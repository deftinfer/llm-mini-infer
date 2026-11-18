"""The raw ASGI application of a frontend process: no FastAPI or pydantic on the hot path (architecture.md, "HTTP
and streaming").

Routes: POST /v1/chat/completions (streaming and not), POST /tokenize and /detokenize (vLLM's shapes); /v1/models,
/server_info (the effective configuration with the reason for each derived value, and the runner's capabilities),
/health (readiness: 200 once warm-up is done, 503 while draining or once the engine stopped; `/ready` and
`/health?ready=1` are the same), /live (liveness: 200 while the process serves HTTP) and /metrics, with any method;
/debug/stages, /debug/engine, /debug/images and /debug/reset with `observability.debug_endpoints`. With an API key set,
every path but /health, /ready, /live and /metrics needs it, and OPTIONS reaches no other handler (`auth`).

A request's id is its `X-Request-Id` header when the client sends a usable one (`request_id`); with
`observability.log_requests` each chat request ends with one INFO line of metadata (`App.log_request`): never prompts,
images or outputs. The HTTP counters: `HttpTally`. The engine marker: `llm_mini_infer.version`. Stopping: `begin_drain`
and `stop_now`, which the launcher's signals call (`api.server`); from the start of a drain every response carries
`Connection: close`. The context limit: `check_length` and `_chat_request`; the schema checks: `grammar`.
"""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import logging
import os
import re
import time
from collections.abc import Callable, Iterator
from enum import IntEnum
from typing import Any

import msgspec
import numpy as np

from ..config import Config, tokenizer_eos_token
from ..engine.core import PHASES
from ..engine.messages import AddRequest
from ..grammar import GrammarError, NestingError
from ..grammar.spec import Pending, thought
from ..lifecycle import Lifecycle
from ..metrics import Metrics
from ..output.detokenizer import Detokenizer
from ..output.logprobs import RequestLogprobs, TokenText, content
from ..output.reasoning import ReasoningFamily, ReasoningParser, effort_kwargs, thinking_flag
from ..output.stop import OutputText, StopStrings
from ..output.tool_calls import QWEN3, Family, ToolCallParser, ToolFeed, api_finish_reason
from ..render.jinja_env import MAX_OPERATIONS, compile_template, operations
from ..render.messages import to_template_messages
from ..render.renderer import Renderer, TemplateError
from ..render.tool_results import NoSentinel, has_tool_text, locate
from ..version import ENGINE_HEADER, engine, fingerprint, version
from .auth import OPEN_PATHS, ApiKeys
from .engine_client import EngineClient, RequestState, sampling_seed, sampling_temperature
from .images import ImageInputs, Prepared
from .protocol import (
    ApiError,
    ChatCompletionRequest,
    JsonObjectFormat,
    JsonSchemaFormat,
    LogprobsLimit,
    check_chat_template,
    check_messages,
    decode_chat,
    decode_detokenize,
    decode_tokenize,
    logprobs_param,
    sanitize,
    validate_chat,
    validation_error,
)
from .sse import DONE, ChunkTemplate
from .tools import ToolRequest, check_tools, template_tool, tool_request

log = logging.getLogger(__name__)
clock = time.perf_counter_ns

MAX_BODY = 32 << 20
# A tool-call grammar's bound (`App.tool_grammar`): 4 × (the schemas' bytes + 128 and the name's bytes a tool) + 64
# KiB. The test corpora's and the recorded tools' grammars take at most 2.4 bytes a byte of their schemas and about
# 110 a tool, below half of it, and the schemas' cap (structured_output.max_schema_bytes) bounds the schemas' part. A
# `$defs` chain copied into each Qwen parameter can take 50 times its schema: the build counts its bytes as it goes and
# stops at the bound (`grammar.tools._Rules`)
GRAMMAR_PER_BYTE, GRAMMAR_PER_TOOL, GRAMMAR_SLACK = 4, 128, 64 << 10
COMPILE_THREADS = 2  # a frontend's threads for grammar builds and checks (grammar.threads)
JSON = [(b"content-type", b"application/json")]
SSE = [(b"content-type", b"text/event-stream"), (b"cache-control", b"no-cache"), (b"x-accel-buffering", b"no")]
_enc = msgspec.json.Encoder()
# A usable X-Request-Id: 1-256 printable ASCII characters, no space, quote or backslash (it goes into JSON, SSE
# templates and logs as is)
_REQUEST_ID = re.compile(rb"[\x21\x23-\x5b\x5d-\x7e]{1,256}")
STOPPED = "the server stopped before the request finished; retry it"
# vLLM's 503s past its queue caps (`App.admit`), its texts
QUEUE_FULL = ("The engine is currently busy and cannot accept new requests. Please try again later or on a different "
              "instance.")
BACKLOG_FULL = ("The engine has reached its prefill token backlog limit. Please try again later or on a different "
                "instance.")
# The paths no HTTP counter counts (`HttpTally`): the six that vLLM's instrumentator leaves out, each searched
# for in the path, and this server's other probes
UNCOUNTED = re.compile("|".join(map(re.escape, ("/metrics", "/health", "/load", "/ping", "/version", "/server_info",
                                                "/ready", "/live"))))
POST_ROUTES = frozenset(("/v1/chat/completions", "/tokenize", "/detokenize"))  # every other route takes any method
DEBUG_ROUTES = frozenset(("/debug/stages", "/debug/engine", "/debug/images", "/debug/reset"))


class ClientGone(Exception):
    """The client left before its response started; `stream`: whether it asked for a stream (False while its body was
    read)."""

    def __init__(self, stream: bool = False) -> None:
        super().__init__()
        self.stream = stream


class Mode(IntEnum):
    """The frontend's mode (`App.mode`): a stop moves it forward only. `App._set` sets the flags the requests read:

    | mode     | entered by                                    | draining | accepting | engine_ok | next     |
    |----------|-----------------------------------------------|----------|-----------|-----------|----------|
    | SERVING  | the start                                     | no       | yes       | yes       | DRAINING |
    | DRAINING | `begin_drain`: /health 503; the grace runs    | yes      | yes       | yes       | REFUSING |
    | REFUSING | the grace's end: new chat requests get a 503  | yes      | no        | yes       | FAILING  |
    | FAILING  | the drain's timeout: every request fails      | yes      | no        | yes       |          |
    | STOPPED  | `stop_now`: every request fails, it stops     | yes      | no        | no        |          |

    STOPPED can follow any mode but itself."""

    SERVING = 0
    DRAINING = 1
    REFUSING = 2
    FAILING = 3
    STOPPED = 4


_MODES = {Mode.SERVING: (Mode.DRAINING, Mode.STOPPED), Mode.DRAINING: (Mode.REFUSING, Mode.STOPPED),
          Mode.REFUSING: (Mode.FAILING, Mode.STOPPED), Mode.FAILING: (Mode.STOPPED,), Mode.STOPPED: ()}


class Connection:
    """A chat request's connection once its body is read (`App.watch_disconnect`): `left` once it went away, and `st`,
    the request, once it was sent to the engine. A request's own ladder: gated (`App.counted`), body read, prepared
    (its grammar and images, a checkpoint after each await: `App.checkpoint`), sent (`st`), then streaming or
    complete."""

    __slots__ = ("left", "st")

    def __init__(self) -> None:
        self.left = False
        self.st: RequestState | None = None


def request_id(scope: dict) -> str:
    """The client's X-Request-Id when it is usable (1-256 printable ASCII characters, no space, quote or
    backslash), else a new id (24 hex digits)."""
    for name, value in scope.get("headers", ()):
        if name == b"x-request-id":
            if _REQUEST_ID.fullmatch(value):
                return value.decode("ascii")
            break
    return os.urandom(12).hex()


CLOSE = (b"connection", b"close")


class HttpTally:
    """One request in the HTTP counters: counted once, as its status line is sent (`sent`; a stream's with its first
    chunk) or as it ends without one (`left`). Each response counts in `http_requests_total{method,status,handler}`,
    the default counter of prometheus-fastapi-instrumentator, as vLLM's server mounts it (`status` grouped as
    `2xx`, `4xx`, `5xx`; `handler` the route's path, or `none` for a path it does not serve or a method its route does
    not take), and in `llm_mini_infer:http_responses_total{handler,method,status}` with the exact code. One that never
    sends a status line counts once it ends: a 499 in the second, and in the first what vLLM's counter records
    then, `2xx` for a client that left a request in flight and `4xx` for one that left while sending the body; an
    exception that escapes is uvicorn's 500. The first leaves out 401s (`sent`), and the `UNCOUNTED` paths get no
    tally."""

    __slots__ = ("done", "handler", "method", "metrics", "route")

    def __init__(self, metrics: Metrics, route: str, method: str, handler: str):
        self.metrics, self.route, self.method, self.handler = metrics, route, method, handler
        self.done = False

    def sent(self, status: int) -> None:
        if self.done:
            return
        self.done = True
        m = self.metrics
        m.http_responses.inc(1.0, self.route, self.method, str(status))
        if status != 401:  # vLLM's key check answers before the counter: no 401 in http_requests_total
            m.http_requests.inc(1.0, self.method, f"{status // 100}xx", self.handler)

    def left(self, prom_status: str) -> None:
        """The client left before a status line: a 499 in the exact-code counter, and `prom_status` (2xx or 4xx)
        in `http_requests_total`."""
        if self.done:
            return
        self.done = True
        self.metrics.http_responses.inc(1.0, self.route, self.method, "499")
        self.metrics.http_requests.inc(1.0, self.method, prom_status, self.handler)

    def disconnected(self, stream: bool) -> None:
        self.metrics.client_disconnects.inc(1.0, self.route, "true" if stream else "false")


def tally_for(metrics: Metrics, routes: frozenset[str], path: str, method: str) -> HttpTally | None:
    """A request's place in the HTTP counters (`HttpTally`): None for a path they leave out."""
    if UNCOUNTED.search(path):
        return None
    route = path if path in routes else "none"
    handler = route if route not in POST_ROUTES or method == "POST" else "none"  # as vLLM's: a full match only
    return HttpTally(metrics, route, method, handler)


def _with_headers(send: Callable, extra: list[tuple[bytes, bytes]], app: App, tally: HttpTally | None) -> Callable:
    """`send` that adds headers to the response start (the request id, the engine marker), and `Connection: close`
    once a drain has begun (uvicorn then closes the connection after the response); and counts the response there
    (`tally`)."""
    async def send_with(msg: dict) -> None:
        if msg["type"] == "http.response.start":
            msg["headers"] = [*msg.get("headers", ()), *extra, *((CLOSE,) if app.draining else ())]
            if tally is not None:
                tally.sent(msg["status"])
        await send(msg)

    return send_with


class RequestLog:
    """What one chat request's log line says (`App.log_request`): metadata, never contents."""

    __slots__ = ("error", "param", "rid", "st", "status", "t0")

    def __init__(self, rid: str):
        self.rid, self.t0 = rid, clock()
        self.status = 0
        self.error = self.param = None
        self.st: RequestState | None = None


async def read_body(receive: Callable) -> bytes:
    chunks, size = [], 0
    while True:
        msg = await receive()
        if msg["type"] == "http.disconnect":
            raise ClientGone
        body = msg.get("body", b"")
        if body:
            size += len(body)
            if size > MAX_BODY:
                raise ApiError(413, "request body too large")
            chunks.append(body)
        if not msg.get("more_body"):
            return chunks[0] if len(chunks) == 1 else b"".join(chunks)


def log_api_key(keys: ApiKeys, host: str) -> None:
    """The startup line on the API key (never the key): what it guards, or, with none, that every path is open, a
    WARNING when the server listens beyond loopback."""
    if keys.enabled:
        log.info("API key: required on every path except OPTIONS, /health, /ready, /live and /metrics "
                 "(Authorization: Bearer; from %s)", " or ".join(keys.sources))
        return
    try:
        local = host == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError:  # a name other than localhost
        local = False
    log.log(logging.INFO if local else logging.WARNING, "API key: none (LLM_MINI_INFER_API_KEY and VLLM_API_KEY "
            "unset): every path is open, /tokenize and /server_info included")


async def respond(send: Callable, status: int, body: bytes, headers: list | None = None) -> None:
    await send({"type": "http.response.start", "status": status,
                "headers": [*(headers or JSON), (b"content-length", str(len(body)).encode())]})
    await send({"type": "http.response.body", "body": body})


class App:
    def __init__(self, cfg: Config, index: int, ipc_path: str, pool_state: Any = None):
        self.cfg = cfg
        self.index = index
        self.model_name = cfg.model.served_model_name or cfg.model.path
        self.max_model_len = cfg.model.max_model_len
        self.renderer = Renderer(cfg.model.path, cfg.model.tokenizer)
        # the logits' width: a token id in a request must be below it (the tokenizer's size when config.json gives none)
        self.vocab = cfg.vocab_size or self.renderer.tokenizer.get_vocab_size(with_added_tokens=True)
        self.token_text = TokenText(self.renderer.tokenizer)  # logprobs' token strings (`output.logprobs`)
        # The longest token's characters: vLLM bounds a prompt by that many a token of `max_model_len`, before it
        # tokenizes (`check_length`)
        self.max_chars_per_token = max(map(len, self.renderer.tokenizer.get_vocab(with_added_tokens=True)))
        if self.renderer.templates:
            try:
                compile_template(self.renderer.default_template)  # compile once, before serving
                _ = self.renderer.content_format, self.renderer.template_variables
            except TemplateError:
                pass  # a template list without a default: compile on demand; each request answers a 400 instead
        # built before serving: the tool results' plain tokenizer and the prompt's segment cache (`Renderer.encode`)
        _ = self.renderer.neutral, self.renderer.segment_cache
        self.metrics = Metrics(self.model_name, self.max_model_len)
        self.metrics.set_cache_info(cfg.dev.cache.page_size, cfg.dev.cache.num_pages, cfg.dev.cache.num_state_slots,
                                    cfg.cache.enable_prefix_caching)
        debug = cfg.observability.debug_endpoints
        self.client = EngineClient(ipc_path, index, self.metrics, cfg.dev.engine.inline_threshold,
                                   debug_records=100_000 if debug else 0, spin_ms=cfg.dev.server.spin_ms or 0.0)
        self.images: ImageInputs | None = None
        im = cfg.images
        if im.enabled:
            from ..vision import family as vf

            fam = vf.load(cfg.model.path, min_pixels=im.min_pixels, max_pixels=im.max_pixels,
                          max_soft_tokens=im.max_soft_tokens)
            if fam is None:
                raise ValueError("images.enabled, but the checkpoint has no image family")
            if fam.name == "gemma4":  # its image token ids come from config.json: they must be the tokenizer's
                fam.check_tokenizer(self.renderer.tokenizer, self.renderer.tokenizer_config)
            self.images = ImageInputs(fam, index=index, num_frontends=cfg.server.num_frontends or 1,
                                      arena_bytes=(cfg.dev.images.arena_mb or 0) << 20, pool_dir=cfg.pool_dir,
                                      workers=im.workers or 1, max_bytes=im.max_bytes,
                                      fetch_timeout_s=im.fetch_timeout_s, max_per_request=im.max_per_request,
                                      allow_urls=im.allow_urls, allowed_networks=im.allowed_networks,
                                      allowed_domains=im.allowed_domains, max_source_pixels=im.max_source_pixels,
                                      pool_state=pool_state, span_key_bits=cfg.dev.images.span_key_bits,
                                      max_request_decode_bytes=im.max_request_decode_bytes,
                                      max_request_download_mb=im.max_request_download_mb)
            self.client.on_released = self.images.release
            if self.images.pool is not None:
                self.images.pool.on_done = lambda d: self.metrics.image_stage.observe(
                    (d.decode_ns + d.write_ns) / 1e9, "pool")
                self.images.pool.on_restart = lambda i, epoch, failed, resent: log.warning(
                    "image pool worker %d stopped (epoch %d next): %d of this frontend's images failed with it, %d "
                    "re-sent", i, epoch, failed, resent)
        # The stop ids a finish reports as `stop_reason` (`stop_reasons`): the model's (eos_token_ids, from
        # generation_config.json) but the tokenizer's own EOS
        eos = tokenizer_eos_token(cfg)
        self.primary_eos = self.renderer.tokenizer.token_to_id(eos) if eos else None
        self.model_stops = frozenset(cfg.eos_token_ids) - {self.primary_eos}
        # tool_calls.parser's format (`output.tool_calls`; None: none, and tools get its 400s: `tool_parser_400`)
        self.tool_family: Family | None = None
        if cfg.tool_calls.parser not in (None, "none"):
            self.tool_family = Family(cfg.tool_calls.parser, self.renderer.tokenizer)
            self.metrics.enable_tool_parser()
        # reasoning.parser's markers (`output.reasoning`; None: none, every output content) and the effort map
        self.reasoning_family: ReasoningFamily | None = None
        if cfg.reasoning.parser not in (None, "none"):
            self.reasoning_family = ReasoningFamily(cfg.reasoning.parser, self.renderer.tokenizer)
        self.effort_map: dict[str, str] = dict(cfg.reasoning.effort_map or {})
        self.grammars = None
        if cfg.structured_output.enabled:
            from ..grammar.spec import GrammarSpecs
            from ..grammar.threads import CompileThreads

            so = cfg.structured_output
            self.grammars = GrammarSpecs(backend=so.backend, disable_any_whitespace=so.disable_any_whitespace,
                                         threads=CompileThreads(COMPILE_THREADS, "grammar"))
        self.defaults = cfg.sampling
        self.fingerprint = fingerprint()
        self.engine_header = (ENGINE_HEADER, engine().encode())
        self.api_keys = ApiKeys()  # from the environment only (auth)
        if index == 0:
            log_api_key(self.api_keys, cfg.server.host)
        self.created = int(time.time())
        self.mode = Lifecycle("the frontend's mode", Mode.SERVING, _MODES)  # moved by `_set` only
        # The mode's flags, read per request (`Mode`; recomputed by `_set`)
        self.draining = False  # a stop began: /health answers 503
        self.accepting = True  # new chat requests are served (until a drain's grace ends)
        self.engine_ok = True  # false once `stop_now` ran
        self.failed: str | None = None  # the message every request failed with: `stop_now`, or a drain's timeout
        self.log_requests = cfg.observability.log_requests
        self.routes = frozenset(("/v1/models", *POST_ROUTES, *(DEBUG_ROUTES if debug else ())))  # counted by path
        self.inflight = 0  # chat requests from the gate to their end (`counted`)
        self.max_queued_reqs = cfg.scheduler.max_num_queued_reqs  # vLLM's queue caps (`admit`; None: off)
        self.max_queued_tokens = cfg.scheduler.max_num_queued_tokens
        self.on_ready: Callable[[], None] | None = None
        self.on_drained: Callable[[], None] | None = None
        self._drained = asyncio.Event()  # set when `inflight` drops to 0
        self._grace_cut = asyncio.Event()

    # -- lifecycle -------------------------------------------------------------------------------------------

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        kind = scope["type"]
        if kind == "http":
            await self.http(scope, receive, send)
        elif kind == "lifespan":
            await self.lifespan(receive, send)

    async def lifespan(self, receive: Callable, send: Callable) -> None:
        while True:
            msg = await receive()
            if msg["type"] == "lifespan.startup":
                self.client.start()
                if self.images is not None:
                    self.images.start()
                await send({"type": "lifespan.startup.complete"})
                if self.on_ready is not None:
                    self.on_ready()
            elif msg["type"] == "lifespan.shutdown":
                await self.client.close()
                if self.images is not None:
                    await self.images.close()
                await send({"type": "lifespan.shutdown.complete"})
                return

    def begin_drain(self, grace: float = 0.0) -> None:
        """SIGTERM (grace: `server.drain_grace_s`) or SIGINT (none): /health answers 503 at once, so a load balancer
        that health-checks stops routing here; new chat requests are served for `grace` seconds more, then refused
        with 503; in-flight requests finish, up to `server.drain_timeout_s` (then they fail); then the server stops. A
        later call cuts the grace short when it has none of its own (SIGINT), and does nothing else."""
        if self.mode.phase is not Mode.SERVING:
            if grace <= 0:
                self._grace_cut.set()
            return
        self._set(Mode.DRAINING)
        log.info("draining: %d requests in flight; new ones served for %g s more (server.drain_grace_s)",
                 self.inflight, grace)
        asyncio.get_running_loop().create_task(self._drain(grace))

    async def _drain(self, grace: float) -> None:
        """The drain `begin_drain` started. A stop meanwhile (`stop_now`) moved the mode past its steps: they leave it
        as it is, and the rest runs as without it."""
        if grace > 0:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._grace_cut.wait(), grace)
        if self.mode.phase < Mode.REFUSING:
            self._set(Mode.REFUSING)
        if self.inflight:
            log.info("refusing new chat requests; waiting for %d in flight (up to %g s)", self.inflight,
                     self.cfg.server.drain_timeout_s)
        if not await self._wait_idle(self.cfg.server.drain_timeout_s):
            log.warning("drain timeout: failing %d requests", self.inflight)
            if self.mode.phase < Mode.FAILING:
                self._set(Mode.FAILING)
            self._fail(STOPPED)
            await self._wait_idle(5.0)
        log.info("drained")
        if self.on_drained is not None:
            self.on_drained()

    def stop_now(self, message: str = STOPPED) -> None:
        """SIGUSR1 from the launcher (a process failure, a second stop signal, or a stop signal during the start:
        `launcher`), or the launcher is gone. Every in-flight request fails at once (503, or an error event on a
        stream), then the server stops. Once stopped, a call does nothing."""
        if self.mode.phase is Mode.STOPPED:
            return
        self._set(Mode.STOPPED)
        self._grace_cut.set()
        log.error("stopping now: failing %d requests in flight", self.inflight)
        self._fail(message)
        asyncio.get_running_loop().create_task(self._stop_soon())

    async def _stop_soon(self) -> None:
        await self._wait_idle(5.0)
        if self.on_drained is not None:
            self.on_drained()

    def _set(self, mode: Mode) -> None:
        """Move to `mode` (RuntimeError off `Mode`'s edges) and set its flags."""
        self.mode.to(mode)
        self.draining = mode >= Mode.DRAINING
        self.accepting = mode < Mode.REFUSING
        self.engine_ok = mode != Mode.STOPPED

    def _fail(self, message: str) -> None:
        """Every request fails with `message`: `failed` first, which those still being prepared read at their next
        checkpoint, then those sent to the engine (`fail_all`)."""
        self.failed = message
        self.fail_all(message)

    def fail_all(self, message: str) -> None:
        """The requests sent to the engine fail now; those still being prepared fail at their next checkpoint
        (`checkpoint`, after `failed` is set)."""
        for st in list(self.client.reqs.values()):
            st.fail(message)

    @contextlib.contextmanager
    def counted(self) -> Iterator[None]:
        """A chat request in flight (`inflight`), from the gate to its end: a drain waits for it, its body's read
        included."""
        self.inflight += 1
        try:
            yield
        finally:
            self.inflight -= 1
            if not self.inflight:
                self._drained.set()

    def checkpoint(self, conn: Connection, stream: bool = False) -> None:
        """After each await of a chat request not yet sent to the engine: ClientGone if its client left, a 503 once
        every request in flight failed (`failed`)."""
        if conn.left:
            raise ClientGone(stream)
        if self.failed is not None:
            raise ApiError(503, self.failed, kind="ServiceUnavailableError")

    async def _wait_idle(self, timeout: float) -> bool:
        """Until no chat request is in flight (True), or `timeout` (False)."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while self.inflight:
            self._drained.clear()
            left = deadline - loop.time()
            if left <= 0:
                return False
            try:
                await asyncio.wait_for(self._drained.wait(), left)
            except TimeoutError:
                return not self.inflight
        return True

    # -- routing ---------------------------------------------------------------------------------------------

    async def http(self, scope: dict, receive: Callable, send: Callable) -> None:
        path, method = scope["path"], scope["method"]
        rid = request_id(scope)
        extra = [self.engine_header]
        if self.cfg.server.request_id_header:
            extra.append((b"x-request-id", rid.encode()))
        tally = tally_for(self.metrics, self.routes, path, method)
        send = _with_headers(send, extra, self, tally)
        try:
            if self.api_keys.guards(method, path) and not self.api_keys.allows(scope.get("headers", ())):
                await self.unauthorized(send, rid, path)
            elif method == "OPTIONS" and path not in OPEN_PATHS:
                await self.refuse_options(send, path)
            elif path == "/v1/chat/completions":
                if method != "POST":
                    raise ApiError(405, "use POST")
                await self.chat_logged(receive, send, rid, tally)
            elif path in ("/health", "/ready"):
                await self.readiness(send)
            elif path == "/live":
                await respond(send, 200, b'{"status":"ok"}')
            elif path == "/metrics":
                if self.images is not None and self.images.pool is not None:
                    self.images.pool_metrics(self.metrics)
                gcs = self.client.gc
                gcs.maybe_warn()
                self.metrics.set_gc(gcs.process, gcs.ns, gcs.collections)
                self.metrics.chat_inflight.set(self.inflight)
                await respond(send, 200, self.metrics.render(),
                              [(b"content-type", b"text/plain; version=0.0.4; charset=utf-8")])
            elif path == "/v1/models":
                await respond(send, 200, self.models_body())
            elif path == "/tokenize":
                if method != "POST":
                    raise ApiError(405, "use POST")
                await self.tokenize(receive, send)
            elif path == "/detokenize":
                if method != "POST":
                    raise ApiError(405, "use POST")
                await self.detokenize(receive, send)
            elif path == "/server_info":
                await respond(send, 200, self.server_info_body())
            elif path.startswith("/debug/") and self.cfg.observability.debug_endpoints:
                await respond(send, 200, self.debug(path, scope.get("query_string", b"")))
            else:
                raise ApiError(404, f"no route for {path}", kind="NotFoundError")
        except ApiError as exc:
            await respond(send, exc.status, exc.body())
        except ClientGone as exc:  # while it sent its body, or while its request was prepared
            if tally is not None:
                tally.disconnected(exc.stream)
                tally.left("4xx")
        except Exception:
            if tally is not None:
                tally.sent(500)  # uvicorn answers it
            raise
        if tally is not None and not tally.done:  # the client left while its request ran
            tally.left("2xx")

    async def refuse_options(self, send: Callable, path: str) -> None:
        """OPTIONS outside the open paths (`auth`): it needs no key, and several routes answer any method, so it reaches
        no handler. A path the server serves answers 405 with the method it takes (`Allow`); any other, the 404."""
        if path not in self.routes and path != "/server_info":  # the routes: the debug ones when they are on
            raise ApiError(404, f"no route for {path}", kind="NotFoundError")
        allow = "POST" if path in POST_ROUTES else "GET"
        err = ApiError(405, f"use {allow}")
        await respond(send, 405, err.body(), [*JSON, (b"allow", allow.encode())])

    async def unauthorized(self, send: Callable, rid: str, path: str) -> None:
        """A request without the right key (`auth`): an OpenAI-style 401 that never quotes what was sent."""
        self.metrics.unauthorized.inc()
        err = ApiError(401, "Unauthorized: this server needs an API key (Authorization: Bearer <key>)",
                       kind="AuthenticationError")
        if self.log_requests and path == "/v1/chat/completions":
            rec = RequestLog(rid)
            rec.status, rec.error = 401, err.kind
            self.log_request(rec)
        await respond(send, 401, err.body(), [*JSON, (b"www-authenticate", b"Bearer")])

    async def readiness(self, send: Callable) -> None:
        """Warm-up done (this process starts serving only then) and no stop begun: 200; else 503, which a load balancer
        counts as a failed health check."""
        if self.engine_ok and not self.draining:
            await respond(send, 200, b'{"status":"ok","ready":true}')
            return
        state = b"draining" if self.engine_ok else b"engine stopped"
        await respond(send, 503, b'{"status":"' + state + b'","ready":false}')

    def models_body(self) -> bytes:
        return _enc.encode({"object": "list", "data": [{
            "id": self.model_name, "object": "model", "created": self.created, "owned_by": "llm-mini-infer",
            "root": self.cfg.model_source.split("@")[0] or self.cfg.model.path, "parent": None,
            "max_model_len": self.max_model_len, "permission": []}]})

    def server_info_body(self) -> bytes:
        """The effective configuration, the reason for each derived value, and the runner's capabilities
        (architecture.md, "Overview", principle 7; vLLM's /server_info). No secrets live in the configuration
        (architecture.md, "Configuration"); `auth` says whether a key is required, never the key."""
        cfg = msgspec.to_builtins(self.cfg)
        reasons = cfg.pop("reasons", {})
        info = cfg.pop("runner_info", {})
        return _enc.encode({"version": version(), "engine": self.fingerprint, "config": cfg, "reasons": reasons,
                            "auth": "api key" if self.api_keys.enabled else "none", "runner": info})

    def check_model(self, model: str | None) -> None:
        """None or "": the served model; any other name is a 404 that names the served ones, as vLLM's
        `_check_model` does, its message sanitized as vLLM sanitizes every error's (a path, the model's as given
        among them, becomes `<path>`)."""
        if model and model != self.model_name:
            raise ApiError(404, sanitize(f"The model `{model}` does not exist. Valid aliases: {self.model_name}."),
                           "model", "NotFoundError")

    def admit(self) -> None:
        """vLLM's queue caps (`scheduler.max_num_queued_reqs`, `max_num_queued_tokens`), checked where vLLM checks them:
        at its preflight (after the model's check, before the chat template's), and again once the request is
        prepared, as vLLM's add does, so the requests prepared meanwhile count. Nothing is sent before either, so a
        stream is refused with a plain 503. Counted on `vllm:admission_rejections_total` by the cap, the requests'
        first. The counts are this frontend's own (`EngineClient.reqs`, `queued_tokens`): the requests sent to the
        engine and not ended, which a request joins after its preparation, as vLLM counts those it added; and their
        prompt tokens until their first output, each whole. The token cap looks at that backlog alone, not at the new
        prompt."""
        cap = self.max_queued_reqs
        if cap is not None and len(self.client.reqs) + 1 > cap:
            self._refuse("max_num_queued_reqs", QUEUE_FULL)
        cap = self.max_queued_tokens
        if cap is not None and self.client.queued_tokens >= cap:
            self._refuse("max_num_queued_tokens", BACKLOG_FULL)

    def _refuse(self, reason: str, message: str) -> None:
        self.metrics.admission_rejections.inc(1.0, reason)
        raise ApiError(503, message, kind="Service Unavailable")  # vLLM's type for these: the status's phrase

    # -- chat ------------------------------------------------------------------------------------------------

    def render_prompt(self, messages: list, add_generation_prompt: bool, kwargs: dict | None) -> str:
        return self.render_conversation(messages, add_generation_prompt, kwargs)[0]

    def _template_messages(self, msgs: list[dict], where: list[int] | None = None) -> list[dict]:
        r = self.renderer
        return to_template_messages(msgs, r.content_format, r.supports_developer_role,
                                    self.images.family.placeholder if self.images is not None else "", where)

    def template_kwargs(self, kwargs: dict | None, effort: str | None = None) -> dict:
        """The chat template's effective kwargs, before the template's own filter: `model.chat_template_kwargs`, under
        the request's `chat_template_kwargs`, under what its `reasoning_effort` sets through the effort map
        (`output.reasoning.effort_kwargs`, vLLM's `build_chat_params`)."""
        out = {**self.cfg.model.chat_template_kwargs, **(kwargs or {})}
        if effort is not None:
            out.update(effort_kwargs(effort, self.effort_map, kwargs))
        return out

    def render_conversation(self, messages: list, add_generation_prompt: bool, kwargs: dict | None,
                            where: list[int] | None = None, tools: list[dict] | None = None,
                            effort: str | None = None) -> tuple[str, list[dict], dict, list[dict]]:
        """(prompt text, the template's conversation, the template kwargs used, the request's messages as builtins).
        `tools`: their template form (`api.tools`); `effort`: the request's `reasoning_effort` (`template_kwargs`). A
        400 for a tool call's `arguments` nested deeper than `json.loads` parses, or a conversation the chat template
        cannot render (`TemplateError`; a mapped effort the template refuses keeps the template's 400)."""
        r = self.renderer
        msgs = [msgspec.to_builtins(m) for m in messages]
        try:
            try:
                conv = self._template_messages(msgs, where)
            except RecursionError as exc:
                raise ApiError(400, "the messages are nested too deeply", "messages") from exc
            kw = r.filter_template_kwargs(self.template_kwargs(kwargs, effort))
            return r.render(conv, tools=tools, add_generation_prompt=add_generation_prompt, **kw), conv, kw, msgs
        except TemplateError as exc:
            # select_template (no template / no default), content_format / supports_developer_role and
            # filter_template_kwargs all raise it before the render's own handler; all are a client 400.
            raise validation_error(str(exc)) from exc  # vLLM's: the template's reason alone

    def tool_spans(self, msgs: list[dict], text: str, add_generation_prompt: bool, kw: dict,
                   tools: list[dict] | None) -> list[tuple[int, int]] | None:
        """Where the prompt `text` holds the tool messages' own text (`render.tool_results`), for a request with a
        tool message's text; None for one without (today's tokenization). Rendered again in a budget of its own."""
        if not has_tool_text(msgs):
            return None
        r = self.renderer

        def render(m: list[dict]) -> str:
            return r.render(self._template_messages(m), tools=tools, add_generation_prompt=add_generation_prompt,
                            **kw)

        with operations(MAX_OPERATIONS):
            try:
                return locate(msgs, text, render)
            except TemplateError:
                log.warning("tool results not neutralized: the chat template failed on the located render")
                return []
            except NoSentinel as exc:
                raise ApiError(400, f"the tool results cannot be located: {exc}", "messages") from exc

    async def chat_logged(self, receive: Callable, send: Callable, rid: str, tally: HttpTally | None = None) -> None:
        rec = RequestLog(rid)
        try:
            await self.chat(receive, send, rec, tally)
        except ApiError as exc:
            rec.status, rec.error, rec.param = exc.status, exc.kind, exc.param
            raise
        except ClientGone:
            rec.status = 499
            raise
        except Exception:  # uvicorn answers it with a 500 (or drops a response that started), as `http` tallies it
            rec.status, rec.error = 500, "InternalServerError"
            raise
        finally:
            if self.log_requests:
                self.log_request(rec)

    def log_request(self, rec: RequestLog) -> None:
        """One line per chat request (observability.log_requests): its id, HTTP status (499: the client left before
        the response started), finish reason, token counts, images, TTFT and end-to-end time from its receipt (the
        log's time for e2e); then the body's receipt, the TTFT and e2e that the histograms observe (`metric_*`: from
        the request parsed to the first output and the finish received), and under MTP its verify steps and their
        tokens. Never contents."""
        e2e = (clock() - rec.t0) / 1e6
        st = rec.st
        if st is None:
            log.info("request id=%s status=%d error=%s%s e2e_ms=%.1f", rec.rid, rec.status, rec.error or "-",
                     f" param={rec.param}" if rec.param else "", e2e)
            return
        status = rec.status or (499 if st.cancelled else 200)
        ttft = f"{(st.t_first_recv - st.t_recv) / 1e6:.1f}" if st.t_first_recv else "-"
        metric_ttft = f"{(st.t_first_recv - st.t_arrival) / 1e6:.1f}" if st.t_first_recv else "-"
        metric_e2e = f"{(st.t_end - st.t_arrival) / 1e6:.1f}" if st.t_end else "-"
        steps, vtok = (st.verify_steps, st.verify_tokens) if st.verify_steps else ("-", "-")
        log.info("request id=%s status=%d finish=%s stream=%d prompt=%d cached=%d images=%d out=%d ttft_ms=%s "
                 "e2e_ms=%.1f receive_ms=%.1f metric_ttft_ms=%s metric_e2e_ms=%s verify_steps=%s verify_tokens=%s",
                 rec.rid, status, st.finish_reason or "-", st.stream, st.prompt_len, st.cached, st.n_images, st.n_out,
                 ttft, e2e, (st.t_body - st.t_recv) / 1e6, metric_ttft, metric_e2e, steps, vtok)

    async def chat(self, receive: Callable, send: Callable, rec: RequestLog, tally: HttpTally | None = None) -> None:
        """A chat request, counted from the gate (`counted`). Once its body is read, `watch_disconnect` is the only
        reader of `receive`, and each await before the request goes to the engine is followed by a checkpoint."""
        t_recv = clock()
        if not (self.accepting and self.engine_ok):
            raise ApiError(503, "the server is shutting down", kind="ServiceUnavailableError")
        with self.counted():
            body = await read_body(receive)
            conn = Connection()
            watcher = asyncio.get_running_loop().create_task(self.watch_disconnect(receive, conn))
            try:
                await self._chat(body, send, rec, tally, t_recv, conn)
            finally:
                watcher.cancel()

    async def _chat(self, body: bytes, send: Callable, rec: RequestLog, tally: HttpTally | None, t_recv: int,
                    conn: Connection) -> None:
        t_body = clock()
        self.checkpoint(conn)
        req: ChatCompletionRequest = decode_chat(body)
        validate_chat(req, self.vocab, images=self.images is not None)
        tools = tool_request(req.tools, req.tool_choice, req.parallel_tool_calls)
        self.check_model(req.model)
        self.admit()  # the queue caps, vLLM's preflight: a full server answers 503 even to a request it would refuse
        check_chat_template(req)  # vLLM's order: after the caps; before the grammar, images and render
        # The origin of the TTFT and e2e histograms, as vLLM's: the request parsed and checked, before its
        # grammar, its images and its render
        t_arrival = clock()
        if tools is not None and self.tool_family is None:
            self.tool_parser_400(tools)
        try:
            grammar = self.grammar(req, tools, warn=True)
        except Pending as p:
            grammar = await self.grammar_built(p, req, tools)
            self.checkpoint(conn, req.stream)
        prep: Prepared | None = None
        if self.images is not None:
            urls = self.images.urls(req.messages)
            if urls:  # fetched, sized and hashed; the pool decodes while the prompt renders and tokenizes
                prep = await self.images.prepare(urls)
        try:
            if prep is not None:
                self.checkpoint(conn, req.stream)
            try:
                st, add = self._chat_request(req, t_recv, prep, rec.rid, tools)
            except LogprobsLimit as exc:
                if not req.stream:
                    raise
                # as vLLM answers it: inside the stream it already started, one error event (its compact JSON, as
                # pydantic's model_dump_json writes it), then [DONE]
                await send({"type": "http.response.start", "status": 200, "headers": SSE})
                await send({"type": "http.response.body", "body": b"data: " + exc.body() + b"\n\n" + DONE})
                rec.status = 200
                if prep is not None:
                    self.images.discard(prep)
                return
            if grammar and st.case in ("b", "c"):  # thinking on: the thought in front (`grammar.spec.thought`)
                try:
                    grammar = self.grammar(req, tools, st.case)
                except Pending as p:
                    grammar = await self.grammar_built(p, req, tools, st.case)
                    self.checkpoint(conn, req.stream)
            st.t_body, st.t_arrival = t_body, t_arrival
            rec.st = st
            add.grammar = grammar
            if prep is not None:
                await self.images.wait(prep)
                self.checkpoint(conn, req.stream)
                add.images = prep.refs(add.images)  # the spans' offsets, turned into references
                st.image_ns, st.n_images = prep.image_ns, len(prep.images)
                st.t_fetched, st.t_sized, st.t_pixels = prep.t_fetched, prep.t_sized, prep.t_pixels
            try:  # the caps again, after the last await: nothing runs between this and the send below
                self.admit()
            except ApiError:
                rec.st = None  # logged as the first check's refusal is, with its error
                raise
        except BaseException:
            if prep is not None:
                self.images.discard(prep)
            raise
        client = self.client
        if prep is not None:
            self.images.adopt(st.rid, prep)
            m = self.metrics
            m.images.inc(len(prep.images))
            m.image_stage.observe((prep.t_fetched - prep.t_start) / 1e9, "fetch")
            m.image_stage.observe((prep.t_sized - prep.t_fetched) / 1e9, "probe")
            m.image_stage.observe((prep.t_pixels - st.t_tokenized) / 1e9 if prep.t_pixels > st.t_tokenized else 0.0,
                                  "pool_wait")
        client.add(st, add)
        conn.st = st
        try:
            # 200 once the response started (a stream whose client then left too), 499 if the client left before it
            if req.stream:
                started = await self.stream(st, send)
            else:
                started = await self.complete(st, send, st.request_id)
            rec.status = 200 if started else 499
        finally:
            if st.left and tally is not None:
                tally.disconnected(req.stream)

    def grammar(self, req: ChatCompletionRequest, tools: ToolRequest | None = None, case: str = "a",
                warn: bool = False) -> str:
        """The request's grammar text: with tools, a call grammar where one applies (`tool_grammar`), else its
        `response_format`'s (`response_format_grammar`). `case`: where the output starts (`output.reasoning`); b and c
        put the thought in front (`thought_of`), a and d are the thinking-off grammar, which `chat` checks first.
        `warn`: `chat`'s first call, which logs the request's warnings (once a request, whatever builds follow)."""
        if tools is not None and self.tool_family is not None:
            got = self.tool_grammar(req, tools, case, warn)
            if got is not None:
                return got
        return self.response_format_grammar(req.response_format, case)

    async def grammar_built(self, p: Pending, req: ChatCompletionRequest, tools: ToolRequest | None,
                            case: str = "a") -> str:
        """`grammar` after a cache miss: each grammar it lacks built on the compile threads (`GrammarSpecs.resolve`),
        then looked up again, with what the builds gave."""
        done: dict[bytes, Any] = {}
        while True:
            done[p.key] = await self.grammars.resolve(p)
            try:
                with self.grammars.using(done):
                    return self.grammar(req, tools, case)
            except Pending as q:
                p = q

    def thought_of(self, case: str) -> str | None:
        """The thought a grammar takes in front where the output starts with one (cases b and c), else None."""
        fam = self.reasoning_family
        if fam is None or case not in ("b", "c"):
            return None
        return thought(fam.name, case, fam.start, fam.end)

    def response_format_grammar(self, rf: Any, case: str = "a") -> str:
        """The grammar text of a `response_format` ("" without one), checked: a 400 otherwise, with llguidance's
        message, or vLLM's for JSON nested past its cap (no param). A schema over
        `structured_output.max_schema_bytes` as received is a 400 before it is decoded. Cases b and c: the thought in
        front (`GrammarSpecs.thinking`)."""
        if not isinstance(rf, JsonObjectFormat | JsonSchemaFormat):
            return ""
        if self.grammars is None:
            raise ApiError(400, "structured output is off on this server (structured_output.enabled)",
                           "response_format")
        size = len(rf.json_schema.schema) if isinstance(rf, JsonSchemaFormat) else 0
        cap = self.cfg.structured_output.max_schema_bytes
        if size > cap:
            raise ApiError(400, f"the JSON schema in 'response_format' is {size} bytes, more than this server takes: "
                                f"{cap} (structured_output.max_schema_bytes)", "response_format")
        prefix = self.thought_of(case)
        try:
            if prefix is not None:
                fam = self.reasoning_family
                return self.grammars.thinking(rf.json_schema.schema if isinstance(rf, JsonSchemaFormat) else None,
                                              prefix, f"{fam.name}:{case}".encode(), fam.grammar_token_ids)
            if isinstance(rf, JsonObjectFormat):
                return self.grammars.json_object()
            return self.grammars.json_schema(rf.json_schema.schema)
        except NestingError as exc:
            raise ApiError(400, str(exc)) from exc
        except GrammarError as exc:
            raise ApiError(400, f"invalid JSON schema in 'response_format': {exc}", "response_format") from exc

    @staticmethod
    def tool_parser_400(tools: ToolRequest) -> None:
        """tool_calls.parser none: a request with tools gets vLLM's 400s of a server without a tool parser,
        with the setting where its texts name its flags (no parameter suffix, param null); tool_choice "none" and
        null are served, the tools rendered."""
        if tools.choice == "auto":
            raise ApiError(400, '"auto" tool choice requires tool_calls.parser to be set')
        if tools.choice == "required":
            raise ApiError(400, 'tool_choice="required" requires tool_calls.parser to be set')
        if tools.choice == "named":
            raise ApiError(400, f'tool_choice=function "{tools.name}" requires tool_calls.parser to be set')

    def tool_grammar(self, req: ChatCompletionRequest, tools: ToolRequest, case: str = "a",
                     warn: bool = False) -> str | None:
        """The call grammar a request with tools takes (`grammar.tools`), checked; None where none applies:
        - `required` or a named tool: calls in the model's syntax (a `response_format` is ignored, with a warning when
          `warn`);
        - `auto` with a `response_format`: calls or the schema's JSON;
        - Qwen's `auto` with a `strict` tool or `parallel_tool_calls` false: free text, then calls.
        With structured output off, the first two are a 400 and `auto` is served unconstrained. Under `auto`, the
        `response_format` is checked first, alone, with its own 400s (param `response_format`). The tools' schemas
        count against `structured_output.max_schema_bytes` with `response_format`'s; a schema the grammar cannot take
        is a 400 that names its keyword or `$ref`, and so is a grammar above its bound (`GRAMMAR_PER_BYTE`; param
        `tools`). Cases b and c: the thought in front of the calls (Qwen's triggered free text takes it already)."""
        rf = req.response_format
        has_rf = isinstance(rf, JsonObjectFormat | JsonSchemaFormat)
        fam = self.tool_family
        choice = tools.choice
        triggered = False
        if choice in ("required", "named"):
            if has_rf and warn:
                log.warning("request %s: tool_choice %s ignores its response_format", req.user or "-", choice)
        elif choice == "auto" and has_rf:
            self.response_format_grammar(rf)
        elif choice == "auto" and fam.name == QWEN3 and (not tools.parallel or any(
                t["function"].get("strict") is True for t in tools.tools)):
            triggered = True
        else:
            return None
        if self.grammars is None:
            if triggered:
                return ""  # auto, unconstrained
            raise ApiError(400, "structured output is off on this server (structured_output.enabled)",
                           "response_format" if choice == "auto" else "tool_choice")
        schema = rf.json_schema.schema if isinstance(rf, JsonSchemaFormat) else None
        size = sum(len(_enc.encode(t["function"]["parameters"])) for t in tools.tools) + len(schema or b"")
        cap = self.cfg.structured_output.max_schema_bytes
        if size > cap:
            raise ApiError(400, f"the tools' and response_format's JSON schemas are {size} bytes, more than this "
                                f"server takes: {cap} (structured_output.max_schema_bytes)", "tools")
        from ..grammar.tools import ToolSchemaError, call_grammar

        g = self.grammars
        with_rf = has_rf and choice == "auto"
        prefix = None if triggered else self.thought_of(case)
        key = _enc.encode([fam.name, tools.tools, choice, tools.name, tools.parallel, triggered,
                           (msgspec.Raw(schema) if schema else "object") if with_rf else None,
                           *((case,) if prefix is not None else ())])
        max_bytes = GRAMMAR_PER_BYTE * (size + sum(GRAMMAR_PER_TOOL + len(t["function"]["name"].encode())
                                                   for t in tools.tools)) + GRAMMAR_SLACK

        def build() -> str:
            try:
                item = g.json_item(schema) if with_rf else None
                return call_grammar(fam.name, tools.tools, choice, tools.name, tools.parallel,
                                    forcing=g.backend == "guidance", response_format=item, triggered=triggered,
                                    ids=fam.grammar_ids, max_bytes=max_bytes, thought=prefix)[0]
            except ToolSchemaError as exc:
                raise GrammarError(f"tools: {exc}") from exc

        try:
            return g.tool_calls(key, build, fam.grammar_token_ids)
        except GrammarError as exc:
            msg = str(exc)
            if msg.startswith("tools: "):
                raise ApiError(400, msg.removeprefix("tools: "), "tools") from exc
            raise ApiError(400, f"invalid tool or response_format schema: {msg}", "tools") from exc

    def _chat_request(self, req: ChatCompletionRequest, t_recv: int, prep: Prepared | None,
                      rid: str, tools: ToolRequest | None = None) -> tuple[RequestState, AddRequest]:
        """Render, tokenize (with the image spans expanded) and check a request: its state and its AddRequest
        (`images` holds the spans' offsets until the pixels are there). vLLM's limits: first `check_length`, the prompt
        alone; then, with the images expanded, a 400 when the prompt reaches `model.max_model_len`, else `max_tokens`,
        a cap, clamped to the room left. `tools`: the request's (`api.tools`), rendered with the prompt and its
        hints."""
        where: list[int] = []  # each message's index in `conv`
        tkw_tools = {"tools": tools.tools} if tools is not None else {}
        with operations(MAX_OPERATIONS):  # one budget for the prompt's render and its hints' (a hint past it: dropped)
            text, conv, tkw, msgs = self.render_conversation(req.messages, req.add_generation_prompt,
                                                             req.chat_template_kwargs, where, **tkw_tools,
                                                             effort=req.reasoning_effort)
            t_rendered = clock()
            spans = self.tool_spans(msgs, text, req.add_generation_prompt, tkw, tkw_tools.get("tools"))
            prompt, max_tokens = self.check_length(req, text, spans)
            hinted = sorted({where[i] for i, m in enumerate(req.messages) if m.cache_hint})
            hints = self.renderer.hint_offsets(conv, hinted, text, prompt, **tkw_tools, **tkw) if hinted else []
        offsets: list[int] = []
        if prep is not None:
            prompt, offsets, hints = self.images.expand(prompt, prep, hints)
        elif self.images is not None and (prompt == self.images.family.pad_id).any():
            raise ApiError(400, "the prompt holds image placeholders but the request has no images", "messages")
        t_tokenized = clock()
        n = len(prompt)
        limit = self.max_model_len
        # vLLM's: at n == limit no token is left, which it refuses as max_tokens 0 (before the input
        # processor's length check)
        if n > limit:  # a ValueError there: param null
            raise ApiError(400, f"Input length ({n}) exceeds model's maximum context length ({limit}).")
        if n == limit:
            raise validation_error("max_tokens must be at least 1, got 0.", "max_tokens", 0)
        lp = logprobs_param(req, self.vocab)  # after the length checks, as vLLM orders them
        if lp >= 0 and not self.cfg.server.logprobs:
            raise ApiError(400, "logprobs are off on this server (server.logprobs)", "logprobs")
        d = self.defaults
        max_tokens = min(max_tokens if max_tokens is not None else (d.max_tokens or limit), limit - n)
        stops = [req.stop] if isinstance(req.stop, str) else (req.stop or [])
        stop_ids = req.stop_token_ids or []
        client = self.client
        erid = client.new_rid()
        request_id = f"chatcmpl-{rid}"  # vLLM's form: chatcmpl-<X-Request-Id>
        skip = req.skip_special_tokens
        parser = None
        if tools is not None and self.tool_family is not None:
            if tools.choice != "none":  # the markers must reach the stop strings, as vLLM's output text has them
                skip = False
            parser = ToolCallParser(self.tool_family, tools.schemas, tools.choice, tools.parallel, skip)
        fam = self.reasoning_family
        case, lazy, hide = "a", None, False
        if fam is not None:  # every request of the family is parsed (`output.reasoning`), its markers in the text
            kw = self.template_kwargs(req.chat_template_kwargs, req.reasoning_effort)
            case = fam.case(prompt, thinking_flag(fam.name, kw))
            if parser is None and case in ("a", "d"):
                lazy = (fam, case, req.skip_special_tokens)  # parsed once a marker comes (`RequestState.on_output`)
            else:
                parser = ReasoningParser(fam, case, parser, req.skip_special_tokens)
            skip = False
            hide = not req.include_reasoning  # vLLM's hide_metadata: no logprobs or output ids either
        st = RequestState(client, erid, n, Detokenizer(self.renderer.tokenizer, prompt, skip),
                          OutputText(StopStrings(stops, req.include_stop_str_in_output) if stops else None),
                          ChunkTemplate(request_id, int(time.time()), self.model_name, self.fingerprint), req.stream,
                          bool(req.stream_options and req.stream_options.include_usage), req.include_stop_str_in_output,
                          self.stop_reasons(stop_ids, req.ignore_eos), max_tokens, t_recv, t_rendered, t_tokenized)
        st.request_id = request_id
        st.case, st.lazy, st.tooled = case, lazy, tools is not None and self.tool_family is not None
        if fam is not None:
            st.reasoning, st.null_reasoning = True, hide
        if parser is not None:
            st.tools = ToolFeed(parser)
        if req.return_token_ids:
            st.out_ids = None if hide else []
            st.prompt_ids = prompt.tolist()
        if hide:
            lp = -1  # asked (and checked) but never sent: the engine computes none
        if lp >= 0:
            st.lp = RequestLogprobs(self.token_text, lp, req.stream)
        top_k = req.top_k if req.top_k is not None else d.top_k
        add = AddRequest(
            rid=erid, frontend=self.index, prompt=prompt, max_tokens=max_tokens,
            temperature=sampling_temperature(req.temperature if req.temperature is not None else d.temperature),
            top_p=req.top_p if req.top_p is not None else d.top_p,
            top_k=top_k if 0 < top_k < self.vocab else 0,  # -1, 0 and the vocabulary or more: off (no filter)
            min_p=req.min_p if req.min_p is not None else d.min_p, seed=sampling_seed(req.seed),
            stop_token_ids=stop_ids, ignore_eos=req.ignore_eos, priority=req.priority, cache_salt=req.cache_salt,
            t_recv_ns=t_recv, cache_hints=hints, request_id=rid, stop_strings=bool(stops), logprobs=lp)
        add.images = offsets
        return st, add

    def stop_reasons(self, stop_ids: list[int], ignore_eos: bool) -> frozenset[int]:
        """The stop ids a request's finish reports as `stop_reason`: its own `stop_token_ids`, and unless `ignore_eos`
        the model's (the generation config's); the tokenizer's own EOS is null unless `ignore_eos`, even when the
        request lists it."""
        if ignore_eos:
            return frozenset(stop_ids)
        return (frozenset(stop_ids) | self.model_stops) - {self.primary_eos}

    def check_length(self, req: ChatCompletionRequest, text: str,
                     spans: list[tuple[int, int]] | None = None) -> tuple[np.ndarray, int | None]:
        """The first limit, as vLLM applies it: the requested output tokens within `model.max_model_len`, then the
        rendered prompt alone, its characters (at most `max_chars_per_token` a token) and then its tokens (one
        placeholder token per image); a 400 otherwise, with vLLM's texts. No room is kept for the output: `max_tokens`
        is a cap, which `_chat_request` clamps to what the prompt leaves. vLLM tokenizes at most one token more of the
        prompt than the limit, so it counts "at least" limit + 1; the 400 says that here too, whatever the whole
        prompt's count. Returns the prompt's tokens and the requested output tokens (None: not given)."""
        limit = self.max_model_len
        if req.max_completion_tokens is not None:
            asked, param = req.max_completion_tokens, "max_completion_tokens"
        else:
            asked, param = req.max_tokens, "max_tokens"
        out = asked or 0
        if out > limit:
            raise validation_error(f"{param}={out} cannot be greater than max_model_len={limit}. Please request fewer "
                                   "output tokens.", param, out)
        chars = limit * self.max_chars_per_token
        if len(text) > chars:
            raise validation_error(f"This model's maximum context length is {limit} tokens. However, your prompt "
                                   f"contains {len(text)} characters (more than {chars} characters, which is the upper "
                                   f"bound for {limit} input tokens). Please reduce the length of the input prompt.",
                                   "input_text", len(text))
        # `spans`: the tool results' text, neutralized (`render.tool_results`); None or []: the segment-cached path
        prompt = self.renderer.encode(text) if not spans else self.renderer.neutral.encode(text, spans)[0]
        if len(prompt) > limit:
            raise validation_error(f"This model's maximum context length is {limit} tokens. However, your prompt "
                                   f"contains at least {limit + 1} input tokens, which exceeds this limit even without "
                                   "any output tokens. Please reduce the length of the input prompt.", "input_tokens",
                                   limit + 1)
        return prompt, asked

    @staticmethod
    async def watch_disconnect(receive: Callable, conn: Connection) -> None:
        """Until the client leaves: then a request sent to the engine is cancelled, and one still being prepared
        leaves at its next checkpoint (`checkpoint`)."""
        try:
            while (await receive())["type"] != "http.disconnect":
                pass
        except asyncio.CancelledError:
            raise
        except Exception:  # the connection failed: the request goes as on a disconnect
            pass
        conn.left = True
        st = conn.st
        if st is None:
            return
        try:
            st.cancel()
        except Exception:
            log.exception("request %d: its cancel after the client left failed", st.rid)

    async def stream(self, st: RequestState, send: Callable) -> bool:
        """Sends the request's server-sent events as they come; returns whether the response started."""
        started = False
        try:
            while True:
                await st.wait()
                if st.cancelled:
                    return started
                data = b"".join(st.pending)
                st.pending.clear()
                if not started:
                    await send({"type": "http.response.start", "status": 200, "headers": SSE})
                    started = True
                    await send({"type": "http.response.body", "body": data, "more_body": not st.done})
                    self.client.record_first_byte(st)
                else:
                    await send({"type": "http.response.body", "body": data, "more_body": not st.done})
                if st.done and not st.pending:
                    return started
        except Exception as exc:  # the client went away mid-stream (servers raise different errors for it)
            log.debug("stream %d ended: %s", st.rid, type(exc).__name__)
            st.cancel()
        return started

    async def complete(self, st: RequestState, send: Callable, request_id: str) -> bool:
        """Sends the request's whole response once it is done; returns False when the client left first."""
        while not st.done:
            await st.wait()
        if st.cancelled:
            return False
        if st.finish_reason == "error":
            raise ApiError(st.error_status, st.error,
                           kind="ServiceUnavailableError" if st.error_status == 503 else "InternalServerError")
        text = st.text.take(final=True)
        calls: list[dict] = []
        reasoning = None
        finish = st.finish_reason
        if st.tools is not None:  # the parsers' content, reasoning and calls (`output.reasoning`, `output.tool_calls`)
            p = st.tools.parser
            text = p.content()
            if st.tools.reasoning:
                reasoning = p.reasoning()
            calls = [{"id": c.id, "type": "function", "function": {"name": c.name.strip(), "arguments": c.arguments}}
                     for c in p.kept_calls()]
            finish = api_finish_reason(finish, p)
            if self.metrics.tool_parser is not None and st.tooled:
                self.metrics.tool_parser.inc(1.0, "non_streaming", "tool_call" if p.choice != "none" and p.found()
                                             else "no_tool_call", "chat_completions")
        elif st.reasoning:  # a parsed reply whose output stayed content (`RequestState.lazy`): null when empty
            text = text or None
        # vLLM's message: its keys (`tool_calls` only with calls), `reasoning` null without one
        message = {"role": "assistant", "content": text, "refusal": None, "annotations": None, "audio": None,
                   "function_call": None}
        if calls:
            message["tool_calls"] = calls
        message["reasoning"] = None if st.null_reasoning else reasoning
        choice = {"index": 0, "message": message,
                  "logprobs": msgspec.Raw(content(st.lp_entries)) if st.lp is not None else None,
                  "finish_reason": finish, "stop_reason": st.stop_reason}
        usage = {"prompt_tokens": st.prompt_len, "total_tokens": st.prompt_len + st.n_out,
                 "completion_tokens": st.n_out, "prompt_tokens_details": {"cached_tokens": st.cached},
                 "completion_tokens_details": {"reasoning_tokens": st.reasoning_tokens()} if st.reasoning else None}
        out = {"id": request_id, "object": "chat.completion", "created": int(time.time()), "model": self.model_name,
               "system_fingerprint": self.fingerprint, "choices": [choice], "usage": usage}
        if st.out_ids is not None:  # return_token_ids (the output's hidden with include_reasoning false)
            choice["token_ids"] = st.out_ids
        if st.prompt_ids is not None:
            out["prompt_token_ids"] = st.prompt_ids
        body = _enc.encode(out)
        await respond(send, 200, body)
        self.client.record_first_byte(st, streaming=False)
        return True

    # -- tokenize --------------------------------------------------------------------------------------------

    async def tokenize(self, receive: Callable, send: Callable) -> None:
        req = decode_tokenize(await read_body(receive))
        self.check_model(req.model)
        if (req.prompt is None) == (req.messages is None):
            raise ApiError(400, "give exactly one of 'prompt' and 'messages'")
        if req.messages is not None:
            check_chat_template(req)  # right after the model's check, as vLLM does (the prompt form ignores it)
        if req.continue_final_message:
            raise ApiError(400, "'continue_final_message' is not supported yet", "continue_final_message")
        spans = None
        if req.messages is not None:
            check_messages(req.messages, images=self.images is not None)
            tools = None  # rendered as a chat request's, an empty list too; no tool_choice, so no tool 400s
            if req.tools is not None:
                check_tools(req.tools)
                tools = [template_tool(t) for t in req.tools]
            text, _, kw, msgs = self.render_conversation(req.messages, req.add_generation_prompt,
                                                         req.chat_template_kwargs, tools=tools)
            spans = self.tool_spans(msgs, text, req.add_generation_prompt, kw, tools)
            add_special = bool(req.add_special_tokens)
            if self.images is not None:
                urls = self.images.urls(req.messages)
                if urls:  # sizing only: the placeholder spans, no pixels
                    if req.return_token_strs:  # before any fetch
                        raise ApiError(400, "'return_token_strs' is not supported with images", "return_token_strs")
                    prep = await self.images.prepare(urls, submit=False)
                    ids, _ = self.tokenize_text(text, add_special, spans)
                    ids, _, _ = self.images.expand(np.asarray(ids, np.int64), prep)
                    body = {"count": len(ids), "max_model_len": self.max_model_len, "tokens": ids.tolist(),
                            "token_strs": None}
                    await respond(send, 200, _enc.encode(body))
                    return
        else:
            text = req.prompt
            add_special = req.add_special_tokens is None or req.add_special_tokens
        ids, strs = self.tokenize_text(text, add_special, spans)
        body: dict[str, Any] = {"count": len(ids), "max_model_len": self.max_model_len, "tokens": ids,
                                "token_strs": strs if req.return_token_strs else None}
        await respond(send, 200, _enc.encode(body))

    def tokenize_text(self, text: str, add_special: bool, spans: list[tuple[int, int]] | None
                      ) -> tuple[list[int], list[str]]:
        """/tokenize's ids and token strings: the tokenizer's, with the tool results' text neutralized (`spans`;
        `render.tool_results`), as a chat request's prompt."""
        if spans:
            ids, strs = self.renderer.neutral.encode(text, spans, add_special_tokens=add_special)
            return ids.tolist(), strs
        enc = self.renderer.tokenizer.encode(text, add_special_tokens=add_special)
        return enc.ids, enc.tokens

    async def detokenize(self, receive: Callable, send: Callable) -> None:
        """/detokenize: token IDs to text, special tokens kept."""
        req = decode_detokenize(await read_body(receive))
        self.check_model(req.model)
        vocab = self.renderer.tokenizer.get_vocab_size(with_added_tokens=True)
        bad = next((t for t in req.tokens if not 0 <= t < vocab), None)
        if bad is not None:
            raise ApiError(400, f"token id {bad} is out of the vocabulary (0..{vocab - 1})", "tokens")
        text = self.renderer.tokenizer.decode(req.tokens, skip_special_tokens=False)
        await respond(send, 200, _enc.encode({"prompt": text}))

    # -- debug -----------------------------------------------------------------------------------------------

    def debug(self, path: str, query: bytes) -> bytes:
        c = self.client
        if path == "/debug/stages":
            return _enc.encode(list(c.records or ()))
        if path == "/debug/engine":
            s = c.engine_stats
            stats = {f: getattr(s, f) for f in s.__struct_fields__} if s else None  # array_like: no to_builtins
            return _enc.encode({"stats": stats, "phases": list(PHASES),
                                "steps": list(c.step_samples or ()), "pid": os.getpid(),
                                "engine_pid": int(os.environ.get("LLM_MINI_INFER_ENGINE_PID", 0))})
        if path == "/debug/images":
            im = self.images
            pool = im.pool if im is not None else None
            st = pool.state if pool is not None else None
            n = range(st.workers if st is not None else 0)
            return _enc.encode({"arena_used": im.alloc.used if im else 0, "owned": len(im.owned) if im else 0,
                                "pool_outstanding": list(pool._outstanding) if pool else [],
                                "pool_pids": [st.pid(i) for i in n], "pool_epochs": [st.epoch(i) for i in n],
                                "pool_ready": [int(st.ready(i)) for i in n],
                                "pool_busy": [int(st.since(i) != 0) for i in n],
                                "pool_failed": pool.failed if pool else 0, "pool_resent": pool.resent if pool else 0,
                                "pool_hangs": st.hangs if st else 0, "pool_closed": int(st.closed) if st else 0})
        if path == "/debug/reset":
            if c.records is not None:
                c.records.clear()
            if c.step_samples is not None:
                c.step_samples.clear()
            return b"{}"
        raise ApiError(404, f"no route for {path}", kind="NotFoundError")
