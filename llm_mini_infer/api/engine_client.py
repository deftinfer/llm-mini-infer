"""The frontend's link to the engine: sends AddRequest, Abort and Ack (backpressure: `engine.process`), reads
StepOutputs (tokens, finishes, and the engine statistics behind /metrics), and turns tokens into text.

Detokenization and stop strings run in the output reader as each StepOutputs arrives, so a stop string sends its
Abort at once, whatever the client's read speed. The reader leaves ready-to-send bytes on each request (one SSE
event per engine step) and wakes the request's coroutine, which only writes them.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections import deque
from collections.abc import Callable, Collection
from typing import Any

import msgspec
import zmq

from ..engine import ipc
from ..engine.core import PHASES
from ..engine.messages import (
    ABORT_DISCONNECT,
    ABORT_FAILED,
    ABORT_REASONS,
    ABORT_STOP,
    CHECKPOINT_SOURCES,
    LOST_CAUSES,
    MASK_CLASSES,
    OPENAI_FINISH,
    Abort,
    Ack,
    AddRequest,
    EngineStats,
    FinishReason,
    StepOutputs,
)
from ..gcstats import GcStats
from ..metrics import Metrics
from ..output.detokenizer import Detokenizer
from ..output.logprobs import RequestLogprobs, content
from ..output.reasoning import GEMMA4, LABEL, ReasoningParser
from ..output.stop import OutputText
from ..output.tool_calls import ToolFeed, api_finish_reason, delta_json
from .sse import DONE, ChunkTemplate

log = logging.getLogger(__name__)
clock = time.perf_counter_ns
_enc = msgspec.json.Encoder()
_WS = re.compile(r"\s*")
ACK_EVERY = 8  # acknowledge the engine's outputs every this many messages (backpressure, `engine.process`)


class RequestState:
    """A chat request in the frontend, from `App._chat_request` to its end. It ends one of four ways, once: `done`,
    `counted` (`client.on_finished` was called) and the rest as below. The booleans stay plain slots, read per chunk.

    | kind                    | how                                                 | cancelled | left | finish_reason |
    |-------------------------|-----------------------------------------------------|-----------|------|---------------|
    | finished                | the engine's finish (`on_output`)                   | no        | no   | the engine's  |
    | errored                 | `fail` (a stop), `crash`, or the engine's error     | no        | no   | "error"       |
    | cancelled: client left  | `cancel()`: a disconnect, or a stream's failed send | yes       | yes  | "abort"       |
    | cancelled by the server | `cancel(left=False)` (`EngineClient.close`)         | yes       | no   | "abort"       |

    One edge leaves an end: `crash` re-opens `done` to finish the request again with an error, whatever state its
    failed processing left it in."""

    __slots__ = ("cached", "cancelled", "case", "client", "counted", "detok", "done", "error", "error_status",
                 "finish_reason", "first_sent", "image_ns", "include_stop", "include_usage", "lazy", "lead", "left",
                 "lp", "lp_entries", "max_tokens", "n_images", "n_out", "null_reasoning", "out_ids", "pending",
                 "prompt_ids", "prompt_len", "queued", "reasoning", "request_id", "rid", "runner_ns", "step_ids",
                 "stop_ids",
                 "stop_reason", "stream", "t_admit", "t_arrival", "t_body", "t_end", "t_fetched", "t_first_out",
                 "t_first_recv", "t_last", "t_last_out", "t_launch", "t_pixels", "t_queued", "t_recv", "t_rendered",
                 "t_sized", "t_tokenized", "text", "tooled", "tools", "tpl", "verify_steps", "verify_tokens", "waiter")

    def __init__(self, client: EngineClient, rid: int, prompt_len: int, detok: Detokenizer, text: OutputText,
                 tpl: ChunkTemplate, stream: bool, include_usage: bool, include_stop: bool, stop_ids: Collection[int],
                 max_tokens: int, t_recv: int, t_rendered: int, t_tokenized: int):
        self.client = client
        self.rid = rid
        self.prompt_len = prompt_len
        self.detok = detok
        self.text = text
        self.tpl = tpl
        self.stream = stream
        self.include_usage = include_usage
        self.include_stop = include_stop
        self.stop_ids = stop_ids  # those a finish reports as `stop_reason` (`App.stop_reasons`)
        self.max_tokens = max_tokens
        self.n_out = 0
        self.first_sent = False
        self.done = False
        self.cancelled = False
        self.counted = False  # `client.on_finished` called for it (`finish`)
        self.queued = False  # its prompt counts in `client.queued_tokens`: sent, no output yet (`EngineClient._enter`)
        self.left = False  # cancelled because the client disconnected
        self.finish_reason: str | None = None
        self.stop_reason: str | int | None = None
        self.pending: list[bytes] = []
        self.waiter: asyncio.Future | None = None
        # The frontend's clock: the request received (its handler's start), its body read, it parsed and checked
        # (`t_arrival`: TTFT and e2e start there; `App.chat`), rendered, tokenized; its first and
        # last outputs received, and its finish received (e2e's end). The engine's: queued (`Scheduler.add`),
        # admitted, its first step launched, its first and last outputs sent (the engine-side series)
        self.t_recv, self.t_rendered, self.t_tokenized = t_recv, t_rendered, t_tokenized
        self.t_body = self.t_arrival = t_recv
        self.t_admit = self.t_launch = self.runner_ns = self.t_first_out = self.t_first_recv = self.t_queued = 0
        self.t_last = self.t_last_out = self.t_end = 0
        self.cached = 0  # prompt tokens the engine served from the prefix cache
        self.request_id = ""
        self.n_images = 0
        self.image_ns = 0  # `Prepared.image_ns`, left out of the TTFT overhead
        self.t_fetched = self.t_sized = self.t_pixels = 0
        self.error = "the engine rejected or failed the request"  # what an error finish tells the client
        self.error_status = 500
        # MTP: its verify steps and the tokens they delivered, as the engine reports them with its finish (0: none)
        self.verify_steps = self.verify_tokens = 0
        # Logprobs (`output.logprobs`; None: not asked): the entries not sent yet (a stream's step; the whole answer's)
        self.lp: RequestLogprobs | None = None
        self.lp_entries: list[bytes] = []
        # return_token_ids (None: not asked): the output's ids so far, the prompt's, and a stream's step not sent yet
        self.out_ids: list[int] | None = None
        self.prompt_ids: list[int] | None = None
        self.step_ids: list[int] = []
        # A request with tools or a reasoning parser: its parser, fed what the stop strings release
        # (`output.tool_calls`, `output.reasoning`; None: none); `tooled`: it has tools (the tool-parser metric)
        self.tools: ToolFeed | None = None
        self.tooled = False
        # A reasoning parser (`App._chat_request`): where the output starts (its case), the parser not built yet for
        # an output that starts as content without tools (`lazy`: built at the first marker or special token, as
        # until then the output is its text), and include_reasoning false (`null_reasoning`: vLLM's
        # `"reasoning": null` on each delta, and no thought sent)
        self.case = "a"
        self.lazy: tuple | None = None
        self.lead = 0  # the output's leading whitespace so far, held while the output is still content (`lazy_take`)
        self.reasoning = self.null_reasoning = False

    def on_output(self, toks: list[int], fin: int, now: int, sent: int, lp: tuple | None = None) -> bool:
        """New tokens (and maybe a finish) from one engine step, received at `now` and sent by the engine at `sent`,
        with their logprobs (`lp`: each one's logprob, top ids and their logprobs; requests that asked). Returns True if
        the engine must abort it."""
        if self.done:
            return False
        if self.n_out == 0:
            self.t_first_recv = now
        else:  # ITL: the engine's send times of a request's consecutive outputs, not the frontend's receipt times
            self.client.metrics.itl.observe((sent - self.t_last_out) / 1e9)
        self.t_last, self.t_last_out = now, sent
        stop_tok = None
        out = toks
        if fin == FinishReason.STOP and toks and not self.include_stop:
            stop_tok = toks[-1]  # the EOS or stop token itself is not detokenized
            toks = toks[:-1]
        hit = None
        consumed = 0
        feed = self.tools
        lazy = self.lazy
        first = self.n_out == 0
        for t in toks:
            consumed += 1
            piece = self.detok.step(t)
            if lazy is not None and (t in lazy[0].watch or (first and consumed == 1 and lazy[0].name == GEMMA4 and (
                    piece == "thought" or piece.startswith(LABEL)))):
                feed = self.start_parser(first and consumed == 1)
                lazy = None
            if feed is not None:
                feed.push(t, piece, len(self.text.text) + len(piece))
            if piece and self.text.append(piece) is not None:
                hit = self.text.stop_reason
                break
        # A stop string's tokens end at the one completing it: the step's stop token after it is not output
        counted = consumed + (stop_tok is not None and hit is None)
        self.n_out += counted
        if self.lp is not None and lp is not None and counted:  # the output tokens' entries (the stop token's too)
            self.lp_entries += self.lp.entries(out[:counted], lp[0][:counted], lp[1][:counted], lp[2][:counted])
        if self.out_ids is not None and counted:  # return_token_ids: the stop token's id too
            step = out[:counted]
            self.out_ids += step
            self.step_ids += step
        if hit is not None:
            if feed is not None:
                feed.cut(len(self.text.text))
            self.finish("stop", hit, now)
            self.wake()
            return fin == FinishReason.NONE
        if fin:
            reason = OPENAI_FINISH[FinishReason(fin)]
            last = out[-1] if fin == FinishReason.STOP and out else None  # the stop token, its text kept or not
            stop_reason = last if last is not None and last in self.stop_ids else None
            if feed is not None and stop_tok is not None:  # the parser sees the stop token (its text is none)
                feed.push(stop_tok, "", len(self.text.text))
            self.finish(reason, stop_reason, now)
        elif self.stream:
            if not self.first_sent:
                self.pending.append(self.tpl.role(self.prompt_ids))
                self.first_sent = True
            if feed is not None:
                self.tool_event()
                self.wake()
                return False
            delta = self.lazy_take() if self.reasoning and not self.text.sent else self.text.take(final=False)
            if self.reasoning:  # a parsed reply whose output is still content: vLLM's deltas (`delta_json`)
                lp_b = content(self.lp_entries) if self.lp is not None and self.lp_entries else None
                ids = self.step_ids if self.out_ids is not None and self.step_ids else None
                if delta and lp_b is None and ids is None and not self.null_reasoning:
                    self.pending.append(self.tpl.event(delta))  # the same bytes as `delta_json`'s, sooner
                elif delta or lp_b is not None or ids is not None:
                    self.pending.append(self.tpl.delta(delta_json(delta, [], None if self.null_reasoning else ""),
                                                       lp_b, ids))
                self.lp_entries = [] if lp_b is not None else self.lp_entries
                if ids is not None:
                    self.step_ids = []
            elif self.out_ids is not None:  # a step's tokens go out with their ids (and logprobs), text or not
                if self.step_ids:
                    lp_b = content(self.lp_entries) if self.lp is not None and self.lp_entries else None
                    self.pending.append(self.tpl.event(delta, lp_b, self.step_ids))
                    self.lp_entries, self.step_ids = [], []
            elif self.lp is not None and self.lp_entries:  # a step's tokens go out with their logprobs, text or not
                self.pending.append(self.tpl.event(delta, content(self.lp_entries)))
                self.lp_entries = []
            elif delta:
                self.pending.append(self.tpl.event(delta))
        self.wake()
        return False

    def lazy_take(self) -> str:
        """`text.take` for a parsed reply whose output is still content, before it sent any: whitespace before the
        content's first other character is held, as the parse holds it (`output.reasoning`), until text after it can
        be released (a stop string may hold some back); then it goes with that text, or at the finish. The leading
        whitespace is scanned once, as it comes."""
        text = self.text.text
        self.lead = _WS.match(text, self.lead).end()
        if self.lead == len(text):
            return ""
        delta = self.text.take(final=False)
        if self.text.sent <= self.lead:  # only whitespace released: still held
            self.text.sent = 0
            return ""
        return delta

    def tool_event(self) -> None:
        """A stream's step with a parser: what the stop strings released goes to the parser, and its deltas out (with
        the step's logprobs and ids, text or not: a token whose text is held, or a marker, still sends its entries)."""
        feed = self.tools
        self.text.take(final=False)  # advances what the stop strings release
        feed.release(self.text.sent)
        reasoning, content_, calls = feed.take()
        m = self.client.metrics.tool_parser
        if m is not None and self.tooled:
            m.inc(1.0, "streaming", "tool_call" if calls else "no_tool_call", "chat_completions")
        if self.null_reasoning:
            reasoning = None
        lp_b = content(self.lp_entries) if self.lp is not None and self.lp_entries else None
        ids = self.step_ids if self.out_ids is not None and self.step_ids else None
        if content_ or calls or reasoning or lp_b is not None or ids is not None:
            self.pending.append(self.tpl.delta(delta_json(content_, calls, reasoning), lp_b, ids))
        self.lp_entries = [] if lp_b is not None else self.lp_entries
        if ids is not None:
            self.step_ids = []

    def start_parser(self, first: bool) -> ToolFeed:
        """The reasoning parser of an output that started as content (`lazy`), built at its first marker or special
        token (`first`: that token is the output's first): the text before is content, the sent part already out."""
        fam, case, skip = self.lazy
        self.lazy = None
        p = ReasoningParser(fam, case, None, skip)
        p.first = first and p.first
        text, sent = self.text.text, self.text.sent
        if sent:
            p.preload(text[:sent])
        feed = self.tools = ToolFeed(p)
        if len(text) > sent:
            feed.push(-1, text[sent:], len(text))  # text alone (no token's id)
        return feed

    def reasoning_tokens(self) -> int:
        """`usage.completion_tokens_details.reasoning_tokens` (`output.reasoning`, "The count")."""
        feed = self.tools
        return feed.parser.tokens if feed is not None and feed.reasoning else 0

    def finish(self, reason: str, stop_reason: str | int | None, now: int = 0) -> None:
        """`now`: when the output that finished it was received (0: it finished otherwise, now). `reason` is the
        engine's (the metrics' and the log's); with tools, the response's may be `tool_calls`."""
        self.done = True
        self.t_end = now or clock()
        self.finish_reason = reason
        self.stop_reason = stop_reason
        feed = self.tools
        if feed is not None and reason != "error":
            feed.release(len(self.text.text))
            feed.parser.finish()
        if self.stream:
            rest = self.text.take(final=True) if reason != "error" else ""  # an error sends no text
            if reason == "error":
                kind = "ServiceUnavailableError" if self.error_status == 503 else "InternalServerError"
                self.pending.append(b"data: " + _enc.encode({"error": {"message": self.error, "type": kind,
                                                                       "code": self.error_status}}) + b"\n\n")
            else:
                if not self.first_sent:
                    self.pending.append(self.tpl.role(self.prompt_ids))
                lp = None
                if self.lp is not None:
                    lp, self.lp_entries = content(self.lp_entries), []
                ids = None
                if self.out_ids is not None:
                    ids, self.step_ids = self.step_ids, []
                if feed is not None:
                    reasoning, content_, calls = feed.take()
                    m = self.client.metrics.tool_parser
                    if m is not None and self.tooled:
                        m.inc(1.0, "streaming", "tool_call" if calls else "no_tool_call", "chat_completions")
                    self.pending.append(self.tpl.final_delta(
                        delta_json(content_, calls, None if self.null_reasoning else reasoning),
                        api_finish_reason(reason, feed.parser), stop_reason, lp, ids))
                elif self.reasoning:  # a parsed reply: `{}` when nothing is pending, as vLLM's
                    self.pending.append(self.tpl.final_delta(delta_json(rest, [], None if self.null_reasoning else ""),
                                                             reason, stop_reason, lp, ids))
                else:
                    self.pending.append(self.tpl.final(rest, reason, stop_reason, lp, ids))
            self.first_sent = True
            if self.include_usage:
                self.pending.append(self.tpl.usage(self.prompt_len, self.n_out, self.cached,
                                                   self.reasoning_tokens() if self.reasoning else None))
            self.pending.append(DONE)
        self.counted = True
        try:
            self.client.on_finished(self)
        finally:
            self.wake()  # whatever on_finished raised: the request is done, and its waiter must see it

    def fail(self, message: str, status: int = 503) -> None:
        """The server cannot finish it (`App.stop_now`, or a drain that timed out): aborted in the engine (harmless if
        it is gone) and finished with an error: `status` and `message` for a non-streamed response, an error event on a
        stream."""
        if self.done:
            return
        self.error, self.error_status = message, status
        self.client.abort([self.rid], ABORT_FAILED)
        self.finish("error", None)  # which wakes its waiter

    def crash(self) -> None:
        """Processing this request's outputs raised (`EngineClient._on_step`): aborted in the engine and finished with
        an error, whatever state the exception left it in. If that finish raises too, the request still ends: done
        with the error, out of `client.reqs`, its waiter woken. The fallback adds only what that finish left undone: a
        stream's error event and `[DONE]`, and the call to `on_finished`."""
        self.done = False
        self.error, self.error_status = "the server failed to process the request's output", 500
        queued = len(self.pending)
        try:
            self.client.abort([self.rid], ABORT_FAILED)
            self.finish("error", None)
        except Exception:
            log.exception("request %s: its error finish failed too", self.request_id or self.rid)
            self.done, self.finish_reason, self.t_end = True, "error", self.t_end or clock()
            if self.stream:
                if len(self.pending) == queued:
                    self.pending.append(b"data: " + _enc.encode({"error": {
                        "message": self.error, "type": "InternalServerError", "code": 500}}) + b"\n\n")
                if self.pending[-1] != DONE:
                    self.pending.append(DONE)
            if self.counted:  # `on_finished` itself raised
                self.client._leave(self.rid)
                return
            self.counted = True
            try:
                self.client.on_finished(self)  # its metrics, as any error finish's
            except Exception:
                self.client._leave(self.rid)
        finally:
            self.wake()

    def cancel(self, left: bool = True) -> None:
        """The client left (`left`), or the server stops: abort in the engine and stop streaming."""
        try:
            if not self.done:
                self.done = self.cancelled = True
                self.left = left
                self.finish_reason = "abort"
                self.client.abort([self.rid], ABORT_DISCONNECT if left else ABORT_FAILED)
                self.counted = True
                self.client.on_finished(self)
        finally:
            self.wake()  # as `finish`'s

    def wake(self) -> None:
        w = self.waiter
        if w is not None and not w.done():
            w.set_result(None)

    async def wait(self) -> None:
        """Until there are bytes to send or the request is done."""
        while not self.pending and not self.done:
            self.waiter = asyncio.get_running_loop().create_future()
            try:
                await self.waiter
            finally:
                self.waiter = None


class EngineClient:
    def __init__(self, ipc_path: str, index: int, metrics: Metrics, inline_threshold: int = 256,
                 debug_records: int = 0, spin_ms: float = 0.0):
        self.index = index
        self.spin_ns = int(spin_ms * 1e6)
        self._spinning = False
        self._spin_until = 0
        self._loop: asyncio.AbstractEventLoop | None = None
        self.metrics = metrics
        self._sctx = zmq.Context()
        self.push = self._sctx.socket(zmq.PUSH)  # sends never block: no HWM, and the engine binds first
        ipc.configure_socket(self.push)
        self.push.connect(ipc.engine_input_addr(ipc_path))
        self.pull = self._sctx.socket(zmq.PULL)
        ipc.configure_socket(self.pull)
        self.pull.connect(ipc.frontend_output_addr(ipc_path, index))
        self.enc = ipc.Encoder(inline_threshold)
        self.dec = ipc.Decoder()
        # The requests sent to the engine and not yet finished or aborted (in at `add`, out at `_leave` only), and the
        # prompt tokens of those of them without an output yet: vLLM's queue caps count these (`App.admit`)
        self.reqs: dict[int, RequestState] = {}
        self.queued_tokens = 0
        self._next_rid = 1
        self._fd: int | None = None
        self.engine_stats: EngineStats | None = None
        self.records: deque[dict[str, Any]] | None = deque(maxlen=debug_records) if debug_records else None
        self.on_released: Callable[[list[int]], None] | None = None  # pixels the engine no longer needs
        self.step_samples: deque[tuple[int, int, int, int]] | None = deque(maxlen=100_000) if debug_records else None
        self._seq = self._acked = 0  # the last StepOutputs processed, and the last acknowledged
        self.gc = GcStats("frontend")  # this process's collections: installed by `start`

    def new_rid(self) -> int:
        rid = self._next_rid
        self._next_rid += 1
        return rid

    def start(self) -> None:
        """Read engine outputs from the event loop: a reader callback on the socket's FD (no task, no futures)."""
        loop = self._loop = asyncio.get_running_loop()
        self.gc.install()
        self._fd = self.pull.get(zmq.FD)
        loop.add_reader(self._fd, self._on_readable)
        loop.call_soon(self._on_readable)  # the FD is edge-triggered: drain whatever arrived before

    async def close(self) -> None:
        self.gc.uninstall()
        if self._fd is not None:
            asyncio.get_running_loop().remove_reader(self._fd)
        try:
            for st in list(self.reqs.values()):
                try:  # each request is cancelled and woken, whatever another's cancel raised
                    st.cancel(left=False)
                except Exception:
                    log.exception("request %d: its cancel at shutdown failed", st.rid)
        finally:
            self.push.close(0)
            self.pull.close(0)
            self._sctx.term()

    def add(self, st: RequestState, msg: AddRequest) -> None:
        self._enter(st)
        self.push.send_multipart(self.enc.encode(msg), copy=False)
        if self.spin_ns and not self._spinning and self._loop is not None:
            self._spinning = True
            self._loop.call_soon(self._spin)

    def _spin(self) -> None:
        """Keep the event loop polling instead of sleeping in epoll: a ready callback per loop turn makes it poll
        with a zero timeout, so this core stays warm (no wake-up latency on the TTFT path) and the engine's outputs
        are read as they arrive. Runs while requests are in flight and `spin_ns` after the last one."""
        self._on_readable()
        now = clock()
        if self.reqs:
            self._spin_until = now + self.spin_ns
        if now < self._spin_until:
            self._loop.call_soon(self._spin)
        else:
            self._spinning = False

    def _enter(self, st: RequestState) -> None:
        """A request sent to the engine: in `reqs`, its prompt in `queued_tokens` until its first output."""
        self.reqs[st.rid] = st
        self.queued_tokens += st.prompt_len
        st.queued = True

    def _leave(self, rid: int) -> None:
        """A request out of `reqs`, the only place it leaves (an abort, a finish, a crash): its prompt out of
        `queued_tokens` if it still counted there. A request may leave twice (`cancel` and `fail` abort, then finish;
        a stop string's abort is followed by its finish): the second finds nothing."""
        st = self.reqs.pop(rid, None)
        if st is not None and st.queued:
            st.queued = False
            self.queued_tokens -= st.prompt_len

    def abort(self, rids: list[int], reason: int = ABORT_DISCONNECT) -> None:
        for rid in rids:
            self._leave(rid)
        self.push.send_multipart(self.enc.encode(Abort(frontend=self.index, rids=rids, reason=reason)), copy=False)

    def _on_readable(self) -> None:
        # ZMQ's FD only says "events may have changed": drain until ZMQ_EVENTS reports no message.
        pull, dec = self.pull, self.dec
        while pull.get(zmq.EVENTS) & zmq.POLLIN:
            frames = pull.recv_multipart(zmq.NOBLOCK, copy=False)
            try:
                msg = dec.decode(frames)
                if isinstance(msg, StepOutputs):
                    try:
                        self._on_step(msg)
                    finally:
                        self._seq = msg.seq
            except Exception:
                log.exception("failed to process engine outputs")
        if self._seq - self._acked >= ACK_EVERY:
            self._acked = self._seq
            self.push.send_multipart(self.enc.encode(Ack(frontend=self.index, seq=self._seq)), copy=False)

    def _on_step(self, msg: StepOutputs) -> None:
        """A step's outputs: the requests' tokens first, then the pixels the engine released and its statistics, each
        on its own, so that a failure in one loses none of the others (logged, `step_output_errors`)."""
        try:
            self._deliver(msg)
        finally:
            m = self.metrics
            if msg.released and self.on_released is not None:
                try:
                    self.on_released(msg.released)
                except Exception:
                    log.exception("releasing the pixels of %d requests failed", len(msg.released))
                    m.step_output_errors.inc(1.0, "release")
            if msg.stats is not None:
                try:
                    self._on_stats(msg.stats)
                except Exception:
                    log.exception("the engine's statistics failed to update the metrics")
                    m.step_output_errors.inc(1.0, "stats")

    def _deliver(self, msg: StepOutputs) -> None:
        now = clock()
        reqs = self.reqs
        if len(msg.first_rids):
            queued = msg.first_t_queued_ns or [0] * len(msg.first_rids)
            for rid, adm, lau, run, cached, que in zip(msg.first_rids.tolist(), msg.first_t_admit_ns.tolist(),
                                                       msg.first_t_launch_ns.tolist(), msg.first_runner_ns.tolist(),
                                                       msg.first_cached.tolist(), queued):
                st = reqs.get(rid)
                if st is not None:
                    st.t_admit, st.t_launch, st.runner_ns, st.t_first_out = adm, lau, run, msg.t_sent_ns
                    st.cached, st.t_queued = cached, que
                    if st.queued:  # its first output: out of the prefill backlog
                        st.queued = False
                        self.queued_tokens -= st.prompt_len
        for rid, n, t in zip(msg.spec_rids, msg.spec_rows, msg.spec_tokens):  # before their finishes are processed
            st = reqs.get(rid)
            if st is not None:
                st.verify_steps, st.verify_tokens = n, t
        n_new = msg.n_new.tolist()
        fins = msg.finish.tolist()
        toks = msg.tokens.tolist()
        lps = msg.logprobs
        if lps is not None:
            lps = (lps.token.tolist(), lps.top_ids.tolist(), lps.top.tolist())
        aborts = []
        off = 0
        for i, rid in enumerate(msg.rids.tolist()):
            n = n_new[i]
            t = toks[off:off + n] if n != 1 else [toks[off]]
            st = reqs.get(rid)
            if st is None:
                off += n
                continue
            lp = (lps[0][off:off + n], lps[1][off:off + n], lps[2][off:off + n]) if lps is not None else None
            off += n
            try:
                if st.on_output(t, fins[i], now, msg.t_sent_ns, lp):
                    aborts.append(rid)
            except Exception:  # this request's alone: it finishes with an error, the others go on
                log.exception("request %s: processing its output failed; it finishes with an error",
                              st.request_id or rid)
                if rid in reqs:  # it had not finished
                    st.crash()
        if aborts:  # stop strings
            self.abort(aborts, ABORT_STOP)

    def _on_stats(self, s: EngineStats) -> None:
        m = self.metrics
        self.engine_stats = s
        self.gc.maybe_warn()
        m.running.set(s.running)
        m.waiting.set(s.waiting)
        m.kv_usage.set(s.kv_usage)
        m.state_usage.set(s.state_usage)
        m.checkpoint_usage.set(s.checkpoint_usage)
        m.prefix_queries.set_total(s.prefix_queries)
        m.prefix_hits.set_total(s.prefix_hits)
        m.preempted_hits.set_total(s.readmit_hits)
        m.ckpt_hits.set_total(s.ckpt_hits)
        m.ckpt_misses.set_total(s.ckpt_misses)
        for source, n in zip(CHECKPOINT_SOURCES, s.ckpts_taken):
            m.ckpts_taken.set_total(n, source)
        for source, n in zip(CHECKPOINT_SOURCES, s.ckpt_resumes):
            m.ckpt_resumes.set_total(n, source)
        for i, n in enumerate(s.ckpts_evicted):
            m.ckpt_evictions.set_total(n, CHECKPOINT_SOURCES[i // 2], ("no", "yes")[i % 2])
        m.ckpts_skipped.set_total(s.ckpts_skipped)
        m.cache_waits.set_total(s.cache_waits)
        for cause, n in zip(LOST_CAUSES, s.lost_tokens):
            m.lost_tokens.set_total(n, cause)
        m.refines.set_total(s.refines)
        m.precise_tokens.set_total(s.precise_tokens)
        m.precise_hits.set_total(s.precise_hits)
        m.precise_steps.set_total(s.precise_steps)
        m.evicted_pages.set_total(s.evicted_pages)
        m.evicted_ckpts.set_total(s.evicted_ckpts)
        m.evicted_window_pages.set_total(s.evicted_window_pages)
        m.windows.set(s.windows)
        m.cached_pages.set(s.cached_pages)
        m.failed.set_total(s.num_failed)
        m.attention_replans.set_total(s.attention_replans)
        m.attention_plans_kept.set_total(s.attention_plans_kept)
        m.attention_plans_unsplit.set_total(s.attention_plans_unsplit)
        m.vision_plan_waits.set_total(s.vision_plan_waits)
        head_steps = m.grammar_head_steps  # the grammar-aware heads: zeros from a runner without them
        head_steps.set_total(s.grammar_head_sparse, "sparse", "union")
        head_steps.set_total(s.grammar_head_dense_off, "dense", "off")
        head_steps.set_total(s.grammar_head_dense_unconstrained, "dense", "unconstrained")
        head_steps.set_total(s.grammar_head_dense_over_cap, "dense", "over_cap")
        if s.grammar_head_union_counts:
            m.grammar_head_union_tokens.set_cumulative(s.grammar_head_union_counts, s.grammar_head_union_sum)
        m.draft_universe_tokens.set(s.draft_universe_tokens)
        m.draft_universe_steps.set_total(s.draft_universe_steps)
        m.draft_universe_rebuilds.set_total(s.draft_universe_rebuilds)
        m.preemptions.set_total(s.num_preemptions)
        for reason, n in zip(ABORT_REASONS, s.aborts):
            m.engine_aborts.set_total(n, reason)
        m.gpu_busy.set_total(s.gpu_busy_ns / 1e9)
        m.gpu_idle.set_total(s.gpu_idle_ns / 1e9)
        m.gpu_sample.set_total(s.gpu_sample_ns / 1e9)
        m.boundary_wait.set_total(s.boundary_ns / 1e9)
        for phase, ns in zip(PHASES, s.phase_ns):
            m.engine_phase.set_total(ns / 1e9, phase)
        if s.gc_ns:
            m.set_gc("engine", s.gc_ns, s.gc_collections)
        if s.encoder_images or s.encoder_hits:
            m.encoder_hits.set_total(s.encoder_hits)
            m.encoder_misses.set_total(s.encoder_images)
            m.encoder_tokens.set_total(s.encoder_tokens)
            m.encoder_time.set_total(s.encoder_ns / 1e9)
            if s.step_encoder_ns:
                m.image_stage.observe(s.step_encoder_ns / 1e9, "vit")
        m.backpressure_pauses.set_total(s.backpressure_pauses)
        m.dropped_samples.set_total(s.dropped_samples)
        m.paused_frontends.set(s.paused_frontends)
        m.backpressure_waits.set_total(s.backpressure_waits)
        if s.structured_requests or s.grammar_rejects:
            m.structured.set_total(s.structured_requests)
            m.grammar_compiles.set_total(s.grammar_compiles)
            m.grammar_compile_time.set_total(s.grammar_compile_ns / 1e9)
            m.grammar_rejects.set_total(s.grammar_rejects)
            m.grammar_masks.set_total(s.grammar_masks)
            m.grammar_mask_time.set_total(s.grammar_mask_ns / 1e9)
            m.grammar_errors.set_total(s.grammar_errors)
            m.grammar_end_waits.set_total(s.grammar_end_waits)
            m.grammar_forced_requests.set_total(s.grammar_forced_requests)
            m.grammar_forced_tokens.set_total(s.grammar_forced_tokens)
            if s.grammar_eos_free:
                m.grammar_eos_free.set_total(s.grammar_eos_free[0], "no")
                m.grammar_eos_free.set_total(s.grammar_eos_free[1], "yes")
            m.grammar_eos_free_gated.set_total(s.grammar_eos_free_gated)
            for c, n in zip(MASK_CLASSES, s.spec_first_rejections):
                m.spec_first_rejections.set_total(n, c)
            if s.spec_window_mask_counts:
                m.spec_window_mask_tokens.set_cumulative(s.spec_window_mask_counts, s.spec_window_mask_sum)
        if s.spec_accepted_per_pos:
            sm = m.spec_metrics(len(s.spec_accepted_per_pos))
            sm.num_drafts.set_total(s.spec_drafts)
            sm.num_draft_tokens.set_total(s.spec_draft_tokens)
            sm.num_accepted_tokens.set_total(s.spec_accepted)
            for i, n in enumerate(s.spec_accepted_per_pos):
                sm.num_accepted_tokens_per_pos.set_total(n, str(i))
            for i, n in enumerate(s.spec_verified_per_pos):
                sm.num_verified_per_pos.set_total(n, str(i))
            for c, n in enumerate(s.spec_steps_by_count):
                if c:
                    sm.steps_by_count.set_total(n, str(c))
        if s.step_rows_counts:
            m.step_decode_rows.set_cumulative(s.step_rows_counts, s.step_rows_sum)
            m.step_encoder_images.set_cumulative(s.step_images_counts, s.step_images_sum)
        m.page_waits.set_total(s.page_waits)
        if s.step_tokens:
            m.iteration_tokens.observe(s.step_tokens)
            m.step_cpu.observe(s.step_cpu_ns / 1e9)
            m.step_runner_calls.observe(s.step_runner_calls_ns / 1e9)
            m.step_runner.observe(s.step_runner_ns / 1e9)
            if self.step_samples is not None:
                self.step_samples.append((s.step_tokens, s.step_cpu_ns, s.step_runner_calls_ns, s.step_runner_ns))

    def on_finished(self, st: RequestState) -> None:
        """A request ended: the tokens it generated count, and unless its client abandoned it, the finished-request
        statistics (`request_success`, e2e and the per-request histograms). An abort, which a client's disconnect
        takes, updates none of them; `llm_mini_infer:client_disconnects_total` counts the abandoned ones.

        The series start and end where vLLM's do (the frontend's `t_arrival` and its receipt of the finish
        for e2e; the engine's clock for the others): queue from the engine's receipt of the request to its admission,
        prefill to the engine's first output, decode from there to its last, inference both, the time per output
        token decode's over the tokens after the first (0 for a request of one token)."""
        self._leave(st.rid)
        m = self.metrics
        m.generation_tokens.inc(st.n_out)
        if st.left:
            return
        m.request_success.inc(1.0, st.finish_reason or "abort")
        m.request_prompt_tokens.observe(st.prompt_len)
        m.request_max_tokens.observe(st.max_tokens)
        m.request_generation_tokens.observe(st.n_out)
        st.t_end = st.t_end or clock()  # a request cancelled as the server stops: now
        m.e2e.observe((st.t_end - st.t_arrival) / 1e9)
        if st.t_admit:
            if st.t_queued:
                m.queue_time.observe(max(0, st.t_admit - st.t_queued) / 1e9)
            if st.t_first_out:
                m.inference_time.observe((st.t_last_out - st.t_admit) / 1e9)
                m.prefill_time.observe((st.t_first_out - st.t_admit) / 1e9)
                m.decode_time.observe((st.t_last_out - st.t_first_out) / 1e9)
        if st.t_first_out:  # observed for each finished request that produced an output (0 for one token)
            m.tpot.observe((st.t_last_out - st.t_first_out) / 1e9 / (st.n_out - 1) if st.n_out > 1 else 0.0)
        if st.verify_steps and m.spec is not None:
            m.spec.tokens_per_verify_step.observe(st.verify_tokens / st.verify_steps)

    def record_first_byte(self, st: RequestState, streaming: bool = True) -> None:
        """The first bytes went out: TTFT and its overhead breakdown (architecture.md, "Observability"). A
        non-streaming response goes out only at the end, so it records no `deliver` stage, TTFT overhead or
        /debug/stages entry."""
        m = self.metrics
        t = clock()
        m.ttft.observe(((st.t_first_recv or t) - st.t_arrival) / 1e9)
        m.prompt_tokens.inc(st.prompt_len)
        if not st.t_first_out:
            return
        stage = m.stage
        stage.observe((st.t_body - st.t_recv) / 1e9, "receive")
        stage.observe((st.t_rendered - st.t_body) / 1e9, "render")
        stage.observe((st.t_tokenized - st.t_rendered) / 1e9, "tokenize")
        stage.observe((st.t_admit - st.t_tokenized) / 1e9, "to_admit")
        stage.observe((st.t_launch - st.t_admit) / 1e9, "to_launch")
        stage.observe((st.t_first_out - st.t_launch) / 1e9, "to_first_token")
        if not streaming:
            return
        stage.observe((t - st.t_first_out) / 1e9, "deliver")
        m.ttft_overhead.observe((t - st.t_body - st.runner_ns - st.image_ns) / 1e9)
        if self.records is not None:
            rec = {"prompt_tokens": st.prompt_len, "recv": st.t_recv, "body": st.t_body, "arrival": st.t_arrival,
                   "queued": st.t_queued, "rendered": st.t_rendered,
                   "tokenized": st.t_tokenized, "admitted": st.t_admit, "launched": st.t_launch,
                   "first_token_out": st.t_first_out, "first_recv": st.t_first_recv, "first_byte": t,
                   "runner_ns": st.runner_ns}
            if st.n_images:
                rec.update(images=st.n_images, fetched=st.t_fetched, sized=st.t_sized, pixels=st.t_pixels,
                           image_ns=st.image_ns)
            self.records.append(rec)


def sampling_seed(seed: int | None) -> int | None:
    """A request's seed as the engine takes it: unsigned; -1 is unseeded."""
    return None if seed is None or seed == -1 else seed & ((1 << 64) - 1)


MIN_TEMPERATURE = 0.01


def sampling_temperature(temperature: float) -> float:
    """A request's temperature as the engine takes it: one in (0, 0.01) is raised to 0.01, as vLLM's
    SamplingParams raises it, so only 0 is greedy (the sampler's 1e-5 threshold then sees no other)."""
    return MIN_TEMPERATURE if 0 < temperature < MIN_TEMPERATURE else temperature

