#!/usr/bin/env python3
"""A load generator for any OpenAI-compatible chat completions server (llm-mini-infer, vLLM, SGLang, ...): the same
requests at the same arrivals against each engine, so they can be compared on one machine.

    uv run python bench/load/run.py --base-url http://127.0.0.1:8000 --model <served name> \\
        --workload scene --tokenizer <model dir or Hugging Face repo id> --rates 2,4 --slots 32,64
    uv run python bench/load/run.py --base-url ... --model ... --workload chat --concurrency 1,8,32
    uv run python bench/load/run.py --base-url ... --model ... --workload catalog --rates 2,4 --slots 32,64
    uv run python bench/load/run.py --base-url ... --model ... --workload <folder> --rates 2 --slots 32

**Workloads** (`--workload`):
- `scene` (the default): the host check's synthetic workload (`llm_mini_infer.synthetic`): per request one generated
  test-pattern image (`--tall-share` of them tall), a system prompt of `--prompt-tokens` tokens (counted with
  `--tokenizer`) shared by every request, a short user text and a strict JSON schema
  (`llm_mini_infer.synthetic.scene_schema`). The image budget is the server's own setting (`images.max_soft_tokens`):
  a client cannot set it.
- `chat`: plain chat, text only: each request a different prompt of `--input-tokens` tokens, no schema. With
  `--turns N` (closed loop only) each client holds a conversation: its next request is the previous messages, the
  reply it got, and a user message of `--tool-tokens` tokens of random words (seeded by the seed, the conversation and
  the turn); after N turns it starts a new conversation. `--think S [--think-sigma X]` makes a client pause after a
  reply, before building the next turn, for S * exp(X * z) seconds (z a standard normal drawn from the seed, the
  conversation and the turn: every engine's clients pause alike; a pause ends at the end of the window and tail at the
  latest); none after the last turn or a failed request. With think time the conversations in flight are fewer than
  `--concurrency`: the report's `in_flight` shows how many were.
  The replies are kept in memory for the next turn, never in the report. `--shared-prefix-tokens N` starts every
  request's prompt (with `--turns`, each conversation's first message) with the same N tokens of random words, seeded
  by the seed alone and followed by a blank line: the server's prefix cache holds them after the first requests, so
  a later request's TTFT is a short turn over a long cached history.
- `catalog`: text only, structured: each request asks for a numbered catalog's entries
  (tests/support/catalog_schema.py's prompts, so no two are the same), with the catalog schema
  (tests/data/catalog.schema.json) as a strict response format, its list of entries held to at least
  `CATALOG_MIN_ENTRIES`.
- a folder in the host check's mounted layout (docs/operations.md, "The host check"). A run that needs more images
  than the folder holds stops before it starts (rates) or ends its load early (concurrency).

**Loads**, each measured in turn after `--warmup` unmeasured seconds, over `--duration` seconds:
- `--rates R1,R2,...`: Poisson arrivals at each rate (requests/s). `--slots` bounds the requests in flight (one value,
  or one per rate): at most that many requests are in flight, and an arrival beyond
  that is dropped and counted, so overload shows as drops.
- `--concurrency C1,C2,...`: a closed loop of C clients, each sending its next request when the last one ends.
- `--concurrency C1,... --requests N`: each load is exactly N requests (workload indexes `offset .. offset + N - 1`),
  all measured, taken by the clients in turn. The window runs from the load's first send to its last finish, so
  `--warmup`, `--duration` and `--tail` do not apply to it (the report records them as given). Not with `--rates` or
  `--turns`.
- `--first-index N`: the workload index of the run's first measured request (default 0); each later load of the run
  continues from where the last ended. A server that has seen an index serves it again from its prefix cache, so several
  runs against one server must not repeat each other's indexes: give each run its own range (a request count apart).
  Not with `--find-rate` (a probe's indexes come from its rate) or a folder workload (its images are finite).
- `--warmup-requests W`: before the first load, W unrecorded requests at concurrency min(W, 4), from the workload built
  with `--seed + 1` at indexes from `WARMUP_OFFSET`, so every pass starts in the same state (the shared prefix cached,
  the server's first-request costs paid) and the measured indexes do not change. Default 20 with `--requests`, else 0.
  A failed warm-up request ends the run. A folder workload takes none (its images are finite).
- `--find-rate LO,HI --bound-file PATH [--resolution R] [--confirm N]`: a search for the highest Poisson rate (between
  LO and HI requests/s) at which the server stays within a bound, one `--slots` value (default 128). The bound file is
  JSON: `{"model": ..., "workload": "scene", "metric": "e2e_ms.p99", "value": <ms>, ...}` (its model and workload must
  be this run's; the metric is e2e_ms, ttft_ms or tpot_ms with p50, p90, p99 or mean; other keys are copied into the
  report). A probe is an open-loop load that passes when nothing failed or was dropped, the server served at least 95%
  of the offered rate, and the metric is within the value. The search probes LO, doubles up to HI until a probe fails,
  then bisects until the passing and failing rates are `--resolution` apart (default (HI - LO) / 8); `--confirm N`
  probes the best rate N more times (at most 9), and a failure there moves the best rate down. At most `MAX_PROBES`
  probes. A probe's arrivals depend on its rate only, and its requests on its rate and its repeat (the n-th probe of a
  rate starts `n * REPEAT_STRIDE` indexes above the rate's first, so a confirmation sends new requests, not ones the
  server has cached), so every engine sees the same ones at a given rate and repeat. The window of the top rate must
  fit `REPEAT_STRIDE` requests. The
  report holds a load per probe and a `search` block (the verdict: the best rate, or None); a failing probe at the top
  of the search is expected, and the exit code is 1 when a request failed in any probe, so read `search`, not the code.

**Tokens per verify step**, one definition for every engine: Σ over the load's successful requests (warm-up seconds,
window and tail: every request the load sent) of (completion tokens - 1), over the server's verify steps
(`llm_mini_infer.hostcheck.VERIFY_SERIES`: vLLM's drafts, SGLang's verify calls) over the same requests. Each load
reads `/metrics` once just before its first request and once after its last has finished and 1 s has passed, and its
entry's `spec` holds the series, the steps, the requests, the summed tokens, `tokens_per_verify` and `no_usage` (the
requests without a token count, left out of the sum); `spec` is null without both pages, a verify series or a step.
SGLang counts a request when it ends, so its figure is complete only then. llm-mini-infer also delivers tokens no
verify step makes (grammar-forced prefixes; under `ignore_eos`, drafts past a drafted EOS), so its ratio is above its
drafter's own acceptance on structured workloads: compare engines' acceptance on the chat workload.

`--meta JSON` (an object) is stored as the report's `meta`; `gpu` holds the first GPU's name, power limit and maximum,
SM clock, memory and driver when `nvidia-smi` answers, and is absent otherwise.

The requests that arrive (rates) or start (concurrency) inside the window are the measured ones; the load goes on
after the window until they have all finished (at most `--tail` seconds), so the last of them run under the same load.

**Output:** a table row per load, and one JSON report (`--out`) with an entry per load: the measured requests, how
many succeeded, were dropped or failed (by error), their finish reasons; latency percentiles in ms of the end-to-end
time, the time to the first streamed token, content or reasoning (TTFT), and the time per output token after the first
(TPOT); the reply's and the prompt's tokens (the server's `usage`); the prefix cache: the hit rate (the cached tokens
the server reports in `usage.prompt_tokens_details`, over the prompt tokens) and the uncached tokens per request, both
absent (None) from a server that does not report them, and by turn index in a session load (`cache_by_turn`);
throughput; and the requests in flight. Every answer of the load, warm-up and tail included, is kept by its workload
index (a session's later turns: "index.turn") as a digest of its content (the first 16 hex digits of its SHA-256, as
the host check's), its tokens and finish reason (`answers`), so that two runs can be compared answer by answer
(bench/load/compare.py; a session's later turns follow each run's own replies, so only its first turn compares). Of
the requests' and answers' contents nothing else is kept but the schema's name (from its title) and the prompt's and
schema's sizes (a session's replies live in memory until its next turn is built, then go). A short answer from a small
schema can be found again from its digest: a folder workload's report is as private as its answers. The server's
counters (its `/metrics`) are read as the window opens and as it closes, and their change is reported (`engine`,
`llm_mini_infer.hostcheck.engine_delta`); a page that cannot be read gives none, as does `--no-metrics`.

Requests are streamed (`stream_options.include_usage`) unless `--no-stream` (then there is no TTFT). The default
`--extra-body` turns a chat template's thinking off (`chat_template_kwargs`, which llm-mini-infer, vLLM and SGLang
accept).
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import dataclasses
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from llm_mini_infer import synthetic  # noqa: E402
from llm_mini_infer.hostcheck import (  # noqa: E402
    VERIFY_SERIES,
    counter,
    engine_delta,
    gpu_info,
    in_flight,
    mounted_inputs,
    pcts,
    poisson_arrivals,
    prometheus,
    synthetic_inputs,
    total,
)
from tests.support import paths  # noqa: E402

RESULTS = paths.RESULTS / "load"
WARMUP_OFFSET = 10**12  # the first workload index of the warm-up requests, far above any measured one
WARMUP_REQUESTS = 20  # the default warm-up of a count-bounded run or a rate search
MAX_PROBES = 16  # the most loads a rate search runs
PROBE_STRIDE = 100_000  # the workload indexes a probe's rate reserves: its first index is round(rate * 1000) * this
REPEAT_STRIDE = 10_000  # the indexes of one probe of a rate: its n-th repeat starts n * this above the rate's first
SETTLE_S = 1.0  # a load's closing counters are read this long after its last request ended
GPU_KEYS = ("name", "power_limit_w", "power_max_w", "sm_clock_mhz", "memory_mib", "driver")
REQUIRED_SERVED = 0.95  # a probe passes only when the server served this share of the offered rate
METRIC_KEYS = ("e2e_ms", "ttft_ms", "tpot_ms")
METRIC_STATS = ("p50", "p90", "p99", "mean")
DEFAULT_EXTRA = {"chat_template_kwargs": {"enable_thinking": False}}
WORDS = (
    "time", "year", "way", "day", "thing", "world", "school", "state", "group", "country", "problem", "hand", "part",
    "place", "case", "week", "system", "program", "question", "work", "number", "night", "point", "home", "water",
    "room", "area", "money", "story", "fact", "month", "lot", "study", "book", "eye", "job", "word", "issue", "side",
    "kind", "head", "house", "service", "power", "hour", "game", "line", "end", "law", "car", "city", "name", "team",
    "minute", "idea", "information", "back", "level", "office", "door", "art", "result", "change", "morning", "reason",
    "moment", "air", "force", "river", "garden", "window", "market", "bridge", "valley", "island", "forest", "letter",
    "music", "paper", "picture", "voice", "station", "winter", "summer", "bird", "horse", "table", "street", "stone",
    "cloud", "field", "road", "train", "lamp", "chair", "clock", "bottle", "apple", "orange", "bread", "salt",
    "coffee", "tea", "glass", "shell", "wave", "sand", "hill", "lake", "ocean", "storm", "rain", "snow", "wind",
    "leaf", "tree", "flower", "seed", "root", "branch",
)


# -- workloads ---------------------------------------------------------------------------------------------------------

@dataclass
class Workload:
    """`body(i)`: the i-th request's JSON body without the run's template (`Run.template`); `count`: the requests it
    can give (a folder's images), None when unbounded; `summary`: what the report records of it."""
    name: str
    body: Callable[[int], dict[str, Any]]
    count: int | None
    summary: dict[str, Any] = field(default_factory=dict)
    tool_message: Callable[[int, int], str] | None = None  # (conversation index, turn) -> the next user message


def load_tokenizer(name: str) -> Any:
    from tokenizers import Tokenizer

    path = Path(name) / "tokenizer.json"
    return Tokenizer.from_file(str(path)) if path.is_file() else Tokenizer.from_pretrained(name)


def image_turn(prompt: str, user_text: str, jpeg: bytes) -> list[dict[str, Any]]:
    url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
    return [{"role": "system", "content": [{"type": "text", "text": prompt}]},
            {"role": "user", "content": [{"type": "text", "text": user_text},
                                         {"type": "image_url", "image_url": {"url": url}}]}]


def schema_workload(inputs: Any, max_tokens: int) -> Workload:
    """Requests of one image each, the shared system prompt, the user text and the strict schema (`inputs`: the host
    check's `Inputs`)."""
    rf = {"type": "json_schema", "json_schema": {"name": inputs.schema_name, "strict": True, "schema": inputs.schema}}

    def body(i: int) -> dict[str, Any]:
        return {"messages": image_turn(inputs.prompt, inputs.user_text, inputs.frame(i)), "max_tokens": max_tokens,
                "response_format": rf}

    count = None if inputs.mode == "synthetic" else inputs.count
    return Workload("scene" if inputs.mode == "synthetic" else "folder", body, count,
                    {"mode": inputs.mode, **inputs.summary, "schema_name": inputs.schema_name,
                     "schema_bytes": len(json.dumps(inputs.schema, separators=(",", ":"))),
                     "prompt_chars": len(inputs.prompt), "max_tokens": max_tokens})


def tool_text(tok: Any, tokens: int, seed: int, conversation: int, turn: int) -> str:
    """A session's user message after a reply: `tokens` tokens of random words, seeded by the seed, the conversation's
    index and the turn, cut to that many tokens and decoded."""
    rng = random.Random(f"{seed}:{conversation}:{turn}")
    text = f"Tool result {conversation}.{turn}: " + " ".join(rng.choice(WORDS) for _ in range(tokens))
    return tok.decode(tok.encode(text, add_special_tokens=False).ids[:tokens])


def next_turn_messages(messages: list[dict[str, Any]], reply: str, user_text: str) -> list[dict[str, Any]]:
    """A conversation's next request: the previous messages, the assistant's reply, a new user message."""
    return [*messages, {"role": "assistant", "content": reply}, {"role": "user", "content": user_text}]


def shared_prefix(tok: Any, tokens: int, seed: int) -> str:
    """`tokens` tokens of random words seeded by the seed alone (the same for every request), cut to that many tokens
    and decoded, then a blank line that ends the prefix."""
    rng = random.Random(f"shared:{seed}")
    text = " ".join(rng.choice(WORDS) for _ in range(tokens))
    return tok.decode(tok.encode(text, add_special_tokens=False).ids[:tokens]) + "\n\n"


def chat_workload(tok: Any, input_tokens: int, max_tokens: int, seed: int, tool_tokens: int = 0,
                  shared_tokens: int = 0, prefix_seed: int | None = None) -> Workload:
    """Text requests, each prompt `input_tokens` tokens of random words (a request's index first, so no two share a
    prefix), cut to that many tokens and decoded; `tool_tokens`: the size of a session's follow-up messages;
    `shared_tokens`: tokens of one text every prompt starts with, before its own part, drawn from `prefix_seed`
    (default: `seed`; the warm-up's workload keeps the measured one's, so its requests cache the same prefix)."""
    prefix = shared_prefix(tok, shared_tokens, seed if prefix_seed is None else prefix_seed) if shared_tokens else ""

    def body(i: int) -> dict[str, Any]:
        rng = random.Random(seed * 1_000_003 + i)
        head = f"Request {i}. Repeat these words in reverse order: "
        text = head + " ".join(rng.choice(WORDS) for _ in range(input_tokens))
        ids = tok.encode(text, add_special_tokens=False).ids[:input_tokens]
        return {"messages": [{"role": "user", "content": prefix + tok.decode(ids)}], "max_tokens": max_tokens}

    summary = {"mode": "chat", "input_tokens": input_tokens, "max_tokens": max_tokens}
    if shared_tokens:
        summary["shared_prefix_tokens"] = shared_tokens
    return Workload("chat", body, None, summary,
                    tool_message=lambda conversation, turn: tool_text(tok, tool_tokens, seed, conversation, turn))


CATALOG_MIN_ENTRIES = 3
SCENE_MAX_TOKENS = 512  # a test pattern holds no object: the answers are short, and 512 leaves room
SCHEMA_MAX_TOKENS = 1024  # room for the largest document of the catalog schema or of a folder's schema
CHAT_MAX_TOKENS = 128


def catalog_workload(max_tokens: int) -> Workload:
    from tests.support import catalog_schema  # the repository's: only this workload needs the tests package

    schema = json.loads(catalog_schema.PATH.read_text())
    schema["properties"]["entries"]["minItems"] = CATALOG_MIN_ENTRIES
    rf = {"type": "json_schema", "json_schema": {"name": catalog_schema.NAME, "strict": True, "schema": schema}}

    def body(i: int) -> dict[str, Any]:
        return {"messages": [{"role": "user", "content": catalog_schema.prompt(i, numbered=True)}],
                "max_tokens": max_tokens, "response_format": rf}

    return Workload("catalog", body, None, {"mode": "catalog", "schema_name": catalog_schema.NAME,
                                          "schema_bytes": len(json.dumps(schema, separators=(",", ":"))),
                                          "min_entries": CATALOG_MIN_ENTRIES, "max_tokens": max_tokens})


def make_workload(args: argparse.Namespace) -> Workload:
    if args.workload == "catalog":
        return catalog_workload(args.max_tokens or SCHEMA_MAX_TOKENS)
    if args.workload not in ("scene", "chat"):
        folder = Path(args.workload)
        if not folder.is_dir():
            raise SystemExit(f"--workload {args.workload}: neither scene, chat, catalog nor a folder")
        try:
            inputs = mounted_inputs(folder)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        return schema_workload(inputs, args.max_tokens or SCHEMA_MAX_TOKENS)
    tok = load_tokenizer(args.tokenizer or args.model)
    if args.workload == "chat":
        return chat_workload(tok, args.input_tokens, args.max_tokens or CHAT_MAX_TOKENS, args.seed, args.tool_tokens,
                             getattr(args, "shared_prefix_tokens", 0), getattr(args, "prefix_seed", None))
    inputs = synthetic_inputs(lambda text: len(tok.encode(text).ids), args.prompt_tokens, args.tall_share)
    return schema_workload(inputs, args.max_tokens or SCENE_MAX_TOKENS)


# -- one request -------------------------------------------------------------------------------------------------------

def error_kind(status: int, text: str) -> str:
    """An HTTP error's kind: its status, then the OpenAI error's `type` (or `code`) when the body has one."""
    try:
        err = json.loads(text).get("error")
        kind = err.get("type") or err.get("code") if isinstance(err, dict) else None
    except (ValueError, AttributeError):
        kind = None
    return f"{status} {kind}" if kind else f"{status}"


def usage_cached(usage: dict[str, Any] | None) -> int | None:
    """The prompt tokens the server served from its cache (`usage.prompt_tokens_details.cached_tokens`); None when it
    does not say (no details, no field, or null), never 0."""
    details = (usage or {}).get("prompt_tokens_details")
    cached = details.get("cached_tokens") if isinstance(details, dict) else None
    return cached if isinstance(cached, int) and not isinstance(cached, bool) else None


async def send(session: Any, url: str, headers: dict[str, str], body: dict[str, Any], timeout: float) -> dict:
    """One request: its status, end-to-end and first-token times (ms), reply and prompt tokens (from `usage`), finish
    reason, cached prompt tokens (None when not reported) and its answer's digest (the content's SHA-256 as it
    arrives, 16 hex digits), or its error. A successful request also returns the reply's text (`reply`), for a session's
    next turn: the caller takes it out of the record."""
    import aiohttp

    rec: dict[str, Any] = {"status": 0}
    digest = hashlib.sha256()
    reply: list[str] = []
    t0 = time.perf_counter()
    try:
        async with session.post(url, json=body, headers=headers,
                                timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            rec["status"] = resp.status
            if resp.status != 200:
                rec["error"] = error_kind(resp.status, await resp.text())
            elif not body.get("stream"):
                out = await resp.json()
                rec["finish"] = out["choices"][0].get("finish_reason")
                content = (out["choices"][0].get("message") or {}).get("content") or ""
                digest.update(content.encode())
                reply.append(content)
                rec["usage"] = out.get("usage") or {}
            else:
                async for raw in resp.content:
                    line = raw.decode().strip()
                    if not line.startswith("data:") or line == "data: [DONE]":
                        continue
                    chunk = json.loads(line[5:])
                    if chunk.get("error"):
                        rec["error"] = "stream " + str(chunk["error"].get("type") or "error")
                    if chunk.get("usage"):
                        rec["usage"] = chunk["usage"]
                    for ch in chunk.get("choices") or []:
                        delta = ch.get("delta") or {}
                        if "ttft_ms" not in rec and any(delta.get(k) for k in ("content", "reasoning_content",
                                                                                 "reasoning")):
                            rec["ttft_ms"] = (time.perf_counter() - t0) * 1e3
                        if delta.get("content"):
                            digest.update(delta["content"].encode())
                            reply.append(delta["content"])
                        rec["finish"] = ch.get("finish_reason") or rec.get("finish")
                if "finish" not in rec and "error" not in rec:
                    rec["error"] = "stream ended without a finish reason"
    except TimeoutError:
        rec["error"] = "timeout"
    except Exception as exc:  # a connection error: a lost request
        rec["error"] = type(exc).__name__
    rec["e2e_ms"] = (time.perf_counter() - t0) * 1e3
    if rec["status"] == 200 and "error" not in rec:
        rec["digest"] = digest.hexdigest()[:16]
        rec["reply"] = "".join(reply)
    rec["cached_tokens"] = usage_cached(rec.get("usage"))
    rec["tokens"] = (rec.get("usage") or {}).get("completion_tokens")
    rec["prompt_tokens"] = (rec.get("usage") or {}).get("prompt_tokens")
    rec.pop("usage", None)
    return rec


# -- loads -------------------------------------------------------------------------------------------------------------

@dataclass
class Run:
    """What a load's requests share: the endpoint, the workload, the request template and the window."""
    session: Any
    url: str
    headers: dict[str, str]
    workload: Workload
    template: dict[str, Any]
    timeout: float
    warmup: float
    duration: float
    tail: float
    offset: int = 0  # the workload's next index: each load takes new ones (new images, new prompts)
    seeds: bool = False  # each request's seed its workload index
    metrics_url: str | None = None  # the server's /metrics; None: not read
    turns: int = 1  # a closed-loop client's requests per conversation
    think: float | None = None  # a conversation's pause (median seconds) between a reply and its next turn
    think_sigma: float = 1.0  # the shape of the pause's log-normal distribution
    seed: int = 0  # the pauses' seed

    def body(self, i: int) -> dict[str, Any]:
        """The request of workload index `offset + i`."""
        j = self.offset + i
        return {**self.template, **self.workload.body(j), **({"seed": j} if self.seeds else {})}


def think_time(run: Run, i: int, turn: int) -> float:
    """The pause after turn `turn` of the conversation of workload index `offset + i`: log-normal with median
    `run.think` and shape `run.think_sigma`, drawn from the seed, the conversation and the turn, so every engine's
    clients pause alike."""
    z = random.Random(f"{run.seed}:{run.offset + i}:{turn}:think").gauss(0.0, 1.0)
    return (run.think or 0.0) * math.exp(run.think_sigma * z)


def session_body(run: Run, prev: dict[str, Any], reply: str, i: int, turn: int) -> dict[str, Any]:
    """The request of turn `turn` (from 1) of the conversation of workload index `offset + i`: `prev`, the last request,
    with the reply and a new user message appended."""
    text = run.workload.tool_message(run.offset + i, turn)
    return {**prev, "messages": next_turn_messages(prev["messages"], reply, text)}


async def scrape(run: Run) -> tuple[float, dict[str, float] | None]:
    """(now, the server's counters now: its /metrics page parsed, or None when it is not read or cannot be)."""
    import aiohttp

    t = time.monotonic()
    if run.metrics_url is None:
        return t, None
    try:
        async with run.session.get(run.metrics_url, headers=run.headers,
                                   timeout=aiohttp.ClientTimeout(total=10)) as resp:
            return t, prometheus(await resp.text()) if resp.status == 200 else None
    except Exception:  # an error or a timeout loses the window's counters, never the run
        return t, None


def engine(marks: dict[str, tuple[float, dict[str, float] | None]], recs: list[dict]) -> dict[str, Any] | None:
    """What the server did between the two marks (`llm_mini_infer.hostcheck.engine_delta`), its per-request figures over
    the requests that succeeded between them; None without both pages."""
    (t0, a), (t1, b) = marks.get("start", (0.0, None)), marks.get("end", (0.0, None))
    if a is None or b is None:
        return None
    return engine_delta(a, b, sum(1 for r in recs if r.get("digest") is not None and t0 <= r["t_end"] < t1))


async def load_mark(run: Run, settled: bool = False) -> tuple[float, dict[str, float] | None]:
    """The counters bounding a whole load (not read without a metrics URL): `settled`: once `SETTLE_S` has passed after
    its last request ended, as a server may count a request after it has answered."""
    if run.metrics_url is None:
        return time.monotonic(), None
    if settled:
        await asyncio.sleep(SETTLE_S)
    return await scrape(run)


def spec_block(a: dict[str, float] | None, b: dict[str, float] | None, recs: list[dict]) -> dict[str, Any] | None:
    """Tokens per verify step over a whole load (module docstring): Σ (completion tokens - 1) over its successful
    requests, over the server's verify steps between the pages `a` and `b`; None without both pages, without a verify
    series or with no step."""
    if a is None or b is None or (series := counter(b, VERIFY_SERIES)) is None:
        return None
    steps = round(series[1] - total(a, series[0]))
    if steps <= 0:
        return None
    done = [r for r in recs if r.get("status") == 200 and not r.get("error")]
    counted = [r["tokens"] - 1 for r in done if r.get("tokens") is not None]
    return {"series": series[0], "verify_steps": steps, "requests": len(done),
            "completion_tokens_minus_one": sum(counted), "tokens_per_verify": round(sum(counted) / steps, 3),
            "no_usage": len(done) - len(counted)}


def arrivals(rate: float, warmup: float, duration: float, tail: float, seed: int) -> list[float]:
    end = warmup + duration + tail
    return [t for t in poisson_arrivals(rate, math.ceil(end * rate) + 1, seed) if t < end]


async def open_loop(run: Run, rate: float, slots: int, seed: int) -> dict[str, Any]:
    """Poisson arrivals at `rate`, at most `slots` requests in flight (an arrival beyond is dropped)."""
    at_list = arrivals(rate, run.warmup, run.duration, run.tail, seed)
    bodies = await asyncio.to_thread(lambda: [run.body(i) for i in range(len(at_list))])
    first = run.offset
    run.offset += len(at_list)
    load_start = await load_mark(run)
    end = run.warmup + run.duration
    recs: list[dict] = []
    events: list[tuple[float, int]] = []
    state = {"in_flight": 0, "measured_left": 0}
    scrapes: dict[str, asyncio.Task] = {}  # the counters as the window opens and closes, read beside the arrivals
    tasks = []

    async def one(i: int, at: float, measured: bool) -> None:
        rec = await send(run.session, run.url, run.headers, bodies[i], run.timeout)
        rec.update(index=first + i, at=at, measured=measured, turn=0, t_end=time.monotonic())
        rec.pop("reply", None)
        state["in_flight"] -= 1
        state["measured_left"] -= measured
        events.append((time.monotonic(), -1))
        recs.append(rec)

    start = time.monotonic() + 0.1
    for i, at in enumerate(at_list):
        if "start" not in scrapes and at >= run.warmup:
            await asyncio.sleep(max(0.0, start + run.warmup - time.monotonic()))
            scrapes["start"] = asyncio.create_task(scrape(run))
        if "end" not in scrapes and at >= end:
            await asyncio.sleep(max(0.0, start + end - time.monotonic()))
            scrapes["end"] = asyncio.create_task(scrape(run))
        if at >= end and not state["measured_left"]:
            break
        await asyncio.sleep(max(0.0, start + at - time.monotonic()))
        measured = run.warmup <= at < end
        if state["in_flight"] >= slots:
            recs.append({"at": at, "measured": measured, "dropped": True})
            continue
        state["in_flight"] += 1
        state["measured_left"] += measured
        events.append((time.monotonic(), +1))
        tasks.append(asyncio.create_task(one(i, at, measured)))
    if "end" not in scrapes:
        await asyncio.sleep(max(0.0, start + end - time.monotonic()))
        scrapes["end"] = asyncio.create_task(scrape(run))
    await asyncio.gather(*tasks)
    marks = {k: await t for k, t in scrapes.items()}
    load_end = await load_mark(run, settled=True)
    out = summarize(recs, events, start + run.warmup, start + end, run.duration)
    return {"load": f"rate {rate:g}/s, {slots} slots", "rate": rate, "slots": slots, **out,
            "engine": engine(marks, recs), "spec": spec_block(load_start[1], load_end[1], recs),
            "answers": answers(recs)}


async def closed_loop(run: Run, conc: int, requests: int | None = None) -> dict[str, Any]:
    """`conc` clients in a closed loop (module docstring). A workload that runs out of requests ends the load, and the
    window with it. With `requests`, the load is exactly that many requests, all measured, and the window runs from
    the first send to the last finish: no warm-up, duration or tail."""
    recs: list[dict] = []
    events: list[tuple[float, int]] = []
    state: dict[str, Any] = {"next": 0, "measured_left": 0, "exhausted_at": None}
    marks: dict[str, tuple[float, dict[str, float] | None]] = {}
    load_start = await load_mark(run)
    start = time.monotonic()
    w0, w1 = (start, math.inf) if requests is not None else (start + run.warmup, start + run.warmup + run.duration)
    limit = run.workload.count - run.offset if run.workload.count is not None else None

    async def mark() -> None:  # the counters as the window opens and as it closes
        await asyncio.sleep(max(0.0, w0 - time.monotonic()))
        marks["start"] = await scrape(run)
        await asyncio.sleep(max(0.0, w1 - time.monotonic()))
        marks["end"] = await scrape(run)

    async def client() -> None:
        i = turn = prev = reply = None  # the conversation's workload index, turn, last request and its reply
        while True:
            now = time.monotonic()
            if (now >= w1 and not state["measured_left"]) or now >= w1 + run.tail:
                return
            if i is None:
                if requests is not None and state["next"] >= requests:
                    return
                if limit is not None and state["next"] >= limit:
                    state["exhausted_at"] = state["exhausted_at"] or time.monotonic()
                    return
                i, turn = state["next"], 0
                state["next"] += 1
                body = await asyncio.to_thread(run.body, i)
            else:
                body = await asyncio.to_thread(session_body, run, prev, reply, i, turn)
            t = time.monotonic()
            measured = w0 <= t < w1
            state["measured_left"] += measured
            events.append((t, +1))
            rec = await send(run.session, run.url, run.headers, body, run.timeout)
            reply = rec.pop("reply", None)
            rec.update(index=run.offset + i, at=t - start, measured=measured, turn=turn, t_end=time.monotonic())
            state["measured_left"] -= measured
            events.append((time.monotonic(), -1))
            recs.append(rec)
            if reply is not None and turn + 1 < run.turns:
                if run.think:  # thinking: not in flight; never past the window and tail (inf with --requests)
                    pause = min(think_time(run, i, turn), w1 + run.tail - time.monotonic())
                    if pause <= 0:
                        return
                    await asyncio.sleep(pause)
                prev, turn = body, turn + 1
            else:  # the conversation is over, or failed: a new one
                i = prev = reply = None

    if requests is None:
        marker = asyncio.create_task(mark())
    else:
        marks["start"] = await scrape(run)
    await asyncio.gather(*(client() for _ in range(conc)))
    if requests is not None:
        marks["end"] = await scrape(run)
    elif time.monotonic() < w1:  # the workload ran out before the window closed: the counters as the load ends
        marker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await marker
        if "start" in marks:
            marks["end"] = await scrape(run)
    else:
        await marker
    run.offset += state["next"]
    load_end = await load_mark(run, settled=True)
    if requests is not None:  # the first send to the last finish
        w0 = min((t for t, d in events if d > 0), default=start)
        t1 = max((r["t_end"] for r in recs), default=w0)
    else:
        t1 = w1 if state["exhausted_at"] is None else min(w1, max(w0, state["exhausted_at"]))
    out = summarize(recs, events, w0, t1, t1 - w0)
    if state["exhausted_at"] is not None:
        out["exhausted"] = (f"the workload ran out of requests {state['exhausted_at'] - start:.1f} s into the load: "
                            f"a window of {t1 - w0:.1f} s")
    name = f"concurrency {conc}" if requests is None else f"concurrency {conc}, {requests} requests"
    return {"load": name, "concurrency": conc, **({} if requests is None else {"requests": requests}), **out,
            "engine": engine(marks, recs), "spec": spec_block(load_start[1], load_end[1], recs),
            "answers": answers(recs)}


def summarize(recs: list[dict], events: list[tuple[float, int]], t0: float, t1: float, seconds: float) -> dict:
    """The measured requests' counts, latencies and throughput over the window [t0, t1)."""
    meas = [r for r in recs if r["measured"]]
    sent = [r for r in meas if not r.get("dropped")]
    ok = [r for r in sent if r["status"] == 200 and not r.get("error")]
    errors: dict[str, int] = {}
    for r in sent:
        if r not in ok:
            errors[r.get("error") or str(r["status"])] = errors.get(r.get("error") or str(r["status"]), 0) + 1
    finish: dict[str, int] = {}
    for r in ok:
        finish[str(r.get("finish"))] = finish.get(str(r.get("finish")), 0) + 1
    tokens = [r["tokens"] for r in ok if r.get("tokens") is not None]
    per_s = 1 / seconds if seconds > 0 else 0.0
    tpot = [(r["e2e_ms"] - r["ttft_ms"]) / (r["tokens"] - 1) for r in ok
            if r.get("ttft_ms") is not None and (r.get("tokens") or 0) > 1]
    return {
        "measured": len(meas), "ok": len(ok), "dropped": len(meas) - len(sent), "failed": len(sent) - len(ok),
        "errors": errors, "finish": finish, "seconds": round(seconds, 3),
        "offered_rps": round(len(meas) * per_s, 3), "served_rps": round(len(ok) * per_s, 3),
        "output_tokens_per_s": round(sum(tokens) * per_s, 1) if tokens else None,
        "e2e_ms": pcts([r["e2e_ms"] for r in ok]), "ttft_ms": pcts([r["ttft_ms"] for r in ok if "ttft_ms" in r]),
        "tpot_ms": pcts(tpot, 2), "reply_tokens": pcts([float(t) for t in tokens], 0),
        "prompt_tokens": pcts([float(r["prompt_tokens"]) for r in ok if r.get("prompt_tokens") is not None], 0),
        "in_flight": in_flight(events, t0, t1) if events and t1 > t0 else None, **cache_stats(ok)}


def cache_stats(ok: list[dict]) -> dict[str, Any]:
    """The prefix cache over the successful requests that report both their prompt and their cached tokens:
    `cache_hit_rate` (Σ cached / Σ prompt), `uncached_tokens` (prompt - cached per request, percentiles) and
    `cache_reported` (how many requests report); None for the first two without any. `cache_by_turn`: per turn index
    of a session load (turn 0 included) the requests, hit rate and median uncached tokens; None for single requests."""
    rep = [r for r in ok if r.get("cached_tokens") is not None and r.get("prompt_tokens") is not None]

    def rate(rs: list[dict]) -> float | None:
        total = sum(r["prompt_tokens"] for r in rs)
        return round(sum(r["cached_tokens"] for r in rs) / total, 4) if total else None

    uncached = [float(r["prompt_tokens"] - r["cached_tokens"]) for r in rep]
    by_turn = None
    if any(r.get("turn") for r in rep):
        by_turn = [{"turn": t, "requests": len(rs), "hit_rate": rate(rs),
                    "uncached_p50": pcts([float(r["prompt_tokens"] - r["cached_tokens"]) for r in rs], 0)["p50"]}
                   for t in sorted({r.get("turn", 0) for r in rep})
                   for rs in [[r for r in rep if r.get("turn", 0) == t]]]
    return {"cache_reported": len(rep), "cache_hit_rate": rate(rep),
            "uncached_tokens": pcts(uncached, 0) if rep else None, "cache_by_turn": by_turn}


def answers(recs: list[dict]) -> dict[str, list]:
    """Every answer of a load, warm-up and tail included, by its workload index (a session's later turns "index.turn"):
    [digest, tokens, finish reason]."""
    return {f"{r['index']}.{r['turn']}" if r.get("turn") else str(r["index"]): [r["digest"], r.get("tokens"),
                                                                              r.get("finish")]
            for r in sorted(recs, key=lambda r: r.get("index", -1)) if r.get("digest") is not None}


# -- the rate search --------------------------------------------------------------------------------------------------

def passes(load: dict[str, Any], metric: str, value: float) -> tuple[bool, str]:
    """Whether a probe's load is within the bound, and why not: nothing failed or dropped, the served rate at least
    `REQUIRED_SERVED` of the offered, and `metric` ("e2e_ms.p99") at most `value`."""
    if load["failed"]:
        return False, f"{load['failed']} failed"
    if load["dropped"]:
        return False, f"{load['dropped']} dropped"
    if load["served_rps"] < REQUIRED_SERVED * load["offered_rps"]:
        return False, f"served {load['served_rps']:g}/s of {load['offered_rps']:g}/s offered"
    key, stat = metric.split(".")
    got = (load.get(key) or {}).get(stat)
    if not load["ok"] or got is None:
        return False, "no successful request"
    if got > value:
        return False, f"{metric} {got:.0f} > {value:g}"
    return True, "ok"


class _ProbeLimit(Exception):
    pass


def find_rate(probe: Callable[[float], tuple[bool, dict]], lo: float, hi: float, resolution: float,
              confirm: int) -> dict[str, Any]:
    """The highest rate in [lo, hi] whose probe passes (module docstring). `probe(rate)` gives (passed, its load).
    Returns `best_rate` (None when none passes), `first_failing_rate`, `confirmed`, `stopped` and `probes`: one
    {rate, passed, confirmation (None, or the repeat's number), load} per probe, in the order run."""
    lo, hi = round(lo, 4), round(hi, 4)
    probes: list[dict[str, Any]] = []

    def run(rate: float, confirmation: int | None = None) -> bool:
        if len(probes) >= MAX_PROBES:
            raise _ProbeLimit
        passed, load = probe(rate)
        probes.append({"rate": rate, "passed": passed, "confirmation": confirmation, "load": load})
        return passed

    best: float | None = None  # the highest rate that passed
    fail: float | None = None  # the lowest rate above it that failed
    confirmed, stopped = False, None
    try:
        if not run(lo):
            fail = lo
        else:
            best = lo
            while fail is None and best < hi:
                rate = min(round(2 * best, 4), hi)
                if run(rate):
                    best = rate
                else:
                    fail = rate
            while True:
                while fail is not None and fail - best > resolution:
                    mid = round((best + fail) / 2, 4)
                    if mid <= best or mid >= fail:
                        break
                    if run(mid):
                        best = mid
                    else:
                        fail = mid
                if not confirm:
                    break
                if all(run(best, n) for n in range(1, confirm + 1)):
                    confirmed = True
                    break
                fail = best  # the best rate did not hold: look again below it
                best = max((q["rate"] for q in probes
                            if q["passed"] and q["confirmation"] is None and q["rate"] < fail), default=None)
                if best is None:
                    break
    except _ProbeLimit:
        stopped = "probe limit"
    return {"best_rate": best, "first_failing_rate": fail, "confirmed": confirmed, "stopped": stopped,
            "probes": probes}


def probe_plan(seed: int, rate: float) -> tuple[int, int]:
    """A probe's arrival seed and the first workload index of its rate: they depend on the rate only, so every engine
    gets the same arrivals and requests. A repeat of the rate keeps the seed and starts REPEAT_STRIDE higher."""
    key = round(rate * 1000)
    return seed + key, key * PROBE_STRIDE


def read_bound(path: Path, model: str, workload: str) -> dict[str, Any]:
    """The bound file's contents, checked against the run's model and workload."""
    try:
        bound = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise SystemExit(f"--bound-file {path}: {exc}") from exc
    if not isinstance(bound, dict):
        raise SystemExit(f"--bound-file {path}: a JSON object")
    if bound.get("model") != model:
        raise SystemExit(f"--bound-file {path}: made for model {bound.get('model')!r}, this run is {model!r}")
    if bound.get("workload") != workload:
        raise SystemExit(f"--bound-file {path}: made for workload {bound.get('workload')!r}, this run is {workload!r}")
    key, _, stat = str(bound.get("metric")).partition(".")
    if key not in METRIC_KEYS or stat not in METRIC_STATS:
        raise SystemExit(f"--bound-file {path}: metric {bound.get('metric')!r} is not one of "
                         f"{{{'|'.join(METRIC_KEYS)}}}.{{{'|'.join(METRIC_STATS)}}}")
    if isinstance(bound.get("value"), bool) or not isinstance(bound.get("value"), (int, float)):
        raise SystemExit(f"--bound-file {path}: value is not a number")
    return bound


async def search(run: Run, args: argparse.Namespace, bound: dict[str, Any]) -> tuple[list[dict], dict[str, Any]]:
    """Run the rate search: (the probes' load entries in order run, the report's `search` block)."""
    lo, hi = args.find_rate
    resolution = args.resolution if args.resolution is not None else (hi - lo) / 8
    slots = args.slots[0] if args.slots else 128
    loop = asyncio.get_running_loop()
    window = args.warmup + args.duration + args.tail
    if math.ceil(window * hi) + 1 > REPEAT_STRIDE:
        raise SystemExit(f"--find-rate {lo:g},{hi:g}: a probe at {hi:g}/s over {window:g} s could send more than "
                         f"{REPEAT_STRIDE} requests, the indexes one probe may use: lower HI or shorten the window")
    repeats: dict[int, int] = {}  # the probes run so far, by rate

    async def one(rate: float) -> tuple[bool, dict]:
        seed, first = probe_plan(args.seed, rate)
        n = repeats.get(round(rate * 1000), 0)
        repeats[round(rate * 1000)] = n + 1
        run.offset = first + n * REPEAT_STRIDE
        print(f"[{time.strftime('%H:%M:%S')}] probe at {rate:g}/s: {args.warmup:g} s warm-up, "
              f"{args.duration:g} s measured", file=sys.stderr, flush=True)
        load = await open_loop(run, rate, slots, seed)
        passed, reason = passes(load, bound["metric"], bound["value"])
        load["probe"] = {"rate": rate, "passed": passed, "reason": reason}
        print(f"  {'pass' if passed else 'fail'}: {reason}", file=sys.stderr, flush=True)
        return passed, load

    found = await asyncio.to_thread(
        find_rate, lambda rate: asyncio.run_coroutine_threadsafe(one(rate), loop).result(), lo, hi, resolution,
        args.confirm)
    loads = []
    for q in found["probes"]:
        if q["confirmation"]:
            q["load"]["load"] += f" (confirmation {q['confirmation']})"
        loads.append(q["load"])
    block = {"bound": bound, "low": lo, "high": hi, "resolution": resolution, "confirm": args.confirm,
             "best_rate": found["best_rate"], "first_failing_rate": found["first_failing_rate"],
             "confirmed": found["confirmed"],
             "probes": [q["load"]["probe"] for q in found["probes"]], "stopped": found["stopped"]}
    return loads, block


def search_line(block: dict[str, Any]) -> str:
    """The table's closing line about a rate search."""
    best = block["best_rate"]
    if best is None:
        line = f"none at or above {block['low']:g}/s"
    else:
        line = f"{best:g}/s ({'confirmed' if block['confirmed'] else 'not confirmed'})"
    return f"best rate within the bound: {line}" + (f"; stopped: {block['stopped']}" if block["stopped"] else "")


# -- the run -----------------------------------------------------------------------------------------------------------

def table(loads: list[dict]) -> str:
    def p(d: dict, k: str) -> str:
        return "-" if not d or d.get("n", 0) == 0 else f"{d[k]:g}"

    head = ("load", "measured", "ok", "dropped", "failed", "req/s", "out tok/s", "e2e p50", "e2e p99", "ttft p50",
            "ttft p99", "tpot p50", "reply", "hit rate", "uncached p50", "uncached p90")
    rows = [head]
    for r in loads:
        rate = r.get("cache_hit_rate")
        hit = "-" if rate is None else f"{rate:.1%}"
        tps = "-" if r["output_tokens_per_s"] is None else f"{r['output_tokens_per_s']:g}"
        rows.append((r["load"], str(r["measured"]), str(r["ok"]), str(r["dropped"]), str(r["failed"]),
                     f"{r['served_rps']:g}", tps,
                     p(r["e2e_ms"], "p50"), p(r["e2e_ms"], "p99"), p(r["ttft_ms"], "p50"), p(r["ttft_ms"], "p99"),
                     p(r["tpot_ms"], "p50"), p(r["reply_tokens"], "mean"), hit, p(r.get("uncached_tokens"), "p50"),
                     p(r.get("uncached_tokens"), "p90")))
    widths = [max(len(row[i]) for row in rows) for i in range(len(head))]
    lines = ["  ".join(c.rjust(w) if i else c.ljust(w) for i, (c, w) in enumerate(zip(row, widths)))
             for row in rows]
    return "\n".join([lines[0], "  ".join("-" * w for w in widths), *lines[1:], "(latencies in ms; reply: mean "
                      "output tokens; hit rate: cached over prompt tokens, - when the server does not report them)"])


def numbers(text: str, kind: type) -> list:
    try:
        return [kind(x) for x in text.split(",") if x.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{text!r}: comma-separated numbers") from exc


def warm_up_args(args: argparse.Namespace) -> argparse.Namespace:
    """The warm-up's arguments: `--seed + 1`, and the measured run's shared prefix (`--shared-prefix-tokens`), which
    the warm-up requests then put in the server's prefix cache."""
    prefix = args.seed if getattr(args, "prefix_seed", None) is None else args.prefix_seed
    return argparse.Namespace(**{**vars(args), "seed": args.seed + 1, "prefix_seed": prefix})


async def warm_up(run: Run, args: argparse.Namespace, count: int) -> None:
    """`count` unrecorded requests at concurrency min(count, 4), from the workload built with `warm_up_args`, at the
    indexes from `WARMUP_OFFSET`: `run.offset` is not touched. A failed request ends the run."""
    warm = dataclasses.replace(run, workload=make_workload(warm_up_args(args)),
                               offset=WARMUP_OFFSET)
    taken = 0

    async def client() -> None:
        nonlocal taken
        while taken < count:
            j, taken = taken, taken + 1
            body = await asyncio.to_thread(warm.body, j)
            rec = await send(run.session, run.url, run.headers, body, run.timeout)
            if rec.get("status") != 200 or rec.get("error"):
                raise SystemExit(f"warm-up request {j} failed: {rec.get('error') or rec.get('status')}")

    print(f"[{time.strftime('%H:%M:%S')}] {count} warm-up requests", file=sys.stderr, flush=True)
    await asyncio.gather(*(client() for _ in range(min(count, 4))))


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    import aiohttp

    workload = make_workload(args)
    if args.warmup_requests and workload.count is not None:
        raise SystemExit(f"--warmup-requests {args.warmup_requests}: the folder workload's images are finite, "
                         "so a warm-up cannot take its own (use --warmup-requests 0)")
    if args.first_index and workload.count is not None:
        raise SystemExit(f"--first-index {args.first_index}: the folder workload's images are finite, so it starts "
                         "at 0")
    bound = None
    if args.find_rate:
        if workload.count is not None:
            raise SystemExit("--find-rate needs a workload without a finite count (scene, chat or catalog)")
        bound = read_bound(args.bound_file, args.model, workload.name)
    if args.rates:
        slots = args.slots or [128]
        if len(slots) not in (1, len(args.rates)):
            raise SystemExit("--slots: one value, or one per rate")
        slots = slots * len(args.rates) if len(slots) == 1 else slots
        need = sum(len(arrivals(r, args.warmup, args.duration, args.tail, args.seed + n))
                   for n, r in enumerate(args.rates))
        if workload.count is not None and need > workload.count:
            raise SystemExit(f"--workload {args.workload}: {workload.count} images, and this run needs up to {need} "
                             "(each is sent once): add images, or shorten --warmup, --duration or --tail")
    template: dict[str, Any] = {"model": args.model, "temperature": args.temperature, **args.extra_body,
                                "stream": not args.no_stream}
    if not args.no_stream:
        template["stream_options"] = {"include_usage": True}
    if args.ignore_eos:
        template["ignore_eos"] = True
    key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    headers = {"authorization": f"Bearer {key}"} if key else {}
    url = args.base_url.rstrip("/") + "/v1/chat/completions"
    report: dict[str, Any] = {
        "when": time.strftime("%Y-%m-%d %H:%M:%S"), "client": {"host": platform.platform(),
                                                              "python": platform.python_version()},
        "model": args.model, "workload": {"name": workload.name, **workload.summary}, "stream": not args.no_stream,
        "temperature": args.temperature, "ignore_eos": args.ignore_eos, "extra_body": args.extra_body,
        "request_seeds": args.request_seeds, "metrics": not args.no_metrics, "warmup_s": args.warmup,
        "duration_s": args.duration, "seed": args.seed, "requests": args.requests,
        "warmup_requests": args.warmup_requests, "first_index": args.first_index, "meta": args.meta, "loads": []}
    gpus = gpu_info().get("gpus")
    if gpus:
        report["gpu"] = {k: gpus[0].get(k) for k in GPU_KEYS}
    if args.turns > 1:
        report["workload"].update(turns=args.turns, tool_tokens=args.tool_tokens, think_s=args.think,
                                  think_sigma=args.think_sigma if args.think else None)
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        run = Run(session, url, headers, workload, template, args.timeout, args.warmup, args.duration, args.tail,
                  offset=args.first_index, seeds=args.request_seeds, turns=args.turns,
                  think=args.think, think_sigma=args.think_sigma, seed=args.seed,
                  metrics_url=None if args.no_metrics else args.base_url.rstrip("/") + "/metrics")
        if args.warmup_requests:
            await warm_up(run, args, args.warmup_requests)
        if bound is not None:
            report["loads"], report["search"] = await search(run, args, bound)
        for n, value in enumerate([] if bound is not None else args.rates or args.concurrency):
            what = (f"{args.requests} requests" if args.requests else
                    f"{args.warmup:g} s warm-up, {args.duration:g} s measured")
            print(f"[{time.strftime('%H:%M:%S')}] {'rate' if args.rates else 'concurrency'} {value:g}: {what}",
                  file=sys.stderr, flush=True)
            r = await (open_loop(run, value, slots[n], args.seed + n) if args.rates
                       else closed_loop(run, int(value), args.requests))
            report["loads"].append(r)
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000", help="the server (default: %(default)s)")
    ap.add_argument("--model", required=True, help="the served model's name, as the server lists it in /v1/models")
    ap.add_argument("--workload", default="scene", help="scene, chat, catalog, or a folder in the host check's layout")
    ap.add_argument("--tokenizer", help="a model folder or Hugging Face repo id whose tokenizer sizes the scene's "
                                        "and chat's prompts (default: --model)")
    load = ap.add_mutually_exclusive_group(required=True)
    load.add_argument("--rates", type=lambda s: numbers(s, float), help="Poisson arrival rates, requests/s: 2,4")
    load.add_argument("--concurrency", type=lambda s: numbers(s, int), help="closed-loop clients: 1,8,32")
    load.add_argument("--find-rate", type=lambda s: numbers(s, float), metavar="LO,HI",
                      help="search for the highest Poisson rate (requests/s) between LO and HI within the bound of "
                           "--bound-file, probing with --slots (one value)")
    ap.add_argument("--bound-file", type=Path,
                    help="with --find-rate: JSON {model, workload, metric: e2e_ms.p99, value: ms, ...}")
    ap.add_argument("--resolution", type=float,
                    help="with --find-rate: the search stops when the passing and failing rates are this close "
                         "(default (HI - LO) / 8)")
    ap.add_argument("--confirm", type=int, default=0,
                    help="with --find-rate: probe the best rate this many more times; a failure moves it down "
                         "(default: %(default)s)")
    ap.add_argument("--slots", type=lambda s: numbers(s, int),
                    help="with --rates: the most requests in flight, one value or one per rate (default 128)")
    ap.add_argument("--requests", type=int,
                    help="with --concurrency: each load is exactly this many requests, all measured, from its first "
                         "send to its last finish (--warmup, --duration and --tail do not apply)")
    ap.add_argument("--first-index", type=int, default=0,
                    help="the workload index of the first measured request (default 0): runs against one server "
                         "must not repeat each other's indexes, which its prefix cache would serve again; not with "
                         "--find-rate or a folder workload")
    ap.add_argument("--warmup-requests", type=int,
                    help="unrecorded requests before the first load, from another seed (default: "
                         f"{WARMUP_REQUESTS} with --requests, else 0)")
    ap.add_argument("--warmup", type=float, default=10.0, help="unmeasured seconds before each load's window")
    ap.add_argument("--duration", type=float, default=60.0, help="each load's measured seconds")
    ap.add_argument("--tail", type=float, default=60.0,
                    help="at most this long after the window, the load goes on while measured requests run")
    ap.add_argument("--max-tokens", type=int,
                    help=f"per request (default: {SCENE_MAX_TOKENS} for scene, {SCHEMA_MAX_TOKENS} for catalog and "
                         f"a folder, {CHAT_MAX_TOKENS} for chat)")
    ap.add_argument("--prompt-tokens", type=int, default=synthetic.PROMPT_TOKENS,
                    help="scene: the system prompt's tokens (default: %(default)s)")
    ap.add_argument("--tall-share", type=float, default=synthetic.TALL_SHARE,
                    help="scene: the share of tall images, 0 to 1 (default: %(default)s)")
    ap.add_argument("--input-tokens", type=int, default=512, help="chat: each prompt's tokens")
    ap.add_argument("--shared-prefix-tokens", type=int, default=0, metavar="N",
                    help="chat: every prompt starts with the same N tokens (with --turns, each conversation's first "
                         "message), so the prefix cache holds them (default: %(default)s, none)")
    ap.add_argument("--turns", type=int, default=1,
                    help="chat, with --concurrency: requests per conversation, each one the last one's messages, its "
                         "reply and a new user message (default: %(default)s, single requests)")
    ap.add_argument("--tool-tokens", type=int, default=128,
                    help="with --turns: the tokens of each new user message (default: %(default)s)")
    ap.add_argument("--think", type=float,
                    help="with --turns: a client's pause after a reply, before its next turn, in seconds (the median "
                         "of a log-normal; default: none)")
    ap.add_argument("--think-sigma", type=float, default=1.0,
                    help="with --think: the pause's log-normal shape (default: %(default)s)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--ignore-eos", action="store_true", help="every reply runs to --max-tokens")
    ap.add_argument("--no-stream", action="store_true", help="not streamed (no TTFT)")
    ap.add_argument("--request-seeds", action="store_true", help="each request's seed: its workload index")
    ap.add_argument("--no-metrics", action="store_true", help="do not read the server's /metrics around each window")
    ap.add_argument("--extra-body", type=json.loads, default=DEFAULT_EXTRA,
                    help=f"JSON merged into every request (default: {json.dumps(DEFAULT_EXTRA)}; a value replaces it)")
    ap.add_argument("--api-key", help="a bearer token (default: OPENAI_API_KEY, if set)")
    ap.add_argument("--timeout", type=float, default=300.0, help="per request, seconds")
    ap.add_argument("--seed", type=int, default=0, help="the arrivals' and chat prompts' seed")
    ap.add_argument("--meta", type=json.loads,
                    help="a JSON object stored in the report as `meta` (the engine, the arm, ...)")
    ap.add_argument("--out", type=Path, help="the JSON report (default: results/load/<time>.json)")
    args = ap.parse_args(argv)
    if not 0 <= args.tall_share <= 1:
        ap.error(f"--tall-share {args.tall_share}: a share from 0 to 1")
    if args.prompt_tokens < 1:
        ap.error(f"--prompt-tokens {args.prompt_tokens}: at least 1")
    if args.shared_prefix_tokens < 0:
        ap.error(f"--shared-prefix-tokens {args.shared_prefix_tokens}: at least 0")
    if args.shared_prefix_tokens and args.workload != "chat":
        ap.error("--shared-prefix-tokens needs --workload chat")
    if args.turns < 1:
        ap.error(f"--turns {args.turns}: at least 1")
    if args.turns > 1 and (args.rates or args.find_rate or args.workload != "chat"):
        ap.error("--turns needs --concurrency (a closed loop) and --workload chat")
    if args.turns > 1 and args.tool_tokens < 1:
        ap.error(f"--tool-tokens {args.tool_tokens}: at least 1")
    if args.think is not None and (args.think <= 0 or args.turns < 2):
        ap.error("--think needs seconds above 0 and --turns above 1")
    if args.think_sigma < 0:
        ap.error(f"--think-sigma {args.think_sigma}: at least 0")
    if args.meta is not None and not isinstance(args.meta, dict):
        ap.error("--meta: a JSON object")
    if args.requests is not None and args.requests < 1:
        ap.error(f"--requests {args.requests}: at least 1")
    if args.requests is not None and (args.rates or args.find_rate or args.turns > 1):
        ap.error("--requests needs --concurrency and a single request per client (not --rates or --turns)")
    if args.first_index < 0:
        ap.error(f"--first-index {args.first_index}: at least 0")
    if args.warmup_requests is None:
        args.warmup_requests = WARMUP_REQUESTS if args.requests or args.find_rate else 0
    if args.warmup_requests < 0:
        ap.error(f"--warmup-requests {args.warmup_requests}: at least 0")
    if args.find_rate:
        lo, hi = args.find_rate if len(args.find_rate) == 2 else (0.0, -1.0)
        if not 0 < lo <= hi:
            ap.error(f"--find-rate {args.find_rate}: two rates LO,HI with 0 < LO <= HI")
        if not args.bound_file:
            ap.error("--find-rate needs --bound-file")
        if args.slots and len(args.slots) != 1:
            ap.error("--find-rate takes one --slots value")
        if args.resolution is not None and args.resolution <= 0:
            ap.error(f"--resolution {args.resolution}: above 0")
        if not 0 <= args.confirm < PROBE_STRIDE // REPEAT_STRIDE:
            ap.error(f"--confirm {args.confirm}: from 0 to {PROBE_STRIDE // REPEAT_STRIDE - 1}")
        if args.first_index:
            ap.error("--first-index does not go with --find-rate (a probe's indexes come from its rate)")
    elif args.bound_file or args.resolution is not None or args.confirm:
        ap.error("--bound-file, --resolution and --confirm need --find-rate")
    report = asyncio.run(main_async(args))
    out = args.out or RESULTS / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(table(report["loads"]))
    if "search" in report:
        print(search_line(report["search"]))
    print(f"wrote {out}", file=sys.stderr)
    return 1 if any(r["failed"] for r in report["loads"]) else 0


if __name__ == "__main__":
    sys.exit(main())
