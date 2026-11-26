#!/usr/bin/env python3
"""A verify step of Gemma 4 31B FP8 with its assistant (`--k` drafts, 5 by default) under structured output, profiled
at a few decode rows on one GPU: where a decode step's time goes at low concurrency.

The real scheduler and engine loop drive a `GemmaRunner` (`make_runner`: CUDA graphs, the KV arena in `--memory-gb`,
which skips the memory profile and leaves the rest of the GPU's memory to Nsight Compute) with a grammar that allows
every token (`AllowAll`: all-ones masks, never stops). Every decode row is MASKED, so every step splits into its
forward graph (`decode_f`) and its sampling graph (`decode_s`), as every step under structured output does. The masks
cost nothing here; under a real grammar llguidance fills 1 + k of them per row between the two graphs.

**The grammar-aware heads** (`runner.cuda.heads`; architecture.md, "Grammar-aware heads"): `--sparse-head` and
`--draft-universe` (both on by default, as the config derives them for Gemma 4 under structured output) pass to the
runner. All-ones masks make every step's union the whole vocabulary, so under `AllowAll` both heads stand aside (the
dense head, and the assistant's whole head). `--grammar subset:<U>` gives every request a grammar whose every mask
allows the same U tokens (`SubsetGrammar`: evenly spaced over the vocabulary, no EOS id), so each step's union and the
draft universe are those U tokens: the sparse head scores those rows of the tied head, the assistant drafts over them.
Both heads have static shapes (the graphs'): every sparse step gathers and multiplies `dev.runner.cuda.sparse_head_rows`
rows, the union's ids padded, so U moves the cost little as long as it fits them (above, the step is dense: `over_cap`).
The sampling graph is then the heads' variant (`decode_q`, `decode_p`, `decode_u`); the spy records which head each
measured step took, and a window whose steps took more than one is an error.

**Requests:** one random "system prompt" of `--prefix` tokens (a long shared system prompt) and a random tail of
`--tail` tokens per request (256 by default: an image's soft tokens and the template around them, as text ids: a
decode step does not tell them from image tokens), greedy, `ignore_eos`,
`--max-tokens` each. The first request runs alone to its first token with a cache hint at the prefix, so every later
one hits there (where the shared system prompt ends) and the rows share the prefix's full pages and window. The rest
are then added, and once all decode, `--decode-before` more steps run.

**Per row count N** (`--rows`, descending; the extra requests are aborted to go down): `--warm-s` seconds of steps
(steady clocks under the power limit; a row that finishes is replaced at once). With `--fresh` (the default) the N
rows are then replaced by new requests on the cached prefix, which decode `--decode-before` steps: the measured rows
hold a short reply's length and share the prefix's window, not the warm-up's hundreds of tokens. `--no-fresh`
measures the warmed rows themselves (the warm-up then also replaces a row too near `--max-tokens` to last the
measurement). Then:
- `--mode torch`: K = `--steps` steps unprofiled: the device time per step (the runner's CUDA events,
  `StepResult.gpu_busy_ns`), the GPU idle per step (between a step's forward and its sampling, and before it:
  `EngineCore.gpu_idle_ns`), wall time and tokens per step, and the board's power, SM clock and clock-event reasons
  (NVML: mean, p10, p90; the power limit); also each step's split (`StepResult.gpu_sample_ns`: its forward and its
  sampling part) and the engine's host time a turn by phase (`engine.core.PHASES`: the scheduler, the launch...; not
  the poll and the send, which only the process loop runs), the numbers planning at the boundary rests on
  (`--boundary` runs each turn that way; `--no-always-split` captures the fused graphs too). Then K more steps under
  torch.profiler (CUPTI, which sees the kernels inside graph replays), between `torch.cuda.synchronize()` calls: per
  step the kernel time and count, the time and share of each category (`VERIFY_CATEGORIES`), the top 30 kernels, and
  every kernel's name, time and calls, so that the categories can be redone from the JSON (`other` lists the names no
  category matched); and from the window's trace (`timeline`) the same split per graph launch: the forward graph (the
  target's layers), the sampling graph (the tied head, the sampler, verification and the assistant's drafts), what
  runs outside them, and the gaps between the kernels inside each graph.
- `--mode ncu`: K steps inside an NVTX range `verify_b{N}` and a `cudaProfilerStart`/`Stop` window, for
  `ncu --profile-from-start off` (torch.profiler never runs in the same process).
- `--mode none`: the unprofiled steps alone (any rows, quickly).
A spy on `launch_forward` checks that every measured step had exactly N device rows, all MASKED and verifying k
drafts, and nothing else (no prefill row, admission, free, copy or image; page growth and window releases are part of
decode steps): a step that does not is an error. It records each plan's shape and the runner's mode (on CUDA: the
`decode` graphs of N's bucket, split), and from the runner's `head_counters` around each step's `launch_sample` the
head its sampling took (sparse, or dense and why), its union, and whether its drafts scored the universe.

    python3 -m tests.scripts.profile_verify --model <FP8 dir> --assistant <assistant dir> --out verify.json \
        [--rows 8,4] [--k 5] [--steps <K>] [--warm-s 10] [--mode torch|ncu] [--trace <dir>] \
        [--grammar allow-all|subset:<U>] [--no-sparse-head] [--no-draft-universe]

profile_tower.py imports the kernel categories, the CUPTI summary, the NVML sampler and the ncu window from here.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import threading
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from llm_mini_infer.engine.core import PHASES, EngineCore
from llm_mini_infer.engine.messages import Abort, AddRequest
from llm_mini_infer.grammar.engine import Grammars, words
from llm_mini_infer.sched.plan import MASKED, PRECISE, SAMPLE
from llm_mini_infer.sched.scheduler import Scheduler

GRAMMAR = "allow-all"  # every request's grammar text (AllowAll compiles any text)


# -- the permissive grammar ---------------------------------------------------------------------------------------


class AllowAllMatcher:
    """`grammar.engine.Matcher` that takes every token and never stops: it only counts what it consumed."""

    def __init__(self) -> None:
        self.n = 0

    def consume_tokens(self, tokens: list[int]) -> bool:
        self.n += len(tokens)
        return True

    def try_consume_tokens(self, tokens: list[int]) -> int:
        self.n += len(tokens)
        return len(tokens)

    def validate_tokens(self, tokens: list[int]) -> int:
        return len(tokens)

    def rollback(self, num_tokens: int) -> None:
        if not 0 <= num_tokens <= self.n:
            raise ValueError(f"rollback of {num_tokens} tokens after {self.n}")
        self.n -= num_tokens

    def is_stopped(self) -> bool:
        return False

    def is_error(self) -> bool:
        return False

    def get_error(self) -> str:
        return ""

    def compute_ff_tokens(self) -> list[int]:
        return []  # forces nothing: the forced prefix takes no token

    def deep_copy(self) -> AllowAllMatcher:
        m = AllowAllMatcher()
        m.n = self.n
        return m


class AllowAll:
    """`grammar.engine.Backend` whose grammars allow every token: masks of all ones (llguidance's draft rule: a row's
    mask, then one after each draft, all of which it takes). No EOS ids: a grammar never terminates, so its rows stay
    MASKED whatever they sample."""

    name = "allow-all"

    def __init__(self, vocab_size: int, eos: tuple[int, ...] = ()):
        self.vocab_size, self.eos = vocab_size, tuple(eos)

    def compile(self, text: str) -> str:
        return text

    def instance(self, compiled: str) -> AllowAllMatcher:
        return AllowAllMatcher()

    def fill(self, singles: list[tuple[Any, int]], drafts: list[tuple[Any, int, list[int]]], buf: np.ndarray) -> None:
        for _, i in singles:
            buf[i] = -1
        for _, i, d in drafts:
            if not d:
                raise ValueError("Draft tokens must not be empty")  # as llguidance's
            buf[i:i + 1 + len(d)] = -1

    @staticmethod
    def only(m: Any, token: int) -> bool:
        return False  # no mask allows one token alone


class SubsetMatcher(AllowAllMatcher):
    """`AllowAllMatcher` whose consumed tokens must lie in a set (`allowed`, bool over the vocabulary): one outside it
    puts it in the error state, as llguidance's on a token it rejects, so a sample its mask did not allow fails its
    request (a sparse head that sampled outside the union stops the measurement). Drafts it takes all, as AllowAll's
    (`validate_tokens`): the scheduler then leaves none of a row's masks to its EOS-only rule, and every mask of every
    step is the set. Verification never reads a mask after a draft outside the set: no masked sample equals that
    draft."""

    def __init__(self, allowed: np.ndarray) -> None:
        super().__init__()
        self.allowed, self.err = allowed, ""

    def _takes(self, t: int) -> bool:
        return 0 <= t < len(self.allowed) and bool(self.allowed[t])

    def consume_tokens(self, tokens: list[int]) -> bool:
        if not self.err:
            bad = next((t for t in tokens if not self._takes(t)), None)
            if bad is None:
                self.n += len(tokens)
                return True
            self.err = f"token {bad} is not one of the subset's (after {self.n} tokens)"
        return False

    def try_consume_tokens(self, tokens: list[int]) -> int:
        n = 0
        while not self.err and n < len(tokens) and self._takes(tokens[n]):
            n += 1
        self.n += n
        return n

    def is_error(self) -> bool:
        return bool(self.err)

    def get_error(self) -> str:
        return self.err

    def deep_copy(self) -> SubsetMatcher:
        m = SubsetMatcher(self.allowed)
        m.n, m.err = self.n, self.err
        return m


class SubsetGrammar(AllowAll):
    """`grammar.engine.Backend` whose grammars allow the same `size` tokens at every position and never stop
    (`--grammar subset:<U>`): the vocabulary without `exclude` (the model's EOS ids: never sampled, as a JSON
    document's masks never allow one before its end) in `size` equal runs, the middle id of each. Every mask is those
    ids' words (along the drafts too, as AllowAll's), so every step's union is exactly them, whatever the rows and the
    drafts: the grammar-aware heads' work at a union of `size`. No EOS ids, as AllowAll's."""

    def __init__(self, vocab_size: int, size: int, exclude: tuple[int, ...] = ()):
        super().__init__(vocab_size)
        keep = np.setdiff1d(np.arange(vocab_size), np.asarray(exclude, np.int64))
        if not 0 < size <= len(keep):
            raise ValueError(f"subset:{size}: from 1 to {len(keep)} tokens (a vocabulary of {vocab_size} without "
                             f"its EOS ids {sorted(exclude)})")
        self.name = f"subset:{size}"
        self.ids = keep[(2 * np.arange(size) + 1) * len(keep) // (2 * size)]
        self.allowed = np.zeros(vocab_size, bool)
        self.allowed[self.ids] = True
        bits = np.zeros(words(vocab_size) * 32, bool)
        bits[self.ids] = True
        self.mask = np.packbits(bits, bitorder="little").view(np.int32)  # bit t % 32 of word t // 32

    def instance(self, compiled: str) -> SubsetMatcher:
        return SubsetMatcher(self.allowed)

    def fill(self, singles: list[tuple[Any, int]], drafts: list[tuple[Any, int, list[int]]], buf: np.ndarray) -> None:
        for _, i in singles:
            buf[i] = self.mask
        for _, i, d in drafts:
            if not d:
                raise ValueError("Draft tokens must not be empty")  # as llguidance's
            buf[i:i + 1 + len(d)] = self.mask


def grammar_spec(text: str) -> str:
    """`--grammar`: allow-all, or subset:<U> with U > 0 (checked against the vocabulary once the model is loaded)."""
    kind, _, size = text.partition(":")
    if text == "allow-all" or (kind == "subset" and size.isdigit() and int(size) > 0):
        return text
    raise argparse.ArgumentTypeError(f"{text!r}: allow-all or subset:<tokens>")


def grammar_backend(spec: str, vocab_size: int, eos: tuple[int, ...] = ()) -> AllowAll:
    """The backend `--grammar` names (`grammar_spec`): AllowAll, or a SubsetGrammar without the model's EOS ids."""
    if spec == "allow-all":
        return AllowAll(vocab_size)
    return SubsetGrammar(vocab_size, int(grammar_spec(spec).partition(":")[2]), exclude=eos)


def row_counts(spec: str, max_num_seqs: int) -> list[int]:
    """`--rows`: decode row counts as a comma list in any order (e.g. 8,7,6,5,4,3,2,1), each once, measured descending;
    each from 1 to `max_num_seqs`."""
    try:
        rows = sorted({int(x) for x in spec.split(",")}, reverse=True)
    except ValueError:
        raise ValueError(f"--rows {spec!r}: a comma list of row counts") from None
    if not 0 < rows[-1] <= rows[0] <= max_num_seqs:
        raise ValueError(f"--rows must lie in 1..--max-num-seqs ({max_num_seqs})")
    return rows


# -- kernel categories and the CUPTI summary ----------------------------------------------------------------------

MEMCPY = re.compile(r"^Mem(cpy|set)\b")  # torch.profiler's names of copies and sets ("Memcpy HtoD (Pinned -> Device)")
OURS = (r"\bg4_\w+_kernel|gelu_tanh_mul\w*_kernel|quant_fp8_kernel|rmsnorm\w*_kernel|qk_norm_rope_kernel|"
        r"vision_qkv_rope_kernel|silu_mul\w*_kernel|sigmoid_gate\w*_kernel")  # kernels/csrc: norms, GELU, quantizers
GEMM = r"nvjet|gemm|gemv|cublas|cutlass|xmma|splitKreduce|dot_kernel|reduce_1Block_kernel"  # cuBLAS(Lt)'s, any dtype

# A verify step's kernels, first match wins: our kernels before the FP8 GEMMs (their signatures may name e4m3), the
# FP8 GEMMs before cuBLAS (the vendored CUTLASS blockwise kernel, `lmk_fp8::cutlass_3x_gemm_fp8_blockwise`;
# torch._scaled_mm's when `fp8_gemm` picks it), attn512 (the full layers and the
# assistant's full layer), FlashInfer and our window kernel (the sliding layers; the assistant's sliding layers are
# FlashInfer's either way), our sampler, then the
# BF16 GEMMs: the tied head, the assistant's layers and head (cuBLAS: nvjet, its CUTLASS kernels, gemv, split-K; our
# small-M kernel at 1-2 rows), and torch.index_select's kernels (indexSelectSmallIndex up to 16 ids,
# indexSelectLargeIndex above): the grammar-aware heads' gathers, the sparse head's rows of the tied head (its cap,
# 1,024 rows of 5,376 BF16: 11 MB a step) and the universe head's rebuild (when the universe grows), next to the step's
# small selects of hidden rows, positions and slot tables (0.07-0.08 ms a step on the 31B at k = 3), so a head's gather
# is the heads-on run's excess over the heads-off run's. Everything else (torch's elementwise, other index and copy
# kernels: the softcap, the fp32 logits, verification, the drafter's glue) is "other".
VERIFY_CATEGORIES: list[tuple[str, re.Pattern[str]]] = [
    ("memcpy", MEMCPY),
    ("norm_act_quant", re.compile(OURS)),
    ("fp8_gemm", re.compile(r"lmk_fp8|fp8_blockwise|gemm_fp8|fp8_gemm|[Bb]lockwise|f8f6f4|e4m3\w*gemm|"
                            r"gemm\w*e4m3|scaled_mm|deep_gemm")),
    ("full_attention", re.compile(r"attn512")),
    ("sliding_attention", re.compile(r"flashinfer|BatchPrefill|BatchDecode|MergeStates|attn_window")),
    ("sampler", re.compile(r"\bsample_kernel|[Ss]ampl(?:er|ing)\w*[Kk]ernel")),
    ("bf16_gemm", re.compile(GEMM + r"|linear_small")),
    ("gather", re.compile(r"\bindexSelect(?:Small|Large)Index\b")),
]
# The vision tower's: cuBLAS's BF16 GEMMs, FlashInfer's ragged attention (with the copy of its output, a torch kernel,
# in the glue), our fused norms, GELU and attention input (`g4_vision_qkv`), and torch's glue (the pooling; where our
# attention-input kernel does not apply, the q/k/v norms, the 2-D RoPE's slices and the pad to 128: elementwise,
# reduce, cat, pad and index kernels).
TOWER_CATEGORIES: list[tuple[str, re.Pattern[str]]] = [
    ("memcpy", MEMCPY),
    ("fused_ops", re.compile(OURS)),
    ("flashinfer_attention", re.compile(r"flashinfer|BatchPrefill|Ragged|MergeStates")),
    ("cublas_gemm", re.compile(GEMM)),
    ("torch_glue", re.compile(r"at::native|at::cuda|\bcub::|elementwise|reduce_kernel|CatArray|[Pp]ad\w*_kernel|"
                              r"[Ii]ndex\w*[Kk]ernel")),
]


def category(name: str, table: list[tuple[str, re.Pattern[str]]]) -> str:
    for cat, rx in table:
        if rx.search(name):
            return cat
    return "other"


def summarize_kernels(rows: list[tuple[str, float, int]], steps: int, table: list[tuple[str, re.Pattern[str]]],
                      top: int = 30) -> dict[str, Any]:
    """CUPTI's totals over `steps` steps (or calls), as (name, device µs, count) rows, per step: the device time and
    the kernel time (without copies and sets), the kernels launched, each category's time, share of the device time and
    calls, the `top` kernels, the names no category matched (`other`), and every kernel (`all`, by time)."""
    cats: dict[str, list[float]] = {c: [0.0, 0.0] for c, _ in table}
    cats["other"] = [0.0, 0.0]
    kernels = []
    for name, us, count in rows:
        c = category(name, table)
        cats[c][0] += us
        cats[c][1] += count
        kernels.append({"ms": round(us / steps / 1e3, 4), "calls": round(count / steps, 2),
                        "us_per_call": round(us / max(count, 1), 2), "cat": c, "kernel": name})
    kernels.sort(key=lambda r: -r["ms"])
    total_us = sum(v[0] for v in cats.values())
    mem_us, mem_n = cats.get("memcpy", [0.0, 0.0])
    return {
        "steps": steps,
        "device_ms": round(total_us / steps / 1e3, 4),
        "kernel_ms": round((total_us - mem_us) / steps / 1e3, 4),
        "kernels": round((sum(v[1] for v in cats.values()) - mem_n) / steps, 2),
        "categories": {c: {"ms": round(us / steps / 1e3, 4), "share": round(us / total_us, 4) if total_us else 0.0,
                           "calls": round(n / steps, 2)} for c, (us, n) in cats.items()},
        "top": kernels[:top],
        "other": [{k: v for k, v in r.items() if k != "cat"} for r in kernels if r["cat"] == "other"],
        "all": kernels,
    }


def activities() -> list[Any]:
    """torch.profiler's activities: CUDA's alone (CUPTI: kernels, copies, sets); the CPU's without a GPU, where a
    window records no kernel (the harness's tests on CPU)."""
    import torch

    a = torch.profiler.ProfilerActivity
    return [a.CUDA] if torch.cuda.is_available() else [a.CPU]


def init_cupti() -> None:
    """CUPTI's one-time start-up (up to seconds, the first time a profiler starts), out of the way: a profile of one
    kernel, so that each measured window's profiler starts promptly after its synchronization (an idle GPU's clocks rise
    under a power limit)."""
    import torch

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]):
        torch.ones(1, device="cuda").add_(1)
        torch.cuda.synchronize()


def profiler_rows(prof: Any) -> list[tuple[str, float, int]]:
    """(name, device µs, count) of each kernel, copy and set a torch.profiler run saw (its `key_averages`, by self
    device time: with CUDA activity alone every row is one of those)."""
    out = []
    for e in prof.key_averages():
        us = getattr(e, "self_device_time_total", None)
        if us is None:
            us = getattr(e, "self_cuda_time_total", 0.0)
        if us and us > 0:
            out.append((e.key, float(us), int(e.count)))
    return out


# -- the trace's launches ------------------------------------------------------------------------------------------

DEVICE_EVENTS = ("kernel", "gpu_memcpy", "gpu_memset")  # the device events of torch.profiler's Chrome trace


def device_events(trace: dict[str, Any]) -> list[dict[str, Any]]:
    """The kernels, copies and sets of a Chrome trace torch.profiler exported (complete events: `ts`, `dur` in µs)."""
    return [e for e in trace["traceEvents"] if e.get("ph") == "X" and e.get("cat") in DEVICE_EVENTS and "dur" in e]


def busy_us(events: list[dict[str, Any]]) -> float:
    """The union of the events' intervals (µs): the time the device was running one of them."""
    total, end = 0.0, float("-inf")
    for t, d in sorted((e["ts"], e["dur"]) for e in events):
        if t + d > end:
            total += t + d - max(t, end)
            end = t + d
    return total


def span_us(events: list[dict[str, Any]]) -> float:
    return max(e["ts"] + e["dur"] for e in events) - min(e["ts"] for e in events) if events else 0.0


def kernel_rows(events: list[dict[str, Any]]) -> list[tuple[str, float, int]]:
    """(name, µs, count) of the events, by name (as `profiler_rows`)."""
    agg: dict[str, list[float]] = {}
    for e in events:
        a = agg.setdefault(e["name"], [0.0, 0])
        a[0] += e["dur"]
        a[1] += 1
    return [(name, us, int(n)) for name, (us, n) in agg.items()]


def timeline(trace: dict[str, Any], calls: int, table: list[tuple[str, re.Pattern[str]]],
             parts: tuple[str, ...] = ()) -> dict[str, Any]:
    """Per step (or call), the device events of a window's trace by launch: CUPTI gives every kernel of one graph
    launch that launch's correlation id, so the events of an id with two kernels or more are a graph launch, and the
    rest were launched one by one (a pageable copy's chunks share an id too). Each graph launch's span against its
    busy time gives the gaps between its kernels (launch latency inside the graph, what programmatic dependent launch
    would hide); the device's busy time against the window's span gives its idle time. With `parts`, the graph
    launches go in turn to the named parts (a split verify step: its forward graph, then its sampling graph), each
    with its kernels by category."""
    events = device_events(trace)
    by: dict[Any, list[dict[str, Any]]] = {}
    for e in events:
        by.setdefault(e.get("args", {}).get("correlation"), []).append(e)

    def is_graph(c: Any, g: list[dict[str, Any]]) -> bool:
        return c is not None and sum(e["cat"] == "kernel" for e in g) > 1

    graphs = sorted((g for c, g in by.items() if is_graph(c, g)), key=lambda g: min(e["ts"] for e in g))
    single = [e for c, g in by.items() if not is_graph(c, g) for e in g]

    def ms(us: float) -> float:
        return round(us / calls / 1e3, 4)

    g_busy, g_span = sum(busy_us(g) for g in graphs), sum(span_us(g) for g in graphs)
    out: dict[str, Any] = {
        "graph_launches": round(len(graphs) / calls, 2), "graph_events": round(sum(map(len, graphs)) / calls, 2),
        "graph_busy_ms": ms(g_busy), "graph_span_ms": ms(g_span), "graph_gaps_ms": ms(g_span - g_busy),
        "single_events": round(len(single) / calls, 2), "single_busy_ms": ms(busy_us(single)),
        "device_busy_ms": ms(busy_us(events)), "window_span_ms": ms(span_us(events))}
    if parts:
        if len(graphs) != len(parts) * calls:
            out["parts"] = f"{len(graphs)} graph launches, not {len(parts)} per step: no split"
            return out
        for i, name in enumerate(parts):
            gs = graphs[i::len(parts)]
            busy, span = sum(busy_us(g) for g in gs), sum(span_us(g) for g in gs)
            sk = summarize_kernels(kernel_rows([e for g in gs for e in g]), calls, table, top=10)
            out[name] = {"span_ms": ms(span), "busy_ms": ms(busy), "gaps_ms": ms(span - busy),
                         **{k: v for k, v in sk.items() if k not in ("steps", "all", "other")}}
    return out


def stats(values: list[float] | np.ndarray, nd: int = 3) -> dict[str, float]:
    v = np.asarray(values, np.float64)
    if not len(v):
        return {"n": 0}
    p10, p50, p90 = np.percentile(v, [10, 50, 90])
    return {"n": len(v), "mean": round(float(v.mean()), nd), "p10": round(float(p10), nd),
            "p50": round(float(p50), nd), "p90": round(float(p90), nd), "min": round(float(v.min()), nd),
            "max": round(float(v.max()), nd)}


# -- NVML -----------------------------------------------------------------------------------------------------------

# NVML's clock-event reasons: nvml.h's nvmlClocksEventReason* bits (tests check them against pynvml's where installed)
REASONS = {0x1: "idle", 0x2: "app_clocks", 0x4: "sw_power_cap", 0x8: "hw_slowdown", 0x10: "sync_boost",
           0x20: "sw_thermal", 0x40: "hw_thermal", 0x80: "hw_power_brake", 0x100: "display_clocks"}
_FIELD_VALUE = {0: "dVal", 1: "uiVal", 2: "ulVal", 3: "ullVal", 4: "sllVal", 5: "siVal", 6: "usVal"}


class PowerSampler:
    """The board's power draw (NVML's averaged reading and its instant one), SM clock, temperature and clock-event
    reasons every `interval` seconds on a thread, each sample labelled with the current `phase`. Inert, with `error`
    set, without pynvml or a GPU."""

    def __init__(self, device_index: int = 0, interval: float = 0.02):
        self.interval = interval
        self.phase = ""
        self.samples: list[tuple[str, float, float, float, float, int]] = []  # phase, W, instant W, MHz, °C, reasons
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.nv: Any = None
        self.h: Any = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self.nv, self.h = pynvml, _nvml_handle(pynvml, device_index)
        except Exception as exc:  # no NVML here: nothing is sampled
            self.error = f"{type(exc).__name__}: {exc}"
        self._instant = getattr(self.nv, "NVML_FI_DEV_POWER_INSTANT", None) if self.nv is not None else None
        self._reasons = None
        if self.nv is not None:
            self._reasons = getattr(self.nv, "nvmlDeviceGetCurrentClocksEventReasons", None) or \
                getattr(self.nv, "nvmlDeviceGetCurrentClocksThrottleReasons", None)

    def info(self) -> dict[str, Any]:
        """The board's name, driver, enforced power limit and its range, and its top SM clock."""
        if self.nv is None:
            return {"error": self.error}
        nv, h, out = self.nv, self.h, {}
        for key, fn in (("name", lambda: nv.nvmlDeviceGetName(h)), ("driver", nv.nvmlSystemGetDriverVersion),
                        ("power_limit_w", lambda: nv.nvmlDeviceGetEnforcedPowerLimit(h) / 1e3),
                        ("power_limit_range_w", lambda: [x / 1e3 for x in
                                                         nv.nvmlDeviceGetPowerManagementLimitConstraints(h)]),
                        ("max_sm_mhz", lambda: nv.nvmlDeviceGetMaxClockInfo(h, nv.NVML_CLOCK_SM)),
                        ("max_mem_mhz", lambda: nv.nvmlDeviceGetMaxClockInfo(h, nv.NVML_CLOCK_MEM))):
            try:
                v = fn()
                out[key] = v.decode() if isinstance(v, bytes) else v
            except Exception as exc:
                out[key] = f"error: {type(exc).__name__}"
        return out

    def _sample(self) -> tuple[float, float, float, float, int]:
        nv, h = self.nv, self.h
        watts = nv.nvmlDeviceGetPowerUsage(h) / 1e3
        instant = float("nan")
        if self._instant is not None:
            try:
                v = nv.nvmlDeviceGetFieldValues(h, [self._instant])[0]
                if v.nvmlReturn == 0:
                    instant = getattr(v.value, _FIELD_VALUE.get(v.valueType, "uiVal")) / 1e3
                else:
                    self._instant = None
            except Exception:
                self._instant = None
        mhz = nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM)
        temp = nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU)
        reasons = int(self._reasons(h)) if self._reasons is not None else 0
        return watts, instant, float(mhz), float(temp), reasons

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.samples.append((self.phase, *self._sample()))
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
                return
            self._stop.wait(self.interval)

    def start(self) -> PowerSampler:
        if self.nv is not None and self._thread is None:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def summary(self, phase: str, tail: float = 1.0) -> dict[str, Any]:
        """The samples of `phase` (its last `tail` share): power (averaged and instant), SM clock and temperature
        (mean, p10, p90), and the share of samples under each clock-event reason."""
        rows = [s for s in self.samples if s[0] == phase]
        rows = rows[int(len(rows) * (1 - tail)):]
        if not rows:
            return {"samples": 0, **({"error": self.error} if self.error else {})}
        a = np.asarray([s[1:5] for s in rows], np.float64)
        out: dict[str, Any] = {"samples": len(rows), "power_w": stats(a[:, 0], 1), "sm_mhz": stats(a[:, 2], 0),
                               "temp_c": stats(a[:, 3], 0)}
        if not np.isnan(a[:, 1]).all():
            out["power_instant_w"] = stats(a[~np.isnan(a[:, 1]), 1], 1)
        bits = np.asarray([s[5] for s in rows], np.int64)
        out["reasons"] = {name: round(float(((bits & b) != 0).mean()), 3) for b, name in REASONS.items()
                          if ((bits & b) != 0).any()}
        return out


def _nvml_handle(nv: Any, device_index: int) -> Any:
    """The NVML device of torch's CUDA device `device_index` (by UUID: CUDA_VISIBLE_DEVICES may renumber), else NVML's
    device of that index."""
    try:
        import torch

        uuid = str(torch.cuda.get_device_properties(device_index).uuid)
        return nv.nvmlDeviceGetHandleByUUID(uuid if uuid.startswith("GPU-") else f"GPU-{uuid}")
    except Exception:
        return nv.nvmlDeviceGetHandleByIndex(device_index)


def gpu_info(sampler: PowerSampler | None) -> dict[str, Any]:
    """torch's view of the device (name, SMs, L2, memory) and NVML's (driver, power limit, clocks)."""
    import torch

    out: dict[str, Any] = {}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(torch.cuda.current_device())
        out = {"name": p.name, "sm_count": p.multi_processor_count, "capability": f"{p.major}.{p.minor}",
               "memory_gb": round(p.total_memory / 2**30, 1), "l2_bytes": getattr(p, "L2_cache_size", None)}
    if sampler is not None:
        out["nvml"] = sampler.info()
    return out


def versions() -> dict[str, Any]:
    import platform

    import torch

    out: dict[str, Any] = {"python": platform.python_version(), "torch": torch.__version__,
                           "cuda": torch.version.cuda}
    try:
        from llm_mini_infer.kernels.flashinfer import versions as fi

        out.update(fi())
    except Exception as exc:
        out["flashinfer"] = f"error: {type(exc).__name__}"
    return out


# -- Nsight Compute's window ----------------------------------------------------------------------------------------


def ncu_window(label: str, fn: Callable[[], Any]) -> Any:
    """`fn` inside an NVTX range `label` and a cudaProfilerStart/Stop window (`ncu --profile-from-start off` profiles
    the kernels launched there), between `torch.cuda.synchronize()` calls."""
    import torch

    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push(label)
    torch.cuda.cudart().cudaProfilerStart()
    try:
        out = fn()
        torch.cuda.synchronize()
    finally:
        torch.cuda.cudart().cudaProfilerStop()
        torch.cuda.nvtx.range_pop()
    return out


# -- the plan spy ---------------------------------------------------------------------------------------------------


def plan_shape(plan: Any) -> dict[str, Any]:
    """What a plan carries, as counts (read at launch: the plan's arrays are reused two plans later)."""
    flags = plan.sched_flags
    return {"rows": int(plan.n_sched), "device": int((plan.sched_tok_off < 0).sum()),
            "sample": int(((flags & SAMPLE) != 0).sum()), "masked": int(((flags & MASKED) != 0).sum()),
            "precise": int(((flags & PRECISE) != 0).sum()), "drafts": sorted(set(plan.sched_n_draft.tolist())),
            "tokens": int(plan.n_query_tokens), "admit": len(plan.admit_slot), "free": len(plan.free_slot),
            "fold": len(plan.fold_slot), "copies": len(plan.kv_copy_src) + len(plan.wkv_copy_src) +
            len(plan.state_copy_src), "ckpt": len(plan.ckpt_sched), "encode": len(plan.enc_key),
            "spans": len(plan.span_sched), "pages": len(plan.page_id), "wpages": len(plan.wpage_id)}


def verify_problems(shape: dict[str, Any], n: int, k: int) -> list[str]:
    """How a plan's shape differs from a verify step of exactly n device rows, all MASKED, each verifying k drafts,
    and nothing else (page growth and window releases are allowed)."""
    want = {"rows": n, "device": n, "sample": n, "masked": n, "precise": 0, "drafts": [k] if n else [],
            "tokens": n * (1 + k), "admit": 0, "free": 0, "fold": 0, "copies": 0, "ckpt": 0, "encode": 0, "spans": 0}
    return [f"{key} {shape[key]} (want {v})" for key, v in want.items() if shape[key] != v]


# The grammar-aware heads' counters (`runner.base.HeadCounters`): a split step counts in one path, the sparse head or
# the dense one for a reason (its switch off; a sample token without a mask or a mask of every token; a union over the
# head's rows)
HEAD_PATHS = ("sparse", "dense_off", "dense_unconstrained", "dense_over_cap")
HEAD_TOTALS = (*HEAD_PATHS, "union_sum", "universe", "universe_steps", "universe_rebuilds")


def head_counts(runner: Any) -> dict[str, int] | None:
    """The runner's grammar-aware head counters (`Runner.head_counters`: cumulative host ints, read by field so that a
    runner without them gives None), with `unions` the steps whose union was measured."""
    fn = getattr(runner, "head_counters", None)
    if fn is None:
        return None
    c = fn()
    return {**{f: int(getattr(c, f, 0)) for f in HEAD_TOTALS}, "unions": int(sum(getattr(c, "union_counts", ())))}


def head_step(a: dict[str, int], b: dict[str, int]) -> dict[str, Any]:
    """One step's heads from the counters before (`a`) and after (`b`) its sampling launch: the path it took (None:
    the runner counted none, as one without the heads), its union's tokens (None: none measured, as for a sample token
    without a mask), the universe after it, whether its drafts scored the universe, and whether the universe head was
    rebuilt for it (the universe grew)."""
    return {"path": "+".join(p for p in HEAD_PATHS if b[p] != a[p]) or None,
            "union": b["union_sum"] - a["union_sum"] if b["unions"] - a["unions"] == 1 else None,
            "universe": b["universe"], "drafted_universe": b["universe_steps"] > a["universe_steps"],
            "rebuilt": b["universe_rebuilds"] > a["universe_rebuilds"]}


class StepSpy:
    """Wraps a runner's `launch_forward`, `_mode`, `launch_sample` and `wait`: while `on`, records each plan the engine
    launches (its shape), the body the runner picked for it ((kind, bucket, split): "decode" graphs, "break" graphs or
    "eager"), the grammar-aware heads its sampling took (`head_step`, from the runner's counters around the engine's
    `launch_sample`; nothing on a runner without them), and each step result the engine waited for."""

    def __init__(self, runner: Any):
        self.runner = runner
        self.on = False
        self.plans: list[dict[str, Any]] = []
        self.modes: list[tuple[str, int, bool]] = []
        self.heads: list[dict[str, Any]] = []
        self.results: list[Any] = []
        launch, mode, sample, wait = runner.launch_forward, runner._mode, runner.launch_sample, runner.wait
        counted = head_counts(runner) is not None

        def spy_launch(plan: Any, _internal: bool = False, _capture: Any = None) -> Any:
            if self.on and not _internal:
                self.plans.append(plan_shape(plan))
            return launch(plan, _internal=_internal, _capture=_capture)

        def spy_mode(lay: Any, split: bool = False) -> tuple[str, int]:
            out = mode(lay, split)
            if self.on:
                self.modes.append((out[0], int(out[1]), bool(split)))
            return out

        def spy_sample(h: Any, n_masks: int = 0, **internal: Any) -> None:
            if not (self.on and counted) or internal:  # the runner's own steps (its warm-up) name their heads
                return sample(h, n_masks, **internal)
            before = head_counts(runner)
            sample(h, n_masks)
            self.heads.append(head_step(before, head_counts(runner)))
            return None

        def spy_wait(h: Any) -> Any:
            res = wait(h)
            if self.on:
                self.results.append(res)
            return res

        runner.launch_forward, runner._mode, runner.launch_sample, runner.wait = spy_launch, spy_mode, spy_sample, \
            spy_wait

    def start(self) -> None:
        self.plans, self.modes, self.heads, self.results = [], [], [], []
        self.on = True

    def stop(self) -> None:
        self.on = False

    def check(self, n: int, k: int) -> dict[str, Any]:
        """Raises unless every recorded plan is a verify step of n rows (`verify_problems`) that the runner split (and
        on CUDA graphs ran from the decode graphs of n's bucket), and every one's sampling took the same grammar-aware
        head (on a runner that counts them); returns the plans' shapes, modes and heads (`heads`), counted."""
        bad = [(i, p) for i, s in enumerate(self.plans) if (p := verify_problems(s, n, k))]
        r = self.runner
        if not bad and len(self.modes) != len(self.plans):
            bad.append((-1, [f"{len(self.modes)} modes for {len(self.plans)} plans"]))
        bucket = next((b for b in getattr(r, "decode_buckets", []) if b >= n), None)
        for i, (kind, size, split) in enumerate(self.modes):
            if not split:
                bad.append((i, [f"not split ({kind} {size})"]))
            elif getattr(r, "use_graphs", False) and (kind, size) != ("decode", bucket):
                bad.append((i, [f"ran {kind} {size}, not the decode graphs of bucket {bucket}"]))
        if bad:
            raise AssertionError(f"{len(bad)} of {len(self.plans)} measured steps are not verify steps of {n} masked "
                                 f"rows: " + "; ".join(f"step {i}: {', '.join(p)}" for i, p in bad[:5]))
        shapes = Counter(json.dumps({k_: v for k_, v in s.items() if k_ in ("pages", "wpages")}, sort_keys=True)
                         for s in self.plans)
        return {"steps": len(self.plans), "rows": n, "drafts": k,
                "page_writes": {s: c for s, c in sorted(shapes.items())},
                "modes": dict(Counter(f"{kind}/{size}/{'split' if split else 'fused'}"
                                      for kind, size, split in self.modes)),
                "heads": self.check_heads()}

    def check_heads(self) -> dict[str, Any] | None:
        """The measured steps' grammar-aware heads (None: the runner counts none): the one path they all took (a
        window that mixes two would average them: an error), the union's tokens over the steps (`stats`), the universe
        at the end, the steps whose drafts scored it, and its rebuilds."""
        if not self.heads:
            return None
        if len(self.heads) != len(self.plans):
            raise AssertionError(f"{len(self.heads)} sampling launches for {len(self.plans)} measured steps")
        paths = [s["path"] or "uncounted" for s in self.heads]
        if len(set(paths)) > 1:
            raise AssertionError(f"the {len(self.plans)} measured steps did not all take one grammar-aware head: "
                                 f"{dict(Counter(paths))}")
        unions = [s["union"] for s in self.heads if s["union"] is not None]
        return {"path": paths[0], "union": stats(unions, 1) if unions else None,
                "universe": self.heads[-1]["universe"],
                "universe_steps": sum(s["drafted_universe"] for s in self.heads),
                "universe_rebuilds": sum(s["rebuilt"] for s in self.heads)}


# -- requests and the engine ----------------------------------------------------------------------------------------


def token_range(vocab: int) -> tuple[int, int]:
    """Random prompt ids: ordinary text ids of a real vocabulary, clear of a tiny model's special ids at its top."""
    return (1000, min(vocab, 150_000)) if vocab > 4000 else (16, vocab - 16)


class Prompts:
    """The shared random prefix, and a new random tail after it for each request."""

    def __init__(self, prefix: int, tail: int, vocab: int, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.lo, self.hi = token_range(vocab)
        self.prefix_len, self.tail = prefix, tail
        self.prefix = self.rng.integers(self.lo, self.hi, prefix).astype(np.int32)

    def next(self) -> np.ndarray:
        return np.concatenate([self.prefix, self.rng.integers(self.lo, self.hi, self.tail).astype(np.int32)])


def eos_ids(model: Path) -> tuple[int, ...]:
    """The model's stop ids as the server derives them (generation_config.json's eos_token_id, else config.json's);
    the requests ignore them."""
    f = model / "generation_config.json"
    gen = json.loads(f.read_text()) if f.exists() else {}
    hf = json.loads((model / "config.json").read_text())
    eos = gen.get("eos_token_id", hf.get("text_config", hf).get("eos_token_id", hf.get("eos_token_id")))
    return tuple(sorted({int(e) for e in (eos if isinstance(eos, list) else [eos]) if e is not None}))


class Harness:
    """The scheduler (window mode, prefix caching, MTP, the permissive grammar or `grammar`'s: `grammar_backend`) and
    the engine loop over a runner, as `engine.process` builds them for a sliding-window model, in-process: requests by
    rid, the tokens delivered to each, and each one's prompt tokens served from the cache."""

    def __init__(self, runner: Any, pc: Any, eos: tuple[int, ...] = (), grammar: str = "allow-all",
                 boundary: bool = False):
        self.runner, self.pc = runner, pc
        self.boundary = boundary  # each turn first polls until the step in flight's forward ends
        self.caps = caps = runner.capabilities()
        self.cuda = bool(getattr(runner, "cuda", False))
        self.grammars = Grammars(grammar_backend(grammar, caps.vocab_size, eos))
        grid = pc.window_page_size or caps.checkpoint_grid
        self.sched = Scheduler(
            max_num_seqs=pc.max_num_seqs, max_num_batched_tokens=pc.max_num_batched_tokens,
            max_model_len=pc.max_model_len, page_size=pc.page_size, num_pages=pc.num_pages,
            num_state_slots=pc.num_state_slots, eos_token_ids=eos, steps_in_flight=caps.max_steps_in_flight,
            mtp_k=caps.mtp_depth, num_live_slots=pc.max_num_seqs if caps.window else None,
            hybrid=caps.state_slot_bytes > 0, prefix_caching=True, checkpoint_grid=grid, grammars=self.grammars,
            window=caps.window, window_page_size=pc.window_page_size, draft_kv=caps.draft_kv)
        self.core = EngineCore(self.sched, runner)
        self.got: dict[int, int] = {}
        self.cached: dict[int, int] = {}
        self.finished: dict[int, int] = {}
        self.live: list[int] = []  # added, not finished or aborted, in order
        self.rid = 0
        self.steps = 0

    def add(self, prompt: np.ndarray, *, max_tokens: int, hints: tuple[int, ...] = ()) -> int:
        rid = self.rid
        self.rid += 1
        self.core.handle(AddRequest(rid=rid, frontend=0, prompt=np.asarray(prompt, np.int32), max_tokens=max_tokens,
                                    temperature=0.0, top_p=1.0, top_k=0, min_p=0.0, seed=1 + rid, stop_token_ids=[],
                                    ignore_eos=True, cache_hints=list(hints), grammar=GRAMMAR))
        self.got[rid] = 0
        self.live.append(rid)
        return rid

    def abort(self, rids: list[int]) -> None:
        self.core.handle(Abort(frontend=0, rids=list(rids)))
        self.live = [r for r in self.live if r not in set(rids)]

    def step(self) -> Any:
        if self.boundary:
            self.core.await_boundary(lambda: 0)
        rep = self.core.step()
        self.steps += 1
        for fo in rep.outputs.values() if rep is not None else ():
            for rid, cached in zip(fo.first_rids.tolist(), fo.first_cached.tolist()):
                self.cached[rid] = cached
            for rid, n, fin in zip(fo.rids.tolist(), fo.n_new.tolist(), fo.finish.tolist()):
                self.got[rid] = self.got.get(rid, 0) + n
                if fin:
                    self.finished[rid] = fin
                    if rid in self.live:
                        self.live.remove(rid)
        return rep

    def run_until(self, done: Callable[[], bool], limit: int = 2000) -> int:
        n = 0
        while not done():
            if n >= limit:
                raise RuntimeError(f"not done after {limit} steps")
            self.step()
            n += 1
        return n

    def decoding(self, rids: list[int] | None = None) -> bool:
        """Whether every request of `rids` (default: every live one) has its first token: its rows are decode rows."""
        return all(self.got.get(r, 0) > 0 for r in (self.live if rids is None else rids))

    def sync(self) -> None:
        if self.cuda:
            import torch

            torch.cuda.synchronize()

    def rows(self, prefix: int) -> dict[str, Any]:
        """The live rows: KV length, tokens generated, prompt tokens served from the cache, the full pages they share
        from position 0, and the positions of each one's window that lie in the shared prefix's window."""
        s, w = self.sched, self.caps.window
        reqs = [s.reqs[(0, r)] for r in self.live]
        seq = [int(s.mirror[q.slot]) for q in reqs]
        pages = [list(q.pages) for q in reqs]
        common = 0
        while pages and all(len(p) > common for p in pages) and len({p[common] for p in pages}) == 1:
            common += 1
        return {"rids": list(self.live), "seq_len": stats(seq, 1),
                "generated": stats([self.got[r] for r in self.live], 1),
                "cached": [self.cached.get(r, -1) for r in self.live], "shared_full_pages": common,
                "shared_window_positions": stats([max(0, prefix - max(0, n - w + 1)) for n in seq], 1) if w else None}


def setup_rows(h: Harness, prompts: Prompts, n: int, max_tokens: int, decode_before: int) -> dict[str, Any]:
    """The first request alone to its first token (a cache hint at the prefix: a checkpoint there), then n - 1 more,
    stepped until every one decodes, then `decode_before` more steps."""
    t0, s0 = time.perf_counter(), h.steps
    first = h.add(prompts.next(), max_tokens=max_tokens, hints=(prompts.prefix_len,))
    h.run_until(lambda: h.got[first] > 0)
    for _ in range(n - 1):
        h.add(prompts.next(), max_tokens=max_tokens)
    h.run_until(h.decoding)
    for _ in range(decode_before):
        h.step()
    return {"steps": h.steps - s0, "seconds": round(time.perf_counter() - t0, 2),
            "cached": {str(r): h.cached.get(r, -1) for r in h.live}}


def shrink(h: Harness, n: int) -> list[int]:
    """Abort the latest live requests beyond n."""
    extra = h.live[n:]
    if extra:
        h.abort(extra)
    return extra


def warm(h: Harness, prompts: Prompts, n: int, seconds: float, *, min_steps: int, max_tokens: int, keep: int = 0,
         sampler: PowerSampler | None = None, phase: str = "") -> dict[str, Any]:
    """Step n decoding rows for `seconds` and at least `min_steps` steps. A row that finishes, or has fewer than `keep`
    tokens left (the measurement's), is replaced at once; the warm-up ends with every row decoding and holding `keep`
    tokens (checked after the last step, not before it), 3 steps past any change."""
    if keep > max_tokens // 2:
        raise ValueError(f"rows of {max_tokens} tokens cannot keep {keep} for the measurement: raise --max-tokens")
    if sampler is not None:
        sampler.phase = phase
    t0, steps, replaced, last = time.perf_counter(), 0, 0, 0
    while True:
        short = [r for r in h.live if h.sched.reqs[(0, r)].max_out - h.got[r] < keep]
        if short:
            h.abort(short)
        while len(h.live) < n:
            h.add(prompts.next(), max_tokens=max_tokens)
            replaced += 1
            last = steps
        if steps >= min_steps and time.perf_counter() - t0 >= seconds and h.decoding() and steps - last >= 3:
            break
        if time.perf_counter() - t0 > seconds + 60:
            raise RuntimeError(f"the warm-up of {n} rows did not settle in {seconds + 60:.0f} s ({replaced} replaced)")
        h.step()
        steps += 1
    if sampler is not None:
        sampler.phase = ""
    return {"seconds": round(time.perf_counter() - t0, 2), "steps": steps, "replaced": replaced}


def refresh(h: Harness, prompts: Prompts, n: int, max_tokens: int, decode_before: int) -> dict[str, Any]:
    """Replace the live rows with n new requests on the cached prefix, stepped until they decode, then
    `decode_before` more steps."""
    t0, s0 = time.perf_counter(), h.steps
    h.abort(list(h.live))
    new = [h.add(prompts.next(), max_tokens=max_tokens) for _ in range(n)]
    h.run_until(lambda: h.decoding(new))
    for _ in range(decode_before):
        h.step()
    return {"steps": h.steps - s0, "seconds": round(time.perf_counter() - t0, 2)}


def check_budget(h: Harness, steps: int, k: int) -> None:
    """Raises when a live row could reach max_tokens within `steps` steps (1 + k tokens each): it would leave the
    measured steps."""
    for r in h.live:
        left = h.sched.reqs[(0, r)].max_out - h.got[r]
        if left < (steps + 2) * (1 + k):
            raise RuntimeError(f"request {r} has {left} tokens left, fewer than {steps} steps of {1 + k} may take: "
                               "raise --max-tokens or lower --steps")


def measure_steps(h: Harness, spy: StepSpy, n: int, k: int, calls: int) -> tuple[list[float], dict[str, Any]]:
    """`calls` engine turns with the spy on: each launches one step, which must be a verify step of n masked rows
    (`StepSpy.check`). Returns the turns' start times (and the end) and the checked plans."""
    spy.start()
    t = []
    try:
        for _ in range(calls):
            t.append(time.perf_counter())
            h.step()
        t.append(time.perf_counter())
    finally:
        spy.stop()
    return t, spy.check(n, k)


def unprofiled(h: Harness, spy: StepSpy, n: int, k: int, steps: int,
               sampler: PowerSampler | None = None) -> dict[str, Any]:
    """K steps without the profiler, in the engine's own rhythm: K + 1 turns, the steps launched by the first K
    measured (their results come in turns 2..K + 1): device time and idle from the runner's events, wall time per turn,
    tokens per step, and NVML's power and clocks over the turns."""
    c0 = h.core
    idle0, waits0, ph0 = c0.gpu_idle_ns, c0.steps, list(c0.phase_ns)
    if sampler is not None:
        sampler.phase = f"unprofiled_b{n}"
    t, plans = measure_steps(h, spy, n, k, steps + 1)
    if sampler is not None:
        sampler.phase = ""
    res = spy.results[1:steps + 1]
    busy = [r.gpu_busy_ns / 1e6 for r in res]
    inside = [r.gpu_idle_inside_ns / 1e6 for r in res]
    before = [r.gpu_idle_before_ns / 1e6 for r in res]
    sample = [getattr(r, "gpu_sample_ns", 0) / 1e6 for r in res]  # the split's sampling part
    toks = [int(np.asarray(r.n_out).sum()) for r in res]
    turns = steps + 1  # the engine's host time a turn, by phase (engine.core.PHASES), but the poll and the send,
    # which only the process loop runs (no frontends here)
    host = {name: round((b - a) / 1e6 / turns, 4) for name, a, b in zip(PHASES, ph0, c0.phase_ns)
            if name not in ("poll", "send")}
    out = {"steps": steps, "plans": plans, "busy_ms": stats(busy), "idle_inside_ms": stats(inside),
           "idle_before_ms": stats(before), "sample_ms": stats(sample),
           "forward_ms": stats([b - s for b, s in zip(busy, sample)]), "host_ms_per_turn": host,
           "core_idle_ms_per_step": round((c0.gpu_idle_ns - idle0) / 1e6 / max(1, c0.steps - waits0), 4),
           "wall_ms": round((t[-1] - t[1]) / steps * 1e3, 3), "tokens_per_step": stats(toks, 2),
           "accepted_per_row": round(float(np.mean(toks)) / n - 1, 3) if toks else None}
    if sampler is not None:
        out["nvml"] = sampler.summary(f"unprofiled_b{n}")
    return out


def profiled(h: Harness, spy: StepSpy, n: int, k: int, steps: int, sampler: PowerSampler | None = None,
             trace: Path | None = None) -> dict[str, Any]:
    """K steps under torch.profiler (CUDA activity: CUPTI's kernels, copies and sets, graph replays included), between
    `torch.cuda.synchronize()` calls: the steps launched in the window are exactly the K measured. Their kernels by
    category, and by launch from the window's trace (`timeline`: the forward graph, the sampling graph, what runs
    outside them, the gaps inside each graph); NVML's power and clocks over the window (the synchronization idles the
    GPU briefly, and its kernels run at those clocks). `trace`: keep the trace there."""
    import torch

    h.sync()
    if sampler is not None:
        sampler.phase = f"profiled_b{n}"
    t0 = time.perf_counter()
    with torch.profiler.profile(activities=activities()) as prof:
        _, plans = measure_steps(h, spy, n, k, steps)
        h.sync()
    wall = time.perf_counter() - t0
    if sampler is not None:
        sampler.phase = ""
    out = {"plans": plans, "wall_ms": round(wall / steps * 1e3, 3),
           **summarize_kernels(profiler_rows(prof), steps, VERIFY_CATEGORIES)}
    out["timeline"] = timeline(export_trace(prof, trace, f"verify_b{n}"), steps, VERIFY_CATEGORIES,
                               parts=("forward", "sample"))
    if sampler is not None:
        out["nvml"] = sampler.summary(f"profiled_b{n}")
    return out


def export_trace(prof: Any, keep: Path | None, name: str) -> dict[str, Any]:
    """The profile's Chrome trace (torch.profiler's documented export), kept as `keep`/`name`.json when asked."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = (keep or Path(tmp)) / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(path))
        return json.loads(path.read_text())


def ncu_steps(h: Harness, spy: StepSpy, n: int, k: int, steps: int) -> dict[str, Any]:
    t0 = time.perf_counter()
    _, plans = ncu_window(f"verify_b{n}", lambda: measure_steps(h, spy, n, k, steps))
    return {"plans": plans, "window_s": round(time.perf_counter() - t0, 2), "nvtx": f"verify_b{n}"}


def profile(h: Harness, spy: StepSpy, prompts: Prompts, *, rows: list[int], k: int, steps: int, warm_s: float,
            min_warm_steps: int, decode_before: int, max_tokens: int, fresh: bool, mode: str,
            sampler: PowerSampler | None = None, trace: Path | None = None, freeze_gc: bool = False,
            log: Callable[[str], None] = print, out: dict[str, Any] | None = None) -> dict[str, Any]:
    """The whole run over a harness (module docstring): the setup, then per row count (descending) the warm-up, the
    optional refresh and the measurement (`mode`: torch, ncu, or none: the unprofiled steps alone). `freeze_gc`: the
    collector's objects frozen after the setup, as the engine process does (dev.engine.gc_freeze). `out`: the dict
    filled as the run goes (returned), which keeps the row counts measured so far when a later one fails."""
    rows = sorted(set(rows), reverse=True)
    out = {} if out is None else out
    out["setup"] = setup_rows(h, prompts, rows[0], max_tokens, decode_before)
    log(f"setup: {len(h.live)} rows decoding after {out['setup']['steps']} steps ({out['setup']['seconds']} s); "
        f"cached {out['setup']['cached']}")
    if freeze_gc:
        gc.collect()
        gc.freeze()
        gc.set_threshold(10_000, 50, 100)
    for n in rows:
        r: dict[str, Any] = {"aborted": shrink(h, n)}
        out[str(n)] = r  # filled in place: what a failure leaves measured stays
        r["warm"] = warm(h, prompts, n, warm_s, min_steps=min_warm_steps, max_tokens=max_tokens,
                         keep=0 if fresh else (2 * steps + 4) * (1 + k), sampler=sampler, phase=f"warm_b{n}")
        if sampler is not None:
            r["warm"]["nvml"] = sampler.summary(f"warm_b{n}", tail=0.5)
        if fresh:
            r["fresh"] = refresh(h, prompts, n, max_tokens, decode_before)
        check_budget(h, 2 * steps + 2, k)
        r["context"] = h.rows(prompts.prefix_len)
        if mode in ("torch", "none"):
            r["unprofiled"] = unprofiled(h, spy, n, k, steps, sampler)
        if mode == "torch":
            r["torch"] = profiled(h, spy, n, k, steps, sampler, trace)
            r["kernel_share_of_busy"] = round(r["torch"]["kernel_ms"] / r["unprofiled"]["busy_ms"]["mean"], 4) \
                if r["unprofiled"]["busy_ms"].get("mean") else None
        elif mode == "ncu":
            r["ncu"] = ncu_steps(h, spy, n, k, steps)
        log(line(n, r))
    out["prefix_hits"] = {"queries": h.sched.prefix_queries, "hits": h.sched.prefix_hits}
    out["grammar"] = {"backend": h.grammars.backend.name, "masks": h.grammars.mask_rows, "errors": h.grammars.errors}
    out["heads"] = head_counts(h.runner)  # the whole run's, warm-ups included (None: a runner without the heads)
    return out


def heads_text(heads: dict[str, Any] | None) -> str:
    """A measurement's heads (`StepSpy.check_heads`) in a few words: the path, the union's mean tokens, the universe."""
    if not heads:
        return "-"
    union = heads["union"]["mean"] if heads["union"] else None
    return heads["path"] + (f" {union:g}" if union is not None else "") + \
        (f", universe {heads['universe']}" if heads["universe"] else "")


def line(n: int, r: dict[str, Any]) -> str:
    u, t = r.get("unprofiled"), r.get("torch")
    parts = [f"{n} rows: warm {r['warm']['seconds']} s / {r['warm']['steps']} steps",
             f"context {r['context']['seq_len'].get('mean')} (generated {r['context']['generated'].get('mean')})"]
    m = u or t or r.get("ncu")
    if m and m["plans"].get("heads"):
        parts.append(f"head {heads_text(m['plans']['heads'])}")
    if u:
        nv = u.get("nvml", {})
        parts.append(f"busy {u['busy_ms'].get('mean')} ms, idle {u['idle_inside_ms'].get('mean')} + "
                     f"{u['idle_before_ms'].get('mean')} ms, wall {u['wall_ms']} ms, "
                     f"{u['tokens_per_step'].get('mean')} tokens/step; {nv.get('power_w', {}).get('mean', '-')} W, "
                     f"{nv.get('sm_mhz', {}).get('mean', '-')} MHz")
    if t:
        parts.append(f"kernels {t['kernel_ms']} ms ({t['kernels']} per step): " +
                     ", ".join(f"{c} {v['ms']:.2f}" for c, v in t["categories"].items() if v["ms"]))
    if r.get("ncu"):
        parts.append(f"ncu window {r['ncu']['window_s']} s")
    return "; ".join(parts)


def table(res: dict[str, Any], rows: list[int]) -> str:
    """The summary per row count: the step's device time, idle, tokens, power and clock (the unprofiled steps) and its
    grammar-aware head (the path and the union's tokens); with a CUPTI profile, its graphs (busy + gaps), kernels, and
    each category's ms and share."""
    keys = [str(n) for n in sorted(rows, reverse=True) if "unprofiled" in res.get(str(n), {})]
    if not keys:
        return ""
    out = [f"{'':22}" + "".join(f"{f'{n} rows':>20}" for n in keys)]

    def row(label: str, fn: Callable[[dict[str, Any]], str]) -> None:
        out.append(f"{label:22}" + "".join(f"{fn(res[n]):>20}" for n in keys))

    def part(r: dict[str, Any], name: str) -> str:
        p = r["torch"]["timeline"].get(name)
        return f"{p['busy_ms']:.2f} + {p['gaps_ms']:.2f}" if isinstance(p, dict) else "-"

    def heads(r: dict[str, Any]) -> dict[str, Any]:
        return r["unprofiled"]["plans"].get("heads") or {}

    row("busy ms/step", lambda r: f"{r['unprofiled']['busy_ms']['mean']:.2f}")
    row("idle ms/step", lambda r: f"{r['unprofiled']['idle_inside_ms']['mean']:.3f} + "
                                  f"{r['unprofiled']['idle_before_ms']['mean']:.3f}")
    row("tokens/step", lambda r: f"{r['unprofiled']['tokens_per_step']['mean']:.2f}")
    row("power W (mean)", lambda r: str(r["unprofiled"].get("nvml", {}).get("power_w", {}).get("mean", "-")))
    row("SM MHz (mean)", lambda r: str(r["unprofiled"].get("nvml", {}).get("sm_mhz", {}).get("mean", "-")))
    if any(heads(res[n]) for n in keys):
        row("head", lambda r: heads(r).get("path", "-"))
        row("union tokens (mean)", lambda r: f"{heads(r)['union']['mean']:g}" if heads(r).get("union") else "-")
    if not all("torch" in res[n] for n in keys):
        return "\n".join(out)
    row("forward graph + gaps", lambda r: part(r, "forward"))
    row("sample graph + gaps", lambda r: part(r, "sample"))
    row("outside graphs ms", lambda r: f"{r['torch']['timeline']['single_busy_ms']:.2f}")
    row("kernel ms/step", lambda r: f"{r['torch']['kernel_ms']:.2f}")
    row("kernel share of busy", lambda r: f"{100 * (r.get('kernel_share_of_busy') or 0):.1f}%")
    row("kernels/step", lambda r: f"{r['torch']['kernels']:.0f}")
    for c in res[keys[0]]["torch"]["categories"]:
        row(c, lambda r, c=c: f"{r['torch']['categories'][c]['ms']:.2f} "
                              f"({100 * r['torch']['categories'][c]['share']:.1f}%)")
    return "\n".join(out)


# -- the runner -------------------------------------------------------------------------------------------------------


def make_runner(model: str, *, max_num_seqs: int, budget: int, max_model_len: int, memory_gb: float | None = None,
                fp8_gemm: str = "auto", mtp_k: int = 0, assistant: str | None = None, **opts: Any):
    """Gemma 4's `GemmaRunner` on CUDA with its KV arena sized as the engine sizes it (window pages of 16 positions,
    full pages of as many bytes) in `memory_gb` (None: the free memory, profiled), configured; returns it and its
    `PoolConfig`. `opts`: the GemmaRunner's switches (the grammar-aware heads, PDL, always-split graphs, the
    batch-invariant parts), each passed only when not None (None: the runner's default)."""
    from llm_mini_infer.cache.pools import size_arena
    from llm_mini_infer.models.gemma4.model import is_gemma4
    from llm_mini_infer.runner.base import PoolConfig
    from llm_mini_infer.runner.cuda.gemma import GemmaRunner

    if not is_gemma4(model):
        raise ValueError(f"{model}: this harness profiles Gemma 4")
    runner = GemmaRunner(model, device="cuda", max_num_seqs=max_num_seqs, max_num_batched_tokens=budget,
                         max_model_len=max_model_len, memory_gb=memory_gb, fp8_gemm=fp8_gemm, mtp_k=mtp_k,
                         assistant=assistant, **{k: v for k, v in opts.items() if v is not None})
    caps = runner.capabilities()
    ws = size_arena(memory_bytes=caps.memory_free_bytes, kv_bytes_per_token=caps.kv_bytes_per_token,
                    window_kv_bytes_per_token=caps.window_kv_bytes_per_token, window=caps.window, window_page_size=16,
                    max_num_seqs=max_num_seqs, max_model_len=max_model_len, max_num_batched_tokens=budget)
    pc = PoolConfig(page_size=ws.page_size, num_pages=ws.num_pages, num_state_slots=max_num_seqs + 1,
                    max_num_seqs=max_num_seqs, max_num_batched_tokens=budget, max_model_len=max_model_len,
                    eos_token_ids=(), mtp_k=mtp_k, window_page_size=16)
    runner.configure(pc)
    return runner, pc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", required=True, help="the FP8 checkpoint (RedHatAI/gemma-4-31B-it-FP8-block)")
    ap.add_argument("--assistant", default=None, help="the MTP assistant (google/gemma-4-31B-it-assistant)")
    ap.add_argument("--k", type=int, default=5, help="MTP drafts (0: MTP off; the engine's default with an assistant)")
    ap.add_argument("--rows", default="8,4", help="decode rows to measure: a comma list (any order: measured "
                    "descending), e.g. 8,7,6,5,4,3,2,1")
    ap.add_argument("--prefix", type=int, default=2048, help="the shared system prompt's tokens")
    ap.add_argument("--tail", type=int, default=256, help="each request's own tokens after the prefix")
    ap.add_argument("--steps", type=int, default=None, help="measured steps (K) per row count (default 20; ncu: 3)")
    ap.add_argument("--warm-s", type=float, default=10.0,
                    help="seconds of steps before each row count's measurement (clocks settle under the power limit)")
    ap.add_argument("--min-warm-steps", type=int, default=8)
    ap.add_argument("--decode-before", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=1024,
                    help="tokens a request generates: enough for the measured steps (checked at start)")
    ap.add_argument("--fresh", action=argparse.BooleanOptionalAction, default=True,
                    help="measure new rows on the cached prefix after each warm-up (module docstring)")
    ap.add_argument("--mode", choices=("torch", "ncu", "none"), default="torch")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--trace", type=Path, default=None, help="also each window's Chrome trace, in this directory")
    ap.add_argument("--memory-gb", type=float, default=16.0, help="the KV arena (skips the memory profile)")
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--budget", type=int, default=2048, help="max_num_batched_tokens")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--fp8-gemm", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grammar", type=grammar_spec, default="allow-all",
                    help="the requests' grammar: allow-all (all-ones masks) or subset:<U> (every mask the same U "
                         "tokens, no EOS: the grammar-aware heads at a union of U; module docstring)")
    ap.add_argument("--sparse-head", action=argparse.BooleanOptionalAction, default=True,
                    help="the grammar-sparse target head (dev.runner.cuda.sparse_head)")
    ap.add_argument("--draft-universe", action=argparse.BooleanOptionalAction, default=True,
                    help="the assistant's head over the grammar's universe (dev.speculative.draft_universe; with "
                         "--k > 0)")
    ap.add_argument("--pdl", action=argparse.BooleanOptionalAction, default=True,
                    help="programmatic dependent launch (dev.runner.cuda.pdl; default on, as the server's)")
    ap.add_argument("--always-split", action=argparse.BooleanOptionalAction, default=True,
                    help="every step a forward and a sampling graph, no fused graphs captured (dev.runner.cuda."
                         "always_split; default on, as the server's with structured output)")
    ap.add_argument("--attn512-split-keys", type=int, default=0,
                    help="the full layers' fixed split, keys a split (kernels/attn512.py; answer reproducibility; "
                         "0: the wave split; the server's with dev.runner.cuda.batch_invariant: 256)")
    ap.add_argument("--window-split-keys", type=int, default=0,
                    help="the sliding layers on our kernel, a decode row split by that many keys "
                         "(kernels/attn_window.py; 0: FlashInfer; the server's with dev.runner.cuda.batch_invariant: "
                         "256)")
    ap.add_argument("--head-fixed-m", action="store_true",
                    help="the sparse head's GEMM at one M (answer reproducibility; the server's with "
                         "dev.runner.cuda.batch_invariant)")
    ap.add_argument("--boundary", action=argparse.BooleanOptionalAction, default=False,
                    help="each turn first polls until the step in flight's forward ends (engine.plan_at_boundary; "
                         "default off)")
    args = ap.parse_args()
    args.steps = args.steps or (3 if args.mode == "ncu" else 20)
    if args.k and not args.assistant:
        ap.error("--k > 0 needs --assistant")
    try:
        rows = row_counts(args.rows, args.max_num_seqs)
    except ValueError as exc:
        ap.error(str(exc))
    if args.prefix + args.tail + args.max_tokens + 3 * (1 + args.k) > args.max_model_len:
        ap.error("--prefix + --tail + --max-tokens must leave room under --max-model-len")
    # a row must not finish within the measured steps (`check_budget`): after its warm-up's keep (without --fresh,
    # at most half of max_tokens) or its fresh decode_before steps, 2 * steps + 4 steps of 1 + k tokens
    need = (2 * args.steps + 4) * (1 + args.k)
    least = need + (args.decode_before + 1) * (1 + args.k) if args.fresh else 2 * need
    if args.max_tokens < least:
        ap.error(f"--max-tokens {args.max_tokens} is too few for {args.steps} steps at k = {args.k}: at least {least}")
    import torch

    def log(msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    t0 = time.perf_counter()
    runner, pc = make_runner(args.model, max_num_seqs=args.max_num_seqs, budget=args.budget,
                             max_model_len=args.max_model_len, memory_gb=args.memory_gb, fp8_gemm=args.fp8_gemm,
                             mtp_k=args.k, assistant=args.assistant if args.k else None, sparse_head=args.sparse_head,
                             draft_universe=args.draft_universe, pdl=args.pdl,
                             always_split=args.always_split, attn512_split_keys=args.attn512_split_keys,
                             window_split_keys=args.window_split_keys, head_fixed_m=args.head_fixed_m)
    log(f"runner ready in {time.perf_counter() - t0:.0f} s: {pc.num_pages} arena pages, graphs "
        f"{runner.stats.get('graphs')}; the sparse head {runner.stats.get('sparse_head', '-')}, the draft universe "
        f"{runner.stats.get('draft_universe', '-')}; PDL {runner.stats.get('pdl', '-')}; always split "
        f"{runner.stats.get('always_split', '-')}; planning at the boundary {args.boundary}; grammar {args.grammar}; "
        f"attn512's fixed split {args.attn512_split_keys}, the window kernel's {args.window_split_keys}, the head at "
        f"one M {args.head_fixed_m}")
    if args.mode == "torch":
        init_cupti()
    sampler = PowerSampler(torch.cuda.current_device()).start() if args.mode != "ncu" else None
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config.update(rows=rows, versions=versions(), gpu=gpu_info(sampler or PowerSampler(torch.cuda.current_device())),
                  pools={"page_size": pc.page_size, "window_page_size": pc.window_page_size, "num_pages": pc.num_pages},
                  runner=json.loads(json.dumps(runner.stats, default=str)))
    h = Harness(runner, pc, eos=eos_ids(Path(args.model)), grammar=args.grammar, boundary=args.boundary)
    spy = StepSpy(runner)
    prompts = Prompts(args.prefix, args.tail, h.caps.vocab_size, seed=args.seed)
    res: dict[str, Any] = {}
    try:
        profile(h, spy, prompts, rows=rows, k=args.k, steps=args.steps, warm_s=args.warm_s,
                min_warm_steps=args.min_warm_steps, decode_before=args.decode_before, max_tokens=args.max_tokens,
                fresh=args.fresh, mode=args.mode, sampler=sampler, trace=args.trace, freeze_gc=True, log=log, out=res)
    except BaseException as exc:  # what was measured is written all the same, with the error
        res["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if sampler is not None:
            sampler.stop()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"config": config, **res}, indent=1, default=str))
        log(f"wrote {args.out}{' (partial: the run failed)' if 'error' in res else ''}")
    print(table(res, rows), flush=True)
    runner.shutdown()


if __name__ == "__main__":
    main()
