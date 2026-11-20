"""The window cache with MTP: its harness, workloads and driver (tests/sched/test_window_mtp.py)."""

from __future__ import annotations

from llm_mini_infer.cache.pools import pages_for_one_request, window_pages_for_one_request
from llm_mini_infer.engine.messages import FinishReason
from llm_mini_infer.sched.plan import STOP_W, reserve_positions

from .fuzzing import event, st
from .harness import Harness, Spec, check_nan_invariants
from .sched_cases import VOCAB, check_plans, nan_events, output_of
from .window_workloads import arena_pressure, gemma_image_workloads, window_workloads


def bound(window: int, page_size: int, full: int, max_tokens: int, max_model_len: int, k: int) -> int:
    """The arena pages one request may need at once under MTP k (`cache.pools.size_arena`'s startup bound)."""
    reserve = reserve_positions(2, k)
    return pages_for_one_request(max_model_len, full, reserve, True) + \
        window_pages_for_one_request(window, max_tokens, page_size, reserve)


def harness(specs: list[Spec], *, k: int = 3, window: int = 16, page_size: int = 4, full: int = 0,
            num_pages: int = 0, max_tokens: int = 64, max_model_len: int = 512, draft_kv: bool = False,
            **kw) -> Harness:
    full = full or 5 * page_size
    if not num_pages:
        num_pages = 4 * bound(window, page_size, full, max_tokens, max_model_len, k) + 16
    return Harness(specs=specs, vocab=VOCAB, window=window, window_page_size=page_size, page_size=full,
                   num_pages=num_pages, grid=kw.pop("grid", page_size), max_tokens=max_tokens,
                   max_model_len=max_model_len, mtp_k=k, draft_kv=draft_kv, **kw)


def with_mtp(draw, w: dict, *, stops: bool = True) -> dict:
    """A window-mode workload with MTP on top: k drafts (most often 5, the assistant's default; 3, an MTP layer's
    default; up to 7, the verify window's limit), an acceptance rate, a drafter with or without KV, stop tokens (more
    than the runner knows, sometimes), and the arena at least the MTP startup bound."""
    k = draw(st.sampled_from([1, 2, 3, 3, 5, 5, 5, 7]))
    w = dict(w, mtp_k=k, accept_p=draw(st.sampled_from([0.0, 0.5, 0.8, 0.95, 1.0])),
             draft_kv=draw(st.sampled_from([False, False, True])))
    w["max_tokens"] = max(w["max_tokens"], 4 * (1 + k))
    need = bound(w["window"], w.get("window_page_size") or w["page_size"], w["page_size"], w["max_tokens"],
                 w["max_model_len"], k) + 1
    w["num_pages"] = max(w["num_pages"], need + draw(st.integers(0, need)))
    if stops:
        specs = []
        for s in w["specs"]:
            if draw(st.booleans()) and draw(st.booleans()):  # stop tokens from its own predicted output
                out = output_of(s)
                n = draw(st.sampled_from([1, 2, STOP_W + 1]))
                s.stop_ids = sorted({draw(st.sampled_from(out)) for _ in range(n)} | set(range(1, n)))[:n]
            specs.append(s)
        w["specs"] = specs
    return w


@st.composite
def window_mtp_workloads(draw):
    """`window_workloads.window_workloads` (windows from one key up, pools from the startup bound, shared prefixes,
    follow-up turns, hints, aborts, priorities, NaNs) with MTP: verify rows sliding their windows from a lower bound,
    rejected KV in both kinds of page, decode windows at commit ends, finishes and aborts with rows in flight."""
    return with_mtp(draw, draw(window_workloads()))


@st.composite
def arena_mtp_workloads(draw):
    """`window_workloads.arena_pressure` with MTP: the arena near its MTP startup bound, so verify rows' wider
    reservations, preemptions with rows in flight and window evictions compete for pages."""
    w = draw(arena_pressure())
    return with_mtp(draw, w, stops=False)


@st.composite
def arena_mtp_count_workloads(draw):
    """`arena_mtp_workloads` (Gemma 4's: a drafter without KV, one arena near its bound) with a draft count per step."""
    from .mtp_workloads import with_counts

    return with_counts(draw, draw(arena_mtp_workloads()))


@st.composite
def gemma_images_mtp_workloads(draw):
    """`window_workloads.gemma_image_workloads` (bidirectional spans, encoder budgets, hints inside spans) with MTP."""
    return with_mtp(draw, draw(gemma_image_workloads()), stops=False)


def run_window_mtp_workload(w: dict, bidirectional: bool = False) -> Harness:
    w = dict(w)
    vocab = 1000 if bidirectional else VOCAB
    h = Harness(vocab=vocab, bidirectional=bidirectional, **w)
    h.run(check_every_turn=True)
    check_plans(h)
    check_nan_invariants(h)
    h.check(failed_ok=True)
    s = h.sched
    nan_events(h)
    assert not any(p["fold"] for p in h.plans), "FOLD in window mode"
    for spec in h.specs.values():
        r = h.results[spec.rid]
        if spec.nonfinite_after is not None and r.finish not in (FinishReason.ABORT,):
            out = h.expected(spec)
            stops = [i for i, t in enumerate(out) if (t == 999 and not spec.ignore_eos) or t in spec.stop_ids]
            if spec.nonfinite_after < min(stops[0] + 1 if stops else len(out), spec.max_tokens):
                assert r.finish == FinishReason.ERROR, "the NaN request was not failed"
    if h.runner.counts["accepted"]:
        event("drafts accepted")
    if s.steps_by_count[1:s.k].sum():
        event("MTP: steps below k")
    if s.count_raises:
        event("MTP: a raise (k drafted for a higher count)")
    if s.prefix_hits:
        event("prefix hit")
    for src in ("decode", "finish", "branch"):
        if s.ckpts_taken[src]:
            event(f"{src} window")
    if s.num_preemptions:
        event("preempted")
    if s.cache is not None and s.cache.evicted_ckpts:
        event("window evicted")
    if s.num_overruns:
        event("a stop token past the runner's")
    if h.runner.counts["kind_switch"]:
        event("a page reused as the other kind")
    if not w.get("draft_kv", True):
        event("a drafter without KV")
    return h
