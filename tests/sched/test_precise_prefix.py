"""The precise prefix (architecture.md, "The precise prefix") through the scheduler and the validating fake.

A block-FP8 model's cached prompt prefix is computed once and every request that resumes from it inherits its
numerics; computed with FP8-quantized activations it carries one draw of their rounding noise into every answer. The
scheduler's `precise` policy marks the rows that compute such a prefix PRECISE (BF16 activations) and keeps their KV
and checkpoints in a precise namespace of the cache (`cache.radix.precise_ns`). The fake computes PRECISE rows like any
other (tokens do not depend on precision), so every output check still holds, and it tracks which states and KV
positions PRECISE rows computed from a precise start: a PRECISE row that continues anything else raises PlanError, and
`check_precise` checks that the precise namespace holds nothing else.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given

from llm_mini_infer.cache.pools import pages_for_one_request
from llm_mini_infer.cache.radix import precise_ns
from llm_mini_infer.engine.messages import CHECKPOINT_SOURCES, FinishReason
from llm_mini_infer.sched.plan import reserve_positions
from llm_mini_infer.sched.scheduler import (
    REFINE_MIN,
    RUNNING,
    Admit,
    Scheduler,
    derive_precise_cost,
    derive_refine_chunk_tokens,
)

from ..support.fuzzing import campaign_settings, saved_on_failure
from ..support.harness import EOS, Harness, Spec, check_precise
from ..support.precise_cases import (
    check_refine_plans,
    precise_rows,
    precise_workloads,
    run_precise_workload,
    watch_refines,
    window_precise_workloads,
)
from ..support.sched_cases import VOCAB, check_plans, output_of, shared, stem
from ..support.window_workloads import arena_bound
from ..support.window_workloads import harness as window_harness
from ..support.workloads import run_cache_workload


def run_checked(h: Harness) -> int:
    """Run to the end, checking the precise namespace after every loop turn; returns the precise nodes seen last."""
    seen = 0
    orig = h.core.step

    def step(*a, **kw):
        nonlocal seen
        r = orig(*a, **kw)
        seen = check_precise(h)
        return r

    h.core.step = step
    h.run(check_every_turn=True)
    return seen


def test_reuse_refines_a_shared_prefix_at_its_second_reuse() -> None:
    """A text-only shape, scaled down (a 710-token shared instruction, short unique items, one request at a time):
    request 0 computes the prompt on the FP8 path; its end-of-prefill checkpoint (704) lies on the shared part.
    Request 1 resumes from it (its first reuse: free); request 2 would be its second reuse, so it recomputes [0, 704)
    in PRECISE rows instead (one 704-token chunk ending exactly there) and then its own 26 tokens on the FP8 path;
    every later request resumes from the precise copy."""
    specs = [Spec(rid=i, prompt_len=0, prompt=shared(710, 20, i), max_tokens=4, seed=7000 + i, arrive=10 * i)
             for i in range(8)]
    h = Harness(specs=specs, vocab=VOCAB, max_num_seqs=2, max_tokens=2048, page_size=16, num_pages=400,
                checkpoint_slots=6, grid=64, max_model_len=1024, precise="reuse")
    assert run_checked(h) > 0
    check_plans(h)
    h.check()
    s = h.sched
    assert [h.results[i].cached for i in range(8)] == [0, 704, 0, 704, 704, 704, 704, 704]
    assert s.refines == 1 and s.precise_hits == 5 and s.ckpts_taken["refine"] == 1
    rows = precise_rows(h)
    assert [(p0, q) for _, p0, q in rows] == [(0, 704)]  # the refiner's one PRECISE row: [0, 704)
    assert s.precise_tokens == 704
    # the refiner recomputes the FP8 copy's 704 by policy; every later request the prefix's 6 tokens past 704
    assert s.lost_tokens == {"writing": 0, "checkpoint": 7 * 6, "page": 0, "precise": 704}


@pytest.mark.parametrize("n_prefix, refined", [(200, False), (710, True)])
def test_the_default_refines_long_prefixes_only(n_prefix: int, refined: bool) -> None:
    """The scheduler's default `refine_min` (256 tokens): a shared prefix whose FP8 checkpoint lies less than that above
    the deepest precise one (none here) is resumed from on the FP8 path however often it is reused (a refine's step runs
    eagerly and dequantizes every weight, whatever it recomputes); a longer one is refined at its second reuse."""
    specs = [Spec(rid=i, prompt_len=0, prompt=shared(n_prefix, 20, i), max_tokens=4, seed=7100 + i, arrive=10 * i)
             for i in range(5)]
    h = Harness(specs=specs, vocab=VOCAB, max_num_seqs=2, max_tokens=2048, page_size=16, num_pages=400,
                checkpoint_slots=6, grid=64, max_model_len=1024, precise="reuse", refine_min=REFINE_MIN)
    run_checked(h)
    check_plans(h)
    h.check()
    hit = n_prefix // 64 * 64  # the first request's end-of-prefill checkpoint, on the shared part
    assert [h.results[i].cached for i in range(5)] == [0, hit, *([0] if refined else [hit]), hit, hit]
    assert h.sched.refines == int(refined) and bool(precise_rows(h)) == refined


def test_a_shallow_fp8_checkpoint_above_the_precise_copy_is_passed_over() -> None:
    """A teacher-forced measurement's shape: a request whose prompt is the shared prefix, its two reuses (the second
    refines [0, 704)), then prompts that extend it. The first request's end left FP8 checkpoints past the refined
    position, on the prefix's FP8 copy; with the default `refine_min` none is worth a refine, and resuming from one
    would bring the FP8 copy's draw back, so every later prompt resumes from the precise copy at 704 and computes the
    rest on the FP8 path (without this rule the deeper FP8 checkpoints win every match and no prompt hits the precise
    copy)."""
    pre = shared(710, 0, 0)
    specs = [Spec(rid=i, prompt_len=0, prompt=pre, max_tokens=1, seed=7200 + i, arrive=10 * i) for i in range(3)]
    specs += [Spec(rid=i, prompt_len=0, prompt=shared(710, 20, i), max_tokens=4, seed=7200 + i, arrive=10 * i)
              for i in range(3, 7)]
    h = Harness(specs=specs, vocab=VOCAB, max_num_seqs=2, max_tokens=2048, page_size=16, num_pages=400,
                checkpoint_slots=6, grid=64, max_model_len=1024, precise="reuse", refine_min=REFINE_MIN)
    assert run_checked(h) > 0
    check_plans(h)
    h.check()
    assert [h.results[i].cached for i in range(7)] == [0, 704, 0, 704, 704, 704, 704]
    assert h.sched.refines == 1 and h.sched.precise_hits == 4


def test_an_fp8_checkpoint_on_the_precise_copy_is_used() -> None:
    """The other side of the rule above: an FP8 checkpoint a grid or more above the precise copy that a request
    standing on the precise copy wrote (`Node.on_precise`) carries no FP8 copy's draw, so later requests resume from it
    as they would without the policy (concurrent requests that share tokens past a shared instruction: passing such
    checkpoints over would make each of them recompute those tokens, which the rule accepts below a grid only:
    `test_a_checkpoint_less_than_a_grid_above_the_precise_copy_is_passed_over`). Requests 0-3 share a 710-token prefix
    (the third refines it); requests 4-6 share 100 more tokens, and request 4's end-of-prefill checkpoint (768) lies on
    them, one grid (64) above the copy."""
    ext = [(j * 11 + 5) % (VOCAB - 1) for j in range(100)]
    specs = [Spec(rid=i, prompt_len=0, prompt=shared(710, 20, i), max_tokens=4, seed=7300 + i, arrive=10 * i)
             for i in range(4)]
    specs += [Spec(rid=i, prompt_len=0, prompt=stem(0, 710) + ext + [(i * 13 + j * 7 + 1) % (VOCAB - 1)
                                                                      for j in range(20)],
                   max_tokens=4, seed=7300 + i, arrive=10 * i) for i in range(4, 7)]
    h = Harness(specs=specs, vocab=VOCAB, max_num_seqs=2, max_tokens=2048, page_size=16, num_pages=400,
                checkpoint_slots=8, grid=64, max_model_len=1024, precise="reuse", refine_min=REFINE_MIN)
    assert run_checked(h) > 0
    check_plans(h)
    h.check()
    assert [h.results[i].cached for i in range(7)] == [0, 704, 0, 704, 704, 768, 768]
    assert h.sched.refines == 1


def test_an_fp8_checkpoint_from_the_fp8_copy_is_passed_over_a_grid_above_the_precise_copy() -> None:
    """The test above with the checkpoint at 768 written on the prefix's FP8 copy: request 0 runs first, cold, and its
    end-of-prefill checkpoint lies on the 810 tokens that requests 4-6 share. Requests 1-3 share the 710-token prefix
    (the first computes it, the third refines [0, 704)). A grid above the precise copy and less than `refine_min`, the
    checkpoint carries the FP8 copy's draw, so requests 4-6 pass it over and resume from the copy (the `short` rule's
    clause on `Node.on_precise`, which alone holds from a grid above the copy)."""
    pre = stem(0, 710) + [(j * 11 + 5) % (VOCAB - 1) for j in range(100)]

    def tail(i: int) -> list[int]:
        return [(i * 13 + j * 7 + 1) % (VOCAB - 1) for j in range(20)]

    specs = [Spec(rid=0, prompt_len=0, prompt=pre + tail(0), max_tokens=3, seed=7400, arrive=0)]
    specs += [Spec(rid=i, prompt_len=0, prompt=shared(710, 20, i), max_tokens=3, seed=7400 + i, arrive=10 * i)
              for i in range(1, 4)]
    specs += [Spec(rid=i, prompt_len=0, prompt=pre + tail(i), max_tokens=3, seed=7400 + i, arrive=10 * i)
              for i in range(4, 7)]
    h = Harness(specs=specs, vocab=VOCAB, max_num_seqs=2, max_tokens=2048, page_size=16, num_pages=400,
                checkpoint_slots=8, grid=64, max_model_len=1024, precise="reuse", refine_min=REFINE_MIN)
    assert run_checked(h) > 0
    check_plans(h)
    h.check()
    assert [h.results[i].cached for i in range(7)] == [0, 0, 704, 0, 704, 704, 704]
    assert h.sched.refines == 1


def test_requests_wait_for_a_checkpoint_a_grid_above_the_precise_copy() -> None:
    """`test_an_fp8_checkpoint_on_the_precise_copy_is_used` with requests 4-6 arriving together: request 4 resumes from
    the copy and writes the checkpoint at 768, a grid above it, which is pending for the other two in that plan. They
    wait for it (`Scheduler._admit_one`; one wait each) and resume from it, as they would from any checkpoint they do
    not pass over. That checkpoint, written by the plan below their branch point, would also hold them alone
    (`Scheduler._branch_wait`); in `restamp_harness` an earlier step wrote it."""
    ext = [(j * 11 + 5) % (VOCAB - 1) for j in range(100)]
    specs = [Spec(rid=i, prompt_len=0, prompt=shared(710, 20, i), max_tokens=4, seed=7300 + i, arrive=10 * i)
             for i in range(4)]
    specs += [Spec(rid=i, prompt_len=0, prompt=stem(0, 710) + ext + [(i * 13 + j * 7 + 1) % (VOCAB - 1)
                                                                      for j in range(20)],
                   max_tokens=4, seed=7300 + i, arrive=100, hints=[768] if i == 4 else []) for i in range(4, 7)]
    h = Harness(specs=specs, vocab=VOCAB, max_num_seqs=4, max_tokens=2048, page_size=16, num_pages=400,
                checkpoint_slots=8, grid=64, max_model_len=1024, precise="reuse", refine_min=REFINE_MIN)
    assert run_checked(h) > 0
    check_plans(h)
    h.check()
    assert [h.results[i].cached for i in range(7)] == [0, 704, 0, 704, 704, 768, 768]
    assert h.sched.refines == 1 and h.sched.cache_waits == 2


def restamp_harness(end: int, hints: list[int], precise: str, refine_min: int) -> Harness:
    """Requests 0-3 of `test_an_fp8_checkpoint_on_the_precise_copy_is_used` (a precise copy at 704); request 4 adds 100
    tokens past the prefix, resumes from the copy, writes FP8 checkpoints on it at `hints` (or its end-of-prefill one,
    768) and leaves. Then, in one turn,
    request 5, whose prompt is the first `end` of request 4's tokens, and requests 6 and 7, which extend them. A hit
    ends before a prompt's last token, so request 5 resumes below the checkpoint at `end` and computes the span up to
    it; its insert swaps the node's pages, which the tree alone holds, for its own, which stamps the node as written by
    its step (`RadixCache._adopt`). The checkpoint is then pending for requests 6 and 7 (`RadixCache.match`); an
    earlier step wrote the checkpoint itself, so `Scheduler._branch_wait` does not hold them."""
    pre = stem(0, 710) + [(j * 11 + 5) % (VOCAB - 1) for j in range(100)]

    def tail(i: int) -> list[int]:
        return [(i * 13 + j * 7 + 1) % (VOCAB - 1) for j in range(20)]

    specs = [Spec(rid=i, prompt_len=0, prompt=shared(710, 20, i), max_tokens=4, seed=7300 + i, arrive=10 * i)
             for i in range(4)]
    specs.append(Spec(rid=4, prompt_len=0, prompt=pre + tail(4), max_tokens=4, seed=7304, arrive=40, hints=hints))
    specs.append(Spec(rid=5, prompt_len=0, prompt=pre[:end], max_tokens=4, seed=7305, arrive=100))
    specs += [Spec(rid=i, prompt_len=0, prompt=pre[:end] + tail(i), max_tokens=4, seed=7300 + i, arrive=100)
              for i in (6, 7)]
    return Harness(specs=specs, vocab=VOCAB, max_num_seqs=4, max_tokens=2048, page_size=16, num_pages=400,
                   checkpoint_slots=8, grid=64, max_model_len=1024, precise=precise, refine_min=refine_min)


@pytest.mark.parametrize("end, hints, precise, refine_min, cached, refines", [
    (768, [], "reuse", REFINE_MIN, [704, 768, 768], 1),  # a grid above the copy
    (800, [768, 800], "reuse", REFINE_MIN, [768, 800, 800], 1),  # 96 above the copy, 32 above request 5's hit
    (744, [744], "reuse", 32, [704, 744, 704], 2),  # 40 above it, `refine_min` below the grid: the second reuse refines
    (744, [744], "bf16", REFINE_MIN, [704, 744, 744], 0),  # no `short` rule
])
def test_requests_wait_for_a_checkpoint_the_short_rule_uses(end: int, hints: list[int], precise: str,
                                                           refine_min: int, cached: list[int], refines: int) -> None:
    """A pending FP8 checkpoint that the `short` rule does not pass over once visible (`restamp_harness`): requests 6
    and 7 wait for it (one wait each), then resume from it or, its second reuse past `refine_min`, refine it. They
    wait unless it lies less than the smaller of a grid and `refine_min` above the precise copy (`Scheduler._admit_one`;
    `test_requests_do_not_wait_for_a_checkpoint_less_than_a_grid_above_the_precise_copy`), counted from the copy, not
    from the FP8 hit, and only under `reuse`."""
    h = restamp_harness(end, hints, precise, refine_min)
    assert run_checked(h) > 0
    check_plans(h)
    h.check()
    assert [h.results[i].cached for i in range(5, 8)] == cached
    assert h.sched.cache_waits == 2 and h.sched.refines == refines


def test_requests_do_not_wait_for_a_restamped_checkpoint_less_than_a_grid_above_the_precise_copy() -> None:
    """The burst of `test_requests_do_not_wait_for_a_checkpoint_less_than_a_grid_above_the_precise_copy` with state
    checkpoints (`restamp_harness`): the checkpoint at 744 lies 40 above the copy, less than a grid, so requests 6 and 7
    do not wait for it and resume from the copy, as the `short` rule would have them do a step later."""
    h = restamp_harness(744, [744], "reuse", REFINE_MIN)
    assert run_checked(h) > 0
    check_plans(h)
    h.check()
    assert [h.results[i].cached for i in range(5, 8)] == [704, 704, 704]
    assert h.sched.cache_waits == 0 and h.sched.refines == 1


def test_requests_wait_for_a_shallow_checkpoint_without_a_precise_copy() -> None:
    """Without a precise copy below it nothing is passed over, so requests wait for a pending FP8 checkpoint however
    shallow: request 0 leaves one at 40 (a hint), less than a grid deep, and leaves; request 1, whose prompt is those 40
    tokens, computes them again and stamps the node (as in `restamp_harness`); requests 2 and 3 wait and resume from
    it."""
    pre = stem(0, 40)
    specs = [Spec(rid=0, prompt_len=0, prompt=shared(40, 20, 0), max_tokens=4, seed=7600, arrive=0, hints=[40]),
             Spec(rid=1, prompt_len=0, prompt=pre, max_tokens=4, seed=7601, arrive=30)]
    specs += [Spec(rid=i, prompt_len=0, prompt=shared(40, 20, i), max_tokens=4, seed=7600 + i, arrive=30)
              for i in (2, 3)]
    h = Harness(specs=specs, vocab=VOCAB, max_num_seqs=4, max_tokens=2048, page_size=16, num_pages=400,
                checkpoint_slots=8, grid=64, max_model_len=1024, precise="reuse", refine_min=REFINE_MIN)
    run_checked(h)
    check_plans(h)
    h.check()
    assert [h.results[i].cached for i in range(4)] == [0, 0, 40, 40]
    assert h.sched.refines == 0 and h.sched.cache_waits == 2


def near_copy_harness(burst: int = 0, near: int = 301) -> Harness:
    """Window mode, grid 8: requests 0-4 share a 300-token prefix and the fourth refines [0, 296) (as in
    `test_window_mode_refines_with_its_window`). Request 5 resumes from the precise copy and leaves a hint's window at
    `near` on it (`Node.on_precise`), by default 301, five tokens above it; requests 6 and 7 share those tokens, and
    `burst` more arrive together at turn 200, when every earlier request has left."""
    specs = [Spec(rid=i, prompt_len=0, prompt=shared(300, 12, i), max_tokens=3, seed=3000 + i, arrive=10 * i)
             for i in range(5)]
    specs += [Spec(rid=i, prompt_len=0, prompt=shared(near, 12, i), max_tokens=3, seed=3000 + i, arrive=10 * i,
                   hints=[near] if i == 5 else []) for i in range(5, 8)]
    specs += [Spec(rid=i, prompt_len=0, prompt=shared(near, 12, i), max_tokens=3, seed=3000 + i, arrive=200)
              for i in range(8, 8 + burst)]
    return window_harness(specs, window=32, page_size=8, max_tokens=512, max_model_len=640, precise="reuse",
                          refine_min=24)


def test_a_checkpoint_less_than_a_grid_above_the_precise_copy_is_passed_over() -> None:
    """An FP8 checkpoint written on the precise copy but less than a checkpoint grid above it (`near_copy_harness`:
    five tokens): requests 6 and 7 resume from the precise copy below it and compute the five tokens on the FP8 path,
    so every admission locks the copy. Resuming from the checkpoint lets the copy's lock drain while requests still
    use the prefix, and under pressure an eviction pass that spares locked windows takes it: the next admission then
    refines a second copy (`tests/sched/test_gemma_spans.py`, the one-token chunk before an image). A checkpoint a whole
    grid above the copy is still used (`test_an_fp8_checkpoint_on_the_precise_copy_is_used`)."""
    h = near_copy_harness()
    assert run_checked(h) > 0
    check_plans(h)
    h.check()
    s = h.sched
    assert s.refines == 1
    assert [h.results[i].cached for i in range(8)] == [0, 0, 296, 0, 296, 296, 296, 296]
    assert s.lost_tokens["precise"] == 296 + 2 * 5  # the refine's, and the five tokens of requests 6 and 7


@pytest.mark.parametrize("near", [301, 303])
def test_requests_do_not_wait_for_a_checkpoint_less_than_a_grid_above_the_precise_copy(near: int) -> None:
    """The checkpoint above (`near_copy_harness`), five or seven tokens (a grid less one) above the copy, and four
    requests that arrive in one turn, once the tree alone holds its node's tail page: the first admission resumes from
    the precise copy below the checkpoint, and its insert swaps that page for its own, which stamps the node as written
    by its step, so the checkpoint is pending for the other three (`RadixCache.match`). They would pass it over a step
    later (`Scheduler._admit_one`, `short`), so they do not wait for it: no cache waits, each costing a step of TTFT,
    and every request resumes from the copy at 296."""
    h = near_copy_harness(burst=4, near=near)
    assert run_checked(h) > 0
    check_plans(h)
    h.check()
    s = h.sched
    assert s.refines == 1 and s.cache_waits == 0
    assert [h.results[i].cached for i in range(12)] == [0, 0, 296, 0, 296, *[296] * 7]
    assert s.lost_tokens["writing"] == 3 * (near - 296)  # the tokens the first of the four was writing, for the others


def test_a_refine_and_the_eviction_of_a_served_refined_window_are_logged(caplog) -> None:
    """One INFO line per refine (its PRECISE rows and pieces), and a WARNING when a refined window that served resumes
    is evicted (here by an eviction of every checkpoint after `near_copy_harness`'s run); `cache.clear`, which
    `Harness.check` calls, evicts the same window without a line."""
    caplog.set_level(logging.INFO, logger="llm_mini_infer")

    def lines(level: int, text: str = "") -> list[str]:
        return [r.getMessage() for r in caplog.records if r.levelno == level and text in r.getMessage()]

    h = near_copy_harness()
    run_checked(h)
    s = h.sched
    piece = min(s.precise_cap or s.piece, s.max_tokens)
    n = -(-296 // piece)
    assert lines(logging.INFO, "refines a shared prefix") == [
        f"request 0:3 refines a shared prefix: PRECISE rows [0, 296) in {n} piece{'s' if n > 1 else ''} of up to "
        f"{piece} tokens"]
    assert s.refines == 1 and not lines(logging.WARNING)
    s.cache.evict_checkpoints(1 << 62, s.step_id + 1)
    warned = lines(logging.WARNING)
    assert len(warned) == 1 and "served 4 resumes" in warned[0] and "position 296" in warned[0], warned  # 4-7
    h.check()
    assert lines(logging.WARNING) == warned
    caplog.clear()
    h = near_copy_harness()
    run_checked(h)
    h.check()
    assert h.sched.cache.evicted_by[CHECKPOINT_SOURCES.index("refine")] == [0, 1] and not lines(logging.WARNING)


@pytest.mark.parametrize("precise", ["reuse", "bf16"])
def test_other_precise_windows_are_evicted_without_a_line(caplog, precise: str) -> None:
    """The WARNING is for a refined window that served resumes alone. Requests 0-3 of `near_copy_harness`: under
    `reuse` the fourth refines [0, 296) and no request resumes from the copy; under `bf16` every cold prompt's windows
    are precise, and the branch window at 296 serves the third and the fourth request. Evicting every checkpoint takes
    both windows without a line."""
    caplog.set_level(logging.INFO, logger="llm_mini_infer")
    specs = [Spec(rid=i, prompt_len=0, prompt=shared(300, 12, i), max_tokens=3, seed=3000 + i, arrive=10 * i)
             for i in range(4)]
    h = window_harness(specs, window=32, page_size=8, max_tokens=512, max_model_len=640, precise=precise,
                       refine_min=24)
    run_checked(h)
    c = h.sched.cache
    before = [list(e) for e in c.evicted_by]
    c.evict_checkpoints(1 << 62, h.sched.step_id + 1)
    gone = {s: [c.evicted_by[i][j] - before[i][j] for j in (0, 1)] for i, s in enumerate(CHECKPOINT_SOURCES)}
    assert gone["refine"] == ([1, 0] if precise == "reuse" else [0, 0]) and gone["branch"] == [0, 1], gone
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]
    h.check()


def test_a_refine_in_several_steps_is_not_duplicated() -> None:
    """PRECISE chunks of 256 tokens (the harness's cap: a refine of several pieces) refine [0, 704) in three steps,
    and only the last chunk takes a checkpoint, so before it nothing in the precise namespace is pending: requests
    that arrive meanwhile with the same prefix, with budget to spare in those steps, wait (the in-flight refine,
    `_refining_now`) instead of starting a second refine of the same span (a convoy of refines), and then resume from
    the precise copy."""
    specs = [Spec(rid=i, prompt_len=0, prompt=shared(710, 20, i), max_tokens=4, seed=7400 + i, arrive=12 * i)
             for i in range(3)]
    specs += [Spec(rid=i, prompt_len=0, prompt=shared(710, 20, i), max_tokens=4, seed=7400 + i, arrive=25 + i % 2)
              for i in range(3, 7)]
    h = Harness(specs=specs, vocab=VOCAB, max_num_seqs=6, max_tokens=2048, page_size=16, num_pages=600,
                checkpoint_slots=8, grid=64, max_model_len=1024, precise="reuse", refine_min=REFINE_MIN)
    h.sched.precise_cap = 256
    assert run_checked(h) > 0
    check_plans(h)
    h.check()
    assert h.sched.refines == 1 and h.sched.precise_tokens == 704
    assert [h.results[i].cached for i in range(3, 7)] == [704] * 4 and h.sched.cache_waits >= 4


def test_window_refine_in_several_steps_is_not_duplicated() -> None:
    """The test above in window mode (Gemma 4: windows as checkpoints): PRECISE chunks of 128 tokens refine [0, 592)
    in five steps, and the requests that arrive meanwhile wait for the refine in flight instead of refining the same
    span again (4 refines and 2,368 PRECISE tokens without `_refining_now`)."""
    specs = [Spec(rid=i, prompt_len=0, prompt=shared(600, 20, i), max_tokens=4, seed=7500 + i, arrive=12 * i)
             for i in range(3)]
    specs += [Spec(rid=i, prompt_len=0, prompt=shared(600, 20, i), max_tokens=4, seed=7500 + i, arrive=25 + i % 2)
              for i in range(3, 7)]
    h = window_harness(specs, window=64, page_size=16, max_tokens=1024, max_model_len=1024, max_num_seqs=6,
                       precise="reuse", refine_min=REFINE_MIN)
    h.sched.precise_cap = 128
    assert run_checked(h) > 0
    check_plans(h)
    h.check()
    assert h.sched.refines == 1 and h.sched.precise_tokens == 592
    assert [h.results[i].cached for i in range(3, 7)] == [592, 0, 592, 592]  # request 4 (first to arrive) refines


@pytest.mark.parametrize("cap", [0, 128])
def test_the_scheduler_keeps_only_the_refines_in_progress(cap: int) -> None:
    """30 shared prefixes of 300 tokens, each reused by 4 requests in turn: each is refined once, by its third request,
    in one step or in two (PRECISE chunks of 128). The scheduler needs a refiner (`_refining_now`) only while its
    PRECISE rows have not reached their target: after every step each entry it keeps is such a refine, and none is
    left at the end. An entry dropped only when `_refining_now` is asked about the same checkpoint again would stay for
    the server's lifetime, since no request asks once the precise copy wins every match: one entry, and a finished
    request, per refine."""
    specs = [Spec(rid=i, prompt_len=0, prompt=stem(i // 4, 300) + stem(40 + i % 4, 6), max_tokens=4, seed=8000 + i,
                  arrive=8 * i) for i in range(120)]
    h = Harness(specs=specs, vocab=VOCAB, max_num_seqs=2, max_tokens=512, page_size=16, num_pages=400,
                checkpoint_slots=8, grid=64, max_model_len=512, precise="reuse", refine_min=REFINE_MIN)
    s, orig = h.sched, h.core.step
    s.precise_cap = cap

    def step(*a, **kw):
        out = orig(*a, **kw)
        for r in s._refining.values():
            assert r.status == RUNNING and r.slot and s.slot_req[r.slot] is r and r.pend < r.refine, r.rid
        return out

    h.core.step = step
    h.run(check_every_turn=True)
    assert s.refines == 30 and not s._refining
    assert [h.results[4 * k + 3].cached for k in range(30)] == [256] * 30  # the fourth: the precise copy
    h.check()


def path_pages(h: Harness, ns, key: list[int], upto: int) -> list[int]:
    """The tree's KV pages of positions [0, upto) along `key` in namespace `ns` (upto page-aligned)."""
    cache = h.sched.cache
    m = cache.match(ns, np.asarray(key, np.int32), 1 << 30, len(key))
    out: list[int] = []
    for node in m.path:
        lo, hi = node.start // cache.ps, min(node.end, upto) // cache.ps
        if hi > lo:
            out.extend(node.pages[:hi - lo])
    return out[:upto // cache.ps]


def test_fp8_continuations_read_the_precise_copys_pages() -> None:
    """The requests that resume from the precise copy keep their own FP8 continuations in the FP8 namespace, under the
    shared prefix's FP8 copy; the first one's insert adopts its pages for the shared part there (`RadixCache.insert`:
    the tree swaps in a request's pages wherever it alone holds its own), and those are the precise copy's. So a later
    hit on an FP8-namespace continuation (a conversation's next turn, a deeper branch checkpoint) reads the shared
    prefix's precise KV, and the FP8 copy's pages go back to the pool: only what a request computed itself carries FP8
    noise."""
    specs = [Spec(rid=i, prompt_len=0, prompt=shared(710, 20, i), max_tokens=4, seed=7000 + i, arrive=10 * i)
             for i in range(5)]
    h = Harness(specs=specs, vocab=VOCAB, max_num_seqs=2, max_tokens=2048, page_size=16, num_pages=400,
                checkpoint_slots=6, grid=64, max_model_len=1024, precise="reuse")
    run_checked(h)
    assert [h.results[i].cached for i in range(5)] == [0, 704, 0, 704, 704] and h.sched.refines == 1
    key = shared(710, 20, 4)
    fp8, pre = path_pages(h, None, key, 704), path_pages(h, precise_ns(None), key, 704)
    assert len(pre) == 44 and fp8 == pre
    assert h.runner.kv_prec[pre].all()  # written by PRECISE rows
    h.check()  # (clears the cache)


@pytest.mark.parametrize("refine_after", [0, 1])
def test_a_conversation_never_refines_its_history(refine_after: int) -> None:
    """Four turns of one conversation: each resumes from the previous turn's finish checkpoint, once. With the
    default (`refine_after` 1) nothing is recomputed. Refining at the first reuse (0) makes every later turn recompute
    the previous turn (its message and answer, from the precise copy the turn before left) in PRECISE rows first: the
    precise copy grows with the conversation, one turn's worth of BF16 prefill per turn (the reason for the default)."""
    turns: list[Spec] = []
    prompt = stem(2, 37)
    for t in range(4):
        spec = Spec(rid=t, prompt_len=0, prompt=list(prompt), max_tokens=12, seed=11 + t, arrive=40 * t,
                    ignore_eos=True)
        turns.append(spec)
        prompt = prompt + output_of(spec) + [5 + t, 6 + t, 7 + t]
    h = Harness(specs=turns, vocab=VOCAB, max_tokens=256, page_size=4, num_pages=200, checkpoint_slots=6, grid=16,
                precise="reuse", refine_after=refine_after)
    run_checked(h)
    h.check()
    s = h.sched
    if refine_after:
        assert s.refines == 0 and s.precise_tokens == 0
        assert all(h.results[t].cached == len(turns[t - 1].prompt) + 12 - 1 for t in range(1, 4))
    else:
        assert s.refines == 3  # every later turn recomputes the previous one in PRECISE rows
        finish = [len(turns[t].prompt) + 12 - 1 for t in range(3)]  # where each turn stopped computing
        assert [p0 for _, p0, _ in precise_rows(h)] == [0, finish[0], finish[1]]
        assert s.precise_tokens == finish[2]


def test_bf16_mode_marks_every_chunk_of_a_cold_prompt() -> None:
    """`bf16` mode: a 300-token prompt in 128-token steps is PRECISE in each of its chunks, not only the first; a later
    identical prompt resumes from its precise end-of-prefill checkpoint and runs its last token on the FP8 path."""
    a = Spec(rid=0, prompt_len=0, prompt=stem(3, 300), max_tokens=3, seed=21)
    b = Spec(rid=1, prompt_len=0, prompt=stem(3, 300), max_tokens=3, seed=22, arrive=20)
    h = Harness(specs=[a, b], vocab=VOCAB, max_num_seqs=2, max_tokens=128, page_size=4, num_pages=300,
                checkpoint_slots=4, grid=16, max_model_len=512, precise="bf16")
    run_checked(h)
    h.check()
    assert [(p0, q) for _, p0, q in precise_rows(h)] == [(0, 128), (128, 128), (256, 44)]
    assert h.results[1].cached == 288 and h.sched.precise_hits == 1 and h.sched.refines == 0


def test_a_verified_request_gives_the_precise_copy_nothing_past_its_hit() -> None:
    """Attention-only `bf16` mode, pages of 2, no NaN (a case a targeted fuzz found). R1's cold prompt
    [22, 53, 84, 115] is PRECISE (the precise node [0, 4)), and its finish puts its pages into the FP8 namespace too. R0
    shares 3 of its tokens and resumes from the precise copy at 2 (a tie goes to it); in attention-only mode its precise
    lock is the node holding position 1, [0, 4), which runs past the hit and past R0's own token at 3. R0's chunk [2, 6)
    is unverified, so its FP8 insert leaves the FP8 namespace's page of index 1 (R1's) in place; R0's first sample
    verifies its pages, and the tree takes them along its locks, along the precise one only what R0 computed precise.
    Taken to that lock's end, [0, 4) would get R0's FP8 page of index 1 (R0's token 46 at 3 instead of 115), and R3
    (R1's prompt and one token more), resuming from the precise copy at 4, would sample from it."""
    specs = [Spec(rid=1, prompt_len=0, prompt=[22, 53, 84, 115], max_tokens=2, seed=701),
             Spec(rid=3, prompt_len=0, prompt=[22, 53, 84, 115, 820], max_tokens=1, seed=703, priority=1),
             Spec(rid=0, prompt_len=0, prompt=[22, 53, 84, 46, 100, 422, 347], max_tokens=1, seed=700, arrive=2)]
    h = Harness(specs=specs, vocab=VOCAB, page_size=2, grid=8, hybrid=False, precise="bf16", max_num_seqs=1,
                chunk_cap=4, policy="priority", num_pages=40, max_tokens=64, max_model_len=160)
    run_checked(h)
    assert [h.results[i].cached for i in (0, 3)] == [2, 4] and h.sched.precise_hits == 2
    h.check()


def test_readoption_under_pressure_gives_the_precise_copy_nothing_past_a_hit() -> None:
    """The test above through `_readopt` (page pressure), with nothing unverified.
    X's 6-token prompt runs in one chunk from its precise hit at 2 (its precise lock [0, 4) again) while Y, resumed from
    the precise copy at 4, holds R1's page of index 1: X's FP8 insert leaves that page in place, a duplicate of X's own
    that X's lock pins once Y leaves. Y leaves; Z's cold prompt makes the pool short while X decodes, and the scheduler
    frees duplicates by readopting the running requests' pages along their locks, along X's precise one only what X
    computed precise. Taken to its end, [0, 4) would get X's page of index 1, and R3 would resume from it."""
    a = [22, 53, 84, 115]
    specs = [Spec(rid=1, prompt_len=0, prompt=a, max_tokens=2, seed=701),
             Spec(rid=2, prompt_len=0, prompt=[*a, 7, 20], max_tokens=2, seed=702, arrive=3, ignore_eos=True),
             Spec(rid=0, prompt_len=0, prompt=[22, 53, 84, 46, 11, 48], max_tokens=12, seed=700, arrive=4,
                  ignore_eos=True),
             Spec(rid=4, prompt_len=0, prompt=stem(9, 12), max_tokens=8, seed=704, arrive=6, ignore_eos=True),
             Spec(rid=3, prompt_len=0, prompt=[*a, 820], max_tokens=1, seed=703, arrive=40)]
    h = Harness(specs=specs, vocab=VOCAB, page_size=2, grid=8, hybrid=False, precise="bf16", max_num_seqs=3,
                num_pages=pages_for_one_request(32, 2, reserve_positions(2, 0), True) + 1, max_tokens=64,
                max_model_len=32)
    s, readopted = h.sched, []
    readopt = s._readopt

    def record(force: bool = False) -> None:
        readopted.append([r.rid for r in s.slot_req if r is not None and r.dup])
        readopt(force)

    s._readopt = record
    run_checked(h)
    assert [h.results[i].cached for i in (2, 0)] == [4, 2] and [0] in readopted
    h.check()


def test_a_long_shared_prefix_is_refined_across_chunks() -> None:
    """A 700-token shared prefix with 256-token steps: the first request's end-of-prefill checkpoint (704) lies past the
    shared part, so the second recomputes from 0 and checkpoints the branch point (640); the third resumes there; the
    fourth refines [0, 640) in PRECISE chunks (the last ending exactly at 640), and the rest hit the precise copy at
    640. The chunks are pieces of 192 from position 0 (256 - 2 rows x 1 on the 64-token grid: a piece fits next to a
    full batch of decode rows)."""
    specs = [Spec(rid=i, prompt_len=0, prompt=shared(700, 20, i), max_tokens=4, seed=7100 + i, arrive=12 * i)
             for i in range(6)]
    h = Harness(specs=specs, vocab=VOCAB, max_num_seqs=2, max_tokens=256, page_size=16, num_pages=400,
                checkpoint_slots=6, grid=64, max_model_len=1024, precise="reuse")
    run_checked(h)
    h.check()
    assert [h.results[i].cached for i in range(6)] == [0, 0, 640, 0, 640, 640]
    assert [(p0, q) for _, p0, q in precise_rows(h)] == [(0, 192), (192, 192), (384, 192), (576, 64)]
    assert h.sched.refines == 1


def test_siblings_wait_for_the_refined_copy() -> None:
    """Six requests sharing a 100-token prefix arrive together. The first computes it on the FP8 path, with a branch
    checkpoint at 96 for the others, which wait one step. In the next plan the second resumes from 96 (its first
    reuse), the third refines [0, 96) (a PRECISE row), and the other three wait once more, for the precise copy that
    plan writes, and then resume from it."""
    specs = [Spec(rid=i, prompt_len=0, prompt=shared(100, 10, i), max_tokens=3, seed=1000 + i) for i in range(6)]
    h = Harness(specs=specs, vocab=VOCAB, max_num_seqs=6, max_tokens=512, page_size=4, num_pages=300,
                checkpoint_slots=8, grid=16, max_model_len=256, precise="reuse")
    run_checked(h)
    h.check()
    s = h.sched
    assert [h.results[i].cached for i in range(6)] == [0, 96, 0, 96, 96, 96]
    assert s.refines == 1 and s.precise_hits == 3 and s.cache_waits == 5 + 3
    assert [(p0, q) for _, p0, q in precise_rows(h)] == [(0, 96)]


def test_a_refiner_preempted_mid_refine_resumes_from_its_precise_state(caplog) -> None:
    """The refiner (priority 1) is preempted after its first PRECISE chunk, its slot taken by a higher-priority
    request: it donates its live state into the precise namespace (all it computed is precise), and its readmission
    resumes there and refines the rest. Both admissions log the refine and count it: one copy."""
    caplog.set_level(logging.INFO, logger="llm_mini_infer.sched.scheduler")
    # request 3 refines [0, 640) in pieces of 256 (the harness's cap): preempt it after its first piece
    first = 36  # request 3's arrival turn: admitted with its first PRECISE chunk in this turn's plan
    specs = [Spec(rid=i, prompt_len=0, prompt=shared(700, 20, i), max_tokens=4, seed=7200 + i, arrive=12 * i,
                  priority=1, ignore_eos=True, preempt_at=first + 1 if i == 3 else None) for i in range(4)]
    specs.append(Spec(rid=9, prompt_len=0, prompt=stem(9, 40), max_tokens=3, seed=99, arrive=first + 1, priority=0))
    h = Harness(specs=specs, vocab=VOCAB, max_num_seqs=1, max_tokens=256, page_size=16, num_pages=400,
                checkpoint_slots=6, grid=64, max_model_len=1024, policy="priority", precise="reuse", precise_cap=256)
    run_checked(h)
    h.check()
    assert [(p0, q) for _, p0, q in precise_rows(h)] == [(0, 256), (256, 256), (512, 128)]
    assert h.sched.num_preemptions == 1 and h.sched.ckpts_taken["refine"] == 1 and h.sched.refines == 2
    # the readmission resumed from the state it donated at 256, into the precise namespace
    assert any(p["seq_len"] == [256] and p["precise"] == [True] for p in h.plans)
    assert [r.getMessage() for r in caplog.records if "refines a shared prefix" in r.getMessage()] == [
        "request 0:3 refines a shared prefix: PRECISE rows [0, 640) in 3 pieces of up to 256 tokens",
        "request 0:3 refines a shared prefix: PRECISE rows [256, 640) in 2 pieces of up to 256 tokens"]


def test_window_mode_refines_with_its_window() -> None:
    """Sliding-window models (Gemma 4): the refiner's PRECISE rows compute the window too, and the precise copy's
    checkpoint is the refiner's own window at the refine target; later requests resume from it. The first request's
    end-of-prefill window (304) lies past the 300 shared tokens, so the second takes the branch window (296), the third
    reuses it and the fourth refines it."""
    specs = [Spec(rid=i, prompt_len=0, prompt=shared(300, 12, i), max_tokens=3, seed=3000 + i, arrive=10 * i)
             for i in range(6)]
    h = window_harness(specs, window=32, page_size=8, max_tokens=512, max_model_len=640, precise="reuse")
    assert run_checked(h) > 0
    check_plans(h)
    h.check()
    s = h.sched
    assert [h.results[i].cached for i in range(6)] == [0, 0, 296, 0, 296, 296]
    assert s.refines == 1 and [(p0, q) for _, p0, q in precise_rows(h)] == [(0, 296)]


def test_an_aged_arena_keeps_the_refined_prefix() -> None:
    """An aged server: 400 requests share a 300-token prefix, each with a unique tail, a few at a time, through an
    arena of twice one request's bound. Every finished request leaves a leaf (a full page, never hit again) and two
    windows, so the arena fills. The windows of unlocked nodes and those leaves go before the windows that running
    requests' paths hold, so the shared prefix keeps its precise window: one refine, and no request after the first
    third misses the prefix. With every window evicted before any leaf, locked or not, this run refines the prefix 8
    times, and 14 of the later requests compute it from scratch."""
    n = 400
    specs = [Spec(rid=i, prompt_len=0, prompt=shared(300, 12, i), max_tokens=6, seed=5000 + i, arrive=i)
             for i in range(n)]
    h = window_harness(specs, window=32, page_size=8, full=40, max_tokens=512, max_model_len=640,
                       num_pages=2 * arena_bound(32, 8, 40, 512, 640), precise="reuse")
    evicted = []
    orig = h.core.step

    def step(*a, **kw):
        r = orig(*a, **kw)
        evicted.append(h.sched.cache.evicted_pages)
        return r

    h.core.step = step
    h.run()
    check_plans(h)
    h.check()
    assert max(evicted) > 0  # the arena filled: leaves were evicted while requests ran
    assert h.sched.refines == 1
    assert all(h.results[i].cached == 296 for i in range(n // 3, n))


def test_precise_tokens_weigh_more_in_the_mixed_step_bound() -> None:
    """With decode rows in a step, its prefill gets K / rows tokens; a PRECISE token counts `precise_cost` of them: next
    to one decode row and K = 200 at a cost of 2, a cold prompt's PRECISE chunk (`bf16` mode) takes at most 100
    tokens, an FP8 chunk 200 (a refine's pieces pass the bound whole:
    `test_fixed_pieces_pass_the_mixed_step_bound_but_fp8_chunks_keep_it`)."""
    long = Spec(rid=0, prompt_len=0, prompt=stem(7, 20), max_tokens=200, seed=5, ignore_eos=True)
    specs = [long] + [Spec(rid=i, prompt_len=0, prompt=shared(400, 8, i), max_tokens=2, seed=4000 + i,
                           arrive=5 + 30 * i) for i in range(1, 5)]
    h = Harness(specs=specs, vocab=VOCAB, max_num_seqs=2, max_tokens=512, page_size=16, num_pages=400,
                checkpoint_slots=6, grid=16, max_model_len=1024, row_tokens=200, precise="bf16",
                precise_cost=2.0)
    run_checked(h)
    check_plans(h)
    h.check()
    assert any(sum(p["device"]) and any(p["precise"]) for p in h.plans)  # PRECISE rows next to the decode row
    for p in h.plans:
        dec = sum(p["device"])
        if dec:
            pre = sum(q for q, pr in zip(p["n_query"], p["precise"]) if pr)
            fp8 = p["n_tokens"] - pre - dec
            assert 2 * pre + fp8 <= max(16, 200 // dec), p


def test_nonfinite_precise_rows_roll_back_and_fail_their_dependents() -> None:
    """`bf16` mode: a cold prompt's PRECISE final chunk reports NONFINITE; what it wrote into the precise namespace is
    rolled back, and the siblings that waited for its branch checkpoint and resumed from it fail too."""
    specs = [Spec(rid=0, prompt_len=0, prompt=shared(100, 10, 0), max_tokens=3, seed=900, nonfinite_after=0)] + \
        [Spec(rid=i, prompt_len=0, prompt=shared(100, 10, i), max_tokens=3, seed=900 + i) for i in range(1, 4)]
    h = Harness(specs=specs, vocab=VOCAB, max_num_seqs=4, max_tokens=512, page_size=4, num_pages=300,
                checkpoint_slots=8, grid=16, max_model_len=256, precise="bf16")
    run_checked(h)
    h.check(failed_ok=True)
    assert h.results[0].finish == FinishReason.ERROR
    assert all(h.results[i].finish == FinishReason.ERROR for i in range(1, 4))
    assert h.sched.cache.rolled_back > 0


# -- how a refine is stepped (fixed pieces, dev.cache.refine_chunk_tokens) ---------------------------------------------


def gemma_refine(n_dec: int, prefix: int, **kw) -> Harness:
    """Gemma 4 31B's serving shape, scaled down: window mode (window pages of 16 positions, the checkpoint grid),
    2,048-token steps, MTP k = 3 from a drafter without KV, the 31B's measured mixed-step bound (30,816) and PRECISE
    cost (1.80), the default `refine_min`. `n_dec` requests decode throughout; five share a `prefix`-token system
    prompt, 25 turns apart: the first one's end-of-prefill checkpoint lies past the shared part, so the second computes
    from 0 and checkpoints the branch point r, the third resumes there, the fourth refines [0, r) and the fifth hits
    the precise copy."""
    specs = [Spec(rid=100 + d, prompt_len=0, prompt=stem(50 + d, 20), max_tokens=400, seed=9000 + d, ignore_eos=True)
             for d in range(n_dec)]
    specs += [Spec(rid=i, prompt_len=0, prompt=shared(prefix, 20, i), max_tokens=4, seed=8000 + i, arrive=5 + 25 * i)
              for i in range(5)]
    return window_harness(specs, window=64, page_size=16, full=80, grid=16, max_tokens=2048, max_model_len=4096,
                          num_pages=(n_dec + 8) * arena_bound(64, 16, 80, 2048, 4096) + 64, max_num_seqs=36, mtp_k=3,
                          draft_kv=False, row_tokens=30816, precise_cost=1.80, refine_min=REFINE_MIN, precise="reuse",
                          **kw)


def refine_rows(h: Harness) -> list[tuple[int, int]]:
    return [(p0, q) for _, p0, q in precise_rows(h)]


def run_refine(h: Harness) -> Harness:
    watch_refines(h)
    run_checked(h)
    check_plans(h)
    check_refine_plans(h)
    h.check()
    return h


@pytest.mark.parametrize("prefix, rows", [(1330, [(0, 1328)]), (2480, [(0, 1904), (1904, 576)])])
@pytest.mark.parametrize("n_dec", [0, 3, 12, 32])
def test_fixed_refine_pieces_do_not_follow_the_load(n_dec: int, prefix: int, rows: list) -> None:
    """A refine is cut only at multiples of the piece size (derived: min(2,048, 2,048 - 36 rows x 4) = 1,904, which
    fits next to a full batch of decode rows) and at its end, each piece whole in one step past the mixed-step bound,
    so the precise copy has the same boundaries at every load (cut like any chunk, the same 1,328 tokens took one, two
    and three steps next to 0, 3-12 and 32 decode rows). The refine still ends exactly at its FP8 checkpoint, and the
    fifth request hits the precise copy there."""
    h = gemma_refine(n_dec, prefix)
    assert h.sched.piece == 1904
    run_refine(h)
    r = sum(rows[-1])
    assert refine_rows(h) == rows and h.sched.precise_steps == len(rows)
    assert [h.results[i].cached for i in range(5)] == [0, 0, r, 0, r]


def test_a_mixed_step_piece_fits_next_to_a_full_batch() -> None:
    """An explicit piece must fit next to a full batch of decode rows (at most max_num_batched_tokens - max_num_seqs x
    (1 + k): the derived piece without its 2,048 cap). A larger one runs only in steps with fewer decode rows: a piece
    of 2,048 at 2,048-token steps fits next to none, so a waiting refiner would hold the queue until nothing decodes,
    and under steady load a refine's next piece would starve while every request on its prefix waits for it (with the
    fake, a 4,200-token refine's second piece runs in the 1,215th step, against the 82nd with the derived piece). The
    largest piece accepted here (2,048 - 36 x 4 = 1,904) runs whole next to a full batch."""
    for piece in (1920, 2048):
        with pytest.raises(ValueError, match=r"at most 1904 \(.*: a piece must fit next to a full batch"):
            gemma_refine(0, 2480, refine_chunk_tokens=piece)
    h = run_refine(gemma_refine(35, 2480, refine_chunk_tokens=1904))
    assert refine_rows(h) == [(0, 1904), (1904, 576)]
    for step, _, _ in precise_rows(h):
        assert sum(next(p for p in h.plans if p["step"] == step)["device"]) == 35


def test_a_harness_cap_replaces_the_piece_within_the_step_budget() -> None:
    """`precise_cap` (the harness's cap on PRECISE chunks) replaces the piece size, at most max_num_batched_tokens: a
    larger cap gives pieces of the whole budget, not a piece that never fits."""
    h = run_refine(gemma_refine(3, 2480, precise_cap=4096))
    assert refine_rows(h) == [(0, 2048), (2048, 432)]


def test_fixed_pieces_count_from_position_0() -> None:
    """Pieces end at multiples of the piece size counted from position 0, not from a refine's start: four requests
    share a 710-token prefix (the third refines [0, 704) in pieces of 512), then six a 1,330-token one that extends it
    (the eighth refines [704, 1280) from the precise checkpoint at 704: cut at 1024, not at 704 + 512). So a refiner
    readmitted after a preemption mid-refine continues on the pieces of an uninterrupted refine."""
    specs = [Spec(rid=i, prompt_len=0, prompt=shared(710 if i < 4 else 1330, 20, i), max_tokens=4, seed=7500 + i,
                  arrive=10 * i) for i in range(10)]
    h = run_refine(Harness(specs=specs, vocab=VOCAB, max_num_seqs=2, max_tokens=2048, page_size=16, num_pages=600,
                           checkpoint_slots=8, grid=64, max_model_len=2048, precise="reuse", refine_min=REFINE_MIN,
                           refine_chunk_tokens=512))
    assert refine_rows(h) == [(0, 512), (512, 192), (704, 320), (1024, 256)] and h.sched.refines == 2
    assert [h.results[i].cached for i in range(10)] == [0, 704, 0, 704, 704, 704, 1280, 704, 1280, 1280]


def test_a_fixed_refiner_preempted_mid_refine_continues_on_the_same_pieces() -> None:
    """The scenario above, one request at a time in 576-token steps: the eighth request refines [704, 1280) from the
    precise checkpoint at 704, and is preempted after its first piece, cut at 1024 (a multiple of 512 from position 0;
    pieces counted from the refine's start would end at 1216, and the step's budget would take all 576 tokens at
    once), its slot taken by a higher-priority request. It donates its precise state there, and its readmission
    resumes at 1024 and refines [1024, 1280): the pieces of an uninterrupted refine."""
    # the eighth arrives at turn 70, admitted with its first piece in that turn's plan: preempt it after that piece
    specs = [Spec(rid=i, prompt_len=0, prompt=shared(710 if i < 4 else 1330, 20, i), max_tokens=4, seed=7500 + i,
                  arrive=10 * i, priority=1, ignore_eos=True, preempt_at=71 if i == 7 else None) for i in range(10)]
    specs.append(Spec(rid=99, prompt_len=0, prompt=stem(9, 40), max_tokens=3, seed=99, arrive=71, priority=0))
    h = run_refine(Harness(specs=specs, vocab=VOCAB, max_num_seqs=1, max_tokens=576, page_size=16, num_pages=600,
                           checkpoint_slots=8, grid=64, max_model_len=2048, policy="priority", precise="reuse",
                           refine_min=REFINE_MIN, refine_chunk_tokens=512))
    assert refine_rows(h) == [(0, 512), (512, 192), (704, 320), (1024, 256)]
    assert h.sched.num_preemptions == 1 and h.sched.ckpts_taken["refine"] == 2
    assert any(p["seq_len"] == [1024] and p["precise"] == [True] for p in h.plans)
    assert [h.results[i].cached for i in range(10)] == [0, 704, 0, 704, 704, 704, 1280, 704, 1280, 1280]


def test_a_fixed_piece_waits_for_room_instead_of_shrinking() -> None:
    """A refiner (the fourth request) arrives while a 600-token prompt's last chunk (88 tokens) takes part of a
    256-token step. Its first 192-token piece does not fit the 168 tokens left: it waits one step rather than shrink to
    them (which would move the copy's boundaries with the load), and runs whole next to that prompt's first decode
    row."""
    specs = [Spec(rid=i, prompt_len=0, prompt=shared(300, 10, i), max_tokens=4, seed=7600 + i, arrive=12 * i)
             for i in range(4)]
    specs.append(Spec(rid=4, prompt_len=0, prompt=stem(9, 600), max_tokens=8, seed=7690, arrive=35))
    h = Harness(specs=specs, vocab=VOCAB, max_num_seqs=4, max_tokens=256, page_size=4, num_pages=700,
                checkpoint_slots=6, grid=16, max_model_len=1024, precise="reuse", refine_chunk_tokens=192)
    s, waits = h.sched, []
    admit = s._admit_one

    def record(req):
        out = admit(req)
        if out is Admit.NO_ROOM and req.rid == 3:
            waits.append(s.step_id + 1)  # the plan being built: its piece did not fit
        return out

    s._admit_one = record
    run_refine(h)
    assert refine_rows(h) == [(0, 192), (192, 96)]
    f0 = precise_rows(h)[0][0]
    assert waits == [f0 - 1]  # a step later, whole
    before, first = (next(p for p in h.plans if p["step"] == step) for step in (f0 - 1, f0))
    assert (512, 88) in zip(before["pos0"], before["n_query"]) and not any(before["precise"])
    assert first["n_tokens"] == 192 + 1


def test_fixed_pieces_pass_the_mixed_step_bound_but_fp8_chunks_keep_it() -> None:
    """The scenario of `test_precise_tokens_weigh_more_in_the_mixed_step_bound` (K = 200 next to one decode row, a
    PRECISE cost of 2) under `reuse`: the refine's 400 tokens go in one piece next to the decode row, past the bound by
    design (`check_plans` exempts a refine's PRECISE rows), while FP8 chunks keep it."""
    long = Spec(rid=0, prompt_len=0, prompt=stem(7, 20), max_tokens=200, seed=5, ignore_eos=True)
    specs = [long] + [Spec(rid=i, prompt_len=0, prompt=shared(400, 8, i), max_tokens=2, seed=4000 + i,
                           arrive=5 + 30 * i) for i in range(1, 5)]
    h = run_refine(Harness(specs=specs, vocab=VOCAB, max_num_seqs=2, max_tokens=512, page_size=16, num_pages=400,
                           checkpoint_slots=6, grid=16, max_model_len=1024, row_tokens=200, precise="reuse",
                           precise_cost=2.0))
    assert refine_rows(h) == [(0, 400)] and h.sched.refines == 1
    step = precise_rows(h)[0][0]
    assert sum(next(p for p in h.plans if p["step"] == step)["device"]) == 1
    for p in h.plans:
        dec = sum(p["device"])
        if dec:
            pre = sum(q for q, pr in zip(p["n_query"], p["precise"]) if pr)
            assert p["n_tokens"] - pre - dec <= max(16, 200 // dec), p


def test_fixed_pieces_keep_image_spans_whole() -> None:
    """A refine over a bidirectional image span (Gemma 4's images; 36 tokens at [100, 136) in the 200 shared ones)
    next to three decode rows, in pieces of 112: a piece is cut before a span that straddles the piece's end (on the
    grid, which leaves [96, 100)), and the piece that starts at the span takes it whole, past the multiple. The fake
    raises on any split span."""
    specs = [Spec(rid=100 + d, prompt_len=0, prompt=stem(50 + d, 20), max_tokens=300, seed=9100 + d, ignore_eos=True)
             for d in range(3)]
    specs += [Spec(rid=i, prompt_len=0, prompt=shared(200, 12, i), max_tokens=3, seed=9200 + i, arrive=5 + 20 * i,
                   images=[(100, 12, 12, 1)]) for i in range(6)]
    h = run_refine(window_harness(specs, window=32, page_size=8, full=40, grid=8, max_tokens=128, max_model_len=512,
                                  max_num_seqs=6, encoder_rows=256, encoder_budget=128, row_tokens=160,
                                  precise_cost=1.8, precise="reuse", bidirectional=True, refine_chunk_tokens=112))
    assert refine_rows(h) == [(0, 96), (96, 4), (100, 36), (136, 64)] and h.sched.refines == 1
    assert [h.results[i].cached for i in range(6)] == [0, 0, 200, 0, 200, 200]


@pytest.mark.parametrize("cold_first", [False, True])
def test_a_refine_piece_shares_its_step_with_a_cold_prompt(cold_first: bool) -> None:
    """A refiner and a cold 100-token prompt arrive together with nothing running: in either order both go into one
    step, the refiner's 704-token piece beside the cold prompt's chunk."""
    specs = [Spec(rid=i, prompt_len=0, prompt=shared(710, 20, i), max_tokens=4, seed=7700 + i, arrive=10 * i)
             for i in range(3)]
    cold = Spec(rid=9, prompt_len=0, prompt=stem(9, 100), max_tokens=4, seed=7709, arrive=20)
    specs = specs[:2] + ([cold, specs[2]] if cold_first else [specs[2], cold])
    h = run_refine(Harness(specs=specs, vocab=VOCAB, max_num_seqs=4, max_tokens=2048, page_size=16, num_pages=600,
                           checkpoint_slots=8, grid=64, max_model_len=1024, precise="reuse", refine_min=REFINE_MIN))
    assert refine_rows(h) == [(0, 704)]
    i = next(i for i, p in enumerate(h.plans) if any(p["precise"]))
    j = next(j for j, p in enumerate(h.plans) if (0, 100) in zip(p["pos0"], p["n_query"]))  # the cold prompt
    assert j == i and h.plans[i]["n_tokens"] == 704 + 100


def test_the_logged_precise_cost_is_the_one_the_scheduler_applies() -> None:
    """The engine logs and passes `derive_precise_cost`: the measured ratio of a PRECISE prefill step to an FP8 one, at
    least 1 (the two minimums' noise can put it below), as the scheduler applies it; 1 when either is unmeasured."""
    kw = dict(max_num_seqs=2, max_num_batched_tokens=256, max_model_len=512, page_size=16, num_pages=64,
              num_state_slots=8, eos_token_ids=(EOS,), checkpoint_grid=64)
    for precise_ms, prefill_ms, want in ((9.0, 10.0, 1.0), (25.0, 10.0, 2.5), (0.0, 10.0, 1.0), (9.0, 0.0, 1.0)):
        cost = derive_precise_cost(precise_ms, prefill_ms)
        assert cost == pytest.approx(want) and Scheduler(**kw, precise_cost=cost).precise_cost == cost


def test_refine_settings_are_validated() -> None:
    """The piece size: a positive multiple of the checkpoint grid, at most what a step leaves next to a full batch of
    decode rows (by default derived: 2,048 at most, and that room at the drafter's largest k, on the grid)."""
    kw = dict(max_num_seqs=2, max_num_batched_tokens=256, max_model_len=512, page_size=16, num_pages=64,
              num_state_slots=8, eos_token_ids=(EOS,), checkpoint_grid=64)
    for n in (100, 320, -64, 256):  # 256 - 2 rows: no room
        with pytest.raises(ValueError, match=f"refine_chunk_tokens {n}: "):
            Scheduler(**kw, refine_chunk_tokens=n)
    assert Scheduler(**kw, refine_chunk_tokens=128).piece == 128
    assert Scheduler(**kw).piece == 192  # 256 - 2 rows, on the 64-token grid
    # the drafter's largest k, never below the set k: 256 - 7 rows x 8 on the 16-token grid, where the set k = 1 alone
    # leaves 256 - 7 rows x 2
    kw7 = {**kw, "max_num_seqs": 7, "checkpoint_grid": 16}
    assert Scheduler(**kw7, mtp_k=1).piece == 240
    assert Scheduler(**kw7, mtp_k=1, mtp_k_max=7).piece == Scheduler(**kw7, mtp_k=7).piece == 192
    assert Scheduler(**kw7, mtp_k=7, mtp_k_max=3).piece == 192
    cases = [((2048, 64, 3, 16), 1792), ((8192, 64, 3, 16), 2048), ((2048, 36, 3, 16), 1904), ((256, 2, 0, 64), 192),
             ((128, 2, 0, 16), 112), ((16, 5, 3, 8), 8),
             # the default step budget and sequences: the assistant's largest k, 7 (at any k; from the set k = 5 alone
             # it would be 1,664), and Qwen's MTP layer at its k, 3, on Qwen's 64-token grid
             ((2048, 64, 7, 16), 1536), ((2048, 64, 5, 16), 1664), ((2048, 64, 3, 64), 1792)]
    for (tokens, seqs, k, grid), n in cases:
        for drafter in (None, "the assistant"):  # the set k (Qwen's MTP layer), the assistant's largest
            got, why = derive_refine_chunk_tokens(max_num_batched_tokens=tokens, max_num_seqs=seqs, mtp_k_max=k,
                                                  grid=grid, drafter=drafter)
            room = n + seqs * (1 + k) <= tokens
            assert got == n and why.startswith(f"{n:,}: ")
            assert ("full batch" if room else "no room next to" if drafter else "waits") in why
            assert ("at any k" in why) == (bool(drafter) and room)
            assert ("largest k" in why) == bool(drafter)


def test_the_derived_piece_does_not_depend_on_k(tmp_path) -> None:
    """The derived piece leaves room for decode rows of the drafter's largest k, whatever k is set, so a change of k
    never changes the precise copy. With an assistant, 1,536 at the default settings (64 sequences, 2,048-token steps,
    Gemma's 16-token grid) at every k from 0 to 7; Qwen's MTP layer keeps the set k, so its piece and reason follow k
    (1,792 at its default 3); without a drafter 1,984."""
    from llm_mini_infer import config

    from ..support.config_cases import _assistant, _gemma, _model

    gemma, asst = _gemma(tmp_path / "gemma"), _assistant(tmp_path / "assistant")
    for name in ("qwen", "plain"):
        (tmp_path / name).mkdir()
    qwen, plain = _model(tmp_path / "qwen", 248_320, 1), _model(tmp_path / "plain", 248_320, 0)

    def piece(model: Path, grid: int, k: int | None, **kw) -> tuple[int, str]:
        """The piece as the engine derives it, for `model` derived with k set (None: its default)."""
        over = {"model.path": str(model), **kw} | ({} if k is None else {"speculative.num_speculative_tokens": k})
        cfg = config.derive(config.load(None, over))
        assert cfg.mtp_k_max >= cfg.speculative.num_speculative_tokens
        return derive_refine_chunk_tokens(max_num_batched_tokens=cfg.scheduler.max_num_batched_tokens,
                                          max_num_seqs=cfg.scheduler.max_num_seqs, mtp_k_max=cfg.mtp_k_max,
                                          grid=grid, drafter="the assistant" if cfg.speculative.model else None)

    with_assistant = {piece(gemma, 16, k, **{"speculative.model": str(asst)}) for k in (None, *range(8))}
    assert with_assistant == {(1536, "1,536: min(2,048, max_num_batched_tokens 2,048 - 64 rows x 8 (1 + the "
                                     "assistant's largest k, 7)) on the 16-token grid: a piece fits next to a full "
                                     "batch of decode rows at any k, so k never changes the precise copy")}
    for k, n in ((None, 1792), (0, 1984), (1, 1920), (2, 1856), (3, 1792)):  # Qwen: the set k
        rows = (2048 - n) // 64  # 1 + k
        assert piece(qwen, 64, k) == (n, f"{n:,}: min(2,048, max_num_batched_tokens 2,048 - 64 rows x {rows}) on the "
                                         "64-token grid: a piece fits next to a full batch of decode rows")
    assert {piece(gemma, 16, None)[0], piece(plain, 64, None)[0]} == {1984}  # no drafter: 2,048 - 64 rows x 1
    for k in range(8):  # the scheduler's own derivation, given the drafter's largest k as the engine passes it
        s = Scheduler(max_num_seqs=64, max_num_batched_tokens=2048, max_model_len=4096, page_size=16, num_pages=64,
                      num_state_slots=65, eos_token_ids=(EOS,), checkpoint_grid=16, mtp_k=k, mtp_k_max=7)
        assert s.piece == 1536


# -- fuzzing ---------------------------------------------------------------------------------------------------------


@campaign_settings(300)
@given(precise_workloads())
def test_random_precise_workloads(w) -> None:
    with saved_on_failure("precise", w):
        run_precise_workload(w)


def test_the_cache_suite_runs_under_the_precise_prefix() -> None:
    """One of the cache suite's own workload shapes (`run_cache_workload`) with the policy on: nothing it checks
    changes (outputs, refcounts, the end state)."""
    specs = [Spec(rid=i, prompt_len=0, prompt=shared(60, 5, i % 3), max_tokens=6, seed=500 + i, arrive=4 * i)
             for i in range(9)]
    run_cache_workload(dict(specs=specs, page_size=4, grid=16, max_model_len=160, max_num_seqs=3, max_tokens=64,
                            num_pages=200, checkpoint_slots=4, precise="reuse"))


@campaign_settings(300)
@given(window_precise_workloads())
def test_random_window_precise_workloads(w) -> None:
    with saved_on_failure("window_precise", w):
        run_precise_workload(w)
