"""The sliding-window cache (Gemma 4) end to end through the scheduler and the validating fake runner.

In window mode the sliding layers' KV lives in a window pool: a running request holds only the window pages its next
queries can see, and the radix cache's checkpoints are windows (refcounted window pages before a resume point) instead
of GDN state slots. The fake checks, at every step, that a row's window pages hold its own tokens wherever its queries
look (a wrong window, or a page released or reused while still in view, raises PlanError) and samples tokens from the
true context (a wrong reuse changes the outputs, which are predicted). Every run also checks the refcounts of both
pools and that everything comes back once the cache is cleared.
"""

from __future__ import annotations

import pytest
from hypothesis import given

from llm_mini_infer.engine.messages import CHECKPOINT_SOURCES, FinishReason

from ..support.fuzzing import campaign_settings, saved_on_failure
from ..support.harness import Harness, Spec
from ..support.sched_cases import output_of, shared, stem
from ..support.window_workloads import (
    arena_bound,
    arena_hits,
    arena_pressure,
    harness,
    run_window_workload,
    window_pressure,
    window_workloads,
)


def window_pages_held(h: Harness) -> list[int]:
    return [sum(1 for p in r.wpages if p) for r in h.sched.slot_req if r is not None]


def test_a_long_request_holds_only_its_window() -> None:
    """A 200-token prompt and 60 decode steps with a window of 16 (pages of 4): the request holds about 16 / 4 + 2
    window pages at any time, never its whole context; its outputs are the predicted ones."""
    spec = Spec(rid=0, prompt_len=200, max_tokens=60, seed=5, ignore_eos=True)
    h = harness([spec], max_tokens=64, prefix_caching=False)
    peak = 0
    orig = h.core.step

    def step(*a, **kw):
        nonlocal peak
        r = orig(*a, **kw)
        peak = max([peak, *window_pages_held(h)])
        return r

    h.core.step = step
    h.run(check_every_turn=True)
    h.check()
    # a 64-token chunk sees 15 earlier positions: (15 + 64) / 4 + 1 pages at most while prefilling
    assert peak <= -(-(15 + 64) // 4) + 1
    assert h.sched.kv.num_free == h.sched.kv.num_pages - 1


def test_siblings_wait_then_hit_a_window_at_the_branch_point() -> None:
    """Four requests share a 100-token prefix and arrive together: the first computes it and takes a window at the
    branch point (100, on its page grid); the others wait one step and resume there, sharing the window's pages."""
    specs = [Spec(rid=i, prompt_len=0, prompt=shared(100, 10, i), max_tokens=3, seed=1000 + i) for i in range(4)]
    h = harness(specs, max_num_seqs=4)
    h.run(check_every_turn=True)
    h.check()
    assert [h.results[i].cached for i in range(4)] == [0, 100, 100, 100]
    assert h.sched.cache_waits == 3 and h.sched.ckpts_taken["branch"] == 1
    second = h.plans[1]
    assert sorted(second["admit"]) == [2, 3, 4] and second["seq_len"] == [100, 100, 100]
    assert second["state_copy"] == 0 and second["kv_copy"] == 0  # windows copy nothing at page-aligned hits


def test_classify_shape_reuses_the_shared_window() -> None:
    """A classification-shaped workload, text only, scaled down: a 700-token shared prefix, short unique items, one
    request at a time with a window of 64: with a hint on the prefix, every request after the first resumes at 700."""
    specs = [Spec(rid=i, prompt_len=0, prompt=shared(700, 20, i), hints=[700], max_tokens=4, seed=7000 + i,
                  arrive=10 * i) for i in range(30)]
    h = harness(specs, window=64, page_size=16, max_tokens=2048, max_model_len=1024)
    h.run()
    h.check()
    assert [h.results[i].cached for i in range(4)] == [0, 700, 700, 700]
    s = h.sched
    assert s.prefix_hits / s.prefix_queries > 0.9  # the ceiling: 29 x 700 of 30 x 720 tokens, 94%


def test_hint_off_the_page_grid_copies_both_tail_pages() -> None:
    """A hint at 50 (pages of 8) ends a chunk there; the next request sharing 55 tokens hits exactly 50 and copies
    the partial page of both pools (the full layers' KV and the window)."""
    a = stem(1, 80)
    b = [*a[:55], 7, 8, 9, 10]
    specs = [Spec(rid=0, prompt_len=0, prompt=a, hints=[50], max_tokens=2, seed=1),
             Spec(rid=1, prompt_len=0, prompt=b, max_tokens=2, seed=2, arrive=20)]
    h = harness(specs, page_size=8)
    h.run(check_every_turn=True)
    h.check()
    assert h.plans[0]["n_query"] == [50]
    assert h.results[1].cached == 50 and h.runner.counts["kv_copy"] == 2


def test_multi_turn_hits_the_finish_window() -> None:
    """Turn 2 is turn 1's prompt and output plus a new message: it resumes where turn 1 stopped, from the window turn
    1 left at its finish."""
    t1 = Spec(rid=0, prompt_len=0, prompt=stem(2, 37), max_tokens=12, seed=11)
    out = output_of(t1)
    t2 = Spec(rid=1, prompt_len=0, prompt=t1.prompt + out + [5, 6, 7], max_tokens=4, seed=12, arrive=40)
    h = harness([t1, t2], window=8)
    h.run(check_every_turn=True)
    h.check()
    assert h.results[1].cached == 37 + 12 - 1 and h.sched.ckpts_taken["finish"] == 2
    assert served(h) == {"finish": 1}  # turn 1's finish window served turn 2


def served(h: Harness) -> dict[str, int]:
    """The resumes each checkpoint source served (llm_mini_infer:checkpoint_resumes_total), those that served any."""
    return {CHECKPOINT_SOURCES[i]: n for i, n in enumerate(h.sched.ckpt_resumes) if n}


def evicted(h: Harness) -> dict[str, list[int]]:
    """The windows evicted, by source: [having served none, having served some]
    (llm_mini_infer:checkpoint_evictions_total)."""
    return {CHECKPOINT_SOURCES[i]: p for i, p in enumerate(h.sched.cache.evicted_by) if any(p)}


def test_one_shot_requests_leave_windows_that_serve_nothing() -> None:
    """Twelve one-shot requests over a shared 60-token prefix, one at a time, in an arena of one request's bound and 4
    pages: the second takes a branch window where they diverge, which serves every later one; each request's prompt-end
    and finish windows serve none and are evicted as the arena fills, freeing window pages;
    the windows alive are those left."""
    specs = [Spec(rid=i, prompt_len=0, prompt=shared(60, 12, i), max_tokens=4, seed=900 + i, arrive=12 * i)
             for i in range(12)]
    h = harness(specs, max_tokens=32, max_model_len=128, max_num_seqs=1,
                num_pages=arena_bound(16, 4, 20, 32, 128) + 4)
    h.run(check_every_turn=True)
    s, c = h.sched, h.sched.cache
    assert {k: v for k, v in s.ckpts_taken.items() if v} == {"prefill": 12, "branch": 1, "finish": 12}
    assert served(h) == {"branch": 10} and [h.results[i].cached for i in range(3)] == [0, 0, 60]
    assert evicted(h) == {"prefill": [11, 0], "finish": [11, 0]} and c.evicted_window_pages > 0
    assert c.num_ckpts == 25 - 22 == s.windows.num_windows
    h.check()


def test_decode_windows_serve_a_continuation() -> None:
    """A long generation takes decode windows every 16 positions (one at a time); a continuation that shares the
    first 60 generated tokens resumes from the deepest one at or below its match."""
    t1 = Spec(rid=0, prompt_len=0, prompt=stem(3, 20), max_tokens=80, seed=21, ignore_eos=True)
    out = output_of(t1)
    t2 = Spec(rid=1, prompt_len=0, prompt=t1.prompt + out[:60] + [9, 9], max_tokens=3, seed=22, arrive=70)
    h = harness([t1, t2], window=8, decode_checkpoint_interval=16)
    h.run(check_every_turn=True)
    h.check()
    assert h.sched.ckpts_taken["decode"] >= 3
    assert h.results[1].cached == 80  # the decode window at 80 (20 prompt + 60 generated)


def test_rows_crossing_pages_together_share_a_window() -> None:
    """Ten requests resume from one window (a 40-token prompt another request computed) and decode in step: every 4
    tokens their rows cross a window page boundary together, every 20 a full page's too, and past the window they give
    back the pages of the window they share, then their own. The strict fake checks every row's window, and the
    scheduler's accounting is checked at every turn. On the steps whose plan is the ten rows alone (a token each), the
    scheduler's `_grow` calls that take pages and its `_wrelease` calls that give pages back are counted: all ten rows
    take a full page in one step, and give back in one step pages of which one was held by two rows (the window they
    resumed from)."""
    prompt = stem(7, 40)
    specs = [Spec(rid=0, prompt_len=0, prompt=prompt, max_tokens=2, seed=70)]
    specs += [Spec(rid=i, prompt_len=0, prompt=prompt, max_tokens=60, seed=70 + i, arrive=8, ignore_eos=True)
              for i in range(1, 11)]
    h = harness(specs, max_num_seqs=10)
    s, steps, cur = h.sched, [], {"grow": 0, "released": []}
    grow, release, schedule = s._grow, s._wrelease, s.schedule

    def counted_grow(slot: int, n: int) -> None:
        cur["grow"] += n > 0  # the decode loop also calls it for rows that need only a window page (n <= 0)
        grow(slot, n)

    def counted_release(slot: int, q: int) -> None:
        a, lo = int(s.wlo[slot]), s._wlow(q)
        if lo > a:  # the row gives pages back: those it holds below its next query's window
            cur["released"].append([p for p in s.slot_req[slot].wpages[a:lo] if p])
        release(slot, q)

    def counted_schedule():  # one step's calls, with what its plan holds
        cur.update(grow=0, released=[])
        planned = schedule()
        if planned is not None:
            inf, held = planned[1], [p for got in cur["released"] for p in got]
            steps.append((len(inf.slots), inf.n_tokens, cur["grow"], len(cur["released"]), len(set(held)) < len(held)))
        return planned

    s._grow, s._wrelease, s.schedule = counted_grow, counted_release, counted_schedule
    h.run(check_every_turn=True)
    h.check()
    assert all(h.results[i].cached == 36 for i in range(1, 11))  # in step: the prompt's end on the page grid
    decode = [st for st in steps if st[:2] == (10, 10)]  # the ten rows alone, a token each
    assert any(g == 10 for _, _, g, _, _ in decode), steps  # the ten cross a full page boundary in one step
    assert any(r == 10 and two for _, _, _, r, two in decode), steps  # and give back a page two of them held


def test_preemption_mid_prefill_resumes_from_its_window() -> None:
    """A request preempted halfway through a long prefill, its slot taken by a higher-priority one, leaves a window
    where it stopped and resumes there."""
    low = Spec(rid=0, prompt_len=150, max_tokens=4, seed=31, priority=1, preempt_at=2)
    high = Spec(rid=1, prompt_len=120, max_tokens=4, seed=32, priority=0, arrive=2)
    h = harness([low, high], window=8, max_tokens=32, max_num_seqs=1, policy="priority", max_model_len=200)
    h.run(check_every_turn=True)
    h.check()
    assert h.sched.num_preemptions >= 1 and h.sched.readmit_hits > 0
    assert h.sched.ckpts_taken["preempt"] >= 1 and served(h)["preempt"] >= 1  # its donation, a source of its own


def test_window_pressure_evicts_windows_before_preempting() -> None:
    """An arena for about two requests and twenty requests over a shared prefix: windows are evicted under pressure,
    requests wait for pages, and every output stays exact."""
    specs = [Spec(rid=i, prompt_len=0, prompt=shared(40, 30 + i, i), max_tokens=10, seed=4000 + i, arrive=i)
             for i in range(20)]
    one = arena_bound(16, 4, 20, 64, 512)
    h = harness(specs, max_tokens=64, max_num_seqs=4, num_pages=2 * one + 1)
    h.run(check_every_turn=True)
    h.check()
    assert h.sched.cache.evicted_ckpts > 0


def test_junction_tails_leave_a_lone_request_its_bound() -> None:
    """The window-mode twin of test_cache_sched's (Gemma 4's mode): full pages of 16, windows of 8 in pages of 4, an
    arena at the startup bound. Six requests branch off one prompt at 90-95, inside the full page [80, 96), with a hint
    at each; then R, alone, resumes at 95 and prefills to 159 in chunks of 32. The junctions at 90-94 lie above R's hit
    inside the page it copies, so they would keep the tails their evicted branches left, pinned by R's lock: R's last
    chunk would never fit beside them, and nothing else would run. Under page pressure they take a child's page for
    their tail."""
    s = stem(1, 160)
    specs = [Spec(rid=i, prompt_len=0, prompt=s[:90 + i] + stem(10 + i, 6), hints=[90 + i], max_tokens=1, seed=1 + i,
                  arrive=3 * i) for i in range(6)]
    specs.append(Spec(rid=6, prompt_len=0, prompt=s[:159], max_tokens=1, seed=99, arrive=28))
    h = harness(specs, window=8, page_size=4, full=16, max_tokens=32, max_model_len=160, max_num_seqs=2,
                num_pages=arena_bound(8, 4, 16, 32, 160) + 1)
    h.run(max_turns=500, check_every_turn=True)
    h.check()
    assert h.results[6].cached == 95 and h.sched.num_preemptions == 0 and h.sched.cache.shared_tails == 5


def test_nonfinite_rolls_back_windows_and_fails_dependents() -> None:
    """A request whose prefill reports NaN fails; the windows it wrote are rolled back, and a sibling that resumed from
    them fails too, while an unrelated request is untouched."""
    a = Spec(rid=0, prompt_len=0, prompt=shared(64, 4, 0), max_tokens=4, seed=51, nonfinite_after=0)
    b = Spec(rid=1, prompt_len=0, prompt=shared(64, 4, 1), max_tokens=4, seed=52)
    c = Spec(rid=2, prompt_len=0, prompt=stem(9, 30), max_tokens=4, seed=53)
    h = harness([a, b, c], max_num_seqs=3)
    h.run(check_every_turn=True)
    assert h.sched.cache.rolled_back > 0 and h.sched.cache.evicted_ckpts == 0 and not evicted(h)  # no eviction
    h.check(failed_ok=True)
    assert h.results[0].finish == FinishReason.ERROR
    assert h.results[2].finish == FinishReason.LENGTH


@pytest.mark.parametrize("window", [1, 2, 3, 4, 5, 9])
def test_tiny_windows(window) -> None:
    """Windows of one key (its own) up to a few pages, with hits, decode and finish windows."""
    specs = [Spec(rid=i, prompt_len=0, prompt=shared(23, 5 + i, i), max_tokens=7, seed=900 + 10 * window + i,
                  arrive=3 * i) for i in range(5)]
    h = harness(specs, window=window, page_size=2, max_tokens=16, decode_checkpoint_interval=4, max_num_seqs=2)
    h.run(check_every_turn=True)
    h.check()


# -- fuzzing -------------------------------------------------------------------------------------------------------


@campaign_settings(300)
@given(window_workloads())
def test_random_window_workloads(w) -> None:
    with saved_on_failure("window", w):
        run_window_workload(w)


@campaign_settings(300)
@given(window_pressure())
def test_random_window_pressure(w) -> None:
    with saved_on_failure("window_pressure", w):
        run_window_workload(w)


# -- the KV arena under pressure -----------------------------------------------------------------------------------


@campaign_settings(300)
@given(arena_pressure())
def test_random_arena_pressure(w) -> None:
    with saved_on_failure("arena_pressure", w):
        run_window_workload(w)


@campaign_settings(300)
@given(arena_hits())
def test_random_arena_hits(w) -> None:
    with saved_on_failure("arena_hits", w):
        run_window_workload(w)
