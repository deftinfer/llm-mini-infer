"""Bidirectional image spans (Gemma 4's `use_bidirectional_attention: "vision"`) through the scheduler in window mode
and the validating fake runner.

A span's queries see the whole span, so the scheduler keeps every span whole in one row: a chunk never ends inside a
span (it is cut before it, or takes all of it past the softer budgets), and no checkpoint or cache hit lies strictly
inside one (architecture.md, "The scheduler"; `Scheduler._span_end`). The fake checks, independently, that every run of
an image's tokens in a row is the whole image (index 0 to its last) and that SPANS names exactly those runs; a split
span or a resume inside one raises PlanError. Outputs are predicted as everywhere (`harness.Harness.check`).
"""

from __future__ import annotations

import numpy as np
from hypothesis import given

from llm_mini_infer.cache.pools import PagePool, StatePool
from llm_mini_infer.cache.radix import RadixCache
from llm_mini_infer.engine.messages import CHECKPOINT_SOURCES, FinishReason

from ..support.fuzzing import campaign_settings, saved_on_failure
from ..support.harness import Harness, Spec
from ..support.sched_cases import VOCAB, check_plans, stem
from ..support.window_workloads import arena_bound, gemma_image_workloads, harness, run_gemma_image_workload
from ..support.workloads import image_prompt, text


def gharness(specs: list[Spec], **kw) -> Harness:
    """Window mode (one arena) with images whose spans attend bidirectionally."""
    kw.setdefault("encoder_rows", 512)
    return harness(specs, bidirectional=True, **kw)


def run(specs: list[Spec], **kw) -> Harness:
    h = gharness(specs, **kw)
    h.run(check_every_turn=True)
    check_plans(h)
    h.check()
    return h


def rows_of(h: Harness, slot: int | None = None) -> list[tuple[int, int, int]]:
    """(slot, pos0, end) of every explicit row in the run (of one slot)."""
    out = []
    for p in h.plans:
        for s, n, pos0, dev in zip(p["slots"], p["n_query"], p["pos0"], p["device"]):
            if not dev and (slot is None or s == slot):
                out.append((s, pos0, pos0 + n))
    return out


def span_slot(h: Harness) -> int:
    """The slot of the first row that prefilled a span."""
    p = next(p for p in h.plans if p["spans"])
    return p["slots"][p["spans"][0][0]]


def test_a_span_is_never_split_and_may_pass_the_mixed_step_budget() -> None:
    """A request decoding next to a prefill whose mixed-step budget (40 tokens next to one decode row) is smaller than
    its 96-token image: the chunk before the image stops at the span (on the window grid), and the span then goes in
    one row of 96 + its text, past the soft budget but within max_num_batched_tokens; SPANS names it once."""
    a = Spec(rid=0, prompt_len=0, prompt=text(1, 30), max_tokens=60, seed=7)
    prompt = image_prompt(text(2, 10), 96, text(3, 20))
    b = Spec(rid=1, prompt_len=0, prompt=prompt, max_tokens=4, seed=8, images=[(10, 16, 24, 1)], arrive=3)
    h = run([a, b], max_tokens=128, row_tokens=40, window=16, page_size=4, grid=8)
    spans = [s for p in h.plans for s in p["spans"]]
    assert [(s[1], s[2]) for s in spans] == [(10, 106)]
    mine = rows_of(h, span_slot(h))
    assert not any(10 < e < 106 for _, _, e in mine)
    big = [r for r in mine if r[1] <= 10 and r[2] >= 106]
    assert len(big) == 1 and big[0][2] - big[0][1] > 40  # past the mixed-step budget


def test_a_span_longer_than_a_step_is_rejected() -> None:
    prompt = image_prompt(text(4, 4), 96, text(5, 4))
    spec = Spec(rid=0, prompt_len=0, prompt=prompt, max_tokens=2, seed=1, images=[(4, 16, 24, 1)])
    h = gharness([spec], max_tokens=64)
    h.run()
    assert h.results[0].finish == FinishReason.ERROR and h.released == {0}


def test_two_images_in_one_prompt_split_only_between_them() -> None:
    """Two 64-token images and a 72-token budget: the chunks cut between the spans, never inside one."""
    prompt = image_prompt(text(6, 8), 64, image_prompt([1, 2, 3], 64, text(7, 10)))
    spec = Spec(rid=0, prompt_len=0, prompt=prompt, max_tokens=4, seed=3, images=[(8, 16, 16, 1), (75, 16, 16, 2)])
    h = run([spec], max_tokens=72, window=16, page_size=4, grid=4)
    assert sorted((s[1], s[2]) for p in h.plans for s in p["spans"]) == [(8, 72), (75, 139)]
    assert not any(8 < e < 72 or 75 < e < 139 for _, _, e in rows_of(h))


def test_a_checkpoint_inside_a_span_moves_to_its_end() -> None:
    """The end-of-prefill checkpoint of an 80-token prompt lands on the grid of 16 at 64, inside the span [10, 74):
    it moves to the span's end, so the same prompt again hits there, past the whole image, not inside it (with causal
    spans, test_image_sched's same prompt hits inside it, at 72)."""
    prompt = image_prompt(text(9, 10), 64, text(10, 6))
    a = Spec(rid=0, prompt_len=0, prompt=prompt, max_tokens=4, seed=21, images=[(10, 16, 16, 1)])
    b = Spec(rid=1, prompt_len=0, prompt=prompt, max_tokens=4, seed=22, images=[(10, 16, 16, 1)], arrive=20)
    h = run([a, b], max_tokens=256, window=16, page_size=4, grid=16)
    assert h.results[1].cached == 74


def test_hits_before_and_after_a_span() -> None:
    """Same text, another image: no hit reaches into the span (the keys differ there; the first request had no window
    at the branch point, so the second computes it); the same image again: past the span."""
    pre, post = text(13, 40), text(14, 12)
    prompt = image_prompt(pre, 16, post)
    specs = [Spec(rid=i, prompt_len=0, prompt=prompt, max_tokens=3, seed=40 + i, images=[(40, 8, 8, iid)],
                  arrive=15 * i) for i, iid in enumerate([1, 2, 2])]
    h = run(specs, max_tokens=256, window=16, page_size=4, grid=4)
    assert h.results[1].cached <= 40  # never into the span of another image
    assert h.results[2].cached >= 56


def test_a_hint_inside_a_span_is_dropped() -> None:
    prompt = image_prompt(text(15, 20), 32, text(16, 10))
    spec = Spec(rid=0, prompt_len=0, prompt=prompt, max_tokens=2, seed=5, images=[(20, 8, 16, 1)], hints=[30])
    h = run([spec], max_tokens=256, window=16, page_size=4, grid=4)
    assert not any(20 < e < 52 for _, _, e in rows_of(h))


def test_the_match_backs_off_from_a_hit_inside_a_span() -> None:
    """The radix match with `forbid`: checkpoints at 32 and 48, a span (40, 60) forbidden: the hit is 32; a pending
    checkpoint inside it is not waited for."""
    kv = PagePool(64)
    states = StatePool(8, 2)
    cache = RadixCache(4, kv, states, hybrid=True)
    key = np.arange(100, dtype=np.int64)
    pages = kv.alloc(25)
    node = cache.insert(None, key, 100, pages, 1, 1)
    for c in (32, 48):
        cache.attach(cache.node_ending_at(node, c), cache.alloc_checkpoint(1), 1, 1, "prefill")
    cache.complete(1, set())
    assert cache.match(None, key, 5, 99).hit == 48
    m = cache.match(None, key, 5, 99, forbid=lambda p: 40 < p < 60)
    assert m.hit == 32 and m.pending == 0


def test_a_one_token_chunk_before_a_span_leaves_one_precise_copy() -> None:
    """The precise prefix under saturation, scaled down: 150 requests share 49 tokens, then each has an image of its
    own (a 20-token span at 49) and 6 tokens; from the sixth on, each comes with a 50-token filler, which spends most
    of a step's budget and keeps the arena (three requests' bound) full, so admissions evict. A cold chunk stops on the
    grid at 48, so the refine's precise copy (the fourth image request) covers [0, 48). A request that resumes there
    in a step whose budget ends inside the span gets the chunk [48, 49) before it (`Scheduler._span_end`), whose window
    at 49 lies on the copy. Later requests resume from the copy below it (`Scheduler._admit_one`, `short`), so the copy
    stays locked and is never evicted: one refine. Resuming from 49 instead would let the copy's lock drain, an eviction
    pass that spares locked windows would take it, and the eleventh image request would refine a second copy, to 49
    (refines 2; the other image requests from the seventh on would resume at 49)."""
    specs, images = [], []
    for k in range(150):
        if k >= 5:
            specs.append(Spec(rid=len(specs), prompt_len=0, prompt=stem(100 + k, 50), max_tokens=6,
                              seed=9000 + len(specs), arrive=2 * k, ignore_eos=True))
        images.append(len(specs))
        tail = [(k * 13 + j * 7 + 1) % (VOCAB - 1) for j in range(6)]
        specs.append(Spec(rid=len(specs), prompt_len=0, prompt=image_prompt(stem(0, 49), 20, tail), max_tokens=6,
                          seed=9000 + len(specs), arrive=2 * k, ignore_eos=True, images=[(49, 4, 20, 10 + k)]))
    h = gharness(specs, window=64, page_size=16, full=80, grid=16, max_tokens=64, max_model_len=256,
                 num_pages=int(3 * arena_bound(64, 16, 80, 64, 256)), max_num_seqs=8, precise="reuse",
                 refine_min=32, encoder_rows=256, encoder_budget=128)
    h.run(check_every_turn=True)
    check_plans(h)
    s, refine = h.sched, CHECKPOINT_SOURCES.index("refine")
    assert any(p0 == 48 and n == 1 for p in h.plans for n, p0 in zip(p["n_query"], p["pos0"]))  # the one-token chunk
    assert s.refines == 1 and s.cache.evicted_by[refine] == [0, 0]  # before `check`, whose `clear` evicts the copy
    cached = [h.results[i].cached for i in images]
    assert cached[2:4] == [48, 0] and cached[4:] == [48] * 146
    h.check()


# -- the fuzz suite (tests/scripts/fuzz_campaign.py: "gemma_images") ---------------------------------------------------


@campaign_settings(300)
@given(gemma_image_workloads())
def test_random_gemma_image_workloads(w) -> None:
    with saved_on_failure("gemma_images", w):
        run_gemma_image_workload(w)


