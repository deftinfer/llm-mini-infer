"""NaN safety of the prefix cache through the scheduler and the validating fake.

A row whose state or KV is non-finite poisons everything computed from it: a cached entry that holds such KV, a state
or a window fails every request that resumes from it. The fake marks what NaN poisoned (`nonfinite_after`: a SAMPLE
row; `nonfinite_write`: the row that computes a prompt position, sampling or not) and spreads it to every row that reads
it; `poisoned_entries` finds it in the cache. By default only SAMPLE rows report NONFINITE, as on the Qwen runner, so a
prefill chunk that is not its prompt's last writes NaN unseen, and the request's next SAMPLE row reports it
(`report_unsampled`: a runner that checks every row, as Gemma 4's does; a NaN it cannot see, one the FP8 path's
activation quantizer turns finite, still goes unseen). The rules under test (`sched.scheduler`, Rollback;
architecture.md, "Failures and rollback"):
- a NaN fails its request and removes what it wrote, but no request that only shares a cached prefix with it: the tree
  never adopts a request's copy of the page holding its hit into the node the hit ends in (the junction keeps its tail
  page and its stamp), and only requests whose hit read a rolled-back write fail with it;
- an empty grammar mask fails its request alone;
- what a request's prefill wrote stays unverified until its next finite SAMPLE row, and is removed when that row is
  NaN, when the request is aborted, and, at a readmission, as far as the readmission does not read it; the tree never
  takes unverified pages in for the pages it holds verified, so a removal takes only the request's own nodes (and the
  checkpoints it attached): nothing it only shared.
The cache suites draw NaN into their workloads; `test_random_nan_workloads` (the campaign suite `nan`) aims small ones
at these classes.
"""

from __future__ import annotations

import pytest
from hypothesis import given

from llm_mini_infer.engine.messages import FinishReason
from llm_mini_infer.sched.scheduler import RUNNING

from ..support.fuzzing import campaign_settings, saved_on_failure
from ..support.harness import Harness, Spec, poisoned_entries
from ..support.nan_cases import nan_workloads, run_nan_workload
from ..support.sched_cases import VOCAB, output_of, shared, stem
from ..support.window_workloads import harness as window_harness

LENGTH, ERROR, ABORT = FinishReason.LENGTH, FinishReason.ERROR, FinishReason.ABORT
MODES = ["fp8", "reuse", "window", "window_reuse"]  # GDN and window mode (Gemma 4's), without and with refine on reuse


def make(mode: str, specs: list[Spec], **kw) -> Harness:
    """GDN mode with pages of 4, or window mode with window pages of 4 and full pages of 20; with the precise
    prefix's refine on reuse (`*reuse`) or without."""
    precise = "reuse" if mode.endswith("reuse") else "fp8"
    if mode.startswith("window"):
        return window_harness(specs, window=16, page_size=4, precise=precise, **kw)
    return Harness(specs=specs, vocab=VOCAB, page_size=4, num_pages=300, checkpoint_slots=6, grid=16, max_tokens=256,
                   precise=precise, **kw)


def finishes(h: Harness) -> list[int]:
    return [h.results[rid].finish for rid in sorted(h.results)]


def junction(mode: str, **last) -> Harness:
    """W computes a 47-token prompt with a hint at 42, inside a page of either pool. R1 resumes from it, then R2 and
    R3 (which decode long), then R4 after R1 left: each copies the page holding 42 (KV_COPY)."""
    specs = [Spec(rid=0, prompt_len=0, prompt=shared(42, 5, 0), hints=[42], max_tokens=2, seed=10)]
    for i, (arrive, n) in enumerate(((10, 3), (11, 40), (11, 40), (22, 3)), 1):
        specs.append(Spec(rid=i, prompt_len=0, prompt=shared(42, 5, i), max_tokens=n, seed=10 + i, arrive=arrive,
                          ignore_eos=n == 40, **(last if i == 4 else {})))
    return make(mode, specs, max_num_seqs=4)


@pytest.mark.parametrize("mode", MODES)
def test_a_nan_at_the_junction_fails_only_its_request(mode: str) -> None:
    """R4's first sample is NaN: what R4 wrote leaves the cache, and R4 fails alone. Were each hitter's insert to
    adopt its copy of the junction's page as the junction's tail, stamping the junction with its own step, R4's
    rollback would remove the junction's whole subtree, and R2 and R3 (decoding on branches below it) would fail with
    it."""
    h = junction(mode, nonfinite_after=0)
    h.run(check_every_turn=True)
    assert finishes(h) == [LENGTH] * 4 + [ERROR]
    assert h.sched.num_failed == 1 and not poisoned_entries(h)
    h.check(failed_ok=True)


@pytest.mark.parametrize("mode", MODES)
def test_an_empty_mask_fails_its_request_without_rollback(mode: str) -> None:
    """R4's grammar mask allows nothing at its first sample (MASK_EMPTY): its token is invalid, but nothing it computed
    is, so it fails alone and nothing is rolled back."""
    h = junction(mode, grammar="fake:7:3:10", mask_empty_after=0)
    h.run(check_every_turn=True)
    assert finishes(h) == [LENGTH] * 4 + [ERROR]
    assert h.sched.num_failed == 1 and h.sched.cache.rolled_back == 0
    h.check(failed_ok=True)


@pytest.mark.parametrize("mode", MODES)
def test_a_hit_inside_a_page_never_restamps_the_junction(mode: str) -> None:
    """R1 and R2 arrive together after W finished and both resume from its hint at 42, inside a page: the junction keeps
    its tail page and its stamp, so R2 resumes in the same plan as R1 (were R1's insert to adopt its copy as the
    junction's tail, R2 would wait a step for it)."""
    specs = [Spec(rid=0, prompt_len=0, prompt=shared(42, 5, 0), hints=[42], max_tokens=2, seed=10)]
    specs += [Spec(rid=i, prompt_len=0, prompt=shared(42, 5, i), max_tokens=3, seed=10 + i, arrive=10) for i in (1, 2)]
    h = make(mode, specs, max_num_seqs=4, refine_after=99)
    cache, swaps = h.sched.cache, []
    adopt = cache._adopt

    def record(node, *a, **kw):
        before = node.tail, node.ready, node.writer
        adopt(node, *a, **kw)
        if node.end == 42 and (node.tail, node.ready, node.writer) != before:
            swaps.append((before, (node.tail, node.ready, node.writer)))

    cache._adopt = record
    h.run(check_every_turn=True)
    h.check()
    assert [h.results[i].cached for i in (1, 2)] == [42, 42]
    assert h.sched.cache_waits == 0 and not swaps, swaps


@pytest.mark.parametrize("mode", ["fp8", "window"])
def test_a_rollback_spares_requests_that_only_locked_the_removed_node(mode: str) -> None:
    """Pages of one position (no copy-on-write anywhere). X recomputes the 50 tokens it shares with R0 while R0 holds
    them, so it keeps its own pages and locks the tree's; R0 leaves. H resumes from X's checkpoint at 48 and adopts its
    own pages into the node [48, 50), which only the tree referenced; H's first sample is NaN, and its rollback removes
    that node with X's locked branch below it. X read nothing H wrote: it runs on (and caches nothing more), not
    failed for its lock alone."""
    x = stem(31, 69)
    specs = [Spec(rid=0, prompt_len=0, prompt=x, max_tokens=10, seed=1287, ignore_eos=True),
             Spec(rid=1, prompt_len=0, prompt=x[:50] + [(j * 11 + 294) % (VOCAB - 1) for j in range(87)],
                  max_tokens=40, seed=5287, arrive=6, ignore_eos=True),
             Spec(rid=2, prompt_len=0, prompt=x[:50] + [(j * 13 + 77) % (VOCAB - 1) for j in range(20)],
                  max_tokens=3, seed=6287, arrive=30, nonfinite_after=0)]
    kw = dict(grid=8, max_model_len=160, max_num_seqs=3, max_tokens=64, num_pages=400)
    h = Harness(specs=specs, vocab=VOCAB, page_size=1, checkpoint_slots=4, **kw) if mode == "fp8" else \
        window_harness(specs, window=16, page_size=1, full=1, **kw)
    s, nocache = h.sched, []
    mark = s._mark_nocache

    def record():
        mark()
        nocache.extend(r.rid for r in s.slot_req if r is not None and r.nocache and r.rid not in nocache)

    s._mark_nocache = record
    h.run(check_every_turn=True)
    assert h.sched.prefix_hits == 48 and h.sched.cache.rolled_back > 0  # H resumed from X's checkpoint
    assert finishes(h) == [LENGTH, LENGTH, ERROR] and h.sched.num_failed == 1 and nocache == [1]
    assert not poisoned_entries(h)
    h.check(failed_ok=True)


def test_a_rollback_spares_a_hit_that_read_none_of_its_pages() -> None:
    """Attention-only, pages of 4. C computes 30 tokens; D and E branch off them at 10 and 8, so [8, 10) is a node whose
    only page is the tail of index 2 (its child holds C's own page of that index). R resumes at 8, computes on its own
    pages from there and branches off at 11: its insert adopts its page as the tails of [8, 10) and [10, 11), stamping
    them with its step. Z resumes at 20 in the next plan, reading C's pages only; R's first sample is NaN, and its
    rollback takes those nodes and the path below them. Z runs on: a hit depends on the nodes whose pages it reads,
    not on every node of its path stamped with a step in flight."""
    c = stem(40, 30)
    specs = [Spec(rid=0, prompt_len=0, prompt=c, max_tokens=2, seed=900),
             Spec(rid=1, prompt_len=0, prompt=[*c[:10], 1, 2, 3], max_tokens=2, seed=901, arrive=8),
             Spec(rid=2, prompt_len=0, prompt=[*c[:8], 4, 5, 6], max_tokens=2, seed=902, arrive=16),
             Spec(rid=3, prompt_len=0, prompt=[*c[:11], 7, 8, 9, 10, 11], max_tokens=2, seed=903, arrive=30,
                  nonfinite_after=0),
             Spec(rid=4, prompt_len=0, prompt=[*c[:20], 12, 13, 14, 15], max_tokens=3, seed=904, arrive=31)]
    h = Harness(specs=specs, vocab=VOCAB, page_size=4, num_pages=200, hybrid=False, max_num_seqs=4)
    h.run(check_every_turn=True)
    assert [h.results[i].cached for i in (1, 2, 4)] == [8, 8, 20] and h.sched.cache.rolled_back > 0
    assert finishes(h) == [LENGTH] * 3 + [ERROR, LENGTH] and not h.cascaded
    assert not poisoned_entries(h)
    h.check(failed_ok=True)


def test_a_node_split_off_a_writes_node_goes_with_its_rollback() -> None:
    """GDN, bf16 mode, one slot. W's 34-token prompt runs in one PRECISE chunk, which writes NaN at 12; W is preempted
    with that step in flight, and R (W's first 19 tokens, a higher priority) takes its slot and computes its prompt from
    position 0: its insert adopts its pages of [0, 16) into W's node [0, 32), stamping it with R's step, and splits it
    at 19, the node [0, 19) keeping W's page of index 2 as its tail. W's sample reports the NaN, and the rollback of its
    step takes that node too (R, which locked it, caches nothing more). Were a node split off recorded under its node's
    last stamp only (R's), W's poisoned page would stay in the precise namespace."""
    p = stem(3, 34)
    specs = [Spec(rid=0, prompt_len=0, prompt=p, hints=[32], max_tokens=3, seed=100, priority=1, nonfinite_write=12,
                  preempt_at=1),
             Spec(rid=1, prompt_len=0, prompt=p[:19], max_tokens=3, seed=101, arrive=1)]
    h = Harness(specs=specs, vocab=VOCAB, page_size=8, grid=8, max_num_seqs=1, policy="priority", precise="bf16",
                max_tokens=64, max_model_len=96, num_pages=40, checkpoint_slots=2)
    s, nocache = h.sched, []
    mark = s._mark_nocache

    def record():
        mark()
        nocache.extend(r.rid for r in s.slot_req if r is not None and r.nocache and r.rid not in nocache)

    s._mark_nocache = record
    h.run(check_every_turn=True)
    assert s.num_preemptions == 1 and nocache == [1] and finishes(h) == [ERROR, LENGTH]  # W recomputes its NaN
    assert not poisoned_entries(h)
    h.check(failed_ok=True)


@pytest.mark.parametrize("report", [False, True])
@pytest.mark.parametrize("how", ["", "abort", "preempt"])
@pytest.mark.parametrize("mode", MODES)
def test_nan_in_a_non_final_chunk_leaves_the_cache(mode: str, how: str, report: bool) -> None:
    """W's 100-token prompt runs in chunks of 16 with a hint at 64 (60 in window mode: page-aligned in both pools, so no
    copy-on-write is involved); its chunk [16, 32) writes NaN and reports nothing (it does not sample). W then fails
    at its final chunk (which reads the NaN), is aborted mid-prefill, or is preempted mid-prefill (a higher-priority
    request takes its slot) and fails after its readmission. R1-R4 share W's 64 first tokens and arrive later: what W
    wrote never reaches them, and the cache still serves them (W's completed chunks and its donation, left in the
    cache, would fail every later hitter). `report`: a runner that reports NONFINITE on rows that do not sample
    (Gemma 4's) fails W at the chunk itself; such a runner alone closes the hazard."""
    hit = 60 if mode.startswith("window") else 64
    specs = [Spec(rid=0, prompt_len=0, prompt=shared(64, 36, 0), hints=[hit], max_tokens=3, seed=500, priority=1,
                  nonfinite_write=20, abort_at=4 if how == "abort" else None,
                  preempt_at=4 if how == "preempt" else None)]
    specs += [Spec(rid=i, prompt_len=0, prompt=shared(64, 6, i), max_tokens=3, seed=500 + i, arrive=20 + 10 * i)
              for i in range(1, 5)]
    if how == "preempt":
        specs.append(Spec(rid=5, prompt_len=0, prompt=stem(9, 10), max_tokens=2, seed=505, arrive=4))
    h = make(mode, specs, max_num_seqs=1, chunk_cap=16, policy="priority", report_unsampled=report)
    h.run(check_every_turn=True)
    assert h.results[0].finish == (ABORT if how == "abort" and not report else ERROR)
    assert finishes(h)[1:5] == [LENGTH] * 4
    if how == "preempt" and not report:
        assert h.sched.num_preemptions >= 1  # (a runner that checks every row fails W before)
    assert not poisoned_entries(h)
    assert max(h.results[i].cached for i in (2, 3, 4)) > 0
    h.check(failed_ok=True)


@pytest.mark.parametrize("report", [False, True])
@pytest.mark.parametrize("mode", ["reuse", "window_reuse"])
def test_nan_in_a_refine_leaves_the_cache(mode: str, report: bool) -> None:
    """The refiner of a shared prefix (refine on reuse, the default policy) writes NaN in its PRECISE rows, which end
    at the FP8 checkpoint it refines and never sample: its next chunk reports it (`report`: the PRECISE row itself, as
    Gemma 4's runner does). Its precise copy leaves the cache with it, and nothing else does: the next request refines
    the prefix's FP8 copy again at once, and the ones after it resume from that precise copy. A removal that took the
    FP8 copy too (everything on the refiner's path from position 0 in its precise range's FP8 sibling) would leave the
    next request to compute the prefix again from nothing."""
    if mode == "reuse":
        specs = [Spec(rid=i, prompt_len=0, prompt=shared(710, 20, i), max_tokens=4, seed=7000 + i, arrive=10 * i)
                 for i in range(8)]
        refiner, full = 2, 704  # its PRECISE row is [0, 704)
        specs[refiner].nonfinite_write = 100
        h = Harness(specs=specs, vocab=VOCAB, max_num_seqs=2, max_tokens=2048, page_size=16, num_pages=400,
                    checkpoint_slots=6, grid=64, max_model_len=1024, precise="reuse", report_unsampled=report)
    else:
        specs = [Spec(rid=i, prompt_len=0, prompt=shared(300, 20, i), max_tokens=4, seed=3000 + i, arrive=10 * i)
                 for i in range(8)]
        refiner, full = 3, 296  # its PRECISE row is [0, 296)
        specs[refiner].nonfinite_write = 100
        h = window_harness(specs, window=32, page_size=8, max_tokens=512, max_model_len=640, precise="reuse",
                           report_unsampled=report)
    s, refiners = h.sched, []
    admit = s._admit_one

    def record(req):
        out = admit(req)
        if req.status == RUNNING and req.refine:
            refiners.append(req.rid)
        return out

    s._admit_one = record
    h.run(check_every_turn=True)
    assert refiners == [refiner, refiner + 1]
    assert finishes(h) == [ERROR if i == refiner else LENGTH for i in range(8)]
    assert [h.results[i].cached for i in range(refiner + 2, 8)] == [full] * (6 - refiner)
    assert not poisoned_entries(h)
    h.check(failed_ok=True)


def test_a_reader_of_an_unverified_write_does_not_reinsert_it() -> None:
    """GDN, MTP with a drafter that writes KV. R1's chunk [0, 1) (a hint) writes NaN unseen. R0 and R2 resume from R1's
    checkpoint at 1 once that step completed, so they depend on no write in flight; R0 copies R1's page 0 (its drafter's
    KV at 0 depends on the token at 1). R1's final chunk is NaN, and what it wrote leaves the cache, with R0's and R2's
    nodes below it. R0 read R1's NaN, but its next chunk would insert the pages it holds again, R1's copied page among
    them: a request whose cached path a removal took caches nothing more."""
    specs = [Spec(rid=0, prompt_len=0, prompt=stem(0, 20), hints=[4], max_tokens=4, seed=100, arrive=2, priority=2),
             Spec(rid=1, prompt_len=0, prompt=[5, 36, 67, 0, 0, 0], hints=[1], max_tokens=4, seed=101,
                  nonfinite_write=0),
             Spec(rid=2, prompt_len=0, prompt=[5, 36, 3], max_tokens=4, seed=102, arrive=2)]
    h = Harness(specs=specs, vocab=VOCAB, page_size=1, grid=8, max_num_seqs=3, max_tokens=16, num_pages=171,
                checkpoint_slots=3, policy="priority", mtp_k=3, accept_p=1.0, max_model_len=160)
    h.run(check_every_turn=True)
    assert [h.results[i].cached for i in (0, 2)] == [-1, -1]  # failed before their first token
    assert finishes(h) == [ERROR] * 3
    assert not poisoned_entries(h)
    h.check(failed_ok=True)


FIRST_CHUNK = (4, 16)  # a first chunk of 16 tokens: 4 full pages of 5 positions and 16 window pages of 1


def readmission_harness(specs: list[Spec]) -> Harness:
    """The readmission tests' pool (window mode, full pages of 5), where R0 (`preempt_at`, `preempt_pages`) is
    preempted for R1's first chunk's pages, and the cache evicts for a first chunk's pages again when the next plan
    starts, R0's donated window among them (as any request's page need would)."""
    h = Harness(specs=specs, vocab=VOCAB, window=2, window_page_size=1, page_size=5, grid=1, max_tokens=16,
                num_pages=54, policy="priority", max_model_len=160)
    s, plan, turn = h.sched, h.sched._schedule, specs[0].preempt_at + 1

    def evicting() -> object:
        if h.turn == turn:
            assert s._ensure_pages(*FIRST_CHUNK)
        return plan()

    s._schedule = evicting
    return h


def test_a_readmission_that_resumes_elsewhere_drops_its_unverified_writes() -> None:
    """Window mode, full pages of 5. R0's first chunk writes NaN unseen, and every later chunk reads it; R0 is preempted
    at 96, before its final chunk, for the pages of a higher-priority request R1's first chunk (R0's first 42 tokens,
    then another). R0's window is evicted, R1 recomputes its prompt cleanly and leaves a branch window at 42, and R0's
    readmission resumes there, inside a page: it copies R1's page, and its finite samples read nothing R0 wrote before.
    Its old writes must leave the cache: the tree takes R0's copy into the node starting at the hit, [42, 48), so R0's
    new pages replace its old ones there, and the readmission's removal takes what they do not (a tree that took no copy
    would keep R0's old page in that node)."""
    p = stem(0, 97)
    specs = [Spec(rid=0, prompt_len=0, prompt=p, max_tokens=4, seed=200, priority=1, nonfinite_write=0, preempt_at=6,
                  preempt_pages=FIRST_CHUNK),
             Spec(rid=1, prompt_len=0, prompt=[*p[:42], 0], max_tokens=4, seed=201, arrive=4)]
    h = readmission_harness(specs)
    h.run(check_every_turn=True)
    assert h.sched.num_preemptions >= 1 and h.sched.readmit_hits == 42
    assert finishes(h) == [LENGTH, LENGTH]
    assert not poisoned_entries(h)
    h.check()


@pytest.mark.parametrize("mode", ["attention", "window"])
def test_a_stateless_resume_does_not_inherit_a_nan_flag(mode: str) -> None:
    """The fake: one slot; B's first sample is NaN; C then resumes from A's prefix in the slot B left. Without recurrent
    state (attention-only and window mode) the fake must not hand C the NaN flag of the slot's previous occupant."""
    specs = [Spec(rid=0, prompt_len=0, prompt=[5, 0, 1, 2], max_tokens=2, seed=1),
             Spec(rid=1, prompt_len=0, prompt=[9, 9], max_tokens=2, seed=2, arrive=4, nonfinite_after=0),
             Spec(rid=2, prompt_len=0, prompt=[5, 0, 1, 2, 3, 4], max_tokens=2, seed=3, arrive=8)]
    h = Harness(specs=specs, vocab=VOCAB, max_num_seqs=1, page_size=2, hybrid=False) if mode == "attention" else \
        window_harness(specs, window=4, page_size=2, max_num_seqs=1)
    h.run(check_every_turn=True)
    assert finishes(h) == [LENGTH, ERROR, LENGTH] and h.results[2].cached > 0
    h.check(failed_ok=True)


def test_an_fp8_continuation_of_an_unverified_precise_copy_is_removed_with_it() -> None:
    """`bf16` mode: R1's PRECISE chunk [0, 16) writes NaN unseen. R0 resumes from R1's precise checkpoint at 48, and its
    FP8 continuation's insert puts the pages below its hit (R1's precise pages) into the FP8 namespace. When R1's final
    chunk reports the NaN, its writes leave the precise namespace and the FP8 one."""
    p0 = [*stem(0, 58), 0]
    specs = [Spec(rid=0, prompt_len=0, prompt=p0, max_tokens=4, seed=300, arrive=1),
             Spec(rid=1, prompt_len=0, prompt=[*p0[:53], 0], max_tokens=4, seed=301, nonfinite_write=0)]
    h = Harness(specs=specs, vocab=VOCAB, page_size=1, grid=8, max_num_seqs=2, max_tokens=16, num_pages=163,
                checkpoint_slots=1, precise="bf16", refine_after=0, max_model_len=160)
    h.run(check_every_turn=True)
    assert h.sched.precise_hits == 1 and finishes(h) == [ERROR, ERROR]
    assert not poisoned_entries(h)
    h.check(failed_ok=True)


@pytest.mark.parametrize("how", ["nan", "abort", "preempt", "nocache"])
@pytest.mark.parametrize("mode", ["gdn", "attention", "window"])
def test_the_page_an_mtp_admission_copies_below_its_hit_is_an_unverified_write(mode: str, how: str) -> None:
    """The MTP twin of the test above (the NaN fuzz's case, window_precise, seed 11). `bf16` mode, MTP with a drafter
    that writes KV, pages of 8 (full pages in window mode), chunks of 4. W's PRECISE chunk [0, 4) writes NaN unseen;
    H resumes from W's precise copy at 8 (its hint's checkpoint; attention-only, its KV) while W is unverified. H's
    drafter recomputes its KV at 7, which depends on H's token at 8, so H copies W's page [0, 8) (KV_COPY), NaN
    included, and its FP8 continuation's insert puts the copy into the FP8 namespace as the node [0, 8). H's first
    chunk [8, 12) does not sample. Then H's final chunk reads the NaN and fails (`nan`), H is aborted (`abort`), H is
    preempted for a higher-priority request and its readmission fails (`preempt`), or W fails first (its prompt is
    shorter), its removal takes H's precise hit, and H runs on caching nothing more until it fails (`nocache`); W
    fails at its final chunk. H's copy leaves the cache with H's other unverified writes. Marked from the first page
    of H's chunk (index 1) only, the copy (index 0) would be nobody's: W's removal looks for W's pages, H's starts at
    8, and [0, 8) would stay. Without MTP, or with a drafter that writes no KV (Gemma 4's assistant), H
    shares W's page, which W's removal finds; a hit inside a page copies the page holding the hit, the chunk's first,
    which was marked."""
    p = stem(0, 24)
    first = how == "nocache"
    specs = [Spec(rid=0, prompt_len=0, prompt=p[:12] if first else p, hints=[8], max_tokens=4, seed=400,
                  nonfinite_write=3),
             Spec(rid=1, prompt_len=0, prompt=[*p[:8], *stem(5, 24 if first else 8)], max_tokens=4, seed=401, arrive=1,
                  priority=1, abort_at=3 if how == "abort" else None, preempt_at=3 if how == "preempt" else None)]
    if how == "preempt":
        specs.append(Spec(rid=2, prompt_len=0, prompt=stem(9, 4), max_tokens=2, seed=402, arrive=3))
    kw = dict(chunk_cap=4, max_num_seqs=2, policy="priority", precise="bf16", mtp_k=3, max_model_len=160)
    if mode == "window":
        h = window_harness(specs, window=8, page_size=8, full=8, **kw)
    else:
        h = Harness(specs=specs, vocab=VOCAB, page_size=8, num_pages=100, checkpoint_slots=4, grid=8, max_tokens=64,
                    hybrid=mode == "gdn", **kw)
    s, nocache = h.sched, []
    mark = s._mark_nocache

    def record():
        mark()
        nocache.extend(r.rid for r in s.slot_req if r is not None and r.nocache and r.rid not in nocache)

    s._mark_nocache = record
    h.run(check_every_turn=True)
    assert s.precise_hits >= 1 and s.num_preemptions == (how == "preempt") and (nocache == [1] or not first)
    assert any(p["kv_copy"] for p in h.plans if 8 in p["seq_len"])  # H's admission copies the page [0, 8)
    assert finishes(h) == [ERROR, ABORT if how == "abort" else ERROR] + [LENGTH] * (how == "preempt")
    assert not poisoned_entries(h)
    h.check(failed_ok=True)


@pytest.mark.parametrize("mode", ["reuse", "window_reuse"])
def test_the_page_an_mtp_admission_copies_below_a_refined_hit_is_an_unverified_write(mode: str) -> None:
    """The test above under refine on reuse (the default policy), where an FP8 continuation of a precise hit inserts
    into the prefix's FP8 copy, which keeps its pages (never the request's copy), unless that copy left the cache. E
    computes 40 tokens in chunks of 4 with a hint at 8, and R refines E's FP8 checkpoint there at once (`refine_after`
    0): R's PRECISE chunk [0, 4) writes NaN unseen. E is aborted mid-prefill, and its unverified writes, the prefix's
    FP8 copy among them, leave the cache. H then resumes from R's precise copy at 8 while R is unverified and copies
    R's page [0, 8) for its drafter, and its FP8 insert puts the copy into the emptied FP8 namespace as the node [0, 8).
    R's final chunk and H's read the NaN and fail, and H's copy leaves the cache with them (R's removal finds R's pages
    only, and H's starts at 8)."""
    e = stem(0, 40)
    specs = [Spec(rid=0, prompt_len=0, prompt=e, hints=[8], max_tokens=2, seed=500, abort_at=5),
             Spec(rid=1, prompt_len=0, prompt=[*e[:8], *stem(5, 8)], max_tokens=2, seed=501, arrive=2,
                  nonfinite_write=3),
             Spec(rid=2, prompt_len=0, prompt=[*e[:8], *stem(7, 8)], max_tokens=2, seed=502, arrive=5)]
    kw = dict(chunk_cap=4, max_num_seqs=3, precise="reuse", refine_after=0, mtp_k=3, max_model_len=160,
              precise_cap=4)  # R's refine in pieces of 4 (the derived piece: one of 8)
    if mode == "window_reuse":
        h = window_harness(specs, window=8, page_size=8, full=8, **kw)
    else:
        h = Harness(specs=specs, vocab=VOCAB, page_size=8, num_pages=100, checkpoint_slots=4, grid=8, max_tokens=64,
                    **kw)
    h.run(check_every_turn=True)
    assert h.sched.refines == 1 and h.sched.precise_hits == 1
    assert any(p["kv_copy"] for p in h.plans if 8 in p["seq_len"])  # H's admission copies the page [0, 8)
    assert finishes(h) == [ABORT, ERROR, ERROR]
    assert not poisoned_entries(h)
    h.check(failed_ok=True)


def test_the_page_an_mtp_admission_copies_from_verified_kv_is_not_an_unverified_write() -> None:
    """The tests above with nothing unverified where H copies, and no NaN. W's PRECISE prompt is done (its hint's
    checkpoint at 8). H resumes from W's precise copy at 8, copies W's page [0, 8) for its drafter, and its FP8 insert
    puts the copy into the empty FP8 namespace as the node [0, 8); R resumes from the precise copy at 8 too and
    finishes, its branch below that node (its finish checkpoint at 13). H is aborted mid-prefill: its removal takes what
    H wrote past its hit, nothing else, and Q, R's next turn, resumes at 13. Were the copy below the hit among H's
    unverified writes whatever it copied, the removal would take the node [0, 8) with R's branch, and Q would resume
    from the precise copy at 8. A copy of an unverified page still is one (the tests above): the copier's NaN removes
    it."""
    r = Spec(rid=2, prompt_len=0, prompt=stem(0, 8) + stem(7, 4), max_tokens=2, seed=12, arrive=6)
    specs = [Spec(rid=0, prompt_len=0, prompt=stem(0, 8) + stem(3, 4), hints=[8], max_tokens=1, seed=10),
             Spec(rid=1, prompt_len=0, prompt=stem(0, 8) + stem(5, 20), max_tokens=2, seed=11, arrive=4, abort_at=7),
             r, Spec(rid=3, prompt_len=0, prompt=r.prompt + output_of(r) + stem(9, 3), max_tokens=2, seed=13,
                     arrive=14)]
    h = Harness(specs=specs, vocab=VOCAB, page_size=8, grid=8, chunk_cap=4, mtp_k=1, accept_p=1.0, precise="bf16",
                max_num_seqs=3, num_pages=100, checkpoint_slots=6, max_model_len=160)
    h.run(check_every_turn=True)
    assert finishes(h) == [LENGTH, ABORT, LENGTH, LENGTH]
    assert [h.results[i].cached for i in (1, 2, 3)] == [-1, 8, 13]
    assert [p["kv_copy"] for p in h.plans if 8 in p["seq_len"]] == [1, 1]  # H's and R's admissions copy [0, 8)
    h.check()


@pytest.mark.parametrize("mode", ["gdn", "window"])
def test_the_page_an_mtp_admission_copies_below_an_unverified_checkpoint_is_an_unverified_write(mode: str) -> None:
    """The test above with the page H copies verified, but not the checkpoint H resumes from. `bf16` mode, MTP with a
    drafter that writes KV, pages of 8 (full pages in window mode), chunks of 4. W's PRECISE prompt is done, with no
    checkpoint at 8. X shares W's first 8 tokens and computes them again (on its own pages, as the tree keeps W's),
    writing NaN at 2 unseen: its branch checkpoint at 8 (a state in GDN mode, a window in window mode) lands on W's
    verified node, computed from X's NaN. H resumes from it and copies W's page [0, 8), where its drafter rewrites its
    KV at 7 from the checkpoint's state (the fake poisons it), and its FP8 insert puts the copy into the empty FP8
    namespace as the node [0, 8). X and H fail at their final chunks, and H's copy leaves the cache with H's other
    unverified writes. Were it one of them only when the page it copied is unverified, [0, 8) would stay with NaN in its
    drafter KV, which fails no request (the drafter's statuses are ignored) and costs every later reader its drafts."""
    p = stem(0, 8)
    specs = [Spec(rid=0, prompt_len=0, prompt=p + stem(3, 20), max_tokens=1, seed=10),
             Spec(rid=1, prompt_len=0, prompt=p + stem(4, 12), max_tokens=2, seed=11, arrive=12, nonfinite_write=2),
             Spec(rid=2, prompt_len=0, prompt=p + stem(5, 20), max_tokens=2, seed=12, arrive=15)]
    kw = dict(chunk_cap=4, max_num_seqs=3, precise="bf16", mtp_k=1, accept_p=1.0, max_model_len=160)
    if mode == "window":
        h = window_harness(specs, window=8, page_size=8, full=8, **kw)
    else:
        h = Harness(specs=specs, vocab=VOCAB, page_size=8, num_pages=100, checkpoint_slots=6, grid=8, **kw)
    h.run(check_every_turn=True)
    assert h.sched.ckpts_taken["branch"] >= 1 and h.sched.precise_hits == 1
    assert any(p["kv_copy"] for p in h.plans if 8 in p["seq_len"])  # H's admission copies the page [0, 8)
    assert finishes(h) == [LENGTH, ERROR, ERROR]
    assert not poisoned_entries(h)
    h.check(failed_ok=True)


# -- what a removal takes: the nodes holding the request's unverified pages, nothing it only shared --------------------


@pytest.mark.parametrize("how", ["abort", "nan"])
@pytest.mark.parametrize("mode", ["fp8", "window"])
def test_an_unverified_prefill_leaves_what_others_cached(mode: str, how: str) -> None:
    """U (100 tokens) is cached; R, a cold 200-token prompt that shares only U's first 5 tokens, runs in chunks of 32
    and is aborted mid-prefill, or writes NaN in a chunk before its last. What R wrote leaves the cache, and nothing
    else does: U's repeat still resumes 96 tokens deep. Were R's chunks to swap their pages into the node [0, 5) that
    U's path shares (the tree alone references U's), or the removal to take everything on R's path from position 0,
    U's entry would go, and with it every prompt sharing R's first token."""
    u, r = stem(0, 5) + stem(3, 95), stem(0, 5) + stem(7, 195)
    specs = [Spec(rid=0, prompt_len=0, prompt=u, max_tokens=2, seed=1),
             Spec(rid=1, prompt_len=0, prompt=r, max_tokens=2, seed=2, arrive=10,
                  abort_at=14 if how == "abort" else None, nonfinite_write=150 if how == "nan" else None),
             Spec(rid=2, prompt_len=0, prompt=u, max_tokens=2, seed=3, arrive=40)]
    h = make(mode, specs, chunk_cap=32, max_num_seqs=2)
    h.run(check_every_turn=True)
    assert finishes(h) == [LENGTH, ABORT if how == "abort" else ERROR, LENGTH]
    assert h.results[2].cached == 96 and not poisoned_entries(h)
    h.check(failed_ok=True)


@pytest.mark.parametrize("mode", ["reuse", "window_reuse"])
def test_an_aborted_refine_leaves_the_prefix_and_what_others_cached(mode: str) -> None:
    """Refine on reuse in PRECISE chunks of 64 (several steps); the refiner of the shared 300-token prefix is aborted
    mid-refine (a client gone). Its precise copy so far leaves the cache, and nothing else does: the next request
    refines again, the ones after it resume from the precise copy, and U, a prompt sharing only the prefix's first 5
    tokens, keeps its entry. A removal of everything on the refiner's path from position 0 in both namespaces (its
    precise range's FP8 sibling) would take the prefix's FP8 copy and every entry sharing its first token."""
    u = stem(0, 5) + stem(3, 95)
    specs = [Spec(rid=9, prompt_len=0, prompt=u, max_tokens=2, seed=99, hints=[96])]
    specs += [Spec(rid=i, prompt_len=0, prompt=shared(300, 20, i), max_tokens=4, seed=3000 + i, arrive=4 + 10 * i)
              for i in range(8)]
    specs.append(Spec(rid=10, prompt_len=0, prompt=u, max_tokens=2, seed=98, arrive=120))
    if mode == "reuse":
        refiner, full = 2, 256  # its PRECISE rows: [0, 256) in four steps (turns 24-27)
        specs[1 + refiner].abort_at = 26
        h = Harness(specs=specs, vocab=VOCAB, max_num_seqs=2, max_tokens=512, page_size=16, num_pages=400,
                    checkpoint_slots=12, grid=64, max_model_len=640, precise="reuse")
    else:
        refiner, full = 3, 296  # its PRECISE rows: [0, 296) in five steps
        specs[1 + refiner].abort_at = 36
        h = window_harness(specs, window=32, page_size=8, max_tokens=512, max_model_len=640, precise="reuse")
    h.sched.precise_cap = 64
    h.run(check_every_turn=True)
    assert finishes(h) == [ABORT if i == refiner else LENGTH for i in range(8)] + [LENGTH, LENGTH]  # then U, U again
    assert h.sched.refines == 2  # the aborted refine, and the next request's
    assert [h.results[i].cached for i in range(refiner + 2, 8)] == [full] * (6 - refiner)
    assert h.results[10].cached == 96
    h.check()


@pytest.mark.parametrize("nan", [False, True])
def test_a_readmission_elsewhere_removes_only_what_it_wrote_before(nan: bool) -> None:
    """The readmission test above, then R1's prompt and R0's again. R0's readmission resumes from R1's branch window at
    42: the pages below 42 it reads are R1's (R1's insert replaced R0's unverified ones there), and what R0 wrote past
    42 before its preemption leaves the cache. Nothing else does: R1's entry stays, and R0, whose insert after its
    readmission replaced its old pages with its new ones, caches its prompt and answer (R0's repeat resumes 96 deep).
    A removal of everything on R0's path from position 0 would take R1's entry, and with it the node R0 has locked
    since its readmission, so R0 would cache nothing more. Without NaN anywhere (`nan` False) as with it."""
    p = stem(0, 97)
    specs = [Spec(rid=0, prompt_len=0, prompt=p, max_tokens=4, seed=200, priority=1,
                  nonfinite_write=0 if nan else None, preempt_at=6, preempt_pages=FIRST_CHUNK),
             Spec(rid=1, prompt_len=0, prompt=[*p[:42], 0], max_tokens=4, seed=201, arrive=4),
             Spec(rid=2, prompt_len=0, prompt=[*p[:42], 0], max_tokens=4, seed=202, arrive=80),
             Spec(rid=3, prompt_len=0, prompt=p, max_tokens=4, seed=203, arrive=90)]
    h = readmission_harness(specs)
    h.run(check_every_turn=True)
    assert h.sched.num_preemptions >= 1 and h.sched.readmit_hits == 42
    assert finishes(h) == [LENGTH] * 4
    assert [h.results[i].cached for i in (2, 3)] == [42, 96] and not poisoned_entries(h)
    h.check()


def test_an_attention_only_readmission_keeps_what_it_reads() -> None:
    """Attention-only (no checkpoints: a hit is the page-aligned KV match): R's 100 tokens run in chunks of 16, and it
    is preempted mid-prefill for a higher-priority request. R's readmission resumes from its own KV, which its next
    SAMPLE row reads: it stays, R finishes caching its prompt, and its repeat resumes 100 deep (were only a donated
    checkpoint R's own, the readmission's removal would take what it reads, and R would cache nothing more)."""
    p = stem(0, 100)
    specs = [Spec(rid=0, prompt_len=0, prompt=p, max_tokens=4, seed=1, priority=1, preempt_at=4),
             Spec(rid=1, prompt_len=0, prompt=stem(9, 30), max_tokens=3, seed=2, arrive=4),
             Spec(rid=2, prompt_len=0, prompt=[*p, 7, 7], max_tokens=2, seed=3, arrive=40)]
    h = Harness(specs=specs, vocab=VOCAB, page_size=4, num_pages=40, hybrid=False, max_num_seqs=1, chunk_cap=16,
                policy="priority", max_model_len=160, max_tokens=64)
    h.run(check_every_turn=True)
    assert h.sched.num_preemptions == 1 and h.sched.readmit_hits == 64
    assert finishes(h) == [LENGTH] * 3 and h.results[2].cached == 100
    h.check()


@pytest.mark.parametrize("mode", ["fp8", "window"])
def test_a_removal_drops_the_checkpoints_it_attached_where_it_only_shared(mode: str) -> None:
    """U's 60 tokens are cached with no checkpoint below 20, where R branches off, so R computes its prompt from
    position 0, in chunks of 16, and its hint at 12 puts its checkpoint on U's node there. R writes NaN at 5: its final
    chunk fails it, its own nodes go, and so does that checkpoint, though the node holding it stays (U's). V branches
    off at 15: it resumes from no NaN (with the checkpoint left in the tree, V failed reading it)."""
    u = stem(0, 60)
    specs = [Spec(rid=0, prompt_len=0, prompt=u, max_tokens=2, seed=1),
             Spec(rid=1, prompt_len=0, prompt=u[:20] + stem(5, 60), hints=[12], max_tokens=2, seed=2, arrive=10,
                  nonfinite_write=5),
             Spec(rid=2, prompt_len=0, prompt=u[:15] + stem(6, 10), max_tokens=2, seed=3, arrive=40)]
    h = make(mode, specs, chunk_cap=16, max_num_seqs=2)
    h.run(check_every_turn=True)
    assert finishes(h) == [LENGTH, ERROR, LENGTH] and h.sched.prefix_hits == 0
    assert not poisoned_entries(h)
    h.check(failed_ok=True)


@pytest.mark.parametrize("mode", ["gdn", "window"])
def test_a_removal_drops_the_branch_checkpoint_its_row_took_for_another(mode: str) -> None:
    """Refine on reuse at the first reuse, no MTP. R1's 13 tokens run in chunks of 4, and its chunk [0, 4) writes NaN
    unseen at 1. R3, R1's first 9 tokens, arrives while R1's chunk [4, 8) is planned: its branch point lies in KV that
    row writes, so the row takes a checkpoint at 8 for it (`_branch_wait`; a window in window mode, a state in GDN
    mode), computed from R1's NaN, and R3 waits a step. R3 then refines [0, 8) and its FP8 insert puts its pages into
    the nodes [0, 4) and [4, 8), in place of R1's unverified ones. R1 is preempted for R2, and its readmission resumes
    from R3's precise copy at 8, elsewhere: what R1 wrote unverified leaves the cache, the checkpoint its row took for
    R3 too: it is among R1's unverified writes, since with R1's pages gone from its node no removal of pages would take
    it."""
    p = [*stem(2, 12), 424]
    specs = [Spec(rid=1, prompt_len=0, prompt=p, max_tokens=1, seed=15407, arrive=3, priority=1, nonfinite_write=1,
                  preempt_at=6),
             Spec(rid=3, prompt_len=0, prompt=p[:9], max_tokens=1, seed=17405, arrive=4),
             Spec(rid=2, prompt_len=0, prompt=[39], max_tokens=1, seed=146166, arrive=6)]
    kw = dict(page_size=4, grid=4, max_tokens=32, max_num_seqs=2, chunk_cap=4, policy="priority", precise="reuse",
              refine_after=0, max_model_len=160, precise_cap=4)  # R3's refine in pieces of 4
    h = Harness(specs=specs, vocab=VOCAB, window=8, window_page_size=2, num_pages=61, **kw) if mode == "window" \
        else Harness(specs=specs, vocab=VOCAB, num_pages=61, checkpoint_slots=8, **kw)
    h.run(check_every_turn=True)
    s = h.sched
    assert s.ckpts_taken["branch"] == 1 and s.refines >= 1 and s.num_preemptions == 1
    assert finishes(h) == [LENGTH] * 3
    assert not poisoned_entries(h)
    h.check()


@pytest.mark.parametrize("mode", ["fp8", "window"])
def test_the_tree_takes_a_verified_requests_pages_at_once(mode: str) -> None:
    """R recomputes U's cached 20 first positions (no checkpoint of U lies on R's path below 20) in chunks of 16: while
    they are unverified the tree keeps U's pages there and R holds its own. R's first sample verifies them, and the tree
    takes R's pages at once (U's go back to the pool), not only when R finishes."""
    u = stem(0, 60)
    specs = [Spec(rid=0, prompt_len=0, prompt=u, max_tokens=2, seed=1),
             Spec(rid=1, prompt_len=0, prompt=u[:20] + stem(5, 40), max_tokens=40, seed=2, arrive=10, ignore_eos=True)]
    h = make(mode, specs, chunk_cap=16, max_num_seqs=2)
    s, seen = h.sched, []
    step = h.core.step

    def check():
        out = step()
        r = s.reqs.get((0, 1))
        if r is not None and r.slot and h.results[1].tokens and not seen:  # R decodes: its first sample is in
            path = s.cache.match(None, s._key(r), 1 << 62, 16).path
            seen.append([p for n in path if n.end <= 16 for p in n.pages] == r.pages[:16 // s.ps])
        return out

    h.core.step = check
    h.run(check_every_turn=True)
    assert h.sched.prefix_hits == 0 and seen == [True]
    h.check()


# -- fuzzing: the NaN classes the cache suites reach rarely ------------------------------------------------------------


@campaign_settings(300)
@given(nan_workloads())
def test_random_nan_workloads(w) -> None:
    with saved_on_failure("nan", w):
        run_nan_workload(w)
