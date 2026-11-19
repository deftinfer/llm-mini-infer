"""GemmaRunner on CPU (the torch reference backends, fp32) against transformers: chunked prefill and decode across a
sliding window whose released pages are reused, a resume that shares and copies pages of both pools, batched rows, the
grammar split path, NONFINITE on rows that do not sample; then the engine end to end (scheduler in window mode, prefix
cache, preemption) against transformers' greedy generation."""

from __future__ import annotations

import types
from contextlib import nullcontext

import numpy as np
import pytest
import torch

from llm_mini_infer.engine.core import EngineCore
from llm_mini_infer.engine.messages import AddRequest, FinishReason
from llm_mini_infer.models.gemma4.model import SLIDING
from llm_mini_infer.runner.base import NONFINITE, PlanError
from llm_mini_infer.sched.plan import SAMPLE, STATE_ZERO, PlanBuilder
from llm_mini_infer.sched.scheduler import Scheduler

from ..support.tiny_gemma import (
    GREEDY,
    GDriver,
    gemma_cpu_runner,
    gemma_pools,
    hf_logits,
    hf_model,
    make_tiny_gemma,
    make_tiny_gemma_fp8,
    poisoned_rope,
)
from ..support.token_ids import ids

TOL = 5e-5
W = 8
POISON = 10  # the position whose sliding-layer RoPE the NaN tests make NaN (`poisoned_rope`)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return make_tiny_gemma(tmp_path_factory.mktemp("g4r"), window=W)


def logits_of(drv: GDriver, slot: int, n: int) -> torch.Tensor:
    return torch.stack([drv.logits[(slot, p)] for p in range(n)])


def test_chunked_prefill_and_decode_match_transformers(tiny) -> None:
    """Prefill in three uneven chunks (one of a single token), then 12 greedy decode steps from device input, with
    window pages released behind each row and reused by a second slot that runs alongside: every position matches one
    full forward of transformers."""
    seq, other = ids(40, 1), ids(30, 2)
    pc = gemma_pools(budget=64, page_size=4, num_pages=32)
    drv = GDriver(gemma_cpu_runner(tiny, pc), pc, window=W)
    steps = [dict(rows=[(1, seq[:13], 0, False)], admit=[(1, GREEDY)]),
             dict(rows=[(1, seq[13:14], 13, False), (2, other[:20], 0, False)], admit=[(2, GREEDY)]),
             dict(rows=[(1, seq[14:40], 14, True), (2, other[20:], 20, True)])]
    steps += [dict(rows=[(1, None, 40 + i, True), (2, None, 30 + i, True)]) for i in range(12)]
    res = drv.run(steps)
    gen1 = [int(r.tokens[0, 0]) for r in res[2:]]
    gen2 = [int(r.tokens[1, 0]) for r in res[2:]]
    for slot, prompt, gen in ((1, seq, gen1), (2, other, gen2)):
        full = prompt + gen
        ref = hf_logits(tiny, full)
        ours = logits_of(drv, slot, len(full) - 1)
        assert (ours - ref[:-1]).abs().max().item() < TOL, slot
        assert [int(x) for x in ref[len(prompt) - 1:len(full) - 1].argmax(-1)] == gen
    assert drv.reused and drv.switched  # released pages were reused, some as the other kind: 31 pages for both


def test_arena_pages_hold_either_kind(tiny) -> None:
    """The KV arena's layout (`runner.cuda.gemma._Arena`): window page p of every sliding layer and full page p of every
    full layer (its sub-pages) are the same bytes, chunk p of every slab, and no other page's."""
    r = gemma_cpu_runner(tiny, gemma_pools(budget=64, page_size=4, num_pages=8))
    a = r.arena
    assert a.page_size == r.pc.page_size == 4 * 4 and a.sub * a.groups == a.page_size
    with torch.inference_mode():
        a.mem.zero_()
        a.wkv[:, 3].fill_(1.0)  # window page 3 of every sliding layer
        full = a.fkv.reshape(a.fkv.shape[0], a.groups, a.num_pages, -1)  # [full layer, sub-page group, page, elements]
        assert bool((full[:, :, 3] == 1.0).all()), "full page 3 is not window page 3's bytes"
        assert float(full[:, :, [p for p in range(a.num_pages) if p != 3]].abs().sum()) == 0.0
    idx, pages = a.expand(np.asarray([0, 2]), np.asarray([5, 3]))
    assert idx.tolist() == [0, 1, 4, 5] and pages.tolist() == [5, 8 + 5, 3, 8 + 3]


def test_resume_shares_and_copies_both_pools(tiny) -> None:
    """Slot 2 resumes at 21 (inside a page of 4) from slot 1's KV: it shares the full pages and the window pages
    before 21 and copies the partial page of each pool; its logits match transformers on its own sequence."""
    a = ids(30, 3)
    b = a[:21] + ids(9, 4)
    pc = gemma_pools(budget=64, page_size=4)
    drv = GDriver(gemma_cpu_runner(tiny, pc), pc, window=W)
    drv.run([dict(rows=[(1, a[:21], 0, False)], admit=[(1, GREEDY)]),
             dict(rows=[(2, b[21:], 21, True), (1, a[21:], 21, True)], admit=[(2, GREEDY)], seq_len={2: 21},
                  share={2: (1, 21)})])
    for slot, s, first in ((1, a, 0), (2, b, 21)):
        ref = hf_logits(tiny, s)
        ours = torch.stack([drv.logits[(slot, p)] for p in range(first, len(s))])
        assert (ours - ref[first:]).abs().max().item() < TOL, slot


@pytest.mark.parametrize("fps, ps", [(16, 4), (32, 8), (4, 4), (8, 8)])
def test_the_warmup_resume_plans_keep_their_pages(fps, ps) -> None:
    """The warm-up's plans 4 and 5 (`GemmaRunner._warmup`): a prompt on pages from 1, then a resume that shares its
    pages before an off-page position and copies the partial page of each kind, on the pages after them. Their arrays
    are byte for byte the ones of the arithmetic inlined here (each row's full pages, then its window pages), for full
    pages longer than window pages and as long; so are `_page_layout`'s pages of several rows with room."""
    from llm_mini_infer.runner.cuda.gemma import GemmaRunner
    from llm_mini_infer.sched.plan import SAMPLE, STATE_ZERO, PlanBuilder

    budget = 256
    r = types.SimpleNamespace(fps=fps, ps=ps, cfg=types.SimpleNamespace(vocab_size=512))
    for name in ("_page_layout", "_synthetic_tokens"):
        setattr(r, name, getattr(GemmaRunner, name).__get__(r))

    def pages(n: int, room: int, first: int) -> tuple[list[int], list[int], int]:  # the oracle
        nf, nw = -(-(n + 2 + 2 + room) // fps), -(-(n + 2 + 2 + room) // ps)
        return list(range(first, first + nf)), list(range(first + nf, first + nf + nw)), first + nf + nw

    layout, after = r._page_layout([(1, 80), (2, 1), (3, 17)], pages_from=3, room=6)
    want, page = [], 3
    for n in (80, 1, 17):
        fp, wp, page = pages(n, 6, page)
        want.append((fp, wp))
    assert (layout, after) == (want, page)

    long = min(budget, fps + 3 if fps > ps else 2 * ps + 3)
    b = PlanBuilder(4, budget, fps, 1024, 0, W, ps)
    got = [GemmaRunner._synthetic(r, b, 4, [(1, long)], pages_from=1, free=[1, 2])]
    (src,), after = r._page_layout([(1, long)], pages_from=1)
    got.append(GemmaRunner._resume_plan(r, b, 5, slot=2, seq_len=long - 1, src=src, page=after))

    b = PlanBuilder(4, budget, fps, 1024, 0, W, ps)
    b.begin()
    b.add_free(1)
    b.add_free(2)
    b.add_admit(1, 1, 0, 0.0, 1.0, 0, 0.0, seed=1235)
    fp, wp, page = pages(long, 0, 1)
    b.add_pages(1, 0, fp)
    b.add_wpages(1, 0, wp)
    b.add_explicit_row(1, r._synthetic_tokens(1, long), 0, STATE_ZERO | SAMPLE)
    want = [b.finish(4)]
    seq_len = long - 1
    b.begin()
    b.add_admit(2, 2, seq_len, 0.0, 1.0, 0, 0.0, seed=99)
    for size, shared, add, copy in ((fps, fp, b.add_pages, b.add_kv_copy), (ps, wp, b.add_wpages, b.add_wkv_copy)):
        whole = seq_len // size
        own = list(range(page, page + -(-(seq_len + 5) // size) - whole))
        page += len(own)
        add(2, 0, shared[:whole] + own)
        if seq_len % size:
            copy(shared[whole], own[0])
    b.add_explicit_row(2, r._synthetic_tokens(2, 3, seq_len), seq_len, SAMPLE)
    want.append(b.finish(5))
    assert want[1].kv_copy_src.size and want[1].wkv_copy_src.size  # a partial page of each kind
    for g, w in zip(got, want):
        for name in type(g).__slots__:
            x, y = getattr(g, name), getattr(w, name)
            assert np.array_equal(x, y) and getattr(x, "dtype", None) == getattr(y, "dtype", None), name


def test_the_mirror_check_sees_device_tables_that_disagree(tiny) -> None:
    """`check_mirror` (dev.runner.cuda.check, which this runner shares with Qwen's) after a step that samples slot 1
    and admits slot 2 mid-prompt: a pending token for the slot that has not sampled, none for the one that has, or a
    seq_len outside the host mirror's range each raise."""
    from llm_mini_infer.runner.cuda.runner import check_mirror

    pc = gemma_pools(budget=64, page_size=4)
    r = gemma_cpu_runner(tiny, pc)
    GDriver(r, pc, window=W).run([dict(rows=[(1, ids(20, 5), 0, True), (2, ids(10, 6), 0, False)],
                                       admit=[(1, GREEDY), (2, GREEDY)])])
    tb, m = r.tables, r.mirror
    assert m.sampled[1] and not m.sampled[2] and m.admitted[1:3].all()
    check_mirror(tb, m, r.cuda)
    for table, slot, bad, what in ((tb.pending, 2, 5, "pending"), (tb.pending, 1, -1, "pending"),
                                   (tb.seq_len, 1, 99, "seq_len")):
        with torch.inference_mode():
            good = int(table[slot])
            table[slot] = bad
        with pytest.raises(AssertionError, match=what):
            check_mirror(tb, m, r.cuda)
        with torch.inference_mode():
            table[slot] = good
    check_mirror(tb, m, r.cuda)


def test_split_steps_with_masks_allowing_everything(tiny) -> None:
    """The grammar split path (forward, then sampling under masks) gives the unsplit tokens when the masks allow
    everything."""
    from llm_mini_infer.sched.plan import MASKED

    seq = ids(20, 5)
    outs = []
    for masked in (False, True):
        pc = gemma_pools(budget=64)
        r = gemma_cpu_runner(tiny, pc)
        drv = GDriver(r, pc, window=W)
        toks = []
        prev = None
        for i in range(6):
            plan = drv.plan(rows=[(1, seq, 0, True)] if i == 0 else [(1, None, 20 + i - 1, True)],
                            admit=[(1, GREEDY)] if i == 0 else ())
            if masked:
                plan.sched_flags |= MASKED
            h = r.launch_forward(plan)
            if prev is not None:
                toks.append(int(r.wait(prev).tokens[0, 0]))
            n = 1 if masked else 0
            if n:
                r.mask_buffer(h)[:n] = -1
            r.launch_sample(h, n)
            prev = h
        toks.append(int(r.wait(prev).tokens[0, 0]))
        outs.append(toks)
    assert outs[0] == outs[1]


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("hook", [False, True])
def test_rows_that_do_not_sample_report_nan(tiny, hook, masked) -> None:
    """An explicit row that does not sample (a prefill chunk before its prompt's last) reports NONFINITE in its own step
    when its last token's final hidden state is not finite, as it is when any KV the row wrote is NaN, so that KV never
    goes into the cache unseen. The sliding layers' RoPE is NaN at position 10, so q and K there are (`poisoned_rope`).
    The first step samples nothing: slot 1's chunk [0, 16) computes position 10 and reports it, slot 3's [0, 7) and
    slot 5's one token, short of it, report nothing. The second: slot 1's next chunk reads the NaN and slot 4's [0, 12)
    computes it, both report it; slot 2's prompt of 6 tokens samples as without the NaN, and slot 3's next token (a row
    the step orders first) reports nothing. With the table restored, the next step's rows report nothing (no flag
    lingers), and a runner that never saw the NaN reports nothing at all. `hook`: the logits hook's path; `masked`:
    the SAMPLE rows take masks that allow everything (the split path)."""
    a, b, c, d, e = ids(30, 11), ids(6, 12), ids(20, 13), ids(12, 14), ids(14, 15)
    steps = [dict(rows=[(1, a[:16], 0, False), (3, c[:7], 0, False), (5, e[:1], 0, False)],
                  admit=[(1, GREEDY), (3, GREEDY), (5, GREEDY)]),
             dict(rows=[(1, a[16:], 16, False), (4, d, 0, False), (2, b, 0, True), (3, c[7:8], 7, False)],
                  admit=[(2, GREEDY), (4, GREEDY)])]
    after = [dict(rows=[(2, None, 6, True), (5, e[1:], 1, False), (3, c[8:], 8, True)])]

    def run(poison: bool) -> tuple[list, GDriver]:
        pc = gemma_pools(max_num_seqs=6, budget=64, page_size=4, num_pages=64)
        drv = GDriver(gemma_cpu_runner(tiny, pc), pc, window=W)
        if not hook:
            drv.r.logits_hook = None
        with poisoned_rope(drv.r, POISON) if poison else nullcontext():
            res = drv.run(steps, masked=masked)
        assert bool(torch.isfinite(drv.r.rotary.cos[SLIDING]).all())
        return res + drv.run(after, masked=masked), drv

    (s1, s2, s3), drv = run(True)
    assert s1.status.tolist() == [NONFINITE, 0, 0] and not s1.n_out.any()
    assert s2.status.tolist() == [NONFINITE, NONFINITE, 0, 0] and s2.n_out.tolist() == [0, 0, 1, 0]
    assert s3.status is None
    clean_res, clean = run(False)
    assert all(r.status is None for r in clean_res)
    for x, y in zip((s1, s2, s3), clean_res):  # every token that samples
        assert np.array_equal(x.tokens, y.tokens) and np.array_equal(x.n_out, y.n_out)
    if hook:  # the rows short of the NaN (or past its restore), bit for bit
        for slot, n in ((2, 7), (3, 20), (5, 14)):
            assert all(torch.equal(drv.logits[(slot, p)], clean.logits[(slot, p)]) for p in range(n)), slot


def test_a_rows_flag_is_its_last_tokens(tiny) -> None:
    """`_check_rows` flags each row of the layout by its last token's final hidden state alone: a NaN there flags that
    row, one at its first token or inside it nothing (with the last token finite, that state went NaN after its last
    KV write: an earlier NaN reaches the last token through the last layer, full attention). The test above cannot
    tell the tokens apart: the CPU reference attention makes every query of a row NaN, those the mask hides the NaN key
    from included (SDPA adds the mask: NaN + −inf is NaN)."""
    pc = gemma_pools(max_num_seqs=6, budget=64, page_size=4, num_pages=64)
    drv = GDriver(gemma_cpu_runner(tiny, pc), pc, window=W)
    drv.run([dict(rows=[(1, ids(16), 0, False), (3, ids(7, 1), 0, False), (5, ids(1, 2), 0, False)],
                  admit=[(1, GREEDY), (3, GREEDY), (5, GREEDY)])])  # laid out as slot 5's token, 1's 16, 3's 7
    flags = []
    with torch.inference_mode():  # the static block is an inference tensor
        for tok in (0, 1, 8, 16, 17, 23):
            full = torch.zeros(24, drv.r.cfg.hidden_size)
            full[tok] = float("nan")
            drv.r._check_rows(full, 3)
            flags.append(drv.r.static.out_nf[:3].tolist())
    assert flags == [[1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1]]


@pytest.fixture(scope="module")
def tiny_fp8(tmp_path_factory):
    return make_tiny_gemma_fp8(tmp_path_factory.mktemp("g4r_fp8"), seed=3)[0]


def test_a_precise_piece_reports_nan(tiny_fp8) -> None:
    """The same for a refine's PRECISE pieces (BF16 activations; on CUDA, eager steps), which end at the FP8 checkpoint
    they refine and never sample: each reports its NaN (the first computes position 10, the second reads it), and the
    FP8 rows beside them, short of it, sample as without the NaN."""
    a, b = ids(30, 16), ids(6, 17)
    steps = [dict(rows=[(1, a[:20], 0, False, True), (2, b, 0, True)], admit=[(1, GREEDY), (2, GREEDY)]),
             dict(rows=[(2, None, 6, True), (1, a[20:], 20, False, True)])]

    def run(poison: bool) -> list:
        pc = gemma_pools(budget=64, page_size=4)
        drv = GDriver(gemma_cpu_runner(tiny_fp8, pc), pc, window=W)
        drv.r.logits_hook = None
        with poisoned_rope(drv.r, POISON) if poison else nullcontext():
            return drv.run(steps)

    nan, clean = run(True), run(False)
    assert [r.status.tolist() for r in nan] == [[NONFINITE, 0], [0, NONFINITE]]
    assert all(r.status is None for r in clean)
    assert all(np.array_equal(x.tokens, y.tokens) for x, y in zip(nan, clean))


def test_fp8_rows_turn_nan_finite(tiny_fp8) -> None:
    """FP8 rows, by contrast, do not report a NaN that their attention computes or reads: the activation quantizer
    leaves it out of the absmax and clamps it to −448 before the next projection (`fp8.quant_ref`, as the kernels
    do), so it never reaches the residual stream, on the CPU as on the GPU. Slot 1's FP8 chunk [0, 12) computes
    position 10 and its final chunk [12, 16) reads it in the window: neither reports it, the second samples, and slot
    2 beside them samples as without the NaN."""
    a, b = ids(16, 18), ids(6, 19)
    steps = [dict(rows=[(1, a[:12], 0, False), (2, b, 0, True)], admit=[(1, GREEDY), (2, GREEDY)]),
             dict(rows=[(2, None, 6, True), (1, a[12:], 12, True)])]

    def run(poison: bool) -> list:
        pc = gemma_pools(budget=64, page_size=4)
        drv = GDriver(gemma_cpu_runner(tiny_fp8, pc), pc, window=W)
        drv.r.logits_hook = None
        with poisoned_rope(drv.r, POISON) if poison else nullcontext():
            return drv.run(steps)

    nan, clean = run(True), run(False)
    assert all(r.status is None for r in nan + clean) and nan[1].n_out.tolist() == [1, 1]
    assert np.array_equal(nan[0].tokens, clean[0].tokens) and np.array_equal(nan[1].tokens[0], clean[1].tokens[0])


# -- contract violations -----------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def runner(tiny):
    return gemma_cpu_runner(tiny, gemma_pools(budget=64, page_size=4))


def _configure(runner):
    pc = gemma_pools(budget=64, page_size=4)
    runner.configure(pc)
    return pc


def _launch(runner, pc, *builds) -> None:
    """One plan per function in `builds`, in the engine's call order (launch N+1, wait N, sample N+1)."""
    b = PlanBuilder(pc.max_num_seqs, pc.max_num_batched_tokens, pc.page_size, pc.max_model_len, 0, W,
                    pc.window_page_size)
    prev = None
    for step, build in enumerate(builds, 1):
        b.begin()
        build(b)
        h = runner.launch_forward(b.finish(step))
        if prev is not None:
            runner.wait(prev)
        runner.launch_sample(h)
        prev = h
    if prev is not None:
        runner.wait(prev)


def _admit(b: PlanBuilder, slot: int, page: int) -> None:
    """ADMIT with full page `page` and window pages `page` + 1 and + 2 (one arena: distinct ids across slots)."""
    b.add_admit(slot, slot, 0, *GREEDY)
    b.add_pages(slot, 0, [page])
    b.add_wpages(slot, 0, [page + 1, page + 2])


def _row(b: PlanBuilder, slot: int) -> None:
    b.add_explicit_row(slot, np.arange(5, dtype=np.int32), 0, STATE_ZERO | SAMPLE)


@pytest.mark.parametrize(("builds", "match"), [
    # `seen[slots] += 1` counted a repeated index once, so the old check never saw a slot twice
    pytest.param([lambda b: (_admit(b, 1, 1), _row(b, 1), _row(b, 1))],
                 "a slot appears twice in SCHED", id="sched-slot-twice"),
    # slot -1 wrapped to the last slot (4), whose page table took the entry
    pytest.param([lambda b: (_admit(b, 1, 1), _admit(b, 4, 4), b.add_pages(-1, 1, [7]))],
                 "full PAGES out of range or for a slot that is not admitted", id="full-pages-slot-minus-1"),
    pytest.param([lambda b: (_admit(b, 1, 1), b.add_pages(5, 0, [7]))],
                 "full PAGES out of range or for a slot that is not admitted", id="full-pages-slot-past-max"),
    pytest.param([lambda b: (_admit(b, 1, 1), _admit(b, 4, 4), b.add_wpages(-1, 2, [7]))],
                 "window PAGES out of range or for a slot that is not admitted", id="window-pages-slot-minus-1"),
    pytest.param([lambda b: (_admit(b, 1, 1), b.add_wpages(5, 0, [7]))],
                 "window PAGES out of range or for a slot that is not admitted", id="window-pages-slot-past-max"),
    pytest.param([lambda b: _admit(b, 1, 1), lambda b: (b.add_free(1), b.add_free(1))],
                 r"a slot appears twice in FREE: \[1, 1\]", id="free-twice"),
    pytest.param([lambda b: (b.add_admit(1, 1, 0, *GREEDY), b.add_admit(1, 1, 0, *GREEDY), b.add_pages(1, 0, [1]))],
                 r"a slot appears twice in ADMIT: \[1, 1\]", id="admit-twice"),
])
def test_repeated_or_out_of_range_slots_raise(runner, builds, match) -> None:
    """A slot twice in SCHED, FREE or ADMIT, and a full or window PAGES entry for a slot outside 1..max_num_seqs each
    raise PlanError."""
    pc = _configure(runner)
    with pytest.raises(PlanError, match=match):
        _launch(runner, pc, *builds)
    _configure(runner)


@pytest.mark.parametrize("builds", [
    pytest.param([lambda b: (_admit(b, 1, 1), _row(b, 1)),
                  lambda b: (b.add_free(1), _admit(b, 1, 4), _row(b, 1))], id="free-and-admit-one-slot"),
    pytest.param([lambda b: (_admit(b, 1, 1), _row(b, 1)),
                  lambda b: (b.add_free(1), b.add_wpages(1, 0, [0]))], id="window-release-of-a-freed-slot"),
])
def test_slot_reuse_and_window_release_in_one_plan_pass(runner, builds) -> None:
    """The checks are within one list: a FREE and an ADMIT of one slot in one plan pass, and so do window PAGES of
    page id 0 (a release) for a slot the same plan FREEs."""
    pc = _configure(runner)
    _launch(runner, pc, *builds)
    _configure(runner)


# -- the engine end to end ----------------------------------------------------------------------------------------


def greedy_reference(model, prompt: list[int], n: int, eos: int) -> list[int]:
    out = []
    seq = list(prompt)
    with torch.no_grad():
        for _ in range(n):
            t = int(model(torch.tensor([seq])).logits[0, -1].argmax())
            out.append(t)
            if t == eos:
                break
            seq.append(t)
    return out


@pytest.mark.parametrize("budget", [16, 64])
def test_engine_with_window_cache_matches_transformers(tiny, budget) -> None:
    """Requests sharing prefixes longer than the window, a follow-up turn, chunked prefill (a budget of 16 or 64) and
    a KV arena small enough to evict windows and wait for pages: every request's greedy output equals transformers'
    generation, and every page comes back."""
    from llm_mini_infer.cache.pools import pages_for_one_request, window_pages_for_one_request

    hf = hf_model(tiny)
    eos = 511
    base = ids(36, 7)
    prompts = [base + ids(5, 10 + i) for i in range(4)] + [ids(9, 20)]
    max_tokens = [7, 9, 5, 11, 6]
    one = window_pages_for_one_request(W, budget, 4, 2) + pages_for_one_request(256, 16, 2, True)
    pc = gemma_pools(max_num_seqs=3, budget=budget, page_size=4, num_pages=2 * one + 1, max_model_len=256)
    runner = gemma_cpu_runner(tiny, pc)
    sched = Scheduler(max_num_seqs=3, max_num_batched_tokens=budget, max_model_len=256, page_size=pc.page_size,
                      num_pages=pc.num_pages, num_state_slots=4, eos_token_ids=(eos,), hybrid=False,
                      checkpoint_grid=4, decode_checkpoint_interval=8, window=W, window_page_size=4)
    core = EngineCore(sched, runner)
    outs: dict[int, list[int]] = {}
    fin: dict[int, int] = {}

    def add(rid: int, prompt: list[int], n: int) -> None:
        core.handle(AddRequest(rid=rid, frontend=0, prompt=np.asarray(prompt, np.int32), max_tokens=n,
                               temperature=0.0, top_p=1.0, top_k=0, min_p=0.0, seed=rid, stop_token_ids=[]))
        outs[rid] = []

    for rid, (p, n) in enumerate(zip(prompts, max_tokens)):
        add(rid, p, n)
    turns = 0
    while core.busy and turns < 500:
        rep = core.step()
        turns += 1
        if rep is None:
            continue
        for o in rep.outputs.values():
            off = 0
            for rid, k, f in zip(o.rids.tolist(), o.n_new.tolist(), o.finish.tolist()):
                outs[rid].extend(o.tokens[off:off + k].tolist())
                off += k
                if f:
                    fin[rid] = f
        if turns == 40 and 5 not in outs:  # a follow-up turn of request 0: its prompt, output and a new message
            add(5, prompts[0] + outs[0] + ids(4, 30), 6)
            prompts.append(prompts[0] + outs[0] + ids(4, 30))
            max_tokens.append(6)
    assert not core.busy
    for rid in outs:
        ref = greedy_reference(hf, prompts[rid], max_tokens[rid], eos)
        assert outs[rid] == ref, rid
        assert fin[rid] in (FinishReason.STOP, FinishReason.LENGTH)
    assert sched.prefix_hits > 0
    sched.check_accounting()
    sched.cache.clear()
    sched.check_accounting()
    assert sched.kv.num_free == sched.kv.num_pages - 1


def test_step_rows_fit_a_request() -> None:
    """The profile's and the graph capture's full-budget steps (both CUDA runners): one row while a request can be that
    long, else rows a request can hold, at most max_num_seqs: a step budget past max_model_len as one row would not
    fit a slot's page table, and the runner would fail to start."""
    from llm_mini_infer.runner.cuda.runner import step_rows

    assert step_rows(2048, 4096, 64) == [2048]
    assert step_rows(4096, 4096, 64) == [4096]  # still one row: a request can be max_model_len long
    rows = step_rows(8192, 4096, 64)
    assert rows == [2731, 2731, 2730] and sum(rows) == 8192 and max(rows) <= 4096 - 64
    assert step_rows(8192, 4096, 2) == [4032, 4032]  # no two requests fill it: the step two can fill
    assert all(sum(step_rows(t, 300, 64)) == t and max(step_rows(t, 300, 64)) <= 300 for t in range(1, 5000, 37))
