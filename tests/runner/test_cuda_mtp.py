"""MTP in the CUDA runner on CPU (torch reference kernels, fp32; architecture.md, "Speculative decoding"):
verification with the sampler's own noise, the drafter and its chain, GDN rollback by replay (window records, the fold
at the next decode step, FOLD, folded checkpoints), the drafter's KV at a resume, through the real scheduler and
engine loop.

The tiny model's MTP layer is random, so its drafts are mostly rejected; an **oracle** that overwrites the drafts
between steps with the true continuation (all or a random number of them correct) drives every accepted count. The
promise under test: a request's tokens are the target's sampled sequence whatever the drafts, so MTP on and off give
the same output, greedy and seeded alike.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from llm_mini_infer.engine.messages import FinishReason
from llm_mini_infer.grammar.engine import Grammars
from llm_mini_infer.sched.plan import SAMPLE, reserve_positions

from ..support.fake_grammar import FakeGrammar
from ..support.mtp_common import EOS, draft_positions_spy, seq
from ..support.qwen_mtp import Run
from ..support.qwen_mtp import model as model  # the fixture


@pytest.mark.parametrize("k", [1, 2, 3])
def test_greedy_outputs_equal_without_mtp(model, k) -> None:
    prompts = [seq(n, i) for i, n in enumerate([5, 17, 40, 1])]
    off = Run(model, 0)
    for i, p in enumerate(prompts):
        off.add(i, p, max_tokens=14)
    off.run()
    for oracle, rng in ((None, None), (off.out, None), (off.out, np.random.default_rng(k))):
        on = Run(model, k)
        for i, p in enumerate(prompts):
            on.add(i, p, max_tokens=14)
        on.run(oracle, rng)
        assert on.out == off.out, (oracle is not None, rng is not None)
        if oracle is not None:
            assert on.sched.spec_accepted > 0
    assert on.r.mirror.slack.max() == 0


@pytest.mark.parametrize("params", [(0.8, 1.0, 0), (1.0, 0.9, 0), (0.7, 1.0, 20)])
def test_seeded_sampling_outputs_equal_without_mtp(model, params) -> None:
    """The coupling's promise: a seeded request samples the same tokens with MTP on and off (each draft and target
    sample at a position uses that position's noise)."""
    temp, top_p, top_k = params
    prompts = [seq(n, 7 + i) for i, n in enumerate([9, 30, 3])]
    runs = []
    for k in (0, 3):
        r = Run(model, k)
        for i, p in enumerate(prompts):
            r.add(i, p, max_tokens=16, temperature=temp, top_p=top_p, top_k=top_k, seed=4_000 + i)
        runs.append(r.run())
    oracle = Run(model, 3)
    for i, p in enumerate(prompts):
        oracle.add(i, p, max_tokens=16, temperature=temp, top_p=top_p, top_k=top_k, seed=4_000 + i)
    oracle.run(runs[0].out, np.random.default_rng(1))
    assert runs[1].out == runs[0].out == oracle.out
    assert oracle.sched.spec_accepted > 0


def test_stop_token_ends_the_accepted_run(model) -> None:
    """Every draft right: a stop token inside the run is the last token of its step (the commit limit), and the
    request ends there exactly as without MTP."""
    p = seq(20, 3)
    off = Run(model, 0)
    off.add(0, p, max_tokens=30)
    off.run()
    stop = off.out[0][9]
    first = off.out[0].index(stop)
    for k in (0, 3):
        r = Run(model, k)
        r.add(0, p, max_tokens=30, stop=[stop])
        r.run({0: off.out[0]} if k else None)
        assert r.out[0] == off.out[0][:first + 1] and r.finish[0] == FinishReason.STOP


def test_cache_hits_resume_exactly_with_the_drafter(model) -> None:
    """Prefix-cache hits under MTP (FOLD on donations, folded decode checkpoints, the drafter's KV recomputed at the
    hit, the page holding it copied on a page boundary): requests sharing prefixes and follow-up turns give the outputs
    they give cold without MTP."""
    base = seq(130, 11)
    first = [base + seq(5, 1), base + seq(9, 2)]
    off = Run(model, 0, budget=256)
    for i, p in enumerate(first):
        off.add(i, p, max_tokens=20)
    off.run()
    turns = [first[0] + off.out[0] + seq(4, 5), first[1] + off.out[1][:13] + seq(6, 6), base + seq(3, 9)]
    cold = Run(model, 0, budget=256)
    for i, p in enumerate(turns):
        cold.add(10 + i, p, max_tokens=12)
    cold.run()
    for oracle in (False, True):
        on = Run(model, 3, caching=True, ckpt_slots=8, interval=64, budget=256)
        for i, p in enumerate(first):
            on.add(i, p, max_tokens=20)
        on.run({i: off.out[i] for i in range(2)} if oracle else None)
        assert on.out == off.out
        for i, p in enumerate(turns):
            on.add(10 + i, p, max_tokens=12)
        on.run({10 + i: cold.out[10 + i] for i in range(3)} if oracle else None)
        assert {r: on.out[r] for r in cold.out} == cold.out
        assert on.cached[10] >= len(first[0]) and on.cached[11] >= 128 and on.cached[12] == 128
        assert on.sched.ckpts_taken["finish"]


def test_preemption_with_decode_rows_in_flight(model) -> None:
    """A pool for about two requests: preemption folds the donated state one record into the in-flight decode row,
    the readmission resumes from it (the drafter's KV recomputed at the resume)."""
    prompts = [seq(30, 20 + i) for i in range(3)]
    off = Run(model, 0)
    for i, p in enumerate(prompts):
        off.add(i, p, max_tokens=40)
    off.run()
    on = Run(model, 3, caching=True, num_pages=1 + 52, max_num_seqs=3, ckpt_slots=4, interval=64)
    for i, p in enumerate(prompts):
        on.add(i, p, max_tokens=40)
    on.run({i: off.out[i] for i in range(3)}, np.random.default_rng(5))
    assert on.sched.num_preemptions > 0
    assert on.out == off.out


def _drafter_trace(model, prompts: list[list[int]], rid_of_interest: int, caching: bool) -> tuple[dict, torch.Tensor]:
    """Run the prompts in order (each to completion: the later ones hit the earlier ones' cache when on); for the
    request of interest, its drafts after each step keyed by seq_len, and the drafter's KV over its prompt once its
    prefill is done."""
    r = Run(model, 3, caching=caching, ckpt_slots=8, interval=64, budget=256)
    drafts: dict[int, tuple[int, ...]] = {}
    kv = None
    r.cached = {}
    layer = r.r.model.mtp.layers[0].self_attn.index
    ps = r.r.pools.page_size
    for i, p in enumerate(prompts):
        r.add(i, p, max_tokens=12)
        while r.core.busy:
            rep = r.core.step()
            if rep is not None:
                for out in rep.outputs.values():
                    r.cached.update(zip(out.first_rids.tolist(), out.first_cached.tolist()))
                    off = 0
                    for rid, nn in zip(out.rids.tolist(), out.n_new.tolist()):
                        r.out[rid].extend(out.tokens[off:off + nn].tolist())
                        off += nn
            slot = next((s for s in range(1, len(r.sched.slot_req)) if r.sched.slot_req[s] is not None
                         and r.sched.slot_req[s].rid == rid_of_interest), None)
            if slot is None or not r.out.get(rid_of_interest):
                continue
            tb = r.r.tables
            drafts[int(tb.seq_len[slot])] = tuple(tb.drafts[slot].tolist())
            if kv is None:  # the prompt's drafter KV (positions [0, len - 1): the last one waits for the sample)
                n = len(prompts[rid_of_interest]) - 1
                pos = torch.arange(n)
                pages = tb.page_table[slot, pos // ps].long()
                kv = r.r.pools.kv[layer][pages, :, pos % ps].clone()
    assert not caching or r.cached[rid_of_interest] == 128, r.cached
    return drafts, kv


def test_drafter_kv_and_drafts_at_a_cache_hit(model) -> None:
    """A request resumed from the prefix cache has the drafter state it has cold: the drafter's KV over its prompt
    (the donor's pages, and the position before the hit recomputed from the checkpoint's hidden vector and the
    request's own next token, in a private copy of the donor's page) and the same drafts at every position. The two
    prompts part exactly at the hit (128), so the donor's drafter KV at 127 is wrong for the second. (Outputs alone
    cannot show any of this: wrong drafts only lower acceptance.)"""
    base = seq(130, 11)
    a, b = base + seq(5, 1), base[:128] + seq(11, 2)
    assert a[128] != b[128]
    cold, kv_cold = _drafter_trace(model, [b], 0, caching=False)
    hit, kv_hit = _drafter_trace(model, [a, b], 1, caching=True)
    assert kv_cold is not None and kv_hit is not None
    torch.testing.assert_close(kv_hit, kv_cold, atol=1e-5, rtol=1e-5)
    common = sorted(set(cold) & set(hit))
    assert len(common) >= 3
    assert all(cold[s] == hit[s] for s in common), [(s, cold[s], hit[s]) for s in common if cold[s] != hit[s]]


@pytest.mark.parametrize("params", [(0.0, 1.0, 0), (0.8, 1.0, 0), (1.0, 0.9, 20)])
def test_outputs_equal_with_a_reduced_draft_head(model, params) -> None:
    """The drafter scoring 128 of the 512 tokens (`DraftHead`) drafts only those tokens, accepts as the tokens allow,
    and the outputs stay the target's, greedy and seeded alike."""
    temp, top_p, top_k = params
    prompts = [seq(n, 3 + i) for i, n in enumerate([11, 26, 4])]
    runs = []
    for k, kw in ((0, {}), (3, {"draft_vocab": 128})):
        r = Run(model, k, **kw)
        for i, p in enumerate(prompts):
            r.add(i, p, max_tokens=16, temperature=temp, top_p=top_p, top_k=top_k, seed=7_000 + i)
        runs.append(r.run())
    assert runs[1].out == runs[0].out
    head = runs[1].r.draft_head
    assert head.reduced and head.size == 128
    ids = set(head.ids.tolist())
    assert set(runs[1].r.tables.drafts[1:].flatten().tolist()) <= ids | {0}


def test_draft_head_ranking() -> None:
    """The ranking: before traffic, the lowest ids and the special tokens; emitted tokens outside the head are counted
    and, past the miss tolerance, ranked in at the next check and installed by `apply`; the logits are the head's
    rows."""
    from llm_mini_infer.runner.cuda import draft_head as dh

    torch.manual_seed(2)
    head = torch.randn(1024, 128) * 0.1
    d = dh.DraftHead(head, size=256, special=[1000, 1001])
    assert d.reduced and d.ids.tolist() == [*range(254), 1000, 1001]
    hidden = torch.randn(3, 128)
    assert torch.equal(d.logits(hidden), torch.nn.functional.linear(hidden, head[d.ids.long()]).float())
    burst = np.repeat(np.arange(600, 700), dh.CHECK_EVERY // 100 + 1)
    d.observe(burst)
    assert d.apply() and d.rebuilds == 1
    new = d.ids.tolist()
    assert set(range(600, 700)) <= set(new) and {1000, 1001} <= set(new) and new == sorted(new) and len(new) == 256
    assert torch.equal(d.logits(hidden), torch.nn.functional.linear(hidden, head[d.ids.long()]).float())
    d.observe(burst)  # nothing missed now: no new ranking
    assert not d.apply()


def test_drafts_take_the_noise_of_the_positions_they_fill(model) -> None:
    """The coupled verification (architecture.md, "Speculative decoding") needs a draft and the target's sample at the
    same position to share that position's noise: the sampler keys every draft at the position it fills (verified next
    step against the target's sample there), not one before. Outputs cannot show this (a wrongly keyed draft is only
    rejected more often at temperature > 0)."""
    r = Run(model, 3)
    seen = draft_positions_spy(r.r, 3)
    for i, p in enumerate([seq(9, 1), seq(20, 2)]):
        r.add(i, p, max_tokens=8, temperature=1.0, seed=50 + i)
    r.run()
    assert len(seen) >= 6
    for keyed, fill in seen:
        assert keyed == fill, (keyed, fill)


class NullPage:
    """Page 0 is the null page: the page-table entry of every position a slot holds no page for, and where a CUDA
    graph's padding rows (slot 0) write their KV, in the same kernels as the step's real rows. So no real row may write
    or read it: what it would read there races. The watch poisons the null page of every KV layer with NaN and records
    (`events`) each real row (slot > 0) whose attention, the target's or the drafter's chain's, reads it and, after
    every step, each layer whose null page a write changed (eagerly on a CPU every row is a real one). `crossings`: the
    SAMPLE explicit rows whose drafter chain (k − 1 positions past the row) reaches a page past the row's last."""

    def __init__(self, run: Run):
        r = run.r
        ps, k = r.pools.page_size, run.k
        self.events: list[tuple] = []
        self.crossings = self.steps = 0
        kv = r.pools.kv

        @torch.inference_mode()
        def poison() -> None:
            kv[:, 0] = float("nan")

        poison()
        launch, step = r.launch_forward, run.core.step

        def launch_forward(plan, *a, **kw):
            self.steps += 1
            final = (plan.sched_tok_off >= 0) & ((plan.sched_flags & SAMPLE) != 0)
            end = (plan.sched_pos0 + plan.sched_n_query)[final].astype(np.int64)
            self.crossings += int(((end + k - 2) // ps > (end - 1) // ps).sum()) if k > 1 else 0
            return launch(plan, *a, **kw)

        def core_step(*a, **kw):
            rep = step(*a, **kw)
            written = ~torch.isnan(kv[:, 0]).flatten(1).all(1)
            if written.any():
                self.events.append(("write", self.steps, written.nonzero().flatten().tolist()))
                poison()
            return rep

        def reads(name: str, attend):
            def spy(meta, index, q, out):
                lengths = meta.kv_len_dev.tolist() if meta.kv_len_dev is not None else meta.row_kv_len_host
                for slot, n in zip(np.asarray(meta.row_slot_host).tolist(), lengths):
                    if slot > 0 and (meta.page_table[slot, :-(-int(n) // ps)] == 0).any():
                        self.events.append(("read", self.steps, name, index, slot, int(n)))
                return attend(meta, index, q, out)

            return spy

        for name in ("attn", "chain_attn"):
            att = getattr(r, name)
            if att is not None:
                att.run = reads(name, att.run)
        r.launch_forward, run.core.step = launch_forward, core_step


@pytest.mark.parametrize("constrained", [False, True])
def test_the_drafter_chain_never_reaches_the_null_page(model, constrained) -> None:
    """The always-split GPU test's workload (`qwen_mtp.requests`: six requests, the first's prompt in
    three chunks, k = 3, pages of 4; free, or every other one under a grammar whose forced prefix joins its final
    chunk): final chunks ending at 35 or 48, among others, where the drafter's chain (the MTP layer's KV at the k − 1
    positions past the row, which its attention reads) crosses into a page past the row's. The chunk's pages cover it:
    with entry 0 there, the chain wrote and read the null page, which a CUDA graph's padding rows write in the same
    kernel, so on a GPU the drafts it gave changed from run to run."""
    from ..support.qwen_mtp import BUDGET, requests

    r = Run(model, 3, budget=BUDGET, grammars=Grammars(FakeGrammar(512, (EOS,))), forced_prefix=True,
            eos_free_finish=True)
    watch = NullPage(r)
    for q in requests(constrained):
        r.add(q["rid"], q["prompt"], max_tokens=q["max_tokens"], ignore_eos=False, grammar=q.get("grammar", ""))
    r.run()
    assert r.sched.grammars.errors == 0 and (r.sched.forced_requests > 0) == constrained
    assert watch.crossings >= 2 and not watch.events, watch.events


@pytest.mark.parametrize("k, page_size, budget, caching, tight", [
    (3, 4, 32, True, False), (2, 8, 48, True, False), (3, 16, 24, False, False), (1, 4, 64, True, False),
    (3, 4, 48, True, True), (2, 8, 32, True, True)])
def test_no_real_row_touches_the_null_page(model, k, page_size, budget, caching, tight) -> None:
    """Workloads drawn around the drafter's geometry: prompts ending on a page boundary, just before one, or anywhere,
    chunked by the budget; follow-up requests that hit the prefix cache (the drafter's KV recomputed at the hit); with
    `tight`, a pool small enough to preempt (a readmission's final chunk ends where its tokens do); an oracle accepting
    a random number of drafts. No real row writes or reads the null page, and the outputs are k = 0's."""
    rng = np.random.default_rng(k * 1_000 + page_size * 10 + budget)
    first = []
    for i in range(4):
        n = page_size * int(rng.integers(2, 80 // page_size + 2)) - int(rng.choice([0, 1, rng.integers(0, page_size)]))
        first.append(seq(max(n, 2), 40 + i))
    tokens = 32 if tight else 12
    need = [-(-(len(p) + tokens + 12 + reserve_positions(2, k)) // page_size) for p in first]  # a follow-up's too
    pages = {"num_pages": 1 + max(max(need), 7 * sum(sorted(need)[-3:]) // 10)} if tight else {}
    runs = []
    for kk in (0, k):
        r = Run(model, kk, budget=budget, page_size=page_size, caching=caching, ckpt_slots=8, interval=32,
                max_num_seqs=3, **pages)
        watch = NullPage(r) if kk else None
        for i, p in enumerate(first):
            r.add(i, p, max_tokens=tokens)
        r.run(runs[0].out if kk else None, rng if kk else None)
        for i, p in enumerate(first):  # follow-up turns: the first ones' prompts and outputs, and a few more tokens
            r.add(10 + i, p + r.out[i] + seq(1 + i, 60 + i), max_tokens=8)
        r.run(runs[0].out if kk else None, rng if kk else None)
        runs.append(r)
    off, on = runs
    assert on.out == off.out
    assert not watch.events, watch.events
    assert watch.crossings > 0 or k == 1
    if caching and not tight:  # (a tight pool may evict the first turns' prefixes before their follow-ups)
        assert any(on.cached[10 + i] for i in range(4))
    assert (on.sched.num_preemptions > 0) == tight
