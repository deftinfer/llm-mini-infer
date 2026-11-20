"""The draft count per step (`dev.speculative.adaptive`, `sched.draft_counts`) in GemmaRunner on CPU (the torch
reference backends, fp32): whatever count each step verifies and drafts, a greedy or seeded request gives the tokens it
gives at a fixed k (architecture.md, "Speculative decoding"). Schedules of counts are forced (`count_schedule`):
lowering, raising (a step that verifies what its rows hold and drafts k), and the policy's own choices; with the
assistant's drafts and an oracle's (`Run._oracle`)."""

from __future__ import annotations

import numpy as np
import pytest

from llm_mini_infer.engine.messages import FinishReason

from ..support.gemma_mtp import PROMPTS, Run, W, counters
from ..support.mtp_common import EOS, seq
from ..support.tiny_gemma import make_tiny_assistant, make_tiny_gemma

K = 5
COUNTS = (1, 2, 3)
SCHEDULES = [(1,), (2, 1), (1, 5), (5, 1, 3, 2), (3, 1, 1, 5, 2)]


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    d = tmp_path_factory.mktemp("g4counts")
    target = make_tiny_gemma(d / "target", window=W, seed=5)
    return target, make_tiny_assistant(d / "assistant", target, seed=6)


def run(pair, k: int, reqs: list[dict], oracle=None, rng=None, **kw) -> Run:
    r = Run(pair, k, **kw)
    for q in reqs:
        r.add(**q)
    return r.run(oracle, rng)


def greedy(max_tokens: int = 18, **extra) -> list[dict]:
    return [dict(rid=i, prompt=p, max_tokens=max_tokens, **extra) for i, p in enumerate(PROMPTS)]


@pytest.mark.parametrize("schedule", SCHEDULES)
def test_greedy_outputs_equal_under_any_schedule_of_counts(pair, schedule) -> None:
    """MTP off's greedy outputs under every schedule, with the assistant's drafts and the oracle's (all right, and a
    random number right): no step reads a draft its rows do not hold; the counters count what each step verified."""
    ref = run(pair, 0, greedy()).out
    for oracle, rng in ((None, None), (ref, None), (ref, np.random.default_rng(len(schedule)))):
        r = run(pair, K, greedy(), oracle, rng, draft_counts=COUNTS, count_schedule=schedule)
        assert r.out == ref, (schedule, oracle is not None, rng is not None)
        counters(r.sched, K)
        if set(schedule) - {K}:
            assert r.sched.spec_verified_per_pos[-1] < r.sched.spec_drafts  # some steps verified fewer than k


@pytest.mark.parametrize("params", [(0.8, 1.0, 0), (1.0, 0.9, 20)])
def test_seeded_sampling_outputs_equal_under_any_schedule_of_counts(pair, params) -> None:
    temp, top_p, top_k = params
    reqs = [dict(rid=i, prompt=p, max_tokens=16, temperature=temp, top_p=top_p, top_k=top_k, seed=4_000 + i)
            for i, p in enumerate(PROMPTS)]
    ref = run(pair, 0, reqs).out
    for schedule in SCHEDULES[2:]:
        assert run(pair, K, reqs, draft_counts=COUNTS, count_schedule=schedule).out == ref, schedule
        assert run(pair, K, reqs, ref, np.random.default_rng(3), draft_counts=COUNTS,
                   count_schedule=schedule).out == ref, schedule


def test_stop_tokens_and_max_tokens_under_a_schedule(pair) -> None:
    """The commit limits end the accepted run at any count: a stop token among the oracle's drafts, and max_tokens
    reached inside a verify window."""
    base = run(pair, 0, [dict(rid=0, prompt=PROMPTS[1], max_tokens=20)])
    stop = base.out[0][6]
    reqs = [dict(rid=0, prompt=PROMPTS[1], max_tokens=20, stop=[stop]), dict(rid=1, prompt=PROMPTS[3], max_tokens=7),
            dict(rid=2, prompt=PROMPTS[0], max_tokens=1)]
    want = run(pair, 0, reqs)
    for schedule in ((1, 5), (3, 2, 5)):
        r = run(pair, K, reqs, {0: base.out[0]}, draft_counts=COUNTS, count_schedule=schedule)
        assert r.out == want.out and r.finish == want.finish, schedule
        assert r.out[0][-1] == stop and r.finish[0] == FinishReason.STOP and len(r.out[1]) == 7


@pytest.mark.parametrize("params", [(0.0, 1.0), (0.8, 0.9)])
def test_masked_outputs_equal_under_a_schedule_of_counts(pair, params) -> None:
    """Structured output: a verify row samples its 1 + v positions under the masks along its first v drafts, whatever
    v is (`Scheduler.fill_masks`); constrained and unconstrained requests share steps; outputs equal MTP off's."""
    from llm_mini_infer.grammar.engine import Grammars

    from ..support.fake_grammar import FakeGrammar, spec

    temp, top_p = params
    reqs = [dict(rid=0, prompt=seq(20, 0), grammar=spec(1, 4, 9)), dict(rid=1, prompt=seq(33, 1)),
            dict(rid=2, prompt=seq(9, 2), grammar=spec(2, 30, 40), max_tokens=12),
            dict(rid=3, prompt=seq(50, 3), grammar=spec(3, 0, 6))]
    reqs = [dict(rid=q["rid"], prompt=q["prompt"], max_tokens=q.get("max_tokens", 14), temperature=temp, top_p=top_p,
                 ignore_eos=False, grammar=q.get("grammar", "")) for q in reqs]
    outs = []
    for k, schedule in ((0, ()), (K, (1, 3)), (K, (5, 2, 1)), (K, (2,))):
        grammar = FakeGrammar(512, (EOS,))
        r = run(pair, k, reqs, grammars=Grammars(grammar), caching=False,
                draft_counts=COUNTS if k else (), count_schedule=schedule)
        assert r.sched.grammars.errors == 0
        outs.append((r.out, r.finish))
    assert all(o == outs[0] for o in outs[1:])
    assert outs[0][1][0] == FinishReason.STOP and outs[0][0][0][-1] == EOS


def test_counts_change_across_a_preemption_and_a_readmission(pair) -> None:
    """Requests preempted mid-decode (with a decode step in flight) and readmitted from the cache while the counts
    change every step: a readmitted slot holds no drafts until its final chunk's SAMPLE row drafts (`Scheduler.held`),
    and no step verifies more than its rows hold (the runner checks it); the outputs are MTP off's."""
    from llm_mini_infer.sched.scheduler import RUNNING

    reqs = [dict(rid=i, prompt=p, max_tokens=30) for i, p in enumerate(PROMPTS)]
    ref = run(pair, 0, reqs).out
    for schedule in ((1, 5, 2, 3), (2,)):
        r = Run(pair, K, draft_counts=COUNTS, count_schedule=schedule)
        for q in reqs:
            r.add(**q)
        plan, turn = r.sched._schedule, [0]

        def preempting(plan=plan, turn=turn, r=r):  # before the plan's rows, inside its construction
            turn[0] += 1
            for rid, at in ((0, 6), (2, 9), (0, 15)):
                req = r.sched.reqs.get((0, rid))
                if at == turn[0] and req is not None and req.status == RUNNING:
                    r.sched._preempt(req.slot)
            return plan()

        r.sched._schedule = preempting
        r.run()
        assert r.sched.num_preemptions >= 2 and r.out == ref, schedule


def test_the_policy_gives_fixed_ks_outputs(pair) -> None:
    """The scheduler's policy (`sched.draft_counts.DraftCounts`) choosing among the counts, probing each: the outputs
    of a fixed k, and every count run at least once."""
    ref = run(pair, 0, greedy(max_tokens=40)).out
    r = run(pair, K, greedy(max_tokens=40), draft_counts=COUNTS, adaptive=True)
    assert r.out == ref
    chosen = r.sched.count_policy.chosen
    assert chosen[K] and sum(1 for n in chosen.values() if n) >= 2, chosen  # k first, then its probes
    counters(r.sched, K)
