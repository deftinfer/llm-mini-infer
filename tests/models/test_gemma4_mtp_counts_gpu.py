"""The draft count per step on CUDA (`dev.speculative.adaptive`; `test_gemma4_mtp_counts` on CPU): decode graphs per
(v, d) (`GemmaRunner._key`), each count's attention backends, and the raise steps that draft k, through the scheduler
in window mode on `gemma_gpu`'s tiny pair (the real head dims, a window of 40). Forced schedules of counts in graphs
give the eager tokens and a fixed k's, up to a near tie of transformers' fp32 logits (`gemma_gpu.agree`); the
counters add up. Their bits beside other requests: `tests/runner/test_batch_invariance_gpu.py`. CUDA only."""

from __future__ import annotations

import numpy as np
import pytest

from ..support.gemma_gpu import W, agree, gpu_run, reference, run_prompts, to_bf16
from ..support.gemma_mtp import counters
from ..support.tiny_gemma import make_tiny_assistant, make_tiny_gemma

pytestmark = pytest.mark.gpu(reason="needs a CUDA GPU and the kernel library")

K = 5
COUNTS = (1, 2, 3)


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    d = tmp_path_factory.mktemp("g4countsgpu")
    target = make_tiny_gemma(d / "target", window=W, gpu=True, seed=3)
    to_bf16(target, ("q_norm.weight", "k_norm.weight"))
    assistant = make_tiny_assistant(d / "assistant", target, hidden=256, intermediate=512, seed=4)
    to_bf16(assistant, ("q_norm.weight",))
    return target, assistant


@pytest.fixture(scope="module")
def fixed(pair):
    """k = 5 at every step (graphs): its tokens, transformers' fp32 logits along them, and how many there are."""
    r = run_prompts(gpu_run(pair, K, draft_counts=COUNTS))
    return r.out, reference(pair[0], r.out), sum(len(o) for o in r.out.values())


@pytest.mark.parametrize("schedule", [(1,), (2, 1), (1, 5), (5, 1, 3, 2)])
def test_schedules_in_graphs_give_the_eager_and_a_fixed_ks_tokens(pair, fixed, schedule) -> None:
    """Every count's decode graphs replay (the raise steps' at (v, k) too); the same tokens eagerly, in graphs and at a
    fixed k, up to a near tie, with the assistant's drafts and an oracle's (a random number right); every step drafts
    what its plan says, −1 past it (`Run.check_drafts`: a graph replayed at another d fails it)."""
    out, ref, n = fixed
    eager = run_prompts(gpu_run(pair, K, graphs=False, draft_counts=COUNTS, count_schedule=schedule).check_drafts())
    assert "decode" not in eager.modes
    graphs = run_prompts(gpu_run(pair, K, draft_counts=COUNTS, count_schedule=schedule).check_drafts())
    assert graphs.modes.count("decode") > 20
    assert agree(graphs.out, eager.out, reference(pair[0], eager.out)) > 0.8 * n
    assert agree(graphs.out, out, ref) > 0.8 * n
    oracle = run_prompts(gpu_run(pair, K, draft_counts=COUNTS, count_schedule=schedule).check_drafts(), out,
                         np.random.default_rng(len(schedule)))
    assert agree(oracle.out, out, ref) > 0.8 * n and oracle.sched.spec_accepted > 0
    for r in (eager, graphs, oracle):
        counters(r.sched, K)
        assert r.drafts_checked.get(K) and any(r.drafts_checked.get(d) for d in COUNTS), r.drafts_checked


@pytest.mark.parametrize("schedule", [(1,), (2, 1)])
def test_a_tight_budget_runs_a_lower_counts_extra_rows_in_graphs(pair, fixed, schedule) -> None:
    """A budget of 20 tokens holds 3 decode rows at k (3 × 6) and all 4 below it: each count's own decode buckets
    reach 4 rows (`GemmaRunner._dbuckets`), so its 4-row steps replay graphs instead of running eagerly; the tokens are
    the eager ones and a fixed k's, up to a near tie."""
    out, ref, n = fixed
    graphs = run_prompts(gpu_run(pair, K, draft_counts=COUNTS, count_schedule=schedule, budget=20).check_drafts())
    assert max(graphs.r.decode_buckets) == 3 and all(max(graphs.r._dbuckets(v)) == 4 for v in COUNTS)
    four = [m for m, rows, v in graphs.decode_rows if rows == 4 and v < K]
    assert len(four) > 10 and set(four) == {"decode"}, graphs.decode_rows
    eager = run_prompts(gpu_run(pair, K, graphs=False, draft_counts=COUNTS, count_schedule=schedule, budget=20))
    assert agree(graphs.out, eager.out, reference(pair[0], eager.out)) > 0.8 * n
    assert agree(graphs.out, out, ref) > 0.8 * n
    counters(graphs.sched, K)


def test_the_policy_in_graphs_gives_a_fixed_ks_tokens(pair) -> None:
    r = run_prompts(gpu_run(pair, K, draft_counts=COUNTS, adaptive=True), n=60)
    want = run_prompts(gpu_run(pair, K, draft_counts=COUNTS), n=60)
    assert agree(r.out, want.out, reference(pair[0], want.out)) > 0.8 * sum(len(o) for o in want.out.values())
    assert sum(1 for c in r.sched.count_policy.chosen.values() if c) >= 2
    counters(r.sched, K)


def test_the_graphs_of_every_count_are_captured(pair) -> None:
    r = gpu_run(pair, K, draft_counts=COUNTS)
    counts = r.r.stats["graphs"]["counts"]
    assert set(counts) == {*COUNTS, K} and all(counts.values())
    r.r.shutdown()
