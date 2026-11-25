"""MTP through the scheduler and the validating fake runner (architecture.md, "The step plan", "The cache").

The fake verifies 1 + k tokens per decode row and accepts a number of drafts drawn from (seed, position), capped by
the request's commit limits, so every request's output must still be exactly its predicted token sequence, delivered
several tokens per step. The fake commits the recurrent state lazily (window records, FOLD) as the CUDA runner does,
and checks that every resume finds a state of exactly its position and prefix: a wrong FOLD count, a decode
checkpoint at the wrong position or a donation past a finish raises `PlanError` or changes later tokens.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given

from llm_mini_infer.engine.messages import FinishReason
from llm_mini_infer.sched.plan import STOP_W

from ..support.fuzzing import campaign_settings, saved_on_failure
from ..support.harness import Harness, Spec, prompt_of
from ..support.mtp_workloads import mtp_count_workloads, mtp_workloads, output_of, run_mtp_workload
from ..support.sched_cases import VOCAB, check_plans, stem


def run(specs, **kw) -> Harness:
    kw.setdefault("vocab", VOCAB)
    h = Harness(specs=specs, **kw)
    h.run(check_every_turn=True)
    check_plans(h)
    h.check()
    return h


@pytest.mark.parametrize("k", [1, 3, 5])
def test_outputs_are_the_target_sequence_several_tokens_per_step(k) -> None:
    specs = [Spec(rid=i, prompt_len=10 + 7 * i, max_tokens=40, seed=0x1234_5678_9 + i) for i in range(4)]
    h = run(specs, mtp_k=k, accept_p=0.9, max_num_seqs=4, max_tokens=128, page_size=4, num_pages=400,
            max_model_len=256)
    dev = [nq for p in h.plans for nq, d in zip(p["n_query"], p["device"]) if d]
    assert dev and set(dev) == {1 + k}
    n = h.runner.counts
    assert n["drafts"] > 0 and 0 < n["accepted"] < n["drafts"]
    assert h.sched.spec_drafts * k == h.sched.spec_draft_tokens and h.sched.spec_accepted == n["accepted"]
    per_pos = h.sched.spec_accepted_per_pos
    assert (np.diff(per_pos) <= 0).all() and per_pos[0] <= h.sched.spec_drafts


@pytest.mark.parametrize("k", [0, 3])
def test_prompt_tokens_count_prefill_not_drafts(k) -> None:
    """EngineStats.prompt_tokens counts the prefill tokens computed (the plans' explicit rows); a decode row's k drafts
    are verify work, counted in spec_draft_tokens. With no cache hit and no preemption: the prompts' lengths."""
    specs = [Spec(rid=i, prompt_len=10 + 7 * i, max_tokens=12, seed=0x5EED_0000 + i, arrive=2 * i) for i in range(6)]
    h = run(specs, mtp_k=k, accept_p=0.5, max_num_seqs=4, max_tokens=64, page_size=4, num_pages=200,
            prefix_caching=False)
    assert h.sched.num_preemptions == 0 and any(d for p in h.plans for d in p["device"])
    explicit = sum(n for p in h.plans for n, d in zip(p["n_query"], p["device"]) if not d)
    assert explicit == sum(s.prompt_len for s in specs)
    assert h.sched.prompt_tokens == explicit


def test_eos_inside_an_accepted_run_ends_the_row() -> None:
    """Every draft accepted (accept_p 1): the EOS at output 5 is the bonus of its row, and nothing after it is
    committed; the request ends exactly at EOS and its donated state resumes a follow-up turn exactly."""
    a = Spec(rid=0, prompt_len=40, prompt=stem(0, 40), max_tokens=30, seed=77_000_001, eos_after=5)
    out = output_of(a, 6)
    turn = list(prompt_of(a, VOCAB)) + out + [5, 6, 7]
    b = Spec(rid=1, prompt_len=0, prompt=turn, max_tokens=4, seed=77_000_002, arrive=40)
    h = run([a, b], mtp_k=3, accept_p=1.0, page_size=4, num_pages=200, checkpoint_slots=4, grid=16,
            max_model_len=256)
    assert h.results[0].tokens == out and h.results[0].finish == FinishReason.STOP
    assert not h.sched.overrun.any()
    # The second turn resumes from the donation: the KV through EOS and the state folded there.
    assert h.results[1].cached >= len(prompt_of(a, VOCAB)) + 5
    assert any(p["fold"] for p in h.plans)


def test_max_tokens_inside_an_accepted_run() -> None:
    specs = [Spec(rid=i, prompt_len=12, max_tokens=m, seed=88_000_000 + i) for i, m in enumerate((1, 2, 5, 9))]
    h = run(specs, mtp_k=3, accept_p=1.0, max_num_seqs=4, page_size=4, num_pages=200)
    for i, m in enumerate((1, 2, 5, 9)):
        assert len(h.results[i].tokens) == m and h.results[i].finish == FinishReason.LENGTH


@pytest.mark.parametrize(("accept_p", "max_tokens"), [(1.0, 29), (1.0, 31), (0.5, 40), (0.8, 55)])
def test_a_length_finish_wastes_no_decode_row(accept_p, max_tokens) -> None:
    """One request at a time: a decode row is scheduled only when the in-flight ones cannot already reach
    max_tokens (each may deliver 1 + k), so every row's result delivers tokens; a wasted row at the end would delay
    the next request's prefill at low concurrency."""
    spec = Spec(rid=0, prompt_len=20, max_tokens=max_tokens, seed=99_000_000 + max_tokens)
    h = run([spec], mtp_k=3, accept_p=accept_p, max_num_seqs=4, page_size=4, num_pages=200)
    assert len(h.results[0].tokens) == max_tokens and h.results[0].finish == FinishReason.LENGTH
    rows = sum(d for p in h.plans for d in p["device"])
    assert rows == h.sched.spec_drafts  # every decode row delivered (a dropped one is not counted as a draft)
    if accept_p == 1.0:
        assert rows == -(-(max_tokens - 1) // 4)


def test_stop_token_beyond_the_runner_limits_cuts_the_row() -> None:
    """A request with more stop tokens than the runner knows (STOP_W): the device may accept past the unknown one,
    the scheduler drops the tokens after it and caches the KV only (no state donation past the finish)."""
    a = Spec(rid=0, prompt_len=20, max_tokens=40, seed=99_000_123, ignore_eos=True)
    out = output_of(a, 40)
    target = out[10]  # mid-row: outputs 9-12 are one row (1 + 3)
    stops = [*[t for t in range(1, VOCAB - 1) if t not in out][:STOP_W], target]
    a.stop_ids = stops
    h = run([a], mtp_k=3, accept_p=1.0, page_size=4, num_pages=200, checkpoint_slots=2, grid=16)
    first = out.index(target)
    assert h.results[0].tokens == out[:first + 1] and h.results[0].finish == FinishReason.STOP
    assert h.sched.num_overruns == 1


def test_preemption_with_a_decode_step_in_flight_keeps_its_tokens() -> None:
    """Pages for about one long request: the second one's growth preempts the first while its decode row is in
    flight; that row's tokens (up to 1 + k) reach the waiting request, and its readmission resumes from the donated
    state, folded to the in-flight row's first token (FOLD 1)."""
    specs = [Spec(rid=i, prompt_len=24, max_tokens=60, seed=(i + 3) * 0x9E3779B9) for i in range(3)]
    h = run(specs, mtp_k=3, accept_p=0.8, page_size=4, num_pages=1 + 36, max_num_seqs=3, max_tokens=64,
            max_model_len=128, checkpoint_slots=4, grid=16)
    assert h.sched.num_preemptions > 0
    assert any(n == 1 for p in h.plans for _, n in p["fold"])


def test_decode_checkpoints_land_where_the_commit_ends() -> None:
    """Decode checkpoints under MTP (interval 16): requested when a row's lowest possible commit reaches the next
    multiple of the interval, attached where the step's commit ended, at most 2k past it. The request keeps its latest
    (in [48, 54] for 60 positions) and a request repeating 55 of its positions resumes from it."""
    a = Spec(rid=0, prompt_len=10, max_tokens=50, seed=4_000_001, ignore_eos=True)
    out = output_of(a, 50)
    b = Spec(rid=1, prompt_len=0, prompt=list(prompt_of(a, VOCAB)) + out[:45] + [3], max_tokens=3,
             seed=4_000_002, arrive=60)
    h = run([a, b], mtp_k=3, accept_p=0.7, page_size=4, num_pages=200, checkpoint_slots=8, grid=16,
            decode_checkpoint_interval=16)
    assert h.sched.ckpts_taken["decode"] == 3  # past 16, 32 and 48
    assert 48 <= h.results[1].cached <= 54


@pytest.mark.parametrize("k", [1, 3])
def test_long_outputs_through_preemption(k) -> None:
    """Outputs past the scheduler's 256-token output ring, several tokens per step, with preemptions that rebuild a
    request's tokens from it (readmission) and cache keys from it (donations, decode checkpoints)."""
    specs = [Spec(rid=i, prompt_len=30, max_tokens=700, seed=(i + 7) * 0x9E3779B97, ignore_eos=True) for i in range(3)]
    h = run(specs, mtp_k=k, accept_p=0.8, page_size=16, num_pages=1 + 2 * 50, max_num_seqs=3, max_tokens=256,
            max_model_len=1024, checkpoint_slots=4, grid=64, decode_checkpoint_interval=128)
    assert h.sched.num_preemptions > 0 and all(len(h.results[i].tokens) == 700 for i in range(3))


@pytest.mark.parametrize("page_size", [4, 16])
def test_hit_on_a_page_boundary_copies_that_page(page_size) -> None:
    """A hit at L on a page boundary: the drafter's KV at L − 1 depends on the token at L, which the new request may
    not share, so the full page holding L − 1 is copied (KV_COPY) before the first chunk rewrites it."""
    base = stem(3, 64)
    a = Spec(rid=0, prompt_len=0, prompt=[*base, 9, 9], max_tokens=3, seed=5_000_001)
    b = Spec(rid=1, prompt_len=0, prompt=[*base, 8, 8, 8], max_tokens=3, seed=5_000_002, arrive=20)
    h = run([a, b], mtp_k=3, page_size=page_size, num_pages=200, checkpoint_slots=4, grid=16)
    hit = h.results[1].cached
    assert hit and hit % page_size == 0
    admit_b = next(p for p in h.plans if p["step"] > 1 and p["admit"])
    assert admit_b["kv_copy"] == 1


@pytest.mark.parametrize("k, draft_kv, pages", [(3, True, 17), (2, True, 17), (1, True, 16), (3, False, 16)])
def test_a_final_chunk_holds_its_drafter_chains_pages(k, draft_kv, pages) -> None:
    """A final chunk's row samples, and a drafter with KV of its own (Qwen's MTP layer) writes its chain's k − 1
    positions past the row and reads them back in the same step: a prompt of 64 (pages of 4) ending on a page boundary
    takes a 17th page with its chunk under k ≥ 2. Its entry would be 0 otherwise, the null page, which a CUDA graph's
    padding rows write in the same kernels (the strict fake raises). k = 1 has no chain, and a drafter without KV
    (Gemma 4's assistant) writes none: 16."""
    a = Spec(rid=0, prompt_len=64, max_tokens=6, seed=5_000_003)
    h = Harness(specs=[a], vocab=VOCAB, mtp_k=k, draft_kv=draft_kv, page_size=4, num_pages=200, max_tokens=128)
    first: list[int] = []
    launch = h.runner.launch_forward

    def record(plan):
        if not first:
            first.append(len(plan.page_id))
        return launch(plan)

    h.runner.launch_forward = record
    h.run(check_every_turn=True)
    check_plans(h)
    h.check()
    assert h.plans[0]["n_query"] == [64] and h.plans[0]["sample"] == [True] and first == [pages]


def test_abort_mid_decode_folds_and_donates() -> None:
    """An abort with a decode row in flight donates the live state folded one record into that row (its pending token);
    the FOLD waits for the next plan (here: when b arrives), which is the first that can read the checkpoint."""
    a = Spec(rid=0, prompt_len=30, max_tokens=60, seed=6_000_001, abort_at=6, ignore_eos=True)
    b = Spec(rid=1, prompt_len=30, max_tokens=4, seed=6_000_002, arrive=12)
    b.prompt = list(prompt_of(a, VOCAB))
    h = run([a, b], mtp_k=3, accept_p=0.6, page_size=4, num_pages=200, checkpoint_slots=2, grid=16)
    assert h.sched.ckpts_taken["finish"] == 2  # a's abort and b's finish
    folds = [p for p in h.plans if p["fold"]]
    assert folds and folds[0]["fold"] == [(1, 1)] and 1 in folds[0]["free"]


# -- the targeted fuzz suite ------------------------------------------------------------------------------------------


@campaign_settings(300)
@given(mtp_workloads())
def test_random_mtp_workloads(w) -> None:
    with saved_on_failure("mtp", w):
        run_mtp_workload(w)


@campaign_settings(300)
@given(mtp_count_workloads())
def test_random_mtp_count_workloads(w) -> None:
    """The MTP workloads with a draft count per step: the same outputs and invariants whatever the counts."""
    with saved_on_failure("mtp_counts", w):
        run_mtp_workload(w)


