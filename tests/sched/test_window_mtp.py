"""MTP in window mode (Gemma 4 with its assistant) through the scheduler and the validating fake runner.

A decode row verifies 1 + k tokens and commits 1 + a of them, so the host knows its next first query only as a lower
bound (`mirror + dev_inflight`) and the window pages below it are released from that bound; the verify window's KV
lands in both kinds of page, its rejected positions holding tokens of no sequence until the next window rewrites them
(rollback is a truncation); decode windows are made where a step's commit ends, finish windows at the lowest committed
position. The fake checks every row's window against the device's true positions and every cached position's
contents (a window or a hit that reads a rejected draft's KV, or a page released while a query still sees it, raises
PlanError), and predicts every output. Gemma 4's assistant writes no KV (`draft_kv=False`); a drafter that does
(Qwen's MTP layer, `draft_kv=True`) is drawn too, for the scheduler's generic paths.
"""

from __future__ import annotations

import pytest
from hypothesis import given

from ..support.fuzzing import campaign_settings, saved_on_failure
from ..support.harness import Harness, Spec
from ..support.sched_cases import check_plans, output_of, shared, stem
from ..support.window_mtp import (
    arena_mtp_count_workloads,
    arena_mtp_workloads,
    gemma_images_mtp_workloads,
    harness,
    run_window_mtp_workload,
    window_mtp_workloads,
)


@pytest.mark.parametrize("k", [3, 5, 7])
@pytest.mark.parametrize("accept_p", [0.0, 0.6, 1.0])
def test_mtp_rows_slide_their_windows(accept_p, k) -> None:
    """A 200-token prompt and 60 decode tokens with a window of 16 (pages of 4), k = 3, 5 (the assistant's default)
    and 7: whatever is accepted, every verify row finds its window from its true first query on (the fake checks the
    device's positions), the request holds about a window's pages, and its outputs are the predicted ones."""
    spec = Spec(rid=0, prompt_len=200, max_tokens=60, seed=5, ignore_eos=True)
    h = harness([spec], k=k, accept_p=accept_p, prefix_caching=False)
    h.run(check_every_turn=True)
    h.check()
    if accept_p == 0.0:
        assert h.runner.counts["accepted"] == 0
    if accept_p == 1.0:
        assert h.runner.counts["accepted"] >= 40  # most steps took 1 + k tokens


def test_a_hit_on_a_finish_window_reads_no_rejected_draft() -> None:
    """Request a decodes with half its drafts rejected, so its last pages hold rejected drafts' KV past its committed
    end; it finishes with a window there. b continues a's whole conversation: it resumes at a's committed end from
    that window, copying the partial page of each kind (rejected KV after the hit, which b's chunk rewrites)."""
    a = Spec(rid=0, prompt_len=0, prompt=stem(4, 41), max_tokens=19, seed=61, ignore_eos=True)
    out = output_of(a)
    b = Spec(rid=1, prompt_len=0, prompt=a.prompt + out + [3, 4, 5], max_tokens=6, seed=62, arrive=60)
    h = harness([a, b], window=8, accept_p=0.5)
    h.run(check_every_turn=True)
    h.check()
    assert h.results[1].cached == 41 + 19 - 1 and h.sched.ckpts_taken["finish"] == 2


def test_mtp_decode_windows_serve_a_continuation() -> None:
    """Decode windows every 16 positions land where each step's commit ends, past the multiple (one at a time: a
    running request keeps its latest); a continuation sharing the first 60 generated tokens, arriving while the first
    request decodes, resumes from its latest decode window, inside the generated tokens."""
    t1 = Spec(rid=0, prompt_len=0, prompt=stem(3, 20), max_tokens=150, seed=21, ignore_eos=True)
    out = output_of(t1)
    t2 = Spec(rid=1, prompt_len=0, prompt=t1.prompt + out[:60] + [9, 9], max_tokens=3, seed=22, arrive=20)
    h = harness([t1, t2], window=8, decode_checkpoint_interval=16, accept_p=0.7, max_model_len=256)
    h.run(check_every_turn=True)
    h.check()
    assert h.sched.ckpts_taken["decode"] >= 3
    assert 20 + 16 <= h.results[1].cached <= 80


def test_preemption_with_verify_rows_in_flight() -> None:
    """A request decoding with MTP is preempted while its verify rows are in flight, its slot taken by a
    higher-priority one: it leaves a window at its lowest committed position and resumes there, its in-flight tokens
    delivered to it (no FOLD in window mode)."""
    low = Spec(rid=0, prompt_len=60, max_tokens=40, seed=31, priority=1, ignore_eos=True, preempt_at=6)
    high = Spec(rid=1, prompt_len=50, max_tokens=8, seed=32, priority=0, arrive=6)
    h = harness([low, high], window=8, max_tokens=32, max_num_seqs=1, policy="priority", max_model_len=200,
                accept_p=0.8)
    h.run(check_every_turn=True)
    h.check()
    assert h.sched.num_preemptions >= 1 and not any(p["fold"] for p in h.plans)
    assert h.sched.readmit_hits > 60  # the readmission resumed inside its generated tokens


def test_siblings_hit_a_branch_window_with_mtp() -> None:
    specs = [Spec(rid=i, prompt_len=0, prompt=shared(100, 10, i), max_tokens=9, seed=1000 + i) for i in range(4)]
    h = harness(specs, max_num_seqs=4, accept_p=0.8)
    h.run(check_every_turn=True)
    h.check()
    assert [h.results[i].cached for i in range(4)] == [0, 100, 100, 100]


def test_no_drafter_kv_no_copy_at_a_page_aligned_hit() -> None:
    """A drafter without KV (Gemma 4's assistant) needs no private copy of the page before a page-aligned hit (Qwen's
    MTP layer rewrites its KV at hit − 1, so its scheduler copies that page); with KV, the copy is made."""
    for draft_kv, copies in ((False, 0), (True, 1)):
        a = Spec(rid=0, prompt_len=0, prompt=stem(5, 64), max_tokens=2, seed=71, hints=[40])
        b = Spec(rid=1, prompt_len=0, prompt=[*stem(5, 64)[:40], 1, 2, 3], max_tokens=2, seed=72, arrive=20)
        h = harness([a, b], window=8, page_size=4, full=20, draft_kv=draft_kv)
        h.run(check_every_turn=True)
        h.check()
        assert h.results[1].cached == 40
        admit = next(p for p in h.plans if 40 in p["seq_len"])
        assert admit["kv_copy"] == copies, draft_kv


def test_mtp_with_bidirectional_image_spans() -> None:
    """Gemma 4's images with MTP: spans whole in their rows, no image look-ahead for a drafter without KV, outputs
    predicted."""
    prompt = list(range(10, 30)) + [0] * 16 + [7, 8, 9] + [0] * 9 + list(range(40, 52))
    specs = [Spec(rid=i, prompt_len=len(prompt), prompt=prompt, max_tokens=12, seed=800 + i, arrive=4 * i,
                  images=[(20, 8, 8, 1 + i % 2), (39, 6, 6, 3)]) for i in range(3)]
    h = Harness(specs=specs, vocab=1000, window=8, window_page_size=4, page_size=8, num_pages=400, grid=4,
                max_tokens=48, max_model_len=192, mtp_k=3, accept_p=0.7, encoder_rows=128, encoder_budget=64,
                bidirectional=True, draft_kv=False)
    h.run(check_every_turn=True)
    check_plans(h)
    h.check()
    assert all(p["next_emb"] == [] or set(p["next_emb"]) <= {-1} for p in h.plans)
    assert h.sched.prefix_hits > 0


# -- fuzzing --------------------------------------------------------------------------------------------------------


@campaign_settings(300)
@given(window_mtp_workloads())
def test_random_window_mtp_workloads(w) -> None:
    with saved_on_failure("window_mtp", w):
        run_window_mtp_workload(w)


@campaign_settings(300)
@given(arena_mtp_workloads())
def test_random_arena_mtp_workloads(w) -> None:
    with saved_on_failure("arena_mtp", w):
        run_window_mtp_workload(w)


@campaign_settings(300)
@given(arena_mtp_count_workloads())
def test_random_arena_mtp_count_workloads(w) -> None:
    """Gemma 4's arena workloads with a draft count per step."""
    with saved_on_failure("arena_mtp_counts", w):
        run_window_mtp_workload(w)


@campaign_settings(300)
@given(gemma_images_mtp_workloads())
def test_random_gemma_images_mtp_workloads(w) -> None:
    with saved_on_failure("gemma_images_mtp", w):
        run_window_mtp_workload(w, bidirectional=True)

