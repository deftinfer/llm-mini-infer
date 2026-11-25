"""The prefix cache end to end through the scheduler and the validating fake runner.

The fake samples each token from (seed, position, fingerprint of the context), and checks every resumed row's state
fingerprint against its own KV (strict mode). Any wrong reuse (a checkpoint of another prefix, a stale or missing
page, a copy at the wrong offset) either raises PlanError or changes the tokens, which are predicted from the true
context. Each run also checks page refcounts and state-slot ownership, and that everything is freed at the end.
"""

from __future__ import annotations

import inspect
from collections import Counter

import numpy as np
import pytest
from hypothesis import given

from llm_mini_infer.cache.pools import pages_for_one_request, window_pages_for_one_request
from llm_mini_infer.engine.messages import CHECKPOINT_SOURCES, FinishReason
from llm_mini_infer.sched.plan import reserve_positions

from ..support.fuzzing import campaign_settings, saved_on_failure
from ..support.harness import Harness, Spec
from ..support.sched_cases import VOCAB, check_plans, output_of, shared, stem
from ..support.workloads import cache_workloads, pressure_pairs, run_cache_workload


@campaign_settings(300)
@given(cache_workloads())
def test_random_cache_workloads(w) -> None:
    with saved_on_failure("cache_sched", w):
        run_cache_workload(w)


@campaign_settings(300)
@given(pressure_pairs())
def test_random_pressure_pairs(w) -> None:
    with saved_on_failure("cache_pressure", w):
        run_cache_workload(w)


def test_siblings_wait_one_step_then_hit_the_branch_point() -> None:
    """Four requests share a 100-token prefix and arrive together. The first computes it; the others wait one step
    (the first one's row takes a checkpoint at the branch point, 96) and then hit 96 tokens."""
    specs = [Spec(rid=i, prompt_len=0, prompt=shared(100, 10, i), max_tokens=3, seed=1000 + i) for i in range(4)]
    h = Harness(specs=specs, vocab=VOCAB, max_num_seqs=4, max_tokens=512, page_size=4, num_pages=200,
                checkpoint_slots=6, grid=16, max_model_len=256)
    h.run(check_every_turn=True)
    h.check()
    assert [h.results[i].cached for i in range(4)] == [0, 96, 96, 96]
    first = h.plans[0]
    assert first["admit"] == [1] and (0, 96) in first["ckpt"]  # branch point on request 0's row
    second = h.plans[1]
    assert sorted(second["admit"]) == [2, 3, 4] and second["seq_len"] == [96, 96, 96]
    assert second["state_copy"] == 3 and second["kv_copy"] == 0  # 96 is page-aligned
    assert h.sched.cache_waits == 3
    # each sibling recomputes the prefix's last 4 tokens past the branch point's checkpoint (its waits count nothing)
    assert h.sched.lost_tokens == {"writing": 0, "checkpoint": 3 * 4, "page": 0, "precise": 0}


def test_classify_shape_reuses_over_90_percent() -> None:
    """A classification-shaped workload, text only, scaled down: a 710-token shared instruction prefix and short
    unique items (730 tokens), one request at a time. Only the first request computes the prefix: its end-of-prefill
    checkpoint (704) lies on the shared part."""
    specs = [Spec(rid=i, prompt_len=0, prompt=shared(710, 20, i), max_tokens=4, seed=7000 + i, arrive=10 * i)
             for i in range(20)]
    h = Harness(specs=specs, vocab=VOCAB, max_num_seqs=2, max_tokens=2048, page_size=16, num_pages=400,
                checkpoint_slots=4, grid=64, max_model_len=1024)
    h.run()
    h.check()
    s = h.sched
    assert s.prefix_hits / s.prefix_queries > 0.9, s.prefix_hits / s.prefix_queries
    assert [h.results[i].cached for i in range(3)] == [0, 704, 704]
    assert s.lost_tokens == {"writing": 0, "checkpoint": 19 * (710 - 704), "page": 0, "precise": 0}


def test_long_items_cost_a_second_miss() -> None:
    """When the end-of-prefill checkpoint lies past the shared part (700 shared, 720 total: 704), the second request
    finds KV but no state on the shared part: it recomputes and checkpoints the branch point (640) for the rest."""
    specs = [Spec(rid=i, prompt_len=0, prompt=shared(700, 20, i), max_tokens=4, seed=7100 + i, arrive=10 * i)
             for i in range(5)]
    h = Harness(specs=specs, vocab=VOCAB, max_num_seqs=2, max_tokens=2048, page_size=16, num_pages=400,
                checkpoint_slots=4, grid=64, max_model_len=1024)
    h.run()
    h.check()
    assert [h.results[i].cached for i in range(5)] == [0, 0, 640, 640, 640]


def test_hint_gives_an_exact_checkpoint_with_copy_on_write() -> None:
    """A hint at 50 (off the grid and off the page boundary) ends a chunk there; the next request sharing 55 tokens
    hits exactly 50 and copies the partial page."""
    a = stem(1, 80)
    b = [*a[:55], 7, 8, 9, 10]
    specs = [Spec(rid=0, prompt_len=0, prompt=a, hints=[50], max_tokens=2, seed=1),
             Spec(rid=1, prompt_len=0, prompt=b, max_tokens=2, seed=2, arrive=20)]
    h = Harness(specs=specs, vocab=VOCAB, max_tokens=256, page_size=8, num_pages=100, checkpoint_slots=4, grid=16)
    h.run(check_every_turn=True)
    h.check()
    assert h.plans[0]["n_query"] == [50] and (0, 50) in h.plans[0]["ckpt"]
    assert h.results[1].cached == 50 and h.runner.counts["kv_copy"] == 1


def test_multi_turn_hits_the_finish_checkpoint() -> None:
    """Turn 2 is turn 1's prompt and output plus a new message: it resumes where turn 1 stopped computing."""
    t1 = Spec(rid=0, prompt_len=0, prompt=stem(2, 37), max_tokens=12, seed=11)
    out = output_of(t1)
    t2 = Spec(rid=1, prompt_len=0, prompt=t1.prompt + out + [5, 6, 7], max_tokens=4, seed=12, arrive=40)
    h = Harness(specs=[t1, t2], vocab=VOCAB, max_tokens=256, page_size=4, num_pages=100, checkpoint_slots=4,
                grid=16)
    h.run(check_every_turn=True)
    h.check()
    # a length finish: turn 1 computed its prompt and all but its last token
    assert h.results[1].cached == 37 + 12 - 1 and h.sched.ckpts_taken["finish"] == 2  # one per finish


def test_resume_from_an_off_grid_hit_runs_in_one_step() -> None:
    """Turn 2 resumes from turn 1's finish checkpoint at 111 (off the 16-grid); its remaining 36 tokens fit the
    budget and run in one chunk. Its end-of-prefill target (desired at 146) goes on the chunk's own grid, at 143 (a
    target that ended the chunk, at 144, would leave the rest a second step). Two siblings arriving with turn 2 still
    wait one step and then hit: one at turn 2's checkpoint (143), one at a branch point added to turn 2's row (127,
    desired at 130). Pages hold one token so that 111 is not mid-page: a hit inside a page makes the hit node pending
    for the plan that copies that page, and siblings would wait on that instead."""
    t1 = Spec(rid=0, prompt_len=0, prompt=stem(15, 100), max_tokens=12, seed=61, ignore_eos=True)
    p2 = t1.prompt + output_of(t1) + [(j * 7 + 3) % (VOCAB - 1) for j in range(35)]  # 147 tokens
    t2 = Spec(rid=1, prompt_len=0, prompt=p2, max_tokens=2, seed=62, arrive=20)
    s3 = Spec(rid=2, prompt_len=0, prompt=[*p2[:130], 1, 2, 3], max_tokens=2, seed=63, arrive=20)
    s4 = Spec(rid=3, prompt_len=0, prompt=[*p2[:146], 4], max_tokens=2, seed=64, arrive=20)
    h = Harness(specs=[t1, t2, s3, s4], vocab=VOCAB, max_num_seqs=4, max_tokens=256, page_size=1, num_pages=1000,
                checkpoint_slots=8, grid=16, max_model_len=256)
    h.run(check_every_turn=True)
    h.check()
    first = next(p for p in h.plans if p["admit"] and p["seq_len"] == [111])
    assert first["n_query"] == [36] and first["sample"] == [True]  # the whole remainder in one step
    assert (0, 32) in first["ckpt"] and (0, 16) in first["ckpt"]  # 143 (≤ 146) and the sibling's 127 (≤ 130)
    assert 146 - 143 <= 16 - 1 and 130 - 127 <= 16 - 1
    assert [h.results[i].cached for i in range(4)] == [0, 111, 127, 143]
    assert h.sched.cache_waits == 2


def test_off_grid_resume_checkpoints_within_one_grid_step() -> None:
    """Turn 2 resumes at turn 1's finish checkpoint, 97 (one past the 16-grid), and prefills to 144. Its
    end-of-prefill checkpoint goes on the chunk's grid within G − 1 of the last prompt token (143): at 129. Desired at
    ⌊143/16⌋·16 = 128 instead, the chunk's grid would put it at 113, 2(G − 1) below. An identical prompt then resumes
    at 129."""
    t1 = Spec(rid=0, prompt_len=0, prompt=stem(30, 86), max_tokens=12, seed=121, ignore_eos=True)
    p2 = t1.prompt + output_of(t1) + [(j * 11 + 5) % (VOCAB - 1) for j in range(46)]  # 144 tokens
    t2 = Spec(rid=1, prompt_len=0, prompt=p2, max_tokens=1, seed=122, arrive=30)
    t3 = Spec(rid=2, prompt_len=0, prompt=p2, max_tokens=1, seed=123, arrive=60)
    h = Harness(specs=[t1, t2, t3], vocab=VOCAB, max_num_seqs=2, max_tokens=256, page_size=1, num_pages=1000,
                checkpoint_slots=8, grid=16, max_model_len=256)
    h.run(check_every_turn=True)
    h.check()
    resume = next(p for p in h.plans if p["seq_len"] == [97])
    assert resume["n_query"] == [47] and (0, 129 - 97) in resume["ckpt"]
    assert 143 - 129 <= 16 - 1
    assert [h.results[i].cached for i in range(3)] == [0, 97, 129]


def test_budget_cut_in_the_last_grid_step_keeps_the_end_of_prefill_checkpoint() -> None:
    """A 100-token prompt under a 97-token budget: the cut is rounded down to the grid (96), and the end-of-prefill
    target (99) lies less than G past it, where the next chunk [96, 100) could place it only at its start. The cut
    takes it as an end-of-chunk checkpoint, so a repeat hits 96 (without that, no checkpoint and a full miss)."""
    p = stem(31, 100)
    specs = [Spec(rid=0, prompt_len=0, prompt=p, max_tokens=2, seed=131),
             Spec(rid=1, prompt_len=0, prompt=p, max_tokens=2, seed=132, arrive=20)]
    h = Harness(specs=specs, vocab=VOCAB, max_num_seqs=1, max_tokens=97, page_size=4, num_pages=200,
                checkpoint_slots=4, grid=16, max_model_len=256)
    h.run(check_every_turn=True)
    h.check()
    assert h.plans[0]["n_query"] == [96] and (0, 96) in h.plans[0]["ckpt"]
    assert h.plans[1]["n_query"] == [4] and not h.plans[1]["ckpt"]
    assert h.results[1].cached == 96


@pytest.mark.parametrize("n, grid", [(64, 16), (80, 16), (96, 16), (100, 16), (1280, 64), (1300, 64)])
def test_identical_prompts_of_every_length_hit(n, grid) -> None:
    """A repeat of a prompt hits its end-of-prefill checkpoint, at ⌊(n − 1)/G⌋·G: for a prompt whose length is a
    multiple of G, a checkpoint at n would lie past the repeat's hit limit of n − 1, and the repeat would miss
    entirely."""
    p = stem(16, n)
    specs = [Spec(rid=0, prompt_len=0, prompt=p, max_tokens=3, seed=71),
             Spec(rid=1, prompt_len=0, prompt=p, max_tokens=3, seed=72, arrive=10)]
    h = Harness(specs=specs, vocab=VOCAB, max_num_seqs=1, max_tokens=2048, page_size=16, num_pages=200,
                checkpoint_slots=4, grid=grid, max_model_len=2048)
    h.run(check_every_turn=True)
    h.check()
    assert h.results[1].cached == (n - 1) // grid * grid


def test_decode_checkpoints_serve_a_continuation() -> None:
    """Turn 2 branches off turn 1's output after position 64. Turn 1 checkpointed every 16 positions while decoding
    but kept only its latest decode checkpoint (64): the earlier ones were released as it went, so a branch before 64
    would find none."""
    t1 = Spec(rid=0, prompt_len=0, prompt=stem(3, 10), max_tokens=60, seed=21, ignore_eos=True)
    out = output_of(t1)
    t2 = Spec(rid=1, prompt_len=0, prompt=t1.prompt + out[:58] + [1, 2], max_tokens=2, seed=22)
    assert out[58] != 1
    t2.arrive = 80  # after turn 1 finished
    h = Harness(specs=[t1, t2], vocab=VOCAB, max_tokens=256, page_size=4, num_pages=100, checkpoint_slots=8,
                grid=16, decode_checkpoint_interval=16)
    h.run(check_every_turn=True)
    h.check()
    assert h.sched.ckpts_taken["decode"] == 4  # positions 16, 32, 48, 64
    assert h.results[1].cached == 64  # the latest decode checkpoint, on turn 2's path


def test_a_long_generation_keeps_one_decode_checkpoint() -> None:
    """A long generation checkpoints every 8 positions but holds at most its latest decode checkpoint, and locks do
    not pin checkpoints: with 3 checkpoint slots, a new prefix arriving meanwhile still gets its checkpoint, and a
    repeat of it hits (locked decode checkpoints kept would fill the slots, and the new prefix's checkpoint would be
    skipped)."""
    a = Spec(rid=0, prompt_len=0, prompt=stem(9, 10), max_tokens=120, seed=91, ignore_eos=True)
    b = Spec(rid=1, prompt_len=0, prompt=stem(10, 41), max_tokens=3, seed=92, arrive=40)
    c = Spec(rid=2, prompt_len=0, prompt=stem(10, 41), max_tokens=3, seed=93, arrive=60)
    h = Harness(specs=[a, b, c], vocab=VOCAB, max_num_seqs=2, max_tokens=256, page_size=4, num_pages=200,
                checkpoint_slots=3, grid=8, max_model_len=256, decode_checkpoint_interval=8)
    held = []
    orig = h.sched._decode_ckpts_done

    def done(inf):
        orig(inf)
        req = h.sched.reqs.get((0, 0))
        if req is not None and req.lock is not None:  # decode checkpoints on request 0's path
            node, n = req.lock, 0
            while node.parent is not None:
                n += bool(node.ckpt) and node.end > 10
                node = node.parent
            held.append(n)

    h.sched._decode_ckpts_done = done
    h.run(check_every_turn=True)
    h.check()
    s = h.sched
    assert s.ckpts_taken["decode"] >= 14 and max(held) == 1, (s.ckpts_taken, max(held))
    assert s.ckpts_skipped == 0 and h.results[2].cached == 40


def test_a_waiting_request_lets_its_match_go_for_the_branch_checkpoint() -> None:
    """Request 2's branch point (66, placed at 64) lies in KV that request 1's running chunk [56, 72) writes in the
    same plan, so _branch_wait wants a checkpoint slot there. Both slots are taken: one by request 1's end-of-prefill
    checkpoint (72, written by this plan) and one by the checkpoint request 2 just matched (40). A checkpoint is pinned
    by the STATE_COPY that reads it, not by the match: request 2 waits and reads nothing, so its match makes room, and
    it resumes at 64 one step later (pinned at the match, it would resume at 40). Every plan pins exactly its
    STATE_COPY sources."""
    x = stem(0, 45)
    tail = [(j * 13 + 500) % (VOCAB - 1) for j in range(32)]
    w = x[:44] + tail  # 76 tokens: chunks [40, 56) and [56, 72) under chunk_cap 16; end-of-prefill checkpoint at 72
    specs = [Spec(rid=0, prompt_len=0, prompt=x, max_tokens=1, seed=101),
             Spec(rid=1, prompt_len=0, prompt=w, max_tokens=2, seed=102, arrive=5),
             Spec(rid=2, prompt_len=0, prompt=[*w[:66], 7, 8, 9, 10, 11, 12, 13, 14], max_tokens=2, seed=103,
                  arrive=6)]
    h = Harness(specs=specs, vocab=VOCAB, max_num_seqs=3, max_tokens=256, chunk_cap=16, page_size=4, num_pages=200,
                checkpoint_slots=2, grid=8, max_model_len=256)
    cache, pins, sources = h.sched.cache, [], []
    unpin, launch = cache.unpin_checkpoints, h.runner.launch_forward

    def record_pins():
        pins.append((h.turn, set(cache._pinned)))
        unpin()

    def record_sources(plan):
        sources.append((h.turn, set(plan.state_copy_src.tolist())))
        return launch(plan)

    cache.unpin_checkpoints, h.runner.launch_forward = record_pins, record_sources
    h.run(check_every_turn=True)
    h.check()
    assert [h.results[i].cached for i in range(3)] == [0, 40, 64]
    assert h.sched.cache_waits == 1
    launched = dict(sources)
    assert all(p == launched.get(turn, set()) for turn, p in pins), (pins, sources)
    assert any(s for _, s in sources)


def test_preempted_request_readmission_hits_its_own_prefix() -> None:
    """A request preempted while decoding gives the only slot to a higher-priority one. The preempted one leaves its
    KV and live state in the cache and keeps the sample that was in flight, so its
    readmission resumes exactly where it stopped computing: position 48 (43 prompt tokens and 5 outputs; the 6th
    output, sampled by the step in flight, came back while it waited), computing one token."""
    specs = [Spec(rid=0, prompt_len=0, prompt=stem(7, 43), max_tokens=30, seed=31, priority=5, preempt_at=6),
             Spec(rid=1, prompt_len=0, prompt=stem(8, 20), max_tokens=3, seed=32, priority=0, arrive=6)]
    h = Harness(specs=specs, vocab=VOCAB, max_num_seqs=1, max_tokens=64, page_size=4, num_pages=1 + 40,
                checkpoint_slots=4, grid=8, max_model_len=128, policy="priority")
    h.run(check_every_turn=True)
    h.check()
    assert h.sched.num_preemptions == 1 and h.sched.readmit_hits == 48
    readmit = [p for p in h.plans if p["seq_len"] == [48]]
    assert len(readmit) == 1 and readmit[0]["n_query"] == [1] and readmit[0]["state_copy"] == 1


def test_a_penalised_readmission_resumes_within_its_prompt() -> None:
    """The same preemption with penalties: the readmission resumes at its prompt's end at most (43; its outputs are
    recomputed, and the runner counts them as it goes: `sched.plan`, ADMIT), its ADMIT carrying the bits of the
    tokens it resumes past and its prompt length. The fake's checks refuse a penalised ADMIT past its prompt."""
    from llm_mini_infer.sched.plan import ADMIT_PENALTIES

    pen = (1.3, 0.5, 0.0)
    specs = [Spec(rid=0, prompt_len=0, prompt=stem(7, 43), max_tokens=30, seed=31, priority=5, preempt_at=6,
                  penalties=pen, bias={3: 2.0}),
             Spec(rid=1, prompt_len=0, prompt=stem(8, 20), max_tokens=3, seed=32, priority=0, arrive=6)]
    h = Harness(specs=specs, vocab=VOCAB, max_num_seqs=1, max_tokens=64, page_size=4, num_pages=1 + 40,
                checkpoint_slots=4, grid=8, max_model_len=128, policy="priority")
    admits: list[tuple] = []
    launch = h.runner.launch_forward

    def record(plan):
        for i in range(len(plan.admit_slot)):
            admits.append((int(plan.admit_seq_len[i]), int(plan.admit_flags[i]) & ADMIT_PENALTIES,
                           int(plan.admit_prompt_len[i]), plan.admit_prompt_bits[i].copy(), int(plan.admit_bias_n[i])))
        return launch(plan)

    h.runner.launch_forward = record
    h.run(check_every_turn=True)
    h.check()
    assert h.sched.num_preemptions == 1
    first, other, again = admits
    assert first[:3] == (0, ADMIT_PENALTIES, 43) and not first[3].any() and first[4] == 1
    assert other[1] == 0 and other[4] == 0
    seq, flag, plen, bits, nb = again
    assert flag and plen == 43 and nb == 1 and 0 < seq <= 43
    want = np.zeros(bits.size * 32, bool)
    want[stem(7, 43)[:seq]] = True
    assert np.array_equal(bits, np.packbits(want, bitorder="little").view("<u4"))


def test_preemption_mid_prefill_resumes_exactly() -> None:
    """A long prompt prefilled 16 tokens per step is preempted after 48, and a higher-priority request takes its slot.
    Its readmission resumes at 48, from the live state it left as a checkpoint (without it, it would find no
    checkpoint below its end-of-prefill target, 96, and recompute from 0)."""
    specs = [Spec(rid=0, prompt_len=0, prompt=stem(11, 100), max_tokens=3, seed=41, priority=5, preempt_at=3),
             Spec(rid=1, prompt_len=0, prompt=stem(12, 10), max_tokens=2, seed=42, priority=0, arrive=3)]
    h = Harness(specs=specs, vocab=VOCAB, max_num_seqs=1, max_tokens=64, chunk_cap=16, page_size=4, num_pages=100,
                checkpoint_slots=4, grid=8, max_model_len=128, policy="priority")
    h.run(check_every_turn=True)
    h.check()
    assert h.sched.num_preemptions == 1 and h.sched.readmit_hits == 48
    assert [p["seq_len"] for p in h.plans if p["admit"]] == [[0], [0], [48]]
    s = h.sched
    assert (s.ckpts_taken["preempt"], s.ckpts_taken["finish"]) == (1, 2)  # the preemption's donation, two finishes
    assert s.ckpt_resumes[CHECKPOINT_SOURCES.index("preempt")] == 1  # the readmission resumed from it


@pytest.mark.parametrize("finish", ["length", "eos"])
def test_held_sample_can_finish_a_preempted_request(finish) -> None:
    """The preempted request's only remaining token was in flight: it arrives while the request waits and finishes
    it (a length or EOS finish), so it is never readmitted."""
    low = Spec(rid=0, prompt_len=0, prompt=stem(13, 40), max_tokens=1 if finish == "length" else 9, seed=51,
               priority=5, eos_after=0 if finish == "eos" else None, preempt_at=1)
    high = Spec(rid=1, prompt_len=0, prompt=stem(14, 12), max_tokens=2, seed=52, priority=0, arrive=1)
    h = Harness(specs=[low, high], vocab=VOCAB, max_num_seqs=1, max_tokens=64, page_size=4, num_pages=100,
                checkpoint_slots=4, grid=8, max_model_len=128, policy="priority")
    h.run(check_every_turn=True)
    h.check()
    r = h.results[0]
    assert len(r.tokens) == 1 and r.finish == (FinishReason.LENGTH if finish == "length" else FinishReason.STOP)
    assert h.sched.num_preemptions == 1 and h.sched._admissions == 2


def test_eviction_under_pressure_keeps_outputs_exact() -> None:
    """Two hot prefixes between unique ones, in a pool that cannot keep everything: LRU evicts the cold entries and
    keeps the hot ones hitting."""
    specs = [Spec(rid=i, prompt_len=0, prompt=(stem(i % 3, 60) if i % 3 < 2 else stem(100 + i, 60)) + [i],
                  max_tokens=5, seed=41 + i, arrive=i) for i in range(30)]
    h = Harness(specs=specs, vocab=VOCAB, max_num_seqs=2, max_tokens=128, page_size=4, num_pages=1 + 64,
                checkpoint_slots=4, grid=8, max_model_len=128)
    h.run(check_every_turn=True)
    h.check()
    c = h.sched.cache
    assert c.evicted_pages > 0 and c.evicted_ckpts > 0 and h.sched.prefix_hits > 0


def test_salt_isolates_namespaces() -> None:
    p = stem(4, 70)
    specs = [Spec(rid=0, prompt_len=0, prompt=p, max_tokens=2, seed=51),
             Spec(rid=1, prompt_len=0, prompt=p, max_tokens=2, seed=52, salt="tenant-b", arrive=10),
             Spec(rid=2, prompt_len=0, prompt=p, max_tokens=2, seed=53, arrive=20)]
    h = Harness(specs=specs, vocab=VOCAB, max_tokens=256, page_size=4, num_pages=200, checkpoint_slots=4, grid=16)
    h.run()
    h.check()
    assert [h.results[i].cached for i in range(3)] == [0, 0, 64]


def test_attention_only_hits_are_page_aligned() -> None:
    p = stem(5, 50)
    specs = [Spec(rid=0, prompt_len=0, prompt=p, max_tokens=3, seed=61),
             Spec(rid=1, prompt_len=0, prompt=[*p[:45], 3, 4], max_tokens=3, seed=62, arrive=10)]
    h = Harness(specs=specs, vocab=VOCAB, max_tokens=256, page_size=4, num_pages=100, hybrid=False)
    h.run(check_every_turn=True)
    h.check()
    assert h.results[1].cached == 44 and h.plans[-1]["state_copy"] == 0


def test_nonfinite_rolls_back_and_fails_dependents() -> None:
    """Request 0's first sample is NaN: what it wrote in that step leaves the cache; request 1, admitted in the next
    plan on its pending checkpoint, fails too; request 2, which arrives later, recomputes and is exact."""
    specs = [Spec(rid=0, prompt_len=0, prompt=shared(80, 5, 0), max_tokens=3, seed=71, nonfinite_after=0),
             Spec(rid=1, prompt_len=0, prompt=shared(80, 5, 1), max_tokens=3, seed=72),
             Spec(rid=2, prompt_len=0, prompt=shared(80, 5, 2), max_tokens=3, seed=73, arrive=20)]
    h = Harness(specs=specs, vocab=VOCAB, max_tokens=512, page_size=4, num_pages=200, checkpoint_slots=4, grid=16)
    h.run(check_every_turn=True)
    h.check(failed_ok=True)
    assert h.results[0].finish == FinishReason.ERROR and h.results[1].finish == FinishReason.ERROR
    assert h.results[2].finish == FinishReason.LENGTH and h.results[2].cached == 0
    assert h.sched.cache.rolled_back > 0


def test_no_prefix_caching_matches_the_cached_outputs() -> None:
    specs = [Spec(rid=i, prompt_len=0, prompt=shared(90, 7, i), max_tokens=6, seed=81 + i, arrive=i % 3)
             for i in range(6)]
    outs = []
    for caching in (True, False):
        h = Harness(specs=[Spec(**vars(s)) for s in specs], vocab=VOCAB, max_tokens=256, page_size=4,
                    num_pages=300, checkpoint_slots=4, grid=16, prefix_caching=caching)
        h.run()
        h.check()
        outs.append({rid: r.tokens for rid, r in h.results.items()})
    assert outs[0] == outs[1]


def test_livelock_under_page_pressure_regression() -> None:
    """A livelock under page pressure, reduced to three requests. Request 2 (high priority) needs pages for its
    first decode row while request 3's final chunk is in flight and every free page sits in tree entries that chunk
    writes. Eviction takes unlocked entries still in flight: were it to skip them, preempting request 3 would free
    nothing, request 2 would preempt itself, and the same plan would repeat forever (request 3 never getting its one
    token)."""
    tail1 = [32, 0, 0, 0, 0, 172, 949, 593, 110, 652, 63, 23, 975, 785, 969, 851, 156]
    tail2 = [39, 787, 646, 688, 861, 138, 847, 781, 37, 662, 564, 252, 199, 764, 48, 240, 736, 446, 644, 0]
    p1 = stem(0, 55) + tail1
    specs = [Spec(rid=1, prompt_len=0, prompt=p1, max_tokens=29, seed=31282448988319, priority=2),
             Spec(rid=2, prompt_len=0, prompt=p1 + tail2, max_tokens=24, seed=50747437347, arrive=10),
             Spec(rid=3, prompt_len=0, prompt=stem(0, 43) + [0] * 5, max_tokens=1, seed=40418636566466, arrive=29,
                  hints=[1], salt="s")]
    h = Harness(specs=specs, vocab=VOCAB, page_size=1, grid=8, max_model_len=160, max_num_seqs=3, max_tokens=123,
                num_pages=163, checkpoint_slots=7, policy="priority")
    h.run(max_turns=500, check_every_turn=True)
    check_plans(h)
    h.check()
    assert h.sched.num_preemptions < 10, h.sched.num_preemptions


def test_duplicate_pages_pinned_by_a_lock_are_freed() -> None:
    """Request 1 recomputes [0, 56) while request 0, which shares its first 50 tokens, still references the tree's
    pages for them, so the tree cannot adopt request 1's copies. Once request 0 is gone, its 50 pages are pinned by
    request 1's lock alone: holding two copies of its prefix, request 1 would never fit its next 64-token chunk in
    the pool, and every plan would be empty from then on. Under page pressure the scheduler re-inserts the locked
    span, which swaps the tree's copies for request 1's pages and frees them."""
    x = stem(31, 69)
    r = x[:50] + [(j * 11 + 294) % (VOCAB - 1) for j in range(87)]
    specs = [Spec(rid=0, prompt_len=0, prompt=x, max_tokens=10, seed=1287, ignore_eos=True),
             Spec(rid=1, prompt_len=0, prompt=r, max_tokens=2, seed=5287, arrive=6, ignore_eos=True)]
    h = Harness(specs=specs, vocab=VOCAB, page_size=1, grid=8, max_model_len=160, max_num_seqs=2, max_tokens=64,
                num_pages=163, checkpoint_slots=1)
    h.run(max_turns=200, check_every_turn=True)
    h.check()
    assert h.sched.cache.kept_dups > 0 and h.sched.num_preemptions == 0


def dedup_harness(workload: str) -> Harness:
    if workload == "tails":  # the GDN workload of test_junction_tails_leave_a_lone_request_its_bound
        return junction_tails_harness("gdn")
    if workload in ("recompute", "rollback"):
        # The test above, then a request that waits for pages once the duplicates are gone (`recompute`), or one that
        # resumes from the recomputing request's checkpoint and whose NaN takes the node it locked (`rollback`: the
        # workload of test_nan_safety.py's test_a_rollback_spares_requests_that_only_locked_the_removed_node)
        x = stem(31, 69)
        r = x[:50] + [(j * 11 + 294) % (VOCAB - 1) for j in range(87)]
        specs = [Spec(rid=0, prompt_len=0, prompt=x, max_tokens=10, seed=1287, ignore_eos=True),
                 Spec(rid=1, prompt_len=0, prompt=r, max_tokens=40, seed=5287, arrive=6, ignore_eos=True)]
        if workload == "rollback":
            specs.append(Spec(rid=2, prompt_len=0, prompt=x[:50] + [(j * 13 + 77) % (VOCAB - 1) for j in range(20)],
                              max_tokens=3, seed=6287, arrive=30, nonfinite_after=0))
            return Harness(specs=specs, vocab=VOCAB, page_size=1, grid=8, max_model_len=160, max_num_seqs=3,
                           max_tokens=64, num_pages=400, checkpoint_slots=4)
        specs.append(Spec(rid=2, prompt_len=0, prompt=stem(40, 120), max_tokens=4, seed=77, arrive=30))
        return Harness(specs=specs, vocab=VOCAB, page_size=1, grid=8, max_model_len=200, max_num_seqs=3,
                       max_tokens=64, num_pages=240, checkpoint_slots=1, decode_checkpoint_interval=16)
    a, b = stem(11, 30), stem(12, 30)  # ten requests over two prefixes, in an arena for about one
    specs = [Spec(rid=i, prompt_len=0, prompt=(a if i % 2 else b)[:20 + i % 7] + stem(20 + i, 12), max_tokens=24,
                  seed=100 + i, arrive=2 * i, ignore_eos=True) for i in range(10)]
    reserve = reserve_positions(2, 0)
    bound = pages_for_one_request(160, 20, reserve, True) + window_pages_for_one_request(16, 64, 4, reserve)
    return Harness(specs=specs, vocab=VOCAB, window=16, window_page_size=4, page_size=20, grid=4, max_num_seqs=6,
                   max_tokens=64, max_model_len=160, num_pages=int(1.2 * bound) + 1)


@pytest.mark.parametrize("workload", ["recompute", "window"])
def test_dedup_walks_only_what_its_full_passes_change(workload: str) -> None:
    """Under page pressure every admission that waits runs `Scheduler._dedup`. When no page is the tree's alone
    (`RadixCache.tree_only`) it walks only the requests whose `dup` flag may be stale (`_readopt`'s `settled`): the
    plans (every page id), each running request's flag after each of them, and the free list are those of the full
    passes at every call. `recompute`: a request recomputes a prefix another still holds, then, once that one is gone,
    takes those pages back in its own insert (a stale flag), and a third waits for pages: a settled pass clears the
    flag. `window`: ten requests in a window-mode arena for about one, most of their passes settled."""

    def run(full: bool) -> tuple[list, list[int], dict[str, int]]:
        h = dedup_harness(workload)
        s, seen = h.sched, {"settled": 0, "cleared": 0}
        tree_only, readopt = s.cache.tree_only, s._readopt

        def only() -> bool:
            seen["settled"] += not tree_only()
            return full or tree_only()

        def walk(force: bool = False, settled: bool = False) -> None:
            before = [r.dup for r in s.slot_req if r is not None]
            readopt(force, settled)
            seen["cleared"] += settled * sum(a and not r.dup for a, r in zip(before, filter(None, s.slot_req)))

        s.cache.tree_only, s._readopt = only, walk
        log, launch = [], h.runner.launch_forward

        def record(plan):
            log.append((plan.page_id.tolist(), plan.wpage_id.tolist(), [(r.slot, r.dup) for r in s.slot_req if r]))
            return launch(plan)

        h.runner.launch_forward = record
        h.run(max_turns=2000, check_every_turn=True)
        h.check()
        return log, list(s.kv._free), seen

    log, free, seen = run(full=False)
    assert (log, free) == run(full=True)[:2]
    assert seen["settled"] >= 20 and seen["cleared"] >= (workload == "recompute")


def watch_locked_paths(s) -> tuple[list[str], Counter]:
    """Check every call of a public method of the scheduler's cache: one that takes a page off a running request's
    locked path (from a node on it, or with the node) bumps `RadixCache.swaps`. Returns the calls that did not, and how
    many calls of each method took one."""
    cache, ps = s.cache, s.cache.ps
    bad, took = [], Counter()

    def held() -> set[tuple[int, int, int]]:
        out = set()
        for slot in np.flatnonzero(s.aid > 0).tolist():
            req = s.slot_req[slot]
            for node in (req.lock, req.plock):
                while node is not None and not node.dead and node.parent is not None:
                    out.update((slot, node.start // ps + j, p) for j, p in enumerate(node.pages))
                    if node.tail:
                        out.add((slot, node.end // ps, node.tail))
                    node = node.parent
        return out

    def watched(name, method):
        def call(*args, **kwargs):
            before, swaps = held(), cache.swaps
            out = method(*args, **kwargs)
            if before - held():
                took[name] += 1
                if cache.swaps == swaps:
                    bad.append(name)
            return out
        return call

    for name, method in inspect.getmembers(cache, inspect.ismethod):
        if not name.startswith("_"):
            setattr(cache, name, watched(name, method))
    return bad, took


@pytest.mark.parametrize("workload", ["recompute", "tails", "rollback"])
def test_pages_leave_a_locked_path_only_with_a_swap(workload: str) -> None:
    """`_readopt`'s `settled` passes skip a request whose duplicate was found since `RadixCache.swaps` last changed: it
    stays one while the request's locked path keeps its pages. So every cache call that takes a page off one bumps the
    count (`watch_locked_paths`): an adoption of a request's own pages (`recompute`: its insert takes them back once
    the other request is gone, `_adopt`), a junction's tail given a child's page (`tails`, `share_tails`), and a
    rollback, which takes a running request's locked nodes with their pages (`rollback`: the recomputing request
    locked the node a NaN takes, `remove_subtree`)."""
    h = dedup_harness(workload)
    bad, took = watch_locked_paths(h.sched)
    h.run(max_turns=2000, check_every_turn=True)
    h.check(failed_ok=workload == "rollback")
    assert not bad, bad
    assert took[{"recompute": "insert", "tails": "share_tails", "rollback": "remove_subtree"}[workload]] > 0, took


@pytest.mark.parametrize("mode", ["gdn", "bf16", "attention"])
def test_junction_tails_leave_a_lone_request_its_bound(mode: str) -> None:
    """Pages of 4, MTP k 1, a pool at the startup bound. A computes a prompt with a hint at 7 and B branches off it at
    5 (a hint there); then R, alone, resumes at 7 and runs to max_model_len. R's hit copies the page holding 4-7, and
    the tree never takes that copy into a node starting before the hit, so the junctions at 5 and 7, whose branches
    were evicted for R's pages, keep tails no child shares, pinned by R's lock: at its last decode step R would need
    one page more than the pool has, preempt itself, and never fit again (a livelock; the saved fuzz failure
    `cache_pressure-junction-tails-past-the-bound`). Under page pressure the junctions take a child's page for their
    tail (`RadixCache.share_tails`). `bf16`: the junctions are the precise namespace's; `attention`: attention-only,
    R's hit is the KV match at 8 and its drafter, which writes KV, makes it copy the page [4, 8) below the hit."""
    h = junction_tails_harness(mode)
    h.run(max_turns=1000, check_every_turn=True)
    h.check()
    hit = 8 if mode == "attention" else 7
    assert h.results[2].cached == hit and h.sched.num_preemptions == 0 and h.sched.cache.shared_tails > 0


def junction_tails_harness(mode: str) -> Harness:
    hit = 8 if mode == "attention" else 7
    s = stem(1, 8)
    specs = [Spec(rid=0, prompt_len=0, prompt=s[:hit] + stem(2, 10), hints=[hit], max_tokens=2, seed=1),
             Spec(rid=1, prompt_len=0, prompt=s[:5] + stem(3, 10), hints=[5], max_tokens=2, seed=2, arrive=10),
             Spec(rid=2, prompt_len=0, prompt=s[:hit] + stem(4, 100 - hit), max_tokens=60, seed=3, arrive=20,
                  ignore_eos=True)]
    pages = pages_for_one_request(160, 4, reserve_positions(2, 1), True) + 1
    return Harness(specs=specs, vocab=VOCAB, page_size=4, grid=4 if mode == "attention" else 8, max_num_seqs=2,
                   max_tokens=64, max_model_len=160, num_pages=pages, checkpoint_slots=8, mtp_k=1, accept_p=0.0,
                   hybrid=mode != "attention", precise="bf16" if mode == "bf16" else "fp8")


def test_an_attention_only_hit_locks_only_what_it_reads() -> None:
    """Attention-only (a hit is the page-aligned KV match, inside the node holding its last position), a pool at the
    startup bound. A computes 128 tokens in one chunk, one node; R, alone, shares A's first 8 and needs most of the
    pool. A lock on A's whole node would leave R's first chunk no room beside it, and nothing else would run: a
    livelock. The node is split at the hit (`RadixCache.end_at_hit`), and what lies past it is evicted."""
    a = stem(0, 128)
    specs = [Spec(rid=0, prompt_len=0, prompt=a, max_tokens=2, seed=1),
             Spec(rid=1, prompt_len=0, prompt=a[:8] + stem(5, 140), max_tokens=4, seed=2, arrive=6)]
    h = Harness(specs=specs, vocab=VOCAB, page_size=4, hybrid=False, max_num_seqs=2, max_tokens=128,
                max_model_len=160, num_pages=pages_for_one_request(160, 4, reserve_positions(2, 0), True) + 1)
    h.run(max_turns=200, check_every_turn=True)
    h.check()
    assert h.results[1].cached == 8 and h.sched.num_preemptions == 0


def test_chunk_starts_stay_on_the_grid() -> None:
    """A long prompt prefilled under a small budget: every budget cut lands on the grid while checkpoints remain
    ahead, so its mid-chunk checkpoint offsets are always multiples of the grid (the fake checks them)."""
    p = stem(6, 150)
    specs = [Spec(rid=0, prompt_len=0, prompt=p, max_tokens=2, seed=91, hints=[37]),
             Spec(rid=1, prompt_len=0, prompt=[*p[:120], 1], max_tokens=2, seed=92, arrive=1)]
    h = Harness(specs=specs, vocab=VOCAB, max_tokens=50, page_size=4, num_pages=200, checkpoint_slots=6, grid=16,
                max_model_len=256)
    h.run(check_every_turn=True)
    h.check()
    starts = [p0 for plan in h.plans for p0, dev in zip(plan["pos0"], plan["device"]) if not dev]
    assert all(s % 16 == 0 or s == 37 for s in starts if s), starts
    assert np.isin([37], starts).all()


def test_on_grid_hints_in_one_chunk_fit_the_plans_checkpoint_records() -> None:
    """A 1,000-token prompt hinted at every multiple of a grid of 16, prefilled in one chunk: on-grid hints do not cut
    the chunk, so each is a mid-chunk checkpoint of that step, and the plan sizes its records by the scheduler's grid
    (by the GDN chunk's 64, the step would hold more than its buffers do)."""
    spec = Spec(rid=0, prompt_len=0, prompt=stem(6, 1000), max_tokens=2, seed=91, hints=list(range(16, 1000, 16)))
    h = Harness(specs=[spec], vocab=VOCAB, max_tokens=1024, max_num_seqs=2, page_size=4, num_pages=2000,
                checkpoint_slots=80, grid=16, max_model_len=1100)
    h.run()
    h.check()
    assert h.results[0].tokens == output_of(spec)
    assert len(h.sched.builder.b.ckpt_dst) >= 1024 // 16
