"""Scheduler properties, against the validating fake runner (which raises on any contract violation).

Every run checks that each request's output is exactly the fake's prediction for its (seed, position) sequence,
cut at its finish condition: a result matched to the wrong request, a lost or duplicated token around a
preemption, or a wasted token that leaked to the client would all fail it.
"""

from __future__ import annotations

import msgspec
import numpy as np
import pytest
from hypothesis import given

from llm_mini_infer.engine.messages import ABORT_DISCONNECT, ABORT_SHUTDOWN, FinishReason
from llm_mini_infer.sched.plan import MASKED, SAMPLE, STATE_ZERO, PlanBuilder, StepPlan
from llm_mini_infer.sched.scheduler import Admit, Scheduler

from ..support.fuzzing import campaign_settings, saved_on_failure
from ..support.harness import EOS, Harness, Spec
from ..support.sched_cases import check_plans
from ..support.workloads import run_workload, workloads


@campaign_settings(300)
@given(workloads())
def test_random_workloads(w) -> None:
    with saved_on_failure("scheduler", w):
        run_workload(w)


def test_idle_before_a_step_launched_ahead_is_the_cpus() -> None:
    """GPU idle due to the CPU: the idle before a step's forward counts when the step was launched while another was
    in flight, not when none was (idle for lack of work); the idle between forward and sampling always counts. Each
    step's own flag decides (with the previous step's flag, and first_tokens_first off, the default, only the idle
    inside steps would count). The fake's device timing is replaced by fixed values: 1 us before each forward, 1 ns
    inside."""
    specs = [Spec(rid=1, prompt_len=9, max_tokens=6, seed=0x1D1E_0001),
             Spec(rid=2, prompt_len=5, max_tokens=4, seed=0x1D1E_0002, arrive=30)]  # after the device went idle
    h = Harness(specs=specs)
    launch, wait = h.runner.launch_forward, h.runner.wait
    in_flight: set[int] = set()
    ahead: dict[int, bool] = {}  # step id -> launched while another step was in flight

    def launch_forward(plan):
        ahead[plan.step_id] = bool(in_flight)
        in_flight.add(plan.step_id)
        return launch(plan)

    def timed_wait(handle):
        in_flight.discard(handle.step_id)
        return msgspec.structs.replace(wait(handle), gpu_idle_before_ns=1000, gpu_idle_inside_ns=1)

    h.runner.launch_forward, h.runner.wait = launch_forward, timed_wait
    h.run()
    h.check()
    n_ahead = sum(ahead.values())
    assert 0 < n_ahead < len(ahead) == h.core.steps  # each request's first step had nothing in flight
    assert h.core.gpu_idle_ns == 1000 * n_ahead + h.core.steps


def test_preemption_and_readmission_recompute_the_same_tokens() -> None:
    # Two long requests in a pool that holds one: the second preempts nothing (FCFS admits only what fits);
    # growth of the first two forces the lowest-priority one out and back in.
    specs = [Spec(rid=i, prompt_len=20, max_tokens=40, seed=(i + 1) * 0x9E3779B9) for i in range(3)]
    h = Harness(specs=specs, page_size=4, num_pages=1 + 30, max_num_seqs=3, max_tokens=64, max_model_len=128)
    h.run()
    check_plans(h)
    h.check()
    assert h.sched.num_preemptions > 0
    readmits = [p for p in h.plans if p["admit"] and p["step"] > 1]
    assert readmits, "a preempted request was readmitted"


def test_page_waits_and_the_per_step_histograms_are_counted() -> None:
    """llm_mini_infer:page_waits counts admissions and prefill chunks that wait for pages (here the pool holds about one
    request); llm_mini_infer:step_decode_rows counts every completed step's decode rows, in the engine."""
    specs = [Spec(rid=i, prompt_len=20, max_tokens=40, seed=(i + 1) * 0x9E3779B9) for i in range(3)]
    h = Harness(specs=specs, page_size=4, num_pages=1 + 30, max_num_seqs=3, max_tokens=64, max_model_len=128)
    h.run()
    h.check()
    assert h.sched.page_waits > 0
    core = h.core
    assert sum(core.rows_hist) == core.steps == sum(core.images_hist) and core.images_sum == 0
    assert core.steps == len(h.plans) and core.rows_sum == sum(sum(p["device"]) for p in h.plans) > 0
    assert core.rows_hist[0] == sum(1 for p in h.plans if not any(p["device"]))  # prefill-only steps


def test_finish_race_wastes_at_most_one_step() -> None:
    # Request 2 arrives later so that plans keep coming after request 1 finished.
    specs = [Spec(rid=1, prompt_len=5, max_tokens=50, seed=12345678901, eos_after=3),
             Spec(rid=2, prompt_len=5, max_tokens=20, seed=98765432109, arrive=3)]
    h = Harness(specs=specs)
    h.run()
    h.check()
    assert h.results[1].tokens[-1] == EOS and h.results[1].finish == FinishReason.STOP
    with_slot = [i for i, p in enumerate(h.plans) if 1 in p["slots"]]
    # prefill (token 0) + 3 decode steps to reach token 3 (EOS) + exactly one more step already in flight
    assert len(with_slot) == 1 + 3 + 1
    assert h.plans[with_slot[-1]]["device"][h.plans[with_slot[-1]]["slots"].index(1)]
    assert 1 in h.plans[with_slot[-1] + 1]["free"]  # freed by the very next plan


def test_length_finish_wastes_nothing() -> None:
    h = Harness(specs=[Spec(rid=1, prompt_len=5, max_tokens=4, seed=987654321987)])
    h.run()
    h.check()
    assert sum(1 in p["slots"] for p in h.plans) == 4  # prefill + 3 decodes, no step beyond max_tokens


def test_slot_reused_in_the_very_next_plan_results_by_position() -> None:
    # One slot. Request 1 finishes by EOS while its next step is in flight; request 2 takes the slot in the very
    # next plan, so the in-flight result for slot 1 belongs to request 1 and must not reach request 2.
    specs = [Spec(rid=1, prompt_len=6, max_tokens=50, seed=1111111111111, eos_after=2),
             Spec(rid=2, prompt_len=4, max_tokens=5, seed=2222222222222)]
    h = Harness(specs=specs, max_num_seqs=1)
    h.run()
    h.check()
    reuse = [p for p in h.plans if p["free"] == [1] and p["admit"] == [1]]
    assert reuse, "slot 1 was freed and readmitted in one plan"
    prev = h.plans[h.plans.index(reuse[0]) - 1]
    assert prev["slots"] == [1] and prev["device"] == [True]  # request 1's wasted step was in flight


def test_aborts_free_everything() -> None:
    specs = [Spec(rid=i, prompt_len=30, max_tokens=60, seed=10_000 + i, abort_at=3 + i) for i in range(4)]
    h = Harness(specs=specs, max_num_seqs=2, max_tokens=16)
    h.run()
    h.check()
    assert h.sched.num_aborts == 4 and h.sched.aborts == [4, 0, 0, 0]  # the harness's aborts are disconnects


def test_abort_all_counts_shutdowns_separately() -> None:
    specs = [Spec(rid=i, prompt_len=8, max_tokens=10, seed=20_000 + i) for i in range(3)]
    h = Harness(specs=specs)
    for spec in specs:
        h.add(spec)

    h.sched.abort_all()

    assert h.sched.num_aborts == 3 and not h.sched.reqs
    assert h.sched.aborts[ABORT_DISCONNECT] == 0
    assert h.sched.aborts[ABORT_SHUTDOWN] == 3


def test_priority_orders_admission_but_never_preempts() -> None:
    """The priority policy orders the waiting requests (the lowest value first, then arrival): a
    high-priority arrival waits for a slot rather than preempt a lower-priority request, then goes before an earlier
    lower-priority one."""
    low = [Spec(rid=i, prompt_len=8, max_tokens=40, seed=5_000_000_000 + i, priority=5) for i in range(3)]
    high = Spec(rid=9, prompt_len=8, max_tokens=4, seed=7_000_000_000, priority=0, arrive=4)
    h = Harness(specs=[*low, high], max_num_seqs=2, policy="priority")
    s, admit, order = h.sched, h.sched._admit_one, []

    def record(req):
        out = admit(req)
        if out is Admit.ADMITTED:
            order.append(req.rid)
        return out

    s._admit_one = record
    h.run()
    h.check()
    assert s.num_preemptions == 0 and order == [0, 1, 9, 2]


def test_the_running_order_takes_any_64_bit_priority() -> None:
    """Running requests in priority order, then arrival (preemption takes from the end): with priorities past 2^23 as
    well, which a sort key packing the priority above 40 bits of arrival wrapped (2^23 went first)."""
    prios = {0: 1 << 23, 1: 0, 2: -(1 << 62), 3: (1 << 63) - 1}
    specs = [Spec(rid=r, prompt_len=8, max_tokens=12, seed=6_000_000_000 + r, priority=p) for r, p in prios.items()]
    h = Harness(specs=specs, policy="priority")
    s, running_order, orders = h.sched, h.sched._running_order, []

    def record():
        out = running_order()
        orders.append([s.slot_req[slot].rid for slot in out.tolist()])
        return out

    s._running_order = record
    h.run()
    h.check()
    assert [2, 1, 0, 3] in orders and all(o == sorted(o, key=prios.get) for o in orders)


# Fields too large for where the engine keeps them (the plans' int32 top_k, the uint64 seed, the int64 priority), and a
# fault of any other kind
ADMISSION_FAULTS = {"top_k": {"top_k": 1 << 31}, "seed": {"seed": -1}, "priority": {"priority": 1 << 63}, "raises": {}}
SECRET = "the request's own text"  # the fault's message (its traceback's source lines show only the name)


@pytest.mark.parametrize("fault", list(ADMISSION_FAULTS))
def test_an_admission_failure_fails_only_its_request(fault, monkeypatch, caplog) -> None:
    """An exception in `Scheduler.add` is that request's ERROR finish, logged without the exception's message; nothing
    of it stays, and the engine serves the others."""
    import llm_mini_infer.sched.scheduler as scheduler

    class Failing(scheduler.Req):
        __slots__ = ()

        def __init__(self, msg, *args):
            if msg.rid == 0:
                raise RuntimeError(SECRET)
            super().__init__(msg, *args)

    specs = [Spec(rid=0, prompt_len=10, max_tokens=5, seed=1, stop_ids=[3]),
             Spec(rid=1, prompt_len=10, max_tokens=5, seed=2, arrive=1)]
    h = Harness(specs=specs, policy="priority")
    handle = h.core.handle

    def bad(msg):
        if msg.rid == 0:
            for name, value in ADMISSION_FAULTS[fault].items():
                setattr(msg, name, value)
        handle(msg)

    h.core.handle = bad
    if fault == "raises":
        monkeypatch.setattr(scheduler, "Req", Failing)
    h.run()
    h.check(failed_ok=True)
    assert h.results[0].finish == FinishReason.ERROR and h.rejected == {0}
    assert h.results[1].finish == FinishReason.LENGTH and len(h.results[1].tokens) == 5
    assert "admission failed" in caplog.text and SECRET not in caplog.text
    assert h.sched.num_waiting == 0 and not h.sched._waiting and h.sched._n_stop_reqs == 0


def test_a_negative_stop_id_keeps_the_stop_table_count() -> None:
    """A request's stop ids count in `_n_stop_reqs` while it runs, whatever the first one is: a negative first id once
    left the count up for the engine's life, and every step compared with the stop table."""
    h = Harness(specs=[Spec(rid=0, prompt_len=10, max_tokens=5, seed=1, stop_ids=[-5, 7]),
                       Spec(rid=1, prompt_len=10, max_tokens=5, seed=2, stop_ids=[-1])])
    h.run()
    h.check()
    assert h.sched._n_stop_reqs == 0


def test_decode_step_is_vectorized() -> None:
    """A steady decode step at batch 64 builds its plan without a Python loop over requests."""
    import time

    s = Scheduler(max_num_seqs=64, max_num_batched_tokens=2048, max_model_len=4096, page_size=16, num_pages=20_000,
                  num_state_slots=65, eos_token_ids=(EOS,))
    from llm_mini_infer.engine.messages import AddRequest

    for i in range(64):
        s.add(AddRequest(rid=i, frontend=0, prompt=np.arange(20, dtype=np.int32), max_tokens=10_000,
                         temperature=0.0, top_p=1.0, top_k=0, min_p=0.0, seed=i, stop_token_ids=[]))
    plan, inf = s.schedule()
    assert plan.n_sched == 64
    tokens = np.ones((64, 1), np.int32)
    n_out = np.ones(64, np.int32)
    s.process(inf, tokens, n_out, 0)
    t = []
    for _ in range(200):
        t0 = time.perf_counter()
        plan, inf = s.schedule()
        s.process(inf, tokens, n_out, 0)
        t.append(time.perf_counter() - t0)
    assert plan.n_sched == 64 and (plan.sched_tok_off == -1).all()
    assert sorted(t)[100] < 0.0005, f"median {sorted(t)[100] * 1e6:.0f} µs for schedule + process at batch 64"


def test_plan_builder_alternates_buffers() -> None:
    b = PlanBuilder(4, 16, 4)
    b.begin()
    b.add_device_rows(np.array([1, 2]))
    p1 = b.finish(1)
    b.begin()
    b.add_device_rows(np.array([3]))
    p2 = b.finish(2)
    assert p1.sched_slot.tolist() == [1, 2] and p2.sched_slot.tolist() == [3]


@pytest.mark.parametrize("images", [False, True])
def test_extend_row_is_the_row_built_with_its_tail(images) -> None:
    """`PlanBuilder.extend_row` (a final chunk's forced tokens, added once every CKPT record of the plan is placed)
    gives the plan that building the row with them would: device rows before and after it (their offsets stay −1),
    explicit rows before and after it (their tokens move up), and with images in the plan the tail's embedding rows
    (−1: text) and M-RoPE deltas (given), while the row's CKPT records (offsets from its pos0), its span (positions)
    and the encodes stay as they were; without images, no embedding or M-RoPE arrays."""
    forced = np.asarray([90, 91, 92], np.int32)

    def build(extend: bool) -> StepPlan:
        b = PlanBuilder(8, 128, 4, 256, 3)
        b.begin()
        b.add_device_rows(np.asarray([1, 2]))
        b.add_explicit_row(3, np.arange(10, 15, dtype=np.int32), 0, STATE_ZERO, 15)
        toks, emb, rope = np.arange(20, 37, dtype=np.int32), np.full(17, -1, np.int32), None
        if images:
            emb[4:8] = [5, 6, 7, 8]
            rope = np.tile(np.arange(17, dtype=np.int32) // 3, (3, 1))
            b.add_encode(0x7F00, 4, (1, 4, 4), (64, 64), 0, np.arange(5, 9))
        tail_rope = np.full((3, 3), 4, np.int32)
        if not extend:
            toks, emb = np.concatenate([toks, forced]), np.concatenate([emb, [-1] * 3])
            rope = None if rope is None else np.concatenate([rope, tail_rope], axis=1)
        b.add_explicit_row(4, toks, 40, SAMPLE | MASKED, -1, emb if images else None, -1, rope,
                           np.full(3, 2, np.int32) if images else None)
        b.add_ckpt(3, 8, 11)
        b.add_ckpt(3, 16, 12)
        b.add_span(3, 44, 48)
        b.add_explicit_row(5, np.arange(50, 56, dtype=np.int32), 8, 0, 56)
        b.add_device_rows(np.asarray([6]))
        b.add_explicit_row(7, np.arange(60, 64, dtype=np.int32), 0, STATE_ZERO | SAMPLE)
        b.add_ckpt(6, -1, 13)
        if extend:
            b.extend_row(3, forced, tail_rope if images else None)
        return b.finish(1)

    want, got = build(False), build(True)
    for name in StepPlan.__slots__:
        a, b = getattr(want, name), getattr(got, name)
        assert np.array_equal(a, b) and np.asarray(a).dtype == np.asarray(b).dtype, (name, a, b)
    assert got.sched_n_query.tolist() == [4, 4, 5, 20, 6, 4, 4] and got.max_query_len == 20
    assert got.sched_tok_off.tolist() == [-1, -1, 0, 5, 25, -1, 31] and got.n_query_tokens == 35 + 3 * 4
    assert got.ckpt_offset.tolist() == [8, 16, -1] and bool(len(got.tok_emb)) == bool(len(got.tok_rope[0])) == images


@pytest.mark.parametrize("k", [0, 3])
def test_row_token_bound_on_prefill_next_to_decode_rows(k) -> None:
    """With `mixed_step_row_tokens` K, a step that carries n decode rows takes at most K / n prefill tokens (at least
    one grid; verify rows count once whatever their 1 + k tokens), a step without decode rows still fills the whole
    budget, and a long prompt arriving while others decode keeps progressing every step until it is done; outputs
    stay as predicted. Fewer decode rows let the prompt take bigger chunks."""
    grid, row_tokens, budget = 16, 192, 128
    specs = [Spec(rid=i, prompt_len=10, max_tokens=60, seed=1000 + i) for i in range(4)]
    specs.append(Spec(rid=9, prompt_len=300, max_tokens=4, seed=4242, arrive=6))
    h = Harness(specs=specs, page_size=4, num_pages=400, max_num_seqs=6, max_tokens=budget, max_model_len=512,
                grid=grid, checkpoint_slots=4, mtp_k=k, accept_p=0.8, row_tokens=row_tokens)
    h.run()
    check_plans(h)
    h.check()
    long_rows = []  # after the first step (the four short prompts), every explicit row is the long prompt's
    for p in h.plans[1:]:
        rows = sum(p["device"])
        long_rows += [(rows, n) for n, d in zip(p["n_query"], p["device"]) if not d]
    assert long_rows, "the long prompt was scheduled"
    assert all(n <= max(grid, row_tokens // rows) for rows, n in long_rows if rows)
    assert sum(n for _, n in long_rows) == 300  # every chunk of it, none lost
    assert max(n for rows, n in long_rows if rows) >= grid  # chunks progress next to decode rows
    assert {rows for rows, _ in long_rows} == {4}  # the four short requests decode throughout
    first = h.plans[0]
    assert first["n_tokens"] == 40 and not any(first["device"])  # the first step: four prompts, no decode rows


def test_row_token_bound_gives_one_decode_row_big_chunks() -> None:
    """One request decoding next to a long prompt: the prompt takes K / 1 tokens a step (up to what the budget leaves),
    where a fixed mixed-step size would cut it as for any number of decode rows."""
    specs = [Spec(rid=0, prompt_len=10, max_tokens=40, seed=5), Spec(rid=1, prompt_len=240, max_tokens=2, seed=6,
                                                                     arrive=3)]
    h = Harness(specs=specs, page_size=4, num_pages=200, max_num_seqs=2, max_tokens=128, max_model_len=512, grid=16,
                checkpoint_slots=2, row_tokens=100)
    h.run()
    check_plans(h)
    h.check()
    chunks = [n for p in h.plans for n, d in zip(p["n_query"], p["device"]) if not d and p["step"] > 1]
    assert chunks[:2] == [96, 96], chunks  # K // 1 = 100, cut to the grid while the prompt's targets lie beyond


def test_row_token_bound_keeps_the_whole_budget_without_decode_rows() -> None:
    """A prompt longer than the bound, alone: no step carries decode rows, so its chunks use the whole token budget
    (the TTFT path at concurrency 1 is unchanged)."""
    h = Harness(specs=[Spec(rid=1, prompt_len=200, max_tokens=3, seed=77)], page_size=4, num_pages=200,
                max_num_seqs=2, max_tokens=64, max_model_len=256, grid=16, checkpoint_slots=2, row_tokens=24)
    h.run()
    h.check()
    prefill = [p["n_query"][0] for p in h.plans if not p["device"][0]]
    assert prefill == [64, 64, 64, 8]


def test_derive_mixed_step_row_tokens() -> None:
    from llm_mini_infer.sched.scheduler import MIXED_STEP_DELAY_MS, derive_mixed_step_row_tokens

    kw = dict(decode_rows=1, prefill_tokens=2048, max_num_batched_tokens=2048, max_num_seqs=64)
    # the 27B's shape: a lone request decodes in 23.6 ms, a 2,048-token prefill step takes 192 ms
    n, why = derive_mixed_step_row_tokens(decode_ms=23.6, prefill_ms=192.0, **kw)
    assert n == round(MIXED_STEP_DELAY_MS / (192.0 / 2048))
    assert "lone request" in why and "192.0 ms" in why and "next to 32" in why
    # Gemma 4 31B's in BF16: 39.0 and 406.4 ms, more costly prefill per token: a tighter bound, whatever its decode step
    g, _ = derive_mixed_step_row_tokens(decode_ms=39.0, prefill_ms=406.4, **kw)
    assert g == round(MIXED_STEP_DELAY_MS / (406.4 / 2048)) and g < n
    assert derive_mixed_step_row_tokens(decode_ms=20.0, prefill_ms=406.4, **kw)[0] == g
    # nothing measured: off, with the reason
    n, why = derive_mixed_step_row_tokens(decode_ms=0.0, prefill_ms=0.0, **kw)
    assert n == 0 and "no step costs" in why
    # never binding (a full budget next to max_num_seqs rows): off
    assert derive_mixed_step_row_tokens(decode_ms=1000.0, prefill_ms=1.0, **kw)[0] == 0


def test_penalties_and_bias_are_checked_at_add() -> None:
    """A request with penalties or a logit bias is taken only by a scheduler whose runner applies them
    (`penalties_vocab`), with at most 1,024 distinct, ascending bias ids in the vocabulary and a value each; its other
    checks still run (an image span outside the prompt, with no encoder here)."""
    from llm_mini_infer.engine.messages import AddRequest, ImageRef

    def add(s: Scheduler, rid: int, **kw) -> bool:
        s.add(AddRequest(rid=rid, frontend=0, prompt=np.arange(20, dtype=np.int32), max_tokens=4, temperature=0.0,
                         top_p=1.0, top_k=0, min_p=0.0, seed=rid, stop_token_ids=[], **kw))
        return (0, rid) in s.reqs

    def sched(vocab: int) -> Scheduler:
        return Scheduler(max_num_seqs=4, max_num_batched_tokens=64, max_model_len=256, page_size=4, num_pages=200,
                         num_state_slots=5, eos_token_ids=(EOS,), penalties_vocab=vocab)

    off, on = sched(0), sched(2048)
    assert add(off, 0) and add(off, 1, repetition_penalty=1.0, bias_ids=[], bias_values=[])  # neutral
    assert not add(off, 2, presence_penalty=0.5) and not add(off, 3, bias_ids=[4], bias_values=[1.0])
    assert add(on, 0, repetition_penalty=1.2, frequency_penalty=-1.0, bias_ids=[0, 4, 2047], bias_values=[1, -2, 3])
    assert add(on, 1, bias_ids=list(range(1024)), bias_values=[1.0] * 1024)
    for rid, ids, vals in ((2, [4, 4], [1, 1]), (3, [5, 4], [1, 1]), (4, [2048], [1]), (5, [-1], [1]), (6, [4], []),
                           (7, list(range(1025)), [1.0] * 1025)):
        assert not add(on, rid, bias_ids=ids, bias_values=vals), rid
    image = ImageRef(key=1, offset=30, tokens=4, grid_t=1, grid_h=2, grid_w=8, height=28, width=112, pixels=0)
    assert not add(on, 8, presence_penalty=0.5, images=[image])  # the image check runs too
    assert not add(on, 9, images=[image])
