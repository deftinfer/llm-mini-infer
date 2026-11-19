"""The CUDA runner's logic on CPU (torch reference kernels, fp32): slot tables, input building, the host mirror and
sampling, and the contract scenarios the fake runner checks, driven through the real scheduler and engine loop.

Outputs are compared with each request run alone (same seed and sampling parameters), which is the property the
contract promises: results by position, never by slot, and sampling a function of (seed, position) only.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from llm_mini_infer.engine.core import EngineCore
from llm_mini_infer.engine.messages import AddRequest, FinishReason
from llm_mini_infer.runner.base import PoolConfig
from llm_mini_infer.runner.fake import PlanError
from llm_mini_infer.sched.plan import SAMPLE, STATE_ZERO, PlanBuilder
from llm_mini_infer.sched.scheduler import Scheduler

from ..support.mtp_common import seq
from ..support.tiny_qwen import GREEDY, Driver, Row, Step, cpu_runner, make_tiny_model, pools

TOL = 2e-5
EOS = 511  # the tiny model's eos_token_id (vocab - 1)
SAMPLED = (0.9, 0.95, 40, 0.02)  # temperature, top_p, top_k, min_p


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    return make_tiny_model(tmp_path_factory.mktemp("m"), seed=1)


@pytest.fixture(scope="module")
def runner(model):
    return cpu_runner(model, pools(max_num_seqs=4, budget=128))


def configure(runner, **kw) -> PoolConfig:
    pc = pools(**kw)
    runner.configure(pc)
    return pc


def close(a: torch.Tensor, b: torch.Tensor) -> bool:
    return (a - b).abs().max().item() < TOL


# -- logits-level consistency ----------------------------------------------------------------------------------


@pytest.mark.parametrize("chunk", [1, 7, 32, 64])
def test_chunked_prefill_equals_single_shot(runner, chunk) -> None:
    s = seq(90)
    params = (*SAMPLED, 777)
    pc = configure(runner, budget=128)
    one = Driver(runner, pc)
    tok_one = one.run([Step(rows=[Row(1, s, 0)], admit=[(1, 1, params)])])[0].tokens[0, 0]
    pc = configure(runner, budget=128)
    many = Driver(runner, pc)
    steps = []
    for p0 in range(0, len(s), chunk):
        part = s[p0:p0 + chunk]
        steps.append(Step(rows=[Row(1, part, p0, sample=p0 + len(part) == len(s))],
                          admit=[(1, 1, params)] if p0 == 0 else []))
    res = many.run(steps)
    assert all(close(one.logits[(1, p)], many.logits[(1, p)]) for p in range(len(s)))
    assert res[-1].tokens[0, 0] == tok_one and all(r.n_out[0] == 0 for r in res[:-1])


def test_batch_invariance_sends_every_prompt_token_through_the_prefill(model) -> None:
    """Under batch invariance (CPU: the torch references) a 1-token chunk beside a decode row takes the GDN prefill
    kernel, not the decode kernel (only device rows do), with the same logits as the default routing within the
    references' rounding."""
    seen = []
    runs = {}
    for inv in (False, True):
        r = cpu_runner(model, pools(max_num_seqs=4, budget=128), batch_invariant=inv)
        prefill = r.gdn.prefill

        def spy(q, *a, prefill=prefill, inv=inv, **kw):
            seen.append((inv, q.shape[0]))
            return prefill(q, *a, **kw)

        r.gdn.prefill = spy
        d = Driver(r, r.pc)
        a, c = seq(30, 2), seq(31, 4)
        d.run([Step(rows=[Row(1, a, 0)], admit=[(1, 1, (*SAMPLED, 1))]),
               Step(rows=[Row(1, None, 30), Row(2, c[:30], 0, sample=False)], admit=[(2, 2, (*SAMPLED, 2))]),
               Step(rows=[Row(1, None, 31), Row(2, c[30:], 30)])])
        runs[inv] = d
    # per GDN layer, the third step's prefill: nothing without the switch (its 1-token row on the decode path), one
    # token with it
    g = len(r.cfg.gdn_layers)
    assert [n for inv, n in seen if not inv] == [30] * 2 * g and [n for inv, n in seen if inv] == [30] * 2 * g + [1] * g
    assert all(close(runs[False].logits[(2, p)], runs[True].logits[(2, p)]) for p in range(31))


def test_decode_after_prefill_equals_longer_prefill(runner) -> None:
    s, n_dec = seq(40, 1), 15
    params = (*SAMPLED, 4242)
    pc = configure(runner, budget=128)
    d = Driver(runner, pc)
    res = d.run([Step(rows=[Row(1, s, 0)], admit=[(1, 1, params)])] +
                [Step(rows=[Row(1, None, 40 + i)]) for i in range(n_dec)])
    gen = [int(r.tokens[0, 0]) for r in res]
    # A fresh slot prefills prompt + the first n_dec generated tokens in one chunk.
    long = s + gen[:n_dec]
    res2 = d.run([Step(rows=[Row(2, long, 0)], admit=[(2, 2, params)])])
    assert all(close(d.logits[(1, p)], d.logits[(2, p)]) for p in range(len(long)))
    assert int(res2[0].tokens[0, 0]) == gen[n_dec]  # same seed, same position: same token


def test_mixed_prefill_and_decode_batches_equal_alone(runner) -> None:
    """A decodes (device input) while B prefills in chunks and C is a 1-token prompt (an explicit 1-token row with
    STATE_ZERO, on the decode path): each matches its run alone."""
    a, b, c = seq(30, 2), seq(50, 3), [17]
    pa, pb, pcc = (*SAMPLED, 1), (*SAMPLED, 2), (*SAMPLED, 3)

    def alone(slot, prompt, params, n_dec):
        pc = configure(runner, budget=128)
        d = Driver(runner, pc)
        res = d.run([Step(rows=[Row(slot, prompt, 0)], admit=[(slot, slot, params)])] +
                    [Step(rows=[Row(slot, None, len(prompt) + i)]) for i in range(n_dec)])
        return d, [int(r.tokens[0, 0]) for r in res]

    da, ga = alone(1, a, pa, 6)
    db, gb = alone(2, b, pb, 3)
    dc, gc = alone(3, c, pcc, 3)
    pc = configure(runner, budget=64)
    d = Driver(runner, pc)
    steps = [Step(rows=[Row(1, a, 0)], admit=[(1, 1, pa)]),
             Step(rows=[Row(1, None, 30), Row(2, b[:20], 0, sample=False)], admit=[(2, 2, pb)]),
             Step(rows=[Row(1, None, 31), Row(2, b[20:], 20), Row(3, c, 0)], admit=[(3, 3, pcc)]),
             Step(rows=[Row(1, None, 32), Row(2, None, 50), Row(3, None, 1)]),
             Step(rows=[Row(1, None, 33), Row(2, None, 51), Row(3, None, 2)]),
             Step(rows=[Row(1, None, 34), Row(2, None, 52), Row(3, None, 3)]),
             Step(rows=[Row(1, None, 35)])]
    res = d.run(steps)
    toks: dict[int, list[int]] = {1: [], 2: [], 3: []}
    plan_rows = [[1], [1, 2], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1]]
    for r, slots in zip(res, plan_rows):
        for i, slot in enumerate(slots):
            if r.n_out[i]:
                toks[slot].append(int(r.tokens[i, 0]))
    assert toks == {1: ga, 2: gb, 3: gc}
    for slot, ref, n in ((1, da, 36), (2, db, 53), (3, dc, 4)):
        assert all(close(ref.logits[(slot, p)], d.logits[(slot, p)]) for p in range(n))


# -- the engine loop: the fake runner's contract scenarios -----------------------------------------------------


def run_engine(runner, reqs: list[dict], *, max_num_seqs: int, budget: int, num_pages: int = 256,
               page_size: int = 4, max_model_len: int = 256, chunk_cap: int = 0, policy: str = "fcfs",
               max_turns: int = 5000) -> tuple[dict[int, tuple[list[int], int]], list[dict], Scheduler]:
    pc = PoolConfig(page_size=page_size, num_pages=num_pages, num_state_slots=max_num_seqs + 1,
                    max_num_seqs=max_num_seqs, max_num_batched_tokens=budget, max_model_len=max_model_len,
                    eos_token_ids=(EOS,))
    runner.logits_hook = None
    runner.configure(pc)
    sched = Scheduler(max_num_seqs=max_num_seqs, max_num_batched_tokens=budget, max_model_len=max_model_len,
                      page_size=page_size, num_pages=num_pages, num_state_slots=max_num_seqs + 1, eos_token_ids=(EOS,),
                      long_prefill_token_threshold=chunk_cap, policy=policy)
    core = EngineCore(sched, runner)
    plans: list[dict] = []
    orig = runner.launch_forward

    def record(plan, _internal=False):
        plans.append({"free": plan.free_slot.tolist(), "admit": plan.admit_slot.tolist(),
                      "slots": plan.sched_slot.tolist(), "device": (plan.sched_tok_off < 0).tolist()})
        return orig(plan, _internal)

    runner.launch_forward = record
    out: dict[int, tuple[list[int], int]] = {}
    try:
        pending = sorted(reqs, key=lambda r: r.get("arrive", 0))
        for turn in range(max_turns):
            while pending and pending[0].get("arrive", 0) <= turn:
                r = pending.pop(0)
                out[r["rid"]] = ([], 0)
                core.handle(AddRequest(rid=r["rid"], frontend=0, prompt=np.asarray(r["prompt"], np.int32),
                                       max_tokens=r["max_tokens"], temperature=r.get("temperature", SAMPLED[0]),
                                       top_p=r.get("top_p", SAMPLED[1]), top_k=r.get("top_k", SAMPLED[2]),
                                       min_p=r.get("min_p", SAMPLED[3]), seed=r["seed"],
                                       stop_token_ids=r.get("stop", []), priority=r.get("priority", 0)))
            report = core.step()
            if report is not None:
                for fo in report.outputs.values():
                    off = 0
                    for rid, n, fin in zip(fo.rids.tolist(), fo.n_new.tolist(), fo.finish.tolist()):
                        toks, done = out[rid]
                        if not done:
                            toks.extend(fo.tokens[off:off + n].tolist())
                            out[rid] = (toks, fin)
                        off += n
            if not pending and not core.busy:
                break
        else:
            raise AssertionError("engine not idle")
    finally:
        del runner.launch_forward
    return out, plans, sched


def solo(runner, r: dict) -> list[int]:
    out, _, _ = run_engine(runner, [{**r, "arrive": 0}], max_num_seqs=1, budget=256)
    return out[r["rid"]][0]


def test_engine_device_input_mixed_batches_match_solo(runner) -> None:
    reqs = [{"rid": i, "prompt": seq(5 + 9 * i, i), "max_tokens": 6 + 3 * i, "seed": 1000 + i, "arrive": i}
            for i in range(6)]
    ref = {r["rid"]: solo(runner, r) for r in reqs}
    out, plans, sched = run_engine(runner, reqs, max_num_seqs=3, budget=24)
    assert {rid: toks for rid, (toks, _) in out.items()} == ref
    assert any(any(p["device"]) and not all(p["device"]) for p in plans), "mixed device-input and prefill rows"
    assert sched.num_running == 0 and sched.kv.num_free == sched.kv.num_pages - 1


def test_engine_slot_reused_in_the_next_plan(runner) -> None:
    """One slot. Request 1 stops on a stop token while its next step is in flight; request 2 takes the slot in the
    very next plan (FREE and ADMIT together), and the in-flight result (matched by position) never reaches it."""
    r1 = {"rid": 1, "prompt": seq(9, 5), "max_tokens": 20, "seed": 31}
    r2 = {"rid": 2, "prompt": seq(6, 6), "max_tokens": 5, "seed": 32}
    t1, t2 = solo(runner, r1), solo(runner, r2)
    r1["stop"] = [t1[2]]
    out, plans, _ = run_engine(runner, [r1, r2], max_num_seqs=1, budget=64)
    assert out[1] == (t1[:3], FinishReason.STOP) and out[2] == (t2, FinishReason.LENGTH)
    reuse = [i for i, p in enumerate(plans) if p["free"] == [1] and p["admit"] == [1]]
    assert reuse and plans[reuse[0] - 1]["device"] == [True]  # request 1's wasted step was in flight


def test_engine_preemption_and_recompute_reproduce_the_same_tokens(runner) -> None:
    """A KV pool too small for three growing requests: the lowest-priority one is preempted, readmitted and
    recomputed from its prompt and generated tokens; seeded sampling reproduces the same tokens."""
    reqs = [{"rid": i, "prompt": seq(20, 10 + i), "max_tokens": 40, "seed": 77 + i} for i in range(3)]
    ref = {r["rid"]: solo(runner, r) for r in reqs}
    out, _, sched = run_engine(runner, reqs, max_num_seqs=3, budget=64, num_pages=1 + 30, max_model_len=128)
    assert sched.num_preemptions > 0
    assert {rid: toks for rid, (toks, _) in out.items()} == ref


# -- contract violations -----------------------------------------------------------------------------------------


def test_drafts_without_mtp_raise(runner) -> None:
    pc = configure(runner, budget=64)
    b = PlanBuilder(pc.max_num_seqs, pc.max_num_batched_tokens, pc.page_size)
    b.begin()
    b.add_admit(1, 1, 0, *GREEDY)
    b.add_pages(1, 0, [1, 2])
    b.add_explicit_row(1, np.arange(5, dtype=np.int32), 0, STATE_ZERO | SAMPLE)
    plan = b.finish(1)
    plan.sched_n_draft = np.array([1], np.int32)
    with pytest.raises(PlanError, match="MTP is off"):
        runner.launch_forward(plan)


def test_plan_violations_raise(runner) -> None:
    pc = configure(runner, budget=64)
    b = PlanBuilder(pc.max_num_seqs, pc.max_num_batched_tokens, pc.page_size)
    b.begin()
    b.add_admit(1, 1, 0, *GREEDY)
    b.add_pages(1, 0, [1, 2])
    b.add_device_rows(np.array([1], np.int32))
    with pytest.raises(PlanError, match="device input before a SAMPLE"):
        runner.launch_forward(b.finish(1))
    configure(runner, budget=64)
    b.begin()
    b.add_admit(1, 1, 0, *GREEDY)
    b.add_pages(1, 0, [1, 2])
    b.add_explicit_row(1, np.arange(5, dtype=np.int32), 3, SAMPLE)
    with pytest.raises(PlanError, match="pos0"):
        runner.launch_forward(b.finish(2))


def _admit(b: PlanBuilder, slot: int, state_slot: int, pages: list[int]) -> None:
    b.add_admit(slot, state_slot, 0, *GREEDY)
    b.add_pages(slot, 0, pages)


def _row(b: PlanBuilder, slot: int) -> None:
    b.add_explicit_row(slot, np.arange(5, dtype=np.int32), 0, STATE_ZERO | SAMPLE)


def _launch(runner, pc: PoolConfig, *builds) -> None:
    """One plan per function in `builds`, in the engine's call order (launch N+1, wait N, sample N+1)."""
    b = PlanBuilder(pc.max_num_seqs, pc.max_num_batched_tokens, pc.page_size)
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


@pytest.mark.parametrize(("builds", "match"), [
    # `seen[slots] += 1` counted a repeated index once, so the old check never saw a slot twice
    pytest.param([lambda b: (_admit(b, 1, 1, [1, 2]), _row(b, 1), _row(b, 1))],
                 "a slot appears twice in SCHED", id="sched-slot-twice"),
    # slot -1 wrapped to the last slot (4), whose page table took the entry
    pytest.param([lambda b: (_admit(b, 1, 1, [1, 2]), _admit(b, 4, 2, [3, 4]), b.add_pages(-1, 1, [5]))],
                 "PAGES out of range or for a slot that is not admitted", id="pages-slot-minus-1"),
    pytest.param([lambda b: (_admit(b, 1, 1, [1, 2]), b.add_pages(5, 0, [5]))],
                 "PAGES out of range or for a slot that is not admitted", id="pages-slot-past-max"),
    pytest.param([lambda b: _admit(b, 1, 1, [1, 2]), lambda b: (b.add_free(1), b.add_free(1))],
                 r"a slot appears twice in FREE: \[1, 1\]", id="free-twice"),
    pytest.param([lambda b: (b.add_admit(1, 1, 0, *GREEDY), b.add_admit(1, 2, 0, *GREEDY),
                             b.add_pages(1, 0, [1, 2]))],
                 r"a slot appears twice in ADMIT: \[1, 1\]", id="admit-twice"),
    pytest.param([lambda b: (b.add_state_copy(1, 3), b.add_state_copy(2, 3))],
                 r"two STATE_COPYs into one slot: \[3, 3\]", id="state-copy-destination-twice"),
])
def test_repeated_or_out_of_range_slots_raise(runner, builds, match) -> None:
    """A slot twice in SCHED, FREE or ADMIT, a PAGES entry for a slot outside 1..max_num_seqs, and one STATE_COPY
    destination twice each raise PlanError."""
    pc = configure(runner, budget=64)
    with pytest.raises(PlanError, match=match):
        _launch(runner, pc, *builds)
    configure(runner, budget=64)


@pytest.mark.parametrize("builds", [
    pytest.param([lambda b: (_admit(b, 1, 1, [1, 2]), _row(b, 1)),
                  lambda b: (b.add_free(1), _admit(b, 1, 2, [3, 4]), _row(b, 1))], id="free-and-admit-one-slot"),
    pytest.param([lambda b: (b.add_admit(1, 2, 0, *GREEDY), b.add_pages(1, 0, [1, 2]), b.add_state_copy(3, 2),
                             _row(b, 1))], id="state-copy-into-the-admits-state-slot"),
    pytest.param([lambda b: (_admit(b, 1, 1, [1, 2]), _row(b, 1)),
                  lambda b: (b.add_free(1), b.add_admit(2, 1, 0, *GREEDY), b.add_pages(2, 0, [3, 4]),
                             b.add_state_copy(3, 1), _row(b, 2))], id="state-copy-into-a-readmitted-state-slot"),
])
def test_slot_reuse_in_one_plan_is_valid(runner, builds) -> None:
    """The checks are within one list: a FREE and an ADMIT of one slot in one plan, a STATE_COPY into the state slot
    an ADMIT takes, and one into a state slot freed and admitted again in the same plan all pass."""
    pc = configure(runner, budget=64)
    _launch(runner, pc, *builds)
    configure(runner, budget=64)


def test_the_mirror_check_sees_device_tables_that_disagree(runner) -> None:
    """`check_mirror` (dev.runner.cuda.check) after a step that samples slot 1 and admits slot 2 mid-prompt: a pending
    token for the slot that has not sampled, none for the one that has, or a seq_len outside the host mirror's range
    each raise."""
    from llm_mini_infer.runner.cuda.runner import check_mirror

    pc = configure(runner, budget=64)
    Driver(runner, pc).run([Step(rows=[Row(1, seq(20), 0), Row(2, seq(30, 1)[:10], 0, sample=False)],
                                 admit=[(1, 1, GREEDY), (2, 2, GREEDY)])])
    tb, m = runner.tables, runner.mirror
    assert m.sampled[1] and not m.sampled[2] and m.admitted[1:3].all()
    check_mirror(tb, m, runner.cuda)
    for table, slot, bad, what in ((tb.pending, 2, 5, "pending"), (tb.pending, 1, -1, "pending"),
                                   (tb.seq_len, 1, 99, "seq_len")):
        with torch.inference_mode():
            good = int(table[slot])
            table[slot] = bad
        with pytest.raises(AssertionError, match=what):
            check_mirror(tb, m, runner.cuda)
        with torch.inference_mode():
            table[slot] = good
    check_mirror(tb, m, runner.cuda)
