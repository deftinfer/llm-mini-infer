"""The penalties' and logit bias's pieces on CPU (`runner.cuda.penalties`, `runner.cuda.sampler`): the reference's terms
(`apply_terms`) equal vLLM's formula (`support.penalty_cases.vllm_terms`) bit for bit, on dense and sparse columns,
with a verify window's drafts; rows without terms are untouched; `sample` with them is `sample` on the adjusted logits,
the non-finite check still the raw logits'; the clamp keeps every candidate finite. The bookkeeping: an admission's
row is rewritten whole (no bit of the plan buffer's previous use leaks in), explicit rows' prompt and output tokens,
the commits' repeats accumulate. `PlanBuilder` sets ADMIT_PENALTIES itself."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from llm_mini_infer.runner.cuda.penalties import (
    BIASED,
    COUNT,
    PROMPT,
    TERM_BIAS,
    TERM_PEN,
    PenaltyTables,
    apply_admit,
    apply_explicit,
    commit_mtp,
    stage_explicit,
)
from llm_mini_infer.runner.cuda.runner import _Staging, stage_terms
from llm_mini_infer.runner.cuda.sampler import F_MAX, NONFINITE, apply_terms, plan_sampling, row_keys, sample
from llm_mini_infer.sched.plan import ADMIT_PENALTIES, MAX_LOGIT_BIAS, SAMPLE, STATE_ZERO, PlanBuilder

from ..support.penalty_cases import vllm_terms

V = 300


def case(rng: np.random.Generator, n: int) -> list[dict]:
    """n rows: a prompt, outputs (repeats), penalties (some neutral), a bias (some none)."""
    rows = []
    for i in range(n):
        rows.append(dict(prompt=rng.integers(0, V, rng.integers(1, 40)).tolist(),
                         outs=rng.integers(0, V // 3, rng.integers(0, 30)).tolist(),
                         rep=[1.0, 1.3, 0.6, 2.0][i % 4], pres=[0.0, 0.5, -1.2, 2.0][(i + 1) % 4],
                         freq=[0.0, 1.0, -2.0, 0.3][(i + 2) % 4],
                         bias={int(t): float(b) for t, b in zip(rng.choice(V, 5, replace=False),
                                                                 rng.uniform(-100, 100, 5))} if i % 3 else {}))
    return rows


def tables(rows: list[dict]) -> SimpleNamespace:
    """The device state of `rows` as the bookkeeping leaves it (per row: its words, bias table and parameters)."""
    n = len(rows)
    cnt = torch.zeros((n, V), dtype=torch.int32)
    bid = torch.zeros((n, MAX_LOGIT_BIAS), dtype=torch.int32)
    bval = torch.zeros((n, MAX_LOGIT_BIAS), dtype=torch.float32)
    terms = torch.zeros(n, dtype=torch.int32)
    for i, r in enumerate(rows):
        cnt[i, r["prompt"]] |= PROMPT
        for t in r["outs"]:
            cnt[i, t] += 1
        for j, (t, b) in enumerate(sorted(r["bias"].items())):
            cnt[i, t] |= BIASED
            bid[i, j], bval[i, j] = t, b
        pen = (r["rep"], r["pres"], r["freq"]) != (1.0, 0.0, 0.0)
        terms[i] = (TERM_PEN if pen else 0) | (TERM_BIAS if r["bias"] else 0)
    f = lambda k: torch.tensor([r[k] for r in rows], dtype=torch.float32)  # noqa: E731
    return SimpleNamespace(cnt=cnt, bias_id=bid, bias_val=bval, bias_n=torch.tensor([len(r["bias"]) for r in rows],
                                                                                   dtype=torch.int32),
                           terms=terms, rep=f("rep"), pres=f("pres"), freq=f("freq"))


def terms_of(logits: torch.Tensor, t: SimpleNamespace, cols: torch.Tensor | None = None,
             window: torch.Tensor | None = None) -> torch.Tensor:
    cols = torch.arange(logits.shape[1]) if cols is None else cols
    return apply_terms(logits, cols, t.terms, t.rep, t.pres, t.freq, t.cnt, t.bias_id, t.bias_val, t.bias_n, window)


def test_the_terms_are_vllms() -> None:
    rng = np.random.default_rng(0)
    rows = case(rng, 12)
    logits = torch.from_numpy(rng.normal(0, 4, (12, V)).astype(np.float32))
    got = terms_of(logits, tables(rows))
    for i, r in enumerate(rows):
        want = vllm_terms(logits[i], r["prompt"], r["outs"], r["rep"], r["pres"], r["freq"], r["bias"])
        assert torch.equal(got[i], want), i


def test_the_terms_on_sparse_columns_and_windows() -> None:
    """Columns of a reduced head (ascending ids, then padding) get their tokens' terms; a verify row's drafts before
    it count once more each (a repeated draft twice)."""
    rng = np.random.default_rng(1)
    rows = case(rng, 6)
    full = torch.from_numpy(rng.normal(0, 4, (6, V)).astype(np.float32))
    ids = torch.from_numpy(np.sort(rng.choice(V, 120, replace=False)))
    for r in rows:  # biased and penalised tokens among the columns
        ids = torch.unique(torch.cat([ids, torch.tensor(list(r["bias"]) + r["outs"][:3], dtype=torch.int64)]))
    cols = torch.cat([ids, torch.full((7,), -1)])
    t = tables(rows)
    window = torch.tensor([[r["outs"][0] if r["outs"] else 5, 9, 9, -1] for r in rows])
    got = terms_of(full[:, cols.clamp(min=0)], t, cols, window)
    for i, r in enumerate(rows):
        drafts = [d for d in window[i].tolist() if d >= 0]
        want = vllm_terms(full[i], r["prompt"], r["outs"] + drafts, r["rep"], r["pres"], r["freq"], r["bias"])
        assert torch.equal(got[i, :len(ids)], want[ids]), i


def test_rows_without_terms_are_untouched() -> None:
    rng = np.random.default_rng(2)
    rows = case(rng, 4)
    for r in rows:
        r.update(rep=1.0, pres=0.0, freq=0.0, bias={})
    logits = torch.from_numpy(rng.normal(0, 4, (4, V)).astype(np.float32))
    logits[0, 3] = -0.0
    assert torch.equal(terms_of(logits, tables(rows)), logits)
    assert torch.signbit(terms_of(logits, tables(rows))[0, 3])


@pytest.mark.parametrize("params", [(0.0, 1.0, 0, 0.0), (0.9, 1.0, 12, 0.0), (1.0, 0.85, 0, 0.0),
                                    (0.7, 0.9, 40, 0.05), (1.2, 1.0, 0, 0.1), (1.0, 1.0, 0, 0.0)])
def test_sampling_with_terms_is_sampling_the_adjusted_logits(params: tuple) -> None:
    """Each sampler path (greedy, top-k, top-p, top-k and min-p, min-p, plain), masks on some rows: the tokens with
    the terms are those of `sample` on the adjusted logits; the status is the raw logits' (a NaN the terms cannot
    hide)."""
    rng = np.random.default_rng(3)
    n = 8
    rows = case(rng, n)
    t = tables(rows)
    raw = torch.from_numpy(rng.normal(0, 3, (n, V)).astype(np.float32))
    raw[5, 17] = float("nan")
    adj = terms_of(raw, t)
    p = [np.full(n, x, dt) for x, dt in zip(params, (np.float32, np.float32, np.int64, np.float32))]
    plan = plan_sampling(*p, V)
    for g in plan.groups:
        g.idx = torch.as_tensor(g.rows)
    temp, top_p, top_k, min_p = (torch.from_numpy(a) for a in p)
    keys = row_keys(torch.arange(n) + 7, torch.arange(n) * 3 + 40)
    mask = torch.from_numpy(rng.integers(-(1 << 31), 1 << 31, (2, -(-V // 32)), dtype=np.int64).astype(np.int32))
    mask_row = torch.tensor([-1, 0, -1, 1, -1, -1, 0, -1])
    args = (temp, top_p, top_k, min_p, keys, torch.arange(V))
    got, st = sample(raw, plan, *args, mask=mask, mask_row=mask_row, adjusted=adj, clamp_rows=t.terms != 0)
    want, _ = sample(torch.where(torch.isnan(raw), 0.0, adj), plan, *args, mask=mask, mask_row=mask_row)
    ok = torch.arange(n) != 5
    assert torch.equal(got[ok], want[ok]) and int(st[5]) == NONFINITE and int(got[5]) == 0
    assert not bool(st[ok].any())


def test_the_clamp_keeps_candidates_finite() -> None:
    """A repetition penalty at float32's smallest normal number scales a seen positive logit past the largest float,
    and a temperature below 1 scales x past it: both clamped, the rows still sample valid tokens, greedy or not."""
    rows = [dict(prompt=[4, 5], outs=[6], rep=1.1754944e-38, pres=0.0, freq=0.0, bias={}) for _ in range(2)]
    logits = torch.full((2, V), -1.0)
    logits[:, 4], logits[:, 5], logits[:, 7] = 5.0, 2.0, 50.0
    adj = terms_of(logits, tables(rows))
    assert float(adj[0, 4]) == F_MAX and float(adj[0, 7]) == 50.0 and float(adj[0, 6]) < 0
    assert torch.isfinite(adj).all()
    p = [np.array([0.0, 0.5], np.float32), np.ones(2, np.float32), np.zeros(2, np.int64), np.zeros(2, np.float32)]
    plan = plan_sampling(*p, V)
    for g in plan.groups:
        g.idx = torch.as_tensor(g.rows)
    keys = row_keys(torch.tensor([1, 2]), torch.tensor([9, 9]))
    toks, st = sample(logits, plan, *(torch.from_numpy(a) for a in p), keys, torch.arange(V), adjusted=adj,
                      clamp_rows=torch.tensor([True, True]))
    assert int(toks[0]) == 4 and int(toks[1]) in (4, 5) and not bool(st.any())


def staging(cap: int = 4096) -> _Staging:
    return _Staging(torch.device("cpu"), 8 * cap, cap, 8, 4, 4, cap_b=cap)


def mirror(slots: int) -> SimpleNamespace:
    return SimpleNamespace(terms=np.zeros(slots, np.int32), prompt_len=np.zeros(slots, np.int64))


def admit_step(b: PlanBuilder, st: _Staging, m: SimpleNamespace, pt: PenaltyTables, step: int,
               admits: list[dict]) -> None:
    """One plan of ADMITs through both runners' staging and device writes."""
    b.begin()
    for a in admits:
        b.add_admit(a["slot"], a["slot"], a.get("seq", 0), 0.0, 1.0, 0, 0.0, 5, rep=a.get("rep", 1.0),
                    prompt_len=a.get("plen", 0), prompt=np.asarray(a.get("prompt", []), np.int32),
                    bias_ids=a.get("bias_ids", ()), bias_val=a.get("bias_val", ()))
    plan = b.finish(step)
    st.reset()
    adm = plan.admit_slot.astype(np.int64)
    fx = stage_terms(plan, st, adm, plan.admit_seq_len.astype(np.int64), m, pt, 1024)
    st.upload(None)
    apply_admit(pt, fx, st)


def test_an_admission_rewrites_its_row_whole() -> None:
    """Plans alternate between two buffers: a penalised admission two plans after another one in the same buffer row
    gets its own bits only (none, with no cached prefix), and an unpenalised admission writes no row."""
    pt, st, m = PenaltyTables(6, V, torch.device("cpu")), staging(), mirror(6)
    b = PlanBuilder(5, 64, 4, 1024, vocab=V)
    admit_step(b, st, m, pt, 1, [dict(slot=1, seq=3, plen=8, rep=1.2, prompt=[5, 9, 5, 2], bias_ids=[3, 280],
                                      bias_val=[1.5, -100.0])])
    assert sorted(torch.nonzero(pt.cnt[1] < 0)[:, 0].tolist()) == [5, 9]  # the first 3 positions: 5, 9, 5
    assert sorted(torch.nonzero(pt.cnt[1] & BIASED)[:, 0].tolist()) == [3, 280]
    assert int(pt.bias_n[1]) == 2 and pt.bias_id[1, :2].tolist() == [3, 280] and pt.bias_val[1, 1] == -100.0
    assert int(pt.terms[1]) == TERM_PEN | TERM_BIAS and float(pt.rep[1]) == pytest.approx(1.2)
    admit_step(b, st, m, pt, 2, [dict(slot=2, seq=1, plen=4, rep=1.5, prompt=[7])])
    pt.cnt[3] = 12345  # garbage of an earlier occupant
    admit_step(b, st, m, pt, 3, [dict(slot=3, seq=0, plen=4, rep=1.1), dict(slot=4)])  # buffer 1 again
    assert not bool(pt.cnt[3].any()) and int(pt.terms[4]) == 0 and m.terms[3] == TERM_PEN
    assert sorted(torch.nonzero(pt.cnt[2] < 0)[:, 0].tolist()) == [7]


def test_explicit_rows_and_commits_count() -> None:
    """A penalised slot's explicit tokens below its prompt length get the prompt bit (a repeat changes nothing), the
    later ones count (repeats accumulate); an unpenalised slot's are not looked at; MTP commits accumulate."""
    pt, st, m = PenaltyTables(4, V, torch.device("cpu")), staging(), mirror(4)
    b = PlanBuilder(3, 64, 4, 1024, vocab=V)
    admit_step(b, st, m, pt, 1, [dict(slot=1, plen=5, rep=1.3), dict(slot=2)])
    b.begin()
    b.add_explicit_row(1, np.array([4, 8, 4, 8, 9, 9, 9, 4], np.int32), 0, STATE_ZERO | SAMPLE)  # prompt: 5 first
    b.add_explicit_row(2, np.array([1, 2, 3], np.int32), 0, STATE_ZERO | SAMPLE)
    plan = b.finish(2)
    st.reset()
    e = stage_explicit(plan, m, st, V)
    st.upload(None)
    apply_explicit(pt, e, st)
    w = pt.cnt[1]
    assert sorted(torch.nonzero(w < 0)[:, 0].tolist()) == [4, 8, 9]
    assert int(w[9] & COUNT) == 2 and int(w[4] & COUNT) == 1 and int(w[8] & COUNT) == 0
    assert not bool(pt.cnt[2].any())
    yw = torch.tensor([[7, 7, 3, 7], [5, 5, 5, 5], [6, 0, 0, 0], [8, 0, 0, 0]])
    nw = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]])  # the last: a non-finite sample
    commit_mtp(pt, torch.tensor([1, 2, 1, 1]), yw, torch.tensor([3, 4, 1, 1]), nw, torch.arange(4))
    assert int(pt.cnt[1, 7] & COUNT) == 2 and int(pt.cnt[1, 3] & COUNT) == 1 and int(pt.cnt[1, 6] & COUNT) == 1
    assert int(pt.cnt[1, 4] & COUNT) == 1 and int(pt.cnt[1, 8] & COUNT) == 0 and not bool(pt.cnt[2].any())


def test_the_builder_sets_the_flag() -> None:
    b = PlanBuilder(3, 64, 4, 1024, vocab=V)
    b.begin()
    b.add_admit(1, 1, 0, 0.0, 1.0, 0, 0.0, 5, flags=ADMIT_PENALTIES)  # neutral: cleared
    b.add_admit(2, 2, 0, 0.0, 1.0, 0, 0.0, 5, pres=0.5)
    b.add_admit(3, 3, 0, 0.0, 1.0, 0, 0.0, 5, bias_ids=[2], bias_val=[1.0])  # a bias alone: no flag
    p = b.finish(1)
    assert [(int(f) & ADMIT_PENALTIES) != 0 for f in p.admit_flags] == [False, True, False]
    assert p.admit_bias_n.tolist() == [0, 0, 1] and p.admit_prompt_bits.shape == (3, -(-V // 32))
    plain = PlanBuilder(3, 64, 4, 1024)
    plain.begin()
    plain.add_admit(1, 1, 0, 0.0, 1.0, 0, 0.0, 5)
    assert plain.finish(1).admit_bias_ids.shape == (1, 0)
    for kw in (dict(rep=1.1), dict(bias_ids=[1], bias_val=[1.0])):
        with pytest.raises(ValueError, match="vocab"):
            plain.add_admit(2, 2, 0, 0.0, 1.0, 0, 0.0, 5, **kw)


def test_the_staging_holds_a_worst_step() -> None:
    """Every slot admitted at once, each penalised with a cached prefix and 1,024 bias entries, beside a full budget of
    a penalised slot's explicit tokens: what `PenaltyTables.staging` reserves holds it."""
    slots, budget = 4, 64
    pt, m = PenaltyTables(slots + 1, V, torch.device("cpu")), mirror(slots + 1)
    pi, pf, pb = pt.staging(slots + 1, budget)
    st = _Staging(torch.device("cpu"), pi, pf, 8, 4, 4, cap_b=pb)
    b = PlanBuilder(slots, budget, 4, 1024, vocab=V)
    ids = list(range(0, 2 * MAX_LOGIT_BIAS, 2))[:MAX_LOGIT_BIAS]
    ids = [i % V for i in ids]
    ids = sorted(set(ids))
    b.begin()
    for s in range(1, slots + 1):
        b.add_admit(s, s, 3, 0.0, 1.0, 0, 0.0, 5, rep=1.2, prompt_len=200, prompt=np.arange(3, dtype=np.int32),
                    bias_ids=ids, bias_val=[1.0] * len(ids))
    b.add_explicit_row(1, np.arange(budget, dtype=np.int32) % V, 3, SAMPLE)
    plan = b.finish(1)
    adm = plan.admit_slot.astype(np.int64)
    fx = stage_terms(plan, st, adm, plan.admit_seq_len.astype(np.int64), m, pt, 1024)
    e = stage_explicit(plan, m, st, V)
    st.upload(None)
    apply_admit(pt, fx, st)
    apply_explicit(pt, e, st)
    assert int(pt.bias_n[slots]) == len(ids) and int((pt.cnt[1] < 0).sum()) == budget
