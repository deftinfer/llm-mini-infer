"""Logprobs (docs/architecture.md, "Sampling"): the values (`runner.cuda.logprobs`) against their definition, on the
dense and the sparse head; the scheduler's path on the fake runner (every token's values, the forced prefix's and the
EOS-free finish's as a sampled step under their one-token masks gives them, MTP verify rows, preemption), with the plans
and tokens of every request the same with logprobs and without; and the targeted fuzz suite."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from hypothesis import assume, given

# Keeps Hypothesis' mined constants, and so a seed's draws, as they were when this file's seeds were chosen (not
# otherwise used).
import llm_mini_infer.config  # noqa: F401
from llm_mini_infer.runner.base import MAX_LOGPROBS
from llm_mini_infer.runner.cuda.logprobs import CHUNK, LogprobsPass, reference, sortable
from llm_mini_infer.sched.scheduler import forced_logprobs

from ..support.fake_grammar import spec
from ..support.forced_cases import structured_hybrid_workloads, structured_window_workloads
from ..support.fuzzing import campaign_settings, saved_on_failure, st
from ..support.harness import Harness, Spec
from ..support.logprob_workload import run_both, run_logprobs_workload
from ..support.mask_words import words_of
from ..support.mtp_workloads import mtp_workloads, with_counts
from ..support.sched_cases import VOCAB
from ..support.workloads import cache_workloads

V = 512


def run_pass(logits: torch.Tensor, tokens: torch.Tensor, rows: list[int], ids: torch.Tensor | None = None,
             mask: tuple[torch.Tensor, torch.Tensor] | None = None, sparse_rows: int = 0, vocab: int = V):
    """`LogprobsPass.run` on CPU over `logits` (the dense buffer's rows, or with `ids` the sparse head's) for the
    samples `rows`: (token values [n], top ids [n, K], top values [n, K])."""
    lp = LogprobsPass(len(logits), vocab, sparse_rows, torch.device("cpu"))
    (lp.sparse if ids is not None else lp.dense)[:len(logits)] = logits
    n = len(rows)
    idx = torch.tensor(rows + [0] * (-n % CHUNK), dtype=torch.int64)
    lp.run(1, idx, n, tokens, ids, mask)
    return lp.tok[1][:n].clone(), lp.ids[1][:n].clone(), lp.top[1][:n].clone()


def test_sortable_keys_order_as_the_floats() -> None:
    x = torch.tensor([float("-inf"), -3e38, -1.5, -1e-30, 0.0, 1e-30, 2.0, 3e38, float("inf")])
    s = sortable(x)
    assert (s[1:] > s[:-1]).all()


def test_the_dense_heads_values_are_the_definitions() -> None:
    """Rows with no mask, a mask of many tokens, of fewer than MAX_LOGPROBS (fillers), of one token (0 and the lowest
    other ids at −inf), and a step whose other samples have no mask; BF16-rounded logits (ties: the lowest id)."""
    g = torch.Generator().manual_seed(1)
    n = 11
    logits = (torch.randn((n, V), generator=g) * 3).bfloat16().float()
    allowed = np.ones((n, V), bool)
    allowed[1] = np.random.default_rng(1).random(V) < 0.3
    allowed[2, :] = False
    allowed[2, [5, 77, 300]] = True
    allowed[3, :] = False
    allowed[3, 41] = True
    words = words_of(allowed)
    mask_row = torch.tensor([-1, 1, 2, 3, -1, 1, 2, 3, -1, -1, 1], dtype=torch.int64)
    tokens = torch.tensor([int(a[i % len(a)]) if (m := int(mask_row[i])) >= 0 and len(a := np.flatnonzero(
        allowed[m])) else 7 + i for i in range(n)])
    tok, ids, top = run_pass(logits, tokens, list(range(n)), mask=(words, mask_row))
    for i in range(n):
        m = int(mask_row[i])
        al = torch.from_numpy(allowed[m]) if m >= 0 else None
        want_t, want_ids, want_top = reference(logits[i], int(tokens[i]), al)
        assert ids[i].tolist() == want_ids, i
        torch.testing.assert_close(tok[i], torch.tensor(want_t), rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(top[i], torch.tensor(want_top), rtol=1e-6, atol=1e-6)
    assert tok[3] == 0.0 and ids[3].tolist() == [41, *[j for j in range(MAX_LOGPROBS) if j != 41]][:MAX_LOGPROBS]
    assert top[3, 0] == 0.0 and torch.isinf(top[3, 1:]).all()
    assert sorted(ids[2, :3].tolist()) == [5, 77, 300] and \
        ids[2, 3:].tolist() == [j for j in range(MAX_LOGPROBS + 3) if j not in (5, 77, 300)][:MAX_LOGPROBS - 3]


def test_the_scheduler_gives_forced_tokens_the_values_of_their_one_token_masks() -> None:
    logits = torch.randn((1, V))
    allowed = np.zeros((1, V), bool)
    allowed[0, 7] = True
    tok, ids, top = run_pass(logits, torch.tensor([7]), [0], mask=(words_of(allowed), torch.tensor([0])))
    t, i, v = forced_logprobs([7])
    assert tok.tolist() == t.tolist() and ids.tolist() == i.tolist() and top.tolist() == v.tolist()


def test_ties_go_to_the_lowest_id_at_the_cut() -> None:
    """Many equal logits (ternary rows): the cut takes the lowest ids among them, in order."""
    x = torch.zeros((2, V))
    x[0, [3, 9, 400]] = 1.0  # three above, then 509 tied at 0
    x[1, ::2] = -1.0
    _, ids, _ = run_pass(x, torch.tensor([9, 1]), [0, 1])
    assert ids[0].tolist() == [3, 9, 400, *range(0, 3), *range(4, 9), *range(10, 10 + MAX_LOGPROBS - 11)]
    assert ids[1].tolist() == list(range(1, 2 * MAX_LOGPROBS, 2))
    assert reference(x[0], 9)[1] == ids[0].tolist() and reference(x[1], 1)[1] == ids[1].tolist()


def test_a_samples_values_do_not_depend_on_the_samples_beside_it() -> None:
    g = torch.Generator().manual_seed(2)
    logits = torch.randn((2 * CHUNK + 3, V), generator=g) * 4
    tokens = torch.randint(0, V, (len(logits),), generator=g)
    rows = list(range(len(logits)))
    tok, ids, top = run_pass(logits, tokens, rows)
    for i in (0, CHUNK - 1, CHUNK + 4, len(rows) - 1):
        t1, i1, v1 = run_pass(logits, tokens, [i])
        assert torch.equal(t1[0], tok[i]) and torch.equal(i1[0], ids[i]) and torch.equal(v1[0], top[i])


@pytest.mark.parametrize("cap", [48, 96])
def test_the_sparse_heads_values_are_the_dense_heads(cap) -> None:
    """The union's columns (ascending ids, −1 padding to the cap) of masked samples: the same tokens at the top, the
    same fillers (ids below 64 in the union or not), the values up to the sums' order."""
    rng = np.random.default_rng(3)
    g = torch.Generator().manual_seed(3)
    n = 6
    dense = torch.randn((n, V), generator=g) * 3
    allowed = np.zeros((n, V), bool)
    allowed[0, [2, 30, 31, 200]] = True
    allowed[1, rng.choice(np.arange(64, V), 25, replace=False)] = True
    allowed[2, 400] = True
    allowed[3, [1, 2, 3, 60, 61]] = True
    allowed[4, rng.choice(V, 12, replace=False)] = True
    allowed[5, [10, 300]] = True
    union = np.flatnonzero(allowed.any(0))
    assert len(union) <= cap
    ids = torch.full((cap,), -1, dtype=torch.int32)
    ids[:len(union)] = torch.from_numpy(union.astype(np.int32))
    sparse = dense[:, ids.clamp(min=0).long()]
    words, mrow = words_of(allowed), torch.arange(n)
    tokens = torch.tensor([int(np.flatnonzero(a)[-1]) for a in allowed])
    a = run_pass(dense, tokens, list(range(n)), mask=(words, mrow))
    b = run_pass(sparse, tokens, list(range(n)), ids=ids, mask=(words, mrow), sparse_rows=cap)
    assert torch.equal(a[1], b[1])
    torch.testing.assert_close(a[0], b[0], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(a[2], b[2], rtol=1e-6, atol=1e-6)


# -- the scheduler on the fake runner ----------------------------------------------------------------------------------


def _structured(mtp_k: int, **kw) -> dict:
    from ..support.sched_cases import stem

    specs = [Spec(rid=0, prompt_len=20, prompt=stem(0, 20), max_tokens=24, seed=11,
                  grammar=spec(5, 3, 9, lead=2, tail=1)),
             Spec(rid=1, prompt_len=30, prompt=stem(1, 30), max_tokens=12, seed=12),
             Spec(rid=2, prompt_len=12, prompt=stem(2, 12), max_tokens=20, seed=13, grammar=spec(6, 2, 14, one_in=3)),
             Spec(rid=3, prompt_len=9, prompt=stem(3, 9), max_tokens=16, seed=14, grammar=spec(7, 0, 4), arrive=3)]
    return {**dict(specs=specs, mtp_k=mtp_k, page_size=4, num_pages=200, max_num_seqs=4, max_tokens=64,
                   max_model_len=160, forced_prefix=True, eos_free_finish=True), **kw}


@pytest.mark.parametrize("k", [0, 1, 3, 5])
@pytest.mark.parametrize("window", [0, 16])
def test_every_token_gets_its_values_and_nothing_else_changes(k, window) -> None:
    """Greedy requests, constrained and not, with the forced prefix and the EOS-free finish, top counts 0, 1, 5, 20,
    and one asking none: the plans and tokens are those without logprobs, every token of a request that asked has its
    values (the forced ones' and the appended EOS's those of their one-token masks: `Harness.check`)."""
    extra = dict(window=window, window_page_size=4, page_size=8, draft_kv=False) if window else {}
    w = _structured(k, **extra)

    def run(w: dict) -> Harness:
        h = Harness(vocab=VOCAB, **w)
        h.run(check_every_turn=True)
        h.check()
        return h

    _, b = run_both(w, run, lambda s: {0: 20, 1: 0, 2: 5, 3: -1}[s.rid])
    assert b.sched.forced_tokens and int(b.sched.eos_free.sum())
    assert all(len(b.results[r].logprobs) == len(b.results[r].tokens) for r in (0, 1, 2))
    assert not b.results[3].logprobs


def test_the_forced_prefix_and_the_eos_free_finish_give_the_sampled_paths_values() -> None:
    """The same requests with both switches on and off: the same tokens and the same values (on, the scheduler gives
    the forced tokens and the EOS theirs; off, the fake samples them under one-token masks)."""
    on = _structured(3)
    off = {**_structured(3), "forced_prefix": False, "eos_free_finish": False}
    res = []
    for w in (on, off):
        for s in w["specs"]:
            s.logprobs = 5
        h = Harness(vocab=VOCAB, **w)
        h.run()
        h.check()
        res.append({rid: (r.tokens, r.logprobs) for rid, r in h.results.items()})
    assert res[0] == res[1]


@pytest.mark.parametrize("k", [0, 3])
def test_a_preempted_requests_values_go_with_its_tokens(k) -> None:
    """Preemptions with a sample in flight (its held row delivers it) and readmissions (their first sample is a final
    chunk's row's): every token's values once, as predicted."""
    specs = [Spec(rid=r, prompt_len=14, max_tokens=30, seed=100 + r, logprobs=3, preempt_at=6 + 3 * r)
             for r in range(3)]
    specs.append(Spec(rid=3, prompt_len=10, max_tokens=30, seed=200, grammar=spec(9, 4, 20, lead=2), logprobs=20,
                      preempt_at=5))
    h = Harness(specs=specs, vocab=VOCAB, mtp_k=k, page_size=4, num_pages=120, max_num_seqs=4, max_tokens=48,
                forced_prefix=True, eos_free_finish=True)
    h.run(check_every_turn=True)
    h.check()
    assert h.sched.num_preemptions >= 3


# -- the targeted fuzz suite -------------------------------------------------------------------------------------------


@st.composite
def logprobs_workloads(draw):
    """Another suite's workload (the scheduler's, MTP's, Gemma 4's and Qwen's structured ones, their switches, NaNs,
    empty masks, preemption, aborts, images) with logprobs asked by some of its requests (top counts 0 to 20)."""
    kind = draw(st.sampled_from(["cache", "mtp", "structured_window", "structured_hybrid"]))
    w = draw({"cache": cache_workloads(), "mtp": mtp_workloads(), "structured_window": structured_window_workloads(),
              "structured_hybrid": structured_hybrid_workloads()}[kind])
    tops = draw(st.lists(st.sampled_from([-1, -1, 0, 1, 5, 20]), min_size=len(w["specs"]), max_size=len(w["specs"])))
    return dict(kind=kind, workload=w, tops=tops)


@st.composite
def logprobs_count_workloads(draw):
    """`logprobs_workloads` of MTP's and the structured suites' kinds at k of 2 or more, with schedules of draft counts
    (`mtp_workloads.with_counts`): a request's values and the plans do not depend on its logprobs as the counts change.
    Not the policy: it does not time a step with logprobs rows (`Inflight.decode_only`), so its counts follow them."""
    kind = draw(st.sampled_from(["mtp", "structured_window", "structured_hybrid"]))
    w = draw({"mtp": mtp_workloads(), "structured_window": structured_window_workloads(),
              "structured_hybrid": structured_hybrid_workloads()}[kind])
    assume(w["mtp_k"] > 1)  # a count below k
    w = with_counts(draw, w, adaptive=False)
    tops = draw(st.lists(st.sampled_from([-1, -1, 0, 1, 5, 20]), min_size=len(w["specs"]), max_size=len(w["specs"])))
    return dict(kind=kind, workload=w, tops=tops)


@campaign_settings(150)
@given(logprobs_workloads())
def test_random_logprobs_workloads(w) -> None:
    with saved_on_failure("logprobs", w):
        run_logprobs_workload(w)


@campaign_settings(150)
@given(logprobs_count_workloads())
def test_random_logprobs_count_workloads(w) -> None:
    with saved_on_failure("logprobs_counts", w):
        run_logprobs_workload(w)


def test_the_scheduler_refuses_what_its_runner_cannot_answer() -> None:
    """More than MAX_LOGPROBS, or logprobs from a runner that computes none (`Capabilities.logprobs`): the request alone
    ends with an error (the frontend refuses both first)."""
    from llm_mini_infer.engine.messages import FinishReason

    specs = [Spec(rid=0, prompt_len=10, max_tokens=4, seed=1, logprobs=MAX_LOGPROBS + 1),
             Spec(rid=1, prompt_len=10, max_tokens=4, seed=2, logprobs=3),
             Spec(rid=2, prompt_len=10, max_tokens=4, seed=3)]
    h = Harness(specs=specs[:1] + specs[2:], vocab=VOCAB)
    h.run()
    assert h.results[0].finish == FinishReason.ERROR and h.results[2].finish == FinishReason.LENGTH
    h = Harness(specs=specs[1:], vocab=VOCAB)
    h.sched.logprobs = False
    h.run()
    assert h.results[1].finish == FinishReason.ERROR and h.results[2].finish == FinishReason.LENGTH
