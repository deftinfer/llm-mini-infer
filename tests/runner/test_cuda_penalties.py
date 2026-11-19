"""Penalties and the logit bias in both CUDA runners on CPU (torch reference kernels, fp32), through the real
scheduler and engine loop (`runner.cuda.penalties`, `server.penalties`): every sampled token is vLLM's (the oracle of
`support.penalty_cases`: its terms over the prompt and the tokens before, on the logits the runner computed there),
greedy and seeded, on each sampler path, under a grammar whose forced prefix the device never samples; a request's
tokens do not depend on its neighbours, on MTP (the drafts before each verify row count, the accepted ones commit), on
a preemption (its outputs recomputed) or on a prefix-cache hit (its prompt's bits come with the admission); and the
setting changes nothing for requests without terms."""

from __future__ import annotations

import numpy as np
import pytest

from llm_mini_infer.runner.base import PlanError
from llm_mini_infer.sched.plan import SAMPLE, STATE_ZERO, PlanBuilder

from ..support.penalty_cases import REQS, check_oracle, check_words, run, seq
from ..support.tiny_gemma import make_tiny_assistant, make_tiny_gemma
from ..support.tiny_qwen import make_tiny_model

PENALISED = [q for q in REQS if q.get("rep") or q.get("pres") or q.get("freq") or q.get("bias")]


@pytest.fixture(scope="module")
def qwen(tmp_path_factory):
    return make_tiny_model(tmp_path_factory.mktemp("pq"), seed=3, mtp=True)


@pytest.fixture(scope="module")
def gemma(tmp_path_factory):
    d = tmp_path_factory.mktemp("pg")
    target = make_tiny_gemma(d / "target", window=8, seed=5)
    return target, make_tiny_assistant(d / "assistant", target, seed=6)


def model_of(kind: str, qwen, gemma):
    return qwen if kind == "qwen" else gemma


@pytest.mark.parametrize(("kind", "k"), [("qwen", 0), ("qwen", 3), ("gemma", 0), ("gemma", 2)])
def test_tokens_are_vllms(kind: str, k: int, qwen, gemma) -> None:
    """Every request's tokens against the oracle, the forced prefix's tokens counted as output, and its slot's words
    at the end; the requests without terms are those of a runner without the setting."""
    model = model_of(kind, qwen, gemma)
    out, logits, sched = run(kind, model, REQS, k=k, max_num_seqs=8)
    assert check_oracle(REQS, out, logits) > 60
    assert check_words(REQS, out, sched.final_words) == 5  # each slot's words: its prompt's bits and tokens' counts
    plain = [q for q in REQS if q not in PENALISED]
    off, _, _ = run(kind, model, plain, k=k, penalties=False, hook=False)
    assert {q["rid"]: out[q["rid"]] for q in plain} == off


@pytest.mark.parametrize("kind", ["qwen", "gemma"])
def test_alone_as_in_a_batch(kind: str, qwen, gemma) -> None:
    """A penalised or biased request's tokens alone are those it gets among the others (seeded ones included)."""
    model = model_of(kind, qwen, gemma)
    out, _, _ = run(kind, model, REQS, hook=False)
    for q in PENALISED[:3]:
        alone, _, _ = run(kind, model, [q], hook=False)
        assert alone[q["rid"]] == out[q["rid"]], q["rid"]


@pytest.mark.parametrize(("kind", "k"), [("qwen", 3), ("gemma", 2)])
def test_mtp_as_without(kind: str, k: int, qwen, gemma) -> None:
    """Under MTP the target's samples are those without it: a verify row counts the drafts before it, and only the
    accepted drafts and the bonus commit (greedy and seeded)."""
    model = model_of(kind, qwen, gemma)
    on, _, _ = run(kind, model, REQS, k=k, hook=False)
    off, _, _ = run(kind, model, REQS, hook=False)
    assert on == off


@pytest.mark.parametrize(("kind", "k"), [("qwen", 0), ("qwen", 3), ("gemma", 0)])
def test_preemption_keeps_the_counts(kind: str, k: int, qwen, gemma) -> None:
    """A pool too small for three growing penalised requests: the preempted one is readmitted within its prompt (its
    outputs recomputed and counted again) and continues as it would have alone."""
    model = model_of(kind, qwen, gemma)
    n = 40 if kind == "qwen" else 80  # Gemma's full pages hold more positions
    reqs = [dict(rid=i, prompt=seq(20, 10 + i), max_tokens=n, rep=1.5, pres=0.5, freq=0.3, bias={7 + i: 1.0},
                 temperature=0.0 if i % 2 else 0.9, seed=77 + i) for i in range(3)]
    ref = {q["rid"]: run(kind, model, [q], k=k, hook=False)[0][q["rid"]] for q in reqs}
    pages = (1 + 30) if kind == "qwen" else 26
    out, logits, sched = run(kind, model, reqs, k=k, num_pages=pages, max_num_seqs=3, ckpt_slots=4)
    assert sched.num_preemptions > 0
    assert out == ref
    assert check_oracle(reqs, out, logits) > 30


@pytest.mark.parametrize("kind", ["qwen", "gemma"])
def test_a_prefix_hit_counts_its_prompt(kind: str, qwen, gemma) -> None:
    """A request that resumes from an earlier one's cached prefix gets the tokens it gets with the cache off: the
    prefix's tokens, which the device never computes for it, come with its admission as prompt tokens (repetition)."""
    model = model_of(kind, qwen, gemma)
    shared = seq(130, 1)  # past Qwen's checkpoint grid (64): its hits land on state checkpoints
    first = dict(rid=0, prompt=[*shared, 1, 2], max_tokens=6)
    second = dict(rid=1, prompt=[*shared, 3, 4, 5], max_tokens=16, rep=3.0, bias={shared[0]: 1.0})
    on, logits, sched = run(kind, model, [first, second], sequential=True, ckpt_slots=4)
    assert sched.prefix_hits >= 64  # the second resumed from the first's prefix
    off, _, _ = run(kind, model, [first, second], sequential=True, caching=False, hook=False)
    assert on == off
    assert check_oracle([second], on, logits) > 10


def test_a_runner_without_the_setting_refuses_them(qwen) -> None:
    """The scheduler without penalties rejects such a request; a runner without them refuses a plan that has them."""
    from ..support.tiny_qwen import cpu_runner, pools

    out, _, _ = run("qwen", qwen, [dict(rid=0, prompt=seq(8), rep=1.2, max_tokens=3)], penalties=False, hook=False)
    assert out[0] == ([], out[0][1]) and out[0][1] != 0  # failed at once, no token
    pc = pools(budget=64)
    r = cpu_runner(qwen, pc)
    b = PlanBuilder(pc.max_num_seqs, pc.max_num_batched_tokens, pc.page_size, pc.max_model_len, vocab=512)
    b.begin()
    b.add_admit(1, 1, 0, 0.0, 1.0, 0, 0.0, 5, rep=1.2, prompt_len=4)
    b.add_pages(1, 0, [1, 2])
    b.add_explicit_row(1, np.arange(4, dtype=np.int32), 0, STATE_ZERO | SAMPLE)
    with pytest.raises(PlanError, match=r"server\.penalties"):
        r.launch_forward(b.finish(1))
    r.shutdown()
