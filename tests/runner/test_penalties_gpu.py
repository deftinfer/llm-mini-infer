"""Penalties and the logit bias on CUDA (`runner.cuda.penalties`, the sampler kernel's terms): both runners on CUDA
graphs (always split, the sparse head, MTP), under the suite's sync check at error. Requests without terms get the
tokens of a runner without the setting; the terms change the penalised ones; greedy penalised tokens are vLLM's (the
oracle of `support.penalty_cases` on the eager runner's logits, near ties aside); each penalised slot's words at the end
are its prompt's, its bias's and its tokens' counts. (A request's tokens alone and in a batch: tests/runner/
test_cuda_penalties.py on CPU, whose reference is exact; on CUDA the model path's invariance is
tests/runner/test_batch_invariance_gpu.py's, and the sampler's rows are independent, tests/kernels/test_kernels.py.)"""

from __future__ import annotations

import pytest

from ..support.gemma_gpu import W
from ..support.gemma_gpu import cuda_runner as gemma_cuda_runner
from ..support.gemma_gpu import pair as pair  # the fixture: the tiny Gemma pair in BF16
from ..support.penalty_cases import REQS, check_oracle, check_words, run
from ..support.qwen_gpu import tiny as qwen_tiny  # noqa: F401 - the fixture: the tiny Qwen in BF16
from ..support.tiny_qwen import cuda_runner as qwen_cuda_runner

pytestmark = pytest.mark.gpu(reason="needs a CUDA GPU and the kernel library")
PENALISED = [q for q in REQS if q.get("rep") or q.get("pres") or q.get("freq") or q.get("bias")]
PLAIN = [q for q in REQS if q not in PENALISED]
def setup(kind: str, qwen_tiny, pair, graphs: bool = True) -> tuple:  # noqa: F811
    if kind == "qwen":
        return qwen_tiny, (lambda m, pc, **rk: qwen_cuda_runner(m, pc, graphs, **rk)), {}
    return pair, (lambda t, pc, **rk: gemma_cuda_runner(t, pc, graphs, **rk)), dict(window=W, wps=16)


@pytest.mark.parametrize("kind", ["qwen", "gemma"])
@pytest.mark.parametrize("k", [0, 3])
@pytest.mark.parametrize("heads", ["dense", "sparse"])
def test_penalties_on_cuda_graphs(qwen_tiny, pair, kind, k, heads) -> None:  # noqa: F811
    model, make, kw = setup(kind, qwen_tiny, pair)
    kw.update(always_split=True, sparse_head=heads == "sparse", draft_universe=heads == "sparse" and k > 0,
              hook=False)
    on, _, _ = run(kind, model, REQS, k=k, make=make, **kw)
    off, _, _ = run(kind, model, PLAIN, k=k, penalties=False, make=make, **kw)
    assert {q["rid"]: on[q["rid"]] for q in PLAIN} == off
    neutral = [{**q, "rep": 1.0, "pres": 0.0, "freq": 0.0, "bias": None} for q in PENALISED]
    base, _, _ = run(kind, model, neutral, k=k, make=make, **kw)
    assert sum(on[q["rid"]] != base[q["rid"]] for q in PENALISED) >= len(PENALISED) // 2  # the terms act


@pytest.mark.parametrize("kind", ["qwen", "gemma"])
@pytest.mark.parametrize("k", [0, 3])
def test_greedy_penalised_tokens_are_vllms_on_cuda(qwen_tiny, pair, kind, k) -> None:  # noqa: F811
    """Eager (the logits hook): each greedy penalised token is the argmax of vLLM's terms on the logits there; a
    near tie (the hook's GEMM has other rows than the sampler's) is skipped."""
    model, make, kw = setup(kind, qwen_tiny, pair, graphs=False)
    greedy = [q for q in PENALISED if q.get("temperature", 0.0) == 0.0]
    out, logits, _ = run(kind, model, greedy, k=k, make=make, **kw)
    assert check_oracle(greedy, out, logits, tie=0.05) >= 25


@pytest.mark.parametrize("kind", ["qwen", "gemma"])
@pytest.mark.parametrize("k", [0, 3])
def test_the_counts_are_the_tokens_on_cuda_graphs(qwen_tiny, pair, kind, k) -> None:  # noqa: F811
    """On CUDA graphs (always split, the sparse head), with and without MTP: at the end each penalised request's slot
    holds its prompt's bits, its bias ids' and the counts of the tokens it was given (the forced prefix's included):
    the sampler's commits, the verify windows' (accepted drafts and the bonus) and the explicit rows' add up."""
    model, make, kw = setup(kind, qwen_tiny, pair)
    kw.update(always_split=True, sparse_head=True, draft_universe=k > 0, hook=False, max_num_seqs=8)
    out, _, sched = run(kind, model, REQS, k=k, make=make, **kw)
    assert check_words(REQS, out, sched.final_words) == 5
