"""The grammar-aware heads in GemmaRunner on CPU (architecture.md, "Grammar-aware heads"; `runner.cuda.heads`): the
sparse target head (a step whose every sample token is masked scores the union of its masks, the sampler reading them
through the ids) and the universe draft head (the assistant scores the cumulative union of the masks), through the
real scheduler in window mode and the engine loop, on the torch reference kernels in fp32 (`test_gemma4_mtp`'s tiny
pair).

The fake grammar's sparse mode (`fake_grammar`: about 1/one_in of `universe` tokens per state) stands for a JSON
grammar: few tokens per mask, a small universe. The promises under test: the heads change no output (greedy and
seeded, constrained and unconstrained requests in the same steps, MTP on and off), the dense head takes every step it
must (a row without a mask, a row whose grammar ended under ignore_eos, a union over the rows), drafts over the universe
raise acceptance where the whole head drafts outside the grammar, and the counters say which head each step took."""

from __future__ import annotations

import logging

import pytest

from llm_mini_infer.engine.messages import FinishReason
from llm_mini_infer.grammar.engine import Grammars

from ..support.fake_grammar import FakeGrammar, spec
from ..support.gemma_heads import VOCAB, json_like
from ..support.gemma_heads import pair as pair  # the fixture
from ..support.gemma_mtp import Run
from ..support.mtp_common import EOS, seq


def go(pair, k: int, reqs: list[dict], *, sparse: bool, universe: bool = False, rows: int = 1024,
       temperature: float = 0.0, top_p: float = 1.0, top_k: int = 0, warm: list[dict] = ()) -> Run:
    grammar = FakeGrammar(VOCAB, (EOS,))
    r = Run(pair, k, grammars=Grammars(grammar), caching=False, sparse_head=sparse, sparse_head_rows=rows,
            draft_universe=universe)
    r.grammar = grammar
    for batch in (warm, reqs):
        if not batch:
            continue
        r.accepted0 = r.sched.spec_accepted
        for q in batch:
            r.add(q["rid"], q["prompt"], max_tokens=q.get("max_tokens", 14), temperature=temperature, top_p=top_p,
                  top_k=top_k, ignore_eos=q.get("ignore_eos", False), grammar=q.get("grammar", ""))
        r.run()
    assert r.sched.grammars.errors == 0
    for q in reqs:  # every constrained token is one its grammar allowed where it was sampled
        if q.get("grammar") and not q.get("ignore_eos"):
            m = grammar.instance(grammar.compile(q["grammar"]))
            for t in r.out[q["rid"]]:
                assert m.allowed()[t] and (m.is_stopped() or m.consume_tokens([t])), q["rid"]
    return r


REQS = [dict(rid=0, prompt=seq(20, 0), grammar=json_like(1, 4, 9)),
        dict(rid=1, prompt=seq(33, 1), max_tokens=3),  # unconstrained: its steps take the dense head
        dict(rid=2, prompt=seq(9, 2), grammar=json_like(2, 10, 18), max_tokens=16),
        dict(rid=3, prompt=seq(50, 3), grammar=json_like(3, 0, 6))]


@pytest.mark.parametrize("params", [(0.0, 1.0, 0), (0.8, 0.9, 0), (1.0, 1.0, 5)], ids=["greedy", "top_p", "top_k"])
@pytest.mark.parametrize("k", [0, 3])
def test_the_sparse_head_gives_the_dense_heads_outputs(pair, k, params) -> None:
    """The same outputs and finishes with the sparse head (and the universe draft head under MTP) as with neither:
    the masked choice over the union's columns is the dense head's (fp32 on CPU: the same logits); unconstrained steps
    take the dense head; every split step is counted once."""
    temp, top_p, top_k = params
    kw = dict(temperature=temp, top_p=top_p, top_k=top_k)
    base = go(pair, k, REQS, sparse=False, **kw)
    runs = [go(pair, k, REQS, sparse=True, **kw)]
    if k:
        runs.append(go(pair, k, REQS, sparse=True, universe=True, **kw))
    for r in runs:
        assert (r.out, r.finish) == (base.out, base.finish)
        c = r.r.head_counters()
        assert c.sparse > 0 and c.dense_unconstrained > 0 and c.dense_off == 0 and c.dense_over_cap == 0, c
        assert sum(c.union_counts) == c.sparse  # every constrained step's union fit
    assert base.finish[0] == FinishReason.STOP and base.out[0][-1] == EOS
    c = base.r.head_counters()
    assert c.sparse == 0 and c.dense_off > 0 and c.universe == 0, c


def test_dead_rows_keep_steps_sparse(pair) -> None:
    """Requests that finish at different steps leave rows in the steps already planned: their masks are singletons
    (`Scheduler.fill_masks`), not all ones, so every split step of constrained requests stays on the sparse head (an
    all-ones row would make the union the whole vocabulary and send most steps to the dense head)."""
    reqs = [dict(rid=i, prompt=seq(7 + 5 * i, i), grammar=json_like(10 + i, 2 + i, 4 + 2 * i), max_tokens=30)
            for i in range(4)]
    base = go(pair, 3, reqs, sparse=False)
    r = go(pair, 3, reqs, sparse=True, universe=True)
    assert (r.out, r.finish) == (base.out, base.finish)
    assert all(f == FinishReason.STOP for f in r.finish.values()), r.finish
    c = r.r.head_counters()
    assert c.sparse > 0 and c.dense_unconstrained == 0 and c.dense_over_cap == 0, c


def test_ignore_eos_past_the_grammar_takes_the_dense_head(pair) -> None:
    """Under ignore_eos a request runs on past its grammar's EOS unconstrained (a finished grammar applies no mask):
    its masks are all ones, so those steps take the dense head, with the same outputs."""
    reqs = [dict(rid=0, prompt=seq(12, 0), grammar=json_like(4, 1, 3), ignore_eos=True, max_tokens=12),
            dict(rid=1, prompt=seq(15, 1), grammar=json_like(5, 4, 30), max_tokens=12)]
    base = go(pair, 3, reqs, sparse=False)
    r = go(pair, 3, reqs, sparse=True, universe=True)
    assert (r.out, r.finish) == (base.out, base.finish)
    assert len(r.out[0]) == 12 and EOS in r.out[0]
    c = r.r.head_counters()
    assert c.sparse > 0 and c.dense_unconstrained > 0, c


def test_a_union_over_the_rows_takes_the_dense_head(pair) -> None:
    """With few rows the larger unions fall back to the dense head (`over_cap`), with the same outputs."""
    reqs = [dict(rid=i, prompt=seq(9 + 4 * i, i), grammar=json_like(20 + i, 30, 40), max_tokens=10) for i in range(4)]
    base = go(pair, 3, reqs, sparse=False)
    r = go(pair, 3, reqs, sparse=True, universe=True, rows=40)
    assert (r.out, r.finish) == (base.out, base.finish)
    c = r.r.head_counters()
    assert c.dense_over_cap > 0, c


def test_universe_drafts_raise_acceptance_and_change_no_output(pair) -> None:
    """The tiny assistant is random: over the whole vocabulary its drafts almost all fall outside a grammar of 6 tokens,
    where they can never be accepted; over the grammar's universe (grown from the first requests: those 6 and the EOS)
    about one in seven is. Drafts change acceptance, never outputs; the universe holds no token the grammars never
    allowed."""
    def tight(i: int) -> str:
        return spec(30 + i, 20, 24, one_in=1, universe=6)  # every state allows tokens 0-5 (the EOS from 20 on)

    warm = [dict(rid=100 + i, prompt=seq(6 + i, 50 + i), grammar=tight(i), max_tokens=24) for i in range(4)]
    reqs = [dict(rid=i, prompt=seq(8 + 3 * i, 70 + i), grammar=tight(10 + i), max_tokens=24) for i in range(4)]
    full = go(pair, 3, reqs, sparse=True, warm=warm)
    uni = go(pair, 3, reqs, sparse=True, universe=True, warm=warm)
    assert (uni.out, uni.finish) == (full.out, full.finish)
    a_full = full.sched.spec_accepted - full.accepted0
    a_uni = uni.sched.spec_accepted - uni.accepted0
    assert a_uni >= a_full and a_uni > 0, (a_uni, a_full)
    c = uni.r.head_counters()
    assert c.universe == 7 and c.universe_steps > 0 and c.universe_rebuilds >= 1, c  # the 6 tokens and the EOS
    assert uni.r.gs.uni_head.n == c.universe


def test_a_universe_over_the_rows_turns_off(pair, caplog) -> None:
    """Grammars whose masks together exceed the rows switch the universe head off for good (logged once): the drafts
    go back to the whole head, the outputs stay."""
    reqs = [dict(rid=i, prompt=seq(9 + 4 * i, 90 + i), grammar=spec(50 + i, 30, 40, one_in=64, universe=256),
                 max_tokens=10) for i in range(4)]  # about 4 tokens a state: steps fit 96 rows, their sum does not
    base = go(pair, 3, reqs, sparse=False)
    with caplog.at_level(logging.WARNING):
        r = go(pair, 3, reqs, sparse=True, universe=True, rows=96)
    assert (r.out, r.finish) == (base.out, base.finish)
    c = r.r.head_counters()
    assert c.universe == 0 and r.r.gs.heads.uni_off, c
    assert sum("universe passed" in m for m in caplog.messages) == 1


def test_the_factory_passes_a_set_attention_kind(pair) -> None:
    """dev.runner.cuda.attention flashinfer reaches Gemma's runner, which refuses it where it cannot run (here: no
    CUDA device), instead of running torch attention for it."""
    from llm_mini_infer import config
    from llm_mini_infer.runner.cuda import make_cuda_runner

    cfg = config.derive(config.load(None, {"model.path": str(pair[0]), "runner.kind": "cuda",
                                           "dev.runner.cuda.device": "cpu", "dev.runner.cuda.attention": "flashinfer"}))
    with pytest.raises(ValueError, match=r"^dev\.runner\.cuda\.attention flashinfer needs a CUDA device$"):
        make_cuda_runner(cfg)


def test_the_runner_takes_the_kernels_prefill_split(pair) -> None:
    """The runner's full layers' prefill split is the kernels' default (`attn512.PREFILL_FROM`, `PREFILL_KEYS`)."""
    from llm_mini_infer import config
    from llm_mini_infer.kernels.attn512 import PREFILL_FROM, PREFILL_KEYS
    from llm_mini_infer.runner.cuda import make_cuda_runner

    cfg = config.derive(config.load(None, {"model.path": str(pair[0]), "runner.kind": "cuda",
                                           "dev.runner.cuda.device": "cpu"}))
    r = make_cuda_runner(cfg)
    assert r.attn512_prefill == (PREFILL_FROM, PREFILL_KEYS)


def test_the_factory_passes_the_batch_invariance_switch(pair, monkeypatch) -> None:
    """`make_cuda_runner` on Gemma 4 with the switch derived on: the runner gets its parts and the switch itself, which
    it reports, so the scheduler's final chunks take their forced prefix whole (`Scheduler.forced_whole`); off, none.
    It leaves the prefill split to the runner's defaults."""
    from llm_mini_infer import config
    from llm_mini_infer.runner.cuda import make_cuda_runner
    from llm_mini_infer.runner.cuda.gemma import GemmaRunner
    from llm_mini_infer.sched.scheduler import Scheduler

    seen, init = {}, GemmaRunner.__init__

    def spy(self, *args, **kw):
        seen.update(kw)
        init(self, *args, **kw)

    monkeypatch.setattr(GemmaRunner, "__init__", spy)
    for on in (True, False):
        cfg = config.derive(config.load(None, {"model.path": str(pair[0]), "runner.kind": "cuda",
                                               "dev.runner.cuda.device": "cpu", "structured_output.enabled": True,
                                               **({} if on else {"dev.runner.cuda.batch_invariant": False})}))
        assert cfg.dev.runner.cuda.batch_invariant is on
        caps = make_cuda_runner(cfg).capabilities()
        assert seen["batch_invariant"] is on and bool(seen["window_split_keys"]) is on and caps.batch_invariant is on
        sched = Scheduler(max_num_seqs=4, max_num_batched_tokens=64, max_model_len=128, page_size=16, num_pages=64,
                          num_state_slots=5, eos_token_ids=(EOS,), window=caps.window, window_page_size=16,
                          grammars=Grammars(FakeGrammar(VOCAB, (EOS,))), forced_prefix=True,
                          batch_invariant=caps.batch_invariant)
        assert sched.forced_whole is on and not sched.aligned
    assert "attn512_prefill" not in seen  # the factory leaves it to the runner's defaults
