"""The grammar-aware heads in the Qwen runner on CPU (`CudaRunner`; architecture.md, "Grammar-aware heads";
`runner.cuda.heads`): the sparse target head (a step whose every sample token is masked scores the union of its masks,
the sampler reading them through the ids) and the universe draft head (the MTP layer's drafts scored over the
cumulative union of the masks instead of its draft head, FR-Spec's ranked one or the whole), through the real scheduler
and the engine loop, on the torch reference kernels in fp32 (`qwen_mtp`'s tiny model and harness).

Test by test as tests/models/test_gemma4_heads.py, its Gemma 4 twin: the heads change no output (greedy and seeded,
constrained and unconstrained requests in the same steps, MTP on and off), the dense head takes every step it must (a
row without a mask, a row whose grammar ended under ignore_eos, a union over the rows), drafts over the universe raise
acceptance where the whole head, or FR-Spec's ranked head without the grammar's tokens, drafts outside the grammar,
and the counters say which head each step took. Then the Qwen runner's own: prefix-cache hits, decode checkpoints and
preemption with the heads on; a tied head; a top_k over the heads' widths; an empty mask; the warm-up's internal steps,
which sample on every variant and move nothing; always split; the factory's switches."""

from __future__ import annotations

import json
import logging
import shutil

import numpy as np
import pytest
import torch

from llm_mini_infer import config
from llm_mini_infer.engine.messages import FinishReason
from llm_mini_infer.grammar.engine import Grammars
from llm_mini_infer.runner.base import MASK_EMPTY
from llm_mini_infer.runner.cuda.heads import NO_HEADS, VARIANT
from llm_mini_infer.runner.cuda.runner import CudaRunner
from llm_mini_infer.sched.plan import MASKED, SAMPLE, STATE_ZERO, PlanBuilder

from ..support.fake_grammar import FakeGrammar, spec
from ..support.gemma_heads import json_like
from ..support.mtp_common import EOS, seq
from ..support.qwen_mtp import Run
from ..support.qwen_mtp import model as model  # the fixture: a tiny Qwen with its MTP layer
from ..support.tiny_qwen import cpu_runner, make_tiny_model, pools

VOCAB = 512
ROWS = VOCAB  # the default 1,024 rows, at most the vocabulary


@pytest.fixture(scope="module")
def tied(tmp_path_factory):
    return make_tiny_model(tmp_path_factory.mktemp("tied"), seed=2, tie=True, mtp=True)


@pytest.fixture(scope="module")
def ranked(model, tmp_path_factory):
    """The tiny model with 128 special tokens from id 300 on (tokenizer_config.json's added_tokens_decoder): FR-Spec's
    ranked head of 128 rows (`draft_vocab`) is those tokens until it ranks again (every `draft_head.CHECK_EVERY`
    emitted tokens), so it holds none of a grammar's tokens below 300."""
    d = tmp_path_factory.mktemp("ranked") / "model"
    shutil.copytree(model, d)
    added = {str(i): {"content": f"<s{i}>", "special": True} for i in range(300, 428)}
    (d / "tokenizer_config.json").write_text(json.dumps({"added_tokens_decoder": added}))
    return d


@pytest.fixture
def sparse_calls(monkeypatch) -> dict:
    """Per runner, the logits' width of each `_sparse_logits` call (a step scored on the sparse head) after
    `configure`: its warm-up's internal steps, which sample on every heads variant, are not counted."""
    calls = {}
    logits, configure = CudaRunner._sparse_logits, CudaRunner.configure

    def spy(self, hidden, buf=None):
        out = logits(self, hidden, buf)
        calls.setdefault(self, []).append(out.shape[1])
        return out

    def configured(self, pools):
        configure(self, pools)
        calls[self] = []

    monkeypatch.setattr(CudaRunner, "_sparse_logits", spy)
    monkeypatch.setattr(CudaRunner, "configure", configured)
    return calls


def tight(i: int) -> str:
    return spec(30 + i, 20, 24, one_in=1, universe=6)  # every state allows tokens 0-5 (the EOS from 20 on)


def go(model, k: int, reqs: list[dict], *, sparse: bool, universe: bool = False, rows: int = 1024,
       temperature: float = 0.0, top_p: float = 1.0, top_k: int = 0, warm: list[dict] = (), **kw) -> Run:
    """`warm`, then `reqs` through the engine loop (`accepted0`: the drafts accepted before `reqs`), every constrained
    token of `reqs` one its grammar allowed; a request's own `top_k` over `top_k`; `kw`: `Run`'s, the runner's options
    among them."""
    grammar = FakeGrammar(VOCAB, (EOS,))
    r = Run(model, k, grammars=Grammars(grammar), sparse_head=sparse, sparse_head_rows=rows, draft_universe=universe,
            **kw)
    for batch in (warm, reqs):
        if not batch:
            continue
        r.accepted0 = r.sched.spec_accepted
        for q in batch:
            r.add(q["rid"], q["prompt"], max_tokens=q.get("max_tokens", 14), temperature=temperature, top_p=top_p,
                  top_k=q.get("top_k", top_k), ignore_eos=q.get("ignore_eos", False), grammar=q.get("grammar", ""))
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
def test_the_sparse_head_gives_the_dense_heads_outputs(model, k, params, sparse_calls) -> None:
    """The same outputs and finishes with the sparse head (and the universe draft head under MTP, over the whole draft
    head or FR-Spec's ranked one on the other steps) as with neither: the masked choice over the union's columns is the
    dense head's (fp32 on CPU: the same logits); unconstrained steps take the dense head; every split step is counted
    once, and each step counted sparse scored the sparse head's rows (the same logits on CPU: only the spy tells)."""
    temp, top_p, top_k = params
    kw = dict(temperature=temp, top_p=top_p, top_k=top_k)
    base = go(model, k, REQS, sparse=False, **kw)
    runs = [go(model, k, REQS, sparse=True, **kw)]
    if k:
        runs.append(go(model, k, REQS, sparse=True, universe=True, **kw))
        runs.append(go(model, k, REQS, sparse=True, universe=True, draft_vocab=128, **kw))
        assert runs[-1].r.draft_head.reduced
    for r in runs:
        assert (r.out, r.finish) == (base.out, base.finish)
        c = r.r.head_counters()
        assert c.sparse > 0 and c.dense_unconstrained > 0 and c.dense_off == 0 and c.dense_over_cap == 0, c
        assert sum(c.union_counts) == c.sparse  # every constrained step's union fit
        assert (c.universe_steps > 0) == r.r.gs.universe, c
        assert len(sparse_calls[r.r]) == c.sparse and set(sparse_calls[r.r]) == {r.r.gs.rows}, (sparse_calls[r.r], c)
    assert base.finish[0] == FinishReason.STOP and base.out[0][-1] == EOS
    c = base.r.head_counters()
    assert c.sparse == 0 and c.dense_off > 0 and c.universe == 0, c
    assert sparse_calls[base.r] == []


def test_dead_rows_keep_steps_sparse(model) -> None:
    """Requests that finish at different steps leave rows in the steps already planned: their masks are singletons
    (`Scheduler.fill_masks`), not all ones, so every split step of constrained requests stays on the sparse head."""
    reqs = [dict(rid=i, prompt=seq(7 + 5 * i, i), grammar=json_like(10 + i, 2 + i, 4 + 2 * i), max_tokens=30)
            for i in range(4)]
    base = go(model, 3, reqs, sparse=False)
    r = go(model, 3, reqs, sparse=True, universe=True)
    assert (r.out, r.finish) == (base.out, base.finish)
    assert all(f == FinishReason.STOP for f in r.finish.values()), r.finish
    c = r.r.head_counters()
    assert c.sparse > 0 and c.dense_unconstrained == 0 and c.dense_over_cap == 0, c


def test_ignore_eos_past_the_grammar_takes_the_dense_head(model) -> None:
    """Under ignore_eos a request runs on past its grammar's EOS unconstrained (a finished grammar applies no mask):
    its masks are all ones, so those steps take the dense head, with the same outputs."""
    reqs = [dict(rid=0, prompt=seq(12, 0), grammar=json_like(4, 1, 3), ignore_eos=True, max_tokens=12),
            dict(rid=1, prompt=seq(15, 1), grammar=json_like(5, 4, 30), max_tokens=12)]
    base = go(model, 3, reqs, sparse=False)
    r = go(model, 3, reqs, sparse=True, universe=True)
    assert (r.out, r.finish) == (base.out, base.finish)
    assert len(r.out[0]) == 12 and EOS in r.out[0]
    c = r.r.head_counters()
    assert c.sparse > 0 and c.dense_unconstrained > 0, c


def test_a_union_over_the_rows_takes_the_dense_head(model) -> None:
    """With few rows the larger unions fall back to the dense head (`over_cap`), with the same outputs (24 rows: the
    unions here hold about 30 tokens)."""
    reqs = [dict(rid=i, prompt=seq(9 + 4 * i, i), grammar=json_like(20 + i, 30, 40), max_tokens=10) for i in range(4)]
    base = go(model, 3, reqs, sparse=False)
    r = go(model, 3, reqs, sparse=True, universe=True, rows=24)
    assert (r.out, r.finish) == (base.out, base.finish)
    c = r.r.head_counters()
    assert c.dense_over_cap > 0, c


WARM = [dict(rid=100 + i, prompt=seq(6 + i, 50 + i), grammar=tight(i), max_tokens=24) for i in range(4)]
TIGHT = [dict(rid=i, prompt=seq(8 + 3 * i, 70 + i), grammar=tight(10 + i), max_tokens=24) for i in range(4)]


def test_universe_drafts_raise_acceptance_and_change_no_output(model) -> None:
    """The tiny MTP layer is random: over the whole vocabulary its drafts almost all fall outside a grammar of 6
    tokens, where they can never be accepted; over the grammar's universe (grown from the first requests: those 6 and
    the EOS) about one in seven is. Drafts change acceptance, never outputs; the universe holds no token the grammars
    never allowed."""
    full = go(model, 3, TIGHT, sparse=True, warm=WARM)
    uni = go(model, 3, TIGHT, sparse=True, universe=True, warm=WARM)
    assert (uni.out, uni.finish) == (full.out, full.finish)
    a_full = full.sched.spec_accepted - full.accepted0
    a_uni = uni.sched.spec_accepted - uni.accepted0
    assert a_uni >= a_full and a_uni > 0, (a_uni, a_full)
    c = uni.r.head_counters()
    assert c.universe == 7 and c.universe_steps > 0 and c.universe_rebuilds >= 1, c  # the 6 tokens and the EOS
    assert uni.r.gs.uni_head.n == c.universe


def test_universe_drafts_where_the_ranked_head_has_none_of_the_grammars_tokens(ranked) -> None:
    """FR-Spec's ranked head (`draft_vocab` 128: the tokens 300-427 here, `ranked`) holds none of the grammar's tokens,
    so no draft it makes is ever accepted (a schema's tokens outside a 32,768-token head on Qwen3.8, until a ranking
    takes them in). On the steps the universe serves it replaces that head, not their intersection (empty here), and
    its drafts are accepted; the ranking keeps counting every step's tokens. The outputs are the whole head's."""
    whole = go(ranked, 3, TIGHT, sparse=True, warm=WARM)
    fr = go(ranked, 3, TIGHT, sparse=True, warm=WARM, draft_vocab=128)
    uni = go(ranked, 3, TIGHT, sparse=True, universe=True, warm=WARM, draft_vocab=128)
    for r in (fr, uni):
        assert (r.out, r.finish) == (whole.out, whole.finish)
        dh = r.r.draft_head
        assert dh.reduced and dh.ids.tolist() == list(range(300, 428)) and dh.rebuilds == 0
        assert int(dh.counts.sum()) > 0  # FR-Spec's ranking counted the emitted tokens, universe steps included
    assert fr.sched.spec_accepted == 0 and uni.sched.spec_accepted - uni.accepted0 > 0
    c = uni.r.head_counters()
    assert c.universe == 7 and uni.r.gs.uni_head.n == 7 and c.universe_steps > 0, c


def test_a_universe_over_the_rows_turns_off(model, caplog) -> None:
    """Grammars whose masks together exceed the rows switch the universe head off for good (logged once): the drafts
    go back to the MTP layer's draft head, the outputs stay."""
    reqs = [dict(rid=i, prompt=seq(9 + 4 * i, 90 + i), grammar=spec(50 + i, 30, 40, one_in=64, universe=256),
                 max_tokens=10) for i in range(4)]  # about 4 tokens a state: steps fit 96 rows, their sum does not
    base = go(model, 3, reqs, sparse=False)
    with caplog.at_level(logging.WARNING):
        r = go(model, 3, reqs, sparse=True, universe=True, rows=96)
    assert (r.out, r.finish) == (base.out, base.finish)
    c = r.r.head_counters()
    assert c.universe == 0 and r.r.gs.heads.uni_off, c
    assert sum("universe passed" in m for m in caplog.messages) == 1


# -- the Qwen runner's own paths -------------------------------------------------------------------------------------


def test_the_heads_with_cache_hits_and_decode_checkpoints(model) -> None:
    """Prefix-cache hits under MTP (FOLD on donations, the drafter's KV recomputed at a hit, end-of-chunk checkpoints
    copied after a split step's sampling), mid-chunk and decode checkpoints, constrained requests and not: with the
    heads on, the dense heads' outputs, hits and checkpoints."""
    base = seq(130, 11)
    first = [base + seq(5, 1), base + seq(9, 2)]
    grammar = json_like(5, 12, 28)
    runs = {}
    for heads in (False, True):
        runs[heads] = go(model, 3, [dict(rid=i, prompt=p, max_tokens=24, grammar=grammar)
                                    for i, p in enumerate(first)],
                         sparse=heads, universe=heads, caching=True, ckpt_slots=8, interval=16, budget=96)
    out = runs[False].out
    assert runs[True].out == out
    turns = [first[0] + out[0] + seq(4, 5), first[1] + out[1][:13] + seq(6, 6), base + seq(3, 9)]
    for r in runs.values():
        for i, p in enumerate(turns):  # the second unconstrained and short: the others' later steps are constrained
            r.add(10 + i, p, max_tokens=4 if i == 1 else 16, ignore_eos=False, grammar=grammar if i != 1 else "")
        r.run()
        assert r.sched.grammars.errors == 0
    dense, sparse = runs[False], runs[True]
    assert (sparse.out, sparse.finish, sparse.cached) == (dense.out, dense.finish, dense.cached)
    assert sparse.sched.ckpts_taken == dense.sched.ckpts_taken
    assert min(sparse.cached[10 + i] for i in range(3)) > 100
    assert sparse.sched.ckpts_taken["decode"] and sparse.sched.ckpts_taken["prefill"]
    c = sparse.r.head_counters()
    assert c.sparse > 0 and c.dense_unconstrained > 0 and c.universe_steps > 0, c


def test_the_heads_with_preemption(model) -> None:
    """A pool for about two of three constrained requests: preemption folds the donated state into the in-flight
    decode row, the readmission resumes from it in a prefill row, its grammar where it was; the heads give the dense
    heads' outputs."""
    reqs = [dict(rid=i, prompt=seq(30, 20 + i), grammar=json_like(40 + i, 30, 40), max_tokens=40) for i in range(3)]
    kw = dict(caching=True, num_pages=1 + 52, max_num_seqs=3, ckpt_slots=4, interval=64)
    base = go(model, 3, reqs, sparse=False, **kw)
    r = go(model, 3, reqs, sparse=True, universe=True, **kw)
    assert r.sched.num_preemptions > 0
    assert (r.out, r.finish) == (base.out, base.finish)
    c = r.r.head_counters()
    assert c.sparse > 0 and c.universe_steps > 0, c


@pytest.mark.parametrize("k", [0, 3])
def test_a_tied_head(tied, k, sparse_calls) -> None:
    """A tied Qwen (`tie_word_embeddings`): the sparse head and the universe gather the embedding's rows, with the
    dense head's outputs; each step counted sparse scored the sparse head's rows."""
    base = go(tied, k, REQS, sparse=False)
    r = go(tied, k, REQS, sparse=True, universe=k > 0)
    assert r.r.gs.head is r.r.model.embed_tokens.weight
    assert k == 0 or r.r.gs.uni_head.full is r.r.model.embed_tokens.weight
    assert (r.out, r.finish) == (base.out, base.finish)
    c = r.r.head_counters()
    assert c.sparse > 0 and c.dense_off == 0, c
    assert len(sparse_calls[r.r]) == c.sparse and set(sparse_calls[r.r]) == {r.r.gs.rows}, (sparse_calls[r.r], c)
    assert sparse_calls[base.r] == []


@pytest.mark.parametrize("k,rows,top_k", [(0, 24, 40), (3, 24, 40), (3, 96, 200)])
def test_a_top_k_over_the_heads_widths(model, k, rows, top_k) -> None:
    """Seeded requests with mixed top_k in a step, one over the sparse head's columns (and under MTP at 96 rows the
    universe's): each step samples with a plan made at its head's width (`plan_sampling`: k at most its columns), so
    the outputs are the dense heads'."""
    reqs = [dict(rid=i, prompt=seq(9 + 5 * i, i), grammar=spec(30 + i, 6, 12, one_in=16, universe=64),
                 top_k=(5, top_k)[i]) for i in range(2)]
    base = go(model, k, reqs, sparse=False, temperature=1.0)
    r = go(model, k, reqs, sparse=True, universe=k > 0, rows=rows, temperature=1.0)
    assert (r.out, r.finish) == (base.out, base.finish)
    c = r.r.head_counters()
    assert c.sparse > 0 and (rows == 24 or c.universe_steps > 0), c


@pytest.mark.parametrize("k", [0, 3])
def test_an_empty_mask_on_the_sparse_head(model, k) -> None:
    """Directly through the runner, as tests/structured/test_cuda_structured.py's empty mask: a MASKED final chunk whose
    mask allows nothing samples as on the dense head, MASK_EMPTY (its union is empty: every column padding; under MTP
    the drafts over the still empty universe are token 0, their statuses ignored); an all-ones mask is unconstrained, so
    it takes the dense head and samples as no mask does."""
    res = {}
    for heads in (False, True):
        pc = pools(max_num_seqs=2, budget=64, mtp_k=k, num_pages=64)
        r = cpu_runner(model, pc, sparse_head=heads, draft_universe=heads)
        for name, fill in (("none", None), ("all", -1), ("empty", 0)):
            r.reset()
            b = PlanBuilder(2, 64, pc.page_size, pc.max_model_len, k)
            b.begin()
            b.add_admit(1, 1, 0, 0.0, 1.0, 0, 0.0, seed=5, max_len=pc.max_model_len, stop=[])
            b.add_pages(1, 0, list(range(1, 9)))
            b.add_explicit_row(1, np.asarray(seq(12), np.int32), 0,
                               STATE_ZERO | SAMPLE | (MASKED if fill is not None else 0))
            h = r.launch_forward(b.finish(1))
            n = 0
            if fill is not None:
                r.mask_buffer(h)[:1] = fill
                n = 1
            r.launch_sample(h, n)
            out = r.wait(h)
            res[heads, name] = (out.tokens[0].tolist(), None if out.status is None else int(out.status[0]))
            if heads and k and name == "empty":
                assert out.drafts[0].tolist() == [0] * k
        if heads:
            c = r.head_counters()
            assert (c.sparse, c.dense_unconstrained, c.union_counts[0], c.universe) == (1, 1, 1, 0), c
            assert c.universe_steps == (1 if k else 0), c
    for name in ("none", "all", "empty"):
        assert res[True, name] == res[False, name], name
    assert res[False, "all"] == res[False, "none"] and res[False, "none"][1] is None
    assert res[False, "empty"][1] == MASK_EMPTY


@pytest.mark.parametrize("k", [0, 3])
def test_the_warm_up_samples_on_every_variant_and_moves_nothing(model, k, caplog) -> None:
    """configure's warm-up (internal steps, `_run`) samples a masked decode step on each heads variant (the dense head,
    the sparse head, and under MTP the sparse head with the universe), the sparse head on the first tokens' ids: no
    counter moves, and the universe stays empty and uninstalled. The runner reports the heads as Gemma 4's does: its
    stats and its startup line."""
    pc = pools(max_num_seqs=4, budget=128, mtp_k=k)
    with caplog.at_level(logging.INFO, logger="llm_mini_infer.runner.cuda.runner"):
        r = CudaRunner(model, device="cpu", dtype=torch.float32, max_num_seqs=pc.max_num_seqs,
                       max_num_batched_tokens=pc.max_num_batched_tokens, max_model_len=pc.max_model_len, check=True,
                       mtp_k=k, sparse_head=True, draft_universe=True)
    want = ["s", "q", "p"] if k else ["s", "q"]
    assert r.gs.variants == want
    assert (r.stats["sparse_head"], r.stats["draft_universe"], r.stats["sparse_head_rows"]) == (True, k > 0, ROWS)
    (line,) = [m for m in caplog.messages if m.startswith("grammar heads:")]
    assert line == (f"grammar heads: the sparse head on, the draft universe {'on' if k else 'off'}, at most {ROWS} "
                    "tokens (0.1 MiB of the target's head rows per sparse step)")
    seen = []
    part = r._part

    def spy(*a, heads=NO_HEADS, **kw):
        if a[5] == "smp":
            seen.append(VARIANT[heads])
        return part(*a, heads=heads, **kw)

    r._part = spy
    r.configure(pc)
    assert sorted(set(seen)) == sorted(want), seen
    c = r.head_counters()
    assert (c.sparse, c.dense_off, c.dense_unconstrained, c.dense_over_cap, sum(c.union_counts), c.union_sum) == \
        (0, 0, 0, 0, 0, 0), c
    assert (c.universe, c.universe_steps, c.universe_rebuilds) == (0, 0, 0), c
    if k:
        g = r.gs
        assert g.uni_head.n == 0 and bool((g.uni_head.ids == -1).all()) and g.heads.pending is None
        assert g.heads.uni_n == 0 and not g.heads.uni_off


@pytest.mark.parametrize("k", [0, 3])
def test_always_split_with_the_heads(model, k) -> None:
    """Always split (every step a forward and a sampling, masked or not) with the heads on: the fused path's outputs and
    head counters (a split step without masks counts in none of them), and the dense heads' outputs."""
    base = go(model, k, REQS, sparse=False)
    runs = {split: go(model, k, REQS, sparse=True, universe=k > 0, always_split=split) for split in (False, True)}
    for r in runs.values():
        assert (r.out, r.finish) == (base.out, base.finish)
    assert runs[True].r.head_counters() == runs[False].r.head_counters()
    assert runs[True].r.head_counters().sparse > 0


def test_the_factory_passes_the_switches(model, monkeypatch) -> None:
    """`make_cuda_runner` on a Qwen with its MTP layer and structured output: every switch config.py derives on for it
    reaches the runner's constructor (PDL among them, which a CPU runner reports off whatever it is given), and the
    runner has them; the prefill split is left to the runner's defaults."""
    from llm_mini_infer.kernels.attn512 import PREFILL_FROM
    from llm_mini_infer.kernels.attn_window import FULL_PREFILL_KEYS
    from llm_mini_infer.runner.cuda import make_cuda_runner

    seen, init = {}, CudaRunner.__init__

    def spy(self, *args, **kw):
        seen.update(kw)
        init(self, *args, **kw)

    monkeypatch.setattr(CudaRunner, "__init__", spy)
    cfg = config.derive(config.load(None, {"model.path": str(model), "runner.kind": "cuda",
                                           "dev.runner.cuda.device": "cpu", "structured_output.enabled": True}))
    o = cfg.dev.runner.cuda
    assert o.pdl and o.always_split and o.sparse_head and cfg.dev.speculative.draft_universe
    r = make_cuda_runner(cfg)
    assert type(r) is CudaRunner and r.k == 3
    switches = ("pdl", "always_split", "sparse_head", "sparse_head_rows", "draft_universe")
    assert {s: seen.get(s) for s in switches} == dict(pdl=True, always_split=True, sparse_head=True,
                                                       sparse_head_rows=1024, draft_universe=True)
    assert (r.stats["sparse_head"], r.stats["draft_universe"], r.stats["sparse_head_rows"]) == (True, True, ROWS)
    assert r.always_split is True and r.stats["always_split"] is True and r.stats["pdl"] is False
    assert seen.get("batch_invariant") is False  # a BF16 checkpoint: off (config.py)
    cfg = config.derive(config.load(None, {"model.path": str(model), "runner.kind": "cuda",
                                           "dev.runner.cuda.device": "cpu", "structured_output.enabled": True,
                                           "dev.runner.cuda.batch_invariant": True}))
    r = make_cuda_runner(cfg)
    assert seen["batch_invariant"] is True and r.inv and r.stats["batch_invariant"] is True
    assert r.capabilities().batch_invariant is True and r.gs.fixed_m
    assert "attn_prefill" not in seen and r.attn_prefill == (PREFILL_FROM, FULL_PREFILL_KEYS)
