"""The forced prefix, the EOS-free finish and the verify-mask counters (architecture.md, "Grammar-aware heads"), from
the grammar backends through the scheduler, the validating fake runner and the frontend.

Both switches must change no output: with `forced_prefix` (a final prefill chunk takes the tokens its grammar forces
at its start as decided output tokens) and `eos_free_finish` (a request whose grammar stopped finishes with the EOS
appended) on, every request's tokens, completion count and finish reason are those of the sampled path, which the
harness predicts (`fake_generate` under the grammar's masks: a forced position's mask allows one token, a stopped
grammar's the EOS alone). What they change is which positions are sampled: a forced token or an appended EOS is never
the sample of a step, so a NaN or an empty mask injected there does not fail the request, while one at any position
still sampled does. The fake grammar's forced runs (`fake_grammar`, `lead` and `tail`) stand for compact JSON's opening
`{"` and first key and its closing brace; the runs go through Gemma 4's shape (window mode, a drafter without KV, MTP
up to k = 7), Qwen's (`gdn`: a hybrid GDN model, its state checkpoints CKPT records, its MTP layer's drafter with KV)
and attention-only mode, with every turn's accounting checked and each request's unreported tokens (`Req.unrep`)
followed through preemption, held samples, readmission and aborts (`Probe`). The verify-mask counters
are compared with an oracle that reads the masks, drafts and tokens of every verify row the fake runner ran
(`MaskOracle`). The catalog schema is checked on llguidance over Gemma 4's tokenizer with ids and counts only.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from hypothesis import given

from llm_mini_infer import config
from llm_mini_infer.cache.pools import pages_for_one_request
from llm_mini_infer.engine.messages import MASK_CLASSES, POPCOUNT_BUCKETS, AddRequest
from llm_mini_infer.grammar.engine import FORCED_MAX, Grammars, words
from llm_mini_infer.runner.fake import unpack_mask
from llm_mini_infer.sched.plan import FOLD_ALL, reserve_positions
from llm_mini_infer.sched.scheduler import Inflight, Req, Scheduler

from ..support.catalog_schema import PATH as CATALOG_SCHEMA
from ..support.config_cases import _gemma, _model
from ..support.fake_grammar import FakeGrammar, spec
from ..support.forced_cases import (
    ABORT,
    CHOICE,
    EOS_CLASS,
    ERROR,
    FORCED,
    FREE,
    LENGTH,
    MODES,
    STOP,
    MaskOracle,
    Probe,
    end_of,
    forced_of,
    make,
    predicted,
    row_at,
    run,
    run_switches_workload,
    structured_hybrid_count_workloads,
    structured_hybrid_workloads,
    structured_window_count_workloads,
    structured_window_workloads,
)
from ..support.fuzzing import campaign_settings, saved_on_failure
from ..support.harness import EOS, Harness, Spec, check_nan_invariants, key_of
from ..support.paths import MODELS_DIR
from ..support.sched_cases import VOCAB, check_plans, stem
from ..support.window_mtp import bound

COMBOS = [(False, False), (True, False), (False, True), (True, True)]  # (forced_prefix, eos_free_finish)
GEMMA = MODELS_DIR / "gemma-4-31b-it"
SYNTHETIC = CATALOG_SCHEMA  # tests/support/catalog_schema.py


# -- instrumentation ---------------------------------------------------------------------------------------------------


def run_combos(mode: str, specs_of: Any, k: int, failed_ok: bool = False, **kw) -> dict[tuple[bool, bool], Harness]:
    """The same workload under each combination of the switches (fresh specs each time), each run fully checked
    (`run`, `Probe`); requests that were not aborted get the same tokens and finish reason under all four. The probe of
    each run is its harness's `probe`."""
    out = {}
    for fp, ef in COMBOS:
        h = make(mode, specs_of(), k, forced_prefix=fp, eos_free_finish=ef, **kw)
        h.probe = Probe(h)
        run(h, failed_ok)
        h.probe.check_counters()
        s = h.sched
        if not fp:
            assert s.forced_requests == 0 and not h.probe.forced
        if not ef:
            assert s.eos_free.tolist() == [0, 0] and s.eos_free_gated == 0 and not h.probe.appended
        assert s.eos_free.sum() == len(h.probe.appended)
        out[(fp, ef)] = h
    base = out[(False, False)]
    for combo, h in out.items():
        for rid, r in h.results.items():
            if r.finish == ABORT or base.results[rid].finish == ABORT:
                continue
            b = base.results[rid]
            assert (r.tokens, r.finish) == (b.tokens, b.finish), (combo, rid, r.finish, b.finish)
    return out


# -- Grammars.forced_prefix on the fake grammar ------------------------------------------------------------------------


def forced_run(lead: int, tail: int, hi: int) -> int:
    """The fake grammar's forced prefix by its definition: positions 0, 1, ... while forced (n < lead, or one of the
    `tail` before `hi`) and the grammar has not stopped (n < hi), at most FORCED_MAX."""
    n = 0
    while n < min(hi, FORCED_MAX) and (n < lead or hi - tail <= n):
        n += 1
    return n


@pytest.mark.parametrize("sparse", [False, True])
def test_fake_forced_prefix_is_the_lead_run(sparse) -> None:
    """Every (lead, tail, lo, hi) of small grammars: the forced prefix is the run of forced positions from the start
    (the tail joins it when nothing lies between), each token the one its mask allows (never an EOS), and the next
    position, if the grammar goes on, is not a forced one; FORCED_MAX caps it."""
    g = Grammars(FakeGrammar(VOCAB, (EOS,)))
    for lead in range(5):
        for tail in range(4):
            for lo, hi in ((0, 0), (2, 2), (3, 3), (5, 5), (1, 6), (4, 12), (0, 2), (9, 9), (40, 60)):
                if tail > hi:
                    continue
                text = spec(lead * 31 + tail * 7 + hi, lo, hi, *((16, 300) if sparse else (0, 0)), lead=lead,
                            tail=tail)
                f = g.forced_prefix(text)
                assert len(f) == forced_run(lead, tail, hi), (lead, tail, lo, hi, f)
                m = g.matcher(g.compile(text))
                for t in f:
                    assert np.flatnonzero(m.allowed()).tolist() == [t] and t != EOS and g.only(m, t)
                    assert m.consume_tokens([t])
                if len(f) < FORCED_MAX and not m.is_stopped():
                    assert m.compute_ff_tokens() == [] and m.allowed().sum() > (0 if sparse else 1)
    assert len(g.forced_prefix(spec(1, 40, 60, lead=40))) == FORCED_MAX == 16


def test_a_fallback_singleton_is_not_forced() -> None:
    """A sparse grammar whose rule allows nothing falls back to one token per position (before `lo`, without the EOS):
    a mask of one token that is no fast-forward token, so the forced prefix stops at the lead run although the next
    masks are singletons too (llguidance forces only what its fast-forward returns)."""
    g = Grammars(FakeGrammar(VOCAB, (EOS,)))
    text = spec(3, 8, 8, one_in=1 << 62, universe=50, lead=1)
    f = g.forced_prefix(text)
    m = g.matcher(g.compile(text))
    assert len(f) == 1 and m.consume_tokens(list(f))
    (t,) = np.flatnonzero(m.allowed()).tolist()
    assert g.only(m, t) and m.compute_ff_tokens() == []


def test_forced_prefixes_are_cached_by_grammar_text() -> None:
    """Computed once per grammar text (one mask a token), then served from an LRU of the grammar cache's size."""
    g = Grammars(FakeGrammar(VOCAB, (EOS,)), cache_size=2)
    calls = Counter()
    only = g.backend.only

    def counted(m: Any, t: int) -> bool:
        calls["only"] += 1
        return only(m, t)

    g.backend.only = counted
    a, b, c = (spec(i, 9, 9, lead=2 + i) for i in range(3))
    first = g.forced_prefix(a)
    assert len(first) == 2 and calls["only"] == 2
    assert g.forced_prefix(a) is first and calls["only"] == 2
    assert len(g.forced_prefix(b)) == 3 and len(g.forced_prefix(c)) == 4 and calls["only"] == 9
    assert g.forced_prefix(a) == first and calls["only"] == 11  # evicted by b and c: computed again
    assert g.forced_prefix(c) is g.forced_prefix(c) and calls["only"] == 11


class _Scripted:
    """A matcher whose fast-forward tokens, mask and acceptance at each of its states are scripted:
    `script[n]` = (fast-forward tokens, the mask allows its first alone, the matcher takes it)."""

    def __init__(self, script: list[tuple[list[int], bool, bool]]):
        self.script, self.n = script, 0

    def compute_ff_tokens(self) -> list[int]:
        return list(self.script[self.n][0]) if self.n < len(self.script) else []

    def consume_tokens(self, tokens: list[int]) -> bool:
        assert len(tokens) == 1, "one forced token at a time"
        ok = self.script[self.n][2]
        self.n += ok
        return ok


class _ScriptedBackend:
    name, vocab_size, eos = "scripted", 100, (99,)

    def __init__(self, script: list[tuple[list[int], bool, bool]]):
        self.script = script

    def compile(self, text: str) -> str:
        return text

    def instance(self, compiled: str) -> _Scripted:
        return _Scripted(self.script)

    def only(self, m: _Scripted, token: int) -> bool:
        return m.script[m.n][1]


@pytest.mark.parametrize("script,want", [
    ([([5, 6], True, True), ([6], True, True), ([99], True, True)], (5, 6)),  # an EOS is never forced
    ([([5], True, True), ([7, 8], False, True)], (5,)),  # a fast-forward token whose mask allows others
    ([([5], True, True), ([7], True, False)], (5,)),  # one the matcher refuses
    ([([5, 6, 7], True, True)], (5,)),  # one token per fast-forward call: the list is never taken whole
    ([([5], True, True)] * 40, (5,) * FORCED_MAX),
])
def test_forced_prefix_stops(script, want) -> None:
    """`Grammars.forced_prefix`'s stops on a scripted backend: the first fast-forward token of each state, while its
    mask allows it alone, it is no EOS and the matcher takes it, at most FORCED_MAX."""
    assert Grammars(_ScriptedBackend(script)).forced_prefix("x") == want


def test_fake_only_agrees_with_the_masks() -> None:
    """`FakeGrammar.only` is the mask's singleton test, along grammars with forced runs (and out-of-range ids)."""
    g = FakeGrammar(VOCAB, (EOS,))
    rng = np.random.default_rng(5)
    for text in (spec(1, 6, 6, lead=2, tail=2), spec(2, 3, 12, lead=1, tail=1), spec(3, 8, 8, 1 << 62, 40, lead=3)):
        m = g.instance(g.compile(text))
        while True:
            a = m.allowed()
            for t in [*np.flatnonzero(a)[:3].tolist(), *rng.integers(0, VOCAB, 4).tolist(), EOS, -1, VOCAB]:
                assert g.only(m, t) == (0 <= t < VOCAB and bool(a[t]) and int(a.sum()) == 1), (text, m.n, t)
            if m.is_stopped():
                break
            a[EOS] = False
            nz = np.flatnonzero(a)
            assert m.consume_tokens([int(nz[m.n % len(nz)])])


# -- llguidance over Gemma 4's tokenizer (backend guidance, compact JSON, the catalog schema) --------------------------


@pytest.fixture(scope="module")
def gemma():
    """llguidance over Gemma 4's tokenizer, its `<eos>` the grammars' EOS and the vocabulary its config's, and the
    synthetic schema (tests/support/catalog_schema.py)."""
    if not (GEMMA / "tokenizer.json").exists() or not SYNTHETIC.exists():
        pytest.skip("tests/.fixtures/models/gemma-4-31b-it or the synthetic schema is absent: "
                    "uv run tests/scripts/fetch_fixtures.py")
    from llm_mini_infer.grammar.engine import LLGuidance

    vocab = int(json.loads((GEMMA / "config.json").read_text())["text_config"]["vocab_size"])
    g = Grammars(LLGuidance(GEMMA / "tokenizer.json", vocab, "<eos>"))
    assert g.eos == {1}
    return g, vocab, json.loads(SYNTHETIC.read_text())


def mask_ids(g: Grammars, m: Any, vocab: int) -> np.ndarray:
    """The ids the backend's mask allows at matcher `m`'s state."""
    buf = np.zeros((1, words(vocab)), np.int32)
    g.fill([(m, 0)], [], buf, 1)
    return np.flatnonzero(unpack_mask(buf[0], vocab))


def test_guidance_compact_json_forces_two_tokens(gemma) -> None:
    """The catalog schema under guidance with compact JSON: its first two masks allow one token each (`{"` then the
    first key's token), the third several (llguidance cannot force the `":` split: Gemma has `:-` tokens, and the first
    property is an integer without a minimum). The forced prefix is those two, each the token its mask allows, and is
    computed once per grammar text."""
    from llm_mini_infer.grammar.spec import GrammarSpecs

    g, vocab, schema = gemma
    text = GrammarSpecs(backend="guidance", disable_any_whitespace=True).json_schema(schema)
    f = g.forced_prefix(text)
    assert len(f) == 2 and f[0] == 14937 and g.forced_prefix(text) is f
    m = g.matcher(g.compile(text))
    for t in f:
        assert mask_ids(g, m, vocab).tolist() == [t] and g.only(m, t) and t not in g.eos
        assert m.consume_tokens([t])
    third = mask_ids(g, m, vocab)
    assert len(third) > 1 and not any(g.only(m, int(t)) for t in third)


def test_flexible_whitespace_and_the_auto_backend_force_nothing(gemma) -> None:
    """Where the first mask allows several tokens nothing is forced: guidance with any whitespace, the auto backend
    (xgrammar's language, no canonical forcing), and compact guidance's `json_object` (an object may start with a key
    or close at once)."""
    from llm_mini_infer.grammar.spec import GrammarSpecs, json_object_grammar

    g, vocab, schema = gemma
    for text in (GrammarSpecs(backend="guidance").json_schema(schema), GrammarSpecs().json_schema(schema),
                 json_object_grammar("guidance", True)):
        assert g.forced_prefix(text) == ()
        assert len(mask_ids(g, g.matcher(g.compile(text)), vocab)) > 1


def test_llguidance_only_agrees_with_the_masks(gemma) -> None:
    """`LLGuidance.only` (one mask into its own buffer, its nonzero words) is the mask's singleton test: along a walk of
    random allowed tokens through the catalog schema's grammar (guidance, compact JSON), for allowed, random, EOS and
    out-of-range ids; and at a complete document's end (an empty object), where the mask is the EOS alone."""
    from llm_mini_infer.grammar.spec import GrammarSpecs, json_object_grammar

    g, vocab, schema = gemma
    rng = np.random.default_rng(17)
    m = g.matcher(g.compile(GrammarSpecs(backend="guidance", disable_any_whitespace=True).json_schema(schema)))
    singles = 0
    for _ in range(40):
        ids = mask_ids(g, m, vocab)
        allowed = set(ids.tolist())
        for t in [*ids[:3].tolist(), *rng.choice(ids, 2).tolist(), *rng.integers(0, vocab, 3).tolist(), 1, -1, vocab]:
            assert g.only(m, t) == (len(ids) == 1 and t in allowed), (len(ids), t)
        singles += len(ids) == 1
        if m.is_stopped():
            break
        assert m.consume_tokens([int(rng.choice(ids))])
    assert singles >= 2
    m = g.matcher(g.compile(json_object_grammar("guidance", True)))
    assert m.consume_tokens(g.backend.tokenizer.tokenize_bytes(b"{}")) and m.is_stopped()
    assert mask_ids(g, m, vocab).tolist() == [1] and g.only(m, 1) and not g.only(m, 2) and not g.only(m, vocab + 1)


# -- the frontend's completion count -----------------------------------------------------------------------------------


def frontend_state(stream: bool, stops: list[str]) -> tuple[Any, list[Any]]:
    """A RequestState over a tiny word-level tokenizer (its pieces fused: " wor" + "ld" is " world"), the client
    reduced to what it calls: the ITL histogram, `on_finished`, `abort`."""
    from tokenizers import Tokenizer, decoders, models

    from llm_mini_infer.api.engine_client import RequestState
    from llm_mini_infer.api.sse import ChunkTemplate
    from llm_mini_infer.output.detokenizer import Detokenizer
    from llm_mini_infer.output.stop import OutputText, StopStrings

    vocab = {"<unk>": 0, "<eos>": 1, "Hel": 2, "lo": 3, " wor": 4, "ld": 5, "!": 6}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.decoder = decoders.Fuse()
    tok.add_special_tokens(["<eos>"])
    finished: list[Any] = []
    client = SimpleNamespace(metrics=SimpleNamespace(itl=SimpleNamespace(observe=lambda s: None)),
                             on_finished=finished.append, abort=lambda rids: finished.append(("abort", rids)))
    st = RequestState(client, 1, 3, Detokenizer(tok, [2, 3]), OutputText(StopStrings(stops) if stops else None),
                      ChunkTemplate("chatcmpl-1", 0, "m"), stream, True, False, [], 16, 0, 0, 0)
    return st, finished


def streamed(st: Any) -> tuple[str, int]:
    """A streamed response's content (its deltas joined) and its usage's completion_tokens."""
    text, usage = "", None
    for chunk in st.pending:
        if not chunk.startswith(b"data: {"):
            continue
        body = json.loads(chunk[len(b"data: "):])
        if body["choices"]:
            text += body["choices"][0]["delta"].get("content") or ""
        else:
            usage = body["usage"]["completion_tokens"]
    return text, usage


@pytest.mark.parametrize("stream", [False, True])
def test_the_frontend_counts_a_stop_strings_finish_by_its_steps(stream) -> None:
    """The frontend's count (every model's, Qwen's too): when a stop string completes at a token, the request finishes
    there and counts up to that token, whichever step the stop token (an EOS) behind it came in: [a, b, EOS] (STOP) and
    [a, b] then [EOS] count 2, as vLLM trims. Without stop strings the count is every token's, 3 both ways. The text
    (streamed or not), the finish and its reason are the same either way."""
    for stops, counts in ((["orld"], (2, 2)), ([], (3, 3))):
        seen = []
        for steps, n in zip(((([4, 5, 1], STOP),), (([4, 5], 0), ([1], STOP))), counts):
            st, finished = frontend_state(stream, stops)
            for toks, fin in steps:
                st.on_output(toks, fin, 1, 1)
            assert st.done and st.n_out == n, (stops, steps, st.n_out)
            text = streamed(st) if stream else (st.text.take(final=True), st.n_out)
            assert text[1] == n
            seen.append((text[0], st.finish_reason, st.stop_reason, [f for f in finished if isinstance(f, tuple)]))
        assert seen[0][:3] == seen[1][:3] == (" w" if stops else " world", "stop", "orld" if stops else None), seen


# -- configuration -----------------------------------------------------------------------------------------------------


def test_the_switches_derive(tmp_path) -> None:
    """Gemma 4: both on with structured output (the forced prefix under every backend, though auto forces nothing: its
    reason says so); the EOS-free finish needs a grammar that ends on one EOS id, so it is off under auto with the
    checkpoint's three stop ids (with its reason) and refused if set; with guidance (the tokenizer's eos_token) it is
    on. Off with structured output off. Qwen3.8 (its generation_config's two stop ids, `<|im_end|>` its tokenizer's
    eos_token) alike: both on with guidance; under auto the forced prefix on (forcing none), the EOS-free finish off
    (two EOS ids) and refused if set. Setting either off is taken."""
    (tmp_path / "q").mkdir()
    gemma, qwen = _gemma(tmp_path / "g"), _model(tmp_path / "q", 248_320, 1)

    def derive(model: Any, **kw: Any) -> config.Config:
        return config.derive(config.load(None, {"model.path": str(model), **kw}))

    so = {"structured_output.enabled": True}
    guidance = {"structured_output.backend": "guidance", "structured_output.disable_any_whitespace": True}
    on = "on: structured output, its grammars ending on one EOS id; a request whose grammar stopped finishes with " \
        "that EOS appended"
    cfg = derive(gemma, **so)  # one stop id (config.json's eos_token_id)
    s = cfg.dev.structured_output
    assert (s.forced_prefix, s.eos_free_finish) == (True, True) and cfg.eos_token_ids == [1]
    assert "backend auto forces none" in cfg.reasons["dev.structured_output.forced_prefix"]
    assert cfg.reasons["dev.structured_output.eos_free_finish"] == on
    for model in (gemma, qwen):
        cfg = derive(model, **{"structured_output.enabled": False})
        s = cfg.dev.structured_output
        assert (s.forced_prefix, s.eos_free_finish) == (False, False)
        why = cfg.reasons["dev.structured_output.forced_prefix"]
        assert why == cfg.reasons["dev.structured_output.eos_free_finish"] == "off: structured output is off"
    (gemma / "generation_config.json").write_text(json.dumps({"eos_token_id": [1, 50, 106]}))  # the checkpoint's
    (qwen / "generation_config.json").write_text(json.dumps({"eos_token_id": [248046, 248044]}))  # Qwen3.8's
    for model, ids in ((gemma, [1, 50, 106]), (qwen, [248044, 248046])):
        cfg = derive(model, **so)
        s = cfg.dev.structured_output
        assert (s.forced_prefix, s.eos_free_finish) == (True, False) and cfg.eos_token_ids == ids
        assert "backend auto forces none" in cfg.reasons["dev.structured_output.forced_prefix"]
        assert cfg.reasons["dev.structured_output.eos_free_finish"] == \
            f"off: backend auto's grammars end on {len(ids)} EOS ids: the sample decides which"
        with pytest.raises(ValueError, match=r"eos_free_finish: backend auto's grammars end on several EOS ids "
                                             rf"\({re.escape(str(ids))}\).*use backend guidance"):
            derive(model, **so, **{"dev.structured_output.eos_free_finish": True})
    (gemma / "tokenizer_config.json").write_text(json.dumps({"eos_token": "<eos>"}))
    (qwen / "tokenizer_config.json").write_text(json.dumps({"eos_token": "<|im_end|>"}))
    for model in (gemma, qwen):
        cfg = derive(model, **so, **guidance)
        s = cfg.dev.structured_output
        assert (s.forced_prefix, s.eos_free_finish) == (True, True)
        assert cfg.reasons["dev.structured_output.forced_prefix"] == \
            "on: structured output; a grammar's forced first tokens go into the final prefill chunk"
        assert cfg.reasons["dev.structured_output.eos_free_finish"] == on
        cfg = derive(model, **so, **{"dev.structured_output.forced_prefix": False,
                                     "dev.structured_output.eos_free_finish": False})
        assert (cfg.dev.structured_output.forced_prefix, cfg.dev.structured_output.eos_free_finish) == (False, False)
    assert derive(qwen, **so, **{"dev.structured_output.forced_prefix": True}).dev.structured_output.forced_prefix


def test_the_scheduler_appends_only_a_lone_grammar_eos_among_its_stop_ids() -> None:
    """The scheduler's own guards: either switch needs grammars (the forced prefix on every shape: GDN, window or
    attention-only); the EOS it appends is the grammars' one EOS id, and only if it stops a request (none with several
    grammar EOS ids: the sample would decide which)."""
    def sched(grammar_eos: tuple[int, ...], stop_ids: tuple[int, ...], **kw: Any) -> Scheduler:
        return Scheduler(max_num_seqs=2, max_num_batched_tokens=64, max_model_len=128, page_size=4, num_pages=64,
                         num_state_slots=3, eos_token_ids=stop_ids, grammars=Grammars(FakeGrammar(VOCAB, grammar_eos)),
                         forced_prefix=True, eos_free_finish=True, **kw)

    assert sched((999,), (999, 991))._grammar_eos == 999
    assert sched((999, 991), (999, 991))._grammar_eos == -1
    assert sched((991,), (999,))._grammar_eos == -1
    assert sched((999,), (999,), hybrid=False).forced_prefix and sched((999,), (999,), window=8).forced_prefix
    assert sched((999,), (999,)).forced_prefix  # GDN: its tokens follow the plan's CKPT records (`_commit_tails`)
    s = Scheduler(max_num_seqs=2, max_num_batched_tokens=64, max_model_len=128, page_size=4, num_pages=64,
                  num_state_slots=3, eos_token_ids=(999,), hybrid=False, forced_prefix=True, eos_free_finish=True)
    assert not s.forced_prefix and not s.eos_free_finish and s._grammar_eos == -1  # no grammars


# -- through the scheduler: the switches change no output --------------------------------------------------------------


def mixed() -> list[Spec]:
    """Constrained requests with forced runs among an unconstrained one that decodes past them all: lo == hi (a JSON
    document's end) and lo < hi; max_tokens of 1, 2, 3 and 4, which cut the forced prefix (the chunk's own sample stays
    within it); a stop id on the second forced token (the run is cut before it: that token is sampled and stops the
    request); ignore_eos past the grammar's end; a prompt of several chunks (its final chunk planned among the running
    requests); the first prompt again later (a cache hit at its end); grammars forced whole (the forced prefix ends
    the document: the chunk samples the EOS, the tail joining the lead run); an abort."""
    def g(seed: int, lo: int, hi: int, lead: int = 0, tail: int = 0) -> str:
        return spec(seed, lo, hi, lead=lead, tail=tail)

    stop = g(7, 9, 9, 3)
    return [Spec(rid=0, prompt_len=0, prompt=stem(1, 30), max_tokens=60, seed=0xF00, grammar=g(1, 10, 10, 2, 1)),
            Spec(rid=1, prompt_len=0, prompt=stem(2, 25), max_tokens=60, seed=0xF01, grammar=g(2, 4, 14, 1), arrive=1),
            Spec(rid=2, prompt_len=0, prompt=stem(3, 12), max_tokens=1, seed=0xF02, grammar=g(3, 6, 6, 3, 2), arrive=1),
            Spec(rid=3, prompt_len=0, prompt=stem(4, 13), max_tokens=2, seed=0xF03, grammar=g(4, 8, 8, 2), arrive=2),
            Spec(rid=4, prompt_len=0, prompt=stem(5, 14), max_tokens=3, seed=0xF04, grammar=g(5, 9, 9, 2, 1), arrive=2),
            Spec(rid=5, prompt_len=0, prompt=stem(6, 15), max_tokens=4, seed=0xF05, grammar=g(6, 7, 7, 3), arrive=3),
            Spec(rid=6, prompt_len=0, prompt=stem(7, 16), max_tokens=40, seed=0xF06, grammar=stop,
                 stop_ids=[forced_of(stop)[1]], arrive=3),
            Spec(rid=7, prompt_len=0, prompt=stem(8, 17), max_tokens=24, seed=0xF07, grammar=g(8, 5, 5, 3, 1),
                 ignore_eos=True, arrive=4),
            Spec(rid=8, prompt_len=0, prompt=stem(9, 18), max_tokens=40, seed=0xF08, ignore_eos=True),
            Spec(rid=9, prompt_len=0, prompt=stem(10, 110), max_tokens=20, seed=0xF09, grammar=g(10, 8, 8, 2, 2),
                 arrive=2),
            Spec(rid=10, prompt_len=0, prompt=stem(1, 30), max_tokens=40, seed=0xF0A, grammar=g(1, 10, 10, 2, 1),
                 arrive=12),
            Spec(rid=11, prompt_len=0, prompt=stem(12, 20), max_tokens=40, seed=0xF0B, grammar=g(12, 2, 2, 2),
                 arrive=5),
            Spec(rid=12, prompt_len=0, prompt=stem(13, 21), max_tokens=40, seed=0xF0C, grammar=g(13, 3, 3, 2, 2),
                 arrive=5),
            Spec(rid=13, prompt_len=0, prompt=stem(14, 22), max_tokens=40, seed=0xF0D, grammar=g(14, 12, 12, 1),
                 arrive=6, abort_at=8)]


@pytest.mark.parametrize("k", [0, 1, 3, 5, 7])
@pytest.mark.parametrize("mode", MODES)
def test_the_switches_change_no_output(mode, k) -> None:
    """`mixed` under the four combinations (`run_combos`): the same tokens and finish reasons, the predicted ones. With
    the forced prefix each request's final chunk takes min(its prefix cut before a stop id, max_tokens − 1, the
    positions `add` reserved past the prompt) tokens (the chunk cap leaves room here), on both planning paths; a
    stop id on a forced token ends the request there; a grammar forced whole ends on the EOS its chunk samples. (Where
    the scheduler takes no forced prefix, `Scheduler.forced_prefix` off, outputs are the same and nothing is forced.)"""
    out = run_combos(mode, mixed, k, max_num_seqs=4, accept_p=0.7, chunk_cap=24)
    reserve = reserve_positions(2, k)
    for (_, ef), h in out.items():
        assert h.results[10].cached >= 24, "the repeated prompt hit the cache"
        if ef:
            assert h.sched.eos_free.sum() > 0
        if not h.sched.forced_prefix:
            continue
        for s in mixed():
            if not s.grammar or s.abort_at is not None:
                continue
            f = forced_of(s.grammar)
            cut = next((j for j, t in enumerate(f) if t in s.stop_ids), len(f))
            want = min(cut, s.max_tokens - 1, reserve)
            assert h.probe.forced.get(s.rid, (0, ""))[0] == want, (s.rid, h.probe.forced.get(s.rid), want)
        assert {c for _, c in h.probe.forced.values()} == {"_admit_one", "_schedule_running_prefills"}
    r = out[(True, True)].results
    stop = forced_of(mixed()[6].grammar)
    assert (r[6].tokens, r[6].finish) == (list(stop[:2]), STOP)
    assert (r[11].tokens, r[11].finish) == ([*forced_of(mixed()[11].grammar), EOS], STOP)
    assert (r[12].tokens, r[12].finish) == ([*forced_of(mixed()[12].grammar), EOS], STOP)
    assert [len(r[i].tokens) for i in (2, 3, 4, 5)] == [1, 2, 3, 4] and r[7].tokens[:3] == list(forced_of(
        mixed()[7].grammar))


@pytest.mark.parametrize("k", [0, 3])
@pytest.mark.parametrize("mode", MODES)
def test_a_refused_forced_prefix_is_sampled(mode, k, monkeypatch, caplog) -> None:
    """A matcher that refuses its forced prefix (never expected: `_force`) is left as it was, and its chunk samples
    those tokens: the engine serves on, every request gets the tokens and finish of a run without the refusal (the
    predicted ones), nothing is forced for the refusing requests, and each refusal is logged. Spoiled prefixes stand
    for the refusal: request 0's (and request 10's: the same grammar) at its second token, where the matcher takes the
    first and is rolled back, and request 1's at its first."""
    spoil = {mixed()[0].grammar: 1, mixed()[1].grammar: 0}

    def spoiled(prefix: Any) -> Any:
        def forced_prefix(text: str) -> tuple[int, ...]:
            f, i = prefix(text), spoil.get(text)
            if i is None:
                return f
            t = (f[i] + 1) % VOCAB  # the position allows f[i] alone
            return (*f[:i], (t + 1) % VOCAB if t == EOS else t, *f[i + 1:])
        return forced_prefix

    out = {}
    for spoil_on in (False, True):
        h = make(mode, mixed(), k, forced_prefix=True, max_num_seqs=4, accept_p=0.7, chunk_cap=24)
        if spoil_on:
            monkeypatch.setattr(h.sched.grammars, "forced_prefix", spoiled(h.sched.grammars.forced_prefix))
        h.probe = Probe(h)
        with caplog.at_level(logging.ERROR, logger="llm_mini_infer.sched.scheduler"):
            run(h)
        h.probe.check_counters()
        out[spoil_on] = h
    clean, bad = out[False], out[True]
    assert not clean.probe.refused and bad.probe.refused == {0, 1, 10}
    assert set(clean.probe.forced) - set(bad.probe.forced) == {0, 1, 10}
    for rid, r in bad.results.items():
        b = clean.results[rid]
        if ABORT not in (r.finish, b.finish):
            assert (r.tokens, r.finish) == (b.tokens, b.finish), rid
    assert sum("refused its own forced prefix" in r.getMessage() for r in caplog.records) == 3


@pytest.mark.parametrize("k", [0, 3, 5])
@pytest.mark.parametrize("mode", MODES)
def test_the_plan_budgets_limit_the_forced_prefix(mode, k) -> None:
    """A final chunk takes forced tokens only within what the plan leaves: under a chunk cap of 24, a prompt of 48
    ends on a chunk of 24 (none taken: they are sampled), one of 47 on 23 (one), one of 46 on 22 (two); and under a
    budget that a chunk of 48 fills (max_num_batched_tokens), none. Outputs are the sampled path's either way."""
    def specs() -> list[Spec]:
        return [Spec(rid=i, prompt_len=0, prompt=stem(20 + i, n), max_tokens=20, seed=0xB0D + i, arrive=6 * i,
                     grammar=spec(30 + i, 6, 6, lead=3, tail=1)) for i, n in enumerate((48, 47, 46))]

    out = run_combos(mode, specs, k, max_num_seqs=1, chunk_cap=24)
    h = out[(True, True)]
    assert [h.probe.forced.get(i, (0,))[0] for i in range(3)] == ([0, 1, 2] if h.sched.forced_prefix else [0, 0, 0])
    out = run_combos(mode, lambda: specs()[:1], k, max_num_seqs=1, max_tokens=max(48, 4 * (1 + k)))
    assert not out[(True, False)].probe.forced


@pytest.mark.parametrize("k", [0, 3, 5])
@pytest.mark.parametrize("mode", ["window16", "gdn"])
def test_image_prompts_take_the_forced_prefix_after_their_spans(mode, k) -> None:
    """Requests with an image and a JSON grammar. Under Gemma 4's bidirectional spans (window mode), the final
    chunk holds an image's span whole and the text after it, then the forced tokens (`_image_inputs` over the chunk
    and them: text positions); a second request with the same image and text hits the prefix past the span (its chunk
    is the text after the hit, then the forced tokens); a third whose span a chunk cap pushes into a step of
    its own takes them on its last text chunk (planned among the running prefills). The same outputs under the four
    combinations; the fake checks every span is whole in its row. Under Qwen's causal M-RoPE spans (`gdn`), the forced
    tokens take the delta of the text after the last image (the request's `rope_off`, which the fake checks on the
    final row's last token and its samples depend on), and chunks may split a span."""
    from ..support.workloads import text

    def specs() -> list[Spec]:
        a = text(1, 6) + [0] * 16 + text(2, 5)
        c = text(3, 20) + [0] * 24 + text(4, 9)
        g = spec(7, 9, 9, lead=2, tail=1)
        return [Spec(rid=0, prompt_len=0, prompt=a, images=[(6, 8, 8, 3)], max_tokens=20, seed=0x1B0, grammar=g),
                Spec(rid=1, prompt_len=0, prompt=a, images=[(6, 8, 8, 3)], max_tokens=20, seed=0x1B1, grammar=g,
                     arrive=8),
                Spec(rid=2, prompt_len=0, prompt=c, images=[(20, 8, 12, 4)], max_tokens=20, seed=0x1B2,
                     grammar=spec(8, 6, 6, lead=3), arrive=1)]

    out = run_combos(mode, specs, k, max_num_seqs=2, bidirectional=mode != "gdn", encoder_rows=128,
                     encoder_budget=64, chunk_cap=30)
    h = out[(True, True)]
    assert h.results[1].cached > 6 + 16 and h.released == {0, 1, 2}  # a hit past the span: no image to encode
    three = min(3, reserve_positions(2, k))  # the positions `add` reserved past the prompt: 2 without MTP
    if mode != "gdn":
        assert h.probe.forced == {0: (2, "_admit_one"), 1: (2, "_admit_one"),
                                  2: (three, "_schedule_running_prefills")}
    elif h.sched.forced_prefix:  # the third's final chunk: 29 tokens under the cap of 30, room for one
        assert h.probe.forced == {0: (2, "_admit_one"), 1: (2, "_admit_one"), 2: (1, "_schedule_running_prefills")}
    else:
        assert not h.probe.forced


@pytest.mark.parametrize("k", [0, 1, 3, 5])
def test_hybrid_gdn_models_force(k) -> None:
    """Qwen's hybrid GDN model (the harness's default shape, checkpoints on a grid of 16): each final chunk takes its
    forced tokens after its prefill checkpoint, a CKPT record inside its row on the grid (`_commit_tails`, once the
    plan's records are placed); the EOS-free finish runs there too. Outputs are the sampled path's under the four
    combinations."""
    def specs() -> list[Spec]:
        return [Spec(rid=i, prompt_len=0, prompt=stem(40 + i, 20 + 9 * i), max_tokens=30, seed=0x6D0 + i, arrive=i,
                     grammar=spec(40 + i, 9, 9, lead=2, tail=1) if i < 3 else "", ignore_eos=i == 3)
                for i in range(4)]

    results = []
    for fp, ef in COMBOS:
        h = Harness(specs=specs(), vocab=VOCAB, mtp_k=k, accept_p=0.7, page_size=4, num_pages=300, checkpoint_slots=6,
                    grid=16, forced_prefix=fp, eos_free_finish=ef)
        assert h.sched.hybrid and not h.sched.window and h.sched.forced_prefix == fp
        h.probe = Probe(h)
        run(h)
        h.probe.check_counters()
        assert h.probe.forced == ({i: (2, "_admit_one") for i in range(3)} if fp else {})
        assert not h.sched.forced_declined and (h.sched.eos_free.sum() > 0) == ef
        for p in h.plans:  # each forced tail follows its row's prefill checkpoint
            for row, slot in enumerate(p["slots"]):
                nf = h.probe.tails.get((p["step"], slot), 0)
                if nf:
                    offs = [o for r, o in p["ckpt"] if r == row]
                    assert offs and all(0 < o < p["n_query"][row] - nf and o % 16 == 0 for o in offs), (p, offs)
        results.append({rid: (r.tokens, r.finish) for rid, r in h.results.items()})
    assert all(r == results[0] for r in results)


# -- the forced prefix and a checkpoint at the prompt's end (Qwen's shape) ---------------------------------------------


@pytest.mark.parametrize("k", [0, 3])
@pytest.mark.parametrize("n", [30, 32])
def test_hints_at_and_before_the_prompts_end(n, k) -> None:
    """Qwen's shape (grid 8), a constrained prompt ending on the grid (32) or off it (30). A hint at its end is dropped
    (`Req.hints` keeps those inside the prompt, as the renderer does: a request resumes at its last position at most),
    so its final row has no record at its n_query: the forced tokens follow its prefill checkpoint (24, inside the
    row), and a later request extending the prompt resumes there. A hint one position before the end ends the chunk
    there (an end-of-chunk checkpoint, off the grid), and the final chunk, the last prompt token, takes the forced
    tokens in the next plan; the later request resumes at the hint. The same under the four combinations."""
    for hint in (n, n - 1):
        def specs(hint: int = hint) -> list[Spec]:
            a = Spec(rid=0, prompt_len=0, prompt=stem(31, n), max_tokens=20, seed=0x4A0, hints=[hint],
                     grammar=spec(31, 8, 8, lead=2, tail=1))
            return [a, Spec(rid=1, prompt_len=0, prompt=[*a.prompt, 5, 6, 7], max_tokens=8, seed=0x4A1, arrive=24)]

        for (fp, _), h in run_combos("gdn", specs, k).items():
            nf = 2 if fp else 0
            if hint == n:
                assert row_at(h, 0) == (n + nf, [24]) and h.results[1].cached == 24
                assert h.probe.forced == ({0: (2, "_admit_one")} if fp else {})
            else:
                assert row_at(h, 0) == (n - 1, [n - 1]) and row_at(h, n - 1) == (1 + nf, [])
                assert h.results[1].cached == n - 1
                assert h.probe.forced == ({0: (2, "_schedule_running_prefills")} if fp else {})
            assert not h.sched.forced_declined


@pytest.mark.parametrize("k", [0, 3])
@pytest.mark.parametrize("n", [30, 32])
def test_a_branch_point_at_a_forcing_rows_prompt_end(n, k) -> None:
    """Qwen's shape (grid 8): a constrained request resumes at 24 (an earlier request's checkpoint) and its final
    chunk is the rest of its prompt; a request whose prompt strictly extends it, arriving with it, is admitted after
    it in the same plan and waits a step for a branch checkpoint that the writer's row takes at its prompt's end
    (`_branch_wait`: a record added to a row already planned, which is why `_commit_tails` runs after the admissions).
    On the grid (32, 8 tokens into the row) the writer's forced tokens follow it, the record inside the row; off it
    (30) the writer samples them (`forced_declined`). The waiting request resumes there under the four
    combinations."""
    def specs() -> list[Spec]:
        a = Spec(rid=0, prompt_len=0, prompt=stem(33, n), max_tokens=20, seed=0x4B0, arrive=20,
                 grammar=spec(33, 8, 8, lead=2, tail=1))
        return [Spec(rid=2, prompt_len=0, prompt=[*a.prompt[:24], 1, 2, 3], max_tokens=4, seed=0x4B2), a,
                Spec(rid=1, prompt_len=0, prompt=[*a.prompt, 5, 6, 7], max_tokens=8, seed=0x4B1, arrive=20)]

    for (fp, _), h in run_combos("gdn", specs, k).items():
        kept = fp and n % 8 == 0
        assert row_at(h, 24) == (n - 24 + (2 if kept else 0), [n - 24])
        assert h.sched.forced_declined == (fp and n % 8 != 0)
        assert h.probe.forced == ({0: (2, "_admit_one")} if kept else {})
        assert h.sched.ckpts_taken["branch"] >= 1 and (h.results[0].cached, h.results[1].cached) == (24, n)


def tree(h: Harness) -> set[tuple[Any, int, tuple[int, ...]]]:
    """Every checkpoint node of the harness's tree: (namespace, depth, the key up to it)."""
    out = set()
    for ns, root in h.sched.cache.roots.items():
        stack = [(c, ()) for c in root.children.values()]
        while stack:
            node, above = stack.pop()
            key = (*above, *node.key[:node.end - node.start].tolist())
            if node.ckpt:
                out.add((ns, node.end, key))
            stack.extend((c, key) for c in node.children.values())
    return out


@pytest.mark.parametrize("k", [0, 3])
def test_later_requests_hit_what_they_hit_without_the_forced_prefix(k) -> None:
    """The cache the forced prefix leaves on Qwen's shape is the sampled path's (an oracle for what later requests
    hit): requests one at a time (the EOS sampled, no decode checkpoint due), prompts that end on and off the grid,
    with hints inside them, one just before the end and one at it (dropped), of one chunk and of several, repeated and
    extended. At the end every checkpoint node is the same with the switch on and off, and so is what a later request
    would hit with each prompt, each prompt and part of its answer (its forced tokens, all of it but its EOS, all of
    it), and each prompt extended."""
    def specs() -> list[Spec]:
        def s(i: int, prompt: list[int], lead: int, tail: int = 1, hints: tuple[int, ...] = ()) -> Spec:
            return Spec(rid=i, prompt_len=0, prompt=prompt, max_tokens=20, seed=0x5C0 + i, arrive=30 * i,
                        hints=list(hints), grammar=spec(60 + i, 8, 8, lead=lead, tail=tail))

        return [s(0, stem(61, 30), 2, hints=(29,)), s(1, stem(62, 32), 3, hints=(32,)),
                s(2, stem(63, 45), 2, 2, hints=(20,)), s(3, stem(64, 70), 2), s(4, stem(62, 32), 1),
                s(5, [*stem(61, 30), 5, 6], 2), s(6, [*stem(61, 30), 5, 6, 7, 8], 2)]

    trees, hits = [], []
    for fp in (False, True):
        h = make("gdn", specs(), k, forced_prefix=fp, max_num_seqs=1, chunk_cap=24, accept_p=0.7)
        h.probe = Probe(h)
        h.run(check_every_turn=True)
        s = h.sched
        assert (s.forced_requests, s.forced_declined) == ((7, 0) if fp else (0, 0))
        keys = []
        for sp in h.specs.values():
            prompt, answer = key_of(sp, VOCAB).tolist(), h.results[sp.rid].tokens
            keys += [prompt + answer[:j] for j in (0, 1, len(forced_of(sp.grammar)), len(answer) - 1, len(answer))]
            keys.append([*prompt, 7, 8, 9])
        hits.append([s.cache.match(None, np.asarray(key, np.int64), s.step_id + 1, len(key) - 1).hit for key in keys])
        trees.append(tree(h))
        check_plans(h)
        check_nan_invariants(h)
        h.check()
    assert hits[0] == hits[1] and trees[0] == trees[1]
    assert {d for _, d, _ in trees[1]} >= {20, 29} and max(hits[1]) > 70  # the hints'; a finish's


# -- preemption, aborts and injections around the forced prefix --------------------------------------------------------


@pytest.mark.parametrize("k", [0, 3, 5])
@pytest.mark.parametrize("mode", ["window16", "gdn", "attention"])
@pytest.mark.parametrize("case", ["final chunk", "decode row", "mid-prefill", "held NaN"])
def test_preemption_around_the_forced_prefix(mode, k, case) -> None:
    """A constrained request is preempted, and a higher-priority one takes the only slot: while its final chunk, which
    took the forced prefix, is in flight (its sample is held: the frontend gets the forced tokens in front of it); while
    its first decode row is in flight behind that chunk's result (the forced tokens already reported); mid-prefill (the
    readmission's final chunk takes the prefix); and while its final chunk is in flight with a transient NaN in it (the
    held sample is dropped, the forced tokens stay with the request and go out with its readmission's first result).
    The prefix is taken once; outputs are the sampled path's under every combination."""
    def specs() -> list[Spec]:
        at = 2 if case in ("decode row", "mid-prefill") else 1
        low = Spec(rid=0, prompt_len=0, prompt=stem(13, 100 if case == "mid-prefill" else 40), max_tokens=14,
                   seed=51, priority=5, grammar=spec(13, 10, 10, lead=2, tail=1), preempt_at=at)
        if case == "held NaN":
            low.nonfinite_write, low.nonfinite_once = 30, True
        return [low, Spec(rid=1, prompt_len=0, prompt=stem(14, 12), max_tokens=4, seed=52, priority=0, arrive=at)]

    out = run_combos(mode, specs, k, max_num_seqs=1, policy="priority", chunk_cap=16 if case == "mid-prefill" else 0)
    for h in out.values():
        e = h.probe.events
        assert h.sched.num_preemptions == 1 and h.results[0].finish == STOP
        if not h.sched.forced_prefix:
            assert not any("unreported" in x for x in e)
            continue
        assert h.probe.forced[0] == (2, "_schedule_running_prefills" if case == "mid-prefill" else "_admit_one")
        want = {"final chunk": {"preempted with unreported tokens": 1, "preempted with its final chunk in flight": 1,
                                "held sample delivered after unreported tokens": 1},
                "decode row": {"preempted with a row in flight": 1},
                "mid-prefill": {},
                "held NaN": {"preempted with unreported tokens": 1, "preempted with its final chunk in flight": 1,
                             "held sample dropped, unreported tokens kept": 1, "readmitted with unreported tokens": 1}}
        assert dict(e) == want[case], dict(e)


@pytest.mark.parametrize("k", [3, 5])
@pytest.mark.parametrize("mode", ["window16", "gdn", "attention"])
def test_page_pressure_preempts_a_final_chunk_in_flight(mode, k) -> None:
    """FCFS in pools from one request's startup bound up: two long requests decode while a constrained one arrives
    at each of 16 turns; growing, they need pages the pool lacks, and the latest admitted, whose final chunk with its
    forced prefix may be in flight, is preempted (its sample held, then delivered after the forced tokens). Every run
    is checked; where the scheduler forces, at least one hits that case, and those are run under all four
    combinations."""
    full = bound(16, 4, 20, 64, 160, k) if mode.startswith("window") else \
        pages_for_one_request(160, 4, reserve_positions(2, k), True)
    hit = forcing = 0
    for extra in (1, 3):
        for arrive in range(16):
            def specs(arrive: int = arrive) -> list[Spec]:
                return [Spec(rid=0, prompt_len=0, prompt=stem(50, 70), max_tokens=80, seed=0x9A0, ignore_eos=True),
                        Spec(rid=1, prompt_len=0, prompt=stem(52, 60), max_tokens=80, seed=0x9A2, ignore_eos=True,
                             arrive=1),
                        Spec(rid=2, prompt_len=0, prompt=stem(51, 20), max_tokens=12, seed=0x9A1, arrive=arrive,
                             grammar=spec(51, 9, 9, lead=2, tail=1))]

            kw = dict(max_num_seqs=3, num_pages=full + extra, max_model_len=160)
            h = make(mode, specs(), k, forced_prefix=True, eos_free_finish=True, **kw)
            h.probe = Probe(h)
            run(h)
            forcing = h.sched.forced_prefix
            if h.probe.events["preempted with its final chunk in flight"]:
                hit += 1
                assert h.probe.events["held sample delivered after unreported tokens"]
                run_combos(mode, specs, k, **kw)
    assert hit or not forcing, "no preemption of a final chunk in flight"


@pytest.mark.parametrize("k", [0, 3])
@pytest.mark.parametrize("mode", ["window16", "gdn", "attention"])
def test_aborts_at_every_turn(mode, k) -> None:
    """A constrained request (its prefix forced, its closing token too) next to a decoding one is aborted at each turn
    of its life: before admission, with its final chunk and forced tokens in flight (they never reach the frontend),
    mid-decode, around its EOS-free finish. Each turn's accounting holds, the other request is unaffected, and
    whatever the aborted one received is a prefix of its prediction."""
    seen, forcing = Counter(), False
    for turn in range(14):
        def specs(turn: int = turn) -> list[Spec]:
            return [Spec(rid=0, prompt_len=0, prompt=stem(61, 30), max_tokens=40, seed=0xAB0, abort_at=turn,
                         grammar=spec(61, 9, 9, lead=2, tail=1)),
                    Spec(rid=1, prompt_len=0, prompt=stem(62, 18), max_tokens=30, seed=0xAB1, ignore_eos=True)]

        out = run_combos(mode, specs, k, max_num_seqs=2)
        for h in out.values():
            seen.update(h.probe.events)
            forcing |= h.sched.forced_prefix
            assert h.results[1].finish == LENGTH
    assert seen["aborted with unreported tokens"] >= (2 if forcing else 0)  # (the forced prefix's two combinations)


INJECTIONS = [  # (output index of the injection, its companion decodes too)
    (0, True), (1, True), (2, True), (9, True), (10, True), (10, False), (11, True)]


@pytest.mark.parametrize("k", [0, 3, 5])
@pytest.mark.parametrize("mode", ["window16", "gdn", "attention"])
@pytest.mark.parametrize("kind", ["nonfinite_after", "mask_empty_after"])
def test_injections_fail_a_request_only_where_it_samples(mode, k, kind) -> None:
    """A NaN or an empty mask at one output position of a grammar (lo == hi = 10, two forced tokens at its start, a
    forced closing token at 9, the EOS at 10). Positions 0 and 1 are sampled only without the forced prefix: with it
    the request runs to its end; 2 (the final chunk's own sample) and 9 (forced, but sampled: the prefix is the start
    only) always fail it. The EOS at 10 is appended by the EOS-free finish next to another decoding request (its row in
    flight is dropped with the NaN in it), so the request finishes; alone, the gate leaves the EOS to that row, which
    samples it and fails it, as does the switch off. Past the end (11), nothing fails. Each case under all four
    combinations; the companion always runs to its length."""
    for pos, companion in INJECTIONS:
        for fp, ef in COMBOS:
            specs = [Spec(rid=0, prompt_len=0, prompt=stem(81, 26), max_tokens=40, seed=0x1A00 + pos,
                          grammar=spec(81, 10, 10, lead=2, tail=1), **{kind: pos})]
            if companion:
                specs.append(Spec(rid=1, prompt_len=0, prompt=stem(82, 20), max_tokens=40, seed=0x2B00,
                                  ignore_eos=True))
            h = make(mode, specs, k, forced_prefix=fp, eos_free_finish=ef, max_num_seqs=2)
            h.probe = Probe(h)
            run(h, failed_ok=True)
            sampled = not (pos < 2 and h.sched.forced_prefix) and not (pos == 10 and ef and companion) and pos < 11
            r = h.results[0]
            assert (r.finish == ERROR) == sampled, (pos, companion, fp, ef, r.finish)
            if not sampled:
                assert r.finish == STOP and r.tokens == predicted(specs[0])[:11]
            if pos == 10:
                assert (0 in h.probe.appended) == (ef and companion)
            if companion:
                assert h.results[1].finish == LENGTH and h.results[1].tokens == predicted(specs[1])


# -- counters ----------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("k", [0, 3, 5])
@pytest.mark.parametrize("mode", ["window16", "gdn", "attention"])
def test_forced_counters_count_each_request_once(mode, k) -> None:
    """`forced_requests` and `forced_tokens`: two per request whose grammar forces two tokens at its start (compact
    JSON's `{"` and first key), counted when its final chunk is planned and once whatever follows (a preemption,
    readmissions behind higher-priority requests); none for a request whose max_tokens leaves no room (1), one whose
    stop id is its first forced token, or an unconstrained one."""
    def specs() -> list[Spec]:
        out = [Spec(rid=i, prompt_len=0, prompt=stem(90 + i, 15 + 4 * i), max_tokens=20, seed=0xC40 + i,
                    arrive=max(0, i - 1) * 2, priority=5 if i < 2 else 0,
                    grammar=spec(90 + i, 9, 9, lead=2, tail=i % 3)) for i in range(5)]
        g = spec(96, 7, 7, lead=2)
        return [*out, Spec(rid=5, prompt_len=0, prompt=stem(95, 12), max_tokens=1, seed=0xC45, grammar=g),
                Spec(rid=6, prompt_len=0, prompt=stem(96, 12), max_tokens=20, seed=0xC46, grammar=g,
                     stop_ids=[forced_of(g)[0]]),
                Spec(rid=7, prompt_len=0, prompt=stem(97, 12), max_tokens=20, seed=0xC47, priority=3, preempt_at=4)]

    out = run_combos(mode, specs, k, max_num_seqs=2, policy="priority")
    for h in out.values():
        assert (h.sched.forced_requests, h.sched.forced_tokens) == ((5, 10) if h.sched.forced_prefix else (0, 0))
    assert sum(h.sched.num_preemptions for h in out.values()) > 0
    r = out[(True, True)].results
    assert (r[6].tokens, r[6].finish) == ([forced_of(spec(96, 7, 7, lead=2))[0]], STOP) and len(r[5].tokens) == 1


def gated_specs(seed: int, tail: int, companion: bool, rid: int = 0, arrive: int = 0) -> list[Spec]:
    """A constrained request (lo == hi = 12, two forced tokens at its start, its closing token forced when `tail`),
    alone or next to an unconstrained one decoding past its end."""
    out = [Spec(rid=rid, prompt_len=0, prompt=stem(70 + seed, 20 + seed % 7), max_tokens=60, seed=0x7A70 + seed,
                arrive=arrive, grammar=spec(70 + seed, 12, 12, lead=2, tail=tail))]
    if companion:
        out.append(Spec(rid=rid + 1, prompt_len=0, prompt=stem(71, 18), max_tokens=60, seed=0x7B70, arrive=arrive,
                        ignore_eos=True))
    return out


def gated_runs(k: int, seed: int, tail: int, companion: bool, mode: str = "window16") -> tuple[Harness, Harness]:
    """`gated_specs` with the forced prefix, the EOS-free finish off and on (both checked fully)."""
    runs = []
    for ef in (False, True):
        h = make(mode, gated_specs(seed, tail, companion), k, forced_prefix=True, eos_free_finish=ef,
                 max_num_seqs=2, accept_p=0.7)
        h.probe = Probe(h)
        runs.append(run(h))
    return runs[0], runs[1]


@pytest.mark.parametrize("k", [0, 3, 5])
@pytest.mark.parametrize("mode", ["window16", "gdn"])
def test_the_gate_leaves_a_lone_request_its_eos_step(mode, k) -> None:
    """A lone request whose closing token is forced: the drafter never drafts it, so it comes as a verify row's
    correction or bonus (without MTP, as a row's one sample), and the request's next row is already in flight (its
    could-end walk stopped at the rejected draft). That row is the step's only one and its result is the EOS anyway:
    the gate leaves it (`eos_free_gated`), and the request finishes with the sampled EOS one step later, at the turn
    and after the steps it takes with the switch off. Next to another decoding request the step runs regardless: the
    request finishes at once with the EOS appended (`eos_free`, its row in flight, dropped), a turn earlier."""
    for seed in range(6):
        off, on = gated_runs(k, seed, 1, False, mode)
        s = on.sched
        assert (s.eos_free.tolist(), s.eos_free_gated) == ([0, 0], 1), seed
        assert on.probe.finished_at[0] == off.probe.finished_at[0] and len(on.plans) == len(off.plans)
        assert on.results[0].tokens == off.results[0].tokens and on.results[0].tokens[-1] == EOS
        off, on = gated_runs(k, seed, 1, True, mode)
        s = on.sched
        assert (s.eos_free.tolist(), s.eos_free_gated) == ([0, 1], 0) and on.probe.appended == {0}, seed
        assert on.probe.finished_at[0] == off.probe.finished_at[0] - 1 and len(on.plans) == len(off.plans)
        assert s.dropped_samples == off.sched.dropped_samples


@pytest.mark.parametrize("k", [3, 5])
@pytest.mark.parametrize("mode", ["window16", "gdn"])
def test_an_eos_free_finish_without_a_row_in_flight_saves_a_step(mode, k) -> None:
    """When nothing else runs, a plan skips a lone request's row while its row in flight could end its grammar (its
    drafts lead there: the could-end walk, `could_end`). If that row ends the document with its correction (a draft
    before the end rejected, the target's own closing token instead), no row of it is in flight: the EOS-free finish
    appends the EOS (`eos_free`, no row), where the switch off plans one more step for it and finishes two turns later.
    Otherwise the gate or a sampled EOS in the row decide, and nothing changes. Over 30 requests, both kinds occur."""
    saved = 0
    for seed in range(30):
        off, on = gated_runs(k, seed, 0, False, mode)
        s = on.sched
        assert on.results[0].tokens == off.results[0].tokens
        if s.eos_free[0]:
            saved += 1
            assert s.eos_free.tolist() == [1, 0] and s.eos_free_gated == 0
            assert len(on.plans) == len(off.plans) - 1 and on.probe.finished_at[0] == off.probe.finished_at[0] - 2
        else:
            assert len(on.plans) == len(off.plans) and on.probe.finished_at[0] == off.probe.finished_at[0]
    assert saved >= 2


# -- the EOS-free finish's donations on hybrid GDN models (Qwen's shape) -----------------------------------------------


def answered(s: Spec) -> list[int]:
    """What a request delivers on the sampled path: its prediction through its first stop."""
    out = predicted(s)
    return out[:end_of(s, out)]


def next_turn(a: Spec, out: list[int], eos: bool, rid: int, arrive: int) -> Spec:
    """A's next turn: its prompt, its answer (with its EOS, as a chat template holds a reply, or without it), then a
    message."""
    return Spec(rid=rid, prompt_len=0, prompt=[*a.prompt, *(out if eos else out[:-1]), 5, 6, 7], max_tokens=12,
                seed=0xF0 + rid, arrive=arrive)


def donor(**kw: Any) -> tuple[Spec, list[int], Spec]:
    """A constrained request whose closing token is forced (so never drafted: it comes as a correction or a bonus,
    the request's next row already in flight), its answer, and a companion decoding past it."""
    a = Spec(rid=0, prompt_len=0, prompt=stem(21, 30), max_tokens=40, seed=0xE0,
             grammar=spec(21, 9, 9, lead=2, tail=1), **kw)
    return a, answered(a), Spec(rid=3, prompt_len=0, prompt=stem(22, 18), max_tokens=60, seed=0xE3, ignore_eos=True)


@pytest.mark.parametrize("k", [0, 1, 3])
def test_a_hybrid_eos_free_finish_donates_its_state_where_its_answer_ends(k) -> None:
    """Qwen's shape: `donor` finishes EOS-free next to its companion, its row in flight dropped, and donates its live
    state at its closing token (the appended EOS never enters the key), folded one record into that row (FOLD 1 in
    the plan that FREEs its slot, before any reader's STATE_COPY; none without MTP, whose live state is already
    there). Its next turns, whose prompts hold the answer without its EOS or with it, resume exactly there (the fake
    checks the state's fingerprint and the outputs are predicted): where the sampled path's finish, which computed the
    EOS, lets the turn with the EOS resume one position deeper, and the turn without it lower (a prefill checkpoint).
    The same tokens either way."""
    runs = {}
    for ef in (False, True):
        a, out, companion = donor()
        h = make("gdn", [a, next_turn(a, out, False, 1, 30), next_turn(a, out, True, 2, 30), companion], k,
                 eos_free_finish=ef, max_num_seqs=4, accept_p=0.7)
        h.probe = Probe(h)
        runs[ef] = run(h)
    off, on = runs[False], runs[True]
    d = len(a.prompt) + len(out) - 1  # through the closing token
    s = on.sched
    assert (s.eos_free.tolist(), on.probe.in_flight, on.probe.donated) == ([0, 1], {0: True}, {0: d})
    assert (on.results[1].cached, on.results[2].cached) == (d, d)
    assert off.results[2].cached == d + 1 and off.results[1].cached < d
    slot = on.plans[0]["admit"][0]  # the donor's: admitted first
    free = next(p for p in on.plans if slot in p["free"])
    assert free["fold"] == ([(slot, 1)] if k else [])
    if not k:
        assert not any(p["fold"] for p in on.plans)
    assert {r: x.tokens for r, x in on.results.items()} == {r: x.tokens for r, x in off.results.items()}


def test_a_hybrid_eos_free_finish_without_a_row_in_flight_donates_its_committed_state() -> None:
    """Qwen's shape at k = 3: a lone constrained request whose last verify row ends its document with its correction
    has no row in flight (the end rule: `test_an_eos_free_finish_without_a_row_in_flight_saves_a_step`); it donates its
    live state at its last committed position, the closing token still pending (FOLD_ALL of the last step's commit, in
    the next plan: its next turn's admission), and that turn resumes there, one position below the closing token."""
    seen = 0
    for seed in range(30):
        a = gated_specs(seed, 0, False)[0]
        out = answered(a)
        h = make("gdn", [a, next_turn(a, out, False, 1, 60)], 3, forced_prefix=True, eos_free_finish=True,
                 max_num_seqs=2, accept_p=0.7)
        h.probe = Probe(h)
        run(h)
        if not h.sched.eos_free[0]:
            continue
        seen += 1
        d = len(a.prompt) + len(out) - 2
        slot = h.plans[0]["admit"][0]
        p = next(p for p in h.plans if slot in p["free"])
        assert p["fold"] == [(slot, FOLD_ALL)] and p["admit"] and h.probe.in_flight == {0: False}
        assert h.probe.donated == {0: d} and h.results[1].cached == d
    assert seen >= 2


@pytest.mark.parametrize("k", [0, 3])
def test_a_nan_in_the_row_an_eos_free_finish_dropped_rolls_its_donation_back(k) -> None:
    """Qwen's shape: `donor` with a NaN at its EOS position. Its EOS-free finish never samples there, so it keeps its
    tokens and STOP; its dropped row samples it and reports the NaN, which rolls back what the finish inserted and
    donated (both stamped with that row's step). Its next turn admitted in the plan right after the finish resumed
    from the donation and fails with it (`_fail_cascade`); one a turn later resumes lower, from what is left."""
    a, out, companion = donor()
    h = make("gdn", [a, companion], k, eos_free_finish=True, max_num_seqs=4, accept_p=0.7)
    h.probe = Probe(h)
    at = run(h).probe.finished_at[0]
    d = len(a.prompt) + len(out) - 1
    for later in (1, 2):
        a, _, companion = donor(nonfinite_after=len(out) - 1)
        h = make("gdn", [a, next_turn(a, out, False, 1, at + later), companion], k, eos_free_finish=True,
                 max_num_seqs=4, accept_p=0.7)
        h.probe = Probe(h)
        run(h, failed_ok=True)
        assert a.seed in h.runner.fired and (h.results[0].tokens, h.results[0].finish) == (out, STOP)
        assert h.probe.in_flight == {0: True} and h.probe.donated == {0: d}
        r = h.results[1]
        if later == 1:
            assert 1 in h.cascaded and r.finish == ERROR
        else:
            assert 1 not in h.cascaded and r.finish != ERROR and 0 < r.cached < d


def verify_specs(lead: int, tail: int, ignore: bool = True) -> list[Spec]:
    """Constrained requests with forced runs (lo == hi: every document ends on an EOS-only position), one with lo < hi,
    one past its grammar's end under ignore_eos (unconstrained masks), and an unconstrained one (no masks)."""
    out = [Spec(rid=i, prompt_len=0, prompt=stem(100 + i, 14 + 5 * i), max_tokens=40, seed=0xD00 + i, arrive=i,
                grammar=spec(100 + i, 8 + 2 * i, 8 + 2 * i, lead=lead, tail=tail)) for i in range(4)]
    out.append(Spec(rid=4, prompt_len=0, prompt=stem(104, 16), max_tokens=40, seed=0xD04, arrive=2,
                    grammar=spec(104, 5, 16, lead=lead, tail=tail)))
    if ignore:
        out.append(Spec(rid=5, prompt_len=0, prompt=stem(105, 17), max_tokens=30, seed=0xD05, arrive=3,
                        grammar=spec(105, 6, 6, lead=lead, tail=tail), ignore_eos=True))
    out.append(Spec(rid=6, prompt_len=0, prompt=stem(106, 18), max_tokens=30, seed=0xD06, ignore_eos=True))
    return out


@pytest.mark.parametrize("k", [1, 3, 5, 7])
@pytest.mark.parametrize("accept_p", [0.0, 0.6, 1.0])
def test_verify_mask_counters_are_the_verify_rows(k, accept_p) -> None:
    """The verify-mask counters equal `MaskOracle`'s count of what verification read, under every combination of the
    switches: the masks read per popcount bucket, and each verify row's first rejected draft by the class of the mask
    its target sample was drawn under. The fake drafts unmasked, so forced positions reject their drafts: lead runs
    while their tokens are sampled (the forced prefix takes them away: fewer such rejections), tail runs always; the
    EOS-only mask after a document (lo == hi, no tail) is counted as `eos`, never `forced`; ignore_eos past a grammar's
    end reads unconstrained masks (`free`). (The popcounts' sum is compared over the mask words here, as the scheduler
    sums them; `test_window_mask_sum_counts_vocabulary_tokens` checks it over the vocabulary.)"""
    by: dict[tuple, np.ndarray] = {}
    for lead, tail in ((2, 0), (3, 2)):
        for fp, ef in COMBOS:
            h = make("window16", verify_specs(lead, tail), k, forced_prefix=fp, eos_free_finish=ef, max_num_seqs=4,
                     accept_p=accept_p)
            o = MaskOracle(h)
            run(h)
            o.check(h.sched)
            assert h.sched.window_masks.sum() > 0 and o.read_classes()[FREE] > 0
            by[(lead, tail, fp, ef)] = h.sched.first_rejections.copy()
    for fp, ef in COMBOS:
        assert by[(3, 2, fp, ef)][FORCED] > 0  # the closing runs: sampled, never drafted
        assert by[(2, 0, False, ef)][FORCED] > 0 and by[(2, 0, True, ef)][FORCED] == 0  # no verify row reads them
        assert by[(2, 0, fp, False)][EOS_CLASS] >= by[(2, 0, fp, True)][EOS_CLASS]
    assert by[(2, 0, False, False)][EOS_CLASS] > 0


@pytest.mark.parametrize("k", [3, 5])
def test_window_mask_sum_counts_vocabulary_tokens(k) -> None:
    """`window_masks_sum` (the histogram's sum: "tokens the grammar masks of MTP verify rows allowed") counts the tokens
    each mask verification read allows. An unconstrained mask (a grammar ended under ignore_eos) allows the vocabulary,
    1,000 tokens here; its 32 int32 words hold 1,024 bits, and the 24 past the vocabulary are no tokens."""
    specs = [Spec(rid=0, prompt_len=0, prompt=stem(107, 20), max_tokens=40, seed=0xD07, ignore_eos=True,
                  grammar=spec(107, 5, 5, lead=1, tail=1))]
    h = make("window16", specs, k, forced_prefix=True, eos_free_finish=True, accept_p=0.6)
    o = MaskOracle(h)
    run(h)
    rej, hist, total = o.expected()
    assert h.sched.first_rejections.tolist() == rej.tolist() and h.sched.window_masks.tolist() == hist.tolist()
    assert h.sched.window_masks_sum == total, (h.sched.window_masks_sum, total)


def test_verify_mask_records_on_hand_set_rows() -> None:
    """`fill_masks`'s records and `_rejections` on hand-set rows of one plan (k = 3), against masks computed from deep
    copies of the rows' matchers: a grammar with one forced token at its start and one before its end (lo == hi = 4),
    its EOS 991 (bit 31 of its word: int32's sign bit, read through the uint32 view). A row along the document
    (classes forced, choice, choice, forced); one past its closing token (choice, forced, the EOS alone, then a mask
    nobody reads: the EOS alone too); one whose grammar terminated (every token); one whose first draft its grammar
    rejects (forced, then unread); an unmasked row and one whose request left (no record). Verification: a choice draft
    rejected, the EOS position's draft rejected (`eos`, not `forced`), a terminated row's (`free`), a forced
    position's (`forced`), and a draft equal to its target sample but refused by a commit limit, which is no
    rejection; the masks read are those up to each row's accepted drafts."""
    k, eos = 3, 991
    g = Grammars(FakeGrammar(VOCAB, (eos,)))
    c = g.compile(spec(5, 4, 4, lead=1, tail=1))

    def at(n: int) -> Any:
        m = g.matcher(c)
        for _ in range(n):
            a = m.allowed()
            a[eos] = False
            assert m.consume_tokens([int(np.flatnonzero(a)[len(np.flatnonzero(a)) // 3])])
        return m

    def nxt(m: Any, salt: int = 0) -> int:  # a token the state allows (not the EOS)
        a = m.allowed()
        a[eos] = False
        nz = np.flatnonzero(a)
        return int(nz[(salt * 37) % len(nz)])

    m0, m2 = at(0), at(2)
    f0 = nxt(m0)
    a1 = at(1)
    c1, c1b = nxt(a1, 1), nxt(a1, 2)
    a2 = at(1)
    assert a2.consume_tokens([c1])
    c2 = nxt(a2, 3)
    t2 = nxt(m2, 4)
    s3 = m2.deep_copy()
    assert s3.consume_tokens([t2])
    f3 = nxt(s3)
    bad = int(np.flatnonzero(~m0.allowed())[7])
    sched = Scheduler(max_num_seqs=8, max_num_batched_tokens=1024, max_model_len=1024, page_size=16, num_pages=64,
                      num_state_slots=9, eos_token_ids=(eos,), mtp_k=k, grammars=g)
    rows = [  # (matcher, drafts, ignore_eos, left)
        (at(0), [f0, c1, c2], False, False),
        (at(2), [t2, f3, 7], False, False),
        (None, [11, 12, 13], True, False),
        (at(0), [bad, 3, 4], False, False),
        (at(1), [c1, 5, 6], False, False),  # unmasked below: no record
        (at(1), [c1, 5, 6], False, True),  # its request left since the plan
        (at(1), [c1b, 8, 9], False, False),
    ]
    for slot, (m, d, ignore, left) in enumerate(rows, 1):
        msg = AddRequest(rid=slot, frontend=0, prompt=np.zeros(4, np.int32), max_tokens=64, temperature=0.0,
                         top_p=1.0, top_k=0, min_p=0.0, seed=slot, stop_token_ids=[], ignore_eos=ignore)
        req = Req(msg, slot, slot, 1024)
        req.matcher = m
        sched.slot_req[slot], sched.aid[slot] = (None, 0) if left else (req, 1000 + slot)
        sched.drafts[slot] = d
    n = len(rows)
    inf = Inflight(step_id=1, slots=np.arange(1, n + 1, dtype=np.int64), aids=1000 + np.arange(1, n + 1),
                   sample=np.ones(n, bool), device=np.ones(n, bool), n_tokens=0,
                   masked=np.asarray([0, 1, 2, 3, 5, 6]))
    buf = np.zeros((n * (1 + k), words(VOCAB)), np.int32)
    sched.fill_masks(inf, buf)

    def pops(m: Any, d: list[int]) -> list[int]:
        """The popcounts of a verify row's masks: at its state, then after each draft its grammar takes (until one it
        rejects or its end); the rest, which verification never reads, allow the EOS alone."""
        m, out = m.deep_copy(), []
        out.append(int(m.allowed().sum()))
        for t in d:
            if m.is_stopped() or t == eos or not m.allowed()[t]:
                break
            assert m.consume_tokens([t])
            out.append(int(m.allowed().sum()))
        return out + [1] * (1 + k - len(out))

    full = VOCAB  # an unconstrained mask counts the vocabulary, not its words' padding bits
    want_pop = [pops(m, d) if m is not None else [full] * (1 + k) for m, d, _, _ in rows]
    want_pop[4] = want_pop[5] = [-1] * (1 + k)
    assert want_pop[0] == [1, *want_pop[0][1:3], 1] and want_pop[1][1:] == [1, 1, 1] and want_pop[3] == [1] * 4
    assert inf.wpop.tolist() == want_pop, inf.wpop.tolist()
    F, E, C, X = FORCED, EOS_CLASS, CHOICE, FREE
    assert inf.wcls.tolist()[:4] == [[F, C, C, F], [C, F, E, E], [X] * 4, [F, E, E, E]] and inf.wcls[6, 0] == C
    assert inf.wdraft.tolist() == [[f0, c1, c2], [t2, f3, 7], [11, 12, 13], [bad, 3, 4], [-1] * 3, [-1] * 3,
                                   [c1b, 8, 9]]
    tokens = np.full((n, 1 + k), -1, np.int32)
    acc = np.asarray([1, 2, 1, 0, 0, 0], np.int64)  # rows 0-4 and 6 verified (row 5 left: not passed)
    tokens[0, :2] = [f0, c1b]  # its second draft rejected: a choice
    tokens[1, :3] = [t2, f3, eos]  # the closing token accepted, the EOS the target's sample: an eos rejection
    tokens[2, :2] = [11, 99]  # unconstrained
    tokens[3, 0] = f0  # the forced token, not the drafted one
    tokens[4, 0] = 1  # unmasked: no record
    tokens[6, 0] = c1b  # equal to its draft: refused by a commit limit, no rejection
    sched._rejections(inf, np.asarray([0, 1, 2, 3, 4, 6]), acc, tokens)
    assert sched.first_rejections.tolist() == [1, 1, 1, 1]
    read = [*want_pop[0][:2], *want_pop[1][:3], *want_pop[2][:2], want_pop[3][0], want_pop[6][0]]
    assert sched.window_masks.tolist() == np.bincount(np.searchsorted(POPCOUNT_BUCKETS, read),
                                                      minlength=len(POPCOUNT_BUCKETS) + 1).tolist()


def test_the_counters_reach_metrics() -> None:
    """The forced prefix's, the EOS-free finish's and the verify-mask counters through a step's stats
    (`EngineLoop._stats`, across the IPC encoding) to a frontend's /metrics:
    llm_mini_infer:grammar_forced_requests_total and _forced_tokens_total,
    llm_mini_infer:grammar_eos_free_finishes_total{in_flight}, llm_mini_infer:grammar_eos_free_gated_total,
    llm_mini_infer:spec_first_rejections_total{mask} and the llm_mini_infer:spec_window_mask_tokens histogram (its
    bucket counts cumulated, its sum and count the scheduler's), after a run that took forced prefixes, finished
    requests EOS-free with and without their row in flight and gated one."""
    import asyncio
    import shutil

    from llm_mini_infer.api.engine_client import EngineClient
    from llm_mini_infer.config import Config
    from llm_mini_infer.engine import ipc
    from llm_mini_infer.engine.core import EngineCore
    from llm_mini_infer.engine.messages import StepOutputs
    from llm_mini_infer.engine.process import EngineLoop
    from llm_mini_infer.metrics import Metrics

    specs = [*verify_specs(2, 1), *gated_specs(0, 1, False, rid=7, arrive=40)]  # the last one alone: gated
    h = make("window16", specs, 3, forced_prefix=True, eos_free_finish=True, max_num_seqs=4, accept_p=0.6)
    run(h)
    s = h.sched
    assert s.forced_requests and s.eos_free[1] and s.eos_free_gated
    assert s.first_rejections[[FORCED, EOS_CLASS, CHOICE]].all()
    path = ipc.ipc_dir()
    loop = EngineLoop(Config(), EngineCore(s, h.runner), path)
    try:
        e64, e32 = np.zeros(0, np.int64), np.zeros(0, np.int32)
        msg = StepOutputs(step=0, rids=e64, n_new=e32, tokens=e32, finish=np.zeros(0, np.uint8), first_rids=e64,
                          first_cached=e32, first_t_admit_ns=e64, first_t_launch_ns=e64, first_runner_ns=e64,
                          t_sent_ns=0, stats=loop._stats(None))
        stats = ipc.Decoder().decode(ipc.Encoder().encode(msg)).stats
        metrics = Metrics("m", 4096)
        client = EngineClient(path, 0, metrics)
        try:
            client._on_stats(stats)
            page = metrics.render().decode()
        finally:
            asyncio.run(client.close())
    finally:
        loop.close()
        shutil.rmtree(path, ignore_errors=True)
    assert (stats.grammar_forced_requests, stats.grammar_forced_tokens) == (s.forced_requests, s.forced_tokens)
    assert stats.grammar_eos_free == s.eos_free.tolist() and stats.grammar_eos_free_gated == s.eos_free_gated
    assert stats.spec_first_rejections == s.first_rejections.tolist()
    assert stats.spec_window_mask_counts == s.window_masks.tolist() and stats.spec_window_mask_sum == s.window_masks_sum
    base = 'model_name="m",engine="0"'
    lines = [f"llm_mini_infer:grammar_forced_requests_total{{{base}}} {float(s.forced_requests)}",
             f"llm_mini_infer:grammar_forced_tokens_total{{{base}}} {float(s.forced_tokens)}",
             f'llm_mini_infer:grammar_eos_free_finishes_total{{{base},in_flight="no"}} {float(s.eos_free[0])}',
             f'llm_mini_infer:grammar_eos_free_finishes_total{{{base},in_flight="yes"}} {float(s.eos_free[1])}',
             f"llm_mini_infer:grammar_eos_free_gated_total{{{base}}} {float(s.eos_free_gated)}",
             *(f'llm_mini_infer:spec_first_rejections_total{{{base},mask="{c}"}} {float(n)}'
               for c, n in zip(MASK_CLASSES, s.first_rejections.tolist())),
             *(f'llm_mini_infer:spec_window_mask_tokens_bucket{{{base},le="{b}"}} {n}'
               for b, n in zip((*POPCOUNT_BUCKETS, "+Inf"), np.cumsum(s.window_masks).tolist())),
             f"llm_mini_infer:spec_window_mask_tokens_sum{{{base}}} {float(s.window_masks_sum)}",
             f"llm_mini_infer:spec_window_mask_tokens_count{{{base}}} {int(s.window_masks.sum())}"]
    for line in lines:
        assert line in page, line

# -- the targeted fuzz suites (campaign suites `structured_window`, `structured_hybrid` and their `_counts`) ----------


@campaign_settings(200)
@given(structured_window_workloads())
def test_random_structured_window_workloads(w) -> None:
    with saved_on_failure("structured_window", w):
        run_switches_workload(w)


@campaign_settings(200)
@given(structured_hybrid_workloads())
def test_random_structured_hybrid_workloads(w) -> None:
    with saved_on_failure("structured_hybrid", w):
        run_switches_workload(w)


@campaign_settings(200)
@given(structured_window_count_workloads())
def test_random_structured_window_count_workloads(w) -> None:
    """Gemma 4's structured workloads with a draft count per step: every check of theirs, the verify-mask counters
    against `MaskOracle` at each step's count."""
    with saved_on_failure("structured_window_counts", w):
        run_switches_workload(w)


@campaign_settings(200)
@given(structured_hybrid_count_workloads())
def test_random_structured_hybrid_count_workloads(w) -> None:
    """Qwen's structured workloads with a draft count per step: every check of theirs, at each step's count."""
    with saved_on_failure("structured_hybrid_counts", w):
        run_switches_workload(w)


# -- edge cases --------------------------------------------------------------------------------------------------------


def _first_token_turn(h: Harness, rid: int) -> int:
    seen: dict[int, int] = {}
    orig = h.collect

    def collect(report) -> None:
        orig(report)
        for r, res in h.results.items():
            if res.tokens and r not in seen:
                seen[r] = h.turn

    h.collect = collect
    h.run(check_every_turn=True)
    h.check()
    return seen[rid]


def test_a_high_priority_request_never_waits_for_its_forced_tokens() -> None:
    """The forced tokens' pages are best-effort (`_chunk_pages`): a high-priority request whose prompt's pages are free
    but not its forced tokens' takes none, rather than wait behind lower-priority requests, so its first token comes
    when it would with the switch off (within two turns)."""
    turns = {}
    for forced in (False, True):
        lows = [Spec(rid=i, prompt_len=67, max_tokens=180, seed=11 + i, ignore_eos=True, priority=1) for i in range(2)]
        high = Spec(rid=9, prompt_len=128, max_tokens=8, seed=19, priority=0, arrive=5, grammar=spec(5, 6, 6, lead=2))
        h = Harness(specs=[*lows, high], vocab=VOCAB, max_num_seqs=3, max_tokens=256, page_size=64, num_pages=7,
                    hybrid=False, prefix_caching=False, policy="priority", max_model_len=256, forced_prefix=forced)
        turns[forced] = _first_token_turn(h, 9)
    assert turns[True] <= turns[False] + 2, turns


def test_a_precise_final_chunk_keeps_its_forced_tokens_within_the_bound() -> None:
    """With the precise prefix (`precise="bf16"`): a cold prompt's final chunk is PRECISE, and its forced tokens count
    at the PRECISE cost against the mixed-step bound (precise_cost x PRECISE + FP8 <= K / rows), as the chunk's own
    tokens do: here none fit beside the chunk, which the bound already fills, and the request samples them."""
    from ..support.harness import check_precise

    for forced in (False, True):
        long = Spec(rid=0, prompt_len=0, prompt=stem(7, 20), max_tokens=60, seed=5, ignore_eos=True)
        cold = Spec(rid=1, prompt_len=0, prompt=stem(9, 100), max_tokens=10, seed=6, arrive=6,
                    grammar=spec(3, 8, 8, lead=4))
        h = Harness(specs=[long, cold], vocab=VOCAB, max_num_seqs=2, max_tokens=512, page_size=4, num_pages=400,
                    grid=16, max_model_len=256, row_tokens=200, precise="bf16", precise_cost=2.0, hybrid=False,
                    forced_prefix=forced)
        orig = h.core.step

        def step(*a, orig=orig, h=h, **kw):
            r = orig(*a, **kw)
            check_precise(h)
            return r

        h.core.step = step
        h.run(check_every_turn=True)
        h.check()
        for p in h.plans:
            dec = sum(p["device"])
            if dec:
                pre = sum(q for q, pr in zip(p["n_query"], p["precise"]) if pr)
                assert 2 * pre + p["n_tokens"] - pre - dec <= max(16, 200 // dec), (forced, p["step"])


@pytest.mark.parametrize("k", [0, 5])
@pytest.mark.parametrize("fail", [{"mask_empty_after": 3}, {"nonfinite_after": 3}])
def test_a_failing_final_chunk_sends_its_forced_tokens_first(k, fail) -> None:
    """A request whose first sampled position after its forced prefix fails (an empty grammar mask, or a NaN there)
    ends ERROR with its forced tokens sent first, not dropped with the request, as on the sampled path, where each was
    a step's sample before the failing one."""
    out = {}
    for forced in (False, True):
        s = Spec(rid=0, prompt_len=30, max_tokens=20, seed=0xBEEF, grammar=spec(4, 12, 12, lead=3), **fail)
        h = Harness(specs=[s], vocab=VOCAB, window=16, window_page_size=4, page_size=20, num_pages=400, grid=4,
                    max_tokens=64, max_model_len=256, mtp_k=k, draft_kv=False, accept_p=0.8, forced_prefix=forced)
        h.run(check_every_turn=True)
        h.check(failed_ok=True)
        out[forced] = (h.results[0].tokens, h.results[0].finish)
    assert out[True] == out[False] and out[True][1] == ERROR and len(out[True][0]) == 3


def test_two_requests_whose_grammars_stop_together_leave_the_step_one_of_them() -> None:
    """The gate counts the requests finishing in the same `process` as gone: two requests alone in the step in flight
    whose grammars stop together do not both finish now (the step would then run for nothing): the first (in the
    step's row order) keeps its row, whose step samples its EOS a turn later; the second finishes now, its EOS
    appended and its row in flight dropped."""
    specs = [Spec(rid=i, prompt_len=12, max_tokens=40, seed=0xB0 + i, grammar=spec(7 + i, 6, 6)) for i in range(2)]
    h = Harness(specs=specs, vocab=VOCAB, mtp_k=0, page_size=4, num_pages=200, eos_free_finish=True)
    h.probe = Probe(h)
    h.run(check_every_turn=True)
    h.check()
    s = h.sched
    assert s.eos_free.tolist() == [0, 1] and s.eos_free_gated == 1 and s.dropped_samples == 1
    assert h.probe.appended == {1} and h.probe.finished_at[0] == h.probe.finished_at[1] + 1


@pytest.mark.parametrize(("mode", "k"), [("attention", 0), ("window16", 0), ("window16", 3), ("window16", 5),
                                         ("gdn", 0), ("gdn", 3)])
def test_a_prefill_chunk_in_flight_counts_against_the_gate(mode, k) -> None:
    """The gate counts every row of the step in flight, a non-final prefill chunk's too (it samples nothing, but the
    step runs for it): a request whose grammar stops while the step in flight holds its row and another request's
    chunk finishes at once, its EOS appended and its row dropped, a turn before the switch off's finish, with the same
    tokens, plans and companion (a gate counting sample rows only would leave it that step). Its closing
    token is forced (`tail`), so the drafter never drafts it and the EOS never comes in the same step. A companion
    arriving after the stop leaves the request alone in the step in flight: the gate leaves it its EOS step. (Where
    the scheduler takes no forced prefix, the grammar stops two steps later, its two forced tokens sampled: the
    companion arrives two turns later too.)"""
    forcing = make(mode, [], k, forced_prefix=True).sched.forced_prefix
    for arrive in (0, 1, 2, 3, 4, 6, 7):
        runs = []
        for ef in (False, True):
            specs = [Spec(rid=0, prompt_len=0, prompt=stem(1, 12), max_tokens=40, seed=0xA1,
                          grammar=spec(3, 6, 6, lead=2, tail=1)),
                     Spec(rid=1, prompt_len=0, prompt=stem(2, 200), max_tokens=5, seed=0xB2, ignore_eos=True,
                          arrive=arrive + (0 if forcing else 2))]
            h = make(mode, specs, k, max_num_seqs=2, chunk_cap=8, accept_p=0.7, forced_prefix=True,
                     eos_free_finish=ef)
            h.probe = Probe(h)
            runs.append(run(h))
        off, on = runs
        s, late = on.sched, arrive > 4
        assert (s.eos_free.tolist(), s.eos_free_gated) == (([0, 0], 1) if late else ([0, 1], 0)), (arrive, s.eos_free)
        assert on.probe.finished_at[0] == off.probe.finished_at[0] - (not late), arrive
        assert on.probe.finished_at[1] == off.probe.finished_at[1] and len(on.plans) == len(off.plans)
        assert [r.tokens for r in on.results.values()] == [r.tokens for r in off.results.values()]


@pytest.mark.parametrize("k", [0, 3, 5])
@pytest.mark.parametrize("mode", ["window16", "gdn", "attention"])
def test_requests_with_stop_strings_skip_the_switches(mode, k) -> None:
    """A request its frontend ends on stop strings (`AddRequest.stop_strings`) takes no forced prefix and no EOS-free
    finish, which move tokens between steps. Its tokens are the sampled path's; the other requests take both."""
    def specs() -> list[Spec]:
        return [Spec(rid=i, prompt_len=0, prompt=stem(40 + i, 14 + 3 * i), max_tokens=30, seed=0xC0 + i, arrive=i,
                     grammar=spec(40 + i, 7, 7, lead=2, tail=1), stop_strings=i % 2 == 0) for i in range(4)]

    out = run_combos(mode, specs, k, accept_p=0.8, max_num_seqs=4)
    h = out[(True, True)]
    assert set(h.probe.forced) == ({1, 3} if h.sched.forced_prefix else set()) and not h.probe.appended & {0, 2}
    assert h.probe.appended or h.sched.eos_free_gated


@pytest.mark.parametrize("forced", [False, True])
def test_a_cascade_fails_a_request_before_its_forced_prefix(forced) -> None:
    """A request failed by another's rolled-back write (`_fail_cascade`: it resumed from a write whose NaN rolled it
    back) fails the sampled path before any output: with its final chunk's forced tokens unreported, it receives none,
    only its ERROR, as the sampled path's; its forced tokens never count as generated."""
    q = Spec(rid=0, prompt_len=0, prompt=stem(1, 24), max_tokens=30, seed=0xA1, nonfinite_write=23)
    r = Spec(rid=1, prompt_len=0, prompt=stem(1, 24) + stem(5, 4), max_tokens=30, seed=0xB2,
             grammar=spec(3, 9, 9, lead=2), arrive=1)
    h = Harness(specs=[q, r], vocab=VOCAB, max_num_seqs=4, page_size=4, num_pages=400, max_tokens=64,
                max_model_len=256, hybrid=False, forced_prefix=forced)
    h.probe = Probe(h)
    h.run(check_every_turn=True)
    h.check(failed_ok=True)
    assert 1 in h.cascaded and (h.results[1].tokens, h.results[1].finish) == ([], ERROR)
    assert h.sched.generation_tokens == 0 and (1 in h.probe.forced) == forced


@pytest.mark.parametrize("kind", ["nonfinite_after", "mask_empty_after"])
def test_a_failed_request_reports_no_first_token(kind) -> None:
    """A request failing at its final chunk's sample receives its forced tokens with its ERROR, and no first-token
    record (`StepOutputs.first_rids`), as any failed request: its frontend records no first-token stages (such a record
    would carry the request's runner time before the step, 0 on a first admission)."""
    sp = Spec(rid=0, prompt_len=0, prompt=stem(1, 20), max_tokens=40, seed=0xA1, grammar=spec(3, 9, 9, lead=2),
              **{kind: 2})
    h = Harness(specs=[sp], vocab=VOCAB, max_num_seqs=2, page_size=4, num_pages=400, max_tokens=64,
                max_model_len=256, hybrid=False, forced_prefix=True)
    firsts, collect = [], h.collect

    def _collect(report: Any) -> None:
        for out in report.outputs.values():
            firsts.extend(out.first_rids.tolist())
        collect(report)

    h.collect = _collect
    h.run(check_every_turn=True)
    h.check(failed_ok=True)
    assert h.results[0].finish == ERROR and len(h.results[0].tokens) == 2 and not firsts
