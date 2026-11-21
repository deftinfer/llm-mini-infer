"""Structured output (architecture.md, "Structured output", "The step loop"): grammars, masks in the step loop, and
their lifecycle.

Through the scheduler and the validating fake runner, a constrained request's output must be exactly
`fake_generate(..., matcher=)`: each token the masked pick of the unmasked one under the grammar's state there. The
fake grammar (`fake_grammar`) allows a hash-chosen two thirds of the vocabulary, so a mask computed at the wrong state
(a token not advanced, a draft not followed, a stale matcher after a preemption) picks another token, and the engine's
grammar then rejects the next one. Under MTP the fake verifies its real drafts against the masked samples.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from hypothesis import given

from llm_mini_infer.engine.messages import FinishReason
from llm_mini_infer.grammar import GrammarError
from llm_mini_infer.grammar.spec import GrammarSpecs, check, json_schema_grammar
from llm_mini_infer.runner.fake import masked_pick, unpack_mask
from llm_mini_infer.sched.scheduler import NEVER_ENDS, could_end, end_drafts

from ..support.fake_grammar import FakeGrammar, spec
from ..support.fixtures import QWEN
from ..support.fuzzing import fuzz_settings, saved_on_failure
from ..support.harness import EOS, Harness, Spec
from ..support.sched_cases import VOCAB
from ..support.structured_cases import SCHEMA, masked_rows, run, run_structured_workload, structured_workloads

TEXT = '{"total": 1, "entries": [{"id": 0, "subjects": ["history"], "formats": null}]}'  # a document of SCHEMA


# -- the frontend's grammar text -------------------------------------------------------------------------------------


def test_json_schema_grammar_and_errors() -> None:
    g = json_schema_grammar(SCHEMA)
    assert g.startswith('%llguidance {"no_forcing": true}\nstart: %json {')
    body = json.loads(g.split("%json ", 1)[1])
    assert body["x-guidance"] == {"whitespace_flexible": True}
    assert list(body["properties"]) == ["total", "entries"]  # property order kept (it is the output's order)
    assert check(g) == ""
    own = json_schema_grammar({"type": "object", "x-guidance": {"whitespace_flexible": False}})
    assert json.loads(own.split("%json ", 1)[1])["x-guidance"] == {"whitespace_flexible": False}
    for bad, msg in (({"type": "strin"}, "Invalid type: strin"), ({"$ref": "#/$defs/x"}, "does not exist"),
                     ({"type": "array", "minItems": 3, "maxItems": 1}, "Unsatisfiable")):
        err = check(json_schema_grammar(bad))
        assert msg in err and "\n" not in err and "%llguidance" not in err, err
    with pytest.raises(GrammarError, match="the keyword 'not' at #/properties/a is not supported"):
        json_schema_grammar({"type": "object", "properties": {"a": {"not": {}}}})  # outside the subset
    with pytest.raises(GrammarError):
        json_schema_grammar([1, 2])


def test_grammar_specs_cache_results_and_errors() -> None:
    gs = GrammarSpecs(size=2)
    a = gs.json_schema(SCHEMA)
    assert gs.json_schema(json.loads(json.dumps(SCHEMA))) is a and gs.hits == 1
    for _ in range(2):
        with pytest.raises(GrammarError, match="Invalid type"):
            gs.json_schema({"type": "strin"})
    assert gs.misses == 2 and gs.hits == 2
    assert gs.json_object().startswith('%llguidance {"no_forcing": true}\nstart: %json {"type":"object"')
    assert len(gs._cache) == 2  # LRU bound


# -- llguidance over the Qwen tokenizer ------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def llg():
    if not (QWEN / "tokenizer.json").exists():
        pytest.skip("tests/.fixtures/models/qwen3.8-27b is absent: uv run tests/scripts/fetch_fixtures.py")
    from llm_mini_infer.grammar.engine import Grammars, LLGuidance

    return Grammars(LLGuidance(QWEN / "tokenizer.json", 248320, (248044, 248046)))


def test_llguidance_masks_along_an_output_and_its_drafts(llg) -> None:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(QWEN / "tokenizer.json"))
    compiled = llg.compile(json_schema_grammar(SCHEMA))
    assert llg.compile(json_schema_grammar(SCHEMA)) is compiled and llg.compiles == 1  # cached by its text
    ids = tok.encode(TEXT, add_special_tokens=False).ids
    words = (248320 + 31) // 32
    buf = np.zeros((8, words), np.int32)
    m = llg.matcher(compiled)
    for i, t in enumerate(ids):
        llg.fill([(m, 0)], [], buf, 1)
        assert unpack_mask(buf[0], 248320)[t], (i, tok.decode([t]))
        assert m.consume_tokens([t])
        if i + 4 < len(ids):  # after the pending token t: masks along three drafts = the masks where they lead
            buf[1:5] = 0
            llg.fill([], [(m, 1, ids[i + 1:i + 4])], buf, 4)
            probe = m.deep_copy()
            for j in range(4):
                llg.fill([(probe, 5)], [], buf, 1)
                assert np.array_equal(buf[1 + j], buf[5]), (i, j)
                if j < 3:
                    assert probe.consume_tokens([ids[i + 1 + j]])
    llg.fill([(m, 0)], [], buf, 1)
    assert np.flatnonzero(unpack_mask(buf[0], 248320)).tolist() == [248044, 248046]  # EOS only where it accepts
    first = llg.matcher(compiled)
    llg.fill([(first, 0)], [], buf, 1)
    assert not unpack_mask(buf[0], 248320)[[248044, 248046]].any()
    # a rejected draft stops the masks there; the matcher is rolled back
    buf[1:5] = 7
    llg.fill([], [(m := llg.matcher(compiled), 1, [ids[0], 12345, ids[2]])], buf, 4)
    assert (buf[3] == 7).all() and (buf[4] == 7).all() and not (buf[2] == 7).all()
    assert m.consume_tokens(ids)


def test_could_end_walks_the_drafts_with_llguidance(llg) -> None:
    """A row could deliver the EOS when the document is complete before it, or after drafts the grammar allows; how
    many drafts it must verify for that (`end_drafts`: up to the one that completes the document); the walk leaves the
    matcher as it was."""
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(QWEN / "tokenizer.json"))
    compiled = llg.compile(json_schema_grammar(SCHEMA))
    ids = tok.encode(TEXT, add_special_tokens=False).ids
    m = llg.matcher(compiled)
    assert m.consume_tokens(ids[:-2])
    assert not could_end(m, [])  # no drafts (no MTP): only a complete document ends the next sample
    assert could_end(m, ids[-2:]) and could_end(m, [*ids[-2:], 248046])
    assert end_drafts(m, ids[-2:]) == end_drafts(m, [*ids[-2:], 248046]) == 2
    assert end_drafts(m, [*ids[-2:], 248046, 11]) == 2
    assert not could_end(m, ids[-2:-1]) and end_drafts(m, ids[-2:-1]) == NEVER_ENDS  # one token short of the end
    assert not could_end(m, [12345, *ids[-2:]])  # a draft the grammar rejects stops the walk
    assert m.consume_tokens(ids[-2:]) and m.is_stopped() and could_end(m, [])  # the walks changed nothing
    assert end_drafts(m, [7]) == 0


def test_a_grammar_the_engine_cannot_compile_raises(llg) -> None:
    with pytest.raises(GrammarError):
        llg.compile('%llguidance {"no_forcing": true}\nstart: %json {"type": "strin"}')
    with pytest.raises(GrammarError):
        llg.compile('%llguidance {"no_forcing": true}\nstart: %json {"type": "strin"}')  # the error is cached too
    assert llg.compile_errors == 1


# -- vLLM's guidance backend over the Gemma 4 tokenizer --------------------------------------------------------------


def test_guidance_grammar_on_gemma() -> None:
    """Backend guidance with compact JSON: llguidance's JSON grammar text (its canonical forcing), the tokenizer's
    `<eos>` the only end, and compact documents only. The default (auto) keeps the Lark emulation of xgrammar. Proven
    equal to vLLM's text, tokenizer and masks in tests/structured/test_grammar_parity.py."""
    gemma = QWEN.parent / "gemma-4-31b-it"
    if not (gemma / "tokenizer.json").exists():
        pytest.skip("tests/.fixtures/models/gemma-4-31b-it is absent: uv run tests/scripts/fetch_fixtures.py")
    from llm_mini_infer.grammar.engine import Grammars, LLGuidance

    text = GrammarSpecs(backend="guidance", disable_any_whitespace=True).json_schema(SCHEMA)
    (grammar,) = json.loads(text)["grammars"]
    assert grammar["json_schema"]["x-guidance"] == {"whitespace_flexible": False} and grammar["lark_grammar"] is None
    assert list(grammar["json_schema"]["properties"]) == ["total", "entries"]
    assert GrammarSpecs().json_schema(SCHEMA).startswith('%llguidance {"no_forcing": true}\nstart: %json {')
    llg = Grammars(LLGuidance(gemma / "tokenizer.json", 262144, "<eos>"))
    assert llg.backend.eos == (1,) and llg.eos == {1}
    with pytest.raises(ValueError, match="no token"):
        LLGuidance(gemma / "tokenizer.json", 262144, "<|im_end|>")
    tok, compiled = llg.backend.tokenizer, llg.compile(text)
    ids = tok.tokenize_bytes(b'{"total":1,"entries":[{"id":0,"subjects":["history"],"formats":null}]}')
    spaced = tok.tokenize_bytes(b'{"total": 1, "entries": [{"id": 0, "subjects": ["history"], "formats": null}]}')
    buf = np.zeros((4, (262144 + 31) // 32), np.int32)
    m = llg.matcher(compiled)
    for i, (t, first) in enumerate(zip(ids, ([14937], [6725]))):  # '{"' and 'total': forced, one token each
        llg.fill([(m, 0)], [], buf, 1)
        assert np.flatnonzero(unpack_mask(buf[0], 262144)).tolist() == first == [t], i
        assert m.consume_tokens([t])
    assert m.consume_tokens(ids[2:-2]) and not could_end(m, []) and could_end(m, [*ids[-2:], 1])
    assert m.consume_tokens(ids[-2:]) and m.is_stopped() and could_end(m, [])
    llg.fill([(m, 0)], [], buf, 1)
    assert np.flatnonzero(unpack_mask(buf[0], 262144)).tolist() == [1]  # not 50 or 106: the tokenizer's EOS only
    s = llg.matcher(compiled)
    assert s.try_consume_tokens(spaced) < len(spaced) and not s.is_error()


def test_guidance_needs_the_tokenizer_eos_among_the_stop_ids(tmp_path) -> None:
    """A completed guidance grammar allows only the tokenizer's eos_token, and a request stops on `eos_token_ids`
    (generation_config.json's), which vLLM joins with the tokenizer's EOS and we do not: without it there, every
    structured request would end in a grammar error. The engine refuses to start. auto ends a grammar on the stop ids
    themselves; Gemma 4's generation_config.json lists `<eos>`."""
    from types import SimpleNamespace

    from llm_mini_infer import config
    from llm_mini_infer.engine.process import _grammars

    gemma = QWEN.parent / "gemma-4-31b-it"
    if not (gemma / "tokenizer.json").exists():
        pytest.skip("tests/.fixtures/models/gemma-4-31b-it is absent: uv run tests/scripts/fetch_fixtures.py")
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        (tmp_path / name).symlink_to(gemma / name)
    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": [106, 50]}))
    caps = SimpleNamespace(grammar_masks=True, backend="fake", vocab_size=262144)

    def grammars(path, backend: str):
        cfg = config.derive(config.load(None, {"model.path": str(path), "runner.kind": "fake",
                                               "structured_output.backend": backend}))
        return cfg, _grammars(cfg, caps)

    with pytest.raises(ValueError, match=r"'<eos>' \(id 1\).* eos_token_ids \[50, 106\]"):
        grammars(tmp_path, "guidance")
    cfg, g = grammars(tmp_path, "auto")
    assert g.eos == {50, 106} and cfg.grammar_eos_token_ids == [50, 106]
    cfg, g = grammars(gemma, "guidance")
    assert g.eos == {1} and cfg.grammar_eos_token_ids == [1] and cfg.eos_token_ids == [1, 50, 106]


# -- the fake grammar --------------------------------------------------------------------------------------------------


def test_fake_grammar_states_and_draft_rule() -> None:
    g = FakeGrammar(64, (63,))
    c = g.compile(spec(7, 2, 4))
    m = g.instance(c)
    a = m.allowed()
    assert 0 < a.sum() < 63 and not a[63]  # no EOS before lo
    toks = []
    for _ in range(4):
        t = masked_pick(5, m.allowed())
        toks.append(t)
        assert m.consume_tokens([t])
        assert m.allowed()[63] == (len(toks) >= 2)
    assert m.is_stopped() and np.flatnonzero(m.allowed()).tolist() == [63]
    m2 = g.instance(c)
    one = m2.deep_copy()
    assert one.consume_tokens([toks[0]])
    bad = int(np.flatnonzero(~one.allowed())[0])  # rejected after the first draft
    assert not one.consume_tokens([bad])
    buf = np.full((6, 2), 9, np.int32)
    g.fill([], [(m2, 1, [toks[0], bad, toks[2]])], buf)
    assert (buf[3] == 9).all() and (buf[4] == 9).all() and not (buf[1] == 9).all() and not (buf[2] == 9).all()
    assert m2.n == 0  # rolled back
    assert m2.try_consume_tokens([toks[0], bad, toks[2]]) == 1 and not m2.is_error()  # llguidance's: no error state
    m2.rollback(1)
    assert m2.n == 0 and not could_end(m2, toks[:1]) and could_end(m2, toks)  # toks reach hi = 4: only EOS after
    assert m2.n == 0
    # llguidance's EOS rule (1.9.1, as 1.7.6): taken while accepting (recorded), or in the call that stops the grammar
    # (not recorded by the fake, where llguidance records it), and never after the end: a later call errs, a try takes
    # nothing
    end = g.instance(c)
    assert end.consume_tokens(toks[:2]) and end.consume_tokens([63]) and end.is_stopped() and end.n == 3
    end.rollback(1)
    assert not end.is_stopped() and end.consume_tokens([toks[2], toks[3], 63]) and end.n == 4
    late = g.instance(c)
    assert late.consume_tokens(toks) and late.try_consume_tokens([63]) == 0 and late.validate_tokens([63]) == 0
    assert not late.is_error() and not late.consume_tokens([63]) and late.is_error()
    # validate_tokens (drafts are cut with it before scheduling): up to and with an EOS, without moving
    v = g.instance(c)
    assert v.consume_tokens(toks[:3]) and v.validate_tokens([toks[3], 63, 63]) == 2 and v.n == 3
    no = int(np.flatnonzero(~v.allowed())[0])
    assert v.validate_tokens([63, toks[3]]) == 1 and v.validate_tokens([no, 63]) == 0 and v.n == 3
    with pytest.raises(GrammarError):
        g.compile("fake:1:5:2")


# -- through the scheduler and the fake runner -------------------------------------------------------------------------


@pytest.mark.parametrize("k", [0, 1, 3])
def test_constrained_outputs_are_the_grammar_s_picks(k) -> None:
    """Constrained and unconstrained requests side by side; the grammars end by forcing EOS at their `hi`."""
    specs = [Spec(rid=i, prompt_len=10 + 7 * i, max_tokens=60, seed=0x5EED_0000 + i,
                  grammar=spec(100 + i, 5 + i, 12 + 3 * i) if i % 3 else "") for i in range(6)]
    h = run(specs, mtp_k=k, accept_p=0.9, max_num_seqs=4, max_tokens=128, page_size=4, num_pages=600)
    for s in specs:
        r = h.results[s.rid]
        if s.grammar:
            assert r.finish == FinishReason.STOP and r.tokens[-1] == EOS  # the grammar ended it
    assert masked_rows(h) > 0 and h.runner.counts["masks"] > 0
    assert h.sched.structured_requests == 4
    if k:
        assert h.runner.counts["masks"] > masked_rows(h)  # 1 + k masks per decode row


def test_drafts_the_grammar_rejects_are_never_accepted() -> None:
    """Every draft the fake makes is the unmasked prediction (accept_p 1): an unconstrained request accepts all k,
    a constrained one only those its grammar allows; outputs stay the grammar's picks. The others do not count as
    drafted (vLLM cuts them before scheduling, none once the grammar has stopped): every draft counted is
    accepted then, but where a commit limit cuts the last row (max_tokens)."""
    a = Spec(rid=0, prompt_len=12, max_tokens=40, seed=0xACE0, ignore_eos=True)
    b = Spec(rid=1, prompt_len=12, max_tokens=40, seed=0xACE1, grammar=spec(5, 30, 60))
    c = Spec(rid=2, prompt_len=12, max_tokens=60, seed=0xACE2, grammar=spec(6, 30, 30))  # ends by its EOS
    ha, hb, hc = (run([s], mtp_k=3, accept_p=1.0, max_num_seqs=2, page_size=4, num_pages=300) for s in (a, b, c))
    assert ha.sched.spec_accepted / ha.sched.spec_draft_tokens > 0.95
    assert ha.sched.spec_draft_tokens == 3 * ha.sched.spec_drafts
    assert hb.sched.spec_draft_tokens < 3 * hb.sched.spec_drafts
    for h in (ha, hb):  # every draft counted is accepted, but those of the last row (max_tokens: a commit limit)
        assert h.sched.spec_draft_tokens - 3 <= h.sched.spec_accepted <= h.sched.spec_draft_tokens
    s = hc.sched
    assert hc.results[2].tokens[-1] == EOS and s.spec_accepted == s.spec_draft_tokens < 3 * s.spec_drafts


def test_preemption_rebuilds_matchers_from_the_generated_tokens() -> None:
    specs = [Spec(rid=i, prompt_len=24, max_tokens=60, seed=(i + 3) * 0x9E3779B9, grammar=spec(i, 50, 59))
             for i in range(3)]
    for k in (0, 3):
        h = run(specs, mtp_k=k, accept_p=0.8, page_size=4, num_pages=1 + 36, max_num_seqs=3, max_tokens=64,
                max_model_len=128, checkpoint_slots=4, grid=16)
        assert h.sched.num_preemptions > 0


def test_aborts_finishes_and_stop_ids_free_matchers() -> None:
    specs = [Spec(rid=0, prompt_len=30, max_tokens=50, seed=11, grammar=spec(1, 40, 49), abort_at=5),
             Spec(rid=1, prompt_len=30, max_tokens=7, seed=12, grammar=spec(2, 40, 49)),  # length
             Spec(rid=2, prompt_len=30, max_tokens=50, seed=13, grammar=spec(3, 2, 6), arrive=3),
             Spec(rid=3, prompt_len=9, max_tokens=50, seed=14, grammar=spec(4, 40, 49), arrive=4, eos_after=1)]
    h = run(specs, mtp_k=3, accept_p=0.7, page_size=4, num_pages=200, max_num_seqs=4)
    assert h.results[1].finish == FinishReason.LENGTH and len(h.results[1].tokens) == 7
    assert h.results[2].finish == FinishReason.STOP


def test_ignore_eos_runs_unconstrained_after_the_grammar_ends() -> None:
    """vLLM's rule: once the grammar accepted an EOS (only possible with ignore_eos), the rest is
    unconstrained; the EOS does not end such a request, so it never waits for it either."""
    for k in (0, 3):
        s = Spec(rid=0, prompt_len=16, max_tokens=40, seed=0xE05, grammar=spec(9, 3, 8), ignore_eos=True)
        h = run([s], mtp_k=k, accept_p=0.9, page_size=4, num_pages=200)
        out = h.results[0].tokens
        assert len(out) == 40 and EOS in out[:9] and h.sched.end_waits == 0


@pytest.mark.parametrize("k", [0, 3])
def test_ignore_eos_past_a_stopped_grammar(k) -> None:
    """A JSON document's grammar allows the EOS only once the document is complete (lo == hi here), and llguidance
    then refuses to consume it in a later call: the forced EOS ends the grammar unconsumed (the matcher
    takes no token once stopped), and the request runs on unconstrained instead of failing."""
    for i in range(4):
        s = Spec(rid=0, prompt_len=16 + i, max_tokens=40, seed=0xE06 + i, grammar=spec(9 + i, 6 + i, 6 + i),
                 ignore_eos=True)
        h = run([s], mtp_k=k, accept_p=0.8, page_size=4, num_pages=200)
        out = h.results[0].tokens
        assert len(out) == 40 and out[6 + i] == EOS and h.sched.grammars.errors == 0, i


@pytest.mark.parametrize("k", [0, 1, 3])
def test_a_lone_request_ending_by_its_grammar_wastes_no_step(k) -> None:
    """Once a request's row in flight could deliver its grammar's EOS, a plan with nothing else skips it, so no step
    runs after the EOS and the device is free for the next request (whose TTFT would otherwise queue behind that
    step). Under MTP the row may not end it after all (a draft before the end rejected): then its next row comes after
    the result, and outputs stay the grammar's picks."""
    for i, p in enumerate((0.0, 0.5, 0.9, 1.0) if k else (0.8,)):
        s = Spec(rid=0, prompt_len=12 + i, max_tokens=60, seed=0xE4D0 + i, grammar=spec(40 + i, 3, 9 + i))
        h = run([s], mtp_k=k, accept_p=p, page_size=4, num_pages=200)
        r = h.results[0]
        assert r.finish == FinishReason.STOP and r.tokens[-1] == EOS
        assert h.sched.dropped_samples == 0 and h.sched.end_waits >= 1, (k, p)
        if not k:
            assert len(h.plans) == len(r.tokens)  # one step per token, none after the EOS


@pytest.mark.parametrize("k", [0, 3])
def test_the_end_rule_skips_rows_only_when_nothing_else_would_run(k) -> None:
    """With another request decoding, the step after the EOS runs anyway: the ending request keeps its row (a skipped
    one could only delay it), which is dropped when its EOS comes back."""
    a = Spec(rid=0, prompt_len=12, max_tokens=60, seed=0xE4E0, grammar=spec(7, 3, 8))
    b = Spec(rid=1, prompt_len=12, max_tokens=60, seed=0xE4E1, ignore_eos=True)  # decodes past a's end
    h = run([a, b], mtp_k=k, accept_p=0.8, page_size=4, num_pages=200)
    assert h.results[0].finish == FinishReason.STOP and h.results[1].finish == FinishReason.LENGTH
    assert h.sched.end_waits == 0 and h.sched.dropped_samples == 1


def test_a_grammar_the_engine_rejects_fails_only_its_request() -> None:
    specs = [Spec(rid=0, prompt_len=10, max_tokens=5, seed=1, grammar="fake:not-a-grammar"),
             Spec(rid=1, prompt_len=10, max_tokens=5, seed=2)]
    h = Harness(specs=specs, vocab=VOCAB)
    h.run()
    assert h.results[0].finish == FinishReason.ERROR and h.results[1].finish == FinishReason.LENGTH
    assert h.sched.grammar_rejects == 1


# -- the targeted fuzz suite -------------------------------------------------------------------------------------------


@fuzz_settings(300)
@given(structured_workloads())
def test_random_structured_workloads(w) -> None:
    with saved_on_failure("structured", w):
        run_structured_workload(w)


def test_json_structural_tokens_are_always_in_a_reduced_draft_head() -> None:
    """JSON allows any whitespace, and long runs of it are late BPE merges (Qwen3.8's 32 newlines: id
    76,374), which a 32,768-token head ranked by id leaves out: every draft then fails. The tokens made only of JSON's
    whitespace and punctuation are forced in, as the special tokens are."""
    import torch

    from llm_mini_infer.runner.cuda import draft_head as dh

    if not (QWEN / "tokenizer.json").exists():
        pytest.skip("tests/.fixtures/models/qwen3.8-27b is absent: uv run tests/scripts/fetch_fixtures.py")
    from tokenizers import Tokenizer

    ids = dh.structural_token_ids(QWEN)
    tok = Tokenizer.from_file(str(QWEN / "tokenizer.json"))
    assert len(ids) == 620 and {198, 76374} <= set(ids) and ids == sorted(ids)
    assert all(set(tok.decode([i])) <= dh.STRUCTURAL for i in ids)
    assert tok.encode("\n" * 32, add_special_tokens=False).ids == [76374]
    head = torch.zeros(248320, 8)
    d = dh.DraftHead(head, size=32768, special=[*dh.special_token_ids(QWEN), *ids])
    assert set(ids) <= set(d.ids.tolist()) and d.size == 32768
    assert dh.structural_token_ids(QWEN.parent / "absent") == []
