"""The tool-call grammars (`grammar.tools`) on the two real tokenizers (Gemma 4, Qwen3.8), compiled as the engine
compiles them (`grammar.engine.LLGuidance`, both backends' EOS): each accepts its chat template's own rendering of a
call and the shapes the models wrote, and refuses a wrong name, a wrong type and a missing required key; a named tool's
opening is the forced prefix under guidance; the end tokens; Qwen's JSON values with any whitespace; `$ref`, names,
keywords, and the OR with a `response_format`."""

from __future__ import annotations

import contextlib
import functools
import json
import random
import re
import tracemalloc

import msgspec
import pytest

from llm_mini_infer.api.app import GRAMMAR_PER_BYTE, GRAMMAR_PER_TOOL, GRAMMAR_SLACK
from llm_mini_infer.api.tools import template_tool
from llm_mini_infer.grammar.engine import Grammars, LLGuidance
from llm_mini_infer.grammar.spec import GrammarSpecs
from llm_mini_infer.grammar.tools import ToolSchemaError, call_grammar
from llm_mini_infer.output.tool_calls import GEMMA4, QWEN3, Family
from llm_mini_infer.render.messages import to_template_messages
from llm_mini_infer.render.renderer import Renderer

from ..support.fixtures import model_dir
from ..support.paths import DATA
from ..support.timing import least_each, under
from ..support.tool_grammar_cases import MALFORMED, chained_defs, check, tool


@functools.cache
def threads():
    """A compile thread: deep grammars compile only on a large stack, as the server's do."""
    from llm_mini_infer.grammar.threads import CompileThreads

    return CompileThreads(1, "test-grammar")


def deep_check(text: str) -> str:
    return threads().run(lambda: check(text))


CASES = json.loads((DATA / "tool_calls" / "cases.json").read_text())
TOOLS = [template_tool(t) for t in CASES["tools"].values()]
ARGS = {"get_weather": {"city": "Paris", "unit": "celsius", "days": 3},
        "search": {"query": "fusion", "sources": ["news", "papers"], "limit": 5, "safe": True},
        "create_event": {"title": "Design review", "start": {"day": "thursday", "time": "14:00"},
                         "attendees": [{"name": "Ana", "email": "ana@example.com"}], "reminder_minutes": 15},
        "book_flight": {"origin": "CDG", "destination": "JFK", "day": "monday",
                        "passengers": [{"kind": "adult", "count": 2}]},
        "get_time": {}}
MODELS = {GEMMA4: "gemma-4-31b-it", QWEN3: "qwen3.8-27b"}
EOS = {(GEMMA4, "guidance"): [1], (GEMMA4, "auto"): [1, 106, 50], (QWEN3, "guidance"): [248046],
       (QWEN3, "auto"): [248046, 248044]}


class Setup:
    def __init__(self, fam: str):
        d = model_dir(MODELS[fam])
        if not (d / "tokenizer.json").exists():
            pytest.skip(f"{d} is absent: uv run tests/scripts/fetch_fixtures.py")
        self.fam, self.r = fam, Renderer(d)
        self.family = Family(fam, self.r.tokenizer)
        cfg = json.loads((d / "config.json").read_text())
        n = (cfg.get("text_config") or cfg)["vocab_size"]
        self.backends = {b: LLGuidance(d / "tokenizer.json", n, EOS[(fam, b)]) for b in ("guidance", "auto")}

    def grammar(self, choice="required", name=None, backend="guidance", tools=TOOLS, **kw) -> str:
        return call_grammar(self.fam, tools, choice, name, forcing=backend == "guidance",
                            ids=self.family.grammar_ids, **kw)[0]

    def rendered_call(self, fn: str, args: dict, tools: list[dict] = TOOLS) -> list[int]:
        """The chat template's own rendering of one call (the history format), as tokens, with the turn's end on
        Gemma."""
        msgs = [{"role": "user", "content": "x"}, {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": fn, "arguments": json.dumps(args)}}]}]
        conv = to_template_messages(msgs, self.r.content_format, self.r.supports_developer_role)
        text = self.r.render(conv, tools=tools, add_generation_prompt=False, enable_thinking=False)
        a, b = ("<|tool_call>", "<tool_call|>") if self.fam == GEMMA4 else ("<tool_call>", "</tool_call>")
        i = text.rindex(a)
        j = text.index(b, i) + len(b)
        return self.ids(text[i:j]) + ([50] if self.fam == GEMMA4 else [])

    def ids(self, text: str) -> list[int]:
        return self.r.tokenizer.encode(text, add_special_tokens=False).ids

    def accepts(self, grammar: str, ids: list[int], backend: str = "guidance") -> bool:
        be = self.backends[backend]
        m = be.instance(be.compile(grammar))
        return bool(m.consume_tokens(ids)) and (m.is_accepting() or m.is_stopped())


_SETUPS: dict[str, Setup] = {}


def get_setup(fam: str) -> Setup:
    """One per family (the backends' tokenizers take seconds to build)."""
    if fam not in _SETUPS:
        _SETUPS[fam] = Setup(fam)
    return _SETUPS[fam]


@pytest.fixture(scope="module", params=[GEMMA4, QWEN3])
def setup(request) -> Setup:
    return get_setup(request.param)


@pytest.mark.parametrize("backend", ["guidance", "auto"])
def test_every_tool_call_the_template_renders_is_accepted(setup, backend) -> None:
    for fn, args in ARGS.items():
        ids = setup.rendered_call(fn, args)
        assert setup.accepts(setup.grammar(backend=backend), ids, backend), fn
        assert setup.accepts(setup.grammar("named", fn, backend=backend), ids, backend), fn
        others = [t for t in TOOLS if t["function"]["name"] != fn][:1]
        assert not setup.accepts(setup.grammar("named", others[0]["function"]["name"], backend=backend), ids,
                                 backend), fn


def test_parallel_calls_and_one(setup) -> None:
    one = setup.rendered_call("get_time", {})
    two = setup.rendered_call("get_weather", {"city": "Rome"})
    if setup.fam == GEMMA4:
        both = one[:-1] + two  # back to back, then the turn's end
    else:
        both = one + setup.ids("\n") + two
    assert setup.accepts(setup.grammar(), both)
    assert not setup.accepts(setup.grammar(parallel=False), both)
    assert setup.accepts(setup.grammar(parallel=False), one)
    same = one[:-1] + one if setup.fam == GEMMA4 else one + setup.ids("\n") + one
    assert not setup.accepts(setup.grammar("named", "get_time"), same)  # a named tool: one call
    assert setup.accepts(setup.grammar("named", "get_time"), one)


def test_wrong_name_type_or_missing_key_is_refused(setup) -> None:
    g = setup.grammar()
    if setup.fam == GEMMA4:
        D = '<|"|>'
        bad = [f"<|tool_call>call:get_wether{{city:{D}Paris{D}}}<tool_call|>",  # a name no tool has
               f"<|tool_call>call:get_weather{{city:{D}Paris{D},days:{D}three{D}}}<tool_call|>",  # a type
               "<|tool_call>call:get_weather{days:3}<tool_call|>",  # the required city missing
               f"<|tool_call>call:get_weather{{unit:{D}kelvin{D},city:{D}Paris{D}}}<tool_call|>"]  # enum, order
        good = f"<|tool_call>call:get_weather{{city:{D}Paris{D},days:3}}<tool_call|>"
        end = [50]
    else:
        def q(name: str, params: str) -> str:
            return f"<tool_call>\n<function={name}>\n{params}</function>\n</tool_call>"
        p = "<parameter={}>\n{}\n</parameter>\n".format
        bad = [q("get_wether", p("city", "Paris")), q("get_weather", p("city", "Paris") + p("days", "three")),
               q("get_weather", p("days", "3")), q("get_weather", p("city", "Paris") + p("unit", "kelvin"))]
        good = q("get_weather", p("city", "Paris") + p("days", "3"))
        end = []
    assert setup.accepts(g, setup.ids(good) + end)
    for text in bad:
        assert not setup.accepts(g, setup.ids(text) + end), text


def test_the_turn_ends_as_the_models_end_it(setup) -> None:
    """Gemma: `<|tool_response>` is named and consumed, under each backend's EOS (guidance's alone is `<eos>`); Qwen:
    the grammar's EOS (`<|im_end|>`) after the last call."""
    call = setup.rendered_call("get_time", {})
    for backend in ("guidance", "auto"):
        be = setup.backends[backend]
        m = be.instance(be.compile(setup.grammar(backend=backend)))
        if setup.fam == GEMMA4:
            assert m.consume_tokens(call[:-1]) and not m.is_accepting()  # a call alone does not end the grammar
            assert m.consume_tokens([50]) and m.is_stopped()
        else:
            assert m.consume_tokens(call) and m.is_accepting()
            assert m.consume_tokens([248046])


def test_a_named_tools_opening_is_the_forced_prefix_under_guidance(setup) -> None:
    g = Grammars(setup.backends["guidance"])
    forced = g.forced_prefix(setup.grammar("named", "get_weather"))
    text = setup.r.tokenizer.decode(list(forced), skip_special_tokens=False)
    # Qwen's: the newline after the parameter's tag is not forced, as the value's first token may take it
    want = '<|tool_call>call:get_weather{city:<|"|>' if setup.fam == GEMMA4 else \
        "<tool_call>\n<function=get_weather>\n<parameter=city>"
    assert text == want, text
    assert Grammars(setup.backends["auto"]).forced_prefix(setup.grammar("named", "get_weather", backend="auto")) == ()


def test_qwen_json_values_take_any_whitespace() -> None:
    """Qwen's non-string values are JSON as `tojson` writes it (", " and ": "), under the image's compact guidance too:
    the call grammar's `%json` items take any whitespace whatever the server's `disable_any_whitespace`."""
    s = get_setup(QWEN3)
    ids = s.rendered_call("create_event", ARGS["create_event"])
    assert '", "' in s.r.tokenizer.decode(ids)  # the template's separators
    assert s.accepts(s.grammar(backend="guidance"), ids, "guidance")
    q = "<tool_call>\n<function=search>\n<parameter=query>\nx\n</parameter>\n<parameter=sources>\n{}\n</parameter>\n" \
        "</function>\n</tool_call>"
    for arr in ('["news","papers"]', '[ "news" , "papers" ]', '["news", "papers"]'):
        assert s.accepts(s.grammar(), s.ids(q.format(arr)))
    assert not s.accepts(s.grammar(), s.ids(q.format('["blogs"]')))  # the items' enum
    t = "<tool_call>\n<function=search>\n<parameter=query>\nx\n</parameter>\n<parameter=safe>\n{}\n</parameter>\n" \
        "</function>\n</tool_call>"
    for spelling in ("true", "True", "false"):  # the parse types each as a boolean
        assert s.accepts(s.grammar(), s.ids(t.format(spelling)))
    assert not s.accepts(s.grammar(), s.ids(t.format("yes")))


def test_refs_names_and_keywords(setup) -> None:
    flight = setup.rendered_call("book_flight", ARGS["book_flight"])
    assert setup.accepts(setup.grammar("named", "book_flight"), flight)  # $ref into $defs, resolved
    bad = {"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {
        "a": {"$ref": "#/$defs/missing"}}}}}
    with pytest.raises(ToolSchemaError, match="does not resolve"):
        setup.grammar(tools=[bad])
    for kw in ("not", "if", "patternProperties", "allOf"):
        t = {"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {
            "a": {"type": "string", kw: {}}}}}}
        with pytest.raises(ToolSchemaError, match=kw):
            setup.grammar(tools=[t])
    t = {"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {
        "a": {"type": "string", "pattern": "^x", "maxLength": 3, "title": "A"}}}}}
    _, loose = call_grammar(setup.fam, [t], "required", ids=setup.family.grammar_ids)
    assert loose == ["maxLength", "pattern"]
    evil = {"type": "function", "function": {"name": 'f"} | /(.|\\n)*/ x: {"', "parameters": None}}
    g = setup.grammar(tools=[evil])
    assert not check(g)  # the name is one literal: the grammar still compiles, and is no wider
    assert not setup.accepts(g, setup.ids("anything"))


def obj(properties: dict, required: list | None = None, **more) -> dict:
    return {"type": "object", "properties": properties,
            "required": list(properties) if required is None else required, **more}


@pytest.mark.parametrize("backend", ["guidance", "auto"])
def test_gemma_writes_keys_in_the_templates_order(backend) -> None:
    """Gemma's template writes a call's keys (nested ones too) in `dictsort` order, which ignores case, ties in their
    order: body, opts, URL, user_id, userId; the grammar takes that rendering, under both backends."""
    s = get_setup(GEMMA4)
    tools = [tool("f", obj({"URL": {"type": "string"}, "body": {"type": "string"}, "user_id": {"type": "integer"},
                            "userId": {"type": "integer"},
                            "opts": obj({"Zeta": {"type": "integer"}, "alpha": {"enum": [{"B": 1, "a": 2}]}})}))]
    ids = s.rendered_call("f", {"URL": "u", "body": "b", "user_id": 1, "userId": 2,
                                "opts": {"Zeta": 1, "alpha": {"B": 1, "a": 2}}}, tools)
    text = s.r.tokenizer.decode(ids, skip_special_tokens=False)
    assert text.startswith('<|tool_call>call:f{body:<|"|>b<|"|>,opts:{alpha:{a:2,B:1},Zeta:1},URL:'), text
    assert s.accepts(s.grammar(tools=tools, backend=backend), ids, backend)


def test_qwen_copies_only_the_definitions_a_parameter_reaches() -> None:
    """Qwen's `%json` items carry the definitions their schema reaches, no others, and parameters with one schema share
    one item: 100 parameters into 100 definitions take about their schema's size. When 100 each reach a chain of 14,
    the grammar would take about 57 times it, and `max_bytes` (the server's bound) refuses it."""
    s = get_setup(QWEN3)
    for chain in (False, True):
        t = chained_defs(chain)
        size = len(json.dumps(t["function"]["parameters"]))
        if not chain:
            g = s.grammar(tools=[t])
            assert len(g) < 2 * size and not check(g)
            continue
        with pytest.raises(ToolSchemaError, match="bytes"):
            call_grammar(QWEN3, [t], "required", ids=s.family.grammar_ids, max_bytes=4 * size + (64 << 10))
    shared = tool("s", obj({f"a{i}": {"type": "array", "items": {"$ref": "#/$defs/D"}} for i in range(100)},
                           [], **{"$defs": {"D": obj({f"f{j}": {"type": "string"} for j in range(50)})}}))
    assert len(s.grammar(tools=[shared])) < 2 * len(json.dumps(shared["function"]["parameters"]))


def test_qwen_takes_draft_07_definitions() -> None:
    """A `$ref` into `definitions` (draft 07's), nested in an array: Qwen's `%json` item resolves it, as Gemma's
    grammar does."""
    t = tool("g", obj({"rows": {"type": "array", "items": {"$ref": "#/definitions/O"}}},
                      **{"definitions": {"O": obj({"n": {"type": "integer"}})}}))
    for fam in (GEMMA4, QWEN3):
        s = get_setup(fam)
        g = s.grammar(tools=[t])
        assert s.accepts(g, s.rendered_call("g", {"rows": [{"n": 1}, {"n": 2}]}, [t])), fam
        assert not s.accepts(g, s.rendered_call("g", {"rows": [{"n": "x"}]}, [t])), fam


@pytest.mark.parametrize("defs", [{"A": {"$ref": "#/$defs/A"}},
                                  {"A": {"$ref": "#/$defs/B"}, "B": {"$ref": "#/$defs/A"}},
                                  {"A": obj({"next": {"$ref": "#/$defs/A"}})},
                                  {"A": {"anyOf": [{"$ref": "#/$defs/A"}, {"$ref": "#/$defs/A"}]}}],
                         ids=["self", "pair", "required-self", "anyof-self"])
def test_a_ref_cycle_that_produces_no_value_is_refused(setup, defs) -> None:
    t = tool("f", obj({"a": {"$ref": "#/$defs/A"}}, **{"$defs": defs}))
    with pytest.raises(ToolSchemaError, match="cycle"):
        setup.grammar(tools=[t])


def test_a_ref_cycle_with_a_value_compiles(setup) -> None:
    """A recursion that can stop (an alternative, an optional key, an array) compiles, and takes values."""
    t = tool("f", obj({"a": {"$ref": "#/$defs/A"}, "n": {"$ref": "#/$defs/Node"}}, **{"$defs": {
        "A": {"anyOf": [{"$ref": "#/$defs/A"}, {"type": "integer"}]},
        "Node": obj({"kids": {"type": "array", "items": {"$ref": "#/$defs/Node"}}, "up": {"$ref": "#/$defs/Node"}},
                    [])}}))
    g = setup.grammar(tools=[t])
    assert not check(g)
    assert setup.accepts(g, setup.rendered_call("f", {"a": 5, "n": {"kids": [{"kids": []}, {"up": {}}]}}, [t]))


def test_an_empty_enum_is_refused(setup) -> None:
    for s in ({"type": "string", "enum": []}, {"anyOf": []}):
        with pytest.raises(ToolSchemaError, match="non-empty"):
            setup.grammar(tools=[tool("f", obj({"a": s}))])


def test_a_diamond_of_refs_builds_in_linear_time(setup) -> None:
    """14 definitions, each an anyOf of two refs to the next (16,384 paths; the deepest the nesting cap takes): each
    definition is walked once, so the grammar builds at once."""
    defs = {f"D{i}": {"anyOf": [{"$ref": f"#/$defs/D{i + 1}"}, {"$ref": f"#/$defs/D{i + 1}"}]} for i in range(14)}
    defs["D14"] = {"type": "integer"}
    t = tool("f", obj({"p": {"$ref": "#/$defs/D0"}}, **{"$defs": defs}))
    g = ""

    def built() -> None:
        nonlocal g
        g = setup.grammar(tools=[t])

    assert under(built, 2) < 2
    assert setup.accepts(g, setup.rendered_call("f", {"p": 7}, [t]))


def server_bound(t: dict) -> int:
    """`api.app.App.tool_grammar`'s bound for one tool."""
    size = len(msgspec.json.encode(t["function"]["parameters"]))
    return GRAMMAR_PER_BYTE * (size + GRAMMAR_PER_TOOL + len(t["function"]["name"].encode())) + GRAMMAR_SLACK


def wrapped_chain(n: int, m: int) -> dict:
    """`n` array parameters, each with its own `maxItems`, into one chain of `m` definitions (each points to the next):
    each Qwen parameter's `%json` item carries the whole chain."""
    defs = {f"D{i}": {"type": "object", "description": "z" * 300,
                      "properties": {"v": {"type": "string"}, "n": {"$ref": f"#/$defs/D{i + 1}"}}} for i in range(m)}
    defs[f"D{m}"] = {"type": "string"}
    props = {f"p{j}": {"type": "array", "maxItems": j + 1, "items": {"$ref": "#/$defs/D0"}} for j in range(n)}
    return tool("amp", {"type": "object", "properties": props, "$defs": defs})


def test_the_grammar_bound_stops_the_build() -> None:
    """Qwen parameters that each reach a definition chain through a wrapper of their own (400 into 14: 37 KB of schema
    for a grammar of 2.2 MB), one tool or four: the build stops when its bytes pass the server's bound, so the 400
    takes time and memory of the order of the bound (4 times the schemas and 64 KB), never of the grammar's full
    size; four tools take about twice the time of one (the bound is twice that), with no slack added."""
    s = get_setup(QWEN3)

    def amps(n: int) -> list[dict]:
        out = []
        for i in range(n):
            t = wrapped_chain(400, 14)
            t["function"]["name"] = f"amp{i}"
            out.append(t)
        return out

    def build(tools: list[dict]) -> None:
        with pytest.raises(ToolSchemaError, match="bytes"):
            call_grammar(QWEN3, tools, "required", ids=s.family.grammar_ids,
                         max_bytes=sum(server_bound(t) for t in tools))

    small_t, big_t = amps(1), amps(4)
    small, big = least_each(lambda: build(small_t), lambda: build(big_t))
    assert big < 4 * small and big < 1, (small, big)
    tracemalloc.start()  # one build, the collector on
    try:
        build(big_t)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 4 * sum(server_bound(t) for t in big_t), (peak, sum(server_bound(t) for t in big_t))


def chain(n: int, wide: int = 0, link: str = "required") -> dict:
    """`n` definitions, each leading to the next, the last an integer; `link` how (each requiring the next, as a
    property: what can produce a value flows from the chain's end; or `deep`: through three nested objects, `ref`: a
    bare `$ref`, `items`, `additional`, `anyof`: an alternative); and `wide` more properties, each into a definition of
    its own, in objects of 320."""
    def step(nxt: dict) -> dict:
        return {"required": obj({"n": nxt}), "ref": nxt, "items": {"type": "array", "items": nxt},
                "additional": {"type": "object", "additionalProperties": nxt},
                "deep": obj({"a": obj({"b": obj({"c": nxt})})}), "anyof": {"anyOf": [nxt, {"type": "null"}]}}[link]

    defs = {f"D{i}": step({"$ref": f"#/$defs/D{i + 1}"}) for i in range(n)}
    defs[f"D{n}"] = {"type": "integer"}
    props = {"p": {"$ref": "#/$defs/D0"}}
    for j in range(wide):
        defs[f"W{j}"] = {"type": "string"}
        props.setdefault(f"g{j // 320}", obj({}, []))["properties"][f"w{j}"] = {"$ref": f"#/$defs/W{j}"}
    return tool("chain", obj(props, **{"$defs": defs}))


def nested(depth: int, ref: bool) -> dict:
    """A two-way `anyOf` nested `depth` deep: an object whose `a` holds the next level, or null; the innermost value a
    `$ref` to an integer (`ref`), or an integer."""
    s: dict = {"$ref": "#/$defs/Leaf"} if ref else {"type": "integer"}
    for _ in range(depth):
        s = {"anyOf": [obj({"a": s}), {"type": "null"}]}
    if ref:
        return tool("nest", obj({"p": {"$ref": "#/$defs/Big"}}, **{"$defs": {"Big": s, "Leaf": {"type": "integer"}}}))
    return tool("nest", obj({"p": s}))


def timed_grammars(setup: Setup, *tools: dict, refused: str | None = None) -> tuple[float, ...]:
    """The time each of `tools`' grammars takes to build, or to be refused with `refused` in the message (`least_each`:
    the runs interleave)."""
    def once(t: dict):
        def run() -> None:
            if refused is None:
                assert "start:" in setup.grammar(tools=[t])
            else:
                with pytest.raises(ToolSchemaError, match=refused):
                    setup.grammar(tools=[t])
        return run
    return least_each(*(once(t) for t in tools))


def nested_many(depth: int, ref: bool, n: int = 35) -> dict:
    """`n` parameters, each a `nested` alternative `depth` deep (35 of 14: 980 alternatives, at their cap)."""
    one = nested(depth, ref)["function"]["parameters"]
    defs = one.get("$defs")
    params = obj({f"q{j}": one["properties"]["p"] for j in range(n)}, **({"$defs": defs} if defs else {}))
    return tool("nest", params)


# Alone on CI's runner (2 cores, 4 vCPUs, two workers): beside a second worker on an earlier 1-core runner, the
# refused chains once measured 8.05 times
@pytest.mark.serial
def test_the_schema_analysis_is_linear(setup) -> None:
    """Which definitions can produce a value, and the grammar's rules, take time linear in the schema: a chain of 12
    definitions beside 960 more (the definitions' cap) takes about 4 times one of 3 beside 240 (the chain sets how far
    that flows); 35 parameters of two alternatives a level, 14 levels deep (the deepest the nesting cap takes, and
    the alternatives' cap), take about 5 times 3 levels, and llguidance compiles them; a chain past the nesting cap is
    refused in time linear in it. Each bound is twice the measured ratio, with no slack added."""
    small, big = timed_grammars(setup, chain(3, 240), chain(12, 960))
    assert big < 8 * small and big < 1, (small, big)
    for ref in (True, False):
        small, big = timed_grammars(setup, nested_many(3, ref), nested_many(14, ref))
        assert big < 10 * small and big < 1, (ref, small, big)
    assert not check(setup.grammar(tools=[nested_many(14, True)]))
    small, big = timed_grammars(setup, chain(200), chain(800), refused="levels deep")
    assert big < 8 * small and big < 1, (small, big)


def ring(m: int) -> dict:
    """A cycle of `m` definitions through optional keys: it can produce values, and a path enters each once."""
    defs = {f"R{i}": obj({"n": {"$ref": f"#/$defs/R{(i + 1) % m}"}}, []) for i in range(m)}
    return tool("ring", obj({"p": {"$ref": "#/$defs/R0"}}, **{"$defs": defs}))


@pytest.mark.parametrize("link", ["required", "deep", "ref", "items", "additional", "anyof", "cycle"])
def test_the_nesting_cap_takes_32_levels(setup, link) -> None:
    """The longest chain of definitions (or cycle) the nesting cap takes, for each link kind, builds on both families
    and llguidance compiles it; one link more nests past `MAX_NESTING` (32) and is a 400 before any grammar is built.
    Gemma's builder, Qwen's `%json` walk and llguidance all recurse a few frames a level, far from any limit here."""
    from llm_mini_infer.grammar.bounds import MAX_NESTING

    make = (lambda n: ring(n)) if link == "cycle" else (lambda n: chain(n, link=link))
    n = 1
    while True:
        try:
            setup.grammar(tools=[make(n + 1)])
        except ToolSchemaError as exc:
            depth = int(re.search(r"nests (\d+) levels deep", str(exc)).group(1))
            break
        n += 1
    per = {"required": 2, "deep": 4, "ref": 1, "items": 2, "additional": 2, "anyof": 2, "cycle": 2}[link]
    assert MAX_NESTING < depth <= MAX_NESTING + per, (n, depth)
    assert not deep_check(setup.grammar(tools=[make(n)])), (link, n)


def ast(n: int) -> dict:
    """A recursive union: `Node`, an anyOf of `n` node types, each an object whose `left` and `args` hold Nodes."""
    defs = {"Node": {"anyOf": [{"$ref": f"#/$defs/T{i}"} for i in range(n)]}}
    for i in range(n):
        defs[f"T{i}"] = obj({"kind": {"const": f"k{i}"}, "left": {"$ref": "#/$defs/Node"},
                             "args": {"type": "array", "items": {"$ref": "#/$defs/Node"}}}, ["kind"])
    return tool("ast", obj({"root": {"$ref": "#/$defs/Node"}}, **{"$defs": defs}))


def test_a_wide_recursive_union_is_shallow(setup) -> None:
    """A union of 200 node types that each refer back to it nests a few levels, whatever its width: every path through
    it enters the union and one node type, so it builds, and llguidance compiles it."""
    g = setup.grammar(tools=[ast(200)])
    assert not check(g)


def test_qwen_follows_a_ref_chain_once() -> None:
    """2,000 Qwen parameters with their own annotations (5 tools of 400), each a `$ref` to the head of a chain of
    definitions that are each a `$ref` to the next: each parameter follows the chain at once, so 4 times the chain
    (28, the longest the nesting cap takes) takes about the same time, with no slack added."""
    s = get_setup(QWEN3)

    def tools_of(m: int) -> list[dict]:
        tools = []
        for t in range(5):
            defs = {f"D{i}": {"$ref": f"#/$defs/D{i + 1}"} for i in range(m)}
            defs[f"D{m}"] = {"type": "integer"}
            props = {f"p{j}": {"$ref": "#/$defs/D0", "description": f"p{j}"} for j in range(400)}
            tools.append(tool(f"aliases{t}", {"type": "object", "properties": props, "$defs": defs}))
        return tools

    a, b = tools_of(7), tools_of(28)
    small, big = least_each(lambda: s.grammar(tools=a), lambda: s.grammar(tools=b))
    assert big < 2 * small and big < 1, (small, big)


def test_the_bound_counts_only_what_the_grammar_holds() -> None:
    """A Qwen parameter's own nested `$defs` is replaced in its `%json` item by the definitions it reaches: a grammar
    that holds them, below the bound, builds under it."""
    s = get_setup(QWEN3)
    t = tool("f", {"type": "object", "properties": {"a": {
        "type": "array", "$defs": {"X": {"type": "string", "description": "y" * 20_000}},
        "items": {"$ref": "#/$defs/D"}}}, "$defs": {"D": {"type": "integer"}}})
    text = call_grammar(QWEN3, [t], "required", ids=s.family.grammar_ids)[0]
    assert len(text.encode()) < 2000
    assert call_grammar(QWEN3, [t], "required", ids=s.family.grammar_ids, max_bytes=2000)[0] == text


@pytest.mark.parametrize("form", MALFORMED, ids=[json.dumps(f) for f in MALFORMED])
def test_a_malformed_schema_value_is_refused(setup, form) -> None:
    """A wrong-typed value where a keyword takes a type, a list of keys or a schema: a `ToolSchemaError` (the API's
    400, param `tools`) at the root and nested, never another exception."""
    for params in ({"type": "object", "properties": {"a": {}}, **form}, obj({"x": form})):
        with pytest.raises(ToolSchemaError):
            setup.grammar(tools=[tool("f", params)])


def test_a_mutated_schema_is_a_grammar_or_a_tool_schema_error(setup) -> None:
    """Every keyword value of the corpus tools' schemas replaced, one at a time, by values of other JSON types: each
    mutant builds a grammar or is a `ToolSchemaError`, never another exception."""
    rng = random.Random(20261103)
    junk = [None, True, False, 0, 1, -2.5, "", "x", "object", [], [1], [None], ["a", 1], [[]], [{}], {}, {"a": 1},
            {"type": 3}, {"$ref": 5}]

    def sites(x: object, path: tuple = ()) -> list[tuple]:
        out = []
        if isinstance(x, dict):
            for k, v in x.items():
                out.append((*path, k))
                out += sites(v, (*path, k))
        elif isinstance(x, list):
            for i, v in enumerate(x):
                out += sites(v, (*path, i))
        return out

    n = 0
    for t in TOOLS:
        params = t["function"]["parameters"]
        if not params:
            continue
        for path in sites(params):
            for value in rng.sample(junk, 6):
                mutant = json.loads(json.dumps(params))
                at = mutant
                for k in path[:-1]:
                    at = at[k]
                at[path[-1]] = value
                with contextlib.suppress(ToolSchemaError):
                    setup.grammar(tools=[{**t, "function": {**t["function"], "parameters": mutant}}])
                n += 1
    assert n > 500, n


def test_a_root_ref_is_the_object(setup) -> None:
    """`parameters` that is a `$ref` (pydantic's recursive model): its definition is the call's object, its required
    key enforced; alternatives or a type other than object at the root are a 400."""
    t = tool("f", {"$ref": "#/$defs/P", "$defs": {"P": obj({"a": {"type": "integer"}})}})
    g = setup.grammar(tools=[t])
    assert setup.accepts(g, setup.rendered_call("f", {"a": 1}, [t]))
    for args in ({}, {"zzz": "x"}, {"a": "x"}):
        assert not setup.accepts(g, setup.rendered_call("f", args, [t])), args
    for root in ({"anyOf": [obj({"a": {"type": "integer"}}), obj({"b": {"type": "string"}})]},
                 {"$ref": "#/$defs/S", "$defs": {"S": {"type": "string"}}}):
        with pytest.raises(ToolSchemaError, match="object schema"):
            setup.grammar(tools=[tool("f", root)])


def test_keywords_are_checked_at_every_level(setup) -> None:
    """A keyword nested in a parameter's schema: the same 400 on both families, and the same keywords logged."""
    for kw, v in (("allOf", [{"type": "string"}]), ("not", {}), ("prefixItems", [])):
        t = tool("f", obj({"o": obj({"x": {"type": "array", "items": {"type": "string", kw: v}}})}))
        with pytest.raises(ToolSchemaError, match=kw):
            setup.grammar(tools=[t])
    t = tool("f", obj({"o": obj({"x": {"type": "array", "items": {"type": "string", "pattern": "^x", "maxLength": 3},
                                       "minItems": 1}})}))
    loose = call_grammar(setup.fam, [t], "required", ids=setup.family.grammar_ids)[1]
    assert loose == ["maxLength", "minItems", "pattern"]
    with pytest.raises(ToolSchemaError, match="'float'"):
        setup.grammar(tools=[tool("f", obj({"x": {"type": "float"}}))])


def test_an_object_without_properties_is_a_dict(setup) -> None:
    """`{"type": "object"}` (a plain dict, as pydantic writes one) takes any keys with any values, nested in a parameter
    and as the whole `parameters`: the template's own rendering of a dict argument passes, and on Gemma a key between
    `<|"|>` too (the parse takes both). `additionalProperties` false takes `{}`; with `properties`, an absent
    `additionalProperties` still takes no other key."""
    tools = [tool("opts", obj({"o": {"type": "object"}})), tool("free", {"type": "object"}),
             tool("closed", obj({"o": {"type": "object", "additionalProperties": False}}))]
    arg = {"Zeta": 1, "alpha": {"k": [1, "s", None, True, 2.5, {}]}, "user id": "x", "e": []}
    g = setup.grammar(tools=tools)
    for fn, args in (("opts", {"o": arg}), ("free", arg), ("free", {}), ("closed", {"o": {}})):
        assert setup.accepts(g, setup.rendered_call(fn, args, tools)), (fn, args)
    assert not setup.accepts(g, setup.rendered_call("closed", {"o": {"a": 1}}, tools))
    assert not setup.accepts(setup.grammar(), setup.rendered_call("get_weather", {"city": "Paris", "extra": 1}))
    nested = {**ARGS["create_event"], "start": {**ARGS["create_event"]["start"], "zone": "UTC"}}  # Qwen's %json too
    assert not setup.accepts(setup.grammar(), setup.rendered_call("create_event", nested))
    if setup.fam == GEMMA4:
        D = '<|"|>'
        quoted = f"<|tool_call>call:free{{{D}k{D}:1,j:{{{D}x{D}:{D}y{D}}}}}<tool_call|>"  # keys between <|"|>
        assert setup.accepts(g, [*setup.ids(quoted), 50])


def test_response_format_or_calls(setup) -> None:
    """auto with a response_format: either calls or the schema's JSON, from the first token."""
    specs = GrammarSpecs(backend="guidance", disable_any_whitespace=True)
    item = specs.json_item(json.dumps({"type": "object", "properties": {"answer": {"type": "string"}},
                                       "required": ["answer"], "additionalProperties": False}).encode())
    g = setup.grammar("auto", response_format=item)
    assert setup.accepts(g, setup.rendered_call("get_time", {}))
    assert setup.accepts(g, setup.ids('{"answer":"It is sunny."}'))
    assert not setup.accepts(g, setup.ids("It is sunny."))
    assert not setup.accepts(g, setup.ids('{"answer": "spaced"}'))  # the server's compact JSON


def test_qwen_strict_auto_is_free_text_then_calls() -> None:
    s = get_setup(QWEN3)
    g = s.grammar("auto", triggered=True)
    call = s.rendered_call("get_time", {})
    think = s.ids("Let me see.\n</think>\n\n")
    assert s.accepts(g, s.ids("Sure, here it is.\n\n") + call)
    assert s.accepts(g, think + call)  # a thought's end is allowed in the free text
    assert s.accepts(g, s.ids("No tool needed."))
    one = s.grammar("auto", triggered=True, parallel=False)
    assert not s.accepts(one, call + s.ids("\n") + call)


def recorded_calls() -> list[tuple[str, dict]]:
    out = []
    d = DATA / "tool_calls" / "recorded"
    for path in sorted(d.glob("*-guidance.json")) if d.exists() else ():
        for r in json.loads(path.read_text())["responses"]:
            if r["choice"] == "auto" and r["calls"] and r["ids"] and not r["stream"]:
                out.append((path.stem, r))
    return out


# A recorded call that breaks its tool's schema, which the grammar refuses: Gemma's template does not render `$defs`,
# so the model never saw the passengers' fields and named its own
SCHEMA_BREAKS = {("gemma4", "flight_defs")}


@pytest.mark.parametrize(("source", "rec"), recorded_calls(), ids=[f"{s}-{r['case']}" for s, r in recorded_calls()])
def test_the_models_own_calls_pass_the_grammars(source, rec) -> None:
    """Each recorded `auto` call that starts the model's reply, as its own tokens: the required grammar accepts it,
    and so does its tool's named grammar where the reply is one call (both backends)."""
    s = get_setup(rec["family"])
    ids = rec["ids"]
    if ids[0] != s.family.opener:
        pytest.skip("text before the call")
    if ids[-1] != (50 if rec["family"] == GEMMA4 else 248046):
        pytest.skip("cut by max_tokens: no complete call")
    tools = [template_tool(t) for t in rec["tools"]]
    body = ids if rec["family"] == GEMMA4 else ids[:-1]  # Qwen's <|im_end|> is the grammar's EOS
    names = {c[0] for c in rec["calls"]}
    want = (rec["family"], rec["case"]) not in SCHEMA_BREAKS
    for backend in ("guidance", "auto"):
        g = s.grammar(tools=tools, backend=backend)  # each call's syntax: parallel_tool_calls false is a count
        assert s.accepts(g, body, backend) == want, backend
        if ids.count(s.family.opener) == 1:  # a named tool takes one call
            g = s.grammar("named", next(iter(names)), tools=tools, backend=backend)
            assert s.accepts(g, body, backend) == want, backend


@pytest.mark.parametrize("fam", [GEMMA4, QWEN3])
def test_the_fixed_output_counts_each_family_s_syntax(fam) -> None:
    """A float `const` in a Qwen `%json` item, whose digits llguidance writes in full
    (30 of 1.5e300 force 9,958 bytes in a row); parameter names that start one another (`k` and `k>\\nz`: Qwen's
    blocks then fix their choice, its `</function>` counted); and a percent-encoded `$ref` prefix, which Qwen's items
    would not rewrite."""
    s = get_setup(fam)
    floats = obj({f"p{i}": {"const": 1.5e300} for i in range(30)})
    heads = obj({"k>\nz": {"const": "x" * 2050}, "k": {"const": "x" * 2050}})
    for params in (floats, heads) if fam == QWEN3 else (heads,):
        with pytest.raises(ToolSchemaError, match="bytes of output in a row"):
            s.grammar(tools=[tool("f", params)])
    assert s.grammar(tools=[tool("f", obj({"k>\nz": {"const": "x" * 10}, "k": {"const": "x" * 10}}))])
    odd = obj({"p": {"$ref": "#/%24defs/x"}}, **{"$defs": {"x": {"type": "integer"}}})
    with pytest.raises(ToolSchemaError, match="does not resolve"):
        s.grammar(tools=[tool("f", odd)])


def test_the_definitions_cap_comes_before_the_fixed_output_count() -> None:
    """300 tools of 100 recursive definitions each: the definitions' cap refuses them before any count runs (counting
    the first 1,010 would be slow)."""
    s = get_setup(GEMMA4)

    def star(n: int) -> dict:
        defs = {"hub": {"anyOf": [{"$ref": f"#/$defs/m{i}"} for i in range(n)] + [{"type": "null"}]}}
        for i in range(n):
            defs[f"m{i}"] = obj({"t": {"const": f"k{i}"}, "c": {"type": "array", "items": {"$ref": "#/$defs/hub"}}},
                                ["t"])
        return obj({"x": {"$ref": "#/$defs/hub"}}, **{"$defs": defs})

    def refused() -> None:
        with pytest.raises(ToolSchemaError, match="more than 1,000"):
            s.grammar(tools=[tool(f"f{i}", star(100)) for i in range(300)])

    assert under(refused, 2) < 2
