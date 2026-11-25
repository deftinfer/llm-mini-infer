"""Schemas inside the JSON Schema subset and its caps: compiled at a bounded cost or refused at once, never the
process's end, and off the event loop.

- **The subset** (`grammar.bounds`): costly shapes, every keyword outside the subset and every merge are a
  400 at once that names the keyword and its JSON pointer; each cap takes its number and refuses one more, over all of
  a grammar's tools; pydantic's and the OpenAI SDK's schemas pass; the costliest shapes at the caps compile at once and
  need a fraction of the compile threads' stack; Qwen's `%json` walk keeps its text, its cost within the grammar's
  bound.
- **The engine** (`grammar.engine`): a deep grammar compiles on its compile thread; a cached error is raised afresh.
- **The compile threads** (`grammar.threads`): a matcher compiled on one is used anywhere; a cache hit stays on the
  caller's thread; one build a key, which a cancelled waiter leaves running and whose failure reaches every waiter; a
  closed pool refuses work; a start that the address-space limit refuses says so.
- **The typing** (`output.tool_args._Typing`): linear on `$ref` cycles and on alternatives that many properties share,
  and the same whichever key it types first.
- **Through the server** (on the fake runner, whose engine compiles with llguidance): each shape gets a 200 or a 400 at
  once, and the server stays up; a slow build leaves another request's stream smooth.

Compiles run on a thread with a large stack, as the engine's do, or in a child process.
"""

from __future__ import annotations

import asyncio
import functools
import http.client
import itertools
import json
import os
import re
import sys
import threading
import time
import traceback
import tracemalloc

import pytest

from llm_mini_infer.grammar import GrammarError, NestingError
from llm_mini_infer.grammar.spec import GrammarSpecs, json_schema_grammar
from llm_mini_infer.grammar.tools import ToolSchemaError, call_grammar
from llm_mini_infer.output.tool_args import fix_types

from ..support.api_util import Server, env_has_model
from ..support.fixtures import QWEN
from ..support.grammar_cases import IDS, MODES, S, bound, bounds, build, child, compiled_on_threads, grammar, obj, tool
from ..support.paths import MODELS_DIR
from ..support.timing import least_each, under

GEMMA = MODELS_DIR / "gemma-4-31b-it"
CHAT = "/v1/chat/completions"


# -- costly shapes -----------------------------------------------------------------------------------------------

def ref_beside_anyof(n: int) -> dict:
    """Definitions chained through a property whose `$ref` sits beside an `anyOf`: llguidance doubled its time and
    memory a level, 2.4 s and 7.4 GB at 21 (guidance-ai/llguidance#436)."""
    defs = {f"D{i}": obj({"p0": {"$ref": f"#/$defs/D{i + 1}", "anyOf": [{"type": "object"}, {"type": "object"}]}},
                         False) for i in range(n)}
    defs[f"D{n}"] = S
    return {**obj({"v": {"$ref": "#/$defs/D0"}}), "$defs": defs}


def pattern_properties(n: int) -> dict:
    """`n` patternProperties keys: llguidance's compile grows as about n^4.5 (69 s at 600, 21 KB;
    guidance-ai/llguidance#438)."""
    patterns = {f"^p{i}_[a-z]+$": {"type": "integer"} for i in range(n)}
    return obj({"v": {"type": "object", "patternProperties": patterns}})


def ref_dag(n: int, fan: int) -> dict:
    """`n` definitions, each an `anyOf` of `$ref`s to the next `fan`: 5.4 s in the frontend's check at 1,500 x 25."""
    defs = {f"M{i}": {"anyOf": [{"$ref": f"#/$defs/M{min(i + j + 1, n)}"} for j in range(fan)]} for i in range(n)}
    defs[f"M{n}"] = S
    return {**obj({"v": {"$ref": "#/$defs/M0"}}), "$defs": defs}


def merged(depth: int, k: int, nprops: int) -> dict:
    """`depth` levels of `k` alternatives merged with `nprops` sibling properties: 9.9 s and 5.9 GB at 12 x 2 x 800."""
    s: dict = S
    for _ in range(depth):
        s = {"type": "object", "properties": {"p": s, **{f"q{i}": {"type": "integer"} for i in range(nprops)}},
             "anyOf": [{"required": ["p"]} for _ in range(k)]}
    return obj({"v": s})


def int_enum(n: int) -> dict:
    return obj({"v": {"enum": list(range(n))}})


TABLE = {"ref-beside-anyOf": (lambda: ref_beside_anyof(21), "'anyOf' beside '$ref'"),
         "patternProperties": (lambda: pattern_properties(600), "'patternProperties'"),
         "dag": (lambda: ref_dag(1500, 25), "1,501 definitions"),
         "merged": (lambda: merged(12, 2, 800), "'type' beside 'anyOf' at #/properties/v"),
         "enum": (lambda: int_enum(100_000), "100,000 enum and const values")}


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("shape", list(TABLE))
def test_costly_shapes_are_refused_at_once(shape, mode) -> None:
    """Each shape, seconds or gigabytes of llguidance's compile within the old bounds, in a `response_format` (both
    backends) and as a tool's parameters (both families): a 400 that names the keyword or the cap, before anything is
    built."""
    make, msg = TABLE[shape]
    if shape == "patternProperties" and mode not in ("auto", "guidance"):
        pytest.skip("the tools' keyword list never took patternProperties")
    schema = make()
    def refused() -> None:
        with pytest.raises((GrammarError, ToolSchemaError), match=re.escape(msg)):
            build(mode, schema)

    assert under(refused, 0.2) < 0.2


# -- the subset -----------------------------------------------------------------------------------------------------

REFUSED = [("allOf", [S, {"maxLength": 3}], "'allOf' at {at} has 2 members"), ("not", {"type": "null"}, None),
           ("if", S, None), ("then", S, None), ("else", S, None), ("patternProperties", {"^a": S}, None),
           ("propertyNames", {"maxLength": 3}, None), ("dependentSchemas", {"a": {"required": ["b"]}}, None),
           ("dependentRequired", {"a": ["b"]}, None), ("unevaluatedProperties", False, None),
           ("unevaluatedItems", False, None), ("contains", {"type": "integer"}, None), ("additionalItems", False, None),
           ("$dynamicRef", "#node", None)]


@pytest.mark.parametrize("backend", ["auto", "guidance"])
def test_a_keyword_outside_the_subset_is_named_with_its_pointer(backend) -> None:
    """Each keyword outside the subset, nested two objects deep in a `response_format`: a 400 that names it and its
    JSON pointer; so are a `$ref` that names no definition (a remote one, a deeper pointer, an anchor) and a nested
    `$id`, under which llguidance would resolve `$ref`s otherwise. The root's `$id`, the annotations and `x-` keys
    pass."""
    at = "#/properties/a/properties/b"
    for kw, value, msg in REFUSED:
        schema = obj({"a": obj({"b": {kw: value}})})
        with pytest.raises(GrammarError, match=re.escape(msg.format(at=at) if msg else
                                                          f"the keyword {kw!r} at {at} is not supported")):
            json_schema_grammar(schema, backend, backend == "guidance")
    defs = {"A": obj({"x": S})}
    for ref in ("https://example.com/s.json", "#/properties/a", "#node", "#/$defs/A/properties/x"):
        with pytest.raises(GrammarError, match=re.escape(f"the $ref {ref!r} at #/properties/r is not supported")):
            json_schema_grammar({**obj({"r": {"$ref": ref}}), "$defs": defs}, backend, backend == "guidance")
    with pytest.raises(GrammarError, match=re.escape("the '$id' at #/properties/a is not supported")):
        json_schema_grammar(obj({"a": {"$id": "https://example.com/a", "type": "string"}}), backend)
    text = json_schema_grammar({"$id": "https://example.com/root", "$schema": "https://json-schema.org/draft/2020-12",
                                **obj({"a": {"$anchor": "a", "type": "string", "readOnly": True, "x-unit": "cm",
                                             "examples": ["x"], "deprecated": False}})}, backend, backend == "guidance")
    assert compiled_on_threads(text) == ""


def test_the_merging_rule() -> None:
    """`anyOf`, `oneOf`, `$ref` and a one-member `allOf` take annotations and definitions alone beside them: anything
    else is a 400 that names it and its pointer, in a `response_format` and in a tool of either family, and so is an
    `allOf` of two. An `allOf` of one `$ref` (older pydantic's) builds the grammar of that `$ref`; pydantic's root
    shapes (a recursive model, a union) keep `$defs` beside them, at the root and nested."""
    a = obj({"k": {"const": "a", "type": "string"}, "x": {"type": "integer"}})

    def doc(v: dict) -> dict:
        return {**obj({"v": v}), "$defs": {"A": a, "B": obj({"k": {"const": "b", "type": "string"}})}}

    ref = {"$ref": "#/$defs/A"}
    for v, msg in (({**ref, "type": "object"}, "'type' beside '$ref'"),
                   ({**ref, "anyOf": [S]}, "'anyOf' beside '$ref'"),
                   ({"anyOf": [S], "type": "string"}, "'type' beside 'anyOf'"),
                   ({"oneOf": [S], "required": ["x"]}, "'required' beside 'oneOf'"),
                   ({"anyOf": [S], "oneOf": [S]}, "'oneOf' beside 'anyOf'"),
                   ({"allOf": [ref], "nullable": True}, "'nullable' beside 'allOf'")):
        for mode in MODES:
            with pytest.raises((GrammarError, ToolSchemaError), match=re.escape(f"{msg} at #/properties/v")):
                build(mode, doc(v))
    for mode in MODES:
        with pytest.raises((GrammarError, ToolSchemaError), match=re.escape("'allOf' at #/properties/v has 2 members")):
            build(mode, doc({"allOf": [ref, {"required": ["x"]}]}))
    with pytest.raises(ToolSchemaError, match=re.escape("tool 'f': 'type' beside '$ref' at #")):
        grammar("gemma4", {"$ref": "#/$defs/A", "type": "object", "$defs": {"A": a}})
    for v in ({"$ref": "#/$defs/A", "description": "d", "title": "T"},
              {"anyOf": [{"$ref": "#/$defs/A"}, {"type": "null"}], "default": None, "title": "V"},
              {"oneOf": [{"$ref": "#/$defs/A"}, {"$ref": "#/$defs/B"}], "discriminator": {"propertyName": "k"}},
              {"allOf": [{"$ref": "#/$defs/A"}], "description": "older pydantic's"}):
        for mode in MODES:
            assert compiled_on_threads(build(mode, doc(v))) == "", (mode, v)
    for fam in ("gemma4", "qwen3_coder"):  # read as its member
        assert grammar(fam, doc({"allOf": [{"$ref": "#/$defs/A"}], "description": "d"})) == \
            grammar(fam, doc({"$ref": "#/$defs/A", "description": "d"}))
    node = obj({"name": S, "kids": {"type": "array", "items": {"$ref": "#/$defs/Node"}}})
    for schema in ({"$defs": {"Node": node}, "$ref": "#/$defs/Node"},
                   {"$defs": doc(S)["$defs"], "discriminator": {"propertyName": "k"},
                    "oneOf": [{"$ref": "#/$defs/A"}, {"$ref": "#/$defs/B"}]},
                   {**obj({"v": {"$defs": {"Z": S}, "$ref": "#/$defs/A"}}), "$defs": {"A": a}}):
        for backend in ("auto", "guidance"):
            assert compiled_on_threads(json_schema_grammar(schema, backend, backend == "guidance")) == ""


def at_cap(kind: str, more: int, mode: str = "gemma4") -> tuple[dict, str]:
    """A schema at the cap `kind` names (`more` 0), or one past it (1), and that refusal's message. JSON nests 64 levels
    in a `response_format` (`mode` auto or guidance; vLLM's cap and text), 124 in a tool."""
    b = bounds()
    if kind == "object":
        n = b.MAX_OBJECT_PROPERTIES + more
        return obj({"v": obj({f"p{i}": S for i in range(n)}, False)}), \
            f"an object at #/properties/v has {n:,} properties, more than {b.MAX_OBJECT_PROPERTIES}"
    if kind == "tuple":
        n = b.MAX_OBJECT_PROPERTIES + more
        return obj({"v": {"type": "array", "prefixItems": [S] * n, "items": False}}), \
            f"an array at #/properties/v has {n:,} tuple items, more than {b.MAX_OBJECT_PROPERTIES}"
    if kind == "properties":  # the root's 13, then 12 objects of 400 and one of 187
        last = b.MAX_PROPERTIES - 13 - 12 * b.MAX_OBJECT_PROPERTIES + more
        sizes = [b.MAX_OBJECT_PROPERTIES] * 12 + [last]
        return obj({f"g{j}": obj({f"p{j}_{i}": S for i in range(k)}, False) for j, k in enumerate(sizes)}), \
            f"{b.MAX_PROPERTIES + more:,} properties and tuple items in all, more than {b.MAX_PROPERTIES:,}"
    if kind == "values":  # an enum, and a const
        return obj({"e": {"enum": [f"v{i}" for i in range(b.MAX_VALUES - 1 + more)]}, "c": {"const": "c"}}), \
            f"{b.MAX_VALUES + more:,} enum and const values in all, more than {b.MAX_VALUES:,}"
    if kind == "alternatives":
        n = b.MAX_ALTERNATIVES + more
        return obj({"v": {"anyOf": [obj({f"k{i}": S}) for i in range(n)]}}), \
            f"{n:,} 'anyOf' and 'oneOf' alternatives in all, more than {b.MAX_ALTERNATIVES:,}"
    if kind == "value-items":  # 100 enum values of 100 items, and one more item
        vals = [list(range(100)) for _ in range(b.MAX_VALUE_ITEMS // 100)]
        vals[0] = vals[0] + [0] * more
        return obj({"v": {"enum": vals}}), \
            f"{b.MAX_VALUE_ITEMS + more:,} array items and object members in enum and const values in all, more " \
            f"than {b.MAX_VALUE_ITEMS:,}"
    if kind == "json":  # a const of nested lists: the root, `properties` and `v` are the first three levels
        rf = mode in ("auto", "guidance")
        c: list = [1]
        for _ in range((b.MAX_JSON_NESTING if rf else b.JSON_DEPTH) - 4 + more):
            c = [c]
        return obj({"v": {"const": c}}), (
            f"The JSON schema is nested too deeply: objects and arrays may nest at most {b.MAX_JSON_NESTING} levels"
            if rf else f"nests more than {b.JSON_DEPTH} JSON levels deep, the most llguidance parses")
    if kind == "definitions":
        n = b.MAX_DEFINITIONS + more
        return {**obj({"v": {"$ref": "#/$defs/D0"}}), "$defs": {f"D{i}": S for i in range(n)}}, \
            f"{n:,} definitions, more than {b.MAX_DEFINITIONS:,}"
    if kind == "pattern-text":  # 16 patterns of 1 KB (classes alone: they fix nothing), and one byte more
        n = b.MAX_PATTERN_BYTES // 16
        pats = ["[ab]" * (n // 4)] * 15 + ["[ab]" * (n // 4 - 1) + "[ab]" + "?" * more]
        return obj({f"p{i}": {"type": "string", "pattern": p} for i, p in enumerate(pats)}), \
            f"{b.MAX_PATTERN_BYTES + more:,} bytes of 'pattern' and 'x-guidance' text in all, more than " \
            f"{b.MAX_PATTERN_BYTES:,}"
    if kind == "counted":  # optional properties under `minProperties`
        n = b.MAX_COUNTED + more
        return obj({"v": obj({f"p{i}": S for i in range(n)}, False, additionalProperties=False, minProperties=1)}), \
            f"an object at #/properties/v has {n:,} optional properties under 'minProperties', more than " \
            f"{b.MAX_COUNTED}"
    if kind == "counted-members":  # 200 copies of 200 members: 199 optional properties and 199 required
        k, r = 199 + more, 199
        v = obj({f"o{i}": S for i in range(k)} | {f"r{i}": S for i in range(r)}, False, additionalProperties=False,
                maxProperties=r + 1) | {"required": [f"r{i}" for i in range(r)]}
        return obj({"v": v}), f"{(k + 1) * (r + 1):,} members in the copies of its objects under 'minProperties' or " \
                              f"'maxProperties' in all, more than {b.MAX_COUNTED_MEMBERS:,}"
    if kind == "formats":
        n = b.MAX_FORMATS + more
        groups = [n // 2, n - n // 2]
        return obj({f"g{j}": obj({f"d{j}_{i}": {"type": "string", "format": "date"} for i in range(k)})
                    for j, k in enumerate(groups)}), \
            f"{n:,} strings with a 'format' in all, more than {b.MAX_FORMATS:,}"
    assert kind == "nesting"  # arrays of arrays: the root, `v`, then one level an array
    s: dict = S
    for _ in range(b.MAX_NESTING - 2 + more):
        s = {"type": "array", "items": s}
    return obj({"v": s}), f"nests {b.MAX_NESTING + more} levels deep, counting its $refs' definitions where they are " \
                          f"used, more than {b.MAX_NESTING}"


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("kind", ["object", "tuple", "properties", "values", "value-items", "alternatives",
                                  "definitions", "nesting", "json", "pattern-text", "formats", "counted",
                                  "counted-members"])
def test_each_cap_takes_its_number_and_refuses_one_more(kind, mode) -> None:
    """At each cap the schema builds and llguidance compiles it; one more is a 400 that names the cap and its number.
    Gemma's grammar never compiles a format (its tools take one without enforcing it): one more builds there."""
    if kind in ("tuple", "counted", "counted-members") and mode not in ("auto", "guidance"):
        pytest.skip("the tools' keyword list never took prefixItems, minProperties or maxProperties")
    schema, _ = at_cap(kind, 0, mode)
    assert compiled_on_threads(build(mode, schema)) == ""
    schema, msg = at_cap(kind, 1, mode)
    if kind == "formats" and mode == "gemma4":
        assert compiled_on_threads(build(mode, schema)) == ""
        return
    with pytest.raises((GrammarError, ToolSchemaError), match=re.escape(msg)) as err:
        build(mode, schema)
    assert isinstance(err.value, NestingError) == (kind == "json" and mode in ("auto", "guidance"))


@pytest.mark.parametrize("fam", ["gemma4", "qwen3_coder"])
def test_the_caps_count_every_tool_in_the_grammar(fam) -> None:
    """Two tools of 2,500 properties each build; one property more is a 400 for the grammar's schemas in all. A named
    tool's grammar holds that tool alone, and counts it alone."""
    def big(n: int, name: str) -> dict:
        sizes = [250] * (n // 250) + ([n % 250] if n % 250 else [])
        return tool(obj({f"g{j}": obj({f"{name}{j}_{i}": S for i in range(k)}, False) for j, k in enumerate(sizes)},
                        False), name)

    def calls(tools: list[dict], choice: str = "required", name: str | None = None) -> str:
        return call_grammar(fam, tools, choice, name, ids=IDS[fam])[0]

    a, b = big(2500 - 10, "a"), big(2500 - 10, "b")  # each object's 10 groups count too
    assert calls([a, b])
    with pytest.raises(ToolSchemaError, match=re.escape("the tools' schemas have 5,001 properties")):
        calls([a, big(2500 - 9, "b")])
    assert calls([a, big(4000, "b")], "named", "a")


@pytest.mark.parametrize("fam", ["gemma4", "qwen3_coder"])
def test_a_tool_s_nested_definitions_and_ids_are_checked(fam) -> None:
    """A tool's nested `$defs`, which Qwen's `%json` items carry, is checked as definitions no `$ref` reaches and
    counted against the definitions' cap; an `$id` is the root's alone; the JSON depth is capped as a
    `response_format`'s. Each is a 400 before anything is built."""
    def nested(defs: dict) -> dict:
        return {**obj({"p": {"$ref": "#/$defs/A", "$defs": defs}}), "$defs": {"A": {"type": "integer"}}}

    assert compiled_on_threads(grammar(fam, nested({"Z": {"type": "string"}}))) == ""
    with pytest.raises(ToolSchemaError, match=re.escape("the keyword 'if' at #/properties/p/$defs/Z is not supported")):
        grammar(fam, nested({"Z": {"if": {}, "then": {}}}))
    with pytest.raises(ToolSchemaError, match="1,002 definitions, more than 1,000"):
        grammar(fam, nested({f"Z{i}": {"type": "string"} for i in range(1001)}))
    with pytest.raises(ToolSchemaError, match=re.escape("the '$id' at #/properties/p is not supported")):
        grammar(fam, obj({"p": {"$id": "https://example.com/p", "type": "integer"}}))
    assert grammar(fam, {"$id": "https://example.com/root", **obj({"p": {"type": "integer"}})})
    c: list = [1]
    for _ in range(130):
        c = [c]
    with pytest.raises(ToolSchemaError, match="nests more than 124 JSON levels deep"):
        grammar(fam, obj({"p": {"const": c}}))


@pytest.mark.parametrize("fam", ["gemma4", "qwen3_coder"])
def test_definitions_beside_an_all_of_and_in_its_member_are_checked(fam) -> None:
    """A one-member `allOf` reads as its member, but the `$defs` beside it are not dropped: both are checked and
    counted (the sibling's were once overwritten by the member's, so 3,000 hidden definitions passed). A nested
    `$defs` that is not an object is a 400, as the root's is."""
    def hidden(defs: object) -> dict:
        return obj({"p": {"$defs": defs, "allOf": [{"$defs": {}, "type": "string"}]}})

    assert compiled_on_threads(grammar(fam, hidden({"Z": {"type": "integer"}}))) == ""
    with pytest.raises(ToolSchemaError, match=re.escape("the keyword 'bogus' at #/properties/p/$defs/Z")):
        grammar(fam, hidden({"Z": {"bogus": 1}}))
    with pytest.raises(ToolSchemaError, match="1,001 definitions, more than 1,000"):
        grammar(fam, hidden({f"Z{i}": {"type": "string"} for i in range(1001)}))
    with pytest.raises(ToolSchemaError, match="1,001 definitions, more than 1,000"):  # the member's count too
        grammar(fam, obj({"p": {"$defs": {"Y": S}, "allOf": [{"$defs": {f"Z{i}": S for i in range(1000)},
                                                               "type": "string"}]}}))
    for bad in ([1], "x", None):
        with pytest.raises(ToolSchemaError, match=re.escape("'$defs' at #/properties/p must be an object")):
            grammar(fam, obj({"p": {"type": "string", "$defs": bad}}))


@pytest.mark.parametrize("mode", MODES)
def test_a_ref_is_decoded_whole_before_it_is_split(mode) -> None:
    """llguidance percent-decodes a `$ref`'s whole pointer before it splits it, so `#/$defs/A%2Fproperties%2Fq` names
    a schema inside A, which the nesting cap never saw (1,000 such links once matched documents 4,001 objects
    deep). A name that decodes to a pointer is a 400; a name with other escapes resolves as llguidance resolves it."""
    a = obj({"q": {"const": "SUB"}})
    with pytest.raises((GrammarError, ToolSchemaError), match="#/\\$defs/A%2Fproperties%2Fq"):
        build(mode, {"$defs": {"A": a}, **obj({"x": {"$ref": "#/$defs/A%2Fproperties%2Fq"}})})
    assert compiled_on_threads(build(mode, {"$defs": {"a b": a}, **obj({"x": {"$ref": "#/$defs/a%20b"}})})) == ""

@pytest.mark.parametrize("fam", ["gemma4", "qwen3_coder"])
def test_a_decoded_ref_types_its_argument(fam) -> None:
    """The grammar reads a tool's `$ref` name decoded, as llguidance does (`%20`, then `~1`), and so does the typing of
    the call's arguments: the integer comes back an integer, in both families (once the string "5").
    A name that resolved only undecoded stays a 400 (`#/$defs/a%20b` naming `a%20b`, `#/$defs/a/b` naming `a/b`)."""
    from llm_mini_infer.output.tool_args import fix_types, gemma4_arguments, gemma4_parse, qwen_arguments

    for key, ref in (("a b", "#/$defs/a%20b"), ("a/b", "#/$defs/a~1b"), ("a%b", "#/$defs/a%25b"), ("A", "#/$defs/A"),
                     ("a b", "#/definitions/a%20b")):
        params = {**obj({"v": {"$ref": ref}}), ("definitions" if "definitions" in ref else "$defs"):
                  {key: {"type": "integer"}}}
        assert compiled_on_threads(grammar(fam, params)) == ""
        if fam == "gemma4":
            got = fix_types(gemma4_arguments("v:5}"), params, gemma4_parse("v:5}"))
        else:
            got = fix_types(qwen_arguments("<parameter=v>\n5\n</parameter>\n"), params)
        assert json.loads(got) == {"v": 5}, (key, ref, got)
    for key, ref in (("a%20b", "#/$defs/a%20b"), ("a/b", "#/$defs/a/b")):
        with pytest.raises(ToolSchemaError, match=re.escape(f"the $ref {ref!r} at #/properties/v does not resolve")):
            grammar(fam, {**obj({"v": {"$ref": ref}}), "$defs": {key: {"type": "integer"}}})


M = -(1 << 63)
OUT_OF_RANGE = [  # outside the supported integer range: an integer's upper bound at the bottom of the 64-bit range, as
    # a double reads it, and such a number in a value that may compile as an integer
    ({"type": "integer", "minimum": M + 512, "maximum": M + 512}, "'maximum'"),
    ({"type": "integer", "minimum": -1e300, "maximum": -1e20}, "'maximum'"),
    ({"type": "integer", "exclusiveMinimum": M - 5000, "exclusiveMaximum": M + 512}, "'exclusiveMaximum'"),
    ({"type": "integer", "minimum": M, "maximum": M, "multipleOf": 2}, "'maximum'"),
    ({"const": M + 512}, "the number"), ({"enum": [[M]]}, "the number"),
    ({"type": "integer", "const": -1e30}, "the number"), ({"type": "integer", "enum": [-1e30]}, "the number"),
    ({"type": "array", "items": {"type": "integer"}, "const": [-1e30]}, "the number"),
    ({"type": "object", "properties": {"a": {"type": "integer"}}, "const": {"a": M - 5}}, "the number"),
    ({"type": ["integer", "string"], "const": -1e30}, "the number"),
    ({"type": "integer", "minimum": -9.223372036854776e18, "maximum": -9.223372036854775e18}, "'maximum'")]
IN_RANGE = [  # what builds, or is llguidance's own 400: a lower bound, a number, a float alone, 513 above
    {"type": "integer", "minimum": M + 513, "maximum": M + 513}, {"type": "integer", "maximum": M},
    {"type": "integer", "minimum": M}, {"type": "integer", "minimum": M, "maximum": M + 5000},
    {"type": "number", "minimum": M, "maximum": M}, {"minimum": M, "maximum": M}, {"const": M + 513},
    {"const": -1e30}, {"const": M - 1}, {"type": "integer", "minimum": M + 600, "maximum": M + 600},
    {"type": "integer", "minimum": M - 10 ** 6, "maximum": float(M + 2048)}]


@pytest.mark.parametrize("mode", MODES)
def test_an_integer_outside_the_supported_range_is_refused(mode) -> None:
    """Each shape outside the supported integer range is a 400 that says so, in a `response_format` and in Qwen's
    tools, whose `%json` items carry bounds and values to llguidance. Gemma's grammar writes values and integers as its
    own text and does not enforce bounds: its call grammar builds and compiles. What is in range builds (or is
    llguidance's own 400)."""
    for schema, what in OUT_OF_RANGE:
        doc = obj({"a": schema})
        if mode == "gemma4":
            got = child("from llm_mini_infer.grammar.tools import call_grammar\n"
                        "from tests.support.grammar_cases import IDS, tool\n"
                        f"text = call_grammar('gemma4', [tool({doc!r})], 'required', ids=IDS['gemma4'])[0]\n"
                        "from tests.support.tool_grammar_cases import check\nprint(repr(check(text)))", 60)
            assert got.returncode == 0 and got.stdout.strip() == "''", (schema, got.returncode, got.stderr[-500:])
            continue
        with pytest.raises((GrammarError, ToolSchemaError),
                           match=re.escape(what) + ".*outside the supported integer range"):
            build(mode, doc)
    if mode != "gemma4":
        for schema in IN_RANGE:
            if schema.get("type") == "integer" and schema.get("maximum") == M:  # llguidance's own 400, the guard's
                with pytest.raises((GrammarError, ToolSchemaError), match="an integer upper bound outside the"):
                    build(mode, obj({"a": schema}))
            else:
                build(mode, obj({"a": schema}))


def test_the_property_count_caps_take_any_closed_object() -> None:
    """llguidance copies an object under `minProperties` whenever no other key is allowed, by `false` or by a schema
    without a value (`{"enum": []}`): the caps count both (100 optional properties beside such a schema once
    built, and the first mask failed). A format beside `enum` values is checked once a value: each counts."""
    b = bounds()
    for addl in (False, {"enum": []}, {"anyOf": []}):
        schema = obj({f"p{i}": S for i in range(b.MAX_COUNTED + 1)}, False, additionalProperties=addl,
                     minProperties=1)
        with pytest.raises(GrammarError, match="optional properties under 'minProperties'"):
            json_schema_grammar(schema, "guidance", True)
    uris = {"type": "string", "format": "uri", "enum": [f"https://example.com/{i}" for i in range(b.MAX_FORMATS + 1)]}
    with pytest.raises(GrammarError, match="strings with a 'format'"):
        json_schema_grammar(obj({"u": uris}), "guidance", True)


def test_pydantic_s_and_the_openai_sdk_s_schemas_pass() -> None:
    """pydantic v2's schemas of the field types clients use (optional fields, nested models, enums, `Literal`, `list`,
    `dict`, unions, discriminated unions, constrained types, recursive models), and the OpenAI SDK's strict form of
    them: llguidance compiles each, as a `response_format` (both backends) and as a tool's parameters (both families).
    A `tuple` (`prefixItems`) passes as a `response_format`; the tools' keyword list does not take it
    (`ToolSchemaError`)."""
    import enum
    from typing import Annotated, Literal

    from openai.lib._pydantic import to_strict_json_schema
    from pydantic import BaseModel, Field, conint, constr

    class Color(enum.Enum):
        red = "red"
        green = "green"

    class Address(BaseModel):
        street: str
        city: str = Field(description="the city")

    class Cat(BaseModel):
        kind: Literal["cat"]
        lives: int

    class Dog(BaseModel):
        kind: Literal["dog"]
        barks: bool

    class Node(BaseModel):
        name: str
        children: list[Node] = []

    class Order(BaseModel):
        note: str | None = None
        address: Address
        color: Color
        size: Literal["s", "m", "l"]
        tags: list[str]
        counts: dict[str, int]
        id: int | str
        pet: Annotated[Cat | Dog, Field(discriminator="kind")]
        stars: conint(ge=0, le=5)
        code: constr(min_length=1, max_length=5, pattern="^[a-z]+$")
        billing: Address | None = None
        tree: Node

    class Point(BaseModel):
        xy: tuple[int, int]

    Node.model_rebuild()
    for model in (Order, Node, Point):
        for schema in (model.model_json_schema(), to_strict_json_schema(model)):
            for mode in MODES:
                if model is Point and mode not in ("auto", "guidance"):
                    with pytest.raises(ToolSchemaError, match="'prefixItems'"):
                        build(mode, schema)
                    continue
                assert compiled_on_threads(build(mode, schema)) == "", (model.__name__, mode)


# -- the costs at the caps ------------------------------------------------------------------------------------------

def nest_wide(depth: int, per: int, required: bool = True, facets: bool = False) -> dict:
    """`depth` objects nested through their last property, each with `per` properties (strings, or with a pattern, a
    format, to the cap on formats, or bounds each)."""
    kinds = [{"type": "string", "pattern": "^[a-z]{3,40}$"}, {"type": "string", "format": "date-time"},
             {"type": "integer", "minimum": -10**6, "maximum": 10**6}, {"type": "number", "multipleOf": 0.5},
             {"type": "string", "minLength": 5, "maxLength": 300}]
    formats = bounds().MAX_FORMATS if facets else 0
    s: dict = S

    def leaf(i: int) -> dict:
        nonlocal formats
        k = kinds[i % len(kinds)] if facets else S
        formats -= "format" in k
        return dict(k if formats >= 0 or "format" not in k else S)

    for _ in range(depth):
        props = {f"p{i}": leaf(i) for i in range(per - 1)}
        props["next"] = s
        s = obj(props, required)
    return s


def compile_once(mode: str, schema: dict) -> None:
    assert compiled_on_threads(build(mode, schema)) == ""


# Alone on CI's runner (2 cores, 4 vCPUs, two workers): beside a second worker on an earlier 1-core runner, the
# facets took 1.5 to 1.7 s
@pytest.mark.serial
def test_the_costliest_shapes_at_the_caps_compile_at_once() -> None:
    """The costliest shapes at the caps: 12 nested objects of 400 optional properties
    (Gemma's longest rules), the same with a facet on each property (formats to their cap), and an `anyOf` of 999
    object definitions. Their build and check take well under a second."""
    wide = {**obj({"v": {"anyOf": [{"$ref": f"#/$defs/A{i}"} for i in range(999)]}}),
            "$defs": {f"A{i}": obj({f"a{i}_{j}": S for j in range(5)}) for i in range(999)}}
    for name, schema, modes in (("optional", nest_wide(12, 400, False), MODES),
                                ("facets", nest_wide(12, 400, facets=True), ["auto", "guidance", "qwen3_coder"]),
                                ("wide", wide, MODES)):
        for mode in modes:
            assert under(functools.partial(compile_once, mode, schema), 1.5) < 1.5, (name, mode)


def test_the_deepest_call_grammar_at_the_caps_needs_a_fraction_of_the_stack() -> None:
    """Gemma's deepest call grammars at the caps (12 nested objects of 400 optional properties, or 31 of 161) compile on
    a thread with a quarter of the compile threads' stack."""
    from llm_mini_infer.grammar.threads import STACK_BYTES, CompileThreads

    assert STACK_BYTES == 32 << 20 and CompileThreads(1, "test").stack == STACK_BYTES
    for schema in (nest_wide(12, 400, False), nest_wide(31, 161, False)):
        assert compiled_on_threads(grammar("gemma4", schema), stack=STACK_BYTES // 4) == ""


def annotated_alternatives(n: int, desc: int) -> dict:
    """An `anyOf` of `n` refs beside a description of `desc` characters, which Qwen's `%json` walk carries into each
    alternative's item."""
    p = {"description": "d" * desc, "anyOf": [{"$ref": f"#/$defs/A{i}"} for i in range(n)]}
    return {**obj({"p": p}), "$defs": {f"A{i}": obj({"a": {"type": "integer"}}) for i in range(n)}}


def test_qwen_estimates_an_items_size_before_it_copies_an_annotation() -> None:
    """999 refs beside a long description would make an item of 999 copies of it: the size is estimated before
    anything is copied and refused, and the walk's keys dump each annotation once (dumping it per `$ref` took
    1.5 s and 0.8 GB at 800 KB). Twice the description takes about twice the time, and the allocations' peak stays
    within a few times the grammar's bound."""
    def refused(desc: int) -> None:
        with pytest.raises(ToolSchemaError, match="bytes"):
            grammar("qwen3_coder", annotated_alternatives(999, desc))

    small, big = least_each(lambda: refused(200_000), lambda: refused(400_000))
    assert big < 3 * small + 0.02 and big < 0.5, (small, big)
    tracemalloc.start()
    try:
        refused(400_000)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 4 * bound([annotated_alternatives(999, 400_000)]), peak


def test_qwen_s_walk_keys_annotations_by_their_content() -> None:
    """Two `$ref`s to one definition with equal list-valued annotations are one alternative in Qwen's `%json` item: keys
    that compared the lists by identity made two."""
    d = obj({"k": {"type": "integer"}})
    for extra in ({"default": [1]}, {"examples": [{"a": 1}]}):
        alts = [{"$ref": "#/$defs/D", **json.loads(json.dumps(extra))} for _ in range(2)]
        text = grammar("qwen3_coder", {**obj({"a": {"anyOf": alts}}), "$defs": {"D": d}})
        item = text.split("%json ", 1)[1].split("\n", 1)[0]
        assert '"anyOf"' not in item, item


def test_a_grammar_past_llguidance_s_symbols_is_refused_by_the_check() -> None:
    """The frontend's check compiles a matcher over a small tokenizer, so it sees llguidance's parser limit, which
    `validate_grammar` does not (the engine refused such a grammar with an error finish): a grammar of 70,000
    alternatives gets the engine's message, one of 20,000 compiles."""
    def alternatives(n: int) -> str:
        return "start: " + " | ".join(f'"v{i}"' for i in range(n)) + "\n"

    assert compiled_on_threads(alternatives(70_000)) == "the grammar needs more than 65,524 symbols, llguidance's limit"
    assert compiled_on_threads(alternatives(20_000)) == ""


# -- the engine -----------------------------------------------------------------------------------------------------


DEEP_COMPILE = f"""
from llm_mini_infer.grammar.engine import Grammars, LLGuidance
from tests.support.grammar_cases import lark_chain
text = lark_chain(5000)
backend = LLGuidance({str(QWEN / "tokenizer.json")!r}, 248320, [248046])
"""


def test_a_deep_grammar_compiles_on_the_compile_thread() -> None:
    """A grammar 5,000 rules deep compiles through the engine's `Grammars` and gives its forced prefix, in a child whose
    main thread waits on the compile thread."""
    if not env_has_model(QWEN):
        pytest.skip("the Qwen3.8 fixture is absent")
    r = child(DEEP_COMPILE + """
from llm_mini_infer.grammar.threads import CompileThreads
g = Grammars(backend, threads=CompileThreads(1, "grammar-compile"))
g.compile(text)
print("forced", len(g.forced_prefix(text)))
""")
    assert r.returncode == 0 and "forced" in r.stdout, r.stderr[-2000:]


def test_a_cached_grammar_error_is_raised_fresh() -> None:
    """The engine's and the frontend's caches keep a refused grammar's error: each hit raises a new one, so its
    traceback does not grow with the hits (by about 2 frames a hit), and the cached error holds no frame of the
    compile that made it."""
    from llm_mini_infer.grammar.engine import Grammars

    from ..support.fake_grammar import FakeGrammar

    def raised(f) -> GrammarError:
        with pytest.raises(GrammarError) as got:
            f()
        return got.value

    g = Grammars(FakeGrammar(64, [1]))
    specs = GrammarSpecs()
    for f, cached in ((lambda: g.compile("not a grammar"), lambda: g._cache["not a grammar"]),
                      (lambda: specs.json_schema({"type": "strin"}), lambda: next(iter(specs._cache.values())))):
        errors = [raised(f) for _ in range(3)]
        assert errors[0] is not errors[1] and str(errors[0]) == str(errors[2])
        depths = [len(traceback.extract_tb(e.__traceback__)) for e in errors]
        assert depths[0] == depths[2], depths
        assert cached().__traceback__ is None


# -- the typing -----------------------------------------------------------------------------------------------------

def cycle_entered_everywhere(n: int) -> tuple[str, dict]:
    defs = {f"D{i}": {"$ref": f"#/$defs/D{(i + 1) % n}"} for i in range(n)}
    params = {"type": "object", "properties": {f"p{i}": {"$ref": f"#/$defs/D{i}"} for i in range(n)}, "$defs": defs}
    return json.dumps({f"p{i}": "1" for i in range(n)}), params


def test_a_ref_cycle_entered_at_every_definition_types_in_linear_time() -> None:
    """A cycle of `$ref`-only definitions, a property into each (quadratic once, 12 s at 4,000): twice the cycle
    (8,000, beside 4,000: about 20 ms) takes about twice the time, with no slack added."""
    a, b = cycle_entered_everywhere(4000), cycle_entered_everywhere(8000)
    small, big = least_each(lambda: fix_types(*a), lambda: fix_types(*b))
    assert big < 3 * small and big < 1, (small, big)
    assert json.loads(fix_types(*a)) == json.loads(a[0])  # a cycle of refs alone types as a string


def typed_cycle(n: int) -> tuple[str, dict]:
    """A cycle of `$ref`s, each beside its own `type` and `enum`, a property into each."""
    defs = {f"D{i}": {"$ref": f"#/$defs/D{(i + 1) % n}", "type": "integer" if i % 3 else "number", "enum": [1, 2]}
            for i in range(n)}
    props = {f"p{i}": {"$ref": f"#/$defs/D{i}"} for i in range(n)}
    return json.dumps({k: "1e3" for k in props}), {"type": "object", "properties": props, "$defs": defs}


def shared_alternatives(n: int) -> tuple[str, dict]:
    """One `anyOf` of `n` refs, which `n` properties share."""
    defs = {f"T{i}": {"type": "integer"} for i in range(n)}
    defs["W"] = {"anyOf": [{"$ref": f"#/$defs/T{i}"} for i in range(n)]}
    props = {f"p{i}": {"$ref": "#/$defs/W"} for i in range(n)}
    return json.dumps({k: "1e3" for k in props}), {"type": "object", "properties": props, "$defs": defs}


@pytest.mark.parametrize("shape", [typed_cycle, shared_alternatives], ids=["typed-cycle", "shared-alternatives"])
def test_the_typing_is_linear_where_no_grammar_bounds_it(shape) -> None:
    """Under `auto`, where no grammar is built and no cap applies, the call's arguments are still typed: a typed `$ref`
    cycle and one wide `anyOf` that many properties share take linear time (they once took 61 s and 47 s at 8,000)."""
    a, b = shape(2000), shape(8000)
    small, big = least_each(lambda: fix_types(*a), lambda: fix_types(*b), runs=3)
    assert big < 8 * small + 0.05 and big < 1, (small, big)


def test_the_typing_does_not_depend_on_key_order() -> None:
    """Three definitions in a cycle through `anyOf` (A: B or a number; B: C or null; C: A). Typed with `c` first, `b`'s
    types were memoized while A was being typed, without A's number, so "1e3" stayed a string; with `a` first it was
    1000. Each order now gives every value the cycle's types."""
    ref = lambda k: {"$ref": f"#/$defs/{k}"}  # noqa: E731
    defs = {"A": {"anyOf": [ref("B"), {"type": "number"}]}, "B": {"anyOf": [ref("C"), {"type": "null"}]},
            "C": {"anyOf": [ref("A")]}}
    params = {"type": "object", "properties": {k: ref(k.upper()) for k in "abc"}, "$defs": defs}
    forward = json.loads(fix_types(json.dumps({k: "1e3" for k in "abc"}), params))
    reverse = json.loads(fix_types(json.dumps({k: "1e3" for k in "cba"}), params))
    assert forward == reverse == {"a": 1000, "b": 1000, "c": 1000}


# -- the compile threads --------------------------------------------------------------------------------------------

def test_the_threads_keep_other_threads_stacks() -> None:
    """The pool's threads take its stack size; `threading.stack_size` is as it was for every thread after."""
    from llm_mini_infer.grammar.threads import CompileThreads

    before = threading.stack_size()
    threads = CompileThreads(2, "test")
    try:
        assert threading.stack_size() == before and len(threads.threads) == 2
        assert threads.run(lambda: threading.current_thread().name).startswith("test-")
    finally:
        threads.close()


def test_the_threads_refuse_work_once_closed() -> None:
    """A job after `close` would wait for a thread that has ended: it raises instead."""
    from llm_mini_infer.grammar.threads import CompileThreads

    threads = CompileThreads(1, "test")
    threads.close()
    got: list[str] = []

    def run() -> None:
        try:
            threads.run(lambda: 1)
            got.append("ran")
        except RuntimeError as exc:
            got.append(str(exc))

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(2)
    assert got == ["the grammar compile threads are closed"]


def test_a_refused_start_says_why(monkeypatch) -> None:
    """Each thread reserves its stack as address space: where a limit refuses it (`RLIMIT_AS`, or
    `vm.overcommit_memory=2`), the pool's start fails with the limit's name, not Python's "can't start new thread"."""
    from llm_mini_infer.grammar.threads import CompileThreads

    def refuse(self) -> None:
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(threading.Thread, "start", refuse)
    with pytest.raises(RuntimeError, match="address-space limit"):
        CompileThreads(2, "test")


@pytest.mark.skipif(sys.platform != "linux", reason="RLIMIT_AS is enforced on Linux")
def test_a_start_under_rlimit_as_says_why() -> None:
    """Under an address-space limit 300 MiB above what the process uses, two threads of 256 MiB cannot start."""
    r = child("""
import resource
used = int(open("/proc/self/status").read().split("VmSize:")[1].split()[0]) << 10
resource.setrlimit(resource.RLIMIT_AS, (used + (300 << 20), resource.RLIM_INFINITY))
from llm_mini_infer.grammar.threads import CompileThreads
try:
    CompileThreads(2, "test", 256 << 20)
    print("started")
except RuntimeError as exc:
    print("refused:", exc)
""")
    assert "refused:" in r.stdout and "address-space limit" in r.stdout, (r.stdout, r.stderr[-1000:])


def test_a_matcher_compiled_on_a_compile_thread_is_used_anywhere() -> None:
    """The same schema compiled on the engine's compile thread and on this thread: along a document, the masks
    llguidance's mask threads fill and the single-token checks on this thread are the same for both."""
    import numpy as np

    from llm_mini_infer.grammar.engine import Grammars, LLGuidance
    from llm_mini_infer.grammar.threads import CompileThreads

    if not env_has_model(QWEN):
        pytest.skip("the Qwen3.8 fixture is absent")
    from tokenizers import Tokenizer

    backend = LLGuidance(QWEN / "tokenizer.json", 248320, [248046], threads=4)
    schema = {"type": "object", "properties": {"title": {"type": "string"}, "year": {"type": "integer"},
                                               "tags": {"type": "array", "items": {"enum": ["sf", "classic"]}}},
              "required": ["title", "year", "tags"], "additionalProperties": False}
    text = json_schema_grammar(schema, "guidance", disable_any_whitespace=True)
    threads = CompileThreads(1, "test")
    try:
        on_thread = Grammars(backend, threads=threads)
        here = Grammars(backend)
        a, b = on_thread.matcher(on_thread.compile(text)), here.matcher(here.compile(text))
        doc = json.dumps({"title": "Dune", "year": 1965, "tags": ["sf", "classic"]}, separators=(",", ":"))
        ids = Tokenizer.from_file(str(QWEN / "tokenizer.json")).encode(doc, add_special_tokens=False).ids
        words = (248320 + 31) // 32
        for i, t in enumerate(ids[:-2]):
            buf = np.zeros((8, words), np.int32)  # a row each, then 1 + 2 rows each along two drafts
            backend.fill([(a, 0), (b, 1)], [(a, 2, ids[i + 1:i + 3]), (b, 5, ids[i + 1:i + 3])], buf)
            assert (buf[0] == buf[1]).all() and (buf[2:5] == buf[5:8]).all(), i
            assert backend.only(a, t) == backend.only(b, t)
            assert a.consume_tokens([t]) and b.consume_tokens([t]), i
    finally:
        threads.close()


class CountingThreads:
    """The compile threads' interface, run on this thread, counting what it is given."""

    def __init__(self) -> None:
        self.submitted = 0

    def submit(self, fn):
        from concurrent.futures import Future

        self.submitted += 1
        fut: Future = Future()
        fut.set_result(fn())
        return fut


def test_a_hit_stays_on_the_callers_thread_and_a_key_builds_once() -> None:
    """A miss raises `Pending`; two requests that miss one key share one build; the lookup after it, and every one
    after, returns the text at once, with nothing submitted."""
    from llm_mini_infer.grammar.spec import Pending

    threads = CountingThreads()
    specs = GrammarSpecs(threads=threads)
    raw = json.dumps({"type": "object", "properties": {"a": {"type": "integer"}}}).encode()

    async def two() -> list:
        got = []
        for _ in range(2):
            try:
                specs.json_schema(raw)
            except Pending as p:
                got.append(p)
        return await asyncio.gather(*(specs.resolve(p) for p in got))

    texts = asyncio.run(two())
    assert threads.submitted == 1 and texts[0] == texts[1] and texts[0].startswith("%llguidance")
    assert specs.json_schema(raw) == texts[0] and threads.submitted == 1


def misses(specs: GrammarSpecs, key: bytes, build, n: int = 2) -> list:
    """`n` lookups of a tool-call grammar that miss, as `n` requests' would (their `Pending`s)."""
    from llm_mini_infer.grammar.spec import Pending

    got = []
    for _ in range(n):
        with pytest.raises(Pending) as p:
            specs.tool_calls(key, build)
        got.append(p.value)
    return got


def test_a_cancelled_waiter_leaves_its_build_running() -> None:
    """Two requests wait for one build on a real compile thread, held at a gate; the first is cancelled (its client
    left). The build goes on: the other gets the text, the cache holds it, and no build is left in flight."""
    from llm_mini_infer.grammar.threads import CompileThreads

    threads = CompileThreads(1, "test")
    specs = GrammarSpecs(threads=threads)
    gate = threading.Event()

    def held() -> str:
        gate.wait(10)
        return 'start: "a"\n'

    async def main() -> None:
        first, second = (asyncio.ensure_future(specs.resolve(p)) for p in misses(specs, b"k", held))
        await asyncio.sleep(0.05)
        assert len(specs._building) == 1
        first.cancel()
        await asyncio.sleep(0.05)
        gate.set()
        assert await second == 'start: "a"\n'
        with pytest.raises(asyncio.CancelledError):
            await first
        assert not specs._building and specs.tool_calls(b"k", held) == 'start: "a"\n'

    try:
        asyncio.run(main())
    finally:
        gate.set()
        threads.close()


def test_a_failed_build_reaches_every_waiter() -> None:
    """A build that raises (not a client's error): both requests waiting for it get its exception, nothing is cached,
    no build is left in flight, and the next lookup builds again."""
    from llm_mini_infer.grammar.spec import Pending
    from llm_mini_infer.grammar.threads import CompileThreads

    threads = CompileThreads(1, "test")
    specs = GrammarSpecs(threads=threads)
    gate = threading.Event()

    def fails() -> str:
        gate.wait(10)
        raise RuntimeError("forced build failure")

    async def main() -> None:
        waiters = [asyncio.ensure_future(specs.resolve(p)) for p in misses(specs, b"k", fails)]
        await asyncio.sleep(0.05)
        gate.set()
        for w in waiters:
            with pytest.raises(RuntimeError, match="forced build failure"):
                await w
        assert not specs._building and not specs._cache
        with pytest.raises(Pending):
            specs.tool_calls(b"k", fails)

    try:
        asyncio.run(main())
    finally:
        gate.set()
        threads.close()


# -- through the server ---------------------------------------------------------------------------------------------

LONG = {"model": {"max_model_len": 262_144}}  # the tools render into the prompt


def gemma_server(tmp, step_ms: float = 0.2, env: dict | None = None) -> Server:
    cfg = {"runner": {"kind": "fake"}, "images": {"enabled": False}, **LONG,
           "structured_output": {"backend": "guidance", "disable_any_whitespace": True}}
    return Server(GEMMA, tmp, step_ms=step_ms, extra=cfg, env=env).start(120)


@pytest.fixture(scope="module", params=["qwen", "gemma"])
def server(request, tmp_path_factory):
    model = QWEN if request.param == "qwen" else GEMMA
    if not env_has_model(model):
        pytest.skip(f"{model} is absent: uv run tests/scripts/fetch_fixtures.py")
    tmp = tmp_path_factory.mktemp("srv")
    srv = gemma_server(tmp) if request.param == "gemma" else Server(QWEN, tmp, step_ms=0.2, extra=LONG).start(120)
    srv.family = "gemma4" if request.param == "gemma" else "qwen3_coder"
    yield srv
    assert srv.stop() == 0, srv.log_path.read_text()[-3000:]


def ask(srv: Server, **kw) -> tuple[int, dict]:
    body = {"messages": [{"role": "user", "content": "x"}], "max_tokens": 8, "temperature": 0.0, **kw}
    if srv.family == "qwen3_coder":
        body["chat_template_kwargs"] = {"enable_thinking": False}
    return srv.post(CHAT, body)


def rf(schema: dict) -> dict:
    return {"type": "json_schema", "json_schema": {"name": "s", "schema": schema}}


def up(srv: Server) -> bool:
    return srv.proc.poll() is None and srv.get("/health")[0] == 200


def ref_chain(levels: int) -> dict:
    """Definitions chained through a property, two levels each, `levels` deep in all."""
    n = (levels - 3) // 2
    defs = {f"D{i}": obj({"x": {"$ref": f"#/$defs/D{i + 1}"}}) for i in range(n)}
    defs[f"D{n}"] = S
    return {**obj({"v": {"$ref": "#/$defs/D0"}}), "$defs": defs}


SHAPES = {  # each a costly shape scaled to what builds at once, or the deepest and widest at the caps
    "ref-beside-anyOf": (lambda: ref_beside_anyof(12), 400),
    "patternProperties": (lambda: pattern_properties(100), 400),
    "dag": (lambda: ref_dag(1100, 2), 400),
    "merged": (lambda: merged(5, 2, 50), 400),
    "enum": (lambda: int_enum(1001), 400),
    "chain-32": (lambda: ref_chain(32), 200),
    "chain-34": (lambda: ref_chain(34), 400),
    "deep-optional": (lambda: nest_wide(12, 400, False), 200),
    "object-401": (lambda: obj({f"p{i}": S for i in range(401)}, False), 400),
}


@pytest.mark.parametrize("as_tool", [False, True], ids=["response_format", "tool"])
@pytest.mark.parametrize("shape", list(SHAPES))
def test_shapes_are_served_or_refused_at_once_and_the_server_stays_up(server, shape, as_tool) -> None:
    """Through the server, as a `response_format` and as a tool's parameters under `required`: the shapes the subset
    refuses get a 400 at once, the deepest and widest it takes a 200, and the server stays up."""
    make, status = SHAPES[shape]
    if shape == "patternProperties" and as_tool:
        status = 400  # the tools' keyword list never took it
    schema = make()
    start = time.perf_counter()
    if as_tool:
        code, r = ask(server, tools=[{"type": "function", "function": {"name": "f", "parameters": schema}}],
                      tool_choice="required")
    else:
        code, r = ask(server, response_format=rf(schema))
    took = time.perf_counter() - start
    assert code == status, (code, r)
    if status == 400:
        assert r["error"]["param"] == ("tools" if as_tool else "response_format"), r
        assert took < 1, took
    assert up(server), server.log_path.read_text()[-3000:]


def stream_gaps(srv: Server, during=None) -> list[float]:
    """The gaps between a streamed request's chunks; `during`: called once the stream's first chunk arrived."""
    c = http.client.HTTPConnection("127.0.0.1", srv.port, timeout=60)
    body = {"messages": [{"role": "user", "content": "x"}], "max_tokens": 300, "temperature": 0.0, "stream": True,
            "ignore_eos": True}
    c.request("POST", CHAT, body=json.dumps(body), headers={"content-type": "application/json"})
    r = c.getresponse()
    assert r.status == 200
    times, worker = [], None
    while True:
        line = r.readline()
        if not line:
            break
        if line.startswith(b"data: "):
            times.append(time.perf_counter())
            if during is not None and worker is None:
                worker = threading.Thread(target=during)
                worker.start()
    c.close()
    if worker is not None:
        worker.join()
    return [b - a for a, b in itertools.pairwise(times)]


SLOW_BUILD = '''
"""A test hook (tests/structured/test_grammar_test.py): a grammar build whose text holds the marker waits 1.5 s on its
compile thread, as a slow build would."""
import time

from llm_mini_infer.grammar import spec

_make = spec.GrammarSpecs._make


def _slow(build, token_ids=()):
    def held():
        text = build()
        if "slow-build-marker" in text:
            time.sleep(1.5)
        return text
    return _make(held, token_ids)


spec.GrammarSpecs._make = staticmethod(_slow)
'''


@pytest.mark.serial  # a server of its own beside the module's: three of Gemma's are 4 GB
def test_a_slow_build_leaves_a_stream_smooth(tmp_path) -> None:
    """A grammar build that takes 1.5 s (held on its compile thread by a hook the test puts on the server's path: no
    schema inside the caps builds that slowly) leaves another request's stream smooth. Its longest gap stays within
    0.75 s, half the build, of the longest of two runs without it: a build on the event loop would add the whole 1.5 s.
    A loaded host stalls a stream at times whatever runs (CI's full suite on an earlier 2-CPU runner: 3.0 s, with the
    server alone never), so a gap over the bound is measured again, up to three times, each with a new schema and so
    a new build: a build on the event loop fails every one."""
    if not env_has_model(GEMMA):
        pytest.skip(f"{GEMMA} is absent")
    hook = tmp_path / "hook"
    hook.mkdir()
    (hook / "sitecustomize.py").write_text(SLOW_BUILD)
    path = os.pathsep.join(filter(None, [str(hook), os.environ.get("PYTHONPATH")]))  # first, not instead
    srv = gemma_server(tmp_path, step_ms=10, env={"PYTHONPATH": path})
    srv.family = "gemma4"
    try:
        tries = []
        for i in range(3):
            slow = obj({"slow-build-marker": {"type": "integer"}, f"try{i}": {"type": "integer"}})  # never cached
            took = []

            def held(slow=slow, took=took) -> None:
                t = time.perf_counter()
                code, _ = ask(srv, response_format=rf(slow), max_tokens=2)
                took.append((code, time.perf_counter() - t))

            base = max(max(stream_gaps(srv)), max(stream_gaps(srv)))
            during = max(stream_gaps(srv, held))
            assert took and took[0][0] == 200 and took[0][1] > 1.4, took
            tries.append((round(base, 3), round(during, 3)))
            if during < base + 0.75:
                break
        else:
            pytest.fail(f"the stream stalled behind the slow build each time: (base, during) {tries}")
    finally:
        assert srv.stop() == 0, srv.log_path.read_text()[-3000:]
