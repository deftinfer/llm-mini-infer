"""The longest run of output a schema fixes, and its cap (`grammar.forced`).

- **The count** is never below llguidance's own forced bytes (`compute_ff_bytes`) at any state of random schemas'
  documents, compact and with any whitespace, and equals them on the plain shapes.
- **The cap:** each shape fixing `MAX_FORCED_RUN` bytes in a row builds, one byte more is a 400 that names the JSON
  pointer and the length, as a `response_format` under both backends and in each family's call grammar.
- **The shapes** that cost seconds under canonical forcing (scripted generators of costly shapes) are
  refused at once, in every mode.
- **Patterns** fix text by the rules measured on llguidance, an anchor read on each alternative; beside a `format`,
  as long as their intersection may fix; a pattern in syntax the count does not read is refused; so is a `$ref` cycle
  that never ends.
- **Numbers** as llguidance reads their bounds (doubles), and a count of properties by the runs it can force, never
  by the sum of its members.
"""

from __future__ import annotations

import json
import random
import re
import time

import numpy as np
import pytest

from llm_mini_infer.grammar import GrammarError
from llm_mini_infer.grammar.spec import json_schema_grammar
from llm_mini_infer.grammar.tools import ToolSchemaError, call_grammar

from ..support.grammar_cases import IDS, MODES, build, obj, tool
from ..support.timing import least_each, under

N = {"type": "null"}
RESPONSE = ["auto", "guidance"]


def forced():
    """The count's module (imported here, so the file collects on a tree without it)."""
    from llm_mini_infer.grammar import forced

    return forced


def cap() -> int:
    return forced().MAX_FORCED_RUN


def count(schema: dict) -> int | None:
    """The count of a `response_format` schema in the subset (None: it produces nothing)."""
    f = forced()
    runs = f.Runs(schema, f.json_syntax(schema))
    runs.definitions(schema, "#", None)
    got = runs.of(schema, "#")
    return None if got is None else got.inner


def refused(mode: str, schema: dict, match: str = "bytes of output in a row") -> float:
    """The seconds a build of `schema` took to be refused (`mode`: a backend or a family)."""
    start = time.perf_counter()
    with pytest.raises((GrammarError, ToolSchemaError), match=match):
        build(mode, schema)
    return time.perf_counter() - start


def dag(k: int, n: int, leaf: dict = N, name_len: int = 2) -> dict:
    """D_i has `k` required properties, each a `$ref` to D_{i+1}; D_n is `leaf`: the whole document is fixed."""
    names = [f"p{j}".ljust(name_len, "x") for j in range(k)]
    defs = {f"D{i}": obj({p: {"$ref": f"#/$defs/D{i + 1}"} for p in names}, additionalProperties=False)
            for i in range(n)}
    defs[f"D{n}"] = leaf
    return {"$ref": "#/$defs/D0", "$defs": defs}


# -- the count against llguidance ------------------------------------------------------------------------------------

def _byte_ids() -> tuple[dict[int, int], dict[int, int]]:
    """`spec`'s check tokenizer: a token per byte (the ByteLevel alphabet, sorted), as byte -> id and id -> byte."""
    from tokenizers import pre_tokenizers

    bs = [*range(ord("!"), ord("~") + 1), *range(ord("¡"), ord("¬") + 1), *range(ord("®"), ord("ÿ") + 1)]
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    ids = {c: i for i, c in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    b2id = {b: ids[chr(c)] for b, c in zip(bs, cs, strict=True)}
    return b2id, {i: b for b, i in b2id.items()}


def longest_forced(grammar: str, seed: int, steps: int = 2000) -> int:
    """The longest run llguidance forces along one random walk of `grammar` over the check tokenizer: the forced bytes
    consumed whole, else a random byte the mask allows (a structural one, often)."""
    import llguidance

    from llm_mini_infer.grammar import spec

    b2id, id2b = _byte_ids()
    m = llguidance.LLMatcher(spec._check_tokenizer(), grammar, log_level=0)
    assert not m.is_error(), m.get_error()
    rng, worst, eos = random.Random(seed), 0, spec.BYTE_TOKENS - 1
    for _ in range(steps):
        if m.is_stopped():
            break
        ff = m.compute_ff_bytes()
        worst = max(worst, len(ff))
        if ff:
            toks = []
            for i, piece in enumerate(re.split(rb"\xff\[(\d+)\]", ff)):
                toks += [int(piece)] if i % 2 else [b2id[b] for b in piece]
            assert m.consume_tokens(toks), m.get_error()
            continue
        bits = np.unpackbits(np.frombuffer(m.compute_bitmask(), np.uint8), bitorder="little")
        ids = [int(i) for i in np.nonzero(bits[:spec.BYTE_TOKENS + spec.SPARE_IDS])[0]]
        if not ids or (eos in ids and rng.random() < 0.5):
            break
        ids = [i for i in ids if i != eos] or ids
        marks = [i for i in ids if id2b.get(i, 0) in b'"{}[],:']
        assert m.consume_token(rng.choice(marks if marks and rng.random() < 0.3 else ids)), m.get_error()
    return worst


PATTERNS = ["^x{3}$", "^x{2,5}$", "^[a-z]{3}$", "^x+x{4}$", "^(x+){5}$", "^(x|xx)x{3}$", "^y{4}(x|xx)x{3}$",
            "^(ab|ac){3}$", "^[x]{6}$", "^a?b{5}$", "^(?:ab){3}c$", "^x*y{5}$", "^\\d{3}-x{4}$", "^[éè]{2}$",
            "^\"{3}$", "^x{5}", "x{5}$", "x{5}", "^x{3}$|^y{4}$", "^(x{2}){3}$", "^(xy?){4}$", "^(a|b|ab){2}c{3}$",
            "^[\\x01\\x02]{2}$", "^((x+)y){3}$", "^.{2}x{4}$", "^[x&&x]{4}$"]


def random_schema(r: random.Random, depth: int, defs: dict) -> dict:
    """A schema in the subset: objects (required and optional keys, open and closed), arrays (tuples, counts),
    values, numbers, patterns, alternatives, `$ref`s and a recursive model."""
    word = lambda: r.choice(["a", "ab", "abc", "k\"q", "é", "zz" * 9, "p" * r.randint(1, 20)])  # noqa: E731
    if depth <= 0 or r.random() < 0.3:
        return r.choice([
            N, {"type": "boolean"}, {"type": "string"}, {"type": "string", "maxLength": r.choice([0, 3])},
            {"const": r.choice([word(), 7, -3, 1.5, True, None, [1, "a"], {"b": 1}])},
            {"enum": [word() for _ in range(r.randint(1, 4))]},
            {"type": "integer", "minimum": (lo := r.randint(-20, 1000)), "maximum": lo + r.choice([0, 1, 9, 99])},
            {"type": "integer", "minimum": 7, "maximum": 13, "multipleOf": 10},
            {"type": "number", "minimum": 1.5, "maximum": 1.5},
            {"type": "string", "pattern": r.choice(PATTERNS), **({"maxLength": r.randint(0, 6)} if r.random() < 0.2
                                                                else {})},
            {"type": "string", "format": r.choice(["date", "uuid", "email"])}])
    k = r.randrange(7)
    if k <= 1:
        props = {word(): random_schema(r, depth - 1, defs) for _ in range(r.randint(0, 4))}
        req = [p for p in props if r.random() < 0.6]
        return {"type": "object", "properties": props, "required": req,
                **({"additionalProperties": False} if r.random() < 0.6 else {})}
    if k == 2:
        lo = r.choice([0, 1, 3])
        return {"type": "array", "items": random_schema(r, depth - 1, defs), "minItems": lo,
                **({"maxItems": lo + r.choice([0, 2])} if r.random() < 0.4 else {})}
    if k == 3:
        return {"type": "array", "prefixItems": [random_schema(r, depth - 1, defs) for _ in range(r.randint(1, 3))],
                **({"items": False} if r.random() < 0.5 else {}), "minItems": r.randint(0, 3)}
    if k == 4:
        return {"anyOf": [random_schema(r, depth - 1, defs) for _ in range(r.randint(1, 3))]}
    if k == 5:
        name = f"D{len(defs)}"
        defs[name] = random_schema(r, depth - 1, defs)
        return {"$ref": f"#/$defs/{name}"}
    name = f"R{len(defs)}"
    defs[name] = obj({"v": random_schema(r, 0, defs), "next": {"anyOf": [{"$ref": f"#/$defs/{name}"}, N]}},
                     additionalProperties=False)
    return {"$ref": f"#/$defs/{name}"}


def test_the_count_is_never_below_llguidance_s_forced_bytes() -> None:
    """At every state of random walks of random schemas' documents, compact and with any whitespace, the bytes
    llguidance forces are at most the count."""
    import llguidance

    from llm_mini_infer.grammar import bounds

    checked = 0
    for seed in range(150):
        r = random.Random(seed)
        defs: dict = {}
        schema = random_schema(r, r.randint(1, 4), defs)
        if defs:
            schema = {"allOf": [schema], "$defs": defs} if "$ref" in schema or "anyOf" in schema else \
                {**schema, "$defs": defs}
        try:
            bounds.check_json_schema(schema)
            n = count(schema)
        except bounds.BoundError:
            continue
        if n is None:
            continue
        for compact in (True, False):
            text = llguidance.LLMatcher.grammar_from_json_schema(json.dumps(schema),
                                                                 defaults={"whitespace_flexible": not compact})
            try:
                got = max(longest_forced(text, seed * 10 + w) for w in range(2))
            except AssertionError as exc:
                if "Unsatisfiable" in str(exc) or "Failed to generate" in str(exc):
                    break
                raise
            assert got <= n, (seed, compact, got, n, schema)
        checked += 1
    assert checked > 100


# -- the differential ---------------------------------------------------------------------------------------------

ATOMS = ["x", "y", "ab", "[x]", "[xy]", "[a-z]", ".", "\\d", "é", "[éè]", "\\.", '"', "[\\x01\\x02]", "\\u0041",
         "(?:xy)", "[^x]", "\\w", "[x&&x]", "[aa]", "\\p{L}"]
FORMATS = ["date", "time", "date-time", "uuid", "email", "ipv4", "ipv6", "uri", "hostname", "duration"]


def regex(r: random.Random, depth: int = 3) -> str:
    """A random regular expression in the reader's syntax."""
    if depth <= 0 or r.random() < 0.3:
        a = r.choice(ATOMS)
    else:
        k = r.randrange(4)
        a = ("".join(regex(r, depth - 1) for _ in range(r.randint(2, 3))) if k == 0 else
             "(" + "|".join(regex(r, depth - 1) for _ in range(r.randint(2, 3))) + ")" if k == 1 else
             "(" + regex(r, depth - 1) + ")")
    q = r.random()
    lo = r.randint(0, 3)
    return a + ("*" if q < 0.15 else "+" if q < 0.3 else "?" if q < 0.4 else f"{{{r.randint(0, 5)}}}" if q < 0.55
                else f"{{{lo},{lo + r.randint(0, 3)}}}" if q < 0.65 else f"{{{lo},}}" if q < 0.7 else "")


class Gen:
    """Random schemas in the subset, objects with required, optional and counted
    keys sharing prefixes, arrays with counts and tuples, values with escapes, numbers of 16 to 19 digits and -0.0,
    random patterns anchored on their alternatives and beside formats, lengths that may leave no string, `$ref` DAGs
    and recursion, separators; long fixed texts, so that runs reach hundreds of bytes."""

    def __init__(self, seed: int):
        self.r, self.defs = random.Random(seed), {}

    def word(self) -> str:
        r = self.r
        return r.choice(["a", "ab", "abd", "k\"q", "é", "\n", "a/b", "p" * r.randint(1, 30), "pq", "zz" * 9]) + \
            r.choice(["", "1", "pp", "x" * r.randint(0, 5)])

    def pattern(self) -> str:
        r, p = self.r, regex(self.r, self.r.randint(1, 4))
        a = r.random()
        return ("^" + p + "$" if a < 0.5 else "^" + p if a < 0.65 else f"^{p}$|^{regex(r, 2)}$" if a < 0.75
                else f"(^{p}$)|({regex(r, 2)})" if a < 0.85 else p)

    def leaf(self) -> dict:
        r = self.r
        k = r.randrange(20)
        if k == 0:
            return {"type": r.choice(["null", "boolean"])}
        if k == 1:
            return {"const": r.choice([self.word(), 7, -3, 1.5, True, None, [1, "a"], {"b": 1},
                                       "y" * r.randint(50, 600), 1e30, 2 ** 60, -0.0])}
        if k == 2:
            stem = "z" * r.randint(0, 300)
            return {"enum": [stem + w for w in r.sample(["", "a", "ab", "b", "\n", "é"], r.randint(1, 4))]}
        if k == 3:
            lo = r.choice([r.randint(-1000, 1000), 10 ** r.randint(15, 18) * r.choice([1, 2, 9]) + r.randint(-9, 9)])
            s = {"type": r.choice(["integer", "integer", "number"]), "minimum": lo,
                 "maximum": lo + r.choice([0, 1, 5, 99, 1000])}
            if r.random() < 0.2:
                s["exclusiveMinimum"] = s.pop("minimum")
            if r.random() < 0.15:
                s["multipleOf"] = r.choice([2, 5, 0.5])
            return s
        if k == 4:
            return {"type": "number", "minimum": r.choice([-0.0, 0.0, -0.0, 1.5]),
                    "maximum": r.choice([-0.0, 0.0, 1.5])}
        if k in (5, 6, 7):
            s = {"type": "string", "pattern": self.pattern()}
            if r.random() < 0.25:
                s[r.choice(["minLength", "maxLength"])] = r.randint(0, 10)
            if r.random() < 0.25:
                s["format"] = r.choice(FORMATS)
            return s
        if k == 8:
            return {"type": "string", "format": r.choice(FORMATS),
                    **({"maxLength": r.randint(0, 12)} if r.random() < 0.3 else {})}
        if k == 9:
            return {"type": "string", "pattern": f"^{r.choice(['x', 'ab', '[x]'])}{{{r.randint(20, 300)}}}$"}
        return r.choice([{"type": "string"}, {"type": "string", "maxLength": r.choice([0, 3])}, {}, N])

    def schema(self, depth: int) -> dict:
        r = self.r
        if depth <= 0 or r.random() < 0.3:
            return self.leaf()
        k = r.randrange(10)
        if k <= 2:
            props = {self.word(): self.schema(depth - 1) for _ in range(r.randint(0, 6))}
            s = {"type": "object", "properties": props, "required": [p for p in props if r.random() < 0.6]}
            if r.random() < 0.7:
                s["additionalProperties"] = False
            elif r.random() < 0.3:
                s["additionalProperties"] = self.leaf()
            if r.random() < 0.15:
                s[r.choice(["minProperties", "maxProperties"])] = len(s["required"]) + r.choice([0, 1, 1, 2])
            return s
        if k == 3:
            lo = r.choice([0, 1, 3, 8, 20])
            return {"type": "array", "items": self.schema(depth - 1), "minItems": lo,
                    **({"maxItems": lo + r.choice([0, 2])} if r.random() < 0.4 else {})}
        if k == 4:
            return {"type": "array", "prefixItems": [self.schema(depth - 1) for _ in range(r.randint(1, 3))],
                    **({"items": False} if r.random() < 0.5 else {}), "minItems": r.randint(0, 3)}
        if k == 5:
            return {"anyOf": [self.schema(depth - 1) for _ in range(r.randint(1, 3))]}
        if k == 6:
            return {"allOf": [self.schema(depth - 1)]}
        if k == 7 and self.defs and r.random() < 0.5:
            return {"$ref": "#/$defs/" + r.choice(list(self.defs))}
        if k <= 7:
            name = f"D{len(self.defs)}"
            self.defs[name] = {}
            self.defs[name] = self.schema(depth - 1)
            return {"$ref": f"#/$defs/{name}"}
        if k == 8:  # a DAG: every key required, the whole document fixed but the leaf
            n, leaf = r.randint(2, 5), self.leaf()
            base = len(self.defs)
            for i in range(n):
                self.defs[f"G{base + i}"] = {"type": "object", "properties": {
                    p: {"$ref": f"#/$defs/G{base + i + 1}"} for p in ("a", "bb")[:r.randint(1, 2)]},
                    "required": ["a", "bb"][:2], "additionalProperties": False}
                self.defs[f"G{base + i}"]["required"] = list(self.defs[f"G{base + i}"]["properties"])
            self.defs[f"G{base + n}"] = leaf
            return {"$ref": f"#/$defs/G{base}"}
        name = f"R{len(self.defs)}"  # recursion with an exit, sometimes under a count
        node = {"type": "object", "properties": {"v": self.leaf(), "next": {"anyOf": [{"$ref": f"#/$defs/{name}"}, N]},
                                                 "o": {"$ref": f"#/$defs/{name}"}},
                "required": ["v", "next"], "additionalProperties": False}
        if r.random() < 0.4:
            node["minProperties"] = 3
            del node["required"]
        self.defs[name] = node
        return {"$ref": f"#/$defs/{name}"}

    def root(self) -> dict:
        s = self.schema(self.r.randint(1, 5))
        if self.defs:
            s = {"allOf": [s], "$defs": self.defs} if any(k in s for k in ("$ref", "anyOf", "allOf")) else \
                {**s, "$defs": self.defs}
        if self.r.random() < 0.1:
            s["x-guidance"] = {"item_separator": self.r.choice([",", ", ", ";;"]),
                               "key_separator": self.r.choice([":", ": "])}
        return s


def test_the_count_bounds_llguidance_on_random_schemas() -> None:
    """At every state of random walks of 300 random schemas' documents, compact and with any whitespace, llguidance
    forces at most the count. Not vacuous: dozens of the walks see a run of
    100 bytes or more, and the longest is in the hundreds."""
    import llguidance

    from llm_mini_infer.grammar import bounds, spec

    checked, long_runs, longest = 0, 0, 0
    for seed in range(300):
        schema = Gen(seed).root()
        try:
            bounds.check_json_schema(schema)
            forced().check_json_schema(schema)
        except bounds.BoundError:
            continue
        n = count(schema)
        for compact in (True, False):
            text = llguidance.LLMatcher.grammar_from_json_schema(json.dumps(schema),
                                                                 defaults={"whitespace_flexible": not compact})
            if spec.check(text):  # llguidance's own 400
                break
            got = max(longest_forced(text, seed * 10 + w, 3000) for w in range(2))
            assert got <= n, (seed, compact, got, n, schema)
            long_runs += got >= 100
            longest = max(longest, got)
        checked += 1
    assert checked > 200 and long_runs > 20 and longest > 300, (checked, long_runs, longest)


FOUND = {  # undercounts found by differential runs (llguidance forced, then the count before the fix), shrunk
    "a number's bounds of one double": {"type": "array", "minItems": 20, "items": {
        "type": "number", "minimum": 2000000000000000005, "maximum": 2000000000000000006}},  # 400, 43
    "bounds llguidance's check refuses": obj({"p" * 31: obj({"x": {  # 37, 36
        "type": "integer", "maximum": 99999999999999995, "exclusiveMinimum": 99999999999999994}}, False,
        additionalProperties=False)}),
    "an array whose item perhaps produces nothing": obj({"k": {"items": {"type": "array", "items": {
        "type": "array", "minItems": 3, "prefixItems": [{"type": "string", "pattern": "[\\x01\\x02]+",
                                                         "format": "email"}]}}, "minItems": 20}},
        False, additionalProperties=False),  # 59, 23
    "a format a length leaves one string": {"type": "array", "minItems": 3, "maxItems": 3, "items": obj({
        "a": N, "b": {"type": "string", "format": "ipv6", "maxLength": 2}}, additionalProperties=False)},
    "a counted object whose optional member has no value": {"type": "array", "minItems": 5, "items": {  # 205, 98
        "type": "array", "minItems": 3, "maxItems": 3, "items": obj({"p": {"type": "array", "minItems": 4, "items": {
            "type": "string", "pattern": "^x*$", "format": "ipv6"}}, "abc": N}, False, additionalProperties=False,
            maxProperties=4) | {"required": ["abc"]}}},
}


@pytest.mark.parametrize("name", FOUND)
def test_the_undercounts_are_counted(name) -> None:
    """Each shape that once counted below llguidance (a number's bounds that are one double, an
    integer's that llguidance's own check refuses, an item that perhaps produces nothing inside an array that needs it,
    `ipv6` within 2 characters: only `::`) now counts at least what llguidance forces."""
    import llguidance

    schema = FOUND[name]
    text = llguidance.LLMatcher.grammar_from_json_schema(json.dumps(schema), defaults={"whitespace_flexible": False})
    assert max(longest_forced(text, w, 4000) for w in range(8)) <= count(schema)


NUMBER_CASES = {  # numbers (llguidance forced, against the count before the fix), each in an array
    "bounds that meet beside an exclusive one": ({"type": "number", "minimum": 123456, "maximum": 123456,
                                                  "exclusiveMinimum": 0}, 600),  # 4,200, 17
    "-0.0 beside a larger exclusive bound": ({"type": "number", "minimum": -0.0, "maximum": 0.0,
                                              "exclusiveMaximum": 1}, 2100),  # 6,300, 17
    "a few multiples between fractional bounds": ({"type": "number", "minimum": 1e-05, "maximum": 0.50001,
                                                   "multipleOf": 0.25}, 900),  # 4,500, 19
    "an exclusive fractional bound and a multiple": ({"type": "number", "exclusiveMinimum": 123.456,
                                                      "maximum": 123.956, "multipleOf": 0.3}, 900),  # 5,400, 19
    "bounds past 10**15": ({"type": "number", "minimum": 1e15, "maximum": 1000000000000000.125}, 1000),  # 17,000, 41
}


@pytest.mark.parametrize("name", NUMBER_CASES)
def test_numbers_leaving_one_value_are_counted(name) -> None:
    """Where llguidance's reading of a number leaves one value the exact reading does not (bounds that meet beside an
    exclusive bound that does not bind; its digit ranges leaving out multiples; digits past a double's), the count
    takes it as fixed throughout: llguidance forces at most the count, and the arrays are past the cap."""
    import llguidance

    item, n = NUMBER_CASES[name]
    for schema in (item, {"type": "array", "items": item, "minItems": 3, "maxItems": 3}):
        text = llguidance.LLMatcher.grammar_from_json_schema(json.dumps(schema),
                                                             defaults={"whitespace_flexible": False})
        assert max(longest_forced(text, w) for w in range(6)) <= count(schema), schema
    for mode in RESPONSE:
        refused(mode, {"type": "array", "items": item, "minItems": n, "maxItems": n})


def test_a_number_s_edges_are_counted() -> None:
    """A multiple's own digits at either end of an integer range (`-0` of multiples of 3 from -1, the `000` of 1000)
    and a negative lower bound's digits after its `-` are in its run; one value past 2**53 whose double is no multiple
    of `multipleOf` may be llguidance's one value, so it builds."""
    import llguidance

    for schema in ({"type": "integer", "minimum": -1, "maximum": 9, "multipleOf": 3},
                   {"type": "integer", "exclusiveMaximum": 1e16, "multipleOf": 7},
                   {"type": "integer", "maximum": 9223372036854775807, "multipleOf": 1000},  # its `000`
                   {"type": "integer", "exclusiveMinimum": -1.5, "minimum": -1.5}):  # `-1`'s 1
        text = llguidance.LLMatcher.grammar_from_json_schema(json.dumps(schema),
                                                             defaults={"whitespace_flexible": False})
        assert max(longest_forced(text, w) for w in range(6)) <= count(schema), schema
    big = {"type": "number", "minimum": 123456789012345678, "maximum": 123456789012345678, "multipleOf": 3}
    for mode in RESPONSE:
        assert json_schema_grammar(obj({"a": big}), mode, mode == "guidance")


@pytest.mark.parametrize("pattern", ["\\s*", "[ ]*", "[ ]?", "[\\n ]*", "[ \\t\\n]{0,4}"])
def test_a_whitespace_pattern_that_only_chooses_builds(pattern) -> None:
    """An `x-guidance` whitespace pattern that fixes nothing at a gap's start builds (`\\s*` fixes at most the rest of
    a character's bytes); one that fixes a byte at every gap is refused: it would join the tokens' runs."""
    schema = obj({"a": {"type": "string"}}) | {"x-guidance": {"whitespace_pattern": pattern}}
    for mode in RESPONSE:
        assert json_schema_grammar(schema, mode, mode == "guidance")
    with pytest.raises(GrammarError, match="whitespace_pattern fixes text"):
        json_schema_grammar(obj({"a": {"type": "string"}}) | {"x-guidance": {"whitespace_pattern": "\\n"}},
                            "guidance", True)


@pytest.mark.parametrize("pattern, why", [("^[a-", "an unclosed class"), ("^a{²}", "a counted repetition"),
                                          ("^(?=a)b", "a lookaround"), ("^(?<!a)b", "a lookaround")])
def test_a_malformed_pattern_is_a_400_that_says_why(pattern, why) -> None:
    """A class range cut short raised IndexError (a 500); a non-ASCII digit gave Python's message."""
    for mode in RESPONSE:
        with pytest.raises(GrammarError, match=why):
            json_schema_grammar(obj({"a": {"type": "string", "pattern": pattern}}), mode, mode == "guidance")
    with pytest.raises(ToolSchemaError, match=why):
        call_grammar("qwen3_coder", [tool(obj({"a": obj({"b": {"type": "string", "pattern": pattern}})}))],
                     "required", ids=IDS["qwen3_coder"])


def test_qwen_s_raw_texts_that_start_one_another_keep_their_json_alternative() -> None:
    """A parameter of raw texts that start one another (`zz`, `zzzzz`) beside a JSON alternative: its run is either's
    (68 forced, 51 counted when the JSON alternative's long value was left out)."""
    params = obj({"a": {"anyOf": [obj({"a/b": {"enum": ["z" * 5000]}}, False), {"enum": ["zz", "zzzzz"]}]}})
    with pytest.raises(ToolSchemaError, match="bytes of output in a row"):
        call_grammar("qwen3_coder", [tool(params)], "required", ids=IDS["qwen3_coder"])


def test_a_tool_call_that_can_produce_no_value_is_refused() -> None:
    """Qwen's `%json` items keep `minItems`: a required parameter that is an array of itself, three at least, has no
    value, and llguidance's forced bytes run on in it (67 forced against no count once). A 400, as for a
    `response_format`; Gemma's grammar takes no counts, so its call builds."""
    params = {**obj({"a": {"$ref": "#/$defs/D"}}), "$defs": {"D": {"type": "array", "items": {"$ref": "#/$defs/D"},
                                                                   "minItems": 3}}}
    with pytest.raises(ToolSchemaError, match="tool 'f''s call can produce no value"):
        call_grammar("qwen3_coder", [tool(params)], "required", ids=IDS["qwen3_coder"])
    assert call_grammar("gemma4", [tool(params)], "required", ids=IDS["gemma4"])
    with pytest.raises(GrammarError, match="can produce no value"):
        json_schema_grammar(params, "guidance", True)


TOOL_IDS = {"gemma4": {"opener": 257, "closer": 258, "end": 259, "quote": 260},
            "qwen3_coder": {"opener": 261, "closer": 262, "end": 263, "think": (264, 265)}}
TOOL_DROP = ("prefixItems", "minProperties", "maxProperties", "x-guidance")


def tool_schema(x):
    """A schema with the keywords tools do not take dropped."""
    if isinstance(x, dict):
        return {k: tool_schema(v) for k, v in x.items() if k not in TOOL_DROP and not (k == "items" and
                                                                                       not isinstance(v, dict))}
    return [tool_schema(v) for v in x] if isinstance(x, list) else x


def test_the_count_bounds_llguidance_on_random_tools(monkeypatch) -> None:
    """Random tools of both families, each choice (`required` with and without parallel calls, a named tool), their call
    grammars walked over the check tokenizer: llguidance forces at most the count of each call's syntax and values
   . Names that start one another, values holding the block's end."""
    from llm_mini_infer.grammar import tools as T

    seen: list[int] = []
    real = forced().cap
    monkeypatch.setattr(forced(), "cap", lambda run, what: (seen.append(run.inner), real(run, what))[1])
    checked = 0
    for seed in range(60):
        r = random.Random(seed)
        tools = []
        for t in range(r.randint(1, 3)):
            g = Gen(seed * 10 + t)
            props, req = {}, []
            for _ in range(r.randint(0, 5)):
                name = r.choice([g.word(), "a", "a:b", "a:t", "k>\nz", "k", "kk", "a,b"])
                props[name] = tool_schema(g.schema(r.randint(0, 3))) if r.random() < 0.7 else r.choice([
                    {"const": r.choice(["x" * r.randint(1, 200), 3, True, None, {"b": "y" * 10}])},
                    {"enum": r.sample(["z" * 20, "z" * 20 + "\n", "z" * 20 + "a", "zz", "true", "1"], r.randint(2, 4))},
                    {"enum": ["v\n</parameter>\n<parameter=" + name + ">\nw", "v"]}, {"type": "boolean"}, N])
                if r.random() < 0.6:
                    req.append(name)
            params = {"type": "object", "properties": props, "required": list(dict.fromkeys(req))}
            if g.defs:
                params["$defs"] = tool_schema(g.defs)
            tools.append({"type": "function", "function": {"name": r.choice(["f", "get_x", "f" * 30]) + str(t),
                                                           "parameters": params}})
        for fam in ("gemma4", "qwen3_coder"):
            for choice, parallel in (("required", True), ("required", False), ("named", True)):
                seen.clear()
                try:
                    text, _ = T.call_grammar(fam, json.loads(json.dumps(tools)), choice, tools[0]["function"]["name"],
                                             parallel, True, ids=TOOL_IDS[fam])
                except T.ToolSchemaError:
                    continue
                try:
                    got = max(longest_forced(text, seed * 100 + w, 3000) for w in range(2))
                except AssertionError as exc:
                    if "nsatisfiable" in str(exc):
                        continue
                    raise
                assert got <= max(seen, default=0), (seed, fam, choice, parallel, got, seen, tools)
                checked += 1
    assert checked > 200, checked


@pytest.mark.parametrize("schema, run", [
    (dag(2, 6), 1075), (dag(2, 9), 8691), (dag(2, 3), 123),
    (obj({"x" * 4000: N}, additionalProperties=False), 4009),
    (obj({"a": {"const": "x" * 4000}}, additionalProperties=False), 4008),
    (obj({"a": {"type": "string", "pattern": "^x{4000}$"}}, additionalProperties=False), 4008),
    ({"type": "array", "items": N, "minItems": 800}, 4000),
    ({"type": "array", "items": N, "minItems": 800, "maxItems": 800}, 4001),
    (obj({"a": {"enum": ["x" * 4000 + "A", "x" * 4000 + "B"]}}, additionalProperties=False), 4006),
    (obj({"a": {"enum": ["A" + "x" * 4000, "B" + "x" * 4000]}}, additionalProperties=False), 4002),
    (dag(2, 9, {"type": "integer", "minimum": 12345, "maximum": 12345}), 9203),
    (obj({"a": {"type": "number", "minimum": 1e100, "maximum": 1e100}}, additionalProperties=False), 107),
    ({"type": "integer", "minimum": -1000, "maximum": -999}, 3),  # `-1`, then `000`
    (obj({"a": {"const": 1}, "b": {"type": "number", "exclusiveMinimum": 1, "exclusiveMaximum": 1}},
         additionalProperties=False) | {"required": ["a"]}, 7),  # `b` takes no value
    ({"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "array", "items": obj(
        {}, additionalProperties=False) | {"minProperties": 2}}}, 10),  # no item: `[[],[],[]]`
])
def test_the_count_equals_llguidance_s_on_plain_shapes(schema, run) -> None:
    """The DAG of `null`s counts 123 bytes at depth 3 and 1,075 at 6, and each of these the run llguidance forces
    (measured with `compute_ff_bytes` on llguidance 1.9.1)."""
    import llguidance

    assert count(schema) == run
    text = llguidance.LLMatcher.grammar_from_json_schema(json.dumps(schema), defaults={"whitespace_flexible": False})
    assert max(longest_forced(text, w, 20_000) for w in range(2)) == run


@pytest.mark.parametrize("schema, llg", [  # each once counted below llguidance's run
    ({"type": "string", "pattern": "^([aa]b){30}$"}, 62),  # `[aa]` is one character
    ({"type": "string", "pattern": "^[a\\x61]cdef$"}, 7),
    ({"const": 1e30}, 31),  # llguidance writes all 31 digits
    ({"const": [1e300] * 2}, 605),
    ({"type": "number", "multipleOf": 3.14159}, 5),
    ({"type": "number", "maximum": 5}, 1),
])
def test_the_count_bounds_llguidance_s_run_on_repeated_classes_and_floats(schema, llg) -> None:
    """A class of repeated members, a float's digits and a fraction's bounds: llguidance forces at most the count."""
    import llguidance

    text = llguidance.LLMatcher.grammar_from_json_schema(json.dumps(schema), defaults={"whitespace_flexible": False})
    forced_now = max(longest_forced(text, w, 20_000) for w in range(4))
    assert forced_now >= llg and count(schema) >= forced_now, (forced_now, count(schema))


def test_scaled_repeated_classes_floats_and_integers_are_refused() -> None:
    """Their scaled forms, past the cap: classes of one member repeated, floats' digits, a whitespace pattern that
    fixes text between tokens, and an integer outside the supported integer range, refused before it is built."""
    for schema, match in (
            (obj({"a": {"type": "string", "pattern": "^([aa]b){3000}$"}}), "bytes of output in a row"),
            (obj({"a": {"const": [1e300] * 14}}), "bytes of output in a row"),
            (obj({"a": N}) | {"x-guidance": {"whitespace_pattern": "\\n{5000}"}}, "whitespace_pattern fixes text"),
            (obj({"a": {"const": -(1 << 63) + 1}}), "an integer outside the supported integer range"),
            (obj({"a": {"enum": [[-(1 << 63)]]}}), "an integer outside the supported integer range")):
        for mode in RESPONSE:
            refused(mode, schema, match)
    assert json_schema_grammar(obj({"a": N}) | {"x-guidance": {"whitespace_pattern": "[ \\t\\n]*"}}, "guidance", True)
    assert json_schema_grammar(obj({"a": {"const": -(1 << 63) + 5000}}), "guidance", True)


def test_a_length_beside_a_format_or_a_pattern_fixes_what_its_choices_can() -> None:
    """A length leaves a format's fixed bytes alone (pydantic's HttpUrl, a URL of 2,083 characters at most, in a
    recursive model), and fixes a pattern's text only where it can leave a choice between lengths one: a name
    pattern of 5,000 characters at most builds, `^(x|yy){5000}$` of 5,000 does not."""
    url = {"type": "string", "format": "uri", "minLength": 1, "maxLength": 2083}
    model = {"$defs": {"C": obj({"id": {"type": "integer"}, "url": url,
                                 "replies": {"type": "array", "items": {"$ref": "#/$defs/C"}}},
                                additionalProperties=False)}, "$ref": "#/$defs/C"}
    assert count(model) < 100
    name = {"type": "string", "pattern": "^[A-Z][a-z]+( [A-Z][a-z]+)*$", "maxLength": 5000}
    assert count(obj({"n": name})) < 100
    for mode in RESPONSE:
        assert json_schema_grammar(model, mode, mode == "guidance")
        refused(mode, obj({"a": {"type": "string", "pattern": "^(x|yy){5000}$", "maxLength": 5000}}))


def test_a_deeply_nested_pattern_is_a_400() -> None:
    """Groups nested past 128 (llguidance's own parser refuses about 130) are the subset's 400, not a 500."""
    with pytest.raises(GrammarError, match="groups nested more than 128 deep"):
        json_schema_grammar(obj({"a": {"type": "string", "pattern": "^" + "(a|" * 250 + "b" + ")" * 250}}),
                            "guidance", True)


# -- the cap ----------------------------------------------------------------------------------------------------------

AT_CAP = {  # a schema fixing `n` bytes in a row, and where
    "key": (lambda n: obj({"x" * (n - 9): N}, additionalProperties=False), "#"),
    "const": (lambda n: obj({"a": {"const": "x" * (n - 8)}}, additionalProperties=False), "#"),
    "pattern": (lambda n: obj({"a": {"type": "string", "pattern": f"^x{{{n - 8}}}$"}}, additionalProperties=False),
                "#"),
    "enum": (lambda n: obj({"a": {"enum": ["A" + "x" * (n - 2), "B" + "x" * (n - 2)]}}, additionalProperties=False),
             "#"),  # after the first letter, the rest of the value, its quote and the object's `}`
    "enum alone": (lambda n: {"enum": ["A" + "x" * (n - 1), "B" + "x" * (n - 1)]}, "#"),
    "minItems": (lambda n: {"type": "array", "items": {"const": "x" * (n - 4)}, "minItems": 1, "maxItems": 1}, "#"),
    "after a choice": (lambda n: obj({"o": {"type": "boolean"}, "a": {"const": "x" * (n - 12)}},
                                     additionalProperties=False), "#"),
    "a definition's end and the next key": (
        lambda n: {"$defs": {"E": obj({"e": {"const": "x" * ((n - 22) // 2)}}, additionalProperties=False)},
                   **obj({"d": {"$ref": "#/$defs/E"}, "z" * ((n - 22) - (n - 22) // 2): N},
                         additionalProperties=False)},
        "#"),
}


@pytest.mark.parametrize("mode", RESPONSE)
@pytest.mark.parametrize("shape", AT_CAP)
def test_a_response_format_fixes_the_cap_and_no_byte_more(shape, mode) -> None:
    make, at = AT_CAP[shape]
    n = cap()
    assert count(make(n)) == n
    assert json_schema_grammar(make(n), mode, mode == "guidance")
    refused(mode, make(n + 1), f"fixes {n + 1:,} bytes of output in a row at {re.escape(at)}, more than {n:,}")


@pytest.mark.parametrize("fam", ["gemma4", "qwen3_coder"])
def test_a_tool_s_call_counts_its_syntax(fam) -> None:
    """A `const` parameter, a long name and a fixed object, in each family's call syntax: the parameter's text joins
    the call's (Gemma's `<|tool_call>call:NAME{key:<|"|>…<|"|>}<tool_call|>`, Qwen's `<parameter=K>` blocks)."""
    def built(params: dict, name: str = "f") -> None:
        call_grammar(fam, [tool(params, name)], "required", ids=IDS[fam])

    n = cap()
    for params in (obj({"a": {"const": "x" * (n - 200)}}), obj({"a": {"const": "x" * (n // 2)}, "b": N}),
                   obj({"a": {"const": 5}, "b": {"enum": ["x" * (n - 200)]}})):
        built(params)
    for params in (obj({"a": {"const": "x" * n}}),
                   obj({"a": {"const": "x" * (n // 2)}, "b": {"const": "y" * (n // 2)}}),
                   obj({"a": obj({k: N for k in ("p" * 400, "q" * 400, "r" * 400)} | {"s" * 3000: N},
                                 additionalProperties=False)})):
        with pytest.raises(ToolSchemaError, match=r"'f''s call fixes .* bytes of output in a row"):
            built(params)
    with pytest.raises(ToolSchemaError, match="bytes of output in a row"):
        built(obj({"a": N}), "g" * n)


# -- the shapes -------------------------------------------------------------------------------------------------------

SHAPES = {  # costly shapes under canonical forcing (guidance-ai/llguidance#440)
    "the null DAG, 15 levels": dag(2, 15),
    "keys of 1,000 letters": dag(2, 15, name_len=1000),
    "keys of 4,000 letters, 12 levels": dag(2, 12, name_len=4000),
    "leaves ^x{1000}$": dag(2, 15, {"type": "string", "pattern": "^x{1000}$"}),
    # scripted generators
    "forced_dag 4 10": dag(4, 10),
    "forced_dag 16 6": dag(16, 6),
    "forced_one key-x 16000": obj({"x" * 16_000: N}, additionalProperties=False),
    "forced_one const-words 16000": obj({"a": {"const": " ".join(["abcdef"] * 2300)}}),
    "pattern_probe rep 100000": obj({"a": {"type": "string", "pattern": "^x{100000}$"}}),
    "pattern_probe dagrep": dag(2, 15, {"type": "string", "pattern": "^x{100000}$"}),
    # hand-built costly shapes
    "optional and required mixed": obj({"o": N, "k" * 5000: N, "z": N}, additionalProperties=False) | {
        "required": ["k" * 5000]},
    "minItems nested, 100 by 100 nulls": {"type": "array", "minItems": 100, "maxItems": 100,
                                          "items": {"type": "array", "items": N, "minItems": 100, "maxItems": 100}},
    "consts in a tuple": {"type": "array", "prefixItems": [{"const": "x" * 100}] * 50, "minItems": 50},
    "integers fixed by their bounds": dag(2, 12, {"type": "integer", "minimum": 7, "maximum": 13, "multipleOf": 10}),
    "a long enum prefix": obj({"a": {"enum": ["z" * 5000 + "a", "z" * 5000 + "b"]}}),
    "an enum of two 20,000-byte values": obj({"a": {"enum": ["A" + "x" * 20_000, "B" + "x" * 20_000]}}),
    "a long alternative after a choice": obj({"a": {"anyOf": [N, {"const": "y" * 5000}]}}),
    "(x{100}){100}": obj({"a": {"type": "string", "pattern": "^(x{100}){100}$"}}),
    "(x+){5000}": obj({"a": {"type": "string", "pattern": "^(x+){5000}$"}}),
    "a length leaving one string": obj({"a": {"type": "string", "pattern": "^(x|yy){5000}$", "maxLength": 5000}}),
    "long separators": obj({k: N for k in "abcdefgh"}) | {"x-guidance": {"item_separator": "," * 600}},
    "a definition's end into the next key": {
        "$defs": {"E": obj({"e": {"const": "x" * 3000}}, additionalProperties=False)},
        **obj({"d": {"$ref": "#/$defs/E"}, "z" * 3000: N}, additionalProperties=False)},
    "numbers of 301 digits": obj({f"n{i}": {"type": "number", "minimum": 1e300, "maximum": 1e300} for i in range(14)},
                                 additionalProperties=False),
}
RAW = {"pattern_probe rep 100000", "(x{100}){100}", "(x+){5000}", "a length leaving one string"}  # Qwen: raw text
FACETS = {"leaves ^x{1000}$", "pattern_probe rep 100000", "pattern_probe dagrep", "integers fixed by their bounds",
          "(x{100}){100}", "(x+){5000}", "a length leaving one string", "numbers of 301 digits"}  # Gemma ignores them


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", SHAPES)
def test_the_forced_run_shapes_are_refused_at_once(name, mode) -> None:
    """Each shape costs a second or more a mask (or a forced prefix past 120 s) under guidance on Gemma 4's
    tokenizer without the cap; with it, a 400 before anything compiles, in every mode. A tool's parameters are an
    object; Gemma's call grammars do not enforce facets, and Qwen's write a parameter's string as raw text, so the
    shapes built on those fix nothing there and build."""
    schema = SHAPES[name]
    if mode in ("gemma4", "qwen3_coder") and (schema.get("type") == "array" or "x-guidance" in schema):
        return  # not a tool's parameters
    if (mode == "gemma4" and name in FACETS) or (mode == "qwen3_coder" and name in RAW):
        assert build(mode, schema)
        return
    assert under(lambda: refused(mode, schema), 0.5) < 0.5


# -- patterns ---------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("pattern, run, llg", [  # the count, and the run llguidance 1.9.1 forces
    ("^x{100}$", 102, 102), ("^x{50,100}$", 51, 51), ("^[a-z]{100}$", 1, 1), ("^.{100}$", 6, 1), ("x{100}", 2, 1),
    ("^y{10}x+x{5}$", 18, 17), ("^(x+){50}$", 51, 51), ("^y{10}(x|xx)x{5}$", 18, 17), ("^[x]{20}$", 22, 22),
    ("^[x&&x]{20}$", 122, 22), ("(?i)^abc$", 5, 1), ('^"{5}$', 12, 12), ("^x{20}$|^y{20}$", 21, 20),
    ("^x{20}", 21, 21),
])
def test_a_pattern_fixes_what_llguidance_forces(pattern, run, llg) -> None:
    """A pattern fixes text from a leading `^`: literals and one-member classes, `x{n}` as n copies, `{n,m}` its n.
    A repetition or an alternation ends the run only where the next bytes differ, so `^(x+){50}$` fixes its 50 copies
    and `^y{10}x+x{5}$` its 10, then 6. An unanchored pattern fixes nothing but the quotes; a class the count does
    not take apart (`&&`) counts as a character of 6 bytes, and `.` as part of an escape."""
    import llguidance

    schema = {"type": "string", "pattern": pattern}
    assert count(schema) == run
    text = llguidance.LLMatcher.grammar_from_json_schema(json.dumps(schema), defaults={"whitespace_flexible": False})
    assert max(longest_forced(text, w) for w in range(6)) == llg


@pytest.mark.parametrize("pattern", ["^a b$(?x)", "^(?x)a  b$", "^x{,5}$", "^a\\bb$", "^a(?=b)$", "^a$b$", "^(^a)b$"])
def test_a_pattern_the_count_cannot_read_is_refused(pattern) -> None:
    """Syntax outside the subset of regex-syntax the count reads (verbose flags, a `{,m}` count, a lookaround, an
    anchor inside) is a 400 that names the pattern's pointer; llguidance refuses most of it too."""
    with pytest.raises(GrammarError, match="#/properties/a"):
        json_schema_grammar(obj({"a": {"type": "string", "pattern": pattern}}), "guidance", True)


FORMATTED = [  # a pattern beside a format, and the run llguidance 1.9.1 forces: the format rules out its choices
    ("email", '^[x"]{60}@a\\.com$', 68), ("email", '^([x"]{100}@c\\.com)?$', 108), ("uuid", "^(a|-)*$", 38),
    ("date-time", "^(1|-|T|:|Z)*$", 20), ("email", "^(x{100})*@a\\.com$", 101), ("hostname", "^(a|\\.)*$", 2),
    ("date", "^(1999-01-01)?$", 12), ("email", "^x*@a\\.com$", 6), ("uri", "^https:", 7),
    ("uri", "^(wss?|wss?\\+unix)://", 7), ("ipv4", "^(1|\\.)*$", 2), ("time", "^(1|:|Z)*$", 11),
    ("duration", "^P(1|D)*$", 3), ("ipv6", "^(a|:)*$", 1)]


@pytest.mark.parametrize("fmt, pattern, llg", FORMATTED)
def test_a_pattern_beside_a_format_counts_what_their_intersection_can_fix(fmt, pattern, llg) -> None:
    """llguidance intersects the two: `^[x"]{60}@a\\.com$` beside `email` forces its 60 x's, which neither forces
    alone, and `^(a|-)*$` beside `uuid` the whole uuid, of a pattern whose shortest string is empty. The count takes
    the pattern's longest string, each repetition without a maximum at its minimum and as many copies as the format's
    shortest string has characters, at most the format's longest: never below what llguidance forces."""
    import llguidance

    schema = {"type": "string", "format": fmt, "pattern": pattern}
    text = llguidance.LLMatcher.grammar_from_json_schema(json.dumps(schema), defaults={"whitespace_flexible": False})
    forced_now = max(longest_forced(text, w) for w in range(12))
    assert forced_now >= llg and count(schema) >= forced_now, (forced_now, count(schema))


def test_a_pattern_beside_a_format_builds_and_its_run_is_capped() -> None:
    """Real cases build (`^https:` beside `uri`); 50 copies of a 108-byte intersection are past the cap."""
    for fmt, pattern in (("uri", "^https:"), ("uri", "^(wss?|wss?\\+unix)://"), ("email", "^[^@]+@[^@]+$")):
        schema = obj({"a": {"type": "string", "format": fmt, "pattern": pattern}})
        for mode in RESPONSE:
            assert json_schema_grammar(schema, mode, mode == "guidance")
    long = {"type": "string", "format": "email", "pattern": '^([x"]{100}@c\\.com)?$'}
    for mode in RESPONSE:
        refused(mode, {"type": "array", "items": long, "minItems": 50})


# -- numbers ----------------------------------------------------------------------------------------------------------

BIG = [  # integer ranges of 16 to 19 digits: llguidance reads each bound as a double, so 10**17 + 5 is 10**17
    (10 ** 17, 10 ** 17 + 5, 18), (9999999999999994, 9999999999999999, 16), (20000000000000007, 20000000000000009, 17),
    (99999999999999994, 99999999999999999, 18), (999999999999999994, 999999999999999999, 19),
    (10 ** 18, 10 ** 18 + 5, 19), (100000000000000000, 100000000000001000, 15), (10 ** 15, 10 ** 15 + 5, 15)]


@pytest.mark.parametrize("kind", ["integer", "number"])
@pytest.mark.parametrize("lo, hi, llg", BIG)
def test_a_number_s_bounds_are_read_as_llguidance_reads_them(lo, hi, llg, kind) -> None:
    """Its bounds as doubles (an integer's moved inward and cast to 64 bits) as llguidance 1.9.1 does:
    10**17..10**17 + 5 is one value, fixed throughout (once counted 17, forced 18), an integer's or a number's.
    The count is what llguidance forces, or above it; in an array of 20, the run joins the items."""
    import llguidance

    for schema in ({"type": kind, "minimum": lo, "maximum": hi},
                   {"type": "array", "items": {"type": kind, "minimum": lo, "maximum": hi}, "minItems": 20}):
        text = llguidance.LLMatcher.grammar_from_json_schema(json.dumps(schema),
                                                             defaults={"whitespace_flexible": False})
        forced_now = max(longest_forced(text, w) for w in range(6))
        assert forced_now <= count(schema), (schema, forced_now, count(schema))
        if "items" not in schema:
            assert kind == "number" or forced_now <= llg <= count(schema), (forced_now, count(schema))


def test_integers_of_one_double_are_refused() -> None:
    """An array of 5,000 such integers forced 95,001 bytes against a count of 18 (13 ms masks), and a `$ref` DAG of 14
    levels 155,002 against 114 (30 ms): both past the cap now, as a `response_format` and in Qwen's `%json` items.
    Gemma's grammar does not enforce bounds, so its call fixes nothing there."""
    leaf = {"type": "integer", "minimum": 10 ** 17, "maximum": 10 ** 17 + 5}
    many = {"type": "array", "items": leaf, "minItems": 5000, "maxItems": 5000}
    assert count(many) == 95_001
    for mode in RESPONSE:
        refused(mode, many)
    for mode in (*RESPONSE, "qwen3_coder"):
        refused(mode, dag(2, 14, leaf))
    assert build("gemma4", dag(2, 14, leaf))
    draft4 = {"type": "integer", "minimum": 999, "exclusiveMinimum": True, "maximum": 1000}  # one value: 1000
    assert count(draft4) == 4


def test_a_minimum_of_negative_zero_writes_its_sign() -> None:
    """llguidance writes `-0` for a number whose minimum is -0.0 and whose bounds meet (once counted 1); an
    array of 1,400 of them fixes 4,201 bytes, past the cap."""
    import llguidance

    neg = {"type": "number", "minimum": -0.0, "maximum": -0.0}
    for schema, n in ((neg, 2), ({"type": "number", "minimum": -0.0, "maximum": 0.0}, 2),
                      ({"type": "number", "minimum": 0.0, "maximum": -0.0}, 1)):
        text = llguidance.LLMatcher.grammar_from_json_schema(json.dumps(schema),
                                                             defaults={"whitespace_flexible": False})
        assert count(schema) == max(longest_forced(text, w) for w in range(4)) == n
    for mode in RESPONSE:
        refused(mode, {"type": "array", "items": neg, "minItems": 1400, "maxItems": 1400})


# -- a count of properties --------------------------------------------------------------------------------------------

def counted_tree(k: int, **count_kw) -> dict:
    """A recursive object: `k` optional properties that are itself and a string, under a count."""
    node = obj({**{f"r{i}": {"$ref": "#/$defs/n"} for i in range(k)}, "b": {"type": "string"}}, False,
               additionalProperties=False)
    return {"$defs": {"n": {**node, **count_kw}}, "$ref": "#/$defs/n"}


def test_a_count_of_properties_bounds_the_runs_it_can_force() -> None:
    """Under `minProperties` or `maxProperties` llguidance makes one optional member required in each of its options,
    or leaves them out: the count runs through one of them at most, never through their sum. Recursive
    shapes (llguidance forces 4 to 6 bytes) counted more than 2**30, and 150 optional 30-byte keys under
    `minProperties` 5,252 (now past the cap on such keys, 32: `test_grammar_test`). They build now, at most 32 keys,
    and llguidance forces at most the count; an and/or/not filter builds too."""
    import llguidance

    keys = {f"k{i}".ljust(30, "x"): N for i in range(32)}
    filt = {"$defs": {"f": {"type": "object", "additionalProperties": False, "minProperties": 1, "maxProperties": 1,
                            "properties": {"and": {"type": "array", "items": {"$ref": "#/$defs/f"}},
                                           "or": {"type": "array", "items": {"$ref": "#/$defs/f"}},
                                           "not": {"$ref": "#/$defs/f"}, "field": {"type": "string"}}}},
            "$ref": "#/$defs/f"}
    shapes = [counted_tree(k, minProperties=1) for k in (2, 5, 10)] + [counted_tree(3, maxProperties=1), filt,
                                                                       obj(keys, False, additionalProperties=False,
                                                                           minProperties=1)]
    for schema in shapes:
        assert count(schema) < 400, count(schema)
        for mode in RESPONSE:
            assert json_schema_grammar(schema, mode, mode == "guidance")
        text = llguidance.LLMatcher.grammar_from_json_schema(json.dumps(schema),
                                                             defaults={"whitespace_flexible": False})
        assert max(longest_forced(text, w, 4000) for w in range(4)) <= count(schema)
    one = obj({"a": {"const": "x" * 100}, "b": N}, False, additionalProperties=False, minProperties=2)
    assert count(one) == count(obj({"a": {"const": "x" * 100}, "b": N}, additionalProperties=False))  # both written


# -- anchors ----------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("pattern", ["^$|(^(?:\\S+\\s+){0,99}\\S+$)", "(^gs://(.+))|(^http://(.+))|(^https://(.+))",
                                     "^BYTES|^KB|^MB", "(?:^ab$)|(?:^cd$)", "((^a{30}$)|(^b{30}$))|^c$",
                                     "(^x{40})|y", "(?i:^abc$)|^d$", "^([^,]|$)",
                                     "^(\\d{4}(-\\d{2})?(-\\d{2})?$)|(^\\d{4}-x$)", "^(^a)$", "(a|b{30})$",
                                     "^(x{30}|y{30})$"])
def test_an_anchor_on_an_alternative_or_a_group_that_is_one_is_read(pattern) -> None:
    """Real patterns and the corpus's: a `^` or a `$` at an alternative's start or end, or at that of a
    group that is a whole alternative between its own anchors, anchors it, as llguidance reads it. Each builds;
    llguidance forces at most the count."""
    import llguidance

    schema = {"type": "string", "pattern": pattern}
    for mode in RESPONSE:
        assert json_schema_grammar(obj({"a": schema}), mode, mode == "guidance")
    text = llguidance.LLMatcher.grammar_from_json_schema(json.dumps(schema), defaults={"whitespace_flexible": False})
    assert max(longest_forced(text, w) for w in range(8)) <= count(schema)


@pytest.mark.parametrize("pattern", ["^a^b", "^(^a)?$", "(^a)?b", "^x(^a|b)", "(^a)b", "^a$b", "^(a$)c"])
def test_another_anchor_inside_a_pattern_is_refused(pattern) -> None:
    with pytest.raises(GrammarError, match="#/properties/a"):
        json_schema_grammar(obj({"a": {"type": "string", "pattern": pattern}}), "guidance", True)


# -- recursion --------------------------------------------------------------------------------------------------------

def test_recursion_is_unfolded_and_a_cycle_that_never_ends_is_refused() -> None:
    """A linked list's closing text grows a level at a time: counted for 32 levels. A required `$ref` cycle with no
    way out can produce no value (llguidance's forced bytes go on until its item limit): a 400."""
    linked = {"$defs": {"N": obj({"v": {"type": "integer"}, "next": {"anyOf": [{"$ref": "#/$defs/N"}, N]}},
                                 additionalProperties=False)}, "$ref": "#/$defs/N"}
    assert count(linked) < 100
    for mode in RESPONSE:
        assert json_schema_grammar(linked, mode, mode == "guidance")
    endless = {"$defs": {"D": obj({"a": {"$ref": "#/$defs/D"}}, additionalProperties=False)}, "$ref": "#/$defs/D"}
    for mode in RESPONSE:
        with pytest.raises(GrammarError, match="can produce no value"):
            json_schema_grammar(endless, mode, mode == "guidance")
    useless = {"$defs": {"D": {"anyOf": [{"const": "x" * 3000}, {"$ref": "#/$defs/D"}]}},
               **obj({"a": {"$ref": "#/$defs/D"}, "b": {"$ref": "#/$defs/D"}}, additionalProperties=False)}
    assert count(useless) == 6015  # `D` is its const alone, so both are fixed: llguidance forces all of it


# -- the count's own cost ---------------------------------------------------------------------------------------------

def _chain(n: int, refs: int) -> dict:
    """D_i an object of `refs` optional `$ref`s to the next definitions and a boolean; the last a boolean."""
    defs = {f"D{i}": obj({**{f"r{j}": {"$ref": f"#/$defs/D{min(i + 1 + j, n - 1)}"} for j in range(refs)},
                          "b": {"type": "boolean"}}, False) for i in range(n - 1)}
    defs[f"D{n - 1}"] = {"type": "boolean"}
    return {"$defs": defs, "$ref": "#/$defs/D0"}


SCALED = {  # past the subset's caps: the count is called alone, its small size a few milliseconds
    "optional keys": (lambda n: obj({f"k{i}": N for i in range(n)}, False), 2000),
    "required keys": (lambda n: obj({f"k{i}": {"enum": ["a", "b"]} for i in range(n)}, additionalProperties=False),
                      2000),
    "a $ref chain": (lambda n: _chain(n, 1), 1000),
    "a $ref DAG": (lambda n: _chain(n, 2), 1000),
    "a recursive star": (lambda n: {"$defs": {"H": obj({f"m{i}": {"$ref": f"#/$defs/M{i}"} for i in range(n)}, False)}
                                    | {f"M{i}": obj({"h": {"$ref": "#/$defs/H"}, "v": {"type": "integer"}}, False)
                                       for i in range(n)}, "$ref": "#/$defs/H"}, 100),
    "an enum": (lambda n: {"enum": [f"{i:06d}v" for i in range(n)]}, 8000),
    "a pattern's alternation": (lambda n: {"type": "string", "pattern": "^(" + "|".join(f"w{i:05d}" for i in range(n))
                                           + ")$"}, 1000),
}


# Alone on CI's runner (2 cores, 4 vCPUs, two workers): beside a second worker on an earlier 1-core runner, "a
# pattern's alternation" once measured 8.8 times
@pytest.mark.serial
@pytest.mark.parametrize("shape", SCALED)
def test_the_count_is_linear_in_the_schema(shape) -> None:
    """Each definition counted once, an object's gaps and an enum's prefixes in one pass, a recursive group's rounds
    a constant: four times the schema takes about four times as long (measured 4.0 to 4.3). The bound, 8, fails a
    quadratic part of a third of the small size's time or more (a quadratic alone gives 16). The collector is off
    while timing: its full passes over many live objects grow faster than the count."""
    make, n = SCALED[shape]
    a, b = make(n), make(4 * n)
    small, big = least_each(lambda: forced().check_json_schema(a), lambda: forced().check_json_schema(b))
    assert big < 8 * small, (shape, small, big)
