"""The plan-diff harness (`tests/scripts/plandiff.py`): it finds no difference between a tree and itself, flags a
planted one-decision change (the mixed-step bound one token lower) with its first differing plan and fields, accepts it
where `--expect-changed` allows it, flags a strategy that draws other workloads or test cases (`--compare-draws`), and
digests decisions, not their storage, timings or neutral ADMIT terms."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from ..scripts import plandiff as pd

BOUND = "self.soft = min(self.soft, max(grid, row_tokens // n))"
TOKENS = "tokens=np.asarray(toks, np.int64)"  # the radix suite's drawn token arrays
COUNT = "n = draw(st.integers(1, 12))"  # the scheduler suite's request count


def _git_ok() -> bool:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=pd.ROOT, capture_output=True).returncode == 0
    except OSError:
        return False


needs_git = pytest.mark.skipif(not _git_ok(), reason="not a git checkout (the harness unpacks revisions)")


def holding(tree: Path, line: str) -> Path:
    """The one file of a tree's tests/ (this one aside) that holds `line`, once: either layout of tests/ (plandiff reads
    both)."""
    hits = [p for p in sorted((tree / "tests").rglob("*.py"))
            if p.name != Path(__file__).name and line in p.read_text()]
    assert len(hits) == 1 and hits[0].read_text().count(line) == 1, f"the planted line moved: update the test {hits}"
    return hits[0]


def suites_of(tree: Path, tmp: Path) -> dict[str, str]:
    """A tree's own fuzz suites and their tests (`fuzz_campaign.SUITES`), as its worker lists them."""
    res = pd.run_worker(tree, {"mode": "suites", "out": str(tmp / "suites.json")}, tmp / "suites.log", 600)
    assert "failed" not in res, res
    return res


@pytest.fixture(scope="module")
def trees(tmp_path_factory):
    """Two copies of HEAD, the second with the mixed-step bound one token lower (`StepBudget.decode_rows`)."""
    tmp = tmp_path_factory.mktemp("plandiff")
    pd.unpack("HEAD", tmp / "clean")
    pd.unpack("HEAD", tmp / "planted")
    path = tmp / "planted" / "llm_mini_infer" / "sched" / "scheduler.py"
    text = path.read_text()
    assert text.count(BOUND) == 1, "the planted line moved: update the test"
    path.write_text(text.replace(BOUND, BOUND.replace("row_tokens //", "(row_tokens - 1) //")))
    return tmp / "clean", tmp / "planted"


def _run(tmp_path, *args: str) -> tuple[int, dict, str]:
    out = tmp_path / "out"
    status = pd.main([*args, "--out", str(out), "--jobs", "4"])
    return status, json.loads((out / "summary.json").read_text()), (out / "report.md").read_text()


@needs_git
def test_a_relative_out_is_the_callers(trees, tmp_path, monkeypatch) -> None:
    clean, _ = trees
    monkeypatch.chdir(tmp_path)
    assert pd.main(["--base", str(clean), "--new", str(clean), "--suites", "radix", "--examples", "5", "--out",
                    "rel"]) == 0
    assert (tmp_path / "rel" / "report.md").exists()


@needs_git
def test_a_tree_against_itself_is_identical(trees, tmp_path) -> None:
    clean, _ = trees
    status, summary, report = _run(tmp_path, "--base", str(clean), "--new", "HEAD", "--suites", "scheduler",
                                   "arena_hits", "radix", "--examples", "40", "--compare-draws")
    assert status == 0, report
    assert summary["verdict"] == "identical"
    for s in summary["suites"]:
        assert s["examples"] >= 20 and s["changed"] == 0 and s["events"] > s["examples"], s
        assert s["only_base"] == s["only_new"] == 0 and s["base_errors"] == s["new_errors"] == 0
        d = s["draws"]
        assert d["workloads"] == [40, 40] and d["cases"][0] >= 40 and d["first_workload"] == d["first_case"] == -1, s
        assert "engine" not in d, s


@needs_git
def test_a_changed_draw_is_flagged(trees, tmp_path) -> None:
    """`--compare-draws`: a strategy that draws the same values in other types (the radix suite's token arrays as
    int32), or the same values through other spans (the scheduler suite's request count mapped through `int`), fails,
    though the base's workloads run alike in both trees."""
    clean, _ = trees
    redrawn = tmp_path / "redrawn"
    pd.unpack("HEAD", redrawn)
    for line, planted in ((TOKENS, TOKENS.replace("int64", "int32")), (COUNT, COUNT.replace("12))", "12).map(int))"))):
        path = holding(redrawn, line)
        path.write_text(path.read_text().replace(line, planted))
    status, summary, report = _run(tmp_path, "--base", str(clean), "--new", str(redrawn), "--suites", "radix",
                                   "scheduler", "--examples", "20", "--compare-draws")
    assert status == 1 and report.count("drawn differently") == 2, report
    by = {s["suite"]: s for s in summary["suites"]}
    assert all(s["changed"] == 0 for s in by.values()), report
    assert by["radix"]["draws"]["first_workload"] == 0 and by["radix"]["draws"]["first_case"] == -1, report
    assert by["scheduler"]["draws"]["first_case"] == 0, report


def test_a_draw_is_compared_with_its_types() -> None:
    """Values of other types hash differently in their `_typed` form, even where `fuzzing._plain` makes them the same
    JSON."""
    for a, b in (([1, 2], (1, 2)), (np.arange(3), np.arange(3, dtype=np.int32)), (np.int32(1), 1), (1, True),
                 (1, 1.0), (0.0, -0.0), ({"a": 1, "b": 2}, {"b": 2, "a": 1})):
        assert pd._repr_hash(pd._typed(a)) != pd._repr_hash(pd._typed(b)), (a, b)
    w = {"tokens": np.arange(5), "spans": [(1, 2)], "eos": None, "p": np.float32(0.5)}
    assert pd._repr_hash(pd._typed(w)) == pd._repr_hash(pd._typed({**w, "tokens": np.arange(5)}))


@needs_git
def test_a_planted_decision_is_flagged_with_its_fields(trees, tmp_path) -> None:
    clean, planted = trees
    # The first seed whose draw reaches the bound in both suites (not window_precise: its refines go in fixed pieces,
    # which pass the bound whole). The draws depend on Hypothesis's version and on the literals of the modules a draw
    # imports, which Hypothesis mixes into them: seed 8 reached it in both on macOS and in one on Linux, a tree apart
    suites = ("arena_hits", "gemma_images_mtp", "radix")
    reached = set()
    for seed in ["8", *(str(i) for i in range(1, 33) if i != 8)]:
        status, summary, report = _run(tmp_path / seed, "--base", str(clean), "--new", str(planted), "--suites",
                                       *suites, "--examples", "60", "--show", "1", "--seed", seed)
        by = {s["suite"]: s for s in summary["suites"]}
        reached |= {s for s in suites[:2] if by[s]["changed"]}
        if all(by[s]["changed"] for s in suites[:2]):
            break
    else:  # a suite that never flags the plant is a broken plandiff; draws that reach it apart, Hypothesis's move
        assert reached == set(suites[:2]), f"no seed's draw flagged the planted bound in {set(suites[:2]) - reached}"
        pytest.skip("no seed of 32 reached the planted bound in both suites at once (Hypothesis's draws moved)")
    assert status == 1, report
    assert by["radix"]["changed"] == 0  # no scheduler there
    changed = [by[s] for s in suites[:2]]
    assert all(s["changed"] and s["identical"] for s in changed), report  # each suite catches it, alone
    d = changed[0]["details"][0]
    assert d["kind"] == "plan" and d["event"] > 0
    assert "sched_n_query" in d["fields"] and "n_query_tokens" in d["fields"], d
    assert "**unexpected**" in report and "UNEXPECTED CHANGES" in report
    # the same change where it is expected passes
    status, summary, report = _run(tmp_path / "expected", "--base", str(clean), "--new", str(planted), "--suites",
                                   *suites, "--examples", "60", "--show", "0", "--seed", seed,
                                   "--expect-changed", ",".join(suites[:2]))
    assert status == 0 and summary["verdict"] == "changed only where expected", report


@needs_git
def test_the_draw_is_the_same_in_two_processes(trees, tmp_path) -> None:
    clean, _ = trees
    test = suites_of(clean, tmp_path)["window_mtp"]
    hashes = []
    for i in range(2):
        out, exs = tmp_path / f"d{i}.json", tmp_path / f"e{i}.jsonl"
        res = pd.run_worker(clean, {"mode": "draw", "suite": "window_mtp", "n": 25, "seed": 3, "examples": str(exs),
                                    "test": test, "out": str(out)}, tmp_path / f"d{i}.log", 600)
        assert "failed" not in res, res
        hashes.append([json.loads(x)["h"] for x in exs.read_text().splitlines()])
        assert {r["outcome"] for r in res["examples"].values()} == {"passed"}
    assert hashes[0] == hashes[1] and len(hashes[0]) >= 15


@dataclass
class _Plan:
    __slots__ = ("admit_slot", "extra", "n_query_tokens", "sched_slot", "step_id")
    step_id: int
    sched_slot: np.ndarray
    admit_slot: np.ndarray
    n_query_tokens: int
    extra: np.ndarray


@dataclass
class _Out:
    rids: np.ndarray
    first_t_admit_ns: np.ndarray
    released: list = field(default_factory=list)


@dataclass
class _Report:
    outputs: dict
    step_id: int
    runner_ns: int


def test_digests_follow_values_not_storage_or_timing() -> None:
    a = _Plan(3, np.array([1, 2], np.int32), np.zeros(0, np.int32), 7, np.zeros(0, np.int64))
    b = _Plan(3, np.array([1, 2], np.int64), np.zeros(0, np.int16), 7, np.zeros(0, np.int8))  # other dtypes, empties
    assert pd.event_digest(*pd.plan_event(a)) == pd.event_digest(*pd.plan_event(b))
    c = _Plan(3, np.array([2, 1], np.int32), np.zeros(0, np.int32), 7, np.zeros(0, np.int64))  # one decision
    assert pd.event_digest(*pd.plan_event(a)) != pd.event_digest(*pd.plan_event(c))
    d = _Plan(3, np.array([1, 2], np.int32), np.zeros(0, np.int32), 7, np.array([0]))  # a field newly set
    assert pd.event_digest(*pd.plan_event(a)) != pd.event_digest(*pd.plan_event(d))
    r1 = _Report({0: _Out(np.array([5]), np.array([111]))}, 4, 900)
    r2 = _Report({0: _Out(np.array([5]), np.array([222]))}, 4, 100)  # timings only
    r3 = _Report({0: _Out(np.array([5]), np.array([111]), [5])}, 4, 900)  # pixels released
    assert pd.event_digest(*pd.report_event(r1)) == pd.event_digest(*pd.report_event(r2))
    assert pd.event_digest(*pd.report_event(r1)) != pd.event_digest(*pd.report_event(r3))


@dataclass
class _Admit:
    __slots__ = ("admit_bias_n", "admit_freq", "admit_pres", "admit_prompt_len", "admit_rep", "admit_slot", "step_id")
    step_id: int
    admit_slot: np.ndarray
    admit_rep: np.ndarray
    admit_pres: np.ndarray
    admit_freq: np.ndarray
    admit_prompt_len: np.ndarray
    admit_bias_n: np.ndarray


@dataclass
class _Bare:
    __slots__ = ("admit_slot", "step_id")
    step_id: int
    admit_slot: np.ndarray


@dataclass
class _Seeded:
    __slots__ = ("admit_seed", "admit_slot", "step_id")
    step_id: int
    admit_slot: np.ndarray
    admit_seed: np.ndarray


def test_neutral_admit_terms_digest_as_their_absence() -> None:
    """The penalty terms every ADMIT carries (`plandiff.NEUTRAL`), all at their neutral values, digest as a plan from a
    tree without them; one value that is not neutral, in any of them, differs. Another all-zero field still counts."""
    def digest(p) -> bytes:
        return pd.event_digest(*pd.plan_event(p))

    slots = np.array([4, 5], np.int32)
    neutral = dict(admit_rep=np.ones(2, np.float32), admit_pres=np.zeros(2, np.float32),
                   admit_freq=np.zeros(2, np.float32), admit_prompt_len=np.zeros(2, np.int32),
                   admit_bias_n=np.zeros(2, np.int32))
    base = digest(_Admit(3, slots, **neutral))
    assert base == digest(_Bare(3, slots))
    for name, v in (("admit_rep", [1.0, 1.2]), ("admit_pres", [0, 0.5]), ("admit_freq", [0, -1]),
                    ("admit_prompt_len", [0, 7]), ("admit_bias_n", [0, 2])):
        assert digest(_Admit(3, slots, **{**neutral, name: np.array(v, neutral[name].dtype)})) != base, name
    assert digest(_Seeded(3, slots, np.zeros(2, np.int32))) != digest(_Bare(3, slots))


def test_first_difference_names_the_event_and_fields() -> None:
    base = [{"kind": "plan", "fields": {"a": [1, 2], "b": 3}}, {"kind": "report", "fields": {"x": [1]}}]
    new = [{"kind": "plan", "fields": {"a": [1, 2], "b": 3}}, {"kind": "report", "fields": {"x": [2], "y": 1}}]
    d = pd.first_difference(base, new)
    assert d["event"] == 1 and d["kind"] == "report"
    assert d["fields"] == {"x": {"base": [1], "new": [2], "lengths": [1, 1], "from": 0},
                           "y": {"base": None, "new": 1}}
    assert pd.first_difference(base, base[:1])["kind"] == "end"
    assert pd.first_difference(base, base)["event"] == -1
    # a long field from its first differing value on, with both lengths (40 tokens against 36)
    long_a = [{"kind": "plan", "fields": {"tokens": list(range(40))}}]
    long_b = [{"kind": "plan", "fields": {"tokens": list(range(36))}}]
    f = pd.first_difference(long_a, long_b)["fields"]["tokens"]
    assert f["lengths"] == [40, 36] and f["from"] == 36 and f["base"][:3] == ["…", 34, 35] and f["new"] == ["…", 34, 35]
    # the stored digests locate a difference the JSON values hide (an int against a float of the same value)
    same = [{"kind": "masks", "fields": {"n": 3}}]
    assert pd.first_difference(same, same, ["aa"], ["bb"])["event"] == 0


def test_scalars_digest_by_value() -> None:
    """A numpy integer where an int was, or a bool where a 1 was, is storage: the same digest (and arrays alike)."""
    assert pd.event_digest(*pd.masks_event(np.int64(3), np.zeros((4, 2), np.int32))) == \
        pd.event_digest(*pd.masks_event(3, np.zeros((4, 2), np.int64)))
    a = _Plan(np.int64(3), np.array([1]), np.zeros(0), np.int32(7), np.zeros(0))
    b = _Plan(3, np.array([1]), np.zeros(0), 7, np.zeros(0))
    assert pd.event_digest(*pd.plan_event(a)) == pd.event_digest(*pd.plan_event(b))


def test_new_failures_fail_the_gate_even_where_changes_are_expected() -> None:
    base = {"a": {"outcome": "passed", "digest": "1", "events": 3}, "b": {"outcome": "passed", "digest": "2",
                                                                            "events": 3}}
    new = {"a": {"outcome": "RuntimeError", "digest": "9", "events": 1}, "b": {"outcome": "passed", "digest": "3",
                                                                               "events": 3}}
    c = pd.compare(base, new)
    assert c["changed"] == ["a", "b"] and c["new_failures"] == ["a"] and c["new_errors"] == 1
    args = argparse.Namespace(expect_changed="s", examples=2, seed=6)
    with tempfile.TemporaryDirectory() as tmp:
        r = {"suite": "s", "compare": {**c, "unserializable": 0}, "details": []}
        assert pd._report([r], args, {"base": "b", "new": "n"}, Path(tmp), 1.0) == 1
        assert "new failures" in (Path(tmp) / "report.md").read_text()
        empty = {"suite": "s", "compare": {**pd.compare({}, {}), "unserializable": 5}, "details": []}
        assert pd._report([empty], args, {"base": "b", "new": "n"}, Path(tmp), 1.0) == 2  # nothing compared


def test_a_broken_run_exits_2(tmp_path) -> None:
    assert pd.main(["--base", "no-such-revision-xyz", "--out", str(tmp_path / "out")]) == 2
