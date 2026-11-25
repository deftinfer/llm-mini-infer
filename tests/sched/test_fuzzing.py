"""The fuzz suites' tools: `fuzzing.st` builds each strategy once per arguments and draws what Hypothesis' own
strategies draw; `FUZZ_CAP` caps the campaign's suites alone; `tests/scripts/fuzz_campaign.py` starts the longest runs
first."""

from __future__ import annotations

import json
import re

import pytest
from hypothesis import HealthCheck, Phase, given, seed, settings, strategies

from ..scripts.fuzz_campaign import SUITES, history, longest_first
from ..support import fuzzing
from ..support.fuzzing import campaign_settings, st
from ..support.harness import Spec
from ..support.paths import ROOT


def test_strategies_are_built_once_per_arguments() -> None:
    builds = (lambda s: s.sampled_from([1, 2, 4]), lambda s: s.none(), lambda s: s.one_of(s.none(), s.integers(0, 3)))
    for build in builds:  # Hypothesis builds these anew on every call
        assert build(strategies) is not build(strategies) and build(st) is build(st)
    assert st.integers is strategies.integers  # any other name is Hypothesis' own (it caches integers itself)
    # told apart by type, not only by value (1 == True == 1.0, 0.0 == -0.0)
    assert st.sampled_from([1, 0]) is not st.sampled_from([True, False])
    assert st.sampled_from([0.0, 1]) is not st.sampled_from([-0.0, 1]) is not st.sampled_from([0, 1])
    specs = [Spec(rid=i, prompt_len=4, max_tokens=2, seed=i) for i in range(2)]
    assert st.sampled_from(specs) is not st.sampled_from(specs)  # mutable values: built each time
    assert st.sampled_from([[1], [2]]) is not st.sampled_from([[1], [2]])
    assert st.sampled_from(range(3)) is not st.sampled_from(range(3))  # labelled otherwise than [0, 1, 2]
    pair = [st.none(), st.integers(0, 3)]
    assert st.one_of(pair) is not st.one_of(pair)  # unhashable: built each time
    # Hypothesis' own strategies, of the same labels
    assert st.sampled_from([None, None, "s"]).label == strategies.sampled_from([None, None, "s"]).label
    assert st.one_of(*pair).label == st.one_of(pair).label == (strategies.none() | strategies.integers(0, 3)).label


def _draws(s) -> list:
    @st.composite
    def workloads(draw):
        n = draw(s.integers(1, 6))
        return [(draw(s.sampled_from([0, 0, 1, 40])), draw(s.one_of(s.none(), s.integers(0, n))),
                 draw(s.lists(s.integers(1, n), max_size=2)), draw(s.booleans()), draw(s.sampled_from(["a", "b"])))
                for _ in range(n)]

    drawn = []

    @seed(5)
    @settings(max_examples=150, phases=(Phase.generate,), database=None, deadline=None,
              suppress_health_check=list(HealthCheck))
    @given(workloads())
    def record(w) -> None:
        drawn.append(w)

    record()
    return drawn


def test_a_seed_draws_what_hypothesis_own_strategies_draw() -> None:
    """The same values, and through the generation phase's mutations the same test cases (which reuse spans of equal
    labels)."""

    class Own:  # `st` without the caching, `a | b` for `one_of`
        integers, sampled_from, lists, booleans, none = (strategies.integers, strategies.sampled_from,
                                                         strategies.lists, strategies.booleans, strategies.none)

        @staticmethod
        def one_of(a, b):
            return a | b

    mine, own = _draws(st), _draws(Own)
    assert mine == own and len(mine) == 150


def test_the_cap_reaches_the_campaign_suites_alone(monkeypatch) -> None:
    """`FUZZ_CAP` caps each test of `SUITES` (`campaign_settings`), which CI's fuzz job runs at a campaign's counts; a
    fuzz test outside them keeps its count, since CI's tests job is the one place it runs."""
    monkeypatch.setattr(fuzzing, "LONG", False)  # a campaign's profile sets its own count
    monkeypatch.setenv("FUZZ_CAP", "7")
    assert campaign_settings(300).max_examples == 7 and campaign_settings(5).max_examples == 5
    monkeypatch.delenv("FUZZ_CAP")
    assert campaign_settings(300).max_examples == 300
    capped = {f"{p.relative_to(ROOT).as_posix()}::{name}" for p in sorted((ROOT / "tests").rglob("test_*.py"))
              for name in re.findall(r"^@campaign_settings\(\d+\)\n(?:.*\n)*?def (\w+)\(", p.read_text(), re.M)}
    assert capped == set(SUITES.values())


def test_the_campaign_orders_runs_by_the_latest_campaign_of_each_suite(tmp_path) -> None:
    def campaign(name: str, examples: int, runs: list[tuple[str, str, float]]) -> None:
        (tmp_path / name).mkdir()
        (tmp_path / name / "summary.json").write_text(json.dumps({"examples": examples, "runs": [
            {"suite": s, "status": status, "seconds": secs} for s, status, secs in runs]}))

    campaign("20261020-100000", 1000, [("radix", "passed", 10.0), ("window", "passed", 40.0)])
    campaign("20261021-100000", 100, [("window", "passed", 2.0), ("window", "passed", 3.0), ("mtp", "FAILED", 1.0)])
    (tmp_path / "20261022-100000").mkdir()  # a campaign still running: no summary yet
    (tmp_path / "failures").mkdir()
    est = history(tmp_path)
    assert est == {"window": pytest.approx(0.025), "radix": pytest.approx(0.01)}  # a failed run's time says nothing
    runs = [("radix", 1), ("window", 1), ("mtp", 1), ("radix", 2), ("window", 2)]
    assert longest_first(runs, est) == [2, 1, 4, 0, 3]  # mtp unknown, then window, then radix; seeds in order
