"""Settings and strategies (`st`) shared by the fuzz suites (`SUITES` in `tests/scripts/fuzz_campaign.py`).

- **CI** (the default profile): each suite runs its own few hundred examples, and Hypothesis shrinks a failure.
- **Long campaigns** (`HYPOTHESIS_PROFILE=long`): `FUZZ_EXAMPLES` examples per test (default 20,000), no shrinking
  and no example database. Hypothesis can spend over 5 minutes shrinking a livelock before giving up, so a campaign
  saves every failing example as JSON instead (under `FUZZ_FAILURES`, default `results/fuzz/failures`), and
  `uv run python -m tests.scripts.fuzz_campaign --replay FILE` runs it again. `tests/scripts/fuzz_campaign.py` runs
  whole campaigns (suites × seeds) in one command. Rare failures need such counts: the livelock of
  `tests/data/fuzz_regressions/cache_sched-livelock-seed11.json` took 20,000 workloads to show up, where 4 × 3,000
  missed it.
- A seed's draws also depend on which non-test package modules the process has loaded (Hypothesis mines their
  literals), so the campaign's one-test-per-process runs and plandiff's worker are the reproducible ways to replay a
  seed.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import hashlib
import json
import os
import traceback
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
from hypothesis import HealthCheck, Phase, settings, strategies
from hypothesis import event as _event
from hypothesis.errors import InvalidArgument

from .paths import DATA, RESULTS

LONG = os.environ.get("HYPOTHESIS_PROFILE") == "long"
_PLAIN = {bool, float, int, str, type(None)}  # values their repr tells apart (1, 1.0, True, "1"; 0.0 and -0.0)


class _Strategies:
    """The suites' `st` (`from .fuzzing import st`): `hypothesis.strategies`, with `sampled_from`, `one_of` and `none`
    built once per distinct arguments. Hypothesis caches its `integers`, `lists` and `booleans` itself, but builds a
    new `sampled_from`, `one_of` (`a | b`) or `none()` on every call and validates, labels and checks it for emptiness
    when it is drawn from: a quarter of the draw's time. A strategy is a function of its arguments, so a cached one has
    the labels, choices and spans a new one would have: a seed draws byte-for-byte the workloads it drew before
    (`uv run python -m tests.scripts.plandiff --base main --compare-draws`). `sampled_from` caches lists and tuples of
    plain values only (told apart by type as well as value); one of other values (`Spec`s, lists) is built each time, so
    that it draws the caller's own objects, as is any other sequence (a `range` labels its strategy otherwise than the
    equal list) and a `one_of` of unhashable arguments (`one_of([a, b])`). `a | b` builds a new strategy each time:
    write `st.one_of(a, b)`, the same strategy. Any other name is Hypothesis' own."""

    def __init__(self) -> None:
        self._one_of = functools.lru_cache(maxsize=4096)(strategies.one_of)
        self.none = functools.cache(strategies.none)
        self._sampled: dict[tuple[str, ...], strategies.SearchStrategy] = {}

    def __getattr__(self, name: str) -> Any:  # Hypothesis' own, looked up there once
        value = getattr(strategies, name)
        setattr(self, name, value)
        return value

    def one_of(self, *args: Any) -> strategies.SearchStrategy:
        try:
            return self._one_of(*args)
        except TypeError:  # an unhashable argument: built each time
            return strategies.one_of(*args)

    def sampled_from(self, values: Sequence[Any]) -> strategies.SearchStrategy:
        if type(values) not in (list, tuple) or not set(map(type, values)) <= _PLAIN:
            return strategies.sampled_from(values)
        key = tuple(map(repr, values))
        out = self._sampled.get(key)
        if out is None:
            if len(self._sampled) >= 4096:  # sequences of a workload's own values (its predicted output) come and go
                self._sampled.clear()
            out = self._sampled[key] = strategies.sampled_from(values)
        return out


st = _Strategies()

settings.register_profile("long", max_examples=int(os.environ.get("FUZZ_EXAMPLES", "20000")),
                          phases=(Phase.explicit, Phase.reuse, Phase.generate), database=None, deadline=None,
                          print_blob=True)
if LONG:
    settings.load_profile("long")


def fuzz_settings(ci_examples: int) -> settings:
    """`@fuzz_settings(n)`: n examples in CI; the long profile's count and phases in a campaign."""
    common: dict[str, Any] = {"deadline": None,
                              "suppress_health_check": [HealthCheck.too_slow, HealthCheck.data_too_large]}
    return settings(**common) if LONG else settings(max_examples=ci_examples, **common)


def event(name: str) -> None:
    """Hypothesis' `event`, also callable outside a test (a replayed example)."""
    with contextlib.suppress(InvalidArgument):
        _event(name)


def failures_dir() -> Path:
    return Path(os.environ.get("FUZZ_FAILURES") or RESULTS / "fuzz" / "failures")


def _plain(x: Any) -> Any:
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return {f.name: _plain(getattr(x, f.name)) for f in dataclasses.fields(x)}
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x


@contextmanager
def saved_on_failure(suite: str, example: Any) -> Iterator[None]:
    """Run a fuzz example; in a long campaign, save it as JSON if it fails (then re-raise)."""
    try:
        yield
    except Exception as exc:
        if LONG:
            body = _plain(example)
            text = json.dumps(body, sort_keys=True)
            path = failures_dir() / f"{suite}-{hashlib.sha1(text.encode()).hexdigest()[:12]}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"suite": suite, "seed": os.environ.get("FUZZ_SEED"),
                                        "error": f"{type(exc).__name__}: {exc}"[:4000],
                                        "traceback": traceback.format_exc()[-8000:], "example": body}, indent=1))
        raise


REGRESSIONS = DATA / "fuzz_regressions"  # saved failures kept as tests


def run_example(suite: str, example: Any) -> None:
    if suite in ("cache_sched", "cache_pressure"):
        from .workloads import run_cache_workload as run
    elif suite == "radix":
        from .radix_model import run_scenario as run
    elif suite in ("mtp", "mtp_counts"):
        from .mtp_workloads import run_mtp_workload as run
    elif suite == "images":
        from .workloads import run_image_workload as run
    elif suite == "structured":
        from .structured_cases import run_structured_workload as run
    elif suite == "compile_async":
        from .compile_cases import run_async_workload as run
    elif suite in ("window", "window_pressure", "arena_pressure", "arena_hits"):
        from .window_workloads import run_window_workload as run
    elif suite == "gemma_images":
        from .window_workloads import run_gemma_image_workload as run
    elif suite in ("precise", "window_precise"):
        from .precise_cases import run_precise_workload as run
    elif suite in ("logprobs", "logprobs_counts"):
        from .logprob_workload import run_logprobs_workload as run
    elif suite == "nan":
        from .nan_cases import run_nan_workload as run
    elif suite in ("structured_window", "structured_hybrid", "structured_window_counts", "structured_hybrid_counts"):
        from .forced_cases import run_switches_workload as run
    elif suite in ("window_mtp", "arena_mtp", "arena_mtp_counts", "gemma_images_mtp"):
        from .window_mtp import run_window_mtp_workload

        def run(ex: Any) -> None:
            run_window_mtp_workload(ex, bidirectional=suite == "gemma_images_mtp")
    else:
        from .workloads import run_workload as run
    run(example)


def load_example(path: str | Path) -> tuple[str, Any]:
    """A saved failure: (suite, example rebuilt as the test receives it)."""
    from .harness import Spec

    data = json.loads(Path(path).read_text())
    suite, ex = data["suite"], data["example"]
    if "specs" in ex:
        ex["specs"] = [Spec(**s) for s in ex["specs"]]
    if isinstance(ex.get("workload"), dict) and "specs" in ex["workload"]:  # a logprobs example wraps another suite's
        ex["workload"]["specs"] = [Spec(**s) for s in ex["workload"]["specs"]]
    for r in ex.get("reqs", ()):
        r["tokens"] = np.asarray(r["tokens"], np.int64)
    if "num_window_pages" in ex:  # separate window and full pools: one arena of both, its pages of one size
        ex["window_page_size"] = ex["page_size"]
        ex["num_pages"] += ex.pop("num_window_pages")
    if "mixed" in ex:  # a fixed mixed-step size: the rows x prefill tokens bound with the same tokens per decode row
        ex["row_tokens"] = ex.pop("mixed")
    # How a refine is stepped: in fixed pieces in mixed steps only (`refine_chunking` "fixed", `refine_steps` "mixed").
    # A workload that names these settings replays without them; one that names neither (its refines cut like any
    # chunk) replays in fixed pieces (the regressions that refine pin `precise_cap` to their chunks). Another value is
    # refused only where it acts, a `reuse` refine: elsewhere it changes nothing (`bf16` and `fp8` never refine), and
    # only its piece may not fit the room check, which then unsets it (the piece is unused there)
    reuse = ex.get("precise", "fp8") == "reuse"
    for key, kept in (("refine_chunking", "fixed"), ("refine_steps", "mixed")):
        value = ex.pop(key, kept)
        if value != kept and reuse:
            raise ValueError(f"{path}: {key} {value!r} is not supported (only {kept!r} is)")
    room = ex.get("max_tokens", 64) - ex.get("max_num_seqs", 4) * (1 + ex.get("mtp_k", 0))
    if not reuse and ex.get("refine_chunk_tokens", 0) > room:
        ex["refine_chunk_tokens"] = 0
    return suite, ex


def nan_write(draw: Any, prompt_len: int) -> dict[str, Any]:
    """The NaN fuzz dimension (`tests/sched/test_nan_safety.py`), as `Spec` fields: about one request in six has a
    prefill chunk write NaN at a prompt position (`nonfinite_write`), half of them a transient fault (`nonfinite_once`:
    the request computing that position again, after a readmission, gets it finite); its next SAMPLE row reports it, or
    the chunk itself under `report_unsampled` (drawn per workload: `nan_report`)."""
    if draw(st.integers(0, 5)):
        return {}
    return {"nonfinite_write": draw(st.integers(0, prompt_len - 1)), "nonfinite_once": draw(st.booleans())}


def nan_report(draw: Any) -> bool:
    """`Harness(report_unsampled=)`: a runner that also reports NONFINITE on rows that do not sample, in one
    workload of four; Qwen's CUDA runner reports it on SAMPLE rows only, Gemma 4's on its explicit rows too."""
    return draw(st.integers(0, 3)) == 0
