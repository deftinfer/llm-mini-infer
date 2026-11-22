"""Long fuzz campaigns in one command (see `tests/support/fuzzing.py`).

    uv run python -m tests.scripts.fuzz_campaign                   # 27 suites × seeds 11, 22, 33, 44 × 20,000 workloads
    uv run python -m tests.scripts.fuzz_campaign --examples 2000 --seeds 1 2 --suites cache_sched
    uv run python -m tests.scripts.fuzz_campaign --replay results/fuzz/<time>/failures/cache_sched-0123abcd4567.json

Each (suite, seed) runs as its own pytest process under `HYPOTHESIS_PROFILE=long`, in parallel, the longest first
(each suite's seconds per workload in the latest campaign that ran it, from `results/fuzz/*/summary.json`; a
suite without one first), so that no long run starts last and keeps the campaign waiting. Logs and the summary go to
`results/fuzz/<time>/` (results in suite and seed order), failing examples to its `failures/`. The exit status
is 1 if any run failed or timed out.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..support.paths import RESULTS, ROOT

SUITES = {
    "cache_sched": "tests/sched/test_cache_sched.py::test_random_cache_workloads",
    "cache_pressure": "tests/sched/test_cache_sched.py::test_random_pressure_pairs",
    "radix": "tests/sched/test_radix.py::test_random_requests_keep_every_invariant",
    "scheduler": "tests/sched/test_scheduler.py::test_random_workloads",
    "mtp": "tests/sched/test_mtp_sched.py::test_random_mtp_workloads",
    "images": "tests/sched/test_image_sched.py::test_random_image_workloads",
    "structured": "tests/structured/test_structured.py::test_random_structured_workloads",
    "compile_async": "tests/structured/test_compile_async.py::test_random_compile_async_workloads",
    "window": "tests/sched/test_window_cache.py::test_random_window_workloads",
    "window_pressure": "tests/sched/test_window_cache.py::test_random_window_pressure",
    "arena_pressure": "tests/sched/test_window_cache.py::test_random_arena_pressure",
    "arena_hits": "tests/sched/test_window_cache.py::test_random_arena_hits",
    "gemma_images": "tests/sched/test_gemma_spans.py::test_random_gemma_image_workloads",
    "window_mtp": "tests/sched/test_window_mtp.py::test_random_window_mtp_workloads",
    "arena_mtp": "tests/sched/test_window_mtp.py::test_random_arena_mtp_workloads",
    "gemma_images_mtp": "tests/sched/test_window_mtp.py::test_random_gemma_images_mtp_workloads",
    "precise": "tests/sched/test_precise_prefix.py::test_random_precise_workloads",
    "window_precise": "tests/sched/test_precise_prefix.py::test_random_window_precise_workloads",
    "nan": "tests/sched/test_nan_safety.py::test_random_nan_workloads",
    "structured_window": "tests/sched/test_forced_tokens.py::test_random_structured_window_workloads",
    "structured_hybrid": "tests/sched/test_forced_tokens.py::test_random_structured_hybrid_workloads",
    "logprobs": "tests/runner/test_logprobs.py::test_random_logprobs_workloads",
    "mtp_counts": "tests/sched/test_mtp_sched.py::test_random_mtp_count_workloads",
    "arena_mtp_counts": "tests/sched/test_window_mtp.py::test_random_arena_mtp_count_workloads",
    "structured_window_counts": "tests/sched/test_forced_tokens.py::test_random_structured_window_count_workloads",
    "structured_hybrid_counts": "tests/sched/test_forced_tokens.py::test_random_structured_hybrid_count_workloads",
    "logprobs_counts": "tests/runner/test_logprobs.py::test_random_logprobs_count_workloads",
}
_STATS = re.compile(r"(\d+) passing(?: examples?)?, (\d+) failing(?: examples?)?,(?: and)? (\d+) invalid")


def run_one(suite: str, seed: int, examples: int, out: Path, timeout: float) -> dict:
    log = out / f"{suite}-seed{seed}.log"
    env = {**os.environ, "HYPOTHESIS_PROFILE": "long", "FUZZ_EXAMPLES": str(examples), "FUZZ_SEED": str(seed),
           "FUZZ_FAILURES": str(out / "failures")}
    cmd = [sys.executable, "-m", "pytest", SUITES[suite], "-q", "-p", "no:cacheprovider", f"--hypothesis-seed={seed}",
           "--hypothesis-show-statistics"]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout)
        text, status = proc.stdout + proc.stderr, ("passed" if proc.returncode == 0 else "FAILED")
    except subprocess.TimeoutExpired as exc:
        text = (exc.stdout or b"").decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        status = "TIMEOUT"
    secs = time.perf_counter() - t0
    log.write_text(text)
    passing = sum(int(m.group(1)) for m in _STATS.finditer(text))
    failing = sum(int(m.group(2)) for m in _STATS.finditer(text))
    return {"suite": suite, "seed": seed, "status": status, "seconds": round(secs, 1), "passing": passing,
            "failing": failing, "log": str(log.relative_to(ROOT))}


def history(results: Path) -> dict[str, float]:
    """Each suite's seconds per workload in the latest campaign under `results` that ran it (its passed runs)."""
    out: dict[str, float] = {}
    for path in sorted(results.glob("*/summary.json"), reverse=True):  # <time> directories: the latest first
        try:
            summary = json.loads(path.read_text())
            per: dict[str, list[float]] = {}
            for r in summary["runs"]:
                if r["status"] == "passed":
                    per.setdefault(r["suite"], []).append(r["seconds"] / summary["examples"])
        except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError):
            continue
        for suite, secs in per.items():
            out.setdefault(suite, sum(secs) / len(secs))
    return out


def longest_first(runs: list[tuple[str, int]], est: dict[str, float]) -> list[int]:
    """The indices of `runs` in the order to start them: suites without an estimate (`history`) first, then the
    longest first."""
    return sorted(range(len(runs)), key=lambda i: (runs[i][0] in est, -est.get(runs[i][0], 0.0)))


def replay(path: str) -> int:
    from ..support.fuzzing import load_example, run_example

    suite, ex = load_example(path)
    print(f"replaying a {suite} example from {path}")
    run_example(suite, ex)
    print("passed (to keep it as a regression test, copy it to tests/data/fuzz_regressions/)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suites", nargs="+", choices=sorted(SUITES), default=list(SUITES))
    ap.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33, 44])
    ap.add_argument("--examples", type=int, default=20_000, help="workloads per (suite, seed)")
    ap.add_argument("--jobs", type=int, default=0, help="parallel runs (default: one per run, up to the CPU count)")
    ap.add_argument("--timeout", type=float, default=4 * 3600, help="seconds per run")
    ap.add_argument("--replay", metavar="FILE", help="run one saved failing example and exit")
    args = ap.parse_args()
    if args.replay:
        return replay(args.replay)
    out = RESULTS / "fuzz" / time.strftime("%Y%m%d-%H%M%S")
    est = history(out.parent)
    out.mkdir(parents=True)
    runs = [(s, seed) for s in args.suites for seed in args.seeds]
    jobs = args.jobs or min(len(runs), os.cpu_count() or 1)
    print(f"{len(runs)} runs × {args.examples:,} workloads, {jobs} at a time; logs in {out.relative_to(ROOT)}")
    t0 = time.perf_counter()
    with ThreadPoolExecutor(jobs) as pool:  # the pool starts them in the order submitted
        started = {i: pool.submit(run_one, *runs[i], args.examples, out, args.timeout)
                   for i in longest_first(runs, est)}
        results = []
        for i in range(len(runs)):
            r = started[i].result()
            results.append(r)
            print(f"  {r['suite']:<12} seed {r['seed']:<6} {r['status']:<8} {r['passing']:>7,} passing "
                  f"{r['failing']:>3} failing  {r['seconds']:>8.1f} s", flush=True)
    wall = time.perf_counter() - t0
    summary = {"examples": args.examples, "wall_seconds": round(wall, 1), "runs": results}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    bad = [r for r in results if r["status"] != "passed"]
    print(f"{len(results) - len(bad)}/{len(results)} passed in {wall / 60:.1f} min wall"
          + (f"; failures in {out.relative_to(ROOT)}/failures" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
