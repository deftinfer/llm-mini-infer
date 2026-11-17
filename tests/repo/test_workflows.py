"""The GitHub Actions workflows: every action pinned by commit, and ci.yml's triggers, permissions and checks (lint, the
chart, the CPU suite, tools/'s tests, the fuzz smoke)."""

from __future__ import annotations

import re
import subprocess

import yaml

from ..support.paths import RESULTS, ROOT

WORKFLOWS = ROOT / ".github" / "workflows"


def load(name: str) -> dict:
    wf = yaml.safe_load((WORKFLOWS / name).read_text())
    wf["on"] = wf.pop(True, wf.get("on"))  # YAML 1.1 reads `on` as true
    return wf


def test_every_action_is_pinned_by_commit() -> None:
    uses = [u for p in sorted(WORKFLOWS.glob("*.yml")) for u in re.findall(r"uses:\s*(\S+)", p.read_text())]
    assert uses
    for u in uses:
        assert re.fullmatch(r"[\w.-]+/[\w./-]+@[0-9a-f]{40}", u), u


def test_ci_runs_on_main_pull_requests_and_dispatch_and_only_reads() -> None:
    wf = load("ci.yml")
    # every pull request runs every job: a path filter would leave the required check pending
    assert wf["on"] == {"push": {"branches": ["main"], "paths-ignore": ["docs/**", "**/*.md"]}, "pull_request": None,
                        "workflow_dispatch": None}
    assert wf["permissions"] == {"contents": "read"}
    assert set(wf["jobs"]) == {"lint", "chart", "tests", "tools", "changes", "fuzz", "required"}
    fuzz = wf["jobs"]["fuzz"]
    assert fuzz["needs"] == "changes" and fuzz["if"] == "needs.changes.outputs.fuzz == 'true'"
    for name, job in wf["jobs"].items():
        if name == "required":  # it checks out nothing (test_the_required_check_waits_for_every_job)
            continue
        checkout = next(s for s in job["steps"] if s.get("uses", "").startswith("actions/checkout@"))
        assert checkout["with"]["persist-credentials"] is False


def test_the_required_check_waits_for_every_job() -> None:
    """main's ruleset requires the check `required`: it needs every other job and runs after a failure too. It passes
    only when every other job passed, and the fuzz smoke also when it was skipped (a change without code)."""
    jobs = load("ci.yml")["jobs"]
    job = jobs["required"]
    assert set(job["needs"]) == set(jobs) - {"required"}
    assert job["if"] == "always()"
    assert not any("uses" in s for s in job["steps"])  # no checkout, no action: only the results
    (step,) = job["steps"]
    assert step["env"] == {name.upper(): f"${{{{ needs.{name}.result }}}}" for name in job["needs"]}

    def passes(**results: str) -> bool:
        env = {name.upper(): results.get(name, "success") for name in job["needs"]}
        return subprocess.run(["/bin/sh", "-c", step["run"]], env=env, capture_output=True).returncode == 0

    assert passes() and passes(fuzz="skipped")
    for name in job["needs"]:
        for result in ("failure", "cancelled", *(() if name == "fuzz" else ("skipped",))):
            assert not passes(**{name: result}), (name, result)


def test_ci_runs_the_checks() -> None:
    jobs = load("ci.yml")["jobs"]
    run = {name: "\n".join(s.get("run", "") for s in job["steps"]) for name, job in jobs.items()}
    assert "uvx \"ruff==$ruff\" check ." in run["lint"]  # uv.lock's ruff
    chart = run["chart"]
    assert "uv run tools/fetch_tools.py helm kubeconform" in chart
    assert all(k in chart for k in ("helm lint", "helm template", "kubeconform -strict"))
    assert "-f tests/data/chart-values.yaml" in chart and "deploy/helm/examples/*.yaml" in chart
    # the CRD catalogue at a commit, not a moving branch
    assert re.search(r"datreeio/CRDs-catalog/[0-9a-f]{40}/", chart), "the CRD catalogue is not pinned to a commit"
    assert "uv sync --locked" in run["tests"] and "tests/scripts/fetch_fixtures.py" in run["tests"]
    # tools/ is its own project: its own lock, environment and tests
    assert "uv sync --project tools --locked" in run["tools"]
    assert "uv run --project tools --locked pytest tools/tests -q -p no:cacheprovider" in run["tools"]
    setup = next(s for s in jobs["tools"]["steps"] if s.get("uses", "").startswith("astral-sh/setup-uv@"))
    assert setup["with"]["cache-dependency-glob"] == "tools/uv.lock"
    assert "python -m tests.scripts.fuzz_campaign --examples 500 --seeds 1 2 --jobs 4" in run["fuzz"]
    assert 'python3 .github/scripts/ci_changes.py "$BASE" "$GITHUB_SHA" >> "$GITHUB_OUTPUT"' in run["changes"]
    checkout = next(s for s in jobs["changes"]["steps"] if s.get("uses", "").startswith("actions/checkout@"))
    assert checkout["with"]["fetch-depth"] == 0  # the base commit is in the history
    # a failure's artifacts are where the suites write them (tests/support/paths.py): a wrong path uploads nothing
    upload = "actions/upload-artifact@"
    uploads = {name: [s["with"]["path"] for s in job["steps"] if s.get("uses", "").startswith(upload)]
               for name, job in jobs.items()}
    results = RESULTS.relative_to(ROOT).as_posix()
    assert uploads == {"lint": [], "chart": [], "tests": [f"{results}/fuzz/failures/"], "tools": [], "changes": [],
                       "fuzz": [f"{results}/fuzz/"], "required": []}


def test_ci_shards_the_cpu_suite_over_all_of_tests() -> None:
    """The CPU suite runs in shards, each in pytest-xdist's workers by file: the named shards take test directories
    apart, and the last takes tests/ without them, so every test runs in exactly one shard."""
    job = load("ci.yml")["jobs"]["tests"]
    assert job["strategy"]["fail-fast"] is False  # one shard's failure leaves the others their results
    *named, rest = job["strategy"]["matrix"]["include"]
    assert len({s["shard"] for s in [*named, rest]}) == len(named) + 1
    taken = [p for s in named for p in s["paths"].split()]
    assert len(set(taken)) == len(taken) and all(p.startswith("tests/") and (ROOT / p).is_dir() for p in taken)
    assert rest["paths"].split() == ["tests", *(f"--ignore={p}" for p in taken)]
    (step,) = [s for s in job["steps"] if s.get("name") == "pytest"]
    assert step["env"] == {"PATHS": "${{ matrix.paths }}"}
    # a hung run ends at `timeout`'s limits, inside the step, so the step's log is kept: the two limits, each with
    # its minute to kill and its sweep, end under the step's limit, and that under the job's
    assert step["timeout-minutes"] < job["timeout-minutes"]
    pytest = "uv run --locked pytest -q -p no:cacheprovider -ra -o faulthandler_timeout=300"
    lines = [line.strip() for line in step["run"].splitlines()]
    assert lines == [
        "status=0",
        'stop="$RUNNER_TEMP/sample-stop" dump="$RUNNER_TEMP/sample-dump-at"',
        # the memory sampler, beside the tests, on the runner's own Python
        'python3 .github/scripts/ci_sample.py --watch $$ --stop "$stop" --dump-at "$dump" &',
        "guarded() {  # a limit in seconds, then pytest's arguments",
        'echo $(( $(date +%s) + $1 - 30 )) > "$dump"',
        'timeout --kill-after=60 "$1" \\',
        f'{pytest} "${{@:2}}" && return',
        "code=$?",
        '[ $code -lt 124 ] || echo "::error::pytest ran into its limit of $1 s (exit $code): see [hang]"',
        # the servers a killed pytest left
        'timeout 30 uv run --locked python -c "from tests.support.api_util import sweep_stale; sweep_stale()"',
        "return $code",
        "}",
        # the workers, then the serial tests alone; both run, and either one's failure fails the step
        'guarded 840 -n 2 --dist loadfile -m "not serial" $PATHS || status=$?',
        "guarded 480 -m serial $PATHS || [ $? -eq 5 ] || status=1",
        'touch "$stop"',
        "wait",
        "exit $status",
    ]
    # each call's limit, its minute to kill and its sweep
    limits = [int(line.split()[1]) + 60 + 30 for line in lines if re.match(r"guarded \d+ ", line)]
    assert len(limits) == 2 and sum(limits) < step["timeout-minutes"] * 60
    assert (WORKFLOWS.parent / "scripts" / "ci_sample.py").is_file()
    assert "\ndef sweep_stale() -> None:" in (ROOT / "tests" / "support" / "api_util.py").read_text()
    # each shard's artifact has its own name: a second upload under one name fails
    (upload,) = [s for s in job["steps"] if s.get("uses", "").startswith("actions/upload-artifact@")]
    assert upload["with"]["name"] == "fuzz-failures-${{ matrix.shard }}"
