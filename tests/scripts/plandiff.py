"""The plan-diff harness (docs/development.md, "The plan-diff harness"): every scheduler decision the fuzz workloads
make, digested per launch, compared between two trees, joined on each example's hash.

    uv run python -m tests.scripts.plandiff --base main                 # the working tree against main
    uv run python -m tests.scripts.plandiff --base main --new HEAD      # two revisions
    uv run python -m tests.scripts.plandiff --base main --suites scheduler mtp --examples 200
    uv run python -m tests.scripts.plandiff --base main --expect-changed structured,structured_window
    uv run python -m tests.scripts.plandiff --base main --compare-draws # a change to the strategies: the same workloads

**What it compares.** Each fuzz suite of `tests/scripts/fuzz_campaign.py` (`SUITES`) drives the scheduler, its engine
turn (`EngineCore.step`) and the strict fake runner over random workloads. For each workload the harness digests, in
order: every `StepPlan` the runner receives (`launch_forward`: every field, values in canonical dtypes, empty arrays and
zero scalars left out, so a field a newer tree adds and leaves empty changes nothing, and so is each ADMIT term that
every admission carries when all its values are neutral, `NEUTRAL`), every set of grammar masks the
engine fills (`launch_sample`: their count and words), and every `StepReport` a turn returns (each frontend's outputs,
its timing fields, the names ending in `_ns`, and the per-request verify counts, `COUNTS`, left out), then the example's
outcome (passed, or the exception's type). The radix suite drives `RadixCache` without a scheduler: its digest is every
match's hit and the pools after each check.
Two trees agree on an example when the digests are equal; an example whose digest differs is a changed decision, and
the report names its first differing event and fields (a second run of that example in both trees, kept in full).

**The same workloads in both trees.** The base tree draws each suite's workloads from its own Hypothesis strategy
with a fixed seed (`--seed`), its test body replaced by a recorder (`hypothesis.inner_test`), serializes them as the
fuzz campaign saves a failure (`fuzzing._plain`, JSON), and both trees run them back through their own
`fuzzing.load_example` and `run_example`: the same inputs by construction, joined on the hash of each one's JSON, as
`fuzzing.saved_on_failure` names a failure. A suite only the new tree has is drawn there and reported, not compared.
With `--compare-draws` the other tree draws them too, from its own strategies: a seed must draw the same workloads in
both (campaigns stay comparable across trees), so a workload drawn differently (types included: a tuple is not a list,
an int32 array not an int64 one), a test case the Hypothesis engine ran differently (its status, choices and spans), or
other counts of its test cases (calls, valid, invalid, overrun, misaligned) fails.

**Isolation.** A revision is unpacked with `git archive` into a temporary directory (the working tree is the default
new side, uncommitted changes included). Workers run one per (tree, suite) as `python -c` with the tree first on
`sys.path` (the venv's editable `.pth` entry comes after it), `PYTHONHASHSEED=0` (set and dict orders of strings), and
this file executed by path: it imports nothing of `tests` or `llm_mini_infer` at module level, so a worker runs the tree
it points at. A tree may have either layout of `tests/`: the worker imports the fuzzing module, the campaign's suites
and the radix suite's model from `tests/support/` and `tests/scripts/`, or, in a tree from before them, from `tests/`
itself. The git-ignored fixtures a tree's tests read (`tests/.fixtures`) are linked from the working tree into an
unpacked one.

**Exit status:** 0 when every compared example is identical (or every changed one is in a suite `--expect-changed`
names), 1 otherwise (also when every base workload of a suite errors: nothing was compared), 2 when a worker
failed. Results land in `results/plandiff/<time>/` (`report.md`, `summary.json`, each tree's digests and the base's
examples).
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WORKER = Path(__file__).resolve()
DIGEST_BYTES = 12
SEED = 6  # the default draw (any fixed value: both trees run what the base drew)
EXAMPLES = 1000  # per suite; the campaign's 20,000 in a few minutes on 16 cores
LINKED = ("tests/.fixtures",)  # git-ignored fixtures some suites' modules look for
SHOW = 3  # changed examples per suite described field by field
TRUNC = 24  # values shown per differing field
# Plan fields set for every ADMIT (the penalties, the logit bias's count): each left out when all its values are the
# neutral one, so a plan without penalties digests as in a tree from before them; their other arrays are empty then
NEUTRAL = {"admit_rep": 1.0, "admit_pres": 0.0, "admit_freq": 0.0, "admit_prompt_len": 0, "admit_bias_n": 0}


# -- canonical events (the worker side; the tree's own classes, duck-typed) --------------------------------------------


def _either(new: str, old: str) -> Any:
    """The module `new`, or `old` in a tree without it (a revision from before tests/support/ and tests/scripts/): a
    base may have either layout. Only `new`'s own absence falls back; an error inside it shows."""
    import importlib

    parts = new.split(".")
    try:
        return importlib.import_module(new)
    except ModuleNotFoundError as exc:
        if exc.name not in {".".join(parts[:i]) for i in range(2, len(parts) + 1)}:
            raise
    return importlib.import_module(old)


def _canon_array(a: Any) -> np.ndarray:
    """Values in a canonical dtype (integers and booleans as int64, floats as float64), so a field whose storage
    dtype changes but whose values do not digests the same."""
    a = np.asarray(a)
    if a.dtype == np.bool_ or np.issubdtype(a.dtype, np.integer):
        return a.astype(np.int64)
    if np.issubdtype(a.dtype, np.floating):
        return a.astype(np.float64)
    return a


def _fields(obj: Any, names: list[str]) -> list[tuple[str, Any]]:
    """(name, value) of the set fields: arrays with elements (not all of them NEUTRAL), non-zero scalars, non-empty
    sequences; `*_ns` timings never."""
    out = []
    for name in sorted(names):
        if name.endswith("_ns"):
            continue
        v = getattr(obj, name, None)
        if v is None:
            continue
        if isinstance(v, np.ndarray):
            if v.size and not (name in NEUTRAL and bool((v == NEUTRAL[name]).all())):
                out.append((name, _canon_array(v)))
        elif isinstance(v, (bool, int, float, np.generic)):
            if v:
                out.append((name, _canon_scalar(v)))
        elif isinstance(v, (list, tuple)):
            if len(v):
                try:
                    a = _canon_array(np.asarray(v))
                except ValueError:  # ragged
                    a = None
                out.append((name, a if a is not None and a.dtype != object else repr(v)))
        elif isinstance(v, dict):
            if v:
                out.append((name, v))
        else:
            out.append((name, repr(v)))
    return out


def _canon_scalar(v: Any) -> int | float:
    """A scalar's value in a canonical type (a bool or any integer an int, any float a float), as `_canon_array`
    does for arrays: a storage change (a numpy integer for an int) changes no digest."""
    if isinstance(v, (bool, int, np.bool_, np.integer)):
        return int(v)
    return float(v)


def _slot_names(obj: Any) -> list[str]:
    names: list[str] = []
    for cls in type(obj).__mro__:
        names += [n for n in getattr(cls, "__slots__", ()) if not n.startswith("_")]
    if not names and hasattr(obj, "__dataclass_fields__"):
        names = list(obj.__dataclass_fields__)
    if not names:
        names = [n for n in vars(obj) if not n.startswith("_")]
    return names


def plan_event(plan: Any) -> tuple[str, list[tuple[str, Any]]]:
    return "plan", _fields(plan, _slot_names(plan))


def masks_event(n: int, rows: np.ndarray) -> tuple[str, list[tuple[str, Any]]]:
    return "masks", [("n", int(n)), ("words", _canon_array(rows[:n]))]


# A frontend's outputs that count, not decide: the verify rows and tokens each finishing request reports (MTP)
COUNTS = frozenset(("spec_rids", "spec_rows", "spec_tokens"))


def report_event(report: Any) -> tuple[str, list[tuple[str, Any]]]:
    fields = [(n, v) for n, v in _fields(report, _slot_names(report)) if n != "outputs"]
    for fe in sorted(report.outputs):
        out = report.outputs[fe]
        fields += [(f"fe{fe}.{n}", v) for n, v in _fields(out, _slot_names(out)) if n not in COUNTS]
    return "report", fields


def _update(h: Any, v: Any) -> None:
    if isinstance(v, np.ndarray) and v.dtype == object:  # its bytes are pointers: digest the values
        h.update(repr(v.tolist()).encode())
    elif isinstance(v, np.ndarray):
        h.update(f"{v.dtype.str}{v.shape}".encode())
        h.update(np.ascontiguousarray(v).tobytes())
    elif isinstance(v, dict):
        h.update(json.dumps(v, sort_keys=True, default=str).encode())
    else:
        h.update(repr(v).encode())


def event_digest(kind: str, fields: list[tuple[str, Any]]) -> bytes:
    h = hashlib.blake2b(kind.encode(), digest_size=DIGEST_BYTES)
    for name, v in fields:
        h.update(b"\0" + name.encode() + b"=")
        _update(h, v)
    return h.digest()


def _plain_value(v: Any) -> Any:
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    return v


class Recorder:
    """One example's events: their digests, and (detail mode) each event in full."""

    def __init__(self) -> None:
        self.digests: list[bytes] = []
        self.full: list[dict] | None = None

    def reset(self, detail: bool = False) -> None:
        self.digests = []
        self.full = [] if detail else None

    def add(self, event: tuple[str, list[tuple[str, Any]]]) -> None:
        kind, fields = event
        self.digests.append(event_digest(kind, fields))
        if self.full is not None:
            self.full.append({"kind": kind, "fields": {n: _plain_value(v) for n, v in fields}})


REC = Recorder()


def install_hooks(suite: str) -> None:
    """Wrap the classes the suites drive (`FakeRunner`, `EngineCore`; the radix suite's `RadixCache` and model) so
    every decision goes through `REC`, in the tree this worker imported."""
    from llm_mini_infer.engine.core import EngineCore
    from llm_mini_infer.runner.fake import FakeRunner

    fwd, smp, step = FakeRunner.launch_forward, FakeRunner.launch_sample, EngineCore.step

    def launch_forward(self, plan, *a, **kw):
        REC.add(plan_event(plan))
        return fwd(self, plan, *a, **kw)

    def launch_sample(self, handle, n_masks=0, *a, **kw):
        if n_masks:
            REC.add(masks_event(n_masks, self.mask_buffer(handle)))
        return smp(self, handle, n_masks, *a, **kw)

    def engine_step(self, *a, **kw):
        emit = kw.get("emit") if "emit" in kw else (a[0] if a else None)
        if emit is not None:  # the process loop's form: the outputs go to `emit`
            def emit_rec(report, _emit=emit):
                REC.add(report_event(report))
                return _emit(report)

            if "emit" in kw:
                kw["emit"] = emit_rec
            else:
                a = (emit_rec, *a[1:])
        report = step(self, *a, **kw)
        if report is not None:
            REC.add(report_event(report))
        return report

    FakeRunner.launch_forward, FakeRunner.launch_sample, EngineCore.step = launch_forward, launch_sample, engine_step
    if suite == "radix":
        from llm_mini_infer.cache.radix import RadixCache

        model = _either("tests.support.radix_model", "tests.test_radix").Model
        match, check = RadixCache.match, model.check

        def radix_match(self, *a, **kw):
            m = match(self, *a, **kw)
            REC.add(("match", [("hit", int(m.hit)), ("kv_len", int(getattr(m, "kv_len", 0)))]))
            return m

        def model_check(self):
            check(self)
            REC.add(("pools", [("refcount", _canon_array(self.kv.refcount)), ("free", int(self.kv.num_free)),
                               ("state_used", _canon_array(self.states.used)),
                               ("state_kind", _canon_array(self.states.kind))]))

        RadixCache.match, model.check = radix_match, model_check


# -- the worker: one tree, one suite ----------------------------------------------------------------------------------


def _example_hash(plain: Any) -> str:
    return hashlib.sha1(json.dumps(plain, sort_keys=True).encode()).hexdigest()[:16]


def _typed(x: Any) -> Any:
    """A drawn workload with its types spelled out (`--compare-draws`): `fuzzing._plain` makes a tuple a list and an
    int32 array or scalar plain ints, so the JSON alone would not see a strategy change a workload's types. Any other
    value counts by its repr, which tells 1, 1.0, True and "1" apart."""
    t = type(x)
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return t.__qualname__, [(f.name, _typed(getattr(x, f.name))) for f in dataclasses.fields(x)]
    if t is dict:
        return "dict", [(_typed(k), _typed(v)) for k, v in x.items()]
    if t is list or t is tuple:
        return t.__name__, [_typed(v) for v in x]
    if t is np.ndarray and x.dtype != object:
        return "ndarray", x.dtype.str, x.shape, x.tobytes()
    if isinstance(x, np.generic):
        return x.dtype.str, x.item()
    return x


def _repr_hash(x: Any) -> str:
    return hashlib.sha1(repr(x).encode()).hexdigest()[:16]


def draw_examples(suite: str, test: str, n: int, seed: int) -> tuple[list[Any], list[str], dict[str, int]]:
    """`n` workloads from the suite's own strategy and seed, its body replaced by a recorder (nothing runs); a hash of
    each test case the Hypothesis engine ran (its status, choices and spans, their labels numbered by first appearance:
    a composite's own label hashes its source); and the engine's counts of them (calls, valid, invalid, overrun,
    misaligned)."""
    import importlib

    from hypothesis import HealthCheck, Phase, settings
    from hypothesis.internal.conjecture.engine import ConjectureRunner

    mod_path, _, name = test.partition("::")
    mod = importlib.import_module(mod_path.removesuffix(".py").replace("/", "."))
    fn = getattr(mod, name)
    drawn: list[Any] = []
    cases: list[str] = []
    labels: dict[int, int] = {}
    counts: dict[str, int] = {}

    def record(*args: Any, **kwargs: Any) -> None:
        drawn.append(copy.deepcopy(args[0] if args else next(iter(kwargs.values()))))

    test_function, run = ConjectureRunner.test_function, ConjectureRunner.run

    def traced(runner: Any, data: Any) -> None:
        try:
            test_function(runner, data)
        finally:  # the last test case ends the run with an exception
            if data.frozen:
                spans = data.spans
                cases.append(_repr_hash((int(data.status), data.choices, list(spans.trail),
                                         [labels.setdefault(x, len(labels)) for x in spans.labels])))

    def counted(runner: Any) -> None:
        run(runner)
        counts.update({k: getattr(runner, a, -1) for k, a in (("calls", "call_count"), ("valid", "valid_test_cases"),
                                                              ("invalid", "invalid_test_cases"),
                                                              ("overrun", "overrun_test_cases"),
                                                              ("misaligned", "misaligned_count"))})

    fn.hypothesis.inner_test = record
    fn._hypothesis_internal_use_seed = seed
    fn._hypothesis_internal_use_settings = settings(max_examples=n, phases=(Phase.generate,), database=None,
                                                    deadline=None, suppress_health_check=list(HealthCheck),
                                                    derandomize=False, print_blob=False)
    ConjectureRunner.test_function, ConjectureRunner.run = traced, counted
    try:
        fn()
    finally:
        ConjectureRunner.test_function, ConjectureRunner.run = test_function, run
    return drawn, cases, counts


def run_examples(suite: str, lines: list[dict], detail: bool = False) -> dict[str, dict]:
    """Each example through the tree's own `load_example` and `run_example`: {hash: outcome, digest, events}."""
    fuzzing = _either("tests.support.fuzzing", "tests.fuzzing")

    out: dict[str, dict] = {}
    with tempfile.TemporaryDirectory(prefix="plandiff-ex-") as tmp:
        path = Path(tmp) / "example.json"
        for line in lines:
            path.write_text(json.dumps({"suite": suite, "example": line["ex"]}))
            REC.reset(detail)
            error = ""
            try:
                s, ex = fuzzing.load_example(path)
                fuzzing.run_example(s, ex)
                outcome = "passed"
            except Exception as exc:
                outcome, error = type(exc).__name__, f"{type(exc).__name__}: {exc}"[:400]
            h = hashlib.blake2b(outcome.encode(), digest_size=DIGEST_BYTES)
            for d in REC.digests:
                h.update(d)
            rec = {"outcome": outcome, "digest": h.hexdigest(), "events": len(REC.digests)}
            if error:
                rec["error"] = error
            if detail:
                rec["full"] = REC.full
                rec["event_digests"] = [d.hex() for d in REC.digests]
            out[line["h"]] = rec
    return out


def worker_main(argv: list[str]) -> int:
    """`argv`: one JSON object: mode (draw: draw then digest; draws: draw only; digest; detail; suites), suite, test,
    n, seed, examples (path of JSONL to read or write), out (path), hashes (detail). A draw's result also lists, in
    order, each drawn workload's hash (`draws`, of its `_typed` form) and each test case's (`cases`), and the engine's
    counts (`engine`)."""
    args = json.loads(argv[0])
    mode = args["mode"]
    if mode == "suites":
        suites = _either("tests.scripts.fuzz_campaign", "tests.fuzz_campaign").SUITES
        Path(args["out"]).write_text(json.dumps(suites))
        return 0
    suite = args["suite"]
    install_hooks(suite)
    t0 = time.perf_counter()
    draws: dict[str, Any] = {}
    if mode in ("draw", "draws"):
        _plain = _either("tests.support.fuzzing", "tests.fuzzing")._plain
        lines, seen, unserial = [], set(), 0
        drawn, cases, engine = draw_examples(suite, args["test"], args["n"], args["seed"])
        for ex in drawn:
            try:
                plain = json.loads(json.dumps(_plain(ex)))
            except (TypeError, ValueError):
                unserial += 1
                continue
            h = _example_hash(plain)
            if h not in seen:
                seen.add(h)
                lines.append({"h": h, "ex": plain})
        draws = {"draws": [_repr_hash(_typed(ex)) for ex in drawn], "cases": cases, "engine": engine}
        if mode == "draws":
            Path(args["out"]).write_text(json.dumps({"suite": suite, **draws,
                                                     "draw_s": round(time.perf_counter() - t0, 2)}))
            return 0
        with open(args["examples"], "w") as f:
            for line in lines:
                f.write(json.dumps(line) + "\n")
        drawn_s = time.perf_counter() - t0
    else:
        lines = [json.loads(x) for x in Path(args["examples"]).read_text().splitlines() if x]
        if mode == "detail":
            lines = [x for x in lines if x["h"] in set(args["hashes"])]
        unserial, drawn_s = 0, 0.0
    t1 = time.perf_counter()
    results = run_examples(suite, lines, detail=mode == "detail")
    Path(args["out"]).write_text(json.dumps({"suite": suite, "examples": results, "unserializable": unserial,
                                             "draw_s": round(drawn_s, 2),
                                             "run_s": round(time.perf_counter() - t1, 2), **draws}))
    return 0


# -- the orchestrator -----------------------------------------------------------------------------------------------


def _git(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def unpack(rev: str, dest: Path) -> str:
    """`rev`'s tree in `dest` (git archive), the working tree's git-ignored fixtures linked in; its commit id."""
    commit = _git("rev-parse", "--verify", f"{rev}^{{commit}}")
    dest.mkdir(parents=True)
    with tempfile.TemporaryFile() as tar:
        subprocess.run(["git", "archive", "--format=tar", commit], cwd=ROOT, check=True, stdout=tar)
        tar.seek(0)
        with tarfile.open(fileobj=tar) as tf:
            tf.extractall(dest, filter="data")
    for rel in LINKED:
        src, dst = ROOT / rel, dest / rel
        if src.exists() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.symlink_to(src)
    return commit


def worker_env(root: Path) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")}
    env.update(PYTHONPATH=str(root), PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1")
    env.pop("HYPOTHESIS_PROFILE", None)  # the fuzz campaign's: its settings would replace the harness's draw
    return env


def run_worker(root: Path, args: dict, log: Path, timeout: float) -> dict:
    """One worker in `root`'s tree; its JSON result (`args["out"]`), or {"failed": ...}."""
    boot = (f"import runpy, sys; sys.path.insert(0, {str(root)!r}); sys.argv = ['plandiff', {json.dumps(args)!r}]; "
            f"runpy.run_path({str(WORKER)!r}, run_name='__plandiff_worker__')")
    try:
        proc = subprocess.run([sys.executable, "-c", boot], cwd=root, env=worker_env(root), capture_output=True,
                              text=True, timeout=timeout)
        log.write_text(proc.stdout + proc.stderr)
        if proc.returncode:
            return {"failed": f"exit {proc.returncode}: {(proc.stdout + proc.stderr).strip()[-1500:]}"}
    except subprocess.TimeoutExpired:
        return {"failed": f"timed out after {timeout:.0f} s"}
    return json.loads(Path(args["out"]).read_text())


def first_difference(a: list[dict], b: list[dict], da: list[str] = (), db: list[str] = ()) -> dict:
    """The first event at which two full runs differ (by their stored digests `da`, `db` when given, else by the
    events' values), and its differing fields: a long field from its first differing value on, with both lengths."""
    n = min(len(a), len(b))
    if da and db:
        i = next((j for j, (x, y) in enumerate(zip(da, db)) if x != y), n)
    else:
        i = next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), n)
    if i < n:
        x, y = a[i], b[i]
        if x["kind"] != y["kind"]:
            return {"event": i, "kind": f"{x['kind']} / {y['kind']}", "fields": {}}
        fields = {}
        for name in sorted(set(x["fields"]) | set(y["fields"])):
            u, v = x["fields"].get(name), y["fields"].get(name)
            if u != v or type(u) is not type(v):
                fields[name] = _window(u, v)
        return {"event": i, "kind": x["kind"], "fields": fields}
    if len(a) != len(b):
        return {"event": n, "kind": "end", "fields": {"events": {"base": len(a), "new": len(b)}}}
    return {"event": -1, "kind": "none", "fields": {}}


def _flat(v: Any) -> Any:
    if isinstance(v, list) and v and isinstance(v[0], list):
        return np.asarray(v, dtype=object).ravel().tolist() if len({len(r) for r in v}) == 1 else \
            [x for r in v for x in r]
    return v


def _window(u: Any, v: Any) -> dict:
    """Two values of a field for the report: lists from their first differing value on (TRUNC of them), with
    their lengths and that index; anything else as is."""
    fu, fv = _flat(u), _flat(v)
    if not (isinstance(fu, list) and isinstance(fv, list)):
        return {"base": u, "new": v}
    i = next((j for j, (x, y) in enumerate(zip(fu, fv)) if x != y), min(len(fu), len(fv)))
    lo = max(0, i - 2)

    def cut(f: list) -> list:
        return (["…"] if lo else []) + f[lo:lo + TRUNC] + (["…"] if len(f) > lo + TRUNC else [])

    return {"base": cut(fu), "new": cut(fv), "lengths": [len(fu), len(fv)], "from": i}


def compare(base: dict[str, dict], new: dict[str, dict]) -> dict:
    """One suite's two digest maps, joined on example hash."""
    both = sorted(set(base) & set(new))
    changed = [h for h in both if base[h]["digest"] != new[h]["digest"]]
    broke = [h for h in both if base[h]["outcome"] == "passed" != new[h]["outcome"]]
    base_failed = [base[h] for h in both if base[h]["outcome"] != "passed"]
    return {"examples": len(both), "identical": len(both) - len(changed), "changed": changed, "new_failures": broke,
            "only_base": len(set(base) - set(new)), "only_new": len(set(new) - set(base)),
            "base_errors": sum(1 for h in both if base[h]["outcome"] != "passed"),
            "new_errors": sum(1 for h in both if new[h]["outcome"] != "passed"),
            "events": sum(new[h]["events"] for h in both),
            "all_errors": (base_failed[0].get("error") or base_failed[0]["outcome"]) if both and
            len(base_failed) == len(both) else ""}


def compare_draws(base: dict, new: dict) -> dict:
    """One suite's draws in the two trees (`--compare-draws`): the workloads and test cases each drew, the index of the
    first workload and of the first test case drawn differently (−1: none), and the engine's counts of both sides where
    they differ."""
    out: dict[str, Any] = {}
    for key, name in (("draws", "workload"), ("cases", "case")):
        a, b = base.get(key) or [], new.get(key) or []
        out[f"{name}s"] = [len(a), len(b)]
        out[f"first_{name}"] = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y),
                                    -1 if len(a) == len(b) else min(len(a), len(b)))
    if base.get("engine") != new.get("engine"):
        out["engine"] = {"base": base.get("engine"), "new": new.get("engine")}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="the old tree: a revision (branch, tag, commit) or a directory")
    ap.add_argument("--new", help="the new tree: a revision or a directory (default: the working tree, uncommitted "
                                  "changes included)")
    ap.add_argument("--suites", nargs="+", help="fuzz suites (default: every suite of tests/scripts/fuzz_campaign.py)")
    ap.add_argument("--examples", type=int, default=EXAMPLES, help="workloads drawn per suite")
    ap.add_argument("--seed", type=int, default=SEED, help="the draw's Hypothesis seed")
    ap.add_argument("--jobs", type=int, default=0, help="parallel workers (default: the CPU count)")
    ap.add_argument("--expect-changed", default="", help="suites allowed to change (comma-separated)")
    ap.add_argument("--compare-draws", action="store_true",
                    help="the other tree draws the workloads too: any workload drawn differently fails")
    ap.add_argument("--show", type=int, default=SHOW, help="changed examples per suite described field by field")
    ap.add_argument("--timeout", type=float, default=3600.0, help="seconds per worker")
    ap.add_argument("--out", type=Path, help="results directory (default: results/plandiff/<time>)")
    ap.add_argument("--keep", action="store_true", help="keep the unpacked trees")
    args = ap.parse_args(argv)
    from tests.support.paths import RESULTS  # here, not at the top: a worker imports nothing of tests

    # absolute: the workers run in their trees' directories
    out = (args.out or RESULTS / "plandiff" / time.strftime("%Y%m%d-%H%M%S")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="plandiff-"))
    t0 = time.perf_counter()
    try:
        return _run(args, out, tmp, t0)
    except Exception as exc:
        print(f"plan-diff failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    finally:
        if args.keep:
            print(f"trees kept in {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


def resolve(spec: str | None, side: str, tmp: Path) -> tuple[Path, str]:
    """A tree to compare: a directory holding one (tests plant changes in a copy), a revision (unpacked into `tmp`),
    or (None) the working tree."""
    if spec is None:
        dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
        return ROOT, f"the working tree ({_git('rev-parse', 'HEAD')[:12]}{' + changes' if dirty else ''})"
    if Path(spec).is_dir():
        return Path(spec).resolve(), str(spec)
    return tmp / side, f"{spec} ({unpack(spec, tmp / side)[:12]})"


def _run(args: argparse.Namespace, out: Path, tmp: Path, t0: float) -> int:
    trees, names = {}, {}
    for side in ("base", "new"):
        trees[side], names[side] = resolve(getattr(args, side), side, tmp)
    suites: dict[str, dict[str, str]] = {}
    for side, root in trees.items():
        res = run_worker(root, {"mode": "suites", "out": str(out / f"suites-{side}.json")}, out / f"suites-{side}.log",
                         args.timeout)
        if "failed" in res:
            print(f"{side}: could not list its suites: {res['failed']}", file=sys.stderr)
            return 2
        suites[side] = res
    wanted = args.suites or list(suites["new"])
    unknown = [s for s in wanted if s not in suites["new"] and s not in suites["base"]]
    if unknown:
        print(f"unknown suites: {unknown}", file=sys.stderr)
        return 2
    jobs = args.jobs or os.cpu_count() or 1
    print(f"plan-diff: {names['base']} against {names['new']}; {len(wanted)} suites × {args.examples:,} workloads "
          f"(seed {args.seed}), {jobs} workers; results in {_rel(out)}", flush=True)

    def one_suite(suite: str) -> dict:
        exs = out / f"examples-{suite}.jsonl"
        drawn_in = "base" if suite in suites["base"] else "new"
        res: dict[str, Any] = {"suite": suite, "drawn_in": drawn_in}
        first = run_worker(trees[drawn_in], {"mode": "draw", "suite": suite, "test": suites[drawn_in][suite],
                                             "n": args.examples, "seed": args.seed, "examples": str(exs),
                                             "out": str(out / f"digests-{drawn_in}-{suite}.json")},
                           out / f"{drawn_in}-{suite}.log", args.timeout)
        if "failed" in first:
            return {**res, "failed": f"{drawn_in}: {first['failed']}"}
        res[drawn_in] = first
        other = "new" if drawn_in == "base" else "base"
        if suite not in suites[other]:
            return {**res, "not_compared": f"no such suite in the {other} tree"}
        drawn = {}
        if args.compare_draws:
            drawn = run_worker(trees[other], {"mode": "draws", "suite": suite, "test": suites[other][suite],
                                              "n": args.examples, "seed": args.seed,
                                              "out": str(out / f"draws-{other}-{suite}.json")},
                               out / f"draws-{other}-{suite}.log", args.timeout)
            if "failed" in drawn:
                return {**res, "failed": f"{other} (draws): {drawn['failed']}"}
        second = run_worker(trees[other], {"mode": "digest", "suite": suite, "examples": str(exs),
                                           "out": str(out / f"digests-{other}-{suite}.json")},
                            out / f"{other}-{suite}.log", args.timeout)
        if "failed" in second:
            return {**res, "failed": f"{other}: {second['failed']}"}
        res[other] = second
        cmp = compare(res["base"]["examples"], res["new"]["examples"])
        cmp["unserializable"] = res[drawn_in].get("unserializable", 0)
        if drawn:
            cmp["draws"] = compare_draws(first, drawn)
        details, detail_failed = [], []
        for h in cmp["changed"][:args.show]:
            full = {}
            for side in ("base", "new"):
                d = run_worker(trees[side], {"mode": "detail", "suite": suite, "examples": str(exs), "hashes": [h],
                                             "out": str(out / f"detail-{side}-{suite}-{h}.json")},
                               out / f"detail-{side}-{suite}-{h}.log", args.timeout)
                if "failed" in d:
                    detail_failed.append(f"{side} {h}: {d['failed']}")
                full[side] = d.get("examples", {}).get(h, {})
            if not (full["base"] and full["new"]):
                details.append({"example": h, "event": -1, "kind": "detail worker failed", "fields": {},
                                "outcomes": {}})
                continue
            diff = first_difference(full["base"].get("full") or [], full["new"].get("full") or [],
                                    full["base"].get("event_digests") or [], full["new"].get("event_digests") or [])
            diff.update(example=h, outcomes={s: full[s].get("error") or full[s].get("outcome") for s in full})
            details.append(diff)
        return {**res, "compare": cmp, "details": details, "detail_failed": detail_failed}

    with ThreadPoolExecutor(min(jobs, len(wanted))) as pool:
        results = list(pool.map(one_suite, wanted))
    return _report(results, args, names, out, time.perf_counter() - t0)


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def _report(results: list[dict], args: argparse.Namespace, names: dict, out: Path, secs: float) -> int:
    allowed = {s for s in args.expect_changed.split(",") if s}
    both = "; both trees drew them" if getattr(args, "compare_draws", False) else ""
    lines = [f"# Plan diff: {names['base']} → {names['new']}", "",
             f"{args.examples:,} workloads per suite drawn with seed {args.seed}{both}; {secs / 60:.1f} min.", "",
             "| suite | workloads | identical | changed | events | errors (base / new) | note |",
             "|---|---|---|---|---|---|---|"]
    status, total, changed_total = 0, 0, 0
    summary = []
    for r in results:
        s = r["suite"]
        if "failed" in r:
            lines.append(f"| {s} | | | | | | worker failed: {r['failed'][:200]} |")
            status = max(status, 2)
            summary.append({"suite": s, "failed": r["failed"]})
            continue
        if "not_compared" in r:
            lines.append(f"| {s} | | | | | | {r['not_compared']} (drawn in {r['drawn_in']}) |")
            summary.append({"suite": s, "not_compared": r["not_compared"]})
            continue
        c = r["compare"]
        n_changed = len(c["changed"])
        total += c["examples"]
        changed_total += n_changed
        note = []
        if n_changed and s not in allowed:
            status = max(status, 1)
            note.append("**unexpected**")
        elif n_changed:
            note.append("expected")
        elif s in allowed:
            note.append("expected to change: unchanged")
        if c["new_failures"]:  # a workload that passed now fails: never expected, whatever --expect-changed says
            status = max(status, 1)
            note.append(f"**{len(c['new_failures'])} new failures**")
        if c["all_errors"]:  # errors on both sides count as identical: a suite that only errors compares nothing
            status = max(status, 1)
            note.append(f"**every base workload errors: {c['all_errors'][:200]}**")
            print(f"{s}: every base workload errors; the first: {c['all_errors']}", file=sys.stderr)
        if not c["examples"]:  # nothing compared is no proof of anything
            status = max(status, 2)
            note.append("**nothing joined**")
        if c["only_base"] or c["only_new"]:
            note.append(f"unjoined {c['only_base']} / {c['only_new']}")
        if c.get("unserializable"):
            note.append(f"{c['unserializable']} drawn workloads not JSON (left out)")
        if "draws" in c:  # `--compare-draws`: a seed must draw the same workloads in both trees
            d = c["draws"]
            diff = [f"{what} {d[k]}" for what, k in (("workload", "first_workload"), ("test case", "first_case"))
                    if d[k] >= 0]
            if diff:
                note.append(f"**drawn differently from {' and '.join(diff)}**")
            if "engine" in d:
                note.append(f"**the engine's counts differ: {d['engine']}**")
            if diff or "engine" in d:
                status = max(status, 1)
            else:
                note.append("drawn alike")
        if r.get("detail_failed"):
            status = max(status, 2)
            note.append(f"**{len(r['detail_failed'])} detail runs failed**")
        lines.append(f"| {s} | {c['examples']:,} | {c['identical']:,} | {n_changed:,} | {c['events']:,} | "
                     f"{c['base_errors']} / {c['new_errors']} | {'; '.join(note)} |")
        summary.append({"suite": s, **{k: v for k, v in c.items() if k != "changed"}, "changed": n_changed,
                        "changed_examples": c["changed"], "details": r["details"],
                        "detail_failed": r.get("detail_failed", [])})
    for r in results:
        for d in r.get("details", ()):
            lines += ["", f"## {r['suite']}: example {d['example']}, first difference at event {d['event']} "
                          f"({d['kind']})", ""]
            if d["outcomes"].get("base") != d["outcomes"].get("new"):
                lines.append(f"- outcome: base {d['outcomes'].get('base')!r}, new {d['outcomes'].get('new')!r}")
            for name, v in d["fields"].items():
                at = f" (lengths {v['lengths'][0]} / {v['lengths'][1]}, from index {v['from']})" if "from" in v else ""
                lines.append(f"- `{name}`{at}: base {json.dumps(v['base'])} → new {json.dumps(v['new'])}")
        for f in r.get("detail_failed", ()):
            lines += ["", f"## {r['suite']}: a detail run failed: {f[:600]}"]
    verdict = ("identical" if not changed_total and status == 0 else
               "changed only where expected" if status == 0 else
               "UNEXPECTED CHANGES" if status == 1 else "WORKER FAILURE")
    lines += ["", f"**{verdict}:** {total - changed_total:,} of {total:,} workloads identical."]
    text = "\n".join(lines) + "\n"
    (out / "report.md").write_text(text)
    (out / "summary.json").write_text(json.dumps({"base": names["base"], "new": names["new"], "seed": args.seed,
                                                  "examples": args.examples, "seconds": round(secs, 1),
                                                  "status": status, "verdict": verdict, "suites": summary},
                                                 indent=1))
    print(text)
    return status


if __name__ == "__plandiff_worker__":
    sys.exit(worker_main(sys.argv[1:]))
elif __name__ == "__main__":
    sys.exit(main())
