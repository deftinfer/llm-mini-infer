"""The strict fake's plan and KV checks against their reference implementations (`FakeRunner(full_checks=True)`).

`test_fast_checks_match_the_reference` draws random workloads of every fuzz suite that runs the strict fake
(`tests/scripts/fuzz_campaign.py`) and runs each in both modes, then again with each of a few faults of distinct kinds,
injected at the first chance from a random step on. Three families of fault. The state of a random running request,
before a step's forward or its sampling: a KV position overwritten, poisoned or unmarked precise, a page-table entry
moved to another page or to none, a page wiped as if released and reused (in one arena: as the other kind) or named by
another slot too, its admission not sampled yet, its device sequence length past its context (by one, or past the page
table); in window mode the same in its window, or the window page its next position writes made that position's full
page. What a row the step checks next reads or writes, once the plan's tables are applied (or before the sampling of an
MTP decode row, which checks its KV there), at a step the clean run shows it has such a row: a page it writes named by
another slot too or by none, full or window; a PRECISE row's precise marks dropped from a position it reads to the end
of that page, full or window; a cached position of a resumed admission's prefix overwritten or never written (a
workload with such rows gets one of these). The plan, before its forward: a field of a random SCHED row (n_draft, next
token, a flag, pos0, n_query, tok_off, its slot, an input token) or a PAGES entry changed. Both modes must end alike
(passed, or the same exception with the same message, which names the step and the first bad position or entity) after
the same outputs at every step, dtypes included.

`test_injected_faults_are_caught`: each kind of fault, in a small workload, fails it alike in both modes, in the check
it targets; together the cases reach every error of the KV checks (`_check_kv`, `_kv_history`) and every SCHED check
a fault can reach (not the row count, the token budget, CKPT without rows: the same code in both).

    FAKE_CHECKS_EXAMPLES=300 uv run pytest tests/sched/test_fake_checks.py     # workloads per suite (default 3)
"""

from __future__ import annotations

import copy
import importlib
import os
import random
import re
from collections.abc import Callable
from typing import Any
from unittest import mock

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from llm_mini_infer.engine.messages import FinishReason
from llm_mini_infer.runner.base import MASK_EMPTY, PlanError, PoolConfig
from llm_mini_infer.runner.fake import FakeRunner
from llm_mini_infer.sched.plan import ADMIT_CONSTRAINED, MASKED, PRECISE, SAMPLE, STATE_ZERO, PlanBuilder, StepPlan

from ..support import harness
from ..support.fuzzing import run_example

EXAMPLES = int(os.environ.get("FAKE_CHECKS_EXAMPLES", "3"))
FAULTS = 4  # faulted runs per workload (each in both modes)
# suite -> (module of tests/support/, strategy): every fuzz suite that runs the strict fake
STRATEGIES = {
    "cache_sched": ("workloads", "cache_workloads"), "cache_pressure": ("workloads", "pressure_pairs"),
    "scheduler": ("workloads", "workloads"), "mtp": ("mtp_workloads", "mtp_workloads"),
    "images": ("workloads", "image_workloads"), "structured": ("structured_cases", "structured_workloads"),
    "window": ("window_workloads", "window_workloads"), "window_pressure": ("window_workloads", "window_pressure"),
    "arena_pressure": ("window_workloads", "arena_pressure"), "arena_hits": ("window_workloads", "arena_hits"),
    "gemma_images": ("window_workloads", "gemma_image_workloads"),
    "window_mtp": ("window_mtp", "window_mtp_workloads"), "arena_mtp": ("window_mtp", "arena_mtp_workloads"),
    "gemma_images_mtp": ("window_mtp", "gemma_images_mtp_workloads"),
    "precise": ("precise_cases", "precise_workloads"),
    "window_precise": ("precise_cases", "window_precise_workloads"), "nan": ("nan_cases", "nan_workloads"),
    "structured_window": ("forced_cases", "structured_window_workloads"),
    "structured_hybrid": ("forced_cases", "structured_hybrid_workloads"),
}
MODULES = {mod: importlib.import_module(f"tests.support.{mod}") for mod, _ in STRATEGIES.values()}
# Faults (module docstring): of a running request's state; of a row the step checks next; of the plan
KINDS = ("kv", "poison", "unprecise", "remap", "reuse", "zero", "shared", "unsampled", "seq")
WINDOW_KINDS = ("wkv", "wpoison", "wunprecise", "wremap", "wzero", "cross")
ROW_KINDS = ("write", "nopage", "precise", "hit", "unwritten", "wwrite", "wnopage", "wprecise")
FLAGS = {"MASKED": MASKED, "PRECISE": PRECISE, "STATE_ZERO": STATE_ZERO, "SAMPLE": SAMPLE}
PLAN_KINDS = ("n_draft", "next_tok", *FLAGS, "pos0", "n_query", "tok_off", "twice", "slot", "token", "page", "wpage")


def kinds_for(r: Probe) -> list[str]:
    """The faults that can find a target in the clean run of runner `r` (a row fault: on a row some step checks)."""
    window = ("wwrite", "wnopage", "wprecise", "wpage", *WINDOW_KINDS)
    return [k for k in KINDS + WINDOW_KINDS + ROW_KINDS + PLAN_KINDS if (r.window or k not in window) and
            (r.precise_rows or k not in ("unprecise", "wunprecise")) and (k not in ROW_KINDS or r.chances.get(k))]


def _other(v: int, vocab: int) -> int:
    return (v + 1) % vocab if v >= 0 else 0


def inject(r: FakeRunner, rng: random.Random, kind: str) -> str:
    """Corrupt the state the checks guard, for a random running request; returns what was done ("" if nothing could
    be: no request has a context yet)."""
    slots = [s for s in np.flatnonzero(r.admitted).tolist() if r.history[s]]
    if not slots:
        return ""
    s = rng.choice(slots)
    n = len(r.history[s])
    if kind == "unsampled":  # a decode row before its admission sampled
        r.sampled[s] = False
        return f"{kind} slot {s}"
    if kind == "seq":  # the device's sequence length past its context: by one, or past the page table
        r.seq_len[s] += rng.choice((1, r.pools.max_model_len))
        return f"{kind} slot {s}: {int(r.seq_len[s])}"
    if kind == "shared":  # the page of its last position named by another slot too
        page = int(r.rows[s, (n - 1) // r.ps])
        r.page_users[page] += bool(page)
        return f"{kind} slot {s} page {page}"
    if kind == "cross":  # the window page its next position writes (none before it) is that position's full page
        i = n // r.wps
        page, new = int(r.wrows[s, i]), int(r.rows[s, n // r.ps])
        if n % r.wps or not page or not new:
            return ""
        r.wpage_users[page] -= 1
        r.wpage_users[new] += 1
        r.wrows[s, i] = new
        return f"{kind} slot {s} window page {i}: {page} -> {new}"
    if kind in WINDOW_KINDS:
        lo = max(0, n - r.window)
        p = rng.randrange(lo, n)
        i, o = p // r.wps, p % r.wps
        page = int(r.wrows[s, i])
        if not page:
            return ""
        if kind == "wkv":
            r.wkv[page, o] = _other(int(r.wkv[page, o]), r.vocab)
        elif kind == "wpoison":
            r.wkv_bad[page, o] = True
        elif kind == "wunprecise":
            r.wkv_prec[page, o] = False
        else:  # to any page, one of its full pages (a page of both kinds) or none
            full = [int(x) for x in r.rows[s] if x]
            new = rng.choice(full) if full and rng.random() < 0.5 else rng.randrange(1, r.pools.num_pages)
            new = 0 if kind == "wzero" else new
            r.wpage_users[page] -= 1
            r.wpage_users[new] += bool(new)
            r.wrows[s, i] = new
            return f"{kind} slot {s} window page {i}: {page} -> {new}"
        return f"{kind} slot {s} position {p} (window page {page})"
    p = rng.randrange(n + (4 if kind in ("zero", "remap") else 0))  # a page-table entry may be one it writes next
    i, o = p // r.ps, p % r.ps
    page = int(r.rows[s, i])
    if not page:
        return ""
    if kind == "kv":
        r.kv[page, o] = _other(int(r.kv[page, o]), r.vocab)
    elif kind == "poison":
        r.kv_bad[page, o] = True
    elif kind == "unprecise":
        r.kv_prec[page, o] = False
    elif kind in ("remap", "zero"):
        new = rng.randrange(1, r.pools.num_pages) if kind == "remap" else 0
        r.page_users[page] -= 1
        r.page_users[new] += bool(new)
        r.rows[s, i] = new
        return f"{kind} slot {s} page {i}: {page} -> {new}"
    elif r.window:  # reused as a window page: its full KV is lost
        r._become(np.asarray([page]), 2)
    else:  # reused by another request: other tokens
        r.kv[page] = np.where(r.kv[page] >= 0, (r.kv[page] + 7) % r.vocab, r.kv[page])
    return f"{kind} slot {s} position {p} (page {page})"


def targets(r: FakeRunner, kind: str, rows: list[tuple[int, int, int, bool, bool]]) -> list:
    """The rows the step checks next (`rows`: slot, first position, the end of what it writes, PRECISE, explicit)
    whose reads or writes a row fault (`inject_row`) can corrupt."""
    if kind in ("precise", "wprecise"):
        return [x for x in rows if x[3] and x[1]]
    if kind in ("hit", "unwritten"):  # the first row of a resumed admission, which reads its prefix from the KV
        return [x for x in rows if x[4] and len(r.history[x[0]]) < x[1]]
    return rows


def inject_row(r: FakeRunner, rng: random.Random, kind: str, rows: list[tuple[int, int, int, bool, bool]]) -> str:
    """Corrupt what a row the step checks next reads or writes (`targets`); "" if no row has a target."""
    rows = targets(r, kind, rows)
    if not rows:
        return ""
    s, p0, end, _, _ = rng.choice(rows)
    if kind in ("precise", "hit", "unwritten"):
        p = rng.randrange(len(r.history[s]) if kind != "precise" else 0, p0)
        page, o = int(r.rows[s, p // r.ps]), p % r.ps
        if kind == "precise":
            r.kv_prec[page, o:] = False
        else:
            r.kv[page, o] = -1 if kind == "unwritten" else _other(int(r.kv[page, o]), r.vocab)
        return f"{kind} slot {s} position {p} (page {page})"
    if kind == "wprecise":
        p = rng.randrange(max(0, p0 - r.window + 1), p0)
        page = int(r.wrows[s, p // r.wps])
        r.wkv_prec[page, p % r.wps:] = False
        return f"{kind} slot {s} position {p} (window page {page})"
    window = kind in ("wwrite", "wnopage")
    ps, table, users = (r.wps, r.wrows, r.wpage_users) if window else (r.ps, r.rows, r.page_users)
    lo = max(0, p0 - r.window + 1) if kind == "wnopage" else p0  # a window row reads its window's pages too
    idx = [i for i in range(lo // ps, (end - 1) // ps + 1) if table[s, i]]
    if not idx:
        return ""
    i = rng.choice(idx)
    page = int(table[s, i])
    if kind in ("write", "wwrite"):  # named by another slot too
        users[page] += 1
    else:  # by none
        users[page] -= 1
        table[s, i] = 0
    return f"{kind} slot {s} {'window ' if window else ''}page {i} ({page})"


def mutate(plan: StepPlan, rng: random.Random, kind: str, r: FakeRunner) -> tuple[StepPlan, str]:
    """A copy of `plan` with one field of a random SCHED row, or a PAGES entry, changed; "" if it has none."""
    p = copy.copy(plan)

    def field(name: str) -> np.ndarray:  # a copy of one of the plan's arrays, to change
        a = getattr(plan, name).copy()
        setattr(p, name, a)
        return a

    if kind in ("page", "wpage"):
        ids = field("page_id" if kind == "page" else "wpage_id")
        if not len(ids):
            return plan, ""
        j = rng.randrange(len(ids))
        ids[j] = rng.randrange(1 if kind == "page" else 0, r.pools.num_pages)  # 0: a window page released
        return p, f"{kind} entry {j} = {ids[j]}"
    n = len(plan.sched_slot)
    rows = [i for i in range(n) if plan.sched_tok_off[i] >= 0] if kind == "token" else list(range(n))
    if not rows or (kind == "twice" and n < 2):
        return plan, ""
    i = rng.choice(rows)
    if kind == "n_draft":
        field("sched_n_draft")[i] += 1
    elif kind == "next_tok":
        a = field("sched_next_tok")
        a[i] = -1 if a[i] >= 0 else rng.randrange(r.vocab)
    elif kind in FLAGS:
        field("sched_flags")[i] ^= FLAGS[kind]
    elif kind == "pos0":
        field("sched_pos0")[i] += rng.choice((-1, 1))
    elif kind == "n_query":  # by one, or past max_model_len
        field("sched_n_query")[i] += rng.choice((-1, 1, r.pools.max_model_len))
    elif kind == "tok_off":  # an explicit row made a device-input row, or the other way
        a = field("sched_tok_off")
        a[i] = 0 if a[i] < 0 else -1
    elif kind == "twice":  # another row's slot
        a = field("sched_slot")
        a[i] = a[rng.choice([j for j in range(n) if j != i])]
    elif kind == "slot":
        field("sched_slot")[i] = rng.randrange(len(r.admitted))
    else:  # an input token of an explicit row: out of the vocabulary, or another
        a = field("tokens")
        j = int(plan.sched_tok_off[i]) + rng.randrange(int(plan.sched_n_query[i]))
        a[j] = rng.choice((-1, r.vocab, _other(int(a[j]), r.vocab)))
    return p, f"{kind} row {i}"


class Probe(FakeRunner):
    """The strict fake recording every step's outputs, with `fault` (step, "forward" or "sample", kind, seed) injected
    once, at the first chance from there on that finds a target: a plan's fault before its forward, a row's after the
    plan's tables or before the sampling, a request's before the forward or the sampling. Without a fault, it notes
    where each row fault finds a target (`chances`)."""

    def __init__(self, *a: Any, fault: tuple[int, str, str, int] | None = None, **kw: Any):
        super().__init__(*a, **kw)
        self.fault, self.done, self.log = fault, "", []
        self.chances: dict[str, list[tuple[int, str]]] = {}

    def _due(self, step: int, at: int) -> bool:
        """Whether the fault may go in at `at` (0: before the forward, 1: after the plan's tables, 2: before the
        sampling) of `step`."""
        if not self.fault or self.done:
            return False
        kind = self.fault[2]
        hooks = (0,) if kind in PLAN_KINDS else (1, 2) if kind in ROW_KINDS else (0, 2)
        return at in hooks and (step, at) >= (self.fault[0], 0 if self.fault[1] == "forward" else 2)

    def _row_fault(self, step: int, at: int, rows: list[tuple[int, int, int, bool, bool]]) -> None:
        """A row fault at `at` (1 or 2, `_due`) of `step`, if due; without a fault, the chances `rows` give."""
        if self.fault is None:
            for kind in ROW_KINDS:
                if targets(self, kind, rows):
                    self.chances.setdefault(kind, []).append((step, "forward" if at == 1 else "sample"))
        elif self._due(step, at):
            self.done = inject_row(self, random.Random(self.fault[3]), self.fault[2], rows)

    def launch_forward(self, plan: StepPlan) -> Any:
        if self._due(self.steps, 0):
            rng = random.Random(self.fault[3])
            if self.fault[2] in PLAN_KINDS:
                plan, self.done = mutate(plan, rng, self.fault[2], self)
            else:
                self.done = inject(self, rng, self.fault[2])
        return super().launch_forward(plan)

    def _apply_tables(self, plan: StepPlan) -> set[int]:
        admitted_now = super()._apply_tables(plan)
        # the plan's rows (`targets`): a device-input row writes its verify window and drafter chain
        if self.fault is None or self._due(self.steps, 1):
            k, rows = self.k, []
            span = 1 if not k else 2 * k if self.draft_kv else 1 + k
            sched = (plan.sched_slot, plan.sched_n_query, plan.sched_tok_off, plan.sched_pos0, plan.sched_flags)
            for s, q, off, p0, f in zip(*(a.tolist() for a in sched)):
                if off < 0:
                    rows.append((s, int(self.seq_len[s]), int(self.seq_len[s]) + span, False, False))
                else:
                    rows.append((s, p0, p0 + q + (k - 1 if k and self.draft_kv and f & SAMPLE else 0),
                                 bool(f & PRECISE), True))
            self._row_fault(self.steps, 1, rows)
        return admitted_now

    def launch_sample(self, handle: Any, n_masks: int = 0) -> None:
        if self.fault is None or self.fault[2] in ROW_KINDS:  # MTP decode rows: their KV is checked here
            slots, pos0, _, _ = handle.rows
            self._row_fault(self.steps - 1, 2, [(s, p, p + 1 + self.k, False, False) for s, p, m in
                                                zip(slots.tolist(), pos0.tolist(), handle.mtp.tolist()) if m])
        elif self._due(self.steps - 1, 2):
            self.done = inject(self, random.Random(self.fault[3]), self.fault[2])
        super().launch_sample(handle, n_masks)
        res = handle.result
        self.log.append(tuple(None if a is None else (a.tolist(), a.dtype.str)
                              for a in (res.tokens, res.n_out, res.status, res.drafts)))


def run(work: Callable[[], Any], full: bool, fault: tuple[int, str, str, int] | None = None) -> tuple[str, Probe]:
    """A workload (`work` builds its harness and runs it) in one mode: its outcome ("passed" or the exception) and
    its runner."""
    made: list[Probe] = []

    def make(**kw: Any) -> Probe:  # the mode replaces any the harness asks for (LLM_MINI_INFER_TEST_FULL_CHECKS)
        made.append(Probe(**{**kw, "full_checks": full}, fault=fault))
        return made[-1]

    with mock.patch.object(harness, "FakeRunner", make):
        try:
            work()
            outcome = "passed"
        except Exception as exc:
            outcome = re.sub(r" at 0x[0-9a-f]+", "", f"{type(exc).__name__}: {exc}")  # no object addresses
    return outcome, made[0]


def check_alike(suite: str, ex: Any, seed: int, faults: int = FAULTS) -> list[str]:
    """Both modes alike on `ex`, clean and with `faults` faults of distinct kinds (drawn from `seed`); returns the
    faulted outcomes."""

    def work() -> None:
        run_example(suite, copy.deepcopy(ex))

    clean, r = run(work, full=False)
    ref, rf = run(work, full=True)
    assert clean == ref == "passed", (clean, ref)
    assert r.log == rf.log
    rng = random.Random(seed)
    kinds = rng.sample(kinds_for(r), faults)
    rare = [k for k in ("precise", "wprecise", "hit", "unwritten") if r.chances.get(k)]
    if rare and not set(rare) & set(kinds):  # one fault on rows few steps have: a PRECISE row that continues a
        kinds[0] = rng.choice(rare)  # precise prefix, the first row of a resumed admission
    out = []
    for kind in kinds if r.steps else ():  # a row fault: at a step where it finds a target
        where = rng.choice(r.chances[kind]) if kind in ROW_KINDS else (rng.randrange(r.steps),
                                                                       rng.choice(("forward", "sample")))
        fault = (*where, kind, rng.randrange(1 << 30))
        got, a = run(work, full=False, fault=fault)
        want, b = run(work, full=True, fault=fault)
        assert a.done == b.done, fault
        assert got == want, f"{fault} {a.done}:\nfast: {got[:2000]}\nfull: {want[:2000]}"
        assert a.log == b.log, f"{fault} {a.done}: the outputs differ"
        out.append(got)
    return out


@pytest.mark.parametrize("suite", sorted(STRATEGIES))
@settings(max_examples=EXAMPLES, deadline=None, suppress_health_check=[HealthCheck.too_slow,
                                                                        HealthCheck.data_too_large])
@given(data=st.data())
def test_fast_checks_match_the_reference(suite: str, data: st.DataObject) -> None:
    mod, name = STRATEGIES[suite]
    ex = data.draw(getattr(MODULES[mod], name)(), label="workload")
    check_alike(suite, ex, data.draw(st.integers(0, 1 << 30), label="faults"))


def _decoding(window: int = 0, mtp_k: int = 0, n: int = 1) -> None:
    """`n` requests decoding after prompts of 40 (pages of 4; `window`: a sliding-window model, window pages of 4 and
    full pages of 20, `mtp_k` drafts by a drafter without KV)."""
    specs = [harness.Spec(rid=i, prompt_len=40, max_tokens=24, seed=5 + i, ignore_eos=True) for i in range(n)]
    if window:
        h = MODULES["window_workloads"].harness(specs, window=window, max_tokens=64, mtp_k=mtp_k, draft_kv=False)
    else:
        h = harness.Harness(specs=specs, max_tokens=64, mtp_k=mtp_k)
    h.run()
    h.check()


def _chunked(window: int = 0) -> None:
    """A prompt of 120 computed in PRECISE chunks of 32 (`precise="bf16"`; `window`: a sliding-window model)."""
    spec = harness.Spec(rid=0, prompt_len=120, max_tokens=4, seed=5)
    if window:
        h = MODULES["window_workloads"].harness([spec], window=window, max_tokens=32, precise="bf16")
    else:
        h = harness.Harness(specs=[spec], max_tokens=32, checkpoint_slots=2, grid=16, precise="bf16")
    h.run()
    h.check()


def _resumed(hybrid: bool) -> None:
    """A request, then one with its prompt and 8 tokens more, which resumes from its cached prefix (pages of 4;
    `hybrid`: from a checkpoint of its recurrent state too)."""
    first = harness.Spec(rid=0, prompt_len=40, max_tokens=4, seed=5)
    prompt = harness.prompt_of(first, 1000).tolist() + list(range(1, 9))
    then = harness.Spec(rid=1, prompt_len=0, prompt=prompt, max_tokens=4, seed=6, arrive=12)
    h = harness.Harness(specs=[first, then], max_tokens=64, hybrid=hybrid, checkpoint_slots=2, grid=16)
    h.run()
    h.check()
    assert h.results[1].cached > 0


SCENARIOS = {"one": _decoding, "mtp": lambda: _decoding(mtp_k=2), "two": lambda: _decoding(n=2),
             "window": lambda: _decoding(16), "window_mtp": lambda: _decoding(16, 2), "precise": _chunked,
             "window_precise": lambda: _chunked(16), "hit": lambda: _resumed(False),
             "hybrid_hit": lambda: _resumed(True)}
PLAN = r"PlanError: step \d+: "
OVERWRITTEN = PLAN + r"slot \d+: KV at position \d+ was overwritten by another request"
SHARED = PLAN + r"slot \d+: KV write at position \d+ into page \d+, which another slot shares"
UNPRECISE = PLAN + r"slot \d+: a PRECISE row reads KV at position \d+ that no PRECISE row wrote"
WINDOW_HOLDS = PLAN + r"slot \d+: window KV at position \d+ \(seen by a query at \d+\) holds "
NO_WINDOW = PLAN + r"slot \d+: window positions \[\d+, \d+\) not covered by window pages"
FAILED = rf"AssertionError: \(0, {FinishReason.ERROR:d}, \d+\)"  # it read NaNs: it failed early
WRONG = r"AssertionError: request \d+: tokens differ from the prediction"


@pytest.mark.parametrize(("kind", "scenario", "step", "seed", "error"), [
    # the KV checks: every error of `_check_kv` and `_kv_history`, and poisoned reads
    ("kv", "one", 12, 3, OVERWRITTEN), ("remap", "one", 12, 3, OVERWRITTEN), ("reuse", "one", 12, 3, OVERWRITTEN),
    ("zero", "one", 12, 3, OVERWRITTEN), ("write", "one", 3, 3, SHARED), ("shared", "one", 12, 3, SHARED),
    ("write", "one", 0, 0, SHARED),  # a page past the first the prompt writes
    ("precise", "precise", 0, 3, UNPRECISE), ("unprecise", "precise", 3, 3, UNPRECISE),
    ("unwritten", "hit", 0, 3, PLAN + r"slot \d+: cached position \d+ was never written"),
    ("hit", "hybrid_hit", 0, 3, PLAN + r"slot \d+: its state slot \d+ holds another prefix than its KV pages"),
    ("hit", "hit", 0, 3, WRONG),  # without recurrent state, the context is the KV
    ("poison", "one", 12, 3, FAILED), ("poison", "one", 12, 2, FAILED),  # a late position, an early one
    ("poison", "mtp", 3, 3, FAILED), ("poison", "mtp", 3, 2, FAILED),  # (MTP: the verify row's read)
    ("kv", "window", 12, 3, OVERWRITTEN), ("reuse", "window", 12, 3, OVERWRITTEN),
    ("wkv", "window", 12, 3, WINDOW_HOLDS + r"\d+, not its token \d+"),
    ("wremap", "window", 12, 3, WINDOW_HOLDS + "-1"),
    *[("wwrite", "window", step, seed, PLAN + r"slot \d+: window KV write at position \d+ into window page \d+, which "
                                              "another slot shares") for step, seed in ((3, 3), (0, 0))],
    ("wprecise", "window_precise", 0, 3, PLAN + r"slot \d+: a PRECISE row reads window KV at position \d+ that no "
                                                "PRECISE row wrote"),
    ("cross", "window_mtp", 3, 3, PLAN + r"slot \d+: KV at position \d+ was overwritten by a window page's write"),
    ("wpoison", "window", 12, 3, FAILED), ("wpoison", "window", 12, 1, FAILED),
    # the SCHED checks
    ("nopage", "one", 0, 3, PLAN + r"slot \d+: positions \[0, 40\) not covered by pages"),
    ("nopage", "one", 3, 3, PLAN + "device-input positions not covered by pages"),
    ("write", "mtp", 3, 3, PLAN + "a decode row's KV writes reach a page another slot shares"),
    ("write", "mtp", 4, 0, PLAN + "a decode row's KV writes reach a page another slot shares"),  # its second page
    ("wnopage", "window", 3, 3, NO_WINDOW), ("wzero", "window", 12, 3, NO_WINDOW),
    ("unsampled", "one", 3, 3, PLAN + r"device input before a SAMPLE step of the admission: \[\d+\]"),
    ("SAMPLE", "one", 0, 3, PLAN + r"device input before a SAMPLE step of the admission: \[\d+\]"),
    ("seq", "one", 3, 3, PLAN + "a decoding slot's live state does not hold its sequence"),
    ("seq", "one", 3, 0, PLAN + "device-input positions beyond the page table"),
    ("seq", "precise", 3, 3, PLAN + r"slot \d+: pos0 \d+ != device seq_len \d+"),
    ("n_draft", "one", 3, 3, PLAN + "n_draft must be 0 on device-input rows and 0 on explicit ones"),
    ("next_tok", "mtp", 3, 3, PLAN + "sched_next_tok must be set exactly on non-final explicit rows"),
    ("MASKED", "one", 3, 3, PLAN + "MASKED row of an admission without ADMIT_CONSTRAINED"),
    ("MASKED", "precise", 0, 3, PLAN + "MASKED on a row without SAMPLE"),
    ("PRECISE", "one", 3, 3, PLAN + "PRECISE rows, but this runner takes none"),
    ("tok_off", "precise", 0, 3, PLAN + "PRECISE on a device-input row"),
    ("STATE_ZERO", "one", 0, 3, PLAN + r"slot \d+: STATE_ZERO must be set exactly on the chunk at position 0"),
    ("STATE_ZERO", "one", 3, 3, PLAN + "STATE_ZERO on a device-input row"),
    ("pos0", "one", 3, 3, PLAN + r"device-input rows need pos0 = -1 and n_query = 1"),
    ("n_query", "mtp", 3, 3, PLAN + r"device-input rows need pos0 = -1 and n_query = 3"),
    ("n_query", "one", 0, 3, PLAN + "positions beyond max_model_len"),
    ("n_query", "one", 0, 0, PLAN + r"slot \d+: explicit tokens \[0, 41\) out of range"),
    ("pos0", "precise", 3, 3, PLAN + r"slot \d+: pos0 \d+ != device seq_len \d+"),
    ("tok_off", "one", 3, 3, PLAN + r"slot \d+: pos0 -1 != device seq_len \d+"),
    ("twice", "two", 3, 3, PLAN + "a slot appears twice in SCHED"),
    ("slot", "one", 3, 3, PLAN + r"SCHED row for a slot that is not admitted: \[\d+\]"),
    ("token", "precise", 0, 3, PLAN + r"slot \d+: token out of vocabulary"),
    # the second of two rows
    ("unsampled", "two", 3, 0, PLAN + r"device input before a SAMPLE step of the admission: \[2\]"),
    ("seq", "two", 3, 7, PLAN + "a decoding slot's live state does not hold its sequence"),
    ("n_query", "two", 0, 5, PLAN + "positions beyond max_model_len"),
    ("MASKED", "two", 3, 0, PLAN + "MASKED row of an admission without ADMIT_CONSTRAINED"),
    ("PRECISE", "two", 3, 0, PLAN + "PRECISE rows, but this runner takes none"),
    ("token", "two", 0, 0, PLAN + r"slot 2: token out of vocabulary"),
])
def test_injected_faults_are_caught(kind: str, scenario: str, step: int, seed: int, error: str) -> None:
    """Each kind of fault, injected into a small workload (`SCENARIOS`) from `step` on, fails it alike in both modes,
    in the check it targets (or, a poisoned read, in the request's outputs)."""
    fault = (step, "forward", kind, seed)
    got, a = run(SCENARIOS[scenario], full=False, fault=fault)
    want, b = run(SCENARIOS[scenario], full=True, fault=fault)
    assert a.done and a.done == b.done and got == want and a.log == b.log and re.match(error, got), (a.done, got[:400])


def test_a_slot_twice_in_sched_raises() -> None:
    """Two SCHED rows of one slot fail the plan (a repeated index counted once in `seen[slots] += 1`: before, the check
    never fired)."""
    r = FakeRunner(vocab_size=1000, max_positions=256)
    pc = PoolConfig(page_size=4, num_pages=16, num_state_slots=4, max_num_seqs=2, max_num_batched_tokens=64,
                    max_model_len=256, eos_token_ids=(999,))
    r.configure(pc)
    b = PlanBuilder(pc.max_num_seqs, pc.max_num_batched_tokens, pc.page_size)
    b.begin()
    b.add_admit(1, 1, 0, 0.0, 1.0, 0, 0.0, 1)
    b.add_pages(1, 0, [1, 2])
    for _ in range(2):
        b.add_explicit_row(1, np.arange(5, dtype=np.int32), 0, STATE_ZERO)
    with pytest.raises(PlanError, match="a slot appears twice in SCHED"):
        r.launch_forward(b.finish(1))


@pytest.mark.parametrize("slot", [-1, 3])
@pytest.mark.parametrize(("where", "full", "match"), [
    pytest.param("PAGES", False, r"^step \d+: PAGES for slot -?\d, which is not admitted", id="pages"),
    pytest.param("window PAGES", False, r"window PAGES for slot -?\d, which is not admitted", id="window-pages"),
    pytest.param("SCHED", False, r"SCHED row for a slot that is not admitted: \[-?\d\]", id="sched"),
    pytest.param("SCHED", True, r"SCHED row for a slot that is not admitted: \[-?\d\]", id="sched-full")])
def test_a_slot_out_of_range_raises(where: str, full: bool, match: str, slot: int) -> None:
    """A PAGES or window PAGES entry or a SCHED row for slot -1 or max_num_seqs + 1 fails the plan, with the last slot
    admitted (before, -1 indexed the last slot's tables and the plan passed; max_num_seqs + 1 raised IndexError)."""
    window = 8 if where == "window PAGES" else 0
    r = FakeRunner(vocab_size=1000, max_positions=256, full_checks=full, window=window)
    pc = PoolConfig(page_size=4, num_pages=16, num_state_slots=4, max_num_seqs=2, max_num_batched_tokens=64,
                    max_model_len=256, eos_token_ids=(999,), window_page_size=4 if window else 0)
    r.configure(pc)
    b = PlanBuilder(pc.max_num_seqs, pc.max_num_batched_tokens, pc.page_size, pc.max_model_len, 0, window, 4)
    b.begin()
    b.add_admit(2, 1, 0, 0.0, 1.0, 0, 0.0, 1)
    b.add_pages(2, 0, [1, 2])
    if where == "PAGES":
        b.add_pages(slot, 2, [3])
    elif where == "window PAGES":
        b.add_wpages(slot, 0, [3])
    else:
        b.add_explicit_row(slot, np.arange(5, dtype=np.int32), 0, STATE_ZERO)
    with pytest.raises(PlanError, match=match):
        r.launch_forward(b.finish(1))


def _allowed(vocab: int, constrained: bool) -> np.ndarray:
    allowed = np.ones(vocab, bool)
    if constrained:
        allowed[3:] = False
    return allowed


def _runner(seed: int, at: int, mtp_k: int = 0) -> FakeRunner:
    r = FakeRunner(vocab_size=1000, max_positions=256, mtp_k=mtp_k, mask_empty_at={seed: {at}})
    r.configure(PoolConfig(page_size=4, num_pages=16, num_state_slots=4, max_num_seqs=2, max_num_batched_tokens=64,
                           max_model_len=256, eos_token_ids=(999,), mtp_k=mtp_k))
    return r


@pytest.mark.parametrize("constrained", [False, True])
def test_the_injected_empty_mask_fires_in_the_mtp_verify_path_only_on_a_constrained_mask(constrained: bool) -> None:
    """`mask_empty_at` in `_mtp_decode` empties the mask at its position unless the mask allows every token (a grammar
    that ended under `ignore_eos`): that one gives no MASK_EMPTY and the seed is not in `fired`; a mask that excludes
    tokens still reports MASK_EMPTY and fires."""
    seed, k = 7, 2
    r = _runner(seed, 1, k)
    r.seed[0] = seed
    r.seq_len[0] = 0  # the first verified position is 1
    r.max_len[0] = 256
    toks, status = r._mtp_decode(0, False, [_allowed(1000, constrained) for _ in range(1 + k)], None)
    if constrained:
        assert status == MASK_EMPTY and toks == [0]
    else:
        assert status == 0 and len(toks) >= 1
    assert (seed in r.fired) == constrained


@pytest.mark.parametrize("constrained", [False, True])
def test_the_injected_empty_mask_fires_in_the_single_row_sample_path_only_on_a_constrained_mask(
        constrained: bool) -> None:
    """The same rule for a MASKED SAMPLE row that is not an MTP verify row: the all-ones mask the scheduler fills for an
    ended grammar samples normally and does not fire; a constrained one reports MASK_EMPTY and fires."""
    seed, n_prompt = 7, 5
    r = _runner(seed, n_prompt)
    b = PlanBuilder(2, 64, 4)
    b.begin()
    b.add_admit(1, 1, 0, 0.0, 1.0, 0, 0.0, seed, ADMIT_CONSTRAINED)
    b.add_pages(1, 0, [1, 2])
    b.add_explicit_row(1, np.arange(n_prompt, dtype=np.int32), 0, SAMPLE | MASKED | STATE_ZERO)
    h = r.launch_forward(b.finish(1))
    r.mask_buffer(h)[0] = 0b111 if constrained else -1
    r.launch_sample(h, 1)
    res = r.wait(h)
    assert (res.status is not None and bool(int(res.status[0]) & MASK_EMPTY)) == constrained
    assert (seed in r.fired) == constrained
