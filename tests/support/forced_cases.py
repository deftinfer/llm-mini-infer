"""The forced-token suites' probes, oracles, workloads and driver (tests/sched/test_forced_tokens.py)."""

from __future__ import annotations

import sys
from collections import Counter
from typing import Any

import numpy as np
from hypothesis import assume

from llm_mini_infer.cache.pools import pages_for_one_request
from llm_mini_infer.engine.messages import MASK_CLASSES, POPCOUNT_BUCKETS, FinishReason
from llm_mini_infer.grammar.engine import Grammars
from llm_mini_infer.runner.fake import fake_generate, unpack_mask
from llm_mini_infer.sched.plan import reserve_positions
from llm_mini_infer.sched.scheduler import RUNNING, WAITING, Inflight, Req, Scheduler

from .fake_grammar import FakeGrammar, spec
from .fuzzing import event, nan_write, st
from .harness import EOS, Harness, Spec, check_nan_invariants, key_of, rope_off_of
from .precise_cases import refine_options
from .sched_cases import VOCAB, check_plans, stem
from .window_mtp import bound
from .window_mtp import harness as window_harness
from .workloads import GRIDS

STOP, LENGTH, ABORT, ERROR = FinishReason.STOP, FinishReason.LENGTH, FinishReason.ABORT, FinishReason.ERROR
FORCED, EOS_CLASS, CHOICE, FREE = (MASK_CLASSES.index(c) for c in ("forced", "eos", "choice", "free"))
FAKE = Grammars(FakeGrammar(VOCAB, (EOS,)))  # the harness's grammars: the oracle's forced prefixes


def forced_of(text: str) -> tuple[int, ...]:
    """The forced prefix of a fake grammar over the harness's vocabulary."""
    return FAKE.forced_prefix(text)


def predicted(s: Spec, max_model_len: int = 512, mrope: bool = False) -> list[int]:
    """The sampled path's output of a request (as `Harness.expected`, without a harness): from its radix key (an image
    span's positions hold its key), with its M-RoPE offset when the images are Qwen's (`mrope`)."""
    key = key_of(s, VOCAB)
    limit = min(s.max_tokens, max_model_len - len(key))
    eos = {len(key) + s.eos_after: EOS} if s.eos_after is not None else {}
    m = FAKE.matcher(FAKE.compile(s.grammar)) if s.grammar else None
    return fake_generate(key, s.seed, limit, VOCAB, (EOS,), eos, rope_off=rope_off_of(s) if mrope else 0, matcher=m,
                         eos_ids=(EOS,))


def end_of(s: Spec, out: list[int]) -> int:
    """How many tokens a request delivers on the sampled path: through its first stop, else all."""
    stops = [i for i, t in enumerate(out) if (t == EOS and not s.ignore_eos) or t in s.stop_ids]
    return stops[0] + 1 if stops else len(out)


def attention_harness(specs: list[Spec], *, k: int = 3, page_size: int = 4, max_tokens: int = 64,
                      max_model_len: int = 512, num_pages: int = 0, **kw) -> Harness:
    """Attention-only mode (no state, no window): hits are KV pages, the forced prefix is allowed (no CKPT records)."""
    if not num_pages:
        num_pages = 4 * pages_for_one_request(max_model_len, page_size, reserve_positions(2, k), True) + 16
    return Harness(specs=specs, vocab=VOCAB, hybrid=False, page_size=page_size, num_pages=num_pages,
                   max_tokens=max_tokens, max_model_len=max_model_len, mtp_k=k, **kw)


def gdn_harness(specs: list[Spec], *, k: int = 3, page_size: int = 4, max_tokens: int = 64, max_model_len: int = 512,
                num_pages: int = 0, grid: int = 8, checkpoint_slots: int = 32, **kw) -> Harness:
    """Qwen's shape: a hybrid GDN model (state checkpoints, which a chunk takes as CKPT records: on the grid inside
    it, or at its n_query), its MTP layer's drafter with KV, NONFINITE reported on SAMPLE rows only."""
    if not num_pages:
        num_pages = 4 * pages_for_one_request(max_model_len, page_size, reserve_positions(2, k), True) + 16
    return Harness(specs=specs, vocab=VOCAB, hybrid=True, page_size=page_size, num_pages=num_pages, grid=grid,
                   checkpoint_slots=checkpoint_slots, max_tokens=max_tokens, max_model_len=max_model_len, mtp_k=k,
                   draft_kv=True, **kw)


def make(mode: str, specs: list[Spec], k: int, **kw) -> Harness:
    """Gemma 4's shape (window 16 or 32 over window pages of 4, full pages of 20, its assistant's drafter without KV),
    Qwen's (`gdn`: checkpoints on a grid of 8) or attention-only mode."""
    if mode == "attention":
        return attention_harness(specs, k=k, **kw)
    if mode == "gdn":
        return gdn_harness(specs, k=k, **kw)
    return window_harness(specs, k=k, window=int(mode.removeprefix("window")), page_size=4, **kw)


def check_unrep(s: Scheduler) -> None:
    """The unreported tokens' bookkeeping (`Req.unrep`, `Scheduler.unrep`): a slot is flagged iff its request holds
    tokens its frontend has not received, the flags are counted, and those tokens are among its outputs."""
    flagged = np.flatnonzero(s.unrep)
    assert s._n_unrep == len(flagged), (s._n_unrep, flagged)
    for slot, req in enumerate(s.slot_req):
        if req is None:
            assert not s.unrep[slot], f"slot {slot}: flagged without a request"
            continue
        assert bool(s.unrep[slot]) == bool(req.unrep), f"request {req.rid}: flag {s.unrep[slot]}, {req.unrep}"
        assert len(req.unrep) <= s.n_out[slot]
    for req in s.reqs.values():
        if req.unrep and req.status != RUNNING:  # waiting: they are among the tokens its readmission prefills
            assert req.tokens[req.prompt_len:req.prompt_len + len(req.unrep)].tolist() == req.unrep


PATHS = ("_admit_one", "_schedule_running_prefills")  # where a chunk is planned: a first chunk, or a later one


class Probe:
    """Test-side instrumentation of a harness's scheduler (methods wrapped on the instance, as the harness wraps
    `_fail_cascade`): the forced prefix each request's final chunk took (tokens, and the planning path that planned
    that chunk: `_admit_one` or `_schedule_running_prefills`; `tails`: the forced tokens of each (plan, slot);
    `refused`: the requests whose matcher refused theirs), the requests finished EOS-free (`appended`; `in_flight`:
    whether a row of theirs was), where each of those donated its state (`donated`), the turn each request's finish
    arrived, and what happened to unreported tokens (`events`): a preemption or an abort with them pending, a held
    sample delivered with them in front or dropped (they stay), a readmission that carries them. Every turn's
    accounting check also runs `check_unrep`."""

    def __init__(self, h: Harness):
        s = self.s = h.sched
        self.forced: dict[int, tuple[int, str]] = {}
        self.refused: set[int] = set()
        self.tails: dict[tuple[int, int], int] = {}
        self.planned: dict[int, str] = {}
        self.appended: set[int] = set()
        self.in_flight: dict[int, bool] = {}
        self.donated: dict[int, int] = {}
        self.finished_at: dict[int, int] = {}
        self.events: Counter = Counter()
        force, eos_free, preempt, held, keep = s._force, s._eos_free, s._preempt, s._deliver_held, s._keep
        admit_one, abort, accounting, collect = s._admit_one, s.abort, s.check_accounting, h.collect
        chunk_row = s._chunk_row

        def _chunk_row(req: Req, *a: Any) -> int:
            f = sys._getframe(1)
            while f.f_code.co_name not in PATHS:  # the planning path that planned the chunk
                f = f.f_back
            self.planned[req.rid] = f.f_code.co_name
            return chunk_row(req, *a)

        def _force(req: Req, slot: int, n: int) -> np.ndarray | None:
            assert req.rid not in self.forced, f"request {req.rid}: a second forced prefix"
            out = force(req, slot, n)
            if out is None:  # its matcher refused them: the chunk samples them
                self.refused.add(req.rid)
                return out
            self.forced[req.rid] = (n, self.planned[req.rid])
            self.tails[(s.step_id + 1, slot)] = n  # in the plan being built
            return out

        def _eos_free(inf: Inflight, sl: np.ndarray, finish: np.ndarray) -> list[int]:
            out = eos_free(inf, sl, finish)
            f = s._last_inf if s._last_inf is not None and s._last_inf.step_id > inf.step_id else None
            for i in out:
                slot = int(sl[i])
                rid = s.slot_req[slot].rid
                self.appended.add(rid)
                self.in_flight[rid] = f is not None and bool(((f.slots == slot) & (f.aids == s.aid[slot]) &
                                                              f.sample).any())
            return out

        def _keep(req: Req, slot: int, source: str) -> bool:
            end = int(s.mirror[slot] + s.dev_inflight[slot])
            donated = keep(req, slot, source)
            if donated and req.rid in self.appended:
                self.donated[req.rid] = end
            return donated

        def _preempt(slot: int) -> None:
            if s.unrep[slot]:
                self.events["preempted with unreported tokens"] += 1
                if s.pending[slot]:
                    self.events["preempted with its final chunk in flight"] += 1
            elif s.pending[slot]:
                self.events["preempted with a row in flight"] += 1
            preempt(slot)

        def _deliver_held(inf: Inflight, *a: Any) -> int:
            waiting = {aid: r for aid, r in s._held.items() if r.unrep}
            n = held(inf, *a)
            for aid, r in waiting.items():
                if aid not in s._held:
                    self.events["held sample delivered after unreported tokens" if not r.unrep else
                                "held sample dropped, unreported tokens kept"] += 1
            return n

        def _admit_one(req: Req) -> Any:
            had = bool(req.unrep)
            out = admit_one(req)
            if had and req.status == RUNNING:
                self.events["readmitted with unreported tokens"] += 1
            return out

        def _abort(frontend: int, rids: list[int], reason: int = 0) -> None:
            for rid in rids:
                req = s.reqs.get((frontend, rid))
                if req is not None and req.unrep:
                    self.events["aborted with unreported tokens"] += 1
            abort(frontend, rids, reason)

        def _accounting(full: bool = False) -> None:
            accounting(full)
            check_unrep(s)

        def _collect(report: Any) -> None:
            collect(report)
            for rid, r in h.results.items():
                if r.finish and rid not in self.finished_at:
                    self.finished_at[rid] = h.turn

        s._force, s._eos_free, s._preempt, s._deliver_held = _force, _eos_free, _preempt, _deliver_held
        s._admit_one, s.abort, s.check_accounting, h.collect = _admit_one, _abort, _accounting, _collect
        s._chunk_row, s._keep = _chunk_row, _keep

    def check_counters(self) -> None:
        """The scheduler's forced-prefix counters are the prefixes the probe saw taken."""
        assert self.s.forced_requests == len(self.forced)
        assert self.s.forced_tokens == sum(n for n, _ in self.forced.values())
        assert all(n > 0 for n, _ in self.forced.values())


class MaskOracle:
    """The verify-mask counters recomputed from what the fake runner verified (`launch_sample` wrapped): for every MTP
    verify row with masks, the v drafts it verified (k, or the step's draft count: `handle.v`), its 1 + v masks, its
    tokens and status. A row counts when its
    request was the row's at `process` and no failure there took it (`_fail` wrapped), or when its request was
    preempted while it was in flight and still waits for it (a held sample, delivered unless its status says NONFINITE
    or an empty mask, which no row counts with): the masks it read are those
    after the pending token and after each accepted draft (`n_out` of them), and its first rejected draft, one that
    differs from the target's sample there (a draft equal to it was refused by a commit limit), falls under the class
    of that mask: one token, the EOS alone (`eos`) or another (`forced`), every token (`free`), or a choice."""

    def __init__(self, h: Harness):
        r, s = h.runner, h.sched
        self.k, self.vocab = s.k, r.vocab
        self.rows: dict[int, list[tuple]] = {}
        self.counted: dict[int, np.ndarray] = {}
        self._failed: set[int] | None = None
        launch, process, fail = r.launch_sample, s.process, s._fail

        def launch_sample(handle: Any, n_masks: int = 0) -> None:
            v = int(handle.v)  # the drafts the step verifies, as the scheduler's `_v` (its masks: 1 + v a row)
            seen = [(i, r.drafts[slot, :v].copy(), r._masks[handle.mask_off[i]:handle.mask_off[i] + 1 + v].copy())
                    for i, slot in enumerate(handle.rows[0].tolist()) if handle.mtp[i] and handle.masked[i]]
            if seen and v < self.k:
                event("verify rows with masks at a count below k")
            step = handle.step_id
            launch(handle, n_masks)
            res = handle.result
            st_ = res.status if res.status is not None else np.zeros(len(res.n_out), np.uint32)
            self.rows[step] = [(i, d, m, res.tokens[i].copy(), int(res.n_out[i]), int(st_[i])) for i, d, m in seen]

        def _fail(req: Req, *a: Any, **kw: Any) -> Any:
            if self._failed is not None:
                self._failed.add(int(s.aid[req.slot]))
            return fail(req, *a, **kw)

        def _process(inf: Inflight, *a: Any, **kw: Any) -> Any:
            valid = s.aid[inf.slots] == inf.aids
            held = inf.sample & np.array([(r := s._held.get(a)) is not None and r.status == WAITING
                                          for a in inf.aids.tolist()], bool)
            self._failed = set()
            try:
                return process(inf, *a, **kw)
            finally:
                self.counted[inf.step_id] = (valid & ~np.isin(inf.aids, list(self._failed))) | held
                self._failed = None

        r.launch_sample, s.process, s._fail = launch_sample, _process, _fail

    def expected(self, vocab_sum: bool = True) -> tuple[np.ndarray, np.ndarray, int]:
        """(first rejections by class, the masks read by popcount bucket, their popcounts summed): over the vocabulary
        (`vocab_sum`), or over the mask words as the scheduler reads them (an all-ones mask counts its padding bits)."""
        rej = np.zeros(len(MASK_CLASSES), np.int64)
        hist = np.zeros(len(POPCOUNT_BUCKETS) + 1, np.int64)
        total = 0
        for step, rows in self.rows.items():
            counted = self.counted.get(step)
            if counted is None:
                continue
            for i, drafts, masks, toks, n, status in rows:
                if not counted[i] or status:
                    continue
                acc = n - 1
                for j in range(acc + 1):
                    p = int(unpack_mask(masks[j], self.vocab).sum())
                    hist[np.searchsorted(POPCOUNT_BUCKETS, p)] += 1
                    total += p if vocab_sum else int(np.bitwise_count(masks[j].view(np.uint32)).sum())
                if acc < len(drafts) and drafts[acc] != toks[acc]:  # a draft verified and rejected
                    a = unpack_mask(masks[acc], self.vocab)
                    p = int(a.sum())
                    rej[(EOS_CLASS if a[EOS] else FORCED) if p == 1 else FREE if a.all() else CHOICE] += 1
        return rej, hist, total

    def read_classes(self) -> Counter:
        """The masks verification read, by class (as the first rejections are classed)."""
        out: Counter = Counter()
        for step, rows in self.rows.items():
            counted = self.counted.get(step)
            for i, _, masks, _, n, status in rows:
                if counted is not None and counted[i] and not status:
                    for j in range(n):
                        a = unpack_mask(masks[j], self.vocab)
                        p = int(a.sum())
                        out[(EOS_CLASS if a[EOS] else FORCED) if p == 1 else FREE if a.all() else CHOICE] += 1
        return out

    def check(self, s: Scheduler, vocab_sum: bool = True) -> None:
        rej, hist, total = self.expected(vocab_sum)
        assert s.first_rejections.tolist() == rej.tolist(), (s.first_rejections.tolist(), rej.tolist())
        assert s.window_masks.tolist() == hist.tolist(), (s.window_masks.tolist(), hist.tolist())
        assert s.window_masks_sum == total, (s.window_masks_sum, total)


def run(h: Harness, failed_ok: bool = False) -> Harness:
    """A full run with every check: each turn's accounting (and `check_unrep`), the plans, the NaN invariants, the
    outputs against the predictions, and the grammars' state at the end."""
    h.run(check_every_turn=True)
    check_plans(h)
    check_nan_invariants(h)
    h.check(failed_ok=failed_ok)
    s = h.sched
    assert s.grammars.errors == 0 and not s.n_masked and not s.masked.any() and not s.unrep.any() and s._n_unrep == 0
    return h


MODES = ["window16", "window32", "gdn", "attention"]


def row_at(h: Harness, pos0: int) -> tuple[int, list[int]]:
    """The explicit row of the first plan that has one starting at `pos0`: (its n_query, its CKPT offsets in order)."""
    p = next(p for p in h.plans if pos0 in p["pos0"])
    row = p["pos0"].index(pos0)
    return p["n_query"][row], sorted(o for r, o in p["ckpt"] if r == row)


def constrained_request(draw: Any, rid: int, prompt: list[int], spans: list[tuple[int, ...]], seed: int,
                        max_model_len: int, arrive: int | None = None, hints: list[int] = (),
                        mrope: bool = False) -> Spec:
    """A request of the targeted suites over `prompt` and its image `spans`: on most a grammar with forced runs (a
    lead of 0-3 tokens, a tail of 0-2, lo == hi most often: JSON documents); max_tokens from 1 up; ignore_eos; an
    arrival (unless given), an abort, a priority, an EOS position now and then; now and then a constrained request its
    frontend ends on stop strings (`AddRequest.stop_strings`: neither switch applies to it); `nan_write`; stop ids on a
    forced token or from the predicted output (`mrope`: under Qwen's M-RoPE images); a NaN (any request) or an empty
    mask (a constrained one that stops at its EOS) at a forced position, the EOS position, the first sampled one or
    around them."""
    grammar = ""
    if draw(st.integers(0, 3)):
        lo = draw(st.integers(0, 14))
        hi = lo if draw(st.integers(0, 3)) else lo + draw(st.integers(1, 10))
        grammar = spec(draw(st.integers(0, 1 << 30)), lo, hi, lead=draw(st.integers(0, 3)),
                       tail=min(hi, draw(st.integers(0, 2))))
    sp = Spec(rid=rid, prompt_len=len(prompt), prompt=prompt, images=spans, seed=seed, grammar=grammar,
              max_tokens=draw(st.integers(5, 40)) if draw(st.integers(0, 3)) else draw(st.integers(1, 4)),
              arrive=draw(st.integers(0, 12)) if arrive is None else arrive, ignore_eos=not draw(st.integers(0, 4)),
              abort_at=None if draw(st.integers(0, 3)) else draw(st.integers(0, 40)),
              priority=draw(st.integers(0, 2)), eos_after=None if draw(st.integers(0, 5)) else draw(
                  st.integers(0, 20)), stop_strings=bool(grammar) and not draw(st.integers(0, 5)), hints=list(hints),
              **nan_write(draw, len(prompt)))
    out = predicted(sp, max_model_len, mrope)
    stop = draw(st.integers(0, 5))
    f = forced_of(grammar) if grammar else ()
    if stop == 0 and f:
        sp.stop_ids = [draw(st.sampled_from(f))]
    elif stop == 1:
        sp.stop_ids = [draw(st.sampled_from(out))]
    inj = draw(st.integers(0, 4))
    if inj in (1, 2):
        end = end_of(sp, out)
        pos = draw(st.sampled_from(sorted({0, 1, max(0, len(f) - 1), len(f), max(0, end - 1), end,
                                           draw(st.integers(0, max(0, end)))})))
        if inj == 2 and grammar and not sp.ignore_eos:
            sp.mask_empty_after = pos
        else:
            sp.nonfinite_after = pos
    return sp


def image_prompt(draw: Any, prompt: list[int]) -> tuple[list[int], list[tuple[int, ...]]]:
    """`prompt` followed by 0-2 images of 1-24 tokens (ids shared: encoder-cache and prefix hits), each whole with text
    after it: (the prompt, its spans)."""
    spans = []
    for _ in range(draw(st.integers(0, 2))):
        iid = draw(st.integers(1, 6))
        gh, gw = GRIDS[iid % len(GRIDS)]
        spans.append((len(prompt), gh, gw, iid))
        prompt = prompt + [0] * (gh * gw // 4) + draw(st.lists(st.integers(0, 979), min_size=1, max_size=5))
    return prompt, spans


@st.composite
def structured_window_workloads(draw):
    """Gemma 4's shape with structured output (the forced prefix and the EOS-free finish are a class of hazard of
    their own, so they get targeted suites: architecture.md, "Testing"): window mode (windows of 8-32 over window
    pages of 2 or 4, full pages of 1-5 window pages), its assistant's drafter without KV at k up to 7 (5 the most
    often), acceptance 0-1; the requests of `constrained_request`, their prompts repeated (hits) or long enough for
    several chunks; priorities, chunk caps, the mixed-step bound and pools from the startup bound up (preemption with
    final chunks in flight); NaN reported by every explicit row as Gemma 4's runner does (three workloads in four); in
    one workload in three, Gemma 4's bidirectional image spans in the prompts (`image_prompt`), as a request may carry
    an image and a JSON grammar; the switches drawn on or off independently."""
    k = draw(st.sampled_from([0, 1, 3, 5, 5, 7]))
    wps = draw(st.sampled_from([2, 4]))
    window = draw(st.sampled_from([8, 16, 32]))
    full = wps * draw(st.sampled_from([1, 2, 5]))
    max_model_len = 160
    stems = [stem(i + 3, draw(st.integers(8, 100))) for i in range(draw(st.integers(1, 3)))]
    images = not draw(st.integers(0, 2))  # Gemma 4's bidirectional image spans, in one workload in three
    n = draw(st.integers(2, 7))
    base = draw(st.integers(1 << 32, 1 << 61))
    specs: list[Spec] = []
    for rid in range(n):
        if specs and not draw(st.integers(0, 4)):  # an earlier prompt again: a hit at its end
            prev = draw(st.sampled_from(specs))
            prompt, spans = list(prev.prompt), list(prev.images)
        else:
            s = draw(st.sampled_from(stems))
            prompt, spans = s[:draw(st.integers(1, len(s)))], []
            if images:
                prompt, spans = image_prompt(draw, prompt)
            prompt = prompt + draw(st.lists(st.integers(0, VOCAB - 2), min_size=1, max_size=8))
            prompt = [t if t != EOS else 0 for t in prompt][:110]
            spans = [im for im in spans if im[0] + im[1] * im[2] // 4 < len(prompt)]
        specs.append(constrained_request(draw, rid, prompt, spans, base + 7919 * rid, max_model_len))
    max_tokens = draw(st.integers(4 * (1 + k), 96))
    need = bound(window, wps, full, max_tokens, max_model_len, k)
    return dict(specs=specs, page_size=full, window_page_size=wps, window=window, grid=wps * draw(st.sampled_from(
        [1, 2])), max_model_len=max_model_len, mtp_k=k, accept_p=draw(st.sampled_from([0.0, 0.6, 0.9, 1.0])),
        draft_kv=False, max_num_seqs=draw(st.integers(1, 5)), max_tokens=max_tokens,
        num_pages=need + 1 + draw(st.sampled_from([0, 1, 2, 3, 5, 8, 13, need])),
        policy=draw(st.sampled_from(["fcfs", "priority"])),
        chunk_cap=draw(st.sampled_from([0, 0, 9, 24])),
        decode_checkpoint_interval=draw(st.sampled_from([8, 16, 256])),
        row_tokens=draw(st.sampled_from([0, 0, 40, 160])), prefix_caching=draw(st.sampled_from([True, True, False])),
        report_unsampled=draw(st.integers(0, 3)) > 0, forced_prefix=draw(st.booleans()),
        eos_free_finish=draw(st.booleans()),
        **(dict(bidirectional=True, encoder_rows=draw(st.sampled_from([32, 48, 128])),
                encoder_budget=draw(st.sampled_from([24, 32, 64]))) if images else {}))


@st.composite
def structured_window_count_workloads(draw):
    """`structured_window_workloads` at k of 2 or more with a draft count per step (`mtp_workloads.with_counts`): the
    forced prefix, the EOS-free finish and the verify-mask counters while the verified and drafted counts change."""
    from .mtp_workloads import with_counts

    w = draw(structured_window_workloads())
    assume(w["mtp_k"] > 1)  # a count below k
    return with_counts(draw, w)


@st.composite
def structured_hybrid_workloads(draw):
    """Qwen's shape with structured output: a hybrid GDN model, whose state checkpoints are CKPT records (on a grid of 8
    or 16 inside a chunk, or at its n_query: on a final chunk, at its prompt's end, only for a branch point that a
    request admitted after it in the same plan waits for, where its forced tokens meet it, on the grid or off it), its
    MTP layer's drafter with KV at k up to 3, acceptance 0-1, NONFINITE reported on SAMPLE rows only (four workloads in
    five); pages of 1-8 positions, 1-6 checkpoint slots, decode checkpoints every 8, 16 or 256 positions, the prefix
    cache on in most workloads; the requests of `constrained_request`, their prompts from shared stems, an earlier
    prompt again, a follow-up turn that continues an earlier request's predicted answer (most often a constrained one's,
    through its grammar's end, with its EOS or without it, as a chat template's next turn holds the reply, or cut
    anywhere: the EOS-free finish's donations are hit), or a strict extension of an earlier prompt arriving with it
    (that branch point), each now and then cut to a whole number of grids (its end on the grid); hints, now and then one
    just before the prompt's end (an end-of-chunk checkpoint, the final chunk its last token) or at it (dropped: hints
    lie inside the prompt); chunk caps on and off the grid (final chunks start and end on and off it); the precise
    prefix (`fp8`, `bf16`, `reuse`: refine targets end chunks); in one workload in three, Qwen's causal M-RoPE image
    spans in the prompts (`image_prompt`: chunks may split them; the forced tokens take the text's delta after the last
    one); priorities, the mixed-step bound and pools from the startup bound up (preemption); the switches drawn on (two
    workloads in three) or off independently."""
    k = draw(st.sampled_from([0, 1, 2, 3, 3]))
    page_size = draw(st.sampled_from([1, 2, 4, 8]))
    grid = draw(st.sampled_from([8, 16]))
    max_model_len = 160
    stems = [stem(i + 3, draw(st.integers(8, 100))) for i in range(draw(st.integers(1, 3)))]
    mrope = not draw(st.integers(0, 2))  # Qwen's causal M-RoPE image spans, in one workload in three
    n = draw(st.integers(2, 8))
    base = draw(st.integers(1 << 32, 1 << 61))
    specs: list[Spec] = []
    for rid in range(n):
        kind = draw(st.sampled_from(["stem", "stem", "again", "turn", "turn", "extend"])) if specs else "stem"
        arrive = None
        if kind == "stem":
            s = draw(st.sampled_from(stems))
            prompt, spans = s[:draw(st.integers(1, len(s)))], []
            if mrope:
                prompt, spans = image_prompt(draw, prompt)
            prompt = prompt + draw(st.lists(st.integers(0, VOCAB - 2), min_size=1, max_size=8))
        else:
            constrained = [sp for sp in specs if sp.grammar and not sp.ignore_eos]
            prev = draw(st.sampled_from(constrained if kind == "turn" and constrained and draw(st.integers(0, 2))
                                        else specs))
            prompt, spans = list(prev.prompt), list(prev.images)
            if kind == "turn":  # the earlier answer as its prompt holds it, then the new message
                out = predicted(prev, max_model_len, mrope)
                end = end_of(prev, out)
                cut = draw(st.sampled_from(["end", "without its EOS", "anywhere"]))
                keep = end if cut == "end" else end - (out[end - 1] == EOS) if cut == "without its EOS" else \
                    draw(st.integers(0, end))
                prompt += out[:keep] + draw(st.lists(st.integers(0, VOCAB - 2), min_size=1, max_size=8))
                arrive = prev.arrive + draw(st.integers(0, 40))
            elif kind == "extend":  # admitted with it: a branch point at its end
                prompt += draw(st.lists(st.integers(0, VOCAB - 2), min_size=1, max_size=12))
                arrive = prev.arrive
        prompt = prompt[:150]
        if len(prompt) > 16 and not draw(st.integers(0, 3)):
            prompt = prompt[:len(prompt) // 16 * 16]
        spans = [im for im in spans if im[0] + im[1] * im[2] // 4 < len(prompt)]
        hints = draw(st.lists(st.integers(1, len(prompt)), max_size=2))
        if not draw(st.integers(0, 3)):
            hints.append(max(1, len(prompt) - draw(st.integers(0, 1))))
        specs.append(constrained_request(draw, rid, prompt, spans, base + 7919 * rid, max_model_len, arrive, hints,
                                         mrope))
    max_tokens = draw(st.integers(4 * (1 + k), 96))
    max_num_seqs = draw(st.integers(1, 5))
    need = pages_for_one_request(max_model_len, page_size, reserve_positions(2, k), True)
    precise = draw(st.sampled_from(["fp8", "fp8", "bf16", "reuse"]))
    policy = {}
    if precise == "reuse":
        policy = dict(refine_after=draw(st.sampled_from([0, 1, 1, 2])), refine_min=draw(st.sampled_from([0, 0, 8, 40])),
                      **refine_options(draw, grid, max_tokens, max_num_seqs, k))
    return dict(specs=specs, page_size=page_size, grid=grid, max_model_len=max_model_len, mtp_k=k,
                accept_p=draw(st.sampled_from([0.0, 0.6, 0.9, 1.0])), draft_kv=True, max_num_seqs=max_num_seqs,
                max_tokens=max_tokens, num_pages=need + 1 + draw(st.sampled_from([0, 1, 2, 3, 5, 8, 13, need])),
                checkpoint_slots=draw(st.sampled_from([1, 2, 3, 4, 6, 6, 6])),
                policy=draw(st.sampled_from(["fcfs", "priority"])),
                chunk_cap=draw(st.sampled_from([0, 0, 8, 9, 13, 16, 24])),
                decode_checkpoint_interval=draw(st.sampled_from([8, 16, 256])),
                row_tokens=draw(st.sampled_from([0, 0, 40, 160])),
                prefix_caching=draw(st.sampled_from([True, True, True, False])),
                report_unsampled=not draw(st.integers(0, 4)), precise=precise,
                precise_cost=draw(st.sampled_from([1.0, 2.5])) if precise != "fp8" else 1.0, **policy,
                forced_prefix=draw(st.integers(0, 2)) > 0, eos_free_finish=draw(st.integers(0, 2)) > 0,
                **(dict(encoder_rows=draw(st.sampled_from([32, 48, 128])),
                        encoder_budget=draw(st.sampled_from([24, 32, 64]))) if mrope else {}))


@st.composite
def structured_hybrid_count_workloads(draw):
    """`structured_hybrid_workloads` at k of 2 or 3 with a draft count per step (`mtp_workloads.with_counts`): Qwen's
    shape, its drafter with KV, while the verified and drafted counts change."""
    from .mtp_workloads import with_counts

    w = draw(structured_hybrid_workloads())
    assume(w["mtp_k"] > 1)  # a count below k
    return with_counts(draw, w)


def run_switches_workload(w: dict) -> Harness:
    """One workload of either targeted suite, every check: outputs, completion counts and finish reasons are the
    predictions (the sampled path's) wherever no injection fired; each turn's accounting and unreported tokens
    (`Probe`); the plans; the NaN invariants; the precise prefix's (`check_precise`, every turn) where it is on; no
    grammar error; the forced-prefix and EOS-free counters against what the probe saw, each zero with its switch off,
    and neither taken by a request with stop strings; the verify-mask counters against `MaskOracle`; each injection's
    outcome: one at a position the request samples fails it, one at a position it never samples (a forced token its
    final chunk took, the EOS an EOS-free finish appended) does not (unless another request's NaN reached it)."""
    from .harness import check_precise

    h = Harness(vocab=VOCAB, **w)
    probe = Probe(h)
    oracle = MaskOracle(h) if w["mtp_k"] else None
    if h.sched.precise != "fp8":
        orig = h.core.step

        def step(*a: Any, **kw: Any) -> Any:
            r = orig(*a, **kw)
            check_precise(h)
            return r

        h.core.step = step
    h.run(check_every_turn=True)
    check_plans(h)
    check_nan_invariants(h)
    h.check(failed_ok=True)
    s = h.sched
    assert s.grammars.errors == 0 and not s.n_masked and not s.unrep.any() and s._n_unrep == 0
    probe.check_counters()
    if not w["forced_prefix"]:
        assert not probe.forced
    if not w["eos_free_finish"]:
        assert s.eos_free.sum() == s.eos_free_gated == 0
    assert s.eos_free.sum() == len(probe.appended)
    strings = {sp.rid for sp in h.specs.values() if sp.stop_strings}
    assert not strings & (set(probe.forced) | probe.appended), (strings, probe.forced, probe.appended)
    if strings:
        event("a constrained request with stop strings")
    if oracle is not None:
        oracle.check(s)
    for sp in h.specs.values():
        pos = sp.nonfinite_after if sp.nonfinite_after is not None else sp.mask_empty_after
        r = h.results[sp.rid]
        if pos is None or r.finish == ABORT or sp.rid in h.rejected or sp.nonfinite_write is not None:
            continue
        end = end_of(sp, h.expected(sp))
        if pos >= end:
            continue
        nf = probe.forced.get(sp.rid, (0, ""))[0]
        appended = sp.rid in probe.appended and pos == end - 1
        if pos >= nf and not appended:
            assert r.finish == ERROR, f"request {sp.rid}: its injection at {pos} (sampled) did not fail it"
        else:
            assert r.finish != ERROR or sp.rid in h.cascaded or sp.seed in h.runner.poison_read, \
                f"request {sp.rid} failed at {pos}, a position it never sampled"
            event("an injection at a position never sampled: " + ("the appended EOS" if appended else "forced"))
            if appended and probe.in_flight[sp.rid] and sp.nonfinite_after is not None:
                event("a NaN in the row an EOS-free finish dropped")
    for name, n in probe.events.items():
        if n:
            event(name)
    if probe.forced:
        event("forced prefix taken")
        if any(h.specs[rid].images for rid in probe.forced):
            event("forced prefix taken after an image span")
        if any(c == "_schedule_running_prefills" for _, c in probe.forced.values()):
            event("forced prefix taken by a running prefill's final chunk")
    if s.forced_declined:
        event("forced tokens sampled: a CKPT record off the grid at the prompt's end")
    for i, what in enumerate(("no row in flight", "its row in flight")):
        if s.eos_free[i]:
            event(f"EOS-free finish, {what}")
    if s.eos_free_gated:
        event("EOS-free finish gated")
    if s.num_preemptions:
        event("preempted")
    if s.first_rejections[FORCED]:
        event("a first rejection at a forced position")
    if s.hybrid and not s.window:
        hybrid_events(h, probe)
    return h


def hybrid_events(h: Harness, probe: Probe) -> None:
    """The hybrid suite's cases: a final chunk's CKPT record at its prompt's end (on or off the grid, with forced
    tokens after it or not), a branch checkpoint for a request admitted in the same plan, and a later request that
    resumed where an EOS-free finish donated its state (its row in flight: FOLD 1; none: FOLD_ALL)."""
    g = h.sched.grid
    for p in h.plans:
        for row, off in p["ckpt"]:
            if off <= 0 or p["device"][row] or not p["sample"][row]:
                continue
            nf = probe.tails.get((p["step"], p["slots"][row]), 0)
            if off == p["n_query"][row] - nf:
                event("a CKPT record at a final chunk's prompt end" + (" off the grid" if off % g else "") +
                      (", forced tokens after it" if nf else ""))
    if h.sched.ckpts_taken["branch"]:
        event("a branch checkpoint for a request admitted in the same plan")
    for rid, d in probe.donated.items():
        done = [*key_of(h.specs[rid], VOCAB).tolist(), *h.results[rid].tokens][:d]
        for sp in h.specs.values():
            if h.results[sp.rid].cached == d and key_of(sp, VOCAB)[:d].tolist() == done:
                event("a later request resumed where an EOS-free finish donated its state, " +
                      ("its row in flight" if probe.in_flight[rid] else "no row in flight"))
