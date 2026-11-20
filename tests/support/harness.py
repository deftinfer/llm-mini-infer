"""Drive the engine loop in-process against the fake runner (no ZMQ, no sleeping) and check what comes out."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from llm_mini_infer.engine.core import EngineCore
from llm_mini_infer.engine.messages import Abort, AddRequest, FinishReason, ImageRef
from llm_mini_infer.grammar.engine import Grammars
from llm_mini_infer.runner.base import PoolConfig
from llm_mini_infer.runner.fake import FakeRunner, fake_generate
from llm_mini_infer.sched.plan import MASKED, PRECISE, SAMPLE, StepPlan
from llm_mini_infer.sched.scheduler import RUNNING, Scheduler
from llm_mini_infer.vision.family import KEY_BASE, ImageSize, Span
from llm_mini_infer.vision.qwen import QwenVision

from .fake_grammar import FakeGrammar

EOS = 999
# Set to 1: every harness without an explicit `batch_invariant` runs the scheduler's batch invariance where its budget
# allows (`uv run python -m tests.scripts.fuzz_campaign`, `tests.scripts.plandiff`: a second pass with it on;
# docs/development.md)
BATCH_INVARIANT_ENV = "LLM_MINI_INFER_TEST_BATCH_INVARIANT"
# Set to 1: the strict checks' reference implementations (`FakeRunner(full_checks=True)`, `check_accounting(full=True)`,
# `RadixCache.check(full=True)`), slower, with the same results and errors (docs/development.md, "Fuzzing")
FULL_CHECKS_ENV = "LLM_MINI_INFER_TEST_FULL_CHECKS"
PAD = 990  # the image placeholder token of the harness's prompts
QWEN = QwenVision(start_id=988, pad_id=PAD, end_id=989, patch=16, merge=2, temporal=2, min_pixels=1,
                  max_pixels=1 << 30, mean=(0.5,) * 3, std=(0.5,) * 3, rescale=1 / 255, mrope=True)


def image_refs(images: list[tuple[int, ...]]) -> list[ImageRef]:
    """(offset, patch rows, patch columns, image id[, digest id]) -> ImageRefs (the key from the image id, the digest
    from the digest id, by default the image id: two images of one id and other digest ids are a span key collision;
    tokens = rows x columns / 4)."""
    return [ImageRef(key=KEY_BASE | im[3], offset=im[0], tokens=im[1] * im[2] // 4, grid_t=1, grid_h=im[1],
                     grid_w=im[2], height=16 * im[1], width=16 * im[2], pixels=4096 * i,
                     digest=(im[4] if len(im) > 4 else im[3]).to_bytes(32, "little"))
            for i, im in enumerate(images)]


@dataclass
class Spec:
    rid: int
    prompt_len: int
    max_tokens: int
    seed: int
    arrive: int = 0  # loop turn at which the request arrives
    eos_after: int | None = None  # the fake emits EOS as this output token (0-based)
    ignore_eos: bool = False
    abort_at: int | None = None  # loop turn at which the frontend aborts it
    priority: int = 0
    stop_ids: list[int] = field(default_factory=list)
    prompt: list[int] | None = None  # explicit prompt tokens (default: derived from rid and prompt_len)
    hints: list[int] = field(default_factory=list)  # cache_control offsets
    salt: str | None = None
    nonfinite_after: int | None = None  # the fake reports NONFINITE when sampling this output token
    # (offset, grid h, grid w, image id[, digest id]: `image_refs`)
    images: list[tuple[int, ...]] = field(default_factory=list)
    grammar: str = ""  # a fake grammar's text (fake_grammar.spec), "" = unconstrained
    # the fake writes NaN at this prompt position (the row computing it reports it only if it samples); once only, with
    # nonfinite_once (a transient fault: the request's next computation of the position is finite)
    nonfinite_write: int | None = None
    nonfinite_once: bool = False
    mask_empty_after: int | None = None  # the fake's grammar mask allows nothing when sampling this output token
    stop_strings: bool = False  # its frontend ends it on stop strings (AddRequest.stop_strings; none applied here)
    # The plan this loop turn builds preempts it first, if it runs: where tests want a preemption at a given moment.
    # With `preempt_pages` (full, window), only if those pages are short after evicting the cache, and then for them,
    # as a running request's page pressure does
    preempt_at: int | None = None
    preempt_pages: tuple[int, int] = (0, 0)
    logprobs: int = -1  # its tokens' logprobs with this many top entries (AddRequest.logprobs; -1: none)
    # Penalties and the logit bias (AddRequest's; the fake's tokens ignore them, its checks read the ADMIT)
    penalties: tuple[float, float, float] = (1.0, 0.0, 0.0)  # repetition, presence, frequency
    bias: dict[int, float] = field(default_factory=dict)


@dataclass
class Result:
    tokens: list[int] = field(default_factory=list)
    finish: int = 0
    first_seen: int = 0
    cached: int = -1
    # with `Spec.logprobs`: each token's (logprob, top ids, their logprobs), as the engine sent them
    logprobs: list[tuple[float, list[int], list[float]]] = field(default_factory=list)


def prompt_of(spec: Spec, vocab: int) -> np.ndarray:
    if spec.prompt is not None:
        out = np.asarray(spec.prompt, np.int32).copy()
    else:
        out = ((np.arange(spec.prompt_len, dtype=np.int32) * 7 + spec.rid) % (vocab - 1)).astype(np.int32)
    for off, gh, gw, *_ in spec.images:
        out[off:off + gh * gw // 4] = PAD
    return out


def key_of(spec: Spec, vocab: int) -> np.ndarray:
    """The prompt as the radix key (and the fake's context): image spans hold their keys."""
    key = prompt_of(spec, vocab).astype(np.int64)
    for off, gh, gw, iid, *_ in spec.images:
        key[off:off + gh * gw // 4] = KEY_BASE | iid
    return key


def rope_off_of(spec: Spec) -> int:
    if not spec.images:
        return 0
    spans = [Span(off, ImageSize(16 * gh, 16 * gw, (1, gh, gw), gh * gw // 4)) for off, gh, gw, *_ in spec.images]
    return QWEN.positions(spec.prompt_len, spans)[1]


class Harness:
    def __init__(self, *, max_num_seqs: int = 4, max_tokens: int = 64, page_size: int = 4, num_pages: int = 64,
                 num_state_slots: int | None = None, checkpoint_slots: int = 0, max_model_len: int = 256,
                 vocab: int = 1000, policy: str = "fcfs", chunk_cap: int = 0, specs: list[Spec] = (),
                 hybrid: bool = True, prefix_caching: bool = True, decode_checkpoint_interval: int = 256,
                 grid: int = 64, strict: bool = True, mtp_k: int = 0,
                 accept_p: float = 0.8, encoder_rows: int = 0, encoder_budget: int = 0,
                 window: int = 0, window_page_size: int = 0, row_tokens: int = 0, bidirectional: bool = False,
                 draft_kv: bool = True, precise: str = "fp8", refine_after: int = 1, precise_cost: float = 1.0,
                 refine_min: int = 0, report_unsampled: bool = False, refine_chunk_tokens: int = 0,
                 precise_cap: int = 0, forced_prefix: bool = False, eos_free_finish: bool = False,
                 batch_invariant: bool | None = None, count_schedule: tuple[int, ...] | list[int] = (),
                 adaptive: bool = False):
        """`bidirectional` (Gemma 4's images): image spans attend bidirectionally, positions are 1-D (no M-RoPE).
        `draft_kv` (MTP): False for a drafter that writes no KV (Gemma 4's assistant). `precise`, `refine_after`,
        `precise_cost`, `refine_min`: the precise prefix's policy (the fake takes PRECISE rows unless it is "fp8");
        `refine_min` 0 here (the scheduler's default is 256), so the workloads' short prompts refine too.
        `report_unsampled`: the fake reports NONFINITE on rows that do not sample too (a runner checking every row, as
        Gemma 4's CUDA runner checks its explicit rows; by default, as Qwen's, only SAMPLE rows report it).
        `refine_chunk_tokens`: a refine's piece size (0: derived, as the scheduler's default); `precise_cap`: the
        scheduler's cap on PRECISE chunks (0: none; a refine's piece size when set), with which harnesses vary the
        precise copy's chunking. `forced_prefix`, `eos_free_finish`: the scheduler's grammar switches (off by default,
        as its own). `batch_invariant`: the scheduler's batch invariance (its cuts, the strict fake checking them
        against the requests' hints); None: off, unless `BATCH_INVARIANT_ENV` asks for it wherever the workload's
        budget lets a prefill progress under it (a pass of the fuzz campaign or the plan-diff harness with it on).
        `count_schedule` (MTP): the scheduler's dev schedule of draft counts; `adaptive`: its policy over every count
        below k (`sched.draft_counts`), which the fake takes all of."""
        self.vocab = vocab
        self.mrope = bool(encoder_rows) and not bidirectional
        self.grammar = FakeGrammar(vocab, (EOS,))
        self.specs = {s.rid: s for s in specs}
        for s in self.specs.values():
            if s.prompt is not None:
                s.prompt_len = len(s.prompt)
        eos_at: dict[int, list[int]] = {}
        nonfinite_at: dict[int, list[int]] = {}
        nonfinite_write_at: dict[int, list[int]] = {}
        mask_empty_at: dict[int, list[int]] = {}
        for s in self.specs.values():
            if s.eos_after is not None:
                eos_at.setdefault(s.seed, []).append(s.prompt_len + s.eos_after)
            if s.nonfinite_after is not None:
                nonfinite_at.setdefault(s.seed, []).append(s.prompt_len + s.nonfinite_after)
            if s.nonfinite_write is not None:
                nonfinite_write_at.setdefault(s.seed, []).append(s.nonfinite_write)
            if s.mask_empty_after is not None:
                mask_empty_at.setdefault(s.seed, []).append(s.prompt_len + s.mask_empty_after)
        nss = num_state_slots or max_num_seqs + 1 + (0 if window else checkpoint_slots)
        # Window mode: one arena of `num_pages` pages, full pages of `page_size` positions and window pages of
        # `window_page_size` (default: page_size)
        wps = (window_page_size or page_size) if window else 0
        want = batch_invariant if batch_invariant is not None else os.environ.get(BATCH_INVARIANT_ENV) == "1"
        self.full_checks = os.environ.get(FULL_CHECKS_ENV) == "1"

        def scheduler(inv: bool) -> Scheduler:
            return Scheduler(max_num_seqs=max_num_seqs, max_num_batched_tokens=max_tokens,
                             max_model_len=max_model_len, page_size=page_size, num_pages=num_pages,
                             num_state_slots=nss, eos_token_ids=(EOS,), policy=policy,
                             long_prefill_token_threshold=chunk_cap, hybrid=hybrid, prefix_caching=prefix_caching,
                             decode_checkpoint_interval=decode_checkpoint_interval, checkpoint_grid=grid,
                             mtp_k=mtp_k, encoder_rows=encoder_rows,
                             encoder_budget=encoder_budget, positions=QWEN.positions if self.mrope else None,
                             grammars=Grammars(self.grammar), window=window, window_page_size=wps,
                             mixed_step_row_tokens=row_tokens, bidirectional_images=bidirectional,
                             draft_kv=draft_kv, precise=precise, refine_after=refine_after, refine_min=refine_min,
                             precise_cost=precise_cost, refine_chunk_tokens=refine_chunk_tokens,
                             forced_prefix=forced_prefix, eos_free_finish=eos_free_finish, batch_invariant=inv,
                             logprobs=True, penalties_vocab=vocab, count_schedule=tuple(count_schedule),
                             draft_counts=tuple(range(1, mtp_k)) if adaptive else ())

        try:
            self.sched = scheduler(want)
        except ValueError:  # the environment's request, where the budget would starve a prefill under it
            if batch_invariant:
                raise
            self.sched = scheduler(False)
        self.batch_invariant = bool(self.sched.aligned or self.sched.forced_whole)
        p = self.sched.count_policy
        if p is not None:  # the fake reports no device time: a step's from its rows and counts, probes every 2 steps
            observe = p.observe_cost
            p.observe_cost = lambda rows, v, d, ns: observe(rows, v, d, int((22 + 0.15 * max(0, rows * (1 + v) - 24)
                                                                             + 0.4 * rows * d) * 1e6))
            p.explore_every, p.refresh_every, p.min_steps = 2, 16, 1
        self.runner = FakeRunner(vocab_size=vocab, max_positions=max_model_len, step_ms=0, sample_ms=0, spin_us=0,
                                 eos_at=eos_at, nonfinite_at=nonfinite_at, strict=strict, checkpoint_grid=grid,
                                 state_slot_bytes=153_954_304 if hybrid and not window else 0, mtp_k=mtp_k,
                                 accept_p=accept_p, encoder_rows=encoder_rows, encoder_budget=encoder_budget,
                                 window=window, window_kv_bytes_per_token=819_200 if window else 0,
                                 bidirectional_images=bidirectional, draft_kv=draft_kv,
                                 precise_rows=precise != "fp8", nonfinite_write_at=nonfinite_write_at,
                                 report_unsampled=report_unsampled, mask_empty_at=mask_empty_at,
                                 nonfinite_once=[s.seed for s in self.specs.values() if s.nonfinite_once],
                                 batch_invariant=self.batch_invariant, full_checks=self.full_checks,
                                 exact_cuts={s.seed: [*s.hints, *(o - d for o, *_ in s.images for d in (0, 1))]
                                             for s in self.specs.values()})
        self.runner.configure(PoolConfig(page_size=page_size, num_pages=num_pages, num_state_slots=nss,
                                         max_num_seqs=max_num_seqs, max_num_batched_tokens=max_tokens,
                                         max_model_len=max_model_len, eos_token_ids=(EOS,), mtp_k=mtp_k,
                                         window_page_size=wps))
        self.sched.precise_cap = precise_cap
        if any(sp.preempt_at is not None for sp in self.specs.values()):
            plan = self.sched._schedule

            def preempting() -> object:  # `Spec.preempt_at`: before the plan's rows, inside its construction
                for sp in self.specs.values():
                    req = self.sched.reqs.get((0, sp.rid))
                    if sp.preempt_at == self.turn and req is not None and req.status == RUNNING:
                        need = sp.preempt_pages
                        if not any(need) or not self.sched._ensure_pages(*need):
                            self.sched._preempt(req.slot)
                            assert not any(need) or self.sched._ensure_pages(*need), "the pages it was preempted for"
                return plan()

            self.sched._schedule = preempting
        self.core = EngineCore(self.sched, self.runner)
        self.plans: list[dict] = []
        orig = self.runner.launch_forward

        def record(plan: StepPlan):
            flags = plan.sched_flags.tolist()  # the rows' flags in Python: numpy costs more per call on so few rows
            self.plans.append({
                "step": plan.step_id, "free": plan.free_slot.tolist(), "admit": plan.admit_slot.tolist(),
                "seq_len": plan.admit_seq_len.tolist(), "slots": plan.sched_slot.tolist(),
                "n_query": plan.sched_n_query.tolist(), "pos0": plan.sched_pos0.tolist(),
                "fold": list(zip(plan.fold_slot.tolist(), plan.fold_n.tolist())),
                "device": [off < 0 for off in plan.sched_tok_off.tolist()], "sample": [bool(f & SAMPLE) for f in flags],
                "n_tokens": plan.n_query_tokens, "kv_copy": len(plan.kv_copy_src),
                "state_copy": len(plan.state_copy_src), "ckpt": list(zip(plan.ckpt_sched.tolist(),
                                                                         plan.ckpt_offset.tolist())),
                "encode": plan.enc_tokens.tolist(), "next_emb": plan.sched_next_emb.tolist(),
                "image_tokens": int((plan.tok_emb >= 0).sum()) if len(plan.tok_emb) else 0,
                "spans": list(zip(plan.span_sched.tolist(), plan.span_start.tolist(), plan.span_end.tolist())),
                "masked": [bool(f & MASKED) for f in flags], "precise": [bool(f & PRECISE) for f in flags]})
            return orig(plan)

        self.runner.launch_forward = record
        # The causes of ERROR finishes (`check_nan_invariants`): requests the scheduler failed for resuming from a
        # rolled-back write (`Scheduler._fail_cascade`), and requests rejected at `add`
        self.cascaded: set[int] = set()
        self.rejected: set[int] = set()
        cascade = self.sched._fail_cascade

        def record_cascade(bad):
            before = {r.rid for r in self.sched.slot_req if r is not None}
            cascade(bad)
            self.cascaded |= before - {r.rid for r in self.sched.slot_req if r is not None}

        self.sched._fail_cascade = record_cascade
        self.results: dict[int, Result] = {}
        self.released: set[int] = set()  # requests whose pixels the engine released
        self.turn = 0
        self.dropped_after_finish = 0
        # every output token the engine sent each request (those after its finish on our side too), and the verify
        # rows and tokens it reported with each finish (MTP): `check_spec_counts`
        self.sent: dict[int, int] = {}
        self.spec: dict[int, tuple[int, int]] = {}

    def expected(self, spec: Spec, logprobs: list | None = None) -> list[int]:
        """The fake's full output for a request: one token per position (a function of seed, position and context),
        EOS where asked; `logprobs`: each one's values appended (`fake_logprobs`)."""
        limit = min(spec.max_tokens, self.sched.max_model_len - spec.prompt_len)
        eos = {spec.prompt_len + spec.eos_after: EOS} if spec.eos_after is not None else {}
        matcher = self.grammar.instance(self.grammar.compile(spec.grammar)) if spec.grammar else None
        return fake_generate(key_of(spec, self.vocab), spec.seed, limit, self.vocab, (EOS,), eos,
                             rope_off=rope_off_of(spec) if self.mrope else 0, matcher=matcher, eos_ids=(EOS,),
                             logprobs=logprobs)

    def add(self, spec: Spec) -> None:
        self.core.handle(AddRequest(rid=spec.rid, frontend=0, prompt=prompt_of(spec, self.vocab),
                                    max_tokens=spec.max_tokens, temperature=0.0, top_p=1.0, top_k=0, min_p=0.0,
                                    seed=spec.seed, stop_token_ids=spec.stop_ids, ignore_eos=spec.ignore_eos,
                                    priority=spec.priority, cache_salt=spec.salt, cache_hints=spec.hints,
                                    images=image_refs(spec.images), grammar=spec.grammar,
                                    stop_strings=spec.stop_strings, logprobs=spec.logprobs,
                                    repetition_penalty=spec.penalties[0], presence_penalty=spec.penalties[1],
                                    frequency_penalty=spec.penalties[2], bias_ids=sorted(spec.bias),
                                    bias_values=[spec.bias[t] for t in sorted(spec.bias)]))
        self.results[spec.rid] = Result()
        if (0, spec.rid) not in self.sched.reqs:
            self.rejected.add(spec.rid)

    def run(self, max_turns: int = 10_000, check_every_turn: bool = False) -> None:
        pending = sorted(self.specs.values(), key=lambda s: s.arrive)
        while self.turn < max_turns:
            while pending and pending[0].arrive <= self.turn:
                self.add(pending.pop(0))
            for s in self.specs.values():
                if s.abort_at == self.turn and s.rid in self.results and not self.results[s.rid].finish:
                    self.core.handle(Abort(frontend=0, rids=[s.rid]))
                    self.results[s.rid].finish = FinishReason.ABORT
            report = self.core.step()
            if report is not None:
                self.collect(report)
            if check_every_turn:
                self.sched.check_accounting(self.full_checks)
            self.turn += 1
            if not pending and not self.core.busy:
                return
        raise AssertionError(f"not idle after {max_turns} turns")

    def collect(self, report) -> None:
        for fe, out in report.outputs.items():
            assert fe == 0
            for rid in out.released:
                assert rid not in self.released, f"request {rid}: pixels released twice"
                self.released.add(rid)
            for rid, cached in zip(out.first_rids.tolist(), out.first_cached.tolist()):
                self.results[rid].cached = cached
            for rid, n, t in zip(out.spec_rids, out.spec_rows, out.spec_tokens):
                assert rid not in self.spec, f"request {rid}: its verify counts reported twice"
                self.spec[rid] = (n, t)
            lp = out.logprobs
            if lp is not None:
                assert len(lp[0]) == len(out.tokens) and lp[1].shape == lp[2].shape == (len(out.tokens), lp[1].shape[1])
            off = 0
            for rid, n, fin in zip(out.rids.tolist(), out.n_new.tolist(), out.finish.tolist()):
                r = self.results[rid]
                toks = out.tokens[off:off + n].tolist()
                k = self.specs[rid].logprobs
                if k >= 0 and n:  # its tokens' logprobs, in the message's first k top columns
                    assert lp is not None and lp[1].shape[1] >= k, f"request {rid}: no logprobs with its tokens"
                    vals = [(float(lp[0][i]), lp[1][i, :k].tolist(), lp[2][i, :k].tolist())
                            for i in range(off, off + n)]
                off += n
                self.sent[rid] = self.sent.get(rid, 0) + n
                if r.finish:  # finished or aborted on our side: the engine may still send what was in flight
                    self.dropped_after_finish += 1
                    continue
                r.tokens.extend(toks)
                if k >= 0 and n:
                    r.logprobs.extend(vals)
                r.finish = fin
            assert off == len(out.tokens)

    def check(self, failed_ok: bool = False) -> None:
        """Every request got a prefix of its predicted output, ending exactly at its finish condition; then every page
        and state slot comes back once the cache is cleared."""
        for rid, spec in self.specs.items():
            r = self.results[rid]
            values: list | None = [] if spec.logprobs >= 0 else None
            exp = self.expected(spec, values)
            assert r.tokens == exp[:len(r.tokens)], f"request {rid}: tokens differ from the prediction"
            if values is not None:  # each token's logprobs, from the logits its sample read (or forced: 0)
                k = spec.logprobs
                assert len(r.logprobs) == len(r.tokens), (rid, len(r.logprobs), len(r.tokens))
                for i, (got, (v, ids, vals)) in enumerate(zip(r.logprobs, values)):
                    assert got == (v, ids[:k], vals[:k]), f"request {rid}, token {i}: logprobs {got}"
            if r.finish == FinishReason.ABORT or (failed_ok and r.finish == FinishReason.ERROR):
                continue
            stops = [i for i, t in enumerate(exp) if (t == EOS and not spec.ignore_eos) or t in spec.stop_ids]
            if stops:
                assert r.finish == FinishReason.STOP and len(r.tokens) == stops[0] + 1, (rid, r.finish, stops)
            else:
                assert r.finish == FinishReason.LENGTH and len(r.tokens) == len(exp), (rid, r.finish, len(r.tokens))
        s = self.sched
        assert s.num_running == 0 and s.num_waiting == 0 and not s.reqs
        self.check_spec_counts()
        with_images = {rid for rid, sp in self.specs.items() if sp.images}
        assert self.released == with_images, f"pixel releases {sorted(self.released)} != {sorted(with_images)}"
        s.check_accounting(self.full_checks)
        if s.cache is not None:  # the checkpoints' counters by source (llm_mini_infer:checkpoint_*_total)
            c = s.cache
            assert sum(map(sum, c.evicted_by)) == c.evicted_ckpts, (c.evicted_by, c.evicted_ckpts)
            if s.hybrid:
                assert sum(s.ckpt_resumes) == s.ckpt_hits, (s.ckpt_resumes, s.ckpt_hits)
        if s.encoder is not None:
            assert all(not e.refs for e in s.encoder.entries.values()), "encoder references leaked"
        if s.cache is not None:
            s.cache.clear()
            s.check_accounting(self.full_checks)
        assert s.kv.num_free == s.kv.num_pages - 1, "pages leaked"
        assert s.states.num_free_live == s.states.n_live and s.states.num_free_ckpt == s.states.n_ckpt, \
            "state slots leaked"


    def check_spec_counts(self) -> None:
        """The speculative-decoding counters reconcile with the tokens sent (MTP): every output token after a request's
        first is a verify row's (drafts and accepted drafts, less the accepted ones a finish cut:
        `Scheduler.spec_trimmed`), a forced prefix's, the EOS an EOS-free finish appended, or a readmission's first
        sample (`Scheduler.resampled`); and the verify rows and tokens each finish reported (`Scheduler.verify_rows`)
        sum to those counters. Exactly when every request finished in the engine; an abort or a failure leaves counted
        tokens unsent (a forced prefix not yet reported, rows of a request that reported none), never the reverse."""
        s = self.sched
        if not s.k:
            assert not self.spec
            return
        after_first = sum(n - 1 for n in self.sent.values() if n)
        verified = s.spec_drafts + s.spec_accepted - s.spec_trimmed
        counted = verified + s.forced_tokens + int(s.eos_free.sum()) + s.resampled
        rows, toks = sum(n for n, _ in self.spec.values()), sum(t for _, t in self.spec.values())
        why = (after_first, rows, toks, s.spec_drafts, s.spec_accepted, s.spec_trimmed, s.forced_tokens,
               s.eos_free.tolist(), s.resampled)
        if all(r.finish in (FinishReason.STOP, FinishReason.LENGTH) for r in self.results.values()):
            assert set(self.spec) == {rid for rid, n in self.sent.items() if n}, why
            assert after_first == counted and (rows, toks) == (s.spec_drafts, verified), why
        else:
            assert after_first <= counted and rows <= s.spec_drafts and toks <= verified, why


def check_precise(h: Harness) -> int:
    """The precise prefix's invariant: every node of a precise namespace holds KV that PRECISE rows wrote (strict
    mode), and every checkpoint there was computed by PRECISE rows from a precise start (window models: its window's
    KV). Returns the precise nodes checked."""
    from llm_mini_infer.cache.radix import is_precise

    s, r = h.sched, h.runner
    cache = s.cache
    if cache is None:
        return 0
    ps, n = s.ps, 0
    for ns, root in cache.roots.items():
        if not is_precise(ns):
            continue
        stack = list(root.children.values())
        while stack:
            node = stack.pop()
            stack.extend(node.children.values())
            n += 1
            if r.strict:
                for pos in range(node.start, node.end):
                    i = pos // ps
                    page = node.pages[i - node.start // ps] if i < node.end // ps else node.tail
                    assert r.kv_prec[page, pos % ps], f"precise node {node}: KV at {pos} not written by a PRECISE row"
            if not node.ckpt:
                continue
            if s.window:
                lo, pages = s.windows.get(node.ckpt)
                wps = s.wps
                for pos in range(max(0, node.end - s.window + 1), node.end):
                    page = pages[pos // wps - lo]
                    assert r.wkv_prec[page, pos % wps], f"precise window at {node.end}: position {pos} not precise"
            else:
                assert r.fp_prec[node.ckpt], f"precise node {node}: its checkpoint is not precise"
    return n


def poisoned_entries(h: Harness) -> list[str]:
    """What the cache holds that NaN poisoned (strict mode, the fake's marks): nodes whose KV holds a poisoned position,
    and checkpoints whose state (GDN) or window (sliding-window models) is poisoned, in every namespace."""
    s, r = h.sched, h.runner
    cache = s.cache
    if cache is None or not r.strict:
        return []
    ps, out = s.ps, []
    for root in cache.roots.values():
        stack = list(root.children.values())
        while stack:
            node = stack.pop()
            stack.extend(node.children.values())
            pos = np.arange(node.start, node.end)
            held = np.asarray([*node.pages, node.tail], np.int64)
            pages = held[np.minimum(pos // ps - node.start // ps, len(node.pages))]
            if r.kv_bad[pages, pos % ps].any():
                out.append(f"{node}: KV at {int(pos[r.kv_bad[pages, pos % ps]][0])}")
            if not node.ckpt:
                continue
            if s.window:
                lo, wp = s.windows.get(node.ckpt)
                seen = np.arange(max(0, node.end - s.window + 1), node.end)
                if r.wkv_bad[np.asarray(wp, np.int64)[seen // s.wps - lo], seen % s.wps].any():
                    out.append(f"{node}: its window")
            elif r.fp_bad[node.ckpt]:
                out.append(f"{node}: its checkpoint")
    return out


def check_nan_invariants(h: Harness) -> None:
    """The NaN invariants of a finished run (the fuzz suites; before `Harness.check` clears the cache): the cache holds
    nothing NaN poisoned, and every request that ended with an error had a cause: its own NaN or empty grammar mask
    (one the fake injected: `fired`), a read of poisoned state or KV, a rolled-back write its hit used
    (`Scheduler._fail_cascade`), a rejection."""
    bad = poisoned_entries(h)
    assert not bad, f"the cache holds poisoned entries: {bad[:3]}"
    r = h.runner
    for rid, res in h.results.items():
        if res.finish != FinishReason.ERROR:
            continue
        seed = h.specs[rid].seed
        assert seed in r.fired or seed in r.poison_read or rid in h.cascaded or rid in h.rejected, \
            f"request {rid} failed without a cause"
