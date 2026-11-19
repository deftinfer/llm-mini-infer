"""The engine process: one Python thread runs the step loop and polls ZMQ inside it (no Python I/O threads; ZMQ's one
I/O thread moves the bytes).

**Supervision** (`launcher`; docs/architecture.md, "Processes"). Each loop turn stamps `heartbeat` (shared memory,
`time.monotonic_ns`): a turn is at most one step, or a 1 s idle poll, and the launcher stops an engine whose stamp is
older than `engine.watchdog_s`. `faulthandler` is on, so a SIGABRT (the watchdog), SIGSEGV or SIGBUS prints every
thread's stack. SIGTERM is ignored while the launcher lives: the launcher drains the frontends first and then sends
`Shutdown`; a supervisor that signals the whole process group must not abort what they are still serving. An orphaned
engine (the launcher gone) stops on SIGTERM, or by itself once idle.

**The wake-up** (architecture.md, "Structured output"). A grammar build that ends on a compile thread writes a byte to
a pipe the poller watches beside the input socket, so a blocked loop admits its requests at once, not after its 1 s
poll. Its write end is never closed while a job may run (the threads are daemons): the process's exit closes it.

**Backpressure** (architecture.md, "Frontend and engine messages"). A frontend more than `LAG_PAUSE` StepOutputs past
its last `Ack` gets no new admissions until it is back within `LAG_RESUME`: its waiting requests wait, its running ones
go on, and nothing is dropped.
"""

from __future__ import annotations

import contextlib
import faulthandler
import gc
import logging
import os
import signal
import sys
import time
from multiprocessing.connection import Connection
from typing import Any

import zmq

from ..cache.pools import size_arena, size_pools
from ..config import Config
from ..gcstats import GcStats
from ..logutil import setup_logging, traceback_without_messages
from ..runner.base import UNION_BUCKETS, PoolConfig, Runner
from ..sched.plan import reserve_positions
from ..sched.scheduler import (
    Scheduler,
    derive_mixed_step_row_tokens,
    derive_precise_cost,
    derive_refine_chunk_tokens,
)
from . import ipc
from .core import P_BOUNDARY, P_LAUNCH, P_POLL, P_SAMPLE, P_SEND, EngineCore, StepReport, derive_plan_at_boundary
from .messages import (
    CHECKPOINT_SOURCES,
    LOST_CAUSES,
    Abort,
    Ack,
    AddRequest,
    EngineFailed,
    EngineReady,
    EngineStats,
    Logprobs,
    Shutdown,
    StepOutputs,
)

log = logging.getLogger("llm_mini_infer.engine")
LAG_PAUSE = 64
LAG_RESUME = 16
# The engine's grammar builds run on these (`grammar.threads`, beside the step loop): one slow build leaves the others a
# thread, and a compile or a mask releases the GIL. A build job uses no buffer or executor call the step loop's masks
# share (`LLGuidance.only`, own), so the threads build at once (tests/structured/test_grammar_builds.py)
COMPILE_THREADS = 2


def build_runner(cfg: Config) -> Runner:
    if cfg.runner.kind == "fake":
        from ..runner.fake import make_fake_runner

        im, on = cfg.dev.images, cfg.images.enabled
        return make_fake_runner(cfg.model.path, cfg.dev.runner.fake, cfg.speculative.num_speculative_tokens or 0,
                                encoder_rows=(im.encoder_cache_tokens or 0) if on else 0,
                                encoder_budget=(im.encoder_budget or 0) if on else 0, logprobs=cfg.server.logprobs,
                                penalties=cfg.server.penalties)
    from ..runner.cuda import make_cuda_runner

    return make_cuda_runner(cfg)


class EngineLoop:
    def __init__(self, cfg: Config, core: EngineCore, ipc_path: str, heartbeat: Any = None):
        self.cfg = cfg
        self.core = core
        self.heartbeat = heartbeat  # multiprocessing RawValue("q"): the last turn's time.monotonic_ns (the watchdog)
        self.parent = os.getppid()  # the launcher
        self.ctx = zmq.Context(io_threads=1)
        self.inp = self.ctx.socket(zmq.PULL)
        ipc.configure_socket(self.inp)
        self.inp.bind(ipc.engine_input_addr(ipc_path))
        self.out: list[zmq.Socket] = []
        for i in range(cfg.server.num_frontends or 1):
            sock = self.ctx.socket(zmq.PUSH)
            ipc.configure_socket(sock)
            sock.bind(ipc.frontend_output_addr(ipc_path, i))
            self.out.append(sock)
        self.poller = zmq.Poller()
        self.poller.register(self.inp, zmq.POLLIN)
        # The wake-up (module docstring): both ends non-blocking; only with compile threads
        self._wake_r = self._wake_w = -1
        g = core.sched.grammars
        if g is not None and g.threads is not None:
            self._wake_r, self._wake_w = os.pipe()
            os.set_blocking(self._wake_r, False)
            os.set_blocking(self._wake_w, False)
            self.poller.register(self._wake_r, zmq.POLLIN)
            g.wake = self._wake
            core.drain = self._drain
        self.enc = ipc.Encoder(cfg.dev.engine.inline_threshold)
        self.dec = ipc.Decoder()
        # Abort everything and leave the loop once idle (`_begin_stop`): a Shutdown, SIGTERM once the launcher is gone,
        # or the launcher found gone while blocked. A request added meanwhile is aborted at once (`_poll`)
        self.stopping = False
        self.idle_reported = True
        self.last_cpu_ns = 0
        self.last_calls_ns = 0
        self._sigterm = False
        self.spin_ns = int((cfg.dev.engine.idle_spin_ms or 0.0) * 1e6)  # poll, not block, this long after going idle
        self._spin_until = 0
        self.boundary = bool(cfg.engine.plan_at_boundary)  # plan once the step in flight's forward ended (core.py)
        self._sent: StepReport | None = None  # what the current turn emitted
        n = len(self.out)
        self.sent_seq, self.acked_seq = [0] * n, [0] * n
        self.paused = core.sched.paused_frontends  # frontends whose admissions wait (backpressure)
        self.pauses = 0
        self.waits = 0  # loop turns that blocked because the only work waited for paused frontends
        self.gc = GcStats("engine")  # installed by `run`

    def _wake(self) -> None:
        """A grammar build ended (`Grammars.wake`, on its compile thread): one byte for the poller."""
        with contextlib.suppress(OSError):  # full (BlockingIOError: the loop wakes anyway), or the process is ending
            os.write(self._wake_w, b"\0")

    def _drain(self) -> None:
        """The wake-up pipe emptied (read until it would block)."""
        with contextlib.suppress(BlockingIOError):
            while os.read(self._wake_r, 4096):
                pass

    def _poll_now(self) -> int:
        """A non-blocking poll inside the boundary wait (`EngineCore.await_boundary`): the ns it spent collecting
        grammar builds (only when the collect did work: a build settled or a request left COMPILING; a spin that
        found nothing stays wait) and handling messages (engine CPU, not the wait), 0 when there was neither."""
        ns = 0
        if self.core.collect_due:
            t = time.perf_counter_ns()
            if self.core.collect():
                ns = time.perf_counter_ns() - t
        if not self.inp.get(zmq.EVENTS) & zmq.POLLIN:
            return ns
        t = time.perf_counter_ns()
        self._poll(block=False)
        return ns + time.perf_counter_ns() - t

    def _poll(self, block: bool) -> None:
        inp = self.inp
        if block:
            events = self.poller.poll(1000)
            if not events:
                return
            if self._wake_r >= 0 and any(s == self._wake_r for s, _ in events):
                self._drain()
        while inp.get(zmq.EVENTS) & zmq.POLLIN:
            msg = self.dec.decode(inp.recv_multipart(zmq.NOBLOCK, copy=False))
            if isinstance(msg, (AddRequest, Abort)):
                if isinstance(msg, AddRequest) and self.stopping:
                    self.core.sched.add(msg)
                    self.core.sched.abort_all()
                    continue
                self.core.handle(msg)
            elif isinstance(msg, Ack):
                self.on_ack(msg.frontend, msg.seq)
            elif isinstance(msg, Shutdown):
                log.info("shutdown requested%s", f": {msg.reason}" if msg.reason else "")
                self._begin_stop()  # even when stopping already: a second Shutdown aborts again

    def _stats(self, report: StepReport | None) -> EngineStats:
        s, c = self.core.sched, self.core
        cache = s.cache
        g = s.grammars
        a = c.runner.attention_counters()
        h = c.runner.head_counters()
        gcs = self.gc
        gcs.maybe_warn()
        # Waiting counts the requests whose grammar is being built (COMPILING), as vLLM's counts them ("deferred")
        return EngineStats(running=s.num_running, waiting=s.num_waiting + s.num_compiling, kv_usage=s.kv_usage,
                           state_usage=s.state_usage, num_preemptions=s.num_preemptions, num_aborts=s.num_aborts,
                           aborts=list(s.aborts),
                           prompt_tokens=s.prompt_tokens, generation_tokens=s.generation_tokens,
                           step_tokens=report.step_tokens if report else 0, step_cpu_ns=self.last_cpu_ns,
                           step_runner_calls_ns=self.last_calls_ns,
                           step_runner_ns=report.runner_ns if report else 0, gpu_busy_ns=c.gpu_busy_ns,
                           gpu_idle_ns=c.gpu_idle_ns, gpu_sample_ns=c.gpu_sample_ns, boundary_ns=c.phase_ns[P_BOUNDARY],
                           steps=c.steps, phase_ns=list(c.phase_ns), gc_ns=list(gcs.ns),
                           gc_collections=list(gcs.collections),
                           checkpoint_usage=s.checkpoint_usage, cached_pages=cache.cached_pages if cache else 0,
                           prefix_queries=s.prefix_queries, prefix_hits=s.prefix_hits, readmit_hits=s.readmit_hits,
                           ckpt_hits=s.ckpt_hits, ckpt_misses=s.ckpt_misses,
                           ckpts_taken=[s.ckpts_taken[k] for k in CHECKPOINT_SOURCES], ckpts_skipped=s.ckpts_skipped,
                           cache_waits=s.cache_waits, lost_tokens=[s.lost_tokens[k] for k in LOST_CAUSES],
                           refines=s.refines, precise_tokens=s.precise_tokens,
                           precise_hits=s.precise_hits, precise_steps=s.precise_steps,
                           evicted_pages=cache.evicted_pages if cache else 0,
                           evicted_ckpts=cache.evicted_ckpts if cache else 0, num_failed=s.num_failed,
                           ckpt_resumes=list(s.ckpt_resumes),
                           ckpts_evicted=[n for pair in cache.evicted_by for n in pair] if cache else [],
                           evicted_window_pages=cache.evicted_window_pages if cache else 0,
                           windows=cache.num_ckpts if cache and s.window else 0,
                           spec_drafts=s.spec_drafts, spec_draft_tokens=s.spec_draft_tokens,
                           spec_accepted=s.spec_accepted,
                           spec_accepted_per_pos=[int(x) for x in s.spec_accepted_per_pos] if s.k else [],
                           attention_replans=a.replans, attention_plans_kept=a.kept, attention_plans_unsplit=a.unsplit,
                           vision_plan_waits=a.vision_waits,
                           grammar_head_sparse=h.sparse, grammar_head_dense_off=h.dense_off,
                           grammar_head_dense_unconstrained=h.dense_unconstrained,
                           grammar_head_dense_over_cap=h.dense_over_cap,
                           grammar_head_union_counts=list(h.union_counts) or [0] * (len(UNION_BUCKETS) + 1),
                           grammar_head_union_sum=h.union_sum, draft_universe_tokens=h.universe,
                           draft_universe_steps=h.universe_steps, draft_universe_rebuilds=h.universe_rebuilds,
                           encoder_images=s.encoder_images, encoder_tokens=s.encoder_tokens,
                           encoder_hits=s.encoder_hits, encoder_ns=c.encoder_ns,
                           step_encoder_ns=c.last_encoder_ns if report else 0,
                           step_rows_counts=list(c.rows_hist), step_rows_sum=c.rows_sum,
                           step_images_counts=list(c.images_hist), step_images_sum=c.images_sum,
                           page_waits=s.page_waits,
                           structured_requests=s.structured_requests, grammar_rejects=s.grammar_rejects,
                           backpressure_pauses=self.pauses, paused_frontends=len(self.paused),
                           backpressure_waits=self.waits,
                           grammar_end_waits=s.end_waits, dropped_samples=s.dropped_samples,
                           **({} if g is None else dict(grammar_compiles=g.compiles, grammar_compile_ns=g.compile_ns,
                                                        grammar_masks=g.mask_rows, grammar_mask_ns=g.mask_ns,
                                                        grammar_errors=g.errors,
                                                        grammar_forced_requests=s.forced_requests,
                                                        grammar_forced_tokens=s.forced_tokens,
                                                        grammar_eos_free=[int(x) for x in s.eos_free],
                                                        grammar_eos_free_gated=s.eos_free_gated,
                                                        spec_first_rejections=[int(x) for x in s.first_rejections],
                                                        spec_window_mask_counts=[int(x) for x in s.window_masks],
                                                        spec_window_mask_sum=s.window_masks_sum)))

    def _emit(self, report: StepReport) -> None:
        """The outputs of a step, sent from inside the turn (before the next step's masks; `EngineCore.step`)."""
        t = time.perf_counter_ns()
        self._send_report(report)
        self.core.phase_ns[P_SEND] += time.perf_counter_ns() - t
        self._sent = report

    def _send(self, fe: int, msg: StepOutputs) -> None:
        self.sent_seq[fe] += 1
        msg.seq = self.sent_seq[fe]
        try:
            self.out[fe].send_multipart(self.enc.encode(msg), flags=zmq.NOBLOCK, copy=False)
        except zmq.Again:
            log.warning("frontend %d is not connected; dropping its outputs", fe)
        if self.sent_seq[fe] - self.acked_seq[fe] > LAG_PAUSE and fe not in self.paused:
            self.paused.add(fe)
            self.pauses += 1
            log.warning("frontend %d is %d outputs behind: pausing its admissions", fe,
                        self.sent_seq[fe] - self.acked_seq[fe])

    def on_ack(self, fe: int, seq: int) -> None:
        if not 0 <= fe < len(self.out):
            return
        self.acked_seq[fe] = max(self.acked_seq[fe], seq)
        if fe in self.paused and self.sent_seq[fe] - self.acked_seq[fe] <= LAG_RESUME:
            self.paused.discard(fe)
            log.info("frontend %d caught up: resuming its admissions", fe)

    def _send_report(self, report: StepReport) -> None:
        stats = self._stats(report)
        now = time.perf_counter_ns()
        for fe, fo in report.outputs.items():
            self._send(fe, StepOutputs(step=report.step_id, rids=fo.rids, n_new=fo.n_new, tokens=fo.tokens,
                                       finish=fo.finish, first_rids=fo.first_rids, first_cached=fo.first_cached,
                                       first_t_admit_ns=fo.first_t_admit_ns, first_t_launch_ns=fo.first_t_launch_ns,
                                       first_runner_ns=fo.first_runner_ns, t_sent_ns=now, stats=stats,
                                       released=fo.released, spec_rids=fo.spec_rids, spec_rows=fo.spec_rows,
                                       spec_tokens=fo.spec_tokens, first_t_queued_ns=fo.first_t_queued_ns,
                                       logprobs=Logprobs(token=fo.logprobs[0], top_ids=fo.logprobs[1],
                                                         top=fo.logprobs[2]) if fo.logprobs is not None else None))

    def _send_idle_stats(self) -> None:
        import numpy as np

        e64, e32, eu8 = np.zeros(0, np.int64), np.zeros(0, np.int32), np.zeros(0, np.uint8)
        stats = self._stats(None)
        for fe in range(len(self.out)):
            self._send(fe, StepOutputs(step=0, rids=e64, n_new=e32, tokens=e32, finish=eu8, first_rids=e64,
                                       first_cached=e32, first_t_admit_ns=e64, first_t_launch_ns=e64,
                                       first_runner_ns=e64, t_sent_ns=time.perf_counter_ns(), stats=stats))

    def run(self) -> None:
        core, ph = self.core, self.core.phase_ns
        clock = time.perf_counter_ns
        hb = self.heartbeat
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, "_sigterm", True))
        self.gc.install()
        while True:
            t0 = clock()
            if hb is not None:
                hb.value = time.monotonic_ns()
            calls0 = ph[P_LAUNCH] + ph[P_SAMPLE]
            blocked = boundary = 0
            if self._sigterm:
                self._sigterm = False
                if os.getppid() == self.parent:
                    log.info("SIGTERM ignored: the launcher stops the engine once the frontends have drained")
                elif not self.stopping:
                    log.info("SIGTERM (the launcher is gone): aborting in-flight requests")
                    self._begin_stop()
            if core.sched.compiling_moved:  # a request entered or left COMPILING: the waiting count changed
                core.sched.compiling_moved = False
                self.idle_reported = False
            busy = core.busy
            if busy and core.paused_only:  # only an Ack can unblock the work: wait for it, as when idle
                self.waits += 1
                self._block()
                blocked = clock() - t0
            elif busy and core.compiling_only:  # only a build's end (the wake-up) or a message brings work
                if not self.idle_reported:  # the frontends see the waiting count while nothing steps
                    self._send_idle_stats()
                    self.idle_reported = True
                self._block()
                blocked = clock() - t0
            elif busy:
                if self.boundary:
                    boundary = core.await_boundary(self._poll_now)
                self._poll(block=False)
                self._spin_until = t0 + self.spin_ns
            else:
                if self.stopping:
                    break
                if not self.idle_reported:
                    self._send_idle_stats()
                    self.idle_reported = True
                if t0 < self._spin_until:  # a warm core for the next request
                    self._poll(block=False)
                    if not core.busy:
                        continue
                else:
                    self._block()
                    blocked = clock() - t0
            t1 = clock()
            ph[P_POLL] += t1 - t0 - blocked - boundary
            self._sent = None
            core.step(emit=self._emit)
            report = self._sent
            wait_ns = 0
            if report is not None:
                wait_ns = report.wait_ns
                self.idle_reported = False
            if report is not None or blocked == 0:
                # Engine CPU of this turn: wall time minus the poll's blocking, the waits for the device (the step's
                # results; at the boundary, its forward's end) and the runner's launch calls.
                calls = ph[P_LAUNCH] + ph[P_SAMPLE] - calls0
                self.last_cpu_ns = clock() - t0 - blocked - boundary - wait_ns - calls
                self.last_calls_ns = calls
        self.gc.uninstall()
        log.info("engine loop stopped after %d steps", core.steps)

    def _block(self) -> None:
        """A blocking poll (a message, or a second: the heartbeat stays fresh), then stop if the launcher is gone."""
        self._poll(block=True)
        if not self.stopping and os.getppid() != self.parent:
            log.warning("the launcher is gone: stopping")
            self._begin_stop()

    def _begin_stop(self) -> None:
        """Stop (`stopping`): every request is aborted now, and the loop ends at its first idle turn. The callers log
        why."""
        self.stopping = True
        self.core.sched.abort_all()

    def close(self) -> None:
        for s in (self.inp, *self.out):
            s.close(0)
        if self._wake_r >= 0:  # the read end only (module docstring, The wake-up): a late write fails quietly
            g = self.core.sched.grammars
            g.wake = None
            os.close(self._wake_r)
            self._wake_r = -1
        self.ctx.term()


def _pin(cpu: int | None) -> None:
    if cpu is None:
        log.info("engine thread not pinned (no CPU affinity on %s)", sys.platform)
        return
    try:
        os.sched_setaffinity(0, {cpu})  # this thread only; ZMQ's I/O thread was created before and keeps its mask
        log.info("engine thread pinned to CPU %d", cpu)
    except (AttributeError, OSError) as exc:
        log.warning("could not pin the engine thread to CPU %d: %s", cpu, exc)


def _runner_info(runner: Runner, caps: Any, startup_s: float) -> dict[str, Any]:
    """The capabilities and the runner's startup report, as plain JSON values (/server_info)."""
    import json

    import msgspec

    stats = getattr(runner, "stats", {}) or {}
    plain = json.loads(json.dumps(stats, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    return {"capabilities": msgspec.to_builtins(caps), "stats": plain, "engine_startup_s": round(startup_s, 2)}


LOGGED_DECODE_ROWS = 32  # the images line's example of a busy step (as the mixed-step bound's reason quotes)


def _images_per_step(*, tokens: int, extra: int, encoder_budget: int, max_num_batched_tokens: int, row_tokens: int,
                     rows: int, width: int, grid: int) -> int:
    """How many images of `tokens` tokens one step prefills whole next to `rows` decode rows of `width` tokens, by
    `Scheduler._schedule`'s budgets: the encoder's, and the prefill tokens the decode rows leave, cut to `row_tokens` /
    rows (at least a checkpoint `grid`) by the mixed-step bound when it is on. Each image comes with the `extra` tokens
    around its span, except a last one: a chunk that starts at a span while the step has prefill tokens left takes
    the whole span, past the bound if need be, within max_num_batched_tokens (`Scheduler._span_end`). A bidirectional
    span (Gemma 4) is prefilled whole in the step that encodes it, so the step budget can bind before the encoder's.
    tests/sched/test_image_sched.py checks this arithmetic for Gemma 4, and that the scheduler reaches it."""
    hard = max_num_batched_tokens - rows * width
    soft = min(hard, max(grid, row_tokens // rows)) if rows and row_tokens else hard
    n = max(0, soft) // (tokens + extra)
    used = n * (tokens + extra)
    if used < soft and used + tokens <= hard:  # the budget has tokens left: one more chunk, starting at a span
        n += 1
    return min(encoder_budget // tokens, n)


def _images_line(cfg: Config, caps: Any, fam: Any, grid: int) -> str:
    """The startup line on images: one image's budget, the encoder's per step and, for bidirectional spans, the step's
    (`_images_per_step`)."""
    t = fam.max_tokens()
    if fam.max_soft_tokens is not None:
        sizing = (f"at most {t} soft tokens an image (images.max_soft_tokens; {fam.max_patches:,} patches, "
                  f"{fam.max_pixels:,} pixels)")
    else:
        sizing = (f"at most {t} tokens an image (images.min_pixels {fam.min_pixels:,} to images.max_pixels "
                  f"{fam.max_pixels:,})")
    n = caps.encoder_budget // t
    line = (f"{sizing}; the encoder takes {caps.encoder_budget:,} image tokens a step ({n} such image"
            f"{'' if n == 1 else 's'}), its cache {caps.encoder_rows:,} rows")
    if not caps.bidirectional_images:
        return line
    sc, row_tokens = cfg.scheduler, cfg.dev.scheduler.mixed_step_row_tokens
    rows = min(LOGGED_DECODE_ROWS, sc.max_num_seqs)
    extra = len(fam.span_prefix) + len(fam.span_suffix)
    kw = dict(tokens=t, extra=extra, encoder_budget=caps.encoder_budget,
              max_num_batched_tokens=sc.max_num_batched_tokens, row_tokens=row_tokens or 0,
              width=1 + caps.mtp_depth, grid=grid)
    bound = (f"dev.scheduler.mixed_step_row_tokens {row_tokens:,}; the last span may run past it"
             if row_tokens else "dev.scheduler.mixed_step_row_tokens off")
    alone = _images_per_step(rows=0, **kw)
    return (f"{line}; a step's prefill holds at most {alone} such image{'' if alone == 1 else 's'} whole "
            f"({t + extra:,} tokens an image with its delimiters), {_images_per_step(rows=rows, **kw)} next to {rows} "
            f"decode rows ({bound})")


def _grammars(cfg: Config, caps: Any) -> Any:
    """The engine's grammars (structured output), or None when it is off. Its llguidance tokenizer covers the logits
    width; a completed grammar allows the backend's EOS ids (grammar.engine): the tokenizer's eos_token with guidance,
    the model's EOS ids with auto and xgrammar (`grammar_eos_token_ids`). They must stop a request: refused if not."""
    so = cfg.structured_output
    if not so.enabled:
        return None
    if not caps.grammar_masks:
        so.enabled = False
        cfg.reasons["structured_output.enabled"] = f"off: the {caps.backend} runner takes no grammar masks"
        log.warning("structured output is off: the %s runner takes no grammar masks", caps.backend)
        return None
    from ..config import tokenizer_eos_token, tokenizer_json
    from ..grammar.engine import Grammars, LLGuidance
    from ..grammar.threads import CompileThreads

    guidance = so.backend == "guidance"
    eos = tokenizer_eos_token(cfg) if guidance else cfg.eos_token_ids
    backend = LLGuidance(tokenizer_json(cfg), caps.vocab_size, eos, cfg.dev.structured_output.threads or 1)
    missing = sorted(set(backend.eos) - set(cfg.eos_token_ids))
    if missing:  # the stop set is the checkpoint's EOS ids; a grammar's end token must be one of them
        raise ValueError(f"structured_output.backend {so.backend} ends a grammar on the tokenizer's eos_token {eos!r} "
                         f"(id {missing[0]}), which does not stop a request: eos_token_ids {cfg.eos_token_ids} "
                         f"({cfg.reasons.get('eos_token_ids', 'set')}): every structured request would end in a "
                         "grammar error")
    cfg.grammar_eos_token_ids = list(backend.eos)
    cfg.reasons["grammar_eos_token_ids"] = (f"the tokenizer's eos_token {eos} (guidance)" if guidance
                                            else "eos_token_ids (auto and xgrammar end on the model's stop ids)")
    log.info("structured output: backend %s%s, llguidance %s over %d tokens, a completed grammar ends on %s; %d "
             "thread(s); tokenizer built in %.2f s", so.backend, ", compact JSON" if so.disable_any_whitespace else "",
             backend.version, caps.vocab_size, cfg.grammar_eos_token_ids, backend.threads, backend.build_s)
    return Grammars(backend, threads=CompileThreads(COMPILE_THREADS, "grammar-compile"))


def engine_main(cfg: Config, ready: Connection, ipc_path: str, heartbeat: Any = None) -> None:
    """Process entry point (spawned by the launcher)."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl-C goes to the launcher, which drains and then stops us
    setup_logging(cfg.observability.log_level, "engine")
    faulthandler.enable()
    try:
        t0 = time.perf_counter()
        runner = build_runner(cfg)
        caps = runner.capabilities()
        sc, cc, dsc = cfg.scheduler, cfg.dev.cache, cfg.dev.scheduler
        len_derived = "model.max_model_len" in cfg.reasons  # not set by the user: the pool check may cap it
        max_model_len = min(cfg.model.max_model_len, caps.max_positions or cfg.model.max_model_len)
        if max_model_len < cfg.model.max_model_len:
            cfg.reasons["model.max_model_len"] = f"capped by the runner's max_positions {caps.max_positions}"
        cfg.model.max_model_len = max_model_len
        if cc.page_size % caps.page_size_multiple:
            raise ValueError(f"page_size {cc.page_size} must be a multiple of {caps.page_size_multiple}")
        if cfg.cache.enable_prefix_caching and caps.state_slot_bytes and caps.checkpoints == "none":
            log.warning("the %s runner takes no checkpoints: prefix caching is off", caps.backend)
            cfg.cache.enable_prefix_caching = False
            cfg.reasons["cache.enable_prefix_caching"] = f"the {caps.backend} runner takes no checkpoints"
        grid = caps.checkpoint_grid
        page_size, window_page_size = cc.page_size, 0
        if caps.window:  # one arena for full and window pages; window checkpoints sit on any window page
            grid = window_page_size = cc.page_size
            sizes = size_arena(memory_bytes=caps.memory_free_bytes, kv_bytes_per_token=caps.kv_bytes_per_token,
                               window_kv_bytes_per_token=caps.window_kv_bytes_per_token, window=caps.window,
                               window_page_size=window_page_size, max_num_seqs=sc.max_num_seqs,
                               max_model_len=max_model_len, max_num_batched_tokens=sc.max_num_batched_tokens,
                               reserve_tokens=reserve_positions(caps.max_steps_in_flight, caps.mtp_depth),
                               num_pages=cc.num_pages, prefix_caching=cfg.cache.enable_prefix_caching,
                               cap_max_model_len=len_derived)
            cc.num_pages, page_size = sizes.num_pages, sizes.page_size
            cc.num_state_slots, cc.checkpoint_slots = sc.max_num_seqs + 1, 0
            num_live = sc.max_num_seqs
            reasons = dict(sizes.reasons)
            log.info("KV arena: dev.cache.page_size %d is the window page; full pages hold %d positions (%s)",
                     window_page_size, page_size, reasons.pop("page_size"))
            cfg.reasons["dev.cache.page_size"] = (f"the window page; full pages of {page_size} positions fill as "
                                                  "many bytes (one KV arena)")
            cfg.reasons["dev.cache.checkpoint_slots"] = ("0: a sliding-window model's checkpoints are windows (KV "
                                                         "arena)")
        else:
            if cfg.cache.enable_prefix_caching and caps.checkpoint_grid % cc.page_size:
                raise ValueError(f"page_size {cc.page_size} must divide the checkpoint grid {caps.checkpoint_grid}")
            sizes = size_pools(memory_bytes=caps.memory_free_bytes, kv_bytes_per_token=caps.kv_bytes_per_token,
                               state_slot_bytes=caps.state_slot_bytes, page_size=cc.page_size,
                               max_num_seqs=sc.max_num_seqs, max_model_len=max_model_len, num_pages=cc.num_pages,
                               num_state_slots=cc.num_state_slots, checkpoint_slots=cc.checkpoint_slots,
                               prefix_caching=cfg.cache.enable_prefix_caching,
                               reserve_tokens=reserve_positions(caps.max_steps_in_flight, caps.mtp_depth),
                               cap_max_model_len=len_derived)
            cc.num_pages, cc.num_state_slots = sizes.num_pages, sizes.num_state_slots
            cc.checkpoint_slots = sizes.num_checkpoint_slots
            num_live = sizes.num_live_slots
            reasons = dict(sizes.reasons)
        if "max_model_len" in reasons:
            cfg.reasons["model.max_model_len"] = reasons.pop("max_model_len")
            max_model_len = cfg.model.max_model_len = sizes.max_model_len
        for k, v in reasons.items():
            cfg.reasons[f"dev.cache.{k}"] = v
        runner.configure(PoolConfig(page_size=page_size, num_pages=cc.num_pages,
                                    num_state_slots=cc.num_state_slots, max_num_seqs=sc.max_num_seqs,
                                    max_num_batched_tokens=sc.max_num_batched_tokens, max_model_len=max_model_len,
                                    eos_token_ids=tuple(cfg.eos_token_ids), mtp_k=caps.mtp_depth,
                                    window_page_size=window_page_size))
        if dsc.mixed_step_row_tokens is None:
            dsc.mixed_step_row_tokens, cfg.reasons["dev.scheduler.mixed_step_row_tokens"] = \
                derive_mixed_step_row_tokens(decode_ms=caps.decode_step_ms, decode_rows=caps.decode_step_rows,
                                             prefill_ms=caps.prefill_step_ms, prefill_tokens=caps.prefill_step_tokens,
                                             max_num_batched_tokens=sc.max_num_batched_tokens,
                                             max_num_seqs=sc.max_num_seqs)
        log.info("dev.scheduler.mixed_step_row_tokens = %d (%s)", dsc.mixed_step_row_tokens,
                 cfg.reasons.get("dev.scheduler.mixed_step_row_tokens", "set"))
        ec = cfg.engine
        if ec.plan_at_boundary is None:
            ec.plan_at_boundary = False
            cfg.reasons["engine.plan_at_boundary"] = (
                "off by default: it brings an arrival's first token a step earlier, but leaves the host only the "
                "step's sampling part to launch the next step instead of its forward, so a host stall idles the "
                "device; set it for streaming clients")
        elif ec.plan_at_boundary and not caps.forward_end:
            raise ValueError(f"engine.plan_at_boundary: the {caps.backend} runner reports no forward's end (Gemma 4 "
                             "on CUDA with structured output does, and the fake runner with dev.runner.fake.sample_ms)")
        elif ec.plan_at_boundary:  # set: warn when the runner's lone decode step leaves the next launch too little
            on, why = derive_plan_at_boundary(forward_end=True, sample_ms=caps.sample_step_ms,
                                              launch_ms=caps.launch_step_ms)
            if not on:
                log.warning("engine.plan_at_boundary is set, but %s", why.removeprefix("off: "))
        log.info("engine.plan_at_boundary = %s (%s)", ec.plan_at_boundary,
                 cfg.reasons.get("engine.plan_at_boundary", "set"))
        positions = None
        if cfg.images.enabled:
            if not caps.images:
                raise ValueError(f"images.enabled, but the {caps.backend} runner has no image encoder")
            from ..vision import family as vf

            im = cfg.images
            fam = vf.load(cfg.model.path, min_pixels=im.min_pixels, max_pixels=im.max_pixels,
                          max_soft_tokens=im.max_soft_tokens)
            positions = fam.positions
            log.info("images: %s", _images_line(cfg, caps, fam, grid))
        grammars = _grammars(cfg, caps)
        # The precise prefix (architecture.md, "The precise prefix"): the policy for a runner that computes PRECISE rows
        # (a block-FP8 model), and their cost in the mixed-step bound (the runner's measured ratio of a PRECISE prefill
        # step to an FP8 one)
        precise = cfg.runner.cuda.fp8_cold_prefill if caps.precise_rows else "fp8"
        precise_cost = derive_precise_cost(caps.precise_step_ms, caps.prefill_step_ms)
        # A refine's piece, derived unless set (`derive_refine_chunk_tokens`, at `config.mtp_k_max`)
        piece = cc.refine_chunk_tokens or 0
        k_max = max(cfg.mtp_k_max, caps.mtp_depth)
        if precise == "reuse" and not piece:
            piece, cfg.reasons["dev.cache.refine_chunk_tokens"] = derive_refine_chunk_tokens(
                max_num_batched_tokens=sc.max_num_batched_tokens, max_num_seqs=sc.max_num_seqs, mtp_k_max=k_max,
                grid=grid, drafter="the assistant" if cfg.speculative.model else None)
            cc.refine_chunk_tokens = piece
        if caps.precise_rows:
            log.info("the precise prefix: %s (a PRECISE token costs %.2f FP8 prefill tokens in the mixed-step bound)",
                     {"reuse": f"a cached prefix is recomputed with BF16 activations once {cc.refine_after + 1} "
                               f"requests reused it, if at least {cc.refine_min} tokens (dev.cache.refine_after, "
                               f"refine_min); its PRECISE rows cut every {piece} tokens from position 0, each piece "
                               "whole in a step (dev.cache.refine_chunk_tokens)",
                      "bf16": "every prefill row of a prompt without a cache hit has BF16 activations",
                      "fp8": "off"}[precise], precise_cost)
        sched = Scheduler(max_num_seqs=sc.max_num_seqs, max_num_batched_tokens=sc.max_num_batched_tokens,
                          max_model_len=max_model_len, page_size=page_size, num_pages=cc.num_pages,
                          num_state_slots=cc.num_state_slots, eos_token_ids=cfg.eos_token_ids, policy=sc.policy,
                          long_prefill_token_threshold=dsc.long_prefill_token_threshold,
                          steps_in_flight=caps.max_steps_in_flight, mtp_k=caps.mtp_depth,
                          num_frontends=cfg.server.num_frontends or 1, seed=sc.seed,
                          num_live_slots=num_live, hybrid=caps.state_slot_bytes > 0,
                          prefix_caching=cfg.cache.enable_prefix_caching, checkpoint_grid=grid,
                          decode_checkpoint_interval=cc.decode_checkpoint_interval,
                          demote_superseded=cc.demote_superseded,
                          mixed_step_row_tokens=dsc.mixed_step_row_tokens,
                          encoder_rows=caps.encoder_rows if cfg.images.enabled else 0,
                          encoder_budget=caps.encoder_budget, positions=positions, grammars=grammars,
                          window=caps.window, window_page_size=window_page_size,
                          bidirectional_images=bool(caps.bidirectional_images and cfg.images.enabled),
                          draft_kv=caps.draft_kv, precise=precise, precise_cost=precise_cost,
                          refine_after=cc.refine_after, refine_min=cc.refine_min,
                          refine_chunk_tokens=piece, mtp_k_max=k_max,
                          forced_prefix=bool(cfg.dev.structured_output.forced_prefix),
                          eos_free_finish=bool(cfg.dev.structured_output.eos_free_finish),
                          batch_invariant=caps.batch_invariant, logprobs=caps.logprobs,
                          penalties_vocab=caps.vocab_size if caps.penalties else 0,
                          compile_timeout_s=cfg.dev.structured_output.compile_timeout_s)
        if grammars is not None:  # the scheduler refuses the EOS-free finish where it cannot keep its outputs
            so = cfg.dev.structured_output
            log.info("structured output: the forced prefix %s, the EOS-free finish %s (dev.structured_output."
                     "forced_prefix, eos_free_finish)",
                     "on" if sched.forced_prefix else "off",
                     "on" if sched.eos_free_finish and sched._grammar_eos >= 0 else
                     "off" + (" (the grammar ends on several EOS ids)" if so.eos_free_finish else ""))
        loop = EngineLoop(cfg, EngineCore(sched, runner), ipc_path, heartbeat)
        _pin(cfg.dev.engine.cpu)
        # the capability this process built the runner's CPU tables with (the fake runner imports no torch)
        if cfg.runner.kind != "fake":
            import torch

            log.info("ATen CPU capability %s", torch.backends.cpu.get_cpu_capability())
        if cfg.dev.engine.gc_freeze:
            gc.collect()
            gc.freeze()
            gc.set_threshold(10_000, 50, 100)
        cfg.runner_info = _runner_info(runner, caps, time.perf_counter() - t0)
        ready.send(EngineReady(cfg, caps, time.perf_counter() - t0))
        ready.close()
    except BaseException as exc:
        log.exception("engine startup failed")
        try:
            ready.send(EngineFailed(f"{type(exc).__name__}: {exc}"))
        finally:
            sys.exit(1)
    try:
        loop.run()
    except Exception as exc:  # its message, and its causes', can quote a request: the types and frames only
        log.error("engine loop failed (exception messages left out)\n%s", traceback_without_messages(exc).rstrip())
        raise SystemExit(1) from None  # unchained: the traceback of a cleanup error below (multiprocessing prints it)
    finally:
        runner.shutdown()
        loop.close()
