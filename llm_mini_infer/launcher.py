"""Starts the engine process and N frontend processes (`spawn`), waits until they are ready, supervises them, drains
on SIGTERM and fails fast when a process dies.

Order (architecture.md, "Processes"): a model given as a Hugging Face repo id is downloaded first (`hub`), and every
process gets its local snapshot; with images on, the pixel arena and the image pool's workers come next; the engine
starts next, then reports its effective configuration; the frontends start with it, so nothing listens on the port,
and `/health` cannot answer, before warm-up is done.

**Stopping** (docs/operations.md, "Health, drain and restarts"). The launcher is the only process a supervisor
signals (docker sends SIGTERM to PID 1 only); it passes the stop on to the frontends, one signal per mode:
- **SIGTERM** (`docker compose stop`) is forwarded as SIGTERM, **SIGINT** (Ctrl-C) as SIGINT: the frontends drain
  (`api.app.App.begin_drain`; SIGINT without the grace). Once they exited, the engine is told to stop, then the pool,
  and the launcher exits 0, or 1 if a frontend failed or the engine died meanwhile.
- **A second signal**, **the engine or a frontend exiting unexpectedly**, **an engine that stops making progress**
  (`engine.watchdog_s`; it is sent SIGABRT first, which dumps its threads' stacks, then SIGKILL), or **an image pool
  in a crash loop** (more than `POOL_DEATHS` worker deaths within `POOL_DEATH_WINDOW_S`, or one after a second
  signal): the frontends get SIGUSR1 and fail their in-flight requests at once (a non-streamed request a 503, a
  streamed one an error event), and the launcher exits 1, so the container's restart policy brings the server back.
  The watchdog (`Watchdog`), armed once the engine is ready, is judged once per supervision turn, and as the pool's
  workers and the frontends start (`_check_started`), with the engine's death and, once its workers are ready, the
  pool's crash loop. Image pool workers that die or hang are respawned (`ImagePool`; what fails: `vision.pool`); on
  any exit the pool is closed first.
- **A stop signal while the server starts** (the engine's load and warm-up, up to 900 s, the image pool's workers, the
  frontends) ends the start (`StopWhileStarting`): the launcher logs the signal, dumps the stacks of the process it was
  waiting for and of the engine, stops the rest and exits with 128 + the signal's number (143 for SIGTERM). During a
  model download, before these handlers, a stop signal ends the process at once with the same code (`exit_on_stop`).
- **The parent's exit**, with `server.stop_with_parent` on (off by default): a thread polls the launcher's parent once
  a second from the start of `serve` and, once it changed, sends the main thread SIGTERM unless a stop is under way, so
  the download, the start or the drain ends as above (`_watch_parent`). A parent of pid 1, or none, is not watched.
The engine and the pool workers ignore SIGTERM while the launcher lives (a supervisor that signals the whole process
group must not abort requests the frontends are still draining); the launcher stops them itself.

**Exit codes:**

| code    | when                                                                                      |
|---------|-------------------------------------------------------------------------------------------|
| 0       | the frontends drained after a stop signal, none failing, the engine alive                 |
| 1       | a process died or hung, a crash loop, a failed drain, start or download, a second signal  |
| 2       | a setting the model or the host refuses (`config.derive`)                                 |
| 128 + n | stop signal n while the server starts (`StopWhileStarting`) or downloads (`exit_on_stop`) |
"""

from __future__ import annotations

import contextlib
import logging
import multiprocessing as mp
import os
import shutil
import signal
import socket
import threading
import time
from collections import deque
from collections.abc import Callable
from enum import IntEnum
from multiprocessing.connection import Connection, wait
from typing import Any

import zmq

from . import hub
from .config import Config, derive, effective_lines
from .engine import ipc
from .engine.messages import EngineFailed, Shutdown
from .lifecycle import Lifecycle
from .logutil import setup_logging

log = logging.getLogger("llm_mini_infer.launcher")
# The image pool's crash-loop breaker: it respawns this many worker deaths within the window, and one more takes the
# exit path; a respawned worker not ready in time is killed, which counts as a death, and so does one killed at the
# job deadline, after STACK_WAIT_S for its stack to reach the log
POOL_DEATHS = 3
POOL_DEATH_WINDOW_S = 60.0
POOL_READY_TIMEOUT_S = 120.0
STACK_WAIT_S = 0.25
TURN_S = 1.0  # a supervision turn's wait for its processes and signals
OVERSLEEP_S = 5.0  # a turn this much later than its wait (or half the watchdog's period, if shorter) overslept
PARENT_POLL_S = 1.0  # server.stop_with_parent: how often the launcher's parent is checked (`_watch_parent`)
PARENT_SETTLE_S = 0.5  # and the wait from its exit to the check for a stop under way


class StopWhileStarting(Exception):
    """A stop signal while the server starts (module docstring): `signum`, during the wait for `proc` (`what`)."""

    def __init__(self, signum: int, proc: Any, what: str):
        super().__init__(f"{signal.Signals(signum).name} while starting (waiting for {what})")
        self.signum, self.proc, self.what = signum, proc, what


def _recv(conn: Connection, proc: Any, timeout: float, what: str, signals: list[int],
          check: Callable[[], None] | None = None) -> Any:
    """`proc`'s ready message: a RuntimeError if it exits or `timeout` passes first, StopWhileStarting on a stop signal
    (the launcher's `signals`, checked every 0.2 s). `check`, called as often, watches the processes already started
    (`_check_started`): it raises to end the start."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if signals:
            raise StopWhileStarting(signals[0], proc, what)
        if check is not None:
            check()
        if conn.poll(0.2):
            try:
                return conn.recv()
            except EOFError:  # it closed its end without a word: it is exiting
                proc.join(5)
                raise RuntimeError(f"{what} exited during startup (code {proc.exitcode})") from None
        if not proc.is_alive():
            raise RuntimeError(f"{what} exited during startup (code {proc.exitcode})")
    raise RuntimeError(f"{what} not ready after {timeout:.0f} s")


def exit_on_stop() -> None:
    """Until `serve` installs its drain: SIGTERM or SIGINT ends the process at once with 128 + the signal's number. A
    model download runs before then, and as PID 1 (a pod without an init) a signal without a handler is ignored, so the
    stop would wait for SIGKILL. `os._exit`, not `SystemExit`: huggingface_hub's executor waits for its threads on the
    way out, each fetching a file of gigabytes. No child process exists yet, and the next start resumes the download."""

    def stop(signum: int, _frame: Any) -> None:
        os.write(2, f"{signal.Signals(signum).name} before the server started: exiting\n".encode())
        os._exit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)


def _watch_parent(cfg: Config, parent: int, signals: list[int]) -> threading.Thread | None:
    """With `server.stop_with_parent` on, a daemon thread that polls the launcher's parent (`parent`, its pid at the
    start of `serve`) every PARENT_POLL_S and, once it changed, sends SIGTERM to the main thread (not to the process:
    the handlers run there, and a process-directed signal may reach this thread instead, leaving the main thread's wait
    uninterrupted), once, and ends. It sends nothing when `serve`'s `signals`, which only its handler writes, holds a
    stop already: a second signal would fail the in-flight requests. It looks PARENT_SETTLE_S after it saw the change,
    so that a SIGTERM the parent sent just before it exited has had its handler run. A parent of pid 1 (an init, or
    what an orphan's parent becomes; a subreaper is watched as any parent) or 0 (none visible: the launcher is pid 1,
    or its parent is in another pid namespace) is not watched."""
    if not cfg.server.stop_with_parent:
        return None
    if parent == 1:
        log.info("server.stop_with_parent: the launcher's parent is pid 1, which outlives it: not watched")
        return None
    if parent <= 0:
        log.info("server.stop_with_parent: the launcher has no parent it can see (pid %d): not watched", parent)
        return None
    log.info("server.stop_with_parent: watching the launcher's parent, pid %d", parent)
    main = threading.main_thread().ident

    def watch() -> None:
        while os.getppid() == parent:
            time.sleep(PARENT_POLL_S)
        time.sleep(PARENT_SETTLE_S)
        if signals:
            log.info("the launcher's parent (pid %d) exited; the server is stopping already", parent)
            return
        log.warning("the launcher's parent (pid %d) exited (server.stop_with_parent): stopping as on SIGTERM", parent)
        signal.pthread_kill(main, signal.SIGTERM)

    t = threading.Thread(target=watch, name="parent-watch", daemon=True)
    t.start()
    return t


def serve(cfg: Config) -> int:
    parent = os.getppid()  # first: a parent that exits while the server starts is still the one watched
    setup_logging(cfg.observability.log_level, "launcher")
    t0 = time.perf_counter()
    exit_on_stop()
    signals: list[int] = []  # every stop signal received, in order (`on_signal`; `_watch_parent` reads it)
    _watch_parent(cfg, parent, signals)
    if cfg.runner.kind == "fake":
        log.warning("runner.kind is fake: this server answers with random tokens (tests and benchmarks only; "
                    "set runner.kind: cuda to serve the model)")
    try:
        hub.resolve(cfg)
    except hub.HubError as exc:
        log.error("%s", exc)
        return 1
    try:
        derive(cfg)
    except ValueError as exc:  # a setting the model or the host refuses: exit 2, as the image's checks do
        log.error("configuration: %s", exc)
        return 2
    cfg.pixel_arena = cfg.pool_dir = ""  # the launcher's own (load() refuses both): it removes only an arena it made
    ipc_path = ipc.ipc_dir()
    ctx = mp.get_context("spawn")

    def on_signal(signum: int, _frame: Any) -> None:
        signals.append(signum)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    wake_r, wake_w = socket.socketpair()  # wakes the supervision wait() at once on a signal
    wake_w.setblocking(False)
    signal.set_wakeup_fd(wake_w.fileno())

    from .api.server import frontend_main
    from .engine.process import engine_main

    procs: list[Any] = []  # the engine and the frontends (the pool's workers are respawned instead: ImagePool)
    frontends: list[Any] = []
    pool: ImagePool | None = None
    engine = None
    heartbeat = ctx.RawValue("q", 0)  # the engine loop's last turn, time.monotonic_ns (engine.watchdog_s)
    code = 1
    try:
        pool = _start_images(cfg, ctx, ipc_path)
        pool_ready = pool.start() if pool is not None else []
        recv, send = ctx.Pipe(duplex=False)
        engine = ctx.Process(target=engine_main, args=(cfg, send, ipc_path, heartbeat), name="llm-mini-infer-engine")
        engine.start()
        send.close()
        procs.append(engine)
        info = _recv(recv, engine, 900, "the engine", signals)
        if isinstance(info, EngineFailed):
            raise RuntimeError(f"engine startup failed: {info.error}")
        cfg = info.cfg
        caps = info.capabilities
        log.info("engine ready in %.1f s (%s runner, %s; pid %d)", info.startup_s, caps.backend, caps.device,
                 engine.pid)
        log.info("effective configuration:\n  %s", "\n  ".join(effective_lines(cfg)))
        watchdog = Watchdog(heartbeat, cfg.engine.watchdog_s or 0.0)  # armed: the engine just reported ready
        for i, r in enumerate(pool_ready):  # a worker failing to start is fatal
            _recv(r, pool.procs[i], 300, f"image pool worker {i}", signals,
                  check=lambda: _check_started(engine, watchdog))
            r.close()
            pool.mark_ready(i)
        if pool is not None:
            log.info("image pool: %d workers ready (pids %s; arena %s, %d MiB; job deadline %.0f s)", len(pool.procs),
                     ", ".join(str(p.pid) for p in pool.procs), cfg.pixel_arena, cfg.dev.images.arena_mb,
                     pool.deadline_s)
        n = cfg.server.num_frontends or 1
        os.environ["LLM_MINI_INFER_ENGINE_PID"] = str(engine.pid)  # for /debug/engine (benchmarks' CPU accounting)
        readies = []
        for i in range(n):
            r, s = ctx.Pipe(duplex=False)
            # the pool's state by keyword: its workers' epochs, pids and readiness (a shared array, passed at the spawn)
            p = ctx.Process(target=frontend_main, args=(cfg, i, ipc_path, s),
                            kwargs={"pool_state": pool.state if pool is not None else None},
                            name=f"llm-mini-infer-frontend-{i}")
            p.start()
            s.close()
            frontends.append(p)
            procs.append(p)
            readies.append(r)
        for i, (r, p) in enumerate(zip(readies, frontends)):
            _recv(r, p, 300, f"frontend {i}", signals, check=lambda: _check_started(engine, watchdog, pool))
        log.info("ready in %.1f s: http://%s:%d (%d frontend%s, %s runner)", time.perf_counter() - t0,
                 cfg.server.host, cfg.server.port, n, "" if n == 1 else "s", cfg.runner.kind)
        code = _supervise(cfg, procs, frontends, engine, watchdog, signals, wake_r, pool)
    except StopWhileStarting as exc:
        log.error("%s: dumping %s, then stopping", exc, "the engine's stacks" if exc.proc is engine else
                  f"the stacks of {exc.what} and of the engine")
        if exc.proc is not engine and exc.proc.is_alive():
            _dump_stack(exc.proc)
            exc.proc.join(STACK_WAIT_S)
        _signal(frontends, signal.SIGUSR1)
        _kill_engine(engine)
        code = 128 + exc.signum
    except RuntimeError as exc:
        log.error("%s", exc)
        _signal(frontends, signal.SIGUSR1)
    finally:
        if pool is not None:
            pool.close()
        _shutdown(cfg, frontends, engine, ipc_path, pool)
        if pool is not None:
            pool.stop()
        if cfg.pixel_arena:
            from .vision import arena

            arena.remove(cfg.pixel_arena)
        shutil.rmtree(ipc_path, ignore_errors=True)
        signal.set_wakeup_fd(-1)
        wake_r.close()
        wake_w.close()
    log.info("stopped (exit code %d)", code)
    return code


def _check_started(engine: Any, watchdog: Watchdog, pool: ImagePool | None = None) -> None:
    """While the image pool's workers and then the frontends start: the engine and, once its workers are ready, the
    pool, as `_supervise` watches them. A death, a hang or a crash loop is a RuntimeError, the start's failure path
    (exit 1, SIGUSR1 to the frontends)."""
    if not engine.is_alive():
        raise RuntimeError(f"{engine.name} exited while the server started ({_exit(engine.exitcode)})")
    stale = watchdog.check()
    if stale is not None:
        _kill_engine(engine)
        raise RuntimeError(f"the engine loop made no progress for {stale:.0f} s (engine.watchdog_s) while the server "
                           "started: stopped it")
    if pool is not None and pool.supervise():
        raise RuntimeError("the image pool's workers kept dying while the server started")


class Stage(IntEnum):
    """`_supervise`'s stop stage: RUNNING, DRAINING once the first stop signal is passed on to the frontends, FAILING
    once a second one sent them SIGUSR1. Two signals queued before a turn take both steps in it."""

    RUNNING = 0
    DRAINING = 1
    FAILING = 2


def _fail_now(frontends: list[Any], fmt: str | None = None, *args: Any) -> int:
    """The supervision's failure exit: `fmt % args` logged as an error (unless the caller logged already), SIGUSR1 to
    the frontends (they fail their in-flight requests at once), and the exit code, 1."""
    if fmt is not None:
        log.error(fmt, *args)
    _signal(frontends, signal.SIGUSR1)
    return 1


def _supervise(cfg: Config, procs: list[Any], frontends: list[Any], engine: Any, watchdog: Watchdog,
               signals: list[int], wake_r: socket.socket, pool: ImagePool | None = None) -> int:
    """0 once the frontends have drained after a stop signal; 1 once they have exited after a second one, or after a
    drain in which a frontend exited non-zero or the engine died, or at a failure, after SIGUSR1 told them to fail
    their in-flight requests (module docstring). `watchdog`: armed when the engine reported ready."""
    stage = Stage.RUNNING
    while True:
        if signals and stage is Stage.RUNNING:
            grace = 0.0 if signals[0] == signal.SIGINT else cfg.server.drain_grace_s
            log.info("%s: draining (new requests served for %g s more, in-flight ones finish within %g s)",
                     signal.Signals(signals[0]).name, grace, grace + cfg.server.drain_timeout_s)
            _signal(frontends, signal.SIGINT if signals[0] == signal.SIGINT else signal.SIGTERM)
            stage = Stage.DRAINING
        if len(signals) > 1 and stage is Stage.DRAINING:
            log.warning("%s again: failing in-flight requests now", signal.Signals(signals[-1]).name)
            _signal(frontends, signal.SIGUSR1)
            stage = Stage.FAILING
        if stage is not Stage.RUNNING and not any(p.is_alive() for p in frontends):
            failed = [p for p in frontends if p.exitcode] + ([] if engine.is_alive() else [engine])
            if failed:
                log.error("frontends drained, but %s", "; ".join(
                    f"{p.name} exited during the drain ({_exit(p.exitcode)})" for p in failed))
                return 1
            log.info("frontends drained")
            return 0 if stage is Stage.DRAINING else 1
        if pool is not None and pool.supervise(failing=stage is Stage.FAILING):  # a crash loop: the exit path
            return _fail_now(frontends)  # `supervise` logged it
        dead = [p for p in procs if not p.is_alive() and not (stage is not Stage.RUNNING and p in frontends)]
        if dead:
            return _fail_now(frontends, "%s exited unexpectedly (code %s): failing in-flight requests and stopping",
                             dead[0].name, dead[0].exitcode)
        stale = watchdog.check()
        if stale is not None:
            # SIGUSR1 first: the stack dump below can take seconds (a stopped process)
            code = _fail_now(frontends, "the engine loop made no progress for %.0f s (engine.watchdog_s): stopping "
                             "it, failing in-flight requests", stale)
            _kill_engine(engine)
            return code
        waits = [*(p.sentinel for p in procs), *(pool.waitables() if pool is not None else ()), wake_r]
        if wake_r in wait(waits, timeout=TURN_S):
            wake_r.recv(64)


class Watchdog:
    """The engine's progress watchdog (`engine.watchdog_s`; module docstring): the engine stamps `heartbeat` every loop
    turn, and `check`, once per supervision turn, finds it stale past `period_s`. A check that comes later than the
    turn's wait by more than OVERSLEEP_S (or half the period) re-arms it instead of judging: the launcher overslept, so
    the host may have frozen it with the engine."""

    def __init__(self, heartbeat: Any, period_s: float, clock: Any = time.monotonic_ns):
        self.heartbeat, self.clock = heartbeat, clock
        self.period_ns = int(period_s * 1e9)
        self.late_ns = int((TURN_S + min(OVERSLEEP_S, period_s / 2)) * 1e9)
        self.last = heartbeat.value = clock()  # armed: the engine just reported ready
        self.rearmed = 0

    def check(self) -> float | None:
        """The engine's staleness in seconds when it is past the period (stop it), else None."""
        now = self.clock()
        gap, self.last = now - self.last, now
        if not self.period_ns:
            return None
        if gap > self.late_ns:
            self.rearmed += 1
            log.warning("the launcher's supervision ran %.1f s after its previous turn (a turn waits %g s): a host "
                        "freeze, a stopped launcher or a slow turn; re-arming the engine's watchdog instead of judging "
                        "it", gap / 1e9, TURN_S)
            self.heartbeat.value = now
            return None
        stale = now - self.heartbeat.value
        return stale / 1e9 if stale > self.period_ns else None


def _signal(procs: list[Any], sig: int) -> None:
    for p in procs:
        if p.is_alive():
            with contextlib.suppress(ProcessLookupError):
                os.kill(p.pid, sig)


def _kill_engine(engine: Any) -> None:
    """A hung engine: SIGABRT first (its faulthandler prints every thread's stack to the log), then SIGKILL."""
    if engine is None or not engine.is_alive():
        return
    os.kill(engine.pid, signal.SIGABRT)
    engine.join(3)
    if engine.is_alive():
        engine.kill()
        engine.join(5)


def _start_images(cfg: Config, ctx: Any, ipc_path: str) -> ImagePool | None:
    """With images on: create the pixel arena, and the image pool that will spawn into it (architecture.md, "The image
    pool")."""
    im = cfg.images
    if not im.enabled:
        return None
    from .vision import arena

    path = arena.arena_path(arena.default_dir(ipc_path), f"{os.getpid()}-{os.path.basename(ipc_path)}")
    try:
        arena.create(path, cfg.dev.images.arena_mb << 20)
    except OSError:
        arena.remove(path)
        raise
    cfg.pixel_arena = path
    cfg.pool_dir = ipc_path
    return ImagePool(cfg, ctx, ipc_path, path)


class Phase(IntEnum):
    """An image pool worker's phase (`Worker.life`): STARTING until it reports ready (READY), RETIRED once `reap` saw
    it dead; it can die starting, and a respawn is a new worker. Nothing moves it back."""

    STARTING = 0
    READY = 1
    RETIRED = 2


_PHASES = {Phase.STARTING: (Phase.READY, Phase.RETIRED), Phase.READY: (Phase.RETIRED,), Phase.RETIRED: ()}


class Kill(IntEnum):
    """Why the launcher killed a pool worker, recorded before the kill (`Worker.mark`), in the order `reap` reads them:
    a higher one wins when two were recorded (a worker that outlived its first kill's join)."""

    NONE = 0
    UNREADY = 1  # not ready POOL_READY_TIMEOUT_S after its respawn (`supervise`)
    STOPPED = 2  # still starting as the server stops (`close`)
    EXPIRED = 3  # past the job deadline (`expire`), with the start of the job that expired (`Worker.since`)


class Worker:
    """One pool worker's process, as long as it is the one at its index: `spawn` makes a new record for a respawn."""

    __slots__ = ("kill", "life", "proc", "since")

    def __init__(self, proc: Any) -> None:
        self.proc = proc
        self.life = Lifecycle("an image pool worker", Phase.STARTING, _PHASES)
        self.kill = Kill.NONE
        self.since = 0  # EXPIRED: the start of the job it held past the deadline

    def mark(self, kill: Kill, since: int = 0) -> None:
        if kill > self.kill:
            self.kill, self.since = kill, since


class ImagePool:
    """The image pool's workers as the launcher runs them (architecture.md, "The image pool"): spawned before the
    engine, and each respawned under a new epoch when it dies or is killed at the job deadline (`vision.pool`), unless
    they keep dying."""

    def __init__(self, cfg: Config, ctx: Any, ipc_path: str, arena: str):
        from .vision.pool import PoolState, job_deadline_s, worker_main

        im = cfg.images
        self.ctx = ctx
        self.target = worker_main
        self.state = PoolState(ctx, im.workers or 1)
        # every worker's parameters, fixed here (the engine's copy of the configuration replaces cfg later); the later
        # ones by keyword, so a new one cannot shift them
        self.args = (cfg.model.path, im.min_pixels, im.max_pixels)
        self.kwargs = {"max_soft_tokens": im.max_soft_tokens, "max_source_pixels": im.max_source_pixels,
                       "state": self.state}
        self.ipc_path, self.arena, self.log_level = ipc_path, arena, cfg.observability.log_level
        self.deadline_s = job_deadline_s(im.max_source_pixels)
        self.workers: list[Worker] = []  # by index: the current worker's record (`spawn` replaces it)
        # respawned workers' ready pipes, until they report: (index, spawned at, the process)
        self.starting: dict[Connection, tuple[int, float, Any]] = {}
        self.deaths: deque[float] = deque()  # within the breaker's window

    @property
    def procs(self) -> list[Any]:
        """The workers' processes, by index."""
        return [w.proc for w in self.workers]

    def start(self) -> list[Connection]:
        """Spawn every worker (epoch 0); their ready pipes, which the launcher awaits before the frontends start."""
        return [self.spawn(i) for i in range(self.state.workers)]

    def spawn(self, i: int, epoch: int = 0) -> Connection:
        r, s = self.ctx.Pipe(duplex=False)
        p = self.ctx.Process(target=self.target, args=(*self.args, i, self.ipc_path, self.arena, s, self.log_level),
                             kwargs={**self.kwargs, "epoch": epoch}, name=f"llm-mini-infer-pool-{i}")
        p.start()
        s.close()
        self.workers[i:i + 1] = [Worker(p)]
        self.state.set(i, epoch=epoch, pid=p.pid or 0, ready=False)
        return r

    def _mark(self, p: Any, kill: Kill) -> None:
        """Record why the launcher kills `p`, on its worker's record (a replaced process has none)."""
        for w in self.workers:
            if w.proc is p:
                w.mark(kill)

    def mark_ready(self, i: int) -> None:
        """Worker i reported ready (the launcher's wait for the first ones, or `supervise` for a respawn)."""
        w = self.workers[i]
        if w.life.phase is Phase.STARTING:
            w.life.to(Phase.READY)
        self.state.set(i, ready=True)

    def waitables(self) -> list[Any]:
        return [*(p.sentinel for p in self.procs), *self.starting]

    def supervise(self, failing: bool = False) -> bool:
        """One turn of the launcher's loop: kill the workers past the job deadline (`expire`), mark respawned workers
        ready (or kill one still starting after POOL_READY_TIMEOUT_S), reap the dead workers (`reap`), then respawn each
        under its new epoch. True: a crash loop (more than POOL_DEATHS deaths within POOL_DEATH_WINDOW_S) or a death
        while the frontends fail everything (`failing`), for the exit path, which respawns none."""
        now = time.monotonic()
        self.expire()
        for r, (i, t0, p) in list(self.starting.items()):
            if r.poll():
                with contextlib.suppress(EOFError, OSError):  # EOF: it died starting (below)
                    r.recv()
                    self.mark_ready(i)
                    log.info("image pool worker %d ready again in %.1f s (pid %d, epoch %d)", i, now - t0, p.pid,
                             self.state.epoch(i))
            elif now - t0 <= POOL_READY_TIMEOUT_S:
                continue
            else:
                log.error("image pool worker %d not ready %.0f s after its respawn: killing it", i, now - t0)
                self._mark(p, Kill.UNREADY)
                p.kill()
                p.join(5)
            r.close()
            del self.starting[r]
        dead = self.reap()
        if not dead:
            return False
        self.deaths.extend(now for _ in dead)
        while now - self.deaths[0] > POOL_DEATH_WINDOW_S:
            self.deaths.popleft()
        if failing or len(self.deaths) > POOL_DEATHS:
            why = ("the frontends are failing every request" if failing else
                   f"{len(self.deaths)} image pool worker deaths in {POOL_DEATH_WINDOW_S:g} s, a crash loop")
            for i, _, how, blamed in dead:
                log.error("%s %s %s: %s; failing in-flight requests and stopping", self.procs[i].name, how,
                          _decoding(blamed), why)
            return True
        for k, (i, _, how, blamed) in enumerate(dead, len(self.deaths) - len(dead) + 1):
            log.error("%s %s %s; its other images are re-sent; respawning it (epoch %d; death %d of the %d the pool "
                      "takes in %g s)", self.procs[i].name, how, _decoding(blamed), self.state.epoch(i), k,
                      POOL_DEATHS, POOL_DEATH_WINDOW_S)
            r = self.spawn(i, self.state.epoch(i))
            self.starting[r] = (i, now, self.procs[i])
        return False

    def expire(self) -> None:
        """Kill each worker that has held one job past the deadline (`vision.pool.job_deadline_s`: a decoder hang): its
        Python stack goes to the log first (SIGUSR2, which its faulthandler dumps, unless the process is stopped), then
        SIGKILL, unless that job ended meanwhile. `reap` then retires it as a crashed one: that job fails (no other it
        started after), the others are re-sent."""
        now = time.monotonic_ns()
        for i, w in enumerate(self.workers):
            p, since = w.proc, self.state.since(i)
            if not since or (now - since) / 1e9 <= self.deadline_s or w.kill is Kill.EXPIRED or not p.is_alive():
                continue
            log.error("image pool worker %d has held one image for %.1f s, past the job deadline (%.0f s): killing it, "
                      "its Python stack first", i, (now - since) / 1e9, self.deadline_s)
            _dump_stack(p)
            p.join(STACK_WAIT_S)
            if self.state.since(i) != since:  # slow, not hung: it finished that image while its stack was printed
                log.warning("image pool worker %d finished that image after all: not killed", i)
                continue
            w.mark(Kill.EXPIRED, since)
            self.state.count_hang()
            p.kill()
            p.join(5)

    def reap(self) -> list[tuple[int, Kill, str, int]]:
        """Every worker dead since the last look, as (index, the kill it died of or NONE, how it ended, the job it was
        decoding or 0): each is retired (`PoolState.retire`: that job recorded, the index not ready, its epoch moved),
        so the frontends fail that job and re-send its others whatever comes next (its respawn, or the exit path:
        `reap_stopping`), and its ready pipe goes with it. A kill blames no image unless it took effect (SIGKILL), or,
        at the job deadline, the job that expired is still the one it was decoding."""
        dead = []
        for i, w in enumerate(self.workers):
            p = w.proc
            if w.life.phase is Phase.RETIRED or p.is_alive():
                continue
            w.life.to(Phase.RETIRED)
            for r, (_, _, q) in list(self.starting.items()):
                if q is p:
                    r.close()
                    del self.starting[r]
            kill = w.kill
            if kill is Kill.EXPIRED:  # blame the image that ran past the deadline, not one started after it
                blame = self.state.since(i) == w.since
                how = f"was killed at the job deadline ({self.deadline_s:.0f} s)"
            elif kill is Kill.STOPPED and p.exitcode == -signal.SIGKILL:  # ours, not a crash just before it
                blame, how = False, "was stopped while starting"
            elif kill is Kill.UNREADY and p.exitcode == -signal.SIGKILL:
                blame, how = False, f"was killed, not ready {POOL_READY_TIMEOUT_S:.0f} s after its respawn,"
            else:
                kill, blame, how = Kill.NONE, True, f"exited unexpectedly ({_exit(p.exitcode)})"
            dead.append((i, kill, how, self.state.retire(i, blame=blame)))
        return dead

    def close(self) -> None:
        """The server stops: no worker is respawned any more. Those still starting are stopped (nothing would mark them
        ready), so once the pool is marked closed a worker that is not ready is gone for good, and the frontends fail
        what waits for one at once (`vision.pool.CLOSED`, a 503) instead of holding its request until the drain's
        deadline. Ready workers serve on until `stop`."""
        for _, _, p in list(self.starting.values()):
            self._mark(p, Kill.STOPPED)
            p.kill()
            p.join(5)
        self.reap_stopping()
        self.state.close()

    def reap_stopping(self) -> None:
        """While the server stops (`close`, `_shutdown`): the deadline and the reaping go on, the respawns do not."""
        self.expire()
        for i, kill, how, blamed in self.reap():
            if kill is Kill.STOPPED:
                log.info("%s %s: the server is stopping", self.procs[i].name, how)
            else:
                log.error("%s %s %s; the server is stopping, so it is not respawned", self.procs[i].name, how,
                          _decoding(blamed))

    def stop(self) -> None:
        """The workers hold no state but their share of the arena, which the launcher removes: SIGKILL (they ignore
        SIGTERM while the launcher lives)."""
        for p in self.procs:
            if p.is_alive():
                p.kill()
        for p in self.procs:
            p.join(5)
        for r in self.starting:
            r.close()
        self.starting.clear()


def _decoding(blamed: int) -> str:
    """What a dead worker was doing (`ImagePool.reap`)."""
    return "while decoding an image, which fails with a 400" if blamed else "between images"


def _dump_stack(p: Any) -> None:
    """A pool worker's or a frontend's Python stack to the log: their faulthandler dumps it on SIGUSR2
    (`vision.pool.worker_main`, `api.server.frontend_main`)."""
    with contextlib.suppress(ProcessLookupError):
        os.kill(p.pid, signal.SIGUSR2)


def _exit(code: int | None) -> str:
    """A process's exit code as the log names it: `code 1`, or `SIGSEGV` for a signal."""
    if code is not None and code < 0:
        with contextlib.suppress(ValueError):
            return signal.Signals(-code).name
    return f"code {code}"


def _shutdown(cfg: Config, frontends: list[Any], engine: Any, ipc_path: str, pool: ImagePool | None = None) -> None:
    """After `_supervise` (or a failed startup): the frontends are exiting (drained, or failing their requests); give
    them their drain time, then stop the engine. Meanwhile the closed pool is still watched (`reap_stopping`): an image
    worker that dies or hangs still has its images failed or sent elsewhere, which a frontend may be waiting on."""
    deadline = time.monotonic() + cfg.server.drain_grace_s + cfg.server.drain_timeout_s + 5
    for p in frontends:
        while p.is_alive() and time.monotonic() < deadline:
            p.join(min(0.25, max(0.0, deadline - time.monotonic())))
            if pool is not None:
                pool.reap_stopping()
        if p.is_alive():
            log.warning("%s did not drain in time; killing it", p.name)
            p.kill()
            p.join(5)
    if engine is not None and engine.is_alive():
        zctx = zmq.Context()
        sock = zctx.socket(zmq.PUSH)
        ipc.configure_socket(sock)
        sock.connect(ipc.engine_input_addr(ipc_path))
        sock.send_multipart(ipc.Encoder().encode(Shutdown(reason="launcher stopping")))
        engine.join(10)
        sock.close(0)
        zctx.term()
        if engine.is_alive():
            log.warning("engine did not stop; killing it")
            engine.kill()
            engine.join(5)
