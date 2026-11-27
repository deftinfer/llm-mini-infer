"""The image process pool (architecture.md, "The image pool"): the exact CPU half of every image, off the frontends'
event loops.

**Workers** (`worker_main`), spawned by the launcher before the engine, so they import torch while it loads the model.
Each runs one torch thread (`OMP_NUM_THREADS=1` and friends are set before torch is imported), so the pool never
oversubscribes the cores with default-sized thread pools. A worker binds a ROUTER socket (`ipc://<dir>/pool-<i>.sock`)
and serves jobs in arrival order: decode, orient and resize the image (the family's `pixels`: Pillow orients it, and
the target turns with it), write the pixels into the arena at the job's offset, hash the image (`family.image_digest`),
and answer with whether it turned the image, the digest and the time it took, or with the error.

**Clients** (`PoolClient`, one per frontend process) connect a DEALER socket to every worker and send each job to the
ready worker with the fewest of this frontend's jobs outstanding (several frontends may queue on one worker; jobs take
about a millisecond). Replies are read from the event loop through reader callbacks on the sockets' file descriptors,
like the engine's outputs (`api.engine_client`), and resolve the job's future.

**A worker that dies or hangs** past the job deadline is respawned by the launcher under a new epoch (`PoolState`, in
shared memory: each worker's epoch, pid, readiness and the job it started). A death blames that one job: the client
that sees the new epoch fails it (`STOPPED`, a 400: the image may have caused it) and re-sends its other jobs there to a
live worker once (`UNSERVED`, a 503, if that worker too stops before decoding it). Every job carries the epoch it was
sent under, and a worker refuses another epoch's job before decoding it (`REFUSED`), so a job left in a queue never
writes into a freed arena range. When the server stops, the launcher closes the pool, and a job waiting for a worker
that is not ready fails at once (`CLOSED`, a 503).
"""

from __future__ import annotations

import asyncio
import contextlib
import faulthandler
import logging
import os
import signal
import time
from collections.abc import Callable
from multiprocessing.connection import Connection
from typing import Any

import msgspec
import zmq

from .probe import MAX_SOURCE_PIXELS  # the standard library only: safe before the workers' thread settings

log = logging.getLogger("llm_mini_infer.pool")

_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS", "TORCH_NUM_THREADS")
EPOCH_POLL_S = 0.1  # how often a client with jobs outstanding looks for a respawned worker
STOPPED = ("the image pool's worker stopped while decoding this image (it crashed, or ran past the job deadline; the "
           "image may have caused it)")
CLOSED = "the image pool stopped before decoding this image: the server is stopping; retry it"
UNSERVED = "two image pool workers stopped before decoding this image; retry it"
MAX_SENDS = 2  # a job is re-sent once: a worker death that never started it, then another, fails it (UNSERVED)
REFUSED = "the image reached the worker that replaced the one it was sent to"  # a stateless client's result only
# A job's deadline: the worst job images.max_source_pixels allows, with a wide margin. At 178.9 million pixels, one
# worker's job took 0.2-1 s on one fast desktop core, and 3.4 s for a JPEG that its EXIF orientation rotates (19 ns a
# pixel; noisier images, past images.max_bytes at that size, took up to 22). 25 ns a pixel bounds it, and 10 times that
# leaves room for a slower core, a throttled cgroup and a busy host: 44.7 s at the limit's default, which bounds how
# long the images queued behind a hung one wait. Never under 5 s (the launcher looks once a second).
JOB_NS_PER_PIXEL = 25
JOB_DEADLINE_MARGIN = 10
JOB_DEADLINE_MIN_S = 5.0
# A job's id: the client's count (wrapping), then the client's frontend index, so a dead worker's record names one
# frontend's job; the record keeps it next to the epoch that ended, in one word
_CLIENT_BITS = 12
_JID_BITS = 48
_JID_MASK = (1 << _JID_BITS) - 1
_COUNT_MAX = (1 << (_JID_BITS - _CLIENT_BITS)) - 1
_TAG = (1 << (63 - _JID_BITS)) - 1


def job_deadline_s(max_source_pixels: int) -> float:
    """Seconds a worker may hold one job before the launcher kills it."""
    return max(JOB_DEADLINE_MIN_S, JOB_DEADLINE_MARGIN * JOB_NS_PER_PIXEL * max_source_pixels / 1e9)


class Job(msgspec.Struct, array_like=True, gc=False):
    jid: int
    fmt: str
    width: int  # the stored size (vision.probe.Header)
    height: int
    out_h: int  # the resize target of the image as stored (vision.family.ImageSize)
    out_w: int
    grid_t: int
    grid_h: int
    grid_w: int
    tokens: int
    offset: int  # where the uint8 [3, out_h, out_w] pixels go in the arena
    epoch: int = 0  # the worker's epoch the client sent it under (PoolState)


class Done(msgspec.Struct, array_like=True, gc=False):
    jid: int
    error: str = ""
    decode_ns: int = 0  # decode, orientation and resize
    write_ns: int = 0  # the copy into the arena and the hash
    transposed: bool = False  # Pillow turned the image by a quarter: the pixels are the job's target turned
    digest: bytes = b""  # `family.image_digest` of the bytes and the oriented target


class PoolState:
    """The pool's shared state, in shared memory: created before the workers and frontends spawn, and passed to them as
    a process argument. Every word has one writer at a time. Per worker:
    - its epoch (how many workers this index had before), pid and readiness: the launcher writes them, every client
      reads them;
    - the job it is running and since when (`time.monotonic_ns`; 0 between jobs): the worker writes them, the launcher
      reads them (the deadline, and the job to blame when the worker dies) and clears them before the next worker;
    - the job it was running when it stopped (`lost`): the launcher writes it before the epoch moves, in one word with
      the epoch that ended, so a client never takes a later death's record for the one it handles.
    For the pool: closed (the server stops: no worker is respawned any more), and the workers killed at the deadline."""

    _HEAD = 2  # closed, hangs
    _PER = 6  # epoch, pid, ready, job, since, lost

    def __init__(self, ctx: Any, workers: int):
        self.workers = workers
        self._a = ctx.RawArray("q", self._HEAD + self._PER * workers)

    def _at(self, i: int, k: int) -> int:
        return self._HEAD + self._PER * i + k

    def epoch(self, i: int) -> int:
        return self._a[self._at(i, 0)]

    def pid(self, i: int) -> int:
        return self._a[self._at(i, 1)]

    def ready(self, i: int) -> bool:
        return bool(self._a[self._at(i, 2)])

    def since(self, i: int) -> int:
        return self._a[self._at(i, 4)]

    @property
    def closed(self) -> bool:
        return bool(self._a[0])

    @property
    def hangs(self) -> int:
        return self._a[1]

    def set(self, i: int, *, epoch: int | None = None, pid: int | None = None, ready: bool | None = None) -> None:
        for k, v in ((0, epoch), (1, pid), (2, ready)):
            if v is not None:
                self._a[self._at(i, k)] = int(v)

    # -- the worker's ------------------------------------------------------------------------------------------------

    def start(self, i: int, jid: int) -> None:
        self._a[self._at(i, 3)] = jid
        self._a[self._at(i, 4)] = time.monotonic_ns()

    def finish(self, i: int) -> None:
        self._a[self._at(i, 4)] = 0

    # -- the launcher's ----------------------------------------------------------------------------------------------

    def retire(self, i: int, blame: bool = True) -> int:
        """Worker i's process is gone: record the job it was running (none without `blame`), clear it for the next
        worker, then move the index's epoch and mark it not ready. The job recorded (0: none)."""
        epoch = self.epoch(i)
        jid = self._a[self._at(i, 3)] if blame and self.since(i) else 0
        self._a[self._at(i, 5)] = (epoch & _TAG) << _JID_BITS | jid
        self._a[self._at(i, 3)] = self._a[self._at(i, 4)] = 0
        self.set(i, epoch=epoch + 1)  # first: a client that sees the index not ready then sees its new epoch
        self.set(i, ready=False)
        return jid

    def close(self) -> None:
        self._a[0] = 1

    def count_hang(self) -> None:
        self._a[1] += 1

    # -- a client's --------------------------------------------------------------------------------------------------

    def lost(self, i: int, epoch: int) -> int | None:
        """The job worker i was running when its `epoch` ended (0: none), or None when the record is another epoch's:
        the index stopped again before this client looked, and which of its jobs was running is unknown."""
        v = self._a[self._at(i, 5)]
        return v & _JID_MASK if v >> _JID_BITS == epoch & _TAG else None


def endpoint(ipc_dir: str, index: int) -> str:
    return f"ipc://{ipc_dir}/pool-{index}.sock"


def worker_main(model_dir: str, min_pixels: int | None, max_pixels: int | None, index: int, ipc_dir: str,
                arena_path: str, ready: Connection | None, log_level: str = "info",
                max_soft_tokens: int | None = None, max_source_pixels: int = MAX_SOURCE_PIXELS, epoch: int = 0,
                state: PoolState | None = None) -> None:
    """Process entry point (spawned by the launcher, which passes the parameters after `log_level` by keyword). The
    family is loaded with the derived `images` sizing, as everywhere: a worker that sized images otherwise would fail
    Gemma 4's pixel-bound check before it reports ready. `epoch`: how many workers this index had before this one; a
    job sent under another epoch is refused undecoded. `state`: the launcher's `PoolState`, where the worker records
    the job it is running (None: tests' workers, which no launcher watches)."""
    for var in _THREAD_VARS:
        os.environ[var] = "1"
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # the launcher stops the pool after the frontends and the engine
    faulthandler.enable()  # a native crash prints the Python stack (never the image) before the launcher respawns it
    faulthandler.register(signal.SIGUSR2, all_threads=True)  # so does a job past its deadline, before the SIGKILL
    from ..logutil import setup_logging

    setup_logging(log_level, f"pool-{index}", without_messages=True)
    import numpy as np
    import torch

    torch.set_num_threads(1)
    with contextlib.suppress(RuntimeError):
        torch.set_num_interop_threads(1)
    from . import family as fam
    from .arena import map_writable
    from .family import ImageSize, image_digest
    from .probe import Header, check_source_pixels

    family = fam.load(model_dir, min_pixels=min_pixels, max_pixels=max_pixels, max_soft_tokens=max_soft_tokens)
    if family is None:
        raise SystemExit(f"{model_dir} has no image family")
    arena = map_writable(arena_path)
    out_view = np.frombuffer(arena.buf, np.uint8)
    ctx = zmq.Context(io_threads=1)
    sock = ctx.socket(zmq.ROUTER)
    sock.setsockopt(zmq.SNDHWM, 0)
    sock.setsockopt(zmq.RCVHWM, 0)
    sock.setsockopt(zmq.LINGER, 0)
    sock.bind(endpoint(ipc_dir, index))
    dec = msgspec.msgpack.Decoder(Job)
    enc = msgspec.msgpack.Encoder()
    # Warm the decoders and the resize once (lazy imports, first-call costs) before reporting ready.
    try:
        _warm(family, np)
    except Exception:  # pragma: no cover - a broken install shows on the first job instead
        log.exception("pool worker %d: warm-up failed", index)
    if ready is not None:
        ready.send("ready")
        ready.close()
    # The launcher stops the pool itself after the frontends drained (SIGKILL); a SIGTERM sent to the whole process
    # group must not stop it before. Orphaned (the launcher gone), the worker stops on SIGTERM or by itself.
    parent = os.getppid()
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=os.getppid() != parent))
    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)
    while not stop["flag"]:
        if not poller.poll(500):
            if os.getppid() != parent:
                break
            continue
        ident, head, data = sock.recv_multipart(copy=False)
        job = dec.decode(head.buffer)
        reply = Done(jid=job.jid)
        if job.epoch != epoch:  # sent to the worker this one replaced: re-sent or failed, its range maybe reused
            reply.error = REFUSED
            sock.send_multipart([ident, enc.encode(reply)], copy=False)
            continue
        if state is not None:  # from here a death blames this job; the launcher's deadline runs
            state.start(index, job.jid)
        t0 = time.perf_counter_ns()
        try:
            header = Header(job.fmt, job.width, job.height)
            check_source_pixels(header, max_source_pixels)  # the frontend refused it already: never let a decoder try
            size = ImageSize(job.out_h, job.out_w, (job.grid_t, job.grid_h, job.grid_w), job.tokens)
            px = family.pixels(data.bytes, header, size)
            t1 = time.perf_counter_ns()
            out = oriented(family, header, size, tuple(px.shape[1:]))
            n = 3 * out.height * out.width
            if job.offset < 0 or job.offset + n > arena.size:
                raise ValueError(f"arena range [{job.offset}, {job.offset + n}) out of bounds")
            out_view[job.offset:job.offset + n] = px.numpy().reshape(-1)
            reply.transposed, reply.digest = out is not size, image_digest(data.buffer, out)
            reply.decode_ns, reply.write_ns = t1 - t0, time.perf_counter_ns() - t1
        except Exception as exc:  # a corrupt image: the client answers 400
            reply.error = f"{type(exc).__name__}: {exc}"[:500]
        if state is not None:  # its pixels are written: a death from here on re-sends it (the same pixels again)
            state.finish(index)
        sock.send_multipart([ident, enc.encode(reply)], copy=False)
    sock.close(0)
    ctx.term()
    del out_view
    arena.close()


class Asymmetric(ValueError):
    """The family sized the image turned otherwise than its target turned (`oriented`): a defect of the family, which
    fails this image (a 400 with this text) before its pixels are written."""


def oriented(family: Any, header: Any, size: Any, shape: tuple[int, ...]) -> Any:
    """The size of a job's pixels: its target (the image as stored), or that target turned when Pillow turned the image
    (`family`'s module docstring). The frontend sized the span and the arena range from the target, which holds only if
    the family sizes the turned image as the turned target: checked here, so a family that does not fails the image
    (`Asymmetric`) instead of writing pixels HF's processors would size otherwise."""
    from .family import transposed
    from .probe import Header

    if shape == (size.height, size.width):
        return size
    turned = transposed(size)
    if shape != (turned.height, turned.width):
        raise ValueError(f"the pixels are {shape[1]}x{shape[0]}, but the target is {size.width}x{size.height}")
    want = family.size(Header(header.format, header.height, header.width))
    if want != turned:
        raise Asymmetric(f"the image family sizes the turned image {want.width}x{want.height} ({want.tokens} tokens), "
                         f"not {turned.width}x{turned.height} ({turned.tokens}): the frontend sized it as stored")
    return turned


def _warm(family: Any, np: Any) -> None:
    import io

    from PIL import Image

    from .family import image_digest
    from .probe import probe

    for fmt in ("JPEG", "PNG"):
        buf = io.BytesIO()
        Image.new("RGB", (64, 48), (10, 20, 30)).save(buf, format=fmt)
        data = buf.getvalue()
        header = probe(data)
        size = family.size(header)
        family.pixels(data, header, size)
        image_digest(data, size)


class PoolClient:
    """A frontend's link to the pool workers (module docstring). `state`: the launcher's `PoolState` (None: workers
    are never respawned, as in tests that run their own); `index`: the frontend's, part of every job id."""

    def __init__(self, ipc_dir: str, workers: int, state: PoolState | None = None, index: int = 0):
        if not 0 <= index < 1 << _CLIENT_BITS:
            raise ValueError(f"a pool client's index must be below {1 << _CLIENT_BITS}, got {index}")
        self._ctx = zmq.Context()
        self._socks: list[zmq.Socket] = []
        for i in range(workers):
            s = self._ctx.socket(zmq.DEALER)
            s.setsockopt(zmq.SNDHWM, 0)
            s.setsockopt(zmq.RCVHWM, 0)
            s.setsockopt(zmq.LINGER, 0)
            s.connect(endpoint(ipc_dir, i))
            self._socks.append(s)
        self._outstanding = [0] * workers
        # jid -> (its future, its worker, the worker's epoch it went under, the job and its bytes: kept to re-send it
        # if the worker stops before decoding it, and how many times it was sent)
        self._futures: dict[int, tuple[asyncio.Future, int, int, Job, Any, int]] = {}
        self._index = index
        self._count = 0
        self._enc = msgspec.msgpack.Encoder()
        self._dec = msgspec.msgpack.Decoder(Done)
        self._loop: asyncio.AbstractEventLoop | None = None
        self.state = state
        self._epochs = [state.epoch(i) for i in range(workers)] if state is not None else [0] * workers
        self._poll: asyncio.TimerHandle | None = None
        self.failed = 0  # jobs failed because their worker stopped while decoding them (metrics)
        self.resent = 0  # jobs re-sent because their worker stopped before it decoded them (metrics)
        self.on_done: Callable[[Done], None] | None = None  # metrics hook
        # (worker, its new epoch, this client's jobs that failed with it, and those re-sent)
        self.on_restart: Callable[[int, int, int, int], None] | None = None

    def start(self) -> None:
        loop = self._loop = asyncio.get_running_loop()
        for i, s in enumerate(self._socks):
            loop.add_reader(s.get(zmq.FD), self._on_readable, i)
            loop.call_soon(self._on_readable, i)

    def close(self) -> None:
        if self._poll is not None:
            self._poll.cancel()
        if self._loop is not None:
            for s in self._socks:
                self._loop.remove_reader(s.get(zmq.FD))
        for fut, *_ in self._futures.values():
            if not fut.done():
                fut.cancel()
        self._futures.clear()
        for s in self._socks:
            s.close(0)
        self._ctx.term()

    def submit(self, job: Job, data: bytes) -> asyncio.Future:
        """Send a job (its `jid` and `epoch` are set here; the client keeps it, and may re-send it, until it resolves);
        the future resolves to its `Done`."""
        self.check_epochs()
        fut = asyncio.get_running_loop().create_future()
        self._send(fut, job, data)
        return fut

    def _send(self, fut: asyncio.Future, job: Job, data: Any, sends: int = 1) -> bool:
        """Send the job under a new id (a re-sent job's old copy may still reach its old worker's successor), or fail
        it at once when no worker will ever serve it (`CLOSED`; False)."""
        self._count = self._count % _COUNT_MAX + 1
        job.jid = jid = self._count << _CLIENT_BITS | self._index
        w = self._pick()
        if w is None:
            if not fut.done():
                fut.set_result(Done(jid=jid, error=CLOSED))
            return False
        job.epoch = self._epochs[w]
        self._outstanding[w] += 1
        self._futures[jid] = (fut, w, job.epoch, job, data, sends)
        s = self._socks[w]
        s.send_multipart([self._enc.encode(job), data], copy=False)
        loop = asyncio.get_running_loop()
        # ZMQ's descriptor signals edges, and a send processes the socket's pending commands: a reply that arrived
        # just before it is then readable without the descriptor ever signalling, and the pipe stays silent until it
        # is drained. Read it now, or requests hang on a worker whose replies sit unread.
        if s.get(zmq.EVENTS) & zmq.POLLIN:
            loop.call_soon(self._on_readable, w)
        if self.state is not None and self._poll is None:
            self._poll = loop.call_later(EPOCH_POLL_S, self._on_poll)
        return True

    def _pick(self) -> int | None:
        """The ready worker with the fewest of this client's jobs; with none ready, any (the job waits for a respawn),
        unless the pool is closed (None: no worker will be respawned)."""
        st = self.state
        workers = range(len(self._socks))
        ready = [i for i in workers if st is None or st.ready(i)]
        if not ready:
            if st.closed:
                return None
            ready = workers
        return min(ready, key=self._outstanding.__getitem__)

    def check_epochs(self) -> None:
        """Handle every worker the launcher replaced since the last look (`_replaced`); then, if the pool is closed,
        fail the jobs that wait for a worker that is not ready (`_fail_unserved`)."""
        st = self.state
        if st is None:
            return
        closed = st.closed  # first: the launcher closes the pool after moving the epochs it saw
        moved = []
        for i in range(len(self._socks)):
            epoch = st.epoch(i)
            if epoch != self._epochs[i]:
                moved.append((i, self._epochs[i]))
                self._epochs[i] = epoch  # every one before any re-send, which may go to another of them
        for i, old in moved:
            self._replaced(i, old)
        if closed:
            self._fail_unserved()

    def _replaced(self, i: int, old: int) -> None:
        """Worker i's `old` epoch ended: the launcher moves the epoch only once the process is gone, so nothing writes
        the ranges of the jobs sent under it any more. Its replies already received come first (a reply follows its
        pixels' write); then the job it had started fails (`STOPPED`), and the others are re-sent (or fail with
        `CLOSED`, see `_send`), once: a job whose second worker also stops before decoding it fails (`UNSERVED`, a
        503), so no image waits through every respawn. Without the record of that death (the index stopped again before
        this client looked), every job sent under an old epoch fails."""
        epoch = self._epochs[i]
        self._on_readable(i)
        started = self.state.lost(i, old)
        jobs = [jid for jid, (_, w, sent, *_) in self._futures.items() if w == i and sent != epoch]
        self._outstanding[i] -= len(jobs)
        failed = resent = unserved = 0
        for jid in jobs:
            fut, _, _, job, data, sends = self._futures.pop(jid)
            if started is None or jid == started:
                failed += 1
                error = STOPPED
            elif sends < MAX_SENDS:
                resent += self._send(fut, job, data, sends + 1)
                continue
            else:
                unserved += 1
                error = UNSERVED
            if not fut.done():
                fut.set_result(Done(jid=jid, error=error))
        self.failed += failed
        self.resent += resent
        if unserved:
            log.warning("image pool worker %d stopped before decoding %d of this frontend's images that another had "
                        "already lost: they failed (503)", i, unserved)
        if self.on_restart is not None:
            self.on_restart(i, epoch, failed, resent)

    def _fail_unserved(self) -> None:
        """The pool is closed: a worker that is not ready now never will be (the launcher stopped those still starting),
        so the jobs waiting for one fail (`CLOSED`, a 503: their images stopped nothing)."""
        st = self.state
        # an index whose epoch moved since this client looked is the next look's (`_replaced`: its dying worker's job is
        # a 400, its others go elsewhere), not a gone worker's
        waiting = [jid for jid, (_, w, *_) in self._futures.items()
                   if not st.ready(w) and st.epoch(w) == self._epochs[w]]
        for jid in waiting:
            fut, w, *_ = self._futures.pop(jid)
            self._outstanding[w] -= 1
            if not fut.done():
                fut.set_result(Done(jid=jid, error=CLOSED))
        if waiting:
            log.warning("the image pool is closed: %d of this frontend's images waiting for a worker failed (503)",
                        len(waiting))

    def _on_poll(self) -> None:
        self._poll = None
        self.check_epochs()
        if self._futures and self._poll is None:  # a re-send armed it already
            self._poll = asyncio.get_running_loop().call_later(EPOCH_POLL_S, self._on_poll)

    def _on_readable(self, i: int) -> None:
        s = self._socks[i]
        while s.get(zmq.EVENTS) & zmq.POLLIN:
            frames = s.recv_multipart(zmq.NOBLOCK, copy=False)
            done = self._dec.decode(frames[-1].buffer)
            if done.error == REFUSED and self.state is not None:
                continue  # a copy that reached its worker's successor: this client re-sends it on seeing the new epoch
            got = self._futures.pop(done.jid, None)
            if got is None:  # failed or re-sent since: a late answer
                continue
            fut, w, *_ = got
            self._outstanding[w] -= 1
            if self.on_done is not None:
                self.on_done(done)
            if not fut.done():
                fut.set_result(done)
