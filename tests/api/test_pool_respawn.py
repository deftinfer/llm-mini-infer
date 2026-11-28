"""A dead image pool worker and a hung one: the launcher respawns it under a new epoch, killing a hung one at the job
deadline first; only the image it was decoding fails (a 400, which a client or proxy does not retry), the images queued
behind it go to another worker, the next jobs succeed, a crash loop still takes the exit path, and once the launcher
closes the pool what waits for a worker that is gone fails at once (a 503). The worker's refusal of another epoch's
job, the clients' failing, re-sending and routing, a frontend's pixel ranges, a real worker past the deadline, and the
launcher's respawns, breaker and closing run without fixtures (real workers on a Gemma 4 config.json alone); the server
tests need the Qwen fixture (the fake runner serves its images)."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import multiprocessing as mp
import os
import shutil
import signal
import threading
import time
from multiprocessing.connection import Connection
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np
import pytest

from llm_mini_infer import config, launcher
from llm_mini_infer.engine import ipc as engine_ipc
from llm_mini_infer.synthetic import pattern
from llm_mini_infer.vision import arena as px_arena
from llm_mini_infer.vision.pool import (
    _COUNT_MAX,
    CLOSED,
    REFUSED,
    STOPPED,
    UNSERVED,
    Done,
    Job,
    PoolClient,
    PoolState,
    endpoint,
    job_deadline_s,
)
from llm_mini_infer.vision.probe import probe

from ..support.api_util import Server, env_has_model
from ..support.fixtures import QWEN
from ..support.vision_pool import GEMMA70, encoded, exif, spawn_pool_worker

SPAWN = mp.get_context("spawn")


def gemma_dir(tmp_path: Path) -> Path:
    """A Gemma 4 checkpoint's config.json alone: images of at most 70 soft tokens."""
    gemma = tmp_path / "gemma"
    gemma.mkdir()
    (gemma / "config.json").write_text(json.dumps({"model_type": "gemma4", "vision_config": {
        "patch_size": 16, "pooling_kernel_size": 3, "default_output_length": 70}}))
    return gemma


def job_for(data: bytes, offset: int) -> Job:
    h = probe(data)
    size = GEMMA70.size(h)
    return Job(jid=0, fmt=h.format, width=h.width, height=h.height,
               out_h=size.height, out_w=size.width, grid_t=size.grid[0], grid_h=size.grid[1], grid_w=size.grid[2],
               tokens=size.tokens, offset=offset)


def tiny() -> Job:
    """A job for the echo workers, which never decode (a new one for each submit: the client keeps it until it is
    answered, to re-send it)."""
    return Job(jid=0, fmt="jpeg", width=1, height=1, out_h=1, out_w=1, grid_t=1, grid_h=1,
               grid_w=1, tokens=1, offset=0)


async def until_async(cond, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < deadline, "timed out"
        await asyncio.sleep(0.005)


def test_a_worker_refuses_another_epochs_job_undecoded(tmp_path) -> None:
    """A respawned worker (epoch 1) answers a job sent under epoch 0 with REFUSED and never writes its arena range: the
    job sat in a DEALER's queue while its worker died, its client has re-sent or failed it, and the range may hold
    another request's pixels by now (the whole arena still holds them when the answer arrives, a worker writing before
    it answers). A job of its own epoch, at another offset, decodes as before, and the worker stays alive."""
    gemma = gemma_dir(tmp_path)
    ipc = engine_ipc.ipc_dir()
    arena = str(tmp_path / "arena")
    px_arena.create(arena, 1 << 22)
    proc = None
    try:
        view = px_arena.map_writable(arena)
        mem = np.frombuffer(view.buf, np.uint8)
        mem[:] = 7  # another request's pixels
        proc, ready = spawn_pool_worker(gemma, None, None, ipc, arena, epoch=1)
        assert ready, f"the worker exited before it was ready (exit code {proc.exitcode})"
        data = encoded(pattern(400, 225), "JPEG")
        state = PoolState(SPAWN, 1)
        state.set(0, epoch=1, ready=True)

        async def main(pixels: np.ndarray) -> tuple[Done, Done]:
            old, new = PoolClient(ipc, 1), PoolClient(ipc, 1, state)  # a client still on epoch 0, one on epoch 1
            old.start()
            new.start()
            try:
                stale = await asyncio.wait_for(old.submit(job_for(data, 1 << 20), data), 60)
                assert (pixels == 7).all()
                ok = await asyncio.wait_for(new.submit(job_for(data, 8192), data), 60)
                return stale, ok
            finally:
                old.close()
                new.close()

        stale, ok = asyncio.run(main(mem))
        assert stale.error == REFUSED and stale.decode_ns == 0  # a client without the state takes it as the answer
        n = job_for(data, 0).out_h * job_for(data, 0).out_w * 3
        assert not ok.error and ok.decode_ns > 0 and proc.is_alive()
        assert (mem[8192:8192 + n] != 7).any() and (mem[:8192] == 7).all() and (mem[8192 + n:] == 7).all()
        del mem
        view.close()
    finally:
        if proc is not None:
            proc.kill()
            proc.join(10)
        px_arena.remove(arena)
        shutil.rmtree(ipc, ignore_errors=True)


class EchoWorkers:
    """Pool workers in threads that answer each job as a real worker does, without decoding: its jid, or REFUSED for an
    epoch other than `epochs[i]`; every job is recorded. A held worker records without answering until released, as a
    worker busy decoding (or stopped) does."""

    def __init__(self, ipc: str, n: int):
        import msgspec
        import zmq

        self.jobs: list[list[Job]] = [[] for _ in range(n)]
        self.epochs = [0] * n
        self.held = [threading.Event() for _ in range(n)]
        self.release = [threading.Event() for _ in range(n)]
        self.stop = threading.Event()
        self.threads = []

        def run(i: int) -> None:
            ctx = zmq.Context()
            sock = ctx.socket(zmq.ROUTER)
            sock.bind(endpoint(ipc, i))
            dec, enc = msgspec.msgpack.Decoder(Job), msgspec.msgpack.Encoder()
            pending = []
            while not self.stop.is_set():
                if sock.poll(10):
                    ident, head, _ = sock.recv_multipart()
                    job = dec.decode(head)
                    self.jobs[i].append(job)
                    pending.append((ident, job))
                if pending and (not self.held[i].is_set() or self.release[i].is_set()):
                    for ident, job in pending:
                        error = "" if job.epoch == self.epochs[i] else REFUSED
                        sock.send_multipart([ident, enc.encode(Done(jid=job.jid, error=error))])
                    pending.clear()
            sock.close(0)
            ctx.term()

        for i in range(n):
            t = threading.Thread(target=run, args=(i,), daemon=True)
            t.start()
            self.threads.append(t)

    def close(self) -> None:
        self.stop.set()
        for t in self.threads:
            t.join(5)


def test_a_dead_worker_fails_only_the_image_it_started() -> None:
    """Worker 0 dies decoding one image with two more queued behind it, one of them another frontend's. The launcher
    retires it (`PoolState.retire`: its record names the job it had started) and moves its epoch; each client takes the
    replies the worker sent before it stopped, fails that one job (STOPPED), and re-sends the others to worker 1 under
    new ids, where they succeed. The successor refuses the old copies, which the clients ignore, as a late answer; they
    route around the worker until it is ready again, then send it jobs under its new epoch."""
    import zmq

    ipc = engine_ipc.ipc_dir()
    workers = EchoWorkers(ipc, 2)
    state = PoolState(SPAWN, 2)
    state.set(0, ready=True)  # worker 1 not ready yet: every job below goes to worker 0

    async def main() -> None:
        a, b = PoolClient(ipc, 2, state, index=0), PoolClient(ipc, 2, state, index=1)  # two frontends
        restarts = []
        a.on_restart = lambda *x: restarts.append(("a", *x))
        b.on_restart = lambda *x: restarts.append(("b", *x))
        try:
            # Not started: replies wait in the sockets, so the dead worker's last answer is unread when its epoch moves
            answered = a.submit(tiny(), b"x")
            await until_async(lambda: a._socks[0].get(zmq.EVENTS) & zmq.POLLIN)
            workers.held[0].set()
            started = a.submit(tiny(), b"x")
            await until_async(lambda: len(workers.jobs[0]) == 2)
            state.start(0, workers.jobs[0][1].jid)  # as the worker records the job it starts
            queued, other = a.submit(tiny(), b"x"), b.submit(tiny(), b"x")
            await until_async(lambda: len(workers.jobs[0]) == 4)
            assert len({j.jid for j in workers.jobs[0]}) == 4  # job ids differ across frontends too
            workers.held[1].set()
            state.set(1, ready=True)
            assert state.retire(0) == workers.jobs[0][1].jid and (state.epoch(0), state.ready(0)) == (1, False)
            a.check_epochs()
            b.check_epochs()
            assert answered.result().error == "" and started.result().error == STOPPED
            assert not queued.done() and not other.done()
            assert (a.failed, a.resent, b.failed, b.resent) == (1, 1, 0, 1)
            assert sorted(restarts) == [("a", 0, 1, 1, 1), ("b", 0, 1, 0, 1)]
            assert a._outstanding == [0, 1] and b._outstanding == [0, 1]
            await until_async(lambda: len(workers.jobs[1]) == 2)
            assert all(j.epoch == 0 and j.jid not in {k.jid for k in workers.jobs[0]} for j in workers.jobs[1])
            a.start()
            b.start()
            workers.release[1].set()
            assert (await asyncio.wait_for(queued, 2)).error == "" and (await asyncio.wait_for(other, 2)).error == ""
            # The successor (epoch 1) refuses the old copies; the clients ignore that, as they do a late answer
            workers.epochs[0] = 1
            workers.release[0].set()
            await asyncio.sleep(0.05)
            assert (a.failed, a.resent, b.failed, b.resent) == (1, 1, 0, 1)
            assert a._outstanding == b._outstanding == [0, 0] and not a._futures and not b._futures
            routed = a.submit(tiny(), b"x")  # worker 0 is not ready: worker 1, under its epoch 0
            assert (await asyncio.wait_for(routed, 2)).error == "" and workers.jobs[1][-1].epoch == 0
            state.set(0, ready=True)
            back = a.submit(tiny(), b"x")  # ready again: worker 0 (a tie), under its new epoch
            assert (await asyncio.wait_for(back, 2)).error == "" and workers.jobs[0][-1].epoch == 1
        finally:
            a.close()
            b.close()

    try:
        asyncio.run(main())
    finally:
        workers.close()
        shutil.rmtree(ipc, ignore_errors=True)


def test_a_refusal_waits_for_the_epoch_and_a_lost_record_fails_everything() -> None:
    """A job that reached its worker's successor before the client saw the new epoch is refused (REFUSED): the client
    keeps it until it sees the epoch (within EPOCH_POLL_S), then re-sends it, since that worker had started nothing.
    When a worker's index stopped twice before a client looked, the record of the first death is gone: every job the
    client sent under that epoch fails (STOPPED), as any of them may have stopped the worker."""
    ipc = engine_ipc.ipc_dir()
    workers = EchoWorkers(ipc, 2)
    state = PoolState(SPAWN, 2)
    state.set(1, ready=True)  # worker 0 not ready: the first job goes to worker 1

    async def main() -> None:
        client = PoolClient(ipc, 2, state)
        client.start()
        try:
            workers.held[1].set()
            refused = client.submit(tiny(), b"x")
            await until_async(lambda: len(workers.jobs[1]) == 1)
            state.set(0, ready=True)
            assert state.retire(1) == 0  # worker 1 died between jobs
            workers.epochs[1] = 1
            workers.release[1].set()  # its successor reads the old job before the client looks
            assert (await asyncio.wait_for(refused, 2)).error == ""  # re-sent to worker 0
            assert (client.resent, client.failed) == (1, 0) and workers.jobs[0][-1].jid != workers.jobs[1][0].jid
            workers.held[0].set()
            state.set(1, ready=False)
            started, queued = client.submit(tiny(), b"x"), client.submit(tiny(), b"x")
            await until_async(lambda: len(workers.jobs[0]) == 3)
            state.start(0, workers.jobs[0][1].jid)
            assert state.retire(0) == workers.jobs[0][1].jid  # epoch 0 ends decoding `started`
            assert state.retire(0) == 0  # its successor dies starting: epoch 1 ends, its record replaces epoch 0's
            client.check_epochs()
            assert started.result().error == queued.result().error == STOPPED and client.failed == 2
        finally:
            client.close()

    try:
        asyncio.run(main())
    finally:
        workers.close()
        shutil.rmtree(ipc, ignore_errors=True)


def test_a_closed_pool_fails_what_waits_for_a_gone_worker() -> None:
    """Once the launcher closes the pool (the server stops: nothing is respawned), a job that waits for a worker that is
    not ready fails at once with CLOSED, a 503 a proxy may retry elsewhere (its image stopped nothing): the job
    queued behind a dying worker's started one, re-sent while no worker was ready and so routed to the dead index before
    the pool closed; a new job; and one re-sent from a worker that died after the close. A ready worker serves on."""
    ipc = engine_ipc.ipc_dir()
    workers = EchoWorkers(ipc, 2)
    state = PoolState(SPAWN, 2)
    state.set(0, ready=True)  # worker 1 is not ready (respawning)

    async def main() -> None:
        client = PoolClient(ipc, 2, state)
        client.start()
        try:
            workers.held[0].set()
            started, queued = client.submit(tiny(), b"x"), client.submit(tiny(), b"x")
            await until_async(lambda: len(workers.jobs[0]) == 2)
            state.start(0, workers.jobs[0][0].jid)
            state.retire(0)  # the death that trips the breaker, before the launcher closes the pool
            client.check_epochs()
            assert started.result().error == STOPPED and not queued.done()  # waits for a respawn that never comes
            state.close()
            client.check_epochs()
            assert queued.result().error == CLOSED and client._outstanding == [0, 0] and not client._futures
            new = client.submit(tiny(), b"x")
            assert new.done() and new.result().error == CLOSED  # at once
            state.set(1, ready=True)  # a worker that was ready before the close serves on
            assert (await asyncio.wait_for(client.submit(tiny(), b"x"), 2)).error == ""
            workers.held[1].set()
            late = client.submit(tiny(), b"x")
            await until_async(lambda: len(workers.jobs[1]) == 2)
            state.retire(1)  # dies between images while the server stops: its job has nowhere to go
            client.check_epochs()
            assert late.result().error == CLOSED and (client.failed, client.resent) == (1, 1)  # `queued` was re-sent
        finally:
            client.close()

    try:
        asyncio.run(main())
    finally:
        workers.close()
        shutil.rmtree(ipc, ignore_errors=True)


def test_an_image_is_re_sent_once(caplog) -> None:
    """A job whose worker stops before decoding it goes to another worker once; if that one stops before decoding it
    too, it fails with UNSERVED (a 503 a proxy may retry elsewhere): no image waits through every respawn of workers
    that keep dying too slowly to trip the breaker (say, a respawn that hangs until POOL_READY_TIMEOUT_S, again and
    again). Job ids stay under the record's tag when the client's count wraps."""
    ipc = engine_ipc.ipc_dir()
    workers = EchoWorkers(ipc, 2)
    state = PoolState(SPAWN, 2)
    state.set(0, ready=True)

    async def main() -> None:
        client = PoolClient(ipc, 2, state, index=5)
        client.start()
        try:
            client._count = _COUNT_MAX - 1
            for held in workers.held:
                held.set()
            job = client.submit(tiny(), b"x")  # worker 0
            await until_async(lambda: len(workers.jobs[0]) == 1)
            state.set(1, ready=True)
            state.retire(0)  # dies before starting it: worker 1 gets it
            client.check_epochs()
            await until_async(lambda: len(workers.jobs[1]) == 1)
            assert not job.done() and client.resent == 1
            state.retire(1)  # so does worker 1: no second re-send
            with caplog.at_level("WARNING", logger="llm_mini_infer.pool"):
                client.check_epochs()
            assert job.result().error == UNSERVED and (client.failed, client.resent) == (0, 1)
            assert ("image pool worker 1 stopped before decoding 1 of this frontend's images that another had already "
                    "lost: they failed (503)") in caplog.text
            assert [workers.jobs[0][0].jid, workers.jobs[1][0].jid] == [_COUNT_MAX << 12 | 5, 1 << 12 | 5]
            assert _COUNT_MAX << 12 | 5 < 1 << 48
        finally:
            client.close()

    try:
        asyncio.run(main())
    finally:
        workers.close()
        shutil.rmtree(ipc, ignore_errors=True)


def test_a_closed_pool_leaves_a_fresh_death_to_the_next_look() -> None:
    """A worker that dies while the closed pool still serves (the frontends finishing) may be retired between a client's
    look at the epochs and its look at the readiness: the index is not ready, but its epoch moved, so its jobs are the
    next look's (`_replaced`: the one it was decoding fails with a 400, the one queued goes to the ready worker), never
    failed as waiting for a gone worker (a 503, which would send the image that may have stopped it elsewhere)."""
    ipc = engine_ipc.ipc_dir()
    workers = EchoWorkers(ipc, 2)
    state = PoolState(SPAWN, 2)
    state.set(0, ready=True)

    async def main() -> None:
        client = PoolClient(ipc, 2, state)
        client.start()
        try:
            workers.held[0].set()
            started, queued = client.submit(tiny(), b"x"), client.submit(tiny(), b"x")  # worker 0: 1 is not ready
            await until_async(lambda: len(workers.jobs[0]) == 2)
            state.start(0, workers.jobs[0][0].jid)
            state.set(1, ready=True)
            state.close()
            state.retire(0)
            client._fail_unserved()  # as the end of a look that read the epochs before the retire
            assert not started.done() and not queued.done()
            client.check_epochs()
            assert started.result().error == STOPPED and (client.failed, client.resent) == (1, 1)
            assert (await asyncio.wait_for(queued, 2)).error == ""
        finally:
            client.close()

    try:
        asyncio.run(main())
    finally:
        workers.close()
        shutil.rmtree(ipc, ignore_errors=True)


def test_a_re_sending_poll_keeps_one_poll() -> None:
    """The poll that re-sends a dead worker's job does not start a second chain of polls: about 10 a second while jobs
    are outstanding, not 10 more for every re-sending poll."""
    ipc = engine_ipc.ipc_dir()
    workers = EchoWorkers(ipc, 2)
    state = PoolState(SPAWN, 2)
    for i in range(2):
        state.set(i, ready=True)

    async def main() -> None:
        client = PoolClient(ipc, 2, state)
        polls, poll = [0], client._on_poll

        def counted() -> None:
            polls[0] += 1
            poll()

        client._on_poll = counted
        client.start()
        try:
            for held in workers.held:
                held.set()
            client.submit(tiny(), b"x")  # worker 0 (a tie)
            await until_async(lambda: len(workers.jobs[0]) == 1)
            state.retire(0)  # the next poll re-sends the job to worker 1
            await until_async(lambda: len(workers.jobs[1]) == 1)
            polls[0] = 0
            await asyncio.sleep(0.55)
            assert polls[0] <= 7, polls[0]  # one chain: 5 or 6; two: about 11
        finally:
            client.close()

    try:
        asyncio.run(main())
    finally:
        workers.close()
        shutil.rmtree(ipc, ignore_errors=True)


def test_a_frontends_image_step_frees_every_range() -> None:
    """A frontend's image step (`api.images.ImageInputs`) on echo workers. A request cancelled while its image decodes
    frees its pixels only once the job resolves. The image a dying worker had started fails its request with a 400 that
    names it, and the image queued behind it is re-sent and succeeds. Once the pool is closed an image with no worker
    left fails with a 503, unless another image of its request is a 400: an image that may have stopped a worker is
    never retried elsewhere. Every pixel range comes back to the arena."""
    from llm_mini_infer.api.images import ImageInputs
    from llm_mini_infer.api.protocol import ApiError

    ipc = engine_ipc.ipc_dir()
    workers = EchoWorkers(ipc, 2)
    state = PoolState(SPAWN, 2)
    state.set(0, ready=True)
    uri = "data:image/jpeg;base64," + base64.b64encode(encoded(pattern(400, 225), "JPEG")).decode()

    async def main() -> None:
        images = ImageInputs(GEMMA70, index=0, num_frontends=1, arena_bytes=8 << 20, pool_dir=ipc, workers=2,
                             max_bytes=1 << 20, fetch_timeout_s=1.0, max_per_request=2, pool_state=state)
        images.start()
        try:
            workers.held[0].set()
            cancelled = await images.prepare([uri])
            task = asyncio.create_task(images.wait(cancelled))
            await until_async(lambda: len(workers.jobs[0]) == 1)
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            held = images.alloc.used
            images.discard(cancelled)
            assert images.alloc.used == held > 0  # worker 0 may still write it
            workers.release[0].set()
            await until_async(lambda: images.alloc.used == 0)
            workers.release[0].clear()
            started = await images.prepare([uri])
            queued = await images.prepare([uri])
            await until_async(lambda: len(workers.jobs[0]) == 3)
            state.start(0, workers.jobs[0][1].jid)
            state.set(1, ready=True)
            state.retire(0)
            with pytest.raises(ApiError) as err:
                await asyncio.wait_for(images.wait(started), 5)
            assert (err.value.status, err.value.message, err.value.param) == (400, f"image: {STOPPED}", "messages")
            images.discard(started)  # as `App.chat` frees a request that never reached the engine
            await asyncio.wait_for(images.wait(queued), 5)  # re-sent to worker 1, which answers at once
            images.discard(queued)
            workers.held[1].set()
            last = await images.prepare([uri])
            both = await images.prepare([uri, uri])
            await until_async(lambda: len(workers.jobs[1]) == 4)  # queued's copy, then 3
            state.start(1, workers.jobs[1][-1].jid)  # worker 1 dies decoding `both`'s second image
            state.close()
            state.retire(1)
            with pytest.raises(ApiError) as err:
                await asyncio.wait_for(images.wait(last), 5)
            assert (err.value.status, err.value.kind) == (503, "ServiceUnavailableError")
            assert err.value.message == f"image: {CLOSED}"
            with pytest.raises(ApiError) as err:  # its first image is CLOSED, its second a 400: the 400 wins
                await asyncio.wait_for(images.wait(both), 5)
            assert (err.value.status, err.value.message) == (400, f"image: {STOPPED}")
            for prep in (last, both):
                images.discard(prep)
            assert images.alloc.used == 0 and images.alloc.free_bytes == images.alloc.size
        finally:
            await images.close()

    try:
        asyncio.run(main())
    finally:
        workers.close()
        shutil.rmtree(ipc, ignore_errors=True)


def test_a_worker_past_the_job_deadline_is_killed_and_only_its_image_fails(tmp_path, monkeypatch) -> None:
    """Real workers through the launcher's ImagePool (Gemma 4's config.json alone), supervised as the launcher does:
    worker 0 decodes an 8,000 x 8,000 image that its EXIF orientation rotates (about a second here, a few tenths on a
    fast runner) past a job deadline shortened to 0.1 s, the wait for its stack to 0.02 s, a small image queued behind
    it. The launcher kills it (its stack to the log first) and respawns it under epoch 1: that image alone fails, the
    record naming it, the queued one is re-sent to worker 1 and its pixels land in its range, worker 1's own job is
    untouched, and the respawned worker 0 serves the next image."""
    ipc = engine_ipc.ipc_dir()
    arena = str(tmp_path / "arena")
    px_arena.create(arena, 4 << 20)
    cfg = config.Config()
    cfg.model.path = str(gemma_dir(tmp_path))
    cfg.images.workers = 2
    cfg.observability.log_level = "warning"
    pool = launcher.ImagePool(cfg, SPAWN, ipc, arena)
    pool.deadline_s = 0.1  # with the stack's wait, well inside the big image's decode: killed before it finishes
    monkeypatch.setattr(launcher, "STACK_WAIT_S", 0.02)
    stop = threading.Event()

    def supervise() -> None:  # the launcher's loop, which looks at least once a second
        while not stop.is_set():
            assert not pool.supervise()
            time.sleep(0.02)

    thread = threading.Thread(target=supervise, daemon=True)
    big = encoded(pattern(8000, 8000, 1), "JPEG", exif=exif(8))
    small = [encoded(pattern(400, 225, seed), "JPEG") for seed in (2, 3, 4)]
    mib = 1 << 20
    try:
        for i, r in enumerate(pool.start()):
            assert r.poll(120) and r.recv() == "ready"
            r.close()
            pool.state.set(i, ready=True)
        thread.start()

        async def main() -> tuple[list[Done], Done, int, int]:
            client = PoolClient(ipc, 2, pool.state)
            client.start()
            try:
                hung = client.submit(job_for(big, 0), big)  # worker 0 (a tie)
                own = client.submit(job_for(small[0], mib), small[0])  # worker 1
                queued = client.submit(job_for(small[1], 2 * mib), small[1])  # worker 0 (a tie), behind the big one
                done = [await asyncio.wait_for(f, 60) for f in (hung, own, queued)]
                await until_async(lambda: pool.state.ready(0) and pool.state.epoch(0) == 1, 60)
                after = await asyncio.wait_for(client.submit(job_for(small[2], 3 * mib), small[2]), 60)  # worker 0
                return done, after, client.failed, client.resent
            finally:
                client.close()

        (hung, own, queued), after, failed, resent = asyncio.run(main())
        assert hung.error == STOPPED and not own.error and not queued.error and not after.error
        assert (failed, resent, pool.state.hangs) == (1, 1, 1)
        assert (pool.state.epoch(0), pool.state.epoch(1), pool.state.lost(0, 0)) == (1, 0, hung.jid)
        view = px_arena.map_readonly(arena)
        mem = np.frombuffer(view.buf, np.uint8)
        for data, off in ((small[1], 2 * mib), (small[2], 3 * mib)):
            h = probe(data)
            size = GEMMA70.size(h)
            n = 3 * size.height * size.width
            assert np.array_equal(mem[off:off + n], GEMMA70.pixels(data, h, size).numpy().reshape(-1))
        del mem
        view.close()
    finally:
        stop.set()
        if thread.is_alive():
            thread.join(10)
        pool.stop()
        px_arena.remove(arena)
        shutil.rmtree(ipc, ignore_errors=True)


class FakeProcess:
    """The spawn context's Process for the launcher's pool, never started: alive until the test kills it (a process
    that died first keeps its exit code, as a SIGKILL to a zombie leaves it; `before_kill` runs first, as what the
    process did up to the signal); its ready pipe kept open (a dup) so the test can report it ready."""

    pids = iter(range(1000, 2000))
    made: ClassVar[list[FakeProcess]] = []

    def __init__(self, *, target, args, kwargs, name):
        self.args, self.kwargs, self.name = args, kwargs, name
        self.pid, self.exitcode, self.killed = next(self.pids), None, False
        self.before_kill = None
        self.made.append(self)

    def start(self) -> None:
        self.ready = Connection(os.dup(self.args[6].fileno()), readable=False)

    def is_alive(self) -> bool:
        return self.exitcode is None

    def die(self, code: int) -> None:
        self.exitcode = code

    def kill(self) -> None:
        if self.before_kill is not None:
            self.before_kill()
        self.killed = True
        if self.exitcode is None:
            self.die(-9)

    def join(self, timeout: float | None = None) -> None:
        pass


class FinishingFrontend:
    """A frontend process alive for two of `launcher._shutdown`'s joins, as one still finishing its requests."""

    name, joins = "llm-mini-infer-frontend-0", 0

    def is_alive(self) -> bool:
        return self.joins < 2

    def join(self, timeout: float | None = None) -> None:
        self.joins += 1


def fake_pool(monkeypatch, workers: int) -> tuple[launcher.ImagePool, list[float], list[int], list[Any]]:
    """The launcher's ImagePool on FakeProcess workers, all ready, with the launcher's clock faked: (the pool, its
    time.monotonic, an offset on its time.monotonic_ns over the real one, which the workers' records use, the
    processes whose stacks it asked for)."""
    now, ns, stacks = [100.0], [0], []
    monkeypatch.setattr(launcher, "time", SimpleNamespace(monotonic=lambda: now[0],
                                                          monotonic_ns=lambda: time.monotonic_ns() + ns[0]))
    monkeypatch.setattr(launcher, "_dump_stack", stacks.append)
    cfg = config.Config()
    cfg.images.workers = workers
    pool = launcher.ImagePool(cfg, SimpleNamespace(Pipe=SPAWN.Pipe, Process=FakeProcess, RawArray=SPAWN.RawArray),
                              "/nonexistent", "/nonexistent-arena")
    for i, r in enumerate(pool.start()):
        r.close()
        pool.state.set(i, ready=True)
    return pool, now, ns, stacks


def test_the_launcher_respawns_dead_workers_until_a_crash_loop(monkeypatch, caplog) -> None:
    """The launcher's ImagePool: a dead worker is retired first (its record written, its epoch moved, so the frontends
    fail or re-send its jobs), then a new worker spawns with that epoch and is ready once it says so. A worker that has
    held one job past the deadline (44.7 s at the pixel limit's default) is killed, its stack asked for first, and
    retired the same way with that job recorded, unless the job ended while its stack was printed (slow, not hung); if
    it started another just before the SIGKILL, the record blames neither. A worker not ready POOL_READY_TIMEOUT_S after
    its respawn is killed and counts as a death; more than POOL_DEATHS deaths within POOL_DEATH_WINDOW_S, or one while
    the frontends fail everything, is a crash loop (True: the exit path), and deaths further apart never are. The
    launcher lets go of the workers it replaced (their pipes' descriptors with them)."""
    pool, now, ns, stacks = fake_pool(monkeypatch, 2)
    assert pool.deadline_s == job_deadline_s(178_956_970) == pytest.approx(44.74, abs=0.01)
    first = list(pool.procs)
    assert not pool.supervise() and pool.procs == first
    with caplog.at_level("INFO", logger="llm_mini_infer.launcher"):
        first[0].die(-11)
        assert not pool.supervise()
    assert ("llm-mini-infer-pool-0 exited unexpectedly (SIGSEGV) between images; its other images are re-sent; "
            "respawning it (epoch 1; death 1 of the 3 the pool takes in 60 s)") in caplog.text
    respawned = pool.procs[0]
    assert respawned is not first[0] and respawned.kwargs["epoch"] == 1 and pool.procs[1] is first[1]
    assert respawned.kwargs["state"] is pool.state
    assert (pool.state.epoch(0), pool.state.ready(0), pool.state.pid(0)) == (1, False, respawned.pid)
    respawned.ready.send("ready")
    now[0] += 2
    assert not pool.supervise() and pool.state.ready(0) and not pool.starting
    # Deaths a window apart never trip the breaker (each respawn reports ready)
    for _ in range(launcher.POOL_DEATHS + 2):
        now[0] += launcher.POOL_DEATH_WINDOW_S + 1
        pool.procs[1].die(1)
        assert not pool.supervise()
        pool.procs[1].ready.send("ready")
        assert not pool.supervise() and pool.state.ready(1)
    epoch = pool.state.epoch(1)
    assert epoch == launcher.POOL_DEATHS + 2
    # A worker past the job deadline: one that finishes its image while its stack is printed lives on
    now[0] += launcher.POOL_DEATH_WINDOW_S + 1
    hung = pool.procs[1]
    pool.state.start(1, 76 << 12)
    ns[0] = int((pool.deadline_s - 1) * 1e9)
    assert not pool.supervise() and not stacks and not hung.killed  # within the deadline
    ns[0] = int((pool.deadline_s + 1) * 1e9)
    monkeypatch.setattr(launcher, "_dump_stack", lambda p: (stacks.append(p), pool.state.finish(1)))
    with caplog.at_level("WARNING", logger="llm_mini_infer.launcher"):
        assert not pool.supervise()
    assert stacks == [hung] and not hung.killed and pool.state.hangs == 0
    assert "image pool worker 1 finished that image after all: not killed" in caplog.text
    # A hung one is killed, its stack first, and retired with its job recorded
    monkeypatch.setattr(launcher, "_dump_stack", stacks.append)
    pool.state.start(1, 77 << 12)
    with caplog.at_level("ERROR", logger="llm_mini_infer.launcher"):
        assert not pool.supervise()
    assert stacks == [hung, hung] and hung.killed and pool.procs[1] is not hung and pool.state.hangs == 1
    assert (pool.state.lost(1, epoch), pool.state.epoch(1), pool.state.since(1)) == (77 << 12, epoch + 1, 0)
    assert "image pool worker 1 has held one image for 45.7 s, past the job deadline (45 s): killing it" in caplog.text
    assert (f"llm-mini-infer-pool-1 was killed at the job deadline (45 s) while decoding an image, which fails with a "
            f"400; its other images are re-sent; respawning it (epoch {epoch + 1}; death 1 of the 3") in caplog.text
    pool.procs[1].ready.send("ready")
    assert not pool.supervise() and pool.state.ready(1)
    # One that finishes it just before the SIGKILL and starts the next: neither is blamed (both are re-sent)
    now[0] += launcher.POOL_DEATH_WINDOW_S + 1
    racer = pool.procs[1]
    pool.state.start(1, 78 << 12)
    racer.before_kill = lambda: (pool.state.finish(1), pool.state.start(1, 79 << 12))
    with caplog.at_level("ERROR", logger="llm_mini_infer.launcher"):
        assert not pool.supervise()
    assert racer.killed and pool.state.lost(1, epoch + 1) == 0 and pool.state.hangs == 2
    assert (f"llm-mini-infer-pool-1 was killed at the job deadline (45 s) between images; its other images "
            f"are re-sent; respawning it (epoch {epoch + 2}") in caplog.text
    pool.procs[1].ready.send("ready")
    assert not pool.supervise() and pool.state.ready(1)
    assert {hung, racer}.isdisjoint(w.proc for w in pool.workers)  # let go once replaced: no record holds them
    ns[0] = 0
    # A respawn that never reports ready is killed after its timeout, and that counts as a death; a respawn that died
    # starting takes its pipe with it
    now[0] += launcher.POOL_DEATH_WINDOW_S + 1
    pool.procs[0].die(1)
    assert not pool.supervise()
    starting = pool.procs[0]
    now[0] += launcher.POOL_READY_TIMEOUT_S + 1
    assert not pool.supervise()
    assert starting.killed and pool.procs[0] is not starting and pool.state.epoch(0) == 3
    assert [i for i, _, _ in pool.starting.values()] == [0] and not pool.procs[1].killed
    # A crash loop: one death more than POOL_DEATHS within the window (the timeout's is the first)
    assert len(pool.deaths) == 1
    for k in range(launcher.POOL_DEATHS - 1):
        pool.procs[k % 2].die(1)
        assert not pool.supervise()
    pool.procs[1].die(1)
    with caplog.at_level("ERROR", logger="llm_mini_infer.launcher"):
        assert pool.supervise()
    assert (f"llm-mini-infer-pool-1 exited unexpectedly (code 1) between images: {launcher.POOL_DEATHS + 1} image pool "
            "worker deaths in 60 s, a crash loop; failing in-flight requests and stopping") in caplog.text
    # While the frontends fail everything (a second stop signal), any death takes the exit path
    calm, _, _, _ = fake_pool(monkeypatch, 2)
    calm.procs[0].die(1)
    assert calm.supervise(failing=True) and calm.state.epoch(0) == 1
    for p in (pool, calm):
        p.stop()
    for p in FakeProcess.made:
        p.ready.close()


def test_a_pool_workers_phase_takes_its_edges_and_no_other(monkeypatch) -> None:
    """A worker's record (`launcher.Worker`): STARTING to READY when it reports, to RETIRED when `reap` finds it dead,
    ready or not; a respawn is a new record. A retired one goes nowhere: `Lifecycle.to` refuses, and the supervision's
    calls check the phase first, so a late ready report changes nothing."""
    pool, _, _, _ = fake_pool(monkeypatch, 3)
    w = pool.workers
    assert [x.life.phase for x in w] == [launcher.Phase.STARTING] * 3  # `fake_pool` marks the state only
    pool.mark_ready(0)
    assert w[0].life.phase is launcher.Phase.READY and pool.state.ready(0)
    w[0].proc.die(1)  # READY to RETIRED
    w[1].proc.die(1)  # STARTING to RETIRED
    first = list(w)
    assert not pool.supervise()
    assert [x.life.phase for x in first[:2]] == [launcher.Phase.RETIRED] * 2
    assert pool.workers[0] is not first[0] and pool.workers[0].life.phase is launcher.Phase.STARTING  # a respawn
    with pytest.raises(RuntimeError, match="an image pool worker: no edge from RETIRED to READY"):
        first[0].life.to(launcher.Phase.READY)
    pool.workers[0] = first[0]  # a late ready report for a retired record: no edge taken
    pool.mark_ready(0)
    assert first[0].life.phase is launcher.Phase.RETIRED
    pool.stop()
    for p in FakeProcess.made:
        p.ready.close()


def test_a_respawn_killed_for_not_getting_ready_is_logged_as_such(monkeypatch, caplog) -> None:
    """A respawned worker not ready POOL_READY_TIMEOUT_S after its respawn is killed: its log line says why, not "exited
    unexpectedly", it blames no image, and the breaker counts it."""
    made = len(FakeProcess.made)
    pool, now, _, _ = fake_pool(monkeypatch, 1)
    pool.procs[0].die(1)
    assert not pool.supervise()
    starting = pool.procs[0]
    now[0] += launcher.POOL_READY_TIMEOUT_S + 1
    caplog.clear()
    with caplog.at_level("ERROR", logger="llm_mini_infer.launcher"):
        assert not pool.supervise()
    assert starting.killed and list(pool.deaths) == [now[0]] and pool.state.epoch(0) == 2  # the first death aged out
    assert ("llm-mini-infer-pool-0 was killed, not ready 120 s after its respawn, between images; its other images are "
            "re-sent; respawning it (epoch 2; death 1 of the 3 the pool takes in 60 s)") in caplog.text
    assert "exited unexpectedly" not in caplog.text
    pool.stop()
    for p in FakeProcess.made[made:]:
        p.ready.close()


def test_a_crash_loop_fails_every_dead_workers_jobs(monkeypatch) -> None:
    """Workers that die in the turn the breaker trips are all retired first (their jobs fail or are re-sent in every
    frontend instead of hanging until the frontends are killed, then reaching the client as a reset), none is
    respawned, and each death is reaped once; a death after the trip, while the frontends finish
    (`launcher._shutdown`), is still retired."""
    pools = []
    for failing in (False, True):
        pool, _, _, _ = fake_pool(monkeypatch, 3)
        pools.append(pool)
        if not failing:
            for _ in range(launcher.POOL_DEATHS):  # worker 2 dies three times and comes back each time
                pool.procs[2].die(-11)
                assert not pool.supervise()
        made = len(FakeProcess.made)
        first = pool.procs[:2]
        for p in first:
            p.die(-9)
        assert pool.supervise(failing=failing)
        assert [(pool.state.epoch(i), pool.state.ready(i)) for i in range(2)] == [(1, False), (1, False)]
        assert pool.procs[:2] == first and len(FakeProcess.made) == made and pool.reap() == []
    pool = pools[0]
    pool.procs[2].die(1)
    epoch = pool.state.epoch(2)
    launcher._shutdown(config.Config(), [FinishingFrontend()], None, "/nonexistent", pool)
    assert pool.state.epoch(2) == epoch + 1 and not pool.state.ready(2)
    for p in pools:
        p.stop()


def test_closing_the_pool_stops_the_workers_still_starting(monkeypatch, caplog) -> None:
    """On any exit (`serve`'s finally), before the frontends finish, the launcher closes the pool: a worker still
    starting is killed and retired without blame (it may have read a job, routed to it while no worker was ready; that
    job is re-sent or fails with a 503, never a 400), unless it crashed on its own just before; a dead one is retired
    with the job it was decoding recorded; a ready one serves on; and the pool is marked closed. While the frontends
    finish, a worker past the deadline is still killed, and nothing is respawned."""
    pool, _, ns, stacks = fake_pool(monkeypatch, 4)
    for i in (1, 3):
        pool.procs[i].die(-11)
    assert not pool.supervise()
    starting, crashed = pool.procs[1], pool.procs[3]  # respawned under epoch 1, not ready yet
    for i, jid in ((1, 5 << 12), (2, 6 << 12), (3, 9 << 12)):
        pool.state.start(i, jid)
    pool.procs[2].die(-11)  # dies decoding as the server stops
    crashed.die(-11)  # so does a worker still starting, which read a job before the launcher marked it ready
    with caplog.at_level("INFO", logger="llm_mini_infer.launcher"):
        pool.close()
    assert starting.killed and crashed.killed and pool.state.closed and not pool.starting
    assert [pool.state.lost(i, e) for i, e in ((1, 1), (2, 0), (3, 1))] == [0, 6 << 12, 9 << 12]
    assert [(pool.state.epoch(i), pool.state.ready(i)) for i in range(4)] == [(0, True), (2, False), (1, False),
                                                                              (2, False)]
    assert "llm-mini-infer-pool-1 was stopped while starting: the server is stopping" in caplog.text
    for i in (2, 3):
        assert (f"llm-mini-infer-pool-{i} exited unexpectedly (SIGSEGV) while decoding an image, "
                "which fails with a 400; the server is stopping, so it is not respawned") in caplog.text
    made = len(FakeProcess.made)
    pool.state.start(0, 7 << 12)
    ns[0] = int((pool.deadline_s + 1) * 1e9)
    launcher._shutdown(config.Config(), [FinishingFrontend()], None, "/nonexistent", pool)
    assert stacks == [pool.procs[0]] and pool.procs[0].killed and pool.state.lost(0, 0) == 7 << 12
    assert len(FakeProcess.made) == made and pool.state.hangs == 1 and not pool.state.ready(0)
    pool.stop()


# -- through the real server ------------------------------------------------------------------------------------------

needs_qwen = pytest.mark.skipif(not env_has_model(QWEN), reason="tests/.fixtures/models/qwen3.8-27b is absent: "
                                                                "uv run tests/scripts/fetch_fixtures.py")
PIXELS = {"min_pixels": 65536, "max_pixels": 327680}


def chat_with_image(seed: int, data: bytes | None = None) -> dict:
    data = data or encoded(pattern(400, 225, seed), "JPEG")
    url = "data:image/jpeg;base64," + base64.b64encode(data).decode()
    return {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                                      {"type": "text", "text": "What is it?"}]}],
            "max_tokens": 4, "seed": seed}


def until(cond, timeout: float = 60.0, poll: float = 0.05) -> None:
    deadline = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(poll)


def debug_images(srv: Server) -> dict:
    return json.loads(srv.get("/debug/images")[1])


def resume(pids: list[int]) -> None:
    """SIGCONT the workers a test stopped: one left stopped outlives a server that a failed test had to kill."""
    for pid in pids:
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGCONT)


@needs_qwen
@pytest.mark.serial  # a server with an image pool
@pytest.mark.parametrize("how", ["crash", "hang"])
def test_a_worker_that_stops_mid_image_fails_only_that_image(tmp_path, how) -> None:
    """Worker 0 stopped (SIGSTOP) in the middle of request A's image (4,800 x 3,600, rotated: a few tenths of a second);
    request B's image waits on worker 1, stopped too, so that request C's image queues behind A's on worker 0; worker 1
    resumes and answers B. Then worker 0 is killed (a crash), or the launcher kills it at the job deadline (a hang; 5 s
    at a pixel limit of 20 million): only A fails, with a 400 that names the stopped worker (a proxy retries only 503s
    and resets), C is re-sent to worker 1 and answers 200, and the respawned worker 0 (epoch 1) serves the next
    image. Neither the engine nor the server restarts, the arena comes back whole, the metrics count it, and the drain
    exits 0."""
    srv = Server(QWEN, tmp_path, extra={"images": {**PIXELS, "workers": 2, "max_source_pixels": 20_000_000},
                                         "observability": {"log_level": "info"}}).start(120)
    results: dict[str, tuple[int, Any]] = {}
    stopped: list[int] = []

    def post(name: str, body: dict) -> threading.Thread:
        t = threading.Thread(target=lambda: results.update({name: srv.post("/v1/chat/completions", body)}))
        t.start()
        return t

    try:
        before = debug_images(srv)
        engine = json.loads(srv.get("/debug/engine")[1])["engine_pid"]
        assert before["pool_ready"] == [1, 1] and before["pool_epochs"] == [0, 0] and all(before["pool_pids"])
        w0, w1 = before["pool_pids"]
        stopped += [w0, w1]
        os.kill(w1, signal.SIGSTOP)
        a = post("a", chat_with_image(1, encoded(pattern(4800, 3600, 1), "JPEG", exif=exif(8))))  # worker 0: a tie
        until(lambda: debug_images(srv)["pool_busy"] == [1, 0], poll=0.002)
        os.kill(w0, signal.SIGSTOP)
        b = post("b", chat_with_image(2))  # worker 1
        until(lambda: debug_images(srv)["pool_outstanding"] == [1, 1])
        c = post("c", chat_with_image(3))  # worker 0 (a tie), behind A's image
        until(lambda: debug_images(srv)["pool_outstanding"] == [2, 1])
        os.kill(w1, signal.SIGCONT)
        b.join(30)
        assert results["b"][0] == 200, results["b"]
        if how == "crash":
            os.kill(w0, signal.SIGKILL)
        a.join(60)
        c.join(60)
        status, out = results["a"]
        assert status == 400, out
        assert (out["error"]["type"], out["error"]["param"]) == ("BadRequestError", "messages")
        assert out["error"]["message"] == f"image: {STOPPED}"
        assert results["c"][0] == 200, results["c"]
        until(lambda: (d := debug_images(srv))["pool_ready"] == [1, 1] and d["pool_epochs"] == [1, 0])
        after = debug_images(srv)
        assert after["pool_pids"][0] != w0 and after["pool_pids"][1] == w1
        assert (after["pool_failed"], after["pool_resent"], after["pool_hangs"]) == (1, 1, int(how == "hang"))
        status, out = srv.post("/v1/chat/completions", chat_with_image(4))  # worker 0, respawned: a tie, ready
        assert status == 200, out
        assert srv.metric("llm_mini_infer:image_pool_restarts_total") == 1
        assert srv.metric("llm_mini_infer:image_pool_hangs_total") == int(how == "hang")
        assert srv.metric("llm_mini_infer:image_pool_jobs_failed_total") == 1
        assert srv.metric("llm_mini_infer:image_pool_jobs_resent_total") == 1
        until(lambda: (d := debug_images(srv))["arena_used"] == 0 and d["owned"] == 0)
        assert json.loads(srv.get("/debug/engine")[1])["engine_pid"] == engine
    finally:
        resume(stopped)
        code = srv.stop()
    log = srv.log_path.read_text()
    assert code == 0, log[-3000:]
    assert "image pool: 2 workers ready (pids " in log and "MiB; job deadline 5 s)" in log
    ended = "was killed at the job deadline (5 s)" if how == "hang" else "exited unexpectedly (SIGKILL)"
    assert (f"llm-mini-infer-pool-0 {ended} while decoding an image, which fails with a 400; "
            "its other images are re-sent; respawning it (epoch 1") in log
    assert ("image pool worker 0 has held one image for" in log) == (how == "hang")
    assert "image pool worker 0 ready again" in log and "crash loop" not in log
    assert ("image pool worker 0 stopped (epoch 1 next): 1 of this frontend's images failed with it, 1 "
            "re-sent") in log


@needs_qwen
@pytest.mark.serial  # a server with an image pool
def test_a_crash_loop_of_pool_workers_stops_the_server(tmp_path) -> None:
    """The breaker: the pool respawns POOL_DEATHS deaths within POOL_DEATH_WINDOW_S; one more takes the exit path
    (the frontends fail their in-flight requests, and the server exits 1 for its restart policy). A request whose image
    waits on the worker that dies last (stopped, so it never started that image) fails at once with a 503, which a
    proxy may retry elsewhere: its image is re-sent to a pool the launcher has closed, with no worker left, instead of
    waiting for the drain's deadline (35 s here)."""
    srv = Server(QWEN, tmp_path,
                 extra={"images": {**PIXELS, "workers": 1}, "observability": {"log_level": "info"}}).start(120)
    result: dict[str, Any] = {}
    stopped: list[int] = []
    try:
        for k in range(launcher.POOL_DEATHS):
            pid = debug_images(srv)["pool_pids"][0]
            os.kill(pid, signal.SIGKILL)
            until(lambda pid=pid, k=k: (d := debug_images(srv))["pool_ready"] == [1]
                  and d["pool_epochs"] == [k + 1] and d["pool_pids"][0] != pid)
        status, out = srv.post("/v1/chat/completions", chat_with_image(5))  # the last respawn serves
        assert status == 200, out
        pid = debug_images(srv)["pool_pids"][0]
        stopped.append(pid)
        os.kill(pid, signal.SIGSTOP)
        t = threading.Thread(target=lambda: result.update(waiting=srv.post("/v1/chat/completions", chat_with_image(6))))
        t.start()
        until(lambda: debug_images(srv)["pool_outstanding"] == [1])
        t0 = time.monotonic()
        os.kill(pid, signal.SIGKILL)
        t.join(60)
        result["s"] = time.monotonic() - t0
        assert srv.proc.wait(60) == 1
    finally:
        resume(stopped)
        srv.stop()
    log = srv.log_path.read_text()
    status, out = result["waiting"]
    assert (status, out["error"]["type"], out["error"]["message"]) == (503, "ServiceUnavailableError",
                                                                       f"image: {CLOSED}"), out
    assert result["s"] < 5, result["s"]
    assert (f"llm-mini-infer-pool-0 exited unexpectedly (SIGKILL) between images: {launcher.POOL_DEATHS + 1} image "
            "pool worker deaths in 60 s, a crash loop") in log, log[-3000:]
