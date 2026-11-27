"""Frontend process entry point: the ASGI app inside uvicorn, running on this process's own event loop.

uvicorn runs in-process (no worker processes of its own), so each frontend process is one HTTP server with one engine
link: `uvicorn.Server` with httptools on our uvloop, serving a socket we bind with SO_REUSEPORT. The server's own signal
handling is disabled; the launcher's signals are ours (`launcher`): SIGTERM drains with `server.drain_grace_s`
(`App.begin_drain`), SIGINT drains without it, SIGUSR1 fails the in-flight requests and stops at once
(`App.stop_now`). A frontend whose launcher is gone stops the same way.
"""

from __future__ import annotations

import asyncio
import contextlib
import faulthandler
import logging
import os
import signal
import socket
from multiprocessing.connection import Connection
from typing import Any

import uvloop

from ..config import Config
from ..logutil import setup_logging
from .app import App

log = logging.getLogger("llm_mini_infer.frontend")


def make_socket(host: str, port: int) -> socket.socket:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if hasattr(socket, "SO_REUSEPORT"):
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    sock.bind((host, port))
    sock.listen(2048)
    sock.setblocking(False)
    return sock


class UvicornServer:
    def __init__(self, app: App, host: str, port: int, log_level: str, keep_alive_s: float = 5.0):
        import uvicorn

        class _Server(uvicorn.Server):
            @contextlib.contextmanager
            def capture_signals(self):  # SIGTERM and SIGINT are ours (`_main`)
                yield

        self.sock = make_socket(host, port)
        self.server = _Server(uvicorn.Config(app, http="httptools", loop="none", lifespan="on", access_log=False,
                                             log_level=max(log_level, "warning", key=_level), log_config=None,
                                             timeout_keep_alive=keep_alive_s))

    async def serve(self) -> None:
        await self.server.serve(sockets=[self.sock])

    def stop(self) -> None:
        self.server.should_exit = True


def _level(name: str) -> int:
    return logging.getLevelName(name.upper()) if isinstance(name, str) else int(name)


async def _main(cfg: Config, index: int, ipc_path: str, ready: Connection | None, pool_state: Any = None) -> None:
    app = App(cfg, index, ipc_path, pool_state)
    server = UvicornServer(app, cfg.server.host, cfg.server.port, cfg.observability.log_level, cfg.server.keep_alive_s)

    def on_ready() -> None:
        log.info("frontend %d serving on %s:%d", index, cfg.server.host, cfg.server.port)
        if ready is not None:
            ready.send("ready")
            ready.close()

    app.on_ready = on_ready
    app.on_drained = server.stop
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, app.begin_drain, cfg.server.drain_grace_s)
    loop.add_signal_handler(signal.SIGINT, app.begin_drain, 0.0)
    loop.add_signal_handler(signal.SIGUSR1, app.stop_now)
    parent = os.getppid()

    def watch_parent() -> None:  # an orphaned frontend (the launcher killed) stops instead of serving on its own
        if os.getppid() != parent:
            log.error("the launcher is gone: stopping")
            app.stop_now()
        else:
            loop.call_later(1.0, watch_parent)

    loop.call_later(1.0, watch_parent)
    await server.serve()


def frontend_main(cfg: Config, index: int, ipc_path: str, ready: Connection | None, pool_state: Any = None) -> None:
    """Process entry point (spawned by the launcher; `pool_state`: the image pool's `vision.pool.PoolState`)."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # until `_main` sets its handler: the launcher handles a Ctrl-C
    faulthandler.enable()  # a native crash prints every thread's stack
    faulthandler.register(signal.SIGUSR2, all_threads=True)  # and so does the launcher, for a start that hangs
    # queued: a stderr that stops reading blocks the log's writer thread, not the event loop every stream waits on
    writer = setup_logging(cfg.observability.log_level, f"frontend-{index}", without_messages=True, queued=True)
    cpus = cfg.dev.server.cpus
    if cpus and index < len(cpus):
        try:
            allowed = os.sched_getaffinity(0)
            os.sched_setaffinity(0, {cpus[index]})  # this thread (Linux): the event loop's
            log.info("frontend %d pinned to CPU %d", index, cpus[index])
            # the log's writer thread, started before, on the cores the engine and the frontends leave (the pool's):
            # on the frontend's own core it would take turns with the loop and its ZMQ threads on every request
            rest = allowed - set(cpus) - {cfg.dev.engine.cpu}
            if writer is not None and rest:
                writer.place(rest)
        except (AttributeError, OSError) as exc:
            log.warning("could not pin frontend %d to CPU %d: %s", index, cpus[index], exc)
    import gc

    gc.collect()
    gc.freeze()
    uvloop.run(_main(cfg, index, ipc_path, ready, pool_state))
