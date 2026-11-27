"""The frontends' queued log (`logutil.setup_logging(queued=True)`, `logutil.Queued`): a stderr that stops reading
holds the log's writer thread, never the caller (a frontend's event loop); the queue is bounded, a line past it is
dropped and counted and the count logged in its place; the lines stay whole and in order, and a process that exits
normally writes every line it queued."""

from __future__ import annotations

import io
import json
import logging
import re
import subprocess
import sys
import threading
import time

from llm_mini_infer import logutil
from llm_mini_infer.logutil import LOG_QUEUE_LINES, Queued, setup_logging

LINE = re.compile(r"INFO \d\d-\d\d \d\d:\d\d:\d\d\.\d{3} \[([\w-]+)\] line (\d+) x+\n")


class Gate(io.TextIOBase):
    """A stderr that takes nothing until it is opened."""

    def __init__(self) -> None:
        self.opened = threading.Event()
        self.lines: list[str] = []

    def write(self, s: str) -> int:
        self.opened.wait()
        self.lines.append(s)
        return len(s)

    def flush(self) -> None:
        pass


def test_a_blocked_stderr_holds_the_writer_and_the_lines_past_the_bound_are_dropped_and_counted(monkeypatch) -> None:
    """With stderr blocked, logging returns at once, at most `LOG_QUEUE_LINES` lines wait (memory is bounded), and
    the rest are dropped and counted; once stderr takes lines again, the next line is preceded by one WARNING with the
    count; the order holds, and the writer's stop (the process's exit) writes every line queued."""
    root = logging.getLogger()
    saved = root.handlers[:], root.level
    gate = Gate()
    monkeypatch.setattr(sys, "stderr", gate)
    log = logging.getLogger("test.log_queue")
    pad = "x" * 40
    writer = None
    try:
        writer = setup_logging("info", "frontend-0", queued=True)
        front = root.handlers[0]
        assert isinstance(front, Queued) and writer is not None
        log.info("line %d %s", 0, pad)
        deadline = time.monotonic() + 5
        while front.queue.qsize() and time.monotonic() < deadline:  # the writer took it and waits on stderr
            time.sleep(0.001)
        extra = 500
        for k in range(1, LOG_QUEUE_LINES + extra + 1):
            log.info("line %d %s", k, pad)
        assert not gate.opened.is_set() and not gate.lines  # every call returned with stderr shut: none waited on it
        assert front.queue.qsize() == LOG_QUEUE_LINES and front.dropped == extra
        gate.opened.set()
        deadline = time.monotonic() + 10
        while front.queue.qsize() and time.monotonic() < deadline:
            time.sleep(0.001)
        log.info("line %d %s", LOG_QUEUE_LINES + extra + 1, pad)
    finally:
        gate.opened.set()  # a failure never leaves the writer blocked at the test process's exit
        if writer is not None:
            writer.stop()
        root.handlers[:], root.level = saved
    got = [LINE.fullmatch(x) for x in gate.lines]
    note = [i for i, m in enumerate(got) if m is None]
    assert len(note) == 1 and note[0] == LOG_QUEUE_LINES + 1  # after the lines that waited, before the next one
    assert re.fullmatch(rf"WARNING .* \[frontend-0\] {extra} log lines were dropped: stderr took none while "
                        rf"{LOG_QUEUE_LINES} lines were queued\n", gate.lines[note[0]])
    seq = [int(m[2]) for m in got if m is not None]
    assert seq == [*range(LOG_QUEUE_LINES + 1), LOG_QUEUE_LINES + extra + 1]


CHILD = r"""
import asyncio, json, logging, sys, time
import uvloop
from llm_mini_infer.logutil import setup_logging

name, n = sys.argv[1], int(sys.argv[2])
setup_logging("info", name, queued=True)
log = logging.getLogger("child")

async def main():
    loop = asyncio.get_running_loop()
    worst, due = 0, time.monotonic_ns() + 1_000_000
    done = loop.create_future()
    state = {"k": 0}

    def tick():
        nonlocal worst, due
        now = time.monotonic_ns()
        worst, due = max(worst, now - due), now + 1_000_000
        for _ in range(5):  # five lines a millisecond, from the loop
            if state["k"] < n:
                log.info("line %d %s", state["k"], "x" * 400)
                state["k"] += 1
        if state["k"] < n:
            loop.call_later(0.001, tick)
        else:
            done.set_result(None)

    loop.call_later(0.001, tick)
    await done
    print(json.dumps({"worst_lag_ms": worst / 1e6}), flush=True)

uvloop.run(main())
"""


def test_a_frontends_loop_runs_on_while_stderr_takes_nothing_and_every_line_arrives_whole(tmp_path) -> None:
    """Two processes log from their event loops through one pipe that nobody reads for two seconds (a stderr that
    stopped reading, as a frontend's may): neither loop waits on it (a direct handler would block it once the pipe's
    64 KiB were full, for the whole two seconds), and once the pipe is read, every line of each arrives whole and in its
    order, all of them before the process exits (fewer than `LOG_QUEUE_LINES`: none dropped). A line is one write, as
    a direct handler's is, so lines from several processes stay whole up to the pipe's atomic write (`PIPE_BUF`: 512
    bytes at least, 4,096 on Linux)."""
    n = 3000
    script = tmp_path / "child.py"
    script.write_text(CHILD)
    import os

    r, w = os.pipe()
    procs = [subprocess.Popen([sys.executable, str(script), f"p{k}", str(n)], stdout=subprocess.PIPE, stderr=w,
                              text=True) for k in range(2)]
    os.close(w)
    try:
        time.sleep(2.0)  # the pipe fills; the children's loops log on
        data = bytearray()
        with os.fdopen(r, "rb") as f:
            while chunk := f.read(1 << 16):
                data += chunk
        reports = [json.loads(p.communicate(timeout=60)[0]) for p in procs]
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
    assert all(p.returncode == 0 for p in procs)
    assert all(rep["worst_lag_ms"] < 50 for rep in reports), reports
    lines = data.decode().splitlines(keepends=True)
    for k in range(2):
        mine = [LINE.fullmatch(x) for x in lines if f"[p{k}]" in x]
        assert all(m is not None and m[0].endswith(" " + "x" * 400 + "\n") for m in mine)  # whole: none cut or mixed
        assert [int(m[2]) for m in mine] == list(range(n))  # in order, none lost


def test_only_the_frontends_queue_their_log() -> None:
    """The frontends queue (they serve the streams); the launcher, the engine and the pool's workers write directly:
    the launcher exits by `os._exit` on a signal and the workers are killed at a stop, so a queue would lose their last
    lines."""
    import inspect

    from llm_mini_infer import launcher
    from llm_mini_infer.api import server
    from llm_mini_infer.engine import process
    from llm_mini_infer.vision import pool

    assert "queued=True" in inspect.getsource(server.frontend_main)
    for mod in (launcher, process, pool):
        assert "queued=True" not in inspect.getsource(mod)


def test_lines_dropped_with_nothing_logged_afterwards_are_counted_at_the_writers_stop(monkeypatch) -> None:
    """The count of dropped lines is written even when nothing is logged after them: the stop queues a note for the
    lines dropped since the last one, after the lines that waited and before it ends."""
    root = logging.getLogger()
    saved = root.handlers[:], root.level
    gate = Gate()
    monkeypatch.setattr(sys, "stderr", gate)
    log = logging.getLogger("test.log_queue")
    pad = "x" * 40
    extra = 300
    writer = None
    try:
        writer = setup_logging("info", "frontend-0", queued=True)
        front = root.handlers[0]
        assert isinstance(front, Queued) and writer is not None
        log.info("line %d %s", 0, pad)
        deadline = time.monotonic() + 5
        while front.queue.qsize() and time.monotonic() < deadline:
            time.sleep(0.001)
        for k in range(1, LOG_QUEUE_LINES + extra + 1):
            log.info("line %d %s", k, pad)
        assert front.dropped == extra
        gate.opened.set()
        writer.stop()
        assert front.dropped == 0 and writer._thread is None
        writer.stop()  # idempotent
    finally:
        gate.opened.set()
        if writer is not None:
            writer.stop()
        root.handlers[:], root.level = saved
    assert re.fullmatch(rf"WARNING .* \[frontend-0\] {extra} log lines were dropped: stderr took none while "
                        rf"{LOG_QUEUE_LINES} lines were queued\n", gate.lines[-1])
    got = [LINE.fullmatch(x) for x in gate.lines[:-1]]
    assert all(m is not None for m in got)
    assert [int(m[2]) for m in got] == list(range(LOG_QUEUE_LINES + 1))  # the lines before it, in order


def test_a_stderr_that_takes_nothing_does_not_hold_the_writers_stop_for_good(monkeypatch) -> None:
    """With stderr blocked for good, the stop gives up after `LOG_STOP_S` (here 0.3 s) and returns, the process exiting
    without the lines instead of hanging."""
    root = logging.getLogger()
    saved = root.handlers[:], root.level
    gate = Gate()
    monkeypatch.setattr(sys, "stderr", gate)
    monkeypatch.setattr(logutil, "LOG_STOP_S", 0.3)
    log = logging.getLogger("test.log_queue")
    writer = None
    try:
        writer = setup_logging("info", "frontend-0", queued=True)
        assert writer is not None
        for k in range(5):
            log.info("line %d", k)
        t = time.monotonic()
        writer.stop()
        assert time.monotonic() - t < 1.0
        assert gate.lines == []
    finally:
        gate.opened.set()
        if writer is not None and writer._thread is not None:
            writer._thread.join(5)
        root.handlers[:], root.level = saved
