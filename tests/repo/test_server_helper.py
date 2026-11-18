"""The test servers' helper (tests/support/api_util.py) leaves nothing running: its process handling, driven with a
stand-in launcher that starts a child as a launcher starts its engine. No model, no server; any OS."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import psutil
import pytest

from llm_mini_infer.vision import arena

from ..support import api_util
from ..support.api_util import Server
from ..support.paths import ROOT

# The stand-in for llm_mini_infer's launcher (the stale-file sweep kills only a command line that names the package):
# `how` (argv[1], "+"-joined) says how it starts and stops. Everything it starts ends by itself within 60 s, so that a
# failing test leaves nothing behind for long. Each SIGTERM it or its child gets is a line in the log.
LAUNCHER = """# a stand-in for llm_mini_infer's launcher
import http.server, os, signal, subprocess, sys, threading, time

CHILD = '''import os, signal, time
signal.signal(signal.SIGTERM, lambda *_: print("SIGTERM to the child", flush=True))  # and it sleeps on
print("armed", os.getpid(), flush=True)
time.sleep(60)
'''
signal.alarm(60)
how, port = sys.argv[1].split("+"), int(sys.argv[2])
child = subprocess.Popen([sys.executable, "-c", CHILD])  # in its group, as an engine is
print("child", child.pid, flush=True)
if "arena" in how:  # what a launcher with images makes, as it makes them (launcher._start_images)
    from llm_mini_infer.engine import ipc
    from llm_mini_infer.vision import arena

    ipc_path = ipc.ipc_dir()
    pixels = arena.arena_path(arena.default_dir(ipc_path), f"{os.getpid()}-{os.path.basename(ipc_path)}")
    arena.create(pixels, 4096)
    print("arena", ipc_path, pixels, flush=True)
if "exit" in how:
    sys.exit(3)


def stop(*_):
    print("SIGTERM to the launcher", flush=True)
    os._exit(0)  # its child, its arena and IPC directory left


signal.signal(signal.SIGTERM, signal.SIG_IGN if "stubborn" in how else stop)
if "never" not in how:
    class Health(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    threading.Thread(target=http.server.HTTPServer(("127.0.0.1", port), Health).serve_forever, daemon=True).start()
time.sleep(60)
"""

# A pytest process that launches a stand-in server and then, as stdin says, exits without its exit hooks (os._exit, as
# a timeout's exit does) or with them (sys.exit), or waits there to be SIGKILLed
OWNER = """
import os, signal, sys
from pathlib import Path

from tests.repo.test_server_helper import StandIn

signal.alarm(60)
srv = StandIn(Path(sys.argv[1]), "serve").start(10)
print(srv.pgid, flush=True)
if sys.stdin.readline().strip() == "exit":
    os._exit(0)
sys.exit(0)
"""


class StandIn(Server):
    """`Server` with the stand-in launcher, through the same launch, registry, start and stop; 2 s to stop."""

    def __init__(self, tmp: Path, how: str):
        super().__init__(tmp / "model", tmp)
        self.how = how

    def command(self, env: dict[str, str]) -> list[str]:
        return [sys.executable, "-c", LAUNCHER, self.how, str(self.port)]

    def stop(self, sig: int = signal.SIGTERM, timeout: float = 2) -> int:
        return super().stop(sig, timeout)

    def logged(self, what: str) -> list[str]:
        """The fields of the stand-in's `what` line, which it prints before anything else."""
        deadline = time.monotonic() + 10
        while True:
            for line in self.log_path.read_text().splitlines():
                if line.startswith(what + " "):
                    return line.split()[1:]
            assert time.monotonic() < deadline, self.log_path.read_text()
            time.sleep(0.02)

    def child(self) -> int:
        return int(self.logged("child")[0])


def sleeper(launcher: bool) -> subprocess.Popen:
    """A process in a session of its own that sleeps 60 s; with `llm_mini_infer` in its command line when `launcher`,
    as a server's launcher has (`sleep`'s holds no interpreter's path, which might hold the name)."""
    cmd = [sys.executable, "-c", "import time; time.sleep(60)  # llm_mini_infer"] if launcher else ["sleep", "60"]
    return subprocess.Popen(cmd, start_new_session=True)


def running(pid: int) -> bool:
    """Whether `pid` runs: gone, or a zombie (dead, waiting for its reaper), it does not."""
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def group(pgid: int) -> list[int]:
    """The pids of the group `pgid` that run."""
    pids = []
    for p in psutil.process_iter():
        with contextlib.suppress(psutil.Error, OSError):
            if os.getpgid(p.pid) == pgid and p.status() != psutil.STATUS_ZOMBIE:
                pids.append(p.pid)
    return pids


def assert_gone(srv: StandIn, child: int, servers: Path) -> None:
    """Nothing of the server runs, and nothing of it is registered."""
    assert not running(srv.proc.pid) and not running(child) and group(srv.pgid) == []
    assert srv.log.closed and srv.pgid not in api_util._LIVE and not (servers / str(srv.pgid)).exists()


@pytest.fixture(autouse=True)
def servers(tmp_path, monkeypatch) -> Path:
    """Each test's own registry and PID files' folder: its sweeps see no other test's server, nor another session's.
    What a failing test left is swept after it."""
    folder = tmp_path / "servers"
    monkeypatch.setenv(api_util.SERVERS_ENV, str(folder))
    monkeypatch.setattr(api_util, "_LIVE", {})
    yield folder
    api_util.sweep()
    api_util.sweep_stale()


def test_a_stop_leaves_nothing_of_the_group_after_the_launcher_exits(tmp_path, servers) -> None:
    """The launcher exits on SIGTERM with 0, its child left running (a launcher killed while it starts its engine
    leaves one): the stop returns the code, then kills the child, and unregisters."""
    srv = StandIn(tmp_path, "serve").start(10)
    child = srv.child()
    assert os.getpgid(srv.proc.pid) == srv.pgid == srv.proc.pid != os.getpgrp()  # a session of its own
    assert running(child) and list(api_util._LIVE.items()) == [(srv.pgid, srv)]
    assert json.loads((servers / str(srv.pgid)).read_text())["owner"] == os.getpid()
    assert srv.stop() == 0
    assert_gone(srv, child, servers)
    assert srv.stop() == 0  # a second stop: the code, nothing more


def test_a_stop_signals_the_launcher_alone(tmp_path, servers) -> None:
    """The stop's SIGTERM goes to the launcher only: its own handling, its drain and its exit code are what the tests
    test (tests/api/test_lifecycle.py). The rest of its group gets the SIGKILL after the launcher's exit, no SIGTERM."""
    srv = StandIn(tmp_path, "serve").start(10)
    child = srv.child()
    srv.logged("armed")  # the child's SIGTERM handler is set: a SIGTERM to it would be a line in the log
    assert srv.stop() == 0
    assert_gone(srv, child, servers)
    log = srv.log_path.read_text()
    assert "SIGTERM to the launcher" in log and "SIGTERM to the child" not in log, log


def test_a_stop_whose_wait_times_out_kills_the_group_and_removes_the_arena(tmp_path, servers) -> None:
    """A launcher that ignores SIGTERM: the stop's wait times out (TimeoutExpired, as before), and the group is killed
    first, the launcher too, and the pixel arena and IPC directory it made, which a killed launcher leaves, are
    removed (the product's helpers name them: /dev/shm on Linux, inside the IPC directory on macOS)."""
    srv = StandIn(tmp_path, "stubborn+arena").start(10)
    child = srv.child()
    ipc_dir, pixels = srv.logged("arena")
    assert os.path.isdir(ipc_dir) and os.path.isfile(pixels)
    with pytest.raises(subprocess.TimeoutExpired):
        srv.stop(timeout=1)
    assert_gone(srv, child, servers)
    assert not os.path.exists(pixels) and not os.path.exists(ipc_dir)
    assert srv.stop() == -signal.SIGKILL


def test_a_launcher_that_exits_by_itself_keeps_what_it_left(tmp_path, servers) -> None:
    """A launcher that exits by itself removes its arena and IPC directory (llm_mini_infer/launcher.py): one that leaves
    them has a bug, which the stop must not hide (CI reports the files left in /dev/shm). The stand-in exits 0 on
    SIGTERM and leaves both: the stop leaves them too, and the test removes them."""
    srv = StandIn(tmp_path, "serve+arena").start(10)
    child = srv.child()
    ipc_dir, pixels = srv.logged("arena")
    try:
        assert srv.stop() == 0
        assert_gone(srv, child, servers)
        assert os.path.isdir(ipc_dir) and os.path.isfile(pixels)
    finally:
        arena.remove(pixels)
        shutil.rmtree(ipc_dir, ignore_errors=True)


@pytest.mark.parametrize(("how", "timeout", "message", "code"), [("exit", 10, "server exited", 3),
                                                                 ("never", 3, "server not ready", -signal.SIGKILL)])
def test_a_failed_start_leaves_nothing_and_stop_still_returns_a_code(tmp_path, servers, how, timeout, message,
                                                                     code) -> None:
    """A launcher that exits at once keeps its exit code (10 s for it to exit, a margin on a loaded host); one never
    ready is killed at the deadline. Either way the start's error is as before, and nothing of the group runs after
    it."""
    srv = StandIn(tmp_path, how)
    with pytest.raises(RuntimeError, match=message):
        srv.start(timeout)
    assert_gone(srv, srv.child(), servers)
    assert srv.stop() == code and srv.proc.returncode == code


def test_a_registration_that_fails_stops_the_server(tmp_path, servers, monkeypatch) -> None:
    """No folder for the PID file (a file in its way): the launch (and so the start) raises that error, with the server
    it launched stopped, its log closed."""
    (tmp_path / "file").write_text("")
    monkeypatch.setenv(api_util.SERVERS_ENV, str(tmp_path / "file" / "servers"))
    srv = StandIn(tmp_path, "serve")
    with pytest.raises(OSError):
        srv.launch()
    assert not running(srv.proc.pid) and group(srv.pgid) == [] and srv.log.closed and api_util._LIVE == {}
    assert srv.stop() == -signal.SIGKILL


def test_sweep_kills_a_server_never_stopped_and_removes_its_pid_file(tmp_path, servers) -> None:
    srv = StandIn(tmp_path, "serve").start(10)
    child = srv.child()
    assert (servers / str(srv.pgid)).is_file()
    api_util.sweep()
    assert_gone(srv, child, servers)
    assert srv.stop() == -signal.SIGKILL


def test_the_context_manager_stops_on_exit_and_keeps_the_bodys_exception(tmp_path, servers) -> None:
    with StandIn(tmp_path, "serve").start(10) as srv:
        child = srv.child()
    assert_gone(srv, child, servers)
    assert srv.stop() == 0
    with pytest.raises(ValueError, match="the body's"), StandIn(tmp_path, "stubborn").start(10) as srv:
        child = srv.child()
        raise ValueError("the body's")  # and the exit's stop times out: its TimeoutExpired is not the one raised
    assert_gone(srv, child, servers)


@pytest.mark.parametrize("how", ["kill", "exit", "return"])
def test_a_killed_pytests_server_dies_at_the_next_stale_file_sweep(tmp_path, servers, how) -> None:
    """A pytest process that is SIGKILLed, or exits without its exit hooks (os._exit), in the middle of a test: its
    server (a stand-in, which does not watch its parent as a launcher does) lives on, until another process's
    stale-file sweep. While that pytest runs, the sweep leaves its server alone (another session's); one that exits
    normally sweeps its own (atexit)."""
    owner = subprocess.Popen([sys.executable, "-c", OWNER, str(tmp_path)], cwd=ROOT, stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, text=True)
    try:
        launcher = int(owner.stdout.readline())
        record = servers / str(launcher)
        assert json.loads(record.read_text())["owner"] == owner.pid and running(launcher)
        api_util.sweep_stale()
        assert record.is_file() and running(launcher)  # its owner runs
        if how == "kill":
            owner.kill()
        else:
            owner.stdin.write(how + "\n")
            owner.stdin.flush()
        owner.wait(30)
        if how == "return":
            assert not record.exists() and not running(launcher) and group(launcher) == []
            return
        assert record.is_file() and running(launcher)  # a survivor
        api_util.sweep_stale()
        assert not record.exists() and not running(launcher) and group(launcher) == []
    finally:
        if owner.poll() is None:
            owner.kill()
            owner.wait(10)


@pytest.mark.parametrize("case", ["its launcher", "not a launcher", "started later", "no start time"])
def test_the_stale_sweep_kills_no_process_but_the_launcher_its_file_names(servers, case) -> None:
    """A PID file whose owner is gone names a live process in a session of its own: killed only when it is that
    server's launcher (`llm_mini_infer` in its command line, started when the file says); else it lives on, and also
    when the file has no start time. The file goes either way."""
    proc = sleeper(launcher=case != "not a launcher")
    try:
        started = psutil.Process(proc.pid).create_time()
        started = {"started later": started - 100, "no start time": None}.get(case, started)
        servers.mkdir()
        path = servers / str(proc.pid)
        # this process as the owner, started in 1970: a later process given the owner's pid, the owner gone
        path.write_text(json.dumps({"owner": os.getpid(), "owner_started": 0.0, "started": started}))
        api_util.sweep_stale()
        assert not path.exists() and running(proc.pid) == (case != "its launcher")
    finally:
        proc.kill()
        proc.wait(10)


def test_the_stale_sweep_raises_nothing_for_files_it_did_not_write(servers) -> None:
    """What the folder holds that this helper never wrote whole (a directory named like a pid, digits that are not a
    pid, no JSON, an owner or a start time that is not one) is no record: the sweep raises nothing and kills nothing
    for it, removes the files and leaves the directory."""
    proc = sleeper(launcher=True)
    try:
        servers.mkdir()
        (servers / "12345").mkdir()
        (servers / "²").write_text("{}")
        (servers / "007").write_text("{}")
        (servers / "123456789").write_text("not JSON")
        (servers / str(2**31)).write_text(json.dumps({"owner": -1, "started": None}))
        record = {"owner": os.getpid(), "owner_started": 0.0, "started": "now"}  # the owner gone (as above)
        (servers / str(proc.pid)).write_text(json.dumps(record))
        api_util.sweep_stale()
        assert [p.name for p in servers.iterdir()] == ["12345"] and running(proc.pid)
    finally:
        proc.kill()
        proc.wait(10)


def test_the_default_folder_is_this_users_own(tmp_path, monkeypatch) -> None:
    """Without $LLM_MINI_INFER_TEST_SERVERS, the PID files go to a private folder under the temporary directory, named
    by the user, which a launch makes and a sweep does not; one that another user owns is refused, with an error that
    names it."""
    uid = os.getuid()
    with monkeypatch.context() as m:  # undone before the fixtures' sweeps
        m.delenv(api_util.SERVERS_ENV)
        m.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
        api_util.sweep_stale()
        folder = tmp_path / f"llm-mini-servers-{uid}"
        assert not folder.exists()
        assert api_util.servers_dir(create=True) == folder and folder.stat().st_mode & 0o777 == 0o700
        (tmp_path / f"llm-mini-servers-{uid + 1}").mkdir()
        m.setattr(os, "getuid", lambda: uid + 1)  # as another user: the folder is not theirs
        with pytest.raises(RuntimeError, match=f"llm-mini-servers-{uid + 1} is not a folder of this user's"):
            api_util.servers_dir()
