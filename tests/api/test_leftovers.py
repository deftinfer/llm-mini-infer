"""What a killed server leaves and the next start removes (`llm_mini_infer.leftovers`): what the sweep removes, what it
keeps, the locks that tell a live server's files from a dead one's, the race with a server being made, and a filesystem
that refuses flock. Every test under its own root (`private_tmp`), with or without an `shm` folder (`layout`: Linux's
/dev/shm, or none as on macOS): nothing here touches /tmp or /dev/shm themselves (the flock check on /dev/shm runs in
the test's own folder there)."""

from __future__ import annotations

import errno
import fcntl
import json
import logging
import os
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_mini_infer import leftovers
from llm_mini_infer.engine import ipc
from llm_mini_infer.leftovers import ARENA_SUFFIX, LOCK_FILE, NEW_SUFFIX, PREFIX
from llm_mini_infer.vision import arena

from ..conftest import HELD, remove_stale_roots
from ..support.paths import ROOT

LOGGER = "llm_mini_infer.leftovers"

# A live server's files, made as the launcher makes them, held until its stdin closes or it is killed
HOLDER = """
import json, os, sys
from llm_mini_infer.engine import ipc
from llm_mini_infer.vision import arena
path, lock = ipc.locked_dir()
pixels = arena.arena_path(arena.default_dir(path), f"{os.getpid()}-{os.path.basename(path)}")
pixels, pixels_lock = arena.create_locked(pixels, 4096)
print(json.dumps([path, pixels]), flush=True)
sys.stdin.read()
"""

# A server's files made as the launcher makes them, the maker stopped after each system call of the making until a
# line on stdin: each step's name on stdout, then "made"
MAKER = """
import fcntl, os, sys, tempfile

from llm_mini_infer.engine import ipc
from llm_mini_infer.vision import arena


def pause(step):
    print(step, flush=True)
    if not sys.stdin.readline():  # the test is gone
        os._exit(1)


def stopping_after(module, name):
    call = getattr(module, name)

    def stopped(*args, **kwargs):
        out = call(*args, **kwargs)
        pause(name)
        return out

    setattr(module, name, stopped)


for module, name in ((tempfile, "mkdtemp"), (os, "open"), (fcntl, "flock"), (os, "rename"), (os, "ftruncate")):
    stopping_after(module, name)
path, lock = ipc.locked_dir()
pixels = arena.arena_path(arena.default_dir(path), f"{os.getpid()}-{os.path.basename(path)}")
pixels, pixels_lock = arena.create_locked(pixels, 4096)
pause("made")
"""
# the directory, its lock file (made, locked, renamed), then the arena (made, locked, renamed, sized)
STEPS = ["mkdtemp", "open", "flock", "rename", "open", "flock", "rename", "ftruncate", "made"]

# A holder of `locked_file`'s lock that spawns a child mapping the file, as the launcher spawns its pool workers
SPAWNER = """
import multiprocessing as mp, os, sys, time

from llm_mini_infer import leftovers
from tests.api.test_leftovers import share_of

if __name__ == "__main__":
    fd = leftovers.locked_file(sys.argv[1])
    os.ftruncate(fd, 1 << 16)
    child = mp.get_context("spawn").Process(target=share_of, args=(sys.argv[1],))
    child.start()
    print(child.pid, flush=True)
    time.sleep(60)
"""


def share_of(path: str) -> None:
    """A pool worker's share of an arena (`vision.arena.map_writable`), for 60 s: SPAWNER's child. `path`.mapped tells
    the test it is mapped."""
    mapping = arena.map_writable(path)
    Path(path + ".mapped").touch()
    time.sleep(60)
    mapping.close()


def _with_shm(root: str) -> str:
    """`root` with a plain `shm` folder (in place of private_tmp's link, whose target its teardown removes)."""
    shm = os.path.join(root, "shm")
    if os.path.islink(shm):
        os.unlink(shm)
    os.makedirs(shm, exist_ok=True)
    return root


@pytest.fixture
def shm_layout(private_tmp) -> str:
    return _with_shm(private_tmp)


@pytest.fixture(params=["shm", "no shm"])
def layout(request, private_tmp) -> str:
    """The test's root, with a writable `shm` folder (Linux's /dev/shm) or none (macOS: arenas in the socket
    directory)."""
    if request.param == "shm":
        return _with_shm(private_tmp)
    shm = os.path.join(private_tmp, "shm")
    if os.path.islink(shm):
        os.unlink(shm)
    return private_tmp


def _line(proc: subprocess.Popen, timeout: float = 60) -> str:
    """`proc`'s next line on stdout (it writes one, then waits for the test)."""
    assert select.select([proc.stdout], [], [], timeout)[0], "the child said nothing"
    line = proc.stdout.readline()
    assert line, "the child exited"
    return line.strip()


def _entries(root: str) -> set[str]:
    """Every folder and file under `root`, `shm`'s too, by its path there; but `private_tmp`'s lock."""
    out = set()
    for folder, dirs, files in os.walk(root, followlinks=True):
        out.update(os.path.relpath(os.path.join(folder, name), root) for name in dirs + files)
    return out - {HELD}


def _free(path: str) -> bool:
    """Whether the flock on `path` is free: another open of it takes it (and drops it at once)."""
    fd = os.open(path, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        return False
    finally:
        os.close(fd)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def make_dead(root: str) -> tuple[str, str]:
    """A server's socket directory and arena as it makes them, in `root`, its owner gone: both locks released. The
    arena's name holds this live process's pid: the sweep does not judge by pid."""
    path, fd = ipc.locked_dir()
    os.close(fd)
    assert os.path.dirname(path) == root
    pixels = arena.arena_path(arena.default_dir(path), f"{os.getpid()}-{os.path.basename(path)}")
    pixels, fd = arena.create_locked(pixels, 4096)
    os.close(fd)
    return path, pixels


def sweep_now() -> None:
    """The sweep, in a thread that must end quickly: a FIFO opened for reading would block it for good."""
    thread = threading.Thread(target=leftovers.sweep, daemon=True)
    thread.start()
    thread.join(20)
    assert not thread.is_alive(), "the sweep blocked"


@pytest.fixture
def live_server(layout):
    """A child process holding a server's socket directory and arena, locked, until its stdin closes or it is
    killed."""
    proc = subprocess.Popen([sys.executable, "-c", HOLDER], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                            cwd=ROOT)
    try:
        path, pixels = json.loads(_line(proc))
        yield SimpleNamespace(proc=proc, dir=path, arena=pixels)
    finally:
        proc.kill()
        proc.wait(10)


# -- removed -----------------------------------------------------------------------------------------------------------

def test_a_dead_servers_files_are_removed(layout) -> None:
    """Its socket directory, whose lock file nobody holds, and its arena: in `shm` where there is one, in the directory
    otherwise, where it goes with it."""
    path, pixels = make_dead(layout)
    assert (os.path.dirname(pixels) == path) == (leftovers.shm_root() is None)
    leftovers.sweep()
    assert not os.path.lexists(path) and not os.path.lexists(pixels)
    assert _entries(layout) <= {"shm"}


def test_a_dead_arena_is_removed_whatever_pid_its_name_holds(shm_layout) -> None:
    """Alone (in a pod, /dev/shm outlives the container, /tmp does not), with a live pid in its name (this process's):
    only the lock counts. In a pod the next launcher has the same pid as the dead one."""
    path, pixels = make_dead(shm_layout)
    shutil.rmtree(path)
    assert os.path.dirname(pixels) == os.path.join(shm_layout, "shm") and str(os.getpid()) in pixels
    leftovers.sweep()
    assert not os.path.lexists(pixels)


def test_a_live_servers_files_stay_until_it_is_killed(live_server) -> None:
    """Its locks keep its files; its death (SIGKILL) releases them for the next sweep."""
    leftovers.sweep()
    assert os.path.isdir(live_server.dir) and os.path.isfile(live_server.arena)
    live_server.proc.kill()
    live_server.proc.wait(10)
    leftovers.sweep()
    assert not os.path.lexists(live_server.dir) and not os.path.lexists(live_server.arena)


# -- kept --------------------------------------------------------------------------------------------------------------

def _directory_without_a_lock_file(root: str, shm: str) -> list[str]:  # an older version's, a test's, however old
    path = ipc.ipc_dir()
    os.utime(path, (0, 0))
    return [path]


def _old_arena(root: str, shm: str) -> list[str]:  # an older version's name, unlocked
    path = os.path.join(shm, f"{PREFIX}{os.getpid()}-x-pixels")
    Path(path).write_bytes(b"x" * 4096)
    return [path]


def _arena_being_made(root: str, shm: str) -> list[str]:
    path = arena.arena_path(shm, "1-x") + NEW_SUFFIX
    Path(path).write_bytes(b"")
    return [path]


def _directory_of_another_name(root: str, shm: str) -> list[str]:
    path = os.path.join(root, "other-dir")
    os.mkdir(path)
    Path(path, LOCK_FILE).write_bytes(b"")
    return [path]


def _arena_of_another_name(root: str, shm: str) -> list[str]:
    path = os.path.join(shm, f"{PREFIX}x")
    Path(path).write_bytes(b"")
    return [path]


def _link_to_a_dead_directory(root: str, shm: str) -> list[str]:
    target, fd = ipc.locked_dir()
    os.close(fd)
    elsewhere = os.path.join(root, "elsewhere")
    os.mkdir(elsewhere)
    os.rename(target, os.path.join(elsewhere, os.path.basename(target)))
    link = os.path.join(root, f"{PREFIX}link")
    os.symlink(os.path.join(elsewhere, os.path.basename(target)), link)
    return [link, os.path.join(elsewhere, os.path.basename(target), LOCK_FILE)]


def _link_to_a_dead_arena(root: str, shm: str) -> list[str]:
    target = os.path.join(root, "dead-arena")
    os.close(arena.create_locked(target, 4096)[1])
    link = arena.arena_path(shm, "1-link")
    os.symlink(target, link)
    return [link, target]


def _lock_file_that_is_a_link(root: str, shm: str) -> list[str]:
    path = ipc.ipc_dir()
    real = os.path.join(root, "real-lock")
    Path(real).write_bytes(b"")
    os.symlink(real, os.path.join(path, LOCK_FILE))
    return [path, real]


def _lock_file_that_is_a_directory(root: str, shm: str) -> list[str]:
    path = ipc.ipc_dir()
    os.mkdir(os.path.join(path, LOCK_FILE))
    return [path]


def _fifo_named_like_an_arena(root: str, shm: str) -> list[str]:  # an open for reading would block
    path = arena.arena_path(shm, "1-fifo")
    os.mkfifo(path)
    return [path]


KEPT = [_directory_without_a_lock_file, _old_arena, _arena_being_made, _directory_of_another_name,
        _arena_of_another_name, _link_to_a_dead_directory, _link_to_a_dead_arena, _lock_file_that_is_a_link,
        _lock_file_that_is_a_directory, _fifo_named_like_an_arena]


@pytest.mark.parametrize("make", KEPT, ids=lambda f: f.__name__.strip("_"))
def test_what_is_not_a_dead_servers_stays(layout, make, caplog) -> None:
    """Not this version's (no lock file, another name), not a directory or regular file of its own (a symbolic link,
    a FIFO), or still being made: kept, never followed, and no warning."""
    kept = make(layout, arena.default_dir(layout))
    sweep_now()
    assert all(os.path.lexists(path) for path in kept), kept
    assert not [r for r in caplog.records if r.name == LOGGER and r.levelno >= logging.WARNING]


def test_another_users_files_stay_and_are_logged_at_debug(shm_layout, monkeypatch, caplog) -> None:
    path, pixels = make_dead(shm_layout)
    uid = os.getuid()
    monkeypatch.setattr(leftovers.os, "getuid", lambda: uid + 1)
    caplog.set_level(logging.DEBUG, logger=LOGGER)
    leftovers.sweep()
    assert os.path.isdir(path) and os.path.isfile(pixels)
    kept = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]
    assert any(path in m for m in kept) and any(pixels in m for m in kept), kept


# -- the log and errors ------------------------------------------------------------------------------------------------

def test_a_removal_is_logged_with_its_size(shm_layout, caplog) -> None:
    caplog.set_level(logging.INFO, logger=LOGGER)
    path, pixels = make_dead(shm_layout)
    leftovers.sweep()
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any(path in m and "socket directory" in m for m in warnings), warnings
    assert any(pixels in m and "(4096 bytes), the arena" in m for m in warnings), warnings
    infos = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert infos == ["removed 2 files of killed servers: 4096 bytes"]


def test_a_failed_removal_is_logged_and_the_sweep_goes_on(shm_layout, monkeypatch, caplog) -> None:
    """A directory and an arena that cannot be removed stay, with a warning each; the others go."""
    (first, first_arena), (second, second_arena) = make_dead(shm_layout), make_dead(shm_layout)
    rmtree, unlink = leftovers.shutil.rmtree, leftovers.os.unlink

    def failing(real, refused):
        def call(path, *args, **kwargs):
            if path == refused:
                raise PermissionError("refused (test)")
            return real(path, *args, **kwargs)
        return call

    monkeypatch.setattr(leftovers.shutil, "rmtree", failing(rmtree, first))
    monkeypatch.setattr(leftovers.os, "unlink", failing(unlink, first_arena))
    leftovers.sweep()
    assert os.path.isdir(first) and os.path.isfile(first_arena)
    assert not os.path.lexists(second) and not os.path.lexists(second_arena)
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any(m.startswith(f"cannot remove {first},") for m in warnings), warnings
    assert any(m.startswith(f"cannot remove {first_arena},") for m in warnings), warnings


# -- the locks and the race --------------------------------------------------------------------------------------------

def test_a_locked_file_holds_its_lock_until_its_descriptor_closes(private_tmp) -> None:
    path = os.path.join(private_tmp, "locked")
    fd = leftovers.locked_file(path)
    assert os.path.isfile(path) and not os.path.lexists(path + NEW_SUFFIX)
    assert os.stat(path).st_mode & 0o777 == 0o600
    assert not _free(path)
    os.close(fd)
    assert _free(path)


def test_a_locked_file_leaves_a_file_being_made_alone(private_tmp) -> None:
    path = os.path.join(private_tmp, "locked")
    Path(path + NEW_SUFFIX).write_text("keep")
    with pytest.raises(FileExistsError):
        leftovers.locked_file(path)
    assert Path(path + NEW_SUFFIX).read_text() == "keep" and not os.path.lexists(path)


def test_a_sweep_never_removes_a_server_being_made(layout) -> None:
    """The race between a starting server and a sweep beside it: a process makes a server's files as the launcher does
    (`ipc.locked_dir`, `arena.create_locked`), stopped after each system call of the making; a sweep at each stop
    removes nothing, from the directory without its lock file to the arena's allocation. Once the maker is killed, the
    next sweep removes what it made."""
    maker = subprocess.Popen([sys.executable, "-c", MAKER], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                             cwd=ROOT)
    steps: list[str] = []
    try:
        for _ in STEPS:
            steps.append(_line(maker))
            there = _entries(layout)
            leftovers.sweep()
            assert _entries(layout) == there, steps
            if steps[-1] != "made":
                maker.stdin.write("\n")
                maker.stdin.flush()
        assert steps == STEPS
        assert any(e.endswith(f"/{LOCK_FILE}") for e in there) and any(e.endswith(ARENA_SUFFIX) for e in there), there
    finally:
        maker.kill()
        maker.wait(10)
    leftovers.sweep()
    assert _entries(layout) <= {"shm"}


@pytest.mark.parametrize("where", ["the tests' root", "/dev/shm"])
def test_a_killed_holders_lock_is_free_at_once(where, private_tmp) -> None:
    """The premise: the flock `locked_file` takes is held until its holder dies, and its SIGKILL frees it at once,
    though a child it spawned (as the launcher spawns its pool workers) still maps the file. On the tests' root, and
    on /dev/shm itself (tmpfs, Linux) in the test's own folder there (`private_tmp`'s `shm`), which the next session
    removes if this one is killed."""
    if where == "/dev/shm":
        shm = os.path.join(private_tmp, "shm")  # a link to it, where the session's root has a folder in /dev/shm
        if not (os.path.isdir(shm) and os.path.realpath(shm).startswith(os.path.realpath("/dev/shm") + os.sep)):
            pytest.skip("no writable /dev/shm on this host")
        folder = tempfile.mkdtemp(prefix="flock-", dir=shm)
    else:
        folder = tempfile.mkdtemp(dir=private_tmp)
    path, child = os.path.join(folder, "held"), 0
    holder = subprocess.Popen([sys.executable, "-c", SPAWNER, path], stdout=subprocess.PIPE, text=True, cwd=ROOT)
    try:
        child = int(_line(holder))
        deadline = time.monotonic() + 60
        while not os.path.exists(path + ".mapped"):
            assert time.monotonic() < deadline and _alive(child), "the child did not map the file"
            time.sleep(0.05)
        assert not _free(path)
        holder.kill()
        holder.wait(10)
        assert _alive(child) and _free(path)
    finally:
        holder.kill()
        holder.wait(10)
        if child:
            os.kill(child, signal.SIGKILL)
            deadline = time.monotonic() + 10
            while _alive(child) and time.monotonic() < deadline:
                time.sleep(0.05)
        shutil.rmtree(folder, ignore_errors=True)


# -- a filesystem that refuses flock -----------------------------------------------------------------------------------

def _refusing(error: OSError):
    def flock(fd, operation):
        raise error
    return flock


def _warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == LOGGER and r.levelno >= logging.WARNING]


@pytest.mark.parametrize("name", ["ENOLCK", "ENOSYS", "EOPNOTSUPP", "ENOTSUP"])
def test_a_filesystem_refusing_flock_gets_files_no_sweep_removes(layout, name, caplog, monkeypatch) -> None:
    """Its socket directory has no lock file and its arena is made under another name, each with one warning naming the
    errno; a sweep, with flock working again, removes neither: nothing tells it that their server is dead."""
    code = getattr(errno, name)
    with monkeypatch.context() as m:  # only while the files are made: the sweep takes the same function
        m.setattr(leftovers.fcntl, "flock", _refusing(OSError(code, os.strerror(code))))
        path, fd = ipc.locked_dir()
        pixels = arena.arena_path(arena.default_dir(path), f"{os.getpid()}-{os.path.basename(path)}")
        made, pfd = arena.create_locked(pixels, 4096)
    assert fd is None and pfd is None
    assert not os.path.lexists(os.path.join(path, LOCK_FILE))
    assert not os.path.lexists(os.path.join(path, LOCK_FILE + NEW_SUFFIX))
    assert made == pixels + leftovers.UNLOCKED_SUFFIX and os.path.getsize(made) == 4096
    assert not os.path.lexists(pixels) and not made.endswith(ARENA_SUFFIX)
    warnings = _warnings(caplog)
    assert len(warnings) == 2, warnings
    assert warnings[0].startswith(f"cannot lock {os.path.join(path, LOCK_FILE)} ({errno.errorcode[code]}: "), warnings
    assert warnings[1].startswith(f"cannot lock {pixels} ({errno.errorcode[code]}: ") and made in warnings[1], warnings
    sweep_now()
    assert os.path.isdir(path) and os.path.isfile(made)
    assert len(_warnings(caplog)) == 2


@pytest.mark.parametrize("error", [BlockingIOError(errno.EAGAIN, "busy"), PermissionError(errno.EACCES, "refused")],
                         ids=["busy", "EACCES"])
def test_a_refused_flock_of_another_kind_still_raises(layout, error, monkeypatch) -> None:
    """Not the filesystem refusing flock (a lock held, a permission): the error reaches the caller, and neither the
    socket directory nor the arena is left."""
    there = _entries(layout)
    pixels = arena.arena_path(arena.default_dir(layout), "1-x")
    with monkeypatch.context() as m:
        m.setattr(leftovers.fcntl, "flock", _refusing(error))
        with pytest.raises(type(error)):
            ipc.locked_dir()
        assert _entries(layout) == there
        with pytest.raises(type(error)):
            arena.create_locked(pixels, 4096)
    for name in (pixels, pixels + NEW_SUFFIX, pixels + leftovers.UNLOCKED_SUFFIX):
        assert not os.path.lexists(name), name
    assert _entries(layout) == there


def test_a_sweep_where_flock_fails_keeps_everything(shm_layout, monkeypatch, caplog) -> None:
    """A dead server's directory and arena whose lock the sweep cannot take (the filesystem refuses flock): both stay,
    with a warning each; once flock works they go."""
    path, pixels = make_dead(shm_layout)
    with monkeypatch.context() as m:
        m.setattr(leftovers.fcntl, "flock", _refusing(OSError(errno.ENOLCK, os.strerror(errno.ENOLCK))))
        leftovers.sweep()
    assert os.path.isdir(path) and os.path.isfile(pixels)
    warnings = _warnings(caplog)
    assert len(warnings) == 2, warnings
    assert any(w.startswith(f"cannot remove {path},") for w in warnings), warnings
    assert any(w.startswith(f"cannot remove {pixels},") for w in warnings), warnings
    leftovers.sweep()
    assert not os.path.lexists(path) and not os.path.lexists(pixels)


# -- the tests' roots --------------------------------------------------------------------------------------------------

def test_a_killed_sessions_private_roots_go_at_the_next_start(private_tmp) -> None:
    """A test's private root (`private_tmp`) whose pytest was killed before the teardown: nobody holds its lock, so
    the next session's start removes it, with its folder in `shm`; a root whose test runs (its lock held, here by this
    process) and one without the lock file yet stay."""
    session = tempfile.mkdtemp(dir=private_tmp)
    os.mkdir(os.path.join(session, "shm"))
    dead, live, new = (os.path.join(session, f"t-{name}") for name in ("dead", "live", "new"))
    twin = os.path.join(session, "shm", "t-twin")
    for path in (dead, live, new, twin):
        os.mkdir(path)
    Path(twin, "arena").write_bytes(b"x")
    os.symlink(twin, os.path.join(dead, "shm"))
    os.close(leftovers.locked_file(os.path.join(dead, HELD)))
    fd = leftovers.locked_file(os.path.join(live, HELD))
    try:
        remove_stale_roots(session)
        assert not os.path.lexists(dead) and not os.path.lexists(twin)
        assert os.path.isdir(live) and os.path.isdir(new)
    finally:
        os.close(fd)


def test_the_files_follow_the_tests_root(layout) -> None:
    """The socket directories in it, the arenas in its `shm` where there is one, else in the socket directory."""
    path = ipc.ipc_dir()
    assert os.path.dirname(path) == layout and os.path.basename(path).startswith(PREFIX)
    shm = os.path.join(layout, "shm") if os.path.isdir(os.path.join(layout, "shm")) else None
    assert leftovers.shm_root() == shm and arena.default_dir(path) == (shm or path)


def test_a_tests_root_outside_the_harness_is_a_warning(monkeypatch, caplog, tmp_path) -> None:
    """The launcher warns once when the tests' root is set outside the harness's own (`launcher._warn_test_root`): the
    harness's root and the private roots under it are quiet, as is no root at all."""
    from llm_mini_infer import launcher

    name = "llm_mini_infer.launcher"
    caplog.set_level(logging.WARNING, logger=name)

    def warned() -> list[logging.LogRecord]:
        return [r for r in caplog.records if r.name == name and r.levelno == logging.WARNING]

    harness = os.path.join("/tmp", f"{leftovers.TEST_ROOT_PREFIX}{os.getuid()}")  # tests/conftest.py's
    for quiet in (harness, os.path.join(harness, "t-x"), harness + "/"):
        monkeypatch.setenv(leftovers.TEST_ROOT_ENV, quiet)
        launcher._warn_test_root()
        assert not warned(), quiet
    monkeypatch.delenv(leftovers.TEST_ROOT_ENV)
    launcher._warn_test_root()
    assert not warned()
    for loud in (str(tmp_path), harness + "-other", os.path.join("/tmp", f"{leftovers.TEST_ROOT_PREFIX}x", "t-x")):
        caplog.clear()
        monkeypatch.setenv(leftovers.TEST_ROOT_ENV, loud)
        launcher._warn_test_root()
        (record,) = warned()
        assert record.getMessage().startswith(f"{leftovers.TEST_ROOT_ENV} is set ({loud}), a test hook"), loud
