"""What a server leaves outside its processes, and how the next start removes it after a crash.

The launcher makes two things its processes share by path (`launcher.serve`): its socket directory in /tmp
(`engine.ipc.ipc_dir`) and, with images on, its pixel arena (`vision.arena`) in /dev/shm where that is writable, which
holds its whole size (`dev.images.arena_mb`) in memory. It removes both on every exit it lives through. A SIGKILL (the
OOM killer, a runtime's stop timeout) leaves them, and in a pod /dev/shm outlives a restarted container.
- **A live server holds a lock on each:** an exclusive `flock` on the lock file in its socket directory (LOCK_FILE) and
  on its arena, from the moment each appears until the launcher has removed it. The kernel drops a flock when the last
  descriptor of its open file closes, which a process's death does; the launcher's children are spawned and inherit
  none of its descriptors, so orphans that outlive it for a moment do not keep its files. Not a pid check: in a pod
  the launcher is often pid 1, after every restart. `flock`, not `fcntl.lockf`: a POSIX lock drops when any
  descriptor of the file closes, and the arena's users open and close it (`vision.arena.Arena`).
- **A file appears locked** (`locked_file`): it is made under another name, locked, then renamed, so a sweep beside a
  starting server never sees it unlocked. A launcher killed before the rename leaves its socket directory, or an empty
  file, for good.
- **The next start removes a dead server's** (`sweep`, called by `launcher.serve` and by the container's entrypoint
  before its /dev/shm check): in tmp_root(), a directory named PREFIX* whose lock file nobody holds; in shm_root(), an
  arena (PREFIX*ARENA_SUFFIX) nobody holds. Only this user's own, never through a symbolic link, and only what this
  version marks as its own: a directory without the lock file (an older version's, a test's, one being made) and a
  file of another name (an older version's arena ends in -pixels) stay, however old. It removes files and stops no
  process: a dead launcher's engine, frontends and pool workers stop by themselves (they watch it).

**The tests' root** (TEST_ROOT_ENV, a test hook, not a setting): the socket directories go there instead of /tmp, and
the arenas to its `shm` when that is a writable directory, else into the socket directory, so that no test sweeps, or
leaves files in, the real ones (tests/conftest.py). /tmp, not $TMPDIR: macOS's and pytest's are too long for a
socket's path (about 104 bytes).
"""

from __future__ import annotations

import contextlib
import errno
import fcntl
import logging
import os
import shutil
import stat
import tempfile

log = logging.getLogger("llm_mini_infer.leftovers")

PREFIX = "llm-mini-infer-"  # a socket directory's name and an arena's start with it
LOCK_FILE = "lock"  # in a socket directory: its server's lock
ARENA_SUFFIX = "-pixel-arena"  # an arena's name ends with it (`vision.arena.arena_path`); an older version's, -pixels
NEW_SUFFIX = ".new"  # a file's name while `locked_file` makes it: no sweep matches it
TEST_ROOT_ENV = "LLM_MINI_INFER_TEST_TMP"


def tmp_root() -> str:
    """Where socket directories go: /tmp (or, without one, the system's temporary directory), or the tests' root."""
    return os.environ.get(TEST_ROOT_ENV) or ("/tmp" if os.path.isdir("/tmp") else tempfile.gettempdir())


def shm_root() -> str | None:
    """Where arenas go: /dev/shm when it is a writable directory (tmpfs), or with the tests' root its `shm` when that is
    one; None: into the socket directory, where its lock covers them (macOS)."""
    root = os.environ.get(TEST_ROOT_ENV)
    path = os.path.join(root, "shm") if root else "/dev/shm"
    return path if os.path.isdir(path) and os.access(path, os.W_OK) else None


def locked_file(path: str) -> int:
    """A new empty file at `path` (mode 0600) and its descriptor, which holds an exclusive flock on it until it is
    closed: close it once the file is removed. The file is made as `path` + NEW_SUFFIX and renamed once locked, so it is
    locked from the moment it appears at `path`. A failure removes what it made."""
    new = path + NEW_SUFFIX
    fd = os.open(new, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)  # nobody else opens that name: never busy
        os.rename(new, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(new)
        os.close(fd)
        raise
    return fd


def sweep() -> None:
    """Remove what dead servers left (the module's docstring), each removal logged with its size. A failure is logged
    and the entry skipped: the sweep never stops a start."""
    removed: list[int] = []
    for root, kind in ((tmp_root(), "socket directory"), (shm_root(), "arena")):
        if root is None:
            continue
        try:
            names = sorted(os.listdir(root))
        except OSError as exc:
            log.warning("cannot look for a dead server's files in %s: %s", root, exc)
            continue
        for name in names:
            if not name.startswith(PREFIX) or (kind == "arena" and not name.endswith(ARENA_SUFFIX)):
                continue
            path = os.path.join(root, name)
            try:
                size = _remove_dir(path) if kind == "socket directory" else _remove_arena(path)
            except FileNotFoundError:  # gone meanwhile: its server stopped, or another start removed it
                continue
            except Exception as exc:
                log.warning("cannot remove %s, a dead server's %s: %s: %s", path, kind, type(exc).__name__, exc)
                continue
            if size is not None:
                log.warning("removed %s (%s), the %s of a server that was killed", path, _size(size), kind)
                removed.append(size)
    if removed:
        log.info("removed %d file%s of killed servers: %s", len(removed), "" if len(removed) == 1 else "s",
                 _size(sum(removed)))


def _size(n: int) -> str:
    return f"{n >> 20} MiB" if n >= 1 << 20 else f"{n} bytes"


def _remove_dir(path: str) -> int | None:
    """Remove the socket directory at `path` if it is a dead server's: this user's directory, holding a lock file
    (LOCK_FILE, a regular file of this user's) nobody holds. Its size, or None when it stays."""
    st = os.lstat(path)
    if not _ours(path, st, stat.S_ISDIR):
        return None
    dfd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        if not _same(os.fstat(dfd), st):  # replaced since the lstat
            return None
        try:
            fd = os.open(LOCK_FILE, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=dfd)
        except FileNotFoundError:  # not this version's, or still being made
            return None
        except OSError as exc:
            if exc.errno == errno.ELOOP:  # a symbolic link: not a lock file of ours
                return None
            raise
        try:
            if not _ours(f"{path}/{LOCK_FILE}", os.fstat(fd), stat.S_ISREG) or not _unlocked(fd):
                return None
            size = sum(e.stat(follow_symlinks=False).st_size for e in os.scandir(dfd))
            if not _same(os.lstat(path), st):
                return None
            shutil.rmtree(path)
            return size
        finally:
            os.close(fd)  # after the removal: a sweep beside this one finds the lock held until then
    finally:
        os.close(dfd)


def _remove_arena(path: str) -> int | None:
    """Remove the arena at `path` if it is a dead server's: a regular file of this user's that nobody holds. Its size,
    or None when it stays."""
    st = os.lstat(path)
    if not _ours(path, st, stat.S_ISREG):  # a FIFO would block the open below
        return None
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        fst = os.fstat(fd)
        if not _same(fst, st) or not _unlocked(fd) or not _same(os.lstat(path), fst):
            return None
        os.unlink(path)
        return fst.st_size
    finally:
        os.close(fd)


def _ours(path: str, st: os.stat_result, kind) -> bool:
    """`st` is of `kind` (S_ISDIR, S_ISREG: never a symbolic link) and this user's; another user's is logged."""
    if not kind(st.st_mode):
        return False
    if st.st_uid != os.getuid():
        log.debug("kept %s: uid %d's", path, st.st_uid)
        return False
    return True


def _same(a: os.stat_result, b: os.stat_result) -> bool:
    return (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino)


def _unlocked(fd: int) -> bool:
    """Take the flock on `fd` if nobody holds it: True, and it stays this process's until `fd` closes."""
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:  # a live server's
        return False
    return True
