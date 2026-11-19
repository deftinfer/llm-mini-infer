"""The pixel arena: one shared-memory file for every image's uint8 pixels, which never travel over ZMQ
(architecture.md, "Frontend and engine messages").

- **Who maps it:** the launcher creates it (`create_locked`), the image pool's workers write the pixels of their jobs
  into it, and the engine reads them when it encodes an image (`map_readonly`). The frontends never map it: each one
  owns a fixed partition and allocates byte ranges there (`Allocator`), passing the offsets around.
- **Where:** `/dev/shm` when it exists and is writable (tmpfs), else the server's private IPC directory. The file is
  unlinked when the launcher stops, and a killed launcher's when the next server starts (`leftovers`): the launcher
  holds a lock on it meanwhile.
- **Lifetime of a range:** the frontend allocates it before it submits the pool job and frees it once the engine says
  the request no longer needs its pixels (`StepOutputs.released`: after the request left the engine and every step
  launched until then completed; an encoder-cache eviction or a preemption may need the pixels again before that), or
  at once when the request fails before reaching the engine. A range whose pool job is still running is freed only
  once the job resolves: its worker answered, or is gone (the launcher moves a worker's epoch only after its process
  exited, a hung one killed at the job deadline first: `vision.pool`). A job re-sent to another worker keeps its range.
"""

from __future__ import annotations

import bisect
import contextlib
import mmap
import os
from pathlib import Path

from .. import leftovers

ALIGN = 4096  # ranges start on page boundaries: a worker's write never shares a page with another range


def default_dir(ipc_dir: str) -> str:
    return leftovers.shm_root() or ipc_dir


def create(path: str, size: int) -> None:
    """Create the arena file with all its pages allocated (`posix_fallocate` where it exists), so the engine's
    pre-populated mapping never takes a page fault on a pixel read."""
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        _allocate(fd, path, size)
    finally:
        os.close(fd)


def create_locked(path: str, size: int) -> int:
    """The launcher's arena: created as `create` does, but locked from the moment it appears (`leftovers.locked_file`).
    The lock's descriptor: close it once the file is removed. A failure removes the file."""
    fd = leftovers.locked_file(path)
    try:
        _allocate(fd, path, size)
    except BaseException:
        remove(path)
        os.close(fd)
        raise
    return fd


def _allocate(fd: int, path: str, size: int) -> None:
    os.ftruncate(fd, size)
    if hasattr(os, "posix_fallocate"):
        try:
            os.posix_fallocate(fd, 0, size)
        except OSError as exc:  # e.g. /dev/shm smaller than the arena: fail now, not on the 100th image
            raise OSError(f"cannot allocate the {size >> 20} MiB pixel arena at {path}: {exc}") from exc


def remove(path: str) -> None:
    with contextlib.suppress(FileNotFoundError):
        os.unlink(path)


class Arena:
    """A mapping of the arena file: writable (the workers) or read-only and pre-populated (the engine)."""

    def __init__(self, path: str, writable: bool):
        self.path = path
        fd = os.open(path, os.O_RDWR if writable else os.O_RDONLY)
        try:
            self.size = os.fstat(fd).st_size
            flags = mmap.MAP_SHARED | (getattr(mmap, "MAP_POPULATE", 0) if not writable else 0)
            prot = mmap.PROT_READ | (mmap.PROT_WRITE if writable else 0)
            self.mm = mmap.mmap(fd, self.size, flags=flags, prot=prot)
        finally:
            os.close(fd)
        self.buf = memoryview(self.mm)

    def close(self) -> None:
        self.buf.release()
        self.mm.close()


def map_readonly(path: str) -> Arena:
    return Arena(path, writable=False)


def map_writable(path: str) -> Arena:
    return Arena(path, writable=True)


class Allocator:
    """First-fit ranges in one frontend's partition [base, base + size), page-aligned, with coalescing frees."""

    def __init__(self, base: int, size: int):
        self.base, self.size = base, size
        self._starts = [base]  # free ranges, sorted by start
        self._lens = {base: size}
        self.used = 0

    def alloc(self, n: int) -> int | None:
        n = -(-max(1, n) // ALIGN) * ALIGN
        for s in self._starts:
            ln = self._lens[s]
            if ln >= n:
                self._starts.remove(s)
                del self._lens[s]
                if ln > n:
                    rest = s + n
                    bisect.insort(self._starts, rest)
                    self._lens[rest] = ln - n
                self.used += n
                return s
        return None

    def free(self, off: int, n: int) -> None:
        n = -(-max(1, n) // ALIGN) * ALIGN
        self.used -= n
        i = bisect.bisect_left(self._starts, off)
        start, length = off, n
        if i < len(self._starts) and self._starts[i] == off + n:  # merge with the next free range
            length += self._lens.pop(self._starts.pop(i))
        if i > 0:
            prev = self._starts[i - 1]
            if prev + self._lens[prev] == off:  # merge with the previous one
                self._lens[prev] += length
                return
        self._starts.insert(i, start)
        self._lens[start] = length

    @property
    def free_bytes(self) -> int:
        return self.size - self.used


def arena_path(directory: str, tag: str) -> str:
    return str(Path(directory) / f"{leftovers.PREFIX}{tag}{leftovers.ARENA_SUFFIX}")
