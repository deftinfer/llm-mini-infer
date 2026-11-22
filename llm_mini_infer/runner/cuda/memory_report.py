"""Memory figures at the end of a CUDA runner's startup (one INFO line), and the dev snapshot of an eager tower call
(`dev.runner.cuda.memory_history`).

- **The device:** the caching allocator's reserved and allocated bytes, and the reserved bytes the memory profile's
  release left unallocated (`released`: segments a fragmented cache keeps; vllm-project/vllm#58430).
- **The host:** torch's pinned host allocator (`torch.cuda.host_memory_stats`), and the cgroup's `memory.stat`: its
  anonymous, file and shared memory, and what it cannot evict.
- **The eager tower** (with images, when `memory_history` names a file): CUDA memory history recorded through one tower
  call at the largest bucket run eagerly, its snapshot dumped to that file (`torch.cuda.memory._dump_snapshot`), and
  its peak above the memory allocated before it, against the device's free memory once the KV arena took its share
  plus what the allocator held unallocated: the memory profile measures the tower through its graphs only.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Any

import torch

log = logging.getLogger(__name__)

GiB = 2**30
# memory.stat's fields that the line reports (cgroup v2; v1 names its shared memory `shmem` too)
CGROUP_FIELDS = ("anon", "file", "shmem", "file_mapped", "unevictable", "kernel")


def cgroup_memory(root: Path = Path("/sys/fs/cgroup")) -> dict[str, int]:
    """The cgroup's memory.stat (v2, else v1's), the CGROUP_FIELDS it has, and memory.current; {} without one."""
    for stat, current in ((root / "memory.stat", root / "memory.current"),
                          (root / "memory" / "memory.stat", root / "memory" / "memory.usage_in_bytes")):
        try:
            fields = dict(line.split()[:2] for line in stat.read_text().splitlines() if line.strip())
        except OSError:
            continue
        out = {k: int(fields[k]) for k in CGROUP_FIELDS if k in fields}
        with contextlib.suppress(OSError, ValueError):
            out["current"] = int(current.read_text())
        return out
    return {}


def host_memory() -> dict[str, int]:
    """torch's pinned host allocator: its allocated and reserved bytes now and at their peak ({} if this torch has no
    `host_memory_stats`)."""
    stats = getattr(torch.cuda, "host_memory_stats", None)
    if stats is None:
        return {}
    s = stats()
    return {k: int(s[k]) for k in ("allocated_bytes.current", "allocated_bytes.peak", "reserved_bytes.current",
                                   "reserved_bytes.peak") if k in s}


def startup_line(device: torch.device, released: int | None) -> str:
    """The memory figures at the end of startup (module docstring), one line."""
    reserved, allocated = torch.cuda.memory_reserved(device), torch.cuda.memory_allocated(device)
    free, total = torch.cuda.mem_get_info(device)
    host = host_memory()
    cg = cgroup_memory()
    parts = [f"device: {reserved / GiB:.2f} GiB reserved, {allocated / GiB:.2f} GiB allocated, {free / GiB:.2f} of "
             f"{total / GiB:.2f} GiB free"]
    if released is not None:
        parts.append(f"after the memory profile's release {released / GiB:.3f} GiB reserved but not allocated")
    if host:
        parts.append("pinned host memory (torch's allocator): " + ", ".join(
            f"{k.replace('_bytes', '')} {v / GiB:.3f} GiB" for k, v in host.items()))
    if cg:
        parts.append("cgroup memory: " + ", ".join(f"{k} {v / GiB:.2f} GiB" for k, v in cg.items()))
    return "memory at the end of startup: " + "; ".join(parts)


def eager_tower_snapshot(img: Any, path: str, free_after_startup: int) -> dict[str, Any]:
    """One tower call of the largest bucket's patches (`img`: a runner's `ImageEncoder`), eagerly, under CUDA memory
    history; the snapshot dumped to `path`. Returns its figures (also logged)."""
    dev = img.device
    # the largest call (`images.family_buckets`): the largest image with one tower call per image (Gemma 4's
    # max_soft_tokens), else the step's budget of image tokens in one call
    tokens = min(img.budget, img.max_soft_tokens) if img.per_image and img.max_soft_tokens else img.budget
    (_, gh, gw), _, tokens = img.synthetic(tokens)
    p = img.cfg.patch
    nbytes = 3 * gh * p * gw * p
    torch.cuda.synchronize(dev)
    base = torch.cuda.memory_allocated(dev)
    cache = torch.cuda.memory_reserved(dev) - base  # the allocator's reserved, unallocated memory: drawn on first
    torch.cuda.reset_peak_memory_stats(dev)
    torch.cuda.memory._record_memory_history(max_entries=200_000)
    try:
        with torch.inference_mode():
            px = torch.from_numpy(img._px_internal[:nbytes].copy()).to(dev)
            patches = img.vision.patches(px.view(3, gh * p, gw * p))
            out = img._tower(patches, [(1, gh, gw)], graphs=False)
            del px, patches, out
        torch.cuda.synchronize(dev)
        peak = torch.cuda.max_memory_allocated(dev) - base
        torch.cuda.memory._dump_snapshot(path)
    finally:
        torch.cuda.memory._record_memory_history(enabled=None)
    room = free_after_startup + cache
    fig = {"tokens": tokens, "patches": gh * gw, "eager_peak_bytes": peak, "free_after_startup_bytes":
           free_after_startup, "allocator_cache_bytes": cache, "snapshot": path}
    log.info("eager tower at the largest bucket (%d image tokens, %d patches): peak %.3f GiB above the memory "
             "allocated before it, against %.3f GiB free once the KV arena took its share and %.3f GiB the allocator "
             "held unallocated (%s); snapshot: %s", tokens, gh * gw, peak / GiB, free_after_startup / GiB, cache / GiB,
             "it fits" if peak < room else "it does not fit", path)
    return fig
