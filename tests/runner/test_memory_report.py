"""The memory figures at the end of a CUDA runner's startup and the eager tower's snapshot (`runner.cuda.memory_report`;
`dev.runner.cuda.memory_history`). No CUDA here: torch.cuda's memory calls and the image encoder are stood in for; the
figures themselves are measured on a GPU host."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from llm_mini_infer.config import Config
from llm_mini_infer.runner.cuda import memory_report

GiB = 2**30


def test_cgroup_memory_reads_v2_then_v1(tmp_path) -> None:
    assert memory_report.cgroup_memory(tmp_path) == {}
    (tmp_path / "memory").mkdir()
    (tmp_path / "memory" / "memory.stat").write_text("cache 5\nshmem 7\nrss 3\n")
    assert memory_report.cgroup_memory(tmp_path) == {"shmem": 7}  # v1: its shared memory, no current
    (tmp_path / "memory.stat").write_text("anon 100\nfile 20\nkernel 3\nshmem 5583457484\nslab 9\nunevictable 0\n")
    (tmp_path / "memory.current").write_text("6000000000\n")
    assert memory_report.cgroup_memory(tmp_path) == {"anon": 100, "file": 20, "shmem": 5583457484, "unevictable": 0,
                                                     "kernel": 3, "current": 6000000000}


def test_the_startup_line(monkeypatch, tmp_path) -> None:
    c = torch.cuda
    monkeypatch.setattr(c, "memory_reserved", lambda d=None: 10 * GiB)
    monkeypatch.setattr(c, "memory_allocated", lambda d=None: 9 * GiB)
    monkeypatch.setattr(c, "mem_get_info", lambda d=None: (2 * GiB, 96 * GiB))
    stats = {"allocated_bytes.current": GiB, "reserved_bytes.current": 2 * GiB, "allocations.current": 3}
    monkeypatch.setattr(c, "host_memory_stats", lambda: stats)
    monkeypatch.setattr(memory_report, "cgroup_memory", lambda: {"shmem": 5 * GiB})
    line = memory_report.startup_line(torch.device("cuda"), released=GiB // 2)
    assert line == ("memory at the end of startup: device: 10.00 GiB reserved, 9.00 GiB allocated, 2.00 of 96.00 GiB "
                    "free; after the memory profile's release 0.500 GiB reserved but not allocated; pinned host memory "
                    "(torch's allocator): allocated.current 1.000 GiB, reserved.current 2.000 GiB; cgroup memory: "
                    "shmem 5.00 GiB")
    monkeypatch.delattr(c, "host_memory_stats")  # a torch without it
    monkeypatch.setattr(memory_report, "cgroup_memory", dict)
    assert memory_report.startup_line(torch.device("cuda"), None) == \
        "memory at the end of startup: device: 10.00 GiB reserved, 9.00 GiB allocated, 2.00 of 96.00 GiB free"


def _cuda_stubs(monkeypatch, log: list) -> None:
    c, mem = torch.cuda, torch.cuda.memory
    monkeypatch.setattr(c, "synchronize", lambda d=None: None)
    monkeypatch.setattr(c, "memory_allocated", lambda d=None: 30 * GiB)
    monkeypatch.setattr(c, "memory_reserved", lambda d=None: 30 * GiB + GiB // 2)
    monkeypatch.setattr(c, "max_memory_allocated", lambda d=None: 31 * GiB)
    monkeypatch.setattr(c, "reset_peak_memory_stats", lambda d=None: log.append("reset"))
    monkeypatch.setattr(mem, "_record_memory_history", lambda enabled="all", **kw: log.append(("history", enabled)))
    monkeypatch.setattr(mem, "_dump_snapshot", lambda path: log.append(("dump", path)))


def test_the_eager_tower_runs_under_memory_history(monkeypatch, tmp_path) -> None:
    """One tower call of the step's budget of image tokens (the largest bucket without one call per image), eagerly,
    between the history's start and its dump to the file; the history stops after a failure too. Its peak is the
    allocator's above what was allocated before it."""
    log: list = []
    _cuda_stubs(monkeypatch, log)
    calls: list = []

    def tower(patches, grids, graphs):
        calls.append((tuple(patches.shape), grids, graphs))
        return patches

    img = SimpleNamespace(device=torch.device("cpu"), budget=280, cfg=SimpleNamespace(patch=16, merge=3),
                          per_image=False, max_soft_tokens=70, _px_internal=np.zeros(3 * 30 * 16 * 84 * 16, np.uint8),
                          synthetic=lambda n: ((1, 30, 84), (480, 1344), n),
                          vision=SimpleNamespace(patches=lambda px: px.reshape(-1, 768)), _tower=tower)
    path = str(tmp_path / "tower.pickle")
    fig = memory_report.eager_tower_snapshot(img, path, free_after_startup=2 * GiB)
    assert calls == [((30 * 84, 768), [(1, 30, 84)], False)]
    assert log == ["reset", ("history", "all"), ("dump", path), ("history", None)]
    assert fig == {"tokens": 280, "patches": 30 * 84, "eager_peak_bytes": GiB, "free_after_startup_bytes": 2 * GiB,
                   "allocator_cache_bytes": GiB // 2, "snapshot": path}
    log.clear()
    img._tower = lambda *a, **kw: (_ for _ in ()).throw(MemoryError("out of memory"))
    with pytest.raises(MemoryError):
        memory_report.eager_tower_snapshot(img, path, free_after_startup=2 * GiB)
    assert log == ["reset", ("history", "all"), ("history", None)]


@pytest.mark.parametrize("per_image,soft,want", [(True, 70, 70), (True, None, 2048), (False, 70, 2048)])
def test_with_one_call_per_image_the_eager_tower_takes_the_largest_image(monkeypatch, tmp_path, per_image, soft,
                                                                        want) -> None:
    """With one tower call per image (batch invariance) no call holds more than the largest image (Gemma 4's
    max_soft_tokens, `images.family_buckets`): the snapshot's one call is of that image, its tokens those `synthetic`
    gives. Without one call per image, or without a soft-token budget (Qwen3.8), it is of the step's budget."""
    _cuda_stubs(monkeypatch, [])
    asked, calls = [], []

    def synthetic(n: int):
        asked.append(n)
        return (1, 21, 30), (336, 480), 70

    img = SimpleNamespace(device=torch.device("cpu"), budget=2048, cfg=SimpleNamespace(patch=16, merge=3),
                          per_image=per_image, max_soft_tokens=soft, _px_internal=np.zeros(3 * 336 * 480, np.uint8),
                          synthetic=synthetic, vision=SimpleNamespace(patches=lambda px: px.reshape(-1, 768)),
                          _tower=lambda patches, grids, graphs: calls.append((tuple(patches.shape), grids)) or patches)
    fig = memory_report.eager_tower_snapshot(img, str(tmp_path / "tower.pickle"), free_after_startup=2 * GiB)
    assert asked == [want] and calls == [((630, 768), [(1, 21, 30)])]
    assert (fig["tokens"], fig["patches"]) == (70, 630)


def test_the_setting_is_off_by_default() -> None:
    assert Config().dev.runner.cuda.memory_history is None
