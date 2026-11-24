"""Gemma's KV arena (runner/cuda/gemma.py `_Arena`) on the CPU: its page copy moves both overlaid views; its layout in
bytes, the full layers in FP8 included."""

from __future__ import annotations

import json
import types

import pytest
import torch

from llm_mini_infer.runner.cuda.gemma import _Arena


def arena_cfg() -> types.SimpleNamespace:
    """4 sliding layers and 2 full layers (groups = 2), a window page of 4 positions: the chunk is 128 elements and
    holds 8 positions of a full layer (sub = 8)."""
    return types.SimpleNamespace(sliding_layers=range(4), full_layers=range(2), sliding_kv_heads=2, sliding_head_dim=8,
                                 full_kv_heads=1, full_head_dim=8, k_eq_v=False)


def test_arena_copy_moves_both_views_of_a_page() -> None:
    """`_Arena.copy` copies whole pages of the one backing tensor. The full layers' sub-page view and the sliding
    layers' window-page view are both full views of it, so a copy through either moves every byte; what the test
    catches is a copy that misses part of a page (a slab, sub-pages indexed as pages, or one layer kind's region).
    One page is filled through both views with distinct values (the window view sets the sliding layers 0 and 1, the
    sub-page view the bytes of full layer 1, the slabs 2 and 3), copied onto another page; the destination reads the
    source's values through both views, and the source and every other page are as they were."""
    a = _Arena(arena_cfg(), 4, 6, torch.float32, torch.device("cpu"))
    assert (a.groups, a.sub, a.chunk) == (2, 8, 128)
    pages, src, dst = list(range(a.num_pages)), 1, 4
    for p in pages:  # every page holds something, so a stray write shows
        a.mem[:, p] = -float(p + 1)
    # The sub-page view as [full layer, group, page, ...]: sub-page j·num_pages + p is chunk p of slab j.
    fkv = a.fkv.reshape(2, a.groups, a.num_pages, *a.fkv.shape[2:])
    wval = torch.arange(a.wkv[:2, src].numel(), dtype=torch.float32).reshape(a.wkv[:2, src].shape) + 100.0
    fval = torch.arange(fkv[1, :, src].numel(), dtype=torch.float32).reshape(fkv[1, :, src].shape) + 1000.0
    a.wkv[:2, src] = wval  # window view: sliding layers 0 and 1, page src
    fkv[1, :, src] = fval  # sub-page view: full layer 1 (slabs 2 and 3), page src
    before = a.mem.clone()
    a.copy(torch.tensor([src]), torch.tensor([dst]))
    assert torch.equal(a.wkv[:2, dst], wval), "window view not copied"
    assert torch.equal(fkv[1, :, dst], fval), "sub-page view not copied"
    assert torch.equal(a.mem[:, dst], before[:, src]), "the page's bytes are not the source's"
    others = [p for p in pages if p != dst]
    assert torch.equal(a.mem[:, others], before[:, others]), "another page changed"


def gemma31b_cfg():
    from llm_mini_infer.models.gemma4.model import Gemma4Config

    from ..support.paths import ROOT

    return Gemma4Config.from_hf(json.loads((ROOT / "tests/data/gemma-4-31b-it-fp8-block.config.json").read_text()))


def test_the_arena_counts_in_bytes() -> None:
    """`_Arena.layout` divides a chunk's bytes by a full position's: the same sub-pages for any one dtype (BF16 and
    fp32 alike), twice the positions when the full layers' KV is FP8 in a BF16 buffer; Gemma 4 31B's full page is 320
    positions in BF16 and 640 in FP8 (12.5 MiB either way: 40 KiB or 20 KiB a position). A dtype other than the
    buffer's or FP8 is refused."""
    cfg = gemma31b_cfg()
    assert _Arena.layout(cfg, 16) == _Arena.layout(cfg, 16, 4) == (5, 64, 2 * 16 * 16 * 256)
    assert _Arena.layout(cfg, 16, 2, torch.float8_e4m3fn) == (5, 128, 2 * 16 * 16 * 256)
    assert _Arena.full_page_size(cfg, 16) == 320 and _Arena.full_page_size(cfg, 16, 2, torch.float8_e4m3fn) == 640
    assert _Arena.layout(arena_cfg(), 4, 4) == (2, 8, 128)
    with pytest.raises(ValueError, match="buffer's dtype"):
        _Arena(arena_cfg(), 4, 6, torch.bfloat16, torch.device("cpu"), torch.float16)


def test_an_fp8_arena_views_the_same_bytes_and_copies_both_kinds() -> None:
    """With the full layers in FP8 (a BF16 buffer): `fkv` is a uint8 view of the buffer's bytes, [full layers,
    groups·pages, sub, heads, dim] with twice the BF16 view's sub; full page p (its sub-pages) is window page p's bytes
    and no other page's; `_Arena.copy` moves both kinds of page, FP8 bytes and BF16 values alike."""
    cfg = types.SimpleNamespace(**{**vars(arena_cfg()), "k_eq_v": True})
    a = _Arena(cfg, 4, 6, torch.bfloat16, torch.device("cpu"), torch.float8_e4m3fn)
    bf = _Arena(cfg, 4, 6, torch.bfloat16, torch.device("cpu"))
    assert a.fkv.dtype == torch.uint8 and a.wkv.dtype == a.mem.dtype == torch.bfloat16
    assert (a.groups, a.sub, a.page_size) == (2, 2 * bf.sub, 2 * bf.page_size) == (2, 32, 64)
    assert a.fkv.shape == (2, a.groups * a.num_pages, a.sub, 1, 8) and a.fkv.data_ptr() == a.mem.data_ptr()
    assert a.fkv.numel() == a.mem.numel() * 2
    fkv = a.fkv.reshape(2, a.groups, a.num_pages, -1)  # [full layer, sub-page group, page, bytes]
    a.wkv[:, 3].fill_(1.0)
    assert bool((fkv[:, :, 3] != 0).all()) and not bool(fkv[:, :, [p for p in range(6) if p != 3]].any())
    a.mem.zero_()
    src, dst = 1, 4
    fval = torch.arange(fkv[1, :, src].numel(), dtype=torch.int64).remainder(251).to(torch.uint8) + 1
    fkv[1, :, src] = fval.reshape(fkv[1, :, src].shape)
    a.wkv[:2, src] = 7.0
    before = a.mem.clone()
    a.copy(torch.tensor([src]), torch.tensor([dst]))
    assert torch.equal(fkv[1, :, dst], fkv[1, :, src]) and bool((a.wkv[:2, dst] == 7.0).all())
    assert torch.equal(a.mem[:, dst], before[:, src])
    assert torch.equal(a.mem[:, [p for p in range(6) if p != dst]], before[:, [p for p in range(6) if p != dst]])
