"""Batch-invariant attention on our kernels (docs/architecture.md, "Batch invariance"; `csrc/attn_window.cu`), head dim
256, every query's result a function of its own keys, whatever the batch: Gemma 4's sliding-window layers
(`WindowAttention`; the reference is `models.gemma4.backend.TorchSlidingAttention`), a query at p seeing the keys
[p − window + 1, p], and, with no window, Qwen3.8's full layers (`FullAttention`).

**Why not FlashInfer** (its FA2 prefill kernel, the one SM120 runs). Each CTA
starts its keys at the window start of its query tile's first query, so a query's KV tiles move with the queries beside
it in its row (an MTP verify row's offset, a prefill chunk's start); its query tile, 16 or 64 rows (and so KV tiles of
64 or 16 keys), follows the batch's average query rows (eagerly) or the graph's row bound; and its KV split is a chunk
size searched per batch (always on in graphs). Each moves a query's last bits with the rows beside it, and no plan
argument pins them all: FlashInfer's deterministic modes fix the KV split sizes, which pins the chunk size and nothing
else (the CTA sizes stay open: FlashInfer issue #2424).

**Work** (`WindowAttention`, as `kernels.attn512`'s; `FullAttention`'s: its class docstring): a row is (slot, first
token, n_query, KV length after the step), a work item (row, tile, KV head, j). A tile's keys run from the window start
of its first query to the largest key end of its queries (the causal end, or an image span's end), read in KV tiles at
absolute multiples of 16 positions:
- **decode rows** (device input: a decode or MTP verify row): tiles of 16 query rows, a group the items of one (row,
  tile, KV head), its keys split at absolute multiples of `split_keys`: every group has `per_tile` items, split s =
  (the tile's first key) // split_keys + j (empty past its last key: the merge skips it), and the merge sums them in
  split order. A decode row's queries are few, and the split spreads them over the GPU (one row, 16 KV heads, 6
  splits of 256 keys: 96 CTAs);
- **prefill rows** (explicit tokens): tiles of 64 query rows (32 positions), each taking its keys in one item, on
  the prefill kernel (`csrc/attn_window.cu`): their tiles are many, and a tile 4 times as tall reads the window's
  keys a quarter as often. A prompt position is computed by a prefill row and an answer position by a decode row, so
  each keeps one rule (the exceptions: docs/architecture.md, "Batch invariance").
Masked keys (outside a query's window, past it) change no bit, so a query's result depends on its position and keys
alone: not on the rows beside it, its tile-mates, a row planned for an upper bound of its length, or the graph.

**CUDA graphs** (`WindowAttention`'s decode buckets): every row of a bucket, padding included, has 1 + k queries and the
same items, so a bucket's item and group tables are built once; before each replay `plan_graph` writes its row table
(padding rows: slot 0's null page, 1 + k keys), and the runner's body the true KV lengths under MTP, as for `attn512`.

**Image spans** (prefill rows only: `models.gemma4.backend`, bidirectional image spans): `plan`'s `span_end` gives
each span query its span's end, and the prefill kernel lets it see the span's later keys in the same pass (keys [p −
window + 1, span end)), where FlashInfer runs the span queries again without the causal mask.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from . import build
from .attn512 import (
    MAX_SPLITS,
    PREFILL_FROM,
    ROWS,
    launches,
    plan_items,
    range_splits,
    split_major,
    tile_items,
    tile_positions,
    tiles,
)
from .ops import _check, _p, _stream

D = 256  # the kernels' head dim
KT = 16  # keys a KV tile
PROWS = 64  # query rows a prefill tile (csrc/attn_window.cu)
SPLIT_KEYS = 256  # a decode row's split, in keys (module docstring)
# Full attention's prefill split base past attn512.PREFILL_FROM keys (`FullAttention`): a prefill tile is one CTA an SM,
# so a short chunk over a long history runs on a few dozen CTAs unless split (attn-timing on SM120)
FULL_PREFILL_KEYS = 2048


def prefill_items(nq: np.ndarray, group: int, kv_heads: int, first_row: int) -> np.ndarray:
    """Prefill rows' items int32 [n, 4] (row, tile of `PROWS` query rows, KV head, 0) for rows `first_row`.. of `nq`
    queries each."""
    per_row = (-(-np.asarray(nq, np.int64) * group // PROWS)) * kv_heads
    local = np.arange(int(per_row.sum())) - np.repeat(np.cumsum(per_row) - per_row, per_row)
    row = np.repeat(np.arange(len(per_row)) + first_row, per_row)
    return np.stack([row, local // kv_heads, local % kv_heads, np.zeros_like(row)], axis=1).astype(np.int32)


def splits_per_tile(window: int, positions: int, split_keys: int) -> int:
    """The most absolute splits a tile's keys can meet: a tile of `positions` query positions sees at most window +
    positions − 1 keys, which meet at most ceil(that / split_keys) + 1 splits of `split_keys`."""
    return -(-(window + positions - 1) // split_keys) + 1


class WindowAttention:
    """The sliding layers' backend (`plan`, `plan_graph`, `run`) for `models.gemma4.backend.Gemma4Context`, in place of
    `kernels.flashinfer.FlashInferWindowAttention`'s causal instance."""

    name = "attn_window"
    replans = kept = unsplit = 0  # FlashInfer's chunk-count hazard (`runner.base.sum_attention_counters`): none here

    def __init__(self, pools: Any, num_heads: int, num_kv_heads: int, head_dim: int, device: torch.device,
                 window: int, max_tokens: int, max_rows: int, scale: float = 1.0, qo_per_row: int = 1,
                 split_keys: int = SPLIT_KEYS, share: WindowAttention | None = None, graph_only: bool = False):
        """`window`: keys a query sees, its own included; `max_rows`: decode rows a step holds at most;
        `qo_per_row`: a decode row's queries (1 + k under MTP); `split_keys`: a decode row's split (module
        docstring). `share`: an instance whose split partials this one uses when large enough (one step runs one
        width's sliding backend, and each call merges its splits before the next call's split on the same stream).
        `graph_only`: decode graphs only (`plan_graph`), so no eager tables."""
        if head_dim != D or num_heads % num_kv_heads or ROWS % (num_heads // num_kv_heads):
            raise ValueError(f"attn_window: head dim {D} and query heads per KV head dividing {ROWS} (got head dim "
                             f"{head_dim}, {num_heads} heads over {num_kv_heads})")
        if split_keys < KT or split_keys % KT:
            raise ValueError(f"attn_window: a split of {split_keys} keys (a multiple of {KT})")
        self.pools = pools
        self.h, self.hkv, self.scale = num_heads, num_kv_heads, scale
        self.group = num_heads // num_kv_heads
        self.window_left = window - 1
        self.split_keys = split_keys
        self.per_tile = splits_per_tile(window, ROWS // self.group, split_keys)
        if self.per_tile > MAX_SPLITS:
            raise ValueError(f"attn_window: a window of {window} keys meets {self.per_tile} splits of {split_keys} "
                             f"keys (the merge takes {MAX_SPLITS})")
        self.device, self.qo_per_row = device, qo_per_row
        # Split partials: every decode row's groups at `per_tile` splits (a step's decode rows come first, so do their
        # items)
        self.max_split_items = max(1, max_rows) * tiles(qo_per_row, self.group) * num_kv_heads * self.per_tile
        if share is not None and share.o_part.shape[0] >= self.max_split_items:
            self.o_part, self.ml_part = share.o_part, share.ml_part
        else:
            self.o_part = torch.empty((self.max_split_items, ROWS, D), dtype=torch.float32, device=device)
            self.ml_part = torch.empty((self.max_split_items, ROWS, 2), dtype=torch.float32, device=device)
        # Eager tables, per parity (a step is planned while the previous one may be queued): the decode rows' items and
        # groups, the prefill rows' items, the span queries' key ends
        self.graph_only = graph_only
        max_groups = max(1, max_rows) * tiles(qo_per_row, self.group) * num_kv_heads
        self.max_prefill_items = (-(-max_tokens * self.group // PROWS) + max_rows) * num_kv_heads
        n = 0 if graph_only else 2
        self.rows = [torch.zeros((max_rows + 1, 4), dtype=torch.int32, device=device) for _ in range(n)]
        self.items = [torch.zeros((self.max_split_items, 4), dtype=torch.int32, device=device) for _ in range(n)]
        self.groups = [torch.zeros((max_groups, 2), dtype=torch.int32, device=device) for _ in range(n)]
        self.p_items = [torch.zeros((self.max_prefill_items, 4), dtype=torch.int32, device=device) for _ in range(n)]
        self.kend = [torch.zeros(max_tokens, dtype=torch.int32, device=device) for _ in range(n)]
        self.g_tabs: dict[int, tuple[torch.Tensor, torch.Tensor, int, torch.Tensor, int]] = {}

    @staticmethod
    def staging_bound(max_rows: int, max_tokens: int, group: int, kv_heads: int, qo_per_row: int, window: int,
                      split_keys: int) -> int:
        """The int32 staging one step's plan takes at most (its row, decode item, group and prefill item tables and the
        span ends), for the runner's page-locked buffer."""
        groups = max(1, max_rows) * tiles(qo_per_row, group) * kv_heads
        per_tile = splits_per_tile(window, ROWS // group, split_keys)
        prefill = (-(-max_tokens * group // PROWS) + max_rows) * kv_heads
        return 4 * (max_rows + 1) + 4 * groups * per_tile + 2 * groups + 4 * prefill + max_tokens

    def plan(self, meta: Any, *, parity: int, rows: np.ndarray, stage: Any, fixed_rows: int = 0,
             span_end: np.ndarray | None = None) -> None:
        """Eager steps: `rows` int32 [n, 4] (slot, first token, n_query, KV length after the step; the decode rows'
        planned for their upper bound), the first `fixed_rows` of them decode rows, split; the rest prefill rows, whole
        (module docstring). `span_end`: int32 [the step's tokens], each image-span query's span end, 0 elsewhere (None:
        no spans). `stage` as in `attn512.Attn512.plan`."""
        if self.graph_only:
            raise RuntimeError("attn_window: an instance for decode graphs only plans no eager step")
        nq = rows[:, 2].astype(np.int64)
        dr, di, dg, pi = self.rows[parity], self.items[parity], self.groups[parity], self.p_items[parity]
        dr[:len(rows)].copy_(stage(rows.reshape(-1)).view(-1, 4), non_blocking=True)
        n_items = n_groups = 0
        if fixed_rows:
            items, grp = plan_items(nq[:fixed_rows], self.group, self.hkv, self.per_tile)
            if len(items) > self.max_split_items:
                raise RuntimeError(f"attn_window: {fixed_rows} decode rows over the partials' "
                                   f"{self.max_split_items} items")
            di[:len(items)].copy_(stage(items.reshape(-1)).view(-1, 4), non_blocking=True)
            dg[:len(grp)].copy_(stage(grp.reshape(-1)).view(-1, 2), non_blocking=True)
            n_items, n_groups = len(items), len(grp)
        pre = prefill_items(nq[fixed_rows:], self.group, self.hkv, fixed_rows)
        if len(pre) > self.max_prefill_items:
            raise RuntimeError(f"attn_window: {len(pre)} prefill items over {self.max_prefill_items}")
        if len(pre):
            pi[:len(pre)].copy_(stage(pre.reshape(-1)).view(-1, 4), non_blocking=True)
        kend = None
        if span_end is not None:
            kend = self.kend[parity]
            kend[:len(span_end)].copy_(stage(span_end), non_blocking=True)
        meta.window_plan = (dr, di, n_items, dg, n_groups, pi, len(pre), kend)

    def graph_tables(self, bucket: int) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor, int]:
        """A decode bucket's static row table and its item and group tables, built once (every row 1 + k queries and
        `per_tile` splits), with their sizes."""
        got = self.g_tabs.get(bucket)
        if got is None:
            items, grp = plan_items(np.full(bucket, self.qo_per_row, np.int64), self.group, self.hkv,
                                    self.per_tile)
            got = self.g_tabs[bucket] = (torch.zeros((bucket, 4), dtype=torch.int32, device=self.device),
                                         torch.as_tensor(items, device=self.device), len(items),
                                         torch.as_tensor(grp, device=self.device), len(grp))
        return got

    def plan_graph(self, meta: Any, *, bucket: int, rows: np.ndarray, stage: Any) -> None:
        """Before a decode graph's replay: its row table (int32 [bucket, 4], padding rows included) into the static
        buffer the captured launch reads."""
        dr, it, n_items, gt, n_groups = self.graph_tables(bucket)
        dr.copy_(stage(rows.reshape(-1)).view(-1, 4), non_blocking=True)
        meta.window_plan = (dr, it, n_items, gt, n_groups, None, 0, None)

    def run(self, meta: Any, layer: Any, q: torch.Tensor, out: torch.Tensor) -> None:
        rows, items, n_items, groups, n_groups, p_items, n_pre, kend = meta.window_plan
        if meta.fix_len:  # the true KV lengths, decided on the device (as attn512's)
            rows[:meta.n_rows, 3].copy_(meta.kv_len_dev[:meta.n_rows])
        kv = self.pools.wkv[layer.index]  # [pages, 2, page_size, Hkv, 256]
        wpt = meta.wpage_table
        lib = build.load()
        if n_items:
            _check(lib.lmk_attn_window(
                _p(q), self.h, self.hkv, _p(kv), kv.stride(0), kv.stride(1), kv.shape[2], _p(wpt), wpt.stride(0),
                self.window_left, self.scale, _p(rows), _p(items), n_items, _p(groups), n_groups, self.split_keys,
                _p(out), _p(self.o_part), _p(self.ml_part), _stream(q)), "attn_window")
        if n_pre:
            _check(lib.lmk_attn_window_prefill(
                _p(q), self.h, self.hkv, _p(kv), kv.stride(0), kv.stride(1), kv.shape[2], _p(wpt), wpt.stride(0),
                self.window_left, self.scale, _p(rows), _p(p_items), n_pre, _p(kend), _p(out), _stream(q)),
                "attn_window_prefill")


class FullAttention:
    """Full causal attention on the same kernels (`csrc/attn_window.cu`'s `lmk_attn_full*`): Qwen3.8's full layers
    under batch invariance, the backend (`plan`, `plan_graph`, `fix_lengths`, `run`) its runner gives `ForwardContext`
    in place of `kernels.flashinfer.FlashInferAttention` (docs/architecture.md, "Batch invariance"). Head dim 256, any
    GQA group (Qwen3.8's 6: a tile's rows are (position, head) pairs, so a position's heads may straddle two tiles), no
    window: a query at p sees the keys [0, p], read in KV tiles at absolute multiples of 16 by absolute page from the
    slot's own page table. The output gate and the partial RoPE stay outside: q comes rotated, K and V from the pool.

    **Decode rows** (device input: a decode or MTP verify row) split their keys as `kernels.attn512`'s fixed split does
    (its module docstring): each query at multiples of its own size from `split_keys`, a group taking the most splits
    its rows may need between their lowest possible and planned lengths. **Prefill rows** run in tiles of 64 query
    rows, as `WindowAttention`'s do (module docstring): a tile whose queries all hold at most `prefill_from` keys, or
    that holds one such query, on the unsplit kernel, whose code stays the unsplit path's; a tile with a query past
    that, also in its splits (sizes from `prefill_keys`), each split an item whose partials the prefill merge combines
    in split order for those queries only. `prefill_from` is at least `prefill_keys`, so a query past it takes two
    splits or more and every tile holding it splits. A step's split items past the partials run in more launches."""

    name = "attn_window"
    replans = kept = unsplit = 0  # as `WindowAttention`'s

    def __init__(self, pools: Any, num_heads: int, num_kv_heads: int, head_dim: int, device: torch.device,
                 max_tokens: int, max_rows: int, max_kv: int, scale: float, qo_per_row: int = 1,
                 split_keys: int = SPLIT_KEYS, prefill_from: int = PREFILL_FROM,
                 prefill_keys: int = FULL_PREFILL_KEYS):
        """`max_rows`: decode rows a step holds at most; `max_kv`: the longest row's keys (max_model_len + 1 + k);
        `qo_per_row`: a decode row's queries (1 + k under MTP); `split_keys`: a decode row's split base,
        `prefill_from` and `prefill_keys` a prefill query's (class docstring; the tests make them small)."""
        if head_dim != D or num_heads % num_kv_heads:
            raise ValueError(f"attention on attn_window's kernels: head dim {D} and whole GQA groups (got head dim "
                             f"{head_dim}, {num_heads} heads over {num_kv_heads})")
        for what, keys in (("decode", split_keys), ("prefill", prefill_keys)):
            if keys < KT or keys % KT:
                raise ValueError(f"attention on attn_window's kernels: a {what} split of {keys} keys (a multiple of "
                                 f"{KT})")
        if prefill_from < prefill_keys:  # so a query past it takes 2 splits or more, and every tile holding it splits
            raise ValueError(f"attention on attn_window's kernels: prefill queries whole up to {prefill_from} keys, "
                             f"below the prefill split's {prefill_keys} (a query between them would take one split, "
                             "on the unsplit kernel or in the splits by its tile)")
        self.max_splits = int(range_splits(0, max_kv - 1, split_keys))  # at most MAX_SPLITS
        self.pools = pools
        self.h, self.hkv, self.scale = num_heads, num_kv_heads, scale
        self.group = num_heads // num_kv_heads
        self.split_keys, self.device, self.qo_per_row = split_keys, device, qo_per_row
        self.prefill_from, self.prefill_keys = prefill_from, prefill_keys
        per_row = tiles(qo_per_row, self.group) * num_kv_heads  # a decode row's groups
        self.max_split_items = max(1, max_rows) * per_row * self.max_splits
        # Split partials: the decode rows' [items, 16, D] and the prefill split's [items, 64, D] views of one buffer
        # (their launches run in stream order), the prefill's at least one group's MAX_SPLITS items
        self.max_prefill_split = max(self.max_split_items * ROWS // PROWS, MAX_SPLITS)
        n = max(self.max_split_items * ROWS, self.max_prefill_split * PROWS)
        o_buf = torch.empty(n * D, dtype=torch.float32, device=device)
        ml_buf = torch.empty(n * 2, dtype=torch.float32, device=device)
        self.o_part = o_buf[:self.max_split_items * ROWS * D].view(self.max_split_items, ROWS, D)
        self.ml_part = ml_buf[:self.max_split_items * ROWS * 2].view(self.max_split_items, ROWS, 2)
        self.o_part_p = o_buf[:self.max_prefill_split * PROWS * D].view(self.max_prefill_split, PROWS, D)
        self.ml_part_p = ml_buf[:self.max_prefill_split * PROWS * 2].view(self.max_prefill_split, PROWS, 2)
        # Tables, per parity, as `WindowAttention`'s but without the span ends; the prefill split's items (a group's
        # up to MAX_SPLITS) and groups
        self.max_prefill_items = (-(-max_tokens * self.group // PROWS) + max_rows) * num_kv_heads
        max_groups = max(1, max_rows) * per_row
        self.rows = [torch.zeros((max_rows + 1, 4), dtype=torch.int32, device=device) for _ in range(2)]
        self.items = [torch.zeros((self.max_split_items, 4), dtype=torch.int32, device=device) for _ in range(2)]
        self.groups = [torch.zeros((max_groups, 2), dtype=torch.int32, device=device) for _ in range(2)]
        self.p_items = [torch.zeros((self.max_prefill_items, 4), dtype=torch.int32, device=device) for _ in range(2)]
        self.s_items = [torch.zeros((self.max_prefill_items * MAX_SPLITS, 4), dtype=torch.int32, device=device)
                        for _ in range(2)]
        self.s_groups = [torch.zeros((self.max_prefill_items, 2), dtype=torch.int32, device=device) for _ in range(2)]
        self.s_orders = [torch.zeros(self.max_prefill_items * MAX_SPLITS, dtype=torch.int32, device=device)
                         for _ in range(2)]
        # decode buckets: the static row, item and group tables, and the split counts their item and group tables
        # were last built for
        self.g_tabs: dict[int, tuple[torch.Tensor, torch.Tensor, int, torch.Tensor, int]] = {}
        self.g_last: dict[int, bytes] = {}

    @staticmethod
    def staging_bound(max_rows: int, max_tokens: int, group: int, kv_heads: int, qo_per_row: int, split_keys: int,
                      max_kv: int) -> int:
        """The int32 staging one step's plan takes at most (its row, item, group and prefill item tables, the prefill
        split's items, their split-major orders and its groups), for the runner's page-locked buffer."""
        groups = max(1, max_rows) * tiles(qo_per_row, group) * kv_heads
        prefill = (-(-max_tokens * group // PROWS) + max_rows) * kv_heads
        n_splits = int(range_splits(0, max_kv - 1, split_keys))
        return 4 * (max_rows + 1) + 4 * groups * n_splits + 2 * groups + 4 * prefill * (1 + MAX_SPLITS) + \
            prefill * MAX_SPLITS + 2 * prefill

    def plan(self, meta: Any, *, parity: int, rows: np.ndarray, stage: Any, fixed_rows: int = 0,
             lb: np.ndarray | None = None) -> None:
        """Eager steps and breakable graphs: `rows` int32 [n, 4] (slot, first token, n_query, KV length after the step;
        the decode rows' planned for their upper bound), the first `fixed_rows` of them decode rows, split by their
        queries' positions from their lowest possible KV length `lb` (per row; None: the planned ones); the rest
        prefill rows (class docstring). `stage` as in `attn512.Attn512.plan`. The plan goes into `meta.attn_plan`."""
        nq, kv = rows[:, 2].astype(np.int64), rows[:, 3].astype(np.int64)
        dr, di, dg, pi = self.rows[parity], self.items[parity], self.groups[parity], self.p_items[parity]
        dr[:len(rows)].copy_(stage(rows.reshape(-1)).view(-1, 4), non_blocking=True)
        n_items = n_groups = 0
        if fixed_rows:
            d = slice(0, fixed_rows)
            low = kv[d] if lb is None else np.asarray(lb, np.int64)[d]
            items, grp = plan_items(nq[d], self.group, self.hkv, range_splits(low - nq[d], kv[d] - 1, self.split_keys))
            if len(items) > self.max_split_items:
                raise RuntimeError(f"attention: {fixed_rows} decode rows over the partials' {self.max_split_items} "
                                   "items")
            di[:len(items)].copy_(stage(items.reshape(-1)).view(-1, 4), non_blocking=True)
            dg[:len(grp)].copy_(stage(grp.reshape(-1)).view(-1, 2), non_blocking=True)
            n_items, n_groups = len(items), len(grp)
        # prefill rows: each tile's split count; the tiles with a query taken whole (or all) on the unsplit kernel
        pnq, pkv = nq[fixed_rows:], kv[fixed_rows:]
        first, last = tile_positions(pnq, pkv, self.group, PROWS)
        n_tile = range_splits(first, last, self.prefill_keys, self.prefill_from)
        whole = (n_tile == 1) | (first + 1 <= self.prefill_from)
        pre = prefill_items(pnq, self.group, self.hkv, fixed_rows)
        pre = pre[np.repeat(whole, self.hkv)]
        if len(pre) > self.max_prefill_items:
            raise RuntimeError(f"attention: {len(pre)} prefill items over {self.max_prefill_items}")
        if len(pre):
            pi[:len(pre)].copy_(stage(pre.reshape(-1)).view(-1, 4), non_blocking=True)
        runs: list[tuple[int, int, int, int]] = []
        if (n_tile > 1).any():  # the split tiles' items, in launches within the prefill partials
            items, grp = tile_items(pnq, self.group, self.hkv, np.where(n_tile > 1, n_tile, 0), PROWS)
            keep = grp[:, 1] > 0
            items[:, 0] += fixed_rows
            grp = grp[keep]
            si, sg, so = self.s_items[parity], self.s_groups[parity], self.s_orders[parity]
            si[:len(items)].copy_(stage(items.reshape(-1)).view(-1, 4), non_blocking=True)
            for i0, ni, g0, ng in launches(grp, self.max_prefill_split):
                rel = grp[g0:g0 + ng].copy()
                rel[:, 0] -= i0
                sg[g0:g0 + ng].copy_(stage(rel.reshape(-1)).view(-1, 2), non_blocking=True)
                so[i0:i0 + ni].copy_(stage(split_major(items[i0:i0 + ni])), non_blocking=True)  # (as attn512's)
                runs.append((i0, ni, g0, ng))
        meta.attn_plan = (dr, di, n_items, dg, n_groups, pi, len(pre), tuple(runs), parity)

    def graph_tables(self, bucket: int) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor, int]:
        """A decode bucket's static row table and its item and group tables at its capacity (every row at
        `max_splits`), with their sizes."""
        got = self.g_tabs.get(bucket)
        if got is None:
            n_groups = bucket * tiles(self.qo_per_row, self.group) * self.hkv
            cap = n_groups * self.max_splits
            got = self.g_tabs[bucket] = (torch.zeros((bucket, 4), dtype=torch.int32, device=self.device),
                                         torch.full((cap, 4), -1, dtype=torch.int32, device=self.device), cap,
                                         torch.full((n_groups, 2), -1, dtype=torch.int32, device=self.device),
                                         n_groups)
            self.g_last[bucket] = b""
        return got

    def plan_graph(self, meta: Any, *, bucket: int, rows: np.ndarray, stage: Any, real: int,
                   lb: np.ndarray | None = None) -> None:
        """Before a decode graph's replay: its row table (int32 [bucket, 4], padding rows included) into the static
        buffer the captured launch reads; the item and group tables when its first `real` rows' split counts changed
        (from their lowest possible KV lengths `lb`, None: the planned ones; padding rows take one split: their KV is
        the null page's 1 + k positions)."""
        dr, it, cap, gt, n_groups = self.graph_tables(bucket)
        dr.copy_(stage(rows.reshape(-1)).view(-1, 4), non_blocking=True)
        nq, kv = rows[:real, 2].astype(np.int64), rows[:real, 3].astype(np.int64)
        low = kv if lb is None else np.asarray(lb, np.int64)[:real]
        ns = np.ones(bucket, np.int64)
        ns[:real] = range_splits(low - nq, kv - 1, self.split_keys)
        key = ns.tobytes()
        if self.g_last[bucket] != key:  # stream order: replays already queued read the old tables first
            items, grp = plan_items(np.full(bucket, self.qo_per_row, np.int64), self.group, self.hkv, ns)
            items = np.concatenate([items, np.full((cap - len(items), 4), -1, np.int32)])
            it.copy_(stage(items.reshape(-1)).view(-1, 4), non_blocking=True)
            gt.copy_(stage(grp.reshape(-1)).view(-1, 2), non_blocking=True)
            self.g_last[bucket] = key
        meta.attn_plan = (dr, it, cap, gt, n_groups, None, 0, (), 0)

    def fix_lengths(self, plan: Any, rows: int, true_len: torch.Tensor) -> None:
        """Rewrite rows [0, rows)'s KV lengths in the plan's row table from their true lengths on the device (int64
        [rows]; MTP's upper-bound plans); graph-safe."""
        plan[0][:rows, 3].copy_(true_len[:rows])

    def graph_page_indices(self, rows: int, page_table: torch.Tensor, slots: torch.Tensor) -> None:
        """FlashInfer's packed page indices in a decode graph: none here (the kernels read the page table)."""

    def run(self, meta: Any, index: int, q: torch.Tensor, out: torch.Tensor) -> None:
        rows, items, n_items, groups, n_groups, p_items, n_pre, runs, parity = meta.attn_plan
        if meta.fix_len:  # the true KV lengths, decided on the device (as attn512's)
            rows[:meta.n_rows, 3].copy_(meta.kv_len_dev[:meta.n_rows])
        kv = self.pools.kv[index]  # [pages, 2, page_size, Hkv, 256]
        pt = meta.page_table
        lib = build.load()
        if n_items:
            _check(lib.lmk_attn_full(
                _p(q), self.h, self.hkv, _p(kv), kv.stride(0), kv.stride(1), kv.shape[2], _p(pt), pt.stride(0),
                self.scale, _p(rows), _p(items), n_items, _p(groups), n_groups, self.split_keys, _p(out),
                _p(self.o_part), _p(self.ml_part), _stream(q)), "attn_full")
        if n_pre:
            _check(lib.lmk_attn_full_prefill(
                _p(q), self.h, self.hkv, _p(kv), kv.stride(0), kv.stride(1), kv.shape[2], _p(pt), pt.stride(0),
                self.scale, _p(rows), _p(p_items), n_pre, _p(out), _stream(q)), "attn_full_prefill")
        si, sg, so = self.s_items[parity], self.s_groups[parity], self.s_orders[parity]
        for i0, ni, g0, ng in runs:  # after the unsplit tiles: a straddling tile's split queries are written last
            _check(lib.lmk_attn_full_prefill_split(
                _p(q), self.h, self.hkv, _p(kv), kv.stride(0), kv.stride(1), kv.shape[2], _p(pt), pt.stride(0),
                self.scale, _p(rows), _p(si[i0:]), ni, _p(sg[g0:]), ng, self.prefill_from, self.prefill_keys,
                _p(so[i0:]), _p(out), _p(self.o_part_p), _p(self.ml_part_p), _stream(q)), "attn_full_prefill_split")
