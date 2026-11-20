"""Gemma 4's full-attention layers on CUDA: our paged attention kernel for head dim 512 with K derived from the stored
V plane (`csrc/attn512.cu`; the reference is `models.gemma4.backend.TorchFullAttention`).

**Planning** (`Attn512.plan`, on the host from the step's planned lengths: under MTP each decode row's upper bound,
below). A row is (slot, first token, n_query, KV length after the step); a work item is (row, tile of 16 query rows, KV
head, split), and a group the items of one (row, tile, KV head). The plan copies its row, item and group tables to the
device in stream order, from the runner's page-locked staging (two parities, as FlashInfer's plans: a step is planned
while the one before may still be queued).

**The split** (`choose_splits`: the wave split). A launch with fewer groups than the GPU holds CTAs at once splits each
group's keys, so more of them run: as many splits as fill one wave of resident CTAs, floor(resident / groups)
(`resident` from the CUDA occupancy API: 2 CTAs to an SM on SM120). A wave left under `WAVE_FILL` (60%) full takes
about 5/3 of a wave instead; rounding up past a wave is never taken, since a second wave of a few items nearly doubles
the kernel. At most 32 splits (the merge's lanes) and, on eager plans, each at least `MIN_SPLIT_KEYS` keys of the
longest row. In a decode graph the split count comes from the replay's real rows: a bucket's padding rows get one item
per group (their KV is the null page's 1 + k positions). The split count sets the merge order, so another count gives
other last bits.

**The fixed split** (`split_keys` > 0; docs/architecture.md, "Batch invariance"). The wave split makes a row's merge
order depend on the rows beside it, and its chunks on its tile's causal end. With a fixed split each query's splits
come from its own absolute position p alone (`split_size`, `query_splits`): a decode row's query splits its keys [0, p]
at absolute multiples of `split_keys` · 2^m, the least m >= 0 that keeps them within the merge's `MAX_SPLITS` (so 256
keys a split up to 8,192 keys, 512 up to 16,384, ...); a prefill row's query takes its keys whole while they are at
most `prefill_from`, and past that splits them as a decode query does, from `prefill_keys`. Split j of a query holds the
keys [j × S, (j + 1) × S) of its size S, whatever the batch, the tile, the chunk it lands in or the row's planned
length. A group (row, tile, KV head) takes the most splits any position its rows may hold needs (`range_splits`: under
MTP a decode row's queries lie anywhere from its lowest possible length to its planned one, and the count drops where
the size doubles); a split past a row's keys is empty, and the merge skips it. A tile whose rows straddle a size
boundary loops the union of its rows' ranges, each row masked to its own. A step whose split items pass the partials
runs them in more launches (`plan`'s launches), each its groups' items and merge: the same arithmetic either way.

**CUDA graphs** (decode buckets): every row has `qo_per_row` queries (one, or the 1 + k of an MTP verify row). The
captured launches read static row, item and group tables of a bucket's capacity (padding items exit at once); before
each replay `plan_graph` rewrites the row table, and the item and group tables when they changed: its real rows under
the wave split, its rows' split counts under the fixed split.

**Lengths on the device (MTP).** The kernel reads each row's KV length from the device row table and computes its
splits' key ranges from it at run time (an empty split keeps m = −inf and the merge skips it), so a plan made for an
upper bound runs exactly once the table's lengths are rewritten on the device: `run` does it from `meta.kv_len_dev`
when `meta.fix_len` is set (the drafter's rows), the runner's step body otherwise.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from . import build
from .ops import _check, _p, _stream

ROWS = 16  # query rows per tile
KT = 16  # keys per KV tile
MIN_SPLIT_KEYS = 256  # an eager plan's splits hold at least this many keys of the longest row
MAX_SPLITS = 32  # the merge's lanes (csrc/attn512.cu MAX_MERGE_SPLITS)
WAVE_FILL = 0.6  # a wave left under this share full takes about 5/3 of a wave (module docstring; fitted on SM120)
SPLIT_KEYS = 256  # the fixed split's base for decode rows (module docstring, `split_size`)
PREFILL_FROM = 8192  # the fixed split: a prefill query with at most this many keys takes them whole (module docstring)
# The fixed split's base for a prefill query past PREFILL_FROM (`split_size`): its full layers' prefill tiles of 2
# positions already fill the GPU from a few hundred queries, so a split pays for a short chunk over a long history
# only, and a smaller base costs a long chunk more than 3% (attn-timing on SM120)
PREFILL_KEYS = 16384


def tiles(n_query: int, group: int) -> int:
    return -(-n_query * group // ROWS)


def tile_items(nq: np.ndarray, group: int, kv_heads: int, n_tile: np.ndarray,
               rows: int = ROWS) -> tuple[np.ndarray, np.ndarray]:
    """(items int32 [n_items, 4]: (row, tile, KV head, split | n << 16); groups int32 [n_groups, 2]: (first item, n)),
    grouped by (row, tile, KV head) with a group's n splits consecutive; `n_tile`: the split count of each (row, tile),
    in that order, for all of the tile's KV heads; `rows`: a tile's query rows."""
    nq = np.asarray(nq, np.int64)
    t_row = -(-nq * group // rows)  # tiles per row
    n_tile = np.asarray(n_tile, np.int64)
    if len(n_tile) != int(t_row.sum()):
        raise ValueError(f"attn512: {len(n_tile)} tile split counts for {int(t_row.sum())} tiles")
    g_row = np.repeat(np.arange(len(nq)), t_row * kv_heads)
    g_local = np.arange(len(g_row)) - np.repeat(np.cumsum(t_row * kv_heads) - t_row * kv_heads, t_row * kv_heads)
    g_ns = np.repeat(n_tile, kv_heads)
    first = np.cumsum(g_ns) - g_ns
    i_grp = np.repeat(np.arange(len(g_row)), g_ns)
    split = np.arange(len(i_grp)) - first[i_grp]
    items = np.stack([g_row[i_grp], g_local[i_grp] // kv_heads, g_local[i_grp] % kv_heads,
                      split | (g_ns[i_grp] << 16)], axis=1).astype(np.int32)
    return items, np.stack([first, g_ns], axis=1).astype(np.int32)


def plan_items(nq: np.ndarray, group: int, kv_heads: int, n_splits: int | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`tile_items` with `n_splits` per row, or one for all."""
    nq = np.asarray(nq, np.int64)
    ns = np.broadcast_to(np.asarray(n_splits, np.int64), nq.shape)
    return tile_items(nq, group, kv_heads, np.repeat(ns, -(-nq * group // ROWS)))


def choose_splits(n_groups: int, max_kv: int, sms: int, resident: int = 0) -> int:
    """Splits per group (module docstring, the split). `resident`: CTAs the GPU holds at once (0: 2 × SMs)."""
    r = resident or 2 * sms
    if n_groups >= r:
        return 1
    s = r // n_groups  # one wave
    if n_groups * s < WAVE_FILL * r:  # it would leave CTAs idle: about 5/3 of a wave (module docstring)
        s = (10 * r + 3 * n_groups) // (6 * n_groups)  # 5r / 3g rounded: 3 (r / 2 < g < 0.6 r here)
    return max(1, min(s, MAX_SPLITS, max(1, max_kv // MIN_SPLIT_KEYS)))


def bucket_splits(real: int, per_row: int, sms: int, resident: int, max_items: int) -> int:
    """The splits of each real row's group in a decode graph's replay of `real` real rows, each `per_row` groups
    (padding rows one item a group: module docstring), within `max_items` split items."""
    s = choose_splits(real * per_row, 1 << 20, sms, resident)
    return max(1, min(s, max_items // (real * per_row)))


def split_size(pos: np.ndarray | int, keys: int) -> np.ndarray:
    """The fixed split of a query at absolute position `pos` (its keys [0, pos]): `keys` · 2^m, the least m >= 0 that
    keeps its pos + 1 keys within `MAX_SPLITS` splits (module docstring)."""
    pos = np.asarray(pos, np.int64)
    s = np.full(pos.shape, keys, np.int64)
    while (big := pos + 1 > s * MAX_SPLITS).any():
        s = np.where(big, 2 * s, s)
    return s


def query_splits(pos: np.ndarray | int, keys: int, whole: int = 0) -> np.ndarray:
    """A query's split count under the fixed split: one when its pos + 1 keys are at most `whole` (a prefill query
    taking them whole), else ceil((pos + 1) / `split_size`)."""
    pos = np.asarray(pos, np.int64)
    return np.where(pos + 1 <= whole, 1, -(-(pos + 1) // split_size(pos, keys)))


def range_splits(lo: np.ndarray | int, hi: np.ndarray | int, keys: int, whole: int = 0) -> np.ndarray:
    """The most splits a query at any position in [lo, hi] takes (`query_splits`): the count rises with the position
    within a size and drops where the size doubles, so it is the count at hi, or `MAX_SPLITS` when the range crosses a
    size boundary (the position just before it takes MAX_SPLITS)."""
    lo, hi = np.broadcast_arrays(np.asarray(lo, np.int64), np.asarray(hi, np.int64))
    split = np.maximum(lo, whole)  # the first position that splits
    n = np.where(split_size(split, keys) == split_size(hi, keys), query_splits(hi, keys, whole), MAX_SPLITS)
    return np.where(hi + 1 <= whole, 1, n)


def tile_positions(nq: np.ndarray, kv: np.ndarray, group: int, rows: int = ROWS) -> tuple[np.ndarray, np.ndarray]:
    """Each (row, tile)'s first and last query position, in (row, tile) order, for rows of `nq` queries ending at KV
    length `kv` (a tile is `rows` flattened (position, head) rows, position-major, the last tile cut at nq)."""
    nq, kv = np.asarray(nq, np.int64), np.asarray(kv, np.int64)
    t_row = -(-nq * group // rows)
    t = np.arange(int(t_row.sum())) - np.repeat(np.cumsum(t_row) - t_row, t_row)
    q0 = np.repeat(kv - nq, t_row)
    last = np.minimum((t * rows + rows - 1) // group, np.repeat(nq, t_row) - 1)
    return q0 + t * rows // group, q0 + last


def launches(groups: np.ndarray, cap: int) -> list[tuple[int, int, int, int]]:
    """A step's launches under the fixed split: (first item, items, first group, groups) each, consecutive groups whose
    items' indexes in their launch stay within `cap` partials (one launch when every group fits, or none splits)."""
    n_items = int(groups[-1, 0] + groups[-1, 1]) if len(groups) else 0
    if n_items <= cap or not (groups[:, 1] > 1).any():
        return [(0, n_items, 0, len(groups))]
    end = groups[:, 0] + groups[:, 1]
    out, g = [], 0
    while g < len(groups):
        start = int(groups[g, 0])
        h = max(g + 1, int(np.searchsorted(end, start + cap, side="right")))
        out.append((start, int(end[h - 1]) - start, g, h - g))
        g = h
    return out


def bucket_capacity(bucket: int, per_row: int, sms: int, resident: int, max_items: int) -> tuple[int, bool]:
    """The most items any real row count makes in a decode bucket, and whether any splits (the captured launch's
    capacity and kernel)."""
    cap, split = bucket * per_row, False
    for real in range(1, bucket + 1):
        s = bucket_splits(real, per_row, sms, resident, max_items)
        cap = max(cap, real * per_row * s + (bucket - real) * per_row)
        split |= s > 1
    return cap, split


class Attn512:
    """The full-attention backend (`plan`, `plan_graph`, `run`) for `models.gemma4.backend.Gemma4Context`."""

    name = "attn512"

    def __init__(self, pools: Any, rotary: Any, num_heads: int, num_kv_heads: int, rot: int, device: torch.device,
                 max_tokens: int, max_rows: int, scale: float = 1.0, qo_per_row: int = 1, share: Attn512 | None = None,
                 split_keys: int = 0, max_kv: int = 0, prefill_from: int = PREFILL_FROM,
                 prefill_keys: int = PREFILL_KEYS):
        """`qo_per_row`: a decode graph row's queries (1 + k under MTP). `share`: an instance whose split partials this
        one uses when large enough (the MTP drafter's attention runs after the target's, on the same stream).
        `split_keys`: the fixed split's base for decode rows (module docstring; 0: the wave split), for rows of up to
        `max_kv` keys; `prefill_from` and `prefill_keys`: a prefill query's (the tests make them small)."""
        self.pools = pools
        self.cos, self.sin = rotary.cos["full_attention"], rotary.sin["full_attention"]
        self.h, self.hkv, self.rot, self.scale = num_heads, num_kv_heads, rot, scale
        self.group = num_heads // num_kv_heads
        self.sms = torch.cuda.get_device_properties(device).multi_processor_count
        lib = build.load()
        occ = lib.lmk_attn512_occupancy() if lib is not None else -1
        self.resident = (occ if occ > 0 else 2) * self.sms  # CTAs at once (2 an SM on SM120)
        self.device = device
        self.qo_per_row = qo_per_row
        self.split_keys, self.max_splits = split_keys, 0
        self.prefill_from, self.prefill_keys = prefill_from, prefill_keys
        if split_keys:
            if split_keys % KT or prefill_keys < KT or prefill_keys % KT or prefill_from < 0 or max_kv < 1:
                raise ValueError(f"attn512: a fixed split of {split_keys} keys and {prefill_keys} past "
                                 f"{prefill_from} (multiples of {KT}) for rows of {max_kv} keys")
            self.max_splits = int(range_splits(0, max_kv - 1, split_keys))  # a decode row's, at most MAX_SPLITS
        per_row = max(1, max_rows * tiles(qo_per_row, self.group) * num_kv_heads)
        # Split partials for the most items a split launch can have: at most one wave of split items, or under
        # WAVE_FILL 3 splits of fewer than 0.6 × resident groups (at most 9 × resident // 5); every row of a
        # decode-sized step's tiles, times MAX_SPLITS, caps it for small steps. The fixed split: every decode row's
        # groups at the most splits a row takes (its rows come first in a step, so their items do)
        max_items = min(per_row * MAX_SPLITS, 9 * self.resident // 5 + 2 * MAX_SPLITS)
        if split_keys:
            max_items = max(max_items, per_row * self.max_splits)
        if share is not None and share.max_split_items >= max_items:
            self.max_split_items, self.o_part, self.ml_part = share.max_split_items, share.o_part, share.ml_part
        else:
            self.max_split_items = max_items
            self.o_part = torch.empty((max_items, ROWS, 512), dtype=torch.float32, device=device)
            self.ml_part = torch.empty((max_items, ROWS, 2), dtype=torch.float32, device=device)
        # Tables, per parity (module docstring): under the fixed split a prefill group may take up to MAX_SPLITS items
        max_groups = (tiles(max_tokens, self.group) + max_rows) * num_kv_heads
        self.rows = [torch.zeros((max_rows + 1, 4), dtype=torch.int32, device=device) for _ in range(2)]
        n_items = max_groups * MAX_SPLITS if split_keys else max_groups + max_items
        self.items = [torch.zeros((n_items, 4), dtype=torch.int32, device=device) for _ in range(2)]
        self.groups = [torch.zeros((max_groups, 2), dtype=torch.int32, device=device) for _ in range(2)]
        # decode buckets: static row, item and group tables, the launch's sizes (items, groups, split), the host
        # tables per real row count (the wave split) and what the bucket's last replay's tables were made for (its
        # real rows; the fixed split: every row's split count)
        self.g_rows: dict[int, torch.Tensor] = {}
        self.g_tabs: dict[int, tuple[torch.Tensor, torch.Tensor, int, int, int]] = {}
        self.g_host: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        self.g_last: dict[int, Any] = {}

    def _splits(self, n_groups: int, max_kv: int) -> int:
        s = choose_splits(n_groups, max_kv, self.sms, self.resident)
        if n_groups * s <= self.max_split_items:
            return s
        return max(1, self.max_split_items // n_groups)

    def plan(self, meta: Any, *, parity: int, rows: np.ndarray, stage: Any, fixed_rows: int = 0,
             lb: np.ndarray | None = None) -> None:
        """Eager steps: `rows` int32 [n, 4] (slot, first token, n_query, KV length after the step); `stage` puts an
        int32 array into the step's page-locked staging and returns its host view (the runner's `put32`). Under the
        fixed split, the first `fixed_rows` rows are decode rows, their queries anywhere from their lowest possible KV
        length `lb` (per row; None: the planned ones) to their planned one, and the rest prefill rows (module
        docstring)."""
        nq, kv_len = rows[:, 2].astype(np.int64), rows[:, 3].astype(np.int64)
        if self.split_keys:
            low = kv_len if lb is None else np.asarray(lb, np.int64)
            d = slice(0, fixed_rows)
            n_dec = range_splits(low[d] - nq[d], kv_len[d] - 1, self.split_keys)
            first, last = tile_positions(nq[fixed_rows:], kv_len[fixed_rows:], self.group)
            n_tile = np.concatenate([np.repeat(n_dec, -(-nq[d] * self.group // ROWS)),
                                     range_splits(first, last, self.prefill_keys, self.prefill_from)])
            items, grp = tile_items(nq, self.group, self.hkv, n_tile)
            parts = launches(grp, self.max_split_items)
        else:
            groups = int(np.sum(-(-nq * self.group // ROWS))) * self.hkv
            splits = self._splits(groups, int(kv_len.max()))
            items, grp = plan_items(nq, self.group, self.hkv, splits)
            parts = [(0, len(items), 0, len(grp))]
        dr, di, dg = self.rows[parity], self.items[parity], self.groups[parity]
        dr[:len(rows)].copy_(stage(rows.reshape(-1)).view(-1, 4), non_blocking=True)
        di[:len(items)].copy_(stage(items.reshape(-1)).view(-1, 4), non_blocking=True)
        runs = []
        for i0, ni, g0, ng in parts:  # each launch's group table counts its items from its own first one
            split = bool((grp[g0:g0 + ng, 1] > 1).any())
            if split:
                rel = grp[g0:g0 + ng].copy()
                rel[:, 0] -= i0
                dg[g0:g0 + ng].copy_(stage(rel.reshape(-1)).view(-1, 2), non_blocking=True)
            runs.append((i0, ni, g0, ng, int(split)))
        meta.full_plan = (dr, di, dg, fixed_rows, tuple(runs))

    def _graph_host(self, bucket: int, ns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """A bucket's item and group tables for split counts `ns` per row (padding rows 1), padded to the bucket's
        capacity with padding items (row −1)."""
        items, grp = plan_items(np.full(bucket, self.qo_per_row, np.int64), self.group, self.hkv, ns)
        _, _, cap_i, _, _ = self.g_tabs[bucket]
        if len(items) > cap_i:
            raise RuntimeError(f"attn512: {len(items)} items for bucket {bucket} (capacity {cap_i})")
        return np.concatenate([items, np.full((cap_i - len(items), 4), -1, np.int32)]), grp

    def _wave_host(self, bucket: int, real: int) -> tuple[np.ndarray, np.ndarray]:
        """The wave split's tables for `real` real rows (the rest padding), built once per bucket and real rows."""
        key = (bucket, real)
        got = self.g_host.get(key)
        if got is None:
            per_row = tiles(self.qo_per_row, self.group) * self.hkv
            ns = np.ones(bucket, np.int64)
            ns[:real] = bucket_splits(real, per_row, self.sms, self.resident, self.max_split_items)
            got = self.g_host[key] = self._graph_host(bucket, ns)
        return got

    def graph_tables(self, bucket: int) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor, int, int]:
        """A decode bucket's static row table and its item and group tables (`qo_per_row` queries per row), with the
        sizes its captured launch takes: items, groups, split."""
        if bucket not in self.g_tabs:
            per_row = tiles(self.qo_per_row, self.group) * self.hkv
            n_groups = bucket * per_row
            if self.split_keys:  # every row at the most splits a row takes
                cap, any_split = n_groups * self.max_splits, self.max_splits > 1
            else:
                cap, any_split = bucket_capacity(bucket, per_row, self.sms, self.resident, self.max_split_items)
            self.g_tabs[bucket] = (torch.full((cap, 4), -1, dtype=torch.int32, device=self.device),
                                   torch.full((n_groups, 2), -1, dtype=torch.int32, device=self.device), cap,
                                   n_groups, int(any_split))
            self.g_last[bucket] = None
            self.g_rows[bucket] = torch.zeros((bucket, 4), dtype=torch.int32, device=self.device)
        it, gt, n_items, n_groups, split = self.g_tabs[bucket]
        return self.g_rows[bucket], it, n_items, gt, n_groups, split

    def plan_graph(self, meta: Any, *, bucket: int, rows: np.ndarray, stage: Any, real: int | None = None,
                   lb: np.ndarray | None = None) -> None:
        """Before a decode graph's replay: its row table (int32 [bucket, 4], padding rows included) into the static
        buffer the captured launch reads; `real`, the real rows (the first ones; None: all of them), whose groups split
        to fill the wave, or by their own positions under the fixed split (module docstring; `lb`: the real rows'
        lowest possible KV lengths, None: the planned ones): the item and group tables are rewritten when they
        changed."""
        dr, it, n_items, gt, n_groups, split = self.graph_tables(bucket)
        dr.copy_(stage(rows.reshape(-1)).view(-1, 4), non_blocking=True)
        real = bucket if real is None else real
        if self.split_keys:
            nq, kv = rows[:real, 2].astype(np.int64), rows[:real, 3].astype(np.int64)
            low = kv if lb is None else np.asarray(lb, np.int64)[:real]
            ns = np.ones(bucket, np.int64)
            ns[:real] = range_splits(low - nq, kv - 1, self.split_keys)
            key: Any = ns.tobytes()
        else:
            key = real
        if self.g_last[bucket] != key:  # stream order: replays already queued read the old tables first
            items, grp = self._graph_host(bucket, ns) if self.split_keys else self._wave_host(bucket, real)
            it.copy_(stage(items.reshape(-1)).view(-1, 4), non_blocking=True)
            gt.copy_(stage(grp.reshape(-1)).view(-1, 2), non_blocking=True)
            self.g_last[bucket] = key
        meta.full_plan = (dr, it, gt, bucket, ((0, n_items, 0, n_groups, split),))

    def run(self, meta: Any, layer: Any, q: torch.Tensor, out: torch.Tensor) -> None:
        rows, items, groups, n_dec, runs = meta.full_plan
        if meta.fix_len:  # the true KV lengths, decided on the device (module docstring)
            rows[:meta.n_rows, 3].copy_(meta.kv_len_dev[:meta.n_rows])
        kv = self.pools.fkv[layer.index]  # [pages, page_size, Hkv, 512]
        pt = meta.page_table
        lib = build.load()
        for i0, n_items, g0, n_groups, split in runs:  # a step past the partials: its launches in stream order
            _check(lib.lmk_attn512(
                _p(q), self.h, self.hkv, _p(kv), kv.stride(0), kv.shape[1], _p(pt), pt.stride(0),
                _p(layer.k_norm.weight), _p(self.cos), _p(self.sin), self.rot, self.scale, _p(rows), _p(items[i0:]),
                n_items, _p(groups[g0:]), n_groups, split, self.split_keys, n_dec, self.prefill_from,
                self.prefill_keys, _p(out), _p(self.o_part) if split else _p(None),
                _p(self.ml_part) if split else _p(None), _stream(q)), "attn512")
