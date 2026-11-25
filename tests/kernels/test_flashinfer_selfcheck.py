"""The FlashInfer self-check's reading of the plan info (`kernels/flashinfer.py`), against stand-in wrappers on CPU:
FlashInfer's layout (0.6.18's and 0.7.2's) passes; a moved chunk-size offset, a missing split flag or a missing plan
argument are reported (the runner then plans uncertain lengths without a split). Then the rules a split plan for
uncertain lengths follows, at k = 3 and 5, and a sweep of the sliding layers' decode-graph plans under MTP (k up to 7,
verify and drafter rows, every decode bucket up to 64 rows) against a model of FlashInfer's CUDA-graph planner
(`graph_plan`). tests/kernels/test_flashinfer_lengths.py runs the real FlashInfer on a GPU, and checks the model against
it."""

from __future__ import annotations

import functools
import itertools
import types
from collections import Counter
from typing import Any

import numpy as np
import pytest
import torch

from llm_mini_infer.kernels import flashinfer as fl

from ..support.flashinfer_plan import BUCKETS, DEFAULT_BUCKETS, HKV, HQ, PS, GraphPlan, W, graph_plan


class Wrapper:
    """FlashInfer's plan() (0.6.18 to 0.7.2): a 15-entry plan info, the work items (query tiles times chunks) at 0, the
    query tile at 3, the split flag at 14, the chunk size (tokens) written into the page-locked buffer at the offset
    entry 9 names. `layout` shifts things to simulate another version."""

    def __init__(self, layout: str = "0.6.18"):
        self.layout = layout
        self._pin_memory_int_workspace_buffer = None
        self._plan_info = None

    def plan(self, qo_indptr, kv_indptr, kv_indices, last_page_len, hq, hkv, d, page_size, causal=False,
             q_data_type=None, kv_data_type=None, non_blocking=True, seq_lens=None, max_token_per_sequence=None,
             fixed_split_size=None, disable_split_kv=False):
        info = [0] * 15
        pages = int(kv_indptr[-1])
        chunk = pages if disable_split_kv or not fixed_split_size else fixed_split_size
        split = int(not disable_split_kv and chunk < pages)
        off = 48
        self._pin_memory_int_workspace_buffer[off:off + 4].view(torch.int32)[0] = chunk * page_size
        tile = 16 if int(qo_indptr[-1]) * hq // hkv <= 16 else 64
        info[0] = -(-int(qo_indptr[-1]) * hq // hkv // tile) * -(-pages // chunk)
        info[3], info[9], info[14] = tile, off, split
        if self.layout == "offset moved":
            info[9] = off + 4
        elif self.layout == "flag moved":
            info[14], info[13] = 0, split
        elif self.layout == "longer":
            info.append(0)
        elif self.layout == "tile moved":
            info[3], info[4] = 0, tile
        elif self.layout == "work moved":
            info[0], info[1] = 0, info[0]
        self._plan_info = info


class OldWrapper(Wrapper):
    def plan(self, qo_indptr, kv_indptr, kv_indices, last_page_len, hq, hkv, d, page_size, causal=False,
             q_data_type=None, kv_data_type=None, non_blocking=True, seq_lens=None, max_token_per_sequence=None,
             fixed_split_size=None):
        return super().plan(qo_indptr, kv_indptr, kv_indices, last_page_len, hq, hkv, d, page_size,
                            fixed_split_size=fixed_split_size)


def attn_with(wrapper: Wrapper):
    return types.SimpleNamespace(wrappers=[wrapper], pools=types.SimpleNamespace(page_size=16), qo_per_row=4,
                                 int_workspace=torch.zeros(8, dtype=torch.uint8),
                                 pinned=[torch.zeros(4096, dtype=torch.uint8)], hq=8, hkv=2, d=128,
                                 dtype=torch.bfloat16)


@pytest.fixture(autouse=True)
def unpinned(monkeypatch) -> None:
    """Pinning opens a CUDA context on a GPU host: the check's index tensors stay unpinned."""
    monkeypatch.setattr(fl, "_pinned32", lambda a: torch.as_tensor(np.asarray(a, np.int32)))


def test_the_0_6_18_layout_passes() -> None:
    assert fl.check_plan_info(attn_with(Wrapper())) is None


def test_other_layouts_are_reported() -> None:
    assert "chunk size at offset" in fl.check_plan_info(attn_with(Wrapper("offset moved")))
    assert "split_kv" in fl.check_plan_info(attn_with(Wrapper("flag moved")))
    assert "16 entries" in fl.check_plan_info(attn_with(Wrapper("longer")))
    assert "query tile" in fl.check_plan_info(attn_with(Wrapper("tile moved")))
    assert "work items" in fl.check_plan_info(attn_with(Wrapper("work moved")))
    assert "disable_split_kv" in fl.check_plan_info(attn_with(OldWrapper()))


def stub_checks(monkeypatch, info: list[str | None], lengths) -> list[str]:
    """`self_check` over stand-in checks: the plan info's findings in call order, the length check's per caller
    (`lengths(attn)`). Returns the checks' calls."""
    runs: list[str] = []
    found = iter(info)
    monkeypatch.setattr(fl, "_checked", {})
    monkeypatch.setattr(fl, "_plan_info_ok", True)
    monkeypatch.setattr(fl, "versions", lambda: {"flashinfer": "0.7.2"})
    monkeypatch.setattr(fl, "check_plan_info", lambda attn: runs.append("info") or next(found))
    monkeypatch.setattr(fl, "check_lengths", lambda attn, kv_dtype: runs.append("lengths") or lengths(attn))
    return runs


def test_a_failed_self_check_fails_every_call(monkeypatch) -> None:
    """The self-check runs once per process and kind of caller, and a later call fails as the first did: a length
    rewrite that does not take, and, for a caller with decode graphs, an unreadable plan info."""
    graphs = types.SimpleNamespace(max_rows=16)
    runs = stub_checks(monkeypatch, [None], lambda attn: "the rewrite did not take")
    for _ in range(2):
        with pytest.raises(RuntimeError, match="the rewrite did not take"):
            fl.self_check(graphs, torch.bfloat16)
    assert runs == ["info", "lengths"]
    stub_checks(monkeypatch, ["the chunk size moved"], lambda attn: None)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="decode graphs"):
            fl.self_check(graphs, torch.bfloat16)
    assert not fl._plan_info_ok


def test_a_graph_caller_is_checked_after_a_graphless_one(monkeypatch) -> None:
    """The length check of a caller with decode graphs also reads its graph wrappers' buffers: a graph-less caller's
    pass does not stand for it. The plan info is the first caller's: a later kind's finding leaves `_plan_info_ok`."""
    plain, graphs = types.SimpleNamespace(max_rows=0), types.SimpleNamespace(max_rows=16)
    runs = stub_checks(monkeypatch, [None, "the chunk size moved"],
                       lambda attn: "a wrapper's buffer is not its own" if attn.max_rows else None)
    first = fl.self_check(plain, torch.bfloat16)
    assert first == {"flashinfer": "0.7.2", "plan_info": "ok", "lengths": "ok"}
    first["lengths"] = "changed by its caller"
    assert fl.self_check(plain, torch.bfloat16)["lengths"] == "ok"  # a copy each time
    for _ in range(2):
        with pytest.raises(RuntimeError, match="a wrapper's buffer is not its own"):
            fl.self_check(graphs, torch.bfloat16)
    assert runs == ["info", "lengths"] * 2 and fl._plan_info_ok


def test_unsafe_chunks_and_safe_sizes() -> None:
    """The split rule the self-check protects: a chunk boundary inside some row's [lb, ub) is unsafe."""
    lb, ub = np.asarray([128]), np.asarray([131])
    assert fl.unsafe_chunks(lb, ub, 16, 8)
    c = fl.safe_chunk_pages(lb, ub, 16, 8)
    assert c > 8 and not fl.unsafe_chunks(lb, ub, 16, c)


def _direct_unsafe(lb: np.ndarray, ub: np.ndarray, ps: int, chunk: int) -> bool:
    """The rule without a window, stated directly: a chunk start c x chunk >= lb that the plan made, on uncertain
    rows."""
    u = lb < ub
    if not u.any():
        return False
    first = np.maximum(-(-lb[u] // (chunk * ps)), 1)
    return bool((first * chunk < -(-ub[u] // ps)).any())


def test_without_a_window_the_rule_is_the_direct_one() -> None:
    """Full attention (Qwen's layers): the kernel's chunks start at 0, so the rule is a planned chunk start between a
    row's bounds (`_direct_unsafe`), and the safe size an eager plan gets is the smallest from the plan's own up that
    has none."""
    rng = np.random.default_rng(0)
    for _ in range(3000):
        n = int(rng.integers(1, 9))
        ub = rng.integers(1, 3000, n)
        lb = np.maximum(ub - rng.integers(0, 8, n), 1)
        chunk = int(rng.integers(1, 40))
        assert fl.unsafe_chunks(lb, ub, 16, chunk) == _direct_unsafe(lb, ub, 16, chunk)
        if _direct_unsafe(lb, ub, 16, chunk):
            direct = chunk
            while direct < int((-(-ub // 16)).max()) and _direct_unsafe(lb, ub, 16, direct):
                direct += 1
            assert fl.safe_chunk_pages(lb, ub, 16, chunk) == direct


def test_a_window_caps_the_lengths_that_matter() -> None:
    """With a window, the kernel counts a row's chunks over at most window_left + query tile positions: a boundary
    between the lower and upper bound past that cap changes nothing (Gemma 4's sliding layers: 1,023 + 64)."""
    lb, ub = np.asarray([1102]), np.asarray([1105])
    assert fl.unsafe_chunks(lb, ub, 16, 23)  # without a window the boundary at 3 x 368 = 1,104 matters
    assert not fl.unsafe_chunks(lb, ub, 16, 23, window_left=1023, cta_tile_q=64)
    lb, ub = np.asarray([1038]), np.asarray([1041])  # the boundary at 1,040 lies below the cap: it matters
    assert fl.unsafe_chunks(lb, ub, 16, 65, window_left=1023, cta_tile_q=64)
    assert not fl.unsafe_chunks(lb, ub, 16, 65, window_left=1023, cta_tile_q=16)  # cap 1,039


def test_a_graph_keeps_its_split_within_its_grid() -> None:
    """16 decode rows of 65 and 66 window pages, each known within 3 positions: FlashInfer's 23 work items fit only
    chunks of 65 pages, whose boundary at 1,040 falls inside some rows' bounds, and no other chunking keeps a split
    there; the sliding layers' larger grid fits chunks of 64. An eager plan may drop the split (66: one chunk each)."""
    lay = dict(window_left=1023, cta_tile_q=64)
    ub = np.asarray([1042, 1030] * 8)
    lb = ub - 3
    assert fl.unsafe_chunks(lb, ub, 16, 65, **lay)
    assert fl.safe_chunk_pages(lb, ub, 16, 65, keep_split=True, max_work=23, tiles=1, **lay) is None
    assert fl.safe_chunk_pages(lb, ub, 16, 65, keep_split=True, max_work=50, tiles=1, **lay) == 64
    assert fl.safe_chunk_pages(lb, ub, 16, 65, **lay) == 66


def test_a_graph_keeps_its_split_within_its_grid_at_k_5() -> None:
    """k = 5: verify rows of 6 queries, each known within 5 positions, and the drafter's rows of one, within 10. 16
    verify rows of 65 and 66 window pages: FlashInfer's 23 work items fit only chunks of 65 pages (its own choice on
    188 SMs), whose boundary at 1,040 falls inside some rows' bounds, and no other chunking keeps a split there; the
    16-row graphs' grid (49 work items, `graph_wrapper`) fits chunks of 64. The drafter's rows whose lower bound is the
    window's 1,024 positions make 64 unsafe too: 63."""
    lay = dict(window_left=1023, cta_tile_q=64)
    ub = np.asarray([1042] * 4 + [1035] * 12)
    lb = ub - 5
    assert graph_plan(np.full(16, 6), -(-ub // 16), 1088, 188).chunk == 65  # 1,088: the 16-row graphs' row bound
    assert fl.unsafe_chunks(lb, ub, 16, 65, **lay)
    assert fl.safe_chunk_pages(lb, ub, 16, 65, keep_split=True, max_work=23, tiles=1, **lay) is None
    assert fl.safe_chunk_pages(lb, ub, 16, 65, keep_split=True, max_work=49, tiles=1, **lay) == 64
    ub = np.asarray([1042] * 4 + [1034] * 12)
    lb = ub - 10
    assert graph_plan(np.ones(16), -(-ub // 16), 1088, 188).chunk == 65
    assert fl.unsafe_chunks(lb, ub, 16, 65, **lay) and fl.unsafe_chunks(lb, ub, 16, 64, **lay)
    assert fl.safe_chunk_pages(lb, ub, 16, 65, keep_split=True, max_work=23, tiles=1, **lay) is None
    assert fl.safe_chunk_pages(lb, ub, 16, 65, keep_split=True, max_work=49, tiles=1, **lay) == 63


# -- FlashInfer's CUDA-graph plan on CPU (0.6.18 to 0.7.2) -------------------------------------------------------------


class GraphWrapper:
    """A CUDA-graph wrapper whose plan() is `graph_plan` on a GPU of `sms` SMs: the plan info, the chunk size in the
    page-locked buffer (as `Wrapper`), the row bound and the refusals as FlashInfer's; `plans` records every
    plan."""

    def __init__(self, sms: int, *args: Any, **bufs: Any):
        self.sms = sms
        self._max_total_num_rows: int | None = None
        self._pin_memory_int_workspace_buffer = None
        self._plan_info: list[int] | None = None
        self.plans: list[GraphPlan] = []

    def plan(self, qo_indptr, kv_indptr, kv_indices, last_page_len, hq, hkv, d, page_size, *, window_left=-1,
             fixed_split_size=None, disable_split_kv=False, **kw):
        assert d == 256 and not disable_split_kv  # what `graph_plan` models
        qo, pages = np.diff(qo_indptr.numpy()), np.diff(kv_indptr.numpy())
        if self._max_total_num_rows is None:
            self._max_total_num_rows = int(qo.sum())
        assert qo.sum() <= self._max_total_num_rows  # FlashInfer raises
        p = graph_plan(qo, pages, self._max_total_num_rows, self.sms, fixed_split_size or 0, window_left=window_left,
                       hq=hq, hkv=hkv, ps=page_size)
        assert p.work <= p.padded, f"FlashInfer refuses {p.work} work items in a grid of {p.padded}: {p}"
        off = 48
        self._pin_memory_int_workspace_buffer[off:off + 4].view(torch.int32)[0] = p.chunk * page_size
        info = [0] * fl._PLAN_INFO_LEN
        info[fl._PLAN_PADDED], info[fl._PLAN_CTA_TILE_Q] = p.padded, p.tile
        info[fl._PLAN_CHUNK_SIZE_OFFSET], info[fl._PLAN_SPLIT_KV] = off, int(p.split)
        self._plan_info = info
        self.plans.append(p)


def graph_attn(qo: int, sms: int) -> fl.FlashInferWindowAttention:
    """The sliding layers' attention of decode rows of `qo` queries under MTP (`upper_bounds`) without CUDA or
    FlashInfer: what `graph_wrapper`, `plan_graph` and `_plan_safely` read, set as `__init__` sets it on a GPU of `sms`
    SMs, with decode-graph wrappers that plan as FlashInfer does (`GraphWrapper`)."""
    a = object.__new__(fl.FlashInferWindowAttention)
    a.pools, a.hq, a.hkv, a.d, a.dtype = types.SimpleNamespace(page_size=PS), HQ, HKV, 256, torch.bfloat16
    a.window, a.scale, a.qo_per_row = W, 1.0, qo
    a.workspace, a.int_workspace = None, torch.zeros(8, dtype=torch.uint8)
    a.pinned = [torch.zeros(64, dtype=torch.uint8) for _ in range(2)]
    a.replans = a.kept = a.unsplit = 0
    a.g_qo = a.g_kv_indptr = a.g_last = a.g_kv_indices = torch.zeros(BUCKETS[-1] + 1, dtype=torch.int32)
    a.g_wrappers, a.split_items, a._cls = {}, 2 * sms // HKV, functools.partial(GraphWrapper, sms)
    return a


def mtp_bounds(seq: np.ndarray, slack: np.ndarray, k: int, drafter: bool) -> tuple[np.ndarray, np.ndarray]:
    """Decode rows' lowest and highest KV lengths under MTP, from the window page of each row's lowest first query, as
    `GemmaRunner._layout` and `_stage_mtp` stage them: `seq` committed positions on the host (the device is up to
    `slack` ahead: k while the previous step is in flight, else 0). A verify row of 1 + k queries holds KV
    [seq + 1 + k, seq + slack + 1 + k]; the drafter's row, one query at the last committed position, [seq + 1,
    seq + slack + k + 1]."""
    base = np.maximum(seq - W + 1, 0) // PS * PS
    lb = seq + (1 if drafter else 1 + k) - base
    return lb, lb + slack + (k if drafter else 0)


def plan_bucket(attn: fl.FlashInferWindowAttention, rows: int, qo: int, lb: np.ndarray, ub: np.ndarray) -> Any:
    """`attn.plan_graph` of a decode bucket's rows as the runner makes it (`GemmaRunner._plan_decode_graph`,
    `_drafter_meta`): lb only when some row is uncertain. Returns the bucket's wrapper."""
    npg = -(-ub // PS)
    i32 = lambda a: torch.as_tensor(np.asarray(a, np.int32))  # noqa: E731 (host tensors; nothing is copied)
    attn.plan_graph(None, rows=rows, parity=0, qo_indptr=i32(np.arange(rows + 1) * qo),
                    kv_indptr=i32(np.concatenate([[0], np.cumsum(npg)])), last_page_len=i32(ub - (npg - 1) * PS),
                    kv_lens=i32(ub), max_q=qo, lb=lb if (lb < ub).any() else None)
    return attn.g_wrappers[rows]


def _draws(rng: np.random.Generator, rows: int, prev: int, k: int):
    """(the lengths' kind, real rows, committed positions, slack) of the decode steps of a bucket above `prev` rows (the
    rest are padding rows): long ones, past the window (every row at each offset in its first window page, then
    offsets and slack drawn per row), lengths across [0, 2 × window), and every row at one drawn length."""
    for r in range(PS):
        yield "long", rows, np.full(rows, W - 1 + r + 7 * PS), np.full(rows, k)
    for _ in range(12):
        n = int(rng.integers(prev + 1, rows + 1))
        slack = np.where(rng.random(n) < 0.75, k, 0)
        yield "long", n, rng.integers(W, 8 * W, n), slack
        yield "uniform", n, rng.integers(0, 2 * W, n), slack
        yield "equal", n, np.full(n, rng.integers(0, 2 * W)), np.full(n, k)


def test_decode_graph_replans_always_find_a_split_that_fits() -> None:
    """The sliding layers' decode graphs under MTP (k = 3, 5 and 7) on 188 SMs (an RTX PRO 6000) and 170 (an RTX 5090),
    every bucket with its padding rows, verify rows (lengths known within k) and the drafter's (within 2k), at lengths
    past the window and others: wherever FlashInfer's own plan (`graph_plan`) could lose a chunk to the true lengths,
    `plan_graph` plans again with a chunk count none changes, keeping the captured split and fitting the captured grid;
    no plan is `kept`. The 12- and 16-row graphs' grids hold two chunks per row (49 work items at 16 rows on 188 SMs,
    `graph_wrapper`); buckets of 24 rows or more exceed FlashInfer's split work items, so its own plans give one chunk
    per row, which no true length changes."""
    rng = np.random.default_rng(8)
    replans: Counter[tuple[int, str]] = Counter()
    for sms, k, drafter in itertools.product((188, 170), (3, 5, 7), (False, True)):
        qo = 1 if drafter else 1 + k
        for rows in BUCKETS:
            prev = max((b for b in DEFAULT_BUCKETS if b < rows), default=0)  # 16 rows: 9-16 real, as served
            attn = graph_attn(qo, sms)
            seq = np.full(rows, 3)  # the capture's plan (`_decode_capture_plan`): the row bound, unless preset
            w = plan_bucket(attn, rows, qo, *mtp_bounds(seq, np.zeros(rows, np.int64), k, drafter))
            if rows < attn.split_items < 2 * rows:  # the 12- and 16-row buckets
                assert w.plans[0].padded >= 2 * rows, (sms, k, drafter, rows, w.plans[0])
            assert (rows, sms) != (16, 188) or w.plans[0].padded == 49, (k, drafter, w.plans[0])
            for mode, n, seq, slack in _draws(rng, rows, prev, k):
                lb, ub = (np.concatenate([b, np.full(rows - n, qo)]) for b in mtp_bounds(seq, slack, k, drafter))
                before = len(w.plans)
                plan_bucket(attn, rows, qo, lb, ub)
                own, last = w.plans[before], w.plans[-1]
                ctx = (sms, k, "drafter" if drafter else "target", rows, mode, lb[:n].tolist(), ub[:n].tolist(), own)
                unsafe = fl.unsafe_chunks(lb, ub, PS, own.chunk, window_left=W - 1, cta_tile_q=own.tile)
                assert (len(w.plans) > before + 1) == unsafe, ctx  # planned again exactly where FlashInfer's was unsafe
                assert last.split and (last.tile, last.padded) == (own.tile, own.padded), (ctx, last)
                assert not fl.unsafe_chunks(lb, ub, PS, last.chunk, window_left=W - 1, cta_tile_q=last.tile), \
                    (ctx, last)
                assert np.array_equal(last.counts, fl.chunk_counts(ub, PS, last.chunk, W - 1, last.tile)), (ctx, last)
                assert rows < 24 or own.counts.max() == 1, ctx
                replans[rows, mode] += unsafe
            assert attn.kept == 0, (sms, k, drafter, rows)
    assert replans[16, "long"] > 0 and replans[12, "long"] > 0, replans
