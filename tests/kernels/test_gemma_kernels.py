"""Gemma 4's CUDA kernels against their torch references, on random inputs. CUDA only (skipped on a Mac).

The quantizing producers (the RMS norm, the sandwich, GELU-tanh x up) equal their BF16 kernel followed by the
standalone quantization (`fp8.quant_ref`) bit for bit, the kernel path enforced (a fallback would call the reference,
which the tests forbid while the producer runs).

Norms, the sandwich, GELU and the q/k/v store round at the references' points, so they agree to one BF16 ulp (reduction
order). The head-512 attention (`csrc/attn512.cu`, K derived from the stored V plane) and FlashInfer's sliding window
(from each row's first window page) are compared with fp32 attention over the same BF16 inputs, within BF16's rounding
of P and of the output.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from llm_mini_infer.kernels import build, fp8, ops
from llm_mini_infer.kernels import gemma_ops as gops
from llm_mini_infer.runner.cuda.graphs import no_gc

from ..support.kernel_cases import (
    BF,
    DEV,
    GuardedPool,
    _full_case,
    _Layer,
    _Pools,
    _rope_tables,
    _Rot,
    _window_ref,
    _window_run,
    _window_setup,
    kv_slots,
    ref,
)

pytestmark = pytest.mark.gpu(reason="needs a CUDA GPU and the kernel library")


def ulp_close(a: torch.Tensor, b: torch.Tensor, max_ulps: int = 1, min_equal: float = 0.98) -> None:
    ia = a.contiguous().view(torch.int16).to(torch.int32)
    ib = b.contiguous().view(torch.int16).to(torch.int32)
    d = (ia - ib).abs()
    assert int(d.max()) <= max_ulps, f"max {int(d.max())} ulps"
    assert float((d == 0).float().mean()) >= min_equal, float((d == 0).float().mean())


def rope_close(a: torch.Tensor, b: torch.Tensor) -> None:
    """RoPE outputs [T, H, D]: `ulp_close`, but a value that cancels to near zero (under 2^-16 of its head's largest)
    by its absolute error instead, within 2^-20 of that largest: its ulps count the rounding of two fp32 products of the
    head's scale, which the kernel and the reference order differently."""
    fa, fb = a.float(), b.float()
    scale = fa.abs().amax(-1, keepdim=True).expand_as(fa)
    tiny = (fa.abs() < scale * 2 ** -16) & (fb.abs() < scale * 2 ** -16)
    ulp_close(torch.where(tiny, 0, a), torch.where(tiny, 0, b))
    assert bool(((fa - fb).abs() <= scale * 2 ** -20)[tiny].all())


@pytest.mark.parametrize("d", [256, 5376])
def test_rmsnorm_and_sandwich(d) -> None:
    torch.manual_seed(0)
    x = torch.randn(37, d, device=DEV, dtype=BF) * 3
    res = torch.randn(37, d, device=DEV, dtype=BF) * 10
    w1, w2 = torch.randn(d, device=DEV, dtype=BF) * 0.5, torch.randn(d, device=DEV, dtype=BF) * 0.5
    ulp_close(gops.rmsnorm(x, w1, 1e-6), ref(gops.rmsnorm, x, w1, 1e-6))
    for scalar in (None, torch.tensor([0.7], device=DEV, dtype=BF)):
        r, h = gops.sandwich(x, res, w1, w2, 1e-6, scalar=scalar)
        r2, h2 = ref(gops.sandwich, x, res, w1, w2, 1e-6, scalar=scalar)
        ulp_close(r, r2)
        ulp_close(h, h2)


def test_gelu_tanh_mul() -> None:
    torch.manual_seed(1)
    gu = torch.randn(29, 2 * 2688, device=DEV, dtype=BF) * 2
    ulp_close(gops.gelu_tanh_mul(gu), ref(gops.gelu_tanh_mul, gu))


def q_equal(a: fp8.QuantAct, b: fp8.QuantAct) -> None:
    m = a.q.shape[0]
    assert torch.equal(a.q.view(torch.uint8), b.q.view(torch.uint8)), "quantized bytes differ"
    assert torch.equal(a.s[:, :m], b.s[:, :m]), "scales differ"


def _kernel_only(monkeypatch) -> None:
    """While a producer runs, its reference path (BF16 op, then quant_ref) must not: the fused kernel ran."""
    def boom(*a, **kw):
        raise AssertionError("the quantizing producer fell back to the reference")
    monkeypatch.setattr(fp8, "quant_ref", boom)


def _heavy(t: int, d: int, seed: int) -> torch.Tensor:
    """Rows of every magnitude, a few large channels, and an all-zero group: the quantization's edge cases."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn(t, d, device=DEV, generator=g) * torch.logspace(-2, 1.5, t, device=DEV)[:, None]
    x[:, 7::331] *= 40
    if t > 2:
        x[2, :128] = 0
    return x.to(BF)


@pytest.mark.parametrize("d", [256, 1024, 5376])
@pytest.mark.parametrize("t", [1, 2, 3, 33, 517])
def test_rmsnorm_quant_exact(d, t, monkeypatch) -> None:
    x = _heavy(t, d, 1)
    w = torch.randn(d, device=DEV, dtype=BF) * 0.5
    want = fp8.quant_ref(gops.rmsnorm(x, w, 1e-6))  # the BF16 kernel, then the reference quantization
    _kernel_only(monkeypatch)
    q_equal(gops.rmsnorm_quant(x, w, 1e-6), want)


@pytest.mark.parametrize("d", [256, 5376, 8192])
@pytest.mark.parametrize("t", [1, 2, 7, 64, 300])
def test_sandwich_quant_exact(d, t, monkeypatch) -> None:
    a, res = _heavy(t, d, 2), _heavy(t, d, 3) * 4
    w1, w2 = torch.randn(d, device=DEV, dtype=BF) * 0.5, torch.randn(d, device=DEV, dtype=BF) * 0.5
    for scalar in (None, torch.tensor([0.7], device=DEV, dtype=BF)):
        r_want, x = gops.sandwich(a, res, w1, w2, 1e-6, scalar=scalar)
        want = fp8.quant_ref(x)
        with monkeypatch.context() as mp:
            _kernel_only(mp)
            r, q = gops.sandwich_quant(a, res, w1, w2, 1e-6, scalar=scalar)
        assert torch.equal(r, r_want)
        q_equal(q, want)


@pytest.mark.parametrize("i", [256, 21504])
@pytest.mark.parametrize("t", [1, 3, 64, 260])
def test_gelu_tanh_mul_quant_exact(i, t, monkeypatch) -> None:
    gu = _heavy(t, 2 * i, 4)
    big = torch.empty((t, 2 * i + 64), device=DEV, dtype=BF)
    big[:, 64:] = gu
    for x in (gu, big[:, 64:]):  # contiguous, and a strided view
        want = fp8.quant_ref(gops.gelu_tanh_mul(x))
        with monkeypatch.context() as mp:
            _kernel_only(mp)
            q_equal(gops.gelu_tanh_mul_quant(x), want)


def test_attention_output_quantization_exact() -> None:
    """The attention output (the sliding layers' after both passes) goes through the standalone quantization."""
    o = _heavy(19, 32 * 256, 5)
    q_equal(ops.quant_fp8(o), fp8.quant_ref(o))


def test_producers_non_finite_exact(monkeypatch) -> None:
    """NaN and ±Inf through the quantizing producers, bit for bit with the BF16 kernel and the reference quantization
    (tests/runner/test_fp8.py pins it: a NaN leaves the absmax and quantizes to −448, an infinite value makes the scale
    Inf). A norm makes a non-finite input's row NaN (every group all-NaN), a NaN norm weight a column (a NaN among
    finite values in every other row), an infinite one an infinite column; the sandwich reduces its absmax over
    half-warps."""
    nan, inf = float("nan"), float("inf")
    t, d = 5, 1024
    x, res = _heavy(t, d, 6), _heavy(t, d, 7)
    x[0, 9], x[1, 400] = nan, inf
    w, w_post = torch.randn(d, device=DEV, dtype=BF) * 0.5, torch.randn(d, device=DEV, dtype=BF) * 0.5
    w[5], w[300] = nan, inf
    want = fp8.quant_ref(gops.rmsnorm(x, w, 1e-6))
    with monkeypatch.context() as mp:
        _kernel_only(mp)
        q_equal(gops.rmsnorm_quant(x, w, 1e-6), want)
    want = fp8.quant_ref(gops.sandwich(x, res, w_post, w, 1e-6)[1])
    with monkeypatch.context() as mp:
        _kernel_only(mp)
        q_equal(gops.sandwich_quant(x, res, w_post, w, 1e-6)[1], want)
    gu = _heavy(t, 2 * d, 8)
    gu[0, 3] = nan
    gu[1, d + 128:d + 256] = nan
    gu[3, d + 300], gu[4, 10] = inf, -inf
    want = fp8.quant_ref(gops.gelu_tanh_mul(gu))
    with monkeypatch.context() as mp:
        _kernel_only(mp)
        q_equal(gops.gelu_tanh_mul_quant(gu), want)


@pytest.mark.parametrize(("d", "k_eq_v"), [(256, False), (512, True), (512, False)])
def test_qkv_store(d, k_eq_v) -> None:
    """q, and K and V (or with `k_eq_v` V alone) into exactly their slots of a guarded pool
    (tests/support/kernel_cases.py: `GuardedPool`, `kv_slots`)."""
    torch.manual_seed(2)
    h, hkv, ps = 8, 2, 16
    page, off = kv_slots(ps)
    t = len(page)
    qkv = torch.randn(t, (h + hkv * (1 if k_eq_v else 2)) * d, device=DEV, dtype=BF)
    qkv[-1] = qkv[-2]  # the padding tokens
    qw, kw = torch.randn(d, device=DEV, dtype=BF) * 0.3 + 1, torch.randn(d, device=DEV, dtype=BF) * 0.3 + 1
    pos = torch.randint(0, 4000, (t,), device=DEV)
    pos[-2:] = 0
    cos, sin = _rope_tables(d, d // 2 if d == 256 else 64)
    pool = GuardedPool((10, ps, hkv, d) if k_eq_v else (10, 2, ps, hkv, d), seed=3)
    b1, b2 = pool.backing.clone(), pool.backing.clone()
    kw_args = dict(heads=h, kv_heads=hkv, head_dim=d, k_eq_v=k_eq_v, q_w=qw, k_w=kw, eps=1e-6, positions=pos,
                   cos_cache=cos, sin_cache=sin, kv_page=page, kv_off=off)
    q1 = gops.qkv_store(qkv, kv_layer=pool.view(b1), **kw_args)
    q2 = ref(gops.qkv_store, qkv, kv_layer=pool.view(b2), **kw_args)
    rope_close(q1, q2)
    pool.check(b1, b2, page, off)


# -- the head-512 attention ------------------------------------------------------------------------------------------


def _full_ref(rows, fkv, table, q, kw, cos, sin, ps):
    out = torch.empty_like(q)
    s = 0
    for r, (nq, kv) in enumerate(rows):
        pg = table[r + 1, :-(-kv // ps)].long()
        v = fkv[pg].reshape(-1, fkv.shape[2], fkv.shape[3])[:kv]
        kpos = torch.arange(kv, device=DEV)
        k = gops.k_from_v(v, kw, cos[kpos], sin[kpos])
        qpos = torch.arange(kv - nq, kv, device=DEV)
        out[s:s + nq] = gops.attention_ref(q[s:s + nq].float(), k.float(), v.float(), qpos, kpos).to(BF)
        s += nq
    return out


@pytest.mark.parametrize(("rows", "h", "hkv", "ps"), [
    ([(1, 1), (1, 17), (1, 300), (1, 5000)], 32, 4, 16),  # decode rows, short to long
    ([(64, 64), (37, 200), (1, 33), (130, 900)], 32, 4, 16),  # prefill chunks, from 0 and resumed
    ([(5, 5), (8, 1100)], 4, 1, 32),  # GQA 4 (the tiny GPU model), pages of 32
    ([(6, 6), (6, 17), (6, 300), (6, 5000)], 32, 4, 16),  # verify rows at k = 5: 3 tiles of 16 rows per KV head
    ([(8, 8), (8, 33), (8, 257), (8, 5000)], 32, 4, 16),  # at k = 7: 4 tiles
    ([(6, 6), (6, 300), (1, 1000), (37, 200)], 32, 4, 24),  # pages of 24: a tile may cross one (its keys' own pages)
])
def test_attn512_matches_reference(rows, h, hkv, ps) -> None:
    from llm_mini_infer.kernels.attn512 import Attn512

    fkv, table, q, kw, cos, sin = _full_case(rows, h, hkv, ps, seed=len(rows) + h)
    want = _full_ref(rows, fkv, table, q, kw, cos, sin, ps)
    a = Attn512(_Pools(fkv[None], ps), _Rot(cos, sin), h, hkv, 64, torch.device(DEV), max_tokens=4096,
                max_rows=8)
    nq = np.asarray([r[0] for r in rows])
    kv = np.asarray([r[1] for r in rows])
    starts = np.concatenate([[0], np.cumsum(nq)[:-1]])
    rr = np.stack([np.arange(1, len(rows) + 1), starts, nq, kv], axis=1).astype(np.int32)
    meta = type("M", (), {"page_table": table, "fix_len": False})()
    stage = lambda a_: torch.from_numpy(np.ascontiguousarray(a_, np.int32)).pin_memory()  # noqa: E731
    a.plan(meta, parity=0, rows=rr, stage=stage)
    got = torch.full_like(q, float("nan"))
    a.run(meta, _Layer(kw), q, got)
    torch.cuda.synchronize()
    err = (got.float() - want.float()).abs()
    assert torch.isfinite(got).all()
    assert err.max().item() < 2e-2 and err.mean().item() < 2e-3, (err.max().item(), err.mean().item())


@pytest.mark.parametrize("qo", [1, 6, 8])
def test_attn512_split_counts_agree(qo) -> None:
    """The same rows of `qo` queries (decode rows; verify rows at k = 5 and 7: 3 and 4 tiles of 16 rows per KV head)
    with 1 to 31 splits per group, the wave split's odd counts among them, and with a split count of each row's own
    (one row's groups whole, the others split; padding items and groups that exit): the reference within BF16
    rounding, and each split count within BF16 rounding of the others."""
    from llm_mini_infer.kernels import attn512 as am

    rows = [(qo, 3000), (qo, 50), (qo, qo)]
    fkv, table, q, kw, cos, sin = _full_case(rows, 32, 4, 16, seed=9)
    want = _full_ref(rows, fkv, table, q, kw, cos, sin, 16)
    nq, kv = np.full(3, qo, np.int64), np.asarray([3000, 50, qo])
    rr = torch.as_tensor(np.stack([np.arange(1, 4), np.arange(3) * qo, nq, kv], axis=1).astype(np.int32), device=DEV)
    tiles = {1: 1, 6: 3, 8: 4}[qo]
    first = None
    for splits in (1, 3, 4, 5, 7, 11, 16, 23, 31, [7, 1, 3]):
        items, groups = am.plan_items(nq, 8, 4, splits)
        per_row = np.broadcast_to(np.asarray(splits), (3,))
        assert len(items) == tiles * 4 * int(per_row.sum()) and len(groups) == 3 * tiles * 4
        items = np.concatenate([items, np.full((5, 4), -1, np.int32)])  # padding items: their CTAs exit
        groups = np.concatenate([groups, np.full((2, 2), -1, np.int32)])
        it, gt = torch.as_tensor(items, device=DEV), torch.as_tensor(groups, device=DEV)
        split = int(per_row.max() > 1)
        o_part = torch.empty(len(items), 16, 512, device=DEV)
        ml = torch.empty(len(items), 16, 2, device=DEV)
        got = torch.full_like(q, float("nan"))
        ops._check(build.load().lmk_attn512(
            ops._p(q), 32, 4, ops._p(fkv), fkv.stride(0), 16, ops._p(table), table.stride(0), ops._p(kw), ops._p(cos),
            ops._p(sin), 64, 1.0, ops._p(rr), ops._p(it), len(items), ops._p(gt), len(groups), split, 0, 0, 0, 0,
            ops._p(None), ops._p(got), ops._p(o_part), ops._p(ml), ops._stream(q)), "attn512")
        torch.cuda.synchronize()
        assert torch.isfinite(got).all(), splits
        assert (got.float() - want.float()).abs().max().item() < 2e-2, splits
        if first is None:
            first = got.clone()
        # only the merge order differs: within BF16 rounding (an absolute bound: values near 0 may change sign, which
        # an ulp count over the bits would read as thousands)
        assert (got.float() - first.float()).abs().max().item() < 2e-2, splits


@pytest.mark.parametrize("ps", [16, 64, 24])
@pytest.mark.parametrize("splits", [1, 3])
def test_attn512_staged_tables_change_no_bit(ps, splits) -> None:
    """Each tile's cos and sin rows staged in shared memory (rotations of up to 64 dims, Gemma 4's) and its page read
    once (pages of a multiple of 16 positions) give the bits of the tables and pages read per key, the same items each
    time: the same KV in pages of 8 positions (a tile's keys on two pages, read per key), a rotation of 66 dims (the
    tables read directly: its two extra dims turn by angle 0, as Gemma 4's proportional RoPE's do), and both (tables
    and pages read per key). Pages of 24 positions: a tile can cross one, and its page is read per key."""
    from llm_mini_infer.kernels import attn512 as am

    rows = [(6, 2700), (6, 33), (1, 1300), (40, 400)]
    fkv, table, q, kw, cos, sin = _full_case(rows, 32, 4, ps, seed=5)
    m = ps // 8  # the same positions in pages of 8: page p's positions [8 j, 8 j + 8) are page m p + j
    fkv8 = fkv.view(-1, 8, *fkv.shape[2:])
    table8 = table.repeat_interleave(m, dim=1) * m + torch.arange(table.shape[1] * m, dtype=torch.int32,
                                                                  device=DEV) % m
    nq = np.asarray([r[0] for r in rows])
    kv = np.asarray([r[1] for r in rows])
    starts = np.concatenate([[0], np.cumsum(nq)[:-1]])
    rr = torch.as_tensor(np.stack([np.arange(1, len(rows) + 1), starts, nq, kv], axis=1).astype(np.int32), device=DEV)
    items, groups = am.plan_items(nq, 8, 4, splits)
    it, gt = torch.as_tensor(items, device=DEV), torch.as_tensor(groups, device=DEV)
    o_part = torch.empty(len(items), 16, 512, device=DEV)
    ml = torch.empty(len(items), 16, 2, device=DEV)
    outs = []
    for pool, size, pt, rot in ((fkv, ps, table, 64), (fkv8, 8, table8, 64), (fkv, ps, table, 66),
                                (fkv8, 8, table8, 66)):
        got = torch.full_like(q, float("nan"))
        ops._check(build.load().lmk_attn512(
            ops._p(q), 32, 4, ops._p(pool), pool.stride(0), size, ops._p(pt), pt.stride(0), ops._p(kw), ops._p(cos),
            ops._p(sin), rot, 1.0, ops._p(rr), ops._p(it), len(items), ops._p(gt), len(groups), int(splits > 1), 0, 0,
            0, 0, ops._p(None), ops._p(got), ops._p(o_part), ops._p(ml), ops._stream(q)), "attn512")
        torch.cuda.synchronize()
        outs.append(got)
    assert torch.isfinite(outs[0]).all()
    assert all(torch.equal(outs[0], o) for o in outs[1:])


@pytest.mark.parametrize(("bucket", "reals"), [(8, (1, 3, 8, 5, 5, 2, 8)), (24, (16, 18, 17, 24, 18))])
def test_attn512_waves_replay_their_real_rows(bucket, reals) -> None:
    """A decode bucket's graph, captured once, replayed with 1 to `bucket` real rows (the rest padding rows on the null
    page): each replay splits the real rows' groups to fill the wave (the item and group tables rewritten when the real
    rows change) and gives the reference on every real row. Bucket 24 at 16-18 real rows takes the under-filled wave's
    3 splits (576-648 items: 1.5-1.7 waves of the RTX PRO 6000's), at 24 one whole-KV item a group."""
    from types import SimpleNamespace

    from llm_mini_infer.kernels.attn512 import Attn512, bucket_splits, tiles

    h, hkv, ps, qo = 32, 4, 16, 6
    kvs = ([2600, 2650, 2701, 17, 2999, 300, 2560, 2577] + [1500 + 97 * i for i in range(16)])[:bucket]
    fkv, table, q, kw, cos, sin = _full_case([(qo, kv) for kv in kvs], h, hkv, ps, seed=33)
    stage = lambda a_: torch.from_numpy(np.ascontiguousarray(a_, np.int32)).pin_memory()  # noqa: E731
    a = Attn512(_Pools(fkv[None], ps), _Rot(cos, sin), h, hkv, 64, torch.device(DEV), max_tokens=4096, max_rows=bucket,
                qo_per_row=qo)
    meta = SimpleNamespace(page_table=table, fix_len=False, n_rows=bucket)
    got = torch.full_like(q, float("nan"))

    def rows_for(real: int) -> np.ndarray:
        rr = np.zeros((bucket, 4), np.int32)
        rr[:, 1] = np.arange(bucket) * qo
        rr[:, 2] = qo
        rr[:, 3] = qo  # padding rows: slot 0 (the null page), KV 1 + k
        rr[:real, 0] = np.arange(1, real + 1)
        rr[:real, 3] = kvs[:real]
        return rr

    a.plan_graph(meta, bucket=bucket, rows=rows_for(bucket), stage=stage, real=bucket)
    g = _replay_on_a_side_stream(lambda: a.run(meta, _Layer(kw), q, got))
    per_row = tiles(qo, h // hkv) * hkv
    for real in reals:
        a.plan_graph(meta, bucket=bucket, rows=rows_for(real), stage=stage, real=real)
        got.fill_(float("nan"))
        g.replay()
        torch.cuda.synchronize()
        n = real * qo
        want = _full_ref([(qo, kv) for kv in kvs[:real]], fkv, table, q[:n].contiguous(), kw, cos, sin, ps)
        err = (got[:n].float() - want.float()).abs()
        assert torch.isfinite(got).all(), real  # padding rows too: written, finite
        assert err.max().item() < 2e-2 and err.mean().item() < 2e-3, (real, err.max().item())
        # the real rows' own split count (`bucket_splits`), the padding's one item a group
        s = int(a.g_host[(bucket, real)][1][0, 1])
        assert s == bucket_splits(real, per_row, a.sms, a.resident, a.max_split_items), (real, s)
        assert (a.g_host[(bucket, real)][1][real * per_row:, 1] == 1).all()


# -- the sliding window on FlashInfer --------------------------------------------------------------------------------


def test_flashinfer_window_from_the_first_window_page() -> None:
    """Rows whose first query is past the window read their pages from the first window page on (the earlier ones
    hold garbage), with window_left = W − 1 and scale 1, against the reference sliding attention."""
    from llm_mini_infer.kernels.flashinfer import FlashInferWindowAttention, window_page_indices
    from llm_mini_infer.models.gemma4.backend import Gemma4Meta, Gemma4Pools, TorchSlidingAttention

    torch.manual_seed(5)
    w, ps, h, hkv, d = 64, 16, 8, 4, 256
    rows = [(1, 1), (1, 300), (40, 40), (24, 500), (100, 150)]  # (n_query, kv_len)
    npg = [-(-kv // ps) for _, kv in rows]
    n_pages = sum(npg) + 1
    wkv = (torch.randn(1, n_pages, 2, ps, hkv, d, device=DEV) * 0.5).to(BF)
    pools = Gemma4Pools(wkv=wkv, fkv=torch.zeros(1), page_size=ps, window=w)
    table = torch.zeros(len(rows) + 1, max(npg) + 2, dtype=torch.int32, device=DEV)
    o = 1
    for r, n in enumerate(npg):
        table[r + 1, :n] = torch.arange(o, o + n, dtype=torch.int32)
        o += n
    nq = np.asarray([r[0] for r in rows])
    kv = np.asarray([r[1] for r in rows])
    q0 = kv - nq
    wlo = np.maximum(q0 - w + 1, 0) // ps
    for r in range(len(rows)):  # pages before the window: garbage that must not be read
        wkv[0, table[r + 1, :wlo[r]].long()] = float("nan")
    t = int(nq.sum())
    q = (torch.randn(t, h, d, device=DEV) * 0.06).to(BF)
    starts = np.concatenate([[0], np.cumsum(nq)[:-1]])
    meta = Gemma4Meta(num_tokens=t, positions=None, f_page=None, f_off=None, w_page=None, w_off=None,
                      n_rows=len(rows), row_start_host=starts, row_nq_host=nq, row_kv_len_host=kv,
                      row_slot_host=np.arange(1, len(rows) + 1), row_wlo_host=wlo, wpage_table=table)
    layer = type("L", (), {"index": 0})()
    want = torch.empty(q.shape, device=DEV)
    pools_f = Gemma4Pools(wkv=wkv.float(), fkv=torch.zeros(1), page_size=ps, window=w)
    TorchSlidingAttention(pools_f).run(meta, layer, q.float(), want)
    att = FlashInferWindowAttention(pools, h, hkv, d, BF, torch.device(DEV), w)
    kv_w = kv - wlo * ps
    n_wp = (kv_w + ps - 1) // ps
    indptr = np.concatenate([[0], np.cumsum(n_wp)])

    def p32(a):
        return torch.from_numpy(np.asarray(a, np.int32)).pin_memory()

    dev = lambda a: torch.as_tensor(np.asarray(a, np.int64), device=DEV)  # noqa: E731
    idx = window_page_indices(table, dev(np.arange(1, len(rows) + 1)), dev(wlo), dev(n_wp), dev(indptr),
                              int(n_wp.max()), int(indptr[-1]), torch.arange(64, device=DEV))
    att.plan(meta, parity=0, qo_indptr=p32(np.concatenate([[0], np.cumsum(nq)])), kv_indptr=p32(indptr),
             kv_indices=idx, last_page_len=p32(kv_w - (n_wp - 1) * ps), kv_lens=p32(kv_w), max_q=int(nq.max()))
    got = torch.empty_like(q)
    att.run(meta, layer, q, got)
    torch.cuda.synchronize()
    assert torch.isfinite(got).all()
    err = (got.float() - want.float()).abs()
    assert err.max().item() < 2e-2 and err.mean().item() < 2e-3, (err.max().item(), err.mean().item())


# -- MTP: the assistant's Q-only layers; KV lengths known within bounds at planning -----------------------------------


@pytest.mark.parametrize("d", [256, 512])
def test_q_rope_matches_reference(d) -> None:
    """The assistant's input side: the q/k/v kernel with no KV heads (q = RoPE(q_norm(q)), nothing stored)."""
    torch.manual_seed(4)
    t, h = 19, 8
    q = torch.randn(t, h * d, device=DEV, dtype=BF)
    qw = torch.randn(d, device=DEV, dtype=BF) * 0.3 + 1
    pos = torch.randint(0, 4000, (t,), device=DEV)
    cos, sin = _rope_tables(d, d // 2 if d == 256 else 64)
    kw = dict(heads=h, head_dim=d, q_w=qw, eps=1e-6, positions=pos, cos_cache=cos, sin_cache=sin)
    ulp_close(gops.q_rope(q, **kw), ref(gops.q_rope, q, **kw))


def _replay_on_a_side_stream(fn):
    """Run `fn` once eagerly on a side stream, then capture it in a CUDA graph there; returns the graph."""
    g, s = torch.cuda.CUDAGraph(), torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
        torch.cuda.synchronize()
        with no_gc(), torch.cuda.graph(g, stream=s):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    return g


# (n_query, how far a row's true KV length may lie above its lowest): verify rows of 1 + k queries, k (k = 3, 5, 7);
# the drafter's rows of one query, up to 2k (its p moves with the slot's step in flight and with the step's own
# verification): 3, and 14 at k = 7
UPPER_BOUND_ROWS = [(4, 3), (1, 3), (6, 5), (8, 7), (1, 14)]


@pytest.mark.parametrize(("qo", "spread"), UPPER_BOUND_ROWS)
def test_attn512_rows_planned_for_upper_bounds(qo, spread) -> None:
    """Rows of `qo` queries (a verify row: 1 + k) planned for an upper bound of their KV length (`spread` above the
    lowest), run with the true lengths written into the device row table (`fix_len`), eagerly and in a decode bucket's
    graph replayed with other true lengths: the reference at the true lengths. Splits of the bound left empty by the
    true length (a row of 4 keys among 1,300) are skipped."""
    from types import SimpleNamespace

    from llm_mini_infer.kernels.attn512 import Attn512

    h, hkv, ps = 32, 4, 16
    lb = np.asarray([qo, 17, 256, 257, 511, 1300])
    ub = lb + spread
    n = len(lb)
    fkv, table, q, kw, cos, sin = _full_case([(qo, int(u)) for u in ub], h, hkv, ps, seed=21)
    a = Attn512(_Pools(fkv[None], ps), _Rot(cos, sin), h, hkv, 64, torch.device(DEV), max_tokens=4096, max_rows=8,
                qo_per_row=qo)
    stage = lambda a_: torch.from_numpy(np.ascontiguousarray(a_, np.int32)).pin_memory()  # noqa: E731
    rr = np.stack([np.arange(1, n + 1), np.arange(n) * qo, np.full(n, qo), ub], axis=1).astype(np.int32)
    lens = torch.zeros(n, dtype=torch.int64, device=DEV)
    meta = SimpleNamespace(page_table=table, fix_len=True, n_rows=n, kv_len_dev=lens)
    got = torch.full_like(q, float("nan"))

    def check(true: np.ndarray) -> None:
        torch.cuda.synchronize()
        want = _full_ref([(qo, int(t)) for t in true], fkv, table, q, kw, cos, sin, ps)
        err = (got.float() - want.float()).abs()
        assert torch.isfinite(got).all()
        assert err.max().item() < 2e-2 and err.mean().item() < 2e-3, (true, err.max().item(), err.mean().item())

    rng = np.random.default_rng(2)
    for true in (lb, ub, lb + rng.integers(0, spread + 1, n)):  # eager: planned for ub, run at the true lengths
        lens.copy_(torch.as_tensor(true))
        a.plan(meta, parity=0, rows=rr, stage=stage)
        a.run(meta, _Layer(kw), q, got)
        check(true)
    a.plan_graph(meta, bucket=n, rows=rr, stage=stage)
    g = _replay_on_a_side_stream(lambda: a.run(meta, _Layer(kw), q, got))
    for _ in range(4):
        true = lb + rng.integers(0, spread + 1, n)
        a.plan_graph(meta, bucket=n, rows=rr, stage=stage)
        lens.copy_(torch.as_tensor(true))
        got.fill_(float("nan"))
        g.replay()
        check(true)


def test_attn512_fixed_split_is_batch_invariant() -> None:
    """The fixed split (`split_keys`; answer reproducibility): a query at a given position gives the same bits alone
    or beside other rows, at any offset of its verify window (which positions share its tile), planned for its true
    KV length or an upper bound (MTP), eagerly or in a decode bucket's tables with padding rows, and beside a prefill
    row; and within BF16 rounding of the reference. The wave split, the control, gives other bits alone than beside
    eight rows."""
    from types import SimpleNamespace

    from llm_mini_infer.kernels.attn512 import Attn512

    h, hkv, ps, qo, d = 32, 4, 16, 6, 512
    torch.manual_seed(41)
    lens = [3200, 2700, 3050, 1200, 400, 2999, 33, 2600, 1800, 2100]  # slot 1 the subject's, the rest companions'
    npg = [-(-n // ps) for n in lens]
    fkv = (torch.randn(sum(npg) + 1, ps, hkv, d, device=DEV) * 0.5).to(BF)
    table = torch.zeros(len(lens) + 1, max(npg) + 2, dtype=torch.int32, device=DEV)
    perm = torch.randperm(sum(npg), device=DEV) + 1
    o = 0
    for i, n in enumerate(npg):
        table[i + 1, :n] = perm[o:o + n].int()
        o += n
    qsub = (torch.randn(lens[0], h, d, device=DEV) * 0.06).to(BF)  # the subject's query at each position
    kw = (torch.randn(d, device=DEV) * 0.3 + 1).to(BF)
    cos, sin = _rope_tables(d, 64, 8192)
    stage = lambda a_: torch.from_numpy(np.ascontiguousarray(a_, np.int32)).pin_memory()  # noqa: E731

    def make(split_keys: int) -> Attn512:
        return Attn512(_Pools(fkv[None], ps), _Rot(cos, sin), h, hkv, 64, torch.device(DEV), max_tokens=4096,
                       max_rows=24, qo_per_row=qo, split_keys=split_keys, max_kv=4096 + qo)

    def run(a: Attn512, rows: list[tuple[int, int, int, int]], *, bucket: int = 0, decode: int | None = None):
        """rows: (slot, first query position, queries, planned KV length above the true one); decode rows first.
        Returns the output and each row's first output row."""
        rr, qs, starts, true, at = [], [], [], [], 0
        for slot, p0, nq, extra in rows:
            rr.append((slot, at, nq, p0 + nq + extra))
            true.append(p0 + nq)
            # the companions' queries: any (they never reach the subject's output)
            qs.append(qsub[p0:p0 + nq] if slot == 1 else (torch.randn(nq, h, d, device=DEV) * 0.06).to(BF))
            starts.append(at)
            at += nq
        n = len(rows)
        if bucket:  # padding rows: slot 0 (the null page), 1 + k keys
            rr += [(0, at + qo * i, qo, qo) for i in range(bucket - n)]
            qs.append(torch.zeros((bucket - n) * qo, h, d, dtype=BF, device=DEV))
            true += [qo] * (bucket - n)
        q = torch.cat(qs).contiguous()
        kv_dev = torch.as_tensor(true, dtype=torch.int64, device=DEV)
        meta = SimpleNamespace(page_table=table, fix_len=True, n_rows=len(rr), kv_len_dev=kv_dev)
        rows_np = np.asarray(rr, np.int32)
        if bucket:
            a.plan_graph(meta, bucket=bucket, rows=rows_np, stage=stage, real=n)
        else:
            a.plan(meta, parity=0, rows=rows_np, stage=stage, fixed_rows=n if decode is None else decode)
        out = torch.full_like(q, float("nan"))
        a.run(meta, _Layer(kw), q, out)
        torch.cuda.synchronize()
        return out, starts

    fixed, waves = make(256), make(0)
    for p in (255, 256, 1000, 2559, 2560, 2600, 3190):
        def at(j: int, extra: int = 0, p: int = p) -> tuple[int, int, int, int]:  # the subject's row, p its j-th query
            return (1, p - j, qo, extra)

        others = [(s, lens[s - 1] - qo, qo, 0) for s in range(2, 10)]  # eight companions' verify rows
        configs = [
            ([at(0)], {}),  # alone
            ([at(3, 5)], {}),  # another offset, planned for an upper bound
            ([*others, at(5)], {}),  # beside eight rows, last
            ([at(1, 2), *others[:2]], {"bucket": 8}),  # a decode bucket with padding rows, an upper bound
            ([*others, at(4)], {"bucket": 24}),  # a bigger bucket
            ([at(2), (2, 600, 40, 0)], {"decode": 1}),  # beside a prefill row (one split, the whole of its keys)
        ]
        outs = []
        for rows, kw_run in configs:
            out, starts = run(fixed, rows, **kw_run)
            i = next(r for r, row in enumerate(rows) if row[0] == 1)
            outs.append(out[starts[i] + p - rows[i][1]])
        assert torch.isfinite(outs[0]).all()
        for c, got in enumerate(outs[1:], 1):
            assert torch.equal(outs[0], got), (p, c)
        want = _full_ref([(1, p + 1)], fkv, table[[0, 1]], qsub[p:p + 1].contiguous(), kw, cos, sin, ps)[0]
        assert (outs[0].float() - want.float()).abs().max().item() < 2e-2, p
    # the control: the wave split's merge order depends on the batch (31 splits alone, 3 beside eight rows)
    p = 2600
    alone, s1 = run(waves, [(1, p, qo, 0)])
    beside, s2 = run(waves, [*[(s, lens[s - 1] - qo, qo, 0) for s in range(2, 10)], (1, p, qo, 0)])
    assert not torch.equal(alone[s1[0]], beside[s2[-1]])


# -- the fixed split by each query's own position (long contexts; prefill) -------------------------------------------


def _split_case(lens: list[int], seed: int):
    """Slots 1.. of `lens[i]` positions on shuffled pages, and slot 1's (the subject's) query at each of its positions:
    a query at a position has the same values wherever a test places it."""
    from types import SimpleNamespace

    h, hkv, ps, d = 32, 4, 16, 512
    torch.manual_seed(seed)
    npg = [-(-n // ps) for n in lens]
    fkv = (torch.randn(sum(npg) + 1, ps, hkv, d, device=DEV) * 0.5).to(BF)
    table = torch.zeros(len(lens) + 1, max(npg) + 2, dtype=torch.int32, device=DEV)
    perm = torch.randperm(sum(npg), device=DEV) + 1
    o = 0
    for i, n in enumerate(npg):
        table[i + 1, :n] = perm[o:o + n].int()
        o += n
    q = (torch.randn(lens[0], h, d, device=DEV) * 0.06).to(BF)
    kw = (torch.randn(d, device=DEV) * 0.3 + 1).to(BF)
    cos, sin = _rope_tables(d, 64, 8192)
    return SimpleNamespace(fkv=fkv, table=table, q=q, kw=kw, cos=cos, sin=sin, h=h, hkv=hkv, ps=ps, d=d)


def _split_attn(case, *, max_rows: int, max_kv: int, qo: int = 6, split_keys: int = 16, prefill_from: int = 64,
                prefill_keys: int = 32):
    from llm_mini_infer.kernels.attn512 import Attn512

    return Attn512(_Pools(case.fkv[None], case.ps), _Rot(case.cos, case.sin), case.h, case.hkv, 64,
                   torch.device(DEV), max_tokens=4096, max_rows=max_rows, qo_per_row=qo, split_keys=split_keys,
                   max_kv=max_kv, prefill_from=prefill_from, prefill_keys=prefill_keys)


def _split_run(case, a, rows: list[tuple[int, int, int, int]], *, bucket: int = 0, fixed: int | None = None,
               graph: bool = False, gap: int = 0, qo: int = 6, unordered: bool = False):
    """One step of `rows` (slot, first query position, queries, planned KV length above the true one), the first
    `fixed` (None: all) decode rows and the rest prefill rows, each decode row planned from the lowest KV length `gap`
    below its true one. With `bucket`: the decode bucket's tables (padding rows on the null page after the real ones),
    and with `graph` run in a CUDA graph. Returns slot 1's output by query position (at most one row of the step is
    slot 1's) and the step's launches."""
    from types import SimpleNamespace

    stage = lambda a_: torch.from_numpy(np.ascontiguousarray(a_, np.int32)).pin_memory()  # noqa: E731
    rr, qs, true, at, where = [], [], [], 0, None
    for slot, p0, nq, extra in rows:
        rr.append((slot, at, nq, p0 + nq + extra))
        true.append(p0 + nq)
        if slot == 1:  # the other rows' queries are any (they never reach slot 1's output)
            qs.append(case.q[p0:p0 + nq])
            where = (at, p0, nq)
        else:
            qs.append((torch.randn(nq, case.h, case.d, device=DEV) * 0.06).to(BF))
        at += nq
    n = len(rows)
    lb = np.maximum(np.asarray(true) - gap, [r[2] for r in rows])
    if bucket:
        rr += [(0, at + qo * i, qo, qo) for i in range(bucket - n)]
        qs.append(torch.zeros((bucket - n) * qo, case.h, case.d, dtype=BF, device=DEV))
        true += [qo] * (bucket - n)
    q = torch.cat(qs).contiguous()
    meta = SimpleNamespace(page_table=case.table, fix_len=True, n_rows=len(rr),
                           kv_len_dev=torch.as_tensor(true, dtype=torch.int64, device=DEV))
    rows_np = np.asarray(rr, np.int32)
    out = torch.full_like(q, float("nan"))
    if bucket:
        a.plan_graph(meta, bucket=bucket, rows=rows_np, stage=stage, real=n, lb=lb)
    else:
        a.plan(meta, parity=0, rows=rows_np, stage=stage, fixed_rows=n if fixed is None else fixed, lb=lb)
    if unordered:  # the split launches in the items' own order, not split-major: the same bits
        dr, di, dg, nd, runs, do = meta.full_plan
        meta.full_plan = (dr, di, dg, nd, tuple((*r[:5], False) for r in runs), do)
    if graph:
        g = _replay_on_a_side_stream(lambda: a.run(meta, _Layer(case.kw), q, out))
        out.fill_(float("nan"))
        g.replay()
    else:
        a.run(meta, _Layer(case.kw), q, out)
    torch.cuda.synchronize()
    start, p0, nq = where
    assert torch.isfinite(out).all()
    return {p0 + j: out[start + j].clone() for j in range(nq)}, meta.full_plan[4]


def test_attn512_query_sizes_are_batch_invariant() -> None:
    """The fixed split sizes each query's splits from its own position (`split_keys` · 2^m, doubling where its keys
    pass 32 splits; 16 keys here, so past 512, 1,024, ... keys): a verify row's queries at positions across a size
    boundary (the split count drops there, a tile's two positions may straddle it) give the same bits for each
    position whatever else shares the step and wherever its row's window lies: alone, with its window ending a few
    positions earlier or later, beside eight other verify rows, in a decode bucket with padding rows (eagerly and in a
    CUDA graph), planned for an upper bound with the lowest possible length below the true one (a true length just
    under a boundary while the bound is past it), and beside a prefill row; and within BF16 rounding of the fp32
    reference."""
    ps, qo = 16, 6
    lens = [2200, 1100, 700, 2100, 300, 1500, 900, 2050, 64]  # slot 1 the subject's, the rest companions'
    case = _split_case(lens, seed=47)
    a = _split_attn(case, max_rows=16, max_kv=lens[0] + qo)
    others = [(s, lens[s - 1] - qo, qo, 0) for s in range(2, 10)]  # eight companions' verify rows
    sizes = ((516, 512), (1026, 1024), (2050, 2048), (300, 0), (800, 0), (1500, 0))  # window end, size boundary
    for end, bound in sizes:  # the window covers p + 1 = end - 5 .. end: 511..516, 1021..1026, 2045..2050
        def at(shift: int, extra: int = 0, end: int = end) -> tuple[int, int, int, int]:
            return (1, end + shift - qo, qo, extra)

        configs = [
            ([at(0)], {}),  # alone: the base
            ([at(1)], {}),  # its window one later: the tile's positions pair differently
            ([at(-1, 4)], {"gap": 2}),  # one earlier, planned for an upper bound
            ([at(3, 6)], {"gap": 3}),
            ([*others, at(0)], {}),  # beside eight rows, last
            ([at(-2), *others[:3]], {}),  # first
            ([at(1, 5), *others[:3]], {"bucket": 16, "gap": 2}),  # a decode bucket's tables with padding rows
            ([*others[:5], at(-3, 3)], {"bucket": 16, "gap": 1, "graph": True}),  # and in a CUDA graph
            ([at(0), (2, 600, 40, 0)], {"fixed": 1}),  # beside a prefill row (its queries past prefill_from split)
            ([*others[:3], at(2), (5, 100, 70, 0)], {"fixed": 4}),
        ]
        if bound:  # the true length under the boundary, the bound past it: its split count is the most (32)
            configs += [([(1, bound - qo, qo, 8)], {"gap": 2}),
                        ([(1, bound - qo, qo, 8), *others[:3]], {"bucket": 16, "gap": 2, "graph": True})]
        base, _ = _split_run(case, a, configs[0][0])
        assert len(base) == qo
        for rows, kw_run in configs[1:]:
            got, _ = _split_run(case, a, rows, **kw_run)
            shared = [p for p in got if p in base]
            assert shared, (end, rows)
            for p in shared:
                assert torch.equal(base[p], got[p]), (end, p, rows, kw_run)
        for p, row in base.items():
            want = _full_ref([(1, p + 1)], case.fkv, case.table[[0, 1]], case.q[p:p + 1].contiguous(), case.kw,
                             case.cos, case.sin, ps)[0]
            assert (row.float() - want.float()).abs().max().item() < 2e-2, (end, p)


def test_attn512_prefill_split_is_chunk_invariant() -> None:
    """A prefill query takes its keys whole up to `prefill_from` and splits them past it at absolute multiples of its
    own size (`prefill_keys` · 2^m): one prompt's positions 40..700 (`prefill_from` 64, 32 keys a split) give the same
    bits for each query as one prefill row and as consecutive chunks of other lengths and offsets (a chunk of one
    token, a chunk that cuts a tile of 16 query rows between two positions, chunks that begin before and past 64),
    each chunk alone and beside decode rows (verify rows first) and another prefill row; and within BF16 rounding of
    the reference. The step's split items pass the partial buffer: the chunk runs in several launches."""
    ps, qo, p0, end = 16, 6, 40, 701
    lens = [end, 1100, 900, 400]
    case = _split_case(lens, seed=53)
    a = _split_attn(case, max_rows=16, max_kv=1200)
    dec = [(2, lens[1] - qo, qo, 0), (3, lens[2] - qo, qo, 3)]  # decode rows, the second planned above its length
    side = (4, 100, 50, 0)  # another prefill row

    def prompt(chunks: list[int], beside: bool) -> torch.Tensor:
        outs, at, many = {}, p0, 0
        for n in chunks:
            row = (1, at, n, 0)
            if beside:
                got, runs = _split_run(case, a, [*dec, row, side], fixed=len(dec), gap=2)
            else:
                got, runs = _split_run(case, a, [row], fixed=0)
            outs.update(got)
            many = max(many, len(runs))
            at += n
        assert sorted(outs) == list(range(p0, end))
        assert many > 1, chunks  # the chunks' items pass the partial buffer: several launches
        return torch.stack([outs[p] for p in range(p0, end)])

    base = prompt([end - p0], beside=False)
    for chunks in ([end - p0], [24, 637], [100, 1, 560], [333, 328]):
        for beside in (False, True):
            assert torch.equal(prompt(chunks, beside), base), (chunks, beside)
    want = _full_ref([(end - p0, end)], case.fkv, case.table[[0, 1]], case.q[p0:end].contiguous(), case.kw, case.cos,
                     case.sin, ps)
    assert (base.float() - want.float()).abs().max().item() < 2e-2


def test_attn512_prefill_below_the_threshold_is_unsplit() -> None:
    """The prefill split changes no bit of a query whose keys are at most `prefill_from` (p + 1 <= 64): positions
    41..63 of a prompt equal the unsplit path's (an instance whose `prefill_from` is past the prompt), though the tile
    of positions 63 and 64 splits for 64's sake and the merge of its one live split for 63 must equal the direct
    write; the later positions differ (the control: the split took effect). And the split launches cut by a small
    partial buffer (`max_rows` 1: a few hundred items) give the bits of one launch, for every query."""
    ps, p0, end = 16, 41, 301
    case = _split_case([end], seed=59)
    split = _split_attn(case, max_rows=32, max_kv=end)
    small = _split_attn(case, max_rows=1, max_kv=end)
    whole = _split_attn(case, max_rows=1, max_kv=end, prefill_from=1 << 20)
    row = [(1, p0, end - p0, 0)]
    got, runs = _split_run(case, split, row, fixed=0)
    cut, cut_runs = _split_run(case, small, row, fixed=0)
    unsplit, one = _split_run(case, whole, row, fixed=0)
    assert len(runs) == 1 and len(cut_runs) > 2 and len(one) == 1
    assert sorted(got) == sorted(cut) == sorted(unsplit) == list(range(p0, end))
    plain, _ = _split_run(case, split, row, fixed=0, unordered=True)
    for p in range(p0, end):
        assert torch.equal(got[p], cut[p]), p  # one launch and several
        assert torch.equal(got[p], plain[p]), p  # split-major and group-major
    for p in range(p0, 64):
        assert torch.equal(got[p], unsplit[p]), p  # p + 1 <= 64
    assert any(not torch.equal(got[p], unsplit[p]) for p in range(64, end))
    want = _full_ref([(end - p0, end)], case.fkv, case.table[[0, 1]], case.q[p0:end].contiguous(), case.kw, case.cos,
                     case.sin, ps)
    for p in range(p0, end):
        assert (got[p].float() - want[p - p0].float()).abs().max().item() < 2e-2, p


# -- the sliding window on our kernel (answer reproducibility: kernels/attn_window.py) --------------------------------


class _WPools:
    def __init__(self, wkv, ps):
        self.wkv, self.page_size = wkv, ps


@pytest.mark.parametrize("split_keys", [48, 64, 256])
def test_attn_window_matches_reference(split_keys) -> None:
    """Decode rows of 1 and 6 queries (split at absolute multiples of `split_keys`) and prefill rows (whole, tiles of
    64 query rows), shorter and longer than the window, from position 0 and resumed, one with an image span: the fp32
    reference within BF16 rounding."""
    from llm_mini_infer.kernels.attn_window import WindowAttention

    h, hkv, d, ps, w = 32, 16, 256, 16, 1024
    lens = [6, 300, 1030, 2700, 3001, 900, 2100]
    pool, table = _window_setup(lens, hkv, d, ps, seed=3)
    a = WindowAttention(_WPools(pool[None], ps), h, hkv, d, torch.device(DEV), w, max_tokens=4096, max_rows=8,
                        qo_per_row=6, split_keys=split_keys)
    # (slot, first position, queries): decode rows, then prefill rows
    spec = [(1, 0, 6), (2, 294, 6), (3, 1024, 6), (4, 2699, 1), (5, 2995, 6), (6, 0, 900), (7, 1900, 200)]
    span = {7: (1950, 2016)}  # slot 7's image span: 66 queries see its later keys
    qs = [(torch.randn(n, h, d, device=DEV) * 0.06).to(BF) for _, _, n in spec]
    rows, ends, at = [], [], 0
    for (slot, p0, n) in spec:
        rows.append((slot, at, n, p0 + n))
        ends += [span[slot][1] if slot in span and span[slot][0] <= p0 + i < span[slot][1] else 0 for i in range(n)]
        at += n
    got = _window_run(a, pool, table, rows, torch.cat(qs), decode=5, span_end=ends)
    at = 0
    for (slot, p0, n), q in zip(spec, qs):
        want = _window_ref(pool, table, slot, q, p0, w, ps, span.get(slot))
        err = (got[at:at + n].float() - want).abs()
        assert torch.isfinite(got[at:at + n]).all(), slot
        assert err.max().item() < 2e-2 and err.mean().item() < 2e-3, (slot, err.max().item(), err.mean().item())
        at += n


def test_attn_window_is_batch_invariant() -> None:
    """A query at a given position gives the same bits as a decode row's query alone or beside other rows, at any
    offset of its verify window, planned for its true KV length or an upper bound (MTP), in a decode bucket's static
    tables with padding rows, and beside a prefill row; and the same bits again as a prefill row's query whatever the
    chunk and the rows beside it, an image span's query too. The two kinds are each one rule (prefill rows whole in
    tiles of 64 query rows, decode rows split)."""
    from llm_mini_infer.kernels.attn_window import WindowAttention

    h, hkv, d, ps, w, qo = 32, 16, 256, 16, 1024, 6
    lens = [3200, 2700, 3050, 1200, 400, 2999, 33, 2600, 1800, 2100]  # slot 1 the subject's
    pool, table = _window_setup(lens, hkv, d, ps, seed=7)
    torch.manual_seed(8)
    qsub = (torch.randn(lens[0], h, d, device=DEV) * 0.06).to(BF)
    a = WindowAttention(_WPools(pool[None], ps), h, hkv, d, torch.device(DEV), w, max_tokens=4096, max_rows=24,
                        qo_per_row=qo, split_keys=256)

    def go(rows_spec, spans=(), **kw):
        """rows_spec: (slot, first position, queries, KV planned above the true length); spans: the subject's image
        spans [a, e) (inside its prefill row); returns the subject's output rows by position."""
        rows, qs, true, ends, at, where = [], [], [], [], 0, {}
        for slot, p0, n, extra in rows_spec:
            rows.append((slot, at, n, p0 + n + extra))
            true.append(p0 + n)
            qs.append(qsub[p0:p0 + n] if slot == 1 else (torch.randn(n, h, d, device=DEV) * 0.06).to(BF))
            ends += [next((e for a_, e in spans if slot == 1 and a_ <= p0 + i < e), 0) for i in range(n)]
            if slot == 1:
                where.update({p0 + i: at + i for i in range(n)})
            at += n
        bucket = kw.get("bucket", 0)
        for _ in range(bucket - len(rows)):  # padding rows: slot 0's null page, 1 + k keys
            rows.append((0, at, qo, qo))
            true.append(qo)
            qs.append(torch.zeros(qo, h, d, dtype=BF, device=DEV))
            ends += [0] * qo
            at += qo
        out = _window_run(a, pool, table, rows, torch.cat(qs), true=true, span_end=ends if spans else None, **kw)
        return {p: out[i] for p, i in where.items()}

    others = [(s, lens[s - 1] - qo, qo, 0) for s in range(2, 10)]
    for p in (100, 1023, 1024, 1040, 2559, 2600, 3190):
        def at(j: int, extra: int = 0, p: int = p) -> tuple[int, int, int, int]:  # the subject's row, p its j-th query
            return (1, p - j, qo, extra)

        decode = [go([at(0)]), go([at(3, 5)]), go([*others, at(5)]), go([at(1, 2), *others[:2]], bucket=8),
                  go([*others, at(4)], bucket=24), go([at(2), (2, 600, 40, 0)], decode=1)]
        for c, got in enumerate(decode[1:], 1):
            assert torch.equal(decode[0][p], got[p]), (p, c)
        prefill = [go([(1, p, 1, 0)], decode=0), go([(1, p - 37, 40, 0), *others[:3]], decode=0),
                   go([*others[:2], (1, max(0, p - 300), min(p, 300) + 1, 0)], decode=2)]
        for c, got in enumerate(prefill[1:], 1):
            assert torch.equal(prefill[0][p], got[p]), (p, c)
        want = _window_ref(pool, table, 1, qsub[p:p + 1], p, w, ps)[0]
        for got in (decode[0][p], prefill[0][p]):
            assert (got.float() - want).abs().max().item() < 2e-2, p
        # an image span of 66 queries about p (inside the subject's keys): its queries alone, from a chunk starting
        # elsewhere, beside other rows
        sp = [(min(max(0, p - 20), lens[0] - 66 - 14), min(max(0, p - 20), lens[0] - 66 - 14) + 66)]
        a0, e0 = sp[0]
        spanned = [go([(1, a0, 66, 0)], spans=sp, decode=0),
                   go([*others[:2], (1, max(0, a0 - 45), 66 + a0 - max(0, a0 - 45) + 9, 0)], spans=sp, decode=2),
                   go([(2, 600, 40, 0), (1, a0, 80, 0), (3, 100, 77, 0)], spans=sp, decode=0)]
        for c, got in enumerate(spanned[1:], 1):
            for x in range(a0, e0):
                assert torch.equal(spanned[0][x], got[x]), (p, x, c)
        want = _window_ref(pool, table, 1, qsub[a0:e0], a0, w, ps, sp[0])
        got = torch.stack([spanned[0][x] for x in range(a0, e0)])
        assert (got.float() - want).abs().max().item() < 2e-2, p


@pytest.mark.parametrize(("qo", "spread"), UPPER_BOUND_ROWS)
def test_flashinfer_window_planned_for_upper_bounds(qo, spread) -> None:
    """Window rows planned for an upper bound of their KV length (`spread` above the lowest) and named from the first
    window page of the lowest possible first query (the pages before it garbage), run with the true named lengths fixed
    on the device, eagerly (`fix_len`) and in a decode graph (window pages gathered at the static first column, lengths
    fixed inside, replayed with other true lengths): the reference sliding attention at the true lengths."""
    from llm_mini_infer.kernels.flashinfer import FlashInferWindowAttention, window_page_indices
    from llm_mini_infer.models.gemma4.backend import Gemma4Meta, Gemma4Pools, TorchSlidingAttention

    torch.manual_seed(6)
    w, ps, h, hkv, d = 64, 16, 8, 4, 256
    lb = np.asarray([qo, 40, 63 + qo, 64, 129, 300])  # each row's lowest KV length after the step
    ub = lb + spread
    n = len(lb)
    wlo = np.maximum(lb - qo - w + 1, 0) // ps  # from the lowest first query
    npg = -(-ub // ps)
    wkv = (torch.randn(1, int(npg.sum()) + 1, 2, ps, hkv, d, device=DEV) * 0.5).to(BF)
    table = torch.zeros(n + 1, int(npg.max()) + 2, dtype=torch.int32, device=DEV)
    o = 1
    for r in range(n):
        table[r + 1, :npg[r]] = torch.arange(o, o + npg[r], dtype=torch.int32)
        o += int(npg[r])
        wkv[0, table[r + 1, :wlo[r]].long()] = float("nan")  # released below the window: never read
    pools = Gemma4Pools(wkv=wkv, fkv=torch.zeros(1), page_size=ps, window=w)
    pools_f = Gemma4Pools(wkv=wkv.float(), fkv=torch.zeros(1), page_size=ps, window=w)
    q = (torch.randn(n * qo, h, d, device=DEV) * 0.06).to(BF)
    layer = type("L", (), {"index": 0})()
    slots = np.arange(1, n + 1)
    kv_w = ub - wlo * ps
    n_wp = (kv_w + ps - 1) // ps
    indptr = np.concatenate([[0], np.cumsum(n_wp)])
    p32 = lambda a_: torch.from_numpy(np.asarray(a_, np.int32)).pin_memory()  # noqa: E731
    dev = lambda a_: torch.as_tensor(np.asarray(a_, np.int64), device=DEV)  # noqa: E731
    plan = dict(parity=0, qo_indptr=p32(np.arange(n + 1) * qo), kv_indptr=p32(indptr),
                last_page_len=p32(kv_w - (n_wp - 1) * ps), kv_lens=p32(kv_w), max_q=qo, lb=lb - wlo * ps)

    def meta_at(true: np.ndarray) -> Gemma4Meta:
        return Gemma4Meta(num_tokens=n * qo, positions=None, f_page=None, f_off=None, w_page=None, w_off=None,
                          n_rows=n, row_start_host=np.arange(n) * qo, row_nq_host=np.full(n, qo),
                          row_kv_len_host=true, row_slot_host=slots, row_wlo_host=wlo, wpage_table=table)

    named = torch.zeros(n, dtype=torch.int64, device=DEV)
    got = torch.empty_like(q)

    def check(true: np.ndarray) -> None:
        torch.cuda.synchronize()
        want = torch.empty(q.shape, device=DEV)
        TorchSlidingAttention(pools_f).run(meta_at(true), layer, q.float(), want)
        err = (got.float() - want).abs()
        assert torch.isfinite(got).all()
        assert err.max().item() < 2e-2 and err.mean().item() < 2e-3, (true, err.max().item(), err.mean().item())

    att = FlashInferWindowAttention(pools, h, hkv, d, BF, torch.device(DEV), w, max_rows=n,
                                    max_pages=int(n_wp.max()) + 1, qo_per_row=qo)
    idx = window_page_indices(table, dev(slots), dev(wlo), dev(n_wp), dev(indptr), int(n_wp.max()), int(indptr[-1]),
                              torch.arange(64, device=DEV))
    rng = np.random.default_rng(5)
    for true in (lb, ub, lb + rng.integers(0, spread + 1, n)):
        m = meta_at(ub)
        m.fix_len, m.kv_named_dev = True, named
        named.copy_(torch.as_tensor(true - wlo * ps))
        att.plan(m, kv_indices=idx, **plan)
        att.run(m, layer, q, got)
        check(true)
    gm = meta_at(ub)
    att.plan_graph(gm, rows=n, **plan)
    att.g_first[:n].copy_(dev(wlo))
    slots_d = dev(slots)

    def body() -> None:
        att.fix_lengths(gm.sliding_plan, n, named)
        att.graph_window_indices(n, table, slots_d)
        att.run(gm, layer, q, got)

    g = _replay_on_a_side_stream(body)
    for _ in range(4):
        true = lb + rng.integers(0, spread + 1, n)
        att.plan_graph(gm, rows=n, **plan)
        named.copy_(torch.as_tensor(true - wlo * ps))
        got.fill_(float("nan"))
        g.replay()
        check(true)
