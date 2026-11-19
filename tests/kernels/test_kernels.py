"""Our CUDA kernels against their torch references (kernels/ops.py), on random inputs. CUDA only (skipped on a Mac).

Norms and activations round at the same points as the references; only the order of a reduction differs, so outputs
agree to one BF16 ulp. The conv is bit-exact. The GDN decode kernel keeps the vendored kernel's arithmetic (fast exp,
warp-reduction order), so it is compared with a tolerance. The sampler must reproduce the reference token for token.
The FP8 quantizing producers equal their BF16 kernel followed by `fp8.quant_ref` bit for bit (SiLU-and-mul its fp32
product and `quant_ref`'s division); the block-FP8 GEMM matches the dequantized fp32 product within a BF16 rounding.
The Qwen kernels' tests run with programmatic dependent launch off and on (`pdl`): on, their outputs are the serialized
launches' bit for bit.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from llm_mini_infer.kernels import fp8, ops
from llm_mini_infer.runner.cuda.graphs import no_gc
from llm_mini_infer.runner.cuda.sampler import plan_sampling, row_keys, sample

from ..support.kernel_cases import BF, DEV, GuardedPool, kv_slots, ref, ulp_close
from ..support.sampler_cases import (
    FIT_CASES,
    FIT_DRAWS,
    FIT_SEED,
    assert_fits,
    fit_row,
    largest_noise_rows,
    target_distribution,
    terms_tables,
)

pytestmark = pytest.mark.gpu(reason="needs a CUDA GPU and the kernel library")


@pytest.fixture(params=[False, True], ids=["serialized", "pdl"])
def pdl(request) -> bool:
    """The kernel library's programmatic-dependent-launch flag (kernels/csrc/common.cuh), off and then on, for a test
    of kernels launched through `lmk_launch`: each waits for the kernel before it as its first statement, so with the
    flag on its outputs are the serialized launch's bit for bit, and a slip in a launch (its grid, shared memory or an
    argument) shows. A test whose comparison with the reference is exact proves that by passing both ways; the others
    also compare with the serialized launch (`same_as_serialized`). The conftest restores the flag after the test."""
    assert ops.set_pdl(request.param) is request.param
    return request.param


def serialized(fn, *args, **kw):
    """`fn(*args, **kw)` with the library's PDL flag off (every launch serialized), then the flag as it was."""
    on = ops.pdl()
    ops.set_pdl(False)
    try:
        return fn(*args, **kw)
    finally:
        ops.set_pdl(on)


def bits(t: torch.Tensor) -> torch.Tensor:
    """A tensor's bytes (a NaN equals itself)."""
    return t.contiguous().view(torch.uint8)


def _outputs(x) -> list[torch.Tensor]:
    """The tensors of an op's result: a tensor, a QuantAct (its m tokens' scales), or a tuple of them."""
    if x is None:
        return []
    if isinstance(x, torch.Tensor):
        return [x]
    if isinstance(x, fp8.QuantAct):
        return [x.q, x.s[:, :x.m], *_outputs(x.bf16)]
    return [t for y in x for t in _outputs(y)]


def same_as_serialized(pdl: bool, got, fn, *args, **kw) -> None:
    """With the PDL flag on, `got` (an op's result) is what `fn(*args, **kw)` returns with every launch serialized, bit
    for bit; for ops that write nothing but what they return."""
    if pdl:
        for a, b in zip(_outputs(got), _outputs(serialized(fn, *args, **kw)), strict=True):
            assert torch.equal(bits(a), bits(b))


@pytest.mark.parametrize("d", [1024, 5120])
def test_rmsnorm(d, pdl) -> None:
    torch.manual_seed(0)
    x = torch.randn(37, d, device=DEV, dtype=BF) * 3
    res = torch.randn(37, d, device=DEV, dtype=BF)
    w = torch.randn(d, device=DEV, dtype=BF) * 0.2
    y = ops.rmsnorm(x, w, 1e-6)
    ulp_close(y, ref(ops.rmsnorm, x, w, 1e-6))
    same_as_serialized(pdl, y, ops.rmsnorm, x, w, 1e-6)
    o, r = ops.rmsnorm(x, w, 1e-6, res)
    o2, r2 = ref(ops.rmsnorm, x, w, 1e-6, res)
    assert torch.equal(r, r2)  # the rounded sum is exact
    ulp_close(o, o2)
    same_as_serialized(pdl, (o, r), ops.rmsnorm, x, w, 1e-6, res)


def test_gated_rmsnorm_silu_mul_gate(pdl) -> None:
    torch.manual_seed(1)
    x = torch.randn(29, 16, 128, device=DEV, dtype=BF)
    zbig = torch.randn(29, 16 * 128 + 64, device=DEV, dtype=BF)
    z = zbig[:, 64:].view(29, 16, 128)  # a strided view, as the in_proj split gives
    w = torch.randn(128, device=DEV, dtype=BF)
    y = ops.gated_rmsnorm(x, z, w, 1e-6)
    ulp_close(y, ref(ops.gated_rmsnorm, x, z, w, 1e-6))
    same_as_serialized(pdl, y, ops.gated_rmsnorm, x, z, w, 1e-6)
    gu = torch.randn(29, 2 * 3584, device=DEV, dtype=BF) * 2
    assert torch.equal(ops.silu_mul(gu), ref(ops.silu_mul, gu))
    o = torch.randn(29, 8 * 256, device=DEV, dtype=BF)
    qg = torch.randn(29, 8, 512, device=DEV, dtype=BF)
    gate = qg[..., 256:]
    assert torch.equal(ops.sigmoid_gate(o, gate), ref(ops.sigmoid_gate, o, gate))


@pytest.mark.parametrize("gated", [True, False])
def test_qk_norm_rope_and_kv_write(gated, pdl) -> None:
    """q, and K and V into exactly their slots of a guarded pool (`GuardedPool`, `kv_slots`); V, copied as is, bit for
    bit."""
    torch.manual_seed(2)
    h, hkv, d, rot, ps = 8, 2, 256, 64, 16
    page, off = kv_slots(ps)
    t = len(page)
    qkv = torch.randn(t, h * d * (2 if gated else 1) + 2 * hkv * d, device=DEV, dtype=BF)
    qkv[-1] = qkv[-2]  # the padding tokens
    qw, kw = torch.randn(d, device=DEV, dtype=BF) * 0.1, torch.randn(d, device=DEV, dtype=BF) * 0.1
    pos = torch.randint(0, 4000, (t,), device=DEV)
    pos[-2:] = 0
    freqs = torch.outer(torch.arange(4096.0, device=DEV), 1 / 1e7 ** (torch.arange(0, rot, 2, device=DEV) / rot))
    cos, sin = freqs.cos().to(BF), freqs.sin().to(BF)
    pool = GuardedPool((10, 2, ps, hkv, d), seed=3)
    b1, b2 = pool.backing.clone(), pool.backing.clone()
    kw_args = dict(heads=h, kv_heads=hkv, head_dim=d, gated=gated, q_w=qw, k_w=kw, eps=1e-6, positions=pos,
                   cos_cache=cos, sin_cache=sin, rotary_dim=rot, kv_page=page, kv_off=off)
    q1 = ops.qk_norm_rope(qkv, kv_layer=pool.view(b1), **kw_args)
    q2 = ref(ops.qk_norm_rope, qkv, kv_layer=pool.view(b2), **kw_args)
    ulp_close(q1, q2)
    pool.check(b1, b2, page, off)
    v1, v2 = (pool.view(b)[page, 1, off].view(torch.int16) for b in (b1, b2))
    assert torch.equal(v1, v2)
    if pdl:  # the serialized launch's bits, the pool's included
        b3 = pool.backing.clone()
        assert torch.equal(q1, serialized(ops.qk_norm_rope, qkv, kv_layer=pool.view(b3), **kw_args))
        assert torch.equal(bits(b1), bits(b3))


def test_qk_norm_rope_mrope(pdl) -> None:
    """Interleaved M-RoPE (images): [3, T] positions whose components differ, against the reference; and equal
    components give the 1-D result bit for bit."""
    torch.manual_seed(12)
    t, h, hkv, d, rot, ps = 41, 8, 2, 256, 64, 16
    qkv = torch.randn(t, h * d * 2 + 2 * hkv * d, device=DEV, dtype=BF)
    qw, kw = torch.randn(d, device=DEV, dtype=BF) * 0.1, torch.randn(d, device=DEV, dtype=BF) * 0.1
    freqs = torch.outer(torch.arange(4096.0, device=DEV), 1 / 1e7 ** (torch.arange(0, rot, 2, device=DEV) / rot))
    cos, sin = freqs.cos().to(BF), freqs.sin().to(BF)
    page = torch.arange(t, device=DEV) // ps + 1
    off = torch.arange(t, device=DEV) % ps
    base = torch.randint(0, 3000, (t,), device=DEV)
    pos3 = torch.stack([base, base + torch.randint(0, 40, (t,), device=DEV), base + torch.randint(0, 40, (t,),
                                                                                             device=DEV)])
    kw_args = dict(heads=h, kv_heads=hkv, head_dim=d, gated=True, q_w=qw, k_w=kw, eps=1e-6, cos_cache=cos,
                   sin_cache=sin, rotary_dim=rot, kv_page=page, kv_off=off, mrope=(11, 10))
    kv1 = torch.zeros(4, 2, ps, hkv, d, device=DEV, dtype=BF)
    kv2 = kv1.clone()
    q1 = ops.qk_norm_rope(qkv, kv_layer=kv1, positions=pos3, **kw_args)
    ulp_close(q1, ref(ops.qk_norm_rope, qkv, kv_layer=kv2, positions=pos3, **kw_args))
    ulp_close(kv1, kv2)
    if pdl:  # the serialized launch's bits, the KV's included
        kv5 = torch.zeros_like(kv1)
        assert torch.equal(q1, serialized(ops.qk_norm_rope, qkv, kv_layer=kv5, positions=pos3, **kw_args))
        assert torch.equal(kv1, kv5)
    same = base[None].expand(3, -1).contiguous()
    kv3, kv4 = torch.zeros_like(kv1), torch.zeros_like(kv1)
    one = {k: v for k, v in kw_args.items() if k != "mrope"}
    assert torch.equal(ops.qk_norm_rope(qkv, kv_layer=kv3, positions=same, **kw_args),
                       ops.qk_norm_rope(qkv, kv_layer=kv4, positions=base, **one))
    assert torch.equal(kv3, kv4)


@pytest.mark.parametrize(("d", "pad"), [(72, 128), (64, 64)])
def test_vision_qkv_rope_exact(d, pad) -> None:
    """The vision tower's rotary and padding: bit-exact with the torch reference."""
    torch.manual_seed(13)
    n, h = 300, 16
    qkv = torch.randn(n, 3 * h * d, device=DEV, dtype=BF)
    ang = torch.rand(n, d // 2, device=DEV) * 40
    ang = torch.cat([ang, ang], dim=-1)
    cos, sin = ang.cos(), ang.sin()
    got = ops.vision_qkv_rope(qkv, heads=h, head_dim=d, cos=cos, sin=sin, pad_to=pad)
    want = ref(ops.vision_qkv_rope, qkv, heads=h, head_dim=d, cos=cos, sin=sin, pad_to=pad)
    for a, b in zip(got, want):
        assert a.shape == (n, h, pad) and torch.equal(a, b)


@pytest.mark.parametrize(("d", "pad", "n", "h"), [(72, 128, 648, 16), (72, 128, 300, 16), (72, 72, 37, 16),
                                                  (16, 64, 50, 2), (48, 64, 9, 3), (72, 128, 1, 8)])
def test_gemma_vision_qkv_exact(d, pad, n, h) -> None:
    """Gemma 4's tower: its attention's input in one kernel (`models.gemma4.vision.vision_qkv`) equals the torch ops
    that define it bit for bit: q and k RMS-normed with their weights and rotated by the 2-D RoPE, v normed, padded
    with zeros; at a bucket of 648 patches (16 heads of 72), with rows of zeros and of large magnitude, the tiny GPU
    model's heads of 16, and a qkv whose rows lie in a wider buffer (under 16 rows of 72, where torch's mean sums in
    another order, the torch ops run: tests/vision/test_gemma_vision.py)."""
    from llm_mini_infer.models.gemma4.vision import vision_qkv

    torch.manual_seed(d + n)
    wide = torch.randn(n, 3 * h * d + 40, device=DEV, dtype=BF) * 2
    if n > 5:
        wide[3] = 0  # a padding patch of zeros: rms 0, eps alone
        wide[5] *= 300  # large magnitudes: the sum of squares near its rounding's edges
    qkv = wide[:, 7:7 + 3 * h * d].view(n, 3, h, d)  # a row stride past the row
    qw = (torch.randn(d, device=DEV) * 0.5 + 1).to(BF)
    kw = (torch.randn(d, device=DEV) * 0.5 + 1).to(BF)
    ang = torch.rand(n, d // 2, device=DEV) * 40  # the column's angles, then the row's
    cos, sin = ang.cos(), ang.sin()
    got = vision_qkv(qkv, qw, kw, 1e-6, cos, sin, pad)
    want = ref(vision_qkv, qkv, qw, kw, 1e-6, cos, sin, pad)
    assert not torch.equal(want[0], ref(vision_qkv, qkv, kw, qw, 1e-6, cos, sin, pad)[0])  # the weights matter
    for a, b in zip(got, want):
        assert a.shape == (n, h, pad) and a.is_contiguous()
        assert torch.equal(a, b), (a.float() - b.float()).abs().max().item()


@pytest.mark.parametrize("n_dec", [0, 3])
def test_causal_conv_exact_with_checkpoints(n_dec, pdl) -> None:
    torch.manual_seed(3)
    c, slots = 6144, 9
    lens = [1, 1, 1, 200, 64, 3, 130]  # decode rows first, then prefill rows of every kind
    cu = torch.tensor(np.concatenate([[0], np.cumsum(lens)]), device=DEV)
    t = int(cu[-1])
    xbig = torch.randn(t, c + 100, device=DEV, dtype=BF)
    x = xbig[:, :c]  # strided, as the in_proj split gives
    w = torch.randn(c, 4, device=DEV, dtype=BF)
    st1 = torch.randn(slots, 2, c, 3, device=DEV, dtype=BF)  # [slots, layers, C, 3]: use layer 1's view
    st2 = st1.clone()
    rows = torch.tensor([1, 2, 3, 4, 5, 6, 7], device=DEV)  # live state slots of the rows
    # mid-chunk checkpoints of row 3 (200 tokens) at 64 and 128, into the free slots 8 and 0
    ck = (torch.tensor([0, 0, 0, 0, 2, 2, 2, 2], device=DEV), torch.tensor([64, 128], device=DEV),
          torch.tensor([8, 0], device=DEV))
    o1 = ops.causal_conv(x, w, st1[:, 1], cu, rows, ck, n_dec)
    o2 = ref(ops.causal_conv, x, w, st2[:, 1], cu, rows, ck)
    assert torch.equal(o1, o2)
    assert torch.equal(st1, st2)


@pytest.mark.parametrize("n_dec", [0, 2])
def test_causal_conv_tiles(n_dec, pdl) -> None:
    """Rows around the kernel's 32-token tiles, shorter than the window, and checkpoints at every kind of offset (in
    the old state's reach, at tile boundaries, in a later tile)."""
    torch.manual_seed(4)
    c = 1000  # not a multiple of the block
    lens = [1, 1, 2, 31, 32, 33, 96, 65, 3, 1]
    cu = torch.tensor(np.concatenate([[0], np.cumsum(lens)]), device=DEV)
    x = torch.randn(int(cu[-1]), c, device=DEV, dtype=BF)
    w = torch.randn(c, 4, device=DEV, dtype=BF)
    st1 = torch.randn(24, c, 3, device=DEV, dtype=BF)
    st2 = st1.clone()
    rows = torch.arange(1, len(lens) + 1, device=DEV)  # live slots 1..10; checkpoints go to 11..
    per_row = {2: [1], 4: [0, 1, 2, 31], 5: [31, 32], 6: [33, 64, 70, 95], 7: [64], 8: [2]}
    ptr, off, dst = [0], [], []
    for r in range(len(lens)):
        for o in per_row.get(r, []):
            off.append(o)
            dst.append(11 + len(dst))
        ptr.append(len(off))
    ck = tuple(torch.tensor(a, device=DEV) for a in (ptr, off, dst))
    o1 = ops.causal_conv(x, w, st1, cu, rows, ck, n_dec)
    o2 = ref(ops.causal_conv, x, w, st2, cu, rows, ck)
    assert torch.equal(o1, o2)
    assert torch.equal(st1, st2)


@pytest.mark.parametrize("hk, hv", [(16, 16), (16, 48)])
def test_gdn_prep(hk, hv, pdl) -> None:
    torch.manual_seed(6)
    t, dk = 77, 128
    xbig = torch.randn(t, 2 * hk * dk + hv * dk + 32, device=DEV, dtype=BF) * 2
    x = xbig[:, :-32]
    ba = torch.randn(t, 2 * hv, device=DEV, dtype=BF) * 3
    b, a = ba[:, :hv], ba[:, hv:]
    a_log = torch.randn(hv, device=DEV) * 0.5
    dt_bias = torch.randn(hv, device=DEV, dtype=BF)
    kw = dict(a_log=a_log, dt_bias=dt_bias, hk=hk, hv=hv, dk=dk)
    got, want = ops.gdn_prep(x, a, b, **kw), ref(ops.gdn_prep, x, a, b, **kw)
    ulp_close(got[0], want[0])
    ulp_close(got[1], want[1])
    for g, w in zip(got[2:], want[2:]):  # v, alpha, beta: exact
        assert torch.equal(g, w)
    same_as_serialized(pdl, got, ops.gdn_prep, x, a, b, **kw)


@pytest.mark.parametrize("hk, hv", [(16, 16), (16, 48)])
def test_gdn_decode(hk, hv, pdl) -> None:
    torch.manual_seed(4)
    n, dk, slots = 13, 128, 20
    kd = hk * dk
    mixed = torch.randn(n, 2 * kd + hv * dk, device=DEV, dtype=BF)
    ba = torch.randn(n, 2 * hv, device=DEV, dtype=BF)
    b, a = ba[:, :hv], ba[:, hv:]
    a_log = torch.randn(hv, device=DEV) * 0.5
    dt_bias = torch.randn(hv, device=DEV, dtype=BF)
    zbig = torch.randn(n, hv * dk + 8, device=DEV, dtype=BF)
    z = zbig[:, 8:].view(n, hv, dk)
    norm_w = torch.randn(dk, device=DEV, dtype=BF)
    pool1 = torch.randn(slots, 3, hv, dk, dk, device=DEV) * 0.1
    pool2, pool3 = pool1.clone(), pool1.clone()
    sl = torch.tensor([3, 5, 1, 7, 9, 11, 2, 4, 6, 8, 10, 12, 0], device=DEV)  # the last row: padding (slot 0)
    kw = dict(a_log=a_log, dt_bias=dt_bias, slots=sl, z=z, norm_w=norm_w, eps=1e-6, hk=hk, dk=dk)
    o1 = ops.gdn_decode(mixed, a, b, state=pool1[:, 2], **kw)
    o2 = ref(ops.gdn_decode, mixed, a, b, state=pool2[:, 2], **kw)
    assert torch.equal(o1[-1], torch.zeros_like(o1[-1]))
    assert (o1.float() - o2.float()).abs().max() < 0.05
    assert ((o1.float() - o2.float()).abs().mean() / o2.float().abs().mean()) < 5e-3
    assert torch.allclose(pool1, pool2, atol=1e-4, rtol=1e-3)
    if pdl:  # the serialized launch's bits, the states' included
        assert torch.equal(o1, serialized(ops.gdn_decode, mixed, a, b, state=pool3[:, 2], **kw))
        assert torch.equal(bits(pool1), bits(pool3))


# -- GDN replay for MTP verification --------------------------------------------------------------------------------


def _mtp_case(hk: int, hv: int, n: int, width: int, req_slots: int = 9, slots: int = 20, layers: int = 3):
    torch.manual_seed(7)
    dk = 128
    kd = hk * dk
    t = n * width
    mixed = torch.randn(t, 2 * kd + hv * dk + 16, device=DEV, dtype=BF)[:, 16:]  # strided rows
    ba = torch.randn(t, 2 * hv, device=DEV, dtype=BF)
    b, a = ba[:, :hv], ba[:, hv:]
    z = torch.randn(t, hv * dk + 8, device=DEV, dtype=BF)[:, 8:].view(t, hv, dk)
    pool = torch.randn(slots, layers, hv, dk, dk, device=DEV) * 0.1
    rec_k = torch.randn(req_slots, layers, hv, width, dk, device=DEV, dtype=BF)
    rec_v = torch.randn(req_slots, layers, hv, width, dk, device=DEV, dtype=BF)
    rec_ab = torch.randn(req_slots, layers, hv, width, 2, device=DEV, dtype=BF) * 2
    a_log = torch.randn(layers, hv, device=DEV) * 0.5
    dt_bias = torch.randn(layers, hv, device=DEV, dtype=BF)
    return dict(mixed=mixed, a=a, b=b, z=z, pool=pool, rec_k=rec_k, rec_v=rec_v, rec_ab=rec_ab, a_log=a_log,
                dt_bias=dt_bias, norm_w=torch.randn(dk, device=DEV, dtype=BF))


@pytest.mark.parametrize("hk, hv, width", [(16, 48, 4), (16, 16, 2), (2, 8, 3)])
def test_gdn_mtp_matches_reference(hk, hv, width, pdl) -> None:
    """The fused fold + output-only window against the torch reference: outputs within the decode kernel's
    tolerance, the folded live states within fp32 noise, the window's records exact."""
    c = _mtp_case(hk, hv, n=6, width=width)
    layer = 1
    sl = torch.tensor([3, 5, 1, 0, 9, 11], device=DEV, dtype=torch.int32)  # row 3: padding
    req = torch.tensor([1, 4, 2, 0, 7, 8], device=DEV, dtype=torch.int32)
    fold_n = torch.tensor([0, width, 1, 0, 2 % (width + 1), 0, 0, width - 1, 0], device=DEV, dtype=torch.int32)
    outs = []
    for call in (None, ref, serialized) if pdl else (None, ref):  # the kernel, the reference, the serialized kernel
        pool, rk, rv, rab = (c[x].clone() for x in ("pool", "rec_k", "rec_v", "rec_ab"))
        mtp = ops.GdnMtp(width, req, fold_n, rk[:, layer], rv[:, layer], rab[:, layer])
        kw = dict(a_log=c["a_log"][layer], dt_bias=c["dt_bias"][layer], state=pool[:, layer], slots=sl, z=c["z"],
                  norm_w=c["norm_w"], eps=1e-6, hk=hk, dk=128, mtp=mtp)
        o = ops.gdn_decode_mtp(c["mixed"], c["a"], c["b"], **kw) if call is None else \
            call(ops.gdn_decode_mtp, c["mixed"], c["a"], c["b"], **kw)
        outs.append((o, pool, rk, rv, rab))
    (o1, p1, k1, v1, ab1), (o2, p2, k2, v2, ab2) = outs[:2]
    assert torch.equal(o1[3 * width:4 * width], torch.zeros_like(o1[:width]))  # padding row
    assert (o1.float() - o2.float()).abs().max() < 0.05
    assert ((o1.float() - o2.float()).abs().mean() / o2.float().abs().mean()) < 5e-3
    assert torch.allclose(p1, p2, atol=1e-4, rtol=1e-3)
    real = req[sl > 0].long()
    for g, w in ((k1, k2), (v1, v2), (ab1, ab2)):
        assert torch.equal(g[real], w[real])
    if pdl:  # the serialized launch's bits: the outputs, the folded states and the records
        for g, w in zip(outs[0], outs[2], strict=True):
            assert torch.equal(bits(g), bits(w))


@pytest.mark.parametrize("hk, hv, width", [(16, 48, 4), (2, 8, 3)])
def test_gdn_fold_is_the_window_replayed(hk, hv, width, pdl) -> None:
    """A fold replays exactly what the window computed: verify window A (nothing to fold), then fold its records with
    the fold kernel into a copy, and verify window B with the fused fold of A's records into the live state; the two
    folded states are bit-identical. The fold kernel also matches its reference."""
    c = _mtp_case(hk, hv, n=2, width=width, layers=1)
    pool, rk, rv, rab = c["pool"], c["rec_k"], c["rec_v"], c["rec_ab"]
    sl = torch.tensor([3, 5], device=DEV, dtype=torch.int32)
    req = torch.tensor([1, 2], device=DEV, dtype=torch.int32)
    fold_n = torch.zeros(9, device=DEV, dtype=torch.int32)
    mtp = ops.GdnMtp(width, req, fold_n, rk[:, 0], rv[:, 0], rab[:, 0])
    kw = dict(a_log=c["a_log"][0], dt_bias=c["dt_bias"][0], state=pool[:, 0], slots=sl, z=c["z"], norm_w=c["norm_w"],
              eps=1e-6, hk=hk, dk=128, mtp=mtp)
    ops.gdn_decode_mtp(c["mixed"], c["a"], c["b"], **kw)  # window A: records only
    n = torch.tensor([width, width - 1], device=DEV)
    src, dst = sl.long(), torch.tensor([12, 13], device=DEV)
    before = pool.clone()
    ops.gdn_fold(pool, rk, rv, rab, a_log=c["a_log"], dt_bias=c["dt_bias"], req=req.long(), src=src, dst=dst, n=n)
    want = before.clone()
    ref(ops.gdn_fold, want, rk, rv, rab, a_log=c["a_log"], dt_bias=c["dt_bias"], req=req.long(), src=src, dst=dst,
        n=n)
    assert torch.allclose(pool[dst], want[dst], atol=1e-4, rtol=1e-3)
    fold_n[1], fold_n[2] = width, width - 1
    ops.gdn_decode_mtp(torch.roll(c["mixed"], 1, 0), c["a"], c["b"], **kw)  # window B folds A's commit in place
    assert torch.equal(pool[src], pool[dst])


def test_causal_conv_mtp_and_fold_exact(pdl) -> None:
    torch.manual_seed(5)
    c, width, layers, slots, req_slots = 1000, 4, 2, 16, 9
    lens = [width] * 3 + [40, 7]  # three verify windows, then prefill rows
    cu = torch.tensor(np.concatenate([[0], np.cumsum(lens)]), device=DEV)
    x = torch.randn(int(cu[-1]), c + 30, device=DEV, dtype=BF)[:, 30:]
    w = torch.randn(c, 4, device=DEV, dtype=BF)
    st1 = torch.randn(slots, layers, c, 3, device=DEV, dtype=BF)
    rec1 = torch.randn(req_slots, layers, c, width, device=DEV, dtype=BF)
    st2, rec2 = st1.clone(), rec1.clone()
    rows = torch.tensor([1, 2, 3, 4, 5], device=DEV)
    req = torch.tensor([3, 0, 6], device=DEV)  # the second window: padding-like request slot 0
    fold_n = torch.tensor([0, 0, 0, 4, 0, 0, 2, 0, 0], device=DEV)
    o1 = ops.causal_conv(x, w, st1[:, 1], cu, rows, None, 0, ops.ConvMtp(3, width, req, fold_n, rec1[:, 1]))
    o2 = ref(ops.causal_conv, x, w, st2[:, 1], cu, rows, None, 0, ops.ConvMtp(3, width, req, fold_n, rec2[:, 1]))
    assert torch.equal(o1, o2) and torch.equal(st1, st2) and torch.equal(rec1, rec2)
    r, a, d, n = (torch.tensor(v, device=DEV) for v in ([3, 6, 0], [1, 3, 2], [10, 3, 11], [4, 2, 0]))
    ops.conv_fold(st1, rec1, r, a, d, n)
    ref(ops.conv_fold, st2, rec2, r, a, d, n)
    assert torch.equal(st1, st2)


# -- sampler -------------------------------------------------------------------------------------------------------


def _ref_tokens(logits, temp, top_p, top_k, min_p, seed, qpos, ids=None, mask=None, mask_row=None):
    v = logits.shape[1]
    plan = plan_sampling(temp, top_p, top_k, min_p, v)
    for g in plan.groups:
        g.idx = torch.as_tensor(g.rows, device=DEV)
    keys = row_keys(torch.as_tensor(seed, device=DEV), torch.as_tensor(qpos + 1, device=DEV))
    toks, status = sample(logits, plan, torch.as_tensor(temp, device=DEV), torch.as_tensor(top_p, device=DEV),
                          torch.as_tensor(top_k, device=DEV), torch.as_tensor(min_p, device=DEV), keys,
                          torch.arange(v, device=DEV) if ids is None else ids.long(), mask=mask, mask_row=mask_row)
    return toks.cpu().numpy(), status.cpu().numpy()


def _kernel_tokens(logits, temp, top_p, top_k, min_p, seed, qpos, ids=None, mask=None, mask_row=None, padded=False):
    n = logits.shape[0]
    slots = n + 1  # tables indexed by slot; row r uses slot r + 1
    t = lambda a, dt: torch.as_tensor(np.concatenate([[0], a]).astype(dt), device=DEV)  # noqa: E731
    out = torch.zeros(n, dtype=torch.int64, device=DEV)
    status = torch.zeros(n, dtype=torch.int64, device=DEV)
    pending = torch.full((slots,), -1, dtype=torch.int64, device=DEV)
    ops.sample(logits, row_slot=torch.arange(1, n + 1, device=DEV), qpos=torch.as_tensor(qpos, device=DEV),
               temperature=t(temp, np.float32), top_p=t(top_p, np.float32), top_k=t(top_k, np.int64),
               min_p=t(min_p, np.float32), seed=t(seed, np.int64), out=out, status=status, pending=pending, ids=ids,
               mask=mask, mask_row=mask_row, padded=padded)
    assert torch.equal(pending[1:], out)
    return out.cpu().numpy(), status.cpu().numpy()


@pytest.mark.parametrize("vocab, bf16", [(248_320, True), (248_320, False), (4096, True)])
def test_sampler_matches_reference(vocab, bf16) -> None:
    rng = np.random.default_rng(vocab + bf16)
    torch.manual_seed(5)
    combos = [  # temperature, top_p, top_k, min_p
        (0.0, 1.0, 0, 0.0), (1.0, 1.0, 0, 0.0), (0.7, 0.8, 20, 0.0), (1.0, 0.95, 0, 0.0), (0.6, 0.95, 20, 0.0),
        (1.0, 1.0, 1, 0.0), (1.3, 1.0, 50, 0.05), (1.0, 0.5, 0, 0.1), (0.9, 0.99, 3000, 0.0), (2.0, 0.999, 0, 0.0),
        (1.0, 1.0, 0, 0.2), (0.8, 0.7, 40, 0.02), (1.0, 0.9, 2048, 0.0), (5.0, 0.9999, 0, 0.0)]
    reps = 6
    n = len(combos) * reps
    temp, top_p, top_k, min_p = (np.array([c[i] for c in combos for _ in range(reps)]) for i in range(4))
    temp, top_p, min_p = temp.astype(np.float32), top_p.astype(np.float32), min_p.astype(np.float32)
    top_k = top_k.astype(np.int64)
    seed = rng.integers(0, 1 << 62, n).astype(np.int64)
    qpos = rng.integers(0, 100_000, n).astype(np.int64)
    scale = torch.linspace(1, 8, n, device=DEV)[:, None]  # sharp to flat
    logits = torch.randn(n, vocab, device=DEV) * scale
    if bf16:  # the model's logits are a BF16 GEMM: many exact ties
        logits = logits.to(BF).float()
    a, sa = _kernel_tokens(logits, temp, top_p, top_k, min_p, seed, qpos)
    b, sb = _ref_tokens(logits, temp, top_p, top_k, min_p, seed, qpos)
    assert not sa.any() and not sb.any()
    bad = np.flatnonzero(a != b)
    assert len(bad) == 0, [(i, combos[i // reps], a[i], b[i]) for i in bad[:10]]


def test_sampler_keeps_the_top_token_alone_below_a_top_p_of_one_over_the_vocabulary() -> None:
    # The API takes any top_p above 0, as vLLM does: only the top token's S is at most top_p * Z (1e-46: 0 in
    # fp32), on each top-p path (the top-k set in one batch, across batches, the whole row)
    rng = np.random.default_rng(46)
    torch.manual_seed(46)
    combos = [(1.0, p, k, 0.0) for p in (1e-9, 1e-12, 1e-45, 1e-46) for k in (20, 3000, 0)]
    reps = 4
    n = len(combos) * reps
    temp, top_p, top_k, min_p = (np.array([c[i] for c in combos for _ in range(reps)]) for i in range(4))
    temp, top_p, min_p = temp.astype(np.float32), top_p.astype(np.float32), min_p.astype(np.float32)
    top_k = top_k.astype(np.int64)
    seed = rng.integers(0, 1 << 62, n).astype(np.int64)
    qpos = rng.integers(0, 100_000, n).astype(np.int64)
    logits = torch.randn(n, 248_320, device=DEV) * 4
    a, sa = _kernel_tokens(logits, temp, top_p, top_k, min_p, seed, qpos)
    b, sb = _ref_tokens(logits, temp, top_p, top_k, min_p, seed, qpos)
    assert not sa.any() and not sb.any()
    best = logits.argmax(-1).cpu().numpy()
    bad = np.flatnonzero((a != best) | (b != best))
    assert len(bad) == 0, [(i, combos[i // reps], a[i], b[i], best[i]) for i in bad[:10]]


@pytest.mark.parametrize("size", [32_768, 1_000])
def test_sampler_over_a_reduced_vocabulary(size) -> None:
    """The drafter's head (`dev.speculative.draft_vocab`): logits over `size` token ids (ascending); the kernel's
    tokens are the reference's, which are the full vocabulary's with the other tokens masked
    (tests/runner/test_sampler.py)."""
    rng = np.random.default_rng(size)
    torch.manual_seed(6)
    ids = torch.as_tensor(np.sort(rng.choice(248_320, size, replace=False)), dtype=torch.int32, device=DEV)
    combos = [(0.0, 1.0, 0, 0.0), (1.0, 1.0, 0, 0.0), (0.7, 0.8, 20, 0.0), (1.0, 0.95, 0, 0.0), (0.9, 0.99, 3000, 0.0),
              (1.3, 1.0, 50, 0.05)]
    reps = 6
    n = len(combos) * reps
    temp, top_p, top_k, min_p = (np.array([c[i] for c in combos for _ in range(reps)]) for i in range(4))
    temp, top_p, min_p = temp.astype(np.float32), top_p.astype(np.float32), min_p.astype(np.float32)
    top_k = top_k.astype(np.int64)
    seed = rng.integers(0, 1 << 62, n).astype(np.int64)
    qpos = rng.integers(0, 100_000, n).astype(np.int64)
    logits = (torch.randn(n, size, device=DEV) * torch.linspace(1, 8, n, device=DEV)[:, None]).to(BF).float()
    a, sa = _kernel_tokens(logits, temp, top_p, top_k, min_p, seed, qpos, ids=ids)
    b, sb = _ref_tokens(logits, temp, top_p, top_k, min_p, seed, qpos, ids=ids)
    assert not sa.any() and not sb.any()
    assert np.array_equal(a, b), np.flatnonzero(a != b)[:10]
    assert np.isin(a, ids.cpu().numpy()).all()


@pytest.mark.parametrize("vocab", [248_320, 4096])
def test_sampler_with_grammar_masks_matches_reference(vocab) -> None:
    """Grammar masks: rows sample among the tokens their mask allows (masks with a few tokens, as inside a JSON enum
    or key, and with most of the vocabulary, as before a value), unmasked rows (−1) as without masks; every parameter
    path of the sampler; an empty mask flags MASK_EMPTY. Token for token with the reference."""
    rng = np.random.default_rng(vocab)
    torch.manual_seed(7)
    combos = [(0.0, 1.0, 0, 0.0), (1.0, 1.0, 0, 0.0), (0.7, 0.8, 20, 0.0), (1.0, 0.95, 0, 0.0), (0.6, 0.95, 20, 0.0),
              (1.3, 1.0, 50, 0.05), (1.0, 0.5, 0, 0.1), (0.9, 0.99, 3000, 0.0), (1.0, 1.0, 1, 0.0)]
    reps = 8
    n = len(combos) * reps
    temp, top_p, top_k, min_p = (np.array([c[i] for c in combos for _ in range(reps)]) for i in range(4))
    temp, top_p, min_p = temp.astype(np.float32), top_p.astype(np.float32), min_p.astype(np.float32)
    top_k = top_k.astype(np.int64)
    seed = rng.integers(0, 1 << 62, n).astype(np.int64)
    qpos = rng.integers(0, 100_000, n).astype(np.int64)
    logits = (torch.randn(n, vocab, device=DEV) * torch.linspace(1, 8, n, device=DEV)[:, None]).to(BF).float()
    m = 6
    allowed = np.zeros((m, vocab), bool)
    allowed[0, rng.choice(vocab, 3, replace=False)] = True  # a few tokens
    allowed[1, rng.choice(vocab, 40, replace=False)] = True
    allowed[2] = rng.random(vocab) < 0.6  # most of the vocabulary
    allowed[3] = True
    allowed[4, [vocab - 1]] = True  # one token, the last
    words = np.packbits(np.pad(allowed, ((0, 0), (0, -vocab % 32))), axis=1, bitorder="little").view(np.int32)
    mask = torch.as_tensor(words.copy(), device=DEV)
    rows = np.where(np.arange(n) % 7 == 6, -1, np.arange(n) % 5).astype(np.int64)  # some rows unmasked
    rows[3] = 5  # mask 5 allows nothing
    mask_row = torch.as_tensor(rows, device=DEV)
    a, sa = _kernel_tokens(logits, temp, top_p, top_k, min_p, seed, qpos, mask=mask, mask_row=mask_row)
    b, sb = _ref_tokens(logits, temp, top_p, top_k, min_p, seed, qpos, mask=mask, mask_row=mask_row)
    assert sa.tolist() == sb.tolist() and sa[3] == 2 and sa.sum() == 2
    bad = np.flatnonzero(a != b)
    assert len(bad) == 0, [(i, combos[i // reps], rows[i], a[i], b[i]) for i in bad[:10]]
    ok = rows >= 0
    ok[3] = False
    assert allowed[rows[ok], a[ok]].all()


TERMS_COMBOS = [(0.0, 1.0, 0, 0.0), (1.0, 1.0, 0, 0.0), (0.7, 0.8, 20, 0.0), (1.0, 0.95, 0, 0.0),
                (1.3, 1.0, 50, 0.05), (1.0, 0.5, 0, 0.1), (0.9, 0.99, 3000, 0.0), (1.0, 1.0, 0, 0.2)]


def _terms_run(logits, rng, k: int, ids=None, mask=None, mask_row=None, padded=False, reps: int = 8, vocab: int = 0):
    """`TERMS_COMBOS` rows with penalties, a bias (1,024 entries on some), both or neither
    (`sampler_cases.terms_tables`), each at a verify window position (k > 0): the kernel's tokens, statuses and counts
    after its commit, and the reference's (`apply_terms`, then `sample`; a penalised row's token counted). `vocab`: the
    tables' (default: the logits' width)."""
    from llm_mini_infer.runner.cuda.penalties import TERM_PEN
    from llm_mini_infer.runner.cuda.sampler import apply_terms

    n = len(TERMS_COMBOS) * reps
    vocab = vocab or logits.shape[1]
    tables, drafts, w = terms_tables(rng, n, vocab, k, DEV)
    temp, top_p, top_k, min_p = (np.array([c[i] for c in TERMS_COMBOS for _ in range(reps)]) for i in range(4))
    temp, top_p, min_p, top_k = temp.astype(np.float32), top_p.astype(np.float32), min_p.astype(np.float32), \
        top_k.astype(np.int64)
    seed = rng.integers(0, 1 << 62, n).astype(np.int64)
    qpos = rng.integers(0, 100_000, n).astype(np.int64)
    slots = torch.arange(1, n + 1, device=DEV)
    window = None
    if k:  # the static block's tokens: row r's sample token is window token w[r] (-1 - w), or an explicit one (w 0)
        w_tok = torch.as_tensor(np.where(w > 0, -1 - w, 5), device=DEV)
        window = (w_tok, torch.arange(n, device=DEV))
    t = lambda a, dt: torch.as_tensor(np.concatenate([[0], a]).astype(dt), device=DEV)  # noqa: E731
    out = torch.zeros(n, dtype=torch.int64, device=DEV)
    status = torch.zeros(n, dtype=torch.int64, device=DEV)
    pending = torch.full((n + 1,), -1, dtype=torch.int64, device=DEV)
    before = tables.cnt.clone()
    ops.sample(logits, row_slot=slots, qpos=torch.as_tensor(qpos, device=DEV), temperature=t(temp, np.float32),
               top_p=t(top_p, np.float32), top_k=t(top_k, np.int64), min_p=t(min_p, np.float32),
               seed=t(seed, np.int64), out=out, status=status, pending=pending, ids=ids, mask=mask, mask_row=mask_row,
               padded=padded, terms=tables, drafts=drafts if k else None, window=window)
    got_cnt = tables.cnt.clone()
    cols = torch.arange(logits.shape[1], device=DEV) if ids is None else ids.long()
    win = None
    if k:
        wt = torch.as_tensor(w, device=DEV)
        d = drafts.index_select(0, slots)
        win = torch.where(torch.arange(k, device=DEV)[None, :] < wt[:, None], d, -1)
    f = lambda a: a.index_select(0, slots)  # noqa: E731
    flags = f(tables.terms)
    adj = apply_terms(logits, cols, flags, f(tables.rep), f(tables.pres), f(tables.freq), before.index_select(0, slots),
                      f(tables.bias_id), f(tables.bias_val), f(tables.bias_n), win)
    plan = plan_sampling(temp, top_p, top_k, min_p, logits.shape[1])
    for g in plan.groups:
        g.idx = torch.as_tensor(g.rows, device=DEV)
    keys = row_keys(torch.as_tensor(seed, device=DEV), torch.as_tensor(qpos + 1, device=DEV))
    want, wst = sample(logits, plan, torch.as_tensor(temp, device=DEV), torch.as_tensor(top_p, device=DEV),
                       torch.as_tensor(top_k, device=DEV), torch.as_tensor(min_p, device=DEV), keys,
                       cols, mask=mask, mask_row=mask_row, adjusted=adj, clamp_rows=flags != 0)
    want_cnt = before.clone()
    on = ((flags & TERM_PEN) != 0) & (wst == 0)
    want_cnt.view(-1).index_add_(0, slots * tables.cnt.shape[1] + want, on.to(torch.int32))
    params = (temp, top_p, top_k, min_p, seed, qpos)
    return (out.cpu().numpy(), status.cpu().numpy(), got_cnt), (want.cpu().numpy(), wst.cpu().numpy(), want_cnt), \
        flags, params


@pytest.mark.parametrize("vocab", [248_320, 262_144])
@pytest.mark.parametrize("k", [0, 5])
def test_sampler_with_terms_matches_reference(vocab, k) -> None:
    """Penalties and the logit bias (runner/cuda/penalties.py) at each real vocabulary, on every sampler path, greedy
    included, with a verify window's drafts (k 5): token for token with the reference, the statuses alike, and the
    commit's counts (a penalised row's token, once). Rows without terms are the kernel's without them."""
    rng = np.random.default_rng(vocab + k)
    torch.manual_seed(11)
    n = len(TERMS_COMBOS) * 8
    logits = (torch.randn(n, vocab, device=DEV) * torch.linspace(1, 8, n, device=DEV)[:, None]).to(BF).float()
    (a, sa, ca), (b, sb, cb), flags, params = _terms_run(logits, rng, k)
    assert sa.tolist() == sb.tolist() and not sa.any()
    bad = np.flatnonzero(a != b)
    assert len(bad) == 0, [(i, int(flags[i]), a[i], b[i]) for i in bad[:10]]
    assert torch.equal(ca, cb)
    plain = (flags == 0).cpu().numpy()
    c, _ = _kernel_tokens(logits, *params)
    assert np.array_equal(a[plain], c[plain])


@pytest.mark.parametrize("vocab", [248_320, 262_144])
def test_sampler_with_terms_under_masks_and_the_sparse_head(vocab) -> None:
    """The same under grammar masks (a biased token the mask refuses is never sampled) and over the grammar-sparse
    head's columns (the union's ids, padded): token for token with the reference."""
    rng = np.random.default_rng(vocab + 1)
    torch.manual_seed(12)
    n = len(TERMS_COMBOS) * 8
    allowed = np.zeros((3, vocab), bool)
    allowed[0, rng.choice(vocab, 40, replace=False)] = True
    allowed[1] = rng.random(vocab) < 0.5
    allowed[2, rng.choice(vocab, 3, replace=False)] = True
    words = np.packbits(np.pad(allowed, ((0, 0), (0, -vocab % 32))), axis=1, bitorder="little").view(np.int32)
    mask = torch.as_tensor(words.copy(), device=DEV)
    mask_row = torch.as_tensor(np.where(np.arange(n) % 4 == 3, -1, np.arange(n) % 3), device=DEV)
    logits = (torch.randn(n, vocab, device=DEV) * 4).to(BF).float()
    (a, sa, ca), (b, sb, cb), _, _ = _terms_run(logits, rng, 0, mask=mask, mask_row=mask_row)
    assert sa.tolist() == sb.tolist() and np.array_equal(a, b) and torch.equal(ca, cb)
    union = np.flatnonzero(allowed[:2].any(0))  # every row masked by mask 0 or 1: the sparse head's columns
    pad = 64
    ids = torch.as_tensor(np.concatenate([union, np.full(pad, -1)]).astype(np.int32), device=DEV)
    rows = torch.as_tensor(np.arange(n) % 2, device=DEV)
    sparse = (torch.randn(n, len(union) + pad, device=DEV) * 4).to(BF).float()
    (a, sa, ca), (b, sb, cb), _, _ = _terms_run(sparse, np.random.default_rng(3), 0, ids=ids, mask=mask,
                                                mask_row=rows, padded=True, vocab=vocab)
    assert sa.tolist() == sb.tolist() and np.array_equal(a, b) and torch.equal(ca, cb)
    assert np.isin(a, union).all()


# temperature, top_p, top_k, min_p: top-k and top-p cuts inside tie groups of thousands, one batch (k <= 2,048) or
# several (k > 2,048, and top-p alone: the Z pass), greedy rows and min-p's
TIES_COMBOS = [(0.0, 1.0, 0, 0.0), (1.0, 1.0, 50, 0.0), (1.0, 1.0, 1500, 0.0), (1.0, 1.0, 2048, 0.0),
               (1.0, 1.0, 3000, 0.0), (1.0, 0.5, 0, 0.0), (1.0, 0.9, 2000, 0.0), (0.7, 0.95, 100, 0.0),
               (1.0, 0.99, 0, 0.0), (1.0, 1.0, 0, 0.5), (0.5, 0.8, 5000, 0.0)]


@pytest.mark.parametrize("vocab", [248_320, 262_144])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("shape", ["even", "few_top"])
def test_sampler_with_many_ties_at_the_cut_matches_reference(vocab, masked, shape) -> None:
    """Ternary logits ({−1, 0, +1}) over the whole vocabulary put tens of thousands of tokens
    on one value, so top-k and top-p cut inside a tie group of more than the 1,024 the kernel ranks in shared memory
    (`kEqCap`: sampler.cu's ordered pass over the row takes the lowest ids); `few_top`: a few hundred at +1, the cut
    then inside the 0 group. `masked`: rows under masks of most of the vocabulary (their ties still over 1,024), of a
    few hundred tokens, or none. Token for token with the reference."""
    rng = np.random.default_rng(vocab + 2 * masked + (shape == "few_top"))
    reps = 6
    n = len(TIES_COMBOS) * reps
    temp, top_p, top_k, min_p = (np.array([c[i] for c in TIES_COMBOS for _ in range(reps)]) for i in range(4))
    temp, top_p, min_p = temp.astype(np.float32), top_p.astype(np.float32), min_p.astype(np.float32)
    top_k = top_k.astype(np.int64)
    seed = rng.integers(0, 1 << 62, n).astype(np.int64)
    qpos = rng.integers(0, 100_000, n).astype(np.int64)
    vals = rng.integers(-1, 2, (n, vocab)).astype(np.float32)
    if shape == "few_top":  # a few hundred tokens at +1, the rest at 0 or −1
        vals[vals == 1] = 0
        for r in range(n):
            vals[r, rng.choice(vocab, int(rng.integers(100, 900)), replace=False)] = 1
    logits = torch.as_tensor(vals, device=DEV)
    mask = mask_row = None
    if masked:
        allowed = np.stack([rng.random(vocab) < 0.6, rng.random(vocab) < 0.9, rng.random(vocab) < 0.002])
        words = np.packbits(np.pad(allowed, ((0, 0), (0, -vocab % 32))), axis=1, bitorder="little").view(np.int32)
        mask = torch.as_tensor(words.copy(), device=DEV)
        mask_row = torch.as_tensor(np.arange(n) % 4 - 1, dtype=torch.int64, device=DEV)  # −1: unmasked
    a, sa = _kernel_tokens(logits, temp, top_p, top_k, min_p, seed, qpos, mask=mask, mask_row=mask_row)
    b, sb = _ref_tokens(logits, temp, top_p, top_k, min_p, seed, qpos, mask=mask, mask_row=mask_row)
    assert not sa.any() and not sb.any()
    bad = np.flatnonzero(a != b)
    assert len(bad) == 0, [(i, TIES_COMBOS[i // reps], a[i], b[i]) for i in bad[:10]]
    # the top-k cut lies in a group of more than 1,024 equal logits (the ordered pass) in most rows that have one
    x = vals.copy()
    if masked:
        mr = np.arange(n) % 4 - 1
        x[mr >= 0] = np.where(allowed[mr[mr >= 0]], vals[mr >= 0], -np.inf)
    rows = np.flatnonzero((top_k > 0) & (np.isfinite(x).sum(1) > top_k))
    kth = -np.sort(-x[rows], axis=1)[np.arange(len(rows)), top_k[rows] - 1]
    group = (x[rows] == kth[:, None]).sum(1)
    assert (group > 1024).mean() > 0.5, group


def test_sampler_flags_nonfinite_rows() -> None:
    logits = torch.randn(4, 1000, device=DEV)
    logits[1, 5] = float("nan")
    logits[2, 9] = float("-inf")
    p = np.ones(4, np.float32)
    toks, status = _kernel_tokens(logits, np.array([1, 1, 0, 0], np.float32), p, np.zeros(4, np.int64),
                                  np.zeros(4, np.float32), np.arange(4), np.arange(4))
    assert status.tolist() == [0, 1, 1, 0] and toks[1] == 0 and toks[2] == 0


def test_sampler_draws_the_references_largest_noise() -> None:
    """The clamp's value in the kernel: tests/support/sampler_cases.py's `largest_noise_rows` sample [3980, 398] only
    where the top uniform's noise is within 0.01 of 16.64, the reference's (the next uniform's, 15.94, or +inf would
    sample one token in both rows). Token for token with the reference."""
    rows = largest_noise_rows().to(DEV)
    ones = np.ones(2, np.float32)
    args = (ones, ones, np.zeros(2, np.int64), np.zeros(2, np.float32), np.full(2, 5738, np.int64),
            np.zeros(2, np.int64))  # position qpos + 1: 1
    a, sa = _kernel_tokens(rows, *args)
    b, sb = _ref_tokens(rows, *args)
    assert not sa.any() and not sb.any()
    assert a.tolist() == b.tolist() == [3980, 398], (a, b)


def test_sampler_never_samples_a_token_far_below_the_top() -> None:
    """The plain pass over a 262,144-token vocabulary: one token at logit 0 and the rest at -40, temperature 1, over
    1,000 positions. The noise is finite (the uniform clamped below 1), so no off-peak token is sampled, though 14 of
    them draw the top uniform, whose noise would be +inf unclamped (tests/runner/test_sampler.py, on the same rows).
    Token for token with the reference."""
    v, n, peak = 262_144, 1000, 12_345
    row = torch.full((1, v), -40.0, device=DEV)
    row[0, peak] = 0.0
    logits = row.expand(n, -1)  # row stride 0: one row in memory
    ones = np.ones(n, np.float32)
    args = (ones, ones, np.zeros(n, np.int64), np.zeros(n, np.float32), np.zeros(n, np.int64), np.arange(n))
    a, sa = _kernel_tokens(logits, *args)  # positions qpos + 1: 1 to 1,000
    b, sb = _ref_tokens(logits, *args)
    assert not sa.any() and not sb.any()
    assert (a == peak).all(), np.flatnonzero(a != peak)[:10]
    assert np.array_equal(a, b)


@pytest.mark.parametrize("case", FIT_CASES)
def test_sampler_frequencies_fit_the_target_distribution(case) -> None:
    """tests/support/sampler_cases.py's goodness of fit on the kernel, with the same draws: one row at many positions,
    every parameter path, the excluded tokens never drawn and the others as often as the target distribution says. Token
    for token with the reference."""
    temp, top_p, top_k, min_p, masked = case
    logits, allowed = fit_row()
    v, n = len(logits), FIT_DRAWS
    args = (np.full(n, temp, np.float32), np.full(n, top_p, np.float32), np.full(n, top_k, np.int64),
            np.full(n, min_p, np.float32), np.full(n, FIT_SEED, np.int64), np.arange(n))  # positions qpos + 1
    mask = _words(allowed[None]) if masked else None
    mask_row = torch.zeros(n, dtype=torch.int64, device=DEV) if masked else None
    rows = torch.as_tensor(logits, device=DEV)[None].expand(n, -1)
    a, sa = _kernel_tokens(rows, *args, mask=mask, mask_row=mask_row)
    b, sb = _ref_tokens(rows, *args, mask=mask, mask_row=mask_row)
    assert not sa.any() and not sb.any()
    assert np.array_equal(a, b), np.flatnonzero(a != b)[:10]
    assert_fits(a, target_distribution(logits, temp, top_p, top_k, min_p, allowed if masked else np.ones(v, bool)))


# The grammar-sparse head samples the union of the step's masks, the assistant's head the grammar's universe, each as
# ascending ids padded with -1 to a fixed width. A top-k above kCap (2,048) takes several batches, or with top-p the Z
# pass; the masks here leave fewer allowed tokens than it, where the Z pass's select runs out.
SAMPLING_COMBOS = [  # temperature, top_p, top_k, min_p
    (0.0, 1.0, 0, 0.0), (1.0, 1.0, 0, 0.0), (0.6, 1.0, 0, 0.0), (0.7, 0.8, 20, 0.0), (1.0, 0.95, 0, 0.0),
    (0.6, 0.95, 20, 0.0), (1.3, 1.0, 50, 0.05), (1.0, 0.5, 0, 0.1), (0.9, 0.99, 3000, 0.0), (1.0, 0.6, 3000, 0.0),
    (1.0, 1.0, 2500, 0.0), (1.0, 1.0, 1, 0.0)]


def _sampling_rows(rng, reps):
    """SAMPLING_COMBOS's parameters, each `reps` rows in a row, with random seeds and positions."""
    n = len(SAMPLING_COMBOS) * reps
    temp, top_p, top_k, min_p = (np.array([c[i] for c in SAMPLING_COMBOS for _ in range(reps)]) for i in range(4))
    return (temp.astype(np.float32), top_p.astype(np.float32), top_k.astype(np.int64), min_p.astype(np.float32),
            rng.integers(0, 1 << 62, n).astype(np.int64), rng.integers(0, 100_000, n).astype(np.int64))


def _padded_ids(ids: np.ndarray, cap: int) -> torch.Tensor:
    out = torch.full((cap,), -1, dtype=torch.int32, device=DEV)
    out[:len(ids)] = torch.as_tensor(ids, dtype=torch.int32, device=DEV)
    return out


def _words(allowed: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.packbits(allowed, axis=1, bitorder="little").view(np.int32).copy(), device=DEV)


@pytest.mark.parametrize("cap", [1024, 4096])
def test_sampler_over_padded_ids_matches_reference(cap) -> None:
    """Padded ids: a padding column (finite, as the head gathers a real row; lifted here above every real logit) is
    never a candidate. Without masks, then with masks over the whole vocabulary tested through the ids (a few of the
    ids; some with other tokens; most of the vocabulary; tokens outside the ids only, and nothing: MASK_EMPTY),
    unmasked rows among them; NaN rows; only padding: MASK_EMPTY. Token for token with the reference."""
    rng = np.random.default_rng(cap)
    torch.manual_seed(8)
    vocab, u = 262_144, cap * 3 // 4
    ids_np = np.sort(rng.choice(vocab, u, replace=False))
    ids = _padded_ids(ids_np, cap)
    args = _sampling_rows(rng, reps=8)
    n = len(args[0])
    logits = (torch.randn(n, cap, device=DEV) * torch.linspace(1, 8, n, device=DEV)[:, None]).to(BF).float()
    logits[:, u:] += 100.0
    logits[5] = float("nan")
    allowed = np.zeros((5, vocab), bool)
    allowed[0, rng.choice(ids_np, 3, replace=False)] = True  # a few of the ids
    allowed[1, rng.choice(ids_np, 40, replace=False)] = True
    allowed[1, rng.choice(vocab, 200, replace=False)] = True  # and tokens that are no column
    allowed[2] = rng.random(vocab) < 0.6  # most of the vocabulary
    allowed[3] = True
    allowed[3, ids_np] = False  # tokens outside the ids only: nothing allowed among the columns
    rows = np.where(np.arange(n) % 7 == 6, -1, np.arange(n) % 5).astype(np.int64)  # some rows unmasked; mask 4: none
    for mask, mask_row in ((None, None), (_words(allowed), torch.as_tensor(rows, device=DEV))):
        a, sa = _kernel_tokens(logits, *args, ids=ids, mask=mask, mask_row=mask_row, padded=True)
        b, sb = _ref_tokens(logits, *args, ids=ids, mask=mask, mask_row=mask_row)
        assert sa.tolist() == sb.tolist()
        bad = np.flatnonzero(a != b)
        assert len(bad) == 0, [(i, SAMPLING_COMBOS[i // 8], a[i], b[i]) for i in bad[:10]]
        ok = sa == 0
        assert np.isin(a[ok], ids_np).all()
        if mask is None:
            assert np.flatnonzero(sa).tolist() == [5] and sa[5] == 1
        else:
            assert (sa[(rows == 3) | (rows == 4)] == 2).all() and sa[5] == 1
            ok &= rows >= 0
            assert allowed[rows[ok], a[ok]].all()
    a, sa = _kernel_tokens(logits, *args, ids=torch.full_like(ids, -1), padded=True)  # only padding
    assert not a.any() and sa.tolist() == [1 if i == 5 else 2 for i in range(n)]


@pytest.mark.parametrize("cap", [1024, 4096])
def test_sampler_on_the_unions_columns_matches_the_dense_logits(cap) -> None:
    """The sparse head's exactness on the kernel: its columns (the union of the step's masks, ascending, padded with
    -1 to `cap`) sampled under the same masks give the dense logits' tokens and statuses, on every parameter path
    (greedy and plain sum nothing; the top-p sums differ only in order, as against the reference), with BF16 ties at
    the top, masks shared by rows, empty masks and a NaN row."""
    rng = np.random.default_rng(cap + 1)
    torch.manual_seed(9)
    vocab, n_masks = 262_144, 12
    args = _sampling_rows(rng, reps=8)
    n = len(args[0])
    allowed = np.zeros((n_masks, vocab), bool)
    for i in range(n_masks):
        allowed[i, rng.choice(vocab, int(rng.integers(1, 65)), replace=False)] = True
    allowed[:2] = False  # two masks allow nothing
    allowed[2, 0] = True  # token 0, whose head row a padding id gathers
    mask_row = rng.integers(0, n_masks, n)
    mask_row[1], mask_row[2::12] = 0, 2
    union = np.flatnonzero(allowed.any(0))
    dense = (torch.randn(n, vocab, device=DEV) * torch.linspace(1, 8, n, device=DEV)[:, None]).to(BF).float()
    for r in range(0, n, 3):  # two allowed tokens tie at the top
        toks = np.flatnonzero(allowed[mask_row[r]])
        if len(toks) >= 2:
            tie = torch.as_tensor(rng.choice(toks, 2, replace=False), device=DEV)
            dense[r, tie] = dense[r, torch.as_tensor(toks, device=DEV)].max() + 1.0
    dense[n - 1] = float("nan")
    ids = _padded_ids(union, cap)
    cols = dense.index_select(1, ids.clamp(min=0).long())
    cols[:, len(union):] += 100.0  # padding: row 0's logit, lifted above every real one
    mask, mrow = _words(allowed), torch.as_tensor(mask_row, device=DEV)
    a, sa = _kernel_tokens(dense, *args, mask=mask, mask_row=mrow)
    b, sb = _kernel_tokens(cols, *args, ids=ids, mask=mask, mask_row=mrow, padded=True)
    assert sa.tolist() == sb.tolist() and sa[1] == 2 and sa[-1] == 1
    bad = np.flatnonzero(a != b)
    assert len(bad) == 0, [(i, SAMPLING_COMBOS[i // 8], a[i], b[i]) for i in bad[:10]]
    c, sc = _ref_tokens(cols, *args, ids=ids, mask=mask, mask_row=mrow)
    assert np.array_equal(b, c) and np.array_equal(sb, sc), np.flatnonzero(b != c)[:10]
    ok = sa == 0
    assert allowed[mask_row[ok], a[ok]].all()


# -- the small-M BF16 linear (in_proj_ba on decode and verify steps) ---------------------------------------------------


@pytest.mark.parametrize("n, k", [(96, 5120), (100, 1024)])
def test_linear_small_matches_reference_and_is_batch_invariant(n, k) -> None:
    """`ops.linear_small` against the fp32 product rounded once, at every M the kernel takes; each row's output is the
    same whatever M is up to LINEAR_SMALL_MAX_M (the kernel computes every row the same way); cuBLAS above."""
    torch.manual_seed(11)
    w = torch.randn(n, k, device=DEV, dtype=BF) * 0.05
    x = torch.randn(256, k, device=DEV, dtype=BF)
    want = (x.float() @ w.float().T).to(BF)
    assert torch.equal(ops.linear_small(x, w), torch.nn.functional.linear(x, w))  # 256 rows: cuBLAS
    small = ops.linear_small(x[:ops.LINEAR_SMALL_MAX_M], w)
    ulp_close(small, want[:ops.LINEAR_SMALL_MAX_M], max_ulps=1, min_equal=0.98)
    for m in range(1, ops.LINEAR_SMALL_MAX_M + 1):
        assert torch.equal(ops.linear_small(x[:m], w), small[:m]), m
    # rows of a strided view (the GDN layer's BF16 norm output is contiguous; a view must work too)
    xs = torch.randn(4, k + 64, device=DEV, dtype=BF)[:, 64:]
    assert torch.equal(ops.linear_small(xs, w), ops.linear_small(xs.contiguous(), w))
    # beyond LINEAR_SMALL_MAX_M: cuBLAS
    big = torch.randn(ops.LINEAR_SMALL_MAX_M + 1, k, device=DEV, dtype=BF)
    assert torch.equal(ops.linear_small(big, w), torch.nn.functional.linear(big, w))


# -- FP8 activation quantization and the block-FP8 GEMM ------------------------------------------------------------


def q_equal(a, b) -> None:
    """Two QuantActs carry the same bytes and scales (over their m tokens)."""
    m = a.q.shape[0]
    assert torch.equal(a.q.view(torch.uint8), b.q.view(torch.uint8))
    assert torch.equal(a.s[:, :m], b.s[:, :m])


def test_quant_fp8_exact() -> None:
    torch.manual_seed(7)
    big = (torch.randn(37, 5120 + 64, device=DEV) * torch.logspace(-3, 2, 37, device=DEV)[:, None]).to(BF)
    x = big[:, 64:]  # a strided view
    x[3, :128] = 0  # an all-zero group
    q_equal(ops.quant_fp8(x), fp8.quant_ref(x))


@pytest.mark.parametrize("d", [1024, 5120])
def test_rmsnorm_quant_exact(d, pdl) -> None:
    """The fused norm + quantization equals the BF16 kernel followed by the quantization, bit for bit."""
    torch.manual_seed(8)
    x = torch.randn(29, d, device=DEV, dtype=BF) * 3
    res = torch.randn(29, d, device=DEV, dtype=BF)
    w = torch.randn(d, device=DEV, dtype=BF) * 0.2
    a, r = ops.rmsnorm_quant(x, w, 1e-6, res, bf16=True)
    y, r2 = ops.rmsnorm(x, w, 1e-6, res)
    assert torch.equal(r, r2) and torch.equal(a.bf16, y)
    q_equal(a, fp8.quant_ref(y))
    same_as_serialized(pdl, (a, r), ops.rmsnorm_quant, x, w, 1e-6, res, bf16=True)
    a2, none = ops.rmsnorm_quant(x, w, 1e-6)
    assert none is None and a2.bf16 is None
    q_equal(a2, fp8.quant_ref(ops.rmsnorm(x, w, 1e-6)))
    same_as_serialized(pdl, (a2, none), ops.rmsnorm_quant, x, w, 1e-6)


def test_gated_rmsnorm_sigmoid_gate_silu_mul_quant_exact(pdl) -> None:
    torch.manual_seed(9)
    t, h = 33, 48
    x = torch.randn(t, h, 128, device=DEV, dtype=BF)
    zbig = torch.randn(t, h * 128 + 64, device=DEV, dtype=BF)
    z = zbig[:, 64:].view(t, h, 128)
    w = torch.randn(128, device=DEV, dtype=BF)
    out = fp8.empty_act(t + 5, h * 128, torch.device(DEV))
    view = fp8.QuantAct(out.q[5:], out.s[:, 5:])  # the GDN prefill rows' slice of a step's buffer
    ops.gated_rmsnorm_quant(x, z, w, 1e-6, view)
    q_equal(view, fp8.quant_ref(ops.gated_rmsnorm(x, z, w, 1e-6).reshape(t, h * 128)))
    if pdl:  # the serialized launch's bytes and scales
        q_equal(view, serialized(ops.gated_rmsnorm_quant, x, z, w, 1e-6, fp8.empty_act(t, h * 128, torch.device(DEV))))
    o = torch.randn(t, 24 * 256, device=DEV, dtype=BF)
    qg = torch.randn(t, 24, 512, device=DEV, dtype=BF)
    gate = qg[..., 256:]
    a = ops.sigmoid_gate_quant(o, gate)
    q_equal(a, fp8.quant_ref(ops.sigmoid_gate(o, gate)))
    same_as_serialized(pdl, a, ops.sigmoid_gate_quant, o, gate)
    gu = torch.randn(t, 2 * 17408, device=DEV, dtype=BF) * 2
    g, u = gu.float().chunk(2, dim=-1)
    q_equal(ops.silu_mul_quant(gu), fp8.quant_ref(torch.nn.functional.silu(g) * u, divide=True))


def test_quant_non_finite_exact(pdl) -> None:
    """NaN and ±Inf, bit for bit with the reference (tests/runner/test_fp8.py pins it: a NaN leaves the absmax and
    quantizes to −448, an infinite value makes the scale Inf): a NaN among finite values, an all-NaN group, +Inf and
    −Inf, and a NaN with an Inf in one group, through the standalone quantization; then through the fused producers,
    whose op spreads them first (a norm makes a NaN input's row NaN; a NaN or infinite norm weight, a column)."""
    torch.manual_seed(14)
    nan, inf = float("nan"), float("inf")
    t, k = 7, 1024
    x = torch.randn(t, k, device=DEV, dtype=BF)
    x[0, 5] = nan
    x[1, 128:256] = nan
    x[2, 300], x[2, 301] = inf, -inf
    x[3, 10], x[3, 11] = nan, inf
    q_equal(ops.quant_fp8(x), fp8.quant_ref(x))
    w = torch.randn(k, device=DEV, dtype=BF) * 0.2
    w[5], w[300] = nan, inf
    res = torch.randn(t, k, device=DEV, dtype=BF)
    q_equal(ops.rmsnorm_quant(x, w, 1e-6, res)[0], fp8.quant_ref(ops.rmsnorm(x, w, 1e-6, res)[0]))
    q_equal(ops.rmsnorm_quant(x, w, 1e-6)[0], fp8.quant_ref(ops.rmsnorm(x, w, 1e-6)))
    xh, z = x.view(t, k // 128, 128), torch.randn(t, k // 128, 128, device=DEV, dtype=BF)
    z[4, 2, 7] = nan
    out = fp8.empty_act(t, k, torch.device(DEV))
    q_equal(ops.gated_rmsnorm_quant(xh, z, w[:128], 1e-6, out),
            fp8.quant_ref(ops.gated_rmsnorm(xh, z, w[:128], 1e-6).reshape(t, k)))
    gate = torch.randn(t, k // 256, 256, device=DEV, dtype=BF)
    gate[5, 1, 9] = nan
    q_equal(ops.sigmoid_gate_quant(x, gate), fp8.quant_ref(ops.sigmoid_gate(x, gate)))
    gu = torch.cat([x, torch.randn(t, k, device=DEV, dtype=BF)], dim=1)
    gu[6, k + 600] = inf
    g, u = gu.float().chunk(2, dim=-1)
    q_equal(ops.silu_mul_quant(gu), fp8.quant_ref(torch.nn.functional.silu(g) * u, divide=True))


@pytest.mark.parametrize("non_finite", [False, True])
def test_gdn_decode_quantized_output(non_finite, pdl) -> None:
    """The decode kernel's FP8 epilogue equals its BF16 output quantized per value head; padding rows quantize
    zeros. With `non_finite`, the epilogue's own copy of the quantizer takes NaN and Inf as the reference does
    (tests/runner/test_fp8.py pins it): a NaN gate value (a NaN among finite values), an infinite one (the scale Inf),
    and a value head whose v is NaN (its state and output NaN: an all-NaN group)."""
    torch.manual_seed(10)
    n, hk, hv, dk = 9, 16, 48, 128
    mixed = torch.randn(n, 2 * hk * dk + hv * dk, device=DEV, dtype=BF)
    ba = torch.randn(n, 2 * hv, device=DEV, dtype=BF)
    b, a = ba[:, :hv], ba[:, hv:]
    z = torch.randn(n, hv, dk, device=DEV, dtype=BF)
    if non_finite:
        z[1, 5, 7], z[3, 2, 4] = float("nan"), float("inf")
        mixed[2, (2 * hk + 9) * dk:(2 * hk + 10) * dk] = float("nan")
    pool = torch.randn(12, hv, dk, dk, device=DEV) * 0.1
    kw = dict(a_log=torch.randn(hv, device=DEV) * 0.5, dt_bias=torch.randn(hv, device=DEV, dtype=BF), z=z,
              norm_w=torch.randn(dk, device=DEV, dtype=BF), eps=1e-6, hk=hk, dk=dk,
              slots=torch.tensor([3, 5, 1, 7, 9, 11, 2, 4, 0], device=DEV))
    p1, p2 = pool.clone(), pool.clone()
    o = ops.gdn_decode(mixed, a, b, state=p1, **kw)
    buf = fp8.empty_act(n + 2, hv * dk, torch.device(DEV))
    qa = fp8.QuantAct(buf.q[:n], buf.s[:, :n])
    assert ops.gdn_decode(mixed, a, b, state=p2, quant=qa, **kw) is None
    assert torch.equal(p1.view(torch.int32), p2.view(torch.int32))
    q_equal(qa, fp8.quant_ref(o.reshape(n, hv * dk)))
    assert (qa.q[-1].view(torch.uint8) == 0).all()


@pytest.mark.parametrize("non_finite", [False, True])
def test_gdn_mtp_quantized_output(non_finite, pdl) -> None:
    """The MTP kernel's FP8 epilogue, likewise: its BF16 outputs quantized per value head, a padding row's zeros at the
    floor's scale; with `non_finite`, a NaN and an infinite gate value, and a NaN v on a window's first token, which
    makes its value head's outputs NaN through the window (all-NaN groups)."""
    hk, hv, dk, width, layer = 16, 48, 128, 3, 1
    c = _mtp_case(hk, hv, n=4, width=width)
    if non_finite:
        c["z"][1, 5, 7], c["z"][10, 2, 4] = float("nan"), float("inf")
        c["mixed"][width, (2 * hk + 9) * dk:(2 * hk + 10) * dk] = float("nan")  # row 1's first token
    sl = torch.tensor([3, 5, 0, 9], device=DEV, dtype=torch.int32)  # row 2: padding
    req = torch.tensor([1, 4, 0, 7], device=DEV, dtype=torch.int32)
    fold_n = torch.tensor([0, 0, 0, 0, width, 0, 0, 1, 0], device=DEV, dtype=torch.int32)
    qa = fp8.empty_act(4 * width, hv * dk, torch.device(DEV))
    outs = []
    for quant in (None, qa):
        pool, rk, rv, rab = (c[x].clone() for x in ("pool", "rec_k", "rec_v", "rec_ab"))
        mtp = ops.GdnMtp(width, req, fold_n, rk[:, layer], rv[:, layer], rab[:, layer])
        o = ops.gdn_decode_mtp(c["mixed"], c["a"], c["b"], a_log=c["a_log"][layer], dt_bias=c["dt_bias"][layer],
                               state=pool[:, layer], slots=sl, z=c["z"], norm_w=c["norm_w"], eps=1e-6, hk=hk, dk=dk,
                               mtp=mtp, quant=quant)
        outs.append((o, pool))
    (o, p1), (none, p2) = outs
    assert none is None and torch.equal(p1.view(torch.int32), p2.view(torch.int32))
    q_equal(qa, fp8.quant_ref(o.reshape(4 * width, hv * dk)))
    assert (qa.q[2 * width:3 * width].view(torch.uint8) == 0).all()


GEMM_SHAPES = [(256, 384), (5120, 6144), (14336, 5120), (5376, 21504), (18432, 5376)]  # Gemma 4's down, full qkv


@pytest.mark.cutlass(reason="needs the CUTLASS library")
@pytest.mark.parametrize("config", [0, 1, 2, 3])
@pytest.mark.parametrize("m", [1, 3, 17, 64, 65, 200, 300])
def test_fp8_gemm_cutlass_matches_reference(config, m) -> None:
    torch.manual_seed(11 + m)
    for n, k in GEMM_SHAPES:
        w, ws = fp8.quantize_weight(torch.randn(n, k, device=DEV) * 0.05)
        a = ops.quant_fp8(torch.randn(m, k, device=DEV, dtype=BF))
        out = fp8.gemm_cutlass(a, w, ws, config)
        want = fp8.gemm_ref(a, w, ws, torch.float32)
        err = (out.float() - want).abs()
        assert (err <= want.abs() * 2 ** -7 + 1e-3).all(), (n, k, float(err.max()))  # within a BF16 rounding


@pytest.mark.cutlass(reason="needs the CUTLASS library")
@pytest.mark.parametrize("m", [1, 7, 64, 100, 128, 257, 1000])
def test_fp8_gemm_configs_give_the_same_bits(m) -> None:
    """Every CUTLASS configuration sums each output in the same order (they give the same bits at every M), so the
    dispatch table chooses speed only."""
    torch.manual_seed(13 + m)
    for n, k in GEMM_SHAPES:
        w, ws = fp8.quantize_weight(torch.randn(n, k, device=DEV) * 0.05)
        a = ops.quant_fp8(torch.randn(m, k, device=DEV, dtype=BF) * 2)
        outs = [fp8.gemm_cutlass(a, w, ws, c) for c in (0, 1, 2, 3)]
        assert all(torch.equal(outs[0], o) for o in outs[1:]), (n, k)


def test_dequant_fp8_and_linear_bf16_exact() -> None:
    """PRECISE rows' weight (the precise prefix): w x its block's scale in fp32, rounded once to BF16, as the reference;
    linear_bf16 is then F.linear on it."""
    torch.manual_seed(14)
    for n, k in [(256, 384), (14336, 5120)]:
        w, ws = fp8.quantize_weight(torch.randn(n, k, device=DEV) * 0.05)
        ws = ws * torch.logspace(-2, 1, ws.numel(), device=DEV).view_as(ws)  # scales of every magnitude
        got = ops.dequant_fp8(w, ws)
        assert torch.equal(got, fp8.dequant_weight(w, ws).to(BF))
        x = torch.randn(33, k, device=DEV, dtype=BF)
        assert torch.equal(fp8.linear_bf16(x, w, ws), torch.nn.functional.linear(x, got))


@pytest.mark.cutlass(reason="needs the CUTLASS library")
@pytest.mark.parametrize("t, first", [(70, 40), (2048, 1500), (130, 1), (9, 0)])
def test_split_fp8_linear_gives_each_paths_bits(t, first) -> None:
    """Per-row precision: in a step of t tokens whose tokens [first, t) are PRECISE, an FP8 projection's first rows are
    the FP8 path's bits for them alone (any M: every configuration gives the same bits), its last rows the BF16 path's
    for them alone; SiLU-and-mul's split (`fp8.SplitAct`) keeps the fused kernel's quantization for the FP8 rows."""
    from llm_mini_infer.models.fp8 import Fp8Linear

    torch.manual_seed(15)
    for n, k in [(512, 1024), (14336, 5120)]:
        lin = Fp8Linear(k, n).to(DEV)
        lin.weight.data, lin.weight_scale.data = fp8.quantize_weight(torch.randn(n, k, device=DEV) * 0.05)
        x = torch.randn(t, k, device=DEV, dtype=BF)
        with fp8.precise_rows(t, first):
            y = lin(x)
        if first:
            assert torch.equal(y[:first], fp8.gemm(ops.quant_fp8(x[:first]), lin.weight, lin.weight_scale))
        assert torch.equal(y[first:], fp8.linear_bf16(x[first:], lin.weight, lin.weight_scale))
    if first:
        gu = torch.randn(t, 2 * 1024, device=DEV, dtype=BF) * 2
        lin = Fp8Linear(1024, 512).to(DEV)
        lin.weight.data, lin.weight_scale.data = fp8.quantize_weight(torch.randn(512, 1024, device=DEV) * 0.05)
        y = lin(fp8.SplitAct(ops.silu_mul_quant(gu[:first]), ops.silu_mul(gu[first:])))
        assert torch.equal(y[:first], fp8.gemm(ops.silu_mul_quant(gu[:first]), lin.weight, lin.weight_scale))
        assert torch.equal(y[first:], fp8.linear_bf16(ops.silu_mul(gu[first:]), lin.weight, lin.weight_scale))


@pytest.mark.cutlass(reason="needs the CUTLASS library")
def test_fp8_linear_in_a_cuda_graph() -> None:
    """Quantization and the GEMM (every dispatch path) replay from a CUDA graph with new inputs."""
    torch.manual_seed(12)
    n, k = 512, 1024
    w, ws = fp8.quantize_weight(torch.randn(n, k, device=DEV) * 0.05)
    for m in (2, 100, 400):
        x = torch.randn(m, k, device=DEV, dtype=BF)
        eager = fp8.gemm(ops.quant_fp8(x), w, ws)
        g = torch.cuda.CUDAGraph()
        with no_gc(), torch.cuda.graph(g):
            y = fp8.gemm(ops.quant_fp8(x), w, ws)
        x.copy_(torch.randn(m, k, device=DEV, dtype=BF))
        g.replay()
        assert torch.equal(y, fp8.gemm(ops.quant_fp8(x), w, ws)) and not torch.equal(y, eager)
