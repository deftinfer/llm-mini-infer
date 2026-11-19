"""Fused ops: our CUDA kernels (`csrc/`, built by `kernels.build`) on CUDA tensors, the torch reference elsewhere (the
sampler is CUDA only: its reference is `runner.cuda.sampler`).

Each reference is the model's unfused torch computation, with its rounding points (models/qwen3_5/model.py): fp32
intermediates, one rounding to the model dtype. The kernels round at the same points and only reduction orders differ,
but for the GDN decode, verify and fold kernels, which keep the vendored decode kernel's arithmetic with fast exp
(tests/kernels/test_kernels.py bounds each difference on a GPU). `enabled(False)` forces the references, also on CUDA,
the sampler excepted (for A/B runs; the references sync where they need host values, so CUDA graphs require the
kernels).
"""

from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from . import build

if TYPE_CHECKING:
    from .fp8 import QuantAct

_enabled = True
_pdl = False  # the kernel library's PDL flag, mirrored for the launchers outside it (`set_pdl`: the CUTLASS GEMMs)


def enabled(flag: bool | None = None) -> bool:
    """Read or set whether CUDA tensors go to the kernels."""
    global _enabled
    if flag is not None:
        _enabled = flag
    return _enabled


def lib() -> Any:
    return build.load() if _enabled else None


def use_kernels(t: torch.Tensor) -> bool:
    return t.is_cuda and _enabled and build.load() is not None


def _p(t: torch.Tensor | None) -> ctypes.c_void_p:
    return ctypes.c_void_p(t.data_ptr() if t is not None else None)


def _stream(t: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(torch.cuda.current_stream(t.device).cuda_stream)


def _check(rc: int, name: str) -> None:
    if rc != 0:
        raise RuntimeError(f"{name} failed: {'unsupported shape' if rc == -1 else f'CUDA error {rc}'}")


# -- launches ------------------------------------------------------------------------------------------------------


def set_pdl(on: bool) -> bool:
    """Programmatic dependent launch for the kernels that support it (csrc/common.cuh; docs/architecture.md, "CUDA
    graphs"): one flag for the library, always off without the library. CUDA graphs keep the launches they captured,
    so a runner sets it before capturing. Returns whether it is now on."""
    global _pdl
    lib = build.load()
    if lib is None:
        _pdl = False
        return False
    lib.lmk_set_pdl(1 if on else 0)
    _pdl = bool(lib.lmk_get_pdl())
    return _pdl


def pdl() -> bool:
    return _pdl


# -- norms and activations -----------------------------------------------------------------------------------------


def _rms(xf: torch.Tensor, eps: float) -> torch.Tensor:
    return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float,
            residual: torch.Tensor | None = None) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Gemma RMSNorm `x / rms(x) * (1 + w)` over the last dim, rounded once. With `residual`: normalises the fp32
    sum x + residual and also returns that sum rounded (the new residual)."""
    if not use_kernels(x) or x.dim() != 2 or x.stride(-1) != 1:
        if residual is None:
            return (_rms(x.float(), eps) * (1.0 + w.float())).to(x.dtype)
        s = x.float() + residual.float()
        return (_rms(s, eps) * (1.0 + w.float())).to(x.dtype), s.to(x.dtype)
    n, d = x.shape
    out = torch.empty((n, d), dtype=x.dtype, device=x.device)
    res_out = None
    if residual is not None:
        residual = residual.contiguous()
        res_out = torch.empty_like(out)
    _check(build.load().lmk_gemma_rmsnorm(_p(x), x.stride(0), _p(residual), _p(w), _p(out), _p(res_out), n, d, eps,
                                          _stream(x)), "gemma_rmsnorm")
    return out if residual is None else (out, res_out)


def gated_rmsnorm(x: torch.Tensor, z: torch.Tensor, w: torch.Tensor, eps: float,
                  out: torch.Tensor | None = None) -> torch.Tensor:
    """The GDN output norm per head: `w * x / rms(x) * silu(z)`, x and z [T, H, D], fp32, rounded once. Writes into
    `out` (contiguous, like x) when given."""
    if not use_kernels(x) or not x.is_contiguous() or z.stride(2) != 1 or z.stride(1) != z.shape[2] or \
            (out is not None and not out.is_contiguous()):
        y = (_rms(x.float(), eps) * w.float() * F.silu(z.float())).to(x.dtype)
        if out is None:
            return y
        out.copy_(y)
        return out
    t, h, d = x.shape
    out = torch.empty_like(x) if out is None else out
    _check(build.load().lmk_gated_rmsnorm(_p(x), _p(z), z.stride(0), _p(w), _p(out), t, h, d, eps, _stream(x)),
           "gated_rmsnorm")
    return out


def silu_mul(gu: torch.Tensor) -> torch.Tensor:
    """gu [T, 2I] (gate, up) -> silu(gate) * up [T, I], fp32, rounded once."""
    if not use_kernels(gu) or gu.stride(-1) != 1:
        g, u = gu.chunk(2, dim=-1)
        return (F.silu(g.float()) * u.float()).to(gu.dtype)
    t, i2 = gu.shape
    out = torch.empty((t, i2 // 2), dtype=gu.dtype, device=gu.device)
    _check(build.load().lmk_silu_mul(_p(gu), gu.stride(0), _p(out), t, i2 // 2, _stream(gu)), "silu_mul")
    return out


def sigmoid_gate(o: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """o [T, H*D] times sigmoid(gate [T, H, D]), fp32, rounded once (the attention output gate)."""
    t, h, d = gate.shape
    if not use_kernels(o) or not o.is_contiguous() or gate.stride(2) != 1:
        return (o.float() * torch.sigmoid(gate.reshape(t, h * d).float())).to(o.dtype)
    out = torch.empty_like(o)
    _check(build.load().lmk_sigmoid_gate(_p(o), _p(gate), gate.stride(0), gate.stride(1), _p(out), t, h, d,
                                         _stream(o)), "sigmoid_gate")
    return out


LINEAR_SMALL_MAX_M = 8  # the kernel's own ceiling: the most rows `linear_small` takes


def linear_small(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """x [M, K] @ w [N, K]^T in BF16 with fp32 accumulation, rounded once (`F.linear`'s result up to the summation
    order). Our kernel for a few rows (`csrc/linear.cu`: cuBLAS runs in_proj_ba at 2-4 rows, an MTP verify window at
    batch 1, much slower than at 1 or 8; every output row is computed the same way at any M up to 8); cuBLAS for more
    rows or other layouts."""
    m, k = x.shape
    if not use_kernels(x) or m > LINEAR_SMALL_MAX_M or x.dtype != torch.bfloat16 or w.dtype != torch.bfloat16 or \
            x.stride(-1) != 1 or w.stride(-1) != 1 or k % 8 or x.stride(0) % 8 or w.stride(0) % 8 or \
            x.data_ptr() % 16 or w.data_ptr() % 16:
        return F.linear(x, w)
    n = w.shape[0]
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    _check(build.load().lmk_linear_small(_p(x), x.stride(0), _p(w), w.stride(0), _p(out), n, m, n, k, _stream(x)),
           "linear_small")
    return out


def linear_rows_slices(k: int) -> int:
    """The slices `linear_rows` cuts K into (csrc/linear.cu): 8, 4, 2 or 1, the most keeping whole 64-element stages."""
    s = 8
    while s > 1 and k % (64 * s):
        s //= 2
    return s


def linear_rows(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """x [M, K] @ w [N, K]^T in BF16 with fp32 accumulation, rounded once, every row computed the same way at any M
    (`csrc/linear.cu`'s `linear_rows`; docs/architecture.md, "Batch invariance"). The reference: the fp32 product,
    rounded once."""
    m, k = x.shape
    if not use_kernels(x):
        return F.linear(x.float(), w.float()).to(x.dtype)
    if x.dtype != torch.bfloat16 or w.dtype != torch.bfloat16 or x.stride(-1) != 1 or w.stride(-1) != 1 or k % 64 or \
            x.stride(0) % 8 or w.stride(0) % 8 or x.data_ptr() % 16 or w.data_ptr() % 16:
        raise ValueError(f"linear_rows: BF16 rows of a multiple of 64 elements, 16-byte aligned (x {tuple(x.shape)} "
                         f"{x.dtype}, w {tuple(w.shape)} {w.dtype})")
    n, sl = w.shape[0], linear_rows_slices(k)
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    part = torch.empty((sl, m, n), dtype=torch.float32, device=x.device)
    _check(build.load().lmk_linear_rows(_p(x), x.stride(0), _p(w), w.stride(0), _p(out), n, _p(part), m, n, k, sl,
                                        _stream(x)), "linear_rows")
    return out


# -- attention input: QK norm, RoPE, KV write ---------------------------------------------------------------------


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rotary_dim: int) -> torch.Tensor:
    """NeoX-style rotation of the first `rotary_dim` dims of each head: x [T, H, D], cos/sin [T, rotary_dim / 2]
    fp32; computed in fp32, rounded once."""
    half = rotary_dim // 2
    x1, x2 = x[..., :half].float(), x[..., half:rotary_dim].float()
    c, s = cos[:, None, :], sin[:, None, :]
    out = x.clone()
    out[..., :half] = x1 * c - x2 * s
    out[..., half:rotary_dim] = x2 * c + x1 * s
    return out


def mrope_components(half: int, sections: tuple[int, int]) -> torch.Tensor:
    """Which position component each of the `half` rotary frequencies reads under interleaved M-RoPE (Qwen3.5/3.8,
    HF's `recomposition_frequencies`): 1 (h) at j % 3 == 1 below 3 x sections[0], 2 (w) at j % 3 == 2 below
    3 x sections[1], else 0 (t)."""
    j = torch.arange(half)
    comp = torch.zeros(half, dtype=torch.long)
    comp[(j % 3 == 1) & (j < 3 * sections[0])] = 1
    comp[(j % 3 == 2) & (j < 3 * sections[1])] = 2
    return comp


def rope_cos_sin(positions: torch.Tensor, cos_cache: torch.Tensor, sin_cache: torch.Tensor,
                 mrope: tuple[int, int] | None) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token cos and sin [T, rotary_dim / 2] in fp32: positions [T], or [3, T] with `mrope` (each frequency reads
    its component's position)."""
    if positions.dim() == 1:
        return cos_cache[positions].float(), sin_cache[positions].float()
    half = cos_cache.shape[1]
    comp = mrope_components(half, mrope).to(positions.device)
    p = positions[comp].t()  # [T, half]: the position each frequency reads
    j = torch.arange(half, device=positions.device)
    return cos_cache[p, j].float(), sin_cache[p, j].float()


def qk_norm_rope(qkv: torch.Tensor, *, heads: int, kv_heads: int, head_dim: int, gated: bool, q_w: torch.Tensor,
                 k_w: torch.Tensor, eps: float, positions: torch.Tensor, cos_cache: torch.Tensor,
                 sin_cache: torch.Tensor, rotary_dim: int, kv_layer: torch.Tensor, kv_page: torch.Tensor,
                 kv_off: torch.Tensor, mrope: tuple[int, int] | None = None) -> torch.Tensor:
    """The attention layer's input side: per-head Gemma RMSNorm of q and k (rounded), partial NeoX RoPE (fp32 against
    the cos/sin cache in the model dtype, rounded), and K and V written to their KV pages (`kv_layer`
    [pages, 2, page_size, Hkv, D]; `kv_page`/`kv_off` per token). Returns q [T, H, D]. `positions` are the RoPE
    positions: [T], or [3, T] (t, h, w) with `mrope` = (h, w) sections for interleaved M-RoPE (images)."""
    t = qkv.shape[0]
    h, hkv, d = heads, kv_heads, head_dim
    q_size = h * d * (2 if gated else 1)
    if positions.dim() == 2 and mrope is None:
        raise ValueError("3-D positions need the M-RoPE sections")
    if not use_kernels(qkv) or qkv.stride(-1) != 1:
        q, k, v = qkv.split([q_size, hkv * d, hkv * d], dim=-1)
        q = q.view(t, h, 2 * d)[..., :d] if gated else q.view(t, h, d)
        cos, sin = rope_cos_sin(positions, cos_cache, sin_cache, mrope)
        q = apply_rope(rmsnorm_heads(q, q_w, eps), cos, sin, rotary_dim)
        k = apply_rope(rmsnorm_heads(k.view(t, hkv, d), k_w, eps), cos, sin, rotary_dim)
        kv_layer[:, 0][kv_page, kv_off] = k
        kv_layer[:, 1][kv_page, kv_off] = v.reshape(t, hkv, d)
        return q.contiguous()
    q_out = torch.empty((t, h, d), dtype=qkv.dtype, device=qkv.device)
    pstride = positions.stride(0) if positions.dim() == 2 else 0
    sec_h, sec_w = mrope if positions.dim() == 2 else (0, 0)
    _check(build.load().lmk_qk_norm_rope(
        _p(qkv), qkv.stride(0), 2 * d if gated else d, q_size, q_size + hkv * d, _p(q_w), _p(k_w), _p(positions),
        pstride, sec_h, sec_w, _p(cos_cache), _p(sin_cache), _p(q_out), _p(kv_layer), kv_layer.stride(0),
        kv_layer.shape[2], _p(kv_page), _p(kv_off), t, h, hkv, d, rotary_dim, eps, _stream(qkv)), "qk_norm_rope")
    return q_out


def vision_qkv_rope(qkv: torch.Tensor, *, heads: int, head_dim: int, cos: torch.Tensor, sin: torch.Tensor,
                    pad_to: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The Qwen3.8 vision tower's attention input: qkv [n, 3 x heads x head_dim] -> q, k with the 2-D rotary
    (fp32 cos/sin [n, head_dim], rotate_half over the whole head, rounded once) and v, each [n, heads, pad_to] with
    zeros past head_dim (FlashInfer's head dims)."""
    n, h, d = qkv.shape[0], heads, head_dim
    if not use_kernels(qkv) or qkv.stride(-1) != 1:
        x = qkv.view(n, 3, h, d)
        c, s = cos[:, None, :], sin[:, None, :]

        def rot(y: torch.Tensor) -> torch.Tensor:
            yf = y.float()
            r = torch.cat([-yf[..., d // 2:], yf[..., :d // 2]], dim=-1)
            return (yf * c + r * s).to(qkv.dtype)

        out = [rot(x[:, 0]), rot(x[:, 1]), x[:, 2]]
        if pad_to > d:
            out = [F.pad(y, (0, pad_to - d)) for y in out]
        return tuple(y.contiguous() for y in out)
    q = torch.empty((n, h, pad_to), dtype=qkv.dtype, device=qkv.device)
    k, v = torch.empty_like(q), torch.empty_like(q)
    _check(build.load().lmk_vision_qkv_rope(_p(qkv), qkv.stride(0), _p(cos), _p(sin), _p(q), _p(k), _p(v), n, h, d,
                                            pad_to, _stream(qkv)), "vision_qkv_rope")
    return q, k, v


def rmsnorm_heads(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    return (_rms(x.float(), eps) * (1.0 + w.float())).to(x.dtype)


# -- GDN -----------------------------------------------------------------------------------------------------------


class ConvMtp:
    """MTP rows of the causal conv (GDN replay; docs/architecture.md, "Speculative decoding"): rows [0, n) are verify
    windows of L tokens each. The live conv state of each first absorbs the first `fold_n[req]` inputs of the slot's
    previous window (its committed tokens, from `rec`), is written back, and then convolves the window without absorbing
    it; the window's inputs are stored in `rec` for the next fold. `rec` is a layer's view [req slots, C, L]; `req` [n]
    int64 the rows' request slots; `fold_n` [req slots] int64."""

    __slots__ = ("fold_n", "n", "rec", "req", "width")

    def __init__(self, n: int, width: int, req: torch.Tensor, fold_n: torch.Tensor, rec: torch.Tensor):
        self.n, self.width, self.req, self.fold_n, self.rec = n, width, req, fold_n, rec


def causal_conv(x: torch.Tensor, w: torch.Tensor, state: torch.Tensor, cu: torch.Tensor, slots: torch.Tensor,
                ckpt: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None, n_dec: int = 0,
                mtp: ConvMtp | None = None) -> torch.Tensor:
    """Depthwise causal conv (width 4) + SiLU over each row's (conv state ++ tokens): x [T, C] (any row stride), w
    [C, 4], state [slots, C, 3] (a layer's view of the conv pool, oldest first), cu [R + 1] int64 token offsets of the
    rows, slots [R] int64 their live state slots. Each product is rounded to the model dtype and summed in fp32; the
    output is rounded once. Stores each row's last 3 inputs as its new state, and with `ckpt` = (ptr [R + 1], off, dst)
    the state after `off` tokens of a row into slot `dst` (mid-chunk checkpoints).
    `n_dec`: rows [0, n_dec) have one token each (lets the kernel skip looking them up; 0 if unknown). `mtp`: rows
    [0, mtp.n) are MTP verify windows (`ConvMtp`)."""
    if use_kernels(x) and x.stride(-1) == 1 and state.stride(2) == 1 and state.stride(1) == 3 and \
            (mtp is None or (mtp.rec.stride(2) == 1 and mtp.rec.stride(1) == mtp.width)):
        t, c = x.shape
        out = torch.empty((t, c), dtype=x.dtype, device=x.device)
        ck_ptr, ck_off, ck_dst = ckpt if ckpt is not None else (None, None, None)
        _check(build.load().lmk_causal_conv(_p(x), x.stride(0), _p(w), _p(state), state.stride(0), _p(cu), _p(slots),
                                            _p(ck_ptr), _p(ck_off), _p(ck_dst), _p(out), len(slots), c, t, n_dec,
                                            mtp.n if mtp else 0, mtp.width if mtp else 0,
                                            _p(mtp.req if mtp else None), _p(mtp.fold_n if mtp else None),
                                            _p(mtp.rec if mtp else None), mtp.rec.stride(0) if mtp else 0,
                                            _stream(x)),
               "causal_conv")
        return out
    # Reference: one row at a time (host loop; on CUDA it syncs to read the row bounds).
    out = torch.empty(x.shape, dtype=torch.float32, device=x.device)
    cu_h, sl_h = cu.tolist(), slots.tolist()
    n_mtp = mtp.n if mtp is not None else 0
    if n_mtp:
        req_h = mtp.req.tolist()
        for r in range(n_mtp):
            s, q = sl_h[r], req_h[r]
            nf = int(mtp.fold_n[q])
            if nf:  # the previous window's committed inputs, absorbed into the live state
                ext = torch.cat([state[s].transpose(0, 1), mtp.rec[q, :, :nf].transpose(0, 1)])
                state[s] = ext[nf:nf + 3].transpose(0, 1)
    cks: dict[int, list[tuple[int, int]]] = {}
    if ckpt is not None:
        ptr, off, dst = (a.tolist() for a in ckpt)
        for r in range(len(sl_h)):
            cks[r] = list(zip(off[ptr[r]:ptr[r + 1]], dst[ptr[r]:ptr[r + 1]]))
    w0, w1, w2, w3 = (w[:, j] for j in range(4))
    for r, s in enumerate(sl_h):
        t0, t1 = cu_h[r], cu_h[r + 1]
        ext = torch.cat([state[s].transpose(0, 1), x[t0:t1]])  # [3 + n, C]
        n = t1 - t0
        for o, d in cks.get(r, ()):
            state[d] = ext[o:o + 3].transpose(0, 1)
        acc = (ext[0:n] * w0).float()
        acc += ext[1:n + 1] * w1
        acc += ext[2:n + 2] * w2
        acc += ext[3:n + 3] * w3
        out[t0:t1] = acc
        if r < n_mtp:  # a verify window: kept for the next fold, not absorbed
            mtp.rec[int(mtp.req[r]), :, :n] = x[t0:t1].transpose(0, 1)
        else:
            state[s] = ext[n:n + 3].transpose(0, 1)
    return F.silu(out).to(x.dtype)


def conv_fold(state: torch.Tensor, rec: torch.Tensor, req: torch.Tensor, src: torch.Tensor, dst: torch.Tensor,
              n: torch.Tensor) -> None:
    """FOLD and MTP decode checkpoints for the conv state of every GDN layer: state [slots, layers, C, 3], rec [req
    slots, layers, C, L]; per row, the state of slot `src` absorbs the first `n` inputs of request slot `req`'s window
    and goes to slot `dst` (src itself: in place). All int64 [rows]."""
    if use_kernels(state) and state.is_contiguous() and rec.is_contiguous():
        rows, (_, layers, c, _w1) = len(req), state.shape
        _check(build.load().lmk_conv_fold(_p(state), _p(rec), _p(req), _p(src), _p(dst), _p(n), rows, layers, c,
                                          rec.shape[3], _stream(state)), "conv_fold")
        return
    for q, a, d, k in zip(req.tolist(), src.tolist(), dst.tolist(), n.tolist()):
        ext = torch.cat([state[a], rec[q, :, :, :k]], dim=-1)  # [layers, C, 3 + k]
        state[d] = ext[..., k:k + 3]


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """x / sqrt(sum(x^2) + eps) over the last dim (FLA's convention)."""
    return x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) + eps)


def recurrent_step(h: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, decay: torch.Tensor,
                   beta: torch.Tensor) -> torch.Tensor:
    """One token of the gated delta rule, fp32, updating `h` in place: `h ← decay h`, `h ← h + beta (v - h k) kᵀ`,
    `o = h q`. h [N, HV, V, K]; q (already scaled) and k [N, HV, K]; v [N, HV, V]; decay = exp(g), beta [N, HV]."""
    h.mul_(decay[..., None, None])
    kv = torch.matmul(h, k.unsqueeze(-1)).squeeze(-1)
    delta = (v - kv) * beta.unsqueeze(-1)
    h.addcmul_(delta.unsqueeze(-1), k.unsqueeze(-2))
    return torch.matmul(h, q.unsqueeze(-1)).squeeze(-1)


def gdn_gates(a: torch.Tensor, b: torch.Tensor, a_log: torch.Tensor,
              dt_bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The log decay g = -exp(A_log) softplus(a + dt_bias) and beta = sigmoid(b), fp32 [T, HV]."""
    g = -torch.exp(a_log.float()) * F.softplus(a.float() + dt_bias.float())
    return g, torch.sigmoid(b.float())


def gdn_prep(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor, *, a_log: torch.Tensor, dt_bias: torch.Tensor,
             hk: int, hv: int, dk: int) -> tuple[torch.Tensor, ...]:
    """The GDN prefill kernel's inputs from the conv output x [T, 2 HK K + HV V]: q and k L2-normalised and rounded
    to the model dtype [T, HK, K], v contiguous [T, HV, V], and the decay alpha = exp(g) and beta [T, HV] fp32
    (`gdn_gates`)."""
    t = x.shape[0]
    kd = hk * dk
    if use_kernels(x) and dk == 128 and x.stride(-1) == 1 and a.stride(-1) == 1 and b.stride(-1) == 1 and \
            dt_bias.dtype == torch.bfloat16:
        qn = torch.empty((t, hk, dk), dtype=x.dtype, device=x.device)
        kn = torch.empty_like(qn)
        v = torch.empty((t, hv, dk), dtype=x.dtype, device=x.device)
        alpha = torch.empty((t, hv), dtype=torch.float32, device=x.device)
        beta = torch.empty_like(alpha)
        _check(build.load().lmk_gdn_prep(_p(x), x.stride(0), _p(a), a.stride(0), _p(b), b.stride(0), _p(a_log),
                                         _p(dt_bias), _p(qn), _p(kn), _p(v), _p(alpha), _p(beta), t, hk, hv,
                                         _stream(x)), "gdn_prep")
        return qn, kn, v, alpha, beta
    q = x[:, :kd].reshape(t, hk, dk)
    k = x[:, kd:2 * kd].reshape(t, hk, dk)
    v = x[:, 2 * kd:].reshape(t, hv, -1).contiguous()
    g, beta = gdn_gates(a, b, a_log, dt_bias)
    return l2norm(q.float()).to(x.dtype), l2norm(k.float()).to(x.dtype), v, torch.exp(g), beta


def gdn_decode(mixed_qkv: torch.Tensor, a: torch.Tensor, b: torch.Tensor, *, a_log: torch.Tensor,
               dt_bias: torch.Tensor, state: torch.Tensor, slots: torch.Tensor, z: torch.Tensor,
               norm_w: torch.Tensor, eps: float, hk: int, dk: int, out: torch.Tensor | None = None,
               slots_i32: torch.Tensor | None = None, cu_i32: torch.Tensor | None = None,
               accepted_i32: torch.Tensor | None = None, quant: QuantAct | None = None) -> torch.Tensor | None:
    """The GDN decode step of one-token rows, fused with the output norm (the vendored kernel, `csrc/gdn_decode.cu`):
    mixed_qkv [N, 2 HK K + HV V] after the conv, a/b [N, HV], z [N, HV, V]; state [slots, HV, V, K] (a layer's view)
    updated in place at `slots` [N] int64 (a row with slot 0 writes zeros and leaves the state alone: padding).
    Returns the normed output [N, HV, V] (in `out` when given, unless the kernel runs and `out` is not contiguous).
    The kernel takes int32 slots, cu_seqlens and accepted counts, which a CUDA graph keeps in static buffers
    (`slots_i32` [N, 1], `cu_i32` [N + 1] = 0..N, `accepted_i32` = 1).

    With `quant` (views q [N, HV * V] and s [HV, >= N] of the block-FP8 out_proj's input), the normed output is
    also quantized, one group per value head (V = 128), and the BF16 output is skipped unless `out` is given; returns
    `out` (None then)."""
    n = mixed_qkv.shape[0]
    hv, v_dim = z.shape[1], z.shape[2]
    if use_kernels(mixed_qkv) and dk == 128 and v_dim == 128 and mixed_qkv.stride(-1) == 1 and \
            state.stride(-1) == 1 and state.stride(-2) == dk and state.stride(-3) == dk * v_dim and \
            (quant is None or (quant.q.is_contiguous() and quant.s.stride(1) == 1)):
        if slots_i32 is None:
            slots_i32 = slots.to(torch.int32).view(n, 1)
            cu_i32 = torch.arange(n + 1, dtype=torch.int32, device=slots.device)
            accepted_i32 = torch.ones(n, dtype=torch.int32, device=slots.device)
        if out is not None and not out.is_contiguous():
            out = None  # the caller copies a fresh output into its view
        if out is None and quant is None:
            out = torch.empty((n, hv, v_dim), dtype=mixed_qkv.dtype, device=mixed_qkv.device)
        _check(build.load().lmk_gdn_decode(
            _p(mixed_qkv), mixed_qkv.stride(0), _p(a), a.stride(0), _p(b), b.stride(0), _p(a_log), _p(dt_bias),
            {torch.float32: 0, torch.bfloat16: 1, torch.float16: 2}[dt_bias.dtype], _p(slots_i32), slots_i32.shape[1],
            _p(cu_i32), _p(accepted_i32), _p(state), state.stride(0), int(state.dtype == torch.bfloat16), _p(z),
            z.stride(0), _p(norm_w), int(norm_w.dtype == torch.bfloat16), _p(out),
            _p(quant.q if quant is not None else None), _p(quant.s if quant is not None else None),
            quant.s.stride(0) if quant is not None else 0, n, hk, hv, dk ** -0.5, eps, 0, _stream(mixed_qkv)),
            "gdn_decode")
        return out
    kd = hk * dk
    q = mixed_qkv[:, :kd].reshape(n, hk, dk)
    k = mixed_qkv[:, kd:2 * kd].reshape(n, hk, dk)
    v = mixed_qkv[:, 2 * kd:].reshape(n, hv, v_dim)
    g, beta = gdn_gates(a, b, a_log, dt_bias)
    real = slots > 0
    h = state[slots]  # a gathered copy
    rep = hv // hk
    qf, kf = l2norm(q.float()), l2norm(k.float())
    if rep > 1:
        qf, kf = qf.repeat_interleave(rep, 1), kf.repeat_interleave(rep, 1)
    o = recurrent_step(h, qf * dk ** -0.5, kf, v.float(), torch.exp(g), beta)
    if bool(real.all()):
        state[slots] = h
    elif bool(real.any()):
        state[slots[real]] = h[real]
    o = gated_rmsnorm(o.to(mixed_qkv.dtype), z, norm_w, eps)
    o = o.masked_fill(~real[:, None, None], 0) if not bool(real.all()) else o
    if quant is not None:
        from .fp8 import quant_ref

        quant_ref(o.reshape(n, hv * v_dim), quant)
        if out is None:
            return None
    if out is not None:
        out.copy_(o)
        return out
    return o


class GdnMtp:
    """MTP verify rows of the GDN decode kernel (GDN replay; docs/architecture.md, "Speculative decoding"). Each row is
    a window of L = 1 + k tokens. Its live state first replays the first `fold_n[req]` tokens of the slot's previous
    window from their records and is written back (the fold); the window's outputs are then computed from the folded
    state in registers, without writing it, and the window's records are stored for the next fold. Records are the
    kernel's own inputs, per value head, so a replay is the same arithmetic: k (before the L2 norm) and v [req slots,
    HV, L, 128], and `ab` [req slots, HV, L, 2] (a, b; layer views); `req` int32 [N], `fold_n` int32 [req slots]."""

    __slots__ = ("ab", "fold_n", "k", "req", "v", "width")

    def __init__(self, width: int, req: torch.Tensor, fold_n: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                 ab: torch.Tensor):
        self.width, self.req, self.fold_n, self.k, self.v, self.ab = width, req, fold_n, k, v, ab


def _replay(h: torch.Tensor, k_raw: torch.Tensor, v: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
            a_log: torch.Tensor, dt_bias: torch.Tensor) -> None:
    """One recorded token into state h [HV, V, K] (in place): the decode kernel's update without its output."""
    g, beta = gdn_gates(a[None], b[None], a_log, dt_bias)
    kf = l2norm(k_raw.float())[None]
    recurrent_step(h[None], torch.zeros_like(kf), kf, v.float()[None], torch.exp(g), beta)


def gdn_decode_mtp(mixed_qkv: torch.Tensor, a: torch.Tensor, b: torch.Tensor, *, a_log: torch.Tensor,
                   dt_bias: torch.Tensor, state: torch.Tensor, slots: torch.Tensor, z: torch.Tensor,
                   norm_w: torch.Tensor, eps: float, hk: int, dk: int, mtp: GdnMtp, out: torch.Tensor | None = None,
                   quant: QuantAct | None = None) -> torch.Tensor | None:
    """`gdn_decode` for MTP verify windows (`GdnMtp`): row r owns tokens [r L, (r + 1) L) of mixed_qkv [N L, ...],
    a, b and z; `slots` int32 [N] (int64 also on the reference) the live state slots (0: padding, zeros out). Returns
    the normed outputs [N L, HV, V] (or writes `out`/`quant` like `gdn_decode`)."""
    n_tok = mixed_qkv.shape[0]
    hv, v_dim = z.shape[1], z.shape[2]
    width = mtp.width
    n = n_tok // width
    if use_kernels(mixed_qkv) and dk == 128 and v_dim == 128 and mixed_qkv.stride(-1) == 1 and \
            state.stride(-1) == 1 and state.stride(-2) == dk and state.stride(-3) == dk * v_dim and \
            (quant is None or (quant.q.is_contiguous() and quant.s.stride(1) == 1)) and mtp.k.stride(-1) == 1:
        if out is not None and not out.is_contiguous():
            out = None
        if out is None and quant is None:
            out = torch.empty((n_tok, hv, v_dim), dtype=mixed_qkv.dtype, device=mixed_qkv.device)
        _check(build.load().lmk_gdn_mtp(
            _p(mixed_qkv), mixed_qkv.stride(0), _p(a), a.stride(0), _p(b), b.stride(0), _p(a_log), _p(dt_bias),
            {torch.float32: 0, torch.bfloat16: 1, torch.float16: 2}[dt_bias.dtype], _p(slots), _p(mtp.req),
            _p(mtp.fold_n), _p(state), state.stride(0), int(state.dtype == torch.bfloat16), _p(z), z.stride(0),
            _p(norm_w), int(norm_w.dtype == torch.bfloat16), _p(out),
            _p(quant.q if quant is not None else None), _p(quant.s if quant is not None else None),
            quant.s.stride(0) if quant is not None else 0, _p(mtp.k), _p(mtp.v), _p(mtp.ab), mtp.k.stride(0), width,
            n, hk, hv, dk ** -0.5, eps, 0, _stream(mixed_qkv)), "gdn_mtp")
        return out
    kd = hk * dk
    rep = hv // hk
    res = torch.zeros((n_tok, hv, v_dim), dtype=mixed_qkv.dtype, device=mixed_qkv.device)
    for r in range(n):
        ss, q = int(slots[r]), int(mtp.req[r])
        t0, t1 = r * width, (r + 1) * width
        x = mixed_qkv[t0:t1]
        kr = x[:, kd:2 * kd].reshape(width, hk, dk).repeat_interleave(rep, 1)  # raw k per value head
        vr = x[:, 2 * kd:].reshape(width, hv, v_dim)
        if ss > 0:
            h = state[ss].float().clone()
            for t in range(int(mtp.fold_n[q])):
                _replay(h, mtp.k[q, :, t], mtp.v[q, :, t], mtp.ab[q, :, t, 0], mtp.ab[q, :, t, 1], a_log, dt_bias)
            if int(mtp.fold_n[q]):
                state[ss] = h.to(state.dtype)
            g, beta = gdn_gates(a[t0:t1], b[t0:t1], a_log, dt_bias)
            qf = l2norm(x[:, :kd].reshape(width, hk, dk).float()).repeat_interleave(rep, 1)
            kf = l2norm(kr.float())
            for t in range(width):
                o = recurrent_step(h[None], qf[t:t + 1] * dk ** -0.5, kf[t:t + 1], vr[t:t + 1].float(),
                                   torch.exp(g[t:t + 1]), beta[t:t + 1])
                res[t0 + t] = o[0].to(res.dtype)
            res[t0:t1] = gated_rmsnorm(res[t0:t1], z[t0:t1], norm_w, eps)
        mtp.k[q, :, :width] = kr.transpose(0, 1)
        mtp.v[q, :, :width] = vr.transpose(0, 1)
        mtp.ab[q, :, :width, 0] = a[t0:t1].transpose(0, 1)
        mtp.ab[q, :, :width, 1] = b[t0:t1].transpose(0, 1)
    if quant is not None:
        from .fp8 import quant_ref

        quant_ref(res.reshape(n_tok, hv * v_dim), quant)
        if out is None:
            return None
    if out is not None:
        out.copy_(res)
        return out
    return res


def gdn_fold(state: torch.Tensor, rec_k: torch.Tensor, rec_v: torch.Tensor, rec_ab: torch.Tensor, *,
             a_log: torch.Tensor, dt_bias: torch.Tensor, req: torch.Tensor, src: torch.Tensor, dst: torch.Tensor,
             n: torch.Tensor) -> None:
    """FOLD and MTP decode checkpoints for the recurrent state of every GDN layer: state [slots, layers, HV, V, K];
    records [req slots, layers, HV, L, 128] and [req slots, layers, HV, L, 2]; a_log [layers, HV] fp32 and dt_bias
    [layers, HV]. Per row, the state of slot `src` replays the first `n` records of request slot `req` (the decode
    kernel's arithmetic) and goes to slot `dst`. All int64 [rows]."""
    if use_kernels(state) and state.is_contiguous() and rec_k.is_contiguous() and rec_v.is_contiguous() and \
            rec_ab.is_contiguous() and state.shape[-1] == 128 and state.shape[-2] == 128:
        rows, (_, layers, hv, _, _) = len(req), state.shape
        _check(build.load().lmk_gdn_fold(
            _p(state), _p(rec_k), _p(rec_v), _p(rec_ab), _p(a_log), _p(dt_bias),
            {torch.float32: 0, torch.bfloat16: 1, torch.float16: 2}[dt_bias.dtype], _p(req), _p(src), _p(dst), _p(n),
            rows, layers, hv, rec_k.shape[3], int(state.dtype == torch.bfloat16), _stream(state)), "gdn_fold")
        return
    for q, a, d, k in zip(req.tolist(), src.tolist(), dst.tolist(), n.tolist()):
        for layer in range(state.shape[1]):
            h = state[a, layer].float().clone()
            for t in range(k):
                _replay(h, rec_k[q, layer, :, t], rec_v[q, layer, :, t], rec_ab[q, layer, :, t, 0],
                        rec_ab[q, layer, :, t, 1], a_log[layer], dt_bias[layer])
            state[d, layer] = h.to(state.dtype)


# -- FP8 activation quantization (block-FP8 linear layers, kernels.fp8) ---------------------------------------------
# Each producer equals its BF16 op followed by `fp8.quant_ref`, except SiLU-and-mul, which quantizes the fp32 product
# and divides by the scale (`silu_mul_quant` in csrc/quant.cu; `quant_ref(..., divide=True)`, as vLLM's fused kernel).
# Scales are written MN-major [K / 128, ld] (fp8.QuantAct).


def _aligned4(t: torch.Tensor, row_stride: int) -> bool:
    return t.data_ptr() % 8 == 0 and row_stride % 4 == 0


def _bf16(*ts: torch.Tensor) -> bool:
    return all(t.dtype == torch.bfloat16 for t in ts)


def quant_fp8(x: torch.Tensor) -> QuantAct:
    """x [T, K] (K a multiple of 128) -> the per-(token, 128-group) e4m3 quantization."""
    from .fp8 import empty_act, quant_ref

    t, k = x.shape
    if not use_kernels(x) or not _bf16(x) or x.stride(-1) != 1 or k % 128 or not _aligned4(x, x.stride(0)):
        return quant_ref(x)
    a = empty_act(t, k, x.device)
    _check(build.load().lmk_quant_fp8(_p(x), x.stride(0), _p(a.q), _p(a.s), a.s.stride(0), t, k, _stream(x)),
           "quant_fp8")
    return a


def rmsnorm_quant(x: torch.Tensor, w: torch.Tensor, eps: float, residual: torch.Tensor | None = None,
                  bf16: bool = False) -> tuple[QuantAct, torch.Tensor | None]:
    """`rmsnorm` (with the residual add when given) quantized for a block-FP8 consumer, in one kernel. Returns the
    QuantAct (carrying the BF16 output too when `bf16`) and the new residual (None without `residual`)."""
    from .fp8 import empty_act, quant_ref

    n, d = x.shape
    if not use_kernels(x) or not _bf16(x, w, *([residual] if residual is not None else [])) or x.stride(-1) != 1 or \
            d % 128:
        r = rmsnorm(x, w, eps, residual)
        y, res = r if residual is not None else (r, None)
        a = quant_ref(y)
        a.bf16 = y if bf16 else None
        return a, res
    a = empty_act(n, d, x.device)
    out = torch.empty((n, d), dtype=x.dtype, device=x.device) if bf16 else None
    res_out = None
    if residual is not None:
        residual = residual.contiguous()
        res_out = torch.empty((n, d), dtype=x.dtype, device=x.device)
    _check(build.load().lmk_gemma_rmsnorm_quant(_p(x), x.stride(0), _p(residual), _p(w), _p(out), _p(res_out),
                                                _p(a.q), _p(a.s), a.s.stride(0), n, d, eps, _stream(x)),
           "gemma_rmsnorm_quant")
    a.bf16 = out
    return a, res_out


def gated_rmsnorm_quant(x: torch.Tensor, z: torch.Tensor, w: torch.Tensor, eps: float, out: QuantAct) -> QuantAct:
    """`gated_rmsnorm` quantized into `out` (views q [T, H * D], s [H * D / 128, >= T]); fused for D = 128 (one
    group per head)."""
    from .fp8 import quant_ref

    t, h, d = x.shape
    if not use_kernels(x) or not _bf16(x, z, w) or d != 128 or not x.is_contiguous() or z.stride(2) != 1 or \
            z.stride(1) != d or not out.q.is_contiguous() or out.s.stride(1) != 1:
        return quant_ref(gated_rmsnorm(x, z, w, eps).reshape(t, h * d), out)
    _check(build.load().lmk_gated_rmsnorm_quant(_p(x), _p(z), z.stride(0), _p(w), _p(out.q), _p(out.s),
                                                out.s.stride(0), t, h, d, eps, _stream(x)), "gated_rmsnorm_quant")
    return out


def silu_mul_quant(gu: torch.Tensor) -> QuantAct:
    """gu [T, 2I] (gate, up) -> silu(gate) * up quantized from fp32, unrounded (csrc/quant.cu)."""
    from .fp8 import empty_act, quant_ref

    t, i2 = gu.shape
    if not use_kernels(gu) or not _bf16(gu) or gu.stride(-1) != 1 or (i2 // 2) % 128 or \
            not _aligned4(gu, gu.stride(0)):
        g, u = gu.chunk(2, dim=-1)
        return quant_ref(F.silu(g.float()) * u.float(), divide=True)
    a = empty_act(t, i2 // 2, gu.device)
    _check(build.load().lmk_silu_mul_quant(_p(gu), gu.stride(0), _p(a.q), _p(a.s), a.s.stride(0), t, i2 // 2,
                                           _stream(gu)), "silu_mul_quant")
    return a


def sigmoid_gate_quant(o: torch.Tensor, gate: torch.Tensor) -> QuantAct:
    """`sigmoid_gate` (rounded to BF16) quantized for a block-FP8 o_proj; head_dim a multiple of 128."""
    from .fp8 import empty_act, quant_ref

    t, h, d = gate.shape
    if not use_kernels(o) or not _bf16(o, gate) or not o.is_contiguous() or gate.stride(2) != 1 or d % 128 or \
            not _aligned4(gate, gate.stride(0)) or gate.stride(1) % 4 or not _aligned4(o, h * d):
        return quant_ref(sigmoid_gate(o, gate))
    a = empty_act(t, h * d, o.device)
    _check(build.load().lmk_sigmoid_gate_quant(_p(o), _p(gate), gate.stride(0), gate.stride(1), _p(a.q), _p(a.s),
                                               a.s.stride(0), t, h, d, _stream(o)), "sigmoid_gate_quant")
    return a


def dequant_fp8(w: torch.Tensor, ws: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """A block-FP8 weight w [N, K] (e4m3, scales ws [N / 128, K / 128]) in BF16: each value times its block's scale
    in fp32, rounded once (`fp8.dequant_weight`, then a cast). Writes into `out` ([N, K], contiguous) when given."""
    from .fp8 import dequant_weight

    n, k = w.shape
    if not use_kernels(w) or n % 128 or k % 128 or w.data_ptr() % 16 or not w.is_contiguous() or \
            not ws.is_contiguous() or (out is not None and (not out.is_contiguous() or out.data_ptr() % 16)):
        y = dequant_weight(w, ws).to(torch.bfloat16)
        if out is None:
            return y
        out.copy_(y)
        return out
    out = torch.empty((n, k), dtype=torch.bfloat16, device=w.device) if out is None else out
    _check(build.load().lmk_dequant_fp8(_p(w), _p(ws), _p(out), n, k, _stream(w)), "dequant_fp8")
    return out


# -- sampling ------------------------------------------------------------------------------------------------------


def sample(logits: torch.Tensor, *, row_slot: torch.Tensor, qpos: torch.Tensor, temperature: torch.Tensor,
           top_p: torch.Tensor, top_k: torch.Tensor, min_p: torch.Tensor, seed: torch.Tensor, out: torch.Tensor,
           status: torch.Tensor, pending: torch.Tensor | None = None, ids: torch.Tensor | None = None,
           mask: torch.Tensor | None = None, mask_row: torch.Tensor | None = None, padded: bool = False,
           terms: Any = None, drafts: torch.Tensor | None = None,
           window: tuple[torch.Tensor, torch.Tensor] | None = None) -> None:
    """The fused counter-based sampler (CUDA only; the definition and the torch reference are in
    runner/cuda/sampler.py). logits [n, V] fp32; per row: `row_slot` [n] int64 into the per-slot parameter tables
    (temperature, top_p, top_k int64, min_p, seed int64) and `qpos` [n] int64, the position of the row's last query
    token. Writes the tokens into out[:n] and the status (NONFINITE, MASK_EMPTY bits) into status[:n] (int64), and each
    token into pending[row_slot] when `pending` is given. The non-finite check reads every column.

    `ids` (int32 [V], ascending): the token id of each logit column, for a reduced vocabulary (the drafter's head);
    the noise and the tokens are those ids'. With `padded` (the grammar-sparse head over the union of the step's
    masks, the drafter's head over the grammar's universe) the valid ids come first and entries < 0 follow,
    padding: never sampled, nor counted by min-p, top-k or top-p. Without it every id must be >= 0.

    `mask` (int32 [m, words], over token ids) and `mask_row` (int64 [n]): grammar masks; row r samples only the
    tokens mask[mask_row[r]] allows (-1: all), column v tested by the bit of its token (ids[v] with `ids`). A row with
    no allowed column (its mask allows none of the columns, or they are all padding) gets token 0 and status 2
    (MASK_EMPTY); a non-finite logit wins (status 1).

    `terms` (`runner.cuda.penalties.PenaltyTables`): the penalties and logit bias of each row's slot, applied where a
    logit is read (the reference's `apply_terms`); with `pending`, a penalised row's token counts. `window` (MTP):
    the static block's tokens and each row's sample token, from which a verify row's drafts before it (`drafts`
    [slots, k]) count too. None: the kernel without terms."""
    n, v = logits.shape
    if padded and ids is None:
        raise ValueError("padded ids need ids")
    t = terms
    if t is None:
        targs: tuple = (None, None, None, None, None, 0, None, None, None, 0, None, 0, None, None, 0)
    else:
        w_tok, w_idx = window if window is not None else (None, None)
        targs = (_p(t.terms), _p(t.rep), _p(t.pres), _p(t.freq), _p(t.cnt), t.cnt.stride(0), _p(t.bias_n),
                 _p(t.bias_id), _p(t.bias_val), t.bias_id.stride(0), _p(drafts if window is not None else None),
                 drafts.shape[1] if window is not None else 0, _p(w_tok), _p(w_idx), int(pending is not None))
    _check(build.load().lmk_sample(_p(logits), logits.stride(0), v, n, _p(row_slot), _p(qpos), _p(temperature),
                                   _p(top_p), _p(top_k), _p(min_p), _p(seed), _p(out), _p(status), _p(pending),
                                   _p(ids), int(padded), _p(mask), mask.stride(0) if mask is not None else 0,
                                   _p(mask_row), *targs, _stream(logits)), "sample")
