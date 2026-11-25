"""Gemma 4's fused ops: our CUDA kernels (`csrc/gemma.cu`) on CUDA tensors, the torch reference elsewhere.

Each reference defines the op's numerics (models/gemma4/model.py: a compiled graph's rule, fp32 inside a fused op, one
rounding where it stores); the kernels round at the same points and differ only in reduction order
(tests/kernels/test_gemma_kernels.py bounds it on a GPU). `ops.enabled(False)` forces the references on CUDA too.

**Quantizing producers** (block-FP8 checkpoints): `rmsnorm_quant`, `sandwich_quant` and `gelu_tanh_mul_quant` return
their op's output quantized for a block-FP8 projection (`kernels.fp8.QuantAct`), in one kernel each that equals the BF16
kernel followed by the standalone quantization bit for bit (`fp8.quant_ref` of the BF16 output). Without the kernels
they are exactly that: the BF16 op, then `fp8.quant_ref`.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from . import build, fp8
from .fp8 import QuantAct
from .ops import _check, _p, _stream, use_kernels


def _rms(xf: torch.Tensor, eps: float) -> torch.Tensor:
    return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)


def rmsnorm(x: torch.Tensor, w: torch.Tensor | None, eps: float) -> torch.Tensor:
    """`x / rms(x) * w` over the last dim (no weight: `x / rms(x)`), fp32, rounded once."""
    if w is not None and use_kernels(x) and x.dim() == 2 and x.stride(-1) == 1 and x.dtype == torch.bfloat16:
        n, d = x.shape
        out = torch.empty((n, d), dtype=x.dtype, device=x.device)
        _check(build.load().lmk_g4_rmsnorm(_p(x), x.stride(0), _p(w), _p(out), n, d, eps, _stream(x)), "g4_rmsnorm")
        return out
    y = _rms(x.float(), eps)
    if w is not None:
        y = y * w.float()
    return y.to(x.dtype)


def sandwich(a: torch.Tensor, residual: torch.Tensor, w_post: torch.Tensor, w_next: torch.Tensor, eps: float,
             scalar: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """The end of a sandwich block and the next norm: the new residual `(residual + a / rms(a) * w_post) * scalar`
    (fp32, rounded once; an eager layer rounds after the norm, the add and the scale, a compiled graph may fuse them)
    and the next layer's input, that rounded residual normed with `w_next`, rounded once. a, residual [T, D]; scalar a
    1-element tensor or None."""
    if use_kernels(a) and a.dim() == 2 and a.stride(-1) == 1 and a.dtype == torch.bfloat16:
        t, d = a.shape
        residual = residual.contiguous()
        r_out = torch.empty((t, d), dtype=a.dtype, device=a.device)
        x_out = torch.empty_like(r_out)
        _check(build.load().lmk_g4_sandwich(_p(a), a.stride(0), _p(residual), _p(w_post), _p(w_next), _p(scalar),
                                            _p(r_out), _p(x_out), t, d, eps, _stream(a)), "g4_sandwich")
        return r_out, x_out
    s = residual.float() + _rms(a.float(), eps) * w_post.float()
    if scalar is not None:
        s = s * scalar.float()
    r = s.to(a.dtype)
    return r, (_rms(r.float(), eps) * w_next.float()).to(a.dtype)


def gelu_tanh_mul(gu: torch.Tensor) -> torch.Tensor:
    """gu [T, 2I] (gate, up) -> gelu_tanh(gate) * up [T, I], fp32, rounded once (torch's tanh-approximate GELU)."""
    if use_kernels(gu) and gu.stride(-1) == 1 and gu.dtype == torch.bfloat16:
        t, i2 = gu.shape
        out = torch.empty((t, i2 // 2), dtype=gu.dtype, device=gu.device)
        _check(build.load().lmk_gelu_tanh_mul(_p(gu), gu.stride(0), _p(out), t, i2 // 2, _stream(gu)),
               "gelu_tanh_mul")
        return out
    g, u = gu.chunk(2, dim=-1)
    return (F.gelu(g.float(), approximate="tanh") * u.float()).to(gu.dtype)


def rmsnorm_quant(x: torch.Tensor, w: torch.Tensor, eps: float) -> QuantAct:
    """`rmsnorm(x, w)` rounded to x's dtype, then quantized per token and group of 128 (D a multiple of 128)."""
    if use_kernels(x) and x.dim() == 2 and x.stride(-1) == 1 and x.dtype == w.dtype == torch.bfloat16 and \
            x.shape[1] % 128 == 0:
        n, d = x.shape
        a = fp8.empty_act(n, d, x.device)
        rc = build.load().lmk_g4_rmsnorm_quant(_p(x), x.stride(0), _p(w), _p(a.q), _p(a.s), a.s.stride(0), n, d, eps,
                                               _stream(x))
        if rc != -1:
            _check(rc, "g4_rmsnorm_quant")
            return a
    return fp8.quant_ref(rmsnorm(x, w, eps))


def sandwich_quant(a: torch.Tensor, residual: torch.Tensor, w_post: torch.Tensor, w_next: torch.Tensor, eps: float,
                   scalar: torch.Tensor | None = None) -> tuple[torch.Tensor, QuantAct]:
    """`sandwich`, its second output (the next layer's normed input) quantized instead of stored: (the new residual,
    the quantization of the next norm's output rounded to the model dtype)."""
    if use_kernels(a) and a.dim() == 2 and a.stride(-1) == 1 and a.dtype == torch.bfloat16 and a.shape[1] % 128 == 0:
        t, d = a.shape
        residual = residual.contiguous()
        r_out = torch.empty((t, d), dtype=a.dtype, device=a.device)
        q = fp8.empty_act(t, d, a.device)
        rc = build.load().lmk_g4_sandwich_quant(_p(a), a.stride(0), _p(residual), _p(w_post), _p(w_next), _p(scalar),
                                                _p(r_out), _p(q.q), _p(q.s), q.s.stride(0), t, d, eps, _stream(a))
        if rc != -1:
            _check(rc, "g4_sandwich_quant")
            return r_out, q
    r, x = sandwich(a, residual, w_post, w_next, eps, scalar)
    return r, fp8.quant_ref(x)


def gelu_tanh_mul_quant(gu: torch.Tensor) -> QuantAct:
    """`gelu_tanh_mul(gu)` rounded to gu's dtype, then quantized per token and group of 128 (I a multiple of 128):
    vLLM stores Gemma 4's GELU output in BF16 and quantizes it (its activation-quant fusion is SiLU's only)."""
    if use_kernels(gu) and gu.dim() == 2 and gu.stride(-1) == 1 and gu.dtype == torch.bfloat16 and \
            (gu.shape[1] // 2) % 128 == 0:
        t, i2 = gu.shape
        a = fp8.empty_act(t, i2 // 2, gu.device)
        rc = build.load().lmk_gelu_tanh_mul_quant(_p(gu), gu.stride(0), _p(a.q), _p(a.s), a.s.stride(0), t, i2 // 2,
                                                  _stream(gu))
        if rc != -1:
            _check(rc, "gelu_tanh_mul_quant")
            return a
    return fp8.quant_ref(gelu_tanh_mul(gu))


def embed_scale(e: torch.Tensor, normalizer: torch.Tensor) -> torch.Tensor:
    """The scaled embedding: one multiply in the model dtype (HF's `embed_scale.to(dtype)`)."""
    return e * normalizer


# -- the attention input side --------------------------------------------------------------------------------------

E4M3_MAX = 448.0  # float8_e4m3fn's largest finite value


def kv_fp8(x: torch.Tensor) -> torch.Tensor:
    """x as an FP8 KV pool stores it (`cache.kv_cache_dtype fp8`): E4M3 bytes (uint8) of x clamped to ±448 (PTX's
    `.satfinite`: an overflow or an infinity saturates, a NaN stays a NaN) and rounded to nearest even. x is the
    value a BF16 pool would hold: the store rounds twice, and a kernel that stores FP8 must round the same way."""
    return x.float().clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).view(torch.uint8)


def kv_cast(x: torch.Tensor, to_fp8: bool) -> torch.Tensor:
    """The kernels' own FP8 KV conversions (csrc/kv_fp8.cuh; tests): BF16 values to E4M3 bytes as the FP8 store rounds
    them, or E4M3 bytes to BF16 as the FP8 attention reads them. x contiguous on CUDA."""
    if x.dtype != (torch.bfloat16 if to_fp8 else torch.uint8) or not x.is_contiguous() or not x.is_cuda:
        raise ValueError(f"kv_cast: a contiguous CUDA tensor of {'bfloat16' if to_fp8 else 'uint8'}, not {x.dtype}")
    out = torch.empty(x.shape, dtype=torch.uint8 if to_fp8 else torch.bfloat16, device=x.device)
    _check(build.load().lmk_g4_kv_cast(_p(x), _p(out), x.numel(), int(to_fp8), _stream(x)), "g4_kv_cast")
    return out


def kv_read(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Values read from a KV pool, in `dtype`: an FP8 pool's bytes (uint8) dequantized (exact into BF16), any other
    pool's as they are."""
    return x.view(torch.float8_e4m3fn).to(dtype) if x.dtype == torch.uint8 else x


def _kv_store(x: torch.Tensor, pool: torch.Tensor) -> torch.Tensor:
    return kv_fp8(x) if pool.dtype == torch.uint8 else x


def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """NeoX rotation over the whole head: x [T, H, D] fp32, cos/sin [T, D / 2] (pairs d and d + D / 2); fp32."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    c, s = cos[:, None, :].float(), sin[:, None, :].float()
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)


def qkv_store(qkv: torch.Tensor, *, heads: int, kv_heads: int, head_dim: int, k_eq_v: bool, q_w: torch.Tensor,
              k_w: torch.Tensor, eps: float, positions: torch.Tensor, cos_cache: torch.Tensor,
              sin_cache: torch.Tensor, kv_layer: torch.Tensor, kv_page: torch.Tensor,
              kv_off: torch.Tensor) -> torch.Tensor:
    """A Gemma 4 attention layer's input side: q = RoPE(q_norm(q)) (rounded once), and into the KV pages at
    (`kv_page`, `kv_off`) per token: K = RoPE(k_norm(k)) and V = v_norm(v) (no weight), or with `k_eq_v` (a full
    layer) V = v_norm(k) alone, the plane the attention derives K from (models/gemma4/model.py). `kv_layer` is a layer's
    view of its pool: [pages, 2, page_size, Hkv, D] (K and V planes), or [pages, page_size, Hkv, D] with `k_eq_v`; a
    uint8 view is an FP8 pool, which stores each value as `kv_fp8` rounds it (the kernel: with `k_eq_v`). Returns q
    [T, H, D]."""
    t = qkv.shape[0]
    h, hkv, d = heads, kv_heads, head_dim
    fp8 = kv_layer.dtype == torch.uint8
    if use_kernels(qkv) and qkv.stride(-1) == 1 and qkv.dtype == torch.bfloat16 and d % 64 == 0 and d <= 512 and \
            (kv_layer.dtype == qkv.dtype or (fp8 and k_eq_v)):
        q_out = torch.empty((t, h, d), dtype=qkv.dtype, device=qkv.device)
        _check(build.load().lmk_g4_qkv(
            _p(qkv), qkv.stride(0), int(k_eq_v), _p(q_w), _p(k_w), _p(positions), _p(cos_cache), _p(sin_cache),
            _p(q_out), _p(kv_layer), kv_layer.stride(0), kv_layer.shape[-3], int(fp8), _p(kv_page), _p(kv_off), t, h,
            hkv, d, eps, _stream(qkv)), "g4_qkv")
        return q_out
    q = qkv[:, :h * d].reshape(t, h, d)
    k = qkv[:, h * d:(h + hkv) * d].reshape(t, hkv, d)
    v = k if k_eq_v else qkv[:, (h + hkv) * d:(h + 2 * hkv) * d].reshape(t, hkv, d)
    cos, sin = cos_cache[positions], sin_cache[positions]
    q_out = rope(_rms(q.float(), eps) * q_w.float(), cos, sin).to(qkv.dtype)
    v_out = _rms(v.float(), eps).to(qkv.dtype)
    if k_eq_v:
        kv_layer[kv_page, kv_off] = _kv_store(v_out, kv_layer)
    else:
        kv_layer[:, 0][kv_page, kv_off] = _kv_store(rope(_rms(k.float(), eps) * k_w.float(), cos, sin).to(qkv.dtype),
                                                    kv_layer)
        kv_layer[:, 1][kv_page, kv_off] = _kv_store(v_out, kv_layer)
    return q_out


def q_rope(q: torch.Tensor, *, heads: int, head_dim: int, q_w: torch.Tensor, eps: float, positions: torch.Tensor,
           cos_cache: torch.Tensor, sin_cache: torch.Tensor) -> torch.Tensor:
    """A Q-only layer's input side (the MTP assistant's): RoPE(q_norm(q)), rounded once, as `qkv_store` computes q,
    with no KV write (the kernel with no KV heads). q [T, H·D]; returns [T, H, D]."""
    t, h, d = q.shape[0], heads, head_dim
    if use_kernels(q) and q.stride(-1) == 1 and q.dtype == torch.bfloat16 and d % 64 == 0 and d <= 512:
        q_out = torch.empty((t, h, d), dtype=q.dtype, device=q.device)
        _check(build.load().lmk_g4_qkv(
            _p(q), q.stride(0), 0, _p(q_w), _p(None), _p(positions), _p(cos_cache), _p(sin_cache), _p(q_out),
            _p(None), 0, 0, 0, _p(None), _p(None), t, h, 0, d, eps, _stream(q)), "g4_qkv")
        return q_out
    cos, sin = cos_cache[positions], sin_cache[positions]
    return rope(_rms(q.reshape(t, h, d).float(), eps) * q_w.float(), cos, sin).to(q.dtype)


def k_from_v(u: torch.Tensor, k_w: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """A k_eq_v layer's keys from its stored plane: K = RoPE(u ⊙ k_norm.weight), fp32, rounded to u's dtype (what the
    attention kernel multiplies). u [n, Hkv, D]; cos/sin [n, D / 2] at the keys' positions."""
    return rope(u.float() * k_w.float(), cos, sin).to(u.dtype)


def attention_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, qpos: torch.Tensor, kpos: torch.Tensor,
                  window: int = 0, qlim: torch.Tensor | None = None) -> torch.Tensor:
    """Reference attention of one row, scale 1.0 (Gemma 4): q [n, H, D], k and v [m, Hkv, D] (GQA: query head h reads KV
    head h // (H / Hkv)), causal by position (key position <= query position; with `qlim` [n], <= each query's limit
    instead: an image span's queries see the whole span) and, with `window` W > 0, a key no older than W − 1 positions.
    Softmax in fp32 (SDPA); returns [n, H, D] in q's dtype."""
    mask = kpos[None, :] <= (qpos if qlim is None else qlim)[:, None]
    if window:
        mask &= kpos[None, :] > qpos[:, None] - window
    o = F.scaled_dot_product_attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None],
                                       attn_mask=mask, scale=1.0, enable_gqa=True)
    return o[0].transpose(0, 1)


def lib() -> Any:
    return build.load()
