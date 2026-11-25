"""Block-FP8 linear layers (docs/architecture.md, "FP8"): dynamic activation quantization and the FP8 GEMM.

**Format** (both checkpoints' formats: `models.fp8`): a weight [N, K] in e4m3 with one FP32 scale per 128 × 128 block
[N / 128, K / 128] (the dequantized weight is `w * scale`); activations quantized per token and per group of 128 along
K, dynamically, to e4m3 with an FP32 scale.

**Activations** (`QuantAct`): q [m, K] e4m3 and the scales "MN-major" [K / 128, ld] (ld >= m; token t's scale of group g
at s[g, t]), the layout the CUTLASS blockwise GEMM reads. The numerics (`quant_ref`), vLLM's: scale = max(absmax, 1e-10)
/ 448, q = e4m3_rn(clamp(x · (1 / scale), ±448)), the reciprocal one IEEE division a group; the SiLU-and-mul fusion
divides, q = e4m3_rn(clamp(x / scale, ±448)), as vLLM's fused kernel does. The quantizing kernels (`csrc/quant.cu`,
`csrc/gemma.cu` and the GDN decode and verify kernels' epilogues; `kernels.ops`, `kernels.gemma_ops`) reproduce it bit
for bit, NaN and ±Inf included: a NaN leaves the absmax and quantizes to −448, so the FP8 path turns it finite.

**GEMM** (`gemm`): D = Σ_kb (A_kb · B_kbᵀ) · sa[kb] · sb[kb], bf16 out, fp32 accumulation; `gemm_ref` dequantizes and
multiplies in fp32 (the definition; kernels differ only in summation order). Backends, chosen per (M, N) by the dispatch
table (`table`):
- `cutlass`: the CUTLASS SM120 blockwise kernel, vendored into our CUTLASS library (`kernels/cutlass/`), in its
  three configurations: 0 swap-AB 128×32×128, 1 ping-pong 64×128×128, 2 cooperative 128×128×128; and 3 swap-AB
  128×64×128, SGLang's tile for 128 < M ≤ 256;
- `torch`: `torch._scaled_mm` with 1×128 and 128×128 block scales (cuBLASLt), where torch supports it.
The default table is set from measurements on SM120, see `DEFAULT_TABLE`. Every backend runs on the caller's stream
with no host synchronisation, so the GEMMs are captured in the runner's CUDA graphs; the CUTLASS workspace is
allocated once per device (before any capture: the runner's first steps are eager). Every configuration gives the
same bits, those of vLLM's CUTLASS blockwise path, so the table is a speed choice only.

**BF16 activations** (per row: `precise_rows`, `bf16_rows`, `linear_bf16`): a block-FP8 projection takes the inputs
of a step's PRECISE rows (sched/plan.py) in BF16 and multiplies them with the weight dequantized to BF16 (one rounding
of w × scale) in a BF16 GEMM with fp32 accumulation, as the BF16 model's projection does; the step's other rows keep
the FP8 path, bit for bit. Which rows are PRECISE is the scheduler's choice (docs/architecture.md, "The precise
prefix").
"""

from __future__ import annotations

import ctypes
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch

from . import build

log = logging.getLogger(__name__)

GROUP = 128  # activation group and weight block (K and N)
FP8_MAX = 448.0
QUANT_EPS = 1e-10
FP8 = torch.float8_e4m3fn

# (largest M, largest N, backend, config): the first row covering the call. Measured on SM120 (RTX PRO 6000, 188
# SMs, on Qwen3.8 27B's shapes, confirmed on an RTX 5090): swap-AB (128 of N x 32 of M per tile) is the fastest at
# every decode M; at M = 128 it still beats 128-row tiles on the 5,120-wide outputs (40 N tiles leave most SMs idle)
# and on gate_up (34,816), while qkv and qkvz (one wave of 128 x 128 tiles) prefer the cooperative 128x128; ping-pong
# 64x128 wins at M <= 256, cooperative 128x128 above. On Gemma 4 31B's shapes: the same picture at decode M and above
# 256, but its two N that Qwen has not want the cooperative 128x128 earlier: gate_up (43,008) at M = 128 and 256, the
# full layers' qkv (18,432) at 256. Those rows only cover N that Qwen's projections do not (5,120, 14,336, 16,384,
# 34,816), so Qwen's choices stay as above (tests/runner/test_fp8.py). SGLang's swap-AB 128x64 (config 3) is never best
# by a clear margin.
BIG = 1 << 30
DEFAULT_TABLE: list[tuple[int, int, str, int]] = [
    (96, BIG, "cutlass", 0), (128, 8192, "cutlass", 0), (128, 20480, "cutlass", 2), (128, 36864, "cutlass", 0),
    (128, BIG, "cutlass", 2), (256, 16384, "cutlass", 1), (256, 20480, "cutlass", 2), (256, 36864, "cutlass", 1),
    (256, BIG, "cutlass", 2), (BIG, BIG, "cutlass", 2)]
CUTLASS_WORKSPACE_BYTES = 32 << 20


@dataclass
class QuantAct:
    """A quantized activation: q [m, K] e4m3, s [K / 128, ld] fp32 (MN-major, ld >= m). `bf16` optionally carries
    the unquantized BF16 values too, for a consumer that is not FP8 (the GDN layers' BF16 in_proj_ba)."""

    q: torch.Tensor
    s: torch.Tensor
    bf16: torch.Tensor | None = None

    @property
    def m(self) -> int:
        return self.q.shape[0]


def scale_ld(m: int) -> int:
    """The leading dimension of a scale buffer for m tokens: packed (m). The CUTLASS kernels derive the scales' layout
    from m; the A/B backend (cuBLAS) pads its own copies."""
    return m


def empty_act(m: int, k: int, device: torch.device) -> QuantAct:
    return QuantAct(torch.empty((m, k), dtype=FP8, device=device),
                    torch.empty((k // GROUP, scale_ld(m)), dtype=torch.float32, device=device))


# -- references (the definitions; the CPU path and the tests) --------------------------------------------------------


def quant_ref(x: torch.Tensor, out: QuantAct | None = None, divide: bool = False) -> QuantAct:
    """Per (token, group of 128) e4m3 quantization of x [m, K] (any float dtype; computed in fp32): x times the
    reciprocal of the group's scale (one IEEE division a group), or with `divide` x divided by it (the SiLU-and-mul
    fusion's form). Writes into `out` when given (q [m, K] and s [K / 128, >= m] views).

    Non-finite values as the kernels' fmaxf and fminf take them (torch's fmax and fmin drop NaN too): a NaN is left out
    of its group's absmax and clamped to −448 (byte 0xFE) under the scale of the group's other values (an all-NaN
    group: the floor's, 1e-10 / 448, whose reciprocal is finite); an infinite value makes its group's scale Inf, its
    reciprocal 0, its own byte 0xFE (Inf · 0, or Inf / Inf, is NaN) and the finite values' ±0."""
    m, k = x.shape
    xf = x.float().reshape(m, k // GROUP, GROUP)
    lim = torch.full((), FP8_MAX, device=x.device)
    scale = _div(xf.abs().fmax(torch.zeros_like(lim)).amax(-1).clamp_min(QUANT_EPS), FP8_MAX)  # [m, G]
    if divide:
        y = xf / scale[..., None]
    else:  # a tensor division: torch would turn a scalar one into a multiply by an inexact reciprocal (`_div`)
        y = xf * (torch.ones_like(scale) / scale)[..., None]
    q = y.fmax(-lim).fmin(lim).to(FP8).reshape(m, k)
    if out is None:
        out = QuantAct(torch.empty((m, k), dtype=FP8, device=x.device),
                       torch.empty((k // GROUP, scale_ld(m)), dtype=torch.float32, device=x.device))
    out.q.copy_(q)
    out.s[:, :m].copy_(scale.T)
    return out


def _div(t: torch.Tensor, d: float) -> torch.Tensor:
    """t / d, IEEE-rounded. On CUDA, torch divides by a Python scalar as a multiply by its reciprocal, which differs in
    the last bit (1/448 is inexact) and then flips e4m3 rounding ties; the kernels divide."""
    return t / torch.full_like(t, d)


def dequant_act(a: QuantAct) -> torch.Tensor:
    m, k = a.q.shape
    return (a.q.float().reshape(m, k // GROUP, GROUP) * a.s[:, :m].T[..., None]).reshape(m, k)


def dequant_weight(w: torch.Tensor, ws: torch.Tensor) -> torch.Tensor:
    """w [N, K] e4m3, ws [N / 128, K / 128] -> fp32 [N, K]."""
    n, k = w.shape
    return (w.float().reshape(n // GROUP, GROUP, k // GROUP, GROUP) * ws[:, None, :, None]).reshape(n, k)


def gemm_ref(a: QuantAct, w: torch.Tensor, ws: torch.Tensor, out_dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    return (dequant_act(a) @ dequant_weight(w, ws).T).to(out_dtype)


def quantize_weight(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Block-quantize a weight [N, K] (both multiples of 128) the way FP8 checkpoints are made: one scale per 128 × 128
    block, amax / 448."""
    n, k = w.shape
    wb = w.float().reshape(n // GROUP, GROUP, k // GROUP, GROUP)
    scale = _div(wb.abs().amax(dim=(1, 3)).clamp_min(QUANT_EPS), FP8_MAX)
    q = (wb / scale[:, None, :, None]).clamp(-FP8_MAX, FP8_MAX).to(FP8).reshape(n, k)
    return q, scale


# -- backends -------------------------------------------------------------------------------------------------------

_table: list[tuple[int, int, str, int]] = list(DEFAULT_TABLE)
_workspace: dict[int, torch.Tensor] = {}
_sm_count: dict[int, int] = {}
_l2_bytes: dict[int, int] = {}


def table(rows: list[tuple[int, int, str, int]] | None = None) -> list[tuple[int, int, str, int]]:
    """Read or replace the dispatch table: (largest M, largest N, backend, config) rows, searched in order."""
    global _table
    if rows is not None:
        _table = list(rows)
    return _table


def use_backend(name: str) -> None:
    """Route every GEMM to one backend (A/B runs, `dev.runner.cuda.fp8_gemm`): 'auto' restores the measured table;
    'cutlass' uses the original kernel's dispatch (M <= 64 swap-AB, M <= 256 ping-pong, else cooperative);
    'sglang' SGLang's on SM120 (M <= 128 swap-AB 128×32, M <= 256 swap-AB 128×64, else cooperative)."""
    if name == "auto":
        table(list(DEFAULT_TABLE))
    elif name == "cutlass":
        table([(64, BIG, "cutlass", 0), (256, BIG, "cutlass", 1), (BIG, BIG, "cutlass", 2)])
    elif name == "sglang":
        table([(128, BIG, "cutlass", 0), (256, BIG, "cutlass", 3), (BIG, BIG, "cutlass", 2)])
    elif name == "torch":
        table([(BIG, BIG, "torch", 0)])
    else:
        raise ValueError(f"unknown FP8 GEMM backend {name!r}")


def _pick(m: int, n: int) -> tuple[str, int]:
    for max_m, max_n, backend, config in _table:
        if m <= max_m and n <= max_n:
            return backend, config
    return _table[-1][2], _table[-1][3]


def _device_info(dev: torch.device) -> tuple[int, int]:
    i = dev.index if dev.index is not None else torch.cuda.current_device()
    if i not in _sm_count:
        p = torch.cuda.get_device_properties(i)
        _sm_count[i] = p.multi_processor_count
        _l2_bytes[i] = getattr(p, "L2_cache_size", 0)
    return _sm_count[i], _l2_bytes[i]


def cutlass_workspace(dev: torch.device) -> torch.Tensor:
    i = dev.index if dev.index is not None else torch.cuda.current_device()
    ws = _workspace.get(i)
    if ws is None:
        ws = _workspace[i] = torch.empty(CUTLASS_WORKSPACE_BYTES, dtype=torch.uint8, device=dev)
    return ws


def cutlass_available() -> bool:
    try:
        return build.load_cutlass() is not None
    except Exception:  # a failed build (`build.load_cutlass`): unavailable
        return False


def _p(t: torch.Tensor | None) -> ctypes.c_void_p:
    return ctypes.c_void_p(t.data_ptr() if t is not None else None)


def gemm_cutlass(a: QuantAct, w: torch.Tensor, ws: torch.Tensor, config: int,
                 out: torch.Tensor | None = None) -> torch.Tensor:
    """Our CUTLASS library. a.s must be packed (ld == m): the kernel derives its scale layout from m. It launches with
    programmatic dependent launch while our kernels do (`ops.set_pdl`)."""
    from . import ops

    m, k = a.q.shape
    n = w.shape[0]
    if a.s.shape[1] != m:
        raise ValueError(f"the CUTLASS GEMM needs packed scales (ld {a.s.shape[1]} != m {m})")
    lib = build.load_cutlass()
    sm, l2 = _device_info(a.q.device)
    swizzle = 8 if l2 and n * k > l2 else 1  # group tiles for L2 reuse when the weight exceeds the L2
    out = torch.empty((m, n), dtype=torch.bfloat16, device=a.q.device) if out is None else out
    wsp = cutlass_workspace(a.q.device)
    rc = lib.lmk_fp8_gemm_sm120(_p(a.q), _p(w), _p(a.s), _p(ws), _p(out), m, n, k, config, swizzle, sm,
                                a.q.device.index or 0, _p(wsp), wsp.numel(), int(ops._pdl),
                                ctypes.c_void_p(torch.cuda.current_stream(a.q.device).cuda_stream))
    if rc != 0:
        raise RuntimeError(f"fp8 GEMM (CUTLASS config {config}, m {m} n {n} k {k}) failed: {rc}")
    return out


def _padded(a: QuantAct, mult: int = 4) -> QuantAct:
    """`a`, or a copy with m rounded up to `mult` rows and packed scales (cuBLAS reads the scales with 16-byte
    alignment). The padding rows are zero."""
    m, k = a.q.shape
    mp = -(-m // mult) * mult
    if mp == m and a.s.shape[1] == m:
        return a
    q = torch.zeros((mp, k), dtype=FP8, device=a.q.device)
    s = torch.ones((k // GROUP, mp), dtype=torch.float32, device=a.q.device)
    q[:m] = a.q
    s[:, :m] = a.s[:, :m]
    return QuantAct(q, s)


def gemm_torch(a: QuantAct, w: torch.Tensor, ws: torch.Tensor) -> torch.Tensor:
    """torch._scaled_mm with BlockWise1x128 activations ([m, K / 128], M-major) and BlockWise128x128 weights."""
    m = a.m
    ap = _padded(a)
    out = torch._scaled_mm(ap.q, w.t(), scale_a=ap.s.t(), scale_b=ws.t(), out_dtype=torch.bfloat16)
    return out[:m]


def gemm(a: QuantAct, w: torch.Tensor, ws: torch.Tensor, out_dtype: torch.dtype = torch.bfloat16,
         out: torch.Tensor | None = None) -> torch.Tensor:
    """[m, N] = the activations times the block-FP8 weight (w [N, K] e4m3, ws [N / 128, K / 128] fp32). The kernels
    write bf16; the reference (CPU, or `ops.enabled(False)`) any `out_dtype`. `out`: an [m, N] contiguous tensor to
    write into (the rows of a step's FP8 tokens, next to its PRECISE ones: `bf16_rows`)."""
    from . import ops

    if not (a.q.is_cuda and ops.enabled()):
        y = gemm_ref(a, w, ws, out_dtype)
    else:
        if out_dtype != torch.bfloat16:
            raise ValueError(f"the FP8 GEMM kernels write bf16, not {out_dtype}")
        backend, config = _pick(a.m, w.shape[0])
        if backend == "cutlass":
            return gemm_cutlass(a, w, ws, config, out=out)
        y = gemm_torch(a, w, ws)
    if out is None:
        return y
    out.copy_(y)
    return out


# -- BF16 activations (PRECISE rows: the precise prefix) -----------------------------------------------------------
#
# The runner orders a step's PRECISE rows (module docstring) last, so their tokens are the activations' last rows:
# `precise_rows(t, first)` marks tokens [first, t) of a t-token step. While it is active the producers write BF16
# (`activation_quant()` is False) and each `Fp8Linear` splits its input by `bf16_rows`: the first rows quantized
# (`ops.quant_fp8`, which equals every fused producer's quantization bit for bit, the norms, gates and GELU included,
# since each of those quantizes its BF16-rounded output by the same reciprocal) through the FP8 GEMM, the last ones
# through `linear_bf16`. SiLU-and-mul is the one producer that quantizes its fp32 product before rounding, and by
# division (csrc/quant.cu, as vLLM's), so its MLP splits it itself (`SplitAct`). Activations of other sizes in such a
# step (the MTP drafter's chain) keep the FP8 path.

_act_quant = True
_split: tuple[int, int] | None = None  # (t, first): tokens [first, t) of the step's t-token activations take BF16
_scratch: dict[int, torch.Tensor] = {}


@dataclass
class SplitAct:
    """An activation whose first rows are quantized (`head`, a QuantAct) and whose last rows stay BF16 (`tail`): a
    producer whose fused quantization is not its BF16 output's (SiLU-and-mul) in a step with PRECISE rows."""

    head: QuantAct
    tail: torch.Tensor

    @property
    def m(self) -> int:
        return self.head.m + self.tail.shape[0]


def activation_quant(flag: bool | None = None) -> bool:
    """Whether the producing ops quantize their outputs for block-FP8 projections (fused, the default). False with
    every activation in BF16 (`bf16_activations`) and while a step holds PRECISE rows (`precise_rows`): the producers
    write BF16 and each FP8 projection takes it (`bf16_rows`). Read as a forward runs: CUDA graphs keep the mode they
    were captured in."""
    global _act_quant
    if flag is not None:
        _act_quant = flag
    return _act_quant and _split is None


def bf16_rows(m: int) -> int:
    """How many trailing rows of an m-row activation take BF16: all of them with `bf16_activations`, the PRECISE
    rows' when m is the step's token count (`precise_rows`), else none."""
    if not _act_quant:
        return m
    if _split is not None and m == _split[0]:
        return m - _split[1]
    return 0


@contextmanager
def bf16_activations() -> Iterator[None]:
    prev = _act_quant
    activation_quant(False)
    try:
        yield
    finally:
        activation_quant(prev)


@contextmanager
def precise_rows(t: int, first: int) -> Iterator[None]:
    """A step of t tokens whose tokens [first, t) belong to PRECISE rows (the module comment above)."""
    global _split
    if not 0 <= first < t:
        raise ValueError(f"PRECISE tokens [{first}, {t}) of a {t}-token step")
    prev = _split
    _split = (t, first)
    try:
        yield
    finally:
        _split = prev


def reserve_bf16_weight(numel: int, dev: torch.device) -> torch.Tensor:
    """The per-device buffer `linear_bf16` dequantizes each weight into: allocated once, for the largest projection
    (before the runner's memory profile, so the pools leave room for it)."""
    i = dev.index if dev.index is not None else torch.cuda.current_device()
    buf = _scratch.get(i)
    if buf is None or buf.numel() < numel:
        buf = _scratch[i] = torch.empty(numel, dtype=torch.bfloat16, device=dev)
    return buf


def linear_bf16(x: torch.Tensor, w: torch.Tensor, ws: torch.Tensor, out_dtype: torch.dtype = torch.bfloat16,
                out: torch.Tensor | None = None) -> torch.Tensor:
    """x [m, K] in BF16 times the block-FP8 weight dequantized to BF16 (w [N, K] e4m3, ws [N / 128, K / 128]): a BF16
    GEMM with fp32 accumulation. The reference (CPU, or `ops.enabled(False)`) multiplies in fp32. `out`: an [m, N]
    contiguous tensor to write into."""
    from . import ops

    if not (x.is_cuda and ops.enabled()):
        y = (x.float() @ dequant_weight(w, ws).T).to(out_dtype)
        if out is None:
            return y
        out.copy_(y)
        return out
    n, k = w.shape
    buf = reserve_bf16_weight(n * k, w.device)
    wb = ops.dequant_fp8(w, ws, out=buf[:n * k].view(n, k))
    if out is None:
        return torch.nn.functional.linear(x, wb)
    return torch.mm(x, wb.t(), out=out)


def describe() -> dict[str, Any]:
    return {"table": [{"max_m": m, "max_n": n, "backend": b, "config": c} for m, n, b, c in _table],
            "cutlass": build.stats_cutlass.get("state", "not loaded")}
