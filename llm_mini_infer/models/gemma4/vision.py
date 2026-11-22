"""The Gemma 4 vision tower (`model.vision_tower.*`) and its embedder (`model.embed_vision.*`) as plain `nn.Module`s,
numerics after transformers' `Gemma4VisionModel` and `Gemma4MultimodalEmbedder`, the reference the embeddings are
tested against.

- **Pixels to patches, on the device** (`patches`): the image pool's uint8 pixels [3, H, W] rescaled exactly as HF's
  processor does, cut into 16 × 16 patches flattened as (row, column, channel). The patches go in **pooling order**:
  each 3 × 3 block of the patch grid (row-major over the blocks) is nine consecutive patches, so the pooling is a
  reshape. Every op before the pooling is per patch, and attention within an image does not depend on the order, so
  the order changes nothing but the pooling's summation order.
- **Patch embedding** (`embed`): `input_proj` (768 → 1152) plus the learned position embeddings of the patch's column
  and row, summed in BF16 as HF does.
- **27 encoder layers** (pre-norm sandwich, as the text model's): q, k and v from one fused projection, then their norms
  and the 2-D RoPE (θ 100: the first half of each 72-dim head by the patch's column, the second by its row) in one
  kernel (`vision_qkv`); attention bidirectional within each image, scale 1.0; a GELU-tanh gated MLP (4304). The norms,
  RoPE and GELU compute in fp32 inside each op and round once where a tensor is stored (`vision_qkv`, and the text
  model's fused ops for the rest, `kernels.gemma_ops`), where HF's BF16 path rounds after every op: closer to HF's fp32
  than HF's own BF16.
- **Pooling and the embedder** (`pool_project`): each 3 × 3 block's mean, scaled and standardised, then an RMSNorm
  without weight and `embedding_projection` (1152 → the text model's 5376): one embedding per soft token, which replaces
  the token's input embedding as it is (unscaled, as HF's `masked_scatter`).

**Attention backends:** Qwen's tower's two (`models.qwen3_5.vision.TorchVisionAttention`, the reference, and
`kernels.flashinfer.FlashInferVisionAttention`; `runner.cuda.images` picks), with this tower's scale of 1.0.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ...kernels import build
from ...kernels import gemma_ops as gops
from ...kernels.ops import _check, _p, _stream, use_kernels
from ..common import Coverage, rope_inv_freq, to_device_int32

GRID_CACHE = 256  # patch grids whose per-patch metadata stays on the device


@dataclass(frozen=True)
class GemmaVisionConfig:
    depth: int
    hidden: int
    heads: int
    head_dim: int
    intermediate: int
    patch: int
    pooling: int
    pos_size: int  # entries of each axis' position-embedding table
    rope_theta: float
    eps: float
    standardize: bool
    out_hidden: int  # the text model's hidden size
    rescale: float
    normalize: bool
    mean: tuple[float, ...]
    std: tuple[float, ...]

    @property
    def merge(self) -> int:
        """Patches per soft token along each side (the runner's encoder interface: tokens = patches / merge²)."""
        return self.pooling

    @property
    def max_grid(self) -> int:
        """The longest side of a patch grid the tower takes: the position table's entries (`GemmaVisionEncoder.meta`;
        the runner's synthetic images stay within it)."""
        return self.pos_size

    @property
    def patch_dim(self) -> int:
        return 3 * self.patch * self.patch

    @property
    def scale(self) -> float:
        return 1.0  # q_norm and k_norm take the role of 1/sqrt(d) (HF's `self.scaling = 1.0`)

    @classmethod
    def from_dir(cls, model_dir: str | Path) -> GemmaVisionConfig:
        model_dir = Path(model_dir)
        hf = json.loads((model_dir / "config.json").read_text())
        vc = hf["vision_config"]
        tc = hf.get("text_config", hf)
        path = model_dir / "processor_config.json"
        proc = json.loads(path.read_text()) if path.exists() else {}
        ip = proc.get("image_processor", proc)
        rope = vc.get("rope_parameters") or {}
        if vc.get("use_clipped_linears"):
            raise NotImplementedError("clipped linears in the vision tower (use_clipped_linears)")
        if vc.get("hidden_activation", "gelu_pytorch_tanh") != "gelu_pytorch_tanh":
            raise NotImplementedError(f"vision hidden_activation {vc['hidden_activation']!r}")
        heads = int(vc["num_attention_heads"])
        if int(vc.get("num_key_value_heads", heads)) != heads:
            raise NotImplementedError("grouped-query attention in the vision tower")
        head_dim = int(vc.get("head_dim") or vc["hidden_size"] // heads)
        if head_dim % 4:
            raise ValueError(f"the 2-D RoPE needs a head dim divisible by 4, got {head_dim}")
        return cls(depth=int(vc["num_hidden_layers"]), hidden=int(vc["hidden_size"]), heads=heads, head_dim=head_dim,
                   intermediate=int(vc["intermediate_size"]), patch=int(vc.get("patch_size", 16)),
                   pooling=int(vc.get("pooling_kernel_size", 3)),
                   pos_size=int(vc.get("position_embedding_size", 10240)),
                   rope_theta=float(rope.get("rope_theta", 100.0)), eps=float(vc.get("rms_norm_eps", 1e-6)),
                   standardize=bool(vc.get("standardize", False)), out_hidden=int(tc["hidden_size"]),
                   rescale=float(ip.get("rescale_factor", 1 / 255)), normalize=bool(ip.get("do_normalize", False)),
                   mean=tuple(ip.get("image_mean", (0.0, 0.0, 0.0))), std=tuple(ip.get("image_std", (1.0, 1.0, 1.0))))


# -- per-image metadata ------------------------------------------------------------------------------------------------


def pool_order(h: int, w: int, k: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """(column, row) of each patch of an h × w patch grid in pooling order: k × k blocks row-major, each block's patches
    row-major."""
    ys, xs = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    shape = (h // k, k, w // k, k)
    return xs.reshape(shape).transpose(1, 2).reshape(-1), ys.reshape(shape).transpose(1, 2).reshape(-1)


@dataclass
class GemmaEncodeMeta:
    """One encoder call: its images' patch offsets and the per-patch tensors the tower reads (`PER_PATCH`)."""

    n: int  # patches
    cu: torch.Tensor  # int32 [images + 1]: patch offsets (cu_seqlens)
    cu_host: list[int]
    cos: torch.Tensor  # fp32 [N, head_dim / 2]: the column's angles, then the row's
    sin: torch.Tensor
    pos_x: torch.Tensor  # int64 [N]: the patch's column and row (position-embedding indices)
    pos_y: torch.Tensor
    attn: Any = None  # the attention backend's plan

    PER_PATCH = ("cos", "sin", "pos_x", "pos_y")  # what `runner.cuda.images.VisionGraphs` keeps in static buffers


# -- the modules -------------------------------------------------------------------------------------------------------


def rope_2d(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """HF's `apply_multidimensional_rope` with two axes, in fp32: x [N, H, D]; cos/sin [N, D / 2] (the column's D / 4
    angles, then the row's). Each half of the head is a NeoX rotation of its own (pairs d and d + D / 4)."""
    half = x.shape[-1] // 2
    q = half // 2
    out = []
    for a, (lo, hi) in enumerate(((0, half), (half, 2 * half))):
        part = x[..., lo:hi]
        c = cos[:, None, a * q:(a + 1) * q]
        s = sin[:, None, a * q:(a + 1) * q]
        x1, x2 = part[..., :q], part[..., q:]
        out += [x1 * c - x2 * s, x2 * c + x1 * s]
    return torch.cat(out, dim=-1)


class GemmaVisionAttention(nn.Module):
    def __init__(self, cfg: GemmaVisionConfig, dtype: torch.dtype):
        super().__init__()
        self.heads, self.d, self.eps = cfg.heads, cfg.head_dim, cfg.eps
        self.qkv = nn.Linear(cfg.hidden, 3 * cfg.hidden, bias=False, dtype=dtype)  # q, k, v fused at load
        self.o_proj = nn.Linear(cfg.hidden, cfg.hidden, bias=False, dtype=dtype)
        self.q_norm = nn.Parameter(torch.ones(cfg.head_dim, dtype=dtype), requires_grad=False)
        self.k_norm = nn.Parameter(torch.ones(cfg.head_dim, dtype=dtype), requires_grad=False)

    def forward(self, x: torch.Tensor, meta: GemmaEncodeMeta, backend: Any) -> torch.Tensor:
        n, h, d = x.shape[0], self.heads, self.d
        qkv = F.linear(x, self.qkv.weight).view(n, 3, h, d)
        pad = getattr(backend, "pad", d)
        q, k, v = vision_qkv(qkv, self.q_norm, self.k_norm, self.eps, meta.cos, meta.sin, pad)
        o = backend.run(meta, q, k, v)
        if pad != d:
            o = o[..., :d]
        return F.linear(o.reshape(n, h * d), self.o_proj.weight)


def vision_qkv(qkv: torch.Tensor, q_w: torch.Tensor, k_w: torch.Tensor, eps: float, cos: torch.Tensor,
               sin: torch.Tensor, pad: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """q = rope(q_norm(q)), k = rope(k_norm(k)), v = v_norm(v) (no weight), each fp32 inside and rounded once;
    zero-padded to `pad` dims for the attention kernel. qkv [N, 3, H, D] in the model dtype. The torch ops below define
    it (57 kernels a layer on CUDA) and run on CPU, without our kernels (`fused_ops` off, no library) and for the shapes
    the kernel does not take; otherwise one kernel computes it (`csrc/vision.cu` g4_vision_qkv), bit for bit: it sums
    the squares in the order of the pinned torch's mean over a last dimension below 128 and rounds where these ops round
    (tests/kernels/test_kernels.py, `torch.equal`), for a head dim that is a multiple of 4, at most 96 (Gemma 4's 72),
    with at least 16 (patch, head) rows or a head dim of at most 64: with fewer rows torch's mean widens its block to 64
    lanes and adds element l + 64 before l + 32, so these ops run there (never in Gemma 4's tower: 16 heads)."""
    n, _, h, d = qkv.shape
    if use_kernels(qkv) and qkv.dtype == torch.bfloat16 and d % 4 == 0 and d <= 96 and pad >= d and \
            (d <= 64 or n * h >= 16) and qkv.stride()[1:] == (h * d, d, 1) and \
            cos.dtype == sin.dtype == torch.float32 and cos.is_contiguous() and sin.is_contiguous() and \
            q_w.is_contiguous() and k_w.is_contiguous():
        out = [torch.empty((n, h, pad), dtype=qkv.dtype, device=qkv.device) for _ in range(3)]
        factor = float(np.float32(n * h) / np.float32(n * h * d))  # torch's mean: outputs over elements, in fp32
        _check(build.load().lmk_g4_vision_qkv(_p(qkv), qkv.stride(0), _p(q_w), _p(k_w), _p(cos), _p(sin), eps, factor,
                                              *(_p(t) for t in out), n, h, d, pad, _stream(qkv)), "g4_vision_qkv")
        return out[0], out[1], out[2]
    q, k, v = qkv.unbind(1)
    dt = qkv.dtype
    q = rope_2d(gops._rms(q.float(), eps) * q_w.float(), cos, sin).to(dt)
    k = rope_2d(gops._rms(k.float(), eps) * k_w.float(), cos, sin).to(dt)
    v = gops._rms(v.float(), eps).to(dt)
    if pad != q.shape[-1]:
        extra = (0, pad - q.shape[-1])
        q, k, v = F.pad(q, extra), F.pad(k, extra), F.pad(v, extra)
    return q, k, v


class GemmaVisionLayer(nn.Module):
    def __init__(self, cfg: GemmaVisionConfig, dtype: torch.dtype):
        super().__init__()
        self.self_attn = GemmaVisionAttention(cfg, dtype)
        self.gate_up_proj = nn.Linear(cfg.hidden, 2 * cfg.intermediate, bias=False, dtype=dtype)  # fused at load
        self.down_proj = nn.Linear(cfg.intermediate, cfg.hidden, bias=False, dtype=dtype)
        ones = lambda: nn.Parameter(torch.ones(cfg.hidden, dtype=dtype), requires_grad=False)  # noqa: E731
        self.input_layernorm, self.post_attention_layernorm = ones(), ones()
        self.pre_feedforward_layernorm, self.post_feedforward_layernorm = ones(), ones()
        self.eps = cfg.eps

    def forward(self, x: torch.Tensor, residual: torch.Tensor, meta: GemmaEncodeMeta,
                backend: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """x: this layer's input, already input-normed; residual: the stream before it. Returns (the MLP's output before
        its post norm, the stream after the attention block), as the text model's layers (`models.gemma4.model`)."""
        a = self.self_attn(x, meta, backend)
        residual, h = gops.sandwich(a, residual, self.post_attention_layernorm, self.pre_feedforward_layernorm,
                                    self.eps)
        return F.linear(gops.gelu_tanh_mul(F.linear(h, self.gate_up_proj.weight)), self.down_proj.weight), residual


def vision_buffers(cfg: GemmaVisionConfig) -> dict[str, torch.Tensor]:
    """The RoPE's inverse frequencies (HF's `compute_axial_rope_parameters`: head_dim / 4 of them, CPU, fp32)."""
    return {"inv_freq": rope_inv_freq(cfg.rope_theta, cfg.head_dim // 2)}


class GemmaVisionEncoder(nn.Module):
    """uint8 pixels to soft-token embeddings in the text model's space (module docstring): `patches`, `meta`, then
    `forward`."""

    def __init__(self, cfg: GemmaVisionConfig, dtype: torch.dtype):
        super().__init__()
        self.cfg = cfg
        self.dtype = dtype
        self.input_proj = nn.Linear(cfg.patch_dim, cfg.hidden, bias=False, dtype=dtype)
        self.position_embedding_table = nn.Parameter(torch.zeros(2, cfg.pos_size, cfg.hidden, dtype=dtype),
                                                     requires_grad=False)
        self.layers = nn.ModuleList([GemmaVisionLayer(cfg, dtype) for _ in range(cfg.depth)])
        self.std_bias = nn.Parameter(torch.zeros(cfg.hidden, dtype=dtype), requires_grad=False)
        self.std_scale = nn.Parameter(torch.ones(cfg.hidden, dtype=dtype), requires_grad=False)
        self.embedding_projection = nn.Linear(cfg.hidden, cfg.out_hidden, bias=False, dtype=dtype)
        for name, value in vision_buffers(cfg).items():
            self.register_buffer(name, value, persistent=False)
        self._grid_cache: dict[tuple, tuple[torch.Tensor, ...]] = {}

    # -- pixels --------------------------------------------------------------------------------------------------

    def patches(self, px: torch.Tensor) -> torch.Tensor:
        """uint8 [3, H, W] (H, W multiples of patch × pooling) -> fp32 [N, 3 · patch²] in pooling order: HF's
        `pixel_values` rows (rescaled as its processor does), reordered."""
        c = self.cfg
        x = px.float()
        if c.normalize:  # HF's fused rescale and normalisation, a tensor division (torchvision's normalize)
            r = 1.0 / c.rescale
            mean = torch.tensor(c.mean, dtype=torch.float32, device=x.device) * r
            std = torch.tensor(c.std, dtype=torch.float32, device=x.device) * r
            x = (x - mean.view(3, 1, 1)) / std.view(3, 1, 1)
        else:
            x = x * c.rescale
        p, k = c.patch, c.pooling
        _, hgt, wid = x.shape
        gh, gw = hgt // p, wid // p
        x = x.reshape(3, gh // k, k, p, gw // k, k, p).permute(1, 4, 2, 5, 3, 6, 0)  # (by, bx, dy, dx, py, px, c)
        return x.reshape(gh * gw, p * p * 3)

    # -- metadata ------------------------------------------------------------------------------------------------

    def grid_meta(self, h: int, w: int, device: torch.device) -> tuple[torch.Tensor, ...]:
        """(cos, sin, column, row) of one h × w patch grid in pooling order, cached by grid."""
        key = (h, w, str(device))
        got = self._grid_cache.get(key)
        if got is None:
            xs, ys = pool_order(h, w, self.cfg.pooling, device)
            inv = self.inv_freq.to(device)
            freqs = torch.cat([xs[:, None].float() * inv, ys[:, None].float() * inv], dim=-1)  # HF: positions @ inv
            got = (freqs.cos(), freqs.sin(), xs, ys)
            if len(self._grid_cache) >= GRID_CACHE:
                self._grid_cache.pop(next(iter(self._grid_cache)))
            self._grid_cache[key] = got
        return got

    def meta(self, grids: list[tuple[int, int, int]], device: torch.device) -> GemmaEncodeMeta:
        """A batch of images (grids (t, h, w) in patches; t must be 1, and no side past the position table: a request's
        image never is, its longest side pooling x images.max_soft_tokens patches at most)."""
        parts, cu = [], [0]
        k, pos = self.cfg.pooling, self.cfg.pos_size
        for t, h, w in grids:
            if t != 1:
                raise NotImplementedError("video (temporal grids > 1) is not supported")
            if h % k or w % k:
                raise ValueError(f"a patch grid of {h} x {w} does not pool by {k}")
            if h > pos or w > pos:
                raise ValueError(f"a patch grid of {h} x {w} is past the position table ({pos} entries); lower "
                                 "images.max_soft_tokens or this image's tokens")
            parts.append(self.grid_meta(h, w, device))
            cu.append(cu[-1] + h * w)
        cos, sin, xs, ys = (x[0] if len(parts) == 1 else torch.cat(list(x)) for x in zip(*parts))
        return GemmaEncodeMeta(n=cu[-1], cu=to_device_int32(cu, device), cu_host=cu, cos=cos, sin=sin, pos_x=xs,
                               pos_y=ys)

    # -- the tower -----------------------------------------------------------------------------------------------

    def embed(self, patches: torch.Tensor, meta: GemmaEncodeMeta) -> torch.Tensor:
        """The patch embedding plus the position embeddings (HF's `Gemma4VisionPatchEmbedder`, BF16 adds)."""
        x = F.linear((2 * (patches - 0.5)).to(self.dtype), self.input_proj.weight)
        t = self.position_embedding_table
        return x + (t[0].index_select(0, meta.pos_x) + t[1].index_select(0, meta.pos_y))

    def forward(self, patches: torch.Tensor, meta: GemmaEncodeMeta, backend: Any) -> torch.Tensor:
        """fp32 patches [N, patch_dim] (pooling order) -> soft-token embeddings [N / pooling², out_hidden]."""
        c = self.cfg
        residual = self.embed(patches, meta)
        layers = self.layers
        x = gops.rmsnorm(residual, layers[0].input_layernorm, c.eps)
        last = len(layers) - 1
        for i, layer in enumerate(layers):
            m, residual = layer(x, residual, meta, backend)
            # the residual after the MLP block and the next layer's input norm; the pooler reads the last residual
            # (no final norm: the last call's second output is not used)
            post = layer.post_feedforward_layernorm
            residual, x = gops.sandwich(m, residual, post, layers[i + 1].input_layernorm if i < last else post, c.eps)
        return self.pool_project(residual)

    def pool_project(self, h: torch.Tensor) -> torch.Tensor:
        """HF's pooler (each block's fp32 mean, rounded, × sqrt(hidden) in fp32), the standardisation (fp32), the
        embedder's weightless RMSNorm and projection. h [N, hidden] in pooling order."""
        c = self.cfg
        k2 = c.pooling ** 2
        pooled = (h.float().view(-1, k2, c.hidden).sum(1) * (1.0 / k2)).to(self.dtype)
        y = pooled.float() * (c.hidden ** 0.5)
        if c.standardize:
            y = (y - self.std_bias.float()) * self.std_scale.float()
        y = gops._rms(y.to(self.dtype).float(), c.eps).to(self.dtype)
        return F.linear(y, self.embedding_projection.weight)


# -- weights -----------------------------------------------------------------------------------------------------------

_LAYER = "model.vision_tower.encoder.layers."
_QKV = {"self_attn.q_proj.linear.weight": 0, "self_attn.k_proj.linear.weight": 1, "self_attn.v_proj.linear.weight": 2}
_GU = {"mlp.gate_proj.linear.weight": 0, "mlp.up_proj.linear.weight": 1}
_SIMPLE = {"self_attn.o_proj.linear.weight": "self_attn.o_proj.weight", "mlp.down_proj.linear.weight":
           "down_proj.weight", "self_attn.q_norm.weight": "self_attn.q_norm", "self_attn.k_norm.weight":
           "self_attn.k_norm", "input_layernorm.weight": "input_layernorm", "post_attention_layernorm.weight":
           "post_attention_layernorm", "pre_feedforward_layernorm.weight": "pre_feedforward_layernorm",
           "post_feedforward_layernorm.weight": "post_feedforward_layernorm"}
_TOP = {"model.vision_tower.patch_embedder.input_proj.weight": "input_proj.weight",
        "model.vision_tower.patch_embedder.position_embedding_table": "position_embedding_table",
        "model.vision_tower.std_bias": "std_bias", "model.vision_tower.std_scale": "std_scale",
        "model.embed_vision.embedding_projection.weight": "embedding_projection.weight"}


_VISION = ("model.vision_tower.", "model.embed_vision.", "vision_tower.", "embed_vision.")  # as gemma4._SKIP_PREFIXES


def vision_weight_name(name: str, hidden: int, intermediate: int) -> tuple[str, int] | None:
    """A checkpoint name -> (our parameter, its first row in a fused one), or None outside the tower and the embedder
    (`_VISION`); a name inside them that is not ours raises (the Qwen tower's loader too)."""
    if name in _TOP:
        return _TOP[name], 0
    if not name.startswith(_LAYER):
        if name.startswith(_VISION):
            raise ValueError(f"unexpected vision weight {name}")
        return None
    i, _, rest = name[len(_LAYER):].partition(".")
    if rest in _QKV:
        return f"layers.{i}.self_attn.qkv.weight", _QKV[rest] * hidden
    if rest in _GU:
        return f"layers.{i}.gate_up_proj.weight", _GU[rest] * intermediate
    if rest in _SIMPLE:
        return f"layers.{i}.{_SIMPLE[rest]}", 0
    raise ValueError(f"unexpected vision weight {name}")


@torch.inference_mode()
def load_gemma_vision(model_dir: str | Path, device: torch.device, dtype: torch.dtype) -> GemmaVisionEncoder:
    """Build on the meta device, allocate on `device`, load the tower and the embedder (BF16 in every Gemma 4
    checkpoint)."""
    from safetensors import safe_open

    from .model import checkpoint_files

    cfg = GemmaVisionConfig.from_dir(model_dir)
    with torch.device("meta"):
        enc = GemmaVisionEncoder(cfg, dtype)
    enc = enc.to_empty(device=device).requires_grad_(False)
    for name, value in vision_buffers(cfg).items():  # to_empty leaves the buffers uninitialised
        setattr(enc, name, value.to(device))
    params = dict(enc.named_parameters())
    cover = Coverage()
    dev = str(device) if device.type == "cpu" else f"cuda:{device.index or 0}"
    for path in checkpoint_files(Path(model_dir)):
        with safe_open(str(path), framework="pt", device=dev) as f:
            for name in f.keys():  # noqa: SIM118 - safe_open is not a mapping
                target = vision_weight_name(name, cfg.hidden, cfg.intermediate)
                if target is None:
                    continue
                pname, row0 = target
                p = params[pname]
                t = f.get_tensor(name)
                dst = p.data.narrow(0, row0, t.shape[0]) if p.dim() > 1 and pname.endswith(("qkv.weight",
                                                                                            "gate_up_proj.weight")) \
                    else p.data
                if dst.shape != t.shape:
                    raise ValueError(f"{name} {tuple(t.shape)} does not fit {pname}[{row0}:] {tuple(dst.shape)}")
                dst.copy_(t.to(p.dtype))
                cover.add(pname, row0, t.shape[0])
    missing = sorted(cover.incomplete(params))
    if missing:
        raise ValueError(f"vision weights missing or partial in the checkpoint: {missing[:6]}")
    return enc.eval()
