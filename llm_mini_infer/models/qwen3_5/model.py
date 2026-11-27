"""Qwen3.5 / Qwen3.8 dense text model (`Qwen3_5ForConditionalGeneration`, text only) as plain `nn.Module`s.

Its references are HF transformers' `modeling_qwen3_5.py` and vLLM's `qwen3_5.py`, `qwen3_next.py` and
`qwen_gdn_linear_attn.py`:
- every 4th layer is **gated full attention**: QK-norm, partial NeoX RoPE (`partial_rotary_factor` of each head) and
  an output gate `o * sigmoid(gate)`; q and the gate interleave per head in `q_proj`;
- the other layers are **Gated DeltaNet** (GDN): a causal depthwise conv over [q|k|v] with SiLU, L2-normalised q/k,
  decay `g = -exp(A_log) * softplus(a + dt_bias)`, `beta = sigmoid(b)`, the gated delta rule, and a gated RMSNorm
  `w * norm(o) * silu(z)` on its output;
- Gemma-style RMSNorm `x / rms(x) * (1 + w)` everywhere else; the residual stream is kept in the model dtype.

Batch metadata comes from a per-step `ForwardContext`: the model sees one flat batch of tokens and calls
`ctx.qk_rope_store(...)`, `ctx.attention(...)` and `ctx.gdn(...)`, whose backends (FlashInfer and our kernels on CUDA,
torch references on CPU; `backend.py`) read and write the paged KV pool and the recurrent-state pool. Norms and
activations go through `kernels.ops`: fused CUDA kernels with the rounding points below, or the torch reference.

**Numerics** follow vLLM's CUDA path for this model (docs/vllm-compatibility.md, "Numerics"), so
that parity is limited by batch noise and not by rounding choices:
- RMSNorms reduce in fp32 and round once; the fused residual add normalises the fp32 sum and keeps the rounded sum
  as the residual (a native `fused_add_rms_norm`, which the Gemma norm uses because its weight is fp32);
- q/k are rounded after the QK-norm, and RoPE runs in fp32 against a cos/sin cache in the model dtype, rounding once
  (as vLLM's fused qk-norm-rope-gate kernel);
- the attention gate, SiLU-and-mul and the gated RMSNorm compute in fp32 and round once (what Inductor emits for the
  native ops);
- the conv products are rounded to the model dtype and summed in fp32; the gating and the recurrence are fp32. The
  decode step is the vendored fused decode kernel (`kernels/csrc/gdn_decode.cu`), which rounds the recurrence output to
  the model dtype before the gated norm, as the prefill path does.

**Positions.** Qwen's interleaved M-RoPE takes (t, h, w) positions; it collapses to ordinary RoPE when the three are
equal, as for every text token. RoPE reads `ForwardContext.meta.rope` when set (with images: every token's RoPE
position, [3, T], or [T] when the components are equal; it differs from the sequence position at image tokens and the
text after them; `ops.qk_norm_rope`), and the sequence positions otherwise. The KV slots always come from the sequence
positions.

**Inputs.** `forward(..., embeds=)` takes the input embeddings ready-made: the runner puts the image encoder's rows at
image placeholders (`models.qwen3_5.vision`).

**Weights:** text weights and the MTP layer; `visual.*` is loaded by `models.qwen3_5.vision`. Projections are fused
at load: q(+gate), k, v → `qkv_proj`; the GDN [q|k|v] and z → `in_proj_qkvz`; b, a →
`in_proj_ba`; gate, up → `gate_up_proj`.

**MTP** (`mtp.*`; vLLM's `qwen3_5_mtp.py`). One extra decoder layer predicts the token after next: at position i it
takes the target's final hidden state h_i (after the last norm) and the token at i + 1,
`fc(cat(pre_fc_norm_embedding(embed(t)), pre_fc_norm_hidden(h)))` (2 × hidden → hidden), then a full-attention decoder
layer whose KV is the pool's last attention layer, and `norm`; the embedding and lm_head are the target's. Its
projections are block-FP8 where the checkpoint
says so (Qwen3.8-27B-FP8: the layer's projections are, `fc` is not).

**Block-FP8 checkpoints** (`quantization_config` with `quant_method` fp8, `weight_block_size` [128, 128], dynamic
activations; Qwen3.8-27B-FP8): the projections the checkpoint quantizes are `Fp8Linear`s, which keep the e4m3 weight
[N, K] and its FP32 block scales [N / 128, K / 128] (`weight_scale_inv`) as stored, the layout the GEMM reads
(`kernels.fp8`). Fused shards concatenate along N, block rows with them; every shard must be a multiple of 128 rows.
Embeddings, lm_head, norms, the conv, dt_bias and in_proj_ba stay BF16 and A_log fp32, unquantized as the checkpoint
stores them (`modules_to_not_convert`). The activations of an FP8 projection are quantized by the op that produces
them: the input and post-attention RMSNorms (with the residual add), the attention output gate, the GDN output norm
(also inside the GDN decode kernel) and SiLU-and-mul, each one kernel (`kernels.ops`); the GDN layers' input norm also
writes its BF16 output for in_proj_ba. PRECISE rows: `kernels.fp8`.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import nn

from ...kernels import fp8, ops
from ...kernels.fp8 import QuantAct
from ..common import Coverage
from ..fp8 import Fp8Linear, QuantConfig, check_tensor, fused_fp8

if TYPE_CHECKING:
    from .backend import ForwardContext

log = logging.getLogger(__name__)

FULL, LINEAR = "full_attention", "linear_attention"


# Our fused projections and the checkpoint modules they are made of.
_FUSED_MODULES = {
    "self_attn.qkv_proj": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
    "linear_attn.in_proj_qkvz": ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z"),
    "linear_attn.in_proj_ba": ("linear_attn.in_proj_b", "linear_attn.in_proj_a"),
    "mlp.gate_up_proj": ("mlp.gate_proj", "mlp.up_proj"),
}


@dataclass(frozen=True)
class Qwen35Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    layer_types: tuple[str, ...]
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rotary_dim: int
    rope_theta: float
    rms_norm_eps: float
    lin_k_heads: int
    lin_v_heads: int
    lin_k_dim: int
    lin_v_dim: int
    conv_kernel: int
    tie_word_embeddings: bool
    max_position_embeddings: int
    attn_output_gate: bool = True
    mtp_layers: int = 0  # MTP layers in the checkpoint (mtp_num_hidden_layers); the runner loads one
    mrope_section: tuple[int, ...] = ()  # interleaved M-RoPE's (t, h, w) sections (images)
    quant: QuantConfig | None = None  # block-FP8 checkpoints

    @property
    def num_layers(self) -> int:
        return len(self.layer_types)

    @property
    def attn_layers(self) -> list[int]:
        return [i for i, t in enumerate(self.layer_types) if t == FULL]

    @property
    def gdn_layers(self) -> list[int]:
        return [i for i, t in enumerate(self.layer_types) if t == LINEAR]

    @property
    def key_dim(self) -> int:
        return self.lin_k_heads * self.lin_k_dim

    @property
    def value_dim(self) -> int:
        return self.lin_v_heads * self.lin_v_dim

    @property
    def conv_dim(self) -> int:
        return 2 * self.key_dim + self.value_dim

    def fp8_module(self, name: str) -> bool:
        """Whether our projection `name` (e.g. `layers.3.self_attn.qkv_proj`) is block-FP8 in this checkpoint
        (`fused_fp8`)."""
        q = self.quant
        if q is None:
            return False
        if name == "mtp.fc":
            return not q.skips(name, [p + name for p in ("", "model.")])
        m = _LAYER.fullmatch(name)
        if m is None:
            return False  # embeddings and lm_head stay BF16
        pre = m.group(1) or ""
        parts = [f"{pre}layers.{m.group(2)}.{p}" for p in _FUSED_MODULES.get(m.group(3), (m.group(3),))]
        return fused_fp8(q, name, parts, ("",) if pre else _STRIP_PREFIXES)

    @classmethod
    def from_hf(cls, hf: dict[str, Any]) -> Qwen35Config:
        """From an HF `config.json`: `Qwen3_5ForConditionalGeneration` (with `text_config`) or the text-only
        `Qwen3_5ForCausalLM` (flat)."""
        quant = QuantConfig.from_hf(hf["quantization_config"], strip=_STRIP_PREFIXES) \
            if hf.get("quantization_config") else None
        tc = hf.get("text_config", hf)
        model_type = tc.get("model_type", hf.get("model_type", ""))
        if model_type not in ("qwen3_5_text", "qwen3_5"):
            raise NotImplementedError(f"model_type {model_type!r} is not Qwen3.5/3.8 dense (qwen3_5_text)")
        if tc.get("num_experts"):
            raise NotImplementedError("Qwen3.5 MoE is not supported yet")
        if tc.get("hidden_act", "silu") != "silu":
            raise NotImplementedError(f"hidden_act {tc['hidden_act']!r}")
        if tc.get("layer_scale"):
            raise NotImplementedError("layer_scale")
        n = int(tc["num_hidden_layers"])
        types = tc.get("layer_types")
        if not types:
            interval = int(tc.get("full_attention_interval", 4))
            types = [FULL if (i + 1) % interval == 0 else LINEAR for i in range(n)]
        if len(types) != n or set(types) - {FULL, LINEAR}:
            raise ValueError(f"bad layer_types {types}")
        rope = dict(tc.get("rope_parameters") or {})
        theta = float(rope.get("rope_theta", tc.get("rope_theta", 10_000.0)))
        partial = float(rope.get("partial_rotary_factor", tc.get("partial_rotary_factor", 0.25)))
        if rope.get("rope_type", "default") != "default":
            raise NotImplementedError(f"rope_type {rope.get('rope_type')!r}")
        head_dim = int(tc.get("head_dim") or int(tc["hidden_size"]) // int(tc["num_attention_heads"]))
        rotary_dim = int(head_dim * partial)
        if rotary_dim % 2 or rotary_dim > head_dim:
            raise ValueError(f"rotary_dim {rotary_dim}")
        conv = int(tc.get("linear_conv_kernel_dim", 4))
        kh, vh = int(tc["linear_num_key_heads"]), int(tc["linear_num_value_heads"])
        if vh % kh:
            raise ValueError("linear_num_value_heads must be a multiple of linear_num_key_heads")
        tie = bool(tc.get("tie_word_embeddings", hf.get("tie_word_embeddings", False)))
        return cls(vocab_size=int(tc["vocab_size"]), hidden_size=int(tc["hidden_size"]),
                   intermediate_size=int(tc["intermediate_size"]), layer_types=tuple(types),
                   num_heads=int(tc["num_attention_heads"]), num_kv_heads=int(tc["num_key_value_heads"]),
                   head_dim=head_dim, rotary_dim=rotary_dim, rope_theta=theta,
                   rms_norm_eps=float(tc.get("rms_norm_eps", 1e-6)), lin_k_heads=kh, lin_v_heads=vh,
                   lin_k_dim=int(tc["linear_key_head_dim"]), lin_v_dim=int(tc["linear_value_head_dim"]),
                   conv_kernel=conv, tie_word_embeddings=tie,
                   max_position_embeddings=int(tc.get("max_position_embeddings", 32768)),
                   attn_output_gate=bool(tc.get("attn_output_gate", True)),
                   mtp_layers=int(tc.get("mtp_num_hidden_layers", 0) or 0),
                   mrope_section=tuple(rope.get("mrope_section") or ()), quant=quant)

    @classmethod
    def from_dir(cls, model_dir: str | Path) -> Qwen35Config:
        return cls.from_hf(json.loads((Path(model_dir) / "config.json").read_text()))


# -- layers ------------------------------------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Gemma-style RMSNorm, `x / rms(x) * (1 + w)`, reduced in fp32 and rounded once."""

    def __init__(self, dim: int, eps: float, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim, dtype=dtype), requires_grad=False)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ops.rmsnorm(x, self.weight, self.eps)

    def forward_add(self, x: torch.Tensor, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Fused residual add: normalises the fp32 sum; the new residual is the sum rounded to the model dtype."""
        return ops.rmsnorm(x, self.weight, self.eps, residual)

    def apply(self, x: torch.Tensor, residual: torch.Tensor | None,
              quant: str | None) -> tuple[torch.Tensor | QuantAct, torch.Tensor]:
        """The norm, with the residual add when `residual` is given: (its output, the new residual). `quant` "fp8"
        quantizes the output for a block-FP8 projection in the same kernel; "fp8+bf16" keeps the BF16 output too
        (`QuantAct.bf16`)."""
        if quant is None or not fp8.activation_quant():
            return (self(x), x) if residual is None else self.forward_add(x, residual)
        a, res = ops.rmsnorm_quant(x, self.weight, self.eps, residual, bf16=quant == "fp8+bf16")
        return a, (x if residual is None else res)


def make_linear(cfg: Qwen35Config, name: str, in_features: int, out_features: int,
                dtype: torch.dtype) -> nn.Module:
    """`Fp8Linear` where the checkpoint quantizes the projection `name`, else a BF16 `nn.Linear`."""
    if cfg.fp8_module(name):
        return Fp8Linear(in_features, out_features, dtype)
    return nn.Linear(in_features, out_features, bias=False, dtype=dtype)


def _rows(x: torch.Tensor | QuantAct) -> int:
    return x.m if isinstance(x, QuantAct) else x.shape[0]


class GatedRMSNorm(nn.Module):
    """The GDN output norm: `w * x / rms(x) * silu(z)` per head (norm before gate), fp32, rounded once."""

    def __init__(self, dim: int, eps: float, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype), requires_grad=False)
        self.eps = eps

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return ops.gated_rmsnorm(x, z, self.weight, self.eps)


class Rotary:
    """cos/sin cache [max_len, rotary_dim / 2] in the model dtype (as HF, which rounds cos and sin to the dtype),
    computed in fp32."""

    def __init__(self, cfg: Qwen35Config, max_len: int, device: torch.device, dtype: torch.dtype):
        rd = cfg.rotary_dim
        inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, rd, 2, dtype=torch.float32, device=device) / rd))
        freqs = torch.outer(torch.arange(max_len, dtype=torch.float32, device=device), inv_freq)
        self.cos = freqs.cos().to(dtype)
        self.sin = freqs.sin().to(dtype)
        self.rotary_dim = rd
        self.max_len = max_len
        sec = cfg.mrope_section
        self.mrope: tuple[int, int] | None = (int(sec[1]), int(sec[2])) if len(sec) == 3 else None  # (h, w) sections


class Attention(nn.Module):
    def __init__(self, cfg: Qwen35Config, index: int, dtype: torch.dtype, prefix: str):
        super().__init__()
        self.index = index  # among the full-attention layers (KV pool layer)
        self.h, self.hkv, self.d = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
        self.gated = cfg.attn_output_gate
        self.q_size = self.h * self.d * (2 if self.gated else 1)
        self.kv_size = self.hkv * self.d
        self.rotary_dim = cfg.rotary_dim
        self.qkv_proj = make_linear(cfg, f"{prefix}.qkv_proj", cfg.hidden_size, self.q_size + 2 * self.kv_size, dtype)
        self.o_proj = make_linear(cfg, f"{prefix}.o_proj", self.h * self.d, cfg.hidden_size, dtype)
        self.q_norm = RMSNorm(self.d, cfg.rms_norm_eps, dtype)
        self.k_norm = RMSNorm(self.d, cfg.rms_norm_eps, dtype)
        self.input_quant = "fp8" if isinstance(self.qkv_proj, Fp8Linear) else None
        self.out_fp8 = isinstance(self.o_proj, Fp8Linear)

    def forward(self, x: torch.Tensor | QuantAct, ctx: ForwardContext) -> torch.Tensor:
        t = _rows(x)
        qkv = self.qkv_proj(x)
        q = ctx.qk_rope_store(self, qkv)  # QK norm and RoPE; K and V written to this layer's KV pages
        o = ctx.attention(self.index, q)
        if self.gated:
            gate = qkv[:, :self.q_size].view(t, self.h, 2 * self.d)[..., self.d:]
            o = ops.sigmoid_gate_quant(o, gate) if self.out_fp8 and fp8.activation_quant() else \
                ops.sigmoid_gate(o, gate)
        return self.o_proj(o)


class GatedDeltaNet(nn.Module):
    def __init__(self, cfg: Qwen35Config, index: int, dtype: torch.dtype, prefix: str):
        super().__init__()
        self.index = index  # among the GDN layers (state pool layer)
        self.hk, self.hv = cfg.lin_k_heads, cfg.lin_v_heads
        self.dk, self.dv = cfg.lin_k_dim, cfg.lin_v_dim
        self.key_dim, self.value_dim, self.conv_dim = cfg.key_dim, cfg.value_dim, cfg.conv_dim
        self.in_proj_qkvz = make_linear(cfg, f"{prefix}.in_proj_qkvz", cfg.hidden_size, self.conv_dim + self.value_dim,
                                        dtype)
        self.in_proj_ba = make_linear(cfg, f"{prefix}.in_proj_ba", cfg.hidden_size, 2 * self.hv, dtype)
        self.conv_weight = nn.Parameter(torch.empty(self.conv_dim, cfg.conv_kernel, dtype=dtype), requires_grad=False)
        self.dt_bias = nn.Parameter(torch.ones(self.hv, dtype=dtype), requires_grad=False)
        self.A_log = nn.Parameter(torch.zeros(self.hv, dtype=torch.float32), requires_grad=False)  # fp32
        self.norm = GatedRMSNorm(self.dv, cfg.rms_norm_eps, dtype)
        self.out_proj = make_linear(cfg, f"{prefix}.out_proj", self.value_dim, cfg.hidden_size, dtype)
        fp8_in = [isinstance(p, Fp8Linear) for p in (self.in_proj_qkvz, self.in_proj_ba)]
        self.input_quant = None if not any(fp8_in) else "fp8" if all(fp8_in) else "fp8+bf16"
        self.out_fp8 = isinstance(self.out_proj, Fp8Linear)
        # Set by the runner under batch invariance: a BF16 in_proj_ba through `ops.linear_rows`, each row's bits its own
        self.ba_rows = False

    def forward(self, x: torch.Tensor | QuantAct, ctx: ForwardContext) -> torch.Tensor:
        t = _rows(x)
        xb = x.bf16 if isinstance(x, QuantAct) else x  # for a BF16 projection next to an FP8 one
        qkvz = self.in_proj_qkvz(x if isinstance(self.in_proj_qkvz, Fp8Linear) else xb)
        mixed_qkv, z = qkvz.split([self.conv_dim, self.value_dim], dim=-1)
        if isinstance(self.in_proj_ba, Fp8Linear):
            ba = self.in_proj_ba(x)
        elif self.ba_rows:
            ba = ops.linear_rows(xb, self.in_proj_ba.weight)
        else:  # our small-M kernel up to 8 rows, cuBLAS beyond
            ba = ops.linear_small(xb, self.in_proj_ba.weight)
        b, a = ba.split([self.hv, self.hv], dim=-1)
        # [T, HV, V] after the gated norm, or (FP8 out_proj) its quantization
        o = ctx.gdn(self, mixed_qkv, z.view(t, self.hv, self.dv), a, b,
                    quant=self.out_fp8 and fp8.activation_quant())
        return self.out_proj(o if isinstance(o, QuantAct) else o.reshape(t, self.value_dim))


class MLP(nn.Module):
    def __init__(self, cfg: Qwen35Config, dtype: torch.dtype, prefix: str):
        super().__init__()
        self.gate_up_proj = make_linear(cfg, f"{prefix}.gate_up_proj", cfg.hidden_size, 2 * cfg.intermediate_size,
                                        dtype)
        self.down_proj = make_linear(cfg, f"{prefix}.down_proj", cfg.intermediate_size, cfg.hidden_size, dtype)
        self.input_quant = "fp8" if isinstance(self.gate_up_proj, Fp8Linear) else None
        self.down_fp8 = isinstance(self.down_proj, Fp8Linear)

    def forward(self, x: torch.Tensor | QuantAct) -> torch.Tensor:
        gu = self.gate_up_proj(x)
        if not self.down_fp8:
            return self.down_proj(ops.silu_mul(gu))
        # SiLU-and-mul quantizes its fp32 product (as vLLM's), not its BF16 output: a step with PRECISE rows
        # splits it (`fp8.SplitAct`)
        t = gu.shape[0]
        nb = fp8.bf16_rows(t)
        if not nb:
            return self.down_proj(ops.silu_mul_quant(gu))
        if nb == t:
            return self.down_proj(ops.silu_mul(gu))
        return self.down_proj(fp8.SplitAct(ops.silu_mul_quant(gu[:t - nb]), ops.silu_mul(gu[t - nb:])))


class DecoderLayer(nn.Module):
    def __init__(self, cfg: Qwen35Config, layer: int, index: int, dtype: torch.dtype, prefix: str | None = None,
                 kind: str | None = None):
        super().__init__()
        prefix = prefix or f"layers.{layer}"
        self.is_attn = (kind or cfg.layer_types[layer]) == FULL
        if self.is_attn:
            self.self_attn = Attention(cfg, index, dtype, f"{prefix}.self_attn")
        else:
            self.linear_attn = GatedDeltaNet(cfg, index, dtype, f"{prefix}.linear_attn")
        self.mlp = MLP(cfg, dtype, f"{prefix}.mlp")
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype)
        self.mixer_quant = (self.self_attn if self.is_attn else self.linear_attn).input_quant

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None,
                ctx: ForwardContext) -> tuple[torch.Tensor, torch.Tensor]:
        h, residual = self.input_layernorm.apply(x, residual, self.mixer_quant)
        h = self.self_attn(h, ctx) if self.is_attn else self.linear_attn(h, ctx)
        h, residual = self.post_attention_layernorm.apply(h, residual, self.mlp.input_quant)
        return self.mlp(h), residual


class MTP(nn.Module):
    """The multi-token-prediction layer (module docstring). `index` is its attention layer's index in the KV pool."""

    def __init__(self, cfg: Qwen35Config, index: int, dtype: torch.dtype):
        super().__init__()
        h = cfg.hidden_size
        self.fc = make_linear(cfg, "mtp.fc", 2 * h, h, dtype)
        self.pre_fc_norm_embedding = RMSNorm(h, cfg.rms_norm_eps, dtype)
        self.pre_fc_norm_hidden = RMSNorm(h, cfg.rms_norm_eps, dtype)
        self.layers = nn.ModuleList([DecoderLayer(cfg, 0, index, dtype, prefix="mtp.layers.0", kind=FULL)])
        self.norm = RMSNorm(h, cfg.rms_norm_eps, dtype)
        self.fc_fp8 = isinstance(self.fc, Fp8Linear)

    def inputs(self, emb: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """The decoder layer's input: fc(cat(norm(embedding), norm(hidden)))."""
        x = torch.cat([self.pre_fc_norm_embedding(emb), self.pre_fc_norm_hidden(hidden)], dim=-1)
        return self.fc(x)

    def forward(self, emb: torch.Tensor, hidden: torch.Tensor, ctx: ForwardContext,
                select: torch.Tensor | None = None) -> torch.Tensor:
        """Embeddings of the next tokens and the hidden states at their positions → the layer's output after `norm`
        [T, hidden] (only the rows in `select` when given)."""
        h, residual = self.layers[0](self.inputs(emb, hidden), None, ctx)
        if select is not None:
            h, residual = h.index_select(0, select), residual.index_select(0, select)
        return self.norm.forward_add(h, residual)[0]

    def store_kv(self, emb: torch.Tensor, hidden: torch.Tensor, ctx: ForwardContext) -> None:
        """Only the layer's K and V at these positions (the KV a cache hit recomputes): the attention input side."""
        layer = self.layers[0]
        a = layer.self_attn
        x, _ = layer.input_layernorm.apply(self.inputs(emb, hidden), None, layer.mixer_quant)
        ctx.qk_rope_store(a, a.qkv_proj(x))


class Qwen35ForCausalLM(nn.Module):
    def __init__(self, cfg: Qwen35Config, dtype: torch.dtype, mtp: bool = False):
        super().__init__()
        self.cfg = cfg
        self.dtype = dtype
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size, dtype=dtype)
        n_attn = n_gdn = 0
        layers = []
        for i, kind in enumerate(cfg.layer_types):
            if kind == FULL:
                layers.append(DecoderLayer(cfg, i, n_attn, dtype))
                n_attn += 1
            else:
                layers.append(DecoderLayer(cfg, i, n_gdn, dtype))
                n_gdn += 1
        self.layers = nn.ModuleList(layers)
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype)
        if not cfg.tie_word_embeddings:
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False, dtype=dtype)
        self.mtp = MTP(cfg, n_attn, dtype) if mtp else None  # its attention: the KV pool's last layer

    @property
    def lm_head_weight(self) -> torch.Tensor:
        return self.embed_tokens.weight if self.cfg.tie_word_embeddings else self.lm_head.weight

    def forward(self, input_ids: torch.Tensor, ctx: ForwardContext, select: torch.Tensor | None = None,
                embeds: torch.Tensor | None = None) -> torch.Tensor:
        """Final hidden states [T, hidden] (after the last norm), or only the rows in `select` (logit rows). The MTP
        drafter needs every position's, so the runner selects nothing when it drafts. `embeds`: the input embeddings
        (image rows in place), instead of looking up `input_ids`."""
        h = self.embed_tokens(input_ids) if embeds is None else embeds
        residual = None
        for layer in self.layers:
            h, residual = layer(h, residual, ctx)
        if select is not None:
            h, residual = h.index_select(0, select), residual.index_select(0, select)
        return self.norm.forward_add(h, residual)[0]

    def logits(self, hidden: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        """[n, vocab] fp32, from a GEMM in the model dtype; into `out`'s first n rows when given (the logprobs'
        buffer: `runner.cuda.logprobs`), returned."""
        y = F.linear(hidden, self.lm_head_weight)
        return y.float() if out is None else out[:len(y)].copy_(y)


# -- weights -----------------------------------------------------------------------------------------------------

_SKIP_PREFIXES = ("model.visual.", "visual.")
_MTP_PREFIXES = ("mtp.", "model.mtp.")
_STRIP_PREFIXES = ("model.language_model.", "language_model.model.", "model.")
_LAYER = re.compile(r"(mtp\.)?layers\.(\d+)\.(.+)")


def _shards(cfg: Qwen35Config) -> dict[str, tuple[str, int]]:
    """Checkpoint suffix (within a layer) → (our parameter suffix, first row in the fused parameter)."""
    q = cfg.num_heads * cfg.head_dim * (2 if cfg.attn_output_gate else 1)
    kv = cfg.num_kv_heads * cfg.head_dim
    return {
        "self_attn.q_proj.weight": ("self_attn.qkv_proj.weight", 0),
        "self_attn.k_proj.weight": ("self_attn.qkv_proj.weight", q),
        "self_attn.v_proj.weight": ("self_attn.qkv_proj.weight", q + kv),
        "linear_attn.in_proj_qkv.weight": ("linear_attn.in_proj_qkvz.weight", 0),
        "linear_attn.in_proj_z.weight": ("linear_attn.in_proj_qkvz.weight", cfg.conv_dim),
        "linear_attn.in_proj_b.weight": ("linear_attn.in_proj_ba.weight", 0),
        "linear_attn.in_proj_a.weight": ("linear_attn.in_proj_ba.weight", cfg.lin_v_heads),
        "linear_attn.conv1d.weight": ("linear_attn.conv_weight", 0),
        "mlp.gate_proj.weight": ("mlp.gate_up_proj.weight", 0),
        "mlp.up_proj.weight": ("mlp.gate_up_proj.weight", cfg.intermediate_size),
    }


def map_weight_name(cfg: Qwen35Config, name: str, mtp: bool = False) -> tuple[str, int] | None:
    """HF checkpoint name → (parameter name, first row), or None for weights we do not load. A block-FP8 weight's
    `weight_scale_inv` (HF `fp8`) or `weight_scale` (`compressed-tensors`, selected via `cfg.quant.scale_suffix`) maps to the projection's `weight_scale`, at the shard's first row / 128 (block rows). `mtp`:
    load the MTP layer (`mtp.*`; the first one when the checkpoint has several)."""
    if name.startswith(_SKIP_PREFIXES):
        return None
    suffix = "." + cfg.quant.scale_suffix if cfg.quant is not None else ".weight_scale_inv"
    if name.startswith(_MTP_PREFIXES):
        name = name[name.index("mtp."):]
        if not mtp or (name.startswith("mtp.layers.") and not name.startswith("mtp.layers.0.")):
            return None
        scale = name.endswith(suffix)
        if scale:
            name = name[:-len(suffix)] + ".weight"
        pname, row0 = name, 0
        m = _LAYER.fullmatch(name)
        if m:
            target = _shards(cfg).get(m.group(3))
            if target is not None:
                pname, row0 = f"mtp.layers.0.{target[0]}", target[1]
        if scale:
            if row0 % fp8.GROUP:
                raise ValueError(f"the FP8 shard {name} starts at row {row0}, inside a 128-row block of {pname}")
            return pname.removesuffix("weight") + "weight_scale", row0 // fp8.GROUP
        return pname, row0
    scale = name.endswith(suffix)
    if scale:
        name = name[:-len(suffix)] + ".weight"
    if name in ("lm_head.weight", "language_model.lm_head.weight"):
        if scale:
            raise NotImplementedError("an FP8 lm_head (Qwen's FP8 checkpoints keep it BF16)")
        return None if cfg.tie_word_embeddings else ("lm_head.weight", 0)
    for p in _STRIP_PREFIXES:
        if name.startswith(p):
            name = name[len(p):]
            break
    pname, row0 = name, 0
    m = _LAYER.fullmatch(name)
    if m:
        target = _shards(cfg).get(m.group(3))
        if target is not None:
            pname, row0 = f"layers.{m.group(2)}.{target[0]}", target[1]
    if scale:
        if row0 % fp8.GROUP:
            raise ValueError(f"the FP8 shard {name} starts at row {row0}, inside a 128-row block of {pname}")
        return pname.removesuffix("weight") + "weight_scale", row0 // fp8.GROUP
    return pname, row0


def checkpoint_files(model_dir: Path) -> list[Path]:
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        names = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        return [model_dir / n for n in names]
    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors files in {model_dir}")
    return files


@torch.inference_mode()
def load_weights(model: Qwen35ForCausalLM, model_dir: Path, device: torch.device) -> dict[str, int]:
    """Copy checkpoint tensors into the (fused) parameters. Returns counts: loaded, skipped."""
    from safetensors import safe_open

    params = dict(model.named_parameters())
    cover = Coverage()
    loaded = skipped = 0
    unknown: list[str] = []
    dev = str(device) if device.type == "cpu" else f"cuda:{device.index or 0}"
    mtp = model.mtp is not None
    for path in checkpoint_files(model_dir):
        with safe_open(str(path), framework="pt", device=dev) as f:
            for name in f.keys():  # noqa: SIM118 - safe_open is not a mapping
                target = map_weight_name(model.cfg, name, mtp)
                if target is None:
                    skipped += 1
                    continue
                pname, row0 = target
                param = params.get(pname)
                if param is None:
                    unknown.append(name)
                    continue
                t = f.get_tensor(name)
                check_tensor(name, param, t, row0)
                if pname.endswith("conv_weight"):
                    t = t.reshape(t.shape[0], -1)
                dst = param.data.narrow(0, row0, t.shape[0]) if param.dim() else param.data
                if dst.shape != t.shape:
                    raise ValueError(f"{name} {tuple(t.shape)} does not fit {pname}[{row0}:] {tuple(dst.shape)}")
                dst.copy_(t)
                cover.add(pname, row0, t.shape[0] if t.dim() else 1)
                loaded += 1
    if unknown:
        raise ValueError(f"unexpected checkpoint weights: {unknown[:8]}{' ...' if len(unknown) > 8 else ''}")
    missing = cover.incomplete(params)
    if missing:
        raise ValueError(f"weights missing or partial in the checkpoint: {missing[:8]}")
    return {"loaded": loaded, "skipped": skipped}


def build_model(model_dir: str | Path, device: torch.device, dtype: torch.dtype,
                mtp: bool = False) -> Qwen35ForCausalLM:
    """Construct on the meta device, allocate on `device` without initialising, then load the checkpoint (`mtp`: with
    its MTP layer)."""
    model_dir = Path(model_dir)
    cfg = Qwen35Config.from_dir(model_dir)
    if mtp and not cfg.mtp_layers:
        raise ValueError(f"{model_dir} has no MTP layer (mtp_num_hidden_layers)")
    with torch.device("meta"):
        model = Qwen35ForCausalLM(cfg, dtype, mtp=mtp)
    model = model.to_empty(device=device).requires_grad_(False)
    counts = load_weights(model, model_dir, device)
    n_fp8 = sum(isinstance(m, Fp8Linear) for m in model.modules())
    log.info("loaded %d tensors (%d skipped: vision%s) from %s%s", counts["loaded"], counts["skipped"],
             "" if mtp else ", MTP", model_dir, f"; {n_fp8} block-FP8 projections" if n_fp8 else "")
    return model.eval()
