# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Team. All rights reserved.
# The proportional RoPE frequencies are adapted from src/transformers/modeling_rope_utils.py of transformers 5.17.0
# (https://github.com/huggingface/transformers), and changed for llm-mini-infer.
"""Gemma 4 dense text model (`Gemma4ForConditionalGeneration`'s language model, text only) as plain `nn.Module`s.

Its references are HF transformers' `modeling_gemma4.py` (the tests' oracle) and vLLM's `gemma4.py`. The 31B
IT checkpoint: 60 layers, 5 sliding to 1 full (every 6th layer, the last included, is full), hidden 5,376, 32
query heads, a GELU-tanh gated MLP of 21,504, a tied 262,144-token head with final logit softcapping 30:
- **sliding layers:** head dim 256, 16 KV heads, a causal window of `sliding_window` keys counting the query's own
  (HF's `kv_idx > q_idx - window`: a query at q sees keys [q - W + 1, q]),
  default NeoX RoPE over the whole head (θ 10k);
- **full layers:** head dim 512, 4 KV heads, `attention_k_eq_v`: no v_proj, V = v_norm(k_proj(x)) (an RMSNorm without
  weight, before RoPE), K = RoPE(k_norm(k_proj(x))); "proportional" RoPE (θ 1M) over the first 64 of each half's dims
  (`Rotary`);
- q_norm and k_norm (RMSNorm with weight) per head, attention scale 1.0 (the norms take the role of 1/sqrt(d));
- sandwich norms around attention and the MLP, then the residual stream times the layer's `layer_scalar`; plain
  RMSNorm `x / rms(x) * w`; embeddings scaled by sqrt(hidden) in the model dtype.

**K and V of the full layers** (`attention_k_eq_v`). The KV pool stores V only (one plane): the attention kernel derives
K = RoPE(V ⊙ k_norm.weight) per key, which halves the full layers' KV memory and reads (`kernels.attn512`; the pools:
`models.gemma4.backend`). V cannot be derived from a stored K (RoPE and the weight are not invertible in BF16), so the
stored plane is V.

**Numerics** follow a compiled graph (Inductor fuses each chain of elementwise ops around a reduction and computes it
in fp32, rounding where a tensor is stored; docs/vllm-compatibility.md, "Numerics"), so that parity with vLLM
is limited by batch noise: every norm reduces in fp32 and rounds once; q and k round once after norm and
RoPE (RoPE in fp32 against the cos/sin cache in the model dtype); the post-attention (post-feedforward) norm, the
residual add (and the layer scalar) round once, as the residual; the next norm reads that rounded residual;
GELU-tanh(gate) × up rounds once; the logits round in BF16, then three BF16 ops softcap them (`softcap`; HF's
`Gemma4ForCausalLM` does the same).

**Batch metadata** comes from a per-step context (`models.gemma4.backend.Gemma4Context`): the model sees one flat batch
of tokens and calls `ctx.qkv_store(...)` (norms, RoPE, the paged KV writes) and `ctx.attention(...)` per layer.

**Weights:** the text weights (`model.language_model.*`, or `model.*` of `Gemma4ForCausalLM`); the vision tower and its
embedder (`models.gemma4.vision` loads them) and any audio tower are skipped. Projections are fused at load: q, k(, v) →
`qkv_proj`; gate, up → `gate_up_proj`.

**Block-FP8 checkpoints** (`models.fp8`): the projections the checkpoint quantizes are `Fp8Linear`s (in a
compressed-tensors block-FP8 checkpoint, every projection of the text model); the embeddings (and so the tied head), the
norms, the vision tower and the KV cache stay BF16 (the full layers' KV in FP8 with `dev.cache.kv_cache_dtype fp8`:
`runner.cuda.gemma._Arena`). The full layers' qkv_proj is q and k only (`k_eq_v`: V is the K projection, so it is
computed once; the product is the same). Each FP8 projection's input comes quantized from the op
that produces it (`kernels.gemma_ops`), but the attention output, which the standalone kernel (`ops.quant_fp8`)
quantizes once the layer's attention is complete: no attention kernel here quantizes its output. PRECISE rows:
`kernels.fp8`.
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
from ...kernels import gemma_ops as gops
from ...kernels.fp8 import QuantAct
from ..common import Coverage
from ..fp8 import Fp8Linear, QuantConfig, check_tensor, fused_fp8

if TYPE_CHECKING:
    from .backend import Gemma4Context

log = logging.getLogger(__name__)

SLIDING, FULL = "sliding_attention", "full_attention"
# rope_parameters per layer type when a config has none for it (HF's Gemma4TextConfig defaults); the assistant's too
DEFAULT_ROPE = {SLIDING: {"rope_type": "default", "rope_theta": 10_000.0},
                FULL: {"rope_type": "proportional", "partial_rotary_factor": 0.25, "rope_theta": 1_000_000.0}}
MODEL_TYPES = ("gemma4", "gemma4_text")


@dataclass(frozen=True)
class Gemma4Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    layer_types: tuple[str, ...]
    num_heads: int
    sliding_kv_heads: int
    sliding_head_dim: int
    full_kv_heads: int
    full_head_dim: int
    sliding_window: int  # keys a sliding query sees, its own included
    sliding_theta: float
    full_theta: float
    full_rope_angles: int  # rotating frequency pairs of a full head (the rest have frequency 0)
    rms_norm_eps: float
    final_logit_softcapping: float | None
    tie_word_embeddings: bool
    max_position_embeddings: int
    k_eq_v: bool
    bidirectional: str | None = None  # use_bidirectional_attention: "vision" (images) or None
    quant: QuantConfig | None = None  # block-FP8 checkpoints

    @property
    def num_layers(self) -> int:
        return len(self.layer_types)

    @property
    def sliding_layers(self) -> list[int]:
        return [i for i, t in enumerate(self.layer_types) if t == SLIDING]

    @property
    def full_layers(self) -> list[int]:
        return [i for i, t in enumerate(self.layer_types) if t == FULL]

    def head_dim(self, kind: str) -> int:
        return self.sliding_head_dim if kind == SLIDING else self.full_head_dim

    def kv_heads(self, kind: str) -> int:
        return self.sliding_kv_heads if kind == SLIDING else self.full_kv_heads

    def stores_v(self, kind: str) -> bool:
        """Whether a layer kind's KV pool has a V plane next to K: sliding layers yes; full layers with k_eq_v store
        one plane (V, from which the kernel derives K)."""
        return kind == SLIDING or not self.k_eq_v

    def fp8_module(self, name: str) -> bool:
        """Whether our projection `name` (e.g. `layers.3.self_attn.qkv_proj`) is block-FP8 in this checkpoint
        (`fused_fp8`)."""
        m = _LAYER.fullmatch(name)
        if self.quant is None or m is None:
            return False  # the embeddings (and the tied head) stay BF16
        layer, rest = int(m.group(1)), m.group(2)
        parts = list(_FUSED_MODULES.get(rest, (rest,)))
        if rest == "self_attn.qkv_proj" and self.k_eq_v and self.layer_types[layer] == FULL:
            parts = parts[:2]  # q and k: a full layer of a k_eq_v model has no v_proj
        return fused_fp8(self.quant, name, [f"layers.{layer}.{p}" for p in parts], _CKPT_PREFIXES)

    def kv_bytes_per_token(self, kind: str, dtype_bytes: int) -> int:
        """KV bytes of one position over all layers of a kind."""
        n = len(self.sliding_layers if kind == SLIDING else self.full_layers)
        planes = 2 if self.stores_v(kind) else 1
        return n * planes * self.kv_heads(kind) * self.head_dim(kind) * dtype_bytes

    @classmethod
    def from_hf(cls, hf: dict[str, Any]) -> Gemma4Config:
        """From an HF `config.json`: `Gemma4ForConditionalGeneration` (with `text_config`) or the text-only
        `Gemma4ForCausalLM` (flat). Raises NotImplementedError for the variants this build does not serve."""
        tc = hf.get("text_config", hf)
        model_type = tc.get("model_type", hf.get("model_type", ""))
        if model_type not in MODEL_TYPES and hf.get("model_type") not in MODEL_TYPES:
            raise NotImplementedError(f"model_type {model_type!r} is not Gemma 4 (gemma4_text)")
        quant = QuantConfig.from_hf(hf["quantization_config"], strip=_STRIP_PREFIXES) \
            if hf.get("quantization_config") else None
        for key, what in (("enable_moe_block", "Gemma 4 MoE"), ("use_double_wide_mlp", "the double-wide MLP")):
            if tc.get(key):
                raise NotImplementedError(f"{what} ({key})")
        if tc.get("hidden_size_per_layer_input"):
            raise NotImplementedError("per-layer embeddings (hidden_size_per_layer_input: the E2B/E4B models)")
        if tc.get("num_kv_shared_layers"):
            raise NotImplementedError("KV-shared layers (num_kv_shared_layers)")
        if tc.get("attention_bias"):
            raise NotImplementedError("attention_bias")
        if tc.get("attn_logit_softcapping"):
            raise NotImplementedError("attention logit softcapping")
        if tc.get("hidden_activation", "gelu_pytorch_tanh") != "gelu_pytorch_tanh":
            raise NotImplementedError(f"hidden_activation {tc['hidden_activation']!r}")
        bidir = tc.get("use_bidirectional_attention")
        if bidir not in (None, "vision"):
            raise NotImplementedError(f"use_bidirectional_attention {bidir!r}")
        n = int(tc["num_hidden_layers"])
        types = tc.get("layer_types") or [SLIDING if (i + 1) % 6 else FULL for i in range(n)]
        types = list(types)
        if len(types) != n or set(types) - {SLIDING, FULL}:
            raise ValueError(f"bad layer_types {types}")
        if types[-1] != FULL:  # HF forces the last layer to full attention
            types[-1] = FULL
        rope = tc.get("rope_parameters") or {}
        sl, fu = rope.get(SLIDING, DEFAULT_ROPE[SLIDING]), rope.get(FULL, DEFAULT_ROPE[FULL])
        if sl.get("rope_type", "default") != "default" or sl.get("partial_rotary_factor", 1.0) != 1.0:
            raise NotImplementedError(f"sliding RoPE {sl}")
        if fu.get("rope_type") not in ("proportional", "default") or fu.get("factor", 1.0) != 1.0:
            raise NotImplementedError(f"full-attention RoPE {fu}")
        head = int(tc.get("head_dim", 256))
        full_head = int(tc.get("global_head_dim", 512))
        k_eq_v = bool(tc.get("attention_k_eq_v", False))
        full_kv = int(tc.get("num_global_key_value_heads") or tc["num_key_value_heads"]) if k_eq_v else \
            int(tc["num_key_value_heads"])
        # transformers' save_pretrained writes the full layers' overrides as `per_layer_config` (by layer index) instead
        # of global_head_dim / num_global_key_value_heads
        overrides: dict[str, dict] = {}
        for k, v in (tc.get("per_layer_config") or {}).items():
            kind = types[int(k)] if str(k).isdigit() else k
            if kind != FULL:
                raise NotImplementedError(f"per_layer_config for {kind} layers")
            if overrides.setdefault(FULL, v) != v:
                raise NotImplementedError("full-attention layers of different geometries")
        if FULL in overrides:
            full_head = int(overrides[FULL].get("head_dim", full_head))
            full_kv = int(overrides[FULL].get("num_key_value_heads", full_kv))
        partial = float(fu.get("partial_rotary_factor", 1.0))
        if fu.get("rope_type") == "default" and partial != 1.0:
            raise NotImplementedError("partial default RoPE on full layers")
        angles = int(partial * full_head // 2)  # HF's `rope_angles`
        tie = bool(tc.get("tie_word_embeddings", hf.get("tie_word_embeddings", True)))
        return cls(vocab_size=int(tc["vocab_size"]), hidden_size=int(tc["hidden_size"]),
                   intermediate_size=int(tc["intermediate_size"]), layer_types=tuple(types),
                   num_heads=int(tc["num_attention_heads"]), sliding_kv_heads=int(tc["num_key_value_heads"]),
                   sliding_head_dim=head, full_kv_heads=full_kv, full_head_dim=full_head,
                   sliding_window=int(tc.get("sliding_window", 512)), sliding_theta=float(sl.get("rope_theta", 1e4)),
                   full_theta=float(fu.get("rope_theta", 1e6)), full_rope_angles=angles,
                   rms_norm_eps=float(tc.get("rms_norm_eps", 1e-6)),
                   final_logit_softcapping=tc.get("final_logit_softcapping"), tie_word_embeddings=tie,
                   max_position_embeddings=int(tc.get("max_position_embeddings", 131_072)), k_eq_v=k_eq_v,
                   bidirectional=bidir, quant=quant)

    @classmethod
    def from_dir(cls, model_dir: str | Path) -> Gemma4Config:
        return cls.from_hf(json.loads((Path(model_dir) / "config.json").read_text()))


def is_gemma4(model_dir: str | Path) -> bool:
    """Whether a checkpoint directory holds a Gemma 4 model (by config.json's model_type)."""
    path = Path(model_dir) / "config.json"
    if not path.exists():
        return False
    hf = json.loads(path.read_text())
    return hf.get("model_type") in MODEL_TYPES or hf.get("text_config", {}).get("model_type") in MODEL_TYPES


# -- layers ------------------------------------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Gemma 4's RMSNorm, `x / rms(x) * w` (with `weight=False`: no weight, the V norm), fp32, rounded once."""

    def __init__(self, dim: int, eps: float, dtype: torch.dtype, weight: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype), requires_grad=False) if weight else None
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return gops.rmsnorm(x, self.weight, self.eps)


class Rotary:
    """cos/sin caches [max_len, head_dim / 2] in the model dtype (as HF, which rounds cos and sin to the dtype), one
    per layer kind, computed in fp32 from HF's inverse frequencies: default RoPE over the whole sliding head; the full
    layers' proportional RoPE (`full_rope_angles` frequencies of a 512-dim head, the rest 0)."""

    def __init__(self, cfg: Gemma4Config, max_len: int, device: torch.device, dtype: torch.dtype):
        pos = torch.arange(max_len, dtype=torch.float32, device=device)
        self.cos: dict[str, torch.Tensor] = {}
        self.sin: dict[str, torch.Tensor] = {}
        for kind in (SLIDING, FULL):
            inv = inv_freq(cfg, kind).to(device)
            freqs = torch.outer(pos, inv)
            self.cos[kind] = freqs.cos().to(dtype)
            self.sin[kind] = freqs.sin().to(dtype)
        self.max_len = max_len


def inv_freq(cfg: Gemma4Config, kind: str) -> torch.Tensor:
    """HF's inverse frequencies (fp32, [head_dim / 2]): `compute_default_rope_parameters` for sliding layers,
    `_compute_proportional_rope_parameters` for full ones."""
    d = cfg.head_dim(kind)
    if kind == SLIDING:
        return 1.0 / (cfg.sliding_theta ** (torch.arange(0, d, 2, dtype=torch.float) / d))
    a = cfg.full_rope_angles
    rot = 1.0 / (cfg.full_theta ** (torch.arange(0, 2 * a, 2, dtype=torch.int64).to(torch.float) / d))
    return torch.cat([rot, torch.zeros(d // 2 - a, dtype=torch.float32)]) if d // 2 > a else rot


class Attention(nn.Module):
    def __init__(self, cfg: Gemma4Config, layer: int, index: int, dtype: torch.dtype):
        super().__init__()
        self.kind = cfg.layer_types[layer]
        self.index = index  # among the layers of its kind (KV pool layer)
        self.h, self.hkv, self.d = cfg.num_heads, cfg.kv_heads(self.kind), cfg.head_dim(self.kind)
        self.k_eq_v = cfg.k_eq_v and self.kind == FULL
        self.q_size = self.h * self.d
        self.kv_size = self.hkv * self.d
        out = self.q_size + self.kv_size * (1 if self.k_eq_v else 2)
        self.qkv_proj = make_linear(cfg, f"layers.{layer}.self_attn.qkv_proj", cfg.hidden_size, out, dtype)
        self.o_proj = make_linear(cfg, f"layers.{layer}.self_attn.o_proj", self.q_size, cfg.hidden_size, dtype)
        self.q_norm = RMSNorm(self.d, cfg.rms_norm_eps, dtype)
        self.k_norm = RMSNorm(self.d, cfg.rms_norm_eps, dtype)
        self.eps = cfg.rms_norm_eps
        self.in_fp8 = isinstance(self.qkv_proj, Fp8Linear)  # its producer quantizes the input
        self.out_fp8 = isinstance(self.o_proj, Fp8Linear)

    def forward(self, x: torch.Tensor | QuantAct, ctx: Gemma4Context) -> torch.Tensor:
        qkv = project(self.qkv_proj, x)
        q = ctx.qkv_store(self, qkv)  # q/k norms, RoPE, v norm; K and V (or V alone) into this layer's KV pages
        o = ctx.attention(self, q)
        if self.out_fp8 and fp8.activation_quant():
            o = ops.quant_fp8(o)  # after both passes of a sliding layer with image spans
        return project(self.o_proj, o)


SMALL_M = 2  # rows up to which `linear` runs our small-M kernel: from 1, 2, 4 and 8 rows measured on an RTX PRO 6000


def linear(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """x [M, K] @ w [N, K]ᵀ: our small-M kernel up to `SMALL_M` rows on CUDA (`ops.linear_small`: one CTA per weight
    row, fp32 FMAs on the CUDA cores), `F.linear` (cuBLAS) otherwise and on CPU. On an RTX PRO 6000 at the board's
    default power limit our kernel runs a decode step's linears faster than cuBLAS at one or two rows; with more rows
    its FMAs reach the board's power limit and its clocks drop, which slows the whole step."""
    return ops.linear_small(x, w) if x.shape[0] <= SMALL_M else F.linear(x, w)


def project(lin: nn.Module, x: torch.Tensor | QuantAct) -> torch.Tensor:
    """A projection: an `Fp8Linear` runs itself (the block-FP8 GEMM, or BF16 activations against its dequantized
    weight), a BF16 one goes through `linear`."""
    return lin(x) if isinstance(lin, Fp8Linear) else linear(x, lin.weight)


def make_linear(cfg: Gemma4Config, name: str, in_features: int, out_features: int, dtype: torch.dtype) -> nn.Module:
    """`Fp8Linear` where the checkpoint quantizes the projection `name`, else a BF16 `nn.Linear`."""
    if cfg.fp8_module(name):
        return Fp8Linear(in_features, out_features, dtype)
    return nn.Linear(in_features, out_features, bias=False, dtype=dtype)


class MLP(nn.Module):
    def __init__(self, cfg: Gemma4Config, layer: int, dtype: torch.dtype):
        super().__init__()
        i = cfg.intermediate_size
        self.gate_up_proj = make_linear(cfg, f"layers.{layer}.mlp.gate_up_proj", cfg.hidden_size, 2 * i, dtype)
        self.down_proj = make_linear(cfg, f"layers.{layer}.mlp.down_proj", i, cfg.hidden_size, dtype)
        self.in_fp8 = isinstance(self.gate_up_proj, Fp8Linear)
        self.down_fp8 = isinstance(self.down_proj, Fp8Linear)

    def forward(self, x: torch.Tensor | QuantAct) -> torch.Tensor:
        gu = project(self.gate_up_proj, x)
        act = gops.gelu_tanh_mul_quant(gu) if self.down_fp8 and fp8.activation_quant() else gops.gelu_tanh_mul(gu)
        return project(self.down_proj, act)


class DecoderLayer(nn.Module):
    def __init__(self, cfg: Gemma4Config, layer: int, index: int, dtype: torch.dtype):
        super().__init__()
        self.self_attn = Attention(cfg, layer, index, dtype)
        self.mlp = MLP(cfg, layer, dtype)
        h, eps = cfg.hidden_size, cfg.rms_norm_eps
        self.input_layernorm = RMSNorm(h, eps, dtype)
        self.post_attention_layernorm = RMSNorm(h, eps, dtype)
        self.pre_feedforward_layernorm = RMSNorm(h, eps, dtype)
        self.post_feedforward_layernorm = RMSNorm(h, eps, dtype)
        self.layer_scalar = nn.Parameter(torch.ones(1, dtype=dtype), requires_grad=False)
        self.eps = eps

    def forward(self, x: torch.Tensor | QuantAct, residual: torch.Tensor,
                ctx: Gemma4Context) -> tuple[torch.Tensor, torch.Tensor]:
        """x: this layer's input, already input-normed (quantized for a block-FP8 qkv_proj); residual: the stream
        before it. Returns (the MLP's output before its post norm, the stream after the attention block): the model
        finishes the layer with the next layer's input norm (`gemma_ops.sandwich`)."""
        a = self.self_attn(x, ctx)
        sw = gops.sandwich_quant if self.mlp.in_fp8 and fp8.activation_quant() else gops.sandwich
        residual, h = sw(a, residual, self.post_attention_layernorm.weight, self.pre_feedforward_layernorm.weight,
                         self.eps)
        return self.mlp(h), residual


class Gemma4ForCausalLM(nn.Module):
    def __init__(self, cfg: Gemma4Config, dtype: torch.dtype):
        super().__init__()
        self.cfg = cfg
        self.dtype = dtype
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size, dtype=dtype)
        counts = {SLIDING: 0, FULL: 0}
        layers = []
        for i, kind in enumerate(cfg.layer_types):
            layers.append(DecoderLayer(cfg, i, counts[kind], dtype))
            counts[kind] += 1
        self.layers = nn.ModuleList(layers)
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype)
        if not cfg.tie_word_embeddings:
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False, dtype=dtype)
        # sqrt(hidden) in the model dtype (HF's `embed_scale.to(dtype)`): 73.5 in BF16
        self.register_buffer("normalizer", torch.tensor(cfg.hidden_size ** 0.5, dtype=dtype), persistent=False)

    @property
    def lm_head_weight(self) -> torch.Tensor:
        return self.embed_tokens.weight if self.cfg.tie_word_embeddings else self.lm_head.weight

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return gops.embed_scale(self.embed_tokens(input_ids), self.normalizer)

    def forward(self, input_ids: torch.Tensor, ctx: Gemma4Context, select: torch.Tensor | None = None,
                embeds: torch.Tensor | None = None) -> torch.Tensor:
        """Final hidden states [T, hidden] (after the final norm), or only the rows in `select` (logit rows).
        `embeds`: the input embeddings ready-made (scaled), instead of looking up `input_ids`."""
        residual = self.embed(input_ids) if embeds is None else embeds
        layers = self.layers
        quant = fp8.activation_quant()
        first = layers[0]
        x = gops.rmsnorm_quant(residual, first.input_layernorm.weight, first.eps) \
            if quant and first.self_attn.in_fp8 else first.input_layernorm(residual)
        last = len(layers) - 1
        for i, layer in enumerate(layers):
            m, residual = layer(x, residual, ctx)
            nxt = self.norm.weight if i == last else layers[i + 1].input_layernorm.weight
            sw = gops.sandwich_quant if quant and i < last and layers[i + 1].self_attn.in_fp8 else gops.sandwich
            residual, x = sw(m, residual, layer.post_feedforward_layernorm.weight, nxt, layer.eps,
                             scalar=layer.layer_scalar)
        return x if select is None else x.index_select(0, select)

    def logits(self, hidden: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        """[n, vocab] fp32: the head's GEMM in the model dtype, softcapped (BF16 ops on BF16 logits), then widened for
        the sampler; into `out`'s first n rows when given (the logprobs' buffer: `runner.cuda.logprobs`), returned."""
        y = softcap(linear(hidden, self.lm_head_weight), self.cfg.final_logit_softcapping)
        return y.float() if out is None else out[:len(y)].copy_(y)


def softcap(logits: torch.Tensor, cap: float | None) -> torch.Tensor:
    """HF's Gemma4ForCausalLM: `tanh(logits / cap) * cap`, three ops in the logits' dtype."""
    if not cap:
        return logits
    return torch.tanh(logits / cap) * cap


# -- weights -----------------------------------------------------------------------------------------------------

_SKIP_PREFIXES = ("model.vision_tower.", "model.embed_vision.", "vision_tower.", "embed_vision.", "model.audio_tower.",
                  "model.embed_audio.")
_STRIP_PREFIXES = ("model.language_model.", "language_model.model.", "model.")
_CKPT_PREFIXES = ("model.language_model.", "model.", "")  # the full names compressed-tensors' `ignore` may match
_LAYER = re.compile(r"layers\.(\d+)\.(.+)")
# Our fused projections and the checkpoint modules they are made of
_FUSED_MODULES = {"self_attn.qkv_proj": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
                  "mlp.gate_up_proj": ("mlp.gate_proj", "mlp.up_proj")}


def _shards(cfg: Gemma4Config, layer: int) -> dict[str, tuple[str, int]]:
    """Checkpoint suffix within layer `layer` → (our parameter suffix, first row in the fused parameter)."""
    kind = cfg.layer_types[layer]
    q = cfg.num_heads * cfg.head_dim(kind)
    kv = cfg.kv_heads(kind) * cfg.head_dim(kind)
    out = {"self_attn.q_proj.weight": ("self_attn.qkv_proj.weight", 0),
           "self_attn.k_proj.weight": ("self_attn.qkv_proj.weight", q),
           "mlp.gate_proj.weight": ("mlp.gate_up_proj.weight", 0),
           "mlp.up_proj.weight": ("mlp.gate_up_proj.weight", cfg.intermediate_size)}
    if cfg.stores_v(kind) and not (cfg.k_eq_v and kind == FULL):
        out["self_attn.v_proj.weight"] = ("self_attn.qkv_proj.weight", q + kv)
    return out


def map_weight_name(cfg: Gemma4Config, name: str) -> tuple[str, int] | None:
    """HF checkpoint name → (parameter name, first row), or None for weights we do not load (vision, audio). A
    block-FP8 weight's scale (`weight_scale`, or `weight_scale_inv` in the `fp8` format: `QuantConfig.scale_suffix`)
    maps to the projection's `weight_scale`, at the shard's first row / 128 (block rows)."""
    if name.startswith(_SKIP_PREFIXES):
        return None
    suffix = "." + cfg.quant.scale_suffix if cfg.quant is not None else None
    scale = suffix is not None and name.endswith(suffix)
    if scale:
        name = name[:-len(suffix)] + ".weight"
    if name in ("lm_head.weight", "language_model.lm_head.weight", "model.lm_head.weight"):
        if scale:
            raise NotImplementedError("an FP8 lm_head (Gemma 4 ties it to the BF16 embeddings)")
        return None if cfg.tie_word_embeddings else ("lm_head.weight", 0)
    for p in _STRIP_PREFIXES:
        if name.startswith(p):
            name = name[len(p):]
            break
    pname, row0 = name, 0
    m = _LAYER.fullmatch(name)
    if m:
        layer = int(m.group(1))
        if layer >= cfg.num_layers:
            raise ValueError(f"{name}: layer {layer} beyond num_hidden_layers {cfg.num_layers}")
        if m.group(2) == "self_attn.v_proj.weight" and cfg.k_eq_v and cfg.layer_types[layer] == FULL:
            raise ValueError(f"{name}: a v_proj on a full-attention layer of an attention_k_eq_v model")
        target = _shards(cfg, layer).get(m.group(2))
        if target is not None:
            pname, row0 = f"layers.{layer}.{target[0]}", target[1]
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
def load_weights(model: Gemma4ForCausalLM, model_dir: Path, device: torch.device) -> dict[str, int]:
    """Copy checkpoint tensors into the (fused) parameters. Returns counts: loaded, skipped."""
    from safetensors import safe_open

    params = dict(model.named_parameters())
    cover = Coverage()
    loaded = skipped = 0
    unknown: list[str] = []
    dev = str(device) if device.type == "cpu" else f"cuda:{device.index or 0}"
    for path in checkpoint_files(model_dir):
        with safe_open(str(path), framework="pt", device=dev) as f:
            for name in f.keys():  # noqa: SIM118 - safe_open is not a mapping
                target = map_weight_name(model.cfg, name)
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
                if pname.endswith("layer_scalar"):
                    t = t.reshape(1)
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


def build_model(model_dir: str | Path, device: torch.device, dtype: torch.dtype) -> Gemma4ForCausalLM:
    """Construct on the meta device, allocate on `device` without initialising, then load the checkpoint."""
    model_dir = Path(model_dir)
    cfg = Gemma4Config.from_dir(model_dir)
    with torch.device("meta"):
        model = Gemma4ForCausalLM(cfg, dtype)
    model = model.to_empty(device=device).requires_grad_(False)
    model.normalizer.fill_(cfg.hidden_size ** 0.5)  # a buffer: to_empty left it uninitialised
    counts = load_weights(model, model_dir, device)
    n_fp8 = sum(isinstance(m, Fp8Linear) for m in model.modules())
    log.info("loaded %d tensors (%d skipped: vision) from %s%s", counts["loaded"], counts["skipped"], model_dir,
             f"; {n_fp8} block-FP8 projections ({cfg.quant.method})" if n_fp8 else "")
    return model.eval()
