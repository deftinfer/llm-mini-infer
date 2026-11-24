"""Per-step batch metadata, the forward context and the attention backends of Gemma 4 (docs/architecture.md,
"Sliding-window models", "Models").

**Two kinds of KV page** (`Gemma4Pools`), allocated by the runner:
- **window pages** `wkv` [n_sliding, pages, 2, page_size, Hs, Ds]: the sliding layers' K and V planes (FlashInfer's
  NHD layout per layer), held for the positions a request's next queries can see;
- **full pages** `fkv` [n_full, pages', full_page_size, Hf, Df] (one plane, V, with `attention_k_eq_v`; [.., 2, ..]
  without): every position of every request, shared through the prefix cache. In the runner's dtype, or uint8 bytes
  of FP8 E4M3 values (`dev.cache.kv_cache_dtype fp8`: `gemma_ops.kv_fp8` stores, `kv_read` reads).
In the runner both are views of one arena (`runner.cuda.gemma._Arena`), whose full pages it addresses through sub-pages
of `full_page_size` positions; the backends need only the two views. Page 0 of each is the null page: CUDA-graph
padding rows write their KV there.

**Metadata** (`Gemma4Meta`): token-level tensors are computed on the device; row-level fields (row bounds, KV lengths,
the first window page of each row, the attention plans) are what the eager breaks read (`Gemma4Context`).

**Attention backends:** `TorchSlidingAttention` and `TorchFullAttention` (references: SDPA over the gathered pages,
one row at a time; any device); on CUDA the sliding layers on `kernels.flashinfer.FlashInferWindowAttention`
(FlashInfer's paged prefill with `window_left`) or, under batch invariance, `kernels.attn_window.WindowAttention`, and
the full layers on `kernels.attn512.Attn512` (our kernel, K derived from the stored V plane).

**Bidirectional image spans** (`use_bidirectional_attention: "vision"`). In the sliding layers a query inside an image
span sees every key of its span, later ones too, within the window (HF's `(causal OR blockwise) AND sliding_window`):
keys [q − W + 1, span end − 1]. The full layers stay causal (HF's `create_masks_for_vision_model`). The spans come
with the step (`Gemma4Meta.span_*`, from the plan's SPANS, each whole inside its row): the reference widens each span
query's causal limit, FlashInfer runs the span queries again without the causal mask, and `kernels.attn_window` takes
them in its one pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from ...kernels import gemma_ops as gops
from ..common import weak_view

if TYPE_CHECKING:
    from .model import Attention, Rotary

SLIDING, FULL = "sliding_attention", "full_attention"


@dataclass
class Gemma4Pools:
    wkv: torch.Tensor  # [n_sliding, pages, 2, ps, Hs, Ds]
    fkv: torch.Tensor  # [n_full, pages', fps, Hf, Df] (k_eq_v) or [n_full, pages', 2, fps, Hf, Df]
    page_size: int  # ps: positions of a window page
    window: int  # keys a sliding query sees, its own included
    full_page_size: int = 0  # fps: positions of a page of `fkv` (0: page_size); `Gemma4Meta.page_table` indexes these


@dataclass
class Gemma4Meta:
    """One step's batch as the backends need it. `*_host` fields are numpy, the others device tensors."""

    num_tokens: int
    positions: torch.Tensor | None  # [T] int64
    f_page: torch.Tensor | None  # [T] int64: the full-pool page holding each token's KV
    f_off: torch.Tensor | None
    w_page: torch.Tensor | None  # [T] int64: the window-pool page
    w_off: torch.Tensor | None
    n_rows: int
    row_start_host: np.ndarray  # [n_rows] first token of each row
    row_nq_host: np.ndarray  # [n_rows]
    row_kv_len_host: np.ndarray  # [n_rows] KV length after this step (seq_len + n_query)
    row_slot_host: np.ndarray  # [n_rows]
    row_wlo_host: np.ndarray  # [n_rows] first window page of each row: (max(0, first query − window + 1)) // ps
    page_table: torch.Tensor | None = None  # [slots, max_pages] int32: full pool
    wpage_table: torch.Tensor | None = None  # [slots, max_pages] int32: window pool
    sliding_plan: Any = None  # set by the sliding backend's plan() (FlashInfer's)
    window_plan: Any = None  # set by the sliding backend's plan() (ours: `kernels.attn_window`)
    full_plan: Any = None  # set by the full backend's plan()
    kv_len_dev: torch.Tensor | None = None  # true KV lengths on the device when the host knows bounds (MTP)
    # MTP: the plans are made for upper bounds; with `fix_len` each backend first rewrites its plan's lengths on the
    # device from the true ones (`kv_len_dev`; the sliding rows' KV from their first window page: `kv_named_dev`),
    # static buffers the step body fills, so a breakable graph's eager break reads them at replay
    fix_len: bool = False
    kv_named_dev: torch.Tensor | None = None
    drafter: Any = None  # the MTP assistant's meta for this step (its rows: one query each, after the verify)
    # Bidirectional image spans: (row, start, end) in positions, the span queries' token indices in the step (device
    # int64) and the sliding layers' second, non-causal plan over them (FlashInfer)
    span_rows_host: np.ndarray | None = None
    span_start_host: np.ndarray | None = None
    span_end_host: np.ndarray | None = None
    span_q: torch.Tensor | None = None
    bidi_plan: Any = None


class Gemma4Context:
    """What the model's layers call during one forward pass. With `capture` set (a `runner.cuda.graphs.
    BreakableCapture`), attention is an eager break that reads `self.meta` when it runs (`models.qwen3_5.backend.
    ForwardContext`); the KV writes stay inside the graph (their token-level metadata is in static buffers)."""

    __slots__ = ("_full", "_sliding", "capture", "meta", "pools", "rotary")

    def __init__(self, meta: Gemma4Meta | None, sliding: Any, full: Any, rotary: Rotary, pools: Gemma4Pools,
                 capture: Any = None):
        self.meta = meta
        self._sliding = sliding
        self._full = full
        self.rotary = rotary
        self.pools = pools
        self.capture = capture

    def qkv_store(self, layer: Attention, qkv: torch.Tensor) -> torch.Tensor:
        """q [T, H, D] after its norm and RoPE; K and V (or V alone, k_eq_v) into this layer's KV pages."""
        m, r, p = self.meta, self.rotary, self.pools
        sliding = layer.kind == SLIDING
        kv = p.wkv[layer.index] if sliding else p.fkv[layer.index]
        return gops.qkv_store(qkv, heads=layer.h, kv_heads=layer.hkv, head_dim=layer.d, k_eq_v=layer.k_eq_v,
                              q_w=layer.q_norm.weight, k_w=layer.k_norm.weight, eps=layer.eps, positions=m.positions,
                              cos_cache=r.cos[layer.kind], sin_cache=r.sin[layer.kind], kv_layer=kv,
                              kv_page=m.w_page if sliding else m.f_page, kv_off=m.w_off if sliding else m.f_off)

    def q_rope(self, layer: Any, q: torch.Tensor) -> torch.Tensor:
        """A Q-only layer's queries (the MTP assistant's, `models.gemma4.assistant`): q [T, H·D] -> [T, H, D] after its
        q_norm and RoPE at the meta's positions; nothing is written to the KV (it reads the target's)."""
        m, r = self.meta, self.rotary
        return gops.q_rope(q, heads=layer.h, head_dim=layer.d, q_w=layer.q_norm.weight, eps=layer.eps,
                           positions=m.positions, cos_cache=r.cos[layer.kind], sin_cache=r.sin[layer.kind])

    def attention(self, layer: Attention, q: torch.Tensor) -> torch.Tensor:
        """q [T, H, D] -> the attention output [T, H * D]."""
        backend = self._sliding if layer.kind == SLIDING else self._full
        out = torch.empty_like(q)
        if self.capture is None:
            backend.run(self.meta, layer, q, out)
        else:
            qw, ow = weak_view(q), weak_view(out)

            def run() -> None:  # the step's real tokens; padding tokens are not in the plans
                n = self.meta.num_tokens
                if n:
                    backend.run(self.meta, layer, qw[:n], ow[:n])

            self.capture.eager(run)
        return out.view(q.shape[0], -1)


# -- reference backends --------------------------------------------------------------------------------------------


class TorchSlidingAttention:
    """Reference sliding-window attention: per row, gather its window pages (from the row's first window page) and
    run SDPA with the causal window mask."""

    name = "torch"

    def __init__(self, pools: Gemma4Pools):
        self.pools = pools

    def plan(self, meta: Gemma4Meta, **_: Any) -> None:
        meta.sliding_plan = None

    def run(self, meta: Gemma4Meta, layer: Attention, q: torch.Tensor, out: torch.Tensor) -> None:
        p = self.pools
        kv = p.wkv[layer.index]
        kc, vc = kv[:, 0], kv[:, 1]  # [pages, ps, Hkv, D]
        ps, w = p.page_size, p.window
        hkv, d = kc.shape[2], kc.shape[3]
        lengths = meta.kv_len_dev.tolist() if meta.kv_len_dev is not None else meta.row_kv_len_host
        for r in range(meta.n_rows):
            s, n, length = int(meta.row_start_host[r]), int(meta.row_nq_host[r]), int(lengths[r])
            lo = int(meta.row_wlo_host[r])
            pages = meta.wpage_table[int(meta.row_slot_host[r]), lo:-(-length // ps)].long()
            keys = kc[pages].reshape(-1, hkv, d)[:length - lo * ps]
            vals = vc[pages].reshape(-1, hkv, d)[:length - lo * ps]
            kpos = torch.arange(lo * ps, length, device=q.device)
            qpos = torch.arange(length - n, length, device=q.device)
            qlim = None
            if meta.span_rows_host is not None and (meta.span_rows_host == r).any():  # span queries see the span
                qlim = qpos.clone()
                for a, e in zip(meta.span_start_host[meta.span_rows_host == r].tolist(),
                                meta.span_end_host[meta.span_rows_host == r].tolist()):
                    qlim[(qpos >= a) & (qpos < e)] = e - 1
            out[s:s + n] = gops.attention_ref(q[s:s + n], keys, vals, qpos, kpos, window=w, qlim=qlim)


class TorchFullAttention:
    """Reference full attention of a k_eq_v layer: per row, gather the stored V plane of every position, derive K =
    RoPE(V ⊙ k_norm.weight) at the keys' positions (`gemma_ops.k_from_v`), causal SDPA. Without k_eq_v the pool has K
    and V planes. An FP8 pool (uint8 bytes) is dequantized into the queries' dtype first (`gemma_ops.kv_read`)."""

    name = "torch"

    def __init__(self, pools: Gemma4Pools, rotary: Rotary, k_eq_v: bool):
        self.pools = pools
        self.rotary = rotary
        self.k_eq_v = k_eq_v

    def plan(self, meta: Gemma4Meta, **_: Any) -> None:
        meta.full_plan = None

    def run(self, meta: Gemma4Meta, layer: Attention, q: torch.Tensor, out: torch.Tensor) -> None:
        p = self.pools
        kv = p.fkv[layer.index]
        ps = p.full_page_size or p.page_size
        cos, sin = self.rotary.cos[FULL], self.rotary.sin[FULL]
        lengths = meta.kv_len_dev.tolist() if meta.kv_len_dev is not None else meta.row_kv_len_host
        for r in range(meta.n_rows):
            s, n, length = int(meta.row_start_host[r]), int(meta.row_nq_host[r]), int(lengths[r])
            pages = meta.page_table[int(meta.row_slot_host[r]), :-(-length // ps)].long()
            kpos = torch.arange(length, device=q.device)
            if self.k_eq_v:
                hkv, d = kv.shape[2], kv.shape[3]
                vals = gops.kv_read(kv[pages].reshape(-1, hkv, d)[:length], q.dtype)
                keys = gops.k_from_v(vals, layer.k_norm.weight, cos[kpos], sin[kpos])
            else:
                hkv, d = kv.shape[3], kv.shape[4]
                keys = gops.kv_read(kv[:, 0][pages].reshape(-1, hkv, d)[:length], q.dtype)
                vals = gops.kv_read(kv[:, 1][pages].reshape(-1, hkv, d)[:length], q.dtype)
            qpos = torch.arange(length - n, length, device=q.device)
            out[s:s + n] = gops.attention_ref(q[s:s + n], keys, vals, qpos, kpos)
