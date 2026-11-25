"""The CUDA model runners (docs/architecture.md, "The CUDA runner"): `make_cuda_runner` builds Gemma 4's
(`gemma.GemmaRunner`) for a Gemma 4 checkpoint, else the Qwen runner (`runner.CudaRunner`). Both also run on CPU with
the torch reference kernels, which is how the tests drive them on a laptop."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...config import Config
    from ..base import Runner


def make_cuda_runner(cfg: Config) -> Runner:
    import torch

    from ...models.gemma4.model import is_gemma4
    from .images import VisionOptions
    from .runner import CudaRunner

    o, c, dsp = cfg.dev.runner.cuda, cfg.runner.cuda, cfg.dev.speculative
    dtype = None if o.dtype == "auto" else getattr(torch, o.dtype)
    im, dim = cfg.images, cfg.dev.images
    vision = VisionOptions(encoder_rows=dim.encoder_cache_tokens or 0, encoder_budget=dim.encoder_budget or 0,
                           arena=cfg.pixel_arena, max_soft_tokens=im.max_soft_tokens) if im.enabled else None
    if is_gemma4(cfg.model.path):
        from ...kernels.attn512 import PREFILL_KEYS
        from ...kernels.attn512 import SPLIT_KEYS as FULL_SPLIT_KEYS
        from ...kernels.attn_window import SPLIT_KEYS
        from .gemma import GemmaRunner

        sp = cfg.speculative
        inv = bool(o.batch_invariant)  # batch invariance: each part's switch of the runner, together
        return GemmaRunner(cfg.model.path, device=o.device, dtype=dtype,
                           attention=o.attention,
                           max_num_seqs=cfg.scheduler.max_num_seqs,
                           max_num_batched_tokens=cfg.scheduler.max_num_batched_tokens,
                           max_model_len=cfg.model.max_model_len, memory_utilization=c.memory_utilization,
                           memory_gb=o.memory_gb, check=o.check, graphs=o.graphs, fused_ops=o.fused_ops,
                           decode_buckets=o.decode_buckets, token_buckets=o.token_buckets,
                           grammar=bool(cfg.structured_output.enabled), vision=vision,
                           vision_buckets=o.vision_buckets, fp8_gemm=o.fp8_gemm, fp8_cold_prefill=c.fp8_cold_prefill,
                           mtp_k=sp.num_speculative_tokens or 0, assistant=sp.model, draft_vocab=dsp.draft_vocab or 0,
                           sparse_head=bool(o.sparse_head),
                           sparse_head_rows=o.sparse_head_rows or 1024, draft_universe=bool(dsp.draft_universe),
                           draft_query_next=dsp.draft_query_next, pdl=bool(o.pdl), always_split=bool(o.always_split),
                           attn512_split_keys=FULL_SPLIT_KEYS if inv else 0, attn512_prefill=(
                               (o.prefill_split_from, PREFILL_KEYS) if o.prefill_split_from else None),
                           window_split_keys=SPLIT_KEYS if inv else 0, head_fixed_m=inv, encode_per_image=inv,
                           batch_invariant=inv, sync_check=o.sync_check, memory_history=o.memory_history,
                           logprobs=cfg.server.logprobs, penalties=cfg.server.penalties,
                           draft_counts=tuple(cfg.draft_counts), kv_cache_dtype=cfg.cache.kv_cache_dtype)
    if cfg.cache.kv_cache_dtype != "auto":  # config.derive refuses it first
        raise ValueError(f"cache.kv_cache_dtype {cfg.cache.kv_cache_dtype}: Gemma 4's full layers only")
    from ...kernels.attn_window import FULL_PREFILL_KEYS

    return CudaRunner(cfg.model.path, device=o.device, dtype=dtype, attention=o.attention, gdn=o.gdn,
                      max_num_seqs=cfg.scheduler.max_num_seqs,
                      max_num_batched_tokens=cfg.scheduler.max_num_batched_tokens,
                      max_model_len=cfg.model.max_model_len, memory_utilization=c.memory_utilization,
                      memory_gb=o.memory_gb, check=o.check, graphs=o.graphs, fused_ops=o.fused_ops,
                      decode_buckets=o.decode_buckets, token_buckets=o.token_buckets, fp8_gemm=o.fp8_gemm,
                      fp8_cold_prefill=c.fp8_cold_prefill,
                      mtp_k=cfg.speculative.num_speculative_tokens or 0,
                      draft_vocab=dsp.draft_vocab or 0,
                      vision=vision,
                      vision_buckets=o.vision_buckets, grammar=bool(cfg.structured_output.enabled),
                      pdl=bool(o.pdl), always_split=bool(o.always_split), sparse_head=bool(o.sparse_head),
                      sparse_head_rows=o.sparse_head_rows or 1024, draft_universe=bool(dsp.draft_universe),
                      batch_invariant=bool(o.batch_invariant), attn_prefill=(
                          (o.prefill_split_from, FULL_PREFILL_KEYS) if o.prefill_split_from else None),
                      sync_check=o.sync_check, logprobs=cfg.server.logprobs,
                      penalties=cfg.server.penalties)
