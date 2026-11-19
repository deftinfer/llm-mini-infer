"""`GemmaRunner`: Gemma 4 dense models (text, and images with `VisionOptions`) in BF16 or block-FP8 behind
`runner.base.Runner`, with CUDA graphs and MTP with the assistant model (docs/architecture.md, "The CUDA runner",
"Sliding-window models", "The step plan").

The step path is the Qwen runner's without what only hybrid models need (state slots, checkpoints, GDN) and with Gemma
4's KV cache. From `runner.cuda.runner` it imports the staging, the static block, `mtp_verify` and the timing and step
helpers; the sampler, the graph capture, the FP8 path, the image path and the grammar-aware heads are the shared
`runner.cuda` modules.

- **One KV arena** (`_Arena`) holds both kinds of page. The slot tables (`_Tables`) hold a page table per kind
  (`page_table` for the full layers, per sub-page of the arena; `wpage_table` for the sliding layers' window pages),
  `seq_len`, the pending token and the sampling parameters. Slot 0 is padding: both rows stay zero (the null pages).
  FREE clears both; window PAGES entries of 0 clear released pages.
- **Plan effects** (eager, before the body): FREE, ADMIT, PAGES of both kinds, KV_COPY of both, ENCODE. STATE_COPY,
  CKPT and FOLD are refused: window models resume from KV alone (their checkpoints are scheduler-side windows), and
  the assistant's rollback is a truncation.
- **Attention.** The sliding layers run FlashInfer with `window_left` (`kernels.flashinfer.FlashInferWindowAttention`),
  or under batch invariance our `kernels.attn_window.WindowAttention`; the full layers run `kernels.attn512.Attn512` (K
  derived from the stored V plane). Image spans attend bidirectionally in the sliding layers (the plan's SPANS,
  `models.gemma4.backend`). The plans come from the host mirror's lengths: exact without MTP, upper bounds under it,
  whose true lengths the kernels read from device tables (`_fix_target`, `_drafter_meta`). On CPU both are the torch
  references.
- **The body** (`_body`) runs eagerly, as a full CUDA graph per decode bucket, or as a breakable graph per token bucket
  (both attentions as eager breaks); the logits are softcapped (`models.gemma4.model.softcap`), and with
  logprobs kept for the LOGPROBS rows' pass after the sampling (`runner.cuda.logprobs`, the Qwen runner's too).
- **Grammar masks** split a step into its forward and its sampling, each with graphs of its own. Unlike the Qwen
  runner, this one reports a split step's forward end (`forward_pending`, `Capabilities.forward_end`) and profiles a
  lone masked decode step's sampling and launch, for `engine.plan_at_boundary`.
- **NaN on rows that do not sample.** An explicit row that does not sample writes its KV into the shared cache all the
  same, so outside the decode graphs the body flags each row whose last token's final hidden state is not finite
  (`_check_rows`), and `wait` reports NONFINITE on it. A NaN anywhere in the KV a row writes reaches that state on a
  BF16 or PRECISE row: the position's own query reads its key and value, and every later full layer reads every earlier
  position (Gemma 4's last layer is one). On an FP8 row the activation quantizer turns a NaN into −448 at the next
  projection (`kernels.fp8`), so neither this check nor the sampler's sees it.

**MTP with the assistant model** (`mtp_k` k > 0, `assistant`: `models.gemma4.assistant`; architecture.md,
"Speculative decoding"). Verification is `runner.cuda.runner.mtp_verify`. Rollback is a truncation: the window's KV at
rejected positions stays in the pages past `seq_len`, which no query reads and the next window overwrites. The drafter
runs after the verification, inside the same body (`_draft`): k assistant steps from each sample row's last committed
position p, all at RoPE position p (p + 1 with `draft_query_next`), over the target's KV of [0, p], with attention plans
of its own (`_drafter_meta`).
"""

from __future__ import annotations

import gc
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ...config import FP32_KERNELS
from ...kernels import build as kbuild
from ...kernels import ops
from ...models.gemma4.backend import (
    Gemma4Context,
    Gemma4Meta,
    Gemma4Pools,
    TorchFullAttention,
    TorchSlidingAttention,
)
from ...models.gemma4.model import FULL, SLIDING, Gemma4Config, Rotary, build_model, linear, softcap
from ...sched.plan import (
    ADMIT_LOGPROBS,
    LOGPROBS,
    MASKED,
    PRECISE,
    SAMPLE,
    STATE_ZERO,
    STOP_W,
    PlanBuilder,
    StepPlan,
    page_write_capacity,
    reserve_positions,
    window_page_write_capacity,
)
from ..base import (
    NONFINITE,
    AttentionCounters,
    Capabilities,
    HeadCounters,
    PlanError,
    PoolConfig,
    Runner,
    StepResult,
    sum_attention_counters,
    vision_waits,
)
from ..mtp import MAX_GEMMA_MTP_K
from . import memory_report
from . import sync_check as sync_checks
from .draft_head import DraftHead, special_token_ids, structural_token_ids
from .fp8 import Fp8Runtime
from .graphs import BreakableCapture, graph_name, graph_names, graph_summary, no_gc, parse_graph_name, warm_libraries
from .heads import NO_HEADS, VARIANT_HEADS, GrammarSampling, Heads
from .images import ImageEncoder, VisionOptions
from .logprobs import CHUNK as LP_CHUNK
from .logprobs import LogprobsPass
from .runner import (
    _HOOK_CHUNK,
    _LEAD_CYCLES,
    _NO_SAMPLE,
    _SAMPLE,
    _VERIFY,
    CPU_POOL_BYTES,
    PROFILE_MARGIN_BYTES,
    LogitsChunk,
    _bucket,
    _CpuEvent,
    _Staging,
    _Static,
    check_free_memory,
    check_mirror,
    default_decode_buckets,
    default_token_buckets,
    forward_running,
    mtp_verify,
    plan_bracket,
    require_cuda_device,
    split_step,
    step_rows,
    step_timing,
)
from .sampler import draw, host_plan

log = logging.getLogger(__name__)


class _Arena:
    """The KV arena on the device (architecture.md, "Sliding-window models"): one buffer whose pages hold either kind of
    KV page, laid out layer-major so that each kind keeps the layout its kernels read.

    The buffer is [n_sliding, num_pages, chunk]: chunk = one sliding layer's share of a window page (its K and V planes
    of `window_page_size` positions). The window pages are the buffer viewed as [n_sliding, num_pages, 2, ps, Hs, Ds]:
    page p of sliding layer l is chunk p of slab l, FlashInfer's contiguous per-layer layout. A chunk also holds `sub`
    positions of one full layer's KV (`sub` = chunk / that layer's bytes per position: 64 on Gemma 4 31B), and the slabs
    divide among the full layers, `groups` = n_sliding / n_full each (5 there): full layer f owns slabs
    [f·groups, (f + 1)·groups), viewed as [groups·num_pages, sub, Hf, Df], a paged cache of `sub`-position sub-pages
    whose sub-page j·num_pages + p is chunk p of its slab j. A full page p of the scheduler (`page_size` = groups·sub
    positions: 320) is those `groups` sub-pages, chunk p of every slab: the same bytes as window page p. The full
    layers' page table is therefore kept per sub-page (`expand`), and our kernels read it as a page size of `sub`."""

    def __init__(self, cfg: Gemma4Config, window_page_size: int, num_pages: int, dtype: torch.dtype,
                 device: torch.device):
        self.groups, self.sub, self.chunk = self.layout(cfg, window_page_size)
        self.num_pages, self.window_page_size = num_pages, window_page_size
        self.page_size = self.groups * self.sub
        ns, nf = len(cfg.sliding_layers), len(cfg.full_layers)
        self.mem = torch.zeros((ns, num_pages, self.chunk), dtype=dtype, device=device)
        self.wkv = self.mem.view(ns, num_pages, 2, window_page_size, cfg.sliding_kv_heads, cfg.sliding_head_dim)
        shape = (self.sub, cfg.full_kv_heads, cfg.full_head_dim) if cfg.k_eq_v else \
            (2, self.sub, cfg.full_kv_heads, cfg.full_head_dim)
        self.fkv = self.mem.view(nf, self.groups * num_pages, *shape)

    @staticmethod
    def layout(cfg: Gemma4Config, window_page_size: int) -> tuple[int, int, int]:
        """(groups, sub, chunk elements): see the class docstring; raises when the two kinds cannot share pages."""
        ns, nf = len(cfg.sliding_layers), len(cfg.full_layers)
        chunk = 2 * window_page_size * cfg.sliding_kv_heads * cfg.sliding_head_dim
        per_pos = (1 if cfg.k_eq_v else 2) * cfg.full_kv_heads * cfg.full_head_dim
        if not ns or not nf or ns % nf or chunk % per_pos:
            raise ValueError(f"one KV arena needs the sliding layers ({ns}) to divide among the full layers ({nf}) and "
                             f"a window page's share of a sliding layer ({chunk} elements) to hold whole positions of "
                             f"a full layer ({per_pos} elements each)")
        return ns // nf, chunk // per_pos, chunk

    @staticmethod
    def full_page_size(cfg: Gemma4Config, window_page_size: int) -> int:
        groups, sub, _ = _Arena.layout(cfg, window_page_size)
        return groups * sub

    def expand(self, idx: np.ndarray, page: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Full PAGES entries (index, page) -> the sub-page table's: (index·groups + j, j·num_pages + page), j <
        groups."""
        g = self.groups
        j = np.arange(g, dtype=np.int64)
        return (np.asarray(idx, np.int64)[:, None] * g + j).reshape(-1), \
            (j * self.num_pages + np.asarray(page, np.int64)[:, None]).reshape(-1)

    def copy(self, src: torch.Tensor, dst: torch.Tensor) -> None:
        """Whole pages src -> dst, either kind (every slab's chunk; gather, then scatter)."""
        self.mem.index_copy_(1, dst, self.mem.index_select(1, src))


class _Tables:
    def __init__(self, slots: int, max_fpages: int, max_wpages: int, device: torch.device, k: int = 0):
        self.page_table = torch.zeros((slots, max_fpages), dtype=torch.int32, device=device)  # full: per sub-page
        self.wpage_table = torch.zeros((slots, max_wpages), dtype=torch.int32, device=device)
        self.seq_len = torch.zeros(slots, dtype=torch.int64, device=device)
        self.pending = torch.full((slots,), -1, dtype=torch.int64, device=device)
        self.temperature = torch.zeros(slots, dtype=torch.float32, device=device)
        self.top_p = torch.ones(slots, dtype=torch.float32, device=device)
        self.min_p = torch.zeros(slots, dtype=torch.float32, device=device)
        self.top_k = torch.zeros(slots, dtype=torch.int64, device=device)
        self.seed = torch.zeros(slots, dtype=torch.int64, device=device)
        self.k = k
        if k:  # MTP: the drafts each slot's next decode row verifies, and the commit limits (ADMIT)
            self.drafts = torch.zeros((slots, k), dtype=torch.int64, device=device)
            self.stop = torch.full((slots, STOP_W), -1, dtype=torch.int64, device=device)
            self.max_len = torch.zeros(slots, dtype=torch.int64, device=device)

    def reset(self) -> None:
        self.page_table.zero_()
        self.wpage_table.zero_()
        self.seq_len.zero_()
        self.pending.fill_(-1)
        self.temperature.zero_()
        if self.k:
            self.drafts.zero_()
            self.stop.fill_(-1)


class _Mirror:
    """Host mirrors of what the plans wrote. `seq_len` is exact without MTP; under MTP the device seq_len is in
    [seq_len, seq_len + slack] (`runner.base`)."""

    def __init__(self, slots: int):
        self.admitted = np.zeros(slots, bool)
        self.sampled = np.zeros(slots, bool)
        self.seq_len = np.zeros(slots, np.int64)
        self.slack = np.zeros(slots, np.int64)
        self.epoch = np.zeros(slots, np.int64)  # admissions of the slot so far: results of older ones are stale
        self.temperature = np.zeros(slots, np.float32)
        self.top_p = np.ones(slots, np.float32)
        self.top_k = np.zeros(slots, np.int64)
        self.min_p = np.zeros(slots, np.float32)
        self.lp = np.zeros(slots, bool)  # admitted with ADMIT_LOGPROBS


class _Effects:
    __slots__ = ("n_admit", "n_free", "n_kvc", "n_pages", "n_wpages", "o_adm", "o_admseq", "o_f", "o_free", "o_kvc",
                 "o_maxlen", "o_pid", "o_pidx", "o_pslot", "o_seed", "o_stop", "o_topk", "o_wid", "o_widx", "o_wslot")


_EMPTY = np.zeros(0, np.int64)


class _Layout:
    __slots__ = ("T", "c_row", "constrained", "d", "dplan", "dplan_uni", "dst", "emb", "full_rows", "g_first", "lp",
                 "max_q", "n", "n_dec", "n_dev", "n_masks", "n_mtp", "n_precise", "n_sample", "n_stok", "o_twb",
                 "order", "r_kind", "r_kv", "r_kv_lb", "r_nq", "r_pos0", "r_s0", "r_slot", "s_idx", "s_mask",
                 "s_plan_rows", "s_slot", "spans", "splan", "splan_sp", "starts", "tok_in", "tok_plan_row", "tok_pos",
                 "w", "wlo")


class _Handle:
    __slots__ = ("checks", "enc_events", "events", "fwd_end", "internal", "lp", "mtp_rows", "n_sample", "n_sched",
                 "observe",
                 "parity", "plan_events", "prev_end", "result", "s_plan_rows", "split", "stage", "step_id")


class _GStaging(_Staging):
    """The Qwen runner's staging with room for the attention plans' int32 tables (rows, work items)."""

    def __init__(self, *args: Any, n32: int, **kw: Any):
        super().__init__(*args, **kw)
        pin = self.h32.is_pinned()
        self.h32 = torch.empty(n32, dtype=torch.int32, pin_memory=pin)
        self.n32 = self.h32.numpy()


class GemmaRunner(Runner):
    def __init__(self, model_dir: str | Path, *, device: str = "cuda", dtype: torch.dtype | None = None,
                 attention: str = "auto", max_num_seqs: int = 64, max_num_batched_tokens: int = 2048,
                 max_model_len: int | None = None, memory_utilization: float = 0.9, memory_gb: float | None = None,
                 check: bool = False, graphs: bool = True, fused_ops: bool = True,
                 decode_buckets: list[int] | None = None, token_buckets: list[int] | None = None,
                 grammar: bool = True, vision: VisionOptions | None = None, vision_buckets: list[int] | None = None,
                 fp8_gemm: str = "auto", fp8_cold_prefill: str = "reuse", mtp_k: int = 0,
                 assistant: str | Path | None = None, draft_vocab: int = 0, draft_fp8: bool = False,
                 sparse_head: bool = False, sparse_head_rows: int = 1024, draft_universe: bool = False,
                 draft_query_next: bool = False, pdl: bool = False, always_split: bool = False,
                 attn512_split_keys: int = 0,
                 head_fixed_m: bool = False, encode_per_image: bool = False, window_split_keys: int = 0,
                 batch_invariant: bool = False, sync_check: str | None = None, memory_history: str | None = None,
                 logprobs: bool = False):
        """`mtp_k` > 0 drafts that many tokens per step with the MTP `assistant` (its checkpoint directory;
        `draft_vocab`, `draft_fp8`: its head, `runner.cuda.draft_head`). `sparse_head`, `sparse_head_rows`,
        `draft_universe`: the grammar-aware heads (`runner.cuda.heads`; with `grammar` only, the universe with MTP).
        `draft_query_next`: the assistant's queries at RoPE position p + 1 instead of p (the module docstring). `pdl`:
        programmatic dependent launch. `always_split` (with `grammar`): every step runs as a forward and a
        sampling, masked or not, and only those graphs are captured. `attn512_split_keys`: the full layers' fixed split
        (`kernels.attn512`; 0: the wave split). `head_fixed_m`: the sparse head's GEMM at one M for every step (its
        `max_num_seqs` × (1 + k) rows). `encode_per_image`: one tower call per image (`runner.cuda.images`).
        `window_split_keys`: the sliding layers on our kernel, decode rows split by that many keys, image spans in its
        prefill rows' pass (`kernels.attn_window`; 0: FlashInfer's). `batch_invariant`: reported to the scheduler, whose
        final chunks then take their forced prefix whole (the parts above are the runner's; `make_cuda_runner` sets them
        together). `sync_check`: off, warn or error around the step's launches (`runner.cuda.sync_check`; None: its
        default). `memory_history`: a file for the snapshot of an eager tower call after startup (`memory_report`).
        `logprobs`: takes LOGPROBS rows, its head's logits kept for them (`runner.cuda.logprobs`)."""
        if always_split and not grammar:
            raise ValueError("always-split graphs need the grammar masks' split (structured output)")
        self.device = torch.device(device)
        self.cuda = self.device.type == "cuda"
        if self.cuda:
            require_cuda_device()
            if self.device.index is None:
                self.device = torch.device("cuda", torch.cuda.current_device())
            torch.cuda.set_device(self.device)
            if memory_gb is None:
                check_free_memory(*torch.cuda.mem_get_info(self.device), memory_utilization, str(self.device))
        self.dtype = dtype or (torch.bfloat16 if self.cuda else torch.float32)
        self.model_dir = Path(model_dir)
        self.cfg = cfg = Gemma4Config.from_dir(self.model_dir)
        self.window = cfg.sliding_window
        self.S, self.B = max_num_seqs, max_num_batched_tokens
        self.logprobs = logprobs
        self.lp: LogprobsPass | None = None
        if mtp_k and assistant is None:
            raise ValueError("MTP with Gemma 4 needs its assistant model (speculative.model)")
        if not 0 <= mtp_k <= MAX_GEMMA_MTP_K:
            raise ValueError(f"Gemma 4's MTP drafts at most {MAX_GEMMA_MTP_K} tokens (speculative."
                             f"num_speculative_tokens = {mtp_k})")
        self.k, self.L = mtp_k, 1 + mtp_k
        self.draft_next = draft_query_next and mtp_k > 0
        self.grammar = grammar
        self.always_split = always_split
        self.attn512_split_keys = attn512_split_keys
        self.window_split_keys = window_split_keys
        self.inv = batch_invariant
        self.max_model_len = min(max_model_len or cfg.max_position_embeddings, cfg.max_position_embeddings)
        self.memory_utilization = memory_utilization
        self.memory_gb = memory_gb
        self.memory_history = memory_history
        self._released: int | None = None  # reserved but not allocated after the memory profile's release
        self.check = check
        ok = cfg.sliding_head_dim in (64, 128, 256) and cfg.full_head_dim == 512 and cfg.k_eq_v and \
            16 % (cfg.num_heads // cfg.full_kv_heads) == 0
        if attention == "auto":
            attention = "flashinfer" if self.cuda and ok else "torch"
        elif attention == "flashinfer" and not self.cuda:
            raise ValueError("dev.runner.cuda.attention flashinfer needs a CUDA device")
        elif attention == "flashinfer" and not ok:
            raise ValueError("Gemma 4 on CUDA needs sliding heads of 64, 128 or 256 (what this runner passes to "
                             "FlashInfer) and k_eq_v full heads of 512 with 1, 2, 4, 8 or 16 query heads per KV head "
                             "(our kernel)")
        self.attn_kind = attention
        t0 = time.perf_counter()
        ops.enabled(fused_ops)
        if self.cuda:  # the sampler has no torch fallback on CUDA: fail before the weights load
            kbuild.require("Gemma 4's CUDA runner")
        kernels = self.cuda and fused_ops
        if kernels and self.dtype != torch.bfloat16:
            raise ValueError(FP32_KERNELS)
        # before any capture (the graphs keep the launches they captured); a CPU runner leaves the library alone
        self.pdl = ops.set_pdl(pdl and kernels) if self.cuda else False
        self.stats: dict[str, Any] = {"kernels": kbuild.stats.get("state", "unused") if self.cuda else "cpu",
                                      "pdl": self.pdl, "always_split": always_split}
        if self.cuda and attention == "flashinfer" and not kernels:
            raise RuntimeError("Gemma 4's full attention on CUDA is our attn512 kernel: it needs "
                               "dev.runner.cuda.fused_ops (or dev.runner.cuda.attention torch)")
        self.use_graphs = graphs and kernels and self.attn_kind == "flashinfer"
        self._sync = sync_checks.make(sync_check, cuda=self.cuda, check=check, reference=(
            "the torch references (dev.runner.cuda.fused_ops off)" if not kernels else
            "torch attention" if self.attn_kind != "flashinfer" else "the torch FP8 GEMM" if fp8_gemm == "torch" else
            None))
        self.fp8 = Fp8Runtime(cfg.quant is not None, cuda=self.cuda, kernels=kernels, dtype=self.dtype,
                              fp8_gemm=fp8_gemm, fp8_cold_prefill=fp8_cold_prefill, stats=self.stats)
        rows = min(max_num_seqs, max_num_batched_tokens // self.L)  # a decode step's tokens are within the budget
        self.decode_buckets = sorted(b for b in (decode_buckets or default_decode_buckets(rows)) if 0 < b <= rows)
        self.token_buckets = sorted(b for b in (token_buckets or default_token_buckets(max_num_batched_tokens))
                                    if 0 < b <= max_num_batched_tokens)
        t1 = time.perf_counter()
        self.model = build_model(self.model_dir, self.device, self.dtype)
        if self.cuda:
            torch.cuda.synchronize()
        n_fp8 = self.fp8.attach(self.model, self.device)  # before the memory profile: the PRECISE rows' buffer
        self.assistant = None
        self.draft_head: DraftHead | None = None
        if self.k:  # the MTP assistant: BF16, each layer on the target's last layer of its kind
            from ...models.gemma4.assistant import build_assistant

            self.assistant = build_assistant(assistant, cfg, self.device, self.dtype)
            self.stats["assistant_layers_shared"] = self.assistant.share(self.model.layers, cfg)
            self.stats["assistant_weight_bytes"] = sum(p.numel() * p.element_size()
                                                       for p in self.assistant.parameters())
            if draft_fp8 and self.cuda:  # an FP8 draft head's GEMMs
                kbuild.require_cutlass("dev.speculative.draft_head_dtype bfloat16")
            if self.cuda:  # fresh segments for the head (runner.py: the loader's cached ones would stay pinned)
                torch.cuda.empty_cache()
            forced = special_token_ids(self.model_dir)
            if grammar and 0 < draft_vocab < cfg.vocab_size:  # JSON's whitespace and punctuation (draft_head.py)
                forced += structural_token_ids(self.model_dir)
            self.draft_head = DraftHead(self.assistant.head_weight, size=draft_vocab, fp8_copy=draft_fp8,
                                        special=forced)
            self.stats.update(draft_head=self.draft_head.describe(), draft_head_bytes=self.draft_head.bytes)
            if self.assistant.cfg.final_logit_softcapping:
                raise NotImplementedError("an assistant head with final logit softcapping")
        self.gs = gs = GrammarSampling(grammar=grammar, sparse=sparse_head, universe=draft_universe,
                                       rows=sparse_head_rows, vocab=cfg.vocab_size, head=self.model.lm_head_weight,
                                       draft_weight=self.assistant.head_weight if self.k else None, gemm=linear,
                                       finish=lambda y: softcap(y, cfg.final_logit_softcapping),
                                       fixed_m=head_fixed_m)
        self.stats.update(sparse_head=gs.sparse, draft_universe=gs.universe, sparse_head_rows=gs.rows)
        self.stats.update({
            "load_s": round(time.perf_counter() - t1, 2), "init_s": round(t1 - t0, 2),
            "weight_bytes": sum(p.numel() * p.element_size() for p in self.model.parameters()),
            "attention": self.attn_kind, "dtype": str(self.dtype).removeprefix("torch."), "graphs": self.use_graphs,
            "mtp_k": self.k, "window": self.window, "draft_query_next": self.draft_next})
        log.info("model loaded in %.1f s: %.2f GiB of weights (%s%s), attention %s (sliding window %d), CUDA graphs "
                 "%s%s",
                 self.stats["load_s"], self.stats["weight_bytes"] / 2**30, self.stats["dtype"],
                 f", {n_fp8} block-FP8 projections" if n_fp8 else "", self.attn_kind, self.window,
                 "on" if self.use_graphs else "off",
                 f", MTP k = {self.k} with the assistant ({self.stats['assistant_weight_bytes'] / 2**30:.2f} GiB, its "
                 f"layers on the target's layers {sorted(set(self.stats['assistant_layers_shared']))}), draft head: "
                 f"{self.stats['draft_head']}{', queries at p + 1' if self.draft_next else ''}" if self.k else "")
        self.fp8.log_setup()
        if gs.heads is not None:
            log.info(gs.describe())
        # Images (`runner.cuda.images`): spans attend bidirectionally in the sliding layers when the checkpoint says so
        self.mm = vision is not None
        self.bidi = self.mm and cfg.bidirectional == "vision"
        self.img: ImageEncoder | None = None
        if self.mm:
            self.img = ImageEncoder(self.model_dir, vision, self.device, self.dtype, buckets=vision_buckets)
            self.img.per_image = encode_per_image
            self.stats.update(vision_load_s=self.img.load_s, vision_weight_bytes=self.img.weight_bytes)
            log.info(self.img.describe(self.use_graphs))
        self.copy_stream = torch.cuda.Stream(self.device) if self.cuda else None
        self.logits_hook: Callable[[LogitsChunk], None] | None = None
        self._caps: Capabilities | None = None
        self._ready = False
        self._allocated = False
        self._pool: Any = None
        self._graphs: dict[tuple[str, int], BreakableCapture] = {}

    # -- geometry and memory ---------------------------------------------------------------------------------

    @property
    def kv_bytes_per_token(self) -> int:
        return self.cfg.kv_bytes_per_token(FULL, self.dtype.itemsize)

    @property
    def window_kv_bytes_per_token(self) -> int:
        return self.cfg.kv_bytes_per_token(SLIDING, self.dtype.itemsize)

    def capabilities(self) -> Capabilities:
        if self._caps is None:
            total, free = self._measure_memory()
            name = torch.cuda.get_device_name(self.device) if self.cuda else "cpu (torch reference kernels)"
            self._caps = Capabilities(
                backend="cuda", device=name, vocab_size=self.cfg.vocab_size,
                max_positions=self.cfg.max_position_embeddings, memory_total_bytes=total, memory_free_bytes=free,
                kv_bytes_per_token=self.kv_bytes_per_token, state_slot_bytes=0, window=self.window,
                window_kv_bytes_per_token=self.window_kv_bytes_per_token,
                page_size_multiple=16 if self.window_split_keys and self.attn_kind == "flashinfer" else 1,  # KV tile
                checkpoint_grid=16, max_steps_in_flight=2, mtp_depth=self.k, draft_kv=False,
                cuda_graphs=self.use_graphs,
                checkpoints="window", images=self.mm, grammar_masks=self.grammar,
                encoder_rows=self.img.rows if self.mm else 0, encoder_budget=self.img.budget if self.mm else 0,
                bidirectional_images=self.bidi,
                decode_step_ms=self.stats.get("decode_step_ms", 0.0),
                decode_step_rows=self.stats.get("decode_step_rows", 0),
                prefill_step_ms=self.stats.get("prefill_step_ms", 0.0),
                prefill_step_tokens=self.stats.get("prefill_step_tokens", 0), precise_rows=self.fp8.precise,
                precise_step_ms=self.stats.get("precise_step_ms", 0.0), forward_end=self.grammar and self.cuda,
                sample_step_ms=self.stats.get("sample_step_ms", 0.0),
                launch_step_ms=self.stats.get("launch_step_ms", 0.0), batch_invariant=self.inv,
                logprobs=self.logprobs)
        return self._caps

    def attention_counters(self) -> AttentionCounters:
        """FlashInfer's plan counters (`Runner.attention_counters`) over the sliding layers' instances: the target's
        (`attn_w`), the drafter's (`attn_dw`, MTP) and the image spans' (`attn_wb`); none before `configure` or with
        the torch references. The full layers' attn512 never plans again (an empty split merges harmlessly). With them,
        the vision tower's plan waits (`vision_waits`)."""
        names = ("attn_w", "attn_dw", "attn_wb")
        return vision_waits(sum_attention_counters(getattr(self, n, None) for n in names), getattr(self, "img", None))

    def _measure_memory(self) -> tuple[int, int]:
        """(total, free for the KV arena): as the Qwen runner's (`runner.cuda.runner`)."""
        if not self.cuda:
            gb = self.memory_gb
            n = int(gb * 2**30) if gb is not None else CPU_POOL_BYTES
            return n, n
        if self.memory_gb is not None:
            return torch.cuda.mem_get_info(self.device)[1], int(self.memory_gb * 2**30)
        t0 = time.perf_counter()
        peak, persistent, graph_bytes = self._profile()
        free, total = torch.cuda.mem_get_info(self.device)
        avail = int(total * self.memory_utilization) - (total - free) - persistent - graph_bytes - peak - \
            PROFILE_MARGIN_BYTES
        self.stats.update(profile_s=round(time.perf_counter() - t0, 2), activation_peak_bytes=peak,
                          persistent_bytes=persistent, graph_pool_bytes=graph_bytes, used_after_load_bytes=total - free,
                          pool_bytes=avail)
        log.info("memory: %.2f GiB total, %.2f GiB in use after loading, activation peak %.2f GiB at %d tokens, "
                 "%.2f GiB of persistent buffers, %.2f GiB of CUDA graphs -> %.2f GiB for the KV arena "
                 "(utilisation %.2f)", total / 2**30, (total - free) / 2**30, peak / 2**30, self.B,
                 persistent / 2**30, graph_bytes / 2**30, avail / 2**30, self.memory_utilization)
        if avail <= 0:
            raise RuntimeError("no device memory left for the KV arena")
        return total, avail

    @torch.inference_mode()
    def _profile(self) -> tuple[int, int, int]:
        """Tiny pools, the CUDA graphs and representative steps: (activation peak, persistent bytes, graph pool)."""
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        s, b, wps = self.S, self.B, 16
        fps = _Arena.full_page_size(self.cfg, wps)
        rows = max(1, min(s, b // 4))
        per = b // rows

        def need(n: int) -> int:  # window and full pages of n positions
            return -(-n // wps) + -(-n // fps)

        full = [(i + 1, n) for i, n in enumerate(step_rows(b, self.max_model_len, s))]  # a full-budget step
        pages = max(sum(need(n + 2 + reserve_positions(2, self.k)) for _, n in full),
                    rows * need(per + 2 + 6 * self.L), s * 4) + 2
        pc = PoolConfig(page_size=fps, num_pages=pages, num_state_slots=s + 1, max_num_seqs=s,
                        max_num_batched_tokens=b, max_model_len=self.max_model_len, eos_token_ids=(),
                        mtp_k=self.k, window_page_size=wps)
        base = torch.cuda.memory_allocated(self.device)
        self._allocate(pc)
        pools = self.arena.mem.numel() * self.arena.mem.element_size()
        allocated = torch.cuda.memory_allocated(self.device)
        persistent = allocated - base - pools
        graph_bytes = 0
        t0 = time.perf_counter()
        if self.use_graphs:
            self._run([lambda: self._synthetic(self._builder(), 1, [(1, min(b, 80))], pages_from=1)])
            self._reset_tables()
            torch.cuda.synchronize()
            self.stats["profile_first_calls_s"] = round(time.perf_counter() - t0, 2)
            t0 = time.perf_counter()
            r0 = torch.cuda.memory_reserved(self.device)
            self._capture_all()
            torch.cuda.synchronize()
            graph_bytes = torch.cuda.memory_reserved(self.device) - r0
            allocated = torch.cuda.memory_allocated(self.device)
            self.stats["profile_capture_s"] = round(time.perf_counter() - t0, 2)
            t0 = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(self.device)
        builder = self._builder()
        if self.mm:  # the encoder at its budget (one image of that many tokens) and a row reading it
            self._run([lambda: self._image_plan(builder, 1, slot=1, image_tokens=min(self.img.budget, b - 2),
                                                pages_from=1)])
            self._reset_tables()
        drows = min(rows, b // self.L)
        pr = self.fp8.precise  # PRECISE rows possible (the precise prefix): their BF16 activations are the peak
        self._run([lambda: self._synthetic(builder, 1, full, pages_from=1, precise=pr),
                   lambda: self._synthetic(builder, 2, [(r + 1, per) for r in range(rows)], pages_from=1,
                                           free=[r[0] for r in full], params=(1.0, 0.95, 0, 0.0), room=4 * self.L,
                                           precise=pr),
                   lambda: self._decode_plan(builder, 3, list(range(1, drows + 1)))])
        torch.cuda.synchronize()
        self.stats["profile_steps_s"] = round(time.perf_counter() - t0, 2)
        peak = max(0, torch.cuda.max_memory_allocated(self.device) - allocated)
        with self.fp8.timing():  # the step costs the scheduler derives its budget from: the FP8 path
            res = self._run([lambda: self._synthetic(builder, 4, [(1, per)], pages_from=1,
                                                     free=list(range(1, rows + 1)), room=4 * self.L),
                             lambda: self._decode_plan(builder, 5, [1]), lambda: self._decode_plan(builder, 6, [1]),
                             lambda: self._synthetic(builder, 7, full, pages_from=1, free=[1]),
                             lambda: self._synthetic(builder, 8, full, pages_from=1, free=[r[0] for r in full])],
                            lead=True)
        self.stats.update(decode_step_ms=round(min(r.gpu_busy_ns for r in res[1:3]) / 1e6, 3), decode_step_rows=1,
                          prefill_step_tokens=sum(r[1] for r in full),
                          prefill_step_ms=round(min(r.gpu_busy_ns for r in res[3:]) / 1e6, 3))
        if pr:  # the same full-budget prefill step with every row PRECISE (eager: the BF16 path's own cost)
            res = self._run([lambda: self._synthetic(builder, 9, full, pages_from=1, free=[r[0] for r in full],
                                                     precise=True),
                             lambda: self._synthetic(builder, 10, full, pages_from=1, free=[r[0] for r in full],
                                                     precise=True)], lead=True)
            self.stats["precise_step_ms"] = round(min(r.gpu_busy_ns for r in res) / 1e6, 3)
        if self.grammar:  # planning at the boundary: a lone masked decode step's split, its sampling on the heads a
            # constrained step takes when serving (the last variant: the sparse head and the universe when both are
            # on), each behind a short device sleep that outlasts a decode graph's launch
            hp, lm, t0 = VARIANT_HEADS[self.gs.variants[-1]], [], time.perf_counter()
            with self.fp8.timing():
                res = self._run([lambda: self._synthetic(builder, 11, [(1, per)], pages_from=1,
                                                         free=[r[0] for r in full], room=4 * self.L),
                                 *(lambda s=s: self._decode_plan(builder, s, [1], masked=True) for s in (12, 13, 14))],
                                lead=True, heads=[hp] * 3, launch_ms=lm, lead_cycles=_LEAD_CYCLES // 10)
            self.stats.update(sample_step_ms=round(min(r.gpu_sample_ns for r in res[1:]) / 1e6, 3),
                              launch_step_ms=round(float(np.median(lm[1:])), 3),
                              profile_boundary_s=round(time.perf_counter() - t0, 2))
        log.info("step costs: a lone request's decode step %.2f ms, a %d-token prefill step %.2f ms (%s)%s%s",
                 self.stats["decode_step_ms"], self.stats["prefill_step_tokens"], self.stats["prefill_step_ms"],
                 "CUDA graphs" if self.use_graphs else "eager",
                 f", with PRECISE rows {self.stats['precise_step_ms']:.2f} ms (eager)" if pr else "",
                 f"; a lone masked decode step samples in {self.stats['sample_step_ms']:.2f} ms of the device and "
                 f"launches in {self.stats['launch_step_ms']:.2f} ms of the host" if self.grammar else "")
        self._release()
        self._released = torch.cuda.memory_reserved(self.device) - torch.cuda.memory_allocated(self.device)
        return peak, max(0, persistent), graph_bytes

    # -- configure ---------------------------------------------------------------------------------------------

    def full_page_size(self, window_page_size: int) -> int:
        """The full pages' positions in an arena of window pages of `window_page_size` (as many bytes)."""
        return _Arena.full_page_size(self.cfg, window_page_size)

    def configure(self, pools: PoolConfig) -> None:
        if pools.mtp_k != self.k:
            raise ValueError(f"configured with mtp_k {pools.mtp_k}; this runner drafts {self.k}")
        if not pools.window_page_size or pools.page_size != self.full_page_size(pools.window_page_size):
            raise ValueError(f"Gemma 4's KV arena: window pages of {pools.window_page_size} positions take full pages "
                             f"of {self.full_page_size(pools.window_page_size or 16)}, not {pools.page_size}")
        t0 = time.perf_counter()
        with torch.inference_mode():
            self._allocate(pools)
            self._warmup()
            self.stats["warmup_s"] = round(time.perf_counter() - t0, 2)
            if self.use_graphs:
                t1 = time.perf_counter()
                self._capture_all()
                self.stats["capture_s"] = round(time.perf_counter() - t1, 2)
        self._ready = True
        self.stats["configure_s"] = round(time.perf_counter() - t0, 2)
        if self.cuda:
            self.stats["memory_allocated_bytes"] = torch.cuda.memory_allocated(self.device)
            self.stats["memory_reserved_bytes"] = torch.cuda.memory_reserved(self.device)
            log.info(memory_report.startup_line(self.device, self._released))
            if self.memory_history and self.mm:
                free = torch.cuda.mem_get_info(self.device)[0]
                self.stats["eager_tower"] = memory_report.eager_tower_snapshot(self.img, self.memory_history, free)
        log.info("KV arena: %d pages, window pages of %d positions or full pages of %d (sub-pages of %d); warm-up "
                 "%.1f s%s", pools.num_pages, pools.window_page_size, pools.page_size, self.arena.sub,
                 self.stats["warmup_s"],
                 f"; {len(self._graphs)} CUDA graphs captured in {self.stats['capture_s']:.1f} s"
                 if self.use_graphs else "")
        timings = {k: v for k, v in self.stats.items() if k.endswith("_s")}
        log.info("runner startup (s): %s", " ".join(f"{k.removesuffix('_s')}={v}" for k, v in timings.items()))

    def _builder(self) -> PlanBuilder:
        pc = self.pc
        return PlanBuilder(pc.max_num_seqs, pc.max_num_batched_tokens, pc.page_size, pc.max_model_len, self.k,
                           self.window, pc.window_page_size)

    def _allocate(self, pc: PoolConfig) -> None:
        if self._allocated:
            self._release()
        cfg, dev, dt = self.cfg, self.device, self.dtype
        self.pc = pc
        self.ps, self.fps = pc.window_page_size, pc.page_size  # window pages; full pages (the scheduler's)
        slots = pc.max_num_seqs + 1
        self.reserve = reserve_positions(2, self.k)
        self.arena = _Arena(cfg, self.ps, pc.num_pages, dt, dev)
        self.sub = self.arena.sub  # the full layers' table and kernels address sub-pages of this many positions
        self.max_pages = -(-(pc.max_model_len + self.reserve) // self.ps) + 1  # window pages per slot
        self.max_fpages = -(-(pc.max_model_len + self.reserve) // self.fps) + 1  # full pages per slot
        self.pools = Gemma4Pools(wkv=self.arena.wkv, fkv=self.arena.fkv, page_size=self.ps, window=self.window,
                                 full_page_size=self.sub)
        self.tables = _Tables(slots, self.max_fpages * self.arena.groups, self.max_pages, dev, self.k)
        self.mirror = _Mirror(slots)
        self.rotary = Rotary(cfg, pc.max_model_len + self.reserve + 1, dev, dt)
        L = self.L
        tmax = max(pc.max_num_batched_tokens, pc.max_num_seqs * L)
        smax = min(tmax, pc.max_num_seqs * L)
        self.static = _Static(tmax, pc.max_num_seqs, smax, L, self.k, cfg.hidden_size, dt, dev, mm=self.mm)
        # MTP: the rows' true KV lengths, decided on the device (the module docstring): the target's rows (all of a
        # row's KV, and the sliding layers' from its first window page), the drafter's rows (one per sample row)
        z = (lambda: torch.zeros(slots, dtype=torch.int64, device=dev)) if self.k else (lambda: None)
        self._t_kv, self._t_named, self._d_kv, self._d_named = z(), z(), z(), z()
        self._t_wb, self._d_wb = z(), z()  # each row's first window position (staged per step)
        writes = page_write_capacity(pc.max_num_seqs, pc.max_num_batched_tokens, self.fps, pc.max_model_len,
                                     self.reserve) * self.arena.groups + \
            window_page_write_capacity(pc.max_num_seqs, pc.max_num_batched_tokens, self.ps, self.window,
                                       self.reserve)
        cap_i = 6 * tmax + 3 * writes + 64 * slots + 1024  # (slot, index, id) per page write of either kind
        if self.mm:  # ENCODE's rows; the spans' rows, pages and query tokens
            cap_i += self.img.budget + 2 * tmax + 64
        if self.k:  # ADMIT's commit limits; the rows' first window pages; the drafter's rows
            cap_i += (STOP_W + 2) * slots + 8 * slots
        if self.logprobs:  # the LOGPROBS samples' indices
            cap_i += smax + LP_CHUNK
        groups = (-(-pc.max_num_batched_tokens * (cfg.num_heads // cfg.full_kv_heads) // 16) + slots) * \
            cfg.full_kv_heads
        n32 = 16 * slots + 64 + 8 * (slots + 1) + 4 * (groups * 32 + 4 * slots)  # FlashInfer plans; rows; items
        if self.mm:
            n32 += 4 * (tmax + 2)  # the spans' FlashInfer plan
        if self.k:  # the drafter's: a FlashInfer plan and attn512's rows and items (at most 9/5 of a wave and a margin)
            n32 += 8 * (slots + 1) + 4 * (slots + 4 * 1024)
        self._staging = [_GStaging(dev, cap_i, 3 * slots, slots, self.static.size, self.static.out.numel(), n32=n32,
                                   px_bytes=self.img.px_bytes if self.mm else 0) for _ in range(2)]
        self.gs.allocate(pc.max_num_seqs * L)
        self.lp = LogprobsPass(smax, cfg.vocab_size, self.gs.rows if self.gs.sparse else 0, dev) \
            if self.logprobs else None
        workspace = None  # FlashInfer's float workspace, shared by its instances and the tower's attention
        if self.attn_kind == "flashinfer":
            from ...kernels.attn512 import Attn512
            from ...kernels.attn_window import WindowAttention
            from ...kernels.flashinfer import WORKSPACE_BYTES, FlashInferWindowAttention

            max_rows = max(self.decode_buckets) if self.use_graphs else 0
            # a decode row's window pages at most: the window before its first query (planned from the lowest
            # possible one under MTP, which may lie k below the true one), its 1 + k queries, a page boundary
            wpages = -(-(self.window + self.ps + 2 * self.L) // self.ps) + 1
            if self.window_split_keys:  # our kernel (kernels.attn_window, image spans too); FlashInfer the drafter
                self.attn_w: Any = WindowAttention(self.pools, cfg.num_heads, cfg.sliding_kv_heads,
                                                   cfg.sliding_head_dim, dev, self.window,
                                                   max_tokens=pc.max_num_batched_tokens, max_rows=pc.max_num_seqs,
                                                   qo_per_row=L, split_keys=self.window_split_keys)
                workspace = torch.zeros(WORKSPACE_BYTES, dtype=torch.uint8, device=dev)
            else:
                self.attn_w = FlashInferWindowAttention(self.pools, cfg.num_heads, cfg.sliding_kv_heads,
                                                        cfg.sliding_head_dim, dt, dev, self.window,
                                                        max_rows=max_rows, max_pages=wpages, qo_per_row=L,
                                                        upper_bounds=bool(self.k))
                workspace = self.attn_w.workspace
            self.stats.update(window_split_keys=self.window_split_keys)
            self.attn_f: Any = Attn512(self.pools, self.rotary, cfg.num_heads, cfg.full_kv_heads,
                                       cfg.full_rope_angles, dev, max_tokens=pc.max_num_batched_tokens,
                                       max_rows=pc.max_num_seqs, qo_per_row=L, split_keys=self.attn512_split_keys,
                                       max_kv=self.max_model_len + L)
            self.stats.update(attn512_resident=self.attn_f.resident, attn512_split_keys=self.attn512_split_keys)
            log.info("attn512: %d CTAs resident at once (%d an SM)", self.attn_f.resident,
                     self.attn_f.resident // self.attn_f.sms)
            if self.k:  # the drafter's: one query per sample row, over the target's last sliding and full layers
                self.attn_dw: Any = FlashInferWindowAttention(self.pools, cfg.num_heads, cfg.sliding_kv_heads,
                                                              cfg.sliding_head_dim, dt, dev, self.window,
                                                              max_rows=max_rows, max_pages=wpages,
                                                              workspace=workspace, upper_bounds=True)
                self.attn_df: Any = Attn512(self.pools, self.rotary, cfg.num_heads, cfg.full_kv_heads,
                                            cfg.full_rope_angles, dev, max_tokens=pc.max_num_seqs,
                                            max_rows=pc.max_num_seqs, share=self.attn_f)
                from ...kernels.flashinfer import self_check

                # upper-bound planning (MTP): the target's FlashInfer, or the drafter's beside our kernel
                self.stats["flashinfer_check"] = self_check(self.attn_dw if self.window_split_keys else self.attn_w,
                                                            dt)
            # image spans: the sliding layers' second plan, without the causal mask (its own plan buffers); our kernel
            # takes them in its pass
            self.attn_wb: Any = FlashInferWindowAttention(self.pools, cfg.num_heads, cfg.sliding_kv_heads,
                                                          cfg.sliding_head_dim, dt, dev, self.window,
                                                          workspace=workspace, causal=False) \
                if self.bidi and not self.window_split_keys else None
        else:
            self.attn_w = TorchSlidingAttention(self.pools)
            self.attn_f = TorchFullAttention(self.pools, self.rotary, cfg.k_eq_v)
            self.attn_wb = None
            if self.k:  # the references plan nothing: the drafter's meta carries its rows
                self.attn_dw, self.attn_df = self.attn_w, self.attn_f
        if self.mm:  # the encoder pool and the tower's attention (its graphs: `_capture_all`)
            fi = self.attn_kind == "flashinfer"
            self.img.allocate(cfg.hidden_size, fi, workspace=workspace if fi else None)
        self._arange_pages = torch.arange(self.max_pages, device=dev)
        self._vocab_ids = torch.arange(cfg.vocab_size, device=dev)
        self._bctx = Gemma4Context(None, self.attn_w, self.attn_f, self.rotary, self.pools)
        self._dbctx = Gemma4Context(None, self.attn_dw, self.attn_df, self.rotary, self.pools) if self.k else None
        self._flip = 0
        self._last_end: Any = None
        self._allocated = True

    def _release(self) -> None:
        self._graphs = {}
        self._pool = None
        for name in ("pools", "arena", "tables", "rotary", "_staging", "attn_w", "attn_f", "attn_wb", "attn_dw",
                     "attn_df", "_arange_pages", "_vocab_ids", "_last_end", "static", "_bctx", "_dbctx", "_t_kv",
                     "_t_named", "_d_kv", "_d_named", "_t_wb", "_d_wb"):
            self.__dict__.pop(name, None)
        self.lp = None
        self.gs.release()
        if self.img is not None:
            self.img.release()
        self._allocated = False
        gc.collect()
        if self.cuda:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    @torch.inference_mode()
    def _reset_tables(self) -> None:
        if self.cuda:
            torch.cuda.synchronize()
        self.tables.reset()
        self.mirror = _Mirror(self.pc.max_num_seqs + 1)
        self._last_end = None
        for st in self._staging:
            st.handle = None

    def reset(self) -> None:
        self._reset_tables()

    # -- internal plans (profile, warm-up, capture) -------------------------------------------------------------

    def _warmup(self) -> None:
        """Every kernel variant once, eagerly: prefill rows with every sampler path, decode rows (and the split path
        with masks), and a resume that shares pages and copies a partial page of each kind."""
        pc = self.pc
        b = self._builder()
        budget = pc.max_num_batched_tokens
        n = max(1, min(4, pc.max_num_seqs))
        lens = [min(80, max(1, budget // 2)), 1, min(17, max(1, budget // 4)), min(3, max(1, budget // 8))][:n]
        while sum(lens) > budget:
            lens.pop()
        n = len(lens)
        params = [(0.0, 1.0, 0, 0.0), (1.0, 1.0, 5, 0.0), (0.8, 0.9, 0, 0.05), (1.0, 1.0, 0, 0.0)]
        heads = [VARIANT_HEADS[v] for v in self.gs.variants[1:]] if self.grammar else []  # the grammar heads' paths
        room = (2 + len(heads)) * self.L  # decode steps after the prompt (their verify windows under MTP)
        if sum(self._pages_of(x + 2 + 2 + room) for x in lens) >= pc.num_pages:
            log.warning("the KV arena is too small for the warm-up; skipping it")
            return
        rows = [(i + 1, lens[i]) for i in range(n)]
        # with PRECISE rows possible, the first row is one: a step of both paths (the precise prefix)
        plans = [lambda: self._synthetic(b, 1, rows, pages_from=1, params=params[:n], room=room,
                                         precise=[self.fp8.precise and i == 0 for i in range(n)])]
        if n * self.L <= budget:  # with grammar masks: the split path (under MTP 1 + k masks per row)
            plans.append(lambda: self._decode_plan(b, 2, [r[0] for r in rows], masked=self.grammar))
            for j in range(len(heads)):  # and each grammar heads' sampling path once (`_run`), a decode step apiece
                plans.append(lambda j=j: self._decode_plan(b, 8 + j, [r[0] for r in rows], masked=True))
        else:
            heads = []
        many = min(32, pc.max_num_seqs, pc.max_num_batched_tokens // 4)
        admitted = [r[0] for r in rows]
        if many > n and many * self._pages_of(6 + 2 + room) < pc.num_pages:
            first = list(admitted)
            plans.append(lambda: self._synthetic(b, 3, [(i + 1, 4) for i in range(many)], pages_from=1, free=first,
                                                 room=room))
            admitted = list(range(1, many + 1))
            if self.k and many * self.L <= budget:  # many rows verifying (attention plans of many uncertain rows)
                plans.append(lambda: self._decode_plan(b, 7, list(range(1, many + 1))))
        # a resume at an off-page position from slot 1's pages (shared full and window pages, a copied partial page of
        # each kind): past one full page when the budget allows
        long = min(pc.max_num_batched_tokens, self.fps + 3 if self.fps > self.ps else 2 * self.ps + 3)
        if 2 * self._pages_of(long + 4) + 2 < pc.num_pages:
            (src,), after = self._page_layout([(1, long)], pages_from=1)  # slot 1's pages in the plan before the resume
            plans.append(lambda: self._synthetic(b, 4, [(1, long)], pages_from=1, free=admitted))
            plans.append(lambda: self._resume_plan(b, 5, slot=2, seq_len=long - 1, src=src, page=after))
        img = min(16, self.img.budget, pc.max_num_batched_tokens - 2) if self.mm else 0
        if img > 0 and self._pages_of(img + 6) + 2 < pc.num_pages:
            # images: the tower (its attention's JIT), image rows, a bidirectional span; on the arena's last pages, in
            # a slot no earlier plan holds (or one it frees)
            def image_step() -> StepPlan:
                free = np.flatnonzero(~self.mirror.admitted[1:]) + 1
                slot = int(free[0]) if len(free) else 1
                return self._image_plan(b, 6, slot=slot, image_tokens=img, free=[] if len(free) else [1],
                                        pages_from=pc.num_pages - self._pages_of(img + 6))
            plans.append(image_step)
        graphs, self.use_graphs = self.use_graphs, False
        try:
            self._run(plans, heads=[NO_HEADS, *heads])
        finally:
            self.use_graphs = graphs
        self._reset_tables()

    def _image_plan(self, b: PlanBuilder, step: int, *, slot: int, image_tokens: int, pages_from: int,
                    free: list[int] = ()) -> StepPlan:
        """A plan admitting a prompt of one text token, one synthetic image of `image_tokens` tokens (its pixels from
        `ImageEncoder.synthetic`) and one text token, with its ENCODE and, on a bidirectional model, its span
        (profile and warm-up); full pages then window pages consecutive from `pages_from` (`_pages_of(image_tokens +
        6)` of them: the prompt and room for two decode steps)."""
        grid, hw = self.img.synthetic(image_tokens)
        n = image_tokens + 2
        b.begin()
        for s_ in free:
            b.add_free(s_)
        b.add_admit(slot, slot, 0, 0.0, 1.0, 0, 0.0, seed=7 + slot)
        nf, nw = -(-(n + 4) // self.fps), -(-(n + 4) // self.ps)
        b.add_pages(slot, 0, list(range(pages_from, pages_from + nf)))
        b.add_wpages(slot, 0, list(range(pages_from + nf, pages_from + nf + nw)))
        rows = np.arange(image_tokens, dtype=np.int64) % self.img.rows
        b.add_encode(1 << 62, image_tokens, grid, hw, 0, rows)
        emb = np.full(n, -1, np.int32)
        emb[1:1 + image_tokens] = rows
        b.add_explicit_row(slot, self._synthetic_tokens(slot, n), 0, STATE_ZERO | SAMPLE, emb=emb)
        if self.bidi:
            b.add_span(0, 1, 1 + image_tokens)
        return b.finish(step)

    def _pages_of(self, n: int) -> int:
        """Arena pages of a row that holds n positions of both kinds: its full pages and its window pages."""
        return -(-n // self.fps) + -(-n // self.ps)

    def _page_layout(self, rows: list[tuple[int, int]], pages_from: int,
                     room: int = 0) -> tuple[list[tuple[list[int], list[int]]], int]:
        """The pages of `_synthetic`'s (slot, prompt length) rows, in row order: each row's full pages and window pages
        for its prompt, two decode steps after it and `room` positions, consecutive arena pages from `pages_from` (its
        full pages, then its window pages); and the first page after them."""
        page, layout = pages_from, []
        for _, n in rows:
            nf, nw = -(-(n + 2 + 2 + room) // self.fps), -(-(n + 2 + 2 + room) // self.ps)
            layout.append((list(range(page, page + nf)), list(range(page + nf, page + nf + nw))))
            page += nf + nw
        return layout, page

    def _synthetic(self, b: PlanBuilder, step: int, rows: list[tuple[int, int]], *, pages_from: int,
                   free: list[int] = (), params: Any = None, room: int = 0,
                   precise: bool | list[bool] = False) -> StepPlan:
        """A plan admitting (slot, prompt length) rows as complete prompts (SAMPLE), each with full and window pages
        for two decode steps after it plus `room` positions (`_page_layout`). `precise`: the rows are PRECISE (one flag
        per row, or one for all)."""
        b.begin()
        for s in free:
            b.add_free(s)
        layout, _ = self._page_layout(rows, pages_from, room)
        for i, ((slot, n), (fp, wp)) in enumerate(zip(rows, layout)):
            p = params[i] if isinstance(params, list) else (params or (0.0, 1.0, 0, 0.0))
            b.add_admit(slot, slot, 0, p[0], p[1], p[2], p[3], seed=1234 + slot)
            b.add_pages(slot, 0, fp)
            b.add_wpages(slot, 0, wp)
            pre = precise[i] if isinstance(precise, list) else precise
            b.add_explicit_row(slot, self._synthetic_tokens(slot, n), 0, STATE_ZERO | SAMPLE | (PRECISE if pre else 0))
        return b.finish(step)

    def _resume_plan(self, b: PlanBuilder, step: int, *, slot: int, seq_len: int, src: tuple[list[int], list[int]],
                     page: int) -> StepPlan:
        """Admit `slot` at `seq_len` sharing the pages before it of a row whose pages are `src` (its full and window
        pages, as `_page_layout` gave them) and copying the partial page of each, then prefill a few tokens on pages
        from `page`."""
        b.begin()
        b.add_admit(slot, slot, seq_len, 0.0, 1.0, 0, 0.0, seed=99)
        for kind, ps, shared in (("full", self.fps, src[0]), ("window", self.ps, src[1])):
            whole = seq_len // ps
            n_own = -(-(seq_len + 3 + 2) // ps) - whole
            own = list(range(page, page + n_own))
            page += n_own
            (b.add_pages if kind == "full" else b.add_wpages)(slot, 0, shared[:whole] + own)
            if seq_len % ps:
                (b.add_kv_copy if kind == "full" else b.add_wkv_copy)(shared[whole], own[0])
        b.add_explicit_row(slot, self._synthetic_tokens(slot, 3, seq_len), seq_len, SAMPLE)
        return b.finish(step)

    def _synthetic_tokens(self, slot: int, n: int, start: int = 0) -> np.ndarray:
        return ((np.arange(start, start + n, dtype=np.int64) * 7919 + slot) % self.cfg.vocab_size).astype(np.int32)

    def _decode_plan(self, b: PlanBuilder, step: int, slots: list[int], masked: bool = False) -> StepPlan:
        b.begin()
        b.add_device_rows(np.asarray(slots, np.int32), np.ones(len(slots), bool) if masked else None)
        return b.finish(step)

    def _run(self, plans: list[Callable[[], StepPlan]], lead: bool = False,
             heads: list[Heads] = (), launch_ms: list[float] | None = None,
             lead_cycles: int = _LEAD_CYCLES) -> list[StepResult]:
        """Internal steps (profile, warm-up): masks of every token, sampled on the heads `heads` names per masked step
        in order (NO_HEADS after), with the sparse head's ids those of the first tokens: no counter or universe moves.
        `lead`: a device sleep of `lead_cycles` before each, so its device time excludes the host's; `launch_ms`
        collects each `launch_forward`'s host time."""
        prev, hook = None, self.logits_hook
        self.logits_hook = None
        out: list[StepResult] = []
        it = iter(heads)
        try:
            for make in plans:
                plan = make()
                if lead and self.cuda:
                    torch.cuda._sleep(lead_cycles)
                t = time.perf_counter()
                h = self.launch_forward(plan, _internal=True)
                if launch_ms is not None:
                    launch_ms.append((time.perf_counter() - t) * 1e3)
                if prev is not None:
                    out.append(self.wait(prev))
                n = h.split[3].n_masks if h.split is not None else 0
                if n:
                    self.gs.mask_np[h.parity][:n] = -1
                self.launch_sample(h, n, _heads=next(it, NO_HEADS) if n else None)
                prev = h
            if prev is not None:
                out.append(self.wait(prev))
        finally:
            self.logits_hook = hook
        return out

    # -- CUDA graph capture -----------------------------------------------------------------------------------

    def _capture_all(self) -> None:
        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        stream = torch.cuda.Stream(self.device)
        pc = self.pc
        b = self._builder()
        torch.cuda.synchronize()
        naming = dict(grammar=self.grammar, always_split=self.always_split, variants=self.gs.variants)
        warm_libraries(self.device)  # cuBLAS's handle before the first capture
        with no_gc(), torch.cuda.stream(stream):
            for t in sorted(self.token_buckets, reverse=True):
                rows = [(i + 1, n) for i, n in enumerate(step_rows(t, pc.max_model_len, pc.max_num_seqs))]
                if sum(n for _, n in rows) < t or sum(self._pages_of(n + 4) for _, n in rows) + 1 >= pc.num_pages:
                    continue  # unreachable, or no room
                for name in graph_names("break", **naming):
                    self._capture_one((name, t), lambda rows=rows: self._synthetic(b, 1, rows, pages_from=1))
            for n in sorted(self.decode_buckets, reverse=True):
                for name in graph_names("decode", **naming):
                    self._capture_one((name, n), lambda n=n: self._decode_capture_plan(b, n))
            if self.mm:  # the vision tower, by bucket of patches
                self.img.capture(self._pool)
        torch.cuda.synchronize()
        self._reset_tables()
        self.stats["graphs"] = graph_summary(self._graphs, variants=self.gs.variants if self.grammar else [],
                                             vision=self.img.graph_buckets() if self.mm else [])

    def _capture_one(self, key: tuple[str, int], make: Callable[[], StepPlan]) -> None:
        self._reset_tables()
        h = self.launch_forward(make(), _internal=True, _capture=key)
        self.wait(h)

    def _decode_capture_plan(self, b: PlanBuilder, n: int) -> StepPlan:
        """Slots 1..n admitted and sampled at a short context, sharing arena pages from 1 on (their full pages, then
        their window pages: contents do not matter for a capture); a decode plan of n rows."""
        m, t = self.mirror, self.tables
        sl = np.arange(1, n + 1)
        m.admitted[sl] = m.sampled[sl] = True
        m.seq_len[sl] = 3
        span = 3 + 2 * self.L  # the context and two verify windows after it
        nfp = -(-span // self.fps)  # full pages 1 .. nfp, then window pages
        fi, fp = self.arena.expand(np.arange(nfp), np.arange(1, nfp + 1))
        t.page_table[1:n + 1, torch.as_tensor(fi)] = torch.as_tensor(fp, dtype=torch.int32, device=t.page_table.device)
        nwp = -(-span // self.ps)
        t.wpage_table[1:n + 1, :nwp] = torch.arange(nfp + 1, nfp + 1 + nwp, dtype=torch.int32,
                                                    device=t.wpage_table.device)
        t.seq_len[1:n + 1] = 3
        t.pending[1:n + 1] = 0
        if self.k:
            t.max_len[1:n + 1] = self.pc.max_model_len
        return self._decode_plan(b, 1, sl.tolist())

    # -- the step path -----------------------------------------------------------------------------------------

    def launch_forward(self, plan: StepPlan, _internal: bool = False,
                       _capture: tuple[str, int] | None = None) -> _Handle:
        """The step's forward (`Runner.launch_forward`), under the sync check unless internal (warm-up, capture)."""
        if self._sync is None or _internal:
            return self._launch_forward(plan, _internal, _capture)
        with self._sync.launch():
            return self._launch_forward(plan, _internal, _capture)

    @torch.inference_mode()
    def _launch_forward(self, plan: StepPlan, _internal: bool = False,
                        _capture: tuple[str, int] | None = None) -> _Handle:
        if not (self._ready or _internal):
            raise RuntimeError("configure() first")
        if not self.k and plan.sched_n_draft.any():
            raise PlanError(f"step {plan.step_id}: drafts to verify, but MTP is off")
        if len(plan.fold_slot):
            raise PlanError(f"step {plan.step_id}: FOLD for an attention-only model (its MTP rollback is a truncation)")
        st = self._staging[self._flip]
        h = _Handle()
        h.parity, h.step_id, h.stage, h.internal = self._flip, plan.step_id, st, _internal
        h.observe = self.draft_head is not None and self.draft_head.reduced and not _internal
        self._flip ^= 1
        if st.handle is not None and st.handle.result is None:
            log.warning("staging buffer reused before wait() of step %d", st.handle.step_id)
            sync_checks.blocked(f"launch_forward: the staging buffer of step {st.handle.step_id} reused before its "
                                "wait()")
            self.wait(st.handle)
        st.handle = h
        st.reset()
        h.events = [torch.cuda.Event(enable_timing=True) for _ in range(5)] if self.cuda else \
            [_CpuEvent() for _ in range(5)]
        h.prev_end = self._last_end
        h.result = None
        h.fwd_end = h.plan_events = None
        fx = self._stage_effects(plan, st)
        enc = self.img.stage(plan, st, _internal) if len(plan.enc_key) else None
        lay = self._layout(plan, st)
        split = split_step(lay.n_masks, lay.n, self.always_split) and _capture is None
        if _capture is not None:
            kind, size = parse_graph_name(_capture[0])[0], _capture[1]
        else:
            kind, size = self._mode(lay, split)
        if lay.n:
            self._stage_static(lay, st, size, kind)
        # The step's device work starts at its upload, the host's layout and staging before it: when the device waits
        # for them, `wait` counts that time idle before the step, not busy
        h.events[0].record()
        st.upload(self.static if lay.n else None)
        self._apply_effects(fx, st)
        h.enc_events = None
        if enc is not None:  # ENCODE: after the copies, before the body (stream order)
            h.enc_events = [torch.cuda.Event(enable_timing=True) for _ in range(2)] if self.cuda else \
                [_CpuEvent() for _ in range(2)]
            h.enc_events[0].record()
            self.img.encode(plan, st, enc, self.use_graphs)
            h.enc_events[1].record()
        if h.observe:  # a re-ranked draft head, gathered before this step's body (stream order)
            self.draft_head.apply()
        h.n_sched, h.n_sample, h.s_plan_rows = lay.n, lay.n_sample, lay.s_plan_rows
        h.lp = lay.lp
        # The explicit rows that do not sample (their rows in the layout, and in the plan): `wait` reads their NaN
        # flags (the module docstring)
        chk = np.flatnonzero(lay.r_kind == _NO_SAMPLE) if lay.n else _EMPTY
        h.checks = (chk, lay.order[chk]) if len(chk) else None
        h.split = None
        h.mtp_rows = None
        if lay.n_mtp:  # resolved at wait(): the device advance of each verify row
            sr = np.flatnonzero(lay.r_kind[lay.c_row] == _VERIFY)
            slots = lay.r_slot[lay.c_row[sr]]
            h.mtp_rows = (sr, slots, self.mirror.epoch[slots].copy())
        if lay.n:
            with plan_bracket(h, self.cuda and _capture is None):
                meta = self._row_meta(lay, st, h.parity, kind, size)
            if _capture is not None:
                self._capture(_capture[0], size, meta, lay)
            elif split:
                ctx = self._part(kind, size, meta, lay, plan.step_id, "fwd")
                h.split = (kind, size, meta, lay, ctx)
                h.events[1].record()
                h.fwd_end = h.events[1] if self.cuda else None  # `forward_pending`
                return h
            elif kind == "eager":
                self._eager(meta, lay, plan.step_id)
            else:
                g = self._graphs[(kind, size)]
                if kind == "break":
                    self._bctx.meta = meta
                    if self.k:
                        self._dbctx.meta = meta.drafter
                g.replay()
            self._logprobs(h, lay, NO_HEADS)
        self._outputs(h, lay)
        return h

    def _outputs(self, h: _Handle, lay: _Layout, split: bool = False) -> None:
        ev, st = h.events, h.stage
        copy = lay.n_sample or h.checks is not None  # sampled tokens, or the NaN flags of rows that do not sample
        if copy:
            st.dout.copy_(self.static.out)
        if not split:
            ev[1].record()
            ev[2].record()
        ev[3].record()
        if copy:
            if self.cuda:
                self.copy_stream.wait_event(ev[3])
                with torch.cuda.stream(self.copy_stream):
                    st.tok.copy_(st.dout, non_blocking=True)
                    if h.lp is not None:
                        self.lp.copy_out(h.parity, h.lp[2])
                ev[4].record(self.copy_stream)
            else:
                st.tok.copy_(st.dout)
                if h.lp is not None:
                    self.lp.copy_out(h.parity, h.lp[2])
        self._last_end = ev[3]

    def _part(self, kind: str, size: int, meta: Gemma4Meta, lay: _Layout, step_id: int, part: str,
              ctx: Gemma4Context | None = None, heads: Heads = NO_HEADS) -> Gemma4Context | None:
        if kind == "eager":
            if part == "fwd":
                return self._eager(meta, lay, step_id, part="fwd")
            t, st, rp, cr = self._sizes("eager", lay.T, lay)
            self._body(ctx, t, st, rp, cr, "eager", lay, part="smp", dctx=self._drafter_ctx(meta), heads=heads)
            return ctx
        if kind == "break":
            self._bctx.meta = meta
            if self.k:
                self._dbctx.meta = meta.drafter
        self._graphs[(graph_name(kind, part, heads), size)].replay()
        return None

    def mask_buffer(self, h: _Handle) -> np.ndarray:
        if not self.grammar:
            raise PlanError("this runner was built without grammar masks")
        return self.gs.mask_buffer(h.parity)

    def launch_sample(self, h: _Handle, n_masks: int = 0, _heads: Heads | None = None) -> None:
        """The step's sampling (`Runner.launch_sample`), under the sync check unless internal."""
        if self._sync is None or h.internal:
            return self._launch_sample(h, n_masks, _heads)
        with self._sync.launch():
            return self._launch_sample(h, n_masks, _heads)

    @torch.inference_mode()
    def _launch_sample(self, h: _Handle, n_masks: int = 0, _heads: Heads | None = None) -> None:
        sp = h.split
        if sp is None:
            if n_masks:
                raise PlanError(f"step {h.step_id}: {n_masks} masks for a step without MASKED rows")
            if self.check:
                check_mirror(self.tables, self.mirror, self.cuda)
            return
        kind, size, meta, lay, ctx = sp
        h.split = None
        if n_masks != lay.n_masks:
            raise PlanError(f"step {h.step_id}: {n_masks} masks for MASKED rows that draw {lay.n_masks}")
        heads = self.gs.choose(h.parity, n_masks, lay.constrained, internal=h.internal, internal_heads=_heads)
        h.events[2].record()  # the sampling's device work starts here: the host's choice above is not device time
        self.gs.upload(h.parity, n_masks, heads, self.copy_stream)
        self.gs.install_universe(heads, h.internal)
        self._part(kind, size, meta, lay, h.step_id, "smp", ctx, heads=heads)
        self._logprobs(h, lay, heads)
        self._outputs(h, lay, split=True)
        if self.check:
            check_mirror(self.tables, self.mirror, self.cuda)

    def wait(self, h: _Handle) -> StepResult:
        if h.result is not None:
            return h.result
        if h.split is not None:
            raise RuntimeError(f"wait() of step {h.step_id} before its launch_sample()")
        ev = h.events
        if self.cuda:
            (ev[4] if h.n_sample or h.checks is not None else ev[3]).synchronize()
        L, k = self.L, self.k
        tokens = np.full((h.n_sched, L), -1, np.int32)
        n_out = np.zeros(h.n_sched, np.int32)
        drafts = np.full((h.n_sched, k), -1, np.int32) if k else None
        status = None
        if h.n_sample:
            n, s = h.n_sample, self.static.s
            out = h.stage.ntok
            toks = out[:s * L].reshape(s, L)[:n]
            nn = out[s * L:s * (L + 1)][:n] if k else np.ones(n, np.int64)
            st = out[s * (L + 1):s * (L + 2)][:n]
            tokens[h.s_plan_rows] = toks
            n_out[h.s_plan_rows] = nn
            if k:
                drafts[h.s_plan_rows] = out[s * (L + 2):s * (L + 2 + k)].reshape(s, k)[:n]
            if st.any():
                status = np.zeros(h.n_sched, np.uint32)
                status[h.s_plan_rows] = st.astype(np.uint32)
            if h.observe:  # the draft head's ranking counts the emitted tokens
                self.draft_head.observe(toks[np.arange(L)[None, :] < nn[:, None]])
            if h.mtp_rows is not None:  # the mirror's range closes: the device advanced each verify row by n_out
                sr, slots, epochs = h.mtp_rows
                m = self.mirror
                live = m.admitted[slots] & (m.epoch[slots] == epochs)
                sl = slots[live]
                m.seq_len[sl] += nn[sr[live]] - 1
                m.slack[sl] -= k
        if h.checks is not None:  # the explicit rows that do not sample: their last token's final hidden state
            rows, plan_rows = h.checks
            o = self.static.s * (L + 2 + k)  # `_Static.out_nf`
            bad = h.stage.ntok[o:o + self.static.s][rows] != 0
            if bad.any():
                if status is None:
                    status = np.zeros(h.n_sched, np.uint32)
                status[plan_rows[bad]] |= NONFINITE
        lp = self.lp.result(h.parity, h.lp[0], L) if h.lp is not None else None
        h.result = StepResult(tokens=tokens, n_out=n_out, status=status, drafts=drafts, logprobs=lp,
                              **step_timing(h, self.cuda))
        return h.result

    def forward_pending(self, h: _Handle) -> bool:
        return forward_running(h)

    def _mode(self, lay: _Layout, split: bool = False) -> tuple[str, int]:
        if not self.use_graphs or self.logits_hook is not None or not self._graphs or \
                self.fp8.precise_step(lay.n_precise):
            return "eager", lay.T

        def have(kind: str, b: int | None) -> bool:
            return b is not None and ((f"{kind}_f", b) in self._graphs and (f"{kind}_s", b) in self._graphs
                                      if split else (kind, b) in self._graphs)

        if lay.n_dev == lay.n and lay.n_sample == lay.n:
            b = _bucket(self.decode_buckets, lay.n)
            if have("decode", b):
                return "decode", b
        t = _bucket(self.token_buckets, lay.T)
        if have("break", t):
            return "break", t
        return "eager", lay.T

    # -- the body ----------------------------------------------------------------------------------------------

    def _sizes(self, kind: str, size: int, lay: _Layout | None) -> tuple[int, int, int, int]:
        """(tokens, sample tokens, rows, sample rows) the body runs: the bucket's static sizes in graphs, the step's
        own eagerly."""
        if kind == "decode":
            return size * self.L, size * self.L, size, size
        if kind == "break":
            rp = min(size, self.S)
            return size, min(size, self.S * self.L), rp, rp
        return lay.T, lay.n_stok, lay.n, lay.n_sample

    def _ids(self, t: int) -> torch.Tensor:
        """The tokens' input ids: explicit tokens, or the slot's pending token and (MTP) drafts on device rows."""
        s, tb = self.static, self.tables
        slot, tok = s.slot[:t], s.tok[:t]
        if self.k:
            w = (-1 - tok).clamp(min=0)  # device tokens: −1 − (window index)
            dev = torch.where(w == 0, tb.pending.index_select(0, slot), tb.drafts[slot, (w - 1).clamp(min=0)])
            return torch.where(tok >= 0, tok, dev)
        return torch.where(tok >= 0, tok, tb.pending.index_select(0, slot))

    def _positions(self, ctx: Gemma4Context, t: int) -> torch.Tensor:
        s, tb = self.static, self.tables
        slot, pin = s.slot[:t], s.pos[:t]
        if self.k:  # device rows: seq_len + the window index
            pos = torch.where(pin >= 0, pin, tb.seq_len.index_select(0, slot) + (-1 - pin).clamp(min=0))
        else:
            pos = torch.where(pin >= 0, pin, tb.seq_len.index_select(0, slot))
        m = ctx.meta
        m.positions = pos
        m.f_page, m.f_off = tb.page_table[slot, pos // self.sub].long(), pos % self.sub  # the arena's sub-pages
        m.w_page, m.w_off = tb.wpage_table[slot, pos // self.ps].long(), pos % self.ps
        return pos

    def _body(self, ctx: Gemma4Context, t: int, st: int, rp: int, cr: int, mode: str, lay: _Layout | None = None,
              hook: Any = None, part: str = "all", dctx: Gemma4Context | None = None, heads: Heads = NO_HEADS) -> None:
        """Inputs, the forward (outside decode graphs with each row's NaN flag, `_check_rows`), the softcapped logits of
        the sample tokens, sampling (MTP: verification, then the drafter, `_draft`) and the seq_len advance, for `t`
        tokens, `st` sample tokens, `rp` rows and `cr` sample rows (padding included); graph-safe. `part`: "fwd" ends at
        the sample tokens' hidden states (in `hid_out`), "smp" samples from them under the masks (split steps), on the
        heads `heads` names (the sparse target head, the universe draft head: `runner.cuda.heads`). `dctx`: the
        drafter's context (MTP)."""
        s, tb, k = self.static, self.tables, self.k
        splan = (lay.splan_sp if heads[0] else lay.splan) if lay is not None else None
        pos = self._positions(ctx, t)
        r_slot = s.r_slot[:rp]
        if part != "smp":
            ids = self._ids(t)
            if k and mode == "decode":
                self._fix_target(ctx.meta, rp, r_slot)
            if mode == "decode" and hasattr(self.attn_w, "graph_window_indices"):
                self.attn_w.graph_window_indices(rp, tb.wpage_table, r_slot)
            embeds = None
            if self.mm and mode != "decode":  # image tokens take their encoder rows (unscaled, as HF's)
                embeds = self.img.embeds(s.emb[:t], self.model.embed(ids))
            sel = s.s_idx[:st]
            if mode == "decode":  # device rows: each samples, and its logits show a NaN
                hidden = self.model(ids, ctx, select=sel, embeds=embeds)
            else:  # every token's final hidden state: the logits hook's, and the rows' NaN flags (the module docstring)
                full = self.model(ids, ctx, embeds=embeds)
                if hook is not None:
                    hook(full)
                hidden = full.index_select(0, sel)  # what the model's `select` returns, bit for bit
                self._check_rows(full, rp)
                del full  # before the logits: the activation peak (and a graph's pool) as with `select`
            if part == "fwd":
                s.hid_out[:st].copy_(hidden)
                return
        else:
            hidden = s.hid_out[:st]
        mask = (self.gs.mask_src, s.s_mask[:st]) if part == "smp" else None
        if not k:
            if st:
                sel = s.s_idx[:st]
                self._sample(hidden, pos.index_select(0, sel), s.s_slot[:st], s.out_tok[:st, 0], s.out_st[:st],
                             tb.pending, splan, mask=mask, sparse=heads[0])
            tb.seq_len.index_add_(0, r_slot, s.r_nq[:rp])
            return
        sel = s.s_idx[:st]
        spos = pos.index_select(0, sel)
        if st:
            self._sample(hidden, spos, s.s_slot[:st], s.y[:st], s.y_st[:st], None, splan, mask=mask, sparse=heads[0])
        a, _, _ = mtp_verify(s, tb, r_slot, rp, cr, st, k)
        if cr:
            self._draft(hidden, spos, a, st, cr, mode, lay, dctx, uni=heads[1])

    def _check_rows(self, full: torch.Tensor, rp: int) -> None:
        """Each row's NaN flag (`_Static.out_nf`): 1 when its last token's final hidden state in `full` (every token's)
        is not finite (padding rows, of no tokens: token 0's, never read); graph-safe."""
        s = self.static
        last = (s.r_start[:rp] + s.r_nq[:rp] - 1).clamp(min=0)
        s.out_nf[:rp].copy_(~torch.isfinite(full.index_select(0, last)).all(-1))

    def _draft(self, hidden: torch.Tensor, spos: torch.Tensor, a: torch.Tensor, st: int, cr: int, mode: str,
               lay: _Layout | None, dctx: Gemma4Context, uni: bool = False) -> None:
        """The assistant's k drafts for every sample row (the module docstring), after the verification: from the
        target's hidden state at the row's last committed position p (sample token r_s0 + a: window index a of a verify
        row, a SAMPLE row's one) and the token after it (now pending); draft j fills p + 2 + j and is sampled with that
        position's noise (`_sample` keys qpos + 1). `uni`: the assistant scores the grammar's universe only
        (`runner.cuda.heads`)."""
        s, tb, k = self.static, self.tables, self.k
        c_row, c_slot = s.c_row[:cr], s.c_slot[:cr]
        src = (s.r_s0.index_select(0, c_row) + a.index_select(0, c_row)).clamp(max=max(st - 1, 0))
        p = spos.index_select(0, src)
        h = hidden.index_select(0, src)
        dm = dctx.meta
        dm.positions = p + 1 if self.draft_next else p  # RoPE only: the KV and the window follow p (below)
        self._d_kv[:cr] = p + 1  # each drafter row's KV: [0, p]
        self._d_named[:cr] = self._d_kv[:cr] - self._d_wb[:cr]  # its sliding rows' KV from its first window page
        # The plans' lengths inside a decode graph (other steps: the backends' breaks, `fix_len`)
        if mode == "decode" and hasattr(self.attn_dw, "graph_window_indices"):
            self.attn_dw.fix_lengths(dm.sliding_plan, cr, self._d_named[:cr])
            self.attn_dw.graph_window_indices(cr, tb.wpage_table, c_slot)
            dm.full_plan[0][:cr, 3].copy_(self._d_kv[:cr])
        tok = tb.pending.index_select(0, c_slot).clamp(min=0)  # the token after p (padding rows: slot 0's)
        dplan = (lay.dplan_uni if uni else lay.dplan) if lay is not None else None
        x, hb = self.assistant(self.model.embed(tok), h, dctx)
        self._sample(x, p + 1, c_slot, s.drafts[0, :cr], s.d_st[:cr], None, dplan, draft=True, uni=uni)
        for j in range(1, k):
            x, hb = self.assistant(self.model.embed(s.drafts[j - 1, :cr]), hb, dctx)
            self._sample(x, p + 1 + j, c_slot, s.drafts[j, :cr], s.d_st[:cr], None, dplan, draft=True, uni=uni)
        dr = s.drafts[:, :cr].t()
        tb.drafts.index_copy_(0, c_slot, dr)
        s.out_dr[:cr] = dr  # returned with the outputs (StepResult.drafts)

    def _fix_target(self, meta: Gemma4Meta, rp: int, r_slot: torch.Tensor) -> None:
        """The target's rows' true KV lengths (seq_len + n_query) into its attention plans (graph-safe: decode graphs;
        `_row_meta` calls it eagerly for other steps whose lengths are uncertain)."""
        tb, s = self.tables, self.static
        self._t_kv[:rp] = tb.seq_len.index_select(0, r_slot) + s.r_nq[:rp]
        self._t_named[:rp] = self._t_kv[:rp] - self._t_wb[:rp]
        if self.attn_kind == "flashinfer":
            if self.window_split_keys:  # our sliding kernel's rows, as attn512's
                meta.window_plan[0][:rp, 3].copy_(self._t_kv[:rp])
            else:
                self.attn_w.fix_lengths(meta.sliding_plan, rp, self._t_named[:rp])
            meta.full_plan[0][:rp, 3].copy_(self._t_kv[:rp])

    def _drafter_ctx(self, meta: Gemma4Meta) -> Gemma4Context | None:
        return Gemma4Context(meta.drafter, self.attn_dw, self.attn_df, self.rotary, self.pools) if self.k else None

    def _sample(self, hidden: torch.Tensor, qpos: torch.Tensor, slots: torch.Tensor, out: torch.Tensor,
                status: torch.Tensor, pending: torch.Tensor | None, splan: Any,
                mask: tuple[torch.Tensor, torch.Tensor] | None = None, draft: bool = False, sparse: bool = False,
                uni: bool = False) -> None:
        """One token per row of `hidden` at position qpos + 1 (the sampler keys it there). `draft`: `hidden` is the
        assistant's draft hidden, scored by its head (`DraftHead`, over its token ids), or with `uni` by the universe's
        (`SubsetHead`); `sparse`: the target's head scores this step's union only (`_sparse_logits`), the masks read
        through its ids. Padded ids (−1) are never drawn (`ops.sample`)."""
        if draft:
            head = self.gs.uni_head if uni else self.draft_head
            ids, padded, logits = head.ids, uni, head.logits(hidden)
        elif sparse:
            ids, padded = self.gs.sp_src, True
            logits = self._sparse_logits(hidden, self.lp.sparse if self.lp is not None else None)
        else:
            ids, padded = None, False
            logits = self.model.logits(hidden, self.lp.dense if self.lp is not None else None)
        draw(logits, ids, padded, cuda=self.cuda, tables=self.tables, slots=slots, qpos=qpos, out=out, status=status,
             pending=pending, plan=splan, mask=mask, vocab_ids=self._vocab_ids)

    def _logprobs(self, h: _Handle, lay: _Layout, heads: Heads) -> None:
        """The step's LOGPROBS samples' values, after its sampling on `heads`, from the logits it read
        (`runner.cuda.logprobs`)."""
        if h.lp is not None:
            s = self.static
            self.lp.step(h.lp, h.parity, h.stage, s.y if self.k else s.out_tok[:, 0],
                         self.gs.sp_src if heads[0] else None, (self.gs.mask_src, s.s_mask) if lay.n_masks else None)

    def _sparse_logits(self, hidden: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        """[n, rows] fp32: `GrammarSampling.sparse_logits` with Gemma's head GEMM (`models.gemma4.model.linear`) and
        BF16 softcap, as `Gemma4ForCausalLM.logits` scores every token; graph-safe. On CUDA with `fused_ops`, n up to
        `models.gemma4.model.SMALL_M` and no `head_fixed_m`: both heads run our small-M kernel, so the masked choice is
        exactly the dense head's. `out`: the logprobs' buffer (`GrammarSampling.sparse_logits`)."""
        return self.gs.sparse_logits(hidden, out)

    def head_counters(self) -> HeadCounters:
        return self.gs.counters()

    def _eager(self, meta: Gemma4Meta, lay: _Layout, step_id: int, part: str = "all") -> Gemma4Context:
        ctx = Gemma4Context(meta, self.attn_w, self.attn_f, self.rotary, self.pools)
        hook = None
        if self.logits_hook is not None:
            def hook(full: torch.Tensor) -> None:
                for c0 in range(0, lay.T, _HOOK_CHUNK):
                    c1 = min(lay.T, c0 + _HOOK_CHUNK)
                    with sync_checks.allowed():  # a test's: it reads the logits and positions on the host
                        self.logits_hook(LogitsChunk(step_id, lay.tok_plan_row[c0:c1],
                                                     ctx.meta.positions[c0:c1].cpu().numpy(),
                                                     self.model.logits(full[c0:c1])))
        t, st, rp, cr = self._sizes("eager", lay.T, lay)
        with self.fp8.activations(t, lay.n_precise):  # the PRECISE rows' tokens (last) with BF16 activations
            self._body(ctx, t, st, rp, cr, "eager", lay, hook=hook, part=part, dctx=self._drafter_ctx(meta))
        return ctx

    def _capture(self, name: str, size: int, meta: Gemma4Meta, lay: _Layout) -> None:
        kind, part, heads = parse_graph_name(name)
        t, st, rp, cr = self._sizes(kind, size, None)
        dctx = None
        if kind == "decode":
            ctx = Gemma4Context(meta, self.attn_w, self.attn_f, self.rotary, self.pools)
            dctx = self._drafter_ctx(meta)
        else:
            ctx = self._bctx
            ctx.meta = meta
            if self.k:
                dctx = self._dbctx
                dctx.meta = meta.drafter
        torch.cuda.current_stream().synchronize()
        cap = BreakableCapture(self._pool)
        ctx.capture = cap if kind == "break" else None
        if dctx is not None:
            dctx.capture = ctx.capture
        try:
            with cap:
                self._body(ctx, t, st, rp, cr, kind, part=part, dctx=dctx, heads=heads)
        finally:
            ctx.capture = None
            if dctx is not None:
                dctx.capture = None
        self._graphs[(name, size)] = cap

    # -- host-side staging -------------------------------------------------------------------------------------

    def _stage_effects(self, plan: StepPlan, st: _Staging) -> _Effects:
        m, s_max = self.mirror, self.pc.max_num_seqs
        fx = _Effects()
        if len(plan.state_copy_src) or len(plan.ckpt_sched):
            raise PlanError(f"step {plan.step_id}: STATE_COPY or CKPT for a sliding-window model (its checkpoints are "
                            "windows in the scheduler)")
        if (len(plan.enc_key) or len(plan.tok_emb)) and not self.mm:
            raise PlanError(f"step {plan.step_id}: image inputs, but this runner has no vision tower")
        free, adm = plan.free_slot.astype(np.int64), plan.admit_slot.astype(np.int64)
        fx.n_free, fx.n_admit, fx.n_pages, fx.n_wpages = len(free), len(adm), len(plan.page_id), len(plan.wpage_id)
        for what, a in (("FREE", free), ("ADMIT", adm)):  # a slot at most once per list (`CudaRunner._stage_effects`)
            if len(a) > 1 and len(set(a.tolist())) != len(a):
                raise PlanError(f"step {plan.step_id}: a slot appears twice in {what}: {a.tolist()}")
        if fx.n_free:
            if (free < 1).any() or (free > s_max).any() or not m.admitted[free].all():
                raise PlanError(f"step {plan.step_id}: FREE of a slot that is not admitted: {free.tolist()}")
            m.admitted[free] = m.sampled[free] = m.lp[free] = False
            m.seq_len[free] = m.slack[free] = 0
            fx.o_free = st.put(free)
        if fx.n_admit:
            seq = plan.admit_seq_len.astype(np.int64)
            if (adm < 1).any() or (adm > s_max).any() or m.admitted[adm].any():
                raise PlanError(f"step {plan.step_id}: ADMIT into a slot in use: {adm.tolist()}")
            if (seq < 0).any() or (seq >= self.pc.max_model_len).any():
                raise PlanError(f"step {plan.step_id}: ADMIT with seq_len {seq.tolist()}")
            m.admitted[adm], m.sampled[adm], m.seq_len[adm] = True, False, seq
            m.slack[adm] = 0
            m.epoch[adm] += 1
            m.temperature[adm], m.top_p[adm] = plan.admit_temperature, plan.admit_top_p
            m.top_k[adm], m.min_p[adm] = plan.admit_top_k, plan.admit_min_p
            m.lp[adm] = (plan.admit_flags & ADMIT_LOGPROBS) != 0
            if self.lp is None and m.lp[adm].any():
                raise PlanError(f"step {plan.step_id}: ADMIT_LOGPROBS, but this runner was built without logprobs")
            fx.o_adm, fx.o_admseq = st.put(adm), st.put(seq)
            fx.o_topk = st.put(plan.admit_top_k)
            fx.o_seed = st.put(np.ascontiguousarray(plan.admit_seed).view(np.int64))
            fx.o_f = st.putf(np.concatenate([plan.admit_temperature, plan.admit_top_p, plan.admit_min_p]))
            if self.k:  # the commit limits (plan.py ADMIT)
                fx.o_maxlen = st.put(plan.admit_max_len.astype(np.int64))
                fx.o_stop = st.put(plan.admit_stop.astype(np.int64).reshape(-1))
        n_arena = self.pc.num_pages
        for pool, slot_a, idx_a, id_a, max_idx, n in (
                ("full", plan.page_slot, plan.page_index, plan.page_id, self.max_fpages, fx.n_pages),
                ("window", plan.wpage_slot, plan.wpage_index, plan.wpage_id, self.max_pages, fx.n_wpages)):
            if not n:
                continue
            lo = 1 if pool == "full" else 0  # window entries of 0 clear released pages
            ok_slot = (slot_a >= 1) & (slot_a <= s_max)
            if ok_slot.all():
                ok_slot = m.admitted[slot_a] | (np.isin(slot_a, free) & (id_a == 0) if pool == "window" else False)
            if not ok_slot.all() or (idx_a < 0).any() or (idx_a >= max_idx).any() or (id_a < lo).any() or \
                    (id_a >= n_arena).any():
                raise PlanError(f"step {plan.step_id}: {pool} PAGES out of range or for a slot that is not admitted")
            keep = m.admitted[slot_a]  # a release of a slot this plan FREEs needs no write (plan.py)
            if pool == "full":  # each full page is `groups` sub-pages of the arena (`_Arena`)
                sub_idx, sub_id = self.arena.expand(idx_a, id_a)
                fx.n_pages = len(sub_idx)
                fx.o_pslot = st.put(np.repeat(slot_a.astype(np.int64), self.arena.groups))
                fx.o_pidx, fx.o_pid = st.put(sub_idx), st.put(sub_id)
            else:
                fx.n_wpages = int(keep.sum())
                fx.o_wslot, fx.o_widx, fx.o_wid = st.put(slot_a[keep]), st.put(idx_a[keep]), st.put(id_a[keep])
        # copies of either kind are copies of arena pages
        fx.n_kvc = len(plan.kv_copy_src) + len(plan.wkv_copy_src)
        if fx.n_kvc:
            kv = np.concatenate([plan.kv_copy_src, plan.wkv_copy_src, plan.kv_copy_dst,
                                 plan.wkv_copy_dst]).astype(np.int64)
            if (kv < 1).any() or (kv >= n_arena).any():
                raise PlanError(f"step {plan.step_id}: KV_COPY page out of range")
            fx.o_kvc = st.put(kv)
        return fx

    def _apply_effects(self, fx: _Effects, st: _Staging) -> None:
        t = self.tables
        if fx.n_free:
            fs = st.i(fx.o_free, fx.n_free)
            t.page_table.index_fill_(0, fs, 0)
            t.wpage_table.index_fill_(0, fs, 0)
        if fx.n_admit:
            n = fx.n_admit
            sl = st.i(fx.o_adm, n)
            t.seq_len.index_copy_(0, sl, st.i(fx.o_admseq, n))
            t.pending.index_fill_(0, sl, -1)
            t.top_k.index_copy_(0, sl, st.i(fx.o_topk, n))
            t.seed.index_copy_(0, sl, st.i(fx.o_seed, n))
            f = st.f(fx.o_f, 3 * n)
            t.temperature.index_copy_(0, sl, f[:n])
            t.top_p.index_copy_(0, sl, f[n:2 * n])
            t.min_p.index_copy_(0, sl, f[2 * n:])
            if self.k:
                t.max_len.index_copy_(0, sl, st.i(fx.o_maxlen, n))
                t.stop.index_copy_(0, sl, st.i(fx.o_stop, n * STOP_W).view(n, STOP_W))
                t.drafts.index_fill_(0, sl, 0)
        if fx.n_pages:
            n = fx.n_pages
            t.page_table[st.i(fx.o_pslot, n), st.i(fx.o_pidx, n)] = st.i(fx.o_pid, n).to(torch.int32)
        if fx.n_wpages:
            n = fx.n_wpages
            t.wpage_table[st.i(fx.o_wslot, n), st.i(fx.o_widx, n)] = st.i(fx.o_wid, n).to(torch.int32)
        if fx.n_kvc:  # copy-on-write of shared partial pages of either kind: whole arena pages
            n = fx.n_kvc
            self.arena.copy(st.i(fx.o_kvc, n), st.i(fx.o_kvc + n, n))

    def _layout(self, plan: StepPlan, st: _Staging) -> _Layout:
        """SCHED → this step's row order, tokens, sample tokens and both attentions' metadata (MTP: each row's KV
        length within [lower, upper] bounds, the drafter's rows); advances the mirror."""
        m, lay, k, L = self.mirror, _Layout(), self.k, self.L
        slots = plan.sched_slot.astype(np.int64)
        n = lay.n = len(slots)
        lay.n_sample = lay.n_stok = lay.n_masks = lay.n_mtp = 0
        lay.splan = lay.dplan = lay.s_mask = lay.w = lay.full_rows = lay.g_first = lay.emb = lay.spans = None
        lay.lp = None
        lay.d = lay.splan_sp = lay.dplan_uni = None
        lay.constrained = False
        lay.s_plan_rows = np.zeros(0, np.int64)
        lay.n_precise = 0
        if n == 0:
            lay.T = lay.n_dec = lay.n_dev = 0
            return lay
        nq = plan.sched_n_query.astype(np.int64)
        tok_off = plan.sched_tok_off.astype(np.int64)
        dev = tok_off < 0
        flags = plan.sched_flags
        sample_rows = (flags & SAMPLE) != 0
        zero = (flags & STATE_ZERO) != 0
        pc, step = self.pc, plan.step_id
        if n > pc.max_num_seqs or (slots < 1).any() or (slots > pc.max_num_seqs).any() or \
                not m.admitted[slots].all():
            raise PlanError(f"step {step}: SCHED rows for slots that are not admitted: {slots.tolist()}")
        if n > 1 and len(set(slots.tolist())) != n:
            raise PlanError(f"step {step}: a slot appears twice in SCHED")
        t_total = int(nq.sum())
        if (nq < 1).any() or t_total > pc.max_num_batched_tokens:
            raise PlanError(f"step {step}: {t_total} query tokens (budget {pc.max_num_batched_tokens})")
        seq = m.seq_len[slots]
        slack = m.slack[slots]
        if dev.any():
            if (nq[dev] != L).any() or (plan.sched_pos0[dev] != -1).any() or zero[dev].any() or \
                    (plan.sched_n_draft[dev] != k).any():
                raise PlanError(f"step {step}: device-input rows need n_query {L}, n_draft {k}, pos0 -1 and no "
                                "STATE_ZERO")
            if not m.sampled[slots[dev]].all():
                raise PlanError(f"step {step}: device input before a SAMPLE step of the admission")
        ex = ~dev
        if ex.any():
            if (plan.sched_pos0[ex] != seq[ex]).any() or (slack[ex] != 0).any():
                raise PlanError(f"step {step}: pos0 {plan.sched_pos0[ex].tolist()} != seq_len {seq[ex].tolist()} "
                                "(or explicit tokens while a decode step of the slot is in flight)")
            if (zero[ex] != (seq[ex] == 0)).any():
                raise PlanError(f"step {step}: STATE_ZERO must be set exactly on the chunk at position 0")
            if (tok_off[ex] + nq[ex] > len(plan.tokens)).any():
                raise PlanError(f"step {step}: explicit tokens out of range")
            if (plan.sched_n_draft[ex] != 0).any():
                raise PlanError(f"step {step}: drafts on an explicit row")
        kv_len = seq + slack + nq  # the upper bound (exact without MTP)
        if int(np.where(dev, seq + 1, seq + nq).max()) > pc.max_model_len:
            raise PlanError(f"step {step}: positions beyond max_model_len {pc.max_model_len}")
        precise = (flags & PRECISE) != 0
        if precise.any():
            if not self.fp8.precise:
                raise PlanError(f"step {step}: PRECISE rows, but this runner computes none (no block-FP8 "
                                "projections, or runner.cuda.fp8_cold_prefill fp8)")
            if (precise & dev).any():
                raise PlanError(f"step {step}: PRECISE on a device-input row")
            lay.n_precise = int(nq[precise].sum())
        # Row order: device rows, then (without MTP) explicit 1-token rows, then the other rows, then PRECISE rows
        # (their tokens last: `kernels.fp8.precise_rows`)
        key = np.where(dev, 0, 2) if k else np.where(dev, 0, np.where(nq == 1, 1, 2))
        key = np.where(precise, 3, key)
        order = np.argsort(key, kind="stable") if n > 1 and (key[1:] < key[:-1]).any() else np.arange(n)
        r_slot, r_nq, r_pos0, r_kv = slots[order], nq[order], seq[order], kv_len[order]
        r_sample, r_dev = sample_rows[order], dev[order]
        n_dev = int(dev.sum())
        starts = np.cumsum(r_nq) - r_nq
        tok_row = np.repeat(np.arange(n), r_nq)
        widx = np.arange(t_total) - np.repeat(starts, r_nq)
        lay.tok_pos = np.repeat(r_pos0, r_nq) + widx
        tok_in = -1 - widx
        n_dtok = int(r_nq[:n_dev].sum())
        ex_idx = None
        if t_total > n_dtok:
            ex_rows, ex_nq = order[n_dev:], r_nq[n_dev:]
            ex_starts = starts[n_dev:] - n_dtok
            j = np.arange(t_total - n_dtok)
            ex_idx = np.repeat(tok_off[ex_rows] - ex_starts, ex_nq) + j
            ex_tok = plan.tokens[ex_idx]
            if int(ex_tok.min()) < 0 or int(ex_tok.max()) >= self.cfg.vocab_size:
                raise PlanError(f"step {step}: token out of vocabulary")
            tok_in[n_dtok:] = ex_tok
        if self.mm:  # each token's encoder row (−1: its token's embedding)
            lay.emb = np.full(t_total, -1, np.int64)
            if ex_idx is not None and len(plan.tok_emb):
                lay.emb[n_dtok:] = plan.tok_emb[ex_idx]
                self.img.check_rows(lay.emb, step)
        if len(plan.span_sched):
            lay.spans = self._layout_spans(plan, st, order, dev, starts, seq, r_nq, lay.emb, step)
        lay.tok_plan_row = order[tok_row]
        # Row kinds and sample tokens: a verify row samples its 1 + k tokens, a SAMPLE row its last
        r_kind = np.where(r_dev & (k > 0), _VERIFY,
                          np.where(r_sample | r_dev, _SAMPLE, _NO_SAMPLE)).astype(np.int64)
        n_tok_s = np.where(r_kind == _VERIFY, L, (r_kind == _SAMPLE).astype(np.int64))
        r_s0 = np.cumsum(n_tok_s) - n_tok_s
        s_idx = np.where(np.repeat(r_kind == _VERIFY, n_tok_s),
                         np.repeat(starts, n_tok_s) + (np.arange(int(n_tok_s.sum())) - np.repeat(r_s0, n_tok_s)),
                         np.repeat(starts + r_nq - 1, n_tok_s))
        lay.s_idx, lay.n_stok = s_idx, len(s_idx)
        lay.s_slot = np.repeat(r_slot, n_tok_s)
        lay.r_s0, lay.r_kind = r_s0, r_kind
        masked = (flags & MASKED) != 0
        if masked.any():  # grammar masks: in plan order, 1 + k per verify row, 1 per other SAMPLE row (runner.base)
            if not self.grammar:
                raise PlanError(f"step {step}: MASKED rows, but this runner was built without grammar masks")
            if (masked & ~sample_rows).any():
                raise PlanError(f"step {step}: MASKED on a row without SAMPLE")
            per = np.where(masked, np.where(dev & (k > 0), L, 1), 0)
            moff = np.cumsum(per) - per
            lay.n_masks = int(per.sum())
            within = np.arange(len(s_idx)) - np.repeat(r_s0, n_tok_s)
            lay.s_mask = np.where(np.repeat(masked[order], n_tok_s), np.repeat(moff[order], n_tok_s) + within, -1)
            lay.constrained = bool((lay.s_mask >= 0).all())  # every sample token masked: the grammar heads may serve
        c_row = np.flatnonzero(r_kind != _NO_SAMPLE)
        lay.c_row, lay.n_sample = c_row, len(c_row)
        lay.s_plan_rows = order[c_row]
        if self.lp is not None:  # LOGPROBS rows: 1 + k samples each under MTP (`runner.cuda.logprobs`)
            lay.lp = self.lp.layout(plan, sample_rows | dev, m.lp[slots], order, c_row, r_s0, lay.n_stok, L, st)
        elif (flags & LOGPROBS).any():
            raise PlanError(f"step {step}: LOGPROBS rows, but this runner was built without logprobs")
        if not self.cuda:  # the reference's sampling plans, per logits width (the grammar heads': their rows)
            lay.splan = host_plan(m, lay.s_slot, self.cfg.vocab_size)
            if self.gs.sparse and lay.constrained:
                lay.splan_sp = host_plan(m, lay.s_slot, self.gs.rows)
            if k:
                cs = r_slot[c_row]
                lay.dplan = host_plan(m, cs, self.draft_head.size)
                if self.gs.universe and lay.constrained:
                    lay.dplan_uni = host_plan(m, cs, self.gs.rows)
        # Both attentions: each row's first window page (from its lowest possible first query under MTP), the full
        # rows (planned for the upper bound of their KV), and the lower bounds
        ps, W = self.ps, self.window
        wlo = np.maximum(r_pos0 - W + 1, 0) // ps
        lay.wlo = wlo
        lay.r_kv_lb = r_pos0 + r_nq
        lay.full_rows = np.stack([r_slot, starts, r_nq, r_kv], axis=1).astype(np.int32)
        if self.attn_kind == "flashinfer" and not self.window_split_keys:  # FlashInfer's: each row's window pages
            kv_w = r_kv - wlo * ps
            n_wp = (kv_w + ps - 1) // ps
            indptr = np.concatenate([[0], np.cumsum(n_wp)])
            lay.w = (st.put(r_slot), st.put(wlo), st.put(n_wp), st.put(indptr), kv_w, n_wp, indptr,
                     int(n_wp.max()), int(indptr[-1]))
            lay.max_q = int(r_nq.max())
        if k and lay.n_sample:  # the drafter's rows: KV [0, p], p the last committed position of each sample row
            cr = c_row
            ver = r_kind[cr] == _VERIFY
            p_lb = np.where(ver, r_pos0[cr], r_pos0[cr] + r_nq[cr] - 1)
            p_ub = np.where(ver, r_pos0[cr] + slack[order][cr] + k, p_lb)
            lay.d = (r_slot[cr], np.maximum(p_lb - W + 1, 0) // ps, p_lb + 1, p_ub + 1)
        lay.order, lay.T, lay.n_dev, lay.n_dec = order, t_total, n_dev, int(((nq == 1) & ~precise).sum())
        lay.n_mtp = n_dev if k else 0
        lay.r_slot, lay.r_nq, lay.r_kv, lay.r_pos0, lay.starts = r_slot, r_nq, r_kv, r_pos0, starts
        lay.tok_in = tok_in
        m.seq_len[slots] = np.where(dev, seq + 1, seq + nq)  # device rows: at least the pending token
        if k:
            m.slack[slots[dev]] += k
        m.sampled[slots[sample_rows | dev]] = True
        return lay

    def _layout_spans(self, plan: StepPlan, st: _Staging, order: np.ndarray, dev: np.ndarray, starts: np.ndarray,
                      seq: np.ndarray, r_nq: np.ndarray, emb: np.ndarray | None, step: int) -> tuple:
        """SPANS: validate them (whole, inside their explicit rows, image tokens only) and stage the sliding layers'
        second plan: each span a row of its queries over its request's window pages [wlo(start) × ps, end)."""
        if not self.bidi:
            raise PlanError(f"step {step}: SPANS, but this model's image spans are causal")
        n = len(order)
        inv = np.empty(n, np.int64)
        inv[order] = np.arange(n)
        sp = plan.span_sched.astype(np.int64)
        a, e = plan.span_start.astype(np.int64), plan.span_end.astype(np.int64)
        if (sp < 0).any() or (sp >= n).any():
            raise PlanError(f"step {step}: SPANS of rows out of range")
        rr = inv[sp]
        p0 = seq[sp]
        if dev[sp].any() or (a < p0).any() or (a >= e).any() or (e > p0 + plan.sched_n_query[sp]).any():
            raise PlanError(f"step {step}: SPANS {list(zip(sp.tolist(), a.tolist(), e.tolist()))} outside their rows")
        q_idx = np.concatenate([np.arange(starts[r] + x - p, starts[r] + y - p)
                                for r, x, y, p in zip(rr.tolist(), a.tolist(), e.tolist(), p0.tolist())])
        if emb is None or (emb[q_idx] < 0).any() or len(np.unique(q_idx)) != len(q_idx):
            raise PlanError(f"step {step}: SPANS over text tokens, or overlapping")
        ps = self.ps
        wlo = np.maximum(a - self.window + 1, 0) // ps
        kv_w = e - wlo * ps
        n_wp = (kv_w + ps - 1) // ps
        indptr = np.concatenate([[0], np.cumsum(n_wp)])
        r_slot = plan.sched_slot.astype(np.int64)[sp]
        return (rr, a, e, st.put(q_idx), len(q_idx), st.put(r_slot), st.put(wlo), st.put(n_wp), st.put(indptr),
                kv_w, n_wp, indptr, q_idx)

    def _stage_static(self, lay: _Layout, st: _Staging, size: int, kind: str) -> None:
        s, a = self.static, st.ns
        t, n = lay.T, lay.n
        tp, sp, rp, cp = self._sizes(kind, size, lay)
        o = s.off
        a[o["slot"]:o["slot"] + t] = np.repeat(lay.r_slot, lay.r_nq)
        a[o["tok"]:o["tok"] + t] = lay.tok_in
        a[o["pos"]:o["pos"] + t] = np.where(lay.tok_in >= 0, lay.tok_pos, lay.tok_in)
        if tp > t:
            for f in ("slot", "tok", "pos"):
                a[o[f] + t:o[f] + tp] = 0
            if kind == "decode" and self.k:  # padding rows: token 0 at positions 0 .. k of the null page (KV 1 + k)
                a[o["pos"] + t:o["pos"] + tp] = np.tile(np.arange(self.L), (tp - t) // self.L)
        if self.mm:  # padding tokens: their token's embedding
            a[o["emb"]:o["emb"] + t] = lay.emb
            a[o["emb"] + t:o["emb"] + tp] = -1
        ns = lay.n_stok
        a[o["s_idx"]:o["s_idx"] + ns] = lay.s_idx
        a[o["s_slot"]:o["s_slot"] + ns] = lay.s_slot
        a[o["s_put"]:o["s_put"] + ns] = lay.s_idx
        a[o["s_mask"]:o["s_mask"] + sp] = -1
        if lay.s_mask is not None:
            a[o["s_mask"]:o["s_mask"] + ns] = lay.s_mask
        if sp > ns:
            if kind == "decode":  # padding rows sample their own tokens
                a[o["s_idx"] + ns:o["s_idx"] + sp] = np.arange(ns, sp)
                a[o["s_put"] + ns:o["s_put"] + sp] = np.arange(ns, sp)
            else:
                a[o["s_idx"] + ns:o["s_idx"] + sp] = 0
                a[o["s_put"] + ns:o["s_put"] + sp] = s.tmax
            a[o["s_slot"] + ns:o["s_slot"] + sp] = 0
        a[o["r_slot"]:o["r_slot"] + n] = lay.r_slot
        a[o["r_nq"]:o["r_nq"] + n] = lay.r_nq
        a[o["r_kind"]:o["r_kind"] + n] = lay.r_kind
        a[o["r_start"]:o["r_start"] + n] = lay.starts
        a[o["r_s0"]:o["r_s0"] + n] = lay.r_s0
        if rp > n:
            a[o["r_slot"] + n:o["r_slot"] + rp] = 0
            if kind == "decode":  # padding rows: 1 + k tokens and samples each, on slot 0 (verified on it: nothing)
                pr = np.arange(n, rp)
                a[o["r_nq"] + n:o["r_nq"] + rp] = self.L
                a[o["r_kind"] + n:o["r_kind"] + rp] = _VERIFY if self.k else _SAMPLE
                a[o["r_start"] + n:o["r_start"] + rp] = pr * self.L
                a[o["r_s0"] + n:o["r_s0"] + rp] = pr * self.L
            else:
                for f in ("r_nq", "r_kind", "r_start", "r_s0"):
                    a[o[f] + n:o[f] + rp] = 0
        c = lay.n_sample
        a[o["c_row"]:o["c_row"] + c] = lay.c_row
        a[o["c_slot"]:o["c_slot"] + c] = lay.r_slot[lay.c_row]
        if cp > c:
            a[o["c_row"] + c:o["c_row"] + cp] = np.arange(c, cp) if kind == "decode" else 0
            a[o["c_slot"] + c:o["c_slot"] + cp] = 0
        if kind == "decode" and self.attn_kind == "flashinfer" and not self.window_split_keys:
            # FlashInfer's decode graphs: the window's first column of every graph row
            first = np.zeros(size, np.int64)
            first[:n] = lay.wlo
            lay.g_first = st.put(first)
        if self.k:
            self._stage_mtp(lay, st, kind, rp, cp)

    def _stage_mtp(self, lay: _Layout, st: _Staging, kind: str, rp: int, cr: int) -> None:
        """MTP: the inputs of `_row_meta` and `_drafter_meta`, prepared before the upload that precedes the plans
        (`_Staging.upload`): the window bases they read on the device (the target's rows, then each drafter row of
        `lay.d`), and for the drafter's FlashInfer plan its rows' KV upper and lower bounds from their first window
        page, their page counts and indptr (host arrays), and the staged arrays its page indices come from (decode
        graphs: each graph row's first window page)."""
        ps = self.ps
        wb = np.zeros(rp, np.int64)
        wb[:lay.n] = lay.wlo * ps
        lay.o_twb, lay.dst = st.put(wb), None
        if lay.d is None:
            return
        d_slot, d_wlo, d_lb, d_ub = lay.d
        m = lay.n_sample
        dwb = np.zeros(cr, np.int64)
        dwb[:m] = d_wlo * ps
        o_dwb = st.put(dwb)
        if self.attn_kind != "flashinfer":
            lay.dst = (o_dwb,)
            return
        rows = cr if kind == "decode" else m
        kvw = np.ones(rows, np.int64)  # padding rows: one key of the null page
        kvw[:m] = d_ub - d_wlo * ps
        lb = kvw.copy()
        lb[:m] = d_lb - d_wlo * ps
        npg = (kvw + ps - 1) // ps
        indptr = np.concatenate([[0], np.cumsum(npg)])
        if kind == "decode":
            first = np.zeros(rows, np.int64)
            first[:m] = d_wlo
            dev = (st.put(first),)
        else:
            dev = (st.put(d_slot), st.put(d_wlo), st.put(npg[:m]), st.put(indptr))
        lay.dst = (o_dwb, rows, kvw, lb, npg, indptr, dev)

    def _row_meta(self, lay: _Layout, st: _Staging, parity: int, kind: str, size: int) -> Gemma4Meta:
        n = lay.n
        tb = self.tables
        meta = Gemma4Meta(num_tokens=lay.T, positions=None, f_page=None, f_off=None, w_page=None, w_off=None,
                          n_rows=n, row_start_host=lay.starts, row_nq_host=lay.r_nq, row_kv_len_host=lay.r_kv,
                          row_slot_host=lay.r_slot, row_wlo_host=lay.wlo, page_table=tb.page_table,
                          wpage_table=tb.wpage_table)
        if lay.spans is not None:
            meta.span_rows_host, meta.span_start_host, meta.span_end_host = lay.spans[:3]
        ps = self.ps
        # MTP: the plans are for each row's upper bound; its true length is written in later (module docstring)
        uncertain = bool(self.k) and bool((lay.r_kv_lb < lay.r_kv).any())
        if self.k:
            rp = self._sizes(kind, size, lay)[2]
            self._t_wb[:rp].copy_(st.i(lay.o_twb, rp))
            # A step without sample rows (prefill chunks short of their prompts' ends) drafts for none: an empty meta,
            # as a breakable graph replays the drafter for its padding rows and its attention breaks skip an empty step
            meta.drafter = self._drafter_meta(lay, st, parity, kind, size) if lay.d is not None else \
                Gemma4Meta(num_tokens=0, positions=None, f_page=None, f_off=None, w_page=None, w_off=None, n_rows=0,
                           row_start_host=_EMPTY, row_nq_host=_EMPTY, row_kv_len_host=_EMPTY, row_slot_host=_EMPTY,
                           row_wlo_host=_EMPTY)
            if self.check:  # the window bases reached the device (staged before the upload: `_stage_mtp`)
                assert np.array_equal(self._t_wb[:n].cpu().numpy(), lay.wlo * ps), "target window bases"
                if lay.d is not None:
                    assert np.array_equal(self._d_wb[:lay.n_sample].cpu().numpy(), lay.d[1] * ps), \
                        "drafter window bases"
        if self.attn_kind != "flashinfer":
            self.attn_w.plan(meta)
            self.attn_f.plan(meta)
            if uncertain:  # the references read the true lengths
                self._fix_target(meta, n, self.static.r_slot[:n])
                meta.kv_len_dev = self._t_kv[:n]
            return meta
        if kind == "decode":
            self._plan_decode_graph(meta, lay, st, parity, size)
            return meta
        from ...kernels.flashinfer import window_page_indices

        if self.window_split_keys:  # our sliding kernel: the full layers' rows, the spans' ends (kernels.attn_window)
            span_end = None
            if lay.spans is not None:
                a, e, q_idx = lay.spans[1], lay.spans[2], lay.spans[-1]
                span_end = np.zeros(lay.T, np.int32)
                span_end[q_idx] = np.repeat(e, e - a)
            self.attn_w.plan(meta, parity=parity, rows=lay.full_rows, stage=st.put32, fixed_rows=lay.n_dev,
                             span_end=span_end)
        else:
            o_slot, o_wlo, o_nwp, o_ip, kv_w, n_wp, indptr, max_pages, total = lay.w
            kv_indices = window_page_indices(tb.wpage_table, st.i(o_slot, n), st.i(o_wlo, n), st.i(o_nwp, n),
                                             st.i(o_ip, n + 1), max_pages, total, self._arange_pages)
            self.attn_w.plan(meta, parity=parity, qo_indptr=st.put32(np.concatenate([[0], np.cumsum(lay.r_nq)])),
                             kv_indptr=st.put32(indptr), kv_indices=kv_indices,
                             last_page_len=st.put32(kv_w - (n_wp - 1) * ps), kv_lens=st.put32(kv_w), max_q=lay.max_q,
                             lb=lay.r_kv_lb - lay.wlo * ps if uncertain else None)
        self.attn_f.plan(meta, parity=parity, rows=lay.full_rows, stage=st.put32, fixed_rows=lay.n_dev)
        if uncertain:  # eagerly, after the plans' copies (stream order), before the body
            self._fix_target(meta, n, self.static.r_slot[:n])
        if lay.spans is not None and self.attn_wb is not None:  # the span queries' second plan, without the causal mask
            _, a, e, o_q, n_q, o_slot, o_wlo, o_nwp, o_ip, kv_w, n_wp, indptr, _ = lay.spans
            k = len(a)
            idx = window_page_indices(tb.wpage_table, st.i(o_slot, k), st.i(o_wlo, k), st.i(o_nwp, k),
                                      st.i(o_ip, k + 1), int(n_wp.max()), int(indptr[-1]), self._arange_pages)
            self.attn_wb.plan(meta, parity=parity, qo_indptr=st.put32(np.concatenate([[0], np.cumsum(e - a)])),
                              kv_indptr=st.put32(indptr), kv_indices=idx, kv_lens=st.put32(kv_w),
                              last_page_len=st.put32(kv_w - (n_wp - 1) * ps), max_q=int((e - a).max()))
            meta.span_q = st.i(o_q, n_q)
        return meta

    def _plan_decode_graph(self, meta: Gemma4Meta, lay: _Layout, st: _Staging, parity: int, rows: int) -> None:
        """A decode bucket: real rows in the step's order, then padding rows of 1 + k tokens on the null page (KV
        length 1 + k); under MTP each row planned for its upper bound (the body writes the true lengths in)."""
        ps, n, L = self.ps, lay.n, self.L
        full = np.zeros((rows, 4), np.int32)  # both kernels' rows
        full[:, 1] = np.arange(rows) * L
        full[:, 2] = L
        full[:, 3] = L
        full[:n] = lay.full_rows
        if self.window_split_keys:  # our sliding kernel (kernels.attn_window): its bucket's items are static
            self.attn_w.plan_graph(meta, bucket=rows, rows=full, stage=st.put32)
        else:
            kv_w = lay.w[4]
            kvw = np.full(rows, L, np.int64)
            kvw[:n] = kv_w
            lb = kvw.copy()
            lb[:n] = lay.r_kv_lb - lay.wlo * ps
            npg = (kvw + ps - 1) // ps
            indptr = np.concatenate([[0], np.cumsum(npg)])
            self.attn_w.plan_graph(meta, rows=rows, parity=parity, qo_indptr=st.put32(np.arange(rows + 1) * L),
                                   kv_indptr=st.put32(indptr), last_page_len=st.put32(kvw - (npg - 1) * ps),
                                   kv_lens=st.put32(kvw), max_q=L, lb=lb if (lb < kvw).any() else None)
            self.attn_w.g_first[:rows].copy_(st.i(lay.g_first, rows))
        self.attn_f.plan_graph(meta, bucket=rows, rows=full, stage=st.put32, real=n)

    def _drafter_meta(self, lay: _Layout, st: _Staging, parity: int, kind: str, size: int) -> Gemma4Meta:
        """The drafter's rows (MTP): one query per sample row at its last committed position p, over the target's KV of
        [0, p] (`lay.d`: slot, first window page, lowest and highest KV length p + 1 per sample row), and their plans,
        made for the upper bound; the body writes the true lengths (`_d_kv`, `_d_named`) after the verification, and
        the backends read them (`fix_len`; inside decode graphs the body fixes the plans itself)."""
        tb, ps = self.tables, self.ps
        cr = self._sizes(kind, size, lay)[3]
        m = lay.n_sample
        d_slot, d_wlo, _, d_ub = lay.d
        self._d_wb[:cr].copy_(st.i(lay.dst[0], cr))
        dm = Gemma4Meta(num_tokens=m, positions=None, f_page=None, f_off=None, w_page=None, w_off=None, n_rows=m,
                        row_start_host=np.arange(m), row_nq_host=np.ones(m, np.int64), row_kv_len_host=d_ub,
                        row_slot_host=d_slot, row_wlo_host=d_wlo, page_table=tb.page_table, wpage_table=tb.wpage_table,
                        kv_len_dev=self._d_kv[:m], kv_named_dev=self._d_named[:m],
                        fix_len=kind != "decode" and self.attn_kind == "flashinfer")
        if self.attn_kind != "flashinfer":
            return dm
        _, rows, kvw, lb, npg, indptr, dev = lay.dst
        args = dict(parity=parity, qo_indptr=st.put32(np.arange(rows + 1)), kv_indptr=st.put32(indptr),
                    last_page_len=st.put32(kvw - (npg - 1) * ps), kv_lens=st.put32(kvw), max_q=1,
                    lb=lb if (lb < kvw).any() else None)
        full = np.zeros((rows, 4), np.int32)
        full[:, 1] = np.arange(rows)
        full[:, 2] = 1
        full[:, 3] = 1
        full[:m, 0] = d_slot
        full[:m, 3] = d_ub
        if kind == "decode":
            self.attn_dw.plan_graph(dm, rows=rows, **args)
            self.attn_dw.g_first[:rows].copy_(st.i(dev[0], rows))
            self.attn_df.plan_graph(dm, bucket=rows, rows=full, stage=st.put32, real=m)
        else:
            from ...kernels.flashinfer import window_page_indices

            o_slot, o_wlo, o_npg, o_ip = dev
            kv_indices = window_page_indices(tb.wpage_table, st.i(o_slot, m), st.i(o_wlo, m), st.i(o_npg, m),
                                             st.i(o_ip, m + 1), int(npg[:m].max()), int(indptr[m]),
                                             self._arange_pages)
            self.attn_dw.plan(dm, kv_indices=kv_indices, **args)
            self.attn_df.plan(dm, parity=parity, rows=full, stage=st.put32)
        return dm

    def shutdown(self) -> None:
        if self._allocated:
            self._release()
        self._ready = False
