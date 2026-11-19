# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# check_free_memory is adapted from vllm/v1/worker/utils.py of vLLM v0.30.0 (https://github.com/vllm-project/vllm), and
# changed for llm-mini-infer.
"""`CudaRunner`: Qwen3.5/3.8 dense models (text, and images with `VisionOptions`) in BF16 or block-FP8 behind
`runner.base.Runner`, with CUDA graphs and MTP speculative decoding (architecture.md, "The step plan", "The CUDA
runner").

**Device-side slot tables** (`_Tables`), written only through plans and stream-ordered on the compute stream:
page-table rows `[max_num_seqs + 1, max_pages]` (int32), and per slot the live state slot, the sampling parameters and
seed, `seq_len` (committed positions) and the pending token; under MTP also the drafts, the commit limits and the
window records (below). Slot 0 is padding: its page-table row stays zero (the null page), its state slot is 0 (the null
state slot) and it samples greedily. Plus the KV pool of every attention layer (the MTP layer's last) and the conv and
recurrent state pools (`models.qwen3_5.backend.Pools`; under MTP each state slot also holds the drafter's hidden vector
at its position − 1).

**launch_forward** writes the step's int64 host arrays into a pinned staging buffer (two alternate: a plan's data must
survive until its step runs on the device) and copies them to the device in one H2D copy, plus one for the float
sampling parameters of admitted requests and one for the **static input block** the step body reads; the attention
plans copy their own int32 tables. It then enqueues the plan's effects eagerly (`sched.plan`), and the drafter's KV at
`pos0 − 1` of a resumed admission's first row (`_mtp_resume_kv`).

**The step body** (`_body`) is one function: it builds the inputs on the device from the static block, runs the model,
samples the sample tokens (`kernels.ops.sample`) and advances `seq_len` and the pending token. It runs eagerly
(`_mode`), as a full CUDA graph per decode batch bucket or as a breakable graph per token bucket (architecture.md,
"CUDA graphs"). After it, end-of-chunk and end-of-step checkpoints copy live state slots, and the outputs go to a
per-parity device buffer and from there to the host on the copy stream; `wait` synchronises on that copy's event.

**A split step** (a step with MASKED rows; every step with rows under `always_split`) runs its forward in
`launch_forward` and its sampling in `launch_sample` (`part`), each with graphs of its own (`graphs.graph_names`). This
runner reports no forward end (`Capabilities.forward_end`: architecture.md, "Planning at the forward/sampling
boundary"). Its heads are `runner.cuda.heads.GrammarSampling`, shared with Gemma 4's runner.

**MTP** (`mtp_k` k > 0; architecture.md, "Speculative decoding"). Verification (`mtp_verify`, shared with Gemma 4's
runner), the MTP layer (`models.qwen3_5.model.MTP`) and its chain run inside the body, every graph included. The drafter
runs over the step's own token layout, so it reuses the target's attention plan; the chain has a plan of its own
(`chain_attn`). The GDN state rolls back by replay (`ops.GdnMtp`, `ops.ConvMtp`): a decode step's commit stays as window
records, which the slot's next verify step or a FOLD folds into the live state (an end-of-step checkpoint folds them
into its copy).

**The host mirror** (`_Mirror`) of `seq_len` is exact without MTP; under it a range [lo, lo + slack], widened by k at
each launch and closed at `wait`, so a row is planned at most k above its true length with one step in flight. Attention
is planned for the upper bound and the kernels read the true lengths on the device (`kernels.flashinfer`). The step path
syncs with the device only in two rare waits for a page-locked buffer: the vision tower's attention planning, and a
draft head's rebuild.

Block-FP8 checkpoints and PRECISE rows: `runner.cuda.fp8.Fp8Runtime`, shared with Gemma 4's runner (a step with
PRECISE rows runs eagerly, those rows last). Images: `runner.cuda.images`. Logprobs: `runner.cuda.logprobs`, shared
too (the target head's logits kept in its buffer; the LOGPROBS rows' pass after the sampling, before the outputs' copy).
Step timing: `step_timing`. Batch invariance: the parts that architecture.md, "Batch invariance" lists for Qwen3.8,
reported as `Capabilities.batch_invariant`.

**On CPU** the same runner runs with the torch reference kernels (`models.qwen3_5.backend`, `kernels.ops`): tables,
staging, the body and its input building are the same code; sampling is the torch reference; work is synchronous and
timing comes from the host clock.
"""

from __future__ import annotations

import gc
import logging
import math
import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ...config import FP32_KERNELS
from ...kernels import build as kbuild
from ...kernels import ops
from ...models.qwen3_5.backend import (
    GDN,
    CkptMeta,
    ForwardContext,
    MtpMeta,
    Pools,
    StepMeta,
    TorchAttention,
    torch_gdn_prefill,
)
from ...models.qwen3_5.model import Qwen35Config, Rotary, build_model
from ...sched.plan import (
    ADMIT_LOGPROBS,
    CKPT_END_OF_STEP,
    CKPT_GRID,
    FOLD_ALL,
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
)
from ..base import (
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
from ..mtp import MAX_CUDA_MTP_K
from . import sync_check as sync_checks
from .draft_head import DraftHead, special_token_ids, structural_token_ids
from .fp8 import Fp8Runtime
from .graphs import BreakableCapture, graph_name, graph_names, graph_summary, no_gc, parse_graph_name, warm_libraries
from .heads import NO_HEADS, VARIANT_HEADS, GrammarSampling, Heads
from .images import ImageEncoder, VisionOptions
from .logprobs import CHUNK as LP_CHUNK
from .logprobs import LogprobsPass
from .sampler import draw, host_plan

log = logging.getLogger(__name__)

PROFILE_MARGIN_BYTES = 512 << 20  # left unused on top of the measured peak (fragmentation, lazy allocations)
CPU_POOL_BYTES = 1 << 30  # pool memory reported on CPU unless dev.runner.cuda.memory_gb says otherwise
_HOOK_CHUNK = 256  # tokens per logits-hook call (256 × 248k × 4 B = 254 MB of fp32 logits)
_LEAD_CYCLES = 200_000_000  # a device sleep of about 70-100 ms (2-3 GHz) ahead of a timed profile step

# Row kinds (the static block's r_kind)
_NO_SAMPLE, _SAMPLE, _VERIFY = 0, 1, 2


def check_free_memory(free: int, total: int, utilization: float, device: str) -> None:
    """The startup check: this server claims `utilization` of the device's memory, so that much must be free when it
    starts. On a time-sliced GPU another process or container can hold memory on the same device; starting anyway
    would shrink the KV and state pools, or fail minutes later after the weights loaded."""
    want = math.ceil(total * utilization)
    if free < want:
        raise RuntimeError(
            f"Free memory on device {device} ({free / 2**30:.2f}/{total / 2**30:.2f} GiB) on startup is less than "
            f"llm-mini-infer's share (runner.cuda.memory_utilization {utilization:g}: {want / 2**30:.2f} GiB): "
            f"{(total - free) / 2**30:.2f} GiB are in use, by another process (on a time-sliced Kubernetes GPU, "
            "another pod) or a previous server still exiting. Free the GPU, or lower runner.cuda.memory_utilization.")


def require_cuda_device() -> None:
    """A CUDA runner's first check: torch must see a GPU, or startup fails, naming the setting. Nothing falls back
    (runner.kind is never derived from what is visible), so a server whose GPU went missing fails here instead of
    serving the fake runner's random tokens."""
    if torch.cuda.is_available():
        return
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    why = ("torch was built without CUDA" if not torch.backends.cuda.is_built() else
           f"no device or driver (CUDA_VISIBLE_DEVICES {'unset' if cvd is None else repr(cvd)})")
    raise RuntimeError(f"runner.kind cuda (the default) needs a CUDA GPU, and torch {torch.__version__} sees none: "
                       f"{why}. runner.kind fake answers with random tokens (tests); dev.runner.cuda.device cpu runs "
                       "the torch reference")


def default_decode_buckets(max_num_seqs: int) -> list[int]:
    """Batch sizes with a decode graph: 1, 2, 4, 8, then every 8 up to 32, every 16 above and `max_num_seqs` (a padded
    row costs a little attention and GDN work; the weights are read once whatever the batch)."""
    out = [b for b in (1, 2, 4, 8, 16, 24, 32) if b < max_num_seqs]
    out += list(range(48, max_num_seqs, 16))
    return sorted({*out, max_num_seqs})


def step_rows(tokens: int, max_model_len: int, max_rows: int) -> list[int]:
    """The prompt lengths of a synthetic step of `tokens` tokens (the memory profile, graph capture): one row when a
    request can be that long, else the fewest rows of at most max_model_len − 64 tokens (room for the decode steps a
    synthetic row reserves, so that each fits a slot's page table). At most `max_rows` (max_num_seqs): a step no set of
    requests can fill is cut to one that can."""
    if tokens <= max_model_len:
        return [tokens]
    longest = max(1, max_model_len - 64)
    n = -(-tokens // longest)
    if n > max_rows:
        return [longest] * max_rows
    q, r = divmod(tokens, n)
    return [q + (i < r) for i in range(n)]


def default_token_buckets(max_tokens: int) -> list[int]:
    """Token counts with a breakable graph: 16, 32, 64, 128, then every 128 up to the budget (padding wastes at most
    one 128-row GEMM tile above 128 tokens)."""
    out = [t for t in (16, 32, 64, 128) if t < max_tokens] + list(range(256, max_tokens, 128))
    return sorted({*out, max_tokens})


@dataclass
class LogitsChunk:
    """Passed to a CUDA runner's `logits_hook` (tests): full logits of some of a step's tokens."""

    step_id: int
    rows: np.ndarray  # SCHED row (plan order) of each token
    positions: np.ndarray  # position of each token: the step's own (device rows: the device's seq_len, exact under MTP)
    logits: torch.Tensor  # [n, vocab] fp32, on the device


class _CpuEvent:
    """CUDA-event stand-in on CPU, where work runs synchronously."""

    __slots__ = ("t",)

    def __init__(self) -> None:
        self.t = 0

    def record(self, stream: Any = None) -> None:
        self.t = time.perf_counter_ns()

    def synchronize(self) -> None:
        pass

    def elapsed_time(self, end: _CpuEvent) -> float:
        return (end.t - self.t) / 1e6


class _Static:
    """The static device inputs and outputs of the step body, at fixed addresses for every CUDA graph.

    One int64 block, filled by one H2D copy per step:
    - per token: `slot`, `tok` (explicit token; −1 − i for device input: window token i, the pending token at 0),
      `pos` (explicit position; −1 − i: the slot's seq_len + i), `dtok` (MTP: the drafter's input token at this
      position, the token after it; −1: the target's sample here);
    - per sample token (MTP decode rows: their 1 + k tokens; SAMPLE rows: the last): `s_idx` (its token), `s_slot`,
      `s_put` (where its sample goes in `y_tok`; tmax: the dump), `s_mask` (its grammar mask's row in the mask buffer,
      −1: none; read by the split steps' sampling only);
    - per row: `r_slot`, `r_nq` (n_query; the seq_len advance of explicit rows), `r_kind` (no sample, SAMPLE,
      verify), `r_start` (first token), `r_s0` (first sample token);
    - per sample row: `c_row` (its row; the drafter chains from it), `c_slot` (its slot; padding: 0).
    Outputs per sample row (`out`): tokens [S, 1 + k], n_out, status, and under MTP the drafts it made [S, k]; then
    per row `out_nf`, 1 where its last token's final hidden state is not finite (Gemma 4's runner, outside decode
    graphs: `runner.cuda.gemma`; this runner leaves it 0 and reads none of it).
    `hid_out` holds the forward's hidden states: of every token under MTP (mid-chunk checkpoints and the split steps'
    sampling read them), of the sample tokens without. Padding: slot 0 (`_stage_static`)."""

    def __init__(self, tmax: int, s: int, smax: int, width: int, k: int, hidden: int, dtype: torch.dtype,
                 device: torch.device, mm: bool = False, mrope: bool = False):
        self.tmax, self.s, self.smax, self.width = tmax, s, smax, width
        names = [("slot", tmax), ("tok", tmax), ("pos", tmax), ("dtok", tmax), ("s_idx", smax), ("s_slot", smax),
                 ("s_put", smax), ("s_mask", smax), ("r_slot", s), ("r_nq", s), ("r_kind", s), ("r_start", s),
                 ("r_s0", s), ("c_row", s), ("c_slot", s)]
        # Images: per token the encoder row of its input (−1: its token's embedding), of the drafter's input (MTP), and
        # its three M-RoPE deltas (consecutive: `rd` is a [3, tmax] view)
        if mm:
            names.append(("emb", tmax))
            if k:
                names.append(("demb", tmax))
        if mrope:
            names += [("rd0", tmax), ("rd1", tmax), ("rd2", tmax)]
        self.off: dict[str, int] = {}
        o = 0
        for name, n in names:
            self.off[name] = o
            o += n
        self.size = o
        self.block = torch.zeros(o, dtype=torch.int64, device=device)
        for name, n in names:
            setattr(self, name, self.block[self.off[name]:self.off[name] + n])
        if mrope:
            self.rd = self.block[self.off["rd0"]:self.off["rd0"] + 3 * tmax].view(3, tmax)
        self.out = torch.zeros(s * (width + 3 + k), dtype=torch.int64, device=device)
        self.out_tok = self.out[:s * width].view(s, width)
        self.out_n = self.out[s * width:s * (width + 1)]
        self.out_st = self.out[s * (width + 1):s * (width + 2)]
        self.out_dr = self.out[s * (width + 2):s * (width + 2 + k)].view(s, k)
        self.out_nf = self.out[s * (width + 2 + k):]
        self.hid_out = torch.zeros((tmax, hidden), dtype=dtype, device=device)
        self.y = torch.zeros(smax, dtype=torch.int64, device=device)  # the samples of the sample tokens
        self.y_st = torch.zeros(smax, dtype=torch.int64, device=device)
        self.arange64 = torch.arange(max(tmax, smax, s) + 1, dtype=torch.int64, device=device)
        self.arange32 = torch.arange(s + 1, dtype=torch.int32, device=device)
        self.ones32 = torch.ones(s, dtype=torch.int32, device=device)
        if k:
            self.y_tok = torch.zeros(tmax + 1, dtype=torch.int64, device=device)
            self.drafts = torch.zeros((k, s), dtype=torch.int64, device=device)  # the chain's drafts, step-major
            self.d_st = torch.zeros(s, dtype=torch.int64, device=device)  # the drafter's statuses (ignored)
            self.d_len = torch.zeros(s, dtype=torch.int64, device=device)  # the chain's true KV lengths
            self.kv_len = torch.zeros(s, dtype=torch.int64, device=device)  # the target's true KV lengths


class _Staging:
    """Pinned host arrays for one step and their device twin. Two alternate; see the module docstring."""

    def __init__(self, device: torch.device, cap_i: int, cap_f: int, cap_rows: int, static_size: int, out_size: int,
                 px_bytes: int = 0):
        pin = device.type == "cuda"
        self.px = torch.empty(px_bytes, dtype=torch.uint8, pin_memory=pin) if px_bytes else None  # ENCODE's pixels
        self.npx = self.px.numpy() if self.px is not None else None
        self.hi = torch.empty(cap_i, dtype=torch.int64, pin_memory=pin)
        self.hf = torch.empty(cap_f, dtype=torch.float32, pin_memory=pin)
        self.h32 = torch.empty(16 * cap_rows + 32, dtype=torch.int32, pin_memory=pin)
        self.hs = torch.zeros(static_size, dtype=torch.int64, pin_memory=pin)  # the static input block
        self.tok = torch.empty(out_size, dtype=torch.int64, pin_memory=pin)  # D2H target: the static outputs
        self.ni, self.nf, self.n32 = self.hi.numpy(), self.hf.numpy(), self.h32.numpy()
        self.ns, self.ntok = self.hs.numpy(), self.tok.numpy()
        # The device twins, on CPU too: an array staged after `upload` never reaches its twin, on CPU as on the GPU, so
        # the CPU tests catch it
        self.di = torch.empty(cap_i, dtype=torch.int64, device=device)
        self.df = torch.empty(cap_f, dtype=torch.float32, device=device)
        self.dout = torch.empty(out_size, dtype=torch.int64, device=device)  # the step's outputs, kept
        self.ki = self.kf = self.k32 = 0
        self.handle: _Handle | None = None  # the step that last used this buffer

    def reset(self) -> None:
        self.ki = self.kf = self.k32 = 0

    def put(self, a: np.ndarray) -> int:
        off, n = self.ki, len(a)
        if off + n > len(self.ni):
            raise RuntimeError(f"staging buffer too small ({off + n} > {len(self.ni)})")
        self.ni[off:off + n] = a
        self.ki = off + n
        return off

    def putf(self, a: np.ndarray) -> int:
        off, n = self.kf, len(a)
        self.nf[off:off + n] = a
        self.kf = off + n
        return off

    def put32(self, a: np.ndarray) -> torch.Tensor:
        """A pinned int32 host view (attention planning reads it and copies it to the device in stream order)."""
        off, n = self.k32, len(a)
        if off + n > len(self.n32):
            raise RuntimeError(f"int32 staging too small ({off + n} > {len(self.n32)})")
        self.n32[off:off + n] = a
        self.k32 = off + n
        return self.h32[off:off + n]

    def upload(self, static: _Static | None) -> None:
        if self.ki:
            self.di[:self.ki].copy_(self.hi[:self.ki], non_blocking=True)
        if self.kf:
            self.df[:self.kf].copy_(self.hf[:self.kf], non_blocking=True)
        if static is not None:
            static.block.copy_(self.hs, non_blocking=True)

    def i(self, off: int, n: int) -> torch.Tensor:
        return self.di[off:off + n]

    def f(self, off: int, n: int) -> torch.Tensor:
        return self.df[off:off + n]


class _Tables:
    """The device-side slot tables (architecture.md, "The step plan"), and under MTP the drafts, commit limits and
    window records."""

    def __init__(self, slots: int, max_pages: int, device: torch.device, k: int = 0, cfg: Qwen35Config | None = None,
                 dtype: torch.dtype = torch.bfloat16):
        self.page_table = torch.zeros((slots, max_pages), dtype=torch.int32, device=device)
        self.state_slot = torch.zeros(slots, dtype=torch.int64, device=device)
        self.seq_len = torch.zeros(slots, dtype=torch.int64, device=device)
        self.pending = torch.full((slots,), -1, dtype=torch.int64, device=device)
        self.temperature = torch.zeros(slots, dtype=torch.float32, device=device)
        self.top_p = torch.ones(slots, dtype=torch.float32, device=device)
        self.min_p = torch.zeros(slots, dtype=torch.float32, device=device)
        self.top_k = torch.zeros(slots, dtype=torch.int64, device=device)
        self.seed = torch.zeros(slots, dtype=torch.int64, device=device)
        self.rope_off = torch.zeros(slots, dtype=torch.int64, device=device)  # M-RoPE offset of device-input rows
        self.k = k
        if k:
            c, w = cfg, 1 + k
            n_gdn = len(c.gdn_layers)
            self.drafts = torch.zeros((slots, k), dtype=torch.int64, device=device)
            self.stop = torch.full((slots, STOP_W), -1, dtype=torch.int64, device=device)
            self.max_len = torch.zeros(slots, dtype=torch.int64, device=device)
            self.fold_n = torch.zeros(slots, dtype=torch.int32, device=device)
            # Window records per request slot (GDN replay, `ops.GdnMtp`, `ops.ConvMtp`)
            self.rec_k = torch.zeros((slots, n_gdn, c.lin_v_heads, w, c.lin_k_dim), dtype=dtype, device=device)
            self.rec_v = torch.zeros((slots, n_gdn, c.lin_v_heads, w, c.lin_v_dim), dtype=dtype, device=device)
            self.rec_ab = torch.zeros((slots, n_gdn, c.lin_v_heads, w, 2), dtype=dtype, device=device)
            self.rec_conv = torch.zeros((slots, n_gdn, c.conv_dim, w), dtype=dtype, device=device)
            self.rec_hid = torch.zeros((slots, w, c.hidden_size), dtype=dtype, device=device)

    def record_bytes(self) -> int:
        if not self.k:
            return 0
        return sum(t.numel() * t.element_size() for t in (self.rec_k, self.rec_v, self.rec_ab, self.rec_conv,
                                                           self.rec_hid))

    def reset(self) -> None:
        self.page_table.zero_()
        self.state_slot.zero_()
        self.seq_len.zero_()
        self.pending.fill_(-1)
        self.temperature.zero_()  # slot 0 (padding) samples greedily
        self.rope_off.zero_()
        if self.k:
            self.drafts.zero_()
            self.stop.fill_(-1)
            self.fold_n.zero_()


class _Mirror:
    """Host mirrors of what the plans wrote. `seq_len` is exact without MTP; under MTP the device seq_len is in
    [seq_len, seq_len + slack] (see the module docstring)."""

    def __init__(self, slots: int):
        self.admitted = np.zeros(slots, bool)
        self.sampled = np.zeros(slots, bool)  # a SAMPLE step of this admission was launched: device input is valid
        self.seq_len = np.zeros(slots, np.int64)
        self.slack = np.zeros(slots, np.int64)
        self.epoch = np.zeros(slots, np.int64)  # admissions of the slot so far: results of older ones are stale
        self.resumed = np.zeros(slots, bool)  # MTP: admitted with seq_len > 0 and no row yet (recompute pos0 − 1)
        self.state_slot = np.zeros(slots, np.int64)  # live state slot of each admitted slot
        self.temperature = np.zeros(slots, np.float32)
        self.top_p = np.ones(slots, np.float32)
        self.top_k = np.zeros(slots, np.int64)
        self.min_p = np.zeros(slots, np.float32)
        self.lp = np.zeros(slots, bool)  # admitted with ADMIT_LOGPROBS


class _Effects:
    __slots__ = ("fold", "n_admit", "n_free", "n_kvc", "n_pages", "n_sc", "n_zero", "o_adm", "o_admseq", "o_f",
                 "o_free", "o_kvc", "o_maxlen", "o_pid", "o_pidx", "o_pslot", "o_ropeoff", "o_sc", "o_seed", "o_ss",
                 "o_stop", "o_topk", "o_zero")


class _Layout:
    """Host side of one step: the row order and every offset into the staging buffer."""

    __slots__ = (
        "T",
        "c_row",
        "chain",
        "ck_mid",
        "constrained",
        "demb",
        "dplan",
        "dplan_uni",
        "dtok",
        "emb",
        "eoc",
        "eos",
        "eos_mtp",
        "full_rows",
        "kv_indptr32",
        "kv_lb",
        "kv_lens32",
        "last32",
        "lp",
        "max_pages",
        "max_q",
        "n",
        "n_dec",
        "n_dev",
        "n_masks",
        "n_mtp",
        "n_precise",
        "n_sample",
        "n_stok",
        "o_kvindptr",
        "o_npages",
        "o_pfcu",
        "o_rcu",
        "o_rslot",
        "order",
        "pf_cu",
        "qo_indptr32",
        "r_kind",
        "r_kv",
        "r_kv_lb",
        "r_nq",
        "r_s0",
        "r_slot",
        "rd",
        "resume",
        "s_idx",
        "s_mask",
        "s_plan_rows",
        "s_put",
        "s_slot",
        "splan",
        "splan_sp",
        "starts",
        "tok_in",
        "tok_plan_row",
        "tok_pos",
        "total_pages",
        "verify",
        "zero_states",
    )


class _Handle:
    __slots__ = ("enc_events", "events", "fwd_end", "internal", "lp", "mtp_rows", "n_sample", "n_sched", "observe",
                 "parity", "plan_events", "prev_end", "result", "s_plan_rows", "split", "stage", "step_id")


def _resolve(kind: str, cuda: bool, ok: bool, what: str) -> str:
    if kind == "auto":
        return "flashinfer" if cuda and ok else "torch"
    if kind == "flashinfer" and not (cuda and ok):
        raise ValueError(f"FlashInfer {what} needs a CUDA device and supported head dims")
    return kind


def _bucket(buckets: list[int], n: int) -> int | None:
    for b in buckets:
        if b >= n:
            return b
    return None


def mtp_verify(s: _Static, tb: Any, r_slot: torch.Tensor, rp: int, cr: int, st: int,
               k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MTP verification (architecture.md, "Speculative decoding") in both CUDA runners' step bodies (graph-safe). The
    target's samples of the step's sample tokens are `s.y[:st]` (statuses `s.y_st`). A verify row (kind _VERIFY) is
    the window [pending, k drafts]: its drafts are accepted while each equals the target's sample at its position, is
    no stop token, keeps the bonus below the slot's `max_len` and the sample there has status 0 (finite, mask not
    empty). The bonus (the target's sample after the `a` accepted drafts) becomes pending, and seq_len += 1 + a.
    Explicit rows advance by n_query; a SAMPLE row's sample becomes pending. Writes each sample row's outputs (tokens
    [cr, 1 + k], n_out, status); returns per row (a, verify rows, rows that sampled)."""
    kind = s.r_kind[:rp]
    ar = s.arange64[:1 + k]
    win = (s.r_s0[:rp, None] + ar[None, :]).clamp(max=max(st - 1, 0))
    yw, nw = s.y[win], s.y_st[win]
    ver = (kind == _VERIFY) & (r_slot > 0)
    cur = tb.seq_len.index_select(0, r_slot)
    lim = tb.max_len.index_select(0, r_slot)
    ok = (tb.drafts.index_select(0, r_slot) == yw[:, :k]) & \
        ~(tb.stop.index_select(0, r_slot)[:, :, None] == yw[:, None, :k]).any(1) & \
        (cur[:, None] + 2 + ar[None, :k] <= lim[:, None] - 1) & (nw[:, :k] == 0)
    a = ok.long().cumprod(1).sum(1) * ver
    bonus = yw.gather(1, a[:, None])[:, 0]
    bst = nw.gather(1, a[:, None])[:, 0]
    sampled = kind > _NO_SAMPLE
    c_row = s.c_row[:cr]
    n_out = torch.where(ver, 1 + a, sampled.long())
    s.out_tok[:cr] = yw[c_row]
    s.out_n[:cr] = n_out[c_row]
    s.out_st[:cr] = bst[c_row]
    adv = torch.where(ver, 1 + a, s.r_nq[:rp]) * (r_slot > 0)
    tb.seq_len.index_add_(0, r_slot, adv)
    tb.pending.index_copy_(0, r_slot, torch.where(sampled, bonus, tb.pending.index_select(0, r_slot)))
    return a, ver, sampled


def split_step(n_masks: int, n_rows: int, always_split: bool) -> bool:
    """Whether a step runs as its forward (`launch_forward`) and its sampling (`launch_sample`): it has MASKED rows
    (their masks come in between), or, always split, any rows at all."""
    return n_masks > 0 or (always_split and n_rows > 0)


def forward_running(h: Any) -> bool:
    """`Runner.forward_pending` of a step's handle: its forward-end event (`fwd_end`: a split step's `events[1]`,
    recorded between the forward and the sampling) not reached yet on the device. A fused step records that event only
    after its sampling, and an eager CPU step has no device: no `fwd_end` for either, False."""
    return h.fwd_end is not None and not h.fwd_end.query()


@contextmanager
def plan_bracket(h: Any, timed: bool) -> Iterator[None]:
    """Brackets a step's attention planning (`timed`: a CUDA step, not a capture) for `step_timing`: when the device has
    run out of work before the host finished the plans (a late launch), the handle keeps the bracket (`plan_events`) and
    `wait` counts that wait idle before the step; otherwise the time between the two events is the plans' own copies,
    busy."""
    e = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) if timed else None
    if e is not None:
        e[0].record()
    yield
    if e is not None:
        late = e[0].query()  # the device reached the plans' start: it waits for the host
        e[1].record()
        h.plan_events = e if late else None


def step_timing(h: Any, cuda: bool) -> dict[str, int]:
    """A waited step's timing fields of `StepResult` from its handle's events (`events`: the device work's start, the
    forward's end, the sampling's start, the end; `prev_end`, `plan_events`, `enc_events`, `fwd_end`): busy = start to
    end less the gap between a split step's forward and its sampling (idle inside), and less a late launch's planning
    (`plan_bracket`), which counts idle before the step with the time since the previous step's end; a split step's
    sampling (0 for the others); the encoder's time. An eager body also idles inside itself whenever the host enqueues
    kernels slower than they run, which counts as busy here. On CPU nothing idles."""
    ev = h.events
    sample = ev[2].elapsed_time(ev[3])
    busy = ev[0].elapsed_time(ev[1]) + sample
    if cuda:
        inside = ev[1].elapsed_time(ev[2])
        before = h.prev_end.elapsed_time(ev[0]) if h.prev_end is not None else 0.0
        if h.plan_events is not None:  # a late launch: the device waited for the attention plans
            gap = max(0.0, h.plan_events[0].elapsed_time(h.plan_events[1]))
            busy -= gap
            before = max(0.0, before) + gap
    else:
        inside = before = 0.0
    enc_ns = int(h.enc_events[0].elapsed_time(h.enc_events[1]) * 1e6) if h.enc_events is not None else 0
    return dict(gpu_busy_ns=int(busy * 1e6), gpu_idle_before_ns=max(0, int(before * 1e6)),
                gpu_idle_inside_ns=max(0, int(inside * 1e6)), encoder_busy_ns=enc_ns,
                gpu_sample_ns=int(sample * 1e6) if h.fwd_end is not None else 0)


def check_mirror(tables: Any, mirror: Any, cuda: bool) -> None:
    """Debug (dev.runner.cuda.check), both runners' check after a step's sampling: the device seq_len lies in the host
    mirror's range for admitted slots, and an admitted slot has a pending token exactly when the mirror says it has
    sampled. Syncs."""
    if cuda:
        torch.cuda.synchronize()
    dev = tables.seq_len.cpu().numpy()
    m = mirror
    bad = np.flatnonzero(m.admitted & ((dev < m.seq_len) | (dev > m.seq_len + m.slack)))
    if bad.size:
        raise AssertionError(f"device seq_len {dev[bad].tolist()} outside the host mirror "
                             f"[{m.seq_len[bad].tolist()}, +{m.slack[bad].tolist()}] for slots {bad.tolist()}")
    pend = tables.pending.cpu().numpy()
    if (m.admitted & m.sampled & (pend < 0)).any() or (m.admitted & ~m.sampled & (pend >= 0)).any():
        raise AssertionError("pending tokens disagree with the host mirror's sampled flags")


class CudaRunner(Runner):
    def __init__(self, model_dir: str | Path, *, device: str = "cuda", dtype: torch.dtype | None = None,
                 attention: str = "auto", gdn: str = "auto", max_num_seqs: int = 64,
                 max_num_batched_tokens: int = 2048, max_model_len: int | None = None,
                 memory_utilization: float = 0.9, memory_gb: float | None = None, check: bool = False,
                 graphs: bool = True, fused_ops: bool = True, decode_buckets: list[int] | None = None,
                 token_buckets: list[int] | None = None, fp8_gemm: str = "auto", fp8_cold_prefill: str = "reuse",
                 mtp_k: int = 0, draft_vocab: int = 0, draft_fp8: bool = False, vision: VisionOptions | None = None,
                 vision_buckets: list[int] | None = None, grammar: bool = True, pdl: bool = False,
                 always_split: bool = False, sparse_head: bool = False, sparse_head_rows: int = 1024,
                 draft_universe: bool = False, batch_invariant: bool = False, attn_split_keys: int = 0,
                 sync_check: str | None = None, logprobs: bool = False):
        """`batch_invariant`: a row's bits do not depend on the rows beside it (architecture.md, "Batch invariance");
        `attn_split_keys`: then a decode row's split of its full layers' keys (0: `attn512.fixed_split_keys` of the
        longest row). `logprobs`: takes LOGPROBS rows, its head's logits kept for them (`runner.cuda.logprobs`)."""
        if always_split and not grammar:
            raise ValueError("always-split graphs need the grammar masks' split (structured output)")
        self.device = torch.device(device)
        self.cuda = self.device.type == "cuda"
        if self.cuda:
            require_cuda_device()
            if self.device.index is None:
                self.device = torch.device("cuda", torch.cuda.current_device())
            torch.cuda.set_device(self.device)
            if memory_gb is None:  # the share must be free before anything loads (a GPU shared with another pod)
                check_free_memory(*torch.cuda.mem_get_info(self.device), memory_utilization, str(self.device))
        self.dtype = dtype or (torch.bfloat16 if self.cuda else torch.float32)
        self.model_dir = Path(model_dir)
        self.cfg = cfg = Qwen35Config.from_dir(self.model_dir)
        if mtp_k and not cfg.mtp_layers:
            raise ValueError(f"MTP needs an MTP layer in the checkpoint; {self.model_dir} has none")
        if self.cuda and mtp_k > MAX_CUDA_MTP_K:
            raise ValueError(f"the CUDA runner drafts at most {MAX_CUDA_MTP_K} tokens (speculative."
                             f"num_speculative_tokens = {mtp_k})")
        self.k, self.L = mtp_k, 1 + mtp_k
        self.S, self.B = max_num_seqs, max_num_batched_tokens
        self.logprobs = logprobs
        self.lp: LogprobsPass | None = None
        self.grammar = grammar  # MASKED rows (structured output): mask buffers and the split steps' graphs
        self.always_split = always_split  # every step with rows split, masked or not; only the split graphs
        self.inv = batch_invariant
        self.max_model_len = min(max_model_len or cfg.max_position_embeddings, cfg.max_position_embeddings)
        if batch_invariant:  # the full layers' fixed split (kernels.attn512's rule, on kernels.attn_window's kernels)
            from ...kernels.attn512 import fixed_split_keys

            attn_split_keys = attn_split_keys or fixed_split_keys(self.max_model_len + 1 + mtp_k)
        self.attn_split_keys = attn_split_keys if batch_invariant else 0
        self.memory_utilization = memory_utilization
        self.memory_gb = memory_gb
        self.check = check
        self.attn_kind = _resolve(attention, self.cuda, cfg.head_dim in (64, 128, 256), "attention")
        self.gdn_kind = _resolve(gdn, self.cuda, cfg.lin_k_dim == cfg.lin_v_dim == 128, "GDN prefill")
        t0 = time.perf_counter()
        ops.enabled(fused_ops)
        if self.cuda:
            kbuild.require("the CUDA runner")
        kernels = self.cuda and fused_ops
        if kernels and self.dtype != torch.bfloat16:
            raise ValueError(FP32_KERNELS)
        # `pdl` (`ops.set_pdl`): set before any capture, as the graphs keep the launches they captured; a CPU runner
        # leaves the library alone
        self.pdl = ops.set_pdl(pdl and kernels) if self.cuda else False
        self.stats: dict[str, Any] = {"kernels": kbuild.stats.get("state", "unused") if self.cuda else "cpu",
                                      "pdl": self.pdl, "always_split": always_split,
                                      "batch_invariant": batch_invariant}
        if kbuild.stats.get("load_s") is not None:
            self.stats["kernel_load_s"] = kbuild.stats["load_s"]
        # CUDA graphs need the kernels (the torch references read host values) and FlashInfer attention.
        self.use_graphs = graphs and kernels and self.attn_kind == "flashinfer"
        # The sync check (`runner.cuda.sync_check`; None: its default) around the step's launches
        self._sync = sync_checks.make(sync_check, cuda=self.cuda, check=check, reference=(
            "the torch references (dev.runner.cuda.fused_ops off)" if not kernels else
            "torch attention" if self.attn_kind != "flashinfer" else
            "torch GDN prefill" if self.gdn_kind != "flashinfer" else "the torch FP8 GEMM" if fp8_gemm == "torch" else
            None))
        rows = min(max_num_seqs, max_num_batched_tokens // self.L)  # a decode step's tokens are within the budget
        self.decode_buckets = sorted(b for b in (decode_buckets or default_decode_buckets(rows)) if 0 < b <= rows)
        self.token_buckets = sorted(b for b in (token_buckets or default_token_buckets(max_num_batched_tokens))
                                    if 0 < b <= max_num_batched_tokens)
        self.fp8 = Fp8Runtime(cfg.quant is not None, cuda=self.cuda, kernels=kernels, dtype=self.dtype,
                              fp8_gemm=fp8_gemm, fp8_cold_prefill=fp8_cold_prefill, stats=self.stats)
        t1 = time.perf_counter()
        self.model = build_model(self.model_dir, self.device, self.dtype, mtp=self.k > 0)
        if self.cuda:
            torch.cuda.synchronize()
        n_fp8 = self.fp8.attach(self.model, self.device)
        if batch_invariant and kernels:  # in_proj_ba's every row as at any M (`ops.linear_rows`)
            for layer in self.model.layers:
                if not layer.is_attn:
                    layer.linear_attn.ba_rows = True
        self.stats.update({
            "load_s": round(time.perf_counter() - t1, 2),
            "weight_bytes": sum(p.numel() * p.element_size() for p in self.model.parameters()),
            "attention": self.attn_kind, "gdn_prefill": self.gdn_kind, "dtype": str(self.dtype).removeprefix("torch."),
            "graphs": self.use_graphs, "init_s": round(t1 - t0, 2), "mtp_k": self.k})
        self.draft_head: DraftHead | None = None
        if self.k:
            mtp = self.model.mtp
            self.stats["mtp_weight_bytes"] = sum(p.numel() * p.element_size() for p in mtp.parameters())
            if draft_fp8 and self.cuda:  # an FP8 draft head's GEMMs
                kbuild.require_cutlass("dev.speculative.draft_head_dtype bfloat16")
            if self.cuda:  # fresh segments for the head, else it would keep the loader's cached ones pinned
                torch.cuda.empty_cache()
            forced = special_token_ids(self.model_dir)
            if grammar and 0 < draft_vocab < self.model.lm_head_weight.shape[0]:
                forced += structural_token_ids(self.model_dir)
            self.draft_head = DraftHead(self.model.lm_head_weight, size=draft_vocab, fp8_copy=draft_fp8,
                                        special=forced)
            self.stats.update(draft_head=self.draft_head.describe(), draft_head_bytes=self.draft_head.bytes)
            gdn_layers = [layer.linear_attn for layer in self.model.layers if not layer.is_attn]
            # every GDN layer's gating parameters, for the fold kernel (all layers in one launch)
            self._a_log = torch.stack([g.A_log for g in gdn_layers]).contiguous()
            self._dt_bias = torch.stack([g.dt_bias for g in gdn_layers]).contiguous()
        # The grammar masks and the grammar-aware heads (`runner.cuda.heads`), whose GEMM and `.float()` defaults are
        # `Qwen35ForCausalLM.logits`'s. The universe draft head (MTP) takes its rows from the target's head (the MTP
        # layer's head too) and is allocated here: the memory profile counts it as loaded
        head = self.model.lm_head_weight
        assert head.dtype == self.dtype, f"a {head.dtype} lm_head (Qwen's FP8 checkpoints keep it BF16)"
        self.gs = gs = GrammarSampling(grammar=grammar, sparse=sparse_head, universe=draft_universe,
                                       rows=sparse_head_rows, vocab=cfg.vocab_size, head=head,
                                       draft_weight=head if self.k else None, fixed_m=batch_invariant)
        self.stats.update(sparse_head=gs.sparse, draft_universe=gs.universe, sparse_head_rows=gs.rows)
        log.info("model loaded in %.1f s: %.2f GiB of weights (%s%s), attention %s, GDN prefill %s, CUDA graphs %s%s",
                 self.stats["load_s"], self.stats["weight_bytes"] / 2**30, self.stats["dtype"],
                 f", {n_fp8} block-FP8 projections" if n_fp8 else "", self.attn_kind, self.gdn_kind,
                 "on" if self.use_graphs else "off",
                 f", MTP k = {self.k}, draft head: {self.stats['draft_head']} "
                 f"({self.stats['draft_head_bytes'] / 2**30:.2f} GiB)" if self.k else "")
        self.fp8.log_setup()
        if gs.heads is not None:
            log.info(gs.describe())
        # Images: the vision tower and the pixel arena (`runner.cuda.images`)
        self.mm = vision is not None
        self.mrope = self.mm and len(cfg.mrope_section) == 3
        self.img: ImageEncoder | None = None
        if self.mm:
            self.img = ImageEncoder(self.model_dir, vision, self.device, self.dtype, buckets=vision_buckets)
            self.img.per_image = batch_invariant  # one tower call per image (batch invariance)
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
        c = self.cfg
        return (len(c.attn_layers) + (1 if self.k else 0)) * 2 * c.num_kv_heads * c.head_dim * self.dtype.itemsize

    @property
    def state_slot_bytes(self) -> int:
        c = self.cfg
        rec = c.lin_v_heads * c.lin_v_dim * c.lin_k_dim * 4
        conv = c.conv_dim * (c.conv_kernel - 1) * self.dtype.itemsize
        hid = c.hidden_size * self.dtype.itemsize if self.k else 0  # the drafter's hidden vector (MTP)
        return len(c.gdn_layers) * (rec + conv) + hid

    def capabilities(self) -> Capabilities:
        if self._caps is None:
            total, free = self._measure_memory()
            name = torch.cuda.get_device_name(self.device) if self.cuda else "cpu (torch reference kernels)"
            self._caps = Capabilities(
                backend="cuda", device=name, vocab_size=self.cfg.vocab_size,
                max_positions=self.cfg.max_position_embeddings, memory_total_bytes=total, memory_free_bytes=free,
                kv_bytes_per_token=self.kv_bytes_per_token, state_slot_bytes=self.state_slot_bytes,
                page_size_multiple=16 if self.inv and self.attn_kind == "flashinfer" else 1,  # our kernel's KV tile
                checkpoint_grid=CKPT_GRID, max_steps_in_flight=2, mtp_depth=self.k,
                cuda_graphs=self.use_graphs, checkpoints="every_n", images=self.mm, grammar_masks=self.grammar,
                encoder_rows=self.img.rows if self.mm else 0, encoder_budget=self.img.budget if self.mm else 0,
                decode_step_ms=self.stats.get("decode_step_ms", 0.0),
                decode_step_rows=self.stats.get("decode_step_rows", 0),
                prefill_step_ms=self.stats.get("prefill_step_ms", 0.0),
                prefill_step_tokens=self.stats.get("prefill_step_tokens", 0), precise_rows=self.fp8.precise,
                precise_step_ms=self.stats.get("precise_step_ms", 0.0), batch_invariant=self.inv,
                logprobs=self.logprobs)
        return self._caps

    def attention_counters(self) -> AttentionCounters:
        """FlashInfer's plan counters (`Runner.attention_counters`) over the target's instance (`attn`) and the
        drafter's chain's (`chain_attn`, MTP); none before `configure` or with the torch references. With them, the
        vision tower's plan waits (`vision_waits`)."""
        return vision_waits(sum_attention_counters(getattr(self, name, None) for name in ("attn", "chain_attn")),
                            getattr(self, "img", None))

    def _measure_memory(self) -> tuple[int, int]:
        """(total, free for pools): `memory_gb` when set, else on CPU `CPU_POOL_BYTES` and on CUDA total × utilisation
        − everything in use after loading − the persistent buffers configure() allocates − the CUDA-graph pool − the
        activation peak of profiled steps at max_num_batched_tokens − `PROFILE_MARGIN_BYTES`."""
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
                 "%.2f GiB of persistent buffers%s, %.2f GiB of CUDA graphs -> %.2f GiB for the KV and state pools "
                 "(utilisation %.2f)", total / 2**30, (total - free) / 2**30, peak / 2**30, self.B,
                 persistent / 2**30,
                 f" ({self.stats['record_bytes'] / 2**30:.2f} GiB of MTP window records)" if self.k else "",
                 graph_bytes / 2**30, avail / 2**30, self.memory_utilization)
        if avail <= 0:
            raise RuntimeError("no device memory left for the KV and state pools")
        return total, avail

    @torch.inference_mode()
    def _profile(self) -> tuple[int, int, int]:
        """Allocate tiny pools, capture the CUDA graphs and run representative steps; measure (activation peak,
        persistent buffer bytes, CUDA-graph pool bytes) and the step costs (`stats`), then release everything."""
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        s, b, ps = self.S, self.B, 16
        rows = max(1, min(s, b // 4))
        per = b // rows
        full = [(i + 1, 1, n) for i, n in enumerate(step_rows(b, self.max_model_len, s))]  # a full-budget step
        pages = max(sum(-(-(n + 2 + reserve_positions(2, self.k)) // ps) for _, _, n in full),
                    rows * -(-(per + 2 + 6 * self.L) // ps), s * 2) + 2
        pc = PoolConfig(page_size=ps, num_pages=pages, num_state_slots=3, max_num_seqs=s,
                        max_num_batched_tokens=b, max_model_len=self.max_model_len, eos_token_ids=(), mtp_k=self.k)
        base = torch.cuda.memory_allocated(self.device)
        self._allocate(pc)
        pools = sum(t.numel() * t.element_size() for t in (self.pools.kv, self.pools.conv, self.pools.rec,
                                                            *((self.pools.hid,) if self.pools.hid is not None else ())))
        allocated = torch.cuda.memory_allocated(self.device)
        persistent = allocated - base - pools
        graph_bytes = 0
        t0 = time.perf_counter()
        if self.use_graphs:
            self._run([lambda: self._synthetic(self._builder(), 1, [(1, 1, min(b, 80))], pages_from=1)])
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
                                                pages_from=1, free=[])])
            self._reset_tables()
        # 1) the whole budget in prefill rows (`step_rows`), with a checkpoint (state slot 2) when long enough;
        # 2) `rows` prefill rows with full-vocabulary top-p sampling, all sharing state slot 1 (contents do not matter,
        # sizes do); 3) those rows decoding. With PRECISE rows possible (the precise prefix), 1) and 2) are PRECISE:
        # their activations in BF16 and the dequantized weights are the peak.
        ck = [(0, CKPT_GRID, 2)] if b > CKPT_GRID else []
        drows = min(rows, b // self.L)
        pr = self.fp8.precise
        self._run([lambda: self._synthetic(builder, 1, full, pages_from=1, ckpts=ck, precise=pr),
                   lambda: self._synthetic(builder, 2, [(r + 1, 1, per) for r in range(rows)], pages_from=1,
                                           free=[r[0] for r in full], params=(1.0, 0.95, 0, 0.0), room=4 * self.L,
                                           precise=pr),
                   lambda: self._decode_plan(builder, 3, list(range(1, drows + 1)))])
        torch.cuda.synchronize()
        self.stats["profile_steps_s"] = round(time.perf_counter() - t0, 2)
        peak = max(0, torch.cuda.max_memory_allocated(self.device) - allocated)
        # Step costs for the scheduler's mixed-step budget (`Capabilities.decode_step_ms`): the device time of warm
        # steps (every kernel variant ran above), launched with `_run`'s `lead`: a lone greedy request's decode step
        # (one row: the profile's rows share a state slot and sample with top-p, which a served lone request does not)
        # and a full-budget prefill, twice each, the faster.
        with self.fp8.timing():
            res = self._run([lambda: self._synthetic(builder, 4, [(1, 1, per)], pages_from=1,
                                                     free=list(range(1, rows + 1)), room=4 * self.L),
                             lambda: self._decode_plan(builder, 5, [1]), lambda: self._decode_plan(builder, 6, [1]),
                             lambda: self._synthetic(builder, 7, full, pages_from=1, free=[1]),
                             lambda: self._synthetic(builder, 8, full, pages_from=1, free=[r[0] for r in full])],
                            lead=True)
        self.stats.update(decode_step_ms=round(min(r.gpu_busy_ns for r in res[1:3]) / 1e6, 3), decode_step_rows=1,
                          prefill_step_tokens=sum(r[2] for r in full),
                          prefill_step_ms=round(min(r.gpu_busy_ns for r in res[3:]) / 1e6, 3))
        if pr:  # the same full-budget prefill step with every row PRECISE (eager: the BF16 path's own cost)
            res = self._run([lambda: self._synthetic(builder, 9, full, pages_from=1, free=[r[0] for r in full],
                                                     precise=True),
                             lambda: self._synthetic(builder, 10, full, pages_from=1, free=[r[0] for r in full],
                                                     precise=True)], lead=True)
            self.stats["precise_step_ms"] = round(min(r.gpu_busy_ns for r in res) / 1e6, 3)
        log.info("step costs: a lone request's decode step %.2f ms, a %d-token prefill step %.2f ms (%s)%s",
                 self.stats["decode_step_ms"], self.stats["prefill_step_tokens"], self.stats["prefill_step_ms"],
                 "CUDA graphs" if self.use_graphs else "eager",
                 f", with PRECISE rows {self.stats['precise_step_ms']:.2f} ms (eager)" if pr else "")
        self.stats["record_bytes"] = self.tables.record_bytes()
        self._release()
        return peak, max(0, persistent), graph_bytes

    # -- configure ---------------------------------------------------------------------------------------------

    def configure(self, pools: PoolConfig) -> None:
        if pools.mtp_k != self.k:
            raise ValueError(f"configured with mtp_k {pools.mtp_k}; this runner drafts {self.k}")
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
        self.stats["record_bytes"] = self.tables.record_bytes()
        if self.cuda:
            self.stats["memory_allocated_bytes"] = torch.cuda.memory_allocated(self.device)
            self.stats["memory_reserved_bytes"] = torch.cuda.memory_reserved(self.device)
        log.info("pools: %d pages of %d tokens, %d state slots%s; warm-up %.1f s%s", pools.num_pages,
                 pools.page_size, pools.num_state_slots,
                 f"; MTP window records {self.stats['record_bytes'] / 2**30:.2f} GiB" if self.k else "",
                 self.stats["warmup_s"],
                 f"; {len(self._graphs)} CUDA graphs captured in {self.stats['capture_s']:.1f} s"
                 if self.use_graphs else "")
        timings = {k: v for k, v in self.stats.items() if k.endswith("_s")}
        log.info("runner startup (s): %s", " ".join(f"{k.removesuffix('_s')}={v}" for k, v in timings.items()))

    def _builder(self) -> PlanBuilder:
        pc = self.pc
        return PlanBuilder(pc.max_num_seqs, pc.max_num_batched_tokens, pc.page_size, pc.max_model_len, self.k)

    def _allocate(self, pc: PoolConfig) -> None:
        if self._allocated:
            self._release()
        cfg, dev, dt = self.cfg, self.device, self.dtype
        self.pc = pc
        self.ps = pc.page_size
        slots = pc.max_num_seqs + 1
        self.reserve = reserve_positions(2, self.k)
        self.max_pages = -(-(pc.max_model_len + self.reserve) // pc.page_size) + 1
        n_attn, n_gdn = len(cfg.attn_layers) + (1 if self.k else 0), len(cfg.gdn_layers)
        kv = torch.zeros((n_attn, pc.num_pages, 2, pc.page_size, cfg.num_kv_heads, cfg.head_dim), dtype=dt,
                         device=dev)
        conv = torch.zeros((pc.num_state_slots, n_gdn, cfg.conv_dim, cfg.conv_kernel - 1), dtype=dt, device=dev)
        rec = torch.zeros((pc.num_state_slots, n_gdn, cfg.lin_v_heads, cfg.lin_v_dim, cfg.lin_k_dim),
                          dtype=torch.float32, device=dev)
        hid = torch.zeros((pc.num_state_slots, cfg.hidden_size), dtype=dt, device=dev) if self.k else None
        self.pools = Pools(kv=kv, conv=conv, rec=rec, page_size=pc.page_size, hid=hid)
        self.tables = _Tables(slots, self.max_pages, dev, self.k, cfg, dt)
        self.mirror = _Mirror(slots)
        self.rotary = Rotary(cfg, pc.max_model_len + self.reserve + 1, dev, dt)
        tmax = max(pc.max_num_batched_tokens, pc.max_num_seqs * self.L)
        smax = min(tmax, pc.max_num_seqs * self.L)
        self.static = _Static(tmax, pc.max_num_seqs, smax, self.L, self.k, cfg.hidden_size, dt, dev, mm=self.mm,
                              mrope=self.mrope)
        max_page_writes = page_write_capacity(pc.max_num_seqs, pc.max_num_batched_tokens, pc.page_size,
                                              pc.max_model_len, self.reserve)
        cap_i = 4 * tmax + 64 * slots + 3 * max_page_writes + (STOP_W + 4) * slots + 1024
        if self.gdn_kind == "flashinfer":  # the GDN prefill's cut rows on checkpoint steps (`flashinfer.gdn_segments`)
            cap_i += tmax + (pc.max_num_batched_tokens // CKPT_GRID + 2) * (4 * slots + 1)
        if self.mm:
            cap_i += 2 * self.img.budget + 8 * slots + 64  # ENCODE's rows, the resumes' rows and rope deltas
        if self.logprobs:  # the LOGPROBS samples' indices
            cap_i += smax + LP_CHUNK
        self._staging = [_Staging(dev, cap_i, 3 * slots, slots, self.static.size, self.static.out.numel(),
                                  self.img.px_bytes if self.mm else 0) for _ in range(2)]
        if self.inv and self.attn_kind == "flashinfer":  # our attention's plan tables beside the chain's
            from ...kernels.attn_window import FullAttention

            for st in self._staging:
                st.h32 = torch.empty(16 * slots + 32 + FullAttention.staging_bound(
                    pc.max_num_seqs, pc.max_num_batched_tokens, cfg.num_heads // cfg.num_kv_heads, cfg.num_kv_heads,
                    self.L, self.attn_split_keys, self.max_model_len + self.L), dtype=torch.int32,
                    pin_memory=st.h32.is_pinned())
                st.n32 = st.h32.numpy()
        self.gs.allocate(pc.max_num_seqs * self.L)
        self.lp = LogprobsPass(smax, cfg.vocab_size, self.gs.rows if self.gs.sparse else 0, dev) \
            if self.logprobs else None
        self.chain_attn: Any = None
        workspace = None  # FlashInfer's float workspace, shared by its instances and the tower's attention
        if self.attn_kind == "flashinfer":
            from ...kernels.flashinfer import WORKSPACE_BYTES, FlashInferAttention

            max_rows = max(self.decode_buckets) if self.use_graphs else 0
            if self.inv:  # our kernel for the full layers and the MTP layer's pass over the step (batch invariance)
                from ...kernels.attn_window import FullAttention

                self.attn: Any = FullAttention(self.pools, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, dev,
                                               max_tokens=pc.max_num_batched_tokens, max_rows=pc.max_num_seqs,
                                               max_kv=self.max_model_len + self.L, scale=cfg.head_dim ** -0.5,
                                               qo_per_row=self.L, split_keys=self.attn_split_keys)
                workspace = torch.zeros(WORKSPACE_BYTES, dtype=torch.uint8, device=dev)
            else:
                self.attn = FlashInferAttention(self.pools, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, dt, dev,
                                                max_rows=max_rows, max_pages=self.max_pages, qo_per_row=self.L)
                workspace = self.attn.workspace
            if self.k:  # the drafter's chain: one query per sample row, planned in the same steps as the target
                self.chain_attn = FlashInferAttention(self.pools, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, dt,
                                                      dev, max_rows=max_rows, max_pages=self.max_pages,
                                                      workspace=workspace)
                from ...kernels.flashinfer import self_check

                # upper-bound planning (MTP): the target's FlashInfer, or the chain's beside our kernel
                self.stats["flashinfer_check"] = self_check(self.chain_attn if self.inv else self.attn, dt)
        else:
            self.attn = TorchAttention(self.pools, cfg.head_dim)
            if self.k:
                self.chain_attn = TorchAttention(self.pools, cfg.head_dim)
        if self.gdn_kind == "flashinfer":
            from ...kernels.flashinfer import flashinfer_gdn_prefill

            self.gdn = GDN(self.pools, flashinfer_gdn_prefill)  # both paths: FlashInfer's per-sequence route
        else:
            self.gdn = GDN(self.pools, torch_gdn_prefill)
        if self.mm:  # the encoder pool and the tower's attention (its graphs: `_capture_all`)
            fi = self.attn_kind == "flashinfer"
            self.img.allocate(cfg.hidden_size, fi, workspace=workspace)
        self._arange_pages = torch.arange(self.max_pages, device=dev)
        self._vocab_ids = torch.arange(cfg.vocab_size, device=dev)
        # Mid-chunk checkpoints: a workspace laid out every 64 tokens of every prefill row, at most B / 64 per step.
        self._ckpt_ws = torch.empty((max(1, pc.max_num_batched_tokens // CKPT_GRID), cfg.lin_v_heads, cfg.lin_v_dim,
                                     cfg.lin_k_dim), dtype=torch.float32, device=dev) if n_gdn else None
        self._bctx = ForwardContext(None, self.attn, self.gdn, self.rotary, self.pools)  # breakable graphs' context
        self._cctx = ForwardContext(None, self.chain_attn, self.gdn, self.rotary, self.pools) if self.k else None
        self._flip = 0
        self._last_end: Any = None
        self._allocated = True

    def _release(self) -> None:
        self._graphs = {}
        self._pool = None
        for name in ("pools", "tables", "rotary", "_staging", "attn", "chain_attn", "gdn", "_arange_pages",
                     "_vocab_ids", "_last_end", "_ckpt_ws", "static", "_bctx", "_cctx"):
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
        """Forget every admission (after internal plans; also for harnesses that reuse a runner across
        schedulers, whose last FREEs never reach it)."""
        if self.cuda:
            torch.cuda.synchronize()
        self.tables.reset()
        self.mirror = _Mirror(self.pc.max_num_seqs + 1)
        self._last_end = None
        for st in self._staging:
            st.handle = None

    def reset(self) -> None:
        """Public `_reset_tables` (tests)."""
        self._reset_tables()

    def _warmup(self) -> None:
        """Run the step path once per kernel variant (FlashInfer JIT, the GDN prefill and the checkpointed prefill's
        cut rows, the copies, every sampler path, the split path and each grammar heads' sampling, and under MTP
        verification, the drafter and its chain, FOLD and the drafter's KV recompute of a resume), then clear the
        tables. Sized to fit the configured pools. Eager: graphs are captured afterwards."""
        pc = self.pc
        n = max(1, min(4, pc.max_num_seqs, pc.num_state_slots - 1))
        b = self._builder()
        budget = pc.max_num_batched_tokens
        lens = [min(80, max(1, budget // 2)), 1, min(17, max(1, budget // 4)), min(3, max(1, budget // 8))][:n]
        while sum(lens) > budget:
            lens.pop()
        n = len(lens)
        # greedy; top-k; top-p + min-p; plain (full-vocabulary Gumbel-max): every sampler path
        params = [(0.0, 1.0, 0, 0.0), (1.0, 1.0, 5, 0.0), (0.8, 0.9, 0, 0.05), (1.0, 1.0, 0, 0.0)]
        heads = [VARIANT_HEADS[v] for v in self.gs.variants[1:]] if self.grammar else []  # the grammar heads' paths
        room = len(heads) * self.L  # their decode steps (verify windows under MTP), after the two a row has room for
        need = sum(-(-(x + 2 + 2 * self.L + room) // pc.page_size) for x in lens)
        if need >= pc.num_pages:
            log.warning("pools too small for the warm-up; skipping it")
            return
        rows = [(i + 1, i + 1, lens[i]) for i in range(n)]
        # with PRECISE rows possible, the first row is one: a step of both paths (the precise prefix)
        plans = [lambda: self._synthetic(b, 1, rows, pages_from=1, params=params[:n], room=room,
                                         precise=[self.fp8.precise and i == 0 for i in range(n)])]
        if n * self.L <= budget:  # with grammar masks: the split path (forward, then sampling under masks)
            plans.append(lambda: self._decode_plan(b, 2, [r[0] for r in rows], ckpt_row=0 if self.k else None,
                                                   ckpt_dst=n + 1 if self.k and n + 1 < pc.num_state_slots else 0,
                                                   masked=self.grammar))
            for j in range(len(heads)):  # and each grammar heads' sampling path once (`_run`), a decode step apiece
                plans.append(lambda j=j: self._decode_plan(b, 8 + j, [r[0] for r in rows], masked=True))
        else:
            heads = []
        many = min(32, pc.max_num_seqs, pc.num_state_slots - 1, pc.max_num_batched_tokens // 4)
        admitted = [r[0] for r in rows]
        if many > n and many * -(-(6 + 2 * self.L) // pc.page_size) < pc.num_pages:
            first = list(admitted)
            plans.append(lambda: self._synthetic(b, 3, [(i + 1, i + 1, 4) for i in range(many)], pages_from=1,
                                                 free=first, fold=first if self.k else ()))
            admitted = list(range(1, many + 1))
        # The prefix-cache paths: a checkpointed prefill (its rows cut into calls at the checkpoints), then a resume
        # from that checkpoint with copies of a page and a state slot.
        long = 2 * CKPT_GRID + 3
        spare = pc.num_state_slots - 1  # never slot 1, the live slot used below
        npg = -(-(long + 2 + 2 * self.L) // pc.page_size)
        if spare > 1 and long <= pc.max_num_batched_tokens and 2 * npg + 2 < pc.num_pages:
            ck = [(0, CKPT_GRID, spare)] + ([(0, long, spare - 1)] if spare - 1 > 1 else [])
            plans.append(lambda: self._synthetic(b, 4, [(1, 1, long)], pages_from=1, free=admitted, ckpts=ck))
            plans.append(lambda: self._resume_plan(b, 5, slot=1, state_slot=1, src_state=spare, seq_len=CKPT_GRID,
                                                   pages=list(range(1 + npg, 1 + 2 * npg)), copy_page=(1, 1 + npg)))
        img = min(16, self.img.budget, pc.max_num_batched_tokens - 2) if self.mm else 0
        img_pages = -(-(img + 4 + 2 * self.L) // pc.page_size)
        if img > 0 and img_pages + 2 < pc.num_pages:
            # images: the encoder (its attention's JIT), image rows, M-RoPE; on the pool's last pages, in a slot no
            # earlier plan holds (or one it frees)
            def image_step() -> StepPlan:
                m = self.mirror
                free = np.flatnonzero(~m.admitted[1:]) + 1
                slot = int(free[0]) if len(free) else 1
                return self._image_plan(b, 6, slot=slot, image_tokens=img, pages_from=pc.num_pages - img_pages,
                                        free=[] if len(free) else [1])
            plans.append(image_step)
        graphs, self.use_graphs = self.use_graphs, False
        try:
            self._run(plans, heads=[NO_HEADS, *heads])
        finally:
            self.use_graphs = graphs
        self._reset_tables()

    def _synthetic(self, b: PlanBuilder, step: int, rows: list[tuple[int, int, int]], *, pages_from: int,
                   free: list[int] = (), params: Any = None, ckpts: list[tuple[int, int, int]] = (),
                   fold: list[int] = (), room: int = 0, precise: bool | list[bool] = False) -> StepPlan:
        """A plan admitting (slot, state slot, prompt length) rows as complete prompts (STATE_ZERO | SAMPLE), with
        optional (row, offset, destination state slot) checkpoints, FOLDs (all records) and FREEs before. Each row
        gets pages for two decode steps after it, plus `room` positions. `precise`: the rows are PRECISE (one flag per
        row, or one for all)."""
        b.begin()
        for s in fold:
            b.add_fold(s, FOLD_ALL)
        for s in free:
            b.add_free(s)
        page = pages_from
        for i, (slot, ss, n) in enumerate(rows):
            p = params[i] if isinstance(params, list) else (params or (0.0, 1.0, 0, 0.0))
            b.add_admit(slot, ss, 0, p[0], p[1], p[2], p[3], seed=1234 + slot, max_len=self.pc.max_model_len,
                        stop=[])
            npg = -(-(n + 2 + 2 * self.L + room) // self.ps)
            b.add_pages(slot, 0, list(range(page, page + npg)))
            page += npg
            pre = precise[i] if isinstance(precise, list) else precise
            b.add_explicit_row(slot, self._synthetic_tokens(slot, n), 0, STATE_ZERO | SAMPLE | (PRECISE if pre else 0))
        for row, off, dst in ckpts:
            b.add_ckpt(row, off, dst)
        return b.finish(step)

    def _image_plan(self, b: PlanBuilder, step: int, *, slot: int, image_tokens: int, pages_from: int,
                    free: list[int]) -> StepPlan:
        """A plan admitting a prompt of one text token, one synthetic image of `image_tokens` tokens (its pixels from
        `ImageEncoder.synthetic`) and one text token, with its ENCODE and M-RoPE deltas (profile and warm-up)."""
        from ...vision.family import ImageSize, Span
        from ...vision.qwen import QwenVision

        grid, hw = self.img.synthetic(image_tokens)
        p = self.img.cfg.patch
        n = image_tokens + 2
        rows = np.arange(image_tokens, dtype=np.int64) % self.img.rows
        fam = QwenVision(start_id=0, pad_id=0, end_id=0, patch=p, merge=2, temporal=2, min_pixels=1,
                         max_pixels=1 << 40, mean=(0.5,) * 3, std=(0.5,) * 3, rescale=1 / 255, mrope=True)
        pos, rope_off = fam.positions(n, [Span(1, ImageSize(hw[0], hw[1], grid, image_tokens))])
        b.begin()
        for s_ in free:
            if self.k:
                b.add_fold(s_, FOLD_ALL)
            b.add_free(s_)
        m = self.mirror
        taken = m.state_slot[m.admitted & ~np.isin(np.arange(len(m.admitted)), free)]
        ss = int(np.setdiff1d(np.arange(1, self.pc.num_state_slots), taken)[0])
        b.add_admit(slot, ss, 0, 0.0, 1.0, 0, 0.0, seed=7 + slot, max_len=self.pc.max_model_len, stop=[],
                    rope_off=rope_off if self.mrope else 0)
        npg = -(-(n + 2 + 2 * self.L) // self.ps)
        b.add_pages(slot, 0, list(range(pages_from, pages_from + npg)))
        b.add_encode(1 << 62, image_tokens, grid, hw, 0, rows)
        emb = np.full(n, -1, np.int32)
        emb[1:1 + image_tokens] = rows
        rope = (pos - np.arange(n)[None, :]).astype(np.int32) if self.mrope else None
        b.add_explicit_row(slot, self._synthetic_tokens(slot, n), 0, STATE_ZERO | SAMPLE, emb=emb, rope=rope)
        return b.finish(step)

    def _synthetic_tokens(self, slot: int, n: int, start: int = 0) -> np.ndarray:
        return ((np.arange(start, start + n, dtype=np.int64) * 7919 + slot) % self.cfg.vocab_size).astype(np.int32)

    def _resume_plan(self, b: PlanBuilder, step: int, *, slot: int, state_slot: int, src_state: int, seq_len: int,
                     pages: list[int], copy_page: tuple[int, int]) -> StepPlan:
        """Free `slot`, readmit it at `seq_len` from checkpoint `src_state` (STATE_COPY), copy one page (KV_COPY)
        and prefill a few tokens with an end-of-step checkpoint back into `src_state`."""
        b.begin()
        if self.k:
            b.add_fold(slot, FOLD_ALL)
        b.add_free(slot)
        b.add_admit(slot, state_slot, seq_len, 0.0, 1.0, 0, 0.0, seed=99, max_len=self.pc.max_model_len, stop=[])
        b.add_pages(slot, 0, pages)
        b.add_kv_copy(*copy_page)
        b.add_state_copy(src_state, state_slot)
        b.add_explicit_row(slot, self._synthetic_tokens(slot, 3, seq_len), seq_len, SAMPLE)
        b.add_ckpt(0, CKPT_END_OF_STEP, src_state)
        return b.finish(step)

    def _decode_plan(self, b: PlanBuilder, step: int, slots: list[int], ckpt_row: int | None = None,
                     ckpt_dst: int = 0, masked: bool = False) -> StepPlan:
        b.begin()
        b.add_device_rows(np.asarray(slots, np.int32), np.ones(len(slots), bool) if masked else None)
        if ckpt_row is not None and ckpt_dst:
            b.add_ckpt(ckpt_row, CKPT_END_OF_STEP, ckpt_dst)
        return b.finish(step)

    def _run(self, plans: list[Callable[[], StepPlan]], lead: bool = False,
             heads: list[Heads] = ()) -> list[StepResult]:
        """Drive internal plans (profile, warm-up) in the engine's call order: launch N+1, wait N, sample N+1.
        Each plan is built just before it is launched: PlanBuilder reuses its buffers two plans later. Returns the
        steps' results. Masks allow every token, and each masked step samples on the next heads of `heads` (then
        NO_HEADS; `GrammarSampling.choose`). `lead`: a device sleep before each step, longer than its CPU launch, so its
        timing is the device's alone (CUDA)."""
        prev, hook = None, self.logits_hook
        self.logits_hook = None
        out: list[StepResult] = []
        it = iter(heads)
        try:
            for make in plans:
                plan = make()
                if lead and self.cuda:
                    torch.cuda._sleep(_LEAD_CYCLES)
                h = self.launch_forward(plan, _internal=True)
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
        """Capture every graph, largest first within each kind (later captures reuse the shared pool's memory): each
        token bucket's and decode bucket's graphs (`graphs.graph_names`), then the vision tower's. A bucket's capture
        runs a synthetic step on scratch slots and leaves garbage in them (the eager breaks run during capture, the
        graph segments do not), so the tables are reset after."""
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
                rows = [(i + 1, 1, n) for i, n in enumerate(step_rows(t, pc.max_model_len, pc.max_num_seqs))]
                npg = sum(-(-(n + 2 + 2 * self.L) // self.ps) for _, _, n in rows)
                if sum(n for _, _, n in rows) < t or npg + 1 >= pc.num_pages:  # unreachable, or no room
                    continue
                for name in graph_names("break", **naming):
                    self._capture_one((name, t), lambda rows=rows: self._synthetic(b, 1, rows, pages_from=1))
            for n in sorted(self.decode_buckets, reverse=True):
                for name in graph_names("decode", **naming):
                    self._capture_one((name, n), lambda n=n: self._decode_capture_plan(b, n))
            if self.mm:
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
        """Slots 1..n admitted and sampled, at a short context on page 1, sharing state slot 1 (scratch); then a
        decode plan of n device-input rows."""
        m, t = self.mirror, self.tables
        sl = np.arange(1, n + 1)
        m.admitted[sl] = m.sampled[sl] = True
        m.seq_len[sl] = 3
        m.state_slot[sl] = 1
        span = -(-(3 + 2 * self.L) // self.ps)
        t.page_table[1:n + 1, :span] = 1
        t.state_slot[1:n + 1] = 1
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
        st = self._staging[self._flip]
        h = _Handle()
        h.parity, h.step_id, h.stage, h.internal = self._flip, plan.step_id, st, _internal
        h.observe = self.draft_head is not None and self.draft_head.reduced and not _internal
        self._flip ^= 1
        if st.handle is not None and st.handle.result is None:
            # Only if the two-steps-in-flight call order was broken: the buffer's previous step must have run.
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
        if len(plan.enc_key) and not self.mm:
            raise PlanError(f"step {plan.step_id}: ENCODE, but this runner has no vision tower")
        enc = self.img.stage(plan, st, _internal) if len(plan.enc_key) else None
        lay = self._layout(plan, st)
        if lay.zero_states is not None:
            fx.n_zero, fx.o_zero = len(lay.zero_states), st.put(lay.zero_states)
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
        h.mtp_rows = None
        h.split = None
        if lay.n_mtp:  # resolved at wait(): the device advance of each verify row
            sr = np.flatnonzero(lay.r_kind[lay.c_row] == _VERIFY)
            slots = lay.r_slot[lay.c_row[sr]]
            h.mtp_rows = (sr, slots, self.mirror.epoch[slots].copy())
        if lay.n:
            if lay.resume is not None:
                self._mtp_resume_kv(lay, st)
            with plan_bracket(h, self.cuda and _capture is None):
                meta = self._row_meta(lay, st, h.parity, kind, size)
            if _capture is not None:
                self._capture(_capture[0], size, meta, lay)
            elif split:  # the forward now, its sampling in launch_sample (module docstring)
                ctx = self._part(kind, size, meta, lay, plan.step_id, "fwd")
                h.split = (kind, size, meta, lay, ctx)
                h.events[1].record()  # the forward's end: until the sampling starts, the device idles
                h.fwd_end = h.events[1] if self.cuda else None  # its sampling part timed (`step_timing`)
                return h
            elif kind == "eager":
                self._eager(meta, lay, plan.step_id)
            else:
                g = self._graphs[(kind, size)]
                if kind == "break":
                    self._bctx.meta = meta
                    if self.k:
                        self._cctx.meta = meta.chain
                g.replay()
            self._logprobs(h, lay, NO_HEADS)
            self._after_body(lay, st)
        self._outputs(h, lay)
        return h

    def _outputs(self, h: _Handle, lay: _Layout, split: bool = False) -> None:
        """After the body: the step's outputs to its device buffer and on to the host (copy stream); the timing
        events (ev[1] and ev[2] bracket the gap between a split step's forward and its sampling, empty otherwise)."""
        ev, st = h.events, h.stage
        if lay.n_sample:
            st.dout.copy_(self.static.out)  # the next step's body rewrites the static outputs
        if not split:
            ev[1].record()
            ev[2].record()
        ev[3].record()
        if lay.n_sample:
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

    def _part(self, kind: str, size: int, meta: StepMeta, lay: _Layout, step_id: int, part: str,
              ctx: ForwardContext | None = None, heads: Heads = NO_HEADS) -> ForwardContext | None:
        """One part of a split step (`_body`): its graph, or eagerly (returns the eager context the parts share); the
        sampling on the heads `heads` names (`runner.cuda.heads`)."""
        if kind == "eager":
            if part == "fwd":
                return self._eager(meta, lay, step_id, part="fwd")
            if self.k:
                self._cctx.meta = meta.chain
            t, st, rp, cr = self._sizes("eager", lay.T, lay)
            with self.fp8.activations(t, lay.n_precise):  # the drafter runs over the step's layout here
                self._body(ctx, t, st, rp, cr, "eager", lay, part="smp", heads=heads)
            return ctx
        if kind == "break":
            self._bctx.meta = meta
            if self.k:
                self._cctx.meta = meta.chain
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
        """A fused step sampled inside its body (`launch_forward`). A split one samples now: its heads chosen from its
        masks on the host (`runner.cuda.heads`; internal steps take `_heads`), its masks and the sparse head's ids
        copied to the device on the copy stream (while the forward may still run; a step without masks, always split,
        copies none and samples every row unmasked on the dense head), then its sampling part (graph or eager), its
        checkpoints (`_after_body`) and the outputs' copy."""
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
        self._after_body(lay, h.stage)
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
            (ev[4] if h.n_sample else ev[3]).synchronize()
        L = self.L
        tokens = np.full((h.n_sched, L), -1, np.int32)
        n_out = np.zeros(h.n_sched, np.int32)
        drafts = np.full((h.n_sched, self.k), -1, np.int32) if self.k else None
        status = None
        if h.n_sample:
            n, s = h.n_sample, self.static.s
            out = h.stage.ntok
            toks = out[:s * L].reshape(s, L)[:n]
            nn = out[s * L:s * (L + 1)][:n] if self.k else np.ones(n, np.int64)
            st = out[s * (L + 1):s * (L + 2)][:n]
            tokens[h.s_plan_rows] = toks
            n_out[h.s_plan_rows] = nn
            if self.k:
                drafts[h.s_plan_rows] = out[s * (L + 2):s * (L + 2 + self.k)].reshape(s, self.k)[:n]
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
                m.slack[sl] -= self.k
        lp = self.lp.result(h.parity, h.lp[0], L) if h.lp is not None else None
        h.result = StepResult(tokens=tokens, n_out=n_out, status=status, drafts=drafts, logprobs=lp,
                              **step_timing(h, self.cuda))
        return h.result

    def _mode(self, lay: _Layout, split: bool = False) -> tuple[str, int]:
        """Which body runs this step: a decode graph (every row a device-input row), a breakable graph (by token
        count), or the eager body; a split step (grammar masks, or always split) needs both of its graphs."""
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
            return size * self.L, size * self.L if self.k else size, size, size
        if kind == "break":
            rp = min(size, self.S)
            return size, min(size, self.S * self.L), rp, rp
        return lay.T, lay.n_stok, lay.n, lay.n_sample

    def _ids(self, t: int) -> torch.Tensor:
        """The tokens' input ids: explicit tokens, or the slot's pending token and drafts on device rows."""
        s, tb = self.static, self.tables
        slot, tok = s.slot[:t], s.tok[:t]
        if self.k:
            w = (-1 - tok).clamp(min=0)  # device tokens: −1 − (window index)
            dev = torch.where(w == 0, tb.pending.index_select(0, slot), tb.drafts[slot, (w - 1).clamp(min=0)])
            return torch.where(tok >= 0, tok, dev)
        return torch.where(tok >= 0, tok, tb.pending.index_select(0, slot))

    def _positions(self, ctx: ForwardContext, t: int, mode: str) -> torch.Tensor:
        """The tokens' sequence positions (device rows: from the device seq_len), and the meta's KV slots and RoPE
        positions. Both parts of a split step compute them: no table changes in between."""
        s, tb = self.static, self.tables
        slot, pin = s.slot[:t], s.pos[:t]
        if self.k:
            pos = torch.where(pin >= 0, pin, tb.seq_len.index_select(0, slot) + (-1 - pin).clamp(min=0))
        else:
            pos = torch.where(pin >= 0, pin, tb.seq_len.index_select(0, slot))
        m = ctx.meta
        m.positions, m.kv_page, m.kv_off = pos, tb.page_table[slot, pos // self.ps].long(), pos % self.ps
        if self.mrope:  # RoPE: the positions plus each token's deltas, or the slot's ADMIT offset on device rows
            off = tb.rope_off.index_select(0, slot)
            m.rope = pos + off if mode == "decode" else pos[None] + torch.where(pin[None] >= 0, s.rd[:, :t], off[None])
        return pos

    def _body(self, ctx: ForwardContext, t: int, st: int, rp: int, cr: int, mode: str, lay: _Layout | None = None,
              hook: Any = None, part: str = "all", heads: Heads = NO_HEADS) -> None:
        """Inputs from the static block, the forward, the logits of the sample tokens, sampling, verification and
        drafting (MTP), and the seq_len advance, for `t` tokens, `st` sample tokens, `rp` rows and `cr` sample rows
        (padding included). Graph-safe: every shape is a host int, every input a static buffer or table.

        `part`: "all" runs the whole step (a fused step); a split step runs "fwd" (to the hidden states, kept in the
        static `hid_out`) in `launch_forward` and "smp" (the rest) in `launch_sample`, on the heads `heads` names
        (`runner.cuda.heads`)."""
        s, tb, k, L = self.static, self.tables, self.k, self.L
        splan = (lay.splan_sp if heads[0] else lay.splan) if lay is not None else None
        pos = self._positions(ctx, t, mode)
        m = ctx.meta
        r_slot = s.r_slot[:rp]
        if part != "smp":
            ids = self._ids(t)
            embeds = None
            if self.mm and mode != "decode":  # image tokens take their encoder rows
                embeds = self.img.embeds(s.emb[:t], self.model.embed_tokens(ids))
            if mode == "decode":  # every row a device-input row: the row metadata comes from the static rows
                m.row_state = tb.state_slot.index_select(0, r_slot)
                m.row_cu = s.arange64[:rp + 1] * L
                if k:
                    m.row_state_i32 = m.row_state.to(torch.int32)
                    m.mtp = self._mtp_meta(rp, r_slot)
                    s.kv_len[:rp] = tb.seq_len.index_select(0, r_slot) + s.r_nq[:rp]
                    if hasattr(self.attn, "fix_lengths"):
                        self.attn.fix_lengths(m.attn_plan, rp, s.kv_len[:rp])
                else:
                    m.decode_i32 = (m.row_state.to(torch.int32).view(rp, 1), s.arange32[:rp + 1], s.ones32[:rp])
                self.attn.graph_page_indices(rp, tb.page_table, r_slot)
            if not k:
                sel = s.s_idx[:st]
                if hook is None:
                    hidden = self.model(ids, ctx, select=sel, embeds=embeds)
                else:
                    full = self.model(ids, ctx, embeds=embeds)
                    hook(full)
                    hidden = full.index_select(0, sel)
                if part == "fwd":
                    s.hid_out[:st].copy_(hidden)
                    return
            else:
                hidden = self.model(ids, ctx, embeds=embeds)
                if hook is not None:
                    hook(hidden)
                if mode != "decode" or part == "fwd":
                    s.hid_out[:t].copy_(hidden)
                if part == "fwd":
                    return
        else:
            hidden = s.hid_out[:t] if k else s.hid_out[:st]
        mask = (self.gs.mask_src, s.s_mask[:st]) if part == "smp" else None
        if not k:
            if st:
                sel = s.s_idx[:st]
                self._sample(hidden, pos.index_select(0, sel), s.s_slot[:st], s.out_tok[:st, 0], s.out_st[:st],
                             tb.pending, splan, mask=mask, sparse=heads[0])
            tb.seq_len.index_add_(0, r_slot, s.r_nq[:rp])
            return
        sel = s.s_idx[:st]
        if st:
            self._sample(hidden.index_select(0, sel), pos.index_select(0, sel), s.s_slot[:st], s.y[:st], s.y_st[:st],
                         None, splan, mask=mask, sparse=heads[0])
        a, ver, _ = mtp_verify(s, tb, r_slot, rp, cr, st, k)
        kind = s.r_kind[:rp]
        ar = s.arange64[:L]
        r_start, r_nq = s.r_start[:rp], s.r_nq[:rp]
        # The window's target hidden states, kept for FOLD and checkpoints; an explicit row's last one goes to its
        # live state slot (the hidden vector a checkpoint of it holds).
        tok_win = (r_start[:, None] + ar[None, :]).clamp(max=t - 1)
        tb.rec_hid.index_copy_(0, r_slot, torch.where(ver[:, None, None], hidden[tok_win],
                                                      tb.rec_hid.index_select(0, r_slot)))
        live = m.row_state if part == "all" and m.row_state is not None and mode == "decode" else \
            tb.state_slot.index_select(0, r_slot)
        expl = (kind != _VERIFY) & (r_slot > 0)
        last = (r_start + r_nq - 1).clamp(min=0, max=t - 1)
        self.pools.hid.index_copy_(0, live, torch.where(expl[:, None], hidden[last],
                                                        self.pools.hid.index_select(0, live)))
        c_row = s.c_row[:cr]
        dsrc = torch.where(ver, r_start + a, r_start + r_nq - 1).clamp(min=0, max=t - 1)  # the drafter's source token
        tb.fold_n.index_copy_(0, r_slot, torch.where(ver, 1 + a, tb.fold_n.index_select(0, r_slot).long())
                              .to(torch.int32))
        # The drafter over the step's own layout (the target's attention plan), then its chain.
        s.y_tok.index_copy_(0, s.s_put[:st], s.y[:st])
        dtok = s.dtok[:t]
        ids_d = torch.where(dtok >= 0, dtok, s.y_tok[:t])
        mtp = self.model.mtp
        emb_d = self.model.embed_tokens(ids_d)
        if self.mm and mode != "decode":  # at a position whose next token is an image token: that token's row
            emb_d = self.img.embeds(s.demb[:t], emb_d)
        out_d = mtp(emb_d, hidden, ctx)
        if not cr:  # an eager step without sample rows: the drafter's KV only
            return
        src = dsrc[c_row]
        c_slot = s.c_slot[:cr]
        p0 = pos[src]
        h_d = out_d[src]
        # The MTP output at position i predicts the token at i + 2: the drafts fill p0 + 2, p0 + 3, ..., and each is
        # sampled with that position's noise (`_sample` keys qpos + 1), the noise the target's sample there takes
        # (a draft keyed with any other position's noise would cost acceptance at temperature > 0). The universe
        # (`heads[1]`): scored over the grammar's universe instead of the draft head (`runner.cuda.heads`)
        uni = heads[1]
        dplan = (lay.dplan_uni if uni else lay.dplan) if lay is not None else None
        self._sample(h_d, p0 + 1, c_slot, s.drafts[0, :cr], s.d_st[:cr], None, dplan, draft=True, uni=uni)
        if k > 1:
            cctx = self._cctx
            cm = cctx.meta
            if mode == "decode":
                self.chain_attn.graph_page_indices(cr, tb.page_table, c_slot)
            c_off = tb.rope_off.index_select(0, c_slot) if self.mrope else None
            for j in range(1, k):
                pj = p0 + j
                cm.positions, cm.kv_page, cm.kv_off = pj, tb.page_table[c_slot, pj // self.ps].long(), pj % self.ps
                if c_off is not None:  # the chain drafts text after the last image
                    cm.rope = pj + c_off
                s.d_len[:cr] = pj + 1
                cm.kv_len_dev = s.d_len[:cr]
                if mode == "decode" and hasattr(self.chain_attn, "fix_lengths"):
                    self.chain_attn.fix_lengths(cm.attn_plan, cr, s.d_len[:cr])
                h_d = mtp(self.model.embed_tokens(s.drafts[j - 1, :cr]), h_d, cctx)
                self._sample(h_d, pj + 1, c_slot, s.drafts[j, :cr], s.d_st[:cr], None, dplan, draft=True, uni=uni)
        dr_rows = s.drafts[:, :cr].t()
        tb.drafts.index_copy_(0, c_slot, dr_rows)
        s.out_dr[:cr] = dr_rows  # returned with the outputs (StepResult.drafts)

    def _sample(self, hidden: torch.Tensor, qpos: torch.Tensor, slots: torch.Tensor, out: torch.Tensor,
                status: torch.Tensor, pending: torch.Tensor | None, splan: Any, draft: bool = False,
                mask: tuple[torch.Tensor, torch.Tensor] | None = None, sparse: bool = False, uni: bool = False) -> None:
        """Sample one token per row of `hidden` (its logits) at position qpos + 1 (`sampler.draw`, which writes
        `pending[slot]` too when given). `draft`: the logits of the drafter's head, over its token ids (`DraftHead`), or
        with `uni` the universe's (`SubsetHead`). `sparse`: the target's head scores this step's union only
        (`_sparse_logits`), the masks read through its ids. `mask`: (the mask buffer, each row's mask in it or −1)."""
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
        """[n, rows] fp32: `GrammarSampling.sparse_logits`, with the GEMM and the widening of
        `Qwen35ForCausalLM.logits`; graph-safe. The masked choice is the dense head's up to the GEMM's rounding: cuBLAS
        picks its kernel by N, so no M gives the dense head's bits. `out`: the logprobs' buffer."""
        return self.gs.sparse_logits(hidden, out)

    def head_counters(self) -> HeadCounters:
        return self.gs.counters()

    def _mtp_meta(self, n: int, req: torch.Tensor) -> MtpMeta:
        tb = self.tables
        return MtpMeta(n=n, width=self.L, req=req, req_i32=req.to(torch.int32), fold_n=tb.fold_n,
                       rec_conv=tb.rec_conv, rec_k=tb.rec_k, rec_v=tb.rec_v, rec_ab=tb.rec_ab,
                       fold_n64=tb.fold_n.long())

    def _eager(self, meta: StepMeta, lay: _Layout, step_id: int, part: str = "all") -> ForwardContext:
        ctx = ForwardContext(meta, self.attn, self.gdn, self.rotary, self.pools)
        if self.k:
            self._cctx.meta = meta.chain
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
        with self.fp8.activations(t, lay.n_precise):
            self._body(ctx, t, st, rp, cr, "eager", lay, hook=hook, part=part)
        return ctx

    def _capture(self, name: str, size: int, meta: StepMeta, lay: _Layout) -> None:
        """Capture graph `name` (`graphs.parse_graph_name`) for `size` rows (decode) or tokens (break)."""
        kind, part, heads = parse_graph_name(name)
        t, st, rp, cr = self._sizes(kind, size, None)
        if kind == "decode":
            meta = self._decode_meta(size, meta)
            ctx = ForwardContext(meta, self.attn, self.gdn, self.rotary, self.pools)
            if self.k:
                self._cctx.meta = meta.chain
        else:
            ctx = self._bctx
            ctx.meta = meta
            if self.k:
                self._cctx.meta = meta.chain
        torch.cuda.current_stream().synchronize()
        cap = BreakableCapture(self._pool)
        ctx.capture = cap if kind == "break" else None
        if self.k:
            self._cctx.capture = ctx.capture
        try:
            with cap:
                self._body(ctx, t, st, rp, cr, kind, part=part, heads=heads)
        finally:
            ctx.capture = None
            if self.k:
                self._cctx.capture = None
        self._graphs[(name, size)] = cap

    def _decode_meta(self, n: int, meta: StepMeta) -> StepMeta:
        """A decode graph's row metadata: every row 1 + k tokens; the device fields are filled inside the body."""
        e = np.zeros(0, np.int64)
        out = StepMeta(num_tokens=n * self.L, positions=None, kv_page=None, kv_off=None, n_rows=n,
                       n_dec=0 if self.k else n, row_state=None, row_cu=None, pf_cu=None, pf_cu_host=e,
                       row_start_host=e, row_nq_host=e, row_kv_len_host=e, row_slot_host=e,
                       page_table=self.tables.page_table, attn_plan=meta.attn_plan)
        out.chain = meta.chain
        return out

    # -- after the body: checkpoints of live states -----------------------------------------------------------

    def _copy_states(self, src_slots: torch.Tensor, dst: torch.Tensor) -> None:
        """Copy whole state slots (conv and recurrent state of every GDN layer, and the drafter's hidden vector): the
        live slots of `src_slots` (request slots, looked up on the device) into checkpoint slots `dst`."""
        src = self.tables.state_slot.index_select(0, src_slots)
        p = self.pools
        p.conv.index_copy_(0, dst, p.conv.index_select(0, src))
        p.rec.index_copy_(0, dst, p.rec.index_select(0, src))
        if p.hid is not None:
            p.hid.index_copy_(0, dst, p.hid.index_select(0, src))

    def _fold(self, req: torch.Tensor, dst: torch.Tensor | None, n: torch.Tensor) -> None:
        """Apply the first n window records of request slots `req` to their live states (in place, `dst` None) or to
        a copy in state slots `dst`: recurrent and conv state of every GDN layer, and the hidden vector (n > 0)."""
        tb, p = self.tables, self.pools
        src = tb.state_slot.index_select(0, req)
        to = src if dst is None else dst
        ops.gdn_fold(p.rec, tb.rec_k, tb.rec_v, tb.rec_ab, a_log=self._a_log, dt_bias=self._dt_bias, req=req, src=src,
                     dst=to, n=n)
        ops.conv_fold(p.conv, tb.rec_conv, req, src, to, n)
        rh = tb.rec_hid.index_select(0, req).gather(1, (n - 1).clamp(min=0)[:, None, None].expand(
            -1, 1, tb.rec_hid.shape[2]))[:, 0]
        p.hid.index_copy_(0, to, torch.where((n > 0)[:, None], rh, p.hid.index_select(0, src)))

    def _after_body(self, lay: _Layout, st: _Staging) -> None:
        """Checkpoints after the body: end-of-chunk and end-of-step copies of live states; under MTP also decode rows'
        end-of-step copies (folded with the step's commit) and the hidden vectors of mid-chunk checkpoints."""
        if lay.eoc is not None:  # end-of-chunk checkpoints: the rows' live states after the forward
            self._copy_states(st.i(lay.eoc[0], lay.eoc[2]), st.i(lay.eoc[1], lay.eoc[2]))
        if lay.eos is not None:  # end-of-step checkpoints (MTP: explicit rows' only): the live states after sampling
            self._copy_states(st.i(lay.eos[0], lay.eos[2]), st.i(lay.eos[1], lay.eos[2]))
        if lay.eos_mtp is not None:  # of decode rows: a folded copy at the step's committed position
            req, dst = st.i(lay.eos_mtp[0], lay.eos_mtp[2]), st.i(lay.eos_mtp[1], lay.eos_mtp[2])
            self._fold(req, dst, self.tables.fold_n.index_select(0, req).long())
        if self.k and lay.ck_mid is not None:  # mid-chunk checkpoints hold the hidden vector at their position − 1
            o_src_tok, o_dst, n = lay.ck_mid[5]
            self.pools.hid.index_copy_(0, st.i(o_dst, n), self.static.hid_out.index_select(0, st.i(o_src_tok, n)))

    def _mtp_resume_kv(self, lay: _Layout, st: _Staging) -> None:
        """The drafter's KV at pos0 − 1 of the first row of a resumed admission (a cache hit or a readmission), from the
        hidden vector its live state slot received with the checkpoint and the row's first token: the KV there depends
        on the token at pos0, which may differ from the one it was computed with (architecture.md, "Checkpoints").
        Eager; the attention-input side of the MTP layer only."""
        o_slot, o_tok, o_pos, n = lay.resume[:4]
        slots, toks, pos = st.i(o_slot, n), st.i(o_tok, n), st.i(o_pos, n)
        tb = self.tables
        meta = StepMeta(num_tokens=n, positions=pos, kv_page=tb.page_table[slots, pos // self.ps].long(),
                        kv_off=pos % self.ps, n_rows=n, n_dec=0, row_state=None, row_cu=None, pf_cu=None,
                        pf_cu_host=np.zeros(0, np.int64), row_start_host=np.zeros(0, np.int64),
                        row_nq_host=np.zeros(0, np.int64), row_kv_len_host=np.zeros(0, np.int64),
                        row_slot_host=np.zeros(0, np.int64))
        emb = self.model.embed_tokens(toks)
        if self.mm:  # the first token may be an image token; pos0 − 1 may be one too (its rope deltas)
            o_emb, o_prev = lay.resume[4:6]
            emb = self.img.embeds(st.i(o_emb, n), emb)
            if self.mrope:
                meta.rope = pos[None] + st.i(o_prev, 3 * n).view(3, n)
        ctx = ForwardContext(meta, self.attn, self.gdn, self.rotary, self.pools)
        hidden = self.pools.hid.index_select(0, tb.state_slot.index_select(0, slots))
        self.model.mtp.store_kv(emb, hidden, ctx)

    # -- host-side staging -------------------------------------------------------------------------------------

    def _stage_effects(self, plan: StepPlan, st: _Staging) -> _Effects:
        """FOLD, FREE, ADMIT, PAGES, KV_COPY and STATE_COPY: validate against the mirrors, update them, stage the
        device writes."""
        m, s_max = self.mirror, self.pc.max_num_seqs
        fx = _Effects()
        fx.fold = None
        free, adm = plan.free_slot.astype(np.int64), plan.admit_slot.astype(np.int64)
        fx.n_free, fx.n_admit, fx.n_pages, fx.n_zero = len(free), len(adm), len(plan.page_id), 0
        # a slot at most once per list (a FREE and an ADMIT of one slot is its reuse): `m.epoch[adm] += 1` counts a
        # repeated index once, and `index_copy_` keeps one of its entries, in no set order
        for what, a in (("FREE", free), ("ADMIT", adm)):
            if len(a) > 1 and len(set(a.tolist())) != len(a):
                raise PlanError(f"step {plan.step_id}: a slot appears twice in {what}: {a.tolist()}")
        if len(plan.state_copy_dst) > 1 and len(set(plan.state_copy_dst.tolist())) != len(plan.state_copy_dst):
            raise PlanError(f"step {plan.step_id}: two STATE_COPYs into one slot: {plan.state_copy_dst.tolist()}")
        if len(plan.fold_slot):
            if not self.k:
                raise PlanError(f"step {plan.step_id}: FOLD without MTP")
            fs, fn = plan.fold_slot.astype(np.int64), plan.fold_n.astype(np.int64)
            if (fs < 1).any() or (fs > s_max).any() or not m.admitted[fs].all() or not np.isin(fs, free).all() or \
                    (fn < FOLD_ALL).any() or (fn > self.L).any():
                raise PlanError(f"step {plan.step_id}: FOLD {fs.tolist()} {fn.tolist()}: slots must be admitted and "
                                "FREEd in the same plan")
            fx.fold = (st.put(fs), st.put(fn), len(fs))
        if fx.n_free:
            if (free < 1).any() or (free > s_max).any() or not m.admitted[free].all():
                raise PlanError(f"step {plan.step_id}: FREE of a slot that is not admitted: {free.tolist()}")
            m.admitted[free] = m.sampled[free] = m.lp[free] = False
            m.seq_len[free] = m.slack[free] = 0
            m.resumed[free] = False
            fx.o_free = st.put(free)
        n_states = self.pc.num_state_slots
        if fx.n_admit:
            ss = plan.admit_state_slot.astype(np.int64)
            seq = plan.admit_seq_len.astype(np.int64)
            if (adm < 1).any() or (adm > s_max).any() or m.admitted[adm].any():
                raise PlanError(f"step {plan.step_id}: ADMIT into a slot in use: {adm.tolist()}")
            if (ss < 1).any() or (ss >= n_states).any():
                raise PlanError(f"step {plan.step_id}: ADMIT with an invalid state slot: {ss.tolist()}")
            if (seq < 0).any() or (seq >= self.pc.max_model_len).any():
                raise PlanError(f"step {plan.step_id}: ADMIT with seq_len {seq.tolist()}")
            m.admitted[adm], m.sampled[adm], m.seq_len[adm], m.state_slot[adm] = True, False, seq, ss
            m.slack[adm] = 0
            m.epoch[adm] += 1
            m.resumed[adm] = self.k > 0 and seq > 0
            m.temperature[adm], m.top_p[adm] = plan.admit_temperature, plan.admit_top_p
            m.top_k[adm], m.min_p[adm] = plan.admit_top_k, plan.admit_min_p
            m.lp[adm] = (plan.admit_flags & ADMIT_LOGPROBS) != 0
            if self.lp is None and m.lp[adm].any():
                raise PlanError(f"step {plan.step_id}: ADMIT_LOGPROBS, but this runner was built without logprobs")
            fx.o_adm, fx.o_ss, fx.o_admseq = st.put(adm), st.put(ss), st.put(seq)
            fx.o_topk = st.put(plan.admit_top_k)
            fx.o_seed = st.put(np.ascontiguousarray(plan.admit_seed).view(np.int64))
            fx.o_f = st.putf(np.concatenate([plan.admit_temperature, plan.admit_top_p, plan.admit_min_p]))
            if self.k:
                fx.o_maxlen = st.put(plan.admit_max_len.astype(np.int64))
                fx.o_stop = st.put(plan.admit_stop.astype(np.int64).reshape(-1))
            if self.mrope:
                fx.o_ropeoff = st.put(plan.admit_rope_off.astype(np.int64))
        if fx.n_pages:
            pslot, pidx, pid = plan.page_slot, plan.page_index, plan.page_id
            if (pslot < 1).any() or (pslot > s_max).any() or not m.admitted[pslot].all() or (pidx < 0).any() or \
                    (pidx >= self.max_pages).any() or (pid < 1).any() or (pid >= self.pc.num_pages).any():
                raise PlanError(f"step {plan.step_id}: PAGES out of range or for a slot that is not admitted")
            fx.o_pslot, fx.o_pidx, fx.o_pid = st.put(pslot), st.put(pidx), st.put(pid)
        fx.n_kvc, fx.n_sc = len(plan.kv_copy_src), len(plan.state_copy_src)
        if fx.n_kvc:
            kv = np.concatenate([plan.kv_copy_src, plan.kv_copy_dst]).astype(np.int64)
            if (kv < 1).any() or (kv >= self.pc.num_pages).any():
                raise PlanError(f"step {plan.step_id}: KV_COPY page out of range")
            fx.o_kvc = st.put(kv)
        if fx.n_sc:
            sc = np.concatenate([plan.state_copy_src, plan.state_copy_dst]).astype(np.int64)
            dst = sc[fx.n_sc:]
            if (sc < 1).any() or (sc >= n_states).any():
                raise PlanError(f"step {plan.step_id}: STATE_COPY slot out of range")
            now = m.state_slot[adm] if fx.n_admit else np.zeros(0, np.int64)
            if (np.isin(dst, m.state_slot[m.admitted]) & ~np.isin(dst, now)).any():
                raise PlanError(f"step {plan.step_id}: STATE_COPY into the live state of a running request")
            fx.o_sc = st.put(sc)
        return fx

    def _apply_effects(self, fx: _Effects, st: _Staging) -> None:
        t = self.tables
        if fx.fold is not None:  # FOLD: before FREE clears the slots' tables
            o_s, o_n, n = fx.fold
            req, cnt = st.i(o_s, n), st.i(o_n, n)
            self._fold(req, None, torch.where(cnt >= 0, cnt, t.fold_n.index_select(0, req).long()))
            t.fold_n.index_fill_(0, req, 0)
        if fx.n_free:
            fs = st.i(fx.o_free, fx.n_free)
            t.page_table.index_fill_(0, fs, 0)
            if self.k:
                t.fold_n.index_fill_(0, fs, 0)
        if fx.n_admit:
            n = fx.n_admit
            sl = st.i(fx.o_adm, n)
            t.state_slot.index_copy_(0, sl, st.i(fx.o_ss, n))
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
                t.fold_n.index_fill_(0, sl, 0)
                t.drafts.index_fill_(0, sl, 0)
            if self.mrope:
                t.rope_off.index_copy_(0, sl, st.i(fx.o_ropeoff, n))
        if fx.n_pages:
            n = fx.n_pages
            t.page_table[st.i(fx.o_pslot, n), st.i(fx.o_pidx, n)] = st.i(fx.o_pid, n).to(torch.int32)
        p = self.pools
        if fx.n_kvc:  # copy-on-write of a shared page, every attention layer (gather, then scatter)
            n = fx.n_kvc
            src, dst = st.i(fx.o_kvc, n), st.i(fx.o_kvc + n, n)
            p.kv.index_copy_(1, dst, p.kv.index_select(1, src))
        if fx.n_sc:  # cached checkpoints into live slots: conv and recurrent state of every GDN layer, hidden vector
            n = fx.n_sc
            src, dst = st.i(fx.o_sc, n), st.i(fx.o_sc + n, n)
            p.conv.index_copy_(0, dst, p.conv.index_select(0, src))
            p.rec.index_copy_(0, dst, p.rec.index_select(0, src))
            if p.hid is not None:
                p.hid.index_copy_(0, dst, p.hid.index_select(0, src))
        if fx.n_zero:  # STATE_ZERO rows start from zero conv and recurrent state: zero their live slots
            z = st.i(fx.o_zero, fx.n_zero)
            p.conv.index_fill_(0, z, 0)
            p.rec.index_fill_(0, z, 0)

    def _layout(self, plan: StepPlan, st: _Staging) -> _Layout:
        """SCHED → this step's row order and kernel metadata, all on the host; advances the mirror."""
        m, lay, k, L = self.mirror, _Layout(), self.k, self.L
        slots = plan.sched_slot.astype(np.int64)
        n = lay.n = len(slots)
        lay.n_sample = lay.n_stok = lay.n_mtp = lay.n_precise = 0
        lay.eoc = lay.eos = lay.eos_mtp = lay.ck_mid = lay.splan = lay.dplan = lay.zero_states = lay.resume = None
        lay.emb = lay.demb = lay.rd = lay.s_mask = lay.splan_sp = lay.dplan_uni = None
        lay.n_masks = 0
        lay.constrained = False
        lay.chain = lay.lp = None
        lay.s_plan_rows = np.zeros(0, np.int64)
        if n == 0:
            if len(plan.ckpt_sched):
                raise PlanError(f"step {plan.step_id}: CKPT without SCHED rows")
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
        # Row order: device rows, then (without MTP) explicit 1-token rows (both decode-shaped), then multi-token rows,
        # then PRECISE rows (their tokens last: `kernels.fp8.precise_rows`). Under batch invariance every prompt token
        # takes the GDN prefill kernel, a 1-token chunk's too: only device rows go to the decode kernel.
        key = np.where(dev, 0, 2) if k or self.inv else np.where(dev, 0, np.where(nq == 1, 1, 2))
        key = np.where(precise, 3, key)
        order = np.argsort(key, kind="stable") if n > 1 and (key[1:] < key[:-1]).any() else np.arange(n)
        r_slot, r_nq, r_pos0, r_kv = slots[order], nq[order], seq[order], kv_len[order]
        r_zero, r_sample, r_dev = zero[order], sample_rows[order], dev[order]
        n_dev = int(dev.sum())
        n_dec = 0 if k else n_dev if self.inv else int(((nq == 1) & ~precise).sum())
        starts = np.cumsum(r_nq) - r_nq
        tok_row = np.repeat(np.arange(n), r_nq)
        # Per-token explicit tokens and positions (device rows: −1 − window index, read on the device).
        widx = np.arange(t_total) - np.repeat(starts, r_nq)
        lay.tok_pos = np.repeat(r_pos0, r_nq) + widx
        tok_in = -1 - widx
        n_dtok = int(r_nq[:n_dev].sum())
        ex_idx = None
        if t_total > n_dtok:
            ex_rows, ex_nq = order[n_dev:], r_nq[n_dev:]
            ex_starts = starts[n_dev:] - n_dtok
            j = np.arange(t_total - n_dtok)
            ex_idx = np.repeat(tok_off[ex_rows] - ex_starts, ex_nq) + j  # each explicit token's index in the plan
            ex_tok = plan.tokens[ex_idx]
            if int(ex_tok.min()) < 0 or int(ex_tok.max()) >= self.cfg.vocab_size:
                raise PlanError(f"step {step}: token out of vocabulary")
            tok_in[n_dtok:] = ex_tok
        # Images: each token's encoder row (−1: its token's embedding) and M-RoPE deltas (device rows: the ADMIT
        # offset, read on the device)
        lay.emb = lay.rd = None
        if self.mm:
            lay.emb = np.full(t_total, -1, np.int64)
            if ex_idx is not None and len(plan.tok_emb):
                lay.emb[n_dtok:] = plan.tok_emb[ex_idx]
                self.img.check_rows(lay.emb, step)
        elif len(plan.tok_emb) or len(plan.enc_key):
            raise PlanError(f"step {step}: image inputs, but this runner has no vision tower")
        if self.mrope:
            lay.rd = np.zeros((3, t_total), np.int64)
            if ex_idx is not None and len(plan.tok_rope[0]):
                lay.rd[:, n_dtok:] = plan.tok_rope[:, ex_idx]
        lay.tok_plan_row = order[tok_row]
        lay.o_rslot = st.put(r_slot)
        lay.o_rcu = st.put(np.concatenate([[0], np.cumsum(r_nq)]))
        n_first = n_dev if k else n_dec  # the first prefill row
        lay.pf_cu = np.concatenate([[0], np.cumsum(r_nq[n_first:])]).astype(np.int64)
        if n_first < n:
            lay.o_pfcu = st.put(lay.pf_cu)
        lay.zero_states = m.state_slot[r_slot[r_zero]] if r_zero.any() else None  # zeroed before the forward
        # Row kinds and sample tokens: a verify row samples its 1 + k tokens, a SAMPLE row its last.
        r_kind = np.where(r_dev & (k > 0), _VERIFY, np.where(r_sample, _SAMPLE, _NO_SAMPLE)).astype(np.int64)
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
        if k:
            # The drafter's input token at each position: the next explicit token; at a chunk's last, its next token
            # (−1 on a final chunk); −1 on device rows: the target's sample there.
            dtok = np.full(t_total, -1, np.int64)
            lay.demb = np.full(t_total, -1, np.int64) if self.mm else None
            if t_total > n_dtok:
                dtok[n_dtok:-1] = tok_in[n_dtok + 1:]
                ends = (starts + r_nq - 1)[n_dev:]
                dtok[ends] = plan.sched_next_tok[order[n_dev:]]
                if self.mm:  # the drafter's input at an image position is that token's embedding row
                    lay.demb[n_dtok:-1] = lay.emb[n_dtok + 1:]
                    lay.demb[ends] = plan.sched_next_emb[order[n_dev:]] if len(plan.sched_next_emb) else -1
            lay.dtok = dtok
            lay.n_mtp = n_dev
            if self.check and n_dev < n and (dtok[(starts + r_nq - 1)[n_dev:]] >= 0).any() and \
                    r_sample[n_dev:][dtok[(starts + r_nq - 1)[n_dev:]] >= 0].any():
                raise PlanError(f"step {step}: a final chunk with a next token")
            # A resumed admission's first row recomputes the drafter's KV at pos0 − 1 (eager, before the body).
            res = m.resumed[r_slot] & ~r_dev
            if res.any():
                idx = np.flatnonzero(res)
                lay.resume = (st.put(r_slot[idx]), st.put(tok_in[starts[idx]]), st.put(r_pos0[idx] - 1), len(idx))
                if self.mm:  # the first token's row, and the rope deltas at pos0 − 1
                    prev = plan.sched_rope_prev[order[idx]].T if len(plan.sched_rope_prev) else \
                        np.zeros((3, len(idx)), np.int64)
                    lay.resume += (st.put(lay.emb[starts[idx]]), st.put(np.ascontiguousarray(prev).reshape(-1)))
            m.resumed[r_slot] = False
        if not self.cuda:  # the torch sampler's plans, per logits width (the grammar heads': their rows)
            lay.splan = host_plan(m, lay.s_slot, self.cfg.vocab_size)
            if self.gs.sparse and lay.constrained:
                lay.splan_sp = host_plan(m, lay.s_slot, self.gs.rows)
            if k:
                cs = r_slot[c_row]
                lay.dplan = host_plan(m, cs, self.draft_head.size)
                if self.gs.universe and lay.constrained:
                    lay.dplan_uni = host_plan(m, cs, self.gs.rows)
        # Attention: planned for the upper bound of each KV length; under MTP the lower bound decides whether the
        # plan's chunking is safe (kernels.flashinfer).
        lay.r_kv_lb = r_pos0 + r_nq
        if self.inv and self.attn_kind == "flashinfer":  # our kernel's rows: (slot, first token, n_query, KV length)
            lay.full_rows = np.stack([r_slot, starts, r_nq, r_kv], axis=1).astype(np.int32)
        elif self.attn_kind == "flashinfer":
            ps = self.ps
            n_pages = (r_kv + ps - 1) // ps
            kv_indptr = np.concatenate([[0], np.cumsum(n_pages)])
            lay.o_npages, lay.o_kvindptr = st.put(n_pages), st.put(kv_indptr)
            lay.qo_indptr32 = st.put32(np.concatenate([[0], np.cumsum(r_nq)]))
            lay.kv_indptr32 = st.put32(kv_indptr)
            lay.last32 = st.put32(r_kv - (n_pages - 1) * ps)
            lay.kv_lens32 = st.put32(r_kv)
            lay.max_pages, lay.total_pages, lay.max_q = int(n_pages.max()), int(kv_indptr[-1]), int(r_nq.max())
        if k and k > 1 and lay.n_sample:
            # The drafter's chain: one query per sample row; at chain step j the KV length is p0 + j + 1, p0 the
            # row's last committed position (seq_len − 1 after the step): in [lo, lo + slack + k] for verify rows.
            # Planned for the last step's upper bound; the lower bound is the first step's.
            cr = c_row
            ver = r_kind[cr] == _VERIFY
            p0_lb = np.where(ver, r_pos0[cr], r_pos0[cr] + r_nq[cr] - 1)
            p0_ub = np.where(ver, r_pos0[cr] + slack[order][cr] + k, p0_lb)
            ub = p0_ub + k
            c_pages = (ub + self.ps - 1) // self.ps
            c_indptr = np.concatenate([[0], np.cumsum(c_pages)])
            lay.chain = (p0_lb + 2, ub, st.put(r_slot[cr]), st.put(c_pages), st.put(c_indptr), int(c_pages.max()),
                         int(c_indptr[-1]))
        # Checkpoints, then the mirror: explicit rows advance by n_query; device rows by at least 1 (+ k of slack).
        lay.order, lay.T, lay.n_dev, lay.n_dec = order, t_total, n_dev, n_dec
        lay.r_slot, lay.r_nq, lay.r_kv, lay.starts = r_slot, r_nq, r_kv, starts
        lay.tok_in = tok_in
        if len(plan.ckpt_sched):
            self._layout_ckpts(plan, st, lay, slots, nq, order, r_nq, n_first, dev)
        m.seq_len[slots] = np.where(dev, seq + 1, seq + nq)
        if k:
            m.slack[slots[dev]] += k
        m.sampled[slots[sample_rows]] = True
        return lay

    def _stage_static(self, lay: _Layout, st: _Staging, size: int, kind: str) -> None:
        """The static input block for `size` (`_sizes`): the token, sample-token, row and sample-row arrays, padded
        (padding: slot 0, token 0; position 0 outside decode graphs)."""
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
            if kind == "decode":  # padding rows: token 0 at positions 0 .. L − 1 of the null page (kv length L)
                a[o["pos"] + t:o["pos"] + tp] = np.tile(np.arange(self.L), (tp - t) // self.L)
        if self.k:
            a[o["dtok"]:o["dtok"] + t] = lay.dtok
            if tp > t:
                a[o["dtok"] + t:o["dtok"] + tp] = -1 if kind == "decode" else 0
        if self.mm:  # padding tokens: their token's embedding
            a[o["emb"]:o["emb"] + t] = lay.emb
            a[o["emb"] + t:o["emb"] + tp] = -1
            if self.k:
                a[o["demb"]:o["demb"] + t] = lay.demb
                a[o["demb"] + t:o["demb"] + tp] = -1
        if self.mrope:
            for c in range(3):
                f = o[f"rd{c}"]
                a[f:f + t] = lay.rd[c]
                a[f + t:f + tp] = 0
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
            if kind == "decode":  # padding rows: 1 + k tokens and samples of their own on slot 0, never verified
                pr = np.arange(n, rp)
                a[o["r_nq"] + n:o["r_nq"] + rp] = self.L
                a[o["r_kind"] + n:o["r_kind"] + rp] = _VERIFY if self.k else _SAMPLE
                a[o["r_start"] + n:o["r_start"] + rp] = pr * self.L
                a[o["r_s0"] + n:o["r_s0"] + rp] = pr * (self.L if self.k else 1)
            else:
                for f in ("r_nq", "r_kind", "r_start", "r_s0"):
                    a[o[f] + n:o[f] + rp] = 0
        c = lay.n_sample
        a[o["c_row"]:o["c_row"] + c] = lay.c_row
        a[o["c_slot"]:o["c_slot"] + c] = lay.r_slot[lay.c_row]
        if cp > c:  # padding sample rows: their own rows in a decode graph (all padding rows sample), else row 0
            a[o["c_row"] + c:o["c_row"] + cp] = np.arange(c, cp) if kind == "decode" else 0
            a[o["c_slot"] + c:o["c_slot"] + cp] = 0

    def _row_meta(self, lay: _Layout, st: _Staging, parity: int, kind: str, size: int) -> StepMeta:
        """The step's row metadata (staged device views), the attention plans (the target's, and the drafter's
        chain under MTP) and their device length fix-ups."""
        n = lay.n
        r_slot = st.i(lay.o_rslot, n)
        tb = self.tables
        row_state = tb.state_slot.index_select(0, r_slot) if kind != "decode" else None
        meta = StepMeta(
            num_tokens=lay.T, positions=None, kv_page=None, kv_off=None, n_rows=n, n_dec=lay.n_dec,
            row_state=row_state, row_cu=st.i(lay.o_rcu, n + 1),
            pf_cu=st.i(lay.o_pfcu, len(lay.pf_cu)) if len(lay.pf_cu) > 1 else None, pf_cu_host=lay.pf_cu,
            row_start_host=lay.starts, row_nq_host=lay.r_nq, row_kv_len_host=lay.r_kv, row_slot_host=lay.r_slot,
            page_table=tb.page_table)
        meta.chain = None
        if row_state is not None:
            meta.row_state_i32 = row_state.to(torch.int32)
        if lay.n_dec and kind != "decode" and self.cuda:
            meta.decode_i32 = (row_state[:lay.n_dec].to(torch.int32).view(lay.n_dec, 1),
                               self.static.arange32[:lay.n_dec + 1], self.static.ones32[:lay.n_dec])
        if lay.n_mtp and kind != "decode":
            meta.mtp = self._mtp_meta(lay.n_mtp, r_slot[:lay.n_mtp])
        if self.k and kind != "decode":  # the true KV lengths (eager torch attention and the fix-ups below)
            s = self.static
            s.kv_len[:n] = tb.seq_len.index_select(0, r_slot) + st.i(lay.o_rcu + 1, n) - st.i(lay.o_rcu, n)
            meta.kv_len_dev = s.kv_len[:n]
        if lay.ck_mid is not None:
            cu, o_src, o_dst, k, conv = lay.ck_mid[:5]
            meta.ckpt = CkptMeta(every=CKPT_GRID, cu_starts_host=cu, workspace=self._ckpt_ws, src=st.i(o_src, k),
                                 dst=st.i(o_dst, k), conv=(st.i(conv[0], n + 1), st.i(conv[1], k), st.i(conv[2], k)),
                                 segments=lay.ck_mid[6])
        uncertain = bool((lay.r_kv_lb < lay.r_kv).any())
        if self.inv and self.attn_kind == "flashinfer":  # our kernel (batch invariance); the chain's FlashInfer below
            if kind == "decode":
                self._plan_decode_graph(meta, lay, st, parity, size)
            else:
                self.attn.plan(meta, parity=parity, rows=lay.full_rows, stage=st.put32, fixed_rows=lay.n_dev)
                if self.k:  # the true KV lengths, decided on the device (the decode graphs' body does its own)
                    self.attn.fix_lengths(meta.attn_plan, n, meta.kv_len_dev)
        elif self.attn_kind == "flashinfer":
            if kind == "decode":
                self._plan_decode_graph(meta, lay, st, parity, size)
            else:
                from ...kernels.flashinfer import page_indices

                kv_indices = page_indices(tb.page_table, r_slot, st.i(lay.o_npages, n),
                                          st.i(lay.o_kvindptr, n + 1), lay.max_pages, lay.total_pages,
                                          self._arange_pages)
                w = self.attn.plan(meta, parity=parity, qo_indptr=lay.qo_indptr32, kv_indptr=lay.kv_indptr32,
                                   kv_indices=kv_indices, last_page_len=lay.last32, kv_lens=lay.kv_lens32,
                                   max_q=lay.max_q, lb=lay.r_kv_lb if uncertain else None)
                if uncertain:
                    self.attn.fix_lengths(w, n, meta.kv_len_dev)
        else:
            self.attn.plan(meta)
        if self.k:
            meta.chain = self._chain_meta(lay, st, parity, kind, size)
        return meta

    def _chain_meta(self, lay: _Layout, st: _Staging, parity: int, kind: str, size: int) -> StepMeta:
        """The drafter's chain: one query per sample row (`cr` rows: the bucket's padding included in a decode graph),
        planned for the upper bound of its last step's KV length; every chain step fixes the lengths on the device."""
        cr = self._sizes(kind, size, lay)[3]
        e = np.zeros(0, np.int64)
        m = lay.n_sample
        cmeta = StepMeta(num_tokens=cr if kind == "decode" else m, positions=None, kv_page=None, kv_off=None,
                         n_rows=cr if kind == "decode" else m, n_dec=0, row_state=None, row_cu=None, pf_cu=None,
                         pf_cu_host=e, row_start_host=np.arange(m), row_nq_host=np.ones(m, np.int64),
                         row_kv_len_host=e, row_slot_host=lay.r_slot[lay.c_row] if m else e,
                         page_table=self.tables.page_table)
        rows = cr if kind == "decode" else m
        # The chain steps' true KV lengths: a static buffer each chain step rewrites in the body (a breakable graph's
        # attention break reads them through this meta at replay)
        cmeta.kv_len_dev = self.static.d_len[:rows]
        cmeta.fix_len = kind != "decode"
        if lay.chain is None or self.chain_attn is None or self.attn_kind != "flashinfer":
            return cmeta
        lb, ub, o_slot, o_pages, o_indptr, max_pages, total = lay.chain
        ps = self.ps
        kv = np.ones(rows, np.int64)
        kv[:m] = ub
        lo = kv.copy()
        lo[:m] = lb
        n_pages = (kv + ps - 1) // ps
        indptr = np.concatenate([[0], np.cumsum(n_pages)])
        args = dict(parity=parity, qo_indptr=st.put32(np.arange(rows + 1)), kv_indptr=st.put32(indptr),
                    last_page_len=st.put32(kv - (n_pages - 1) * ps), kv_lens=st.put32(kv), max_q=1, lb=lo)
        if kind == "decode":
            self.chain_attn.plan_graph(cmeta, rows=rows, **args)
        else:
            from ...kernels.flashinfer import page_indices

            kv_indices = page_indices(self.tables.page_table, st.i(o_slot, m), st.i(o_pages, m),
                                      st.i(o_indptr, m + 1), max_pages, total, self._arange_pages)
            self.chain_attn.plan(cmeta, kv_indices=kv_indices, **args)
        return cmeta

    def _plan_decode_graph(self, meta: StepMeta, lay: _Layout, st: _Staging, parity: int, rows: int) -> None:
        """Plan a decode bucket: real rows in the step's row order, then padding rows of 1 + k tokens on the null
        page."""
        ps, n, L = self.ps, lay.n, self.L
        if self.inv:  # our kernel: the row table, padding rows on slot 0's null page with 1 + k keys
            full = np.zeros((rows, 4), np.int32)
            full[:n] = lay.full_rows
            pr = np.arange(n, rows)
            full[n:, 1], full[n:, 2], full[n:, 3] = pr * L, L, L
            self.attn.plan_graph(meta, bucket=rows, rows=full, stage=st.put32, real=n)
            return
        kv = np.full(rows, L, np.int64)
        kv[:n] = lay.r_kv
        lb = kv.copy()
        lb[:n] = lay.r_kv_lb
        n_pages = (kv + ps - 1) // ps
        indptr = np.concatenate([[0], np.cumsum(n_pages)])
        self.attn.plan_graph(meta, rows=rows, parity=parity, qo_indptr=st.put32(np.arange(rows + 1) * L),
                             kv_indptr=st.put32(indptr), last_page_len=st.put32(kv - (n_pages - 1) * ps),
                             kv_lens=st.put32(kv), max_q=L, lb=lb if (lb < kv).any() else None)

    def _layout_ckpts(self, plan: StepPlan, st: _Staging, lay: _Layout, slots: np.ndarray, nq: np.ndarray,
                      order: np.ndarray, r_nq: np.ndarray, n_first: int, dev: np.ndarray) -> None:
        """CKPT records: validate them and stage the kinds (mid-chunk, end of chunk, end of step; under MTP the end of
        step of a decode row folds a copy)."""
        m, step, n = self.mirror, plan.step_id, len(slots)
        rows = plan.ckpt_sched.astype(np.int64)
        offs = plan.ckpt_offset.astype(np.int64)
        dst = plan.ckpt_dst.astype(np.int64)
        if (rows < 0).any() or (rows >= n).any():
            raise PlanError(f"step {step}: CKPT for a SCHED row out of range: {rows.tolist()}")
        row_nq = nq[rows]
        mid = (offs > 0) & (offs < row_nq)
        end = offs == row_nq
        eos = offs == CKPT_END_OF_STEP
        if not (mid | end | eos).all() or (offs[mid] % CKPT_GRID).any():
            raise PlanError(f"step {step}: CKPT offsets {offs.tolist()} for rows of {row_nq.tolist()} tokens")
        if (dst < 1).any() or (dst >= self.pc.num_state_slots).any() or len(set(dst.tolist())) != len(dst):
            raise PlanError(f"step {step}: CKPT destinations out of range or repeated: {dst.tolist()}")
        if np.isin(dst, m.state_slot[m.admitted]).any():
            raise PlanError(f"step {step}: CKPT into the live state slot of a running request")
        mtp_row = dev[rows] if self.k else np.zeros(len(rows), bool)
        if self.k and (mtp_row & ~eos).any():
            raise PlanError(f"step {step}: a decode row checkpoints only at the end of the step under MTP")
        if end.any():
            lay.eoc = (st.put(slots[rows[end]]), st.put(dst[end]), int(end.sum()))
        if (eos & ~mtp_row).any():
            e = eos & ~mtp_row
            lay.eos = (st.put(slots[rows[e]]), st.put(dst[e]), int(e.sum()))
        if (eos & mtp_row).any():
            e = eos & mtp_row
            lay.eos_mtp = (st.put(slots[rows[e]]), st.put(dst[e]), int(e.sum()))
        if mid.any():
            inv = np.empty(n, np.int64)
            inv[order] = np.arange(n)
            r = inv[rows[mid]]  # rows in the runner's order: mid-chunk ones are multi-token, so in the prefill group
            counts = r_nq[n_first:] // CKPT_GRID
            cu = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
            src = cu[r - n_first] + offs[mid] // CKPT_GRID - 1
            k = int(mid.sum())
            # The conv kernel walks each row's tokens: its checkpoints by row (runner order), then by offset.
            by = np.lexsort((offs[mid], r))
            conv_ptr = np.searchsorted(r[by], np.arange(n + 1))
            conv = (st.put(conv_ptr), st.put(offs[mid][by]), st.put(dst[mid][by]))
            hid = None
            if self.k:  # the hidden vector at the checkpoint's position − 1: the row's token there
                tok = lay.starts[r] + offs[mid] - 1
                hid = (st.put(tok), st.put(dst[mid]), k)
            seg = None
            if self.gdn_kind == "flashinfer":  # its rows cut there (`kernels.flashinfer._gdn_cut_rows`)
                from ...kernels.flashinfer import gdn_segments

                seg = gdn_segments(lay.pf_cu, r - n_first, offs[mid], src, lambda a: st.i(st.put(a), len(a)))
            lay.ck_mid = (cu, st.put(src), st.put(dst[mid]), k, conv, hid, seg)

    def shutdown(self) -> None:
        if self._allocated:
            self._release()
        self._ready = False
