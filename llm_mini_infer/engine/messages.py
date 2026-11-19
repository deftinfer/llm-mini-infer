"""Messages to and from the engine: the frontends' requests, aborts and acks, the launcher's `Shutdown`, and the
engine's `StepOutputs` (docs/architecture.md, "Frontend and engine messages"); and the engine's start, reported to the
launcher over a pipe (`EngineReady`, `EngineFailed`).

msgspec Structs (`array_like`, `gc=False`) behind a one-byte type frame; the wire format is `ipc`'s. Request IDs are
frontend-local ints: the engine keys requests by (frontend, rid), and every output message goes to one frontend only.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import msgspec
import numpy as np


class FinishReason(IntEnum):
    NONE = 0
    STOP = 1  # EOS or a stop token ID (engine), or a stop string (frontend)
    LENGTH = 2  # max_tokens or max_model_len
    ABORT = 3  # aborted by the engine as it stops (Shutdown, the launcher gone)
    ERROR = 4  # rejected or failed by the engine


OPENAI_FINISH = {FinishReason.STOP: "stop", FinishReason.LENGTH: "length", FinishReason.ABORT: "abort",
                 FinishReason.ERROR: "error"}


class _Msg(msgspec.Struct, array_like=True, gc=False):
    pass


class ImageRef(_Msg):
    """One image of a request: its placeholder span in the prompt, its patch grid, its content hash (as the span's radix
    key) and where its uint8 pixels are in the pixel arena (`vision.arena`)."""

    key: int  # radix-key value of every position of the span (vision.family.span_key), outside the int32 range
    offset: int  # prompt position of the span's first placeholder token
    tokens: int  # placeholder tokens (the encoder's embeddings)
    grid_t: int  # patch grid (before the spatial merge)
    grid_h: int
    grid_w: int
    height: int  # pixels: uint8 [3, height, width] at `pixels` in the arena
    width: int
    pixels: int  # arena offset
    digest: bytes = b""  # the image's full content hash (32 bytes, `key`'s source): checked on every cache hit


class AddRequest(_Msg):
    rid: int
    frontend: int
    prompt: np.ndarray  # int32 token IDs
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int  # 0 = off
    min_p: float
    seed: int | None  # None: the engine draws one, so every sample is reproducible from the debug log
    stop_token_ids: list[int]  # on top of the model's EOS IDs
    ignore_eos: bool = False
    priority: int = 0  # lower runs first
    cache_salt: str | None = None  # a separate prefix-cache namespace
    t_recv_ns: int = 0  # when the frontend received the HTTP request
    cache_hints: list[int] = []  # prompt offsets where a message marked with cache_control ends (exact checkpoints)
    images: list[ImageRef] = []  # in prompt order; their pixels stay valid until the engine releases the request
    grammar: str = ""  # structured output: the grammar text the engine compiles (grammar.spec); "" = none
    request_id: str = ""  # the request's id (X-Request-Id, or the frontend's), for the engine's log lines
    # The frontend ends it on stop strings; it takes no forced prefix and no EOS-free finish, which move tokens between
    # steps
    stop_strings: bool = False
    logprobs: int = -1  # its output tokens' logprobs with this many top entries (0..runner.base.MAX_LOGPROBS); -1: none
    # Penalties (vLLM's; neutral: 1, 0, 0) and the logit bias: distinct token ids, ascending, and their values in
    # [-100, 100] (at most `sched.plan.MAX_LOGIT_BIAS`); empty: none
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    bias_ids: list[int] = []
    bias_values: list[float] = []


class Abort(_Msg):
    frontend: int
    rids: list[int]
    reason: int = 0  # why the frontend aborts them: an ABORT_REASONS index


class Ack(_Msg):
    """From a frontend: it has processed its StepOutputs up to `seq` (backpressure, `engine.process`)."""

    frontend: int
    seq: int


class Shutdown(_Msg):
    """From the launcher: abort every request, stop the step loop once the step in flight has completed, and exit."""

    reason: str = ""


# A checkpoint's source (`RadixCache.attach`), in EngineStats.ckpts_taken, ckpt_resumes and ckpts_evicted order: a
# prefill's end, a branch point, a client's hint, a decode interval, a request leaving (finished or aborted), a refine,
# a preempted request's donation
CHECKPOINT_SOURCES = ("prefill", "branch", "hint", "decode", "finish", "refine", "preempt")
# Abort.reason, EngineStats.aborts order: the client left, a stop string ended the request in the frontend, the
# frontend failed it (the engine stopped, a drain timed out)
ABORT_REASONS = ("disconnect", "stop", "failed")
ABORT_DISCONNECT, ABORT_STOP, ABORT_FAILED = range(3)
# EngineStats.lost_tokens order: why an admission recomputed what the prefix cache held for it (`cache.radix`, Lost
# tokens; `precise`: `sched.scheduler._count_lost`)
LOST_CAUSES = ("writing", "checkpoint", "page", "precise")
# For EngineStats.spec_first_rejections and spec_window_mask_counts: a grammar mask's class, by what it allows: one
# token the grammar forces, the EOS alone (the end of a document), a choice of several, or every token (unconstrained: a
# grammar ended under ignore_eos); and the bucket bounds of the popcounts of the masks verification read
MASK_CLASSES = ("forced", "eos", "choice", "free")
POPCOUNT_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128, 256)


class EngineStats(_Msg):
    """The engine's gauges and counters, plus timing of the step these outputs come from."""

    running: int
    waiting: int
    kv_usage: float  # share of the pages that running requests reference (cached-only pages count as free)
    state_usage: float  # share of live state slots in use
    num_preemptions: int  # cumulative
    num_aborts: int  # cumulative, frontend Aborts that found a live request
    prompt_tokens: int  # cumulative prompt tokens computed (prefill), recompute included
    generation_tokens: int  # cumulative output tokens of live requests
    step_tokens: int  # tokens scheduled in this step (vllm:iteration_tokens_total)
    step_cpu_ns: int  # engine-thread time of the previous loop turn, minus blocking (poll, device) and runner calls
    step_runner_calls_ns: int  # CPU time of the runner's launch_forward + launch_sample in that turn
    step_runner_ns: int  # runner (GPU) busy time of this step
    gpu_busy_ns: int  # cumulative
    gpu_idle_ns: int  # cumulative idle time with work queued (the GPU waiting for the CPU)
    steps: int = 0  # cumulative steps completed
    phase_ns: list[int] = []  # cumulative ns per loop phase (engine.core.PHASES)
    # prefix cache (cumulative unless noted)
    checkpoint_usage: float = 0.0  # checkpoint slots in use, as a share (gauge); window models: pages only windows hold
    cached_pages: int = 0  # KV pages the cache holds (gauge)
    prefix_queries: int = 0  # prompt tokens of first admissions
    prefix_hits: int = 0  # of which served from the cache
    readmit_hits: int = 0  # tokens served to readmissions after preemption
    ckpt_hits: int = 0  # admissions resumed from a checkpoint
    ckpt_misses: int = 0  # admissions without one (hybrid models)
    ckpts_taken: list[int] = []  # checkpoints created, per CHECKPOINT_SOURCES
    ckpts_skipped: int = 0  # wanted, but no checkpoint slot could be freed
    cache_waits: int = 0  # admissions deferred one step for KV or a checkpoint being written
    lost_tokens: list[int] = []  # prompt tokens the cache held for an admission but it recomputed, per LOST_CAUSES
    aborts: list[int] = []  # num_aborts per ABORT_REASONS
    # the precise prefix: admissions that recomputed a cached prefix with BF16 activations, tokens of PRECISE rows
    # scheduled, admissions resumed from a precise checkpoint
    refines: int = 0
    precise_tokens: int = 0
    precise_hits: int = 0
    # ... and steps that carried PRECISE rows
    precise_steps: int = 0
    evicted_pages: int = 0
    evicted_ckpts: int = 0
    # checkpoints by source (CHECKPOINT_SOURCES): the resumes each served, and those evicted (cumulative), two per
    # source: evicted having served none, and having served some; the window pages evicted windows freed; the windows
    # alive now (sliding-window models)
    ckpt_resumes: list[int] = []
    ckpts_evicted: list[int] = []
    evicted_window_pages: int = 0
    windows: int = 0
    num_failed: int = 0  # requests failed: NONFINITE (their own or a rolled-back write), MASK_EMPTY, a rejected token
    # MTP speculative decoding (cumulative; empty per-position list when MTP is off)
    spec_drafts: int = 0  # verify steps of requests (`vllm:spec_decode_num_drafts`)
    spec_draft_tokens: int = 0
    spec_accepted: int = 0
    spec_accepted_per_pos: list[int] = []
    # FlashInfer's attention-plan counters (`runner.base.AttentionCounters`)
    attention_replans: int = 0
    attention_plans_kept: int = 0
    attention_plans_unsplit: int = 0
    vision_plan_waits: int = 0  # the vision tower's plans that waited for their buffer (`AttentionCounters`)
    # the grammar-aware heads' counters (`runner.base.HeadCounters`; cumulative but draft_universe_tokens, a gauge),
    # with zero union counts from a runner without the heads
    grammar_head_sparse: int = 0
    grammar_head_dense_off: int = 0
    grammar_head_dense_unconstrained: int = 0
    grammar_head_dense_over_cap: int = 0
    grammar_head_union_counts: list[int] = []
    grammar_head_union_sum: int = 0
    draft_universe_tokens: int = 0
    draft_universe_steps: int = 0
    draft_universe_rebuilds: int = 0
    # images (cumulative)
    encoder_images: int = 0  # images encoded (encoder cache misses)
    encoder_tokens: int = 0  # embeddings computed
    encoder_hits: int = 0  # image spans prefilled from embeddings already cached
    encoder_ns: int = 0  # device time of the encoder
    step_encoder_ns: int = 0  # the encoder's device time in this step (0: no image encoded)
    # every completed step's decode (verify) rows and encoded images (EngineCore.rows_hist, images_hist): counts per
    # bucket since start (metrics.STEP_ROWS and STEP_IMAGES, the last one +Inf) and sums
    step_rows_counts: list[int] = []
    step_rows_sum: int = 0
    step_images_counts: list[int] = []
    step_images_sum: int = 0
    page_waits: int = 0  # cumulative: admissions and running prefill chunks that waited for KV pages
    # structured output (cumulative)
    structured_requests: int = 0  # requests with a grammar the engine accepted
    grammar_compiles: int = 0  # grammars compiled (cache misses, failed ones included)
    grammar_compile_ns: int = 0
    grammar_rejects: int = 0  # requests whose grammar the engine could not compile, or not in time
    grammar_masks: int = 0  # masks filled
    grammar_mask_ns: int = 0  # engine time filling them
    grammar_errors: int = 0  # requests failed by a token their grammar rejected (never expected)
    grammar_end_waits: int = 0  # decode rows skipped behind a row in flight that could end its grammar
    dropped_samples: int = 0  # sample rows computed for requests that had left (a stop, an abort, a preemption)
    # the forced prefix and the EOS-free finish (cumulative): requests whose final prefill chunk took their grammar's
    # forced prefix, and its tokens; EOS-free finishes [no row of the request in flight, its row in flight], and the
    # ones the gate left to the step in flight; MTP verify rows' first rejected draft by the class of its sample's mask
    # (MASK_CLASSES), and the popcounts of the masks verification read (bucket counts over POPCOUNT_BUCKETS, the last
    # one +Inf; sum)
    grammar_forced_requests: int = 0
    grammar_forced_tokens: int = 0
    grammar_eos_free: list[int] = []
    grammar_eos_free_gated: int = 0
    spec_first_rejections: list[int] = []
    spec_window_mask_counts: list[int] = []
    spec_window_mask_sum: int = 0
    # backpressure: admissions of a frontend paused while it lags behind its outputs (cumulative; gauge), and the loop
    # turns that blocked on the poll because the only work waited for paused frontends (cumulative)
    backpressure_pauses: int = 0
    paused_frontends: int = 0
    backpressure_waits: int = 0
    # split steps (cumulative): the device time of their sampling parts (`StepResult.gpu_sample_ns`), and the engine's
    # polling for the step in flight's forward to end (planning at the boundary: phase_ns's "boundary")
    gpu_sample_ns: int = 0
    boundary_ns: int = 0
    # the engine's garbage collections and their time, per generation (cumulative from its loop's start: `gcstats`)
    gc_ns: list[int] = []
    gc_collections: list[int] = []


class Logprobs(_Msg):
    """The logprobs of a StepOutputs' tokens, aligned with its `tokens`: those of a request that asked
    (`AddRequest.logprobs`), the rest unused (NaN, id 0). `top_ids`/`top`: K columns, the most any of the message's
    requests asked; a request reads its own first ones (docs/architecture.md, "Sampling")."""

    token: np.ndarray  # float32 [n_tokens]: each token's logprob
    top_ids: np.ndarray  # int32 [n_tokens, K]: the top tokens, in descending order
    top: np.ndarray  # float32 [n_tokens, K]: their logprobs


class StepOutputs(_Msg):
    """At most one per step per frontend: what the step produced for that frontend's requests (results by position)."""

    step: int
    rids: np.ndarray  # int64 [n]
    n_new: np.ndarray  # int32 [n], new tokens per request (0 when only finishing)
    tokens: np.ndarray  # int32 [sum(n_new)]
    finish: np.ndarray  # uint8 [n], FinishReason
    # Requests whose first tokens are in this message (a subset of rids), with their engine-side timestamps.
    first_rids: np.ndarray  # int64 [m]
    first_cached: np.ndarray  # int32 [m], prompt tokens served from the prefix cache
    first_t_admit_ns: np.ndarray  # int64 [m]
    first_t_launch_ns: np.ndarray  # int64 [m], when the first step that computed the request was launched
    first_runner_ns: np.ndarray  # int64 [m], runner time of the steps before the first token (TTFT budget)
    t_sent_ns: int
    stats: EngineStats | None = None
    # Requests with images whose pixels the frontend may free: they left the engine and every step launched until
    # then completed (an abort in flight included). Sent once per request, finished or not.
    released: list[int] = []
    seq: int = 0  # this frontend's message number (from 1), which it acknowledges (`Ack`)
    # MTP: requests finishing in this message, with their verify rows and the tokens those delivered (all admissions)
    spec_rids: list[int] = []
    spec_rows: list[int] = []
    spec_tokens: list[int] = []
    first_t_queued_ns: list[int] = []  # per first row: when the engine took the request (`Scheduler.add`)
    logprobs: Logprobs | None = None  # with a request that asked for them


@dataclass
class EngineReady:
    """The engine's start, as it reports it to the launcher (over its ready pipe, pickled): its effective
    configuration, its runner's capabilities and the seconds it took."""

    cfg: Any
    capabilities: Any
    startup_s: float


@dataclass
class EngineFailed:
    """The engine's start failed (its ready pipe): the exception's type and message."""

    error: str


MESSAGE_TYPES: dict[type, bytes] = {AddRequest: b"A", Abort: b"X", Ack: b"K", Shutdown: b"S", StepOutputs: b"O"}
