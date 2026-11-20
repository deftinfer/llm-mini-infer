"""The model-runner interface: what the engine's step loop needs from a backend (CUDA, or the fake).

Lifecycle, all on the engine thread:

    runner = build_runner(cfg)                       # engine.process: load weights
    caps = runner.capabilities()                     # geometry, memory and features; the engine sizes pools
    runner.configure(PoolConfig(...))                # allocate pools and slot tables, warm up (graphs, JIT)
    loop:                                            # one step ahead of the device (architecture.md, "The step loop")
        h1 = runner.launch_forward(plan_n1)          # enqueue plan effects + forward of step N+1; returns at once
        res = runner.wait(h0)                        # tokens of step N
        ...                                          # the engine processes N, sends its outputs
        n = fill(runner.mask_buffer(h1))             # grammar masks of N+1's MASKED rows, if any
        runner.launch_sample(h1, n)                  # enqueue sampling of N+1 (+ MTP drafts for N+2)
    runner.shutdown()

Contract (architecture.md, "The step plan"; the plan's sections, effects and lifetime: `sched.plan`):
- **Order.** All effects are stream-ordered: plan N+1's table updates and forward run after step N's sampling, in
  call order, so a slot or page freed and reused in a later plan never races work still in flight. Calls always
  alternate `launch_forward(N+1)`, [`wait(N)`], `launch_sample(N+1)`; at most two steps are in flight.
- **Slot tables.** The runner keeps, per slot: its page-table rows, the live state slot (models with recurrent state),
  the sampling parameters and seed, device `seq_len` and the pending token, the commit limits (MTP) and the rope offset
  (M-RoPE). Only plans write them, except that sampling advances `seq_len` and the pending token. The runner never
  allocates an ID and never decides that a request is finished.
- **Device input.** A SCHED row with `tok_off == -1` reads its input from the slot's pending token and its position
  from the device `seq_len`. The host never supplies that input, so the step launches before the previous step's
  tokens reach the host.
- **MTP** (`mtp_depth` k > 0). A device-input row verifies `1 + v` tokens: the pending token and the first v of the
  drafts the runner drafted for the slot in its previous SAMPLE step (v = k, or one of `draft_counts`: the plan's
  `sched_n_draft`, the same on every device row of a plan; never more than that step drafted). It returns `n_out = 1 +
  accepted` tokens (the accepted drafts, then the bonus token, which becomes pending) and advances `seq_len` by as
  much. Drafts are accepted while they equal the target's samples (the sampler's noise of their position), so the
  tokens are exactly the target's sampled sequence; acceptance also stops at the slot's commit limits (ADMIT). Every
  SAMPLE row drafts k tokens for the next step, or the plan's `draft_n` (one of `draft_counts`), and returns them
  (`StepResult.drafts`, −1 past them): the engine computes the next step's masks along them. While device-input rows of
  a slot are in flight, the host knows its `seq_len` only as a range: each advances it by 1 to 1 + k.
- **Results** are per SCHED row, in plan order (`StepResult`). `wait` must not block on anything but the device
  work of that step.
- **NONFINITE** may be reported on any row. A runner may see a row's NaN only at a later SAMPLE row that reads it (the
  Qwen runner checks SAMPLE rows' logits alone), so the scheduler trusts what a request's prefill wrote only once one
  of its SAMPLE rows was finite (architecture.md, "Failures and rollback").
- **Seeds.** Sampling noise is a function of (seed, position, token), independent of the batch.
- **Grammar masks** (`grammar_masks`). A MASKED row's masks depend on the tokens the slot's previous step delivered (and
  on its drafts), which the engine learns only from `wait` of that step, after `launch_forward` of this one. So the
  engine fills them between the two calls, into `mask_buffer(handle)` (int32 bit words: token t is allowed when bit t %
  32 of word t // 32 is set; one row per mask, in plan order: 1 + v per MTP device-input row, 1 per other row), and
  passes their count to `launch_sample`, which may run none of the step's sampling before. A step without MASKED rows
  needs no masks (count 0). A mask is applied to the target's logits before the sampler's noise, so verification
  compares the drafts with masked samples. A used sample whose mask allows nothing reports `MASK_EMPTY`.
- **Sliding windows** (`window` W > 0). Window pages and full pages are `num_pages` pages of equal bytes, one id space
  (`cache.pools.size_arena`): a runner lays its KV out so that a page id addresses the same bytes as either kind.
- **Images** (`images`): the runner owns an image encoder and a pool of `encoder_rows` embedding rows, which only plans
  fill (ENCODE). With `bidirectional_images` it masks each span the plan names (SPANS) within its step alone.
- **Logprobs** (`logprobs`): for each LOGPROBS row, the values of every sample it draws, from the logits its sampler
  read, its grammar mask applied: vLLM's default `raw_logprobs` (before temperature and the filters), over the
  whole vocabulary (`LogprobRows`). They come back with the tokens; computing them never changes a token.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

import msgspec
import numpy as np

from ..sched.plan import StepPlan

MAX_LOGPROBS = 20  # the top entries a sample's logprobs carry (OpenAI's limit; vLLM's --max-logprobs default)

# StepResult.status bits
NONFINITE = 1  # a non-finite logit or state: the row's tokens are invalid and what it wrote is poisoned (contract)
MASK_EMPTY = 2  # a used sample's grammar mask allowed no token (MASKED rows): its token is invalid


class PlanError(ValueError):
    """The scheduler sent a plan that breaks the runner contract (every runner raises it)."""


class Capabilities(msgspec.Struct, frozen=True, kw_only=True):
    """Reported once at startup. The engine adapts to it instead of branching on backend."""

    backend: str  # "cuda", "fake"
    device: str
    vocab_size: int  # logits width, padding included
    max_positions: int  # the model's position limit
    # Memory and geometry: the engine does the pool arithmetic (architecture.md, "Pools").
    memory_total_bytes: int
    memory_free_bytes: int  # usable for the KV and state pools, after weights, activations and graphs
    kv_bytes_per_token: int  # all full-attention layers, the MTP drafter's included, at the KV dtype
    state_slot_bytes: int  # one recurrent-state slot: all linear-attention layers + MTP hidden; 0 if none
    window: int = 0  # sliding-window layers' window, the query's own key included (0: none)
    window_kv_bytes_per_token: int = 0  # all sliding-window layers' KV of one position (the window pool)
    page_size_multiple: int = 1  # page sizes the attention kernels accept are multiples of this
    checkpoint_grid: int = 64  # GDN chunk size; state checkpoints sit on this grid
    # Features
    max_steps_in_flight: int = 2  # D in `sched.plan.reserve_positions`
    mtp_depth: int = 0  # k; 0 = no MTP
    # MTP: the counts below k a plan may verify and draft (`sched.plan`: `sched_n_draft`, `draft_n`); empty: k alone
    draft_counts: tuple[int, ...] = ()
    # The MTP drafter writes KV of its own at the target's positions (Qwen's MTP layer; `sched.plan`'s module docstring,
    # "MTP drafting"). False: it reads the target's KV and writes none (Gemma 4's assistant, cross-model KV sharing)
    draft_kv: bool = True
    cuda_graphs: bool = False
    images: bool = False  # runs ENCODE: an image encoder and its pool of embedding rows
    encoder_rows: int = 0  # embedding rows of the encoder pool (the scheduler's encoder cache)
    encoder_budget: int = 0  # image tokens one step may encode (the memory profile ran this many)
    # An image span's tokens attend to each other in both directions; the scheduler keeps each span whole in one row
    # (SPANS: `sched.plan`)
    bidirectional_images: bool = False
    grammar_masks: bool = False  # takes MASKED rows: `mask_buffer` and masks in `launch_sample`
    logprobs: bool = False  # takes LOGPROBS rows (`sched.plan`) and returns their `LogprobRows`
    penalties: bool = False
    checkpoints: str = "none"  # "none", "every_n" (recurrent-state slots), "window" (sliding-window KV)
    # Measured step costs: device time of a decode step of `decode_step_rows` rows and of a prefill step of
    # `prefill_step_tokens` tokens, timed warm by the memory profile; 0 = not measured. The engine derives the
    # scheduler's mixed-step budget from them (`sched.scheduler.derive_mixed_step_row_tokens`).
    decode_step_ms: float = 0.0
    decode_step_rows: int = 0
    prefill_step_ms: float = 0.0
    prefill_step_tokens: int = 0
    # PRECISE rows (`sched.plan`; architecture.md, "The precise prefix"): a block-FP8 model computes them with BF16
    # activations, and the precise prefix is on (`runner.cuda.fp8_cold_prefill` reuse or bf16). `precise_step_ms`: the
    # device time of the same prefill step as `prefill_step_ms` with every row PRECISE (0 = not measured): the scheduler
    # weighs PRECISE tokens in the mixed-step bound by its ratio to `prefill_step_ms`
    precise_rows: bool = False
    precise_step_ms: float = 0.0
    # `Runner.forward_pending` (architecture.md, "Planning at the forward/sampling boundary"): the runner reports when a
    # split step's forward ends on the device, and its memory profile timed a lone decode step's split: the device time
    # of its sampling part (`sample_step_ms`) and the host time of launching it (`launch_step_ms`). The engine checks a
    # set `engine.plan_at_boundary` against them (`engine.core.derive_plan_at_boundary`)
    forward_end: bool = False
    sample_step_ms: float = 0.0
    launch_step_ms: float = 0.0
    # Batch invariance (`dev.runner.cuda.batch_invariant`; architecture.md, "Batch invariance"): a row's bits do not
    # depend on the rows beside it, so the scheduler keeps a prompt's chunking from depending on them too (a GDN model's
    # chunks on the grid, every final chunk's forced prefix whole: `sched.scheduler`)
    batch_invariant: bool = False


class PoolConfig(msgspec.Struct, frozen=True, kw_only=True):
    """Sent once to `configure`. Page 0 and state slot 0 are null/padding."""

    page_size: int  # positions of a full-attention KV page (the radix tree's page)
    num_pages: int  # page 0 included; sliding-window models: the arena's pages, of either kind
    num_state_slots: int  # slot 0 included
    max_num_seqs: int  # request slots are 1..max_num_seqs; slot 0 is padding
    max_num_batched_tokens: int
    max_model_len: int
    # Read by the fake alone: no unmasked sample is an EOS, except at `eos_at`. Finishing stays in the engine
    eos_token_ids: tuple[int, ...]
    mtp_k: int = 0
    window_page_size: int = 0  # sliding-window models: positions of a window page (one arena); 0 without


class LogprobRows(msgspec.Struct, kw_only=True):
    """The logprobs of a step's LOGPROBS rows (`Capabilities.logprobs`), per sample each row draws: L = 1 + k on an MTP
    device-input row (the first `n_out` delivered, the rest unused), 1 on any other row (L = 1 + k columns either way
    under MTP, the second on unused). Each value is `(x_t − max x) − log Σ exp(x − max x)` in fp32 over the sample's
    logits x, a token its mask refuses at −inf; `top_ids` holds the MAX_LOGPROBS tokens with the highest logits, ties to
    the lowest id, then (fewer allowed) the lowest refused ids at −inf."""

    rows: np.ndarray  # int32 [m]: their SCHED rows (plan order)
    token: np.ndarray  # float32 [m, L]: the sampled token's logprob
    top_ids: np.ndarray  # int32 [m, L, MAX_LOGPROBS], in descending order
    top: np.ndarray  # float32 [m, L, MAX_LOGPROBS]: their logprobs


class StepResult(msgspec.Struct, kw_only=True):
    """Per SCHED row, in plan order. Timing fields are device time in ns (0 when the backend cannot measure)."""

    tokens: np.ndarray  # int32 [n_sched, 1 + k]; -1 padding; row i holds n_out[i] tokens (a finish token last)
    n_out: np.ndarray  # int32 [n_sched]: 0 without SAMPLE, 1 with, 1 + accepted drafts on MTP decode rows
    status: np.ndarray | None = None  # uint32 [n_sched]: bit 0 NONFINITE (contract), bit 1 MASK_EMPTY; None = all ok
    drafts: np.ndarray | None = None  # int32 [n_sched, k] (MTP): each SAMPLE row's drafts for its slot's next step (−1
    # past the plan's `draft_n`)
    gpu_busy_ns: int = 0  # device time of this step's forward + sampling
    gpu_idle_before_ns: int = 0  # device idle right before this step's forward (nothing was queued)
    gpu_idle_inside_ns: int = 0  # device idle between this step's forward and sampling (sampling launched late)
    encoder_busy_ns: int = 0  # device time of the step's ENCODE (part of gpu_busy_ns)
    gpu_sample_ns: int = 0  # device time of a split step's sampling part (part of gpu_busy_ns; 0: not split)
    logprobs: LogprobRows | None = None  # the LOGPROBS rows' values (None: the step has none)


class AttentionCounters(msgspec.Struct, frozen=True, kw_only=True):
    """FlashInfer's attention plans for rows whose KV lengths the host knows only within bounds (MTP; the chunk-count
    hazard of `kernels.flashinfer`'s module docstring), cumulative over a runner's instances since `configure` made
    them, its warm-up and captures included (`Runner.attention_counters`; llm_mini_infer:attention_*_total on /metrics:
    a plan that attends wrongly shows otherwise only as falling acceptance)."""

    # plans made again with chunks whose count no true length can change (`FlashInferAttention._plan_safely`); a
    # decode graph's re-plan that FlashInfer reports unsplit is undone and counts in `kept` too
    replans: int = 0
    kept: int = 0  # decode-graph plans kept although no safe split fit their grid: those rows may attend wrongly
    unsplit: int = 0  # plans made without a split instead (the self-check's fallback: correct, slower)
    # the vision tower's plans that waited for their page-locked buffer's last copy (`FlashInferVisionAttention.waits`:
    # more encoder calls in flight than its buffers), a host wait in the step's launch
    vision_waits: int = 0


def sum_attention_counters(attns: Iterable[Any]) -> AttentionCounters:
    """The counters of FlashInfer attention instances (`FlashInferAttention.replans`, `kept`, `unsplit`: host ints its
    planning keeps, so no device sync), summed: each instance once (defensive: no runner holds one under two names),
    none of those without them (None, the torch references)."""
    seen = {id(a): a for a in attns if hasattr(a, "replans")}.values()
    return AttentionCounters(replans=sum(a.replans for a in seen), kept=sum(a.kept for a in seen),
                             unsplit=sum(a.unsplit for a in seen))


def vision_waits(counters: AttentionCounters, img: Any) -> AttentionCounters:
    """`counters` with the vision tower's plan waits (`FlashInferVisionAttention.waits`, a host int) of a runner's image
    encoder `img` (None: no images; its torch attention has none)."""
    waits = getattr(getattr(img, "attn", None), "waits", 0)
    return msgspec.structs.replace(counters, vision_waits=waits) if waits else counters


# Powers of two up to 4,096, fine where a grammar's mask unions usually fall and around the sparse head's default cap
# (1,024 rows), then powers of four up to Gemma 4's vocabulary
UNION_BUCKETS = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 16384, 65536, 262144)  # union-size histogram bounds


class HeadCounters(msgspec.Struct, frozen=True, kw_only=True):
    """The grammar-aware heads' counters (`Runner.head_counters`): host ints, cumulative since start.

    Every step with MASKED rows (its sampling runs in `launch_sample`, after the engine filled the masks) counts in
    `sparse` or in one `dense_*` reason. llm_mini_infer:grammar_head_* and llm_mini_infer:draft_universe_* on
    /metrics."""

    sparse: int = 0  # split steps sampled on the sparse head
    dense_off: int = 0  # split steps on the dense head because dev.runner.cuda.sparse_head is off
    dense_unconstrained: int = 0  # ... because a sample token had no mask, or a mask allowed every token
    dense_over_cap: int = 0  # ... because the union exceeded sparse_head_rows
    # per UNION_BUCKETS bound (non-cumulative counts; len == len(UNION_BUCKETS) + 1, the last for unions above the last
    # bound; () on a runner without the heads), over every step with all sample tokens masked and a head on. A union of
    # s tokens counts at bisect_left(UNION_BUCKETS, s): Prometheus's le, as `metrics.Histogram.observe` buckets
    union_counts: tuple[int, ...] = ()
    union_sum: int = 0  # sum of those unions' sizes
    universe: int = 0  # tokens in the draft universe now (0: off or empty)
    universe_steps: int = 0  # steps whose drafts used the universe head
    universe_rebuilds: int = 0  # times the universe head was rebuilt (it grew)


class Runner(ABC):
    @abstractmethod
    def capabilities(self) -> Capabilities: ...

    @abstractmethod
    def configure(self, pools: PoolConfig) -> None:
        """Allocate pools and slot tables, then warm up. Returns once the runner can take plans."""

    @abstractmethod
    def launch_forward(self, plan: StepPlan) -> Any:
        """Enqueue the plan's effects before SCHED (`sched.plan`), then the forward pass over SCHED. Must not block on
        the device. Returns a handle for `launch_sample` and `wait`."""

    @abstractmethod
    def launch_sample(self, handle: Any, n_masks: int = 0) -> None:
        """Enqueue sampling for the SAMPLE rows of that step, with the `n_masks` masks of its MASKED rows the engine
        wrote into `mask_buffer(handle)`, and the copy of the sampled tokens to the host. Advances device `seq_len`
        and pending tokens (`sched.plan`)."""

    def mask_buffer(self, handle: Any) -> np.ndarray:
        """Where the engine writes that step's grammar masks before `launch_sample`: int32 [capacity, words], words =
        ceil(vocab_size / 32), C-contiguous (page-locked on CUDA); valid until `launch_sample` returns."""
        raise PlanError("this runner takes no grammar masks")

    @abstractmethod
    def wait(self, handle: Any) -> StepResult:
        """Block until that step's tokens are on the host and return them."""

    def forward_pending(self, handle: Any) -> bool:
        """Whether that step, launched and sampled (`launch_sample` returned), still runs its forward on the device:
        True only for a step split into a forward and a sampling (`Capabilities.forward_end`) whose forward has not
        ended. The engine polls until it is False before planning the next step (`engine.plan_at_boundary`); a runner
        that cannot tell says False, and the engine plans at once (architecture.md, "Planning at the forward/sampling
        boundary")."""
        return False

    def attention_counters(self) -> AttentionCounters:
        """FlashInfer's attention-plan counters (`AttentionCounters`), which the engine reads into every step's stats:
        host ints, no device work. A runner without FlashInfer attention (the fake, the torch references) has none."""
        return AttentionCounters()

    def head_counters(self) -> HeadCounters:
        """The grammar-aware heads' counters (`HeadCounters`), read as `attention_counters` are. A runner without
        those heads (the fake) has none."""
        return HeadCounters()

    def shutdown(self) -> None:  # noqa: B027 - optional hook
        """Release device resources."""
