"""The scheduler: one `StepPlan` per step, and each step's results folded back in (architecture.md, "The scheduler").

`Scheduler` owns the requests (`Req`), the slot arrays, the pools' bookkeeping, the radix cache (`cache.radix`) and the
encoder cache's bookkeeping (`cache.encoder`). The engine calls `add` and `abort` as messages arrive, `schedule` for
step N + 1's plan, `process` with step N's results (it returns each frontend's outputs, `FrontendOutputs`), then
`fill_masks` for step N + 1's grammar masks. `Inflight` keeps what reading a plan's results needs; `StepBudget` what
the plan under construction may still take. The policies are architecture.md's: "The scheduler", "The cache",
"Structured output", "Grammar-aware heads" and "Batch invariance".

**State.** Per-request state the hot path touches is struct-of-arrays, indexed by slot. A steady decode step is a
handful of numpy operations over the running slots; Python runs per request only when its state changes (admitted,
finished, preempted, crossing a page boundary, first token, a checkpoint), and for a constrained request's grammar.

**A request's state** is its `status`, its entry in the waiting heap (`_waiting`, cleaned lazily: an entry whose
request no longer waits is dropped once it reaches the head), the count of waiting requests (`num_waiting`), its entry
in the compiling set (`_compiling`) and its place in `reqs`. Three methods change them together: `_to_waiting` (added,
its grammar built, preempted, or deferred by `_admit`), `_to_running` (admitted: the heap's head, which `_admit_one`
made RUNNING with its slot) and `_to_finished` (finished, failed or aborted). A request whose grammar is being built
(`Grammars.submit`) starts COMPILING: in `reqs` with its arrival number and its seed, outside the heap, so later ready
requests overtake it; `collect` makes it WAITING once its grammar is built, or FINISHED when the build fails or takes
longer than `compile_timeout_s`. COMPILING goes to WAITING or FINISHED (also an abort), WAITING to RUNNING or FINISHED
(an abort, a held sample's finish, a grammar that fails at admission), RUNNING back to WAITING (a preemption) or to
FINISHED, which is final. `check_accounting` checks them.

**Bookkeeping rules** (architecture.md, "The step plan"):
- `mirror[slot]` mirrors the device `seq_len`: explicit tokens advance it by `n_query` when scheduled, device-input
  decode by `n_out` when the result arrives; `dev_inflight[slot]` counts device-input rows in flight. Without MTP
  `mirror + dev_inflight` is what the device will have computed once they ran; under MTP an in-flight decode row
  computes at least its first token (the pending one, at `mirror`).
- Before each decode row, pages cover positions `< mirror + reserve` (`plan.reserve_positions`: two steps of 1 + k
  tokens, plus the drafter's chain). A final chunk's pages cover its row and, with a drafter that writes KV, the
  k − 1 positions its chain writes past the row (`_chunk_pages`).
- Results are matched by position in the in-flight plan. A row counts only if its slot still holds the same admission
  (`aid`): rows of finished, aborted or failed requests are dropped, so the finish race (the request is already in the
  next plan when its EOS arrives) costs at most one wasted step. A preempted request's sample in flight is delivered to
  it while it waits, and its readmission waits for it (`_deliver_held`).
- **MTP:** the runner never accepts past a commit limit (ADMIT: `max_len`, the first `STOP_W` stop tokens), so a finish
  token ends its row; tokens after a stop token beyond `STOP_W` are dropped, and the finish caches the KV only
  (`overrun`). A finish, abort or preemption with a decode step in flight caches the KV up to `mirror + 1` (that step
  always computes the pending token) and donates the live state folded to the same position: the plan that FREEs the
  slot FOLDs it first (`sched.plan`).
- A slot freed by a finish, abort or preemption is cleared by FREE in the next plan, and can be readmitted in that same
  plan (FREE runs before ADMIT). Pages and state slots are reusable at once: every later plan runs after the steps in
  flight on the device stream.
- No step is scheduled for a request whose samples in flight may already reach `max_tokens`, so a length finish wastes
  nothing. Under MTP a decode row in flight may deliver 1 + k tokens, and the request waits for its result when that
  could finish it.

**Admission** matches the request's tokens in the radix tree under its `cache_salt` and resumes at the hit (at most its
tokens − 1: the prompt, plus on a readmission what it generated). It shares the tree's full pages (PAGES), copies a
partial last page (KV_COPY) and, on a GDN model, the checkpoint into its live state slot (STATE_COPY); a window
checkpoint's pages are shared by reference. A miss starts from STATE_ZERO. A prefill chunk's KV and checkpoints go
into the tree when the chunk is planned, stamped with its step; a decode checkpoint when its step's result arrives.

**Rollback** (architecture.md, "Failures and rollback"). A row reporting NONFINITE fails its request (which rows report
it: `runner.base`). What the request wrote in that step and in the steps still in flight is rolled back, and so are its
**unverified writes**: what its prefill wrote before a SAMPLE row of it could show it finite (`_unverified`,
`RadixCache.unverified`). The removal takes the nodes that hold those pages, with the paths below them and the
checkpoints the request attached meanwhile: nothing it only shared, unless page pressure made the tree take its pages.
Running requests that resumed from a rolled-back write fail too (`deps`, `_fail_cascade`); one that only locked a
removed node runs on but caches nothing more (`nocache`). An abort removes the request's unverified writes, and a
readmission keeps those it reads (`_readmit_unverified`). An empty grammar mask (MASK_EMPTY) fails its request alone:
its token is invalid, not what it computed.

**The precise prefix** (`precise`: `runner.cuda.fp8_cold_prefill` on a runner with PRECISE rows; architecture.md, "The
precise prefix"). PRECISE rows compute a shared prefix with BF16 activations into the precise namespace
(`cache.radix.precise_ns`). Under `reuse`, an admission whose best hit is an FP8 checkpoint that has served
`refine_after` resumes (`Node.hits`) and lies at least `refine_min` past the deepest precise checkpoint on its path is a
**refiner**: its PRECISE rows end exactly at that checkpoint (source `refine`), then the rest of its prompt runs on the
FP8 path. A shorter span is resumed from as a first reuse is, unless its writer stood on the FP8 copy
(`Node.on_precise` false) or it is shorter than a checkpoint grid: the request then resumes from the precise copy below
it (keeping it locked) and computes the span on the FP8 path. Refines of one checkpoint never overlap
(`_refining_now`). A request whose computed positions are all precise donates into the precise namespace (`_keep`).
"""

from __future__ import annotations

import contextlib
import functools
import heapq
import logging
import math
import random
import time
from collections.abc import Callable, Hashable, Iterator
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Literal, NamedTuple

import numpy as np

from ..cache.encoder import EncoderCache, Entry
from ..cache.pools import PagePool, SlotPool, StatePool, WindowStore, window_pages_for_one_request
from ..cache.radix import KEY_DTYPE, Match, Node, RadixCache, Spans, is_precise, lost_tokens, precise_ns
from ..engine.messages import (
    ABORT_REASONS,
    CHECKPOINT_SOURCES,
    LOST_CAUSES,
    MASK_CLASSES,
    POPCOUNT_BUCKETS,
    AddRequest,
    FinishReason,
)
from ..grammar import GrammarError
from ..logutil import traceback_without_messages
from .draft_counts import DraftCounts
from .plan import (
    ADMIT_CONSTRAINED,
    ADMIT_LOGPROBS,
    CKPT_END_OF_STEP,
    CKPT_GRID,
    FOLD_ALL,
    LOGPROBS,
    MASKED,
    MAX_LOGIT_BIAS,
    PRECISE,
    SAMPLE,
    STATE_ZERO,
    PlanBuilder,
    StepPlan,
    reserve_positions,
)

# The family's M-RoPE positions (`vision.family.ImageFamily.positions`)
Positions = Callable[[int, list], tuple]

log = logging.getLogger(__name__)

WAITING, RUNNING, FINISHED, COMPILING = 0, 1, 2, 3
GRAMMAR_FAILED = "its grammar does not compile with this tokenizer"  # a rejection's reason (llguidance's quotes it)


class Unverified(IntEnum):
    """What becomes of a request's unverified writes (`_unverified`) when it ends, or when a held sample of it reads
    them (`_settle_unverified`): NONE, nothing, as a finish leaves them without a call (the row that finished it read
    them); KEEP, they are verified (a finite sample read them all), or valid (an empty mask computed nothing wrong);
    DROP, removed now (a NaN, or an abort: nothing will read them); ORPHAN, removed at the next `process` (a plan being
    built may read them)."""

    NONE = 0
    KEEP = 1
    DROP = 2
    ORPHAN = 3

NONFINITE = 1  # StepResult.status bit 0
MASK_EMPTY = 2  # StepResult.status bit 1
MAX_LOGPROBS = 20  # runner.base.MAX_LOGPROBS: the top entries a sample's logprobs carry


def forced_logprobs(tokens: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The logprobs of tokens no sampler drew (the forced prefix, the EOS-free finish), as `process` carries them (each
    token's logprob [n], the top ids [n, MAX_LOGPROBS], their logprobs): each one's mask allows it alone
    (`Grammars.forced_prefix`, a stopped grammar's one EOS), so its logprob is 0 whatever the logits, and the rest of
    its top entries are the lowest other ids at −inf, as a sampled step under that mask gives them
    (`runner.base.LogprobRows`)."""
    n = len(tokens)
    ids = np.zeros((n, MAX_LOGPROBS), np.int32)
    for i, t in enumerate(tokens):
        ids[i] = [t, *(j for j in range(MAX_LOGPROBS) if j != t)][:MAX_LOGPROBS]
    top = np.full((n, MAX_LOGPROBS), float("-inf"), np.float32)
    top[:, 0] = 0.0
    return np.zeros(n, np.float32), ids, top


def _lp_cat(parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return tuple(np.concatenate([p[j] for p in parts]) for j in range(3))  # type: ignore[return-value]


def _lp_unused(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Logprobs for n tokens of requests that asked for none (never read)."""
    return (np.full(n, np.nan, np.float32), np.zeros((n, MAX_LOGPROBS), np.int32),
            np.full((n, MAX_LOGPROBS), np.nan, np.float32))
_OUT_W = 256  # per-slot ring of generated tokens; full rows are flushed to the request (needed for recompute)
GROW_ROWS = 4  # decode rows crossing a page boundary in one step from which `_grow_rows` takes their pages at once


class Req:
    """One request; its hot fields live in the slot arrays (module docstring, State)."""

    __slots__ = (
        "arrival",
        "base_precise",
        "bias",
        "cached",
        "compiled",
        "continues",
        "cow",
        "cow_unv",
        "decode_ckpt",
        "demoted",
        "deps",
        "dup",
        "dup_swaps",
        "enc",
        "first_done",
        "forced",
        "frontend",
        "future",
        "grammar",
        "held",
        "hints",
        "hit",
        "ignore_eos",
        "images",
        "key",
        "key64",
        "label",
        "left",
        "lock",
        "logprobs",
        "matcher",
        "max_out",
        "min_p",
        "n_out",
        "nocache",
        "ns",
        "num_preemptions",
        "out_chunks",
        "own_ckpts",
        "pages",
        "pen",
        "pend",
        "plock",
        "priority",
        "prompt_len",
        "refine",
        "resumed",
        "rid",
        "rope",
        "rope_off",
        "runner_ns",
        "seed",
        "slot",
        "sp_end",
        "sp_id",
        "sp_key",
        "sp_start",
        "spans",
        "state_slot",
        "status",
        "stop_ids",
        "stop_list",
        "stop_strings",
        "t_admit_ns",
        "t_launch_ns",
        "t_queued_ns",
        "t_recv_ns",
        "targets",
        "temperature",
        "tokens",
        "top_k",
        "top_p",
        "unrep",
        "unv",
        "unv_aids",
        "unv_ckpts",
        "unv_ns",
        "unv_pages",
        "verify_rows",
        "verify_tokens",
        "waits",
        "wpages",
    )

    def __init__(self, msg: AddRequest, seed: int, arrival: int, max_model_len: int):
        self.key = (msg.frontend, msg.rid)
        self.label = msg.request_id or f"{msg.frontend}:{msg.rid}"  # in log lines: the request's id (X-Request-Id)
        self.frontend = msg.frontend
        self.rid = msg.rid
        self.tokens = np.ascontiguousarray(msg.prompt, dtype=np.int32).copy()  # prompt (+ outputs after preemption)
        self.prompt_len = len(self.tokens)
        self.max_out = min(msg.max_tokens, max_model_len - self.prompt_len)
        # top_k, the seed and the priority checked against what the plans and slot arrays hold (int32, uint64, int64):
        # a value that does not fit fails `Scheduler.add`, so its request alone, not the plan that would admit it
        self.temperature, self.top_p, self.min_p = msg.temperature, msg.top_p, msg.min_p
        self.top_k = int(np.int32(msg.top_k))
        self.seed = int(np.uint64(seed))
        # Penalties (repetition, presence, frequency; None: neutral) and the logit bias (ids ascending, values; None)
        pen = (float(msg.repetition_penalty), float(msg.presence_penalty), float(msg.frequency_penalty))
        self.pen = pen if pen != (1.0, 0.0, 0.0) else None
        self.bias = (np.asarray(msg.bias_ids, np.int32), np.asarray(msg.bias_values, np.float32)) \
            if len(msg.bias_ids) else None
        self.stop_ids = np.asarray(msg.stop_token_ids, np.int32)
        self.ignore_eos = msg.ignore_eos
        self.stop_list: list[int] = []  # the commit limit's stop tokens (set by the scheduler under MTP)
        # the frontend ends it on stop strings (architecture.md, "Grammar-aware heads")
        self.stop_strings = msg.stop_strings
        self.priority = int(np.int64(msg.priority))
        self.arrival = arrival
        self.t_recv_ns = msg.t_recv_ns
        self.t_queued_ns = time.perf_counter_ns()  # the engine took it: queue time starts
        self.status = WAITING
        self.slot = 0
        self.state_slot = 0
        self.pages: list[int] = []
        # The output ring: while the request runs, its slot's `n_out` counts the tokens it generated, `out_chunks`
        # holds those of the rows of the slot's `out_buf` that filled (a copy each, flushed as it filled:
        # `Scheduler._flush_full`; or one array of whole rows after a readmission), and `out_buf` the rest, the row
        # being written. While it waits after a preemption (`_preempt`), `out_chunks` is one array of all `n_out`
        # tokens, which a held sample extends (`_deliver_held`) and the readmission splits back (`_admit_one`).
        # `_seq_tokens` reads both
        self.out_chunks: list[np.ndarray] = []
        self.n_out = 0
        self.num_preemptions = 0
        self.t_admit_ns = 0
        self.t_launch_ns = 0
        self.runner_ns = 0
        self.first_done = False
        self.verify_rows = self.verify_tokens = 0  # MTP: its verify rows and the tokens they delivered, while it waits
        self.ns = msg.cache_salt
        self.hints = sorted({int(h) for h in msg.cache_hints if 0 < int(h) < self.prompt_len})
        self.key64: np.ndarray | None = None  # radix key: the tokens as int64 (image spans: the images' keys)
        self.cached = -1  # prompt tokens served from the cache at the first admission
        self.lock: Node | None = None  # deepest radix node this request holds locked (the FP8 namespace)
        self.plock: Node | None = None  # and in its precise namespace (the precise prefix's copies)
        self.refine = 0  # its prefill rows before this position are PRECISE (the precise prefix; 0: none)
        self.pend = 0  # what it computed is precise up to here (a precise hit, then its PRECISE rows)
        self.base_precise = False  # it stands on a precise copy (a precise hit, a refine, or an FP8 hit on one)
        self.deps: set[tuple[int, int]] = set()  # (aid, step) of in-flight cache writes its hit used
        self.targets: list[tuple[int, str]] = []  # desired checkpoint positions of this prefill, with their source
        self.waits = 0
        self.decode_ckpt: tuple[Node, int, int] | None = None  # latest decode checkpoint: (node, slot, step)
        # What its deeper checkpoints supersede (`Scheduler._supersede`): the checkpoint this admission resumed from,
        # and the end-of-prefill and branch checkpoints it took, each (node, slot, step); whether it continues where an
        # earlier request stopped (it resumed from a finish, preemption or decode checkpoint, or its key runs through a
        # cached path's whole last node); the checkpoints it demoted (undone if it fails); and the checkpoint its last
        # preemption left
        self.resumed: tuple[Node, int, int] | None = None
        self.own_ckpts: list[tuple[Node, int, int]] = []
        self.continues = False
        self.demoted: list[tuple[Node, int, int]] = []
        self.left: tuple[Node, int, int] | None = None
        self.wpages: list[int] = []  # window-pool pages by page index (0: released below the window, or none)
        self.held = 0  # preempted with a sample in flight: the admission ID that sample belongs to
        self.dup = False  # may hold pages that duplicate tree pages its lock pins (RadixCache.kept_dups)
        self.dup_swaps = -1  # RadixCache.swaps when `dup` was last found true in this admission (`_readopt`)
        # Its admission's hit, and its copy of the tree's page holding it (KV_COPY), which the tree never adopts into
        # a node that starts before the hit (RadixCache.insert); whether that copy, under MTP with drafter KV the full
        # page below the hit, copied unverified KV or rewrote its drafter KV at hit − 1 from an unverified checkpoint,
        # which makes it one of its unverified writes (`_unverified`)
        self.hit = 0
        self.cow = 0
        self.cow_unv = False
        self.nocache = False  # a removal took a node it locked: it caches nothing more (module docstring, Rollback)
        # Its unverified writes (module docstring, Rollback): the start of the first of its prefill chunks that no
        # finite SAMPLE row of it has read yet (-1: none; with `cow_unv`, its admission's copy of the page below its hit
        # is one of them too), the namespaces they went into, the admissions (aid) that wrote them, the pages those
        # marked (RadixCache.unverified) and the checkpoints computed from them: (node, slot, stamp)
        self.unv = -1
        self.unv_ns: set = set()
        self.unv_aids: list[int] = []
        self.unv_pages: list[int] = []
        self.unv_ckpts: list[tuple[Node, int, int]] = []
        # Images: spans [sp_start, sp_end) in the prompt with their radix keys, and as the radix tree checks them (with
        # their images' full digests: `cache.radix`, Image spans); each image's identity in the encoder cache (its key,
        # its digest and the request's cache_salt, last: two salts share no entry); the encoder-cache entries this
        # request references (spans it has not finished prefilling); M-RoPE deltas of the prompt [3, prompt_len] (None:
        # positions are the sequence indices) and of every later token
        self.images = msg.images
        self.sp_start = np.asarray([im.offset for im in msg.images], np.int64)
        self.sp_end = self.sp_start + np.asarray([im.tokens for im in msg.images], np.int64)
        self.sp_key = np.asarray([im.key for im in msg.images], np.int64)
        self.sp_id = [(im.key, im.digest, self.ns) for im in msg.images]
        self.spans = Spans(self.sp_start.tolist(), self.sp_end.tolist(), [im.digest for im in msg.images]) \
            if msg.images else None
        self.enc: list[Entry | None] = [None] * len(msg.images)
        self.rope: np.ndarray | None = None
        self.rope_off = 0
        # Structured output: the grammar text, its compiled form, while COMPILING the build's future
        # (`Grammars.submit`) and, while admitted, the matcher
        self.grammar = msg.grammar
        self.compiled = None
        self.future: Any = None
        self.matcher = None
        # The forced prefix (architecture.md, "Grammar-aware heads"): the tokens its grammar forces at its start (cut
        # before a stop token), and the output tokens it was given that its frontend has not received yet (forced ones,
        # while the chunk that computes them is in flight, or after a preemption dropped their result)
        self.forced: tuple[int, ...] = ()
        self.unrep: list[int] = []
        self.logprobs = msg.logprobs  # its tokens' logprobs, with this many top entries (-1: none)


@dataclass
class Inflight:
    """What the scheduler needs to read a step's results: its SCHED rows, by position."""

    step_id: int
    slots: np.ndarray  # int64 [n]
    aids: np.ndarray  # int64 [n], the admission each row belongs to
    sample: np.ndarray  # bool [n]
    device: np.ndarray  # bool [n], device-input rows
    n_tokens: int
    admitted: list[Req] = field(default_factory=list)
    decode_ckpts: list[tuple[int, int, int]] = field(default_factory=list)  # (slot, aid, dst)
    masked: np.ndarray | None = None  # MASKED rows (structured output): their plan indices; None without any
    invalid: np.ndarray | None = None  # MTP, per row: how many drafts its grammar refuses (`fill_masks`), not counted
    # MTP, per masked verify row (`fill_masks`; the verify-mask counters): its masks' popcounts [rows, 1 + k] (−1: a
    # row without them) and classes (`MASK_CLASSES` indices), and the drafts it verifies [rows, k]
    wpop: np.ndarray | None = None
    wcls: np.ndarray | None = None
    wdraft: np.ndarray | None = None
    n_encode: int = 0  # images the plan encodes (ENCODE sections; llm_mini_infer:step_encoder_images)
    # MTP: the drafts the plan's device rows verify and those its SAMPLE rows make (`sched.plan`; −1: k), and whether
    # every row of it is a decode row without logprobs (the step's time is the decode part's: `DraftCounts`)
    v: int = -1
    d: int = -1
    decode_only: bool = False


@dataclass
class FrontendOutputs:
    """One frontend's share of a step: rows in plan order, plus requests finished without a token."""

    rids: np.ndarray
    n_new: np.ndarray
    tokens: np.ndarray
    finish: np.ndarray
    first_rids: np.ndarray
    first_cached: np.ndarray
    first_t_admit_ns: np.ndarray
    first_t_launch_ns: np.ndarray
    first_runner_ns: np.ndarray
    released: list[int] = field(default_factory=list)  # requests whose pixels the frontend may free
    first_t_queued_ns: list[int] = field(default_factory=list)  # the first rows' requests' `Req.t_queued_ns`
    # MTP: requests finishing in these rows, with their verify rows and the tokens those delivered (`verify_rows`)
    spec_rids: list[int] = field(default_factory=list)
    spec_rows: list[int] = field(default_factory=list)
    spec_tokens: list[int] = field(default_factory=list)
    # With a request that asked for logprobs: those of `tokens`, aligned with them (each one's logprob [n], the top ids
    # [n, K] and their logprobs, K the most its requests asked; a request's that asked for none unused)
    logprobs: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None


_E64 = np.zeros(0, np.int64)
_E32 = np.zeros(0, np.int32)


class Admit(IntEnum):
    """`_admit_one`'s outcome: ADMITTED; WAIT, deferred a step for a better hit; FAILED, its grammar could not be
    rebuilt (it is finished, and still the waiting heap's head); NO_ROOM, its first chunk does not fit (the budget,
    encoder rows or pages), which ends the step's admissions."""

    ADMITTED = 0
    WAIT = 1
    FAILED = 2
    NO_ROOM = 3


class GrammarStart(IntEnum):
    """`_grammar_start`'s outcome for an admission: ACTIVE, a matcher over the tokens the request generated (its slot
    is masked); ENDED, its grammar accepted an EOS among them (ignore_eos: admitted constrained, without a mask);
    FAILED, they no longer parse (it fails)."""

    ACTIVE = 0
    ENDED = 1
    FAILED = 2


class Orphan(NamedTuple):
    """Unverified writes that a request left with, or that its readmission will not read (`_orphan_unverified`,
    `_readmit_unverified`), removed at the next `process` (`_remove_unverified`'s arguments): the namespaces, the
    request's own, its key, the admissions that wrote them, the checkpoints they attached, the position each namespace
    is removed from (none: 0), the pages whose marks then go (None: none) and its image spans."""

    nss: set
    ns: Hashable
    key: np.ndarray
    aids: list[int]
    ckpts: list
    lo: dict
    pages: list[int] | None
    spans: Spans | None

# The waiting a step's prefill may add to the requests decoding in it, summed over them (architecture.md, "The
# scheduler")
MIXED_STEP_DELAY_MS = 3500.0

# The precise prefix (architecture.md, "The precise prefix"): the resumes an FP8 checkpoint serves before the next one
# recomputes it with BF16 activations. 1, not 0: the first reuse is free.
REFINE_AFTER = 1
# ... and the fewest tokens a refine recomputes (its FP8 checkpoint minus the deepest precise one below it): a refine
# pays only for a long prefix. Without a minimum, unrelated prompts that share only a chat template's first tokens
# (Gemma 4's sliding-window cache hits short prefixes) would refine those few tokens again and again.
REFINE_MIN = 256
PRECISE_MODES = ("reuse", "bf16", "fp8")
# The sources of the checkpoints a request's deeper checkpoint may demote (`Scheduler._supersede`): a conversation's
# own states. A client's hint and a refine's end stay live
DEMOTABLE = frozenset(CHECKPOINT_SOURCES.index(s) for s in ("prefill", "branch", "decode", "finish", "preempt"))
# ... and those a request continues from, where an earlier request stopped (`Req.continues`)
CONTINUED = frozenset(CHECKPOINT_SOURCES.index(s) for s in ("decode", "finish", "preempt"))
PREFILL, PREEMPT = CHECKPOINT_SOURCES.index("prefill"), CHECKPOINT_SOURCES.index("preempt")
# The longest a request waits for its grammar's build, from `add` (`dev.structured_output.compile_timeout_s`); a build
# that runs on is not stopped: it fills the cache when it ends
COMPILE_TIMEOUT_S = 60.0
# How a refine is stepped (architecture.md, "The precise prefix"): fixed pieces of at most REFINE_CHUNK tokens by
# default (on larger budgets it, not the budget, bounds a piece; a step may carry several pieces)
REFINE_CHUNK = 2048


def derive_mixed_step_row_tokens(*, decode_ms: float, decode_rows: int, prefill_ms: float, prefill_tokens: int,
                                 max_num_batched_tokens: int, max_num_seqs: int) -> tuple[int, str]:
    """The default `dev.scheduler.mixed_step_row_tokens` and its reason, from the runner's measured step costs (its
    memory profile times a full-budget prefill step, and the decode step of a lone request, `decode_rows` = 1, which
    the reason quotes for scale): the decode rows × prefill tokens whose summed delay to the decoding requests is
    `MIXED_STEP_DELAY_MS`, at the prefill step's cost per token (prefill costs about the same per token from a few
    hundred tokens up). 0 (off) when nothing was measured, or when every step of max_num_seqs decode rows may take a
    full budget of prefill."""
    if not (decode_ms > 0 and prefill_ms > 0 and prefill_tokens > 0):
        return 0, "off: the runner measured no step costs (CPU, the fake runner, or dev.runner.cuda.memory_gb set)"
    per_token = prefill_ms / prefill_tokens
    n = round(MIXED_STEP_DELAY_MS / per_token)
    lone = "a lone request's decode step" if decode_rows == 1 else f"a decode step of {decode_rows} rows"
    why = (f"the prefill of a step delays its decode rows by at most {MIXED_STEP_DELAY_MS / 1000:g} s summed over them "
           f"({MIXED_STEP_DELAY_MS / decode_ms:.0f} x {lone}, {decode_ms:.1f} ms), at {per_token:.4f} ms per prefill "
           f"token (a {prefill_tokens}-token prefill step took {prefill_ms:.1f} ms in the memory profile): "
           f"{n:,} rows x tokens, e.g. {min(max_num_batched_tokens, n // 8):,} prefill tokens next to 8 decode rows, "
           f"{min(max_num_batched_tokens, n // 32):,} next to 32")
    if n >= max_num_seqs * max_num_batched_tokens:
        return 0, f"off: every step may take a full budget of prefill next to max_num_seqs rows ({why})"
    return n, why


def derive_precise_cost(precise_step_ms: float, prefill_step_ms: float) -> float:
    """A PRECISE token's cost in FP8 prefill tokens of a step's time (the mixed-step bound): the measured ratio of a
    PRECISE prefill step to an FP8 one, at least 1 (the two timings' noise can put it below), 1 when either is
    unmeasured. The scheduler applies the same floor to a given cost."""
    return max(1.0, precise_step_ms / prefill_step_ms) if precise_step_ms and prefill_step_ms else 1.0


def derive_refine_chunk_tokens(*, max_num_batched_tokens: int, max_num_seqs: int, mtp_k_max: int, grid: int,
                               drafter: str | None = None) -> tuple[int, str]:
    """The default `dev.cache.refine_chunk_tokens` (a refine's fixed piece) and its reason: `REFINE_CHUNK`, at most what
    a step leaves next to a full batch of decode rows, on the checkpoint grid, so that a piece never waits for the
    load to drop; at least one grid. A decode row counts 1 + `mtp_k_max` tokens: with a `drafter` (its name: the
    assistant) its largest k, so that changing k never changes the precise copy (`config.mtp_k_max`); without one, the
    set k (Qwen's MTP layer)."""
    rows = max_num_seqs * (1 + mtp_k_max)
    n = max(min(REFINE_CHUNK, max_num_batched_tokens - rows) // grid * grid, min(grid, max_num_batched_tokens))
    k = f" (1 + {drafter}'s largest k, {mtp_k_max})" if drafter else ""
    why = (f"{n:,}: min({REFINE_CHUNK:,}, max_num_batched_tokens {max_num_batched_tokens:,} - {max_num_seqs} rows x "
           f"{1 + mtp_k_max}{k}) on the {grid}-token grid")
    if n + rows > max_num_batched_tokens:  # at the set k the scheduler checks the piece itself (it refuses one that
        # cannot fit next to its decode rows)
        return n, f"{why}, at least one grid" + (": no room next to a full batch of decode rows of that k" if drafter
                                                 else ": a piece waits for steps with fewer decode rows")
    return n, f"{why}: a piece fits next to a full batch of decode rows" + (
        " at any k, so k never changes the precise copy" if drafter else "")


NEVER_ENDS = 1 << 30  # `end_drafts`: no prefix of the drafts ends the grammar


def end_drafts(m: Any, drafts: list[int]) -> int:
    """How many of `drafts` a sample row whose window starts at matcher `m`'s state must verify before it could deliver
    the grammar's EOS (architecture.md, "Structured output"): 0 when the grammar allows nothing else now, j when it
    does after the first j drafts, NEVER_ENDS when no prefix of them gets there. The target accepts a draft only if it
    is its own masked sample, so only drafts the grammar allows can lead there: llguidance's `try_consume_tokens`
    consumes those, stopping at the first it rejects (never entering its error state) or after the one that stops the
    grammar, so a grammar it leaves stopped stopped at the last draft it took; they are rolled back."""
    if m.is_stopped():
        return 0
    if not drafts:
        return NEVER_ENDS
    n = m.try_consume_tokens(drafts)
    end = n if n and m.is_stopped() else NEVER_ENDS
    if n:
        m.rollback(n)
    return end


def could_end(m: Any, drafts: list[int]) -> bool:
    """Whether a sample row whose window starts at matcher `m`'s state and verifies `drafts` (MTP; [] without) could
    deliver the grammar's EOS (`end_drafts`)."""
    return end_drafts(m, drafts) <= len(drafts)


def advance_matcher(m: Any, toks: list[int], eos: frozenset[int]) -> tuple[bool, bool]:
    """Advance matcher `m` over tokens a request was given: (accepted, ended). The tokens before the first EOS among
    them, then that EOS ends the grammar (under ignore_eos, the only way one reaches here: the tokens after it were
    sampled unconstrained). llguidance takes an EOS before the grammar stops, or in the call whose tokens stopped it,
    never in a later call (its error state), and a JSON document stops at its closing brace: so a stopped grammar
    ends without consuming it."""
    end = next((j for j, t in enumerate(toks) if t in eos), -1)
    if end < 0:
        return bool(m.consume_tokens(toks)), False
    if end and not m.consume_tokens(toks[:end]):
        return False, False
    return bool(m.is_stopped() or m.consume_tokens(toks[end:end + 1])), True


class StepBudget:
    """What the plan under construction may still schedule (architecture.md, "The scheduler"). `hard`:
    max_num_batched_tokens left, which a bidirectional span or a refine's piece may take whole. `soft`: the prefill
    budget left, which a step's decode rows bound to `mixed_step_row_tokens` / n, at least one checkpoint grid
    (`bounded`); while bounded, a PRECISE token weighs `cost` FP8 prefill tokens there (the runner's measured ratio),
    and once in `hard`. `cap`: a harness's cap on PRECISE chunks (0: none); `chunk_cap`: one request's chunk at most
    (`long_prefill_token_threshold`, 0: none). A span or a piece can leave `soft` below 0: nothing more is planned
    then."""

    __slots__ = ("bounded", "cap", "chunk_cap", "cost", "hard", "soft")

    def __init__(self, tokens: int, cost: float = 1.0, cap: int = 0, chunk_cap: int = 0):
        self.hard = self.soft = tokens
        self.bounded = False
        self.cost, self.cap, self.chunk_cap = cost, cap, chunk_cap

    def decode_rows(self, n: int, width: int, row_tokens: int, grid: int) -> None:
        """The step carries n decode rows of `width` tokens: they take their tokens, and with `row_tokens` they bound
        the prefill that holds them back (rows × prefill tokens)."""
        self.soft -= n * width
        self.hard = self.soft
        if row_tokens:
            self.soft = min(self.soft, max(grid, row_tokens // n))
            self.bounded = True

    def tokens(self, precise: bool) -> int:
        """The tokens a chunk may take from what is left: its own count, or a PRECISE chunk's weighed while bounded."""
        if not precise:
            return self.soft
        n = min(self.hard, int(self.soft / self.cost)) if self.bounded else min(self.soft, self.hard)
        return min(n, self.cap) if self.cap else n

    def chunk(self, left: int, precise: bool) -> int:
        """The most a request with `left` tokens to prefill may take in this step's chunk."""
        return min(left, self.tokens(precise), self.chunk_cap or left)

    def room(self, chunk: int, precise: bool) -> int:
        """The tokens a chunk of `chunk` tokens may add (its forced prefix): what the plan and the chunk cap leave."""
        return min(self.hard, self.tokens(precise), self.chunk_cap or self.hard) - chunk

    def charge(self, tokens: int, precise: bool) -> None:
        """A chunk of `tokens` was planned (a PRECISE one weighed while bounded)."""
        self.soft -= math.ceil(tokens * self.cost) if precise and self.bounded else tokens
        self.hard -= tokens


def among(a: np.ndarray, ids: tuple[int, ...]) -> np.ndarray:
    """Which elements of `a` are one of `ids`: `np.isin` for the few ids of a stop set, one comparison per id, cheaper
    than `isin` on a step's few rows."""
    out = a == ids[0] if ids else np.zeros(a.shape, bool)
    for x in ids[1:]:
        out |= a == x
    return out


def page_counts(pages: list[int], n: int) -> np.ndarray | None:
    """`Scheduler.check_accounting`: how often each page of a pool of n pages occurs in `pages` (int64), or None when
    one is not a page of the pool."""
    if pages and (min(pages) < 0 or max(pages) >= n):
        return None
    return np.bincount(np.asarray(pages, np.int64), minlength=n)


@functools.lru_cache(maxsize=4)
def eos_only(words: int, eos: int) -> np.ndarray:
    """The mask of `words` int32 words that allows token `eos` alone (read-only, shared): what `Scheduler.fill_masks`
    writes for the samples nobody reads. Its bit goes in through a uint32 view, as bit 31 of an int32 word does not fit
    an int32 assignment."""
    row = np.zeros(words, np.int32)
    row.view(np.uint32)[eos >> 5] = np.uint32(1 << (eos & 31))
    row.flags.writeable = False
    return row


class Scheduler:
    def __init__(self, *, max_num_seqs: int, max_num_batched_tokens: int, max_model_len: int, page_size: int,
                 num_pages: int, num_state_slots: int, eos_token_ids: tuple[int, ...] | list[int],
                 policy: Literal["fcfs", "priority"] = "fcfs", long_prefill_token_threshold: int = 0,
                 steps_in_flight: int = 2, mtp_k: int = 0, num_frontends: int = 1, seed: int | None = None,
                 num_live_slots: int | None = None, hybrid: bool = True, prefix_caching: bool = True,
                 checkpoint_grid: int = CKPT_GRID, decode_checkpoint_interval: int = 256,
                 demote_superseded: bool = True, mixed_step_row_tokens: int = 0, encoder_rows: int = 0,
                 encoder_budget: int = 0, positions: Positions | None = None, grammars: Any = None,
                 window: int = 0, window_page_size: int = 0, bidirectional_images: bool = False,
                 draft_kv: bool = True, precise: str = "fp8", refine_after: int = REFINE_AFTER,
                 refine_min: int = REFINE_MIN,
                 precise_cost: float = 1.0, refine_chunk_tokens: int = 0, mtp_k_max: int = 0,
                 forced_prefix: bool = False, eos_free_finish: bool = False, batch_invariant: bool = False,
                 logprobs: bool = False, penalties_vocab: int = 0, compile_timeout_s: float = COMPILE_TIMEOUT_S,
                 draft_counts: tuple[int, ...] = (), count_schedule: tuple[int, ...] = (),
                 mixed_draft_counts: bool = False):
        self.max_num_seqs = max_num_seqs
        self.max_tokens = max_num_batched_tokens
        self.max_model_len = max_model_len
        self.ps = page_size
        self.policy = policy
        self.chunk_cap = long_prefill_token_threshold
        self.row_tokens = mixed_step_row_tokens  # decode rows x prefill tokens of a step (0: off)
        self.k = mtp_k
        self.reserve = reserve_positions(steps_in_flight, mtp_k)  # at k, whatever count a step runs (the largest)
        # The draft count per step (`dev.speculative.adaptive`, `sched.draft_counts`): the counts below k a plan may
        # verify and draft (`Capabilities.draft_counts`; empty: k on every step), or a schedule of counts cycled per
        # plan with decode rows (`dev.speculative.count_schedule`: tests)
        self.count_policy = DraftCounts(draft_counts, mtp_k) if mtp_k and draft_counts else None
        self.count_schedule = tuple(int(c) for c in count_schedule) if mtp_k else ()
        if any(not 1 <= c <= mtp_k for c in self.count_schedule):
            raise ValueError(f"count_schedule {list(count_schedule)}: counts between 1 and k ({mtp_k})")
        self._counts = self.count_policy.counts if self.count_policy is not None else self.count_schedule or (mtp_k,)
        # a plan with explicit rows drafts its count too (`Capabilities.mixed_draft_counts`), else k: under continuous
        # admission about half the steps carry prefill rows, and drafting k on them wastes the passes past the count
        self.mixed_draft_counts = mixed_draft_counts
        self._schedule_i = 0
        self.count_raises = 0  # plans whose count exceeded what their rows held: they verified that and drafted k
        # The MTP drafter writes KV of its own (Qwen's MTP layer: `Capabilities.draft_kv`): a hit on a page boundary
        # copies the page of hit − 1, an image chunk's next token needs its embedding (the drafter's input there), and
        # a final chunk's pages cover the positions its drafter's chain writes past its row (`chain`: `_chunk_pages`)
        self.draft_kv = bool(mtp_k) and draft_kv
        self.chain = mtp_k - 1 if self.draft_kv else 0
        self.num_frontends = num_frontends
        self.eos_ids = np.asarray(sorted(set(eos_token_ids)), np.int32)
        self._eos = tuple(self.eos_ids.tolist())  # the same, for `among`
        self.kv = PagePool(num_pages)
        self.states = StatePool(num_state_slots, max_num_seqs if num_live_slots is None else num_live_slots)
        if self.states.n_live < max_num_seqs:
            raise ValueError(f"{self.states.n_live} live state slots for max_num_seqs {max_num_seqs}")
        self.slots = SlotPool(max_num_seqs + 1, "slot")
        # Sliding-window models: window pages, and window checkpoints in the cache; both kinds of page come from one
        # arena, `kv` (architecture.md, "Sliding-window models")
        self.window = window
        self.wps = window_page_size or page_size  # positions of a window page
        self.wkv: PagePool | None = None
        self.windows: WindowStore | None = None
        if window:  # MTP too: its rollback is a truncation of the KV, which windows read like any row
            hybrid = True  # hits land on window checkpoints, as on GDN state checkpoints
            self.wkv = self.kv
            self.windows = WindowStore(self.kv)
        # Penalties and the logit bias (the runner's `Capabilities.penalties`: its vocabulary; 0: none)
        self.pen_vocab = penalties_vocab
        self.builder = PlanBuilder(max_num_seqs, max_num_batched_tokens, page_size, max_model_len, mtp_k, window,
                                   self.wps, checkpoint_grid, vocab=penalties_vocab)
        self._rng = random.Random(seed)
        self.hybrid = hybrid
        self.grid = checkpoint_grid
        grid_unit = self.wps if window else page_size  # window checkpoints sit on window pages
        if checkpoint_grid % grid_unit:
            raise ValueError(f"the checkpoint grid {checkpoint_grid} must be a multiple of the "
                             f"{'window ' if window else ''}page size {grid_unit}")
        self.ckpt_interval = decode_checkpoint_interval if hybrid else 0
        self.demote = demote_superseded  # `_supersede` (dev.cache.demote_superseded)
        self.cache: RadixCache | None = None
        if prefix_caching and (not hybrid or self.states.n_ckpt or window):
            self.cache = RadixCache(page_size, self.kv, self.windows if window else self.states, hybrid=hybrid)
        elif prefix_caching:
            log.info("prefix caching is off: a hybrid model needs checkpoint state slots")
        # The precise prefix (module docstring): which prefill rows are PRECISE. Refines need checkpoint hits (hybrid
        # and sliding-window models); the precise namespace needs the cache
        if precise not in PRECISE_MODES:
            raise ValueError(f"precise {precise!r}: one of {', '.join(PRECISE_MODES)}")
        if precise == "reuse" and (self.cache is None or not hybrid):
            precise = "fp8"  # nothing is reused, or hits are not checkpoints (attention-only models)
        self.precise = precise
        self.pns_on = self.cache is not None and precise != "fp8"  # the precise namespace is in use
        self.refine_after = refine_after
        self.refine_min = refine_min
        # An FP8 checkpoint's node id -> the request refining it, while its PRECISE rows have not reached their target
        # (`_refining_now`; `_refined`)
        self._refining: dict[int, Req] = {}
        # Unverified writes (module docstring, Rollback): the requests that have some, whether a removal took nodes
        # since the last `_mark_nocache`, and the writes a plan under construction orphaned, removed at the next
        # `process` (`Orphan`)
        self._n_unv = 0
        self._removed = False
        self._orphans: list[Orphan] = []
        self.precise_cost = max(1.0, float(precise_cost))  # a PRECISE token in FP8 prefill tokens of a step's time
        self.precise_cap = 0  # a cap on PRECISE chunks' tokens (0: none; harnesses vary the precise copy's chunking)
        # How a refine is stepped (architecture.md, "The precise prefix"): fixed pieces of `piece` tokens from position
        # 0. Unless set, the piece leaves room for decode rows of `mtp_k_max` drafts (`config.mtp_k_max`, at least
        # `mtp_k`), as the engine derives it
        if refine_chunk_tokens and not (0 < refine_chunk_tokens <= max_num_batched_tokens and
                                        refine_chunk_tokens % checkpoint_grid == 0):
            raise ValueError(f"refine_chunk_tokens {refine_chunk_tokens}: a positive multiple of the checkpoint grid "
                             f"({checkpoint_grid}), at most max_num_batched_tokens ({max_num_batched_tokens})")
        room = max_num_batched_tokens - max_num_seqs * (1 + mtp_k)
        if refine_chunk_tokens and refine_chunk_tokens > room:
            # A piece waits whole: one that does not fit next to a full batch of decode rows runs only in steps with
            # fewer, so under steady load a refine's next piece would starve (and a waiting refiner hold the queue)
            raise ValueError(f"refine_chunk_tokens {refine_chunk_tokens}: at most {room} (max_num_batched_tokens "
                             f"{max_num_batched_tokens} - max_num_seqs {max_num_seqs} decode rows x {1 + mtp_k}: a "
                             "piece must fit next to a full batch of decode rows)")
        self.piece = refine_chunk_tokens or derive_refine_chunk_tokens(
            max_num_batched_tokens=max_num_batched_tokens, max_num_seqs=max_num_seqs,
            mtp_k_max=max(mtp_k, mtp_k_max), grid=checkpoint_grid)[0]
        self._plan_precise = False  # the plan being built has PRECISE rows

        s = max_num_seqs + 1
        self.aid = np.zeros(s, np.int64)  # admission ID; 0 = free slot
        self.fe = np.zeros(s, np.int32)
        self.fe_rid = np.zeros(s, np.int64)
        self.mirror = np.zeros(s, np.int64)  # device seq_len as known to the CPU
        self.dev_inflight = np.zeros(s, np.int64)  # device-input rows in flight
        self.last_step = np.zeros(s, np.int64)  # last step that computed tokens for the slot's admission
        self.num_sched = np.zeros(s, np.int64)  # prompt (+ recomputed output) tokens scheduled or cached so far
        self.total = np.zeros(s, np.int64)  # tokens to prefill for this admission
        self.n_out = np.zeros(s, np.int64)  # tokens generated so far (all admissions)
        self.max_out = np.zeros(s, np.int64)
        self.pending = np.zeros(s, np.int64)  # in-flight SAMPLE steps
        self.decoding = np.zeros(s, bool)  # the final prefill chunk has been scheduled
        self.ignore_eos = np.zeros(s, bool)
        self.num_pages = np.zeros(s, np.int64)
        self.num_wpages = np.zeros(s, np.int64)  # window mode: entries of the slot's window-pool page table
        self.wlo = np.zeros(s, np.int64)  # window mode: the first window page the slot still holds
        self.prio = np.zeros(s, np.int64)
        self.admit_order = np.zeros(s, np.int64)
        self.has_first = np.zeros(s, bool)
        self.runner_ns = np.zeros(s, np.int64)
        # MTP, per request (all admissions): its verify rows whose results it got, and the tokens they delivered
        # (llm_mini_infer:request_tokens_per_verify_step, the request log line)
        self.verify_rows = np.zeros(s, np.int64)
        self.verify_tokens = np.zeros(s, np.int64)
        self.out_buf = np.zeros((s, _OUT_W), np.int32)  # the slots' output rings (`Req.out_chunks`)
        self.stop_ids = np.full((s, 4), -1, np.int32)
        self.next_ckpt = np.zeros(s, np.int64)  # the decode checkpoint target: a multiple of the interval
        self.ckpt_pending = np.zeros(s, bool)  # a decode checkpoint of the slot's admission is in flight
        self.overrun = np.zeros(s, bool)  # MTP: the device committed past the request's finish (KV only at release)
        self.mark = np.zeros(s, np.int64)
        self._mark_id = 0
        self.slot_req: list[Req | None] = [None] * s

        self.reqs: dict[tuple[int, int], Req] = {}
        self._waiting: list[tuple[int, int, Req]] = []  # heap of (priority key, arrival, req); lazy deletion
        self.num_waiting = 0
        self.pending_free: list[int] = []
        self.pending_fold: list[tuple[int, int]] = []  # (slot, records) to FOLD before the FREE of the next plan
        # window mode: (slot, first index, n) entries of released window pages to clear in the next plan built (a
        # plan may be abandoned for want of rows; a slot FREEd meanwhile needs none: FREE clears its table)
        self.pending_wclear: list[tuple[int, int, int]] = []
        self._extra: list[tuple[int, int, int]] = []  # (frontend, rid, FinishReason) finished without a token
        self._held: dict[int, Req] = {}  # admission ID -> request preempted while a sample of it was in flight
        # (fe, rid, tokens, finish, first req, (verify rows, verify tokens) of a finishing request under MTP)
        # (frontend, rid, tokens, finish, its first-token request, its verify counts, its logprobs and their width)
        self._held_rows: list[tuple[int, int, list[int], int, Req | None, tuple[int, int] | None,
                                    tuple[tuple, int] | None]] = []
        self._order = _E64
        self._order_dirty = False
        self._next_aid = 1
        self._next_arrival = 0
        self._admissions = 0
        self._n_stop_reqs = 0  # running requests with extra stop token IDs
        self.step_id = 0
        self._done_step = 0  # results of every step up to this one have been processed (`process`)
        self._plan_rows: dict[int, tuple[int, int, int, int]] = {}  # aid -> (SCHED row, pos0, end, slot), plan built
        self._plan_admits: set[int] = set()  # aids admitted by the plan being built
        self._ckpt_inflight: set[int] = set()  # checkpoint slots of decode CKPTs whose step has not completed
        # counters (EngineStats)
        self.num_preemptions = 0
        self.num_aborts = 0
        self.aborts = [0] * len(ABORT_REASONS)  # num_aborts by reason
        self.num_failed = 0
        self.prompt_tokens = 0
        self.generation_tokens = 0
        self.prefix_queries = 0  # prompt tokens of first admissions (vllm:prefix_cache_queries)
        self.prefix_hits = 0  # of which served from the cache (vllm:prefix_cache_hits)
        self.readmit_hits = 0  # tokens served from the cache to readmissions after preemption
        self.ckpt_hits = 0  # admissions resumed from a checkpoint (hybrid and sliding-window models)
        self.ckpt_misses = 0  # admissions that found no usable checkpoint
        self.ckpts_taken: dict[str, int] = dict.fromkeys(CHECKPOINT_SOURCES, 0)
        self.ckpt_resumes = [0] * len(CHECKPOINT_SOURCES)  # admissions resumed from a checkpoint, by its source
        self.ckpts_skipped = 0  # wanted, but no checkpoint slot could be freed
        self.cache_waits = 0  # admissions deferred one step for KV or a checkpoint being written
        # Prompt tokens the tree held for an admission but it recomputed, by cause (`_count_lost`)
        self.lost_tokens: dict[str, int] = dict.fromkeys(LOST_CAUSES, 0)
        # The precise prefix: admissions that recomputed a cached prefix with BF16 activations (refiners), the tokens of
        # PRECISE rows scheduled, and admissions resumed from a precise checkpoint
        self.refines = 0
        self.precise_tokens = 0
        self.precise_hits = 0
        # ... and the steps that carried PRECISE rows
        self.precise_steps = 0
        # MTP (vllm:spec_decode_*): verified decode rows, their drafts, the accepted ones, and acceptance per position
        self.spec_drafts = 0
        self.spec_draft_tokens = 0
        self.spec_accepted = 0
        self.spec_accepted_per_pos = np.zeros(max(1, mtp_k), np.int64)
        self.spec_verified_per_pos = np.zeros(max(1, mtp_k), np.int64)  # verify rows whose window held each position
        self.steps_by_count = np.zeros(mtp_k + 1, np.int64)  # plans of decode rows by the drafts they verified
        self.num_overruns = 0  # MTP rows whose device commit ran past a finish the runner did not know (KV-only finish)
        # ... their tokens past that finish, in spec_accepted but never delivered; and the tokens readmissions' final
        # prefill chunks sampled (a request's next token after a preemption, from no verify row). Σ over requests of
        # (output tokens − 1) = spec_drafts + spec_accepted − spec_trimmed + forced_tokens + eos_free.sum() + resampled
        # (tests/support/harness.py, `check_spec_counts`)
        self.spec_trimmed = 0
        self.resampled = 0
        # Images: the encoder cache, the encoder's budget of image tokens per step, the family's M-RoPE positions, and
        # the pixel releases owed to the frontends: (frontend, rid, step that must complete first)
        self.encoder = EncoderCache(encoder_rows) if encoder_rows else None
        self.enc_budget = encoder_budget or encoder_rows
        self._enc_left = 0
        self._enc_short = False  # an image within the budget found the cache's rows referenced (`_image_window`)
        self.positions = positions
        self.bidi = bidirectional_images  # image spans attend bidirectionally: keep each whole
        self._budget = StepBudget(0)  # the plan being built's (`_schedule`)
        self._released: list[tuple[int, int, int]] = []
        self.encoder_images = 0  # images encoded (encoder-cache misses)
        self.page_waits = 0  # admissions and later prefill chunks that waited for KV pages
                             # (llm_mini_infer:page_waits_total)
        self.encoder_tokens = 0  # embeddings computed
        self.encoder_hits = 0  # image spans prefilled from embeddings already in the cache
        self._building = False  # inside schedule(): encoder references dropped now are released after the plan
        self._unrefs: list[Entry] = []
        # Structured output: the engine's grammars (grammar.engine.Grammars; None: off), per slot whether its SAMPLE
        # rows are MASKED and the drafts its last SAMPLE step made (what its next decode row verifies)
        self.grammars = grammars
        self.masked = np.zeros(s, bool)
        self.drafts = np.zeros((s, max(1, mtp_k)), np.int32)
        # ... and how many it made (`StepPlan.draft_n`): set when that row is planned, which is when its count is
        # known; a slot's next decode row verifies at most that many (0 since its admission)
        self.held = np.zeros(s, np.int64)
        self.n_masked = 0  # slots with masked set
        # the drafts the slot's next sample row must verify before it could deliver its grammar's EOS (`end_drafts`)
        self.end_at = np.full(s, NEVER_ENDS, np.int64)
        self.end_waits = 0  # decode rows skipped behind a row in flight that could end its request's grammar
        self.dropped_samples = 0  # sample rows whose request had left before their result (a stop, abort, preemption)
        self.structured_requests = 0
        # The forced prefix and the EOS-free finish (architecture.md, "Grammar-aware heads"). A final chunk's forced
        # tokens go into its row once the plan's CKPT records are placed (`_commit_tails`): (row, slot, request, how
        # many) of the plan being built
        self.forced_prefix = forced_prefix and grammars is not None
        self.eos_free_finish = eos_free_finish and grammars is not None
        # Logprobs (the runner's `Capabilities.logprobs`): per slot the top entries its request asked for (-1: none),
        # whose SAMPLE rows are LOGPROBS
        self.logprobs = logprobs
        self.lp = np.full(s, -1, np.int32)
        self.n_lp = 0  # slots with lp set
        self._tails: list[tuple[int, int, Req, int]] = []
        # Batch invariance (architecture.md, "Batch invariance"): a GDN model's chunks are cut on the grid from where
        # each prompt's computation starts (`_chunk_end`), and every model's final chunk takes its forced prefix whole
        # (`_n_forced`)
        self.aligned = batch_invariant and self.hybrid and not window
        self.forced_whole = batch_invariant and self.forced_prefix
        self.image_cuts = 0  # aligned chunks cut off the grid before an image the step cannot have (`_image_chunk`)
        self._check_progress(long_prefill_token_threshold)
        # The EOS a stopped grammar allows, when it is one id (guidance's: the tokenizer's eos_token) that ends a
        # request: the EOS-free finish appends it; −1: off (a grammar with several EOS ids leaves the pick to the model)
        geos = sorted(grammars.eos) if grammars is not None else []
        self._grammar_eos = geos[0] if len(geos) == 1 and geos[0] in set(self.eos_ids.tolist()) else -1
        self.unrep = np.zeros(s, bool)  # its request has output tokens its frontend has not received (`Req.unrep`)
        self._n_unrep = 0
        self._last_inf: Inflight | None = None  # the last plan built (in flight while `process` reads the one before)
        self.forced_requests = 0  # requests whose final prefill chunk took forced tokens
        self.forced_tokens = 0  # ... and those tokens
        self.forced_declined = 0  # final chunks that sampled theirs: a CKPT record off the grid at the prompt's end
        self.eos_free = np.zeros(2, np.int64)  # EOS-free finishes: no row of the request in flight, its row in flight
        self.eos_free_gated = 0  # requests left to the EOS the step in flight samples (the gate: nothing else stays)
        # The verify-mask counters (`_rejections`): MTP verify rows' first rejected draft by the class of the mask
        # its target sample was drawn under (MASK_CLASSES), and the popcounts of the masks verification read
        self.first_rejections = np.zeros(len(MASK_CLASSES), np.int64)
        self.window_masks = np.zeros(len(POPCOUNT_BUCKETS) + 1, np.int64)  # counts per bucket (the last one +Inf)
        self.window_masks_sum = 0
        # Backpressure (engine.process): frontends lagging behind their outputs; their waiting requests are not admitted
        self.paused_frontends: set[int] = set()
        self.grammar_rejects = 0  # requests whose grammar the engine could not compile, or not in time
        # Requests whose grammar is being built (COMPILING), in arrival order; how long one may wait for it, from `add`;
        # and whether one entered or left the set since the engine loop last looked (it then reports the waiting count)
        self._compiling: dict[Req, None] = {}
        self.compile_timeout_ns = int(compile_timeout_s * 1e9)
        self.compiling_moved = False

    # -- requests in and out -------------------------------------------------------------------------------

    def add(self, msg: AddRequest) -> None:
        """Queue a new request, or reject it: an ERROR finish to its frontend. An exception while it is checked or built
        (a field no check foresaw, say) rejects it the same way and leaves nothing of it behind: it fails that request,
        not the engine loop. A request whose grammar is being built waits COMPILING (module docstring, A request's
        state) until `collect` finds it built."""
        try:
            req = self._new_request(msg)
        except Exception as exc:  # its message can quote the request: the types and frames only
            log.error("rejecting request %s: admission failed (exception messages left out)\n%s",
                      msg.request_id or f"{msg.frontend}:{msg.rid}", traceback_without_messages(exc).rstrip())
            req = None
        if req is None:
            self._extra.append((msg.frontend, msg.rid, FinishReason.ERROR))
            if msg.images:
                self._released.append((msg.frontend, msg.rid, self.step_id))
            return
        if req.compiled is not None:
            self.structured_requests += 1
        self._next_arrival += 1
        self.reqs[req.key] = req
        if req.future is not None:
            req.status = COMPILING
            self._compiling[req] = None
            self.compiling_moved = True
            return
        self._to_waiting(req)

    def _check_penalties(self, msg: AddRequest) -> str | None:
        """Why the runner cannot take a request's penalties or logit bias (None: it can)."""
        ids = np.asarray(msg.bias_ids, np.int64)
        if not self.pen_vocab:
            return "penalties or a logit bias, but this runner applies none"
        if len(ids) > MAX_LOGIT_BIAS or len(msg.bias_values) != len(ids) or \
                (len(ids) and (ids[0] < 0 or ids[-1] >= self.pen_vocab or (np.diff(ids) <= 0).any())):
            return f"a logit bias of {len(ids)} entries that are not distinct ascending ids below {self.pen_vocab}"
        return None

    def _new_request(self, msg: AddRequest) -> Req | None:
        """`add`'s checks and its request, built; None (the reason logged) when a check rejects it."""
        key = (msg.frontend, msg.rid)
        n = len(msg.prompt)
        reason = None
        if key in self.reqs:
            reason = "duplicate request id"
        elif n == 0:
            reason = "empty prompt"
        elif n >= self.max_model_len:
            reason = f"prompt of {n} tokens >= max_model_len {self.max_model_len}"
        elif msg.max_tokens < 1:
            reason = "max_tokens < 1"
        elif -(-(n + self.reserve) // self.ps) + self._window_peak() > self.kv.num_pages - 1:
            reason = f"prompt of {n} tokens does not fit in the KV pool"
        elif msg.logprobs >= 0 and not self.logprobs:
            reason = "logprobs, but this runner computes none"
        elif msg.logprobs > MAX_LOGPROBS:
            reason = f"{msg.logprobs} top logprobs (at most {MAX_LOGPROBS})"
        elif msg.images:
            reason = self._check_images(msg, n)
        if reason is None and ((msg.repetition_penalty, msg.presence_penalty, msg.frequency_penalty) != (1.0, 0.0, 0.0)
                               or len(msg.bias_ids)):
            reason = self._check_penalties(msg)
        built = fut = None
        if reason is None and msg.grammar:
            if self.grammars is None:
                reason = "structured output is off in this engine"
            else:  # done now when the grammar is cached or no compile threads run (`Grammars.submit`)
                fut = self.grammars.submit(msg.grammar, self._wants_prefix(msg.stop_strings))
                if fut.done():
                    built, fut = fut.result(), None
                    if isinstance(built, GrammarError):  # its message quotes the schema: request content, not logged
                        self.grammar_rejects += 1
                        reason = GRAMMAR_FAILED
        if reason is not None:
            log.warning("rejecting request %s: %s", msg.request_id or f"{key[0]}:{key[1]}", reason)
            return None
        seed = msg.seed if msg.seed is not None else self._rng.getrandbits(63)
        if msg.seed is None:
            log.debug("request %s: drew seed %d", msg.request_id or f"{key[0]}:{key[1]}", seed)
        req = Req(msg, seed, self._next_arrival, self.max_model_len)
        if built is not None:
            self._set_grammar(req, *built)
        req.future = fut
        if self.bidi and req.hints and len(req.sp_start):  # a checkpoint never lies inside a span
            req.hints = [h for h in req.hints if not self._in_span(req, h)]
        if msg.images and self.positions is not None:
            from ..vision.family import ImageSize, Span

            spans = [Span(im.offset, ImageSize(im.height, im.width, (im.grid_t, im.grid_h, im.grid_w), im.tokens))
                     for im in msg.images]
            rope, req.rope_off = self.positions(n, spans)
            if rope is not None:
                req.rope = (rope - np.arange(n, dtype=np.int32)[None, :]).astype(np.int32)
        return req

    def _wants_prefix(self, stop_strings: Any) -> bool:
        """Whether a request's grammar gives it a forced prefix: not when its frontend ends it on stop strings."""
        return self.forced_prefix and not stop_strings

    def _set_grammar(self, req: Req, compiled: Any, forced: tuple[int, ...] | None) -> None:
        """The request's compiled grammar and, when it wants one, its forced prefix: the build's (`forced`), or else
        `Grammars.forced_prefix`'s (computed here without compile threads, a cache hit with them), cut before a stop
        token."""
        req.compiled = compiled
        if self._wants_prefix(req.stop_strings):
            f = forced if forced is not None else self.grammars.forced_prefix(req.grammar)
            stop = {*req.stop_ids.tolist(), *(() if req.ignore_eos else self.eos_ids.tolist())}
            req.forced = f[:next((j for j, t in enumerate(f) if t in stop), len(f))]  # cut before a stop token

    def _window_peak(self) -> int:
        """Window pages one request may hold at once, in the same pool as its full pages (0 without a window)."""
        if not self.window:
            return 0
        return window_pages_for_one_request(self.window, self.max_tokens, self.wps, self.reserve)

    def _check_images(self, msg: AddRequest, n: int) -> str | None:
        """Spans in order, inside the prompt, separated by text, and each image encodable on its own. The text between
        spans (the template's end and start tokens) is what guarantees progress: a chunk can always finish one image and
        drop its encoder reference before it needs the next one, also with a drafter that writes KV and reads one token
        ahead (`_image_window`)."""
        if self.encoder is None:
            return "images, but this server runs no image encoder"
        end = -1
        for im in msg.images:
            if im.tokens < 1 or im.offset <= end or im.offset + im.tokens > n:
                return (f"image span [{im.offset}, {im.offset + im.tokens}) empty, outside the prompt, out of order or "
                        "adjacent to the previous one")
            if im.tokens > self.enc_budget or EncoderCache.blocks_for(im.tokens) > self.encoder.n_blocks:
                return (f"an image of {im.tokens} tokens exceeds the encoder budget ({self.enc_budget}) or cache "
                        f"({self.encoder.rows} rows)")
            if self.bidi and im.tokens > self.max_tokens:
                return (f"an image of {im.tokens} tokens exceeds max_num_batched_tokens ({self.max_tokens}): its "
                        "bidirectional span is prefilled in one step")
            end = im.offset + im.tokens
        return None

    def abort(self, frontend: int, rids: list[int], reason: int = 0) -> None:
        """The frontend's Abort of `rids`, for `reason` (an ABORT_REASONS index)."""
        for rid in rids:
            req = self.reqs.get((frontend, rid))
            if req is None:
                continue  # already finished: its outputs crossed the Abort
            self.num_aborts += 1
            self.aborts[reason] += 1
            if req.status == RUNNING:
                self._release(req, keep=req.unv < 0)  # unverified KV is not kept: nothing will verify it
            self._settle_unverified(req, Unverified.DROP)
            self._to_finished(req)
        if self._removed:
            self._mark_nocache()

    def collect(self) -> bool:
        """The grammar builds that ended (`Grammars.settle`, also when no request waits for them), and the COMPILING
        requests they decide: built, a request waits (its arrival number keeps its place among the waiting); failed or
        past `compile_timeout_s`, it finishes with an ERROR. Called once an engine turn, and in the boundary wait. Never
        raises: a request's failure is that request's. Whether a request left COMPILING."""
        g = self.grammars
        if g is None:
            return False
        settled = g.settle()
        compiling = self._compiling
        if not compiling:
            return False
        moved = False
        if settled:
            for req in [r for r in compiling if r.future.done()]:
                self._built(req)
                moved = True
        now = time.perf_counter_ns()
        while compiling:
            req = next(iter(compiling))  # the oldest: the first past the limit
            if now - req.t_queued_ns < self.compile_timeout_ns:
                break
            self.grammar_rejects += 1
            log.warning("rejecting request %s: its grammar did not compile within "
                        "dev.structured_output.compile_timeout_s (%g s)", req.label, self.compile_timeout_ns / 1e9)
            self._reject(req)
            moved = True
        return moved

    def _built(self, req: Req) -> None:
        """`collect`: the COMPILING request's build is done; its outcome read from its own future (the cache may have
        evicted it)."""
        fut, req.future = req.future, None
        try:
            built = fut.result()
            if isinstance(built, GrammarError):  # its message quotes the schema: request content, not logged
                self.grammar_rejects += 1
                log.warning("rejecting request %s: %s", req.label, GRAMMAR_FAILED)
                self._reject(req)
                return
            self._set_grammar(req, *built)
        except Exception as exc:  # as `add`'s: its message can quote the request
            log.error("rejecting request %s: admission failed (exception messages left out)\n%s", req.label,
                      traceback_without_messages(exc).rstrip())
            self._reject(req)
            return
        self.structured_requests += 1
        self._to_waiting(req)

    def _reject(self, req: Req) -> None:
        """A COMPILING request fails: an ERROR finish to its frontend (`_to_finished` releases its images)."""
        self._extra.append((req.frontend, req.rid, FinishReason.ERROR))
        self._to_finished(req)

    @property
    def num_compiling(self) -> int:
        """Requests waiting for their grammar's build (COMPILING)."""
        return len(self._compiling)

    def abort_all(self, reason: FinishReason = FinishReason.ABORT) -> None:
        """Engine-side abort of everything (shutdown): the frontends get a finish row for each request."""
        for req in list(self.reqs.values()):
            self._extra.append((req.frontend, req.rid, reason))
            self.abort(req.frontend, [req.rid])

    @property
    def num_running(self) -> int:
        return self.max_num_seqs - self.slots.num_free

    def has_work(self) -> bool:
        return self.num_waiting > 0 or self.num_running > 0

    def has_unsent(self) -> bool:
        return bool(self._extra) or bool(self._released)

    def paused_only(self) -> bool:
        """Whether the only work is waiting requests of paused frontends (backpressure): nothing runs or is unsent,
        and every waiting request waits for its frontend, none for a held sample or a better hit (`_admit`). Only an
        `Ack` can change that, and the engine's poll brings it (`EngineLoop.run` blocks)."""
        paused = self.paused_frontends
        if not paused or self.num_running or not self.num_waiting or self.has_unsent():
            return False
        return all(r.frontend in paused and not r.held for r in self._waiting_reqs())

    def _release_pixels(self, req: Req) -> None:
        """The request left: its frontend may free its pixels once every step planned so far has completed."""
        if req.images:
            self._released.append((req.frontend, req.rid, self.step_id))

    # -- the step plan ---------------------------------------------------------------------------------------

    def schedule(self) -> tuple[StepPlan, Inflight] | None:
        b = self.builder
        b.begin()
        self._enc_left = self.enc_budget
        self._building = True
        try:
            return self._schedule()
        finally:
            self._building = False
            self._flush_unrefs()

    def _schedule(self) -> tuple[StepPlan, Inflight] | None:
        b = self.builder
        enc0 = self.encoder_images
        self._mark_id += 1  # mark[slot] == _mark_id: the slot has a row in the plan being built
        self._plan_rows = {}
        self._plan_admits.clear()
        budget = self._budget = StepBudget(self.max_tokens, self.precise_cost, self.precise_cap, self.chunk_cap)
        self._plan_precise = False
        preempted = stop_running = False
        hold: Req | None = None  # its chunk waits for pages behind a request short of encoder rows
        n_pre = self.num_preemptions
        admitted: list[Req] = []
        decode_ckpts: list[tuple[int, int, int]] = []
        order = self._running_order()
        width = 1 + self.k  # tokens of a decode row
        v = d = c = self.k  # the drafts the plan's decode rows verify, those its SAMPLE rows make, the count chosen
        if len(order):
            dec = order[self.decoding[order]]
            if len(dec):
                # in-flight samples deliver at most 1 token each, decode rows (device input) 1 + k (at most)
                dec = dec[self.n_out[dec] + self.pending[dec] + self.k * self.dev_inflight[dec] < self.max_out[dec]]
                if self.n_masked and len(dec) and self._end_wait(order, dec):
                    dec = dec[:0]
                fit = budget.soft // (1 + min(self._counts))  # the rows the budget holds at the narrowest count
                if len(dec) and fit and (self.count_policy is not None or self.count_schedule):
                    # one count for all the rows, at most what each holds; a count above that verifies what they hold
                    # and drafts k, which every later count can verify (a raise). Decided on every row, then cut for
                    # the width it gives: the rows the cut (or a preemption) keeps hold at least as many
                    c = self._choose_count(min(len(dec), fit), budget.soft)
                    held = int(self.held[dec].min())
                    v, d = (c, c) if c <= held else (held, self.k)
                    if c > held:
                        self.count_raises += 1
                        if not self.count_schedule:
                            self.count_policy.raised()
                    width = 1 + v
                if len(dec) * width > budget.soft:
                    dec = dec[:budget.soft // width]
            if len(dec):
                need = (self.mirror[dec] + (self.reserve + self.ps - 1)) // self.ps - self.num_pages[dec]
                needw = None
                if self.window:  # each row's first query is at mirror + dev_inflight or later (MTP): slide its window
                    self._wslide(dec, self.mirror[dec] + self.dev_inflight[dec])
                    needw = (self.mirror[dec] + (self.reserve + self.wps - 1)) // self.wps - self.num_wpages[dec]
                    short = np.flatnonzero((need > 0) | (needw > 0))
                    ok = not short.size or self._ensure_pages(int(need[need > 0].sum()), int(needw[needw > 0].sum()))
                else:
                    short = np.flatnonzero(need > 0)
                    ok = not short.size or self._ensure_pages(int(need[short].sum()))
                if not ok:
                    dec, stop_running = self._decode_with_preemption(order, dec, need, needw)
                    preempted = True
                elif short.size:
                    self._grow_rows(dec[short], need[short], None if needw is None else needw[short])
                if len(dec):
                    row0 = b.n_sched
                    b.add_device_rows(dec, self.masked[dec] if self.n_masked else None,
                                      self.lp[dec] >= 0 if self.n_lp else None, v)
                    self.pending[dec] += 1
                    self.mark[dec] = self._mark_id
                    budget.decode_rows(len(dec), 1 + v, self.row_tokens, self.grid)
                    if self.k:
                        self.steps_by_count[v] += 1
                        if self.count_policy is not None or self.count_schedule:
                            log.debug("step %d: %d decode rows verify %d drafts (count %d)", self.step_id + 1,
                                      len(dec), v, c)
                    if self.ckpt_interval and self.cache is not None:
                        self._decode_checkpoints(dec, row0, decode_ckpts)
                    self.dev_inflight[dec] += 1
            if budget.soft > 0 and not stop_running:
                p, blocked, hold = self._schedule_running_prefills(order)
                preempted |= p | blocked
        if not preempted:
            self._admit(admitted)
        if self.cache is not None:
            self.cache.unpin_checkpoints()  # the plan is built: its STATE_COPY sources may go
        if self._tails:
            self._commit_tails()
        if b.n_sched == 0:
            if hold is not None and self.num_preemptions == n_pre and self._done_step == self.step_id:
                self._drop_images(hold)  # else every later plan is this one (architecture.md, "The scheduler")
            return None
        for slot, n in self.pending_fold:  # before the FREEs: a donated live state is folded first (MTP)
            b.add_fold(slot, n)
        self.pending_fold.clear()
        for slot in self.pending_free:
            b.add_free(slot)
        self.pending_free.clear()
        for slot, first, n in self.pending_wclear:
            b.add_wpages(slot, first, [0] * n)
        self.pending_wclear.clear()
        if self._plan_precise:
            self.precise_steps += 1
        if b.n_sched > b.n_device and not self.mixed_draft_counts:
            d = self.k  # explicit rows: a step on the runners' token-bucket graphs, which draft k (`sched.plan`)
        if self.k:
            b.draft_n = d if d < self.k else 0
        self.step_id += 1
        plan = b.finish(self.step_id)
        slots = plan.sched_slot.astype(np.int64)
        self.last_step[slots] = self.step_id
        flags = plan.sched_flags
        sample = (flags & SAMPLE) != 0
        if self.k:
            self.held[slots[sample]] = d
        inf = Inflight(step_id=self.step_id, slots=slots, aids=self.aid[slots], sample=sample,
                       device=plan.sched_tok_off < 0, n_tokens=plan.n_query_tokens, admitted=admitted,
                       decode_ckpts=decode_ckpts, n_encode=self.encoder_images - enc0, v=v, d=d,
                       decode_only=b.n_sched == b.n_device and not (flags & LOGPROBS).any())
        if self.n_masked:
            masked = np.flatnonzero(flags & MASKED)
            if len(masked):
                inf.masked = masked
        # Prefill tokens computed: the explicit rows (prompt chunks, recompute and refines included; a final chunk's
        # forced tokens too); a decode row's k drafts are verify work (spec_draft_tokens)
        self.prompt_tokens += b.n_tokens
        self._last_inf = inf  # the EOS-free finish's gate reads the step in flight (`_eos_free`)
        return plan, inf

    def _choose_count(self, rows: int, tokens: int) -> int:
        """The draft count of a plan with `rows` decode rows and `tokens` of budget for them: the dev schedule's next,
        else the policy's."""
        if self.count_schedule:
            c = self.count_schedule[self._schedule_i % len(self.count_schedule)]
            self._schedule_i += 1
            return c
        return self.count_policy.choose(rows, tokens)

    def _end_wait(self, order: np.ndarray, dec: np.ndarray) -> bool:
        """Whether this plan skips its decode rows `dec` (architecture.md, "Structured output"): each has a row in
        flight (in the last plan built) that could deliver its grammar's EOS with the drafts that plan verifies, and the
        plan would carry nothing else (no waiting request, no prefill chunk; the decoding requests outside `dec` wait
        for a length finish)."""
        v = self._v(self._last_inf) if self._last_inf is not None else self.k
        if self.num_waiting or not ((self.end_at[dec] <= v) & (self.pending[dec] > 0)).all() or \
                not self.decoding[order].all():
            return False
        self.end_waits += len(dec)
        return True

    @staticmethod
    def on_launched(inf: Inflight, t_ns: int) -> None:
        """The plan was launched: stamp first launches (TTFT stage timestamps)."""
        for req in inf.admitted:
            if not req.t_launch_ns:
                req.t_launch_ns = t_ns

    def _decode_checkpoints(self, dec: np.ndarray, row0: int, out: list[tuple[int, int, int]]) -> None:
        """CKPT −1 on decode rows whose step commits up to the next multiple of `decode_checkpoint_interval` or past
        it: a row's committed position after the step is at least `mirror + dev_inflight + 1` (exactly that without
        MTP), and a request has one decode checkpoint in flight at a time; it lands where the step's commit ends. A
        running request keeps only its latest (architecture.md, "Checkpoints"): taking a new one releases the previous
        one first, so the new one reuses its slot instead of evicting another prefix's checkpoint."""
        lo = self.mirror[dec] + self.dev_inflight[dec] + 1
        due = np.flatnonzero((lo >= self.next_ckpt[dec]) & ~self.ckpt_pending[dec])
        iv = self.ckpt_interval
        for j in due.tolist():
            slot = int(dec[j])
            req = self.slot_req[slot]
            if req.nocache:
                continue
            if self.window:  # a window, made where the step's commit ends once its result is in (no plan record)
                self.ckpt_pending[slot] = True
                out.append((slot, int(self.aid[slot]), 0))
                continue
            if req.decode_ckpt is not None:
                self.cache.drop_checkpoint(*req.decode_ckpt)  # if that node still holds it
                req.decode_ckpt = None
            dst = self.cache.alloc_checkpoint(self.step_id + 1)
            if not dst:
                self.ckpts_skipped += 1
                self.next_ckpt[slot] = (int(lo[j]) // iv + 1) * iv
                continue
            self.builder.add_ckpt(row0 + j, CKPT_END_OF_STEP, dst)
            self._ckpt_inflight.add(dst)
            self.ckpt_pending[slot] = True
            out.append((slot, int(self.aid[slot]), dst))

    def _schedule_running_prefills(self, order: np.ndarray) -> tuple[bool, bool, Req | None]:
        """Next chunks of running requests whose prompt is not fully scheduled. Returns whether a request was
        preempted, whether one was blocked for lack of pages, and that one if a request ahead of it got no chunk for
        lack of encoder rows (architecture.md, "The scheduler")."""
        preempted = blocked = starved = False
        for slot in order[~self.decoding[order]].tolist():
            if self._budget.soft <= 0:
                break
            if self.aid[slot] == 0:
                continue
            req = self.slot_req[slot]
            start = int(self.num_sched[slot])
            self._enc_short = False
            chunk = self._next_chunk(req, start, int(self.total[slot]))
            if chunk is None:
                starved |= self._enc_short  # its next image found the encoder cache's rows referenced
                continue  # nothing fits what the step leaves (a PRECISE token, a piece, its next image)
            end, enc_new, nf, precise = chunk
            if self.window:  # the chunk's first query reads from start − W + 1 on
                self._wrelease(slot, start)
            need, needw, nf = self._chunk_pages(end, nf, int(self.num_pages[slot]), int(self.num_wpages[slot]),
                                                final=end == int(self.total[slot]))
            if not self._ensure_pages(need, needw):
                # Decode rows were scheduled first, so only lower-priority requests without a row in this plan
                # can give pages back. If that is not enough, the chunk waits (keeping its pages).
                for victim in order[::-1].tolist():
                    if self._ensure_pages(need, needw) or victim == slot:
                        break
                    if self.aid[victim] and self.mark[victim] != self._mark_id:
                        self._preempt(victim)
                        preempted = True
                if not self._ensure_pages(need, needw):
                    self.page_waits += 1
                    blocked = True
                    if enc_new is not None:
                        self.encoder.rollback()  # its images' reservation (`_image_window`)
                    if starved:  # a request ahead waits for encoder rows its references may hold
                        return preempted, blocked, req
                    break
            self._grow(slot, need)
            self._wgrow(slot, needw)
            self.mark[slot] = self._mark_id
            self._cache_chunk(req, slot, self._chunk_row(req, slot, start, end, enc_new, nf, precise), start, end)
        return preempted, blocked, None

    def _next_chunk(self, req: Req, start: int, total: int) -> tuple[int, list[int] | None, int, bool] | None:
        """The next chunk [start, end) of a prefill that the plan can take, first chunks and later ones alike: (end, the
        images it encodes, None without images; the forced tokens a final chunk takes; whether its rows are PRECISE, the
        precise prefix's), or None when nothing fits."""
        precise = start < req.refine
        if precise and req.refine < total:  # a reuse refine's piece (bf16's cold prompt: refine == total)
            avail = self._piece_tokens(req, start)
        else:
            avail = self._budget.chunk(total - start, precise)
        if avail <= 0:
            return None
        end, enc_new = self._chunk_to(req, start, total, avail)
        if end <= start:
            return None
        nf = self._n_forced(req, end - start, precise) if end == total else 0
        if nf < 0:  # its forced prefix whole does not fit next to the prompt's end: a shorter chunk now, the end later
            if enc_new is not None:
                self.encoder.rollback()  # the images' reservation (`_image_window`), made again for the shorter chunk
            end, enc_new = self._chunk_to(req, start, total, min(avail, total - start - 1))
            if end <= start:
                return None
            nf = 0
        return end, enc_new, nf, precise

    def _chunk_to(self, req: Req, start: int, total: int, avail: int) -> tuple[int, list[int] | None]:
        """(end, the images it encodes or None without images) of the next chunk given `avail` tokens; end == start
        when nothing fits."""
        if len(req.sp_start):
            return self._image_chunk(req, start, total, avail) if avail > 0 else (start, [])
        return (self._chunk_end(req, start, total, avail) if avail > 0 else start), None

    def _chunk_pages(self, end: int, nf: int, held: int, wheld: int, copy: int = 0, wcopy: int = 0,
                     final: bool = False) -> tuple[int, int, int]:
        """The pages a chunk ending at `end` needs beyond those its slot holds (`held` full and `wheld` window pages, of
        which an admission copies `copy` and `wcopy`: KV_COPY): (full pages, window pages, nf), with its `nf` forced
        tokens only if their pages are free at once (best-effort: nothing waits or is preempted for them), unless
        `forced_whole`. A `final` chunk's row samples, and under MTP a drafter with KV of its own writes its chain's
        k − 1 positions past the row's last, after the forced tokens, and reads them back (`plan`, MTP drafting): its
        pages cover them, or the chain would write and read the null page, where other rows (a CUDA graph's padding)
        write too."""
        def need(n: int) -> tuple[int, int]:
            return copy + -(-n // self.ps) - held, wcopy + -(-n // self.wps) - wheld if self.window else 0

        chain = self.chain if final else 0
        pages, wpages = need(end + nf + chain)
        if nf and not self.forced_whole and not self._ensure_pages(pages, wpages):
            nf = 0
            pages, wpages = need(end + chain)
        return pages, wpages, nf

    def _chunk_row(self, req: Req, slot: int, start: int, end: int, enc_new: list[int] | None, nf: int,
                   precise: bool) -> int:
        """Plan the chunk [start, end) of the request in `slot`, its pages secured: its images' encodes, its row, its
        spans, the slot's counters and the step's budget, the budget of its `nf` forced tokens included, which
        `_commit_tails` adds to the row. Returns its row."""
        b, total = self.builder, int(self.total[slot])
        row = b.n_sched
        if enc_new is not None:
            self._image_commit(req, start, end, total, enc_new)
        toks = req.tokens[start:end]
        final = end == total
        b.add_explicit_row(slot, toks, start, (STATE_ZERO if start == 0 else 0) | (PRECISE if precise else 0) |
                           ((SAMPLE | (MASKED if self.masked[slot] else 0) | (LOGPROBS if self.lp[slot] >= 0 else 0))
                            if final else 0),
                           -1 if final else int(req.tokens[end]), *self._image_inputs(req, start, end, total))
        if self.bidi:
            self._add_spans(row, req, start, end)
        self.num_sched[slot] = end
        self.mirror[slot] += end - start
        self._budget.charge(end - start + nf, precise)
        if nf:
            self._tails.append((row, slot, req, nf))
        if final:
            self.decoding[slot] = True
            self.pending[slot] += 1
            if self.masked[slot] and not req.ignore_eos:  # its first token: an EOS if the grammar is stopped
                self.end_at[slot] = 0 if req.matcher.is_stopped() else NEVER_ENDS
        return row

    def _decode_with_preemption(self, order: np.ndarray, dec: np.ndarray, need: np.ndarray,
                                needw: np.ndarray | None = None) -> tuple[np.ndarray, bool]:
        """The rare path: not enough free pages for every decode row, even after evicting the cache. Walk the rows in
        priority order and preempt from the lowest-priority end until each fits. Returns the rows to schedule and
        whether a row preempted itself (then no further running request is scheduled). `needw`: window-pool pages per
        row (sliding-window models)."""
        selected: list[int] = []
        stop_running = False
        for i, slot in enumerate(dec.tolist()):
            if self.aid[slot] == 0:
                continue
            n = max(0, int(need[i]))
            nw = max(0, int(needw[i])) if needw is not None else 0
            while (n > 0 or nw > 0) and not self._ensure_pages(n, nw):
                victim = self._pick_victim(order)
                self._preempt(victim)
                if victim == slot:
                    break
            if self.aid[slot] == 0:
                stop_running = True
                break
            self._grow(slot, n)
            self._wgrow(slot, nw)
            selected.append(slot)
        return np.asarray(selected, np.int64), stop_running

    def _ensure_pages(self, n: int, nw: int = 0) -> bool:
        """At least n free pages, evicting cached pages if needed, then freeing duplicates (`_dedup`). Window mode: n
        full and nw window pages from the one arena, freed in a fixed order (architecture.md, "Sliding-window models").
        Freeing duplicates can swap the tree's pages on a locked path (`_readopt`, `RadixCache.share_tails`), so no
        caller holds tree page ids across this call: admission reads its hit's pages only after securing its own."""
        if self.window:
            need = max(0, n) + max(0, nw)
            kv, cache = self.kv, self.cache
            if kv.num_free < need and cache is not None:
                step = self.step_id + 1

                def enough() -> bool:
                    return kv.num_free >= need

                cache.evict_checkpoints(1 << 62, step, spare_locked=True, enough=enough)
                if kv.num_free < need:
                    cache.evict_pages(need - kv.num_free)
                self._dedup(need)
                if kv.num_free < need:  # the last resort before preemption: windows on paths running requests hold
                    cache.evict_checkpoints(1 << 62, step, enough=enough)
            return kv.num_free >= need
        if self.kv.num_free < n and self.cache is not None:
            self.cache.evict_pages(n - self.kv.num_free)
            self._dedup(n)
        return self.kv.num_free >= n

    def _dedup(self, need: int) -> None:
        """What eviction left of page pressure (`_ensure_pages`): free the pages the locked paths hold twice, until
        `need` pages are free. The tree's copies of a running request's own pages (`_readopt`), then the tails of
        junctions that no child shares, along every locked path, the hit an admission locks among them
        (`RadixCache.share_tails`); with unverified pages last in both (a removal of them then takes the nodes they went
        into). Under page pressure this runs at every admission that waits for pages, mostly to find nothing: when no
        page is the tree's alone (`RadixCache.tree_only`), nothing can be swapped or freed, and only the `dup` flags of
        requests whose duplicates may have gone can change (`_readopt`'s `settled`)."""
        kv, cache = self.kv, self.cache
        if kv.num_free >= need:
            return
        if not cache.tree_only():
            self._readopt(settled=True)
            return
        self._readopt()
        if kv.num_free < need:
            cache.share_tails()
        if kv.num_free < need and self._n_unv:
            self._readopt(force=True)
            if kv.num_free < need:
                cache.share_tails(force=True)

    def _readopt(self, force: bool = False, settled: bool = False) -> None:
        """Free the tree pages that duplicate a running request's own. A request that computed a span the tree held
        for another request, which still referenced it, keeps its own pages for that span; once the other request
        leaves, the tree's copy is pinned by this request's lock alone. `RadixCache.readopt` along the lock adopts the
        request's pages and frees the copies. Without this, a request alone could hold two copies of a prefix and
        never fit its next chunk: its prefill would block forever, with nothing else to run. A request's unverified
        pages are never adopted where the tree's are verified (`cache.radix`), which keeps such copies too: `force`
        adopts them as well, the last resort (a removal of them then takes the nodes they went into). `settled`: no
        page is the tree's alone, so nothing is adopted (`_dedup`), and a request whose `dup` was found true since the
        tree last gave a page up (`Req.dup_swaps`, `RadixCache.swaps`) still holds that duplicate, pinned by another
        request: its walk would leave its flag as it is, and it is skipped."""
        swaps = self.cache.swaps
        for slot in np.flatnonzero(self.aid > 0).tolist():
            req = self.slot_req[slot]
            if not req.dup or req.nocache or (settled and req.dup_swaps == swaps):
                continue
            aid = int(self.aid[slot])
            if aid in self._plan_admits:
                continue
            # Only what earlier steps wrote, so the rows of the plan being built may read what the tree then holds
            # (stream order): the positions of this plan's own chunk stay out, and so does a request admitted in
            # this plan, whose pages come from this step's copies and forward.
            info = self._plan_rows.get(aid)
            whole, remain = True, False
            for lk in (req.lock, req.plock):
                if lk is None or lk.dead:
                    continue
                full = self._lock_end(req, lk)
                end = full if info is None else min(full, info[1])
                remain |= self.cache.readopt(lk, req.pages, end, int(self.last_step[slot]), aid, req.cow, req.hit,
                                             force)
                whole &= end == full
            if whole:
                req.dup = remain
                if remain:
                    req.dup_swaps = self.cache.swaps

    @staticmethod
    def _lock_end(req: Req, lk: Node) -> int:
        """Where the request's pages may replace the tree's along its lock `lk` (`_readopt`, `_adopt_verified`): the
        lock's end, but along its precise lock only what it computed precise (`pend`): it computes the positions past
        that on the FP8 path. (An attention-only precise hit locks the node ending at the hit, `RadixCache.end_at_hit`,
        not the node holding hit − 1, which could run past the hit and past the request's own tokens.)"""
        return lk.end if lk is req.lock else min(lk.end, req.pend)

    # -- admission ---------------------------------------------------------------------------------------------

    def _admit(self, admitted: list[Req]) -> None:
        deferred: list[Req] = []
        while self.num_waiting and self._budget.soft > 0:
            req = self._peek_waiting()
            if self.slots.num_free == 0 or self.states.num_free_live == 0:
                break
            if req.held or (self.paused_frontends and req.frontend in self.paused_frontends):
                # held: its in-flight sample comes with this turn's results, readmit it with that token; paused: its
                # frontend lags behind its outputs (backpressure)
                self._pop_waiting()
                deferred.append(req)
                continue
            done = self._admit_one(req)
            if done is Admit.WAIT:
                self._pop_waiting()
                deferred.append(req)
                continue
            if done is Admit.FAILED:  # its grammar could not be rebuilt: finished (and counted out), still the head
                self._dequeue_head(req)
                continue
            if done is Admit.NO_ROOM:
                break
            if done is not Admit.ADMITTED:
                raise RuntimeError(f"request {req.label}: admission's outcome {done!r}")
            self._to_running(req)
            admitted.append(req)
        for req in deferred:
            self._to_waiting(req)

    def _admit_one(self, req: Req) -> Admit:
        """Admit `req` with its first chunk (charged to the step's budget), or not (`Admit`)."""
        b, cache, ps = self.builder, self.cache, self.ps
        total = len(req.tokens)
        step = self.step_id + 1
        m = pm = Match()
        refine, short = 0, False
        if cache is not None:
            forbid = (lambda pos: self._in_span(req, pos)) if self.bidi and len(req.sp_start) else None
            key = self._key(req)
            # A penalised readmission resumes within its prompt at most: the runner counts its outputs as it
            # recomputes them (`sched.plan`, ADMIT), and only a prompt's tokens can come with an admission
            cap = total if req.pen is None else min(total, req.prompt_len + 1)
            m = cache.match(req.ns, key, step, cap - 1, forbid=forbid, spans=req.spans)
            if self.pns_on:  # the precise prefix's copies (module docstring)
                pm = cache.match(precise_ns(req.ns), key, step, cap - 1, forbid=forbid, spans=req.spans)
            # The hit is pinned by its STATE_COPY below, not here: a request that waits reads nothing, and its
            # matched checkpoint may then make room for the branch checkpoint it waits for. Until the STATE_COPY,
            # nothing can evict it: `_branch_wait` waits whenever its allocation succeeds, and page eviction skips the
            # hit's path once it is locked.
            best = max(m.hit, pm.hit)
            pending = m.pending
            if pending and self.precise == "reuse" and pm.hit and pending - pm.hit < min(self.grid, self.refine_min):
                # an FP8 checkpoint that the `short` rule below passes over whatever its writer: once it is visible,
                # the request resumes from the precise copy all the same, so a wait would only cost it a step
                pending = 0
            wait = pending > best or self._branch_wait(m, cap, step, floor=pm.hit)
            if not wait and self.pns_on:
                # a precise copy at least as deep as the FP8 hit is being written by this plan: it comes next step
                wait = (pm.pending > pm.hit and pm.pending >= m.hit) or \
                    self._branch_wait(pm, cap, step, floor=m.hit)
            if not wait and self.precise == "reuse" and m.hit > pm.hit:
                gap = m.hit - pm.hit
                if pm.hit and gap < self.refine_min and (gap < self.grid or not m.node.on_precise):
                    # an FP8 checkpoint less than a refine above the precise copy whose state sits on the prefix's
                    # FP8 copy (the end of a request whose prompt was the shared prefix, computed before the refine),
                    # or less than a grid above it whatever its writer (a chunk cut short at an image span): resume
                    # from the precise one, which stays locked, and compute the rest on the FP8 path
                    short = True
                elif gap >= max(1, self.refine_min) and m.node.hits >= self.refine_after:
                    if pm.pending > pm.hit or self._refining_now(m.node):
                        # a precise prefix of its path is being written, or another request refines this checkpoint
                        # in several steps (its chunks before the last leave no checkpoint): resume next step
                        wait = True
                    else:
                        refine = m.hit  # a refiner: [pm.hit, m.hit) in PRECISE rows
            if wait:
                req.waits += 1
                self.cache_waits += 1
                return Admit.WAIT
        # It continues where an earlier request stopped (`Req.continues`) if its key runs through the whole of a cached
        # path's last node: the end of what that request computed, where nothing branches
        extends = bool(m.path) and m.kv_len == m.path[-1].end and not m.path[-1].children
        if refine or short or (pm.hit and pm.hit >= m.hit):
            use = pm  # a precise hit (or a refiner's start: the deepest precise checkpoint, or 0)
        else:
            use = m
        hit = use.hit
        if not refine and self.precise == "bf16" and not hit:
            refine = total  # every prefill row of a prompt without a hit is PRECISE
        on_fp8 = use is m and m.node is not None and m.node.on_precise
        req.base_precise = bool(refine) or (use is pm and bool(hit)) or on_fp8
        if hit:
            if use.node.end != hit:  # attention-only: lock what the hit reads, no more (`RadixCache.end_at_hit`)
                cache.end_at_hit(use)
            cache.lock(use.node)
            if self.window:  # its window's pages are read below, after `_ensure_pages` (which evicts windows)
                cache.pin_checkpoint(use.node)
        req.refine = refine
        req.targets = self._targets(req, hit, max(m.kv_len, pm.kv_len), total)

        def give_up(result: Admit) -> Admit:  # not admitted: the hit and the images' reservation go
            if hit:
                cache.unlock(use.node)
            if len(req.sp_start):
                self.encoder.rollback()
            return result

        chunk = self._next_chunk(req, hit, total)
        if chunk is None:
            return give_up(Admit.NO_ROOM)
        end, enc_new, nf, precise = chunk
        # The copy of a partial last page; under MTP also of a full one ending at the hit, whose drafter KV at hit − 1
        # the first chunk rewrites (it depends on the token at the hit), if the drafter has KV of its own. Window mode:
        # the window's pages before the hit are shared; a partial last one is copied like the KV's
        wps = self.wps
        cow = 1 if hit % ps or (self.draft_kv and hit) else 0
        n_new, nw_new, nf = self._chunk_pages(end, nf, -(-hit // ps), -(-hit // wps), cow, 1 if hit % wps else 0,
                                              final=end == total)
        if not self._ensure_pages(n_new, nw_new):
            self.page_waits += 1
            return give_up(Admit.NO_ROOM)
        start, matcher = None, None
        if req.compiled is not None:
            start, matcher = self._grammar_start(req)
            if start is GrammarStart.FAILED:
                return give_up(Admit.FAILED)
        # After _ensure_pages: freeing duplicates may swap the tree's pages on the (locked) hit path.
        full, tail = cache.hit_pages(use) if hit else ([], 0)
        below = False
        if cow and not tail:
            tail = full.pop()  # the full page holding hit − 1: copied, like a partial one
            # It holds another admission's unverified writes, or its drafter KV at hit − 1 comes from a state computed
            # from some (`_unverified`)
            below = bool(cache.unverified[tail]) or self._unverified_ckpt(use.node)
        slot = self.slots.alloc()
        live = self.states.alloc_live()
        own = self.kv.alloc(n_new)
        assert slot is not None and live is not None and own is not None
        if self._n_unv or self._orphans:  # a page freed while marked unverified carries no mark into its next use
            cache.unverified[own] = 0
        self.kv.incref(full)
        pages = full + own
        wl, wshared, wtail, wown = 0, [], 0, []
        if self.window:
            wl = self._wlow(hit)
            if hit:
                lo, wp = self.windows.get(use.node.ckpt)
                if lo != wl or len(wp) != -(-hit // wps) - lo:
                    raise RuntimeError(f"the window at {hit} covers pages [{lo}, {lo + len(wp)})")
                wshared = wp[:hit // wps - lo]
                wtail = wp[-1] if hit % wps else 0
            wown = self.wkv.alloc(nw_new)
            assert wown is not None
            self.wkv.incref(wshared)
        aid = self._next_aid
        self._next_aid += 1
        self._admissions += 1
        self._plan_admits.add(aid)
        req.status, req.slot, req.state_slot, req.pages = RUNNING, slot, live, pages
        req.lock = req.plock = None
        req.resumed, req.own_ckpts, req.continues, req.demoted = None, [], extends, []
        req.nocache = False
        req.dup_swaps = -1
        req.pend = 0
        if hit:
            if use is pm:
                req.plock, req.pend = use.node, hit  # what it resumes from is precise
                self.precise_hits += 0 if refine else 1
            else:
                req.lock = use.node
                use.node.hits += 1  # an FP8 checkpoint's resumes: refined once it has served `refine_after`
                if use.node.ckpt:
                    req.resumed = (use.node, use.node.ckpt, use.node.ckpt_ready)
                    req.continues |= use.node.ckpt_src in CONTINUED
            if use.node.ckpt:  # a checkpoint's resume (hybrid and sliding-window models), in either namespace
                use.node.served += 1
                self.ckpt_resumes[use.node.ckpt_src] += 1
        if refine and self.precise == "reuse":
            self.refines += 1
            piece = min(self.precise_cap or self.piece, self.max_tokens)
            n = -(-refine // piece) - hit // piece
            log.info("request %s refines a shared prefix: PRECISE rows [%d, %d) in %d %s of up to %d tokens",
                     req.label, hit, refine, n, "piece" if n == 1 else "pieces", piece)
        req.deps = cache.pending_writers(use.path, hit, use.node) if hit else set()
        if not req.t_admit_ns:
            req.t_admit_ns = time.perf_counter_ns()
        first = req.cached < 0
        if first:
            req.cached = hit
            self.prefix_queries += req.prompt_len
            self.prefix_hits += hit
        else:
            self.readmit_hits += hit
        if cache is not None:
            self._count_lost(m, pm, use, hit)
        if self.hybrid and cache is not None:
            if hit:
                self.ckpt_hits += 1
            else:
                self.ckpt_misses += 1
        self.slot_req[slot] = req
        self.aid[slot] = aid
        self.fe[slot] = req.frontend
        self.fe_rid[slot] = req.rid
        self.mirror[slot] = self.num_sched[slot] = hit  # its chunk's row moves both (`_chunk_row`)
        self.dev_inflight[slot] = 0
        self.total[slot] = total
        self.n_out[slot] = req.n_out
        self.max_out[slot] = req.max_out
        self.next_ckpt[slot] = (total // self.ckpt_interval + 1) * self.ckpt_interval if self.ckpt_interval else 1 << 62
        self.ckpt_pending[slot] = False
        self.overrun[slot] = False
        self.pending[slot] = 0
        self.decoding[slot] = False  # until its final chunk (`_chunk_row`)
        self.end_at[slot] = NEVER_ENDS
        self.ignore_eos[slot] = req.ignore_eos
        self.num_pages[slot] = len(pages)
        if self.window:
            req.wpages = [0] * wl + wshared + wown
            self.num_wpages[slot] = len(req.wpages)
            self.wlo[slot] = wl
        self.prio[slot] = req.priority if self.policy == "priority" else 0
        self.admit_order[slot] = req.arrival
        self.has_first[slot] = req.first_done
        self.runner_ns[slot] = req.runner_ns
        self.verify_rows[slot], self.verify_tokens[slot] = req.verify_rows, req.verify_tokens
        self.held[slot] = 0  # the slot's drafts are another admission's: its final chunk's SAMPLE row drafts anew
        if req.n_out:  # readmission after preemption: refill the ring's partial row
            gen = req.out_chunks[0]  # _preempt left exactly the n_out generated tokens here
            full_rows = req.n_out - req.n_out % _OUT_W
            self.out_buf[slot, :req.n_out - full_rows] = gen[full_rows:]
            req.out_chunks = [gen[:full_rows]] if full_rows else []
        self._set_stop_ids(slot, req.stop_ids)
        self._order_dirty = True
        flags = 0
        if start is not None:
            flags = ADMIT_CONSTRAINED  # ENDED too: constrained, without a mask
            if start is GrammarStart.ACTIVE:
                req.matcher = matcher
                self.masked[slot] = True
                self.n_masked += 1
            elif start is GrammarStart.ENDED:
                req.matcher = None
            else:  # FAILED returned above
                raise RuntimeError(f"unknown grammar start {start!r}")
        if req.unrep:  # a readmission with output tokens its frontend has not received: they go with its first result
            self.unrep[slot] = True  # (never with a forced prefix, `_n_forced`)
            self._n_unrep += 1
        if req.logprobs >= 0:
            flags |= ADMIT_LOGPROBS
            self.lp[slot] = req.logprobs
            self.n_lp += 1
        terms: dict[str, Any] = {}
        if req.pen is not None:
            rp, pp, fp = req.pen
            terms = dict(rep=rp, pres=pp, freq=fp, prompt_len=req.prompt_len, prompt=req.tokens)
        if req.bias is not None:
            terms.update(bias_ids=req.bias[0], bias_val=req.bias[1])
        if self.k:
            if not req.stop_list:
                eos = [] if req.ignore_eos else self.eos_ids.tolist()
                req.stop_list = list(dict.fromkeys([*eos, *req.stop_ids.tolist()]))
            b.add_admit(slot, live, hit, req.temperature, req.top_p, req.top_k, req.min_p, req.seed, flags,
                        max_len=req.prompt_len + req.max_out, stop=req.stop_list, rope_off=req.rope_off, **terms)
        else:
            b.add_admit(slot, live, hit, req.temperature, req.top_p, req.top_k, req.min_p, req.seed, flags,
                        rope_off=req.rope_off, **terms)
        b.add_pages(slot, 0, pages)
        req.hit, req.cow, req.cow_unv = hit, own[0] if tail else 0, below
        if tail:
            b.add_kv_copy(tail, own[0])
        if self.window:
            if wshared or wown:
                b.add_wpages(slot, wl, wshared + wown)
            if wtail:
                b.add_wkv_copy(wtail, wown[0])
        elif hit and self.hybrid:
            b.add_state_copy(use.node.ckpt, live)
            cache.pin_checkpoint(use.node)  # read by this plan: no later allocation while it is built may evict it
        row = self._chunk_row(req, slot, hit, end, enc_new, nf, precise)
        if req.unv >= 0:
            self._readmit_unverified(req, use is pm, hit, use.node)
        self._cache_chunk(req, slot, row, hit, end)
        if refine and self.precise == "reuse" and req.pend < refine:
            self._refining[m.node.id] = req  # until its PRECISE rows reach `refine` (`_refining_now`)
        return Admit.ADMITTED

    # -- structured output ------------------------------------------------------------------------------------

    def _grammar_start(self, req: Req) -> tuple[GrammarStart, Any]:
        """An admission's grammar (`GrammarStart`): ACTIVE with a clone of the compiled grammar's matcher, advanced over
        the tokens the request generated before a preemption; ENDED if its grammar accepted an EOS among them (under
        ignore_eos); FAILED if they no longer parse (never expected): the request then fails."""
        g = self.grammars
        m = g.matcher(req.compiled)
        gen = req.tokens[req.prompt_len:].tolist()
        if not gen:
            return GrammarStart.ACTIVE, m
        ok, ended = advance_matcher(m, gen, g.eos)
        if not ok:
            self._grammar_failed(req, m, "rebuilding it after a preemption")
            self._extra.append((req.frontend, req.rid, FinishReason.ERROR))
            self.num_failed += 1
            self._to_finished(req)  # still waiting, and the heap's head (`_admit`)
            self._settle_unverified(req, Unverified.ORPHAN)  # this plan, being built, may read them
            return GrammarStart.FAILED, None
        return (GrammarStart.ENDED, None) if ended else (GrammarStart.ACTIVE, m)

    def _n_forced(self, req: Req, chunk: int, precise: bool = False) -> int:
        """How many of its grammar's forced tokens a request's final prefill chunk of `chunk` tokens takes
        (architecture.md, "Grammar-aware heads"): only before its first output, and as many as leave room for the
        chunk's own sample within max_tokens (its positions then stay below ADMIT's commit limit), fit what the plan has
        left (max_num_batched_tokens, the step's budget, weighed for a PRECISE chunk's tokens,
        long_prefill_token_threshold) and the positions `add` reserved past the prompt (`reserve`: a prompt it took fits
        the pool with them, but for a drafter's chain after them, `_chunk_pages`). They are best-effort: a chunk that
        would wait or preempt for their pages takes none (`_chunk_pages`), and one whose row has a CKPT record off the
        grid at the prompt's end samples them (`_commit_tails`).

        Under batch invariance (`forced_whole`) a final chunk takes them whole, as many as the request alone decides:
        within max_tokens and the positions `add` reserved, less a drafter's chain after them (their pages are then
        needed like the chunk's own, `_chunk_pages`), past the step's prefill bound and the chunk cap but within
        max_num_batched_tokens; −1 when they do not fit next to the chunk (it is cut short of the prompt's end,
        `_next_chunk`). A PRECISE chunk (the precise prefix, batch-dependent anyway) takes what its budget leaves."""
        if not req.forced or req.n_out or req.unrep:
            return 0
        room = self._budget.room(chunk, precise)
        if self.forced_whole:
            f = min(len(req.forced), req.max_out - 1, self.reserve - self.chain)
            if precise:
                return max(0, min(f, room))
            return f if f <= self._budget.hard - chunk else -1
        return max(0, min(len(req.forced), req.max_out - 1, room, self.reserve))

    def _force(self, req: Req, slot: int, n: int) -> np.ndarray | None:
        """The chunk was planned with n forced tokens: its matcher takes them, and they are the request's first output
        tokens from now on (its output ring, `n_out`), unreported until the chunk's result arrives (`unrep`). None if
        the matcher refuses them (never expected: they were computed along a fresh matcher of the same grammar): it is
        left as it was and the chunk samples them (`_commit_tails`), so a grammar that rejects the sample too fails
        that request alone (`_advance_grammars`), not the engine."""
        toks = list(req.forced[:n])
        took = req.matcher.try_consume_tokens(toks)  # up to the first refused token, without llguidance's error state
        if took < n:
            if took:
                req.matcher.rollback(took)
            log.error("request %s: its grammar refused its own forced prefix: its chunk samples it", req.label)
            return None
        self.out_buf[slot, :n] = toks
        self.n_out[slot] = n
        req.unrep = toks
        self.unrep[slot] = True
        self._n_unrep += 1
        self.forced_requests += 1
        self.forced_tokens += n
        return np.asarray(toks, np.int32)

    def _commit_tails(self) -> None:
        """The plan's final chunks' forced tokens go into their rows now that every CKPT record of the plan is placed
        (architecture.md, "Grammar-aware heads"): a record at a row's n_query, the prompt's end, then lies inside the
        row, where a runner takes a checkpoint only on the grid (the GDN prefill's 64-token chunks, from the row's
        start). A row with one off the grid there keeps the sampled path's shape, and its request samples its forced
        tokens, whose budget and pages it holds anyway; so does one whose matcher refuses them (`_force`). Only an
        end-of-chunk checkpoint can be off the grid, and only a branch point that a request admitted after it in this
        plan waits for (`_branch_wait`) puts one at a final row's prompt end: hints lie strictly inside the prompt
        (`Req.hints`), and a final chunk's own checkpoints on the grid below its end. The test stays general, as a
        safeguard."""
        b, g = self.builder, self.grid
        rows, offs = b.b.ckpt_sched[:b.n_ckpt], b.b.ckpt_offset[:b.n_ckpt]
        for row, slot, req, nf in self._tails:
            o = offs[rows == row]
            if ((o > 0) & (o % g != 0)).any():
                self.forced_declined += 1
                continue
            toks = self._force(req, slot, nf)
            if toks is None:
                continue
            b.extend_row(row, toks, None if req.rope is None else np.full((3, nf), req.rope_off, np.int32))
            self.mirror[slot] += nf
            if self.masked[slot] and not req.ignore_eos:  # its first token: an EOS if the forced tokens end its grammar
                self.end_at[slot] = 0 if req.matcher.is_stopped() else NEVER_ENDS
        self._tails.clear()

    def _grammar_failed(self, req: Req, m: Any, when: str) -> None:
        """llguidance's error message quotes the tokens it parsed (the output): not logged."""
        self.grammars.errors += 1
        log.error("request %s: its grammar rejected a token %s", req.label, when)

    def _advance_grammars(self, s: np.ndarray, rows: np.ndarray, tok: np.ndarray, n_new: np.ndarray | None,
                          finish: np.ndarray) -> None:
        """Advance the matchers of masked slots over the tokens their rows delivered (`rows`: indices into `s`).
        A finishing row needs nothing; an EOS under ignore_eos terminates the grammar (`advance_matcher`; unmasked
        after); a token the grammar rejects finishes the request with an error (`finish`, in place). Otherwise, without
        ignore_eos, `end_at` is computed for the slot's next sample row (`end_drafts`; `self.drafts` already holds its
        window's drafts; `_end_wait` compares it with the count that row verifies)."""
        eos = self.grammars.eos
        off = np.cumsum(n_new) - n_new if n_new is not None else None
        for i in rows.tolist():
            if finish[i]:
                continue
            slot = int(s[i])
            req = self.slot_req[slot]
            toks = tok[off[i]:off[i] + n_new[i]].tolist() if off is not None else [int(tok[i])]
            ok, ended = advance_matcher(req.matcher, toks, eos)
            if not ok:
                self._grammar_failed(req, req.matcher, f"at output {int(self.n_out[slot])}")
                finish[i] = FinishReason.ERROR
                self.num_failed += 1
            elif ended:
                self._unmask(slot)
                req.matcher = None
            elif not req.ignore_eos:
                d = self.drafts[slot]  # the drafts this step made (−1 past them: `StepPlan.draft_n`)
                self.end_at[slot] = end_drafts(req.matcher, d[d >= 0].tolist() if self.k else [])

    def fill_masks(self, inf: Inflight, buf: np.ndarray) -> int:
        """The masks of a plan's MASKED rows, in plan order (`plan.py`), after `process` of the step before it:
        returns how many rows of `buf` it wrote, every one of them defined (the sparse head ORs a step's masks into the
        union it samples over, so none may keep what an earlier step left there):
        - a row whose request finished, failed or left since the plan (its slot holds another admission, or none)
          allows the grammar's EOS alone (`eos_only`, the lowest EOS id) in each of its masks: its sample is dropped
          (results are matched by admission), so no output changes, and the union gains one token at most (all ones
          would send the whole step to the dense head);
        - a request alive whose grammar terminated (an EOS under ignore_eos) is unconstrained: all ones (vLLM's
          rule), as are an MTP row's masks past a drafted EOS under ignore_eos (ours verifies those drafts
          unconstrained; vLLM cuts them);
        - along an MTP row's drafts the backend writes the mask at the matcher's state and one after each of the first
          w = min(v, e) drafts: v, those the grammar takes in a row (`validate_tokens`, up to and with an EOS; none
          once it has stopped), e, the index of the first EOS among them (k: none). Verification never reads the masks
          after those (a draft the grammar rejects is never a masked sample; an EOS ends the row, or the grammar under
          ignore_eos). They allow the EOS alone too, written after the fill. v comes from the whole draft list, which
          gives the same w as the list cut at a drafted EOS (nothing before e is an EOS): one walk a row.
        Under MTP it also counts the drafts each row's grammar refuses (`Inflight.invalid`; all of them once the
        grammar has stopped): `vllm:spec_decode_num_draft_tokens` counts the drafts the grammar takes (and under
        ignore_eos the accepted ones past a drafted EOS: `_count_verify`), so these are not counted as drafted, here
        or in `process`."""
        k, g = self._v(inf) if self.k else 0, self.grammars  # the drafts the plan's device rows verify
        eos = g.eos
        only = eos_only(buf.shape[1], min(eos)) if eos else -1  # a grammar with no EOS ids: all ones
        singles: list[tuple[Any, int]] = []
        drafts: list[tuple[Any, int, list[int]]] = []
        unread: list[tuple[int, int]] = []  # masks along drafts that verification never reads: [start, end) in `buf`
        invalid = np.zeros(len(inf.slots), np.int64) if k else None
        windows: list[tuple[int, int, int]] = []  # live verify rows (verify-mask counters): (plan row, offset, slot)
        off = 0
        for r in inf.masked.tolist():
            slot = int(inf.slots[r])
            n = 1 + k if k and inf.device[r] else 1
            req = self.slot_req[slot]
            if self.aid[slot] != inf.aids[r] or req is None:
                buf[off:off + n] = only
            elif req.matcher is None:
                buf[off:off + n] = -1  # its grammar terminated (ignore_eos): unconstrained
                if n > 1:
                    windows.append((r, off, slot))
            elif n == 1:
                singles.append((req.matcher, off))
            else:
                windows.append((r, off, slot))
                m, d = req.matcher, self.drafts[slot, :k].tolist()
                v = 0 if m.is_stopped() else m.validate_tokens(d)
                invalid[r] = k - v
                e = k if eos.isdisjoint(d) else next(j for j, t in enumerate(d) if t in eos)
                if e < k and req.ignore_eos:  # past a drafted EOS the grammar has terminated: unconstrained
                    buf[off + e + 1:off + n] = -1
                    d = d[:e]
                if d:
                    drafts.append((m, off, d))
                else:
                    singles.append((m, off))
                w = min(v, e)
                if w < len(d):
                    unread.append((off + 1 + w, off + 1 + len(d)))
            off += n
        g.fill(singles, drafts, buf, off)
        for start, end in unread:  # after the fill, so they never depend on what the backend wrote past w
            buf[start:end] = only
        inf.invalid = invalid
        if windows:
            self._window_masks(inf, buf, windows)
        return off

    def _window_masks(self, inf: Inflight, buf: np.ndarray, windows: list[tuple[int, int, int]]) -> None:
        """The verify-mask counters (`_rejections`): the popcount and class (`MASK_CLASSES`) of each mask of the
        verify rows `windows` (plan row, offset in `buf`, slot), and the drafts each row verifies, for `process` to
        class its first rejected draft and count the masks verification read. One pass over the rows' masks, behind
        the forward."""
        k, n, g = self._v(inf), len(inf.slots), self.grammars
        rows = np.asarray([w[0] for w in windows], np.int64)
        idx = (np.asarray([w[1] for w in windows], np.int64)[:, None] + np.arange(1 + k)).ravel()
        block = buf[idx].view(np.uint32)
        vocab = g.backend.vocab_size  # an all-ones mask (unconstrained) counts the vocabulary, not its padding bits
        pop = np.minimum(np.bitwise_count(block).sum(axis=1, dtype=np.int64), vocab)
        n_eos = np.zeros(len(block), np.int64)  # the grammar's EOS ids a mask allows (several under auto, xgrammar)
        for e in g.eos:
            n_eos += (block[:, e >> 5] >> np.uint32(e & 31)) & np.uint32(1)
        cls = np.where((pop == n_eos) & (n_eos > 0), 1, np.where(pop == 1, 0, np.where(pop == vocab, 3, 2)))
        inf.wpop = np.full((n, 1 + k), -1, np.int64)
        inf.wpop[rows] = pop.reshape(-1, 1 + k)
        inf.wcls = np.zeros((n, 1 + k), np.int8)
        inf.wcls[rows] = cls.reshape(-1, 1 + k)
        inf.wdraft = np.full((n, k), -1, np.int32)
        inf.wdraft[rows] = self.drafts[np.asarray([w[2] for w in windows], np.int64), :k]

    # -- images -----------------------------------------------------------------------------------------------

    def _image_chunk(self, req: Req, start: int, total: int, avail: int) -> tuple[int, list[int]]:
        """The end of the next chunk [start, end) of a request with images, and the images it must encode (indices).
        Every image whose span overlaps the chunk, or the token after a non-final chunk with a drafter that writes KV
        (its input there), must have its embeddings: cached, or encoded in this plan within the encoder budget and the
        encoder cache. The chunk is `_chunk_end`'s, cut before the first image that cannot (and before its look-ahead
        with drafter KV); end == start when even the first token cannot run in this plan."""
        end = self._span_end(req, start, total, self._chunk_end(req, start, total, avail))
        while end > start:
            cut, enc = self._image_window(req, start, end, total)
            if cut >= end:
                return end, enc
            if cut <= start:
                break
            end = self._span_end(req, start, total, min(cut, self._chunk_end(req, start, total, cut - start)))
            if end <= start and self.aligned and (self._enc_left == self.enc_budget or
                                                  self._finishes_own_span(req, start, cut)):
                # Batch invariance: no grid point before the image it cannot have, though the step's whole encoder
                # budget was its own, or the image waits for the encoder rows of a span the request itself has not
                # finished: cut exactly there (architecture.md, "Batch invariance"), or it would wait forever
                self.image_cuts += 1
                end = cut
        return start, []

    # -- bidirectional image spans (architecture.md, "The scheduler") ------------------------------------------

    @staticmethod
    def _finishes_own_span(req: Req, start: int, cut: int) -> bool:
        """Whether a chunk [start, cut) would finish an image span the request already references (one it entered in an
        earlier chunk), releasing its encoder rows."""
        i = int(np.searchsorted(req.sp_end, start, "right"))
        return i < len(req.sp_start) and req.sp_start[i] < start and req.sp_end[i] <= cut and req.enc[i] is not None

    @staticmethod
    def _in_span(req: Req, pos: int) -> bool:
        """Whether position `pos` lies strictly inside one of the request's image spans (s < pos < e)."""
        i = int(np.searchsorted(req.sp_end, pos, "right"))  # the first span that ends after pos
        return i < len(req.sp_start) and int(req.sp_start[i]) < pos

    def _span_end(self, req: Req, start: int, total: int, end: int) -> int:
        """The end of a chunk [start, end) that no bidirectional span straddles: cut before a span it would end inside
        (on the grid when that stays past `start`); a chunk starting at a span takes all of it, within the plan's
        `max_num_batched_tokens` left, else nothing (end == start)."""
        if not self.bidi or not len(req.sp_start):
            return end
        g = self.grid
        while self._in_span(req, end):
            i = int(np.searchsorted(req.sp_end, end, "right"))
            s, e = int(req.sp_start[i]), int(req.sp_end[i])
            if s > start:
                end = s - s % g if s % g and s - s % g > start else s  # may land in an earlier span: check again
                continue
            if s < start:
                raise RuntimeError(f"request {req.label}: a chunk starts at {start}, inside the image span [{s}, {e})")
            return e if e - start <= self._budget.hard and e <= total else start
        return end

    def _ckpt_pos(self, req: Req, c: int, start: int, end: int, d: int) -> int:
        """Where a chunk [start, end) takes a checkpoint placed at c for a target desired at d, when c would lie inside
        a bidirectional span: the span's end if the chunk holds it and that is at or below d, else the span's start
        (`start`: none)."""
        if not self.bidi or not self._in_span(req, c):
            return c
        i = int(np.searchsorted(req.sp_end, c, "right"))
        s, e = int(req.sp_start[i]), int(req.sp_end[i])
        if e <= min(d, end):
            return e
        return s if s > start else start

    def _add_spans(self, row: int, req: Req, start: int, end: int) -> None:
        """SPANS: the bidirectional spans row `row` prefills ([start, end), every one whole: `_span_end`)."""
        if not len(req.sp_start):
            return
        i = int(np.searchsorted(req.sp_end, start, "right"))
        j = int(np.searchsorted(req.sp_start, end, "left"))
        for x in range(i, j):
            s, e = int(req.sp_start[x]), int(req.sp_end[x])
            if s < start or e > end:
                raise RuntimeError(f"request {req.label}: the chunk [{start}, {end}) splits the span [{s}, {e})")
            self.builder.add_span(row, s, e)

    def _image_window(self, req: Req, start: int, end: int, total: int) -> tuple[int, list[int]]:
        """(end, or the cut before the first image that cannot be had; the images to encode). The chunk reserves its
        images in the encoder cache in span order (`EncoderCache.reserve`): a cached one is referenced, so no later
        allocation of the chunk evicts it, and a new one is allocated within the plan's encoder budget
        (`EncoderCache.alloc`). One that does not fit, or that an earlier allocation of the chunk evicted, ends the
        chunk before it, and the chunk's reservation is rolled back; otherwise it stays open for `_image_commit` (or a
        rollback, when the chunk does not get its pages). `_enc_short` records a refusal within the budget: the cache's
        rows are referenced (`_schedule_running_prefills`)."""
        look = 1 if self.draft_kv and end < total else 0
        enc = self.encoder
        i = int(np.searchsorted(req.sp_end, start, "right"))
        j = int(np.searchsorted(req.sp_start, end + look, "left"))
        budget, held, new = self._enc_left, set(), []
        for x in range(i, j):
            k = req.sp_id[x]
            if req.enc[x] is not None or k in held:
                continue
            if k in enc.entries:
                e = enc.reserve(k)
            else:
                n = int(req.sp_end[x] - req.sp_start[x])
                e = enc.reserve(k, n) if n <= budget else None
                if e is not None:
                    budget -= n
                    new.append(x)
                elif n <= budget:
                    self._enc_short = True
            if e is None:
                enc.rollback()
                s = int(req.sp_start[x])
                return (s - (1 if self.draft_kv else 0) if s > start else start), []
            held.add(k)
        return end, new

    def _image_commit(self, req: Req, start: int, end: int, total: int, new: list[int]) -> None:
        """Commit the chunk's reservation (`_image_window`): the request references each of its images, an ENCODE
        writes each new one, and the spans this chunk finishes drop their references after the plan is built. Their
        recency follows the cached images, then the new ones, then repeats of new ones, in span order."""
        look = 1 if self.draft_kv and end < total else 0
        enc = self.encoder
        i = int(np.searchsorted(req.sp_end, start, "right"))
        j = int(np.searchsorted(req.sp_start, end + look, "left"))
        fresh = {req.sp_id[x] for x in new}
        seen: set[tuple[int, bytes, str | None]] = set()

        def take(x: int) -> Entry:  # an image's first span takes the reservation's reference, a repeat one more
            k = req.sp_id[x]
            e = enc.get(k)
            if k in seen:
                enc.ref(e)
            seen.add(k)
            req.enc[x] = e
            return e

        for x in range(i, j):
            if req.enc[x] is None and req.sp_id[x] not in fresh:
                take(x)
                self.encoder_hits += 1
        for x in new:
            e, n = take(x), int(req.sp_end[x] - req.sp_start[x])
            im = req.images[x]
            self.builder.add_encode(im.key, n, (im.grid_t, im.grid_h, im.grid_w), (im.height, im.width), im.pixels,
                                    e.rows)
            self._enc_left -= n
            self.encoder_images += 1
            self.encoder_tokens += n
        for x in range(i, j):  # a repeat of an image encoded just above
            if req.enc[x] is None:
                take(x)
        enc.commit()
        for x in range(i, j):
            if req.sp_end[x] <= end and req.enc[x] is not None:
                self._unref(req.enc[x])
                req.enc[x] = None

    def _image_inputs(self, req: Req, start: int, end: int, total: int) -> tuple:
        """The chunk's image inputs for `PlanBuilder.add_explicit_row`: (embedding rows or None, the next token's row
        (with drafter KV) or −1, M-RoPE deltas [3, n] or None, the deltas at start − 1)."""
        emb, next_emb = None, -1
        if len(req.sp_start):
            i = int(np.searchsorted(req.sp_end, start, "right"))
            n_sp = len(req.sp_start)
            while i < n_sp and req.sp_start[i] < end:
                s, e = int(req.sp_start[i]), int(req.sp_end[i])
                lo, hi = max(s, start), min(e, end)
                if emb is None:
                    emb = np.full(end - start, -1, np.int32)
                rows = (req.enc[i].rows if req.enc[i] is not None else self.encoder.get(req.sp_id[i]).rows)
                emb[lo - start:hi - start] = rows[lo - s:hi - s]
                i += 1
            if self.draft_kv and end < total:
                x = int(np.searchsorted(req.sp_end, end, "right"))
                if x < n_sp and req.sp_start[x] <= end:
                    e = req.enc[x] or self.encoder.get(req.sp_id[x])
                    next_emb = int(e.rows[end - int(req.sp_start[x])])
        rope = prev = None
        if req.rope is not None:
            p, off = req.prompt_len, req.rope_off
            if end <= p:
                rope = req.rope[:, start:end]
            elif start >= p:
                rope = np.full((3, end - start), off, np.int32)
            else:
                rope = np.concatenate([req.rope[:, start:], np.full((3, end - p), off, np.int32)], axis=1)
            prev = req.rope[:, start - 1] if 0 < start <= p else np.full(3, off if start > p else 0, np.int32)
        return emb, next_emb, rope, prev

    def _unref(self, e: Entry) -> None:
        if self._building:
            self._unrefs.append(e)  # a later chunk of this plan must not evict rows this one reads
        else:
            self.encoder.unref(e)

    def _flush_unrefs(self) -> None:
        for e in self._unrefs:
            self.encoder.unref(e)
        self._unrefs.clear()

    def _drop_images(self, req: Req, failed: bool = False) -> None:
        """Drop the request's encoder references (it left or was preempted, or its chunk waits for pages behind a
        request short of encoder rows and every later plan would be the same: `_schedule`); a failed request's images
        are forgotten too, unless another request still uses them (their embeddings may hold the non-finite values)."""
        for x, e in enumerate(req.enc):
            if e is not None:
                self._unref(e)
                req.enc[x] = None
        if failed and self.encoder is not None:
            self._flush_unrefs()
            # by key in its set's order, the order the freed blocks (and so later plans' rows) follow, then by digest
            for key in set(req.sp_key.tolist()):
                for k in dict.fromkeys(i for i in req.sp_id if i[0] == key):
                    e = self.encoder.entries.get(k)
                    if e is not None and not e.refs:
                        self.encoder.drop(k)

    def _count_lost(self, m: Match, pm: Match, use: Match, hit: int) -> None:
        """An admission's lost tokens: what the tree held for it within its limit (the deeper of its two namespaces'
        matches) past its hit, which it recomputes, by cause (`cache.radix`, Lost tokens); and `precise`: an FP8 copy's
        usable hit above the precise copy it resumed from (a refine, or a resume just below an FP8 checkpoint)."""
        lo = hit
        if use is pm and m.hit > hit:
            self.lost_tokens["precise"] += m.hit - hit
            lo = m.hit
        deep = m if m.held >= pm.held else pm
        lost_tokens(deep, lo, deep.held, self.lost_tokens)

    def _branch_wait(self, m: Match, total: int, step: int, floor: int = 0) -> bool:
        """The branch point lies in KV that the plan being built writes: add a checkpoint there to the writing row
        (if it can take one) and defer the request one step. Hybrid and sliding-window models only (`hybrid`). `m` is
        one namespace's match (the writer's row inserts into it); `floor`: the request's hit in the other one, which
        the wait must deepen."""
        if not self.hybrid:
            return False
        bp = min(m.kv_len, total - 1)  # the desired branch point, as in `_targets`; placed on the writer's grid below
        if bp <= max(m.hit, floor):
            return False
        node = self.cache.node_covering(m.path, bp - 1)
        if node is None or node.ready != step:
            return False
        info = self._plan_rows.get(node.writer)
        if info is None:
            return False
        row, start, end, wslot = info
        if not start < bp <= end:
            return False
        c = self._grid_point(start, end, bp)  # on the writing chunk's grid, as that row's own targets
        if self.aligned and (c - start) % self.grid:
            # Batch invariance: not at a final row's prompt end off its grid, which its forced prefix would pass and
            # which a hit would not replay (a resume there cuts on another grid): on the grid below it
            c = start + (c - start) // self.grid * self.grid
        if self.bidi:
            c = self._ckpt_pos(self.slot_req[wslot], c, start, end, bp)
        if c <= max(start, m.hit, floor):
            return False  # no position that row can checkpoint would deepen the hit
        target = self.cache.node_ending_at(node, c)
        if target.ckpt:
            return target.ckpt_ready == step
        w = self.slot_req[wslot]
        if self.window:  # the writer holds its window pages before c: the chunk reads them
            self.cache.attach(target, self._make_window(w, c), step, node.writer, "branch",
                              on_precise=w.base_precise)
        else:
            dst = self.cache.alloc_checkpoint(step)
            if not dst:
                self.ckpts_skipped += 1
                return False
            self.cache.attach(target, dst, step, node.writer, "branch", on_precise=w.base_precise)
            self.builder.add_ckpt(row, c - start, dst)
        if w.unv >= 0:  # computed from the writer's unverified writes: among them (module docstring, Rollback)
            w.unv_ckpts.append((target, target.ckpt, step))
        self.ckpts_taken["branch"] += 1
        return True

    def _targets(self, req: Req, hit: int, kv_len: int, total: int) -> list[tuple[int, str]]:
        """Desired checkpoint positions of this prefill (architecture.md, "Checkpoints": the end of a prompt, a branch
        point, a client hint), with their source: (position, "prefill" | "branch" | "hint" | "refine"). Hints and a
        refiner's target (`req.refine`, the FP8 checkpoint it recomputes: its PRECISE rows end there) are exact;
        `_cache_chunk` places the others."""
        if not self.hybrid or self.cache is None:
            return []
        out: dict[int, str] = {}
        # The desired positions, unrounded: `_grid_point` places each on the grid of the chunk that contains it, within
        # G − 1 below. Rounding them to the absolute grid first would put the end-of-prefill checkpoint up to 2(G − 1)
        # below total − 1 after a resume from an off-grid hit.
        for c, src in ((total - 1, "prefill"), (min(kv_len, total - 1), "branch")):
            if c > hit:
                out.setdefault(c, src)
        for h in req.hints:
            if hit < h <= total:
                out[h] = "hint"
        if hit < req.refine < total:
            out[req.refine] = "refine"
        return sorted(out.items())

    def _grid_point(self, start: int, end: int, d: int) -> int:
        """Where a chunk [start, end) can checkpoint a non-hint target desired at d (start < d ≤ end): the deepest
        position at or below d among start + k·G (mid-chunk) and end (end of chunk). `start` itself means none."""
        return end if d >= end else start + (d - start) // self.grid * self.grid

    def _chunk_end(self, req: Req, start: int, total: int, avail: int) -> int:
        """End of the next chunk of [start, total) given `avail` tokens of budget (architecture.md, "The scheduler"). A
        refiner's PRECISE rows end exactly at its target (the precise prefix), unless a hint before it ends them
        first."""
        end = start + avail
        r = req.refine
        cut = start < r < total and r <= end
        if cut:
            end = r
        g = self.grid
        for c, src in req.targets:
            if c <= start:
                continue
            if c >= end:
                break
            if src == "hint" and (c - start) % g:
                return c  # an end-of-chunk checkpoint exactly at the hint
        if self.aligned and start >= r:  # batch invariance: whole grids from the chunk's start;
            return end if end >= total else start + (end - start) // g * g  # PRECISE rows keep their pieces
        if not req.targets:
            return end
        if not cut and end < total and end % g and req.targets[-1][0] > end:
            aligned = end - end % g
            if aligned > start:
                end = aligned
        return end

    def _check_progress(self, chunk_cap: int) -> None:
        """Under batch invariance, a prefill must always be able to progress (architecture.md, "Batch invariance"): a
        request that prefills holds a slot, so at most max_num_seqs − 1 rows decode beside it, and what they leave must
        hold one grid (aligned cuts) or one token, plus a final chunk's whole forced prefix; a chunk cap below the grid
        would cut every aligned chunk to nothing."""
        if not (self.aligned or self.forced_whole):
            return
        forced = self.reserve - self.chain if self.forced_whole else 0
        need = (self.grid if self.aligned else 1) + forced
        left = self.max_tokens - (self.max_num_seqs - 1) * (1 + self.k)
        if left < need:
            raise ValueError(f"batch invariance: max_num_batched_tokens {self.max_tokens} leaves {left} tokens next to "
                             f"{self.max_num_seqs - 1} decode rows of {1 + self.k}, under the {need} a prefill needs "
                             f"({'a ' + str(self.grid) + '-token grid' if self.aligned else 'a token'}"
                             f"{f' and its forced prefix ({forced})' if forced else ''})")
        if self.aligned and 0 < chunk_cap < self.grid:
            raise ValueError(f"batch invariance: long_prefill_token_threshold {chunk_cap} is below the {self.grid}-"
                             "token grid every chunk is cut on")

    def _refining_now(self, node: Node) -> bool:
        """Another request is refining the FP8 checkpoint at `node` and has PRECISE rows left: it still runs in its
        slot and its precise prefix has not reached its target (a refine in several steps: several pieces, or a piece
        that a hint, an image span or the encoder budget cuts; one preempted or finished is not)."""
        r = self._refining.get(node.id)
        if r is None:
            return False
        if r.status == RUNNING and r.slot > 0 and self.slot_req[r.slot] is r and r.refine and r.pend < r.refine:
            return True
        del self._refining[node.id]
        return False

    def _refined(self, req: Req) -> None:
        """The request refines nothing any more: its PRECISE rows reached their target, or it left its slot. Forget it
        (`_refining_now`), so a finished request is not kept for the server's lifetime."""
        for nid, r in self._refining.items():
            if r is req:
                del self._refining[nid]
                return

    def _piece_tokens(self, req: Req, start: int) -> int:
        """A refine's PRECISE piece from `start` to the refine's end or the next multiple of the piece size (from
        position 0), if it fits what the plan has left of max_num_batched_tokens, else 0: a piece is never cut by the
        load, it waits (`precise_cap`, a harness's, replaces the piece size; an empty plan fits either)."""
        p = min(self.precise_cap or self.piece, self.max_tokens)
        n = min(req.refine, (start // p + 1) * p) - start
        return n if n <= self._budget.hard else 0

    def _cache_chunk(self, req: Req, slot: int, row: int, start: int, end: int) -> None:
        """A chunk [start, end) was planned: insert its KV, move the lock, and add its checkpoints. A PRECISE chunk
        (the precise prefix: start < `req.refine`) inserts into the request's precise namespace, the others into its
        FP8 namespace."""
        self._plan_rows[int(self.aid[slot])] = (row, start, end, slot)
        precise = start < req.refine
        if precise:
            self.precise_tokens += end - start
            req.pend = end  # its PRECISE rows continue a precise start (position 0 or a precise checkpoint)
            self._plan_precise = True
            if end >= req.refine and self._refining:
                self._refined(req)
        cache = self.cache
        if cache is None or req.nocache:
            return
        step = self.step_id + 1
        aid = int(self.aid[slot])
        before = cache.kept_dups
        ns = precise_ns(req.ns) if precise else req.ns
        if end < int(self.total[slot]) or req.unv >= 0:  # unverified until the request's next finite SAMPLE row
            self._unverified(req, aid, ns, start, end)
        deep = cache.insert(ns, self._key(req), end, req.pages, step, aid, req.cow, req.hit, req.spans)
        if cache.kept_dups != before:
            req.dup, req.dup_swaps = True, cache.swaps
        self._relock(req, deep)
        for d, src in req.targets:
            if d <= start:
                continue
            exact = src in ("hint", "refine")
            if d > end:
                # The chunk that contains a non-hint target less than G past this chunk's end can place it only at
                # its own start, this chunk's end, which is then taken here: after a budget cut rounded down to the
                # grid, nothing else makes it a checkpoint. (A hint, or a refiner's target, ends its own chunk, so
                # targets after one do not start at this end.)
                if exact or d - end >= self.grid:
                    break
                c = end
            else:
                c = d if exact else self._grid_point(start, end, d)  # a hint inside is on the grid
            if self.bidi:
                c = self._ckpt_pos(req, c, start, end, d)
            if c <= start:
                continue  # within G − 1 above the chunk's start, which is a checkpoint (the hit, a hint, or above)
            node = cache.node_ending_at(deep, c)
            if node.ckpt:
                continue
            if self.window:  # a window of the request's own pages: nothing to copy, no plan record
                cache.attach(node, self._make_window(req, c), step, aid, src, on_precise=req.base_precise)
                self.ckpts_taken[src] += 1
            else:
                dst = cache.alloc_checkpoint(step)
                if not dst:
                    self.ckpts_skipped += 1
                    continue
                cache.attach(node, dst, step, aid, src, on_precise=req.base_precise)
                self.builder.add_ckpt(row, c - start, dst)
                self.ckpts_taken[src] += 1
            if src in ("prefill", "branch"):
                req.own_ckpts.append((node, node.ckpt, step))
            if req.unv >= 0:
                req.unv_ckpts.append((node, node.ckpt, step))
            self._supersede(req, node)  # a deeper checkpoint of its own: the ones above it are superseded

    def _relock(self, req: Req, node: Node) -> None:
        """Lock `node`, the deepest the request holds in its namespace (the FP8 one: `lock`; the precise prefix's:
        `plock`), and release the one it held there before."""
        held = req.plock if is_precise(node.ns) else req.lock
        if node is held:
            return
        self.cache.lock(node)
        if held is not None:
            self.cache.unlock(held)
        if is_precise(node.ns):
            req.plock = node
        else:
            req.lock = node

    # -- results -------------------------------------------------------------------------------------------------

    def process(self, inf: Inflight, tokens: np.ndarray, n_out: np.ndarray, gpu_busy_ns: int,
                status: np.ndarray | None = None, drafts: np.ndarray | None = None,
                logprobs: Any = None) -> dict[int, FrontendOutputs]:
        """Fold one step's results in (vectorized) and return the outputs for each frontend. `drafts` (MTP): each
        SAMPLE row's drafts for its slot's next step (kept for the masks of structured output). `logprobs`: the
        LOGPROBS rows' values (`runner.base.LogprobRows`), which go out with their tokens."""
        slots = inf.slots
        valid = self.aid[slots] == inf.aids
        self._done_step = inf.step_id
        if self.count_policy is not None and inf.decode_only and len(slots) and not inf.n_encode:
            self.count_policy.observe_cost(len(slots), self._v(inf), self._d(inf), gpu_busy_ns)
        if self._orphans:  # unverified writes nothing will verify (`_admit_one`); the plans that read them are built
            for nss, ns, key, aids, ckpts, lo, pages, spans in self._orphans:
                self._remove_unverified(nss, ns, key, aids, ckpts, lo, spans)
                if pages is not None:
                    self._unmark(pages, aids)
            self._orphans.clear()
        bad = nan = None
        if status is not None:
            bad = (status & (NONFINITE | MASK_EMPTY)) != 0
            nan = (status & NONFINITE) != 0
        bad_aids = set(inf.aids[nan].tolist()) if nan is not None and nan.any() else set()
        if self.cache is not None:
            self.cache.complete(inf.step_id, bad_aids)  # confirms this step's writes, or rolls back the NaN ones
        if self._n_unv:  # a final chunk's finite SAMPLE row (an empty mask's too) read all its request wrote
            ok = valid & inf.sample & ~inf.device if nan is None else valid & inf.sample & ~inf.device & ~nan
            for slot in slots[ok].tolist():
                req = self.slot_req[slot]
                if req.unv >= 0:
                    self._clear_unverified(req)
                    if req.dup:
                        self._adopt_verified(req, inf.step_id)
        if bad_aids:
            rolled = {(a, inf.step_id) for a in bad_aids}
            for slot in slots[valid & nan].tolist():
                rolled |= self._fail(self.slot_req[slot])
            self._fail_cascade(rolled)
            valid = self.aid[slots] == inf.aids
        if bad is not None and (valid & bad & ~nan).any():  # an empty grammar mask: the request fails alone
            for slot in slots[valid & bad & ~nan].tolist():
                self._fail(self.slot_req[slot], rollback=False)
            valid = self.aid[slots] == inf.aids
        dropped = inf.sample & ~valid
        if dropped.any():
            n_drop = int(dropped.sum())
            if self._held:
                n_drop -= self._deliver_held(inf, tokens, n_out, valid, bad, nan, gpu_busy_ns, logprobs)
            self.dropped_samples += n_drop
        if bad_aids or self._removed:
            self._mark_nocache()
        live = slots[valid]
        if len(live):
            waiting_first = live[~self.has_first[live]]
            if len(waiting_first):
                self.runner_ns[waiting_first] += gpu_busy_ns
        dm = valid & inf.device
        if dm.any():
            ds = slots[dm]
            self.mirror[ds] += n_out[dm]
            self.dev_inflight[ds] -= 1
            if self.k:
                self._count_verify(inf, np.flatnonzero(dm), n_out[dm], tokens)
        m = valid & inf.sample
        s = slots[m]
        if not len(s):
            self._decode_ckpts_done(inf)
            return self._outputs(_E64, _E32, _E32, None, None, inf.step_id)
        self.pending[s] -= 1
        if self.k and drafts is not None:
            self.drafts[s] = drafts[m]
        # The logprobs of the rows' tokens, laid out as they are [rows, 1 + k] (None: no request among them asked)
        lp = self._lp_rows(logprobs, m) if logprobs is not None and (self.lp[s] >= 0).any() else None
        if self.k:
            delivered, tok, finish = self._cut(s, tokens[m], n_out[m].astype(np.int64))
            if lp is not None:
                keep = np.arange(tokens.shape[1])[None, :] < delivered[:, None]
                lp = (lp[0][keep], lp[1][keep], lp[2][keep])
            dev = inf.device[m]
            if dev.any():  # verify rows; a final chunk's sample is its request's first token, or a readmission's next
                self.verify_rows[s[dev]] += 1
                self.verify_tokens[s[dev]] += delivered[dev]
            if not dev.all():
                self.resampled += int(delivered[~dev & self.has_first[s]].sum())
        else:
            tok = tokens[m, 0].astype(np.int32)
            delivered = None
            if lp is not None:
                lp = (lp[0][:, 0], lp[1][:, 0], lp[2][:, 0])
            n = self.n_out[s] + 1
            self.n_out[s] = n
            col = (n - 1) % _OUT_W
            self.out_buf[s, col] = tok
            for i in np.flatnonzero(col == _OUT_W - 1).tolist():  # a full row (`Req.out_chunks`; `_flush_full`)
                self.slot_req[s[i]].out_chunks.append(self.out_buf[s[i]].copy())
            stop = among(tok, self._eos) & ~self.ignore_eos[s]
            if self._n_stop_reqs:
                stop |= (self.stop_ids[s] == tok[:, None]).any(axis=1)
            length = n >= self.max_out[s]
            finish = np.where(stop, np.uint8(FinishReason.STOP),
                              np.where(length, np.uint8(FinishReason.LENGTH), np.uint8(0))).astype(np.uint8)
            self.generation_tokens += len(s)
        if self.n_masked:
            rows = np.flatnonzero(self.masked[s])
            if len(rows):
                self._advance_grammars(s, rows, tok, delivered, finish)
        # Output tokens the frontend receives with these (`Req.unrep`): before a row's, the forced prefix its
        # final chunk took (or a preemption's unreported tokens); after them, the EOS an EOS-free finish appends
        pre = post = None
        if self._n_unrep:
            ur = np.flatnonzero(self.unrep[s])
            if len(ur):
                pre = {}
                for i in ur.tolist():
                    slot = int(s[i])
                    req = self.slot_req[slot]
                    pre[i] = req.unrep
                    self.generation_tokens += len(req.unrep)
                    req.unrep = []
                    self._clear_unrep(slot)
        if self.eos_free_finish and self.n_masked:
            post = self._eos_free(inf, s, finish)
        if pre or post:
            tok, delivered, lp = self._amend(tok, delivered, len(s), pre or {}, post or [], lp)
        first = ~self.has_first[s]
        first_idx = np.flatnonzero(first) if first.any() else None
        if first_idx is not None:
            self.has_first[s[first_idx]] = True
        self._decode_ckpts_done(inf)
        out = self._outputs(s, tok, finish, first_idx, delivered, inf.step_id, lp)
        done = np.flatnonzero(finish)
        for i in done.tolist():
            self._finish(int(s[i]))
        return out

    def _count_verify(self, inf: Inflight, rows: np.ndarray, n_out: np.ndarray, tokens: np.ndarray) -> None:
        """The speculative-decoding counters (`vllm:spec_decode_*`; one draft is one request's verify step) and the
        verify-mask counters for the MTP verify rows `rows` (plan indices) whose results their requests get, with
        `n_out` tokens each: the rows of running requests (`process`) and those held for preempted ones
        (`_deliver_held`)."""
        acc = n_out.astype(np.int64) - 1
        v = self._v(inf)
        self.spec_drafts += len(rows)
        # the drafts counted are those the grammar takes (`fill_masks`); under ignore_eos those after a drafted EOS are
        # verified unconstrained, so a row counts at least the drafts it accepted
        self.spec_draft_tokens += len(rows) * v if inf.invalid is None else \
            int(np.maximum(v - inf.invalid[rows], acc).sum())
        self.spec_accepted += int(acc.sum())
        self.spec_accepted_per_pos[:v] += (acc[:, None] > np.arange(v)[None, :]).sum(axis=0)
        self.spec_verified_per_pos[:v] += len(rows)
        if self.count_policy is not None:
            self.count_policy.observe_accept(v, acc)
        if inf.wpop is not None:
            self._rejections(inf, rows, acc, tokens)

    def _v(self, inf: Inflight) -> int:
        """The drafts the plan's device rows verified (`Inflight.v`; −1: k)."""
        return self.k if inf.v < 0 else inf.v

    def _d(self, inf: Inflight) -> int:
        return self.k if inf.d < 0 else inf.d

    def _eos_free(self, inf: Inflight, s: np.ndarray, finish: np.ndarray) -> list[int]:
        """The EOS-free finish (architecture.md, "Grammar-aware heads"): the rows (indices into `s`) whose request's
        grammar stopped with this step's tokens get the grammar's EOS now, counted, and finish, instead of a step
        sampling it; `finish` is set in place. Not when the request may not take it (ignore_eos, max_tokens, stop
        strings), nor when the step in flight samples that EOS and would run for nothing else (the gate): when no row of
        it (a prefill chunk's included) belongs to a request that stays (the requests finishing in this call leave), the
        first such request waits for its row, and the others may go, since the step then runs for it."""
        eos = self._grammar_eos
        cand = np.flatnonzero((finish == 0) & self.masked[s] & ~self.ignore_eos[s] & (self.n_out[s] < self.max_out[s]))
        if eos < 0 or not len(cand):
            return []
        f = self._last_inf if self._last_inf is not None and self._last_inf.step_id > inf.step_id else None
        stays: set[int] = set()  # slots with a row in the step in flight (a prefill chunk's too) whose request stays
        if f is not None:
            live = self.aid[f.slots] == f.aids
            stays = set(f.slots[live].tolist()) - set(s[finish != 0].tolist())
        stopped = [i for i in cand.tolist()
                   if not (r := self.slot_req[int(s[i])]).stop_strings and r.matcher.is_stopped()]
        stays -= {int(s[i]) for i in stopped}
        out = []
        for i in stopped:
            slot = int(s[i])
            in_flight = f is not None and bool(((f.slots == slot) & (f.aids == self.aid[slot]) & f.sample).any())
            if in_flight and not stays:
                self.eos_free_gated += 1
                stays.add(slot)
                continue
            out.append(i)
            self._ring_push(slot, eos)
            self.generation_tokens += 1
            finish[i] = FinishReason.STOP
            self.eos_free[int(in_flight)] += 1
        return out

    def _ring_push(self, slot: int, t: int) -> None:
        """Append one output token to the slot's ring (flushing a full row to the request)."""
        n = int(self.n_out[slot])
        c = n % _OUT_W
        self.out_buf[slot, c] = t
        self.n_out[slot] = n + 1
        if c == _OUT_W - 1:
            self._flush_full(slot)

    def _flush_full(self, slot: int) -> None:
        """The slot's ring row just filled: a copy goes to its request's `out_chunks` (`Req.out_chunks`), as the row
        is written over next."""
        self.slot_req[slot].out_chunks.append(self.out_buf[slot].copy())

    def _amend(self, tok: np.ndarray, n_new: np.ndarray | None, rows: int, pre: dict[int, list[int]],
               post: list[int], lp: tuple | None = None) -> tuple[np.ndarray, np.ndarray, tuple | None]:
        """The step's output tokens (flattened, `n_new` per row; None: one each) with `pre[i]` before row i's and the
        grammar's EOS after those of each row in `post` (the forced prefix, the EOS-free finish); `lp`: their logprobs,
        amended the same way (`forced_logprobs` for the tokens added)."""
        n = np.ones(rows, np.int64) if n_new is None else n_new.astype(np.int64)
        off = np.concatenate([[0], np.cumsum(n)])
        parts, new, lparts = [], n.copy(), []
        tail = set(post)
        for i in range(rows):
            p = pre.get(i)
            if p:
                parts.append(np.asarray(p, np.int32))
                new[i] += len(p)
                if lp is not None:
                    lparts.append(forced_logprobs(p))
            parts.append(tok[off[i]:off[i + 1]])
            if lp is not None:
                lparts.append(tuple(a[off[i]:off[i + 1]] for a in lp))
            if i in tail:
                parts.append(np.asarray([self._grammar_eos], np.int32))
                new[i] += 1
                if lp is not None:
                    lparts.append(forced_logprobs([self._grammar_eos]))
        return np.concatenate(parts).astype(np.int32), new.astype(np.int32), _lp_cat(lparts) if lp is not None else None

    def _lp_rows(self, logprobs: Any, m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The LOGPROBS rows' values (`runner.base.LogprobRows`) of the plan rows `m` selects, [rows, 1 + k] as their
        tokens; a row without them (its request asked for none) unused."""
        w = logprobs.token.shape[1]
        at = np.full(len(m), -1, np.int64)
        at[logprobs.rows] = np.arange(len(logprobs.rows))
        j = at[np.flatnonzero(m)]
        has = j >= 0
        tok = np.full((len(j), w), np.nan, np.float32)
        ids = np.zeros((len(j), w, MAX_LOGPROBS), np.int32)
        top = np.full((len(j), w, MAX_LOGPROBS), np.nan, np.float32)
        tok[has], ids[has], top[has] = logprobs.token[j[has]], logprobs.top_ids[j[has]], logprobs.top[j[has]]
        return tok, ids, top

    def _rejections(self, inf: Inflight, rows: np.ndarray, acc: np.ndarray, tokens: np.ndarray) -> None:
        """The verify-mask counters (`llm_mini_infer:spec_first_rejections_total`,
        `llm_mini_infer:spec_window_mask_tokens`) for this step's verify rows `rows` (plan indices, live) with `acc`
        accepted drafts each: the popcounts of the masks verification read (the samples after the pending token and
        after each accepted draft), and the class of the mask the first rejected draft's position was sampled under. A
        draft equal to its target sample, refused only because that sample ends the request (a stop token, or
        max_tokens: the commit limits), is no rejection."""
        k = self._v(inf)
        pop = inf.wpop[rows]
        has = pop[:, 0] >= 0
        if not has.any():
            return
        rows, acc, pop = rows[has], acc[has], pop[has]
        p = pop[np.arange(1 + k)[None, :] <= acc[:, None]]
        self.window_masks += np.bincount(np.searchsorted(POPCOUNT_BUCKETS, p), minlength=len(self.window_masks))
        self.window_masks_sum += int(p.sum())
        rej = acc < k
        if rej.any():
            r, a = rows[rej], acc[rej]
            real = inf.wdraft[r, a] != tokens[r, a]
            if real.any():
                cls = inf.wcls[r[real], a[real]].astype(np.int64)
                self.first_rejections += np.bincount(cls, minlength=len(MASK_CLASSES))

    def _cut(self, s: np.ndarray, toks: np.ndarray, n_tok: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """MTP rows of up to 1 + k tokens: deliver each row's tokens up to its first finish (EOS, a stop token, or
        `max_out`), append them to the slots' output rings, and return (tokens per row, the tokens flattened, finish
        reasons). Tokens after a finish (only for a stop token the runner did not know) mark the slot `overrun`.
        Vectorized over [rows, 1 + k]; only a row whose ring fills in this step takes a Python path."""
        w = toks.shape[1]
        n0 = self.n_out[s]
        col = np.arange(w)[None, :]
        valid = col < n_tok[:, None]
        stop = among(toks, self._eos) & ~self.ignore_eos[s][:, None]
        if self._n_stop_reqs:
            stop |= (self.stop_ids[s][:, :, None] == toks[:, None, :]).any(axis=1)
        length = n0[:, None] + col + 1 >= self.max_out[s][:, None]
        cut, finish, ended = self._finish_at(stop, length, valid, n_tok)
        if ended:
            over = cut < n_tok
            if over.any():
                self.overrun[s[over]] = True
                self.num_overruns += int(over.sum())
                self.spec_trimmed += int((n_tok - cut)[over].sum())
        keep = col < cut[:, None]
        start = n0 % _OUT_W
        wraps = np.flatnonzero(start + cut > _OUT_W)  # rows whose ring wraps in this step (rare): in order
        if len(wraps):
            for i in wraps.tolist():
                slot = int(s[i])
                for j in range(int(cut[i])):
                    c = (int(n0[i]) + j) % _OUT_W
                    self.out_buf[slot, c] = toks[i, j]
                    if c == _OUT_W - 1:
                        self._flush_full(slot)
            keep[wraps] = False
        rr, cc = np.nonzero(keep)
        if len(rr):
            self.out_buf[s[rr], start[rr] + cc] = toks[rr, cc]
            for i in np.flatnonzero(start + cut == _OUT_W).tolist():  # rows exactly full now (`_flush_full`, inline)
                self.slot_req[int(s[i])].out_chunks.append(self.out_buf[int(s[i])].copy())
        self.n_out[s] = n0 + cut
        self.generation_tokens += int(cut.sum())
        return cut.astype(np.int32), toks[col < cut[:, None]].astype(np.int32), finish

    @staticmethod
    def _finish_at(stop: np.ndarray, length: np.ndarray, valid: np.ndarray, n_tok: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray, bool]:
        """The finish rule over rows of tokens ([rows, n] masks): a row ends at its first valid token that is a stop
        (an EOS unless ignore_eos, or a stop id: `stop`) or reaches max_tokens (`length`), STOP winning over LENGTH at
        the same token. Returns the tokens each row keeps (all its `n_tok` when it does not end), each row's finish
        reason (0: none) and whether any row ends. `_cut` (running requests' MTP rows) and `_deliver_held` (a
        preempted request's held sample) share it; `process` applies it inline to a step of one token a row."""
        hit = (stop | length) & valid
        any_hit = hit.any(axis=1)
        first = hit.argmax(axis=1)
        cut = np.where(any_hit, first + 1, n_tok)
        finish = np.zeros(len(hit), np.uint8)
        ended = bool(any_hit.any())
        if ended:
            r = np.flatnonzero(any_hit)
            finish[r] = np.where(stop[r, first[r]], np.uint8(FinishReason.STOP), np.uint8(FinishReason.LENGTH))
        return cut, finish, ended

    def _deliver_held(self, inf: Inflight, tokens: np.ndarray, n_out: np.ndarray, valid: np.ndarray,
                      bad: np.ndarray | None, nan: np.ndarray | None, gpu_busy_ns: int, logprobs: Any = None) -> int:
        """Samples of requests preempted while this step was in flight (module docstring): append them to the waiting
        request up to a finish (which finishes it), and queue its output row; they verify its unverified writes. A
        NONFINITE row's samples are dropped: its writes were rolled back, its unverified writes are removed, and the
        readmission recomputes. An empty mask's sample (an invalid token) is dropped too. A delivered verify row counts
        as a running request's does (`_count_verify`, the request's verify rows), and its logprobs go with its tokens
        (`logprobs`). Returns the rows delivered."""
        delivered = 0
        for i in np.flatnonzero(inf.sample & ~valid).tolist():
            req = self._held.pop(int(inf.aids[i]), None)
            if req is None:
                continue
            req.held = 0
            if req.status != WAITING:
                continue  # aborted meanwhile
            if bad is not None and bad[i]:
                if nan[i]:
                    self._settle_unverified(req, Unverified.DROP)  # it waits on: its readmission recomputes
                continue
            delivered += 1
            self._settle_unverified(req, Unverified.KEEP)
            n = max(1, int(n_out[i]))
            row = tokens[i:i + 1, :n]  # its tokens up to its first finish (`_finish_at`, as a running row's)
            m = row.shape[1]
            stop = among(row, self._eos) & (not req.ignore_eos)
            if len(req.stop_ids):
                stop |= (req.stop_ids[:, None] == row[0]).any(axis=0)
            length = req.n_out + np.arange(1, m + 1) >= req.max_out
            cut, fin, _ = self._finish_at(stop, length[None, :], np.ones((1, m), bool), np.array([m]))
            new: list[int] = row[0, :int(cut[0])].tolist()
            finish = int(fin[0])
            if self.k and inf.device[i]:
                self._count_verify(inf, np.array([i]), n_out[i:i + 1], tokens)
                self.spec_trimmed += n - len(new)
                req.verify_rows += 1
                req.verify_tokens += len(new)
            elif self.k and req.first_done:  # a readmission's final chunk
                self.resampled += len(new)
            gen = req.out_chunks[0] if req.out_chunks else _E32  # _preempt left one chunk with every token
            arr = np.asarray(new, np.int32)
            req.out_chunks = [np.concatenate([gen, arr])]
            req.tokens = np.concatenate([req.tokens, arr])
            req.key64 = None
            req.n_out += len(new)
            self.generation_tokens += len(new)
            first = None
            if not req.first_done:
                req.runner_ns += gpu_busy_ns
                req.first_done = True
                first = req
            lp = None
            if req.logprobs >= 0 and logprobs is not None:
                j = np.flatnonzero(logprobs.rows == i)
                if len(j):
                    j, c = int(j[0]), len(new)
                    lp = (logprobs.token[j, :c], logprobs.top_ids[j, :c], logprobs.top[j, :c])
            if req.unrep:  # its forced prefix (already among its tokens) goes first
                self.generation_tokens += len(req.unrep)
                if lp is not None:
                    lp = _lp_cat([forced_logprobs(req.unrep), lp])
                new, req.unrep = req.unrep + new, []
            spec = (req.verify_rows, req.verify_tokens) if finish and self.k else None
            self._held_rows.append((req.frontend, req.rid, new, int(finish), first, spec,
                                    (lp, req.logprobs) if lp is not None else None))
            if finish:
                self._to_finished(req)  # still waiting
        return delivered

    def _decode_ckpts_done(self, inf: Inflight) -> None:
        """Decode checkpoints of this step: insert the KV they sit on and attach them where the step's commit ended
        (the slot's `mirror`, already advanced by this step's `n_out`), or give their slot back (the request left
        since the plan, or its device commit ran past a finish: the state is past its end; or it caches nothing
        more)."""
        iv = self.ckpt_interval
        for slot, aid, dst in inf.decode_ckpts:
            self._ckpt_inflight.discard(dst)
            req = self.slot_req[slot]
            if self.aid[slot] != aid or req is None or self.overrun[slot]:
                if dst:
                    self.states.release(dst)
                continue
            if req.nocache:
                if dst:
                    self.states.release(dst)
                self.ckpt_pending[slot] = False
                continue
            pos = int(self.mirror[slot])
            self.ckpt_pending[slot] = False
            self.next_ckpt[slot] = (pos // iv + 1) * iv
            key = self._seq_tokens(req, slot, pos)
            before = self.cache.kept_dups
            deep = self.cache.insert(req.ns, key, pos, req.pages, inf.step_id, aid, req.cow, req.hit, req.spans)
            if self.cache.kept_dups != before:
                req.dup, req.dup_swaps = True, self.cache.swaps
            self._relock(req, deep)
            node = self.cache.node_ending_at(deep, pos)
            if node.ckpt:
                if dst:
                    self.states.release(dst)
                continue
            if self.window:  # the request still holds its pages before pos (`_wrelease` keeps them until then)
                if req.decode_ckpt is not None:  # one decode checkpoint per running request
                    self.cache.drop_checkpoint(*req.decode_ckpt)
                dst = self._make_window(req, pos)
            self.cache.attach(node, dst, inf.step_id, aid, "decode", on_precise=req.base_precise)
            self.ckpts_taken["decode"] += 1
            req.decode_ckpt = (node, dst, inf.step_id)
            self._supersede(req, node)

    def _outputs(self, s: np.ndarray, tok: np.ndarray, finish: np.ndarray, first_idx: np.ndarray | None,
                 n_new: np.ndarray | None, done_step: int, lp: tuple | None = None) -> dict[int, FrontendOutputs]:
        """Group rows by frontend (one message per frontend per step). `n_new`: tokens per row (None: one each), with
        `tok` holding them flattened in row order, and `lp` their logprobs (None: no request among them asked). Pixel
        releases go out once `done_step` (every step up to it has completed) reaches the step they wait for."""
        extra = self._extra
        self._extra = []
        result: dict[int, FrontendOutputs] = {}
        lp_k: dict[int, int] = {}  # per frontend with logprobs: the most top entries its requests asked
        if len(s):
            rids = self.fe_rid[s]
            fes = self.fe[s] if self.num_frontends > 1 else None
            if first_idx is not None:
                first_slots = s[first_idx]
                first_reqs = [self.slot_req[x] for x in first_slots.tolist()]
                for r, rn in zip(first_reqs, self.runner_ns[first_slots].tolist()):
                    r.first_done = True
                    r.runner_ns = rn
                f_rids = rids[first_idx]
                f_cached = np.fromiter((max(0, r.cached) for r in first_reqs), np.int32, len(first_reqs))
                f_admit = np.fromiter((r.t_admit_ns for r in first_reqs), np.int64, len(first_reqs))
                f_launch = np.fromiter((r.t_launch_ns for r in first_reqs), np.int64, len(first_reqs))
                f_runner = self.runner_ns[first_slots].copy()
                f_queued = np.fromiter((r.t_queued_ns for r in first_reqs), np.int64, len(first_reqs))
                f_fes = self.fe[first_slots] if fes is not None else None
            spec = None
            if self.k and finish.any():  # MTP: the finishing requests' verify rows and tokens
                fi = np.flatnonzero(finish)
                spec = (rids[fi], self.verify_rows[s[fi]], self.verify_tokens[s[fi]],
                        fes[fi] if fes is not None else None)
            if n_new is not None and fes is not None:
                tok_row = np.repeat(np.arange(len(s)), n_new)  # the row of each flattened token
            for fe in (range(self.num_frontends) if fes is not None else (0,)):
                if fes is not None:
                    sel = fes == fe
                    if not sel.any():
                        continue
                    r_, f_ = rids[sel], finish[sel]
                    t_ = tok[sel] if n_new is None else tok[sel[tok_row]]
                    nn = np.ones(len(r_), np.int32) if n_new is None else n_new[sel]
                else:
                    r_, t_, f_ = rids, tok, finish
                    nn = np.ones(len(r_), np.int32) if n_new is None else n_new
                fo = FrontendOutputs(rids=r_, n_new=nn, tokens=t_, finish=f_,
                                     first_rids=_E64, first_cached=_E32, first_t_admit_ns=_E64,
                                     first_t_launch_ns=_E64, first_runner_ns=_E64)
                if first_idx is not None:
                    fsel = slice(None) if f_fes is None else (f_fes == fe)
                    fo.first_rids = f_rids[fsel]
                    fo.first_cached = f_cached[fsel]
                    fo.first_t_admit_ns = f_admit[fsel]
                    fo.first_t_launch_ns = f_launch[fsel]
                    fo.first_runner_ns = f_runner[fsel]
                    fo.first_t_queued_ns = f_queued[fsel].tolist()
                if spec is not None:
                    ssel = slice(None) if spec[3] is None else (spec[3] == fe)
                    fo.spec_rids, fo.spec_rows, fo.spec_tokens = (a[ssel].tolist() for a in spec[:3])
                if lp is not None:
                    k = int(self.lp[s if fes is None else s[sel]].max())
                    if k >= 0:
                        lp_k[fe] = k
                        tsel = slice(None) if fes is None else (sel if n_new is None else sel[tok_row])
                        fo.logprobs = (lp[0][tsel], lp[1][tsel], lp[2][tsel])
                result[fe] = fo
        held = self._held_rows
        if held:
            self._held_rows = []
            for fe, rid, ts, fin, freq, spec, hlp in held:
                fo = result.get(fe)
                if fo is None:
                    fo = result[fe] = FrontendOutputs(rids=_E64, n_new=_E32, tokens=_E32, finish=np.zeros(0, np.uint8),
                                                      first_rids=_E64, first_cached=_E32, first_t_admit_ns=_E64,
                                                      first_t_launch_ns=_E64, first_runner_ns=_E64)
                if hlp is not None or fo.logprobs is not None:  # its tokens' logprobs, after the message's
                    have = fo.logprobs if fo.logprobs is not None else _lp_unused(len(fo.tokens))
                    fo.logprobs = _lp_cat([have, hlp[0] if hlp is not None else _lp_unused(len(ts))])
                    if hlp is not None:
                        lp_k[fe] = max(lp_k.get(fe, -1), hlp[1])
                fo.rids = np.append(fo.rids, np.int64(rid))
                fo.n_new = np.append(fo.n_new, np.int32(len(ts)))
                fo.tokens = np.append(fo.tokens, np.asarray(ts, np.int32))
                fo.finish = np.append(fo.finish, np.uint8(fin))
                if freq is not None:
                    fo.first_rids = np.append(fo.first_rids, np.int64(rid))
                    fo.first_cached = np.append(fo.first_cached, np.int32(max(0, freq.cached)))
                    fo.first_t_admit_ns = np.append(fo.first_t_admit_ns, np.int64(freq.t_admit_ns))
                    fo.first_t_launch_ns = np.append(fo.first_t_launch_ns, np.int64(freq.t_launch_ns))
                    fo.first_runner_ns = np.append(fo.first_runner_ns, np.int64(freq.runner_ns))
                    fo.first_t_queued_ns.append(freq.t_queued_ns)
                if spec is not None:
                    fo.spec_rids.append(rid)
                    fo.spec_rows.append(spec[0])
                    fo.spec_tokens.append(spec[1])
        for fe, rid, reason in extra:
            fo = result.get(fe)
            row = (np.array([rid], np.int64), np.zeros(1, np.int32), np.array([reason], np.uint8))
            if fo is None:
                result[fe] = FrontendOutputs(rids=row[0], n_new=row[1], tokens=_E32, finish=row[2], first_rids=_E64,
                                             first_cached=_E32, first_t_admit_ns=_E64, first_t_launch_ns=_E64,
                                             first_runner_ns=_E64)
            else:
                fo.rids = np.concatenate([fo.rids, row[0]])
                fo.n_new = np.concatenate([fo.n_new, row[1]])
                fo.finish = np.concatenate([fo.finish, row[2]])
        if self._released:
            keep = []
            for fe, rid, after in self._released:
                if after > done_step:
                    keep.append((fe, rid, after))
                    continue
                fo = result.get(fe)
                if fo is None:
                    fo = result[fe] = FrontendOutputs(rids=_E64, n_new=_E32, tokens=_E32, finish=np.zeros(0, np.uint8),
                                                      first_rids=_E64, first_cached=_E32, first_t_admit_ns=_E64,
                                                      first_t_launch_ns=_E64, first_runner_ns=_E64)
                fo.released.append(rid)
            self._released = keep
        for fe, k in lp_k.items():  # each frontend's top entries: as many as its requests asked for at most
            t, ids, top = result[fe].logprobs
            result[fe].logprobs = (t, ids[:, :k], top[:, :k])
        return result

    def take_extra(self) -> dict[int, FrontendOutputs]:
        """Rejections, engine-side aborts and pixel releases to send in a turn without a step result: every step they
        wait for has completed."""
        return self._outputs(_E64, _E32, _E32, None, None, 1 << 62)

    # -- state changes ---------------------------------------------------------------------------------------

    def _finish(self, slot: int) -> None:
        req = self.slot_req[slot]
        self._release(req, keep=True)
        self._to_finished(req)

    def _fail(self, req: Req, rollback: bool = True, cascade: bool = False) -> set[tuple[int, int]]:
        """The request ends with an error. NONFINITE (`rollback`): its writes still in flight are rolled back, and its
        unverified writes removed; returns the (aid, step) pairs of the former (requests that used them must fail too).
        MASK_EMPTY: its token is invalid, not what it computed, so nothing is rolled back. `cascade`: it used another
        request's rolled-back write (`_fail_cascade`)."""
        aid = int(self.aid[req.slot])
        rolled: set[tuple[int, int]] = set()
        if rollback and self.cache is not None:
            for step in range(self.cache.done_step + 1, self.step_id + 1):
                self.cache.drop_writes(step, {aid})
                rolled.add((aid, step))
        self._settle_unverified(req, Unverified.DROP if rollback else Unverified.KEEP)  # before the slot's release
        if rollback:  # in-flight writes are gone: the checkpoints they superseded are live again (conservative)
            for node, slot, step in req.demoted:
                if self.cache._owns(node, slot, step):
                    self.cache.promote(node)
        if self.unrep[req.slot] and not cascade:
            # Its forced prefix, then ERROR: the sampled path's output when it fails where it samples
            # after them (an empty mask, a NaN arising there); no first-token record, as for any failed request
            self.generation_tokens += len(req.unrep)
            self._held_rows.append((req.frontend, req.rid, req.unrep, int(FinishReason.ERROR), None, None,
                                    (forced_logprobs(req.unrep), req.logprobs) if req.logprobs >= 0 else None))
            req.unrep = []
        else:  # a cascade fails the sampled path before any output: the forced prefix is never sent
            req.unrep = []
            self._extra.append((req.frontend, req.rid, FinishReason.ERROR))
        self.num_failed += 1
        self._release(req, keep=False)
        self._drop_images(req, failed=rollback)
        self._to_finished(req)
        if rollback:
            log.warning("request %s failed: non-finite logits or state (its own or its cache hit's)", req.label)
        else:
            log.error("request %s failed: its grammar mask allowed no token", req.label)
        return rolled

    def _fail_cascade(self, bad: set[tuple[int, int]]) -> None:
        """Fail running requests that resumed from a rolled-back write, transitively. One that only locked a removed
        node used none of it: it runs on (`_mark_nocache`)."""
        while bad:
            victims = [r for r in self.slot_req if r is not None and r.deps & bad]
            bad = set()
            for r in victims:
                if r.status == RUNNING:
                    bad |= self._fail(r, cascade=True)

    def _mark_nocache(self) -> None:
        """A rollback or removal took nodes that running requests locked: each runs on but caches nothing more (no
        chunk, decode checkpoint or donation), and lets its locks go. It may hold pages the removal judged unsafe (it
        read them, and its own next SAMPLE row will tell), which an insert would put back. Locks are only retention
        (architecture.md, "Eviction"), and its own would pin duplicates that `_readopt` no longer frees."""
        self._removed = False
        cache = self.cache
        for req in self.slot_req:
            if req is None or req.nocache or not any(lk is not None and lk.dead for lk in (req.lock, req.plock)):
                continue
            req.nocache = True
            for lk in (req.lock, req.plock):
                if lk is not None:
                    cache.unlock(lk)
            req.lock = req.plock = None
            req.dup = False

    def _unverified(self, req: Req, aid: int, ns: Hashable, start: int, end: int) -> None:
        """A chunk [start, end) of the request is planned before a SAMPLE row of it could show finite what it wrote (a
        chunk before its last, or any once it has such writes): mark its pages with its admission
        (`RadixCache.unverified`: the tree adopts them only where its own pages are unverified too). An admission's
        first chunk also wrote its copy of the page holding hit − 1 (`cow`, KV_COPY): the chunk's first page or, under
        MTP with drafter KV, the full page ending at the hit, which goes into any node its insert makes below the hit
        (an FP8 continuation of a precise hit's). That full page is marked too (`cow_unv`), so that a removal takes it,
        when the page it copied was unverified or its drafter KV at hit − 1 is rewritten from an unverified
        checkpoint's hidden vector (`_unverified_ckpt`); a NaN there fails no row (the drafter's statuses are ignored)
        but costs every later reader its drafts. A copy of verified KV and state holds nothing to remove (marked, an
        abort would take the branches others made below it), and one of what a final chunk in flight wrote fails with
        that chunk's NaN (the hit's `deps`), its writes in flight with it."""
        if req.unv < 0:
            self._n_unv += 1
            req.unv = start
        req.unv = min(req.unv, start)
        req.unv_ns.add(ns)
        if aid not in req.unv_aids:
            req.unv_aids.append(aid)
        lo = start - 1 if req.cow_unv and start == req.hit else start  # the copy holds hit − 1 (`_admit_one`)
        pages = req.pages[lo // self.ps:-(-end // self.ps)]
        self.cache.unverified[pages] = aid
        req.unv_pages += pages

    def _unverified_ckpt(self, node: Node) -> bool:
        """`node`'s checkpoint was computed from unverified writes: it is among a request's (`unv_ckpts`), or among
        those of one that left with them, removed at the next `process` (`_orphans`)."""
        if not node.ckpt or not (self._n_unv or self._orphans):
            return False
        lists = [r.unv_ckpts for r in self.reqs.values() if r.unv_ckpts] + [o.ckpts for o in self._orphans]
        return any(n is node and s == node.ckpt and t == node.ckpt_ready for ck in lists for n, s, t in ck)

    def _adopt_verified(self, req: Req, step: int) -> None:
        """A finite SAMPLE row of step `step` verified what the running request wrote: the tree takes its pages where
        it kept its own while they were unverified (`cache.radix`), stamped with that step, which is complete (nothing
        it takes rolls back with a later one)."""
        remain = False
        for lk in (req.lock, req.plock):
            if lk is not None and not lk.dead:
                remain |= self.cache.readopt(lk, req.pages, self._lock_end(req, lk), step, int(self.aid[req.slot]),
                                             req.cow, req.hit)
        req.dup = remain
        if remain:
            req.dup_swaps = self.cache.swaps

    def _clear_unverified(self, req: Req, unmark: bool = True) -> None:
        """The request has no unverified writes any more: a finite SAMPLE row of it read them all, or they are removed
        (`unmark` False: at the next `process`, which needs their marks)."""
        if unmark:
            self._unmark(req.unv_pages, req.unv_aids)
        req.unv = -1
        req.unv_ns.clear()
        req.unv_aids, req.unv_pages, req.unv_ckpts = [], [], []
        self._n_unv -= 1

    def _unmark(self, pages: list[int], aids: list[int]) -> None:
        """Clear the marks these admissions left on these pages (a page freed and reused since holds another's mark,
        or none, which stays)."""
        if pages:
            p = np.asarray(pages, np.int64)
            u = self.cache.unverified
            u[p[np.isin(u[p], aids)]] = 0

    def _settle_unverified(self, req: Req, how: Unverified) -> None:
        """Settle the request's unverified writes, if it has any, as `how` says (`Unverified`). Each caller orders it
        against its other steps: an abort after its slot's release (which keeps no unverified KV), a failure before
        it."""
        if req.unv < 0 or how is Unverified.NONE:
            return
        if how is Unverified.KEEP:
            self._clear_unverified(req)
        elif how is Unverified.DROP:
            self._drop_unverified(req)
        elif how is Unverified.ORPHAN:
            self._orphan_unverified(req)
        else:
            raise RuntimeError(f"unknown unverified disposition {how!r}")

    def _drop_unverified(self, req: Req) -> None:
        """Remove the request's unverified writes (module docstring, Rollback): its row that would verify them is NaN,
        or none will come (an abort). Never while a plan is built (`_orphan_unverified`)."""
        if req.unv >= 0:
            self._remove_unverified(req.unv_ns, req.ns, self._key(req), req.unv_aids, req.unv_ckpts, spans=req.spans)
            self._clear_unverified(req)

    def _orphan_unverified(self, req: Req) -> None:
        """The request failed while a plan is built, which may read its unverified writes: they are removed at the next
        `process`."""
        self._orphans.append(Orphan(set(req.unv_ns), req.ns, self._key(req).copy(), req.unv_aids, req.unv_ckpts, {},
                                    req.unv_pages, req.spans))
        self._clear_unverified(req, unmark=False)

    def _readmit_unverified(self, req: Req, precise: bool, hit: int, node: Node | None) -> None:
        """A readmission of a request with unverified writes resumes at `hit`, in its precise namespace or its FP8 one
        (module docstring, Rollback). Its next finite SAMPLE row reads the pages below the hit there; if it resumes from
        a checkpoint its writes left (its donation, or one its chunks attached), that state was computed from all it
        wrote before it, so it verifies that too. Those stay unverified. The rest nothing will read (what it wrote past
        the hit; its checkpoints below it, and its other namespace there, unless it resumes from its own): it goes at
        the next `process`, once the plan that admits it is built."""
        own = node is not None and any(n is node and s == node.ckpt and t == node.ckpt_ready
                                       for n, s, t in req.unv_ckpts)
        ns, other = (precise_ns(req.ns), req.ns) if precise else (req.ns, precise_ns(req.ns))
        keep = [c for c in req.unv_ckpts if own and c[0].end <= hit]
        self._orphans.append(Orphan(set(req.unv_ns), req.ns, self._key(req).copy(), list(req.unv_aids),
                                    [c for c in req.unv_ckpts if not (own and c[0].end <= hit)],
                                    {ns: hit, other: hit if own else 0}, None, req.spans))
        req.unv_ckpts = keep

    def _remove_unverified(self, nss: set, ns: Hashable, key: np.ndarray, aids: list[int], ckpts: list,
                           lo: dict | None = None, spans: Spans | None = None) -> None:
        """Remove what the admissions `aids` wrote unverified along `key`'s path in the namespaces `nss`, from the
        position `lo` gives each (`RadixCache.remove_unverified`), and the checkpoints `ckpts` they attached, if still
        there. A precise namespace's FP8 sibling `ns` too: an FP8 continuation of a precise hit inserts the precise
        pages below its hit there. The FP8 namespace goes first: the order pages are freed in, and so the plans after,
        never depend on the hash seed."""
        if any(is_precise(x) for x in nss):
            nss = nss | {ns}
        for x in sorted(nss, key=is_precise):
            self._removed |= self.cache.remove_unverified(x, key, aids, lo.get(x, 0) if lo else 0, spans) > 0
        for node, slot, step in ckpts:
            self.cache.drop_checkpoint(node, slot, step)

    def _release(self, req: Req, keep: bool = False, source: str = "finish") -> None:
        """Free a running request's slot, state slot and pages; FREE its slot in the next plan. With `keep` (finish,
        abort, preemption), its KV goes into the cache and its live state becomes the checkpoint at the last position
        it computed, of `source` (finish, or preempt)."""
        slot = req.slot
        assert not any(t[1] == slot for t in self._tails), f"request {req.label}: released with its forced tokens due"
        donated = False
        cache = self.cache
        if cache is not None:
            if keep and not req.nocache and any(lk is not None and not lk.dead for lk in (req.lock, req.plock)):
                donated = self._keep(req, slot, source)
            for lk in (req.lock, req.plock):
                if lk is not None:
                    cache.unlock(lk)
            req.lock = req.plock = None
            req.resumed, req.own_ckpts, req.demoted = None, [], []
        self.kv.release(req.pages)
        req.pages = []
        req.hit = req.cow = 0
        req.cow_unv = False
        if req.refine and self._refining:  # left mid-refine (preempted, aborted or failed)
            self._refined(req)
        if self.window:
            held = [p for p in req.wpages if p]
            if held:
                self.wkv.release(held)
            req.wpages = []
            self.num_wpages[slot] = self.wlo[slot] = 0
            if self.pending_wclear:  # FREE clears its table
                self.pending_wclear = [c for c in self.pending_wclear if c[0] != slot]
        if req.images:
            self._drop_images(req)
        req.matcher = None
        if self.masked[slot]:
            self._unmask(slot)
        if self.lp[slot] >= 0:
            self.lp[slot] = -1
            self.n_lp -= 1
        self.end_at[slot] = NEVER_ENDS
        if self.unrep[slot]:  # its unreported tokens stay with the request (`Req.unrep`: a preemption's readmission)
            self._clear_unrep(slot)
        if not donated:
            self.states.release(req.state_slot)
        self.slots.release(slot)
        if len(req.stop_ids):  # as `_set_stop_ids` counted it
            self._n_stop_reqs -= 1
        self.stop_ids[slot] = -1
        self.aid[slot] = 0
        self.decoding[slot] = False
        self.pending[slot] = 0
        self.dev_inflight[slot] = 0
        self.slot_req[slot] = None
        self.pending_free.append(slot)
        self._order_dirty = True
        req.slot = req.state_slot = 0
        req.deps = set()

    def _keep(self, req: Req, slot: int, source: str) -> bool:
        """Insert what the request computed and donate its live state as the checkpoint there. Returns whether the
        live slot was donated. During plan construction (a preemption), the slot it may evict for the donation is
        never one the plan reads (pinned) or writes with a stamp of its own step. A checkpoint an earlier donation of
        the same construction made is stamped with its earlier step: it can go, though its FOLD is in this plan (it is
        the most recently used, so only if nothing else can); the node then simply loses it."""
        cache = self.cache
        end = int(self.mirror[slot] + self.dev_inflight[slot])  # positions computed once the in-flight steps ran
        step = int(self.last_step[slot])
        aid = int(self.aid[slot])
        if self.overrun[slot]:  # the device committed past the finish: cache the delivered tokens' KV only
            end = min(end, req.prompt_len + int(self.n_out[slot]))
        if end <= 0:
            return False
        key = self._seq_tokens(req, slot, end)
        # What it computed goes into the precise namespace if all of it is precise (a refiner stopped in its PRECISE
        # rows), else into the FP8 one
        ns = precise_ns(req.ns) if self.pns_on and req.pend >= end else req.ns
        deep = cache.insert(ns, key, end, req.pages, step, aid, req.cow, req.hit, req.spans)
        self._relock(req, deep)
        if not self.hybrid or self.overrun[slot]:
            return False
        node = cache.node_ending_at(deep, end)
        if node.ckpt:
            self._supersede(req, node)
            return False
        if self.window:  # its own window pages before `end` become the checkpoint; nothing is donated
            cache.attach(node, self._make_window(req, end), step, aid, source, on_precise=req.base_precise)
            self.ckpts_taken[source] += 1
            if req.unv >= 0:  # computed from its unverified writes: a readmission that resumes here verifies them
                req.unv_ckpts.append((node, node.ckpt, node.ckpt_ready))
            self._supersede(req, node)
            return False
        demoted = self._supersede(req, node)  # first: the donation's eviction takes them before any live one
        if not self.states.donate(req.state_slot) and not (cache.evict_checkpoints(1, self.step_id + 1) and
                                                            self.states.donate(req.state_slot)):
            for n in demoted:  # nothing deeper after all: they stay its conversation's latest
                cache.promote(n)
            self.ckpts_skipped += 1
            return False
        cache.attach(node, req.state_slot, step, aid, source, on_precise=req.base_precise)
        if source == "preempt":
            req.left = (node, node.ckpt, step)
        if req.unv >= 0:
            req.unv_ckpts.append((node, node.ckpt, node.ckpt_ready))
        if self.k:
            # The live state lags a decode step's commit (FOLD in plan.py): fold it to `end` first among the effects of
            # the next plan, which is the first that can read the checkpoint (it is stamped with `step`, which that
            # plan follows; within a plan FOLD runs before STATE_COPY).
            self.pending_fold.append((slot, 1 if self.dev_inflight[slot] else FOLD_ALL))
        self.ckpts_taken[source] += 1
        return True

    def _supersede(self, req: Req, deep: Node) -> list[Node]:
        """`req` left a checkpoint at `deep` on its path: an end-of-prefill or branch checkpoint as its chunk is planned
        (`_cache_chunk`), a decode checkpoint (`_decode_ckpts_done`), or the one at the end of what it computed at its
        finish or preemption (`_keep`). Demote the checkpoints of its conversation that `deep` supersedes
        (architecture.md, "Eviction"), so that they are evicted before any live one. They are the checkpoint it resumed
        from, its end-of-prefill and branch checkpoints and its latest decode checkpoint, each if its node still holds
        it and lies above `deep` on its path in its FP8 namespace, unless another request may still want it: one that
        another request resumed from since it was attached (the one it resumed from: more than once), one a client hint
        or a refine took, one below which the tree forks before any deeper checkpoint (a shared prefix, a branch point:
        `RadixCache.forks`), another request's preemption checkpoint, and its own end-of-prefill checkpoints unless it
        continues where an earlier request stopped (`Req.continues`): a first request's may be the prefix that its
        siblings resume from, and nothing shows that yet. Returns the checkpoints demoted."""
        cache = self.cache
        if not self.demote or deep.ns != req.ns:
            return []
        cands = []
        if req.resumed is not None and (req.resumed[0].ckpt_src != PREEMPT or req.resumed == req.left):
            cands.append((req.resumed, 1))  # another request's preemption checkpoint is its writer's to resume
        cands += [(c, 0) for c in req.own_ckpts if req.continues or c[0].ckpt_src != PREFILL]
        if req.decode_ckpt is not None:
            cands.append((req.decode_ckpt, 0))
        out = []
        for c, own in cands:
            node, slot, step = c
            if (node is not deep and not node.dead and cache._owns(node, slot, step) and node.end < deep.end
                    and node.ns == req.ns and node.served <= own and node.ckpt_src in DEMOTABLE
                    and not cache.forks(node) and cache.demote(node)):
                out.append(node)
                req.demoted.append(c)
        return out

    @staticmethod
    def _key(req: Req) -> np.ndarray:
        """The request's tokens as a radix key: int64, with each image span's positions set to the image's key (a hit
        never crosses an image whose content differs, while the model sees the placeholders)."""
        if req.key64 is None:
            key = req.tokens.astype(KEY_DTYPE)
            for s, e, k in zip(req.sp_start.tolist(), req.sp_end.tolist(), req.sp_key.tolist()):
                key[s:e] = k
            req.key64 = key
        return req.key64

    def _seq_tokens(self, req: Req, slot: int, n: int) -> np.ndarray:
        """The request's first n tokens (prompt, then generated) as a radix key."""
        if n <= len(req.tokens):
            return self._key(req)[:n]
        k = int(self.n_out[slot])
        done = sum(len(c) for c in req.out_chunks)
        i = req.prompt_len
        key = np.empty(min(n, i + k), KEY_DTYPE)  # no concatenation: a batch's rows take decode checkpoints together
        key[:i] = self._key(req)[:i]
        for c in (*req.out_chunks, self.out_buf[slot, :k - done]):
            m = min(len(c), len(key) - i)
            key[i:i + m] = c[:m]
            i += m
        return key

    def _preempt(self, slot: int) -> None:
        req = self.slot_req[slot]
        k = req.n_out = int(self.n_out[slot])  # the slot array is the source of truth while running
        done = sum(len(c) for c in req.out_chunks)
        gen = np.concatenate([*req.out_chunks, self.out_buf[slot, :k - done]]) if k else _E32
        req.out_chunks = [gen] if k else []
        req.tokens = np.concatenate([req.tokens[:req.prompt_len], gen]).astype(np.int32)
        req.key64 = None
        req.runner_ns = int(self.runner_ns[slot])
        req.verify_rows, req.verify_tokens = int(self.verify_rows[slot]), int(self.verify_tokens[slot])
        if self.pending[slot] == 1:  # one step in flight samples for it: keep that token (process delivers it)
            req.held = int(self.aid[slot])
            self._held[req.held] = req
        if self.cache is not None:  # its prefix outlives colder cache entries: the readmission comes soon
            self.cache.touch_path(req.lock)
            self.cache.touch_path(req.plock)
        self._release(req, keep=True, source="preempt")  # its KV and live state: a checkpoint where it stopped
        req.num_preemptions += 1
        self.num_preemptions += 1
        self._to_waiting(req)
        log.debug("preempted request %s after %d tokens", req.label, k)

    def _pick_victim(self, order: np.ndarray) -> int:
        for slot in order[::-1].tolist():
            if self.aid[slot]:
                return slot
        raise RuntimeError("no running request to preempt")

    def _grow(self, slot: int, n: int) -> None:
        if n <= 0:
            return
        pages = self.kv.alloc(n)
        if pages is None:
            raise RuntimeError("page allocation failed after the capacity check")
        if self._n_unv or self._orphans:  # as at admission
            self.cache.unverified[pages] = 0
        req = self.slot_req[slot]
        first = len(req.pages)
        req.pages.extend(pages)
        self.num_pages[slot] = first + n
        self.builder.add_pages(slot, first, pages)

    def _grow_rows(self, slots: np.ndarray, need: np.ndarray, needw: np.ndarray | None) -> None:
        """`_grow`, then `_wgrow`, for each of the decode rows `slots` in turn, at once (a batch whose rows cross a page
        boundary together): one allocation, cut in the order those calls take pages off the free stack (a row's full
        pages, then its window pages), so every page id and plan entry is theirs. Python only extends the requests'
        page tables. Fewer than `GROW_ROWS` rows take those calls: their arrays would cost more."""
        if len(slots) < GROW_ROWS:
            for slot, a, c in zip(slots.tolist(), need.tolist(), [0] * len(slots) if needw is None else needw.tolist()):
                self._grow(slot, a)
                self._wgrow(slot, c)
            return
        n = np.maximum(need, 0).tolist()
        nw = np.maximum(needw, 0).tolist() if needw is not None else [0] * len(n)
        pages = self.kv.alloc(sum(n) + sum(nw))  # `wkv` is `kv` in window mode
        if pages is None:
            raise RuntimeError("page allocation failed after the capacity check")
        fs, fi, fn, fp = [], [], [], []  # full pages: the rows' slots, first new page indices and counts; the pages
        ws, wi, wn, wp = [], [], [], []  # window pages: the same
        i = 0
        for slot, a, c in zip(slots.tolist(), n, nw):
            req = self.slot_req[slot]
            if a:
                got = pages[i:i + a]
                fs.append(slot)
                fi.append(len(req.pages))
                fn.append(a)
                fp += got
                req.pages += got
                i += a
            if c:
                got = pages[i:i + c]
                ws.append(slot)
                wi.append(len(req.wpages))
                wn.append(c)
                wp += got
                req.wpages += got
                i += c
        if fs:
            if self._n_unv or self._orphans:  # as at admission
                self.cache.unverified[fp] = 0
            self.num_pages[fs] = np.add(fi, fn)
            self.builder.add_page_runs(fs, fi, fn, fp)
        if ws:
            self.num_wpages[ws] = np.add(wi, wn)
            self.builder.add_page_runs(ws, wi, wn, wp, window=True)

    # -- the window pool (sliding-window models) ---------------------------------------------------------------

    def _wgrow(self, slot: int, n: int) -> None:
        """n more window pages at the end of the slot's window-pool page table."""
        if n <= 0:
            return
        pages = self.wkv.alloc(n)
        if pages is None:
            raise RuntimeError("window page allocation failed after the capacity check")
        req = self.slot_req[slot]
        first = len(req.wpages)
        req.wpages.extend(pages)
        self.num_wpages[slot] = first + n
        self.builder.add_wpages(slot, first, pages)

    def _wlow(self, q: int) -> int:
        """The first window page a query at position q reads: the one holding max(0, q − W + 1)."""
        return max(0, q - self.window + 1) // self.wps

    def _wrelease(self, slot: int, q: int) -> None:
        """Release the slot's window pages below the first one a query at q reads: its next query is at q or later
        (the caller's row), so nothing it plans reads them again. Steps in flight that still read them run before
        any later plan's (stream order), so the pages are reusable at once."""
        lo = self._wlow(q)
        a = int(self.wlo[slot])
        if lo <= a:
            return
        req = self.slot_req[slot]
        old = [p for p in req.wpages[a:lo] if p]
        req.wpages[a:lo] = [0] * (lo - a)
        if old:
            self.wkv.release(old)
            self.pending_wclear.append((slot, a, lo - a))  # cleared in the runner's table too
        self.wlo[slot] = lo

    def _wslide(self, slots: np.ndarray, q: np.ndarray) -> None:
        """`_wrelease` for decode rows, at once: Python only clears the tables of the rows that cross a page boundary,
        and their pages go back in one release, row after row, as the per-row releases free them; per row only when two
        rows hold the same page (a window both resumed from), which the later of their releases frees."""
        lo = np.maximum(q - self.window + 1, 0) // self.wps
        rows = np.flatnonzero(lo > self.wlo[slots])
        if not len(rows):
            return
        slots, lo = slots[rows], lo[rows]
        old: list[list[int]] = []
        for slot, a, b in zip(slots.tolist(), self.wlo[slots].tolist(), lo.tolist()):
            table = self.slot_req[slot].wpages
            got = [p for p in table[a:b] if p]
            table[a:b] = [0] * (b - a)
            if got:
                old.append(got)
                self.pending_wclear.append((slot, a, b - a))  # cleared in the runner's table too
        flat = [p for got in old for p in got]
        if len(set(flat)) == len(flat):
            self.wkv.release(flat)
        else:
            for got in old:
                self.wkv.release(got)
        self.wlo[slots] = lo

    def _make_window(self, req: Req, c: int) -> int:
        """A window checkpoint at position c from the request's own window pages: the ones holding positions
        [c − W + 1, c), which it still holds (it releases a page only once its next query no longer sees it)."""
        lo, hi = self._wlow(c), -(-c // self.wps)
        pages = req.wpages[lo:hi]
        if len(pages) != hi - lo or 0 in pages:
            raise RuntimeError(f"request {req.label}: window at {c} needs pages [{lo}, {hi}) it does not hold "
                               f"({len(req.wpages)} entries)")
        return self.windows.make(lo, pages)

    def _unmask(self, slot: int) -> None:
        """The slot takes no grammar mask any more (its grammar ended, or its request left): the flag and its count."""
        self.masked[slot] = False
        self.n_masked -= 1

    def _clear_unrep(self, slot: int) -> None:
        """The slot flags no unreported tokens any more (they went with its result, or stay with its request as it
        leaves the slot): the flag and its count."""
        self.unrep[slot] = False
        self._n_unrep -= 1

    def _set_stop_ids(self, slot: int, ids: np.ndarray) -> None:
        if not len(ids):
            return
        if len(ids) > self.stop_ids.shape[1]:
            wider = np.full((self.stop_ids.shape[0], len(ids)), -1, np.int32)
            wider[:, :self.stop_ids.shape[1]] = self.stop_ids
            self.stop_ids = wider
        self.stop_ids[slot, :len(ids)] = ids
        self._n_stop_reqs += 1

    def _running_order(self) -> np.ndarray:
        if self._order_dirty:
            run = np.flatnonzero(self.aid > 0)
            self._order = run[np.lexsort((self.admit_order[run], self.prio[run]))]  # by priority, then arrival
            self._order_dirty = False
        return self._order

    # -- a request's state (module docstring) ---------------------------------------------------------------------

    def _to_waiting(self, req: Req) -> None:
        """`req` waits: added, its grammar built (it leaves COMPILING), preempted, or deferred by `_admit` (popped,
        still WAITING)."""
        if req.status == COMPILING:
            self._leave_compiling(req)
        req.status = WAITING
        key = req.priority if self.policy == "priority" else 0
        heapq.heappush(self._waiting, (key, req.arrival, req))
        self.num_waiting += 1

    def _to_running(self, req: Req) -> None:
        """`req`, the heap's head, was admitted (`_admit_one` made it RUNNING, with its slot): it leaves the heap and
        the count."""
        self._dequeue_head(req)
        self.num_waiting -= 1

    def _to_finished(self, req: Req) -> None:
        """`req` ends (finished, failed or aborted), waiting or running: it leaves `reqs`, and the count if it waited
        (its heap entry is dropped when it reaches the head); its pixels go once the steps planned so far complete."""
        self.reqs.pop(req.key, None)
        if req.status == WAITING:
            self.num_waiting -= 1
        elif req.status == COMPILING:  # its build goes on, never cancelled: it may be shared, and fills the cache
            self._leave_compiling(req)
        req.status = FINISHED
        self._release_pixels(req)

    def _leave_compiling(self, req: Req) -> None:
        del self._compiling[req]
        req.future = None
        self.compiling_moved = True

    def _dequeue_head(self, req: Req) -> None:
        """Pop the waiting heap's head, `req` (RuntimeError if it is not), which `_admit_one` admitted or finished, so
        that `_pop_waiting` would skip it."""
        if self._waiting[0][2] is not req:
            raise RuntimeError(f"request {req.label}: not the waiting heap's head")
        heapq.heappop(self._waiting)

    def _waiting_reqs(self) -> Iterator[Req]:
        """The waiting requests: the heap's live entries (`check_accounting` walks the heap on its own)."""
        return (r for _, _, r in self._waiting if r.status == WAITING)

    def _peek_waiting(self) -> Req:
        while self._waiting[0][2].status != WAITING:
            heapq.heappop(self._waiting)
        return self._waiting[0][2]

    def _pop_waiting(self) -> Req:
        req = self._peek_waiting()
        heapq.heappop(self._waiting)
        self.num_waiting -= 1
        return req

    # -- observability ---------------------------------------------------------------------------------------

    @property
    def kv_usage(self) -> float:
        """Pages that running requests reference (`vllm:kv_cache_usage_perc`: cached-only pages count as free); in
        window mode, pages of either kind of the one arena."""
        if self.cache is None:
            return self.kv.usage
        cached = self.cache.tree_refs[1:] + (self.windows.refs[1:] if self.window else 0)
        used = np.count_nonzero(self.kv.refcount[1:] > cached)
        return used / (self.kv.num_pages - 1)

    @property
    def state_usage(self) -> float:
        return self.states.live_usage

    @property
    def checkpoint_usage(self) -> float:
        """Checkpoint slots in use; in window mode, the share of the arena's pages only windows hold."""
        if self.window:
            only = (self.kv.refcount[1:] == self.windows.refs[1:]) & (self.windows.refs[1:] > 0)
            return float(np.count_nonzero(only)) / (self.kv.num_pages - 1)
        return self.states.ckpt_usage

    def check_accounting(self, full: bool = False) -> None:
        """Tests: page refcounts equal the cache's references plus the running requests' page tables, and every
        state slot in use is a live slot of a running request, a checkpoint of the tree or a decode checkpoint in
        flight. Window mode: the arena's refcounts also count window pages (the running requests' window tables and
        the windows), and no page is a full page and a window page at once. The slots' flags match their counts and
        their phases. Between turns only. `full`: the reference implementation of every check, and the waiting heap's
        walk. The default counts the page tables at once (`_page_tables`) and walks the radix tree only when it
        changed (`RadixCache.check`), and fails where the reference fails, with the same error
        (`tests/sched/test_accounting_checks.py`), the heap's walk aside."""
        held, wheld, live = self._page_tables(full)
        tree = self.cache.tree_refs if self.cache is not None else 0
        expect = held + tree
        if self.window:
            wins = wheld + self.windows.refs
            both = ((expect > 0) & (wins > 0))[1:]
            assert not both.any(), f"pages {(np.flatnonzero(both) + 1).tolist()} are full and window pages at once"
            expect = expect + wins
            assert self.windows.num_windows == (self.cache.num_ckpts if self.cache is not None else 0), \
                "windows leaked"
        expect[0] = 1
        assert np.array_equal(self.kv.refcount, expect), "page refcounts != cache references + page tables"
        assert self.kv.num_free == int((self.kv.refcount == 0).sum()), "free list out of sync"
        used = set(np.flatnonzero(self.states.used).tolist())
        assert live <= used and not live & self._ckpt_inflight, "live state slots out of sync"
        n_ckpts = self.cache.num_ckpts if self.cache is not None and not self.window else 0
        assert len(used - live - self._ckpt_inflight) == n_ckpts, "state slots leaked"
        assert self._n_unv == sum(r.unv >= 0 for r in self.reqs.values()), "unverified writes out of sync"
        assert all(r.status == RUNNING or (r.resumed is None and not r.own_ckpts) for r in self.reqs.values()), \
            "a request out of its slot holds checkpoints to supersede"
        # The COMPILING requests (module docstring, A request's state): each in the compiling set with its build's
        # future, unsettled and in flight, and nothing admitted. A build in flight that no request waits for any more
        # (aborted, timed out) is settled by the `collect` after its job ends
        assert self.num_compiling == sum(r.status == COMPILING for r in self.reqs.values()), \
            "compiling count out of sync"
        if self._compiling:
            g = self.grammars
            flying = {id(f) for f in g._inflight.values()} if g is not None else set()
            for r in self._compiling:
                assert r.status == COMPILING and self.reqs.get(r.key) is r, \
                    f"request {r.label}: compiling set out of sync"
                assert r.future is not None and not r.future.done() and id(r.future) in flying, \
                    f"request {r.label}: compiling, but its build is not in flight"
                assert not (r.slot or r.state_slot or r.pages or r.wpages or r.compiled is not None) and \
                    r.lock is None and all(e is None for e in r.enc), f"request {r.label}: compiling, but admitted"
        assert self._n_stop_reqs == sum(r is not None and len(r.stop_ids) > 0 for r in self.slot_req), \
            "stop-table count out of sync"
        # The flags' counts, and each slot's phase: a free slot (aid 0) neither decodes nor holds flags; samples and
        # device-input rows in flight only once the final chunk was scheduled, after which nothing is left to prefill
        assert self.n_masked == int(self.masked.sum()), "masked-slot count out of sync"
        assert self.n_lp == int((self.lp >= 0).sum()), "logprobs-slot count out of sync"
        free = self.aid == 0
        assert not (free & (self.decoding | self.masked | (self.lp >= 0))).any(), "a free slot holds a request's state"
        assert not (~self.decoding & ((self.pending != 0) | (self.dev_inflight != 0))).any(), \
            "rows in flight for a slot that is not decoding"
        assert (self.num_sched[self.decoding] == self.total[self.decoding]).all(), "a decoding slot left to prefill"
        if full:  # the waiting heap, lazily cleaned: its live entries are the waiting requests, each once
            heap = [r for _, _, r in self._waiting if r.status == WAITING]
            ids = {id(r) for r in heap}
            assert len(heap) == self.num_waiting, "waiting requests out of sync"
            assert len(ids) == len(heap), "a request is in the waiting heap twice"
            assert ids == {id(r) for r in self.reqs.values() if r.status == WAITING}, "waiting heap out of sync"
            compiling = {id(r) for r in self._compiling}
            assert compiling == {id(r) for r in self.reqs.values() if r.status == COMPILING}, \
                "compiling set out of sync"
            assert not compiling & {id(r) for _, _, r in self._waiting}, "a compiling request in the waiting heap"
        # The forced prefix: a slot flags its request's unreported output tokens, which lie in its ring
        assert self._n_unrep == int(self.unrep.sum()), "unreported tokens out of sync"
        for slot in np.flatnonzero(self.unrep).tolist():
            req = self.slot_req[slot]
            assert req is not None, f"slot {slot}: unreported tokens flagged without a request"
            n = len(req.unrep)
            assert n and n <= self.n_out[slot], "unreported tokens out of sync"
            assert self.out_buf[slot, :n].tolist() == req.unrep, "unreported tokens not in the ring"
        if self.cache is not None:
            marked = np.flatnonzero((self.cache.unverified != 0) & (self.kv.refcount > 0))
            if len(marked):  # a page in use marked unverified: by a request, or an orphan removal, still pending
                pending = {a for r in self.reqs.values() for a in r.unv_aids}
                pending |= {a for o in self._orphans for a in o.aids}
                assert set(self.cache.unverified[marked].tolist()) <= pending, "unverified marks out of sync"
            self.cache.check(full)
        if self.encoder is not None:
            self.encoder.check()
            refs: dict[Hashable, int] = {}
            for req in self.reqs.values():
                for x, e in enumerate(req.enc):
                    if e is not None:
                        assert e.key == req.sp_id[x], f"a request holds encoder entry {e.key!r}: another image or salt"
                        refs[e.key] = refs.get(e.key, 0) + 1
            for e in self.encoder.entries.values():
                held = refs.get(e.key, 0)
                assert e.refs == held, f"encoder entry {e.key!r}: {e.refs} refs, {held} held by requests"
            assert not set(refs) - set(self.encoder.entries), "a request references an evicted encoder entry"

    def _page_tables(self, full: bool) -> tuple[np.ndarray, np.ndarray | None, set[int]]:
        """`check_accounting`: the references the running requests' page tables hold on each page and, in window mode,
        on each window page; and their live state slots. Unless `full`, every table is counted at once where the
        reference, request by request, cannot fail: every page table a list of the pool's pages, every window table in
        sync. Elsewhere, and where counting raises (an id that is no integer, a table of another type), the reference
        runs, and fails as it does."""
        n = self.kv.num_pages
        if not full:
            with contextlib.suppress(Exception):  # the reference below says what is wrong
                reqs = [req for req in self.slot_req if req is not None]
                lists = all(type(req.pages) is list for req in reqs)  # np.asarray reads a list as iterating it does
                held = page_counts([p for req in reqs for p in req.pages], n) if lists else None
                wheld = None
                if self.window and all(len(req.wpages) == self.num_wpages[req.slot]
                                       and not any(req.wpages[:self.wlo[req.slot]]) for req in reqs):
                    wheld = page_counts([p for req in reqs for p in req.wpages if p], n)
                if held is not None and (wheld is not None or not self.window):
                    return held, wheld, {req.state_slot for req in reqs}
        held = np.zeros(n, np.int64)
        wheld = np.zeros(n, np.int64) if self.window else None
        live = set()
        for req in self.slot_req:
            if req is not None:
                np.add.at(held, np.asarray(req.pages, np.int64), 1)
                live.add(req.state_slot)
                if self.window:
                    np.add.at(wheld, np.asarray([p for p in req.wpages if p], np.int64), 1)
                    assert len(req.wpages) == self.num_wpages[req.slot] and not any(req.wpages[:self.wlo[req.slot]]), \
                        "a window page table out of sync"
        return held, wheld, live
