"""A deterministic fake runner: the engine, scheduler and API run on a laptop against it (no GPU, no threads).

**State fingerprints.** Each recurrent-state slot carries the fingerprint of the token prefix its state represents:
its length and a rolling hash `H = H × B + token + 1 (mod 2^64)` over the prefix (`context_hash`). A forward pass
extends the fingerprint of the row's live slot by the row's input tokens; STATE_ZERO resets it; STATE_COPY and
CKPT copy it (a mid-chunk checkpoint gets the fingerprint of the prefix up to its offset). A row that resumes
without STATE_ZERO must find a state of exactly its own length (`pos0`), and in strict mode exactly the prefix its
KV pages hold, or `PlanError` is raised. An attention-only model resumes from its KV pages (strict mode only), which
give its fingerprint.

**Tokens.** The token sampled for a request is `splitmix64(admit_seed XOR position XOR H) mod vocab` in wrapping
uint64 arithmetic, where `position` is the index the new token will occupy and `H` the fingerprint of the context
before it (`fake_token`). So output depends on the context, as with a real model: a state or KV reused for the
wrong prefix changes every later token, and a test that predicts outputs from the true context catches it, as does
a result matched to the wrong request. So that a request ends only where a test asks, a token that happens to be an
EOS ID is replaced by the next ID, except at the positions in `eos_at`. `scripts` gives a request (by seed) a recorded
token sequence from a position on: those are its samples there, EOS IDs included, masked and drafted as any other.

**NaN injection** (tests): `nonfinite_at` poisons what a request's SAMPLE row at a given position writes, and
`nonfinite_write_at` its explicit row whose chunk covers a position (`nonfinite_once`: a transient fault). Any row that
reads poisoned state or KV is poisoned too, as NaNs propagate on a GPU. Only SAMPLE rows report NONFINITE, as on the
Qwen runner, unless `report_unsampled` (Gemma 4's runner). `mask_empty_at`: the MASKED sample at a position finds its
mask empty, unless the mask allows every token (a grammar that ended under `ignore_eos`), which a mask a grammar
computed never does; so whether it fires does not depend on which row samples the position. `poison_read` collects the
seeds of requests whose rows read poisoned state or KV, `fired` those whose injection took effect.

**Device stream.** Simulated without threads: `launch_forward` and `launch_sample` record when their work would run on a
serial device stream (`start = max(now, stream end)`, then `step_ms` plus `prefill_us_per_token` per query token, or
`sample_ms`), and `wait` passes the time until the step's sampling would end (`wait_mode`: spinning, or sleeping most of
it). Idle gaps on that timeline are reported in `StepResult`. With `sample_ms` > 0 every step is a forward, then a
sampling part, whose boundary `forward_pending` reports.

**MTP** (`mtp_k` k > 0). A device-input row verifies 1 + k tokens and returns n_out = 1 + a: the target's tokens are
unchanged (`fake_token` at each position), and the number of accepted drafts a is drawn from (seed, first position),
each draft accepted with probability `accept_p` while the previous ones were, then capped by the slot's commit limits: a
stop token ADMIT carries or the token at `max_len − 1` ends the accepted run. The recurrent state commits lazily, as the
CUDA runner's does: the committed tokens stay pending window records until the slot's next decode row or a FOLD folds
them; a −1 checkpoint of a decode row folds a copy. Explicit rows on a slot with pending records, FOLD counts beyond
them, a `sched_next_tok` that is not the next chunk's first token, and drafter KV writes into pages another admitted
slot names raise `PlanError`. **A drafter without KV** (`draft_kv=False`, Gemma 4's assistant) writes the verify window
only, and its rollback is a truncation: a window or a hit that reads past a committed position fails the content checks.

**Grammar masks and drafts.** Every MTP SAMPLE row returns k drafts (`StepResult.drafts`): the unmasked chain of what
the target would sample, with the draft after the accepted count made wrong. A MASKED row samples under the masks the
engine wrote (`masked_pick`), and an MTP row verifies its drafts against those masked samples. `fake_generate(...,
matcher=)` predicts the outputs with a fresh grammar matcher (`tests/support/fake_grammar.py`), so a mask at the wrong
grammar state, or along the wrong drafts, changes them.

**Logprobs** (`logprobs`). A LOGPROBS row's samples get `fake_logprobs`'s values: a mask that allows the sample alone
gives it 0 and the lowest other ids at −inf, as the CUDA runners do (so the forced prefix's and the EOS-free finish's
tokens, which the scheduler gives those values, look the same sampled or not); any other sample gets a value from
(seed, position, context) and is its own top token, then the next tokens (the allowed ones, cyclically), a quarter lower
each, then the lowest refused ids at −inf. A LOGPROBS row without SAMPLE, or of a slot admitted without ADMIT_LOGPROBS,
raises.

**Images** (`encoder_rows` > 0). An explicit token whose `tok_emb` names an encoder-pool row stands for that row's image
key in the fingerprints and the strict KV, so a span fed the wrong image's rows changes every later token. Checked: rows
written before they are read, in token order within a span; under MTP, the image token in `sched_next_emb`'s row as its
chunk runs is the one in the next chunk's first row (another copy when the request gave its images back meanwhile:
architecture.md, "The scheduler"); a final chunk's last token carries the ADMIT rope offset as its delta; the encoder
budget; with `bidirectional_images`, that every span in a row is whole and named in SPANS. M-RoPE: the sampled token
also depends on the slot's ADMIT rope offset, so a wrong offset changes the outputs (`fake_generate(..., rope_off=)`).

**Sliding windows** (`window` W > 0). A row whose first query is at position q must have window pages for
[max(0, q − W + 1), its end) holding its own tokens (strict mode). Full pages and window pages share one id space, as
on the GPU: a page named by both kinds at once raises, and in strict mode a page written as one kind loses what it held
as the other.

**PRECISE rows** (`precise_rows`). The fake tracks which states and (strict mode) KV positions PRECISE rows computed
from a precise start. A PRECISE row that continues an imprecise state, or reads KV that no PRECISE row wrote, raises
`PlanError`; tests check that the cache's precise namespace holds only precise pages and checkpoints.

**Validation.** Every plan is checked against the slot tables the fake keeps, and any violation raises `PlanError`:
`pos0` equal to the device `seq_len`; device input only after a SAMPLE step of the same admission; pages covering every
position written; no slot twice in SCHED; the token budget; slot and state-slot ownership; copies and checkpoints into
valid slots; checkpoint offsets on the grid; under batch invariance on a GDN model with the requests' exact cuts given
(`exact_cuts`), every prefill cut on the grid from where its computation started (`_check_cut`). With `strict`, the fake
also stores the token written at every KV position and checks, at every step, that the pages of each scheduled request
still hold its own tokens and that no row writes into a page another admitted slot names. `full_checks` runs the
reference implementations of the SCHED and KV checks: slower, with the same results and errors
(`tests/sched/test_fake_checks.py` compares the two under injected faults). The checks are in `fake_checks.py`.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ..sched.plan import (
    ADMIT_CONSTRAINED,
    ADMIT_LOGPROBS,
    CKPT_END_OF_STEP,
    CKPT_GRID,
    FOLD_ALL,
    LOGPROBS,
    MASKED,
    PRECISE,
    STATE_ZERO,
    StepPlan,
    reserve_positions,
)
from .base import (
    MASK_EMPTY,
    MAX_LOGPROBS,
    NONFINITE,
    Capabilities,
    LogprobRows,
    PlanError,
    PoolConfig,
    Runner,
    StepResult,
)
from .fake_checks import HASH_BASE, MASK64, FakeChecks, context_hash

_U64 = np.uint64
_SLEEP_FRACTION = 0.75 if sys.platform == "darwin" else 1.0

log = logging.getLogger(__name__)


def splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK64
    return x ^ (x >> 31)


def splitmix64_np(x: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        x = x.astype(_U64) + _U64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> _U64(30))) * _U64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> _U64(27))) * _U64(0x94D049BB133111EB)
        return x ^ (x >> _U64(31))


def rope_mix(rope_off: int) -> int:
    """What a request's M-RoPE offset adds to the fake's sampling noise (0 for 0)."""
    return (rope_off * 0x9E3779B97F4A7C15) & MASK64


def fake_token(seed: int, position: int, vocab: int, avoid: Iterable[int] = (), ctx: int = 0, rope_off: int = 0) -> int:
    """The token the fake runner samples at `position` for a request admitted with `seed`, after a context whose
    `context_hash` is `ctx`."""
    avoid = set(avoid)
    t = splitmix64((seed ^ position ^ ctx ^ rope_mix(rope_off)) & MASK64) % vocab
    while t in avoid:
        t = (t + 1) % vocab
    return t


def fake_logprobs(seed: int, pos: int, ctx: int, mix: int, t: int, allowed: np.ndarray | None,
                  vocab: int) -> tuple[float, list[int], list[float]]:
    """The fake's logprobs of its sample `t` at `pos` after a context hashed `ctx` (`mix`: its M-RoPE offset's), under
    a mask (bool [vocab]) or none: (t's value, the MAX_LOGPROBS top ids, their values; module docstring)."""
    nz = np.flatnonzero(allowed) if allowed is not None else None
    if nz is not None and len(nz) == 1:
        return 0.0, [t, *(i for i in range(MAX_LOGPROBS) if i != t)][:MAX_LOGPROBS], \
            [0.0] + [float("-inf")] * (MAX_LOGPROBS - 1)
    h = splitmix64((seed ^ pos ^ int(ctx) ^ mix ^ 0x10C0) & MASK64)
    v0 = -float((h >> 11) % 2048) / 256.0
    if nz is None:
        ids = [(t + j) % vocab for j in range(MAX_LOGPROBS)]
    else:
        i = int(np.searchsorted(nz, t))
        ids = [int(nz[(i + j) % len(nz)]) for j in range(min(MAX_LOGPROBS, len(nz)))]
    vals = [v0 - 0.25 * j for j in range(len(ids))]
    if len(ids) < MAX_LOGPROBS:  # the lowest refused ids
        refused = np.flatnonzero(~allowed[:2 * MAX_LOGPROBS + len(nz)])[:MAX_LOGPROBS - len(ids)].tolist()
        ids += refused
        vals += [float("-inf")] * len(refused)
    return v0, ids, vals


def masked_pick(t: int, allowed: np.ndarray) -> int:
    """The token the fake samples under a grammar mask (bool [vocab]) when its unmasked choice is `t`: t if the mask
    allows it, else the first allowed token after it (cyclically); −1 if the mask allows nothing."""
    if allowed[t]:
        return t
    nz = np.flatnonzero(allowed)
    if not len(nz):
        return -1
    return int(nz[np.searchsorted(nz, t) % len(nz)])


def unpack_mask(words: np.ndarray, vocab: int) -> np.ndarray:
    """int32 mask words -> bool [vocab]."""
    return np.unpackbits(words.view(np.uint8), bitorder="little")[:vocab].astype(bool)


def fake_generate(prompt: Iterable[int], seed: int, n: int, vocab: int, avoid: Iterable[int] = (),
                  eos_positions: Mapping[int, int] | None = None, rope_off: int = 0, matcher: Any = None,
                  eos_ids: Iterable[int] = (), logprobs: list | None = None) -> list[int]:
    """The fake's first n output tokens for a prompt (`eos_positions`: position -> EOS token to emit there). With
    images, `prompt` is the radix key (each span's positions hold its image key) and `rope_off` the request's M-RoPE
    offset. With a grammar `matcher` (a fresh one; `tests/support/fake_grammar.py`), each token is the masked
    pick of the unmasked one (`masked_pick`), until one of `eos_ids` ends the grammar (after which, with ignore_eos,
    it is unconstrained): it consumes the EOS, or once stopped just ends there (llguidance takes no EOS then).
    `logprobs`: each token's `fake_logprobs` appended."""
    toks = [int(t) for t in prompt]
    h = context_hash(toks)
    eos = set(eos_ids)
    out = []
    for i in range(n):
        pos = len(toks) + i
        t = (eos_positions or {}).get(pos)
        if t is None:
            t = fake_token(seed, pos, vocab, avoid, h, rope_off)
        allowed = matcher.allowed() if matcher is not None else None
        if matcher is not None:
            t = masked_pick(t, allowed)
            assert t >= 0, "the grammar allows nothing"
            assert (t in eos and matcher.is_stopped()) or matcher.consume_tokens([t]), "the grammar rejected its pick"
            if t in eos:
                matcher = None
        if logprobs is not None:
            logprobs.append(fake_logprobs(seed, pos, h, rope_mix(rope_off), t, allowed, vocab))
        out.append(t)
        h = context_hash((t,), h)
    return out


def model_geometry(model_dir: str | Path) -> dict[str, int]:
    """Sizes from an HF `config.json`, as a real runner would report them, for realistic pool sizing.

    KV per token covers every full-attention layer plus the MTP drafter's (BF16 K and V); a state slot holds the
    FP32 recurrent state and BF16 conv state of every linear-attention (GDN) layer plus the MTP hidden vector.
    For Qwen3.8-27B this gives 1,114,112 B per 16-token page and 153,954,304 B per slot.
    """
    cfg = json.loads((Path(model_dir) / "config.json").read_text())
    tc = cfg.get("text_config", cfg)
    if tc.get("model_type", cfg.get("model_type")) in ("gemma4", "gemma4_text"):  # sliding window + full layers
        from ..models.gemma4.model import FULL, SLIDING, Gemma4Config

        g = Gemma4Config.from_hf(cfg)
        return {"kv_bytes_per_token": g.kv_bytes_per_token(FULL, 2), "state_slot_bytes": 0,
                "window_kv_bytes_per_token": g.kv_bytes_per_token(SLIDING, 2), "window": g.sliding_window,
                "vocab_size": g.vocab_size, "max_positions": g.max_position_embeddings,
                "bidirectional_images": g.bidirectional == "vision" and bool(cfg.get("vision_config"))}
    n_layers = int(tc.get("num_hidden_layers", 0))
    types = tc.get("layer_types")
    if not types:
        interval = int(tc.get("full_attention_interval", 1) or 1)
        types = ["full_attention" if (i + 1) % interval == 0 else "linear_attention" for i in range(n_layers)]
    n_attn = sum(t == "full_attention" for t in types)
    n_lin = sum(t == "linear_attention" for t in types)
    mtp = int(tc.get("mtp_num_hidden_layers", 0) or 0)
    hidden = int(tc.get("hidden_size", 0))
    head_dim = int(tc.get("head_dim") or (hidden // max(1, int(tc.get("num_attention_heads", 1)))))
    kv_heads = int(tc.get("num_key_value_heads", tc.get("num_attention_heads", 1)))
    kv_bytes_per_token = (n_attn + mtp) * 2 * kv_heads * head_dim * 2
    state = 0
    if n_lin:
        kh, kd = int(tc.get("linear_num_key_heads", 0)), int(tc.get("linear_key_head_dim", 0))
        vh, vd = int(tc.get("linear_num_value_heads", 0)), int(tc.get("linear_value_head_dim", 0))
        conv = int(tc.get("linear_conv_kernel_dim", 4))
        recurrent = vh * kd * vd * 4
        conv_state = (2 * kh * kd + vh * vd) * (conv - 1) * 2
        state = n_lin * (recurrent + conv_state) + (hidden * 2 if mtp else 0)
    vocab = int(tc.get("vocab_size", cfg.get("vocab_size", 0)))
    max_pos = int(tc.get("max_position_embeddings", cfg.get("max_position_embeddings", 0)) or 0)
    return {"kv_bytes_per_token": kv_bytes_per_token, "state_slot_bytes": state, "vocab_size": vocab,
            "max_positions": max_pos}


class _Handle:
    __slots__ = ("bad", "deadline", "end_ckpts", "fwd_dur", "fwd_end", "idle_before", "idle_inside", "lpf",
                 "mask_off", "masked", "mtp", "n_masks", "plan", "result", "rows", "step_id")


class FakeRunner(FakeChecks, Runner):
    def __init__(self, *, vocab_size: int, max_positions: int, kv_bytes_per_token: int = 69_632,
                 state_slot_bytes: int = 153_954_304, memory_bytes: int = 66_571_993_088, step_ms: float = 5.0,
                 sample_ms: float = 0.0, prefill_us_per_token: float = 0.0,
                 eos_at: Mapping[int, Iterable[int]] | None = None,
                 scripts: Iterable[tuple[int, int, Iterable[int]]] | None = None,
                 nonfinite_at: Mapping[int, Iterable[int]] | None = None, strict: bool = True, spin_us: float = 300.0,
                 wait_mode: str = "spin", checkpoint_grid: int = CKPT_GRID, mtp_k: int = 0, accept_p: float = 0.8,
                 encoder_rows: int = 0, encoder_budget: int = 0, window: int = 0,
                 window_kv_bytes_per_token: int = 0, bidirectional_images: bool = False, draft_kv: bool = True,
                 precise_rows: bool = False, nonfinite_write_at: Mapping[int, Iterable[int]] | None = None,
                 report_unsampled: bool = False, mask_empty_at: Mapping[int, Iterable[int]] | None = None,
                 nonfinite_once: Iterable[int] = (), batch_invariant: bool = False,
                 exact_cuts: Mapping[int, Iterable[int]] | None = None, full_checks: bool = False,
                 logprobs: bool = True):
        self.vocab = vocab_size
        self.logprobs = logprobs  # takes LOGPROBS rows (module docstring)
        self.full_checks = full_checks  # the reference implementations of the SCHED and KV checks (Validation)
        self.precise_rows = precise_rows  # takes PRECISE rows (the precise prefix; module docstring)
        self.draft_kv = draft_kv  # MTP: the drafter writes KV of its own (module docstring)
        self.window = window
        self.bidi = bidirectional_images
        self.encoder_rows = encoder_rows
        self.encoder_budget = encoder_budget or encoder_rows
        self.k = mtp_k
        self.accept_p = accept_p
        self.grid = checkpoint_grid  # tests use a small grid to checkpoint short prompts
        self.has_state = state_slot_bytes > 0  # attention-only models resume from KV alone
        self._caps = Capabilities(backend="fake", device="fake device", vocab_size=vocab_size,
                                  max_positions=max_positions, memory_total_bytes=memory_bytes,
                                  memory_free_bytes=memory_bytes, kv_bytes_per_token=kv_bytes_per_token,
                                  state_slot_bytes=state_slot_bytes, checkpoint_grid=checkpoint_grid,
                                  checkpoints="window" if window else "every_n" if state_slot_bytes else "none",
                                  mtp_depth=mtp_k, images=encoder_rows > 0, encoder_rows=encoder_rows,
                                  encoder_budget=self.encoder_budget, grammar_masks=True, window=window,
                                  window_kv_bytes_per_token=window_kv_bytes_per_token if window else 0,
                                  bidirectional_images=bidirectional_images, draft_kv=draft_kv,
                                  precise_rows=precise_rows, forward_end=sample_ms > 0,
                                  sample_step_ms=float(sample_ms), batch_invariant=batch_invariant,
                                  logprobs=logprobs)
        # Batch invariance on a GDN model (module docstring, Validation): every prefill cut a whole number of grids from
        # where the prompt's computation started, but at a SAMPLE row, a PRECISE row or one of `exact_cuts` (a
        # request's hints and image cuts, by seed; checked only when given)
        self.aligned = batch_invariant and state_slot_bytes > 0 and not window and exact_cuts is not None
        self.exact_cuts = {int(s): {int(p) for p in ps} for s, ps in (exact_cuts or {}).items()}
        self.step_s = step_ms / 1e3
        self.sample_s = sample_ms / 1e3
        self.per_token_s = prefill_us_per_token / 1e6
        self.spin_s = spin_us / 1e6
        self.wait_mode = wait_mode  # "spin" (CUDA's default spin-wait) or "sleep"
        self.eos_at = {int(s): {int(p) for p in ps} for s, ps in (eos_at or {}).items()}
        self.scripts = {int(s): (int(p), [int(t) for t in toks]) for s, p, toks in (scripts or ())}
        self.nonfinite_at = {int(s): {int(p) for p in ps} for s, ps in (nonfinite_at or {}).items()}
        # NaN injection (module docstring)
        self.nonfinite_write_at = {int(s): {int(p) for p in ps} for s, ps in (nonfinite_write_at or {}).items()}
        self.nonfinite_once = {int(s) for s in nonfinite_once}
        self.report_unsampled = report_unsampled
        self.mask_empty_at = {int(s): {int(p) for p in ps} for s, ps in (mask_empty_at or {}).items()}
        self.poison_read: set[int] = set()
        self.fired: set[int] = set()
        self.strict = strict
        self._stream_end = 0.0
        self.steps = 0
        self.configured = False
        self.counts = {"kv_copy": 0, "state_copy": 0, "ckpt": 0, "resumed": 0, "folds": 0, "drafts": 0, "encoded": 0,
                       "accepted": 0, "masked": 0, "masks": 0, "kind_switch": 0}

    def capabilities(self) -> Capabilities:
        return self._caps

    def configure(self, pools: PoolConfig) -> None:
        if pools.mtp_k != self.k:
            raise PlanError(f"configured with mtp_k {pools.mtp_k}; this fake runner drafts {self.k}")
        self.pools = pools
        s = pools.max_num_seqs + 1
        self.ps = pools.page_size
        self.wps = pools.window_page_size or pools.page_size  # window pages (one arena with the full ones)
        self.reserve = reserve_positions(2, self.k)
        self.max_pages_per_seq = -(-(pools.max_model_len + self.reserve) // pools.page_size) + 1
        self.max_wpages_per_seq = -(-(pools.max_model_len + self.reserve) // self.wps) + 1
        self.admitted = np.zeros(s, bool)
        self.seed = np.zeros(s, _U64)
        self.state_slot = np.zeros(s, np.int32)
        self.seq_len = np.zeros(s, np.int64)
        self.pending = np.full(s, -1, np.int64)
        self.sampled = np.zeros(s, bool)
        self.rows = np.zeros((s, self.max_pages_per_seq), np.int32)
        n_states = pools.num_state_slots
        self.state_owner = np.zeros(n_states, np.int32)  # live state slot -> request slot
        self.fp_len = np.full(n_states, -1, np.int64)  # -1: holds no state
        self.fp_hash = np.zeros(n_states, _U64)
        self.fp_bad = np.zeros(n_states, bool)
        self.fp_prec = np.zeros(n_states, bool)  # the state was computed by PRECISE rows alone, from position 0
        self.page_users = np.zeros(pools.num_pages, np.int32)  # admitted rows naming each page
        if self.window:  # the window pages' table and users, over the arena's ids
            if not pools.window_page_size:
                raise PlanError("a sliding-window runner needs PoolConfig.window_page_size (one arena)")
            self.wrows = np.zeros((s, self.max_wpages_per_seq), np.int32)
            self.wpage_users = np.zeros(pools.num_pages, np.int32)
            self.kind = np.zeros(pools.num_pages, np.int8)  # strict mode: the kind each page was last written as
        # MTP: commit limits, and the window records a live state has not folded yet (the committed tokens of the
        # slot's last decode row), with whether they are poisoned
        self.max_len = np.zeros(s, np.int64)
        self.stop = [frozenset()] * s
        self.fold_n = np.zeros(s, np.int64)
        self.rec_tok: list[list[int]] = [[] for _ in range(s)]
        self.rec_bad = np.zeros(s, bool)
        self.resumed = np.zeros(s, bool)  # admitted with seq_len > 0 and no row yet: the drafter recomputes pos0 − 1
        self.next_tok = np.full(s, -1, np.int64)  # the token the previous chunk promised for this one's first
        self.rope_off = np.zeros(s, np.int64)  # ADMIT's M-RoPE offset
        self.constrained = np.zeros(s, bool)  # ADMIT_CONSTRAINED: the request has a grammar (MASKED rows allowed)
        self.lp_slot = np.zeros(s, bool)  # ADMIT_LOGPROBS: the request asks for logprobs (LOGPROBS rows allowed)
        self.anchor = np.zeros(s, np.int64)  # prefill grid start (`aligned`): the hit, an exact cut, a PRECISE end
        self.drafts = np.zeros((s, max(1, self.k)), np.int64)  # MTP: the drafts each slot's next decode row verifies
        self._masks = np.zeros((max(1, pools.max_num_seqs * (1 + self.k)), (self.vocab + 31) // 32), np.int32)
        self.next_img = np.full((s, 2), -1, np.int64)  # the previous chunk's look-ahead: (key, index), or (-1, -1)
        self.last_img = np.full((s, 2), -1, np.int64)  # (key, index) of the slot's last explicit token, (-1, -1): text
        self.enc_key = np.zeros(max(1, self.encoder_rows), np.int64)  # 0: never written
        self.enc_idx = np.full(max(1, self.encoder_rows), -1, np.int64)
        self.key_n: dict[int, int] = {}  # image key -> its tokens
        self._eff = np.zeros(0, np.int64)
        self.eos_ids = tuple(pools.eos_token_ids)
        pw = np.empty(pools.max_model_len + self.reserve + 2, _U64)
        x = 1
        for i in range(len(pw)):
            pw[i] = x
            x = (x * HASH_BASE) & MASK64
        self._pow = pw
        if self.strict:
            self.kv = np.full((pools.num_pages, pools.page_size), -1, np.int64)
            self.kv_bad = np.zeros((pools.num_pages, pools.page_size), bool)
            self.kv_prec = np.zeros((pools.num_pages, pools.page_size), bool)  # written by a PRECISE row
            if self.window:
                self.wkv = np.full((pools.num_pages, self.wps), -1, np.int64)
                self.wkv_bad = np.zeros((pools.num_pages, self.wps), bool)
                self.wkv_prec = np.zeros((pools.num_pages, self.wps), bool)
            self.kv_img = np.full((pools.num_pages, pools.page_size), -1, np.int64)  # image token index, -1: text
            self.history: list[list[int]] = [[] for _ in range(s)]
            self.img_hist: list[list[int]] = [[] for _ in range(s)]
            # `_check_kv` reads and writes them through flat views (one index, page × page size + offset, for every
            # array of a kind), with each position's page index and offset precomputed
            names = ("kv", "kv_bad", "kv_prec", "kv_img") + (("wkv", "wkv_bad", "wkv_prec") if self.window else ())
            self.flat = {a: getattr(self, a).reshape(-1) for a in names}
            self._pdiv, self._pmod = np.divmod(np.arange(self.max_pages_per_seq * self.ps), self.ps)
            if self.window:
                self._wdiv, self._wmod = np.divmod(np.arange(self.max_wpages_per_seq * self.wps), self.wps)
        self.configured = True

    # -- fingerprints ---------------------------------------------------------------------------------------------

    def _extend(self, h: Any, toks: np.ndarray) -> _U64:
        """`context_hash(toks, h)` as a uint64: a loop over a few tokens, NumPy over more (its cost per call would
        outweigh a short loop)."""
        k = len(toks)
        if k <= 16:
            return _U64(context_hash(toks.tolist(), int(h)))
        with np.errstate(over="ignore"):
            s = np.sum((toks.astype(_U64) + _U64(1)) * self._pow[k - 1::-1], dtype=_U64)
            return _U64(h) * self._pow[k] + s

    # -- plan effects ---------------------------------------------------------------------------------------------

    def _fail(self, plan: StepPlan, msg: str) -> None:
        raise PlanError(f"step {plan.step_id}: {msg}\n{plan.describe()}")

    def _fold(self, slot: int, n: int, dst: int | None = None) -> None:
        """Apply the first n pending window records of `slot` to its live state (or, with `dst`, to a copy of it in
        state slot `dst`, leaving the live state alone: a −1 checkpoint of a decode row)."""
        ss = int(self.state_slot[slot])
        h, ln = context_hash(self.rec_tok[slot][:n], int(self.fp_hash[ss])), self.fp_len[ss] + n
        bad = bool(self.fp_bad[ss] or (n and self.rec_bad[slot]))
        prec = bool(self.fp_prec[ss]) and not n  # a decode row's commit is never PRECISE
        if dst is None:
            self.fp_hash[ss], self.fp_len[ss], self.fp_bad[ss], self.fp_prec[ss] = h, ln, bad, prec
            self.rec_tok[slot] = self.rec_tok[slot][n:]
            self.fold_n[slot] -= n
        else:
            self.fp_hash[dst], self.fp_len[dst], self.fp_bad[dst], self.fp_prec[dst] = h, ln, bad, prec

    def _apply_tables(self, plan: StepPlan) -> set[int]:
        nseq, n_states = self.pools.max_num_seqs, self.pools.num_state_slots
        for slot, n in zip(plan.fold_slot.tolist(), plan.fold_n.tolist()):
            if not self.k:
                self._fail(plan, "FOLD without MTP")
            if not 1 <= slot <= nseq or not self.admitted[slot]:
                self._fail(plan, f"FOLD of slot {slot}, which is not admitted")
            if slot not in plan.free_slot:
                self._fail(plan, f"FOLD of slot {slot}, which the plan does not FREE")
            n = int(self.fold_n[slot]) if n == FOLD_ALL else n
            if not 0 <= n <= self.fold_n[slot]:
                self._fail(plan, f"FOLD of {n} records of slot {slot}, which has {int(self.fold_n[slot])} pending")
            self._fold(slot, n)
            self.counts["folds"] += 1
        for slot in plan.free_slot.tolist():
            if not 1 <= slot <= nseq or not self.admitted[slot]:
                self._fail(plan, f"FREE of slot {slot}, which is not admitted")
            row = self.rows[slot]
            np.subtract.at(self.page_users, row[row > 0], 1)
            row[:] = 0
            if self.window:
                wrow = self.wrows[slot]
                np.subtract.at(self.wpage_users, wrow[wrow > 0], 1)
                wrow[:] = 0
            self.state_owner[self.state_slot[slot]] = 0
            self.admitted[slot] = self.sampled[slot] = False
            self.pending[slot] = -1
            self.fold_n[slot] = 0
            self.rec_tok[slot] = []
        admitted_now: set[int] = set()
        for i, slot in enumerate(plan.admit_slot.tolist()):
            ss = int(plan.admit_state_slot[i])
            h = int(plan.admit_seq_len[i])
            if h and not self.has_state and not self.strict:
                self._fail(plan, "the fake resumes attention-only models from KV contents: strict mode only")
            if not 1 <= slot <= nseq or self.admitted[slot]:
                self._fail(plan, f"ADMIT into slot {slot}, which is in use")
            if not 1 <= ss < n_states or self.state_owner[ss]:
                self._fail(plan, f"ADMIT with state slot {ss}, which is invalid or in use")
            if not 0 <= h < self.pools.max_model_len:
                self._fail(plan, f"ADMIT with seq_len {h}")
            self.admitted[slot] = True
            admitted_now.add(slot)
            self.state_slot[slot] = ss
            self.state_owner[ss] = slot
            self.fp_bad[ss] = False  # a resume without recurrent state takes nothing from the slot's last occupant
            self.seed[slot] = plan.admit_seed[i]
            self.seq_len[slot] = h
            self.anchor[slot] = h
            self.sampled[slot] = False
            self.pending[slot] = -1
            self.max_len[slot] = int(plan.admit_max_len[i]) if self.k else 1 << 40
            self.stop[slot] = frozenset(int(t) for t in plan.admit_stop[i].tolist() if t >= 0) if self.k else \
                frozenset()
            self.fold_n[slot] = 0
            self.rec_tok[slot] = []
            self.resumed[slot] = h > 0
            self.next_tok[slot] = -1
            self.next_img[slot] = -1
            self.rope_off[slot] = int(plan.admit_rope_off[i]) if len(plan.admit_rope_off) else 0
            self.constrained[slot] = bool(int(plan.admit_flags[i]) & ADMIT_CONSTRAINED)
            self.lp_slot[slot] = bool(int(plan.admit_flags[i]) & ADMIT_LOGPROBS)
            if self.lp_slot[slot] and not self.logprobs:
                self._fail(plan, "ADMIT_LOGPROBS, but this runner takes no LOGPROBS rows")
            self.drafts[slot] = -1
            self.last_img[slot] = (-1, -1) if h == 0 else (-2, -2)  # a resume's last token: unknown until read
            if h:
                self.counts["resumed"] += 1
            if self.strict:
                self.history[slot] = []
                self.img_hist[slot] = []
        for slot, idx, page in zip(plan.page_slot.tolist(), plan.page_index.tolist(), plan.page_id.tolist()):
            if not 1 <= slot <= nseq or not self.admitted[slot]:  # the range first: a Python index wraps
                self._fail(plan, f"PAGES for slot {slot}, which is not admitted")
            if not 0 <= idx < self.max_pages_per_seq or not 1 <= page < self.pools.num_pages:
                self._fail(plan, f"PAGES entry ({slot}, {idx}) = {page} out of range")
            old = self.rows[slot, idx]
            if old:
                self.page_users[old] -= 1
            self.rows[slot, idx] = page
            self.page_users[page] += 1
        if len(plan.wpage_id) and not self.window:
            self._fail(plan, "window PAGES, but this model has no sliding window")
        freed = set(plan.free_slot.tolist())
        for slot, idx, page in zip(plan.wpage_slot.tolist(), plan.wpage_index.tolist(), plan.wpage_id.tolist()):
            if not 1 <= slot <= nseq or not self.admitted[slot]:
                if page == 0 and slot in freed:
                    continue  # a release of the request this plan FREEs (plan.py)
                self._fail(plan, f"window PAGES for slot {slot}, which is not admitted")
            if not 0 <= idx < self.max_wpages_per_seq or not 0 <= page < self.pools.num_pages:
                self._fail(plan, f"window PAGES entry ({slot}, {idx}) = {page} out of range")
            old = self.wrows[slot, idx]
            if old:
                self.wpage_users[old] -= 1
            self.wrows[slot, idx] = page  # 0: released
            if page:
                self.wpage_users[page] += 1
        if self.window:  # one arena: a page is one kind at a time
            for page in plan.page_id.tolist() + plan.wpage_id.tolist():
                if self.page_users[page] > 0 and self.wpage_users[page] > 0:
                    self._fail(plan, f"page {page} is named as a full page and as a window page at once")
        if len(plan.kv_copy_src):
            src, dst = plan.kv_copy_src.astype(np.int64), plan.kv_copy_dst.astype(np.int64)
            n_pages = self.pools.num_pages
            if (src < 1).any() or (src >= n_pages).any() or (dst < 1).any() or (dst >= n_pages).any():
                self._fail(plan, "KV_COPY page out of range")
            if (self.page_users[dst] > 1).any() or len(set(dst.tolist())) != len(dst):
                self._fail(plan, "KV_COPY into a shared page")
            if self.window and (self.wpage_users[dst] > 0).any():
                self._fail(plan, "KV_COPY into a page that is a window page")
            if self.strict:
                self._become(dst, 1)
                self.kv[dst] = self.kv[src].copy()
                self.kv_bad[dst] = self.kv_bad[src].copy()
                self.kv_prec[dst] = self.kv_prec[src].copy()
                self.kv_img[dst] = self.kv_img[src].copy()
            self.counts["kv_copy"] += len(dst)
        if len(plan.wkv_copy_src):
            if not self.window:
                self._fail(plan, "window KV_COPY, but this model has no sliding window")
            src, dst = plan.wkv_copy_src.astype(np.int64), plan.wkv_copy_dst.astype(np.int64)
            n_pages = self.pools.num_pages
            if (src < 1).any() or (src >= n_pages).any() or (dst < 1).any() or (dst >= n_pages).any():
                self._fail(plan, "window KV_COPY page out of range")
            if (self.wpage_users[dst] > 1).any() or len(set(dst.tolist())) != len(dst):
                self._fail(plan, "window KV_COPY into a shared page")
            if (self.page_users[dst] > 0).any():
                self._fail(plan, "window KV_COPY into a page that is a full page")
            if self.strict:
                self._become(dst, 2)
                self.wkv[dst] = self.wkv[src].copy()
                self.wkv_bad[dst] = self.wkv_bad[src].copy()
                self.wkv_prec[dst] = self.wkv_prec[src].copy()
            self.counts["kv_copy"] += len(dst)
        if len(plan.state_copy_src):
            src, dst = plan.state_copy_src.astype(np.int64), plan.state_copy_dst.astype(np.int64)
            if (src < 1).any() or (src >= n_states).any() or (dst < 1).any() or (dst >= n_states).any():
                self._fail(plan, "STATE_COPY slot out of range")
            if len(set(dst.tolist())) != len(dst):
                self._fail(plan, "two STATE_COPYs into one slot")
            if (self.fp_len[src] < 0).any():
                self._fail(plan, f"STATE_COPY from a slot that holds no state: {src[self.fp_len[src] < 0].tolist()}")
            for d in dst.tolist():
                owner = int(self.state_owner[d])
                if owner and owner not in admitted_now:
                    self._fail(plan, f"STATE_COPY into state slot {d}, the live state of running slot {owner}")
            # gather, then scatter: every source is read before any destination is written
            self.fp_len[dst], self.fp_hash[dst], self.fp_bad[dst], self.fp_prec[dst] = (
                self.fp_len[src].copy(), self.fp_hash[src].copy(), self.fp_bad[src].copy(), self.fp_prec[src].copy())
            self.counts["state_copy"] += len(dst)
        if len(plan.enc_key):
            self._encode(plan)
        self._eff = self._effective(plan)
        return admitted_now

    def _encode(self, plan: StepPlan) -> None:
        """ENCODE: record which image key and token index each named row now holds."""
        if not self.encoder_rows:
            self._fail(plan, "ENCODE, but this runner has no image encoder")
        n = plan.enc_tokens.astype(np.int64)
        if (n < 1).any() or int(n.sum()) != len(plan.enc_rows):
            self._fail(plan, "ENCODE token counts do not match its rows")
        if int(n.sum()) > self.encoder_budget:
            self._fail(plan, f"ENCODE of {int(n.sum())} image tokens > the encoder budget {self.encoder_budget}")
        rows = plan.enc_rows
        if (rows < 0).any() or (rows >= self.encoder_rows).any() or len(set(rows.tolist())) != len(rows):
            self._fail(plan, "ENCODE rows out of range or repeated")
        if (plan.enc_key < (1 << 62)).any():
            self._fail(plan, "ENCODE key outside the image-key range")
        for k, cnt, r0 in zip(plan.enc_key.tolist(), n.tolist(), plan.enc_row0.tolist()):
            r = rows[r0:r0 + cnt]
            self.enc_key[r] = k
            self.enc_idx[r] = np.arange(cnt)
            self.key_n[k] = cnt
        self.counts["encoded"] += len(plan.enc_key)

    def _effective(self, plan: StepPlan) -> np.ndarray:
        """The plan's explicit tokens as the context the fake hashes: each image token stands for its image key."""
        eff = plan.tokens.astype(np.int64)
        if len(plan.tok_emb):
            emb = plan.tok_emb
            m = emb >= 0
            if m.any():
                r = emb[m]
                if (r >= self.encoder_rows).any() or (self.enc_key[r] == 0).any():
                    self._fail(plan, "an image token reads an encoder row never written")
                eff[m] = self.enc_key[r]
        return eff

    def _sample(self, seed: int, pos: int, ctx: Any, mix: int = 0) -> int:
        """The token the fake samples at `pos` after a context with fingerprint hash `ctx`."""
        script = self.scripts.get(seed)
        if script is not None and 0 <= pos - script[0] < len(script[1]):
            return script[1][pos - script[0]]
        if pos in self.eos_at.get(seed, ()) and self.eos_ids:
            return self.eos_ids[0]
        t = int(splitmix64((seed ^ pos ^ int(ctx) ^ mix) & MASK64) % self.vocab)
        while t in self.eos_ids:
            t = (t + 1) % self.vocab
        return t

    def _mtp_decode(self, slot: int, read_bad: bool, masks: list[np.ndarray] | None,
                    lp: list | None = None) -> tuple[list[int], int]:
        """One MTP decode row of `slot` from its live state (already folded to its seq_len S): the target's samples
        at S + 1 .. S + 1 + a and the row's status (NONFINITE: the bonus, the last, is non-finite; MASK_EMPTY: its
        mask allowed nothing). Drafts are accepted while they equal the (masked) samples, within the commit limits
        and before a non-finite sample. `masks` (MASKED rows): 1 + k bool masks along [pending, drafts]. `lp`
        (LOGPROBS rows): each sample's `fake_logprobs` appended."""
        seed, S, k = int(self.seed[slot]), int(self.seq_len[slot]), self.k
        ss = int(self.state_slot[slot])
        if read_bad or self.fp_bad[ss]:
            return [0], NONFINITE  # NaNs everywhere: nothing accepted, the bonus is non-finite
        nan = self.nonfinite_at.get(seed, ())
        drafts = self.drafts[slot]
        ctx = context_hash((int(self.pending[slot]),), int(self.fp_hash[ss]))
        out: list[int] = []
        mix = rope_mix(int(self.rope_off[slot]))
        for i in range(k + 1):
            pos = S + 1 + i
            t = self._sample(seed, pos, ctx, mix)
            if masks is not None:
                t = masked_pick(t, masks[i])
                if t < 0 or (pos in self.mask_empty_at.get(seed, ()) and not masks[i].all()):
                    if t >= 0:
                        self.fired.add(seed)
                    return [*out, 0], MASK_EMPTY
            out.append(t)
            if lp is not None:
                lp.append(fake_logprobs(seed, pos, int(ctx), mix, t, masks[i] if masks is not None else None,
                                        self.vocab))
            if pos in nan:
                self.fired.add(seed)
                return out, NONFINITE
            if i == k or t != drafts[i] or t in self.stop[slot] or pos + 1 > self.max_len[slot] - 1:
                return out, 0
            ctx = context_hash((t,), ctx)
        return out, 0

    def _draft(self, slot: int, ctx: int) -> None:
        """The k drafts of `slot`'s next decode row, after its pending token (at the device seq_len S; `ctx`: the
        fingerprint through it): the unmasked chain of the target's samples, with the one after the accepted count drawn
        from (seed, S) made wrong (module docstring)."""
        seed, S, k = int(self.seed[slot]), int(self.seq_len[slot]), self.k
        h = splitmix64((seed ^ (S * 0x9E3779B97F4A7C15) ^ 0x5BD1E995) & MASK64)
        drawn = 0
        while drawn < k:
            if drawn and not drawn % 4:  # 16 bits a draft: a fresh word every 4 (k up to 7)
                h = splitmix64(h)
            if ((h >> (16 * (drawn % 4))) & 0xFFFF) >= self.accept_p * 65536:
                break
            drawn += 1
        mix = rope_mix(int(self.rope_off[slot]))
        for j in range(k):
            t = self._sample(seed, S + 1 + j, ctx, mix)
            if j == drawn:
                t = (t + 1) % self.vocab
            self.drafts[slot, j] = t
            ctx = context_hash((t,), ctx)

    # -- Runner API -----------------------------------------------------------------------------------------------

    def launch_forward(self, plan: StepPlan) -> _Handle:
        if not self.configured:
            raise RuntimeError("configure() first")
        now = time.perf_counter()
        self._apply_tables(plan)
        h = _Handle()
        h.step_id, h.plan = plan.step_id, plan  # the plan may be read until launch_sample returns
        resumed = self.resumed.copy() if self.k and self.draft_kv and self.strict else None  # `_check_sched` clears it
        h.rows = slots, pos0, nq, sample = self._check_sched(plan)
        n = len(slots)
        h.bad = np.zeros(n, bool)
        h.mtp = np.zeros(n, bool)
        dev = plan.sched_tok_off < 0
        # Masks: one per sample a MASKED row draws (1 + k for an MTP decode row), in plan order (runner.base).
        h.masked = (plan.sched_flags & MASKED) != 0
        h.lpf = (plan.sched_flags & LOGPROBS) != 0
        if h.masked.any():
            per = np.where(h.masked, np.where(dev & (self.k > 0), 1 + self.k, 1), 0)
            h.mask_off, h.n_masks = np.cumsum(per) - per, int(per.sum())
        else:
            h.mask_off, h.n_masks = np.zeros(n, np.int64), 0
        ck_by_row: dict[int, list[tuple[int, int]]] = {}
        for r, off, d in zip(plan.ckpt_sched.tolist(), plan.ckpt_offset.tolist(), plan.ckpt_dst.tolist()):
            ck_by_row.setdefault(r, []).append((off, d))
        self.counts["ckpt"] += len(plan.ckpt_sched)
        h.end_ckpts = []
        if n:
            live = self.state_slot[slots]
            sl, p0l, nql, offl = slots.tolist(), pos0.tolist(), nq.tolist(), plan.sched_tok_off.tolist()
            h.mtp = mtp = dev if self.k else np.zeros(n, bool)  # decode and verify with the masks in launch_sample
            explicit = [i for i in range(n) if offl[i] >= 0]
            decode = [] if self.k else [i for i in range(n) if offl[i] < 0]  # decode rows without MTP
            zero = (plan.sched_flags & STATE_ZERO) != 0
            precise = (plan.sched_flags & PRECISE) != 0
            # PRECISE rows continue a precise state: position 0, or one PRECISE rows computed from there (the BF16
            # prefix); without recurrent state the KV they read is checked instead (`_check_kv`)
            start_prec = zero | self.fp_prec[live]
            if self.has_state and (precise & ~start_prec).any():
                i = int(np.flatnonzero(precise & ~start_prec)[0])
                self._fail(plan, f"slot {int(slots[i])}: a PRECISE row continues a state that is not precise")
            row_prec = precise & start_prec
            bad = h.bad
            bad[:] = np.where(zero, False, self.fp_bad[live])  # NaNs in the state a row starts from
            if self.strict:  # before any fingerprint moves: a resumed slot's state is checked against its KV
                for i in explicit if self.k else range(n):  # (an MTP decode row's KV: in launch_sample)
                    a, b_ = offl[i], offl[i] + nql[i]
                    toks = img = None
                    if a >= 0:  # explicit inputs (a decode row's is its pending token)
                        toks = self._eff[a:b_]
                        if len(plan.tok_emb):
                            e = plan.tok_emb[a:b_]
                            img = np.where(e >= 0, self.enc_idx[np.maximum(e, 0)], -1)
                    if self._check_kv(plan, sl[i], p0l[i], nql[i], toks, img=img, precise=bool(precise[i])):
                        bad[i] = True
            draft_bad = None
            if resumed is not None:  # the first row of a resumed admission rewrites the drafter's KV at pos0 − 1 from
                # what it resumed from (its state's hidden vector; without recurrent state, the KV it reads)
                draft_bad = resumed[slots] & ~dev & (self.fp_bad[live] if self.has_state else bad)
            if bad.any():  # rows that read poisoned state or KV
                self.poison_read.update(self.seed[slots[bad]].tolist())
            if self.nonfinite_at:
                for i in np.flatnonzero(sample & ~mtp).tolist():
                    seed = int(self.seed[sl[i]])
                    if p0l[i] + nql[i] in self.nonfinite_at.get(seed, ()):
                        bad[i] = True
                        self.fired.add(seed)
            if self.nonfinite_write_at:
                for i in explicit:
                    seed, p0, end = int(self.seed[sl[i]]), p0l[i], p0l[i] + nql[i]
                    at = self.nonfinite_write_at.get(seed, ())
                    if any(p0 <= p < end for p in at):
                        bad[i] = True
                        self.fired.add(seed)
                        if seed in self.nonfinite_once:
                            self.nonfinite_write_at[seed] = {p for p in at if not p0 <= p < end}
            for i in decode:  # each live state takes its pending token
                ss = int(live[i])
                self.fp_hash[ss] = (int(self.fp_hash[ss]) * HASH_BASE + int(self.pending[sl[i]]) + 1) & MASK64
                self.fp_len[ss] += 1
                self.fp_bad[ss] = bad[i]
                self.fp_prec[ss] = False
            for i in explicit:
                ss, p0, q = int(live[i]), p0l[i], nql[i]
                toks = self._eff[offl[i]:offl[i] + q]
                h0 = _U64(0) if zero[i] else self.fp_hash[ss]
                for off, d in ck_by_row.get(i, ()):
                    if 0 < off < q:
                        self.fp_len[d], self.fp_hash[d], self.fp_bad[d] = p0 + off, self._extend(h0, toks[:off]), bad[i]
                        self.fp_prec[d] = row_prec[i]
                self.fp_hash[ss] = self._extend(h0, toks)
                self.fp_len[ss] = p0 + q
                self.fp_bad[ss] = bad[i]
                self.fp_prec[ss] = row_prec[i]
            if self.strict and (bad & ~mtp).any():  # NaNs spread to the KV the row wrote
                for i in np.flatnonzero(bad & ~mtp).tolist():
                    self._poison_kv(sl[i], p0l[i], nql[i])
            if draft_bad is not None and draft_bad.any():  # and to the drafter's KV at pos0 − 1 (a full page)
                for i in np.flatnonzero(draft_bad).tolist():
                    p = p0l[i] - 1
                    self.kv_bad[self.rows[sl[i], p // self.ps], p % self.ps] = True
            for i, cks in ck_by_row.items():  # end-of-chunk now; end-of-step after sampling
                ss = int(live[i])
                for off, d in cks:
                    if mtp[i] and off != CKPT_END_OF_STEP:
                        self._fail(plan, "an MTP decode row checkpoints only at the end of the step")
                    if off == nql[i]:
                        self.fp_len[d], self.fp_hash[d], self.fp_bad[d] = self.fp_len[ss], self.fp_hash[ss], bad[i]
                        self.fp_prec[d] = self.fp_prec[ss]
                    elif off == CKPT_END_OF_STEP:
                        h.end_ckpts.append((ss, d, sl[i] if mtp[i] else 0))
        start = max(now, self._stream_end)
        h.idle_before = start - self._stream_end if self._stream_end else 0.0
        h.fwd_dur = self.step_s + self.per_token_s * plan.n_query_tokens
        h.fwd_end = start + h.fwd_dur
        self._stream_end = h.fwd_end
        h.result = None
        self.steps += 1
        return h

    def _poison_kv(self, slot: int, pos0: int, nq: int) -> None:
        wpos = np.arange(pos0, pos0 + nq)
        self.kv_bad[self.rows[slot, wpos // self.ps], wpos % self.ps] = True
        if self.window:
            self.wkv_bad[self.wrows[slot, wpos // self.wps], wpos % self.wps] = True

    def mask_buffer(self, handle: _Handle) -> np.ndarray:
        return self._masks

    def launch_sample(self, handle: _Handle, n_masks: int = 0) -> None:
        now = time.perf_counter()
        slots, pos0, nq, sample = handle.rows
        n = len(slots)
        if n_masks != handle.n_masks:
            raise PlanError(f"step {handle.step_id}: {n_masks} masks for MASKED rows that draw {handle.n_masks}")
        tokens = np.full((n, 1 + self.k), -1, np.int32)
        drafts = np.full((n, self.k), -1, np.int32) if self.k else None
        n_out = np.zeros(n, np.int32)
        status = np.zeros(n, np.uint32)
        if n:
            k, vocab = self.k, self.vocab
            masked, off = handle.masked, handle.mask_off
            if masked.any():
                self.counts["masked"] += int(masked.sum())
                self.counts["masks"] += n_masks

            def mask(i: int, j: int = 0) -> np.ndarray:
                return unpack_mask(self._masks[off[i] + j], vocab)

            after = pos0 + nq
            bad = handle.bad
            mtp, sampling = handle.mtp.tolist(), np.flatnonzero(sample).tolist()
            lpf = handle.lpf
            lp_vals: dict[int, list] = {}  # LOGPROBS rows: each sample's values, in order
            for i in (i for i in range(n) if mtp[i]):  # verify rows: the replay folds the previous commit first
                slot, p0 = int(slots[i]), int(pos0[i])
                self._fold(slot, int(self.fold_n[slot]))
                rb = self.strict and self._read_bad(slot, p0)
                if rb or self.fp_bad[self.state_slot[slot]]:
                    self.poison_read.add(int(self.seed[slot]))
                toks, st = self._mtp_decode(slot, rb or bool(bad[i]),
                                            [mask(i, j) for j in range(1 + k)] if masked[i] else None,
                                            lp_vals.setdefault(i, []) if lpf[i] else None)
                a = len(toks) - 1
                committed = [int(self.pending[slot]), *toks[:a]]
                if self.strict:
                    self._check_kv(handle.plan, slot, p0, 1 + k, np.asarray(committed, np.int64), k - a)
                    if st & NONFINITE:
                        self._poison_kv(slot, p0, 1 + k)
                bad[i] = st & NONFINITE
                status[i] = st
                self.rec_tok[slot] = committed
                self.rec_bad[slot] = bool(st & NONFINITE)
                self.fold_n[slot] = 1 + a
                self.counts["drafts"] += k
                self.counts["accepted"] += a
                after[i] = p0 + len(toks)
                tokens[i, :len(toks)] = toks
                n_out[i] = len(toks)
                self.pending[slot] = toks[-1]
            self.seq_len[slots] = after
            for i in (i for i in sampling if not mtp[i]):  # the other SAMPLE rows: a token each
                slot, pos = int(slots[i]), int(after[i])
                seed = int(self.seed[slot])
                ctx, mix = self.fp_hash[self.state_slot[slot]], rope_mix(int(self.rope_off[slot])) if \
                    self.encoder_rows else 0
                t = self._sample(seed, pos, ctx, mix)
                allowed = mask(i) if masked[i] else None
                if masked[i]:
                    t = masked_pick(t, allowed)
                    if pos in self.mask_empty_at.get(seed, ()) and not allowed.all():  # never an unconstrained mask
                        t = -1
                        self.fired.add(seed)
                    if t < 0:
                        status[i] |= MASK_EMPTY
                    t = max(t, 0)
                if lpf[i]:
                    lp_vals[i] = [fake_logprobs(seed, pos, int(ctx), mix, t, allowed, vocab)]
                self.pending[slot] = t
                tokens[i, 0] = t
                n_out[i] = 1
                if bad[i]:
                    status[i] |= NONFINITE
            self.sampled[slots[sample]] = True
            if self.report_unsampled:  # a runner that checks the rows that do not sample too
                status[bad & ~sample] |= NONFINITE
            if k:  # every SAMPLE row drafts for its slot's next step (after its pending token)
                for i in sampling:
                    slot = int(slots[i])
                    if status[i]:
                        continue
                    c = int(self.fp_hash[self.state_slot[slot]])
                    if mtp[i]:  # the live state lags this row's commit (window records)
                        c = context_hash(self.rec_tok[slot], c)
                    self._draft(slot, context_hash((int(self.pending[slot]),), c))
                    drafts[i] = self.drafts[slot]
            if status.any():
                failed = status != 0
                tokens[failed] = -1
                n_out[failed] = 0
            for ss, d, slot in handle.end_ckpts:
                if slot:  # an MTP decode row: the live state folded with this step's commit, into a copy
                    self._fold(slot, int(self.fold_n[slot]), d)
                else:
                    self.fp_len[d], self.fp_hash[d], self.fp_bad[d] = self.fp_len[ss], self.fp_hash[ss], self.fp_bad[ss]
                    self.fp_prec[d] = self.fp_prec[ss]
        start = max(now, handle.fwd_end)
        handle.idle_inside = start - handle.fwd_end
        handle.deadline = start + self.sample_s
        self._stream_end = max(self._stream_end, handle.deadline)
        busy = handle.fwd_dur + self.sample_s
        handle.plan = None
        lp = None
        if n and handle.lpf.any():  # each LOGPROBS row's samples (1 + k columns under MTP, unused ones 0)
            rows = np.flatnonzero(handle.lpf)
            w = 1 + self.k
            lp = LogprobRows(rows=rows.astype(np.int32), token=np.zeros((len(rows), w), np.float32),
                             top_ids=np.zeros((len(rows), w, MAX_LOGPROBS), np.int32),
                             top=np.zeros((len(rows), w, MAX_LOGPROBS), np.float32))
            for r, i in enumerate(rows.tolist()):
                for j, (v, ids, vals) in enumerate(lp_vals.get(i, [])):
                    lp.token[r, j], lp.top_ids[r, j], lp.top[r, j] = v, ids, vals
        handle.result = StepResult(tokens=tokens, n_out=n_out, status=status if status.any() else None,
                                   drafts=drafts, logprobs=lp, gpu_busy_ns=int(busy * 1e9),
                                   gpu_idle_before_ns=int(handle.idle_before * 1e9),
                                   gpu_idle_inside_ns=int(handle.idle_inside * 1e9),
                                   gpu_sample_ns=int(self.sample_s * 1e9))  # 0 without a sampling part: not split

    def forward_pending(self, handle: _Handle) -> bool:
        """The simulated stream has not reached the end of that step's forward (`Runner.forward_pending`); False
        without `sample_ms` (module docstring)."""
        return self.sample_s > 0 and time.perf_counter() < handle.fwd_end

    def wait(self, handle: _Handle) -> StepResult:
        if handle.result is None:
            raise RuntimeError("wait() before launch_sample()")
        remaining = handle.deadline - time.perf_counter()
        if self.wait_mode == "sleep":
            # macOS coalesces timers: sleep() overshoots, which would stretch every simulated step. Sleep for part of
            # the wait and spin the rest.
            nap = remaining * _SLEEP_FRACTION - self.spin_s
            if nap > 0:
                time.sleep(nap)
        # "spin" mirrors CUDA's default synchronization (cudaDeviceScheduleAuto spin-waits when there are fewer
        # contexts than cores): the engine thread stays on its core with warm caches between steps.
        while time.perf_counter() < handle.deadline:
            pass
        return handle.result


def load_model(seconds: float) -> None:
    """The fake's model load: `dev.runner.fake.load_s` of sleep, a slow or hung start for tests (its stack names it)."""
    log.warning("loading for %g s (dev.runner.fake.load_s)", seconds)
    time.sleep(seconds)


def make_fake_runner(model_dir: str | Path, opts: Any, mtp_k: int = 0, encoder_rows: int = 0,
                     encoder_budget: int = 0, logprobs: bool = True) -> FakeRunner:
    """Build from the model's config.json (geometry), `dev.runner.fake` options, the MTP depth, (images) the encoder
    cache's rows and the encoder's budget, and whether it takes LOGPROBS rows."""
    if opts.load_s:
        load_model(opts.load_s)
    geo = model_geometry(model_dir) if (Path(model_dir) / "config.json").exists() else {}
    vocab = opts.vocab_size or geo.get("vocab_size") or 32000
    eos_at: dict[int, list[int]] = {}
    for seed, pos in opts.eos:
        eos_at.setdefault(int(seed), []).append(int(pos))
    return FakeRunner(vocab_size=vocab, max_positions=geo.get("max_positions") or 32768,
                      kv_bytes_per_token=geo.get("kv_bytes_per_token") or 69_632,
                      state_slot_bytes=geo.get("state_slot_bytes", 0) if geo else 153_954_304,
                      memory_bytes=int(opts.memory_gb * (1 << 30)), step_ms=opts.step_ms, sample_ms=opts.sample_ms,
                      prefill_us_per_token=opts.prefill_us_per_token, eos_at=eos_at, scripts=opts.scripts,
                      strict=opts.strict or bool(geo.get("window")), wait_mode=opts.wait, mtp_k=mtp_k,
                      accept_p=opts.accept_p, encoder_rows=encoder_rows, encoder_budget=encoder_budget,
                      window=geo.get("window", 0), window_kv_bytes_per_token=geo.get("window_kv_bytes_per_token", 0),
                      bidirectional_images=bool(geo.get("bidirectional_images")) and encoder_rows > 0,
                      draft_kv=not geo.get("window"),  # Gemma 4's drafter (its assistant) writes no KV
                      precise_rows=opts.precise_rows, batch_invariant=opts.batch_invariant, logprobs=logprobs)
