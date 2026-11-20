"""`StepPlan`: what the scheduler tells the runner to do in one step (architecture.md, "The step plan").

Struct-of-arrays numpy views, handed to the runner by reference. Sections and their order of effect:

0. **FOLD** `fold_slot/fold_n` (MTP on a model with recurrent state) — apply the first `fold_n` (−1: all) of a slot's
   pending window records, the last decode step's commit, which the runner folds into the live state lazily
   (architecture.md, "Speculative decoding"), and set its drafter hidden vector to the last one's. The scheduler folds a
   live slot it donates as a checkpoint (finish, abort, preemption) in the plan that FREEs it: −1 when no step for the
   slot is in flight, 1 when a decode step is (that step folds the previous commit itself and always commits its first
   token, the pending one). A FOLDed slot must be FREEd in the same plan.
1. **FREE** `free_slot` — clear these slot rows (request finished, aborted or preempted).
2. **ADMIT** `admit_*` — initialise a slot row: live state slot, starting `seq_len` (0, or the length of the cached
   prefix on a prefix-cache hit), sampling parameters (temperature 0 = greedy; top_p 1, top_k 0 and min_p 0 = off)
   and the seed, which is always set. The first step after an ADMIT carries explicit tokens at
   `pos0 = seq_len`. Under MTP, ADMIT also carries the request's **commit limits**: `admit_max_len` (positions
   `>= max_len` never hold a token of the request: prompt plus `max_tokens`) and `admit_stop` [n, STOP_W] (the
   tokens that end it: EOS unless ignored, then its stop token IDs; −1 pads). Verification never accepts a draft
   whose target sample is one of those stop tokens or would put the bonus token at `max_len` or beyond, so such a
   finish is the last token of its step and the committed state never runs past it. A stop token beyond `STOP_W` can
   be overrun: the scheduler drops what follows (`overrun`).
   **Penalties and the logit bias** (`Capabilities.penalties`; architecture.md, "Sampling"), in plans of a builder
   made with the vocabulary (`vocab`; otherwise their widths are 0 and every admission is neutral): `admit_rep`,
   `admit_pres`, `admit_freq` (neutral 1, 0, 0) and `admit_prompt_len`; `ADMIT_PENALTIES` when one of the three is not
   neutral, and then `admit_prompt_bits` [n, ceil(V / 32)] holds bit t % 32 of word t / 32 for each token t at the
   positions [0, seq_len) (the cached prefix: never past the prompt, the scheduler caps such a request's match there;
   zero for the plan's other admissions; width 0 in a plan without a penalised one); `admit_bias_n` entries of
   `admit_bias_ids` [n, w] (distinct, ascending; w the plan's largest count, the rest zero) and `admit_bias_val`. The
   runner counts what follows itself: an explicit token at a position below `prompt_len` is a prompt token, any later
   one (a readmission's recomputed outputs, a forced prefix) and every sample it commits an output token.
3. **PAGES** `page_slot/page_index/page_id` — set `row[slot][index] = page_id` (set semantics: writing an
   existing entry replaces it). Page 0 is the null page. Rows of different slots may name the same page: shared
   pages hold a cached prefix and are read-only.
4. **KV_COPY** `kv_copy_src/kv_copy_dst` — copy whole pages (every attention layer): copy-on-write of a shared
   partial last page, when a hit ends inside a page. Then **STATE_COPY** `state_copy_src/state_copy_dst` — copy whole
   state slots (conv and recurrent state of every linear-attention layer): a cached checkpoint into a live slot on a
   hit. Within each section, every source is read before any destination is written (gather, then scatter), so a
   plan never depends on the order of its own copies.
5. The forward pass over **SCHED** (rows are independent within a step, so a runner may run them in any order; results
   and masks follow plan order). **CKPT** `ckpt_sched/ckpt_offset/ckpt_dst` snapshots the state of SCHED row
   `ckpt_sched` into state slot `ckpt_dst` at `ckpt_offset` tokens after its `pos0`: a multiple of `CKPT_GRID`
   strictly inside the chunk (mid-chunk), `n_query` (end of the chunk), or −1 (end of the step, after sampling; used
   for decode checkpoints). Under MTP a −1 checkpoint of a device-input row holds the state at the step's committed
   position: the live state folded with the step's commit, written to `ckpt_dst` without changing the live slot. A
   checkpoint also holds the drafter's hidden vector at its position − 1 (MTP).
6. Sampling, which advances `seq_len` and the pending token; then the −1 CKPTs. It is enqueued by the time
   `launch_sample` returns; a step without MASKED rows may sample in `launch_forward`.

SCHED rows:
- `sched_slot`, `sched_n_query` (tokens computed: a prefill chunk, or `1 + n_draft` for decode);
- `sched_tok_off`: offset of the explicit tokens in `tokens`, or −1 for **device input**: the slot's pending token
  plus its `n_draft` drafts (drafted by the runner in the slot's previous SAMPLE step), which the host never supplies;
- `sched_pos0`: position of the first query token, equal to the device `seq_len`; −1 for device input;
- `sched_next_tok` (MTP): the token after an explicit row's chunk, −1 on final chunks and device rows;
- `sched_n_draft`: MTP drafts to verify: the plan's count v (1 ≤ v ≤ k) on every device-input row under MTP (a
  uniform `1 + v` query: k unless the scheduler chose fewer, `Capabilities.draft_counts`), else 0;
- `sched_flags`: `SAMPLE` (sample after the last query token: the final prefill chunk, and every decode row);
  `STATE_ZERO` (start from zero recurrent and conv state: the first chunk of an uncached prompt); `MASKED` (a SAMPLE
  row whose samples a grammar constrains: below); `PRECISE` (below); `LOGPROBS` (below).

**PRECISE rows** (`Capabilities.precise_rows`; architecture.md, "The precise prefix"): an explicit row whose block-FP8
projections take BF16 activations against their weights dequantized to BF16, instead of quantizing the activations;
the step's other rows keep the FP8 path, bit for bit (`kernels.fp8.precise_rows`). The scheduler sets it on the rows
that compute a shared prefix from a precise start (STATE_ZERO, or a resume from a precise checkpoint). Never on
device-input rows; a runner that computes none refuses it.

A step moves the device `seq_len` by `n_query` for explicit tokens and by `1 + accepted drafts` for device input; the
sampled (bonus) token becomes pending. Results come back per SCHED row, in plan order (never by slot): a slot can be
freed and readmitted in the very next plan while the old request's result is still in flight.

**MTP drafting.** Every SAMPLE row also drafts tokens for the slot's next step: k, or `draft_n` when the scheduler chose
fewer (0: k); a slot's next device row verifies at most that many, and the runner returns −1 for the drafts past them.
A drafter that writes KV of its own (`Capabilities.draft_kv`, Qwen's MTP layer) keeps it in the pages like every
attention layer's; its entry at position i depends on the token at i + 1. An explicit row computes it at its own
positions, at its last one from `sched_next_tok` (a final chunk uses its sampled token). The first row after an ADMIT
with `seq_len > 0` also recomputes the drafter's KV at `pos0 − 1`, from the hidden vector its live state slot received
with the checkpoint: the page holding `pos0 − 1` must be the slot's own, so an admission that hits the cache on a page
boundary copies that page (KV_COPY). The drafter's chain writes KV at the d − 1 positions after the row's last
committed position (d its drafts): decode rows write up to `2k − 1` positions past their first, which the page
reservation covers (`reserve_positions`, at k whatever the counts), and a SAMPLE explicit row d − 1 past its last token,
which the final chunk's pages cover (sized for k). Gemma 4's assistant writes no KV.

**Grammar masks** (`grammar`; architecture.md, "Structured output"). A `MASKED` row samples only the tokens its masks
allow. The engine writes them between `launch_forward` and `launch_sample` (`runner.base`), one per sample the row
draws, in plan order: an MTP device-input row draws 1 + k (the target's sample after the pending token and after each of
its drafts: mask i holds the grammar's state after the pending token and the first i drafts), any other SAMPLE row one.
Verification never reads the masks after a draft the grammar rejects, which allow the EOS alone. Drafts are sampled
unmasked. `ADMIT_CONSTRAINED` in `admit_flags` marks a request
with a grammar; its SAMPLE rows are MASKED until the grammar terminates (an EOS it accepted, with `ignore_eos`),
unconstrained after that. MASKED is never set on rows without SAMPLE.

**Logprobs** (`Capabilities.logprobs`; architecture.md, "Sampling"). `ADMIT_LOGPROBS` in `admit_flags` marks a request
that asks for them, and `LOGPROBS` each of its SAMPLE rows: the runner returns the values of every sample such a row
draws (`StepResult.logprobs`: 1 + k on an MTP device-input row), and of no other row. LOGPROBS is never set on rows
without SAMPLE, nor on a slot admitted without ADMIT_LOGPROBS.

**Images** (`vision`; architecture.md, "Images"). Three additions, all empty in a plan without images:
- **ENCODE** `enc_*`, after the copies and before the forward: run the image encoder on each image's uint8 pixels
  [3, `enc_hw`] at arena offset `enc_pixels` (`vision.arena`; valid until this step completes), patch grid
  `enc_grid` (t, h, w), and write its `enc_tokens` embeddings, in token order, to the encoder-pool rows
  `enc_rows[enc_row0 : enc_row0 + enc_tokens]`. `enc_key` is the image's span key (for runners that check plans).
  The rows are the scheduler's (`cache.encoder`): the runner never allocates one.
- **Embedding rows** `tok_emb` [n_tokens], aligned with `tokens`: −1 for a text token, else the encoder-pool row that
  replaces it (an image placeholder). `sched_next_emb` (MTP with a drafter that writes KV): the row of the token after
  a non-final chunk (`sched_next_tok`), −1 when it is text: the drafter's input at an image position is the image's
  embedding.
- **M-RoPE** (families whose RoPE takes 3-D positions, `vision.family.positions`): `tok_rope` [3, n_tokens], aligned
  with `tokens`: each explicit token's rope position minus its sequence position (0 for every token of a request
  without images). `sched_rope_prev` [n_sched, 3]: the deltas at `pos0 − 1` (the drafter's KV a resumed admission
  recomputes). `admit_rope_off` [n_admit]: the delta of every token after the request's last image, which a
  device-input row (and the drafter's chain) adds to its device-side positions. Explicit rows of a request with images
  carry the `MROPE` flag. Sequence positions (KV slots, `seq_len`, the sampler's noise) are unchanged: only RoPE sees
  the deltas.

**SPANS** `span_sched/span_start/span_end` (`Capabilities.bidirectional_images`, Gemma 4's
`use_bidirectional_attention: "vision"`): the image spans an explicit row prefills, by its index in SCHED and in
sequence positions [start, end). The model's queries inside a span see every key of the span, later ones included (in
the layers the model says: Gemma 4's sliding layers, within their window); every other query stays causal. The
scheduler never splits a span across rows or steps and never resumes inside one, so a span always lies inside its
row's chunk. Empty for other models.

**Window pages** (sliding-window models, `Capabilities.window`). The sliding layers' KV lives in window pages with a
page table of their own per slot: `wpage_slot/wpage_index/wpage_id` set its entries (PAGES, same semantics; an entry set
to 0, the null page, is one the scheduler released), `wkv_copy_src/wkv_copy_dst` copy whole window pages (KV_COPY, with
the other copies: gather, then scatter). A row whose first query is at position q reads the sliding layers' KV of
positions ≥ q − W + 1 only, from page index `(max(0, q − W + 1)) // window_page_size` on: the scheduler releases older
pages, clearing their entries, and may reuse them in the same plan (steps in flight read them first, stream order). FREE
clears both tables; the scheduler sends no clearing entry for a slot the plan FREEs, and a runner skips one. Window
pages and full pages are pages of one arena (`cache.pools.size_arena`), one id space: a page is one kind from its
allocation to its free, so a page id is named by PAGES or by window PAGES, never both at once. Plans of other models
leave both empty.

Lifetime: the arrays are views of scheduler-owned buffers that are reused two plans later. The runner may read
them until its `launch_sample` for this step returns, and must copy anything it needs after that.
"""

from __future__ import annotations

import numpy as np

# sched_flags
SAMPLE = 1
STATE_ZERO = 2
MASKED = 4  # a SAMPLE row's samples are grammar-constrained (masks in launch_sample)
MROPE = 8  # the row's request has images (its tokens' M-RoPE deltas are in tok_rope)
PRECISE = 16  # an explicit row computed with BF16 activations (the precise prefix, above)
LOGPROBS = 32  # a SAMPLE row whose samples' logprobs the runner returns (above)

# admit_flags
ADMIT_LOGPROBS = 1  # the request asks for logprobs (its SAMPLE rows are LOGPROBS)
ADMIT_CONSTRAINED = 2  # the request has a grammar (its SAMPLE rows are MASKED until it terminates)
ADMIT_PENALTIES = 4  # a penalty is not neutral: the runner keeps the request's prompt and output counts

CKPT_GRID = 64  # GDN chunk size: mid-chunk checkpoint offsets are multiples of it (where a cut replays its chunks)
CKPT_END_OF_STEP = -1
FOLD_ALL = -1  # fold_n: every pending window record
STOP_W = 8  # admit_stop width: a request's stop tokens the runner knows (more: the scheduler cuts)
MAX_LOGIT_BIAS = 1024  # admit_bias_ids width: a request's logit_bias entries at most (vLLM's v2 runner's limit)

I32 = np.int32


def reserve_positions(steps_in_flight: int, k: int) -> int:
    """Positions past the device `seq_len` mirror that a decode row's pages must cover: D steps of 1 + k tokens each,
    plus the drafter's chain (k − 1 positions past the verify window of the last one) under MTP."""
    return steps_in_flight * (1 + k) + max(0, k - 1)


class StepPlan:
    __slots__ = (
        "admit_bias_ids",
        "admit_bias_n",
        "admit_bias_val",
        "admit_flags",
        "admit_freq",
        "admit_max_len",
        "admit_min_p",
        "admit_pres",
        "admit_prompt_bits",
        "admit_prompt_len",
        "admit_rep",
        "admit_rope_off",
        "admit_seed",
        "admit_seq_len",
        "admit_slot",
        "admit_state_slot",
        "admit_stop",
        "admit_temperature",
        "admit_top_k",
        "admit_top_p",
        "ckpt_dst",
        "ckpt_offset",
        "ckpt_sched",
        "draft_n",
        "enc_grid",
        "enc_hw",
        "enc_key",
        "enc_pixels",
        "enc_row0",
        "enc_rows",
        "enc_tokens",
        "fold_n",
        "fold_slot",
        "free_slot",
        "kv_copy_dst",
        "kv_copy_src",
        "max_query_len",
        "n_query_tokens",
        "page_id",
        "page_index",
        "page_slot",
        "sched_flags",
        "sched_n_draft",
        "sched_n_query",
        "sched_next_emb",
        "sched_next_tok",
        "sched_pos0",
        "sched_rope_prev",
        "sched_slot",
        "sched_tok_off",
        "span_end",
        "span_sched",
        "span_start",
        "state_copy_dst",
        "state_copy_src",
        "step_id",
        "tok_emb",
        "tok_rope",
        "tokens",
        "wkv_copy_dst",
        "wkv_copy_src",
        "wpage_id",
        "wpage_index",
        "wpage_slot",
    )

    step_id: int
    fold_slot: np.ndarray
    fold_n: np.ndarray
    free_slot: np.ndarray
    admit_slot: np.ndarray
    admit_state_slot: np.ndarray
    admit_seq_len: np.ndarray
    admit_temperature: np.ndarray
    admit_top_p: np.ndarray
    admit_top_k: np.ndarray
    admit_min_p: np.ndarray
    admit_seed: np.ndarray
    admit_flags: np.ndarray
    admit_max_len: np.ndarray
    admit_stop: np.ndarray
    # penalties and the logit bias (widths 0 without them)
    admit_rep: np.ndarray
    admit_pres: np.ndarray
    admit_freq: np.ndarray
    admit_prompt_len: np.ndarray
    admit_prompt_bits: np.ndarray
    admit_bias_n: np.ndarray
    admit_bias_ids: np.ndarray
    admit_bias_val: np.ndarray
    page_slot: np.ndarray
    page_index: np.ndarray
    page_id: np.ndarray
    kv_copy_src: np.ndarray
    kv_copy_dst: np.ndarray
    state_copy_src: np.ndarray
    state_copy_dst: np.ndarray
    sched_slot: np.ndarray
    sched_n_query: np.ndarray
    sched_tok_off: np.ndarray
    sched_pos0: np.ndarray
    sched_n_draft: np.ndarray
    sched_next_tok: np.ndarray
    sched_flags: np.ndarray
    ckpt_sched: np.ndarray
    ckpt_offset: np.ndarray
    ckpt_dst: np.ndarray
    tokens: np.ndarray
    n_query_tokens: int
    max_query_len: int
    draft_n: int  # MTP: the drafts every SAMPLE row of the plan makes when fewer than k (0: k; the module docstring)
    # images (empty without)
    enc_key: np.ndarray
    enc_tokens: np.ndarray
    enc_grid: np.ndarray
    enc_hw: np.ndarray
    enc_pixels: np.ndarray
    enc_row0: np.ndarray
    enc_rows: np.ndarray
    tok_emb: np.ndarray
    sched_next_emb: np.ndarray
    tok_rope: np.ndarray
    sched_rope_prev: np.ndarray
    admit_rope_off: np.ndarray
    # bidirectional image spans (empty without)
    span_sched: np.ndarray
    span_start: np.ndarray
    span_end: np.ndarray
    # the window pool (empty without)
    wpage_slot: np.ndarray
    wpage_index: np.ndarray
    wpage_id: np.ndarray
    wkv_copy_src: np.ndarray
    wkv_copy_dst: np.ndarray

    @property
    def n_sched(self) -> int:
        return len(self.sched_slot)

    def describe(self) -> str:
        out = (f"plan {self.step_id}: " +
               (f"fold {list(zip(self.fold_slot.tolist(), self.fold_n.tolist()))} " if len(self.fold_slot) else "") +
               f"free {self.free_slot.tolist()} admit {self.admit_slot.tolist()} "
               f"seq_len {self.admit_seq_len.tolist()} pages {len(self.page_id)} sched slot {self.sched_slot.tolist()} "
               f"n_query {self.sched_n_query.tolist()} tok_off {self.sched_tok_off.tolist()} "
               f"pos0 {self.sched_pos0.tolist()} flags {self.sched_flags.tolist()}")
        if len(self.wpage_id) or len(self.wkv_copy_src):
            out += (f" window pages {len(self.wpage_id)} wkv_copy "
                    f"{list(zip(self.wkv_copy_src.tolist(), self.wkv_copy_dst.tolist()))}")
        if len(self.kv_copy_src) or len(self.state_copy_src) or len(self.ckpt_sched):
            out += (f" kv_copy {list(zip(self.kv_copy_src.tolist(), self.kv_copy_dst.tolist()))}"
                    f" state_copy {list(zip(self.state_copy_src.tolist(), self.state_copy_dst.tolist()))}"
                    f" ckpt {list(zip(self.ckpt_sched.tolist(), self.ckpt_offset.tolist(), self.ckpt_dst.tolist()))}")
        if len(self.enc_key):
            out += f" encode {list(zip([hex(k) for k in self.enc_key.tolist()], self.enc_tokens.tolist()))}"
        if len(self.tok_emb):
            out += f" image tokens {int((self.tok_emb >= 0).sum())}"
        if len(self.span_sched):
            out += f" spans {list(zip(self.span_sched.tolist(), self.span_start.tolist(), self.span_end.tolist()))}"
        return out


class _Buffers:
    def __init__(self, max_seqs: int, max_tokens: int, max_pages: int, max_ckpts: int, max_wpages: int = 0,
                 vocab: int = 0):
        s = max_seqs + 1
        self.wpage_slot = np.empty(max_wpages, I32)
        self.wpage_index = np.empty(max_wpages, I32)
        self.wpage_id = np.empty(max_wpages, I32)
        self.wkv_copy_src = np.empty(s, I32)  # at most one copy-on-write window page per admission
        self.wkv_copy_dst = np.empty(s, I32)
        self.fold_slot = np.empty(s, I32)
        self.fold_n = np.empty(s, I32)
        self.free_slot = np.empty(s, I32)
        self.admit_slot = np.empty(s, I32)
        self.admit_state_slot = np.empty(s, I32)
        self.admit_seq_len = np.empty(s, I32)
        self.admit_temperature = np.empty(s, np.float32)
        self.admit_top_p = np.empty(s, np.float32)
        self.admit_top_k = np.empty(s, I32)
        self.admit_min_p = np.empty(s, np.float32)
        self.admit_seed = np.empty(s, np.uint64)
        self.admit_flags = np.empty(s, np.uint32)
        self.admit_max_len = np.empty(s, I32)
        self.admit_stop = np.empty((s, STOP_W), I32)
        self.admit_rep = np.empty(s, np.float32)
        self.admit_pres = np.empty(s, np.float32)
        self.admit_freq = np.empty(s, np.float32)
        self.admit_prompt_len = np.empty(s, I32)
        self.admit_prompt_bits = np.empty((s, -(-vocab // 32)), np.uint32)  # np.empty: touched by penalised ones only
        self.admit_bias_n = np.empty(s, I32)
        self.admit_bias_ids = np.empty((s, MAX_LOGIT_BIAS if vocab else 0), I32)
        self.admit_bias_val = np.empty((s, MAX_LOGIT_BIAS if vocab else 0), np.float32)
        self.admit_rope_off = np.zeros(s, I32)
        self.page_slot = np.empty(max_pages, I32)
        self.page_index = np.empty(max_pages, I32)
        self.page_id = np.empty(max_pages, I32)
        self.sched_slot = np.empty(s, I32)
        self.sched_n_query = np.empty(s, I32)
        self.sched_tok_off = np.empty(s, I32)
        self.sched_pos0 = np.empty(s, I32)
        self.sched_n_draft = np.zeros(s, I32)
        self.sched_next_tok = np.full(s, -1, I32)
        self.sched_flags = np.empty(s, np.uint32)
        self.tokens = np.empty(max_tokens, I32)
        self.tok_emb = np.empty(max_tokens, I32)
        self.sched_next_emb = np.full(s, -1, I32)
        self.tok_rope = np.empty((3, max_tokens), I32)
        self.sched_rope_prev = np.zeros((s, 3), I32)
        self.kv_copy_src = np.empty(s, I32)  # at most one copy-on-write page per admission
        self.kv_copy_dst = np.empty(s, I32)
        self.state_copy_src = np.empty(s, I32)  # at most one checkpoint restore per admission
        self.state_copy_dst = np.empty(s, I32)
        self.ckpt_sched = np.empty(max_ckpts, I32)
        self.ckpt_offset = np.empty(max_ckpts, I32)
        self.ckpt_dst = np.empty(max_ckpts, I32)
        self.span_sched = np.empty(max_tokens, I32)  # a span holds at least one token
        self.span_start = np.empty(max_tokens, I32)
        self.span_end = np.empty(max_tokens, I32)


def page_write_capacity(max_seqs: int, max_tokens: int, page_size: int, max_model_len: int, reserve: int = 2) -> int:
    """Page writes one plan may carry: each decode row crosses at most a few page boundaries for the positions it
    reserves (`reserve_positions`), prefill chunks write at most max_tokens / page_size pages in all, plus a few
    each, and an admission that hits the prefix cache writes every shared page of its prefix."""
    return max_seqs * (3 + -(-reserve // page_size) + -(-max_model_len // page_size)) + max_tokens // page_size + 64


def window_page_write_capacity(max_seqs: int, max_tokens: int, page_size: int, window: int, reserve: int = 2) -> int:
    """Window-pool page writes one plan may carry: as `page_write_capacity`, but a hit shares only a window's pages
    (the last `window − 1` positions before it); plus the clears of the pages the rows moved past since the previous
    plan: a row moves by at most its previous chunk, so the clears are at most as many as the new pages (one per page
    the plan's tokens fill and per row's boundary), hence 2 x the per-row 3 and 2 x max_tokens // page_size (a bound,
    not an exact count)."""
    return max_seqs * (6 + -(-reserve // page_size) + -(-window // page_size)) + 2 * (max_tokens // page_size) + 64


class PlanBuilder:
    """Builds plans into two alternating buffer sets, the second a margin: the runner reads a plan's arrays only until
    it launches the plan's sampling, which the engine does before it builds the next plan (the contract above).
    `page_size`: the full-attention KV page; `window_page_size` (sliding-window models; 0: `page_size`): the window
    page; `grid`: the checkpoint grid (`Capabilities.checkpoint_grid`); `vocab` (0: none): the vocabulary size of
    the penalties' and the logit bias's arrays (`Capabilities.penalties`)."""

    def __init__(self, max_seqs: int, max_tokens: int, page_size: int, max_model_len: int = 0, mtp_k: int = 0,
                 window: int = 0, window_page_size: int = 0, grid: int = CKPT_GRID, vocab: int = 0):
        max_pages = page_write_capacity(max_seqs, max_tokens, page_size, max_model_len, reserve_positions(2, mtp_k))
        self.k = mtp_k
        # Checkpoints per step: mid-chunk ones are at most one per `grid` query tokens; every row may add an
        # end-of-chunk or end-of-step one, and a waiting sibling a branch point on another row.
        max_ckpts = max_tokens // grid + 2 * (max_seqs + 1) + 16
        max_wpages = window_page_write_capacity(max_seqs, max_tokens, window_page_size or page_size, window,
                                                reserve_positions(2, mtp_k)) if window else 0
        self.vocab = vocab
        self._bufs = tuple(_Buffers(max_seqs, max_tokens, max_pages, max_ckpts, max_wpages, vocab) for _ in range(2))
        self._flip = 0
        self.b = self._bufs[0]
        self.n_fold = self.n_free = self.n_admit = self.n_pages = self.n_sched = self.n_tokens = self.n_device = 0
        self.n_kv_copy = self.n_state_copy = self.n_ckpt = self.n_wpages = self.n_wkv_copy = self.n_spans = 0
        self.max_query_len = 0
        self.v = mtp_k  # the drafts the plan's device rows verify
        self.draft_n = 0  # the drafts its SAMPLE rows make when fewer than k (0: k)
        self.emb_on = self.rope_on = False
        self.enc: list[tuple[int, int, tuple[int, int, int], tuple[int, int], int, np.ndarray]] = []

    def begin(self) -> None:
        self._flip ^= 1
        self.b = self._bufs[self._flip]
        self.n_fold = self.n_free = self.n_admit = self.n_pages = self.n_sched = self.n_tokens = self.n_device = 0
        self.n_kv_copy = self.n_state_copy = self.n_ckpt = self.n_wpages = self.n_wkv_copy = self.n_spans = 0
        self.max_query_len = 0
        self.v, self.draft_n = self.k, 0
        self.emb_on = self.rope_on = False  # the plan has image tokens / M-RoPE rows
        self.enc = []
        self.max_bias, self.any_pen = 0, False  # the widths of the ADMIT terms' arrays (`finish`)

    def add_fold(self, slot: int, n: int) -> None:
        """FOLD: apply the first n pending window records of `slot` to its live state (FOLD_ALL: every one)."""
        self.b.fold_slot[self.n_fold], self.b.fold_n[self.n_fold] = slot, n
        self.n_fold += 1

    def add_free(self, slot: int) -> None:
        self.b.free_slot[self.n_free] = slot
        self.n_free += 1

    def add_admit(self, slot: int, state_slot: int, seq_len: int, temperature: float, top_p: float, top_k: int,
                  min_p: float, seed: int, flags: int = 0, max_len: int = 1 << 30,
                  stop: np.ndarray | list[int] = (), rope_off: int = 0, rep: float = 1.0, pres: float = 0.0,
                  freq: float = 0.0, prompt_len: int = 0, prompt: np.ndarray | None = None,
                  bias_ids: np.ndarray | list[int] = (), bias_val: np.ndarray | list[float] = ()) -> None:
        """ADMIT. Penalties (not neutral: `ADMIT_PENALTIES` is set here) and the logit bias need a builder made with
        `vocab`; `prompt`: the request's tokens at positions [0, seq_len), prompt tokens all (their bits)."""
        b, i = self.b, self.n_admit
        nb = len(bias_ids)
        pen = rep != 1.0 or pres != 0.0 or freq != 0.0
        if (pen or nb) and not self.vocab:
            raise ValueError("penalties or a logit bias in a plan without them (PlanBuilder's vocab)")
        b.admit_rep[i], b.admit_pres[i], b.admit_freq[i] = rep, pres, freq
        b.admit_prompt_len[i] = prompt_len
        b.admit_bias_n[i] = nb
        if nb:
            b.admit_bias_ids[i, :nb] = bias_ids
            b.admit_bias_val[i, :nb] = bias_val
            self.max_bias = max(self.max_bias, nb)
        if pen:
            self.any_pen = True
            flags |= ADMIT_PENALTIES
            seen = np.zeros(b.admit_prompt_bits.shape[1] * 32, bool)
            if seq_len:
                seen[np.asarray(prompt)[:seq_len]] = True
            b.admit_prompt_bits[i] = np.packbits(seen, bitorder="little").view("<u4")
        else:
            flags &= ~ADMIT_PENALTIES
        b.admit_rope_off[i] = rope_off
        b.admit_max_len[i] = min(max_len, (1 << 31) - 1)
        b.admit_stop[i] = -1
        n = min(len(stop), STOP_W)
        if n:
            b.admit_stop[i, :n] = stop[:n]
        b.admit_slot[i] = slot
        b.admit_state_slot[i] = state_slot
        b.admit_seq_len[i] = seq_len
        b.admit_temperature[i] = temperature
        b.admit_top_p[i] = top_p
        b.admit_top_k[i] = top_k
        b.admit_min_p[i] = min_p
        b.admit_seed[i] = seed
        b.admit_flags[i] = flags
        self.n_admit = i + 1

    def add_pages(self, slot: int, first_index: int, page_ids: list[int]) -> None:
        n, i = len(page_ids), self.n_pages
        if i + n > len(self.b.page_id):
            raise RuntimeError("page-write capacity exceeded")
        self.b.page_slot[i:i + n] = slot
        self.b.page_index[i:i + n] = np.arange(first_index, first_index + n, dtype=I32)
        self.b.page_id[i:i + n] = page_ids
        self.n_pages = i + n

    def add_wpages(self, slot: int, first_index: int, page_ids: list[int]) -> None:
        """PAGES of the window pool (sliding-window models)."""
        n, i = len(page_ids), self.n_wpages
        if i + n > len(self.b.wpage_id):
            raise RuntimeError("window page-write capacity exceeded")
        self.b.wpage_slot[i:i + n] = slot
        self.b.wpage_index[i:i + n] = np.arange(first_index, first_index + n, dtype=I32)
        self.b.wpage_id[i:i + n] = page_ids
        self.n_wpages = i + n

    def add_page_runs(self, slots: list[int], first: list[int], counts: list[int], page_ids: list[int],
                      window: bool = False) -> None:
        """`add_pages` (`add_wpages` if `window`) for several slots in turn, at once: slot `slots[j]` gets `counts[j]`
        pages from page index `first[j]`, the next ones of `page_ids`."""
        b, n = self.b, len(page_ids)
        dst = (b.wpage_slot, b.wpage_index, b.wpage_id) if window else (b.page_slot, b.page_index, b.page_id)
        i = self.n_wpages if window else self.n_pages
        if i + n > len(dst[2]):
            raise RuntimeError(f"{'window ' if window else ''}page-write capacity exceeded")
        if n == len(slots):  # a page each (rows crossing a page boundary together)
            dst[0][i:i + n] = slots
            dst[1][i:i + n] = first
        else:
            k = np.asarray(counts)
            dst[0][i:i + n] = np.repeat(slots, k)
            dst[1][i:i + n] = np.arange(n) + np.repeat(np.asarray(first) - (np.cumsum(k) - k), k)
        dst[2][i:i + n] = page_ids
        if window:
            self.n_wpages = i + n
        else:
            self.n_pages = i + n

    def add_wkv_copy(self, src: int, dst: int) -> None:
        """KV_COPY of a whole window page."""
        b, i = self.b, self.n_wkv_copy
        b.wkv_copy_src[i], b.wkv_copy_dst[i] = src, dst
        self.n_wkv_copy = i + 1

    def add_kv_copy(self, src: int, dst: int) -> None:
        b, i = self.b, self.n_kv_copy
        b.kv_copy_src[i], b.kv_copy_dst[i] = src, dst
        self.n_kv_copy = i + 1

    def add_state_copy(self, src: int, dst: int) -> None:
        b, i = self.b, self.n_state_copy
        b.state_copy_src[i], b.state_copy_dst[i] = src, dst
        self.n_state_copy = i + 1

    def add_ckpt(self, row: int, offset: int, dst: int) -> None:
        """Snapshot SCHED row `row`'s state at `offset` tokens after its pos0 (or −1: end of step) into `dst`."""
        b, i = self.b, self.n_ckpt
        if i == len(b.ckpt_dst):
            raise RuntimeError("checkpoint capacity exceeded")
        b.ckpt_sched[i], b.ckpt_offset[i], b.ckpt_dst[i] = row, offset, dst
        self.n_ckpt = i + 1

    def add_span(self, row: int, start: int, end: int) -> None:
        """A bidirectional image span [start, end) of SCHED row `row` (positions; inside the row's chunk)."""
        b, i = self.b, self.n_spans
        b.span_sched[i], b.span_start[i], b.span_end[i] = row, start, end
        self.n_spans = i + 1

    def add_encode(self, key: int, tokens: int, grid: tuple[int, int, int], hw: tuple[int, int], pixels: int,
                   rows: np.ndarray) -> None:
        """ENCODE one image into encoder-pool `rows` (its `tokens` embeddings, in order)."""
        self.enc.append((key, tokens, grid, hw, pixels, rows))

    def add_device_rows(self, slots: np.ndarray, masked: np.ndarray | None = None,
                        logprobs: np.ndarray | None = None, n_draft: int | None = None) -> None:
        """Decode rows with device input (`tok_off` = `pos0` = −1, `SAMPLE`; 1 + v query tokens, v drafts), vectorized.
        `masked` and `logprobs` (bool per row): the rows that also get `MASKED`, `LOGPROBS`. `n_draft`: v, the plan's
        drafts to verify (one count for all its device rows; default k)."""
        b, i, n = self.b, self.n_sched, len(slots)
        k = (self.v if self.n_device else self.k) if n_draft is None else n_draft
        if not (1 if self.k else 0) <= k <= self.k or (self.n_device and k != self.v):
            raise ValueError(f"device rows verify {k} drafts; the plan's are {self.v} (1 to {self.k})")
        self.v = k
        b.sched_slot[i:i + n] = slots
        b.sched_n_query[i:i + n] = 1 + k
        b.sched_n_draft[i:i + n] = k
        b.sched_next_tok[i:i + n] = -1
        b.sched_tok_off[i:i + n] = -1
        b.sched_pos0[i:i + n] = -1
        b.sched_flags[i:i + n] = SAMPLE if masked is None else np.where(masked, SAMPLE | MASKED, SAMPLE)
        if logprobs is not None:
            b.sched_flags[i:i + n] |= np.where(logprobs, LOGPROBS, 0).astype(np.uint32)
        b.sched_next_emb[i:i + n] = -1
        b.sched_rope_prev[i:i + n] = 0
        self.n_sched = i + n
        self.n_device += n
        if n:
            self.max_query_len = max(self.max_query_len, 1 + k)

    def add_explicit_row(self, slot: int, tokens: np.ndarray, pos0: int, flags: int, next_tok: int = -1,
                         emb: np.ndarray | None = None, next_emb: int = -1, rope: np.ndarray | None = None,
                         rope_prev: np.ndarray | None = None) -> None:
        """`emb`: the tokens' encoder-pool rows (−1: text) when the chunk holds image tokens; `rope` [3, n] and
        `rope_prev` [3]: M-RoPE deltas of a request with images (`MROPE` is then set)."""
        b, i, n, off = self.b, self.n_sched, len(tokens), self.n_tokens
        b.tokens[off:off + n] = tokens
        b.sched_slot[i] = slot
        b.sched_n_query[i] = n
        b.sched_n_draft[i] = 0
        b.sched_next_tok[i] = next_tok
        b.sched_tok_off[i] = off
        b.sched_pos0[i] = pos0
        if emb is not None or next_emb >= 0:
            if not self.emb_on:
                self.emb_on = True
                b.tok_emb[:off] = -1
                b.sched_next_emb[:i] = -1
            b.tok_emb[off:off + n] = -1 if emb is None else emb
        elif self.emb_on:
            b.tok_emb[off:off + n] = -1
        b.sched_next_emb[i] = next_emb
        if rope is not None:
            flags |= MROPE
            if not self.rope_on:
                self.rope_on = True
                b.tok_rope[:, :off] = 0
                b.sched_rope_prev[:i] = 0
            b.tok_rope[:, off:off + n] = rope
            b.sched_rope_prev[i] = 0 if rope_prev is None else rope_prev
        elif self.rope_on:
            b.tok_rope[:, off:off + n] = 0
            b.sched_rope_prev[i] = 0
        b.sched_flags[i] = flags
        self.n_sched = i + 1
        self.n_tokens = off + n
        self.max_query_len = max(self.max_query_len, n)

    def extend_row(self, i: int, tokens: np.ndarray, rope: np.ndarray | None = None) -> None:
        """Append `tokens` to explicit SCHED row i (a final chunk's forced tokens, which the scheduler adds once every
        CKPT record of the plan is placed): the explicit tokens of the rows after it move up; its CKPT offsets (from
        its pos0), its spans (positions) and the encodes are unchanged. `rope` [3, n]: their M-RoPE deltas when the
        plan has M-RoPE rows (0 if None); their embedding rows, when the plan has any, are −1 (text)."""
        b, n, nt = self.b, len(tokens), self.n_tokens
        off = int(b.sched_tok_off[i]) + int(b.sched_n_query[i])
        b.tokens[off + n:nt + n] = b.tokens[off:nt]  # numpy copies overlapping slices as if through a buffer
        b.tokens[off:off + n] = tokens
        if self.emb_on:
            b.tok_emb[off + n:nt + n] = b.tok_emb[off:nt]
            b.tok_emb[off:off + n] = -1
        if self.rope_on:
            b.tok_rope[:, off + n:nt + n] = b.tok_rope[:, off:nt]
            b.tok_rope[:, off:off + n] = 0 if rope is None else rope
        later = b.sched_tok_off[i + 1:self.n_sched]  # a view: device rows keep −1
        later[later >= 0] += n
        b.sched_n_query[i] += n
        self.n_tokens = nt + n
        self.max_query_len = max(self.max_query_len, int(b.sched_n_query[i]))

    def finish(self, step_id: int) -> StepPlan:
        b, p = self.b, StepPlan()
        p.step_id = step_id
        p.fold_slot, p.fold_n = b.fold_slot[:self.n_fold], b.fold_n[:self.n_fold]
        p.free_slot = b.free_slot[:self.n_free]
        na = self.n_admit
        p.admit_slot = b.admit_slot[:na]
        p.admit_state_slot = b.admit_state_slot[:na]
        p.admit_seq_len = b.admit_seq_len[:na]
        p.admit_temperature = b.admit_temperature[:na]
        p.admit_top_p = b.admit_top_p[:na]
        p.admit_top_k = b.admit_top_k[:na]
        p.admit_min_p = b.admit_min_p[:na]
        p.admit_seed = b.admit_seed[:na]
        p.admit_flags = b.admit_flags[:na]
        p.admit_max_len = b.admit_max_len[:na]
        p.admit_stop = b.admit_stop[:na]
        p.admit_rep, p.admit_pres, p.admit_freq = b.admit_rep[:na], b.admit_pres[:na], b.admit_freq[:na]
        p.admit_prompt_len, p.admit_bias_n = b.admit_prompt_len[:na], b.admit_bias_n[:na]
        w = self.max_bias  # the arrays' every entry set (plans are compared whole: tests/scripts/plandiff.py)
        if w:
            for i in np.flatnonzero(b.admit_bias_n[:na] < w).tolist():
                b.admit_bias_ids[i, b.admit_bias_n[i]:w] = 0
                b.admit_bias_val[i, b.admit_bias_n[i]:w] = 0
        p.admit_bias_ids, p.admit_bias_val = b.admit_bias_ids[:na, :w], b.admit_bias_val[:na, :w]
        if self.any_pen:
            b.admit_prompt_bits[:na][(b.admit_flags[:na] & ADMIT_PENALTIES) == 0] = 0
            p.admit_prompt_bits = b.admit_prompt_bits[:na]
        else:
            p.admit_prompt_bits = b.admit_prompt_bits[:na, :0]
        npg = self.n_pages
        p.page_slot = b.page_slot[:npg]
        p.page_index = b.page_index[:npg]
        p.page_id = b.page_id[:npg]
        n = self.n_kv_copy
        p.kv_copy_src, p.kv_copy_dst = b.kv_copy_src[:n], b.kv_copy_dst[:n]
        n = self.n_state_copy
        p.state_copy_src, p.state_copy_dst = b.state_copy_src[:n], b.state_copy_dst[:n]
        n = self.n_wpages
        p.wpage_slot, p.wpage_index, p.wpage_id = b.wpage_slot[:n], b.wpage_index[:n], b.wpage_id[:n]
        n = self.n_wkv_copy
        p.wkv_copy_src, p.wkv_copy_dst = b.wkv_copy_src[:n], b.wkv_copy_dst[:n]
        n = self.n_ckpt
        p.ckpt_sched, p.ckpt_offset, p.ckpt_dst = b.ckpt_sched[:n], b.ckpt_offset[:n], b.ckpt_dst[:n]
        n = self.n_spans
        p.span_sched, p.span_start, p.span_end = b.span_sched[:n], b.span_start[:n], b.span_end[:n]
        ns = self.n_sched
        p.sched_slot = b.sched_slot[:ns]
        p.sched_n_query = b.sched_n_query[:ns]
        p.sched_tok_off = b.sched_tok_off[:ns]
        p.sched_pos0 = b.sched_pos0[:ns]
        p.sched_n_draft = b.sched_n_draft[:ns]
        p.sched_next_tok = b.sched_next_tok[:ns]
        p.sched_flags = b.sched_flags[:ns]
        p.tokens = b.tokens[:self.n_tokens]
        p.max_query_len = self.max_query_len
        p.n_query_tokens = self.n_tokens + self.n_device * (1 + self.v)
        p.draft_n = self.draft_n
        p.admit_rope_off = b.admit_rope_off[:na]
        nt = self.n_tokens
        p.tok_emb = b.tok_emb[:nt] if self.emb_on else _EMPTY32
        p.sched_next_emb = b.sched_next_emb[:ns] if self.emb_on else _EMPTY32
        p.tok_rope = b.tok_rope[:, :nt] if self.rope_on else _EMPTY_ROPE
        p.sched_rope_prev = b.sched_rope_prev[:ns] if self.rope_on else _EMPTY_PREV
        enc = self.enc
        if enc:
            p.enc_key = np.asarray([e[0] for e in enc], np.int64)
            p.enc_tokens = np.asarray([e[1] for e in enc], I32)
            p.enc_grid = np.asarray([e[2] for e in enc], I32).reshape(-1, 3)
            p.enc_hw = np.asarray([e[3] for e in enc], I32).reshape(-1, 2)
            p.enc_pixels = np.asarray([e[4] for e in enc], np.int64)
            p.enc_row0 = np.concatenate([[0], np.cumsum(p.enc_tokens)[:-1]]).astype(I32)
            p.enc_rows = np.concatenate([e[5] for e in enc]).astype(np.int64)
        else:
            p.enc_key, p.enc_tokens, p.enc_grid, p.enc_hw = _EMPTY64, _EMPTY32, _EMPTY_GRID, _EMPTY_HW
            p.enc_pixels, p.enc_row0, p.enc_rows = _EMPTY64, _EMPTY32, _EMPTY64
        return p


_EMPTY32 = np.zeros(0, I32)
_EMPTY64 = np.zeros(0, np.int64)
_EMPTY_ROPE = np.zeros((3, 0), I32)
_EMPTY_PREV = np.zeros((0, 3), I32)
_EMPTY_GRID = np.zeros((0, 3), I32)
_EMPTY_HW = np.zeros((0, 2), I32)
