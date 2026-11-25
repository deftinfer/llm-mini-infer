"""The drafters' LM heads (docs/architecture.md, "Speculative decoding"): `DraftHead`, a drafter's head (Qwen's MTP
layer's is the target's, Gemma 4's assistant's its own), whole or reduced; `SubsetHead`, a head over the grammar's
universe (`runner.cuda.heads`); and `fp32_reduction`, the fp32 partial sums of narrow BF16 heads.

Drafts never change a request's tokens (verification keeps the target's samples), only how many are accepted, so a
drafter may score them with a cheaper head than the target's (on Qwen3.8-27B, 248,320 × 5,120 BF16, 2.5 GB):

- **A reduced vocabulary** (FR-Spec's frequency-ranked draft vocabulary): the drafter scores `size` tokens. Which ones
  is derived, never a per-workload file: the tokens the target emits most on this server, counted from the steps'
  results (`observe`), and before (and below) any traffic the tokenizer's merge order, since BPE ids put frequent
  pieces first. The special tokens are always in, and with structured output the tokens made only of JSON's whitespace
  and punctuation (`structural_token_ids`). When too many emitted tokens fall outside the head, the ranking is redone
  and the head rebuilt in place, between steps on the compute stream (`apply`). A target token outside the head is
  never drafted, which costs acceptance at that position, never a wrong token.

The sampler draws each draft with the noise of its true token id (`ids`, ascending, so ties still go to the lowest
id), so a draft and the target's sample at a position share their noise as with the whole vocabulary.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ...kernels import fp8
from . import sync_check as sync_checks

log = logging.getLogger(__name__)

CHECK_EVERY = 4096  # emitted tokens between two checks of the ranking
MISS_TOL = 0.002  # rebuild when more than this share of them was outside the head


STRUCTURAL = frozenset(' \t\n\r{}[],:"')  # JSON's whitespace and punctuation


def structural_token_ids(model_dir: Path) -> list[int]:
    """The tokens made only of JSON's whitespace and punctuation (620 of Qwen3.8's), for a reduced head under
    structured output. JSON allows any whitespace between tokens, and a model can emit long runs of it, which BPE
    merged late: one token of 32 newlines is id 76,374, outside a 32,768-token head ranked by id. Outside the head,
    every draft fails until the next ranking, and such a run advances about one token per verify step. Empty without a
    tokenizer.json."""
    tj = model_dir / "tokenizer.json"
    if not tj.exists():
        return []
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(tj))
    texts = tok.decode_batch([[i] for i in range(tok.get_vocab_size())], skip_special_tokens=False)
    return [i for i, t in enumerate(texts) if t and set(t) <= STRUCTURAL]


def special_token_ids(model_dir: Path) -> list[int]:
    """The tokenizer's added (special) tokens: `tokenizer_config.json`'s added_tokens_decoder, else
    `tokenizer.json`'s added_tokens; empty when neither is there."""
    tc = model_dir / "tokenizer_config.json"
    if tc.exists():
        dec = json.loads(tc.read_text()).get("added_tokens_decoder") or {}
        if dec:
            return sorted(int(i) for i in dec)
    tj = model_dir / "tokenizer.json"
    if tj.exists():
        return sorted(int(t["id"]) for t in json.loads(tj.read_text()).get("added_tokens") or [])
    return []


class DraftHead:
    """The drafter's head: `logits(hidden)` [n, size] fp32 over `ids` (int32 [size], ascending; None: the whole
    vocabulary in order). `head` is the drafter's head [V, hidden] (Qwen's MTP layer: the target's), kept as is."""

    def __init__(self, head: torch.Tensor, *, size: int = 0, special: Iterable[int] = ()):
        self.full = head
        self.V, self.hidden = head.shape
        size = -(-size // fp8.GROUP) * fp8.GROUP if size else 0  # whole 128-row blocks
        self.reduced = 0 < size < self.V
        self.size = size if self.reduced else self.V
        dev = head.device
        self.rebuilds = 0
        self.w = self.ids = None
        if self.reduced:
            self.w = torch.empty((self.size, self.hidden), dtype=head.dtype, device=dev)
        if not self.reduced:
            return
        self.ids = torch.empty(self.size, dtype=torch.int32, device=dev)
        self._ids_host = torch.empty(self.size, dtype=torch.int64, pin_memory=dev.type == "cuda")
        self._copied: torch.cuda.Event | None = None
        self.counts = np.zeros(self.V, np.int64)
        self.prior = 1.0 - np.arange(self.V, dtype=np.float64) / self.V  # token-id (merge) order, below one emission
        forced = np.asarray([i for i in special if 0 <= i < self.V], np.int64)
        self.prior[forced] = np.inf
        self.hot = np.zeros(self.V, bool)
        self.cur = np.zeros(0, np.int64)
        self._seen = self._missed = 0
        self._pending: np.ndarray | None = None
        self._install(self._rank())

    @property
    def bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.w, self.ids) if t is not None)

    def describe(self) -> str:
        what = f"{self.size:,} of {self.V:,} tokens" if self.reduced else "the whole vocabulary"
        return f"{what}, {str(self.full.dtype).removeprefix('torch.')}"

    # -- the step path ---------------------------------------------------------------------------------------------

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """[n, size] fp32: a GEMM in the model dtype, then fp32."""
        return F.linear(hidden, self.w if self.reduced else self.full).float()

    # -- the ranking (host) ------------------------------------------------------------------------------------------

    def observe(self, tokens: np.ndarray) -> None:
        """Count tokens the target emitted (a step's results); every CHECK_EVERY of them, rank again if the head
        missed more than MISS_TOL of them. The new head is installed by the next `apply`."""
        if not self.reduced or not len(tokens):
            return
        np.add.at(self.counts, tokens, 1)
        self._seen += len(tokens)
        self._missed += int((~self.hot[tokens]).sum())
        if self._seen < CHECK_EVERY:
            return
        if self._missed > MISS_TOL * self._seen:
            new = self._rank()
            if not np.array_equal(new, self.cur):
                self._pending = new
        self._seen = self._missed = 0

    def _rank(self) -> np.ndarray:
        """The `size` best tokens by emitted count, then by token id; the special tokens always in. Ascending."""
        score = self.counts + self.prior
        top = np.argpartition(-score, self.size - 1)[:self.size]
        return np.sort(top)

    def apply(self) -> bool:
        """Install a pending ranking: gather the rows on the current stream, so the next step uses it. Graph-safe
        between replays: the head and `ids` keep their addresses."""
        if self._pending is None:
            return False
        new, self._pending = self._pending, None
        added = int((~self.hot[new]).sum())
        self._install(new)
        self.rebuilds += 1
        log.info("draft head re-ranked: %d tokens in, from %d emitted tokens' counts (rebuild %d)", added,
                 int(self.counts.sum()), self.rebuilds)
        return True

    def _install(self, ids: np.ndarray) -> None:
        if self._copied is not None:  # the pinned staging buffer's previous copy is done
            sync_checks.host_wait(self._copied, "the draft head's ids copied again before their last copy ran")
        self.cur = ids
        self.hot[:] = False
        self.hot[ids] = True
        self._ids_host.numpy()[:] = ids
        idx = self._ids_host.to(self.full.device, non_blocking=True)
        self.ids.copy_(idx)
        torch.index_select(self.full, 0, idx, out=self.w)
        if self.full.is_cuda:
            self._copied = torch.cuda.Event()
            self._copied.record()


@contextmanager
def fp32_reduction() -> Iterator[None]:
    """cuBLAS's BF16 GEMMs with fp32 partial sums only, for a head over a few rows. torch allows BF16 partial sums by
    default (`allow_bf16_reduced_precision_reduction`), and cuBLAS splits K that way for a GEMM narrow in N: on SM120 a
    1,024-row head puts many of its logits off the fp32-accumulated result, some by their sign, where the 262,144-row
    head stays within a few ulps, so argmaxes over a mask's tokens would part from the dense head's. With fp32 sums the
    narrow head is at least as close to the fp32 result as the dense one. Read when the GEMM is issued: a CUDA graph
    keeps what its capture saw."""
    m = torch.backends.cuda.matmul
    prev = m.allow_bf16_reduced_precision_reduction
    m.allow_bf16_reduced_precision_reduction = False
    try:
        yield
    finally:
        m.allow_bf16_reduced_precision_reduction = prev


class SubsetHead:
    """A head over a set of token ids padded to a static capacity (the grammar universe's draft head,
    `runner.cuda.heads`): `logits(hidden)` [n, cap] fp32 over `ids` (int32 [cap]: the set's ids ascending, then −1,
    which the sampler never draws: `ops.sample(padded=True)`), the rows gathered from `full` in the model dtype (row 0's
    for the padding: finite, as the NaN check reads every column). `install` rebuilds it in place on the current stream,
    between steps: the buffers keep their addresses, so the CUDA graphs that read them stay valid."""

    def __init__(self, full: torch.Tensor, cap: int):
        self.full, self.cap = full, cap
        self.V, self.hidden = full.shape
        dev = full.device
        self.w = torch.zeros((cap, self.hidden), dtype=full.dtype, device=dev)
        self.ids = torch.full((cap,), -1, dtype=torch.int32, device=dev)
        self._host = torch.full((cap,), -1, dtype=torch.int32, pin_memory=dev.type == "cuda")
        self._copied: torch.cuda.Event | None = None
        self.n = 0

    @property
    def bytes(self) -> int:
        return self.w.numel() * self.w.element_size() + self.ids.numel() * 4

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """[n, cap] fp32: the GEMM in the model dtype over the gathered rows (fp32 partial sums: `fp32_reduction`), then
        fp32."""
        with fp32_reduction():
            return F.linear(hidden, self.w).float()

    def install(self, ids: np.ndarray) -> None:
        """Score `ids` (ascending int32, at most `cap`) from the next step on."""
        n = len(ids)
        if n > self.cap:
            raise ValueError(f"{n} ids for a head of {self.cap} rows")
        if self._copied is not None:  # the pinned staging buffer's previous copy is done
            sync_checks.host_wait(self._copied, "the draft universe's ids copied again before their last copy ran "
                                                "(SubsetHead.install)")
        h = self._host.numpy()
        h[:n] = ids
        h[n:] = -1
        self.ids.copy_(self._host, non_blocking=True)
        torch.index_select(self.full, 0, self.ids.clamp(min=0).long(), out=self.w)
        if self.full.is_cuda:
            self._copied = torch.cuda.Event()
            self._copied.record()
        self.n = n
