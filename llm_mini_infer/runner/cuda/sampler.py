"""Counter-based Gumbel-max sampling (docs/architecture.md, "Sampling"): the torch reference, and the definition the
fused CUDA sampler (`kernels/csrc/sampler.cu`, `kernels.ops.sample`) reproduces token for token.

The noise for token `v` of a request is a hash of (seed, position, v): `h = splitmix64(key + v * φ)` with
`key = splitmix64(seed ^ splitmix64(position))`, `u = ((h >> 40) + 0.5) / 2^24` in fp32 (clamped below 1), then
`-log(-log(u))`. `position` is the index the sampled token will occupy: the row's last query position + 1, each row of
a verify window and each draft at its own. The noise therefore depends only on the seed, the position and the token,
never on the batch: a request recomputed after a preemption draws the same noise. The hash is exact 64-bit integer
arithmetic (two's-complement wrapping), so CPU and CUDA compute the same u; their fp32 logs may round differently.

Per row: a temperature below 1e-5 (vLLM's threshold; a request's is 0 or at least 0.01, `api.engine_client`)
is greedy (`argmax`, lowest id on ties).
Otherwise `x = logits / temperature` (fp32) and the token is the lowest-id argmax of `x + noise` over the tokens that
survive, in vLLM's order:
- **min-p:** `x >= max(x) + log(min_p)`, i.e. `p >= min_p * p_max`;
- **top-k:** the first k tokens in the order (x descending, id ascending);
- **top-p:** over the survivors, renormalised: a token survives if the mass of the tokens before it in that order is
  at most `top_p` (the top token always survives). Masses are `exp(x - max(x))` in fp32, summed in fp64.

That equals sampling from the truncated, renormalised distribution. The details the kernel must reproduce: ties in x
are ordered by id (a stable sort, not `topk`, whose order among ties is unspecified), the top-p sums are not an fp32
softmax and cumsum (whose rounding depends on the reduction order), and ties in `x + noise` go to the lowest id. The
kernel sums the masses in another order (a top-k set over 2,048 tokens in compensated fp32 per thread, fp64 across
threads), so a token can differ only when a top-p sum lands within about 1e-13 of its threshold.

Rows with a non-finite logit are flagged NONFINITE (status bit 0); the check reads every column of the logits given.

**Grammar masks.** A row with a mask samples among the tokens it allows only: the others' logits are -inf for
everything above (the kernel skips them instead, with the same result). The non-finite check reads the logits
before the mask; a finite row with no allowed column flags MASK_EMPTY (status bit 1) with token 0. Masks are over
token ids: over a reduced vocabulary the bit tested for column v is that of its token, `vocab_ids[v]`.

**Padding.** `vocab_ids` may end with -1 entries: the grammar-sparse head scores the union of the step's masks
(ascending ids) padded to a fixed width for the graphs, and the drafter's head the grammar's universe the same way. A
padding column is excluded like a masked token (never sampled, in no filter), but the non-finite check still reads it,
and a row whose columns are all padding is MASK_EMPTY too. Every token a row's mask allows is in the union, so sampling
the union's columns gives the dense path's tokens and statuses exactly (tests/runner/test_sampler.py).

**Penalties and the logit bias** (`runner.cuda.penalties`, `server.penalties`; vLLM's terms and order): a row with
terms samples from `y`, its logits after `apply_terms`, in place of the logits everywhere above (greedy included), and
its `y` and `y / T` are clamped to the finite floats; the non-finite check still reads the raw logits. Rows without
terms are untouched.

A CUDA runner samples through `draw`: the fused sampler on CUDA, this reference on CPU with a `host_plan`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from ...kernels import ops

GREEDY_EPS = 1e-5  # temperatures below it are greedy (vLLM's threshold)
NONFINITE = 1  # StepResult.status bit 0
MASK_EMPTY = 2  # StepResult.status bit 1
_PLAIN_ROWS = 8  # rows per full-vocabulary noise chunk: 8 × 248k × 8 B = 16 MB per temporary
U_MAX = 1.0 - 2.0**-24  # the Gumbel uniform's ceiling (`gumbel`): the largest fp32 below 1
F_MAX = float(np.finfo(np.float32).max)  # a penalised row's logits and x are clamped to [-F_MAX, F_MAX]


def _i64(c: int) -> int:
    return c - (1 << 64) if c >= 1 << 63 else c


PHI = _i64(0x9E3779B97F4A7C15)
_M1 = _i64(0xBF58476D1CE4E5B9)
_M2 = _i64(0x94D049BB133111EB)


def _srl(x: torch.Tensor, s: int) -> torch.Tensor:
    """Logical right shift of int64 (torch's >> is arithmetic)."""
    return (x >> s) & ((1 << (64 - s)) - 1)


def splitmix64(x: torch.Tensor) -> torch.Tensor:
    x = x + PHI
    x = (x ^ _srl(x, 30)) * _M1
    x = (x ^ _srl(x, 27)) * _M2
    return x ^ _srl(x, 31)


def row_keys(seed: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
    return splitmix64(seed ^ splitmix64(position))


def gumbel(keys: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """Gumbel noise [m, C] fp32 for token ids [C] or [m, C] of rows with keys [m]. 24 bits: u = ((h >> 40) + 0.5) /
    2^24 in fp32, clamped to `U_MAX` as in the kernel: the top value rounds to 1 (noise +inf, which would win over any
    logit) and draws 16.64 instead, one token in 2^24 (the next value draws 15.94). The noise is finite, from -2.85
    (u = 2^-25) to 16.64, so a candidate more than 19.49 below the largest candidate x (a relative probability below
    3.5e-9) is never sampled."""
    h = splitmix64(keys[:, None] + ids * PHI)
    u = ((_srl(h, 40).to(torch.float32) + 0.5) * (1.0 / (1 << 24))).clamp_(max=U_MAX)
    return -torch.log(-torch.log(u))


def _noisy(x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """x + noise, with the tokens already excluded (x = -inf: min-p, top-k, top-p, a grammar mask) kept at -inf, as
    the kernel never adds noise to them. The noise is finite (`gumbel`), so -inf + noise is -inf already; the guard
    is defensive: an infinite noise would make NaN, which argmax takes for the largest."""
    return torch.where(x == float("-inf"), x, x + noise)


@dataclass
class _Group:
    kind: str  # "topk", "sort", "plain"
    rows: np.ndarray  # indices into the step's sample rows
    kmax: int = 0
    uniform_k: bool = True
    top_p: bool = False
    min_p: bool = False
    idx: torch.Tensor | None = None  # `rows` as a tensor on the logits' device (`host_plan`)


@dataclass
class SamplePlan:
    n: int
    groups: list[_Group] = field(default_factory=list)


def plan_sampling(temperature: np.ndarray, top_p: np.ndarray, top_k: np.ndarray, min_p: np.ndarray,
                  vocab: int) -> SamplePlan:
    """Host side, for the torch reference: group the non-greedy sample rows by the work they need: `topk` (top_k > 0:
    the first kmax candidates), `sort` (top_k = 0, top_p < 1: the whole row sorted), `plain` (neither: Gumbel-max over
    the whole row, a few rows at a time to bound memory)."""
    n = len(temperature)
    plan = SamplePlan(n)
    rnd = temperature >= GREEDY_EPS
    if not rnd.any():
        return plan
    k = np.minimum(top_k, vocab)
    for kind, sel in (("topk", rnd & (k > 0)), ("sort", rnd & (k <= 0) & (top_p < 1.0)),
                      ("plain", rnd & (k <= 0) & (top_p >= 1.0))):
        rows = np.flatnonzero(sel)
        if not len(rows):
            continue
        g = _Group(kind, rows.astype(np.int64), top_p=bool((top_p[rows] < 1.0).any()),
                   min_p=bool((min_p[rows] > 0).any()))
        if kind == "topk":
            g.kmax = int(k[rows].max())
            g.uniform_k = bool((k[rows] == g.kmax).all())
        plan.groups.append(g)
    return plan


def host_plan(mirror: Any, slots: np.ndarray, width: int) -> SamplePlan:
    """A runner's plan for sample rows on `slots` (`plan_sampling` over the sampling parameters its host mirror keeps
    per slot, each group's rows a host tensor in `idx`), for logits `width` columns wide: the vocabulary, or a reduced
    head's columns (the drafter's, a grammar head's rows)."""
    plan = plan_sampling(mirror.temperature[slots], mirror.top_p[slots], mirror.top_k[slots], mirror.min_p[slots],
                         width)
    for g in plan.groups:
        g.idx = torch.as_tensor(g.rows)
    return plan


def unpack_masks(words: torch.Tensor, vocab: int) -> torch.Tensor:
    """int32 mask words [m, W] -> bool [m, vocab] (bit v % 32 of word v // 32)."""
    bits = (words[:, :, None] >> torch.arange(32, dtype=torch.int32, device=words.device)) & 1
    return bits.reshape(len(words), -1)[:, :vocab].bool()


def apply_terms(logits: torch.Tensor, cols: torch.Tensor, terms: torch.Tensor, rep: torch.Tensor, pres: torch.Tensor,
                freq: torch.Tensor, cnt: torch.Tensor, bias_id: torch.Tensor, bias_val: torch.Tensor,
                bias_n: torch.Tensor, window: torch.Tensor | None = None) -> torch.Tensor:
    """The logits of rows with terms after vLLM's bias and penalties (`runner.cuda.penalties`), in fp32 op by op as
    the kernel computes them, clamped to [-F_MAX, F_MAX]; other rows unchanged (a copy). logits [n, C]; `cols` [C] the
    columns' token ids (-1: padding, whatever its value); per row: `terms`, `rep`, `pres`, `freq`, the slot's `cnt` row
    [n, V], its bias table (`bias_id` ascending, `bias_val`, `bias_n`) and `window` [n, k]: the drafts before the row in
    its verify window (-1: none), each counted once more."""
    out = logits.clone()
    rows = torch.nonzero(terms != 0)[:, 0]
    if not len(rows):
        return out
    from .penalties import BIASED, COUNT, TERM_PEN

    y = logits.index_select(0, rows)
    tok = cols.clamp(min=0)
    e = cnt.index_select(0, rows).index_select(1, tok)  # [r, C]: the columns' words
    b = (e & BIASED) != 0
    if bool(b.any()):
        bid, bval = bias_id.index_select(0, rows).long(), bias_val.index_select(0, rows)
        bn = bias_n.index_select(0, rows)
        big = torch.iinfo(torch.int64).max  # past the entries: never found
        keys = torch.where(torch.arange(bid.shape[1], device=bid.device)[None, :] < bn[:, None], bid, big)
        at = torch.searchsorted(keys, tok[None, :].expand(len(rows), -1).contiguous()).clamp(max=bid.shape[1] - 1)
        y = torch.where(b, y + bval.gather(1, at), y)
    pen = ((terms.index_select(0, rows) & TERM_PEN) != 0)[:, None]
    c = e & COUNT
    if window is not None:
        c = c + (window.index_select(0, rows)[:, :, None] == tok[None, None, :]).sum(1).to(c.dtype)
    seen = (e < 0) | (c > 0)
    r = rep.index_select(0, rows)[:, None]
    scaled = torch.where(y > 0, y * (1.0 / r), y * r)  # 1 / rep rounded once in fp32, as vLLM's 1.0 / scale
    y = torch.where(pen & seen, scaled, y)
    y = torch.where(pen & (c > 0), (y - freq.index_select(0, rows)[:, None] * c.to(y.dtype)) -
                    pres.index_select(0, rows)[:, None], y)
    out.index_copy_(0, rows, y.clamp(-F_MAX, F_MAX))
    return out


def sample(logits: torch.Tensor, plan: SamplePlan, temperature: torch.Tensor, top_p: torch.Tensor,
           top_k: torch.Tensor, min_p: torch.Tensor, keys: torch.Tensor, vocab_ids: torch.Tensor,
           mask: torch.Tensor | None = None, mask_row: torch.Tensor | None = None, adjusted: torch.Tensor | None = None,
           clamp_rows: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """logits [n, V] fp32; per-row parameters [n] on the device; keys [n] int64 (`row_keys`); `vocab_ids` [V] int64,
    the token id of each logit column, ascending (`arange(V)`, or a reduced vocabulary: the MTP drafter's head, whose
    noise and tokens are those of the ids), then possibly -1 entries (padding, never sampled: module docstring).
    `mask` int32 [m, W] and `mask_row` int64 [n]: each row's grammar mask over token ids (-1: none). Returns the tokens
    [n] int64 and the status [n] int64 (NONFINITE for rows with a non-finite logit, MASK_EMPTY for rows with no
    allowed column; their token is 0). `adjusted` (`apply_terms`): the logits sampled from, in place of `logits`
    after the non-finite check; `clamp_rows` [n] bool: the rows whose x is clamped to [-F_MAX, F_MAX] (with terms)."""
    status = (~torch.isfinite(logits).all(-1)).to(torch.int64) * NONFINITE
    if adjusted is not None:
        logits = adjusted
    real = vocab_ids >= 0
    rows, allowed = None, None
    if mask is not None:
        rows = torch.nonzero(mask_row >= 0)[:, 0]
        if len(rows):  # the bits of the columns' tokens, over the masks' full width
            words = mask.index_select(0, mask_row.index_select(0, rows))
            allowed = unpack_masks(words, words.shape[1] * 32).index_select(1, vocab_ids.clamp(min=0)) & real
    if not bool(real.all()):  # padding: every row keeps to the real columns (a masked row, to its mask's among them)
        full = real.expand(len(logits), -1).clone()
        if allowed is not None:
            full[rows] = allowed
        rows, allowed = torch.arange(len(logits), device=logits.device), full
    if allowed is not None:
        logits = logits.clone()
        logits[rows] = logits[rows].masked_fill(~allowed, float("-inf"))
        st = status.index_select(0, rows)
        status.index_copy_(0, rows, torch.where((st == 0) & ~allowed.any(-1), MASK_EMPTY, st))
    out = logits.argmax(dim=-1)
    for g in plan.groups:
        idx = g.idx
        x = logits.index_select(0, idx) / temperature.index_select(0, idx)[:, None]
        if clamp_rows is not None:
            x = torch.where(clamp_rows.index_select(0, idx)[:, None], x.clamp(-F_MAX, F_MAX), x)
        if g.min_p:
            mp = min_p.index_select(0, idx)
            x = x.masked_fill(x < x.amax(-1, keepdim=True) + torch.log(mp)[:, None], float("-inf"))
        kk = keys.index_select(0, idx)
        if g.kind == "plain":
            toks = torch.empty(len(g.rows), dtype=torch.int64, device=x.device)
            for r0 in range(0, len(g.rows), _PLAIN_ROWS):
                r1 = min(r0 + _PLAIN_ROWS, len(g.rows))
                toks[r0:r1] = _noisy(x[r0:r1], gumbel(kk[r0:r1], vocab_ids)).argmax(-1)
        else:
            vals, ids = x.sort(dim=-1, descending=True, stable=True)  # (x desc, id asc)
            if g.kind == "topk":
                vals, ids = vals[:, :g.kmax], ids[:, :g.kmax]
                if not g.uniform_k:
                    col = torch.arange(g.kmax, device=x.device)
                    vals = vals.masked_fill(col[None, :] >= top_k.index_select(0, idx)[:, None], float("-inf"))
            if g.top_p:
                e = torch.exp(vals - vals[:, :1]).double()  # fp32 masses (the top token is the max), fp64 sums
                drop = e.cumsum(-1) - e > top_p.index_select(0, idx).double()[:, None] * e.sum(-1, keepdim=True)
                vals = vals.masked_fill(drop, float("-inf"))
            score = torch.full_like(x, float("-inf"))
            # back to id order: ties go to the lowest id
            score.scatter_(1, ids, _noisy(vals, gumbel(kk, vocab_ids[ids])))
            toks = score.argmax(-1)
        out.index_copy_(0, idx, toks)
    return vocab_ids[out].masked_fill(status != 0, 0), status


def draw(logits: torch.Tensor, ids: torch.Tensor | None, padded: bool, *, cuda: bool, tables: Any,
         slots: torch.Tensor, qpos: torch.Tensor, out: torch.Tensor, status: torch.Tensor,
         pending: torch.Tensor | None, plan: SamplePlan | None, mask: tuple[torch.Tensor, torch.Tensor] | None,
         vocab_ids: torch.Tensor, terms: bool = False, window: tuple[torch.Tensor, torch.Tensor] | None = None) -> None:
    """A CUDA runner's sampling of one token per row of `logits` at position qpos + 1, with the parameters and seed its
    device `tables` keep for each row's slot: the fused sampler on CUDA (`kernels.ops.sample`), the reference with the
    host `plan` on CPU. `ids`: the columns' token ids (None: `vocab_ids`, every id), padded with −1 when `padded`;
    `mask`: (the mask buffer, each row's mask in it or −1). Writes `out`, `status` (NONFINITE, MASK_EMPTY) and, when
    given, the tokens to `pending[slot]`.

    `terms` (the target's samples): the penalties and logit bias of the tables' `pen` (`runner.cuda.penalties`; None:
    off), and with `pending` (no MTP: the sample is committed) each penalised row's token counts. `window` (MTP): the
    static block's tokens and each row's sample token (`s_idx`), from which a verify row's place in its window comes:
    the drafts before it (`tables.drafts`) count for its penalties."""
    tb = tables
    mbuf, mrow = mask if mask is not None else (None, None)
    pt = tb.pen if terms else None
    if cuda:
        ops.sample(logits, row_slot=slots, qpos=qpos, temperature=tb.temperature, top_p=tb.top_p, top_k=tb.top_k,
                   min_p=tb.min_p, seed=tb.seed, out=out, status=status, pending=pending, ids=ids, mask=mbuf,
                   mask_row=mrow, padded=padded, terms=pt, drafts=tb.drafts if window is not None else None,
                   window=window)
        return
    cols = vocab_ids if ids is None else ids.long()
    adj = flags = None
    if pt is not None:
        flags = pt.terms.index_select(0, slots)
        if bool(flags.any()):
            win = None
            if window is not None:  # a verify row's sample token w (window token w is its input): drafts 0 .. w − 1
                tk = window[0].index_select(0, window[1])
                w = torch.where(tk < 0, -1 - tk, 0)
                d = tb.drafts.index_select(0, slots)
                win = torch.where(torch.arange(d.shape[1], device=d.device)[None, :] < w[:, None], d, -1)
            adj = apply_terms(logits, cols, flags, pt.rep.index_select(0, slots), pt.pres.index_select(0, slots),
                              pt.freq.index_select(0, slots), pt.cnt.index_select(0, slots),
                              pt.bias_id.index_select(0, slots), pt.bias_val.index_select(0, slots),
                              pt.bias_n.index_select(0, slots), win)
        else:
            flags = None
    toks, st = sample(logits, plan, tb.temperature.index_select(0, slots), tb.top_p.index_select(0, slots),
                      tb.top_k.index_select(0, slots), tb.min_p.index_select(0, slots),
                      row_keys(tb.seed.index_select(0, slots), qpos + 1), cols, mask=mbuf, mask_row=mrow,
                      adjusted=adj, clamp_rows=flags != 0 if flags is not None else None)
    out.copy_(toks)
    status.copy_(st)
    if pending is not None:
        pending.index_copy_(0, slots, toks)
        if flags is not None:  # the committed samples of penalised rows count (`runner.cuda.penalties`)
            from .penalties import TERM_PEN

            on = ((flags & TERM_PEN) != 0) & (st == 0)
            pt.cnt.view(-1).index_add_(0, slots * pt.vocab + toks, on.to(torch.int32))
