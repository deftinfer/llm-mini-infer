"""Penalties and the logit bias on the CUDA runners (`server.penalties`; architecture.md, "Sampling"; `sched.plan`).

The sampler applies vLLM's terms where it reads a logit, never writing the logits, so logprobs stay raw: per token t of
a row, in fp32, `y = l + bias[t]`; when t is in the prompt or the output so far, `y = y > 0 ? y * (1 / rep) : y * rep`;
then `y -= freq * count[t]` and `y -= pres` when `count[t] > 0` (output tokens only); then temperature and the
filters (`runner.cuda.sampler`). A row's verify window adds the drafts before it to the counts.

**Device tables** (`PenaltyTables`, per slot): `cnt` int32 [slots, V], one word per token: bit 31 (`PROMPT`) the
token is in the prompt, bit 30 (`BIASED`) it has a bias entry, bits 0-29 (`COUNT`) its output count; `terms`
(`TERM_PEN`, `TERM_BIAS`), `rep`, `pres`, `freq`; the bias table `bias_n`, `bias_id` [slots, MAX_LOGIT_BIAS]
(ascending), `bias_val`. A slot with no term reads none of it and is never updated.

**Filling the counts** (vLLM's sets: the prompt's tokens; every token after the prompt), stream-ordered:
- ADMIT (`stage_admit`, `apply_admit`): the row of a penalised or biased admission is rewritten whole: `PROMPT` on the
  tokens of its cached prefix (the plan's bitset; the scheduler resumes a penalised request within its prompt), zero
  elsewhere, then `BIASED` on its bias ids.
- Explicit rows of a penalised slot (`stage_explicit`, `apply_explicit`), before the step's body: a token at a
  position below the slot's prompt length gets `PROMPT`, a later one (a readmission's recomputed outputs, a forced
  prefix) counts.
- Committed samples, in the body: without MTP the sampler counts the token it writes as pending (`sampler.draw`);
  under MTP `commit_mtp` counts each verify row's accepted drafts and its bonus, and an explicit SAMPLE row's sample.
  Repeats accumulate (`scatter_add_`, `index_add_`): never `t[idx] += 1`. Draft samples never count.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ...sched.plan import ADMIT_PENALTIES, MAX_LOGIT_BIAS, StepPlan

PROMPT = -(1 << 31)  # bit 31 of a `cnt` word: the token is in the prompt
BIASED = 1 << 30  # bit 30: the token has a logit_bias entry
COUNT = (1 << 30) - 1  # bits 0-29: the token's output count
TERM_PEN = 1  # `terms`: a penalty is not neutral
TERM_BIAS = 2  # `terms`: a logit bias


class PenaltyTables:
    """The device tables of the module docstring, for `slots` slots (slot 0: padding, never penalised) and a vocabulary
    of `vocab` tokens."""

    def __init__(self, slots: int, vocab: int, device: torch.device):
        self.vocab, self.words = vocab, -(-vocab // 32)
        self.terms = torch.zeros(slots, dtype=torch.int32, device=device)
        self.rep = torch.ones(slots, dtype=torch.float32, device=device)
        self.pres = torch.zeros(slots, dtype=torch.float32, device=device)
        self.freq = torch.zeros(slots, dtype=torch.float32, device=device)
        self.cnt = torch.zeros((slots, vocab), dtype=torch.int32, device=device)
        self.bias_n = torch.zeros(slots, dtype=torch.int32, device=device)
        self.bias_id = torch.zeros((slots, MAX_LOGIT_BIAS), dtype=torch.int32, device=device)
        self.bias_val = torch.zeros((slots, MAX_LOGIT_BIAS), dtype=torch.float32, device=device)
        self._bit = torch.arange(32, dtype=torch.int32, device=device)

    @staticmethod
    def bytes_for(slots: int, vocab: int) -> int:
        return slots * (vocab * 4 + MAX_LOGIT_BIAS * 8 + 20)

    def reset(self) -> None:
        self.terms.zero_()

    def staging(self, slots: int, max_tokens: int) -> tuple[int, int, int]:
        """What a step stages at most: int64 entries (the admissions' rows and bias indices, the explicit tokens'
        indices), fp32 entries (the three penalties, the bias values) and int32 words (bitsets, bias ids)."""
        nb = slots * MAX_LOGIT_BIAS
        return 2 * slots + 2 * nb + max_tokens, 3 * slots + nb, slots * self.words + nb


@dataclass
class AdmitTerms:
    """One plan's staged ADMIT terms (offsets into the staging buffer)."""

    n: int  # admissions
    o_slot: int  # int64: their slots
    o_terms: int  # int64: their `terms`
    o_nb: int  # int64: their bias counts
    o_f: int  # fp32: rep, pres, freq [3, n]
    n_init: int = 0  # admissions whose `cnt` row is rewritten (penalised or biased)
    o_rows: int = 0  # int64: their slots
    n_bits: int = 0  # of those, the penalised ones (a bitset each), first
    o_bits: int = 0  # int32: their bitsets [n_bits, words]
    n_e: int = 0  # bias entries
    o_tab: int = 0  # int64: slot * MAX_LOGIT_BIAS + j
    o_cnt: int = 0  # int64: slot * V + id
    o_id: int = 0  # int32: ids
    o_val: int = 0  # fp32: values


def stage_admit(plan: StepPlan, mirror: object, st: object, adm: np.ndarray, vocab: int) -> AdmitTerms:
    """Stage a plan's ADMIT terms and record each admission's `terms` and prompt length in the host mirror (`terms`,
    `prompt_len`)."""
    n = len(adm)
    pen = (plan.admit_flags & ADMIT_PENALTIES) != 0
    nb = plan.admit_bias_n.astype(np.int64) if len(plan.admit_bias_n) else np.zeros(n, np.int64)
    terms = np.where(pen, TERM_PEN, 0) | np.where(nb > 0, TERM_BIAS, 0)
    mirror.terms[adm] = terms
    mirror.prompt_len[adm] = plan.admit_prompt_len if len(plan.admit_prompt_len) else 0
    neutral = np.zeros(n, np.float32)
    f = np.concatenate([plan.admit_rep, plan.admit_pres, plan.admit_freq]) if len(plan.admit_rep) else \
        np.concatenate([neutral + 1, neutral, neutral])
    out = AdmitTerms(n, st.put(adm), st.put(terms.astype(np.int64)), st.put(nb), st.putf(f))
    init = np.flatnonzero(terms != 0)
    if not len(init):
        return out
    init = np.concatenate([init[pen[init]], init[~pen[init]]])  # the penalised ones first (their bitsets)
    rows = adm[init]
    out.n_init, out.o_rows = len(rows), st.put(rows)
    out.n_bits = int(pen[init].sum())
    if out.n_bits:
        out.o_bits = st.putb(np.ascontiguousarray(plan.admit_prompt_bits[init[:out.n_bits]]).view(np.int32)
                             .reshape(-1))
    biased = np.flatnonzero(nb)
    if len(biased):
        cnt = nb[biased]
        slots = np.repeat(adm[biased], cnt)
        j = np.arange(int(cnt.sum())) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        ids = np.concatenate([plan.admit_bias_ids[i, :c] for i, c in zip(biased, cnt)]).astype(np.int64)
        vals = np.concatenate([plan.admit_bias_val[i, :c] for i, c in zip(biased, cnt)])
        out.n_e = len(ids)
        out.o_tab, out.o_cnt = st.put(slots * MAX_LOGIT_BIAS + j), st.put(slots * vocab + ids)
        out.o_id, out.o_val = st.putb(ids.astype(np.int32)), st.putf(vals)
    return out


def apply_admit(pt: PenaltyTables, a: AdmitTerms, st: object) -> None:
    """The device writes of `stage_admit` (stream order: after FREE, before the step's explicit rows)."""
    n = a.n
    sl = st.i(a.o_slot, n)
    pt.terms.index_copy_(0, sl, st.i(a.o_terms, n).to(torch.int32))
    pt.bias_n.index_copy_(0, sl, st.i(a.o_nb, n).to(torch.int32))
    f = st.f(a.o_f, 3 * n)
    pt.rep.index_copy_(0, sl, f[:n])
    pt.pres.index_copy_(0, sl, f[n:2 * n])
    pt.freq.index_copy_(0, sl, f[2 * n:])
    if not a.n_init:
        return
    slots = st.i(a.o_rows, a.n_init)
    if a.n_bits:  # a few rows at a time: two int32 vocabulary rows of temporaries each
        words = st.b(a.o_bits, a.n_bits * pt.words).view(a.n_bits, pt.words)
        for r0 in range(0, a.n_bits, 8):
            r1 = min(a.n_bits, r0 + 8)
            bits = ((words[r0:r1, :, None] >> pt._bit) & 1).reshape(r1 - r0, -1)[:, :pt.vocab]
            pt.cnt.index_copy_(0, slots[r0:r1], bits * PROMPT)
    if a.n_init > a.n_bits:
        pt.cnt.index_fill_(0, slots[a.n_bits:], 0)
    if a.n_e:
        tab = st.i(a.o_tab, a.n_e)
        pt.bias_id.view(-1).index_copy_(0, tab, st.b(a.o_id, a.n_e))
        pt.bias_val.view(-1).index_copy_(0, tab, st.f(a.o_val, a.n_e))
        flat, idx = pt.cnt.view(-1), st.i(a.o_cnt, a.n_e)  # distinct ids: one write each
        flat.index_copy_(0, idx, flat.index_select(0, idx) | BIASED)


def stage_explicit(plan: StepPlan, mirror: object, st: object, vocab: int) -> tuple[int, int, int, int] | None:
    """The explicit rows' tokens of penalised slots (after this plan's ADMITs reached the mirror): (offset, n) of the
    prompt tokens' and of the output tokens' `cnt` indices (slot * V + token), or None when there are none."""
    terms = mirror.terms
    if not terms.any():
        return None
    slots, off = plan.sched_slot, plan.sched_tok_off
    rows = np.flatnonzero((off >= 0) & ((terms[slots] & TERM_PEN) != 0))
    if not len(rows):
        return None
    prompt, output = [], []
    for r in rows.tolist():
        s, o, nq, p0 = int(slots[r]), int(off[r]), int(plan.sched_n_query[r]), int(plan.sched_pos0[r])
        idx = s * vocab + plan.tokens[o:o + nq].astype(np.int64)
        cut = min(max(int(mirror.prompt_len[s]) - p0, 0), nq)
        prompt.append(idx[:cut])
        output.append(idx[cut:])
    p, q = np.concatenate(prompt), np.concatenate(output)
    return st.put(p), len(p), st.put(q), len(q)


def apply_explicit(pt: PenaltyTables, e: tuple[int, int, int, int], st: object) -> None:
    """The device writes of `stage_explicit`: `PROMPT` on the prompt tokens (gathered, then written: a repeated token
    writes the same word), then the output tokens' counts (accumulated)."""
    flat = pt.cnt.view(-1)
    o_p, n_p, o_q, n_q = e
    if n_p:
        idx = st.i(o_p, n_p)
        flat.index_copy_(0, idx, flat.index_select(0, idx) | PROMPT)
    if n_q:
        flat.index_add_(0, st.i(o_q, n_q), torch.ones(n_q, dtype=torch.int32, device=flat.device))


def commit_mtp(pt: PenaltyTables, r_slot: torch.Tensor, yw: torch.Tensor, n_out: torch.Tensor,
               ar: torch.Tensor) -> None:
    """Under MTP, after the verification (`runner.mtp_verify`; graph-safe): each row's committed samples, the first
    `n_out` of its window's (the accepted drafts' samples and the bonus; an explicit SAMPLE row's one sample), count
    for its slot when it is penalised. Other entries add 0."""
    on = (pt.terms.index_select(0, r_slot) & TERM_PEN) != 0
    take = (ar[None, :] < n_out[:, None]) & on[:, None]
    idx = r_slot[:, None] * pt.vocab + yw
    pt.cnt.view(-1).scatter_add_(0, idx.reshape(-1), take.reshape(-1).to(torch.int32))
