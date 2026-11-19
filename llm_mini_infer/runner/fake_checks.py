"""The fake runner's oracle: the checks of a plan against the slot tables the fake keeps, as a mixin `FakeRunner`
inherits (`runner/fake.py` describes what they verify). `context_hash` is the fingerprint hash the checks and the
fake's tokens share."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from ..sched.plan import CKPT_END_OF_STEP, LOGPROBS, MASKED, MROPE, PRECISE, SAMPLE, STATE_ZERO, StepPlan

MASK64 = (1 << 64) - 1
HASH_BASE = 0x100000001B3  # odd: the rolling hash of a prefix (`context_hash`)


def context_hash(tokens: Iterable[int], h: int = 0) -> int:
    """The fingerprint hash of a token prefix, extended from `h` (the hash of what precedes `tokens`)."""
    for t in tokens:
        h = (h * HASH_BASE + int(t) + 1) & MASK64
    return h


def _first_diff(a: list, b: list) -> int:
    """The first index where two lists differ (they must differ within the shorter one)."""
    return next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)


class FakeChecks:
    """The plan checks of `FakeRunner`; they report through its `_fail`."""

    def _check_ckpts(self, plan: StepPlan) -> None:
        if not len(plan.ckpt_sched):
            return
        n_states, n = self.pools.num_state_slots, len(plan.sched_slot)
        rows, offs, dsts = plan.ckpt_sched.tolist(), plan.ckpt_offset.tolist(), plan.ckpt_dst.tolist()
        if len(set(dsts)) != len(dsts):
            self._fail(plan, "two CKPTs into one state slot")
        for r, off, d in zip(rows, offs, dsts):
            if not 0 <= r < n:
                self._fail(plan, f"CKPT for SCHED row {r} of {n}")
            nq = int(plan.sched_n_query[r])
            if not (off in (CKPT_END_OF_STEP, nq) or (0 < off < nq and off % self.grid == 0)):
                self._fail(plan, f"CKPT offset {off} for a row of {nq} tokens (grid {self.grid})")
            if not 1 <= d < n_states:
                self._fail(plan, f"CKPT into state slot {d}, out of range")
            if self.state_owner[d]:
                self._fail(plan, f"CKPT into state slot {d}, the live state of slot {self.state_owner[d]}")

    def _check_sched(self, plan: StepPlan) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Validate SCHED (in Python over the rows' values: a plan has a handful of rows, where NumPy's cost per call
        dominates; `full_checks`: the reference implementation, `_check_sched_full`). Returns per row: slot, first
        position, n_query, SAMPLE."""
        if self.full_checks:
            return self._check_sched_full(plan)
        slots = plan.sched_slot.astype(np.int64)
        n = len(slots)
        if n > self.pools.max_num_seqs:
            self._fail(plan, f"{n} SCHED rows > max_num_seqs")
        if plan.n_query_tokens > self.pools.max_num_batched_tokens:
            self._fail(plan, f"{plan.n_query_tokens} query tokens > max_num_batched_tokens")
        if n == 0:
            if len(plan.ckpt_sched):
                self._fail(plan, "CKPT without SCHED rows")
            e = np.zeros(0, np.int64)
            return e, e, e, np.zeros(0, bool)
        sl = slots.tolist()
        if len(set(sl)) != n:
            self._fail(plan, "a slot appears twice in SCHED")
        if min(sl) < 1 or max(sl) > self.pools.max_num_seqs:  # the range first: an index of -1 wraps
            self._fail(plan, f"SCHED row for a slot that is not admitted: {sl}")
        if not self.admitted[slots].all():
            self._fail(plan, f"SCHED row for a slot that is not admitted: {slots[~self.admitted[slots]].tolist()}")
        k = self.k
        nq = plan.sched_n_query.astype(np.int64)
        flags = plan.sched_flags
        seq = self.seq_len[slots]
        dev = plan.sched_tok_off < 0
        ps = self.ps
        live = self.state_slot[slots]
        fl, nql, seql, p0l, devl = flags.tolist(), nq.tolist(), seq.tolist(), plan.sched_pos0.tolist(), dev.tolist()
        if plan.sched_n_draft.tolist() != [k if d else 0 for d in devl]:
            self._fail(plan, f"n_draft must be {k} on device-input rows and 0 on explicit ones")
        if k and any((t >= 0) != (not d and not f & SAMPLE) for t, d, f in zip(plan.sched_next_tok.tolist(), devl, fl)):
            self._fail(plan, "sched_next_tok must be set exactly on non-final explicit rows")
        if any(f & MASKED for f in fl):
            masked = (flags & MASKED) != 0
            if (masked & ((flags & SAMPLE) == 0)).any():
                self._fail(plan, "MASKED on a row without SAMPLE")
            if not self.constrained[slots[masked]].all():
                self._fail(plan, "MASKED row of an admission without ADMIT_CONSTRAINED")
        if any(f & LOGPROBS for f in fl):
            lpf = (flags & LOGPROBS) != 0
            if (lpf & ((flags & SAMPLE) == 0)).any():
                self._fail(plan, "LOGPROBS on a row without SAMPLE")
            if not self.lp_slot[slots[lpf]].all():
                self._fail(plan, "LOGPROBS row of an admission without ADMIT_LOGPROBS")
        if any(f & PRECISE for f in fl):
            if not self.precise_rows:
                self._fail(plan, "PRECISE rows, but this runner takes none")
            if (((flags & PRECISE) != 0) & dev).any():
                self._fail(plan, "PRECISE on a device-input row")
        di = [i for i in range(n) if devl[i]]
        for i in di:
            self.resumed[sl[i]] = False
        if di:
            if any(p0l[i] != -1 or nql[i] != 1 + k for i in di):
                self._fail(plan, f"device-input rows need pos0 = -1 and n_query = {1 + k}")
            if not all(self.sampled[sl[i]] for i in di):
                ds = slots[dev]
                self._fail(plan, f"device input before a SAMPLE step of the admission: {ds[~self.sampled[ds]]}")
            if any(fl[i] & STATE_ZERO for i in di):
                self._fail(plan, "STATE_ZERO on a device-input row")
            # The positions the row writes: the verify window and, under MTP, the drafter's chain after it (a drafter
            # without KV: the window alone); their pages, one slice of the row's page table
            span = 1 if not k else 2 * k if self.draft_kv else 1 + k
            if any((seql[i] + span - 1) // ps >= self.max_pages_per_seq for i in di):
                self._fail(plan, "device-input positions beyond the page table")
            wpages = [self.rows[sl[i], seql[i] // ps:(seql[i] + span - 1) // ps + 1] for i in di]
            if not all(p.all() for p in wpages):
                self._fail(plan, "device-input positions not covered by pages")
            if self.window:
                for i in di:
                    self._check_window_pages(plan, sl[i], seql[i], seql[i] + span)
            if k and any(self.page_users[p].max() > 1 for p in wpages):
                self._fail(plan, "a decode row's KV writes reach a page another slot shares")
            if any(self.fp_len[live[i]] + self.fold_n[sl[i]] != seql[i] for i in di):
                self._fail(plan, "a decoding slot's live state does not hold its sequence")
        pos0 = np.where(dev, seq, plan.sched_pos0)
        if any((seql[i] + 1 if devl[i] else p0l[i] + nql[i]) > self.pools.max_model_len for i in range(n)):
            self._fail(plan, "positions beyond max_model_len")
        spans = self._check_spans(plan, dev, pos0, nq)
        for i in range(n):
            if not devl[i]:
                self._check_explicit(plan, i, sl[i], nql[i], int(plan.sched_tok_off[i]), p0l[i], seql[i], fl[i],
                                     int(live[i]), spans.get(i, []))
        self._check_ckpts(plan)
        return slots, pos0, nq, (flags & SAMPLE) != 0

    def _check_sched_full(self, plan: StepPlan) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """`_check_sched`'s reference implementation (`full_checks`): the checks across rows vectorized."""
        slots = plan.sched_slot.astype(np.int64)
        n = len(slots)
        if n > self.pools.max_num_seqs:
            self._fail(plan, f"{n} SCHED rows > max_num_seqs")
        if plan.n_query_tokens > self.pools.max_num_batched_tokens:
            self._fail(plan, f"{plan.n_query_tokens} query tokens > max_num_batched_tokens")
        if n == 0:
            if len(plan.ckpt_sched):
                self._fail(plan, "CKPT without SCHED rows")
            e = np.zeros(0, np.int64)
            return e, e, e, np.zeros(0, bool)
        if len(set(slots.tolist())) != n:
            self._fail(plan, "a slot appears twice in SCHED")
        if (slots < 1).any() or (slots > self.pools.max_num_seqs).any():
            self._fail(plan, f"SCHED row for a slot that is not admitted: {slots.tolist()}")
        if not self.admitted[slots].all():
            self._fail(plan, f"SCHED row for a slot that is not admitted: {slots[~self.admitted[slots]].tolist()}")
        k = self.k
        nq = plan.sched_n_query.astype(np.int64)
        flags = plan.sched_flags
        seq = self.seq_len[slots]
        dev = plan.sched_tok_off < 0
        ps = self.ps
        live = self.state_slot[slots]
        if (plan.sched_n_draft != np.where(dev, k, 0)).any():
            self._fail(plan, f"n_draft must be {k} on device-input rows and 0 on explicit ones")
        if k and ((plan.sched_next_tok >= 0) != (~dev & ((flags & SAMPLE) == 0))).any():
            self._fail(plan, "sched_next_tok must be set exactly on non-final explicit rows")
        masked = (flags & MASKED) != 0
        if masked.any():
            if (masked & ((flags & SAMPLE) == 0)).any():
                self._fail(plan, "MASKED on a row without SAMPLE")
            if not self.constrained[slots[masked]].all():
                self._fail(plan, "MASKED row of an admission without ADMIT_CONSTRAINED")
        lpf = (flags & LOGPROBS) != 0
        if lpf.any():
            if (lpf & ((flags & SAMPLE) == 0)).any():
                self._fail(plan, "LOGPROBS on a row without SAMPLE")
            if not self.lp_slot[slots[lpf]].all():
                self._fail(plan, "LOGPROBS row of an admission without ADMIT_LOGPROBS")
        precise = (flags & PRECISE) != 0
        if precise.any():
            if not self.precise_rows:
                self._fail(plan, "PRECISE rows, but this runner takes none")
            if (precise & dev).any():
                self._fail(plan, "PRECISE on a device-input row")
        self.resumed[slots[dev]] = False
        if dev.any():
            ds = slots[dev]
            if (plan.sched_pos0[dev] != -1).any() or (nq[dev] != 1 + k).any():
                self._fail(plan, f"device-input rows need pos0 = -1 and n_query = {1 + k}")
            if not self.sampled[ds].all():
                self._fail(plan, f"device input before a SAMPLE step of the admission: {ds[~self.sampled[ds]]}")
            if (flags[dev] & STATE_ZERO).any():
                self._fail(plan, "STATE_ZERO on a device-input row")
            span = 1 if not k else 2 * k if self.draft_kv else 1 + k
            wpos = seq[dev][:, None] + np.arange(span)[None, :]
            if (wpos // ps >= self.max_pages_per_seq).any():
                self._fail(plan, "device-input positions beyond the page table")
            wpages = self.rows[ds[:, None], wpos // ps]
            if (wpages == 0).any():
                self._fail(plan, "device-input positions not covered by pages")
            if self.window:
                for slot, q in zip(ds.tolist(), seq[dev].tolist()):
                    self._check_window_pages(plan, slot, q, q + span)
            if k and (self.page_users[wpages] > 1).any():
                self._fail(plan, "a decode row's KV writes reach a page another slot shares")
            if (self.fp_len[live[dev]] + self.fold_n[ds] != seq[dev]).any():
                self._fail(plan, "a decoding slot's live state does not hold its sequence")
        pos0 = np.where(dev, seq, plan.sched_pos0)
        if (np.where(dev, pos0 + 1, pos0 + nq) > self.pools.max_model_len).any():
            self._fail(plan, "positions beyond max_model_len")
        spans = self._check_spans(plan, dev, pos0, nq)
        for i in np.flatnonzero(~dev).tolist():
            self._check_explicit(plan, i, int(slots[i]), int(nq[i]), int(plan.sched_tok_off[i]),
                                 int(plan.sched_pos0[i]), int(seq[i]), int(flags[i]), int(live[i]), spans.get(i, []))
        self._check_ckpts(plan)
        return slots, pos0, nq, (flags & SAMPLE) != 0

    def _check_spans(self, plan: StepPlan, dev: np.ndarray, pos0: np.ndarray,
                     nq: np.ndarray) -> dict[int, list[tuple[int, int]]]:
        """SPANS (bidirectional image spans), by SCHED row; each inside its explicit row."""
        spans: dict[int, list[tuple[int, int]]] = {}
        if len(plan.span_sched):
            if not self.bidi:
                self._fail(plan, "SPANS, but this model's image spans are causal")
            for r, a, b_ in zip(plan.span_sched.tolist(), plan.span_start.tolist(), plan.span_end.tolist()):
                if not 0 <= r < len(dev) or dev[r] or not pos0[r] <= a < b_ <= pos0[r] + nq[r]:
                    self._fail(plan, f"SPANS ({r}, {a}, {b_}) outside its explicit row")
                spans.setdefault(r, []).append((a, b_))
        return spans

    def _check_explicit(self, plan: StepPlan, i: int, slot: int, q: int, off: int, p0: int, seq: int, flags: int,
                        ss: int, spans: list[tuple[int, int]]) -> None:
        """Explicit SCHED row i (`_check_sched`): its inputs, pages, drafter chain, window, MTP look-ahead, images,
        batch-invariant cut, and the state it resumes from (live state slot `ss`)."""
        ps = self.ps
        if p0 != seq:
            self._fail(plan, f"slot {slot}: pos0 {p0} != device seq_len {seq}")
        if q < 1 or off + q > len(plan.tokens):
            self._fail(plan, f"slot {slot}: explicit tokens [{off}, {off + q}) out of range")
        if bool(flags & STATE_ZERO) != (p0 == 0):
            self._fail(plan, f"slot {slot}: STATE_ZERO must be set exactly on the chunk at position 0")
        inputs = plan.tokens[off:off + q]
        if int(inputs.min()) < 0 or int(inputs.max()) >= self.vocab:
            self._fail(plan, f"slot {slot}: token out of vocabulary")
        if not self.rows[slot, p0 // ps:(p0 + q - 1) // ps + 1].all():
            self._fail(plan, f"slot {slot}: positions [{p0}, {p0 + q}) not covered by pages")
        # A SAMPLE row's drafter chain (MTP, a drafter with KV) writes k − 1 positions past the row, and reads them
        chain = self.k - 1 if self.k and self.draft_kv and flags & SAMPLE else 0
        if chain:
            cp = self.rows[slot, (p0 + q) // ps:(p0 + q + chain - 1) // ps + 1]
            if not cp.all():
                self._fail(plan, f"slot {slot}: the drafter's chain at [{p0 + q}, {p0 + q + chain}) not covered "
                                 "by pages")
            if (self.page_users[cp] > 1).any():
                self._fail(plan, f"slot {slot}: the drafter's chain at [{p0 + q}, {p0 + q + chain}) writes a page "
                                 "another slot shares")
        if self.window:
            self._check_window_pages(plan, slot, p0, p0 + q + chain)
        if self.k:
            if self.fold_n[slot]:
                self._fail(plan, f"slot {slot}: explicit tokens while its live state has pending window records")
            if self.draft_kv and self.resumed[slot] and self.page_users[self.rows[slot, (p0 - 1) // ps]] > 1:
                self._fail(plan, f"slot {slot}: the drafter recomputes its KV at {p0 - 1}, in page "
                                 f"{int(self.rows[slot, (p0 - 1) // ps])}, which another slot shares")
            if self.next_tok[slot] >= 0 and int(inputs[0]) != self.next_tok[slot]:
                self._fail(plan, f"slot {slot}: the previous chunk's next token {int(self.next_tok[slot])} is not "
                                 f"this chunk's first {int(inputs[0])}")
            nt = int(plan.sched_next_tok[i])
            if nt >= self.vocab:
                self._fail(plan, f"slot {slot}: next token {nt} out of vocabulary")
            self.next_tok[slot] = nt
        if self.encoder_rows:
            self._check_images(plan, i, slot, q, off, p0, bool(flags & SAMPLE), spans)
        elif len(plan.tok_emb) or len(plan.tok_rope[0]) or len(plan.enc_key):
            self._fail(plan, "image inputs, but this runner has no image encoder")
        if self.aligned:
            self._check_cut(plan, slot, p0 + q, flags)
        self.resumed[slot] = False
        if p0 and self.has_state and self.fp_len[ss] != p0:
            self._fail(plan, f"slot {slot} resumes at position {p0} but its state slot {ss} holds a prefix of "
                             f"{int(self.fp_len[ss])} tokens")

    def _check_cut(self, plan: StepPlan, slot: int, end: int, flags: int) -> None:
        """Batch invariance (`aligned`): an explicit row that does not sample ends a whole number of grids past its
        slot's anchor, or at one of its request's exact cuts, which moves the anchor; a PRECISE row may end anywhere
        and moves it too (a refine's pieces and end)."""
        if flags & SAMPLE:
            return
        off = (end - int(self.anchor[slot])) % self.grid
        if off and not flags & PRECISE and end not in self.exact_cuts.get(int(self.seed[slot]), ()):
            self._fail(plan, f"slot {slot}: a prefill cut at {end}, off the {self.grid}-token grid from "
                             f"{int(self.anchor[slot])} (batch invariance)")
        if flags & PRECISE or off:
            self.anchor[slot] = end

    def _check_images(self, plan: StepPlan, i: int, slot: int, q: int, off: int, p0: int, final: bool,
                      spans: list[tuple[int, int]] = ()) -> None:
        """An explicit row's image inputs (module docstring): spans read in token order, whole, from written rows;
        the MTP look-ahead row; the M-RoPE deltas of a final chunk; bidirectional spans whole in the row, as SPANS
        names them."""
        emb = plan.tok_emb[off:off + q] if len(plan.tok_emb) else None
        if self.bidi:
            runs: list[tuple[int, int]] = []
            if emb is not None:
                img = emb >= 0
                idx = np.where(img, self.enc_idx[np.maximum(emb, 0)], -1)
                for t in np.flatnonzero(img).tolist():
                    if not runs or runs[-1][1] != t or idx[t] == 0:
                        runs.append((t, t + 1))
                    else:
                        runs[-1] = (runs[-1][0], t + 1)
                for a, b_ in runs:
                    key = int(self.enc_key[int(emb[a])])
                    if idx[a] != 0 or idx[b_ - 1] != self.key_n.get(key, -1) - 1:
                        self._fail(plan, f"slot {slot}: the image tokens at [{p0 + a}, {p0 + b_}) are image tokens "
                                         f"{int(idx[a])}..{int(idx[b_ - 1])} of {self.key_n.get(key)}: a bidirectional "
                                         "span split across rows, or resumed inside")
            if [(p0 + a, p0 + b_) for a, b_ in runs] != sorted(spans):
                self._fail(plan, f"slot {slot}: SPANS {sorted(spans)} != the row's image spans "
                                 f"{[(p0 + a, p0 + b_) for a, b_ in runs]}")
        pk, pi = (int(x) for x in self.last_img[slot])
        if pk == -2 and p0 and self.strict:  # a resume: the token before pos0 is in the (shared) KV
            pg, o = self.rows[slot, (p0 - 1) // self.ps], (p0 - 1) % self.ps
            pi = int(self.kv_img[pg, o])
            pk = int(self.kv[pg, o]) if pi >= 0 else -1
        first = int(emb[0]) if emb is not None else -1
        got = (int(self.enc_key[first]), int(self.enc_idx[first])) if first >= 0 else (-1, -1)
        if self.k and self.next_img[slot, 0] >= 0 and got != tuple(self.next_img[slot].tolist()):
            self._fail(plan, f"slot {slot}: the previous chunk's look-ahead read (image key, token) "
                             f"{tuple(self.next_img[slot].tolist())}, this chunk's first row holds {got}")
        if emb is not None or pi >= 0:
            for t in range(q):
                r = int(emb[t]) if emb is not None else -1
                if r >= 0:
                    k, x = int(self.enc_key[r]), int(self.enc_idx[r])
                    if pk == k and pi >= 0:  # the same span, or the same image again right after it
                        ok = x == pi + 1 or (x == 0 and pi == self.key_n.get(k, -1) - 1)
                    elif pk == -2:
                        ok = True  # resumed without strict KV: unknown
                    else:
                        ok = x == 0 and (pi < 0 or pi == self.key_n.get(pk, -1) - 1)
                    if not ok:
                        self._fail(plan, f"slot {slot}: position {p0 + t} reads image token {x} after "
                                         f"{'text' if pi < 0 else f'image token {pi}'}")
                    pk, pi = k, x
                else:
                    if pi >= 0 and pi != self.key_n.get(pk, -1) - 1:
                        self._fail(plan, f"slot {slot}: text at position {p0 + t} inside an image span (after its "
                                         f"token {pi})")
                    pk, pi = -1, -1
                    if emb is None:
                        break  # the rest of the row is text
            if emb is None:
                pk, pi = -1, -1
        self.last_img[slot] = (pk, pi)
        r = int(plan.sched_next_emb[i]) if len(plan.sched_next_emb) else -1
        if r >= self.encoder_rows:
            self._fail(plan, f"slot {slot}: look-ahead row {r} outside the encoder pool")
        self.next_img[slot] = (self.enc_key[r], self.enc_idx[r]) if r >= 0 else (-1, -1)
        rope = plan.tok_rope[:, off:off + q] if len(plan.tok_rope[0]) else None
        if plan.sched_flags[i] & MROPE and rope is None:
            self._fail(plan, f"slot {slot}: an MROPE row without M-RoPE deltas")
        if final and self.rope_off[slot] and (rope is None or (rope[:, -1] != self.rope_off[slot]).any()):
            self._fail(plan, f"slot {slot}: the final chunk's last delta "
                             f"{None if rope is None else rope[:, -1].tolist()} is not the ADMIT rope offset "
                             f"{int(self.rope_off[slot])}")

    def _check_window_pages(self, plan: StepPlan, slot: int, q: int, end: int) -> None:
        """A row whose first query is at q and that writes up to `end` reads the window pages from q − W + 1."""
        lo, wps = max(0, q - self.window + 1), self.wps
        if end // wps >= self.max_wpages_per_seq or not self.wrows[slot, lo // wps:(end - 1) // wps + 1].all():
            self._fail(plan, f"slot {slot}: window positions [{lo}, {end}) not covered by window pages")

    def _become(self, pages: np.ndarray, kind: int) -> bool:
        """Strict mode, one arena: `pages` (int array) are written as `kind` (1 full, 2 window); a page that held the
        other kind loses it, as on the GPU, where both kinds of a page id are the same bytes (its positions read as
        never written). Returns whether a page changed kind."""
        if not self.window:
            return False
        was = self.kind[pages]
        if all(w == kind for w in was.tolist()):
            return False
        flip = pages[was != kind]
        if kind == 1:
            self.wkv[flip], self.wkv_bad[flip], self.wkv_prec[flip] = -1, False, False
        else:
            self.kv[flip], self.kv_bad[flip], self.kv_img[flip], self.kv_prec[flip] = -1, False, -1, False
        self.counts["kind_switch"] += int(np.count_nonzero(self.kind[flip]))
        self.kind[flip] = kind
        return True

    def _kv_history(self, plan: StepPlan, slot: int, pos0: int, inputs: np.ndarray | None,
                    img: np.ndarray | None) -> tuple[list[int], list[int]]:
        """Strict mode: the tokens (and image token indices) of `slot`'s context, extended by a row's inputs at pos0
        (`_check_kv`). A resumed slot takes its cached prefix from its pages, which must match its state's
        fingerprint."""
        ps = self.ps
        hist = self.history[slot]
        imgs = self.img_hist[slot]
        if len(hist) < pos0:  # resumed from the prefix cache: the cached positions come from the shared pages
            pos = np.arange(len(hist), pos0)
            got = self.kv[self.rows[slot, pos // ps], pos % ps]
            if (got < 0).any():
                self._fail(plan, f"slot {slot}: cached position {int(pos[got < 0][0])} was never written")
            hist.extend(got.tolist())
            imgs.extend(self.kv_img[self.rows[slot, pos // ps], pos % ps].tolist())
            ss = int(self.state_slot[slot])
            if not self.has_state:  # no recurrent state: the context is the KV alone
                self.fp_len[ss], self.fp_hash[ss], self.fp_bad[ss] = pos0, context_hash(hist), False
            elif context_hash(hist) != int(self.fp_hash[ss]):
                self._fail(plan, f"slot {slot}: its state slot {ss} holds another prefix than its KV pages")
        if inputs is None:
            hist.append(int(self.pending[slot]))
            imgs.append(-1)
        else:
            hist.extend(inputs.tolist())
            imgs.extend(img.tolist() if img is not None else [-1] * len(inputs))
        del imgs[len(hist):]
        return hist, imgs

    def _check_kv(self, plan: StepPlan, slot: int, pos0: int, nq: int, inputs: np.ndarray | None,
                  garbage: int = 0, img: np.ndarray | None = None, precise: bool = False) -> bool:
        """Strict mode: write this row's input tokens to the fake KV and check the whole context still holds this
        request's tokens (`_kv_history`: a resumed slot's cached prefix). Returns whether the row read poisoned KV. An
        MTP decode row passes its committed tokens and the number of rejected positions after them (`garbage`): their
        KV holds tokens of no sequence until rewritten. `img`: the inputs' image token indices (−1: text), kept next to
        the KV. `precise`: a PRECISE row, which must read only KV that PRECISE rows wrote (the precise prefix), and
        whose writes are marked so. `full_checks` runs the reference implementation instead (`_check_kv_full`), with
        the same results and errors."""
        if self.full_checks:
            return self._check_kv_full(plan, slot, pos0, nq, inputs, garbage, img, precise)
        ps, end, f = self.ps, pos0 + nq - garbage, self.flat
        hist, imgs = self._kv_history(plan, slot, pos0, inputs, img)
        pages = self.rows[slot][self._pdiv[:end + garbage]]
        at = pages * ps + self._pmod[:end + garbage]  # the flat index of every position the row reads or writes
        wpages, wat = pages[pos0:], at[pos0:]
        if max(self.page_users[wpages].tolist()) > 1:
            bad = pos0 + int(np.flatnonzero(self.page_users[wpages] > 1)[0])
            self._fail(plan, f"slot {slot}: KV write at position {bad} into page {int(self.rows[slot, bad // ps])}, "
                             f"which another slot shares")
        read_bad = bool(f["kv_bad"][at[:pos0]].any())
        if precise:
            prec = f["kv_prec"][at[:pos0]]
            if not prec.all():
                self._fail(plan, f"slot {slot}: a PRECISE row reads KV at position {int(prec.argmin())} that no "
                                 "PRECISE row wrote")
        self._become(wpages, 1)
        f["kv"][wat] = hist[pos0:end] + [-2] * garbage
        f["kv_img"][wat] = imgs[pos0:end] + [-1] * garbage
        f["kv_bad"][wat] = False
        f["kv_prec"][wat] = precise
        got = f["kv"][at[:end]].tolist()
        if got != hist[:end]:
            self._fail(plan, f"slot {slot}: KV at position {_first_diff(got, hist)} was overwritten by another request")
        if self.window:  # window pages: the positions this row's queries see must hold its own tokens
            wps = self.wps
            lo = max(0, pos0 - self.window + 1)
            n = pos0 - lo  # the positions [lo, pos0) its queries see, then the ones it writes
            wpages = self.wrows[slot][self._wdiv[lo:end + garbage]]
            wat = wpages * wps + self._wmod[lo:end + garbage]
            if max(self.wpage_users[wpages[n:]].tolist()) > 1:
                bad = pos0 + int(np.flatnonzero(self.wpage_users[wpages[n:]] > 1)[0])
                self._fail(plan, f"slot {slot}: window KV write at position {bad} into window page "
                                 f"{int(self.wrows[slot, bad // wps])}, which another slot shares")
            got = f["wkv"][wat[:n]].tolist()
            if got != hist[lo:pos0]:
                bad = lo + _first_diff(got, hist[lo:pos0])
                self._fail(plan, f"slot {slot}: window KV at position {bad} (seen by a query at {pos0}) holds "
                                 f"{int(self.wkv[self.wrows[slot, bad // wps], bad % wps])}, not its token "
                                 f"{hist[bad]}: a wrong window, or a page released while still in view")
            read_bad |= bool(f["wkv_bad"][wat[:n]].any())
            if precise:
                prec = f["wkv_prec"][wat[:n]]
                if not prec.all():
                    self._fail(plan, f"slot {slot}: a PRECISE row reads window KV at position "
                                     f"{lo + int(prec.argmin())} that no PRECISE row wrote")
            flipped = self._become(wpages[n:], 2)
            f["wkv"][wat[n:]] = hist[pos0:end] + [-2] * garbage
            f["wkv_bad"][wat[n:]] = False
            f["wkv_prec"][wat[n:]] = precise
            if flipped:  # its full pages were not overwritten as window pages (nothing else rewrites them here)
                got = f["kv"][at[:end]].tolist()
                if got != hist[:end]:
                    self._fail(plan, f"slot {slot}: KV at position {_first_diff(got, hist)} was overwritten by a "
                                     "window page's write")
        return read_bad

    def _check_kv_full(self, plan: StepPlan, slot: int, pos0: int, nq: int, inputs: np.ndarray | None,
                       garbage: int = 0, img: np.ndarray | None = None, precise: bool = False) -> bool:
        """`_check_kv`'s reference implementation (`full_checks`): every array indexed by (page, offset), the whole
        context re-read after the window writes too."""
        ps, end = self.ps, pos0 + nq - garbage
        hist, imgs = self._kv_history(plan, slot, pos0, inputs, img)
        pos = np.arange(end)
        wpos = np.arange(pos0, end + garbage)
        wpages = self.rows[slot, wpos // ps]
        if (self.page_users[wpages] > 1).any():
            bad = int(wpos[self.page_users[wpages] > 1][0])
            self._fail(plan, f"slot {slot}: KV write at position {bad} into page {int(self.rows[slot, bad // ps])}, "
                             f"which another slot shares")
        read_bad = bool(self.kv_bad[self.rows[slot, pos[:pos0] // ps], pos[:pos0] % ps].any())
        if precise and not self.kv_prec[self.rows[slot, pos[:pos0] // ps], pos[:pos0] % ps].all():
            bad = int(np.flatnonzero(~self.kv_prec[self.rows[slot, pos[:pos0] // ps], pos[:pos0] % ps])[0])
            self._fail(plan, f"slot {slot}: a PRECISE row reads KV at position {bad} that no PRECISE row wrote")
        self._become(wpages, 1)
        self.kv[wpages, wpos % ps] = hist[pos0:end] + [-2] * garbage
        self.kv_img[wpages, wpos % ps] = imgs[pos0:end] + [-1] * garbage
        self.kv_bad[wpages, wpos % ps] = False
        self.kv_prec[wpages, wpos % ps] = precise
        got = self.kv[self.rows[slot, pos // ps], pos % ps]
        want = np.asarray(hist[:end], np.int64)
        if not np.array_equal(got, want):
            bad = int(np.flatnonzero(got != want)[0])
            self._fail(plan, f"slot {slot}: KV at position {bad} was overwritten by another request")
        if self.window:  # window pages: the positions this row's queries see must hold its own tokens
            wps = self.wps
            lo = max(0, pos0 - self.window + 1)
            seen = np.arange(lo, pos0)
            wp = self.wrows[slot, wpos // wps]
            if (self.wpage_users[wp] > 1).any():
                bad = int(wpos[self.wpage_users[wp] > 1][0])
                self._fail(plan, f"slot {slot}: window KV write at position {bad} into window page "
                                 f"{int(self.wrows[slot, bad // wps])}, which another slot shares")
            sp = self.wrows[slot, seen // wps]
            got = self.wkv[sp, seen % wps]
            if not np.array_equal(got, want[lo:pos0]):
                bad = int(seen[np.flatnonzero(got != want[lo:pos0])[0]])
                self._fail(plan, f"slot {slot}: window KV at position {bad} (seen by a query at {pos0}) holds "
                                 f"{int(self.wkv[self.wrows[slot, bad // wps], bad % wps])}, not its token "
                                 f"{int(want[bad])}: a wrong window, or a page released while still in view")
            read_bad |= bool(self.wkv_bad[sp, seen % wps].any())
            if precise and not self.wkv_prec[sp, seen % wps].all():
                bad = int(seen[np.flatnonzero(~self.wkv_prec[sp, seen % wps])[0]])
                self._fail(plan, f"slot {slot}: a PRECISE row reads window KV at position {bad} that no PRECISE row "
                                 "wrote")
            self._become(wp, 2)
            self.wkv[wp, wpos % wps] = hist[pos0:end] + [-2] * garbage
            self.wkv_bad[wp, wpos % wps] = False
            self.wkv_prec[wp, wpos % wps] = precise
            got = self.kv[self.rows[slot, pos // ps], pos % ps]  # its full pages were not overwritten as window pages
            if not np.array_equal(got, want):
                bad = int(np.flatnonzero(got != want)[0])
                self._fail(plan, f"slot {slot}: KV at position {bad} was overwritten by a window page's write")
        return read_bad

    def _read_bad(self, slot: int, end: int) -> bool:
        """Strict mode: whether `slot`'s KV at [0, end) holds a poisoned position (`_check_kv`'s flat index;
        `full_checks`: by page and offset)."""
        if self.full_checks:
            return bool(self.kv_bad[self.rows[slot, np.arange(end) // self.ps], np.arange(end) % self.ps].any())
        return bool(self.flat["kv_bad"][self.rows[slot][self._pdiv[:end]] * self.ps + self._pmod[:end]].any())
