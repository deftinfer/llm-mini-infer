"""The tiny Gemma 4 pair's engine driver and prompts on the CPU (tests/models/test_gemma4_mtp.py,
tests/models/test_gemma4_mtp_depth.py)."""

from __future__ import annotations

import numpy as np
import torch

from llm_mini_infer.engine.core import EngineCore
from llm_mini_infer.engine.messages import AddRequest, ImageRef
from llm_mini_infer.sched.scheduler import Scheduler
from llm_mini_infer.vision import family as vf

from .mtp_common import EOS, seq
from .tiny_gemma import gemma_cpu_runner, gemma_pools

W = 8


class Run:
    """The scheduler (window mode) and the engine loop over a GemmaRunner (`make`: CPU by default), with an optional
    oracle drafter. A runner with images (its `vision` option) gets the scheduler's encoder cache and image spans;
    `forced_prefix`: the scheduler's (with `grammars`), and its batch invariance the runner's. `draft_counts`: the
    runner's counts below k; `count_schedule` cycles counts over the plans with decode rows, `adaptive` lets the
    scheduler's policy choose among them (`sched.draft_counts`)."""

    def __init__(self, pair, k: int, *, caching: bool = True, max_num_seqs: int = 4, budget: int = 64,
                 num_pages: int = 512, interval: int = 8, grammars=None, make=gemma_cpu_runner, window: int = W,
                 wps: int = 4, forced_prefix: bool = False, draft_counts: tuple[int, ...] = (),
                 count_schedule: tuple[int, ...] = (), adaptive: bool = False, **runner_kw):
        target, assistant = pair
        if draft_counts:
            runner_kw["draft_counts"] = draft_counts
        pc = gemma_pools(max_num_seqs=max_num_seqs, budget=budget, page_size=wps, num_pages=num_pages,
                         max_model_len=256, mtp_k=k)
        self.r = make(target, pc, mtp_k=k, assistant=assistant if k else None, **runner_kw)
        self.k = k
        caps = self.r.capabilities()
        mm = dict(encoder_rows=caps.encoder_rows, encoder_budget=caps.encoder_budget,
                  bidirectional_images=caps.bidirectional_images, positions=vf.load(target).positions) \
            if caps.images else {}
        self.sched = Scheduler(max_num_seqs=max_num_seqs, max_num_batched_tokens=budget, max_model_len=256,
                               page_size=pc.page_size, num_pages=pc.num_pages, num_state_slots=max_num_seqs + 1,
                               eos_token_ids=(EOS,), mtp_k=k, prefix_caching=caching, checkpoint_grid=wps,
                               decode_checkpoint_interval=interval, window=window, window_page_size=wps,
                               draft_kv=caps.draft_kv, grammars=grammars, forced_prefix=forced_prefix,
                               batch_invariant=caps.batch_invariant, count_schedule=count_schedule,
                               draft_counts=caps.draft_counts if adaptive else (), **mm)
        self.core = EngineCore(self.sched, self.r)
        self.out: dict[int, list[int]] = {}
        self.finish: dict[int, int] = {}
        self.cached: dict[int, int] = {}
        self.prompts: dict[int, list[int]] = {}
        self.after_step = None  # called after every engine turn (tests read the tables)

    def add(self, rid: int, prompt: list[int], max_tokens: int = 12, temperature: float = 0.0, seed: int = 0,
            stop: list[int] = (), ignore_eos: bool = True, top_p: float = 1.0, top_k: int = 0,
            grammar: str = "", images: list[ImageRef] = ()) -> None:
        self.prompts[rid] = list(prompt)
        self.out[rid] = []
        self.core.handle(AddRequest(rid=rid, frontend=0, prompt=np.asarray(prompt, np.int32), max_tokens=max_tokens,
                                    temperature=temperature, top_p=top_p, top_k=top_k, min_p=0.0,
                                    seed=seed or 100 + rid, stop_token_ids=list(stop), ignore_eos=ignore_eos,
                                    grammar=grammar, images=list(images)))

    def run(self, oracle: dict[int, list[int]] | None = None, rng: np.random.Generator | None = None) -> Run:
        n = 0
        while self.core.busy and n < 3000:
            rep = self.core.step()
            n += 1
            if oracle is not None:
                self._oracle(oracle, rng)
            if self.after_step is not None:
                self.after_step()
            if rep is None:
                continue
            for out in rep.outputs.values():
                for rid, cached in zip(out.first_rids.tolist(), out.first_cached.tolist()):
                    self.cached[rid] = cached
                off = 0
                for rid, nn, fin in zip(out.rids.tolist(), out.n_new.tolist(), out.finish.tolist()):
                    self.out[rid].extend(out.tokens[off:off + nn].tolist())
                    off += nn
                    if fin:
                        self.finish[rid] = fin
        assert not self.core.busy
        self.sched.check_accounting()
        return self

    @torch.inference_mode()
    def _oracle(self, ref: dict[int, list[int]], rng: np.random.Generator | None) -> None:
        """Every running slot's drafts become its next k reference tokens (with `rng`, a random number of them; the
        rest wrong): the verify window after the pending token at seq_len."""
        tb = self.r.tables
        for slot in range(1, len(self.sched.slot_req)):
            req = self.sched.slot_req[slot]
            if req is None or req.rid not in ref:
                continue
            full = self.prompts[req.rid] + ref[req.rid]
            at = int(tb.seq_len[slot])
            good = self.k if rng is None else int(rng.integers(0, self.k + 1))
            for j in range(self.k):
                p = at + 1 + j
                tb.drafts[slot, j] = full[p] if j < good and p < len(full) else (full[p] + 1) % 500 \
                    if p < len(full) else 7


PROMPTS = [seq(11, 1), seq(29, 2), seq(4, 3), seq(40, 4)]  # two past the window, one short


def counters(s: Scheduler, k: int, masked: bool = False) -> np.ndarray:
    """A run's acceptance counters add up: every verify row counts its k drafts (with a grammar, only those it takes:
    `Scheduler.spec_draft_tokens`), the accepted ones are a prefix of them (the count per position does not grow with
    the position, the first at most the rows) and sum to the accepted total. Returns the counts per position."""
    per, offered = s.spec_accepted_per_pos, s.spec_offered_per_pos
    assert len(per) == k and s.spec_drafts > 0
    assert 0 <= s.spec_accepted <= s.spec_draft_tokens <= s.spec_drafts * k
    if not masked:
        assert s.spec_draft_tokens == int(offered.sum())  # k each, or the count the plan verified
    assert int(per.sum()) == s.spec_accepted
    assert (np.diff(per) <= 0).all() and per[0] <= s.spec_drafts, per
    assert (per <= offered).all() and (np.diff(offered) <= 0).all() and offered[0] == s.spec_drafts, offered
    return per
