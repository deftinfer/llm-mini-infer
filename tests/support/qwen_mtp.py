"""The tiny Qwen with its MTP layer, its engine driver, and the always-split runs' requests
(tests/runner/test_cuda_mtp.py, tests/structured/test_qwen_always_split_gpu.py)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from llm_mini_infer.engine.core import EngineCore
from llm_mini_infer.engine.messages import AddRequest
from llm_mini_infer.sched.scheduler import Scheduler

from .fake_grammar import spec
from .mtp_common import EOS, seq
from .tiny_qwen import cpu_runner, make_tiny_model, pools


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    return make_tiny_model(tmp_path_factory.mktemp("mtp"), seed=1, mtp=True)


class Run:
    """The scheduler and engine loop over a CudaRunner (`make`: CPU by default), with an optional oracle drafter;
    `grammars`: the scheduler's (structured output), with its grammar switches (`forced_prefix`, `eos_free_finish`). A
    runner with images (its `vision` option) gets the scheduler's encoder cache and M-RoPE positions, and one with batch
    invariance its scheduler's (`Capabilities`)."""

    def __init__(self, model, k: int, *, caching: bool = False, max_num_seqs: int = 4, budget: int = 128,
                 num_pages: int = 512, ckpt_slots: int = 0, interval: int = 256, page_size: int = 4, grammars=None,
                 forced_prefix: bool = False, eos_free_finish: bool = False, make=cpu_runner, max_model_len: int = 1024,
                 **runner_kw):
        pc = pools(max_num_seqs=max_num_seqs, budget=budget, mtp_k=k, num_pages=num_pages, page_size=page_size,
                   state_slots=max_num_seqs + 1 + ckpt_slots, max_model_len=max_model_len)
        self.r = make(model, pc, **runner_kw)
        self.k = k
        caps = self.r.capabilities()
        mm = {}
        if caps.images:
            from llm_mini_infer.vision import family as vf

            mm = dict(encoder_rows=caps.encoder_rows, encoder_budget=caps.encoder_budget,
                      positions=vf.load(model).positions)
        self.sched = Scheduler(max_num_seqs=max_num_seqs, max_num_batched_tokens=budget,
                               max_model_len=pc.max_model_len, page_size=pc.page_size, num_pages=pc.num_pages,
                               num_state_slots=pc.num_state_slots, eos_token_ids=(EOS,), mtp_k=k,
                               prefix_caching=caching, decode_checkpoint_interval=interval, grammars=grammars,
                               forced_prefix=forced_prefix, eos_free_finish=eos_free_finish,
                               batch_invariant=caps.batch_invariant, **mm)
        self.core = EngineCore(self.sched, self.r)
        self.out: dict[int, list[int]] = {}
        self.finish: dict[int, int] = {}
        self.cached: dict[int, int] = {}
        self.prompts: dict[int, list[int]] = {}

    def add(self, rid: int, prompt: list[int], max_tokens: int = 12, temperature: float = 0.0, seed: int = 0,
            stop: list[int] = (), ignore_eos: bool = True, top_p: float = 1.0, top_k: int = 0,
            grammar: str = "", images: list = ()) -> None:
        self.prompts[rid] = list(prompt)
        self.out[rid] = []
        self.core.handle(AddRequest(rid=rid, frontend=0, prompt=np.asarray(prompt, np.int32), max_tokens=max_tokens,
                                    temperature=temperature, top_p=top_p, top_k=top_k, min_p=0.0,
                                    seed=seed or 100 + rid, stop_token_ids=list(stop), ignore_eos=ignore_eos,
                                    grammar=grammar, images=list(images)))

    def run(self, oracle: dict[int, list[int]] | None = None, rng: np.random.Generator | None = None) -> Run:
        """Until idle. `oracle`: rid -> the true continuation; after each step every running slot's drafts become the
        next k reference tokens (with `rng`, only a random number of them; the rest wrong)."""
        n = 0
        while self.core.busy and n < 2000:
            rep = self.core.step()
            n += 1
            if oracle is not None:
                self._oracle(oracle, rng)
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


BUDGET = 64  # the first prompt (150 tokens) in three chunks: the first two steps without sample rows


def requests(constrained: bool, n: int = 6) -> list[dict]:
    """`n` requests, the first's prompt longer than the step budget; with `constrained`, every other one under a
    JSON-like grammar with forced runs (its first two tokens, the forced prefix, and its last before the EOS), the
    others free and shorter: steps of both, then steps of the constrained ones alone (the sparse heads)."""
    out = []
    for i in range(n):
        q = {"rid": i, "prompt": seq(150 if i == 0 else 9 + 13 * i, i + 1), "max_tokens": 24}
        if constrained and i % 2 == 0:
            q.update(grammar=spec(i, 12, 20, one_in=16, universe=128, lead=2, tail=1))
        elif constrained:
            q.update(max_tokens=8)
        out.append(q)
    return out
