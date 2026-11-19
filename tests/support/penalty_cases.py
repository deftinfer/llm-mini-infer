"""The penalties' and logit bias's runs on the tiny models (tests/runner/test_cuda_penalties.py): requests through the scheduler and engine loop, each token's logits from the
runner's hook, and the oracle, vLLM's terms written from its v2 sampler (`vllm_terms`)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from llm_mini_infer.engine.core import EngineCore
from llm_mini_infer.engine.messages import AddRequest
from llm_mini_infer.grammar.engine import Grammars
from llm_mini_infer.runner.cuda.sampler import GREEDY_EPS, plan_sampling, row_keys, sample, unpack_masks
from llm_mini_infer.sched.scheduler import Scheduler

from .fake_grammar import FakeGrammar, spec
from .tiny_gemma import gemma_cpu_runner, gemma_pools
from .tiny_qwen import cpu_runner, pools

EOS = 511
VOCAB = 512


def seq(n: int, salt: int = 0) -> list[int]:
    """A prompt that repeats tokens (penalties see repeats): n tokens over a range of about n / 2 ids."""
    return [(i * 37 + 11 + salt * 97) % max(7, n // 2) + 3 * salt for i in range(n)]


# Requests with every kind of term (`run`'s fields): greedy and seeded, each sampler path, a grammar with a forced
# prefix (its tokens are outputs the device never samples), unpenalised neighbours
REQS = [dict(rid=0, prompt=seq(24, 0), rep=1.6, max_tokens=16),
        dict(rid=1, prompt=seq(30, 1), pres=1.5, freq=0.7, max_tokens=16, temperature=0.9, seed=11),
        dict(rid=2, prompt=seq(18, 2), bias={5: 4.0, 17: -100.0, 40: 2.5}, max_tokens=12),
        dict(rid=3, prompt=seq(21, 3), rep=0.7, freq=-1.0, bias={9: 1.5}, max_tokens=14, temperature=1.0, top_k=20,
             seed=12),
        dict(rid=4, prompt=seq(16, 4), rep=1.3, pres=0.8, max_tokens=12, temperature=0.8, top_p=0.9, min_p=0.05,
             seed=13),
        dict(rid=5, prompt=seq(20, 5), grammar=spec(1, 4, 9, lead=2, tail=1), rep=2.0, freq=1.0, bias={3: 3.0}),
        dict(rid=6, prompt=seq(12, 6), max_tokens=10),  # unpenalised
        dict(rid=7, prompt=seq(15, 7), max_tokens=10, temperature=1.0, seed=14)]  # unpenalised, seeded


def add_request(q: dict) -> AddRequest:
    bias = sorted((q.get("bias") or {}).items())
    return AddRequest(rid=q["rid"], frontend=0, prompt=np.asarray(q["prompt"], np.int32),
                      max_tokens=q.get("max_tokens", 14), temperature=q.get("temperature", 0.0),
                      top_p=q.get("top_p", 1.0), top_k=q.get("top_k", 0), min_p=q.get("min_p", 0.0),
                      seed=q.get("seed", 100 + q["rid"]), stop_token_ids=[], grammar=q.get("grammar", ""),
                      repetition_penalty=q.get("rep", 1.0), presence_penalty=q.get("pres", 0.0),
                      frequency_penalty=q.get("freq", 0.0), bias_ids=[t for t, _ in bias],
                      bias_values=[v for _, v in bias])


def run(kind: str, model: Any, reqs: list[dict], *, k: int = 0, penalties: bool = True, budget: int = 64,
        num_pages: int = 512, max_num_seqs: int = 4, ckpt_slots: int = 0, caching: bool = True,
        sequential: bool = False, hook: bool = True, window: int = 8, wps: int = 4,
        make: Any = None) -> tuple[dict, dict, Scheduler]:
    """`reqs` through the scheduler and engine loop until idle (`sequential`: each to completion before the next is
    added, so later ones hit earlier ones' prefixes): ({rid: (tokens, finish)}, {(rid, position): the logits there,
    the last computed}, the scheduler). `penalties`: the runner and scheduler take them (`server.penalties`); `make`:
    the runner (default: on CPU)."""
    if kind == "qwen":
        pc = pools(max_num_seqs=max_num_seqs, budget=budget, page_size=wps, mtp_k=k, num_pages=num_pages,
                   state_slots=max_num_seqs + 1 + ckpt_slots)
        r = (make or cpu_runner)(model, pc, penalties=penalties)
        extra: dict = {}
    else:
        pc = gemma_pools(max_num_seqs=max_num_seqs, budget=budget, page_size=wps, mtp_k=k, num_pages=num_pages)
        r = (make or gemma_cpu_runner)(model[0], pc, penalties=penalties, mtp_k=k, assistant=model[1] if k else None)
        extra = dict(window=window, window_page_size=pc.window_page_size, draft_kv=False, checkpoint_grid=wps)
    sched = Scheduler(max_num_seqs=max_num_seqs, max_num_batched_tokens=budget, max_model_len=pc.max_model_len,
                      page_size=pc.page_size, num_pages=pc.num_pages, num_state_slots=pc.num_state_slots,
                      eos_token_ids=(EOS,), mtp_k=k, grammars=Grammars(FakeGrammar(VOCAB, (EOS,))),
                      forced_prefix=True, eos_free_finish=True, prefix_caching=caching,
                      batch_invariant=r.capabilities().batch_invariant, penalties_vocab=VOCAB if penalties else 0,
                      **extra)
    core = EngineCore(sched, r)
    logits: dict[tuple[int, int], torch.Tensor] = {}
    rids: list[int] = []
    orig = r.launch_forward

    def launch(plan: Any, *a: Any, **kw: Any) -> Any:
        rids[:] = [sched.slot_req[s].rid for s in plan.sched_slot.tolist()]
        return orig(plan, *a, **kw)

    def on_logits(chunk: Any) -> None:
        lg = chunk.logits.detach().to("cpu", torch.float32)
        for i, (row, pos) in enumerate(zip(chunk.rows.tolist(), chunk.positions.tolist())):
            logits[(rids[row], pos)] = lg[i].clone()

    r.launch_forward = launch
    if hook:
        r.logits_hook = on_logits
    out: dict[int, tuple[list[int], int]] = {q["rid"]: ([], 0) for q in reqs}

    def drain() -> None:
        n = 0
        while core.busy and n < 4000:
            rep = core.step()
            n += 1
            if rep is None:
                continue
            for o in rep.outputs.values():
                off = 0
                for rid, nn, f in zip(o.rids.tolist(), o.n_new.tolist(), o.finish.tolist()):
                    toks, fin = out[rid]
                    toks.extend(o.tokens[off:off + nn].tolist())
                    off += nn
                    out[rid] = (toks, f or fin)
        assert not core.busy

    for q in reqs:
        core.handle(add_request(q))
        if sequential:
            drain()
    drain()
    assert sched.grammars.errors == 0
    r.shutdown()
    return out, logits, sched


def vllm_terms(x: torch.Tensor, prompt: list[int], outs: list[int], rep: float = 1.0, pres: float = 0.0,
               freq: float = 0.0, bias: dict | None = None) -> torch.Tensor:
    """vLLM v0.33.0's v2 sampler on one row (`gpu/sample/logit_bias.py`, `penalties.py`), in fp32: the bias added, then
    repetition over the prompt and the output (`logits *= where(logits > 0, 1 / scale, scale)`), frequency and
    presence over the output."""
    x = x.to(torch.float32).clone()
    v = len(x)
    for t, b in (bias or {}).items():
        x[t] += b
    cnt = torch.bincount(torch.tensor(outs, dtype=torch.long), minlength=v)[:v] if outs else \
        torch.zeros(v, dtype=torch.long)
    seen = torch.zeros(v, dtype=torch.bool)
    seen[torch.tensor(prompt, dtype=torch.long)] = True
    scale = torch.where(seen | (cnt > 0), torch.tensor(rep, dtype=torch.float32), torch.tensor(1.0))
    x = x * torch.where(x > 0, 1.0 / scale, scale)
    x = x - freq * cnt.to(torch.float32)
    return x - pres * (cnt > 0).to(torch.float32)


def check_oracle(reqs: list[dict], out: dict, logits: dict, tie: float = 1e-4) -> int:
    """Each sampled token against the oracle: vLLM's terms (`vllm_terms`) over the prompt and the tokens before it, on
    the logits the runner computed at its position (its hook), its grammar's mask applied, then greedy's argmax or the
    reference sampler with the row's seed and position. A greedy token whose top two adjusted logits lie within `tie`
    is skipped (the hook's GEMM has other rows than the sampler's), and so is a token the grammar forced. Returns how
    many were checked."""
    grammar = FakeGrammar(VOCAB, (EOS,))
    checked = 0
    for q in reqs:
        toks, _ = out[q["rid"]]
        m = grammar.instance(grammar.compile(q["grammar"])) if q.get("grammar") else None
        plen = len(q["prompt"])
        for i, t in enumerate(toks):
            allowed = torch.from_numpy(m.allowed().copy()) if m is not None else None
            forced = allowed is not None and int(allowed.sum()) == 1
            if m is not None:
                if t == EOS:
                    m = None
                else:
                    assert m.consume_tokens([t])
            if forced:
                continue
            x = vllm_terms(logits[(q["rid"], plen + i - 1)], q["prompt"], toks[:i], q.get("rep", 1.0),
                           q.get("pres", 0.0), q.get("freq", 0.0), q.get("bias"))
            if allowed is not None:
                x = x.masked_fill(~allowed, float("-inf"))
            temp = q.get("temperature", 0.0)
            if temp < GREEDY_EPS:
                top = torch.topk(x, 2).values
                if float(top[0] - top[1]) < tie:
                    continue
                assert t == int(x.argmax()), (q["rid"], i, t, int(x.argmax()))
            else:
                p = (np.array([temp], np.float32), np.array([q.get("top_p", 1.0)], np.float32),
                     np.array([q.get("top_k", 0)], np.int64), np.array([q.get("min_p", 0.0)], np.float32))
                plan = plan_sampling(*p, VOCAB)
                for g in plan.groups:
                    g.idx = torch.as_tensor(g.rows)
                keys = row_keys(torch.tensor([q.get("seed", 100 + q["rid"])]), torch.tensor([plen + i]))
                want, _ = sample(x[None], plan, *(torch.from_numpy(a) for a in p), keys, torch.arange(VOCAB))
                assert t == int(want[0]), (q["rid"], i, t, int(want[0]))
            checked += 1
    return checked


__all__ = ["EOS", "REQS", "VOCAB", "add_request", "check_oracle", "run", "seq", "unpack_masks", "vllm_terms"]
