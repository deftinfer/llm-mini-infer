"""Teacher forcing through the real engine: a grammar backend that allows a reference's next token alone at every
position (then its EOS), so a request's answer is the reference whatever the logits; a capture of each sample token's
final hidden state and logits, kept where the inputs up to it are the reference's; and a forced run of a few requests
through the scheduler and a GemmaRunner (`ForcedRun`) or the Qwen runner (`QwenForcedRun`) that returns one request's
captures, its images' encoder rows among them. tests/repo/test_repro_harness.py checks them on the CPU;
tests/runner/test_batch_invariance_gpu.py and tests/runner/test_qwen_batch_invariance_gpu.py use them to compare a
request's bits alone and beside others.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# -- the forced grammar ------------------------------------------------------------------------------------------------


class ForcedMatcher:
    """A request's forced grammar: the reference's next token alone at each position, then its EOS
    (`grammar.engine.Matcher`); its first `forced` tokens are forced (the forced prefix)."""

    def __init__(self, seq: list[int], eos: int, forced: int):
        self.seq, self.eos, self.forced = seq, eos, forced
        self.n, self.err = 0, ""

    def next(self, i: int | None = None) -> int:
        i = self.n if i is None else i
        return self.seq[i] if i < len(self.seq) else self.eos

    def _takes(self, t: int, i: int) -> bool:
        return i <= len(self.seq) and t == self.next(i)

    def consume_tokens(self, tokens: list[int]) -> bool:
        for t in tokens:
            if self.err or not self._takes(t, self.n):
                self.err = self.err or f"token {t} at {self.n} is not the reference's"
                return False
            self.n += 1
        return True

    def try_consume_tokens(self, tokens: list[int]) -> int:
        k = 0
        while not self.err and k < len(tokens) and self._takes(tokens[k], self.n + k):
            k += 1
        self.n += k
        return k

    def validate_tokens(self, tokens: list[int]) -> int:
        k = 0
        while k < len(tokens) and self._takes(tokens[k], self.n + k):
            k += 1
        return k

    def rollback(self, num_tokens: int) -> None:
        if not 0 <= num_tokens <= self.n:
            raise ValueError(f"rollback of {num_tokens} tokens after {self.n}")
        self.n -= num_tokens

    def is_stopped(self) -> bool:
        return self.n >= len(self.seq)

    def is_error(self) -> bool:
        return bool(self.err)

    def get_error(self) -> str:
        return self.err

    def compute_ff_tokens(self) -> list[int]:
        return [self.seq[self.n]] if self.n < min(self.forced, len(self.seq)) else []

    def deep_copy(self) -> ForcedMatcher:
        m = ForcedMatcher(self.seq, self.eos, self.forced)
        m.n, m.err = self.n, self.err
        return m


class Forced:
    """`grammar.engine.Backend` of the forced grammars: a grammar's text names its reference (`forced:<i>`)."""

    name = "forced"

    def __init__(self, vocab_size: int, eos: int, refs: dict[int, list[int]], forced: int):
        self.vocab_size, self.eos, self.refs, self.forced = vocab_size, (eos,), refs, forced

    def compile(self, text: str) -> list[int]:
        return self.refs[int(text.partition(":")[2])]

    def instance(self, compiled: list[int]) -> ForcedMatcher:
        return ForcedMatcher(compiled, self.eos[0], self.forced)

    @staticmethod
    def _one(buf: np.ndarray, row: int, token: int) -> None:
        buf[row] = 0
        buf.view(np.uint32)[row, token >> 5] = np.uint32(1) << np.uint32(token & 31)

    def fill(self, singles: list[tuple[Any, int]], drafts: list[tuple[Any, int, list[int]]], buf: np.ndarray) -> None:
        for m, i in singles:
            self._one(buf, i, m.next())
        for m, i, d in drafts:  # along the drafts, as llguidance's: a mask after each, to the first it rejects
            if not d:
                raise ValueError("Draft tokens must not be empty")
            for j in range(len(d) + 1):
                self._one(buf, i + j, m.next(m.n + j))
                if j < len(d) and not m._takes(d[j], m.n + j):
                    break

    @staticmethod
    def only(m: ForcedMatcher, token: int, own: bool = False) -> bool:  # `own`: no state shared either way
        return token == m.next()


# -- the captures ------------------------------------------------------------------------------------------------------


class Logits:
    """The target head's logits at each sampling, as the sampling computes them (in its graphs too). Installed on a
    GemmaRunner before `configure` (its graphs then hold the copies), it wraps the sparse head (`_sparse_logits`) and
    the dense one (the model's `logits`) so that each copies its logits (fp32 [n, columns]) and its columns' token ids
    (the union's ascending ids, padded with −1 to the head's rows, `runner.gs.rows`: `sparse_head_rows`, at most the
    vocabulary; or every id) into static buffers, which `Capture` clones after each sampling launch, stream-ordered.
    The assistant's heads are left alone."""

    def __init__(self, runner: Any):
        import torch

        with torch.inference_mode():  # written inside the runner's inference mode, as its own buffers
            self.logits = torch.zeros((runner.S * runner.L, runner.cfg.vocab_size), dtype=torch.float32,
                                      device=runner.device)
            self.ids = torch.full((runner.cfg.vocab_size,), -1, dtype=torch.int32, device=runner.device)
        sparse, dense = runner._sparse_logits, runner.model.logits
        runner._sparse_logits = lambda hidden, out=None: self._put(sparse(hidden, out), runner.gs.sp_src)
        runner.model.logits = lambda hidden, out=None: self._put(dense(hidden, out), runner._vocab_ids)

    def _put(self, logits: Any, ids: Any) -> Any:
        n, w = logits.shape
        self.logits[:n, :w].copy_(logits)
        self.ids[:w].copy_(ids)
        self.ids[w:].fill_(-1)
        return logits


class Capture:
    """Each step's sample tokens on a CUDA runner: their slot, position and input token, their final hidden state
    (copied after the forward, before the sampling graph reads it), with `logits` (`Logits`, installed before the
    runner's `configure`) their logits and the columns' token ids (copied after the sampling launch); and each plan's
    slots' request ids. `true` holds each request's prompt and reference: the inputs a valid state's window must have.
    """

    def __init__(self, runner: Any, core: Any, logits: Logits | None = None):
        import torch

        from llm_mini_infer.runner.cuda.runner import CudaRunner

        self.runner, self.sched, self.logits = runner, core.sched, logits
        # The Qwen runner under MTP keeps every token's final hidden state in `hid_out` (its drafter reads them), Gemma
        # 4's and Qwen's without MTP the sample tokens' alone
        every = isinstance(runner, CudaRunner) and runner.k > 0
        self.steps: list[tuple[Any, ...]] = []  # (step, slots, positions, input tokens, states[, logits, their ids])
        self.rids: dict[int, dict[int, int]] = {}  # step -> slot -> rid
        self.true: dict[int, np.ndarray] = {}  # rid -> its prompt and reference
        orig_lf, orig_ls = runner.launch_forward, runner.launch_sample

        def launch_forward(plan: Any, _internal: bool = False, _capture: Any = None) -> Any:
            if not _internal and _capture is None:
                self.rids[int(plan.step_id)] = {s: int(self.sched.fe_rid[s]) for s in plan.sched_slot.tolist()}
            return orig_lf(plan, _internal=_internal, _capture=_capture)

        def launch_sample(h: Any, n_masks: int = 0, _heads: Any = None) -> Any:
            sp, rec = h.split, None
            if sp is not None and not h.internal:
                kind, size, _, lay, _ = sp
                t, _, _, _ = runner._sizes(kind, size, lay)
                n = lay.n_stok
                if n:
                    s, tb = runner.static, runner.tables
                    pin, slot = s.pos[:t], s.slot[:t]
                    pos = torch.where(pin >= 0, pin, tb.seq_len.index_select(0, slot) + (-1 - pin).clamp(min=0))
                    sel = s.s_idx[:n]
                    rec = (int(h.step_id), s.s_slot[:n].clone(), pos.index_select(0, sel),
                           runner._ids(t).index_select(0, sel),
                           s.hid_out.index_select(0, sel) if every else s.hid_out[:n].clone())
            out = orig_ls(h, n_masks, _heads)
            if rec is not None:
                if self.logits is not None:  # behind the sampling just launched
                    rec += (self.logits.logits[:len(rec[1])].clone(), self.logits.ids.clone())
                self.steps.append(rec)
            return out

        runner.launch_forward = launch_forward
        runner.launch_sample = launch_sample

    def drain(self, keep: dict[tuple[int, int], np.ndarray],
              logits: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] | None = None) -> tuple[int, int]:
        """The captured steps into `keep` ((rid, position) -> the hidden state's bits, uint16) and `logits` (the same
        keys -> the columns' token ids, ascending, and the logits' bits, uint32), valid states only: a window's states
        count up to its first input that is not the reference's. Returns (kept, dropped)."""
        import torch

        kept = dropped = 0
        for step, slot, pos, tok, hid, *lg in self.steps:
            slot, pos, tok = slot.cpu().numpy(), pos.cpu().numpy(), tok.cpu().numpy()
            hid = hid.view(dtype=torch.int16).cpu().numpy().view(np.uint16)
            if lg:
                ids = lg[1].cpu().numpy()
                cols = np.flatnonzero(ids >= 0)
                cols = cols[np.argsort(ids[cols], kind="stable")]
                bits = lg[0].cpu().numpy()[:, cols].view(np.uint32)
                ids = ids[cols]
            by_rid = self.rids.get(step, {})
            ok_row: dict[tuple[int, int], bool] = {}
            for j in range(len(slot)):
                rid = by_rid.get(int(slot[j]))
                seq = self.true.get(rid) if rid is not None else None
                if seq is None:
                    continue
                key = (int(step), int(slot[j]))
                p = int(pos[j])
                ok = ok_row.get(key, True) and p < len(seq) and int(tok[j]) == int(seq[p])
                ok_row[key] = ok
                if ok:
                    keep[(rid, p)] = hid[j]
                    if lg and logits is not None:
                        logits[(rid, p)] = (ids, bits[j])
                    kept += 1
                else:
                    dropped += 1
        self.steps.clear()
        return kept, dropped


# -- a forced run ------------------------------------------------------------------------------------------------------


@dataclass
class Bits:
    """A request's captures in a forced run, by position: its final hidden state's bits (uint16), its logits (the
    columns' token ids, ascending, and their bits, uint32) and, by each image's first position, the image's encoder rows
    as its ENCODE wrote them (uint16 [tokens, width]); and the run's grammar-head counters, the images each ENCODE held
    and the images each tower call held."""

    hidden: dict[int, np.ndarray]
    logits: dict[int, tuple[np.ndarray, np.ndarray]]
    images: dict[int, np.ndarray]
    heads: Any
    encodes: list[int]
    calls: list[int]


class ForcedRun:
    """Requests whose answers are forced (`Forced`), through the scheduler and the engine over a GemmaRunner on
    `device` (`gemma_mtp.Run`, whose other arguments `kw` takes, the runner's switches among them), each sample
    token's final hidden state and logits captured (`Logits`, `Capture`), and each image's encoder rows after its
    ENCODE. `prompts`: each request's parts, token lists and PNG images; a model with a vision tower takes the images
    through a pixel arena at `arena`. `add` a request, run `steps` of the engine, then `bits` runs the rest and returns
    one request's captures."""

    def __init__(self, pair: tuple[Path, Path], prompts: dict[int, list], answers: dict[int, list[int]], *,
                 device: str = "cuda", arena: str | None = None, k: int = 3, forced: int = 0, **kw: Any):
        """`forced`: an answer's first tokens its grammar forces (the forced prefix's), 0: none."""
        from llm_mini_infer.grammar.engine import Grammars
        from llm_mini_infer.runner.cuda.images import VisionOptions

        from .gemma_images import Pixels

        Run, eos, vocab = self._driver(pair[0])
        self.device, self.answers = device, answers
        self.encodes: list[int] = []
        self.calls: list[int] = []
        self.images: dict[int, Any] = {}  # span key -> its encoder rows, as its first ENCODE wrote them
        self.pixels = Pixels(pair[0], arena) if arena else None
        if arena:
            kw["vision"] = VisionOptions(encoder_rows=512, encoder_budget=160, arena=arena)
        self.head: Logits | None = None
        self.run = Run(self._model(pair), k, grammars=Grammars(Forced(vocab, eos, answers, forced)), make=self._make,
                       **kw)
        self.encodes.clear()  # the requests' only
        self.calls.clear()
        self.images.clear()
        self.cap = Capture(self.run.r, self.run.core, self.head)
        self.prompts = {rid: self.pixels.prompt(parts)[:2] if self.pixels else ([t for p in parts for t in p], [])
                        for rid, parts in prompts.items()}
        for rid, (ids, _) in self.prompts.items():
            self.cap.true[rid] = np.asarray(ids + answers[rid], np.int64)

    @staticmethod
    def _driver(model_dir: Path) -> tuple[Any, int, int]:
        """The engine driver of Gemma 4's runner, its EOS and the model's vocabulary."""
        from llm_mini_infer.models.gemma4.model import Gemma4Config

        from .gemma_mtp import Run
        from .mtp_common import EOS

        return Run, EOS, Gemma4Config.from_dir(model_dir).vocab_size

    @staticmethod
    def _model(pair: tuple[Path, Path]) -> Any:
        return pair  # the target and its assistant

    def _runner(self, model_dir: Path, pc: Any, **kw: Any) -> Any:
        """A GemmaRunner as tests/models/test_gemma4_gpu.py's `cuda_runner` makes one (`tiny_gemma.gemma_cpu_runner`'s
        on the CPU)."""
        import torch

        from llm_mini_infer.runner.cuda.gemma import GemmaRunner

        cuda = self.device == "cuda"
        return GemmaRunner(model_dir, device=self.device, dtype=None if cuda else torch.float32,
                           max_num_seqs=pc.max_num_seqs, max_num_batched_tokens=pc.max_num_batched_tokens,
                           max_model_len=pc.max_model_len, memory_gb=2.0 if cuda else None, check=True, **kw)

    def _make(self, model_dir: Path, pc: Any, **kw: Any) -> Any:
        """The runner (`_runner`), its head's logits captured before its graphs are; then its ENCODEs and tower calls
        counted, and each ENCODE's rows copied behind it (stream-ordered)."""
        import torch

        r = self._runner(model_dir, pc, **kw)
        self.head = Logits(r)
        r.configure(pc)
        if r.img is not None:
            encode, tower = r.img.encode, r.img._tower

            def counted_encode(plan: Any, *a: Any) -> Any:
                self.encodes.append(len(plan.enc_key))
                out = encode(plan, *a)
                for key, r0, n in zip(plan.enc_key.tolist(), plan.enc_row0.tolist(), plan.enc_tokens.tolist()):
                    if key not in self.images:
                        rows = torch.as_tensor(plan.enc_rows[r0:r0 + n], device=r.img.pool.device)
                        self.images[key] = r.img.pool.index_select(0, rows)
                return out

            def counted_tower(patches: Any, grids: list, graphs: bool) -> Any:
                self.calls.append(len(grids))
                return tower(patches, grids, graphs)

            r.img.encode, r.img._tower = counted_encode, counted_tower
        return r

    def add(self, rid: int) -> None:
        ids, refs = self.prompts[rid]
        self.run.add(rid, ids, max_tokens=max(80, len(self.answers[rid]) + 8), grammar=f"forced:{rid}",
                     ignore_eos=False, images=refs)

    def steps(self, n: int) -> None:
        """`n` turns of the engine, their outputs kept (as `Run.run`)."""
        r = self.run
        for _ in range(n):
            rep = r.core.step()
            for out in rep.outputs.values() if rep is not None else ():
                off = 0
                for rid, nn, fin in zip(out.rids.tolist(), out.n_new.tolist(), out.finish.tolist()):
                    r.out[rid].extend(out.tokens[off:off + nn].tolist())
                    off += nn
                    if fin:
                        r.finish[rid] = fin

    def bits(self, rid: int = 0) -> Bits:
        """Run to the end (every answer the forced one, then the runner shut down) and return `rid`'s captures where
        its inputs are the reference's (`Capture.drain`), with its images' encoder rows."""
        import torch

        r = self.run
        r.run()
        assert r.sched.grammars.errors == 0
        assert all(r.out[q][:len(self.answers[q])] == self.answers[q] for q in r.out)  # every answer the forced one
        hidden: dict[tuple[int, int], np.ndarray] = {}
        logits: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        self.cap.drain(hidden, logits)
        images = {ref.offset: self.images[ref.key].view(dtype=torch.int16).cpu().numpy().view(np.uint16)
                  for ref in self.prompts[rid][1]}
        heads = r.r.head_counters()
        r.r.shutdown()
        return Bits({p: v for (q, p), v in hidden.items() if q == rid},
                    {p: v for (q, p), v in logits.items() if q == rid}, images, heads, self.encodes, self.calls)


class QwenForcedRun(ForcedRun):
    """`ForcedRun` over the Qwen runner (`CudaRunner`, as tests/support/tiny_qwen.py's `cuda_runner` makes one) and its
    engine driver (`qwen_mtp.Run`: `caching` and `ckpt_slots` for the prefix cache), on a tiny Qwen (`model`: its
    folder; images through a pixel arena when it has a tower)."""

    def __init__(self, model: Path, prompts: dict[int, list], answers: dict[int, list[int]], **kw: Any):
        super().__init__((model, model), prompts, answers, **kw)

    @staticmethod
    def _driver(model_dir: Path) -> tuple[Any, int, int]:
        from llm_mini_infer.models.qwen3_5.model import Qwen35Config

        from .mtp_common import EOS
        from .qwen_mtp import Run

        return Run, EOS, Qwen35Config.from_dir(model_dir).vocab_size

    @staticmethod
    def _model(pair: tuple[Path, Path]) -> Any:
        return pair[0]

    def _runner(self, model_dir: Path, pc: Any, **kw: Any) -> Any:
        import torch

        from llm_mini_infer.runner.cuda.runner import CudaRunner

        cuda = self.device == "cuda"
        return CudaRunner(model_dir, device=self.device, dtype=None if cuda else torch.float32,
                          max_num_seqs=pc.max_num_seqs, max_num_batched_tokens=pc.max_num_batched_tokens,
                          max_model_len=pc.max_model_len, memory_gb=1.0 if cuda else None, check=True, mtp_k=pc.mtp_k,
                          **kw)
