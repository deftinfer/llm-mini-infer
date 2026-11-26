#!/usr/bin/env python3
"""Offline fp32 analysis of teacher-forced captures (tests/scripts/numerics_forced.py): against the checkpoint run in
fp32, where each arm's accuracy goes, per layer and per kernel, how its error accumulates along the residual stream, and
what that costs the drafter's acceptance and the target's choices. Qwen3.5/3.8 (Gated DeltaNet and full-attention
layers, the MTP layer drafting) and Gemma 4 (sliding and full layers, the MTP assistant drafting).

**The reference** (`QwenRef`, `GemmaRef`): the checkpoint's own modules (`models.qwen3_5.model`, `models.gemma4.model`,
`models.gemma4.assistant`) on the torch reference ops (`ops.enabled(False)`), in fp32 (`--dtype`), without FP8
activation quantization (`fp8.bf16_activations`), TF32 off (`reference_numerics`). A layer is read from the checkpoint
when it runs, each block-FP8 projection's weight dequantized to fp32 (`kernels.fp8.dequant_weight`), and dropped after;
the embeddings and the head stay as stored, widened a vocabulary chunk at a time. Attention is the reference SDPA over
pages of one position (`TorchAttention`; `TorchSlidingAttention`, `TorchFullAttention`); the GDN recurrence is
`torch_gdn_prefill`'s token loop with every source of a request in one loop (`recurrence`). RoPE reads the captured
positions (Qwen's M-RoPE, decode positions past the last image included); Gemma's sliding layers see image spans
bidirectionally when the checkpoint says so. `--dtype float64` stores and multiplies in float64, but the reference ops
still reduce their norms and activations in fp32.

**Inputs:** arm folders of one forced run (`<run>/<arm>/r<rid>.pt`, or the run's folder for all its arms), the same
references in each, in tests/scripts/numerics_forced.py's capture format; the checkpoints (folders or repo ids,
`llm_mini_infer.hub`) from the captures or `--model`/`--assistant`. A capture keeps a position where the reference's
inputs computed it (`hvalid`, `tvalid`). A source's position it lacks (a prefix-cache hit, say) is taken from another
arm that has it, else from the fp32 chain (which `layers` then runs too), and counted (`fill`); errors count an arm's
own positions only.

**Streams**, as the model code computes them (checked against it):
- Qwen: the stream entering layer l is s_l = rnd(u_l) (`RMSNorm.apply`; s_0 the embeddings, image rows included), where
  u_l = m_f(l − 1) + mid(l − 1) summed in fp32 is what the input norm reads (the fused add-norm normalises the fp32 sum
  and keeps its rounding as the residual; u_0 = s_0); the mixer's output m_a (`self_attn`/`linear_attn`) from norm(u_l);
  the post-attention norm reads s_l + m_a in fp32 and keeps mid = rnd(s_l + m_a); the MLP's output m_f from that norm;
  s_{l+1} = rnd(mid + m_f) = rnd(u_{l+1}); the final hidden state H = norm(u_L), (1 + w).
- Gemma: the stream r_l is the layer's `residual` argument (r_0: the scaled embeddings, image rows unscaled), and each
  norm reads a rounded stream: r_mid = rnd(r_l + rms(m_a) w_post_attention) (`gemma_ops.sandwich`); m_f from rms(r_mid)
  w_pre_feedforward; r_{l+1} = rnd((r_mid + rms(m_f) w_post_feedforward) layer_scalar); H = rms(r_L) w. The layer
  scalar scales the whole stream, so a layer's delta r_{l+1} − r_l holds (layer_scalar − 1) r_l too.
rnd rounds once to the engine's dtype (the captures'); the fp32 reference rounds nowhere.

**Subcommands** (each writes `<name>.json` into `--out`, default `results/numerics/fp32/<UTC time>/`, and prints
its markdown tables):
- `layers`: per request, layer and arm with taps, from what the arm's norms read (its captures; Qwen's u_l from its taps
  of layer l − 1): the fp32 mixer against its m_a, the fp32 MLP from its post-attention sum (Qwen: s_l + m_a, unrounded;
  Gemma: r_mid) against its m_f, and the whole fp32 layer from its s_l (and u_l) against its s_{l+1}, relative to the
  fp32 layer's own delta from s_l. Errors are relative RMS ||x − ref|| / ||ref|| and max |x − ref| / max |ref|, over
  prompt and answer positions apart. With `core` captures, each kernel in situ: the fp32 recurrence (GDN prefill rows,
  each from the initial state the kernel read) or SDPA from the kernel's captured inputs against its captured output.
  Tables per layer and pooled per kind.
- `chain`: the fp32 model layer by layer from `--base`'s stream[0] (the engine's embeddings and image rows; without
  taps, text prompts from the table, while an image's rows need an arm with taps): H_fp32 at every position, each arm's
  hidden against it and each arm's stream[l] against the chain's (the error accumulated by layer l). Saves
  `chain/r<rid>.pt`: H_fp32, the embeddings and, for Gemma, the fp32 KV of the two layers its assistant reads.
- `drafter`: the drafter in fp32, greedy, `--k` drafts a round (the captures' k), reading each hidden source: every
  arm's hidden and H_fp32 ("fp32"). Rounds start at every p from the drafter's first (the prompt's last position, past
  the forced prefix when the final chunk takes it whole: the capture's `forced`, settings `forced_prefix`; from an arm's
  first round if earlier) while draft 1 has a reference token; a(p) = the leading drafts equal to the reference there.
  Qwen: the MTP layer over every position i < N − 1 from the next token's embedding (an image position's: its encoder
  row) and H[i], its KV from the same source, the first draft from p's output; then k − 1 chained steps, each round's on
  its own pages above p, at position p + j with RoPE p + j plus the M-RoPE offset (`runner.cuda.runner`'s drafter and
  chain); the full head. Gemma: the assistant from H[p] and the target's scaled embedding of t[p + 1] (in the engine's
  dtype), k steps at RoPE position p (p + 1 with `draft_query_next`) over the chain's fp32 KV of [0, p], for every
  source (so only the hidden state differs); its own head. Reported per source: the walk p <- p + 1 + a(p) (the
  engine's) and its tokens per verify step, the mean over every p, the acceptance per draft position; per arm, its
  in-engine rounds (`accept` at `accept_pos` − 1, each cut to the drafts the reference reaches: past it they are checked
  against the EOS), and the share of its rounds whose a equals each source's a(p) at the same p (on a CPU fp32 engine,
  its own hidden source: every round).
- `target`: per arm, the fp32 head on its hidden against on H_fp32 at the positions predicting the answer: the share
  whose argmax is H_fp32's, the share whose argmax is the reference token, the top-1 margins (Gemma's softcapped) and
  H_fp32's margins where the arm disagrees.
- `pairs`: an arm against another on the same references (by default each FP8-KV arm, `numerics_forced --kv-dtype`,
  against its BF16 twin: its name without `-kv8[all][s]` or `-kvfp8`), from the fp32 head on each one's hidden state
  alone, at the positions predicting the answer both computed: the top-1 agreement and its first miss in each answer,
  the margins, and each arm's tokens per verify step; with `--chain` also each one's argmax = H_fp32's share, their
  discordant positions (b: only the first agrees with H_fp32; c: only the second) and the difference. "free" counts the
  positions past the grammar's forced prefix only (the prefix's tokens are near-certain). The intervals resample
  requests (a cluster bootstrap, `--boot` draws, seed 0), since positions within an answer are correlated: a one-sided
  95% lower bound on the agreement, two-sided 95% intervals on the differences (a percentile bootstrap: it means
  little with few requests, and with no miss in any request the bound is the share itself). By workload (forced runs'
  run.json).
- `all`: `layers` and `chain` in one pass over the layers, then `drafter`, `target` and `report`.
- `report DIR`: one markdown summary of the results in DIR (`report.md`).
`drafter` and `target` run the chain first unless `--chain` names a chain's results; `pairs` runs no chain.

    python3 -m tests.scripts.numerics_fp32 all <run> [--rids 0-3,7] [--out DIR] [--device cuda]
    python3 -m tests.scripts.numerics_fp32 layers <run>/on-eager <run>/off-eager
    python3 -m tests.scripts.numerics_fp32 drafter <run> --chain <a chain's --out> [--k 3]
    python3 -m tests.scripts.numerics_fp32 pairs <run> [--pair on-eager:on] [--chain <a chain's --out>]
    python3 -m tests.scripts.numerics_fp32 report <out>

The function API (tests/repo/test_numerics.py drives it on the tiny models, CPU, fp32): `make_ref`, `Request` (or
`load_requests`), `sweep` (layers and chain), `drafter`, `target`, `pair`; `draft_qwen`/`draft_gemma` return the
drafts.

**Fixed differences from the engine:** the cos/sin caches in fp32 (BF16 there); Gemma's full-layer keys from the fp32
stored plane (from the BF16-rounded V there); every projection on fp32 activations (FP8 there, BF16 on PRECISE rows);
the drafter's full head (a reduced draft head there: setting `draft_vocab`). On a CPU fp32 engine whose checkpoint
has no FP8 projection the reference is the engine's own arithmetic, up to summation order.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .numerics_forced import HelpFormatter, md, parse_ids, resolve, results_dir

if TYPE_CHECKING:
    import torch
    from torch import nn

BANDS = ("prompt", "answer")
MEASURES = ("mixer", "mlp", "layer")
ROWS = 4096  # rows of a position-wise computation (an MLP) at a time
HEAD_CHUNK = 16384  # vocabulary rows of the fp32 head at a time
CAPTURE = re.compile(r"r(\d+)\.pt")
KV_TAG = re.compile(r"-kv(?:8(?:all)?s?|fp8)(?=-|$)")  # numerics_forced's FP8-KV arms (`arm_name`)
BOOT = 10_000  # cluster-bootstrap draws (`pairs`)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


@contextmanager
def reference_numerics() -> Iterator[None]:
    """The reference's process-global switches while it runs, restored after: the torch reference ops
    (`ops.enabled(False)`), every projection on unquantized activations (`fp8.bf16_activations`), no TF32, SDPA's
    math backend (plain matmuls and softmax in the inputs' dtype), inference mode."""
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from llm_mini_infer.kernels import fp8, ops

    prev = ops.enabled(), torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    ops.enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    try:
        with fp8.bf16_activations(), sdpa_kernel(SDPBackend.MATH), torch.inference_mode():
            yield
    finally:
        ops.enabled(prev[0])
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = prev[1], prev[2]


# -- the captures ------------------------------------------------------------------------------------------------------


class Request:
    """One request's captures, one per arm (`caps`: arm -> the capture dict), checked to hold the same tokens."""

    def __init__(self, rid: int, caps: dict[str, dict[str, Any]]):
        import torch

        if not caps:
            raise ValueError(f"request {rid}: no captures")
        first = next(iter(caps.values()))
        for arm, c in caps.items():
            if not torch.equal(c["tokens"], first["tokens"]) or int(c["prompt_len"]) != int(first["prompt_len"]):
                raise ValueError(f"request {rid}: arm {arm}'s tokens differ from the others' (another reference?)")
        self.rid, self.caps = rid, caps
        self.tokens = first["tokens"].long()
        self.n, self.prompt_len = len(self.tokens), int(first["prompt_len"])
        self.forced = int(first.get("forced") or 0)  # the tokens the grammar forces at the answer's start
        self.images = [(int(o), int(t)) for o, t in first.get("images") or ()]
        self.rope = None if first.get("rope") is None else first["rope"].long()
        self.settings: dict[str, Any] = dict(first.get("settings") or {})
        self.engine_dtype = first["hidden"].dtype  # the engine's: BF16 on CUDA, fp32 on a CPU runner
        # positions RoPE reaches, a drafter's chain's k aside
        self.n_pos = max(self.n, int(self.rope.max()) + 1 if self.rope is not None else 0) + 8

    def taps(self) -> dict[str, dict[str, Any]]:
        return {arm: c["taps"] for arm, c in self.caps.items() if c.get("taps")}

    def hvalid(self, arm: str) -> torch.Tensor:
        return _valid(self.caps[arm].get("hvalid"), self.n)

    def tvalid(self, arm: str) -> torch.Tensor:
        return _valid(self.caps[arm]["taps"].get("tvalid"), self.n)

    def bands(self, valid: torch.Tensor, device: torch.device) -> dict[str, torch.Tensor]:
        """Positions [0, len(valid)) where `valid`, split into the prompt's and the answer's (bool, on `device`)."""
        import torch

        pos = torch.arange(len(valid))
        return {"prompt": (valid & (pos < self.prompt_len)).to(device),
                "answer": (valid & (pos >= self.prompt_len)).to(device)}


def _valid(v: torch.Tensor | None, n: int) -> torch.Tensor:
    import torch

    return torch.ones(n, dtype=torch.bool) if v is None else v.bool()


def arm_folders(paths: list[Path]) -> dict[str, Path]:
    """Arm folders by name: each path is one (holding r<rid>.pt) or a forced run's folder of them."""

    def holds(d: Path) -> bool:
        return any(CAPTURE.fullmatch(f.name) for f in d.iterdir())

    out: dict[str, Path] = {}
    for p in paths:
        found = [p] if holds(p) else sorted(d for d in p.iterdir() if d.is_dir() and holds(d))
        if not found:
            raise FileNotFoundError(f"{p}: no captures (r<rid>.pt) in it or in its folders")
        for d in found:
            out[d.name if d.name not in out else f"{d.parent.name}/{d.name}"] = d
    return out


def workloads(paths: list[Path]) -> dict[int, str]:
    """Each request's workload, from the forced runs' run.json beside their captures."""
    out: dict[int, str] = {}
    for d in arm_folders(paths).values():
        if (d / "run.json").exists():
            for q in json.loads((d / "run.json").read_text()).get("requests", []):
                out.setdefault(int(q["rid"]), q.get("workload"))
    return out


def load_requests(paths: list[Path], rids: list[int] | None = None) -> list[Request]:
    """The captures of `paths`' arms (`arm_folders`) by request, memory-mapped (taps are large), `rids` only."""
    import torch

    files: dict[int, dict[str, Path]] = {}
    for arm, d in arm_folders(paths).items():
        for f in sorted(d.iterdir()):
            m = CAPTURE.fullmatch(f.name)
            if m and (rids is None or int(m.group(1)) in rids):
                files.setdefault(int(m.group(1)), {})[arm] = f
    if not files:
        raise FileNotFoundError(f"no captures of requests {rids} in {[str(p) for p in paths]}")
    return [Request(rid, {arm: torch.load(f, map_location="cpu", mmap=True, weights_only=True)
                          for arm, f in arms.items()}) for rid, arms in sorted(files.items())]


def round_start(req: Request) -> int:
    """The drafter's first round: from the prompt's last position, or from the forced prefix's last when the final
    prefill chunk takes it whole (the capture's `forced` with settings `forced_prefix`)."""
    forced = req.forced if req.settings.get("forced_prefix") else 0
    return req.prompt_len - 1 + min(forced, req.n - req.prompt_len)


# -- the reference model -----------------------------------------------------------------------------------------------


class Checkpoint:
    """A safetensors checkpoint's tensors under our parameter names (`mapper`: the model family's `map_weight_name`,
    fused projections by their first row), read on demand: one module's at a time."""

    def __init__(self, model_dir: Path, mapper: Callable[[str], tuple[str, int] | None], device: torch.device):
        from safetensors import safe_open

        from llm_mini_infer.models.qwen3_5.model import checkpoint_files

        index = model_dir / "model.safetensors.index.json"
        if index.exists():
            names = [(n, model_dir / f) for n, f in json.loads(index.read_text())["weight_map"].items()]
        else:
            names = []
            for f in checkpoint_files(model_dir):
                with safe_open(str(f), framework="pt") as h:
                    names += [(n, f) for n in h.keys()]  # noqa: SIM118 - safe_open is not a mapping
        self.where: dict[str, list[tuple[Path, str, int]]] = {}
        for name, f in names:
            target = mapper(name)
            if target is not None:
                self.where.setdefault(target[0], []).append((f, name, target[1]))
        self.dev = "cpu" if device.type == "cpu" else f"cuda:{device.index or 0}"

    def tensor(self, pname: str) -> torch.Tensor:
        from safetensors import safe_open

        (f, name, _), = self.where[pname]
        with safe_open(str(f), framework="pt", device=self.dev) as h:
            return h.get_tensor(name)

    def load(self, module: nn.Module, prefix: str) -> None:
        """Every parameter of `module` from the checkpoint's `prefix` + its name, shards at their rows (as the model
        families' `load_weights`)."""
        from safetensors import safe_open

        params = dict(module.named_parameters())
        rows = dict.fromkeys(params, 0)
        by_file: dict[Path, list[tuple[str, str, int]]] = {}
        for pn in params:
            for f, name, row0 in self.where.get(prefix + pn, ()):
                by_file.setdefault(f, []).append((name, pn, row0))
        for f, items in by_file.items():
            with safe_open(str(f), framework="pt", device=self.dev) as h:
                for name, pn, row0 in items:
                    t, p = h.get_tensor(name), params[pn]
                    if pn.endswith("conv_weight"):
                        t = t.reshape(t.shape[0], -1)
                    elif pn.endswith("layer_scalar"):
                        t = t.reshape(1)
                    dst = p.data.narrow(0, row0, t.shape[0]) if p.dim() else p.data
                    if dst.shape != t.shape:
                        raise ValueError(f"{name} {tuple(t.shape)} does not fit {prefix}{pn}[{row0}:]")
                    dst.copy_(t)
                    rows[pn] += t.shape[0] if t.dim() else 1
        missing = [n for n, p in params.items() if rows[n] != (p.shape[0] if p.dim() else 1)]
        if missing:
            raise ValueError(f"{prefix}: weights missing or partial in the checkpoint: {missing[:8]}")


def dequantize(module: nn.Module, dtype: torch.dtype) -> nn.Module:
    """`module` with each block-FP8 projection (`Fp8Linear`) replaced by a plain linear of its weight dequantized
    (`kernels.fp8.dequant_weight`: each e4m3 value times its block's scale, in fp32)."""
    import torch
    from torch import nn

    from llm_mini_infer.kernels import fp8
    from llm_mini_infer.models.fp8 import Fp8Linear

    for name, child in list(module.named_children()):
        if isinstance(child, Fp8Linear):
            with torch.device("meta"):
                lin = nn.Linear(child.in_features, child.out_features, bias=False)
            lin.weight = nn.Parameter(fp8.dequant_weight(child.weight, child.weight_scale).to(dtype),
                                      requires_grad=False)
            setattr(module, name, lin)
        else:
            dequantize(child, dtype)
    return module


def rowwise(fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, rows: int = ROWS) -> torch.Tensor:
    """`fn` over x's rows `rows` at a time (a position-wise computation, its temporaries bounded)."""
    import torch

    return fn(x) if x.shape[0] <= rows else torch.cat([fn(x[i:i + rows]) for i in range(0, x.shape[0], rows)])


def head_top(h: torch.Tensor, w: torch.Tensor, cap: float | None = None,
             chunk: int = HEAD_CHUNK) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The head w [V, D] (any dtype, widened a chunk at a time) on h [M, D] in h's dtype: each row's best token (the
    lowest id on a tie, as the sampler's greedy draw), its logit and the runner-up's; `cap`: Gemma's final softcap on
    both (monotone: the argmax is the uncapped one's)."""
    import torch

    m = h.shape[0]
    best = torch.full((m,), -math.inf, dtype=h.dtype, device=h.device)
    second = best.clone()
    idx = torch.zeros(m, dtype=torch.long, device=h.device)
    for c0 in range(0, w.shape[0], chunk):
        lg = h @ w[c0:c0 + chunk].to(h.dtype).T
        top = lg.topk(min(2, lg.shape[1]), dim=-1).values
        c2 = top[:, 1] if top.shape[1] > 1 else torch.full_like(best, -math.inf)
        win = top[:, 0] > best
        second = torch.maximum(torch.maximum(second, c2), torch.where(win, best, top[:, 0]))
        idx = torch.where(win, lg.argmax(-1) + c0, idx)
        best = torch.where(win, top[:, 0], best)
    if cap:
        best, second = torch.tanh(best / cap) * cap, torch.tanh(second / cap) * cap
    return idx, best, second


def recurrence(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor,
               h: torch.Tensor) -> torch.Tensor:
    """The gated delta rule token by token, every source at once (`models.qwen3_5.backend.torch_gdn_prefill`'s
    arithmetic in h's dtype): q, k [S, T, HK, K] (L2-normalised), v [S, T, HV, V], the decay alpha = exp(g) and beta [S,
    T, HV], the states h [S, HV, V, K], updated in place. Returns o [S, T, HV, V]."""
    import torch

    from llm_mini_infer.kernels import ops

    s, t, hk, dk = q.shape
    rep, scale = v.shape[2] // hk, dk ** -0.5
    out = torch.empty((s, t, *v.shape[2:]), dtype=h.dtype, device=h.device)
    for i in range(t):
        qi, ki = q[:, i].to(h.dtype), k[:, i].to(h.dtype)
        if rep > 1:
            qi, ki = qi.repeat_interleave(rep, 1), ki.repeat_interleave(rep, 1)
        out[:, i] = ops.recurrent_step(h, qi * scale, ki, v[:, i].to(h.dtype), alpha[:, i].to(h.dtype),
                                       beta[:, i].to(h.dtype))
    return out


def prefill_rows(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor,
                 pool: torch.Tensor, slots: torch.Tensor, meta: Any) -> torch.Tensor:
    """The GDN prefill kernel of the sweep (`models.qwen3_5.backend.GDN`'s `prefill`, as `torch_gdn_prefill`): its rows
    are the sources of one request, one length each, in one token loop (`recurrence`), from their slots' states, which
    take the final ones."""
    cu = meta.pf_cu_host
    rows, n = len(cu) - 1, int(cu[1] - cu[0])
    if (np.diff(cu) != n).any():
        raise ValueError("the batched recurrence takes rows of one length")
    h = pool[slots]
    o = recurrence(q.reshape(rows, n, *q.shape[1:]), k.reshape(rows, n, *k.shape[1:]),
                   v.reshape(rows, n, *v.shape[1:]), alpha.reshape(rows, n, -1), beta.reshape(rows, n, -1), h)
    pool[slots] = h
    return o.reshape(v.shape).to(v.dtype)


def qwen_rows(s: int, n: int, dev: torch.device, rope: torch.Tensor | None) -> Any:
    """`models.qwen3_5.backend.StepMeta` of s sequences of n tokens from position 0, one prefill row each (state slot 1
    + i), a page a position (row i's position j on page 1 + i n + j); RoPE from `rope` [3, >= n] when given."""
    import torch

    from llm_mini_infer.models.qwen3_5.backend import StepMeta

    pos = torch.arange(n, device=dev).repeat(s)
    page = torch.arange(s * n, device=dev) + 1
    table = torch.zeros((s + 1, n), dtype=torch.int32, device=dev)
    table[1:] = page.view(s, n)
    cu = torch.arange(s + 1, device=dev) * n
    cu_h = np.arange(s + 1) * n
    return StepMeta(num_tokens=s * n, positions=pos, kv_page=page, kv_off=torch.zeros_like(pos), n_rows=s, n_dec=0,
                    row_state=torch.arange(1, s + 1, device=dev), row_cu=cu, pf_cu=cu, pf_cu_host=cu_h,
                    row_start_host=cu_h[:-1], row_nq_host=np.full(s, n), row_kv_len_host=np.full(s, n),
                    row_slot_host=np.arange(1, s + 1), page_table=table,
                    rope=None if rope is None else rope[:, :n].to(dev).repeat(1, s))


class _Ref:
    """What `QwenRef` and `GemmaRef` share: the checkpoint read a module at a time, RoPE caches, the per-layer API the
    sweep calls (`layer`, `mixer`, `mid_engine`, `mid_fp32`, `mlp`, `out_fp32`, `final_norm`, `embeds`; with
    `fused_add`, `norm_input`)."""

    family = ""
    fused_add = False  # the input norms read the sum before the rounding the stream keeps (Qwen's: `norm_input`)
    kv_layers: tuple[int, ...] = ()  # the target layers whose KV the drafter reads (Gemma's assistant)
    softcap: float | None = None
    attn_scale = 1.0

    def __init__(self, model_dir: Path, device: str | torch.device, dtype: torch.dtype | None,
                 mapper: Callable[[str], tuple[str, int] | None]):
        import torch

        self.dir = Path(model_dir)
        self.dev = torch.device(device)
        self.dtype = dtype or torch.float32
        self.ckpt = Checkpoint(self.dir, mapper, self.dev)
        self.embed = self.ckpt.tensor("embed_tokens.weight")
        self._rot: Any = None
        self._drafter: Any = None

    def _load(self, module: nn.Module, prefix: str) -> nn.Module:
        module = module.to_empty(device=self.dev).requires_grad_(False)
        self.ckpt.load(module, prefix)
        return dequantize(module, self.dtype).eval()

    def _rotary(self, n: int) -> Any:
        raise NotImplementedError

    def rotary(self, n: int) -> Any:
        """The model's RoPE caches (fp32: `Rotary` in the reference's dtype) for positions below at least n."""
        if self._rot is None or self._rot.max_len < n:
            self._rot = self._rotary(n)
        return self._rot


class QwenRef(_Ref):
    """Qwen3.5/3.8 in fp32 (the module docstring): its decoder layers and MTP layer one at a time, the head."""

    family = "qwen"
    fused_add = True

    def __init__(self, model_dir: Path, *, device: str | torch.device = "cpu", dtype: torch.dtype | None = None):
        from llm_mini_infer.models.qwen3_5.model import LINEAR, Qwen35Config, map_weight_name

        cfg = self.cfg = Qwen35Config.from_dir(model_dir)
        super().__init__(model_dir, device, dtype, lambda n: map_weight_name(cfg, n, mtp=cfg.mtp_layers > 0))
        self.kinds = ["gdn" if t == LINEAR else "full" for t in cfg.layer_types]
        self.eps = cfg.rms_norm_eps
        self.head = self.embed if cfg.tie_word_embeddings else self.ckpt.tensor("lm_head.weight")
        self.norm_w = self.ckpt.tensor("norm.weight").to(self.dtype)
        self.attn_scale = cfg.head_dim ** -0.5

    def _rotary(self, n: int) -> Any:
        from llm_mini_infer.models.qwen3_5.model import Rotary

        return Rotary(self.cfg, n, self.dev, self.dtype)

    def layer(self, index: int) -> nn.Module:
        import torch

        from llm_mini_infer.models.qwen3_5.model import DecoderLayer

        with torch.device("meta"):
            module = DecoderLayer(self.cfg, index, 0, self.dtype)
        return self._load(module, f"layers.{index}.")

    def mtp(self) -> nn.Module:
        """The MTP layer (its attention on KV pool layer 0), loaded once."""
        import torch

        from llm_mini_infer.models.qwen3_5.model import MTP

        if not self.cfg.mtp_layers:
            raise ValueError(f"{self.dir} has no MTP layer")
        if self._drafter is None:
            with torch.device("meta"):
                module = MTP(self.cfg, 0, self.dtype)
            self._drafter = self._load(module, "mtp.")
        return self._drafter

    def embeds(self, tokens: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Text tokens' input embeddings as an engine in `dtype` computes them (the table's rows)."""
        return self.embed[tokens.to(self.dev)].to(dtype)

    def mixer(self, layer: nn.Module, s: torch.Tensor, req: Request,
              kv_row: int | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        """Layer `layer`'s mixer output [S, N, D] from what its input norm reads, s [S, N, D] (u_l: the module
        docstring; `_Sweep.mixer_in`): S sources of `req`, each its own sequence (its own pages and state slot). No KV
        is kept (`kv_row` is Gemma's)."""
        import functools

        import torch

        from llm_mini_infer.models.qwen3_5.backend import GDN, ForwardContext, Pools, TorchAttention

        cfg = self.cfg
        rows, n, d = s.shape
        z = functools.partial(torch.zeros, dtype=self.dtype, device=self.dev)
        if layer.is_attn:
            pools = Pools(kv=z(1, rows * n + 1, 2, 1, cfg.num_kv_heads, cfg.head_dim), conv=z(1), rec=z(1),
                          page_size=1)
        else:
            pools = Pools(kv=z(1), conv=z(rows + 1, 1, cfg.conv_dim, cfg.conv_kernel - 1),
                          rec=z(rows + 1, 1, cfg.lin_v_heads, cfg.lin_v_dim, cfg.lin_k_dim), page_size=1)
        ctx = ForwardContext(qwen_rows(rows, n, self.dev, req.rope), TorchAttention(pools, cfg.head_dim),
                             GDN(pools, prefill_rows), self.rotary(req.n_pos), pools)
        h = layer.input_layernorm(s.reshape(rows * n, d))
        a = layer.self_attn(h, ctx) if layer.is_attn else layer.linear_attn(h, ctx)
        return a.view(rows, n, d), None

    def norm_input(self, s: torch.Tensor, a: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """What the engine's next input norm (or its final norm) reads after a layer, from the layer's captured stream
        s, mixer output a and MLP output m: the fused add-norm's fp32 sum of m and the post-attention residual
        mid = rnd(s + a), unrounded (u_{l+1}; the next stream is its rounding)."""
        return m.float() + (s.float() + a.float()).to(s.dtype).float()

    def mid_engine(self, layer: nn.Module, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """What the engine's post-attention norm reads, from its captured stream and mixer output: their fp32 sum,
        unrounded (the fused add-norm; the residual it keeps, mid, is its rounding)."""
        return s.float() + a.float()

    def mid_fp32(self, layer: nn.Module, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return s + a

    def mlp(self, layer: nn.Module, mid: torch.Tensor) -> torch.Tensor:
        """The MLP's output from mids [M, D] (the post-attention norm first), position by position."""
        return rowwise(lambda x: layer.mlp(layer.post_attention_layernorm(x)), mid)

    def out_fp32(self, layer: nn.Module, mid: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        return mid + m

    def final_norm(self, s: torch.Tensor) -> torch.Tensor:
        from llm_mini_infer.kernels import ops

        return ops.rmsnorm(s, self.norm_w, self.eps)


class GemmaRef(_Ref):
    """Gemma 4 in fp32 (the module docstring): its decoder layers one at a time, the head, the MTP assistant."""

    family = "gemma"

    def __init__(self, model_dir: Path, assistant_dir: Path | None = None, *, device: str | torch.device = "cpu",
                 dtype: torch.dtype | None = None):
        from llm_mini_infer.models.gemma4.model import SLIDING, Gemma4Config, map_weight_name

        cfg = self.cfg = Gemma4Config.from_dir(model_dir)
        super().__init__(model_dir, device, dtype, lambda n: map_weight_name(cfg, n))
        self.assistant_dir = Path(assistant_dir) if assistant_dir else None
        self.kinds = ["sliding" if t == SLIDING else "full" for t in cfg.layer_types]
        self.kv_layers = tuple(max(ls) for ls in (cfg.sliding_layers, cfg.full_layers) if ls)
        self.eps = cfg.rms_norm_eps
        self.head = self.embed if cfg.tie_word_embeddings else self.ckpt.tensor("lm_head.weight")
        self.norm_w = self.ckpt.tensor("norm.weight").to(self.dtype)
        self.softcap = cfg.final_logit_softcapping

    def _rotary(self, n: int) -> Any:
        from llm_mini_infer.models.gemma4.model import Rotary

        return Rotary(self.cfg, n, self.dev, self.dtype)

    def layer(self, index: int) -> nn.Module:
        import torch

        from llm_mini_infer.models.gemma4.model import DecoderLayer

        with torch.device("meta"):
            module = DecoderLayer(self.cfg, index, 0, self.dtype)
        return self._load(module, f"layers.{index}.")

    def assistant(self) -> nn.Module:
        """The MTP assistant (`build_assistant`, in the reference's dtype), loaded once, its layers on KV pool layer 0
        of their kind, the full one deriving keys with the target's last full layer's k_norm."""
        from torch import nn

        from llm_mini_infer.models.gemma4.assistant import build_assistant
        from llm_mini_infer.models.gemma4.model import RMSNorm

        if self.assistant_dir is None:
            raise ValueError("Gemma's drafter needs the assistant's checkpoint (--assistant)")
        if self._drafter is None:
            asst = build_assistant(self.assistant_dir, self.cfg, self.dev, self.dtype)
            target = nn.Module()
            target.k_norm = RMSNorm(self.cfg.full_head_dim, self.eps, self.dtype).to(self.dev)
            target.k_norm.weight.copy_(self.ckpt.tensor(f"layers.{self.kv_layers[1]}.self_attn.k_norm.weight"))
            for layer in asst.layers:
                layer.self_attn.index, layer.self_attn.target = 0, [target]
            self._drafter = asst
        return self._drafter

    def embeds(self, tokens: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Text tokens' scaled input embeddings as an engine in `dtype` computes them (`Gemma4ForCausalLM.embed`:
        the row times sqrt(hidden) in `dtype`, one rounding)."""
        import torch

        e = self.embed[tokens.to(self.dev)].to(dtype)
        return e * torch.tensor(self.cfg.hidden_size ** 0.5, dtype=dtype, device=self.dev)

    def spans(self, req: Request) -> list[tuple[int, int]]:
        """The image spans the sliding layers see bidirectionally (positions [start, end)), if the checkpoint does."""
        return [(o, o + t) for o, t in req.images] if self.cfg.bidirectional == "vision" else []

    def qlim(self, req: Request, n: int) -> torch.Tensor | None:
        """Each query's last key on a sliding layer (`gemma_ops.attention_ref`'s qlim) when image spans widen it."""
        import torch

        spans = self.spans(req)
        if not spans:
            return None
        q = torch.arange(n, device=self.dev)
        lim = q.clone()
        for a, e in spans:
            lim[(q >= a) & (q < e)] = e - 1
        return lim

    def mixer(self, layer: nn.Module, s: torch.Tensor, req: Request,
              kv_row: int | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        """Layer `layer`'s attention output [S, N, D] from streams s [S, N, D] (S sources of `req`, each its own
        sequence and pages), and with `kv_row` the KV that source's row wrote (fp32 [N, heads, dim], on the CPU)."""
        import functools

        import torch

        from llm_mini_infer.models.gemma4.backend import (
            Gemma4Context,
            Gemma4Meta,
            Gemma4Pools,
            TorchFullAttention,
            TorchSlidingAttention,
        )
        from llm_mini_infer.models.gemma4.model import SLIDING

        cfg, dev = self.cfg, self.dev
        rows, n, d = s.shape
        att = layer.self_attn
        sliding = att.kind == SLIDING
        z = functools.partial(torch.zeros, dtype=self.dtype, device=dev)
        pages, hkv, hd = rows * n + 1, cfg.kv_heads(att.kind), cfg.head_dim(att.kind)
        wkv = z(1, pages, 2, 1, hkv, hd) if sliding else z(1, 1, 2, 1, 1, 1)
        fkv = z(1, 1, 1, 1, 1) if sliding else z(1, pages, 1, hkv, hd) if cfg.k_eq_v else z(1, pages, 2, 1, hkv, hd)
        pools = Gemma4Pools(wkv=wkv, fkv=fkv, page_size=1, window=cfg.sliding_window)
        pos = torch.arange(n, device=dev).repeat(rows)
        page = torch.arange(rows * n, device=dev) + 1
        table = page.view(rows, n).int()
        spans = self.spans(req)
        sp = dict(span_rows_host=np.repeat(np.arange(rows), len(spans)),
                  span_start_host=np.tile([a for a, _ in spans], rows),
                  span_end_host=np.tile([e for _, e in spans], rows)) if spans else {}
        meta = Gemma4Meta(num_tokens=rows * n, positions=pos, f_page=page, f_off=pos * 0, w_page=page, w_off=pos * 0,
                          n_rows=rows, row_start_host=np.arange(rows) * n, row_nq_host=np.full(rows, n),
                          row_kv_len_host=np.full(rows, n), row_slot_host=np.arange(rows),
                          row_wlo_host=np.zeros(rows, np.int64), page_table=table, wpage_table=table, **sp)
        rot = self.rotary(req.n_pos)
        ctx = Gemma4Context(meta, TorchSlidingAttention(pools), TorchFullAttention(pools, rot, cfg.k_eq_v), rot, pools)
        a = att(layer.input_layernorm(s.reshape(rows * n, d)), ctx).view(rows, n, d)
        if kv_row is None:
            return a, None
        pg = 1 + kv_row * n + torch.arange(n, device=dev)
        if sliding:
            return a, {"sliding_k": wkv[0, pg, 0, 0].cpu(), "sliding_v": wkv[0, pg, 1, 0].cpu()}
        if cfg.k_eq_v:
            return a, {"full_v": fkv[0, pg, 0].cpu()}
        return a, {"full_k": fkv[0, pg, 0, 0].cpu(), "full_v": fkv[0, pg, 1, 0].cpu()}

    def mid_engine(self, layer: nn.Module, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """r_mid from a stream and an attention output (`sandwich`: rounded once to their dtype), which the
        pre-feedforward norm reads: the engine's from its captures, the fp32 one in the reference's dtype."""
        from llm_mini_infer.kernels import gemma_ops as gops

        return gops.sandwich(a, s, layer.post_attention_layernorm.weight, layer.pre_feedforward_layernorm.weight,
                             layer.eps)[0]

    mid_fp32 = mid_engine

    def mlp(self, layer: nn.Module, mid: torch.Tensor) -> torch.Tensor:
        """The MLP's output from mids [M, D] (the pre-feedforward norm first), position by position."""
        from llm_mini_infer.kernels import gemma_ops as gops

        return rowwise(lambda x: layer.mlp(gops.rmsnorm(x, layer.pre_feedforward_layernorm.weight, layer.eps)), mid)

    def out_fp32(self, layer: nn.Module, mid: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        from llm_mini_infer.kernels import gemma_ops as gops

        w = layer.post_feedforward_layernorm.weight
        return gops.sandwich(m, mid, w, w, layer.eps, scalar=layer.layer_scalar)[0]

    def final_norm(self, s: torch.Tensor) -> torch.Tensor:
        from llm_mini_infer.kernels import gemma_ops as gops

        return gops.rmsnorm(s, self.norm_w, self.eps)


def make_ref(model_dir: Path, assistant_dir: Path | None = None, *, device: str | torch.device = "cpu",
             dtype: torch.dtype | None = None) -> QwenRef | GemmaRef:
    """The fp32 reference of a checkpoint folder (Gemma 4 with its assistant's folder, or Qwen3.5/3.8)."""
    from llm_mini_infer.models.gemma4.model import is_gemma4

    if is_gemma4(model_dir):
        return GemmaRef(model_dir, assistant_dir, device=device, dtype=dtype)
    return QwenRef(model_dir, device=device, dtype=dtype)


# -- errors ------------------------------------------------------------------------------------------------------------


def errors(x: torch.Tensor, ref: torch.Tensor, bands: dict[str, torch.Tensor]) -> dict[str, dict[str, float]]:
    """x against ref ([N, ...] each) over each band's positions: the sums the tables pool (`pooled`)."""
    d = x.to(ref.dtype) - ref
    out = {}
    for name, m in bands.items():
        dm, rm, n = d[m], ref[m], int(m.sum())
        out[name] = {"n": n, "d2": float(dm.double().square().sum()), "r2": float(rm.double().square().sum()),
                     "dmax": float(dm.abs().max()) if n else 0.0, "rmax": float(rm.abs().max()) if n else 0.0}
    return out


def pooled(items: list[dict[str, float]]) -> dict[str, float]:
    tot = {"n": 0, "d2": 0.0, "r2": 0.0, "dmax": 0.0, "rmax": 0.0}
    for e in items:
        for key in ("n", "d2", "r2"):
            tot[key] += e[key]
        for key in ("dmax", "rmax"):
            tot[key] = max(tot[key], e[key])
    return tot


def rel(e: dict[str, float]) -> str:
    """Relative RMS error ||x - ref|| / ||ref||."""
    return f"{math.sqrt(e['d2'] / e['r2']):.2e}" if e["n"] and e["r2"] else "-"


def rel_max(e: dict[str, float]) -> str:
    """max |x - ref| / max |ref|."""
    return f"{e['dmax'] / e['rmax']:.2e}" if e["n"] and e["rmax"] else "-"


def core_gdn(c: dict[str, Any], req: Request, dev: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    """A GDN prefill kernel in situ: the fp32 recurrence from its captured inputs (q, k, v, alpha, beta, rows
    concatenated in `rows`' order: (first position, tokens) each) against its captured output o. Each row starts from
    its captured initial state (`init` [rows, HV, V, K], the state the kernel read); without `init`, rows from position
    0 start from zero and a row continuing the last one from its fp32 state, and a row starting elsewhere is
    skipped."""
    import torch

    rows = [(int(a), int(b)) for a, b in c["rows"]]
    offs = np.concatenate([[0], np.cumsum([b for _, b in rows])]).astype(int)
    init = c.get("init")
    sums: dict[str, list] = {b: [] for b in BANDS}
    h, end, used, skipped = None, -1, 0, 0
    for i in sorted(range(len(rows)), key=lambda i: rows[i][0]):
        pos0, n = rows[i]
        if init is not None:
            h = init[i:i + 1].to(dev, dtype, copy=True)  # the recurrence updates it in place
        elif pos0 == 0:
            h = torch.zeros((1, c["v"].shape[1], c["v"].shape[2], c["q"].shape[2]), dtype=dtype, device=dev)
        elif h is None or pos0 != end:
            h, skipped = None, skipped + 1
            continue
        sl = slice(offs[i], offs[i] + n)
        q, k, v, alpha, beta = (c[name][sl].to(dev)[None] for name in ("q", "k", "v", "alpha", "beta"))
        o = recurrence(q, k, v, alpha, beta, h)[0]
        pos = torch.arange(pos0, pos0 + n)
        e = errors(c["o"][sl].to(dev).reshape(n, -1), o.reshape(n, -1),
                   {"prompt": (pos < req.prompt_len).to(dev), "answer": (pos >= req.prompt_len).to(dev)})
        for band in BANDS:
            sums[band].append(e[band])
        end, used = pos0 + n, used + 1
    return {**{band: pooled(sums[band]) for band in BANDS}, "rows": used, "skipped": skipped}


def core_attention(ref: QwenRef | GemmaRef, layer: nn.Module, kind: str, c: dict[str, Any],
                   req: Request) -> dict[str, Any]:
    """An attention kernel in situ: fp32 SDPA (the model's scale, causal; on Gemma's sliding layers its window and
    image spans) from its captured q, k, v by position against its captured output, at positions whose inputs and
    earlier keys are all valid (`valid`). Without k (Gemma's full layers store V alone), the keys come from the stored
    plane (`gemma_ops.k_from_v`, fp32)."""
    import torch

    from llm_mini_infer.kernels import gemma_ops as gops

    dev, dt = ref.dev, ref.dtype
    q, v = c["q"].to(dev, dt), c["v"].to(dev, dt)
    n = q.shape[0]
    pos = torch.arange(n, device=dev)
    if c.get("k") is not None:
        k = c["k"].to(dev, dt)
    else:
        from llm_mini_infer.models.gemma4.model import FULL

        rot = ref.rotary(req.n_pos)
        k = gops.k_from_v(v, layer.self_attn.k_norm.weight, rot.cos[FULL][pos], rot.sin[FULL][pos])
    valid = _valid(c.get("valid"), n)
    ok = valid & (torch.cumprod(valid.long(), 0) > 0)
    sliding = kind == "sliding"
    o = gops.attention_ref(q * ref.attn_scale, k, v, pos, pos, window=ref.cfg.sliding_window if sliding else 0,
                           qlim=ref.qlim(req, n) if sliding else None)
    return {**errors(c["o"].to(dev).reshape(n, -1), o.reshape(n, -1), req.bands(ok, dev)), "rows": 0, "skipped": 0}


# -- the sweep: layers and the chain -----------------------------------------------------------------------------------


@dataclass
class Chain:
    """The fp32 chain of one request: H_fp32 [N, D], the embeddings it started from (the engine's dtype) and where
    they came from (`base`), and for Gemma the KV of the layers the assistant reads (fp32 [N, heads, dim]: the last
    sliding layer's `sliding_k`/`sliding_v`, the last full layer's stored plane `full_v`, and `full_k` without
    k_eq_v)."""

    rid: int
    base: str
    hidden: torch.Tensor
    embeds: torch.Tensor
    kv: dict[str, torch.Tensor]

    def save(self, path: Path) -> None:
        import torch

        torch.save({"rid": self.rid, "base": self.base, "hidden": self.hidden.cpu(), "embeds": self.embeds.cpu(),
                    "kv": {k: v.cpu() for k, v in self.kv.items()}}, path)

    @classmethod
    def load(cls, path: Path) -> Chain:
        import torch

        d = torch.load(path, map_location="cpu", weights_only=True)
        return cls(d["rid"], d["base"], d["hidden"], d["embeds"], d["kv"])


@dataclass
class Swept:
    result: dict[str, Any]  # JSON: fill, layers, core (layers); base, stream, hidden (chain)
    chain: Chain | None


def stream0(ref: QwenRef | GemmaRef, req: Request, base: str = "on") -> tuple[str, torch.Tensor]:
    """The embeddings the chain starts from, in the engine's dtype on the device, and their source: `base`'s
    stream[0] (the arm of that name, else the first of that switch: `on` takes `on-eager`; else the first arm with
    taps), a position it lacks from another arm's, then from the table (text tokens: `embeds`; an image position no arm
    holds is an error). Without taps: the table ("table")."""
    import torch

    taps = req.taps()
    order = sorted(taps, key=lambda a: (a != base, a.split("-")[0] != base))
    x = ref.embeds(req.tokens, req.engine_dtype)
    need = torch.ones(req.n, dtype=torch.bool)
    for arm in order:
        take = need & req.tvalid(arm)
        x[take.to(ref.dev)] = taps[arm]["stream"][0][take].to(ref.dev, x.dtype)
        need &= ~take
    img = torch.zeros(req.n, dtype=torch.bool)
    for o, t in req.images:
        img[o:o + t] = True
    if (need & img).any():
        raise ValueError(f"request {req.rid}: no arm's taps hold its image rows (a forced run with --taps)")
    return (order[0] if order else "table"), x


class _Sweep:
    """One request's part of `sweep`: its arms' taps, the chain's stream, the results so far."""

    def __init__(self, ref: QwenRef | GemmaRef, req: Request, *, layers: bool, chain: bool, base: str):
        self.ref, self.req, self.dev, self.dt = ref, req, ref.dev, ref.dtype
        self.taps = req.taps()
        for arm, t in self.taps.items():
            if list(t["kinds"]) != ref.kinds:
                raise ValueError(f"request {req.rid}: arm {arm}'s taps are of layers {list(t['kinds'])}, not "
                                 f"{ref.kinds}")
        tvalid = {arm: req.tvalid(arm) for arm in self.taps}
        self.bands = {arm: req.bands(v, self.dev) for arm, v in tvalid.items()}
        self.layers = layers and bool(self.taps)
        self.res: dict[str, Any] = {"rid": req.rid, "n": req.n, "prompt_len": req.prompt_len}
        # each arm's positions to take from another arm's taps (in order), the rest from the chain's stream
        self.fill: dict[str, tuple[list[tuple[str, torch.Tensor]], torch.Tensor]] = {}
        for arm in self.taps:
            need, plan = ~tvalid[arm], []
            for other in self.taps:
                take = need & tvalid[other]
                if other != arm and take.any():
                    plan.append((other, take.nonzero()[:, 0]))
                    need &= ~take
            self.fill[arm] = (plan, need.nonzero()[:, 0])
        self.res["fill"] = {arm: {"from_arms": sum(len(i) for _, i in plan), "rest": len(rest)}
                            for arm, (plan, rest) in self.fill.items()}
        self.s: torch.Tensor | None = None
        if chain or (self.layers and any(len(rest) for _, rest in self.fill.values())):  # the chain fills the rest
            self.base, self.s0 = stream0(ref, req, base)
            self.s = self.s0.to(self.dt)
            self.kv: dict[str, torch.Tensor] = {}
            self.res.update(base=self.base, stream=[self.against_chain(0)])
        if self.layers:
            self.res.update(layers=[], core=[])

    def tap(self, arm: str, name: str, index: int, idx: torch.Tensor | None = None) -> torch.Tensor:
        """The arm's tap `name` of layer `index` on the device, at positions `idx` (None: all)."""
        t = self.taps[arm][name][index]
        return (t if idx is None else t[idx]).to(self.dev)

    def _filled(self, arm: str, rows: Callable[[str, torch.Tensor | None], torch.Tensor]) -> torch.Tensor:
        """`rows(a, idx)` (arm a's positions idx, None: all, in the reference's dtype on the device) for the arm, the
        positions it lacks from the arms its `fill` names, the rest from the chain's stream."""
        x = rows(arm, None)
        plan, rest = self.fill[arm]
        for other, idx in plan:
            x[idx.to(self.dev)] = rows(other, idx)
        if len(rest) and self.s is not None:
            x[rest.to(self.dev)] = self.s[rest.to(self.dev)]
        return x

    def stream_in(self, arm: str, index: int) -> torch.Tensor:
        """The arm's stream entering layer `index`, its invalid positions filled (`fill`)."""
        return self._filled(arm, lambda a, idx: self.tap(a, "stream", index, idx).to(self.dt))

    def mixer_in(self, arm: str, index: int) -> torch.Tensor:
        """What the arm's input norm of layer `index` read, filled alike: with `fused_add` past layer 0 the unrounded
        sum (`norm_input` of the arm's taps of the layer before: u_l), else its stream."""
        if not (index and self.ref.fused_add):
            return self.stream_in(arm, index)
        return self._filled(arm, lambda a, idx: self.ref.norm_input(
            *(self.tap(a, name, index - 1, idx) for name in ("stream", "mixer", "mlp"))).to(self.dt))

    def against_chain(self, index: int) -> dict[str, Any]:
        return {arm: errors(self.tap(arm, "stream", index), self.s, self.bands[arm]) for arm in self.taps}

    def step(self, index: int, kind: str, layer: nn.Module) -> None:
        """Layer `index` for every source of the request: the arms' own inputs (`layers`: their streams, and what
        their input norms read) and the chain's."""
        import torch

        ref, n = self.ref, self.req.n
        arms = list(self.taps) if self.layers else []
        srcs = [self.stream_in(arm, index) for arm in arms] + ([self.s] if self.s is not None else [])
        if not srcs:
            return
        s = torch.stack(srcs)
        x = torch.stack([*(self.mixer_in(arm, index) for arm in arms), *srcs[len(arms):]]) \
            if arms and index and ref.fused_add else s
        a, kv = ref.mixer(layer, x, self.req, len(srcs) - 1 if self.s is not None and index in ref.kv_layers else None)
        del x
        if kv:
            self.kv.update(kv)
        mid = ref.mid_fp32(layer, s, a)
        d = s.shape[-1]
        mids = [ref.mid_engine(layer, self.tap(arm, "stream", index), self.tap(arm, "mixer", index)).to(self.dt)
                for arm in arms]
        m = ref.mlp(layer, torch.cat([*mids, mid.reshape(-1, d)]))
        m_x, m_f = m[:len(arms) * n].view(len(arms), n, d), m[len(arms) * n:].view(len(srcs), n, d)
        out = ref.out_fp32(layer, mid, m_f)
        if arms:
            row: dict[str, Any] = {"index": index, "kind": kind, **{meas: {} for meas in MEASURES}}
            for i, arm in enumerate(arms):
                b = self.bands[arm]
                row["mixer"][arm] = errors(self.tap(arm, "mixer", index), a[i], b)
                row["mlp"][arm] = errors(self.tap(arm, "mlp", index), m_x[i], b)
                row["layer"][arm] = errors(self.tap(arm, "stream", index + 1).to(self.dt) - srcs[i], out[i] - srcs[i],
                                           b)
            self.res["layers"].append(row)
            self.core(index, kind, layer)
        if self.s is not None:
            self.s = out[-1]
            self.res["stream"].append(self.against_chain(index + 1))

    def core(self, index: int, kind: str, layer: nn.Module) -> None:
        for arm, t in self.taps.items():
            cores = t.get("core") or {}
            c = cores.get(index, cores.get(str(index)))
            if c is not None:
                e = core_gdn(c, self.req, self.dev, self.dt) if kind == "gdn" else \
                    core_attention(self.ref, layer, kind, c, self.req)
                self.res["core"].append({"index": index, "kind": kind, "arm": arm, **e})

    def finish(self) -> Swept:
        if self.s is None:
            return Swept(self.res, None)
        req = self.req
        h = self.ref.final_norm(self.s)
        self.res["hidden"] = {arm: errors(c["hidden"].to(self.dev), h, req.bands(req.hvalid(arm), self.dev))
                              for arm, c in req.caps.items()}
        return Swept(self.res, Chain(req.rid, self.base, h, self.s0, self.kv))


def sweep(ref: QwenRef | GemmaRef, requests: list[Request], *, layers: bool = True, chain: bool = True,
          base: str = "on", say: Callable[[str], None] = log) -> dict[int, Swept]:
    """One pass over the checkpoint's layers, each loaded once for every request: `layers` (every arm with taps, from
    its own captured inputs) and the fp32 `chain` from `base`'s embeddings (the module docstring)."""
    import torch

    with reference_numerics():
        parts = [_Sweep(ref, req, layers=layers, chain=chain, base=base) for req in requests]
        for index, kind in enumerate(ref.kinds):
            t0 = time.perf_counter()
            layer = ref.layer(index)
            for part in parts:
                part.step(index, kind, layer)
            del layer
            mem = f", {torch.cuda.max_memory_allocated(ref.dev) / 2**30:.1f} GiB peak" if ref.dev.type == "cuda" else ""
            say(f"layer {index + 1}/{len(ref.kinds)} ({kind}): {time.perf_counter() - t0:.1f} s{mem}")
        return {part.req.rid: part.finish() for part in parts}


# -- the drafter -------------------------------------------------------------------------------------------------------


def draft_qwen(ref: QwenRef, req: Request, hidden: torch.Tensor, embeds: torch.Tensor, p: torch.Tensor,
               k: int) -> torch.Tensor:
    """Qwen's MTP drafts [S, R, k] for the rounds starting at p [R], from each source's hidden states [S, N, D]
    (the module docstring): the MTP layer over every position i < N - 1 from embeds[i + 1] (its KV on the source's
    pages), round p's first draft from its output at p; then k - 1 chained steps, each (source, round)'s on its own
    pages above p (the main pass's KV below), as the runner's drafter and chain."""
    import torch

    from llm_mini_infer.models.qwen3_5.backend import ForwardContext, Pools, StepMeta, TorchAttention

    cfg, dev = ref.cfg, ref.dev
    s, n, d = hidden.shape[0], req.n - 1, hidden.shape[2]
    r = len(p)
    mtp = ref.mtp()
    rot = ref.rotary(req.n_pos + k)
    priv0 = 1 + s * n  # the chains' pages after the main pass's
    kv = torch.zeros((1, priv0 + s * r * (k - 1), 2, 1, cfg.num_kv_heads, cfg.head_dim), dtype=ref.dtype, device=dev)
    pools = Pools(kv=kv, conv=kv[:0], rec=kv[:0], page_size=1)
    attn = TorchAttention(pools, cfg.head_dim)
    meta = qwen_rows(s, n, dev, req.rope)
    rr, pr = torch.arange(s * r, device=dev), p.repeat(s)  # chain row (source-major) and its round's p
    col, at = torch.arange(req.n + k, device=dev), pr[:, None]
    table = torch.zeros((1 + s + s * r, req.n + k), dtype=torch.int32, device=dev)
    table[1:1 + s, :n] = meta.page_table[1:]
    table[1 + s:] = torch.where(col <= at, 1 + (rr // r)[:, None] * n + col,
                                torch.where((col > at) & (col < at + k), priv0 + rr[:, None] * (k - 1) + col - at - 1,
                                            0))
    meta.page_table = table
    x = mtp(embeds[1:].repeat(s, 1), hidden[:, :n].reshape(s * n, d), ForwardContext(meta, attn, None, rot, pools))
    h = x.view(s, n, d)[:, p].reshape(s * r, d)
    drafts = [head_top(h, ref.head)[0]]
    off = int(req.rope[0, -1]) - (req.n - 1) if req.rope is not None else 0  # the M-RoPE offset past the last image
    cu, host = torch.arange(s * r + 1, device=dev), np.arange(s * r)
    for j in range(1, k):
        pj = pr + j
        cm = StepMeta(num_tokens=s * r, positions=pj, kv_page=priv0 + rr * (k - 1) + j - 1, kv_off=torch.zeros_like(pj),
                      n_rows=s * r, n_dec=0, row_state=rr, row_cu=cu, pf_cu=cu, pf_cu_host=np.arange(s * r + 1),
                      row_start_host=host, row_nq_host=np.ones(s * r, np.int64),
                      row_kv_len_host=(pj + 1).cpu().numpy(), row_slot_host=1 + s + host, page_table=table,
                      rope=None if req.rope is None else pj + off)
        h = mtp(ref.embeds(drafts[-1], ref.dtype), h, ForwardContext(cm, attn, None, rot, pools))
        drafts.append(head_top(h, ref.head)[0])
    return torch.stack(drafts, -1).view(s, r, k)


def draft_gemma(ref: GemmaRef, req: Request, hidden: torch.Tensor, kv: dict[str, torch.Tensor], p: torch.Tensor,
                k: int, query_next: bool = False) -> torch.Tensor:
    """Gemma's assistant drafts [S, R, k] for the rounds starting at p [R], from each source's hidden states [S, N, D]
    and the target's KV `kv` (`Chain.kv`, read as the target's last sliding and full layers'): k steps at RoPE position
    p (p + 1 with `query_next`) over the KV of [0, p], the first from H[p] and t[p + 1], each next from the previous
    draft and backbone hidden state, as the runner's `_draft`."""
    import torch

    from llm_mini_infer.models.gemma4.backend import (
        Gemma4Context,
        Gemma4Meta,
        Gemma4Pools,
        TorchFullAttention,
        TorchSlidingAttention,
    )

    cfg, dev, dt = ref.cfg, ref.dev, ref.dtype
    s, n, d = hidden.shape
    m = s * len(p)
    asst = ref.assistant()
    sk = kv["sliding_k"].to(dev, dt)
    wkv = torch.zeros((1, n + 1, 2, 1, *sk.shape[1:]), dtype=dt, device=dev)
    wkv[0, 1:, 0, 0], wkv[0, 1:, 1, 0] = sk, kv["sliding_v"].to(dev, dt)
    fv = kv["full_v"].to(dev, dt)
    if cfg.k_eq_v:
        fkv = torch.zeros((1, n + 1, 1, *fv.shape[1:]), dtype=dt, device=dev)
        fkv[0, 1:, 0] = fv
    else:
        fkv = torch.zeros((1, n + 1, 2, 1, *fv.shape[1:]), dtype=dt, device=dev)
        fkv[0, 1:, 0, 0], fkv[0, 1:, 1, 0] = kv["full_k"].to(dev, dt), fv
    pools = Gemma4Pools(wkv=wkv, fkv=fkv, page_size=1, window=cfg.sliding_window)
    rot = ref.rotary(req.n_pos + k)
    table = torch.arange(1, n + 1, dtype=torch.int32, device=dev)[None]
    pr = p.repeat(s)
    meta = Gemma4Meta(num_tokens=m, positions=pr + int(query_next), f_page=None, f_off=None, w_page=None, w_off=None,
                      n_rows=m, row_start_host=np.arange(m), row_nq_host=np.ones(m, np.int64),
                      row_kv_len_host=(pr + 1).cpu().numpy(), row_slot_host=np.zeros(m, np.int64),
                      row_wlo_host=np.zeros(m, np.int64), page_table=table, wpage_table=table)
    ctx = Gemma4Context(meta, TorchSlidingAttention(pools), TorchFullAttention(pools, rot, cfg.k_eq_v), rot, pools)

    def emb(tok: torch.Tensor) -> torch.Tensor:  # the target's scaled embedding, as the engine rounds it
        return ref.embeds(tok, req.engine_dtype).to(dt)

    x, hb = asst(emb(req.tokens.to(dev)[pr + 1]), hidden[:, p].reshape(m, d), ctx)
    drafts = [head_top(x, asst.head_weight)[0]]
    for _ in range(1, k):
        x, hb = asst(emb(drafts[-1]), hb, ctx)
        drafts.append(head_top(x, asst.head_weight)[0])
    return torch.stack(drafts, -1).view(s, len(p), k)


def accepted(drafts: torch.Tensor, tokens: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """a(p) per source [S, R]: the leading drafts (draft j fills p + 2 + j) equal to the reference there, among those
    the reference reaches."""
    import torch

    pos = p[:, None] + 2 + torch.arange(drafts.shape[-1], device=p.device)
    ok = (drafts == tokens[pos.clamp(max=len(tokens) - 1)]) & (pos < len(tokens))
    return ok.long().cumprod(-1).sum(-1)


def walk(a: list[int], p0: int, start: int, n: int) -> list[tuple[int, int]]:
    """The rounds the engine takes over a(p) (`a[p - p0]`): p <- p + 1 + a(p) from `start`, while draft 1 has a
    reference token (p + 2 < n)."""
    out, p = [], start
    while p + 2 < n:
        out.append((p, a[p - p0]))
        p += 1 + a[p - p0]
    return out


def engine_rounds(c: dict[str, Any], n: int) -> list[tuple[int, int]]:
    """An arm's in-engine rounds (p, a): each verify window's accepted drafts (`accept`) at the round its drafter ran,
    p = the window's first position − 1 (`accept_pos`; the hidden state the drafter read), a cut to the drafts the
    reference reaches (n − p − 2: a draft past it is checked against the EOS)."""
    return [(q - 1, min(x, n - q - 1)) for q, x in zip(c["accept_pos"].tolist(), c["accept"].tolist()) if q + 1 < n]


def hidden_sources(ref: QwenRef | GemmaRef, req: Request, chain: Chain | None,
                   need: torch.Tensor) -> tuple[list[str], torch.Tensor, dict[str, int]]:
    """Each arm's final hidden states and H_fp32 ("fp32") as [S, N, D] on the device, a position of `need` (bool
    [N]) an arm lacks from H_fp32, else from another arm's; returns (the names, the states, positions filled by arm)."""
    import torch

    names, hs, fills = [], [], {}
    valid = {arm: req.hvalid(arm) for arm in req.caps}
    for arm, c in req.caps.items():
        h = c["hidden"].to(ref.dev, ref.dtype)
        miss = need & ~valid[arm]
        fills[arm] = int(miss.sum())
        if chain is not None:
            h[miss.to(ref.dev)] = chain.hidden.to(ref.dev, ref.dtype)[miss.to(ref.dev)]
        else:
            for other, oc in req.caps.items():
                take = miss & valid[other]
                h[take.to(ref.dev)] = oc["hidden"][take].to(ref.dev, ref.dtype)
                miss &= ~take
            if miss.any():
                raise ValueError(f"request {req.rid}: no arm holds {arm}'s hidden state at {int(miss.sum())} "
                                 "positions the drafter reads (run the chain)")
        names.append(arm)
        hs.append(h)
    if chain is not None:
        names.append("fp32")
        hs.append(chain.hidden.to(ref.dev, ref.dtype))
    return names, torch.stack(hs), fills


def drafter(ref: QwenRef | GemmaRef, req: Request, chain: Chain | None = None, *, k: int | None = None,
            base: str = "on") -> dict[str, Any]:
    """The drafter in fp32 on every hidden source (the module docstring): a(p) for every round start p from `p0` (the
    walk's `start`, or an arm's first round if earlier), and each arm's in-engine rounds (`engine_rounds`) with the
    share of them each source matches. `chain`: H_fp32 (a source), the embeddings and Gemma's KV; Qwen without it
    reads the embeddings from the taps or the table (`stream0`)."""
    import torch

    with reference_numerics():
        k = int(k or req.settings.get("k", 3))
        start, n = round_start(req), req.n
        rounds = {arm: engine_rounds(c, n) for arm, c in req.caps.items() if c.get("accept_pos") is not None}
        p0 = min([start, *(r[0][0] for r in rounds.values() if r)])  # a forced prefix taken in part starts earlier
        p = torch.arange(p0, max(p0, n - 2), device=ref.dev)  # rounds whose first draft the reference reaches
        need = torch.zeros(n, dtype=torch.bool)
        if ref.family == "qwen":
            need[:n - 1] = True
        else:
            need[p.cpu()] = True
        names, hidden, fills = hidden_sources(ref, req, chain, need)
        if not len(p):
            drafts = torch.zeros((len(names), 0, k), dtype=torch.long, device=ref.dev)
        elif isinstance(ref, QwenRef):
            embeds = chain.embeds if chain is not None else stream0(ref, req, base)[1]
            drafts = draft_qwen(ref, req, hidden, embeds.to(ref.dev, ref.dtype), p, k)
        else:
            if chain is None or "full_v" not in chain.kv:
                raise ValueError("Gemma's drafter reads the fp32 chain's KV: run the chain")
            drafts = draft_gemma(ref, req, hidden, chain.kv, p, k, bool(req.settings.get("draft_query_next")))
        a = accepted(drafts, req.tokens.to(ref.dev), p).cpu().numpy()
        res: dict[str, Any] = {"rid": req.rid, "n": n, "prompt_len": req.prompt_len, "p0": p0, "start": start, "k": k,
                               "draft_vocab": int(req.settings.get("draft_vocab", 0) or 0), "fill": fills,
                               "sources": {name: a[i].tolist() for i, name in enumerate(names)}, "engine": {}}
        for arm, rs in rounds.items():
            res["engine"][arm] = {"accept": req.caps[arm]["accept"].tolist(), "rounds": rs,
                                  "same": {name: sum(int(a[i][q - p0] == x) for q, x in rs)
                                           for i, name in enumerate(names)}}
        return res


# -- the target --------------------------------------------------------------------------------------------------------


def target(ref: QwenRef | GemmaRef, req: Request, chain: Chain) -> dict[str, Any]:
    """The fp32 head on each arm's hidden against on H_fp32, at the positions whose next token is the answer's: the
    argmax agreements, the reference's share and the top-1 margins (the module docstring)."""
    import torch

    def margins(x: torch.Tensor) -> list[float]:
        return np.round(x.double().cpu().numpy(), 5).tolist()

    with reference_numerics():
        dev, dt = ref.dev, ref.dtype
        pos = torch.arange(req.prompt_len - 1, req.n - 1)
        nxt = req.tokens[pos + 1].to(dev)
        fi, f1, f2 = head_top(chain.hidden.to(dev, dt)[pos.to(dev)], ref.head, ref.softcap)
        res: dict[str, Any] = {"rid": req.rid, "n": req.n, "prompt_len": req.prompt_len, "arms": {},
                               "fp32": {"n": len(pos), "ref": int((fi == nxt).sum()), "margins": margins(f1 - f2)}}
        for arm, c in req.caps.items():
            v = req.hvalid(arm)[pos]
            if not v.any():
                continue
            vd = v.to(dev)
            xi, x1, x2 = head_top(c["hidden"][pos[v]].to(dev, dt), ref.head, ref.softcap)
            dis = xi != fi[vd]
            res["arms"][arm] = {"n": int(v.sum()), "agree": int((~dis).sum()), "ref": int((xi == nxt[vd]).sum()),
                                "margins": margins(x1 - x2), "dis_fp32_margins": margins((f1 - f2)[vd][dis])}
        return res


# -- arm against arm ---------------------------------------------------------------------------------------------------


def twin(arm: str) -> str | None:
    """An FP8-KV arm's BF16 twin, its name without the KV tag (`on-kv8-cache` -> `on-cache`); None for another arm."""
    t = KV_TAG.sub("", arm, count=1)
    return t if t != arm else None


def default_pairs(arms: list[str]) -> list[tuple[str, str]]:
    """Each FP8-KV arm with its twin, where the twin is among `arms`."""
    return [(a, t) for a in arms if (t := twin(a)) in arms]


def pair(ref: QwenRef | GemmaRef, req: Request, a: str, b: str, chain: Chain | None = None) -> dict[str, Any] | None:
    """Arm `a` against arm `b` on one request (`pairs`, the module docstring), at the positions predicting the answer
    that both computed (None: none): per band ("all"; "free": past the forced prefix) the positions, the top-1
    agreements and with `chain` each arm's argmax = H_fp32's (fp32_a, fp32_b) and the discordant counts (b, c); the
    first disagreement among the free positions (its index in the answer; None: none); a's top-1 margins and b's where
    they disagree (free positions); each arm's verify steps and accepted drafts (its capture's `accept`)."""
    import torch

    def r5(x: torch.Tensor) -> list[float]:
        return np.round(x.double().cpu().numpy(), 5).tolist()

    with reference_numerics():
        dev, dt = ref.dev, ref.dtype
        pos = torch.arange(req.prompt_len - 1, req.n - 1)
        pos = pos[req.hvalid(a)[pos] & req.hvalid(b)[pos]]
        if not len(pos):
            return None
        ai, a1, a2 = head_top(req.caps[a]["hidden"][pos].to(dev, dt), ref.head, ref.softcap)
        bi, b1, b2 = head_top(req.caps[b]["hidden"][pos].to(dev, dt), ref.head, ref.softcap)
        free = (pos >= req.prompt_len - 1 + req.forced).to(dev)
        same = ai == bi
        fi = head_top(chain.hidden.to(dev, dt)[pos.to(dev)], ref.head, ref.softcap)[0] if chain is not None else None
        bands: dict[str, Any] = {}
        for band, m in (("all", torch.ones_like(free)), ("free", free)):
            e = {"n": int(m.sum()), "agree": int((same & m).sum())}
            if fi is not None:
                ok_a, ok_b = (ai == fi) & m, (bi == fi) & m
                e.update(fp32_a=int(ok_a.sum()), fp32_b=int(ok_b.sum()), b=int((ok_a & ~ok_b).sum()),
                         c=int((~ok_a & ok_b).sum()))
            bands[band] = e
        miss = torch.nonzero(free & ~same).flatten().cpu()
        first = int(pos[int(miss[0])]) - (req.prompt_len - 1) if len(miss) else None
        acc = {arm: req.caps[arm]["accept"].long() for arm in (a, b)}
        return {"rid": req.rid, "pair": [a, b], "bands": bands, "first_miss": first,
                "answer": req.n - req.prompt_len, "margins": r5((a1 - a2)[free]),
                "miss_margins": r5((b1 - b2)[free & ~same]),
                "accept": {arm: {"steps": len(x), "accepted": int(x.sum())} for arm, x in acc.items()}}


def bootstrap(cols: dict[str, np.ndarray], stat: Callable[[dict[str, np.ndarray]], np.ndarray], draws: int = BOOT,
              seed: int = 0) -> np.ndarray:
    """`stat` of the columns' sums over `draws` resamples of their rows (the requests) with replacement."""
    k = len(next(iter(cols.values())))
    idx = np.random.default_rng(seed).integers(0, k, size=(draws, k))
    return stat({key: col[idx].sum(1) for key, col in cols.items()})


def pair_summary(results: list[dict[str, Any]], band: str, draws: int = BOOT) -> dict[str, Any]:
    """One pair's requests pooled (`pair`'s results): positions, the agreement share and its one-sided 95% lower bound,
    the first misses, the margins, each arm's tokens per verify step and the difference (first arm − second) with its
    two-sided 95% interval, and with the chain the argmax = H_fp32 counts, b, c and the difference of their shares
    with its interval (`pairs`: a cluster bootstrap over the requests)."""
    e = [r["bands"][band] for r in results]
    n, agree = (np.array([x[key] for x in e], np.float64) for key in ("n", "agree"))
    a, b = results[0]["pair"]
    acc = {arm: (np.array([r["accept"][arm]["steps"] for r in results], np.float64),
                 np.array([r["accept"][arm]["accepted"] for r in results], np.float64)) for arm in (a, b)}
    tpv = {arm: float(1 + x.sum() / s.sum()) if s.sum() else None for arm, (s, x) in acc.items()}
    out: dict[str, Any] = {
        "pair": [a, b], "band": band, "requests": len(results), "n": int(n.sum()), "agree": int(agree.sum()),
        "share": float(agree.sum() / n.sum()) if n.sum() else None, "lower95": None,
        "first_miss": [r["first_miss"] for r in results], "answers": [r["answer"] for r in results],
        "margins": [x for r in results for x in r["margins"]],
        "miss_margins": [x for r in results for x in r["miss_margins"]], "tpv": tpv, "tpv_delta": None}
    if n.sum():
        d = bootstrap({"a": agree, "n": n}, lambda s: s["a"] / np.maximum(s["n"], 1), draws)
        out["lower95"] = float(np.quantile(d, 0.05))
    if tpv[a] is not None and tpv[b] is not None:
        cols = {"sa": acc[a][0], "xa": acc[a][1], "sb": acc[b][0], "xb": acc[b][1]}
        d = bootstrap(cols, lambda s: s["xa"] / np.maximum(s["sa"], 1) - s["xb"] / np.maximum(s["sb"], 1), draws)
        out["tpv_delta"] = [tpv[a] - tpv[b], *np.quantile(d, [0.025, 0.975]).tolist()]
    if all("fp32_a" in x for x in e) and n.sum():
        ok_a, ok_b = (np.array([x[key] for x in e], np.float64) for key in ("fp32_a", "fp32_b"))
        d = bootstrap({"a": ok_a, "b": ok_b, "n": n}, lambda s: (s["a"] - s["b"]) / np.maximum(s["n"], 1), draws)
        out["fp32"] = {"fp32_a": int(ok_a.sum()), "fp32_b": int(ok_b.sum()), "b": sum(x["b"] for x in e),
                       "c": sum(x["c"] for x in e),
                       "delta": [float((ok_a.sum() - ok_b.sum()) / n.sum()), *np.quantile(d, [0.025, 0.975]).tolist()]}
    return out


# -- tables ------------------------------------------------------------------------------------------------------------


def _tps(a: list[int]) -> str:
    return f"{1 + sum(a) / len(a):.3f}" if a else "-"


def _pct(x: int, n: int) -> str:
    return f"{100 * x / n:.1f}%" if n else "-"


def _q(xs: list[float], f: float) -> str:
    return f"{float(np.quantile(xs, f)):.3f}" if xs else "-"


def layers_tables(res: dict[str, Any]) -> str:
    reqs = [r for r in res["requests"] if r.get("layers")]
    if not reqs:
        return "No arm has taps (an eager forced run with --taps): no per-layer errors."
    rows0 = reqs[0]["layers"]
    arms = list(dict.fromkeys(a for r in reqs for row in r["layers"] for a in row["mixer"]))

    def pool(ls: list[int], meas: str, arm: str, band: str) -> dict[str, float]:
        return pooled([r["layers"][i][meas][arm][band] for r in reqs for i in ls if arm in r["layers"][i][meas]])

    def cell(ls: list[int], meas: str, arm: str, peak: bool = False) -> str:
        p, q = pool(ls, meas, arm, "prompt"), pool(ls, meas, arm, "answer")
        return f"{rel(p)} / {rel(q)}" + (f" (max {rel_max(q)})" if peak else "")

    out = ["### Per layer: relative RMS error against fp32, prompt / answer positions", "",
           "mixer: the fp32 mixer from the arm's stream; mlp: the fp32 MLP from the arm's mid; layer: the fp32 layer "
           "from the arm's stream, relative to the fp32 layer's delta.", "",
           md(["layer", "kind", *(f"{a} {m}" for a in arms for m in MEASURES)],
              [[row["index"], row["kind"], *(cell([i], m, a) for a in arms for m in MEASURES)]
               for i, row in enumerate(rows0)])]
    kinds = list(dict.fromkeys(row["kind"] for row in rows0))
    of = {kind: [i for i, row in enumerate(rows0) if row["kind"] == kind] for kind in kinds}
    agg = [(f"{kind} mixer", of[kind], "mixer") for kind in kinds] + [("MLP", list(range(len(rows0))), "mlp")] + \
        [(f"{kind} layer", of[kind], "layer") for kind in kinds]
    out += ["", "### Pooled per kind: relative RMS prompt / answer (max relative error, answer)", "",
            md(["", *arms], [[label, *(cell(ls, meas, a, True) for a in arms)] for label, ls, meas in agg])]
    core = [c for r in reqs for c in r.get("core", ())]
    if core:
        keys = list(dict.fromkeys((c["index"], c["kind"], c["arm"]) for c in core))
        crow = []
        for key in keys:
            cs = [c for c in core if (c["index"], c["kind"], c["arm"]) == key]
            p, q = pooled([c["prompt"] for c in cs]), pooled([c["answer"] for c in cs])
            crow.append([*key, f"{rel(p)} / {rel(q)}", f"{rel_max(p)} / {rel_max(q)}",
                         f"{sum(c['rows'] for c in cs)} / {sum(c['skipped'] for c in cs)}" if key[1] == "gdn" else "-"])
        out += ["", "### Kernels in situ: fp32 from the kernel's captured inputs against its captured output", "",
                md(["layer", "kind", "arm", "rel RMS prompt / answer", "max rel prompt / answer",
                    "GDN rows used / skipped"], crow)]
    fills = {a: [sum(r["fill"][a][key] for r in reqs if a in r["fill"]) for key in ("from_arms", "rest")]
             for a in arms}
    if any(sum(v) for v in fills.values()):
        out += ["", "Positions an arm lacks, taken from another arm's stream / the chain's: " +
                ", ".join(f"{a} {v[0]} / {v[1]}" for a, v in fills.items())]
    return "\n".join(out)


def chain_tables(res: dict[str, Any]) -> str:
    reqs = res["requests"]
    arms = list(dict.fromkeys(a for r in reqs for a in r["hidden"]))
    rows = []
    for a in arms:
        p, q = (pooled([r["hidden"][a][band] for r in reqs if a in r["hidden"]]) for band in BANDS)
        rows.append([a, rel(p), rel(q), rel_max(q), q["n"]])
    bases = ", ".join(dict.fromkeys(r["base"] for r in reqs))
    out = [f"### Final hidden state against the fp32 chain's H_fp32 (its embeddings: {bases})", "",
           md(["arm", "rel RMS prompt", "rel RMS answer", "max rel answer", "answer positions"], rows)]
    taps = list(dict.fromkeys(a for r in reqs for a in r["stream"][0]))
    if taps:
        kinds = res["meta"]["kinds"]
        srow = []
        for i in range(len(reqs[0]["stream"])):
            cells = []
            for a in taps:
                p, q = (pooled([r["stream"][i][a][band] for r in reqs if a in r["stream"][i]]) for band in BANDS)
                cells.append(f"{rel(p)} / {rel(q)}")
            srow.append([i, kinds[i] if i < len(kinds) else "final norm", *cells])
        out += ["", "### Accumulated: each arm's stream entering layer l against the chain's (relative RMS, prompt / "
                "answer)", "", md(["l", "layer l", *taps], srow)]
    return "\n".join(out)


def drafter_tables(res: dict[str, Any]) -> str:
    reqs = res["requests"]
    names = list(dict.fromkeys(nm for r in reqs for nm in r["sources"]))
    k = max(r["k"] for r in reqs)
    rows = []
    for nm in names:
        walked, every, reach, hit = [], [], np.zeros(k), np.zeros(k)
        for r in reqs:
            a = r["sources"].get(nm)
            if a is None:
                continue
            walked += [x for _, x in walk(a, r["p0"], r["start"], r["n"])]
            every += a
            for i, x in enumerate(a):
                for j in range(r["k"]):
                    if r["p0"] + i + 2 + j < r["n"]:
                        reach[j] += 1
                        hit[j] += x > j
        rows.append([nm, len(walked), _tps(walked), _tps(every),
                     " ".join(f"{h / c:.3f}" if c else "-" for h, c in zip(hit, reach))])
    out = ["### The drafter in fp32 on each hidden source", "",
           "walk: the engine's rounds (p <- p + 1 + a(p)); every p: the mean of 1 + a(p) over every round start.", "",
           md(["source", "rounds", "tokens/step (walk)", "tokens/step (every p)", "acceptance per draft position"],
              rows)]
    engines = list(dict.fromkeys(a for r in reqs for a in r["engine"]))
    if engines:
        erow = []
        for arm in engines:
            raw, rounds, same = [], [], dict.fromkeys(names, 0)
            for r in reqs:
                e = r["engine"].get(arm)
                if e is not None:
                    raw += e["accept"]
                    rounds += [x for _, x in e["rounds"]]
                    for nm, v in e["same"].items():
                        same[nm] += v
            erow.append([arm, len(raw), _tps(raw), len(rounds), _tps(rounds),
                         *(_pct(same[nm], len(rounds)) for nm in names)])
        out += ["", "### In-engine acceptance (`accept`), and the share of its rounds each source's a(p) matches", "",
                md(["arm", "verify steps", "tokens/step", "rounds in the reference", "tokens/step there",
                    *(f"= {nm}" for nm in names)], erow)]
    if len(reqs) > 1:
        out += ["", "### Tokens per verify step by request (walk; in-engine in the reference)", "",
                md(["request", *names, *(f"{a} in-engine" for a in engines)],
                   [[r["rid"], *(_tps([x for _, x in walk(r["sources"][nm], r["p0"], r["start"], r["n"])])
                                 if nm in r["sources"] else "-" for nm in names),
                     *(_tps([x for _, x in r["engine"][a]["rounds"]]) if a in r["engine"] else "-" for a in engines)]
                    for r in reqs])]
    fills = {a: sum(r["fill"].get(a, 0) for r in reqs) for a in names}
    if any(fills.values()):
        out += ["", "Hidden states an arm lacks where the drafter reads them, taken from H_fp32 (or another arm): " +
                ", ".join(f"{a} {v}" for a, v in fills.items() if v)]
    if any(r["draft_vocab"] for r in reqs):
        out += ["", "The engine drafted with a reduced head (draft_vocab); the fp32 drafter scores the full one."]
    return "\n".join(out)


def target_tables(res: dict[str, Any]) -> str:
    reqs = res["requests"]
    fm = [x for r in reqs for x in r["fp32"]["margins"]]
    fn = sum(r["fp32"]["n"] for r in reqs)
    rows = [["fp32", fn, "-", _pct(sum(r["fp32"]["ref"] for r in reqs), fn), _q(fm, 0.1), _q(fm, 0.5), "-", "-"]]
    for a in dict.fromkeys(a for r in reqs for a in r["arms"]):
        es = [r["arms"][a] for r in reqs if a in r["arms"]]
        n, agree = sum(e["n"] for e in es), sum(e["agree"] for e in es)
        ms, dis = [x for e in es for x in e["margins"]], [x for e in es for x in e["dis_fp32_margins"]]
        rows.append([a, n, _pct(agree, n), _pct(sum(e["ref"] for e in es), n), _q(ms, 0.1), _q(ms, 0.5), n - agree,
                     f"{_q(dis, 0.5)} / {max(dis):.3f}" if dis else "-"])
    return "\n".join(["### The fp32 head on each hidden state, at the positions predicting the answer", "",
                      md(["hidden", "positions", "argmax = fp32's", "argmax = reference", "margin p10", "margin p50",
                          "disagreements", "fp32 margin there p50 / max"], rows)])


def pair_groups(results: list[dict[str, Any]]) -> list[tuple[str, str, list[dict[str, Any]]]]:
    """`pair` results by pair and workload (its `workload`), then the pair's every workload together ("all") when it
    has several."""
    out = []
    for p in dict.fromkeys(tuple(r["pair"]) for r in results):
        rs = [r for r in results if tuple(r["pair"]) == p]
        wls = list(dict.fromkeys(r.get("workload") or "-" for r in rs))
        out += [(f"{p[0]} : {p[1]}", w, [r for r in rs if (r.get("workload") or "-") == w]) for w in wls]
        if len(wls) > 1:
            out.append((f"{p[0]} : {p[1]}", "all", rs))
    return out


def pairs_tables(res: dict[str, Any]) -> str:
    draws = int(res.get("meta", {}).get("boot", BOOT))
    agree, fp32, drafts = [], [], []
    for name, wl, rs in pair_groups(res["requests"]):
        for band in ("free", "all"):
            s = pair_summary(rs, band, draws)
            fm = [x for x in s["first_miss"] if x is not None]
            agree.append([name, wl, band, s["requests"], s["n"], _pct(s["agree"], s["n"]),
                          f"{100 * s['lower95']:.1f}%" if s["lower95"] is not None else "-",
                          f"{len(fm)} / {s['requests']}", _q(fm, 0.5), f"{_q(s['margins'], 0.1)} / "
                          f"{_q(s['margins'], 0.5)}", f"{_q(s['miss_margins'], 0.5)} / {max(s['miss_margins']):.3f}"
                          if s["miss_margins"] else "-"])
            if "fp32" in s:
                f, (d, lo, hi) = s["fp32"], s["fp32"]["delta"]
                fp32.append([name, wl, band, s["n"], _pct(f["fp32_a"], s["n"]), _pct(f["fp32_b"], s["n"]),
                             f"{100 * d:+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}]", f["b"], f["c"]])
            if band == "all":
                t, dt = s["tpv"], s["tpv_delta"]
                drafts.append([name, wl, s["requests"], *(f"{t[a]:.3f}" if t[a] is not None else "-"
                                                         for a in s["pair"]),
                               f"{dt[0]:+.3f} [{dt[1]:+.3f}, {dt[2]:+.3f}]" if dt else "-"])
    parts = ["### Top-1 agreement: the fp32 head on each arm's hidden state", "",
             md(["first : second", "workload", "positions", "requests", "n", "agree", "lower 95%",
                 "answers with a miss", "first miss p50", "first's margin p10 / p50",
                 "second's margin at misses p50 / max"], agree), "",
             "### Drafter: tokens per verify step", "",
             md(["first : second", "workload", "requests", "first", "second", "first − second [95%]"], drafts)]
    if fp32:
        parts += ["", "### Against H_fp32: argmax = fp32's (points: first − second)", "",
                  md(["first : second", "workload", "positions", "n", "first", "second", "difference [95%]",
                      "b (first only)", "c (second only)"], fp32)]
    return "\n".join(parts)


SECTIONS = (("layers", "Per layer and kernel", layers_tables), ("chain", "Accumulated along the stream", chain_tables),
            ("drafter", "Drafter", drafter_tables), ("target", "Target", target_tables),
            ("pairs", "Arm against arm", pairs_tables))


def report(d: Path) -> str:
    """Every result in d as one markdown summary, also written to report.md in d."""
    parts, meta = [], None
    for name, title, tables in SECTIONS:
        f = d / f"{name}.json"
        if f.exists():
            res = json.loads(f.read_text())
            meta = meta or res["meta"]
            parts += [f"## {title}", "", tables(res), ""]
    if meta is None:
        raise FileNotFoundError(f"{d}: no results ({', '.join(f'{s[0]}.json' for s in SECTIONS)})")
    settings = [[a, ", ".join(f"{k}={v}" for k, v in s.items() if isinstance(v, bool | int | float | str))]
                for a, s in meta["arms"].items()]
    text = "\n".join([f"# fp32 analysis: {meta['family']}", "",
                      f"{meta['model_dir']}; the reference in {meta['dtype']} on {meta['device']}; requests "
                      f"{meta['rids']}.", "", md(["arm", "settings"], settings), "", *parts])
    (d / "report.md").write_text(text)
    return text


# -- the CLI -----------------------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python3 -m tests.scripts.numerics_fp32", description=__doc__.split("\n\n")[0],
                                 formatter_class=HelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    helps = {"layers": "per layer and kernel, from each arm's own inputs", "chain": "the fp32 model from the "
             "embeddings: H_fp32 and the accumulated error", "drafter": "the fp32 drafter on each hidden source",
             "target": "the fp32 head on each hidden source", "all": "layers and chain in one pass, then drafter, "
             "target and report", "pairs": "an arm against another (each FP8-KV arm against its twin): the head on "
             "both hidden states, and with --chain against H_fp32"}
    for name, text in helps.items():
        p = sub.add_parser(name, help=text, description=text, formatter_class=HelpFormatter)
        p.add_argument("arms", nargs="+", type=Path,
                       help="arm folders of one forced run (r<rid>.pt each), or the run's folder")
        p.add_argument("--model", help="the checkpoint, a folder or a repo id (default: the captures' model_dir)")
        p.add_argument("--assistant", help="Gemma's assistant, a folder or a repo id (default: the captures' "
                       "assistant_dir)")
        p.add_argument("--device", help="cuda (the default when there is one) or cpu")
        p.add_argument("--dtype", choices=("float32", "float64"), default="float32",
                       help="the reference's dtype (the module docstring)")
        p.add_argument("--rids", type=parse_ids, help="these requests only, e.g. 0-3,7")
        if name != "pairs":
            p.add_argument("--base", default="on", help="the arm whose stream[0] the chain starts from, by name or "
                           "switch (on: on-eager, on-cache-eager, ...)")
        p.add_argument("--out", type=Path, help="results folder (default: results/numerics/fp32/<UTC time>)")
        if name in ("drafter", "target"):
            p.add_argument("--chain", type=Path, help="a chain's results (its --out): H_fp32 and the KV, else the "
                           "chain runs first")
        if name == "pairs":
            p.add_argument("--pair", action="append", default=[], metavar="FIRST:SECOND",
                           help="arms to compare by name, e.g. on-eager:on (repeatable; default: each FP8-KV arm "
                           "against its twin)")
            p.add_argument("--chain", type=Path, help="a chain's results (its --out): adds each arm against H_fp32 "
                           "(default: the head on the two hidden states alone, no chain)")
            p.add_argument("--boot", type=int, default=BOOT, help="cluster-bootstrap draws")
        if name in ("drafter", "all"):
            p.add_argument("--k", type=int, help="drafts a round (default: the captures' k)")
    p = sub.add_parser("report", help="one markdown summary of a results folder", formatter_class=HelpFormatter)
    p.add_argument("dir", type=Path, help="a results folder (its <name>.json); report.md goes there")
    args = ap.parse_args(argv)
    if args.cmd == "report":
        print(report(args.dir))
        return
    import torch

    out = (args.out or results_dir("fp32")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    requests = load_requests(args.arms, args.rids)
    first = next(iter(requests[0].caps.values()))
    if not (args.model or first.get("model_dir")):
        ap.error("the captures name no model_dir: give --model")
    asst = args.assistant or first.get("assistant_dir")
    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ref = make_ref(resolve(args.model or first["model_dir"]), resolve(asst) if asst else None, device=dev,
                   dtype=getattr(torch, args.dtype))
    if first.get("model") not in (None, ref.family):
        ap.error(f"the captures are of a {first['model']} model, the checkpoint a {ref.family}")
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    t0 = time.perf_counter()
    arms: dict[str, Any] = {}
    for req in requests:
        for arm, c in req.caps.items():
            arms.setdefault(arm, c.get("settings") or {})
    meta = {"tool": "numerics_fp32", "command": args.cmd, "family": ref.family, "model_dir": str(ref.dir),
            "device": str(dev), "dtype": args.dtype, "kinds": ref.kinds, "rids": [r.rid for r in requests],
            "arms": arms}

    def write(name: str, results: list[dict[str, Any]]) -> None:
        peak = torch.cuda.max_memory_allocated(dev) / 2**30 if dev.type == "cuda" else None
        res = {"meta": {**meta, "seconds": round(time.perf_counter() - t0, 1), "peak_gib": peak},
               "requests": results}
        (out / f"{name}.json").write_text(json.dumps(res, indent=1, default=str))
        print(next(s[2] for s in SECTIONS if s[0] == name)(res), flush=True)
        log(f"wrote {out / name}.json")

    if args.cmd == "pairs":
        names = list(arms)
        pairs = [tuple(x.split(":", 1)) for x in args.pair] if args.pair else default_pairs(names)
        unknown = sorted({x for p in pairs for x in p if x not in names} | {":".join(p) for p in pairs if len(p) < 2})
        if not pairs or unknown:
            ap.error(f"no pair to compare among the arms {names}" + (f": {', '.join(unknown)}" if unknown else ""))
        chains = {}
        if args.chain is not None:
            folder = args.chain / "chain" if (args.chain / "chain").is_dir() else args.chain
            chains = {req.rid: Chain.load(folder / f"r{req.rid}.pt") for req in requests}
        wls = workloads(args.arms)
        results = [{**r, "workload": wls.get(req.rid)} for a, b in pairs for req in requests
                   if a in req.caps and b in req.caps and (r := pair(ref, req, a, b, chains.get(req.rid)))]
        meta["boot"] = args.boot
        write("pairs", results)
        return
    chains: dict[int, Chain | None] = {}
    if args.cmd in ("layers", "chain", "all") or getattr(args, "chain", None) is None:
        swept = sweep(ref, requests, layers=args.cmd in ("layers", "all"), chain=args.cmd != "layers", base=args.base)
        if args.cmd in ("layers", "all"):
            write("layers", [{key: s.result[key] for key in ("rid", "n", "prompt_len", "fill", "layers", "core")
                              if key in s.result} for s in swept.values()])
        if args.cmd != "layers":
            (out / "chain").mkdir(exist_ok=True)
            for rid, s in swept.items():
                s.chain.save(out / "chain" / f"r{rid}.pt")
                chains[rid] = s.chain
            write("chain", [{key: s.result[key] for key in ("rid", "n", "prompt_len", "base", "stream", "hidden")}
                            for s in swept.values()])
    else:
        folder = args.chain / "chain" if (args.chain / "chain").is_dir() else args.chain
        chains = {req.rid: Chain.load(folder / f"r{req.rid}.pt") for req in requests}
    if args.cmd in ("drafter", "all"):
        write("drafter", [drafter(ref, req, chains.get(req.rid), k=args.k, base=args.base) for req in requests])
    if args.cmd in ("target", "all"):
        write("target", [target(ref, req, chains[req.rid]) for req in requests])
    if args.cmd == "all":
        report(out)
        log(f"wrote {out / 'report.md'}")


if __name__ == "__main__":
    main()
