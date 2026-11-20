"""GPU probes of single kernels at Qwen3.8 27B's and Gemma 4 31B's real shapes, each against an fp32 reference and
against the others: how much accuracy the attention and GDN kernels of the two paths batch invariance switches between
(`dev.runner.cuda.batch_invariant`; docs/architecture.md, "Batch invariance") lose, where, and whether FlashInfer's GDN
prefill moves bits between identical calls.

    python3 -m tests.scripts.numerics_kernels smid
    python3 -m tests.scripts.numerics_kernels gdn-race [--repeats 300] [--lens 1000,300,2000,70,700]
    python3 -m tests.scripts.numerics_kernels gdn-accuracy [--lens 500,2000,8000] [--inits zero,warm] [--inputs FILE]
    python3 -m tests.scripts.numerics_kernels attn-accuracy [--parts qwen-full,gemma-sliding,gemma-full] \\
        [--rows 1,16,128]
    python3 -m tests.scripts.numerics_kernels nan-leak
    python3 -m tests.scripts.numerics_kernels timing
    python3 -m tests.scripts.numerics_kernels attn-timing [--parts qwen-full,gemma-full] [--max-model-lens 8184,262144]

They run on an SM120 GPU (an RTX PRO 6000: 188 SMs) with the kernel library and FlashInfer 0.7.2, from the GPU tests'
image (`/opt/llm-mini-infer/test`); `--help` and the host-side helpers (tests/repo/test_numerics.py) run anywhere. Each
subcommand prints markdown tables and writes them, with its JSON results, into `--out` (default
`results/numerics/kernels/<UTC time>/`): `<subcommand>.md` and `<subcommand>.json` (non-finite numbers as null).
`<subcommand> --help` describes each. `attn-timing` times (CUDA events, eagerly and in decode graphs) the attention
kernels of both models' full layers, the batch-invariant path against the switch-off one, at a grid of prefill and
decode shapes and `--max-model-lens`. Inputs are random from `--seed`, reseeded per case, so a case gets the same inputs
alone or among others. A case that raises is recorded in the JSON and the run goes on; the exit status is then 1.

**Shapes.** Qwen3.8 27B's GDN layers: 16 q/k heads, 48 v heads, head dim 128, fp32 states [48, 128, 128], q and k
L2-normalised by the caller, the decay passed as alpha = exp(g) (`kernels.flashinfer.flashinfer_gdn_prefill`); its full
attention: 24 query and 4 KV heads, head dim 256, scale 1/16, pages of 16. Gemma 4 31B's sliding layers: 32 query and
16 KV heads, head dim 256, window 1,024, scale 1; its full layers: 32 query and 4 KV heads, head dim 512, K derived from
the stored V plane (`kernels.attn512`). Verify rows hold 1 + k queries (`--k`, 3), planned for an upper bound of their
KV length k above the lowest (one step in flight under MTP: `runner.cuda.runner`'s docstring).

**Reuse.** The GDN calls are FlashInfer's as our wrapper makes them (`kernels.flashinfer`): the per-sequence path and
its cut rows through `_gdn_call` (tests/support/kernel_cases.py, the kernel tests' cases), the context-parallel route
and the checkpointing variant (which our wrapper does not take: not deterministic on SM120 in 0.7.2, fixed in 0.7.3 by
flashinfer-ai/flashinfer#7785) directly (`_fi`). The attention setups and references are the kernel tests' too
(`_window_setup`, `_window_ref`, `_window_run`, `_full_case`, `_run`, `_stage`); the fp32 GDN reference (`gdn_fp32`) is
tests/scripts/numerics_fp32.py's `recurrence` (`ops.recurrent_step` token by token, the math of
`models.qwen3_5.backend.torch_gdn_prefill`) over all rows and heads at once; the table, id-list, results-folder and help
helpers are tests/scripts/numerics_forced.py's. torch, FlashInfer, Triton and our kernel library are imported inside the
functions.
"""

from __future__ import annotations

import argparse
import functools
import inspect
import itertools
import json
import math
import subprocess
import sys
import time
import traceback
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from llm_mini_infer.config import BATCH_INVARIANT_MAX_LEN

from ..support.paths import ROOT
from .numerics_forced import HelpFormatter, md, parse_ids, results_dir
from .numerics_fp32 import recurrence

DEV = "cuda"
GDN_HK, GDN_HV, GDN_D = 16, 48, 128  # Qwen3.8 27B's GDN layers: q/k heads, v heads, head dim
KEYS = ("q", "k", "v", "a", "b")  # a GDN row's per-token inputs (`_gdn_seq`'s names: a the decay alpha, b beta)
CHUNK = 64  # FlashInfer's GDN chunk: its checkpoints every 64 tokens, our cuts at multiples of it
PS = 16  # the KV page
KV_RMS = 0.5  # the random keys' and values' RMS per dim (`_window_setup`, `_full_case`)
TENSORMAPS = ("gdn_prefill_tensormaps", "gdn_cp_sm120_prefill_tensormaps")  # FlashInfer's per-SM descriptor buffers


# -- host-side helpers (tests/repo/test_numerics.py) -------------------------------------------------------------------


def names(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def cut_points(n: int, parts: int = 3) -> list[int]:
    """Checkpoint offsets in a GDN row of n tokens: the multiples of 64 at or below n × j / parts (j = 1 .. parts −
    1), those strictly inside the row."""
    return sorted({n * j // parts // CHUNK * CHUNK for j in range(1, parts)} - {0, n})


def row_kinds(rows: np.ndarray, cu: np.ndarray, chunk: int = CHUNK) -> np.ndarray:
    """Where rows of a packed GDN call lie (gdn-race), per row index of the call (sequence bounds `cu`): bit 1 when it
    is in its sequence's partial last chunk (a length not a multiple of `chunk`), bit 2 when an earlier sequence's
    partial last chunk reaches it (its `chunk` rows run past that sequence's end, into the rows after it); 0
    elsewhere."""
    rows, cu = np.asarray(rows, np.int64), np.asarray(cu, np.int64)
    lens = np.diff(cu)
    rem = lens % chunk
    reach = np.maximum.accumulate(np.where(rem > 0, cu[1:] + chunk - rem, cu[1:]))  # the furthest row read so far
    seq = np.searchsorted(cu, rows, side="right") - 1
    tail = (rem[seq] > 0) & (rows - cu[seq] >= lens[seq] - rem[seq])
    head = rows < np.concatenate([[0], reach])[seq]
    return tail.astype(np.int64) | (head.astype(np.int64) << 1)


def padded_index(lens: list[int] | np.ndarray, multiple: int = CHUNK) -> tuple[np.ndarray, np.ndarray]:
    """Every row of a packed call padded to a multiple of `multiple` tokens (nan-leak's candidate fix): the padded
    cu_seqlens, and per token of the unpadded call, in order, its row in the padded one."""
    lens = np.asarray(lens, np.int64)
    pcu = np.concatenate([[0], np.cumsum(-(-lens // multiple) * multiple)])
    cu = np.concatenate([[0], np.cumsum(lens)])
    return pcu, np.repeat(pcu[:-1] - cu[:-1], lens) + np.arange(int(cu[-1]))


def pad_row(x: dict[str, Any], multiple: int = CHUNK) -> dict[str, Any]:
    """A GDN row's inputs (`KEYS`, and its state `s`) padded to a multiple of `multiple` tokens with tokens that leave
    an exact recurrence where it is: q = k = v = 0, decay alpha = 1, beta = 0 (nan-leak's candidate fix)."""
    import torch

    p = -x["q"].shape[0] % multiple

    def ext(t: Any, fill: float) -> Any:
        return torch.cat([t, torch.full((p, *t.shape[1:]), fill, dtype=t.dtype, device=t.device)]) if p else t

    return {"q": ext(x["q"], 0.0), "k": ext(x["k"], 0.0), "v": ext(x["v"], 0.0), "a": ext(x["a"], 1.0),
            "b": ext(x["b"], 0.0), "s": x["s"]}


def decode_bucket(n: int, buckets: list[int]) -> int:
    """The decode graph a step of n rows replays: the smallest bucket that holds them."""
    return min(b for b in buckets if b >= n)


@dataclass
class AttnCase:
    """One attention step (attn-accuracy): rows of `nq` queries each, the first `n_dec` verify rows (device input,
    split, planned for `ub`, their true KV lengths anywhere in [lb, ub]), the rest prefill rows (lb = ub, exact)."""

    name: str
    nq: np.ndarray
    lb: np.ndarray  # each row's lowest possible KV length after the step
    ub: np.ndarray  # the length its plan is made for
    n_dec: int

    @property
    def starts(self) -> np.ndarray:
        return np.concatenate([[0], np.cumsum(self.nq)[:-1]])

    @property
    def qo_indptr(self) -> np.ndarray:
        return np.concatenate([[0], np.cumsum(self.nq)])

    @property
    def uncertain(self) -> bool:
        return bool((self.lb < self.ub).any())

    def true(self, mode: str, rng: np.random.Generator) -> np.ndarray:
        """The rows' true KV lengths: the lowest (`lb`), the bound (`ub`) or each uniform between (`rand`)."""
        if mode == "lb":
            return self.lb.copy()
        if mode == "ub":
            return self.ub.copy()
        return self.lb + rng.integers(0, self.ub - self.lb + 1)


def attn_cases(rows: list[int], lo: int, hi: int, k: int, rng: np.random.Generator) -> list[AttnCase]:
    """attn-accuracy's steps: per row count, verify rows of 1 + k queries in lockstep (one lowest length for all, as
    requests of one template) and diverse (lengths uniform in [lo, hi]), each planned k above it; a 2,048-token prefill
    row from 0; prefill rows of 128, 256 and 512 tokens over 4,000 to 5,000 positions; 16 verify rows beside a 512-token
    prefill chunk over 3,000."""
    out = []
    for n in rows:
        for kind in ("lockstep", "diverse") if n > 1 else ("diverse",):
            lb = np.full(n, rng.integers(lo, hi + 1)) if kind == "lockstep" else rng.integers(lo, hi + 1, n)
            out.append(AttnCase(f"verify-{n}-{kind}", np.full(n, 1 + k), lb.astype(np.int64), lb + k, n))
    out.append(AttnCase("prefill-2048", np.asarray([2048]), np.asarray([2048]), np.asarray([2048]), 0))
    tail = np.asarray([128, 256, 512])
    end = np.asarray([4000, 4500, 5000]) + tail
    out.append(AttnCase("prefill-history", tail, end, end.copy(), 0))
    lb = rng.integers(lo, hi + 1, 16)
    out.append(AttnCase("mixed-16+512", np.r_[np.full(16, 1 + k), 512], np.r_[lb, 3512], np.r_[lb + k, 3512], 16))
    return out


def rel_errors(x: Any, ref: Any) -> dict[str, Any]:
    """x against ref (any shapes alike): relative RMS ‖x − ref‖ / ‖ref‖, max |x − ref| / max |ref|, and x's
    non-finite values."""
    import torch

    x, ref = x.float(), ref.float()
    if not x.numel():
        return {"rel_rms": None, "max_rel": None, "nonfinite": 0}
    d = x - ref
    rn, rm = float(ref.norm()), float(ref.abs().max())
    return {"rel_rms": float(d.norm()) / (rn or 1.0), "max_rel": float(d.abs().max()) / (rm or 1.0),
            "nonfinite": int((~torch.isfinite(x)).sum())}


def gdn_fp32(rows: list[dict[str, Any]],
             snaps: dict[int, list[int]] | None = None) -> list[tuple[Any, Any, dict[int, Any]]]:
    """The fp32 GDN recurrence over prefill rows (`KEYS` and the initial state `s` each), each from its state:
    tests/scripts/numerics_fp32.py's `recurrence` (`ops.recurrent_step` token by token, the math of
    `models.qwen3_5.backend.torch_gdn_prefill`: value head h reads q/k head h // (HV / HK), q scaled by K^-1/2), all
    rows and heads at once on any device, a shorter row padded with tokens that leave its state where it is
    (`pad_row`'s). Per row: the outputs fp32 [n, HV, V], the final state, and the states after the `snaps[row]` tokens
    counts."""
    import torch

    lens = [int(r["q"].shape[0]) for r in rows]
    tmax, dev = max(lens), rows[0]["q"].device

    def stack(key: str, fill: float) -> Any:
        out = torch.full((len(rows), tmax, *rows[0][key].shape[1:]), fill, dtype=torch.float32, device=dev)
        for i, r in enumerate(rows):
            out[i, :lens[i]] = r[key].float()
        return out

    x = [stack(key, fill) for key, fill in zip(KEYS, (0.0, 0.0, 0.0, 1.0, 0.0))]
    h = torch.stack([r["s"].float() for r in rows])
    snaps = snaps or {}
    edges = sorted({0, tmax, *(off for offs in snaps.values() for off in offs if 0 < off < tmax)})
    got: list[dict[int, Any]] = [{} for _ in rows]
    outs = []
    for t0, t1 in itertools.pairwise(edges):
        outs.append(recurrence(*(t[:, t0:t1] for t in x), h))  # h advances in place
        for i, offs in snaps.items():
            if t1 in offs:
                got[i][t1] = h[i].clone()
    o = torch.cat(outs, 1)
    return [(o[i, :lens[i]], h[i], got[i]) for i in range(len(rows))]



def case_history(case: AttnCase) -> int:
    """The positions before a single-shape case's rows' queries (attn-timing's grids)."""
    return int(case.ub[0] - case.nq[0])


def timing_cases(short_chunks: list[int], histories: list[int], long_chunk: int, long_histories: list[int],
                 decode_rows: list[int], decode_histories: list[int], k: int) -> list[AttnCase]:
    """attn-timing's steps, their KV lengths exact (lb = ub): one prefill row of each of `short_chunks` queries over
    each of `histories` positions, one of `long_chunk` over each of `long_histories`, and per `decode_rows` count
    verify rows of 1 + k queries over each of `decode_histories`. Each row has the shape's one length."""

    def one(name: str, rows: int, nq: int, history: int, n_dec: int) -> AttnCase:
        kv = np.full(rows, history + nq, np.int64)
        return AttnCase(name, np.full(rows, nq, np.int64), kv, kv.copy(), n_dec)

    out = [one(f"prefill-short-{c}-h{h}", 1, c, h, 0) for c in short_chunks for h in histories]
    out += [one(f"prefill-long-{long_chunk}-h{h}", 1, long_chunk, h, 0) for h in long_histories]
    return out + [one(f"decode-{n}-h{h}", n, 1 + k, h, n) for n in decode_rows for h in decode_histories]


def timing_variants(case: AttnCase, prefill_keys: list[int], splits_prefill: bool) -> list[int | None]:
    """The prefill split bases the batch-invariant path of `case` is timed at: each of `prefill_keys` for a prefill
    step on a part whose prefill rows split (the full layers'), else one, None (the default; a decode row's split comes
    from its own position, whatever the context)."""
    return list(prefill_keys) if splits_prefill and not case.n_dec else [None]


def timing_ratios(recs: list[dict[str, Any]]) -> None:
    """Each timed batch-invariant record's `ratio`: its median over the switch-off record's of its case and mode."""
    off = {(r["case"], r["mode"]): r["median_us"] for r in recs if r.get("impl") == "off" and "median_us" in r}
    for r in recs:
        if r.get("impl") == "on" and "median_us" in r and (r["case"], r["mode"]) in off:
            r["ratio"] = r["median_us"] / off[r["case"], r["mode"]]


# -- output ---------------------------------------------------------------------------------------------------------


def _out(args: argparse.Namespace) -> Path:
    out = (Path(args.out) if args.out else results_dir("kernels")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


def _clean(x: Any) -> Any:
    """JSON-ready: numpy to Python, non-finite floats to None, keys to str."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return _clean(x.tolist())
    if isinstance(x, np.generic):
        return _clean(x.item())
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def _save(out: Path, name: str, result: dict[str, Any], tables: list[str]) -> None:
    (out / f"{name}.json").write_text(json.dumps(_clean(result), indent=1) + "\n")
    (out / f"{name}.md").write_text("\n\n".join(tables) + "\n")


def md_table(title: str, headers: list[str], rows: list[list[Any]]) -> str:
    """A titled markdown table (`md`)."""
    return f"**{title}**\n\n{md(headers, rows)}"


def _e(x: float | None) -> str:
    return "-" if x is None else f"{x:.2e}"


def _cuda(library: bool = False) -> Any:
    """torch, once a CUDA GPU (and with `library` the kernel library) is there; TF32 off for the fp32 references."""
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA GPU")
    if library:
        from llm_mini_infer.kernels import build

        if build.load() is None:
            raise SystemExit("needs the kernel library (llm_mini_infer.kernels.build)")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch


def _device() -> dict[str, Any]:
    import torch

    p = torch.cuda.get_device_properties(0)
    try:
        import flashinfer

        fi = getattr(flashinfer, "__version__", "?")
    except ImportError:
        fi = None
    return {"name": p.name, "sms": p.multi_processor_count, "capability": f"{p.major}.{p.minor}",
            "memory_gb": round(p.total_memory / 2 ** 30, 1), "torch": torch.__version__, "cuda": torch.version.cuda,
            "flashinfer": fi}


def _dev64(a: Any) -> Any:
    import torch

    return torch.as_tensor(np.asarray(a, np.int64), device=DEV)


def _bits(t: Any) -> Any:
    import torch

    return t.contiguous().view(torch.int16 if t.element_size() == 2 else torch.int32)


def _failed(what: str, exc: BaseException) -> dict[str, str]:
    traceback.print_exc()
    print(f"{what}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    return {"error": f"{type(exc).__name__}: {exc}"}


# -- smid -----------------------------------------------------------------------------------------------------------


_SMID_CPP = "void smid_probe(torch::Tensor out, int64_t spin_ns);"
_SMID_CU = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void smid_kernel(int* out, long long spin_ns) {
  unsigned sm, nsm;
  unsigned long long t0, t;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
  asm volatile("mov.u32 %0, %%nsmid;" : "=r"(nsm));
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  do {
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  } while ((long long)(t - t0) < spin_ns);
  if (threadIdx.x == 0) {
    out[2 * blockIdx.x] = (int)sm;
    out[2 * blockIdx.x + 1] = (int)nsm;
  }
}

void smid_probe(torch::Tensor out, int64_t spin_ns) {
  smid_kernel<<<(unsigned)out.size(0), 128, 0, at::cuda::getCurrentCUDAStream()>>>(out.data_ptr<int>(), spin_ns);
}
"""


@functools.cache
def _smid_triton() -> Callable[[Any, int], None]:
    """The probe in Triton (inline PTX): `launch(out int32 [CTAs, 2], spin_ns)`. `triton` and `tl` are bound as module
    globals, where Triton's compiler looks a kernel's names up."""
    global triton, tl
    import triton
    import triton.language as tl

    @triton.jit
    def smid_kernel(out_ptr, spin_ns):
        sm, nsm = tl.inline_asm_elementwise("mov.u32 $0, %smid;\n\tmov.u32 $1, %nsmid;", "=r,=r", [],
                                            dtype=(tl.int32, tl.int32), is_pure=False, pack=1)
        t0 = tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l", [], dtype=tl.int64, is_pure=False, pack=1)
        t = t0
        while t - t0 < spin_ns:
            t = tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l", [], dtype=tl.int64, is_pure=False,
                                          pack=1)
        pid = tl.program_id(0)
        tl.store(out_ptr + 2 * pid, sm)
        tl.store(out_ptr + 2 * pid + 1, nsm)

    def launch(out: Any, spin_ns: int) -> None:
        smid_kernel[(out.shape[0],)](out, spin_ns)

    return launch


@functools.cache
def _smid_cuda() -> Callable[[Any, int], None]:
    """The probe as a `torch.utils.cpp_extension.load_inline` extension (nvcc), when Triton is not there."""
    from torch.utils.cpp_extension import load_inline

    mod = load_inline("llm_mini_infer_smid_probe", _SMID_CPP, cuda_sources=_SMID_CU, functions=["smid_probe"],
                      extra_cuda_cflags=["-O2"])
    return mod.smid_probe


def cmd_smid(args: argparse.Namespace) -> int:
    """Launches of `--blocks-per-sm` × the SM count CTAs (Triton's inline PTX, else a `load_inline` extension), each
    spinning `--spin-us` first so that the launch spreads over every SM, record every CTA's `%smid` and `%nsmid`,
    `--launches` times: the ids' min, max and distinct count against the SM count (`multiProcessorCount`), the ids at
    or above it, those below it never seen, and CTAs per id. FlashInfer's GDN prefill kernels on SM120 place a
    sequence's tail-tile TMA descriptor at `buffer + %smid × 128` in a buffer of SM count × 128 bytes
    (`gdn_kernels/delta_rule_dsl/collective_store_tma.py`; the buffers `gdn_prefill_tensormaps` and
    `gdn_cp_sm120_prefill_tensormaps`), so the probe checks that every id stays below the SM count."""
    torch = _cuda()
    out = _out(args)
    dev = _device()
    sms = dev["sms"]
    blocks, spin = args.blocks_per_sm * sms, int(args.spin_us * 1000)
    result: dict[str, Any] = {"device": dev, "blocks": blocks, "launches": args.launches, "spin_us": args.spin_us,
                              "backend_errors": {}}
    launch = None
    for backend, make in (("triton", _smid_triton), ("load_inline", _smid_cuda)):
        try:
            launch = make()
            probe = torch.full((blocks, 2), -1, dtype=torch.int32, device=DEV)
            launch(probe, spin)
            torch.cuda.synchronize()
            result["backend"] = backend
            break
        except Exception as exc:
            result["backend_errors"][backend] = _failed(f"smid ({backend})", exc)["error"]
            launch = None
    if launch is None:
        _save(out, "smid", result, [])
        return 1
    got = []
    for _ in range(args.launches):
        probe = torch.full((blocks, 2), -1, dtype=torch.int32, device=DEV)
        launch(probe, spin)
        torch.cuda.synchronize()
        got.append(probe.cpu().numpy())
    a = np.concatenate(got)
    ids, nsm = a[:, 0], a[:, 1]
    written = ids >= 0
    seen = np.unique(ids[written])
    if not len(seen):
        result["error"] = "no CTA recorded its %smid"
        _save(out, "smid", result, [])
        return 1
    result.update(unwritten=int((~written).sum()), smid_min=int(seen.min()), smid_max=int(seen.max()),
                  distinct=len(seen), nsmid=np.unique(nsm[written]).tolist(),
                  at_or_above_sm_count=seen[seen >= sms].tolist(),
                  unseen_below_sm_count=sorted(set(range(sms)) - set(seen.tolist())),
                  ctas_per_id={int(i): int(c) for i, c in zip(*np.unique(ids[written], return_counts=True))},
                  past_the_buffer=bool(seen.max() >= sms))
    table = md_table("%smid over every CTA", ["GPU", "SMs", "CTAs a launch", "backend", "%nsmid", "min", "max",
                                              "distinct", ">= SMs", "unseen < SMs", "a descriptor at %smid x 128"],
                     [[dev["name"], sms, blocks, result["backend"], result["nsmid"], result["smid_min"],
                       result["smid_max"], result["distinct"], len(result["at_or_above_sm_count"]),
                       len(result["unseen_below_sm_count"]),
                       "past the buffer" if result["past_the_buffer"] else "inside the buffer"]])
    print(table)
    _save(out, "smid", result, [table])
    return 0


# -- GDN prefill calls ----------------------------------------------------------------------------------------------


def _fi(x: dict[str, Any], init: Any, cu: Any, **kw: Any) -> tuple[Any, Any]:
    """FlashInfer's GDN prefill on packed inputs (`KEYS`) and packed states, called as our wrapper calls it
    (`kernels.flashinfer`, which reads and writes a state pool instead), with `kw` (`use_cp`; or its checkpointing
    variant: `state_checkpoints`, `checkpoint_cu_starts`, `checkpoint_every_n_tokens`, with `use_cp` off, a path our
    wrapper does not take, its state every 64 tokens of every row having moved between identical calls on SM120)."""
    from flashinfer.gdn_prefill import chunk_gated_delta_rule

    return chunk_gated_delta_rule(q=x["q"], k=x["k"], v=x["v"], g=x["a"], beta=x["b"], scale=None, initial_state=init,
                                  output_final_state=True, cu_seqlens=cu, use_qk_l2norm_in_kernel=False, **kw)


def _fi_call(rows: list[dict[str, Any]], *, use_cp: bool = False,
             every: int = 0) -> list[tuple[Any, Any, dict[int, Any]]]:
    """`_fi` over `rows` (`KEYS` and `s`) packed in one call: the route `use_cp` names, or with `every` the
    checkpointing variant, into a fresh NaN workspace and a fresh `checkpoint_cu_starts` per call (FlashInfer's
    every-`every` layout, counted from each row's start). Per row (`_gdn_call`'s): its outputs, its final state, its
    checkpoint states by offset."""
    import torch

    lens = np.asarray([r["q"].shape[0] for r in rows], np.int64)
    cu = np.concatenate([[0], np.cumsum(lens)])
    cat = {x: torch.cat([r[x] for r in rows]).contiguous() for x in KEYS}
    init = torch.stack([r["s"] for r in rows]).contiguous()
    cu_dev = torch.as_tensor(cu, device=DEV)
    starts = np.concatenate([[0], np.cumsum(lens // every)]) if every else np.zeros(len(rows) + 1, np.int64)
    cks: list[dict[int, Any]] = [{} for _ in rows]
    if starts[-1]:
        ws = torch.full((int(starts[-1]), GDN_HV, GDN_D, GDN_D), float("nan"), device=DEV)
        o, final = _fi(cat, init, cu_dev, state_checkpoints=ws, checkpoint_cu_starts=_dev64(starts),
                       checkpoint_every_n_tokens=every, use_cp=False)
        for i in range(len(rows)):
            cks[i] = {j * every: ws[int(starts[i]) + j - 1] for j in range(1, int(starts[i + 1] - starts[i]) + 1)}
    else:
        o, final = _fi(cat, init, cu_dev, use_cp=use_cp)
    torch.cuda.synchronize()
    return [(o[cu[i]:cu[i + 1]], final[i], cks[i]) for i in range(len(rows))]


def _gdn_route(route: str, rows: list[dict[str, Any]],
               cuts: dict[int, list[int]]) -> list[tuple[Any, Any, dict[int, Any]]]:
    """A GDN prefill route over `rows` packed in one call: `cp` (the context-parallel route, use_cp=True), `seq` (the
    per-sequence route: our wrapper, `_gdn_call`), `ckpt` (FlashInfer's checkpointing variant, every 64 tokens)
    or `cut` (our cut rows at `cuts`, our wrapper's checkpoints: `_gdn_call`)."""
    from ..support.kernel_cases import _gdn_call

    if route == "cp":
        return _fi_call(rows, use_cp=True)
    if route == "ckpt":
        return _fi_call(rows, every=CHUNK)
    if route == "seq":
        return _gdn_call(rows)
    if route == "cut":
        return _gdn_call(rows, cuts)
    raise SystemExit(f"unknown GDN route {route!r} (cp, seq, ckpt, cut)")


# -- gdn-race -------------------------------------------------------------------------------------------------------


def _race_worker(args: argparse.Namespace, variant: str, buffers: str) -> dict[str, Any]:
    torch = _cuda()
    from ..support.kernel_cases import _gdn_seq

    dev = torch.device(DEV, torch.cuda.current_device())  # the device FlashInfer keys its buffers by (q.device)
    if buffers == "grown":
        from flashinfer.utils import _get_cache_buf

        for name in TENSORMAPS:
            _get_cache_buf(name, 1 << 20, dev)
    lens = parse_ids(args.lens)
    cu = np.concatenate([[0], np.cumsum(lens)])
    rows = [_gdn_seq(n, args.seed * 1000 + i) for i, n in enumerate(lens)]
    cuts = {i: cut_points(n) for i, n in enumerate(lens)}
    route = {"ckpt": "ckpt", "plain": "seq", "cut": "cut"}[variant]
    first = _gdn_route(route, rows, cuts)
    from flashinfer import utils as fu

    maps = [{"name": key[0], "bytes": int(buf.numel()), "ptr": hex(buf.data_ptr())}
            for key, buf in fu._cache_buf.items() if key[0] in TENSORMAPS]
    calls = Counter()
    kinds = Counter()
    by_seq = [Counter() for _ in lens]
    final_heads, ckpt_slots, per_call = Counter(), Counter(), {}
    max_rel = 0.0
    for c in range(1, args.repeats):
        got = _gdn_route(route, rows, cuts)
        diff_rows, n_final, n_ckpt = [], 0, 0
        for s, ((o, f, ck), (o0, f0, ck0)) in enumerate(zip(got, first)):
            m = (_bits(o) != _bits(o0)).flatten(1).any(1)
            if bool(m.any()):
                diff_rows.append(m.nonzero()[:, 0].cpu().numpy() + cu[s])
                max_rel = max(max_rel, rel_errors(o, o0)["max_rel"])
            for h in (_bits(f) != _bits(f0)).flatten(1).any(1).nonzero()[:, 0].tolist():
                final_heads[f"{s}:{h}"] += 1
                n_final += 1
            for off, st in ck.items():
                heads = (_bits(st) != _bits(ck0[off])).flatten(1).any(1)
                if bool(heads.any()):
                    ckpt_slots[f"{s}@{off}"] += 1
                    n_ckpt += int(heads.sum())
        rows_hit = np.concatenate(diff_rows) if diff_rows else np.zeros(0, np.int64)
        if len(rows_hit) or n_final or n_ckpt:
            kind = row_kinds(rows_hit, cu)
            seq = np.searchsorted(cu, rows_hit, side="right") - 1
            for label, sel in (("tail", (kind & 1) > 0), ("next", (kind & 2) > 0), ("elsewhere", kind == 0)):
                kinds[label] += int(sel.sum())
                for s in seq[sel].tolist():
                    by_seq[s][label] += 1
            kinds["total"] += len(rows_hit)
            calls["outputs"] += bool(len(rows_hit))
            calls["finals"] += bool(n_final)
            calls["ckpts"] += bool(n_ckpt)
            per_call[c] = {"output_rows": len(rows_hit), "final_heads": n_final, "ckpt_heads": n_ckpt}
    unwritten = sum(not bool(torch.isfinite(st).all()) for _, _, ck in first for st in ck.values())
    return {"variant": variant, "buffers": buffers, "repeats": args.repeats, "lens": lens, "cuts": cuts,
            "ckpt_states": sum(len(ck) for _, _, ck in first), "ckpt_unwritten": unwritten, "tensormaps": maps,
            "calls_differing": dict(calls), "output_rows": dict(kinds), "output_rows_by_seq": [dict(c) for c in by_seq],
            "final_heads": dict(final_heads), "ckpt_slots": dict(ckpt_slots), "max_rel_output": max_rel,
            "per_call": per_call, "peak_gb": torch.cuda.max_memory_allocated() / 2 ** 30, "device": _device()}


def cmd_gdn_race(args: argparse.Namespace) -> int:
    """`--repeats` identical calls of FlashInfer's GDN prefill on sequences of mixed lengths (`--lens`, not multiples of
    64: each ends in a partial 64-row chunk) packed in one call, from fixed random inputs
    (tests/support/kernel_cases.py's `_gdn_seq`), in three variants (`--variants`): `ckpt`, its checkpointing variant
    (`state_checkpoints` every 64 tokens of every row, `use_cp` off: `_fi_call`); `plain`, the per-sequence route
    without checkpoints; `cut`, our cut rows (`gdn_segments`: `flashinfer_gdn_prefill` through `_gdn_call`, how our
    wrapper takes checkpoints), two checkpoints a row (`cut_points`). Per variant: the calls whose outputs, checkpoint
    states or final states differ from the first call's bits, where the differing output rows lie (`tail`: in their
    sequence's partial last chunk; `next`: among the rows an earlier sequence's partial last chunk reaches past its end;
    elsewhere), per sequence, the (sequence:head) final states and (sequence@offset) checkpoints that moved, and the
    first call's checkpoint states left unwritten (NaN). Each variant runs twice, each time in a fresh process: with
    FlashInfer's tensormap buffers as it allocates them (`native`) and with both pre-grown to 1 MiB before the first
    call (`grown`, `flashinfer.utils._get_cache_buf`): a race that `grown` removes points at the buffers' size
    (`smid`). A worker that dies (an illegal address, say) is reported with its log's tail. On FlashInfer 0.7.2
    the `ckpt` variant's bits move between calls on SM120, so our wrapper cuts rows; 0.7.3 fixes it
    (flashinfer-ai/flashinfer#7785)."""
    out = _out(args)
    d = out / "gdn-race"
    d.mkdir(parents=True, exist_ok=True)
    if args.worker:
        variant, buffers = args.worker.split(":")
        res = _race_worker(args, variant, buffers)
        (d / f"{variant}-{buffers}.json").write_text(json.dumps(_clean(res), indent=1) + "\n")
        return 0
    results, failed = [], 0
    for variant in names(args.variants):
        for buffers in names(args.buffers):
            tag = f"{variant}-{buffers}"
            path, log = d / f"{tag}.json", d / f"{tag}.log"
            path.unlink(missing_ok=True)
            cmd = [sys.executable, "-m", "tests.scripts.numerics_kernels", "gdn-race", "--worker",
                   f"{variant}:{buffers}", "--out", str(out), "--repeats", str(args.repeats), "--lens", args.lens,
                   "--seed", str(args.seed)]
            print(f"gdn-race: {tag} ({args.repeats} calls) ...", flush=True)
            t0 = time.perf_counter()
            with log.open("w") as f:
                try:
                    rc: Any = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                                             timeout=args.timeout).returncode
                except subprocess.TimeoutExpired:
                    rc = "timeout"
            if rc == 0 and path.exists():
                res = json.loads(path.read_text())
            else:
                failed += 1
                res = {"variant": variant, "buffers": buffers, "error": f"exit {rc}",
                       "log_tail": log.read_text(errors="replace").splitlines()[-25:]}
            res["seconds"] = round(time.perf_counter() - t0, 1)
            results.append(res)
    rows = []
    for r in results:
        if "error" in r:
            rows.append([r["variant"], r["buffers"], "-", "-", "-", "-", "-", "-", r["error"]])
            continue
        c, k = r["calls_differing"], r["output_rows"]
        ckpts = c.get("ckpts", 0) if r["ckpt_states"] else "-"
        where = f"{k.get('tail', 0)} / {k.get('next', 0)} / {k.get('elsewhere', 0)}"
        rows.append([r["variant"], r["buffers"], r["repeats"] - 1, c.get("outputs", 0), ckpts, c.get("finals", 0),
                     where, _e(r["max_rel_output"] or None), "ok"])
    table = md_table(f"Identical GDN prefill calls against the first (sequences {args.lens})",
                     ["variant", "tensormaps", "calls", "outputs differ", "checkpoints differ", "final states differ",
                      "differing rows: tail / next / elsewhere", "max rel", "status"], rows)
    print(table)
    _save(out, "gdn-race", {"lens": parse_ids(args.lens), "repeats": args.repeats, "results": results}, [table])
    return 1 if failed else 0


# -- gdn-accuracy ---------------------------------------------------------------------------------------------------


def _realistic_seq(n: int, seed: int) -> dict[str, Any]:
    """Random-realistic GDN inputs of n tokens (no state): q and k normalised per head, v standard normal, per head a
    decay rate log-uniform over [1e-4, 0.5] with a log-normal jitter per token (alpha = exp(−rate × jitter): about a
    quarter of the heads lose under 1e-3 a token), beta = sigmoid(N(bias, 1)) with a bias per head uniform over
    [−2, 2]."""
    import torch

    g = torch.Generator(device=DEV).manual_seed(seed)
    q = torch.randn(n, GDN_HK, GDN_D, generator=g, device=DEV)
    k = torch.randn(n, GDN_HK, GDN_D, generator=g, device=DEV)
    v = torch.randn(n, GDN_HV, GDN_D, generator=g, device=DEV)
    rate = torch.exp(torch.empty(GDN_HV, device=DEV).uniform_(math.log(1e-4), math.log(0.5), generator=g))
    jitter = torch.exp(0.5 * torch.randn(n, GDN_HV, generator=g, device=DEV))
    bias = torch.empty(GDN_HV, device=DEV).uniform_(-2.0, 2.0, generator=g)
    beta = torch.sigmoid(torch.randn(n, GDN_HV, generator=g, device=DEV) + bias)
    bf = torch.bfloat16
    return {"q": (q / q.norm(dim=-1, keepdim=True)).to(bf), "k": (k / k.norm(dim=-1, keepdim=True)).to(bf),
            "v": v.to(bf), "a": torch.exp(-rate * jitter), "b": beta}


def _initial_state(kind: str, seed: int, warm: int) -> Any:
    """`zero`; `random` (N(0, 0.1²), as `_gdn_seq`'s); `warm`: the fp32 state after `warm` realistic tokens from
    zero."""
    import torch

    if kind == "zero":
        return torch.zeros(GDN_HV, GDN_D, GDN_D, device=DEV)
    if kind == "random":
        g = torch.Generator(device=DEV).manual_seed(seed)
        return torch.randn(GDN_HV, GDN_D, GDN_D, generator=g, device=DEV) * 0.1
    if kind == "warm":
        x = _realistic_seq(warm, seed)
        x["s"] = torch.zeros(GDN_HV, GDN_D, GDN_D, device=DEV)
        return gdn_fp32([x])[0][1].clone()
    raise SystemExit(f"unknown initial state {kind!r} (zero, random, warm)")


def _load_inputs(path: str, layer: int | None) -> tuple[str, list[dict[str, Any]], list[Any] | None]:
    """GDN prefill inputs from a file: a dict of q, k [T, 16, 128], v [T, 48, 128], alpha, beta [T, 48] as the prefill
    kernel got them, optionally `init` (the initial states, [rows, 48, 128, 128]; [48, 128, 128] for one row), `rows`
    [(pos0, n)] (consecutive in the tensors; default one row) and `o` (the kernel's output [T, 48, 128] where they were
    captured); or a forced run's capture, whose `taps.core[--layer]` holds them (tests/scripts/numerics_forced.py). A
    row without a state starts from zero: a row with pos0 > 0 and no `init` is skipped. Returns the case's name, its
    rows, and the captured outputs per row. The file is memory-mapped (a capture with taps holds gigabytes, the layer a
    small part), loaded with torch's `weights_only`, and without it when it holds other objects (the file is the
    caller's own)."""
    import pickle

    import torch

    try:
        obj = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    except pickle.UnpicklingError:
        obj = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    name = Path(path).name
    if "taps" in obj:
        if layer is None:
            raise SystemExit("--inputs is a capture: name its GDN layer (--layer)")
        obj, name = obj["taps"]["core"][layer], f"{name}:layer{layer}"
    spec = [(int(p), int(n)) for p, n in obj.get("rows", [(0, obj["q"].shape[0])])]
    init = obj.get("init")
    if init is not None and init.dim() == 3:
        init = init[None]
    rows, outs, at = [], [], 0
    for i, (p0, n) in enumerate(spec):
        sl = slice(at, at + n)
        at += n
        if init is None and p0 > 0:
            print(f"--inputs: row {i} starts at position {p0} with no initial state: skipped", flush=True)
            continue
        rows.append({"q": obj["q"][sl].to(DEV, torch.bfloat16).contiguous(),
                     "k": obj["k"][sl].to(DEV, torch.bfloat16).contiguous(),
                     "v": obj["v"][sl].to(DEV, torch.bfloat16).contiguous(),
                     "a": obj["alpha"][sl].to(DEV, torch.float32).contiguous(),
                     "b": obj["beta"][sl].to(DEV, torch.float32).contiguous(),
                     "s": (init[i].to(DEV, torch.float32) if init is not None else
                           torch.zeros(GDN_HV, GDN_D, GDN_D, device=DEV)).contiguous()})
        outs.append(obj["o"][sl].to(DEV) if "o" in obj else None)
    if not rows:
        raise SystemExit("--inputs: no row to run")
    return name, rows, outs if all(o is not None for o in outs) else None


def _gdn_cases(args: argparse.Namespace) -> Iterator[tuple[str, Callable[[], tuple[list[dict], list | None]]]]:
    """gdn-accuracy's cases, each made when its turn comes (one case's inputs on the GPU at a time)."""
    if args.inputs:
        name, rows, outs = _load_inputs(args.inputs, args.layer)
        yield name, lambda: (rows, outs)
        return
    lens = parse_ids(args.lens)
    groups = [lens] if args.packed else [[n] for n in lens]
    for c, (kind, group) in enumerate(itertools.product(names(args.inits), groups)):
        def make(kind: str = kind, group: list[int] = group, c: int = c) -> tuple[list[dict], None]:
            rows = []
            for i, n in enumerate(group):
                x = _realistic_seq(n, args.seed * 1000 + 10 * c + i)
                x["s"] = _initial_state(kind, args.seed * 1000 + 10 * c + i + 5, args.warm)
                rows.append(x)
            return rows, None

        yield f"{'+'.join(map(str, group))}/{kind}", make


def _cp_info(lens: list[int]) -> dict[str, Any]:
    """What FlashInfer's "auto" would do with these rows in one call (its own host heuristic,
    `gdn_kernels/delta_rule_dsl/varlen_helper.py`): whether it takes the context-parallel route, and that route's
    chunk length."""
    import torch

    try:
        from flashinfer.gdn_kernels.delta_rule_dsl.varlen_helper import choose_cp_chunk_len_host, should_use_cp_host
    except ImportError:
        return {}
    p = torch.cuda.get_device_properties(0)
    cap, n, total = (p.major, p.minor), len(lens), sum(lens)
    return {"auto_takes_cp": bool(should_use_cp_host(n * GDN_HV, p.multi_processor_count, p.name,
                                                     device_capability=cap)),
            "cp_chunk_len": int(choose_cp_chunk_len_host(total, GDN_HV, p.multi_processor_count,
                                                         device_capability=cap, total_seqlen=total, num_seqs=n,
                                                         device_name=p.name))}


def _ref_check(row: dict[str, Any], m: int = 192) -> dict[str, Any]:
    """`gdn_fp32` against `models.qwen3_5.backend.torch_gdn_prefill` on a row's first m tokens."""
    import torch

    from llm_mini_infer.models.qwen3_5.backend import torch_gdn_prefill

    part = {x: row[x][:m] for x in KEYS}
    n = int(part["q"].shape[0])
    o, f, _ = gdn_fp32([{**part, "s": row["s"]}])[0]
    ft = row["s"][None].clone()  # the pool: one slot
    ot = torch_gdn_prefill(*(part[x] for x in KEYS), ft, torch.zeros(1, dtype=torch.int64, device=ft.device),
                           SimpleNamespace(pf_cu_host=np.asarray([0, n]), ckpt=None))
    return {"tokens": n, "final_max_abs": float((f - ft[0]).abs().max()),
            "outputs_equal_in_bf16": bool(torch.equal(o.to(ot.dtype), ot))}


def _gdn_errors(res: list[tuple], ref: list[tuple], edges: list[int], cuts: dict[int, list[int]]) -> dict[str, Any]:
    """A route's outputs against the reference per position band (positions from each row's start), its final states,
    and its checkpoint states at `cuts`."""
    import torch

    bands = {}
    for lo, hi in zip(edges, [*edges[1:], None]):
        xs = [o[lo:hi] for o, _, _ in res if o.shape[0] > lo]
        if xs:
            rs = [o[lo:hi] for o, _, _ in ref if o.shape[0] > lo]
            bands[f"{lo}-{hi - 1}" if hi else f"{lo}-end"] = rel_errors(torch.cat(xs), torch.cat(rs))
    have = [(res[i][2][off], ref[i][2][off]) for i, offs in cuts.items() for off in offs if off in res[i][2]]
    finals = [f for _, f, _ in res]
    return {"bands": bands, "all": rel_errors(torch.cat([o for o, _, _ in res]), torch.cat([o for o, _, _ in ref])),
            "final": None if finals[0] is None else rel_errors(torch.stack(finals),
                                                              torch.stack([f for _, f, _ in ref])),
            "ckpts": rel_errors(torch.stack([a for a, _ in have]), torch.stack([b for _, b in have])) if have
            else None}


def _gdn_pair(a: list[tuple], b: list[tuple]) -> dict[str, Any]:
    import torch

    oa, ob = torch.cat([o for o, _, _ in a]), torch.cat([o for o, _, _ in b])
    fa, fb = torch.stack([f for _, f, _ in a]), torch.stack([f for _, f, _ in b])
    return {"outputs_rel_rms": rel_errors(oa, ob)["rel_rms"], "outputs_bits_equal": bool(torch.equal(_bits(oa),
                                                                                                    _bits(ob))),
            "finals_rel_rms": rel_errors(fa, fb)["rel_rms"], "finals_bits_equal": bool(torch.equal(_bits(fa),
                                                                                                  _bits(fb)))}


def cmd_gdn_accuracy(args: argparse.Namespace) -> int:
    """FlashInfer's GDN prefill routes against the fp32 recurrence and each other. Inputs: sequences of `--lens`
    tokens, each alone in its call (so FlashInfer's `use_cp` "auto" would take its context-parallel route on a 188-SM
    card; `--packed`: all in one call), random-realistic (q and k normalised, per-head decay rates log-uniform over
    [1e-4, 0.5]: `_realistic_seq`), from a zero, random or warm initial state (`--inits`; warm: the fp32 state after
    `--warm` such tokens); or `--inputs FILE`, inputs as the kernel got them (a dict, or a forced run's capture with
    `--layer`; `_load_inputs`), its captured output (`o`) also compared. Routes (`--routes`): `cp` (use_cp=True), `seq`
    (use_cp=False, the per-sequence route: our wrapper), `ckpt` (the checkpointing variant), `cut` (our cut
    rows); checkpoints at two multiples of 64 a row (`cut_points`). Each route against `gdn_fp32` (checked against
    `torch_gdn_prefill` on the first case's first 192 tokens): relative RMS ‖x − ref‖ / ‖ref‖ and max |x − ref| /
    max |ref| of its outputs per position band (`--bands`, from each row's start), of its final states and of its
    checkpoint states; and each pair of routes: their relative difference and whether they agree bit for bit. Also
    what "auto" would take, and the context-parallel chunk length."""
    torch = _cuda()
    out = _out(args)
    result: dict[str, Any] = {"device": _device(), "settings": {k: v for k, v in vars(args).items() if k != "func"},
                              "cases": []}
    tables, errors = [], 0
    for name, make in _gdn_cases(args):
        rec: dict[str, Any] = {"case": name}
        try:
            _gdn_case(rec, make, names(args.routes), parse_ids(args.bands), check=not result["cases"])
        except Exception as exc:
            errors += 1
            rec.update(_failed(f"gdn-accuracy {name}", exc))
        result["cases"].append(rec)
        made = _gdn_tables(rec)
        tables += made
        print("\n\n".join(made), flush=True)
        _save(out, "gdn-accuracy", result, tables)
        torch.cuda.empty_cache()
    result["peak_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
    _save(out, "gdn-accuracy", result, tables)
    return 1 if errors else 0


def _gdn_case(rec: dict[str, Any], make: Callable[[], tuple[list[dict], list | None]], routes: list[str],
              edges: list[int], check: bool) -> None:
    """One gdn-accuracy case into `rec` (its tensors freed on return)."""
    rows, captured = make()
    lens = [int(r["q"].shape[0]) for r in rows]
    cuts = {i: cut_points(n) for i, n in enumerate(lens)}
    rec.update(lens=lens, cuts=cuts, **_cp_info(lens))
    if check:
        rec["ref_check"] = _ref_check(rows[0])
    t0 = time.perf_counter()
    ref = gdn_fp32(rows, cuts)
    rec["ref_seconds"] = round(time.perf_counter() - t0, 1)
    got = {route: _gdn_route(route, rows, cuts) for route in routes}
    rec["routes"] = {r: _gdn_errors(res, ref, edges, cuts) for r, res in got.items()}
    if captured is not None:  # the kernel's outputs where the inputs were captured: no state to compare
        rec["routes"]["captured"] = _gdn_errors([(o, None, {}) for o in captured], ref, edges, {})
    rec["pairs"] = {f"{a}~{b}": _gdn_pair(got[a], got[b]) for a, b in itertools.combinations(routes, 2)}


def _gdn_tables(rec: dict[str, Any]) -> list[str]:
    if "error" in rec:
        return [f"**{rec['case']}**: {rec['error']}"]
    bands = list(next(iter(rec["routes"].values()))["bands"])
    auto = {True: "cp", False: "seq"}.get(rec.get("auto_takes_cp"), "?")
    title = (f"GDN prefill {rec['case']} against fp32 (relative RMS; \"auto\" takes {auto}, the cp chunk "
             f"{rec.get('cp_chunk_len', '?')} tokens)")
    if "ref_check" in rec:
        c = rec["ref_check"]
        title += (f"; the fp32 reference against torch_gdn_prefill over {c['tokens']} tokens: final states within "
                  f"{c['final_max_abs']:.1e}, outputs {'equal' if c['outputs_equal_in_bf16'] else 'not equal'} in "
                  "BF16")
    rows = [[r, *(_e(e["bands"].get(b, {}).get("rel_rms")) for b in bands), _e(e["all"]["max_rel"]),
             _e(e["final"]["rel_rms"]) if e["final"] else "-", _e(e["ckpts"]["rel_rms"]) if e["ckpts"] else "-",
             e["all"]["nonfinite"]] for r, e in rec["routes"].items()]
    t1 = md_table(title, ["route", *bands, "max rel", "final state", "checkpoints", "non-finite"], rows)
    t2 = md_table(f"GDN prefill {rec['case']}: routes against each other",
                  ["pair", "outputs rel RMS", "outputs bit-equal", "final states rel RMS", "final states bit-equal"],
                  [[p, _e(v["outputs_rel_rms"]), v["outputs_bits_equal"], _e(v["finals_rel_rms"]),
                    v["finals_bits_equal"]] for p, v in rec["pairs"].items()])
    return [t1, t2]


# -- attn-accuracy --------------------------------------------------------------------------------------------------


def _self_check(fi: Any) -> dict[str, Any]:
    """FlashInfer's startup self-check (`kernels.flashinfer.self_check`) on this wrapper, as its runner makes it. The
    check keeps its first result for the process, so it is cleared first: each part checks its own wrapper (the sliding
    one's graph buffers included)."""
    import torch

    from llm_mini_infer.kernels import flashinfer as fl

    fl._checked = {}
    return fl.self_check(fi, torch.bfloat16)


class _AttnPart:
    """One layer kind's probe: its implementations, made once (their buffers, FlashInfer's workspace and counters
    carry over from case to case, as in a server), and per case a KV pool, page table and queries. `impls`: the
    batch-invariant one (ours) last; a `-graph` one runs on verify-only steps."""

    name = ""
    impls: tuple[str, ...] = ()
    fi: Any = None  # the FlashInfer instance whose counters the probe reads
    check: Any = None  # FlashInfer's self-check, as the runner makes it

    def __init__(self, args: argparse.Namespace, dev: Any, h: int, d: int, scale: float):
        from llm_mini_infer.runner.cuda.runner import default_decode_buckets

        self.args, self.dev, self.L = args, dev, 1 + args.k
        self.max_kv = args.max_model_len + self.L
        self.buckets = default_decode_buckets(args.max_num_seqs)
        self.h, self.d = h, d
        self.q_std = args.logit_std / (d ** 0.5 * KV_RMS * scale)  # logits q·k × scale of that spread
        self.q: Any = None
        self.bucket = 0

    def queries(self, case: AttnCase, seed: int) -> Any:
        import torch

        g = torch.Generator(device=DEV).manual_seed(seed + 1)
        return (torch.randn(int(case.nq.sum()), self.h, self.d, generator=g, device=DEV) * self.q_std).to(
            torch.bfloat16)

    def padded(self, n: int) -> tuple[int, Any]:
        """A decode bucket for n verify rows, and the queries with its padding rows' (zeros)."""
        import torch

        b = self.bucket = decode_bucket(n, self.buckets)
        return b, torch.cat([self.q, self.q.new_zeros(((b - n) * self.L, *self.q.shape[1:]))])

    def counters(self) -> dict[str, int] | None:
        return None if self.fi is None else {"replans": int(self.fi.replans), "kept": int(self.fi.kept),
                                             "unsplit": int(self.fi.unsplit)}

    def setup(self, case: AttnCase, seed: int) -> None:
        raise NotImplementedError

    def run(self, impl: str, case: AttnCase, true: np.ndarray) -> Any:
        raise NotImplementedError

    def reference(self, case: AttnCase, true: np.ndarray) -> Any:
        raise NotImplementedError

    def release(self) -> None:
        self.q = None


class _QwenFull(_AttnPart):
    """Qwen3.8's full layers: FlashInfer as the switch-off path runs it, after its self-check (`self_check`): eager
    plans (`FlashInferAttention.plan`, verify rows planned for their bound with `lb`, then `run_fixed`) and a decode
    bucket's CUDA-graph wrapper (`plan_graph` with `lb` over the bucket's rows, padding rows of 1 + k keys on the null
    page, then `fix_lengths`, `graph_page_indices` and the run: the runner's graph body, run eagerly, the kernel a
    graph captures since a graph plan always splits); our `attn_window.FullAttention` (the fixed split, the switch-on
    path); fp32 (`_window_ref` with no window)."""

    name = "qwen-full"
    impls = ("flashinfer", "flashinfer-graph", "attn_window")
    H, HKV, D = 24, 4, 256

    def __init__(self, args: argparse.Namespace, dev: Any):
        import torch

        from llm_mini_infer.kernels.attn512 import SPLIT_KEYS
        from llm_mini_infer.kernels.attn_window import FullAttention
        from llm_mini_infer.kernels.flashinfer import FlashInferAttention

        super().__init__(args, dev, self.H, self.D, self.D ** -0.5)
        self.max_pages = -(-self.max_kv // PS) + 2
        ns = SimpleNamespace(kv=None, page_size=PS)
        self.fi = FlashInferAttention(ns, self.H, self.HKV, self.D, torch.bfloat16, dev, max_rows=self.buckets[-1],
                                      max_pages=self.max_pages, qo_per_row=self.L)
        self.check = _self_check(self.fi)
        self.ours = FullAttention(ns, self.H, self.HKV, self.D, dev, max_tokens=args.max_tokens,
                                  max_rows=args.max_num_seqs, max_kv=self.max_kv, scale=self.D ** -0.5,
                                  qo_per_row=self.L, split_keys=SPLIT_KEYS)
        self.arange = torch.arange(self.max_pages, device=dev)

    def setup(self, case: AttnCase, seed: int) -> None:
        import torch.nn.functional as F

        from ..support.kernel_cases import _window_setup

        self.pool, table = _window_setup([int(x) for x in case.ub], self.HKV, self.D, PS, seed)
        self.table = F.pad(table, (0, self.max_pages - table.shape[1]))  # as wide as the graph wrappers' buffers
        self.ours.pools = SimpleNamespace(kv=self.pool[None], page_size=PS)
        self.q = self.queries(case, seed)

    def run(self, impl: str, case: AttnCase, true: np.ndarray) -> Any:
        import torch

        from llm_mini_infer.kernels.flashinfer import page_indices

        from ..support.kernel_cases import _run, _stage

        n, q, pool, L = len(case.nq), self.q, self.pool, self.L
        slots = np.arange(1, n + 1)
        if impl == "attn_window":
            rows = np.stack([slots, case.starts, case.nq, case.ub], axis=1)
            return _run(self.ours, self.table, rows, q, decode=case.n_dec, true=true)
        if impl == "flashinfer":
            pages = -(-case.ub // PS)
            indptr = np.concatenate([[0], np.cumsum(pages)])
            idx = page_indices(self.table, _dev64(slots), _dev64(pages), _dev64(indptr), int(pages.max()),
                               int(indptr[-1]), self.arange)
            w = self.fi.plan(None, parity=0, qo_indptr=_stage(case.qo_indptr), kv_indptr=_stage(indptr),
                             kv_indices=idx, last_page_len=_stage(case.ub - (pages - 1) * PS), kv_lens=_stage(case.ub),
                             max_q=int(case.nq.max()), lb=case.lb if case.uncertain else None)
            out = torch.full_like(q, float("nan"))
            if case.uncertain:
                self.fi.run_fixed(w, q, pool, out, n, _dev64(true))
            else:
                w.run(q, pool, out=out)
            return out
        b, qb = self.padded(n)
        kv, lb, tl_, sl = np.full(b, L), np.full(b, L), np.full(b, L), np.zeros(b, np.int64)
        kv[:n], lb[:n], tl_[:n], sl[:n] = case.ub, case.lb, true, slots
        pages = -(-kv // PS)
        indptr = np.concatenate([[0], np.cumsum(pages)])
        w = self.fi.plan_graph(None, rows=b, parity=0, qo_indptr=_stage(np.arange(b + 1) * L),
                               kv_indptr=_stage(indptr), last_page_len=_stage(kv - (pages - 1) * PS),
                               kv_lens=_stage(kv), max_q=L, lb=lb if (lb < kv).any() else None)
        self.fi.fix_lengths(w, b, _dev64(tl_))
        self.fi.graph_page_indices(b, self.table, _dev64(sl))
        ob = torch.full_like(qb, float("nan"))
        w.run(qb, pool, out=ob, enable_pdl=False)  # PDL off: the length rewrite is the kernel just before
        return ob[:q.shape[0]]

    def reference(self, case: AttnCase, true: np.ndarray) -> Any:
        import torch

        from ..support.kernel_cases import _window_ref

        ref = torch.empty(self.q.shape, dtype=torch.float32, device=DEV)
        for r, (s, m) in enumerate(zip(case.starts.tolist(), case.nq.tolist())):
            ref[s:s + m] = _window_ref(self.pool, self.table, r + 1, self.q[s:s + m].float() * self.D ** -0.5,
                                       int(true[r]) - m, 1 << 30, PS)
        return ref

    def release(self) -> None:
        self.pool = self.table = self.q = None


class _GemmaSliding(_AttnPart):
    """Gemma 4's sliding layers: `FlashInferWindowAttention` as the switch-off path runs it (`upper_bounds`, as under
    MTP), eagerly and in a decode bucket's wrapper (`plan_graph`, `g_first`, `fix_lengths`, `graph_window_indices`),
    each row named from the first window page of its lowest possible first query (the pages before it NaN, so a read
    of them shows); our `attn_window.WindowAttention` (split at multiples of 256 keys); fp32 (`_window_ref`)."""

    name = "gemma-sliding"
    impls = ("flashinfer", "flashinfer-graph", "attn_window")
    H, HKV, D, W = 32, 16, 256, 1024

    def __init__(self, args: argparse.Namespace, dev: Any):
        import torch

        from llm_mini_infer.kernels.attn_window import SPLIT_KEYS, WindowAttention
        from llm_mini_infer.kernels.flashinfer import FlashInferWindowAttention

        super().__init__(args, dev, self.H, self.D, 1.0)
        wpages = -(-(self.W + PS + 2 * self.L) // PS) + 1  # the runner's: a decode row's window pages at most
        ns = SimpleNamespace(wkv=None, page_size=PS)
        self.fi = FlashInferWindowAttention(ns, self.H, self.HKV, self.D, torch.bfloat16, dev, self.W,
                                            max_rows=self.buckets[-1], max_pages=wpages, qo_per_row=self.L,
                                            upper_bounds=True)
        self.check = _self_check(self.fi)
        self.ours = WindowAttention(ns, self.H, self.HKV, self.D, dev, self.W, max_tokens=args.max_tokens,
                                    max_rows=args.max_num_seqs, qo_per_row=self.L, split_keys=SPLIT_KEYS)
        self.arange = torch.arange(-(-(self.W + args.max_tokens) // PS) + 2, device=dev)

    def setup(self, case: AttnCase, seed: int) -> None:
        from ..support.kernel_cases import _window_setup

        self.pool, self.table = _window_setup([int(x) for x in case.ub], self.HKV, self.D, PS, seed)
        self.wlo = np.maximum(case.lb - case.nq - self.W + 1, 0) // PS
        for r, w in enumerate(self.wlo.tolist()):
            if w:  # released below the window: never read
                self.pool[self.table[r + 1, :w].long()] = float("nan")
        self.ours.pools = SimpleNamespace(wkv=self.pool[None], page_size=PS)
        self.q = self.queries(case, seed)

    def run(self, impl: str, case: AttnCase, true: np.ndarray) -> Any:
        import torch

        from llm_mini_infer.kernels.flashinfer import window_page_indices

        from ..support.kernel_cases import _stage, _window_run

        n, q, pool, L, wb = len(case.nq), self.q, self.pool, self.L, self.wlo * PS
        slots = np.arange(1, n + 1)
        if impl == "attn_window":
            rows = np.stack([slots, case.starts, case.nq, case.ub], axis=1)
            return _window_run(self.ours, pool, self.table, rows, q, decode=case.n_dec, true=true)
        if impl == "flashinfer":
            kvw = case.ub - wb
            pages = -(-kvw // PS)
            indptr = np.concatenate([[0], np.cumsum(pages)])
            idx = window_page_indices(self.table, _dev64(slots), _dev64(self.wlo), _dev64(pages), _dev64(indptr),
                                      int(pages.max()), int(indptr[-1]), self.arange)
            w = self.fi.plan(None, parity=0, qo_indptr=_stage(case.qo_indptr), kv_indptr=_stage(indptr),
                             kv_indices=idx, last_page_len=_stage(kvw - (pages - 1) * PS), kv_lens=_stage(kvw),
                             max_q=int(case.nq.max()), lb=case.lb - wb if case.uncertain else None)
            out = torch.full_like(q, float("nan"))
            if case.uncertain:
                self.fi.run_fixed(w, q, pool, out, n, _dev64(true - wb))
            else:
                w.run(q, pool, out=out)
            return out
        b, qb = self.padded(n)
        kvw, lb, named = np.full(b, L), np.full(b, L), np.full(b, L)
        first, sl = np.zeros(b, np.int64), np.zeros(b, np.int64)  # padding rows: slot 0's null page, column 0
        kvw[:n], lb[:n], named[:n], first[:n], sl[:n] = case.ub - wb, case.lb - wb, true - wb, self.wlo, slots
        pages = -(-kvw // PS)
        indptr = np.concatenate([[0], np.cumsum(pages)])
        w = self.fi.plan_graph(None, rows=b, parity=0, qo_indptr=_stage(np.arange(b + 1) * L),
                               kv_indptr=_stage(indptr), last_page_len=_stage(kvw - (pages - 1) * PS),
                               kv_lens=_stage(kvw), max_q=L, lb=lb if (lb < kvw).any() else None)
        self.fi.g_first[:b].copy_(_dev64(first))
        self.fi.fix_lengths(w, b, _dev64(named))
        self.fi.graph_window_indices(b, self.table, _dev64(sl))
        ob = torch.full_like(qb, float("nan"))
        w.run(qb, pool, out=ob, enable_pdl=False)
        return ob[:q.shape[0]]

    def reference(self, case: AttnCase, true: np.ndarray) -> Any:
        import torch

        from ..support.kernel_cases import _window_ref

        ref = torch.empty(self.q.shape, dtype=torch.float32, device=DEV)
        for r, (s, m) in enumerate(zip(case.starts.tolist(), case.nq.tolist())):
            ref[s:s + m] = _window_ref(self.pool, self.table, r + 1, self.q[s:s + m].float(), int(true[r]) - m,
                                       self.W, PS)
        return ref

    def release(self) -> None:
        self.pool = self.table = self.q = None


class _GemmaFull(_AttnPart):
    """Gemma 4's full layers on `attn512.Attn512`: the wave split (`split_keys` 0, the switch-off path's) eagerly and
    in a decode bucket's tables (`plan_graph`: the split counts of the replay's real rows), against the fixed split
    (`split_size` from `SPLIT_KEYS`, the switch-on path's) and fp32: SDPA's math backend over K = RoPE(V ⊙ k_norm)
    rounded as the kernel rounds it (`gemma_ops.k_from_v`, `attention_ref`), kept in fp32 (test_gemma_kernels.py's
    `_full_ref` rounds it to BF16)."""

    name = "gemma-full"
    impls = ("wave", "wave-graph", "fixed")
    H, HKV, D, ROT = 32, 4, 512, 64

    def __init__(self, args: argparse.Namespace, dev: Any):
        from llm_mini_infer.kernels.attn512 import SPLIT_KEYS, Attn512

        from ..support.kernel_cases import _Pools, _rope_tables, _Rot

        super().__init__(args, dev, self.H, self.D, 1.0)
        self.cos, self.sin = _rope_tables(self.D, self.ROT, self.max_kv + PS)
        kw = dict(max_tokens=args.max_tokens, max_rows=args.max_num_seqs, qo_per_row=self.L)
        self.wave = Attn512(_Pools(None, PS), _Rot(self.cos, self.sin), self.H, self.HKV, self.ROT, dev, **kw)
        self.fixed = Attn512(_Pools(None, PS), _Rot(self.cos, self.sin), self.H, self.HKV, self.ROT, dev,
                             split_keys=SPLIT_KEYS, max_kv=self.max_kv, **kw)

    def setup(self, case: AttnCase, seed: int) -> None:
        from ..support.kernel_cases import _full_case, _Pools

        rows = [(int(m), int(u)) for m, u in zip(case.nq, case.ub)]
        self.fkv, self.table, _, self.kw, _, _ = _full_case(rows, self.H, self.HKV, PS, seed)
        for a in (self.wave, self.fixed):
            a.pools = _Pools(self.fkv[None], PS)
        self.q = self.queries(case, seed)

    def run(self, impl: str, case: AttnCase, true: np.ndarray) -> Any:
        import torch

        from ..support.kernel_cases import _Layer, _stage

        n, q, L = len(case.nq), self.q, self.L
        rows = np.stack([np.arange(1, n + 1), case.starts, case.nq, case.ub], axis=1).astype(np.int32)
        if impl != "wave-graph":
            a = self.wave if impl == "wave" else self.fixed
            meta = SimpleNamespace(page_table=self.table, fix_len=case.uncertain, n_rows=n, kv_len_dev=_dev64(true))
            a.plan(meta, parity=0, rows=rows, stage=_stage, fixed_rows=case.n_dec)
            out = torch.full_like(q, float("nan"))
            a.run(meta, _Layer(self.kw), q, out)
            return out
        b, qb = self.padded(n)
        rb = np.zeros((b, 4), np.int32)
        rb[:, 1], rb[:, 2], rb[:, 3] = np.arange(b) * L, L, L  # padding rows: slot 0's null page, 1 + k keys
        rb[:n] = rows
        tb = np.full(b, L)
        tb[:n] = true
        meta = SimpleNamespace(page_table=self.table, fix_len=True, n_rows=b, kv_len_dev=_dev64(tb))
        self.wave.plan_graph(meta, bucket=b, rows=rb, stage=_stage, real=n)
        ob = torch.full_like(qb, float("nan"))
        self.wave.run(meta, _Layer(self.kw), qb, ob)
        return ob[:q.shape[0]]

    def reference(self, case: AttnCase, true: np.ndarray) -> Any:
        import torch
        from torch.nn.attention import SDPBackend, sdpa_kernel

        from llm_mini_infer.kernels import gemma_ops as gops

        ref = torch.empty(self.q.shape, dtype=torch.float32, device=DEV)
        with sdpa_kernel(SDPBackend.MATH):
            for r, (s, m) in enumerate(zip(case.starts.tolist(), case.nq.tolist())):
                t = int(true[r])
                v = self.fkv[self.table[r + 1, :-(-t // PS)].long()].reshape(-1, self.HKV, self.D)[:t]
                kpos = torch.arange(t, device=DEV)
                k = gops.k_from_v(v, self.kw, self.cos[kpos], self.sin[kpos])
                ref[s:s + m] = gops.attention_ref(self.q[s:s + m].float(), k.float(), v.float(), kpos[t - m:], kpos)
        return ref

    def release(self) -> None:
        self.fkv = self.table = self.kw = self.q = None


PARTS: dict[str, type[_AttnPart]] = {"qwen-full": _QwenFull, "gemma-sliding": _GemmaSliding,
                                     "gemma-full": _GemmaFull}


def _attn_errors(out: Any, ref: Any, case: AttnCase, bad: float) -> dict[str, Any]:
    """An implementation's output against fp32: per query index of the verify rows, over the prefill rows and over
    all; the (row, query) pairs whose own relative error passes `bad` (non-finite ones included)."""
    nd, n = case.n_dec, len(case.nq)
    res: dict[str, Any] = {"all": rel_errors(out, ref)}
    if nd:
        m = int(case.nq[0])
        o, r = out[:nd * m].float().view(nd, m, -1), ref[:nd * m].view(nd, m, -1)
        res["per_query"] = [rel_errors(o[:, j], r[:, j]) for j in range(m)]
        e = (o - r).norm(dim=-1) / r.norm(dim=-1).clamp_min(1e-30)
        over = ~(e <= bad)
        res.update(bad=int(over.sum()), bad_by_query=over.sum(0).tolist(),
                   worst_row_query=float(e.nan_to_num(nan=float("inf")).max()))
    if nd < n:
        s = int(case.starts[nd])
        res["prefill"] = rel_errors(out[s:], ref[s:])
    return res


def _attn_case(part: _AttnPart, case: AttnCase, mode: str, true: np.ndarray, bad: float) -> dict[str, Any]:
    import torch

    rec: dict[str, Any] = {"case": case.name, "true": mode, "rows": len(case.nq), "n_dec": case.n_dec,
                           "lb": case.lb, "ub": case.ub, "true_len": true, "impls": {}}
    ref = part.reference(case, true)
    outs = {}
    for impl in part.impls:
        if impl.endswith("-graph") and case.n_dec < len(case.nq):
            continue
        before = part.counters()
        o = part.run(impl, case, true)
        torch.cuda.synchronize()
        e = _attn_errors(o, ref, case, bad)
        if impl.startswith("flashinfer"):
            after = part.counters()
            e["counters"] = {k: after[k] - before[k] for k in after}
        if impl.endswith("-graph"):
            e["bucket"] = part.bucket
        rec["impls"][impl] = e
        outs[impl] = o
    ours = part.impls[-1]
    for impl, o in outs.items():
        if impl != ours:
            rec["impls"][impl][f"vs_{ours}"] = rel_errors(o, outs[ours])
    return rec


def _attn_table(part: str, recs: list[dict[str, Any]], L: int) -> str:
    rows = []
    for rec in recs:
        if "error" in rec:
            rows.append([rec["case"], "-", rec["error"], *["-"] * (L + 6)])
            continue
        for impl, e in rec["impls"].items():
            pq = e.get("per_query") or [{}] * L
            c = e.get("counters")
            vs = next((v for k, v in e.items() if k.startswith("vs_")), None)
            rows.append([rec["case"], rec["true"], impl, *(_e(x.get("rel_rms")) for x in pq),
                         _e(e["prefill"]["rel_rms"]) if "prefill" in e else "-", _e(e["all"]["max_rel"]),
                         e.get("bad", "-"), e["all"]["nonfinite"], f"{c['replans']}/{c['kept']}/{c['unsplit']}" if c
                         else "-", _e(vs["rel_rms"]) if vs else "-"])
    return md_table(f"{part}: relative RMS error against fp32 (verify rows per query index)",
                    ["case", "true", "impl", *(f"q{j}" for j in range(L)), "prefill", "max rel", "bad", "non-finite",
                     "replans/kept/unsplit", "vs ours"], rows)


def cmd_attn_accuracy(args: argparse.Namespace) -> int:
    """Attention at Qwen3.8's and Gemma 4's shapes: each part's implementations against fp32 and the batch-invariant
    one. Steps (`attn_cases`): verify rows of 1 + k queries, `--rows` of them, in lockstep and diverse, lowest lengths
    in [`--min-len`, `--max-len`], planned k above, run at true lengths (`--true`) `lb`, `rand` and `ub`; a 2,048-token
    prefill row from 0; prefill rows of 128, 256 and 512 tokens over 4,000-5,000 positions; 16 verify rows beside a
    512-token prefill chunk. Parts (`--parts`): `qwen-full` (FlashInfer eager plans and decode-graph wrappers, our
    `FullAttention`), `gemma-sliding` (`FlashInferWindowAttention` the same two ways, our `WindowAttention`),
    `gemma-full` (`Attn512`'s wave split eagerly and in a decode bucket, its fixed split). Per step and implementation:
    the relative RMS error per query index of the verify rows (a FlashInfer plan whose chunk count the true length
    changed merges other queries' partials into every query after the first), over the prefill rows, the max relative
    error, the (row, query) pairs whose own relative error passes `--bad`, non-finite values, FlashInfer's counters
    (`replans`, `kept`, `unsplit`) and the relative difference to ours. Keys and values are random with RMS 0.5 a dim;
    the queries' scale gives the logits q·k × scale a spread of `--logit-std`."""
    torch = _cuda(library=True)
    out = _out(args)
    dev = torch.device(DEV, torch.cuda.current_device())
    cases = attn_cases(parse_ids(args.rows), args.min_len, args.max_len, args.k, np.random.default_rng(args.seed))
    if max(int(c.ub.max()) for c in cases) > args.max_model_len:
        raise SystemExit(f"--max-len + --k past --max-model-len {args.max_model_len}")
    result: dict[str, Any] = {"device": _device(), "settings": {k: v for k, v in vars(args).items() if k != "func"},
                              "parts": {}}
    tables, errors = [], 0
    for pname in names(args.parts):
        recs: list[dict[str, Any]] = []
        result["parts"][pname] = {"cases": recs}
        try:
            part = PARTS[pname](args, dev)
        except Exception as exc:
            errors += 1
            result["parts"][pname].update(_failed(f"attn-accuracy {pname}", exc))
            continue
        result["parts"][pname]["self_check"] = part.check
        for ci, case in enumerate(cases):
            seed = args.seed * 1000 + ci
            rng = np.random.default_rng([args.seed, ci])
            try:
                part.setup(case, seed)
                for mode in names(args.true) if case.uncertain else ["exact"]:
                    recs.append(_attn_case(part, case, mode, case.true(mode, rng), args.bad))
                    print(f"{pname} {case.name} {recs[-1]['true']}: done", flush=True)
            except Exception as exc:
                errors += 1
                recs.append({"case": case.name, **_failed(f"attn-accuracy {pname} {case.name}", exc)})
            finally:
                part.release()
                torch.cuda.empty_cache()
            _save(out, "attn-accuracy", result, tables)
        table = _attn_table(pname, recs, part.L)
        print(table, flush=True)
        tables.append(table)
        del part
        torch.cuda.empty_cache()
    result["peak_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
    _save(out, "attn-accuracy", result, tables)
    return 1 if errors else 0


# -- nan-leak -------------------------------------------------------------------------------------------------------


def _nonfinite(res: list[tuple], lens: list[int]) -> tuple[list[int], list[int]]:
    """The sequences whose outputs (the first lens[i] rows) and whose final states hold a non-finite value."""
    import torch

    o_bad = [i for i, ((o, _, _), n) in enumerate(zip(res, lens)) if not bool(torch.isfinite(o[:n].float()).all())]
    return o_bad, [i for i, (_, f, _) in enumerate(res) if not bool(torch.isfinite(f).all())]


def cmd_nan_leak(args: argparse.Namespace) -> int:
    """Sequences of `--lens` tokens packed in one call (`_gdn_seq`'s inputs), a NaN in the first `--nan-rows` rows of
    one sequence at a time, in q, k or v (`--nan-in`; a and b, the decay and beta, also take), on the per-sequence route
    (our wrapper) and the context-parallel one (use_cp=True): which sequences' outputs and final states come out
    non-finite (the leak is flashinfer-ai/flashinfer#9372, still in 0.7.4). Then the candidate fix (`pad_row`): every
    row padded to a multiple of 64 inside the packed call (q = k = v = 0, decay 1, beta 0 in the padding rows; the
    padded rows' cu_seqlens; the outputs taken back): whether it keeps the NaN in its sequence, and whether without a
    NaN its outputs and final states equal the unpadded call's bit for bit, and by value (+0 = −0): it is usable on the
    batch-invariant path only if they do."""
    _cuda()
    from ..support.kernel_cases import _gdn_seq

    out = _out(args)
    lens = parse_ids(args.lens)
    base = [_gdn_seq(n, args.seed * 1000 + i) for i, n in enumerate(lens)]
    result: dict[str, Any] = {"device": _device(), "lens": lens, "lens_mod_64": [n % CHUNK for n in lens],
                              "nan_rows": args.nan_rows, "routes": {}}
    t_leak, t_equal = [], []
    for route in ("seq", "cp"):
        try:
            result["routes"][route] = _nan_route(route, base, lens, args, t_leak, t_equal)
        except Exception as exc:
            result["routes"][route] = _failed(f"nan-leak {route}", exc)
            t_equal.append([route, result["routes"][route]["error"], *["-"] * 5])
    tables = [md_table(f"A NaN in one sequence's first {args.nan_rows} rows (sequences {lens}, mod 64: "
                       f"{result['lens_mod_64']}): the sequences left non-finite",
                       ["route", "NaN in", "sequence", "outputs", "final states", "padded: outputs",
                        "padded: final states"], t_leak),
              md_table("Padded to multiples of 64 against unpadded, no NaN",
                       ["route", "outputs bit-equal", "final states bit-equal", "outputs equal (±0)",
                        "final states equal (±0)", "outputs max abs", "final states max abs"], t_equal)]
    print("\n\n".join(tables))
    _save(out, "nan-leak", result, tables)
    return 1 if any("error" in r for r in result["routes"].values()) else 0


def _nan_route(route: str, base: list[dict[str, Any]], lens: list[int], args: argparse.Namespace,
               t_leak: list[list[Any]], t_equal: list[list[Any]]) -> dict[str, Any]:
    """nan-leak on one route: the padded call against the unpadded one without NaN, then each NaN placement."""
    import torch

    clean = _gdn_route(route, base, {})
    padded = _gdn_route(route, [pad_row(r) for r in base], {})
    oc, op = torch.cat([o for o, _, _ in clean]), torch.cat([o[:n] for (o, _, _), n in zip(padded, lens)])
    fc, fp = torch.stack([f for _, f, _ in clean]), torch.stack([f for _, f, _ in padded])
    same = {"outputs_bits_equal": bool(torch.equal(_bits(oc), _bits(op))),
            "finals_bits_equal": bool(torch.equal(_bits(fc), _bits(fp))),
            "outputs_values_equal": bool(torch.equal(oc, op)), "finals_values_equal": bool(torch.equal(fc, fp)),
            "outputs_max_abs": float((oc.float() - op.float()).abs().max()),
            "finals_max_abs": float((fc - fp).abs().max())}
    t_equal.append([route, same["outputs_bits_equal"], same["finals_bits_equal"], same["outputs_values_equal"],
                    same["finals_values_equal"], _e(same["outputs_max_abs"]), _e(same["finals_max_abs"])])
    cases = []
    for key, j in itertools.product(names(args.nan_in), range(len(lens))):
        rows = [dict(r) for r in base]
        rows[j][key] = rows[j][key].clone()
        rows[j][key][:args.nan_rows] = float("nan")
        o_bad, s_bad = _nonfinite(_gdn_route(route, rows, {}), lens)
        po_bad, ps_bad = _nonfinite(_gdn_route(route, [pad_row(r) for r in rows], {}), lens)
        cases.append({"nan_in": key, "seq": j, "outputs_nonfinite": o_bad, "finals_nonfinite": s_bad,
                      "padded_outputs_nonfinite": po_bad, "padded_finals_nonfinite": ps_bad,
                      "leaks": any(i != j for i in o_bad + s_bad),
                      "padded_leaks": any(i != j for i in po_bad + ps_bad)})
        t_leak.append([route, key, j, o_bad, s_bad, po_bad, ps_bad])
    return {"padded_vs_unpadded": same, "cases": cases}


# -- timing ---------------------------------------------------------------------------------------------------------


def _timed(fn: Callable[[], Any], iters: int, warmup: int) -> dict[str, float]:
    """CUDA-event time of fn on the current stream: the median, p10 and p90 of `iters` calls after `warmup`."""
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
    for s, e in ev:
        s.record()
        fn()
        e.record()
    torch.cuda.synchronize()
    ms = np.sort([s.elapsed_time(e) for s, e in ev])
    return {"median_ms": float(np.median(ms)), "p10_ms": float(np.percentile(ms, 10)),
            "p90_ms": float(np.percentile(ms, 90))}


def cmd_timing(args: argparse.Namespace) -> int:
    """CUDA-event timings (`_timed`: median, p10, p90 of `--iters` calls after `--warmup`) of FlashInfer's GDN
    prefill as our wrapper calls it: checkpoint steps, its checkpointing variant (`_fi`, every 64 tokens) against our
    cut rows (`flashinfer_gdn_prefill` with `gdn_segments`) and against the per-sequence call without checkpoints (one
    2,048-token row with checkpoints at 704 and 1,408; eight rows of 128, each with one at 64; a mix of six rows); the
    same steps and one row of 2,048 on a layer's view of a state pool laid out as the runner's (16 slots of 48 GDN
    layers), our call (`state_indices`) against the states gathered and scattered around it (`_packed_reference`);
    one sequence of `--single` tokens, the per-sequence route against the context-parallel one; and the padded call
    (nan-leak's fix, its copies in and out included) against the unpadded one, on `--padded` mixes."""
    torch = _cuda()
    from llm_mini_infer.kernels.flashinfer import flashinfer_gdn_prefill, gdn_segments
    from llm_mini_infer.models.qwen3_5.backend import CkptMeta

    from ..support.kernel_cases import _gdn_seq, _packed_reference

    out = _out(args)
    result: dict[str, Any] = {"device": _device(), "iters": args.iters, "warmup": args.warmup, "cases": []}
    rows_t: list[list[Any]] = []
    states = torch.randn((16, 48, GDN_HV, GDN_D, GDN_D), device=DEV).mul_(0.1)
    pool = states[:, 7]  # a layer's view: its slots 48 layers apart

    def packed(lens: list[int], seed: int) -> tuple[dict[str, Any], Any, np.ndarray, Any]:
        rows = [_gdn_seq(n, seed + i) for i, n in enumerate(lens)]
        cu = np.concatenate([[0], np.cumsum(lens)])
        meta = SimpleNamespace(pf_cu=torch.as_tensor(cu, device=DEV), pf_cu_host=cu, ckpt=None)
        return ({x: torch.cat([r[x] for r in rows]).contiguous() for x in KEYS},
                torch.stack([r["s"] for r in rows]).contiguous(), cu, meta)

    def record(case: str, fns: dict[str, Callable[[], Any]], base: str) -> None:
        """Time each of `fns` (the first, `base`, the one the others are compared with); a case that raises is
        recorded and the run goes on."""
        try:
            times = {v: _timed(fn, args.iters, args.warmup) for v, fn in fns.items()}
        except Exception as exc:
            result["cases"].append({"case": case, **_failed(f"timing {case}", exc)})
            rows_t.append([case, "-", "-", "-", "-", result["cases"][-1]["error"]])
            return
        result["cases"].append({"case": case, "base": base, "times": times})
        for v, t in times.items():
            rows_t.append([case, v, f"{t['median_ms']:.3f}", f"{t['p10_ms']:.3f}", f"{t['p90_ms']:.3f}",
                           f"{t['median_ms'] / times[base]['median_ms']:.2f}"])

    def go(fn: Callable[..., Any], x: dict[str, Any], init: Any, meta: Any) -> Callable[[], Any]:
        ids = torch.arange(init.shape[0], device=DEV)  # init cloned as the pool: each call from the same states
        return lambda: fn(x["q"], x["k"], x["v"], x["a"], x["b"], init.clone(), ids, meta)

    def on_pool(fn: Callable[..., Any], x: dict[str, Any], n: int, meta: Any) -> Callable[[], Any]:
        ids = torch.arange(16 - n, 16, device=DEV)
        return lambda: fn(x["q"], x["k"], x["v"], x["a"], x["b"], pool, ids, meta)

    ckpt_cases = {"2048 with 2 checkpoints": ([2048], {0: [704, 1408]}),
                  "8 x 128 with 1 each": ([128] * 8, {i: [64] for i in range(8)}),
                  "mix of 6": ([1000, 128, 128, 300, 70, 400], {0: [512], 1: [64], 2: [64], 4: [64], 5: [128, 320]})}
    for lens in ([2048], [128] * 8):
        x, _, _, meta = packed(lens, args.seed * 1000 + len(lens))
        record(f"pool, {len(lens)} x {lens[0]}", {"gathered": on_pool(_packed_reference, x, len(lens), meta),
                                                  "state_indices": on_pool(flashinfer_gdn_prefill, x, len(lens), meta)},
               "gathered")
    for c, (case, (lens, cuts)) in enumerate(ckpt_cases.items()):
        x, init, cu, meta = packed(lens, args.seed * 1000 + 10 * c)
        starts = np.concatenate([[0], np.cumsum(np.asarray(lens) // CHUNK)])
        ws = torch.empty((int(starts[-1]), GDN_HV, GDN_D, GDN_D), device=DEV)
        want = sorted((r, o) for r, offs in cuts.items() for o in offs)
        src = np.asarray([starts[r] + o // CHUNK - 1 for r, o in want], np.int64)
        seg = gdn_segments(cu, np.asarray([r for r, _ in want]), np.asarray([o for _, o in want]), src, _dev64)
        m_cut = SimpleNamespace(**vars(meta))
        m_cut.ckpt = CkptMeta(every=CHUNK, cu_starts_host=starts, workspace=ws,
                              src=None, dst=None, conv=None, segments=seg)
        starts_dev = _dev64(starts)

        def variant(x: dict[str, Any] = x, init: Any = init, meta: Any = meta, ws: Any = ws,
                    starts: Any = starts_dev) -> Any:
            return _fi(x, init.clone(), meta.pf_cu, state_checkpoints=ws, checkpoint_cu_starts=starts,
                       checkpoint_every_n_tokens=CHUNK, use_cp=False)

        record(case, {"no checkpoints": go(flashinfer_gdn_prefill, x, init, meta),
                      "checkpointing variant": variant, "cut rows": go(flashinfer_gdn_prefill, x, init, m_cut)},
               "no checkpoints")
        record(f"pool, {case}", {"gathered": on_pool(_packed_reference, x, len(lens), m_cut),
                                 "state_indices": on_pool(flashinfer_gdn_prefill, x, len(lens), m_cut)}, "gathered")
    for n in parse_ids(args.single):
        x, init, cu, meta = packed([n], args.seed * 1000 + n)

        def cp(x: dict[str, Any] = x, init: Any = init, meta: Any = meta) -> Any:
            return _fi(x, init.clone(), meta.pf_cu, use_cp=True)

        record(f"1 x {n}", {"per-sequence": go(flashinfer_gdn_prefill, x, init, meta), "context-parallel": cp},
               "per-sequence")
    for spec in args.padded.split(";"):
        lens = parse_ids(spec)
        x, init, cu, meta = packed(lens, args.seed * 1000 + len(lens))
        pcu, valid = padded_index(lens)
        vd, t = _dev64(valid), int(pcu[-1])
        buf = {"q": x["q"].new_zeros((t, GDN_HK, GDN_D)), "k": x["k"].new_zeros((t, GDN_HK, GDN_D)),
               "v": x["v"].new_zeros((t, GDN_HV, GDN_D)), "a": x["a"].new_ones((t, GDN_HV)),
               "b": x["b"].new_zeros((t, GDN_HV))}
        pmeta = SimpleNamespace(pf_cu=_dev64(pcu), pf_cu_host=pcu, ckpt=None)

        ids = torch.arange(init.shape[0], device=DEV)

        def padded(x: dict[str, Any] = x, buf: dict[str, Any] = buf, init: Any = init, pmeta: Any = pmeta,
                   vd: Any = vd, ids: Any = ids) -> Any:
            for key in KEYS:
                buf[key].index_copy_(0, vd, x[key])
            return flashinfer_gdn_prefill(*(buf[key] for key in KEYS), init.clone(), ids, pmeta).index_select(0, vd)

        record(f"padded {lens}", {"unpadded": go(flashinfer_gdn_prefill, x, init, meta),
                                  "padded (kernel)": go(flashinfer_gdn_prefill, buf, init, pmeta),
                                  "padded (with copies)": padded}, "unpadded")
    table = md_table(f"GDN prefill timings (CUDA events, {args.iters} calls after {args.warmup})",
                     ["case", "variant", "median ms", "p10", "p90", "x first"], rows_t)
    print(table)
    _save(out, "timing", result, [table])
    return 1 if any("error" in c for c in result["cases"]) else 0


# -- attn-timing ----------------------------------------------------------------------------------------------------


def _graph(fn: Callable[[], Any]) -> Any:
    """`fn`'s launches captured in a CUDA graph, after a warm-up on a side stream (as torch requires)."""
    import torch

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    return g


class _TimedPart:
    """One layer kind's timing probe: the switch-off path (`off`) and the batch-invariant one (`on`, per prefill split
    base it is timed at), made once, and per case a KV pool, page table and queries. `prepare` plans one of them for the
    case, in a `mode` (eager: the step's own plan; bucket: the decode bucket's plan, run eagerly; graph: the same, for
    the caller to capture), and returns the callable that launches the kernels, nothing else: the output and the
    padded queries are made before it."""

    name = ""
    H = HKV = D = 0
    splits_prefill = True  # whether the batch-invariant path splits prefill rows (`timing_variants`)
    real_rows = True  # whether its plan_graph takes the real rows

    def __init__(self, args: argparse.Namespace, dev: Any, sizes: SimpleNamespace):
        self.args, self.dev, self.sizes, self.L = args, dev, sizes, 1 + args.k
        self.ons: dict[int, Any] = {}
        self.pool = self.table = self.q = self.out = self.pools = self.layer = self._bucket = None

    def off_split_keys(self) -> int | None:
        return None

    def on_split_keys(self) -> int:
        from llm_mini_infer.kernels.attn512 import SPLIT_KEYS

        return SPLIT_KEYS

    def make(self, prefill_keys: int | None) -> Any:
        raise NotImplementedError

    def prefill(self, prefill_keys: int | None) -> dict[str, int]:
        return {"prefill_from": self.args.prefill_from, "prefill_keys": prefill_keys or self.prefill_default()}

    def prefill_default(self) -> int:
        from llm_mini_infer.kernels.attn512 import PREFILL_KEYS

        return PREFILL_KEYS

    def on(self, prefill_keys: int | None) -> Any:
        a = self.ons.get(prefill_keys)
        if a is None:
            a = self.ons[prefill_keys] = self.make(prefill_keys)
        a.pools = self.pools
        return a

    def instances(self) -> list[Any]:
        return list(self.ons.values())

    def pools_of(self, kv: Any) -> Any:
        raise NotImplementedError

    def build(self, case: AttnCase, seed: int) -> None:
        """The case's pool, page table and queries."""
        raise NotImplementedError

    def queries(self, case: AttnCase, seed: int) -> Any:
        import torch

        g = torch.Generator(device=DEV).manual_seed(seed + 1)
        return (torch.randn(int(case.nq.sum()), self.H, self.D, generator=g, device=DEV) * 0.1).to(torch.bfloat16)

    def setup(self, case: AttnCase, seed: int) -> None:
        import torch

        self._bucket = None
        self.build(case, seed)
        self.out = torch.empty_like(self.q)
        for a in self.instances():
            a.pools = self.pools

    def release(self) -> None:
        for a in self.instances():
            a.pools = None
        self.pool = self.table = self.q = self.out = self.pools = self._bucket = None

    def meta(self, rows: int) -> SimpleNamespace:
        return SimpleNamespace(page_table=self.table, wpage_table=self.table, fix_len=False, n_rows=rows,
                               kv_len_dev=None, bidi_plan=None)

    def bucket(self, case: AttnCase) -> tuple[int, np.ndarray, Any, Any]:
        """The decode bucket of the case's rows: its size, its row table (the padding rows on slot 0's null page with
        1 + k keys), the queries with the padding rows' (zeros) and the output."""
        if self._bucket is None:
            n, L = len(case.nq), self.L
            b = decode_bucket(n, self.sizes.buckets)
            rb = np.zeros((b, 4), np.int32)
            rb[:, 1], rb[:, 2], rb[:, 3] = np.arange(b) * L, L, L
            rb[:n] = np.stack([np.arange(1, n + 1), case.starts, case.nq, case.ub], axis=1)
            qb = self.q.new_zeros((b * L, *self.q.shape[1:]))
            qb[:self.q.shape[0]] = self.q
            self._bucket = (b, rb, qb, self.q.new_empty(qb.shape))
        return self._bucket

    def drive(self, a: Any, case: AttnCase, mode: str) -> Callable[[], Any]:
        """Plan one of our backends (`a`) for the case and return its launch."""
        n = len(case.nq)
        if mode == "eager":
            rows = np.stack([np.arange(1, n + 1), case.starts, case.nq, case.ub], axis=1).astype(np.int32)
            meta, q, out = self.meta(n), self.q, self.out
            a.plan(meta, parity=0, rows=rows, stage=_timing_stage, fixed_rows=case.n_dec)
        else:
            b, rb, q, out = self.bucket(case)
            meta = self.meta(b)
            a.plan_graph(meta, bucket=b, rows=rb, stage=_timing_stage, **({"real": n} if self.real_rows else {}))
        return lambda: a.run(meta, self.layer, q, out)

    def off(self, case: AttnCase, mode: str) -> Callable[[], Any]:
        raise NotImplementedError

    def prepare(self, impl: str, case: AttnCase, mode: str, prefill_keys: int | None) -> Callable[[], Any]:
        return self.drive(self.on(prefill_keys), case, mode) if impl == "on" else self.off(case, mode)


def _timing_stage(a: np.ndarray) -> Any:
    from ..support.kernel_cases import _stage

    return _stage(a)


class _TimedQwen(_TimedPart):
    """Qwen3.8's full layers: `FlashInferAttention` (eager plan, and the decode bucket's graph wrapper as the runner
    plans it) against `attn_window.FullAttention`."""

    name = "qwen-full"
    H, HKV, D = 24, 4, 256

    def __init__(self, args: argparse.Namespace, dev: Any, sizes: SimpleNamespace):
        import torch

        from llm_mini_infer.kernels.flashinfer import FlashInferAttention

        super().__init__(args, dev, sizes)
        self.max_pages = -(-sizes.max_kv // PS) + 2
        self.fi = FlashInferAttention(SimpleNamespace(kv=None, page_size=PS), self.H, self.HKV, self.D,
                                      torch.bfloat16, dev, max_rows=sizes.max_rows, max_pages=self.max_pages,
                                      qo_per_row=self.L)
        self.arange = torch.arange(self.max_pages, device=dev)
        self.layer = 0

    def prefill_default(self) -> int:
        from llm_mini_infer.kernels.attn_window import FULL_PREFILL_KEYS

        return FULL_PREFILL_KEYS

    def make(self, prefill_keys: int | None) -> Any:
        from llm_mini_infer.kernels.attn_window import FullAttention

        return FullAttention(None, self.H, self.HKV, self.D, self.dev, max_tokens=self.sizes.max_tokens,
                             max_rows=self.sizes.max_rows, max_kv=self.sizes.max_kv, scale=self.D ** -0.5,
                             qo_per_row=self.L, split_keys=self.on_split_keys(), **self.prefill(prefill_keys))

    def build(self, case: AttnCase, seed: int) -> None:
        import torch.nn.functional as F

        from ..support.kernel_cases import _window_setup

        self.pool, table = _window_setup([int(x) for x in case.ub], self.HKV, self.D, PS, seed)
        self.table = F.pad(table, (0, self.max_pages - table.shape[1]))  # as wide as the graph wrappers' buffers
        self.pools = SimpleNamespace(kv=self.pool[None], page_size=PS)
        self.q = self.queries(case, seed)

    def off(self, case: AttnCase, mode: str) -> Callable[[], Any]:
        from llm_mini_infer.kernels.flashinfer import page_indices

        n, L, fi, pool = len(case.nq), self.L, self.fi, self.pool
        slots = np.arange(1, n + 1)
        if mode == "eager":
            pages = -(-case.ub // PS)
            indptr = np.concatenate([[0], np.cumsum(pages)])
            idx = page_indices(self.table, _dev64(slots), _dev64(pages), _dev64(indptr), int(pages.max()),
                               int(indptr[-1]), self.arange)
            w = fi.plan(None, parity=0, qo_indptr=_timing_stage(case.qo_indptr), kv_indptr=_timing_stage(indptr),
                        kv_indices=idx, last_page_len=_timing_stage(case.ub - (pages - 1) * PS),
                        kv_lens=_timing_stage(case.ub), max_q=int(case.nq.max()))
            q, out = self.q, self.out
            return lambda: w.run(q, pool, out=out)
        b, _, qb, ob = self.bucket(case)
        kv, sl = np.full(b, L), np.zeros(b, np.int64)  # padding rows: slot 0's null page, 1 + k keys
        kv[:n], sl[:n] = case.ub, slots
        pages = -(-kv // PS)
        indptr = np.concatenate([[0], np.cumsum(pages)])
        w = fi.plan_graph(None, rows=b, parity=0, qo_indptr=_timing_stage(np.arange(b + 1) * L),
                          kv_indptr=_timing_stage(indptr), last_page_len=_timing_stage(kv - (pages - 1) * PS),
                          kv_lens=_timing_stage(kv), max_q=L)
        fi.fix_lengths(w, b, _dev64(kv))
        fi.graph_page_indices(b, self.table, _dev64(sl))
        return lambda: w.run(qb, pool, out=ob, enable_pdl=False)


class _TimedSliding(_TimedPart):
    """Gemma 4's sliding layers: `FlashInferWindowAttention` (eager plan, and the decode bucket's graph wrapper, each
    row named from its first window page) against `attn_window.WindowAttention`, whose split does not follow M."""

    name = "gemma-sliding"
    H, HKV, D, W = 32, 16, 256, 1024
    splits_prefill = False
    real_rows = False

    def __init__(self, args: argparse.Namespace, dev: Any, sizes: SimpleNamespace):
        import torch

        from llm_mini_infer.kernels.flashinfer import FlashInferWindowAttention

        super().__init__(args, dev, sizes)
        wpages = -(-(self.W + PS + 2 * self.L) // PS) + 1  # the runner's: a decode row's window pages at most
        self.fi = FlashInferWindowAttention(SimpleNamespace(wkv=None, page_size=PS), self.H, self.HKV, self.D,
                                            torch.bfloat16, dev, self.W, max_rows=sizes.max_rows, max_pages=wpages,
                                            qo_per_row=self.L, upper_bounds=True)
        self.arange = torch.arange(-(-(self.W + sizes.max_tokens) // PS) + 2, device=dev)
        self.layer = SimpleNamespace(index=0)

    def on_split_keys(self) -> int:
        from llm_mini_infer.kernels.attn_window import SPLIT_KEYS

        return SPLIT_KEYS

    def make(self, prefill_keys: int | None) -> Any:
        from llm_mini_infer.kernels.attn_window import SPLIT_KEYS, WindowAttention

        return WindowAttention(None, self.H, self.HKV, self.D, self.dev, self.W, max_tokens=self.sizes.max_tokens,
                               max_rows=self.sizes.max_rows, qo_per_row=self.L, split_keys=SPLIT_KEYS)

    def build(self, case: AttnCase, seed: int) -> None:
        from ..support.kernel_cases import _window_setup

        self.pool, self.table = _window_setup([int(x) for x in case.ub], self.HKV, self.D, PS, seed)
        self.pools = SimpleNamespace(wkv=self.pool[None], page_size=PS)
        self.q = self.queries(case, seed)

    def off(self, case: AttnCase, mode: str) -> Callable[[], Any]:
        from llm_mini_infer.kernels.flashinfer import window_page_indices

        n, L, fi, pool = len(case.nq), self.L, self.fi, self.pool
        slots = np.arange(1, n + 1)
        wlo = np.maximum(case.ub - case.nq - self.W + 1, 0) // PS  # each row's first window page
        kvw = case.ub - wlo * PS
        if mode == "eager":
            pages = -(-kvw // PS)
            indptr = np.concatenate([[0], np.cumsum(pages)])
            idx = window_page_indices(self.table, _dev64(slots), _dev64(wlo), _dev64(pages), _dev64(indptr),
                                      int(pages.max()), int(indptr[-1]), self.arange)
            w = fi.plan(None, parity=0, qo_indptr=_timing_stage(case.qo_indptr), kv_indptr=_timing_stage(indptr),
                        kv_indices=idx, last_page_len=_timing_stage(kvw - (pages - 1) * PS),
                        kv_lens=_timing_stage(kvw), max_q=int(case.nq.max()))
            q, out = self.q, self.out
            return lambda: w.run(q, pool, out=out)
        b, _, qb, ob = self.bucket(case)
        kv, first, sl = np.full(b, L), np.zeros(b, np.int64), np.zeros(b, np.int64)
        kv[:n], first[:n], sl[:n] = kvw, wlo, slots  # padding rows: slot 0's null page, column 0
        pages = -(-kv // PS)
        indptr = np.concatenate([[0], np.cumsum(pages)])
        w = fi.plan_graph(None, rows=b, parity=0, qo_indptr=_timing_stage(np.arange(b + 1) * L),
                          kv_indptr=_timing_stage(indptr), last_page_len=_timing_stage(kv - (pages - 1) * PS),
                          kv_lens=_timing_stage(kv), max_q=L)
        fi.g_first[:b].copy_(_dev64(first))
        fi.fix_lengths(w, b, _dev64(kv))
        fi.graph_window_indices(b, self.table, _dev64(sl))
        return lambda: w.run(qb, pool, out=ob, enable_pdl=False)


class _TimedGemmaFull(_TimedPart):
    """Gemma 4's full layers on `attn512.Attn512`: the wave split (`split_keys` 0) against the fixed split."""

    name = "gemma-full"
    H, HKV, D, ROT = 32, 4, 512, 64

    def __init__(self, args: argparse.Namespace, dev: Any, sizes: SimpleNamespace):
        from ..support.kernel_cases import _rope_tables

        super().__init__(args, dev, sizes)
        self.cos, self.sin = _rope_tables(self.D, self.ROT, sizes.max_kv + PS)
        self.wave = self.make(None, fixed=False)  # the switch-off path

    def off_split_keys(self) -> int | None:
        return 0

    def instances(self) -> list[Any]:
        return [*super().instances(), self.wave]

    def make(self, prefill_keys: int | None, fixed: bool = True) -> Any:
        """The fixed split at a prefill base (None or 0: the default), or the wave split (`fixed` False)."""
        from llm_mini_infer.kernels.attn512 import Attn512

        from ..support.kernel_cases import _Pools, _Rot

        return Attn512(_Pools(None, PS), _Rot(self.cos, self.sin), self.H, self.HKV, self.ROT, self.dev,
                       max_tokens=self.sizes.max_tokens, max_rows=self.sizes.max_rows, qo_per_row=self.L,
                       split_keys=self.on_split_keys() if fixed else 0, max_kv=self.sizes.max_kv,
                       **self.prefill(prefill_keys))

    def build(self, case: AttnCase, seed: int) -> None:
        from ..support.kernel_cases import _full_case, _Layer, _Pools

        rows = [(int(m), int(u)) for m, u in zip(case.nq, case.ub)]
        fkv, self.table, self.q, kw, _, _ = _full_case(rows, self.H, self.HKV, PS, seed)
        self.pool, self.pools, self.layer = fkv, _Pools(fkv[None], PS), _Layer(kw)

    def off(self, case: AttnCase, mode: str) -> Callable[[], Any]:
        return self.drive(self.wave, case, mode)

    def release(self) -> None:
        super().release()
        self.layer = None


TIMED_PARTS: dict[str, type[_TimedPart]] = {"qwen-full": _TimedQwen, "gemma-sliding": _TimedSliding,
                                            "gemma-full": _TimedGemmaFull}


def _time_case(part: _TimedPart, case: AttnCase, args: argparse.Namespace, recs: list[dict[str, Any]]) -> int:
    """Time the switch-off path and the batch-invariant one at each prefill split base of `case` (`timing_variants`),
    in each mode, into `recs`; the number of variants that raised."""
    import torch

    n = len(case.nq)
    base = {"case": case.name, "rows": n, "nq": int(case.nq[0]), "history": case_history(case),
            "kv_len": int(case.ub.max())}
    modes = ("eager", "bucket", "graph") if case.n_dec else ("eager",)
    errors = 0

    def variant(impl: str, sp: int | None, mode: str, split_keys: int | None) -> None:
        nonlocal errors
        rec = {**base, "impl": impl, "sp": sp, "mode": mode, "split_keys": split_keys}
        recs.append(rec)
        try:
            fn = part.prepare(impl, case, mode, sp)
            torch.cuda.synchronize()
            g = _graph(fn) if mode == "graph" else None
            t = _timed(g.replay if g else fn, args.iters, args.warmup)
            rec.update(median_us=t["median_ms"] * 1e3, p10_us=t["p10_ms"] * 1e3, p90_us=t["p90_ms"] * 1e3)
            del g
        except Exception as exc:
            errors += 1
            rec.update(_failed(f"attn-timing {part.name} {case.name} {impl} {mode}", exc))
        torch.cuda.synchronize()

    for mode in modes:
        variant("off", None, mode, part.off_split_keys())
    for sp in timing_variants(case, parse_ids(args.prefill_keys), part.splits_prefill):
        for mode in modes:
            variant("on", sp, mode, part.on_split_keys())
    return errors


def _timing_table(part: str, recs: list[dict[str, Any]], args: argparse.Namespace) -> str:
    def us(r: dict[str, Any], key: str) -> str:
        return f"{r[key]:.1f}" if key in r else "-"

    rows = []
    for r in recs:
        if "case" not in r:
            continue
        impl = "off" if r["impl"] == "off" else "on" if not r["sp"] else f"on, Sp {r['sp']}"
        note = r.get("error", "")[:60]
        rows.append([r["case"], r["rows"], r["nq"], r["history"], impl, r["mode"], r["split_keys"],
                     note or us(r, "median_us"), us(r, "p10_us"), us(r, "p90_us"),
                     f"{r['ratio']:.2f}" if "ratio" in r else "-"])
    return md_table(f"{part}: attention kernel time per call, batch-invariant (on; a prefill step at each prefill "
                    f"split base Sp past {args.prefill_from:,} keys) against switch-off (off; "
                    f"CUDA events, {args.iters} calls after {args.warmup}; the graph rows replay a CUDA graph)",
                    ["case", "rows", "nq", "history", "impl", "mode", "split_keys", "median us", "p10", "p90",
                     "on/off"], rows)


def cmd_attn_timing(args: argparse.Namespace) -> int:
    """CUDA-event timings (`_timed`: median, p10 and p90 of `--iters` calls after `--warmup`) of the attention kernels
    of the full layers (`--parts`: `qwen-full`, `gemma-full`; `gemma-sliding` too), the batch-invariant path (on: our
    kernels, the fixed split, a decode query's from its own position, a prefill query's past `--prefill-from` keys at
    each base of `--prefill-keys`) against the switch-off one (off: FlashInfer, or `Attn512`'s wave split), each sized
    for rows of `--max-model-len` + 1 + k keys. Steps (`timing_cases`, exact KV lengths): one prefill row of each of
    `--short-chunks` queries over each of `--histories` positions; one of `--long-chunk` over each of
    `--long-histories`; per `--decode-rows` count, verify rows of 1 + k queries (`--k`) over each of
    `--decode-histories`. The plan is
    made once per path, outside the timed region, and only the kernels' launches are timed: eagerly, and for the
    verify rows in their decode bucket (`bucket`: its plan, run eagerly; `graph`: replayed from a CUDA graph, the
    padding rows on the null page). Per step, path and mode: the median, p10 and p90 in microseconds and the ratio of
    on to off. The eager rows of a kernel of a few microseconds count its launch's host time too: read the decode
    steps' graph rows. Needs room for the pools: 64 rows of 32k positions are about 8 GiB on either model's full
    layers, and their setup draws them in fp32 first (about 4 times that at its peak); the sliding layers' are 4
    times as large, past a 96 GB card at that step."""
    torch = _cuda(library=True)
    from llm_mini_infer.runner.cuda.runner import default_decode_buckets

    out = _out(args)
    dev = torch.device(DEV, torch.cuda.current_device())
    k = args.k
    cases = timing_cases(parse_ids(args.short_chunks), parse_ids(args.histories), args.long_chunk,
                         parse_ids(args.long_histories), parse_ids(args.decode_rows),
                         parse_ids(args.decode_histories), k)
    if max(c.n_dec for c in cases) > args.max_num_seqs:
        raise SystemExit(f"--decode-rows past --max-num-seqs {args.max_num_seqs}")
    buckets = default_decode_buckets(args.max_num_seqs)
    sizes = SimpleNamespace(
        buckets=buckets, max_rows=max([decode_bucket(c.n_dec, buckets) for c in cases if c.n_dec] + [1]),
        max_tokens=max(int(c.nq.sum()) for c in cases),
        max_kv=max(args.max_model_len + 1 + k, max(int(c.ub.max()) for c in cases)))
    result: dict[str, Any] = {"device": _device(), "settings": {x: v for x, v in vars(args).items() if x != "func"},
                              "parts": {}}
    tables, errors = [], 0
    for pname in names(args.parts):
        recs: list[dict[str, Any]] = []
        result["parts"][pname] = {"records": recs}
        try:
            part = TIMED_PARTS[pname](args, dev, sizes)
        except Exception as exc:
            errors += 1
            result["parts"][pname].update(_failed(f"attn-timing {pname}", exc))
            continue
        for ci, case in enumerate(cases):
            try:
                part.setup(case, args.seed * 1000 + ci)
                errors += _time_case(part, case, args, recs)
                print(f"{pname} {case.name}: done", flush=True)
            except Exception as exc:
                errors += 1
                recs.append({"case": case.name, **_failed(f"attn-timing {pname} {case.name}", exc)})
            finally:
                part.release()
                torch.cuda.empty_cache()
            timing_ratios(recs)
            _save(out, "attn-timing", result, tables)
        table = _timing_table(pname, recs, args)
        print(table, flush=True)
        tables.append(table)
        del part
        torch.cuda.empty_cache()
    result["peak_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
    _save(out, "attn-timing", result, tables)
    return 1 if errors else 0


# -- the command line -----------------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", help="results directory (default: results/numerics/kernels/<UTC time>)")
    seeded = argparse.ArgumentParser(add_help=False, parents=[common])
    seeded.add_argument("--seed", type=int, default=0, help="the random inputs' seed")
    ap = argparse.ArgumentParser(prog="python3 -m tests.scripts.numerics_kernels", description=__doc__.split("\n\n")[0],
                                 formatter_class=HelpFormatter,
                                 epilog="`<subcommand> --help` describes each; the module docstring, the whole.")
    sub = ap.add_subparsers(dest="command", required=True)

    def add(name: str, fn: Callable[[argparse.Namespace], int], what: str,
            seed: bool = True) -> argparse.ArgumentParser:
        p = sub.add_parser(name, parents=[seeded if seed else common], description=inspect.cleandoc(fn.__doc__ or ""),
                           help=what, formatter_class=HelpFormatter)
        p.set_defaults(func=fn)
        return p

    p = add("smid", cmd_smid, "the SM ids CTAs run on, against the SM count", seed=False)
    p.add_argument("--blocks-per-sm", type=int, default=64, help="CTAs a launch, per SM")
    p.add_argument("--launches", type=int, default=10, help="launches, each recording every CTA")
    p.add_argument("--spin-us", type=float, default=20.0, help="each CTA's spin before it records")
    p = add("gdn-race", cmd_gdn_race, "identical GDN prefill calls: which bits move, where")
    p.add_argument("--repeats", type=int, default=300, help="identical calls a variant (the first the reference)")
    p.add_argument("--lens", default="1000,300,2000,70,700", help="the packed sequences' lengths")
    p.add_argument("--variants", default="ckpt,plain,cut",
                   help="ckpt (the checkpointing variant), plain (the per-sequence route), cut (our cut rows)")
    p.add_argument("--buffers", default="native,grown", help="FlashInfer's tensormap buffers as allocated, pre-grown")
    p.add_argument("--timeout", type=float, default=1800.0, help="seconds a worker process")
    p.add_argument("--worker", help=argparse.SUPPRESS)  # variant:buffers, in a worker process
    p = add("gdn-accuracy", cmd_gdn_accuracy, "GDN prefill routes against the fp32 recurrence")
    p.add_argument("--lens", default="500,1000,2000,4000,8000", help="sequence lengths, one case each")
    p.add_argument("--packed", action="store_true", help="all --lens in one call instead")
    p.add_argument("--inits", default="zero,warm", help="initial states: zero, random, warm")
    p.add_argument("--warm", type=int, default=1024, help="tokens a warm initial state has seen")
    p.add_argument("--routes", default="cp,seq,ckpt,cut",
                   help="cp (context-parallel), seq (per-sequence), ckpt (the checkpointing variant), cut (cut rows)")
    p.add_argument("--bands", default="0,64,512,2048,4096", help="position bands' lower edges")
    p.add_argument("--inputs", help="inputs from a file (a dict, or a capture with --layer) instead of random ones")
    p.add_argument("--layer", type=int, help="the GDN layer of a capture's taps.core")
    p = add("attn-accuracy", cmd_attn_accuracy, "attention kernels against fp32, per query of a verify row")
    p.add_argument("--parts", default="qwen-full,gemma-sliding,gemma-full", help="the layer kinds to probe")
    p.add_argument("--rows", default="1,4,16,64,128", help="verify rows a step")
    p.add_argument("--min-len", type=int, default=2000, help="the verify rows' lowest KV lengths: at least")
    p.add_argument("--max-len", type=int, default=6000, help="the verify rows' lowest KV lengths: at most")
    p.add_argument("--k", type=int, default=3, help="drafts: verify rows of 1 + k queries, planned k above")
    p.add_argument("--true", default="rand,lb", help="true lengths: lb, rand, ub")
    p.add_argument("--logit-std", type=float, default=2.0, help="the spread of q·k × scale")
    p.add_argument("--bad", type=float, default=0.05, help="a (row, query)'s relative RMS error counted bad above")
    p.add_argument("--max-model-len", type=int, default=BATCH_INVARIANT_MAX_LEN - 8,
                   help="sizes the fixed splits and the page tables")
    p.add_argument("--max-num-seqs", type=int, default=128, help="decode rows at most (the graph buckets)")
    p.add_argument("--max-tokens", type=int, default=4096, help="a step's tokens at most (our kernels' tables)")
    p = add("nan-leak", cmd_nan_leak, "a NaN across packed GDN sequences, and the padding fix")
    p.add_argument("--lens", default="100,128,70,300,200", help="the packed sequences' lengths")
    p.add_argument("--nan-in", default="q,k,v", help="inputs to put the NaN in: q, k, v, a, b")
    p.add_argument("--nan-rows", type=int, default=1, help="NaN rows at the sequence's start")
    p = add("timing", cmd_timing, "GDN prefill timings: checkpoints, routes, padding")
    p.add_argument("--iters", type=int, default=50, help="timed calls a variant")
    p.add_argument("--warmup", type=int, default=5, help="untimed calls before them")
    p.add_argument("--single", default=f"512,1024,2048,4096,{BATCH_INVARIANT_MAX_LEN - 8}",
                   help="single-sequence lengths")
    p.add_argument("--padded", default="1000,300,2000,70,700;100,128,70,300,200",
                   help="packed mixes for the padded call (;-separated)")
    p = add("attn-timing", cmd_attn_timing, "attention kernels' time: batch-invariant against switch-off")
    p.add_argument("--parts", default="qwen-full,gemma-full", help="the layer kinds to time (gemma-sliding too)")
    p.add_argument("--max-model-len", type=int, default=262144, help="the context both paths are sized for")
    p.add_argument("--prefill-from", type=int, default=8192, help="a prefill query's keys taken whole up to this")
    p.add_argument("--prefill-keys", default="0", help="the prefill split bases to time past it (0: each part's)")
    p.add_argument("--short-chunks", default="64,256", help="short prefill rows' queries")
    p.add_argument("--histories", default="8192,32768,131072", help="the positions before the short prefill rows")
    p.add_argument("--long-chunk", type=int, default=2048, help="the long prefill row's queries")
    p.add_argument("--long-histories", default="0,32768", help="the positions before the long prefill row")
    p.add_argument("--decode-rows", default="1,64", help="verify rows a step")
    p.add_argument("--decode-histories", default="1024,8192,32768", help="the positions before the verify rows' keys")
    p.add_argument("--k", type=int, default=3, help="drafts: verify rows of 1 + k queries (all parts)")
    p.add_argument("--max-num-seqs", type=int, default=128, help="decode rows at most (the graph buckets)")
    p.add_argument("--iters", type=int, default=50, help="timed calls a variant")
    p.add_argument("--warmup", type=int, default=5, help="untimed calls before them")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
