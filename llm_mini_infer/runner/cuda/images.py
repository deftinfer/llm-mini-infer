"""The image path of the CUDA runners, one implementation for every model family (docs/architecture.md, "The vision
encoder and its cache"): the Qwen runner (`runner.py`) and Gemma 4's (`gemma.py`) each own an `ImageEncoder`.

- **The vision tower** of the checkpoint's family (`vision.family.ImageFamily.load_encoder`: Qwen3.5/3.8's
  `models.qwen3_5.vision`, Gemma 4's `models.gemma4.vision`), behind one interface: `patches`, `meta`, `forward`.
- **The encoder pool:** `encoder_rows` embedding rows of the text model's width on the device. The scheduler owns the
  bookkeeping (`cache.encoder`: which image sits in which rows); ENCODE names the rows to write, and the plan's
  `tok_emb` the rows a token reads.
- **ENCODE** (`stage`, then `encode`, after the plan's copies and before the forward, stream-ordered): each image's
  uint8 pixels are copied from the pixel arena (`vision.arena`; or the internal pixels of synthetic plans) into the
  step's pinned staging, uploaded in one H2D copy, rescaled and patchified on the device by the tower, encoded, and
  written to the named rows. The budget (`encoder_budget` image tokens per step) bounds the staging and the graphs.
- **The embedding scatter** (`embeds`): a token whose `tok_emb` names a row takes that row as its input embedding
  instead of its token's (graph-safe: a `torch.where` over the token embeddings).
- **CUDA graphs of the tower** (`VisionGraphs`), one per bucket of patches, captured with the runner's other graphs.
- **Synthetic images** (`synthetic`) for the memory profile and the warm-up: an image of n soft tokens, its grid
  (within the tower's position table) and its pixels from a fixed random buffer; the profile's step encodes a budget
  of the largest images serving takes (`profile_images`).
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ...lifecycle import Lifecycle
from ...sched.plan import StepPlan
from ..base import PlanError

log = logging.getLogger(__name__)


class Phase(IntEnum):
    """An `ImageEncoder`'s phase."""

    LOADED = 0  # the tower (construction, or after `release`)
    ALLOCATED = 1  # its pool and attention backend (`allocate`)
    CAPTURED = 2  # and its graphs (`capture`)


# `allocate` and `release` from any phase (the memory profile releases inside `capabilities`, a cleanup path must not
# fail, and an allocate drops the graphs: the runners release first); `capture` from ALLOCATED alone
_EDGES = {Phase.LOADED: (Phase.ALLOCATED, Phase.LOADED),
         Phase.ALLOCATED: (Phase.ALLOCATED, Phase.CAPTURED, Phase.LOADED),
         Phase.CAPTURED: (Phase.ALLOCATED, Phase.LOADED)}
_ALLOCATED = Phase.ALLOCATED  # `stage` and `encode`'s one compare


@dataclass(frozen=True)
class VisionOptions:
    """Images: the encoder pool's rows (rounded up to the encoder cache's blocks), the image tokens one step may
    encode, the pixel arena file (`vision.arena`), and Gemma 4's image budget (images.max_soft_tokens: the tower's
    graph buckets follow it; None: the checkpoint's)."""

    encoder_rows: int
    encoder_budget: int
    arena: str = ""
    max_soft_tokens: int | None = None


def default_vision_buckets(max_patches: int, unit: int = 4, dense: int = 2048) -> list[int]:
    """Patch counts with a vision-tower graph: every 128 from 256 to `dense` (at least 2,048: Qwen images of 64 to 512
    tokens; Gemma 4: its largest image, 2,520 patches), then about 1.25x apart up to `max_patches`, each rounded up to
    a multiple of `unit` (the patches of one soft token: merge², so the pooling of a bucket's padding stays separate).
    Padding wastes at most a 128-row step up to `dense` (a third at 256-384 patches, under a tenth past 1,280), at most
    a quarter past it. Every call up to the budget (with one call per image, up to the largest image: `family_buckets`)
    replays a graph: an eager call would allocate activations the memory profile does not reserve (it measures the
    tower from its graphs)."""
    out, b = [], 256
    top = max(2048, dense)
    while b < max_patches:
        out.append(b)
        b = b + 128 if b < top else (b * 5 // 4 + 127) // 128 * 128
    if out and out[-1] >= max_patches - 128:
        out.pop()
    out = [-(-x // unit) * unit for x in [*out, max_patches]]
    return sorted(set(out))


def family_buckets(fam: Any, merge: int, budget: int, per_image: bool = False) -> list[int]:
    """The default buckets of a family's tower for an encoder budget (image tokens per step): dense up to the family's
    largest image when that is at most 4,096 patches (Gemma 4: 2,520), else to 2,048 (Qwen's default sizing allows
    images far larger than typical ones). `per_image` (one tower call per image) on a family with a soft-token budget
    (Gemma 4's `max_soft_tokens`, which no image exceeds): the list ends at its first bucket that holds the largest
    image, as no call is larger. It is the whole list up to there, so every call takes the bucket it takes in the
    whole list: the same GEMM shapes, the same bits."""
    unit = merge * merge
    largest = unit * fam.max_tokens()
    out = default_vision_buckets(unit * budget, unit, largest if largest <= 4096 else 2048)
    if per_image and fam.max_soft_tokens is not None:
        cap = unit * min(budget, fam.max_soft_tokens)  # the largest call's patches: one image
        out = out[:next(i for i, b in enumerate(out) if b >= cap) + 1]
    return out


def synthetic_grid(tokens: int, merge: int, max_side: int | None = None) -> tuple[int, int]:
    """The patch grid (h, w) of a synthetic image of `tokens` soft tokens (merge x merge patches each), as near square
    as their divisors allow; where that is a strip with a side past `max_side` patches (a prime count, say),
    isqrt(tokens) rows of tokens // isqrt(tokens) soft tokens, a few tokens fewer (each side then at most
    merge x (isqrt(tokens) + 2))."""
    root = math.isqrt(tokens)
    a = next(d for d in range(root, 0, -1) if tokens % d == 0)
    if max_side is not None and merge * (tokens // a) > max_side:
        a = root
    return merge * a, merge * (tokens // a)


def profile_images(tokens: int, largest: int | None) -> list[int]:
    """The soft tokens of each image the memory profile's image step encodes, `tokens` of them in all: as many images
    of the largest size (`largest`, Gemma 4's `max_soft_tokens`) as they hold, then one of the rest; one image of
    them all without a largest size (None). Each is an image serving can take, its grid one a request's image has (as
    vLLM's profile, which encodes the most images of the largest size the budget holds)."""
    if not largest:
        return [tokens]
    k, rest = divmod(tokens, largest)
    return [largest] * k + ([rest] if rest else [])


class VisionGraphs:
    """Breakable CUDA graphs of a vision tower, one per bucket of patches (`dev.runner.cuda.graphs`): the segments
    between attention calls are graph segments; attention runs eagerly at each replay with the call's plan (FlashInfer's
    ragged kernel over the real patches only). Inputs go into static buffers sized to the largest bucket: the patches
    and each per-patch field of the encoder's metadata (`PER_PATCH`); padding rows keep what they hold and their
    outputs are dropped (every op but attention is per row, and buckets are multiples of a soft token's patches, so the
    pooling of the padding stays apart). An eager tower is launch-bound at small images; a replay is depth + 1 graph
    segments and depth eager attention calls."""

    def __init__(self, enc: Any, attn: Any, buckets: list[int], pool: Any):
        from ...models.common import weak_view

        self.enc, self.attn, self.pool = enc, attn, pool
        self.buckets = sorted(buckets)
        self._weak = weak_view
        top = self.buckets[-1]
        c = enc.cfg
        dev = next(enc.parameters()).device
        sample = enc.meta([(1, c.merge, c.merge)], dev)  # the per-patch fields' dtypes and trailing shapes
        self.meta_cls, self.fields = type(sample), type(sample).PER_PATCH
        self.static = {f: torch.zeros((top, *getattr(sample, f).shape[1:]), dtype=getattr(sample, f).dtype,
                                      device=dev) for f in self.fields}
        self.patches = torch.zeros((top, c.patch_dim), dtype=torch.float32, device=dev)
        self.graphs: dict[int, Any] = {}
        self.outs: dict[int, torch.Tensor] = {}
        self.current: Any = None  # the call's metadata (its attention plan), read by the breaks

    def bucket(self, n: int) -> int | None:
        return next((b for b in self.buckets if b >= n), None)

    def capture(self, b: int) -> None:
        from .graphs import BreakableCapture

        # one image of b patches (the plan only matters for the capture: the breaks read each call's plan)
        meta = self.meta_cls(n=b, cu=torch.tensor([0, b], dtype=torch.int32, device=self.patches.device),
                             cu_host=[0, b], **{f: t[:b] for f, t in self.static.items()})
        self.attn.plan(meta)
        self.current = meta
        cap = BreakableCapture(self.pool)
        backend = _BreakAttention(self, cap)
        torch.cuda.current_stream().synchronize()
        with cap:
            out = self.enc(self.patches[:b], meta, backend)
        self.graphs[b] = cap
        self.outs[b] = out

    def run(self, patches: torch.Tensor, meta: Any) -> torch.Tensor | None:
        """The tower from the graph of the smallest bucket that holds `meta.n` patches, or None (no such graph). The
        replay computes the bucket's padding rows too; why a padded call still replays: `default_vision_buckets`."""
        n = meta.n
        b = self.bucket(n)
        if b is None or b not in self.graphs:
            return None
        self.patches[:n].copy_(patches)
        for f, t in self.static.items():
            t[:n].copy_(getattr(meta, f))
        self.attn.plan(meta)
        self.current = meta
        self.graphs[b].replay()
        return self.outs[b][:n // self.enc.cfg.merge ** 2]


class _BreakAttention:
    """Attention as an eager break of a `VisionGraphs` capture: the output is allocated in the segment before the
    break; at each replay the break runs the attention over the call's real patches with the call's plan (weak views:
    the graph's pool owns the buffers)."""

    def __init__(self, graphs: VisionGraphs, cap: Any):
        self.graphs, self.cap = graphs, cap
        self.pad = getattr(graphs.attn, "pad", graphs.enc.cfg.head_dim)

    def run(self, meta: Any, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(q)
        wk = self.graphs._weak
        qw, kw, vw, ow = (wk(x) for x in (q, k, v, out))
        g = self.graphs

        def run() -> None:
            cur = g.current
            n = cur.n
            ow[:n].copy_(g.attn.run(cur, qw[:n], kw[:n], vw[:n]))

        self.cap.eager(run)
        return out


class ImageEncoder:
    """A runner's image path (module docstring). Built with the runner (it loads the tower), `allocate`d with its
    pools, `capture`d with its graphs (its phase: `Phase`); ENCODE needs it allocated. `buckets`: the tower's graph
    buckets (None: `family_buckets`, which `per_image` shortens); `per_image`: one tower call per image (batch
    invariance)."""

    def __init__(self, model_dir: str | Path, opts: VisionOptions, device: torch.device, dtype: torch.dtype,
                 buckets: list[int] | None = None, per_image: bool = False):
        from ...vision import family as vf

        t0 = time.perf_counter()
        # Only the budget: the buckets need the largest image (Gemma 4's follows images.max_soft_tokens), and the Qwen
        # runner sizes nothing (its buckets are dense to 2,048 patches under its default pixel bounds, as under any
        # whose largest image is at most 2,048 patches or past 4,096: `family_buckets`)
        fam = vf.load(model_dir, max_soft_tokens=opts.max_soft_tokens)
        if fam is None:
            raise ValueError(f"{model_dir} has no vision tower this server runs")
        self.max_soft_tokens = fam.max_soft_tokens
        if device.type == "cuda":
            torch.cuda.empty_cache()
        self.device, self.dtype = device, dtype
        self.vision = fam.load_encoder(model_dir, device, dtype)
        self.cfg = c = self.vision.cfg
        self.rows = -(-opts.encoder_rows // 16) * 16  # whole blocks of the scheduler's encoder cache
        self.budget = opts.encoder_budget
        self.arena: Any = None
        if opts.arena:
            from ...vision.arena import map_readonly

            self.arena = map_readonly(opts.arena)
        side = c.merge * c.patch
        self.px_bytes = 3 * side * side * self.budget  # the uint8 pixels of a budget's soft tokens
        self.buckets = family_buckets(fam, c.merge, self.budget, per_image) if buckets is None else \
            sorted(b for b in buckets if b > 0)
        self.load_s = round(time.perf_counter() - t0, 2)
        self.weight_bytes = sum(p.numel() * p.element_size() for p in self.vision.parameters())
        self.pool: torch.Tensor | None = None
        self.attn: Any = None
        self.graphs: VisionGraphs | None = None
        self.life = Lifecycle("the image encoder", Phase.LOADED, _EDGES)
        self._px_internal: np.ndarray | None = None  # pixels of internal plans (profile, warm-up, capture)
        # One tower call per image (batch invariance): an image's GEMMs then run at its own bucket's M, not the
        # step's images' together, which cuBLAS may give another kernel (other bits) when other images join the step.
        # Read at each ENCODE; the default buckets follow the constructor's (`family_buckets`)
        self.per_image = per_image

    # -- lifetime -------------------------------------------------------------------------------------------------

    def allocate(self, hidden: int, flashinfer: bool, workspace: torch.Tensor | None = None) -> None:
        """The encoder pool (rows of the text model's width `hidden`) and the attention backend."""
        c = self.cfg
        self.pool = torch.zeros((self.rows, hidden), dtype=self.dtype, device=self.device)
        if flashinfer:
            from ...kernels.flashinfer import FlashInferVisionAttention

            self.attn = FlashInferVisionAttention(c.heads, c.head_dim, self.dtype, self.device, workspace=workspace,
                                                  scale=c.scale)
        else:
            from ...models.qwen3_5.vision import TorchVisionAttention

            self.attn = TorchVisionAttention(scale=c.scale)
        self.graphs = None
        self.life.to(Phase.ALLOCATED)

    def release(self) -> None:
        self.pool = self.attn = self.graphs = None
        self.life.to(Phase.LOADED)

    def capture(self, pool: Any) -> None:
        """The tower's graphs, by bucket of patches (largest first: later captures reuse the shared pool's memory); an
        allocated encoder's, kept once every bucket is captured."""
        if not self.buckets:
            return
        if self.life.phase is not Phase.ALLOCATED:
            raise RuntimeError(f"the image encoder: a capture when {self.life.phase.name}, not ALLOCATED")
        graphs = VisionGraphs(self.vision, self.attn, self.buckets, pool)
        for n in sorted(self.buckets, reverse=True):
            graphs.capture(n)
        self.graphs = graphs
        self.life.to(Phase.CAPTURED)

    def graph_buckets(self) -> list[int]:
        return sorted(self.graphs.graphs) if self.graphs is not None else []

    # -- ENCODE ---------------------------------------------------------------------------------------------------

    def stage(self, plan: StepPlan, st: Any, internal: bool) -> tuple[int, int, np.ndarray]:
        """Validate ENCODE and copy the images' pixels from the arena into the step's pinned staging (they are read
        before the step completes, as the contract allows), and stage the destination rows. Returns (the rows'
        staging offset, pixel bytes, per-image byte counts)."""
        if self.life.phase < _ALLOCATED:
            raise RuntimeError(f"step {plan.step_id}: ENCODE before the image encoder is allocated")
        n = plan.enc_tokens.astype(np.int64)
        hw = plan.enc_hw.astype(np.int64)
        g = plan.enc_grid.astype(np.int64)
        p, m = self.cfg.patch, self.cfg.merge
        rows = plan.enc_rows
        if (int(n.sum()) != len(rows) or int(n.sum()) > self.budget or (rows < 0).any() or
                (rows >= self.rows).any() or (g[:, 0] != 1).any() or (hw[:, 0] != g[:, 1] * p).any() or
                (hw[:, 1] != g[:, 2] * p).any() or (g[:, 1] * g[:, 2] != n * m * m).any()):
            raise PlanError(f"step {plan.step_id}: ENCODE of {n.tolist()} tokens, grids {g.tolist()}, sizes "
                            f"{hw.tolist()}: inconsistent, or over the budget ({self.budget}) or the pool")
        nbytes = 3 * hw[:, 0] * hw[:, 1]
        total = int(nbytes.sum())
        if total > len(st.npx):
            raise PlanError(f"step {plan.step_id}: ENCODE of {total} pixel bytes > the staging's {len(st.npx)}")
        off = 0
        for src, nb in zip(plan.enc_pixels.tolist(), nbytes.tolist()):
            st.npx[off:off + nb] = self._pixels(src, nb, internal)
            off += nb
        return st.put(rows.astype(np.int64)), total, nbytes

    def _pixels(self, off: int, n: int, internal: bool) -> np.ndarray:
        if internal:
            return self._px_internal[off:off + n]
        if self.arena is None:
            raise RuntimeError("ENCODE without a pixel arena (VisionOptions.arena)")
        return np.frombuffer(self.arena.buf, np.uint8, n, off)

    def encode(self, plan: StepPlan, st: Any, enc: tuple[int, int, np.ndarray], graphs: bool) -> None:
        """Upload the staged pixels, patchify, run the tower (its graphs when `graphs` and a bucket holds the call)
        and write the rows."""
        if self.life.phase < _ALLOCATED:
            raise RuntimeError(f"step {plan.step_id}: ENCODE before the image encoder is allocated")
        o_rows, total, nbytes = enc
        v = self.vision
        p = self.cfg.patch
        dpx = st.px[:total].to(self.device, non_blocking=True)
        grids = [tuple(x) for x in plan.enc_grid.tolist()]
        parts, off = [], 0
        for (_, gh, gw), nb in zip(grids, nbytes.tolist()):
            parts.append(v.patches(dpx[off:off + nb].view(3, gh * p, gw * p)))
            off += nb
        rows = st.i(o_rows, len(plan.enc_rows))
        if self.per_image and len(parts) > 1:  # each image alone, its rows written before the next call's replay
            r0 = 0
            for g, patches in zip(grids, parts):
                out = self._tower(patches, [g], graphs)
                self.pool.index_copy_(0, rows[r0:r0 + len(out)], out.to(self.pool.dtype))
                r0 += len(out)
            return
        patches = parts[0] if len(parts) == 1 else torch.cat(parts)
        self.pool.index_copy_(0, rows, self._tower(patches, grids, graphs).to(self.pool.dtype))

    def _tower(self, patches: torch.Tensor, grids: list[tuple[int, int, int]], graphs: bool) -> torch.Tensor:
        """The tower over the images of `grids` (their patches, concatenated): its graph of the smallest bucket that
        holds them when `graphs`, else eagerly (no graphs, or more patches than the largest bucket)."""
        v = self.vision
        meta = v.meta(grids, self.device)
        out = self.graphs.run(patches, meta) if graphs and self.graphs is not None else None
        if out is None:
            self.attn.plan(meta)
            out = v(patches, meta, self.attn)
        return out

    def embeds(self, rows: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        """Input embeddings: a token's encoder row where `rows` names one (≥ 0), else its token's embedding (`text`)."""
        return torch.where((rows >= 0)[:, None], self.pool.index_select(0, rows.clamp(min=0)), text)

    def check_rows(self, rows: np.ndarray, step: int) -> None:
        if len(rows) and int(rows.max()) >= self.rows:
            raise PlanError(f"step {step}: an encoder row out of range")

    # -- synthetic images (memory profile, warm-up) ---------------------------------------------------------------

    def synthetic(self, tokens: int) -> tuple[tuple[int, int, int], tuple[int, int], int]:
        """An image of `tokens` soft tokens as near square as its divisors allow, its grid within the tower's
        (`max_grid`: Gemma 4's position table; Qwen's has none), else a few tokens fewer (`synthetic_grid`): (its patch
        grid (1, h, w), its pixel size, its soft tokens), with internal pixels (`stage(..., internal=True)` from
        offset 0) for it."""
        m, p = self.cfg.merge, self.cfg.patch
        gh, gw = synthetic_grid(tokens, m, getattr(self.cfg, "max_grid", None))
        need = 3 * gh * gw * p * p
        if self._px_internal is None or len(self._px_internal) < need:
            self._px_internal = np.random.default_rng(0).integers(0, 256, max(self.px_bytes, need), dtype=np.uint8)
        return (1, gh, gw), (gh * p, gw * p), gh * gw // (m * m)

    def describe(self, graphs: bool) -> str:
        """The startup line. `graphs`: whether the runner captures CUDA graphs (`capture` then takes one per bucket;
        otherwise, or with no buckets, every call runs eagerly)."""
        b = self.buckets
        budget = f"images of at most {self.max_soft_tokens} soft tokens; " if self.max_soft_tokens else ""
        if graphs and b:
            tower = f"tower graphs for {len(b)} buckets of {b[0]:,}-{b[-1]:,} patches"
        else:
            why = "dev.runner.cuda.vision_buckets is empty" if graphs else "CUDA graphs off"
            tower = f"no tower graphs ({why}): the tower runs eagerly"
        return (f"vision tower loaded in {self.load_s:.1f} s: {self.weight_bytes / 2**30:.2f} GiB; encoder pool "
                f"{self.rows} rows, {self.budget} image tokens per step; {budget}{tower}")
