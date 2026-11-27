#!/usr/bin/env python3
"""Gemma 4's vision tower on one image, profiled on one GPU: an encode's device time, kernels and launches, from CUDA
graphs and eagerly, and what the glue around its GEMMs and attention costs (`models/gemma4/vision.py`: the q/k/v
norms, the 2-D RoPE, the pad to 128, the pooling).

The image is a synthetic 1280 x 720 JPEG (`image_jpeg`), sized by the checkpoint's image family at `--max-soft-tokens`
(default 280, Gemma 4's own budget). At 280 the run asserts the image's geometry (`check_reference`): 576 x 1056
pixels, patch grid (1, 36, 66), 2376 patches, 264 soft tokens, the 2439 bucket. It goes
through the server's path, `ImageEncoder.encode` as the runner calls it: the uint8 pixels uploaded from pinned staging,
patchified on the device, the tower replayed from the `VisionGraphs` graph of the smallest bucket that holds the call
(the family's buckets at the budget and the encoder budget, as `runner.cuda.images.family_buckets` derives them
without one call per image; an image in another bucket is recorded), its 27 attention calls FlashInfer's ragged kernel
as eager breaks, and the soft tokens written to the encoder pool. Only the image's bucket is captured
(`--all-buckets`: every one, as the runner does with batch invariance off; with it on, the runner's list ends at the
largest image's bucket, and every image takes the same bucket in both lists). `--eager` compares the same call without
graphs (every kernel launched by itself).

- `--mode torch`: per path, `--warmup` calls, then `--calls` calls timed with CUDA events (synchronized after each:
  the device time's median, p10 and p90, and the wall time), with NVML's power and SM clock; then `--calls` calls
  under torch.profiler (CUPTI, kernels inside graph replays too): per call the kernel time and count, the time and
  share of each category (`TOWER_CATEGORIES`: cuBLAS's GEMMs, FlashInfer's attention, our fused kernels, torch's
  glue, copies), the top 30 kernels and every kernel's name (`other`: the names no category matched); the launches per
  call (the capture's graph segments and eager attention breaks, or every kernel eagerly), and from the window's trace
  (`timeline`, profile_verify.py) the device's busy time per call and the gaps between the kernels inside the graph
  segments. Also the graph's soft tokens against the eager call's (relative L2: the bucket's padding rows change the
  GEMM shapes; BF16 noise). `--trace` keeps the traces.
- `--mode ncu`: per path, `--calls` calls inside an NVTX range `tower_graph` / `tower_eager` and a
  cudaProfilerStart/Stop window, for `ncu --profile-from-start off` (torch.profiler never runs in the same process).

**The glue:** one kernel of ours computes the attention's input (the q and k norms with their weights, v's norm, the
2-D RoPE, the pad to FlashInfer's head dim), bit for bit the torch ops' 57 kernels a layer
(`models.gemma4.vision.vision_qkv`). A torch profile counts its launches per call (`fused_glue_launches`: one a
layer, 27; fewer where the kernel does not apply and `vision_qkv` falls back to the torch ops).

    python3 -m tests.scripts.profile_tower --model <dir> --out tower.json [--eager] [--mode torch|ncu]
"""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from .profile_verify import (
    TOWER_CATEGORIES,
    PowerSampler,
    activities,
    export_trace,
    gpu_info,
    init_cupti,
    ncu_window,
    profiler_rows,
    stats,
    summarize_kernels,
    timeline,
    versions,
)

# The reference image (1280 x 720) at Gemma 4's default budget, 280 soft tokens, on its 16-pixel patches: what the run
# asserts for that image at that budget
REFERENCE = {"image": [1280, 720], "max_soft_tokens": 280, "patch": 16, "size": [576, 1056], "grid": [1, 36, 66],
             "tokens": 264, "patches": 2376, "bucket": 2439}
FUSED_GLUE = "g4_vision_qkv"  # our attention-input kernel's name in a profile (csrc/vision.cu)


def image_jpeg(width: int, height: int, seed: int = 0) -> bytes:
    """A synthetic JPEG of this size: a random background and 20 random rectangles, quality 90 (at 400 x 225, byte for
    byte the first of test_profile_harness's `synthetic_images`)."""
    from PIL import Image, ImageDraw

    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (width, height), tuple(int(x) for x in rng.integers(0, 256, 3)))
    d = ImageDraw.Draw(img)
    for _ in range(20):
        x0, y0 = int(rng.integers(0, width)), int(rng.integers(0, height))
        d.rectangle([x0, y0, x0 + int(rng.integers(4, width // 2)), y0 + int(rng.integers(4, height // 2))],
                    fill=tuple(int(x) for x in rng.integers(0, 256, 3)))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def image_shape(fam: Any, data: bytes, budget: int) -> dict[str, Any]:
    """The image as the family sizes it (its header only), the encoder budget the config derives for a step budget
    (`max(max_num_batched_tokens, the largest image)`), the tower's buckets for it (`family_buckets`, as the runner's
    `ImageEncoder`) and the bucket that holds the image's patches."""
    from llm_mini_infer import hostcheck
    from llm_mini_infer.runner.cuda.images import family_buckets
    from llm_mini_infer.vision.family import transposed
    from llm_mini_infer.vision.probe import probe

    h = probe(data)
    size = fam.size(h)
    image_wh = hostcheck.oriented_size(io.BytesIO(data))  # Pillow's reading, as the pool's decode orients it
    if image_wh != (h.width, h.height):
        size = transposed(size)
    enc_budget = max(budget, fam.max_tokens())
    buckets = family_buckets(fam, fam.pooling, enc_budget)
    patches = size.grid[1] * size.grid[2]
    bucket = next((b for b in buckets if b >= patches), None)
    return {"image": list(image_wh), "max_soft_tokens": fam.max_soft_tokens, "patch": fam.patch,
            "size": [size.height, size.width], "grid": list(size.grid), "tokens": size.tokens, "patches": patches,
            "encoder_budget": enc_budget, "buckets": buckets, "bucket": bucket,
            "padding": None if bucket is None else bucket - patches}


def check_reference(shape: dict[str, Any]) -> list[str]:
    """For the reference image size at images.max_soft_tokens 280 on Gemma 4's 16-pixel patches: the sizes it must have
    (raises), and a note when its bucket is not 2439. Any other image, budget or patch: nothing to check."""
    if any(shape[key] != REFERENCE[key] for key in ("image", "max_soft_tokens", "patch")):
        return []
    for key in ("size", "grid", "tokens", "patches"):
        if shape[key] != REFERENCE[key]:
            raise AssertionError(f"the reference image: {key} {shape[key]}, not {REFERENCE[key]}")
    if shape["bucket"] != REFERENCE["bucket"]:
        return [f"{REFERENCE['patches']} patches land in the {shape['bucket']} bucket, not {REFERENCE['bucket']}"]
    return []


class TowerCall:
    """One ENCODE of the image as the runner runs it: `ImageEncoder.encode` with the image's uint8 pixels in pinned
    staging (on CPU, plain memory), a plan naming its grid and encoder-pool rows 0..tokens - 1, from the graphs or
    eagerly."""

    def __init__(self, enc: Any, fam: Any, data: bytes):
        import torch

        from llm_mini_infer.vision.family import transposed
        from llm_mini_infer.vision.probe import probe

        h = probe(data)
        size = fam.size(h)
        px = fam.pixels(data, h, size)
        if tuple(px.shape[1:]) != (size.height, size.width):  # Pillow turned the image: the decode decides
            size = transposed(size)
        px = px.reshape(-1).contiguous()  # uint8 [3 * H * W]
        self.enc, self.tokens, self.grid = enc, size.tokens, tuple(size.grid)
        self.patches = size.grid[1] * size.grid[2]
        self.nbytes = px.numel()
        self.px = px.pin_memory() if enc.device.type == "cuda" else px
        rows = torch.arange(size.tokens, dtype=torch.int64, device=enc.device)
        self.plan = SimpleNamespace(enc_grid=np.asarray([size.grid], np.int64),
                                    enc_rows=np.arange(size.tokens, dtype=np.int64))
        self.stage = SimpleNamespace(px=self.px, i=lambda off, n: rows[:n])
        self.arg = (0, self.nbytes, np.asarray([self.nbytes], np.int64))

    def __call__(self, graphs: bool) -> None:
        import torch

        with torch.inference_mode():
            self.enc.encode(self.plan, self.stage, self.arg, graphs)

    def output(self) -> Any:
        """The soft tokens the last call wrote (a copy)."""
        return self.enc.pool[:self.tokens].clone()


class ImagesCall:
    """One ENCODE of several images as `TowerCall` runs one, the images given as uint8 pixels [3, H, W] whose sides
    are whole soft tokens (merge × patch pixels): their pixels one after the other in staging, a plan of their grids,
    and encoder-pool rows from 0, each image's soft tokens after the one before's."""

    def __init__(self, enc: Any, pixels: list[Any]):
        import torch

        p, m = enc.cfg.patch, enc.cfg.merge
        self.enc = enc
        self.grids = [(1, px.shape[1] // p, px.shape[2] // p) for px in pixels]
        self.tokens = [h * w // (m * m) for _, h, w in self.grids]
        self.starts = np.concatenate([[0], np.cumsum(self.tokens)[:-1]]).astype(np.int64)
        flat = torch.cat([px.reshape(-1) for px in pixels])
        self.px = flat.pin_memory() if enc.device.type == "cuda" else flat
        n = sum(self.tokens)
        rows = torch.arange(n, dtype=torch.int64, device=enc.device)
        self.plan = SimpleNamespace(enc_grid=np.asarray(self.grids, np.int64), enc_rows=np.arange(n, dtype=np.int64))
        self.stage = SimpleNamespace(px=self.px, i=lambda off, k: rows[:k])
        nbytes = np.asarray([px.numel() for px in pixels], np.int64)
        self.arg = (0, int(nbytes.sum()), nbytes)

    def __call__(self, graphs: bool) -> None:
        import torch

        with torch.inference_mode():
            self.enc.encode(self.plan, self.stage, self.arg, graphs)

    def rows(self, i: int) -> Any:
        """Image i's soft tokens, as the last call wrote them (a copy)."""
        a = int(self.starts[i])
        return self.enc.pool[a:a + self.tokens[i]].clone()


def glue_launches(cupti: dict[str, Any]) -> float:
    """Our attention-input kernel's launches per call in a CUPTI summary (`FUSED_GLUE`): one a layer where it
    applies."""
    return round(sum(k["calls"] for k in cupti["all"] if FUSED_GLUE in k["kernel"]), 2)


def capture(enc: Any, buckets: list[int]) -> dict[str, Any]:
    """The tower's graphs of `buckets` (`VisionGraphs`, largest first, on a side stream with the collector off, as the
    runner captures them), installed as the encoder's."""
    import torch

    from llm_mini_infer.runner.cuda.graphs import no_gc, warm_libraries
    from llm_mini_infer.runner.cuda.images import VisionGraphs

    warm_libraries(enc.device)
    t0 = time.perf_counter()
    with torch.inference_mode(), no_gc(), torch.cuda.stream(torch.cuda.Stream(enc.device)):
        vg = VisionGraphs(enc.vision, enc.attn, enc.buckets, torch.cuda.graph_pool_handle())
        for b in sorted(buckets, reverse=True):
            vg.capture(b)
    torch.cuda.synchronize()
    enc.graphs = vg
    return {"buckets": sorted(buckets), "capture_s": round(time.perf_counter() - t0, 2)}


def launches(enc: Any, call: TowerCall, graphs: bool, kernels: float | None) -> dict[str, Any]:
    """Launches per call: the bucket's graph segments and its eager attention breaks, or (eagerly) every kernel."""
    if not graphs:
        return {"eager_kernels": kernels}
    cap = enc.graphs.graphs[enc.graphs.bucket(call.patches)]
    return {"graph_segments": cap.graphs, "eager_breaks": cap.breaks, "launches": cap.graphs + cap.breaks}


def time_calls(call: TowerCall, graphs: bool, n: int, sampler: PowerSampler | None, phase: str) -> dict[str, Any]:
    """n calls, each between CUDA events and synchronized: device and wall ms per call."""
    import torch

    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    dev, wall = [], []
    if sampler is not None:
        sampler.phase = phase
    for _ in range(n):
        t0 = time.perf_counter()
        a.record()
        call(graphs)
        b.record()
        b.synchronize()
        wall.append((time.perf_counter() - t0) * 1e3)
        dev.append(a.elapsed_time(b))
    if sampler is not None:
        sampler.phase = ""
    out = {"device_ms": stats(dev), "wall_ms": stats(wall)}
    if sampler is not None:
        out["nvml"] = sampler.summary(phase)
    return out


def profile_calls(call: TowerCall, graphs: bool, n: int, trace: Path | None = None) -> dict[str, Any]:
    """n calls under torch.profiler (CUDA activity), between synchronizations: the kernels per call by category, and
    by launch from the window's trace (`timeline`: the graph segments' busy time against their span, the eager
    breaks)."""
    import torch

    torch.cuda.synchronize()
    with torch.profiler.profile(activities=activities()) as prof:
        for _ in range(n):
            call(graphs)
        torch.cuda.synchronize()
    out = summarize_kernels(profiler_rows(prof), n, TOWER_CATEGORIES)
    out["timeline"] = timeline(export_trace(prof, trace, f"tower_{'graph' if graphs else 'eager'}"), n,
                               TOWER_CATEGORIES)
    return out


def rel(a: Any, b: Any) -> float:
    return float((a.float() - b.float()).norm() / b.float().norm())


def profile(enc: Any, call: TowerCall, *, paths: list[str], calls: int, warmup: int, mode: str,
            sampler: PowerSampler | None = None, trace: Path | None = None, log: Any = print) -> dict[str, Any]:
    """Each path ("graph", "eager") of the call: warm-up, then the timing and the CUPTI profile (`mode` torch: with our
    attention-input kernel's launches, `glue_launches`) or the ncu window (ncu); the graph's soft tokens against the
    eager call's."""
    out: dict[str, Any] = {}
    if "graph" in paths:
        b = enc.graphs.bucket(call.patches) if enc.graphs is not None else None
        if b is None or b not in enc.graphs.graphs:
            raise RuntimeError(f"no tower graph holds {call.patches} patches: the graph path would run eagerly")
    for path in paths:
        graphs = path == "graph"
        for _ in range(warmup):
            call(graphs)
        r: dict[str, Any] = {}
        if mode == "torch":
            r["timing"] = time_calls(call, graphs, calls, sampler, f"tower_{path}")
            r["cupti"] = profile_calls(call, graphs, calls, trace)
            r["launches"] = launches(enc, call, graphs, r["cupti"]["kernels"])
            r["fused_glue_launches"] = glue_launches(r["cupti"])
            cats = ", ".join(f"{c} {v['ms']:.2f} ms / {v['calls']:.0f}" for c, v in r["cupti"]["categories"].items()
                             if v["calls"])
            log(f"{path}: {r['timing']['device_ms']['p50']} ms per call (device, p50), {r['cupti']['kernels']:.0f} "
                f"kernels, {r['launches']}, our glue kernel {r['fused_glue_launches']:g} times; {cats}")
        elif mode == "ncu":
            t0 = time.perf_counter()
            ncu_window(f"tower_{path}", lambda graphs=graphs: [call(graphs) for _ in range(calls)])
            r["ncu"] = {"nvtx": f"tower_{path}", "calls": calls, "window_s": round(time.perf_counter() - t0, 2)}
            r["launches"] = launches(enc, call, graphs, None)
        out[path] = r
    if "graph" in paths and "eager" in paths:
        call(True)
        g = call.output()
        call(False)
        out["graph_vs_eager"] = round(rel(g, call.output()), 6)
    return out


def table(res: dict[str, Any]) -> str:
    paths = [p for p in ("graph", "eager") if p in res and "cupti" in res[p]]
    if not paths:
        return ""
    out = [f"{'':24}" + "".join(f"{p:>20}" for p in paths)]

    def row(label: str, fn: Any) -> None:
        out.append(f"{label:24}" + "".join(f"{fn(res[p]):>20}" for p in paths))

    row("device ms/call (p50)", lambda r: f"{r['timing']['device_ms']['p50']:.3f}")
    row("kernel ms/call", lambda r: f"{r['cupti']['kernel_ms']:.3f}")
    row("kernels/call", lambda r: f"{r['cupti']['kernels']:.0f}")
    row("launches/call", lambda r: str(r["launches"].get("launches", r["launches"].get("eager_kernels"))))
    row("device busy ms/call", lambda r: f"{r['cupti']['timeline']['device_busy_ms']:.3f}")
    row("graph launches/call", lambda r: f"{r['cupti']['timeline']['graph_launches']:.0f}")
    row("graph gaps ms/call", lambda r: f"{r['cupti']['timeline']['graph_gaps_ms']:.3f}")
    row("fused glue/call", lambda r: f"{r['fused_glue_launches']:g}" if "fused_glue_launches" in r else "-")
    for c in res[paths[0]]["cupti"]["categories"]:
        row(c, lambda r, c=c: f"{r['cupti']['categories'][c]['ms']:.3f} / {r['cupti']['categories'][c]['calls']:.0f}")
    return "\n".join(out)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", type=Path, required=True, help="a Gemma 4 checkpoint with its vision tower")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mode", choices=("torch", "ncu"), default="torch")
    ap.add_argument("--calls", type=int, default=None, help="calls per path (default 50; ncu: 2)")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--eager", action="store_true", help="also the call without graphs")
    ap.add_argument("--max-soft-tokens", type=int, default=280,
                    help="images.max_soft_tokens (Gemma 4's own budget; the reference image is checked at 280)")
    ap.add_argument("--budget", type=int, default=2048, help="max_num_batched_tokens (the encoder budget's floor)")
    ap.add_argument("--image-size", default="1280x720", help="the synthetic JPEG's width x height")
    ap.add_argument("--all-buckets", action="store_true", help="capture every bucket, as the runner does")
    ap.add_argument("--trace", type=Path, default=None, help="also each path's Chrome trace, in this directory")
    args = ap.parse_args(argv)
    args.calls = args.calls or (2 if args.mode == "ncu" else 50)
    return args


def main() -> None:
    args = parse_args()
    import torch

    from llm_mini_infer.runner.cuda.images import ImageEncoder, VisionOptions
    from llm_mini_infer.vision import family as vf

    def log(msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    width, height = (int(x) for x in args.image_size.lower().split("x"))
    fam = vf.load(args.model, max_soft_tokens=args.max_soft_tokens)
    if fam is None or not hasattr(fam, "pooling"):
        raise SystemExit(f"{args.model} has no Gemma 4 vision tower")
    data = image_jpeg(width, height)
    shape = image_shape(fam, data, args.budget)
    notes = check_reference(shape)
    for n in notes:
        log(f"note: {n}")
    log(f"image {width}x{height}: {shape['size'][1]}x{shape['size'][0]} pixels, grid {shape['grid']}, "
        f"{shape['patches']} patches, {shape['tokens']} soft tokens; bucket {shape['bucket']} of {shape['buckets']}")
    dev = torch.device("cuda", torch.cuda.current_device())
    t0 = time.perf_counter()
    enc = ImageEncoder(args.model, VisionOptions(encoder_rows=2 * shape["encoder_budget"],
                                                 encoder_budget=shape["encoder_budget"],
                                                 max_soft_tokens=args.max_soft_tokens), dev, torch.bfloat16)
    if enc.buckets != shape["buckets"]:
        raise AssertionError(f"the encoder's buckets {enc.buckets} are not the family's {shape['buckets']}")
    enc.allocate(enc.cfg.out_hidden, flashinfer=True)
    call = TowerCall(enc, fam, data)
    call(False)  # one eager call first: cuBLAS, FlashInfer and our kernels set up outside the captures
    torch.cuda.synchronize()
    cap = capture(enc, enc.buckets if args.all_buckets else [shape["bucket"]])
    log(f"tower loaded and {len(cap['buckets'])} bucket(s) captured in {time.perf_counter() - t0:.0f} s")
    sampler = None
    if args.mode == "torch":
        init_cupti()
        sampler = PowerSampler(torch.cuda.current_device()).start()
    try:
        res = profile(enc, call, paths=["graph", "eager"] if args.eager else ["graph"], calls=args.calls,
                      warmup=args.warmup, mode=args.mode, sampler=sampler, trace=args.trace, log=log)
    finally:
        if sampler is not None:
            sampler.stop()
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config.update(versions=versions(), gpu=gpu_info(sampler or PowerSampler(torch.cuda.current_device())),
                  tower={"depth": enc.cfg.depth, "hidden": enc.cfg.hidden, "heads": enc.cfg.heads,
                         "head_dim": enc.cfg.head_dim, "attention_pad": getattr(enc.attn, "pad", None),
                         "weight_bytes": enc.weight_bytes})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"config": config, "image": shape, "notes": notes, "capture": cap, **res},
                                   indent=1))
    log(f"wrote {args.out}")
    print(table(res), flush=True)


if __name__ == "__main__":
    main()
