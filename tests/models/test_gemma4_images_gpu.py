"""Gemma 4 with images on CUDA: GemmaRunner in BF16 with our kernels, FlashInfer's sliding window and its second,
non-causal pass over image spans, the head-512 kernel, the vision tower on FlashInfer (head dim padded, scale 1.0) and
CUDA graphs, against transformers in fp32 on the same BF16 weights. A tiny model with the real head dims (sliding 256,
full 512) and a window of 40 (spans of 50-70 soft tokens are longer: the clamp is exercised). CUDA only."""

from __future__ import annotations

import pytest
import torch

from ..support.gemma_images import Engine, greedy_mm, hf_mm_logits_causal, png, text
from ..support.tiny_gemma import hf_mm_logits, hf_mm_model, make_tiny_gemma_mm

pytestmark = pytest.mark.gpu(reason="needs a CUDA GPU and the kernel library")
W = 40


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """BF16 weights, the q and k norms' weights a quarter (as `test_gemma4_gpu`'s fixture: moderate attention logits at
    scale 1.0)."""
    from safetensors.torch import load_file, save_file

    path = make_tiny_gemma_mm(tmp_path_factory.mktemp("g4imgpu"), window=W, depth=2, gpu=True, seed=4)
    f = path / "model.safetensors"
    sd = load_file(f)
    sd = {k: v * 0.25 if k.endswith(("q_norm.weight", "k_norm.weight")) and "language" in k else v
          for k, v in sd.items()}
    save_file({k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in sd.items()}, f)
    return path


def prompt_of(eng: Engine) -> tuple[list[int], list, list]:
    return eng.prompt([text(30, 1), png(48, 36, 1), text(9, 2), png(30, 60, 2), text(7, 3)])


def logits_of(eng: Engine, rid: int, n: int) -> torch.Tensor:
    return torch.stack([eng.logits[(rid, p)] for p in range(n)]).float()


@pytest.mark.parametrize("per_image", [True, False], ids=["per-image", "one-call"])
def test_the_memory_profile_encodes_images_within_the_position_table(tiny, per_image) -> None:
    """The startup memory profile with images and CUDA graphs at a 91-token budget, in a fresh process (a device-side
    assert ends its CUDA context): its image step encodes an image of 70 soft tokens (the tiny tower's largest) and
    one of 19 (`images.profile_images`), where it encoded one of 89, a prime count, whose 3 x 267-patch strip indexed
    past the tiny tower's 256-entry position table (a device-side assert at startup). With one tower call per image
    (batch invariance's; the image spans on our window kernel) the tower's graphs end at its largest image's bucket,
    with one call for the step's images (the spans on FlashInfer's second pass) at the budget's."""
    import os
    import subprocess
    import sys

    code = ("from llm_mini_infer.runner.cuda.gemma import GemmaRunner\n"
            "from llm_mini_infer.runner.cuda.images import VisionOptions\n"
            f"r = GemmaRunner({str(tiny)!r}, device='cuda', max_num_seqs=4, max_num_batched_tokens=91, "
            f"max_model_len=512, memory_utilization=0.5, graphs=True, encode_per_image={per_image}, "
            f"window_split_keys={16 if per_image else 0}, "
            "vision=VisionOptions(encoder_rows=192, encoder_budget=91, max_soft_tokens=70))\n"
            "plans, plan = [], r._image_plan\n"
            "r._image_plan = lambda *a, **kw: plans.append(kw['images']) or plan(*a, **kw)\n"
            "caps = r.capabilities()\n"
            "assert caps.memory_free_bytes > 0 and r.stats['graph_pool_bytes'] > 0\n"
            "print('images', plans, 'vision graphs', r.stats['graphs']['vision'])\n")
    env = {k: v for k, v in os.environ.items() if k not in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")}
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=600)
    want = [261, 387, 513, 648] if per_image else [261, 387, 513, 648, 819]
    assert res.returncode == 0 and f"images [[70, 19]] vision graphs {want}" in res.stdout, \
        res.stdout[-1000:] + res.stderr[-3000:]


def test_bf16_image_prompt_is_within_the_reference_paths_noise(tiny, tmp_path) -> None:
    """Our kernels in BF16 (chunks cut around both spans: a budget of 96) against transformers in fp32: as close as
    the torch reference path in BF16 (the same runner, torch attention and ops), and much closer to the bidirectional
    reference than to the causal one at the image positions (the span pass took)."""
    ours_eng = Engine(tiny, tmp_path, budget=96, caching=False, device="cuda", graphs=False, wps=16, num_pages=64)
    ids, refs, pils = prompt_of(ours_eng)
    ours_eng.add(0, ids, refs, 4)
    ours_eng.run()
    ref_eng = Engine(tiny, tmp_path, budget=96, caching=False, device="cuda", graphs=False, wps=16, num_pages=64,
                     attention="torch", fused_ops=False)
    ids2, refs2, _ = prompt_of(ref_eng)
    ref_eng.add(0, ids2, refs2, 4)
    ref_eng.run()
    hf = hf_mm_logits(tiny, ids, pils)
    ours, torch_bf16 = logits_of(ours_eng, 0, len(ids)), logits_of(ref_eng, 0, len(ids))
    err, noise = (ours - hf).abs().max().item(), (torch_bf16 - hf).abs().max().item()
    assert err <= 1.25 * noise + 1e-2, (err, noise)
    causal = hf_mm_logits_causal(tiny, ids, pils)
    span = [p for r in refs for p in range(r.offset, r.offset + r.tokens)]
    d_bidi = (ours[span] - hf[span]).abs().mean().item()
    d_causal = (ours[span] - causal[span]).abs().mean().item()
    assert d_bidi * 4 < d_causal, (d_bidi, d_causal)


def spy_modes(eng: Engine) -> list[str]:
    """The step kinds the runner picks (`eager`, `decode`, `break`, ...), in order."""
    modes, mode = [], eng.runner._mode

    def spy(lay, split=False):
        m = mode(lay, split)
        modes.append(m[0])
        return m

    eng.runner._mode = spy
    return modes


def test_graphs_give_the_eager_tokens_with_images(tiny, tmp_path) -> None:
    """The breakable graphs (the span pass inside the attention breaks), the decode graphs and the vision tower's
    graphs give the eager path's greedy tokens; a second request hits the first one's prefix past both images. No
    logits hook: with one, the runner runs every step eagerly."""
    outs = []
    for graphs in (False, True):
        eng = Engine(tiny, tmp_path, budget=96, caching=True, device="cuda", graphs=graphs, wps=16, num_pages=64,
                     hook=False)
        modes = spy_modes(eng)
        ids, refs, _ = prompt_of(eng)
        eng.add(0, ids, refs, 6)
        eng.run()
        eng.add(1, ids, refs, 6)
        eng.run()
        outs.append((eng.outs[0], eng.outs[1], eng.cached[1]))
        if graphs:
            assert eng.runner.img.graph_buckets(), "no vision-tower graph was captured"
            assert "decode" in modes and "break" in modes, modes
    assert outs[0] == outs[1]
    assert outs[0][0] == outs[0][1] and outs[0][2] >= max(r.offset + r.tokens for r in refs)


def test_images_with_mtp_give_mtp_offs_tokens(tiny, tmp_path) -> None:
    """An image prompt with the assistant (k = 3) in BF16 with graphs (decode graphs of verify rows, breakable graphs
    with the span pass): MTP off's greedy tokens up to a near tie of transformers' fp32 logits, and a second request
    hitting past both images."""
    from ..support.gemma_gpu import NEAR, to_bf16
    from ..support.tiny_gemma import make_tiny_assistant

    asst = make_tiny_assistant(tmp_path / "assistant", tiny, hidden=256, intermediate=512, seed=5)
    to_bf16(asst, ("q_norm.weight",))
    outs = []
    for k in (0, 3):
        eng = Engine(tiny, tmp_path, budget=96, caching=True, device="cuda", graphs=True, wps=16, num_pages=64,
                     mtp_k=k, assistant=asst if k else None, hook=False)
        modes = spy_modes(eng)
        ids, refs, pils = prompt_of(eng)
        eng.add(0, ids, refs, 24)
        eng.run()
        eng.add(1, ids, refs, 24)
        eng.run()
        assert eng.cached[1] >= max(r.offset + r.tokens for r in refs), k
        assert modes.count("decode") > 5, (k, modes)
        if k:
            assert eng.sched.spec_draft_tokens > 0
        outs.append(dict(eng.outs))
    same = 0
    for rid in (0, 1):
        on, off = outs[1][rid], outs[0][rid]
        i = next((j for j, (x, y) in enumerate(zip(on, off)) if x != y), None)
        if i is None:
            same += len(off)
            continue
        top = hf_mm_logits(tiny, ids + off, pils)[len(ids) + i - 1].topk(2).values
        assert (top[0] - top[1]).item() < NEAR, (rid, i, (top[0] - top[1]).item())
        same += i
    assert same > 0.6 * 48, same


def test_greedy_matches_transformers_where_the_top_two_are_apart(tiny, tmp_path) -> None:
    """Greedy decoding after an image prompt in BF16: the first tokens equal transformers' fp32 greedy (a random tiny
    model's logits are far apart at its first positions)."""
    eng = Engine(tiny, tmp_path, budget=128, caching=False, device="cuda", graphs=True, wps=16, num_pages=64)
    ids, refs, pils = prompt_of(eng)
    eng.add(0, ids, refs, 4)
    eng.run()
    ref = greedy_mm(hf_mm_model(tiny), ids, pils, 4)
    assert eng.outs[0][:2] == ref[:2], (eng.outs[0], ref)


def test_the_tower_on_flashinfer_matches_the_reference(tiny) -> None:
    """The vision tower in BF16 on FlashInfer (head dim padded, scale 1.0) and in its graphs against the SDPA reference
    in fp32 (the same weights): within BF16 noise; a graph equals eager on the same GEMM shapes (the call plus its
    bucket's padding as a dummy image of zeros); a call past the largest bucket runs eagerly."""
    from llm_mini_infer.kernels.flashinfer import FlashInferVisionAttention
    from llm_mini_infer.models.gemma4.vision import load_gemma_vision
    from llm_mini_infer.models.qwen3_5.vision import TorchVisionAttention
    from llm_mini_infer.runner.cuda.graphs import no_gc
    from llm_mini_infer.runner.cuda.images import VisionGraphs
    from llm_mini_infer.vision import family as vf
    from llm_mini_infer.vision.probe import probe

    fam = vf.load(tiny)
    blobs = [png(48, 36, 7), png(70, 20, 8)]
    sizes = [fam.size(probe(b)) for b in blobs]
    out = {}
    for dt, dev in ((torch.float32, "cpu"), (torch.bfloat16, "cuda")):
        enc = load_gemma_vision(tiny, torch.device(dev), dt)
        pxs = [fam.pixels(b, probe(b), s).to(dev) for b, s in zip(blobs, sizes)]
        patches = torch.cat([enc.patches(px) for px in pxs])
        meta = enc.meta([s.grid for s in sizes], torch.device(dev))
        c = enc.cfg
        attn = TorchVisionAttention(scale=c.scale) if dev == "cpu" else \
            FlashInferVisionAttention(c.heads, c.head_dim, dt, torch.device(dev), scale=c.scale)
        attn.plan(meta)
        with torch.no_grad():
            out[dev] = enc(patches, meta, attn).float().cpu()
            if dev == "cuda":
                pad = 4 * c.merge * c.merge  # a bucket 36 patches past the call
                g = VisionGraphs(enc, attn, [meta.n + pad], torch.cuda.graph_pool_handle())
                s = torch.cuda.Stream()  # graphs are captured off the default stream
                s.wait_stream(torch.cuda.current_stream())
                with no_gc(), torch.cuda.stream(s):
                    for b in sorted(g.buckets, reverse=True):
                        g.capture(b)
                torch.cuda.current_stream().wait_stream(s)
                out["graphs"] = g.run(patches, meta).float().cpu()
                meta2 = enc.meta([s_.grid for s_ in sizes] + [(1, c.merge, pad // c.merge)], torch.device(dev))
                attn.plan(meta2)
                n_tok = meta.n // (c.merge * c.merge)
                out["padded"] = enc(torch.cat([patches, patches.new_zeros(pad, patches.shape[1])]), meta2,
                                    attn)[:n_tok].float().cpu()
                g.buckets = [meta.n - c.merge * c.merge]  # no bucket holds the call: eager
                assert g.run(patches, meta) is None
    ref = out["cpu"]
    rel = ((out["cuda"] - ref).norm() / ref.norm()).item()
    assert rel < 0.02, rel  # BF16 through two layers
    assert torch.equal(out["graphs"], out["padded"])


def test_the_tower_s_fused_glue_changes_no_bit(tiny, monkeypatch) -> None:
    """The tower's attention input in our kernel (`vision_qkv`) gives the image embeddings of the torch ops that define
    it, bit for bit, eagerly and in the tower's graphs. The torch run turns the kernels off for `vision_qkv` alone (the
    only reader of its module's `use_kernels`): the tower's other kernels run in both."""
    from llm_mini_infer.kernels.flashinfer import FlashInferVisionAttention
    from llm_mini_infer.models.gemma4 import vision as gemma4_vision
    from llm_mini_infer.models.gemma4.vision import load_gemma_vision
    from llm_mini_infer.runner.cuda.graphs import no_gc
    from llm_mini_infer.runner.cuda.images import VisionGraphs
    from llm_mini_infer.vision import family as vf
    from llm_mini_infer.vision.probe import probe

    fam = vf.load(tiny)
    blobs = [png(48, 36, 7), png(70, 20, 8), png(36, 36, 9)]
    sizes = [fam.size(probe(b)) for b in blobs]
    dev = torch.device("cuda")
    enc = load_gemma_vision(tiny, dev, torch.bfloat16)
    patches = torch.cat([enc.patches(fam.pixels(b, probe(b), s).to(dev)) for b, s in zip(blobs, sizes)])
    meta = enc.meta([s.grid for s in sizes], dev)
    c = enc.cfg
    attn = FlashInferVisionAttention(c.heads, c.head_dim, torch.bfloat16, dev, scale=c.scale)
    attn.plan(meta)
    out = {}
    for fused in (True, False):
        with monkeypatch.context() as m, torch.no_grad():
            if not fused:
                m.setattr(gemma4_vision, "use_kernels", lambda t: False)
            out[fused] = enc(patches, meta, attn)
            g = VisionGraphs(enc, attn, [meta.n + 4 * c.merge * c.merge], torch.cuda.graph_pool_handle())
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            # no collection mid-capture, as in the runner (`graphs.no_gc`): the first loop's graphs are garbage in a
            # cycle (through their breaks' closures), and freeing a graph during a capture invalidates the capture
            with no_gc(), torch.cuda.stream(s):
                g.capture(g.buckets[0])
            torch.cuda.current_stream().wait_stream(s)
            out[("graphs", fused)] = g.run(patches, meta).clone()
    assert torch.equal(out[True], out[False])
    assert torch.equal(out[("graphs", True)], out[("graphs", False)])


def test_concurrent_image_prompts_in_one_step_are_within_the_reference_noise(tiny, tmp_path) -> None:
    """Three image prompts prefilled in the same steps (several rows with spans in one FlashInfer plan, each span
    gathered and scattered at its row's offset): each request's logits as close to transformers' fp32 as the torch
    reference path's in BF16, and its spans far from the causal reference."""
    parts = [[text(30, 1), png(48, 36, 1), text(9, 2), png(30, 60, 2), text(7, 3)],
             [text(5, 4), png(40, 40, 3), text(21, 5)],
             [text(44, 6), png(60, 30, 4), text(3, 7), png(36, 48, 5), text(12, 8)]]
    logits = {}
    for kind in ("ours", "torch"):
        kw = {} if kind == "ours" else {"attention": "torch", "fused_ops": False}
        eng = Engine(tiny, tmp_path, budget=512, caching=False, device="cuda", graphs=False, wps=16, num_pages=96,
                     max_num_seqs=4, enc_rows=512, enc_budget=400, **kw)
        prompts = [eng.prompt(p) for p in parts]
        for rid, (ids, refs, _) in enumerate(prompts):
            eng.add(rid, ids, refs, 2)
        eng.run()
        logits[kind] = [logits_of(eng, rid, len(ids)) for rid, (ids, _, _) in enumerate(prompts)]
    for rid, (ids, refs, pils) in enumerate(prompts):
        hf = hf_mm_logits(tiny, ids, pils)
        err = (logits["ours"][rid] - hf).abs().max().item()
        noise = (logits["torch"][rid] - hf).abs().max().item()
        assert err <= 1.25 * noise + 1e-2, (rid, err, noise)
        causal = hf_mm_logits_causal(tiny, ids, pils)
        span = [p for r in refs for p in range(r.offset, r.offset + r.tokens)]
        d_bidi = (logits["ours"][rid][span] - hf[span]).abs().mean().item()
        d_causal = (logits["ours"][rid][span] - causal[span]).abs().mean().item()
        assert d_bidi * 4 < d_causal, (rid, d_bidi, d_causal)
