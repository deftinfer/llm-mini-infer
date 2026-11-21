"""The tiny Gemma 4 pair on the GPU: its runner, BF16 and FP8 pairs, runs and references
(tests/models/test_gemma4_gpu.py, tests/models/test_gemma4_mtp_gpu.py, tests/kernels/test_pdl_gpu.py)."""

from __future__ import annotations

import pytest
import torch

from .gemma_mtp import Run
from .mtp_common import seq
from .tiny_gemma import hf_logits, make_tiny_assistant, make_tiny_gemma, make_tiny_gemma_fp8


def cuda_runner(model_dir, pc, graphs: bool, **kw):
    from llm_mini_infer.runner.cuda.gemma import GemmaRunner

    r = GemmaRunner(model_dir, device="cuda", max_num_seqs=pc.max_num_seqs,
                    max_num_batched_tokens=pc.max_num_batched_tokens, max_model_len=pc.max_model_len, memory_gb=2.0,
                    check=True, graphs=graphs, **kw)
    r.configure(pc)
    return r


W = 40
NEAR = 0.3  # twice the BF16 runs' largest logit error against transformers in fp32 (`test_gemma4_gpu`: < 0.15)
PROMPTS = [seq(11, 1), seq(57, 2), seq(4, 3), seq(90, 4)]  # two past the window


def to_bf16(path, scaled: tuple[str, ...]) -> None:
    """Save the checkpoint in BF16 (transformers in fp32 then runs exactly the weights we run), the weights ending in
    `scaled` a quarter (the q and k norms: moderate attention logits at scale 1.0, as `test_gemma4_gpu`'s fixture)."""
    from safetensors.torch import load_file, save_file

    f = path / "model.safetensors"
    sd = load_file(f)
    sd = {k: v * 0.25 if k.endswith(scaled) else v for k, v in sd.items()}
    save_file({k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in sd.items()}, f)


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    d = tmp_path_factory.mktemp("g4mtpgpu")
    target = make_tiny_gemma(d / "target", window=W, gpu=True, seed=3)
    to_bf16(target, ("q_norm.weight", "k_norm.weight"))
    assistant = make_tiny_assistant(d / "assistant", target, hidden=256, intermediate=512, seed=4)
    to_bf16(assistant, ("q_norm.weight",))
    return target, assistant


def gpu_run(pair, k: int, graphs: bool = True, **kw) -> Run:
    r = Run(pair, k, make=lambda t, pc, **rk: cuda_runner(t, pc, graphs, **rk), window=W, wps=16, **kw)
    r.modes = []
    r.decode_rows = []  # each step of decode rows only: (its mode, rows, the drafts they verify)
    r.steps = []  # every step: (its mode, device rows, rows, the drafts its SAMPLE rows make)
    mode = r.r._mode

    def spy(lay, split=False):
        m = mode(lay, split)
        r.modes.append(m[0])
        if lay.n_dev == lay.n:
            r.decode_rows.append((m[0], lay.n, lay.v))
        r.steps.append((m[0], lay.n_dev, lay.n, lay.dn))
        return m

    r.r._mode = spy
    return r


def run_prompts(r: Run, oracle=None, rng=None, n: int = 40) -> Run:
    for i, p in enumerate(PROMPTS):
        r.add(i, p, max_tokens=n)
    r.run(oracle, rng)
    r.r.shutdown()
    return r


def reference(target, outs: dict[int, list[int]], prompts=PROMPTS) -> dict[int, torch.Tensor]:
    return {rid: hf_logits(target, prompts[rid] + o) for rid, o in outs.items()}


def agree(got: dict[int, list[int]], want: dict[int, list[int]], ref: dict[int, torch.Tensor], prompts=PROMPTS) -> int:
    """`got` follows `want` (rid -> tokens) up to a near tie of the reference logits (`want`'s sequences, teacher
    forced); returns how many tokens agree before the partings, every request together."""
    same = 0
    for rid, w in want.items():
        g, start = got[rid], len(prompts[rid])
        i = next((j for j, (x, y) in enumerate(zip(g, w)) if x != y), None)
        if i is None:
            assert len(g) == len(w), rid
            same += len(w)
            continue
        top = ref[rid][start + i - 1].topk(2).values
        assert (top[0] - top[1]).item() < NEAR, (rid, i, g[i], w[i], (top[0] - top[1]).item())
        same += i
    return same


@pytest.fixture(scope="module")
def fp8_pair(tmp_path_factory):
    d = tmp_path_factory.mktemp("pdlgpu")
    target, _ = make_tiny_gemma_fp8(d / "target", seed=5, gpu=True)
    assistant = make_tiny_assistant(d / "assistant", target, hidden=256, intermediate=512, seed=4)
    to_bf16(assistant, ("q_norm.weight",))
    return target, assistant
