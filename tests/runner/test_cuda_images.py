"""Images through the CUDA runner on CPU (torch reference kernels, fp32), the real scheduler and the engine loop,
against transformers' `Qwen3_5ForConditionalGeneration` on a tiny random multimodal model.

Teacher-forced logits of prompts with images match HF's forward (pixel_values from the same pixels, HF's own M-RoPE
positions), whole or chunked across a span; greedy decoding continues HF's sequence (the M-RoPE offset after the
image); MTP's outputs equal MTP off's; a cache hit inside an image span gives the cold run's tokens.
"""

from __future__ import annotations

import pytest
import torch

from ..support.qwen_images import PAD, Engine, Img, prompt_of, text
from ..support.qwen_images import vl as vl  # the fixture

TOL = 1e-4  # fp32 on both sides; the patch embedding is a GEMM here, a Conv3d in HF


def hf_logits(model, ids: list[int], spans) -> torch.Tensor:
    from llm_mini_infer.models.qwen3_5.vision import QwenVisionEncoder

    t = torch.tensor([ids])
    kw = {}
    if spans:
        patches = torch.cat([QwenVisionEncoder.patches(_enc_stub(), im.px) for _, im in spans])
        kw = dict(pixel_values=patches, image_grid_thw=torch.tensor([[1, im.gh, im.gw] for _, im in spans]),
                  mm_token_type_ids=(t == PAD).int())
    with torch.no_grad():
        return model(input_ids=t, **kw).logits[0].float()


def _enc_stub():
    """`QwenVisionEncoder.patches` needs only the config and the normalisation buffers."""
    from llm_mini_infer.models.qwen3_5.vision import QwenVisionEncoder, VisionConfig, vision_buffers

    cfg = VisionConfig(depth=0, hidden=64, heads=4, intermediate=96, patch=16, temporal=2, merge=2, out_hidden=64,
                       num_pos=64, in_channels=3, act="gelu_pytorch_tanh", rope_theta=10000.0, mean=(0.5,) * 3,
                       std=(0.5,) * 3, rescale=1 / 255)
    e = QwenVisionEncoder.__new__(QwenVisionEncoder)
    e.cfg = cfg
    for k, v in vision_buffers(cfg).items():
        setattr(e, k, v)
    return e


@pytest.mark.parametrize("budget", [128, 8], ids=["one-step", "chunked"])
def test_prompt_logits_match_transformers(vl, tmp_path, budget) -> None:
    """One image (a 4 x 6 patch grid: 6 tokens) between text, prefilled in one step or in 8-token chunks that split
    the span: every prompt position's logits match HF's forward."""
    path, model = vl
    ids, spans = prompt_of([text(9), Img(4, 6, 1), text(7, 1)])
    e = Engine(path, tmp_path, budget=budget)
    e.add(0, ids, spans, max_tokens=1)
    e.run()
    ref = hf_logits(model, ids, spans)
    ours = e.prompt_logits(0, len(ids))
    assert (ours - ref).abs().max().item() < TOL
    assert e.out[0] == [int(ref[-1].argmax())]
    assert e.sched.encoder_images == 1


def test_two_images_and_greedy_decoding_continue_transformers(vl, tmp_path) -> None:
    """Two images (one twice as wide) and text between: the prompt matches HF, and 8 greedy tokens continue HF's
    sequence (their RoPE positions carry the offset after the last image)."""
    path, model = vl
    ids, spans = prompt_of([text(5, 2), Img(4, 4, 3), text(3, 3), Img(2, 8, 4), text(6, 4)])
    e = Engine(path, tmp_path, budget=64)
    e.add(0, ids, spans, max_tokens=8)
    e.run()
    gen = e.out[0]
    ref = hf_logits(model, ids + gen, spans)
    assert (e.prompt_logits(0, len(ids)) - ref[:len(ids)]).abs().max().item() < TOL
    assert [int(x) for x in ref[len(ids) - 1:-1].argmax(-1)] == gen


def test_mtp_outputs_equal_mtp_off(vl, tmp_path) -> None:
    """With MTP (k = 2) the drafter reads the image's rows at image positions and the verification keeps exactly the
    target's tokens: greedy outputs equal MTP off's, also when 8-token chunks end inside or right before the span."""
    path, _ = vl
    ids, spans = prompt_of([text(6, 5), Img(4, 6, 5), text(5, 6)])
    outs = []
    for k, budget in ((0, 128), (2, 128), (2, 8)):
        e = Engine(path, tmp_path, budget=budget, k=k)
        e.add(0, ids, spans, max_tokens=12)
        e.run()
        outs.append(e.out[0])
    assert outs[0] == outs[1] == outs[2]


def test_a_hit_inside_the_span_gives_the_cold_tokens(vl, tmp_path) -> None:
    """Three prompts with the same text and image and different text after it, 8-token chunks, the prefix cache on.
    The first leaves checkpoints at its chunk ends (8, 16, 24; the span is [5, 21)); the others resume at 16, inside
    the span, and read the image's cached embeddings for the rest of it. Every request's tokens equal the cache-off
    run's."""
    path, _ = vl
    img = Img(8, 8, 6)
    prompts = [prompt_of([text(4, 7), img, text(3 + i, 8 + i)]) for i in range(3)]
    outs = []
    for caching in (False, True):
        e = Engine(path, tmp_path, budget=8, caching=caching)
        for rid, (ids, spans) in enumerate(prompts):
            e.add(rid, ids, spans, max_tokens=6)
            e.run()
        outs.append(e.out)
    assert outs[0] == outs[1]
    assert e.sched.prefix_hits == 2 * 16 and e.sched.encoder_hits == 2 and e.sched.encoder_images == 1


def test_gemma_buckets_follow_max_soft_tokens() -> None:
    """Gemma 4's tower buckets at a 2,048-token encoder budget follow images.max_soft_tokens: dense every 128 patches
    to the largest image (at least 2,048), then about 1.25x apart to the budget's 18,432 patches; 70 drops the 280
    list's 2,178, 2,304 and 2,439. A 400 x 225 image at 70 (594 patches) takes the 648 bucket."""
    from llm_mini_infer.runner.cuda.images import family_buckets
    from llm_mini_infer.vision.gemma import GemmaVision

    def buckets(k: int) -> list[int]:
        return family_buckets(GemmaVision(boi_id=255999, image_id=258880, eoi_id=258882, max_soft_tokens=k), 3, 2048)

    at70 = [261, 387, 513, 648, 774, 900, 1026, 1152, 1287, 1413, 1539, 1665, 1800, 1926, 2052, 2565, 3204, 4104,
            5121, 6408, 8064, 10116, 12672, 15876, 18432]
    assert buckets(70) == at70
    assert sorted(set(buckets(280)) - set(at70)) == [2178, 2304, 2439] and set(at70) <= set(buckets(280))
    assert next(b for b in at70 if b >= 594) == 648


# Gemma 4's buckets with one tower call per image, by images.max_soft_tokens: (the top bucket, their count), the same at
# every encoder budget
PER_IMAGE_BUCKETS = {70: (648, 4), 140: (1287, 9), 280: (2565, 19), 560: (5121, 19), 1120: (10116, 22)}


def test_gemma_buckets_with_one_call_per_image_end_at_the_largest_image() -> None:
    """With one tower call per image (batch invariance), Gemma 4's buckets end at the first that holds its largest
    image (9 x max_soft_tokens patches, the most `aspect_ratio_size` gives): at each soft-token budget the same list
    at every encoder budget from 2,048 to 32,768 tokens, the start of the whole list, so every image up to the largest
    (each multiple of a soft token's 9 patches) takes the bucket it takes in the whole list, and always one. Without
    one call per image, and for Qwen3.8 (no soft-token budget), the list is the whole one."""
    from llm_mini_infer.runner.cuda.images import family_buckets
    from llm_mini_infer.vision.gemma import SOFT_TOKENS, GemmaVision
    from llm_mini_infer.vision.qwen import QwenVision

    def pick(buckets: list[int], n: int) -> int | None:
        return next((b for b in buckets if b >= n), None)

    assert sorted(PER_IMAGE_BUCKETS) == sorted(SOFT_TOKENS)
    for k in SOFT_TOKENS:
        fam = GemmaVision(boi_id=255999, image_id=258880, eoi_id=258882, max_soft_tokens=k)
        top, count = PER_IMAGE_BUCKETS[k]
        for budget in [*range(2048, 32769, 2048), 3435, 7168, 10240, 12288]:
            whole, ours = family_buckets(fam, 3, budget), family_buckets(fam, 3, budget, per_image=True)
            assert (ours[-1], len(ours)) == (top, count) and ours == whole[:count], (k, budget)
            assert all(pick(ours, n) == pick(whole, n) is not None for n in range(9, 9 * k + 1, 9)), (k, budget)
    qwen = QwenVision(start_id=0, pad_id=0, end_id=0, patch=16, merge=2, temporal=2, min_pixels=65536,
                      max_pixels=16777216, mean=(0.5,) * 3, std=(0.5,) * 3, rescale=1 / 255, mrope=True)
    for budget in (2048, 16384, 32768):
        assert family_buckets(qwen, 2, budget, per_image=True) == family_buckets(qwen, 2, budget)


def test_synthetic_images_fit_the_position_table() -> None:
    """The memory profile's synthetic images (`synthetic_grid`) at every encoder budget's B - 2 tokens up to 65,534
    stay within Gemma 4's 10,240-entry position table: the most square divisor pair where it fits (every token kept),
    else isqrt(t) rows (fewer than isqrt(t) tokens dropped). The divisor pair alone overruns the table at many counts
    from 3,433 on (a prime count's strip, 3 x 10,299 patches; at a 16,384-token budget 6 x 24,573)."""
    import math

    from llm_mini_infer.runner.cuda.images import synthetic_grid

    assert synthetic_grid(16382, 3) == (6, 24573) and synthetic_grid(16382, 3, 10240) == (381, 384)
    assert synthetic_grid(3432, 3, 10240) == synthetic_grid(3432, 3) and synthetic_grid(3433, 3) == (3, 10299)
    for t in range(280, 65535):
        h, w = synthetic_grid(t, 3, 10240)
        tokens = h * w // 9
        assert h % 3 == w % 3 == 0 and max(h, w) <= 10240 and 0 <= t - tokens < math.isqrt(t), t
        pair = synthetic_grid(t, 3)
        if max(pair) <= 10240:
            assert (h, w) == pair, t
        if t % 97 == 0:  # the pair: the largest divisor up to the square root
            a = max(d for d in range(1, math.isqrt(t) + 1) if t % d == 0)
            assert pair == (3 * a, 3 * (t // a)), t


def test_the_profile_encodes_a_budget_of_the_largest_images() -> None:
    """The memory profile's image step (`profile_images`): as many images of max_soft_tokens as the step's image tokens
    hold, then one of the rest; one image of them all below that, or without a soft-token budget (Qwen3.8)."""
    from llm_mini_infer.runner.cuda.images import profile_images

    assert profile_images(2046, 280) == [280] * 7 + [86]
    assert profile_images(32766, 70) == [70] * 468 + [6]
    assert profile_images(560, 280) == [280, 280]
    assert profile_images(198, 280) == [198]
    assert profile_images(2046, None) == [2046]
