# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 the HuggingFace Team. All rights reserved.
# aspect_ratio_size is adapted from src/transformers/models/gemma4/image_processing_gemma4.py of transformers 5.17
# (https://github.com/huggingface/transformers), and changed for llm-mini-infer.
"""Images for Gemma 4 (`Gemma4ForConditionalGeneration`): transformers' `Gemma4ImageProcessor` (its torchvision backend,
the processor class `processor_config.json` names) and `Gemma4Processor`'s prompt expansion.

- **Sizing** (`size`, `aspect_ratio_size`): HF's `get_aspect_ratio_preserving_size`, operation for operation. The
  target holds at most `max_soft_tokens` × pooling² patches of 16 × 16 (645,120 pixels at 280 soft tokens), keeps the
  aspect ratio and makes both sides multiples of pooling × patch (48); small images are scaled up to it, as large ones
  down. The image then takes (h / 16) × (w / 16) / 9 soft tokens: a 16:9 image 264 at 280 (576 × 1056), never more
  than the budget. (vLLM computes the count with its own formula; HF's is the reference here. They differ only on rare
  thin strips at small budgets, where vLLM's count is one lower.)
- **The budget** is `images.max_soft_tokens`, server-wide, else the checkpoint's (`from_model_dir`). `min_pixels` /
  `max_pixels` follow from it and cannot be set apart.
- **Pixels** (`pixels`, in the image pool): the shared exact uint8 path (`family.decode_rgb`), then the backend's
  antialiased bicubic resize on uint8 (`family.resize_bicubic`), skipped when the target is the image's own size (HF
  skips it too). Rescaling and patchifying run on the GPU (`models.gemma4.vision`).
- **The prompt:** the chat template renders one `<|image|>` per image; the frontend expands it to `<|image>`, the
  image's soft tokens (`<|image|>` each: the span) and `<image|>`, as HF's processor does in text. Only the soft
  tokens are the span: they take the encoder's embeddings and, in the sliding layers, attend to each other in both
  directions (`use_bidirectional_attention: "vision"`); `<|image>` and `<image|>` are text. The frontend checks the
  expansion's token ids against the tokenizer at startup (`check_tokenizer`).
- **Positions** are the sequence indices (Gemma 4's RoPE is 1-D).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .family import ImageFamily, ImageSize, Span, decode_rgb, resize_bicubic
from .probe import Header

SOFT_TOKENS = (70, 140, 280, 560, 1120)  # HF's _SUPPORTED_SOFT_TOKENS
BICUBIC = 3  # PIL's resample code, the only one HF's Gemma 4 checkpoints use


def aspect_ratio_size(height: int, width: int, patch: int, max_patches: int, pooling: int) -> tuple[int, int]:
    """transformers' Gemma 4 `get_aspect_ratio_preserving_size`, operation for operation (its `int(math.floor(x))` is
    `math.floor(x)`): (target height, width), the largest that keep the aspect ratio, hold at most `max_patches`
    patches and have sides divisible by pooling × patch. ValueError where HF raises."""
    total_px = height * width
    target_px = max_patches * (patch ** 2)
    factor = math.sqrt(target_px / total_px)
    ideal_height = factor * height
    ideal_width = factor * width
    side_mult = pooling * patch
    target_height = math.floor(ideal_height / side_mult) * side_mult
    target_width = math.floor(ideal_width / side_mult) * side_mult
    if target_height == 0 and target_width == 0:
        raise ValueError("Attempting to resize to a 0 x 0 image. Resized height should be divisible by "
                         f"`pooling_kernel_size * patch_size`={pooling * patch}.")
    max_side_length = (max_patches // pooling ** 2) * side_mult
    if target_height == 0:
        target_height = side_mult
        target_width = min(math.floor(width / height) * side_mult, max_side_length)
    elif target_width == 0:
        target_width = side_mult
        target_height = min(math.floor(height / width) * side_mult, max_side_length)
    if target_height * target_width > target_px:
        raise ValueError(f"Resizing [{height}x{width}] to [{target_height}x{target_width}] but this exceeds "
                         f"{max_patches} patches with patch_size {patch}")
    return target_height, target_width


class GemmaVision(ImageFamily):
    name = "gemma4"
    max_soft_tokens_source = "the family's argument"  # from_model_dir: where the budget came from (config's reason)

    def __init__(self, *, boi_id: int, image_id: int, eoi_id: int, patch: int = 16, pooling: int = 3,
                 max_soft_tokens: int = 280, rescale: float = 1 / 255, normalize: bool = False,
                 mean: tuple[float, ...] = (0.0, 0.0, 0.0), std: tuple[float, ...] = (1.0, 1.0, 1.0)):
        if max_soft_tokens not in SOFT_TOKENS:
            raise ValueError(f"max_soft_tokens must be one of {SOFT_TOKENS}, got {max_soft_tokens}")
        self.start_id, self.pad_id, self.end_id = boi_id, image_id, eoi_id
        self.placeholder = "<|image|>"
        self.span_prefix, self.span_suffix = (boi_id,), (eoi_id,)
        self.patch, self.pooling, self.max_soft_tokens = patch, pooling, max_soft_tokens
        self.max_patches = max_soft_tokens * pooling ** 2
        self.rescale, self.normalize, self.mean, self.std = rescale, normalize, tuple(mean), tuple(std)
        self.min_pixels = (pooling * patch) ** 2  # one soft token
        self.max_pixels = self.max_patches * patch * patch  # every target holds at most this many pixels

    @classmethod
    def from_model_dir(cls, model_dir: Path, hf: dict[str, Any], *, min_pixels: int | None = None,
                       max_pixels: int | None = None, max_soft_tokens: int | None = None) -> GemmaVision:
        """The checkpoint's family at the budget `max_soft_tokens` (images.max_soft_tokens), else the checkpoint's own:
        `processor_config.json`'s image processor, as HF's processor reads it, then
        `vision_config.default_output_length` (the checkpoint's own default when the processor gives none), then
        transformers' 280. `min_pixels` / `max_pixels`, when given, must be the budget's."""
        path = model_dir / "processor_config.json"
        proc = json.loads(path.read_text()) if path.exists() else {}
        ip = proc.get("image_processor", proc)
        vc = hf["vision_config"]
        if ip.get("resample", BICUBIC) != BICUBIC or not ip.get("do_resize", True) or not ip.get("do_rescale", True):
            raise NotImplementedError(f"Gemma 4 image processing with resample {ip.get('resample')}, do_resize "
                                      f"{ip.get('do_resize')}, do_rescale {ip.get('do_rescale')}")
        if max_soft_tokens is not None:
            source = "images.max_soft_tokens"
        elif "max_soft_tokens" in ip:
            max_soft_tokens = ip["max_soft_tokens"]
            source = f"processor_config.json {'image_processor.' if ip is not proc else ''}max_soft_tokens"
        elif "default_output_length" in vc:
            max_soft_tokens, source = vc["default_output_length"], "config.json vision_config.default_output_length"
        else:
            max_soft_tokens, source = 280, "280 (transformers' default)"
        fam = cls(boi_id=int(hf.get("boi_token_id", 255999)), image_id=int(hf.get("image_token_id", 258880)),
                  eoi_id=int(hf.get("eoi_token_id", 258882)), patch=int(ip.get("patch_size", vc.get("patch_size", 16))),
                  pooling=int(ip.get("pooling_kernel_size", vc.get("pooling_kernel_size", 3))),
                  max_soft_tokens=int(max_soft_tokens), rescale=float(ip.get("rescale_factor", 1 / 255)),
                  normalize=bool(ip.get("do_normalize", False)), mean=tuple(ip.get("image_mean", (0.0, 0.0, 0.0))),
                  std=tuple(ip.get("image_std", (1.0, 1.0, 1.0))))
        fam.max_soft_tokens_source = source
        # images.min_pixels / max_pixels are the budget's (config._derive_images derives them for the pixel arena). The
        # check also trips a load site that forgets the budget and so loads at another (the checkpoint's).
        for name, value in (("min_pixels", min_pixels), ("max_pixels", max_pixels)):
            if value is not None and value != getattr(fam, name):
                raise ValueError(f"images.{name}: Gemma 4 sizes images by images.max_soft_tokens ({fam.max_soft_tokens}"
                                 f", {source}); its {name} is {getattr(fam, name)}, not {value}: set "
                                 "images.max_soft_tokens instead of pixel bounds (and pass it wherever the image "
                                 "family is loaded)")
        return fam

    # -- frontend --------------------------------------------------------------------------------------------------

    def size(self, header: Header) -> ImageSize:
        width, height = header.oriented
        h, w = aspect_ratio_size(height, width, self.patch, self.max_patches, self.pooling)
        grid = (1, h // self.patch, w // self.patch)
        return ImageSize(height=h, width=w, grid=grid, tokens=grid[1] * grid[2] // self.pooling ** 2)

    def max_tokens(self) -> int:
        return self.max_soft_tokens

    def check_tokenizer(self, tokenizer: Any, tokenizer_config: dict[str, Any]) -> None:
        """config.json's ids (`boi_token_id`, `image_token_id`, `eoi_token_id`), which the expansion uses, against the
        tokenizer's ids of the tokens HF's processor writes (tokenizer_config.json's `boi_token`, `image_token`,
        `eoi_token`): a ValueError naming the token and the ids where they differ, where the tokenizer lacks the token,
        or where tokenizer_config.json does not name it."""
        for key, want in (("boi_token", self.start_id), ("image_token", self.pad_id), ("eoi_token", self.end_id)):
            text = tokenizer_config.get(key)
            if isinstance(text, dict):  # a serialized AddedToken
                text = text.get("content")
            got = tokenizer.token_to_id(text) if isinstance(text, str) else None
            if got == want:
                continue
            if not isinstance(text, str):
                why = f"tokenizer_config.json names no {key}"
            elif got is None:
                why = f"the tokenizer has no {text}"
            else:
                why = f"the tokenizer's {text} is {got}"
            raise ValueError(f"Gemma 4's {key}_id is {want} in config.json, but {why}: images would not be expanded "
                             "as HF's processor expands them")

    # -- image pool ------------------------------------------------------------------------------------------------

    def pixels(self, data: bytes, header: Header, size: ImageSize) -> Any:
        img = decode_rgb(data, header)
        if tuple(img.shape[1:]) == (size.height, size.width):
            return img.contiguous()
        return resize_bicubic(img, size.height, size.width)

    # -- scheduler -------------------------------------------------------------------------------------------------

    def positions(self, n: int, spans: list[Span]) -> tuple[np.ndarray | None, int]:
        return None, 0

    # -- runner ----------------------------------------------------------------------------------------------------


    def load_encoder(self, model_dir: str | Path, device: Any, dtype: Any) -> Any:
        from ..models.gemma4.vision import load_gemma_vision

        return load_gemma_vision(model_dir, device, dtype)
