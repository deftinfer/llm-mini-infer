# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
# smart_resize is adapted from src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py of transformers
# (https://github.com/huggingface/transformers), and changed for llm-mini-infer.
"""Images for Qwen3.5 / Qwen3.8 (`Qwen3_5ForConditionalGeneration`): the Qwen2-VL image processor's rules.

- **Sizing** (`size`): transformers' `smart_resize` with factor = patch × merge (32), operation for operation, on the
  stored size (symmetric under a transpose: the pool turns the target with the image, `family`). The grid is
  (1, h / 16, w / 16) patches and the image takes h × w / (16 × 16 × 4) tokens: 84 for a 400×225 photo (384 × 224).
- **Pixels** (`pixels`, in the image pool): the shared exact uint8 path (`family.decode_rgb`), then the antialiased
  bicubic resize on uint8 the processor's torchvision backend runs (`family.resize_bicubic`). Normalisation and
  patchifying run on the GPU (`models.qwen3_5.vision`).
- **Positions** (`positions`): interleaved M-RoPE, transformers' `get_rope_index`. Text continues the running position;
  an image's tokens take (t, h, w) = (s, s + row, s + column) over its merged grid, row-major, where s is the position
  the next text token would have had, and the text after it continues from s + max(rows, columns). Every token after
  the last image sits at its sequence index plus a constant offset (HF's `rope_deltas`), which the runner adds to the
  device-side positions of decode steps.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .family import ImageFamily, ImageSize, Span, decode_rgb, resize_bicubic, target
from .probe import Header

MAX_ASPECT = 200  # transformers rejects images whose sides differ by more than this factor


def smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    """transformers' Qwen2-VL `smart_resize`, operation for operation (`round` is Python's: half to even)."""
    if max(height, width) / min(height, width) > MAX_ASPECT:
        raise ValueError(f"absolute aspect ratio must be smaller than {MAX_ASPECT}, got "
                         f"{max(height, width) / min(height, width)}")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


class QwenVision(ImageFamily):
    name = "qwen3_5"

    def __init__(self, *, start_id: int, pad_id: int, end_id: int, patch: int, merge: int, temporal: int,
                 min_pixels: int, max_pixels: int, mean: tuple[float, ...], std: tuple[float, ...],
                 rescale: float, mrope: bool):
        self.start_id, self.pad_id, self.end_id = start_id, pad_id, end_id
        self.placeholder = "<|vision_start|><|image_pad|><|vision_end|>"
        self.patch, self.merge, self.temporal = patch, merge, temporal
        self.factor = patch * merge
        self.min_pixels, self.max_pixels = min_pixels, max_pixels
        self.mean, self.std, self.rescale = mean, std, rescale
        self.mrope = mrope
        if min_pixels > max_pixels:
            raise ValueError(f"images.min_pixels {min_pixels} > images.max_pixels {max_pixels}")

    @classmethod
    def from_model_dir(cls, model_dir: Path, hf: dict[str, Any], *, min_pixels: int | None = None,
                       max_pixels: int | None = None) -> QwenVision:
        pre_path = model_dir / "preprocessor_config.json"
        pre = json.loads(pre_path.read_text()) if pre_path.exists() else {}
        size = pre.get("size") or {}
        vc = hf["vision_config"]
        rope = (hf.get("text_config") or hf).get("rope_parameters") or {}
        return cls(start_id=int(hf.get("vision_start_token_id", 248053)),
                   pad_id=int(hf.get("image_token_id", 248056)), end_id=int(hf.get("vision_end_token_id", 248054)),
                   patch=int(pre.get("patch_size", vc.get("patch_size", 16))),
                   merge=int(pre.get("merge_size", vc.get("spatial_merge_size", 2))),
                   temporal=int(pre.get("temporal_patch_size", vc.get("temporal_patch_size", 2))),
                   min_pixels=int(min_pixels if min_pixels is not None else size.get("shortest_edge", 65536)),
                   max_pixels=int(max_pixels if max_pixels is not None else size.get("longest_edge", 16777216)),
                   mean=tuple(pre.get("image_mean", (0.5, 0.5, 0.5))),
                   std=tuple(pre.get("image_std", (0.5, 0.5, 0.5))),
                   rescale=float(pre.get("rescale_factor", 1 / 255)), mrope=bool(rope.get("mrope_section")))

    # -- frontend --------------------------------------------------------------------------------------------------

    def size(self, header: Header) -> ImageSize:
        h, w = smart_resize(header.height, header.width, self.factor, self.min_pixels, self.max_pixels)
        grid = (1, h // self.patch, w // self.patch)
        return ImageSize(height=h, width=w, grid=grid, tokens=grid[1] * grid[2] // (self.merge * self.merge))

    def max_tokens(self) -> int:
        return self.max_pixels // (self.factor * self.factor)

    # -- image pool ------------------------------------------------------------------------------------------------

    def pixels(self, data: bytes, header: Header, size: ImageSize) -> Any:
        img = decode_rgb(data, header)
        return resize_bicubic(img, *target(img, header, size))

    # -- scheduler -------------------------------------------------------------------------------------------------

    def positions(self, n: int, spans: list[Span]) -> tuple[np.ndarray | None, int]:
        if not self.mrope or not spans:
            return None, 0
        rope = np.empty((3, n), np.int64)
        cur = prev = 0
        m = self.merge
        for sp in spans:
            off, k = sp.offset, sp.size.tokens
            if off < prev or off + k > n:
                raise ValueError(f"image span [{off}, {off + k}) outside the prompt or overlapping another")
            rope[:, prev:off] = cur + np.arange(off - prev)
            cur += off - prev
            gh, gw = sp.size.grid[1] // m, sp.size.grid[2] // m
            rope[0, off:off + k] = cur
            rope[1, off:off + k] = cur + np.repeat(np.arange(gh), gw)
            rope[2, off:off + k] = cur + np.tile(np.arange(gw), gh)
            cur += max(gh, gw)
            prev = off + k
        rope[:, prev:] = cur + np.arange(n - prev)
        return rope.astype(np.int32), cur - prev

    # -- runner ----------------------------------------------------------------------------------------------------


    def load_encoder(self, model_dir: str | Path, device: Any, dtype: Any) -> Any:
        from ..models.qwen3_5.vision import load_vision

        return load_vision(model_dir, device, dtype)
