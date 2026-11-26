"""The per-family image interface: everything model-specific about images, in one object (architecture.md, "The
vision encoder and its cache").

- **Frontend:** the placeholder the chat template renders per image and its token ids, and header-only sizing: the
  resize target and the number of placeholder tokens (`size`).
- **Image pool:** the exact CPU half, decode and resize to the target into uint8 pixels (`pixels`).
- **Scheduler:** the positions the model's RoPE takes over a prompt with images (`positions`): None when the family
  uses 1-D positions, else [3, n] and the offset that later tokens add to their sequence index.
- **Runner:** the vision encoder with the checkpoint's weights (`load_encoder`), which rescales or normalises and
  patchifies the pixels on the GPU.

**The span in the prompt.** The chat template renders one placeholder per image; the frontend expands its token into
`span_prefix`, `size().tokens` copies of `pad_id` (the span: the positions the encoder's embeddings replace) and
`span_suffix`.

`load(model_dir, ...)` returns the family a checkpoint's `config.json` names (Qwen3.5 and Qwen3.8: `vision.qwen`;
Gemma 4: `vision.gemma`), or None: no vision tower, `language_model_only`, or a family not listed here.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .probe import Header

KEY_BASE = 1 << 62  # radix keys of image spans: [2^62, 2^63), outside the int32 token range (cache.radix)


@dataclass(frozen=True, slots=True)
class ImageSize:
    """One image as the model takes it: the resize target in pixels, the patch grid and the placeholder count."""

    height: int
    width: int
    grid: tuple[int, int, int]  # (t, h, w) patches before the spatial merge
    tokens: int  # placeholder tokens, one per merged patch


@dataclass(frozen=True, slots=True)
class Span:
    """An image's placeholder span in a prompt: positions [offset, offset + size.tokens)."""

    offset: int
    size: ImageSize


def span_key(digest: bytes, bits: int = 62) -> int:
    """The radix-key value of every position of an image's span: 62 bits of the image's digest, in [2^62, 2^63)
    (architecture.md, "The radix tree"). `bits`: fewer force collisions (`dev.images.span_key_bits`, tests)."""
    return KEY_BASE | (int.from_bytes(digest[:8], "little") & ((1 << max(0, min(bits, 62))) - 1))


class ImageFamily(ABC):
    name: str
    start_id: int  # tokens around each image's span (rendered by the template, or added by the expansion)
    pad_id: int  # the one placeholder token per image in the rendered prompt; expanded to `size().tokens` copies
    end_id: int
    placeholder: str  # the text of one image in a prompt (a "string"-format template's content)
    span_prefix: tuple[int, ...] = ()  # tokens the expansion puts before each span (module docstring)
    span_suffix: tuple[int, ...] = ()  # and after it
    min_pixels: int  # the smallest and largest resize targets (pixels): the pixel arena is sized from max_pixels
    max_pixels: int
    max_soft_tokens: int | None = None  # Gemma 4's image budget (images.max_soft_tokens); None: sized by pixels

    @abstractmethod
    def size(self, header: Header) -> ImageSize:
        """Header-only sizing; ValueError when the model cannot take the image (a 400)."""

    @abstractmethod
    def pixels(self, data: bytes, header: Header, size: ImageSize) -> Any:
        """The image pool's exact CPU half: decode, orient and resize to `size`, as uint8 [3, height, width]."""

    @abstractmethod
    def positions(self, n: int, spans: list[Span]) -> tuple[np.ndarray | None, int]:
        """RoPE positions of a prompt of n tokens with these image spans: (int32 [3, n], the offset of every later
        token: position = sequence index + offset), or (None, 0) when they are the sequence indices."""

    @abstractmethod
    def max_tokens(self) -> int:
        """The most placeholder tokens one image can take (at `max_pixels`)."""

    @abstractmethod
    def load_encoder(self, model_dir: str | Path, device: Any, dtype: Any) -> Any:
        """The vision encoder with the checkpoint's weights, on `device` (`runner.cuda.images.ImageEncoder`): a module
        with `cfg` (patch, merge, patch_dim, heads, head_dim, scale), `patches(uint8 [3, H, W])`, `meta(grids, device)`
        and `forward(patches, meta, attention)` -> [patches / merge², the text model's width]."""


def load(model_dir: str | Path, *, min_pixels: int | None = None, max_pixels: int | None = None,
         max_soft_tokens: int | None = None) -> ImageFamily | None:
    """The image family of a checkpoint, or None (module docstring; the sizing settings are then unused). Every site
    after `config._derive_images` passes the derived budget, images.max_soft_tokens, so all of them size images alike
    (`tests/vision/test_image_sites.py`): the frontend, the engine and the image pool with the derived pixel bounds too,
    the runner's tower without them (it sizes nothing: its graph buckets follow the budget).

    `max_soft_tokens` sizes Gemma 4's images; a family sized by pixels refuses it: a misplaced setting fails at
    startup."""
    model_dir = Path(model_dir)
    path = model_dir / "config.json"
    if not path.exists():
        return None
    hf = json.loads(path.read_text())
    if not hf.get("vision_config") or hf.get("language_model_only"):
        return None
    model_type = hf.get("model_type", "")
    if model_type in ("qwen3_5", "qwen3_5_moe"):
        from .qwen import QwenVision

        if max_soft_tokens is not None:
            raise ValueError("images.max_soft_tokens is Gemma 4's image budget; Qwen3.5/3.8 size images by "
                             "images.min_pixels and images.max_pixels")
        return QwenVision.from_model_dir(model_dir, hf, min_pixels=min_pixels, max_pixels=max_pixels)
    if model_type == "gemma4":
        from .gemma import GemmaVision

        return GemmaVision.from_model_dir(model_dir, hf, min_pixels=min_pixels, max_pixels=max_pixels,
                                          max_soft_tokens=max_soft_tokens)
    return None


# -- the pool's exact CPU path, shared by the families ----------------------------------------------------------------

_TRANSPOSED = frozenset(range(2, 9))  # the orientations exif_transpose's lookup transposes


def decode_rgb(data: bytes, header: Header) -> Any:
    """The image as vLLM hands it to HF's processors, uint8 [3, H, W] after the orientation: through Pillow, opened as
    the probed format only (`exif_transpose`, then `convert("RGB")`; EXIF that `exif_transpose` cannot read leaves the
    image as stored, as vLLM's `normalize_image` does). An image with transparency (RGBA, LA, PA, or a `transparency`
    entry: a PNG tRNS chunk, an RGB one's colour key included, a GIF's transparent index) is pasted onto white through
    its alpha instead, as vLLM's `convert_image_mode` does.
    Raises ProbeError when the decoded size disagrees with the header: before the decode (a GIF's first frame may
    extend past the logical screen the probe reads, and Pillow sizes the image by it) and after the orientation."""
    import contextlib
    import io

    import torch
    from PIL import Image, ImageOps

    from .probe import ProbeError

    im = Image.open(io.BytesIO(data), formats=(header.format.upper(),))  # the header only (and Pillow's bomb check)
    if im.size != (header.width, header.height):
        raise ProbeError(f"the decoded image is {im.size[0]}x{im.size[1]}, but its header says "
                         f"{header.width}x{header.height}")
    im.load()  # the pixels first, so that only the EXIF's errors are suppressed: a truncated image still fails
    with contextlib.suppress(Exception):  # a broken EXIF leaves the image as stored
        if im.getexif().get(0x0112, 1) in _TRANSPOSED:  # else exif_transpose only copies the image (its cached EXIF)
            im = ImageOps.exif_transpose(im)
    if im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info:
        rgba = im if im.mode == "RGBA" else im.convert("RGBA")  # pasted onto white, as vLLM does
        im = Image.new("RGB", rgba.size, (255, 255, 255))
        im.paste(rgba, mask=rgba.getchannel("A"))
    elif im.mode != "RGB":
        im = im.convert("RGB")
    img = _rgb(im, torch).permute(2, 0, 1)
    if tuple(img.shape[1:]) != (header.oriented[1], header.oriented[0]):
        raise ProbeError(f"the decoded image is {img.shape[2]}x{img.shape[1]}, but its header says "
                         f"{header.oriented[0]}x{header.oriented[1]}")
    return img


def _rgb(im: Any, torch: Any) -> Any:
    """An RGB image's pixels as uint8 [H, W, 3] over one writable buffer that Pillow's raw encoder packs into, chunk by
    chunk, as `tobytes` runs it, without `tobytes`' join of the chunks and a copy for torch: at 12 megapixels each is
    36 MB more to copy and, on linux, to fault in. `_getencoder` is Pillow's private name for it, at the stack's pinned
    Pillow (the JPEG digests and HF's pixel tests read its bytes)."""
    from PIL import Image

    enc = Image._getencoder("RGB", "raw", "RGB")
    enc.setimage(im.im, (0, 0, *im.size))
    buf = bytearray(3 * im.width * im.height)
    view, pos, step = memoryview(buf), 0, max(1 << 16, 4 * im.width)  # tobytes' chunk size
    while True:
        _, status, chunk = enc.encode(step)
        view[pos:pos + len(chunk)] = chunk
        pos += len(chunk)
        if status:
            break
    if status < 0 or pos != len(buf):
        raise RuntimeError(f"Pillow's raw encoder: status {status}, {pos} of {len(buf)} bytes")
    return torch.frombuffer(buf, dtype=torch.uint8).view(im.height, im.width, 3)


def resize_bicubic(img: Any, height: int, width: int) -> Any:
    """The processors' torchvision backend's antialiased bicubic resize on uint8 (exact with HF's pixels), on torch
    alone, call for call as torchvision 0.29's `resize` runs it on a CPU uint8 tensor: the image itself at an unchanged
    size (`interpolate` there is not promised to be the identity); else, for a channels-last input (a decoded image's
    permuted view), the batch stride a batch of one needs to read as channels-last; then `interpolate`."""
    import torch

    if (height, width) == tuple(img.shape[1:]):
        return img.contiguous()
    x = img.unsqueeze(0)
    if x.is_contiguous(memory_format=torch.channels_last) and x.stride(0) != x.numel():
        x = x.as_strided(x.shape, (x.numel(), *x.stride()[1:]))
    out = torch.nn.functional.interpolate(x, size=[height, width], mode="bicubic", align_corners=False, antialias=True)
    return out[0].contiguous()
