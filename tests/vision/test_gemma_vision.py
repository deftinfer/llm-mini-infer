"""Gemma 4's images against transformers on CPU: the resize target and soft-token count of HF's
`Gemma4ImageProcessor` exactly, its `pixel_values` bit for bit (the pool's uint8 path, then the tower's rescale and
patchify) at the checkpoint's budget and at images.max_soft_tokens 70, the prompt expansion of `Gemma4Processor`, and
the vision tower with its embedder on a tiny random configuration, layer by layer, to fp32 rounding."""

from __future__ import annotations

import io
import json
import random
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from llm_mini_infer.vision.family import transposed
from llm_mini_infer.vision.gemma import SOFT_TOKENS, GemmaVision, aspect_ratio_size
from llm_mini_infer.vision.probe import probe

from ..support.paths import DATA, MODELS_DIR

pytest.importorskip("transformers")
from transformers.models.gemma4.image_processing_gemma4 import (
    Gemma4ImageProcessor,
    get_aspect_ratio_preserving_size,
)

FAM = GemmaVision(boi_id=255999, image_id=258880, eoi_id=258882)
FAMS = {k: GemmaVision(boi_id=255999, image_id=258880, eoi_id=258882, max_soft_tokens=k) for k in (280, 70)}


def hf_size(h: int, w: int, k: int) -> tuple[int, int] | None:
    try:
        return get_aspect_ratio_preserving_size(height=h, width=w, patch_size=16, max_patches=k * 9,
                                                pooling_kernel_size=3)
    except ValueError:
        return None


def our_size(h: int, w: int, k: int) -> tuple[int, int] | None:
    try:
        return aspect_ratio_size(h, w, 16, k * 9, 3)
    except ValueError:
        return None


def test_the_resize_target_is_hfs_exactly() -> None:
    """Every soft-token budget; a random sweep of sizes from 1 pixel to 8k, strips of every ratio, squares, the sides
    where a floor flips (multiples of 48 and their neighbours) and photos' usual sizes: the same target or the same
    rejection."""
    rng = random.Random(5)
    sizes = [(rng.randint(1, 8000), rng.randint(1, 8000)) for _ in range(4000)]
    sizes += [(1, n) for n in range(1, 3000, 7)] + [(n, 1) for n in range(1, 3000, 11)]
    sizes += [(n, n) for n in range(1, 2000, 3)] + [(48 * a + d, 48 * b + e) for a in range(1, 30, 3)
                                                     for b in range(1, 30, 4) for d in (-1, 0, 1) for e in (-1, 0, 1)]
    sizes += [(225, 400), (400, 225), (720, 1280), (1080, 1920), (3024, 4032), (4032, 3024), (333, 1000), (2, 3)]
    for k in SOFT_TOKENS:
        for h, w in sizes:
            assert our_size(h, w, k) == hf_size(h, w, k), (h, w, k)


def test_the_token_count_is_the_processors() -> None:
    """The family's sizing from a header (the stored size, as the frontend sizes it) gives the count `Gemma4Processor`
    expands to."""
    from PIL import Image

    proc = Gemma4ImageProcessor()
    for w, h in ((400, 225), (225, 400), (1000, 333), (37, 2900), (5, 5), (1920, 1080)):
        buf = io.BytesIO()
        Image.new("RGB", (w, h), (9, 99, 199)).save(buf, format="PNG")
        size = FAM.size(probe(buf.getvalue()))
        out = proc(images=[Image.open(io.BytesIO(buf.getvalue()))], return_tensors="pt")
        assert size.tokens == int(out["num_soft_tokens_per_image"][0])
        th, tw = hf_size(h, w, 280)
        assert (size.height, size.width) == (th, tw) and size.grid == (1, th // 16, tw // 16)


def _images() -> list[tuple[str, bytes]]:
    """Synthetic edge cases: modes, formats, EXIF orientations, sizes that need no resize, up- and downscaling."""
    from PIL import Image

    rng = np.random.default_rng(3)
    out = []

    def save(kind: str, im: Image.Image, fmt: str, **kw) -> None:
        buf = io.BytesIO()
        im.save(buf, format=fmt, **kw)
        out.append((kind, buf.getvalue()))

    def noise(w: int, h: int, mode: str = "RGB") -> Image.Image:
        c = {"RGB": 3, "RGBA": 4, "L": 1, "LA": 2}[mode]
        a = rng.integers(0, 256, (h, w, c), dtype=np.uint8)
        return Image.fromarray(a[..., 0] if c == 1 else a, mode)

    save("jpeg", noise(400, 225), "JPEG", quality=90)  # a landscape and a portrait size: 400 x 225 and 400 x 711
    save("jpeg_portrait", noise(400, 711), "JPEG", quality=90)
    save("jpeg_up", noise(64, 40), "JPEG", quality=95)
    save("jpeg_big", noise(1600, 900), "JPEG", quality=85)
    save("png_exact", noise(1056, 576), "PNG")  # the target of a 16:9 photo: no resize
    save("png_rgba", noise(300, 500, "RGBA"), "PNG")
    save("png_l", noise(500, 300, "L"), "PNG")
    save("png_la", noise(200, 120, "LA"), "PNG")
    save("png_palette", noise(320, 240).convert("P"), "PNG")
    save("webp", noise(640, 480), "WEBP", quality=80)
    save("gif", noise(150, 150).convert("P"), "GIF")
    save("bmp", noise(123, 77), "BMP")
    save("strip", noise(2000, 3), "PNG")
    save("tiny", noise(3, 3), "PNG")
    for o in (3, 6, 8):  # EXIF orientations: rotated 180, 90 cw, 90 ccw
        im = noise(320, 200)
        ex = im.getexif()
        ex[0x0112] = o
        save(f"jpeg_exif{o}", im, "JPEG", quality=90, exif=ex.tobytes())
    return out


def _hf_pixels(data: bytes, k: int = 280) -> tuple[np.ndarray, np.ndarray]:
    """HF's processor at `k` soft tokens on the image as vLLM hands it over: transformers'
    `load_image` (EXIF orientation, RGB), this file's RGBA and LA images pasted onto white through their alpha first
    (tests/vision/test_vision.py pins that rule): (pixel_values of the real patches, their (x, y))."""
    from PIL import Image, ImageOps
    from transformers.image_utils import load_image

    im = ImageOps.exif_transpose(pil_open(data))
    if im.mode in ("RGBA", "LA"):
        rgba = im.convert("RGBA")
        im = Image.new("RGB", rgba.size, (255, 255, 255))
        im.paste(rgba, mask=rgba.getchannel("A"))
    proc = Gemma4ImageProcessor(max_soft_tokens=k)
    out = proc(images=[load_image(im)], return_tensors="pt")
    pos = out["image_position_ids"][0]
    real = (pos[:, 0] >= 0).numpy()
    return out["pixel_values"][0].numpy()[real], pos.numpy()[real]


def pil_open(data: bytes):
    from PIL import Image

    return Image.open(io.BytesIO(data))


def _ours(data: bytes, fam: GemmaVision = FAM) -> tuple[np.ndarray, np.ndarray]:
    """The pool's path (`GemmaVision.pixels`), then the tower's rescale and patchify (pooling order), reordered to HF's
    row-major patches; with each patch's (x, y)."""
    from llm_mini_infer.models.gemma4.vision import GemmaVisionConfig, GemmaVisionEncoder, pool_order

    cfg = GemmaVisionConfig(depth=0, hidden=8, heads=2, head_dim=4, intermediate=8, patch=16, pooling=3, pos_size=16,
                            rope_theta=100.0, eps=1e-6, standardize=False, out_hidden=8, rescale=1 / 255,
                            normalize=False, mean=(0.0,) * 3, std=(1.0,) * 3)
    with torch.device("meta"):
        enc = GemmaVisionEncoder(cfg, torch.float32)
    header = probe(data)
    size = fam.size(header)
    px = fam.pixels(data, header, size)
    if tuple(px.shape[1:]) != (size.height, size.width):  # Pillow turned a non-square image: the decode decides
        size = transposed(size)
    assert px.dtype == torch.uint8 and tuple(px.shape) == (3, size.height, size.width)
    patches = GemmaVisionEncoder.patches(enc, px).numpy()
    xs, ys = pool_order(size.grid[1], size.grid[2], 3, torch.device("cpu"))
    order = np.argsort((ys * size.grid[2] + xs).numpy())  # pooling order -> row-major
    return patches[order], np.stack([xs.numpy(), ys.numpy()], 1)[order]


@pytest.mark.parametrize("k", sorted(FAMS))
@pytest.mark.parametrize("kind,data", _images(), ids=lambda x: x if isinstance(x, str) else "")
def test_pixels_are_hfs_bit_for_bit(kind: str, data: bytes, k: int) -> None:
    """At the checkpoint's budget (280) and at 70, the processor and the family at the same one."""
    ours, pos = _ours(data, FAMS[k])
    ref, ref_pos = _hf_pixels(data, k)
    assert ours.shape == ref.shape, kind
    assert np.array_equal(pos, ref_pos), kind
    assert np.array_equal(ours, ref), f"{kind}: {int((ours != ref).sum())} values differ"


def _photos() -> list[bytes]:
    """Photo-like JPEGs, as a camera's are: smooth gradients, blurred shapes and a fine grain, at cameras' sizes and
    aspects (landscape and portrait), baseline and progressive, some with an EXIF orientation."""
    from PIL import Image, ImageDraw, ImageFilter

    rng = np.random.default_rng(7)
    sizes = [(4032, 3024), (3024, 4032), (1920, 1080), (1080, 1920), (1280, 960), (800, 600), (640, 480),
             (2048, 1536), (1600, 1200), (720, 1280), (1000, 1000), (3000, 2000)]
    out = []
    for i, (w, h) in enumerate(sizes):
        im = Image.fromarray(rng.integers(0, 256, (6, 8, 3), dtype=np.uint8)).resize((w, h), Image.BICUBIC)
        draw = ImageDraw.Draw(im)
        for _ in range(12):
            x, y, r = int(rng.integers(0, w)), int(rng.integers(0, h)), int(rng.integers(w // 30, w // 6))
            draw.ellipse((x - r, y - r, x + r, y + r), fill=tuple(int(c) for c in rng.integers(0, 256, 3)))
        grain = rng.integers(-6, 7, (h, w, 3))
        im = Image.fromarray(np.clip(np.asarray(im.filter(ImageFilter.GaussianBlur(2)), np.int16) + grain, 0, 255)
                             .astype(np.uint8))
        kw = {"quality": int(rng.integers(80, 96)), "progressive": i % 3 == 0}
        if i % 4 == 1:
            exif = im.getexif()
            exif[0x0112] = (6, 3, 8)[i % 3]
            kw["exif"] = exif.tobytes()
        buf = io.BytesIO()
        im.save(buf, format="JPEG", **kw)
        out.append(buf.getvalue())
    return out


def test_photo_like_images_are_hfs_bit_for_bit() -> None:
    """Photo-like JPEGs (`_photos`) at the checkpoint's budget: the pool's pixels are the processor's."""
    for i, data in enumerate(_photos()):
        ours, _ = _ours(data)
        ref, _ = _hf_pixels(data)
        assert ours.shape == ref.shape and np.array_equal(ours, ref), f"photo {i}"


def test_the_family_reads_the_checkpoints_processor_config(tmp_path) -> None:
    from llm_mini_infer.vision import family as vf

    hf = {"model_type": "gemma4", "boi_token_id": 255999, "image_token_id": 258880, "eoi_token_id": 258882,
          "vision_config": {"patch_size": 16, "pooling_kernel_size": 3, "default_output_length": 280},
          "text_config": {"use_bidirectional_attention": "vision"}}
    (tmp_path / "config.json").write_text(json.dumps(hf))
    (tmp_path / "processor_config.json").write_text(json.dumps({"image_processor": {
        "do_normalize": False, "max_soft_tokens": 560, "patch_size": 16, "pooling_kernel_size": 3, "resample": 3,
        "rescale_factor": 0.00392156862745098}}))
    fam = vf.load(tmp_path)
    assert isinstance(fam, GemmaVision) and fam.max_tokens() == 560 and fam.max_pixels == 560 * 9 * 256
    assert fam.span_prefix == (255999,) and fam.span_suffix == (258882,) and fam.pad_id == 258880
    assert vf.load(tmp_path, min_pixels=fam.min_pixels, max_pixels=fam.max_pixels).max_tokens() == 560
    with pytest.raises(ValueError, match="max_soft_tokens"):
        vf.load(tmp_path, max_pixels=327680)


# -- the budget: images.max_soft_tokens --------------------------------------------------------------------------------


def _checkpoint(tmp: Path) -> Path:
    """google/gemma-4-31B-it's config.json and processor_config.json (tests/data): images of at most 280 soft tokens."""
    tmp.mkdir(parents=True, exist_ok=True)
    shutil.copy(DATA / "gemma-4-31b-it.config.json", tmp / "config.json")
    shutil.copy(DATA / "gemma-4-31b-it.processor_config.json", tmp / "processor_config.json")
    return tmp


def test_the_image_token_ids_are_the_tokenizers(tmp_path) -> None:
    """config.json's boi, image and eoi ids against the tokenizer's for tokenizer_config.json's tokens, as the frontend
    checks them at startup: google/gemma-4-31B-it's agree (its tracked configs, the fixture's tokenizer); a shifted id
    fails naming the token and both ids, and so does a token the tokenizer lacks or tokenizer_config.json leaves out."""
    from tokenizers import Tokenizer

    from llm_mini_infer.vision import family as vf

    tok = MODELS_DIR / "gemma-4-31b-it" / "tokenizer.json"
    if not tok.exists():
        pytest.skip("tests/.fixtures/models/gemma-4-31b-it is absent: uv run tests/scripts/fetch_fixtures.py")
    tokenizer = Tokenizer.from_file(str(tok))
    tc = json.loads((DATA / "gemma-4-31b-it.tokenizer_config.json").read_text())
    path = _checkpoint(tmp_path)
    vf.load(path).check_tokenizer(tokenizer, tc)
    FAM.check_tokenizer(tokenizer, {**tc, "image_token": {"content": "<|image|>", "special": True}})  # an AddedToken
    hf = json.loads((path / "config.json").read_text())
    (path / "config.json").write_text(json.dumps({**hf, "image_token_id": 258881}))
    with pytest.raises(ValueError, match=r"image_token_id is 258881 in config.json, but the tokenizer's <\|image\|> "
                                         r"is 258880: images would not be expanded as HF's processor expands them"):
        vf.load(path).check_tokenizer(tokenizer, tc)
    with pytest.raises(ValueError, match=r"boi_token_id is 255999 in config.json, but the tokenizer has no <\|img>"):
        FAM.check_tokenizer(tokenizer, {**tc, "boi_token": "<|img>"})
    with pytest.raises(ValueError, match=r"eoi_token_id is 258882 in config\.json, but tokenizer_config\.json names "
                                         r"no eoi_token"):
        FAM.check_tokenizer(tokenizer, {k: v for k, v in tc.items() if k != "eoi_token"})


def _jpeg(w: int, h: int, seed: int = 0) -> bytes:
    from PIL import Image

    a = np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, "JPEG", quality=90)
    return buf.getvalue()


def test_the_budget_is_the_setting_else_the_checkpoints(tmp_path) -> None:
    """images.max_soft_tokens, else processor_config.json (as HF's processor reads it), else vision_config's
    default_output_length (vLLM's budget), else transformers' 280; the family records where it came from (the
    derived setting's reason). Pixel bounds of another budget fail and name the setting: the tripwire of a load site
    that forgets to pass it."""
    from llm_mini_infer.vision import family as vf

    path = _checkpoint(tmp_path)

    def budget(**kw) -> tuple[int, str]:
        fam = vf.load(path, **kw)
        return fam.max_soft_tokens, fam.max_soft_tokens_source

    assert budget(max_soft_tokens=70) == (70, "images.max_soft_tokens")
    assert budget() == (280, "processor_config.json image_processor.max_soft_tokens")
    with pytest.raises(ValueError, match=r"images\.max_soft_tokens \(280, .*its max_pixels is 645120, not 161280"):
        vf.load(path, min_pixels=2304, max_pixels=161280)
    assert vf.load(path, min_pixels=2304, max_pixels=161280, max_soft_tokens=70).max_tokens() == 70
    (path / "processor_config.json").unlink()
    hf = json.loads((path / "config.json").read_text())
    hf["vision_config"]["default_output_length"] = 140
    (path / "config.json").write_text(json.dumps(hf))
    assert budget() == (140, "config.json vision_config.default_output_length")
    del hf["vision_config"]["default_output_length"]
    (path / "config.json").write_text(json.dumps(hf))
    assert budget() == (280, "280 (transformers' default)")


def test_400_by_225_and_400_by_711_take_66_soft_tokens_at_70(tmp_path) -> None:
    """At images.max_soft_tokens 70, a 400 x 225 and a 400 x 711 image take 66 soft
    tokens (288 x 528 pixels, 594 patches), against 264 at the checkpoint's 280. HF's processor at 70 agrees: 66 soft
    tokens, 594 real patches padded to 630."""
    from llm_mini_infer.vision import family as vf
    from llm_mini_infer.vision.family import ImageSize

    path = _checkpoint(tmp_path)
    at70, at280 = vf.load(path, max_soft_tokens=70), vf.load(path)
    proc = Gemma4ImageProcessor(max_soft_tokens=70)
    for (w, h), want in (((400, 225), ImageSize(288, 528, (1, 18, 33), 66)),
                         ((400, 711), ImageSize(528, 288, (1, 33, 18), 66))):
        data = _jpeg(w, h)
        assert at70.size(probe(data)) == want
        full = at280.size(probe(data))
        assert full.tokens == 264 and {full.height, full.width} == {576, 1056}
        out = proc(images=[pil_open(data)], return_tensors="pt")
        pos = out["image_position_ids"][0]
        assert int(out["num_soft_tokens_per_image"][0]) == 66
        assert pos.shape[0] == 630 and int((pos[:, 0] >= 0).sum()) == 594


def test_the_soft_token_sets_agree() -> None:
    """images.max_soft_tokens takes exactly the family's budgets, which are transformers'."""
    import typing

    from transformers.models.gemma4 import image_processing_gemma4

    from llm_mini_infer.config import ImageConfig

    literal = next(a for a in typing.get_args(typing.get_type_hints(ImageConfig)["max_soft_tokens"])
                   if a is not type(None))
    assert typing.get_args(literal) == SOFT_TOKENS == image_processing_gemma4._SUPPORTED_SOFT_TOKENS


def test_pool_worker_takes_the_soft_token_budget(tmp_path, capfd) -> None:
    """An image pool worker loads the family at images.max_soft_tokens (the launcher passes it by keyword): at 70 it
    reports ready and writes a 400 x 225 image's pixels at 288 x 528, HF's. Spawned without it, the worker reloads the
    checkpoint's 280, and the family's check stops it before it reports ready, naming the setting (a load site that
    forgets the budget fails at startup, never sizes otherwise; the child's traceback goes to the inherited stderr)."""
    from llm_mini_infer.engine import ipc as engine_ipc
    from llm_mini_infer.vision import arena as px_arena

    from ..support.vision_pool import run_jobs, spawn_pool_worker

    path = _checkpoint(tmp_path / "g")
    fam = FAMS[70]
    ipc = engine_ipc.ipc_dir()
    arena = str(tmp_path / "arena")
    px_arena.create(arena, 1 << 22)
    procs = []
    try:
        proc, ready = spawn_pool_worker(path, fam.min_pixels, fam.max_pixels, ipc, arena, max_soft_tokens=70)
        procs.append(proc)
        assert ready
        data = _jpeg(400, 225, 1)
        header = probe(data)
        size = fam.size(header)
        (done,) = run_jobs(ipc, [(data, header, size, 8192)])
        assert not done.error and done.decode_ns > 0 and (size.height, size.width) == (288, 528)
        view = px_arena.map_readonly(arena)
        got = np.frombuffer(view.buf, np.uint8, 3 * 288 * 528, 8192).reshape(3, 288, 528).copy()
        view.close()
        assert np.array_equal(got, fam.pixels(data, header, size).numpy())
        proc.kill()
        proc.join(10)
        capfd.readouterr()
        proc, ready = spawn_pool_worker(path, fam.min_pixels, fam.max_pixels, ipc, arena)
        procs.append(proc)
        assert not ready and proc.exitcode not in (0, None)
        err = capfd.readouterr().err
        assert ("ValueError: images.max_pixels: Gemma 4 sizes images by images.max_soft_tokens (280, processor_config"
                ".json image_processor.max_soft_tokens); its max_pixels is 645120, not 161280") in err, err
    finally:
        for p in procs:
            p.kill()  # as the launcher stops the pool (a worker ignores SIGTERM while its parent lives)
            p.join(10)
        px_arena.remove(arena)
        shutil.rmtree(ipc, ignore_errors=True)


# -- the tower against transformers (tiny random configuration, CPU, fp32) ---------------------------------------------


@pytest.fixture(scope="module")
def tiny_mm(tmp_path_factory):
    from ..support.tiny_gemma import make_tiny_gemma_mm

    return make_tiny_gemma_mm(tmp_path_factory.mktemp("g4mm"), depth=3)


def _tiny_family(model_dir: Path) -> GemmaVision:
    from llm_mini_infer.vision import family as vf

    fam = vf.load(model_dir)
    assert isinstance(fam, GemmaVision)
    return fam


def _photo(w: int, h: int, seed: int) -> bytes:
    from PIL import Image

    a = np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)
    a[: h // 2] //= 2  # some structure: the halves differ
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="PNG")
    return buf.getvalue()


def test_the_tower_matches_transformers_layer_by_layer(tiny_mm) -> None:
    """Three images of other shapes in one encoder call (a ragged batch): the patch embedding, every layer's residual
    stream and the soft tokens after the pooler, the standardisation and the embedder equal transformers'
    (`get_image_features`, each image padded to the processor's patch budget as HF runs it) to fp32 rounding."""
    from llm_mini_infer.models.gemma4.vision import load_gemma_vision, pool_order
    from llm_mini_infer.models.qwen3_5.vision import TorchVisionAttention

    from ..support.tiny_gemma import hf_image_processor, hf_mm_model

    fam = _tiny_family(tiny_mm)
    enc = load_gemma_vision(tiny_mm, torch.device("cpu"), torch.float32)
    hf = hf_mm_model(tiny_mm)
    blobs = [_photo(96, 54, 1), _photo(30, 70, 2), _photo(201, 13, 3)]
    sizes = [fam.size(probe(b)) for b in blobs]
    assert len({s.tokens for s in sizes}) > 1
    pxs = [fam.pixels(b, probe(b), s) for b, s in zip(blobs, sizes)]
    patches = torch.cat([enc.patches(px) for px in pxs])
    meta = enc.meta([s.grid for s in sizes], torch.device("cpu"))
    ours_layers: list[torch.Tensor] = []
    from llm_mini_infer.kernels import gemma_ops

    orig = gemma_ops.sandwich
    count = [0]

    def spy(*a, **kw):
        r, x = orig(*a, **kw)
        count[0] += 1
        if count[0] % 2 == 0:  # after each layer's MLP block
            ours_layers.append(r.clone())
        return r, x

    gemma_ops.sandwich = spy
    try:
        with torch.no_grad():
            ours = enc(patches, meta, TorchVisionAttention(scale=1.0))
    finally:
        gemma_ops.sandwich = orig
    from PIL import Image

    ims = [Image.open(io.BytesIO(b)) for b in blobs]
    ref_in = hf_image_processor(tiny_mm)(images=ims, return_tensors="pt")
    with torch.no_grad():
        feats = hf.model.get_image_features(ref_in["pixel_values"], ref_in["image_position_ids"], return_dict=True)
        vt = hf.model.vision_tower(pixel_values=ref_in["pixel_values"], pixel_position_ids=ref_in["image_position_ids"],
                                   output_hidden_states=True)
    ref = torch.cat(feats.pooler_output)
    assert ours.shape == ref.shape
    assert (ours - ref).abs().max().item() < 2e-5 * max(1.0, ref.abs().max().item())
    # layer by layer: HF's hidden states (padded, row-major) against ours (pooling order), real patches only
    off = 0
    for i, s in enumerate(sizes):
        n = s.grid[1] * s.grid[2]
        xs, ys = pool_order(s.grid[1], s.grid[2], 3, torch.device("cpu"))
        rowmajor = (ys * s.grid[2] + xs).long()
        # HF records the embeddings, each layer's output but the last's, then the pooled output in its place
        assert len(ours_layers) == 3 and len(vt.hidden_states) == 4
        embed = enc.embed(patches, meta)
        for layer, (o, r) in enumerate(zip([embed, *ours_layers[:2]], vt.hidden_states[:3])):
            got, want = o[off:off + n], r[i][:n][rowmajor]
            assert (got - want).abs().max().item() < 2e-5 * max(1.0, want.abs().max().item()), (i, layer)
        off += n


def test_the_patch_offsets_are_the_grids_patch_counts(tiny_mm, monkeypatch) -> None:
    """The encoder call's patch offsets: each image's patches after those of the images before it, int32 on the
    tower's device, and on a CPU never pinned (that needs CUDA; tests/vision/test_vision_gpu.py checks the copy on
    one)."""
    from llm_mini_infer.models.gemma4.vision import load_gemma_vision

    enc = load_gemma_vision(tiny_mm, torch.device("cpu"), torch.float32)

    def pin_memory(self, *args, **kwargs):
        raise AssertionError("pinned on a CPU")

    monkeypatch.setattr(torch.Tensor, "pin_memory", pin_memory)
    meta = enc.meta([(1, 6, 9), (1, 3, 3), (1, 12, 3)], torch.device("cpu"))
    assert meta.cu_host == [0, 54, 63, 99] and meta.n == 99
    assert meta.cu.dtype == torch.int32 and meta.cu.device.type == "cpu" and meta.cu.tolist() == meta.cu_host


def test_a_grid_past_the_position_table_is_refused_by_name(tiny_mm) -> None:
    """The tiny tower's position table has 256 entries a side: a grid with a longer side (the memory profile's old
    synthetic images of 89 and 254 soft tokens, 3 x 267 and 6 x 381 patches) is refused by name before the tower runs,
    where it indexed past the table (an IndexError on a CPU, a device-side assert on CUDA). The profile's images of
    those counts (`ImageEncoder.synthetic`) now fit, a few tokens fewer, and run."""
    from llm_mini_infer.runner.cuda.images import ImageEncoder, VisionOptions

    cpu = torch.device("cpu")
    enc = ImageEncoder(tiny_mm, VisionOptions(encoder_rows=512, encoder_budget=256, max_soft_tokens=70), cpu,
                       torch.float32)
    enc.allocate(enc.cfg.out_hidden, flashinfer=False)
    v = enc.vision
    assert v.cfg.pos_size == v.cfg.max_grid == 256
    assert v.meta([(1, 255, 3), (1, 3, 255)], cpu).n == 2 * 765
    for _, h, w in ((1, 3, 267), (1, 6, 381), (1, 258, 3)):
        with pytest.raises(ValueError, match=rf"a patch grid of {h} x {w} is past the position table \(256 entries\)"):
            enc._tower(torch.zeros(h * w, v.cfg.patch_dim), [(1, 3, 3), (1, h, w)], graphs=False)
    for t, grid, tokens in ((89, (1, 27, 27), 81), (254, (1, 45, 48), 240), (70, (1, 21, 30), 70)):
        g, hw, n = enc.synthetic(t)
        assert (g, n) == (grid, tokens) and hw == (g[1] * 4, g[2] * 4)
        px = torch.from_numpy(enc._px_internal[:3 * hw[0] * hw[1]].copy()).view(3, *hw)
        with torch.inference_mode():
            out = enc._tower(v.patches(px), [g], graphs=False)
        assert out.shape == (n, enc.cfg.out_hidden) and bool(torch.isfinite(out).all())


def test_every_image_fits_the_31bs_position_table() -> None:
    """At each of the five soft-token budgets the most elongated image `aspect_ratio_size` gives (a strip one soft
    token high, pooling x max_soft_tokens patches long) stays within the position table of google/gemma-4-31B-it
    (config.json's `position_embedding_size`): the tower's check never refuses a request's image."""
    vc = json.loads((DATA / "gemma-4-31b-it.config.json").read_text())["vision_config"]
    table, patch, k = vc["position_embedding_size"], vc["patch_size"], vc["pooling_kernel_size"]
    assert (table, patch, k) == (10240, 16, 3)
    shapes = [(1, 10**6), (10**6, 1), (9, 16), (1080, 1920), (3, 100_000), (100_000, 7)]
    shapes += [(1, w) for w in range(1, 5000, 7)] + [(h, 1) for h in range(1, 5000, 11)]
    for n in SOFT_TOKENS:
        longest = max(max(aspect_ratio_size(h, w, patch, n * k * k, k)) // patch for h, w in shapes)
        assert longest == k * n <= table, n


@pytest.mark.parametrize("name", ["model.vision_tower.patch_embedder.input_proj.weight_scale",
                                  "model.embed_vision.embedding_post_projection_norm.weight",
                                  "vision_tower.std_bias"])
def test_an_unknown_vision_weight_is_refused(name) -> None:
    """A tower or embedder tensor we have no parameter for is refused, as one inside a layer is (it was skipped, and
    the tower ran without it); names outside them are the text loader's."""
    from llm_mini_infer.models.gemma4.vision import vision_weight_name

    with pytest.raises(ValueError, match=f"^unexpected vision weight {name}$"):
        vision_weight_name(name, 1152, 4304)
    assert vision_weight_name("model.language_model.layers.0.mlp.down_proj.weight", 1152, 4304) is None
    assert vision_weight_name("model.audio_tower.layers.0.weight", 1152, 4304) is None


def test_the_tower_reads_the_real_checkpoints_names() -> None:
    """Every vision weight name of google/gemma-4-31B-it (its index) maps to a parameter, and every parameter is
    covered (the shapes come with the real load on the GPU)."""
    from llm_mini_infer.models.gemma4.vision import GemmaVisionConfig, GemmaVisionEncoder, vision_weight_name

    index = MODELS_DIR / "gemma-4-31b-it" / "model.safetensors.index.json"
    if not index.exists():
        pytest.skip("no local copy of the 31B's index")
    cfg = GemmaVisionConfig.from_dir(index.parent)
    assert (cfg.depth, cfg.hidden, cfg.heads, cfg.head_dim, cfg.intermediate, cfg.out_hidden) == \
        (27, 1152, 16, 72, 4304, 5376)
    with torch.device("meta"):
        enc = GemmaVisionEncoder(cfg, torch.bfloat16)
    params = dict(enc.named_parameters())
    hit: set[str] = set()
    for name in json.loads(index.read_text())["weight_map"]:
        t = vision_weight_name(name, cfg.hidden, cfg.intermediate)
        if t is not None:
            assert t[0] in params, name
            hit.add(t[0])
    assert hit == set(params)


@pytest.mark.parametrize(("n", "h", "d", "fused"), [(594, 16, 72, True), (1, 16, 72, True), (1, 8, 64, True),
                                                    (2, 7, 72, False), (1, 8, 72, False)])
def test_the_fused_glue_runs_where_its_sum_is_torchs(monkeypatch, n, h, d, fused) -> None:
    """The tower's fused glue (`vision_qkv`) runs only where its sum of squares takes torch's order, so its output is
    the torch ops' bit for bit: torch's mean (2.13 to 2.14.1) over a last dimension below 128 reduces each output in
    one warp from 16 outputs on; with fewer and a head dim past 64, in a 64-lane block that adds element l + 64 before
    l + 32, and the torch ops run (never in Gemma 4's tower: 16 heads). The kernel library is a stand-in here."""
    from llm_mini_infer.models.gemma4 import vision as gv

    calls = []
    lib = SimpleNamespace(lmk_g4_vision_qkv=lambda *a: calls.append(a) or 0)
    monkeypatch.setattr(gv, "use_kernels", lambda t: True)
    monkeypatch.setattr(gv.build, "load", lambda: lib)
    monkeypatch.setattr(gv, "_stream", lambda t: None)
    qkv = torch.zeros(n, 3, h, d, dtype=torch.bfloat16)
    w = torch.ones(d, dtype=torch.bfloat16)
    cs = torch.zeros(n, d // 2)
    out = gv.vision_qkv(qkv, w, w, 1e-6, cs, cs, 128)
    assert bool(calls) == fused and all(t.shape == (n, h, 128) for t in out)
