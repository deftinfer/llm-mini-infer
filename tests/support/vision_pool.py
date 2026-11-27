"""The image pool's tests: images, a pool worker and its jobs (tests/vision/test_vision.py)."""

from __future__ import annotations

import asyncio
import io
import multiprocessing as mp
import struct
from pathlib import Path

from PIL import Image

from llm_mini_infer.vision.family import ImageSize
from llm_mini_infer.vision.gemma import GemmaVision
from llm_mini_infer.vision.pool import Done
from llm_mini_infer.vision.probe import Header

GEMMA70 = GemmaVision(boi_id=255999, image_id=258880, eoi_id=258882, max_soft_tokens=70)  # Gemma 4 at 70 soft tokens

# A poison image: SOI, a one-component SOF0 declaring 65,535 x 65,535 pixels (12.9 GB of RGB), EOI: 17 bytes
# without a scan, which libjpeg refuses at once, so it is safe without the guard too. Never put a JPEG with real tables
# and sides up to 65,500 over the limit in a test: without the guard, its decode allocates 12.9 GB on the test host.
POISON = b"\xff\xd8\xff\xc0" + struct.pack(">HBHHB", 11, 8, 65535, 65535, 1) + b"\x01\x11\x00\xff\xd9"


def exif(o: int, big: bool = False, typ: int = 3) -> bytes:
    e = ">" if big else "<"
    val = struct.pack(e + "HH", o, 0) if typ == 3 else struct.pack(e + "I", o)
    tiff = (b"MM\x00*" if big else b"II*\x00") + struct.pack(e + "I", 8) + struct.pack(e + "H", 1) + \
        struct.pack(e + "HHI", 0x0112, typ, 1) + val + struct.pack(e + "I", 0)
    return b"Exif\x00\x00" + tiff


def encoded(img: Image.Image, fmt: str, **kw) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt, **kw)
    return buf.getvalue()


def spawn_pool_worker(model_dir: Path, min_pixels: int | None, max_pixels: int | None, ipc: str, arena: str,
                      lopsided: bool = False, **kw) -> tuple[mp.Process, bool]:
    """A real pool worker 0, spawned as the launcher spawns it (`kw`: worker_main's parameters after `log_level`, by
    keyword): (the process, whether it reported ready; one that did not has exited, its exit code set). `lopsided`: its
    family is `lopsided`'s, a defective one whose sizing is not symmetric under a transpose."""
    from llm_mini_infer.vision.pool import worker_main

    ctx = mp.get_context("spawn")
    r, s = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=lopsided_worker_main if lopsided else worker_main,
                       args=(str(model_dir), min_pixels, max_pixels, 0, ipc, arena, s, "warning"), kwargs=kw)
    proc.start()
    s.close()
    try:
        ready = r.poll(120) and r.recv() == "ready"
    except EOFError:  # it exited first
        ready = False
    r.close()
    if not ready:
        proc.join(30)
    return proc, ready


def run_jobs(ipc: str, jobs: list[tuple[bytes, Header, ImageSize, int]]) -> list[Done]:
    """Each (image, its header, its resize target, the pixels' arena offset) as one pool job to worker 0, in order;
    the replies."""
    from llm_mini_infer.vision.pool import Job, PoolClient

    async def main() -> list[Done]:
        client = PoolClient(ipc, 1)
        client.start()
        try:
            out = []
            for data, h, size, offset in jobs:
                job = Job(jid=0, fmt=h.format, width=h.width, height=h.height,
                          out_h=size.height, out_w=size.width, grid_t=size.grid[0],
                          grid_h=size.grid[1], grid_w=size.grid[2], tokens=size.tokens, offset=offset)
                out.append(await asyncio.wait_for(client.submit(job, data), 60))
            return out
        finally:
            client.close()

    return asyncio.run(main())


def lopsided(family: GemmaVision) -> GemmaVision:
    """A copy of `family` that is deliberately not symmetric under a transpose: a portrait image takes one token fewer
    than the landscape one it is turned from."""
    import copy
    import dataclasses

    class Lopsided(type(family)):
        def size(self, header: Header) -> ImageSize:
            size = super().size(header)
            return dataclasses.replace(size, tokens=size.tokens - 1) if header.height > header.width else size

    out = copy.copy(family)
    out.__class__ = Lopsided
    return out


def lopsided_worker_main(*args, **kw) -> None:
    """`worker_main` with the family it loads made `lopsided` (a spawned worker's entry)."""
    from llm_mini_infer.vision import family as fam
    from llm_mini_infer.vision.pool import worker_main

    load = fam.load
    fam.load = lambda *a, **k: lopsided(load(*a, **k))
    worker_main(*args, **kw)
