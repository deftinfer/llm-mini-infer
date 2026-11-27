"""The image path's CPU parts: header probe vs Pillow, sizing vs transformers' smart_resize, M-RoPE positions vs
transformers' get_rope_index, the pool's pixels vs HF's pixel_values, the pixel arena, fetching, a pool worker end to
end, the stored-size limit before any decode (images.max_source_pixels against a poison image), and transparency
composited as vLLM does. All on synthetic images: the test pattern (`llm_mini_infer.synthetic.pattern`) in each
format, colour mode and EXIF/XMP orientation the probe reads, and random pixels for transparency."""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import multiprocessing as mp
import random
import re
import struct
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image, ImageOps, PngImagePlugin

from llm_mini_infer.synthetic import pattern  # the host check's synthetic images too
from llm_mini_infer.vision import arena as px_arena
from llm_mini_infer.vision.family import ImageSize, Span, span_key
from llm_mini_infer.vision.fetch import Fetcher, FetchError, from_data_uri, image_url
from llm_mini_infer.vision.pool import Done
from llm_mini_infer.vision.probe import Header, ProbeError, probe
from llm_mini_infer.vision.qwen import QwenVision, smart_resize

from ..support.fixtures import QWEN, require_qwen
from ..support.vision_pool import GEMMA70, POISON, encoded, exif, run_jobs, spawn_pool_worker

FAMILY = QwenVision(start_id=248053, pad_id=248056, end_id=248054, patch=16, merge=2, temporal=2, min_pixels=65536,
                    max_pixels=327680, mean=(0.5,) * 3, std=(0.5,) * 3, rescale=1 / 255, mrope=True)


def variants() -> list[tuple[str, bytes]]:
    img = pattern(40, 30)
    out = []
    for o in range(1, 9):
        for fmt in ("JPEG", "PNG", "WEBP"):
            out.append((f"{fmt}-exif{o}", encoded(img, fmt, exif=exif(o, big=o % 2 == 0, typ=3 if o < 5 else 4))))
        xmp = (f'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF><rdf:Description tiff:Orientation="{o}"/>'
               f"</rdf:RDF></x:xmpmeta>").encode()
        out.append((f"JPEG-xmp{o}", encoded(img, "JPEG", xmp=xmp)))
        info = PngImagePlugin.PngInfo()
        info.add_itxt("XML:com.adobe.xmp", xmp.decode(), zip=o % 2 == 0)
        out.append((f"PNG-xmp{o}", encoded(img, "PNG", pnginfo=info)))
    out += [(f, encoded(img, f)) for f in ("GIF", "BMP")]
    out += [("progressive", encoded(pattern(64, 48, 1), "JPEG", progressive=True)),
            ("cmyk", encoded(pattern(50, 40, 2).convert("CMYK"), "JPEG")),
            ("gray", encoded(pattern(50, 40, 3).convert("L"), "JPEG")),
            ("rgba", encoded(pattern(50, 40, 4).convert("RGBA"), "PNG")),
            ("palette", encoded(pattern(50, 40, 5).convert("P", palette=Image.ADAPTIVE, colors=16), "PNG")),
            ("gray16", encoded(pattern(50, 40, 6).convert("L").convert("I;16"), "PNG")),
            ("la", encoded(pattern(50, 40, 7).convert("LA"), "PNG")),
            ("lossless-webp", encoded(pattern(50, 40, 8), "WEBP", lossless=True))]
    return out


@pytest.mark.parametrize(("name", "data"), variants(), ids=[v[0] for v in variants()])
def test_probe_matches_pillow(name, data) -> None:
    im = Image.open(io.BytesIO(data))
    h = probe(data)
    assert (h.width, h.height) == im.size
    assert h.oriented == ImageOps.exif_transpose(im).size


def test_probe_rejects_what_it_cannot_read() -> None:
    for data in (b"", b"not an image at all", b"\xff\xd8\xff\xe0\x00\x10JFIF\x00", b"\x89PNG\r\n\x1a\n" + b"\0" * 8):
        with pytest.raises(ProbeError):
            probe(data)


def test_a_png_text_chunk_inflates_at_most_as_far_as_pillow_reads_it() -> None:
    """The probe runs on the frontend's event loop: a compressed text chunk that inflates past 1 MiB (Pillow's
    MAX_TEXT_CHUNK) is refused, as Pillow refuses to open the image, instead of inflating about 1,000 times. At the
    cap both read it."""
    import zlib

    from llm_mini_infer.vision.probe import MAX_TEXT_CHUNK

    assert MAX_TEXT_CHUNK == PngImagePlugin.MAX_TEXT_CHUNK

    def png(text: bytes, kind: bytes) -> bytes:
        plain = encoded(pattern(8, 6, 1), "PNG")
        body = (b"Raw profile type exif\0\0" + zlib.compress(text) if kind == b"zTXt" else
                b"XML:com.adobe.xmp\0\x01\0\0\0" + zlib.compress(text))
        chunk = struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))
        return plain[:33] + chunk + plain[33:]  # after IHDR

    for kind in (b"zTXt", b"iTXt"):
        at_cap = png(b" " * MAX_TEXT_CHUNK, kind)
        h = probe(at_cap)
        assert (h.width, h.height) == Image.open(io.BytesIO(at_cap)).size == (8, 6)
        bomb = png(b" " * (MAX_TEXT_CHUNK + 1), kind)
        assert len(bomb) < 4096  # a few kilobytes that would inflate past the cap
        with pytest.raises(ProbeError, match="inflates past 1 MiB"):
            probe(bomb)
        with pytest.raises(ValueError, match="MAX_TEXT_CHUNK"):
            Image.open(io.BytesIO(bomb)).load()


def test_smart_resize_matches_transformers() -> None:
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize as hf

    rng = random.Random(0)
    sizes = [(225, 400), (711, 400), (1, 1), (33, 31), (6000, 40), (16, 3200)]
    sizes += [(rng.randint(1, 5000), rng.randint(1, 5000)) for _ in range(400)]
    for h, w in sizes:
        for lo, hi in ((65536, 327680), (65536, 16777216), (3136, 1003520)):
            try:
                want = hf(h, w, 32, lo, hi)
            except ValueError:
                with pytest.raises(ValueError):
                    smart_resize(h, w, 32, lo, hi)
                continue
            assert smart_resize(h, w, 32, lo, hi) == want, (h, w, lo, hi)


def test_qwen_tokens_for_400_by_225_and_400_by_711() -> None:
    """A 400 x 225 and a 400 x 711 image under images.min_pixels/max_pixels 65,536-327,680: 84 tokens for
    400x225 (half-to-even: 384, not 416), 264 for 400x711."""
    for (w, h), tokens in (((400, 225), 84), ((400, 711), 264)):
        assert FAMILY.size(probe(encoded(pattern(w, h), "JPEG"))).tokens == tokens


def test_positions_match_transformers() -> None:
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

    fake = types.SimpleNamespace(config=types.SimpleNamespace(vision_config=types.SimpleNamespace(
        spatial_merge_size=2)))
    fake.get_vision_position_ids = lambda *a, **k: Qwen3_5Model.get_vision_position_ids(fake, *a, **k)
    rng = random.Random(1)
    for _ in range(200):
        toks, spans, grids = [], [], []
        for _ in range(rng.randint(0, 3)):
            toks += [5] * rng.randint(0, 7) + [FAMILY.start_id]
            gh, gw = 2 * rng.randint(1, 9), 2 * rng.randint(1, 9)
            size = ImageSize(16 * gh, 16 * gw, (1, gh, gw), gh * gw // 4)
            spans.append(Span(len(toks), size))
            grids.append([1, gh, gw])
            toks += [FAMILY.pad_id] * size.tokens + [FAMILY.end_id]
        toks += [7] * rng.randint(1, 9)
        ids = torch.tensor([toks])
        pos, delta = Qwen3_5Model.get_rope_index(fake, ids, (ids == FAMILY.pad_id).int(),
                                                 torch.tensor(grids) if grids else None)
        rope, off = FAMILY.positions(len(toks), spans)
        if rope is None:
            rope = np.arange(len(toks))[None].repeat(3, 0)
        assert np.array_equal(rope, pos[:, 0].numpy()) and off == int(delta[0, 0])


def hf_pixel_values(data: bytes, family: QwenVision, model_dir: Path) -> np.ndarray:
    from transformers import AutoImageProcessor
    from transformers.image_utils import load_image

    proc = AutoImageProcessor.from_pretrained(str(model_dir))
    img = load_image(Image.open(io.BytesIO(data)))
    out = proc(images=[img], return_tensors="np",
               size={"shortest_edge": family.min_pixels, "longest_edge": family.max_pixels})
    return out["pixel_values"]


def patches(px: torch.Tensor) -> np.ndarray:
    from ..support.vision_tower import hf_patches

    return hf_patches(px).numpy()


@pytest.mark.parametrize(("name", "data"), [v for v in variants() if v[0].endswith(("exif3", "exif6", "exif7"))
                                            or v[0] in ("cmyk", "gray", "rgba", "palette", "gray16", "la",
                                                        "progressive", "GIF", "BMP")],
                         ids=lambda x: x if isinstance(x, str) else "")
def test_pool_pixels_are_hfs_pixel_values(name, data) -> None:
    """The pool's exact path, then the GPU half's normalisation and patchify (here on CPU): bit-identical with HF's
    processor (its torchvision backend, Pillow decode and exif_transpose)."""
    model_dir = require_qwen()
    if not (model_dir / "preprocessor_config.json").exists():
        pytest.skip("no preprocessor_config.json")
    header = probe(data)
    size = FAMILY.size(header)
    px = FAMILY.pixels(data, header, size)
    assert px.dtype == torch.uint8 and tuple(px.shape) == (3, size.height, size.width)
    assert np.array_equal(patches(px), hf_pixel_values(data, FAMILY, model_dir))


def test_span_keys() -> None:
    k = span_key(b"\x01" * 32)
    assert (1 << 62) <= k < (1 << 63) and k != span_key(b"\x02" * 32)


def test_arena_allocator_first_fit_and_coalescing() -> None:
    a = px_arena.Allocator(4096, 10 * 4096)
    x = a.alloc(5000)  # 2 pages
    y = a.alloc(4096)
    z = a.alloc(3 * 4096)
    assert (x, y, z) == (4096, 3 * 4096, 4 * 4096) and a.free_bytes == 4 * 4096
    assert a.alloc(5 * 4096) is None
    a.free(y, 4096)
    a.free(x, 5000)
    assert a.alloc(3 * 4096) == 4096  # x and y coalesced
    a.free(4096, 3 * 4096)
    a.free(z, 3 * 4096)
    assert a.free_bytes == 10 * 4096 and a.alloc(10 * 4096) == 4096


def test_arena_file_maps_shared(tmp_path) -> None:
    path = str(tmp_path / "arena")
    px_arena.create(path, 1 << 16)
    w, r = px_arena.map_writable(path), px_arena.map_readonly(path)
    w.buf[4096:4100] = b"abcd"
    assert bytes(r.buf[4096:4100]) == b"abcd"
    with pytest.raises(ValueError):  # as the engine reads it (runner.cuda.images)
        np.frombuffer(r.buf, np.uint8, 1, 1 << 16)
    w.close()
    r.close()
    px_arena.remove(path)


def test_data_uris() -> None:
    import base64

    data = encoded(pattern(8, 8), "PNG")
    url = "data:image/png;base64," + base64.b64encode(data).decode()
    assert from_data_uri(url, 1 << 20) == data
    assert image_url({"url": url, "detail": "auto"}) == url and image_url(url) == url
    for bad in ("data:image/png,abc", "data:text/plain;base64,QUJD", "data:image/png;base64,***",
                "data:image/png;base64"):
        with pytest.raises(FetchError):
            from_data_uri(bad, 1 << 20)
    with pytest.raises(FetchError):
        from_data_uri(url, 10)
    with pytest.raises(FetchError):
        image_url({"detail": "auto"})


def test_http_fetch_with_limits() -> None:
    """A local server: a 200 with the image, a 404, an oversized body and a slow one; with URLs off (the default),
    none is fetched."""
    from aiohttp import web

    data = encoded(pattern(16, 16), "JPEG")

    async def main() -> None:
        async def ok(_):
            return web.Response(body=data, content_type="image/jpeg")

        async def big(_):
            return web.Response(body=b"x" * 5000)

        async def slow(_):
            await asyncio.sleep(2)
            return web.Response(body=data)

        app = web.Application()
        app.add_routes([web.get("/ok", ok), web.get("/big", big), web.get("/slow", slow)])
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        off = Fetcher(max_bytes=4096, timeout_s=0.5)
        with pytest.raises(FetchError, match=r"images\.allow_urls"):
            await off.fetch(f"http://127.0.0.1:{port}/ok")
        f = Fetcher(max_bytes=4096, timeout_s=0.5, allow_urls=True, allowed_networks=["127.0.0.0/8"])
        try:
            assert await f.fetch(f"http://127.0.0.1:{port}/ok") == data
            for path, msg in (("/missing", "HTTP 404"), ("/big", "larger than"), ("/slow", "longer than")):
                with pytest.raises(FetchError, match=msg):
                    await f.fetch(f"http://127.0.0.1:{port}{path}")
            with pytest.raises(FetchError):
                await f.fetch("ftp://example.com/x.png")
        finally:
            await f.close()
            await runner.cleanup()

    asyncio.run(main())


class StubResolver:
    """An aiohttp resolver with a table of answers, and the names it was asked for."""

    def __init__(self, table: dict[str, list[str]]):
        self.table, self.calls = table, []

    async def resolve(self, host, port=0, family=None):
        import socket

        self.calls.append(host)
        return [{"hostname": host, "host": a, "port": port, "family": socket.AF_INET6 if ":" in a else socket.AF_INET,
                 "proto": 0, "flags": 0} for a in self.table[host]]

    async def close(self) -> None:
        pass


class Site:
    """A local aiohttp server on 127.0.0.1: `routes` (path to handler), the hits and Host headers it saw."""

    def __init__(self, routes: dict):
        from aiohttp import web

        self.hits, self.hosts = [], []

        def wrap(handler):
            async def inner(request):
                self.hits.append(request.path)
                self.hosts.append(request.headers.get("Host"))
                return await handler(request)
            return inner

        app = web.Application()
        app.add_routes([web.get(path, wrap(h)) for path, h in routes.items()])
        self.runner = web.AppRunner(app)

    async def __aenter__(self) -> Site:
        from aiohttp import web

        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        self.port = site._server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *exc) -> None:
        await self.runner.cleanup()


def image_site(data: bytes, **extra) -> Site:
    from aiohttp import web

    async def ok(_):
        return web.Response(body=data, content_type="image/jpeg")

    return Site({"/img": ok, **extra})


def test_a_url_on_the_servers_own_network_is_refused_by_default() -> None:
    """With URLs on and no allowed_networks, a URL to loopback is refused before any request (the handler is hit
    0 times) with the one refusal message, which names no address; with the network allowed it is fetched."""
    from llm_mini_infer.vision import urlguard

    data = encoded(pattern(16, 16), "JPEG")

    async def main() -> None:
        async with image_site(data) as site:
            f = Fetcher(max_bytes=1 << 20, timeout_s=5, allow_urls=True)
            try:
                with pytest.raises(FetchError) as err:
                    await f.fetch(f"http://127.0.0.1:{site.port}/img")
            finally:
                await f.close()
            assert str(err.value) == urlguard.REFUSED and "127.0.0.1" not in str(err.value) and site.hits == []
            f = Fetcher(max_bytes=1 << 20, timeout_s=5, allow_urls=True, allowed_networks=["127.0.0.0/8"])
            try:
                assert await f.fetch(f"http://127.0.0.1:{site.port}/img") == data
            finally:
                await f.close()
            assert site.hits == ["/img"]

    asyncio.run(main())


def redirect(location: str, status: int = 302):
    from aiohttp import hdrs, web

    async def handler(_):
        return web.Response(status=status, headers={hdrs.LOCATION: location})

    return handler


@pytest.mark.parametrize("target", ["http://169.254.169.254/", "http://[::1]:{port}/img", "http://meta.test/img",
                                    "ftp://x/"], ids=["metadata", "ipv6-loopback", "name-to-private", "scheme"])
def test_a_redirect_to_a_blocked_address_is_refused_before_it_is_followed(target) -> None:
    """A fetched URL that redirects to a metadata address, an IPv6 loopback literal, a name that resolves to a private
    address, or another scheme is refused with the one message; the second server sees no request."""
    from llm_mini_infer.vision import urlguard

    data = encoded(pattern(16, 16), "JPEG")

    async def main() -> None:
        async with image_site(data) as second, Site({"/go": redirect(target.format(port=second.port))}) as first:
            stub = StubResolver({"meta.test": ["10.0.0.1"]})
            f = Fetcher(max_bytes=1 << 20, timeout_s=5, allow_urls=True, allowed_networks=["127.0.0.0/8"],
                        resolver=stub)
            try:
                with pytest.raises(FetchError) as err:
                    await f.fetch(f"http://127.0.0.1:{first.port}/go")
            finally:
                await f.close()
            assert str(err.value) == urlguard.REFUSED, err.value
            assert first.hits == ["/go"] and second.hits == []

    asyncio.run(main())


def test_a_url_or_location_yarl_cannot_parse_is_an_invalid_url() -> None:
    """A URL with an unclosed IPv6 bracket, a port past 65535 or a host whose IDNA form does not decode (`xn--`), and a
    redirect to one, are a FetchError (a 400), never an exception the API answers with a 500."""
    from aiohttp import hdrs, web

    async def main() -> None:
        async def bad_location(_):
            return web.Response(status=302, headers={hdrs.LOCATION: "http://[::1"})

        async def bad_idna(_):
            return web.Response(status=302, headers={hdrs.LOCATION: "http://xn--/"})

        async with Site({"/go": bad_location, "/idna": bad_idna}) as site:
            f = Fetcher(max_bytes=1 << 20, timeout_s=5, allow_urls=True, allowed_networks=["127.0.0.0/8"])
            try:
                for url in ("http://[::1/x", "http://x:99999/", "http://xn--/", f"http://127.0.0.1:{site.port}/go",
                            f"http://127.0.0.1:{site.port}/idna"):
                    with pytest.raises(FetchError) as err:
                        await f.fetch(url)
                    assert str(err.value) == "fetching the image failed: an invalid URL", url
            finally:
                await f.close()

    asyncio.run(main())


def test_redirects_are_followed_up_to_five_and_relative_locations_resolve() -> None:
    """A chain of five redirects is fetched, a chain of six is refused, and a relative Location is joined to the
    URL it came from."""
    data = encoded(pattern(16, 16), "JPEG")

    async def main() -> None:
        routes = {f"/r{i}": redirect(f"/r{i + 1}" if i < 5 else "/img", 307) for i in range(6)}
        routes["/rel"] = redirect("img", 301)
        async with image_site(data, **routes) as site:
            f = Fetcher(max_bytes=1 << 20, timeout_s=5, allow_urls=True, allowed_networks=["127.0.0.0/8"])
            try:
                base = f"http://127.0.0.1:{site.port}"
                assert await f.fetch(f"{base}/r1") == data  # r1..r5: five redirects
                assert await f.fetch(f"{base}/rel") == data
                with pytest.raises(FetchError, match="more than 5 redirects"):
                    await f.fetch(f"{base}/r0")  # six
            finally:
                await f.close()

    asyncio.run(main())


def test_the_fetch_deadline_covers_every_redirect_hop() -> None:
    """Hops that each take 0.6 s under a 1 s fetch timeout fail at about 1 s, not after all the hops."""
    import time

    from aiohttp import hdrs, web

    def slow_redirect(location: str):
        async def handler(_):
            await asyncio.sleep(0.6)
            return web.Response(status=302, headers={hdrs.LOCATION: location})
        return handler

    async def main() -> None:
        routes = {f"/s{i}": slow_redirect(f"/s{i + 1}") for i in range(5)}
        async with Site(routes) as site:
            f = Fetcher(max_bytes=1 << 20, timeout_s=1.0, allow_urls=True, allowed_networks=["127.0.0.0/8"])
            try:
                t0 = time.monotonic()
                with pytest.raises(FetchError, match="took longer than"):
                    await f.fetch(f"http://127.0.0.1:{site.port}/s0")
                assert time.monotonic() - t0 < 1.5
            finally:
                await f.close()

    asyncio.run(main())


def test_a_name_is_resolved_once_and_the_request_goes_to_the_checked_address() -> None:
    """A name resolving to an allowed address is asked of the resolver once and connected to at that address, with the
    name in the Host header; an answer holding any blocked address is refused whole."""
    from llm_mini_infer.vision import urlguard

    data = encoded(pattern(16, 16), "JPEG")

    async def main() -> None:
        async with image_site(data) as site:
            stub = StubResolver({"img.test": ["127.0.0.1"], "mixed.test": ["8.8.8.8", "127.0.0.1"]})
            f = Fetcher(max_bytes=1 << 20, timeout_s=5, allow_urls=True, allowed_networks=["127.0.0.0/8"],
                        resolver=stub)
            try:
                assert await f.fetch(f"http://img.test:{site.port}/img") == data
                assert site.hosts == [f"img.test:{site.port}"] and stub.calls == ["img.test"]
            finally:
                await f.close()
            stub = StubResolver({"mixed.test": ["8.8.8.8", "127.0.0.1"]})
            f = Fetcher(max_bytes=1 << 20, timeout_s=5, allow_urls=True, resolver=stub)
            try:
                with pytest.raises(FetchError) as err:
                    await f.fetch(f"http://mixed.test:{site.port}/img")
            finally:
                await f.close()
            assert str(err.value) == urlguard.REFUSED and len(site.hits) == 1

    asyncio.run(main())


def test_pool_worker_round_trip(tmp_path) -> None:
    """A real worker process (spawned, one torch thread): it decodes, orients and resizes into the arena; a corrupt
    image comes back as an error."""
    model_dir = require_qwen()
    import shutil

    from llm_mini_infer.engine import ipc as engine_ipc
    from llm_mini_infer.vision.pool import Job, PoolClient, worker_main

    ipc = engine_ipc.ipc_dir()  # a short path: sun_path holds about 104 bytes
    path = str(tmp_path / "arena")
    px_arena.create(path, 1 << 22)
    ctx = mp.get_context("spawn")
    r, s = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=worker_main, args=(str(model_dir), 65536, 327680, 0, ipc, path, s, "warning"))
    proc.start()
    s.close()
    try:
        assert r.poll(120) and r.recv() == "ready"
        data = encoded(pattern(400, 225, 9), "JPEG", exif=exif(6))
        header = probe(data)
        size = FAMILY.size(header)

        async def main():
            client = PoolClient(ipc, 1)
            client.start()
            try:
                job = Job(jid=0, fmt=header.format, width=header.width, height=header.height,
                          orientation=header.orientation, out_h=size.height,
                          out_w=size.width, grid_t=1, grid_h=size.grid[1], grid_w=size.grid[2], tokens=size.tokens,
                          offset=8192)
                ok = await asyncio.wait_for(client.submit(job, data), 60)
                bad = await asyncio.wait_for(client.submit(Job(**{**{f: getattr(job, f) for f in job.__struct_fields__},
                                                                  "fmt": "png"}), b"garbage"), 60)
                return ok, bad
            finally:
                client.close()

        ok, bad = asyncio.run(main())
        assert not ok.error and ok.decode_ns > 0 and bad.error
        view = px_arena.map_readonly(path)
        got = np.frombuffer(view.buf, np.uint8, 3 * size.height * size.width, 8192).reshape(3, size.height,
                                                                                           size.width).copy()
        assert np.array_equal(got, FAMILY.pixels(data, header, size).numpy())
        view.close()
    finally:
        proc.kill()  # as the launcher stops the pool (a worker ignores SIGTERM while its parent lives)
        proc.join(10)
        px_arena.remove(path)
        shutil.rmtree(ipc, ignore_errors=True)


def test_pool_client_never_loses_a_reply() -> None:
    """Replies that arrive while the client sends on the same DEALER socket are still read: ZMQ's file descriptor
    signals edges, and a send can consume the edge of a reply that just arrived (a reply left unread hangs its request,
    and the least-outstanding dispatch never sends that worker another job). Echo workers in threads; bursts of jobs,
    every reply awaited with a deadline."""
    import shutil
    import threading
    import time

    import msgspec
    import zmq

    from llm_mini_infer.engine import ipc as engine_ipc
    from llm_mini_infer.vision.pool import Done, Job, PoolClient, endpoint

    ipc = engine_ipc.ipc_dir()
    stop = threading.Event()

    def echo(i: int) -> None:
        ctx = zmq.Context()
        sock = ctx.socket(zmq.ROUTER)
        sock.bind(endpoint(ipc, i))
        dec, enc = msgspec.msgpack.Decoder(Job), msgspec.msgpack.Encoder()
        while not stop.is_set():
            if sock.poll(20):
                ident, head, _ = sock.recv_multipart()
                sock.send_multipart([ident, enc.encode(Done(jid=dec.decode(head).jid))])
        sock.close(0)
        ctx.term()

    workers = [threading.Thread(target=echo, args=(i,), daemon=True) for i in range(2)]
    for t in workers:
        t.start()
    job = Job(jid=0, fmt="jpeg", width=1, height=1, orientation=1, out_h=1, out_w=1, grid_t=1, grid_h=1,
              grid_w=1, tokens=1, offset=0)

    async def main() -> int:
        client = PoolClient(ipc, 2)
        client.start()
        done = 0
        try:
            for _ in range(40):
                futs = []
                for _ in range(4):  # a reply lands between two sends to its socket, before the loop polls
                    futs.append(client.submit(job, b"x"))
                    time.sleep(0.002)
                for f in futs:
                    await asyncio.wait_for(f, 2)
                    done += 1
        finally:
            client.close()
        return done

    try:
        assert asyncio.run(main()) == 160
    finally:
        stop.set()
        for t in workers:
            t.join(5)
        shutil.rmtree(ipc, ignore_errors=True)


def test_family_from_the_checkpoint() -> None:
    from llm_mini_infer.vision import family

    model_dir = require_qwen()
    fam = family.load(model_dir, max_pixels=327680)
    hf = json.loads((model_dir / "config.json").read_text())
    assert isinstance(fam, QwenVision) and fam.pad_id == hf["image_token_id"] and fam.max_tokens() == 320
    assert family.load(QWEN.parent / "nonexistent") is None


# -- the poison image: a header over images.max_source_pixels ----------------------------------------------------------


def uri(data: bytes) -> str:
    import base64

    return "data:image/jpeg;base64," + base64.b64encode(data).decode()


def test_check_source_pixels_is_the_decompression_bomb_bound() -> None:
    """178,956,970 (Pillow's decompression-bomb bound, vLLM's VLLM_MAX_IMAGE_PIXELS default) on the stored size,
    strictly above it, in vLLM's words with our setting's name; the setting's default is the same number."""
    from llm_mini_infer import config
    from llm_mini_infer.vision import probe as pr

    pr.check_source_pixels(Header("jpeg", 2, 89_478_485))  # exactly the limit
    with pytest.raises(ProbeError, match="exceed the maximum of 178956970 pixels"):
        pr.check_source_pixels(Header("jpeg", 1, 178_956_971))
    h = probe(POISON)
    assert (h.width, h.height) == (65535, 65535)
    with pytest.raises(ProbeError, match=r"65535x65535 \(4294836225 pixels\)"):
        pr.check_source_pixels(h)
    with pytest.raises(ProbeError, match="exceed the maximum of 10000 pixels"):  # a lowered limit
        pr.check_source_pixels(probe(encoded(pattern(400, 225), "JPEG")), 10_000)
    assert config.ImageConfig().max_source_pixels == pr.MAX_SOURCE_PIXELS == 178_956_970


class RecordingPool:
    """The image pool as a frontend sees it, recording the jobs it is given (each answered at once)."""

    def __init__(self) -> None:
        self.jobs: list = []

    def start(self) -> None:
        pass

    def close(self) -> None:
        pass

    def submit(self, job, data: bytes) -> asyncio.Future:
        self.jobs.append(job)
        fut = asyncio.get_running_loop().create_future()
        fut.set_result(Done(jid=job.jid))
        return fut


@pytest.mark.parametrize("fam", [FAMILY, GEMMA70], ids=["qwen", "gemma"])
def test_the_frontend_refuses_a_poison_header_before_the_pool(fam) -> None:
    """A header declaring more than images.max_source_pixels is a 400 at the probe, before sizing, the pixel arena and
    the pool (neither family's sizing refuses a size: Gemma resizes anything to its budget): a request with one submits
    nothing, not even its valid images, and /tokenize's sizing (submit=False) refuses it too. A lowered limit refuses
    a 400 x 225 image (90,000 pixels) and takes a smaller one."""
    from llm_mini_infer.api.images import ImageInputs
    from llm_mini_infer.api.protocol import ApiError

    def inputs(**kw) -> ImageInputs:
        im = ImageInputs(fam, index=0, num_frontends=1, arena_bytes=16 << 20, pool_dir="", workers=1,
                         max_bytes=20 << 20, fetch_timeout_s=1, max_per_request=2, **kw)
        im.pool = RecordingPool()
        im.start()
        return im

    async def main() -> None:
        im = inputs()
        for submit in (True, False):
            with pytest.raises(ApiError) as err:
                await im.prepare([uri(encoded(pattern(40, 30), "JPEG")), uri(POISON)], submit=submit)
            assert (err.value.status, err.value.param) == (400, "messages")
            assert err.value.message == ("image: Image dimensions 65535x65535 (4294836225 pixels) exceed the maximum "
                                         "of 178956970 pixels (images.max_source_pixels)")
        assert im.pool.jobs == []
        low = inputs(max_source_pixels=80_000)
        with pytest.raises(ApiError, match=r"400x225 \(90000 pixels\) exceed the maximum of 80000 pixels"):
            await low.prepare([uri(encoded(pattern(400, 225), "JPEG"))])
        prep = await low.prepare([uri(encoded(pattern(300, 200), "JPEG"))])
        assert len(prep.images) == 1 and len(low.pool.jobs) == 1
        low.discard(prep)

    asyncio.run(main())


def frontend_inputs(**kw):
    """A frontend's image step (Qwen's sizing) over a pool that records its jobs."""
    from llm_mini_infer.api.images import ImageInputs

    im = ImageInputs(FAMILY, index=0, num_frontends=1, arena_bytes=16 << 20, pool_dir="", workers=1,
                     max_bytes=4 << 20, fetch_timeout_s=5, max_per_request=4, **kw)
    im.pool = RecordingPool()
    im.start()
    return im


def test_a_requests_decode_budget_is_vllms() -> None:
    """vLLM's per-request decode budget (VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES, images.max_request_decode_bytes): 4
    bytes a stored pixel, summed in prompt order; the image that takes the sum past it is a 400 with vLLM's text (the
    sum counts it), no param, before anything is submitted, and so for /tokenize's sizing (submit=False). Each image
    alone passes. 0 is no limit; the defaults are vLLM's."""
    from llm_mini_infer import config
    from llm_mini_infer.api.images import DECODE_BYTES, DOWNLOAD_MB
    from llm_mini_infer.api.protocol import ApiError

    one = uri(encoded(pattern(400, 225), "JPEG"))  # 360,000 bytes at 4 a pixel

    async def main() -> None:
        im = frontend_inputs(max_request_decode_bytes=1_000_000)
        for submit in (True, False):
            with pytest.raises(ApiError) as err:
                await im.prepare([one, one, one], submit=submit)
            e = err.value
            assert (e.status, e.param, e.kind) == (400, None, "BadRequestError")
            assert e.message == ("Request decoded media exceeds images.max_request_decode_bytes=1000000 (1080000 "
                                 "bytes).")
        assert im.pool.jobs == []
        prep = await im.prepare([one, one])  # 720,000
        assert len(im.pool.jobs) == 2
        im.discard(prep)
        off = frontend_inputs(max_request_decode_bytes=0)
        off.discard(await off.prepare([one, one, one, one]))
        assert len(off.pool.jobs) == 4

    asyncio.run(main())
    im = config.ImageConfig()
    assert im.max_request_decode_bytes == DECODE_BYTES == 715_827_880 == 4 * im.max_source_pixels
    assert im.max_request_download_mb == DOWNLOAD_MB == 256


def test_a_requests_download_budget_is_vllms() -> None:
    """vLLM's per-request download budget (VLLM_MAX_MEDIA_REQUEST_DOWNLOAD_SIZE_MB, images.max_request_download_mb):
    every chunk of a request's http(s) downloads is counted as it arrives; past the MiB limit the download stops, and
    the request gets vLLM's 422 (its fetch wrapper's text, the URL sanitized as vLLM sanitizes messages, param
    image_url). Within the limit, and with 0, the downloads complete; a data URI is never counted."""
    import base64
    import re

    from aiohttp import web

    from llm_mini_infer.api.protocol import ApiError
    from llm_mini_infer.vision.fetch import DownloadBudget, Fetcher, fetch_all

    body = b"x" * (600 << 10)  # two make 1.2 MiB

    async def main() -> None:
        async def serve(_):
            return web.Response(body=body, content_type="image/png")

        app = web.Application()
        app.add_routes([web.get("/a", serve), web.get("/img/b.png", serve)])
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        base = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
        f = Fetcher(max_bytes=4 << 20, timeout_s=5, allow_urls=True, allowed_networks=["127.0.0.0/8"])
        try:
            for limit in (2, 0):
                budget = DownloadBudget(limit)
                assert await fetch_all(f, [f"{base}/a", f"{base}/img/b.png"], budget) == [body, body]
                assert budget.used == (2 * len(body) if limit else 0)
            budget = DownloadBudget(1)
            data = "data:image/png;base64," + base64.b64encode(b"y" * (3 << 19)).decode()  # 1.5 MiB, not counted
            assert len((await fetch_all(f, [data], budget))[0]) == 3 << 19 and budget.used == 0
            im = frontend_inputs(allow_urls=True, allowed_networks=["127.0.0.0/8"], max_request_download_mb=1)
            for urls in ([f"{base}/a", f"{base}/a"], [f"{base}/img/b.png", f"{base}/img/b.png"]):
                with pytest.raises(ApiError) as err:
                    await im.prepare(urls)
                e = err.value
                assert (e.status, e.param, e.kind) == (422, "image_url", "UnprocessableEntityError")
                got = re.fullmatch(r"Failed to fetch media from URL: Request media downloads exceed images\."
                                   r"max_request_download_mb=1 \((\d+) bytes received, including retried attempts\)\. "
                                   r"\(parameter=image_url, value=(.*)\)", e.message)
                assert got and 1 << 20 < int(got[1]) <= 1 << 20 | 1 << 16, e.message  # the chunk past it counted
                assert got[2] == (f"{base}/a" if urls[0].endswith("/a") else f"{base}<path>")
            await im.close()
        finally:
            await f.close()
            await runner.cleanup()

    asyncio.run(main())


def test_a_pool_worker_refuses_an_oversized_header_without_decoding(tmp_path) -> None:
    """The pool worker repeats the check on its job's header before anything decodes (a job past the frontend must
    never reach the JPEG decoder, which allocates what the header says): with images.max_source_pixels
    100,000, a 400 x 225 image (90,000 pixels) decodes; a 400 x 300 one (120,000, perfectly decodable) is refused by
    the guard instead of decoded, and so is the poison (the guard's message, not libjpeg's); the worker stays alive
    and decodes the next image."""
    import shutil

    from llm_mini_infer.engine import ipc as engine_ipc

    gemma = tmp_path / "gemma"  # a Gemma 4 checkpoint's config.json alone: images of at most 70 soft tokens
    gemma.mkdir()
    (gemma / "config.json").write_text(json.dumps({"model_type": "gemma4", "vision_config": {
        "patch_size": 16, "pooling_kernel_size": 3, "default_output_length": 70}}))
    ipc = engine_ipc.ipc_dir()
    arena = str(tmp_path / "arena")
    px_arena.create(arena, 1 << 22)
    proc = None
    try:
        proc, ready = spawn_pool_worker(gemma, None, None, ipc, arena, max_source_pixels=100_000)
        assert ready, f"the worker exited before it was ready (exit code {proc.exitcode})"
        blobs = [encoded(pattern(400, 225), "JPEG"), encoded(pattern(400, 300), "JPEG"), POISON,
                 encoded(pattern(400, 225, 1), "JPEG")]
        jobs = [(data, probe(data), GEMMA70.size(probe(data)), 8192) for data in blobs]
        ok, guarded, poison, again = run_jobs(ipc, jobs)
        assert not ok.error and ok.decode_ns > 0
        assert "exceed the maximum of 100000 pixels" in guarded.error, guarded.error
        assert "exceed the maximum of 100000 pixels" in poison.error and "SOS" not in poison.error, poison.error
        assert not again.error and proc.is_alive()
    finally:
        if proc is not None:
            proc.kill()
            proc.join(10)
        px_arena.remove(arena)
        shutil.rmtree(ipc, ignore_errors=True)


def test_a_gif_frame_larger_than_its_screen_fails_before_decoding(monkeypatch) -> None:
    """The probe reads a GIF's logical screen; Pillow sizes the image by its first frame when that is larger. The
    pool's decode refuses the disagreement before it loads a pixel (a frame over the screen's size, up to Pillow's own
    bomb bound, is never decoded)."""
    from PIL import ImageFile

    from llm_mini_infer.vision.family import decode_rgb

    gif = bytearray(encoded(pattern(64, 48).convert("P"), "GIF"))
    struct.pack_into("<HH", gif, 6, 16, 16)  # the logical screen; the frame stays 64 x 48
    gif = bytes(gif)
    header = probe(gif)
    assert (header.width, header.height) == (16, 16) and Image.open(io.BytesIO(gif)).size == (64, 48)

    def load(self):
        raise AssertionError("pixels decoded")

    monkeypatch.setattr(ImageFile.ImageFile, "load", load)
    with pytest.raises(ProbeError, match="is 64x48, but its header says 16x16"):
        decode_rgb(gif, header)


# -- EXIF orientation as Pillow loads it ------------------------------------------------------------------------------

# Struct codes of the TIFF types (RATIONAL and SRATIONAL: two LONGs)
TIFF_CODES = {1: "B", 2: "s", 3: "H", 4: "I", 5: "II", 6: "b", 7: "B", 8: "h", 9: "i", 10: "ii", 11: "f", 12: "d",
              13: "I", 16: "Q"}


def ifd0(entries: list[tuple[int, int, int, bytes]], big: bool = False, trailer: bytes = b"") -> bytes:
    """A TIFF block of IFD0's (tag, type, count, 4-byte value field) entries, `trailer` after them (at offset
    10 + 12 × entries + 4)."""
    e = ">" if big else "<"
    body = b"".join(struct.pack(e + "HHI", tag, typ, count) + field for tag, typ, count, field in entries)
    return (b"MM\x00*" if big else b"II*\x00") + struct.pack(e + "IH", 8, len(entries)) + body + \
        struct.pack(e + "I", 0) + trailer


def orientation_entry(typ: int, values: list, big: bool = False, extra: int = 0) -> bytes:
    """IFD0 with one Orientation entry of `values` in type `typ`: inline when they fit 4 bytes, else in the trailer
    (after `extra` padding bytes)."""
    e = ">" if big else "<"
    flat = [x for v in values for x in (v if isinstance(v, tuple) else (v,))]
    data = struct.pack(e + TIFF_CODES[typ][0] * len(flat), *flat) if typ != 2 else b"".join(values)
    if len(data) <= 4:
        return ifd0([(0x0112, typ, len(values), data.ljust(4, b"\0"))], big)
    return ifd0([(0x0112, typ, len(values), struct.pack(e + "I", 26 + extra))], big, bytes(extra) + data)


def pillow_orientation(data: bytes) -> tuple[int, bool]:
    """The orientation `exif_transpose` applies (its lookup on what Pillow reads: 2..8, else 1), and whether it raises
    after reading one (the image is then served as stored)."""
    try:
        v = Image.open(io.BytesIO(data)).getexif().get(274)
    except Exception:
        v = None
    o = int(v) if v in {2, 3, 4, 5, 6, 7, 8} else 1
    try:
        ImageOps.exif_transpose(Image.open(io.BytesIO(data)))
    except Exception:
        return o, o != 1
    return o, False


ORIENTATIONS = [  # (Orientation entry, the orientation Pillow applies)
    (orientation_entry(3, [6, 6]), 6), (orientation_entry(3, [8, 1, 1]), 8), (orientation_entry(4, [5, 1]), 5),
    (orientation_entry(8, [6]), 6), (orientation_entry(9, [7]), 7), (orientation_entry(13, [6]), 6),
    (orientation_entry(6, [8]), 8), (orientation_entry(16, [6]), 6), (orientation_entry(5, [(12, 2)]), 6),
    (orientation_entry(10, [(-16, -2)]), 8), (orientation_entry(11, [6.0]), 6), (orientation_entry(12, [5.0]), 5),
    (orientation_entry(5, [(6, 0)]), 1), (orientation_entry(11, [6.5]), 1), (orientation_entry(8, [-6]), 1),
    (orientation_entry(1, [6]), 1), (orientation_entry(7, [6]), 1), (orientation_entry(2, [b"6"]), 1),
    (orientation_entry(3, [6, 6], big=True), 6), (orientation_entry(9, [6], big=True), 6),
    # an entry Pillow skips (unknown type, no data) leaves the tag to a later one; a stored one keeps it
    (ifd0([(0x0112, 99, 1, b"\6\0\0\0"), (0x0112, 3, 1, b"\6\0\0\0")]), 6),
    (ifd0([(0x0112, 3, 0, b"\6\0\0\0"), (0x0112, 3, 1, b"\5\0\0\0")]), 5),
    (ifd0([(0x0112, 1, 1, b"\6\0\0\0"), (0x0112, 3, 1, b"\6\0\0\0")]), 1),
    # data past the block's end: Pillow stops reading IFD0; past its running bound of the block's length: skipped
    (ifd0([(0x0112, 3, 4, b"\xf0\0\0\0")]), 1), (ifd0([(0x010F, 2, 6, b"\x40\0\0\0"), (0x0112, 3, 1, b"\6\0\0\0")]), 1),
    (ifd0([(0x010F, 2, 60, b"\x08\0\0\0"), (0x0112, 3, 4, b"\x08\0\0\0"), (0x0112, 3, 1, b"\6\0\0\0")]), 6),
]


@pytest.mark.parametrize(("tiff", "want"), ORIENTATIONS, ids=[str(i) for i in range(len(ORIENTATIONS))])
def test_the_probe_reads_the_orientation_pillow_applies(tiff, want) -> None:
    """Any count (its first value), any numeric type (signed, IFD, float, rational: `exif_transpose` compares numbers),
    in JPEG, PNG and WebP. Once a count other than 1 or a type other than SHORT and LONG counted as 1."""
    img = pattern(40, 20)
    for fmt in ("JPEG", "PNG", "WEBP"):
        data = encoded(img, fmt, exif=b"Exif\x00\x00" + tiff if fmt == "JPEG" else tiff)
        assert pillow_orientation(data)[0] == want, fmt
        assert probe(data).orientation == want, fmt


def random_ifd0(rng: random.Random) -> tuple[bytes, bool]:
    """IFD0 with an Orientation entry among 0-3 others, of random types, counts, offsets and trailers (with the
    orientation's values encoded in its type); and whether Pillow stops reading IFD0 at an earlier entry whose data
    lies past the block's end (the probe stops there too)."""
    from PIL import TiffImagePlugin

    big = rng.random() < 0.5
    e = ">" if big else "<"
    tags = [0x0112] + [rng.choice([0x010F, 0x0110, 0x0131, 0x8769, 0x8825, 0x0132, 0x011A, 0x9286,
                                   rng.randint(0, 0xFFFF)]) for _ in range(rng.randint(0, 3))]
    rng.shuffle(tags)
    head = 10 + 12 * len(tags) + 4
    trailer = bytes(rng.randint(0, 24))
    entries = []
    for tag in tags:
        if tag == 0x0112:
            typ = rng.choice([3, 3, 4, 8, 9, 13, 6, 16, 5, 10, 11, 12, 1, 2, 7, 99])
            count = rng.choice([1, 1, 1, 2, 3, 4, 0]) if typ != 2 else rng.choice([1, 2])
            code = TIFF_CODES.get(typ, "I")
            vals = [rng.choice([6, 8, 5, 7, 3, 2, 1, 0, 9, 100]) for _ in range(count * len(code))]
            if typ in (5, 10) and vals:
                vals[1] = rng.choice([1, 1, 2, 0])
                vals[0] *= vals[1] or 1
            if typ == 10 and rng.random() < 0.5:
                vals[:2] = [-v for v in vals[:2]]
            data = b"6" * count if typ == 2 else struct.pack(e + code[0] * len(vals), *vals) if typ != 99 else b"\6"
            if len(data) <= 4:
                field = data.ljust(4, b"\0")
            else:
                at = rng.choice([head + len(trailer), head + len(trailer), rng.randint(0, 200)])
                if at == head + len(trailer):
                    trailer += data
                field = struct.pack(e + "I", at)
        else:
            typ = rng.choice([3, 3, 4, 1, 2, 5, 7, 9, 10, 11, 12, 13, 0, 99])
            count = rng.choice([1, 1, 1, 0, 2, 4, 100, 0xFFFFFFFF]) if rng.random() < 0.5 else 1
            field = struct.pack(e + "I", rng.choice([8, 26, 40, 200, rng.randint(0, 0xFFFFFFFF)]))
        entries.append((tag, typ, count, field))
    tiff = ifd0(entries, big, trailer)
    # Pillow's loop (TiffImagePlugin.ImageFileDirectory_v2.load) up to the orientation: an earlier tag's data past the
    # block's end stops it
    stored: set[int] = set()
    total, stops_early = 0, False
    for tag, typ, count, field in entries:
        if tag in stored or typ not in TiffImagePlugin.ImageFileDirectory_v2._load_dispatch:
            continue
        size = count * TiffImagePlugin.ImageFileDirectory_v2._load_dispatch[typ][0]
        if size > 4:
            total += size
            if total > len(tiff):
                continue
            if struct.unpack(e + "I", field)[0] + size > len(tiff):
                stops_early = tag != 0x0112
                break
        if tag == 0x0112 and size:
            break
        if size:
            stored.add(tag)
    return tiff, stops_early


def entries_of(tiff: bytes) -> list[tuple[int, int, int]]:
    """IFD0's (tag, type, count) entries of an `ifd0` block."""
    e = "<" if tiff[:2] == b"II" else ">"
    return [struct.unpack_from(e + "HHI", tiff, 10 + 12 * k) for k in range(struct.unpack_from(e + "H", tiff, 8)[0])]


def test_the_probe_agrees_with_pillow_on_random_ifd0s() -> None:
    """Random IFD0s in JPEG, PNG and WebP: the probe's orientation is the one `exif_transpose` applies, Pillow stopping
    at an earlier tag's data past the block's end included, but for one known difference (rare: malformed EXIF):
    `exif_transpose` raising after it read an orientation (it fails rewriting the EXIF; the image is served as stored,
    the probe orients it)."""
    import warnings

    rng = random.Random(1)
    img = pattern(40, 20)
    seen = {"agree": 0, "count": 0, "type": 0, "stops early": 0, "raises": 0}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(1500):
            fmt = rng.choice(["JPEG", "PNG", "WEBP"])
            tiff, stops_early = random_ifd0(rng)
            try:
                data = encoded(img, fmt, exif=b"Exif\x00\x00" + tiff if fmt == "JPEG" else tiff)
            except Exception:
                continue
            want, raises = pillow_orientation(data)
            if raises:
                seen["raises"] += 1
                continue
            assert probe(data).orientation == want, (fmt, tiff.hex())
            seen["agree"] += 1
            seen["stops early"] += stops_early
            if want != 1:
                n, typ = next((c, t) for tag, t, c in entries_of(tiff) if tag == 0x0112 and t != 99 and c)
                seen["count"] += n != 1
                seen["type"] += typ not in (3, 4)
    assert seen["agree"] > 1000 and seen["count"] > 30 and seen["type"] > 30 and seen["stops early"] > 50, seen


def test_the_pool_serves_malformed_exif_as_stored_and_decodes_the_probed_format_only() -> None:
    """EXIF that `exif_transpose` cannot read (not TIFF, a broken header): the image as stored, not a failed
    request. A disagreement with the probe still fails it: `exif_transpose` raising after it read orientation 6, which
    the probe applies. Pillow opens the probed format only."""
    from PIL import UnidentifiedImageError

    from llm_mini_infer.vision.family import decode_rgb

    img = pattern(40, 20)
    for fmt in ("JPEG", "PNG", "WEBP"):
        prefix = b"Exif\x00\x00" if fmt == "JPEG" else b""
        for junk in (b"not a tiff header at all", b"II\x2b\x00\x08\x00\x00\x00" + bytes(8)):
            data = encoded(img, fmt, exif=prefix + junk, **({"lossless": True} if fmt == "WEBP" else {}))
            if fmt != "JPEG":  # a JPEG's DPI read at open swallows the error: exif_transpose then finds no EXIF
                with pytest.raises(Exception):  # noqa: B017 - Pillow's errors vary (SyntaxError, struct.error)
                    ImageOps.exif_transpose(Image.open(io.BytesIO(data)))
            assert pillow_orientation(data) == (1, False)
            h = probe(data)
            assert h.orientation == 1 and tuple(decode_rgb(data, h).shape) == (3, 20, 40), (fmt, junk)
        data = encoded(img, fmt, exif=prefix + orientation_entry(3, [6, 6]))  # once orientation 1 in the probe
        assert tuple(decode_rgb(data, probe(data)).shape) == (3, 40, 20)
        raises = bytes.fromhex("4d4d002a000000080002010f000a00000004000000080112000300000001000600060000000000")
        data = encoded(img, fmt, exif=prefix + raises + bytes(48))
        assert pillow_orientation(data) == (6, True) and probe(data).orientation == 6
        with pytest.raises(ProbeError, match="is 40x20, but its header says 20x40"):
            decode_rgb(data, probe(data))
    gif = encoded(img, "GIF")
    with pytest.raises(UnidentifiedImageError):  # a GIF under a PNG header is not decoded as a GIF
        decode_rgb(gif, Header("png", 40, 20))
    assert tuple(decode_rgb(gif, probe(gif)).shape) == (3, 20, 40)


def xmp_packet(o: int) -> bytes:
    return (f'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF><rdf:Description tiff:Orientation="{o}"/>'
            f"</rdf:RDF></x:xmpmeta>").encode()


def pillow_pixels(data: bytes) -> torch.Tensor:
    """The image as vLLM decodes it (`normalize_image`: `exif_transpose`, the image as stored when it raises), RGB."""
    im = Image.open(io.BytesIO(data))
    with contextlib.suppress(Exception):
        im = ImageOps.exif_transpose(im)
    return torch.from_numpy(np.asarray(im.convert("RGB")).copy()).permute(2, 0, 1)


def agrees_with_pillow(data: bytes) -> int:
    """The probe's orientation, checked against Pillow's: the orientation it reads, the oriented size, the pixels."""
    from llm_mini_infer.vision.family import decode_rgb

    h = probe(data)
    want = pillow_pixels(data)
    assert h.orientation == pillow_orientation(data)[0]
    assert h.oriented == (want.shape[2], want.shape[1])
    assert torch.equal(decode_rgb(data, h), want)
    return h.orientation


def test_a_doubled_exif_prefix_and_junk_exif_beside_xmp_are_read_as_pillow_reads_them() -> None:
    """#23's two cases, at orientations 2..8: an EXIF block behind two `Exif\\0\\0` prefixes, which Pillow strips
    (JPEG APP1, PNG eXIf, WebP EXIF); and EXIF Pillow cannot parse beside an XMP orientation (JPEG and PNG), where
    getexif raises before it reads the XMP and exif_transpose leaves the image as stored, as vLLM serves it."""
    img = pattern(40, 20)
    for o in range(2, 9):
        tiff = exif(o)[6:]
        doubled = [encoded(img, "JPEG", exif=b"Exif\x00\x00" * 2 + tiff),
                   encoded(img, "PNG", exif=b"Exif\x00\x00" * 2 + tiff),  # Pillow writes the eXIf chunk one prefix off
                   encoded(img, "WEBP", exif=b"Exif\x00\x00" * 3 + tiff, lossless=True)]  # and the WebP chunk
        for data in doubled:
            assert Image.open(io.BytesIO(data)).info["exif"] == b"Exif\x00\x00" * 2 + tiff
            assert agrees_with_pillow(data) == o
        info = PngImagePlugin.PngInfo()
        info.add_itxt("XML:com.adobe.xmp", xmp_packet(o).decode())
        junk = [encoded(img, "JPEG", exif=b"Exif\x00\x00not a tiff header", xmp=xmp_packet(o)),
                encoded(img, "PNG", exif=b"not a tiff header", pnginfo=info)]
        for data in junk:
            assert agrees_with_pillow(data) == 1


def test_exif_without_an_orientation_falls_back_to_xmp_and_exif_pillow_cannot_parse_is_1() -> None:
    """An EXIF block beside an XMP orientation of 6, in JPEG, PNG and WebP, read as Pillow's getexif reads it: IFD0's
    orientation when it has one (a TIFF header of either byte order of 42, or a big-endian 43, which Pillow reads as
    classic TIFF); the XMP's when Pillow finds none (the block empty once its prefixes are stripped, no Orientation
    tag, IFD0 or the Orientation's data past the block's end, an entry cut short); 1 when Pillow cannot parse it
    (shorter than a TIFF header, another byte order, another magic number, a little-endian BigTIFF, a third prefix):
    getexif raises before it reads the XMP, and the image is served as stored."""
    p, o3, o3be = b"Exif\x00\x00", orientation_entry(3, [3]), orientation_entry(3, [3], big=True)
    cases = [(p + o3, 3), (p + o3[:2] + b"\x00\x2a" + o3[4:], 3), (p + o3be[:2] + b"\x2a\x00" + o3be[4:], 3),
             (p + o3be[:2] + b"\x00\x2b" + o3be[4:], 3),
             (p, 6), (p * 2, 6), (p + ifd0([(0x010F, 2, 4, b"abc\0")]), 6), (p + b"II*\x00" + struct.pack("<I", 99), 6),
             (p + orientation_entry(3, [3, 3, 3])[:26], 6), (p + ifd0([(0x010F, 2, 4, b"abc\0")] * 2)[:22], 6),
             (p + b"M", 1), (p + o3[:6], 1), (p + o3[:7], 1), (p + b"XY" + o3[2:], 1),
             (p + b"II\x2b\x00" + o3[4:], 1), (p + b"II\x2c\x00" + o3[4:], 1), (p * 3 + o3, 1)]
    img = pattern(40, 20)
    info = PngImagePlugin.PngInfo()
    info.add_itxt("XML:com.adobe.xmp", xmp_packet(6).decode())
    for block, want in cases:
        for data in (encoded(img, "JPEG", exif=block, xmp=xmp_packet(6)), encoded(img, "PNG", exif=block, pnginfo=info),
                     encoded(img, "WEBP", exif=p + block, xmp=xmp_packet(6), lossless=True)):
            assert agrees_with_pillow(data) == want, (data[:4], block)


def test_ifd0_ends_at_the_first_entry_whose_data_lies_past_the_block_as_in_pillow() -> None:
    """Pillow stops reading IFD0 at the first entry whose data lies past the block's end, whatever its tag: an
    Orientation after it is not read, so an XMP orientation counts, else 1. Data past Pillow's running bound of the
    block's length is skipped instead, and the Orientation after it is read. JPEG, PNG and WebP at orientations 2 to 8,
    against Pillow's orientation, size and pixels."""
    img = pattern(40, 20)
    p = b"Exif\x00\x00"
    for o in range(2, 9):
        x = 2 if o != 2 else 3
        past = ifd0([(0x010F, 2, 6, b"\x40\0\0\0"), (0x0112, 3, 1, bytes([o, 0, 0, 0]))])  # 6 bytes at 64 of 38
        skipped = ifd0([(0x010F, 2, 60, b"\x08\0\0\0"), (0x0112, 3, 1, bytes([o, 0, 0, 0]))])  # 60 > 38 in all
        info = PngImagePlugin.PngInfo()
        info.add_itxt("XML:com.adobe.xmp", xmp_packet(x).decode())
        for block, xmp, want in ((past, None, 1), (past, x, x), (skipped, x, o)):
            extra = {} if xmp is None else {"xmp": xmp_packet(xmp)}
            for data in (encoded(img, "JPEG", exif=p + block, **extra),
                         encoded(img, "PNG", exif=block, **({} if xmp is None else {"pnginfo": info})),
                         encoded(img, "WEBP", exif=p + block, lossless=True, **extra)):
                assert agrees_with_pillow(data) == want, (o, xmp, data[:4])


# The JPEG cases' file digests (the first 16 hex digits) and the pixels the decoder before Pillow gave them
# (torchvision's decode_jpeg, oriented by the probe), as sha256 of the HWC uint8 bytes, the same on the Mac and on linux
# x86_64
OLD_JPEG_DIGESTS: dict[str, tuple[str, str]] = {
    "JPEG-exif1": ("4c3561b45288bde1", "722eec33b8d1cca9e20ae1d15863626f5a2776001ec49aafde6362ad7767ee1b"),
    "JPEG-xmp1": ("d6a5a06b2fda5f92", "722eec33b8d1cca9e20ae1d15863626f5a2776001ec49aafde6362ad7767ee1b"),
    "JPEG-exif2": ("c4d981005d6606b8", "1a495426995a21d8cfd1971c1c5127fe217b1c7291edef5f4f501daf04ecab1e"),
    "JPEG-xmp2": ("a08c4064b60e0e42", "1a495426995a21d8cfd1971c1c5127fe217b1c7291edef5f4f501daf04ecab1e"),
    "JPEG-exif3": ("b483719663b5d8fb", "a953c7f20c2cd429f7c7a8f8cf2ba5d02db4b523c388b3c5da96f250f1981334"),
    "JPEG-xmp3": ("4bd7a7c489e90ec2", "a953c7f20c2cd429f7c7a8f8cf2ba5d02db4b523c388b3c5da96f250f1981334"),
    "JPEG-exif4": ("d7d031819823ddaa", "49928308216a0795df0d44a8f79257e4143a5cf16f3f902836c6a35f74b41f57"),
    "JPEG-xmp4": ("134f254298db682f", "49928308216a0795df0d44a8f79257e4143a5cf16f3f902836c6a35f74b41f57"),
    "JPEG-exif5": ("a46917b17a5609d8", "1834a80c73d59fc40857863ab39a3b8e66312b3f175d5e3245093f74b0f19c4f"),
    "JPEG-xmp5": ("6a7343f5faa49727", "1834a80c73d59fc40857863ab39a3b8e66312b3f175d5e3245093f74b0f19c4f"),
    "JPEG-exif6": ("6e99c44c6fbc132a", "28103de473048bb7cf971579ef192ef0ae159c1ce01adc9532861f2606524a04"),
    "JPEG-xmp6": ("aeecd5f3f04b3a43", "28103de473048bb7cf971579ef192ef0ae159c1ce01adc9532861f2606524a04"),
    "JPEG-exif7": ("5780cc73e47cd49e", "0cbbd575e596bacff23e372f946ddfef55b6729b8962fadc65c4ee1157bfa6d9"),
    "JPEG-xmp7": ("2df36ad97103e583", "0cbbd575e596bacff23e372f946ddfef55b6729b8962fadc65c4ee1157bfa6d9"),
    "JPEG-exif8": ("4add62fe11a2d388", "3c69c752f57b1a0ef1f46e582d3877d600b7782d062acdb55c6d115840bc96ae"),
    "JPEG-xmp8": ("62aaa3773c384dba", "3c69c752f57b1a0ef1f46e582d3877d600b7782d062acdb55c6d115840bc96ae"),
    "progressive": ("8bafc054dd7dd2d4", "e017eb4085ff7d232185e8c244ee122d6a583e3f50db393126d326b84089e7a1"),
    "cmyk": ("ed4cf5c84091c2d4", "4c591f70eb44d1c2683a9028b774536a9759258113400eb68a92aeb7d1002919"),
    "gray": ("3e7eb6e9119aeef7", "54e0b7a6f2281daf9ed24cdc579757dd0a31cc64a01b6227542a8c4eae038fa5"),
    "subsampling-444": ("18f92846541cf6f4", "38c551b17e8764096409b59b9e0bd962b1d017315512da8647b1efe6df1c96db"),
    "subsampling-422": ("b2de83491fb3563d", "5fafca7b7ea5a90bef92f6318459af5d13e8a62453350178f4fc205decbd9f1a"),
    "subsampling-420": ("5d7a0ea84f0823b0", "7f437eefe7202e07f45c0b79b6a58539eb5d814e5e9f9af18a21cd2f825fc9f7"),
}


def old_jpeg_cases() -> dict[str, bytes]:
    img = pattern(40, 30)
    cases = {}
    for o in range(1, 9):
        cases[f"JPEG-exif{o}"] = encoded(img, "JPEG", exif=exif(o, big=o % 2 == 0, typ=3 if o < 5 else 4))
        cases[f"JPEG-xmp{o}"] = encoded(img, "JPEG", xmp=xmp_packet(o))
    cases["progressive"] = encoded(pattern(64, 48, 1), "JPEG", progressive=True)
    cases["cmyk"] = encoded(pattern(50, 40, 2).convert("CMYK"), "JPEG")
    cases["gray"] = encoded(pattern(50, 40, 3).convert("L"), "JPEG")
    for ss, name in ((0, "444"), (1, "422"), (2, "420")):
        cases[f"subsampling-{name}"] = encoded(pattern(97, 61, 4), "JPEG", subsampling=ss, quality=85)
    return cases


def test_jpeg_decodes_to_the_old_decoders_pixels() -> None:
    """Pillow's decode of a JPEG gives the pixels the decoder before it gave: EXIF and XMP orientations 1 to 8,
    progressive, CMYK, grayscale and every chroma subsampling (both are libjpeg-turbo)."""
    import hashlib

    from llm_mini_infer.vision.family import decode_rgb

    for name, data in old_jpeg_cases().items():
        file, pixels = OLD_JPEG_DIGESTS[name]
        assert hashlib.sha256(data).hexdigest()[:16] == file, f"{name}: the encoder wrote other bytes"
        got = decode_rgb(data, probe(data)).permute(1, 2, 0).contiguous().numpy()
        assert hashlib.sha256(got.tobytes()).hexdigest() == pixels, name


def test_the_decode_packs_pillows_pixels_as_tobytes_does() -> None:
    """The decode's tensor is Pillow's own bytes, packed straight into one writable buffer: equal to `np.asarray` at
    1 x 1, a single row and column, odd sizes and rows wider than one of the encoder's chunks."""
    from llm_mini_infer.vision.family import _rgb

    rng = np.random.default_rng(5)
    for w, h in ((1, 1), (1, 7), (9, 1), (37, 23), (20000, 3), (3, 20000), (641, 479)):
        im = Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8), "RGB")
        got = _rgb(im, torch)
        assert got.shape == (h, w, 3) and np.array_equal(got.numpy(), np.asarray(im)), (w, h)
        got[0, 0, 0] = 7  # writable: its own buffer


def test_a_pillow_without_its_private_encoder_takes_the_public_path(monkeypatch) -> None:
    """`Image._getencoder` is Pillow's private name: without it (a `PIL.Image` that lacks only that name, as the
    decode imports it; Pillow's own calls keep theirs), the decode packs the same pixels through numpy, every JPEG case
    of the old decoder's and an odd size included."""
    import PIL

    from llm_mini_infer.vision.family import _rgb, decode_rgb

    class Without:
        def __getattr__(self, name: str):
            if name == "_getencoder":
                raise AttributeError(name)
            return getattr(Image, name)

    cases = list(old_jpeg_cases().values())
    want = [decode_rgb(data, probe(data)) for data in cases]
    im = Image.fromarray(np.random.default_rng(6).integers(0, 256, (23, 37, 3), dtype=np.uint8), "RGB")
    monkeypatch.setattr(PIL, "Image", Without())
    for data, pixels in zip(cases, want, strict=True):
        assert torch.equal(decode_rgb(data, probe(data)), pixels)
    got = _rgb(im, torch)
    assert np.array_equal(got.numpy(), np.asarray(im))
    got[0, 0, 0] = 7  # writable: its own buffer


def test_the_resize_is_torchvisions_call_for_call() -> None:
    """`resize_bicubic`, on torch alone, equals torchvision's uint8 antialiased bicubic resize (the dev group keeps
    torchvision for the reference tests): up and down, odd sizes, an unchanged size, on the decode's channels-last view
    and on a contiguous image."""
    import torchvision.transforms.v2.functional as tvF

    from llm_mini_infer.vision.family import resize_bicubic

    rng = random.Random(3)
    gen = torch.Generator().manual_seed(3)
    for k in range(200):
        h, w = rng.randint(1, 160), rng.randint(1, 160)
        img = torch.randint(0, 256, (h, w, 3), dtype=torch.uint8, generator=gen).permute(2, 0, 1)
        if k % 2:
            img = img.contiguous()
        th, tw = (h, w) if k % 10 == 0 else (rng.randint(1, 200), rng.randint(1, 200))
        want = tvF.resize(img.unsqueeze(0), [th, tw], interpolation=tvF.InterpolationMode.BICUBIC, antialias=True)
        got = resize_bicubic(img, th, tw)
        assert got.is_contiguous() and torch.equal(got, want[0]), (h, w, th, tw, k % 2)


def test_the_serving_code_imports_no_torchvision() -> None:
    """torchvision is a dev dependency only: in a child that refuses to import it, the vision package and the pool
    worker's entry import, and both families decode and resize a JPEG, a PNG and a WebP, without asking for it."""
    import subprocess
    import sys

    code = """
import sys

asked = []


class Refuse:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] == "torchvision":
            asked.append(name)
            raise ImportError(f"{name} is refused")


sys.meta_path.insert(0, Refuse())
from llm_mini_infer.synthetic import pattern
from llm_mini_infer.vision import family, gemma, pool, probe, qwen
from tests.support.vision_pool import GEMMA70, encoded, exif

fams = [GEMMA70, qwen.QwenVision(start_id=1, pad_id=2, end_id=3, patch=16, merge=2, temporal=2, min_pixels=65536,
                                 max_pixels=327680, mean=(0.5,) * 3, std=(0.5,) * 3, rescale=1 / 255, mrope=True)]
for fmt in ("JPEG", "PNG", "WEBP"):
    data = encoded(pattern(400, 225), fmt, exif=exif(6) if fmt == "JPEG" else exif(6)[6:])
    h = probe.probe(data)
    for f in fams:
        assert tuple(f.pixels(data, h, f.size(h)).shape)[0] == 3
assert pool.worker_main and not asked and not any(m.split(".")[0] == "torchvision" for m in sys.modules), asked
"""
    root = Path(__file__).resolve().parents[2]
    proc = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr[-3000:]


def test_a_truncated_or_corrupt_jpeg_is_a_400_with_pillows_words(tmp_path) -> None:
    """JPEG decodes through Pillow, as PNG and WebP do: a truncated file and one with a broken Huffman table fail
    their request with a 400 that names Pillow's error; the worker stays up and decodes the next image."""
    import base64
    import shutil

    from llm_mini_infer.api.images import ImageInputs
    from llm_mini_infer.api.protocol import ApiError
    from llm_mini_infer.engine import ipc as engine_ipc

    gemma = tmp_path / "gemma"  # a Gemma 4 checkpoint's config.json alone
    gemma.mkdir()
    (gemma / "config.json").write_text(json.dumps({"model_type": "gemma4", "vision_config": {
        "patch_size": 16, "pooling_kernel_size": 3, "default_output_length": 70}}))
    jpg = encoded(pattern(400, 225), "JPEG", quality=90)
    dht = jpg.index(b"\xff\xc4")
    bad = {"truncated": jpg[:len(jpg) // 2], "a broken Huffman table": jpg[:dht + 5] + b"\xff" * 16 + jpg[dht + 21:]}
    words = {"truncated": r"image: OSError: image file is truncated \(\d+ bytes not processed\)",
             "a broken Huffman table": r"image: OSError: broken data stream when reading image file"}

    def uri(data: bytes) -> str:
        return "data:image/jpeg;base64," + base64.b64encode(data).decode()

    ipc = engine_ipc.ipc_dir()
    arena = str(tmp_path / "arena")
    px_arena.create(arena, 1 << 22)
    proc = None

    async def main() -> None:
        images = ImageInputs(GEMMA70, index=0, num_frontends=1, arena_bytes=1 << 22, pool_dir=ipc, workers=1,
                             max_bytes=1 << 20, fetch_timeout_s=1.0, max_per_request=1)
        images.start()
        try:
            for name, data in bad.items():
                prep = await images.prepare([uri(data)])
                with pytest.raises(ApiError) as err:
                    await asyncio.wait_for(images.wait(prep), 60)
                assert err.value.status == 400 and re.fullmatch(words[name], err.value.message), err.value.message
                images.discard(prep)
            prep = await images.prepare([uri(jpg)])
            await asyncio.wait_for(images.wait(prep), 60)
            images.discard(prep)
        finally:
            await images.close()

    try:
        proc, ready = spawn_pool_worker(gemma, None, None, ipc, arena)
        assert ready, f"the worker exited before it was ready (exit code {proc.exitcode})"
        asyncio.run(main())
        assert proc.is_alive()
    finally:
        if proc is not None:
            proc.kill()
            proc.join(10)
        px_arena.remove(arena)
        shutil.rmtree(ipc, ignore_errors=True)


def test_an_image_past_pillows_pixel_limit_warns_and_twice_it_fails(monkeypatch) -> None:
    """Every format decodes through Pillow, JPEG too: a stored size above Image.MAX_IMAGE_PIXELS decodes with
    Pillow's DecompressionBombWarning, and above twice it is Pillow's DecompressionBombError (a 400 from the pool).
    images.max_source_pixels is at most twice Pillow's default, so the probe refuses such an image first."""
    from llm_mini_infer.vision.family import decode_rgb
    from llm_mini_infer.vision.probe import MAX_SOURCE_PIXELS

    assert MAX_SOURCE_PIXELS == 2 * Image.MAX_IMAGE_PIXELS
    img = pattern(40, 20)
    for fmt in ("JPEG", "PNG", "WEBP"):
        data = encoded(img, fmt)
        monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 40 * 20 - 1)
        with pytest.warns(Image.DecompressionBombWarning):
            assert tuple(decode_rgb(data, probe(data)).shape) == (3, 20, 40)
        monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 40 * 20 // 2 - 1)
        with pytest.raises(Image.DecompressionBombError):
            decode_rgb(data, probe(data))


# -- transparency: vLLM's composite onto white ------------------------------------------------------------------------


def png16(w: int, h: int, ctype: int, samples: np.ndarray, trns: bytes | None = None) -> bytes:
    """A 16-bit PNG of colour type `ctype` (Pillow writes none): `samples` uint16 [h, w, channels]."""
    import zlib

    def chunk(kind: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))

    rows = b"".join(b"\x00" + samples[y].astype(">u2").tobytes() for y in range(h))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 16, ctype, 0, 0, 0)) +
            (chunk(b"tRNS", trns) if trns is not None else b"") + chunk(b"IDAT", zlib.compress(rows)) +
            chunk(b"IEND", b""))


def bmp_rgba(rgba: np.ndarray) -> bytes:
    """A top-down 32-bit BMP whose BITFIELDS masks include alpha (BITMAPV4HEADER): Pillow decodes it as RGBA, and the
    32-bit BMP it writes as RGB."""
    h, w = rgba.shape[:2]
    px = rgba[..., [2, 1, 0, 3]].tobytes()
    v4 = struct.pack("<IiiHHIIiiII", 108, w, -h, 1, 32, 3, len(px), 2835, 2835, 0, 0)
    v4 += struct.pack("<IIII", 0xFF0000, 0xFF00, 0xFF, 0xFF000000) + b"BGRs" + bytes(48)
    return b"BM" + struct.pack("<IHHI", 14 + len(v4) + len(px), 0, 0, 14 + len(v4)) + v4 + px


def transparent_images() -> list[tuple[str, bytes]]:
    """Each kind of transparency Pillow decodes from the probe's formats (48 x 32 random pixels, an alpha that takes
    every value), and the same pixels without it (`plain-`)."""
    rng = np.random.default_rng(30)
    w, h = 48, 32
    rgb = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    a = (np.arange(w * h) % 256).astype(np.uint8).reshape(h, w)
    rgba = Image.frombytes("RGBA", (w, h), np.dstack([rgb, a]).tobytes())
    rgb_im = Image.frombytes("RGB", (w, h), rgb.tobytes())
    la = Image.frombytes("LA", (w, h), np.dstack([rgb[..., 0], a]).tobytes())
    gray = Image.frombytes("L", (w, h), rgb[..., 0].tobytes())
    one = gray.point(lambda v: 255 if v > 127 else 0).convert("1")
    i16 = rng.integers(0, 512, (h, w)).astype(np.uint16)  # above 255 converts to 255
    gray16 = Image.frombytes("I;16", (w, h), i16.astype("<u2").tobytes())
    idx = rng.integers(0, 256, (h, w), dtype=np.uint8)
    idx.flat[:256] = np.arange(256)
    p = Image.frombytes("P", (w, h), idx.tobytes())
    p.putpalette(rng.integers(0, 256, 768, dtype=np.uint8).tobytes())
    alphas = rng.integers(0, 256, 256, dtype=np.uint8)
    alphas[:4] = (0, 255, 0, 128)
    gray_p = Image.frombytes("P", (w, h), idx.tobytes())
    gray_p.putpalette(bytes(v for i in range(256) for v in (i, i, i)))  # a GIF decodes it as L
    p2 = Image.frombytes("P", (w, h), np.roll(idx, 7).tobytes())
    p2.putpalette(p.getpalette())
    s16 = rng.integers(0, 65536, (h, w, 4)).astype(np.uint16)
    s16[..., 3] = (np.arange(w * h) * 257 % 65536).reshape(h, w)
    return [
        ("png-rgba", encoded(rgba, "PNG")),
        ("png-rgba-exif6", encoded(rgba, "PNG", exif=exif(6))),
        ("png-rgba16", png16(w, h, 6, s16)),
        ("png-la", encoded(la, "PNG")),
        ("png-la16", png16(w, h, 4, s16[..., [0, 3]])),  # Pillow: RGBA
        ("png-p-alphas", encoded(p, "PNG", transparency=alphas.tobytes())),
        ("png-p-alphas-exif8", encoded(p, "PNG", transparency=alphas.tobytes(), exif=exif(8))),
        ("png-p-short-alphas", encoded(p, "PNG", transparency=alphas[:10].tobytes())),
        ("png-p-index", encoded(p, "PNG", transparency=3)),
        ("png-l-trns", encoded(gray, "PNG", transparency=int(rgb[0, 0, 0]))),
        ("png-1-trns", encoded(one, "PNG", transparency=0)),
        ("png-i16-trns", encoded(gray16, "PNG", transparency=int(i16[i16 <= 255][0]))),
        ("png-i16-trns-above-255", encoded(gray16, "PNG", transparency=int(i16[i16 > 255][0]))),
        ("png-rgb-trns", encoded(rgb_im, "PNG", transparency=tuple(int(v) for v in rgb[0, 0]))),  # its key composited
        ("png-rgb16-trns", png16(w, h, 2, s16[..., :3], struct.pack(">3H", *(int(v) for v in s16[0, 0, :3])))),
        ("webp-rgba", encoded(rgba, "WEBP", lossless=True)),
        ("webp-rgba-exif3", encoded(rgba, "WEBP", lossless=True, exif=exif(3))),
        ("webp-rgba-animated", encoded(rgba, "WEBP", lossless=True, save_all=True,
                                       append_images=[rgba.transpose(Image.Transpose.ROTATE_180)])),
        ("gif-index", encoded(p, "GIF", transparency=3)),
        ("gif-gray-index", encoded(gray_p, "GIF", transparency=9)),
        ("gif-animated", encoded(p, "GIF", transparency=3, save_all=True, append_images=[p2])),  # the first frame
        ("bmp-rgba", bmp_rgba(np.dstack([rgb, a]))),
        ("plain-png-rgb", encoded(rgb_im, "PNG")),
        ("plain-png-rgb16", png16(w, h, 2, s16[..., :3])),
        ("plain-png-p", encoded(p, "PNG")),
        ("plain-png-l", encoded(gray, "PNG")),
        ("plain-png-i16", encoded(gray16, "PNG")),
        ("plain-webp-rgb", encoded(rgb_im, "WEBP", lossless=True)),
        ("plain-webp-opaque-rgba", encoded(rgb_im.convert("RGBA"), "WEBP", lossless=True)),  # libwebp drops the alpha
        ("plain-gif", encoded(p, "GIF")),
        ("plain-bmp-rgb", encoded(rgb_im, "BMP")),
        ("plain-bmp-rgba-pillow", encoded(rgba, "BMP")),
        ("plain-bmp-p", encoded(p, "BMP")),
    ]


# sha256 of the H x W x 3 pixels the reference vLLM's own code gives for each image above:
# `ImageMediaIO(image_mode="RGB").load_bytes(data).media`, with vllm/multimodal/image.py, media/image.py and
# media/base.py as the reference vLLM has them, on Pillow 13.0.0.
VLLM_DIGESTS: dict[str, str] = {
    "png-rgba": "8fafe86ea22a66955a46303cbb9712915b80dc1e9d6f1d87451235e773c84e03",
    "png-rgba-exif6": "4ab7b8e2cf889b80bb0fcb1c1e6e90e3a4091092778482aaf478c2e9fcea6be4",
    "png-rgba16": "ab82665c3101e91ca3cb43abac2c071cdd37f7248783189918710f887adfe185",
    "png-la": "42c1e8b531b43ecb7107b6f32142c6ab7755c7431a48d3ed1e764feeb13ac9ae",
    "png-la16": "cd8aaea51834573eb117b94a33ae677815425f7b4c5e098636c78b34c1f8d120",
    "png-p-alphas": "f435c9dbda0d84bfa77a083c431d82f8beefc6afb88af527fd65339db63efa15",
    "png-p-alphas-exif8": "50346f91820d21a00c49d0f810e07e2fff40b3802a37de32f25268bd27ea45bf",
    "png-p-short-alphas": "031a20df8d201aabaff2991ffb189f5131ea1a9ab734f1a7d6d7aab0839bb3e5",
    "png-p-index": "da5a3d2b5e6ec3a194484923b248bf36719353ab1fa1ca7eeebc87a50b2a1ed5",
    "png-l-trns": "42d7245a959a568a15a5ddb559b43168c4b31766ab7798f75756417d9443a907",
    "png-1-trns": "d397edef4cf4719aa6670603a4abe242d870f1ac33e619dc894b24dc1eb9b413",
    "png-i16-trns": "fd94d0ac9b25a8ddf24b375b5f783ecfcb8c81e97e16146d565c04e344b5cfe8",
    "png-i16-trns-above-255": "63aa2ccc4edb7856bd1f60a7eaf0c6f659c505bb01c1273cea2280c90299b82d",
    "png-rgb-trns": "83e8d35763c74a2374d5b719f824199aacdf239fe824ea79deca59f89225dde9",
    "png-rgb16-trns": "16ebaf7682c032e875197acc6c4efd149bed555c5c15ab86d48ec4f8ada6e909",
    "webp-rgba": "8fafe86ea22a66955a46303cbb9712915b80dc1e9d6f1d87451235e773c84e03",
    "webp-rgba-exif3": "a01bdac5ebd6a0586a09383a40c5acc37a6f26dfe7b0f41ef3242df886aed5ff",
    "webp-rgba-animated": "8fafe86ea22a66955a46303cbb9712915b80dc1e9d6f1d87451235e773c84e03",
    "gif-index": "da5a3d2b5e6ec3a194484923b248bf36719353ab1fa1ca7eeebc87a50b2a1ed5",
    "gif-gray-index": "851b68ce7ac4c9348f9614c16b867e8ba1d9d5cbb3ea4baf94e842ff095fddfd",
    "gif-animated": "da5a3d2b5e6ec3a194484923b248bf36719353ab1fa1ca7eeebc87a50b2a1ed5",
    "bmp-rgba": "8fafe86ea22a66955a46303cbb9712915b80dc1e9d6f1d87451235e773c84e03",
    "plain-png-rgb": "5229e29701c001aee3b4a59599c38c4d3a0ee4b898dd0007a806a56df06a5169",
    "plain-png-rgb16": "16ebaf7682c032e875197acc6c4efd149bed555c5c15ab86d48ec4f8ada6e909",
    "plain-png-p": "999e30cdcb4ff83f3492ae48eeafc9b5033991228789da4426f25466b22744ca",
    "plain-png-l": "c7c74ed8b0982b97fb50c003acbbeddbe15c70a5ee1b6005e05a30b2d3ec9f8c",
    "plain-png-i16": "885b16bafe834faaea62974bd7df5c9fae0c629d1910d5f567c0eb3183816427",
    "plain-webp-rgb": "5229e29701c001aee3b4a59599c38c4d3a0ee4b898dd0007a806a56df06a5169",
    "plain-webp-opaque-rgba": "5229e29701c001aee3b4a59599c38c4d3a0ee4b898dd0007a806a56df06a5169",
    "plain-gif": "999e30cdcb4ff83f3492ae48eeafc9b5033991228789da4426f25466b22744ca",
    "plain-bmp-rgb": "5229e29701c001aee3b4a59599c38c4d3a0ee4b898dd0007a806a56df06a5169",
    "plain-bmp-rgba-pillow": "5229e29701c001aee3b4a59599c38c4d3a0ee4b898dd0007a806a56df06a5169",
    "plain-bmp-p": "999e30cdcb4ff83f3492ae48eeafc9b5033991228789da4426f25466b22744ca",
}
# Transparency that vLLM leaves as stored: a 16-bit RGB image's tRNS colour, which Pillow never matches to a pixel
STORED = {"png-rgb16-trns"}


@pytest.mark.parametrize(("name", "data"), transparent_images(), ids=[v[0] for v in transparent_images()])
def test_transparency_is_composited_as_vllm_does(name, data) -> None:
    """The pool's decode gives vLLM's pixels: transparency pasted onto white through its alpha, after the
    orientation, an RGB image's tRNS colour included; without transparency (or a 16-bit key no pixel matches),
    Pillow's `convert("RGB")` as before."""
    import hashlib
    import warnings

    from llm_mini_infer.vision.family import decode_rgb

    got = decode_rgb(data, probe(data)).permute(1, 2, 0).contiguous().numpy()
    assert hashlib.sha256(got.tobytes()).hexdigest() == VLLM_DIGESTS[name]
    with warnings.catch_warnings():  # Pillow asks for RGBA when a palette's alpha is dropped
        warnings.simplefilter("ignore")
        dropped = np.asarray(ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB"))
    assert np.array_equal(got, dropped) == (name.startswith("plain-") or name in STORED)


def test_a_transparent_png_reaches_the_pool_composited(tmp_path) -> None:
    """Through the frontend's image step (`api.images.ImageInputs`, data URIs as a request gives them) and a real pool
    worker (Gemma 4 at 70 soft tokens): a transparent PNG's pixels in the arena are those of its composite onto white
    saved as an RGB PNG, not those of its pixels with the alpha dropped."""
    import base64
    import shutil

    from llm_mini_infer.api.images import ImageInputs
    from llm_mini_infer.engine import ipc as engine_ipc

    gemma = tmp_path / "gemma"  # a Gemma 4 checkpoint's config.json alone
    gemma.mkdir()
    (gemma / "config.json").write_text(json.dumps({"model_type": "gemma4", "vision_config": {
        "patch_size": 16, "pooling_kernel_size": 3, "default_output_length": 70}}))
    rgba = Image.frombytes("RGBA", (400, 225), np.random.default_rng(31).integers(0, 256, (225, 400, 4),
                                                                                  dtype=np.uint8).tobytes())
    white = Image.new("RGB", rgba.size, (255, 255, 255))
    white.paste(rgba, mask=rgba.getchannel("A"))
    uris = ["data:image/png;base64," + base64.b64encode(encoded(im, "PNG")).decode()
            for im in (rgba, white, rgba.convert("RGB"))]
    ipc = engine_ipc.ipc_dir()
    arena = str(tmp_path / "arena")
    px_arena.create(arena, 1 << 22)
    proc = None

    async def main() -> list[np.ndarray]:
        images = ImageInputs(GEMMA70, index=0, num_frontends=1, arena_bytes=1 << 22, pool_dir=ipc, workers=1,
                             max_bytes=1 << 20, fetch_timeout_s=1.0, max_per_request=3)
        images.start()
        try:
            prep = await images.prepare(uris)
            await asyncio.wait_for(images.wait(prep), 60)
            view = px_arena.map_readonly(arena)
            out = [np.frombuffer(view.buf, np.uint8, im.nbytes, im.offset).copy() for im in prep.images]
            view.close()
            images.discard(prep)
            return out
        finally:
            await images.close()

    try:
        proc, ready = spawn_pool_worker(gemma, None, None, ipc, arena)
        assert ready, f"the worker exited before it was ready (exit code {proc.exitcode})"
        transparent, composite, dropped = asyncio.run(main())
        assert np.array_equal(transparent, composite) and not np.array_equal(transparent, dropped)
    finally:
        if proc is not None:
            proc.kill()
            proc.join(10)
        px_arena.remove(arena)
        shutil.rmtree(ipc, ignore_errors=True)
