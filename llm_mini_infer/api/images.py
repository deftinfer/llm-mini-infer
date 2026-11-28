"""A frontend's image inputs (architecture.md, "Images"): fetch the bytes, probe the header, size the placeholders,
and have the image pool write the pixels into the arena and hash the image.

Per request (`App.chat`): `prepare` fetches every image concurrently, probes and sizes each from its header (the
stored size: the orientation is the pool's decode's, `vision.family`), allocates its pixels in this frontend's arena
partition and submits the pool job; rendering and tokenizing run while the pool decodes. `expand` turns the one
placeholder token per image of the tokenized prompt into the image's span (its token count does not depend on the
orientation), and `wait` collects the pool's results: whether the pool turned the image (its size and grid then turn)
and its digest (`vision.family.image_digest`, `span_key`), which the request's image references need. From the
moment the AddRequest is sent the engine owns the pixel ranges until it releases the request (`StepOutputs.released`);
before that, `discard` frees them (after the job finished, if it is still running).

Errors are `ApiError` 400s (`prepare`, `wait`): among them a header that declares more pixels than
`images.max_source_pixels`, refused right after the probe so nothing is allocated or decoded for it, a request whose
images would decode past `images.max_request_decode_bytes` (vLLM's per-request budget, charged at each header in
prompt order, before any decode), and an image whose pool worker stopped while decoding it (it may have caused that, so
a client or proxy should not retry it on another server). Downloads past `images.max_request_download_mb` are vLLM's
422. Two are 503s, which another server may serve: no arena space within the wait, and an image the pool could not
serve (the server stopping, or two workers lost on it: `vision.pool`).
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..engine.messages import ImageRef
from ..vision.arena import Allocator
from ..vision.family import ImageFamily, ImageSize, image_digest, span_key, transposed
from ..vision.fetch import DownloadBudget, DownloadBudgetError, Fetcher, FetchError, fetch_all, image_url
from ..vision.pool import CLOSED, UNSERVED, Done, Job, PoolClient, PoolState
from ..vision.probe import MAX_SOURCE_PIXELS, Header, ProbeError, check_source_pixels, probe
from .protocol import ApiError, sanitize, validation_error

clock = time.perf_counter_ns
SPACE_TIMEOUT_S = 30.0
DECODE_BYTES = 715_827_880  # images.max_request_decode_bytes' default
DOWNLOAD_MB = 256  # images.max_request_download_mb's default


@dataclass
class Image:
    header: Header
    size: ImageSize  # as stored until the pool's answer (`ImageInputs.wait`), then as oriented
    nbytes: int  # the uint8 pixels in the arena
    key: int = 0  # the span key and the full content hash, from the pool's answer (`ImageInputs.wait`)
    digest: bytes = b""
    offset: int = -1  # arena offset (-1: not allocated)
    job: asyncio.Future | None = None
    done: Done | None = None


@dataclass
class Prepared:
    images: list[Image] = field(default_factory=list)
    t_start: int = 0
    t_fetched: int = 0
    t_sized: int = 0
    t_pixels: int = 0  # all pool jobs done
    image_ns: int = 0  # time on the TTFT path spent on images: the fetch, and waiting for the pool after tokenizing

    def refs(self, offsets: list[int]) -> list[ImageRef]:
        return [ImageRef(key=im.key, offset=off, tokens=im.size.tokens, grid_t=im.size.grid[0],
                         grid_h=im.size.grid[1], grid_w=im.size.grid[2], height=im.size.height,
                         width=im.size.width, pixels=im.offset, digest=im.digest)
                for im, off in zip(self.images, offsets)]


class ImageInputs:
    def __init__(self, family: ImageFamily, *, index: int, num_frontends: int, arena_bytes: int, pool_dir: str,
                 workers: int, max_bytes: int, fetch_timeout_s: float, max_per_request: int, allow_urls: bool = False,
                 allowed_networks: list[str] | None = None, allowed_domains: list[str] | None = None,
                 max_source_pixels: int = MAX_SOURCE_PIXELS, pool_state: PoolState | None = None,
                 span_key_bits: int = 62, max_request_decode_bytes: int = DECODE_BYTES,
                 max_request_download_mb: int = DOWNLOAD_MB, spare: frozenset[int] = frozenset()):
        self.family = family
        self.span_key_bits = span_key_bits
        part = arena_bytes // max(1, num_frontends) // 4096 * 4096
        self.alloc = Allocator(index * part, part)
        self.pool = PoolClient(pool_dir, workers, pool_state, index) if pool_dir else None
        self.fetcher = Fetcher(max_bytes, fetch_timeout_s, allow_urls, allowed_networks or (), allowed_domains or (),
                               spare=spare)
        self.max_per_request = max_per_request
        self.max_source_pixels = max_source_pixels
        self.max_request_decode_bytes, self.max_request_download_mb = max_request_decode_bytes, max_request_download_mb
        self.owned: dict[int, list[Image]] = {}  # rid -> pixel ranges the engine holds until it releases them
        self._space: asyncio.Event | None = None

    def start(self) -> None:
        self._space = asyncio.Event()
        if self.pool is not None:
            self.pool.start()

    async def close(self) -> None:
        await self.fetcher.close()
        if self.pool is not None:
            self.pool.close()

    def pool_metrics(self, metrics: Any) -> None:
        """The pool's restarts and the workers killed at the job deadline (the launcher's counts, the same in every
        frontend), and this frontend's images that failed with their worker or were re-sent, for a /metrics scrape."""
        pool = self.pool
        st = pool.state
        if st is not None:
            metrics.pool_restarts.set_total(sum(st.epoch(i) for i in range(st.workers)))
            metrics.pool_hangs.set_total(st.hangs)
        metrics.pool_jobs_failed.set_total(pool.failed)
        metrics.pool_jobs_resent.set_total(pool.resent)

    # -- per request -------------------------------------------------------------------------------------------------

    @staticmethod
    def urls(messages: list[Any]) -> list[str]:
        """The image URLs of a request's messages, in prompt order."""
        out = []
        for m in messages:
            if isinstance(m.content, list):
                for part in m.content:
                    if getattr(part, "image_url", None) is not None:
                        try:
                            out.append(image_url(part.image_url))
                        except FetchError as exc:
                            raise ApiError(400, str(exc), "messages") from exc
        return out

    async def prepare(self, urls: list[str], submit: bool = True) -> Prepared:
        """Fetch, probe and size every image; with `submit`, allocate their pixels and start the pool jobs (without a
        pool, as in tests, hash them here instead: as stored)."""
        if len(urls) > self.max_per_request:  # vLLM's `--limit-mm-per-prompt` message (Gemma 4's own limit: none)
            raise validation_error(f"At most {self.max_per_request} image(s) may be provided in one prompt.", "image")
        prep = Prepared(t_start=clock())
        try:
            blobs = await fetch_all(self.fetcher, urls, DownloadBudget(self.max_request_download_mb))
        except DownloadBudgetError as exc:  # vLLM's fetch wrapper: a 422 naming the URL, the message sanitized
            msg = sanitize(f"Failed to fetch media from URL: {exc} (parameter=image_url, value={exc.url})")
            raise ApiError(422, msg, "image_url", "UnprocessableEntityError") from exc
        except FetchError as exc:
            raise ApiError(400, f"image: {exc}", "messages") from exc
        prep.t_fetched = clock()
        fam = self.family
        limit, used = self.max_request_decode_bytes, 0
        for k, data in enumerate(blobs):
            if k:  # each image's probe a turn of the loop of its own (each a bounded walk: `vision.probe`)
                await asyncio.sleep(0)
            try:
                header = probe(data)
                check_source_pixels(header, self.max_source_pixels)
            except (ProbeError, ValueError) as exc:
                raise ApiError(400, f"image: {exc}", "messages") from exc
            if limit > 0:  # vLLM's decode budget: 4 bytes a stored pixel, this image's included in the text
                used += header.width * header.height * 4
                if used > limit:
                    raise ApiError(400, f"Request decoded media exceeds images.max_request_decode_bytes={limit} "
                                        f"({used} bytes).")
            try:
                size = fam.size(header)
            except (ProbeError, ValueError) as exc:
                raise ApiError(400, f"image: {exc}", "messages") from exc
            prep.images.append(Image(header, size, 3 * size.height * size.width))
        prep.t_sized = clock()
        if submit and self.pool is None:
            for im, data in zip(prep.images, blobs):
                self._finish(im, image_digest(data, im.size))
        if submit and self.pool is not None:
            try:
                for im, data in zip(prep.images, blobs):
                    im.offset = await self._alloc(im.nbytes)
                    hd, sz = im.header, im.size
                    im.job = self.pool.submit(Job(jid=0, fmt=hd.format, width=hd.width, height=hd.height,
                                                  out_h=sz.height, out_w=sz.width, grid_t=sz.grid[0],
                                                  grid_h=sz.grid[1], grid_w=sz.grid[2], tokens=sz.tokens,
                                                  offset=im.offset), data)
            except BaseException:
                self.discard(prep)
                raise
        prep.image_ns = prep.t_fetched - prep.t_start
        return prep

    async def wait(self, prep: Prepared) -> None:
        """Until the pool wrote every image's pixels, each image's size then as oriented and its digest as the pool
        gave them. A failed decode is a 400. An image that the pool could not serve is a 503 (module docstring), unless
        another image of the request is a 400: an image that may have stopped a worker is never retried. The jobs are
        shielded: a cancelled request frees its pixels only once their jobs resolve (`discard`)."""
        t0 = clock()
        retry = ""
        for im in prep.images:
            if im.job is None:
                continue
            im.done = await asyncio.shield(im.job)
            if im.done.error in (CLOSED, UNSERVED):
                retry = retry or im.done.error
            elif im.done.error:
                raise ApiError(400, f"image: {im.done.error}", "messages")
            else:
                if im.done.transposed:
                    im.size = transposed(im.size)
                self._finish(im, im.done.digest)
        if retry:
            raise ApiError(503, f"image: {retry}", kind="ServiceUnavailableError")
        prep.t_pixels = clock()
        prep.image_ns += prep.t_pixels - t0

    def expand(self, ids: np.ndarray, prep: Prepared, hints: list[int] = ()) -> tuple[np.ndarray, list[int], list[int]]:
        """The prompt with each image's placeholder token expanded into the family's span prefix, `size.tokens`
        copies of the placeholder (the span) and its suffix (`vision.family`): (ids, the spans' offsets, the
        cache-hint offsets moved past the expansions before them)."""
        n = len(prep.images)
        fam = self.family
        pads = np.flatnonzero(ids == fam.pad_id)
        if len(pads) != n:
            raise ApiError(400, f"the prompt holds {len(pads)} image placeholders for {n} images (is the "
                                "placeholder token in the text?)", "messages")
        if not n:
            return ids, [], list(hints)
        pre, suf = np.asarray(fam.span_prefix, ids.dtype), np.asarray(fam.span_suffix, ids.dtype)
        sizes = np.asarray([im.size.tokens for im in prep.images], np.int64)
        counts = np.ones(len(ids), np.int64)
        counts[pads] = len(pre) + sizes + len(suf)
        starts = np.cumsum(counts) - counts  # where each token's expansion begins
        out = np.repeat(ids, counts)
        offsets = starts[pads] + len(pre)
        if len(pre) or len(suf):
            for off, k in zip(offsets.tolist(), sizes.tolist()):
                out[off - len(pre):off] = pre
                out[off + k:off + k + len(suf)] = suf
        ends = np.append(starts, len(out))
        return out, [int(o) for o in offsets], [int(ends[h]) for h in hints]

    def _finish(self, im: Image, digest: bytes) -> None:
        im.digest, im.key = digest, span_key(digest, self.span_key_bits)

    def discard(self, prep: Prepared) -> None:
        """Free the pixel ranges of a request that never reached the engine (once their jobs finished)."""
        for im in prep.images:
            self._free_when_done(im)
        prep.images = []

    def adopt(self, rid: int, prep: Prepared) -> None:
        """The request goes to the engine: its pixels are the engine's until it releases them."""
        if prep.images:
            self.owned[rid] = prep.images

    def release(self, rids: list[int]) -> None:
        for rid in rids:
            for im in self.owned.pop(rid, ()):
                self._free_when_done(im)

    # -- the arena partition -----------------------------------------------------------------------------------------

    def _free_when_done(self, im: Image) -> None:
        if im.offset < 0:
            return
        if im.job is not None and not im.job.done():
            im.job.add_done_callback(lambda _f, im=im: self._free(im))
        else:
            self._free(im)

    def _free(self, im: Image) -> None:
        if im.offset >= 0:
            self.alloc.free(im.offset, im.nbytes)
            im.offset = -1
            if self._space is not None:
                self._space.set()

    async def _alloc(self, n: int) -> int:
        if n > self.alloc.size:
            raise ApiError(400, f"the image's pixels ({n} bytes) exceed this frontend's arena ({self.alloc.size} "
                                "bytes; dev.images.arena_mb)", "messages")
        deadline = time.monotonic() + SPACE_TIMEOUT_S
        while (off := self.alloc.alloc(n)) is None:
            left = deadline - time.monotonic()
            if left <= 0:
                raise ApiError(503, "image memory is full; retry later (dev.images.arena_mb)",
                               kind="ServiceUnavailableError")
            self._space.clear()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._space.wait(), left)
        return off
