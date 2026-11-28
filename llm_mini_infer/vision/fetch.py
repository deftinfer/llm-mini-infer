# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vllm/connections.py of vLLM v0.33.0 (https://github.com/vllm-project/vllm), and changed for
# llm-mini-infer.
"""Image bytes from a chat request's `image_url`: base64 data URIs and, opt-in, http(s) URLs.

Limits (`images.*`): the decoded or downloaded size (`max_bytes`) and, for URLs, a time limit on the whole download,
redirects included (`fetch_timeout_s`); redirects are followed by hand, up to `MAX_REDIRECTS`. A request's downloads
share one budget (`DownloadBudget`, `max_request_download_mb`). Every failure is a `FetchError`, which the API turns
into a 400, or past the budget a `DownloadBudgetError`, its 422. URLs are fetched with aiohttp, one session per
frontend process, created on first use inside its event loop.

**URLs are off by default** (`images.allow_urls`). When on, the URL and every redirect's are checked before their
request, and a name's addresses when it is resolved (`urlguard`): loopback, private, link-local, metadata and reserved
addresses are refused, but for `images.allowed_networks`, and with `images.allowed_domains` set, every other name. A
refused URL's message is the same wherever it was refused and never names an address; other connection failures name
only their kind.
"""

from __future__ import annotations

import asyncio
import binascii
import contextlib
import logging
import os
from collections.abc import Iterable
from typing import Any

from .urlguard import REFUSED, Blocked, BlockedAddress, Policy, check_url, safe_resolver

try:  # pybase64 (the images extra) decodes faster, and lets other threads run meanwhile; the standard library is the
    # fallback, which holds the GIL throughout
    import pybase64 as _b64
except ImportError:  # pragma: no cover - depends on the environment
    import base64 as _b64
# From this many bytes (decoded), work that lets other threads run goes to the fetcher's thread, off the event loop
# (`Fetcher.offload`): a data URI's base64 decode (with pybase64 only: the standard library's holds the GIL, which would
# hold the loop as long) and an image's hash (`api.images`); below it the thread's hop (tens of microseconds) costs
# about what the work does
OFF_LOOP_BYTES = 256 << 10
OFF_LOOP_CHARS = OFF_LOOP_BYTES * 4 // 3  # the same, as a data URI's characters


def _b64decode(s: str) -> bytes:
    return _b64.b64decode(s, validate=True)

MAX_REDIRECTS = 5  # SGLang's
# The most characters a data URI may hold before its ',' (`data:image/png;base64` and any parameters): the frontend's
# event loop splits and checks them before it decodes anything
MAX_DATA_URI_HEAD = 1024
REDIRECTS = frozenset((301, 302, 303, 307, 308))
MIB = 1 << 20
log = logging.getLogger(__name__)


class FetchError(ValueError):
    """The image could not be read from the request (a 400 for the client)."""


class DownloadBudgetError(FetchError):
    """A request's downloads past its budget (`DownloadBudget`): vLLM's 422, which names the URL being fetched."""

    def __init__(self, message: str, url: str = ""):
        super().__init__(message)
        self.url = url


class DownloadBudget:
    """The http(s) bytes one request's image downloads receive (vLLM's `MediaRequestDownloadBudget`): each chunk of
    each download is charged as it arrives, and the one that takes the total past `limit_mb` MiB stops its download
    with vLLM's text, the setting named in place of VLLM_MAX_MEDIA_REQUEST_DOWNLOAD_SIZE_MB. 0: no limit."""

    def __init__(self, limit_mb: int):
        self.limit_mb, self.used = limit_mb, 0

    def charge(self, size: int) -> None:
        if self.limit_mb <= 0:
            return
        received = self.used + size
        if received > self.limit_mb * MIB:
            raise DownloadBudgetError("Request media downloads exceed images.max_request_download_mb="
                                      f"{self.limit_mb:g} ({received} bytes received, including retried attempts).")
        self.used = received


def _place(cpus: frozenset[int]) -> None:
    """The calling thread's cores (Linux), when there are any to give it."""
    if cpus:
        with contextlib.suppress(AttributeError, OSError):
            os.sched_setaffinity(0, cpus)


def image_url(part_value: Any) -> str:
    """The URL of an `image_url` content part: OpenAI's `{"url": ..., "detail": ...}`, or a bare string."""
    if isinstance(part_value, str):
        return part_value
    if isinstance(part_value, dict) and isinstance(part_value.get("url"), str):
        return part_value["url"]
    raise FetchError("'image_url' must be an object with a string 'url' (or a string)")


def from_data_uri(url: str, max_bytes: int) -> bytes:
    """`data:image/<type>;base64,<data>` → bytes."""
    comma = url.find(",")
    if comma < 0:
        raise FetchError("malformed data URI: no ','")
    if comma > MAX_DATA_URI_HEAD:
        raise FetchError(f"malformed data URI: more than {MAX_DATA_URI_HEAD} characters before its ','")
    head, body = url[:comma], url[comma + 1:]
    meta = head[5:].split(";")
    media = meta[0].strip().lower()
    if media and not media.startswith("image/"):
        raise FetchError(f"data URI media type {media!r} is not an image")
    if "base64" not in (m.strip().lower() for m in meta[1:]):
        raise FetchError("only base64 data URIs are supported for images")
    if len(body) * 3 // 4 > max_bytes + 3:
        raise FetchError(f"image larger than {max_bytes} bytes")
    try:
        data = _b64decode(body.strip())
    except (binascii.Error, ValueError) as exc:
        raise FetchError(f"invalid base64 in data URI: {exc}") from exc
    if len(data) > max_bytes:
        raise FetchError(f"image larger than {max_bytes} bytes")
    if not data:
        raise FetchError("empty image")
    return data


class Fetcher:
    """Per frontend process: resolves `image_url` values to bytes."""

    def __init__(self, max_bytes: int, timeout_s: float, allow_urls: bool = False,
                 allowed_networks: Iterable[str] = (), allowed_domains: Iterable[str] = (), resolver: Any = None,
                 spare: frozenset[int] = frozenset()):
        """`allowed_networks`, `allowed_domains`: the settings' (`urlguard.Policy`); `resolver`: the aiohttp resolver
        whose answers are checked (default: aiohttp's own; the tests' stub names). `spare`: the cores the decode thread
        runs on (`fetch`), off the event loop's own (empty: wherever the thread that starts it may run)."""
        self.max_bytes = max_bytes
        self.timeout_s = timeout_s
        self.allow_urls = allow_urls
        self.policy = Policy(allowed_networks, allowed_domains)
        self._resolver = resolver
        self._session: Any = None
        self._decoder: Any = None  # the off-loop thread (`offload`)
        self._spare = spare

    async def fetch(self, url: str, budget: DownloadBudget | None = None) -> bytes:
        """The image's bytes. `budget`: the request's, charged by an http(s) download (a data URI is not counted)."""
        if url.startswith("data:"):
            if len(url) < OFF_LOOP_CHARS or _b64.__name__ != "pybase64":
                return from_data_uri(url, self.max_bytes)
            return await self.offload(from_data_uri, url, self.max_bytes)
        low = url[:8].lower()
        if low.startswith("http://") or low.startswith("https://"):
            if not self.allow_urls:
                raise FetchError("image URLs are off on this server (images.allow_urls); send the image as a base64 "
                                 "data URI")
            try:
                return await self._http(url, budget)
            except DownloadBudgetError as exc:
                exc.url = url
                raise
        raise FetchError("image URLs must be data URIs" + (" or http(s) URLs" if self.allow_urls else ""))

    async def _http(self, url: str, budget: DownloadBudget | None = None) -> bytes:
        try:
            import aiohttp
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise FetchError("this server cannot fetch image URLs (aiohttp is not installed); send a data URI") \
                from exc
        from yarl import URL

        if self._session is None:
            # aiohttp connects only to the addresses the resolver returned (and keeps them 10 s), so the address
            # checked is the address connected to
            connector = aiohttp.TCPConnector(resolver=safe_resolver(self.policy, self._resolver))
            self._session = aiohttp.ClientSession(connector=connector,
                                                  timeout=aiohttp.ClientTimeout(total=self.timeout_s),
                                                  auto_decompress=True)
        cur = None
        try:
            cur = URL(url)
            # one deadline for every hop: aiohttp's own `total` restarts with each request
            async with asyncio.timeout(self.timeout_s):
                for hop in range(MAX_REDIRECTS + 1):
                    check_url(cur, self.policy)
                    async with self._session.get(cur, allow_redirects=False) as resp:
                        location = resp.headers.get(aiohttp.hdrs.LOCATION) if resp.status in REDIRECTS else None
                        if location is not None:  # its body unread: the connection is not reused
                            if hop == MAX_REDIRECTS:
                                raise FetchError(f"fetching the image took more than {MAX_REDIRECTS} redirects")
                            cur = cur.join(URL(location))
                            continue
                        return await self._body(resp, budget)
        except FetchError:
            raise
        except Blocked as exc:
            log.info("image URL refused: host %s: %s", exc.host, exc.reason)
            raise FetchError(REFUSED) from None
        # yarl's parse of the URL or a Location (a port, an IPv6 bracket), or its host's IDNA decoding (`xn--`): the log
        # names the raw host, which decodes nothing
        except (ValueError, UnicodeError) as exc:
            log.info("image URL failed: host %s: an invalid URL: %s", cur.raw_host if cur is not None else None,
                     type(exc).__name__)
            raise FetchError("fetching the image failed: an invalid URL") from None
        except TimeoutError as exc:
            raise FetchError(f"fetching the image took longer than {self.timeout_s:g} s") from exc
        except aiohttp.ClientConnectorDNSError as exc:
            if isinstance(exc.os_error, BlockedAddress):
                log.info("image URL refused: host %s: an address in its DNS answer is blocked", exc.os_error.host)
                raise FetchError(REFUSED) from None
            log.info("image URL failed: host %s: %s", cur.raw_host, exc.os_error)
            raise FetchError("fetching the image failed: its host did not resolve") from None
        except aiohttp.ClientError as exc:
            # the kind only: the text names hosts, ports and system errors, which tell a client about the network (the
            # log neither: some kinds' text holds the URL, whose query may hold a secret)
            log.info("image URL failed: host %s: %s %s", cur.raw_host, type(exc).__name__, getattr(exc, "os_error", ""))
            raise FetchError(f"fetching the image failed: {type(exc).__name__}") from None
        raise AssertionError("unreachable")  # the loop returns or raises

    async def _body(self, resp: Any, budget: DownloadBudget | None) -> bytes:
        if resp.status != 200:
            raise FetchError(f"fetching the image failed: HTTP {resp.status}")
        if resp.content_length is not None and resp.content_length > self.max_bytes:
            raise FetchError(f"image larger than {self.max_bytes} bytes")
        chunks, n = [], 0
        async for chunk in resp.content.iter_chunked(1 << 16):
            n += len(chunk)
            if n > self.max_bytes:
                raise FetchError(f"image larger than {self.max_bytes} bytes")
            if budget is not None:  # after the image's own bound, as vLLM checks them
                budget.charge(len(chunk))
            chunks.append(chunk)
        if not n:
            raise FetchError("empty image")
        return b"".join(chunks)

    def offload(self, fn: Any, *args: Any) -> asyncio.Future:
        """`fn(*args)` on the fetcher's one thread, off the event loop, on the spare cores (on the loop's own core it
        would take turns with the loop): for work that lets other threads run. A request cancelled meanwhile drops the
        result; the thread is started on first use."""
        if self._decoder is None:
            from concurrent.futures import ThreadPoolExecutor

            self._decoder = ThreadPoolExecutor(1, thread_name_prefix="image-base64", initializer=_place,
                                               initargs=(self._spare,))
        return asyncio.get_running_loop().run_in_executor(self._decoder, fn, *args)

    async def close(self) -> None:
        if self._decoder is not None:  # a decode still running ends on its own (milliseconds); a queued one never runs
            self._decoder.shutdown(wait=False, cancel_futures=True)
            self._decoder = None
        if self._session is not None:
            await self._session.close()
            self._session = None


async def fetch_all(fetcher: Fetcher, urls: list[str], budget: DownloadBudget | None = None) -> list[bytes]:
    """Every image of a request, concurrently, their downloads charged to `budget`. The first failure is raised, and
    the downloads still running are cancelled (and awaited) first, so none runs on to its own timeout; so are all of
    them if the caller is cancelled."""
    if len(urls) == 1:
        return [await fetcher.fetch(urls[0], budget)]
    tasks = [asyncio.ensure_future(fetcher.fetch(u, budget)) for u in urls]
    try:
        return list(await asyncio.gather(*tasks))
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
