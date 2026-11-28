"""A large data URI's base64 decode off the event loop (`vision.fetch.Fetcher.fetch`, `OFF_LOOP_CHARS`): on the
fetcher's thread with pybase64, which lets the loop run meanwhile; inline below the threshold and with the standard
library's decode, which holds the GIL; the same bytes and the same errors either way; a body outside ASCII refused
before its decode; nothing left running after a cancelled fetch and the fetcher's close."""

from __future__ import annotations

import asyncio
import base64
import threading

import pytest

from llm_mini_infer.vision import fetch
from llm_mini_infer.vision.fetch import OFF_LOOP_CHARS, Fetcher, FetchError, from_data_uri


def uri(n: int) -> str:
    """A data URI of at least `n` characters with a valid base64 body."""
    body = base64.b64encode(bytes(range(256)) * (n // 340 + 1)).decode()
    return "data:image/png;base64," + body


def where(monkeypatch) -> list[str]:
    """The names of the threads that decode (`from_data_uri` wrapped)."""
    names: list[str] = []

    def recording(url: str, max_bytes: int) -> bytes:
        names.append(threading.current_thread().name)
        return from_data_uri(url, max_bytes)

    monkeypatch.setattr(fetch, "from_data_uri", recording)
    return names


async def fetch_one(fetcher: Fetcher, url: str) -> bytes:
    try:
        return await fetcher.fetch(url)
    finally:
        await fetcher.close()


@pytest.mark.skipif(fetch._b64.__name__ != "pybase64", reason="pybase64 is absent (the images extra)")
def test_a_large_data_uri_is_decoded_on_the_fetchers_thread_with_the_same_bytes_and_errors(monkeypatch) -> None:
    """Below `OFF_LOOP_CHARS` the decode runs on the loop's thread, from it on the fetcher's own; the bytes and every
    error's text are what the inline decode gives."""
    names = where(monkeypatch)
    small, large = uri(OFF_LOOP_CHARS - 2000), uri(OFF_LOOP_CHARS + 2000)
    assert len(small) < OFF_LOOP_CHARS <= len(large)
    for url in (small, large):
        assert asyncio.run(fetch_one(Fetcher(20 << 20, 10.0), url)) == from_data_uri(url, 20 << 20)
    assert names[0] == threading.current_thread().name and names[1].startswith("image-offload")
    bad = large[:-8] + "!!!!!!!!"  # not base64
    with pytest.raises(FetchError) as inline:
        from_data_uri(bad, 20 << 20)
    with pytest.raises(FetchError) as off:
        asyncio.run(fetch_one(Fetcher(20 << 20, 10.0), bad))
    assert str(off.value) == str(inline.value)
    with pytest.raises(FetchError, match="image larger than 1000 bytes"):
        asyncio.run(fetch_one(Fetcher(1000, 10.0), large))


def test_a_data_uri_outside_ascii_after_its_comma_is_refused_before_its_decode(monkeypatch) -> None:
    """A character outside ASCII after the ',' is never base64: the 400 comes before the decode (on a string that wide
    it holds the GIL, on the fetcher's thread too), with the standard library's text, after the size check as before.
    One in the header leaves the body to the decode, as before."""
    calls: list[int] = []
    monkeypatch.setattr(fetch, "_b64decode", lambda s: calls.append(len(s)) or base64.b64decode(s, validate=True))
    large = uri(OFF_LOOP_CHARS + 2000)
    comma = large.index(",")
    text = "invalid base64 in data URI: string argument should contain only ASCII characters"
    for ch in ("\U0001f600", "\u00e9", "\u3000"):
        for url in (large + ch, large[:comma + 8] + ch + large[comma + 8:]):
            with pytest.raises(FetchError) as exc:
                from_data_uri(url, 20 << 20)
            assert str(exc.value) == text
        with pytest.raises(FetchError, match="image larger than 1000 bytes"):
            from_data_uri(large + ch, 1000)
    assert calls == []
    with pytest.raises(ValueError) as std:
        base64.b64decode(large[comma + 1:] + "\U0001f600", validate=True)
    assert text.endswith(str(std.value))
    named = "data:image/png;name=\u00e9;base64," + large[comma + 1:]
    assert from_data_uri(named, 20 << 20) == from_data_uri(large, 20 << 20) and len(calls) == 2


def test_the_standard_librarys_decode_stays_on_the_loops_thread(monkeypatch) -> None:
    """Without pybase64 the decode holds the GIL, so a thread would hold the loop as long: it runs inline."""
    names = where(monkeypatch)
    monkeypatch.setattr(fetch, "_b64", base64)
    url = uri(OFF_LOOP_CHARS + 2000)
    assert asyncio.run(fetch_one(Fetcher(20 << 20, 10.0), url)) == from_data_uri(url, 20 << 20)
    assert names == [threading.current_thread().name]


@pytest.mark.skipif(fetch._b64.__name__ != "pybase64", reason="pybase64 is absent (the images extra)")
def test_a_cancelled_fetch_and_the_fetchers_close_leave_no_decode_running() -> None:
    """A request cancelled while its image is decoded ends with its cancellation; the fetcher's close shuts its thread
    down, and no decode thread is left once a decode in flight ends."""
    url = uri(16 << 20)

    async def run() -> None:
        fetcher = Fetcher(20 << 20, 10.0)
        task = asyncio.ensure_future(fetcher.fetch(url))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await fetcher.close()
        assert fetcher._decoder is None

    asyncio.run(run())
    deadline = threading.Event()
    for _ in range(200):  # the decode in flight ends on its own: milliseconds
        if not any(t.name.startswith("image-offload") for t in threading.enumerate()):
            break
        deadline.wait(0.05)
    assert not any(t.name.startswith("image-offload") for t in threading.enumerate())


@pytest.mark.skipif(fetch._b64.__name__ != "pybase64" or not hasattr(__import__("os"), "sched_setaffinity"),
                    reason="pybase64 is absent, or no thread affinity on this platform")
def test_the_decode_thread_runs_on_the_spare_cores(monkeypatch) -> None:
    """A frontend gives the fetcher the cores the engine and the frontends leave (`api.server`): its decode thread runs
    there, not on the event loop's core."""
    import os

    seen: list[set[int]] = []

    def recording(url: str, max_bytes: int) -> bytes:
        seen.append(os.sched_getaffinity(0))
        return from_data_uri(url, max_bytes)

    monkeypatch.setattr(fetch, "from_data_uri", recording)
    one = frozenset({min(os.sched_getaffinity(0))})
    url = uri(OFF_LOOP_CHARS + 2000)
    assert asyncio.run(fetch_one(Fetcher(20 << 20, 10.0, spare=one), url)) == from_data_uri(url, 20 << 20)
    assert seen == [set(one)]
