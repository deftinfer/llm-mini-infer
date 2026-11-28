#!/usr/bin/env python3
"""A frontend's event loop beside image requests, on the fake runner: how late the frontend's timers run
(`/debug/loop`) and the gaps between the chunks of steady streams, with and without image requests; then each stage of
an image request's path timed in this process, and whether it lets other threads run. Results:
results/server/<time>-loop-<what>.json (or --out).

    uv run python bench/server/loop.py arms [--repeats 5] [--duration 15] [--files DIR] [--only none,12mp@4]
    uv run python bench/server/loop.py stages [--files DIR]
    uv run python bench/server/loop.py table results/server/<time>-loop-arms.json

**arms.** One server: the model fixture (`--model`), images on with `--workers` pool processes, one frontend, its
logging as by default (a line per request at INFO) into a pipe that a thread of this script reads. Beside every arm but
the `@1` ones, `--streams` streams of `--step-ms` fake steps, one chunk a step, each chunk's arrival recorded by client
processes. Per arm, over `--duration` seconds after `--warmup`: the lag of the frontend's timer (`/debug/loop`; p50,
p99, p99.9 and the maximum, ms), the streams' chunk gaps (the same), and for the requests sent beside them their TTFT,
the requests served a second and, for streamed image requests, the frontend's stage records (`/debug/stages`):
- `none`: the streams alone;
- `<image>@<rate>`: Poisson arrivals of image requests at `--rate` a second, each image a data URI: `phone` (a
  1280x960 JPEG with EXIF), `12mp` (a 4000x3000 JPEG with EXIF) and `png` (a 1920x1080 PNG in 8 KiB IDAT chunks, as
  libpng writes them); `<image>@max`: `--max-conc` clients sending them back to back, as many as the server takes;
- `<image>@1`: one client sending them back to back and no streams, for the TTFT of one image request alone;
- `file:<name>`, with `--files DIR`: each of the folder's images, one client back to back, the window held open
  until its last request ends (a crafted image's stall in full);
- `log`: text requests of one token at `--log-rate` a second, a log line each; `log-stalled`: the same with the
  pipe's reader stopped for `--stall-s` seconds from the window's start, then reading `--drain-bps` bytes a second.
The arms run `--repeats` times, interleaved, each repeat in a new order; `table` prints each arm's median over the
repeats and the spread (min to max) beside it. The `none` arm's spread is the noise floor of every figure.

**stages.** Each image (the three above, and `--files DIR`'s) through the frontend's steps in this process: the body
joined from 256 KiB reads, parsed (`decode_chat`), the data URI decoded (`vision.fetch.from_data_uri`), the header
probed (`vision.probe.probe`) and checked against `images.max_source_pixels`, the image sized (`ImageFamily.size`) and
hashed (BLAKE3 of the bytes and the size, as the frontend does): each one's median time over `--reps`, and the longest
that another thread waited for the GIL meanwhile, against the step's own time (a step that holds the GIL makes another
thread wait as long as it runs; one that releases it, about the switch interval at most).
"""

from __future__ import annotations

import argparse
import asyncio
import io
import itertools
import json
import multiprocessing as mp
import os
import random
import statistics
import struct
import sys
import threading
import time
import zlib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bench.server import run as base  # noqa: E402

KW = base.KW
IMAGES = ("phone", "12mp", "png")
MS = 1e6


# -- the images -------------------------------------------------------------------------------------------------------

def _photo(w: int, h: int, seed: int) -> Any:
    """A photo-like RGB image: gradients and fine noise, which JPEG at quality 90 stores at about a camera's size."""
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    base_ = np.stack([(x * 255 // w), (y * 255 // h), ((x + y) * 127 // (w + h))], -1)
    arr = (base_ + rng.integers(0, 24, (h, w, 3))) % 256
    return Image.fromarray(arr.astype(np.uint8))


def _exif(orientation: int) -> Any:
    from PIL import Image

    exif = Image.Exif()
    exif[0x010F], exif[0x0110], exif[0x0112] = "Maker", "Model 12", orientation
    exif[0x011A] = exif[0x011B] = 72.0
    exif[0x0131], exif[0x0132] = "Firmware 1.0", "2026:01:01 12:00:00"
    return exif


def _png_8k(data: bytes) -> bytes:
    """A PNG with its image data re-cut into 8 KiB IDAT chunks, as libpng writes them."""
    out, i, idat = [data[:8]], 8, []
    while i < len(data):
        n = struct.unpack_from(">I", data, i)[0]
        kind = data[i + 4:i + 8]
        if kind == b"IDAT":
            idat.append(data[i + 8:i + 8 + n])
        else:
            if idat and kind == b"IEND":
                z = b"".join(idat)
                for k in range(0, len(z), 8192):
                    part = z[k:k + 8192]
                    out.append(struct.pack(">I", len(part)) + b"IDAT" + part
                               + struct.pack(">I", zlib.crc32(b"IDAT" + part)))
            out.append(data[i:i + 12 + n])
        i += 12 + n
    return b"".join(out)


def image(name: str) -> tuple[bytes, str]:
    """One of `IMAGES`: (its bytes, its media type)."""
    buf = io.BytesIO()
    if name == "phone":
        _photo(1280, 960, 1).save(buf, "JPEG", quality=90, exif=_exif(6))
        return buf.getvalue(), "image/jpeg"
    if name == "12mp":
        _photo(4000, 3000, 2).save(buf, "JPEG", quality=90, exif=_exif(6))
        return buf.getvalue(), "image/jpeg"
    if name == "png":
        import numpy as np
        from PIL import Image

        rng = np.random.default_rng(3)
        arr = np.full((1080, 1920, 3), 245, np.uint8)  # a screenshot: flat panels, and lines of "text"
        for _ in range(12):
            y0, x0 = rng.integers(0, 900), rng.integers(0, 1600)
            arr[y0:y0 + 160, x0:x0 + 300] = rng.integers(0, 255, 3)
        rows = rng.integers(0, 1080, 400)
        arr[rows[:, None], rng.integers(0, 1920, (400, 600))] = 20
        Image.fromarray(arr).save(buf, "PNG")
        return _png_8k(buf.getvalue()), "image/png"
    raise ValueError(name)


def files(folder: Path | None) -> list[tuple[str, bytes, str]]:
    """The folder's images, by name: (name, bytes, media type from the first bytes)."""
    if folder is None:
        return []
    out = []
    for p in sorted(folder.iterdir()):
        if p.is_file():
            data = p.read_bytes()
            mime = ("image/png" if data[:4] == b"\x89PNG" else "image/webp" if data[8:12] == b"WEBP"
                    else "image/jpeg")
            out.append((p.name, data, mime))
    return out


def uri(data: bytes, mime: str) -> str:
    import base64

    return f"data:{mime};base64," + base64.b64encode(data).decode()


def image_body(url: str, i: int, max_tokens: int = 4) -> dict:
    return {"messages": [{"role": "system", "content": "You describe images."},
                         {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                                      {"type": "text", "text": f"Image {i}: what is it?"}]}],
            "max_tokens": max_tokens, "stream": True, "chat_template_kwargs": KW}


def text_body(i: int) -> dict:
    return {"messages": [{"role": "user", "content": f"Hello {i}!"}], "max_tokens": 1, "chat_template_kwargs": KW}


# -- clients (each in its own process) ----------------------------------------------------------------------------

async def _exchange(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, req: bytes,
                    events: list[int] | None = None) -> tuple[int, int, int, int]:
    """One request on a keep-alive connection: (sent, first event or 0, done, HTTP status); `events` gets each event's
    arrival (monotonic ns; the events of one read share it)."""
    t0 = time.monotonic_ns()
    writer.write(req)
    first = status = 0
    head, tail = b"", b""
    length = -1  # the body's length (-1: chunked), once the headers are in
    got = 0
    while True:
        data = await reader.read(1 << 18)
        now = time.monotonic_ns()
        if not data:
            raise ConnectionError("the server closed the connection")
        if not status:
            head += data
            top, sep, data = head.partition(b"\r\n\r\n")
            if not sep:
                continue
            status = int(top.split(b" ", 2)[1])
            for line in top.split(b"\r\n")[1:]:
                if line.lower().startswith(b"content-length:"):
                    length = int(line.split(b":")[1])
        if length >= 0:
            got += len(data)
            if got >= length:
                return t0, now, now, status
            continue
        chunk = tail + data
        n = chunk.count(b"data: ") - tail.count(b"data: ")
        if n and not first:
            first = now
        if events is not None:
            events.extend([now] * n)
        tail = chunk[-5:]
        if chunk.endswith(b"0\r\n\r\n"):
            return t0, first, now, status


def bodies(spec: tuple, max_tokens: int = 4) -> list[bytes]:
    """A request client's bodies: ("text",), ("image", name) or ("file", path); an image request's `max_tokens`."""
    if spec[0] == "text":
        return [base.request_bytes(text_body(i)) for i in range(256)]
    if spec[0] == "image":
        url = uri(*image(spec[1]))
        return [base.request_bytes(image_body(url, i, max_tokens)) for i in range(4)]
    path = Path(spec[1])
    (_, data, mime), = [f for f in files(path.parent) if f[0] == path.name]
    return [base.request_bytes(image_body(uri(data, mime), 0, max_tokens))]


def _streams(port: int, n: int, max_tokens: int, stop: Any, q: Any) -> None:
    """`n` connections, each streaming long requests until `stop` (an event) is set: each request's event arrivals."""
    import uvloop

    req = base.request_bytes({"messages": [{"role": "user", "content": "Count."}], "max_tokens": max_tokens,
                              "stream": True, "ignore_eos": True, "chat_template_kwargs": KW})

    async def one(out: list) -> None:
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        try:
            while not stop.is_set():
                events: list[int] = []
                out.append(events)
                task = asyncio.ensure_future(_exchange(reader, writer, req, events))
                while not task.done():
                    if stop.is_set():
                        task.cancel()
                        return
                    await asyncio.wait({task}, timeout=0.05)
                task.result()
        finally:
            writer.close()

    async def main() -> list:
        out: list = []
        await asyncio.gather(*(one(out) for _ in range(n)))
        return out

    q.put(("streams", uvloop.run(main())))


def _requests(port: int, spec: tuple, rate: float, conc: int, t_start: int, t_end: int, seed: int, q: Any,
              max_tokens: int = 4) -> None:
    """Requests from `t_start` until `t_end`: Poisson arrivals at `rate` a second (rate > 0, a new connection each), or
    `conc` keep-alive connections back to back; the bodies of `spec` in turn. Each request's (sent, first event, done,
    status)."""
    import uvloop

    reqs = bodies(spec, max_tokens)
    time.sleep(max(0.0, (t_start - time.monotonic_ns()) / 1e9))

    async def send(i: int, out: list) -> None:
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
        except OSError:
            out.append((time.monotonic_ns(), 0, 0, -1))
            return
        try:
            out.append(await _exchange(reader, writer, reqs[i % len(reqs)]))
        except (OSError, ConnectionError):
            out.append((time.monotonic_ns(), 0, 0, -1))
        finally:
            writer.close()

    async def poisson() -> list:
        out: list = []
        rng, tasks, i = random.Random(seed), [], 0
        nxt = time.monotonic_ns()
        while nxt < t_end:
            await asyncio.sleep(max(0.0, (nxt - time.monotonic_ns()) / 1e9))
            tasks.append(asyncio.ensure_future(send(i, out)))
            i += 1
            nxt += int(rng.expovariate(rate) * 1e9)
        await asyncio.gather(*tasks)
        return out

    async def closed() -> list:
        out: list = []

        async def client(c: int) -> None:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            i = c
            try:
                while time.monotonic_ns() < t_end:
                    out.append(await _exchange(reader, writer, reqs[i % len(reqs)]))
                    i += conc
            finally:
                writer.close()

        await asyncio.gather(*(client(c) for c in range(conc)))
        return out

    q.put(("requests", uvloop.run(poisson() if rate > 0 else closed())))


# -- the server's stderr --------------------------------------------------------------------------------------------

class Pipe:
    """The server's stderr: a pipe a thread drains into a file, or stops reading (`stall`) and then reads slowly."""

    def __init__(self, path: Path):
        self.r, self.w = os.pipe()
        self.out = path.open("wb")
        self.until = 0  # monotonic ns: no read before
        self.bps = 0.0  # bytes a second once reading again (0: at once)
        self.lines = 0
        threading.Thread(target=self._run, daemon=True).start()

    def stall(self, seconds: float, bps: float) -> None:
        self.until, self.bps = time.monotonic_ns() + int(seconds * 1e9), bps

    def drain(self) -> None:
        self.until, self.bps = 0, 0.0

    def _run(self) -> None:
        while True:
            now = time.monotonic_ns()
            if now < self.until:
                time.sleep(min(0.01, (self.until - now) / 1e9))
                continue
            n = 1 << 16 if not self.bps else max(1, int(self.bps * 0.05))
            data = os.read(self.r, n)
            if not data:
                return
            self.lines += data.count(b"\n")
            self.out.write(data)
            if self.bps:
                time.sleep(0.05)


# -- arms ----------------------------------------------------------------------------------------------------------

def pcts(xs: list[float]) -> dict[str, float] | None:
    if not xs:
        return None
    s = sorted(xs)

    def at(p: float) -> float:
        return s[min(len(s) - 1, int(p / 100 * len(s)))]

    return {"p50": at(50), "p99": at(99), "p99.9": at(99.9), "max": s[-1], "n": len(s)}


def held(lags: list[float], window: float, q: float) -> float:
    """The time-weighted quantile `q` of the loop's lag, ms: the wait of a callback that arrives at a random moment of
    the window. A sample of lag L ran L late, so a callback arriving in those L ms waited from L down to 0: the time it
    waits more than x is the sum of max(0, L - x), and the quantile is the x where that is (1 - q) of the window. (A
    stall of S seconds is one sample: the samples' own percentiles barely see it.)"""
    budget, s = (1 - q) * window, sorted(lags, reverse=True)
    total = 0.0
    for k, top in enumerate(s, 1):
        total += top
        nxt = s[k] if k < len(s) else 0.0
        if total - k * nxt >= budget:  # the quantile lies in [nxt, top]
            return max(0.0, (total - budget) / k)
    return 0.0


def lag_stats(loop: dict, t0: int, t1: int) -> dict | None:
    """The timer's lag over the window, ms (a sample before its time counts as 0): the samples' percentiles, the
    time-weighted ones (`held`), the share of the window a callback would wait more than 5 ms, and the largest
    samples."""
    lags = [max(0, lag) / MS for t, lag in zip(loop["t"], loop["lag"]) if t0 <= t <= t1]
    out = pcts(lags)
    if out is None:
        return None
    window = (t1 - t0) / MS
    out.update({"held_p99": held(lags, window, 0.99), "held_p99.9": held(lags, window, 0.999),
                "over_5ms_share": sum(max(0.0, x - 5) for x in lags) / window,
                "top": sorted(lags, reverse=True)[:200], "window_ms": window})
    return out


def gap_stats(per_request: list[list[int]], t0: int, t1: int, slow_ms: float) -> dict | None:
    """The gaps between consecutive chunks of one stream, both inside the window, ms; `late_share`: the share of the
    streams' time spent in gaps beyond `slow_ms` (a stall is one gap a stream, which the percentiles barely see)."""
    gaps = [(b - a) / MS for ev in per_request for a, b in itertools.pairwise(ev) if t0 <= a and b <= t1]
    out = pcts(gaps)
    if out is not None:
        out.update({"late_share": sum(max(0.0, g - slow_ms) for g in gaps) / max(sum(gaps), 1e-9),
                    "top": sorted(gaps, reverse=True)[:200]})
    return out


def stage_stats(recs: list[dict]) -> dict:
    """Per image request, the frontend's steps from its body to its tokens (`/debug/stages`), ms: they run on the
    loop with no await between them on these requests (`probe+size` holds the hash too where the frontend hashes)."""
    recs = [r for r in recs if r.get("images")]
    steps = {"parse": ("body", "arrival"), "base64": ("arrival", "fetched"), "probe+size": ("fetched", "sized"),
             "submit+render": ("sized", "rendered"), "tokenize": ("rendered", "tokenized"),
             "on_loop": ("body", "tokenized"), "to_pixels": ("sized", "pixels"),
             "pixels_to_first_byte": ("pixels", "first_byte")}
    return {k: pcts([(r[b] - r[a]) / MS for r in recs if a in r and b in r]) for k, (a, b) in steps.items()}


def arm_plan(args: argparse.Namespace, crafted: list) -> list[str]:
    arms = ["none", *(f"{im}@{args.rate:g}" for im in IMAGES), *(f"{im}@max" for im in IMAGES),
            *(f"{im}@1" for im in IMAGES), "log", "log-stalled"]
    arms += [f"file:{name}" for name, _, _ in crafted]
    if args.only:
        want = args.only.split(",")
        arms = [a for a in arms if a in want]
    return arms


def run_arm(srv: Any, pipe: Pipe, arm: str, args: argparse.Namespace, seed: int) -> dict:
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    now = time.monotonic_ns()
    t0 = now + int(args.warmup * 1e9)
    t1 = t0 + int(args.duration * 1e9)
    streams = not arm.endswith("@1")
    stop = ctx.Event()
    procs = []
    if streams:
        max_tokens = min(8000, int((args.warmup + args.duration + 5) * 1e3 / args.step_ms))
        per = [args.streams // args.stream_procs + (k < args.streams % args.stream_procs)
               for k in range(args.stream_procs)]
        procs += [ctx.Process(target=_streams, args=(srv.port, n, max_tokens, stop, q)) for n in per if n]
    load: tuple | None = None
    if arm != "none":
        name, _, how = arm.partition("@")
        if arm.startswith("log"):
            load = (("text",), args.log_rate, 0)
        elif arm.startswith("file:"):
            load = (("file", str(args.files / arm[5:])), 0.0, 1)
        else:
            load = (("image", name), args.rate if how not in ("max", "1") else 0.0,
                    args.max_conc if how == "max" else 1)
        # a second of warm-up before the window; a file's first request after the reset (its stall would hold the reset)
        start = t0 if arm.startswith("file:") else t0 - int(1e9)
        procs.append(ctx.Process(target=_requests, args=(srv.port, *load, start, t1, seed, q, args.image_tokens)))
    for p in procs:
        p.start()
    time.sleep(max(0.0, (t0 - int(0.2e9) - time.monotonic_ns()) / 1e9))
    srv.get("/debug/reset")
    time.sleep(max(0.0, (t0 - time.monotonic_ns()) / 1e9))
    if arm == "log-stalled":
        pipe.stall(args.stall_s, args.drain_bps)
    time.sleep(max(0.0, (t1 - time.monotonic_ns()) / 1e9))
    pipe.drain()
    results = []
    if load is not None:  # the requests in flight at the window's end finish (a crafted image's stall included)
        while not any(kind == "requests" for kind, _ in results):
            results.append(q.get())
    stop.set()
    t_end = max([t1, *(r[2] for kind, rs in results if kind == "requests" for r in rs)])
    _, loop = srv.get(f"/debug/loop?since={t0}")
    _, recs = srv.get("/debug/stages")
    results += [q.get() for _ in range(len(procs) - len(results))]
    for p in procs:
        p.join()
    out: dict[str, Any] = {"arm": arm, "lag_ms": lag_stats(loop, t0, t_end), "window_s": (t_end - t0) / 1e9}
    if streams:
        out["gap_ms"] = gap_stats([ev for kind, r in results if kind == "streams" for ev in r], t0, t_end,
                                  args.step_ms + 5)
    if load is not None:
        reqs = [r for kind, rs in results if kind == "requests" for r in rs if t0 <= r[0] <= t1]
        ok = [r for r in reqs if r[3] == 200]
        out["requests"] = {"sent": len(reqs), "ok": len(ok), "per_s": len(ok) / args.duration,
                           "ttft_ms": pcts([(r[1] - r[0]) / MS for r in ok if r[1]]),
                           "e2e_ms": pcts([(r[2] - r[0]) / MS for r in ok]),
                           "statuses": sorted({r[3] for r in reqs})}
        out["stages_ms"] = stage_stats(recs)
    return out


def bench_arms(args: argparse.Namespace) -> dict:
    crafted = files(args.files)
    arms = arm_plan(args, crafted)
    base.RESULTS.mkdir(parents=True, exist_ok=True)
    pipe = Pipe(base.RESULTS / f".loop-stderr-{os.getpid()}.log")
    extra = {"images": {"enabled": True, "workers": args.workers},
             "observability": {"log_level": "info", "log_requests": True},
             "server": {"num_frontends": 1}, "scheduler": {"max_num_seqs": max(64, args.streams + args.max_conc + 8)}}
    srv = base.Server(step_ms=args.step_ms, extra=extra, stderr=pipe.w)
    os.close(pipe.w)
    runs = []
    try:
        for rep in range(args.repeats):
            order = arms[rep % len(arms):] + arms[:rep % len(arms)]
            if rep % 2:
                order.reverse()
            for arm in order:
                r = run_arm(srv, pipe, arm, args, seed=rep)
                r["repeat"] = rep
                runs.append(r)
                lag, gap = r["lag_ms"] or {}, r.get("gap_ms") or {}
                print(f"[{rep}] {arm:14s} lag p99 {lag.get('p99', 0):7.2f} max {lag.get('max', 0):8.2f}   "
                      f"gap p99 {gap.get('p99', 0):7.2f} max {gap.get('max', 0):8.2f}"
                      + (f"   {r['requests']['per_s']:.1f}/s" if "requests" in r else ""), flush=True)
                time.sleep(1.0)
    finally:
        pipe.drain()
        srv.stop()
        pipe.out.flush()
    return {"what": "arms", "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "files": [n for n, _, _ in crafted], "stderr_lines": pipe.lines, "host": host(), "runs": runs}


# -- stages in-process ---------------------------------------------------------------------------------------------

def _gil_wait(fn: Any, reps: int) -> float:
    """The longest this thread waited to run while `fn` ran `reps` times in another thread, ms."""
    done, worst = threading.Event(), 0.0

    def work() -> None:
        for _ in range(reps):
            fn()
        done.set()

    th = threading.Thread(target=work)
    last = time.perf_counter_ns()
    th.start()  # the wait for it to start is a wait too
    while True:
        now = time.perf_counter_ns()
        worst, last = max(worst, (now - last) / MS), now
        if done.is_set():
            break
        time.sleep(0.0002)
    th.join()
    return worst


def steps_of(data: bytes, mime: str, fam: Any) -> tuple[dict[str, Any], str]:
    """The frontend's steps for one image, each a call to time, and the probe's outcome."""
    import blake3

    from llm_mini_infer.api.protocol import decode_chat
    from llm_mini_infer.vision.fetch import from_data_uri, image_url
    from llm_mini_infer.vision.probe import check_source_pixels, probe

    body = json.dumps(image_body(uri(data, mime), 0)).encode()
    reads = [body[k:k + (1 << 18)] for k in range(0, len(body), 1 << 18)]
    url = image_url(decode_chat(body).messages[1].content[0].image_url)

    def probe_once() -> Any:
        try:
            h = probe(data)
            check_source_pixels(h)
            return h
        except ValueError as exc:
            return exc

    steps: dict[str, Any] = {"join": lambda: b"".join(reads), "parse": lambda: decode_chat(body),
                             "base64": lambda: from_data_uri(url, 20 << 20), "probe": probe_once}
    header = probe_once()
    if isinstance(header, ValueError):
        return steps, f"{type(header).__name__}: {str(header)[:60]}"
    size = fam.size(header)

    def hash_once() -> bytes:
        h = blake3.blake3(data)
        h.update(struct.pack("<II", size.height, size.width))
        return h.digest()

    steps["size"] = lambda: fam.size(header)
    steps["hash"] = hash_once
    return steps, f"{header.format} {header.width}x{header.height}"


def bench_stages(args: argparse.Namespace) -> dict:
    from llm_mini_infer.vision import family as vf

    fam = vf.load(base.MODEL)
    out = []
    for name, data, mime in [(im, *image(im)) for im in IMAGES] + files(args.files):
        steps, outcome = steps_of(data, mime, fam)
        row: dict[str, Any] = {"name": name, "bytes": len(data), "header": outcome}
        for step, fn in steps.items():
            t = time.perf_counter_ns()
            fn()
            reps = args.reps if time.perf_counter_ns() - t < 1e9 else 1  # a step of seconds (a crafted image): once
            ts = []
            for _ in range(reps):
                t = time.perf_counter_ns()
                fn()
                ts.append((time.perf_counter_ns() - t) / MS)
            row[step] = {"ms": statistics.median(ts), "gil_wait_ms": _gil_wait(fn, max(1, min(reps, 5)))}
        out.append(row)
        print(f"{name[:40]:40s} {len(data):>9d} B " + " ".join(f"{s} {row[s]['ms']:.3f}/{row[s]['gil_wait_ms']:.2f}"
                                                            for s in steps), flush=True)
    return {"what": "stages", "switch_interval_ms": sys.getswitchinterval() * 1e3, "host": host(), "rows": out}


# -- the table ------------------------------------------------------------------------------------------------------

def table(report: dict) -> str:
    """Per arm, the median over the repeats of each figure, and its spread (min-max)."""
    by: dict[str, list[dict]] = {}
    for r in report["runs"]:
        by.setdefault(r["arm"], []).append(r)

    def cell(rs: list[dict], *path: str, fmt: str = ".2f") -> str:
        vs = []
        for r in rs:
            v: Any = r
            for k in path:
                v = v.get(k) if isinstance(v, dict) else None
            if isinstance(v, (int, float)):
                vs.append(v)
        if not vs:
            return "-"
        return f"{statistics.median(vs):{fmt}} ({min(vs):{fmt}}-{max(vs):{fmt}})"

    cols = [("lag p99", ("lag_ms", "p99")), ("lag held p99", ("lag_ms", "held_p99")), ("lag max", ("lag_ms", "max")),
            ("gap p50", ("gap_ms", "p50")), ("gap p99", ("gap_ms", "p99")), ("gap max", ("gap_ms", "max")),
            ("gap late %", ("gap_ms", "late_share")), ("TTFT p50", ("requests", "ttft_ms", "p50")),
            ("served/s", ("requests", "per_s")), ("on loop p50", ("stages_ms", "on_loop", "p50"))]
    lines = ["| arm | " + " | ".join(c for c, _ in cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for arm, rs in by.items():
        cells = [cell(rs, *path, fmt=".3%" if path[-1] == "late_share" else ".2f") for _, path in cols]
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def host() -> dict:
    import platform

    return {"python": platform.python_version(), "machine": platform.machine(), "cpus": os.cpu_count(),
            "processor": platform.processor() or None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("what", choices=["arms", "stages", "table"])
    ap.add_argument("report", nargs="?", type=Path, help="table: an arms report")
    ap.add_argument("--model", type=Path, default=base.MODEL, help="a model directory (default: %(default)s)")
    ap.add_argument("--files", type=Path, help="a folder of images: an arm each, and rows of stages")
    ap.add_argument("--only", help="arms: these arms only, comma-separated")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--duration", type=float, default=15.0, help="each arm's measured seconds")
    ap.add_argument("--warmup", type=float, default=3.0, help="seconds before each arm's window")
    ap.add_argument("--streams", type=int, default=16)
    ap.add_argument("--stream-procs", type=int, default=2, help="client processes for the streams")
    ap.add_argument("--step-ms", type=float, default=10.0, help="the fake runner's step")
    ap.add_argument("--rate", type=float, default=4.0, help="image requests a second in the <image>@<rate> arms")
    ap.add_argument("--max-conc", type=int, default=8, help="clients in the <image>@max arms")
    ap.add_argument("--workers", type=int, default=2, help="images.workers")
    ap.add_argument("--image-tokens", type=int, default=4,
                    help="an image request's max_tokens (1: each takes one fake step, so the pool or the loop bounds "
                         "the <image>@max arms)")
    ap.add_argument("--log-rate", type=float, default=100.0, help="text requests a second in the log arms")
    ap.add_argument("--stall-s", type=float, default=5.0, help="log-stalled: seconds the stderr reader stops")
    ap.add_argument("--drain-bps", type=float, default=8192.0, help="log-stalled: its reads afterwards, bytes/s")
    ap.add_argument("--reps", type=int, default=20, help="stages: timed runs of each step")
    ap.add_argument("--set", action="append", default=[], metavar="GROUP.KEY=VALUE",
                    help="arms: a server config override, e.g. --set observability.log_requests=false (repeatable)")
    ap.add_argument("--out", type=Path, help="write the JSON here instead of results/server/")
    args = ap.parse_args()
    if args.what == "table":
        print(table(json.loads(args.report.read_text())))
        return
    base.MODEL = args.model
    import yaml

    for item in args.set:  # as bench/server/run.py takes them: over the arms' own settings
        key, _, value = item.partition("=")
        node = base.EXTRA
        *parents, leaf = key.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = yaml.safe_load(value)
    report = bench_arms(args) if args.what == "arms" else bench_stages(args)
    out = args.out or base.RESULTS / f"{time.strftime('%Y%m%d-%H%M%S')}-loop-{args.what}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    if args.what == "arms":
        print(table(report))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
