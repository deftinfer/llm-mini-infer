#!/usr/bin/env python3
"""The server's own overhead, on the fake runner (or the CUDA runner on a GPU). Results:
results/server/<time>-<what>[-warm][-sleep].json (or --out); bench/server/README.md says what each measures.

    uv run python bench/server/run.py all                  # everything below
    uv run python bench/server/run.py ttft                 # TTFT overhead at concurrency 1, ~1,024-token chat
    uv run python bench/server/run.py decode --batch 64    # engine CPU per decode step
    uv run python bench/server/run.py idle                 # simulated GPU idle, 5 ms fake steps, concurrency 1/8/32
    uv run python bench/server/run.py sse                  # SSE throughput with one frontend
    python3 bench/server/run.py idle --runner cuda --model <hf dir>   # the CUDA runner on a GPU: device timing

The timings come from the server's own instrumentation (stage timestamps, per-step engine timing, the runner's device
timeline), read from its /debug endpoints; CPU use from psutil on the server's processes, and throttling from the
cgroup's cpu.stat. Load comes from separate client processes (raw asyncio HTTP/1.1, keep-alive).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import multiprocessing as mp
import os
import platform
import signal
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import psutil
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tests.support import paths  # noqa: E402

# --model overrides: any model directory with its tokenizer and configuration (the fake runner reads no weights)
MODEL = paths.MODELS_DIR / "gemma-4-31b-it"
RESULTS = paths.RESULTS / "server"
KW = {"enable_thinking": False}
RUNNER = "fake"  # --runner: "cuda" serves the model in --model on the GPU
EXTRA: dict = {}  # --set group.key=value overrides for every server
IMAGES: dict = {"on": False, "next": 0}  # --images: each classification request carries a new synthetic image
RESPONSE_FORMAT: dict | None = None  # --response-format: classification requests carry it


# -- server ------------------------------------------------------------------------------------------------------

def _merge(dst: dict, src: dict) -> None:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _merge(dst[k], v)
        else:
            dst[k] = v


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Server:
    def __init__(self, step_ms: float, max_num_seqs: int = 64, extra: dict | None = None, stderr: Any = None):
        """`stderr`: the server's (a file or descriptor, as `subprocess.Popen` takes it; default: this process's)."""
        self.port = free_port()
        fake = {"runner": {"fake": {"step_ms": step_ms, "strict": False}}} if RUNNER == "fake" else {}
        # speculative decoding off (a checkpoint with an MTP layer turns it on by default): each decode step then
        # yields one token a row, which the decode benchmark's window and the targets assume (--set turns it on)
        cfg = {"model": {"path": str(MODEL), "served_model_name": "bench", "max_model_len": 8192},
               "speculative": {"num_speculative_tokens": 0}, "runner": {"kind": RUNNER}, "dev": fake,
               "scheduler": {"max_num_seqs": max_num_seqs},
               "server": {"host": "127.0.0.1", "port": self.port},
               "observability": {"debug_endpoints": True, "log_level": "warning"}}
        _merge(cfg, extra or {})
        _merge(cfg, EXTRA)
        RESULTS.mkdir(parents=True, exist_ok=True)
        self.cfg_path = RESULTS / f".server-{self.port}.yaml"
        self.proc: subprocess.Popen | None = None
        try:
            self.cfg_path.write_text(yaml.safe_dump(cfg))
            # the bench's own server: no API key (an exported one would guard every path but the probes from the
            # bench's clients, which send none)
            env = {k: v for k, v in os.environ.items() if k not in ("LLM_MINI_INFER_API_KEY", "VLLM_API_KEY")}
            self.proc = subprocess.Popen([sys.executable, "-m", "llm_mini_infer.cli", "serve", "--config",
                                          str(self.cfg_path)],
                                         cwd=ROOT, env=env, stderr=stderr)
            deadline = time.monotonic() + (60 if RUNNER == "fake" else 1200)  # the GPU loads weights, profiles, JITs
            while time.monotonic() < deadline:
                if self.proc.poll() is not None:
                    raise RuntimeError(f"the server exited during startup (code {self.proc.returncode})")
                try:
                    if self.get("/health?ready=1")[0] == 200:
                        break
                except OSError:
                    pass
                time.sleep(0.1)
            else:
                raise RuntimeError("server did not start")
            _, info = self.get("/debug/engine")
            self.engine = psutil.Process(info["engine_pid"])
            # Every frontend (server.num_frontends): the launcher's spawned children but the engine and the image
            # pool's workers (its resource tracker is no spawned child)
            status, images = self.get("/debug/images")
            pool = set(images.get("pool_pids") or []) if status == 200 and isinstance(images, dict) else set()
            self.frontends = [p for p in self.engine.parent().children()
                              if p.pid != self.engine.pid and p.pid not in pool
                              and "spawn_main" in " ".join(p.cmdline())]
            want = int(cfg["server"].get("num_frontends") or 1)
            if len(self.frontends) != want:
                raise RuntimeError(f"found {len(self.frontends)} frontend processes, server.num_frontends is {want}")
            # The engine pins its (main) thread to one core on Linux (architecture.md, "The step loop"); this reads
            # that thread's mask.
            self.affinity = self.engine.cpu_affinity() if hasattr(self.engine, "cpu_affinity") else None
        except BaseException:
            self.stop()
            raise

    def get(self, path: str) -> tuple[int, Any]:
        import http.client

        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        try:
            c.request("GET", path)
            r = c.getresponse()
            body = r.read()
            return r.status, json.loads(body) if body[:1] in (b"{", b"[") else body
        finally:
            c.close()

    def cpu(self) -> dict[str, float]:
        out = {}
        for name, procs in (("engine", [self.engine]), ("frontend", self.frontends)):
            t = 0.0
            for p in procs:
                c = p.cpu_times()
                t += c.user + c.system
            out[name] = t
        return out

    def stop(self) -> None:
        """SIGTERM, then SIGKILL after 60 s; the child is reaped and the config file removed. Safe to call twice, and
        before the child exists (a failed start). The launcher's children are not in a new session: a killed launcher
        may leave its engine or frontends running."""
        try:
            if self.proc is not None and self.proc.poll() is None:
                self.proc.send_signal(signal.SIGTERM)
                try:
                    self.proc.wait(60)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    with contextlib.suppress(subprocess.TimeoutExpired):  # it raised already: keep that
                        self.proc.wait(10)
        finally:
            self.cfg_path.unlink(missing_ok=True)


# -- load ---------------------------------------------------------------------------------------------------------

def request_bytes(body: dict) -> bytes:
    data = json.dumps(body).encode()
    return (b"POST /v1/chat/completions HTTP/1.1\r\nHost: bench\r\nContent-Type: application/json\r\n"
            b"Content-Length: %d\r\n\r\n" % len(data)) + data


async def _stream(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, req: bytes) -> tuple[int, int, int, int]:
    t0 = time.perf_counter_ns()
    writer.write(req)
    first = events = 0
    tail = b""
    while True:
        data = await reader.read(1 << 18)
        if not data:
            raise ConnectionError("server closed the connection")
        chunk = tail + data
        if not first and b"data: " in chunk:
            first = time.perf_counter_ns()
        events += chunk.count(b"data: ")
        tail = chunk[-5:]
        if chunk.endswith(b"0\r\n\r\n"):
            return t0, first, time.perf_counter_ns(), events


def _client(port: int, bodies: list[dict], n_conn: int, duration: float, max_requests: int, q: Any) -> None:
    import uvloop

    reqs = [request_bytes(b) for b in bodies]

    async def conn(ci: int, out: list) -> None:
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        stop_at = time.monotonic() + duration
        i = ci
        n = 0
        while time.monotonic() < stop_at and (not max_requests or n < max_requests):
            out.append(await _stream(reader, writer, reqs[i % len(reqs)]))
            i += n_conn
            n += 1
        writer.close()

    async def main() -> list:
        out: list = []
        await asyncio.gather(*(conn(i, out) for i in range(n_conn)))
        return out

    q.put(uvloop.run(main()))


def run_load(port: int, bodies: list[dict], conc: int, duration: float, max_requests: int = 0,
             procs: int | None = None) -> list[tuple[int, int, int, int]]:
    """`conc` keep-alive connections spread over client processes; each sends requests back to back."""
    procs = procs or min(conc, 8)
    per = [conc // procs + (1 if i < conc % procs else 0) for i in range(procs)]
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    ps = []
    offset = 0
    for n in per:
        if not n:
            continue
        mine = bodies[offset:] + bodies[:offset]
        offset += n
        p = ctx.Process(target=_client, args=(port, mine, n, duration, max_requests, q))
        p.start()
        ps.append(p)
    out = []
    for _ in ps:
        out.extend(q.get())
    for p in ps:
        p.join()
    return out


# -- workloads ----------------------------------------------------------------------------------------------------

def classify_prompt(target_tokens: int = 1024) -> str:
    """The system prompt every classification request shares: rules added until it reaches `target_tokens` tokens."""
    from llm_mini_infer.render.renderer import Renderer

    tok = Renderer(MODEL).tokenizer
    labels = ["electronics", "clothing", "furniture", "toys", "groceries", "books", "garden", "sports", "beauty"]
    lines = ["You are a product classifier. Read the item and answer with exactly one label from the list below, "
             "followed by a confidence between 0 and 1. Follow every rule."]
    k = 0
    while len(tok.encode("\n".join(lines)).ids) < target_tokens:
        a, b = labels[k % len(labels)], labels[(k * 7 + 3) % len(labels)]
        lines.append(f"Rule {k + 1}: if the item description mentions features typical of {a}, but its packaging "
                     f"or intended use points to {b}, prefer {b} unless the title names {a} explicitly.")
        k += 1
    return "\n".join(lines)


def classify_bodies(n: int, max_tokens: int, stream: bool = True, offset: int = 0) -> tuple[list[dict], int]:
    """`n` classification requests, request i on item i (with --images, synthetic image `offset + i` before the item
    text), and the prompt's token count (text only: the image's span is not counted)."""
    from llm_mini_infer.render.renderer import Renderer

    system = classify_prompt()
    bodies = [{"messages": [{"role": "system", "content": system},
                            {"role": "user", "content": f"Item {i}: a {['red', 'blue', 'green'][i % 3]} widget, "
                                                        f"model {1000 + i}, ships in a cardboard box."}],
               "max_tokens": max_tokens, "stream": stream, "seed": i, "ignore_eos": True,
               "chat_template_kwargs": KW} for i in range(n)]
    r = Renderer(MODEL)
    text = r.render([{"role": m["role"], "content": m["content"]} for m in bodies[0]["messages"]], **KW)
    if RESPONSE_FORMAT is not None:
        for b in bodies:
            b["response_format"] = RESPONSE_FORMAT
    if IMAGES["on"]:
        import base64

        from llm_mini_infer.synthetic import image_jpeg

        for i, b in enumerate(bodies):
            item = b["messages"][1]["content"]
            url = "data:image/jpeg;base64," + base64.b64encode(image_jpeg(offset + i)).decode()
            b["messages"][1]["content"] = [{"type": "image_url", "image_url": {"url": url}},
                                           {"type": "text", "text": item}]
    return bodies, len(r.encode(text))


def pool_cpu(srv: Server) -> float | None:
    """CPU seconds (user + system) the image pool's workers have used so far."""
    try:
        pids = srv.get("/debug/images")[1].get("pool_pids") or []
        return sum(sum(psutil.Process(pid).cpu_times()[:2]) for pid in pids)
    except (OSError, psutil.Error, AttributeError):
        return None


def pct(xs: list[float], p: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, round(p / 100 * (len(s) - 1)))]


def summary(xs: list[float]) -> dict[str, float]:
    return {"p50": pct(xs, 50), "p90": pct(xs, 90), "p99": pct(xs, 99), "mean": statistics.fmean(xs), "n": len(xs)}


def bench_ttft(n: int = 400, warmup: int = 50, fake_wait: str = "spin", srv: Server | None = None) -> dict:
    """`srv`: a running server to reuse (ttft+idle: one model load for both)."""
    own = srv is None
    srv = srv or Server(step_ms=5.0, extra={"dev": {"runner": {"fake": {"wait": fake_wait}}}})
    try:
        srv.get("/debug/reset")
        bodies, prompt_tokens = classify_bodies(n + warmup, max_tokens=4, offset=IMAGES["next"])
        IMAGES["next"] += n + warmup
        client = run_load(srv.port, bodies, conc=1, duration=600, max_requests=n + warmup, procs=1)
        _, recs = srv.get("/debug/stages")
    finally:
        if own:
            srv.stop()
    recs = recs[warmup:]
    client = sorted(client)[warmup:]
    ms = 1e6
    stages = {
        "receive (the body)": [(r["body"] - r["recv"]) / ms for r in recs],
        "render: decode and validate": [(r["arrival"] - r["body"]) / ms for r in recs],
        "render: grammar, images, template": [(r["rendered"] - r["arrival"]) / ms for r in recs],
        "tokenize": [(r["tokenized"] - r["rendered"]) / ms for r in recs],
        "to_admit (DecodeStream prime, ZMQ, engine poll, admit)": [(r["admitted"] - r["tokenized"]) / ms for r in recs],
        "to_launch (plan, launch_forward)": [(r["launched"] - r["admitted"]) / ms for r in recs],
        "to_first_token minus runner time": [(r["first_token_out"] - r["launched"] - r["runner_ns"]) / ms
                                             for r in recs],
        "deliver (ZMQ, detokenize, SSE, first write)": [(r["first_byte"] - r["first_token_out"]) / ms for r in recs],
    }
    overhead = [(r["first_byte"] - r["body"] - r["runner_ns"] - r.get("image_ns", 0)) / ms for r in recs]
    runner = [r["runner_ns"] / ms for r in recs]
    client_ttft = [(c[1] - c[0]) / ms for c in client]
    out = {"runner": RUNNER, "fake_wait": fake_wait, "prompt_tokens": prompt_tokens,
           "engine_affinity": srv.affinity, "overhead_ms": summary(overhead),
           "stages_ms": {k: summary(v) for k, v in stages.items()}, "runner_ms": summary(runner),
           "client_ttft_ms": summary(client_ttft),
           "client_ttft_minus_runner_ms": summary([c - r for c, r in zip(client_ttft, runner)])}
    if IMAGES["on"]:
        wait = [max(0, r["pixels"] - r["tokenized"]) / ms for r in recs]
        out["images"] = {
            "overhead_ms is": "first byte - body received - device time of the request's steps (the ViT included) - "
                              "the image fetch (a data URI's base64 decode) - the wait for the pool after tokenizing",
            # the fetch counted too: an upper bound, its stage measured from the body's receipt
            "overhead_with_fetch_ms": summary([(r["first_byte"] - r["body"] - r["runner_ns"] - r["image_ns"]
                                                + r["fetched"] - r["body"]) / ms for r in recs]),
            "fetch_ms": summary([(r["fetched"] - r["recv"]) / ms for r in recs]),
            "probe_size_hash_ms": summary([(r["sized"] - r["fetched"]) / ms for r in recs]),
            "pool_wait_after_tokenize_ms": summary(wait),
            "pixels_ready_after_recv_ms": summary([(r["pixels"] - r["recv"]) / ms for r in recs])}
    return out


def throttled_us() -> int | None:
    """cgroup v2 CPU throttling so far (µs summed over CPUs), to tell host interference apart from engine cost."""
    try:
        for line in Path("/sys/fs/cgroup/cpu.stat").read_text().splitlines():
            if line.startswith("throttled_usec"):
                return int(line.split()[1])
    except OSError:
        pass
    return None


def stage_summary(recs: list[dict]) -> dict:
    """TTFT overhead (first byte − body received − device time of the request's steps − `image_ns`), runner time,
    `to_admit` and the overhead less `to_admit`, ms. Under load, `to_admit` also holds the wait for a free slot or
    budget (queueing, not CPU)."""
    ms = 1e6
    recs = [r for r in recs if r.get("first_byte")]
    if not recs:
        return {}
    return {"overhead_ms": summary([(r["first_byte"] - r["body"] - r["runner_ns"] - r.get("image_ns", 0)) / ms
                                    for r in recs]),
            "runner_ms": summary([r["runner_ns"] / ms for r in recs]),
            "to_admit_ms": summary([(r["admitted"] - r["tokenized"]) / ms for r in recs]),
            # to_admit holds the wait for the pool too: only the fetch part of image_ns is subtracted on top
            "overhead_minus_to_admit_ms": summary([(r["first_byte"] - r["body"] - r["runner_ns"] - _fetch_ns(r)
                                                    - (r["admitted"] - r["tokenized"])) / ms for r in recs])}


def _fetch_ns(r: dict) -> int:
    """A record's image fetch time: its image_ns minus the wait for the pool after tokenizing."""
    return max(0, r.get("image_ns", 0) - max(0, r.get("pixels", 0) - r["tokenized"])) if r.get("images") else 0


def bench_decode(batch: int, step_ms: float = 10.0, max_tokens: int = 400) -> dict:
    srv = Server(step_ms=step_ms, max_num_seqs=batch)
    try:
        bodies = [{"messages": [{"role": "user", "content": f"Hello {i}!"}], "max_tokens": max_tokens,
                   "stream": True, "seed": i, "ignore_eos": True, "chat_template_kwargs": KW} for i in range(batch)]
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        procs = min(8, batch)
        per = [batch // procs + (1 if i < batch % procs else 0) for i in range(procs)]
        ps, off = [], 0
        for n in per:
            p = ctx.Process(target=_client, args=(srv.port, bodies[off:off + n], n, 600, 1, q))
            off += n
            p.start()
            ps.append(p)
        time.sleep(1.0 + 0.1 * step_ms)  # past admission: every request decoding
        _, e0 = srv.get("/debug/engine")
        srv.get("/debug/reset")
        c0 = srv.cpu()
        t0 = time.monotonic()
        time.sleep(max_tokens * step_ms / 1e3 * 0.5)
        _, e1 = srv.get("/debug/engine")
        c1, dt = srv.cpu(), time.monotonic() - t0
        for _ in ps:
            q.get()
        for p in ps:
            p.join()
    finally:
        srv.stop()
    steps = [s for s in e1["steps"] if s[0] == batch]
    s0, s1 = e0["stats"], e1["stats"]
    nsteps = s1["steps"] - s0["steps"]
    phases = {name: (b - a) / nsteps / 1e3 for name, a, b in zip(e1["phases"], s0["phase_ns"], s1["phase_ns"])}
    return {"batch": batch, "step_ms": step_ms, "steady_steps": len(steps),
            "engine_cpu_us": summary([s[1] / 1e3 for s in steps]),
            "runner_calls_us": summary([s[2] / 1e3 for s in steps]),
            "phases_us_per_step": phases,
            "cpu_cores": {k: (c1[k] - c0[k]) / dt for k in c0}}


def bench_idle(conc: int, step_ms: float = 5.0, duration: float = 15.0, srv: Server | None = None) -> dict:
    own = srv is None
    srv = srv or Server(step_ms=step_ms, max_num_seqs=64)
    try:
        # with images, a body per request of the window (no image repeats within the run, so every request goes
        # through the pool and the encoder); text-only, bodies repeat
        n_bodies = max(64, conc * 4) if not IMAGES["on"] else max(96, conc * 48)
        bodies, prompt_tokens = classify_bodies(n_bodies, max_tokens=64, offset=IMAGES["next"])
        IMAGES["next"] += n_bodies
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        procs = min(conc, 8)
        per = [conc // procs + (1 if i < conc % procs else 0) for i in range(procs)]
        ps, off = [], 0
        warm = 2.0 if RUNNER == "fake" else 10.0  # the real runner: past the first wave of prefills
        for n in per:  # clients keep sending until after the window closes
            p = ctx.Process(target=_client, args=(srv.port, bodies[off:] + bodies[:off], n, warm + duration + 4, 0, q))
            off += n
            p.start()
            ps.append(p)
        time.sleep(warm)
        _, e0 = srv.get("/debug/engine")
        srv.get("/debug/reset")
        c0, th0, p0 = srv.cpu(), throttled_us(), pool_cpu(srv)
        t0 = time.monotonic()
        time.sleep(duration)
        _, e1 = srv.get("/debug/engine")
        dt = time.monotonic() - t0
        c1, th1, p1 = srv.cpu(), throttled_us(), pool_cpu(srv)
        _, recs = srv.get("/debug/stages")
        results = []
        for _ in ps:
            results.extend(q.get())
        for p in ps:
            p.join()
    finally:
        if own:
            srv.stop()
    s0, s1 = e0["stats"], e1["stats"]
    busy = (s1["gpu_busy_ns"] - s0["gpu_busy_ns"]) / 1e9
    idle = (s1["gpu_idle_ns"] - s0["gpu_idle_ns"]) / 1e9
    steps = s1["steps"] - s0["steps"]
    phases = {name: (b - a) / max(1, steps) / 1e3 for name, a, b in zip(e1["phases"], s0["phase_ns"], s1["phase_ns"])}
    ttft = [(r[1] - r[0]) / 1e6 for r in results]
    cpu = [x[1] / 1e3 for x in e1["steps"]]
    return {"conc": conc, "runner": RUNNER, "step_ms": step_ms, "prompt_tokens": prompt_tokens,
            "engine_affinity": srv.affinity, "gpu_idle_with_work_pct": 100 * idle / (idle + busy) if busy else 0.0,
            "device_busy_pct_of_wall": 100 * busy / dt, "steps": steps, "requests": len(recs),
            "generation_tokens_per_s": (s1["generation_tokens"] - s0["generation_tokens"]) / dt,
            "mean_step_device_ms": 1e3 * busy / max(1, steps), "engine_cpu_us": summary(cpu) if cpu else None,
            "phases_us_per_step": phases, "client_ttft_ms": summary(ttft) if ttft else None,
            "ttft_stages": stage_summary(recs), "cpu_cores": {k: (c1[k] - c0[k]) / dt for k in c0}
            | ({"image_pool": (p1 - p0) / dt} if p0 is not None and p1 is not None else {}),
            "images_per_s": (sum(1 for r in recs if r.get("images")) / dt) if IMAGES["on"] else None,
            "cgroup_throttled_ms": (th1 - th0) / 1e3 if th0 is not None and th1 is not None else None}


def bench_sse(conc: int = 128, max_tokens: int = 256, step_ms: float = 0.0, duration: float = 10.0) -> dict:
    srv = Server(step_ms=step_ms, max_num_seqs=max(64, conc))
    try:
        bodies = [{"messages": [{"role": "user", "content": f"Hi {i}"}], "max_tokens": max_tokens, "stream": True,
                   "seed": i, "ignore_eos": True, "chat_template_kwargs": KW} for i in range(conc)]
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        procs = min(8, conc)
        per = [conc // procs + (1 if i < conc % procs else 0) for i in range(procs)]
        ps, off = [], 0
        for n in per:
            p = ctx.Process(target=_client, args=(srv.port, bodies[off:off + n], n, duration + 3, 0, q))
            off += n
            p.start()
            ps.append(p)
        time.sleep(2.0)
        c0, t0, w0 = srv.cpu(), time.monotonic(), time.perf_counter_ns()
        _, e0 = srv.get("/debug/engine")
        time.sleep(duration)
        _, e1 = srv.get("/debug/engine")
        c1, dt, w1 = srv.cpu(), time.monotonic() - t0, time.perf_counter_ns()
        results = []
        for _ in ps:
            results.extend(q.get())
        for p in ps:
            p.join()
    finally:
        srv.stop()
    # Delivered: events of the streams that completed inside the window, as counted by the clients.
    done = [r for r in results if w0 <= r[2] <= w1]
    delivered = sum(r[3] for r in done) / ((w1 - w0) / 1e9)
    produced = (e1["stats"]["generation_tokens"] - e0["stats"]["generation_tokens"]) / dt
    fe = (c1["frontend"] - c0["frontend"]) / dt
    return {"conc": conc, "step_ms": step_ms, "events_per_s": delivered,
            "engine_tokens_per_s": produced, "frontend_cores": fe, "engine_cores": (c1["engine"] - c0["engine"]) / dt,
            "frontend_cpu_us_per_event": 1e6 * fe / delivered if delivered else None,
            "steps_per_s": (e1["stats"]["steps"] - e0["stats"]["steps"]) / dt,
            "stream_ms": summary([(r[2] - r[0]) / 1e6 for r in done]) if done else None}


def result_tag(what: str, warm: bool, fake_wait: str) -> str:
    """The result file's tag: -warm, and -sleep where the fake's wait applies (the TTFT benchmark's own server; decode,
    SSE and idle spin)."""
    sleeps = fake_wait == "sleep" and what in ("all", "ttft", "ttft+idle")
    return ("-warm" if warm else "") + ("-sleep" if sleeps else "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["all", "ttft", "decode", "idle", "sse", "ttft+idle"])
    ap.add_argument("--batch", type=int, nargs="*", default=[64, 256])
    ap.add_argument("--conc", type=int, nargs="*", default=[1, 8, 32])
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--fake-wait", choices=["spin", "sleep"], default="spin",
                    help="how the fake runner passes device time in ttft, and ttft+idle's idle (spin: CUDA's default)")
    ap.add_argument("--warm", action="store_true",
                    help="keep one background process spinning, so the CPU does not clock down between requests "
                         "(approximates a server with a performance governor; see bench/server/README.md)")
    ap.add_argument("--model", type=Path,
                    help="model directory (default: the Gemma 4 31B fixture, tests/scripts/fetch_fixtures.py)")
    ap.add_argument("--runner", choices=["fake", "cuda"], default="fake")
    ap.add_argument("--duration", type=float, default=15.0, help="idle: seconds measured per concurrency")
    ap.add_argument("--out", type=Path, help="write the JSON here instead of results/server/")
    ap.add_argument("--set", action="append", default=[], metavar="GROUP.KEY=VALUE",
                    help="server config override, e.g. --set scheduler.policy=priority (repeatable)")
    ap.add_argument("--images", action="store_true",
                    help="each classification request carries a new synthetic image (llm_mini_infer/synthetic.py), "
                         "sized by the server's image settings (--set images.max_soft_tokens=...)")
    ap.add_argument("--response-format", type=Path, default=None,
                    help="classification requests carry this JSON schema as a strict response_format "
                         "(e.g. tests/data/catalog.schema.json)")
    args = ap.parse_args()
    global RESPONSE_FORMAT
    if args.response_format is not None:
        RESPONSE_FORMAT = {"type": "json_schema", "json_schema": {
            "name": args.response_format.name.split(".")[0], "strict": True,
            "schema": json.loads(args.response_format.read_text())}}
    IMAGES["on"] = args.images
    global MODEL, RUNNER
    MODEL = args.model or MODEL
    RUNNER = args.runner
    for item in args.set:
        key, _, value = item.partition("=")
        node = EXTRA
        *parents, leaf = key.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = yaml.safe_load(value)
    if not (MODEL / "tokenizer.json").exists():
        sys.exit(f"{MODEL} has no tokenizer.json (uv run tests/scripts/fetch_fixtures.py for the default)")
    out: dict[str, Any] = {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "host": platform.platform(),
                           "cpu": platform.processor() or platform.machine(), "cores": os.cpu_count(),
                           "python": platform.python_version(), "warm": args.warm, "runner": RUNNER,
                           "model": str(MODEL)}
    spinner = subprocess.Popen([sys.executable, "-c", "while True: pass"]) if args.warm else None
    try:
        # one model load for both, with the run's --fake-wait as bench_ttft's own server takes it
        shared = (Server(step_ms=5.0, extra={"dev": {"runner": {"fake": {"wait": args.fake_wait}}}})
                  if args.what == "ttft+idle" else None)
        try:
            if args.what in ("all", "ttft", "ttft+idle"):
                out["ttft"] = [bench_ttft(n=args.n, fake_wait=args.fake_wait, srv=shared)]
            if args.what in ("all", "decode"):
                out["decode"] = [bench_decode(b) for b in args.batch]
            if args.what in ("all", "idle", "ttft+idle"):
                out["idle"] = [bench_idle(c, duration=args.duration, srv=shared) for c in args.conc]
        finally:
            if shared is not None:
                shared.stop()
        if args.what in ("all", "sse"):
            out["sse"] = [bench_sse()]
    finally:
        if spinner is not None:
            spinner.kill()
    RESULTS.mkdir(parents=True, exist_ok=True)
    tag = result_tag(args.what, args.warm, args.fake_wait)
    path = args.out or RESULTS / f"{time.strftime('%Y%m%d-%H%M%S')}-{args.what}{tag}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print(f"\nwrote {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
