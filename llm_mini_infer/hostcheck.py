"""The host check (docs/operations.md, "The host check"): one command, run inside the container on the machine that
will serve, that measures what the server's speed depends on and prints each measure against reference numbers.

    llm-mini-infer hostcheck --set model.path=MODEL [--set speculative.model=DRAFTER] [--inputs DIR]

It measures, in order:
1. **the GPU and the host:** the GPU's name, power limit, SM clock and driver; the CPU model, ATen's CPU capability
   with its ISA flags, and the CPU quota;
2. **host speed,** three times: a single-thread Python loop, torch's CUDA launch overhead, the torch + FlashInfer
   import (a slow host skews every latency the server has);
3. **startup:** the server with this container's configuration, started twice, to `/health` 200 (the first start
   also fills an empty cache volume);
4. **the load**, by the model's family.

**Gemma 4: an open loop of structured-output requests with one image** (`request_body`: the system prompt, a short user
text beside one JPEG, a strict json_schema, temperature 0, not streamed). Arrivals are Poisson at each of `--rates`,
with at most `--slots` requests in flight: an arrival beyond that is dropped and counted, so overload shows as drops
rather than as a queue in the client. Its inputs, in one of two modes:
- **synthetic** (no `--inputs`): `llm_mini_infer.synthetic`'s workload (`WORKLOAD_VERSION`, in its label), its system
  prompt `--prompt-tokens` long and a share `--tall-share` of its images tall, at the server's own image budget
  (`images.max_soft_tokens`). A test pattern gives short answers, so the latencies are this workload's, not an
  application's; they tell a healthy host from a slow one.
- **mounted** (`--inputs DIR`, a folder mounted into the container, never in the image): `DIR/frames/` (JPEG images,
  `.jpg` or `.jpeg`, each sent once in sorted order, so no request hits an image cache), `DIR/prompt.md` (or
  `prompt.txt`: the system prompt), `DIR/schema.json` (the json_schema), and optionally `DIR/user.txt` (the user turn's
  text) and `DIR/reference.json` (references for this workload, in `load_reference`'s fields; without it, no verdict
  against a reference). The answers, reply lengths and acceptance are then the workload's own.

**References** (`load_reference`): a load's verdicts compare it with a reference only when one matches the run: the
same power class (`power_class`) and, for each label the reference names, the same GPU, model, drafter, k and workload
(`matching_reference`). Without a match the report has the raw numbers and no verdict against a reference. The packaged
references (hostcheck_references.json) are labelled by what they were measured on; a mounted folder's reference.json
holds its own, from a healthy host's report (`as_reference`).

Per load: e2e and TTFT at p50 / p90 / p99, per image shape too; the reply's length; the requests in flight; from
`/metrics` over the measured window, acceptance per position, GPU idle due to the CPU, the grammar-aware heads' steps,
the draft universe and the forced tokens; FlashInfer's kept plans since the server started (must stay 0); power and SM
clock. **The drafter:** a server without its drafter fails the check, with the reason.

**Qwen3.8: closed loops** of classification-shaped requests (a ~1,024-token instruction, a synthetic image when images
are on, a short item), streamed without a grammar: the TTFT overhead at concurrency 1 and GPU idle due to the CPU at
concurrency 1, 8 and 32, against their bars.

Nothing it sends or receives is logged: the report holds aggregates only, and, per load, a digest of each answer by its
image's index (`answers`: 16 hex digits of its SHA-256), so two runs can be compared answer for answer without the
answers. A short answer from a small schema can be found again from its digest: a mounted workload's report is as
private as its answers. Exit status: 0 when no verdict failed (SLOW and LOW verdicts still exit 0: read them), 1 when
one failed (a model or drafter download, a server that does not start, no drafter, failed requests, a kept FlashInfer
plan) or the check raised (the report then holds `error` and what was measured before it), 2 when the check cannot run
(the GPU busy, `--rates` and `--slots` of different lengths, too few images, an unreadable inputs folder).
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import http.client
import io
import json
import math
import multiprocessing as mp
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .api.auth import client_headers

# The packaged references: the host measures' values (`host`) and labelled loads (`loads`, `load_reference`'s format)
REFERENCES = Path(__file__).with_name("hostcheck_references.json")
HOST_MARGINS = {"python_loop_s": 1.3, "import_s": 1.5, "launch_us": 1.3}  # ok up to these multiples of the reference
KW = {"enable_thinking": False}
STARTUP_BAR_S = 60.0  # a warm start (weights and kernels cached) is ready within this
TTFT_OVERHEAD_BAR_MS = 2.0  # the frontend's and the engine's share of TTFT at concurrency 1, p50
GPU_IDLE_BAR_PCT = 3.0  # the GPU idle while the CPU prepares a step
E2E_MARGIN = 1.25  # a load's e2e p50 and p99 within this multiple of its reference's: ok
LABELS = ("gpu", "model", "drafter", "k")  # what a reference may name beside its power class and workload
DEFAULT_USER_TEXT = "Describe this image."  # a mounted folder's user turn when it has no user.txt
WARM_REQUESTS = 16  # before the first load, unmeasured (concurrency 4): the system prompt's prefix, the grammar


def log(msg: str) -> None:
    print(f"[hostcheck {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pct(xs: list[float], p: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, round(p / 100 * (len(s) - 1)))] if s else float("nan")


def pcts(xs: list[float], nd: int = 1) -> dict[str, Any]:
    """n, p50 / p90 / p99, mean and max (numpy's `nearest` percentiles)."""
    if not xs:
        return {"n": 0}
    return {"n": len(xs), **{f"p{p}": round(pct(xs, p), nd) for p in (50, 90, 99)},
            "mean": round(sum(xs) / len(xs), nd), "max": round(max(xs), nd)}


# -- 1. the GPU and the host -------------------------------------------------------------------------------------------

def gpu_info() -> dict[str, Any]:
    fields = "name,power.limit,power.max_limit,clocks.max.sm,memory.total,memory.used,driver_version"
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
    except (OSError, subprocess.TimeoutExpired):
        return {"error": "nvidia-smi not found or hung"}
    gpus = []
    for line in out:
        v = [x.strip() for x in line.split(",")]
        gpus.append({"name": v[0], "power_limit_w": _num(v[1]), "power_max_w": _num(v[2]), "sm_clock_mhz": _num(v[3]),
                     "memory_mib": _num(v[4]), "memory_used_mib": _num(v[5]), "driver": v[6]})
    return {"gpus": gpus}


def _num(x: str) -> float | None:
    try:
        return float(x)
    except ValueError:
        return None


ISA_FLAGS = ("avx2", "fma", "avx512f", "avx512bw", "avx512vl", "avx512dq")  # what ATen's AVX2 and AVX512 need


def host_info() -> dict[str, Any]:
    from .config import cgroup_cpu_quota

    model = platform.processor() or ""
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text()
    except OSError:
        cpuinfo = ""
    for line in cpuinfo.splitlines():
        if line.startswith("model name"):
            model = line.split(":", 1)[1].strip()
            break
    capability = ""
    try:
        import torch

        capability = torch.backends.cpu.get_cpu_capability()
    except ImportError:
        pass
    quota, why = cgroup_cpu_quota()
    return {"cpu": model, "cpu_capability": capability, "isa": cpu_isa(cpuinfo), "cpus": os.cpu_count(),
            "cpu_quota": quota, "cpu_quota_source": why, "kernel": platform.release()}


def cpu_isa(cpuinfo: str) -> list[str]:
    """The `ISA_FLAGS` on /proc/cpuinfo's first `flags` line: none where there is no such line (aarch64, macOS)."""
    for line in cpuinfo.splitlines():
        if line.startswith("flags"):
            flags = set(line.partition(":")[2].split())
            return [f for f in ISA_FLAGS if f in flags]
    return []


# -- 2. host speed -----------------------------------------------------------------------------------------------------

def cpubench() -> dict[str, float]:
    """Host speed: single-thread Python, the torch + FlashInfer import, torch's CUDA launch."""
    t = time.perf_counter()
    sum(i * i for i in range(5_000_000))
    py = time.perf_counter() - t
    out = subprocess.run([sys.executable, "-c", "import time; t = time.perf_counter(); import torch, flashinfer; "
                          "print(time.perf_counter() - t)"], capture_output=True, text=True)
    imp = float(out.stdout.split()[-1]) if out.returncode == 0 and out.stdout.strip() else float("nan")
    launch = float("nan")
    try:
        import torch

        if torch.cuda.is_available():
            x = torch.zeros(1, device="cuda")
            for _ in range(2000):
                x.add_(1)
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(20000):
                x.add_(1)
            launch = (time.perf_counter() - t) / 20000 * 1e6
            torch.cuda.synchronize()
            del x
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return {"python_loop_s": py, "import_s": imp, "launch_us": launch}


# -- 3. the server -----------------------------------------------------------------------------------------------------

class Server:
    def __init__(self, config: str, port: int, out: Path, sets: list[str], log_requests: bool = False):
        self.port = port
        args = [sys.executable, "-m", "llm_mini_infer.cli", "serve", "--config", config, "--host", "127.0.0.1",
                "--port", str(port), "--set", "observability.debug_endpoints=true",
                "--set", "server.stop_with_parent=true",  # it stops when this check exits without stopping it
                "--set", f"observability.log_requests={'true' if log_requests else 'false'}",
                *(["--set", "observability.log_level=info"] if log_requests else []),  # the request lines are INFO
                *[x for s in sets for x in ("--set", s)]]
        self.log_path = out
        self.log = out.open("w")
        self.t0 = time.perf_counter()
        try:
            self.proc = subprocess.Popen(args, stdout=self.log, stderr=subprocess.STDOUT)
        except BaseException:
            self.log.close()
            raise
        self.ready_s = float("nan")

    def wait_ready(self, timeout: float = 900) -> float:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"the server exited during startup (code {self.proc.returncode}){self.tail()}")
            try:
                if self.get("/health")[0] == 200:
                    self.ready_s = time.perf_counter() - self.t0
                    return self.ready_s
            except OSError:
                pass
            time.sleep(0.1)
        self.stop()
        raise RuntimeError(f"the server was not ready after {timeout:.0f} s{self.tail()}")

    def tail(self, lines: int = 60) -> str:
        """The end of the server's log, for an error: in a container its log may be on a volume nobody reads once the
        container has ended (startup lines only: no request has run, and nothing logs a secret)."""
        try:
            last = self.log_path.read_text(errors="replace").splitlines()[-lines:]
        except OSError:
            return f"; its log: {self.log_path}"
        return f"; the last {len(last)} lines of its log ({self.log_path}):\n" + "\n".join(last)

    def get(self, path: str) -> tuple[int, Any]:
        """GET `path` with the API key the server holds, if any (`client_headers`): every path but the probes needs
        it."""
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        try:
            c.request("GET", path, headers=client_headers())
            r = c.getresponse()
            body = r.read()
            return r.status, json.loads(body) if body[:1] in (b"{", b"[") else body
        finally:
            c.close()

    def stop(self) -> None:
        """SIGINT (drain without a grace), then SIGKILL after 120 s; the child is reaped and the log closed (a second
        call does nothing more). The launcher's children are not in a new session: a launcher that had to be killed
        may leave its engine or frontends running, so after a kill the port is not known to be free."""
        try:
            if self.proc.poll() is None:
                self.proc.send_signal(2)
                try:
                    self.proc.wait(120)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    with contextlib.suppress(subprocess.TimeoutExpired):
                        self.proc.wait(10)
        finally:
            self.log.close()


# -- 4a. Qwen3.8: closed loops -----------------------------------------------------------------------------------------

def instruction(model_dir: str, target: int = 1024) -> str:
    """A classification system prompt of about `target` tokens."""
    from .render.renderer import Renderer

    tok = Renderer(model_dir).tokenizer
    labels = ["electronics", "clothing", "furniture", "toys", "groceries", "books", "garden", "sports", "beauty"]
    lines = ["You are a product classifier. Read the item and answer with exactly one label from the list below, "
             "followed by a confidence between 0 and 1. Follow every rule."]
    k = 0
    while len(tok.encode("\n".join(lines)).ids) < target:
        a, b = labels[k % len(labels)], labels[(k * 7 + 3) % len(labels)]
        lines.append(f"Rule {k + 1}: if the item description mentions features typical of {a}, but its packaging "
                     f"or intended use points to {b}, prefer {b} unless the title names {a} explicitly.")
        k += 1
    return "\n".join(lines)


def image_uri(i: int) -> str:
    """A deterministic 336 x 336 JPEG, different for every i (no encoder-cache hits, as with real images)."""
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(i)
    y, x = np.mgrid[0:336, 0:336]
    px = np.stack([(x + i * 7) % 256, (y * 2 + i * 13) % 256, (x + y + i) % 256], axis=-1).astype(np.int16)
    px = np.clip(px + rng.integers(-20, 20, px.shape), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(px).save(buf, "JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def bodies(model_dir: str, n: int, max_tokens: int, images: bool, offset: int) -> list[dict]:
    system = instruction(model_dir)
    out = []
    for i in range(offset, offset + n):
        item = f"Item {i}: a {['red', 'blue', 'green'][i % 3]} widget, model {1000 + i}, ships in a cardboard box."
        content: Any = item if not images else [{"type": "image_url", "image_url": {"url": image_uri(i)}},
                                                {"type": "text", "text": item}]
        out.append({"messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
                    "max_tokens": max_tokens, "stream": True, "seed": i, "ignore_eos": True,
                    "chat_template_kwargs": KW})
    return out


def _stream(port: int, body: dict) -> tuple[float, float]:
    """One streamed request: (seconds to the first content chunk, to the end)."""
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=300)
    t0 = time.perf_counter()
    first = None
    try:
        c.request("POST", "/v1/chat/completions", body=json.dumps(body),
                  headers={"content-type": "application/json", **client_headers()})
        r = c.getresponse()
        if r.status != 200:
            raise RuntimeError(f"a streamed request got HTTP {r.status}: {r.read()[:200]!r}")
        while True:
            line = r.readline()
            if not line:
                break
            if first is None and line.startswith(b"data: {") and b'"content":"' in line and \
                    b'"content":""' not in line:
                first = time.perf_counter() - t0
        return (first if first is not None else float("nan")), time.perf_counter() - t0
    finally:
        c.close()


def _client(port: int, reqs: list[dict], conc: int, until: float, q: Any) -> None:
    """A closed loop of `conc` threads over `reqs` until `until` (time.time()); puts the (ttft, e2e) list."""
    results: list[tuple[float, float]] = []
    lock = threading.Lock()
    nxt = [0]

    def worker() -> None:
        while time.time() < until:
            with lock:
                body = reqs[nxt[0] % len(reqs)]
                nxt[0] += 1
            r = _stream(port, body)
            with lock:
                results.append(r)

    ts = [threading.Thread(target=worker) for _ in range(conc)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    q.put(results)


def ttft_overhead(srv: Server, model_dir: str, images: bool, n: int, warmup: int) -> dict[str, Any]:
    """The server's overhead on the path to the first token at concurrency 1, from its stage records."""
    srv.get("/debug/reset")
    reqs = bodies(model_dir, n + warmup, 4, images, offset=0)
    client = [_stream(srv.port, b) for b in reqs]
    _, recs = srv.get("/debug/stages")
    recs, client = recs[warmup:], client[warmup:]
    ms = 1e6
    over = [(r["first_byte"] - r["recv"] - r["runner_ns"] - r.get("image_ns", 0)) / ms for r in recs]
    return {"n": len(recs), "overhead_p50_ms": pct(over, 50), "overhead_p99_ms": pct(over, 99),
            "runner_p50_ms": pct([r["runner_ns"] / ms for r in recs], 50),
            "client_ttft_p50_ms": pct([c[0] * 1e3 for c in client], 50),
            "client_ttft_p99_ms": pct([c[0] * 1e3 for c in client], 99)}


def gpu_idle(srv: Server, model_dir: str, images: bool, conc: int, duration: float) -> dict[str, Any]:
    """GPU idle due to the CPU: a closed loop of 55-token answers, measured after a warm-up."""
    reqs = bodies(model_dir, max(64, 4 * conc), 55, images, offset=10_000 + 1000 * conc)
    warm = 10.0
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_client, args=(srv.port, reqs, conc, time.time() + warm + duration + 2, q))
    p.start()
    time.sleep(warm)
    _, e0 = srv.get("/debug/engine")
    t0 = time.monotonic()
    time.sleep(duration)
    _, e1 = srv.get("/debug/engine")
    dt = time.monotonic() - t0
    results = q.get()
    p.join()
    s0, s1 = e0["stats"], e1["stats"]
    busy = (s1["gpu_busy_ns"] - s0["gpu_busy_ns"]) / 1e9
    idle = (s1["gpu_idle_ns"] - s0["gpu_idle_ns"]) / 1e9
    steps = s1["steps"] - s0["steps"]
    return {"conc": conc, "gpu_idle_pct": 100 * idle / (idle + busy) if busy else float("nan"),
            "device_busy_pct_of_wall": 100 * busy / dt, "steps": steps,
            "mean_step_ms": 1e3 * busy / max(1, steps), "requests": len(results),
            "tokens_per_s": (s1["generation_tokens"] - s0["generation_tokens"]) / dt}


def run_qwen(args: argparse.Namespace, srv: Server, model_dir: str, images: bool, report: dict) -> None:
    report["ttft"] = ttft_overhead(srv, model_dir, images, args.requests, args.warmup)
    log(f"TTFT at concurrency 1: {report['ttft']}")
    report["idle"] = []
    for c in [int(x) for x in args.levels.split(",") if x]:
        report["idle"].append(gpu_idle(srv, model_dir, images, c, args.duration or 20.0))
        log(f"GPU idle at {c}: {report['idle'][-1]}")


# -- 4b. Gemma 4: an open loop of structured requests ------------------------------------------------------------------

@dataclass
class Inputs:
    """What the requests carry: `frame(i)` is the i-th image's JPEG and `shape(i)` its (width, height)."""
    mode: str  # mounted | synthetic
    prompt: str
    schema: dict[str, Any]
    schema_name: str  # response_format's json_schema name, unless --schema-name gives one
    user_text: str
    count: int  # images available (synthetic: unbounded)
    frame: Any
    shape: Any
    summary: dict[str, Any] = field(default_factory=dict)
    reference: list[dict[str, Any]] = field(default_factory=list)  # reference.json's, or the packaged ones
    workload: dict[str, Any] = field(default_factory=dict)  # the labels a reference's `workload` is matched against


def schema_name(schema: dict[str, Any]) -> str:
    """A json_schema name from a schema's title (as a pydantic model's class name gives it), else `response`."""
    title = schema.get("title") if isinstance(schema.get("title"), str) else ""
    return re.sub(r"[^a-z0-9_-]+", "_", title.lower()).strip("_") or "response"


def load_reference(path: Path) -> list[dict[str, Any]]:
    """A reference file: one object or a list of them, each with `card` (what it was measured on, as text), `card_w`
    (its power class, `power_class`; default 600), `startup_s` (the warm starts' range), `loads` by rate with `e2e`,
    `ttft`, `reply`, `tpv`, `power` and `sm`, and optionally `accept_per_pos` (the acceptance per draft position) and
    the labels a run must match (`LABELS`, and `workload`: an object)."""
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"--inputs: {path.name} is not readable JSON ({exc})") from exc
    return parse_references(raw, path.name)


def parse_references(raw: Any, name: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for n, entry in enumerate(raw if isinstance(raw, list) else [raw]):
        loads = entry.get("loads") if isinstance(entry, dict) else None
        if not isinstance(loads, dict) or not loads:
            raise ValueError(f"--inputs: {name}, entry {n}: no `loads`")
        card_w = entry.get("card_w", 600)
        if not isinstance(card_w, (int, float)) or isinstance(card_w, bool):
            raise ValueError(f"--inputs: {name}, entry {n}: no power class (card_w {card_w!r}): a reference "
                             "needs the card's power limit in W, which the host that recorded it did not report")
        ref: dict[str, Any] = {"card": str(entry.get("card", "")), "card_w": power_class(float(card_w)), "loads": {}}
        ref.update({k: entry[k] for k in LABELS if entry.get(k) is not None})
        if isinstance(entry.get("workload"), dict):
            ref["workload"] = dict(entry["workload"])
        if entry.get("startup_s"):
            ref["startup_s"] = tuple(entry["startup_s"])
        if entry.get("accept_per_pos"):
            ref["accept_per_pos"] = list(entry["accept_per_pos"])
        for rate, r in loads.items():
            missing = [k for k in ("e2e", "ttft", "reply", "tpv", "power", "sm") if r.get(k) is None]
            if missing:
                raise ValueError(f"--inputs: {name}, entry {n}, rate {rate}: no {', '.join(missing)}")
            ref["loads"][float(rate)] = {**r, "e2e": tuple(r["e2e"]), "ttft": tuple(r["ttft"])}
        out.append(ref)
    return out


def packaged_references() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The packaged host measures (by key: (value, what it is)) and load references."""
    raw = json.loads(REFERENCES.read_text())
    host = {k: tuple(v) if isinstance(v, list) else v for k, v in raw["host"].items()}
    return host, parse_references(raw["loads"], REFERENCES.name)


def mounted_inputs(folder: Path) -> Inputs:
    """`folder`'s images (`frames/*.jpg` and `*.jpeg`, sorted), prompt (`prompt.md` or `prompt.txt`), schema
    (`schema.json`), and optionally its user turn (`user.txt`) and references (`reference.json`). Image names are never
    logged; only counts and sizes are."""
    from .vision.probe import probe

    frames_dir = folder / "frames"
    files = sorted(p for p in frames_dir.glob("*") if p.suffix.lower() in (".jpg", ".jpeg")) \
        if frames_dir.is_dir() else []
    prompt_file = next((folder / n for n in ("prompt.md", "prompt.txt") if (folder / n).is_file()), None)
    missing = [what for what, ok in (("frames/ with .jpg or .jpeg", files), ("prompt.md or prompt.txt", prompt_file),
                                     ("schema.json", (folder / "schema.json").is_file())) if not ok]
    if missing:
        raise ValueError(f"--inputs {folder}: no {', no '.join(missing)} (docs/operations.md, \"The host check\": "
                         "the folder's layout)")
    schema = json.loads((folder / "schema.json").read_text())
    user = folder / "user.txt"
    user_text = user.read_text().rstrip("\n") if user.is_file() else DEFAULT_USER_TEXT
    reference = load_reference(folder / "reference.json") if (folder / "reference.json").is_file() else []
    shapes: dict[str, int] = {}
    sizes: list[tuple[int, int]] = []
    for f in files:
        w, h = probe(f.read_bytes()).oriented  # the header only: nothing decodes
        sizes.append((w, h))
        shapes[f"{w}x{h}"] = shapes.get(f"{w}x{h}", 0) + 1
    top = dict(sorted(shapes.items(), key=lambda kv: -kv[1])[:4])
    if len(shapes) > len(top):
        top["other"] = len(files) - sum(top.values())
    return Inputs("mounted", prompt_file.read_text(), schema, schema_name(schema), user_text, len(files),
                  lambda i: files[i].read_bytes(), lambda i: sizes[i],
                  {"frames": len(files), "sizes": top, "user_text": "user.txt" if user.is_file() else "default",
                   "reference": "reference.json" if reference else "none"}, reference, {"mode": "mounted"})


def synthetic_inputs(count_tokens: Any, prompt_tokens: int, tall_share: float) -> Inputs:
    from . import synthetic

    prompt = synthetic.system_prompt(count_tokens, prompt_tokens)
    return Inputs("synthetic", prompt, synthetic.scene_schema(), synthetic.SCHEMA_NAME, synthetic.USER_TEXT, 10**9,
                  lambda i: synthetic.image_jpeg(i, tall_share), lambda i: synthetic.image_size(i, tall_share),
                  {"frames": "generated", "sizes": {f"{w}x{h}": f"{share:.0%}" for (w, h), share in
                                                    zip(synthetic.SIZES, (1 - tall_share, tall_share), strict=True)},
                   "prompt_tokens": prompt_tokens},
                  packaged_references()[1], {"mode": "synthetic", "version": synthetic.WORKLOAD_VERSION,
                                             "prompt_tokens": prompt_tokens, "tall_share": tall_share})


def request_body(model: str, prompt: str, image: bytes, schema: dict, name: str, max_tokens: int,
                 user_text: str = DEFAULT_USER_TEXT) -> bytes:
    """One request, as an OpenAI client sends it, in raw JSON: `chat_template_kwargs` at the top level, where the
    OpenAI client takes it as `extra_body`."""
    url = "data:image/jpeg;base64," + base64.b64encode(image).decode()
    return json.dumps({
        "model": model, "temperature": 0, "max_tokens": max_tokens,
        "messages": [{"role": "system", "content": [{"type": "text", "text": prompt}]},
                     {"role": "user", "content": [{"type": "text", "text": user_text},
                                                  {"type": "image_url", "image_url": {"url": url}}]}],
        "response_format": {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}},
        "chat_template_kwargs": KW}, separators=(",", ":")).encode()


def poisson_arrivals(rate: float, n: int, seed: int) -> list[float]:
    """`n` offsets of a Poisson process at `rate`, the first at 0, its n gaps scaled to sum to n / rate, so the
    offered rate is exact."""
    rng = random.Random(seed)
    gaps = [rng.expovariate(rate) for _ in range(n)]
    k = (n / rate) / sum(gaps) if n else 1.0
    out, t = [], 0.0
    for g in gaps:
        out.append(t)
        t += g * k
    return out


def schedule(rate: float, warmup: float, seconds: float, tail: float, seed: int) -> list[float]:
    """A load's arrivals up to the tail's end: one image is reserved for each."""
    end = warmup + seconds + tail
    return [t for t in poisson_arrivals(rate, math.ceil(end * rate) + 1, seed) if t < end]


def frames_needed(args: argparse.Namespace) -> int:
    """The images the warm-up and every load take, each sent once."""
    return WARM_REQUESTS + sum(len(schedule(float(r), args.load_warmup, args.duration or 120.0, args.tail,
                                            args.seed + n)) for n, r in enumerate(args.rates.split(",")))


_SAMPLE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+(\S+)")


def prometheus(text: str) -> dict[str, float]:
    """A /metrics page: `name{labels}` -> value (the samples of one name and label set summed)."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        m = _SAMPLE.match(line)
        if m and line[0] != "#":
            with contextlib.suppress(ValueError):
                out[m[1] + (m[2] or "")] = out.get(m[1] + (m[2] or ""), 0.0) + float(m[3])
    return out


def total(sample: dict[str, float], name: str, **match: str) -> float:
    """A metric's samples summed (those whose labels include `match`)."""
    return sum(v for k, v in sample.items() if k.partition("{")[0] == name
               and all(f'{a}="{b}"' in k for a, b in match.items()))


VERIFY_SERIES = ("vllm:spec_decode_num_drafts_total", "sglang:spec_verify_calls_total")  # one count per verify step
GENERATION_SERIES = ("vllm:generation_tokens_total", "sglang:generation_tokens_total")


def counter(sample: dict[str, float], names: tuple[str, ...]) -> tuple[str, float] | None:
    """The first of `names` that the page exports (under any label set) and its total over every label set; None when
    it exports none."""
    for name in names:
        if any(k.partition("{")[0] == name for k in sample):
            return name, total(sample, name)
    return None


def by_label(sample: dict[str, float], name: str, label: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in sample.items():
        if k.partition("{")[0] == name and (m := re.search(rf'(?<!\w){label}="([^"]*)"', k)):
            out[m[1]] = out.get(m[1], 0.0) + v
    return out


class GpuSampler:
    """nvidia-smi every 100 ms: power draw and SM clock."""

    def __init__(self) -> None:
        self.samples: list[tuple[float, float, float]] = []
        self.proc: subprocess.Popen | None = None
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        if not shutil.which("nvidia-smi"):
            return
        self.proc = subprocess.Popen(["nvidia-smi", "--query-gpu=power.draw,clocks.sm", "--format=csv,noheader,"
                                      "nounits", "-i", os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0],
                                      "-lms", "100"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                                     bufsize=1)
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self) -> None:
        assert self.proc and self.proc.stdout
        for line in self.proc.stdout:
            with contextlib.suppress(ValueError):
                p, c = (float(x) for x in line.split(","))
                self.samples.append((time.monotonic(), p, c))

    def stop(self) -> None:
        if self.proc:
            self.proc.terminate()
            with contextlib.suppress(Exception):
                self.proc.wait(10)
        if self.thread:
            self.thread.join(5)

    def window(self, t0: float, t1: float) -> dict[str, Any] | None:
        rows = [s for s in self.samples if t0 <= s[0] < t1]
        if not rows:
            return None
        return {"samples": len(rows), "power_w": round(sum(r[1] for r in rows) / len(rows), 1),
                "power_w_p90": pct([r[1] for r in rows], 90), "sm_mhz": round(sum(r[2] for r in rows) / len(rows))}


_REQUEST_LINE = re.compile(r"request id=(\S+) status=(\d+) .*? out=(\d+) ttft_ms=(\S+) e2e_ms=([\d.]+)")


def server_requests(log_path: Path, prefix: str, answered: set[str] = frozenset(), wait_s: float = 5.0
                    ) -> dict[str, float]:
    """The server's TTFT (ms) of each request whose id starts with `prefix`, from its request log
    (observability.log_requests: the request's receipt to its first token in the frontend). A frontend writes a
    request's line once its response is sent, so a client can read the response first: up to `wait_s`, it waits for
    the lines of the `answered` ids."""
    deadline = time.monotonic() + wait_s
    while True:
        out: dict[str, float] = {}
        seen = set()
        with contextlib.suppress(OSError):
            for m in _REQUEST_LINE.finditer(log_path.read_text(errors="replace")):
                if m[1].startswith(prefix):
                    seen.add(m[1])
                    if m[4] != "-":
                        out[m[1]] = float(m[4])
        if answered <= seen or time.monotonic() >= deadline:
            return out
        time.sleep(0.05)


class Client:
    """A client over raw HTTP: one request, its status, reply length, finish and its answer's digest (no contents
    kept)."""

    def __init__(self, port: int, timeout: float = 110.0) -> None:
        self.port, self.timeout = port, timeout
        self.headers = {"content-type": "application/json", **client_headers()}

    def send(self, rid: str, body: bytes) -> dict[str, Any]:
        rec: dict[str, Any] = {"id": rid, "status": 0}
        t0 = time.monotonic()
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=self.timeout)
        try:
            c.request("POST", "/v1/chat/completions", body=body, headers={**self.headers, "x-request-id": rid})
            r = c.getresponse()
            data = r.read()
            rec["status"] = r.status
            if r.status == 200:
                out = json.loads(data)
                rec["tokens"] = out["usage"]["completion_tokens"]
                rec["finish"] = out["choices"][0]["finish_reason"]
                content = out["choices"][0]["message"]["content"] or ""
                rec["digest"] = hashlib.sha256(content.encode()).hexdigest()[:16]
            else:
                rec["error"] = (json.loads(data).get("error") or {}).get("type") if data[:1] == b"{" else "http"
        except Exception as exc:  # a connection error or the client's timeout: a lost request
            rec["error"] = type(exc).__name__
        finally:
            c.close()
        rec["e2e_ms"] = round((time.monotonic() - t0) * 1e3, 1)
        return rec


def open_loop(srv: Server, client: Client, inputs: Inputs, model: str, schema_name: str, max_tokens: int, rate: float,
              slots: int, warmup: float, seconds: float, tail: float, seed: int, offset: int, tag: str) -> dict:
    """One load: Poisson arrivals at `rate` over images `offset` on, `slots` requests in flight at most (an arrival
    beyond is dropped). Arrivals in [warmup, warmup + seconds) are measured; the tail's arrivals go on while a measured
    request is in flight, so the last measured ones finish under the same load."""
    end = warmup + seconds
    arrivals = schedule(rate, warmup, seconds, tail, seed)
    body = [request_body(model, inputs.prompt, inputs.frame(offset + i), inputs.schema, schema_name, max_tokens,
                         inputs.user_text) for i in range(len(arrivals))]
    recs: list[dict] = []
    events: list[tuple[float, int]] = []  # (monotonic, +1 / -1): requests in flight
    lock = threading.Lock()
    state = {"in_flight": 0, "measured_left": 0}
    threads: list[threading.Thread] = []
    gpu = GpuSampler()
    marks: dict[str, dict[str, float]] = {}

    def metrics() -> dict[str, float]:
        c = http.client.HTTPConnection("127.0.0.1", srv.port, timeout=10)
        try:
            c.request("GET", "/metrics")
            return prometheus(c.getresponse().read().decode())
        finally:
            c.close()

    def one(i: int, at: float, measured: bool) -> None:
        w, h = inputs.shape(offset + i)
        rec = client.send(f"{tag}-{i:06d}", body[i])
        rec.update(at=at, measured=measured, tall=h > w, t_end=time.monotonic(), frame=offset + i)
        with lock:
            state["in_flight"] -= 1
            state["measured_left"] -= measured
            events.append((time.monotonic(), -1))
            recs.append(rec)

    gpu.start()
    try:
        time.sleep(0.5)  # the sampler's first reads
        start = time.monotonic() + 0.2
        ended = False
        for i, at in enumerate(arrivals):
            if "start" not in marks and at >= warmup:  # the window opens: the counters at its start
                time.sleep(max(0.0, start + warmup - time.monotonic()))
                marks["start"] = metrics()
            if not ended and at >= end:
                time.sleep(max(0.0, start + end - time.monotonic()))
                marks["end"], ended = metrics(), True
            if ended and not state["measured_left"]:
                break
            time.sleep(max(0.0, start + at - time.monotonic()))
            measured = warmup <= at < end
            with lock:
                if state["in_flight"] >= slots:
                    recs.append({"id": f"{tag}-{i:06d}", "at": at, "measured": measured, "dropped": True})
                    continue
                state["in_flight"] += 1
                state["measured_left"] += measured
                events.append((time.monotonic(), +1))
            t = threading.Thread(target=one, args=(i, at, measured), daemon=True)
            t.start()
            threads.append(t)
        if not ended:
            time.sleep(max(0.0, start + end - time.monotonic()))
            marks["end"] = metrics()
        for t in threads:
            t.join()
    finally:
        gpu.stop()
    ttft = server_requests(srv.log_path, tag + "-", {r["id"] for r in recs if r.get("status") == 200})
    out = summarize_load(recs, events, marks, start + warmup, start + end, ttft, rate, slots, seconds)
    out["gpu"] = gpu.window(start + warmup, start + end)
    out["frames_reserved"] = len(arrivals)
    # every answer of the load, warm-up and tail included, by its image's index
    out["answers"] = {str(r["frame"]): r["digest"] for r in sorted(recs, key=lambda r: r.get("frame", -1))
                      if r.get("digest")}
    return out


def summarize_load(recs: list[dict], events: list[tuple[float, int]], marks: dict[str, dict[str, float]], t0: float,
                   t1: float, ttft: dict[str, float], rate: float, slots: int, seconds: float) -> dict[str, Any]:
    meas = [r for r in recs if r["measured"]]
    sent = [r for r in meas if not r.get("dropped")]
    ok = [r for r in sent if r["status"] == 200 and not r.get("error")]
    errors: dict[str, int] = {}
    for r in sent:
        if r not in ok:
            k = f"{r['status']} {r.get('error')}"
            errors[k] = errors.get(k, 0) + 1
    out: dict[str, Any] = {
        "rate": rate, "slots": slots, "seconds": seconds, "measured": len(meas),
        "offered_rps": round(len(meas) / seconds, 3), "served_rps": round(len(ok) / seconds, 3),
        "dropped": len(meas) - len(sent), "failed": len(sent) - len(ok), "errors": errors,
        "finish": {f: sum(1 for r in ok if r.get("finish") == f) for f in {r.get("finish") for r in ok}},
        "e2e_ms": pcts([r["e2e_ms"] for r in ok]), "ttft_ms": pcts([ttft[r["id"]] for r in ok if r["id"] in ttft]),
        "reply_tokens": pcts([float(r["tokens"]) for r in ok], 0), "in_flight": in_flight(events, t0, t1)}
    for shape, tall in (("wide", False), ("tall", True)):
        rs = [r for r in ok if r["tall"] == tall]
        out[shape] = {"e2e_ms": pcts([r["e2e_ms"] for r in rs]),
                      "ttft_ms": pcts([ttft[r["id"]] for r in rs if r["id"] in ttft]),
                      "reply_tokens": pcts([float(r["tokens"]) for r in rs], 0)}
    done = sum(1 for r in recs if r.get("status") == 200 and not r.get("error") and t0 <= r["t_end"] < t1)
    if "start" in marks and "end" in marks:  # per request: those that completed in the window, as the counters count
        out["engine"] = engine_delta(marks["start"], marks["end"], done)
    return out


def in_flight(events: list[tuple[float, int]], t0: float, t1: float) -> dict[str, Any]:
    """The requests in flight over [t0, t1), time-weighted: mean, p99 and max."""
    level, at, share = 0, t0, {}
    for t, d in sorted(events):
        if t <= t0:
            level += d
            continue
        if t >= t1:
            break
        share[level] = share.get(level, 0.0) + t - at
        level, at = level + d, t
    share[level] = share.get(level, 0.0) + t1 - at
    span, acc, p99 = t1 - t0, 0.0, None
    for k in sorted(share):
        acc += share[k] / span
        if p99 is None and acc >= 0.99:
            p99 = k
    return {"mean": round(sum(k * v for k, v in share.items()) / span, 2), "p99": p99,
            "max": max(k for k, v in share.items() if v > 0)}


def engine_delta(a: dict[str, float], b: dict[str, float], completed: int) -> dict[str, Any]:
    """What the server did over the measured window: acceptance per position, GPU idle due to the CPU, the grammar-aware
    heads (the steps by head, the masks' unions: `union_sizes`, the draft universe), the forced prefix, the EOS-free
    finishes, the MTP verify rows' first rejections by mask, FlashInfer's kept plans, failed requests and grammar
    errors. A count that `counted` checks is None, not 0, when the server does not export its counter."""
    def d(name: str, **match: str) -> float:
        return total(b, name, **match) - total(a, name, **match)

    def counted(name: str) -> bool:
        return any(k.partition("{")[0] == name for k in b)

    def delta_by(name: str, label: str) -> dict[str, int] | None:
        if not counted(name):
            return None
        la, lb = by_label(a, name, label), by_label(b, name, label)
        return {k: round(v - la.get(k, 0.0)) for k, v in lb.items()}

    out: dict[str, Any] = {}
    drafts = d("vllm:spec_decode_num_drafts_total")
    verify, gen = counter(b, VERIFY_SERIES), counter(b, GENERATION_SERIES)
    out["verify_steps"] = round(verify[1] - total(a, verify[0])) if verify else None
    pa, pb = (by_label(s, "vllm:spec_decode_num_accepted_tokens_per_pos_total", "position") for s in (a, b))
    if drafts > 0:
        per = [round((pb[p] - pa.get(p, 0.0)) / drafts, 3) for p in sorted(pb, key=int)]
        out["accept"] = {"drafts": round(drafts), "per_pos": per, "tokens_per_verify": round(1 + sum(per), 3),
                         "verify_steps_per_request": round(drafts / completed, 2) if completed else None}
    busy, idle = d("llm_mini_infer:gpu_busy_seconds_total"), d("llm_mini_infer:gpu_idle_seconds_total")
    out["gpu_idle_pct"] = round(100 * idle / (idle + busy), 4) if busy else None
    ha, hb = (by_label(s, "llm_mini_infer:grammar_head_steps_total", "path") for s in (a, b))
    heads = {k: round(v - ha.get(k, 0.0)) for k, v in hb.items()}
    out["head_steps"] = heads
    reasons_a, reasons_b = (by_label(s, "llm_mini_infer:grammar_head_steps_total", "reason") for s in (a, b))
    out["head_dense_reasons"] = {k: round(v - reasons_a.get(k, 0.0)) for k, v in reasons_b.items()
                                 if k != "union" and v - reasons_a.get(k, 0.0)}
    out["sparse_share"] = round(heads.get("sparse", 0) / sum(heads.values()), 4) if sum(heads.values()) else None
    out["union_tokens"] = union_sizes(a, b)
    out["draft_universe_tokens"] = total(b, "llm_mini_infer:draft_universe_tokens") or None
    for key, name in (("draft_universe_steps", "llm_mini_infer:draft_universe_steps_total"),
                      ("draft_universe_rebuilds", "llm_mini_infer:draft_universe_rebuilds_total"),
                      ("forced_requests", "llm_mini_infer:grammar_forced_requests_total"),
                      ("forced_tokens", "llm_mini_infer:grammar_forced_tokens_total"),
                      ("eos_free_gated", "llm_mini_infer:grammar_eos_free_gated_total"),
                      ("requests_failed", "llm_mini_infer:requests_failed_total"),
                      ("grammar_errors", "llm_mini_infer:grammar_errors_total")):
        out[key] = round(d(name)) if counted(name) else None
    out["generation_tokens"] = round(gen[1] - total(a, gen[0])) if gen else None
    forced = out["forced_tokens"]
    out["forced_tokens_per_request"] = round(forced / completed, 2) if completed and forced is not None else None
    out["eos_free_finishes"] = delta_by("llm_mini_infer:grammar_eos_free_finishes_total", "in_flight")  # no, yes
    out["spec_first_rejections"] = delta_by("llm_mini_infer:spec_first_rejections_total", "mask")
    out["plans_kept"] = round(d("llm_mini_infer:attention_plans_kept_total"))
    out["plans_kept_since_start"] = round(total(b, "llm_mini_infer:attention_plans_kept_total"))
    return out


def union_sizes(a: dict[str, float], b: dict[str, float]) -> dict[str, Any] | None:
    """The unions of grammar masks over the window (`llm_mini_infer:grammar_head_union_tokens`): their count, mean
    tokens and the bucket bounds their p50 and p90 fall under; None without one."""
    name = "llm_mini_infer:grammar_head_union_tokens"
    n = total(b, name + "_count") - total(a, name + "_count")
    if n <= 0:
        return None
    ca, cb = by_label(a, name + "_bucket", "le"), by_label(b, name + "_bucket", "le")
    cum = sorted((float(le), v - ca.get(le, 0.0)) for le, v in cb.items())  # cumulative counts by upper bound

    def bound(q: float) -> int | None:
        le = next((le for le, c in cum if c >= q * n), None)
        return None if le is None or math.isinf(le) else round(le)

    return {"steps": round(n), "mean": round((total(b, name + "_sum") - total(a, name + "_sum")) / n, 1),
            "p50": bound(0.5), "p90": bound(0.9)}


def drafter_status(info: dict[str, Any], log_text: str, gemma: bool) -> dict[str, Any]:
    """Whether the server drafts with its assistant, from `/server_info`'s speculative settings and the CUDA runner's
    startup line (`MTP k = N with the assistant`): a Gemma 4 server without it fails, with the reason (the loads then
    check that its drafts were verified)."""
    cfg = info.get("config") or {}
    sp = cfg.get("speculative") or {}
    k = sp.get("num_speculative_tokens") or 0
    cuda = (cfg.get("runner") or {}).get("kind") == "cuda"
    m = re.search(r"MTP k = (\d+) with the assistant", log_text)
    out = {"model": cfg.get("speculative_source") or sp.get("model"), "k": k, "startup_line": m[0] if m else None,
           "reason": (info.get("reasons") or {}).get("speculative.num_speculative_tokens")}
    if not gemma:
        out["ok"] = True
    elif not sp.get("model"):
        out.update(ok=False, why="speculative.model is unset: Gemma 4 drafts with its assistant "
                                 "(--set speculative.model=google/gemma-4-31B-it-assistant)")
    elif not k or (cuda and not m):
        out.update(ok=False, why=f"the assistant is set but the runner drafts no tokens (k = {k}"
                                 f"{', no startup line' if cuda and not m else ''}): {out['reason']}")
    else:
        out["ok"] = True
    return out


def power_class(power_limit: float | None) -> int | None:
    """A card's power limit to the nearest 50 W (None: unknown): a reference applies to its own power class only, as a
    card at a lower limit runs GPU-bound work slower."""
    return None if power_limit is None else int(50 * math.floor(power_limit / 50 + 0.5))


def run_labels(report: dict[str, Any], info: dict[str, Any], inputs: Inputs | None, max_tokens: int) -> dict[str, Any]:
    """What a reference is matched against (`matching_reference`): the GPU, its power class, the model and the drafter
    as given (a repo id, or a path), k, and the workload with the server's image budget."""
    g = (report["gpu"].get("gpus") or [{}])[0]
    cfg = info.get("config") or {}
    d = report.get("drafter") or {}
    workload = dict(inputs.workload) if inputs else {}
    workload["max_soft_tokens"] = (cfg.get("images") or {}).get("max_soft_tokens")
    if workload.get("mode") == "mounted":
        workload["max_tokens"] = max_tokens
    return {"gpu": g.get("name"), "card_w": power_class(g.get("power_limit_w")),
            "model": (cfg.get("model_source") or "").split("@")[0] or (cfg.get("model") or {}).get("path"),
            "drafter": (d.get("model") or "").split("@")[0] or None, "k": d.get("k") or 0, "workload": workload}


def matching_reference(refs: list[dict[str, Any]], run: dict[str, Any]) -> dict[str, Any] | None:
    """The first reference whose power class is the run's and whose other labels, those it names, are the run's: its
    GPU, model, drafter, k and each key of its workload. None: no verdict against a reference."""
    for ref in refs:
        if run.get("card_w") is None or ref.get("card_w") != run["card_w"] or any(
                ref[k] != run.get(k) for k in LABELS if k in ref):
            continue
        if all((run.get("workload") or {}).get(k) == v for k, v in (ref.get("workload") or {}).items()):
            return ref
    return None


def load_verdicts(load: dict[str, Any], ref: dict[str, Any] | None, mode: str) -> dict[str, str]:
    """Each measure's verdict; one that starts with FAILED fails the check (`serve_and_load`)."""
    v: dict[str, str] = {}
    eng = load.get("engine") or {}
    v["requests"] = "ok" if not load["failed"] and not load["dropped"] else (
        f"FAILED: {load['failed']} failed ({load['errors']})" if load["failed"] else
        f"SLOW: {load['dropped']} dropped with every slot busy")
    v["plans_kept"] = "ok" if not eng.get("plans_kept_since_start") else \
        "FAILED: FlashInfer kept a plan (an MTP decode that may attend wrongly): investigate before serving"
    if eng.get("requests_failed") or eng.get("grammar_errors"):
        v["requests"] = f"FAILED: the engine failed {eng.get('requests_failed')} ({eng.get('grammar_errors')} grammar)"
    idle = eng.get("gpu_idle_pct")
    v["gpu_idle"] = "n/a" if idle is None else "ok" if idle < GPU_IDLE_BAR_PCT else \
        "SLOW: the host's CPU holds the GPU back"
    share = eng.get("sparse_share")
    v["heads"] = "n/a" if share is None else "ok" if share >= 0.95 else \
        f"dense steps: {eng.get('head_dense_reasons')} (dev.runner.cuda.sparse_head, sparse_head_rows)"
    r = (ref or {}).get("loads", {}).get(load["rate"])
    if r is None:
        v["e2e"] = v["accept"] = "no reference at this rate" if ref else "no reference"
        return v
    lim = E2E_MARGIN
    e = load["e2e_ms"]
    v["e2e"] = "n/a" if not e.get("n") else "ok" if e["p50"] <= r["e2e"][0] * lim and e["p99"] <= r["e2e"][1] * lim \
        else f"SLOW: beyond {lim:g}x the reference"
    tpv = (eng.get("accept") or {}).get("tokens_per_verify")
    per_pos = " / ".join(f"{a:.2f}" for a in ref.get("accept_per_pos") or [])
    v["accept"] = "n/a" if tpv is None else "ok" if tpv >= 0.9 * r["tpv"] else \
        "LOW: below 0.9x the reference" + (f" (per position: {per_pos})" if per_pos else "")
    reply = load["reply_tokens"].get("mean")
    if mode == "mounted" and reply:
        v["reply"] = "ok" if 0.8 * r["reply"] <= reply <= 1.25 * r["reply"] else \
            "differs from the reference's: are these the reference's images, prompt and schema?"
    return v


def run_gemma(args: argparse.Namespace, srv: Server, inputs: Inputs, report: dict) -> None:
    """The warm-up, then each load in turn over the next images; their results into `report["loads"]`."""
    client = Client(srv.port)
    _, models = srv.get("/v1/models")
    model = models["data"][0]["id"]  # the served name, the one requests send
    report["served_model"] = model
    rates = [float(x) for x in args.rates.split(",")]
    slots = [int(x) for x in args.slots.split(",")]
    name = args.schema_name or inputs.schema_name
    warm = [request_body(model, inputs.prompt, inputs.frame(inputs.count - 1 - i if inputs.mode == "mounted"
                                                            else 10**6 + i), inputs.schema, name,
                         args.max_tokens, inputs.user_text) for i in range(WARM_REQUESTS)]
    t0 = time.monotonic()
    lock, nxt, lost = threading.Lock(), iter(range(WARM_REQUESTS)), []

    def worker() -> None:
        for i in nxt:
            r = client.send(f"hostcheck-warm-{i:03d}", warm[i])
            if r["status"] != 200:
                with lock:
                    lost.append(f"{r['status']} {r.get('error')}")

    ts = [threading.Thread(target=worker) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    log(f"warm-up: {WARM_REQUESTS} requests in {time.monotonic() - t0:.1f} s" + (f", lost {lost}" if lost else ""))
    offset = 0
    report["loads"] = []
    for n, (rate, s) in enumerate(zip(rates, slots, strict=True)):
        log(f"load {rate:g} requests/s, {s} slots: {args.load_warmup:g} s warm-up, {args.duration or 120:g} s measured")
        r = open_loop(srv, client, inputs, model, name, args.max_tokens, rate, s, args.load_warmup,
                      args.duration or 120.0, args.tail, args.seed + n, offset, f"hostcheck-{n}")
        offset += r["frames_reserved"]
        report["loads"].append(r)
        e, t, eng = r["e2e_ms"], r["ttft_ms"], r.get("engine") or {}
        log(f"load {rate:g}/s: served {r['served_rps']}/s, dropped {r['dropped']}, failed {r['failed']}; e2e "
            f"p50/p90/p99 {e.get('p50')}/{e.get('p90')}/{e.get('p99')} ms, TTFT p50 {t.get('p50')} ms, reply mean "
            f"{r['reply_tokens'].get('mean')}; accept {(eng.get('accept') or {}).get('per_pos')}; GPU idle "
            f"{eng.get('gpu_idle_pct')}%; {r.get('gpu')}")


# -- the report --------------------------------------------------------------------------------------------------------

def verdict(key: str, value: float, host: dict[str, Any] | None = None) -> str:
    """`ok` up to a margin above the reference value (`host`: the packaged host measures), or under a bar, else
    `SLOW`: a slower host skews every latency. NaN is `n/a`."""
    if value != value:
        return "n/a"
    if key == "gpu_idle_pct":
        return "ok" if value < GPU_IDLE_BAR_PCT else "SLOW"
    if key == "ttft_overhead_p50_ms":
        return "ok" if value <= TTFT_OVERHEAD_BAR_MS else "SLOW"
    ref = (host if host is not None else packaged_references()[0])[key][0]
    return "ok" if value <= ref * HOST_MARGINS[key] else "SLOW"


def main(argv: list[str] | None = None) -> int:
    from . import synthetic  # the defaults in the help

    ap = argparse.ArgumentParser(prog="llm-mini-infer hostcheck", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.environ.get("LLM_MINI_INFER_CONFIG", "/etc/llm-mini-infer/config.yaml"))
    ap.add_argument("--port", type=int, default=18900, help="the check's own server (default %(default)s; not the "
                    "service's port)")
    ap.add_argument("--starts", type=int, default=2)
    ap.add_argument("--no-server", action="store_true", help="only the GPU, the host and cpubench")
    ap.add_argument("--out", type=Path, help="the JSON report (default: /cache/hostcheck/<time>.json, else "
                    "./hostcheck/<time>.json)")
    ap.add_argument("--set", action="append", default=[], metavar="GROUP.KEY=VALUE")
    ap.add_argument("--duration", type=float, help="seconds measured per load (Gemma 4: per rate, 120; Qwen: GPU "
                    "idle per concurrency, 20)")
    g = ap.add_argument_group("Gemma 4: the open loop")
    g.add_argument("--inputs", type=Path, help="the mounted mode: a folder with frames/ (.jpg or .jpeg images), "
                   "prompt.md or prompt.txt (the system prompt), schema.json (the json_schema), optionally user.txt "
                   "(the user turn's text) and reference.json (references for this workload); else the synthetic one")
    g.add_argument("--rates", default="2,4", help="requests/s of each load (default %(default)s)")
    g.add_argument("--slots", default="32,64", help="requests in flight at most, per load (default %(default)s; an "
                   "arrival beyond is dropped and counted)")
    g.add_argument("--load-warmup", type=float, default=30.0, help="unmeasured seconds of arrivals before each window "
                   "(default %(default)s)")
    g.add_argument("--tail", type=float, default=15.0, help="seconds of arrivals after the window, while a measured "
                   "request is in flight (default %(default)s)")
    g.add_argument("--seed", type=int, default=0, help="the first load's arrival schedule (default %(default)s; the "
                   "next: +1)")
    g.add_argument("--schema-name", help="response_format's json_schema name (default: the schema's title in lower "
                   "case, else `response`)")
    g.add_argument("--max-tokens", type=int, default=1024, help="each request's max_tokens (default %(default)s)")
    g.add_argument("--prompt-tokens", type=int, default=None, help="the synthetic mode: the system prompt's length in "
                   f"the model's tokens (default: llm_mini_infer.synthetic.PROMPT_TOKENS, {synthetic.PROMPT_TOKENS})")
    g.add_argument("--tall-share", type=float, default=None, help="the synthetic mode: the share of tall images, 0 to "
                   f"1 (default: llm_mini_infer.synthetic.TALL_SHARE, {synthetic.TALL_SHARE:g})")
    q = ap.add_argument_group("Qwen3.8: the closed loops")
    q.add_argument("--requests", type=int, default=150, help="measured TTFT requests at concurrency 1 (default "
                   "%(default)s)")
    q.add_argument("--warmup", type=int, default=50)
    q.add_argument("--levels", default="1,8,32")
    args = ap.parse_args(argv)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = args.out.parent if args.out else (Path("/cache/hostcheck") if os.access("/cache", os.W_OK)
                                               else Path.cwd() / "hostcheck")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out or out_dir / f"{stamp}.json"
    report: dict[str, Any] = {"time": stamp, "build": os.environ.get("LLM_MINI_INFER_BUILD", ""), "config": args.config}
    report["gpu"], report["host"] = gpu_info(), host_info()
    log(f"GPU: {report['gpu']}; host: {report['host']}")
    used = [g.get("memory_used_mib") or 0 for g in report["gpu"].get("gpus", [])]
    if used and max(used) > 2048 and not args.no_server:
        log(f"the GPU already has {max(used):.0f} MiB in use: stop what uses it first (another process or container)")
        return 2
    if len(args.rates.split(",")) != len(args.slots.split(",")):
        log(f"--rates {args.rates} and --slots {args.slots}: one slot count per rate")
        return 2
    if args.tall_share is not None and not 0 <= args.tall_share <= 1:
        log(f"--tall-share {args.tall_share}: a share from 0 to 1")
        return 2
    if args.prompt_tokens is not None and args.prompt_tokens < 1:
        log(f"--prompt-tokens {args.prompt_tokens}: at least 1")
        return 2
    inputs = None
    if args.inputs and not args.no_server:
        try:
            inputs = mounted_inputs(args.inputs)
        except ValueError as exc:
            log(str(exc))
            return 2
        if inputs.count < frames_needed(args):
            log(f"--inputs: {inputs.count} images, and the loads need {frames_needed(args)} (each image is sent once: "
                "a repeat would hit the image caches): mount more images, or shorten --duration")
            return 2
    failed = False
    try:
        report["cpubench"] = [cpubench() for _ in range(3)]
        log(f"cpubench: {report['cpubench']}")
        if not args.no_server:
            failed = serve_and_load(args, report, inputs, out_dir, stamp)
    except Exception as exc:  # the report keeps what was measured before it
        log(f"the check raised:\n{traceback.format_exc()}")
        report["error"] = f"{type(exc).__name__}: {exc}"
        failed = True
    out.write_text(json.dumps(report, indent=1, default=str))
    try:
        print_table(report)
    except Exception as exc:  # a partial report may lack what the table reads: the JSON has it
        log(f"the table could not be printed ({type(exc).__name__}: {exc})")
    print(f"\nreport: {out}")
    return 1 if failed else 0


def serve_and_load(args: argparse.Namespace, report: dict, inputs: Inputs | None, out_dir: Path, stamp: str) -> bool:
    """The downloads, the starts, the loads and their verdicts; True when one of them, or the drafter, failed."""
    from . import config as cfgmod
    from . import hub

    cfg = cfgmod.load(args.config, dict(cfgmod.parse_override(s) for s in args.set))
    try:
        hub.resolve(cfg)  # a repo id: downloaded once here, so the timed starts below read it from the cache
    except hub.HubError as exc:
        drafter = cfg.speculative.model and str(exc).startswith(f"model {cfg.speculative.model} ")
        report["drafter" if drafter else "error"] = {"ok": False, "why": f"its download failed: {exc}"} \
            if drafter else str(exc)
        log(f"{'the drafter' if drafter else 'the model'}: {exc}")
        return True
    cfgmod.derive(cfg)
    model_dir, images = cfg.model.path, bool(cfg.images.enabled)
    hf = json.loads((Path(model_dir) / "config.json").read_text())
    gemma = (hf.get("model_type") or (hf.get("text_config") or {}).get("model_type")) in cfgmod.GEMMA4_TYPES
    report["family"] = "gemma4" if gemma else "qwen"
    if gemma:
        if inputs is None:
            from . import synthetic
            from .render.renderer import Renderer

            tok = Renderer(model_dir).tokenizer
            inputs = synthetic_inputs(lambda text: len(tok.encode(text).ids),
                                      args.prompt_tokens or synthetic.PROMPT_TOKENS,
                                      synthetic.TALL_SHARE if args.tall_share is None else args.tall_share)
            log("the synthetic workload (no --inputs): test-pattern images, a generated prompt and schema; its "
                "latencies are its own")
        report["inputs"] = {"mode": inputs.mode, **inputs.summary, "frames_needed": frames_needed(args),
                            "schema_name": args.schema_name or inputs.schema_name, "max_tokens": args.max_tokens}
    elif inputs is not None:
        log("--inputs: unused (Qwen3.8's closed loops send their own requests)")
    if inputs is not None and inputs.mode == "mounted" and (args.prompt_tokens or args.tall_share is not None):
        log("--prompt-tokens and --tall-share: unused (the mounted mode sends its own prompt and images)")
    report["startup_s"] = []
    srv = None
    for i in range(args.starts):
        srv = Server(args.config, args.port, out_dir / f"{stamp}-server{i}.log", args.set, log_requests=gemma)
        try:
            report["startup_s"].append(round(srv.wait_ready(), 1))
        except RuntimeError as exc:
            srv.stop()
            report["error"] = f"start {i + 1}: {str(exc).splitlines()[0]}"
            log(str(exc))
            return True
        except BaseException:
            srv.stop()
            raise
        log(f"start {i + 1}: ready in {report['startup_s'][-1]} s")
        if i + 1 < args.starts:
            srv.stop()
    assert srv is not None
    try:
        _, info = srv.get("/server_info")
        report["drafter"] = drafter_status(info, srv.log_path.read_text(errors="replace"), gemma)
        log(f"drafter: {report['drafter']}")
        if gemma:
            run_gemma(args, srv, inputs, report)
        else:
            run_qwen(args, srv, model_dir, images, report)
    finally:
        srv.stop()
    if gemma:
        run = run_labels(report, info, inputs, args.max_tokens)
        ref = matching_reference(inputs.reference, run)
        report["reference"] = {"mode": inputs.mode, "run": run, "card": (ref or {}).get("card"), "values": ref}
        if ref is None:
            log(f"no reference matches this run ({'reference.json' if inputs.mode == 'mounted' else REFERENCES.name}):"
                " the loads get no verdict against a reference")
        report["as_reference"] = as_reference(report)
        for load in report["loads"]:
            load["verdicts"] = load_verdicts(load, ref, inputs.mode)
            drafts = ((load.get("engine") or {}).get("accept") or {}).get("drafts")
            if report["drafter"].get("ok") and not drafts:
                report["drafter"].update(ok=False, why=f"no drafts verified at {load['rate']:g}/s")
    return not report["drafter"].get("ok") or any(v.startswith("FAILED") for load in report.get("loads", [])
                                                  for v in load["verdicts"].values())


def as_reference(r: dict[str, Any]) -> dict[str, Any]:
    """This run's results in a reference file's fields (`load_reference`), labelled by the run (`run_labels`): what a
    mounted folder's reference.json takes, from a run on a host known to be healthy."""
    g = (r["gpu"].get("gpus") or [{}])[0]
    warm = r.get("startup_s", [])[1:] or r.get("startup_s", [])
    run = (r.get("reference") or {}).get("run") or {}
    out: dict[str, Any] = {"card": f"{g.get('name')}, {g.get('power_limit_w') or 0:.0f} W",
                           "card_w": power_class(g.get("power_limit_w")),
                           **{k: run[k] for k in (*LABELS, "workload") if run.get(k) is not None},
                           "startup_s": [min(warm), max(warm)] if warm else None, "loads": {}}
    for load in r.get("loads", []):
        acc, gpu = ((load.get("engine") or {}).get("accept") or {}), load.get("gpu") or {}
        e, t = load["e2e_ms"], load["ttft_ms"]
        out["loads"][f"{load['rate']:g}"] = {
            "e2e": [e.get("p50"), e.get("p99")], "ttft": [t.get("p50"), t.get("p99")],
            "reply": load["reply_tokens"].get("mean"), "tpv": acc.get("tokens_per_verify"),
            "power": gpu.get("power_w"), "sm": gpu.get("sm_mhz")}
        out.setdefault("accept_per_pos", acc.get("per_pos"))
    return out


def print_table(r: dict[str, Any]) -> None:
    rows: list[tuple[str, str, str, str]] = []
    gpus = r["gpu"].get("gpus") or [{}]
    g = gpus[0]
    host = packaged_references()[0]
    gemma = r.get("family") == "gemma4"
    ref = ((r.get("reference") or {}).get("values") or {}) if gemma else {}
    rows.append(("GPU", f"{g.get('name')}, driver {g.get('driver')}", ref.get("gpu") or "an SM120 GPU (the kernels')",
                 ""))
    if g.get("power_limit_w") is not None:
        rows.append(("power limit", f"{g['power_limit_w']:.0f} W (max {g.get('power_max_w') or 0:.0f} W)",
                     f"{ref['card_w']} W" if ref else "", ""))
    rows.append(("CPU", f"{r['host']['cpu']}, quota {r['host']['cpu_quota']:g}", host["hosts"], ""))
    cb = r.get("cpubench") or []
    for key, label, fmt in (("python_loop_s", "Python loop", "{:.3f} s"), ("import_s", "torch + flashinfer import",
                                                                             "{:.2f} s"),
                            ("launch_us", "torch CUDA launch", "{:.2f} us")):
        if not cb:
            break
        best = min(x[key] for x in cb)
        rows.append((label, fmt.format(best), host[key][1], verdict(key, best, host)))
    if r.get("error"):
        rows.append(("error", r["error"], "", "FAILED"))
    if r.get("startup_s"):
        warm = r["startup_s"][-1]
        src = f"bar {STARTUP_BAR_S:g} s" if not ref.get("startup_s") else (
            f"ready in {ref['startup_s'][0]:g}-{ref['startup_s'][1]:g} s warm ({ref['card']}); bar {STARTUP_BAR_S:g} s")
        rows.append(("startup to ready", " / ".join(f"{s:g} s" for s in r["startup_s"]), src,
                     "ok" if warm <= STARTUP_BAR_S else "SLOW"))
    if "drafter" in r:
        d = r["drafter"]
        rows.append(("the drafter", f"{d.get('model')}, k = {d.get('k')}" if d.get("ok") else str(d.get("why")),
                     f"{ref['drafter']}, k = {ref.get('k')}" if ref.get("drafter") else
                     "the model's assistant, at llm-mini-infer's default k" if gemma else "",
                     "ok" if d.get("ok") else "FAILED"))
    if gemma and "inputs" in r:
        rows += gemma_rows(r)
    if "ttft" in r:
        t = r["ttft"]
        rows.append(("TTFT overhead c=1, p50 / p99", f"{t['overhead_p50_ms']:.2f} / {t['overhead_p99_ms']:.2f} ms",
                     f"bar {TTFT_OVERHEAD_BAR_MS:g} ms (p50)", verdict("ttft_overhead_p50_ms", t["overhead_p50_ms"])))
        rows.append(("client TTFT c=1, p50 (4 tokens)", f"{t['client_ttft_p50_ms']:.1f} ms",
                     "GPU-bound: depends on the GPU and its power limit", ""))
    for x in r.get("idle", []):
        rows.append((f"GPU idle due to the CPU, c={x['conc']}", f"{x['gpu_idle_pct']:.4f}%",
                     f"bar {GPU_IDLE_BAR_PCT:g}%", verdict("gpu_idle_pct", x["gpu_idle_pct"])))
    w = [max(len(row[i]) for row in rows) for i in range(4)]
    print()
    for row in [("measure", "this host", "reference", "verdict"), *rows]:
        print("  ".join(c.ljust(w[i]) for i, c in enumerate(row)).rstrip())
    if gemma and not ref and "loads" in r:
        print(f"\nNo reference matches this run: the loads have raw numbers only. The report's reference.run holds "
              f"what a reference must match (its GPU, power class, model, drafter, k and workload); {REFERENCES.name} "
              "and a mounted reference.json hold the references.")
    if gemma and r.get("inputs", {}).get("mode") == "synthetic":
        print("\nThe synthetic workload: test-pattern images get short answers, so these latencies are this "
              "workload's, not an application's. Mount your own images, prompt and schema (--inputs) to measure those.")


def gemma_rows(r: dict[str, Any]) -> list[tuple[str, str, str, str]]:
    inp, rf = r["inputs"], r.get("reference") or {}
    ref = rf.get("values") or {}
    card = f" ({rf.get('card')})" if rf.get("card") else ""
    sizes = ", ".join(f"{k} {v}" for k, v in inp["sizes"].items())
    rows = [("inputs", f"{inp['mode']}: {inp['frames']} images ({sizes}), schema '{inp['schema_name']}', max_tokens "
             f"{inp['max_tokens']}", f"references{card}" if ref else "no matching reference", "")]
    for load in r.get("loads", []):
        v, eng, lr = load.get("verdicts") or {}, load.get("engine") or {}, ref.get("loads", {}).get(load["rate"]) or {}
        acc, gpu, e, t = eng.get("accept") or {}, load.get("gpu") or {}, load["e2e_ms"], load["ttft_ms"]
        head = f"{load['rate']:g}/s, {load['slots']} slots"
        rows.append((f"{head}: requests", f"{load['measured']} measured, {load['dropped']} dropped, {load['failed']} "
                     f"failed; in flight mean {load['in_flight']['mean']}, p99 {load['in_flight']['p99']}",
                     "none dropped or failed", v.get("requests", "")))
        rows.append((f"{head}: e2e p50 / p90 / p99", f"{e.get('p50')} / {e.get('p90')} / {e.get('p99')} ms",
                     f"{lr['e2e'][0]} / - / {lr['e2e'][1]} ms" if lr else "", v.get("e2e", "")))
        for shape in ("wide", "tall"):
            se, st = load[shape]["e2e_ms"], load[shape]["ttft_ms"]
            rows.append((f"  {shape} images: e2e, TTFT p50 / p99", f"{se.get('p50')} / {se.get('p99')}, "
                         f"{st.get('p50')} / {st.get('p99')} ms ({se['n']} requests)", "", ""))
        rows.append((f"{head}: TTFT p50 / p90 / p99", f"{t.get('p50')} / {t.get('p90')} / {t.get('p99')} ms",
                     f"{lr['ttft'][0]} / - / {lr['ttft'][1]} ms" if lr else "", ""))
        rows.append((f"{head}: reply tokens mean / p99 / max", f"{load['reply_tokens'].get('mean')} / "
                     f"{load['reply_tokens'].get('p99')} / {load['reply_tokens'].get('max')}",
                     f"mean {lr['reply']}" if lr else "", v.get("reply", "")))
        rows.append((f"{head}: acceptance per position", f"{acc.get('per_pos')} ({acc.get('tokens_per_verify')} a "
                     "verify step)", f"{lr['tpv']} a verify step" if lr else "", v.get("accept", "")))
        rows.append((f"{head}: GPU idle due to the CPU", f"{eng.get('gpu_idle_pct')}%", "under 3%",
                     v.get("gpu_idle", "")))
        rows.append((f"{head}: power, SM clock", f"{gpu.get('power_w')} W, {gpu.get('sm_mhz')} MHz",
                     f"{lr['power']} W, {lr['sm']} MHz" if lr else "", ""))
        rows.append((f"{head}: grammar heads", f"sparse {eng.get('sparse_share')} of {sum(eng['head_steps'].values())}"
                     f" steps, universe {eng.get('draft_universe_tokens')}, forced "
                     f"{eng.get('forced_tokens_per_request')} a request" if eng.get("head_steps") is not None else "",
                     "sparse ~1.0", v.get("heads", "")))
        rows.append((f"{head}: FlashInfer kept plans", f"{eng.get('plans_kept_since_start')}", "0",
                     v.get("plans_kept", "")))
    return rows


if __name__ == "__main__":
    sys.exit(main())
