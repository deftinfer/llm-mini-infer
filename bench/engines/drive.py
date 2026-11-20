#!/usr/bin/env python3
"""The benchmark's driver: runs one family of units (a server start, the smoke checks, the loads, a stop) for each of
the engine comparison's arms (bench/engines/arms.py), one engine after another, and writes everything a report needs.

    python3 bench/engines/drive.py --out DIR --model gemma|qwen --family FAMILY [--engines ours,vllm,sglang] \\
        [--rotate N] [--arm NAME ...] [--engine-bin ENGINE=PATH ...] [--inherit-env] [--client-pythonpath DIR] \\
        [--bound-file PATH] [--meta-extra JSON] [--m-concurrency 48] [--tokenizer ID] [--context N] [--skip-validity] \\
        [--dry-run]
    python3 bench/engines/drive.py --out DIR --family control --label start|middle|end [--python PATH]

**Families** (`--family`): `smoke` (the start and the smoke checks), `A1` (closed-loop loads of the head arms), `A2`
(the highest rate within a latency bound, needs `--bound-file`), `C` (two passes of every arm and the cost of batch
invariance), `D` (cold and warm start, then the structured-output validity run, and the grammar cost for
llm-mini-infer), `B` (vLLM's own draft counts), `I` (each engine in both modes, c=64 scene and chat on fresh servers:
the cost of batch invariance), `L` (long contexts, with `--context`: family I's arms under decode, prefill and
cached-history loads; a load whose prompt and reply do not fit the context is skipped and named), `M` (qwen only:
multi-turn sessions), `Q` (the head arm of each engine in its vendor image), `control` (the GPU control,
bench/engines/gpu_control.py).

**Layout** under `--out`: `<model>/<family>/<unit>/` holds `server.log` (private), `startup.json`, `smoke.json`, one
`<load>.json` per load, the `/metrics` pages around it, `gpu.csv`, `manifest.json` and, last, `done`. A unit whose
`done` exists is skipped. A unit that did not finish well writes its manifest with the status and no `done`, so a rerun
tries it again. `<out>/private/` holds the logs of a crashed engine, `<out>/drive.log` the driver's own progress.

The driver never prints or writes the value of an environment variable: a manifest holds the names only.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import importlib.util
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT / "bench/load"), str(HERE), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import arms  # noqa: E402
import gpu_control  # noqa: E402
import startup  # noqa: E402

HF_HOME = arms.HF_HOME
START_TIMEOUT_S = 1800
VALIDITY_LIMIT_S = 1800
LOAD_LIMIT_S = 3 * 3600
HEALTH_WAIT_S = 30
SCENE_INDEX = 10**9 - 1  # the smoke checks' workload index: no load ever reaches it
WARMUP_REQUESTS = 20  # run.py's default with --requests or --find-rate
# The client arguments every load gets, unless its family says otherwise
FIXED = {"--seed": "0", "--warmup": "10", "--duration": "60", "--tail": "30", "--timeout": "120"}
SCENE = ["--workload", "scene"]
CHAT = ["--workload", "chat", "--input-tokens", "1024", "--max-tokens", "256", "--ignore-eos"]
# Each load of a server starts at its own workload index (run.py's --first-index), so no load resends a request an
# earlier one put in the server's prefix cache; the same load starts at the same index on every engine and family (a
# family C or B load at c=64 sends A1's c=64 requests)
FIRST = {"c1": ["--first-index", "0"], "c16": ["--first-index", "1000000"], "c64": ["--first-index", "2000000"],
         "h1k": ["--first-index", "3000000"], "h8k": ["--first-index", "3100000"], "h32k": ["--first-index", "3200000"],
         "p32k": ["--first-index", "4000000"], "p128k": ["--first-index", "4100000"],
         "s32k": ["--first-index", "5000000"], "s128k": ["--first-index", "5100000"]}
VERSION_PACKAGES = ("vllm", "sglang", "sglang-kernel", "llm-mini-infer", "torch", "flashinfer-python",
                    "flashinfer-cubin", "flashinfer-jit-cache", "triton", "xgrammar", "llguidance", "transformers",
                    "deep-gemm", "sgl-kernel")
# A start that fails with one of these lines is a refusal the engine documents, not a failure
EXPECTED_REFUSALS = ("batch_invariant mode is not supported for GDN_ATTN",)
FAMILIES = ("smoke", "A1", "A2", "C", "D", "B", "I", "L", "M", "Q", "control")


# -- the engines' startup logs -----------------------------------------------------------------------------------------
# One regex per resolved field, applied to each line of the server log with the logger's prefix cut off; the last
# match wins and a field with no match is null. Each comment names the engine's source line at its tag.

def _num(text: str) -> int:
    return int(text.replace(",", ""))


def _quoted(text: str) -> str | None:
    return None if text == "None" else text.strip("'")


def _float_or_none(text: str) -> float | None:
    return None if text == "None" else float(text)


def _vllm_spec(text: str) -> str:
    try:
        cfg = ast.literal_eval(text)
        return f"{cfg.get('method')} k={cfg.get('num_speculative_tokens')}"
    except (ValueError, SyntaxError, AttributeError):
        return text[:120]


LOG_PATTERNS: dict[str, dict[str, re.Pattern]] = {
    "vllm": {
        # vllm/v1/core/kv_cache_utils.py: "GPU KV cache size: 1,234,560 tokens, Maximum concurrency ..." (INFO, once)
        "kv_tokens": re.compile(r"KV cache size: ([\d,]+) tokens"),
        # vllm/entrypoints/common/startup.py log_non_default_args: the dict lists the arguments that differ from the
        # defaults, so a default (prefix caching on, 128 sequences) prints nothing
        "max_running_requests": re.compile(r"non-default args: .*'max_num_seqs': (\d+)"),
        "prefix_cache": re.compile(r"non-default args: .*'enable_prefix_caching': (True|False)"),
        "context": re.compile(r"non-default args: .*'max_model_len': (\d+)"),
        "spec": re.compile(r"non-default args: .*'speculative_config': (\{[^}]*\})"),
        # vllm/platforms/cuda.py: "Using FLASH_ATTN attention backend out of potential backends: [...]", or when one is
        # forced "Using AttentionBackendEnum.TRITON_ATTN backend."
        "attention_backend": re.compile(r"Using (?:AttentionBackendEnum\.)?(\w+) (?:attention )?backend\b"),
        # vllm/model_executor/kernels/linear/__init__.py: "Selected <kernel class> for <module>", once per class and
        # module (`ACCUMULATE`: every distinct class)
        "fp8_kernel": re.compile(r"^Selected (\S+) for \S+$"),
        # vllm/config/scheduler.py: "Chunked prefill is enabled with max_num_batched_tokens=2048."
        "chunk_tokens": re.compile(r"Chunked prefill is enabled with max_num_batched_tokens=(\d+)\."),
        # vllm/v1/worker/gpu_worker.py: "... Desired GPU memory utilization is (0.92, 72.87 GiB). ..."
        "memory_fraction": re.compile(r"Desired GPU memory utilization is \(([\d.]+), "),
        # vllm/v1/worker/gpu_model_runner.py: "Graph capturing finished in 12 secs, took 0.85 GiB"
        "cuda_graphs": re.compile(r"Graph capturing finished in \d+ secs, took ([\d.]+) GiB"),
        # vllm/v1/worker/gpu_worker.py: "Using V2 Model Runner" (no line names the classic runner)
        "model_runner": re.compile(r"^(Using V2 Model Runner)$"),
        # vllm/config/cache.py: "Using fp8 data type to store kv cache." (quantized dtypes only)
        "kv_dtype": re.compile(r"^Using (\S+) data type to store kv cache"),
    },
    "sglang": {
        # python/sglang/srt/managers/scheduler.py: "max_total_num_tokens=N, chunked_prefill_size=N,
        # max_prefill_tokens=N, max_running_requests=N, context_len=N, ..." (the resolved values; TP0 only)
        "kv_tokens": re.compile(r"max_total_num_tokens=(\d+)"),
        "max_running_requests": re.compile(r"max_running_requests=(\d+)"),
        "chunk_tokens": re.compile(r"chunked_prefill_size=(-?\d+)"),
        "context": re.compile(r"context_len=(\d+)"),
        # python/sglang/srt/entrypoints/engine.py: "server_args={...}", the resolved ServerArgs as a dict
        "prefix_cache": re.compile(r"'disable_radix_cache': (True|False)"),
        "attention_backend": re.compile(r"'attention_backend': ('[^']*'|None)"),
        "fp8_kernel": re.compile(r"'fp8_gemm_runner_backend': ('[^']*'|None)"),  # the requested value, not the resolved
        "memory_fraction": re.compile(r"'mem_fraction_static': ([0-9.]+|None)"),
        "spec": re.compile(r"'speculative_algorithm': (None|'[^']*')"),
        "spec_steps": re.compile(r"'speculative_num_steps': (\d+|None)"),
        "cuda_graphs": re.compile(r"'disable_cuda_graph': (True|False)"),
        "kv_dtype": re.compile(r"'kv_cache_dtype': '([^']*)'"),
        "batch_invariant_active": re.compile(r"'enable_deterministic_inference': (True|False)"),
    },
}
CONVERT: dict[str, dict[str, Any]] = {
    "vllm": {"kv_tokens": _num, "max_running_requests": int, "prefix_cache": lambda s: s == "True", "context": int,
             "spec": _vllm_spec, "chunk_tokens": int, "memory_fraction": float,
             "cuda_graphs": lambda s: f"captured, {s} GiB", "model_runner": lambda s: "v2"},
    "sglang": {"kv_tokens": int, "max_running_requests": int, "chunk_tokens": int, "context": int,
               "prefix_cache": lambda s: s == "False", "attention_backend": _quoted, "fp8_kernel": _quoted,
               "memory_fraction": _float_or_none, "spec": _quoted,
               "spec_steps": lambda s: None if s == "None" else int(s),
               "cuda_graphs": lambda s: s == "False", "batch_invariant_active": lambda s: s == "True"},
}
# Fields that list every distinct match in order of first appearance instead of keeping the last
ACCUMULATE = {"vllm": ("fp8_kernel",)}
# vLLM v0.33.0 logs no line saying batch invariance is on; with VLLM_BATCH_INVARIANT=1 torch.compile traces
# determinism/batch_invariant.py (a Dynamo UserWarning and a traceback line name it) and a sampled request warns; the
# lines may lack the logger's prefix (raw stderr)
VLLM_INVARIANT_MARKERS = (re.compile(r"determinism/batch_invariant\.py|'batch_invariant\.py'"),
                          re.compile(r"Random sampling without an explicit seed may not be batch invariant"))
# The logger's own prefix: vLLM's "(EngineCore pid=1) INFO 11-12 10:00:00 [file.py:1] ", SGLang's "[<time> TP0] "
_PREFIXES = (re.compile(r"^(?:\([^)]*\) )?[A-Z]+ +\d\d-\d\d \d\d:\d\d:\d\d(?:\.\d+)? \[[^\]]*\] "),
             re.compile(r"^\[\d{4}-\d\d-\d\d \d\d:\d\d:\d\d[^\]]*\] "))
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def log_message(line: str) -> str:
    """A server log line without the ANSI colors and the logger's prefix."""
    line = _ANSI.sub("", line.rstrip("\n"))
    for p in _PREFIXES:
        if m := p.match(line):
            return line[m.end():]
    return line


def parse_log(engine: str, lines: list[str]) -> dict[str, Any]:
    """The resolved fields of one engine's server log (`LOG_PATTERNS`): the last match of each, converted; null where
    nothing matched. Never a raw log line."""
    patterns, convert = LOG_PATTERNS.get(engine, {}), CONVERT.get(engine, {})
    found: dict[str, str] = {}
    distinct: dict[str, list[str]] = {}
    marked = False
    for line in lines:
        msg = log_message(line)
        if engine == "vllm" and not marked:
            raw = _ANSI.sub("", line)
            marked = any(rx.search(raw) or rx.search(msg) for rx in VLLM_INVARIANT_MARKERS)
        for field_name, rx in patterns.items():
            if m := rx.search(msg):
                if field_name in ACCUMULATE.get(engine, ()):
                    if m.group(1) not in distinct.setdefault(field_name, []):
                        distinct[field_name].append(m.group(1))
                    found[field_name] = ", ".join(distinct[field_name])
                else:
                    found[field_name] = m.group(1)
    out: dict[str, Any] = {}
    for field_name, text in found.items():
        try:
            out[field_name] = convert.get(field_name, str)(text)
        except (ValueError, TypeError):
            out[field_name] = None
    if engine == "vllm":  # a log that reached its KV cache line without a marker: invariance is off
        out["batch_invariant_active"] = True if marked else (False if "kv_tokens" in found else None)
    steps = out.pop("spec_steps", None)
    if out.get("spec") is not None and steps is not None:
        out["spec"] = f"{out['spec']} steps={steps}"
    return {k: out.get(k) for k in RESOLVED_FIELDS}


RESOLVED_FIELDS = ("kv_tokens", "max_running_requests", "prefix_cache", "attention_backend", "fp8_kernel",
                   "chunk_tokens", "memory_fraction", "batch_invariant_active", "spec", "context", "cuda_graphs",
                   "model_runner", "kv_dtype", "notes")


_OURS_FULL_PAGE = re.compile(r"full pages hold (\d+) positions")
OURS_FP8 = "CUTLASS (llm-mini-infer's own block-FP8 GEMMs)"


def resolve_ours(info: dict[str, Any], log_lines: list[str] | None = None) -> dict[str, Any]:
    """The allow-listed fields of llm-mini-infer's `GET /server_info`. A sliding-window model's KV arena (Gemma 4)
    holds window pages and full pages of the same bytes, `dev.cache.page_size` being the window page's positions: its
    `kv_tokens` are the full layers' positions if every page held them (the startup line "KV arena: ... full pages hold
    N positions"), as a full-attention model's are its pages times their positions."""
    def dig(node: Any, *path: str) -> Any:
        for key in path:
            node = node.get(key) if isinstance(node, dict) else None
        return node

    cfg, stats = info.get("config") or {}, dig(info, "runner", "stats") or {}
    pages, page = dig(cfg, "dev", "cache", "num_pages"), dig(cfg, "dev", "cache", "page_size")
    full = [int(m.group(1)) for ln in log_lines or () if (m := _OURS_FULL_PAGE.search(ln))]
    window = bool(full)
    if full:
        page = full[-1]
    fp8 = stats.get("fp8_gemm") or (OURS_FP8 if cfg else None)
    graphs = stats.get("graphs")
    k = dig(cfg, "speculative", "num_speculative_tokens")
    out: dict[str, Any] = dict.fromkeys(RESOLVED_FIELDS)
    out.update(
        kv_tokens=pages * page if isinstance(pages, int) and isinstance(page, int) else None,
        max_running_requests=dig(cfg, "scheduler", "max_num_seqs"),
        prefix_cache=dig(cfg, "cache", "enable_prefix_caching"),
        attention_backend=stats.get("attention"), fp8_kernel=fp8,
        chunk_tokens=dig(cfg, "scheduler", "max_num_batched_tokens"),
        memory_fraction=dig(cfg, "runner", "cuda", "memory_utilization"),
        batch_invariant_active=dig(cfg, "dev", "runner", "cuda", "batch_invariant"),
        spec=f"mtp k={k}" if k else None, context=dig(cfg, "model", "max_model_len"),
        cuda_graphs=bool(graphs) if graphs is not None else None,
        notes="a KV arena shared by window and full pages: kv_tokens counts the full layers' positions if every page "
              "held them" if window else None)
    return out


_PATH = re.compile(r"(?<![\w.])/[\w.@+-]+(?:/[\w.@+-]*)*")
_SECRET = re.compile(r"(?:hf|gh[pousr]|rpa|sk)[_-][A-Za-z0-9]{12,}")


def clean_line(text: str) -> str:
    """A log line fit for a manifest: the logger's prefix, absolute paths and secret-looking words cut out."""
    text = _SECRET.sub("<token>", _PATH.sub("<path>", log_message(text)))
    return text.strip()[:300]


def error_line(lines: list[str]) -> str:
    """The last line of a log that names an error, else its last non-empty line."""
    errors = [ln for ln in lines if re.search(r"Error|Exception|error:|Traceback|FATAL|CUDA out of memory", ln)]
    pool = errors or [ln for ln in lines if ln.strip()]
    return clean_line(pool[-1]) if pool else ""


# -- small helpers -----------------------------------------------------------------------------------------------------

@dataclass
class Ctx:
    args: argparse.Namespace
    model: arms.Model
    out: Path
    base_url: str
    client_dir: Path
    tokenizer: str
    bins: dict[str, str]
    meta_extra: dict[str, Any]
    run_module: Any = None
    tok_cache: dict = field(default_factory=dict)

    def log(self, msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        if not self.args.dry_run:
            self.out.mkdir(parents=True, exist_ok=True)
            with open(self.out / "drive.log", "a") as f:
                f.write(line + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def http(url: str, body: dict | None = None, timeout: float = 600) -> tuple[int, bytes]:
    """One request with the standard library: (status, body), or (0, b"") when the connection fails."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except (OSError, urllib.error.URLError, ValueError):
        return 0, b""


def save_metrics(ctx: Ctx, path: Path) -> str:
    status, body = http(ctx.base_url + "/metrics", timeout=30)
    text = body.decode(errors="replace") if status == 200 else ""
    path.write_text(text)
    return text


def alive(h: startup.Handle, ctx: Ctx, wait_s: float = HEALTH_WAIT_S) -> bool:
    """The server process runs and `/health` answers 200, within `wait_s`."""
    deadline = time.monotonic() + wait_s
    while True:
        if h.proc.poll() is not None:
            return False
        if http(ctx.base_url + "/health", timeout=10)[0] == 200:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(1)


def run_child(argv: list[str], env: dict[str, str], cwd: Path, log_path: Path, limit: float | None) -> tuple[int, bool]:
    """Run a command in its own process group with its output appended to `log_path`. Returns (exit code, timed out);
    a command past `limit` seconds is killed with its group."""
    with open(log_path, "ab") as log:
        proc = subprocess.Popen(argv, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                start_new_session=True)
        try:
            return proc.wait(timeout=limit), False
        except subprocess.TimeoutExpired:
            with contextlib.suppress(OSError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            return -9, True
        except BaseException:
            with contextlib.suppress(OSError):
                os.killpg(proc.pid, signal.SIGKILL)
            raise


def client_env(ctx: Ctx) -> dict[str, str]:
    """The environment of the client children: this process's, with the Hugging Face cache offline. The API key and
    the Hugging Face token stay behind: the servers take no key and the cache needs no network."""
    env = dict(os.environ)
    for name in ("OPENAI_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        env.pop(name, None)
    env.update(HF_HOME=HF_HOME, HF_HUB_OFFLINE="1")
    if ctx.args.client_pythonpath:
        env["PYTHONPATH"] = str(ctx.client_dir)
    return env


def model_dir(repo: str) -> Path:
    return Path(HF_HOME) / "hub" / ("models--" + repo.replace("/", "--"))


def revision(repo: str) -> str | None:
    try:
        text = (model_dir(repo) / "refs" / "main").read_text().strip()
    except OSError:
        return None
    return text if re.fullmatch(r"[0-9a-f]{40}", text) else None


def engine_python(binary: str) -> str:
    return os.path.join(os.path.dirname(binary), "python3")


def package_versions(binary: str, env: dict[str, str]) -> dict[str, str]:
    """The versions of `VERSION_PACKAGES` installed in the engine's interpreter (the `python3` beside its command);
    the missing ones are left out."""
    py = engine_python(binary)
    if not os.path.exists(py):
        return {}
    code = ("import importlib.metadata as m, json\nout = {}\nfor n in " + repr(VERSION_PACKAGES) + ":\n    try:\n"
            "        out[n] = m.version(n)\n    except Exception:\n        pass\nprint(json.dumps(out))")
    try:
        res = subprocess.run([py, "-c", code], env=env, capture_output=True, text=True, timeout=120)
        return json.loads(res.stdout.strip().splitlines()[-1])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return {}


def host_block() -> dict[str, Any]:
    """The GPU and the host, from the host check's readers."""
    keys = ("gpu", "power_limit_w", "power_max_w", "sm_clock_max_mhz", "memory_mib", "driver", "cuda", "cpu", "cpus",
            "cpu_quota", "kernel")
    out: dict[str, Any] = dict.fromkeys(keys)
    try:
        from llm_mini_infer import hostcheck

        gpus = hostcheck.gpu_info().get("gpus") or []
        if gpus:
            g = gpus[0]
            out.update(gpu=g.get("name"), power_limit_w=g.get("power_limit_w"), power_max_w=g.get("power_max_w"),
                       sm_clock_max_mhz=g.get("sm_clock_mhz"), memory_mib=g.get("memory_mib"), driver=g.get("driver"))
        h = hostcheck.host_info()
        out.update({k: h.get(k) for k in ("cpu", "cpus", "cpu_quota", "kernel")})
    except Exception:  # a host without the pieces: the fields stay null
        out.update(cpus=os.cpu_count(), kernel=platform.release())
    try:
        text = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30).stdout
        if m := re.search(r"CUDA Version: (\d+\.\d+)", text):
            out["cuda"] = m.group(1)
    except (OSError, subprocess.SubprocessError):
        pass
    return out


# -- the units ---------------------------------------------------------------------------------------------------------

@dataclass
class LoadSpec:
    name: str
    args: list[str]
    fixed: dict[str, str] = field(default_factory=dict)  # replaces entries of FIXED


@dataclass
class Unit:
    arm: arms.Arm
    family: str
    name: str  # the unit's folder
    passno: int | None = None  # family C: 1 or 2


def rotated(engines: list[str], n: int) -> list[str]:
    """`arms.ENGINES` rotated by `n`, then cut to `engines`."""
    k = n % len(arms.ENGINES)
    order = list(arms.ENGINES[k:] + arms.ENGINES[:k])
    return [e for e in order if e in engines]


def build_units(args: argparse.Namespace) -> list[Unit]:
    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    bad = [e for e in engines if e not in arms.ENGINES]
    if bad:
        raise SystemExit(f"--engines: unknown {', '.join(bad)}")
    table = arms.arms(args.model, args.context)
    family = args.family
    if family == "Q":
        pool = table["A1"]
    elif family in table:
        pool = table[family]
    else:
        raise SystemExit(f"--family {family}: the {args.model} model has no such family")
    pool = [a for a in pool if a.engine in engines]
    if args.arm:
        known = {a.name for a in pool}
        missing = [n for n in args.arm if n not in known]
        if missing:
            raise SystemExit(f"--arm {', '.join(missing)}: not an arm of this family (have {', '.join(sorted(known))})")
        pool = [a for a in pool if a.name in args.arm]
    pool = [override(a, args) for a in pool]
    place = {e: i for i, e in enumerate(rotated(engines, args.rotate))}
    pool = sorted(pool, key=lambda a: place[a.engine])  # a stable sort: an engine's own arms keep their order
    units: list[Unit] = []
    for a in pool:
        if family == "C":
            units += [Unit(a, family, f"{a.name}-p1", 1), Unit(a, family, f"{a.name}-p2", 2)]
        else:
            units.append(Unit(a, family, a.name))
    return units


def _pairs(items: list[str], flag: str) -> list[tuple[str, str]]:
    out = []
    for item in items:
        name, sep, value = item.partition("=")
        if not sep or not name or not value:
            raise SystemExit(f"{flag} {item}: ARM=VALUE")
        out.append((name, value))
    return out


def override(arm: arms.Arm, args: argparse.Namespace) -> arms.Arm:
    """The arm with the documented fallbacks for it (`--attention`, `--drop`, `--add-arg`), each a named
    deviation with the reasons given by `--deviation`. The unit keeps the arm's name."""
    changes: dict[str, Any] = {}
    notes: list[str] = []
    for name, value in _pairs(args.attention, "--attention"):
        if name == arm.name:
            changes["attention"] = value
            notes.append(f"attention backend {value} forced")
    drops = [v for n, v in _pairs(args.drop, "--drop") if n == arm.name]
    if drops:
        changes["drop"] = arm.drop + tuple(drops)
        notes += [f"{v} left out" for v in drops]
    extra = [v for n, v in _pairs(args.add_arg, "--add-arg") if n == arm.name]
    if extra:
        changes["extra"] = arm.extra + tuple(extra)
        notes.append(f"added: {' '.join(extra)}")
    notes += [v for n, v in _pairs(args.deviation, "--deviation") if n == arm.name]
    if not changes and not notes:
        return arm
    import dataclasses

    return dataclasses.replace(arm, deviations=arm.deviations + tuple(notes), **changes)


def unit_dir(ctx: Ctx, family: str, name: str) -> Path:
    return ctx.out / ctx.args.model / family / name


def a1_rate(ctx: Ctx, arm: arms.Arm) -> float:
    path = unit_dir(ctx, "A1", arm.name) / "scene-c64.json"
    try:
        rate = json.loads(path.read_text())["loads"][0]["served_rps"]
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        raise LookupError(f"no served rate in A1's scene-c64 report for {arm.name}") from None
    if not isinstance(rate, int | float) or rate <= 0:
        raise LookupError(f"A1's scene-c64 report for {arm.name} served no requests")
    return float(rate)


# Family L's "32k" and "128k" prompts, in tokens (`long_loads`), and the room a load keeps for the chat template
# and the request's own head beside its prompt and reply
LONG_TOKENS = (32_000, 130_000)
LONG_MARGIN = 64


def tokens_of(args: list[str], flag: str) -> int:
    """The integer after `flag` in a load's arguments; 0 when it is absent."""
    return int(args[args.index(flag) + 1]) if flag in args else 0


def long_loads(context: int) -> tuple[list[LoadSpec], list[str]]:
    """Family L's loads that fit `context` (a prompt, its shared prefix and the reply's tokens within it), and a note
    for each load left out. Decode: one client over a prompt of 1k, 8k and 32k tokens. Chat: family I's cell at c=16,
    and at c=64 three times, each from its own first index. Prefill: every prompt distinct, one token back, so the time
    to first token is a cold prefill. Cached: a short turn after a long shared prefix, which the warm-up requests
    write into the cache. A "32k" or "128k" prompt is `LONG_TOKENS`' 32,000 or 130,000 tokens, so it fits a context of
    32,768 or 131,072 beside its reply. The loads with `--requests` end with their requests (run.py ignores
    `--warmup`, `--duration` and `--tail` for them); the long ones replace FIXED's `--timeout` (a long prompt takes
    longer than 120 s on a slow prefill)."""
    k32, k128 = (str(n) for n in LONG_TOKENS)
    decode = ["--workload", "chat", "--max-tokens", "256", "--ignore-eos", "--concurrency", "1",
              "--warmup-requests", "2"]
    prefill = ["--workload", "chat", "--max-tokens", "1", "--concurrency", "1", "--warmup-requests", "1"]
    cached = ["--workload", "chat", "--input-tokens", "128", "--max-tokens", "16", "--concurrency", "1",
              "--requests", "8", "--warmup-requests", "2"]
    long = {"--timeout": "900"}
    c64 = [LoadSpec(f"chat-c64{'' if r == 1 else f'-r{r}'}",
                    [*CHAT, "--concurrency", "64", "--first-index", str(int(FIRST["c64"][1]) + 100_000 * (r - 1))])
           for r in (1, 2, 3)]
    every = [
        LoadSpec("decode-c1-h1k", [*decode, "--input-tokens", "1024", "--requests", "8", *FIRST["h1k"]]),
        LoadSpec("decode-c1-h8k", [*decode, "--input-tokens", "8192", "--requests", "6", *FIRST["h8k"]]),
        LoadSpec("decode-c1-h32k", [*decode, "--input-tokens", k32, "--requests", "4", *FIRST["h32k"]]),
        LoadSpec("chat-c16", [*CHAT, "--concurrency", "16", *FIRST["c16"]]),
        *c64,
        LoadSpec("prefill-32k", [*prefill, "--input-tokens", k32, "--requests", "3", *FIRST["p32k"]], long),
        LoadSpec("prefill-128k", [*prefill, "--input-tokens", k128, "--requests", "3", *FIRST["p128k"]], long),
        LoadSpec("cached-32k", [*cached, "--shared-prefix-tokens", k32, *FIRST["s32k"]], long),
        LoadSpec("cached-128k", [*cached, "--shared-prefix-tokens", k128, *FIRST["s128k"]], long)]
    kept, skipped = [], []
    for spec in every:
        need = tokens_of(spec.args, "--shared-prefix-tokens") + tokens_of(spec.args, "--input-tokens")
        need += tokens_of(spec.args, "--max-tokens")
        if need + LONG_MARGIN <= context:
            kept.append(spec)
        else:
            skipped.append(f"load {spec.name} skipped: its prompt and reply take {need:,} tokens, the context is "
                           f"{context:,}")
    return kept, skipped


def loads_for(ctx: Ctx, unit: Unit) -> list[LoadSpec]:
    """The loads of a unit, in order, only those `--load` names when it names any (a named deviation). Raises
    LookupError when A2 has no A1 rate to start from."""
    loads = family_loads(ctx, unit)
    keep = getattr(ctx.args, "load", None) or []
    return [spec for spec in loads if spec.name in keep] if keep else loads


def family_loads(ctx: Ctx, unit: Unit) -> list[LoadSpec]:
    """The loads of a unit's family, in order (`loads_for`)."""
    fam, args = unit.family, ctx.args
    if fam in ("smoke", "D"):
        return []
    if fam == "A1":
        return [LoadSpec("scene-c1", [*SCENE, "--concurrency", "1", "--requests", "150", *FIRST["c1"]]),
                LoadSpec("chat-c1", [*CHAT, "--concurrency", "1", "--requests", "80", *FIRST["c1"]]),
                LoadSpec("scene-c16", [*SCENE, "--concurrency", "16", *FIRST["c16"]]),
                LoadSpec("scene-c64", [*SCENE, "--concurrency", "64", *FIRST["c64"]]),
                LoadSpec("chat-c16", [*CHAT, "--concurrency", "16", *FIRST["c16"]]),
                LoadSpec("chat-c64", [*CHAT, "--concurrency", "64", *FIRST["c64"]])]
    if fam == "A2":
        rate = a1_rate(ctx, unit.arm)
        lo, hi, res = round(0.4 * rate, 3), round(1.0 * rate, 3), round(0.1 * rate, 3)
        return [LoadSpec("find-rate", [*SCENE, "--find-rate", f"{lo:g},{hi:g}", "--resolution", f"{res:g}",
                                               "--confirm", "1", "--slots", "128", "--bound-file", "{bound}"])]
    if fam == "C":
        loads = [LoadSpec("stability", [*SCENE, "--concurrency", "16", "--requests", "500", *FIRST["c1"]])]
        if unit.passno == 1 and unit.arm.batch_invariant is not None:
            loads += [LoadSpec("scene-c64", [*SCENE, "--concurrency", "64", *FIRST["c64"]]),
                      LoadSpec("chat-c64", [*CHAT, "--concurrency", "64", *FIRST["c64"]])]
        return loads
    if fam == "I":
        return [LoadSpec("scene-c64", [*SCENE, "--concurrency", "64", *FIRST["c64"]]),
                LoadSpec("chat-c64", [*CHAT, "--concurrency", "64", *FIRST["c64"]])]
    if fam == "L":
        kept, skipped = long_loads(unit.arm.context)
        for note in skipped:
            ctx.log(f"{unit.name}: {note}")
        return kept
    if fam in ("B", "Q"):
        return [LoadSpec("scene-c1", [*SCENE, "--concurrency", "1", "--requests", "100", *FIRST["c1"]]),
                LoadSpec("scene-c64", [*SCENE, "--concurrency", "64", *FIRST["c64"]])]
    if fam == "M":
        return [LoadSpec("session", ["--workload", "chat", "--concurrency", str(args.m_concurrency), "--turns", "8",
                                     "--input-tokens", "2048", "--tool-tokens", "256", "--max-tokens", "256",
                                     "--ignore-eos", "--think", "20", "--think-sigma", "1.0"],
                         {"--warmup": "60", "--duration": "420", "--tail": "120"})]
    raise ValueError(fam)


def engine_command(ctx: Ctx, arm: arms.Arm) -> tuple[list[str], dict[str, str], str, dict[str, str]]:
    """The arm's argv, the environment the server gets, its command and the arm's own variables (the only names a
    manifest records, even with `--inherit-env`)."""
    binary = ctx.bins.get(arm.engine) or arms.BIN[arm.engine]
    arm_env = arms.env(arm, binary)
    env = {**os.environ, **arm_env} if ctx.args.inherit_env else arm_env
    return arms.argv(arm, binary), env, binary, arm_env


def warmup_requests_of(args: list[str]) -> int | None:
    """The unrecorded warm-up requests run.py takes for these arguments: an explicit --warmup-requests, else
    WARMUP_REQUESTS with --requests or --find-rate (run.py's own rule), else None: a duration-bounded load warms up
    for `warmup_s` seconds instead."""
    for i, a in enumerate(args):
        if a == "--warmup-requests" and i + 1 < len(args):
            return int(args[i + 1])
        if a.startswith("--warmup-requests="):
            return int(a.split("=", 1)[1])
    return WARMUP_REQUESTS if any(a in ("--requests", "--find-rate") for a in args) else None


def curand_note(cuda_home: str, binary: str) -> str:
    """SGLang's cuRAND headers as the manifest names them. Its speculative-sampling JIT kernel includes curand.h; the
    pod's venv step copies the headers from SGLang's venv (the binary's `bin` folder's parent) into the CUDA toolkit's
    include folder."""
    target = Path(cuda_home) / "include" / "curand.h"
    venv = Path(binary).parent.parent
    theirs = sorted(venv.glob("lib/python3*/site-packages/nvidia/cu*/include/curand.h"))
    if not target.is_file():
        return ("no curand.h in the CUDA toolkit's include folder: SGLang's speculative-sampling JIT kernel fails on a "
                "sampled request")
    mine = target.read_bytes()
    if any(p.read_bytes() == mine for p in theirs):
        return ("the cuRAND headers of SGLang's venv (nvidia/cu13/include/curand*.h, the folder as found) copied into "
                "the CUDA toolkit's include folder: its speculative-sampling JIT kernel includes curand.h, which the "
                "GPU tests' image lacks (SGLang's own image is CUDA devel)")
    return ("curand.h in the CUDA toolkit's include folder differs from SGLang's venv's"
            + (" (none found there)" if not theirs else "") + ": its speculative-sampling JIT kernel includes curand.h")


# -- the smoke checks --------------------------------------------------------------------------------------------------

def run_module(ctx: Ctx) -> Any:
    """bench/load/run.py of the client, loaded as a module (its workloads build the smoke requests)."""
    if ctx.run_module is None:
        path = ctx.client_dir / "bench/load/run.py"
        spec = importlib.util.spec_from_file_location("bench_engines_client_run", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        if str(ctx.client_dir) not in sys.path:
            sys.path.insert(0, str(ctx.client_dir))
        spec.loader.exec_module(mod)
        ctx.run_module = mod
    return ctx.run_module


def workload_body(ctx: Ctx, kind: str) -> dict[str, Any]:
    """The request body of run.py's `kind` workload at the smoke checks' index, with its defaults (chat: 1024 input
    tokens, 64 max tokens)."""
    run = run_module(ctx)
    ns = SimpleNamespace(workload=kind, tokenizer=ctx.tokenizer, model=ctx.model.id, seed=0, tool_tokens=128,
                         prompt_tokens=run.synthetic.PROMPT_TOKENS, tall_share=run.synthetic.TALL_SHARE,
                         input_tokens=1024 if kind == "chat" else 512, max_tokens=64 if kind == "chat" else None)
    old = {k: os.environ.get(k) for k in ("HF_HOME", "HF_HUB_OFFLINE")}
    os.environ.update(HF_HOME=HF_HOME, HF_HUB_OFFLINE="1")
    try:
        return run.make_workload(ns).body(SCENE_INDEX)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def chat_request(ctx: Ctx, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """One non-streamed, greedy request with thinking off: (status, the reply's JSON or {})."""
    req = {**body, "model": ctx.model.id, "temperature": 0, "stream": False,
           "chat_template_kwargs": {"enable_thinking": False}}
    status, data = http(ctx.base_url + "/v1/chat/completions", req, timeout=600)
    try:
        return status, json.loads(data) if status == 200 else {}
    except ValueError:
        return status, {}


def message(reply: dict[str, Any]) -> dict[str, Any]:
    try:
        return reply["choices"][0]["message"] or {}
    except (KeyError, IndexError, TypeError):
        return {}


def smoke(ctx: Ctx, arm: arms.Arm) -> dict[str, Any]:
    """The checks every unit passes before any load: the scene request twice (the second must hit the prefix cache),
    its reply (valid, compact JSON), a chat request, no reasoning in any reply, and verify steps above zero."""
    t0 = time.monotonic()
    fails: list[str] = []
    out: dict[str, Any] = dict.fromkeys(
        ("ok", "failures", "scene_prompt_tokens", "scene_completion_tokens", "scene_cached_tokens_second",
         "scene_valid", "scene_compact", "chat_prompt_tokens", "chat_completion_tokens", "reasoning_free",
         "verify_steps"))
    from llm_mini_infer import hostcheck

    try:
        scene_body = workload_body(ctx, "scene")
        chat_body = workload_body(ctx, "chat")
    except Exception as exc:  # the tokenizer or the workload cannot be built
        out.update(ok=False, failures=[f"cannot build the requests: {type(exc).__name__}"], seconds=0.0)
        return out
    before = hostcheck.prometheus(http(ctx.base_url + "/metrics", timeout=30)[1].decode(errors="replace"))
    s1, r1 = chat_request(ctx, scene_body)
    s2, r2 = chat_request(ctx, scene_body)
    s3, r3 = chat_request(ctx, chat_body)
    after = hostcheck.prometheus(http(ctx.base_url + "/metrics", timeout=30)[1].decode(errors="replace"))
    for name, status in (("scene", s1), ("scene (second)", s2), ("chat", s3)):
        if status != 200:
            fails.append(f"{name} request: HTTP {status}" if status else f"{name} request: no answer")
    u1, u2, u3 = (r.get("usage") or {} for r in (r1, r2, r3))
    out.update(scene_prompt_tokens=u1.get("prompt_tokens"), scene_completion_tokens=u1.get("completion_tokens"),
               chat_prompt_tokens=u3.get("prompt_tokens"), chat_completion_tokens=u3.get("completion_tokens"))
    cached = (u2.get("prompt_tokens_details") or {}).get("cached_tokens") if s2 == 200 else None
    out["scene_cached_tokens_second"] = cached
    if s2 == 200:
        if isinstance(cached, int) and cached <= 0:
            fails.append("the second scene request reports no cached prompt tokens")
        elif cached is None and arm.engine == "ours":
            fails.append("llm-mini-infer does not report cached prompt tokens")
    if s1 == 200:
        content = message(r1).get("content") or ""
        try:
            doc = json.loads(content)
        except ValueError:
            doc = None
            out["scene_valid"] = False
            fails.append("the scene reply is not JSON")
        else:
            out["scene_compact"] = json.dumps(doc, separators=(",", ":"), ensure_ascii=False) == content
            if not out["scene_compact"]:
                fails.append("the scene reply is not compact JSON")
            try:
                import jsonschema

                jsonschema.Draft202012Validator(scene_body["response_format"]["json_schema"]["schema"]).validate(doc)
                out["scene_valid"] = True
            except ImportError:
                out["scene_valid"] = None
            except Exception:  # a validation error: its text may quote the reply
                out["scene_valid"] = False
                fails.append("the scene reply does not validate against the schema")
    msgs = [message(r) for r, s in ((r1, s1), (r2, s2), (r3, s3)) if s == 200]
    bad = [m for m in msgs if m.get("reasoning_content") or m.get("reasoning")
           or (m.get("content") or "").lstrip().startswith("<think") or "<|channel>" in (m.get("content") or "")]
    out["reasoning_free"] = not bad if msgs else None
    if bad:
        fails.append("a reply holds reasoning")
    a, b = hostcheck.counter(before, hostcheck.VERIFY_SERIES), hostcheck.counter(after, hostcheck.VERIFY_SERIES)
    if b is None:
        fails.append("/metrics exports no verify-step series")
    else:
        steps = b[1] - (a[1] if a and a[0] == b[0] else 0.0)
        out["verify_steps"] = steps
        if steps <= 0:
            fails.append("no verify steps over the smoke requests")
    out.update(ok=not fails, failures=fails, seconds=round(time.monotonic() - t0, 2))
    return out


# -- one unit ----------------------------------------------------------------------------------------------------------

@dataclass
class Outcome:
    status: str = "ok"
    status_line: str | None = None
    startup: dict[str, Any] = field(default_factory=dict)
    starts: list[dict] = field(default_factory=list)
    resolved: dict[str, Any] = field(default_factory=dict)
    smoke: dict[str, Any] | None = None
    loads: list[dict] = field(default_factory=list)
    stopped: dict | None = None


def scrub_wiped(items: list[dict]) -> list[dict]:
    return [{"removed": i.get("removed")} for i in items]


def scrub_evicted(items: list[dict]) -> list[dict]:
    return [{k: v for k, v in i.items() if k != "dir"} for i in items]


def start_server(ctx: Ctx, unit: Unit, argv: list[str], env: dict[str, str], log_path: Path) -> tuple[Any, dict]:
    """Launch the arm's server and wait for readiness and its first answer. Stops (raises SystemExit) when anything
    listens on the port already. Raises StartError when the start fails."""
    if startup._port_open(arms.HOST, arms.PORT):
        ctx.log(f"{unit.name}: something listens on port {arms.PORT} already; not starting beside it")
        raise SystemExit(3)
    ctx.log(f"{unit.name}: start {' '.join(startup.redact(argv))[:400]}")
    h = startup.launch(argv, env=env, cwd="/", log_path=log_path, base_url=ctx.base_url)
    rec: dict[str, Any] = {}
    try:
        rec["ready_s"] = startup.wait_ready(h, "/health", START_TIMEOUT_S)
        rec["first_answer_s"] = startup.first_answer(h, ctx.model.id, START_TIMEOUT_S)
    except BaseException:  # a failed start, or an interrupt: no server is left running
        startup.stop(h)
        raise
    ctx.log(f"{unit.name}: ready in {rec['ready_s']} s, first answer at {rec['first_answer_s']} s")
    return h, rec


def classify_failure(unit: Unit, err: startup.StartError, log_path: Path) -> tuple[str, str]:
    """(status, status_line) of a start that failed: `unsupported` for a documented refusal, else `failed`."""
    try:
        lines = log_path.read_text(errors="replace").splitlines()
    except OSError:
        lines = []
    if unit.arm.engine == "vllm" and arms.batch_invariant(unit.arm):
        for ln in lines:
            if any(p in ln for p in EXPECTED_REFUSALS):
                return "unsupported", clean_line(ln)
    return "failed", error_line(lines) or clean_line(str(err))


def run_load(ctx: Ctx, unit: Unit, spec: LoadSpec, udir: Path, bound: Path | None, sampler: gpu_control.Sampler,
             meta: dict) -> dict:
    name = spec.name
    report = udir / f"{name}.json"
    report.unlink(missing_ok=True)  # a stale report must not pass for this run's
    save_metrics(ctx, udir / f"metrics-{name}-before.txt")
    fixed = {**FIXED, **spec.fixed}
    cmd = [sys.executable, str(ctx.client_dir / "bench/load/run.py"), "--base-url", ctx.base_url, "--model",
           ctx.model.id, "--tokenizer", ctx.tokenizer, "--out", str(report), "--meta", json.dumps(meta),
           *[str(bound) if a == "{bound}" else a for a in spec.args], *[x for kv in fixed.items() for x in kv]]
    ctx.log(f"{unit.name}: load {name}")
    t0 = time.monotonic()
    code, timed_out = run_child(cmd, client_env(ctx), ctx.client_dir, ctx.out / "drive.log", LOAD_LIMIT_S)
    t1 = time.monotonic()
    save_metrics(ctx, udir / f"metrics-{name}-after.txt")
    ctx.log(f"{unit.name}: load {name} exited {code} after {t1 - t0:.0f} s" + (" (timed out)" if timed_out else ""))
    error = report_failure(code, timed_out, report)
    client = {"seed": int(fixed["--seed"]), "warmup_s": float(fixed["--warmup"]),
              "duration_s": float(fixed["--duration"]), "tail_s": float(fixed["--tail"]),
              "timeout_s": float(fixed["--timeout"]),
              "warmup_requests": warmup_requests_of([*spec.args, *[x for kv in fixed.items() for x in kv]])}
    return {"name": name, "file": f"{name}.json", "exit": code, "seconds": round(t1 - t0, 1),
            **gpu_control.summarize(sampler.samples, t0, t1), "ok": error is None,
            **({"error": error} if error else {}), "client": client}


def report_failure(code: int, timed_out: bool, report: Path) -> str | None:
    """Why a client child failed, else None. Exit 1 is a result (some requests or answers failed), not a crash; a
    negative code is a signal; the report must exist and be a JSON object."""
    if timed_out:
        return "timed out"
    if code < 0:
        return f"killed by signal {-code}"
    if code not in (0, 1):
        return f"exit {code}"
    try:
        doc = json.loads(report.read_text())
    except OSError:
        return "wrote no report"
    except ValueError:
        return "report is not JSON"
    return None if isinstance(doc, dict) else "report is not a JSON object"


def run_unit(ctx: Ctx, unit: Unit, order: list[str]) -> str:
    """One unit, start to manifest. Returns its status."""
    arm, model = unit.arm, ctx.model
    udir = unit_dir(ctx, unit.family, unit.name)
    udir.mkdir(parents=True, exist_ok=True)
    log_path = udir / "server.log"
    for stale in ("server.log", "gpu.csv", "done"):
        (udir / stale).unlink(missing_ok=True)
    argv, env, binary, arm_env = engine_command(ctx, arm)
    t_start = time.monotonic()
    oc = Outcome()
    sampler = gpu_control.Sampler(1000, udir / "gpu.csv")
    sampler.start()
    h = None
    bound: Path | None = None
    meta = {"engine": arm.engine, "arm": arm.name, "family": unit.family, "model": model.id, "k": arm.draft_k,
            "batch_invariant": arms.batch_invariant(arm), "unit": unit.name}
    try:
        try:
            loads = loads_for(ctx, unit)
        except LookupError as exc:
            oc.status, oc.status_line = "failed", str(exc)
            ctx.log(f"{unit.name}: {exc}")
            loads = []
        else:
            if unit.family == "A2":
                bound = udir / "bound.json"
                shutil.copyfile(ctx.args.bound_file, bound)
        if oc.status == "ok":
            h = unit_session(ctx, unit, udir, log_path, argv, env, loads, bound, sampler, meta, oc)
    except startup.StartError as err:
        oc.status, oc.status_line = classify_failure(unit, err, log_path)
        ctx.log(f"{unit.name}: start failed ({oc.status}): {oc.status_line}")
    except ValueError as exc:  # the cold start's wipe refused a directory
        oc.status, oc.status_line = "failed", clean_line(str(exc))
    finally:
        if h is not None:
            oc.stopped = startup.stop(h)
        sampler.stop()
    if unit.family == "D" and arm.engine == "ours" and oc.status == "ok":
        oc.loads.append(gc := grammar_cost(ctx, unit, udir))
        if not gc["ok"]:
            oc.status, oc.status_line = "failed", f"load grammar-cost: {gc['error']}"
            ctx.log(f"{unit.name}: {oc.status_line}")
    if oc.status == "crashed":
        priv = ctx.out / "private"
        priv.mkdir(parents=True, exist_ok=True)
        for src, dst in ((log_path, f"{model.key}-{arm.name}-server.log"),
                         (udir / "validity.json", f"{model.key}-{arm.name}-validity.json")):
            if src.exists():
                shutil.move(str(src), str(priv / dst))
    write_unit(ctx, unit, udir, log_path, binary, env, arm_env, oc, order)
    seconds = round(time.monotonic() - t_start, 1)
    if oc.status in ("ok", "unsupported"):
        write_json(udir / "done", {"status": oc.status, "seconds": seconds})
    ctx.log(f"{unit.name}: {oc.status} after {seconds:.0f} s" + (f" ({oc.status_line})" if oc.status_line else ""))
    return oc.status


def unit_session(ctx: Ctx, unit: Unit, udir: Path, log_path: Path, argv: list[str], env: dict[str, str],
                 loads: list[LoadSpec], bound: Path | None, sampler: gpu_control.Sampler, meta: dict,
                 oc: Outcome) -> Any:
    """The start(s), smoke checks, resolved settings, loads and liveness check of a unit. Fills `oc`; returns the
    running server's handle (the caller stops it)."""
    arm = unit.arm
    h = None
    if unit.family == "D":
        wiped = startup.wipe(arms.cache_roots(arm.engine))
        repos = [ctx.model.id, *([ctx.model.drafter] if ctx.model.drafter else [])]
        evicted = startup.evict([str(model_dir(r)) for r in repos])
        oc.startup.update(wiped=scrub_wiped(wiped), evicted=scrub_evicted(evicted))
        h, rec = start_server(ctx, unit, argv, env, log_path)
        cold = {**rec, "stopped": startup.stop(h)}
        h = None
        write_json(udir / "startup-cold.json", cold)
        oc.starts.append({"kind": "cold", **cold})
        oc.startup["cold"] = {"ready_s": rec["ready_s"], "first_answer_s": rec["first_answer_s"]}
    h, rec = start_server(ctx, unit, argv, env, log_path)
    kind = "warm" if unit.family == "D" else "start"
    oc.starts.append({"kind": kind, **rec})
    if unit.family == "D":
        write_json(udir / "startup-warm.json", rec)
    oc.startup.update(ready_s=rec["ready_s"], first_answer_s=rec["first_answer_s"])
    write_json(udir / "startup.json", {"starts": oc.starts, **{k: oc.startup.get(k) for k in ("wiped", "evicted")}})
    oc.smoke = smoke(ctx, arm)
    write_json(udir / "smoke.json", oc.smoke)
    if not oc.smoke["ok"]:
        oc.status, oc.status_line = "smoke-failed", "; ".join(oc.smoke["failures"])[:300]
        ctx.log(f"{unit.name}: smoke failed: {oc.status_line}")
        return h
    oc.resolved = resolved_settings(ctx, arm, log_path)
    for spec in loads:
        oc.loads.append(ld := run_load(ctx, unit, spec, udir, bound, sampler, meta))
        if not ld["ok"]:
            oc.status, oc.status_line = "failed", f"load {spec.name}: {ld['error']}"
            ctx.log(f"{unit.name}: {oc.status_line}")
            return h
    if unit.family == "D" and not ctx.args.skip_validity:
        run_validity(ctx, unit, udir, h, sampler, oc)
    if oc.status == "ok" and not alive(h, ctx):
        oc.status, oc.status_line = "crashed", "the server died or stopped answering /health after the loads"
    return h


def run_validity(ctx: Ctx, unit: Unit, udir: Path, h: Any, sampler: gpu_control.Sampler, oc: Outcome) -> None:
    cmd = [sys.executable, str(ctx.client_dir / "bench/structured/validity.py"), "--base-url", ctx.base_url,
           "--model", ctx.model.id, "--out", str(udir / "validity.json"), "--timeout", "120"]
    ctx.log(f"{unit.name}: validity")
    t0 = time.monotonic()
    code, timed_out = run_child(cmd, client_env(ctx), ctx.client_dir, ctx.out / "drive.log", VALIDITY_LIMIT_S)
    t1 = time.monotonic()
    error = report_failure(code, timed_out, udir / "validity.json")
    oc.loads.append({"name": "validity", "file": "validity.json", "exit": code, "seconds": round(t1 - t0, 1),
                     **gpu_control.summarize(sampler.samples, t0, t1), "ok": error is None,
                     **({"error": error} if error else {}), "client": None})
    if timed_out:
        oc.status, oc.status_line = "crashed", f"validity.py did not finish within {VALIDITY_LIMIT_S} s"
    elif not alive(h, ctx):
        oc.status, oc.status_line = "crashed", "the server died or stopped answering /health during the validity run"
    elif error:
        oc.status, oc.status_line = "failed", f"load validity: {error}"
        ctx.log(f"{unit.name}: {oc.status_line}")


def grammar_cost(ctx: Ctx, unit: Unit, udir: Path) -> dict:
    cmd = [sys.executable, str(ctx.client_dir / "bench/structured/grammar_cost.py"), "--backend", "guidance",
           "--compact", "--catalog", "--k", str(unit.arm.draft_k), "--out", str(udir / "grammar-cost.json")]
    if ctx.args.model == "qwen":
        cmd += ["--model", "tests/.fixtures/models/qwen3.8-27b"]
    ctx.log(f"{unit.name}: grammar cost")
    (udir / "grammar-cost.json").unlink(missing_ok=True)
    t0 = time.monotonic()
    code, timed_out = run_child(cmd, client_env(ctx), ctx.client_dir, ctx.out / "drive.log", 3600)
    error = report_failure(code, timed_out, udir / "grammar-cost.json")
    return {"name": "grammar-cost", "file": "grammar-cost.json", "exit": code,
            "seconds": round(time.monotonic() - t0, 1), "sm_mhz_median": None, "power_w_median": None, "throttle": [],
            "ok": error is None, **({"error": error} if error else {}), "client": None}


def resolved_settings(ctx: Ctx, arm: arms.Arm, log_path: Path) -> dict[str, Any]:
    """The engine's resolved settings: llm-mini-infer's from `/server_info`, the others' from the server log."""
    if arm.engine == "ours":
        status, data = http(ctx.base_url + "/server_info", timeout=30)
        try:
            lines = log_path.read_text(errors="replace").splitlines()
        except OSError:
            lines = []
        try:
            return resolve_ours(json.loads(data), lines) if status == 200 else dict.fromkeys(RESOLVED_FIELDS)
        except ValueError:
            return dict.fromkeys(RESOLVED_FIELDS)
    try:
        lines = log_path.read_text(errors="replace").splitlines()
    except OSError:
        lines = []
    return parse_log(arm.engine, lines)


def write_unit(ctx: Ctx, unit: Unit, udir: Path, log_path: Path, binary: str, env: dict[str, str],
               arm_env: dict[str, str], oc: Outcome, order: list[str]) -> None:
    arm, model = unit.arm, ctx.model
    source = "vendor image" if ctx.args.inherit_env else ("image venv" if arm.engine == "ours" else "venv")
    if not (udir / "startup.json").exists():
        write_json(udir / "startup.json", {"starts": oc.starts, "wiped": None, "evicted": None})
    deviations = list(arm.deviations)
    if getattr(ctx.args, "load", None):
        deviations.append(f"only the loads {', '.join(ctx.args.load)} (--load)")
    if unit.family == "L":
        deviations += long_loads(arm.context)[1]
    if unit.family == "D" and ctx.args.skip_validity:
        deviations.append("no structured-output validity run (--skip-validity)")
    if arm.engine == "sglang" and not ctx.args.inherit_env:  # a vendor image has no venv to compare with
        deviations.append(curand_note(arm_env.get("CUDA_HOME", "/usr/local/cuda"), binary))
    manifest: dict[str, Any] = {
        "schema": 1, "arm": arm.name, "family": unit.family, "engine": arm.engine, "model": model.id,
        "drafter": model.drafter,
        "revisions": {r: revision(r) for r in (model.id, *([model.drafter] if model.drafter else []))},
        "k": arm.draft_k, "k_source": arms.k_source(arm), "batch_invariant": arms.batch_invariant(arm),
        "batch_invariant_source": "default" if arm.batch_invariant is None else "set",
        "argv": startup.redact(arms.argv(arm, binary)), "env_names": sorted(arm_env),
        "env_inherited": bool(ctx.args.inherit_env), "deviations": deviations,
        "status": oc.status, "status_line": oc.status_line, "versions": package_versions(binary, env),
        "source": source, "resolved": oc.resolved, "smoke": oc.smoke,
        "startup": {"ready_s": oc.startup.get("ready_s"), "first_answer_s": oc.startup.get("first_answer_s"),
                    "cold": oc.startup.get("cold"), "wiped": oc.startup.get("wiped"),
                    "evicted": oc.startup.get("evicted"), "stopped": oc.stopped},
        "loads": oc.loads, "host": host_block(),
        "client": {"python": platform.python_version()},
        "order": order, "when": time.strftime("%Y-%m"), **ctx.meta_extra}
    write_json(udir / "manifest.json", manifest)


# -- the families ------------------------------------------------------------------------------------------------------

def compare_passes(ctx: Ctx, arm: arms.Arm) -> None:
    """Family C: after both passes of an arm, compare.py over their `stability.json` reports."""
    p1, p2 = (unit_dir(ctx, "C", f"{arm.name}-p{n}") / "stability.json" for n in (1, 2))
    if not (p1.exists() and p2.exists()):
        return
    out = ctx.out / ctx.args.model / "C" / f"{arm.name}-compare.txt"
    cmd = [sys.executable, str(ctx.client_dir / "bench/load/compare.py"), f"p1={p1}", f"p2={p2}"]
    res = subprocess.run(cmd, cwd=ctx.client_dir, env=client_env(ctx), capture_output=True, text=True, timeout=600)
    out.write_text(res.stdout)
    ctx.log(f"compare {arm.name}: exit {res.returncode}")


def dry_run(ctx: Ctx, units: list[Unit]) -> None:
    for u in units:
        argv, _, _, arm_env = engine_command(ctx, u.arm)
        print(f"{u.family} {u.name}: {' '.join(startup.redact(argv))}")
        print(f"  env: {' '.join(sorted(arm_env))}")
        try:
            for spec in loads_for(ctx, u):
                print(f"  load {spec.name}: {' '.join(spec.args)}")
        except LookupError as exc:
            print(f"  loads: {exc}")


def run_control(args: argparse.Namespace) -> int:
    if not args.label:
        raise SystemExit("--family control needs --label start|middle|end")
    out = Path(args.out) / "control" / f"{args.label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    py = args.python or sys.executable
    with open(Path(args.out) / "drive.log", "a") as log:
        log.write(f"{time.strftime('%H:%M:%S')} control {args.label}\n")
    code, _ = run_child([py, str(HERE / "gpu_control.py"), "--out", str(out)], dict(os.environ), ROOT,
                        Path(args.out) / "drive.log", 900)
    return 0 if code == 0 else 1


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", choices=sorted(arms.MODELS))
    ap.add_argument("--family", required=True, choices=FAMILIES)
    ap.add_argument("--engines", default=",".join(arms.ENGINES))
    ap.add_argument("--rotate", type=int, default=0, help="rotate the engine order by N")
    ap.add_argument("--arm", action="append", default=[], help="keep only this arm (repeatable)")
    ap.add_argument("--engine-bin", action="append", default=[], metavar="ENGINE=PATH",
                    help="the engine's command (default: bench/engines/arms.py's BIN)")
    ap.add_argument("--inherit-env", action="store_true",
                    help="start the engines with this process's whole environment under the arm's variables")
    ap.add_argument("--client-pythonpath", metavar="DIR",
                    help="run the clients from DIR with PYTHONPATH=DIR (a vendor image without the repository)")
    ap.add_argument("--bound-file", help="family A2: the latency bound")
    ap.add_argument("--meta-extra", default="{}", help="JSON merged into every manifest")
    ap.add_argument("--m-concurrency", type=int, default=48, help="family M: conversations in flight")
    ap.add_argument("--load", action="append", default=[], metavar="NAME",
                    help="run only this load of the family (repeatable; a named deviation)")
    ap.add_argument("--context", type=int, default=arms.CONTEXT, help="the context every arm serves (default: "
                    "%(default)s); family L's loads that do not fit it are skipped")
    ap.add_argument("--tokenizer", help="the clients' tokenizer: a folder or repo id (default: the model's repo id)")
    ap.add_argument("--label", choices=("start", "middle", "end"), help="family control: the file's name")
    ap.add_argument("--python", help="family control: the interpreter that runs the control")
    ap.add_argument("--attention", action="append", default=[], metavar="ARM=BACKEND",
                    help="force an arm's attention backend (a documented fallback; a named deviation)")
    ap.add_argument("--drop", action="append", default=[], metavar="ARM=FLAG",
                    help="leave a flag out of an arm's command (a documented fallback; a named deviation)")
    ap.add_argument("--add-arg", action="append", default=[], metavar="ARM=ARG",
                    help="add one argument to the end of an arm's command (a recipe's flag; a named deviation)")
    ap.add_argument("--deviation", action="append", default=[], metavar="ARM=TEXT",
                    help="the reason for an arm's deviation, into its manifest")
    ap.add_argument("--skip-validity", action="store_true",
                    help="family D: no structured-output validity run (a named deviation)")
    ap.add_argument("--dry-run", action="store_true", help="print each unit's command and environment names only")
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = parser()
    args = ap.parse_args(argv)
    if args.family == "control":
        return run_control(args)
    if not args.model:
        ap.error("--model is required")
    if args.family == "A2" and not args.bound_file:
        ap.error("--family A2 needs --bound-file")
    if args.family == "A2" and not Path(args.bound_file).is_file():
        ap.error(f"--bound-file {args.bound_file}: no such file")
    try:
        meta_extra = json.loads(args.meta_extra)
        assert isinstance(meta_extra, dict)
    except (ValueError, AssertionError):
        ap.error("--meta-extra: a JSON object")
    bins = {}
    for item in args.engine_bin:
        eng, sep, path = item.partition("=")
        if not sep or eng not in arms.ENGINES:
            ap.error(f"--engine-bin {item}: ENGINE=PATH with ENGINE one of {', '.join(arms.ENGINES)}")
        bins[eng] = path
    client_dir = Path(args.client_pythonpath).resolve() if args.client_pythonpath else ROOT
    if args.client_pythonpath:
        sys.path.insert(0, str(client_dir))
    model = arms.MODELS[args.model]
    ctx = Ctx(args, model, Path(args.out), f"http://{arms.HOST}:{arms.PORT}", client_dir, args.tokenizer or model.id,
              bins, meta_extra)
    units = build_units(args)
    if args.dry_run:
        dry_run(ctx, units)
        return 0
    order = [u.name for u in units]
    ctx.log(f"{args.model} {args.family}: {len(units)} units: {', '.join(order)}")
    statuses: dict[str, str] = {}
    done_arms: set[str] = set()
    for u in units:
        if (unit_dir(ctx, u.family, u.name) / "done").exists():
            ctx.log(f"{u.name}: done already; skipped")
            statuses[u.name] = "skipped"
        else:
            statuses[u.name] = run_unit(ctx, u, order)
        if u.family == "C" and u.passno == 2 and u.arm.name not in done_arms:
            done_arms.add(u.arm.name)
            compare_passes(ctx, u.arm)
    ctx.log("units: " + ", ".join(f"{n} {s}" for n, s in statuses.items()))
    return 0 if all(s in ("ok", "unsupported", "skipped") for s in statuses.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
