"""The benchmark's driver (bench/engines/drive.py) and the GPU control (bench/engines/gpu_control.py) against a fake
engine: a stdlib HTTP server run as an arm's command. No GPU, no model download, no network."""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import socket
import stat
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from ..support.paths import ROOT


def _load(name: str, rel: str):
    if name in sys.modules:
        return sys.modules[name]
    sys.path.insert(0, str(ROOT / "bench/engines"))
    sys.path.insert(0, str(ROOT / "bench/load"))
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


drive = _load("bench_engines_drive", "bench/engines/drive.py")
gpu_control = drive.gpu_control
arms = drive.arms
check = _load("bench_reports_check_for_engines", "bench/reports/check.py")
ORIGINAL_SPAWN = gpu_control.Sampler._spawn
DAY = "-".join(("2000", "01", "01"))  # a fixed date for the fake logs

FAKE = r'''
import hashlib
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

here = Path(__file__).resolve().parent
args = sys.argv[1:]
port = int(args[args.index("--port") + 1]) if "--port" in args else next(
    (int(a.split("=", 1)[1]) for a in args if a.startswith("server.port=")), 8000)  # vLLM's and SGLang's, or ours
mode = (here / "mode").read_text().strip()
if mode == "gdn":
    print("ValueError: batch_invariant mode is not supported for GDN_ATTN", flush=True)
    sys.exit(1)
if mode == "boom":
    print("RuntimeError: boom at /opt/vllm/lib/engine.py line 3", flush=True)
    sys.exit(1)
print("(EngineCore pid=1) INFO 11-12 10:00:00 [kv_cache_utils.py:1147] GPU KV cache size: 1,234,560 tokens, "
      "Maximum concurrency for 8,192 tokens per request: 150.20x", flush=True)
scene = (here / "scene.json").read_text()
lock = threading.Lock()
state = {"drafts": 0, "tokens": 0}
seen = set()


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            self.send(200, "{}")
        elif self.path == "/metrics":
            with lock:
                page = ('# TYPE vllm:spec_decode_num_drafts_total counter\n'
                        f'vllm:spec_decode_num_drafts_total{{engine="0"}} {state["drafts"]}\n'
                        if mode != "nospec" else '') + f'vllm:generation_tokens_total{{engine="0"}} {state["tokens"]}\n'
            self.send(200, page, "text/plain")
        elif self.path == "/server_info":
            self.send(200, json.dumps({"config": {}}))
        else:
            self.send(404, "{}")

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        text = json.dumps(body["messages"])
        key = hashlib.sha256(text.encode()).hexdigest()
        n_prompt = len(text) // 4
        with lock:
            cached = n_prompt // 2 if key in seen else 0
            seen.add(key)
        if "response_format" in body:
            content = scene
            if mode == "spaces":
                content = json.dumps(json.loads(scene), separators=(", ", ": "))
        else:
            content = "ok " * min(body.get("max_tokens", 8), 8)
        n_out = max(2, min(body.get("max_tokens", 8), 8))
        with lock:
            state["drafts"] += n_out // 2 + 1
            state["tokens"] += n_out
        usage = {"prompt_tokens": n_prompt, "completion_tokens": n_out, "total_tokens": n_prompt + n_out,
                 "prompt_tokens_details": {"cached_tokens": cached}}
        if not body.get("stream"):
            msg = {"role": "assistant", "content": content}
            if mode == "reasoning":
                msg["reasoning_content"] = "hmm"
            self.send(200, json.dumps({"choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
                                       "usage": usage}))
            return
        chunks = [{"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]},
                  {"choices": [{"index": 0, "delta": {"content": content}}]},
                  {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                  {"choices": [], "usage": usage}]
        data = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
        self.send(200, data, "text/event-stream")


ThreadingHTTPServer.daemon_threads = True
ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
'''

NVIDIA_SMI = """#!{python}
import sys, time
a = sys.argv[1:]
if a and a[0].startswith("--query-gpu=name"):
    print("Test GPU, 300, 600, 3000, 96000, 0, 570.1")
elif a and a[0].startswith("--query-gpu=timestamp"):
    while True:
        print("{day} 00:00:00.000, 301.5, 2400, 1200, 1000, 55, 0x0000000000000004", flush=True)
        time.sleep(0.2)
else:
    print("| NVIDIA-SMI 570.1   Driver Version: 570.1   CUDA Version: 13.0 |")
"""


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Rig:
    def __init__(self, tmp: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from tokenizers import Tokenizer, models, pre_tokenizers

        self.tmp, self.mp = tmp, monkeypatch
        self.out = tmp / "out"
        self.work = tmp / "fake"
        self.work.mkdir()
        (self.work / "fake.py").write_text(FAKE)
        (self.work / "mode").write_text("ok")
        # a minimal document of the scene schema: every setting attribute null
        from llm_mini_infer import synthetic

        schema = synthetic.scene_schema()
        doc = {"count": 0, "objects": [], "setting": dict.fromkeys(schema["$defs"]["Setting"]["properties"])}
        (self.work / "scene.json").write_text(json.dumps(doc, separators=(",", ":"), ensure_ascii=False))
        self.bin = self.work / "fake-engine"
        self.bin.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{self.work / "fake.py"}" "$@"\n')
        self.bin.chmod(self.bin.stat().st_mode | stat.S_IXUSR)
        self.tok = tmp / "tok"
        self.tok.mkdir()
        tok = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
        tok.pre_tokenizer = pre_tokenizers.Whitespace()
        tok.save(str(self.tok / "tokenizer.json"))
        self.port = free_port()
        monkeypatch.setattr(arms, "PORT", self.port)
        monkeypatch.setattr(drive, "HF_HOME", str(tmp / "hf"))
        monkeypatch.setattr(gpu_control.Sampler, "_spawn", lambda self, fields: None)  # no nvidia-smi
        monkeypatch.setattr(drive, "loads_for", self.tiny_loads)

    def mode(self, text: str) -> None:
        (self.work / "mode").write_text(text)

    def tiny_loads(self, ctx, unit):
        if unit.family in ("smoke", "D"):
            return []
        first = "stability" if unit.family == "C" else "scene-c2"
        return [drive.LoadSpec(first, [*drive.SCENE, "--concurrency", "2", "--requests", "6", "--warmup-requests",
                                            "2", "--prompt-tokens", "64", "--max-tokens", "16"]),
                drive.LoadSpec("chat-c2", ["--workload", "chat", "--input-tokens", "32", "--max-tokens", "8",
                                           "--concurrency", "2", "--requests", "6", "--warmup-requests", "2",
                                           "--ignore-eos"])]

    def run(self, *extra: str, model: str = "gemma", family: str = "smoke", engines: str = "vllm") -> int:
        arm = [] if "--arm" in extra else ["--arm", "vllm"]  # the head arm, unless the test names one
        return drive.main(["--out", str(self.out), "--model", model, "--family", family, "--engines", engines,
                           "--engine-bin", f"vllm={self.bin}", "--tokenizer", str(self.tok), *arm, *extra])

    def unit(self, name: str = "vllm", family: str = "smoke", model: str = "gemma") -> Path:
        return self.out / model / family / name

    def manifest(self, name: str = "vllm", family: str = "smoke", model: str = "gemma") -> dict:
        return json.loads((self.unit(name, family, model) / "manifest.json").read_text())


@pytest.fixture
def rig(tmp_path, monkeypatch):
    return Rig(tmp_path, monkeypatch)


def findings(rig: Rig, manifest: Path) -> list:
    folder = rig.tmp / "published" / "manifests"
    folder.mkdir(parents=True, exist_ok=True)
    copy = folder / "m.json"
    copy.write_text(manifest.read_text())
    return check.check_file(copy, rig.tmp / "published")


def test_smoke_unit_writes_its_files_and_a_checked_manifest(rig, monkeypatch, capsys):
    monkeypatch.setenv("SECRET_PROBE", "hunter2-secret-value")
    assert rig.run("--inherit-env", "--meta-extra", '{"image": "sha256:abc", "commit": "' + "a" * 40 + '"}') == 0
    u = rig.unit()
    assert {p.name for p in u.iterdir()} >= {"startup.json", "smoke.json", "manifest.json", "done", "server.log"}
    assert not list(u.glob("scene-*.json"))  # the smoke family runs no load
    smoke = json.loads((u / "smoke.json").read_text())
    assert smoke["ok"] and smoke["scene_valid"] is True and smoke["scene_compact"] is True
    assert smoke["scene_cached_tokens_second"] > 0 and smoke["verify_steps"] > 0 and smoke["reasoning_free"]
    m = rig.manifest()
    assert m["status"] == "ok" and m["source"] == "vendor image" and m["order"] == ["vllm"]
    assert m["resolved"]["kv_tokens"] == 1234560 and m["resolved"]["spec"] is None
    assert m["argv"][0] == "fake-engine" and m["image"] == "sha256:abc" and m["k"] == 5
    assert "SECRET_PROBE" not in m["env_names"] and "PATH" in m["env_names"] and m["env_inherited"] is True
    assert findings(rig, u / "manifest.json") == []
    done = json.loads((u / "done").read_text())
    assert done["status"] == "ok"
    for p in rig.out.rglob("*"):  # names only: no value of the environment reaches a file
        if p.is_file():
            assert "hunter2-secret-value" not in p.read_text(errors="replace"), p
    assert "hunter2-secret-value" not in capsys.readouterr().out
    # a second run skips the finished unit
    stamp = (u / "manifest.json").stat().st_mtime_ns
    assert rig.run("--inherit-env") == 0
    assert (u / "manifest.json").stat().st_mtime_ns == stamp
    assert "done already" in (rig.out / "drive.log").read_text()


def test_load_unit_writes_reports_metrics_and_gpu_samples(rig, monkeypatch):
    bindir = rig.tmp / "smi"
    bindir.mkdir()
    smi = bindir / "nvidia-smi"
    smi.write_text(NVIDIA_SMI.format(python=sys.executable, day=DAY))
    smi.chmod(smi.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setattr(gpu_control.Sampler, "_spawn", ORIGINAL_SPAWN)  # here the real sampler reads the fake
    assert rig.run(family="A1") == 0
    u = rig.unit(family="A1")
    for load in ("scene-c2", "chat-c2"):
        report = json.loads((u / f"{load}.json").read_text())
        assert report["loads"] and report["meta"]["unit"] == "vllm" and report["meta"]["family"] == "A1"
        assert (u / f"metrics-{load}-before.txt").read_text().count("vllm:spec_decode_num_drafts_total") >= 1
        assert (u / f"metrics-{load}-after.txt").exists()
    assert (u / "gpu.csv").read_text().splitlines()[0].startswith("mono_s,timestamp")
    m = rig.manifest(family="A1")
    assert [ld["name"] for ld in m["loads"]] == ["scene-c2", "chat-c2"]
    assert all(ld["exit"] == 0 and ld["sm_mhz_median"] == 2400.0 and ld["throttle"] == [4] for ld in m["loads"])
    assert m["host"]["gpu"] == "Test GPU" and m["host"]["cuda"] == "13.0"
    assert findings(rig, u / "manifest.json") == []
    assert (u / "done").exists()


def test_no_nvidia_smi_the_unit_still_completes(rig):
    assert rig.run(family="A1") == 0
    assert not (rig.unit(family="A1") / "gpu.csv").exists()
    m = rig.manifest(family="A1")
    assert m["status"] == "ok" and all(ld["sm_mhz_median"] is None and ld["throttle"] == [] for ld in m["loads"])


def test_gdn_refusal_is_unsupported_with_done(rig):
    rig.mode("gdn")
    assert rig.run("--arm", "vllm-bi", model="qwen") == 0
    u = rig.unit("vllm-bi", model="qwen")
    m = rig.manifest("vllm-bi", model="qwen")
    assert m["status"] == "unsupported" and "batch_invariant mode is not supported for GDN_ATTN" in m["status_line"]
    assert m["batch_invariant"] is True and m["batch_invariant_source"] == "set"
    assert (u / "done").exists()
    assert findings(rig, u / "manifest.json") == []


def test_other_start_failure_has_no_done(rig):
    rig.mode("boom")
    assert rig.run() == 1
    m = rig.manifest()
    assert m["status"] == "failed" and "boom" in m["status_line"] and "/opt" not in m["status_line"]
    assert not (rig.unit() / "done").exists()
    assert findings(rig, rig.unit() / "manifest.json") == []


def test_a_server_without_verify_steps_fails_its_smoke_unless_its_arm_drafts_nothing(rig):
    rig.mode("nospec")
    assert rig.run() == 1
    assert rig.manifest()["smoke"]["failures"] == ["/metrics exports no verify-step series"]
    out = ["--out", str(rig.out), "--model", "gemma", "--family", "K", "--engines", "ours", "--engine-bin",
           f"ours={rig.bin}", "--tokenizer", str(rig.tok), "--arm", "ours-k0", "--load", "scene-c1"]
    assert drive.main(out) == 0  # k = 0: its smoke needs no verify step
    m = rig.manifest("ours-k0", "K")
    assert m["smoke"]["ok"] is True and m["smoke"]["verify_steps"] is None and m["k"] == 0


@pytest.mark.parametrize(("mode", "failure"), [("spaces", "compact"), ("reasoning", "reasoning")])
def test_smoke_failure_runs_no_load(rig, mode, failure):
    rig.mode(mode)
    assert rig.run(family="A1") == 1
    u = rig.unit(family="A1")
    m = rig.manifest(family="A1")
    assert m["status"] == "smoke-failed" and failure in m["status_line"]
    assert m["smoke"]["ok"] is False and m["loads"] == []
    assert not (u / "done").exists() and not list(u.glob("scene-*.json")) and not list(u.glob("metrics-*"))


def test_rotate_orders_the_arms_and_records_it(rig, capsys):
    def order(*extra: str) -> list[str]:
        capsys.readouterr()
        assert drive.main(["--out", str(rig.out), "--model", "gemma", "--family", "smoke", "--dry-run", *extra]) == 0
        return [ln.split(":")[0].split()[-1] for ln in capsys.readouterr().out.splitlines() if ln[:5] == "smoke"]

    base = order()
    assert base[:2] == ["ours", "ours-nobi"] and [n.split("-")[0] for n in base] == [
        "ours", "ours", "vllm", "vllm", "vllm", "sglang", "sglang"]
    one = [n.split("-")[0] for n in order("--rotate", "1")]
    assert one == ["vllm", "vllm", "vllm", "sglang", "sglang", "ours", "ours"]
    assert [n.split("-")[0] for n in order("--rotate", "1", "--engines", "sglang,ours")] == ["sglang", "sglang", "ours",
                                                                                           "ours"]
    only = order("--arm", "vllm", "--arm", "sglang")
    assert only == ["vllm", "sglang"]
    with pytest.raises(SystemExit):
        order("--arm", "nonsense")


def test_dry_run_starts_nothing_writes_nothing_and_redacts(rig, capsys):
    rig.mp.undo()  # the real loads_for
    rc = drive.main(["--out", str(rig.out), "--model", "gemma", "--family", "A1", "--engines", "vllm", "--dry-run",
                     "--engine-bin", f"vllm={rig.bin}"])
    out = capsys.readouterr().out
    assert rc == 0 and not rig.out.exists()
    assert "A1 vllm: fake-engine serve" in out and str(rig.work) not in out and "env: " in out
    assert "load scene-c1: --workload scene --concurrency 1 --requests 150" in out


def test_a2_needs_a_bound_file_and_derives_its_search_from_a1(rig, capsys):
    with pytest.raises(SystemExit):
        drive.main(["--out", str(rig.out), "--model", "gemma", "--family", "A2", "--engines", "vllm"])
    a1 = rig.unit(family="A1")
    a1.mkdir(parents=True)
    (a1 / "scene-c64.json").write_text(json.dumps({"loads": [{"served_rps": 7.5}]}))
    bound = rig.tmp / "bound.json"
    bound.write_text("{}")
    rig.mp.undo()  # the real loads_for
    args = ["--out", str(rig.out), "--model", "gemma", "--family", "A2", "--bound-file", str(bound), "--dry-run"]
    assert drive.main([*args, "--engines", "vllm"]) == 0
    out = capsys.readouterr().out
    assert "--find-rate 3,7.5 --resolution 0.75 --confirm 1 --slots 128 --bound-file {bound}" in out
    assert drive.main([*args, "--engines", "sglang"]) == 0
    assert "no served rate in A1's scene-c64 report for sglang" in capsys.readouterr().out


def test_c_passes_and_the_cost_of_invariance_loads(rig, capsys):
    rig.mp.undo()
    drive.main(["--out", str(rig.out), "--model", "gemma", "--family", "C", "--engines", "vllm", "--dry-run"])
    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if ln.startswith("C ")]
    assert [ln.split(":")[0] for ln in lines] == ["C vllm-p1", "C vllm-p2", "C vllm-bi-p1", "C vllm-bi-p2",
                                                  "C vllm-bi-nodg-p1", "C vllm-bi-nodg-p2"]
    # the invariance arms' first pass also measures scene-c64 and chat-c64; the default mode's does not
    assert out.count("load scene-c64") == 2 and out.count("load stability") == 6


def test_c_runs_two_passes_and_compares_them(rig):
    assert rig.run(family="C") == 0
    assert [rig.manifest(f"vllm-p{n}", "C")["status"] for n in (1, 2)] == ["ok", "ok"]
    assert (rig.unit("vllm-p2", "C") / "stability.json").exists()
    compare = rig.out / "gemma" / "C" / "vllm-compare.txt"
    assert compare.exists() and compare.read_text().strip()
    assert rig.manifest("vllm-p1", "C")["order"] == ["vllm-p1", "vllm-p2"]


def test_d_starts_cold_then_warm_then_runs_validity(rig):
    rig.mp.setattr(arms, "CACHE_ROOT", str(rig.tmp / "cache"))
    junk = rig.tmp / "cache" / "vllm" / "stale"
    junk.mkdir(parents=True)
    (junk / "x").write_text("x")
    assert rig.run(family="D") in (0, 1)
    u = rig.unit(family="D")
    assert not junk.exists()  # the cold start wiped the engine's cache
    cold, warm = (json.loads((u / f"startup-{k}.json").read_text()) for k in ("cold", "warm"))
    assert cold["ready_s"] > 0 and warm["first_answer_s"] > 0
    m = rig.manifest(family="D")
    assert m["startup"]["cold"]["ready_s"] == cold["ready_s"] and m["startup"]["wiped"] == [{"removed": 1}]
    assert [ld["name"] for ld in m["loads"]] == ["validity"] and (u / "validity.json").exists()
    assert m["loads"][0]["ok"] is True and m["loads"][0]["client"] is None
    assert findings(rig, u / "manifest.json") == []


def test_manifest_keys_are_the_reports_keys(rig):
    assert rig.run() == 0
    m = rig.manifest()
    assert m["env_inherited"] is False and m["client"] == {"python": platform.python_version()}
    assert set(m) <= check.MANIFEST_KEYS | {"image", "commit", "env_inherited"}
    for key, subs in check.MANIFEST_NESTED.items():
        assert set(m[key] or {}) <= subs, key


# -- the log patterns and the other pure parts ------------------------------------------------------------------------

VLLM_LINES = [
    "(EngineCore pid=123) INFO 11-12 10:00:00 [kv_cache_utils.py:1147] GPU KV cache size: 1,234,560 tokens, "
    "Maximum concurrency for 8,192 tokens per request: 150.20x",
    "(APIServer pid=123) INFO 11-12 10:00:00 [startup.py:96] non-default args: {'model': 'org/m', 'max_num_seqs': 64, "
    "'enable_prefix_caching': False, 'max_model_len': 8184, 'speculative_config': {'method': 'mtp', "
    "'num_speculative_tokens': 5}}",
    "(EngineCore pid=123) INFO 11-12 10:00:00 [cuda.py:508] Using FLASH_ATTN attention backend out of potential "
    "backends: ['FLASH_ATTN', 'FLASHINFER'].",
    "(EngineCore pid=123) INFO 11-12 10:00:00 [__init__.py:779] Selected CutlassFp8Kernel for model.layers.0.mlp.down",
    "(EngineCore pid=123) INFO 11-12 10:00:00 [scheduler.py:311] Chunked prefill is enabled with "
    "max_num_batched_tokens=2048.",
    "(EngineCore pid=123) INFO 11-12 10:00:00 [gpu_worker.py:1066] Free memory on device (79.2/79.6 GiB) on startup. "
    "Desired GPU memory utilization is (0.92, 72.87 GiB). Actual usage is 1 GiB.",
    "(EngineCore pid=123) INFO 11-12 10:00:00 [gpu_model_runner.py:6516] Graph capturing finished in 12 secs, took "
    "0.85 GiB",
    "(EngineCore pid=123) INFO 11-12 10:00:00 [gpu_worker.py:513] Using V2 Model Runner",
    "(EngineCore pid=123) INFO 11-12 10:00:00 [cache.py:484] Using fp8 data type to store kv cache. It reduces the "
    "GPU memory footprint.",
]
SGLANG_LINES = [
    "[" + DAY + " 10:00:00 TP0] max_total_num_tokens=654321, chunked_prefill_size=8192, max_prefill_tokens=16384, "
    "max_running_requests=64, context_len=8184, available_gpu_mem=12.3 GB",
    "[" + DAY + " 10:00:00] server_args={'attention_backend': 'triton', 'mem_fraction_static': 0.85, "
    "'disable_radix_cache': False, 'fp8_gemm_runner_backend': 'auto', 'speculative_algorithm': 'FROZEN_KV_MTP', "
    "'speculative_num_steps': 5, 'disable_cuda_graph': False, 'kv_cache_dtype': 'auto', "
    "'enable_deterministic_inference': False}",
]


def test_vllm_log_patterns_match_their_example_lines():
    r = drive.parse_log("vllm", VLLM_LINES)
    assert r == {"kv_tokens": 1234560, "max_running_requests": 64, "prefix_cache": False,
                 "attention_backend": "FLASH_ATTN", "fp8_kernel": "CutlassFp8Kernel", "chunk_tokens": 2048,
                 "memory_fraction": 0.92, "batch_invariant_active": False, "spec": "mtp k=5", "context": 8184,
                 "cuda_graphs": "captured, 0.85 GiB", "model_runner": "v2", "kv_dtype": "fp8",
                 "notes": None}
    assert set(drive.LOG_PATTERNS["vllm"]) <= set(drive.RESOLVED_FIELDS)
    assert all(v is None for v in drive.parse_log("vllm", ["nothing here"]).values())


def test_sglang_log_patterns_match_their_example_lines():
    r = drive.parse_log("sglang", SGLANG_LINES)
    assert r["kv_tokens"] == 654321 and r["max_running_requests"] == 64 and r["chunk_tokens"] == 8192
    assert r["context"] == 8184 and r["prefix_cache"] is True and r["attention_backend"] == "triton"
    assert r["memory_fraction"] == 0.85 and r["spec"] == "FROZEN_KV_MTP steps=5" and r["cuda_graphs"] is True
    assert r["fp8_kernel"] == "auto" and r["kv_dtype"] == "auto" and r["batch_invariant_active"] is False
    assert r["model_runner"] is None
    last = drive.parse_log("sglang", [*SGLANG_LINES, "[" + DAY + " 10:00:01 TP0] max_total_num_tokens=7, "
                                      "chunked_prefill_size=1, max_prefill_tokens=1, max_running_requests=2"])
    assert last["kv_tokens"] == 7  # the last match wins


def test_ours_resolved_settings_take_only_the_allow_listed_fields():
    info = {"version": "1", "engine": "x", "auth": "none", "config": {
        "scheduler": {"max_num_seqs": 64, "max_num_batched_tokens": 2048}, "cache": {"enable_prefix_caching": True},
        "model": {"max_model_len": 8184, "path": "/secret/path"}, "speculative": {"num_speculative_tokens": 5},
        "runner": {"cuda": {"memory_utilization": 0.9}},
        "dev": {"cache": {"num_pages": 1000, "page_size": 16}, "runner": {"cuda": {"batch_invariant": True}}}},
        "runner": {"stats": {"attention": "flashinfer", "graphs": True}}}
    r = drive.resolve_ours(info)
    assert set(r) == set(drive.RESOLVED_FIELDS)
    assert r["kv_tokens"] == 16000 and r["batch_invariant_active"] is True and r["attention_backend"] == "flashinfer"
    assert r["max_running_requests"] == 64 and r["chunk_tokens"] == 2048 and r["spec"] == "mtp k=5"
    assert "/secret/path" not in json.dumps(r)
    assert r["fp8_kernel"] == drive.OURS_FP8 and r["notes"] is None
    assert all(v is None for v in drive.resolve_ours({}).values())
    # A sliding-window model: the arena's pages counted at the full pages' positions, from the startup line
    window = drive.resolve_ours(info, ["INFO KV arena: dev.cache.page_size 16 is the window page; full pages hold 320 "
                                       "positions (the bytes of a window page)"])
    assert window["kv_tokens"] == 320_000 and "window" in window["notes"]


def test_clean_line_cuts_prefixes_paths_and_secrets():
    line = "(EngineCore pid=1) ERROR 11-12 10:00:00 [x.py:1] failed at /opt/vllm/a.py with hf_abcdefghijklmnop1234"
    assert drive.clean_line(line) == "failed at <path> with <token>"
    assert drive.error_line(["ok", "ValueError: bad /root/x", "tail"]) == "ValueError: bad <path>"
    assert drive.error_line(["one", "two"]) == "two"


def test_gpu_control_summary_and_rates_without_a_gpu():
    samples = [(0.0, 1000.0, 100.0, 0), (1.0, 2000.0, 200.0, 4), (2.0, 3000.0, None, 8), (3.0, 9000.0, 900.0, 16),
               (4.0, None, 50.0, None)]
    s = gpu_control.summarize(samples, 0.5, 2.5)
    assert s == {"sm_mhz_median": 2500.0, "power_w_median": 200.0, "throttle": [4, 8]}
    assert gpu_control.summarize(samples, 10.0, 11.0) == {"sm_mhz_median": None, "power_w_median": None,
                                                          "throttle": []}
    assert gpu_control.tflops(8192, 100, 2.0) == pytest.approx(2 * 8192**3 * 100 / 2.0 / 1e12)
    assert gpu_control.copy_gbs(2 * 2**30, 10, 1.0) == pytest.approx(2 * 2 * 2**30 * 10 / 1e9)
    row = DAY + " 00:00:00.000, 301.5, 2400, 1200, 1000, 55, 0x0000000000000004"
    assert gpu_control.parse_row(row, True) == (2400.0, 301.5, 4)
    assert gpu_control.parse_row(row, False) is None  # one column too many for the base query
    assert gpu_control.parse_row(DAY + ", [N/A], [N/A], 1, 2, 3", False) == (None, None, None)
    assert gpu_control.parse_row("garbage", True) is None


def test_the_sessions_fallbacks_change_one_arm_and_are_named_deviations():
    args = drive.parser().parse_args([
        "--out", "x", "--model", "qwen", "--family", "smoke", "--attention", "vllm-bi=TRITON_ATTN",
        "--drop", "sglang=--enable-linear-replayssm-spec", "--add-arg", "sglang=--mem-fraction-static",
        "--add-arg", "sglang=0.85", "--deviation", "sglang=it ran out of memory capturing graphs"])
    units = {u.name: u.arm for u in drive.build_units(args)}
    sg = drive.arms.argv(units["sglang"], "sglang")
    assert "--enable-linear-replayssm-spec" not in sg and sg[-2:] == ["--mem-fraction-static", "0.85"]
    assert units["sglang"].deviations[0].startswith("--max-running-requests 48")  # the arm's own, first
    assert units["sglang"].deviations[1:] == ("--enable-linear-replayssm-spec left out",
                                              "added: --mem-fraction-static 0.85",
                                              "it ran out of memory capturing graphs")
    vb = drive.arms.argv(units["vllm-bi"], "vllm")
    assert vb[vb.index("--attention-backend") + 1] == "TRITON_ATTN"
    assert units["vllm"].deviations == () and "--attention-backend" not in drive.arms.argv(units["vllm"], "vllm")
    assert units["sglang"].name == "sglang"  # the unit keeps its folder


def test_skip_validity_runs_the_starts_only_and_names_the_deviation(rig):
    rig.mp.setattr(arms, "CACHE_ROOT", str(rig.tmp / "cache"))
    assert rig.run("--skip-validity", family="D") == 0
    m = rig.manifest(family="D")
    assert m["loads"] == [] and not (rig.unit(family="D") / "validity.json").exists()
    assert "no structured-output validity run (--skip-validity)" in m["deviations"]
    assert (rig.unit(family="D") / "done").exists()


# -- family I, the arms and the failed-load rule -----------------------------------------------------------------------

def test_family_i_runs_both_modes_of_each_engine_at_c64(rig, capsys):
    rig.mp.undo()  # the real loads_for
    assert drive.main(["--out", str(rig.out), "--model", "gemma", "--family", "I", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert [ln.split(":")[0] for ln in out.splitlines() if ln.startswith("I ")] == [
        "I ours", "I ours-nobi", "I vllm", "I vllm-bi-nodg", "I sglang", "I sglang-bi"]
    assert out.count("load scene-c64: --workload scene --concurrency 64 --first-index 2000000") == 6
    assert out.count("load chat-c64: --workload chat") == 6 and "stability" not in out
    assert "VLLM_USE_DEEP_GEMM" in out


def test_family_k_runs_our_own_draft_counts_on_a1s_loads_and_the_mixed_load(rig, capsys):
    rig.mp.undo()  # the real loads_for
    assert drive.main(["--out", str(rig.out), "--model", "gemma", "--family", "K", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert [ln.split(":")[0] for ln in out.splitlines() if ln.startswith("K ")] == [
        "K ours-k0", "K ours-k1", "K ours-k2", "K ours-k3", "K ours", "K ours-adaptive"]
    assert "speculative.num_speculative_tokens=0" in out and "speculative.num_speculative_tokens=5" in out
    assert out.count("dev.speculative.adaptive=true") == 1
    assert out.count("load scene-c1: ") == 6 and out.count("load chat-c64: --workload chat") == 6
    assert out.count("load mixed-c64-chat: --workload chat --input-tokens 1024 --max-tokens 256 --ignore-eos "
                     "--concurrency 32 --first-index 3000000") == 6
    assert out.count("load mixed-c64-scene: --workload scene --concurrency 32 --first-index 3000000") == 6
    k = {a.name: a for a in arms.arms("qwen")["K"]}
    assert sorted(k) == ["ours", "ours-k0", "ours-k1", "ours-k2"]
    assert "OURS_OWN_K" in arms.k_source(k["ours-k1"]) and arms.k_source(k["ours"]) == arms.MODELS["qwen"].k_source


def test_load_keeps_only_the_named_loads_and_refuses_an_unknown_one(rig, capsys):
    rig.mp.undo()  # the real loads_for
    base = ["--out", str(rig.out), "--model", "qwen", "--family", "K", "--dry-run", "--arm", "ours-k1"]
    assert drive.main([*base, "--load", "chat-c64", "--load", "mixed-c64"]) == 0
    loads = [ln.strip().split(":")[0] for ln in capsys.readouterr().out.splitlines() if ln.startswith("  load ")]
    assert loads == ["load chat-c64", "load mixed-c64-chat", "load mixed-c64-scene"]
    with pytest.raises(SystemExit):
        drive.main([*base, "--load", "chat-c64", "--load", "chat-c99"])  # every name must be known
    assert "--load chat-c99: not a load of this family" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        drive.main([*base, "--load", "chat-c99"])
    assert "--load chat-c99: not a load of this family" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        drive.main(["--out", str(rig.out), "--model", "qwen", "--family", "smoke", "--dry-run", "--load", "x"])
    assert "(have no loads)" in capsys.readouterr().err
    gemma = ["--out", str(rig.out), "--model", "gemma", "--family", "C", "--dry-run", "--load", "scene-c64"]
    assert drive.main(gemma) == 0  # a second pass has only `stability`: left out, not refused
    out = capsys.readouterr().out
    assert "-p2:" not in out and "C ours-nobi-p1:" in out and "load stability" not in out
    with pytest.raises(SystemExit):
        drive.main(["--out", str(rig.out), "--family", "control", "--label", "start", "--load", "x"])
    assert "the control family has no loads" in capsys.readouterr().err


def test_a_load_of_parts_runs_one_client_per_part_each_with_its_report(rig, monkeypatch):
    each = ["--concurrency", "2", "--requests", "4", "--warmup-requests", "1", "--max-tokens", "8"]
    spec = [drive.LoadSpec("mix", [], parts=(
        ("chat", ["--workload", "chat", "--input-tokens", "32", "--ignore-eos", *each]),
        ("scene", [*drive.SCENE, "--prompt-tokens", "64", *each])))]
    monkeypatch.setattr(drive, "loads_for", lambda ctx, unit: spec)
    assert rig.run(family="A1") == 0
    u, m = rig.unit(family="A1"), rig.manifest(family="A1")
    ld = m["loads"][0]
    assert ld["file"] == ["mix-chat.json", "mix-scene.json"] and ld["ok"] is True and len(ld["exit"]) == 2
    for part in ("chat", "scene"):
        report = json.loads((u / f"mix-{part}.json").read_text())
        assert report["meta"]["part"] == part and report["meta"]["unit"] == "vllm"
        assert report["workload"]["name"] == part and report["loads"][0]["ok"] == 4
    assert not (u / "mix.json").exists() and (u / "metrics-mix-after.txt").exists()


def test_arms_config_cache_roots_and_the_family_i_variants(monkeypatch):
    ours = arms.Arm("ours", "gemma")
    monkeypatch.setenv("LLM_MINI_INFER_CONFIG", "/x/c.yaml")
    cmd = arms.argv(ours, "/opt/venv/bin/llm-mini-infer")
    assert cmd[cmd.index("--config") + 1] == "/x/c.yaml"
    monkeypatch.delenv("LLM_MINI_INFER_CONFIG")
    cmd = arms.argv(ours, "/opt/venv/bin/llm-mini-infer")
    assert cmd[cmd.index("--config") + 1] == arms.OURS_CONFIG
    assert "~/.cache/sglang" in arms.cache_roots("sglang") and arms.cache_roots("vllm") == ["/cache/vllm"]
    nodg = {a.name: a for a in arms.arms("gemma")["I"]}["vllm-bi-nodg"]
    env = arms.env(nodg, "/opt/vllm/bin/vllm", base={})
    assert env["VLLM_USE_DEEP_GEMM"] == "0" and env["VLLM_BATCH_INVARIANT"] == "1"
    sg = {a.name: a for a in arms.arms("qwen")["A1"]}["sglang"]
    cmd = arms.argv(sg, "/opt/sglang/bin/sglang")
    assert cmd[cmd.index("--max-running-requests") + 1] == "48"
    assert cmd[cmd.index("--max-mamba-cache-size") + 1] == "240"
    assert any("--max-running-requests 48" in d for d in sg.deviations)


def test_run_children_share_one_deadline_and_kill_only_what_outlives_it(tmp_path):
    quick = [sys.executable, "-c", "raise SystemExit(3)"]
    slow = [sys.executable, "-c", "import time; time.sleep(120)"]
    t0 = time.monotonic()
    out = drive.run_children([slow, quick, slow], dict(os.environ), tmp_path, tmp_path / "log", 8)
    assert out == [(-9, True), (3, False), (-9, True)]
    assert time.monotonic() - t0 < 14  # one deadline for all: a limit per child would take 16 s


def test_an_interrupted_start_kills_every_group_already_running(tmp_path, monkeypatch):
    pidfile = tmp_path / "pid"
    first = ["/bin/sh", "-c", f"sleep 120 & echo $! > {pidfile}"]  # returns at once; the sleep stays in its group
    real, procs = drive.subprocess.Popen, []

    def popen(*a, **kw):
        if procs:  # the second start: once the first has exited, leaving its sleep
            procs[0].wait(timeout=10)
            raise KeyboardInterrupt
        procs.append(real(*a, **kw))
        return procs[0]

    monkeypatch.setattr(drive.subprocess, "Popen", popen)
    with pytest.raises(KeyboardInterrupt):
        drive.run_children([first, ["true"]], dict(os.environ), tmp_path, tmp_path / "log", 60)
    pid = int(pidfile.read_text().strip())
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        os.kill(pid, 9)
        pytest.fail("the exited leader's sleep outlived the interrupt")


# -- the context and family L ------------------------------------------------------------------------------------------

def test_the_default_context_leaves_every_arm_as_it_was():
    assert arms.CONTEXT == 8184
    for family_arms in arms.arms("gemma").values():
        for a in family_arms:
            assert a.context == arms.CONTEXT and "ctx" not in a.name
            cmd = arms.argv(a, "/x/bin")
            flag = {"ours": "--set", "vllm": "--max-model-len", "sglang": "--context-length"}[a.engine]
            want = "model.max_model_len=8184" if a.engine == "ours" else "8184"
            assert want in [cmd[i + 1] for i, c in enumerate(cmd) if c == flag]
    assert [a.name for a in arms.arms("gemma")["I"]] == [
        "ours", "ours-nobi", "vllm", "vllm-bi-nodg", "sglang", "sglang-bi"]


def test_a_longer_context_is_in_every_engines_flags_and_the_unit_name():
    for a in arms.arms("gemma", 32768)["A1"]:
        cmd = arms.argv(a, "/x/bin")
        assert a.name == f"{a.engine}-ctx32768" and a.context == 32768
        want = "model.max_model_len=32768" if a.engine == "ours" else "32768"
        assert want in cmd and "8184" not in " ".join(cmd)
    qwen = arms.argv(arms.Arm("vllm", "qwen", context=65536), "/x/bin")
    assert qwen[qwen.index("--max-model-len") + 1] == "65536"


def test_ours_is_invariant_by_default_on_gemma_and_on_qwen_up_to_its_limit_and_family_i_and_l_pair_it_with_the_other():
    default = arms.default_batch_invariant
    assert default("ours", 8184, "gemma") and default("ours", 262144, "gemma")
    assert default("ours", 32768, "qwen") and not default("ours", 32769, "qwen")
    assert not default("vllm", 8184, "gemma") and not default("sglang", 32768, "gemma")
    for family in ("I", "L"):
        ours = [a for a in arms.arms("gemma", 32768)[family] if a.engine == "ours"]
        assert [a.name for a in ours] == ["ours-ctx32768", "ours-ctx32768-nobi"]
        assert [arms.batch_invariant(a) for a in ours] == [True, False]
        assert "dev.runner.cuda.batch_invariant=false" in arms.argv(ours[1], "/x/bin")
        assert not any("batch_invariant" in x for x in arms.argv(ours[0], "/x/bin"))
        assert [a.name for a in arms.arms("gemma", 32768)[family] if a.engine == "vllm"] == [
            "vllm-ctx32768", "vllm-ctx32768-bi-nodg"]
    for context, names in ((32768, ["ours-ctx32768", "ours-ctx32768-nobi"]),
                           (65536, ["ours-ctx65536", "ours-ctx65536-bi"])):
        ours = [a for a in arms.arms("qwen", context)["I"] if a.engine == "ours"]  # on at 32,768, off past it
        assert [a.name for a in ours] == names


def test_the_invariance_limit_is_the_packages():
    from llm_mini_infer import config

    assert arms.OURS_INVARIANT_MAX_LEN == {"gemma": None, "qwen": config.QWEN_INVARIANT_MAX_LEN}


def test_family_l_keeps_the_loads_that_fit_the_context():
    def fits(context):
        kept, skipped = drive.long_loads(context)
        return [s.name for s in kept], skipped

    chat = ["chat-c16", "chat-c64", "chat-c64-r2", "chat-c64-r3"]
    names = ["decode-c1-h1k", "decode-c1-h8k", "decode-c1-h32k", *chat, "prefill-32k", "prefill-128k", "cached-32k",
             "cached-128k"]
    kept, skipped = fits(8184)
    assert kept == ["decode-c1-h1k", *chat] and len(skipped) == 6
    assert "decode-c1-h8k skipped" in skipped[0] and "8,448" in skipped[0] and "8,184" in skipped[0]
    kept, skipped = fits(32768)  # the "32k" loads fit a 32,768 context beside their replies
    assert kept == [*names[:7], "prefill-32k", "cached-32k"]
    assert [n.split()[1] for n in skipped] == ["prefill-128k", "cached-128k"]
    assert fits(131072) == (names, [])
    assert fits(262144) == (names, [])


def test_family_l_loads_have_their_own_first_indexes_and_their_own_timeouts():
    kept, _ = drive.long_loads(262144)
    firsts = [s.args[s.args.index("--first-index") + 1] for s in kept]
    assert len(set(firsts)) == len(firsts)
    by = {s.name: s for s in kept}
    first = {n: by[n].args[by[n].args.index("--first-index") + 1] for n in by}
    assert first["chat-c16"] == "1000000" and first["chat-c64"] == "2000000"  # where family I's start
    assert (first["chat-c64-r2"], first["chat-c64-r3"]) == ("2100000", "2200000")
    assert by["prefill-128k"].args[by["prefill-128k"].args.index("--input-tokens") + 1] == "130000"
    assert by["cached-32k"].args[by["cached-32k"].args.index("--shared-prefix-tokens") + 1] == "32000"
    bounded = [n for n in by if n.startswith(("decode", "prefill", "cached"))]
    assert all("--requests" in by[n].args and "--duration" not in by[n].args for n in bounded)
    assert all(by[n].fixed == {"--timeout": "900"} for n in bounded if not n.startswith("decode"))
    assert "--requests" not in by["chat-c16"].args


def test_family_l_dry_run_names_the_skipped_loads_and_the_manifest_does_too(rig, capsys):
    rig.mp.undo()  # the real loads_for
    argv = ["--out", str(rig.out), "--model", "gemma", "--family", "L", "--engines", "ours", "--context", "32768"]
    assert drive.main([*argv, "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert [ln.split(":")[0] for ln in out.splitlines() if ln.startswith("L ")] == ["L ours-ctx32768",
                                                                                  "L ours-ctx32768-nobi"]
    assert "load prefill-32k:" in out and "load prefill-128k:" not in out and "prefill-128k skipped" in out
    assert not rig.out.exists()


def test_load_keeps_only_the_named_loads():
    args = drive.parser().parse_args(["--out", "x", "--model", "gemma", "--family", "L", "--context", "262144",
                                      "--load", "prefill-128k", "--load", "cached-128k"])
    unit = drive.build_units(args)[0]
    assert [s.name for s in drive.loads_for(SimpleNamespace(args=args), unit)] == ["prefill-128k", "cached-128k"]
    args = drive.parser().parse_args(["--out", "x", "--model", "gemma", "--family", "L", "--context", "32768",
                                      "--load", "prefill-128k", "--load", "cached-32k"])
    with pytest.raises(LookupError, match=r"^--load prefill-128k: not among family L's loads here \(.*cached-32k"):
        drive.loads_for(SimpleNamespace(args=args, log=print), drive.build_units(args)[0])  # past the context


def test_a_manifest_names_the_loads_kept(rig):
    rig.mp.setattr(drive, "loads_for", lambda ctx, unit: [])
    assert rig.run("--context", "32768", "--arm", "vllm-ctx32768", "--load", "chat-c64", family="L") == 0
    assert "only the loads chat-c64 (--load)" in rig.manifest("vllm-ctx32768", "L")["deviations"]


def test_a_family_l_manifest_records_the_context_and_the_skipped_loads(rig):
    rig.mp.setattr(drive, "loads_for", lambda ctx, unit: [])
    assert rig.run("--context", "32768", "--arm", "vllm-ctx32768", family="L") == 0
    m = rig.manifest("vllm-ctx32768", "L")
    assert m["arm"] == "vllm-ctx32768" and m["argv"][m["argv"].index("--max-model-len") + 1] == "32768"
    assert any("load prefill-128k skipped" in d for d in m["deviations"])


def fake_children(monkeypatch, code=0, report: str | None = "{}", timed_out=False, only="run.py"):
    """`run_child` for the client scripts named `only`: no process, the given exit and report."""
    real = drive.run_child
    calls = []

    def fake(argv, env, cwd, log_path, limit):
        if not argv[1].endswith(only):
            return real(argv, env, cwd, log_path, limit)
        calls.append(argv)
        out = Path(argv[argv.index("--out") + 1])
        if report is not None:
            out.write_text(report)
        return code, timed_out

    monkeypatch.setattr(drive, "run_child", fake)
    return calls


@pytest.mark.parametrize(("kw", "reason"), [
    ({"code": 2}, "exit 2"), ({"code": -9}, "killed by signal 9"), ({"report": None}, "wrote no report"),
    ({"report": "[1]"}, "report is not a JSON object"), ({"report": "{"}, "report is not JSON"),
    ({"timed_out": True, "code": -9}, "timed out")])
def test_a_load_that_crashed_or_wrote_no_report_fails_its_unit(rig, monkeypatch, kw, reason):
    calls = fake_children(monkeypatch, **kw)
    assert rig.run(family="A1") == 1
    m = rig.manifest(family="A1")
    assert m["status"] == "failed" and m["status_line"] == f"load scene-c2: {reason}"
    assert [ld["name"] for ld in m["loads"]] == ["scene-c2"] and len(calls) == 1  # no further load
    assert m["loads"][0]["ok"] is False and m["loads"][0]["error"] == reason
    assert not (rig.unit(family="A1") / "done").exists()


def test_exit_one_with_a_report_is_a_result_and_a_stale_report_is_deleted(rig, monkeypatch):
    fake_children(monkeypatch, code=1)
    assert rig.run(family="A1") == 0
    assert all(ld["ok"] is True and "error" not in ld for ld in rig.manifest(family="A1")["loads"])
    assert (rig.unit(family="A1") / "done").exists()
    (rig.unit(family="A1") / "done").unlink()
    fake_children(monkeypatch, code=0, report=None)  # the stale scene-c2.json from the run above is still there
    assert (rig.unit(family="A1") / "scene-c2.json").exists()
    assert rig.run(family="A1") == 1
    assert rig.manifest(family="A1")["status_line"] == "load scene-c2: wrote no report"
    assert not (rig.unit(family="A1") / "scene-c2.json").exists()


def test_failed_validity_fails_the_unit(rig, monkeypatch):
    rig.mp.setattr(arms, "CACHE_ROOT", str(rig.tmp / "cache"))
    fake_children(monkeypatch, code=3, report=None, only="validity.py")
    assert rig.run(family="D") == 1
    m = rig.manifest(family="D")
    assert m["status"] == "failed" and m["status_line"] == "load validity: exit 3"
    assert m["loads"][0]["ok"] is False and not (rig.unit(family="D") / "done").exists()


def test_each_load_records_the_client_arguments_it_used(rig, monkeypatch):
    spec = [drive.LoadSpec("a", ["--workload", "scene", "--requests", "5"]),
            drive.LoadSpec("b", ["--workload", "scene"], {"--duration": "5", "--warmup": "1"})]
    monkeypatch.setattr(drive, "loads_for", lambda ctx, unit: spec)
    fake_children(monkeypatch)
    assert rig.run(family="A1") == 0
    m = rig.manifest(family="A1")
    base = {"seed": 0, "warmup_s": 10.0, "duration_s": 60.0, "tail_s": 30.0, "timeout_s": 120.0}
    assert m["loads"][0]["client"] == {**base, "warmup_requests": drive.WARMUP_REQUESTS}
    assert m["loads"][1]["client"] == {**base, "warmup_s": 1.0, "duration_s": 5.0, "warmup_requests": None}
    assert m["client"] == {"python": platform.python_version()}


def test_warmup_requests_follow_run_pys_rule():
    w = drive.WARMUP_REQUESTS
    assert drive.warmup_requests_of(["--requests", "5"]) == w
    assert drive.warmup_requests_of(["--find-rate", "1,8", "--resolution", "0.5"]) == w
    assert drive.warmup_requests_of(["--concurrency", "4"]) is None
    assert drive.warmup_requests_of(["--requests", "5", "--warmup-requests", "2"]) == 2
    assert drive.warmup_requests_of(["--find-rate", "1,8", "--warmup-requests=0"]) == 0
    assert drive.warmup_requests_of(["--rates", "1,2", "--warmup-requests", "3"]) == 3


def test_vllm_attention_backend_forms():
    def backend(line: str):
        return drive.parse_log("vllm", [f"(EngineCore pid=1) INFO 11-12 10:00:00 [cuda.py:5] {line}"])[
            "attention_backend"]

    assert backend("Using FLASH_ATTN attention backend out of potential backends: ['FLASH_ATTN'].") == "FLASH_ATTN"
    assert backend("Using AttentionBackendEnum.TRITON_ATTN backend.") == "TRITON_ATTN"
    assert backend("Using FLASHINFER attention backend") == "FLASHINFER"
    assert backend("Using V2 Model Runner") is None
    both = drive.parse_log("vllm", ["Using FLASH_ATTN attention backend out of potential backends: [].",
                                    "Using AttentionBackendEnum.TRITON_ATTN backend."])
    assert both["attention_backend"] == "TRITON_ATTN"  # the last match wins


def test_vllm_fp8_kernels_are_every_distinct_one_in_order():
    def kernels(*names: str):
        lines = [f"INFO 11-12 10:00:00 [x.py:1] Selected {n} for model.layers.{i}.mlp" for i, n in enumerate(names)]
        return drive.parse_log("vllm", lines)["fp8_kernel"]

    assert kernels("CutlassFp8Kernel", "CutlassFp8Kernel") == "CutlassFp8Kernel"
    assert kernels("DeepGemmKernel", "CutlassFp8Kernel", "DeepGemmKernel") == "DeepGemmKernel, CutlassFp8Kernel"
    assert kernels() is None
    assert drive.parse_log("sglang", SGLANG_LINES)["fp8_kernel"] == "auto"  # the other engines are unchanged


KV_LINE = "(EngineCore pid=1) INFO 11-12 10:00:00 [kv_cache_utils.py:1] GPU KV cache size: 1,000 tokens, x"


@pytest.mark.parametrize(("lines", "active"), [
    ([KV_LINE], False), ([], None), (["nothing"], None),
    (["(EngineCore pid=1) WARNING 11-12 10:00:00 [c.py:1] UserWarning: x 'batch_invariant.py' is traced"], True),
    ([KV_LINE, '  File "/opt/vllm/lib/python3.12/site-packages/vllm/model_executor/determinism/batch_invariant.py", '
               "line 3"], True),
    ([KV_LINE, "\x1b[33mRandom sampling without an explicit seed may not be batch invariant\x1b[0m"], True),
    (["Random sampling without an explicit seed may not be batch invariant"], True)])
def test_vllm_batch_invariance_from_its_log(lines, active):
    assert drive.parse_log("vllm", lines)["batch_invariant_active"] is active


def test_curand_note_in_its_three_cases(tmp_path):
    cuda, venv = tmp_path / "cuda", tmp_path / "sg"
    (cuda / "include").mkdir(parents=True)
    inc = venv / "lib/python3.12/site-packages/nvidia/cu13/include"
    inc.mkdir(parents=True)
    (venv / "bin").mkdir()
    binary = str(venv / "bin/sglang")
    (inc / "curand.h").write_text("header")
    assert drive.curand_note(str(cuda), binary).startswith("no curand.h in the CUDA toolkit's include folder")
    (cuda / "include/curand.h").write_text("header")
    assert drive.curand_note(str(cuda), binary).startswith("the cuRAND headers of SGLang's venv (nvidia/cu13/include/")
    (cuda / "include/curand.h").write_text("other")
    assert "differs from SGLang's venv's" in drive.curand_note(str(cuda), binary)


def test_every_sglang_manifest_carries_the_curand_note(rig, monkeypatch):
    real_env = arms.env
    cuda = rig.tmp / "cuda"  # never the real /usr/local/cuda
    monkeypatch.setattr(arms, "env", lambda *a, **k: {**real_env(*a, **k), "CUDA_HOME": str(cuda)})
    assert rig.run("--arm", "sglang", "--engine-bin", f"sglang={rig.bin}", engines="sglang") == 0
    devs = rig.manifest("sglang")["deviations"]
    assert len(devs) == 1 and devs[0].startswith("no curand.h in the CUDA toolkit's include folder")


def test_inherit_env_leaves_out_the_curand_note(rig, monkeypatch):
    real_env = arms.env
    cuda = rig.tmp / "cuda"
    monkeypatch.setattr(arms, "env", lambda *a, **k: {**real_env(*a, **k), "CUDA_HOME": str(cuda)})
    assert rig.run("--arm", "sglang", "--engine-bin", f"sglang={rig.bin}", "--inherit-env", engines="sglang") == 0
    assert rig.manifest("sglang")["deviations"] == []
