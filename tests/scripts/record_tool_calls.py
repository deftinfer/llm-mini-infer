"""Record what a model answers to the tool-calling cases (`tests/data/tool_calls/cases.json`, synthetic): one server
per backend, every case sent one at a time, not streamed and streamed, with the response's and each event's bytes,
the prompt's and the output's token ids (`return_token_ids`), and the MTP and step counters around each request.

In the GPU tests' image on a GPU host (docs/development.md, "Numerics" runs its tools the same way), from its tree:

    python3 -m tests.scripts.record_tool_calls --model RedHatAI/gemma-4-31B-it-FP8-block --revision <commit> \\
        --set speculative.model=google/gemma-4-31B-it-assistant --set speculative.revision=<commit> \\
        --cases basic --out-dir rec

It starts `llm-mini-infer serve --model M --config deploy/config.yaml --set model.revision=R --set
cache.enable_prefix_caching=false` (prefix caching off, so that each case's bits are its own: every case shares the
tools' prefix), waits for `/health?ready=1`, and stops the server by its PID before the next backend. `--backends
guidance auto`: under `auto` (`structured_output.backend=auto`, `disable_any_whitespace=false`) only the cases marked
for both backends run again. Requests go at temperature 0 with `skip_special_tokens: false`. One JSON per model and
backend lands in `--out-dir`, with the server's log beside it. `--runner fake` runs it on the CPU against the fake
runner (tests/repo/test_record_tool_calls.py).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / "tests" / "data" / "tool_calls" / "cases.json"
AUTO = ["structured_output.backend=auto", "structured_output.disable_any_whitespace=false"]
# The counters scraped around each request: MTP's, and the engine's steps (sums and counts)
METRICS = re.compile(r"^(vllm:spec_decode_\w+|vllm:iteration_tokens_total_(?:sum|count)"
                     r"|llm_mini_infer:engine_step_\w+_(?:sum|count))(\{[^}]*\})? ([0-9.eE+-]+)$")


def family(model: str) -> str:
    """The cases' family: a local checkpoint's `model_type` (config.json), else a repo id's name, its last part (a
    directory's path may hold "gemma" anywhere, a worktree's name included)."""
    config = Path(model) / "config.json"
    name = json.loads(config.read_text()).get("model_type", "") if config.is_file() else Path(model).name
    return "gemma" if "gemma" in name.lower() else "qwen"


def load_cases(path: Path, select: str, fam: str) -> tuple[dict, list[dict]]:
    """The cases for one family: `basic`, `extended` (basic and the rest), `all`, or names separated by commas."""
    data = json.loads(path.read_text())
    cases = [c for c in data["cases"] if c.get("family", fam) == fam]
    if select == "basic":
        cases = [c for c in cases if c["group"] == "basic"]
    elif select in ("extended", "all"):
        pass
    else:
        names = select.split(",")
        missing = sorted(set(names) - {c["name"] for c in cases})
        if missing:
            raise SystemExit(f"no such case for {fam}: {', '.join(missing)}")
        cases = [c for c in cases if c["name"] in names]
    return data["tools"], cases


def request(case: dict, tools: dict, stream: bool) -> dict:
    body: dict[str, Any] = {"messages": case["messages"], "max_tokens": case["max_tokens"], "temperature": 0,
                            "return_token_ids": True, "skip_special_tokens": False, "stream": stream}
    if case["tools"] is not None:
        body["tools"] = [tools[t] if isinstance(t, str) else t for t in case["tools"]]
    if "kwargs" in case:
        body["chat_template_kwargs"] = case["kwargs"]
    body.update(case.get("extra", {}))
    return body


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def serve_command(args: argparse.Namespace, port: int, backend: str) -> list[str]:
    exe = shutil.which("llm-mini-infer")
    cmd = [exe] if exe else [sys.executable, "-m", "llm_mini_infer.cli"]
    cmd += ["serve", "--model", args.model, "--config", str(args.config), "--port", str(port), "--host", "127.0.0.1"]
    sets = ["cache.enable_prefix_caching=false", *(["model.revision=" + args.revision] if args.revision else []),
            *(["runner.kind=fake"] if args.runner == "fake" else []), *(AUTO if backend == "auto" else []),
            *args.set]
    for s in sets:
        cmd += ["--set", s]
    return cmd


class ServerProcess:
    def __init__(self, cmd: list[str], log_path: Path, port: int):
        self.cmd, self.log_path, self.port = cmd, log_path, port
        self.proc: subprocess.Popen | None = None

    def start(self) -> None:
        # never a token (public checkpoints only): neither of the variables huggingface_hub reads
        # no Hugging Face token, and no API key: the recording's server serves only this script, which sends none
        drop = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "LLM_MINI_INFER_API_KEY", "VLLM_API_KEY")
        env = {k: v for k, v in os.environ.items() if k not in drop}
        self._log = self.log_path.open("w")
        self.proc = subprocess.Popen(self.cmd, stdout=self._log, stderr=subprocess.STDOUT, env=env,
                                     start_new_session=True)

    def stop(self) -> int:
        """SIGTERM to its PID (the launcher stops its engine and frontends), then SIGKILL after 60 s."""
        p = self.proc
        if p is None or p.poll() is not None:
            return p.returncode if p else 0
        p.send_signal(signal.SIGTERM)
        try:
            return p.wait(60)
        except subprocess.TimeoutExpired:
            p.kill()
            return p.wait(10)
        finally:
            self._log.close()


async def wait_ready(session: Any, base: str, proc: ServerProcess, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.proc.poll() is not None:
            raise RuntimeError(f"the server exited ({proc.proc.returncode}): see {proc.log_path}")
        try:
            async with session.get(base + "/health?ready=1") as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        await asyncio.sleep(1.0)
    raise TimeoutError(f"the server was not ready in {timeout:.0f} s: see {proc.log_path}")


async def scrape(session: Any, base: str) -> dict[str, float]:
    async with session.get(base + "/metrics") as r:
        text = await r.text()
    out = {}
    for line in text.splitlines():
        m = METRICS.match(line)
        if m:
            out[m.group(1) + (m.group(2) or "")] = float(m.group(3))
    return out


def delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    return {k: after[k] - before.get(k, 0.0) for k in after if after[k] != before.get(k, 0.0)}


async def send(session: Any, base: str, body: dict) -> dict:
    """One request: its status, its bytes (each event's when streamed) and its times (ns from the send)."""
    t0 = time.perf_counter_ns()
    out: dict[str, Any] = {"stream": body["stream"]}
    async with session.post(base + "/v1/chat/completions", json=body) as r:
        out["status"] = r.status
        if body["stream"] and r.status == 200:
            events, first = [], 0
            buf = b""
            async for chunk in r.content.iter_any():
                if not first:
                    first = time.perf_counter_ns() - t0
                buf += chunk
                while b"\n\n" in buf:
                    event, buf = buf.split(b"\n\n", 1)
                    events.append(event.decode())
            out["events"] = events
            out["t_first_ns"] = first
        else:
            raw = await r.read()
            out["body"] = raw.decode()
            out["t_first_ns"] = time.perf_counter_ns() - t0
    out["t_end_ns"] = time.perf_counter_ns() - t0
    return out


async def record_backend(args: argparse.Namespace, backend: str, tools: dict, cases: list[dict], out: Path) -> Path:
    import aiohttp

    port = free_port()
    stem = re.sub(r"[^a-z0-9.]+", "-", Path(args.model.rstrip("/")).name.lower()) + f"-{backend}"
    proc = ServerProcess(serve_command(args, port, backend), out / f"{stem}.log", port)
    base = f"http://127.0.0.1:{port}"
    timeout = aiohttp.ClientTimeout(total=args.request_timeout)
    proc.start()
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            await wait_ready(session, base, proc, args.start_timeout)
            async with session.get(base + "/server_info") as r:
                info = await r.json()
            results = []
            run = [c for c in cases if backend == "guidance" or backend in c.get("backends", ())]
            for i, case in enumerate(run):
                for stream in (False, True):
                    body = request(case, tools, stream)
                    before = await scrape(session, base)
                    got = await send(session, base, body)
                    got["metrics"] = delta(before, await scrape(session, base))
                    got.update(case=case["name"], request=body)
                    results.append(got)
                print(f"[{backend}] {i + 1}/{len(run)} {case['name']}: "
                      f"{results[-2]['status']} {results[-1]['status']}", flush=True)
        path = out / f"{stem}.json"
        path.write_text(json.dumps({"model": args.model, "revision": args.revision, "backend": backend,
                                    "family": family(args.model), "command": proc.cmd, "server_info": info,
                                    "results": results}, indent=1, ensure_ascii=False) + "\n")
        return path
    finally:
        code = proc.stop()
        print(f"[{backend}] server {proc.proc.pid} stopped ({code})", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, help="a Hugging Face repo id or a local checkpoint directory")
    ap.add_argument("--revision", help="the repo's commit (model.revision)")
    ap.add_argument("--cases", default="basic", help="basic, extended, all, or case names separated by commas")
    ap.add_argument("--cases-file", type=Path, default=CASES)
    ap.add_argument("--backends", nargs="+", default=["guidance"], choices=["guidance", "auto"],
                    help="guidance: the image's configuration; auto: the constrained cases again under auto")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=ROOT / "deploy" / "config.yaml")
    ap.add_argument("--runner", choices=["cuda", "fake"], default="cuda")
    ap.add_argument("--set", action="append", default=[], metavar="GROUP.KEY=VALUE", help="more settings")
    ap.add_argument("--start-timeout", type=float, default=1800.0)
    ap.add_argument("--request-timeout", type=float, default=900.0)
    args = ap.parse_args(argv)
    fam = family(args.model)
    tools, cases = load_cases(args.cases_file, args.cases, fam)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for backend in args.backends:
        path = asyncio.run(record_backend(args, backend, tools, cases, args.out_dir))
        print(f"wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
