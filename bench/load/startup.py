"""Time how long a server command takes to become ready and to answer its first request.

Usable as a CLI and as a library. Each start launches the command in its own process group, polls the readiness path
(`ready_s`), then sends one fixed chat request until it succeeds (`first_answer_s`, the headline: engines differ in
what their readiness path means), and stops the server. Standard library only.

    python3 bench/load/startup.py --out report.json --model NAME [--base-url http://127.0.0.1:8000] \
        [--ready-path /health] [--starts 1] [--wipe DIR]... [--evict DIR]... [--hold] [--timeout 1800] \
        [--cwd /] [--log PATH] -- <server argv...>
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from llm_mini_infer import server_process

POLL_S = 0.25
# Seconds to wait after SIGINT, SIGTERM and SIGKILL, and for the port to come free.
STOP_WAITS = (60.0, 20.0, 10.0)
PORT_WAIT_S = 60.0
SECRET_FLAGS = ("--api-key", "--hf-token", "--token")
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".gguf", ".pt", ".pth")


class StartError(RuntimeError):
    """A start that failed. `log_tail` is the end of the server log, `exit_code` the child's exit code if it exited."""

    def __init__(self, message: str, log_tail: str = "", exit_code: int | None = None) -> None:
        super().__init__(message)
        self.log_tail = log_tail
        self.exit_code = exit_code


@dataclass
class Handle:
    """A launched server: the process, the monotonic time just before launch, its log file and its base URL."""

    proc: subprocess.Popen
    t0: float
    log_path: Path
    base_url: str


def redact(argv: list[str]) -> list[str]:
    """Copy argv with secret values replaced and the executable cut to its basename."""
    out: list[str] = []
    hide_next = False
    for i, arg in enumerate(argv):
        if hide_next:
            out.append("<redacted>")
            hide_next = False
        elif i == 0:
            out.append(os.path.basename(arg))
        elif arg in SECRET_FLAGS:
            out.append(arg)
            hide_next = True
        elif any(arg.startswith(f + "=") for f in SECRET_FLAGS):
            out.append(arg.split("=", 1)[0] + "=<redacted>")
        else:
            out.append(arg)
    return out


def launch(argv: list[str], *, env: dict[str, str] | None, cwd: str, log_path: Path, base_url: str) -> Handle:
    """Start the server in a new session with stdout and stderr appended to `log_path`; `t0` is taken just before."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "ab") as log:
        t0 = time.monotonic()
        proc = server_process.launch(argv, cwd=cwd, env=env if env is not None else os.environ.copy(), stdout=log,
                                     stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    return Handle(proc, t0, log_path, base_url.rstrip("/"))


def _exited(h: Handle, what: str) -> StartError | None:
    code = h.proc.poll()
    if code is None:
        return None
    return StartError(f"server exited with code {code} before {what}", server_process.log_tail(h.log_path) or "", code)


def _timed_out(h: Handle, what: str, timeout: float) -> StartError:
    stop(h)
    return StartError(f"no {what} within {timeout:g} s", server_process.log_tail(h.log_path) or "", None)


def wait_ready(h: Handle, ready_path: str = "/health", timeout: float = 1800) -> float:
    """Seconds from launch to the first 200 on `base_url + ready_path`. Raises StartError if the child exits first
    or the timeout passes (the child is stopped first)."""
    url = h.base_url + ready_path
    while True:
        err = _exited(h, "becoming ready")
        if err:
            raise err
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return round(time.monotonic() - h.t0, 2)
        except (OSError, urllib.error.URLError, ValueError):
            pass  # connection refused, reset, HTTP error status or timeout: keep polling
        if time.monotonic() - h.t0 > timeout:
            raise _timed_out(h, "readiness", timeout)
        time.sleep(POLL_S)


def _error_type(body: bytes) -> str:
    try:
        err = json.loads(body).get("error")
        if isinstance(err, dict):
            return str(err.get("type") or err.get("code") or "unknown")
        if isinstance(err, str):
            return "error"
        return str(json.loads(body).get("type") or "unknown")
    except (ValueError, AttributeError):
        return "unknown"


def first_answer(h: Handle, model: str, timeout: float = 1800) -> float:
    """Seconds from launch to the first 200 for one fixed, non-streamed chat request. Retries on a connection error
    or a 5xx; a 4xx raises StartError with the status and the error type."""
    payload = json.dumps({
        "model": model, "messages": [{"role": "user", "content": "Say hello."}], "max_tokens": 16, "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    url = h.base_url + "/v1/chat/completions"
    while True:
        err = _exited(h, "the first answer")
        if err:
            raise err
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                r.read()
                if r.status == 200:
                    return round(time.monotonic() - h.t0, 2)
        except urllib.error.HTTPError as e:
            if e.code < 500:
                body = e.read()
                stop(h)
                msg = f"first request refused: HTTP {e.code} ({_error_type(body)})"
                raise StartError(msg, server_process.log_tail(h.log_path) or "") from None
        except (OSError, urllib.error.URLError, ValueError):
            pass
        if time.monotonic() - h.t0 > timeout:
            raise _timed_out(h, "first answer", timeout)
        time.sleep(POLL_S)


def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def stop(h: Handle) -> dict:
    """Stop the server's process group (SIGINT, then SIGTERM, then SIGKILL), reap it, and wait for its port to come
    free. Returns {"signal", "seconds", "port_free"}; "signal" is the one that ended it, or null if it had exited."""
    t = time.monotonic()
    used = server_process.stop(h.proc, first_signal=signal.SIGINT, escalation_signals=(signal.SIGTERM,),
                               waits=STOP_WAITS, signal_group_signals=True)
    parsed = urllib.parse.urlparse(h.base_url)
    host, port = parsed.hostname or "127.0.0.1", parsed.port or (443 if parsed.scheme == "https" else 80)
    deadline = time.monotonic() + PORT_WAIT_S
    free = not _port_open(host, port)
    while not free and time.monotonic() < deadline:
        time.sleep(POLL_S)
        free = not _port_open(host, port)
    return {"signal": used, "seconds": round(time.monotonic() - t, 2), "port_free": free}


def _holds_weights(root: Path) -> bool:
    for _, dirs, files in os.walk(root):
        if any(d.startswith("models--") for d in dirs) or any(f.endswith(WEIGHT_SUFFIXES) for f in files):
            return True
    return False


def wipe(dirs: list[str]) -> list[dict]:
    """Empty each directory (created if missing). Raises ValueError, before removing anything, for a directory that
    holds model weights, is `/`, the home directory or a parent of the current directory."""
    cwd = Path.cwd().resolve()
    roots = [Path(d).expanduser().resolve() for d in dirs]
    for root in roots:
        if root == Path("/") or root == Path.home().resolve() or root == cwd or root in cwd.parents:
            raise ValueError(f"refusing to wipe {root}: it is the root, the home directory or holds the current folder")
        if root.is_dir() and _holds_weights(root):
            raise ValueError(f"refusing to wipe {root}: it holds model weights")
    out = []
    for root in roots:
        root.mkdir(parents=True, exist_ok=True)
        removed = 0
        for entry in root.iterdir():
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry)
            else:
                entry.unlink()
            removed += 1
        out.append({"dir": str(root), "removed": removed})
    return out


def _fincore(files: list[str]) -> int | None:
    exe = shutil.which("fincore")
    if not exe or not files:
        return None
    total = 0
    try:
        for i in range(0, len(files), 200):
            res = subprocess.run([exe, "--bytes", "--noheadings", "--output", "RES", *files[i:i + 200]],
                                 capture_output=True, text=True, timeout=120)
            if res.returncode != 0:
                return None
            total += sum(int(x) for x in res.stdout.split())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    return total


def evict(dirs: list[str]) -> list[dict]:
    """Ask the kernel to drop cached pages of every file under each directory (symlinks followed, each target once).
    The report holds files and bytes advised, errors, and resident bytes from `fincore` before and after if it
    exists (else null). The call does not say whether pages were dropped."""
    out = []
    for d in dirs:
        seen: set[tuple[int, int]] = set()
        files: list[str] = []
        sizes = 0
        errors = 0
        for dirpath, _, names in os.walk(d, followlinks=True):
            for name in names:
                p = os.path.join(dirpath, name)
                try:
                    st = os.stat(p)
                except OSError:
                    errors += 1
                    continue
                if not os.path.isfile(p) or (st.st_dev, st.st_ino) in seen:
                    continue
                seen.add((st.st_dev, st.st_ino))
                files.append(p)
        before = _fincore(files)
        advised = 0
        for p in files:
            try:
                fd = os.open(p, os.O_RDONLY)
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                    advised += 1
                    sizes += os.fstat(fd).st_size
                finally:
                    os.close(fd)
            except (OSError, AttributeError):
                errors += 1
        out.append({"dir": d, "files": advised, "bytes": sizes, "errors": errors,
                    "resident_before": before, "resident_after": _fincore(files)})
    return out


def _start_once(args: argparse.Namespace, log_path: Path) -> tuple[dict, Handle | None]:
    rec: dict = {"ready_s": None, "first_answer_s": None, "error": None, "stopped": None}
    h = launch(args.argv, env=None, cwd=args.cwd, log_path=log_path, base_url=args.base_url)
    try:
        rec["ready_s"] = wait_ready(h, args.ready_path, args.timeout)
        rec["first_answer_s"] = first_answer(h, args.model, args.timeout)
    except StartError as e:
        rec["error"] = f"{e}\n{e.log_tail}".strip()
    return rec, h


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--ready-path", default="/health")
    ap.add_argument("--model", required=True)
    ap.add_argument("--starts", type=int, default=1)
    ap.add_argument("--wipe", action="append", default=[])
    ap.add_argument("--evict", action="append", default=[])
    ap.add_argument("--hold", action="store_true")
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--cwd", default="/")
    ap.add_argument("--log")
    ap.add_argument("server_argv", nargs=argparse.REMAINDER)
    args = ap.parse_args(argv)
    args.argv = args.server_argv[1:] if args.server_argv[:1] == ["--"] else args.server_argv
    if not args.argv:
        ap.error("the server command is missing; put it after --")
    log_path = Path(args.log) if args.log else Path(args.out).with_suffix(".log")

    try:
        wiped = wipe(args.wipe)
    except ValueError as e:
        print(f"startup: {e}", file=sys.stderr)
        return 2
    evicted = evict(args.evict)
    report = {
        "argv": redact(args.argv), "cwd": args.cwd, "base_url": args.base_url, "ready_path": args.ready_path,
        "model": args.model, "env_names": sorted(os.environ), "wiped": wiped, "evicted": evicted, "starts": [],
    }
    ok = True
    last: Handle | None = None
    for i in range(args.starts):
        rec, h = _start_once(args, log_path)
        last = h
        keep = args.hold and i == args.starts - 1 and rec["error"] is None
        if not keep:
            rec["stopped"] = stop(h)
        report["starts"].append(rec)
        if rec["error"]:
            ok = False
            print(f"startup: start {i + 1} failed: {rec['error'].splitlines()[0]}", file=sys.stderr)
            break
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    if args.hold and ok and last is not None:
        stopping = {"now": False}

        def on_signal(signum, frame):
            stopping["now"] = True

        signal.signal(signal.SIGTERM, on_signal)
        signal.signal(signal.SIGINT, on_signal)
        while not stopping["now"] and last.proc.poll() is None:
            time.sleep(0.2)
        stop(last)
        return 0
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
