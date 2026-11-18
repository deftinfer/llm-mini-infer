"""Start the real server (launcher, engine and frontend processes) against the fake runner, and talk HTTP to it.

No server outlives its test session. Each starts in a session of its own, whose process group (the launcher's pid)
holds everything it starts; `Server.stop` kills what is left of the group once the launcher is gone, and a failed
`Server.start` stops what it launched. Until it is stopped, a server is registered here and in a PID file
(`servers_dir`): `sweep` kills the servers this process never stopped (pytest's session finish, tests/conftest.py, and
this process's exit), and `sweep_stale` those of a pytest process that was killed and so ran neither (the OOM killer, a
timeout). A launcher stops by itself once the process that started it exits (`server.stop_with_parent`, on in the
configuration written here); these remain for what that misses: a launcher killed outright (its pixel arena and IPC
directory stay), a pytest process stopped rather than killed, and the entrypoint's checks, which run before the
launcher watches its parent. `Server.start(under_parent=True)` puts a process between the two, for the tests to kill."""

from __future__ import annotations

import atexit
import contextlib
import glob
import http.client
import json
import os
import re
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import psutil
import yaml

from llm_mini_infer.output.detokenizer import Detokenizer
from llm_mini_infer.render.messages import to_template_messages
from llm_mini_infer.render.renderer import Renderer
from llm_mini_infer.runner.fake import fake_generate
from llm_mini_infer.vision import arena

from .paths import ROOT

SERVERS_ENV = "LLM_MINI_INFER_TEST_SERVERS"  # the PID files' folder, when set (`servers_dir`)
IPC_PREFIX = "llm-mini-infer-"  # an IPC directory's name starts so (`engine.ipc.ipc_dir`)
_LIVE: dict[int, Server] = {}  # the servers this process launched and has not stopped, by process group id


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def merge(into: dict, extra: dict) -> None:
    """`extra` into a configuration's dict, group by group (the dev group's subgroups too)."""
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(into.get(key), dict):
            merge(into[key], value)
        else:
            into[key] = value


_LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')  # a Prometheus label pair


class Server:
    def __init__(self, model_dir: Path, tmp: Path, eos: list[tuple[int, int]] = (),
                 step_ms: float = 1.0, extra: dict[str, Any] | None = None, mtp: int = 0,
                 env: dict[str, str] | None = None, entrypoint: bool = False, runner: str | None = "fake"):
        """`mtp`: MTP drafts (the derived default is 3 from an MTP layer, 5 from an assistant; these tests pin it, 0
        unless asked). `env`: extra environment (a Hugging Face token found in the tests' own environment never reaches
        the server, nor, when `env` points HF_ENDPOINT at a local fake hub, HF_HUB_OFFLINE, as in test_hub.resolve).
        `entrypoint`:
        start through the container's entrypoint (`llm_mini_infer.entrypoint serve`) instead of `llm-mini-infer serve`.
        `runner`: runner.kind (None: not set, the default).
        Images are off unless `extra` names the images group (`{"images": {}}` for the defaults): a pool worker and the
        pixel arena are half of a server's memory, and CI's runner holds few servers (ci.yml)."""
        self.port = free_port()
        self.model_dir = model_dir
        self.env = env or {}
        self.entrypoint = entrypoint
        kind = {"kind": runner} if runner is not None else {}
        cfg = {
            "model": {"path": str(model_dir), "max_model_len": 4096, "served_model_name": "qwen-test"},
            "runner": kind,
            "server": {"host": "127.0.0.1", "port": self.port, "drain_timeout_s": 20, "stop_with_parent": True},
            "observability": {"debug_endpoints": True, "log_level": "warning"},
            "speculative": {"num_speculative_tokens": mtp},
            "dev": {"runner": {"fake": {"step_ms": step_ms, "strict": True, "eos": [list(e) for e in eos]}}},
        }
        merge(cfg, extra or {})
        cfg.setdefault("images", {"enabled": False})
        self.cfg_path = tmp / f"server-{self.port}.yaml"
        self.cfg_path.write_text(yaml.safe_dump(cfg))
        self.log_path = tmp / f"server-{self.port}.log"
        self.proc: subprocess.Popen | Adopted | None = None
        self.parent: subprocess.Popen | None = None  # an intermediate parent's (`start(under_parent=True)`)
        self.pgid: int | None = None  # the launcher's process group (its pid), once launched and registered
        self._owner = 0  # the process that launched it (a forked child's copy of the registry is not its own)
        self._started: float | None = None  # the launcher's start time, as its PID file has it
        self._pid_file: Path | None = None  # once written
        self._ended = False

    def __enter__(self) -> Server:
        """`with Server(...).start() as srv:` stops it when the block ends."""
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: Any) -> None:
        if exc_type is None:
            self.stop()
            return
        with contextlib.suppress(Exception):  # the body's exception is the one to see, not a stop that timed out
            self.stop()

    def start(self, timeout: float = 60, under_parent: bool = False) -> Server:
        """`launch` (or `launch_under_parent`), then a wait until the server is ready. A start that fails stops what it
        launched: a launcher that exited ("server exited") keeps its exit code for `stop`, one not ready by the deadline
        is killed."""
        try:
            self.launch_under_parent() if under_parent else self.launch()
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if self.proc.poll() is not None:
                    raise RuntimeError(f"server exited: {self.log_path.read_text()[-3000:]}")
                try:
                    if self.get("/health?ready=1")[0] == 200:
                        return self
                except OSError:
                    pass
                time.sleep(0.1)
            raise RuntimeError("server not ready")
        except BaseException:
            self._end()
            raise

    def command(self, env: dict[str, str]) -> list[str]:
        """The launcher's command line (`env`, its environment, takes the entrypoint's configuration variable)."""
        if self.entrypoint:
            env["LLM_MINI_INFER_CONFIG"] = str(self.cfg_path)
            return [sys.executable, "-m", "llm_mini_infer.entrypoint", "serve"]
        return [sys.executable, "-m", "llm_mini_infer.cli", "serve", "--config", str(self.cfg_path)]

    def launch(self) -> Server:
        """Start the server's process, without waiting for it to be ready: in a session of its own, so that its process
        group holds the engine, the frontends and the pool; registered until it is stopped. A registration that fails
        (no folder or file for its PID file) stops it and raises its own error."""
        env, cmd = self._prepare()
        self.proc = subprocess.Popen(cmd, cwd=ROOT, stdout=self.log, stderr=subprocess.STDOUT, env=env,
                                     start_new_session=True)
        try:
            self._register()
        except BaseException:
            self._end()
            raise
        return self

    def launch_under_parent(self) -> Server:
        """`launch`, with a process between this one and the launcher, `parent` (`start_under_parent`), which a test
        kills (`kill_parent`) to see the launcher lose its parent. The launcher is not this process's child: `proc`
        stands in for its Popen (`Adopted`), and its exit code is the one its log gives: the configuration's log level
        must be INFO or DEBUG (`extra`), else a ValueError."""
        level = yaml.safe_load(self.cfg_path.read_text())["observability"]["log_level"]
        if level not in ("debug", "info"):
            raise ValueError(f"a launcher under a parent logs its exit code at INFO, and this one logs at {level}")
        env, cmd = self._prepare()
        try:
            self.parent, pid = start_under_parent(cmd, env, self.log)
            self.proc = Adopted(pid, self.log_path)
            self._register()
        except BaseException:
            self._end()
            raise
        return self

    def kill_parent(self) -> None:
        """SIGKILL to the intermediate parent (`launch_under_parent`), and a wait for it: the launcher is orphaned."""
        self.parent.kill()
        self.parent.wait(10)

    def _prepare(self) -> tuple[dict[str, str], list[str]]:
        """A launch's start: the log opened, the state of a previous launch cleared, the environment and the command."""
        self.log = self.log_path.open("w")
        # no Hugging Face token, and no API key unless the test sets one (`env`): the tests' requests send none
        drop = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "LLM_MINI_INFER_API_KEY", "VLLM_API_KEY")
        env = {k: v for k, v in os.environ.items() if k not in drop}
        if "HF_ENDPOINT" in self.env:  # a test's local fake hub: the tests' own offline switch would refuse it
            env.pop("HF_HUB_OFFLINE", None)
        env.update(self.env)
        cmd = self.command(env)
        self.pgid, self._pid_file, self._ended, self.parent = None, None, False, None
        return env, cmd

    def stop(self, sig: int = signal.SIGTERM, timeout: float = 30) -> int:
        """`sig` to the launcher alone (its own SIGTERM and SIGINT handling, its drain and its exit code are what
        the tests test), and its exit code once it exited; then, also when the wait timed out (TimeoutExpired, raised
        after), what is left of its group is killed (`_end`). A second stop returns the code and does nothing more; a
        server never launched stops with 0."""
        if self.proc is None:
            return 0
        if self._ended:
            return self.proc.returncode
        try:
            if self.proc.poll() is None:
                self.proc.send_signal(sig)
            return self.proc.wait(timeout)
        finally:
            self._end()

    def _end(self) -> None:
        """The end of a launched server, whatever ended it (`stop`, a failed `start` or registration, `sweep`); once.
        A launcher still running is killed with its group; one that exited keeps its exit code, and what is left of
        its group is killed. A launcher that died of a signal (killed here or by the test) leaves its pixel arena and
        IPC directory, which are removed; one that exited by itself removes them itself, and what it left is its own
        bug, to stay visible (CI reports the files left in /dev/shm). Then the server is unregistered and its log
        closed. A group is killed only while its id is known to be this server's: its launcher running and not reaped
        yet, or reaped with no process given its pid since. An intermediate parent (`launch_under_parent`) is killed
        after its launcher; a launcher of its that exited is then reaped by an init, which `_end_parent` waits for."""
        if self._ended:
            return
        self._ended = True
        try:
            pgid = self.pgid
            if pgid is not None:
                if self.proc.poll() is None:
                    _kill_group(pgid)
                    with contextlib.suppress(subprocess.TimeoutExpired):
                        self.proc.wait(10)
                    self._end_parent()
                else:
                    self._end_parent()
                    if not _exists(pgid):
                        _kill_group(pgid)
                code = self.proc.returncode
                # died of a signal, and no other process has the pid since (the arena named by it would be its own)
                if code is not None and code < 0 and not _running(pgid):
                    _remove_leftovers(pgid)
                self._unregister()
        finally:
            self._end_parent()
            if getattr(self, "log", None) is not None:
                self.log.close()

    def _end_parent(self) -> None:
        """The intermediate parent, if any, killed and reaped; then, for up to 5 s, the wait until its launcher, if it
        exited, is reaped by an init (one that does not reap leaves a zombie, which `_end` takes for a live group)."""
        parent, self.parent = self.parent, None
        if parent is None:
            return
        if parent.poll() is None:
            parent.kill()
        with contextlib.suppress(subprocess.TimeoutExpired):
            parent.wait(10)
        parent.stderr.close()
        deadline = time.monotonic() + 5
        while (self.pgid is not None and self.proc.poll() is not None and _exists(self.pgid)
               and time.monotonic() < deadline):
            time.sleep(0.05)

    def _register(self) -> None:
        """Into the registry (`sweep`) and a PID file named by the group id (`sweep_stale`, CI's failure path), with
        its owner (this process) and both start times, which tell a later process given the same pid."""
        pid = getattr(self.proc, "pid", None)
        if not isinstance(pid, int) or pid <= 0:  # a stand-in for Popen that started nothing (test_api_key's)
            return
        self.pgid, self._owner, self._started = pid, os.getpid(), _start_time(pid)
        _LIVE[pid] = self  # a server displaced here had its launcher reaped long ago: the pid was free
        folder = servers_dir(create=True)
        record = {"owner": self._owner, "owner_started": _start_time(self._owner), "started": self._started,
                  "log": str(self.log_path)}
        part = folder / f".{pid}.part"  # renamed into place: a sweep elsewhere never reads half a record
        part.write_text(json.dumps(record))
        os.replace(part, folder / str(pid))
        self._pid_file = folder / str(pid)

    def _unregister(self) -> None:
        if _LIVE.get(self.pgid) is self:  # not when a later server of this process was given the pid, and the file
            del _LIVE[self.pgid]
            if self._pid_file is not None:
                _forget(self._pid_file, (self._owner, self._started))

    def conn(self) -> http.client.HTTPConnection:
        return http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)

    def get(self, path: str) -> tuple[int, bytes]:
        c = self.conn()
        try:
            c.request("GET", path)
            r = c.getresponse()
            return r.status, r.read()
        finally:
            c.close()

    def post(self, path: str, body: Any, raw: bytes | None = None) -> tuple[int, Any]:
        c = self.conn()
        try:
            c.request("POST", path, body=raw if raw is not None else json.dumps(body),
                      headers={"content-type": "application/json"})
            r = c.getresponse()
            data = r.read()
            try:
                return r.status, json.loads(data)
            except ValueError:
                return r.status, data
        finally:
            c.close()

    def stream(self, body: dict) -> list[Any]:
        """POST a streaming chat request; returns the decoded events ("[DONE]" kept as a string)."""
        c = self.conn()
        try:
            c.request("POST", "/v1/chat/completions", body=json.dumps({**body, "stream": True}),
                      headers={"content-type": "application/json"})
            r = c.getresponse()
            assert r.status == 200, (r.status, r.read())
            assert r.getheader("content-type", "").startswith("text/event-stream")
            events = []
            while True:
                line = r.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                assert line.startswith(b"data: "), line
                payload = line[6:]
                events.append("[DONE]" if payload == b"[DONE]" else json.loads(payload))
            return events
        finally:
            c.close()

    def metric(self, name: str, default: float | None = None, **labels: str) -> float:
        """The sum of `name`'s series whose labels include `labels`; `default` when there is none (else a failure)."""
        _, text = self.get("/metrics")
        total = 0.0
        found = False
        for line in text.decode().splitlines():
            if line.startswith(name + "{") or line.startswith(name + " "):
                if labels:
                    have = dict(_LABEL.findall(line.rsplit(" ", 1)[0][len(name):]))
                    if any(have.get(k) != v for k, v in labels.items()):
                        continue
                total += float(line.rsplit(" ", 1)[1])
                found = True
        if not found and default is not None:
            return default
        assert found, (name, labels)
        return total


_PARENT = """import json, os, subprocess, sys, time
owner = os.getppid()
p = subprocess.Popen(json.loads(sys.argv[1]), stderr=subprocess.STDOUT, start_new_session=True)
print(p.pid, file=sys.stderr, flush=True)
deadline = time.monotonic() + 3600
while os.getppid() == owner and time.monotonic() < deadline:
    time.sleep(0.2)
"""  # an intermediate parent (`start_under_parent`): it lives until it is killed or its own parent is gone


def start_under_parent(cmd: list[str], env: dict[str, str], out: Any) -> tuple[subprocess.Popen, int]:
    """Start `cmd` (its stdout and stderr to the file `out`) under an intermediate parent: a process between this one
    and `cmd`'s, which a test kills to orphan it. `cmd` runs in a session of its own (its pid is its process group, as
    `launch`'s), from ROOT, and its parent waits, until it is killed or this process is gone. The parent's Popen
    (its stderr a pipe, `Server._end_parent` closes it) and `cmd`'s pid."""
    parent = subprocess.Popen([sys.executable, "-c", _PARENT, json.dumps(cmd)], cwd=ROOT, env=env, stdout=out,
                              stderr=subprocess.PIPE, text=True)
    line = parent.stderr.readline()
    try:
        return parent, int(line)
    except ValueError:
        parent.kill()
        parent.wait(10)
        why = line + parent.stderr.read()
        parent.stderr.close()
        raise RuntimeError(f"the intermediate parent started nothing: {why[-2000:]}") from None


class Adopted:
    """A launcher that is not this process's child (`Server.launch_under_parent`), with the part of Popen's interface
    that `Server` uses. It has exited once it is gone or a zombie (`_alive`: its reaper may be an init that does not
    reap), and its exit code is the last one its log gives (`stopped (exit code N)`, an INFO line), else -SIGKILL: a
    death by a signal, whose number only its reaper learns, or an exit before `serve`'s main path, which logs no code
    (a refused setting or download, `exit_on_stop`)."""

    def __init__(self, pid: int, log_path: Path):
        self.pid, self.log_path = pid, log_path
        self.started = _start_time(pid)
        self.returncode: int | None = None

    def poll(self) -> int | None:
        if self.returncode is None and not _alive(self.pid, self.started):
            codes = re.findall(r"stopped \(exit code (-?\d+)\)", self.log_path.read_text(errors="replace"))
            self.returncode = int(codes[-1]) if codes else -signal.SIGKILL
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        deadline = None if timeout is None else time.monotonic() + timeout
        while self.poll() is None:
            if deadline is not None and time.monotonic() > deadline:
                raise subprocess.TimeoutExpired(str(self.pid), timeout)
            time.sleep(0.05)
        return self.returncode

    def send_signal(self, sig: int) -> None:
        if self.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.kill(self.pid, sig)

    def kill(self) -> None:
        self.send_signal(signal.SIGKILL)


class Predictor:
    """What the fake runner will generate for a chat request, as tokens and as text."""

    def __init__(self, model_dir: Path, vocab: int = 248320, eos_ids: tuple[int, ...] = (248044, 248046)):
        self.r = Renderer(model_dir)
        self.vocab = vocab
        self.eos_ids = eos_ids

    def prompt(self, messages: list[dict], **kwargs: Any) -> list[int]:
        conv = to_template_messages(messages, self.r.content_format, self.r.supports_developer_role)
        return self.r.encode(self.r.render(conv, **kwargs)).tolist()

    def tokens(self, prompt: list[int], seed: int, n: int, eos_at: int | None = None) -> list[int]:
        """The fake's first n tokens after `prompt` (a function of seed, position and context)."""
        eos = {eos_at: self.eos_ids[0]} if eos_at is not None else None
        return fake_generate(prompt, seed, n, self.vocab, self.eos_ids, eos)

    def text(self, prompt: list[int], tokens: list[int]) -> tuple[str, list[str]]:
        d = Detokenizer(self.r.tokenizer, prompt, skip_special_tokens=True)
        pieces = [d.step(t) for t in tokens]
        return "".join(pieces), pieces


def content_of(events: list[Any]) -> str:
    return "".join(e["choices"][0]["delta"].get("content", "") for e in events
                   if isinstance(e, dict) and e.get("choices"))


def env_has_model(model_dir: Path) -> bool:
    return (model_dir / "tokenizer.json").exists() and (model_dir / "config.json").exists() and os.access(
        model_dir, os.R_OK)


# -- no survivors: the registry, the PID files and the group kills ----------------------------------------------------

def servers_dir(create: bool = False) -> Path:
    """The folder of the live servers' PID files, each named by its server's process group id:
    $LLM_MINI_INFER_TEST_SERVERS when set (a test's, or a wrapper's that wants its own), the caller's, used as given;
    else this user's folder under the system's temporary directory, made private and used only when this user owns
    it: in a shared /tmp another user could make it first, and plant records in it. Made when missing with `create`
    (a launch); a sweep makes none."""
    folder = os.environ.get(SERVERS_ENV)
    if folder:
        if create:
            Path(folder).mkdir(parents=True, exist_ok=True)
        return Path(folder)
    path = Path(tempfile.gettempdir()) / f"llm-mini-servers-{os.getuid()}"
    if create:
        path.mkdir(mode=0o700, exist_ok=True)
    elif not os.path.lexists(path):
        return path
    st = path.lstat()
    if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.getuid():
        raise RuntimeError(f"{path} is not a folder of this user's (its owner: uid {st.st_uid}), and the test servers' "
                           f"PID files need one: remove it, or set {SERVERS_ENV}")
    return path


def sweep() -> None:
    """Kill every server this process launched and has not stopped, with its group, and remove what it leaves: a test
    or a fixture that failed before its stop, or a run interrupted. At pytest's session finish (tests/conftest.py) and
    at this process's exit."""
    for srv in list(_LIVE.values()):
        if srv._owner == os.getpid():  # not a forked child's copy of its parent's servers
            srv._end()


def sweep_stale() -> None:
    """Kill the servers of the PID files whose owner, the pytest process that launched them, is gone: one killed (the
    OOM killer, a timeout) ran neither its teardown nor its exit hook, and its launchers live on. A file whose owner
    runs belongs to a session at work (this one's worker, or another terminal's session) and is left alone. A
    launcher is killed, with its group, only when its pid is still that launcher: the leader of its own group,
    `llm_mini_infer` in its command line, started when the file says; otherwise only the file goes. A file that holds
    no record written whole by this helper (`_read`) is never a reason to kill: it is removed, a directory is left."""
    try:
        paths = [p for p in servers_dir().iterdir() if p.name.isdigit()]
    except OSError:  # no folder: no server was launched here
        return
    for path in paths:
        record = _read(path)
        if record is None:  # no owner to judge it by (or gone meanwhile: its owner stopped it)
            _forget(path, None)
            continue
        if _alive(record["owner"], record.get("owner_started")):
            continue
        pid = int(path.name)
        if _is_launcher(pid, record.get("started")):
            _kill_group(pid)
        if not _running(pid):  # a dead launcher's arena and IPC directory
            _remove_leftovers(pid)
        _forget(path, _key(record))


def _read(path: Path) -> dict | None:
    """The record of the PID file `path` (`Server._register`), or None for what is not one written whole by this
    helper: a directory, a name that is not a pid, not JSON, an owner or a start time of another type."""
    try:
        pid = int(path.name)
        record = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if str(pid) != path.name or not isinstance(record, dict) or type(record.get("owner")) is not int:
        return None
    if not (0 < pid < 2**31 and 0 < record["owner"] < 2**31):  # pids as the system's calls take them
        return None
    if not all(t is None or isinstance(t, int | float) for t in (record.get("owner_started"), record.get("started"))):
        return None
    return record


def _key(record: dict | None) -> tuple[int, float | None] | None:
    """What tells one server's record from another's under the same pid: its owner and its launcher's start time."""
    return None if record is None else (record["owner"], record.get("started"))


def _forget(path: Path, key: tuple[int, float | None] | None) -> None:
    """Remove the PID file `path` if it still holds the record `key` names (`_key`; None: a plain file that holds
    none), never one that a later server given the same pid wrote meanwhile, nor a directory; never an error."""
    with contextlib.suppress(OSError):
        if path.is_file() and _key(_read(path)) == key:
            path.unlink()


def _start_time(pid: int) -> float | None:
    try:
        return psutil.Process(pid).create_time()
    except psutil.Error:
        return None


def _exists(pid: int) -> bool:
    """Whether any process (a zombie too) has the pid."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # another user's
        return True
    return True


def _alive(pid: int, started: float | None) -> bool:
    """Whether `pid` runs (a zombie, dead and waiting for its reaper, does not) and, when `started` is known, is the
    process that started then rather than a later one given its pid."""
    try:
        p = psutil.Process(pid)
        return p.status() != psutil.STATUS_ZOMBIE and (started is None or abs(p.create_time() - started) < 1)
    except psutil.NoSuchProcess:
        return False
    except psutil.Error:  # there, but not ours to read
        return True


def _running(pid: int) -> bool:
    return _alive(pid, None)


def _is_launcher(pid: int, started: float | None) -> bool:
    """Whether `pid` runs a server's launcher: the leader of its own group, `llm_mini_infer` in its command line, and
    started at `started`. Never when that is unknown (a launcher already gone at its registration): a later launcher
    given the pid would pass the other checks."""
    if started is None:
        return False
    try:
        p = psutil.Process(pid)
        return (p.status() != psutil.STATUS_ZOMBIE and abs(p.create_time() - started) < 1
                and os.getpgid(pid) == pid and any("llm_mini_infer" in arg for arg in p.cmdline()))
    except (psutil.Error, OSError):
        return False


def _group(pgid: int) -> list[psutil.Process]:
    """The processes of the group `pgid` that run."""
    found = []
    for p in psutil.process_iter():
        with contextlib.suppress(psutil.Error, OSError):
            if os.getpgid(p.pid) == pgid and p.status() != psutil.STATUS_ZOMBIE:
                found.append(p)
    return found


def _kill_group(pgid: int, timeout: float = 10) -> None:
    """SIGKILL to every process of the group `pgid`, then a wait of up to `timeout` until none runs (a zombie, which
    holds no memory, may wait longer for its reaper). Only for a group known to be a server's (`Server._end`,
    `sweep_stale`)."""
    try:
        os.killpg(pgid, 0)
    except OSError:  # empty, the common case after a launcher that exited by itself; or not this user's
        return
    procs = _group(pgid)
    with contextlib.suppress(OSError):
        os.killpg(pgid, signal.SIGKILL)
    deadline = time.monotonic() + timeout
    while procs and time.monotonic() < deadline:
        time.sleep(0.02)
        procs = [p for p in procs if _still_runs(p)]


def _still_runs(p: psutil.Process) -> bool:
    try:
        return p.is_running() and p.status() != psutil.STATUS_ZOMBIE  # is_running: not a later process given its pid
    except psutil.Error:
        return False


def _remove_leftovers(pid: int) -> None:
    """What a launcher that died of a signal as `pid` leaves (one that exits by itself removes them): its pixel arena,
    named by its pid and its IPC directory's name (`launcher._start_images`; in /dev/shm on Linux, inside the IPC
    directory on macOS), and that IPC directory. A server without images leaves only its IPC directory, a few
    sockets, which nothing names: it stays. An arena this user cannot remove (another user's, in a shared /dev/shm)
    stays, with its IPC directory."""
    base = "/tmp" if os.path.isdir("/tmp") else tempfile.gettempdir()  # where `engine.ipc.ipc_dir` makes it
    head, tail = os.path.basename(arena.arena_path(base, "{}")).split("{}")  # the arena's name around its tag
    for path in glob.glob(arena.arena_path(arena.default_dir(os.path.join(base, IPC_PREFIX + "*")), f"{pid}-*")):
        name = os.path.basename(path).removeprefix(head).removesuffix(tail).removeprefix(f"{pid}-")
        try:
            arena.remove(path)
        except OSError:
            continue
        if name.startswith(IPC_PREFIX) and os.sep not in name:
            shutil.rmtree(os.path.join(base, name), ignore_errors=True)


atexit.register(sweep)
