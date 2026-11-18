"""The server's lifecycle through the real server (launcher, engine, frontend; the fake runner, except where a test
leaves runner.kind at its default, cuda): readiness, request ids, the drain on SIGTERM that loses no request, failing
fast when the engine dies or hangs or no GPU is there, and logs that never carry request contents, an engine crash's
included."""

from __future__ import annotations

import base64
import contextlib
import http.client
import io
import json
import logging
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest

from llm_mini_infer.launcher import Watchdog, _watch_parent

from ..support.api_util import Adopted, Predictor, Server, content_of, start_under_parent
from ..support.fixtures import require_qwen
from ..support.paths import ROOT

MSG = [{"role": "user", "content": "Hello!"}]
KW = {"enable_thinking": False}


def chat(**kw) -> dict:
    return {"messages": MSG, "chat_template_kwargs": KW, **kw}


def request(srv: Server, body: dict, headers: dict | None = None, timeout: float = 60) -> tuple[int, dict, Any]:
    """POST a chat request: (status, response headers, the JSON body or the stream's events)."""
    c = http.client.HTTPConnection("127.0.0.1", srv.port, timeout=timeout)
    try:
        c.request("POST", "/v1/chat/completions", body=json.dumps(body),
                  headers={"content-type": "application/json", **(headers or {})})
        r = c.getresponse()
        hdrs = {k.lower(): v for k, v in r.getheaders()}
        data = r.read()
        if hdrs.get("content-type", "").startswith("text/event-stream"):
            events = [e[6:] for e in data.decode().split("\n\n") if e.startswith("data: ")]
            return r.status, hdrs, [e if e == "[DONE]" else json.loads(e) for e in events]
        return r.status, hdrs, json.loads(data) if data else None
    finally:
        c.close()


def engine_pid(srv: Server) -> int:
    return json.loads(srv.get("/debug/engine")[1])["engine_pid"]


@pytest.fixture(scope="module")
def model():
    return require_qwen()


@pytest.fixture(scope="module")
def server(model, tmp_path_factory):
    srv = Server(model, tmp_path_factory.mktemp("srv"), extra={"observability": {"log_level": "info"}}).start()
    yield srv
    assert srv.stop() == 0, srv.log_path.read_text()[-2000:]


def test_health_is_readiness_and_live_is_liveness(server) -> None:
    for path in ("/health", "/health?ready=1", "/ready", "/live"):
        status, body = server.get(path)
        assert status == 200 and json.loads(body)["status"] == "ok", path


def test_request_ids_are_echoed_and_name_the_completion(server) -> None:
    rid = "router-7d3f1c2e-0b9a-4c1e-9f00-1234567890ab"
    status, hdrs, body = request(server, chat(max_tokens=3, seed=5), {"X-Request-Id": rid})
    assert status == 200 and hdrs["x-request-id"] == rid and body["id"] == f"chatcmpl-{rid}"
    status, hdrs, events = request(server, chat(max_tokens=3, seed=5, stream=True), {"X-Request-Id": rid})
    assert status == 200 and hdrs["x-request-id"] == rid
    assert {e["id"] for e in events if isinstance(e, dict)} == {f"chatcmpl-{rid}"}
    status, hdrs, body = request(server, chat(max_tokens=2, seed=6))  # none sent: one is made, header = body
    assert status == 200 and body["id"] == f"chatcmpl-{hdrs['x-request-id']}" and len(hdrs["x-request-id"]) == 24
    status, hdrs, body = request(server, chat(max_tokens=2, seed=6), {"X-Request-Id": 'bad "id" with spaces'})
    assert status == 200 and '"' not in hdrs["x-request-id"] and body["id"] == f"chatcmpl-{hdrs['x-request-id']}"
    status, hdrs, body = request(server, {"messages": []}, {"X-Request-Id": rid})  # errors carry it too
    assert status == 400 and hdrs["x-request-id"] == rid
    c = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
    c.request("GET", "/health", headers={"X-Request-Id": "probe-1"})
    assert c.getresponse().getheader("x-request-id") == "probe-1"
    c.close()
    log = server.log_path.read_text()
    assert f"request id={rid} status=200 finish=length stream=0 prompt=" in log
    assert f"request id={rid} status=400 error=BadRequestError param=messages" in log


def test_an_unexpected_error_logs_the_500_uvicorn_answers(caplog) -> None:
    """A chat request that fails with neither an API error nor a client gone is answered with a 500 by uvicorn: its
    request line says so (it said 200, or 0 before the request state existed)."""
    import asyncio
    import logging
    from types import SimpleNamespace

    from llm_mini_infer.api.app import App

    async def chat(*args) -> None:
        raise RuntimeError("an engine bug")

    app = SimpleNamespace(log_requests=True, chat=chat)
    app.log_request = lambda rec: App.log_request(app, rec)
    with caplog.at_level(logging.INFO, logger="llm_mini_infer.api.app"), pytest.raises(RuntimeError):
        asyncio.run(App.chat_logged(app, None, None, "boom"))
    assert [r.getMessage().split(" e2e_ms")[0] for r in caplog.records] == [
        "request id=boom status=500 error=InternalServerError"]


def _raw(srv: Server, body: dict, rid: str) -> socket.socket:
    """A chat request sent on a raw socket, so that the test can close it before or after the response starts."""
    s = socket.create_connection(("127.0.0.1", srv.port))
    data = json.dumps(body).encode()
    s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: llm-mini-infer\r\nContent-Type: application/json\r\n"
              + f"X-Request-Id: {rid}\r\nContent-Length: {len(data)}\r\n\r\n".encode() + data)
    return s


def test_the_request_log_tells_a_client_that_left_by_whether_its_response_started(server) -> None:
    """status=499 when the client left before the response started (an abandoned non-streamed request), 200 when it
    left after (a stream cut short): log_request's meaning, as the HTTP metrics count them."""
    s = _raw(server, chat(max_tokens=3000, seed=21, stream=True), "left-stream")
    got = b""
    while got.count(b"data: ") < 3:
        got += s.recv(65536)
    s.close()
    s = _raw(server, chat(max_tokens=3000, seed=22), "left-whole")
    time.sleep(0.3)
    s.close()
    lines: dict[str, str] = {}
    deadline = time.monotonic() + 30
    while len(lines) < 2 and time.monotonic() < deadline:
        time.sleep(0.1)
        lines = dict(re.findall(r"request id=(left-stream|left-whole) (status=\d+ finish=\S+ stream=\d)",
                                server.log_path.read_text()))
    assert lines == {"left-stream": "status=200 finish=abort stream=1",
                     "left-whole": "status=499 finish=abort stream=0"}


def test_sigterm_drain_with_grace_loses_no_request(model, tmp_path) -> None:
    """docker compose stop behind a health-checking router: /health turns 503 at once, requests arriving during the
    grace are served, later ones are refused, every in-flight request finishes, and the server exits 0."""
    pred = Predictor(model)
    prompt = pred.prompt(MSG, **KW)
    with Server(model, tmp_path, step_ms=5,
                extra={"server": {"drain_grace_s": 1.5, "drain_timeout_s": 20},
                       "observability": {"log_level": "info"}}).start() as srv:
        results: dict[int, Any] = {}

        def one(i: int, n: int = 600) -> None:
            results[i] = request(srv, chat(max_tokens=n, seed=900 + i, stream=bool(i % 2)))

        threads = [threading.Thread(target=one, args=(i,)) for i in range(6)]
        for t in threads:
            t.start()
        time.sleep(0.3)  # all six are running (600 steps of 5 ms: they outlast the grace)
        srv.proc.send_signal(signal.SIGTERM)
        time.sleep(0.2)
        assert srv.get("/health")[0] == 503 and srv.get("/live")[0] == 200
        late = threading.Thread(target=one, args=(10, 20))  # arrives during the grace: served
        late.start()
        time.sleep(1.8)  # past the grace
        assert request(srv, chat(max_tokens=2))[0] == 503
        for t in [*threads, late]:
            t.join(30)
        for i in [*range(6), 10]:
            status, _, body = results[i]
            n = 20 if i == 10 else 600
            text = pred.text(prompt, pred.tokens(prompt, 900 + i, n))[0]
            assert status == 200, (i, status, body)
            got = content_of(body) if isinstance(body, list) else body["choices"][0]["message"]["content"]
            assert got == text, i
        assert srv.proc.wait(30) == 0
        srv.log.close()
    log = srv.log_path.read_text()
    assert "SIGTERM: draining" in log and "frontends drained" in log


def test_sigint_drains_without_the_grace(model, tmp_path) -> None:
    with Server(model, tmp_path, step_ms=5, extra={"server": {"drain_grace_s": 30}}).start() as srv:
        results: dict[int, Any] = {}
        t = threading.Thread(target=lambda: results.setdefault(0, request(srv, chat(max_tokens=100, seed=3))))
        t.start()
        time.sleep(0.2)
        t0 = time.monotonic()
        srv.proc.send_signal(signal.SIGINT)
        time.sleep(0.2)
        with contextlib.suppress(OSError):  # refused at once (no grace), or already closed
            assert request(srv, chat(max_tokens=2), timeout=5)[0] == 503
        t.join(30)
        assert results[0][0] == 200 and srv.proc.wait(30) == 0 and time.monotonic() - t0 < 20
        srv.log.close()


@pytest.mark.parametrize("how", ["kill", "hang"])
def test_a_dead_or_hung_engine_fails_in_flight_requests_and_exits_1(model, tmp_path, how) -> None:
    """An engine crash (SIGKILL) or a hang (SIGSTOP, caught by engine.watchdog_s): in-flight requests fail at once
    (503; an error event on a stream that started), /health turns 503, and the server exits 1 for its restart policy."""
    with Server(model, tmp_path, step_ms=5,
                extra={"engine": {"watchdog_s": 2.0}, "server": {"drain_timeout_s": 30}}).start() as srv:
        results: dict[int, Any] = {}

        def one(i: int) -> None:
            results[i] = request(srv, chat(max_tokens=2000, seed=i, stream=bool(i % 2)))

        threads = [threading.Thread(target=one, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        time.sleep(0.5)
        pid = engine_pid(srv)
        t0 = time.monotonic()
        os.kill(pid, signal.SIGKILL if how == "kill" else signal.SIGSTOP)
        for t in threads:
            t.join(30)
        took = time.monotonic() - t0
        try:
            assert srv.proc.wait(30) == 1
        finally:
            if how == "hang":
                with contextlib.suppress(ProcessLookupError):
                    os.kill(pid, signal.SIGKILL)
        assert took < (8 if how == "hang" else 4), took  # not the 30 s drain timeout
        for i in range(4):
            status, _, body = results[i]
            if i % 2:  # streams had started: an error event, then [DONE]
                assert status == 200 and body[-1] == "[DONE]"
                err = next(e for e in body if isinstance(e, dict) and "error" in e)["error"]
                assert err["code"] == 503 and "retry" in err["message"]
            else:
                assert status == 503 and body["error"]["type"] == "ServiceUnavailableError", body
        srv.log.close()
    log = srv.log_path.read_text()
    assert ("llm-mini-infer-engine exited unexpectedly" in log) if how == "kill" else ("made no progress" in log)


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT], ids=["SIGTERM", "SIGINT"])
def test_a_stop_signal_while_starting_dumps_the_engines_stacks(model, tmp_path, sig) -> None:
    """A start held in the engine's model load (dev.runner.fake.load_s), then SIGTERM (a supervisor stopping a start
    it judged hung) or SIGINT: the launcher names the signal and the wait, the engine's faulthandler prints its threads'
    stacks (armed before the load: they show it), every process stops, and the launcher exits 128 + the signal."""
    srv = Server(model, tmp_path, extra={"dev": {"runner": {"fake": {"load_s": 600}}}}).launch()
    try:
        deadline = time.monotonic() + 60
        while "dev.runner.fake.load_s" not in srv.log_path.read_text():
            assert time.monotonic() < deadline and srv.proc.poll() is None, srv.log_path.read_text()[-2000:]
            time.sleep(0.1)
        t0 = time.monotonic()
        srv.proc.send_signal(sig)
        code = srv.proc.wait(30)
        took = time.monotonic() - t0  # the launcher's exit, not the stop's sweep below
    finally:
        if srv.proc.poll() is None:
            srv.proc.kill()
        srv.stop()  # an engine still in its model load does not watch its parent: the stop kills what is left
    log = srv.log_path.read_text()
    assert code == 128 + sig and took < 15, log[-3000:]
    assert f"{signal.Signals(sig).name} while starting (waiting for the engine)" in log, log[-3000:]
    assert "Fatal Python error: Aborted" in log and "in load_model" in log and "in engine_main" in log, log[-3000:]


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
def test_a_stop_before_the_drain_exits_at_once(sig) -> None:
    """Before the launcher's drain exists (a model download in the entrypoint's checks or the launcher), SIGTERM and
    SIGINT exit at once with 128 + the signal's number: as PID 1 (a pod without an init) a signal without a handler
    would be ignored until SIGKILL. At once also while a thread pool fetches files, as huggingface_hub's does: a
    `SystemExit` would wait for its threads."""
    code = ("import time\nfrom concurrent.futures import ThreadPoolExecutor\n"
            "from llm_mini_infer.launcher import exit_on_stop\n"
            "exit_on_stop()\nwith ThreadPoolExecutor(2) as ex:\n    ex.submit(time.sleep, 60)\n"
            "    print('ready', flush=True)\n    time.sleep(60)\n")
    p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                         cwd=ROOT)
    try:
        assert p.stdout.readline().strip() == "ready"
        t0 = time.monotonic()
        p.send_signal(sig)
        assert p.wait(timeout=10) == 128 + sig
        assert time.monotonic() - t0 < 5
        assert f"{signal.Signals(sig).name} before the server started" in p.stderr.read()
    finally:
        if p.poll() is None:
            p.kill()


def test_the_watchdog_rearms_after_an_oversleep() -> None:
    """A fake clock, a 10 s period: checks a second apart judge the heartbeat (stale past the period: stop the engine);
    a check 30 s after the previous one (the host froze the launcher with the engine) re-arms instead, and an engine
    that stays hung is then caught one period later. Off (0), it never judges."""
    beat, now = SimpleNamespace(value=0), [0]

    def at(s: float, w: Watchdog) -> float | None:
        now[0] = int(s * 1e9)
        return w.check()

    w = Watchdog(beat, 10.0, clock=lambda: now[0])
    assert all(at(s, w) is None for s in range(1, 11)) and at(11, w) == 11.0
    w = Watchdog(beat, 10.0, clock=lambda: now[0])  # armed at 11
    assert at(12, w) is None and at(13, w) is None
    assert at(43, w) is None and w.rearmed == 1 and beat.value == 43e9
    assert all(at(s, w) is None for s in range(44, 54)) and at(54, w) == 11.0
    w = Watchdog(beat, 0.0, clock=lambda: now[0])
    assert at(1000, w) is None and w.rearmed == 0


def test_a_host_freeze_does_not_trip_the_watchdog(model, tmp_path) -> None:
    """The engine and the launcher stopped together for 5 s (a whole-host freeze), the launcher resumed first: its next
    turn comes about 5 s late, so it re-arms the 2 s watchdog instead of stopping the engine, which resumes and serves
    on."""
    srv = Server(model, tmp_path, step_ms=5,
                 extra={"engine": {"watchdog_s": 2.0}, "observability": {"log_level": "info"}}).start()
    pid = None  # until the engine's is known: a failure before that still reaches the stop
    try:
        pid = engine_pid(srv)
        os.kill(pid, signal.SIGSTOP)
        srv.proc.send_signal(signal.SIGSTOP)
        time.sleep(5)
        srv.proc.send_signal(signal.SIGCONT)
        time.sleep(0.5)
        os.kill(pid, signal.SIGCONT)
        assert request(srv, chat(max_tokens=4))[0] == 200
        time.sleep(3)  # past the watchdog's period
        assert srv.proc.poll() is None and request(srv, chat(max_tokens=4))[0] == 200
    finally:
        if pid is not None:
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGCONT)
        srv.proc.send_signal(signal.SIGCONT)
        code = srv.stop()
    log = srv.log_path.read_text()
    assert code == 0 and "re-arming the engine's watchdog" in log and "made no progress" not in log, log[-3000:]


def test_a_config_without_runner_kind_needs_a_gpu(model, tmp_path) -> None:
    """runner.kind defaults to cuda: a config that names no runner, on a host where torch sees no GPU (this one, or a
    GPU host with CUDA_VISIBLE_DEVICES empty), exits 1 at startup naming the setting; it never answers with the fake
    runner's random tokens behind a passing /health."""
    srv = Server(model, tmp_path, runner=None, env={"CUDA_VISIBLE_DEVICES": ""},
                 extra={"images": {"enabled": False}})
    try:
        with pytest.raises(RuntimeError, match="server exited"):
            srv.start(timeout=120)
    finally:
        code = srv.stop()
    log = srv.log_path.read_text()
    assert code == 1 and "needs a CUDA GPU" in log and "runner.kind is fake" not in log, log[-2000:]
    assert re.search(r"engine startup failed: \w+: .*needs a CUDA GPU", log), log[-2000:]  # the engine's report


def test_a_second_signal_stops_at_once(model, tmp_path) -> None:
    with Server(model, tmp_path, step_ms=5, extra={"server": {"drain_timeout_s": 60}}).start() as srv:
        results: dict[int, Any] = {}
        t = threading.Thread(target=lambda: results.setdefault(0, request(srv, chat(max_tokens=3000, seed=3))))
        t.start()
        time.sleep(0.3)
        srv.proc.send_signal(signal.SIGTERM)
        time.sleep(0.3)
        srv.proc.send_signal(signal.SIGTERM)
        t.join(15)
        assert results[0][0] == 503 and srv.proc.wait(15) == 1
        srv.log.close()


@pytest.mark.serial  # a server with an image pool
def test_orphans_stop_when_the_launcher_is_killed(model, tmp_path) -> None:
    """The engine, the frontend and the image pool stop by themselves when the launcher is SIGKILLed. The launcher's
    IPC directory (its sockets) and pixel arena, which it removes on every exit it lives through, are the test's to
    remove here (/tmp/llm-mini-infer-*, the arena in /dev/shm on Linux)."""
    import shutil

    from llm_mini_infer.vision import arena

    psutil = pytest.importorskip("psutil")
    with Server(model, tmp_path, extra={"images": {}}).start() as srv:
        cfg = json.loads(srv.get("/server_info")[1])["config"]
        ipc_dir, pixels = cfg["pool_dir"], cfg["pixel_arena"]  # images on: the pool's directory is the IPC directory
        assert os.path.isdir(ipc_dir) and os.path.isfile(pixels), cfg
        try:
            children = psutil.Process(srv.proc.pid).children(recursive=True)
            assert len(children) >= 2  # the engine and the frontend
            srv.proc.kill()
            srv.proc.wait(10)
            _, alive = psutil.wait_procs(children, timeout=15)
            for p in alive:
                p.kill()
            srv.log.close()
            assert not alive, [p.name() for p in alive]
        finally:
            arena.remove(pixels)
            shutil.rmtree(ipc_dir, ignore_errors=True)
        assert not os.path.exists(ipc_dir) and not os.path.exists(pixels)


INFO = {"observability": {"log_level": "info"}}  # the launcher's `stopped (exit code N)` line, and the parent watch's


def test_the_launcher_stops_when_its_parent_exits(model, tmp_path) -> None:
    """server.stop_with_parent on (as the test Server has it): the launcher's parent SIGKILLed while the server serves,
    the launcher logs it, drains as on SIGTERM (an idle server at once) and exits 0, and every process it started stops
    with it."""
    psutil = pytest.importorskip("psutil")
    with Server(model, tmp_path, extra=INFO).start(under_parent=True) as srv:
        parent = srv.parent.pid
        children = psutil.Process(srv.proc.pid).children(recursive=True)
        assert len(children) >= 2  # the engine and the frontend
        t0 = time.monotonic()
        srv.kill_parent()
        code = srv.proc.wait(15)
        took = time.monotonic() - t0
        _, alive = psutil.wait_procs(children, timeout=15)
        for p in alive:
            p.kill()
        assert not alive, [p.name() for p in alive]
    log = srv.log_path.read_text()
    assert code == 0 and took < 15, log[-3000:]
    assert f"the launcher's parent (pid {parent}) exited (server.stop_with_parent): stopping as on SIGTERM" in log
    assert "SIGTERM: draining" in log, log[-3000:]


def test_a_launcher_whose_parent_exits_while_it_starts_stops(model, tmp_path) -> None:
    """The parent SIGKILLed while the engine loads (dev.runner.fake.load_s): the start ends as on SIGTERM
    (`StopWhileStarting`), and the launcher exits 143."""
    srv = Server(model, tmp_path, extra={**INFO, "dev": {"runner": {"fake": {"load_s": 600}}}}).launch_under_parent()
    try:
        deadline = time.monotonic() + 60
        while "dev.runner.fake.load_s" not in srv.log_path.read_text():
            assert time.monotonic() < deadline and srv.proc.poll() is None, srv.log_path.read_text()[-2000:]
            time.sleep(0.1)
        parent = srv.parent.pid
        t0 = time.monotonic()
        srv.kill_parent()
        code = srv.proc.wait(30)
        took = time.monotonic() - t0
    finally:
        if srv.proc.poll() is None:
            srv.proc.kill()
        srv.stop()
    log = srv.log_path.read_text()
    assert code == 143 and took < 15, log[-3000:]
    assert f"the launcher's parent (pid {parent}) exited" in log, log[-3000:]
    assert "SIGTERM while starting (waiting for the engine)" in log, log[-3000:]


def test_a_parent_that_stopped_the_launcher_before_it_exits_sends_no_second_signal(model, tmp_path) -> None:
    """A supervisor that sends SIGTERM and then exits (a wrapper, a test's stop): the watch finds the stop under way and
    sends nothing, so the drain (4 s of grace here: the watch looks meanwhile) ends with 0, not with a second signal's
    exit 1."""
    with Server(model, tmp_path, extra={**INFO, "server": {"drain_grace_s": 4}}).start(under_parent=True) as srv:
        parent = srv.parent.pid
        srv.proc.send_signal(signal.SIGTERM)
        srv.kill_parent()
        code = srv.proc.wait(30)
    log = srv.log_path.read_text()
    assert code == 0, log[-3000:]
    assert f"the launcher's parent (pid {parent}) exited; the server is stopping already" in log, log[-3000:]
    assert "again: failing in-flight requests" not in log and "stopping as on SIGTERM" not in log, log[-3000:]


def test_a_parent_that_exits_during_a_download_ends_it_at_once(tmp_path) -> None:
    """The watch runs from the start of `serve`, before its drain exists (a model download): the parent's exit reaches
    the main thread while it waits on a thread pool, as huggingface_hub's, and `exit_on_stop` ends the process at once
    (its line is the last thing it does before `os._exit(143)`; the exit code goes to an init, not to this test)."""
    code = ("import logging, os, time\nfrom concurrent.futures import ThreadPoolExecutor\n"
            "from types import SimpleNamespace\nfrom llm_mini_infer.launcher import _watch_parent, exit_on_stop\n"
            "logging.basicConfig(level=logging.INFO)\nexit_on_stop()\n"
            "_watch_parent(SimpleNamespace(server=SimpleNamespace(stop_with_parent=True)), os.getppid(), [])\n"
            "with ThreadPoolExecutor(2) as ex:\n    f = ex.submit(time.sleep, 60)\n    print('ready', flush=True)\n"
            "    f.result()\n")
    out_path = tmp_path / "child.log"
    with out_path.open("w") as out:
        parent, pid = start_under_parent([sys.executable, "-c", code], dict(os.environ), out)
    child = Adopted(pid, out_path)
    try:
        deadline = time.monotonic() + 30
        while "ready" not in out_path.read_text():
            assert time.monotonic() < deadline and child.poll() is None, out_path.read_text()
            time.sleep(0.05)
        t0 = time.monotonic()
        parent.kill()
        parent.wait(10)
        child.wait(10)
        took = time.monotonic() - t0
    finally:
        child.kill()
        parent.kill()
        parent.wait(10)
        parent.stderr.close()
    text = out_path.read_text()
    assert took < 5, text
    assert "exited (server.stop_with_parent): stopping as on SIGTERM" in text, text
    assert text.rstrip().endswith("SIGTERM before the server started: exiting"), text


def test_the_launcher_outlives_its_parent_by_default(model, tmp_path) -> None:
    """server.stop_with_parent off (the default): the parent's exit changes nothing, as for a server started with
    nohup; the launcher serves on until it is stopped."""
    with Server(model, tmp_path, extra={**INFO, "server": {"stop_with_parent": False}}).start(under_parent=True) as srv:
        srv.kill_parent()
        time.sleep(3)  # three of the watch's looks, were it on
        assert srv.proc.poll() is None and srv.get("/health")[0] == 200
        assert srv.stop() == 0, srv.log_path.read_text()[-3000:]
    assert "the launcher's parent" not in srv.log_path.read_text()


def test_a_parent_of_pid_1_or_0_is_not_watched(caplog) -> None:
    """Pid 1 (an init, or what an orphan's parent becomes) outlives the launcher, and 0 is no parent it can see (it is
    pid 1, or its parent is in another pid namespace): no thread, one line saying so. Off, nothing is watched."""
    on = SimpleNamespace(server=SimpleNamespace(stop_with_parent=True))
    with caplog.at_level(logging.INFO, logger="llm_mini_infer.launcher"):
        assert _watch_parent(on, 1, []) is None and _watch_parent(on, 0, []) is None
    assert "the launcher's parent is pid 1, which outlives it: not watched" in caplog.text, caplog.text
    assert "the launcher has no parent it can see (pid 0): not watched" in caplog.text, caplog.text
    assert _watch_parent(SimpleNamespace(server=SimpleNamespace(stop_with_parent=False)), os.getppid(), []) is None


@pytest.mark.serial  # a server with an image pool
def test_logs_never_carry_request_contents(model, tmp_path) -> None:
    """At DEBUG, nothing a client sent (text, system prompt, schema, stop strings, image bytes) or got back reaches the
    log, failing requests included."""
    from PIL import Image

    with Server(model, tmp_path,
                extra={"observability": {"log_level": "debug"},
                       "images": {"min_pixels": 65536, "max_pixels": 327680, "workers": 1}}).start(120) as srv:
        secret = "zq-secret-marker-7731"
        buf = io.BytesIO()
        Image.new("RGB", (64, 48), (10, 200, 30)).save(buf, "PNG")
        uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        schema = {"type": "object", "properties": {secret: {"type": "string", "description": secret}},
                  "required": [secret], "additionalProperties": False}
        msgs = [{"role": "system", "content": f"You are {secret}."},
                {"role": "user", "content": [{"type": "text", "text": f"Describe {secret}"},
                                             {"type": "image_url", "image_url": {"url": uri}}]}]
        outputs = []
        for body in ({"messages": msgs, "max_tokens": 12, "seed": 1, "stop": [secret]},
                     {"messages": msgs, "max_tokens": 12, "seed": 2, "stream": True},
                     {"messages": msgs, "max_tokens": 40, "seed": 3,
                      "response_format": {"type": "json_schema", "json_schema": {"name": "x", "schema": schema}}},
                     {"messages": msgs, "max_tokens": 5, "response_format": {
                         "type": "json_schema", "json_schema": {"name": "x", "schema": {"type": secret}}}},  # a 400
                     {"messages": [{"role": secret, "content": secret}]},  # a 400 that quotes the role in its message
                     {"messages": msgs, "max_tokens": 4, "logit_bias": {"1": 1}, "user": secret}):  # 400
            status, _, out = request(srv, {**body, "chat_template_kwargs": KW})
            if status == 200:
                text = content_of(out) if isinstance(out, list) else out["choices"][0]["message"]["content"]
                outputs.extend(w for w in text.split() if len(w) >= 12)  # long output words: specific enough to grep
        assert srv.stop() == 0
    log = srv.log_path.read_text()
    assert "request id=" in log and "status=200" in log and "status=400" in log  # the request lines are there
    for needle in (secret, uri[30:90], *outputs):
        assert needle not in log, needle


# The engine process's entry point, its step loop replaced by one that fails while handling a request: its exception
# carries the request's content in its message and a note, and so does the exception it was raised from
CRASH = """
import sys
from types import SimpleNamespace

from llm_mini_infer import config
from llm_mini_infer.engine import process

SECRET = "zq-secret-marker-7731"


def parse(text):
    raise ValueError(f"cannot parse {text!r}")


def run(self):
    try:
        parse(f"Describe {SECRET}")
    except ValueError as exc:
        err = RuntimeError(f"step failed for {SECRET}")
        err.add_note(SECRET)
        raise err from exc


process.EngineLoop.run = run
cfg = config.derive(config.load(None, {"model.path": sys.argv[1], "runner.kind": "fake", "images.enabled": False}))
process.engine_main(cfg, SimpleNamespace(send=lambda info: None, close=lambda: None), sys.argv[2])
"""


def test_a_frontends_logged_exceptions_carry_no_message() -> None:
    """The frontends' and the image pool's handler (`setup_logging(..., without_messages=True)`): an exception logged
    with its traceback, as uvicorn logs an application's ("Exception in ASGI application"), keeps its type and frames
    but not its message, which can quote a request."""
    import logging

    from llm_mini_infer.logutil import setup_logging

    root = logging.getLogger()
    saved = root.handlers[:], root.level
    stream = io.StringIO()
    try:
        setup_logging("info", "frontend-0", without_messages=True)
        root.handlers[0].stream = stream
        secret = "a secret " + "prompt"  # built at run time, as a request's contents are: no source line shows it
        try:
            raise ValueError(secret)
        except ValueError:
            logging.getLogger("uvicorn.error").exception("Exception in ASGI application")
    finally:
        root.handlers[:], root.level = saved
    out = stream.getvalue()
    assert "Exception in ASGI application" in out and "ValueError" in out and "test_lifecycle.py" in out
    assert "a secret prompt" not in out


def test_an_engine_crash_logs_no_exception_text(model) -> None:
    """The engine loop's crash path logs the exception's type and frames, and those of the exception it was raised
    from, but none of their messages or notes, which can quote a request; the engine exits 1. An exception raised while
    handling another is printed after it, as Python prints them, and an exception group's members after the group."""
    import shutil

    from llm_mini_infer.engine import ipc
    from llm_mini_infer.logutil import traceback_without_messages

    secret = "zq-secret-marker-7731"

    def fail() -> None:
        raise OSError(secret)

    try:
        try:
            {}[secret]
        except KeyError:
            fail()  # the KeyError is the OSError's context
    except OSError as exc:
        text = traceback_without_messages(exc)
    during = text.index("During handling of the above exception, another exception occurred:")
    assert secret not in text and text.index("\nKeyError\n") < during < text.index("in fail") and \
        text.endswith("\nOSError\n"), text
    try:
        try:
            fail()
        except OSError as exc:
            raise ExceptionGroup(secret, [exc, KeyError(secret)]) from None
    except ExceptionGroup as group:
        text = traceback_without_messages(group)
    assert secret not in text and text.index("\nExceptionGroup\n+---------------- 1 ----------------\n") < \
        text.index("|   File") < text.index("in fail") < text.index("| OSError\n+---------------- 2 ----------------\n"
                                                                     "| KeyError\n+-----"), text

    ipc_dir = ipc.ipc_dir()
    env = {k: v for k, v in os.environ.items() if k not in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")}
    try:
        out = subprocess.run([sys.executable, "-c", CRASH, str(model), ipc_dir], cwd=ROOT, env=env,
                             capture_output=True, text=True, timeout=120)
    finally:
        shutil.rmtree(ipc_dir, ignore_errors=True)
    log = out.stderr
    assert out.returncode == 1 and "engine loop failed (exception messages left out)" in log, log[-3000:]
    assert "zq-secret-marker-7731" not in log and "cannot parse" not in log and "step failed" not in log, log[-3000:]
    cause = log.index("The above exception was the direct cause of the following exception:")
    assert log.index("in parse") < log.index("\nValueError\n") < cause < log.index("\nRuntimeError\n")
    assert log.rindex("in run") > cause and log.index("in engine_main") > cause
