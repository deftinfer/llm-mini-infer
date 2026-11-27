"""The benchmark kit (bench/README.md) against the code it drives: the load generator's workloads, summary and answer
digests, and short loads of it against the fake runner's server (its answers, and the engine's counters over the
window); the comparison of arms; the grammar-cost script's documents, which the catalog schema must accept; the validity
script's requests and verdicts. The scripts are loaded from their files (bench/ is no package)."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import os
import random
import re
import signal
import sys
import time

import jsonschema
import psutil
import pytest

from llm_mini_infer.hostcheck import GENERATION_SERIES, VERIFY_SERIES, counter, engine_delta, prometheus

from ..support.api_util import Server, env_has_model
from ..support.catalog_schema import NAME as CATALOG
from ..support.catalog_schema import PATH as CATALOG_SCHEMA
from ..support.catalog_schema import prompt as catalog_prompt
from ..support.metrics_pages import VLLM_METRICS
from ..support.paths import MODELS_DIR, ROOT
from ..support.schema_corpus import CORPUS

GEMMA = MODELS_DIR / "gemma-4-31b-it"
needs_gemma = pytest.mark.skipif(not env_has_model(GEMMA), reason="tests/.fixtures/models/gemma-4-31b-it is absent: "
                                                                  "uv run tests/scripts/fetch_fixtures.py")


def script(name: str, rel: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


load = script("bench_load_run", "bench/load/run.py")
compare = script("bench_load_compare", "bench/load/compare.py")
grammar_cost = script("bench_grammar_cost", "bench/structured/grammar_cost.py")
validity = script("bench_validity", "bench/structured/validity.py")
server_run = script("bench_server_run", "bench/server/run.py")
server_report = script("bench_server_report", "bench/server/report.py")


# -- the load generator --------------------------------------------------------------------------------------------


def test_summary_counts_the_measured_requests_and_their_rates() -> None:
    ok = {"measured": True, "status": 200, "e2e_ms": 100.0, "ttft_ms": 20.0, "tokens": 9, "prompt_tokens": 50,
          "finish": "stop"}
    recs = [ok, {**ok, "e2e_ms": 300.0, "tokens": 5, "finish": "length"},
            {"measured": True, "dropped": True},
            {"measured": True, "status": 400, "error": "400 BadRequestError", "e2e_ms": 3.0},
            {"measured": True, "status": 0, "error": "timeout", "e2e_ms": 9.0},
            {**ok, "measured": False}]  # warm-up and tail: not counted
    events = [(0.0, +1), (0.5, +1), (1.0, -1), (1.5, -1)]
    s = load.summarize(recs, events, 0.0, 2.0, 2.0)
    assert (s["measured"], s["ok"], s["dropped"], s["failed"]) == (5, 2, 1, 2)
    assert s["errors"] == {"400 BadRequestError": 1, "timeout": 1} and s["finish"] == {"stop": 1, "length": 1}
    assert (s["offered_rps"], s["served_rps"], s["output_tokens_per_s"]) == (2.5, 1.0, 7.0)
    assert s["e2e_ms"]["p50"] == 100.0 and s["e2e_ms"]["max"] == 300.0 and s["ttft_ms"]["n"] == 2
    assert s["tpot_ms"]["max"] == 70.0 and s["tpot_ms"]["p50"] == 10.0  # (e2e - ttft) / (tokens - 1)
    assert s["in_flight"]["max"] == 2 and s["in_flight"]["mean"] == 1.0
    assert load.summarize([], [], 0.0, 0.0, 0.0)["served_rps"] == 0.0  # an empty window divides by nothing


def test_arrivals_are_seeded_and_at_the_rate() -> None:
    a = load.arrivals(5.0, 10.0, 60.0, 0.0, seed=3)
    assert a == load.arrivals(5.0, 10.0, 60.0, 0.0, seed=3) != load.arrivals(5.0, 10.0, 60.0, 0.0, seed=4)
    assert a == sorted(a) and a[0] == 0.0 and a[-1] < 70.0 and abs(len(a) - 350) <= 2


def test_the_schema_workloads_send_one_new_image_each(tmp_path) -> None:
    from llm_mini_infer.hostcheck import mounted_inputs, synthetic_inputs
    from llm_mini_infer.synthetic import write_inputs

    scene = load.schema_workload(synthetic_inputs(lambda text: len(text) // 4, 256, 0.25), 64)
    a, b = scene.body(0), scene.body(1)
    assert scene.count is None and a["max_tokens"] == 64
    assert a["response_format"]["json_schema"]["strict"] and a["response_format"]["json_schema"]["name"] == "scene"
    urls = [x["messages"][1]["content"][1]["image_url"]["url"] for x in (a, b)]
    assert urls[0].startswith("data:image/jpeg;base64,") and urls[0] != urls[1]
    assert a["messages"][0] == b["messages"][0]  # the system prompt every request shares
    write_inputs(tmp_path / "inputs", 3, lambda text: len(text) // 4)
    folder = load.schema_workload(mounted_inputs(tmp_path / "inputs"), 32)
    assert folder.count == 3 and folder.summary["frames"] == 3 and folder.name == "folder"


def test_the_catalog_workload_asks_for_distinct_boxes_under_the_strict_schema() -> None:
    """Text requests under the catalog schema, strict, its list of entries at least three long (so that every answer
    decodes some); the validity script's prompts with the box numbered, so no two are the same; with request
    seeds, each request's seed its workload index."""
    catalog = load.catalog_workload(256)
    bodies = [catalog.body(i) for i in range(16)]
    assert catalog.count is None and catalog.name == "catalog" and bodies[0]["max_tokens"] == 256
    rf = bodies[0]["response_format"]["json_schema"]
    sent = json.loads(json.dumps(rf["schema"]))
    min_items = sent["properties"]["entries"].pop("minItems")
    assert min_items == load.CATALOG_MIN_ENTRIES == catalog.summary["min_entries"] == 3
    assert rf["strict"] and rf["name"] == CATALOG and sent == json.loads(CATALOG_SCHEMA.read_text())
    texts = [b["messages"][0]["content"] for b in bodies]
    assert len(set(texts)) == 16 and texts[3] == catalog_prompt(3, numbered=True) and " box 3 " in texts[3]
    assert [r["body"]["messages"][0]["content"] for r in validity.build_requests(8, 64)[:8]] == \
        [catalog_prompt(i) for i in range(8)]
    assert all("seed" not in b for b in bodies)
    run = load.Run(None, "", {}, catalog, {"model": "m"}, 1.0, 0.0, 1.0, 0.0, offset=5, seeds=True)
    assert run.body(2)["seed"] == 7 and run.body(2)["model"] == "m" and run.body(2)["messages"] == bodies[7]["messages"]


def test_a_slow_metrics_page_holds_back_no_arrival(monkeypatch) -> None:
    """The open loop reads the server's counters beside its arrivals: a /metrics page that takes 0.4 s delays none of
    them, and the counters are still read as the window opens and as it closes."""
    sent, reads = [], []

    async def scrape(run):
        reads.append(time.monotonic())
        await asyncio.sleep(0.4)
        return reads[-1], {}

    async def send(session, url, headers, body, timeout):
        sent.append(time.monotonic())
        return {"status": 200, "e2e_ms": 1.0, "tokens": 1, "finish": "stop", "digest": "0" * 16}

    monkeypatch.setattr(load, "scrape", scrape)
    monkeypatch.setattr(load, "send", send)
    run = load.Run(None, "", {}, load.catalog_workload(8), {}, 5.0, 0.3, 0.3, 0.3)
    r = asyncio.run(load.open_loop(run, 40.0, 64, seed=0))
    at = load.arrivals(40.0, 0.3, 0.3, 0.3, seed=0)
    t0 = sent[0] - at[0]  # the first arrival is at 0
    assert len(sent) > 15 and max(t - t0 - a for t, a in zip(sent, at)) < 0.2  # each sent when it arrives
    assert len(reads) == 2 and abs(reads[0] - t0 - 0.3) < 0.1 and abs(reads[1] - t0 - 0.6) < 0.1
    assert r["engine"] is not None and r["dropped"] == 0


class _Response:
    """An aiohttp response as `send` reads it: a status, then a JSON body or SSE lines."""

    def __init__(self, status: int, payload: dict | None = None, lines: tuple[str, ...] = ()):
        self.status, self.payload, self.content = status, payload, self._lines(lines)

    @staticmethod
    async def _lines(lines):
        for line in lines:
            yield line.encode()

    async def json(self) -> dict:
        return self.payload

    async def text(self) -> str:
        return json.dumps(self.payload)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc) -> bool:
        return False


class _Session:
    def __init__(self, resp: _Response):
        self.resp = resp

    def post(self, url: str, **kwargs) -> _Response:
        return self.resp


def test_an_answers_digest_is_its_contents_streamed_or_not() -> None:
    """The digest of an answer's content as it arrives (16 hex digits of its SHA-256, as the host check's) is that of
    the whole content, streamed or not; a failed request has none; a load keeps its answers by workload index."""
    text = '{"n":1,"b":[]}'
    want = hashlib.sha256(text.encode()).hexdigest()[:16]
    chunks = [{"choices": [{"delta": {"role": "assistant"}}]},
              *({"choices": [{"delta": {"content": c}}]} for c in ('{"n":1', ",", '"b":[]}')),
              {"choices": [{"delta": {}, "finish_reason": "stop"}]},
              {"choices": [], "usage": {"completion_tokens": 6, "prompt_tokens": 9}}]
    lines = (*(f"data: {json.dumps(c)}\n" for c in chunks), "data: [DONE]\n")
    streamed = asyncio.run(load.send(_Session(_Response(200, lines=lines)), "u", {}, {"stream": True}, 5.0))
    plain = asyncio.run(load.send(_Session(_Response(200, {
        "choices": [{"message": {"content": text}, "finish_reason": "stop"}], "usage": {"completion_tokens": 6}})),
        "u", {}, {}, 5.0))
    assert streamed["digest"] == plain["digest"] == want and streamed["tokens"] == plain["tokens"] == 6
    failed = asyncio.run(load.send(_Session(_Response(400, {"error": {"type": "BadRequestError"}})), "u", {}, {}, 5.0))
    assert failed["error"] == "400 BadRequestError" and "digest" not in failed
    recs = [{**streamed, "index": 7}, {**plain, "index": 3, "finish": "length"}, {**failed, "index": 4},
            {"dropped": True}]
    assert load.answers(recs) == {"3": [want, 6, "length"], "7": [want, 6, "stop"]}


@needs_gemma
def test_the_chat_workload_sizes_its_distinct_prompts() -> None:
    tok = load.load_tokenizer(str(GEMMA))
    chat = load.chat_workload(tok, 200, 16, seed=0)
    texts = [chat.body(i)["messages"][0]["content"] for i in range(4)]
    assert len(set(texts)) == 4 and all(abs(len(tok.encode(t, add_special_tokens=False).ids) - 200) <= 4
                                        for t in texts)
    assert chat.body(0) == load.chat_workload(tok, 200, 16, seed=0).body(0)


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("bench")
    srv = Server(GEMMA, tmp, step_ms=1.0, extra={"model": {"max_model_len": 8192}, "images": {}}).start(120)
    yield srv
    assert srv.stop() == 0, srv.log_path.read_text()[-3000:]


@pytest.mark.serial  # the module's Gemma server has an image pool, for the scene load's images
@needs_gemma
@pytest.mark.parametrize("args", [
    ["--workload", "chat", "--concurrency", "2", "--input-tokens", "64", "--max-tokens", "8"],
    ["--workload", "scene", "--rates", "4", "--slots", "4", "--no-metrics", "--max-tokens", "16"],
    ["--workload", "catalog", "--rates", "4", "--slots", "4", "--request-seeds", "--temperature", "0.8",
     "--max-tokens", "24"],
], ids=["chat-concurrency", "scene-rate", "catalog-rate"])
def test_a_short_load_against_the_fake_server(server, tmp_path, args) -> None:
    """Each load keeps every answer's digest, tokens and finish reason by workload index, and (unless --no-metrics)
    the engine's counters over its window: the tokens it generated (within the answers'), the grammar heads (zeros
    from the fake runner), the forced prefix, the EOS-free finish, no failed request."""
    out = tmp_path / "load.json"
    status = load.main(["--base-url", f"http://127.0.0.1:{server.port}", "--model", "qwen-test", "--tokenizer",
                        str(GEMMA), "--warmup", "0.5", "--duration", "1.5", "--tail", "20", "--out", str(out), *args])
    report = json.loads(out.read_text())
    (r,) = report["loads"]
    assert status == 0 and r["failed"] == 0 and r["ok"] > 0 and r["ok"] + r["dropped"] == r["measured"]
    assert r["ttft_ms"]["n"] == r["ok"] and r["reply_tokens"]["max"] <= int(args[-1])
    assert report["workload"]["name"] == args[1] and report["request_seeds"] == ("--request-seeds" in args)
    answers = r["answers"]  # warm-up and tail included
    assert len(answers) >= r["ok"] and all(re.fullmatch(r"[0-9a-f]{16}", d) and 0 < n <= int(args[-1]) and f
                                           for d, n, f in answers.values())
    if "--no-metrics" in args:
        assert r["engine"] is None
        return
    eng = r["engine"]
    assert eng["requests_failed"] == 0 and eng["grammar_errors"] == 0 and eng["plans_kept_since_start"] == 0
    assert "accept" not in eng and 0 <= eng["gpu_idle_pct"] <= 100  # no drafts on the fake runner
    # the window's tokens: some, and no more than every answer's (warm-up and tail included): the counters were read
    # as the window opened, then as it closed
    assert 0 < eng["generation_tokens"] <= sum(n for _, n, _ in answers.values())
    if args[1] == "catalog":  # structured: the grammar's counters are exported, the heads' zeros
        assert set(eng["head_steps"]) == {"sparse", "dense"} and eng["union_tokens"] is None
        assert eng["forced_requests"] is not None and eng["draft_universe_steps"] == 0
        assert set(eng["eos_free_finishes"]) == {"no", "yes"} and eng["eos_free_gated"] is not None


# -- the comparison of arms ----------------------------------------------------------------------------------------


def _arm(root, name: str, digests: list[str], engine: dict | None, texts: list[str] | None = None) -> None:
    """An arm's folder: a run.py report of one catalog load, a file that is no report, and optionally validity.py's."""
    d = root / name
    d.mkdir()
    row = {"load": "rate 2/s, 50 slots", "measured": 20, "ok": 20, "finish": {"stop": 19, "length": 1},
           "served_rps": 2.0, "output_tokens_per_s": 101.5, "e2e_ms": {"n": 20, "p50": 900.0, "p99": 1500.0},
           "ttft_ms": {"n": 20, "p50": 80.0, "p99": 200.0}, "tpot_ms": {"n": 20, "p50": 12.5, "p99": 20.25},
           "reply_tokens": {"n": 20, "mean": 50.0}, "engine": engine,
           "answers": {str(100 + i): [x, 50, "stop"] for i, x in enumerate(digests)}}
    (d / "catalog.json").write_text(json.dumps({"workload": {"name": "catalog", "mode": "catalog"}, "loads": [row]}))
    (d / "server_info.json").write_text(json.dumps({"config": {}}))
    if texts:
        (d / "validity.json").write_text(json.dumps({
            "summary": {"all": {"requests": 3, "stop": 3, "valid": 3, "invalid": 0, "errors": 0}},
            "results": [{"id": f"catalog-{i}", "text": t, "tokens": 9, "finish": "stop"}
                        for i, t in enumerate(texts)]}))


def test_compare_tables_the_arms_and_judges_their_answers_against_the_floor(tmp_path, capsys) -> None:
    """Per load, the client's and the engine's rows of each arm (a dash where a server does not count: a vLLM arm
    has its acceptance and none of llm-mini-infer's counters, an arm without a page dashes only); each pair's agreement,
    judged against two passes of one arm: an arm that agrees with A1 as A2 does or better is not below the floor, one
    that agrees on half is."""
    eng = {"accept": {"drafts": 100, "per_pos": [0.9, 0.6], "tokens_per_verify": 2.5}, "sparse_share": 0.97,
           "union_tokens": {"steps": 50, "mean": 40.54, "p50": 32, "p90": 64}, "draft_universe_tokens": 351,
           "forced_tokens_per_request": 2.0, "eos_free_finishes": {"no": 3, "yes": 1}, "gpu_idle_pct": 0.5,
           "requests_failed": 0, "grammar_errors": 0}
    base = [f"{i:016x}" for i in range(20)]
    _arm(tmp_path, "A1", base, eng, ["x", "y", "z"])
    _arm(tmp_path, "A2", ["f" * 16, "e" * 16, *base[2:]], eng)  # two answers moved: the floor, 90%
    _arm(tmp_path, "B1", base, eng, ["x", "y", "w"])
    _arm(tmp_path, "C", ["c" * 16] * 10 + base[10:], None)
    vllm = engine_delta(*(prometheus(VLLM_METRICS % kw) for kw in (dict(drafts=10, p0=8, p1=6, p2=4, gen=100),
                                                                   dict(drafts=110, p0=88, p1=66, p2=44, gen=500))),
                        completed=20)
    _arm(tmp_path, "vllm", base, vllm)
    status = compare.main([str(tmp_path / n) for n in ("A1", "A2", "B1", "vllm")] + [f"V={tmp_path / 'C'}",
                                                                                    "--floor", "A1,A2"])
    out = capsys.readouterr().out
    assert status == 0 and "== catalog: rate 2/s, 50 slots" in out and "== validity.py" in out
    a1 = [line.split() for line in out.splitlines() if line.startswith("A1 ")]
    assert ["A1", "20/20", "length:1", "stop:19", "2", "101.5", "900", "1500", "80", "200", "12.5", "20.25", "50",
            "-"] in a1
    assert ["A1", "2.5", "0.90/0.60", "0.97", "40.5", "<=64", "351", "2", "4", "(1)", "0.5", "0/0"] in a1
    assert ["vllm", "2.8", "0.80/0.60/0.40", *["-"] * 7, "-/-"] in [line.split() for line in out.splitlines()
                                                                      if line.startswith("vllm ")]
    assert ["V", *["-"] * 10] in [line.split() for line in out.splitlines()]  # the engine's row of a server without
    assert "A1 vs A2: 18/20 identical (90.0%)" in out
    assert re.search(r"A1 vs B1: 20/20 identical \(100\.0%\); against the floor \(A1 vs A2, 20 answers\): gap \+10\.0 "
                     r"points \[[^]]+\]: not below the floor", out)
    assert re.search(r"A1 vs V: 10/20 identical \(50\.0%\); .*: gap -40\.0 points \[[^]]+\]: BELOW the floor", out)
    assert "A1 vs B1: 2/3 identical (66.7%)" in out.split("== validity.py")[1]
    with pytest.raises(SystemExit, match="two arms named A1"):
        compare.main([str(tmp_path / "A1"), str(tmp_path / "A1")])
    with pytest.raises(SystemExit, match="no arm named X"):
        compare.main([str(tmp_path / "A1"), "--floor", "A1,X"])


# -- structured output ---------------------------------------------------------------------------------------------


def test_the_grammar_cost_documents_are_valid_catalogs() -> None:
    check = jsonschema.Draft202012Validator(grammar_cost.SCHEMA)
    rng = random.Random(0)
    docs = grammar_cost.documents(40, rng) + grammar_cost.documents(40, rng, compact=True)
    for d in docs:
        check.validate(json.loads(d))
    assert all(not re.search(r"\s", d) for d in docs[40:]) and any(json.loads(d)["entries"] for d in docs)


def test_the_validity_requests_and_verdicts() -> None:
    assert {r["group"] for r in validity.build_requests(0, 128)} == {"corpus"}
    reqs = validity.build_requests(8, 128)
    assert len(reqs) == 8 + 3 * len(CORPUS) and {r["group"] for r in reqs} == {"catalog", "corpus"}
    strict = [r["body"]["response_format"]["json_schema"]["strict"] for r in reqs if r["group"] == "catalog"]
    assert strict.count(True) == 5 and any(r["stream"] for r in reqs) and not all(r["stream"] for r in reqs)
    catalog = validity.SCHEMA
    assert validity.validate('{"total": 0, "entries": []}', catalog) is None
    assert validity.validate('{"total": 0, "entries": [], "total": 1}', catalog).startswith("duplicate keys")
    assert validity.validate('{"total": 0, "entries": [', catalog).startswith("not JSON")
    assert validity.validate('{"total": "zero", "entries": []}', catalog).startswith("invalid")


def test_the_server_benchmarks_tag_and_report_name_what_they_ran(tmp_path, monkeypatch, capsys) -> None:
    """bench/server: a result's -sleep tag only where the fake's wait applies (the TTFT benchmark's server); the report
    names every host its files come from, and warns when they differ (it showed the last file's alone)."""
    tag = server_run.result_tag
    assert [tag(w, False, "sleep") for w in ("decode", "sse", "idle")] == ["", "", ""]
    assert [tag(w, False, "sleep") for w in ("ttft", "ttft+idle", "all")] == ["-sleep"] * 3
    assert tag("idle", True, "spin") == "-warm" and tag("ttft", True, "sleep") == "-warm-sleep"
    files = []
    for name, host in (("a.json", "Linux-a"), ("b.json", "Linux-b"), ("c.json", "Linux-a")):
        (tmp_path / name).write_text(json.dumps({"host": host, "cores": 8, "python": "3.12"}))  # no tables
        files.append(str(tmp_path / name))

    def report(*paths: str) -> tuple[str, str]:
        monkeypatch.setattr(sys, "argv", ["report.py", *paths])
        server_report.main()
        out = capsys.readouterr()
        return out.out.splitlines()[0], out.err

    line, err = report(*files)
    assert line == ("Hosts (MIXED): Linux-a, 8 cores, Python 3.12 (a.json, c.json); Linux-b, 8 cores, Python 3.12 "
                    "(b.json)")
    assert "2 hosts" in err
    assert report(files[0], files[2]) == ("Host: Linux-a, 8 cores, Python 3.12", "")


# -- the server wrapper ---------------------------------------------------------------------------------------------


@needs_gemma
def test_the_bench_server_starts_on_the_fake_runner_and_stops_idempotently() -> None:
    srv = server_run.Server(1.0)
    try:
        assert srv.cfg_path.exists() and srv.proc.poll() is None and srv.frontends and srv.engine.is_running()
    finally:
        srv.stop()
    assert srv.proc.returncode is not None and not srv.cfg_path.exists()
    srv.stop()  # a second stop does nothing


def test_the_bench_server_passes_its_shutdown_policy_to_the_shared_helper(tmp_path, monkeypatch) -> None:
    calls = []

    def stop(proc, *, first_signal, waits):
        calls.append((proc, first_signal, waits))

    monkeypatch.setattr(server_run.server_process, "stop", stop)
    srv = object.__new__(server_run.Server)
    srv.proc, srv.cfg_path, srv._stopped = object(), tmp_path / ".server-1.yaml", False
    srv.cfg_path.write_text("server: {}")
    srv.stop(wait=60, kill_wait=10)
    assert calls == [(srv.proc, signal.SIGTERM, (60, 10))]
    assert not srv.cfg_path.exists()
    srv.stop()
    assert calls == [(srv.proc, signal.SIGTERM, (60, 10))]
    srv.proc = None
    srv._stopped = False
    srv.cfg_path.write_text("server: {}")
    srv.stop()
    assert not srv.cfg_path.exists()


def _running_group(pgid: int) -> list[int]:
    pids = []
    for proc in psutil.process_iter():
        try:
            if os.getpgid(proc.pid) == pgid:
                pids.append(proc.pid)
        except (OSError, psutil.Error):
            pass
    return pids


@needs_gemma
@pytest.mark.skipif(not hasattr(os, "killpg"), reason="no process groups on this platform")
def test_the_bench_server_kills_its_group_after_the_launcher_is_stopped(monkeypatch) -> None:
    monkeypatch.setattr(server_run, "MODEL", GEMMA)
    monkeypatch.setattr(server_run, "RUNNER", "fake")
    srv = server_run.Server(1.0)
    pgid = srv.proc.pid
    try:
        assert os.getpgid(srv.proc.pid) == pgid
        os.kill(srv.proc.pid, signal.SIGSTOP)
        srv.stop(wait=0.1)
        assert srv.proc.returncode == -signal.SIGKILL
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and _running_group(pgid):
            time.sleep(0.05)
        assert _running_group(pgid) == []
        assert not srv.cfg_path.exists()
    finally:
        if srv.proc.poll() is None:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except OSError:
                pass
            srv.proc.wait(timeout=10)
        srv.stop()


def test_a_failed_start_stops_the_bench_server_and_removes_its_config(monkeypatch) -> None:
    started = []
    real = server_run.subprocess.Popen

    def popen(*a, **kw):
        started.append(real(*a, **kw))
        return started[-1]

    def no_engine(self, path):
        raise RuntimeError("no engine")

    monkeypatch.setattr(server_run.subprocess, "Popen", popen)
    monkeypatch.setattr(server_run.Server, "get", lambda self, path: (200, b"") if path.startswith("/health") else
                        no_engine(self, path))
    with pytest.raises(RuntimeError, match="no engine"):
        server_run.Server(1.0)
    (proc,) = started
    assert proc.returncode is not None
    assert not list(server_run.RESULTS.glob(f".server-{proc.args[-1].split('-')[-1]}"))


def test_the_bench_server_notices_a_child_that_exited_and_waits_through_non_200s(monkeypatch) -> None:
    class Dead:
        returncode = 3

        def poll(self):
            return 3

        def wait(self, timeout=None):
            return 3

    monkeypatch.setattr(server_run.subprocess, "Popen", lambda *a, **kw: Dead())
    monkeypatch.setattr(server_run.Server, "get", lambda self, path: pytest.fail("polled a dead child's port"))
    with pytest.raises(RuntimeError, match=r"exited during startup \(code 3\)"):
        server_run.Server(1.0)
    answers = iter([503, 503, 200])
    slept = []
    monkeypatch.setattr(server_run.time, "sleep", slept.append)

    class Live:
        returncode = None

        def poll(self):
            return None

        def send_signal(self, sig):
            pass

        def wait(self, timeout):
            return 0

    monkeypatch.setattr(server_run.subprocess, "Popen", lambda *a, **kw: Live())
    monkeypatch.setattr(server_run.Server, "get", lambda self, path: (next(answers), b""))
    with pytest.raises(StopIteration):  # past the third answer the stub has nothing more: the loop got there
        server_run.Server(1.0)
    assert len(slept) == 2  # one sleep for each 503


SGLANG_PAGE = """sglang:spec_verify_calls_total{tp_rank="0",model_name="m"} %(calls)s
sglang:spec_verify_calls_total{tp_rank="1",model_name="m"} %(calls)s
sglang:generation_tokens_total{model_name="m"} %(gen)s
"""


def test_counter_takes_the_first_exported_series_summed_over_its_labels() -> None:
    vllm = prometheus(VLLM_METRICS % dict(drafts=10, p0=8, p1=6, p2=4, gen=100))
    assert counter(vllm, VERIFY_SERIES) == ("vllm:spec_decode_num_drafts_total", 10.0)
    sglang = prometheus(SGLANG_PAGE % dict(calls=5, gen=40))
    assert counter(sglang, VERIFY_SERIES) == ("sglang:spec_verify_calls_total", 10.0)
    assert counter(sglang, GENERATION_SERIES) == ("sglang:generation_tokens_total", 40.0)
    assert counter(prometheus("other_total 1\n"), VERIFY_SERIES) is None


def test_engine_delta_reads_sglangs_verify_steps_and_generation_tokens() -> None:
    a, b = (prometheus(SGLANG_PAGE % dict(calls=c, gen=g)) for c, g in ((5, 40), (20, 340)))
    out = engine_delta(a, b, completed=10)
    assert out["verify_steps"] == 30 and out["generation_tokens"] == 300 and "accept" not in out
    plain = engine_delta(prometheus("x_total 1\n"), prometheus("x_total 2\n"), completed=1)
    assert plain["verify_steps"] is None and plain["generation_tokens"] is None


def test_compare_prints_the_hit_rate_column_and_the_cache_by_turn_line(tmp_path, capsys) -> None:
    _arm(tmp_path, "A", [f"{i:016x}" for i in range(3)], None)
    path = tmp_path / "A" / "catalog.json"
    report = json.loads(path.read_text())
    report["loads"][0].update(cache_hit_rate=0.5, cache_by_turn=[
        {"turn": 0, "requests": 4, "hit_rate": 0.0}, {"turn": 1, "requests": 3, "hit_rate": 0.8125}])
    path.write_text(json.dumps(report))
    _arm(tmp_path, "B", [f"{i:016x}" for i in range(3)], None)
    assert compare.main([str(tmp_path / "A"), str(tmp_path / "B")]) == 0
    out = capsys.readouterr().out
    assert "hit rate" in out and "  A by turn: 0 0.0% (4), 1 81.2% (3)" in out and "B by turn" not in out
    assert [line.split()[-1] for line in out.splitlines() if line.startswith(("A ", "B "))][:2] == ["50.0%", "-"]
