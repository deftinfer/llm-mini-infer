"""The load generator's cache columns and session mode (bench/load/run.py), on hand-made records and a fake `send`:
no server, no network, no tokenizer download."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import sys
import time
from types import SimpleNamespace

import pytest

from ..support.paths import ROOT


def _load():
    name = "bench_load_run"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "bench/load/run.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


load = _load()


class WordTokenizer:
    """A stand-in tokenizer: one token per whitespace-separated word."""

    def __init__(self) -> None:
        self.vocab: list[str] = []
        self.index: dict[str, int] = {}

    def encode(self, text: str, add_special_tokens: bool = True) -> SimpleNamespace:
        ids = []
        for w in text.split():
            if w not in self.index:
                self.index[w] = len(self.vocab)
                self.vocab.append(w)
            ids.append(self.index[w])
        return SimpleNamespace(ids=ids)

    def decode(self, ids: list[int]) -> str:
        return " ".join(self.vocab[i] for i in ids)


def ok(prompt: int, cached: int | None, turn: int = 0, **extra) -> dict:
    return {"measured": True, "status": 200, "e2e_ms": 10.0, "tokens": 4, "prompt_tokens": prompt,
            "cached_tokens": cached, "finish": "stop", "turn": turn, **extra}


def summary(recs: list[dict]) -> dict:
    return load.summarize(recs, [], 0.0, 1.0, 1.0)


def test_the_usage_field_is_none_unless_the_server_reports_it() -> None:
    cached = load.usage_cached
    assert cached({"prompt_tokens_details": {"cached_tokens": 12}}) == 12
    assert cached({"prompt_tokens_details": {"cached_tokens": 0}}) == 0
    assert cached({"prompt_tokens_details": {"cached_tokens": None}}) is None
    assert cached({"prompt_tokens_details": {}}) is None
    assert cached({"prompt_tokens_details": None}) is None
    assert cached({"prompt_tokens": 5}) is None and cached({}) is None and cached(None) is None


class _Response:
    def __init__(self, payload: dict | None = None, lines: tuple[str, ...] = ()):
        self.status, self.payload, self.content = 200, payload, self._lines(lines)

    @staticmethod
    async def _lines(lines):
        for line in lines:
            yield line.encode()

    async def json(self) -> dict:
        return self.payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc) -> bool:
        return False


class _Session:
    def __init__(self, resp: _Response):
        self.resp = resp

    def post(self, url: str, **kwargs) -> _Response:
        return self.resp


def test_send_keeps_the_cached_tokens_and_the_reply_streamed_or_not() -> None:
    """The cached tokens come from the non-streamed usage and from the streamed final usage chunk alike, None without
    the field; the reply is the content, accumulated from the deltas when streamed."""
    usage = {"completion_tokens": 3, "prompt_tokens": 40, "prompt_tokens_details": {"cached_tokens": 32}}
    chunks = [{"choices": [{"delta": {"content": c}}]} for c in ("Hel", "lo")]
    chunks += [{"choices": [{"delta": {}, "finish_reason": "stop"}]}, {"choices": [], "usage": usage}]
    lines = (*(f"data: {json.dumps(c)}\n" for c in chunks), "data: [DONE]\n")
    streamed = asyncio.run(load.send(_Session(_Response(lines=lines)), "u", {}, {"stream": True}, 5.0))
    plain = asyncio.run(load.send(_Session(_Response({
        "choices": [{"message": {"content": "Hello"}, "finish_reason": "stop"}], "usage": usage})), "u", {}, {}, 5.0))
    bare = asyncio.run(load.send(_Session(_Response({
        "choices": [{"message": {"content": "Hello"}, "finish_reason": "stop"}],
        "usage": {"completion_tokens": 3, "prompt_tokens": 40}})), "u", {}, {}, 5.0))
    assert streamed["cached_tokens"] == plain["cached_tokens"] == 32 and streamed["prompt_tokens"] == 40
    assert streamed["reply"] == plain["reply"] == "Hello" and streamed["digest"] == plain["digest"]
    assert bare["cached_tokens"] is None


def test_summary_gives_the_hit_rate_and_the_uncached_tokens() -> None:
    s = summary([ok(100, 0), ok(100, 50), ok(200, 150), ok(100, 100)])
    assert (s["cache_reported"], s["cache_hit_rate"]) == (4, 0.6)  # (0 + 50 + 150 + 100) / 500
    assert s["uncached_tokens"]["n"] == 4 and s["uncached_tokens"]["p50"] == 50.0
    assert s["uncached_tokens"]["p90"] == 100.0 and s["uncached_tokens"]["max"] == 100.0
    assert s["cache_by_turn"] is None  # single requests


def test_summary_marks_a_server_without_the_field_absent_not_zero() -> None:
    s = summary([ok(100, None), ok(80, None)])
    assert s["cache_hit_rate"] is None and s["uncached_tokens"] is None and s["cache_reported"] == 0
    assert s["cache_by_turn"] is None
    empty = summary([])
    assert empty["cache_hit_rate"] is None and empty["uncached_tokens"] is None
    row = load.table([{"load": "x", "measured": 2, "ok": 2, "dropped": 0, "failed": 0, "served_rps": 1.0,
                       "output_tokens_per_s": None, "e2e_ms": {}, "ttft_ms": {}, "tpot_ms": {}, "reply_tokens": {},
                       **s}]).splitlines()
    assert "hit rate" in row[0] and "uncached p50" in row[0] and "uncached p90" in row[0]
    assert row[2].split()[-3:] == ["-", "-", "-"]


def test_summary_computes_over_the_requests_that_report() -> None:
    s = summary([ok(100, 80), ok(300, None), ok(100, 20), ok(50, None, status=500, error="500")])
    assert s["cache_reported"] == 2 and s["cache_hit_rate"] == 0.5  # (80 + 20) / 200
    assert s["uncached_tokens"]["n"] == 2


def test_summary_gives_the_hit_rate_by_turn_of_a_session() -> None:
    recs = [ok(100, 0, 0), ok(100, 10, 0), ok(200, 100, 1), ok(220, 120, 1), ok(300, 280, 2), ok(300, None, 2)]
    s = summary(recs)
    assert s["cache_by_turn"] == [
        {"turn": 0, "requests": 2, "hit_rate": 0.05, "uncached_p50": 90.0},
        {"turn": 1, "requests": 2, "hit_rate": round(220 / 420, 4), "uncached_p50": 100.0},
        {"turn": 2, "requests": 1, "hit_rate": round(280 / 300, 4), "uncached_p50": 20.0}]


def test_a_session_builds_each_turn_from_the_history_and_the_reply() -> None:
    tok = WordTokenizer()
    chat = load.chat_workload(tok, 6, 16, seed=3, tool_tokens=5)
    first = chat.body(4)["messages"]
    assert len(first) == 1 and first[0]["role"] == "user"
    run = load.Run(None, "", {}, chat, {"model": "m"}, 1.0, 0.0, 1.0, 0.0, offset=10, turns=3)
    body1 = run.body(2)
    second = load.session_body(run, body1, "reply one", 2, 1)
    assert second["model"] == "m" and second["max_tokens"] == 16
    m = second["messages"]
    assert m[:1] == body1["messages"] and m[1] == {"role": "assistant", "content": "reply one"}
    assert m[2]["role"] == "user" and len(tok.encode(m[2]["content"]).ids) == 5 and len(m) == 3
    third = load.session_body(run, second, "reply two", 2, 2)["messages"]
    assert third[:3] == m and third[3] == {"role": "assistant", "content": "reply two"} and len(third) == 5
    assert len(tok.encode(third[4]["content"]).ids) == 5 and third[4]["content"] != m[2]["content"]
    again = load.session_body(run, second, "reply two", 2, 2)["messages"]
    assert again == third  # seeded by the seed, the conversation and the turn
    assert body1["messages"] == run.body(2)["messages"] and len(body1["messages"]) == 1  # the input is not mutated


def test_a_closed_loop_client_holds_a_conversation_for_n_turns_then_starts_another(monkeypatch) -> None:
    tok = WordTokenizer()
    chat = load.chat_workload(tok, 4, 8, seed=0, tool_tokens=3)
    sent: list[dict] = []

    async def send(session, url, headers, body, timeout):
        sent.append(body)
        await asyncio.sleep(0.01)
        n = len(sent)
        return {"status": 200, "e2e_ms": 1.0, "finish": "stop", "digest": f"d{n}", "reply": f"r{n}", "tokens": 2,
                "prompt_tokens": 10 * len(body["messages"]), "cached_tokens": 5 * len(body["messages"])}

    async def scrape(run):
        return 0.0, None

    monkeypatch.setattr(load, "send", send)
    monkeypatch.setattr(load, "scrape", scrape)
    run = load.Run(None, "", {}, chat, {}, 5.0, 0.0, 0.3, 0.3, turns=3, metrics_url=None)

    r = asyncio.run(load.closed_loop(run, 1))
    assert len(sent) >= 4
    assert [len(b["messages"]) for b in sent[:7]] == [1, 3, 5, 1, 3, 5, 1][:len(sent[:7])]
    assert sent[1]["messages"][1] == {"role": "assistant", "content": "r1"}
    assert sent[4]["messages"][1] == {"role": "assistant", "content": "r4"}
    assert sent[3]["messages"] != sent[0]["messages"]  # a new conversation, a new first prompt
    assert r["cache_by_turn"] is not None and [t["turn"] for t in r["cache_by_turn"]] == [0, 1, 2]
    assert '"reply"' not in json.dumps(r) and "r1" not in json.dumps(r["answers"])
    assert any(k.endswith(".1") for k in r["answers"]) and "0" in r["answers"]


def test_shared_prefix_tokens_zero_leaves_the_bodies_unchanged() -> None:
    plain = load.chat_workload(WordTokenizer(), 6, 16, seed=3, tool_tokens=5)
    zero = load.chat_workload(WordTokenizer(), 6, 16, seed=3, tool_tokens=5, shared_tokens=0)
    assert all(plain.body(i) == zero.body(i) for i in range(5))
    assert "shared_prefix_tokens" not in zero.summary


def test_shared_prefix_tokens_start_every_prompt_alike() -> None:
    tok = WordTokenizer()
    chat = load.chat_workload(tok, 6, 16, seed=3, shared_tokens=20)
    a, b = (chat.body(i)["messages"][0]["content"] for i in (1, 7))
    ids_a, ids_b = tok.encode(a).ids, tok.encode(b).ids
    assert ids_a[:20] == ids_b[:20] and ids_a[20:] != ids_b[20:] and len(ids_a) == 26
    assert a.startswith(load.shared_prefix(tok, 20, 3)) and a.split("\n\n", 1)[1].startswith("Request 1.")
    other = load.chat_workload(tok, 6, 16, seed=4, shared_tokens=20).body(1)["messages"][0]["content"]
    assert tok.encode(other).ids[:20] != ids_a[:20]  # the seed picks the prefix
    assert chat.summary["shared_prefix_tokens"] == 20


def test_the_warm_up_caches_the_measured_runs_shared_prefix() -> None:
    tok = WordTokenizer()
    args = argparse.Namespace(seed=3, other=1)
    warm = load.warm_up_args(args)
    assert (warm.seed, warm.prefix_seed, warm.other) == (4, 3, 1) and args.seed == 3
    measured = load.chat_workload(tok, 12, 16, seed=3, shared_tokens=20).body(0)["messages"][0]["content"]
    warmed = load.chat_workload(tok, 12, 16, seed=warm.seed, shared_tokens=20, prefix_seed=warm.prefix_seed)
    content = warmed.body(0)["messages"][0]["content"]
    assert tok.encode(content).ids[:20] == tok.encode(measured).ids[:20] and content != measured


@pytest.mark.parametrize("argv", [["--workload", "catalog"], ["--workload", "scene"]])
def test_shared_prefix_tokens_is_refused_outside_chat(argv, capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        load.main(["--model", "m", "--concurrency", "1", "--shared-prefix-tokens", "8", *argv])
    assert exc.value.code == 2 and "--shared-prefix-tokens needs --workload chat" in capsys.readouterr().err


@pytest.mark.parametrize(("argv", "why"), [
    (["--rates", "2", "--workload", "chat", "--turns", "3"], "--turns needs --concurrency"),
    (["--concurrency", "2", "--workload", "catalog", "--turns", "3"], "--turns needs --concurrency"),
    (["--concurrency", "2", "--workload", "chat", "--turns", "0"], "--turns 0: at least 1"),
    (["--concurrency", "2", "--workload", "chat", "--turns", "2", "--tool-tokens", "0"], "--tool-tokens 0: at least 1"),
])
def test_turns_is_refused_outside_a_closed_loop_chat(argv, why, capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        load.main(["--model", "m", *argv])
    assert exc.value.code == 2 and why in capsys.readouterr().err


def test_compare_holds_only_the_first_turns_answers_of_a_session_load(tmp_path) -> None:
    """A session load's later turns follow each engine's own replies: compare.py reads the first turns' answers only."""
    spec = importlib.util.spec_from_file_location("bench_load_compare", ROOT / "bench/load/compare.py")
    compare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compare)
    answers = {"0": ["a", 3, "stop"], "0.1": ["b", 3, "stop"], "1": ["c", 3, "stop"], "1.2": ["d", 3, "stop"]}
    report = {"workload": {"name": "chat"}, "loads": [{"load": "c2", "answers": answers}]}
    (tmp_path / "r.json").write_text(json.dumps(report))
    _, sections = compare.read_arm(str(tmp_path / "r.json"))
    held = sections[("chat", "c2")]["answers"]
    assert sorted(held) == ["0", "1"]
    other = {"0": ("a", 3, "stop"), "0.1": ("x", 3, "stop"), "1": ("c", 3, "stop")}
    assert "100.0%" in "\n".join(compare.agreement({"A": held, "B": other}, []))


def _fake_send(sent: list[dict], fail_from: int | None = None):
    async def send(session, url, headers, body, timeout):
        sent.append(body)
        await asyncio.sleep(0.005)
        if fail_from is not None and len(sent) > fail_from:
            return {"status": 500, "error": "500", "e2e_ms": 1.0}
        return {"status": 200, "e2e_ms": 1.0, "finish": "stop", "digest": f"d{len(sent)}", "reply": "r", "tokens": 2,
                "prompt_tokens": 10, "cached_tokens": None}

    return send


async def _no_scrape(run):
    return 0.0, None


def _indexed(sent: list[dict]) -> list[int]:
    return [int(b["messages"][0]["content"]) for b in sent]


def _count_workload(seed: int = 0) -> object:
    return load.Workload("scene", lambda i: {"messages": [{"role": "user", "content": str(i)}]}, None,
                         {"mode": "fake", "seed": seed})


def test_a_count_bounded_load_sends_exactly_n_requests_at_the_next_indexes(monkeypatch) -> None:
    sent: list[dict] = []
    monkeypatch.setattr(load, "send", _fake_send(sent))
    monkeypatch.setattr(load, "scrape", _no_scrape)
    seen = []
    for _ in range(2):
        sent.clear()
        run = load.Run(None, "", {}, _count_workload(), {}, 5.0, 10.0, 60.0, 60.0, metrics_url=None)
        r = asyncio.run(load.closed_loop(run, 3, requests=7))
        seen.append(sorted(_indexed(sent)))
        assert r["load"] == "concurrency 3, 7 requests" and r["requests"] == 7
        assert (r["measured"], r["ok"], r["failed"], r["dropped"]) == (7, 7, 0, 0)
        assert sorted(int(k) for k in r["answers"]) == list(range(7)) and run.offset == 7
        assert r["seconds"] > 0 and abs(r["offered_rps"] * r["seconds"] - 7) < 0.5  # rps is rounded to 3 decimals
    assert seen[0] == seen[1] == list(range(7))


def test_the_warmup_requests_come_from_another_seed_and_leave_the_measured_indexes(monkeypatch, tmp_path) -> None:
    sent: list[dict] = []
    built: list[int] = []

    def make(args):
        built.append(args.seed)
        return _count_workload(args.seed)

    monkeypatch.setattr(load, "send", _fake_send(sent))
    monkeypatch.setattr(load, "scrape", _no_scrape)
    monkeypatch.setattr(load, "make_workload", make)
    argv = ["--model", "m", "--concurrency", "2", "--requests", "5", "--seed", "3", "--out", str(tmp_path / "r.json")]
    assert load.main(argv) == 0
    report = json.loads((tmp_path / "r.json").read_text())
    assert built == [3, 4] and report["warmup_requests"] == 20 and report["requests"] == 5
    idx = _indexed(sent)
    assert len(idx) == 25 and sorted(idx[:20]) == [load.WARMUP_OFFSET + j for j in range(20)]
    assert sorted(idx[20:]) == list(range(5))
    assert sorted(int(k) for k in report["loads"][0]["answers"]) == list(range(5))  # none of the warm-up's


def test_a_failed_warmup_request_ends_the_run_before_any_load(monkeypatch, tmp_path) -> None:
    sent: list[dict] = []
    monkeypatch.setattr(load, "send", _fake_send(sent, fail_from=2))
    monkeypatch.setattr(load, "scrape", _no_scrape)
    monkeypatch.setattr(load, "make_workload", lambda args: _count_workload())
    argv = ["--model", "m", "--concurrency", "1", "--requests", "5", "--warmup-requests", "4",
            "--out", str(tmp_path / "r.json")]
    with pytest.raises(SystemExit, match=r"warm-up request [0-9] failed: 500"):
        load.main(argv)
    assert len(sent) >= 3 and not (tmp_path / "r.json").exists()


def test_a_folder_workload_takes_no_warmup(monkeypatch) -> None:
    folder = load.Workload("folder", lambda i: {}, 4, {})
    monkeypatch.setattr(load, "make_workload", lambda args: folder)
    args = SimpleNamespace(warmup_requests=3)
    with pytest.raises(SystemExit, match="images are finite"):
        asyncio.run(load.main_async(args))


@pytest.mark.parametrize(("argv", "why"), [
    (["--rates", "2", "--requests", "5"], "--requests needs --concurrency"),
    (["--concurrency", "2", "--workload", "chat", "--turns", "2", "--requests", "5"], "--requests needs --concurrency"),
    (["--concurrency", "2", "--requests", "0"], "--requests 0: at least 1"),
    (["--concurrency", "2", "--warmup-requests", "-1"], "--warmup-requests -1: at least 0"),
])
def test_requests_is_refused_outside_a_count_bounded_closed_loop(argv, why, capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        load.main(["--model", "m", *argv])
    assert exc.value.code == 2 and why in capsys.readouterr().err


def test_the_default_warmup_is_20_with_requests_and_0_without(monkeypatch, tmp_path) -> None:
    got = []

    async def fake_main(args):
        got.append((args.requests, args.warmup_requests))
        return {"loads": []}

    monkeypatch.setattr(load, "main_async", fake_main)
    out = ["--out", str(tmp_path / "r.json")]
    load.main(["--model", "m", "--concurrency", "2", "--requests", "5", *out])
    load.main(["--model", "m", "--concurrency", "2", *out])
    load.main(["--model", "m", "--concurrency", "2", "--requests", "5", "--warmup-requests", "0", *out])
    assert got == [(5, 20), (None, 0), (5, 0)]


def _load_entry(**over) -> dict:
    stat = {"n": 5, "p50": 900.0, "p90": 1200.0, "p99": 1400.0, "mean": 950.0, "max": 1500.0}
    return {"measured": 10, "ok": 10, "failed": 0, "dropped": 0, "offered_rps": 2.0, "served_rps": 2.0,
            "output_tokens_per_s": None, "reply_tokens": stat, "e2e_ms": stat, "ttft_ms": stat, "tpot_ms": stat, **over}


def test_passes_gives_the_verdict_and_the_reason() -> None:
    passes = load.passes
    assert passes(_load_entry(), "e2e_ms.p99", 1500) == (True, "ok")
    assert passes(_load_entry(failed=1), "e2e_ms.p99", 1500) == (False, "1 failed")
    assert passes(_load_entry(dropped=2), "e2e_ms.p99", 1500) == (False, "2 dropped")
    assert passes(_load_entry(served_rps=1.8), "e2e_ms.p99", 1500)[0] is False
    assert passes(_load_entry(served_rps=1.9), "e2e_ms.p99", 1500)[0] is True  # exactly 95%
    ok, why = passes(_load_entry(e2e_ms={"n": 5, "p99": 1830.4}), "e2e_ms.p99", 1500)
    assert (ok, why) == (False, "e2e_ms.p99 1830 > 1500")
    assert passes(_load_entry(ok=0, ttft_ms={"n": 0}), "ttft_ms.p50", 100) == (False, "no successful request")
    assert passes(_load_entry(), "tpot_ms.mean", 1000)[0] is True


def _threshold_probe(threshold: float, seen: list[float]):
    def probe(rate: float):
        seen.append(rate)
        return rate <= threshold, {"rate": rate}

    return probe


@pytest.mark.parametrize("threshold", [0.5, 1.0, 3.3, 7.77, 9.9, 11.0, 16.0])
def test_find_rate_finds_a_monotone_threshold_within_the_resolution(threshold) -> None:
    seen: list[float] = []
    lo, hi, res = 1.0, 16.0, 0.5
    r = load.find_rate(_threshold_probe(threshold, seen), lo, hi, res, 0)
    assert len(seen) <= load.MAX_PROBES and all(lo <= x <= hi for x in seen)
    assert [q["rate"] for q in r["probes"]] == seen and r["stopped"] is None
    if threshold < lo:
        assert r["best_rate"] is None and r["first_failing_rate"] == lo and seen == [lo]
    elif threshold >= hi:
        assert r["best_rate"] == hi and r["first_failing_rate"] is None
    else:
        assert r["best_rate"] <= threshold < r["first_failing_rate"]
        assert r["first_failing_rate"] - r["best_rate"] <= res


def test_find_rate_doubles_from_lo_and_stops_at_hi() -> None:
    seen: list[float] = []
    r = load.find_rate(_threshold_probe(100.0, seen), 1.5, 10.0, 1.0, 0)
    assert seen == [1.5, 3.0, 6.0, 10.0] and r["best_rate"] == 10.0 and r["first_failing_rate"] is None


def test_find_rate_confirmation_moves_a_noisy_best_rate_down() -> None:
    calls: dict[float, int] = {}

    def probe(rate: float):
        calls[rate] = calls.get(rate, 0) + 1
        flaky = rate == 6.0 and calls[rate] > 1  # passes once, then fails on every repeat
        return rate <= 6.5 and not flaky, {}

    r = load.find_rate(probe, 1.5, 12.0, 1.0, 2)
    assert r["best_rate"] is not None and r["best_rate"] < 6.0 and r["confirmed"] is True
    reps = [q for q in r["probes"] if q["confirmation"]]
    assert reps[0]["rate"] == 6.0 and reps[0]["passed"] is False and reps[0]["confirmation"] == 1
    assert [q["confirmation"] for q in r["probes"] if q["rate"] == r["best_rate"]][-2:] == [1, 2]


def test_find_rate_confirmation_can_leave_no_rate_and_stops_at_the_probe_limit() -> None:
    calls: dict[float, int] = {}

    def once(rate: float):
        calls[rate] = calls.get(rate, 0) + 1
        return calls[rate] == 1 and rate == 2.0, {}

    r = load.find_rate(once, 2.0, 8.0, 1.0, 1)
    assert r["best_rate"] is None and r["first_failing_rate"] == 2.0 and r["confirmed"] is False
    seen: list[float] = []
    r = load.find_rate(_threshold_probe(5.0, seen), 1.0, 1000.0, 1e-9, 0)
    assert len(seen) == load.MAX_PROBES and r["stopped"] == "probe limit" and r["confirmed"] is False


def test_a_probe_depends_on_its_rate_only() -> None:
    assert load.probe_plan(5, 2.5) == (5 + 2500, 2500 * 100_000)
    assert load.probe_plan(5, 2.5) == load.probe_plan(5, round(2.5, 4))


def _bound(tmp_path, **over):
    bound = {"model": "m", "workload": "scene", "metric": "e2e_ms.p99", "value": 1500, "note": "kept", **over}
    path = tmp_path / "bound.json"
    path.write_text(json.dumps(bound))
    return path


def _search_args(tmp_path, bound, *extra):
    return ["--model", "m", "--find-rate", "1,8", "--bound-file", str(bound), "--duration", "1", "--warmup", "0",
            "--tail", "0", "--out", str(tmp_path / "r.json"), *extra]


def test_a_search_builds_the_same_arrivals_for_a_rate_in_any_order(monkeypatch, tmp_path) -> None:
    plans: list[dict] = []

    def fake(threshold):
        async def open_loop(run, rate, slots, seed):
            plans.append({"rate": rate, "seed": seed, "offset": run.offset, "slots": slots})
            ok_ = rate <= threshold
            return {"load": f"rate {rate:g}/s, {slots} slots", **_load_entry(failed=0 if ok_ else 1)}

        return open_loop

    monkeypatch.setattr(load, "make_workload", lambda args: _count_workload())
    monkeypatch.setattr(load, "send", _fake_send([]))
    monkeypatch.setattr(load, "scrape", _no_scrape)
    runs = []
    for threshold in (3.0, 5.0):
        plans.clear()
        monkeypatch.setattr(load, "open_loop", fake(threshold))
        argv = _search_args(tmp_path, _bound(tmp_path), "--seed", "7", "--warmup-requests", "0")
        assert load.main(argv) == 1  # a failing probe at the top of the search makes the exit code 1
        runs.append({p["rate"]: p for p in plans})
        report = json.loads((tmp_path / "r.json").read_text())
        s = report["search"]
        assert s["bound"]["note"] == "kept" and s["low"] == 1 and s["high"] == 8 and s["resolution"] == 7 / 8
        assert s["best_rate"] <= threshold < s["first_failing_rate"] and s["stopped"] is None
        assert [p["rate"] for p in s["probes"]] == [p["rate"] for p in plans]
        assert [ld["probe"] for ld in report["loads"]] == s["probes"] and plans[0]["slots"] == 128
        assert s["probes"][-1]["passed"] is not (s["probes"][-1]["reason"] != "ok")
    for rate in runs[0].keys() & runs[1].keys():
        assert runs[0][rate] == runs[1][rate] == {"rate": rate, "slots": 128, **dict(zip(
            ("seed", "offset"), load.probe_plan(7, rate), strict=True))}
    assert runs[0].keys() != runs[1].keys()  # the searches visited the rates in different orders


def test_a_confirmation_probe_is_named_and_keeps_its_arrivals_with_new_requests(monkeypatch, tmp_path) -> None:
    plans: list[tuple] = []

    async def open_loop(run, rate, slots, seed):
        plans.append((rate, seed, run.offset))
        return {"load": f"rate {rate:g}/s, {slots} slots", **_load_entry()}

    monkeypatch.setattr(load, "make_workload", lambda args: _count_workload())
    monkeypatch.setattr(load, "open_loop", open_loop)
    argv = _search_args(tmp_path, _bound(tmp_path), "--confirm", "2", "--warmup-requests", "0")
    assert load.main(argv) == 0
    report = json.loads((tmp_path / "r.json").read_text())
    assert [ld["load"] for ld in report["loads"]][-2:] == ["rate 8/s, 128 slots (confirmation 1)",
                                                         "rate 8/s, 128 slots (confirmation 2)"]
    assert report["search"]["confirmed"] is True
    (rate, seed, first), (_, seed1, second), (_, seed2, third) = plans[-3:]
    assert rate == 8 and seed == seed1 == seed2 == load.probe_plan(0, 8)[0]
    assert (second, third) == (first + load.REPEAT_STRIDE, first + 2 * load.REPEAT_STRIDE)
    assert first == load.probe_plan(0, 8)[1] and load.REPEAT_STRIDE == 10_000


def test_a_search_refuses_a_hi_whose_window_could_exceed_the_repeat_stride(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(load, "make_workload", lambda args: _count_workload())
    argv = _search_args(tmp_path, _bound(tmp_path), "--warmup-requests", "0")
    argv[argv.index("1,8")] = "1,200"
    argv[argv.index("--duration") + 1] = "60"
    with pytest.raises(SystemExit, match="more than 10000 requests"):
        load.main(argv)


def test_first_index_starts_the_measured_indexes_and_is_recorded(monkeypatch, tmp_path) -> None:
    sent: list[dict] = []
    monkeypatch.setattr(load, "make_workload", lambda args: _count_workload())
    monkeypatch.setattr(load, "send", _fake_send(sent))
    monkeypatch.setattr(load, "scrape", _no_scrape)
    argv = ["--model", "m", "--concurrency", "1", "--requests", "3", "--warmup-requests", "0", "--first-index", "5",
            "--out", str(tmp_path / "r.json")]
    assert load.main(argv) == 0
    report = json.loads((tmp_path / "r.json").read_text())
    assert sorted(int(k) for k in report["loads"][0]["answers"]) == [5, 6, 7] and report["first_index"] == 5
    assert _indexed(sent) == [5, 6, 7]


@pytest.mark.parametrize(("extra", "why"), [
    (["--find-rate", "1,4", "--bound-file", "b"], "--first-index does not go with --find-rate"),
    (["--concurrency", "1", "--first-index", "-1"], "at least 0"),
])
def test_first_index_is_refused_with_find_rate_or_below_zero(extra, why, capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        load.main(["--model", "m", "--first-index", "2", *extra] if "-1" not in extra else ["--model", "m", *extra])
    assert exc.value.code == 2 and why in capsys.readouterr().err


def test_the_default_warmup_of_a_search_is_20(monkeypatch, tmp_path) -> None:
    got = []

    async def fake_main(args):
        got.append(args.warmup_requests)
        return {"loads": []}

    monkeypatch.setattr(load, "main_async", fake_main)
    load.main(_search_args(tmp_path, _bound(tmp_path)))
    assert got == [20]


@pytest.mark.parametrize(("argv", "why"), [
    (["--rates", "2", "--find-rate", "1,4"], "not allowed with argument"),
    (["--concurrency", "2", "--find-rate", "1,4"], "not allowed with argument"),
    (["--find-rate", "1,4"], "--find-rate needs --bound-file"),
    (["--find-rate", "4,1", "--bound-file", "b"], "0 < LO <= HI"),
    (["--find-rate", "1,4", "--bound-file", "b", "--slots", "1,2"], "one --slots value"),
    (["--find-rate", "1,4", "--bound-file", "b", "--requests", "5"], "--requests needs --concurrency"),
    (["--concurrency", "2", "--confirm", "1"], "need --find-rate"),
])
def test_find_rate_is_refused_with_another_load_or_without_its_bound(argv, why, capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        load.main(["--model", "m", *argv])
    assert exc.value.code == 2 and why in capsys.readouterr().err


@pytest.mark.parametrize(("over", "why"), [
    ({"model": "other"}, "made for model 'other'"),
    ({"workload": "chat"}, "made for workload 'chat'"),
    ({"metric": "e2e_ms.p95"}, "metric 'e2e_ms.p95'"),
    ({"metric": "queue_ms.p99"}, "metric 'queue_ms.p99'"),
    ({"value": "fast"}, "value is not a number"),
])
def test_a_bound_file_for_another_model_workload_or_metric_is_refused(monkeypatch, tmp_path, over, why) -> None:
    monkeypatch.setattr(load, "make_workload", lambda args: _count_workload())
    with pytest.raises(SystemExit, match=why):
        load.main(_search_args(tmp_path, _bound(tmp_path, **over)))


def test_the_search_line_says_the_best_rate_or_none() -> None:
    base = {"low": 2.0, "best_rate": 6.5, "confirmed": True, "stopped": None}
    assert load.search_line(base) == "best rate within the bound: 6.5/s (confirmed)"
    assert load.search_line({**base, "best_rate": None}) == "best rate within the bound: none at or above 2/s"


def _pages(drafts_a: float, drafts_b: float, name: str = "vllm:spec_decode_num_drafts_total"):
    def page(n: float) -> dict[str, float]:
        return load.prometheus(f'{name}{{engine="0"}} {n}\n' if name else "other_total 1\n")

    return page(drafts_a), page(drafts_b)


def test_a_loads_spec_block_is_the_summed_tokens_over_the_verify_steps(monkeypatch) -> None:
    sent: list[dict] = []
    send = _fake_send(sent)
    a, b = _pages(100, 106)
    calls = []

    async def scrape(run):
        calls.append(1)
        return 0.0, a if len(calls) == 1 else b

    async def counted_send(session, url, headers, body, timeout):
        rec = await send(session, url, headers, body, timeout)
        index = int(body["messages"][0]["content"])
        if index == 1:  # no completion-token count
            rec["tokens"] = None
        elif index == 2:
            rec["tokens"] = 7
        return rec

    monkeypatch.setattr(load, "send", counted_send)
    monkeypatch.setattr(load, "scrape", scrape)
    monkeypatch.setattr(load, "SETTLE_S", 0.0)
    run = load.Run(None, "", {}, _count_workload(), {}, 5.0, 0.0, 1.0, 0.0, metrics_url="http://x/metrics")
    r = asyncio.run(load.closed_loop(run, 2, 5))
    assert r["spec"] == {"series": "vllm:spec_decode_num_drafts_total", "verify_steps": 6, "requests": 5,
                         "completion_tokens_minus_one": 3 * 1 + 6, "tokens_per_verify": 1.5, "no_usage": 1}
    assert len(calls) == 4  # load start, window start, window end, load end


def test_a_loads_spec_is_null_without_metrics_a_verify_series_or_a_step(monkeypatch) -> None:
    monkeypatch.setattr(load, "send", _fake_send([]))
    monkeypatch.setattr(load, "SETTLE_S", 0.0)
    recs = [{"status": 200, "tokens": 3}]
    a, b = _pages(5, 9)
    assert load.spec_block(a, b, recs)["tokens_per_verify"] == 0.5
    assert load.spec_block(*_pages(1, 1), recs) is None  # no step
    assert load.spec_block(*_pages(0, 0, ""), recs) is None  # no verify series
    assert load.spec_block(None, b, recs) is None and load.spec_block(a, None, recs) is None
    monkeypatch.setattr(load, "scrape", _no_scrape)
    run = load.Run(None, "", {}, _count_workload(), {}, 5.0, 0.0, 1.0, 0.0, metrics_url=None)
    assert asyncio.run(load.closed_loop(run, 1, 2))["spec"] is None
    sglang = load.prometheus('sglang:spec_verify_calls_total{tp_rank="0"} 4\n'
                             'sglang:spec_verify_calls_total{tp_rank="1"} 4\n')
    assert load.spec_block({}, sglang, recs)["verify_steps"] == 8


def test_meta_and_the_gpu_entry_land_in_the_report(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(load, "send", _fake_send([]))
    monkeypatch.setattr(load, "scrape", _no_scrape)
    monkeypatch.setattr(load, "make_workload", lambda args: _count_workload())
    argv = ["--model", "m", "--concurrency", "1", "--requests", "2", "--warmup-requests", "0", "--no-metrics",
            "--meta", '{"engine": "e", "draft": 3}', "--out", str(tmp_path / "r.json")]
    gpu = {"gpus": [{"name": "G", "power_limit_w": 300.0, "power_max_w": 600.0, "sm_clock_mhz": 2000.0,
                     "memory_mib": 1000.0, "memory_used_mib": 5.0, "driver": "1.2"}]}
    monkeypatch.setattr(load, "gpu_info", lambda: {"error": "nvidia-smi not found or hung"})
    assert load.main(argv) == 0
    report = json.loads((tmp_path / "r.json").read_text())
    assert report["meta"] == {"engine": "e", "draft": 3} and "gpu" not in report
    monkeypatch.setattr(load, "gpu_info", lambda: gpu)
    assert load.main(argv) == 0
    report = json.loads((tmp_path / "r.json").read_text())
    assert report["gpu"] == {"name": "G", "power_limit_w": 300.0, "power_max_w": 600.0, "sm_clock_mhz": 2000.0,
                             "memory_mib": 1000.0, "driver": "1.2"}
    assert report["loads"][0]["spec"] is None


def test_meta_must_be_an_object(capsys) -> None:
    with pytest.raises(SystemExit):
        load.main(["--model", "m", "--concurrency", "1", "--meta", "[1]"])
    assert "--meta" in capsys.readouterr().err


def _think_run(monkeypatch, sleeps: list[float], think: float = 5.0, seed: int = 3):
    chat = load.chat_workload(WordTokenizer(), 4, 8, seed=seed, tool_tokens=3)
    real_sleep = asyncio.sleep
    n = {"sent": 0}

    async def send(session, url, headers, body, timeout):
        n["sent"] += 1
        await real_sleep(0)
        return {"status": 200, "e2e_ms": 1.0, "finish": "stop", "digest": "d", "reply": "r", "tokens": 2}

    async def scrape(run):
        return 0.0, None

    async def sleep(delay, *a):
        if delay > 1.0:  # the pauses; the loop's own waits are shorter
            sleeps.append(delay)
            delay = 0
        await real_sleep(delay)

    monkeypatch.setattr(load, "send", send)
    monkeypatch.setattr(load, "scrape", scrape)
    monkeypatch.setattr(load.asyncio, "sleep", sleep)
    return load.Run(None, "", {}, chat, {}, 5.0, 0.0, 0.3, 0.3, turns=3, metrics_url=None, think=think, seed=seed,
                    think_sigma=0.5)


def test_think_time_is_seeded_by_the_conversation_and_the_turn(monkeypatch) -> None:
    run = _think_run(monkeypatch, [])
    assert load.think_time(run, 1, 0) == load.think_time(run, 1, 0)
    assert load.think_time(run, 1, 0) != load.think_time(run, 1, 1)
    assert load.think_time(run, 1, 0) != load.think_time(run, 2, 0)
    run.offset = 5
    assert load.think_time(run, 1, 0) == load.think_time(run, 1, 0) and load.think_time(run, 1, 0) > 0


def test_a_client_pauses_between_turns_and_not_after_the_last(monkeypatch) -> None:
    sleeps: list[float] = []
    run = _think_run(monkeypatch, sleeps)
    asyncio.run(load.closed_loop(run, 1, requests=6))  # no window: a pause is never capped
    assert sleeps and len(sleeps) >= 2
    run.offset = 0
    first = [load.think_time(run, 0, 0), load.think_time(run, 0, 1)]
    assert sleeps[:2] == first  # turns 0 and 1 pause, turn 2 (the last) does not: the third pause is the next one's
    assert sleeps[2:3] != [load.think_time(run, 0, 2)]


def test_a_pause_ends_at_the_window_and_tail_not_after_the_draw(monkeypatch) -> None:
    real_sleep = asyncio.sleep
    run = _think_run(monkeypatch, [])
    monkeypatch.setattr(load.asyncio, "sleep", real_sleep)
    monkeypatch.setattr(load, "think_time", lambda run, i, turn: 1000.0)
    t0 = time.monotonic()
    asyncio.run(load.closed_loop(run, 1, requests=None))  # duration 0.3 + tail 0.3
    assert time.monotonic() - t0 < 2.0


def test_think_is_refused_without_turns(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        load.main(["--model", "m", "--concurrency", "2", "--workload", "chat", "--think", "2"])
    assert exc.value.code == 2 and "--think needs" in capsys.readouterr().err
