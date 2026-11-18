"""End to end: the real server (launcher, engine and frontend processes over ZMQ) against the fake runner. Every
expected text is predicted from the fake's (seed, position) tokens."""

from __future__ import annotations

import json
import re
import signal
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from llm_mini_infer.api.app import tally_for
from llm_mini_infer.api.engine_client import sampling_temperature
from llm_mini_infer.engine.core import PHASES
from llm_mini_infer.metrics import Metrics
from llm_mini_infer.runner.base import UNION_BUCKETS

from ..support.api_util import Predictor, Server, content_of
from ..support.fixtures import require_qwen

MSG = [{"role": "user", "content": "Hello!"}]
KW = {"enable_thinking": False}
SEED = 1_234_567
EOS_SEED = 4_242_424_242


def chat(**kw) -> dict:
    return {"messages": MSG, "chat_template_kwargs": KW, **kw}  # kw may override messages


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    model = require_qwen()
    pred = Predictor(model)
    prompt = pred.prompt(MSG, **KW)
    srv = Server(model, tmp_path_factory.mktemp("srv"), eos=[(EOS_SEED, len(prompt) + 3)])
    srv.start()
    srv.pred, srv.prompt_ids = pred, prompt
    yield srv
    assert srv.stop() == 0, srv.log_path.read_text()[-2000:]


def test_stream_matches_prediction_one_event_per_step(server) -> None:
    events = server.stream(chat(max_tokens=8, seed=SEED))
    assert events[-1] == "[DONE]"
    chunks = events[:-1]
    toks = server.pred.tokens(server.prompt_ids, SEED, 8)
    text, pieces = server.pred.text(server.prompt_ids, toks)
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}  # OpenAI's role-only first chunk
    assert all("role" not in c["choices"][0]["delta"] for c in chunks[1:])
    assert content_of(events) == text
    got = [c["choices"][0]["delta"]["content"] for c in chunks[1:]]
    assert got == [*[p for p in pieces[:-1] if p], pieces[-1]]  # one event per step with text
    assert chunks[-1]["choices"][0]["finish_reason"] == "length"
    assert all(c["choices"][0]["finish_reason"] is None for c in chunks[:-1])
    assert chunks[0]["model"] == "qwen-test" and chunks[0]["id"].startswith("chatcmpl-")


def test_non_stream_and_usage(server) -> None:
    status, r = server.post("/v1/chat/completions", chat(max_tokens=6, seed=SEED + 1))
    assert status == 200, r
    toks = server.pred.tokens(server.prompt_ids, SEED + 1, 6)
    assert r["choices"][0]["message"] == {"role": "assistant", "content": server.pred.text(server.prompt_ids, toks)[0],
                                          "refusal": None, "annotations": None, "audio": None, "function_call": None,
                                          "reasoning": None}  # vLLM's keys; `tool_calls` only with calls
    assert r["choices"][0]["finish_reason"] == "length"
    assert r["usage"] == {"prompt_tokens": len(server.prompt_ids), "completion_tokens": 6,
                          "total_tokens": len(server.prompt_ids) + 6, "prompt_tokens_details": {"cached_tokens": 0},
                          "completion_tokens_details": {"reasoning_tokens": 0}}  # Qwen's reasoning parser, on
    assert r["object"] == "chat.completion"


def test_stream_include_usage(server) -> None:
    events = server.stream(chat(max_tokens=5, seed=SEED + 2, stream_options={"include_usage": True}))
    assert events[-1] == "[DONE]"
    usage = events[-2]
    assert usage["choices"] == [] and usage["usage"] == {"prompt_tokens": len(server.prompt_ids),
                                                          "total_tokens": len(server.prompt_ids) + 5,
                                                          "completion_tokens": 5,
                                                          "prompt_tokens_details": {"cached_tokens": 0},
                                                          "completion_tokens_details": {"reasoning_tokens": 0}}


def boundary_stops(pred: Predictor, prompt: list[int], n: int = 1) -> list[tuple[int, str, int, str]]:
    """`n` seeds whose output has a stop string spanning two tokens (i and i + 1), occurring first at that boundary:
    (seed, stop, i, the text before it)."""
    out = []
    for seed in range(10_000, 14_000):
        toks = pred.tokens(prompt, seed, 12)
        text, pieces = pred.text(prompt, toks)
        for i in range(2, 10):
            a, b = pieces[i], pieces[i + 1]
            if len(a) >= 2 and len(b) >= 2 and a.isprintable() and b.isprintable():
                stop = a[-2:] + b[:2]
                at = len("".join(pieces[:i + 1])) - 2
                if text.find(stop) == at:
                    out.append((seed, stop, i, text[:at]))
                    if len(out) == n:
                        return out
                    break
    raise AssertionError("no suitable seeds")


def find_boundary_stop(pred: Predictor, prompt: list[int]) -> tuple[int, str, int, str]:
    return boundary_stops(pred, prompt)[0]


@pytest.mark.parametrize("stream", [True, False])
def test_stop_string_split_across_tokens(server, stream) -> None:
    seed, stop, i, expected = find_boundary_stop(server.pred, server.prompt_ids)
    body = chat(max_tokens=12, seed=seed, stop=["@@never@@", stop])
    if stream:
        events = server.stream(body)
        assert content_of(events) == expected
        assert all(stop not in (e["choices"][0]["delta"].get("content", "") if isinstance(e, dict) else "")
                   for e in events)
        final = events[-2]["choices"][0]
    else:
        status, r = server.post("/v1/chat/completions", body)
        assert status == 200, r
        final = r["choices"][0]
        assert final["message"]["content"] == expected
        assert r["usage"]["completion_tokens"] == i + 2  # up to the token that completed the stop string
    assert final["finish_reason"] == "stop" and final["stop_reason"] == stop


def test_a_stop_string_abort_counts_as_a_stop(tmp_path) -> None:
    """A stop string ends a request the engine still runs (steps of 20 ms, so the frontend reads each step's outputs
    long before the next, on any host): the frontend's abort finds it, and counts as a stop string's."""
    model = require_qwen()
    pred = Predictor(model)
    prompt = pred.prompt(MSG, **KW)
    seed, stop, _, expected = find_boundary_stop(pred, prompt)
    srv = Server(model, tmp_path, step_ms=20).start()
    try:
        status, r = srv.post("/v1/chat/completions", chat(max_tokens=2000, seed=seed, stop=[stop]))
        assert status == 200 and r["choices"][0]["message"]["content"] == expected, r
        assert r["choices"][0]["finish_reason"] == "stop"
        _until(lambda: srv.metric("llm_mini_infer:engine_aborts_total", 0.0, reason="stop") == 1,
               "no stop abort counted")
        assert srv.metric("llm_mini_infer:engine_aborts_total") == 1
    finally:
        assert srv.stop() == 0, srv.log_path.read_text()[-2000:]


def test_a_stop_string_and_the_eos_in_one_mtp_step_count_up_to_the_stop(tmp_path) -> None:
    """MTP (every draft accepted: steps of 1 + 3 tokens): a stop string completes, and the EOS right after it comes in
    the same step. The usage counts the tokens up to the one that completed the stop string, not the step's
    tokens."""
    model = require_qwen()
    pred = Predictor(model)
    prompt = pred.prompt(MSG, **KW)
    cases = boundary_stops(pred, prompt, 8)
    eos = [(seed, len(prompt) + i + 2) for seed, _, i, _ in cases]  # the token after the stop string's last one
    srv = Server(model, tmp_path, mtp=3, eos=eos, extra={"dev": {"runner": {"fake": {"accept_p": 1.0}}}}).start()
    try:
        for n, (seed, stop, i, expected) in enumerate(cases):
            body = chat(max_tokens=12, seed=seed, stop=[stop])
            if n % 2:
                events = srv.stream({**body, "stream_options": {"include_usage": True}})
                assert content_of(events) == expected and events[-3]["choices"][0]["finish_reason"] == "stop"
                usage = events[-2]["usage"]
            else:
                status, r = srv.post("/v1/chat/completions", body)
                assert status == 200 and r["choices"][0]["message"]["content"] == expected, r
                assert r["choices"][0]["finish_reason"] == "stop" and r["choices"][0]["stop_reason"] == stop
                usage = r["usage"]
            assert usage["completion_tokens"] == i + 2, (seed, i, usage)
    finally:
        assert srv.stop() == 0, srv.log_path.read_text()[-2000:]


def test_eos_and_ignore_eos(server) -> None:
    plen = len(server.prompt_ids)
    status, r = server.post("/v1/chat/completions", chat(max_tokens=10, seed=EOS_SEED))
    assert status == 200, r
    toks = server.pred.tokens(server.prompt_ids, EOS_SEED, 10, eos_at=plen + 3)
    # the fake's EOS is the model's first stop id, 248044 (<|endoftext|>): not the tokenizer's own, so reported
    assert r["choices"][0]["finish_reason"] == "stop" and r["choices"][0]["stop_reason"] == 248044
    assert r["usage"]["completion_tokens"] == 4  # three tokens and the EOS
    assert r["choices"][0]["message"]["content"] == server.pred.text(server.prompt_ids, toks[:3])[0]
    status, r = server.post("/v1/chat/completions", chat(max_tokens=10, seed=EOS_SEED, ignore_eos=True))
    assert status == 200, r
    assert r["choices"][0]["finish_reason"] == "length" and r["usage"]["completion_tokens"] == 10
    assert r["choices"][0]["message"]["content"] == server.pred.text(server.prompt_ids, toks)[0]


def test_stop_token_ids(server) -> None:
    toks = server.pred.tokens(server.prompt_ids, SEED + 3, 10)
    status, r = server.post("/v1/chat/completions", chat(max_tokens=10, seed=SEED + 3, stop_token_ids=[toks[4]]))
    assert status == 200, r
    assert r["choices"][0]["finish_reason"] == "stop" and r["choices"][0]["stop_reason"] == toks[4]
    assert r["usage"]["completion_tokens"] == 5


BAD = [
    ({"foo": 1}, 400), ({"n": 2}, 400), ({"logprobs": True, "top_logprobs": 21}, 400),
    ({"frequency_penalty": 0.5}, 400), ({"presence_penalty": 1.0}, 400), ({"repetition_penalty": 1.1}, 400),
    ({"top_logprobs": 2}, 400),
    ({"logit_bias": {"1": 2.0}}, 400), ({"tool_choice": "required"}, 400), ({"min_tokens": 5}, 400),
    ({"tool_choice": "auto"}, 400), ({"tools": []}, 400), ({"tools": [], "tool_choice": "none"}, 400),
    ({"tools": [{"type": "function", "function": {"parameters": {}}}]}, 400),
    ({"response_format": {"type": "json_schema"}}, 400), ({"response_format": {"type": "structural_tag"}}, 400),
    ({"response_format": {"type": "json_schema", "json_schema": {"name": "x"}}}, 400),
    ({"response_format": {"type": "json_schema", "json_schema": {"schema": {"type": "object"}}}}, 400),
    ({"response_format": {"type": "json_schema", "json_schema": {"name": "x", "schema": None}}}, 400),
    ({"response_format": {"type": "json_schema", "json_schema": {"name": "x", "schema": [1]}}}, 400),
    ({"response_format": {"type": "json_object", "schema": {}}}, 400),
    ({"response_format": {"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "strin"}}}}, 400),
    ({"guided_json": {"type": "object"}}, 400), ({"structured_outputs": {"json": {"type": "object"}}}, 400),
    ({"temperature": 3.0}, 400), ({"top_p": 0.0}, 400),
    ({"max_tokens": 0}, 400), ({"stream_options": {"include_usage": True}}, 400),
    ({"stream": True, "stream_options": {"continuous_usage_stats": True}}, 400), ({"max_tokens": 5000}, 400),
    ({"messages": []}, 400), ({"messages": "hi"}, 400), ({"messages": [{"role": "robot", "content": "x"}]}, 400),
    ({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "http://x/y.png"}}]}]}, 400),
    ({"messages": [{"role": "user", "content": [{"type": "text", "text": "x", "extra": 1}]}]}, 400),
    ({"messages": [{"role": "user", "content": "word " * 5000}]}, 400),
    ({"messages": [{"role": "user", "content": "a"}, {"role": "system", "content": "late"}]}, 400),
    ({"model": "other-model"}, 404), ({"echo": True}, 400), ({"continue_final_message": True}, 400),
]


@pytest.mark.parametrize(("extra", "status"), BAD, ids=[json.dumps(b[0])[:40] for b in BAD])
def test_error_paths(server, extra, status) -> None:
    code, body = server.post("/v1/chat/completions", chat(**{"max_tokens": 4, **extra}))
    assert code == status, body
    assert body["error"]["code"] == status and body["error"]["message"]


TOOL = {"type": "function", "function": {"name": "f", "parameters": {}}}
# What vLLM refuses too, with its messages (the `(parameter=P, value=V)` suffix, and param) and order
SHARED_REFUSALS = [
    ({"tools": []}, "tools",
     "`tools` must not be an empty array. Either provide at least one tool or omit the field entirely. "
     "(parameter=tools)"),
    ({"tool_choice": "auto"}, "tool_choice", "When using `tool_choice`, `tools` must be set. (parameter=tool_choice)"),
    ({"tool_choice": "required", "tools": None}, "tool_choice",
     "When using `tool_choice`, `tools` must be set. (parameter=tool_choice)"),
    ({"tool_choice": "foo", "tools": [TOOL]}, "tool_choice",
     'Invalid value for `tool_choice`: foo! Only named tools, "none", "auto" or "required" are supported. '
     "(parameter=tool_choice)"),
    ({"response_format": {"type": "json_schema"}}, "response_format",
     "When response_format type is 'json_schema', the 'json_schema' field must be provided. "
     "(parameter=response_format)"),
    ({"stream_options": {"include_usage": True}}, "stream_options",
     "Stream options can only be defined when `stream=True`. (parameter=stream_options)"),
    ({"temperature": -1}, "temperature", "temperature must be non-negative, got -1.0. (parameter=temperature, "
                                         "value=-1.0)"),
    ({"temperature": 2.5}, "temperature", "temperature must be in [0, 2], got 2.5. (parameter=temperature, value=2.5)"),
    ({"top_p": 0}, "top_p", "top_p must be in (0, 1], got 0.0. (parameter=top_p, value=0.0)"),
    ({"top_p": 1.5}, "top_p", "top_p must be in (0, 1], got 1.5. (parameter=top_p, value=1.5)"),
    ({"top_k": -2}, None, "top_k must be 0 (disable), or at least 1, got -2."),
    ({"min_p": 1.5}, None, "min_p must be in [0, 1], got 1.5."),
    ({"max_tokens": 0}, "max_tokens", "max_tokens must be at least 1, got 0. (parameter=max_tokens, value=0)"),
    ({"max_tokens": None, "max_completion_tokens": -1}, "max_tokens",
     "max_tokens must be at least 1, got -1. (parameter=max_tokens, value=-1)"),
    ({"messages": [{"role": "user", "content": "a"}, {"role": "system", "content": "late"}]}, None,
     "System message must be at the beginning."),  # the chat template's reason alone
    ({"stop": ""}, None, "stop cannot contain an empty string."),
    ({"stop": ["a", ""]}, None, "stop cannot contain an empty string."),
    *[({"cache_salt": salt}, "cache_salt", "Parameter 'cache_salt' must be a non-empty string if provided. "
                                           "(parameter=cache_salt)") for salt in ("", 5, ["a"])],
    *[({"cache_salt": salt}, "cache_salt", "Parameter 'cache_salt' must be at most 128 characters and must not "
                                           "contain '@', '/', '\\\\', or NUL. (parameter=cache_salt)")
      for salt in ("x" * 129, "a/b", "a@b", "a\\b", "a\x00b")],
]


@pytest.mark.parametrize(("extra", "param", "message"), SHARED_REFUSALS,
                         ids=[json.dumps(r[0])[:40] for r in SHARED_REFUSALS])
def test_what_vllm_refuses_is_refused_with_its_message(server, extra, param, message) -> None:
    code, body = server.post("/v1/chat/completions", chat(**{"max_tokens": 4, **extra}))
    assert code == 400 and (body["error"]["param"], body["error"]["message"]) == (param, message), body


def test_vllms_seeds_and_top_p(server) -> None:
    """Seeds and top_p as vLLM takes them: a seed of -1 is unseeded; the signed 64-bit bounds are seeds (negative
    ones as their unsigned 64 bits); a top_p of any size above 0 is taken (`test_sampler`: below 1 / the vocabulary,
    the top token alone)."""
    def text(seed: int) -> str:
        return server.pred.text(server.prompt_ids, server.pred.tokens(server.prompt_ids, seed & ((1 << 64) - 1), 6))[0]

    code, r = server.post("/v1/chat/completions", chat(max_tokens=6, seed=-1))
    assert code == 200 and r["choices"][0]["message"]["content"] != text(-1), r
    for seed in ((1 << 63) - 1, -(1 << 63), -2):
        code, r = server.post("/v1/chat/completions", chat(max_tokens=6, seed=seed))
        assert code == 200 and r["choices"][0]["message"]["content"] == text(seed), (seed, r)
    for top_p in (1e-12, 1e-46):
        assert server.post("/v1/chat/completions", chat(max_tokens=2, top_p=top_p))[0] == 200


def test_a_temperature_below_a_hundredth_is_raised_to_it() -> None:
    """As vLLM's SamplingParams: a temperature in (0, 0.01) samples at 0.01, so only 0 is greedy."""
    temps = (0.0, 1e-9, 1e-6, 1e-5, 0.0099, 0.01, 0.7, 2.0)
    assert [sampling_temperature(t) for t in temps] == [0.0, 0.01, 0.01, 0.01, 0.01, 0.01, 0.7, 2.0]


def test_what_vllm_takes_is_taken(server) -> None:
    """vLLM takes `max_completion_tokens` when it is given, and checks only it; and an empty `stream_options`
    without `stream` (it refuses a non-empty one)."""
    code, body = server.post("/v1/chat/completions", chat(max_tokens=0, max_completion_tokens=2))
    assert code == 200 and body["usage"]["completion_tokens"] <= 2, body
    code, body = server.post("/v1/chat/completions", chat(max_tokens=2, stream_options={}))
    assert code == 200 and "usage" in body, body
    code, body = server.post("/v1/chat/completions", chat(max_tokens=2, stream_options={"include_usage": False}))
    assert code == 400 and body["error"]["param"] == "stream_options", body


VOCAB = 248_320  # the fixture's config.json
# Integers past what the engine keeps (once each failed the engine loop, or the frontend's encoding: a 500), and stop
# ids outside the vocabulary (vLLM's message) or too many of them
BAD_INTS = [
    {"stop_token_ids": [1 << 31]}, {"stop_token_ids": [1 << 32]}, {"stop_token_ids": [1 << 64]},
    {"stop_token_ids": [-5, 7]}, {"stop_token_ids": [VOCAB]}, {"stop_token_ids": list(range(17))},
    {"priority": 1 << 63}, {"priority": -(1 << 63) - 1}, {"seed": 1 << 63}, {"seed": -(1 << 63) - 1},
]


def test_out_of_range_integers_are_refused_or_clamped_and_the_server_serves_on(server) -> None:
    for extra in BAD_INTS:
        code, body = server.post("/v1/chat/completions", chat(max_tokens=4, **extra))
        assert code == 400 and body["error"]["param"] == next(iter(extra)), (extra, body)
    code, body = server.post("/v1/chat/completions", chat(max_tokens=4, stop_token_ids=[3, VOCAB, -1]))
    assert body["error"]["message"] == ("token_id(s) [248320, -1] in stop_token_ids contain out-of-vocab token ids. "
                                        "Vocabulary size: 248320 (parameter=stop_token_ids, value=[248320, -1])")
    toks = server.pred.tokens(server.prompt_ids, SEED + 4, 6)
    text = server.pred.text(server.prompt_ids, toks)[0]
    for top_k in (VOCAB, 1 << 31, 1 << 40):  # the vocabulary or more filters nothing
        code, r = server.post("/v1/chat/completions", chat(max_tokens=6, seed=SEED + 4, top_k=top_k))
        assert code == 200 and r["choices"][0]["message"]["content"] == text, (top_k, r)
    # deduplicated: 40 copies of one id are one, and stop it
    code, r = server.post("/v1/chat/completions", chat(max_tokens=6, seed=SEED + 4, stop_token_ids=[toks[2]] * 40))
    assert code == 200 and r["choices"][0]["stop_reason"] == toks[2] and r["usage"]["completion_tokens"] == 3
    code, r = server.post("/v1/chat/completions", chat(max_tokens=6, seed=SEED + 4, priority=(1 << 63) - 1))
    assert code == 200 and r["choices"][0]["message"]["content"] == text
    assert server.get("/health")[0] == 200


def test_known_fields_at_defaults_are_accepted(server) -> None:
    code, body = server.post("/v1/chat/completions", chat(
        max_tokens=2, n=1, logprobs=False, frequency_penalty=0, presence_penalty=0.0, repetition_penalty=1.0,
        tool_choice="none", top_logprobs=0, logit_bias={}, user="u", min_tokens=0, top_k=-1, min_p=0.0,
        max_completion_tokens=2, response_format={"type": "text"}, priority=0, model="qwen-test"))
    assert code == 200, body
    assert body["usage"]["completion_tokens"] == 2
    code, body = server.post("/v1/chat/completions", chat(max_tokens=2, model=""))  # an empty name is the served model
    assert code == 200, body


def test_other_error_routes(server) -> None:
    assert server.post("/v1/chat/completions", None, raw=b"{")[0] == 400
    for path, raw in (("/v1/chat/completions", b'{"messages": [{"role": "user", "content": "\xff"}], "max_tokens": 2}'),
                      ("/tokenize", b'{"prompt": "caf\xc3"}')):  # strings that are not UTF-8
        code, body = server.post(path, None, raw=raw)
        assert code == 400 and body["error"]["message"].startswith("invalid JSON: 'utf-8' codec"), (path, body)
    assert server.get("/v1/chat/completions")[0] == 405
    assert server.get("/nope")[0] == 404


def test_tokenize_models_health_metrics(server) -> None:
    code, r = server.post("/tokenize", {"messages": MSG, "chat_template_kwargs": KW})
    assert code == 200 and r["tokens"] == server.prompt_ids and r["count"] == len(server.prompt_ids)
    assert r["max_model_len"] == 4096 and r["token_strs"] is None
    code, r = server.post("/tokenize", {"prompt": "Hello world", "return_token_strs": True})
    assert code == 200 and r["tokens"] == server.pred.r.tokenizer.encode("Hello world").ids and r["token_strs"]
    code, r = server.post("/tokenize", {"prompt": "x", "messages": MSG})
    assert code == 400
    code, body = server.get("/v1/models")
    assert code == 200 and json.loads(body)["data"][0]["id"] == "qwen-test"
    assert server.get("/health")[0] == 200
    assert server.get("/health?ready=1")[0] == 200
    assert server.get("/ready")[0] == 200
    code, text = server.get("/metrics")
    text = text.decode()
    for name in ("vllm:num_requests_running{", "vllm:num_requests_waiting{", "vllm:kv_cache_usage_perc{",
                 "vllm:time_to_first_token_seconds_bucket{", "vllm:e2e_request_latency_seconds_count{",
                 "vllm:request_success_total{",
                 "vllm:prompt_tokens_total{", "vllm:generation_tokens_total{", "vllm:prefix_cache_queries_total{",
                 "vllm:num_preemptions_total{", "# TYPE vllm:request_success_total counter",
                 "llm_mini_infer:request_stage_seconds_bucket{", "llm_mini_infer:ttft_overhead_seconds_count{",
                 "llm_mini_infer:engine_step_cpu_seconds_bucket{", "vllm:iteration_tokens_total_bucket{",
                 "llm_mini_infer:step_decode_rows_bucket{", "llm_mini_infer:step_encoder_images_bucket{",
                 'model_name="qwen-test",engine="0"'):
        assert name in text, name
    assert "vllm:spec_decode" not in text  # MTP off: no speculative-decoding metrics
    assert "gpu_cache_usage_perc" not in text  # no alias of kv_cache_usage_perc: vLLM exports none
    # the engine loop's time by phase (engine.core.PHASES), every phase a series; the steps above planned and waited
    phases = {p: server.metric("llm_mini_infer:engine_phase_seconds_total", phase=p) for p in PHASES}
    assert all(v >= 0 for v in phases.values()) and phases["schedule"] > 0 and phases["wait"] > 0, phases
    assert server.metric("llm_mini_infer:engine_phase_seconds_total", phase="boundary") == \
        server.metric("llm_mini_infer:boundary_wait_seconds_total")
    # checkpoints by source (tests/sched/test_window_cache.py): resumes, and evictions by whether they served one
    from llm_mini_infer.engine.messages import CHECKPOINT_SOURCES

    for src in CHECKPOINT_SOURCES:
        assert server.metric("llm_mini_infer:checkpoint_resumes_total", source=src) >= 0
        for served in ("no", "yes"):
            assert server.metric("llm_mini_infer:checkpoint_evictions_total", source=src, served=served) >= 0
    assert server.metric("llm_mini_infer:checkpoint_resumes_total") == \
        server.metric("llm_mini_infer:checkpoint_hits_total")
    assert server.metric("llm_mini_infer:prefix_cache_windows") == 0  # not a sliding-window model
    assert server.metric("llm_mini_infer:prefix_cache_evicted_window_pages_total") == 0
    # garbage collections (tests/api/test_gcstats.py), the engine's and this frontend's, every generation a series
    for process in ("engine", "frontend"):
        for g in "012":
            assert server.metric("llm_mini_infer:gc_collections_total", process=process, generation=g) >= 0
    # the grammar-aware heads (tests/api/test_head_counters.py): every series, all 0 (the fake has neither head): the
    # steps' four paths and reasons, the union histogram's buckets, sum and count, the universe's three
    heads = [ln for ln in text.splitlines()
             if ln.startswith(("llm_mini_infer:grammar_head_", "llm_mini_infer:draft_universe_"))]
    assert len(heads) == 4 + len(UNION_BUCKETS) + 3 + 3, heads
    assert all(ln.endswith((" 0", " 0.0")) for ln in heads), heads


def test_mtp_stream_and_spec_decode_metrics(tmp_path) -> None:
    """MTP on (the fake runner drafts 3 and accepts some): the same text as without MTP, several tokens per event,
    the `vllm:spec_decode_*` counters, and per request its verify steps and their tokens (the request log line,
    llm_mini_infer:request_tokens_per_verify_step: tests/sched/test_spec_counts.py)."""
    model = require_qwen()
    pred = Predictor(model)
    prompt = pred.prompt(MSG, **KW)
    srv = Server(model, tmp_path, mtp=3,
                 extra={"observability": {"log_requests": True, "log_level": "info"}}).start()
    try:
        for i in range(4):
            events = srv.stream(chat(max_tokens=24, seed=SEED + 100 + i, ignore_eos=True))
            assert events[-1] == "[DONE]"
            assert content_of(events) == pred.text(prompt, pred.tokens(prompt, SEED + 100 + i, 24))[0]
        drafts = srv.metric("vllm:spec_decode_num_drafts_total")
        draft_tokens = srv.metric("vllm:spec_decode_num_draft_tokens_total")
        accepted = srv.metric("vllm:spec_decode_num_accepted_tokens_total")
        assert drafts > 0 and draft_tokens == 3 * drafts and 0 < accepted <= draft_tokens
        _, text = srv.get("/metrics")
        per_pos = {}
        for line in text.decode().splitlines():
            if line.startswith("vllm:spec_decode_num_accepted_tokens_per_pos_total{"):
                per_pos[line.split('position="')[1].split('"')[0]] = float(line.rsplit(" ", 1)[1])
        assert sorted(per_pos) == ["0", "1", "2"] and sum(per_pos.values()) == accepted
        assert per_pos["0"] >= per_pos["1"] >= per_pos["2"]
        # every request decodes 23 tokens after its first: fewer steps than tokens
        assert srv.metric("vllm:generation_tokens_total") == 4 * 24
        # one request at a time: each step after a prefill holds its one verify row (llm_mini_infer:step_decode_rows)
        rows, steps = (srv.metric("llm_mini_infer:step_decode_rows_sum"),
                       srv.metric("llm_mini_infer:step_decode_rows_count"))
        assert rows == drafts and steps >= rows
        # FlashInfer's attention-plan counters (tests/api/test_attention_counters.py): 0, the fake plans none
        for name in ("replans", "plans_kept", "plans_unsplit"):
            assert srv.metric(f"llm_mini_infer:attention_{name}_total") == 0
        # per request: each line's verify steps and tokens; every token after the first is a verify step's here (no
        # grammar, no preemption), so they sum to the `vllm:spec_decode_*` counters, and their ratios fill the
        # histogram's 1 to 4 buckets
        name = "llm_mini_infer:request_tokens_per_verify_step"
        lines = []
        deadline = time.monotonic() + 10
        while len(lines) < 4 and time.monotonic() < deadline:
            lines = re.findall(r" out=(\d+) .* verify_steps=(\d+) verify_tokens=(\d+)", srv.log_path.read_text())
            time.sleep(0.05)
        rows = [tuple(map(int, x)) for x in lines]
        assert len(rows) == 4 and all(out - 1 == toks for out, _, toks in rows), rows
        assert sum(n for _, n, _ in rows) == drafts and sum(t for _, _, t in rows) == drafts + accepted
        assert srv.metric(name + "_count") == 4
        assert srv.metric(name + "_sum") == pytest.approx(sum(t / n for _, n, t in rows))
        buckets = re.findall(name + r'_bucket\{[^}]*le="([^"]+)"\}', text.decode())
        assert buckets == [str(1 + i / 4) for i in range(13)] + ["+Inf"], buckets
    finally:
        assert srv.stop() == 0, srv.log_path.read_text()[-2000:]


def test_vllm_latency_series_start_and_end_where_vllms_do(tmp_path) -> None:
    """vLLM's TTFT and e2e start at its arrival_time, the request parsed and checked, and its queue, prefill,
    decode and inference times run on the engine's clock (queue from the engine's receipt of the request). A body sent
    300 ms after its headers: the wait is the `receive` stage and in the log line's ttft_ms (from the request's
    receipt), but neither in the TTFT histogram and the line's metric_ttft_ms nor in the TTFT overhead. Over
    every request, prefill is to_launch + to_first_token, inference prefill + decode, the time per output token
    decode's over the tokens after the first (0 for one token, observed as 0), and the queue lies within
    to_admit (which adds the IPC)."""
    model = require_qwen()
    srv = Server(model, tmp_path, step_ms=1.0,
                 extra={"observability": {"log_requests": True, "log_level": "info"}}).start()
    try:
        for i in range(3):
            assert srv.post(CHAT, chat(max_tokens=6, seed=SEED + 300 + i, stream=bool(i % 2)))[0] == 200
        ttft0, recv0 = srv.metric("vllm:time_to_first_token_seconds_sum"), srv.metric(
            "llm_mini_infer:request_stage_seconds_sum", stage="receive")
        over0 = srv.metric("llm_mini_infer:ttft_overhead_seconds_sum")
        body = json.dumps(chat(max_tokens=6, seed=SEED + 310, stream=True)).encode()
        with socket.create_connection(("127.0.0.1", srv.port)) as sock:
            sock.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                         b"X-Request-Id: slow-body\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n")
            time.sleep(0.3)
            sock.sendall(body)
            got = b""
            while b"[DONE]" not in got:
                got += sock.recv(65536)
        line = ""
        deadline = time.monotonic() + 10
        while not line and time.monotonic() < deadline:
            line = next((x for x in srv.log_path.read_text().splitlines() if "request id=slow-body " in x), "")
            time.sleep(0.05)
        f = dict(re.findall(r" (\w+_ms)=([\d.]+)", line))
        assert float(f["receive_ms"]) >= 290 and float(f["ttft_ms"]) >= float(f["receive_ms"]), line
        assert float(f["metric_ttft_ms"]) < float(f["ttft_ms"]) - 250 and float(f["metric_e2e_ms"]) < 250, line
        assert srv.metric("vllm:time_to_first_token_seconds_sum") - ttft0 < 0.25
        assert srv.metric("llm_mini_infer:request_stage_seconds_sum", stage="receive") - recv0 >= 0.29
        assert srv.metric("llm_mini_infer:ttft_overhead_seconds_sum") - over0 < 0.25

        def total(name: str, **labels: str) -> float:
            return srv.metric(name + "_sum", **labels)

        prefill, decode = total("vllm:request_prefill_time_seconds"), total("vllm:request_decode_time_seconds")
        assert prefill == pytest.approx(total("llm_mini_infer:request_stage_seconds", stage="to_launch") +
                                        total("llm_mini_infer:request_stage_seconds", stage="to_first_token"))
        assert total("vllm:request_inference_time_seconds") == pytest.approx(prefill + decode)
        queue = total("vllm:request_queue_time_seconds")  # observed for every request: the engine's stamp reached it
        assert srv.metric("vllm:request_queue_time_seconds_count") == 4 and 0 < queue <= \
            total("llm_mini_infer:request_stage_seconds", stage="to_admit")
        tpot = total("vllm:request_time_per_output_token_seconds")
        assert tpot == pytest.approx(decode / 5)  # 6 tokens each
        assert srv.post(CHAT, chat(max_tokens=1, seed=SEED + 320))[0] == 200
        deadline = time.monotonic() + 10
        while srv.metric("vllm:request_time_per_output_token_seconds_count") < 5 and time.monotonic() < deadline:
            time.sleep(0.05)
        assert srv.metric("vllm:request_time_per_output_token_seconds_count") == 5
        assert total("vllm:request_time_per_output_token_seconds") == tpot
    finally:
        assert srv.stop() == 0, srv.log_path.read_text()[-2000:]


CHAT = "/v1/chat/completions"


def test_http_counters(server) -> None:
    """Each response once in the instrumentator's `http_requests_total` (method, status grouped, handler: the route, or
    `none` for a path or a method no route serves) and with its exact code in llm_mini_infer:http_responses_total; the
    probes, the scrapes and /server_info in neither."""
    instrumentator = [("POST", "2xx", CHAT), ("POST", "4xx", CHAT), ("GET", "4xx", "none"),
                      ("POST", "2xx", "/tokenize"), ("GET", "2xx", "/v1/models")]
    ours = [(CHAT, "POST", "200"), (CHAT, "POST", "400"), (CHAT, "POST", "404"), (CHAT, "GET", "405"),
            ("none", "GET", "404"), ("/tokenize", "POST", "200"), ("/v1/models", "GET", "200")]

    def counts() -> list[float]:
        return [*(server.metric("http_requests_total", 0.0, method=m, status=s, handler=h)
                  for m, s, h in instrumentator),
                *(server.metric("llm_mini_infer:http_responses_total", 0.0, handler=h, method=m, status=s)
                  for h, m, s in ours)]

    before = counts()
    assert server.post(CHAT, chat(max_tokens=2, seed=SEED + 50))[0] == 200
    assert server.post(CHAT, chat(max_tokens=0))[0] == 400
    assert server.post(CHAT, chat(max_tokens=2, model="other-model"))[0] == 404
    assert server.get(CHAT)[0] == 405
    assert server.get("/nope")[0] == 404
    assert server.post("/tokenize", {"prompt": "hi"})[0] == 200
    assert server.get("/v1/models")[0] == 200
    for path in ("/health", "/health?ready=1", "/ready", "/live", "/server_info"):
        assert server.get(path)[0] == 200
    delta = [b - a for a, b in zip(before, counts())]
    assert delta == [1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1], delta
    _, text = server.get("/metrics")
    text = text.decode()
    assert "# TYPE http_requests_total counter" in text
    instrumentator_series = re.compile(
        r'http_requests_total\{method="[A-Z]+",status="[2-5]xx",handler="[^"]+"\} [0-9.]+')
    lines = [ln for ln in text.splitlines() if ln.startswith("http_requests_total")]
    assert lines and all(instrumentator_series.fullmatch(ln) for ln in lines), lines
    counted = {h for ln in text.splitlines()
               if ln.startswith(("http_requests_total{", "llm_mini_infer:http_responses_total"))
               for h in re.findall(r'handler="([^"]+)"', ln)}
    debug = {f"/debug/{d}" for d in ("stages", "engine", "images", "reset")}
    assert counted <= {CHAT, "/tokenize", "/detokenize", "/v1/models", "none", *debug}, counted


def test_http_tally_counts_each_response_once() -> None:
    """Without a server: a 503 and a 500 told apart, both 5xx in `http_requests_total`; a status line counted once
    whatever follows; the excluded paths matched anywhere in the path, as the instrumentator's patterns are; a method a
    POST route does not take is the instrumentator's `none`, its route here."""
    m = Metrics("m", 4096)
    routes = frozenset((CHAT, "/v1/models"))
    for status in (503, 500, 400, 200):
        t = tally_for(m, routes, CHAT, "POST")
        t.sent(status)
        t.sent(200)  # a later message of the same response
        t.left("2xx")
    tally_for(m, routes, CHAT, "GET").sent(405)
    tally_for(m, routes, "/v1/models", "GET").left("2xx")  # the client left before the status line
    assert all(tally_for(m, routes, p, "GET") is None for p in ("/health", "/healthz", "/v1/load_lora_adapter",
                                                                  "/metrics/x", "/ready", "/live", "/server_info"))
    assert m.http_responses.values == {
        (CHAT, "POST", "503"): 1, (CHAT, "POST", "500"): 1, (CHAT, "POST", "400"): 1, (CHAT, "POST", "200"): 1,
        (CHAT, "GET", "405"): 1, ("/v1/models", "GET", "499"): 1}
    assert m.http_requests.values == {("POST", "5xx", CHAT): 2, ("POST", "4xx", CHAT): 1, ("POST", "2xx", CHAT): 1,
                                      ("GET", "4xx", "none"): 1, ("GET", "2xx", "/v1/models"): 1}
    assert 'http_requests_total{method="POST",status="5xx",handler="/v1/chat/completions"} 2.0' in m.render().decode()


def _send(server, body: dict) -> socket.socket:
    data = json.dumps(body).encode()
    s = socket.create_connection(("127.0.0.1", server.port))
    s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
              b"Content-Length: %d\r\n\r\n" % len(data) + data)
    return s


def _until(cond, what: str, timeout: float = 10) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return
        time.sleep(0.05)
    raise AssertionError(what)


def test_client_disconnect_aborts(server) -> None:
    """A stream cut short after its first chunks: the engine aborts the request (a disconnect), and it counts as a
    disconnect of a stream whose 200 went out. A non-streamed request abandoned: aborted too, a 499 here (vLLM counts
    a 2xx: its chat route sends an empty 200 to the closed connection). Neither updates the finished-request
    statistics (`request_success`, e2e, the per-request histograms)."""
    def counts() -> list[float]:
        return [server.metric("llm_mini_infer:engine_aborts_total", 0.0, reason="disconnect"),
                server.metric("llm_mini_infer:client_disconnects_total", 0.0, handler=CHAT, stream="true"),
                server.metric("llm_mini_infer:client_disconnects_total", 0.0, handler=CHAT, stream="false"),
                server.metric("llm_mini_infer:http_responses_total", 0.0, handler=CHAT, method="POST", status="200"),
                server.metric("llm_mini_infer:http_responses_total", 0.0, handler=CHAT, method="POST", status="499"),
                server.metric("http_requests_total", 0.0, method="POST", status="2xx", handler=CHAT),
                sum(server.metric(name, 0.0) for name in ("vllm:request_success_total",
                                                          "vllm:e2e_request_latency_seconds_count",
                                                          "vllm:request_prompt_tokens_count"))]

    before = counts()
    s = _send(server, chat(max_tokens=3000, seed=SEED + 9, stream=True))
    got = b""
    while got.count(b"data: ") < 3:
        got += s.recv(65536)
    s.close()
    _until(lambda: counts()[0] >= before[0] + 1 and server.metric("vllm:num_requests_running") == 0,
           "the engine did not abort the disconnected stream")
    assert [b - a for a, b in zip(before, counts())] == [1, 1, 0, 1, 0, 1, 0]
    s = _send(server, chat(max_tokens=3000, seed=SEED + 10))
    time.sleep(0.3)
    s.close()
    _until(lambda: counts()[0] >= before[0] + 2 and server.metric("vllm:num_requests_running") == 0,
           "the engine did not abort the abandoned request")
    assert [b - a for a, b in zip(before, counts())] == [2, 1, 1, 1, 1, 2, 0]


def test_64_concurrent_streams(server) -> None:

    def one(i: int) -> bool:
        seed = 777_000 + i
        events = server.stream(chat(max_tokens=24, seed=seed))
        text = server.pred.text(server.prompt_ids, server.pred.tokens(server.prompt_ids, seed, 24))[0]
        return content_of(events) == text and events[-1] == "[DONE]"

    with ThreadPoolExecutor(64) as ex:
        assert all(ex.map(one, range(64)))
    assert server.metric("vllm:num_requests_running") == 0


def test_sigterm_drains_in_flight_requests(tmp_path) -> None:
    model = require_qwen()
    pred = Predictor(model)
    prompt = pred.prompt(MSG, **KW)
    with Server(model, tmp_path, step_ms=5).start() as srv:
        results: dict[int, list] = {}

        def one(i: int) -> None:
            results[i] = srv.stream(chat(max_tokens=120, seed=900 + i))

        threads = [threading.Thread(target=one, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        time.sleep(0.3)  # all four are streaming (120 steps of 5 ms)
        srv.proc.send_signal(signal.SIGTERM)
        time.sleep(0.3)
        try:
            code, _ = srv.post("/v1/chat/completions", chat(max_tokens=2))
            assert code == 503
        except OSError:
            pass  # already closed: also fine
        for t in threads:
            t.join(30)
        for i in range(4):
            text = pred.text(prompt, pred.tokens(prompt, 900 + i, 120))[0]
            assert content_of(results[i]) == text and results[i][-1] == "[DONE]"
        assert srv.proc.wait(30) == 0
        srv.log.close()


def test_default_chat_template_kwargs_sit_under_the_request_s(tmp_path) -> None:
    """model.chat_template_kwargs (vLLM's `--default-chat-template-kwargs`): a default a request may override."""
    model = require_qwen()
    srv = Server(model, tmp_path, extra={"model": {"chat_template_kwargs": KW}}).start()
    try:
        def ids(**kw) -> list[int]:
            status, body = srv.post("/tokenize", {"messages": MSG, **kw})
            assert status == 200, body
            return body["tokens"]

        plain = Predictor(model).prompt(MSG, **KW)
        assert ids() == plain == ids(chat_template_kwargs=KW)
        assert ids(chat_template_kwargs={"enable_thinking": True}) != plain  # the request wins
    finally:
        srv.stop()


# vLLM's 400 for a request's own chat template, byte for byte (no parameter suffix)
UNTRUSTED = (b'{"error":{"message":"Chat template is passed with request, but --trust-request-chat-template is not '
             b'set. Refused request with untrusted chat template.","type":"BadRequestError","param":null,"code":400}}')
TEMPLATE = "{{ messages[0].content }}"


def post_raw(srv: Server, path: str, body: dict) -> tuple[int, bytes]:
    c = srv.conn()
    try:
        c.request("POST", path, body=json.dumps(body), headers={"content-type": "application/json"})
        r = c.getresponse()
        return r.status, r.read()
    finally:
        c.close()


@pytest.mark.parametrize("path", [CHAT, "/tokenize"])
@pytest.mark.parametrize("extra", [{"chat_template": TEMPLATE}, {"chat_template_kwargs": {"chat_template": TEMPLATE}},
                                   {"chat_template_kwargs": {**KW, "chat_template": TEMPLATE}}, {"chat_template": ""},
                                   {"chat_template_kwargs": {**KW, "chat_template": False}}],
                         ids=["top-level", "in-kwargs", "beside-a-key", "empty", "false-in-kwargs"])
def test_a_requests_own_chat_template_is_refused_as_vllm_refuses_it(server, path, extra) -> None:
    """A template the request supplies, top-level or not null in its `chat_template_kwargs` (alone or beside another
    key; any value but null, an empty or false one too), on a chat and on /tokenize: vLLM's 400 and body exactly;
    a streamed chat gets the same plain 400."""
    body = {"messages": MSG, "chat_template_kwargs": KW, **({"max_tokens": 2} if path == CHAT else {}), **extra}
    assert post_raw(server, path, body) == (400, UNTRUSTED)
    if path == CHAT:
        assert post_raw(server, path, {**body, "stream": True}) == (400, UNTRUSTED)


@pytest.mark.parametrize("extra", [{"chat_template": None}, {"chat_template_kwargs": {**KW, "chat_template": None}},
                                   {"chat_template_kwargs": {**KW, "tokenize": True}}],
                         ids=["null", "null-in-kwargs", "tokenize"])
def test_a_null_template_and_tokenize_are_served(server, extra) -> None:
    """A null template, in either place, and `tokenize` (any value, dropped) are served: the prompt is the plain one."""
    code, body = server.post(CHAT, {"messages": MSG, "chat_template_kwargs": KW, "max_tokens": 2, **extra})
    assert code == 200, body
    code, body = server.post("/tokenize", {"messages": MSG, "chat_template_kwargs": KW, **extra})
    assert code == 200 and body["tokens"] == server.prompt_ids, body


def test_the_template_check_comes_after_the_model_check(server) -> None:
    """As vLLM orders them: a wrong model's 404 first, on both routes; on /tokenize the template's 400 before
    `tools` are checked or `continue_final_message` refused; its prompt form takes no template, so one there is
    ignored."""
    chat = {"messages": MSG, "max_tokens": 2, "chat_template": TEMPLATE}
    assert server.post(CHAT, {**chat, "model": "other"})[0] == 404
    assert server.post("/tokenize", {"messages": MSG, "chat_template": TEMPLATE, "model": "other"})[0] == 404
    for extra in ({"tools": [{"type": "function", "function": {"name": "f"}}]}, {"continue_final_message": True}):
        assert post_raw(server, "/tokenize", {"messages": MSG, "chat_template": TEMPLATE, **extra}) == (400, UNTRUSTED)
    code, body = server.post("/tokenize", {"prompt": "Hello", "chat_template": TEMPLATE})
    assert code == 200 and body == server.post("/tokenize", {"prompt": "Hello"})[1], body


def test_an_unknown_model_is_a_404_that_names_the_served_one(server) -> None:
    """vLLM's `_check_model` on its generation and tokenizer routes: the 404 lists the served names after "Valid
    aliases", the message sanitized as vLLM's `sanitize_message` does (a served name that is a path, under /mnt for
    one, becomes `<path>`, its final period with it)."""
    from llm_mini_infer.api.protocol import sanitize

    name = json.loads(server.get("/v1/models")[1])["data"][0]["id"]
    message = sanitize(f"The model `other` does not exist. Valid aliases: {name}.")
    want = {"error": {"message": message, "type": "NotFoundError", "param": "model", "code": 404}}
    for path, body in ((CHAT, {"messages": MSG, "max_tokens": 2}), ("/tokenize", {"prompt": "Hello"}),
                       ("/detokenize", {"tokens": [1]})):
        assert server.post(path, {**body, "model": "other"}) == (404, want), path
    assert sanitize("The model `other` does not exist. Valid aliases: /mnt/models/gemma-4.") == \
        "The model `other` does not exist. Valid aliases: <path>"
    assert sanitize("The model `x` does not exist. Valid aliases: google/gemma-4-31B-it.") == \
        "The model `x` does not exist. Valid aliases: google/gemma-4-31B-it."


def test_a_default_chat_template_or_tokenize_is_dropped(tmp_path) -> None:
    """model.chat_template_kwargs holding `chat_template` and `tokenize`: only the request's own kwargs are checked, so
    the server starts and serves, and the renderer drops both."""
    model = require_qwen()
    srv = Server(model, tmp_path, extra={"model": {"chat_template_kwargs": {**KW, "chat_template": TEMPLATE,
                                                                          "tokenize": True}}}).start()
    try:
        code, body = srv.post("/tokenize", {"messages": MSG})
        assert code == 200 and body["tokens"] == Predictor(model).prompt(MSG, **KW), body
        code, body = srv.post(CHAT, {"messages": MSG, "max_tokens": 2})
        assert code == 200, body
    finally:
        srv.stop()
