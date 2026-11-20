"""The host check (llm_mini_infer.hostcheck) and its synthetic workload (llm_mini_infer.synthetic).

- A server that fails to start shows why in the check's own output, since in a Kubernetes Job
  (deploy/helm/examples/hostcheck-job.yaml) its log sits on the pod's volume, which nothing can read once the pod has
  ended.
- Gemma 4's load: the request's body, the synthetic workload's two image sizes, schema and prompt, a mounted folder
  with its optional user text and references, Poisson arrivals, the server's request log and metrics, and the
  verdicts, among them the drafter's. End to end on the fake runner with Gemma 4's configuration and tokenizer.
"""

from __future__ import annotations

import argparse
import io
import itertools
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from llm_mini_infer import cli, hostcheck, synthetic
from llm_mini_infer.vision.gemma import GemmaVision
from llm_mini_infer.vision.probe import probe

from ..support.api_util import env_has_model, free_port
from ..support.metrics_pages import VLLM_METRICS
from ..support.paths import MODELS_DIR

GEMMA = MODELS_DIR / "gemma-4-31b-it"
needs_gemma = pytest.mark.skipif(not env_has_model(GEMMA),
                                 reason="tests/.fixtures/models/gemma-4-31b-it is absent: "
                                        "uv run tests/scripts/fetch_fixtures.py")


def test_a_server_that_fails_to_start_shows_its_log(tmp_path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"model:\n  path: {tmp_path / 'no-checkpoint'}\nrunner:\n  kind: fake\n")
    srv = hostcheck.Server(str(cfg), free_port(), tmp_path / "server.log", [])
    with pytest.raises(RuntimeError) as err:
        srv.wait_ready(60)
    msg = str(err.value)
    assert msg.startswith("the server exited during startup (code 2); the last ") and "lines of its log" in msg
    assert f"model directory {tmp_path / 'no-checkpoint'} does not exist" in msg


# -- the host ----------------------------------------------------------------------------------------------------------


def test_the_host_record_names_atens_cpu_capability_first() -> None:
    """A log line that cuts the record short (at 240 characters, say) keeps the CPU's facts: they come first, the
    kernel's release last."""
    import torch

    host = hostcheck.host_info()
    assert list(host) == ["cpu", "cpu_capability", "isa", "cpus", "cpu_quota", "cpu_quota_source", "kernel"]
    assert host["cpu_capability"] == torch.backends.cpu.get_cpu_capability()
    cpuinfo = Path("/proc/cpuinfo")
    assert host["isa"] == (hostcheck.cpu_isa(cpuinfo.read_text()) if cpuinfo.exists() else [])


def test_the_isa_flags_come_from_the_first_flags_line() -> None:
    zen4 = ("processor\t: 0\nmodel name\t: AMD EPYC 9554 64-Core Processor\nflags\t\t: fpu sse2 avx fma avx2 avx512f "
            "avx512dq avx512bw avx512vl avx512_bf16 avx512_vnni\n\nprocessor\t: 1\nflags\t\t: fpu\n")
    zen3 = "processor\t: 0\nmodel name\t: AMD EPYC 7443P 24-Core Processor\nflags\t\t: fpu sse2 avx fma avx2 sha_ni\n"
    arm = "processor\t: 0\nBogoMIPS\t: 50.00\nFeatures\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics\n"
    assert hostcheck.cpu_isa(zen4) == ["avx2", "fma", "avx512f", "avx512bw", "avx512vl", "avx512dq"]
    assert hostcheck.cpu_isa(zen3) == ["avx2", "fma"]
    assert hostcheck.cpu_isa(arm) == [] and hostcheck.cpu_isa("") == []


# -- the synthetic workload --------------------------------------------------------------------------------------------

# The synthetic schema's attributes, by name: (values in the set, the list's maximum length, an empty list refused)
OBJECT = {"kind": (8, 2, True), "colour": (10, 3, True), "shape": (5, 2, True), "position": (5, 2, True),
          "material": (7, 3, False), "condition": (3, 1, False), "texture": (5, 2, False), "parts": (7, 4, False)}
SETTING = {"place": (7, 2, True), "light": (4, 2, False), "walls": (4, 2, False)}


def attribute_shape(prop: dict) -> tuple[int, int, bool]:
    arr, null = prop["anyOf"]
    assert null == {"type": "null"} and arr["type"] == "array" and arr["items"]["type"] == "string"
    assert len(set(arr["items"]["enum"])) == len(arr["items"]["enum"])
    return len(arr["items"]["enum"]), arr["maxItems"], arr.get("minItems") == 1


def test_the_synthetic_schemas_shape() -> None:
    s = synthetic.scene_schema()
    obj, setting = s["$defs"]["Object"], s["$defs"]["Setting"]
    for d in (s, obj, setting):  # strict: no other keys, every key required in order
        assert d["type"] == "object" and d["additionalProperties"] is False
        assert d["required"] == list(d["properties"])
    assert list(s["properties"]) == ["count", "objects", "setting"] and s["properties"]["count"]["type"] == "integer"
    assert s["properties"]["objects"] == {"items": {"$ref": "#/$defs/Object"}, "maxItems": synthetic.MAX_OBJECTS,
                                          "title": "Objects", "type": "array"}
    assert synthetic.MAX_OBJECTS == 6 and s["properties"]["setting"] == {"$ref": "#/$defs/Setting"}
    assert list(obj["properties"]) == ["id", *OBJECT, "size", "occluded"]
    assert obj["properties"]["id"]["type"] == "integer"
    assert [n for n, *_ in synthetic.OBJECT] == list(OBJECT)
    assert {a: attribute_shape(obj["properties"][a]) for a in OBJECT} == OBJECT
    assert sum(v[2] for v in OBJECT.values()) == 4 and list(OBJECT.values())[:4] == [v for v in OBJECT.values() if v[2]]
    assert obj["properties"]["size"] == {"enum": list(synthetic.SIZE), "title": "Size", "type": "string"}
    assert obj["properties"]["occluded"]["type"] == "boolean"
    assert {a: attribute_shape(setting["properties"][a]) for a in SETTING} == SETTING
    assert synthetic.scene_schema() == s and hostcheck.schema_name(s) == synthetic.SCHEMA_NAME == "scene"


def test_the_synthetic_images_have_two_sizes_and_never_repeat() -> None:
    family = GemmaVision(boi_id=255999, image_id=258880, eoi_id=258882, max_soft_tokens=280)
    images = [synthetic.image_jpeg(i) for i in range(40)]
    assert len(set(images)) == 40
    sizes = [probe(f).oriented for f in images]
    assert sizes.count((720, 1280)) == 10 and sizes.count((1280, 720)) == 30  # TALL_SHARE: a quarter portrait
    grids = {}
    for f, wh in zip(images, sizes, strict=True):
        size = family.size(probe(f))
        assert size.tokens == 264 and size.grid[1] * size.grid[2] == 2376
        grids[wh] = size.grid[1:]
    assert grids == {(1280, 720): (36, 66), (720, 1280): (66, 36)}  # the portrait one's grid transposed


def test_a_synthetic_workloads_label_carries_its_version() -> None:
    """A reference measured on another definition of the workload never matches."""
    inp = hostcheck.synthetic_inputs(lambda t: len(t.split()), 300, 0.25)
    assert inp.workload["version"] == synthetic.WORKLOAD_VERSION
    assert inp.workload == {"mode": "synthetic", "version": synthetic.WORKLOAD_VERSION, "prompt_tokens": 300,
                            "tall_share": 0.25}


@pytest.mark.parametrize("share,tall", [(0.0, 0), (0.15, 3), (0.25, 5), (0.5, 10), (1.0, 20)])
def test_the_share_of_tall_images_is_spread_and_exact(share, tall) -> None:
    """p tall images in every q (the share p/q in lowest terms), spread out: never two tall images p/q apart less."""
    shapes = [synthetic.image_size(i, share) for i in range(20)]
    assert shapes.count(synthetic.SIZES[1]) == tall and shapes.count(synthetic.SIZES[0]) == 20 - tall
    assert synthetic.image_jpeg(3, share) == synthetic.image_jpeg(3, share)
    if 0 < share < 1:
        idx = [i for i, s in enumerate(shapes) if s == synthetic.SIZES[1]]
        assert min(b - a for a, b in itertools.pairwise(idx)) >= int(1 / share)
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError, match="between 0 and 1"):
            synthetic.image_size(0, bad)


def test_the_synthetic_prompt_reaches_its_length() -> None:
    words = synthetic.system_prompt(lambda t: len(t.split()), target=1500)
    assert 1500 <= len(words.split()) < 1560
    for name in (*OBJECT, *SETTING, "size", "occluded"):
        assert f"\n- {name}: " in words


@needs_gemma
def test_the_synthetic_prompt_reaches_its_length_in_gemmas_tokens() -> None:
    from llm_mini_infer.render.renderer import Renderer

    tok = Renderer(str(GEMMA)).tokenizer
    for target in (synthetic.PROMPT_TOKENS, 600):
        n = len(tok.encode(synthetic.system_prompt(lambda t: len(tok.encode(t).ids), target)).ids)
        assert target <= n <= target + 100


def test_cli_options_have_help() -> None:
    """Every optional CLI argument must document its purpose."""
    parsers = {
        "serve": cli.serve_args(argparse.ArgumentParser()),
        "hostcheck": hostcheck.build_parser(),
    }

    for name, parser in parsers.items():
        for action in parser._actions:
            if action.option_strings:
                assert action.help, f"{name} option {action.option_strings} has no help text"


# -- the request, the arrivals, the parsing ----------------------------------------------------------------------------

def test_the_request() -> None:
    body = json.loads(hostcheck.request_body("M", "the prompt", b"\xff\xd8jpeg", {"type": "object"}, "scene", 1024,
                                             "Look."))
    assert body == {
        "model": "M", "temperature": 0, "max_tokens": 1024,
        "messages": [{"role": "system", "content": [{"type": "text", "text": "the prompt"}]},
                     {"role": "user", "content": [{"type": "text", "text": "Look."},
                                                  {"type": "image_url",
                                                   "image_url": {"url": "data:image/jpeg;base64,/9hqcGVn"}}]}],
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "scene", "strict": True, "schema": {"type": "object"}}},
        "chat_template_kwargs": {"enable_thinking": False}}
    assert "stream" not in body  # not streamed
    default = json.loads(hostcheck.request_body("M", "p", b"x", {}, "n", 8))
    assert default["messages"][1]["content"][0]["text"] == hostcheck.DEFAULT_USER_TEXT


def test_poisson_arrivals_are_normalized_and_fixed_by_their_seed() -> None:
    a = hostcheck.poisson_arrivals(2.5, 400, 3)
    assert a[0] == 0 and a == sorted(a) and a == hostcheck.poisson_arrivals(2.5, 400, 3)
    assert 0.95 * 400 / 2.5 < a[-1] < 400 / 2.5  # the 400th gap ends at 400 / rate exactly
    sched = hostcheck.schedule(5.0, 30, 120, 15, 1)
    assert all(t < 165 for t in sched) and abs(len(sched) - 5.0 * 165) < 3
    args = argparse.Namespace(rates="2,4", load_warmup=30.0, duration=None, tail=15.0, seed=0)
    assert 990 <= hostcheck.frames_needed(args) <= 1020  # 16 + 165 s at 2/s + 165 s at 4/s


METRICS = """# HELP vllm:spec_decode_num_drafts_total x
vllm:spec_decode_num_drafts_total{model_name="m",engine="0"} %(drafts)s
vllm:spec_decode_num_accepted_tokens_per_pos_total{model_name="m",engine="0",position="0"} %(p0)s
vllm:spec_decode_num_accepted_tokens_per_pos_total{model_name="m",engine="0",position="1"} %(p1)s
llm_mini_infer:gpu_busy_seconds_total{model_name="m",engine="0"} %(busy)s
llm_mini_infer:gpu_idle_seconds_total{model_name="m",engine="0"} %(idle)s
llm_mini_infer:grammar_head_steps_total{model_name="m",engine="0",path="sparse",reason="union"} %(sparse)s
llm_mini_infer:grammar_head_steps_total{model_name="m",engine="0",path="dense",reason="over_cap"} %(dense)s
llm_mini_infer:draft_universe_tokens{model_name="m",engine="0"} 351
llm_mini_infer:grammar_forced_tokens_total{model_name="m",engine="0"} %(forced)s
llm_mini_infer:attention_plans_kept_total{model_name="m",engine="0"} %(kept)s
llm_mini_infer:requests_failed_total{model_name="m",engine="0"} %(failed)s
llm_mini_infer:grammar_errors_total{model_name="m",engine="0"} %(gerr)s
llm_mini_infer:grammar_head_union_tokens_bucket{model_name="m",engine="0",le="16"} %(u16)s
llm_mini_infer:grammar_head_union_tokens_bucket{model_name="m",engine="0",le="32"} %(u32)s
llm_mini_infer:grammar_head_union_tokens_bucket{model_name="m",engine="0",le="64"} %(u64)s
llm_mini_infer:grammar_head_union_tokens_bucket{model_name="m",engine="0",le="+Inf"} %(u64)s
llm_mini_infer:grammar_head_union_tokens_sum{model_name="m",engine="0"} %(usum)s
llm_mini_infer:grammar_head_union_tokens_count{model_name="m",engine="0"} %(u64)s
llm_mini_infer:draft_universe_steps_total{model_name="m",engine="0"} %(usteps)s
llm_mini_infer:draft_universe_rebuilds_total{model_name="m",engine="0"} %(rebuilds)s
llm_mini_infer:grammar_forced_requests_total{model_name="m",engine="0"} %(freq)s
llm_mini_infer:grammar_eos_free_finishes_total{model_name="m",engine="0",in_flight="no"} %(eos_no)s
llm_mini_infer:grammar_eos_free_finishes_total{model_name="m",engine="0",in_flight="yes"} %(eos_yes)s
llm_mini_infer:grammar_eos_free_gated_total{model_name="m",engine="0"} %(gated)s
llm_mini_infer:spec_first_rejections_total{model_name="m",engine="0",mask="forced"} 0
llm_mini_infer:spec_first_rejections_total{model_name="m",engine="0",mask="free"} %(rej_free)s
vllm:generation_tokens_total{model_name="m",engine="0"} %(gen)s
"""


def page(**kw: float) -> dict[str, float]:
    base = {"drafts": 0, "p0": 0, "p1": 0, "busy": 0, "idle": 0, "sparse": 0, "dense": 0, "forced": 0, "kept": 0,
            "u16": 0, "u32": 0, "u64": 0, "usum": 0, "usteps": 0, "rebuilds": 0, "freq": 0, "eos_no": 0, "eos_yes": 0,
            "gated": 0, "rej_free": 0, "gen": 0, "failed": 0, "gerr": 0}
    return hostcheck.prometheus(METRICS % {**base, **kw})


def test_the_engines_window_from_its_metrics() -> None:
    a = page(drafts=100, p0=50, p1=20, busy=10, idle=0.01, sparse=90, dense=0, forced=40, u16=10, u32=50, u64=90,
             usum=2000, usteps=80, rebuilds=2, freq=20, eos_no=5, eos_yes=1, gated=1, rej_free=7, gen=900)
    b = page(drafts=300, p0=230, p1=180, busy=30, idle=0.02, sparse=290, dense=10, forced=80, u16=40, u32=150, u64=290,
             usum=7400, usteps=270, rebuilds=3, freq=40, eos_no=20, eos_yes=3, gated=2, rej_free=37, gen=1500,
             failed=2, gerr=1)
    d = hostcheck.engine_delta(a, b, completed=20)
    assert d["accept"] == {"drafts": 200, "per_pos": [0.9, 0.8], "tokens_per_verify": 2.7,
                           "verify_steps_per_request": 10.0}
    assert d["gpu_idle_pct"] == round(100 * 0.01 / 20.01, 4)
    assert d["head_steps"] == {"sparse": 200, "dense": 10} and d["sparse_share"] == round(200 / 210, 4)
    assert d["head_dense_reasons"] == {"over_cap": 10} and d["draft_universe_tokens"] == 351
    assert d["forced_tokens_per_request"] == 2.0 and d["plans_kept"] == 0 and d["plans_kept_since_start"] == 0
    # 200 unions in the window: 30 of at most 16 tokens, 100 of at most 32 (the p50's bucket), 200 of at most 64
    assert d["union_tokens"] == {"steps": 200, "mean": 27.0, "p50": 32, "p90": 64}
    assert (d["draft_universe_steps"], d["draft_universe_rebuilds"]) == (190, 1)
    assert (d["forced_requests"], d["forced_tokens"], d["generation_tokens"]) == (20, 40, 600)
    assert (d["requests_failed"], d["grammar_errors"]) == (2, 1)
    assert d["eos_free_finishes"] == {"no": 15, "yes": 2} and d["eos_free_gated"] == 1
    assert d["spec_first_rejections"] == {"forced": 0, "free": 30}
    assert hostcheck.engine_delta(a, a, completed=0)["union_tokens"] is None  # no union in the window
    # vLLM's page: acceptance per position from its counters, none of llm-mini-infer's
    v = [hostcheck.prometheus(VLLM_METRICS % kw) for kw in (dict(drafts=10, p0=8, p1=6, p2=4, gen=100),
                                                            dict(drafts=110, p0=88, p1=66, p2=44, gen=500))]
    dv = hostcheck.engine_delta(*v, completed=10)
    assert dv["accept"] == {"drafts": 100, "per_pos": [0.8, 0.6, 0.4], "tokens_per_verify": 2.8,
                            "verify_steps_per_request": 10.0}
    assert dv["generation_tokens"] == 400 and dv["head_steps"] == {} and dv["sparse_share"] is None
    assert all(dv[k] is None for k in ("union_tokens", "draft_universe_steps", "draft_universe_rebuilds",
                                       "forced_requests", "forced_tokens", "forced_tokens_per_request",
                                       "eos_free_finishes", "eos_free_gated", "spec_first_rejections", "gpu_idle_pct",
                                       "draft_universe_tokens", "requests_failed", "grammar_errors"))


def test_ttft_comes_from_the_servers_request_log(tmp_path) -> None:
    """Each request's TTFT from its line; a line written after its response was read (the frontend logs a request
    once its response is sent) is waited for, a little."""
    import threading
    import time

    log = tmp_path / "server.log"
    log.write_text("INFO 10-14 [frontend-0] request id=hostcheck-0-000001 status=200 finish=stop stream=0 prompt=2700 "
                   "cached=2624 images=1 out=101 ttft_ms=85.3 e2e_ms=720.1\n"
                   "INFO 10-14 [frontend-0] request id=hostcheck-1-000001 status=200 finish=stop stream=0 prompt=2700 "
                   "cached=2624 images=1 out=99 ttft_ms=90.0 e2e_ms=700.0\n"
                   "INFO 10-14 [frontend-0] request id=hostcheck-0-000002 status=400 error=BadRequestError "
                   "e2e_ms=3.0\n")
    assert hostcheck.server_requests(log, "hostcheck-0-") == {"hostcheck-0-000001": 85.3}
    late = ("INFO 10-14 [frontend-0] request id=hostcheck-0-000003 status=200 finish=stop stream=0 prompt=2700 "
            "cached=2624 images=1 out=98 ttft_ms=70.5 e2e_ms=690.0\n")
    writer = threading.Timer(0.3, lambda: log.write_text(log.read_text() + late))
    writer.start()
    t0 = time.monotonic()
    got = hostcheck.server_requests(log, "hostcheck-0-", {"hostcheck-0-000001", "hostcheck-0-000003"})
    assert got == {"hostcheck-0-000001": 85.3, "hostcheck-0-000003": 70.5} and time.monotonic() - t0 < 3
    writer.join()
    t0 = time.monotonic()  # a line that never comes: the wait ends
    assert hostcheck.server_requests(log, "hostcheck-0-", {"hostcheck-0-000009"}, wait_s=0.2) == got
    assert time.monotonic() - t0 >= 0.2


# -- the verdicts ------------------------------------------------------------------------------------------------------

def info(model: str | None, k: int, kind: str = "cuda") -> dict:
    return {"config": {"speculative": {"model": model, "num_speculative_tokens": k}, "runner": {"kind": kind},
                       "speculative_source": f"{model}@abc" if model else ""}, "reasons": {}}


def test_the_drafters_verdict() -> None:
    line = "INFO model loaded in 9.1 s: ..., MTP k = 5 with the assistant (0.92 GiB, ...)"
    ok = hostcheck.drafter_status(info("/hf/assistant", 5), line, gemma=True)
    assert ok["ok"] and ok["k"] == 5 and ok["startup_line"] == "MTP k = 5 with the assistant"
    unset = hostcheck.drafter_status(info(None, 0), "", gemma=True)
    assert not unset["ok"] and unset["why"].startswith("speculative.model is unset")
    silent = hostcheck.drafter_status(info("/hf/assistant", 5), "no such line", gemma=True)
    assert not silent["ok"] and "no startup line" in silent["why"]
    assert hostcheck.drafter_status(info("/hf/assistant", 5, "fake"), "", gemma=True)["ok"]  # the fake logs none
    assert hostcheck.drafter_status(info(None, 3), "", gemma=False)["ok"]  # Qwen drafts with its own MTP layer


def load(**kw) -> dict:
    eng = {"accept": {"tokens_per_verify": 4.5, "drafts": 900}, "gpu_idle_pct": 0.01, "sparse_share": 1.0,
           "plans_kept_since_start": 0, "requests_failed": 0, "grammar_errors": 0, "head_dense_reasons": {}}
    eng.update(kw.pop("engine", {}))
    return {"rate": 2.5, "failed": 0, "dropped": 0, "errors": {}, "e2e_ms": {"n": 300, "p50": 600, "p99": 900},
            "reply_tokens": {"mean": 100}, "engine": eng, **kw}


# A reference file's entry, as a mounted folder's reference.json holds it: a made-up 450 W card
REFERENCE = {"card": "a 450 W card", "card_w": 450, "startup_s": [26.0, 28.0], "accept_per_pos": [0.9, 0.8],
             "loads": {"2.5": {"e2e": [640, 880], "ttft": [77, 108], "reply": 97, "tpv": 4.35, "power": 316,
                               "sm": 2421}}}


def test_the_loads_verdicts(tmp_path) -> None:
    (tmp_path / "reference.json").write_text(json.dumps(REFERENCE))
    (ref,) = hostcheck.load_reference(tmp_path / "reference.json")
    good = hostcheck.load_verdicts(load(), ref, "mounted")
    assert {k: v for k, v in good.items() if v != "ok"} == {}
    bad = hostcheck.load_verdicts(load(failed=2, errors={"500 InternalServerError": 2}, e2e_ms={
        "n": 300, "p50": 2000, "p99": 5000}, engine={"plans_kept_since_start": 1, "accept": {"tokens_per_verify": 2.5},
                                                     "gpu_idle_pct": 4.0}, reply_tokens={"mean": 30}),
                                  ref, "mounted")
    assert bad["requests"].startswith("FAILED") and bad["plans_kept"].startswith("FAILED")
    assert bad["e2e"] == "SLOW: beyond 1.25x the reference" and bad["gpu_idle"].startswith("SLOW")
    assert bad["accept"] == "LOW: below 0.9x the reference (per position: 0.90 / 0.80)"
    assert bad["reply"].startswith("differs")
    assert "reply" not in hostcheck.load_verdicts(load(reply_tokens={"mean": 30}), ref, "synthetic")
    assert hostcheck.load_verdicts(load(rate=7.0), ref, "mounted")["e2e"] == "no reference at this rate"
    none = hostcheck.load_verdicts(load(), None, "mounted")  # no reference matches
    assert none["e2e"] == none["accept"] == "no reference" and "reply" not in none and none["requests"] == "ok"


def run(**kw) -> dict:
    """A run's labels (`run_labels`), a made-up card's."""
    return {"gpu": "Card X", "card_w": 450, "model": "org/model", "drafter": "org/drafter", "k": 4,
            "workload": {"mode": "synthetic", "prompt_tokens": 1000, "tall_share": 0.5, "max_soft_tokens": 140}, **kw}


def test_a_reference_applies_to_its_own_power_class_and_labels_only() -> None:
    """Power classes are the limit to the nearest 50 W. A reference without labels (a reference.json from an older
    report) matches on its power class alone; one with labels needs each of them; no other card's is taken."""
    assert [hostcheck.power_class(w) for w in (250.0, 262.0, 449.0, 476.0, None)] == [250, 250, 450, 500, None]
    bare = {"card_w": 450, "loads": {}}
    labelled = {"card_w": 450, "gpu": "Card X", "model": "org/model", "drafter": "org/drafter", "k": 4,
                "workload": {"mode": "synthetic", "prompt_tokens": 1000, "tall_share": 0.5, "max_soft_tokens": 140},
                "loads": {}}
    other = {**labelled, "card_w": 250}
    assert hostcheck.matching_reference([other, labelled], run()) is labelled
    assert hostcheck.matching_reference([other, bare], run()) is bare
    assert hostcheck.matching_reference([other], run()) is None  # a 250 W card's: no verdict, no fallback
    for label, value in (("gpu", "Card Y"), ("model", "org/other"), ("drafter", None), ("k", 3)):
        assert hostcheck.matching_reference([labelled], run(**{label: value})) is None, label
    for key, value in (("prompt_tokens", 2000), ("tall_share", 0.25), ("max_soft_tokens", 280), ("mode", "mounted")):
        changed = run(workload={**run()["workload"], key: value})
        assert hostcheck.matching_reference([labelled], changed) is None, key
    assert hostcheck.matching_reference([bare], run(card_w=None)) is None  # no GPU: no power class


def test_the_packaged_references_are_labelled() -> None:
    """Every packaged load reference names what it was measured on, so a run on anything else gets no verdict; the host
    measures name their hosts."""
    host, refs = hostcheck.packaged_references()
    assert host["hosts"] and all(len(host[k]) == 2 for k in hostcheck.HOST_MARGINS)
    assert refs
    for ref in refs:
        assert {"card", "card_w", "gpu", "model", "workload", "startup_s", "loads"} <= set(ref), ref.get("card")
        assert ref["workload"]["mode"] == "synthetic" and ref["card_w"] == hostcheck.power_class(ref["card_w"])
    assert hostcheck.verdict("python_loop_s", host["python_loop_s"][0], host) == "ok"
    assert hostcheck.verdict("python_loop_s", 2 * host["python_loop_s"][0], host) == "SLOW"
    assert hostcheck.verdict("ttft_overhead_p50_ms", 1.9) == "ok" and hostcheck.verdict("gpu_idle_pct", 3.5) == "SLOW"


def test_every_packaged_load_reference_names_its_workloads_version() -> None:
    """A reference's workload carries the synthetic workload's version (an integer), so a run of another definition of
    the workload never matches it."""
    _, refs = hostcheck.packaged_references()
    for ref in refs:
        version = ref["workload"].get("version")
        assert isinstance(version, int) and not isinstance(version, bool), ref.get("card")
        same = {k: ref[k] for k in ("gpu", "card_w", "model", "drafter", "k") if k in ref}
        same["workload"] = {**ref["workload"], "version": synthetic.WORKLOAD_VERSION}
        other = {**same, "workload": {**same["workload"], "version": version + 1}}
        assert hostcheck.matching_reference([ref], other) is None, ref.get("card")
        assert (hostcheck.matching_reference([ref], same) is ref) == (version == synthetic.WORKLOAD_VERSION)
        assert hostcheck.matching_reference([ref], {**same, "workload": {**same["workload"], "version": None}}) is None



def test_the_packaged_load_reference_is_what_a_default_run_measures() -> None:
    """The packaged entry's labels are a default synthetic run's on its GPU, model and drafter (Gemma 4's own image
    budget, 280): a change to the workload's defaults without a new version fails here, not silently in the field."""
    _, refs = hostcheck.packaged_references()
    ref = refs[0]
    inputs = hostcheck.synthetic_inputs(lambda t: len(t.split()), synthetic.PROMPT_TOKENS, synthetic.TALL_SHARE)
    report = {"gpu": {"gpus": [{"name": ref["gpu"], "power_limit_w": ref["card_w"]}]},
              "drafter": {"model": ref["drafter"], "k": ref["k"]}}
    info = {"config": {"model_source": ref["model"], "images": {"max_soft_tokens": 280}}}
    labels = hostcheck.run_labels(report, info, inputs, max_tokens=1024)
    assert labels["workload"] == ref["workload"]
    assert hostcheck.matching_reference(refs, labels) is ref

def streaming_server(status: int):
    """A local HTTP server in a thread: a streamed body (or `status`), the Authorization header recorded."""
    seen: list[str | None] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            seen.append(self.headers.get("Authorization"))
            self.send_response(status)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            if status == 200:
                self.wfile.write(b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\ndata: [DONE]\n\n')

        def log_message(self, *args) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, seen


def test_a_streamed_request_sends_the_api_key_and_a_refusal_raises(monkeypatch) -> None:
    monkeypatch.delenv("LLM_MINI_INFER_API_KEY", raising=False)
    monkeypatch.setenv("VLLM_API_KEY", "the-key")
    for status in (200, 401):
        server, thread, seen = streaming_server(status)
        try:
            port = server.server_address[1]
            if status == 200:
                first, total = hostcheck._stream(port, {"model": "m", "stream": True})
                assert 0 <= first <= total or first != first
            else:
                with pytest.raises(RuntimeError, match="401"):
                    hostcheck._stream(port, {"model": "m", "stream": True})
            assert seen == ["Bearer the-key"]
        finally:
            server.shutdown()
            server.server_close()
            thread.join(5)


def test_a_reports_own_results_make_a_reference_file(tmp_path) -> None:
    report = {"gpu": {"gpus": [{"name": "GPU", "power_limit_w": 600.0}]}, "startup_s": [40.0, 26.5],
              "loads": [{"rate": 2.5, "e2e_ms": {"p50": 640.0, "p99": 880.0}, "ttft_ms": {"p50": 77.0, "p99": 108.0},
                         "reply_tokens": {"mean": 97.0}, "gpu": {"power_w": 316.0, "sm_mhz": 2421},
                         "engine": {"accept": {"tokens_per_verify": 4.35, "per_pos": [0.9, 0.8]}}}]}
    report["reference"] = {"run": run(gpu="GPU", card_w=600)}
    out = hostcheck.as_reference(report)
    assert out["startup_s"] == [26.5, 26.5] and out["card_w"] == 600 and out["accept_per_pos"] == [0.9, 0.8]
    assert out["gpu"] == "GPU" and out["model"] == "org/model" and out["workload"] == run()["workload"]
    (tmp_path / "reference.json").write_text(json.dumps(out))
    (ref,) = hostcheck.load_reference(tmp_path / "reference.json")
    assert ref["loads"][2.5] == {"e2e": (640.0, 880.0), "ttft": (77.0, 108.0), "reply": 97.0, "tpv": 4.35,
                                 "power": 316.0, "sm": 2421}
    assert hostcheck.matching_reference([ref], run(gpu="GPU", card_w=600)) is ref  # the same host, model and workload


# -- the mounted mode's folder -----------------------------------------------------------------------------------------

def folder(tmp_path: Path, frames: int) -> Path:
    (tmp_path / "frames").mkdir(parents=True)
    for i in range(frames):
        buf = io.BytesIO()
        synthetic.pattern(*synthetic.image_size(i), seed=i).save(buf, "JPEG")
        (tmp_path / "frames" / f"f{i:04d}.jpg").write_bytes(buf.getvalue())
    (tmp_path / "prompt.md").write_text("a prompt")
    (tmp_path / "schema.json").write_text(json.dumps(synthetic.scene_schema()))
    return tmp_path


def test_the_mounted_mode_reads_a_folder(tmp_path) -> None:
    inp = hostcheck.mounted_inputs(folder(tmp_path, 20))
    assert inp.mode == "mounted" and inp.count == 20 and inp.prompt == "a prompt" and inp.schema_name == "scene"
    assert inp.user_text == hostcheck.DEFAULT_USER_TEXT and inp.reference == []  # neither file: the defaults
    assert inp.summary == {"frames": 20, "sizes": {"1280x720": 15, "720x1280": 5}, "user_text": "default",
                           "reference": "none"}
    assert inp.shape(1) == (1280, 720) and inp.frame(0) == (tmp_path / "frames" / "f0000.jpg").read_bytes()
    (tmp_path / "user.txt").write_text("Count the objects.\n")
    (tmp_path / "reference.json").write_text(json.dumps([REFERENCE, {**REFERENCE, "card_w": 250}]))
    inp = hostcheck.mounted_inputs(tmp_path)
    assert inp.user_text == "Count the objects." and [r["card_w"] for r in inp.reference] == [450, 250]
    assert inp.reference[0]["loads"][2.5]["e2e"] == (640, 880) and inp.summary["reference"] == "reference.json"
    (tmp_path / "reference.json").write_text(json.dumps({"loads": {"2.5": {"e2e": [1, 2]}}}))
    with pytest.raises(ValueError, match=r"reference\.json, entry 0, rate 2\.5: no ttft, reply, tpv, power, sm"):
        hostcheck.mounted_inputs(tmp_path)
    (tmp_path / "schema.json").unlink()
    with pytest.raises(ValueError, match=r"no schema\.json"):
        hostcheck.mounted_inputs(tmp_path)


def test_the_synthetic_workload_written_as_a_mounted_folder(tmp_path) -> None:
    synthetic.write_inputs(tmp_path, 20, lambda t: len(t.split()))
    inp = hostcheck.mounted_inputs(tmp_path)
    assert inp.count == 20 and inp.user_text == synthetic.USER_TEXT and inp.schema == synthetic.scene_schema()
    assert inp.schema_name == "scene" and inp.frame(3) == synthetic.image_jpeg(3)
    assert inp.summary["sizes"] == {"1280x720": 15, "720x1280": 5}
    synthetic.write_inputs(tmp_path / "half", 20, lambda t: len(t.split()), prompt_tokens=300, tall_share=0.5)
    half = hostcheck.mounted_inputs(tmp_path / "half")
    assert half.summary["sizes"] == {"1280x720": 10, "720x1280": 10} and 300 <= len(half.prompt.split()) < 360


def test_too_few_images_stop_the_check_before_anything_runs(tmp_path, capsys) -> None:
    assert hostcheck.main(["--inputs", str(folder(tmp_path / "in", 30)), "--out", str(tmp_path / "r.json"),
                           "--config", str(tmp_path / "absent.yaml")]) == 2
    assert "30 images, and the loads need" in capsys.readouterr().out
    assert not (tmp_path / "r.json").exists()


# -- end to end on the fake runner -------------------------------------------------------------------------------------

@pytest.mark.serial  # a Gemma server with an image pool
@needs_gemma
def test_the_gemma_load_end_to_end_on_the_fake_runner(tmp_path, monkeypatch, capsys) -> None:
    """Gemma 4's configuration and tokenizer, the image's settings (guidance, compact JSON) with one image a request at
    the model's own budget, the fake runner drafting as the assistant at k = 5: both loads run, short, in synthetic
    mode, with no reference to match (no GPU, so no power class): raw numbers, no verdict against a reference. Replies
    stop at 64 tokens and the seeds are fixed: the fake samples anywhere its grammar allows, and documents of 400-670
    tokens take 1.4-3.6 s on a busy host, which fills the 8 slots at 3 requests a second and drops arrivals."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(hostcheck, "cpubench", lambda: {"python_loop_s": 0.2, "import_s": 2.0, "launch_us": 4.0})
    (tmp_path / "assistant").mkdir()
    (tmp_path / "assistant" / "config.json").write_text(json.dumps(
        {"model_type": "gemma4_assistant", "text_config": {"model_type": "gemma4_text"}}))
    cfg = tmp_path / "config.yaml"
    cfg.write_text(json.dumps({
        "model": {"path": str(GEMMA), "max_model_len": 4096}, "runner": {"kind": "fake"},
        "dev": {"runner": {"fake": {"step_ms": 0.5}}}, "scheduler": {"seed": 0},
        "images": {"max_per_request": 1, "workers": 2},
        "structured_output": {"backend": "guidance", "disable_any_whitespace": True},
        "speculative": {"model": str(tmp_path / "assistant")}}))
    code = hostcheck.main(["--config", str(cfg), "--port", str(free_port()), "--starts", "1", "--out",
                           str(tmp_path / "report.json"), "--rates", "3,6", "--slots", "8,16", "--load-warmup", "1",
                           "--duration", "4", "--tail", "1", "--max-tokens", "64"])
    r = json.loads((tmp_path / "report.json").read_text())
    assert r["family"] == "gemma4" and r["inputs"]["mode"] == "synthetic" and r["served_model"] == str(GEMMA)
    assert r["inputs"]["schema_name"] == "scene" and r["reference"]["values"] is None
    run_ = r["reference"]["run"]
    assert run_["model"] == str(GEMMA) and run_["k"] == 5 and run_["workload"] == {
        "mode": "synthetic", "version": synthetic.WORKLOAD_VERSION, "prompt_tokens": synthetic.PROMPT_TOKENS,
        "tall_share": synthetic.TALL_SHARE,
        "max_soft_tokens": 280}
    assert set(r["as_reference"]["loads"]) == {"3", "6"}
    assert r["drafter"]["ok"] and r["drafter"]["k"] == 5, r["drafter"]
    for ld in r["loads"]:
        assert ld["failed"] == 0 and ld["dropped"] == 0 and ld["e2e_ms"]["n"] >= ld["measured"] - 1 > 0
        assert ld["ttft_ms"]["n"] == ld["e2e_ms"]["n"]  # every request's TTFT from the server's log
        assert ld["wide"]["e2e_ms"]["n"] + ld["tall"]["e2e_ms"]["n"] == ld["e2e_ms"]["n"]
        assert ld["engine"]["accept"]["drafts"] > 0 and len(ld["engine"]["accept"]["per_pos"]) == 5
        assert ld["engine"]["plans_kept_since_start"] == 0 and ld["verdicts"]["plans_kept"] == "ok"
        assert ld["verdicts"]["e2e"] == ld["verdicts"]["accept"] == "no reference"
        # each answer's digest by its image (16 hex digits), the load's images only: never an answer's text
        assert len(ld["answers"]) >= ld["e2e_ms"]["n"] and all(re.fullmatch("[0-9a-f]{16}", d)
                                                               for d in ld["answers"].values())
    a, b = (set(map(int, ld["answers"])) for ld in r["loads"])
    assert not a & b and max(a) < min(b)  # the second load's images follow the first's
    assert code == 0
    out = capsys.readouterr().out
    assert "these latencies are this workload's" in out and "the drafter" in out
    assert "No reference matches this run" in out


# -- the server's stop and a check that raises -------------------------------------------------------------------------

class StubbornChild:
    """A child that ignores SIGINT: the first wait times out, the kill ends it, the wait after it reaps it."""

    def __init__(self) -> None:
        self.returncode = None
        self.calls: list[str] = []

    def poll(self):
        return self.returncode

    def send_signal(self, sig) -> None:
        self.calls.append(f"signal {sig}")

    def kill(self) -> None:
        self.calls.append("kill")
        self.returncode = -9

    def wait(self, timeout):
        self.calls.append(f"wait {timeout}")
        if self.returncode is None:
            import subprocess

            raise subprocess.TimeoutExpired("server", timeout)
        return self.returncode


def test_stop_kills_a_child_that_ignores_sigint_waits_for_it_and_closes_the_log(tmp_path) -> None:
    srv = object.__new__(hostcheck.Server)
    srv.proc, srv.log = StubbornChild(), (tmp_path / "server.log").open("w")
    srv.stop()
    assert srv.proc.calls == ["signal 2", "wait 120", "kill", "wait 10"] and srv.proc.returncode == -9
    assert srv.log.closed
    srv.stop()  # a second stop signals nothing more
    assert srv.proc.calls == ["signal 2", "wait 120", "kill", "wait 10"]


def test_a_server_that_exited_during_startup_is_reaped_and_its_log_closed_by_stop(tmp_path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"model:\n  path: {tmp_path / 'no-checkpoint'}\nrunner:\n  kind: fake\n")
    srv = hostcheck.Server(str(cfg), free_port(), tmp_path / "server.log", [])
    with pytest.raises(RuntimeError, match="exited during startup"):
        srv.wait_ready(60)
    srv.stop()
    assert srv.proc.returncode == 2 and srv.log.closed


def gemma_check(tmp_path, monkeypatch) -> tuple[list[str], list[hostcheck.Server]]:
    """`hostcheck.main`'s arguments on the fake runner with Gemma 4's configuration, and the servers it starts."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(hostcheck, "cpubench", lambda: {"python_loop_s": 0.2, "import_s": 2.0, "launch_us": 4.0})
    servers: list[hostcheck.Server] = []

    class Recorded(hostcheck.Server):
        def __init__(self, *a, **kw) -> None:
            super().__init__(*a, **kw)
            servers.append(self)

    monkeypatch.setattr(hostcheck, "Server", Recorded)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(json.dumps({
        "model": {"path": str(GEMMA), "max_model_len": 4096}, "runner": {"kind": "fake"},
        "dev": {"runner": {"fake": {"step_ms": 0.5}}}, "images": {"max_per_request": 1, "workers": 2},
        "structured_output": {"backend": "guidance", "disable_any_whitespace": True}}))
    return ["--config", str(cfg), "--port", str(free_port()), "--starts", "1", "--out", str(tmp_path / "report.json"),
            "--rates", "3", "--slots", "8", "--load-warmup", "1", "--duration", "2", "--tail", "1"], servers


@pytest.mark.serial  # a Gemma server with an image pool
@needs_gemma
def test_a_load_that_raises_leaves_a_partial_report_and_no_server(tmp_path, monkeypatch, capsys) -> None:
    argv, servers = gemma_check(tmp_path, monkeypatch)

    def boom(*a, **kw):
        raise ValueError("the load broke")

    monkeypatch.setattr(hostcheck, "run_gemma", boom)
    assert hostcheck.main(argv) == 1
    r = json.loads((tmp_path / "report.json").read_text())
    assert r["error"] == "ValueError: the load broke" and r["family"] == "gemma4"
    assert len(r["startup_s"]) == 1 and len(r["cpubench"]) == 3 and "loads" not in r
    (srv,) = servers
    assert srv.proc.returncode is not None and srv.log.closed
    out = capsys.readouterr().out
    assert "ValueError: the load broke" in out and f"report: {tmp_path / 'report.json'}" in out


@pytest.mark.serial  # a Gemma server with an image pool
@needs_gemma
def test_an_exception_in_the_start_loop_stops_the_server(tmp_path, monkeypatch) -> None:
    argv, servers = gemma_check(tmp_path, monkeypatch)

    def boom(self, timeout=900):
        raise KeyError("a reply the check did not expect")

    monkeypatch.setattr(hostcheck.Server, "wait_ready", boom)
    assert hostcheck.main(argv) == 1
    assert "KeyError" in json.loads((tmp_path / "report.json").read_text())["error"]
    (srv,) = servers
    assert srv.proc.returncode is not None and srv.log.closed


@pytest.mark.serial  # a Gemma server with an image pool
@needs_gemma
def test_a_startup_failure_also_writes_the_report_and_stops_the_server(tmp_path, monkeypatch) -> None:
    argv, servers = gemma_check(tmp_path, monkeypatch)

    def exited(self, timeout=900):
        raise RuntimeError("the server exited during startup (code 2)\nits log")

    monkeypatch.setattr(hostcheck.Server, "wait_ready", exited)
    assert hostcheck.main(argv) == 1
    r = json.loads((tmp_path / "report.json").read_text())
    assert r["error"] == "start 1: the server exited during startup (code 2)"
    (srv,) = servers
    assert srv.proc.returncode is not None and srv.log.closed


def test_the_table_tolerates_a_report_cut_short(tmp_path, monkeypatch, capsys) -> None:
    """A check that raised after a load, before its verdicts, or before the host speed: the table prints what is
    there."""
    load = hostcheck.summarize_load([], [], {}, 0.0, 1.0, {}, 3.0, 8, 1.0)
    report = {"gpu": {}, "host": {"cpu": "cpu", "cpu_quota": 4.0}, "family": "gemma4", "loads": [load],
              "inputs": {"mode": "synthetic", "frames": "generated", "sizes": {}, "schema_name": "scene",
                         "max_tokens": 64}, "error": "ValueError: x"}
    hostcheck.print_table(report)
    out = capsys.readouterr().out
    assert "error" in out and "3/s, 8 slots: requests" in out

    def boom(*a, **kw):
        raise ValueError("a load broke")

    monkeypatch.setattr(hostcheck, "gpu_info", lambda: {})
    monkeypatch.setattr(hostcheck, "cpubench", boom)
    assert hostcheck.main(["--no-server", "--out", str(tmp_path / "r.json")]) == 1
    r = json.loads((tmp_path / "r.json").read_text())
    assert r["error"] == "ValueError: a load broke" and "cpubench" not in r
    assert "report:" in capsys.readouterr().out
