"""The tool-calling recorder (`tests.scripts.record_tool_calls`) on the CPU: two cases against the fake runner, through
the image's configuration file, so its mechanics are checked before any GPU host and in CI."""

from __future__ import annotations

import json
import sys

from ..scripts import record_tool_calls as rec
from ..support.fixtures import require_qwen


def test_the_recorder_writes_both_forms_of_each_case(tmp_path) -> None:
    model = require_qwen()
    assert rec.main(["--model", str(model), "--runner", "fake", "--cases", "weather,control_weather",
                     "--out-dir", str(tmp_path), "--set", "model.max_model_len=4096",
                     "--set", "model.served_model_name=qwen-test", "--start-timeout", "120"]) == 0
    [path] = tmp_path.glob("*.json")
    assert path.name == "qwen3.8-27b-guidance.json"
    data = json.loads(path.read_text())
    assert data["family"] == "qwen" and data["server_info"]["config"]["structured_output"]["backend"] == "guidance"
    assert data["server_info"]["config"]["cache"]["enable_prefix_caching"] is False
    results = data["results"]
    assert [(r["case"], r["stream"]) for r in results] == [("weather", False), ("weather", True),
                                                            ("control_weather", False), ("control_weather", True)]
    for r in results:
        assert r["status"] == 200 and r["request"]["temperature"] == 0 and r["request"]["return_token_ids"]
        assert r["t_end_ns"] >= r["t_first_ns"] > 0
        steps = sum(v for k, v in r["metrics"].items() if k.startswith("vllm:iteration_tokens_total_count{"))
        assert steps >= 1, r["metrics"]
        if r["stream"]:
            assert r["events"][-1] == "data: [DONE]"
            assert "prompt_token_ids" in json.loads(r["events"][0].removeprefix("data: "))
        else:
            body = json.loads(r["body"])
            assert body["prompt_token_ids"] and body["choices"][0]["token_ids"]
    assert ("tools" in results[0]["request"]) and ("tools" not in results[2]["request"])
    assert (tmp_path / "qwen3.8-27b-guidance.log").stat().st_size  # the server's own log, beside the JSON


def test_the_server_gets_no_token(tmp_path, monkeypatch) -> None:
    """The server the recorder starts gets neither Hugging Face token variable (stand-in values here; a stub command
    prints which of them it sees)."""
    monkeypatch.setenv("HF_TOKEN", "stand-in")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "stand-in")
    names = "('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN')"
    stub = [sys.executable, "-c", f"import os; print([k for k in {names} if k in os.environ])"]
    proc = rec.ServerProcess(stub, tmp_path / "stub.log", 0)
    proc.start()
    assert proc.proc.wait(30) == 0
    proc.stop()
    assert (tmp_path / "stub.log").read_text().strip() == "[]"


def test_the_family_comes_from_the_checkpoint_not_its_path(tmp_path) -> None:
    qwen = tmp_path / "llm-mini-infer-gemma-work" / "model"
    qwen.mkdir(parents=True)
    (qwen / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    assert rec.family(str(qwen)) == "qwen"
    assert rec.family("RedHatAI/gemma-4-31B-it-FP8-block") == "gemma"
    assert rec.family("Qwen/Qwen3.8-27B-FP8") == "qwen"
