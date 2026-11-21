"""The deploy contract: an example values file (tests/data/chart-values.yaml), rendered by the chart, give a
configuration the container's preflight passes and that derives what the values ask for: the served name, the drafter
at llm-mini-infer's own k, the image budget, the images a request, the context and with it batch invariance, the rows,
the seed, the memory share, the keep-alive and the grammar, with the pixel arena inside the chart's /dev/shm. The
image's own configuration (deploy/config.yaml) plus the host check Job's --set, and the chart's defaults with the same
model, both derive the image's: the model's own context (which leaves batch invariance off) and image budget, the
engine's defaults, the image's seed and grammar.

The model is Gemma 4 31B's own configuration (tests/data: the FP8 checkpoint's config.json and generation_config.json,
google/gemma-4-31B-it's processor_config.json and tokenizer_config.json, the same bytes in the FP8 repository), a
placeholder tokenizer.json (derive checks only that it exists) and an assistant's config.json. hub.download is
stubbed: nothing is downloaded, and every model repository id resolves to these files, so a wrong id shows as a wrong
served name. Values written for another model fail the contract both ways: Qwen's pixel bounds make the preflight
refuse Gemma 4 (the container exits 2), and without them the served name, the drafter, the image budget and the
grammar are not the ones asked for.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from llm_mini_infer import config, entrypoint, hub

from ..support.helm import HELM, HELM_BIN, VALUES, kind, render
from ..support.paths import DATA, ROOT

MODEL = "RedHatAI/gemma-4-31B-it-FP8-block"  # served under its repository id
ASSISTANT = "google/gemma-4-31B-it-assistant"
helm = pytest.mark.skipif(HELM_BIN is None, reason="no helm in tools/bin: uv run tools/fetch_tools.py helm")


@pytest.fixture
def checkpoints(tmp_path, monkeypatch) -> None:
    target, assistant = tmp_path / "target", tmp_path / "assistant"
    target.mkdir()
    assistant.mkdir()
    for src, dst in (("gemma-4-31b-it-fp8-block.config.json", "config.json"),
                     ("gemma-4-31b-it.generation_config.json", "generation_config.json"),
                     ("gemma-4-31b-it.processor_config.json", "processor_config.json"),
                     ("gemma-4-31b-it.tokenizer_config.json", "tokenizer_config.json")):
        shutil.copy(DATA / src, target / dst)
    (target / "tokenizer.json").write_text("{}")
    (assistant / "config.json").write_text(json.dumps({"architectures": ["Gemma4AssistantForCausalLM"],
                                                       "model_type": "gemma4_assistant"}))
    for var in entrypoint.CACHES:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(hub, "_resolved", {})
    monkeypatch.setattr(hub, "cache_dir", lambda: tmp_path / "hf")
    monkeypatch.setattr(hub, "download", lambda repo, revision: str(assistant if repo == ASSISTANT else target))


def default_k() -> int:
    """llm-mini-infer's own k with an assistant: what derive gives the model and its assistant when no k is set."""
    cfg = config.load(None, {"model.path": MODEL, "speculative.model": ASSISTANT})
    hub.resolve(cfg)
    return config.derive(cfg).speculative.num_speculative_tokens


def example_settings() -> dict:
    """What the example values ask for (tests/data/chart-values.yaml), read from them. Batch invariance is on
    at any context for Gemma 4 (the example's model)."""
    values = yaml.safe_load(VALUES.read_text())
    cfg = values["config"]
    so = cfg["structured_output"]
    return {"served name": values["model"]["id"], "drafter": cfg["speculative"]["model"],
            "image tokens": cfg["images"]["max_soft_tokens"], "images per request": cfg["images"]["max_per_request"],
            "context": cfg["model"]["max_model_len"],
            "batch invariant": True,
            "rows": cfg["scheduler"]["max_num_seqs"], "seed": cfg["scheduler"]["seed"],
            "runner": (cfg["runner"]["kind"], cfg["runner"]["cuda"]["memory_utilization"]),
            "keep-alive": cfg["server"]["keep_alive_s"],
            "grammar": (so["backend"], so["disable_any_whitespace"])}


EXAMPLE = example_settings()
IMAGE_CONFIG = yaml.safe_load((ROOT / "deploy" / "config.yaml").read_text())
# What the image serves Gemma 4 31B FP8 with: the model's own context (262,144 positions) and
# image budget (280 soft tokens, processor_config.json's), the engine's defaults (16 images, 64 rows, a memory share
# of 0.9, keep-alive 5 s), and the seed and grammar of deploy/config.yaml
IMAGE = {**EXAMPLE, "image tokens": 280, "images per request": 16, "context": 262_144, "batch invariant": True,
         "rows": 64, "runner": (IMAGE_CONFIG["runner"]["kind"], 0.9), "keep-alive": 5,
         "seed": IMAGE_CONFIG["scheduler"]["seed"],
         "grammar": (IMAGE_CONFIG["structured_output"]["backend"],
                     IMAGE_CONFIG["structured_output"]["disable_any_whitespace"])}


def violations(path: Path, extra: list[str] | None = None, shm_mib: int | None = None,
               want: dict | None = None) -> list[str]:
    """The preflight's problems (the container exits 2 on any), else every setting that differs from `want` (default:
    what the example values ask for), and k: set at all, or effectively other than llm-mini-infer's default with an
    assistant. The host's own /dev/shm is not the deploy's: `shm_mib` is the chart's."""
    extra = extra or []
    problems = [p for p in entrypoint.preflight(str(path), extra) if not p.startswith("/dev/shm")]
    if problems:
        return problems
    cfg = entrypoint.load(str(path), extra)
    k_set = cfg.speculative.num_speculative_tokens  # None: left to llm-mini-infer
    hub.resolve(cfg)
    config.derive(cfg)
    so = cfg.structured_output
    got = {
        "served name": cfg.model.served_model_name,
        "drafter": cfg.speculative_source.split("@")[0],
        "image tokens": cfg.image_tokens_max,
        "images per request": cfg.images.max_per_request,
        "context": cfg.model.max_model_len,
        "batch invariant": cfg.dev.runner.cuda.batch_invariant,
        "rows": cfg.scheduler.max_num_seqs,
        "seed": cfg.scheduler.seed,
        "runner": (cfg.runner.kind, cfg.runner.cuda.memory_utilization),
        "keep-alive": cfg.server.keep_alive_s,
        "grammar": (so.backend, so.disable_any_whitespace),
    }
    checks = {name: (got[name], w) for name, w in (want or EXAMPLE).items()}
    if shm_mib is not None:
        checks["pixel arena in /dev/shm"] = (cfg.dev.images.arena_mb <= shm_mib, True)
    problems = [f"{name}: {got!r}, wanted {want!r}" for name, (got, want) in checks.items() if got != want]
    k = cfg.speculative.num_speculative_tokens
    if k_set is not None:
        problems.append(f"k: {k_set} set, where llm-mini-infer's own default belongs")
    elif k != (own := default_k()):
        problems.append(f"MTP drafts: {k}, llm-mini-infer's own {own}")
    return problems


def mib(size: str) -> int:
    for unit, scale in (("Gi", 1024), ("Mi", 1)):
        if size.endswith(unit):
            return int(size.removesuffix(unit)) * scale
    raise ValueError(f"a /dev/shm size in Gi or Mi, got {size!r}")


def chart(values: Path, tmp: Path) -> tuple[Path, int]:
    """The configuration file llm-mini-infer's chart renders from `values`, and the MiB of its /dev/shm."""
    docs = render("llm-mini-infer", values)
    path = tmp / "config.yaml"
    path.write_text(kind(docs, "ConfigMap")[0]["data"]["config.yaml"])
    vols = {v["name"]: v for v in kind(docs, "Deployment")[0]["spec"]["template"]["spec"]["volumes"]}
    return path, mib(vols["shm"]["emptyDir"]["sizeLimit"])


@helm
def test_the_values_serve_what_they_ask_for(checkpoints, tmp_path) -> None:
    path, shm = chart(VALUES, tmp_path)
    assert violations(path, shm_mib=shm) == []


def test_the_host_check_job_runs_the_images_configuration(checkpoints) -> None:
    """The image's configuration (deploy/config.yaml) plus the Job's --set (the model and its assistant)."""
    pod = yaml.safe_load((HELM / "examples" / "hostcheck-job.yaml").read_text())["spec"]["template"]["spec"]
    args = pod["containers"][0]["args"]
    vols = {v["name"]: v for v in pod["volumes"]}
    assert args[0] == "hostcheck"
    shm = mib(vols["shm"]["emptyDir"]["sizeLimit"])
    assert violations(ROOT / "deploy" / "config.yaml", args[1:], shm, want=IMAGE) == []


@helm
def test_the_charts_defaults_serve_as_the_image(checkpoints, tmp_path) -> None:
    """The chart's configuration file replaces the image's, so its defaults repeat the image's settings: installed with
    a model and its assistant alone, the chart serves as the image does."""
    docs = render("llm-mini-infer", None, f"model.id={MODEL}", f"config.speculative.model={ASSISTANT}")
    path = tmp_path / "config.yaml"
    path.write_text(kind(docs, "ConfigMap")[0]["data"]["config.yaml"])
    vols = {v["name"]: v for v in kind(docs, "Deployment")[0]["spec"]["template"]["spec"]["volumes"]}
    assert violations(path, shm_mib=mib(vols["shm"]["emptyDir"]["sizeLimit"]), want=IMAGE) == []


@helm
def test_a_k_in_the_values_fails_the_contract(checkpoints, tmp_path) -> None:
    """The values pass no k, so the assistant drafts llm-mini-infer's own default; values that pass one fail."""
    values = yaml.safe_load(VALUES.read_text())
    assert "num_speculative_tokens" not in values["config"]["speculative"]
    assert default_k() > 0  # the assistant drafts
    values["config"]["speculative"]["num_speculative_tokens"] = 3
    k3 = tmp_path / "values-k3.yaml"
    k3.write_text(yaml.safe_dump(values))
    path, shm = chart(k3, tmp_path)
    assert violations(path, shm_mib=shm) == ["k: 3 set, where llm-mini-infer's own default belongs"]


@helm
def test_values_for_another_model_fail_the_contract(checkpoints, tmp_path) -> None:
    """Values written for Qwen3.8 27B (its repository, its pixel bounds, no drafter, no grammar settings of their own)
    on Gemma 4: the preflight refuses the pixel bounds (the container would exit 2 and restart forever); without them,
    every request would name a model the server does not serve (a 404), with no MTP and the image budget missing. The
    grammar is the chart's default, the image's."""
    values = yaml.safe_load(VALUES.read_text())
    values["model"]["id"] = "Qwen/Qwen3.8-27B-FP8"
    del values["config"]["speculative"], values["config"]["structured_output"]
    values["config"]["images"] = {"min_pixels": 65536, "max_pixels": 327680, "max_per_request": 1}
    other = tmp_path / "values-other.yaml"
    other.write_text(yaml.safe_dump(values))
    path, shm = chart(other, tmp_path)
    (problem,) = violations(path, shm_mib=shm, want={**EXAMPLE, "grammar": IMAGE["grammar"]})
    assert "images.min_pixels" in problem, problem
    for key in ("min_pixels", "max_pixels"):
        del values["config"]["images"][key]
    other.write_text(yaml.safe_dump(values))
    path, shm = chart(other, tmp_path)
    found = {v.split(":")[0] for v in violations(path, shm_mib=shm, want={**EXAMPLE, "grammar": IMAGE["grammar"]})}
    assert {"served name", "MTP drafts", "drafter", "image tokens"} <= found and "grammar" not in found, found
