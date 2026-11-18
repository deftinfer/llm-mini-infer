"""Settings of the configuration (architecture.md, "Configuration"): derived defaults (the MTP drafter's head, Gemma 4's
assistant and image budget, the structured-output backend, the grammar-aware heads' and the kernels' switches, the
reasoning parser and its effort map);
runner.kind, the one never derived; the operator's settings, the dev group and the server's state kept apart; and
deleted settings, refused at load."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from llm_mini_infer import config

from ..support.config_cases import _assistant, _gemma, _model
from ..support.paths import DATA, MODELS_DIR

MAXL = config.BATCH_INVARIANT_MAX_LEN - 8  # batch invariance's default range of model.max_model_len


def test_draft_head_defaults(tmp_path) -> None:
    cfg = config.derive(config.load(None, {"model.path": str(_model(tmp_path, 248_320, 1))}))
    assert cfg.speculative.num_speculative_tokens == 3
    assert cfg.dev.speculative.draft_vocab == config.DRAFT_VOCAB and cfg.dev.speculative.draft_head_dtype == "bfloat16"
    assert "ranked from the tokens this server emits" in cfg.reasons["dev.speculative.draft_vocab"]
    small = tmp_path / "small"
    small.mkdir()
    cfg = config.derive(config.load(None, {"model.path": str(_model(small, 65_536, 1))}))
    assert cfg.dev.speculative.draft_vocab == 0  # under 4 x 32,768: the whole vocabulary
    cfg = config.derive(config.load(None, {"model.path": str(_model(small, 248_320, 0))}))
    assert cfg.dev.speculative.draft_vocab == 0  # no MTP
    cfg = config.derive(config.load(None, {"model.path": str(_model(small, 248_320, 1)),
                                           "dev.speculative.draft_vocab": 0}))
    assert cfg.dev.speculative.draft_vocab == 0  # an override wins


def test_a_drafter_model_turns_mtp_on(tmp_path) -> None:
    """Gemma 4's MTP drafter is its assistant model (`speculative.model`): k = 5 by default with one (its drafts keep
    paying deep into the chain), 0 without, and a set k stays as set; any other pairing is refused at derive time."""
    import pytest

    target, asst = _gemma(tmp_path / "g"), _assistant(tmp_path / "a")
    cfg = config.derive(config.load(None, {"model.path": str(target), "speculative.model": str(asst)}))
    why = cfg.reasons["speculative.num_speculative_tokens"]
    assert cfg.speculative.num_speculative_tokens == 5 and "assistant" in why and "deep into the chain" in why
    assert config.derive(config.load(None, {"model.path": str(target), "speculative.model": str(asst),
                                            "speculative.num_speculative_tokens": 3})).speculative \
        .num_speculative_tokens == 3  # a set k stays
    assert cfg.dev.speculative.draft_vocab == 0  # the whole head
    assert "assistant" in cfg.reasons["dev.speculative.draft_vocab"]
    cfg = config.derive(config.load(None, {"model.path": str(target)}))
    assert cfg.speculative.num_speculative_tokens == 0
    with pytest.raises(ValueError, match="needs an MTP layer"):
        config.derive(config.load(None, {"model.path": str(target), "speculative.num_speculative_tokens": 3}))
    with pytest.raises(ValueError, match="only Gemma 4's assistant"):
        config.derive(config.load(None, {"model.path": str(_model(tmp_path, 248_320, 1)),
                                         "speculative.model": str(asst)}))
    with pytest.raises(ValueError, match=r"has no config\.json"):
        config.derive(config.load(None, {"model.path": str(target), "speculative.model": str(tmp_path / "none")}))


def test_the_drafters_largest_k(tmp_path) -> None:
    """`mtp_k_max`, which the derived refine piece leaves room for, so that k never changes the precise copy: the
    assistant's largest k whatever k is set (7, the verify window's), never below the set k; Qwen's MTP layer the set
    k, from which its piece is derived; 0 without a drafter; one definition of each cap, the runners' own."""
    import pytest

    from llm_mini_infer.runner import mtp

    target, asst = _gemma(tmp_path / "g"), _assistant(tmp_path / "a")
    (tmp_path / "q").mkdir()
    qwen = _model(tmp_path / "q", 248_320, 1)

    def derive(model: Path, **kw) -> config.Config:
        return config.derive(config.load(None, {"model.path": str(model), **kw}))

    for k in (None, 0, 3, 5, 7):
        cfg = derive(target, **{"speculative.model": str(asst)},
                     **({} if k is None else {"speculative.num_speculative_tokens": k}))
        assert cfg.mtp_k_max == mtp.MAX_GEMMA_MTP_K == mtp.MAX_MTP_K == 7
        assert cfg.reasons["mtp_k_max"].startswith("7: the assistant model's largest k, whatever k is set")
    for k, want in ((None, 3), (0, 0), (2, 2), (3, 3), (5, 5)):
        cfg = derive(qwen, **({} if k is None else {"speculative.num_speculative_tokens": k}))
        assert cfg.mtp_k_max == want and cfg.reasons["mtp_k_max"].startswith(f"{want}: the set k: the checkpoint's MTP")
    cfg = derive(target)
    assert cfg.mtp_k_max == 0 and cfg.reasons["mtp_k_max"].startswith("0: no drafter")
    with pytest.raises(ValueError, match=r"must be 0-7 \(the verify window is 1 \+ k <= 8 tokens\)"):
        derive(target, **{"speculative.model": str(asst), "speculative.num_speculative_tokens": 8})
    from llm_mini_infer.runner.cuda import gemma, runner

    assert (gemma.MAX_GEMMA_MTP_K, runner.MAX_CUDA_MTP_K) == (mtp.MAX_GEMMA_MTP_K, mtp.MAX_CUDA_MTP_K)


def test_a_step_budget_or_slot_count_the_scheduler_cannot_use_is_refused(tmp_path) -> None:
    """A token budget below a decode row (1 + k) or fewer than 1 slot: the scheduler would never plan a step, and the
    server hung. Each is refused, naming its setting; the smallest that works starts."""
    import pytest

    target, asst = _gemma(tmp_path / "g"), _assistant(tmp_path / "a")

    def derive(**kw) -> config.Config:
        return config.derive(config.load(None, {"model.path": str(target), "speculative.model": str(asst),
                                                "speculative.num_speculative_tokens": 3, **kw}))

    for budget in (0, 3, -5):
        with pytest.raises(ValueError, match=rf"^scheduler.max_num_batched_tokens {budget}: at least 1 \+ speculative"
                                             r".num_speculative_tokens = 4"):
            derive(**{"scheduler.max_num_batched_tokens": budget})
    for seqs in (0, -1):
        with pytest.raises(ValueError, match=rf"^scheduler.max_num_seqs {seqs}: at least 1$"):
            derive(**{"scheduler.max_num_seqs": seqs})
    cfg = derive(**{"scheduler.max_num_batched_tokens": 4, "scheduler.max_num_seqs": 1})
    assert (cfg.scheduler.max_num_batched_tokens, cfg.scheduler.max_num_seqs) == (4, 1)


def test_a_compile_timeout_of_zero_or_less_is_refused(tmp_path) -> None:
    """`dev.structured_output.compile_timeout_s` is more than 0 and finite (NaN and infinity refused), 60 s by
    default."""
    import pytest

    target = _gemma(tmp_path / "g")

    def derive(**kw) -> config.Config:
        return config.derive(config.load(None, {"model.path": str(target), **kw}))

    assert derive().dev.structured_output.compile_timeout_s == 60.0
    for bad in (0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match=r"^dev.structured_output.compile_timeout_s .*: more than 0, and finite$"):
            derive(**{"dev.structured_output.compile_timeout_s": bad})
    assert derive(**{"dev.structured_output.compile_timeout_s": 0.5}).dev.structured_output.compile_timeout_s == 0.5


def test_the_grammar_heads_switches(tmp_path) -> None:
    """The grammar-aware heads' switches (`runner/cuda/heads.py`), alike on both models: the sparse head on with
    structured output, the draft universe with drafts too (Gemma 4's assistant, Qwen's MTP layer), off otherwise, each
    with its reason; 1,024 rows by default, their share of the model's vocabulary in the reason; overrides win; an
    explicit universe without drafts is refused."""
    import pytest

    target, asst = _gemma(tmp_path / "g"), _assistant(tmp_path / "a")
    (tmp_path / "q").mkdir()
    (tmp_path / "q0").mkdir()
    qwen, qwen0 = _model(tmp_path / "q", 248_320, 1), _model(tmp_path / "q0", 248_320, 0)
    so = {"structured_output.enabled": True}

    def derive(model: Path, **kw) -> config.Config:
        return config.derive(config.load(None, {"model.path": str(model), **kw}))

    cfg = derive(target, **so, **{"speculative.model": str(asst)})
    o = cfg.dev.runner.cuda
    assert o.sparse_head and cfg.dev.speculative.draft_universe
    assert o.sparse_head_rows == config.SPARSE_HEAD_ROWS == 1024
    assert "the union of its masks" in cfg.reasons["dev.runner.cuda.sparse_head"]
    assert "grammar's universe" in cfg.reasons["dev.speculative.draft_universe"]
    assert "0.39% of the 262,144-row head" in cfg.reasons["dev.runner.cuda.sparse_head_rows"]
    cfg = derive(target, **so)  # no assistant: no drafts to restrict
    assert cfg.dev.runner.cuda.sparse_head and not cfg.dev.speculative.draft_universe
    assert cfg.reasons["dev.speculative.draft_universe"].startswith("off: no drafts")
    cfg = derive(target, **{"speculative.model": str(asst), "structured_output.enabled": False})
    assert not cfg.dev.runner.cuda.sparse_head and not cfg.dev.speculative.draft_universe
    assert cfg.reasons["dev.runner.cuda.sparse_head"] == "off: structured output is off"
    cfg = derive(target, **so, **{"speculative.model": str(asst), "speculative.num_speculative_tokens": 0})
    assert cfg.dev.runner.cuda.sparse_head and not cfg.dev.speculative.draft_universe  # no drafts
    cfg = derive(target, **so, **{"speculative.model": str(asst), "dev.runner.cuda.sparse_head": False,
                                  "dev.speculative.draft_universe": False, "dev.runner.cuda.sparse_head_rows": 512})
    o = cfg.dev.runner.cuda
    assert (o.sparse_head, cfg.dev.speculative.draft_universe, o.sparse_head_rows) == (False, False, 512)
    cfg = derive(qwen, **so)  # its MTP layer drafts 3
    o = cfg.dev.runner.cuda
    assert o.sparse_head and cfg.dev.speculative.draft_universe and o.sparse_head_rows == 1024
    assert "the union of its masks" in cfg.reasons["dev.runner.cuda.sparse_head"]
    assert "(the MTP layer)" in cfg.reasons["dev.speculative.draft_universe"]
    assert cfg.reasons["dev.runner.cuda.sparse_head_rows"].startswith("1,024: 0.41% of the 248,320-row head")
    for model, kw in ((qwen, {"speculative.num_speculative_tokens": 0}), (qwen0, {})):  # no drafts
        cfg = derive(model, **so, **kw)
        assert cfg.dev.runner.cuda.sparse_head and not cfg.dev.speculative.draft_universe
        with pytest.raises(ValueError, match="draft_universe: needs drafts"):
            derive(model, **so, **kw, **{"dev.speculative.draft_universe": True})
    cfg = derive(qwen)
    assert not cfg.dev.runner.cuda.sparse_head and not cfg.dev.speculative.draft_universe
    assert cfg.reasons["dev.speculative.draft_universe"] == "off: structured output is off"
    with pytest.raises(ValueError, match="draft_universe: needs drafts"):
        derive(target, **so, **{"dev.speculative.draft_universe": True})  # no assistant
    # the drafter's queries at p + 1: Gemma 4's assistant only
    assert derive(target, **{"speculative.model": str(asst), "dev.speculative.draft_query_next": True})
    for model in (qwen, target):  # no assistant on either
        with pytest.raises(ValueError, match="draft_query_next: Gemma 4's assistant only"):
            derive(model, **{"dev.speculative.draft_query_next": True})
    with pytest.raises(ValueError, match="1-65,536 tokens"):
        derive(target, **{"dev.runner.cuda.sparse_head_rows": 0})


def test_structured_output_backend_settings(tmp_path) -> None:
    """vLLM's `--structured-outputs-config` keys. guidance ends a finished grammar on the tokenizer's eos_token,
    which must be named (checked before the weights load); `disable_any_whitespace` needs guidance (vLLM refuses
    it with auto, and xgrammar's compact form is not reproduced); vLLM's other backends and keys are unknown."""
    import pytest

    from ..support.fixtures import QWEN
    from ..support.paths import MODELS_DIR

    gemma = MODELS_DIR / "gemma-4-31b-it"
    if not (gemma / "tokenizer.json").exists():
        pytest.skip("tests/.fixtures/models/gemma-4-31b-it is absent: uv run tests/scripts/fetch_fixtures.py")
    base = {"model.path": str(gemma), "runner.kind": "fake"}
    cfg = config.derive(config.load(None, base))
    so = cfg.structured_output
    assert (so.backend, so.disable_any_whitespace) == ("auto", False) and cfg.eos_token_ids == [1, 50, 106]
    assert "the model's stop ids end a grammar" in cfg.reasons["structured_output.backend"]
    cfg = config.derive(config.load(None, {**base, "structured_output.backend": "guidance",
                                           "structured_output.disable_any_whitespace": True}))
    why = cfg.reasons["structured_output.backend"]
    assert "GuidanceBackend" in why and "compact" in why and "eos_token <eos>" in why
    assert cfg.grammar_eos_token_ids == []  # the engine fills it
    for backend in ("auto", "xgrammar"):
        with pytest.raises(ValueError, match="disable_any_whitespace"):
            config.derive(config.load(None, {**base, "structured_output.backend": backend,
                                             "structured_output.disable_any_whitespace": True}))
    for key, value in (("backend", "outlines"), ("disable_additional_properties", True)):
        with pytest.raises(ValueError, match="invalid configuration"):
            config.load(None, {**base, f"structured_output.{key}": value})
    plain = _model(tmp_path, 65_536, 0)
    (plain / "tokenizer_config.json").write_text(json.dumps({"bos_token": "<s>"}))
    guided = {"model.path": str(plain), "runner.kind": "fake", "structured_output.backend": "guidance"}
    with pytest.raises(ValueError, match="names no eos_token"):
        config.derive(config.load(None, {**guided, "structured_output.enabled": True}))
    off = config.derive(config.load(None, {**guided, "structured_output.enabled": False}))
    assert not off.structured_output.enabled  # nothing to end

    # transformers' rule (5.17): special_tokens_map.json's wins unless tokenizer_config.json has added_tokens_decoder
    def eos(tc: dict, stm: dict | None = None) -> str | None:
        (plain / "tokenizer_config.json").write_text(json.dumps(tc))
        (plain / "special_tokens_map.json").unlink(missing_ok=True)
        if stm is not None:
            (plain / "special_tokens_map.json").write_text(json.dumps(stm))
        return config.tokenizer_eos_token(config.load(None, {"model.path": str(plain)}))

    assert eos({"eos_token": "<a>"}) == "<a>" and eos({}, {"eos_token": "<b>"}) == "<b>"
    assert eos({"eos_token": "<a>"}, {"eos_token": {"content": "<b>", "lstrip": False}}) == "<b>"
    assert eos({"eos_token": "<a>", "added_tokens_decoder": {}}, {"eos_token": "<b>"}) == "<a>"
    assert eos({"eos_token": {"__type": "AddedToken", "content": "<a>", "special": True}}) == "<a>"
    assert eos({}) is None and eos({"eos_token": None}, {}) is None
    assert config.tokenizer_eos_token(config.load(None, {"model.path": str(plain),
                                                         "model.tokenizer": str(gemma)})) == "<eos>"
    names = {"<eos>": (gemma, 1), "<|im_end|>": (QWEN, 248_046)}
    for name, (path, _) in names.items():
        if (path / "tokenizer_config.json").exists():
            assert config.tokenizer_eos_token(config.load(None, {"model.path": str(path)})) == name
    transformers = pytest.importorskip("transformers")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("HF_HUB_OFFLINE", "1")
        mp.delenv("HF_TOKEN", raising=False)
        for name, (path, token_id) in names.items():
            if (path / "tokenizer.json").exists():
                tok = transformers.AutoTokenizer.from_pretrained(path, local_files_only=True)
                assert (tok.eos_token, tok.eos_token_id) == (name, token_id)


def test_the_schema_cap_is_a_setting(tmp_path) -> None:
    """structured_output.max_schema_bytes: an operator's setting, 1 MiB unless set (the frontend refuses a larger
    schema)."""
    path = str(_model(tmp_path, 65_536, 0))
    assert "structured_output.max_schema_bytes" in config.SETTINGS
    assert config.derive(config.load(None, {"model.path": path})).structured_output.max_schema_bytes == 1 << 20
    cfg = config.derive(config.load(None, {"model.path": path, "structured_output.max_schema_bytes": 65_536}))
    assert cfg.structured_output.max_schema_bytes == 65_536


def _gemma_checkpoint(tmp: Path) -> Path:
    """google/gemma-4-31B-it's config.json and processor_config.json (tests/data): images of at most 280 soft tokens."""
    tmp.mkdir(parents=True, exist_ok=True)
    shutil.copy(DATA / "gemma-4-31b-it.config.json", tmp / "config.json")
    shutil.copy(DATA / "gemma-4-31b-it.processor_config.json", tmp / "processor_config.json")
    return tmp


def test_images_max_soft_tokens_sizes_gemma_images(tmp_path) -> None:
    """Gemma 4's image budget is a setting (vLLM's `--mm-processor-kwargs` max_soft_tokens). Unset, it is the
    checkpoint's, with its source as the reason; set, it gives the pixel bounds (reasons naming it), the largest image
    and the encoder budget's images per step. Pixel bounds of their own on Gemma 4 and a budget outside HF's five are
    refused; with images off the key is unused."""
    import pytest

    path = str(_gemma_checkpoint(tmp_path / "g"))
    cfg = config.derive(config.load(None, {"model.path": path}))
    im = cfg.images
    assert (im.max_soft_tokens, cfg.image_tokens_max, im.max_pixels) == (280, 280, 645_120)
    assert "processor_config.json" in cfg.reasons["images.max_soft_tokens"]
    cfg = config.derive(config.load(None, {"model.path": path, "images.max_soft_tokens": 70}))
    im = cfg.images
    assert (im.max_soft_tokens, cfg.image_tokens_max, im.min_pixels, im.max_pixels) == (70, 70, 2_304, 161_280)
    for key in ("images.min_pixels", "images.max_pixels"):  # not preprocessor_config.json's: Gemma has none
        assert "images.max_soft_tokens" in cfg.reasons[key], key
    dim = cfg.dev.images
    assert dim.encoder_budget == 2048 and "29 images of 70 tokens" in cfg.reasons["dev.images.encoder_budget"]
    assert dim.arena_mb == 256
    with pytest.raises(ValueError, match=r"images\.max_soft_tokens"):
        config.derive(config.load(None, {"model.path": path, "images.min_pixels": 65_536}))
    with pytest.raises(ValueError, match="Invalid enum value 100"):
        config.load(None, {"model.path": path, "images.max_soft_tokens": 100})
    cfg = config.derive(config.load(None, {"model.path": path, "images.enabled": False, "images.max_soft_tokens": 70}))
    assert not cfg.images.enabled and cfg.image_tokens_max == 0


def _qwen_vl(tmp: Path) -> Path:
    tmp.mkdir(parents=True, exist_ok=True)
    tc = {"model_type": "qwen3_5_text", "vocab_size": 248_320, "max_position_embeddings": 4096, "eos_token_id": 1}
    vc = {"patch_size": 16, "spatial_merge_size": 2, "temporal_patch_size": 2}
    (tmp / "config.json").write_text(json.dumps({"model_type": "qwen3_5", "text_config": tc, "vision_config": vc}))
    return tmp


def test_a_pixel_sized_family_refuses_max_soft_tokens(tmp_path) -> None:
    """Qwen3.5/3.8 size images by pixels: images.max_soft_tokens on them is a configuration error (transformers'
    processors for them ignore it), and without it they derive as before."""
    import pytest

    path = str(_qwen_vl(tmp_path))
    with pytest.raises(ValueError, match="Gemma 4's image budget"):
        config.derive(config.load(None, {"model.path": path, "images.max_soft_tokens": 70}))
    cfg = config.derive(config.load(None, {"model.path": path, "images.max_pixels": 327_680}))
    im = cfg.images
    assert im.max_soft_tokens is None and "unused" in cfg.reasons["images.max_soft_tokens"]
    assert (im.min_pixels, im.max_pixels, cfg.image_tokens_max) == (65_536, 327_680, 320)
    assert cfg.reasons["images.min_pixels"] == "preprocessor_config.json size.shortest_edge"


def test_max_source_pixels_is_validated(tmp_path) -> None:
    """images.max_source_pixels (against a poison image): Pillow's decompression-bomb bound by default; it can be
    lowered, not raised or turned off."""
    import pytest

    path = str(_gemma_checkpoint(tmp_path))
    cfg = config.derive(config.load(None, {"model.path": path}))
    assert cfg.images.enabled and cfg.images.max_source_pixels == 178_956_970
    cfg = config.derive(config.load(None, {"model.path": path, "images.max_source_pixels": 1_000_000}))
    assert cfg.images.max_source_pixels == 1_000_000
    for bad in (0, 178_956_971):
        with pytest.raises(ValueError, match=r"images\.max_source_pixels must be 1\.\.178956970"):
            config.derive(config.load(None, {"model.path": path, "images.max_source_pixels": bad}))


def test_the_request_budgets_are_validated(tmp_path) -> None:
    """images.max_request_decode_bytes and max_request_download_mb (vLLM's per-request budgets): vLLM's defaults, 0
    (no limit) or more."""
    import pytest

    path = str(_gemma_checkpoint(tmp_path))
    cfg = config.derive(config.load(None, {"model.path": path}))
    assert (cfg.images.max_request_decode_bytes, cfg.images.max_request_download_mb) == (715_827_880, 256)
    for key in ("max_request_decode_bytes", "max_request_download_mb"):
        assert getattr(config.derive(config.load(None, {"model.path": path, f"images.{key}": 0})).images, key) == 0
        with pytest.raises(ValueError, match=rf"images\.{key} must be 0 \(no limit\) or more, got -1"):
            config.derive(config.load(None, {"model.path": path, f"images.{key}": -1}))


def test_the_url_allow_lists_are_validated(tmp_path) -> None:
    """images.allowed_networks takes CIDRs and images.allowed_domains exact hostnames, both empty by default; a bad
    entry is a ValueError naming the setting, at config time."""
    import re

    import pytest

    path = str(_gemma_checkpoint(tmp_path))
    cfg = config.derive(config.load(None, {"model.path": path}))
    assert cfg.images.allowed_networks == [] and cfg.images.allowed_domains == []
    cfg = config.derive(config.load(None, {"model.path": path, "images.allowed_networks": ["10.0.0.0/8", "fd00::/8"],
                                           "images.allowed_domains": ["Images.Example.com"]}))
    assert cfg.images.allowed_networks == ["10.0.0.0/8", "fd00::/8"]
    with pytest.raises(ValueError, match=r"images\.allowed_networks: 'not-a-cidr'"):
        config.derive(config.load(None, {"model.path": path, "images.allowed_networks": ["10.0.0.0/8", "not-a-cidr"]}))
    for bad in ("http://x", "*.example.com"):
        with pytest.raises(ValueError, match=r"images\.allowed_domains: " + re.escape(repr(bad))):
            config.derive(config.load(None, {"model.path": path, "images.allowed_domains": [bad]}))


DELETED = [  # (group, key, a value it took): every deleted setting, refused whatever the value, its old default too
    ("engine", "first_tokens_first", False),  # under MTP its decode-only test never matches
    # the HTTP server's choice (Granian or uvicorn), KV events, and A/B switches of which one side is kept
    ("server", "http", "uvicorn"), ("cache", "kv_events", False), ("cache", "kv_events_endpoint", "tcp://*:5557"),
    ("cache", "kv_events_topic", "x"), ("cache", "refine_chunking", "fixed"), ("cache", "refine_steps", "mixed"),
    ("runner.cuda", "attn512_split", "waves"), ("runner.cuda", "vision_glue", "fused"),
    ("speculative", "method", "mtp"),  # one value, mtp, the only method there is
    ("dev.runner.cuda", "fp8_small_m", 0),  # the small-M FP8 GEMM's switch; 0 its last default
]


@pytest.mark.parametrize(("group", "key", "value"), DELETED, ids=[f"{g}.{k}" for g, k, _ in DELETED])
def test_deleted_settings_are_refused(tmp_path, monkeypatch, capsys, caplog, group, key, value) -> None:
    """A deleted setting fails the load whatever its value, from a file or an override, and the cli and the
    container's entrypoint (its preflight) exit 2 naming it."""
    import yaml

    from llm_mini_infer import cli, entrypoint

    refused = f"unknown field `{key}`"
    nested: dict = {key: value}
    for part in reversed(group.split(".")):
        nested = {part: nested}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(nested))
    for path_, overrides in ((str(path), {}), (None, {f"{group}.{key}": value})):
        with pytest.raises(ValueError, match=refused):
            config.load(path_, overrides)
    with pytest.raises(SystemExit) as exc:
        cli.main(["serve", "--model", str(tmp_path), "--set", f"{group}.{key}={yaml.safe_dump(value).split()[0]}"])
    assert exc.value.code == 2 and refused in capsys.readouterr().err
    assert refused in entrypoint.preflight(str(path), [])[0]
    monkeypatch.setenv("LLM_MINI_INFER_CONFIG", str(path))
    with pytest.raises(SystemExit) as exc:
        entrypoint.main(["serve"])
    logged = [r.getMessage() for r in caplog.records
              if r.name == "llm_mini_infer.entrypoint" and r.levelname == "ERROR"]
    assert exc.value.code == 2 and len(logged) == 1 and refused in logged[0]  # not merely a missing model


def test_the_runner_is_cuda_unless_fake_is_set(tmp_path, monkeypatch) -> None:
    """runner.kind is cuda unless a config sets fake, and never derived from what torch sees: a config that names no
    runner serves the model or fails at startup, never random tokens behind a passing /health. Here torch sees no GPU,
    as on a host whose GPU went missing."""
    import pytest
    import torch

    from llm_mini_infer.engine.process import build_runner

    assert config.load(None, {}).runner.kind == "cuda"
    assert config.load(None, {"runner.kind": "fake"}).runner.kind == "fake"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cfg = config.derive(config.load(None, {"model.path": str(_gemma(tmp_path / "g"))}))
    with pytest.raises(RuntimeError, match=r"runner\.kind cuda .* needs a CUDA GPU"):
        build_runner(cfg)


def test_the_runner_kind_refuses_mlx(capsys) -> None:
    """runner.kind names only the runners that exist: mlx, not built (architecture.md, "Not yet"), fails the load and
    the cli's --runner, before the engine starts."""
    import pytest

    from llm_mini_infer import cli

    with pytest.raises(ValueError, match=r"Invalid enum value 'mlx' - at `\$\.runner\.kind`"):
        config.load(None, {"runner.kind": "mlx"})
    with pytest.raises(SystemExit) as exc:
        cli.main(["serve", "--runner", "mlx"])
    assert exc.value.code == 2 and "invalid choice: 'mlx'" in capsys.readouterr().err


def test_refine_settings_are_validated() -> None:
    """How a refine is stepped (the precise prefix): in fixed pieces, so that every start cuts the precise copy at the
    same boundaries at any load, the piece derived unless set. Stepping has no modes: either mode key is refused at
    load, whatever its value."""
    import pytest

    assert config.load(None, {}).dev.cache.refine_chunk_tokens is None
    assert config.load(None, {"dev.cache.refine_chunk_tokens": 1024}).dev.cache.refine_chunk_tokens == 1024
    for key in ("cache.refine_chunking", "cache.refine_steps"):
        for value in ("fixed", "bound", "mixed", "exclusive"):
            with pytest.raises(ValueError, match="invalid configuration"):
                config.load(None, {key: value})


def test_a_refine_piece_is_checked_before_the_engine_starts(tmp_path) -> None:
    """`dev.cache.refine_chunk_tokens`, when set, is checked at derive time (the launcher's, before the engine loads any
    weight): positive, at most max_num_batched_tokens, and at most what a step leaves next to a full batch of decode
    rows (2,048 - 64 x (1 + k) by default). The engine checks its grid too."""
    import pytest

    def derive(model: Path, **kw) -> config.Config:
        return config.derive(config.load(None, {"model.path": str(model), **kw}))

    plain = _model(tmp_path, 248_320, 0)
    (tmp_path / "mtp").mkdir()
    mtp = _model(tmp_path / "mtp", 248_320, 1)
    for n in (0, -16, 4096):
        with pytest.raises(ValueError, match=r"cache\.refine_chunk_tokens .*: positive"):
            derive(plain, **{"dev.cache.refine_chunk_tokens": n})
    with pytest.raises(ValueError, match=r"at most 1984 \(.*: a piece must fit next to a full batch"):
        derive(plain, **{"dev.cache.refine_chunk_tokens": 2048})
    with pytest.raises(ValueError, match=r"at most 1792 \("):
        derive(mtp, **{"dev.cache.refine_chunk_tokens": 1808})  # k = 3: 2,048 - 64 x 4
    with pytest.raises(ValueError, match=r"at most 1792 \("):
        derive(mtp, **{"dev.cache.refine_chunk_tokens": 2048})  # 2,048 at 2,048-token steps: no decode row's room
    assert derive(plain, **{"dev.cache.refine_chunk_tokens": 1984}).dev.cache.refine_chunk_tokens == 1984
    assert derive(mtp, **{"dev.cache.refine_chunk_tokens": 1792}).dev.cache.refine_chunk_tokens == 1792
    assert derive(plain).dev.cache.refine_chunk_tokens is None  # derived by the engine
    gemma = {"speculative.model": str(_assistant(tmp_path / "asst"))}  # an assistant drafts 5 by default
    with pytest.raises(ValueError, match=r"at most 1664 \("):
        derive(_gemma(tmp_path / "gemma"), **gemma, **{"dev.cache.refine_chunk_tokens": 1792})  # 2,048 - 64 x 6
    cfg = derive(tmp_path / "gemma", **gemma, **{"dev.cache.refine_chunk_tokens": 1664})
    assert cfg.dev.cache.refine_chunk_tokens == 1664


def test_the_kernel_switches(tmp_path) -> None:
    """Programmatic dependent launch defaults on for Gemma 4 and for Qwen; an override wins. Batch invariance is on
    (CUTLASS at every M); the small-M FP8 GEMM has no switch (`DELETED`)."""
    g = str(_gemma(tmp_path / "g"))
    cfg = config.derive(config.load(None, {"model.path": g}))
    assert cfg.dev.runner.cuda.batch_invariant is True and cfg.dev.runner.cuda.pdl is True
    assert "on" in cfg.reasons["dev.runner.cuda.pdl"]
    assert "CUTLASS at every M" in cfg.reasons["dev.runner.cuda.batch_invariant"]
    cfg = config.derive(config.load(None, {"model.path": g, "dev.runner.cuda.pdl": False}))
    assert cfg.dev.runner.cuda.pdl is False
    q = str(_model(tmp_path, 248_320, 1))
    cfg = config.derive(config.load(None, {"model.path": q}))
    assert cfg.dev.runner.cuda.pdl is True and cfg.reasons["dev.runner.cuda.pdl"].startswith("on: ")
    for on in (False, True):
        assert config.derive(config.load(None, {"model.path": q, "dev.runner.cuda.pdl": on})).dev.runner.cuda.pdl is on


def test_batch_invariance_needs_a_short_context(tmp_path) -> None:
    """Batch invariance defaults on for Gemma 4 while a decode row's keys, its verify window's included, fit the full
    layers' fixed split at 256 keys a split (32 splits): on at the range's top, off one step past it, the reason naming
    what forcing it on costs; a set value wins. Off too where a prefill could starve (a token and a final chunk's
    forced prefix next to max_num_seqs − 1 decode rows) or the window pages are not a multiple of 16 (our sliding
    kernel's KV tile), which a set value refuses."""
    g = str(_gemma(tmp_path / "g", positions=8192))

    def derive(n: int, **kw) -> config.Config:
        return config.derive(config.load(None, {"model.path": g, "model.max_model_len": n,
                                                "structured_output.enabled": True, **kw}))

    cfg = derive(MAXL)
    assert cfg.dev.runner.cuda.batch_invariant is True
    assert cfg.reasons["dev.runner.cuda.batch_invariant"].startswith("on: Gemma 4")
    cfg = derive(config.BATCH_INVARIANT_MAX_LEN)
    assert cfg.dev.runner.cuda.batch_invariant is False
    why = cfg.reasons["dev.runner.cuda.batch_invariant"]
    assert why.startswith(f"off: model.max_model_len {config.BATCH_INVARIANT_MAX_LEN} is past {MAXL:,}")
    assert "fewer, longer splits" in why and "a prefill row one however long its history" in why
    forced = derive(config.BATCH_INVARIANT_MAX_LEN, **{"dev.runner.cuda.batch_invariant": True})
    assert forced.dev.runner.cuda.batch_invariant is True
    assert derive(MAXL, **{"dev.runner.cuda.batch_invariant": False}).dev.runner.cuda.batch_invariant is False
    # 64 sequences, no MTP: a token and the forced prefix (2 x 1) next to 63 decode rows of 1 token
    assert derive(MAXL, **{"scheduler.max_num_batched_tokens": 66}).dev.runner.cuda.batch_invariant is True
    tight = derive(MAXL, **{"scheduler.max_num_batched_tokens": 65})
    assert tight.dev.runner.cuda.batch_invariant is False
    assert tight.reasons["dev.runner.cuda.batch_invariant"] == (
        "off: scheduler.max_num_batched_tokens 65 is below 66: a prefill token, with its forced prefix, next to "
        "max_num_seqs - 1 = 63 decode rows of 1 tokens")
    with pytest.raises(ValueError, match=r"batch_invariant: scheduler\.max_num_batched_tokens 65 is below 66"):
        derive(MAXL, **{"scheduler.max_num_batched_tokens": 65, "dev.runner.cuda.batch_invariant": True})
    assert derive(MAXL, **{"scheduler.max_num_batched_tokens": 65, "structured_output.enabled": False}
                  ).dev.runner.cuda.batch_invariant is True  # no forced prefix
    pages = derive(MAXL, **{"dev.cache.page_size": 8})
    assert pages.dev.runner.cuda.batch_invariant is False and pages.reasons["dev.runner.cuda.batch_invariant"] == (
        "off: dev.cache.page_size 8 is not a multiple of 16 (our attention kernels' KV tile)")


def test_qwen_batch_invariance(tmp_path) -> None:
    """The Qwen runner's batch invariance defaults on for a block-FP8 checkpoint up to the range's top while a step can
    always cut a prefill on the 64-token grid (max_num_batched_tokens past a grid, a final chunk's forced prefix and
    max_num_seqs − 1 decode rows of 1 + k tokens; no chunk cap below the grid), its reason naming what it covers; off
    for a BF16 checkpoint (its projections run at the step's M), past it, or where a prefill could starve, which a
    set value refuses; a set value wins otherwise."""
    for d in ("q", "b", "long"):
        (tmp_path / d).mkdir()
    q = _model(tmp_path / "q", 248_320, 1)
    conf = json.loads((q / "config.json").read_text())
    conf["quantization_config"] = {"quant_method": "fp8", "weight_block_size": [128, 128]}
    (q / "config.json").write_text(json.dumps(conf))
    conf["text_config"]["max_position_embeddings"] = 8192
    (tmp_path / "long" / "config.json").write_text(json.dumps(conf))

    def derive(**kw) -> config.Config:  # no refine piece to fit next to the decode rows (tight budgets below)
        return config.derive(config.load(None, {"model.path": str(q), "structured_output.enabled": True,
                                                "runner.cuda.fp8_cold_prefill": "fp8", **kw}))

    cfg = derive()
    assert cfg.dev.runner.cuda.batch_invariant is True
    why = cfg.reasons["dev.runner.cuda.batch_invariant"]
    assert why.startswith("on: Qwen3.8 with a block-FP8 checkpoint") and "in_proj_ba" in why and "64-token grid" in why
    bf16 = derive(**{"model.path": str(_model(tmp_path / "b", 248_320, 1))})
    assert bf16.dev.runner.cuda.batch_invariant is False
    assert bf16.reasons["dev.runner.cuda.batch_invariant"].startswith("off: a BF16 checkpoint")
    long = derive(**{"model.path": str(tmp_path / "long")})  # its position limit, 8,192
    assert long.dev.runner.cuda.batch_invariant is False
    assert long.reasons["dev.runner.cuda.batch_invariant"].startswith(f"off: model.max_model_len 8192 is past {MAXL:,}")
    # k = 3 (the MTP layer): 64 + 2 x 4 + 63 x 4 = 324 tokens a step at 64 sequences
    assert derive(**{"scheduler.max_num_batched_tokens": 324}).dev.runner.cuda.batch_invariant is True
    tight = derive(**{"scheduler.max_num_batched_tokens": 323})
    assert tight.dev.runner.cuda.batch_invariant is False
    assert tight.reasons["dev.runner.cuda.batch_invariant"] == (
        "off: scheduler.max_num_batched_tokens 323 is below 324: a prefill cut on the 64-token grid, with its forced "
        "prefix, next to max_num_seqs - 1 = 63 decode rows of 4 tokens")
    with pytest.raises(ValueError, match=r"batch_invariant: scheduler\.max_num_batched_tokens 323 is below 324"):
        derive(**{"scheduler.max_num_batched_tokens": 323, "dev.runner.cuda.batch_invariant": True})
    assert derive(**{"scheduler.max_num_batched_tokens": 316, "structured_output.enabled": False}
                  ).dev.runner.cuda.batch_invariant is True  # no forced prefix: 64 + 63 x 4
    cap = derive(**{"dev.scheduler.long_prefill_token_threshold": 48})
    assert cap.dev.runner.cuda.batch_invariant is False and "48 is below the 64-token grid" in \
        cap.reasons["dev.runner.cuda.batch_invariant"]
    assert derive(**{"dev.runner.cuda.batch_invariant": False}).dev.runner.cuda.batch_invariant is False
    with pytest.raises(ValueError, match=r"dev\.cache\.page_size 24 is not a multiple of 16"):
        derive(**{"dev.cache.page_size": 24, "dev.runner.cuda.batch_invariant": True})
    assert bf16.dev.runner.cuda.batch_invariant is False
    assert config.derive(config.load(None, {"model.path": str(tmp_path / "b"), "dev.runner.cuda.batch_invariant":
                                            True})).dev.runner.cuda.batch_invariant is True


def test_the_budget_for_batch_invariance_is_the_schedulers(tmp_path) -> None:
    """Config turns batch invariance off exactly where the scheduler would refuse it (`Scheduler._check_progress`): at
    the smallest max_num_batched_tokens config allows the scheduler starts, one token below both refuse. Gemma 4 with
    its assistant (k = 5, a drafter without KV of its own: the forced prefix keeps its chain's positions) and an FP8
    Qwen3.8 (k = 3, an MTP layer that writes KV, a 64-token grid)."""
    from llm_mini_infer.grammar.engine import Grammars
    from llm_mini_infer.sched.scheduler import Scheduler

    from ..support.fake_grammar import FakeGrammar

    for d in ("g", "a", "q"):
        (tmp_path / d).mkdir()
    q = _model(tmp_path / "q", 248_320, 1)
    conf = json.loads((q / "config.json").read_text())
    conf["quantization_config"] = {"quant_method": "fp8", "weight_block_size": [128, 128]}
    (q / "config.json").write_text(json.dumps(conf))
    cases = [({"model.path": str(_gemma(tmp_path / "g")), "speculative.model": str(_assistant(tmp_path / "a"))},
              dict(mtp_k=5, draft_kv=False, hybrid=False, checkpoint_grid=16, window=1024, window_page_size=16)),
             ({"model.path": str(q), "runner.cuda.fp8_cold_prefill": "fp8"},  # no refine piece (budgets below)
              dict(mtp_k=3, draft_kv=True, hybrid=True, checkpoint_grid=64))]
    def derive(model: dict, n: int | None = None) -> config.Config:
        budget = {} if n is None else {"scheduler.max_num_batched_tokens": n}
        return config.derive(config.load(None, {**model, "structured_output.enabled": True, **budget}))

    def scheduler(n: int, seqs: int, shape: dict) -> Scheduler:
        return Scheduler(max_num_seqs=seqs, max_num_batched_tokens=n, max_model_len=MAXL, page_size=16, num_pages=4096,
                         num_state_slots=seqs + 8, eos_token_ids=(511,), forced_prefix=True,
                         grammars=Grammars(FakeGrammar(512, (511,))), batch_invariant=True, **shape)

    for model, shape in cases:
        cfg = derive(model)
        k, seqs = cfg.speculative.num_speculative_tokens, cfg.scheduler.max_num_seqs
        assert k == shape["mtp_k"]
        why = derive(model, seqs * (1 + k)).reasons["dev.runner.cuda.batch_invariant"]
        need = int(re.search(r"is below (\d+)", why).group(1))
        assert derive(model, need).dev.runner.cuda.batch_invariant is True
        assert derive(model, need - 1).dev.runner.cuda.batch_invariant is False
        scheduler(need, seqs, shape)
        with pytest.raises(ValueError, match=f"under the {need - (seqs - 1) * (1 + k)} a prefill needs"):
            scheduler(need - 1, seqs, shape)


def test_malformed_yaml_is_a_configuration_error(tmp_path) -> None:
    """A YAML syntax error in the file or a `--set` value is a ValueError, which the CLI and the image's checks turn
    into exit 2 (a configuration error), not a traceback and exit 1."""
    bad = tmp_path / "bad.yaml"
    bad.write_text("scheduler: [max_num_seqs: 8\n")
    with pytest.raises(ValueError, match=r"bad\.yaml: not valid YAML"):
        config.load(str(bad), {})
    with pytest.raises(ValueError, match=r"scheduler\.max_num_seqs: the value is not valid YAML"):
        config.parse_override("scheduler.max_num_seqs=[8")
    r = subprocess.run([sys.executable, "-m", "llm_mini_infer.cli", "serve", "--config", str(bad)], capture_output=True,
                       text=True, timeout=60)
    assert r.returncode == 2 and "not valid YAML" in r.stderr and "Traceback" not in r.stderr, r.stderr


def test_a_setting_derive_refuses_exits_2_from_the_cli() -> None:
    """`llm-mini-infer serve` outside the image: a setting that `derive` refuses (here a context past the model's
    position limit) is one log line and exit 2, as through the image's checks, not a traceback and exit 1."""
    from ..support.fixtures import QWEN, require_qwen

    require_qwen()
    r = subprocess.run([sys.executable, "-m", "llm_mini_infer.cli", "serve", "--model", str(QWEN), "--max-model-len",
                        "300000", "--set", "runner.kind=fake"], capture_output=True, text=True, timeout=120)
    assert r.returncode == 2 and "Traceback" not in r.stderr, r.stderr
    assert "configuration: model.max_model_len 300000: at most the model's position limit" in r.stderr, r.stderr


def test_state_fields_are_refused_at_load(tmp_path, monkeypatch, capsys, caplog) -> None:
    """The server's state (Config's fields that are not groups, which the hub, derive(), the launcher and the engine
    fill) is refused from a file and from an override, and the cli and the entrypoint's preflight exit 2 naming it.
    Were it accepted, `--set pixel_arena=<path>` with images off would have the launcher unlink that path at exit, a set
    model_source would name the served model, and a set grammar_eos_token_ids would show in /server_info."""
    import msgspec
    import pytest
    import yaml

    from llm_mini_infer import cli, entrypoint

    assert config.STATE_FIELDS == ("model_source", "speculative_source", "eos_token_ids", "grammar_eos_token_ids",
                                   "mtp_k_max", "vocab_size", "cpu_budget", "image_tokens_max", "pixel_arena",
                                   "pool_dir", "runner_info", "reasons")
    groups = {f.name for f in msgspec.structs.fields(config.Config)} - set(config.STATE_FIELDS)
    assert groups == {"model", "sampling", "scheduler", "cache", "speculative", "images", "structured_output",
                      "tool_calls", "reasoning", "runner", "server", "engine", "observability", "dev"}
    keep = tmp_path / "keep"
    keep.write_text("not the server's")
    for name in config.STATE_FIELDS:
        path = tmp_path / f"{name}.yaml"
        path.write_text(yaml.safe_dump({name: str(keep)}))
        for path_, overrides in ((str(path), {}), (None, {name: str(keep)}), (None, {f"{name}.x": 1})):
            with pytest.raises(ValueError, match=rf"invalid configuration: {name}: the server's own state"):
                config.load(path_, overrides)
    with pytest.raises(ValueError, match="model_source, pixel_arena: the server's own state"):
        config.load(None, {"pixel_arena": str(keep), "model_source": "org/name@abc"})
    with pytest.raises(SystemExit) as exc:
        cli.main(["serve", "--model", str(tmp_path), "--runner", "fake", "--set", f"pixel_arena={keep}"])
    assert exc.value.code == 2 and "pixel_arena: the server's own state" in capsys.readouterr().err
    path = tmp_path / "pixel_arena.yaml"
    assert "pixel_arena: the server's own state" in entrypoint.preflight(str(path), [])[0]
    monkeypatch.setenv("LLM_MINI_INFER_CONFIG", str(tmp_path / "none.yaml"))
    (tmp_path / "none.yaml").write_text("runner:\n  kind: fake\n")
    with pytest.raises(SystemExit) as exc:
        entrypoint.main(["serve", "--set", f"pixel_arena={keep}"])  # the container's --set route
    logged = [r.getMessage() for r in caplog.records
              if r.name == "llm_mini_infer.entrypoint" and r.levelname == "ERROR"]
    assert exc.value.code == 2 and len(logged) == 1 and "pixel_arena: the server's own state" in logged[0]


def test_superseded_checkpoints_are_demoted_by_default() -> None:
    """Checkpoints a conversation has moved past go first under checkpoint pressure unless a benchmark turns it off."""
    assert config.load(None, {}).dev.cache.demote_superseded is True


def test_the_settings_split(tmp_path, monkeypatch, capsys, caplog) -> None:
    """An operator's settings are fewer than 60, the tests' and benchmarks' live in the dev group under the same names,
    and the server's state apart. A dev setting given outside the dev group, from a file or an override, fails the load
    naming it and where it goes, and so do the cli and the entrypoint's preflight (exit 2); a dev subgroup (the fake
    runner's) is refused whole."""
    import pytest
    import yaml

    from llm_mini_infer import cli, entrypoint

    dev = [k.removeprefix("dev.") for k in config.leaves(config.Config) if k.startswith("dev.")]
    assert len(config.SETTINGS) == 59 < 60 and len(dev) == 61 and not set(dev) & set(config.SETTINGS)
    for name, value in (("runner.cuda.fused_ops", False), ("cache.num_pages", 10), ("runner.fake.step_ms", 1.0),
                        ("scheduler.mixed_step_row_tokens", 100)):
        got = config.load(None, {f"dev.{name}": value}).dev
        for part in name.split("."):
            got = getattr(got, part)
        assert got == value
        refused = "runner.fake" if ".fake." in name else name  # the fake runner's group, whole
        with pytest.raises(ValueError, match=rf"{refused}: dev settings \(tests.*: set dev\."):
            config.load(None, {name: value})
    path = tmp_path / "old.yaml"
    cuda = {"graphs": False, "memory_utilization": 0.8}
    path.write_text(yaml.safe_dump({"runner": {"kind": "fake", "fake": {"step_ms": 1}, "cuda": cuda},
                                    "cache": {"enable_prefix_caching": True, "page_size": 32}}))
    msg = "cache.page_size, runner.cuda.graphs, runner.fake: dev settings"  # the file's order
    with pytest.raises(ValueError, match=msg) as err:
        config.load(str(path), {})
    assert "set dev.cache.page_size, dev.runner.cuda.graphs, dev.runner.fake" in str(err.value)
    cfg = config.load(None, {"runner.cuda.memory_utilization": 0.8, "dev.runner.fake.step_ms": 1,
                             "dev.cache.page_size": 32})
    assert (cfg.runner.cuda.memory_utilization, cfg.dev.runner.fake.step_ms, cfg.dev.cache.page_size) == (0.8, 1, 32)
    with pytest.raises(SystemExit) as exc:
        cli.main(["serve", "--model", str(tmp_path), "--set", "runner.cuda.graphs=false"])
    assert exc.value.code == 2 and "runner.cuda.graphs: dev settings" in capsys.readouterr().err
    assert msg in entrypoint.preflight(str(path), [])[0]
    monkeypatch.setenv("LLM_MINI_INFER_CONFIG", str(path))
    with pytest.raises(SystemExit) as exc:
        entrypoint.main(["serve"])
    logged = [r.getMessage() for r in caplog.records
              if r.name == "llm_mini_infer.entrypoint" and r.levelname == "ERROR"]
    assert exc.value.code == 2 and len(logged) == 1 and "dev settings (tests" in logged[0]
    lines = config.effective_lines(config.Config())  # the startup log: the settings, then the dev group, then state
    assert [line.split(" = ")[0] for line in lines[:len(config.SETTINGS)]] == list(config.SETTINGS)
    assert lines[len(config.SETTINGS)].startswith("dev.") and lines[-1].startswith("runner_info = ")


def test_the_launcher_removes_only_an_arena_it_made(tmp_path, monkeypatch) -> None:
    """A Config built in code carrying a pixel arena, with images off: the launcher clears its own fields before it
    starts anything, so its exit removes no file it did not create (load() refuses the field; this is the second
    guard). The launcher stops here at its first step, as when the image pool fails to start."""
    import signal

    from llm_mini_infer import launcher

    keep = tmp_path / "keep"
    keep.write_text("not the server's")
    cfg = config.Config()
    cfg.images.enabled = False
    cfg.pixel_arena = cfg.pool_dir = str(keep)
    monkeypatch.setattr(launcher, "setup_logging", lambda *a: None)
    monkeypatch.setattr(launcher.hub, "resolve", lambda c: None)
    monkeypatch.setattr(launcher, "derive", lambda c: c)
    monkeypatch.setattr(signal, "signal", lambda *a: None)  # the test process keeps its own handlers

    def fail(*_a):
        raise RuntimeError("the image pool did not start (test)")

    monkeypatch.setattr(launcher, "_start_images", fail)
    assert launcher.serve(cfg) == 1
    assert keep.read_text() == "not the server's" and cfg.pixel_arena == cfg.pool_dir == ""


def test_fp32_on_cuda_needs_the_torch_references(tmp_path) -> None:
    """Our kernels are BF16: dev.runner.cuda.dtype float32 on CUDA with dev.runner.cuda.fused_ops on is refused at
    load; with fused_ops off, on the CPU device or the fake runner it loads."""
    model = str(_model(tmp_path, 248_320, 0))

    def derive(**kw) -> config.Config:
        return config.derive(config.load(None, {"model.path": model, "dev.runner.cuda.dtype": "float32", **kw}))

    for device in ("cuda", "cuda:1"):
        with pytest.raises(ValueError, match=r"^dev\.runner\.cuda\.dtype float32 on CUDA needs "
                                             r"dev\.runner\.cuda\.fused_ops false \(our kernels are BF16\)$"):
            derive(**{"dev.runner.cuda.device": device})
    for kw in ({"dev.runner.cuda.fused_ops": False}, {"dev.runner.cuda.device": "cpu"}, {"runner.kind": "fake"}):
        assert derive(**kw).dev.runner.cuda.dtype == "float32"


def test_a_set_context_past_the_models_position_limit_is_refused(tmp_path) -> None:
    """A set model.max_model_len past config.json's smallest position limit is refused at load, naming the limit
    (the runners would cap it to the limit unseen); at the limit, or derived, it loads."""
    model = str(_model(tmp_path, 248_320, 0))

    def derive(**kw) -> config.Config:
        return config.derive(config.load(None, {"model.path": model, **kw}))

    with pytest.raises(ValueError, match=r"^model\.max_model_len 4097: at most the model's position limit, "
                                         r"config\.json max_position_embeddings 4096$"):
        derive(**{"model.max_model_len": 4097})
    assert derive(**{"model.max_model_len": 4096}).model.max_model_len == 4096
    assert derive().model.max_model_len == 4096


def test_no_room_for_a_derived_refine_piece_is_refused_at_load(tmp_path) -> None:
    """A block-FP8 checkpoint under runner.cuda.fp8_cold_prefill reuse derives a refine piece of at least one checkpoint
    grid: where a full batch of decode rows leaves less, it is refused at load, naming the settings that make room (the
    scheduler refused the derived piece at startup as if it were set). A BF16 checkpoint, or another fp8_cold_prefill,
    loads."""
    for d in ("q", "b"):
        (tmp_path / d).mkdir()
    q = _model(tmp_path / "q", 248_320, 1)
    conf = json.loads((q / "config.json").read_text())
    conf["quantization_config"] = {"quant_method": "fp8", "weight_block_size": [128, 128]}
    (q / "config.json").write_text(json.dumps(conf))

    def derive(model: Path, **kw) -> config.Config:
        return config.derive(config.load(None, {"model.path": str(model), "scheduler.max_num_seqs": 512,
                                                "scheduler.max_num_batched_tokens": 1024, **kw}))

    with pytest.raises(ValueError, match=r"^scheduler\.max_num_batched_tokens 1024 leaves -1024 tokens next to "
                                         r"max_num_seqs 512 decode rows x 4, under one 64-token piece"):
        derive(q)
    derive(q, **{"scheduler.max_num_batched_tokens": 512 * 4 + 64})  # one piece fits
    derive(q, **{"runner.cuda.fp8_cold_prefill": "bf16"})
    derive(_model(tmp_path / "b", 248_320, 1))


def test_a_checkpoint_without_an_eos_id_says_so(tmp_path, caplog) -> None:
    """No eos_token_id in generation_config.json or config.json: the reason and a warning say none was found (not
    "config.json eos_token_id"), and the EOS-free finish is off for that reason, refused for it when set (not as one
    on several ids)."""
    import logging

    conf = {"text_config": {"model_type": "qwen3_5_text", "vocab_size": 1000, "max_position_embeddings": 4096}}
    (tmp_path / "config.json").write_text(json.dumps(conf))

    def derive(**kw) -> config.Config:
        return config.derive(config.load(None, {"model.path": str(tmp_path), "structured_output.enabled": True,
                                                "structured_output.backend": "xgrammar", **kw}))

    with caplog.at_level(logging.WARNING, logger="llm_mini_infer.config"):
        cfg = derive()
    assert cfg.eos_token_ids == [] and cfg.reasons["eos_token_ids"].startswith("none: no eos_token_id")
    assert any("eos_token_ids: none" in r.message for r in caplog.records)
    assert cfg.reasons["dev.structured_output.eos_free_finish"] == "off: the checkpoint names no EOS id"
    with pytest.raises(ValueError, match=r"eos_free_finish: the checkpoint names no EOS id \(eos_token_ids is empty\)"):
        derive(**{"dev.structured_output.eos_free_finish": True})


def _with_tokenizer(tmp: Path, fixture: str, template: str | None = None) -> Path:
    """A model directory's tokenizer and chat template from a fixture (`template`: another chat template)."""
    src = MODELS_DIR / fixture
    if not (src / "tokenizer.json").exists():
        pytest.skip(f"{src} is absent: uv run tests/scripts/fetch_fixtures.py")
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy(src / name, tmp / name)
    (tmp / "chat_template.jinja").write_text(template if template is not None else
                                             (src / "chat_template.jinja").read_text())
    return tmp


def test_the_reasoning_settings(tmp_path) -> None:
    """reasoning.parser derived by family and template, a set one checked against the tokenizer; reasoning.effort_map
    the family's by default, a set one replacing it whole, and a refused one stopping the start with vLLM's message
    (the setting's name for its flag's)."""
    from llm_mini_infer.output.reasoning import EFFORT_MAPS

    def derived(model: Path, **kw):
        return config.derive(config.load(None, {"model.path": str(model), **kw}))

    def model(name: str, kind: str, fixture: str, template: str | None = None) -> Path:
        d = tmp_path / name
        d.mkdir()
        if kind == "gemma":
            _gemma(d)
        elif kind == "qwen":
            _model(d, 248_320, 1)
        else:
            (d / "config.json").write_text(json.dumps({"model_type": "llama", "vocab_size": 248_320,
                                                       "max_position_embeddings": 4096, "eos_token_id": 1}))
        return _with_tokenizer(d, fixture, template)

    gemma, qwen = model("gemma", "gemma", "gemma-4-31b-it"), model("qwen", "qwen", "qwen3.8-27b")
    cfg = derived(gemma)
    assert (cfg.reasoning.parser, cfg.reasoning.effort_map) == ("gemma4", {})
    assert cfg.reasons["reasoning.parser"].startswith("gemma4: Gemma 4")
    cfg = derived(qwen)
    assert (cfg.reasoning.parser, cfg.reasoning.effort_map) == ("qwen3", EFFORT_MAPS["qwen3"])
    cfg = derived(model("neither", "llama", "qwen3.8-27b"))
    assert (cfg.reasoning.parser, cfg.reasoning.effort_map) == ("none", {})
    assert cfg.reasons["reasoning.parser"] == "none: the model is neither Gemma 4 nor Qwen3.5/3.8"
    cfg = derived(model("plain", "qwen", "qwen3.8-27b", "{% for m in messages %}{{ m.content }}{% endfor %}"))
    assert cfg.reasoning.parser == "none"
    assert cfg.reasons["reasoning.parser"] == "none: the chat template does not render Qwen3.5/3.8's thought markers"
    cfg = derived(model("plain-gemma", "gemma", "gemma-4-31b-it", "{% for m in messages %}{{ m.content }}{% endfor %}"))
    assert (cfg.reasoning.parser, cfg.reasoning.effort_map) == ("none", {})
    assert cfg.reasons["reasoning.parser"] == "none: the chat template does not render Gemma 4's thought markers"
    assert derived(qwen, **{"reasoning.parser": "none"}).reasoning.parser == "none"
    with pytest.raises(ValueError) as e:
        derived(qwen, **{"reasoning.parser": "gemma4"})
    assert str(e.value) == ("invalid configuration: reasoning.parser 'gemma4' does not fit the tokenizer, which has no "
                            "<|channel>, <channel|> - at `$.reasoning.parser`")
    assert derived(qwen, **{"reasoning.effort_map": {"high": "low"}}).reasoning.effort_map == {"high": "low"}
    assert derived(gemma, **{"reasoning.effort_map": {"none": "none"}}).reasoning.effort_map == {"none": "none"}
    shape = "Error: reasoning.effort_map must be a JSON object mapping reasoning efforts to strings."
    for value, message in (([1], shape), ("high", shape), ({"high": 3}, shape),
                           ({"ultra": "x", "low": "low"}, "Error: reasoning.effort_map has unknown efforts ['ultra']; "
                            "keys must be one of ['none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'].")):
        with pytest.raises(ValueError) as e:
            derived(qwen, **{"reasoning.effort_map": value})
        assert str(e.value) == message, value
