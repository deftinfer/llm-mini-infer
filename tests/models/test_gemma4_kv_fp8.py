"""Gemma 4's full-attention layers in an FP8 KV pool (`cache.kv_cache_dtype fp8`) on the CPU: the torch references'
store and read (`kernels.gemma_ops.kv_fp8`, `kv_read`), the runner's geometry and refusals, the fake runner's."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from llm_mini_infer.kernels import gemma_ops as gops
from llm_mini_infer.models.gemma4.backend import FULL, Gemma4Pools, TorchFullAttention

from ..scripts.numerics_forced import e4m3_round
from ..support.paths import ROOT
from ..support.tiny_gemma import make_tiny_gemma

F8 = torch.float8_e4m3fn


def every_bf16() -> torch.Tensor:
    return torch.arange(1 << 16, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)


def test_the_store_rounds_every_bf16_value_as_the_fake_quant() -> None:
    """`kv_fp8` on every BF16 bit pattern: the E4M3 value it stores is `numerics_forced.e4m3_round`'s (the stage that
    measured the accuracy) for every finite value and ±inf, both of which saturate to ±448; a NaN stays a NaN."""
    x = every_bf16()
    got = gops.kv_fp8(x)
    assert got.dtype == torch.uint8 and got.shape == x.shape
    back = got.view(F8).to(torch.bfloat16)
    nan = x.isnan()
    assert int(nan.sum()) == 254 and bool(back[nan].isnan().all()) and not bool(back[~nan].isnan().any())
    assert torch.equal(back[~nan], e4m3_round(x[~nan]))
    inf = x.isinf()
    assert back[inf].tolist() == [448.0 if v > 0 else -448.0 for v in x[inf].tolist()]


def test_the_read_dequantizes_every_byte_exactly() -> None:
    """`kv_read` of an FP8 pool's bytes is torch's E4M3 -> BF16 conversion on all 256 bytes (exact: every E4M3 value is
    a BF16 value; the two NaN bytes stay NaN), and a pool in another dtype is returned as it is."""
    b = torch.arange(256, dtype=torch.int32).to(torch.uint8)
    got = gops.kv_read(b, torch.bfloat16)
    want = b.view(F8).float()
    assert got.dtype == torch.bfloat16 and int(got.isnan().sum()) == 2 == int(want.isnan().sum())
    ok = ~want.isnan()
    assert torch.equal(got[ok].float(), want[ok]) and torch.equal(gops.kv_fp8(got[ok]), b[ok])
    x = torch.randn(4, 8, dtype=torch.bfloat16)
    assert gops.kv_read(x, torch.bfloat16) is x


def _qkv(t: int, h: int, hkv: int, d: int, seed: int) -> tuple[torch.Tensor, dict]:
    g = torch.Generator().manual_seed(seed)
    qkv = (torch.randn(t, (h + hkv) * d, generator=g) * 3).to(torch.bfloat16)
    pos = torch.arange(t, dtype=torch.int64) * 7
    ang = torch.randn(int(pos.max()) + 1, d // 2, generator=g)
    kw = dict(heads=h, kv_heads=hkv, head_dim=d, k_eq_v=True, q_w=(1 + torch.randn(d, generator=g) / 4).bfloat16(),
              k_w=(1 + torch.randn(d, generator=g) / 4).bfloat16(), eps=1e-6, positions=pos,
              cos_cache=ang.cos().bfloat16(), sin_cache=ang.sin().bfloat16())
    return qkv, kw


def test_the_reference_store_writes_the_bf16_value_rounded() -> None:
    """`qkv_store`'s reference (a full layer, `k_eq_v`) into a uint8 pool: q as into a BF16 pool, and at each written
    slot the E4M3 bytes of the value the BF16 pool holds there (V = bf16(k / rms(k)), then E4M3: two roundings, the
    fake-quant's); every other slot untouched."""
    t, h, hkv, d, pages, ps = 12, 4, 2, 64, 5, 8
    qkv, kw = _qkv(t, h, hkv, d, 0)
    page = torch.tensor([1, 1, 2, 3, 3, 3, 4, 4, 2, 1, 4, 3])
    off = torch.tensor([0, 5, 1, 2, 7, 4, 0, 3, 6, 2, 1, 0])
    bf = torch.zeros(pages, ps, hkv, d, dtype=torch.bfloat16)
    f8 = torch.full((pages, ps, hkv, d), 0x55, dtype=torch.uint8)
    q_bf = gops.qkv_store(qkv, **kw, kv_layer=bf, kv_page=page, kv_off=off)
    q_f8 = gops.qkv_store(qkv, **kw, kv_layer=f8, kv_page=page, kv_off=off)
    assert torch.equal(q_bf, q_f8)
    assert torch.equal(f8[page, off], gops.kv_fp8(bf[page, off]))
    assert float(bf[page, off].float().abs().max()) <= 64 ** 0.5  # |k / rms(k)| <= sqrt(D): never saturated
    written = torch.zeros(pages, ps, dtype=torch.bool)
    written[page, off] = True
    assert bool((f8[~written] == 0x55).all())


def test_the_reference_attention_reads_an_fp8_pool_as_its_bf16_values() -> None:
    """`TorchFullAttention` over an FP8 pool equals it over a BF16 pool holding the same values dequantized, bit for
    bit (the one rounding is the store's), and differs from it over the values before the rounding."""
    g = torch.Generator().manual_seed(1)
    h, hkv, d, ps, pages, n = 4, 1, 64, 8, 6, 30
    vals = torch.randn(pages, ps, hkv, d, generator=g)
    bf = (vals / vals.pow(2).mean(-1, keepdim=True).sqrt()).bfloat16()
    f8 = gops.kv_fp8(bf)
    deq = gops.kv_read(f8, torch.bfloat16)
    ang = torch.randn(64, d // 2, generator=g)
    rot = SimpleNamespace(cos={FULL: ang.cos().bfloat16()}, sin={FULL: ang.sin().bfloat16()})
    layer = SimpleNamespace(index=0, k_norm=SimpleNamespace(weight=(1 + torch.randn(d, generator=g) / 4).bfloat16()))
    meta = SimpleNamespace(kv_len_dev=None, n_rows=2, row_start_host=np.array([0, 3]), row_nq_host=np.array([3, 2]),
                           row_kv_len_host=np.array([n, 13]), row_slot_host=np.array([1, 2]),
                           page_table=torch.tensor([[0, 0, 0, 0], [1, 2, 3, 4], [5, 3, 0, 0]], dtype=torch.int32))
    q = (torch.randn(5, h, d, generator=g) / 4).bfloat16()

    def attend(pool: torch.Tensor) -> torch.Tensor:
        p = Gemma4Pools(wkv=torch.zeros(0), fkv=pool[None], page_size=ps, window=8, full_page_size=ps)
        out = torch.empty_like(q)
        TorchFullAttention(p, rot, True).run(meta, layer, q, out)
        return out

    assert torch.equal(attend(f8), attend(deq)) and not torch.equal(attend(f8), attend(bf))


@pytest.fixture(scope="module")
def tiny(tmp_path_factory) -> Path:
    return make_tiny_gemma(tmp_path_factory.mktemp("kv-fp8") / "model")


def test_the_runner_reports_the_full_layers_bytes_and_pages(tiny: Path) -> None:
    """A BF16 CPU runner with `kv_cache_dtype` fp8: its full layers' KV bytes a position are half auto's (the window's
    unchanged), a full page holds twice the positions, its arena's full view is uint8 and the runner reports it; a pool
    config with the BF16 page is refused by name."""
    from llm_mini_infer.runner.cuda.gemma import GemmaRunner

    from ..support.tiny_gemma import gemma_pools

    kw = dict(device="cpu", dtype=torch.bfloat16, max_num_seqs=4, max_num_batched_tokens=64, max_model_len=256)
    bf, f8 = GemmaRunner(tiny, **kw), GemmaRunner(tiny, **kw, kv_cache_dtype="fp8")
    cb, c8 = bf.capabilities(), f8.capabilities()
    assert c8.kv_bytes_per_token * 2 == cb.kv_bytes_per_token == bf.cfg.kv_bytes_per_token(FULL, 2)
    assert c8.window_kv_bytes_per_token == cb.window_kv_bytes_per_token
    assert f8.full_page_size(4) == 2 * bf.full_page_size(4) == 32
    assert (bf.stats["full_kv_dtype"], f8.stats["full_kv_dtype"]) == ("bfloat16", "float8_e4m3fn")
    with pytest.raises(ValueError, match="window pages of 4 positions take full pages of 32, not 16"):
        f8.configure(gemma_pools(budget=64, page_size=4, num_pages=32, max_model_len=256))
    f8.configure(gemma_pools(budget=64, page_size=4, num_pages=32, max_model_len=256, full=32))
    assert f8.arena.fkv.dtype == torch.uint8 and f8.arena.sub == 2 * 8 and f8.pools.full_page_size == 16


def test_the_runner_refuses_what_is_not_built(tiny: Path, tmp_path) -> None:
    from llm_mini_infer.runner.cuda.gemma import GemmaRunner

    kw = dict(device="cpu", max_num_seqs=4, max_num_batched_tokens=64, max_model_len=256)
    with pytest.raises(ValueError, match=r"needs a BF16 runner, not torch\.float32"):
        GemmaRunner(tiny, **kw, kv_cache_dtype="fp8")  # the CPU's default
    with pytest.raises(ValueError, match="auto or fp8"):
        GemmaRunner(tiny, **kw, dtype=torch.bfloat16, kv_cache_dtype="fp8_e5m2")
    kv = make_tiny_gemma(tmp_path / "kv", k_eq_v=False)
    with pytest.raises(ValueError, match="attention_k_eq_v"):
        GemmaRunner(kv, **kw, dtype=torch.bfloat16, kv_cache_dtype="fp8")


def test_the_fake_runner_takes_the_real_geometry(tmp_path) -> None:
    """The fake runner's KV bytes (`model_geometry`, `make_fake_runner`) follow the setting as the CUDA runner's do:
    Gemma 4 31B's full layers 40,960 B a position, 20,480 B in FP8, the window's 819,200 either way; a model other than
    Gemma 4 is refused."""
    from llm_mini_infer.config import FakeRunnerConfig
    from llm_mini_infer.runner.fake import make_fake_runner, model_geometry

    g = tmp_path / "g"
    g.mkdir()
    shutil.copy(ROOT / "tests/data/gemma-4-31b-it-fp8-block.config.json", g / "config.json")
    auto, fp8 = model_geometry(g), model_geometry(g, "fp8")
    assert (auto["kv_bytes_per_token"], fp8["kv_bytes_per_token"]) == (40_960, 20_480)
    assert auto["window_kv_bytes_per_token"] == fp8["window_kv_bytes_per_token"] == 819_200
    caps = make_fake_runner(g, FakeRunnerConfig(), kv_cache_dtype="fp8").capabilities()
    assert caps.kv_bytes_per_token == 20_480 and caps.window_kv_bytes_per_token == 819_200
    q = tmp_path / "q"
    q.mkdir()
    (q / "config.json").write_text(json.dumps({"text_config": {"model_type": "qwen3_5_text", "vocab_size": 512}}))
    for path in (q, tmp_path / "none"):
        with pytest.raises(ValueError, match="Gemma 4's full layers only"):
            make_fake_runner(path, FakeRunnerConfig(), kv_cache_dtype="fp8")
