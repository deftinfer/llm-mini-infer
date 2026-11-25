"""Programmatic dependent launch (kernels/csrc/common.cuh) where it needs no GPU: the library's flag
without a library, the CPU runners leaving it alone, FlashInfer serialized right after a length rewrite, in the full
and the sliding attention and in the startup check of the rewrite (FlashInfer launches with PDL by default and reads
each row's KV length before its wait), and the kernel sources: every kernel launched with the attribute waits first. The
bit-for-bit tests are tests/kernels/test_kernels.py's, tests/models/test_gemma4_fp8_gpu.py's and
tests/kernels/test_pdl_gpu.py's."""

from __future__ import annotations

import re
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from llm_mini_infer.kernels import flashinfer, ops
from llm_mini_infer.kernels.flashinfer import FlashInferAttention, FlashInferWindowAttention


def test_the_flag_without_a_library(monkeypatch) -> None:
    """Without the kernel library (a Mac) there is nothing to launch: the flag stays off."""
    monkeypatch.setattr(ops.build, "load", lambda: None)
    assert ops.set_pdl(True) is False and ops.pdl() is False


def test_a_cpu_runner_leaves_the_library_alone(tmp_path, monkeypatch) -> None:
    """A Gemma runner on the CPU (the torch references) neither loads the kernel library for the flag (on a GPU host
    with a stale cache that would build it) nor changes the flag a CUDA runner set; it reports PDL off."""
    from ..support.tiny_gemma import gemma_cpu_runner, gemma_pools, make_tiny_gemma

    model = make_tiny_gemma(tmp_path / "g", seed=1)

    def load():
        raise AssertionError("a CPU runner loaded the kernel library")

    monkeypatch.setattr(ops.build, "load", load)
    monkeypatch.setattr(ops, "_pdl", True)
    r = gemma_cpu_runner(model, gemma_pools(), pdl=True)
    assert r.stats["pdl"] is False and ops.pdl() is True
    r.shutdown()


def test_a_qwen_cpu_runner_leaves_the_library_alone(tmp_path, monkeypatch) -> None:
    """The Qwen runner on the CPU likewise: no kernel library loaded for the flag, the flag as a CUDA runner set it, PDL
    reported off."""
    from ..support.tiny_qwen import cpu_runner, make_tiny_model, pools

    model = make_tiny_model(tmp_path / "q", seed=1)

    def load():
        raise AssertionError("a CPU runner loaded the kernel library")

    monkeypatch.setattr(ops.build, "load", load)
    monkeypatch.setattr(ops, "_pdl", True)
    r = cpu_runner(model, pools(), pdl=True)
    assert r.stats["pdl"] is False and ops.pdl() is True
    r.shutdown()


def test_the_sliding_attention_waits_out_a_length_rewrite() -> None:
    """MTP's non-decode steps rewrite the plan's last-page lengths on the device (`fix_len`) just before FlashInfer
    runs: that run launches without PDL, so FlashInfer cannot read the lengths before the rewrite lands; other runs
    keep FlashInfer's default."""
    calls: list[tuple[str, dict]] = []

    class Plan:
        def run(self, q, kv, **kw):
            calls.append(("run", kw))

    plan = Plan()
    stub = SimpleNamespace(pools=SimpleNamespace(wkv={3: "kv"}),
                           fix_lengths=lambda w, rows, true_len: calls.append(("fix", {"rows": rows})))
    stub.run_fixed = partial(FlashInferAttention.run_fixed, stub)
    layer = SimpleNamespace(index=3)
    for fix in (True, False):
        calls.clear()
        meta = SimpleNamespace(fix_len=fix, sliding_plan=plan, n_rows=2, kv_named_dev=None, bidi_plan=None)
        FlashInferWindowAttention.run(stub, meta, layer, "q", "out")
        if fix:
            assert calls == [("fix", {"rows": 2}), ("run", {"out": "out", "enable_pdl": False})]
        else:
            assert calls == [("run", {"out": "out"})]


def test_the_full_attention_waits_out_a_length_rewrite() -> None:
    """The MTP drafter's chain on the Qwen runner's non-decode steps rewrites its full attention's last-page lengths on
    the device (`fix_len`) just before FlashInfer runs: that run launches without PDL, as the sliding attention's; other
    runs keep FlashInfer's default."""
    calls: list[tuple[str, dict]] = []

    class Plan:
        def run(self, q, kv, **kw):
            calls.append(("run", kw))

    plan = Plan()
    stub = SimpleNamespace(pools=SimpleNamespace(kv={3: "kv"}),
                           fix_lengths=lambda w, rows, true_len: calls.append(("fix", {"rows": rows})))
    stub.run_fixed = partial(FlashInferAttention.run_fixed, stub)
    for fix in (True, False):
        calls.clear()
        meta = SimpleNamespace(fix_len=fix, attn_plan=plan, n_rows=2, kv_len_dev=None)
        FlashInferAttention.run(stub, meta, 3, "q", "out")
        if fix:
            assert calls == [("fix", {"rows": 2}), ("run", {"out": "out", "enable_pdl": False})]
        else:
            assert calls == [("run", {"out": "out"})]


def test_the_startup_check_waits_out_its_length_rewrite(monkeypatch) -> None:
    """`check_lengths` (the startup check of upper-bound planning, MTP) rewrites one plan's lengths and runs it at once:
    that run launches without PDL as the serving path's does; the exact plan's keeps FlashInfer's default."""
    calls: list[tuple[str, dict]] = []

    class Plan:
        def __init__(self, fix: bool):
            self.fix = fix

        def run(self, q, kv, *, out, **kw):
            calls.append(("run", {"fix": self.fix, **kw}))
            out.zero_()

    class Attn:
        pools = SimpleNamespace(page_size=4)
        int_workspace = torch.empty(0)
        qo_per_row, hq, hkv, d, dtype, max_rows = 1, 2, 1, 8, torch.float32, 0

        def plan(self, meta, *, lb, **kw):
            return Plan(lb is not None)

        def fix_lengths(self, w, rows, true_len):
            calls.append(("fix", {"rows": rows}))

        run_fixed = FlashInferAttention.run_fixed

    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None)
    # pinning opens a CUDA context on a GPU host
    monkeypatch.setattr(flashinfer, "_pinned32", lambda a: torch.as_tensor(np.asarray(a, np.int32)))
    assert flashinfer.check_lengths(Attn(), torch.float32) is None
    assert calls == [("fix", {"rows": 3}), ("run", {"fix": True, "enable_pdl": False}), ("run", {"fix": False})]


# -- the kernels' sources ------------------------------------------------------------------------------------------

CSRC = Path(ops.__file__).resolve().parent / "csrc"
# FlashInfer's attention, which launches with PDL by its own default, follows these: they trigger only with the flag on
TRIGGER_ON_FLAG = {"g4_qkv_kernel", "qk_norm_rope_kernel"}
# Launched serialized (`<<<>>>`): outside the CUDA graphs (the folds, the weights' dequantization), or followed by
# FlashInfer's vision attention (the towers' attention inputs)
SERIALIZED = {"conv_fold_kernel", "gdn_fold_kernel", "dequant_fp8_kernel", "vision_qkv_rope_kernel",
              "g4_vision_qkv_kernel"}


def _sources() -> str:
    """Every .cu file of the kernel library, its comments removed."""
    text = "\n".join(p.read_text(encoding="utf-8") for p in sorted(CSRC.glob("*.cu")))
    return re.sub(r"//[^\n]*", "", re.sub(r"/\*.*?\*/", "", text, flags=re.S))


def _first_argument(src: str, at: int) -> str:
    """The first argument of the call whose arguments start at `at` (up to its first comma outside parentheses and
    template brackets), whitespace collapsed."""
    depth = 0
    for i in range(at, len(src)):
        if src[i] in "(<":
            depth += 1
        elif src[i] in ")>":
            depth -= 1
        elif src[i] == "," and depth == 0:
            return " ".join(src[at:i].split())
    raise AssertionError(f"a call without arguments at {at}")


def _first_statements(src: str, kernel: str) -> list[str]:
    """The first two statements of the kernel's body, whitespace collapsed."""
    (m,) = re.finditer(r"__global__\s[^;{]*?\b" + kernel + r"\s*\(", src)
    depth, i = 1, m.end()
    while depth:
        depth += {"(": 1, ")": -1}.get(src[i], 0)
        i += 1
    body = src[src.index("{", i) + 1:]
    return [" ".join(s.split()) + ";" for s in body.split(";", 2)[:2]]


def test_every_pdl_launch_waits_first() -> None:
    """Every kernel launched through `lmk_launch`, which adds the PDL attribute while the library's flag is on, waits
    for the kernel before it as its first statement (common.cuh): before any read, write or early return, in every CTA,
    so its bits are the serialized launch's. Its second statement lets the next one launch, unconditionally but in the
    kernels FlashInfer follows (`TRIGGER_ON_FLAG`): those alone are templated on kTrigger, launched by the flag's value
    and trigger only with it on. Every other kernel launches serialized (`SERIALIZED`)."""
    src = _sources()
    launched: dict[str, set[str]] = {}
    for m in re.finditer(r"\blmk_launch\(", src):
        arg = _first_argument(src, m.end())
        names = set(re.findall(r"\b(\w+_kernel)\b", arg))
        assert len(names) == 1, arg
        launched.setdefault(names.pop(), set()).add(arg)
    defined = re.findall(r"__global__\s[^;{]*?\b(\w+_kernel)\s*\(", src)
    assert len(defined) == src.count("__global__") == len(set(defined))
    serialized = set(re.findall(r"\b(\w+_kernel)\s*(?:<[^<>;]*>)?\s*<<<", src))
    assert serialized == SERIALIZED and not serialized & launched.keys()
    assert launched.keys() | serialized == set(defined)
    templated = re.findall(r"template\s*<\s*bool\s+kTrigger\s*>\s*__global__\s[^;{]*?\b(\w+_kernel)\s*\(", src)
    assert set(templated) == TRIGGER_ON_FLAG
    for name, args in sorted(launched.items()):
        wait, trigger = _first_statements(src, name)
        assert wait == "lmk_pdl_wait();", (name, wait)
        if name in TRIGGER_ON_FLAG:
            assert args == {f"lmk_pdl_flag() ? {name}<true> : {name}<false>"}, (name, args)
            assert trigger == "if (kTrigger) lmk_pdl_trigger();", (name, trigger)
        else:
            assert trigger == "lmk_pdl_trigger();", (name, trigger)
