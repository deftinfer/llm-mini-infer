"""The profiling harnesses: tests/scripts/profile_verify.py (a verify step of Gemma 4 31B FP8 with its assistant under
structured output) and tests/scripts/profile_tower.py (the vision tower on one image), their logic on CPU with the tiny
Gemma models: the permissive grammar against the scheduler, the subset grammar (every mask the same U tokens), the
request setup on a shared cached prefix, the spy that makes every measured step N masked verify rows and records the
grammar-aware heads each took (the runner's head counters), the switches to the runner, the kernel categories, the
reference image's sizing and the tower call through the runner's `ImageEncoder.encode`. On a CUDA GPU with the kernel
library, both harnesses also run their CUPTI profile on tiny models (decode graphs split by the masks, the heads'
sampling graphs, the tower's graph and eager paths), and the categories are checked against the kernels this GPU
launches: run this file on the GPU before profiling the 31B."""

from __future__ import annotations

import argparse
import io
import re
import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from llm_mini_infer.engine.messages import FinishReason

from ..scripts import profile_tower as pt
from ..scripts import profile_verify as pv
from ..support.paths import MODELS_DIR
from ..support.tiny_gemma import gemma_cpu_runner, gemma_pools, make_tiny_assistant, make_tiny_gemma, make_tiny_gemma_mm

GEMMA = MODELS_DIR / "gemma-4-31b-it"
K = 3
PREFIX, TAIL = 40, 6  # the tiny model's grid is its window page (4): the prefix's hint lands on it
cuda = pytest.mark.gpu(reason="needs a CUDA GPU and the kernel library")


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    d = tmp_path_factory.mktemp("profile")
    target = make_tiny_gemma(d / "target", window=8, seed=5)
    return target, make_tiny_assistant(d / "assistant", target, seed=6)


def harness(pair, max_num_seqs: int = 4, k: int = K, grammar: str = "allow-all", **heads):
    """`heads`: the GemmaRunner's grammar-aware heads (sparse_head, sparse_head_rows, draft_universe; default off)."""
    target, assistant = pair
    pc = gemma_pools(max_num_seqs=max_num_seqs, budget=64, page_size=4, num_pages=512, max_model_len=256, mtp_k=k)
    runner = gemma_cpu_runner(target, pc, mtp_k=k, assistant=assistant if k else None, **heads)
    h = pv.Harness(runner, pc, eos=pv.eos_ids(target), grammar=grammar)
    return h, pv.StepSpy(runner), pv.Prompts(PREFIX, TAIL, h.caps.vocab_size, seed=1)


# -- the permissive grammar -------------------------------------------------------------------------------------


def test_allow_all_matcher_takes_every_token_and_never_stops() -> None:
    m = pv.AllowAll(512).instance(pv.AllowAll(512).compile("anything"))
    assert m.consume_tokens([1, 2, 3]) and m.try_consume_tokens([4, 5]) == 2 and m.validate_tokens([6, 7, 8]) == 3
    assert not m.is_stopped() and not m.is_error() and m.get_error() == ""
    m.rollback(2)
    assert m.n == 3 and m.deep_copy().n == 3
    with pytest.raises(ValueError):
        m.rollback(4)


def test_allow_all_masks_are_all_ones_along_drafts() -> None:
    """A single row's mask, and 1 + len(drafts) masks for a row with drafts (llguidance's rule, every draft taken);
    rows not named stay as they were."""
    buf = np.zeros((8, 16), np.int32)
    m = pv.AllowAllMatcher()
    pv.AllowAll(512).fill([(m, 0)], [(m, 2, [5, 6, 7])], buf)
    assert (buf[0] == -1).all() and (buf[1] == 0).all() and (buf[2:6] == -1).all() and (buf[6:] == 0).all()
    with pytest.raises(ValueError):
        pv.AllowAll(512).fill([], [(m, 0, [])], buf)


def test_requests_run_to_their_lengths_under_the_permissive_grammar(pair) -> None:
    """Every row MASKED under MTP (masks along the drafts, the matchers advanced over what each step delivers):
    the requests reach max_tokens, never an error, and the scheduler's accounting holds."""
    h, _, prompts = harness(pair)
    want = {h.add(prompts.next(), max_tokens=n): n for n in (12, 9, 15)}
    h.run_until(lambda: not h.core.busy)
    assert {r: h.got[r] for r in want} == want
    assert all(h.finished[r] == FinishReason.LENGTH for r in want)
    assert h.grammars.errors == 0 and h.grammars.mask_rows >= sum(want.values())
    h.sched.check_accounting()


# -- the subset grammar -------------------------------------------------------------------------------------------


def test_subset_grammar_allows_the_same_tokens_in_every_mask() -> None:
    """subset:<U>: the middle id of each of U equal runs of the vocabulary without its EOS ids, the same words in
    every mask, along the drafts too (as AllowAll's, so every step's union is those U tokens); rows not named stay as
    they were; no EOS ids of its own; a set larger than the vocabulary without its EOS ids is refused."""
    g = pv.SubsetGrammar(512, 8, exclude=(1, 511))
    assert g.name == "subset:8" and g.eos == () and g.vocab_size == 512
    assert g.ids.tolist() == [32, 96, 160, 224, 287, 351, 415, 479]
    assert np.flatnonzero(np.unpackbits(g.mask.view(np.uint8), bitorder="little")).tolist() == g.ids.tolist()
    buf = np.zeros((8, 16), np.int32)
    m = g.instance(g.compile("anything"))
    g.fill([(m, 0)], [(m, 2, [5, 6, 7])], buf)
    assert (buf[[0, 2, 3, 4, 5]] == g.mask).all() and not buf[1].any() and not buf[6:].any()
    with pytest.raises(ValueError):
        g.fill([], [(m, 0, [])], buf)
    assert pv.SubsetGrammar(512, 510, exclude=(1, 511)).allowed.sum() == 510
    with pytest.raises(ValueError, match="from 1 to 510 tokens"):
        pv.SubsetGrammar(512, 511, exclude=(1, 511))


def test_subset_matcher_fails_on_a_token_outside_its_set() -> None:
    """It never stops and takes every draft (`validate_tokens`: the scheduler leaves no mask to its EOS-only rule),
    but a consumed token outside the set is the error state, as llguidance's on a token it rejects: a sample outside
    the union fails its request. `try_consume_tokens` stops before it without the error."""
    g = pv.SubsetGrammar(512, 8)
    m = g.instance(g.compile("x"))
    a, b = g.ids[:2].tolist()
    assert m.consume_tokens([a, b]) and m.validate_tokens([a, 5, b]) == 3 and m.n == 2
    assert m.try_consume_tokens([a, 5, b]) == 1 and m.n == 3 and not m.is_error()
    m.rollback(1)
    c = m.deep_copy()
    assert not m.consume_tokens([b, 5]) and m.is_error() and m.get_error().startswith("token 5 ")
    assert not m.consume_tokens([a]) and m.try_consume_tokens([a]) == 0 and not m.is_stopped()  # the error stays
    assert not c.is_error() and c.consume_tokens([a]) and c.n == 3  # a copy made before it does not share it


def test_grammar_option_names_a_backend() -> None:
    assert [pv.grammar_spec(s) for s in ("allow-all", "subset:512")] == ["allow-all", "subset:512"]
    for bad in ("subset", "subset:", "subset:0", "subset:-3", "subset:x", "allow", "subset:8:1"):
        with pytest.raises(argparse.ArgumentTypeError):
            pv.grammar_spec(bad)
    assert type(pv.grammar_backend("allow-all", 512, eos=(511,))) is pv.AllowAll
    g = pv.grammar_backend("subset:16", 512, eos=(511,))
    assert isinstance(g, pv.SubsetGrammar) and len(g.ids) == 16 and 511 not in g.ids.tolist()


def test_requests_run_to_their_lengths_under_the_subset_grammar(pair) -> None:
    """Every token delivered is one of the set's (a matcher would fail its request otherwise), on the dense head and
    on the grammar-aware heads; the requests reach max_tokens and the scheduler's accounting holds."""
    for heads in ({}, {"sparse_head": True, "sparse_head_rows": 128, "draft_universe": True}):
        h, _, prompts = harness(pair, grammar="subset:8", **heads)
        want = {h.add(prompts.next(), max_tokens=n): n for n in (12, 9, 15)}
        h.run_until(lambda core=h.core: not core.busy)
        assert {r: h.got[r] for r in want} == want and h.grammars.errors == 0, heads
        assert all(h.finished[r] == FinishReason.LENGTH for r in want)
        h.sched.check_accounting()


# -- the request setup and the measured steps -----------------------------------------------------------------


def test_setup_reaches_n_decoding_rows_on_the_shared_prefix(pair) -> None:
    """The first request runs alone to its first token with a hint at the prefix; every later one hits the prefix
    there and shares its full pages; then every row decodes, and the next steps are N masked verify rows."""
    h, spy, prompts = harness(pair)
    setup = pv.setup_rows(h, prompts, 3, max_tokens=100, decode_before=2)
    first, *rest = h.live
    assert len(h.live) == 3 and h.decoding()
    assert setup["cached"] == {str(first): 0, **{str(r): PREFIX for r in rest}}
    rows = h.rows(PREFIX)
    assert rows["shared_full_pages"] == PREFIX // h.pc.page_size and rows["cached"] == [0, PREFIX, PREFIX]
    _, plans = pv.measure_steps(h, spy, 3, K, 4)
    assert plans["steps"] == 4 and plans["modes"] == {"eager/12/split": 4}  # 3 rows x (1 + k) tokens, on CPU


@pytest.mark.parametrize("fresh", [True, False])
def test_measured_steps_are_exactly_n_masked_device_rows(pair, fresh) -> None:
    """The whole run without a profiler (mode none), 3 rows then 2 (one aborted): every measured step passes the spy,
    the measured rows hit the cached prefix (fresh ones too), and the results line up with the steps."""
    h, spy, prompts = harness(pair)
    res = pv.profile(h, spy, prompts, rows=[2, 3], k=K, steps=4, warm_s=0.0, min_warm_steps=4, decode_before=2,
                     max_tokens=120, fresh=fresh, mode="none", log=lambda _: None)
    for n in (3, 2):
        r = res[str(n)]
        assert r["unprofiled"]["plans"]["steps"] == 5 and r["unprofiled"]["plans"]["rows"] == n
        assert r["unprofiled"]["busy_ms"]["n"] == 4 and r["unprofiled"]["tokens_per_step"]["min"] >= n
        assert r["context"]["cached"] == ([PREFIX] * n if fresh else [0] + [PREFIX] * (n - 1))
    assert res["2"]["aborted"] == [res["3"]["context"]["rids"][-1]]
    assert res["grammar"]["errors"] == 0
    h.sched.check_accounting()


def test_a_step_that_is_not_n_masked_verify_rows_fails_the_measurement(pair) -> None:
    """A request admitted into the measured steps (a prefill row next to the decode rows), or a row count the step
    does not have, is an error."""
    h, spy, prompts = harness(pair)
    pv.setup_rows(h, prompts, 2, max_tokens=100, decode_before=1)
    with pytest.raises(AssertionError, match="not verify steps of 3 masked rows"):
        pv.measure_steps(h, spy, 3, K, 2)
    h.add(prompts.next(), max_tokens=100)
    with pytest.raises(AssertionError, match=r"admit 1 \(want 0\)"):
        pv.measure_steps(h, spy, 2, K, 2)


def test_rows_that_would_finish_inside_the_measurement_are_replaced_or_refused(pair) -> None:
    """A warm-up replaces the rows with fewer tokens left than the measurement may take (and those that finish), so
    the measured steps keep their N rows; the check before the measurement refuses such a row."""
    h, _, prompts = harness(pair)
    pv.setup_rows(h, prompts, 2, max_tokens=40, decode_before=0)
    pv.check_budget(h, 2, K)  # (2 + 2) x (1 + k) = 16 tokens
    h.run_until(lambda: min(40 - h.got[r] for r in h.live) < 16)
    with pytest.raises(RuntimeError, match="tokens left"):
        pv.check_budget(h, 2, K)
    w = pv.warm(h, prompts, 2, 0.0, min_steps=4, max_tokens=40, keep=16)
    assert w["replaced"] >= 1 and len(h.live) == 2 and h.decoding()
    pv.check_budget(h, 2, K)
    with pytest.raises(ValueError, match="cannot keep"):
        pv.warm(h, prompts, 2, 0.0, min_steps=1, max_tokens=40, keep=21)


def test_a_warm_up_ends_with_every_row_holding_the_measurements_tokens(pair) -> None:
    """A row that falls below `keep` in the warm-up's last step is replaced too (a step delivers a token a row or
    more): the warm-up cannot end on it, which `check_budget` would then refuse (--no-fresh)."""
    h, _, prompts = harness(pair)
    pv.setup_rows(h, prompts, 2, max_tokens=40, decode_before=0)
    h.run_until(lambda: min(40 - h.got[r] for r in h.live) <= 23)
    keep = min(h.sched.reqs[(0, r)].max_out - h.got[r] for r in h.live) - 3  # a row short after 4 steps; <= 40 / 2
    w = pv.warm(h, prompts, 2, 0.0, min_steps=4, max_tokens=40, keep=keep)
    assert w["replaced"] >= 1 and h.decoding()
    assert all(h.sched.reqs[(0, r)].max_out - h.got[r] >= keep for r in h.live)


def test_a_failed_row_count_keeps_the_ones_measured_before(pair, monkeypatch) -> None:
    """`profile` fills the caller's dict as it goes: the first row count's measurement stays when the second fails, and
    what the second did before it failed (main() writes it with the error)."""
    h, spy, prompts = harness(pair)
    budget, calls = pv.check_budget, []

    def failing(*a) -> None:
        calls.append(a)
        if len(calls) == 2:
            raise RuntimeError("the second row count fails")
        budget(*a)

    monkeypatch.setattr(pv, "check_budget", failing)
    res: dict = {}
    with pytest.raises(RuntimeError, match="second row count"):
        pv.profile(h, spy, prompts, rows=[3, 2], k=K, steps=2, warm_s=0.0, min_warm_steps=3, decode_before=1,
                   max_tokens=100, fresh=True, mode="none", log=lambda _: None, out=res)
    assert res["3"]["unprofiled"]["plans"]["steps"] == 3 and "setup" in res
    assert res["2"]["aborted"] and "fresh" in res["2"] and "unprofiled" not in res["2"]


def test_torch_mode_on_cpu_fills_the_records_and_the_table(pair, monkeypatch) -> None:
    """The torch mode's windows on CPU (the profiler records no kernel there): the profiled steps pass the spy too, and
    the summary table has each measured row count and category."""
    monkeypatch.setattr(pv, "activities", lambda: [torch.profiler.ProfilerActivity.CPU])  # CUPTI opens a CUDA context
    h, spy, prompts = harness(pair)
    res = pv.profile(h, spy, prompts, rows=[2], k=K, steps=2, warm_s=0.0, min_warm_steps=3, decode_before=1,
                     max_tokens=100, fresh=True, mode="torch", sampler=pv.PowerSampler(), log=lambda _: None)
    t = res["2"]["torch"]
    assert t["plans"]["steps"] == 2 and t["kernels"] == 0 and res["2"]["kernel_share_of_busy"] == 0
    assert t["timeline"]["device_busy_ms"] == 0 and "no split" in t["timeline"]["parts"]
    table = pv.table(res, [2])
    assert "2 rows" in table and "busy ms/step" in table and all(c in table for c in t["categories"])


def test_ncu_mode_brackets_exactly_the_measured_steps(pair, monkeypatch) -> None:
    """ncu's window per row count: synchronized, an NVTX range named for it, cudaProfilerStart, the K steps (no
    profiler, no unprofiled run), synchronized, cudaProfilerStop, the range closed."""
    calls: list[str] = []

    class Runtime:
        def cudaProfilerStart(self) -> None:
            calls.append(f"start after {spy.on}")

        def cudaProfilerStop(self) -> None:
            calls.append(f"stop with {len(spy.plans)} plans")

    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: calls.append("sync"))
    monkeypatch.setattr(torch.cuda.nvtx, "range_push", lambda label: calls.append(f"push {label}"))
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", lambda: calls.append("pop"))
    monkeypatch.setattr(torch.cuda, "cudart", Runtime)
    h, spy, prompts = harness(pair)
    res = pv.profile(h, spy, prompts, rows=[3, 2], k=K, steps=2, warm_s=0.0, min_warm_steps=3, decode_before=1,
                     max_tokens=100, fresh=True, mode="ncu", log=lambda _: None)
    assert calls == [c for n in (3, 2) for c in ("sync", f"push verify_b{n}", "start after False", "sync",
                                                 "stop with 2 plans", "pop")]
    assert [res[n]["ncu"]["plans"]["steps"] for n in ("3", "2")] == [2, 2]
    assert "unprofiled" not in res["3"] and "torch" not in res["3"]


def test_plan_problems_name_what_differs() -> None:
    good = {"rows": 4, "device": 4, "sample": 4, "masked": 4, "precise": 0, "drafts": [3], "tokens": 16, "admit": 0,
            "free": 0, "fold": 0, "copies": 0, "ckpt": 0, "encode": 0, "spans": 0, "pages": 2, "wpages": 3}
    assert pv.verify_problems(good, 4, 3) == []
    assert pv.verify_problems({**good, "masked": 3, "free": 1}, 4, 3) == ["masked 3 (want 4)", "free 1 (want 0)"]


# -- the grammar-aware heads each measured step took ----------------------------------------------------------------

HEADS = {"sparse_head": True, "sparse_head_rows": 128, "draft_universe": True}  # 128 rows: an all-ones mask is over


@pytest.mark.parametrize(("grammar", "heads", "path", "union", "universe"), [
    ("subset:8", HEADS, "sparse", 8, 8),
    ("subset:8", {**HEADS, "draft_universe": False}, "sparse", 8, 0),
    ("subset:8", {**HEADS, "sparse_head": False}, "dense_off", 8, 8),
    ("subset:8", {}, "dense_off", None, 0),  # both off: no union taken
    ("subset:200", HEADS, "dense_over_cap", 200, 0),
    ("allow-all", HEADS, "dense_unconstrained", 512, 0),  # all ones: the tiny vocabulary
], ids=["both", "sparse", "universe", "off", "over-cap", "allow-all"])
def test_each_measured_step_records_its_head(pair, grammar, heads, path, union, universe) -> None:
    """The runner's head counters around each measured step's sampling: the path its sampling took, its union (the
    subset's U tokens, or the whole vocabulary under all-ones masks), the universe and the steps drafting over it; the
    universe grows once, in the setup's first step (its rebuild), never in a measurement."""
    h, spy, prompts = harness(pair, grammar=grammar, **heads)
    res = pv.profile(h, spy, prompts, rows=[3, 2], k=K, steps=4, warm_s=0.0, min_warm_steps=4, decode_before=2,
                     max_tokens=120, fresh=True, mode="none", log=lambda _: None)
    for n in ("3", "2"):
        got = res[n]["unprofiled"]["plans"]["heads"]
        assert (got["path"], got["universe"], got["universe_steps"], got["universe_rebuilds"]) == (
            path, universe, 5 if universe else 0, 0)
        assert (got["union"] or {}).get("mean") == union and (got["union"] or {}).get("n", 5) == 5
    assert res["heads"]["universe_rebuilds"] == (1 if universe else 0) and res["heads"]["universe"] == universe
    assert res["grammar"] == {"backend": grammar, "masks": h.grammars.mask_rows, "errors": 0}
    assert f"head {path}" in pv.line(2, res["2"]) and path in pv.table(res, [3, 2])


def test_a_window_whose_steps_took_two_heads_fails_the_measurement(pair) -> None:
    """A window mixing two heads would average them: an error (a stand-in for the runner's counters alternating the
    sparse head and the over-cap one). A runner without the counters records none."""
    h, _, prompts = harness(pair)
    runner, sample = h.runner, h.runner.launch_sample
    c = dict.fromkeys(pv.HEAD_TOTALS, 0)

    def counting(hd, n_masks: int = 0, **internal) -> None:
        sample(hd, n_masks, **internal)
        if not internal:
            c["sparse" if c["sparse"] <= c["dense_over_cap"] else "dense_over_cap"] += 1

    runner.launch_sample = counting
    runner.head_counters = lambda: types.SimpleNamespace(**c)  # no union_counts: none measured
    spy = pv.StepSpy(runner)
    pv.setup_rows(h, prompts, 2, max_tokens=100, decode_before=1)
    with pytest.raises(AssertionError, match=r"the 3 measured steps did not all take one grammar-aware head: "
                                             r"\{'(sparse|dense_over_cap)': [12], '(sparse|dense_over_cap)': [12]\}"):
        pv.measure_steps(h, spy, 2, K, 3)
    h2, _, prompts2 = harness(pair)
    h2.runner.head_counters = None
    spy2 = pv.StepSpy(h2.runner)
    pv.setup_rows(h2, prompts2, 2, max_tokens=100, decode_before=1)
    _, plans = pv.measure_steps(h2, spy2, 2, K, 3)
    assert plans["steps"] == 3 and plans["heads"] is None and spy2.heads == []


def test_cli_passes_the_heads_to_the_runner(monkeypatch, tmp_path) -> None:
    """--sparse-head and --draft-universe, both on by default (the config derives them on for Gemma 4 under structured
    output), go to make_runner; a malformed --grammar stops before any model loads."""
    seen: list[dict] = []

    class Loaded(Exception):
        pass

    def make_runner(model: str, **kw) -> None:
        seen.append(kw)
        raise Loaded

    monkeypatch.setattr(pv, "make_runner", make_runner)
    base = ["profile_verify.py", "--model", "m", "--assistant", "a", "--out", str(tmp_path / "v.json"), "--k", "5"]
    for extra, want in (([], (True, True)), (["--no-sparse-head", "--grammar", "subset:512"], (False, True)),
                        (["--no-draft-universe", "--grammar", "allow-all"], (True, False))):
        monkeypatch.setattr(sys, "argv", base + extra)
        with pytest.raises(Loaded):
            pv.main()
        assert (seen[-1]["sparse_head"], seen[-1]["draft_universe"], seen[-1]["mtp_k"]) == (*want, 5)
    monkeypatch.setattr(sys, "argv", [*base, "--grammar", "subset:0"])
    with pytest.raises(SystemExit):
        pv.main()
    assert len(seen) == 3


def test_cli_takes_row_counts(monkeypatch, tmp_path) -> None:
    """--rows takes any comma list of row counts (each once, measured descending: 8,7,6,5,4,3,2,1); a row count out of
    1..--max-num-seqs stops before any model loads."""
    seen: list[dict] = []

    class Loaded(Exception):
        pass

    def make_runner(model: str, **kw) -> None:
        seen.append(kw)
        raise Loaded

    monkeypatch.setattr(pv, "make_runner", make_runner)
    base = ["profile_verify.py", "--model", "m", "--assistant", "a", "--out", str(tmp_path / "v.json"), "--k", "5"]
    for extra in ([], ["--rows", "8,7,6,5,4,3,2,1"], ["--rows", "1,8,4"]):
        monkeypatch.setattr(sys, "argv", base + extra)
        with pytest.raises(Loaded):
            pv.main()
        assert (seen[-1]["mtp_k"], seen[-1]["sparse_head"]) == (5, True)
    for bad in (["--rows", "0,4"], ["--rows", "65"], ["--rows", "8,x"]):
        monkeypatch.setattr(sys, "argv", base + bad)
        with pytest.raises(SystemExit):
            pv.main()
    assert len(seen) == 3
    assert pv.row_counts("8,7,6,5,4,3,2,1", 64) == [8, 7, 6, 5, 4, 3, 2, 1] == pv.row_counts("1,2,3,4,5,6,7,8,8", 8)
    for spec in ("0,4", "9", "8,x", "", "8,,4"):
        with pytest.raises(ValueError):
            pv.row_counts(spec, 8)


# -- kernel categories ---------------------------------------------------------------------------------------------

CUTLASS_FP8 = ("void cutlass::device_kernel<lmk_fp8::cutlass_3x_gemm_fp8_blockwise<cutlass::bfloat16_t, 128, 1, 128, "
               "cute::tuple<cute::C<128>, cute::C<32>, cute::C<128> >, "
               "cute::tuple<cute::C<1>, cute::C<1>, cute::C<1> >, cutlass::epilogue::collective::EpilogueScheduleAuto, "
               "cutlass::gemm::KernelTmaWarpSpecializedBlockwiseCooperativeSm120, true>::GemmKernel>("
               "lmk_fp8::cutlass_3x_gemm_fp8_blockwise<cutlass::bfloat16_t>::GemmKernel::Params)")
VERIFY_NAMES = {
    CUTLASS_FP8: "fp8_gemm",
    "void flashinfer::gemm::GemmGroupwiseKernel<cutlass::float_e4m3_t>(...)": "fp8_gemm",
    "nvjet_tst_192x112_64x5_2x1_v_bz_coopB_TNT": "bf16_gemm",
    "void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x1_tn_align8>(cutlass_80_wmma_"
    "tensorop_bf16_s161616gemm_bf16_32x32_128x1_tn_align8::Params)": "bf16_gemm",
    "void gemv2T_kernel_val<int, int, __nv_bfloat16, __nv_bfloat16, float, 128, 16, 4, 4, false, false>(...)":
        "bf16_gemm",
    "void cublasLt::splitKreduce_kernel<32, 16, int, __nv_bfloat16, __nv_bfloat16, float, __nv_bfloat16, false>(...)":
        "bf16_gemm",
    "void (anonymous namespace)::linear_small_kernel<8>(__nv_bfloat16 const*, long, __nv_bfloat16 const*, long, "
    "__nv_bfloat16*, long, int, int)": "bf16_gemm",
    "void flashinfer::BatchPrefillWithPagedKVCacheKernel<flashinfer::KernelTraits<(flashinfer::MaskMode)1, 16u, 1u, "
    "1u, 16u, 2u, 1u, 4u>, flashinfer::DefaultAttention<false, true, false, false>>(...)": "sliding_attention",
    "void flashinfer::PersistentVariableLengthMergeStatesKernel<8u, 32u, 4u, 4u, __nv_bfloat16, __nv_bfloat16, int>"
    "(__nv_bfloat16*, float*, int*, __nv_bfloat16*, float*, unsigned int, unsigned int*, unsigned int)":
        "sliding_attention",
    "void (anonymous namespace)::attn512_kernel<true>(__nv_bfloat16 const*, int, int, int, ...)": "full_attention",
    "(anonymous namespace)::attn512_merge_kernel(float const*, float const*, int, __nv_bfloat16*)": "full_attention",
    "(anonymous namespace)::g4_sandwich_quant_kernel(__nv_bfloat16 const*, long, __nv_fp8_e4m3*, float*, long)":
        "norm_act_quant",
    "(anonymous namespace)::gelu_tanh_mul_quant_kernel(__nv_bfloat16 const*, long, unsigned char*, float*, long, int,"
    " int)": "norm_act_quant",
    "(anonymous namespace)::g4_qkv_kernel(__nv_bfloat16 const*, long, int, ...)": "norm_act_quant",
    "(anonymous namespace)::quant_fp8_kernel(__nv_bfloat16 const*, long, unsigned char*, float*, long, int, int)":
        "norm_act_quant",
    "(anonymous namespace)::sample_kernel(float const*, long, int, long const*, long const*, float const*)": "sampler",
    "Memcpy HtoD (Pinned -> Device)": "memcpy",
    "Memset (Device)": "memcpy",
    # names from profiles of Gemma 4 on SM120: cuBLAS's GEMV, our templates
    "std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, "
    "__nv_bfloat16, float, false, true, true, false, 7, false, cublasGemvParamsEx<int>>(...)": "bf16_gemm",
    "void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8>(...)": "bf16_gemm",
    "void (anonymous namespace)::attn512_kernel<false>(__nv_bfloat16 const*, int, int, int, ...)": "full_attention",
    "void (anonymous namespace)::g4_sandwich_quant_kernel<4>(__nv_bfloat16 const*, long, ...)": "norm_act_quant",
    "(anonymous namespace)::g4_rmsnorm_quant_kernel(__nv_bfloat16 const*, long, __nv_bfloat16 const*, ...)":
        "norm_act_quant",
    "Memcpy DtoD (Device -> Device)": "memcpy",
    "Memcpy DtoH (Device -> Pinned)": "memcpy",
    "void at::native::vectorized_elementwise_kernel<4, at::native::tanh_kernel_cuda(at::TensorIteratorBase&)>(...)":
        "other",
    # torch.index_select's: the step's small selects and the grammar-aware heads' gathers (the sparse head's 1,024 rows
    # of the tied head: the large-index kernel); torch's other gathers stay "other"
    "void at::native::(anonymous namespace)::indexSelectSmallIndex<c10::BFloat16, long, unsigned int, 2, 2, -2>(...)":
        "gather",
    "void at::native::(anonymous namespace)::indexSelectSmallIndex<long, long, unsigned int, 1, 1, -2>(at::cuda::"
    "detail::TensorInfo<long, unsigned int>, at::cuda::detail::TensorInfo<long const, unsigned int>, ...)": "gather",
    "void at::native::(anonymous namespace)::indexSelectLargeIndex<c10::BFloat16, long, unsigned int, 2, 2, -2, true>"
    "(at::cuda::detail::TensorInfo<c10::BFloat16, unsigned int>, at::cuda::detail::TensorInfo<c10::BFloat16 const, "
    "unsigned int>, at::cuda::detail::TensorInfo<long const, unsigned int>, int, int, unsigned int, unsigned int, "
    "long)": "gather",
    "void at::native::vectorized_gather_kernel<16, long>(char*, char*, long*, int, long, long, long, long, bool)":
        "other",
    "void at::native::_scatter_gather_elementwise_kernel<128, 8, at::native::_cuda_scatter_gather_internal_kernel<"
    "false, at::native::OpaqueType<8>, long>::operator()<long>(...)": "other",
}
TOWER_NAMES = {
    "nvjet_tst_128x64_64x6_2x1_v_bz_coopB_TNT": "cublas_gemm",
    "void flashinfer::BatchPrefillWithRaggedKVCacheKernel<flashinfer::KernelTraits<(flashinfer::MaskMode)0, 128u, 4u,"
    " 1u, 8u, 2u, 1u, 4u>>(...)": "flashinfer_attention",
    "(anonymous namespace)::g4_sandwich_kernel(__nv_bfloat16 const*, long, __nv_bfloat16 const*)": "fused_ops",
    "(anonymous namespace)::gelu_tanh_mul_kernel(__nv_bfloat16 const*, long, __nv_bfloat16*, int, int)": "fused_ops",
    "void (anonymous namespace)::g4_sandwich_vec_kernel<4>(__nv_bfloat16 const*, long, ...)": "fused_ops",
    "(anonymous namespace)::g4_rmsnorm_kernel(__nv_bfloat16 const*, long, __nv_bfloat16 const*, ...)": "fused_ops",
    "void at::native::(anonymous namespace)::indexFuncLargeIndex<c10::BFloat16, long, unsigned int, 2, 2, -2, "
    "true>(...)": "torch_glue",
    "void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::MeanOps<float, float>>>(...)":
        "torch_glue",
    "void at::native::(anonymous namespace)::CatArrayBatchedCopy<at::native::(anonymous namespace)::OpaqueType<4u>, "
    "unsigned int, 3, 64, 64>(...)": "torch_glue",
    "void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_"
    "cuda(at::TensorIteratorBase&)>(...)": "torch_glue",
    "Memcpy DtoD (Device -> Device)": "memcpy",
}


@pytest.mark.parametrize(("names", "table"), [(VERIFY_NAMES, pv.VERIFY_CATEGORIES),
                                              (TOWER_NAMES, pv.TOWER_CATEGORIES)])
def test_categories_name_the_kernels_this_code_launches(names, table) -> None:
    assert {n: pv.category(n, table) for n in names} == names


def test_summary_per_step_by_category_with_the_unmatched_names() -> None:
    rows = [(CUTLASS_FP8, 30_000.0, 240), ("Memcpy HtoD (Pinned -> Device)", 100.0, 20),
            ("(anonymous namespace)::sample_kernel(float const*)", 400.0, 20), ("mystery_kernel", 500.0, 40)]
    s = pv.summarize_kernels(rows, 10, pv.VERIFY_CATEGORIES, top=2)
    assert s["device_ms"] == 3.1 and s["kernel_ms"] == 3.09 and s["kernels"] == 30
    c = s["categories"]
    assert c["fp8_gemm"] == {"ms": 3.0, "share": round(30_000 / 31_000, 4), "calls": 24}
    assert c["memcpy"]["calls"] == 2 and c["bf16_gemm"] == {"ms": 0.0, "share": 0.0, "calls": 0}
    assert abs(sum(v["share"] for v in c.values()) - 1) < 1e-3
    assert [t["kernel"] for t in s["top"]] == [CUTLASS_FP8, "mystery_kernel"] and len(s["all"]) == 4
    assert s["other"] == [{"ms": 0.05, "calls": 4.0, "us_per_call": 12.5, "kernel": "mystery_kernel"}]


def test_power_sampler_samples_by_phase_or_says_why_not() -> None:
    """With NVML (a GPU host) each phase summarizes its samples; without (no NVIDIA GPU) the sampler is inert and the
    summary says why, so a run never fails for it."""
    s = pv.PowerSampler(interval=0.005)
    s.phase = "x"
    s.start()
    time.sleep(0.05)
    s.stop()
    out = s.summary("x")
    if s.nv is None:
        assert out["samples"] == 0 and out["error"] and s.info() == {"error": s.error}
    else:
        assert out["samples"] > 0 and out["power_w"]["mean"] > 0 and out["sm_mhz"]["n"] == out["samples"]


def test_clock_event_reasons_are_nvmls_bits() -> None:
    """Each reason the sampler names is NVML's bit of that name (pynvml's constants: where pynvml is installed)."""
    nv = pytest.importorskip("pynvml")
    names = {"idle": "GpuIdle", "app_clocks": "ApplicationsClocksSetting", "sw_power_cap": "SwPowerCap",
             "hw_slowdown": "HwSlowdown", "sync_boost": "SyncBoost", "sw_thermal": "SwThermalSlowdown",
             "hw_thermal": "HwThermalSlowdown", "hw_power_brake": "HwPowerBrakeSlowdown",
             "display_clocks": "DisplayClockSetting"}
    for bit, name in pv.REASONS.items():
        want = getattr(nv, f"nvmlClocksEventReason{names[name]}", None)
        assert bit == (want if want is not None else getattr(nv, f"nvmlClocksThrottleReason{names[name]}")), name


def ev(name: str, ts: float, dur: float, corr: int, cat: str = "kernel") -> dict:
    return {"ph": "X", "cat": cat, "name": name, "ts": ts, "dur": dur, "args": {"correlation": corr, "stream": 7}}


SAMPLE = "(anonymous namespace)::sample_kernel(float const*)"
NORM = "(anonymous namespace)::g4_sandwich_quant_kernel(__nv_bfloat16 const*)"
ATTN = "void (anonymous namespace)::attn512_kernel<true>(...)"
WINDOW = "void flashinfer::BatchPrefillWithPagedKVCacheKernel<...>(...)"


def split_step_trace(steps: int = 2) -> dict:
    """A split verify step's device events, as torch.profiler's trace lists them: the staging upload, the forward
    graph (one correlation id), the masks' upload, the sampling graph, the tokens' download; CPU and flow events."""
    out = [{"ph": "X", "cat": "cpu_op", "name": "aten::mm", "ts": 0, "dur": 900}, {"ph": "f", "cat": "ac2g", "id": 1}]
    for i in range(steps):
        t, c = 1000.0 * i, 10 * i
        out += [ev("Memcpy HtoD (Pinned -> Device)", t, 5, c, "gpu_memcpy"),
                ev(CUTLASS_FP8, t + 10, 100, c + 1), ev(NORM, t + 112, 8, c + 1), ev(ATTN, t + 121, 30, c + 1),
                ev("Memcpy HtoD (Pinned -> Device)", t + 160, 2, c + 2, "gpu_memcpy"),
                ev("nvjet_tst_64x8_64x16_4x1_v_bz_TNT", t + 170, 50, c + 3), ev(SAMPLE, t + 221, 5, c + 3),
                ev(WINDOW, t + 227, 10, c + 3), ev("Memcpy DtoH (Device -> Pinned)", t + 240, 1, c + 4, "gpu_memcpy")]
    return {"traceEvents": out}


def test_timeline_splits_a_step_by_graph_launch_with_its_gaps() -> None:
    """Events sharing a correlation id are one graph launch: per step the forward graph's busy time (138 µs) against its
    span (141: 3 µs of gaps between its kernels), the sampling graph's, the three copies launched alone, and the
    kernels of each graph by category."""
    tl = pv.timeline(split_step_trace(), 2, pv.VERIFY_CATEGORIES, parts=("forward", "sample"))
    assert tl["graph_launches"] == 2 and tl["graph_events"] == 6 and tl["single_events"] == 3
    assert (tl["graph_busy_ms"], tl["graph_span_ms"], tl["graph_gaps_ms"]) == (0.203, 0.208, 0.005)
    assert (tl["single_busy_ms"], tl["device_busy_ms"], tl["window_span_ms"]) == (0.008, 0.211, 0.6205)
    f, s = tl["forward"], tl["sample"]
    assert (f["busy_ms"], f["span_ms"], f["gaps_ms"], f["kernels"]) == (0.138, 0.141, 0.003, 3)
    assert {c: v["ms"] for c, v in f["categories"].items() if v["ms"]} == \
        {"fp8_gemm": 0.1, "norm_act_quant": 0.008, "full_attention": 0.03}
    assert {c: v["ms"] for c, v in s["categories"].items() if v["ms"]} == \
        {"bf16_gemm": 0.05, "sampler": 0.005, "sliding_attention": 0.01}
    assert (s["busy_ms"], s["gaps_ms"]) == (0.065, 0.002)
    assert "graph launches" in pv.timeline(split_step_trace(), 3, pv.VERIFY_CATEGORIES, ("a", "b"))["parts"]
    chunks = {"traceEvents": [ev("Memcpy HtoD (Pageable -> Device)", t, 1, 99, "gpu_memcpy") for t in (0, 2)]}
    assert pv.timeline(chunks, 1, pv.VERIFY_CATEGORIES)["graph_launches"] == 0  # a pageable copy's chunks
    assert pv.busy_us([ev("a", 0, 10, 1), ev("b", 5, 10, 1), ev("c", 30, 1, 2)]) == 16


def test_tower_table_lists_each_path() -> None:
    cupti = pv.summarize_kernels([("nvjet_tst_x", 900.0, 60), ("void at::native::vectorized_elementwise_kernel<4>()",
                                                                 300.0, 1200)], 3, pv.TOWER_CATEGORIES)
    cupti["timeline"] = pv.timeline(split_step_trace(3), 3, pv.TOWER_CATEGORIES)
    path = {"timing": {"device_ms": pv.stats([6.1, 6.0, 6.3])}, "cupti": cupti,
            "launches": {"graph_segments": 28, "eager_breaks": 27, "launches": 55}}
    table = pt.table({"graph": path, "eager": {**path, "launches": {"eager_kernels": 420.0}}})
    assert "launches/call" in table and "55" in table and "420.0" in table and "torch_glue" in table
    assert "graph gaps ms/call" in table and "0.005" in table
    # our attention-input kernel's launches (the glue), "-" for a record without them
    glued = pt.table({"graph": {**path, "fused_glue_launches": 27.0}, "eager": {**path, "fused_glue_launches": 0.0}})
    assert [line.split()[-2:] for line in (glued + "\n" + table).splitlines() if line.startswith("fused glue")] == [
        ["27", "0"], ["-", "-"]]


# -- the tower's image and call ------------------------------------------------------------------------------------


def synthetic_images(n: int, seed: int = 0) -> list[bytes]:
    """Synthetic JPEGs from one seed: five sizes in turn, each a random background and 20 random rectangles, quality
    90."""
    from PIL import Image, ImageDraw

    rng = np.random.default_rng(seed)
    out = []
    sizes = [(400, 225), (400, 711), (640, 480), (320, 320), (1000, 333)]
    for i in range(n):
        w, h = sizes[i % len(sizes)]
        img = Image.new("RGB", (w, h), tuple(int(x) for x in rng.integers(0, 256, 3)))
        d = ImageDraw.Draw(img)
        for _ in range(20):
            x0, y0 = int(rng.integers(0, w)), int(rng.integers(0, h))
            d.rectangle([x0, y0, x0 + int(rng.integers(4, w // 2)), y0 + int(rng.integers(4, h // 2))],
                        fill=tuple(int(x) for x in rng.integers(0, 256, 3)))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=90)
        out.append(buf.getvalue())
    return out


def test_tower_image_is_the_first_synthetic_image() -> None:
    assert pt.image_jpeg(400, 225) == synthetic_images(1)[0]


@pytest.mark.skipif(not (GEMMA / "config.json").exists() or not (GEMMA / "processor_config.json").exists(),
                    reason="tests/.fixtures/models/gemma-4-31b-it's configs are absent: "
                           "uv run tests/scripts/fetch_fixtures.py")
@pytest.mark.parametrize("budget", [2048, 8192])
def test_reference_image_takes_2376_patches_in_the_2439_bucket(budget) -> None:
    from llm_mini_infer.vision import family as vf

    fam = vf.load(GEMMA, max_soft_tokens=280)
    shape = pt.image_shape(fam, pt.image_jpeg(1280, 720), budget)
    assert (shape["size"], shape["grid"], shape["tokens"], shape["patches"]) == ([576, 1056], [1, 36, 66], 264, 2376)
    assert shape["bucket"] == 2439 and shape["padding"] == 63 and shape["encoder_budget"] == budget
    assert pt.check_reference(shape) == []
    with pytest.raises(AssertionError, match="tokens"):
        pt.check_reference({**shape, "tokens": 65})
    assert pt.check_reference({**shape, "bucket": 2304}) == ["2376 patches land in the 2304 bucket, not 2439"]


@pytest.fixture(scope="module")
def tiny_mm(tmp_path_factory):
    return make_tiny_gemma_mm(tmp_path_factory.mktemp("profile_mm"), depth=2, seed=3)


def test_tower_call_writes_the_towers_soft_tokens(tiny_mm) -> None:
    """The image through `ImageEncoder.encode` as the runner calls it (pixels from staging, the plan's grid and rows),
    eagerly on CPU: the encoder pool's rows are the tower's output for the image, and the image is sized as on the
    real tower (its patches are 4 pixels: the same grid at 70 soft tokens)."""
    from llm_mini_infer.runner.cuda.images import ImageEncoder, VisionOptions
    from llm_mini_infer.vision import family as vf
    from llm_mini_infer.vision.probe import probe

    fam = vf.load(tiny_mm, max_soft_tokens=70)
    data = pt.image_jpeg(400, 225)
    shape = pt.image_shape(fam, data, 2048)
    assert (shape["grid"], shape["tokens"], shape["bucket"]) == ([1, 18, 33], 66, 648)
    assert pt.check_reference(shape) == []  # 4-pixel patches: nothing to check
    enc = ImageEncoder(tiny_mm, VisionOptions(encoder_rows=2 * shape["encoder_budget"],
                                              encoder_budget=shape["encoder_budget"], max_soft_tokens=70),
                       torch.device("cpu"), torch.float32)
    assert enc.buckets == shape["buckets"]
    enc.allocate(enc.cfg.out_hidden, flashinfer=False)
    call = pt.TowerCall(enc, fam, data)
    call(False)
    v, hdr = enc.vision, probe(data)
    meta = v.meta([call.grid], torch.device("cpu"))
    enc.attn.plan(meta)
    with torch.inference_mode():
        want = v(v.patches(fam.pixels(data, hdr, fam.size(hdr))), meta, enc.attn)
    assert call.patches == 594 and torch.equal(call.output(), want)
    assert not enc.pool[call.tokens:].any()


def test_one_tower_call_per_image_gives_each_image_its_rows_alone(tiny_mm, monkeypatch) -> None:
    """`ImageEncoder.encode` with one tower call per image (`per_image`, batch invariance's), eagerly on CPU: an ENCODE
    of images B, A and C (16, 66 and 27 soft tokens) makes three `_tower` calls of one grid each, and each image's
    encoder rows are those it gets encoded alone, bit for bit; without it, one call holds the three grids."""
    from llm_mini_infer.runner.cuda.images import ImageEncoder, VisionOptions

    enc = ImageEncoder(tiny_mm, VisionOptions(encoder_rows=256, encoder_budget=256, max_soft_tokens=70),
                       torch.device("cpu"), torch.float32)
    enc.allocate(enc.cfg.out_hidden, flashinfer=False)
    grids, tower = [], enc._tower

    def spy(patches: torch.Tensor, g: list, graphs: bool) -> torch.Tensor:
        grids.append(list(g))
        return tower(patches, g, graphs)

    monkeypatch.setattr(enc, "_tower", spy)
    side = enc.cfg.merge * enc.cfg.patch  # one soft token's pixels
    gen = torch.Generator().manual_seed(0)
    b, a, c = (torch.randint(0, 256, (3, side * h, side * w), generator=gen, dtype=torch.uint8)
               for h, w in ((4, 4), (6, 11), (3, 9)))
    alone = []
    for px in (b, a, c):
        one = pt.ImagesCall(enc, [px])
        one(False)
        alone.append(one.rows(0))
    enc.per_image = True
    enc.pool.zero_()
    grids.clear()
    bac = pt.ImagesCall(enc, [b, a, c])
    bac(False)
    assert bac.tokens == [16, 66, 27] and grids == [[g] for g in bac.grids]
    for i in range(3):
        assert torch.equal(bac.rows(i), alone[i]), "BAC"[i]
    assert not enc.pool[sum(bac.tokens):].any()
    enc.per_image = False
    grids.clear()
    bac(False)
    assert grids == [bac.grids]


def test_the_image_encoders_phases(tiny_mm, monkeypatch) -> None:
    """An `ImageEncoder`'s phase (`runner.cuda.images.Phase`): ENCODE and a capture before `allocate` are refused,
    named; allocate and release go from any phase, as the memory profile and the cleanup paths use them; a capture
    keeps its graphs, and is CAPTURED, only once every bucket is captured."""
    from llm_mini_infer.runner.cuda import images
    from llm_mini_infer.runner.cuda.images import ImageEncoder, Phase, VisionOptions

    enc = ImageEncoder(tiny_mm, VisionOptions(encoder_rows=256, encoder_budget=256, max_soft_tokens=70),
                       torch.device("cpu"), torch.float32)
    plan = types.SimpleNamespace(step_id=3)
    assert enc.life.phase is Phase.LOADED and enc.buckets
    unallocated = f"step {plan.step_id}: ENCODE before the image encoder is allocated"
    with pytest.raises(RuntimeError, match=unallocated):
        enc.stage(plan, None, False)
    with pytest.raises(RuntimeError, match=unallocated):
        enc.encode(plan, None, (0, 0, None), False)
    with pytest.raises(RuntimeError, match="a capture when LOADED, not ALLOCATED"):
        enc.capture(None)
    captured: list[int] = []

    class Graphs:  # VisionGraphs without CUDA: the buckets it captures
        def __init__(self, vision, attn, buckets, pool) -> None:
            self.graphs: dict[int, None] = {}

        def capture(self, n: int) -> None:
            captured.append(n)
            self.graphs[n] = None

    monkeypatch.setattr(images, "VisionGraphs", Graphs)
    enc.release()
    for _ in range(2):
        enc.allocate(enc.cfg.out_hidden, flashinfer=False)
        assert enc.life.phase is Phase.ALLOCATED and enc.pool is not None
        enc.capture(None)
        assert enc.life.phase is Phase.CAPTURED and enc.graph_buckets() == sorted(enc.buckets)
        assert captured == sorted(enc.buckets, reverse=True)
        captured.clear()
    with pytest.raises(RuntimeError, match="a capture when CAPTURED, not ALLOCATED"):
        enc.capture(None)
    enc.release()
    assert enc.life.phase is Phase.LOADED and enc.pool is None and enc.graphs is None

    def out_of_memory(self, n: int) -> None:
        raise RuntimeError("out of memory")

    enc.allocate(enc.cfg.out_hidden, flashinfer=False)
    monkeypatch.setattr(Graphs, "capture", out_of_memory)
    with pytest.raises(RuntimeError, match="out of memory"):
        enc.capture(None)
    assert enc.life.phase is Phase.ALLOCATED and enc.graphs is None


def test_the_glue_kernels_launches_and_the_arguments() -> None:
    """Our attention-input kernel's launches in a profile (one a layer), and the arguments: 50 calls a path, 2 under
    ncu, the graph path alone unless --eager."""
    assert pt.glue_launches({"all": [{"kernel": "void (anonymous namespace)::g4_vision_qkv_kernel<72>(...)",
                                      "calls": 27.0}, {"kernel": "nvjet_tst_x", "calls": 82.0}]}) == 27.0
    base = ["--model", "m", "--out", "o.json"]
    a = pt.parse_args(base)
    assert (a.calls, a.mode, a.eager) == (50, "torch", False)
    a = pt.parse_args([*base, "--mode", "ncu", "--eager"])
    assert (a.calls, a.mode, a.eager) == (2, "ncu", True)


# -- on a CUDA GPU ------------------------------------------------------------------------------------------------


def to_bf16(path: Path, scaled: tuple[str, ...]) -> None:
    """The checkpoint in BF16, the weights ending in `scaled` a quarter (moderate attention logits at scale 1.0, as
    gemma_gpu's fixture)."""
    from safetensors.torch import load_file, save_file

    f = path / "model.safetensors"
    sd = load_file(f)
    sd = {k: v * 0.25 if k.endswith(scaled) else v for k, v in sd.items()}
    save_file({k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in sd.items()}, f)


@pytest.fixture(scope="module")
def gpu_pair(tmp_path_factory):
    """gemma_gpu's pair: the real head dims (sliding 256, full 512), a window of 40, a tiny assistant."""
    d = tmp_path_factory.mktemp("profile_gpu")
    target = make_tiny_gemma(d / "target", window=40, gpu=True, seed=3)
    to_bf16(target, ("q_norm.weight", "k_norm.weight"))
    assistant = make_tiny_assistant(d / "assistant", target, hidden=256, intermediate=512, seed=4)
    to_bf16(assistant, ("q_norm.weight",))
    return target, assistant


@cuda
def test_categories_name_this_gpus_kernels() -> None:
    """Our quantizer, the FP8 GEMM (the CUTLASS blockwise kernel where the library has it) and cuBLAS's BF16 GEMM,
    each alone under torch.profiler: every kernel each launches falls in its category."""
    import torch.nn.functional as F

    from llm_mini_infer.kernels import fp8, ops

    x = torch.randn(16, 512, device="cuda", dtype=torch.bfloat16)
    w, ws = fp8.quantize_weight(torch.randn(1024, 512, device="cuda") * 0.05)
    wb = torch.randn(4096, 512, device="cuda", dtype=torch.bfloat16)
    a = ops.quant_fp8(x)
    torch.cuda.synchronize()
    calls = [(lambda: ops.quant_fp8(x), "norm_act_quant"), (lambda: F.linear(x, wb), "bf16_gemm")]
    if fp8.cutlass_available():  # the CUTLASS library (SM 12.x)
        calls.append((lambda: fp8.gemm(a, w, ws), "fp8_gemm"))
    for fn, want in calls:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            fn()
            torch.cuda.synchronize()
        names = [n for n, _, _ in pv.profiler_rows(prof) if not pv.MEMCPY.search(n)]
        assert names and {pv.category(n, pv.VERIFY_CATEGORIES) for n in names} == {want}, names


def gpu_harness(gpu_pair, grammar: str = "allow-all", **heads):
    """The harness on CUDA with graphs over the tiny pair (`heads`: the GemmaRunner's grammar-aware heads)."""
    from llm_mini_infer.runner.cuda.gemma import GemmaRunner

    target, assistant = gpu_pair
    pc = gemma_pools(max_num_seqs=4, budget=64, page_size=16, num_pages=512, max_model_len=256, mtp_k=K)
    runner = GemmaRunner(target, device="cuda", max_num_seqs=pc.max_num_seqs,
                         max_num_batched_tokens=pc.max_num_batched_tokens, max_model_len=pc.max_model_len,
                         memory_gb=2.0, graphs=True, mtp_k=K, assistant=assistant, **heads)
    runner.configure(pc)
    h = pv.Harness(runner, pc, eos=pv.eos_ids(target), grammar=grammar)
    return h, pv.StepSpy(runner), pv.Prompts(64, 10, h.caps.vocab_size)


@cuda
def test_verify_profile_runs_the_split_decode_graphs(gpu_pair) -> None:
    """The harness on CUDA with graphs: every measured step ran the decode graphs of its bucket, split (the spy), and
    CUPTI saw the verify step's kernels in their categories: no kernel of ours, FlashInfer's or a GEMM is "other".
    The grammar-aware heads are off (the runner's default): the dense head, counted so."""
    h, spy, prompts = gpu_harness(gpu_pair)
    sampler = pv.PowerSampler(torch.cuda.current_device()).start()
    try:
        res = pv.profile(h, spy, prompts, rows=[4, 2], k=K, steps=3, warm_s=0.2, min_warm_steps=4, decode_before=2,
                         max_tokens=100, fresh=True, mode="torch", sampler=sampler, log=lambda _: None)
    finally:
        sampler.stop()
    for n in (4, 2):
        r = res[str(n)]
        assert r["torch"]["plans"]["modes"] == {f"decode/{n}/split": 3}
        assert r["unprofiled"]["plans"]["modes"] == {f"decode/{n}/split": 4}
        assert r["torch"]["plans"]["heads"]["path"] == r["unprofiled"]["plans"]["heads"]["path"] == "dense_off"
        c = r["torch"]["categories"]
        for cat in ("sliding_attention", "full_attention", "sampler", "bf16_gemm", "norm_act_quant", "gather"):
            assert c[cat]["calls"] > 0, (n, cat)
        assert c["fp8_gemm"]["calls"] == 0  # a BF16 model
        mine = r"lmk_|flashinfer|attn512|g4_|gelu|sample_kernel|nvjet|gemm|gemv|cutlass|cublas|quant|indexSelect"
        assert not [o["kernel"] for o in r["torch"]["other"] if re.search(mine, o["kernel"])]
        assert r["kernel_share_of_busy"] > 0 and r["unprofiled"]["busy_ms"]["mean"] > 0
        tl = r["torch"]["timeline"]  # per step: the forward graph, then the sampling graph (the drafter's in it)
        assert tl["graph_launches"] == 2 and tl["forward"]["categories"]["full_attention"]["calls"] > 0
        assert tl["sample"]["categories"]["sampler"]["calls"] > 0 and tl["sample"]["gaps_ms"] >= 0
    h.runner.shutdown()


@cuda
def test_verify_profile_with_the_grammar_heads(gpu_pair) -> None:
    """Both heads on under the subset grammar: every measured step ran the decode graphs of its bucket with the heads'
    sampling graph, on the sparse head over the subset's 64 tokens, its drafts over a universe of the same 64; CUPTI
    saw the sparse head's gather of its rows (128, padded: the large-index kernel) in the sampling graph, in "gather".
    128 rows, a quarter of the tiny vocabulary: an all-ones mask (a dropped row's) is over them."""
    h, spy, prompts = gpu_harness(gpu_pair, "subset:64", **HEADS)
    res = pv.profile(h, spy, prompts, rows=[4, 2], k=K, steps=3, warm_s=0.2, min_warm_steps=4, decode_before=2,
                     max_tokens=100, fresh=True, mode="torch", log=lambda _: None)
    for n in (4, 2):
        r = res[str(n)]
        for m in ("unprofiled", "torch"):
            heads, plans = r[m]["plans"]["heads"], r[m]["plans"]
            assert (heads["path"], heads["union"]["mean"], heads["universe"]) == ("sparse", 64, 64), (n, m)
            assert heads["universe_steps"] == plans["steps"] and heads["universe_rebuilds"] == 0
            assert plans["modes"] == {f"decode/{n}/split": plans["steps"]}
        tl = r["torch"]["timeline"]
        assert tl["graph_launches"] == 2 and tl["sample"]["categories"]["gather"]["calls"] > 0
        assert [k for k in r["torch"]["all"] if k["cat"] == "gather" and "indexSelect" in k["kernel"]]  # Small or Large
    assert res["grammar"]["errors"] == 0 and res["heads"]["universe_rebuilds"] == 1
    h.runner.shutdown()


def cuda_tower(tmp_path: Path) -> tuple[object, object, dict]:
    """A tiny tower on CUDA in BF16 with FlashInfer (2 layers, heads of 16, 4-pixel patches: the 400 x 225 image's
    grid): its encoder, the image's call (`TowerCall`) and the image's shape."""
    from safetensors.torch import load_file, save_file

    from llm_mini_infer.runner.cuda.images import ImageEncoder, VisionOptions
    from llm_mini_infer.vision import family as vf

    path = make_tiny_gemma_mm(tmp_path / "mm", depth=2, gpu=True, seed=4)
    f = path / "model.safetensors"
    save_file({k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in load_file(f).items()}, f)
    fam = vf.load(path, max_soft_tokens=70)
    data = pt.image_jpeg(400, 225)
    shape = pt.image_shape(fam, data, 2048)
    enc = ImageEncoder(path, VisionOptions(encoder_rows=2 * shape["encoder_budget"],
                                           encoder_budget=shape["encoder_budget"], max_soft_tokens=70),
                       torch.device("cuda", torch.cuda.current_device()), torch.bfloat16)
    enc.allocate(enc.cfg.out_hidden, flashinfer=True)
    return enc, pt.TowerCall(enc, fam, data), shape


@cuda
def test_tower_profile_replays_the_images_bucket(tmp_path) -> None:
    """The tower harness on CUDA (a tiny tower, 4-pixel patches: the 400 x 225 image's grid): the 648 bucket's graph
    replays (3 segments around its 2 layers' attention breaks; the real tower's 27 layers make 28 and 27), the eager
    path launches every kernel, our attention-input kernel runs once a layer on both, and both write the same soft
    tokens within BF16 noise."""
    enc, call, shape = cuda_tower(tmp_path)
    call(False)
    cap = pt.capture(enc, [shape["bucket"]])
    assert cap["buckets"] == [648]
    res = pt.profile(enc, call, paths=["graph", "eager"], calls=3, warmup=1, mode="torch", log=lambda _: None)
    g, e = res["graph"], res["eager"]
    assert g["launches"] == {"graph_segments": 3, "eager_breaks": 2, "launches": 5}
    assert e["launches"] == {"eager_kernels": e["cupti"]["kernels"]}
    assert g["cupti"]["timeline"]["graph_launches"] == 3 and e["cupti"]["timeline"]["graph_launches"] == 0
    for r in (g, e):
        c = r["cupti"]["categories"]
        assert c["flashinfer_attention"]["calls"] >= 2 and c["cublas_gemm"]["calls"] > 0
        assert c["torch_glue"]["calls"] > 0 and c["fused_ops"]["calls"] > 0 and r["timing"]["device_ms"]["p50"] > 0
        assert r["fused_glue_launches"] == 2
    assert res["graph_vs_eager"] < 0.1
