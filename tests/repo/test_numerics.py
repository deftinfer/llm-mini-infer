"""CPU tests of the numerics tools (tests/scripts/numerics_kernels.py, tests/scripts/numerics_forced.py,
tests/scripts/numerics_fp32.py): their host-side helpers, and the tools on the tiny models (tests/support/tiny_qwen.py,
tests/support/tiny_gemma.py)."""

from __future__ import annotations

import contextlib
import importlib
import io
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from ..scripts import numerics_forced as nfo
from ..scripts import numerics_fp32 as nf
from ..scripts import numerics_kernels as nk
from ..support.paths import ROOT

# -- tests/scripts/numerics_kernels.py: host-side helpers --------------------------------------------------------------


def test_cut_points_are_multiples_of_64_inside_the_row() -> None:
    assert nk.cut_points(70) == []
    assert nk.cut_points(300) == [64, 192]
    assert nk.cut_points(2000) == [640, 1280]
    for n in range(1, 3000, 37):
        pts = nk.cut_points(n)
        assert pts == sorted(set(pts)) and all(p % 64 == 0 and 0 < p < n for p in pts), n


def test_row_kinds_name_partial_chunks_and_the_rows_they_reach() -> None:
    """gdn-race's classification of packed rows: a sequence's partial last chunk (bit 1) and the rows after it that
    chunk's 64 rows run into (bit 2), across a short sequence too."""
    cu = np.concatenate([[0], np.cumsum([100, 10, 300, 128, 64])])  # 0, 100, 110, 410, 538, 602
    want = np.zeros(int(cu[-1]), np.int64)
    want[64:110] |= 1  # sequence 0's last 36 rows, sequence 1's 10
    want[366:410] |= 1  # sequence 2's last 44
    want[100:128] |= 2  # sequence 0's last chunk runs to 128
    want[110:164] |= 2  # sequence 1's to 164, into sequence 2
    want[410:430] |= 2  # sequence 2's to 430; sequence 3's whole chunks reach nothing
    np.testing.assert_array_equal(nk.row_kinds(np.arange(int(cu[-1])), cu), want)


def _gdn_rows(lens: list[int], seed: int, hk: int = 2, hv: int = 6, d: int = 8) -> list[dict]:
    g = torch.Generator().manual_seed(seed)

    def unit(n: int) -> torch.Tensor:
        x = torch.randn(n, hk, d, generator=g)
        return (x / x.norm(dim=-1, keepdim=True)).to(torch.bfloat16)

    return [{"q": unit(n), "k": unit(n), "v": torch.randn(n, hv, d, generator=g).to(torch.bfloat16),
             "a": torch.exp(-torch.rand(n, hv, generator=g) * 0.3), "b": torch.sigmoid(torch.randn(n, hv, generator=g)),
             "s": torch.randn(hv, d, d, generator=g) * 0.1} for n in lens]


def test_padding_layout_and_rows() -> None:
    """`padded_index` (packed) and `pad_row` (per row) pad alike: every row to a multiple of 64, the padding rows
    q = k = v = 0, decay 1, beta 0, the state untouched."""
    lens = [100, 64, 1, 0, 130]
    pcu, valid = nk.padded_index(lens)
    np.testing.assert_array_equal(pcu, [0, 128, 192, 256, 256, 448])
    np.testing.assert_array_equal(valid, np.r_[0:100, 128:192, 192, 256:386])
    rows = _gdn_rows(lens, 0)
    padded = [nk.pad_row(r) for r in rows]
    pad = np.setdiff1d(np.arange(pcu[-1]), valid)
    for key, fill in (("q", 0.0), ("k", 0.0), ("v", 0.0), ("a", 1.0), ("b", 0.0)):
        cat = torch.cat([p[key] for p in padded])
        assert cat.shape[0] == pcu[-1] and cat.dtype == rows[0][key].dtype
        assert torch.equal(cat[valid], torch.cat([r[key] for r in rows])), key
        assert bool((cat[pad] == fill).all()), key
    assert all(p["s"] is r["s"] for p, r in zip(padded, rows))


def test_gdn_fp32_is_torch_gdn_prefill() -> None:
    """The fp32 reference, every row at once, gives `torch_gdn_prefill`'s outputs (in BF16), final states and every-64
    states (its checkpoint workspace)."""
    from llm_mini_infer.models.qwen3_5.backend import CkptMeta, torch_gdn_prefill

    lens = [150, 70, 5]
    rows = _gdn_rows(lens, 1)
    got = nk.gdn_fp32(rows, {0: [64, 128], 1: [64]})
    cu = np.concatenate([[0], np.cumsum(lens)])
    starts = np.concatenate([[0], np.cumsum(np.asarray(lens) // 64)])
    ws = torch.zeros(int(starts[-1]), 6, 8, 8)
    meta = SimpleNamespace(pf_cu_host=cu, ckpt=CkptMeta(every=64, cu_starts_host=starts, workspace=ws,
                                                         src=None, dst=None, conv=None))
    final = torch.stack([r["s"] for r in rows])  # the pool, each row in its own slot
    o = torch_gdn_prefill(*(torch.cat([r[x] for r in rows]) for x in nk.KEYS), final, torch.arange(len(rows)), meta)
    for i, (oi, fi, snaps) in enumerate(got):
        assert torch.equal(oi.to(torch.bfloat16), o[cu[i]:cu[i + 1]]), i
        torch.testing.assert_close(fi, final[i], rtol=1e-6, atol=1e-7)
        for off, state in snaps.items():
            torch.testing.assert_close(state, ws[starts[i] + off // 64 - 1], rtol=1e-6, atol=1e-7)
    assert sorted(got[0][2]) == [64, 128] and sorted(got[1][2]) == [64] and got[2][2] == {}


def test_padding_rows_leave_the_recurrence_where_it_is() -> None:
    """nan-leak's candidate fix for a NaN that crosses packed sequences (flashinfer-ai/flashinfer#9372) in exact
    arithmetic: rows padded to multiples of 64 give the same outputs and final states."""
    rows = _gdn_rows([100, 7], 2)
    for (o, f, _), (po, pf, _), n in zip(nk.gdn_fp32(rows), nk.gdn_fp32([nk.pad_row(r) for r in rows]), [100, 7]):
        assert po.shape[0] == -(-n // 64) * 64
        assert torch.equal(po[:n], o) and torch.equal(pf, f)


def test_attn_cases() -> None:
    rng = np.random.default_rng(0)
    cases = nk.attn_cases([1, 4, 16], 2000, 6000, 3, rng)
    assert [c.name for c in cases] == ["verify-1-diverse", "verify-4-lockstep", "verify-4-diverse",
                                       "verify-16-lockstep", "verify-16-diverse", "prefill-2048", "prefill-history",
                                       "mixed-16+512"]
    for c in cases:
        d = c.n_dec
        assert (c.nq <= c.lb).all() and (c.nq[:d] == 4).all() and (c.ub[:d] - c.lb[:d] == 3).all()
        assert (c.lb[d:] == c.ub[d:]).all() and c.uncertain == bool(d)
        np.testing.assert_array_equal(c.qo_indptr, np.r_[c.starts, c.nq.sum()])
        for mode in ("lb", "ub", "rand"):
            t = c.true(mode, rng)
            assert ((c.lb <= t) & (t <= c.ub)).all(), (c.name, mode)
        if c.name.startswith("verify"):
            assert c.lb.min() >= 2000 and c.lb.max() <= 6000
        if "lockstep" in c.name:
            assert len(set(c.lb.tolist())) == 1
    assert nk.decode_bucket(5, [1, 2, 4, 8, 16]) == 8 and nk.decode_bucket(16, [1, 16]) == 16


def test_attn_timing_grid() -> None:
    """The default grid's steps and their exact lengths, the Ms a step runs at, the on path's split per M."""
    from llm_mini_infer.kernels.attn512 import MAX_SPLITS, fixed_split_keys, fixed_splits

    a = nk.build_parser().parse_args(["attn-timing"])
    ms, k = nfo.parse_ids(a.max_model_lens), a.k
    cases = nk.timing_cases(nfo.parse_ids(a.short_chunks), nfo.parse_ids(a.histories), a.long_chunk,
                            nfo.parse_ids(a.long_histories), nfo.parse_ids(a.decode_rows),
                            nfo.parse_ids(a.decode_histories), k)
    by = {c.name: c for c in cases}
    assert len(cases) == len(by) == 2 * 3 + 2 + 2 * 3 and ms == [8184, 32768, 131072, 262144]
    assert all(not c.uncertain for c in cases)
    c = by["prefill-short-256-h131072"]
    assert (c.n_dec, c.nq.tolist(), c.ub.tolist()) == (0, [256], [131072 + 256])
    c = by["prefill-long-2048-h0"]
    assert (c.n_dec, c.nq.tolist(), c.lb.tolist()) == (0, [2048], [2048]) and nk.case_history(c) == 0
    c = by["decode-64-h32768"]
    assert c.n_dec == 64 and (c.nq == 4).all() and (c.ub == 32768 + 4).all() and nk.case_history(c) == 32768
    np.testing.assert_array_equal(c.starts, np.arange(64) * 4)
    for c in cases:  # prefill rows once (M = any); verify rows at every M, skipped where a row is past M + 1 + k
        got = nk.timing_ms(c, ms, k)
        if not c.n_dec:
            assert got == [(None, False)]
        else:
            assert [m for m, _ in got] == ms
            assert all(skip == (int(c.ub.max()) > m + 1 + k) for m, skip in got)
    assert nk.timing_ms(by["decode-1-h1024"], ms, k, per_m=False) == [(None, False)]
    assert nk.timing_ms(by["decode-1-h32768"], ms, k) == [(8184, True), (32768, False), (131072, False),
                                                           (262144, False)]
    for m in ms:
        c = by["decode-1-h1024"]
        assert nk.on_max_kv(c, m, k) == m + 1 + k and nk.on_split_keys(c, m, k) == fixed_split_keys(m + 1 + k)
        assert int(fixed_splits(m + 1 + k, nk.on_split_keys(c, m, k))) <= MAX_SPLITS
    c = by["decode-1-h1024"]
    assert nk.on_split_keys(c, 8184, k) == 256 and nk.on_split_keys(c, 262144, k) > 256
    assert nk.on_max_kv(by["prefill-short-64-h8192"], None, k) == 8192 + 64


def test_timing_ratios() -> None:
    def rec(impl: str, mode: str, us: float | None, m: int | None = None) -> dict:
        return {"case": "c", "impl": impl, "mode": mode, "m": m, **({} if us is None else {"median_us": us})}

    recs = [rec("off", "eager", 50.0), rec("off", "graph", 20.0), rec("on", "eager", 25.0, 8184),
            rec("on", "graph", 30.0, 8184), rec("on", "eager", None, 32768), rec("on", "bucket", 5.0, 8184)]
    nk.timing_ratios(recs)
    assert [r.get("ratio") for r in recs[2:]] == [0.5, 1.5, None, None]


def test_rel_errors() -> None:
    ref = torch.tensor([3.0, -4.0])
    e = nk.rel_errors(ref + torch.tensor([0.3, 0.4]), ref)
    assert e["rel_rms"] == pytest.approx(0.1) and e["max_rel"] == pytest.approx(0.1) and e["nonfinite"] == 0
    e = nk.rel_errors(torch.tensor([float("nan"), 1.0]), ref)
    assert e["nonfinite"] == 1 and nk._clean(e)["rel_rms"] is None


def test_command_line() -> None:
    """Every subcommand parses with its defaults (on any machine: torch and FlashInfer load inside them); gdn-race's
    sequences end in partial 64-row chunks."""
    p = nk.build_parser()
    for cmd in ("smid", "gdn-race", "gdn-accuracy", "attn-accuracy", "nan-leak", "timing", "attn-timing"):
        assert callable(p.parse_args([cmd]).func)
    assert all(n % 64 for n in nfo.parse_ids(p.parse_args(["gdn-race"]).lens))
    assert nfo.parse_ids("0-3,7, 14") == [0, 1, 2, 3, 7, 14]


# -- the three tools together on the tiny models (CPU, fp32) ---------------------------------------------------------
#
# Each tiny model's requests (text, and text with an image) go through tests/scripts/numerics_forced.py (two arms,
# eager, with taps and core layers) and their captures through tests/scripts/numerics_fp32.py. A CPU runner computes in
# fp32 with the torch reference kernels: the fp32 tool's own arithmetic up to summation order (SDPA's backend, a matmul
# over more rows), so its recomputation of every layer and kernel agrees to TOL (seen: 2.5e-6 at most), and its chain,
# whose differences compound over the layers (about twice a layer on the tiny Gemma), to CHAIN_TOL (seen: under 2e-5).

K = 3  # drafts a verify step
ANSWER = 24  # a reference answer's tokens
TOL = 1e-5  # relative error (RMS, and max |x − ref| / max |ref|) of a recomputed layer or kernel against the engine
CHAIN_TOL = 1e-4  # the same of the fp32 chain's streams and final hidden states against each arm's
TEXT = (3, 490)  # the tiny models' text tokens: below their image tokens (Qwen 496-499, Gemma 500-503) and EOS 511
ARMS = {"on-eager", "off-eager"}


def text(n: int, salt: int) -> list[int]:
    return [(i * 53 + 7 + salt * 131) % (TEXT[1] - TEXT[0]) + TEXT[0] for i in range(n)]


@dataclass
class Tiny:
    """A tiny model's forced run and its fp32 analysis: the references (refs.json's content), the run's folder (arm
    `on-eager`: switch on, one request at a time, the final states from the forward; `off-eager`: switch off, three
    in flight, the final states where graphs keep them, `hid_out`; both with taps and `core` layers), the captures by
    request, the fp32 reference and its sweep (layers and chain)."""

    refs: dict[str, Any]
    run: Path
    core: list[int]
    reqs: list[Any]
    ref: Any
    swept: dict[int, Any]


def drafty(ref: Any, req: Any, rng: np.random.Generator) -> list[int]:
    """An answer of ANSWER tokens the drafter often gets right, so that the forced runs accept drafts (a tiny model's
    drafter is random): `req`'s forced prefix and first answer token, then along the engine's walk, at each round p,
    the fp32 drafter's first a drafts there (on H_fp32; a drawn from 0..K, cut before a draft outside TEXT) and a token
    other than draft a + 1. `req`: a forced run's capture with taps (the prompt's embeddings, image rows included)."""
    cap = next(iter(req.caps.values()))
    stream, pl = cap["taps"]["stream"], req.prompt_len
    toks = req.tokens[:pl + req.forced + 1].tolist()
    p = len(toks) - 2  # the drafter's first round
    while len(toks) < pl + ANSWER:
        n = len(toks)
        s = torch.zeros((stream.shape[0], n, stream.shape[2]), dtype=stream.dtype)
        s[:, :pl] = stream[:, :pl]
        sub = nf.Request(req.rid, {"forced": {
            **cap, "tokens": torch.tensor(toks), "hidden": torch.zeros((n, s.shape[2]), dtype=s.dtype),
            "hvalid": torch.zeros(n, dtype=torch.bool), "rope": None if req.rope is None else req.rope[:, :n],
            "taps": {**cap["taps"], "stream": s, "tvalid": torch.arange(n) < pl}}})
        chain = nf.sweep(ref, [sub], layers=False, say=lambda _: None)[req.rid].chain
        with nf.reference_numerics():
            at = torch.tensor([p])
            drafts = (nf.draft_qwen(ref, sub, chain.hidden[None], chain.embeds.to(ref.dtype), at, K)
                      if ref.family == "qwen" else nf.draft_gemma(ref, sub, chain.hidden[None], chain.kv, at, K))
        drafts = drafts[0, 0].tolist()
        a, want = 0, int(rng.integers(0, K + 1))
        while a < want and TEXT[0] <= drafts[a] < TEXT[1]:
            a += 1
        other = int(rng.integers(TEXT[0], TEXT[1] - 1))
        toks += [*drafts[:a], other + int(a < K and other >= drafts[a])]
        p += 1 + a
    return toks[pl:pl + ANSWER]


def forced_and_swept(tmp: Path, model: Path, assistant: Path | None, parts: list[list[Any]], forced: list[int],
                     core: list[int], soft: int | None) -> Tiny:
    """The references of `parts` (greedy), a forced run of arbitrary answers for the prompts' embeddings, drafty
    answers from them, then the two arms' forced runs (`on-eager` by the command line) and the fp32 sweep."""
    opts = nfo.Options(device="cpu", max_model_len=1024, budget=128, max_num_seqs=4, memory_gb=0.05, soft_tokens=soft)
    reqs = [dict(rid=i, workload="tiny", index=i, parts=x, grammar="", forced=f)
            for i, (x, f) in enumerate(zip(parts, forced))]
    refs = nfo.make_refs(model, reqs, assistant=assistant, opts=opts, max_tokens=ANSWER)
    for q in refs["requests"]:
        q["answer"] = text(ANSWER, 50 + q["rid"])
    nfo.run_arm(refs, switch="on", graphs=False, taps=True, out=tmp / "prompts", opts=opts)
    ref = nf.make_ref(model, assistant, device="cpu")
    rng = np.random.default_rng(0)
    for q, req in zip(refs["requests"], nf.load_requests([tmp / "prompts"])):
        q["answer"] = drafty(ref, req, rng)
    (tmp / "refs.json").write_text(json.dumps(refs, default=str))
    run = tmp / "run"
    nfo.main(["run", "--refs", str(tmp / "refs.json"), "--arm", "on", "--no-graphs", "--taps", "--core-layers",
              ",".join(map(str, core)), "--device", "cpu", "--memory-gb", "0.05", "--max-model-len", "1024",
              "--budget", "128", "--max-num-seqs", "4", "--k", str(K), "--out", str(run)])
    nfo.run_arm(tmp / "refs.json", switch="off", graphs=False, taps=True, core_layers=core, concurrency=3, out=run,
                opts=opts, hid_out=True)
    caps = nf.load_requests([run])
    return Tiny(refs, run, core, caps, ref, nf.sweep(ref, caps, say=lambda _: None))


@pytest.fixture(scope="module")
def tiny_qwen(tmp_path_factory) -> Tiny:
    """Qwen3.5 VL with its MTP layer: a text prompt, one with an image (M-RoPE), a longer one (two chunks)."""
    from ..support.tiny_qwen import VL_TOKENS, make_tiny_vl_model

    tmp = tmp_path_factory.mktemp("numerics-qwen")
    make_tiny_vl_model(tmp / "model", seed=3, mtp=True)
    start, end = VL_TOKENS["vision_start_token_id"], VL_TOKENS["vision_end_token_id"]
    return forced_and_swept(tmp, tmp / "model", None,
                            [[text(40, 0)], [[*text(20, 1), start], {"image": 0}, [end, *text(9, 2)]], [text(150, 3)]],
                            [0, 2, 1], [0, 1], None)


@pytest.fixture(scope="module")
def tiny_gemma(tmp_path_factory) -> Tiny:
    """Gemma 4 with a vision tower (bidirectional image spans) and its MTP assistant, window 8: a text prompt, one
    with an image, a longer one."""
    from ..support.tiny_gemma import make_tiny_assistant, make_tiny_gemma_mm

    tmp = tmp_path_factory.mktemp("numerics-gemma")
    make_tiny_gemma_mm(tmp / "model", window=8, seed=5)
    make_tiny_assistant(tmp / "assistant", tmp / "model", seed=6)
    return forced_and_swept(tmp, tmp / "model", tmp / "assistant",
                            [[text(40, 0)], [text(20, 1), {"image": 0}, text(9, 2)], [text(150, 3)]], [0, 2, 1],
                            [2, 5], 70)


@pytest.fixture(params=["qwen", "gemma"])
def tiny(request) -> Tiny:
    return request.getfixturevalue(f"tiny_{request.param}")


def _rel(e: dict[str, float]) -> float:
    """The larger of an error's relative RMS and its max relative error (`numerics_fp32.errors`' sums)."""
    return max(math.sqrt(e["d2"] / e["r2"]) if e["r2"] else 0.0, e["dmax"] / e["rmax"] if e["rmax"] else 0.0)


def test_forced_captures(tiny: Tiny) -> None:
    """Every request's capture in both arms, in the shared format (loaded as numerics_fp32 loads them: memory-mapped,
    weights only): its tokens, spans and positions; the taps' stream identity in the runner's dtype (fp32 here, each
    sum rounded once: on BF16 the same identity in BF16); the final hidden state the final norm of the last stream (in
    fp32: Qwen's in BF16 normalises the sum before its rounding); the accepted counts along the engine's walk."""
    from llm_mini_infer.kernels import gemma_ops as gops

    ref = tiny.ref
    layers = [ref.layer(i) for i in range(len(ref.kinds))] if ref.family == "gemma" else []
    for req, q in zip(tiny.reqs, tiny.refs["requests"]):
        assert set(req.caps) == ARMS and req.rid == q["rid"]
        n = len(q["prompt"]) + ANSWER
        for arm, c in req.caps.items():
            assert c["tokens"].tolist() == q["prompt"] + q["answer"] and c["prompt_len"] == q["prompt_len"]
            assert c["forced"] == q["forced"] and [list(x) for x in c["images"]] == q["images"]
            assert c["settings"]["forced_prefix"] and c["settings"]["k"] == K and c["settings"]["draft_vocab"] == 0
            if ref.family == "qwen" and q["images"]:
                (off, _), = q["images"]
                assert c["rope"].shape == (3, n) and torch.equal(c["rope"][:, :off], torch.arange(off).expand(3, -1))
                assert int(c["rope"][0, -1]) < n - 1  # the decode positions past the image's M-RoPE offset
            else:
                assert c["rope"] is None
            t, hv = c["taps"], c["hvalid"]
            s, mx, ml, tv = t["stream"], t["mixer"], t["mlp"], t["tvalid"]
            assert t["kinds"] == ref.kinds and s.dtype == c["hidden"].dtype == torch.float32
            assert tv[:-1].all()  # every input is the reference's (the last token's only as an accepted draft)
            for i in range(len(ref.kinds)):
                if ref.family == "qwen":  # s_{l+1} = dtype(dtype(s_l + mixer) + mlp)
                    nxt = (s[i] + mx[i]) + ml[i]
                else:  # r_mid = rnd(r + rms(mixer) w_pa), r_{l+1} = rnd((r_mid + rms(mlp) w_pf) layer_scalar)
                    lay = layers[i]
                    mid = gops.sandwich(mx[i], s[i], lay.post_attention_layernorm.weight,
                                        lay.pre_feedforward_layernorm.weight, lay.eps)[0]
                    w = lay.post_feedforward_layernorm.weight
                    nxt = gops.sandwich(ml[i], mid, w, w, lay.eps, scalar=lay.layer_scalar)[0]
                assert torch.equal(nxt[tv], s[i + 1][tv]), (req.rid, arm, i)
            if arm == "on-eager" or ref.family == "qwen":  # the forward's states, or Qwen's hid_out: every token's
                assert torch.equal(hv, tv)
            else:  # Gemma 4's hid_out: the sample tokens' (the forced prefix's are not)
                assert hv[nf.round_start(req):n - 1].all() and not hv[:q["prompt_len"] - 1].any()
            both = hv & tv
            assert torch.equal(ref.final_norm(s[-1][both]), c["hidden"][both])
            pos, acc = c["accept_pos"].tolist(), c["accept"].tolist()
            assert len(pos) == len(acc) > 0 and pos[-1] < n - 1
            assert all(b == a + 1 + x for a, b, x in zip(pos, pos[1:], acc)), (req.rid, arm)  # the walk
            if arm == "on-eager":  # batch invariance: the final chunk takes the forced prefix whole
                assert pos[0] == q["prompt_len"] + q["forced"]
            gdn = [i for i in tiny.core if ref.kinds[i] == "gdn"]
            for i in gdn:
                g = t["core"][i]
                assert sum(m for _, m in g["rows"]) == q["prompt_len"] + c["forced"] or arm == "off-eager"
                assert all(not g["init"][j].any() for j, (p0, _) in enumerate(g["rows"]) if p0 == 0)
            for i in set(tiny.core) - set(gdn):
                a = t["core"][i]
                assert torch.equal(a["valid"], tv) and ("k" in a) == (ref.family == "qwen")


def test_fp32_layers_replicate_the_engine(tiny: Tiny) -> None:
    """numerics_fp32 `layers`: every layer's mixer, MLP and whole layer recomputed in fp32 from each arm's captured
    inputs, and each captured kernel from its own inputs, agree with the engine's to TOL (the GDN prefill rows, whose
    CPU kernel is the reference recurrence, exactly)."""
    for req in tiny.reqs:
        rid, res = req.rid, tiny.swept[req.rid].result
        assert [row["kind"] for row in res["layers"]] == tiny.ref.kinds
        for arm, c in req.caps.items():  # positions without taps (at most the last), from the other arm or the chain
            assert sum(res["fill"][arm].values()) == int((~c["taps"]["tvalid"]).sum()) <= 1
        for row in res["layers"]:
            for meas in nf.MEASURES:
                assert set(row[meas]) == ARMS
                for arm, bands in row[meas].items():
                    for band, e in bands.items():
                        assert e["n"] and _rel(e) < TOL, (rid, row["index"], meas, arm, band, _rel(e))
        assert {(c["index"], c["arm"]) for c in res["core"]} == {(i, arm) for i in tiny.core for arm in ARMS}
        for c in res["core"]:
            assert c["answer"]["n"] if c["kind"] != "gdn" else (c["rows"] and not c["skipped"])
            for band in nf.BANDS:
                assert _rel(c[band]) < TOL and (c["kind"] != "gdn" or c[band]["d2"] == 0), (rid, c)


def test_fp32_chain_reaches_the_engines_hidden_state(tiny: Tiny) -> None:
    """numerics_fp32 `chain`: the fp32 model from the `on` arm's embeddings (image rows included) gives every arm's
    streams and final hidden states, at the prompt's positions and the answer's, to CHAIN_TOL."""
    for rid, sw in tiny.swept.items():
        res = sw.result
        assert res["base"] == "on-eager" and len(res["stream"]) == len(tiny.ref.kinds) + 1
        assert sw.chain.hidden.shape == (res["n"], tiny.reqs[0].caps["on-eager"]["hidden"].shape[1])
        for i, row in enumerate(res["stream"]):
            for arm, bands in row.items():
                assert all(_rel(e) < CHAIN_TOL for e in bands.values()), (rid, i, arm, bands)
        for arm, bands in res["hidden"].items():
            assert bands["answer"]["n"] and max(map(_rel, bands.values())) < CHAIN_TOL, (rid, arm, bands)
        if tiny.ref.family == "gemma":
            assert set(sw.chain.kv) == {"sliding_k", "sliding_v", "full_v"}


def test_fp32_drafter_reproduces_the_engines_acceptance(tiny: Tiny) -> None:
    """numerics_fp32 `drafter`: on each arm's own hidden states the fp32 drafter accepts, at every round the engine
    verified, the drafts the engine accepted there (the CPU engine's drafter is the same fp32 arithmetic); on H_fp32
    too. The answers make the drafter right often: every accepted count from 0 to K occurs."""
    seen: list[int] = []
    results = [nf.drafter(tiny.ref, req, tiny.swept[req.rid].chain) for req in tiny.reqs]
    for req, res in zip(tiny.reqs, results):
        assert set(res["engine"]) == ARMS and res["k"] == K and res["start"] == nf.round_start(req)
        for arm, e in res["engine"].items():
            a = res["sources"][arm]
            bad = [(p, x, a[p - res["p0"]]) for p, x in e["rounds"] if a[p - res["p0"]] != x]
            assert e["rounds"] and not bad, (req.rid, arm, bad)
            assert e["same"] == {"on-eager": len(e["rounds"]), "off-eager": len(e["rounds"]), "fp32": len(e["rounds"])}
            seen += [x for _, x in e["rounds"]]
        assert res["engine"]["on-eager"]["rounds"][0][0] == res["start"]
    assert set(seen) == set(range(K + 1))
    assert "| fp32 |" in nf.drafter_tables({"requests": results})


def test_fp32_rounds_and_walk() -> None:
    """An arm's rounds sit at its windows' first positions − 1, each cut to the drafts the reference reaches (n = 20:
    the window at 17 verifies drafts at 18, 19 and the EOS's 20); the walk over a(p) from the drafter's first round."""
    c = {"accept_pos": torch.tensor([10, 12, 16, 17]), "accept": torch.tensor([1, 3, 0, 3])}
    assert nf.engine_rounds(c, 20) == [(9, 1), (11, 3), (15, 0), (16, 2)]
    assert nf.engine_rounds(c, 18) == [(9, 1), (11, 3), (15, 0)]  # the window at 17 starts on the last token
    a = [1, 0, 3, 0, 0, 0, 0, 2, 0, 1, 0]  # a(p) for p = 8, 9, ...
    assert nf.walk(a, 8, 9, 20) == [(9, 0), (10, 3), (14, 0), (15, 2)]


def test_fp32_layers_read_what_qwens_norms_read(tiny_qwen: Tiny, monkeypatch) -> None:
    """On BF16 captures (the CPU runs make fp32 ones, where no rounding shows) the fp32 layers start from what the
    engine's norms read: Qwen's fused add-norms normalise each fp32 sum and the stream keeps its rounding, so layer l's
    mixer reads u_l = mlp[l − 1] + rnd(stream[l − 1] + mixer[l − 1]) (u_0 = stream[0]) and its MLP stream[l] + mixer[l],
    both unrounded; a position an arm lacks comes from another arm's alike. The sweep's step feeds the mixer those."""
    req, bf, g = tiny_qwen.reqs[0], torch.bfloat16, torch.Generator().manual_seed(0)
    layers = len(tiny_qwen.ref.kinds)
    caps, sums = {}, {}
    for arm, c in req.caps.items():
        n, d = c["hidden"].shape
        mixer, mlp = (torch.randn(layers, n, d, generator=g).to(bf) for _ in range(2))
        u = [torch.randn(n, d, generator=g).to(bf).float()]
        for i in range(layers):
            u.append(mlp[i].float() + (u[i].to(bf).float() + mixer[i].float()).to(bf).float())
        sums[arm] = u
        caps[arm] = {**c, "hidden": c["hidden"].to(bf), "taps": {
            **c["taps"], "stream": torch.stack(u).to(bf), "mixer": mixer, "mlp": mlp,
            "tvalid": torch.ones(n, dtype=torch.bool)}}
    caps["off-eager"]["taps"]["tvalid"][5] = False
    sw = nf._Sweep(tiny_qwen.ref, nf.Request(req.rid, caps), layers=True, chain=False, base="on")
    assert sw.s is None and sw.res["fill"]["off-eager"] == {"from_arms": 1, "rest": 0}
    rounded = 0
    for i in range(layers):
        for arm, t in caps.items():
            want = sums[arm][i].clone()
            if arm == "off-eager":
                want[5] = sums["on-eager"][i][5]
            assert torch.equal(sw.mixer_in(arm, i), want), (arm, i)
            s, a = t["taps"]["stream"][i], t["taps"]["mixer"][i]
            assert torch.equal(tiny_qwen.ref.mid_engine(None, s, a), s.float() + a.float())
            rounded += int((want != s.float()).sum())
    assert rounded
    ref, seen = tiny_qwen.ref, []
    mixer = ref.mixer
    monkeypatch.setattr(ref, "mixer", lambda layer, x, *rest: (seen.append(x), mixer(layer, x, *rest))[1])
    sw.step(1, ref.kinds[1], ref.layer(1))
    assert all(torch.equal(seen[0][j], sw.mixer_in(arm, 1)) for j, arm in enumerate(sw.taps))


def test_core_layers_need_taps() -> None:
    """The in-situ kernel captures live in the taps (`taps.core`): core layers without taps are refused."""
    with pytest.raises(ValueError, match="--taps"):
        nfo.run_arm({"requests": []}, switch="on", graphs=False, core_layers=[1])


def test_fp32_target_agrees_with_itself(tiny: Tiny) -> None:
    """numerics_fp32 `target`: the fp32 head picks the same token from each arm's hidden state as from H_fp32 at
    every answer position the arm has."""
    for req in tiny.reqs:
        res = nf.target(tiny.ref, req, tiny.swept[req.rid].chain)
        assert set(res["arms"]) == ARMS and res["fp32"]["n"] == ANSWER
        for arm, e in res["arms"].items():
            assert e["n"] >= ANSWER - req.forced and e["agree"] == e["n"] and not e["dis_fp32_margins"], (req.rid, arm)


def test_fp32_command_line(tiny: Tiny, tmp_path, capsys) -> None:
    """`all` on the run's folder (both arms) and `drafter` reusing its chain, then `report`: JSON results, tables."""
    nf.main(["all", str(tiny.run), "--device", "cpu", "--out", str(tmp_path / "all")])
    for name in ("layers", "chain", "drafter", "target"):
        res = json.loads((tmp_path / "all" / f"{name}.json").read_text())
        assert [r["rid"] for r in res["requests"]] == [r.rid for r in tiny.reqs] and set(res["meta"]["arms"]) == ARMS
    nf.main(["drafter", str(tiny.run / "on-eager"), "--device", "cpu", "--chain", str(tmp_path / "all"),
             "--out", str(tmp_path / "drafter")])
    again = json.loads((tmp_path / "drafter" / "drafter.json").read_text())["requests"]
    first = json.loads((tmp_path / "all" / "drafter.json").read_text())["requests"]
    assert [r["sources"]["fp32"] for r in again] == [r["sources"]["fp32"] for r in first]
    capsys.readouterr()
    nf.main(["report", str(tmp_path / "all")])
    out = capsys.readouterr().out
    assert out == (tmp_path / "all" / "report.md").read_text() + "\n"
    assert all(f"## {title}" in out for _, title, _ in nf.SECTIONS)


def test_kernels_read_a_forced_capture(tiny_qwen: Tiny, monkeypatch) -> None:
    """gdn-accuracy's `--inputs` reads a forced capture (`taps.core[layer]`): its prefill rows with their initial
    states and outputs, which the fp32 recurrence reproduces from the inputs as the GPU kernel takes them (BF16)."""
    monkeypatch.setattr(nk, "DEV", "cpu")
    path = tiny_qwen.run / "on-eager" / "r2.pt"
    gdn = tiny_qwen.ref.kinds.index("gdn")
    core = torch.load(path, weights_only=True)["taps"]["core"][gdn]
    name, rows, outs = nk._load_inputs(str(path), gdn)
    assert name == f"r2.pt:layer{gdn}" and outs is not None and len(rows) == len(core["rows"]) > 1
    for row, o, (p0, n), init in zip(rows, outs, core["rows"], core["init"]):
        assert row["q"].shape[0] == n == o.shape[0] and torch.equal(row["s"], init) and (p0 > 0 or not init.any())
        assert nk.rel_errors(nk.gdn_fp32([row])[0][0], o)["rel_rms"] < 1e-2


@pytest.mark.parametrize("module", ["numerics_kernels", "numerics_forced", "numerics_fp32"])
def test_help(module: str) -> None:
    """`python3 -m tests.scripts.<tool> --help` and each subcommand's, on any machine (torch and FlashInfer load inside
    the commands)."""
    env = {k: v for k, v in os.environ.items() if k != "HF_TOKEN"}
    root = ROOT
    top = subprocess.run([sys.executable, "-m", f"tests.scripts.{module}", "--help"], cwd=root, env=env,
                         capture_output=True, text=True, timeout=120)
    assert top.returncode == 0 and "usage: python3 -m tests.scripts." in top.stdout, top.stderr
    subs = {"numerics_kernels": ["smid", "gdn-race", "gdn-accuracy", "attn-accuracy", "nan-leak", "timing",
                                 "attn-timing"],
            "numerics_forced": ["refs", "run"],
            "numerics_fp32": ["layers", "chain", "drafter", "target", "all", "report"]}[module]
    main = importlib.import_module(f"tests.scripts.{module}").main
    for sub in subs:
        with pytest.raises(SystemExit) as exc, contextlib.redirect_stdout(io.StringIO()) as out:
            main([sub, "--help"])
        assert exc.value.code == 0 and sub in out.getvalue().split(), (module, sub)  # argparse wraps at its width
