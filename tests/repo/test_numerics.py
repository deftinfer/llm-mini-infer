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
    """The default grid's steps and their exact lengths, and the prefill split bases a step is timed at."""
    a = nk.build_parser().parse_args(["attn-timing"])
    k = a.k
    cases = nk.timing_cases(nfo.parse_ids(a.short_chunks), nfo.parse_ids(a.histories), a.long_chunk,
                            nfo.parse_ids(a.long_histories), nfo.parse_ids(a.decode_rows),
                            nfo.parse_ids(a.decode_histories), k)
    by = {c.name: c for c in cases}
    assert len(cases) == len(by) == 2 * 3 + 2 + 2 * 3 and a.max_model_len == 262144 and a.prefill_from == 8192
    assert all(not c.uncertain for c in cases)
    c = by["prefill-short-256-h131072"]
    assert (c.n_dec, c.nq.tolist(), c.ub.tolist()) == (0, [256], [131072 + 256])
    c = by["prefill-long-2048-h0"]
    assert (c.n_dec, c.nq.tolist(), c.lb.tolist()) == (0, [2048], [2048]) and nk.case_history(c) == 0
    c = by["decode-64-h32768"]
    assert c.n_dec == 64 and (c.nq == 4).all() and (c.ub == 32768 + 4).all() and nk.case_history(c) == 32768
    np.testing.assert_array_equal(c.starts, np.arange(64) * 4)
    sps = nfo.parse_ids(a.prefill_keys)
    assert sps == [0]  # each part's own base
    sps = [8192, 4096, 2048]
    for c in cases:  # a prefill step at each base on the full layers; decode steps and the sliding layers once
        assert nk.timing_variants(c, sps, True) == ([None] if c.n_dec else sps)
        assert nk.timing_variants(c, sps, False) == [None]


def test_timing_ratios() -> None:
    def rec(impl: str, mode: str, us: float | None, sp: int | None = None) -> dict:
        return {"case": "c", "impl": impl, "mode": mode, "sp": sp, **({} if us is None else {"median_us": us})}

    recs = [rec("off", "eager", 50.0), rec("off", "graph", 20.0), rec("on", "eager", 25.0, 8192),
            rec("on", "graph", 30.0, 8192), rec("on", "eager", None, 4096), rec("on", "bucket", 5.0, 8192)]
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
    assert all(f"## {title}" in out for name, title, _ in nf.SECTIONS if name != "pairs")  # `all` runs no pairs


# -- the KV fake-quant (numerics_forced --kv-dtype) and arm against arm (numerics_fp32 pairs) -----------------------


def e4m3(x: float) -> float:
    """float8_e4m3fn of x by hand: |x| clamped to 448, then to the nearest of its binade's 8 steps (subnormals: steps of
    2^-9 below 2^-6), ties to the even step; NaN stays NaN."""
    if math.isnan(x):
        return x
    a = min(abs(x), nfo.E4M3_MAX)
    if a == 0:
        return math.copysign(0.0, x)
    step = 2.0 ** (max(math.floor(math.log2(a)), -6) - 3)
    q, frac = divmod(a / step, 1.0)
    q += frac > 0.5 or (frac == 0.5 and q % 2 == 1)
    return math.copysign(min(q * step, nfo.E4M3_MAX), x)


def test_e4m3_round_against_the_format() -> None:
    """`e4m3_round` equals the format by hand on every finite BF16 value (ties to even, subnormals, the largest finite
    448, flush to zero at or below half the smallest subnormal), saturates past 448 and infinities to ±448
    (`.satfinite`), keeps NaN, keeps x's dtype; at a power-of-two scale, the format's grid scaled."""
    bits = torch.arange(1 << 16, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    x = bits[torch.isfinite(bits)]
    got = nfo.e4m3_round(x)
    assert got.dtype == torch.bfloat16
    want = torch.tensor([e4m3(v) for v in x.float().tolist()])
    assert torch.equal(got.float(), want)
    special = torch.tensor([1.0625, 1.1875, 3 * 2.0**-10, 2.0**-10, 2.0**-9, 449.0, 464.0, 1e4, -1e30, math.inf,
                            -math.inf, math.nan])
    out = nfo.e4m3_round(special).tolist()
    assert out[:-1] == [1.0, 1.25, 2.0**-8, 0.0, 2.0**-9, 448.0, 448.0, 448.0, -448.0, 448.0, -448.0]
    assert math.isnan(out[-1])
    some = x.float()[::97]
    assert nfo.e4m3_round(some, 4.0).tolist() == [4 * e4m3(v / 4) for v in some.tolist()]
    assert nfo.pow2_scale(448.0) == 1.0 and nfo.pow2_scale(449.0) == 2.0 and nfo.pow2_scale(1000.0) == 4.0
    assert nfo.pow2_scale(math.nan) == 1.0


def test_kv_fake_quant_rounds_the_written_slots_and_counts() -> None:
    """`KvFakeQuant.round` rounds exactly the slots at (page, off), K and V planes or one plane, at the layer's scale;
    every other slot keeps its bits. Its counters take the real rows only (padding rows write the null page, 0): the
    values written, those past 448 × the scale and the largest magnitude, per layer and plane; `report` names the
    power-of-two scales they ask for; `reset` zeroes them in place."""
    g = torch.Generator().manual_seed(0)
    two = (torch.randn((5, 2, 4, 2, 8), generator=g) * 30).to(torch.bfloat16)  # [pages, K/V, page_size, heads, dim]
    one = (torch.randn((5, 4, 2, 8), generator=g) * 30).to(torch.bfloat16)  # a k_eq_v layer: V alone
    page, off = torch.tensor([2, 3, 0, 0]), torch.tensor([1, 3, 2, 2])  # two real rows, two padding rows
    two[3, 1, 3, 0, 0], two[0, 1, 2, 1, 1], one[2, 1, 1, 5] = 1000.0, 5000.0, -600.0
    before2, before1 = two.clone(), one.clone()
    kvq = nfo.KvFakeQuant("fp8", "cpu", {("full", 0): (1.0, 2.0)})
    kvq.round("full", 0, {0: two[:, 0], 1: two[:, 1]}, page, off)
    kvq.round("full", 1, {1: one}, page, off)
    written = torch.zeros((5, 4), dtype=torch.bool)
    written[page, off] = True
    for p, (now, was, s) in enumerate([(two[:, 0], before2[:, 0], 1.0), (two[:, 1], before2[:, 1], 2.0),
                                       (one, before1, 1.0)]):
        assert torch.equal(now[written], nfo.e4m3_round(was[written], s)), p
        assert torch.equal(now[~written], was[~written]), p
    assert float(two[3, 1, 3, 0, 0]) == 896.0 and float(one[2, 1, 1, 5]) == -448.0  # 2 · 448, and the clamp
    rep = kvq.report()
    assert list(rep["layers"]) == ["full:0", "full:1"] and rep["dtype"] == "fp8" and rep["clamp"] == 448.0
    assert rep["layers"]["full:0"]["values"] == [32, 32] and rep["layers"]["full:1"]["values"] == [0, 32]
    assert rep["layers"]["full:0"]["saturated"] == [0, 1] and rep["layers"]["full:1"]["saturated"] == [0, 1]
    real = before2[page[:2], :, off[:2]].float().abs()
    assert rep["layers"]["full:0"]["amax"] == [real[:, 0].max().item(), 1000.0]  # not the null page's 5000
    assert rep["pow2_scales"] == {"full:0": [1.0, 4.0], "full:1": [1.0, 2.0]}
    assert rep["values"] == 96 and rep["saturated"] == 2 and rep["amax"] == 1000.0
    assert rep["scales"] == {"full:0": [1.0, 2.0]} and kvq.scaled
    kvq.reset()
    assert kvq.report()["values"] == 0 and not kvq.amax.any()
    with pytest.raises(ValueError, match="powers of two"):
        nfo.KvFakeQuant("fp8", "cpu", {("full", 0): (1.0, 3.0)})
    with pytest.raises(ValueError, match="KV dtype"):
        nfo.KvFakeQuant("fp16", "cpu")


def test_kv_scales_file(tmp_path) -> None:
    """`--kv-scales`: a JSON object of per-layer scales, or a run.json whose `kv_fake_quant.pow2_scales` is one."""
    (tmp_path / "s.json").write_text(json.dumps({"full:3": [1, 2], "sliding:0": [4, 1]}))
    assert nfo.load_kv_scales(tmp_path / "s.json") == {("full", 3): (1.0, 2.0), ("sliding", 0): (4.0, 1.0)}
    run = {"arm": "off-kv8", "kv_fake_quant": {"pow2_scales": {"full:0": [2, 1]}}}
    (tmp_path / "run.json").write_text(json.dumps(run))
    assert nfo.load_kv_scales(tmp_path / "run.json") == {("full", 0): (2.0, 1.0)}
    (tmp_path / "bf16.json").write_text(json.dumps({"arm": "off", "kv_fake_quant": None}))
    with pytest.raises(ValueError, match="without --kv-dtype"):
        nfo.load_kv_scales(tmp_path / "bf16.json")
    (tmp_path / "bad.json").write_text(json.dumps({"layer3": [1, 2]}))
    with pytest.raises(ValueError, match="KV scales"):
        nfo.load_kv_scales(tmp_path / "bad.json")


def test_arm_names_and_twins() -> None:
    """`<switch>[-kv8|-kv8all][-cache][-eager]` (`s` after the KV tag with scales), and each FP8-KV arm's twin."""
    assert nfo.arm_name("on", False, True) == "on" and nfo.arm_name("off", True, False) == "off-cache-eager"
    assert nfo.arm_name("on", True, False, "fp8") == "on-kv8-cache-eager"
    assert nfo.arm_name("on", False, True, "fp8-all") == "on-kv8all"
    assert nfo.arm_name("off", False, True, "fp8", kv_scaled=True) == "off-kv8s"
    for arm, twin in [("on-kv8", "on"), ("on-kv8-cache-eager", "on-cache-eager"), ("on-kv8all", "on"),
                      ("off-kv8s", "off"), ("on", None), ("on-cache", None), ("off-eager", None)]:
        assert nf.twin(arm) == twin, arm
    assert nf.default_pairs(["on", "on-kv8", "off-kv8", "on-kv8all", "on-cache"]) == [("on-kv8", "on"),
                                                                                       ("on-kv8all", "on")]


def _spy_attention(monkeypatch, seen: dict[tuple[str, int], list[bool]]) -> None:
    """Records, at each attention call of a layer whose KV the step just stored (the assistant's layers store none),
    whether the slots it wrote hold E4M3 values (x == e4m3_round(x)), by (kind, pool index): the rounding happens
    between the store and the attention's read. The store spies go in first, so the fake-quant wraps them."""
    from llm_mini_infer.models.gemma4.backend import SLIDING, Gemma4Context
    from llm_mini_infer.models.qwen3_5.backend import ForwardContext

    def on_grid(x: torch.Tensor) -> bool:
        return bool(torch.equal(x, nfo.e4m3_round(x)))

    gstore, qstore = Gemma4Context.qkv_store, ForwardContext.qk_rope_store
    gemma, qwen = Gemma4Context.attention, ForwardContext.attention
    stored: set[Any] = set()

    def gemma_store(ctx: Any, layer: Any, qkv: torch.Tensor) -> torch.Tensor:
        stored.add(id(layer))
        return gstore(ctx, layer, qkv)

    def qwen_store(ctx: Any, layer: Any, qkv: torch.Tensor) -> torch.Tensor:
        stored.add(layer.index)
        return qstore(ctx, layer, qkv)

    def gemma_attention(ctx: Any, layer: Any, q: torch.Tensor) -> torch.Tensor:
        m, p = ctx.meta, ctx.pools
        if id(layer) in stored:
            stored.discard(id(layer))
            if layer.kind == SLIDING:
                kv = p.wkv[layer.index]
                seen.setdefault(("sliding", layer.index), []).append(
                    on_grid(kv[:, 0][m.w_page, m.w_off]) and on_grid(kv[:, 1][m.w_page, m.w_off]))
            else:
                seen.setdefault(("full", layer.index), []).append(on_grid(p.fkv[layer.index][m.f_page, m.f_off]))
        return gemma(ctx, layer, q)

    def qwen_attention(ctx: Any, index: int, q: torch.Tensor) -> torch.Tensor:
        kv, m = ctx.pools.kv[index], ctx.meta
        if index in stored:
            stored.discard(index)
            seen.setdefault(("full", index), []).append(
                on_grid(kv[:, 0][m.kv_page, m.kv_off]) and on_grid(kv[:, 1][m.kv_page, m.kv_off]))
        return qwen(ctx, index, q)

    for cls, name, f in ((Gemma4Context, "qkv_store", gemma_store), (ForwardContext, "qk_rope_store", qwen_store),
                         (Gemma4Context, "attention", gemma_attention), (ForwardContext, "attention", qwen_attention)):
        monkeypatch.setattr(cls, name, f)


def _kv8_run(tiny: Tiny, out: Path, kv_dtype: str) -> dict[str, Any]:
    opts = nfo.Options(device="cpu", max_model_len=1024, budget=128, max_num_seqs=4, memory_gb=0.05, k=K,
                       soft_tokens=tiny.refs["options"]["soft_tokens"])
    return nfo.run_arm(tiny.refs, switch="on", graphs=False, out=out, opts=opts, kv_dtype=kv_dtype)


def test_kv_fake_quant_in_the_engine(tiny: Tiny, tmp_path, monkeypatch) -> None:
    """A forced run with `--kv-dtype fp8` on the tiny models (CPU, eager): every full-attention layer's slots written in
    a step hold E4M3 values when that step's attention reads them (Qwen's MTP layer included), Gemma 4's sliding
    layers' do not; run.json counts every full layer (Gemma 4's: V alone) and nothing saturated; the store methods are
    the originals again afterwards; the arm's hidden states differ from the BF16 arm's. Then `pairs` on the two arms:
    the default pair, positions and tokens per verify step from the captures."""
    from llm_mini_infer.models.gemma4.backend import Gemma4Context
    from llm_mini_infer.models.qwen3_5.backend import ForwardContext

    seen: dict[tuple[str, int], list[bool]] = {}
    _spy_attention(monkeypatch, seen)
    stores = Gemma4Context.qkv_store, ForwardContext.qk_rope_store
    res = _kv8_run(tiny, tmp_path, "fp8")
    assert (Gemma4Context.qkv_store, ForwardContext.qk_rope_store) == stores
    assert res["arm"] == "on-kv8-eager" and not res["wrong"] and (tmp_path / "on-kv8-eager" / "run.json").exists()
    kinds, kv = tiny.ref.kinds, res["kv_fake_quant"]
    full = [("full", i) for i in range(kinds.count("full") + (tiny.ref.family == "qwen"))]  # Qwen's MTP layer last
    assert [key for key in seen if key[0] == "full"] == full and all(all(seen[key]) for key in full)
    assert res["settings"]["kv_fake_quant"]["layers"] == "full" and kv["saturated"] == 0 and kv["values"] > 0
    assert list(kv["layers"]) == [f"full:{i}" for _, i in full]
    if tiny.ref.family == "gemma":
        sliding = [key for key in seen if key[0] == "sliding"]
        assert len(sliding) == kinds.count("sliding") and not any(any(seen[key]) for key in sliding)
        assert all(e["values"][0] == 0 < e["values"][1] for e in kv["layers"].values())
    else:
        assert all(0 < e["values"][0] == e["values"][1] for e in kv["layers"].values())
    caps = nf.load_requests([tiny.run / "on-eager", tmp_path / "on-kv8-eager"])
    assert any(not torch.equal(r.caps["on-kv8-eager"]["hidden"], r.caps["on-eager"]["hidden"]) for r in caps)
    nf.main(["pairs", str(tiny.run / "on-eager"), str(tmp_path / "on-kv8-eager"), "--device", "cpu", "--boot", "200",
             "--out", str(tmp_path / "pairs")])
    got = json.loads((tmp_path / "pairs" / "pairs.json").read_text())
    assert got["meta"]["boot"] == 200 and [r["pair"] for r in got["requests"]] == [["on-kv8-eager", "on-eager"]] * 3
    for r, req in zip(got["requests"], caps):
        assert r["workload"] == "tiny" and r["bands"]["all"]["n"] == ANSWER
        assert r["bands"]["free"]["n"] == ANSWER - req.forced and "fp32_a" not in r["bands"]["all"]
        for arm in r["pair"]:
            assert r["accept"][arm]["steps"] == len(req.caps[arm]["accept"]) > 0
    assert "### Top-1 agreement" in nf.report(tmp_path / "pairs")


def test_kv_fake_quant_on_every_layer(tiny_gemma: Tiny, tiny_qwen: Tiny, tmp_path, monkeypatch) -> None:
    """`--kv-dtype fp8-all` rounds Gemma 4's sliding layers too (K and V); Qwen3.8 has no sliding layers to add."""
    seen: dict[tuple[str, int], list[bool]] = {}
    _spy_attention(monkeypatch, seen)
    res = _kv8_run(tiny_gemma, tmp_path, "fp8-all")
    kinds = tiny_gemma.ref.kinds
    assert res["arm"] == "on-kv8all-eager" and len(seen) == len(kinds) and all(all(v) for v in seen.values())
    kv = res["kv_fake_quant"]["layers"]
    assert {k for k in kv if k.startswith("sliding:")} == {f"sliding:{i}" for i in
                                                            range(tiny_gemma.ref.kinds.count("sliding"))}
    assert all(0 < e["values"][0] == e["values"][1] for k, e in kv.items() if k.startswith("sliding:"))
    with pytest.raises(ValueError, match="fp8-all"):
        _kv8_run(tiny_qwen, tmp_path, "fp8-all")


def _cap(tokens: list[int], prompt_len: int, forced: int, hidden: torch.Tensor, accept: list[int]) -> dict[str, Any]:
    n = len(tokens)
    return {"tokens": torch.tensor(tokens), "prompt_len": prompt_len, "forced": forced, "hidden": hidden,
            "hvalid": torch.ones(n, dtype=torch.bool), "accept": torch.tensor(accept),
            "accept_pos": torch.arange(len(accept)), "images": [], "rope": None, "settings": {}}


def test_pair_counts_and_the_cluster_bootstrap() -> None:
    """`pair` on hand-made captures (a head of 8 tokens, the identity: each hidden row's argmax is its largest entry):
    the agreements over all answer positions and the free ones (past the forced prefix), the first miss's index in the
    answer, the discordant counts against H_fp32, the margins; `pair_summary` pools requests: the one-sided lower bound
    of a cluster bootstrap below the share (equal to it when every request agrees alike), tokens per verify step."""
    ref = SimpleNamespace(dev=torch.device("cpu"), dtype=torch.float32, head=torch.eye(8), softcap=None)

    def rows(ids: list[int]) -> torch.Tensor:
        return torch.nn.functional.one_hot(torch.tensor(ids), 8).float() * 2 + 0.1 * torch.arange(8.0)

    prompt, answer = [1, 2, 3], [4, 5, 6, 7, 1, 2]  # positions 2..7 predict the answer; the first 2 forced
    toks = prompt + answer
    base = [0, 0, 4, 5, 6, 7, 1, 2, 0]
    other = [0, 0, 3, 5, 6, 0, 1, 2, 0]  # misses at positions 2 (forced) and 5 (free: answer index 3)
    fp32 = [0, 0, 4, 5, 6, 0, 1, 3, 0]
    req = nf.Request(0, {"on": _cap(toks, 3, 2, rows(base), [1, 0, 2]),
                         "on-kv8": _cap(toks, 3, 2, rows(other), [0, 0, 1, 3])})
    chain = SimpleNamespace(hidden=rows(fp32))
    r = nf.pair(ref, req, "on-kv8", "on", chain)
    assert r["bands"]["all"] == {"n": 6, "agree": 4, "fp32_a": 4, "fp32_b": 4, "b": 1, "c": 1}
    assert r["bands"]["free"] == {"n": 4, "agree": 3, "fp32_a": 3, "fp32_b": 2, "b": 1, "c": 0}
    assert r["first_miss"] == 3 and len(r["margins"]) == 4 and len(r["miss_margins"]) == 1
    assert r["accept"] == {"on-kv8": {"steps": 4, "accepted": 4}, "on": {"steps": 3, "accepted": 3}}
    same = nf.Request(1, {"on": _cap(toks, 3, 2, rows(base), [1]), "on-kv8": _cap(toks, 3, 2, rows(base), [1])})
    r1 = nf.pair(ref, same, "on-kv8", "on")
    assert r1["first_miss"] is None and r1["bands"]["free"] == {"n": 4, "agree": 4} and not r1["miss_margins"]
    s = nf.pair_summary([r, r1], "free", draws=2000)
    assert (s["n"], s["agree"], s["share"]) == (8, 7, 7 / 8) and s["lower95"] == 0.75  # resamples of r alone: 3/4
    assert s["tpv"] == {"on-kv8": 1 + 5 / 5, "on": 1 + 4 / 4} and s["tpv_delta"][0] == 0.0
    assert "fp32" not in s and nf.pair_summary([r1, r1], "free", draws=100)["lower95"] == 1.0
    f = nf.pair_summary([r], "all", draws=100)["fp32"]
    assert (f["fp32_a"], f["fp32_b"], f["b"], f["c"]) == (4, 4, 1, 1) and f["delta"][0] == 0.0
    a = nf.bootstrap({"x": np.array([1.0, 0.0])}, lambda c: c["x"], draws=500)
    assert np.array_equal(a, nf.bootstrap({"x": np.array([1.0, 0.0])}, lambda c: c["x"], draws=500))  # seeded
    tables = nf.pairs_tables({"meta": {"boot": 100},
                              "requests": [{**r, "workload": "chat"}, {**r1, "workload": "scene"}]})
    assert "| on-kv8 : on | chat | free |" in tables and "| on-kv8 : on | all |" in tables


def test_long_prompts_are_refused_and_recall_turns_chat_around(tiny_qwen: Tiny) -> None:
    """refs refuses a prompt whose answer (+ 2, the forced run's) could pass max_model_len; `recall` turns chat's
    instruction around, so bench/load/run.py's chat_workload must still ask for `REVERSE`."""
    import inspect

    req = [dict(rid=0, workload="tiny", index=0, parts=[text(40, 0)], grammar="", forced=0)]
    opts = nfo.Options(device="cpu", max_model_len=65, budget=128, max_num_seqs=4, memory_gb=0.05)
    with pytest.raises(ValueError, match="max_model_len 65"):
        nfo.make_refs(tiny_qwen.run.parent / "model", req, opts=opts, max_tokens=24)
    assert nfo.REVERSE in inspect.getsource(nfo._bench_load().chat_workload)


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
            "numerics_fp32": ["layers", "chain", "drafter", "target", "all", "pairs", "report"]}[module]
    main = importlib.import_module(f"tests.scripts.{module}").main
    for sub in subs:
        with pytest.raises(SystemExit) as exc, contextlib.redirect_stdout(io.StringIO()) as out:
            main([sub, "--help"])
        assert exc.value.code == 0 and sub in out.getvalue().split(), (module, sub)  # argparse wraps at its width
