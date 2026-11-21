"""The Qwen runner's batch-invariant kernels on CUDA (docs/architecture.md, "Batch invariance"): full attention on
`attn_window`'s kernels with no window at Qwen3.8's GQA group of 6 (`kernels.attn_window.FullAttention`), the BF16
linear whose rows' bits do not depend on M (`ops.linear_rows`, in_proj_ba), and FlashInfer's GDN prefill pinned to its
per-sequence kernel on the state pool, with checkpoints taken by cutting rows
(`kernels.flashinfer.flashinfer_gdn_prefill`, both paths'). Each against its reference, and its bits alone against
beside other rows, cut otherwise, or at another M. The GDN tests are FlashInfer's: an upgrade that breaks its
per-sequence bits fails them.

CUDA and the kernel library only (FlashInfer for the GDN tests)."""

from __future__ import annotations

import itertools
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from llm_mini_infer.kernels import ops

from ..support.kernel_cases import (
    BF,
    DEV,
    _gdn_call,
    _gdn_seq,
    _gdn_step,
    _packed_reference,
    _run,
    _stage,
    _window_ref,
    _window_setup,
)

pytestmark = pytest.mark.gpu(reason="needs a CUDA GPU and the kernel library")

H, HKV, D, PS = 24, 4, 256, 16  # Qwen3.8 27B's full layers: a GQA group of 6


def _full(max_rows: int = 24, qo: int = 4, split_keys: int = 256, max_kv: int = 4096, pool=None, **split):
    from llm_mini_infer.kernels.attn_window import FullAttention

    return FullAttention(SimpleNamespace(kv=pool[None], page_size=PS), H, HKV, D, torch.device(DEV), max_tokens=4096,
                         max_rows=max_rows, max_kv=max_kv, scale=D ** -0.5, qo_per_row=qo, split_keys=split_keys,
                         **split)


@pytest.mark.parametrize("split_keys", [96, 128, 256])  # 32 splits hold the longest row, 3,001 keys
def test_full_attention_matches_reference(split_keys) -> None:
    """Decode rows of 1 and 4 queries (split at absolute multiples of `split_keys`) and prefill rows (whole, tiles of 64
    query rows: 10 and 2/3 positions at a group of 6), from position 0 and resumed: the fp32 reference within BF16
    rounding."""
    lens = [5, 300, 1030, 2700, 3001, 900, 2100]
    pool, table = _window_setup(lens, HKV, D, PS, seed=3)
    a = _full(max_rows=8, qo=4, split_keys=split_keys, max_kv=3072, pool=pool)
    spec = [(1, 0, 4), (2, 296, 4), (3, 1026, 4), (4, 2699, 1), (5, 2997, 4), (6, 0, 900), (7, 1900, 200)]
    qs = [(torch.randn(n, H, D, device=DEV) * 0.5).to(BF) for _, _, n in spec]
    rows, at = [], 0
    for slot, p0, n in spec:
        rows.append((slot, at, n, p0 + n))
        at += n
    got = _run(a, table, rows, torch.cat(qs), decode=5)
    at = 0
    for (slot, p0, n), q in zip(spec, qs):
        want = _window_ref(pool, table, slot, (q.float() * D ** -0.5).to(BF), p0, 1 << 30, PS)
        err = (got[at:at + n].float() - want).abs()
        assert torch.isfinite(got[at:at + n]).all(), slot
        assert err.max().item() < 2e-2 and err.mean().item() < 2e-3, (slot, err.max().item(), err.mean().item())
        at += n


def test_full_attention_is_batch_invariant() -> None:
    """A query at a given position gives the same bits as a decode row's query alone or beside other rows, at any offset
    of its verify window, planned for its true length or an upper bound (MTP), in a decode bucket's static tables with
    padding rows, and beside a prefill row; and the same bits again as a prefill row's query whatever its chunk (where
    its position's 6 heads straddle two tiles or not) and the rows beside it."""
    qo = 4
    lens = [3200, 2700, 3050, 1200, 400, 2999, 33, 2600, 1800, 2100]  # slot 1 the subject's
    pool, table = _window_setup(lens, HKV, D, PS, seed=7)
    torch.manual_seed(8)
    qsub = (torch.randn(lens[0], H, D, device=DEV) * 0.5).to(BF)
    a = _full(max_rows=24, qo=qo, split_keys=256, pool=pool)

    def go(spec, decode: int | None = None, bucket: int = 0):
        rows, qs, true, at, where = [], [], [], 0, {}
        for slot, p0, n, extra in spec:
            rows.append((slot, at, n, p0 + n + extra))
            true.append(p0 + n)
            qs.append(qsub[p0:p0 + n] if slot == 1 else (torch.randn(n, H, D, device=DEV) * 0.5).to(BF))
            if slot == 1:
                where.update({p0 + i: at + i for i in range(n)})
            at += n
        real = len(rows) if decode is None else decode
        for _ in range(bucket - len(rows)):  # padding rows: slot 0's null page, 1 + k keys
            rows.append((0, at, qo, qo))
            true.append(qo)
            qs.append(torch.zeros(qo, H, D, dtype=BF, device=DEV))
            at += qo
        out = _run(a, table, rows, torch.cat(qs), decode=real, bucket=bucket, true=true)
        return {p: out[i] for p, i in where.items()}

    others = [(s, lens[s - 1] - qo, qo, 0) for s in range(2, 10)]
    for p in (100, 255, 256, 1023, 1040, 2559, 2600, 3190):
        def at(j: int, extra: int = 0, p: int = p) -> tuple[int, int, int, int]:
            return (1, p - j, qo, extra)

        decode = [go([at(0)]), go([at(3, 5)]), go([*others, at(2)]), go([at(1, 2), *others[:2]], bucket=8),
                  go([*others, at(3)], bucket=24), go([at(2), (2, 600, 40, 0)], decode=1)]
        for c, got in enumerate(decode[1:], 1):
            assert torch.equal(decode[0][p], got[p]), (p, c)
        prefill = [go([(1, p, 1, 0)], decode=0), go([(1, p - 37, 40, 0), *others[:3]], decode=0),
                   go([*others[:2], (1, max(0, p - 300), min(p, 300) + 1, 0)], decode=2),
                   go([(1, p - 5, 9, 0)], decode=0)]
        for c, got in enumerate(prefill[1:], 1):
            assert torch.equal(prefill[0][p], got[p]), (p, c)
        want = _window_ref(pool, table, 1, (qsub[p:p + 1].float() * D ** -0.5).to(BF), p, 1 << 30, PS)[0]
        for got in (decode[0][p], prefill[0][p]):
            assert (got.float() - want).abs().max().item() < 2e-2, p


def _step(a, table, rows, q, *, decode: int, bucket: int = 0, true=None, lb=None):
    """`_run` with the rows' lowest possible KV lengths `lb` (per row; None: the planned ones) given to the plan, and
    the step's meta returned beside the output: (output, meta)."""
    rr = np.asarray(rows, np.int32)
    kv = torch.as_tensor(rr[:, 3] if true is None else np.asarray(true), dtype=torch.int64, device=DEV)
    meta = SimpleNamespace(page_table=table, fix_len=True, n_rows=len(rr), kv_len_dev=kv)
    if bucket:
        a.plan_graph(meta, bucket=bucket, rows=rr, stage=_stage, real=decode, lb=lb)
    else:
        a.plan(meta, parity=0, rows=rr, stage=_stage, fixed_rows=decode, lb=lb)
    out = torch.full_like(q, float("nan"))
    a.run(meta, 0, q, out)
    torch.cuda.synchronize()
    return out, meta


def _scaled(q: torch.Tensor) -> torch.Tensor:
    """`q` as the reference takes it: the kernel's softmax scale folded into the BF16 queries."""
    return (q.float() * D ** -0.5).to(BF)


def test_full_attention_query_sizes_are_batch_invariant() -> None:
    """With the split small (`split_keys` 16: a query's split doubles past 512 keys and again past 1,024) a verify row's
    queries, at positions inside a size and straddling the two size boundaries (a row's 4 queries at 6 heads each
    straddle tiles too), give the same bits whatever carries them: the row alone, at another offset of its window,
    beside other verify rows, in a decode bucket's static tables with padding rows, planned for an upper bound of its
    length over a lower bound (the true lengths on the device, as under MTP), and beside a prefill row; and each
    within BF16 rounding of the fp32 reference."""
    qo = 4
    lens = [1300, 1100, 520, 1030, 200, 700, 33, 1290, 600]  # slot 1 the subject's
    pool, table = _window_setup(lens, HKV, D, PS, seed=11)
    torch.manual_seed(12)
    qsub = (torch.randn(lens[0], H, D, device=DEV) * 0.5).to(BF)
    a = _full(max_rows=12, qo=qo, split_keys=16, pool=pool, prefill_from=64, prefill_keys=16)

    def go(spec, decode: int | None = None, bucket: int = 0):
        """spec: (slot, first position, queries, planned extra over the true length, true length over its lower bound);
        the subject's (slot 1) queries by position."""
        rows, qs, true, low, at, where = [], [], [], [], 0, {}
        for slot, p0, n, extra, drop in spec:
            rows.append((slot, at, n, p0 + n + extra))
            true.append(p0 + n)
            low.append(max(n, p0 + n - drop))
            qs.append(qsub[p0:p0 + n] if slot == 1 else (torch.randn(n, H, D, device=DEV) * 0.5).to(BF))
            if slot == 1:
                where.update({p0 + i: at + i for i in range(n)})
            at += n
        real = len(rows) if decode is None else decode
        for _ in range(bucket - len(rows)):  # padding rows: slot 0's null page, 1 + k keys
            rows.append((0, at, qo, qo))
            true.append(qo)
            low.append(qo)
            qs.append(torch.zeros(qo, H, D, dtype=BF, device=DEV))
            at += qo
        out, _ = _step(a, table, rows, torch.cat(qs), decode=real, bucket=bucket, true=true, lb=np.asarray(low))
        return {p: out[i] for p, i in where.items()}

    others = [(s, lens[s - 1] - qo, qo, 0, 0) for s in range(2, 10)]
    for p in (5, 300, 508, 509, 510, 511, 512, 515, 1021, 1022, 1023, 1024, 1027, 1200):
        def at(j: int, extra: int = 0, drop: int = 0, p: int = p) -> tuple[int, int, int, int, int]:
            return (1, p - j, qo, extra, drop)

        got = [go([at(0)]), go([at(3)]), go([at(1, 5), *others[:2]]), go([*others, at(2)]),
               go([at(1, 2, 3), *others[:2]], bucket=8), go([at(3, 40, 40), *others], bucket=16),
               go([at(2, 600, 3)]), go([at(0, 9, 1), *others[2:5]], bucket=8),
               go([at(2), (2, 600, 40, 0, 0)], decode=1), go([*others[:3], at(1), (3, 400, 77, 0, 0)], decode=4)]
        for c, cand in enumerate(got[1:], 1):
            for pos in cand.keys() & got[0].keys():
                assert torch.equal(got[0][pos], cand[pos]), (p, pos, c)
        want = _window_ref(pool, table, 1, _scaled(qsub[p:p + 1]), p, 1 << 30, PS)[0]
        assert (got[0][p].float() - want).abs().max().item() < 2e-2, p


def test_full_attention_prefill_split_is_chunk_invariant() -> None:
    """With the prefill split small (`prefill_from` 64 keys, `prefill_keys` 16: past 512 keys the split doubles) every
    query of a prompt gives the same bits as one prefill row and as chunks of different lengths and offsets (a chunk
    starting below 64 and ending past it, so a tile straddles the threshold; one ending exactly at it; chunks of 1
    token; tiles that start at other positions), run alone, beside decode rows, and beside another prefill row before
    or after it; each within BF16 rounding of the fp32 reference."""
    qo, n, first = 4, 900, 30
    lens = [n, 700, 1000, 420]  # slot 1 the prompt, slot 4 the other prefill row's
    pool, table = _window_setup(lens, HKV, D, PS, seed=21)
    torch.manual_seed(22)
    qs = (torch.randn(n, H, D, device=DEV) * 0.5).to(BF)
    qother = (torch.randn(400, H, D, device=DEV) * 0.5).to(BF)[100:]  # slot 4's positions 100..399
    a = _full(max_rows=8, qo=qo, split_keys=16, pool=pool, prefill_from=64, prefill_keys=16)
    decode_rows = [(s, lens[s - 1] - qo, qo, lens[s - 1]) for s in (2, 3)]

    def prompt(cuts: list[int], beside: str = "") -> torch.Tensor:
        """The prompt's positions [cuts[0], cuts[-1]) by chunks between the cuts, each its own step;
        `beside`: "d" decode rows, "p" another prefill row after the chunk, "q" before it."""
        out = torch.full((n, H, D), float("nan"), device=DEV)
        for lo, hi in itertools.pairwise(cuts):
            rows = [(s, qo * i, qo, kv) for i, (s, _, _, kv) in enumerate(decode_rows)] if "d" in beside else []
            qq = [(torch.randn(qo, H, D, device=DEV) * 0.5).to(BF) for _ in rows]
            decode = len(rows)
            if "q" in beside:
                rows.append((4, qo * decode, 300, 400))
                qq.append(qother)
            at = qo * decode + (300 if "q" in beside else 0)
            rows.append((1, at, hi - lo, hi))
            qq.append(qs[lo:hi])
            if "p" in beside:
                rows.append((4, at + hi - lo, 300, 400))
                qq.append(qother)
            got, _ = _step(a, table, rows, torch.cat(qq), decode=decode)
            out[lo:hi] = got[at:at + hi - lo]
        return out

    base = prompt([first, n])
    want = _window_ref(pool, table, 1, _scaled(qs[first:n]), first, 1 << 30, PS)
    err = (base[first:].float() - want).abs()
    assert err.max().item() < 2e-2 and err.mean().item() < 2e-3, (err.max().item(), err.mean().item())
    cases = [([first, 90, 200, 201, 520, n], ""), ([first, 60, 63, 64, 65, 300, 512, n], ""),
             ([first, 47, 333, n], ""), ([first, 63, 64, 65, 66, 67, n], ""), ([first, 300, n], ""),
             ([first, 100, 101, 600, n], "d"), ([first, 90, 200, 201, 520, n], "p"), ([first, 47, 333, n], "q"),
             ([first, 64, 333, n], "dp"), ([first, n], "d"), ([first, n], "dq")]
    for cuts, beside in cases:
        got = prompt(cuts, beside)
        assert torch.equal(base[first:], got[first:]), (cuts, beside)


def test_full_attention_prefill_below_the_threshold_is_unsplit() -> None:
    """A prefill query holding at most `prefill_from` keys gives the bits of an instance whose threshold lies past the
    prompt (no split: the unsplit kernel), at any chunk offset (a tile straddling the threshold keeps its low queries'
    unsplit bits) while the queries past it differ (the split is in play); and the split's launches cut by a small
    prefill partial buffer (`max_prefill_split` at its floor of 32, one group's items) or a large one give the same
    bits for every query."""
    n = 900
    pool, table = _window_setup([n], HKV, D, PS, seed=31)
    torch.manual_seed(32)
    qs = (torch.randn(n, H, D, device=DEV) * 0.5).to(BF)
    plain = _full(max_rows=1, qo=1, split_keys=16, pool=pool, prefill_from=4096, prefill_keys=16)
    small = _full(max_rows=1, qo=1, split_keys=16, pool=pool, prefill_from=64, prefill_keys=16)
    large = _full(max_rows=24, qo=4, split_keys=16, pool=pool, prefill_from=64, prefill_keys=16)
    assert small.max_prefill_split == 32 and large.max_prefill_split > 32

    for p0 in (0, 3, 30, 59):
        rows, q = [(1, 0, n - p0, n)], qs[p0:]
        want, _ = _step(plain, table, rows, q, decode=0)
        got, meta = _step(small, table, rows, q, decode=0)
        big, big_meta = _step(large, table, rows, q, decode=0)
        low = 64 - p0  # the queries at positions < 64: p + 1 <= 64
        assert torch.equal(want[:low], got[:low]), p0
        assert not torch.equal(want[low:], got[low:]), p0
        assert len(meta.attn_plan[7]) > len(big_meta.attn_plan[7]) >= 1, p0
        assert torch.equal(got, big), p0
        ref = _window_ref(pool, table, 1, _scaled(q[:low + 40]), p0, 1 << 30, PS)
        assert (got[:low + 40].float() - ref).abs().max().item() < 2e-2, p0


@pytest.mark.parametrize(("n", "k"), [(96, 5120), (8, 128)])  # Qwen3.8 27B's in_proj_ba, a tiny model's
def test_linear_rows_matches_reference_and_is_batch_invariant(n, k) -> None:
    """Every row's bits are the same at any M (one row alone, beside 2, 7, 8, 9, 99 and 2,047 others), and the fp32
    product within BF16's rounding."""
    torch.manual_seed(1)
    w = (torch.randn(n, k, device=DEV) * 0.03).to(BF)
    x = torch.randn(2048, k, device=DEV).to(BF)
    full = ops.linear_rows(x, w)
    ref = (x.float() @ w.float().T)
    rel = ((full.float() - ref).abs() / ref.abs().clamp(min=1e-2)).max().item()
    assert rel < 1e-2, rel
    for m in (1, 3, 8, 9, 10, 100, 2048):
        for lo in (0, 2048 - m):
            assert torch.equal(ops.linear_rows(x[lo:lo + m].contiguous(), w), full[lo:lo + m]), (m, lo)


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return bool(torch.equal(a.view(torch.int8), b.view(torch.int8)))


def test_gdn_prefill_on_the_state_pool_is_the_packed_call() -> None:
    """The prefill reads and writes each sequence's slot in a layer's view of a state pool laid out as the runner's
    ([slots, 48 GDN layers, 48, 128, 128] fp32, 9.7 GB: slot offsets past 2^31 elements), with the bits of the call on
    states gathered into a packed tensor and scattered back: the outputs, the checkpoints and the whole pool after it,
    so no other slot or layer is written. Packed rows of lengths about the 64-token chunk, one sequence alone, cut rows
    with checkpoints and a 1-token tail, at the first and last layer and shuffled slots up to the last; a zero-length
    sequence leaves its slot as it was."""
    from llm_mini_infer.kernels.flashinfer import flashinfer_gdn_prefill

    torch.manual_seed(11)
    full = torch.randn(64, 48, 48, 128, 128, device=DEV).mul_(0.1)  # with its clone and a compare's mask: 29 GB
    rng = np.random.default_rng(12)
    cases = [([1, 63, 64, 65, 1400], None), ([2048], None), ([1000, 300, 129, 70], {0: [64, 512, 960], 2: [64, 128]}),
             ([64 * 3 + 1, 30], {0: [64, 128, 192]}), ([5, 700], {1: [640]})]
    for t, (lens, ckpts) in enumerate(cases):
        rows = [_gdn_seq(n, 600 + 10 * t + j) for j, n in enumerate(lens)]
        slots = [int(s) for s in rng.choice(64, size=len(rows), replace=False)]
        if t == 0:
            slots[-1] = 63
        for layer in (0, 47):
            cat, meta, _, _ = _gdn_step(rows, ckpts)
            ids = torch.as_tensor(slots, dtype=torch.int64, device=DEV)
            ref = full.clone()
            want = _packed_reference(*(cat[x] for x in ("q", "k", "v", "a", "b")), ref[:, layer], ids, meta)
            want_ck = meta.ckpt.workspace.clone() if meta.ckpt is not None else None
            if meta.ckpt is not None:
                meta.ckpt.workspace.fill_(float("nan"))
            got = flashinfer_gdn_prefill(cat["q"], cat["k"], cat["v"], cat["a"], cat["b"], full[:, layer], ids, meta)
            torch.cuda.synchronize()
            assert _same(got, want), (t, layer, "outputs")
            if want_ck is not None:
                assert _same(meta.ckpt.workspace.nan_to_num(), want_ck.nan_to_num()), (t, layer, "checkpoints")
            assert _same(full, ref), (t, layer, "the pool")
            del ref
    rows = [_gdn_seq(n, 700 + n) for n in (130, 70)]  # a zero-length sequence between them
    cat, meta, _, _ = _gdn_step(rows)
    meta.pf_cu_host = np.asarray([0, 130, 130, 200], np.int64)
    meta.pf_cu = torch.as_tensor(meta.pf_cu_host, device=DEV)
    full[3, 5], full[9, 5] = rows[0]["s"], rows[1]["s"]
    before = full[:, 5].clone()
    flashinfer_gdn_prefill(cat["q"], cat["k"], cat["v"], cat["a"], cat["b"], full[:, 5],
                           torch.as_tensor([3, 40, 9], dtype=torch.int64, device=DEV), meta)
    torch.cuda.synchronize()
    assert _same(full[40, 5], before[40]), "the zero-length sequence's slot"
    alone = [_gdn_call([r])[0] for r in rows]
    assert _same(full[3, 5], alone[0][1]) and _same(full[9, 5], alone[1][1])


def test_gdn_prefill_is_batch_invariant() -> None:
    """FlashInfer's GDN prefill on its per-sequence kernel (`use_cp` off, its checkpoints taken by cutting rows):
    a sequence's outputs and final state have the same bits alone, packed between others of other lengths, cut into
    calls at multiples of 64 from its start (the fp32 state handed on), with a 1-token tail, and with checkpoints in
    its step, whose states are those of a call stopped there; and a repeat gives the same bits (its checkpointing
    variant's do not). FlashInfer's: an upgrade that breaks its per-sequence bits fails here."""
    rng = np.random.default_rng(5)
    for t in range(12):
        n = int(rng.integers(65, 1400))
        s = _gdn_seq(n, 100 + t)
        others = [_gdn_seq(int(rng.integers(1, 600)), 1000 + 10 * t + j) for j in range(int(rng.integers(1, 6)))]
        alone = _gdn_call([s])[0]
        at = int(rng.integers(0, len(others) + 1))
        packed = _gdn_call([*others[:at], s, *others[at:]])[at]
        assert _same(alone[0], packed[0]) and _same(alone[1], packed[1]), (t, n, "packed")
        grid = list(range(64, n, 64))
        cuts = sorted({int(c) for c in rng.choice(grid, size=min(len(grid), 3), replace=False)})
        ck = _gdn_call([s, *others], {0: cuts})[0]
        assert _same(alone[0], ck[0]) and _same(alone[1], ck[1]), (t, n, "checkpoints", cuts)
        again = _gdn_call([s, *others], {0: cuts})[0]
        assert all(_same(ck[2][c], again[2][c]) for c in cuts) and _same(ck[0], again[0]), (t, "repeat")
        state, outs, prev = s["s"], [], 0
        for c in [*cuts, n]:  # the cut calls: each beside others, from the state the last left
            part = {x: s[x][prev:c] for x in ("q", "k", "v", "a", "b")}
            part["s"] = state
            r = _gdn_call([others[0], part])[1]
            outs.append(r[0])
            if c < n:
                assert _same(r[1], ck[2][c]), (t, c, "checkpoint = a call stopped there")
            state, prev = r[1], c
        assert _same(torch.cat(outs), alone[0]) and _same(state, alone[1]), (t, n, "cut", cuts)
    m = 64 * 3 + 1  # a 1-token tail
    s = _gdn_seq(m, 7)
    whole = _gdn_call([s])[0]
    head = _gdn_call([{**{x: s[x][:m - 1] for x in ("q", "k", "v", "a", "b")}, "s": s["s"]}])[0]
    tail = _gdn_call([_gdn_seq(30, 8), {**{x: s[x][m - 1:] for x in ("q", "k", "v", "a", "b")}, "s": head[1]}])[1]
    assert _same(torch.cat([head[0], tail[0]]), whole[0]) and _same(tail[1], whole[1])


def test_gdn_prefill_takes_the_per_sequence_route(monkeypatch) -> None:
    """The prefill runs the per-sequence route at every step, on the state pool, and takes a step's mid-chunk
    checkpoints by cutting rows, never FlashInfer's checkpointing variant (which in FlashInfer 0.7.2 moves between
    identical calls on SM120, fixed in 0.7.3 by flashinfer-ai/flashinfer#7785; every FlashInfer call is watched):
    repeated identical calls of packed rows of other lengths, several with checkpoints, give the same bits every time;
    each checkpoint is the state of a call stopped there; and one sequence alone, which FlashInfer's own choice would
    send to its context-parallel route, gives its bits from the packed, cut call."""
    import functools
    import inspect

    import flashinfer.gdn_prefill as fg

    # an upgrade that drops the route's pin fails here by name, not with a TypeError at the first prefill
    assert {"use_cp", "state_indices"} <= set(inspect.signature(fg.chunk_gated_delta_rule).parameters)
    calls: list[dict] = []
    orig = fg.chunk_gated_delta_rule

    @functools.wraps(orig)
    def spy(*a, **kw):
        calls.append(kw)
        return orig(*a, **kw)

    monkeypatch.setattr(fg, "chunk_gated_delta_rule", spy)
    rows = [_gdn_seq(int(n), 300 + i) for i, n in enumerate((1000, 300, 2048, 70, 700, 129))]
    ckpts = {0: [64, 512, 960], 2: [1024, 1984], 4: [128], 5: [64, 128]}
    first = _gdn_call(rows, ckpts)
    for rep in range(30):
        for i, (o, s, cks) in enumerate(_gdn_call(rows, ckpts)):
            want = first[i]
            assert _same(o, want[0]) and _same(s, want[1]), (rep, i)
            assert all(_same(cks[c], want[2][c]) for c in ckpts.get(i, [])), (rep, i, "checkpoints")
    for i, offs in ckpts.items():
        for c in offs:
            head = {**{x: rows[i][x][:c] for x in ("q", "k", "v", "a", "b")}, "s": rows[i]["s"]}
            assert _same(first[i][2][c], _gdn_call([head])[0][1]), (i, c, "a call stopped there")
    alone = _gdn_call([rows[2]])[0]
    assert _same(alone[0], first[2][0]) and _same(alone[1], first[2][1]), "alone"
    assert calls and all(kw.get("use_cp") is False and "state_checkpoints" not in kw for kw in calls)
    assert all(kw["state_indices"] is not None and kw["initial_state"] is kw["output_state"] for kw in calls)


def test_gdn_prefill_matches_the_recurrence() -> None:
    """Its outputs and final state against the fp32 recurrence (`models.qwen3_5.backend.torch_gdn_prefill`), token by
    token: within BF16's rounding."""
    from llm_mini_infer.models.qwen3_5.backend import torch_gdn_prefill

    rows = [_gdn_seq(n, 50 + n) for n in (70, 200)]
    got = _gdn_call(rows, {1: [64, 128]})
    cu = np.asarray([0, 70, 270], np.int64)
    meta = SimpleNamespace(pf_cu_host=cu, ckpt=None)
    cat = {x: torch.cat([r[x] for r in rows]) for x in ("q", "k", "v", "a", "b")}
    final = torch.stack([r["s"] for r in rows])
    o = torch_gdn_prefill(cat["q"], cat["k"], cat["v"], cat["a"], cat["b"], final,
                          torch.arange(2, device=DEV), meta)
    for i in range(2):
        want_o, want_s = o[cu[i]:cu[i + 1]].float(), final[i]
        assert ((got[i][0].float() - want_o).abs().max() / want_o.abs().max()).item() < 2e-2, i
        assert ((got[i][1] - want_s).abs().max() / want_s.abs().max()).item() < 2e-2, i
