"""Answer reproducibility on the Qwen runner (architecture.md, "Batch invariance"), as
tests/runner/test_batch_invariance_gpu.py checks it on Gemma 4's: with `batch_invariant` (the GDN prefill on
FlashInfer's per-sequence kernel, its checkpoints taken by cutting rows; full attention on our kernel, decode rows split
at absolute multiples, prefill rows whole; in_proj_ba at any M; the sparse head at one M; the tower one image a call; a
prompt's cuts on the 64-token grid) and the FP8 projections on CUTLASS at every M, a request's final hidden state and
logits at every position it samples at, and its image's encoder rows, have the same bits alone and beside other
requests that arrive while it runs, and whatever its chunking. Its answer is forced by its grammar (`QwenForcedRun`
of tests/support/forced.py), so the runs' inputs agree; the MTP layer drafts freely at k = 3 (its verify windows sit
elsewhere in each run); CUDA graphs; every step split into its forward and sampling graphs. The tiny models have
Qwen3.8 27B's attention shape (head dim 256, a GQA group of 6: `tiny_qwen.QWEN38_ATTN`), so our attention kernel
serves them, and their in_proj_ba is BF16 as the real checkpoints'.

- **Text and an image,** alone against beside (decode rows of others take the prompt's budget, so its cuts move).
- **Long contexts:** the full layers' split sizes double with the position (16 keys to 512, then 32, then 64) and a
  prefill row's queries past 64 keys split by 16 (`attn_prefill`): a request whose prompt passes 64 and whose answer
  passes 512 and 1024 positions, alone and beside requests that arrive while it runs.
- **The chunking:** the same request alone under chunk caps (64, 100: off the grid; 128 with a 1-token final chunk),
  with and without the prefix cache.
- **A prefix hit on a checkpoint its prefill wrote:** a request that resumes at 64 from an earlier request's
  checkpoint against the same request computed from position 0.
- **The forced prefix** (an answer's first tokens forced): taken whole by the final chunk, alone and beside.
- **The dense head** (the sparse head off): its GEMM runs at each step's M; an expected failure that records whether
  its logits' bits move.
- **The golden:** each alone run's digest, per GPU and tiny weights, in tests/data/golden/batch_invariance.json
  (`tests/support/golden.py`; `--refresh-golden` records this GPU's entries).

CUDA, the kernel library, the CUTLASS library and FlashInfer only."""

from __future__ import annotations

import numpy as np
import pytest

from ..support import golden
from ..support.forced import Bits, QwenForcedRun
from ..support.gemma_images import png
from ..support.invariance import LogitsDiffer, logits_compared
from ..support.tiny_qwen import QWEN38_ATTN, make_tiny_fp8_model, make_tiny_vl_fp8_model

pytestmark = [pytest.mark.gpu,
              pytest.mark.cutlass(reason="needs a CUDA GPU, the kernel library and the CUTLASS library")]

VOCAB = 512  # the tiny models' (tests/support/tiny_qwen.py); image tokens 496-499
INVARIANT = dict(sparse_head=True, draft_universe=True, pdl=True, always_split=True, batch_invariant=True,
                 attn_split_keys=64)  # splits of 64 keys: the tiny contexts span several
LONG = dict(max_model_len=1280, attn_split_keys=16, attn_prefill=(64, 16))  # sizes 16, 32, 64; prefill splits past 64
GOLDEN = golden.GOLDEN_DIR / "batch_invariance.json"


def tokens(n: int, seed: int) -> list[int]:
    return np.random.default_rng(seed).integers(3, 490, n).tolist()  # below the image tokens


SHARED = tokens(72, 99)  # a shared prefix: an earlier request's prefill leaves a checkpoint at 64 in it
CASES = {  # each request's prompt (its parts) and forced answer
    "text": ({0: [tokens(300, 1)], 1: [tokens(120, 2)], 2: [tokens(35, 3)], 3: [tokens(90, 4)]},
             {r: tokens(30 + 7 * r, 10 + r) for r in range(4)}),
    "long": ({0: [tokens(600, 41)], 1: [tokens(120, 42)], 2: [tokens(200, 43)], 3: [tokens(90, 44)]},
             {0: tokens(520, 45), 1: tokens(30, 46), 2: tokens(40, 47), 3: tokens(50, 48)}),
    "image": ({0: [tokens(20, 21), png(48, 36, 1), tokens(12, 22)], 1: [tokens(9, 23), png(40, 40, 2), tokens(5, 24)],
               2: [tokens(60, 25)], 3: [tokens(14, 26), png(30, 60, 3), tokens(8, 27)]},
              {r: tokens(30 + 7 * r, 30 + r) for r in range(4)}),
    "tail": ({0: [tokens(129, 5)], 1: [tokens(100, 6)], 2: [tokens(40, 7)], 3: [tokens(70, 8)]},
             {r: tokens(20 + 3 * r, 50 + r) for r in range(4)}),
    "hit": ({0: [SHARED + tokens(40, 9)], 1: [tokens(30, 10)], 2: [tokens(50, 11)], 3: [SHARED + tokens(9, 12)]},
            {r: tokens(25, 60 + r) for r in range(4)}),
}


@pytest.fixture(scope="module")
def text_model(tmp_path_factory):
    return make_tiny_fp8_model(tmp_path_factory.mktemp("qbinv"), seed=5, mtp=True, gpu=True, attn=QWEN38_ATTN)[0]


@pytest.fixture(scope="module")
def vl_model(tmp_path_factory):
    return make_tiny_vl_fp8_model(tmp_path_factory.mktemp("qbinvmm"), seed=6, mtp=True)


def run(model, case: str, arm: str, arena: str | None, forced: int = 0, cap: int = 0, caching: bool = True,
        **switches) -> Bits:
    """Request 0's forced answer (or, in the hit case, request 3's): `arm` alone or beside the others; `cap`: the
    scheduler's chunk cap (long_prefill_token_threshold); `forced`: the forced prefix's tokens; `caching`: the prefix
    cache (without it no checkpoint target rounds a cut to the grid: only batch invariance does)."""
    prompts, answers = CASES[case]
    f = QwenForcedRun(model, prompts, answers, device="cuda", arena=arena if case == "image" else None, forced=forced,
                      max_num_seqs=8, budget=256, num_pages=2048, page_size=16, caching=caching, ckpt_slots=8,
                      forced_prefix=forced > 0, **{**INVARIANT, **switches})
    if "attn_prefill" in switches:  # the settings reached the full layers' kernel (a run without them is invariant)
        a = f.run.r.attn
        assert (a.split_keys, (a.prefill_from, a.prefill_keys)) == (switches["attn_split_keys"],
                                                                     switches["attn_prefill"])
    f.run.sched.chunk_cap = cap
    subject = 3 if case == "hit" else 0
    if case == "hit":  # request 0 writes the shared prefix's checkpoint at 64 first; then 3 resumes there (or not)
        if arm == "hit":
            f.add(0)
            f.steps(40)
        f.add(3)
    elif arm == "alone":
        f.add(0)
    elif case == "image":  # 0's image and 1's in one ENCODE (a tower call each) and one prefill beside 2's decode
        f.add(2)
        f.steps(3)
        f.add(0)
        f.add(1)
        f.steps(4)
        f.add(3)
    elif case == "long":  # 1 decodes while 0's prompt prefills in chunks; 3's prompt prefills beside 0's answer, and
        # 2's (200 tokens) beside it again once it is well into its answer
        f.add(1)
        f.steps(3)
        f.add(0)
        f.steps(4)
        f.add(3)
        f.steps(40)
        f.add(2)
    else:  # 1 and 2 decode while 0's prompt prefills; 3's prompt prefills beside 0's answer
        f.add(1)
        f.add(2)
        f.steps(3)
        f.add(0)
        f.steps(4)
        f.add(3)
    bits = f.bits(subject)
    if case == "hit":
        assert f.run.cached[3] == (64 if arm == "hit" else 0), f.run.cached
    return bits


@pytest.fixture(scope="module")
def runs(text_model, vl_model, tmp_path_factory):
    """`get(case, arm, **kw)`: that run's captures, run once for the module's tests."""
    done: dict[tuple, Bits] = {}

    def get(case: str, arm: str, **kw) -> Bits:
        k = (case, arm, tuple(sorted(kw.items())))
        if k not in done:
            arena = str(tmp_path_factory.mktemp("arena") / "pixels")
            done[k] = run(vl_model if case == "image" else text_model, case, arm, arena, **kw)
        return done[k]

    return get


def same_bits(a: Bits, b: Bits, min_positions: int) -> None:
    """The positions both runs kept hold the same hidden states' and logits' bits (per token id: the runs' mask
    unions differ, so a token's column may move), and their images the same encoder rows."""
    common = sorted(set(a.hidden) & set(b.hidden))
    assert len(common) >= min_positions, (len(common), min_positions)
    differ = [p for p in common if not np.array_equal(a.hidden[p], b.hidden[p])]
    assert not differ, f"{len(differ)} of {len(common)} positions' states differ, the first at {min(differ)}"
    n, _, ldiff = logits_compared(a, b, common)
    assert n >= len(common) and not ldiff, f"{len(ldiff)} of {n} logits differ, the first at {ldiff[:1]}"
    assert set(a.images) == set(b.images)
    bad = [p for p in a.images if not np.array_equal(a.images[p], b.images[p])]
    assert not bad, f"the encoder rows of the image at {bad} differ"


@pytest.mark.parametrize("case", ["text", "image"])
def test_a_requests_bits_do_not_depend_on_the_requests_beside_it(runs, case) -> None:
    alone, beside = runs(case, "alone"), runs(case, "beside")
    if case == "image":  # 0's image alone, then in one ENCODE with 1's (each its own tower call)
        assert alone.encodes == [1] and 2 in beside.encodes and set(beside.calls) == {1}
        assert len(alone.images) == 1
    same_bits(alone, beside, len(CASES[case][1][0]))
    assert alone.heads.sparse > 0 and beside.heads.sparse > 0


def test_a_requests_bits_do_not_depend_on_the_requests_beside_it_past_the_split_sizes(runs) -> None:
    """The full layers' split sizes double with a query's position (16 keys, 32 past 512, 64 past 1024) and a prefill
    row splits the queries past 64 keys by 16: request 0's 600-token prompt (cut on the 64-token grid, its queries past
    64 split) and 520-token answer (its decode queries past 512 and 1024) have the same bits alone and beside requests
    that arrive while it runs, a prefill row of 200 tokens among them."""
    alone, beside = runs("long", "alone", **LONG), runs("long", "beside", **LONG)
    assert max(alone.hidden) > 1024 and min(alone.hidden) < 600  # the answer passes both size boundaries
    same_bits(alone, beside, len(CASES["long"][1][0]))
    assert alone.heads.sparse > 0 and beside.heads.sparse > 0


@pytest.mark.parametrize(("case", "cap", "caching"), [("text", 64, True), ("text", 100, False), ("tail", 128, True),
                                                      ("tail", 64, True), ("text", 0, False)])
def test_a_requests_bits_do_not_depend_on_its_chunking(runs, case, cap, caching) -> None:
    """The same request alone, its prompt cut by the step's budget (256) or by a chunk cap, which the grid rounds down
    (100 to 64), with the prefix cache or without it (no checkpoint target then rounds a cut: the switch alone does);
    the 129-token prompt's last chunk under a cap of 128 is one token, which takes the GDN prefill kernel as any."""
    same_bits(runs(case, "alone"), runs(case, "alone", cap=cap, caching=caching), len(CASES[case][1][0]))


def test_a_hit_on_a_checkpoint_its_prefill_wrote_gives_the_cold_bits(runs) -> None:
    """Request 3 shares 72 tokens with request 0, which ran before it: it resumes at 64 (0's prefill checkpoint, on the
    grid), and its bits are those it gets computing its prompt from position 0."""
    same_bits(runs("hit", "cold"), runs("hit", "hit"), len(CASES["hit"][1][3]))


def test_the_forced_prefix_is_taken_whole(runs) -> None:
    """With the forced prefix (an answer's first 4 tokens forced), the final chunk takes them whole, alone and beside:
    their positions and the first sample after them run in the prefill row either way."""
    alone, beside = runs("text", "alone", forced=4), runs("text", "beside", forced=4)
    n = len(CASES["text"][0][0][0])  # the prompt; its forced positions are not sampled, the one after them is
    assert min(alone.hidden) == n + 4 - 1
    same_bits(alone, beside, len(CASES["text"][1][0]) - 4)


@pytest.mark.xfail(raises=LogitsDiffer, strict=False,
                   reason="the dense head's GEMM runs at each step's M (cuBLAS picks its kernel by M): batch "
                          "invariance fixes the sparse head's M only, so the dense head's logits may move beside "
                          "other requests")
def test_the_dense_heads_logits_beside_other_requests(runs) -> None:
    """The sparse head off, every step on the dense head: the states keep their bits (asserted), the logits may not;
    an XPASS records that they kept them on this GPU."""
    alone, beside = runs("text", "alone", sparse_head=False), runs("text", "beside", sparse_head=False)
    assert alone.heads.sparse == beside.heads.sparse == 0
    common = sorted(set(alone.hidden) & set(beside.hidden))
    assert all(np.array_equal(alone.hidden[p], beside.hidden[p]) for p in common)
    n, _, differ = logits_compared(alone, beside, common)
    assert n == VOCAB * len(common)
    if differ:
        raise LogitsDiffer(f"{len(differ)} of {n} logits differ, the first at (position, token) {differ[0]}")


@pytest.mark.parametrize("case", ["text", "image"])
def test_the_alone_runs_bits_are_the_golden(runs, case, text_model, vl_model, refresh_golden) -> None:
    alone = runs(case, "alone")
    entry = golden.key(f"qwen-{case}", golden.gpu(), golden.weights_digest((text_model,) if case == "text" else
                                                                             (vl_model,)))
    golden.check(GOLDEN, entry, golden.bits_digest(alone.hidden, alone.logits, alone.images), refresh=refresh_golden,
                 stack=golden.versions(), test="tests/runner/test_qwen_batch_invariance_gpu.py")
