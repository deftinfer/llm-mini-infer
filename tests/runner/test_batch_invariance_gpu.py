"""Answer reproducibility on CUDA (architecture.md, "Batch invariance"): with the batch-invariant parts on (the full
layers' fixed split, the sliding layers on our kernel, the sparse head at one M, the tower one image a call) and the
FP8 projections on CUTLASS at every M, a request's final hidden state and logits at every position, and its images'
encoder rows, have the same bits alone and beside other requests that arrive while it runs. Its answer is forced by its
grammar (`Forced` of tests/support/forced.py), so both runs' inputs agree; an FP8 target's assistant drafts freely at
k = 3 (its verify windows sit elsewhere in each run); CUDA graphs; every step split into its forward and sampling
graphs. The logits are the sampling graph's own (`forced.Logits`), compared per token id: the two runs' steps have
different mask unions, so a token's column in the sparse head moves between them.

- **Text:** a tiny FP8 target and its assistant.
- **An image:** a tiny multimodal FP8 model (its tower BF16, as a real FP8 checkpoint's) whose compared request
  carries an image, encoded alone in one run and in one step with another request's image in the other (a tower call
  each, the tower's attention on FlashInfer's ragged kernel); its encoder rows are compared as the ENCODE wrote them,
  and its span, which takes the sliding kernel's bidirectional pass, through the positions after it.
- **The forced prefix** (an answer's first tokens forced): taken whole by the final chunk, alone and beside.
- **The dense head** (the sparse head off): its GEMM runs at each step's M, which batch invariance leaves as it is; an
  expected failure that records whether its logits' bits move.
- **The golden:** each alone run's digest, recorded per GPU (its architecture and SM count) and tiny weights in
  tests/data/golden/batch_invariance.json (`tests/support/golden.py`): a new torch, CUDA, cuBLAS, FlashInfer or kernel
  build shows whether it moved the bits. `--refresh-golden` records this GPU's entries.

CUDA, the kernel library and the CUTLASS library only."""

from __future__ import annotations

import numpy as np
import pytest

from ..support import golden
from ..support.forced import Bits, ForcedRun
from ..support.gemma_gpu import W, to_bf16
from ..support.gemma_gpu import fp8_pair as fp8_pair  # the fixture: an FP8 target and its BF16 assistant
from ..support.gemma_images import png
from ..support.invariance import LogitsDiffer, logits_compared
from ..support.tiny_gemma import make_tiny_assistant, make_tiny_gemma_mm_fp8

pytestmark = [pytest.mark.gpu,
              pytest.mark.cutlass(reason="needs a CUDA GPU, the kernel library and the CUTLASS library")]

VOCAB = 512  # the tiny models' (tests/support/tiny_gemma.py)
INVARIANT = dict(sparse_head=True, draft_universe=True, pdl=True, always_split=True, attn512_split_keys=32,
                 window_split_keys=16, head_fixed_m=True, encode_per_image=True, batch_invariant=True)
GOLDEN = golden.GOLDEN_DIR / "batch_invariance.json"


def tokens(n: int, seed: int, top: int = VOCAB - 3) -> list[int]:
    return np.random.default_rng(seed).integers(3, top, n).tolist()


def text(n: int, seed: int) -> list[int]:
    return tokens(n, seed, 490)  # below the multimodal model's image tokens (500-503)


CASES = {  # each request's prompt (its parts) and forced answer
    "text": ({0: [tokens(70, 1)], 1: [tokens(120, 2)], 2: [tokens(35, 3)], 3: [tokens(90, 4)]},
             {r: tokens(30 + 7 * r, 10 + r) for r in range(4)}),
    "image": ({0: [text(20, 21), png(48, 36, 1), text(12, 22)], 1: [text(9, 23), png(40, 40, 2), text(5, 24)],
               2: [text(60, 25)], 3: [text(14, 26), png(30, 60, 3), text(8, 27)]},
              {r: text(30 + 7 * r, 30 + r) for r in range(4)}),
}


@pytest.fixture(scope="module")
def mm_pair(tmp_path_factory):
    d = tmp_path_factory.mktemp("binvmm")
    target = make_tiny_gemma_mm_fp8(d / "target", window=W, seed=6, gpu=True)
    assistant = make_tiny_assistant(d / "assistant", target, hidden=256, intermediate=512, seed=7)
    to_bf16(assistant, ("q_norm.weight",))
    return target, assistant


def run(pair, case: str, beside: bool, arena: str, forced: int = 0, **switches) -> Bits:
    """Request 0's forced answer, alone or beside requests 1-3; its captures where its inputs are the reference's.
    `forced`: the forced prefix's tokens."""
    prompts, answers = CASES[case]
    f = ForcedRun(pair, prompts, answers, device="cuda", arena=arena if case == "image" else None, window=W, wps=16,
                  max_num_seqs=8, budget=256, num_pages=2048, forced=forced, forced_prefix=forced > 0,
                  **{**INVARIANT, **switches})
    if not beside:
        f.add(0)
    elif case == "text":  # 1 and 2 decode while 0's prompt prefills; 3's prompt prefills beside 0's answer
        f.add(1)
        f.add(2)
        f.steps(3)
        f.add(0)
        f.steps(4)
        f.add(3)
    else:  # 0's image and 1's in one ENCODE (a tower call each) and one prefill beside 2's decode; then 3's image
        f.add(2)
        f.steps(3)
        f.add(0)
        f.add(1)
        f.steps(4)
        f.add(3)
    return f.bits(0)


@pytest.fixture(scope="module")
def runs(fp8_pair, mm_pair, tmp_path_factory):
    """`get(case, beside, **switches)`: that run's captures, run once for the module's tests."""
    done: dict[tuple, Bits] = {}

    def get(case: str, beside: bool, **switches) -> Bits:
        k = (case, beside, tuple(sorted(switches.items())))
        if k not in done:
            arena = str(tmp_path_factory.mktemp("arena") / "pixels")
            done[k] = run(fp8_pair if case == "text" else mm_pair, case, beside, arena, **switches)
        return done[k]

    return get


def same_states(alone: Bits, beside: Bits, case: str, forced: int = 0) -> list[int]:
    """The positions both runs kept, asserted to hold the same hidden states' bits."""
    common = sorted(set(alone.hidden) & set(beside.hidden))
    # the prompt's last position and every answer position but the EOS's (and the forced prefix's, never sampled)
    assert len(common) >= len(CASES[case][1][0]) - forced
    differ = [p for p in common if not np.array_equal(alone.hidden[p], beside.hidden[p])]
    assert not differ, f"{len(differ)} of {len(common)} positions' states differ, the first at {min(differ)}"
    return common


@pytest.mark.parametrize("case", ["text", "image"])
def test_a_requests_bits_do_not_depend_on_the_requests_beside_it(runs, case) -> None:
    alone, beside = runs(case, False), runs(case, True)
    if case == "image":  # 0's image alone, then in one ENCODE with 1's (each its own tower call): its rows first
        assert alone.encodes == [1] and 2 in beside.encodes and set(beside.calls) == {1}, beside.encodes
        assert len(alone.images) == 1 and set(alone.images) == set(beside.images)
        differ = [p for p in alone.images if not np.array_equal(alone.images[p], beside.images[p])]
        assert not differ, f"the encoder rows of the image at {differ} differ"
    common = same_states(alone, beside, case)
    n, moved, differ = logits_compared(alone, beside, common)
    assert n >= len(common)  # at least each position's forced token
    assert not differ, f"{len(differ)} of {n} logits differ, the first at (position, token) {differ[0]}"
    assert alone.heads.sparse > 0 and beside.heads.sparse > 0 and moved > 0  # columns that moved kept their bits


def test_a_requests_bits_do_not_depend_on_the_draft_counts(runs) -> None:
    """A draft count per step (`dev.speculative.adaptive`; forced schedules, `count_schedule`): request 0 alone at a
    fixed k, alone under a schedule, and beside requests 1-3 under others (verify rows 2, 3 and 4 wide; lowering, and
    raising through a step that verifies what its rows hold and drafts k, then 4 wide again after it) has the same
    hidden states' and logits' bits every time: a row's bits depend neither on the rows beside it nor on its window's
    width."""
    alone = runs("text", False)
    for beside, schedule in ((False, (1, 3, 2)), (True, (1, 3, 2)), (True, (2, 1, 1, 3)), (True, (3, 3, 1, 3, 2))):
        other = runs("text", beside, draft_counts=(1, 2), count_schedule=schedule)
        common = same_states(alone, other, "text")
        n, _, differ = logits_compared(alone, other, common)
        assert n >= len(common) and not differ, \
            f"{schedule}: {len(differ)} of {n} logits differ, the first at {differ[:1]}"
        assert other.heads.sparse > 0


def test_the_forced_prefix_is_taken_whole(runs) -> None:
    """With the forced prefix (an answer's first 4 tokens forced), the final chunk takes them whole, alone and beside:
    their positions and the first sample after them run in the prefill row either way."""
    alone, beside = runs("text", False, forced=4), runs("text", True, forced=4)
    assert min(alone.hidden) == len(CASES["text"][0][0][0]) + 4 - 1  # the prompt's; the forced positions not sampled
    common = same_states(alone, beside, "text", forced=4)
    n, _, differ = logits_compared(alone, beside, common)
    assert n >= len(common) and not differ, f"{len(differ)} of {n} logits differ, the first at {differ[:1]}"


@pytest.mark.xfail(raises=LogitsDiffer, strict=False,
                   reason="the dense head's GEMM runs at each step's M (cuBLAS picks its kernel by M): batch "
                          "invariance fixes the sparse head's M only, so the dense head's logits may move beside "
                          "other requests")
def test_the_dense_heads_logits_beside_other_requests(runs) -> None:
    """The sparse head off, every step on the dense head: the states keep their bits (asserted), the logits may not;
    an XPASS records that they kept them on this GPU."""
    alone, beside = runs("text", False, sparse_head=False), runs("text", True, sparse_head=False)
    assert alone.heads.sparse == beside.heads.sparse == 0
    common = same_states(alone, beside, "text")
    n, _, differ = logits_compared(alone, beside, common)
    assert n == VOCAB * len(common)  # every token's
    if differ:
        raise LogitsDiffer(f"{len(differ)} of {n} logits differ, the first at (position, token) {differ[0]}")


@pytest.mark.parametrize("case", ["text", "image"])
def test_the_alone_runs_bits_are_the_golden(runs, case, fp8_pair, mm_pair, refresh_golden) -> None:
    alone = runs(case, False)
    entry = golden.key(case, golden.gpu(), golden.weights_digest(fp8_pair if case == "text" else mm_pair))
    golden.check(GOLDEN, entry, golden.bits_digest(alone.hidden, alone.logits, alone.images), refresh=refresh_golden,
                 stack=golden.versions(), test="tests/runner/test_batch_invariance_gpu.py")
