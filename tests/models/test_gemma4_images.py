"""Gemma 4 with images on CPU: GemmaRunner (the torch reference backends, fp32) with the shared image path
(`runner.cuda.images`), the real scheduler with bidirectional spans and the prefix cache, against transformers'
`Gemma4ForConditionalGeneration` on a tiny random configuration (window 8: image spans are longer than the window, so
the clamp of a span to the window is exercised).

The reference is HF's own forward with `mm_token_type_ids` (its sliding layers' `(causal OR blockwise) AND
sliding_window`, its full layers causal) over the whole prompt at once; every position's logits and every greedy
continuation must match to fp32 rounding, whatever the chunking, prefix hits (before, at and past a span; a hit that
would land inside a span backs off), two images in one prompt, and evictions in a small arena.
"""

from __future__ import annotations

import pytest
import torch

from ..support.gemma_images import Engine, greedy_mm, hf_mm_logits_causal, png, text
from ..support.tiny_gemma import hf_mm_logits, hf_mm_model, make_tiny_gemma_mm

pytest.importorskip("transformers")

TOL = 1e-4  # fp32 on both sides (x86's BLAS: up to 5.03e-5 on a CI runner); logits of order 1-10
W = 8


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return make_tiny_gemma_mm(tmp_path_factory.mktemp("g4img"), window=W, depth=2)


def test_one_image_prompt_matches_transformers_at_every_position(tiny, tmp_path) -> None:
    """One prompt (text, an image of 63 soft tokens, text) prefilled in one step: the logits of every prompt position
    equal HF's bidirectional forward; with the spans causal (HF without `mm_token_type_ids`) they would not."""
    eng = Engine(tiny, tmp_path, budget=128, caching=False)
    ids, refs, pils = eng.prompt([text(9, 1), png(48, 36, 1), text(7, 2)])
    assert len(refs) == 1 and refs[0].tokens > W
    eng.add(0, ids, refs, 3)
    eng.run()
    ours = torch.stack([eng.logits[(0, p)] for p in range(len(ids))])
    ref = hf_mm_logits(tiny, ids, pils)
    assert (ours - ref).abs().max().item() < TOL
    causal = hf_mm_logits_causal(tiny, ids, pils)
    assert (causal - ref).abs().max().item() > 5e-3  # the test can tell the masks apart


@pytest.mark.parametrize("budget,row_tokens", [(72, 0), (96, 40)])
def test_chunked_prefill_around_spans_matches_transformers(tiny, tmp_path, budget, row_tokens) -> None:
    """Two images in one prompt and budgets that would cut through them (72 tokens; 96 with a mixed-step bound of 40
    next to a request decoding): the chunks stop before each span and take it whole; every position's logits equal
    HF's, and the greedy continuation is HF's."""
    eng = Engine(tiny, tmp_path, budget=budget, caching=False, row_tokens=row_tokens)
    ids, refs, pils = eng.prompt([text(30, 3), png(36, 48, 2), text(5, 4), png(60, 30, 3), text(11, 5)])
    assert len(refs) == 2
    if row_tokens:  # a request decoding meanwhile: steps with decode rows bound the prefill
        o_ids, o_refs, _ = eng.prompt([text(12, 6)])
        eng.add(1, o_ids, o_refs, 30)
    eng.add(0, ids, refs, 6)
    eng.run()
    ours = torch.stack([eng.logits[(0, p)] for p in range(len(ids))])
    assert (ours - hf_mm_logits(tiny, ids, pils)).abs().max().item() < TOL
    assert eng.outs[0] == greedy_mm(hf_mm_model(tiny), ids, pils, 6)


def test_prefix_hits_before_at_and_past_spans_match_transformers(tiny, tmp_path) -> None:
    """Requests one after another over one instruction: the same image with another item, another image, that image
    again with a hint at its start, an earlier prompt again, and the image followed by a single token. Hits land where
    the cache's rules put them (before an image whose keys differ, past an image shared whole; never inside a span),
    and every greedy output is HF's."""
    eng = Engine(tiny, tmp_path, budget=128, caching=True)
    inst = text(40, 7)
    a_img, b_img = png(48, 36, 10), png(30, 45, 11)
    reqs = [([inst, a_img, text(6, 8)], []), ([inst, a_img, text(9, 9)], []), ([inst, b_img, text(6, 8)], []),
            ([inst, b_img, text(3, 12)], [41]), ([inst, a_img, text(9, 9)], []), ([inst, a_img, [5]], [])]
    hf = hf_mm_model(tiny)
    spans = []
    for rid, (parts, hints) in enumerate(reqs):
        ids, refs, pils = eng.prompt(parts)
        eng.add(rid, ids, refs, 4, hints)
        eng.run()
        assert eng.outs[rid] == greedy_mm(hf, ids, pils, 4), rid
        sp = [(r.offset, r.offset + r.tokens) for r in refs]
        assert not any(s < eng.cached[rid] < e for s, e in sp), (rid, eng.cached[rid], sp)
        spans.append(sp[0])
    # 1 and 2 miss: nothing was checkpointed below where they leave the tree (past image A; at image B's start)
    assert eng.cached[0] == eng.cached[1] == eng.cached[2] == 0
    assert 0 < eng.cached[3] <= spans[3][0]  # image B again, another item: the window request 2 left below B
    assert eng.cached[4] >= spans[4][1]  # request 1 again: past the image
    # an image followed by one token: the end-of-prefill checkpoint of request 5 is the span's end at best
    assert eng.cached[5] <= spans[5][1]


def test_a_small_arena_evicts_and_recomputes_exactly(tiny, tmp_path) -> None:
    """An arena that holds about two requests: windows and cached pages are evicted, requests wait for pages; outputs
    stay HF's."""
    from llm_mini_infer.cache.pools import pages_for_one_request, window_pages_for_one_request

    one = window_pages_for_one_request(W, 128, 4, 2) + pages_for_one_request(512, 16, 2, True)
    eng = Engine(tiny, tmp_path, budget=128, caching=True, num_pages=2 * one + 1, max_num_seqs=3)
    hf = hf_mm_model(tiny)
    prompts = []
    for rid in range(5):
        ids, refs, pils = eng.prompt([text(20, 3), png(40 + 7 * rid, 30, 20 + rid % 2), text(4, rid)])
        prompts.append((ids, pils))
        eng.add(rid, ids, refs, 5)
    eng.run(max_turns=2000)
    for rid, (ids, pils) in enumerate(prompts):
        assert eng.outs[rid] == greedy_mm(hf, ids, pils, 5), rid


def test_a_hit_inside_a_span_backs_off_and_stays_exact(tiny, tmp_path, monkeypatch) -> None:
    """A checkpoint inside an image span (planted: checkpoints are kept out of spans, so the placement rule is switched
    off here, as for a cache that held one): the next identical prompt's match backs off below the span and
    recomputes it whole; its output is HF's. With the back-off off too, the scheduler's own check refuses the split."""
    from llm_mini_infer.sched import scheduler as sm

    monkeypatch.setattr(sm.Scheduler, "_ckpt_pos", lambda self, req, c, start, end, d: c)
    eng = Engine(tiny, tmp_path, budget=128, caching=True)
    # an image near the end: the end-of-prefill checkpoint on the grid (4) falls inside the span
    ids, refs, pils = eng.prompt([text(10, 1), png(48, 36, 3), [7]])
    s, e = refs[0].offset, refs[0].offset + refs[0].tokens
    assert len(ids) - 1 - (len(ids) - 1) % 4 < e  # the grid point before the last token lies in the span
    hf = hf_mm_model(tiny)
    for rid in (0, 1):
        eng.add(rid, ids, refs, 3)
        eng.run()
        assert eng.outs[rid] == greedy_mm(hf, ids, pils, 3), rid
    ends, todo = [], list(eng.sched.cache.roots.values())
    while todo:  # every checkpoint in the tree
        n = todo.pop()
        todo.extend(n.children.values())
        if n.ckpt:
            ends.append(n.end)
    assert any(s < x < e for x in ends), (ends, s, e)  # a checkpoint inside the span is there to be hit
    assert eng.cached[1] <= s, (eng.cached[1], s, e, ends)


def test_images_with_mtp_match_transformers(tiny, tmp_path) -> None:
    """Gemma 4's images with its assistant (k = 3): prompts with one and two images in one batch, chunks cut around
    spans, drafts right after a span, then the first prompt again, which hits past its image. The greedy outputs equal
    MTP off's and transformers', with the assistant's drafts (mostly rejected: it is random) and with an oracle's (every
    draft right, so windows and pages grow by accepted runs)."""
    from ..support.tiny_gemma import make_tiny_assistant

    asst = make_tiny_assistant(tmp_path / "assistant", tiny, seed=9)
    hf = hf_mm_model(tiny)
    outs, refs, full = [], {}, {}
    for k, oracle in ((0, False), (3, False), (3, True)):
        eng = Engine(tiny, tmp_path, budget=96, mtp_k=k, assistant=asst if k else None)
        p0 = eng.prompt([text(30, 3), png(48, 36, 4), text(5, 5)])
        p1 = eng.prompt([text(30, 3), png(40, 40, 6), text(4, 7), png(32, 48, 8), text(3, 9)])

        @torch.inference_mode()
        def plant(_turn, eng=eng, k=k, full=full):  # every running slot's drafts become its next k reference tokens
            tb = eng.runner.tables
            for slot, req in enumerate(eng.sched.slot_req):
                if req is None or slot == 0:
                    continue
                seq, at = full[req.rid], int(tb.seq_len[slot])
                for j in range(k):
                    tb.drafts[slot, j] = seq[at + 1 + j] if at + 1 + j < len(seq) else 7

        hook = plant if oracle else None
        eng.add(0, p0[0], p0[1], 14)
        eng.add(1, p1[0], p1[1], 12)
        eng.run(on_turn=hook)
        eng.add(2, p0[0], p0[1], 14)
        eng.run(on_turn=hook)
        span = p0[1][0].offset + p0[1][0].tokens
        assert eng.cached[2] >= span, (k, eng.cached[2], span)
        outs.append(dict(eng.outs))
        if not refs:
            refs = {0: greedy_mm(hf, p0[0], p0[2], 14), 1: greedy_mm(hf, p1[0], p1[2], 12)}
            refs[2] = refs[0]
            full = {0: p0[0] + refs[0], 1: p1[0] + refs[1], 2: p0[0] + refs[0]}
        if oracle:
            assert eng.sched.spec_accepted > eng.sched.spec_draft_tokens // 2
    assert outs[0] == refs and outs[1] == refs and outs[2] == refs


def _fake(budget: int, pc):
    from llm_mini_infer.runner.fake import FakeRunner

    fake = FakeRunner(vocab_size=512, max_positions=pc.max_model_len, encoder_rows=2 * budget, encoder_budget=budget,
                      window=W, window_kv_bytes_per_token=1024, bidirectional_images=True, strict=True, step_ms=0,
                      sample_ms=0, spin_us=0)
    fake.configure(pc)
    return fake


@pytest.mark.parametrize("budget,soft", [(32768, 70), (2048, 280), (300, 280), (200, 280), (4096, 1120)])
def test_the_profile_image_plan_holds_a_budget_of_the_largest_images(tiny, budget, soft) -> None:
    """The memory profile's image step (`GemmaRunner._image_plan` of `profile_images`): one prompt of the step's
    budget whose B - 2 soft tokens are images of max_soft_tokens back to back and one of the rest (468 of 70 and one of
    6 at 32,768), each with its ENCODE (keys from 1 << 62, encoder rows consecutive), its span and a grid within the
    tiny tower's 256-entry position table (the 734 tokens left at 4,096 and 1,120 are a prime strip's: 729 of them),
    on the pages of the prompt and two decode steps. The strict fake runner takes it, and refuses it without spans."""
    from types import SimpleNamespace

    import numpy as np

    from llm_mini_infer.runner.base import PlanError, PoolConfig
    from llm_mini_infer.runner.cuda.gemma import GemmaRunner
    from llm_mini_infer.runner.cuda.images import ImageEncoder, VisionOptions, profile_images, synthetic_grid
    from llm_mini_infer.sched.plan import PlanBuilder

    ps, fps = 4, 16
    img = ImageEncoder(tiny, VisionOptions(encoder_rows=2 * budget, encoder_budget=budget, max_soft_tokens=soft),
                       torch.device("cpu"), torch.float32, per_image=True)
    r = SimpleNamespace(fps=fps, ps=ps, img=img, bidi=True, cfg=SimpleNamespace(vocab_size=512))
    for name in ("_image_plan", "_synthetic_tokens"):
        setattr(r, name, getattr(GemmaRunner, name).__get__(r))
    images = profile_images(budget - 2, soft)
    want = [h * w // 9 for h, w in (synthetic_grid(t, 3, 256) for t in images)]
    assert want == images or (budget, soft, images[-1], want[-1]) == (4096, 1120, 734, 729)
    n = sum(want) + 2
    nf, nw = -(-(n + 4) // fps), -(-(n + 4) // ps)
    pc = PoolConfig(page_size=fps, num_pages=nf + nw + 2, num_state_slots=5, max_num_seqs=4,
                    max_num_batched_tokens=budget, max_model_len=budget + 64, eos_token_ids=(), mtp_k=0,
                    window_page_size=ps)

    def plan_of():
        return r._image_plan(PlanBuilder(4, budget, fps, pc.max_model_len, 0, W, ps), 1, slot=1, images=images,
                             pages_from=1)

    plan = plan_of()
    starts = np.cumsum([0, *want])
    assert len(plan.tokens) == n and plan.enc_tokens.tolist() == want
    assert plan.enc_key.tolist() == [(1 << 62) + i for i in range(len(want))]
    assert plan.enc_rows.tolist() == list(range(n - 2)) and plan.enc_row0.tolist() == starts[:-1].tolist()
    assert plan.tok_emb.tolist() == [-1, *range(n - 2), -1]
    assert plan.span_sched.tolist() == [0] * len(want)
    assert plan.span_start.tolist() == (1 + starts[:-1]).tolist()
    assert plan.span_end.tolist() == (1 + starts[1:]).tolist()
    assert [h * w for _, h, w in plan.enc_grid.tolist()] == [9 * t for t in want]
    assert int(plan.enc_grid[:, 1:].max()) <= 256 and (plan.enc_pixels == 0).all()
    assert plan.page_id.tolist() == list(range(1, 1 + nf))
    assert plan.wpage_id.tolist() == list(range(1 + nf, 1 + nf + nw))
    fake = _fake(budget, pc)
    h = fake.launch_forward(plan)
    fake.launch_sample(h)
    fake.wait(h)
    r.bidi = False
    with pytest.raises(PlanError, match="SPANS"):
        _fake(budget, pc).launch_forward(plan_of())


def test_the_runner_stages_and_encodes_469_images_in_a_step(tiny) -> None:
    """The profile's image step at the smallest soft-token budget (70: the most images a step holds) and a
    32,768-token budget (468 images of 70 and one of 6) through
    the CPU runner's host side as `launch_forward` runs it, short of the forward (32,768 tokens through the full
    layers' reference attention would take minutes): its ENCODE and spans fit the step's staging, and with one tower
    call per image (the tower's buckets then end at its largest image's, 648 patches) every image's rows are written,
    each 70-token image's those of the first (the same pixels)."""
    from llm_mini_infer.runner.cuda.gemma import GemmaRunner
    from llm_mini_infer.runner.cuda.images import VisionOptions, profile_images

    from ..support.tiny_gemma import gemma_pools

    budget = 32768
    pc = gemma_pools(max_num_seqs=4, budget=budget, page_size=4, num_pages=budget // 2 + 64,
                     max_model_len=budget + 64)
    r = GemmaRunner(tiny, device="cpu", dtype=torch.float32, max_num_seqs=4, max_num_batched_tokens=budget,
                    max_model_len=pc.max_model_len, check=True, encode_per_image=True,
                    vision=VisionOptions(encoder_rows=2 * budget, encoder_budget=budget, max_soft_tokens=70))
    assert r.img.per_image and r.img.buckets == [261, 387, 513, 648]
    r.configure(pc)
    images = profile_images(budget - 2, 70)
    plan = r._image_plan(r._builder(), 1, slot=1, images=images, pages_from=1)
    assert len(plan.enc_key) == len(plan.span_sched) == 469
    with torch.inference_mode():
        st = r._staging[0]
        st.reset()
        r._stage_effects(plan, st)
        enc = r.img.stage(plan, st, True)
        lay = r._layout(plan, st)
        kind, size = r._mode(lay, False)
        r._stage_static(lay, st, size, kind)
        st.upload(r.static)
        r.img.pool.zero_()
        r.img.encode(plan, st, enc, False)
    rows = r.img.pool[:budget - 2]
    assert len(lay.spans[0]) == 469 and bool((rows.abs().sum(1) > 0).all())
    assert all(torch.equal(rows[70 * i:70 * (i + 1)], rows[:70]) for i in range(1, 468))
