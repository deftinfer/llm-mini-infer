"""Images through the scheduler and the validating fake runner.

The fake hashes each image token as its image's key (the radix key), checks that spans are read in order from rows
an ENCODE wrote, that the MTP look-ahead row matches, and mixes the request's M-RoPE offset into its samples: a
wrong image, a span fed out of order, a stale encoder row or a wrong offset changes the predicted outputs or raises
PlanError (`runner.fake`). Each run also checks the encoder cache's references and the pixel releases.
"""

from __future__ import annotations

from hypothesis import given

from llm_mini_infer.cache.encoder import EncoderCache
from llm_mini_infer.cache.pools import pages_for_one_request
from llm_mini_infer.engine.messages import FinishReason
from llm_mini_infer.sched.plan import reserve_positions

from ..support.fuzzing import campaign_settings, saved_on_failure, st
from ..support.harness import Harness, Spec
from ..support.sched_cases import VOCAB, check_plans
from ..support.workloads import image_prompt, image_workloads, run_image_workload, text


def run(specs: list[Spec], **kw) -> Harness:
    kw.setdefault("encoder_rows", 1024)
    h = Harness(specs=specs, vocab=VOCAB, **kw)
    h.run(check_every_turn=True)
    check_plans(h)
    h.check()
    return h


def test_one_image_request_is_exact() -> None:
    """A 12 x 16 patch grid: 48 tokens at 20, encoded once, in the step that prefills the span."""
    prompt = image_prompt(text(1, 20), 48, text(2, 15))
    spec = Spec(rid=0, prompt_len=0, prompt=prompt, max_tokens=8, seed=5, images=[(20, 12, 16, 7)])
    h = run([spec], max_tokens=256, page_size=4, num_pages=100, checkpoint_slots=4, grid=16)
    assert h.plans[0]["encode"] == [48] and h.plans[0]["image_tokens"] == 48
    assert sum(len(p["encode"]) for p in h.plans) == 1


def test_a_span_split_across_chunks_reads_its_cached_embeddings() -> None:
    """A 32-token budget prefills a 96-token span over several steps: the image is encoded once, in the first, and
    the later chunks read its rows in order (the fake checks the token indices)."""
    prompt = image_prompt(text(3, 10), 96, text(4, 20))
    spec = Spec(rid=0, prompt_len=0, prompt=prompt, max_tokens=6, seed=9, images=[(10, 16, 24, 1)])
    h = run([spec], max_tokens=32, page_size=4, num_pages=100, checkpoint_slots=4, grid=8, encoder_budget=128)
    enc = [p["encode"] for p in h.plans]
    assert sum(len(e) for e in enc) == 1 and enc[0] == [96]
    assert sum(p["image_tokens"] for p in h.plans) == 96 and len([p for p in h.plans if p["image_tokens"]]) >= 3


def test_the_encoder_budget_cuts_a_chunk_before_the_next_image() -> None:
    """Two 64-token images, an encoder budget of 64 per step: the chunk ends before the second image, which the next
    step encodes; at most 64 image tokens are encoded per step (the fake checks the budget too)."""
    prompt = image_prompt(text(5, 8), 64, image_prompt([1, 2, 3], 64, text(6, 10)))
    spec = Spec(rid=0, prompt_len=0, prompt=prompt, max_tokens=4, seed=3, images=[(8, 16, 16, 1), (75, 16, 16, 2)])
    h = run([spec], max_tokens=256, page_size=4, num_pages=100, checkpoint_slots=4, grid=8, encoder_budget=64)
    assert [p["encode"] for p in h.plans[:2]] == [[64], [64]]
    assert h.plans[0]["n_query"] == [72]  # cut at the second span (75), rounded down to the grid like a budget cut


def test_mtp_cuts_one_token_earlier_for_the_drafters_look_ahead() -> None:
    """Under MTP the drafter's input after a non-final chunk is the next token: a chunk that cannot encode the next
    image ends one token before its span (here 74, then the grid's 72; with a 1-token grid, 74). When the image fits,
    a chunk ending right before the span carries the span's first row as its look-ahead (`sched_next_emb`)."""
    prompt = image_prompt(text(7, 8), 64, image_prompt([1, 2, 3], 64, text(8, 10)))
    spec = Spec(rid=0, prompt_len=0, prompt=prompt, max_tokens=6, seed=13, images=[(8, 16, 16, 1), (75, 16, 16, 2)])
    h = run([spec], max_tokens=256, page_size=4, num_pages=100, checkpoint_slots=4, grid=8, encoder_budget=64,
            mtp_k=3)
    assert h.plans[0]["n_query"] == [72] and h.plans[0]["encode"] == [64]
    spec1 = Spec(rid=0, prompt_len=0, prompt=prompt, max_tokens=6, seed=13, images=[(8, 16, 16, 1), (75, 16, 16, 2)])
    h1 = run([spec1], max_tokens=256, page_size=1, num_pages=400, checkpoint_slots=4, grid=1, encoder_budget=64,
             mtp_k=3)
    assert h1.plans[0]["n_query"] == [74] and h1.plans[0]["encode"] == [64]
    spec2 = Spec(rid=0, prompt_len=0, prompt=prompt, max_tokens=6, seed=13, images=[(8, 16, 16, 1), (75, 16, 16, 2)])
    h2 = run([spec2], max_tokens=8, page_size=4, num_pages=100, checkpoint_slots=4, grid=8, encoder_budget=128,
             mtp_k=3)
    assert h2.plans[0]["n_query"] == [8] and h2.plans[0]["next_emb"][0] >= 0  # the first image token's row


def test_a_hit_inside_a_span_reads_the_cached_embeddings_or_encodes_again() -> None:
    """The second request is identical: it resumes from the first's checkpoint inside the image span (the span ends
    past the grid point), so it needs the image's embeddings: from the encoder cache when they are still there, from
    a new ENCODE when a third image evicted them."""
    prompt = image_prompt(text(9, 10), 64, text(10, 6))  # span [10, 74), 80 tokens: end-of-prefill checkpoint at 72
    a = Spec(rid=0, prompt_len=0, prompt=prompt, max_tokens=4, seed=21, images=[(10, 16, 16, 1)])
    b = Spec(rid=1, prompt_len=0, prompt=prompt, max_tokens=4, seed=22, images=[(10, 16, 16, 1)], arrive=20)
    h = run([a, b], max_tokens=256, page_size=4, num_pages=100, checkpoint_slots=4, grid=8)
    assert h.results[1].cached == 72 and h.sched.encoder_hits == 1 and h.sched.encoder_images == 1
    other = image_prompt(text(11, 10), 64, text(12, 6))
    c = Spec(rid=2, prompt_len=0, prompt=other, max_tokens=4, seed=23, images=[(10, 16, 16, 9)], arrive=10)
    b2 = Spec(rid=1, prompt_len=0, prompt=prompt, max_tokens=4, seed=22, images=[(10, 16, 16, 1)], arrive=20)
    a2 = Spec(rid=0, prompt_len=0, prompt=prompt, max_tokens=4, seed=21, images=[(10, 16, 16, 1)])
    h = run([a2, c, b2], max_tokens=256, page_size=4, num_pages=100, checkpoint_slots=4, grid=8, encoder_rows=64)
    assert h.results[1].cached == 72 and h.sched.encoder_images == 3  # image 1 evicted by image 9, then again


SALT_PROMPT = image_prompt(text(9, 10), 64, text(10, 6))
SALT_CFG = dict(max_tokens=256, page_size=4, num_pages=200, checkpoint_slots=6, grid=8)


def salted(salts: list[str | None], gap: int = 20, **kw) -> Harness:
    specs = [Spec(rid=i, prompt_len=0, prompt=SALT_PROMPT, max_tokens=4, seed=21 + i, images=[(10, 16, 16, 1)],
                  arrive=gap * i, salt=s, **kw.get(i, {})) for i, s in enumerate(salts)]
    return run(specs, **SALT_CFG)


def test_two_salts_share_neither_the_embeddings_nor_the_kv() -> None:
    """The same image and text under two salts, 20 turns apart: two encodes, no encoder hit, and no cached prompt
    tokens for the second."""
    h = salted(["a", "b"])
    assert h.sched.encoder_images == 2 and h.sched.encoder_hits == 0
    assert h.results[1].cached == 0
    assert {e.key[2] for e in h.sched.encoder.entries.values()} == {"a", "b"}


def test_one_salt_or_none_shares_the_embeddings() -> None:
    """The same image under one salt, and under no salt: one encode and one hit, the second request resuming from the
    first's checkpoint."""
    for salt in ("a", None):
        h = salted([salt, salt])
        assert h.sched.encoder_images == 1 and h.sched.encoder_hits == 1, salt
        assert h.results[1].cached == 72
        assert {e.key[2] for e in h.sched.encoder.entries.values()} == {salt}


def test_a_salted_and_an_unsalted_request_share_nothing() -> None:
    """A salted request and an unsalted one with the same image: two encodes, no hit, whichever comes first."""
    for salts in (["a", None], [None, "a"]):
        h = salted(salts)
        assert h.sched.encoder_images == 2 and h.sched.encoder_hits == 0, salts
        assert h.results[1].cached == 0
        assert {e.key[2] for e in h.sched.encoder.entries.values()} == set(salts)


def test_concurrent_requests_encode_once_per_salt() -> None:
    """Two requests with the same image in the same turn: two ENCODEs under two salts, one under a single salt."""
    h = salted(["a", "b"], gap=0)
    assert h.sched.encoder_images == 2 and sum(len(p["encode"]) for p in h.plans) == 2
    h = salted(["a", "a"], gap=0)
    assert h.sched.encoder_images == 1 and sum(len(p["encode"]) for p in h.plans) == 1
    assert {e.key[2] for e in h.sched.encoder.entries.values()} == {"a"}


def test_a_failed_request_forgets_its_own_salts_entry_only() -> None:
    """A request under salt "b" fails on a non-finite sample: its image's entry under "b" is dropped, the one under
    "a", left by a finished request with the same image, stays."""
    specs = [Spec(rid=0, prompt_len=0, prompt=SALT_PROMPT, max_tokens=4, seed=21, images=[(10, 16, 16, 1)], salt="a"),
             Spec(rid=1, prompt_len=0, prompt=SALT_PROMPT, max_tokens=8, seed=22, images=[(10, 16, 16, 1)], salt="b",
                  arrive=20, nonfinite_after=2)]
    h = Harness(specs=specs, vocab=VOCAB, encoder_rows=1024, **SALT_CFG)
    h.run(check_every_turn=True)
    check_plans(h)
    h.check(failed_ok=True)
    assert h.results[0].finish != FinishReason.ERROR and h.results[1].finish == FinishReason.ERROR
    assert h.sched.encoder_images == 2
    assert {e.key[2] for e in h.sched.encoder.entries.values()} == {"a"}


def test_a_hit_never_crosses_a_different_image() -> None:
    """Same text, different images: the second request's hit stops before the span (its keys differ), on the branch
    checkpoint; the same image again hits past the span."""
    pre, post = text(13, 40), text(14, 12)
    prompt = image_prompt(pre, 16, post)
    specs = [Spec(rid=i, prompt_len=0, prompt=prompt, max_tokens=3, seed=40 + i, images=[(40, 8, 8, iid)],
                  arrive=15 * i) for i, iid in enumerate([1, 2, 2])]
    h = run(specs, max_tokens=256, page_size=4, num_pages=100, checkpoint_slots=6, grid=8)
    assert h.results[1].cached <= 40  # not into the span of another image
    assert h.results[2].cached > 56  # the same image: past the span (it ends at 56)


def test_colliding_span_keys_are_told_apart_by_the_digest() -> None:
    """Images whose span keys collide (one image id: the same 62 bits) but whose digests differ (digest ids 1 and 2), in
    the same text: the second request's hit stops before the span, as for another image, and its image is encoded
    again, never served the first one's KV or embeddings; then each image hits past its span on its own path."""
    pre, post = text(13, 40), text(14, 12)
    prompt = image_prompt(pre, 16, post)
    specs = [Spec(rid=i, prompt_len=0, prompt=prompt, max_tokens=3, seed=40 + i, images=[(40, 8, 8, 1, d)],
                  arrive=15 * i) for i, d in enumerate([1, 2, 2, 1])]
    h = run(specs, max_tokens=256, page_size=4, num_pages=100, checkpoint_slots=6, grid=8)
    assert h.results[1].cached <= 40 and h.results[2].cached > 56 and h.results[3].cached > 56
    assert h.sched.encoder_images == 2 and h.sched.encoder_hits == 0


def test_a_hint_after_shared_instructions_hits_them_exactly() -> None:
    """Long shared instructions with cache_control (a hint at their end), then a unique image and item per request.
    The end-of-prefill checkpoint lies past the shared part, so without the hint the second request misses; with it,
    every request after the first hits exactly the hint."""
    inst = text(15, 300)
    specs = [Spec(rid=i, prompt_len=0, prompt=image_prompt([*inst, 1], 36, text(16 + i, 12)), max_tokens=3,
                  seed=60 + i, images=[(301, 12, 12, 100 + i)], hints=[300], arrive=10 * i) for i in range(6)]
    h = run(specs, max_tokens=512, page_size=16, num_pages=200, checkpoint_slots=4, grid=64, max_model_len=1024)
    assert [h.results[i].cached for i in range(6)] == [0] + [300] * 5
    nohint = [Spec(rid=i, prompt_len=0, prompt=image_prompt([*inst, 1], 36, text(16 + i, 12)), max_tokens=3,
                   seed=60 + i, images=[(301, 12, 12, 100 + i)], arrive=10 * i) for i in range(6)]
    h = run(nohint, max_tokens=512, page_size=16, num_pages=200, checkpoint_slots=4, grid=64, max_model_len=1024)
    assert [h.results[i].cached for i in range(4)] == [0, 0, 256, 256]  # the branch point's checkpoint at 256


def test_preemption_mid_span_resumes_exactly() -> None:
    """Page pressure preempts a request in the middle of its image span; its readmission references the image again
    (re-encoding it if the cache let it go) and its outputs stay exact."""
    specs = [Spec(rid=i, prompt_len=0, prompt=image_prompt(text(20 + i, 6), 64, text(30 + i, 8)), max_tokens=20,
                  seed=80 + i, images=[(6, 16, 16, 200 + i)], priority=i % 2, arrive=i) for i in range(3)]
    min_pages = pages_for_one_request(160, 4, reserve_positions(2, 0), True) + 1
    h = run(specs, max_num_seqs=3, max_tokens=24, page_size=4, num_pages=min_pages, checkpoint_slots=2, grid=8,
            max_model_len=160, policy="priority", encoder_rows=128)
    assert h.sched.num_preemptions > 0


def test_a_chunk_waiting_for_pages_gives_back_the_rows_a_request_ahead_waits_for() -> None:
    """Request 1's 24-token image fills the 2-block encoder cache, and request 1 is inside its span when the pool runs
    out: its chunk at 64 waits for 8 pages (4 are free). Request 0, admitted first, reaches its own 8-token image at 64
    and gets no chunk (its image waits for rows), so it preempts nothing, and request 1 cannot preempt request 0: unless
    request 1 gives back its reference, every plan is the same empty one. Once a plan comes out empty with nothing in
    flight, it does; in the next step request 0 encodes its image and preempts request 1 for the pages, whose
    readmission encodes its image again."""
    specs = [Spec(rid=0, prompt_len=0, prompt=image_prompt(text(1, 64), 8, text(2, 4)), max_tokens=4, seed=7,
                  images=[(64, 4, 8, 2)]),
             Spec(rid=1, prompt_len=0, prompt=image_prompt(text(3, 48), 24, text(4, 8)), max_tokens=4, seed=8,
                  images=[(48, 8, 12, 4)])]
    h = run(specs, max_num_seqs=2, max_tokens=16, chunk_cap=8, page_size=1, grid=8, num_pages=133, max_model_len=96,
            prefix_caching=False, encoder_rows=32, encoder_budget=24)
    assert [p["pos0"] for p in h.plans[6:9]] == [[48, 48], [56, 56], [64]] and h.plans[8]["encode"] == [8]
    assert h.plans[8]["free"] == [2] and h.sched.num_preemptions == 1 and h.sched.encoder_images == 3


def test_a_look_ahead_given_back_is_read_again_from_another_copy() -> None:
    """The same cycle through a drafter's look-ahead (MTP with a drafter that writes KV): request 1's chunk [48, 56)
    encodes its 24-token image (rows 0-23) for the look-ahead at 56, and its chunk [56, 64) waits for 8 pages (7 are
    free); request 0 stops at 63, before its 4-token image, which cannot have a block. Request 1 gives the image back,
    request 0 encodes its own and finishes on the 7 pages without preempting anyone, and request 1's chunk at 56 encodes
    the image again, in rows from 16: the first row it reads is not the row its look-ahead read, but holds the same
    token (the fake compares the image token)."""
    specs = [Spec(rid=0, prompt_len=0, prompt=image_prompt(text(1, 64), 4, text(2, 1)), max_tokens=1, seed=7,
                  images=[(64, 4, 4, 2)]),
             Spec(rid=1, prompt_len=0, prompt=image_prompt(text(3, 56), 24, text(4, 8)), max_tokens=4, seed=8,
                  images=[(56, 8, 12, 4)])]
    h = run(specs, max_num_seqs=2, max_tokens=16, chunk_cap=8, page_size=1, grid=8, num_pages=127, max_model_len=96,
            prefix_caching=False, encoder_rows=32, encoder_budget=24, mtp_k=2, draft_kv=True, accept_p=0.0)
    assert h.plans[6]["next_emb"] == [-1, 0] and [p["pos0"] for p in h.plans[7:10]] == [[56], [63], [56]]
    assert h.plans[9]["encode"] == [24] and h.plans[9]["next_emb"] == [24]  # the new copy's token 8 (was row 8)
    assert h.sched.num_preemptions == 0 and h.sched.encoder_images == 3


def test_an_abort_releases_the_pixels_after_the_steps_in_flight() -> None:
    prompt = image_prompt(text(40, 10), 64, text(41, 10))
    spec = Spec(rid=0, prompt_len=0, prompt=prompt, max_tokens=40, seed=91, images=[(10, 16, 16, 3)], abort_at=3)
    h = Harness(specs=[spec], vocab=VOCAB, max_tokens=256, page_size=4, num_pages=100, checkpoint_slots=4, grid=8,
                encoder_rows=256)
    h.run(check_every_turn=True)
    assert h.results[0].finish == FinishReason.ABORT and h.released == {0}


def test_images_without_an_encoder_are_rejected() -> None:
    spec = Spec(rid=0, prompt_len=0, prompt=image_prompt(text(1, 4), 16, [1]), max_tokens=2, seed=1,
                images=[(4, 8, 8, 1)])
    h = Harness(specs=[spec], vocab=VOCAB, max_tokens=64, page_size=4, num_pages=50, checkpoint_slots=2, grid=8)
    h.run()
    assert h.results[0].finish == FinishReason.ERROR and h.released == {0}
    big = Spec(rid=0, prompt_len=0, prompt=image_prompt([], 64, [1]), max_tokens=2, seed=1, images=[(0, 16, 16, 1)])
    h = Harness(specs=[big], vocab=VOCAB, max_tokens=64, page_size=4, num_pages=50, checkpoint_slots=2, grid=8,
                encoder_rows=256, encoder_budget=32)
    h.run()
    assert h.results[0].finish == FinishReason.ERROR
    # Adjacent spans (no text between, which no template emits): rejected. Under MTP they could need two images'
    # embeddings at once, more than a small encoder cache holds (found by the image fuzz suite: a hang).
    adj = Spec(rid=0, prompt_len=0, prompt=image_prompt([5, 36], 28, [0]), max_tokens=1, seed=2,
               images=[(2, 4, 4, 1), (6, 8, 12, 4)])
    h = Harness(specs=[adj], vocab=VOCAB, max_tokens=16, page_size=2, num_pages=100, checkpoint_slots=1, grid=8,
                encoder_rows=32, encoder_budget=24, mtp_k=1)
    h.run()
    assert h.results[0].finish == FinishReason.ERROR


# -- the image fuzz suite (tests/scripts/fuzz_campaign.py: "images") ---------------------------------------------------


def cache_state(c: EncoderCache) -> tuple:
    """Everything a later choice of the encoder cache reads: its entries' blocks and references, their recency's
    order, the free list's order and the eviction count."""
    order = sorted(c.entries, key=lambda k: c.entries[k].last_used)
    return ({k: (e.blocks, e.refs) for k, e in c.entries.items()}, order, list(c._free), c.evicted)


@given(st.data())
def test_a_rolled_back_reservation_leaves_the_encoder_cache_as_it_was(data) -> None:
    """Two caches take the same allocations, references and reads; one of them also reserves chunks' images now and
    then (cached ones referenced, new ones allocated, entries evicted for them) and rolls each back: after every
    rollback the two are alike in every entry, reference, recency and free block, so every later allocation evicts
    and gets the same blocks: a chunk the scheduler cuts or does not plan leaves no trace (`_image_window`)."""
    rows = data.draw(st.sampled_from([64, 128, 256]))
    a, b = EncoderCache(rows), EncoderCache(rows)
    keys = st.integers(1, 12)
    for _ in range(data.draw(st.integers(5, 60))):
        op, k = data.draw(st.sampled_from(["alloc", "ref", "unref", "get", "reserve"])), data.draw(keys)
        if op == "alloc" and k not in a.entries:
            n = data.draw(st.integers(1, 80))
            ea, eb = a.alloc(k, n), b.alloc(k, n)
            assert (ea is None) == (eb is None) and (ea is None or ea.blocks == eb.blocks)
        elif op in ("ref", "get") and k in a.entries:
            getattr(a, op)(a.entries[k] if op == "ref" else k)
            getattr(b, op)(b.entries[k] if op == "ref" else k)
        elif op == "unref" and k in a.entries and a.entries[k].refs:
            a.unref(a.entries[k])
            b.unref(b.entries[k])
        elif op == "reserve":
            for k2 in data.draw(st.lists(keys, max_size=5)):
                if k2 in a.entries or data.draw(st.booleans()):
                    a.reserve(k2)
                else:
                    a.reserve(k2, data.draw(st.integers(1, 80)))
            a.rollback()
            a.check()
        assert cache_state(a) == cache_state(b)


@campaign_settings(300)
@given(image_workloads())
def test_random_image_workloads(w) -> None:
    with saved_on_failure("images", w):
        run_image_workload(w)


def test_images_per_step_numbers() -> None:
    """The engine's startup line on images (`_images_per_step`) for Gemma 4: 70 soft tokens an image (72 with
    `<|image>` and `<image|>`), 2,048-token steps and encoder budget, k = 3, a 16-position grid and the mixed-step
    bound measured on Gemma 4 31B FP8 (30,155): 28 images a step alone or next to 8 decode rows, 14 next to 32: 13 in
    the bound's 942 prefill tokens, and the span of a chunk that starts while 6 are left, which runs past the bound
    (`Scheduler._span_end`). At 1,120 soft tokens none fits the 942 tokens, but that one does. The encoder alone would
    take 31 images of 66 tokens; with the bound off, 32 rows leave 1,920 tokens, no
    span past them."""
    from llm_mini_infer.engine.process import _images_per_step

    kw = dict(tokens=70, extra=2, encoder_budget=2048, max_num_batched_tokens=2048, row_tokens=30_155, width=4,
              grid=16)
    assert [_images_per_step(rows=r, **kw) for r in (0, 8, 32)] == [28, 28, 14]
    assert [_images_per_step(**{**kw, "tokens": 1120}, rows=r) for r in (0, 32)] == [1, 1]
    assert _images_per_step(**{**kw, "tokens": 66, "max_num_batched_tokens": 8192}, rows=0) == 31
    assert _images_per_step(**{**kw, "row_tokens": 0}, rows=32) == 1920 // 72
    assert _images_per_step(**{**kw, "row_tokens": 942 * 32 // 2}, rows=32) == 6 + 1  # 471 tokens: 6 x 72 + 39
    assert _images_per_step(**{**kw, "row_tokens": 72 * 13 * 32}, rows=32) == 13  # none left for a 14th to start


def test_the_images_line_is_what_the_scheduler_does() -> None:
    """The startup line's number against the scheduler at the same settings: 32 requests decode while 40 one-image
    requests arrive (`<|image>`, 70 soft tokens, `<image|>`, a token; each finishes with its prefill, so 32 rows keep
    decoding). The first step prefills 12 whole and only the `<|image>` of the others; the next two prefill 13 whole
    in the bound's 942 tokens, then a 14th's span, past the bound (1,006 tokens): 14 in a step, the line's number,
    and no step holds more than the line says for its decode rows."""
    from llm_mini_infer.engine.process import _images_per_step

    specs = [Spec(rid=i, prompt_len=0, prompt=text(i, 20), max_tokens=24, seed=100 + i, ignore_eos=True)
             for i in range(32)]
    specs += [Spec(rid=100 + j, prompt_len=0, prompt=[1, *([0] * 70), 2, 3], max_tokens=1, seed=500 + j,
                   images=[(1, 14, 20, 1 + j)], arrive=6) for j in range(40)]
    h = run(specs, max_num_seqs=96, max_tokens=2048, page_size=16, num_pages=2000, grid=16, mtp_k=3, accept_p=0.0,
            encoder_rows=4096, encoder_budget=2048, row_tokens=30_155, bidirectional=True, hybrid=False,
            draft_kv=False, prefix_caching=False)
    kw = dict(tokens=70, extra=2, encoder_budget=2048, max_num_batched_tokens=2048, row_tokens=30_155, width=4,
              grid=16)
    for p in h.plans:
        assert len(p["spans"]) <= _images_per_step(rows=sum(p["device"]), **kw), p["step"]
    assert max(len(p["spans"]) for p in h.plans if sum(p["device"]) == 32) == 14 == _images_per_step(rows=32, **kw)
