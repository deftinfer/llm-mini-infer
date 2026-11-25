"""Pool sizing at startup (architecture.md, "Pools"): the KV pool must hold one request of max_model_len; a
sliding-window model's KV arena."""

from __future__ import annotations

import json
import logging

import pytest

from llm_mini_infer.cache.pools import (
    arena_full_page_size,
    pages_for_one_request,
    size_arena,
    size_pools,
    window_pages_for_one_request,
)

from ..support.api_util import Server
from ..support.fixtures import require_qwen

GEO = {"kv_bytes_per_token": 1000, "state_slot_bytes": 0, "page_size": 16, "max_num_seqs": 4, "memory_bytes": 1 << 34}


def test_pages_for_one_request() -> None:
    # 4,095 prompt tokens + 2 reserved = 4,097 positions = 257 pages of 16, plus the copy-on-write page
    assert pages_for_one_request(4096, 16, 2, True) == 258
    assert pages_for_one_request(4096, 16, 2, False) == 257
    assert pages_for_one_request(160, 1, 2, True) == 162


def test_startup_refuses_a_pool_shorter_than_one_request() -> None:
    """199 pages of 16 tokens cannot hold a request of 4,096 (258 pages): refused, naming the longest length that
    fits, (199 − 1) × 16 − 2 + 1 = 3,167."""
    with pytest.raises(ValueError, match=r"largest max_model_len that fits is 3,167"):
        size_pools(**GEO, max_model_len=4096, num_pages=200)
    sizes = size_pools(**GEO, max_model_len=3167, num_pages=200)  # exactly full
    assert sizes.max_model_len == 3167 and pages_for_one_request(3167, 16, 2, True) == 199
    with pytest.raises(ValueError, match="3,167"):
        size_pools(**GEO, max_model_len=3168, num_pages=200)


def test_a_derived_max_model_len_is_capped(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="llm_mini_infer.cache.pools"):
        sizes = size_pools(**GEO, max_model_len=4096, num_pages=200, cap_max_model_len=True)
    assert sizes.max_model_len == 3167 and "3,167" in sizes.reasons["max_model_len"]
    assert any("capping the derived max_model_len to 3,167" in r.message for r in caplog.records)


def test_server_refuses_or_caps_at_startup(tmp_path) -> None:
    """End to end: a configured max_model_len the pool cannot hold stops the server at startup with the error; a
    derived one (config.json says 262,144) is capped, and the frontends serve the capped length."""
    model = require_qwen()
    srv = Server(model, tmp_path, extra={"dev": {"cache": {"num_pages": 100}}})
    try:
        with pytest.raises(RuntimeError, match=r"largest max_model_len that fits is 1,567"):
            srv.start()
    finally:
        srv.stop()
    srv = Server(model, tmp_path, extra={"model": {"max_model_len": None},
                                         "dev": {"cache": {"num_pages": 300}}}).start()
    try:
        status, body = srv.get("/v1/models")
        assert status == 200 and json.loads(body)["data"][0]["max_model_len"] == 4767
    finally:
        assert srv.stop() == 0, srv.log_path.read_text()[-2000:]


# -- the KV arena of a sliding-window model ---------------------------------------------------------------------------

GEMMA = {"kv_bytes_per_token": 40_960, "window_kv_bytes_per_token": 819_200, "window": 1024}  # Gemma 4 31B in BF16


def test_arena_full_pages_fill_a_window_page() -> None:
    """A full page holds as many bytes as a window page: 320 positions of Gemma 4 31B's full layers (V only, 40 KiB
    each) next to 16 of its sliding layers (800 KiB each); a byte count that does not divide is refused."""
    assert arena_full_page_size(16, 40_960, 819_200) == 320
    assert arena_full_page_size(8, 40_960, 819_200) == 160
    with pytest.raises(ValueError, match="one arena cannot hold both"):
        arena_full_page_size(16, 40_961, 819_200)


def test_size_arena_takes_all_the_memory_and_logs_concurrency(caplog) -> None:
    """25.37 GiB (Gemma 4 31B's pools in BF16 on an RTX PRO 6000) is 2,078 pages of 12.5 MiB; a request of 4,096
    tokens holds 13 full and 66 window pages, so about 26 run at once, also at max_model_len 4,096."""
    mem = int(25.37 * 2**30)
    with caplog.at_level(logging.INFO, logger="llm_mini_infer.cache.pools"):
        sz = size_arena(memory_bytes=mem, **GEMMA, window_page_size=16, max_num_seqs=64, max_model_len=4096,
                        max_num_batched_tokens=2048)
    assert sz.num_pages == mem // (16 * 819_200) == 2078 and sz.page_size == 320 and sz.max_model_len == 4096
    msg = next(r.message for r in caplog.records if r.message.startswith("KV arena"))
    assert "about 26 requests of 4,096 tokens at once (26 at max_model_len 4,096)" in msg


def test_size_arena_refuses_or_caps_a_request_it_cannot_hold() -> None:
    """One request needs its full pages at max_model_len (14 at 4,096, the copied page included) and its peak of window
    pages (194: a 2,048-token chunk after 1,023 earlier positions): 208 pages; 200 refuse it, or cap a derived
    length."""
    kw = dict(**GEMMA, window_page_size=16, max_num_seqs=64, max_model_len=4096, max_num_batched_tokens=2048,
              memory_bytes=1 << 40)
    peak = window_pages_for_one_request(1024, 2048, 16, 2)
    assert peak == 194 and pages_for_one_request(4096, 320, 2, True) == 14
    size_arena(**kw, num_pages=209)  # exactly full
    with pytest.raises(ValueError, match="largest max_model_len that fits"):
        size_arena(**kw, num_pages=200)
    sz = size_arena(**kw, num_pages=200, cap_max_model_len=True)
    assert sz.max_model_len == (200 - 1 - 194 - 1) * 320 - 2 + 1


def test_an_fp8_arena_holds_twice_the_full_positions_a_page(caplog) -> None:
    """With the full layers' KV in FP8 (`cache.kv_cache_dtype fp8`: 20 KiB a position) a full page is 640
    positions, the window pages unchanged. In 4,000 pages a request of 32,768 tokens then holds 52 full pages and 66
    window pages (118, against 169 in BF16), so 33 fit at once against 23, and 8 of 262,144 against 4; a length the
    arena caps is about twice as long."""
    assert arena_full_page_size(16, 20_480, 819_200) == 640
    kw = dict(memory_bytes=0, window_kv_bytes_per_token=819_200, window=1024, window_page_size=16, max_num_seqs=64,
              max_num_batched_tokens=2048)
    for kv, page, at_32k, at_262k in ((40_960, 320, 23, 4), (20_480, 640, 33, 8)):
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="llm_mini_infer.cache.pools"):
            sz = size_arena(**kw, kv_bytes_per_token=kv, max_model_len=262_144, num_pages=4000, reference_len=32_768)
        msg = next(r.message for r in caplog.records if r.message.startswith("KV arena"))
        assert sz.page_size == page and f"full page of {page} positions" in msg
        assert f"about {at_32k} requests of 32,768 tokens at once ({at_262k} at max_model_len 262,144)" in msg
        capped = size_arena(**kw, kv_bytes_per_token=kv, max_model_len=4096, num_pages=200, cap_max_model_len=True)
        assert capped.max_model_len == (200 - 1 - 194 - 1) * page - 1


def test_a_pool_that_does_not_fit_names_the_settings_that_size_it(caplog) -> None:
    """The errors name the settings that size the state pool: dev.cache.num_state_slots when set (max_num_seqs and
    checkpoint_slots then change nothing), max_num_seqs and checkpoint_slots otherwise, max_num_seqs alone without
    prefix caching, neither without recurrent state. A set checkpoint_slots that is dropped is logged."""
    geo = {**GEO, "state_slot_bytes": 1 << 30, "memory_bytes": 8 << 30}
    with pytest.raises(ValueError, match=r"are free: lower dev\.cache\.num_state_slots$"):
        size_pools(**geo, max_model_len=4096, num_state_slots=9)
    with pytest.raises(ValueError, match=r"are free: lower scheduler\.max_num_seqs or dev\.cache\.checkpoint_slots$"):
        size_pools(**geo, max_model_len=4096, checkpoint_slots=8)
    with pytest.raises(ValueError, match=r"Lower model\.max_model_len or dev\.cache\.num_state_slots, or give"):
        size_pools(**geo, max_model_len=4096, num_state_slots=6, num_pages=200)
    with pytest.raises(ValueError, match=r"Lower model\.max_model_len, scheduler\.max_num_seqs or "
                                         r"dev\.cache\.checkpoint_slots, or give"):
        size_pools(**geo, max_model_len=4096, num_pages=200)
    with caplog.at_level(logging.WARNING, logger="llm_mini_infer.cache.pools"):
        with pytest.raises(ValueError, match=r"Lower model\.max_model_len or scheduler\.max_num_seqs, or give"):
            size_pools(**geo, max_model_len=4096, num_pages=200, checkpoint_slots=3, prefix_caching=False)
        with pytest.raises(ValueError, match=r"Lower model\.max_model_len, or give"):
            size_pools(**GEO, max_model_len=4096, num_pages=200, checkpoint_slots=3)
    assert [r.message for r in caplog.records if "dropped" in r.message] == [
        "dev.cache.checkpoint_slots 3 is dropped: prefix caching off",
        "dev.cache.checkpoint_slots 3 is dropped: no recurrent state: no checkpoints"]
