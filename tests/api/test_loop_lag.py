"""The frontend's loop-lag sampler (`api.app.LoopLag`, /debug/loop): a stall of the loop shows as one late sample."""

from __future__ import annotations

import asyncio
import time

import uvloop

from llm_mini_infer.api.app import LAG_INTERVAL_NS, LoopLag


def test_a_stall_of_the_loop_shows_in_full_and_the_window_and_reset_hold() -> None:
    async def run() -> tuple[dict, dict, int]:
        lag = LoopLag(size=1000)
        lag.start()
        await asyncio.sleep(0.02)
        mid = time.monotonic_ns()
        time.sleep(0.03)  # the loop held for 30 ms
        await asyncio.sleep(0.02)
        lag.stop()
        n = len(lag.samples()["t"])
        after = lag.samples(since=mid)
        lag.reset()
        return after, lag.samples(), n

    after, cleared, n = uvloop.run(run())
    assert n >= 15 and after["interval_ns"] == LAG_INTERVAL_NS and after["lost"] == 0
    assert all(t >= 0 for t in after["t"]) and after["t"] == sorted(after["t"])
    assert 29e6 <= max(after["lag"]) < 60e6  # the stall, not more than it plus a timer's resolution
    assert cleared["t"] == [] and cleared["lag"] == []


def test_the_ring_keeps_the_last_samples_oldest_first() -> None:
    async def run() -> dict:
        lag = LoopLag(size=8)
        lag.start()
        await asyncio.sleep(0.03)
        lag.stop()
        return lag.samples()

    got = uvloop.run(run())
    assert len(got["t"]) == 8 and got["lost"] > 0 and got["t"] == sorted(got["t"])
