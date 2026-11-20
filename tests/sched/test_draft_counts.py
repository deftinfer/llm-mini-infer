"""The draft-count policy (`sched.draft_counts.DraftCounts`) on synthetic acceptance and device times: the count with
the most tokens per device-ms, hysteresis, the deterministic probes, and the survival estimate updated only where a
step verified."""

from __future__ import annotations

import numpy as np

from llm_mini_infer.sched.draft_counts import (
    EXPLORE_EVERY,
    HYSTERESIS,
    MIN_STEPS,
    REFRESH_EVERY,
    DraftCounts,
    row_bucket,
)

K = 5


def cost_ms(rows: int, c: int) -> float:
    """A step's device time: a floor, then about linear in its tokens past it (the shape `profile_verify` measures)."""
    return 22.0 + 0.15 * max(0, rows * (1 + c) - 120) + 0.03 * rows * c


def drive(p: DraftCounts, rows: int, surv: np.ndarray, steps: int) -> list[int]:
    """`steps` plans of `rows` decode rows against rows that accept at least j drafts with probability surv[j - 1]
    (deterministic shares), each step's time from `cost_ms`; returns the counts chosen."""
    out = []
    for _ in range(steps):
        c = p.choose(rows)
        out.append(c)
        acc = np.zeros(rows, np.int64)
        for j in range(1, c + 1):
            acc[: round(float(surv[j - 1]) * rows)] = j
        p.observe_accept(c, acc)
        p.observe_cost(rows, c, c, int(cost_ms(rows, c) * 1e6))
    return out


def test_buckets_follow_the_decode_graphs() -> None:
    assert [row_bucket(n) for n in (1, 2, 3, 5, 9, 17, 25, 33, 48, 49, 64)] == [1, 2, 4, 8, 16, 24, 32, 48, 48, 64,
                                                                                64]


def test_low_acceptance_at_high_load_picks_a_low_count_and_high_acceptance_at_low_load_k() -> None:
    chat = np.array([0.46, 0.13, 0.11, 0.10, 0.09])  # the share accepting at least j drafts
    scene = np.array([0.87, 0.72, 0.65, 0.57, 0.50])
    busy = drive(DraftCounts((1, 2, 3), K), 64, chat, 400)
    assert busy[-1] == 1 and busy[-100:].count(1) > 90
    one = drive(DraftCounts((1, 2, 3), K), 1, scene, 400)
    assert one[-1] == K and one[-100:].count(K) > 90


def test_k_runs_until_the_other_counts_are_measured_and_each_is_probed() -> None:
    p = DraftCounts((1, 2, 3), K)
    chosen = drive(p, 32, np.full(K, 0.5), EXPLORE_EVERY * MIN_STEPS * 3)
    assert chosen[0] == K and set(chosen) == {1, 2, 3, K}
    assert all(p.measured(row_bucket(32), c) for c in (1, 2, 3, K))
    probes = [i for i, c in enumerate(chosen) if c != K]
    assert all(i % EXPLORE_EVERY == EXPLORE_EVERY - 1 for i in probes)  # one step in every EXPLORE_EVERY


def test_hysteresis_keeps_a_count_whose_rival_gains_too_little() -> None:
    p = DraftCounts((3,), K)
    b = row_bucket(16)
    p.surv[:] = 0.5
    p.cost[(b, K, K)] = [10.0, MIN_STEPS]
    # count 3's rate is a hair above 5's: within the hysteresis, k stays
    r5 = 16 * (1 + 0.5 * K) / 10.0
    p.cost[(b, 3, 3)] = [16 * (1 + 0.5 * 3) / (r5 * (1 + HYSTERESIS / 2)), MIN_STEPS]
    p.steps[b] = 1  # past the first probe phase
    assert p.choose(16) == K
    p.cost[(b, 3, 3)] = [16 * (1 + 0.5 * 3) / (r5 * (1 + 2 * HYSTERESIS)), MIN_STEPS]
    assert p.choose(16) == 3


def test_the_refresh_runs_k_and_a_neighbour_on_its_schedule() -> None:
    p = DraftCounts((1, 2, 3), K)
    b = row_bucket(64)
    for c in (1, 2, 3, K):
        p.cost[(b, c, c)] = [cost_ms(64, c), MIN_STEPS]
    p.surv[:] = [0.46, 0.13, 0.11, 0.10, 0.09]
    chosen = [p.choose(64) for _ in range(2 * REFRESH_EVERY)]
    assert chosen[REFRESH_EVERY - 1] == K and chosen[2 * REFRESH_EVERY - 1] == K
    assert chosen[REFRESH_EVERY // 2 - 1] != 1 or chosen[REFRESH_EVERY + REFRESH_EVERY // 2 - 1] != 1
    assert chosen.count(1) >= 2 * REFRESH_EVERY - 4


def test_survival_moves_only_where_a_step_verified() -> None:
    p = DraftCounts((1, 2), K)
    before = p.surv.copy()
    p.observe_accept(2, np.array([0, 0, 1, 2]))
    assert (p.surv[2:] == before[2:]).all()  # positions 3 to 5: not verified
    assert p.surv[0] < 1.0 and p.surv[1] < 1.0 and p.surv[1] < p.surv[0]
    p.observe_accept(0, np.array([]))  # nothing verified: nothing moves
    p.observe_cost(4, 2, 2, 0)  # no device time: nothing recorded
    assert not p.cost


def test_the_choice_is_a_function_of_its_inputs() -> None:
    a, b = DraftCounts((1, 2, 3), K), DraftCounts((1, 2, 3), K)
    surv = np.array([0.6, 0.4, 0.3, 0.2, 0.1])
    assert drive(a, 24, surv, 300) == drive(b, 24, surv, 300)
