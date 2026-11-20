"""The draft-count policy (`sched.draft_counts.DraftCounts`) on synthetic acceptance and device times, driven by the
scheduler's rule for a count above what the rows hold (a raise: `Scheduler._schedule`): the count with the most tokens
per device-ms, hysteresis, the deterministic probes, a probe above the count measured at that count (past a row that
sat out its raise too), the climb back when acceptance rises, the rows each count fits in a tight budget, and the
survival estimate updated only where a step verified."""

from __future__ import annotations

from collections.abc import Callable

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
CHAT = np.array([0.46, 0.13, 0.11, 0.10, 0.09])  # the share accepting at least j drafts
SCENE = np.array([0.87, 0.72, 0.65, 0.57, 0.50])


def cost_ms(rows: int, v: int, d: int) -> float:
    """A step's device time: a floor, then about linear in its verify tokens past it (the shape `profile_verify`
    measures), plus the drafter's passes."""
    return 22.0 + 0.15 * max(0, rows * (1 + v) - 120) + 0.03 * rows * d


def drive(p: DraftCounts, rows: int, surv: np.ndarray | Callable[[int], np.ndarray], steps: int,
          tokens: int | None = None, held: int = K) -> tuple[list[int], list[tuple[int, int, int]], int]:
    """`steps` plans of `rows` decode rows under the scheduler's rule: one count c for all of them; above what they
    hold (`held`: the last step's d) it verifies that and drafts k (a raise, `DraftCounts.raised`); `tokens` keeps the
    rows that fit at 1 + v tokens a row. Rows accept at least j drafts with probability surv[j - 1] (deterministic
    shares; a function of the step index: acceptance that changes), each step's time from `cost_ms`. Returns the counts
    chosen, the (rows, v, d) each step ran, and what the rows hold after."""
    chosen, ran = [], []
    for i in range(steps):
        s = surv(i) if callable(surv) else surv
        c = p.choose(rows, tokens)
        v, d = (c, c) if c <= held else (held, p.k)
        if c > held:
            p.raised()
        n = rows if tokens is None else min(rows, tokens // (1 + v))
        acc = np.zeros(n, np.int64)
        for j in range(1, v + 1):
            acc[: round(float(s[j - 1]) * n)] = j
        p.observe_accept(v, acc)
        p.observe_cost(n, v, d, int(cost_ms(n, v, d) * 1e6))
        held = d
        chosen.append(c)
        ran.append((n, v, d))
    return chosen, ran, held


def test_buckets_follow_the_decode_graphs() -> None:
    assert [row_bucket(n) for n in (1, 2, 3, 5, 9, 17, 25, 33, 48, 49, 64)] == [1, 2, 4, 8, 16, 24, 32, 48, 48, 64,
                                                                                64]


def test_low_acceptance_at_high_load_picks_a_low_count_and_high_acceptance_at_low_load_k() -> None:
    busy = drive(DraftCounts((1, 2, 3), K), 64, CHAT, 400)[0]
    assert busy[-1] == 1 and busy[-100:].count(1) > 90
    one = drive(DraftCounts((1, 2, 3), K), 1, SCENE, 400)[0]
    assert one[-1] == K and one[-100:].count(K) > 90


def test_k_runs_until_the_other_counts_are_measured_and_each_is_probed() -> None:
    p = DraftCounts((1, 2, 3), K)
    chosen = drive(p, 32, np.full(K, 0.5), EXPLORE_EVERY * MIN_STEPS * 3)[0]
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
        p.cost[(b, c, c)] = [cost_ms(64, c, c), MIN_STEPS]
    p.surv[:] = CHAT
    chosen = [p.choose(64) for _ in range(2 * REFRESH_EVERY)]
    assert chosen[REFRESH_EVERY - 1] == K and chosen[2 * REFRESH_EVERY - 1] == K
    assert chosen[REFRESH_EVERY // 2 - 1] != 1 or chosen[REFRESH_EVERY + REFRESH_EVERY // 2 - 1] != 1
    assert chosen.count(1) >= 2 * REFRESH_EVERY - 4


def test_a_probe_above_the_count_is_measured_at_that_count() -> None:
    """Settled below k, the k refresh and the probe of the count above are raises (the rows hold the bucket's count):
    the next plan runs the same count again, now held, so its time and the survival past the bucket's count are
    measured (before, the step after a raise went back to the count, and nothing above it was ever measured)."""
    p = DraftCounts((1, 2, 3), K)
    chosen, ran, _ = drive(p, 64, CHAT, 3 * REFRESH_EVERY + 10)
    assert chosen[-1] == 1
    late = range(REFRESH_EVERY, 3 * REFRESH_EVERY)  # two refreshes, each with its neighbour probe
    raises = [i for i in late if chosen[i] > ran[i][1]]
    assert raises and all(ran[i + 1] == (64, chosen[i], chosen[i]) for i in raises)  # each raise, then its count
    assert sum(1 for i in late if ran[i] == (64, K, K)) >= 2  # k verified at every refresh
    before = p.cost[(64, K, K)][1], p.surv.copy()
    drive(p, 64, SCENE, REFRESH_EVERY, held=ran[-1][2])
    assert p.cost[(64, K, K)][1] > before[0] and (p.surv[1:] > before[1][1:]).all()


def test_a_follow_survives_a_row_that_sat_out_the_raise() -> None:
    """Rows near their max_tokens sit out the plan after one they ran in (`Scheduler._schedule`'s gate): one that sat
    out a raise comes back still holding the bucket's count, so the first follow raises again; the second verifies the
    count (`MAX_FOLLOWS`). One row in 64 sits out each plan: k is verified at every refresh, the neighbour above at its
    probes (with one follow, neither ever is)."""
    p = DraftCounts((1, 2, 3), K)
    held = np.full(64, K)
    ran = []
    for i in range(3 * REFRESH_EVERY + 10):
        run = np.arange(64) != i % 64
        n = int(run.sum())
        c = p.choose(n)
        h = int(held[run].min())
        v, d = (c, c) if c <= h else (h, K)
        if c > h:
            p.raised()
        acc = np.zeros(n, np.int64)
        for j in range(1, v + 1):
            acc[: round(float(CHAT[j - 1]) * n)] = j
        p.observe_accept(v, acc)
        p.observe_cost(n, v, d, int(cost_ms(n, v, d) * 1e6))
        held[run] = d
        ran.append(v)
    late = ran[REFRESH_EVERY:]
    assert ran[-1] == 1 and late.count(K) >= 2 and late.count(2) >= 2, (late.count(K), late.count(2))


def test_a_bucket_below_k_climbs_back_when_acceptance_rises() -> None:
    """A bucket settled at a low count under low acceptance moves up when acceptance rises: its refreshes measure the
    positions past its count, and their neighbour probes the counts above (refresh shortened)."""
    low = np.array([0.2, 0.02, 0.01, 0.01, 0.01])
    p = DraftCounts((1, 2, 3), K)
    p.refresh_every = 32
    first, _, held = drive(p, 32, low, 800)
    assert first[-200:].count(3) + first[-200:].count(K) < 30  # below 3 at low acceptance
    then = drive(p, 32, SCENE, 2000, held=held)[0]
    assert then[-200:].count(3) > 160, {c: then[-200:].count(c) for c in set(then[-200:])}


def test_the_counts_compare_the_rows_each_fits_in_a_tight_budget() -> None:
    """When the step's tokens cannot hold every row at a count, that count runs the rows they hold, is measured in
    their bucket and rated by their tokens: 8 rows in 24 tokens fit at 1 and 2, 6 at 3, 4 at k. Before, the count was
    rated in the bucket of all 8 rows, which k's steps of 4 never measured, so the bucket stayed at k."""
    p = DraftCounts((1, 2, 3), K)
    chosen, ran, _ = drive(p, 8, SCENE, 400, tokens=24)
    assert all(n * (1 + v) <= 24 for n, v, _ in ran)
    assert p.measured(row_bucket(4), K) and p.measured(row_bucket(6), 3) and p.measured(row_bucket(8), 2)
    assert chosen[-100:].count(2) > 90  # 8 rows at 2: the most tokens per device-ms of the four (`cost_ms`)


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
