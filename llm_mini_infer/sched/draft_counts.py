"""The draft count of a verify step (`dev.speculative.adaptive`; architecture.md, "Speculative decoding"): which count
of the captured ones (`Capabilities.draft_counts` and k) the scheduler's next plan verifies, chosen per step from the
load and the acceptance it measured.

A count c serves a step of R decode rows with E(c) = R · (1 + s_1 + ... + s_c) tokens, where s_j is the share of
verify rows that accept at least j drafts (the batch's survival to draft j), at a device time C(R, c). Above the
step's memory floor C grows with the step's tokens, so the count that delivers the most tokens per device-ms falls as
the rows grow and as acceptance falls; a step's prefill tokens cost the same at every count, so the decode part's
time is the one to compare. When the step's token budget cannot hold every row at a count, that count runs the rows
it holds (R_c), and its rate is theirs. `DraftCounts` keeps:
- the survival: per position, the verify rows that held it and those that accepted it, both decayed at every verify
  step, over an optimistic prior (`PRIOR_ROWS` rows accepting every draft) and never rising with depth. A step at c
  says nothing of the drafts past c, so their evidence ages while the count runs lower and their estimate drifts back
  to the prior, until a step verifies them again: a run of rejections cannot leave a position's estimate low for good
  (at one decode row a step's share is 0 or 1, and acceptance comes in runs);
- the time: per decode row bucket (`row_bucket`), and per (v, d) a step ran at, an EMA of the device time of the steps
  whose rows were all decode rows (`observe_cost`; other steps cost what their prefill tokens cost);
- per bucket (of the rows at the narrowest count) the count it runs (k at first) and its steps.
Each choice is a function of those alone (`choose`): a count without two measured steps in its rows' bucket is run one
step, every `EXPLORE_EVERY` of the bucket's steps, nearest first; every `REFRESH_EVERY` steps the bucket runs k once
(the positions its count does not verify) and one neighbour of its count (their times); otherwise it switches to the
count with the most tokens per device-ms only when that beats its own by `HYSTERESIS`. A count above what the rows
hold makes a raise (the scheduler verifies what they hold and drafts k, `raised`), so the bucket's next plan runs that
count again: it is measured there (a row that sat out the raise still holds less, and makes the follow a raise too:
then the plan after it, `MAX_FOLLOWS`). Times are measured, so the schedule is not reproducible from run to run; no
output token depends on it (architecture.md, "Speculative decoding").
"""

from __future__ import annotations

import numpy as np

SURVIVAL_ALPHA = 0.02  # the survival counts' decay at each verify step
PRIOR_ROWS = 2.0  # the survival's prior: as many rows accepting every draft (optimistic, as k runs first)
COST_ALPHA = 0.1  # the time EMA's weight of one step
HYSTERESIS = 0.03  # a switch must gain this share of tokens per device-ms
EXPLORE_EVERY = 8  # a bucket's unmeasured count runs one step in every this many of the bucket's steps
REFRESH_EVERY = 256  # k and a neighbour of the bucket's count run one step each in every this many
MIN_STEPS = 2  # measured steps before a (bucket, count)'s time is trusted
# Plans that run a raised count again (`raised`): a row near its max_tokens sits out the plan after one it ran in, so
# one that sat out the raise holds less at the first follow; by the second every row has drafted k
MAX_FOLLOWS = 2


def row_bucket(rows: int) -> int:
    """The decode row bucket a step of `rows` decode rows runs in (the runners' decode graphs: 1, 2, 4, 8, 16, 24, 32,
    then every 16)."""
    for b in (1, 2, 4, 8, 16, 24, 32):
        if rows <= b:
            return b
    return -(-rows // 16) * 16


class DraftCounts:
    """The policy of the module docstring for the counts `counts` (k among them) of a drafter at most k deep."""

    def __init__(self, counts: tuple[int, ...] | list[int], k: int):
        self.k = k
        self.counts = tuple(sorted({int(c) for c in counts if 1 <= int(c) <= k} | {k}))
        self.surv = np.ones(k, np.float64)  # s_1 .. s_k (optimistic until measured: k runs first)
        self.held_rows = np.zeros(k, np.float64)  # per position: the verify rows that held it, decayed
        self.accepted_rows = np.zeros(k, np.float64)  # ... and those that accepted it
        self.cost: dict[tuple[int, int, int], list[float]] = {}  # (bucket, v, d) -> [device ms, steps]
        self.steps: dict[int, int] = {}  # bucket -> decode steps chosen
        self.current: dict[int, int] = {}  # bucket -> the count it runs
        # bucket -> (the count its next plan runs again, the follows of that raise so far: `raised`)
        self.follow: dict[int, tuple[int, int]] = {}
        self.chosen = dict.fromkeys(self.counts, 0)  # steps planned at each count (`choose`)
        self._last: tuple[int, int, int] | None = None  # the last choice: (bucket, count, the follows it ends)
        # the probe schedule and the trust in a time (the module's constants; tests shorten them)
        self.explore_every, self.refresh_every, self.min_steps = EXPLORE_EVERY, REFRESH_EVERY, MIN_STEPS

    def measured(self, bucket: int, c: int) -> bool:
        e = self.cost.get((bucket, c, c))
        return e is not None and e[1] >= self.min_steps

    def rate(self, rows: int, c: int) -> float:
        """Tokens per device-ms of a step of `rows` decode rows at count c (its time measured in their bucket)."""
        return rows * (1.0 + float(self.surv[:c].sum())) / self.cost[(row_bucket(rows), c, c)][0]

    def choose(self, rows: int, tokens: int | None = None) -> int:
        """The count for a plan with `rows` decode rows, of which a count c runs as many as `tokens` holds at 1 + c
        tokens a row (None: all of them at any count; module docstring)."""
        at = {x: rows if tokens is None else min(rows, tokens // (1 + x)) for x in self.counts}
        b = row_bucket(max(1, at[self.counts[0]]))
        n = self.steps.get(b, 0)
        self.steps[b] = n + 1
        cur = self.current.get(b, self.k)
        f = self.follow.pop(b, None)
        c, depth = f if f is not None and at[f[0]] > 0 else (None, 0)
        if c is None:
            c = self._probe(b, n, cur, at)
        if c is None:
            rates = {x: self.rate(at[x], x) for x in self.counts if at[x] and self.measured(row_bucket(at[x]), x)}
            if rates:
                best = max(rates, key=lambda x: (rates[x], x))
                if cur not in rates or rates[best] > rates[cur] * (1.0 + HYSTERESIS):
                    cur = best
            self.current[b] = c = cur
        self.chosen[c] += 1
        self._last = (b, c, depth)
        return c

    def raised(self) -> None:
        """The last plan chosen (`choose`) raised its count: its rows held fewer drafts, so it verified those and
        drafted k. The bucket's next plan runs that count again, now held, and so measures it (a follow that raises
        again is followed once more: `MAX_FOLLOWS`)."""
        if self._last is not None and self._last[2] < MAX_FOLLOWS:
            b, c, depth = self._last
            self.follow[b] = (c, depth + 1)

    def _probe(self, b: int, n: int, cur: int, at: dict[int, int]) -> int | None:
        """A count to run once for its measurements, or None (module docstring)."""
        every, refresh = self.explore_every, self.refresh_every
        if n % every == every - 1:
            todo = [x for x in self.counts if at[x] and not self.measured(row_bucket(at[x]), x)]
            if todo:
                return min(todo, key=lambda x: (abs(x - cur), x))
        phase = n % refresh
        if phase == refresh - 1 and cur != self.k and at[self.k]:
            return self.k
        if phase == refresh // 2 - 1:
            i = self.counts.index(cur)
            side = (n // refresh) % 2  # alternately the count below and the one above
            j = i - 1 if (side == 0 and i > 0) or i == len(self.counts) - 1 else i + 1
            if 0 <= j < len(self.counts) and j != i and at[self.counts[j]]:
                return self.counts[j]
        return None

    def observe_accept(self, v: int, accepted: np.ndarray) -> None:
        """A verify step of v drafts a row: each row's accepted drafts (`accepted`, ints). Every position's counts
        decay; the first v gain this step's rows (module docstring)."""
        if v <= 0 or not len(accepted):
            return
        a = np.asarray(accepted, np.int64)
        self.held_rows *= 1.0 - SURVIVAL_ALPHA
        self.accepted_rows *= 1.0 - SURVIVAL_ALPHA
        self.held_rows[:v] += len(a)
        self.accepted_rows[:v] += (a[:, None] >= np.arange(1, v + 1)[None, :]).sum(axis=0)
        self.surv = np.minimum.accumulate((self.accepted_rows + PRIOR_ROWS) / (self.held_rows + PRIOR_ROWS))

    def observe_cost(self, rows: int, v: int, d: int, busy_ns: int) -> None:
        """The device time of a step whose `rows` rows were all decode rows, run at (v, d)."""
        if busy_ns <= 0:
            return
        key = (row_bucket(rows), v, d)
        ms = busy_ns / 1e6
        e = self.cost.get(key)
        if e is None:
            self.cost[key] = [ms, 1]
        else:
            e[0] += COST_ALPHA * (ms - e[0])
            e[1] += 1
