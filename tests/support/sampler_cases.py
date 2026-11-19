"""The sampler's fits against its target distribution (tests/runner/test_sampler.py), and the penalties' and logit
bias's tables of a step's rows (`terms_tables`: tests/kernels/test_kernels.py)."""

from __future__ import annotations

import numpy as np
import torch

from llm_mini_infer.runner.cuda.sampler import row_keys


def largest_noise_rows(by: float = 0.01) -> torch.Tensor:
    """Two rows of 4096 logits that read the largest noise's value through sampling, for seed 5738 at position 1:
    token 3980 draws it (the top uniform, clamped) and token 398 the next largest, 11.10. Every logit is 0 but token
    3980's, `by` above and then below the one at which its noise, 16.64, ties token 398's: the rows sample [3980,
    398]. The next uniform's noise, 15.94, would sample 398 in both rows, and +inf (the uniform unclamped) 3980."""
    from llm_mini_infer.runner.cuda.sampler import gumbel

    noise = gumbel(row_keys(torch.as_tensor([5738]), torch.as_tensor([1])), torch.arange(4096))[0]
    top = float(-np.log(-np.log(1 - 2.0**-24)))  # in fp64: the noise of the largest fp32 below 1
    rows = torch.zeros(2, 4096)
    rows[:, 3980] = noise.topk(2).values[1] - top + torch.tensor([by, -by])
    return rows


FIT_CASES = [  # temperature, top_p, top_k, min_p, masked: plain (under a mask; with min-p), sort, topk
    (0.7, 1.0, 0, 0.0, True), (1.0, 1.0, 0, 0.1, False), (1.3, 0.9, 0, 0.0, True), (0.9, 0.95, 12, 0.0, False)]

FIT_DRAWS = 100_000  # draws per case: one row, one seed, positions 1 to FIT_DRAWS
FIT_SEED = 7


def fit_row(v: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """The fitted row: fp32 logits [v] (normal, standard deviation 1.5) and the tokens its mask allows (all but 13:
    12 at random and the largest logit's)."""
    rng = np.random.default_rng(31)
    logits = (rng.standard_normal(v) * 1.5).astype(np.float32)
    allowed = np.ones(v, bool)
    allowed[rng.choice(v, 12, replace=False)] = False
    allowed[np.argmax(logits)] = False
    assert allowed.sum() == v - 13
    return logits, allowed


def target_distribution(logits: np.ndarray, temp: float, top_p: float, top_k: int, min_p: float,
                        allowed: np.ndarray) -> np.ndarray:
    """The distribution a row samples from (runner/cuda/sampler.py's docstring), in fp64: softmax(x), x = logits /
    temperature (fp32), over the tokens the mask allows and min-p, top-k and top-p keep; 0 elsewhere. FIT_CASES keep
    clear of the filters' boundaries, where fp32 and fp64 could keep different tokens."""
    x = np.where(allowed, (logits / np.float32(temp)).astype(np.float64), -np.inf)
    keep = allowed.copy()
    if min_p > 0:
        keep &= x >= x.max() + np.log(min_p)
    order = np.lexsort((np.arange(len(x)), -x))  # x descending, id ascending
    order = order[keep[order]]
    if top_k > 0:
        keep[order[top_k:]] = False
        order = order[:top_k]
    if top_p < 1:
        e = np.exp(x[order] - x[order[0]])
        keep[order[np.cumsum(e) - e > top_p * e.sum()]] = False
    p = np.where(keep, np.exp(x - x.max()), 0.0)
    return p / p.sum()


def assert_fits(tokens: np.ndarray, probs: np.ndarray) -> None:
    """Independent draws (one row at distinct positions) fit `probs`: none where it is 0, and Pearson's chi-square
    over the tokens it allows (one bin pools those expected fewer than 10 draws, and the next least likely until it
    expects 10, so every bin expects at least 10) below df + 10 sqrt(2 df): ten standard
    deviations above the statistic's mean, which correct draws exceed with probability below 4e-8 for the df here (10
    to 49). The seeds fix the draws, so the test cannot flake; the margin is for other seeds' equally correct draws,
    while a temperature off by 5% (either way) fails every case."""
    counts = np.bincount(tokens, minlength=len(probs))
    assert not counts[probs == 0].any(), np.flatnonzero(counts * (probs == 0))
    expected = len(tokens) * probs
    support = np.flatnonzero(probs > 0)
    support = support[np.argsort(expected[support], kind="stable")]
    e = expected[support]  # ascending
    # support[:k + 1] holds every token expected fewer than 10 draws and expects at least 10; the rest expect 10 each
    k = max(int(np.searchsorted(np.cumsum(e), 10.0)), int(np.searchsorted(e, 10.0)) - 1)
    bins = [support[:k + 1], *([t] for t in support[k + 1:])]
    obs = np.array([counts[b].sum() for b in bins], np.float64)
    exp = np.array([expected[b].sum() for b in bins])
    chi2, df = float(((obs - exp) ** 2 / exp).sum()), len(bins) - 1
    assert chi2 < df + 10 * np.sqrt(2 * df), (chi2, df)


def terms_tables(rng: np.random.Generator, n: int, vocab: int, k: int, device: torch.device) -> tuple:
    """Slots 1 .. n (slot 0: padding) in `runner.cuda.penalties`' layout, a row each: penalties, a bias (up to 1,024
    entries), both or neither (on some a repetition penalty at float32's smallest normal: the clamp); prompts and
    outputs with repeats; each row at a verify window position w (0 .. k, k > 0) with k drafts (repeats among them).
    Returns (tables: `PenaltyTables`' fields, drafts [n + 1, k], w [n])."""
    from types import SimpleNamespace

    from llm_mini_infer.runner.cuda.penalties import BIASED, TERM_BIAS, TERM_PEN
    from llm_mini_infer.sched.plan import MAX_LOGIT_BIAS

    s = n + 1
    cnt = np.zeros((s, vocab), np.uint32)  # the words' bits; viewed as int32 below
    bid = np.zeros((s, MAX_LOGIT_BIAS), np.int32)
    bval = np.zeros((s, MAX_LOGIT_BIAS), np.float32)
    terms = np.zeros(s, np.int32)
    rep, pres, freq = np.ones(s, np.float32), np.zeros(s, np.float32), np.zeros(s, np.float32)
    nb = np.zeros(s, np.int32)
    for i in range(1, s):
        kind = i % 4  # 0: none, 1: penalties, 2: a bias, 3: both
        if kind in (1, 3):
            terms[i] |= TERM_PEN
            rep[i] = [0.7, 1.3, 2.0, 1.0][(i // 4) % 4] if i % 16 != 5 else 1.1754944e-38  # the clamp's rows
            pres[i] = [0.0, 0.5, -1.5, 2.0][(i // 8) % 4]
            freq[i] = [0.3, -2.0, 1.0, 0.0][(i // 2) % 4]
            prompt = rng.integers(0, vocab, rng.integers(1, 3000))
            cnt[i, prompt] |= np.uint32(1 << 31)  # PROMPT
            outs = rng.integers(0, vocab, rng.integers(0, 400))
            outs = np.concatenate([outs, np.argsort(-rng.random(vocab))[:20].repeat(5)])  # some tokens many times
            np.add.at(cnt[i], outs, np.uint32(1))
        if kind in (2, 3):
            terms[i] |= TERM_BIAS
            m = MAX_LOGIT_BIAS if i % 8 == 2 else int(rng.integers(1, 40))
            ids = np.sort(rng.choice(vocab, m, replace=False))
            nb[i] = m
            bid[i, :m], bval[i, :m] = ids, rng.uniform(-100, 100, m).astype(np.float32)
            cnt[i, ids] |= np.uint32(BIASED)
    drafts = rng.integers(0, vocab, (s, max(k, 1)))
    drafts[:, -1] = drafts[:, 0]  # a repeated draft
    w = rng.integers(0, k + 1, n) if k else np.zeros(n, np.int64)
    t = lambda a: torch.as_tensor(a, device=device)  # noqa: E731
    tables = SimpleNamespace(terms=t(terms), rep=t(rep), pres=t(pres), freq=t(freq),
                             cnt=t(cnt.view(np.int32)), bias_n=t(nb),
                             bias_id=t(bid), bias_val=t(bval), vocab=vocab)
    return tables, t(drafts[:, :k] if k else drafts[:, :0]), w
