"""Exact tokenization of prompts with long repeated parts (a shared system prompt, earlier turns of a chat).

HF `tokenizers` first cuts the input at added tokens (an Aho-Corasick, leftmost-longest match over the raw text for
tokens with `normalized=False`) and then normalizes, pre-tokenizes and BPE-encodes each piece on its own. So
cutting the text at one such token ourselves and encoding the pieces separately gives the same IDs, if the
token's matches cannot be pre-empted or stretched by the tokenizer:
- `normalized`, `lstrip`, `rstrip` and `single_word` are all false (the match is exactly the token's text);
- no other added token contains it, or ends with one of its proper prefixes (a match starting earlier and
  overlapping it would win under leftmost-longest).
Long pieces are cached by their text; runs of short pieces are joined back and encoded in one call, so a long
shared system prompt is tokenized once and then looked up. tests/api/test_render.py checks exactness on the
chat-template corpus (`tests/scripts/template_corpus.py`) for all five fixture models.
"""

from __future__ import annotations

import logging
from collections import Counter

import numpy as np
from tokenizers import Tokenizer

log = logging.getLogger(__name__)

_EMPTY = np.zeros(0, np.int32)


def pick_split_token(tokenizer: Tokenizer, probe: str | None) -> tuple[str, int] | None:
    """The safe added token occurring most often in `probe` (a rendered sample conversation, or the template's
    source: the message delimiter wins), or None."""
    if not probe:
        return None
    added = dict(tokenizer.get_added_tokens_decoder().items())
    contents = [t.content for t in added.values()]
    counts: Counter[str] = Counter()
    for tok in added.values():
        c = tok.content
        if tok.normalized or tok.lstrip or tok.rstrip or tok.single_word or len(c) < 2:
            continue
        n = probe.count(c)  # first: the scan below reads every added token, so it runs only for those in the probe
        if not n:
            continue
        if any(o != c and (c in o or any(o.endswith(c[:k]) for k in range(1, len(c)))) for o in contents):
            continue
        counts[c] = n
    if not counts:
        return None
    best = max(counts, key=lambda c: (counts[c], len(c)))
    return best, next(i for i, t in added.items() if t.content == best)


class SegmentCache:
    def __init__(self, tokenizer: Tokenizer, split: str, split_id: int, min_chars: int = 512,
                 max_tokens: int = 2_000_000):
        self.tok = tokenizer
        self.split = split
        self.sep = np.asarray([split_id], np.int32)
        self.min_chars = min_chars
        self.max_tokens = max_tokens
        self.cached_tokens = 0
        self._cache: dict[str, np.ndarray] = {}
        self.hits = self.misses = 0

    def _encode(self, text: str) -> np.ndarray:
        if not text:
            return _EMPTY
        return np.asarray(self.tok.encode_batch_fast([text], add_special_tokens=False)[0].ids, np.int32)

    def _long(self, part: str) -> np.ndarray:
        cache = self._cache
        ids = cache.pop(part, None)
        if ids is not None:
            cache[part] = ids  # most recently used
            self.hits += 1
            return ids
        self.misses += 1
        ids = self._encode(part)
        cache[part] = ids
        self.cached_tokens += len(ids)
        while self.cached_tokens > self.max_tokens and cache:
            old = next(iter(cache))
            self.cached_tokens -= len(cache.pop(old))
        return ids

    def encode(self, text: str) -> np.ndarray:
        parts = text.split(self.split)
        if len(parts) == 1 or max(map(len, parts)) < self.min_chars:
            return self._encode(text)
        out: list[np.ndarray] = []
        run: list[str] = []
        for part in parts:
            if len(part) >= self.min_chars:
                if run:
                    out.append(self._encode(self.split.join(run)))
                    out.append(self.sep)
                    run = []
                out.append(self._long(part))
                out.append(self.sep)
            else:
                run.append(part)
        if run:
            out.append(self._encode(self.split.join(run)))
        else:
            out.pop()  # the separator after the last long part
        return np.concatenate(out)
