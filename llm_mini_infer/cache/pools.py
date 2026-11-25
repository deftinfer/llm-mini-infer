"""KV page pool, recurrent-state slot pool, request slots, window checkpoints (`WindowStore`) and the pools' sizing
(`size_pools`, `size_arena`; docs/architecture.md, "Pools").

The pools are independent: page size never depends on the state size. IDs are owned by the engine; the runner only
stores what plans tell it. Index 0 of each pool is the null entry (a padding target that is never handed out).

**Pages** are ref-counted, and the refcount is the single source of truth: running requests hold one reference for each
entry of their page tables, the radix tree (`cache.radix`) one for each page a node keeps, and each window checkpoint
one for each of its pages. A page is free exactly when nobody references it. Cached pages are reclaimed by evicting
tree nodes or windows (LRU), not by the pool.

**State slots** are one physical pool split into two kinds (`StatePool`): *live* slots, one per running request, and
*checkpoint* slots, owned by tree nodes. Running requests never compete with checkpoints for a slot (concurrency
first). A request that leaves (a finish, an abort, a preemption) can donate its live slot as a checkpoint: the slot
changes kind and a free checkpoint slot takes its place among the live ones, so the counts never change.

**The KV arena** (sliding-window models; `size_arena`): one `PagePool` of equal pages holds both kinds of KV page,
each page one kind from its allocation to its free (architecture.md, "Sliding-window models").
"""

from __future__ import annotations

import logging

import msgspec
import numpy as np

log = logging.getLogger(__name__)

# Share of the memory left after the live state slots that the derived checkpoint budget may take.
CHECKPOINT_MEMORY_SHARE = 0.25


class PagePool:
    """Fixed-size KV pages with refcounts."""

    def __init__(self, num_pages: int):
        if num_pages < 2:
            raise ValueError("the KV pool needs at least one page besides the null page")
        self.num_pages = num_pages
        self.refcount = np.zeros(num_pages, np.int32)
        self.refcount[0] = 1  # the null page is never free
        self._free = list(range(num_pages - 1, 0, -1))  # a stack: low IDs first

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def usage(self) -> float:
        return 1.0 - len(self._free) / (self.num_pages - 1)

    def alloc(self, n: int) -> list[int] | None:
        """n pages with refcount 1, or None (nothing allocated) when fewer are free."""
        if n > len(self._free):
            return None
        if n == 0:
            return []
        pages = self._free[-n:]
        del self._free[-n:]
        pages.reverse()
        self.refcount[pages] = 1
        return pages

    def incref(self, pages: list[int]) -> None:
        if len(pages) == 1 and pages[0] > 0 and self.refcount[pages[0]] > 0:  # one page (a node's tail): no array
            self.refcount[pages[0]] += 1
        elif pages:
            arr = np.asarray(pages, np.int64)
            cur = self.refcount[arr]
            if cur.min() <= 0 or not arr.all():
                raise RuntimeError(f"incref of a free or null page: {arr[(cur <= 0) | (arr == 0)].tolist()}")
            self.refcount[arr] = cur + 1

    def release(self, pages: list[int]) -> int:
        """Drop one reference from each page (no duplicates in `pages`); pages reaching zero go back to the free
        list. Returns how many were freed."""
        if not pages:
            return 0
        if len(pages) == 1 and pages[0] > 0 and self.refcount[pages[0]] > 0:  # one page (a node's tail): no array
            p = int(pages[0])
            self.refcount[p] -= 1
            if self.refcount[p]:
                return 0
            self._free.append(p)
            return 1
        arr = np.asarray(pages, np.int64)
        left = self.refcount[arr] - 1
        if left.min() < 0 or not arr.all():
            raise RuntimeError(f"release of the null page or of pages released more often than allocated: "
                               f"{arr[(left < 0) | (arr == 0)].tolist()}")
        self.refcount[arr] = left
        freed = arr[left == 0]
        if len(freed):
            self._free.extend(freed.tolist())
        return len(freed)


class SlotPool:
    """Request slots: IDs 1..n-1."""

    def __init__(self, n: int, what: str):
        self.n = n
        self.what = what
        self._free = list(range(n - 1, 0, -1))
        self._used = np.zeros(n, bool)

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def usage(self) -> float:
        return 1.0 - len(self._free) / max(1, self.n - 1)

    def alloc(self) -> int | None:
        if not self._free:
            return None
        i = self._free.pop()
        self._used[i] = True
        return i

    def release(self, i: int) -> None:
        if not self._used[i]:
            raise RuntimeError(f"{self.what} {i} released twice")
        self._used[i] = False
        self._free.append(i)


LIVE, CHECKPOINT = 1, 2


class StatePool:
    """Recurrent-state slots 1..n-1: `n_live` live slots (running requests) and the rest checkpoint slots (radix
    tree). `kind[i]` is a slot's current role; donation swaps the roles of two slots."""

    def __init__(self, n: int, n_live: int):
        if n_live < 0 or n_live > n - 1:
            raise ValueError(f"{n_live} live slots do not fit in a pool of {n - 1} state slots")
        self.n = n
        self.n_live = n_live
        self.n_ckpt = n - 1 - n_live
        self.kind = np.zeros(n, np.uint8)
        self.kind[1:1 + n_live] = LIVE
        self.kind[1 + n_live:] = CHECKPOINT
        self.used = np.zeros(n, bool)
        self._live = list(range(n_live, 0, -1))
        self._ckpt = list(range(n - 1, n_live, -1))

    @property
    def num_free_live(self) -> int:
        return len(self._live)

    @property
    def num_free_ckpt(self) -> int:
        return len(self._ckpt)

    @property
    def live_usage(self) -> float:
        return 1.0 - len(self._live) / max(1, self.n_live)

    @property
    def ckpt_usage(self) -> float:
        return 1.0 - len(self._ckpt) / max(1, self.n_ckpt)

    def alloc_live(self) -> int | None:
        return self._take(self._live)

    def alloc_ckpt(self) -> int | None:
        return self._take(self._ckpt)

    def _take(self, free: list[int]) -> int | None:
        if not free:
            return None
        i = free.pop()
        self.used[i] = True
        return i

    def release(self, i: int) -> None:
        if not self.used[i]:
            raise RuntimeError(f"state slot {i} released twice")
        self.used[i] = False
        (self._live if self.kind[i] == LIVE else self._ckpt).append(i)

    def donate(self, live: int) -> bool:
        """Turn the (used) live slot `live` into a used checkpoint slot, and a free checkpoint slot into a free live
        slot. False (nothing changed) when no checkpoint slot is free."""
        if not self._ckpt:
            return False
        if self.kind[live] != LIVE or not self.used[live]:
            raise RuntimeError(f"state slot {live} is not a live slot in use")
        c = self._ckpt.pop()
        self.kind[live], self.kind[c] = CHECKPOINT, LIVE
        self._live.append(c)
        return True

    def is_checkpoint(self, i: int) -> bool:
        """Whether `i` is an allocated checkpoint slot (the radix tree's checkpoint handles)."""
        return bool(self.kind[i] == CHECKPOINT and self.used[i])


class WindowStore:
    """Window checkpoints (architecture.md, "Sliding-window models"): the radix tree's checkpoint handles in window
    mode. A window is the set of the arena's window pages that hold the sliding layers' KV of the positions
    [c − W + 1, c) before a resume point c, page by page from page index `lo`: all a request resuming at c needs of the
    sliding layers (the full layers' KV comes from the tree's pages). Making one copies nothing: it takes one reference
    on each page, so overlapping windows and the running requests that hold the same pages share them."""

    def __init__(self, pool: PagePool):
        self.pool = pool
        self._win: dict[int, tuple[int, list[int]]] = {}  # handle -> (first page index, pages)
        self._next = 1
        self.refs = np.zeros(pool.num_pages, np.int32)  # references the windows hold (accounting)

    @property
    def num_windows(self) -> int:
        return len(self._win)

    def make(self, lo: int, pages: list[int]) -> int:
        """A window over page indices [lo, lo + len(pages)) holding `pages` (distinct, non-null; none for a window of
        one key at a page boundary: such a resume needs no earlier position). Returns its handle."""
        if 0 in pages or len(set(pages)) != len(pages):
            raise RuntimeError(f"a window needs distinct non-null pages, got {pages}")
        if pages:
            self.pool.incref(pages)
            self.refs[pages] += 1
        h = self._next
        self._next += 1
        self._win[h] = (lo, list(pages))
        return h

    def get(self, h: int) -> tuple[int, list[int]]:
        return self._win[h]

    def release(self, h: int) -> int:
        """Drop a window's references. Returns the pages that became free."""
        _, pages = self._win.pop(h)
        if not pages:
            return 0
        self.refs[pages] -= 1
        return self.pool.release(pages)

    def is_checkpoint(self, h: int) -> bool:
        return h in self._win

    def freed_by(self, h: int) -> int:
        """The pages releasing window `h` would free: those no other window or request references."""
        _, pages = self._win[h]
        return int((self.pool.refcount[pages] == 1).sum()) if pages else 0


class PoolSizes(msgspec.Struct, frozen=True):
    num_pages: int
    num_state_slots: int  # slot 0 included: 1 + live + checkpoint
    num_live_slots: int
    num_checkpoint_slots: int
    reasons: dict[str, str]
    max_model_len: int  # as given, or capped to what the KV pool holds (`cap_max_model_len`)


def pages_for_one_request(max_model_len: int, page_size: int, reserve_tokens: int, prefix_caching: bool) -> int:
    """KV pages one request of `max_model_len` may hold: its longest prompt (max_model_len − 1 tokens, the scheduler's
    admission bound) plus the positions reserved for the steps in flight (`plan.reserve_positions`), and with prefix
    caching one more for the cached page a hit copies (a partial one, or under MTP the full one holding the drafter's
    KV the hit recomputes), which the tree keeps while the request runs."""
    return -(-(max_model_len - 1 + reserve_tokens) // page_size) + (1 if prefix_caching else 0)


def window_pages_for_one_request(window: int, max_num_batched_tokens: int, page_size: int, reserve_tokens: int) -> int:
    """Window pages one request may hold at once (sliding-window models): a chunk's first query sees
    window − 1 positions before it, so a chunk of up to `max_num_batched_tokens` (or a decode row and its reserved
    positions) spans window − 1 + that many positions, one more page for their misalignment, and one for the partial
    last page of the window a hit copies (the window keeps its own)."""
    return -(-(window - 1 + max(max_num_batched_tokens, reserve_tokens)) // page_size) + 2


REFERENCE_LEN = 4096  # a common max_model_len: the length concurrency is logged for


def arena_full_page_size(window_page_size: int, kv_bytes_per_token: int, window_kv_bytes_per_token: int) -> int:
    """The full layers' page size in a sliding-window model's arena: the positions of the full layers'
    KV that fill exactly the bytes of one window page (`window_page_size` positions of every sliding layer's KV), so
    one page of the arena holds either kind. Gemma 4 31B: a window page of 16 positions is 12.5 MiB, and so are 320
    positions of its full layers (V only, 40 KiB per position)."""
    page = window_page_size * window_kv_bytes_per_token
    if kv_bytes_per_token <= 0 or page % kv_bytes_per_token:
        raise ValueError(f"a window page of {window_page_size} positions ({page:,} bytes) is not a whole number of the "
                         f"full layers' positions ({kv_bytes_per_token:,} bytes each): one arena cannot hold both")
    return page // kv_bytes_per_token


class ArenaSizes(msgspec.Struct, frozen=True):
    num_pages: int  # the arena's pages, page 0 (the null page) included
    page_size: int  # positions of a full page (the radix tree's page: `arena_full_page_size`)
    reasons: dict[str, str]
    max_model_len: int


def size_arena(*, memory_bytes: int, kv_bytes_per_token: int, window_kv_bytes_per_token: int, window: int,
               window_page_size: int, max_num_seqs: int, max_model_len: int, max_num_batched_tokens: int,
               reserve_tokens: int = 2, num_pages: int | None = None, prefix_caching: bool = True,
               cap_max_model_len: bool = False, reference_len: int | None = None) -> ArenaSizes:
    """The KV arena of a sliding-window model (architecture.md, "Sliding-window models"): one pool of equal pages, each
    a window page (`window_page_size` positions of the sliding layers' KV) or a full page (`arena_full_page_size`
    positions of the full layers' KV). A decoding request holds about `window / window_page_size` window pages whatever
    its length, plus its length in full pages; the cache's windows and full pages use what is left.

    All the memory goes to the arena (or `num_pages` is set). It must hold one request of max_model_len: its full pages
    (`pages_for_one_request`) and its peak of window pages (`window_pages_for_one_request`); else this raises, or caps a
    derived max_model_len, as `size_pools` does. The concurrency the arena supports is logged at `reference_len`
    (default `REFERENCE_LEN`) and at max_model_len."""
    reasons: dict[str, str] = {}
    fps = arena_full_page_size(window_page_size, kv_bytes_per_token, window_kv_bytes_per_token)
    page_bytes = window_page_size * window_kv_bytes_per_token
    reasons["page_size"] = (f"{fps} positions of the full layers fill one window page of {window_page_size} positions "
                            f"({page_bytes / 2**20:.2f} MiB): one arena holds both kinds")
    if num_pages is None:
        num_pages = int(memory_bytes // page_bytes)
        reasons["num_pages"] = (f"{memory_bytes / 2**30:.2f} GiB free / {page_bytes / 2**20:.2f} MiB per page: one "
                                "arena for the full and the window pages, split by the load")
    else:
        reasons["num_pages"] = "configured"
    w_peak = window_pages_for_one_request(window, max_num_batched_tokens, window_page_size, reserve_tokens)
    need = pages_for_one_request(max_model_len, fps, reserve_tokens, prefix_caching) + w_peak
    if num_pages - 1 < need:
        extra = 1 if prefix_caching else 0
        fits = (num_pages - 1 - w_peak - extra) * fps - reserve_tokens + 1
        why = (f"the KV arena holds {num_pages - 1:,} pages, but one request of max_model_len {max_model_len:,} needs "
               f"{need:,} ({need - w_peak} full pages of {fps} positions and {w_peak} window pages)")
        if not cap_max_model_len or fits < 2:
            raise ValueError(f"{why}: the largest max_model_len that fits is {max(fits, 0):,}. Lower "
                             "model.max_model_len or scheduler.max_num_batched_tokens, or give the arena more memory")
        log.warning("%s: capping the derived max_model_len to %s", why, f"{fits:,}")
        reasons["max_model_len"] = f"capped to {fits:,}, the longest request the KV arena holds"
        max_model_len = fits

    def per_request(n: int) -> int:  # pages a decoding request of n positions holds
        return -(-(n + reserve_tokens) // fps) + -(-(min(n, window - 1) + reserve_tokens) // window_page_size) + 1

    ref = min(reference_len or REFERENCE_LEN, max_model_len)
    usable = num_pages - 1
    log.info("KV arena: %d pages of %.2f MiB (%.1f GiB), each a window page of %d positions or a full page of %d "
             "positions (window %d); about %d requests of %s tokens at once (%d at max_model_len %s) without shared "
             "prefixes", num_pages, page_bytes / 2**20, num_pages * page_bytes / 2**30, window_page_size, fps, window,
             usable // per_request(ref), f"{ref:,}", usable // per_request(max_model_len), f"{max_model_len:,}")
    if usable // per_request(ref) < max_num_seqs:
        log.warning("the KV arena holds about %d requests of %s tokens, fewer than max_num_seqs (%d): with longer or "
                    "unshared prompts, admissions wait for pages", usable // per_request(ref), f"{ref:,}",
                    max_num_seqs)
    return ArenaSizes(num_pages=int(num_pages), page_size=int(fps), reasons=reasons, max_model_len=int(max_model_len))


def _either(names: list[str]) -> str:
    return " or ".join(names) if len(names) < 3 else ", ".join(names[:-1]) + " or " + names[-1]


def size_pools(*, memory_bytes: int, kv_bytes_per_token: int, state_slot_bytes: int, page_size: int,
               max_num_seqs: int, max_model_len: int, num_pages: int | None = None,
               num_state_slots: int | None = None, checkpoint_slots: int | None = None,
               prefix_caching: bool = True, reserve_tokens: int = 2, cap_max_model_len: bool = False) -> PoolSizes:
    """The pools in this order (architecture.md, "Pools"): the live state slots from the requested concurrency (one per
    running request), then the checkpoint budget, then KV gets the rest. Warns when the KV pool then caps concurrency
    below `max_num_seqs` at `max_model_len`.

    The checkpoint budget defaults to `max_num_seqs` slots (one cached prefix per running request), capped at
    `CHECKPOINT_MEMORY_SHARE` of the memory left after the live slots. It is 0 without recurrent state (attention-only
    models checkpoint nothing) and without prefix caching.

    The KV pool must hold one request of `max_model_len` (`pages_for_one_request`): a request that outgrew
    the whole pool would preempt itself forever. If it does not, this raises, naming the largest length that fits;
    with `cap_max_model_len` (max_model_len was derived, not set) it caps max_model_len to that length instead and
    logs why.
    """
    reasons: dict[str, str] = {}
    kv_page_bytes = kv_bytes_per_token * page_size
    live = max_num_seqs
    if num_state_slots is not None:
        if num_state_slots < live + 1:
            raise ValueError(f"dev.cache.num_state_slots {num_state_slots} < max_num_seqs {max_num_seqs} + 1")
        ckpt = num_state_slots - 1 - live
        reasons["num_state_slots"] = f"configured: {live} live + {ckpt} checkpoint slots + null"
        knobs = ["dev.cache.num_state_slots"]  # what sizes the state pool, for the errors below
    else:
        knobs = ["scheduler.max_num_seqs", "dev.cache.checkpoint_slots"] if state_slot_bytes else []
        if not state_slot_bytes or not prefix_caching:
            ckpt = 0
            why = "no recurrent state: no checkpoints" if not state_slot_bytes else "prefix caching off"
            knobs = knobs[:1]
            if checkpoint_slots is not None:
                log.warning("dev.cache.checkpoint_slots %d is dropped: %s", checkpoint_slots, why)
        elif checkpoint_slots is not None:
            ckpt = checkpoint_slots
            why = "configured"
        else:
            room = max(0, memory_bytes - (live + 1) * state_slot_bytes)
            cap = int(CHECKPOINT_MEMORY_SHARE * room // state_slot_bytes)
            ckpt = min(max_num_seqs, cap)
            why = (f"min(max_num_seqs {max_num_seqs}, {CHECKPOINT_MEMORY_SHARE:.0%} of the "
                   f"{room / 2**30:.2f} GiB left after the live slots = {cap} slots of "
                   f"{state_slot_bytes / 2**20:.1f} MiB)")
        num_state_slots = live + ckpt + 1
        reasons["num_state_slots"] = f"{live} live slots (max_num_seqs) + {ckpt} checkpoint slots ({why}) + null"
    state_bytes = num_state_slots * state_slot_bytes
    if num_pages is None:
        if state_bytes >= memory_bytes:
            raise ValueError(f"{num_state_slots} state slots need {state_bytes / 2**30:.1f} GiB but only "
                             f"{memory_bytes / 2**30:.1f} GiB are free: lower {_either(knobs)}")
        num_pages = (memory_bytes - state_bytes) // kv_page_bytes
        reasons["num_pages"] = (f"({memory_bytes / 2**30:.2f} GiB free - {state_bytes / 2**30:.2f} GiB of state) "
                                f"/ {kv_page_bytes / 2**20:.2f} MiB per page")
    if num_pages < 2:
        raise ValueError("no memory left for the KV pool")
    need = pages_for_one_request(max_model_len, page_size, reserve_tokens, prefix_caching)
    if num_pages - 1 < need:
        extra = 1 if prefix_caching else 0
        fits = (num_pages - 1 - extra) * page_size - reserve_tokens + 1
        why = (f"the KV pool holds {num_pages - 1:,} pages of {page_size} tokens, but one request of max_model_len "
               f"{max_model_len:,} needs {need:,} ({max_model_len - 1:,} prompt tokens + {reserve_tokens} reserved for "
               f"the steps in flight{' + 1 page for copy-on-write' if extra else ''})")
        if not cap_max_model_len or fits < 2:
            raise ValueError(f"{why}: the largest max_model_len that fits is {max(fits, 0):,}. Lower "
                             f"{_either(['model.max_model_len', *knobs])}, or give the pools more memory")
        log.warning("%s: capping the derived max_model_len to %s", why, f"{fits:,}")
        reasons["max_model_len"] = f"capped to {fits:,}, the longest request the KV pool holds"
        max_model_len = fits
    kv_tokens = (num_pages - 1) * page_size
    full_len = kv_tokens / max_model_len
    log.info("KV pool: %d pages of %d tokens (%s usable, %.1f GiB); state pool: %d slots (%d live, %d checkpoint; "
             "%.1f GiB); %.2f concurrent requests at max_model_len %d", num_pages, page_size, f"{kv_tokens:,}",
             num_pages * kv_page_bytes / 2**30, num_state_slots, live, ckpt, state_bytes / 2**30, full_len,
             max_model_len)
    if full_len < max_num_seqs:
        log.warning("the KV pool holds %.2f requests of max_model_len (%d tokens), fewer than max_num_seqs (%d); "
                    "running requests may be preempted when contexts grow long", full_len, max_model_len,
                    max_num_seqs)
    return PoolSizes(num_pages=int(num_pages), num_state_slots=int(num_state_slots), num_live_slots=live,
                     num_checkpoint_slots=ckpt, reasons=reasons, max_model_len=int(max_model_len))
