"""Radix tree properties (hypothesis), against a content model: every page records which token each of its positions
holds (as written by the request that owns it), and every checkpoint slot records the token prefix its state
represents. A hit must then hand out pages holding exactly the request's own prefix, and a checkpoint at exactly the
hit offset; refcounts must always equal the tree's references plus the running requests' page tables."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given

from llm_mini_infer.cache.pools import PagePool, StatePool, WindowStore
from llm_mini_infer.cache.radix import SPAN_BASE, RadixCache, Spans, lost_tokens, precise_ns

from ..support.fuzzing import campaign_settings, saved_on_failure
from ..support.radix_model import Model, run_scenario, scenarios


@campaign_settings(400)
@given(scenarios())
def test_random_requests_keep_every_invariant(s) -> None:
    with saved_on_failure("radix", s):
        run_scenario(s)


def test_hybrid_hit_is_the_deepest_checkpoint_and_salt_isolates() -> None:
    mdl = Model(4, 64, 4, hybrid=True)
    a = np.arange(40, dtype=np.int64)
    assert mdl.request(None, a, 40, [16, 32])
    mdl.finish(0)
    mdl.complete(set())
    b = np.concatenate([a[:35], [99, 98, 97]])
    m = mdl.cache.match(None, b, mdl.step + 1, len(b) - 1)
    assert (m.hit, m.kv_len) == (32, 35)  # KV goes deeper than the deepest state: a branch point at 32
    assert mdl.cache.match("salt", b, mdl.step + 1, len(b) - 1).kv_len == 0
    m = mdl.cache.match(None, a, mdl.step + 1, 31)  # limit below 32: the shallower checkpoint
    assert m.hit == 16


def test_visibility_pending_entries_are_usable_from_the_next_plan() -> None:
    mdl = Model(4, 64, 4, hybrid=True)
    a = np.arange(30, dtype=np.int64)
    assert mdl.request(None, a, 30, [24])
    same = mdl.cache.match(None, a, mdl.step, 29)  # the plan that writes it
    assert same.hit == 0 and same.pending == 24
    nxt = mdl.cache.match(None, a, mdl.step + 1, 29)
    assert nxt.hit == 24 and nxt.pending == 0


def test_partial_page_hit_and_split_tails() -> None:
    mdl = Model(4, 64, 4, hybrid=True)
    a = np.arange(30, dtype=np.int64)
    assert mdl.request(None, a, 30, [10])  # node boundary at 10, mid-page: [0, 10) keeps page 2 as its tail
    mdl.complete(set())
    m = mdl.cache.match(None, np.concatenate([a[:12], [77, 78]]), mdl.step + 1, 13)
    assert m.hit == 10
    full, tail = mdl.cache.hit_pages(m)
    assert len(full) == 2 and tail and mdl.kv.refcount[tail] >= 2  # [0,10)'s tail and [10,30)'s full page
    mdl.check()


def test_eviction_cascade_and_locks() -> None:
    """Locks keep KV, not checkpoints: a running request's checkpoints are evicted by LRU like any other, its pages
    only once it is gone."""
    mdl = Model(4, 16, 2, hybrid=True)
    a = np.arange(20, dtype=np.int64)
    assert mdl.request(None, a, 20, [8, 16])
    mdl.complete(set())
    assert mdl.cache.evict_pages(100) == 0  # locked by the running request
    assert mdl.cache.evict_checkpoints(1, mdl.step + 1) == 1 and mdl.cache.num_nodes >= 2  # a checkpoint alone
    mdl.finish(0)
    assert mdl.cache.num_ckpts == 1
    freed = mdl.cache.evict_pages(100)
    assert freed == 5 and mdl.cache.num_nodes == 0 and mdl.states.num_free_ckpt == 2  # leaves take their checkpoints
    mdl.check()


def test_checkpoints_the_plan_under_construction_writes_or_reads_are_kept() -> None:
    """Checkpoint eviction skips what the plan being built (step S) writes (stamped S) and the checkpoints its
    STATE_COPYs read (pinned until the plan is built); everything else goes by LRU, locked or not."""
    mdl = Model(4, 64, 2, hybrid=True)
    a = np.arange(40, dtype=np.int64)
    assert mdl.request(None, a, 40, [16])  # step s: a checkpoint at 16, locked by the request
    c = mdl.cache
    s = mdl.step
    assert c.evict_checkpoints(1, s) == 0  # written by the plan under construction (step s)
    assert c.evict_checkpoints(1, s + 1) == 1  # next plan: evictable although its node is locked
    assert mdl.request(None, np.concatenate([a, [7, 7]]), 40, [32])  # step s + 1: hits nothing, adds 32
    m = c.match(None, a, s + 3, 39)
    assert m.hit == 32
    c.pin_checkpoint(m.node)  # plan 3's STATE_COPY reads it
    assert c.evict_checkpoints(1, s + 3) == 0
    spare = c.alloc_checkpoint(s + 3)  # the free slot
    assert spare and c.alloc_checkpoint(s + 3) == 0  # nothing else may be taken
    mdl.states.release(spare)
    c.unpin_checkpoints()
    assert c.evict_checkpoints(1, s + 3) == 1 and c.num_ckpts == 0
    mdl.check()


def test_checkpoint_eviction_can_spare_locked_nodes() -> None:
    """Window mode's first pass (`Scheduler._ensure_pages`) spares the checkpoints of locked nodes: the ones running
    requests' paths hold, a shared prefix's among them. The default still evicts them, LRU, and `enough` stops a pass
    as soon as it holds."""
    mdl = Model(4, 64, 3, hybrid=True)
    a = np.arange(40, dtype=np.int64)
    assert mdl.request(None, a, 40, [16])  # a checkpoint at 16, locked by the running request
    b = np.concatenate([np.arange(16, dtype=np.int64), [9] * 24])
    assert mdl.request(None, b, 40, [32])  # another at 32 on b's path, locked too
    c = mdl.cache
    s = mdl.step + 1
    assert c.num_ckpts == 2
    assert c.evict_checkpoints(4, s, spare_locked=True) == 0 and c.num_ckpts == 2
    assert c.evict_checkpoints(4, s, enough=lambda: c.num_ckpts <= 1) == 1 and c.num_ckpts == 1
    assert c.evict_checkpoints(4, s) == 1 and c.num_ckpts == 0
    mdl.check()


def test_eviction_takes_unlocked_entries_still_in_flight() -> None:
    """An unlocked entry whose writing step has not completed is evicted like any other (stream order makes the
    reuse safe), and a later rollback of that step skips the dead node."""
    mdl = Model(4, 16, 2, hybrid=True)
    a = np.arange(20, dtype=np.int64)
    assert mdl.request(None, a, 20, [8, 16])  # its step writes [0, 20) and two checkpoints: still in flight
    assert mdl.cache.done_step == 0
    mdl.finish(0)  # the request is gone (preempted or aborted) before its step completed
    assert mdl.cache.evict_checkpoints(1, mdl.step + 1) == 1  # an in-flight checkpoint alone
    assert mdl.cache.evict_pages(100) == 5 and mdl.cache.num_nodes == 0
    assert mdl.kv.num_free == 15 and mdl.states.num_free_ckpt == 2
    assert mdl.cache.complete(1, {mdl.aid}) == 0  # the step reports NONFINITE: nothing left to roll back
    mdl.check()


def test_rollback_removes_the_subtree_and_its_checkpoints() -> None:
    mdl = Model(4, 64, 4, hybrid=True)
    a = np.arange(30, dtype=np.int64)
    assert mdl.request(None, a, 12, [8])  # the first request, in the first step
    assert mdl.request(None, a, 30, [24])  # the second, in the next: hits 8, extends to 30
    bad = mdl.running[1]["aid"]
    mdl.cache.complete(1, set())
    mdl.cache.complete(2, {bad})
    assert mdl.running[1]["lock"].dead and not mdl.running[0]["lock"].dead
    m = mdl.cache.match(None, a, 3, 29)
    assert m.kv_len == 12 and m.hit == 8
    mdl.finish(1)
    mdl.finish(0)
    mdl.check()


@pytest.mark.parametrize("ps", [1, 4, 16])
def test_split_keeps_pages_consistent(ps) -> None:
    kv = PagePool(200)
    states = StatePool(8, 2)
    cache = RadixCache(ps, kv, states, hybrid=True)
    toks = np.arange(100, dtype=np.int64)
    pages = kv.alloc(-(-100 // ps))
    deep = cache.insert(None, toks, 100, pages, 1, 1)
    for c in (37, 50, 3, 99):
        cache.node_ending_at(deep, c)
        cache.check()
    full, node = [], deep
    while node.parent is not None:  # each full page in the node it ends in, each tail the page holding the node's end
        assert node.tail == (pages[node.end // ps] if node.end % ps else 0)
        full = node.pages + full
        node = node.parent
    assert full == pages[:100 // ps]
    kv.release(pages)
    cache.complete(1)
    cache.clear()
    assert kv.num_free == 199


@pytest.mark.parametrize("cow", [True, False])
def test_insert_never_adopts_the_copy_on_write_page(cow: bool) -> None:
    """A request whose hit ends inside a page copies the tree's page there (KV_COPY) into its own first page; its insert
    never makes that copy the junction's tail, so the junction keeps its page and its stamp. Without `cow` the copy is
    adopted (the tree alone referenced the tail) and the junction is stamped with the request's step and admission."""
    kv = PagePool(64)
    cache = RadixCache(4, kv, StatePool(8, 2), hybrid=True)
    key = np.arange(20, dtype=np.int64)
    w = kv.alloc(3)
    cache.insert(None, key, 10, w, 1, 1)  # W: [0, 10), its tail page w[2]
    cache.complete(1)
    kv.release(w)  # W left
    own = kv.alloc(3)  # H: a hit at 10 shares w[:2] and copies w[2] into own[0]
    kv.incref(w[:2])
    cache.insert(None, key, 20, w[:2] + own, 9, 2, own[0] if cow else 0, 10)
    j = cache.roots[None].children[0]
    assert (j.start, j.end) == (0, 10) and cache.kept_dups == 0
    assert (j.tail, j.ready, j.writer) == ((w[2], 1, 1) if cow else (own[0], 9, 2))
    cache.check()


def test_the_copy_on_write_page_is_adopted_from_its_hit_on() -> None:
    """The copy is never adopted into a node that starts before the hit, but it is into one that starts at the hit:
    there it holds the request's own positions, like any page it computed (X's branch at 10, recomputed by H)."""
    kv = PagePool(64)
    cache = RadixCache(4, kv, StatePool(8, 2), hybrid=True)
    key = np.arange(20, dtype=np.int64)
    w = kv.alloc(3)
    cache.insert(None, key, 10, w, 1, 1)  # W: [0, 10)
    x = kv.alloc(3)
    kv.incref(w[:2])
    cache.insert(None, key, 20, w[:2] + x, 2, 2, x[0], 10)  # X: a hit at 10 (w[2] copied into x[0]), [10, 20)
    cache.complete(2)
    kv.release(w)
    kv.release(w[:2] + x)  # W and X left
    own = kv.alloc(3)  # H: the same hit at 10, recomputes [10, 20)
    kv.incref(w[:2])
    cache.insert(None, key, 20, w[:2] + own, 9, 3, own[0], 10)
    j, x_node = cache.match(None, key, 10, 20).path[:2]
    assert (j.end, j.tail, j.ready) == (10, w[2], 1)
    assert (x_node.start, x_node.pages, x_node.ready) == (10, own, 9)
    kv.release(w[:2] + own)
    cache.complete(9)
    cache.clear()
    cache.check()
    assert kv.num_free == kv.num_pages - 1


def test_share_tails_frees_the_junction_tails_no_child_shares() -> None:
    """Pages of 4. A computes [0, 10) and B branches off it at 5; R resumes at 7, copying the tail of [5, 7), and its
    insert splits A's node there. Neither junction takes R's copy (they start before its hit), and once A's and B's
    branches left, [0, 5) and [5, 7) hold tails no child shares, pinned by R's lock: two pages more on R's path.
    `share_tails` gives each its child's page, children first, so both end on R's copy; it stamps and records nothing,
    shares no page a write in flight holds (R's node, before R's step completes), and an unverified one (R's copy, while
    R's chunk is unverified) only with `force`."""
    kv = PagePool(64)
    cache = RadixCache(4, kv, StatePool(8, 2), hybrid=True)
    s = list(range(1, 8))
    a, b, r = (np.asarray(k, np.int64) for k in ([*s, 70, 71, 72], [*s[:5], 50, 51, 52], [*s, 90, 91, 92, 93, 94]))
    pa = kv.alloc(3)
    cache.insert(None, a, 10, pa, 1, 1)  # A: [0, 10)
    cache.complete(1)
    kv.release(pa)
    pb = kv.alloc(2)
    cache.insert(None, b, 8, pb, 2, 2)  # B: [5, 8) below [0, 5), which takes B's pages (the tree alone held A's)
    kv.release(pb)
    cache.complete(2)
    j5 = cache.roots[None].children[1]
    assert (j5.end, j5.pages, j5.tail) == (5, [pb[0]], pb[1])
    pr = [pb[0], *kv.alloc(2)]  # R: a hit at 7 shares B's page 0 and copies the page holding 4-6 into pr[1]
    kv.incref(pb[:1])
    cache.unverified[pr[1:]] = 3  # R's chunk is not its last
    deep = cache.insert(None, r, 12, pr, 3, 3, pr[1], 7)
    cache.lock(deep)
    j7 = deep.parent
    assert (j7.start, j7.end, j7.tail) == (5, 7, pa[1]) and j5.tail == pb[1]
    cache.remove_subtree(j5.children[int(b[5])])  # B's branch and A's left (evicted)
    cache.remove_subtree(j7.children[int(a[7])])
    stamps = (j5.ready, j7.ready)
    assert cache.share_tails() == 1 and j5.tail == pa[1]  # [5, 7)'s tail: R's node is in flight, its page unverified
    cache.complete(3)
    assert cache.share_tails() == 0 and j7.tail == pa[1]  # R's copy is still unverified
    assert cache.share_tails(force=True) == 1 and j5.tail == j7.tail == pr[1]
    assert (j5.ready, j7.ready) == stamps and not cache._writes and cache.shared_tails == 2
    cache.check()
    cache.unlock(deep)
    kv.release(pr)
    cache.clear()
    cache.check()
    assert kv.num_free == kv.num_pages - 1


def test_an_insert_never_adopts_an_unverified_page_over_a_verified_one() -> None:
    """A page that a request's prefill wrote before a SAMPLE row of it could show it finite (marked `unverified` with
    its admission) never replaces a page the tree holds unmarked: the node keeps its page and its stamp, and the
    request keeps a duplicate (`kept_dups`) that `readopt` adopts only with `force` (page pressure). A page that is
    itself unverified is replaced like any duplicate."""
    kv = PagePool(64)
    cache = RadixCache(4, kv, StatePool(8, 2), hybrid=True)
    key = np.arange(8, dtype=np.int64)
    w = kv.alloc(2)
    cache.insert(None, key, 8, w, 1, 1)  # W: [0, 8)
    cache.complete(1)
    kv.release(w)  # the tree alone holds W's pages
    r = kv.alloc(2)
    cache.unverified[r] = 5  # R (admission 5) recomputed [0, 8) in a chunk before its last
    node = cache.insert(None, key, 8, r, 2, 5)
    assert node.pages == w and (node.ready, node.writer) == (1, 1) and cache.kept_dups == 2
    assert cache.readopt(node, r, 8, 2, 5) and node.pages == w
    assert not cache.readopt(node, r, 8, 2, 5, force=True) and node.pages == r and node.ready == 2
    kv.release(r)  # R left (its pages stay marked: say it was preempted)
    s = kv.alloc(2)
    cache.insert(None, key, 8, s, 3, 6)  # S's pages, unmarked, replace R's
    assert node.pages == s and (node.ready, node.writer) == (3, 6)
    kv.release(s)
    cache.complete(3)
    cache.clear()
    cache.check()
    assert kv.num_free == kv.num_pages - 1


def test_an_insert_ending_inside_an_unverified_page_takes_its_own() -> None:
    """R's earlier admission left [0, 12) with its pages unverified; its new one recomputes [0, 7) into its own pages
    and inserts: the node ending at 7 takes R's new page holding 7 as its tail (the tree's holds R's earlier writes
    past R's end), and the node from 7 on keeps the earlier page, which a removal of the earlier writes from 7 takes
    without the node R now locks."""
    kv = PagePool(64)
    cache = RadixCache(2, kv, StatePool(8, 2), hybrid=True)
    key = np.arange(12, dtype=np.int64)
    old = kv.alloc(6)
    cache.unverified[old] = 4
    cache.insert(None, key, 12, old, 1, 4)
    cache.complete(1)
    kv.release(old)  # R preempted: its earlier writes stay, unverified
    new = kv.alloc(4)
    cache.unverified[new] = 9  # its readmission (admission 9), from position 0
    deep = cache.insert(None, key, 7, new, 2, 9)
    lower = deep.children[int(key[7])]
    assert (deep.end, deep.pages, deep.tail) == (7, new[:3], new[3]) and lower.pages[0] == old[3]
    cache.lock(deep)
    assert cache.remove_unverified(None, key, [4], lo=7) == 1 and lower.dead and not deep.dead
    cache.unlock(deep)
    kv.release(new)
    cache.complete(2)
    cache.clear()
    cache.check()
    assert kv.num_free == kv.num_pages - 1


def test_remove_unverified_takes_the_nodes_holding_the_pages_and_below() -> None:
    """`remove_unverified(ns, key, aids, lo)` walks key's path and removes from the first position at or after lo held
    in a page one of `aids` wrote unverified, split at that page's start, the subtree below included: B's own node,
    not the node [0, 5) it shares with A (B recomputed it, and its pages were not adopted), nor A's branch. A page
    adopted by force (page pressure) takes the shared node with it."""
    kv = PagePool(64)
    cache = RadixCache(2, kv, StatePool(8, 2), hybrid=True)
    a = np.arange(12, dtype=np.int64)
    b = np.concatenate([a[:5], [99, 98, 97, 96, 95]])
    pa = kv.alloc(6)
    cache.insert(None, a, 12, pa, 1, 1)  # A
    cache.complete(1)
    kv.release(pa)
    pb = kv.alloc(5)
    cache.unverified[pb] = 7  # B (admission 7): [0, 10) cold, in a chunk before its last
    deep = cache.insert(None, b, 10, pb, 2, 7)  # [0, 5) keeps A's pages; B's node [5, 10)
    assert cache.remove_unverified(None, b, [7], lo=6) == 1 and deep.dead  # [6, 10), split off [5, 10)
    assert cache.remove_unverified(None, b, [3]) == 0 and cache.remove_unverified("salt", b, [7]) == 0
    assert cache.remove_unverified(None, b, [7]) == 1  # [5, 6)
    assert cache.match(None, a, 3, 12).kv_len == 12 and cache.match(None, b, 3, 10).kv_len == 5
    deep = cache.insert(None, b, 10, pb, 3, 7)
    cache.readopt(deep, pb, 10, 3, 7, force=True)  # page pressure: [0, 5) takes B's pages
    assert cache.remove_unverified(None, b, [7]) == 3  # [0, 5), A's branch and B's
    assert cache.num_nodes == 0 and cache.rolled_back == 5
    kv.release(pb)
    cache.complete(3)
    cache.check()
    assert kv.num_free == kv.num_pages - 1


def test_a_lock_on_a_removed_node_unlocks_cleanly() -> None:
    """Removal takes locked nodes too (as rollback does): the request that held the lock unlocks it as usual, the lock
    counts of the nodes left stay consistent, and the tree frees every page afterwards."""
    kv = PagePool(64)
    cache = RadixCache(2, kv, StatePool(8, 2), hybrid=True)
    a = np.arange(12, dtype=np.int64)
    pa = kv.alloc(6)
    cache.unverified[pa[3:]] = 2
    deep = cache.insert(None, a, 12, pa, 1, 2)
    cache.lock(deep)
    assert cache.remove_unverified(None, a, [2]) == 1 and deep.dead
    cache.check()
    cache.unlock(deep)
    cache.check()
    kv.release(pa)
    cache.clear()
    assert kv.num_free == kv.num_pages - 1 and cache.num_nodes == 0


def test_pending_writers_count_only_the_pages_a_hit_reads() -> None:
    """A hit depends on the writes in flight of the nodes whose pages it reads: the full pages below it, and the page
    holding it (copied). A node whose only page is a tail that a deeper node's full page of the same index overrides is
    not read, so the step that adopted a page as that tail (stamping the node) is no dependency of the hit (otherwise,
    attention-only, a rollback of such a step fails a request that read none of its pages)."""
    kv = PagePool(64)
    cache = RadixCache(4, kv, StatePool(4, 2), hybrid=False)
    key = np.arange(20, dtype=np.int64)
    c = kv.alloc(5)
    deep = cache.insert(None, key, 20, c, 1, 1)
    cache.node_ending_at(cache.node_ending_at(deep, 10), 8)  # [0, 8), [8, 10) (only its tail c[2]), [10, 20)
    cache.complete(1)
    kv.release(c)
    rk = key.copy()
    rk[11] = 99  # R: a hit at 8, then [8, 12) on its own page, branching off at 11
    kv.incref(c[:2])
    r = c[:2] + kv.alloc(1)
    cache.insert(None, rk, 12, r, 2, 2)  # adopts r[2] as the tails of [8, 10) and [10, 11), stamping them (2, 2)
    assert [(n.start, n.end, n.tail, n.ready) for n in cache.match(None, rk, 3, 12).path[1:3]] == \
        [(8, 10, r[2], 2), (10, 11, r[2], 2)]
    m = cache.match(None, key, 3, 20)  # a hit at 20 on the first path, while R's step is in flight
    assert m.hit == 20 and cache.hit_pages(m) == (c, 0)  # none of R's pages
    assert cache.pending_writers(m.path, m.hit, m.node) == set()
    kv.release(r)
    cache.complete(2)
    cache.clear()
    cache.check()


def test_every_adoption_is_recorded_for_rollback() -> None:
    """Two requests adopt their own pages into one node in the same step (the first left before the second's insert):
    the node holds the second's pages though the first stamped it, so a rollback of the second's step removes it too."""
    kv = PagePool(64)
    cache = RadixCache(4, kv, StatePool(8, 2), hybrid=True)
    key = np.arange(8, dtype=np.int64)
    w = kv.alloc(2)
    cache.insert(None, key, 8, w, 1, 1)
    cache.complete(1)
    kv.release(w)
    a = kv.alloc(2)
    cache.insert(None, key, 8, a, 5, 2)  # adopts its pages: stamped (5, 2)
    kv.release(a)
    b = kv.alloc(2)
    node = cache.insert(None, key, 8, b, 5, 3)  # adopts its own in the same step
    assert node.pages == b and (node.ready, node.writer) == (5, 2)
    assert cache.complete(5, {3}) == 1 and node.dead
    kv.release(b)
    cache.check()
    cache.clear()
    assert kv.num_free == kv.num_pages - 1


def test_a_hit_depends_on_every_adoption_of_its_node_in_flight() -> None:
    """A adopts its pages into W's node and leaves, then B adopts its own in the same step: the node holds B's pages
    under A's stamp. A hit planned while that step is in flight reads B's pages, so it depends on B's write too, and
    B's rollback fails it (with the stamp alone it would keep running on the removed pages)."""
    kv = PagePool(64)
    cache = RadixCache(4, kv, StatePool(8, 2), hybrid=False)
    key = np.arange(12, dtype=np.int64)
    w = kv.alloc(3)
    cache.insert(None, key, 12, w, 1, 1)
    cache.complete(1)
    kv.release(w)
    cache.complete(3)  # plan 5 is built: plan 4 is in flight
    a = kv.alloc(3)
    cache.insert(None, key, 12, a, 4, 2)
    kv.release(a)
    b = kv.alloc(3)
    node = cache.insert(None, key, 12, b, 4, 3)
    assert node.pages == b and (node.writer, node.ready) == (2, 4)
    m = cache.match(None, key, 5, 12)
    assert cache.hit_pages(m) == (b, 0)
    assert cache.pending_writers(m.path, m.hit, m.node) == {(2, 4), (3, 4)}
    assert cache.complete(4, {3}) == 1 and node.dead
    kv.release(b)
    cache.check()
    assert kv.num_free == kv.num_pages - 1


def test_a_failed_checkpoint_write_leaves_its_slot_reused_by_a_later_step() -> None:
    """aid 2 attaches the only checkpoint slot at step id 2; step id 3 evicts it while step id 2 is in flight and aid 3
    attaches the same slot to the same node. aid 2 fails before its results arrive: its stale record names that slot,
    but the checkpoint is step id 3's and stays, until step id 3's own rollback."""
    kv = PagePool(64)
    cache = RadixCache(4, kv, StatePool(4, 2), hybrid=True)  # one checkpoint slot
    key = np.arange(8, dtype=np.int64)
    pages = kv.alloc(2)
    node = cache.insert(None, key, 8, pages, 1, 1)
    cache.complete(1)
    slot = cache.alloc_checkpoint(2)
    cache.attach(node, slot, 2, 2, "prefill")
    assert cache.alloc_checkpoint(3) == slot and not node.ckpt
    cache.attach(node, slot, 3, 3, "prefill")
    assert cache.drop_writes(2, {2}) == 0 and (node.ckpt, node.ckpt_ready) == (slot, 3)
    cache.check()
    assert cache.complete(3, {3}) == 1 and not node.ckpt
    kv.release(pages)
    cache.clear()
    cache.check()


def test_window_mode_allocates_no_checkpoint_slot() -> None:
    """A window is made from a request's pages: asking the window store for a slot is a caller's error, and evicts no
    window on the way."""
    kv = PagePool(16)
    windows = WindowStore(kv)
    cache = RadixCache(4, kv, windows, hybrid=True)
    pages = kv.alloc(2)
    node = cache.insert(None, np.arange(8, dtype=np.int64), 8, pages, 1, 1)
    h = windows.make(0, pages)
    cache.attach(node, h, 1, 1, "prefill")
    cache.complete(1)
    with pytest.raises(RuntimeError, match="window mode"):
        cache.alloc_checkpoint(2)
    assert node.ckpt == h and windows.is_checkpoint(h)
    kv.release(pages)
    cache.clear()
    cache.check()


@pytest.mark.parametrize("op", ["incref", "release"])
def test_a_null_page_error_names_the_null_page(op: str) -> None:
    kv = PagePool(4)
    p = kv.alloc(1)
    with pytest.raises(RuntimeError, match=r": \[0\]$"):
        getattr(kv, op)([0, *p])


def test_a_node_split_off_goes_with_every_write_in_flight_of_its_node() -> None:
    """W inserts [0, 32) with its step in flight and leaves; R recomputed [0, 19) into its own pages and inserts it in
    the next step: the node adopts R's pages of [0, 16) (stamped with R's step), and is split at 19, the node [0, 19)
    keeping W's page of index 2 as its tail. W's step reports NONFINITE: its rollback removes that node too, not only
    the part [19, 32) that its record names (were a node split off recorded under its node's last stamp only, W's
    poisoned page would stay in the cache)."""
    kv = PagePool(64)
    cache = RadixCache(8, kv, StatePool(8, 2), hybrid=True)
    key = np.arange(32, dtype=np.int64)
    w = kv.alloc(4)
    cache.insert(None, key, 32, w, 1, 1)  # W, its step in flight
    kv.release(w)  # W preempted: the tree alone holds its pages
    r = kv.alloc(3)
    deep = cache.insert(None, key, 19, r, 2, 2)  # R
    assert (deep.pages, deep.tail, deep.ready, deep.writer) == (r[:2], w[2], 2, 2)
    assert cache.complete(1, {1}) == 2 and deep.dead and cache.num_nodes == 0
    kv.release(r)
    cache.complete(2)
    cache.check()
    assert kv.num_free == kv.num_pages - 1


def _lost(m) -> dict[str, int]:
    out = dict.fromkeys(("writing", "checkpoint", "page"), 0)
    lost_tokens(m, m.hit, m.held, out)
    return out


def test_lost_tokens_attention_only() -> None:
    """Pages of 4; the tree holds [0, 10). A later plan's match hits 8 and loses the partial page [8, 10); a plan of the
    writing step itself sees nothing yet (all 10 tokens are a step in flight's); the limit caps what counts."""
    kv = PagePool(64)
    cache = RadixCache(4, kv, StatePool(8, 2), hybrid=False)
    key = np.arange(16, dtype=np.int64)
    pages = kv.alloc(3)
    cache.insert(None, key, 10, pages, 5, 1)
    m = cache.match(None, key, 5, 15)
    assert (m.kv_len, m.held, m.hit) == (10, 10, 0) and _lost(m) == {"writing": 10, "checkpoint": 0, "page": 0}
    cache.complete(5)
    m = cache.match(None, key, 6, 15)
    assert (m.hit, _lost(m)) == (8, {"writing": 0, "checkpoint": 0, "page": 2})
    m = cache.match(None, key[:10], 6, 9)  # a prompt of the cached 10 tokens: its last one is always computed
    assert (m.held, m.hit, _lost(m)) == (9, 8, {"writing": 0, "checkpoint": 0, "page": 1})
    kv.release(pages)


def test_lost_tokens_hybrid() -> None:
    """The tree holds [0, 12) with checkpoints at 4 and 8. A hit at 8 loses [8, 12) (no checkpoint there); with both
    checkpoints evicted (the one at 4 first: the hit used the other), the KV stays and all 12 are lost; a checkpoint at
    12 that the plan under construction writes loses them to it instead."""
    kv = PagePool(64)
    states = StatePool(8, 2)
    cache = RadixCache(4, kv, states, hybrid=True)
    key = np.arange(20, dtype=np.int64)
    pages = kv.alloc(3)
    deep = cache.insert(None, key, 12, pages, 1, 1)
    for c in (4, 8):
        cache.attach(cache.node_ending_at(deep, c), cache.alloc_checkpoint(1), 1, 1, "prefill")
    cache.complete(1)
    m = cache.match(None, key, 2, 19)
    assert (m.held, m.hit, _lost(m)) == (12, 8, {"writing": 0, "checkpoint": 4, "page": 0})
    assert cache.evict_checkpoints(1, 2) == 1 and cache.match(None, key, 2, 19).hit == 8
    assert cache.evict_checkpoints(1, 2) == 1
    m = cache.match(None, key, 2, 19)
    assert (m.held, m.hit, _lost(m)) == (12, 0, {"writing": 0, "checkpoint": 12, "page": 0})
    # the plan under construction (step id 3) writes a checkpoint at 12
    cache.attach(cache.node_ending_at(deep, 12), cache.alloc_checkpoint(3), 3, 2, "prefill")
    m = cache.match(None, key, 3, 19)
    assert (m.pending, m.hit, _lost(m)) == (12, 0, {"writing": 12, "checkpoint": 0, "page": 0})
    kv.release(pages)


def test_colliding_span_keys_diverge_at_the_span() -> None:
    """Two keys equal everywhere, an image span's value over [10, 18) included (its 62 bits collide), the digests not:
    B's match ends at the span's start, inside A's node, as a different token would end it; B's insert files its span
    beside A's, both under the node that ends there; then each matches its own path whole, a third image neither, and
    an image of A's digest A's."""
    from llm_mini_infer.vision.family import KEY_BASE

    assert SPAN_BASE == KEY_BASE  # the span keys' range, as the frontend makes them
    kv = PagePool(64)
    cache = RadixCache(4, kv, StatePool(8, 2), hybrid=False)
    key = np.asarray([*range(10), *[SPAN_BASE | 7] * 8, *range(100, 106)], np.int64)
    a, b, c = (Spans([10], [18], [bytes([x]) * 32]) for x in (1, 2, 3))
    pa = kv.alloc(6)
    cache.insert(None, key, 24, pa, 1, 1, spans=a)
    cache.complete(1)
    m = cache.match(None, key, 2, 23, spans=b)
    assert (m.kv_len, m.hit) == (10, 8)
    pb = [*pa[:2], *kv.alloc(4)]  # shares [0, 8), copies the page of 8-11, computes the rest
    kv.incref(pa[:2])
    cache.insert(None, key, 24, pb, 2, 2, pb[2], 8, spans=b)
    cache.complete(2)
    cache.check()
    junction = cache.roots[None].children[0]
    assert (junction.end, len(junction.children)) == (10, 2)
    assert sorted(junction.children) == [(SPAN_BASE | 7, a.digests[0]), (SPAN_BASE | 7, b.digests[0])]
    for spans, want in ((a, 24), (b, 24), (c, 10), (Spans([10], [18], [bytes([1]) * 32]), 24)):
        assert cache.match(None, key, 3, 24, spans=spans).kv_len == want
    assert cache.match(None, key, 3, 24, spans=b).path[-1].pages[-2:] == pb[-2:]  # B's own pages past the span
    kv.release(pa)
    kv.release(pb)
    cache.clear()
    cache.check()
    assert kv.num_free == kv.num_pages - 1


def test_a_salts_root_goes_with_its_last_node() -> None:
    """A cache salt's root goes once its last node does, by eviction or by a removal (a rollback) whose locks are
    released later; the unsalted namespaces' roots stay, and a salt seen again gets its root back."""
    kv = PagePool(64)
    cache = RadixCache(4, kv, StatePool(8, 2), hybrid=True)
    a = np.arange(12, dtype=np.int64)
    for ns in (None, precise_ns(None), "s", precise_ns("s"), "t"):
        pages = kv.alloc(3)
        cache.insert(ns, a, 12, pages, 1, 2)
        kv.release(pages)
    assert len(cache.roots) == 5
    version = cache.version
    assert cache.evict_pages(1 << 62) == 15 and cache.num_nodes == 0
    assert set(cache.roots) == {None, precise_ns(None)} and cache.version > version
    cache.check()
    pages = kv.alloc(3)
    deep = cache.insert("s", a, 12, pages, 2, 3)
    cache.lock(deep)
    assert cache.remove_subtree(deep) == 1 and "s" in cache.roots  # still locked
    cache.unlock(deep)
    assert "s" not in cache.roots
    cache.check()
    kv.release(pages)
    with pytest.raises(ValueError):
        cache.insert("u", a, 0, [], 3, 4)  # an empty insert makes no root
    assert "u" not in cache.roots
    cache.clear()
    assert set(cache.roots) == {None, precise_ns(None)} and kv.num_free == kv.num_pages - 1
    cache.check()


@pytest.mark.parametrize("precise", ["fp8", "reuse"])
def test_many_salts_leave_no_root_behind(precise: str) -> None:
    """Requests under 40 cache salts, two of each sharing a prompt (with precise reuse, the second refines it), in a
    pool small enough to evict: a salt's roots stay only while its namespace holds nodes, and once the cache is cleared
    only the unsalted ones are left. The accounting checks pass at every turn. The second of a pair arrives while the
    first decodes, so it resumes from the first's end-of-prefill checkpoint before the first's finish demotes it
    (architecture.md, "Eviction"): under this pressure a demoted one would be the next to go."""
    from ..support.harness import Harness, Spec, check_precise

    specs = []
    for i in range(40):
        prompt = [(7 * i + j) % 900 + 1 for j in range(24)]
        for k in range(2):
            specs.append(Spec(rid=2 * i + k, prompt_len=24, prompt=prompt, max_tokens=6, seed=100 + 2 * i + k,
                              arrive=2 * i + 4 * k, salt=f"tenant-{i}"))
    h = Harness(specs=specs, num_pages=48, max_num_seqs=2, checkpoint_slots=16, grid=8, precise=precise,
                refine_after=0)
    kept = {None, precise_ns(None)}
    seen: set = set()

    def step() -> None:
        roots = h.sched.cache.roots
        seen.update(roots)
        assert all(r.children or r.lock > 1 for ns, r in roots.items() if ns not in kept)

    collect = h.collect
    h.collect = lambda rep: (collect(rep), step())  # type: ignore[method-assign]
    h.run(check_every_turn=True)
    assert len(seen) == (80 if precise == "reuse" else 40)  # with reuse, each salt's precise namespace too
    assert h.sched.refines == (40 if precise == "reuse" else 0)
    assert len(h.sched.cache.roots) <= 12  # the others went with their nodes
    if precise == "reuse":
        check_precise(h)
    h.sched.cache.clear()
    assert set(h.sched.cache.roots) <= kept
    h.sched.check_accounting(True)


def node_at(cache, end: int):
    """The node whose checkpoint ends at `end`."""
    stack = list(cache.roots.values())
    while stack:
        n = stack.pop()
        if n.parent is not None and n.end == end and n.ckpt:
            return n
        stack.extend(n.children.values())
    raise LookupError(end)


def test_demoted_checkpoints_go_first_but_never_the_plans() -> None:
    """Three checkpoints on one path, at 8, 16 and 32. The demoted ones go before every live one, least recently used
    first in each tier; but not one the plan under construction writes (stamped with its step) or reads (pinned): those
    stay whatever their tier, and a live one goes instead. A hit makes a demoted checkpoint live again, so the live
    order takes it in turn."""
    mdl = Model(4, 64, 3, hybrid=True)
    a = np.arange(40, dtype=np.int64)
    assert mdl.request(None, a, 40, [8, 16, 32])
    c, s = mdl.cache, mdl.step
    n8, n16, n32 = node_at(c, 8), node_at(c, 16), node_at(c, 32)
    for limit in (8, 16, 32):  # used in this order: n8 is the least recently used
        assert c.match(None, a, s + 1, limit).hit == limit
    assert c.demote(n32) and not c.demote(n32) and c.num_demoted == 1
    mdl.check()
    assert c.evict_checkpoints(1, s) == 0  # the plan of step s wrote all three
    assert c.evict_checkpoints(1, s + 1) == 1 and not n32.ckpt  # the demoted one first, though the newest
    assert c.num_demoted == 0 and c.evicted_ckpts == 1
    c.demote(n16)
    c.pin_checkpoint(n16)  # the plan under construction reads it
    assert c.evict_checkpoints(1, s + 1) == 1 and n16.ckpt and not n8.ckpt  # the live one goes instead
    c.unpin_checkpoints()
    mdl.check()
    m = c.match(None, a, s + 1, 39)
    assert m.node is n16 and not n16.demoted and c.num_demoted == 0  # the hit made it live
    mdl.check()
    assert mdl.request(None, np.concatenate([a, [7] * 8]), 48, [40])  # a third checkpoint, live and newer
    assert c.evict_checkpoints(1, mdl.step + 1) == 1 and not n16.ckpt  # live order: n16 is the older
    mdl.check()


def test_demoting_and_promoting_bump_the_trees_version() -> None:
    """`RadixCache.check` walks the tree again only when `version` moved, so a demotion and its undoing move it, and a
    demotion that changes nothing does not."""
    mdl = Model(4, 64, 3, hybrid=True)
    assert mdl.request(None, np.arange(40, dtype=np.int64), 40, [8, 16])
    c = mdl.cache
    n16 = node_at(c, 16)
    version = c.version
    assert c.demote(n16) and c.version == version + 1
    assert not c.demote(n16) and c.version == version + 1
    c.promote(n16)
    assert c.version == version + 2 and not n16.demoted
    c.promote(n16)
    assert c.version == version + 2
    mdl.check()


def test_the_check_catches_a_checkpoint_the_heap_lost() -> None:
    """`RadixCache.check` asserts that every checkpoint has an eviction entry at or below its key: with a node's entries
    taken out of the heap, or replaced by one above its key, eviction could never reach it, and the check fails."""
    mdl = Model(4, 64, 3, hybrid=True)
    a = np.arange(40, dtype=np.int64)
    assert mdl.request(None, a, 40, [8, 16])
    c = mdl.cache
    mdl.check()
    n16 = node_at(c, 16)
    kept = list(c._ckpt_heap)
    c._ckpt_heap = [e for e in kept if e[3] is not n16]
    with pytest.raises(AssertionError, match="no heap entry"):
        c.check(full=True)
    c.demote(n16)  # its demotion pushes an entry at its new key: found again
    c.check(full=True)
    c._ckpt_heap = [(1, *e[1:]) if e[3] is n16 else e for e in c._ckpt_heap]  # an entry above its key only
    with pytest.raises(AssertionError, match="no heap entry"):
        c.check(full=True)
