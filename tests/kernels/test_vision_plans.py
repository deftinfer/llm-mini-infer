"""The vision tower's attention plans in flight (`kernels.flashinfer.FlashInferVisionAttention`), on CPU: a stand-in
for FlashInfer's ragged wrapper over a model of the stream. A plan writes its bytes into the page-locked buffer the
wrapper holds and queues that buffer's copy to the wrapper's device int workspace; a run queues a kernel that reads the
int workspace with the plan the host passed it; an event completes when the stream reaches it. The stream lags behind
the host, as when the runner queues a step's ENCODE while the step before still runs, and works only when a test or a
host wait lets it. With one tower call per image (batch invariance), no plan may write a buffer whose copy is still
queued, and every call must run with its own plan: three images in a step, two in each of two steps; when every
buffer's copy is queued, the plan waits for the oldest. tests/kernels/test_vision_plans_gpu.py runs FlashInfer
itself."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from llm_mini_infer.kernels.flashinfer import FlashInferVisionAttention


class Stream:
    """The device's work in the order the host queued it: page-locked buffers' copies, events and kernels, run by
    `drain` (all of it, or up to an event)."""

    def __init__(self) -> None:
        self.queue: list[tuple[Any, ...]] = []
        self.buffers: list[int] = []  # the page-locked buffer of each plan, in order (its address)
        self.overwrites: list[int] = []  # the plans (their index) that wrote a buffer whose copy was still queued
        self.wrong: list[list[int]] = []  # the calls (their plan) whose kernel read another call's plan

    def plan(self, buf: torch.Tensor, plan: list[int], int_ws: torch.Tensor) -> None:
        if any(op[0] == "copy" and op[1] is buf for op in self.queue):
            self.overwrites.append(len(self.buffers))
        self.buffers.append(buf.data_ptr())
        nbytes = 4 * len(plan)
        buf[:nbytes].view(torch.int32).copy_(torch.tensor(plan, dtype=torch.int32))
        self.queue.append(("copy", buf, int_ws, nbytes))

    def run(self) -> None:
        op = self.queue.pop(0)
        if op[0] == "copy":
            _, src, dst, nbytes = op
            dst[:nbytes].copy_(src[:nbytes])
        elif op[0] == "event":
            op[1].done = True
        else:
            _, plan, int_ws = op
            if int_ws[:4 * len(plan)].view(torch.int32).tolist() != plan:
                self.wrong.append(plan)

    def drain(self, until: Event | None = None) -> None:
        while self.queue and (until is None or not until.done):
            self.run()


class Event:
    """A CUDA event on the model's stream: complete before its first record, as CUDA's."""

    def __init__(self, stream: Stream):
        self.stream, self.done = stream, True

    def record(self) -> None:
        self.done = False
        self.stream.queue.append(("event", self))

    def query(self) -> bool:
        return self.done

    def synchronize(self) -> None:
        self.stream.drain(until=self)


def fake_flashinfer(stream: Stream) -> types.ModuleType:
    class Ragged:
        """FlashInfer's `BatchPrefillWithRaggedKVCacheWrapper` as the vision attention calls it: the plan's bytes are
        its patch offsets."""

        def __init__(self, float_workspace_buffer: torch.Tensor, kv_layout: str = "NHD", backend: str = "auto"):
            assert backend == "fa2"  # the kernel the tower's batch invariance was proved on
            self._float_workspace_buffer = float_workspace_buffer
            # FlashInfer allocates both int buffers (the attention may give it others)
            self._int_workspace_buffer = torch.zeros(64, dtype=torch.uint8)
            self._pin_memory_int_workspace_buffer = torch.zeros(64, dtype=torch.uint8)
            self._plan_info: list[int] | None = None

        def plan(self, qo_indptr: torch.Tensor, kv_indptr: torch.Tensor, num_qo_heads: int, num_kv_heads: int,
                 head_dim_qk: int, causal: bool = False, sm_scale: float | None = None, q_data_type: Any = None,
                 kv_data_type: Any = None, non_blocking: bool = True) -> None:
            assert torch.equal(qo_indptr, kv_indptr) and not causal
            self._plan_info = qo_indptr.tolist()
            stream.plan(self._pin_memory_int_workspace_buffer, self._plan_info, self._int_workspace_buffer)

        def run(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            stream.queue.append(("kernel", self._plan_info, self._int_workspace_buffer))
            return torch.empty_like(q)

    mod = types.ModuleType("flashinfer")
    mod.BatchPrefillWithRaggedKVCacheWrapper = Ragged  # type: ignore[attr-defined]
    return mod


@pytest.fixture
def vision(monkeypatch) -> tuple[FlashInferVisionAttention, Stream]:
    stream = Stream()
    monkeypatch.setitem(sys.modules, "flashinfer", fake_flashinfer(stream))
    # pinning opens a CUDA context on a GPU host: the plans' index tensors stay unpinned
    monkeypatch.setattr("llm_mini_infer.kernels.flashinfer._pinned32", lambda a: torch.as_tensor(a, dtype=torch.int32))
    attn = FlashInferVisionAttention(16, 72, torch.bfloat16, torch.device("cpu"),
                                     workspace=torch.zeros(16, dtype=torch.uint8), event=lambda: Event(stream))
    return attn, stream


def call(attn: FlashInferVisionAttention, n: int) -> SimpleNamespace:
    """One tower call of one image of n patches, as `ImageEncoder._tower` makes it: its plan, then its attention (one
    layer's)."""
    meta = SimpleNamespace(n=n, cu_host=[0, n], attn=None)
    attn.plan(meta)
    q = torch.zeros(n, 1, 1)
    attn.run(meta, q, q, q)
    return meta


def test_the_wrappers_share_the_workspaces_and_have_buffers_of_their_own(vision) -> None:
    attn, _ = vision
    assert len(attn.wrappers) == attn.PLANS >= 4
    assert all(w._int_workspace_buffer is attn.int_workspace for w in attn.wrappers)
    assert all(w._float_workspace_buffer is attn.workspace for w in attn.wrappers)
    pinned = [w._pin_memory_int_workspace_buffer for w in attn.wrappers]
    assert len({b.data_ptr() for b in pinned}) == attn.PLANS
    assert all(b.numel() >= attn.int_workspace.numel() for b in pinned)  # FlashInfer allocates within the workspace


def test_three_images_in_a_step_plan_into_three_buffers(vision) -> None:
    """Images of 2,520, 594 and 1,188 patches planned while the stream has run none of their copies."""
    attn, stream = vision
    for n in (2520, 594, 1188):
        call(attn, n)
    assert stream.overwrites == [] and len(set(stream.buffers)) == 3 and attn.waits == 0
    stream.drain()
    assert stream.wrong == []


def test_two_steps_of_two_images_never_write_a_queued_buffer(vision) -> None:
    """Step N + 1's images planned while step N's copies are still queued (at most two steps are in flight); step
    N + 2's after step N ran (the runner's `wait(N)`), into step N's buffers, without a wait."""
    attn, stream = vision
    steps = [(2520, 594), (594, 2520), (1188, 2520)]
    ran = Event(stream)
    for n in steps[0]:
        call(attn, n)
    ran.record()
    for n in steps[1]:
        call(attn, n)
    assert stream.overwrites == [] and len(set(stream.buffers)) == 4
    stream.drain(until=ran)
    for n in steps[2]:
        call(attn, n)
    assert stream.overwrites == [] and stream.buffers[4:] == stream.buffers[:2] and attn.waits == 0
    stream.drain()
    assert stream.wrong == []


def test_a_plan_waits_while_its_buffers_copy_is_queued(vision) -> None:
    """A fifth call with every buffer's copy queued: its plan waits for the first plan's copy (and only for that)
    before it writes the first buffer again."""
    attn, stream = vision
    for n in (2520, 594, 1188, 594):
        call(attn, n)
    assert attn.waits == 0
    call(attn, 2520)
    assert attn.waits == 1 and stream.overwrites == [] and stream.buffers[4] == stream.buffers[0]
    assert sum(op[0] == "copy" for op in stream.queue) == 4  # the wait ran the first copy, not the later ones
    stream.drain()
    assert stream.wrong == []


def test_a_call_runs_before_the_next_one_is_planned(vision) -> None:
    """The wrappers share one device int workspace: a call's attention after another call's plan would read that
    plan, so it raises."""
    attn, _ = vision
    first = SimpleNamespace(n=594, cu_host=[0, 594], attn=None)
    attn.plan(first)
    call(attn, 2520)
    q = torch.zeros(594, 1, 1)
    with pytest.raises(RuntimeError, match="planned"):
        attn.run(first, q, q, q)
