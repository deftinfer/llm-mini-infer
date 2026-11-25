"""The randomized mode of the engine's grammar builds (`tests/support/compile_cases.py`): builds ending at random turns
change when requests run, never their outputs, and keep every accounting check."""

from __future__ import annotations

from hypothesis import given

from ..support.compile_cases import compile_async_workloads, run_async_workload
from ..support.fuzzing import campaign_settings, saved_on_failure


@campaign_settings(150)
@given(compile_async_workloads())
def test_random_compile_async_workloads(w) -> None:
    with saved_on_failure("compile_async", w):
        run_async_workload(w)
