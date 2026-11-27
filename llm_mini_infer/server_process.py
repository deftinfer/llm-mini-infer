"""Standard-library helpers for launching and stopping server process groups."""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import IO, Any


def launch(
    argv: Sequence[str],
    *,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    stdin: int | IO[Any] | None = None,
    stdout: int | IO[Any] | None = None,
    stderr: int | IO[Any] | None = subprocess.STDOUT,
) -> subprocess.Popen:
    """Start a command in a new session so its children share a process group."""
    return subprocess.Popen(argv, cwd=cwd, env=env, stdin=stdin, stdout=stdout, stderr=stderr,
                            start_new_session=True)


def wait(proc: subprocess.Popen, timeout: float) -> bool:
    """Wait up to `timeout` seconds for the process to exit."""
    try:
        proc.wait(timeout=timeout)
        return True
    except subprocess.TimeoutExpired:
        return False


def log_tail(path: str | Path, lines: int = 60) -> str | None:
    """Return the last `lines` from a log, or None when it cannot be read."""
    try:
        return "\n".join(Path(path).read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return None


def _signal_group(proc: subprocess.Popen, sig: int) -> None:
    """Signal the process group created by `launch`; fall back to its leader if needed."""
    try:
        os.killpg(proc.pid, sig)
    except ProcessLookupError:
        pass
    except (AttributeError, PermissionError):
        with contextlib.suppress(OSError):
            proc.send_signal(sig)


def stop(
    proc: subprocess.Popen,
    *,
    first_signal: int,
    waits: Sequence[float],
    escalation_signals: Sequence[int] = (),
    signal_group_signals: bool = False,
) -> str | None:
    """Drain the launcher, escalate if needed, then kill any process-group leftovers.

    The first signal and any escalation signals go to the launcher unless `signal_group_signals` is true. The final
    SIGKILL always goes to the whole process group. `waits` has one timeout for each signal, including that final
    SIGKILL. Returns the name of the last signal sent, or None if the launcher had already exited.
    """
    signals = (first_signal, *escalation_signals, signal.SIGKILL)
    if len(waits) != len(signals):
        raise ValueError(f"expected {len(signals)} wait intervals, got {len(waits)}")

    used: str | None = None
    group_killed = False
    if proc.poll() is None:
        for index, (sig, timeout) in enumerate(zip(signals, waits)):
            used = signal.Signals(sig).name
            is_final = index == len(signals) - 1
            if is_final or signal_group_signals:
                _signal_group(proc, sig)
                group_killed |= is_final
            else:
                proc.send_signal(sig)
            if wait(proc, timeout):
                break
    else:
        proc.wait()

    # A launcher can exit cleanly and still leave workers in its process group.
    if not group_killed:
        _signal_group(proc, signal.SIGKILL)
    if proc.poll() is None:
        try:
            proc.wait(timeout=waits[-1])
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    return used
