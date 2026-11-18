""".github/scripts/ci_sample.py, the memory sampler beside ci.yml's tests: it runs on the runner's own Python, so on the
standard library alone; it prints its lines while the watched process lives, its summary when the stop file appears,
and it ends by itself when the watched process is gone. Its numbers are Linux's /proc; elsewhere the lines are empty."""

from __future__ import annotations

import ast
import subprocess
import sys
import time

from ..support.paths import ROOT

SCRIPT = ROOT / ".github" / "scripts" / "ci_sample.py"


def sample(*args: str, watch: subprocess.Popen) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, str(SCRIPT), "--watch", str(watch.pid), "--every", "0.2", *args],
                            stdout=subprocess.PIPE, text=True)


def test_the_sampler_imports_the_standard_library_alone() -> None:
    tree = ast.parse(SCRIPT.read_text())
    names = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    names |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert names <= set(sys.stdlib_module_names) | {"__future__"}, names


def test_the_stop_file_ends_the_sampler_with_its_summary(tmp_path) -> None:
    watched = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        out = tmp_path / "samples.log"
        p = sample("--stop", str(tmp_path / "stop"), "--out", str(out), watch=watched)
        deadline = time.monotonic() + 20
        while "[mem] +" not in (out.read_text() if out.exists() else "") and time.monotonic() < deadline:
            time.sleep(0.1)
        time.sleep(0.5)
        assert p.poll() is None  # sampling, while the watched process lives
        (tmp_path / "stop").touch()
        text, _ = p.communicate(timeout=20)
    finally:
        watched.kill()
        watched.wait(10)
    lines = text.splitlines()
    assert p.returncode == 0 and lines[0].startswith("[mem] sampler: a line every 0.2 s")
    assert all(f" {key}=" in lines[0] for key in ("mem_total", "cpus", "swap_total")), lines[0]  # the runner's size
    assert sum(line.startswith("[mem] +") for line in lines) >= 2, text  # the samples, and the stop's last one
    end = [line for line in lines if line.startswith("[end] ")]
    assert "peaks: used=" in end[0] and end[1].startswith("[end] Python processes left: ")
    assert any(line.startswith("[end] /dev/shm files left: ") for line in end), text
    assert out.read_text() == text  # --out receives the same lines


def test_the_sampler_ends_when_the_watched_process_is_gone() -> None:
    watched = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(1.5)"])
    p = sample(watch=watched)
    watched.wait(10)  # reaped: a zombie still has its pid
    text, _ = p.communicate(timeout=20)
    assert p.returncode == 0 and "[end] the watched process is gone" in text, text
