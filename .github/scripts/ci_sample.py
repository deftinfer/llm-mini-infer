"""ci.yml's memory sampler, beside the pytest step's tests: the runner's memory, swap, /dev/shm and the Python
processes the step started, a line a minute in the step's log, so that a run the runner ends (its shutdown signal when
memory and swap are both full) leaves its last state there. Usage, from the step's shell, on the runner's
own Python (the standard library and Linux's /proc; elsewhere the lines are empty):

    python3 .github/scripts/ci_sample.py --watch $$ --stop FILE [--dump-at FILE] &

It ends when FILE (--stop) appears, with a summary (the peaks, the Python processes and /dev/shm files left, and a
warning annotation when memory ran low), or when the watched process is gone.

    [mem]  the first line: the sampler's settings and the runner's size, mem_total and swap_total (MB) and cpus;
           then every --every seconds (every --poll seconds under --low MB available; the peaks take every sample):
           +seconds, then MB: used (total - available), avail, free, shmem, swap used, the step's cgroup (current/peak,
           when it has them), /dev/shm (MB/files); psi (memory pressure, some/full avg10); load; py (the Python
           processes started since the sampler, and their RSS sum), srv (server launchers), orph (those whose parent is
           init); oom (the kernel's OOM kills since the start); major faults and swap pages in/out since the last line
    [top]  under --low MB available, every 15 s: the processes with the most resident memory (pid, parent, session,
           RSS, PSS and swap in MB, age, state), and [shm], /dev/shm's files
    [hang] --dump-at's file holds an epoch second, a limit about to end the tests: from then, once, the process forest
"""

from __future__ import annotations

import argparse
import contextlib
import os
import subprocess
import sys
import time
from typing import NamedTuple

PAGE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096
TICK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
SHM = "/dev/shm"
MB = 1 << 20
TOP_LOW = 14.0  # seconds between two [top] tables while memory is low: every third sample at the default --poll


def read(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def table(path: str) -> dict[str, int]:
    """A file of `key: number [kB]` or `key number` lines (meminfo, vmstat, smaps_rollup)."""
    out = {}
    for line in read(path).splitlines():
        parts = line.replace(":", " ").split()
        if len(parts) >= 2 and parts[1].isdigit():
            out[parts[0]] = int(parts[1])
    return out


class Proc(NamedTuple):
    pid: int
    ppid: int
    sid: int
    state: str
    rss: int  # bytes
    start: int  # clock ticks since boot
    python: bool
    cmd: str

    @property
    def key(self) -> tuple[int, int]:
        return self.pid, self.start


def processes() -> list[Proc]:
    out = []
    for name in os.listdir("/proc") if os.path.isdir("/proc") else ():
        if not name.isdigit():
            continue
        head, _, tail = read(f"/proc/{name}/stat").rpartition(")")  # pid (comm) state ppid pgrp session ...
        rest = tail.split()
        if len(rest) < 22:
            continue
        comm = head.partition("(")[2]
        cmd = read(f"/proc/{name}/cmdline").replace("\0", " ").strip()
        try:
            exe = os.path.basename(os.readlink(f"/proc/{name}/exe"))
        except OSError:
            exe = os.path.basename(cmd.split(" ")[0]) if cmd else ""
        out.append(Proc(int(name), int(rest[1]), int(rest[3]), rest[0], int(rest[21]) * PAGE, int(rest[19]),
                        exe.startswith("python"), cmd or f"[{comm}]"))
    return out


def psi(kind: str) -> str:
    """`some/full` avg10 of /proc/pressure/<kind>."""
    got = {}
    for line in read(f"/proc/pressure/{kind}").splitlines():
        parts = line.split()
        if parts:
            got[parts[0]] = next((p[6:] for p in parts if p.startswith("avg10=")), "?")
    return f"{got.get('some', '?')}/{got.get('full', '?')}"


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # another user's: it exists
        pass
    return True


def shm_files() -> list[tuple[str, int]]:
    out = []
    try:
        for name in sorted(os.listdir(SHM)):
            with contextlib.suppress(OSError):
                out.append((name, os.stat(os.path.join(SHM, name)).st_blocks * 512))
    except OSError:
        pass
    return out


class Sampler:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.t0 = args.t0 or time.time()
        self.out = open(args.out, "a") if args.out else None  # noqa: SIM115
        self.cwd = os.getcwd() + "/"
        self.base = {p.key for p in processes() if p.python}  # the runner's own Python processes, and this one
        self.base_shm = {name for name, _ in shm_files()}
        self.vm = table("/proc/vmstat")
        self.vm0 = dict(self.vm)
        self.peak = {"used": 0, "swap": 0, "shm": 0, "py": 0, "rss": 0, "srv": 0, "orph": 0, "cg": 0}
        self.min_avail: int | None = None
        self.dumped: str | None = None
        cgroup = next((line[3:] for line in read(f"/proc/{args.watch}/cgroup").splitlines() if line.startswith("0::")),
                      "")
        self.cgroup = "/sys/fs/cgroup" + cgroup.rstrip("/")  # a container's root has the files, a host's has not

    def say(self, line: str) -> None:
        data = (line + "\n").encode()
        with contextlib.suppress(OSError):
            os.write(1, data)
        if self.out:
            self.out.write(line + "\n")
            self.out.flush()

    def age(self, p: Proc, uptime: float) -> str:
        s = max(0, int(uptime - p.start / TICK))
        return f"{s // 60}:{s % 60:02d}"

    def short(self, cmd: str, width: int = 130) -> str:
        return cmd.replace(self.cwd, "").replace(".venv/bin/", "")[:width]

    def ours(self, procs: list[Proc]) -> list[Proc]:
        return [p for p in procs if p.python and p.key not in self.base and p.pid != os.getpid()]

    def sample(self, say: bool) -> int:
        """One sample into the peaks, and its [mem] line with `say` or under --low; returns the available MB."""
        mem = table("/proc/meminfo")
        total, avail = mem.get("MemTotal", 0) // 1024, mem.get("MemAvailable", 0) // 1024
        swap = (mem.get("SwapTotal", 0) - mem.get("SwapFree", 0)) // 1024
        files = shm_files()
        shm = sum(size for _, size in files) // MB
        ours = self.ours(processes())
        rss = sum(p.rss for p in ours) // MB
        srv = sum(1 for p in ours if "llm_mini_infer.cli serve" in p.cmd or "llm_mini_infer.entrypoint serve" in p.cmd)
        orph = sum(1 for p in ours if p.ppid == 1)
        vm = table("/proc/vmstat")
        d = {k: vm.get(k, 0) - self.vm.get(k, 0) for k in ("pgmajfault", "pswpin", "pswpout")}
        oom = vm.get("oom_kill", 0) - self.vm0.get("oom_kill", 0)
        cg = ""
        if self.cgroup:
            cur, peak = read(f"{self.cgroup}/memory.current").strip(), read(f"{self.cgroup}/memory.peak").strip()
            if cur.isdigit():
                cg = f" cg={int(cur) // MB}/{int(peak) // MB if peak.isdigit() else '?'}"
                self.peak["cg"] = max(self.peak["cg"], int(peak) // MB if peak.isdigit() else int(cur) // MB)
        for key, value in (("used", total - avail), ("swap", swap), ("shm", shm), ("py", len(ours)), ("rss", rss),
                           ("srv", srv), ("orph", orph)):
            self.peak[key] = max(self.peak[key], value)
        self.min_avail = avail if self.min_avail is None else min(self.min_avail, avail)
        if not say and avail >= self.args.low:
            return avail
        self.vm = vm
        load = read("/proc/loadavg").split(" ")[0]
        free, shmem = mem.get("MemFree", 0) // 1024, mem.get("Shmem", 0) // 1024
        self.say(f"[mem] +{time.time() - self.t0:04.0f}s used={total - avail} avail={avail} free={free}"
                 f" shmem={shmem} swap={swap}{cg} shm={shm}/{len(files)} psi={psi('memory')}"
                 f" load={load} py={len(ours)}/{rss} srv={srv} orph={orph} oom={oom}"
                 f" maj={d['pgmajfault']} in={d['pswpin']} out={d['pswpout']}")
        return avail

    def top(self, count: int) -> None:
        procs = processes()
        uptime = float((read("/proc/uptime") or "0").split()[0])
        pss_sum = 0
        for p in self.ours(procs):
            pss_sum += table(f"/proc/{p.pid}/smaps_rollup").get("Pss", 0)
        self.say(f"[top] pid ppid sid rss pss swap age state command (our Python processes' PSS: {pss_sum // 1024} MB)")
        for p in sorted(procs, key=lambda p: -p.rss)[:count]:
            roll = table(f"/proc/{p.pid}/smaps_rollup")
            self.say(f"[top] {p.pid} {p.ppid} {p.sid} {p.rss // MB} {roll.get('Pss', 0) // 1024}"
                     f" {roll.get('Swap', 0) // 1024} {self.age(p, uptime)} {p.state} {self.short(p.cmd)}")
        for name, size in shm_files():
            self.say(f"[shm] {size // MB} MB {name}")

    def run(self, cmd: list[str], limit: float = 20, lines: int = 400) -> None:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=limit, errors="replace")
            text = r.stdout + r.stderr
        except (OSError, subprocess.TimeoutExpired) as e:
            text = f"{type(e).__name__}: {e}"
        for line in text.splitlines()[:lines]:
            self.say("[hang] " + line[:220])

    def hang(self) -> None:
        """The dump before a limit ends the tests: the process forest, /dev/shm, and the [top] table."""
        self.say(f"[hang] +{time.time() - self.t0:04.0f}s the limit is near: the processes, before it ends them")
        columns = "pid,ppid,pgid,sid,stat,etime,rss,args"
        self.run(["ps", "--ppid", "2", "-p", "2", "--deselect", "-o", columns, "--forest", "-ww"])  # no kernel threads
        self.run(["df", "-m", SHM])
        self.top(self.args.top)

    def end(self) -> None:
        procs = processes()
        uptime = float((read("/proc/uptime") or "0").split()[0])
        vm = table("/proc/vmstat")
        self.say(f"[end] +{time.time() - self.t0:04.0f}s peaks: " + " ".join(f"{k}={v}" for k, v in self.peak.items())
                 + f" min_avail={self.min_avail} oom={vm.get('oom_kill', 0) - self.vm0.get('oom_kill', 0)}"
                 f" swapped_out={(vm.get('pswpout', 0) - self.vm0.get('pswpout', 0)) * PAGE // MB}MB")
        left = self.ours(procs)
        self.say(f"[end] Python processes left: {len(left)}")
        for p in left:
            self.say(f"[end] {p.pid} ppid={p.ppid} sid={p.sid} rss={p.rss // MB} age={self.age(p, uptime)}"
                     f" {self.short(p.cmd)}")
        files = [(name, size) for name, size in shm_files() if name not in self.base_shm]
        self.say(f"[end] /dev/shm files left: {len(files)}, {sum(size for _, size in files) // MB} MB")
        for name, size in files:
            self.say(f"[end] {size // MB} MB {name}")
        if self.min_avail is not None and (self.min_avail < self.args.warn or self.peak["swap"] > self.args.warn):
            # GitHub shows the line as an annotation on the run: the tests have outgrown the runner's memory
            self.say(f"::warning title=memory::{self.min_avail} MB were available at the lowest and {self.peak['swap']}"
                     f" MB of swap were used: past {self.args.warn} MB of either, tests that start servers need the"
                     " serial mark or smaller servers (.github/workflows/ci.yml)")

    def loop(self) -> None:
        a = self.args
        mem = table("/proc/meminfo")
        self.say(f"[mem] sampler: a line every {a.every:g} s of samples every {a.poll:g} s, pid {os.getpid()},"
                 f" uid {os.getuid()}, cgroup {self.cgroup}, mem_total={mem.get('MemTotal', 0) // 1024}"
                 f" cpus={os.cpu_count()} swap_total={mem.get('SwapTotal', 0) // 1024}")
        next_sample = next_line = time.time()
        last_top = 0.0
        while True:
            now = time.time()
            if a.stop and os.path.exists(a.stop):
                self.sample(True)
                self.end()
                return
            if not alive(a.watch):
                self.say("[end] the watched process is gone")
                self.end()
                return
            if now >= next_sample:
                next_sample = max(next_sample + min(a.poll, a.every), now)
                line = now >= next_line
                if line:
                    next_line = max(next_line + a.every, now)
                avail = self.sample(line)
                low = avail < a.low
                if (a.top_every and now - last_top >= a.top_every) or (low and now - last_top >= TOP_LOW):
                    last_top = now
                    self.top(a.top)
            if a.dump_at:  # every turn: the limit it announces is 30 s away
                at = read(a.dump_at).strip()
                if at.isdigit() and at != self.dumped and now >= int(at):
                    self.dumped = at
                    self.hang()
            time.sleep(0.5)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--watch", type=int, required=True, help="the step's shell: the sampler ends with it")
    ap.add_argument("--stop", help="a file whose appearance ends the sampler, with its summary")
    ap.add_argument("--every", type=float, default=60, help="seconds between [mem] lines")
    ap.add_argument("--poll", type=float, default=5, help="seconds between samples (the peaks; a line each when low)")
    ap.add_argument("--top-every", type=float, default=0, help="seconds between [top] tables (0: only when low)")
    ap.add_argument("--top", type=int, default=15, help="the [top] table's rows")
    ap.add_argument("--low", type=int, default=400, help="available MB under which every sample prints, with [top]")
    ap.add_argument("--warn", type=int, default=1024, help="the summary warns under this many MB available, or over "
                                                           "it of swap used")
    ap.add_argument("--dump-at", help="a file holding the epoch second from which to dump the processes once")
    ap.add_argument("--out", help="a file that receives the lines too")
    ap.add_argument("--t0", type=float, help="the epoch second that +seconds count from (default: now)")
    args = ap.parse_args()
    try:
        Sampler(args).loop()
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
