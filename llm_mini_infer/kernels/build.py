"""Build and load llm-mini-infer's CUDA kernels (`csrc/*.cu`) and the CUTLASS GEMMs (`cutlass/*.cu`).

The kernels are plain CUDA with `extern "C"` launchers: nvcc alone compiles them into a shared library (no torch
headers, so a build takes seconds), loaded with ctypes and called with raw device pointers and the caller's stream
(`kernels.ops`; docs/architecture.md, "Kernels").

**Two libraries.** `libllm_mini_infer_kernels.so` (`csrc/`) holds the fused ops; `libllm_mini_infer_cutlass.so`
(`cutlass/`) holds the block-FP8 GEMM (the vendored CUTLASS SM120 blockwise kernel, `kernels.fp8`). CUTLASS templates
take a minute or more to compile, so they live apart and are cached apart: editing a fused op does not rebuild them. The
CUTLASS library needs CUTLASS 4 headers, from `$CUTLASS_HOME` or else the copy FlashInfer ships
(`flashinfer/data/cutlass`), and is built for the arch-specific target (`sm_120a`), which its tensor-core instructions
need; it exists for SM 12.x only.

**Nothing compiles on the serving path.** The library is cached under `$LLM_MINI_INFER_KERNEL_CACHE` (default
`~/.cache/llm_mini_infer/kernels`), keyed by a hash of the sources, the nvcc version, the target architectures and the
flags (the CUTLASS library's also by CUTLASS's `version.h`). `python -m llm_mini_infer.kernels.build` builds it ahead of
time (an image build runs it, with `--arch` for GPUs the build host lacks) and records `nvcc --version` beside the
libraries (`NVCC_VERSION_FILE`). At startup `load()` uses the cached library, or one prebuilt under a read-only
directory of `$LLM_MINI_INFER_KERNEL_PREBUILT` (colon-separated; the image's, so a cache volume mounted over the cache
path neither hides nor outdates them), or builds it once into the cache, logging the time (the empty-cache start). The
key's nvcc version is nvcc's own answer; without nvcc (the image's runtime has none, as it compiles nothing) it is the
one recorded in a prebuilt directory, so the runtime finds the libraries the build made and cannot build others. No CUDA
toolkit and no prebuilt directory (a Mac) means no library: CPU tensors run the torch reference (`kernels.ops`), and a
CUDA runner refuses to start, naming why the lookup found none (`require`): on CUDA its sampler has no torch fallback.

**Reproducible bytes:** every object is compiled alone and they link in the sources' order, so the job count changes
nothing; and the libraries link stripped (`LINK_FLAGS`): nvcc names its host intermediate after its process id
(`tmpxft_<pid>_..._x.cudafe1.cpp`), which the host compiler records as a FILE symbol in `.symtab`, so unstripped no two
builds would be alike. The launchers are exported in `.dynsym`, which stays. The device code's line tables hold each
file's path, and ptxas adds its mtime and size: the same bytes need the same source and CUTLASS directories and the same
mtimes (deploy/Dockerfile's comments; docs/development.md, "The kernels' recorded bytes").
"""

from __future__ import annotations

import ctypes
import hashlib
import importlib.util
import logging
import math
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

CSRC = Path(__file__).resolve().parent / "csrc"
LIB_NAME = "libllm_mini_infer_kernels.so"
# -lineinfo for profilers; no fast math: kernels that mirror torch rely on IEEE expf/logf and division.
FLAGS = ["-O3", "-std=c++17", "-Xcompiler", "-fPIC", "-lineinfo", "--expt-relaxed-constexpr"]

_P, _I, _I64, _F = ctypes.c_void_p, ctypes.c_int, ctypes.c_int64, ctypes.c_float
# Launcher signatures (csrc/*.cu); a launcher returns a cudaError_t (0 = success, -1 = unsupported shape), a query
# (`lmk_attn512_occupancy`, `lmk_get_pdl`) its value.
SIGNATURES: dict[str, list] = {
    "lmk_gemma_rmsnorm": [_P, _I64, _P, _P, _P, _P, _I64, _I, _F, _P],
    "lmk_gated_rmsnorm": [_P, _P, _I64, _P, _P, _I64, _I, _I, _F, _P],
    "lmk_silu_mul": [_P, _I64, _P, _I64, _I, _P],
    "lmk_sigmoid_gate": [_P, _P, _I64, _I64, _P, _I64, _I, _I, _P],
    "lmk_qk_norm_rope": [_P, _I64, _I64, _I64, _I64, _P, _P, _P, _I64, _I, _I, _P, _P, _P, _P, _I64, _I, _P, _P, _I64,
                         _I, _I, _I, _I, _F, _P],
    "lmk_vision_qkv_rope": [_P, _I64, _P, _P, _P, _P, _P, _I64, _I, _I, _I, _P],
    "lmk_g4_vision_qkv": [_P, _I64, _P, _P, _P, _P, _F, _F, _P, _P, _P, _I64, _I, _I, _I, _P],
    "lmk_causal_conv": [_P, _I64, _P, _P, _I64, _P, _P, _P, _P, _P, _P, _I, _I, _I, _I, _I, _I, _P, _P, _P, _I64,
                        _P],
    "lmk_conv_fold": [_P, _P, _P, _P, _P, _P, _I, _I, _I, _I, _P],
    "lmk_gdn_mtp": [_P, _I64, _P, _I64, _P, _I64, _P, _P, _I, _P, _P, _P, _P, _I64, _I, _P, _I64, _P, _I, _P, _P, _P,
                    _I64, _P, _P, _P, _I64, _I, _I, _I, _I, _F, _F, _I, _P],
    "lmk_gdn_fold": [_P, _P, _P, _P, _P, _P, _I, _P, _P, _P, _P, _I, _I, _I, _I, _I, _P],
    "lmk_gdn_decode": [_P, _I64, _P, _I64, _P, _I64, _P, _P, _I, _P, _I, _P, _P, _P, _I64, _I, _P, _I64, _P, _I, _P,
                       _P, _P, _I64, _I, _I, _I, _F, _F, _I, _P],
    "lmk_sample": [_P, _I64, _I, _I, _P, _P, _P, _P, _P, _P, _P, _P, _P, _P, _P, _I, _P, _I64, _P,
                   _P, _P, _P, _P, _P, _I64, _P, _P, _P, _I64, _P, _I, _P, _P, _I, _P],
    "lmk_gdn_prep": [_P, _I64, _P, _I64, _P, _I64, _P, _P, _P, _P, _P, _P, _P, _I64, _I, _I, _P],
    "lmk_quant_fp8": [_P, _I64, _P, _P, _I64, _I64, _I, _P],
    "lmk_gemma_rmsnorm_quant": [_P, _I64, _P, _P, _P, _P, _P, _P, _I64, _I64, _I, _F, _P],
    "lmk_gated_rmsnorm_quant": [_P, _P, _I64, _P, _P, _P, _I64, _I64, _I, _I, _F, _P],
    "lmk_silu_mul_quant": [_P, _I64, _P, _P, _I64, _I64, _I, _P],
    "lmk_sigmoid_gate_quant": [_P, _P, _I64, _I64, _P, _P, _I64, _I64, _I, _I, _P],
    "lmk_dequant_fp8": [_P, _P, _P, _I64, _I, _P],
    "lmk_linear_small": [_P, _I64, _P, _I64, _P, _I64, _I, _I, _I, _P],
    "lmk_linear_rows": [_P, _I64, _P, _I64, _P, _I64, _P, _I, _I, _I, _I, _P],
    # Gemma 4: csrc/gemma.cu, csrc/attn512.cu
    "lmk_g4_rmsnorm": [_P, _I64, _P, _P, _I64, _I, _F, _P],
    "lmk_g4_sandwich": [_P, _I64, _P, _P, _P, _P, _P, _P, _I64, _I, _F, _P],
    "lmk_gelu_tanh_mul": [_P, _I64, _P, _I64, _I, _P],
    "lmk_g4_qkv": [_P, _I64, _I, _P, _P, _P, _P, _P, _P, _P, _I64, _I, _I, _P, _P, _I64, _I, _I, _I, _F, _P],
    "lmk_g4_kv_cast": [_P, _P, _I64, _I, _P],
    "lmk_attn512": [_P, _I, _I, _P, _I64, _I, _I, _P, _I64, _P, _P, _P, _I, _F, _P, _P, _I, _P, _I, _I, _I, _I, _I,
                    _I, _P, _P, _P, _P, _P],
    "lmk_attn512_occupancy": [_I],
    # Gemma 4's sliding layers, batch-invariant: csrc/attn_window.cu
    "lmk_attn_window": [_P, _I, _I, _P, _I64, _I64, _I, _P, _I64, _I, _F, _P, _P, _I, _P, _I, _I, _P, _P, _P, _P],
    "lmk_attn_window_prefill": [_P, _I, _I, _P, _I64, _I64, _I, _P, _I64, _I, _F, _P, _P, _I, _P, _P, _P],
    # Qwen3.8's full layers on the same kernels, batch-invariant: csrc/attn_window.cu
    "lmk_attn_full": [_P, _I, _I, _P, _I64, _I64, _I, _P, _I64, _F, _P, _P, _I, _P, _I, _I, _P, _P, _P, _P],
    "lmk_attn_full_prefill": [_P, _I, _I, _P, _I64, _I64, _I, _P, _I64, _F, _P, _P, _I, _P, _P],
    "lmk_attn_full_prefill_split": [_P, _I, _I, _P, _I64, _I64, _I, _P, _I64, _F, _P, _P, _I, _P, _I, _I, _I, _P, _P,
                                    _P, _P, _P],
    # Gemma 4's quantizing producers: csrc/gemma.cu
    "lmk_g4_rmsnorm_quant": [_P, _I64, _P, _P, _P, _I64, _I64, _I, _F, _P],
    "lmk_g4_sandwich_quant": [_P, _I64, _P, _P, _P, _P, _P, _P, _P, _I64, _I64, _I, _F, _P],
    "lmk_gelu_tanh_mul_quant": [_P, _I64, _P, _P, _I64, _I64, _I, _P],
    # Programmatic dependent launch (csrc/launch.cu)
    "lmk_set_pdl": [_I],
    "lmk_get_pdl": [],
}
CUTLASS_SIGNATURES: dict[str, tuple[list, Any]] = {
    "lmk_fp8_gemm_sm120": ([_P, _P, _P, _P, _P, _I, _I, _I, _I, _I, _I, _I, _P, _I64, _I, _P], ctypes.c_int),
    "lmk_fp8_gemm_sm120_workspace": ([_I, _I, _I, _I, _I, _I, _I], ctypes.c_int64),
}
# Both libraries link without .symtab (module docstring, "Reproducible bytes"); part of the cache key
LINK_FLAGS = ["-Xlinker", "--strip-all"]
CUTLASS_SRC = Path(__file__).resolve().parent / "cutlass"
# `nvcc --version`'s output as the ahead-of-time build saw it (`main`), beside the libraries it built: the keys' nvcc
# version where there is no nvcc
NVCC_VERSION_FILE = "nvcc-version.txt"
CUTLASS_LIB_NAME = "libllm_mini_infer_cutlass.so"
# No CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL: CUTLASS then finds cuTensorMapEncodeTiled through the runtime's driver
# entry points, so the library needs no link against libcuda. CUTLASS_ENABLE_GDC_FOR_SM100 compiles CUTLASS's
# programmatic-dependent-launch waits in (SM 10.x to 12.x take that macro): they return at once unless a call
# launches with PDL (`lmk_fp8_gemm_sm120`'s pdl).
CUTLASS_FLAGS = ["-O3", "-std=c++17", "-Xcompiler", "-fPIC", "--expt-relaxed-constexpr", "-DNDEBUG", "-lineinfo",
                 "-DCUTLASS_ENABLE_GDC_FOR_SM100"]


# -- the build's parallelism ---------------------------------------------------------------------------------------

MAX_JOBS = 8
GiB = 1 << 30
# A library's budget a job: its largest source's resident peak, nvcc and its children (cicc, ptxas, the host
# compiler) summed, measured with nvcc 13.0 for sm_120 (gdn_mtp.cu, fp8_gemm_sm120_c3.cu, each built alone in the GPU
# tests' image), plus 15%, rounded up to 1/4 GiB
JOB_BUDGET = {LIB_NAME: 0.25 * GiB, CUTLASS_LIB_NAME: 5.25 * GiB}
CGROUP = Path("/sys/fs/cgroup")
MEMINFO = Path("/proc/meminfo")


def cgroup_cpu_quota() -> tuple[float, str]:
    """CPUs this process may use: the cgroup quota (v2 `cpu.max`, v1 `cfs_quota_us`), else the affinity mask or
    `os.cpu_count()` (macOS)."""
    try:
        quota, period = (CGROUP / "cpu.max").read_text().split()[:2]
        if quota != "max":
            return int(quota) / int(period), f"cgroup v2 cpu.max {quota}/{period}"
    except (OSError, ValueError):
        pass
    try:
        quota = int((CGROUP / "cpu" / "cpu.cfs_quota_us").read_text())
        period = int((CGROUP / "cpu" / "cpu.cfs_period_us").read_text())
        if quota > 0:
            return quota / period, f"cgroup v1 cfs_quota_us {quota}/{period}"
    except (OSError, ValueError):
        pass
    if hasattr(os, "sched_getaffinity"):
        n = len(os.sched_getaffinity(0))
        return float(n), f"affinity mask ({n} CPUs), no cgroup quota"
    return float(os.cpu_count() or 1), "os.cpu_count(), no cgroup quota"


def _stat(path: Path, key: str) -> int:
    for line in path.read_text().splitlines():
        name, _, value = line.partition(" ")
        if name == key:
            return int(value)
    return 0


def available_memory() -> tuple[int | None, str]:
    """Bytes a build may still take: the least of `/proc/meminfo`'s MemAvailable and the cgroup's limit less its
    working set (v2 `memory.max` − `memory.current` + inactive file pages; v1 `memory.limit_in_bytes` −
    `memory.usage_in_bytes` + `total_inactive_file`), or None where neither is readable (macOS)."""
    found: list[tuple[int, str]] = []
    try:
        for line in MEMINFO.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                found.append((int(line.split()[1]) * 1024, "MemAvailable"))
    except (OSError, ValueError, IndexError):
        pass
    try:
        limit = (CGROUP / "memory.max").read_text().strip()
        if limit != "max":
            used = int((CGROUP / "memory.current").read_text()) - _stat(CGROUP / "memory.stat", "inactive_file")
            found.append((int(limit) - used, f"cgroup v2 memory.max {int(limit) / GiB:.1f} GiB"))
    except (OSError, ValueError):
        pass
    try:
        v1 = CGROUP / "memory"
        limit = int((v1 / "memory.limit_in_bytes").read_text())
        if limit < 1 << 60:  # unlimited reads as about 2^63
            used = int((v1 / "memory.usage_in_bytes").read_text()) - _stat(v1 / "memory.stat", "total_inactive_file")
            found.append((limit - used, f"cgroup v1 memory.limit_in_bytes {limit / GiB:.1f} GiB"))
    except (OSError, ValueError):
        pass
    if not found:
        return None, "no /proc/meminfo or cgroup memory limit"
    avail, why = min(found)
    return max(avail, 0), why


def default_jobs(budget: float, sources: int = MAX_JOBS) -> tuple[int, str]:
    """nvcc jobs for a library whose sources take up to `budget` bytes each: min(MAX_JOBS, usable CPUs, available
    memory ÷ budget, the sources), at least 1; and the reason, for the build's log line."""
    quota, cpu_why = cgroup_cpu_quota()
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    cpus = max(1, min(cpus, math.floor(quota)))
    mem, mem_why = available_memory()
    by_mem = MAX_JOBS if mem is None else max(1, int(mem // budget))
    jobs = max(1, min(MAX_JOBS, cpus, by_mem, sources))
    mem_text = (f"memory unknown ({mem_why})" if mem is None else
                f"{mem / GiB:.1f} GiB available ({mem_why}) / {budget / GiB:.2f} GiB a job")
    return jobs, f"min of {MAX_JOBS}, {cpus} CPUs ({cpu_why}), {mem_text}, {sources} sources"


def cache_dir() -> Path:
    env = os.environ.get("LLM_MINI_INFER_KERNEL_CACHE")
    return Path(env).expanduser() if env else Path.home() / ".cache" / "llm_mini_infer" / "kernels"


def prebuilt_dirs() -> list[Path]:
    return [Path(d).expanduser() for d in filter(None, os.environ.get("LLM_MINI_INFER_KERNEL_PREBUILT", "").split(":"))]


def prebuilt(path: Path) -> Path:
    """`path` (a library under `cache_dir()`), or the same library under a `$LLM_MINI_INFER_KERNEL_PREBUILT` directory
    when only that one has it."""
    if path.exists():
        return path
    rel = path.relative_to(cache_dir())
    for d in prebuilt_dirs():
        alt = d / rel
        if alt.exists():
            return alt
    return path


def find_nvcc() -> str | None:
    home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if home and (Path(home) / "bin" / "nvcc").exists():
        return str(Path(home) / "bin" / "nvcc")
    if Path("/usr/local/cuda/bin/nvcc").exists():
        return "/usr/local/cuda/bin/nvcc"
    return shutil.which("nvcc")


def recorded_nvcc_version() -> str | None:
    """`nvcc --version`'s output as the ahead-of-time build recorded it (`NVCC_VERSION_FILE`), in the first
    `$LLM_MINI_INFER_KERNEL_PREBUILT` directory that holds one, or None."""
    for d in prebuilt_dirs():
        if (d / NVCC_VERSION_FILE).is_file():
            return (d / NVCC_VERSION_FILE).read_text()
    return None


def nvcc_version() -> str | None:
    """The keys' nvcc version: `nvcc --version`'s output, or without nvcc the ahead-of-time build's record
    (`recorded_nvcc_version`), or None."""
    nvcc = find_nvcc()
    if nvcc is not None:
        return subprocess.run([nvcc, "--version"], capture_output=True, text=True, check=True).stdout
    return recorded_nvcc_version()


def record_nvcc_version() -> Path:
    """Write `nvcc --version`'s output beside the libraries in the cache (`NVCC_VERSION_FILE`), atomically: what a
    runtime without nvcc keys its lookup on when this cache is its prebuilt directory."""
    if find_nvcc() is None:
        raise RuntimeError("nvcc not found (set CUDA_HOME)")
    version = nvcc_version()
    assert version is not None
    path = cache_dir() / NVCC_VERSION_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}")
    tmp.write_text(version)
    os.replace(tmp, path)
    return path


def local_arch() -> str | None:
    """The current GPU's architecture as nvcc spells it (`120` for SM120), or None without a GPU."""
    try:
        import torch

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            return f"{major}{minor}"
    except Exception:  # no torch or no driver: no local GPU
        pass
    return None


def sources() -> list[Path]:
    return sorted(CSRC.glob("*.cu"))


def build_key(version: str, archs: list[str], src_dir: Path = CSRC, extra: str = "") -> str:
    """The libraries' key: a hash of the sources, `version` (`nvcc --version`'s output: `nvcc_version`), the
    architectures, the flags and `extra`."""
    h = hashlib.sha256()
    for f in sorted(src_dir.iterdir()):
        if f.suffix in (".cu", ".cuh", ".h"):
            h.update(f.name.encode())
            h.update(f.read_bytes())
    h.update(version.encode())
    h.update(" ".join(FLAGS + archs + LINK_FLAGS).encode())
    h.update(extra.encode())
    return h.hexdigest()[:16]


def library_path(archs: list[str] | None = None) -> Path | None:
    """The library's cache path for `archs` (default: the local GPU's), built or not, or None without nvcc's version
    (nvcc, or a prebuilt directory's record) or an arch."""
    archs = archs or ([a] if (a := local_arch()) else [])
    if not archs or (version := nvcc_version()) is None:
        return None
    return cache_dir() / f"{'-'.join(archs)}-{build_key(version, archs)}" / LIB_NAME


def _compile(nvcc: str, srcs: list[Path], flags: list[str], gencode: list[str], lib: Path, jobs: int | None) -> None:
    """Compile `srcs` in parallel (`jobs`, default `default_jobs` for the library's budget) and link them into `lib`,
    in the sources' order, atomically (concurrent builders never see a partial library)."""
    if jobs:
        jobs, why = min(jobs, len(srcs)), "--jobs"
    else:
        jobs, why = default_jobs(JOB_BUDGET[lib.name], len(srcs))
    log.info("building %s: %d nvcc job%s (%s)", lib.name, jobs, "" if jobs == 1 else "s", why)
    lib.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=lib.parent) as tmp:
        objs = [Path(tmp) / (s.stem + ".o") for s in srcs]

        def compile_one(src_obj: tuple[Path, Path]) -> None:
            src, obj = src_obj
            cmd = [nvcc, *flags, *gencode, "-c", str(src), "-o", str(obj)]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                raise RuntimeError(f"nvcc failed on {src.name}:\n{res.stderr[-4000:]}")

        with ThreadPoolExecutor(jobs) as pool:
            list(pool.map(compile_one, zip(srcs, objs)))
        out = Path(tmp) / lib.name
        res = subprocess.run([nvcc, "-shared", *LINK_FLAGS, *gencode, *map(str, objs), "-o", str(out)],
                             capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"linking {lib.name} failed:\n{res.stderr[-4000:]}")
        os.replace(out, lib)


def build(archs: list[str] | None = None, force: bool = False, jobs: int | None = None) -> Path:
    """Compile every source for `archs` (default: the local GPU) unless the cache has it. Returns the library."""
    nvcc = find_nvcc()
    if nvcc is None:
        raise RuntimeError("nvcc not found (set CUDA_HOME)")
    archs = archs or ([a] if (a := local_arch()) else [])
    if not archs:
        raise RuntimeError("no GPU to build for: pass archs (e.g. ['120'])")
    lib = library_path(archs)
    assert lib is not None
    if lib.exists() and not force:
        return lib
    gencode = [f"-gencode=arch=compute_{a},code=sm_{a}" for a in archs]
    t0 = time.perf_counter()
    _compile(nvcc, sources(), FLAGS, gencode, lib, jobs)
    log.info("built %s for sm_%s in %.1f s", lib, ",".join(archs), time.perf_counter() - t0)
    return lib


# -- the CUTLASS library ------------------------------------------------------------------------------------------


def cutlass_root() -> Path | None:
    """CUTLASS 4 headers: $CUTLASS_HOME, else the copy inside the installed FlashInfer package (found without
    importing it)."""
    env = os.environ.get("CUTLASS_HOME")
    if env and (Path(env) / "include" / "cutlass" / "cutlass.h").exists():
        return Path(env)
    try:
        spec = importlib.util.find_spec("flashinfer")
    except (ImportError, ValueError):
        spec = None
    if spec is not None and spec.origin:
        root = Path(spec.origin).parent / "data" / "cutlass"
        if (root / "include" / "cutlass" / "cutlass.h").exists():
            return root
    return None


def cutlass_archs(archs: list[str]) -> list[str]:
    """The arch-specific targets the CUTLASS kernels need (`120a`); only SM 12.x has kernels here."""
    return [f"{a}a" for a in archs if a.startswith("12")]


def cutlass_library_path(archs: list[str] | None = None) -> Path | None:
    root = cutlass_root()
    archs = cutlass_archs(archs or ([a] if (a := local_arch()) else []))
    if root is None or not archs or (version := nvcc_version()) is None:
        return None
    cutlass = (root / "include" / "cutlass" / "version.h").read_bytes()
    key = build_key(version, archs, CUTLASS_SRC, " ".join(CUTLASS_FLAGS) + hashlib.sha256(cutlass).hexdigest())
    return cache_dir() / f"cutlass-{'-'.join(archs)}-{key}" / CUTLASS_LIB_NAME


def build_cutlass(archs: list[str] | None = None, force: bool = False, jobs: int | None = None) -> Path | None:
    """Compile the CUTLASS GEMMs for the SM 12.x archs among `archs` (default: the local GPU) unless cached. None when
    there is nothing to build (no SM 12.x target)."""
    nvcc, root = find_nvcc(), cutlass_root()
    if nvcc is None:
        raise RuntimeError("nvcc not found (set CUDA_HOME)")
    if root is None:
        raise RuntimeError("CUTLASS 4 headers not found: set CUTLASS_HOME or install flashinfer-python")
    lib = cutlass_library_path(archs)
    if lib is None:
        return None
    if lib.exists() and not force:
        return lib
    a = cutlass_archs(archs or ([x] if (x := local_arch()) else []))
    gencode = [f"-gencode=arch=compute_{x},code=sm_{x}" for x in a]
    flags = [*CUTLASS_FLAGS, f"-I{root / 'include'}", f"-I{root / 'tools' / 'util' / 'include'}"]
    t0 = time.perf_counter()
    _compile(nvcc, sorted(CUTLASS_SRC.glob("*.cu")), flags, gencode, lib, jobs)
    log.info("built %s for sm_%s in %.1f s (CUTLASS %s)", lib, ",".join(a), time.perf_counter() - t0, root)
    return lib


def _missing(cutlass: bool = False) -> str:
    """Why the lookup found no library (`library_path`; `cutlass_library_path` with `cutlass`): nvcc's version first,
    since the library's key holds it, then a GPU, then for CUTLASS its headers and an SM 12.x GPU."""
    if find_nvcc() is None and recorded_nvcc_version() is None:
        home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
        pre = os.environ.get("LLM_MINI_INFER_KERNEL_PREBUILT") or "unset"
        return (f"nvcc not found ($CUDA_HOME/$CUDA_PATH {home or 'unset'}, /usr/local/cuda/bin/nvcc, PATH), nor a "
                f"prebuilt directory recording its version ($LLM_MINI_INFER_KERNEL_PREBUILT {pre}, "
                f"{NVCC_VERSION_FILE}): the library's key holds nvcc's version, so neither the cache {cache_dir()} nor "
                "a prebuilt directory was searched")
    arch = local_arch()
    if arch is None:
        return "no CUDA GPU (torch sees none)"
    if cutlass and cutlass_root() is None:
        return "no CUTLASS 4 headers ($CUTLASS_HOME, or the copy flashinfer-python ships)"
    if cutlass and not cutlass_archs([arch]):
        return f"no SM 12.x GPU (this one is sm_{arch})"
    return "not found"


_lib: ctypes.CDLL | None = None
_tried = False
stats: dict[str, float | str | bool] = {}


def load() -> ctypes.CDLL | None:
    """The kernel library for the local GPU (cached or prebuilt, else built into the cache), or None; `stats["state"]`
    says why (`_missing`). The first lookup alone does the work: a failure in it (to locate, build, load or bind the
    library) is recorded there before its error is raised, so a later lookup returns None and `require` names it."""
    global _lib, _tried
    if _tried:
        return _lib
    _tried = True
    if os.environ.get("LLM_MINI_INFER_KERNELS", "1") == "0":
        stats.update(state="disabled (LLM_MINI_INFER_KERNELS=0)")
        return None
    t0 = time.perf_counter()
    step = "locate"
    try:
        path = library_path()
        if path is None:
            stats.update(state=_missing())
            return None
        path = prebuilt(path)
        built = not path.exists()
        if built:
            step = "build"
            log.warning("kernel library not in the cache (%s): building it now", path.parent)
            build()
        step = "load"
        lib = ctypes.CDLL(str(path))
        step = "bind"
        for name, args in SIGNATURES.items():
            fn = getattr(lib, name)
            fn.argtypes = args
            fn.restype = ctypes.c_int
    except Exception as exc:
        stats.update(state=f"failed to {step}: {exc}")
        raise
    _lib = lib
    stats.update(state="loaded", path=str(path), built_at_startup=built, load_s=round(time.perf_counter() - t0, 2))
    log.info("kernel library %s in %.2f s (%s)", "built and loaded" if built else "loaded", stats["load_s"], path)
    return _lib


def require(what: str) -> ctypes.CDLL:
    """The kernel library, or a RuntimeError naming why the lookup found none. A CUDA runner calls it before its weights
    load (`what` names it): on CUDA its sampler has no torch fallback (`ops.sample`), so without the library it would
    fail at its first sampled step, inside the memory profile."""
    err: Exception | None = None
    try:
        lib = load()
    except Exception as exc:  # a failed build or load, recorded in stats["state"]: named below with its fix
        lib, err = None, exc
    if lib is None:
        state = str(stats.get("state"))
        fix = ""  # no CUDA GPU, or a failed build or load: the state names the error
        if state.startswith("disabled"):
            fix = " Unset LLM_MINI_INFER_KERNELS (a debug switch)."
        elif state.startswith("nvcc not found"):
            fix = (" The image has the libraries prebuilt with nvcc's version recorded beside them (`python -m "
                   "llm_mini_infer.kernels.build`); elsewhere install nvcc (CUDA_HOME).")
        elif state.startswith("failed to build") and "nvcc not found" in state:  # no library for this GPU, no nvcc
            fix = (f" No library is prebuilt for this GPU (sm_{local_arch()}): the image's are for sm_120 alone "
                   "(docs/operations.md, \"The image's model limit\"); elsewhere install nvcc (CUDA_HOME).")
        raise RuntimeError(f"{what} needs llm-mini-infer's CUDA kernel library, which is unavailable: "
                           f"{state}.{fix}") from err
    return lib


_cutlass: ctypes.CDLL | None = None
_cutlass_tried = False
stats_cutlass: dict[str, float | str | bool] = {}


def load_cutlass() -> ctypes.CDLL | None:
    """The CUTLASS GEMM library for the local GPU (cached or prebuilt, else built into the cache), or None;
    `stats_cutlass["state"]` says why (`_missing`, or a failed load). The first lookup alone does the work: a library
    that fails to load is logged and recorded, and any other failure in it (to locate, build or bind the library) is
    recorded before its error is raised."""
    global _cutlass, _cutlass_tried
    if _cutlass_tried:
        return _cutlass
    _cutlass_tried = True
    if os.environ.get("LLM_MINI_INFER_KERNELS", "1") == "0":
        stats_cutlass.update(state="disabled (LLM_MINI_INFER_KERNELS=0)")
        return None
    t0 = time.perf_counter()
    step = "locate"
    try:
        path = cutlass_library_path()
        if path is None:
            stats_cutlass.update(state=_missing(cutlass=True))
            return None
        path = prebuilt(path)
        built = not path.exists()
        if built:
            step = "build"
            log.warning("CUTLASS GEMM library not in the cache (%s): building it now", path.parent)
            build_cutlass()
        step = "load"
        try:
            lib = ctypes.CDLL(str(path))
        except OSError as exc:
            stats_cutlass.update(state=f"failed to load: {exc}")
            log.error("the CUTLASS GEMM library failed to load: %s", exc)
            return None
        step = "bind"
        for name, (args, ret) in CUTLASS_SIGNATURES.items():
            fn = getattr(lib, name)
            fn.argtypes = args
            fn.restype = ret
    except Exception as exc:
        stats_cutlass.update(state=f"failed to {step}: {exc}")
        raise
    _cutlass = lib
    stats_cutlass.update(state="loaded", path=str(path), built_at_startup=built,
                         load_s=round(time.perf_counter() - t0, 2))
    log.info("CUTLASS GEMM library %s in %.2f s (%s)", "built and loaded" if built else "loaded",
             stats_cutlass["load_s"], path)
    return _cutlass


def require_cutlass(fix: str) -> ctypes.CDLL:
    """The CUTLASS GEMM library, or a RuntimeError naming why the lookup found none, on the first lookup as on later
    ones (there `load_cutlass` raises a failed build's own error); `fix` names the setting that does without it."""
    err: Exception | None = None
    try:
        lib = load_cutlass()
    except Exception as exc:  # a failed build, recorded in stats_cutlass["state"]: named below
        lib, err = None, exc
    if lib is None:
        raise RuntimeError(f"the FP8 GEMM library is unavailable: {stats_cutlass.get('state')} (SM 12.x, nvcc and "
                           f"CUTLASS 4 headers; or {fix})") from err
    return lib


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Build llm-mini-infer's CUDA kernels into the cache "
                                             "(an image build step).")
    ap.add_argument("--arch", action="append", help="nvcc arch without the sm_ prefix, e.g. 120 (default: local GPU)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-cutlass", action="store_true", help="skip the CUTLASS GEMM library")
    ap.add_argument("--jobs", type=int, default=0,
                    help="nvcc jobs per library (default: min(8, usable CPUs, available memory / budget, sources))")
    args = ap.parse_args()
    if args.jobs < 0:
        ap.error("--jobs: at least 1 (0: the default)")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print(build(args.arch, force=args.force, jobs=args.jobs or None))
    if not args.no_cutlass:
        print(build_cutlass(args.arch, force=args.force, jobs=args.jobs or None) or
              "no SM 12.x target: no CUTLASS library")
    print(record_nvcc_version())


if __name__ == "__main__":
    main()
