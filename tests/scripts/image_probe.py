"""The slim image's facts, gathered inside it as its own user, without a GPU or a network:
tests/deploy/test_image_checks.py runs this with `docker run` and checks what it prints (one JSON object). Standard
library and llm_mini_infer only: it runs on the image's Python.

    python3 -I -B /checks/image_probe.py --model-data /model-data     # Gemma 4's configuration files (tests/data)
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import importlib.metadata as md
import importlib.util
import json
import os
import pkgutil
import re
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

SITE = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
KERNELS = Path("/opt/llm-mini-infer/kernels")
NOTICES = Path("/usr/share/doc/llm-mini-infer/THIRD_PARTY_NOTICES")
# A licence file, by its name: LICENSE, LICENCE, COPYING, NOTICE or COPYRIGHT, with any suffix
LICENCE_FILE = re.compile(r"^(?:LICEN[CS]E|COPYING|NOTICE|COPYRIGHT)", re.I)
# What FlashInfer's cut removes (deploy/Dockerfile's deps stage): files the AOT package's RECORD still lists
CUT = ("flashinfer_jit_cache/providers/sm120f/jit_cache/",)
# The host check Job's settings beside the model and its assistant (deploy/helm/examples/hostcheck-job.yaml): none
JOB_SETS: list[str] = []


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def files(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*")) if p.is_file()}


def distributions() -> dict:
    """Every installed distribution's version, and the files its RECORD lists that are missing: those FlashInfer's cut
    removed counted by package, the rest listed. The RECORD is read as it is: `Distribution.files` leaves out the
    files that are missing (Python 3.12's importlib.metadata), so it can never show one."""
    dists, cut, missing, no_record = {}, {c: 0 for c in CUT}, [], []
    for d in md.distributions():
        dists[d.metadata["Name"]] = d.version
        record = d.read_text("RECORD")
        if record is None:
            no_record.append(d.metadata["Name"])
            continue
        for row in csv.reader(record.splitlines()):
            if not row:
                continue
            p = Path(d.locate_file(row[0]))
            if p.exists():
                continue
            rel = os.path.relpath(p, SITE)
            prefix = next((c for c in CUT if rel.startswith(c)), None)
            if prefix:
                cut[prefix] += 1
            else:
                missing.append(rel)
    return {"dists": dists, "cut": cut, "missing": missing, "no_record": no_record}


def package_metadata() -> dict:
    """llm-mini-infer's own metadata as installed: its licence expression, the sha256 of each licence file its dist-info
    carries (PEP 639's `licenses/`), and its readme's type and sha256 (the description, stripped)."""
    d = md.distribution("llm-mini-infer")
    m = d.metadata
    files = {}
    for f in m.get_all("License-File") or []:
        text = d.read_text(f"licenses/{f}")
        files[f] = None if text is None else hashlib.sha256(text.encode()).hexdigest()
    readme = (m.get_payload() or "").strip()
    return {"license_expression": m.get("License-Expression"), "license_files": files,
            "readme_type": m.get("Description-Content-Type"),
            "readme_sha256": hashlib.sha256(readme.encode()).hexdigest()}


def kernel_lookup() -> list[str]:
    """Where the runtime's own cache keys (nvcc's version as the build recorded it, the installed sources, FlashInfer's
    CUTLASS) look for the prebuilt libraries."""
    from llm_mini_infer.kernels import build

    return [str(build.prebuilt(p)) for p in (build.library_path(["120"]), build.cutlass_library_path(["120"]))]


def aot_providers() -> list[dict]:
    """FlashInfer's AOT providers, as its jit-cache shim finds them (their entry points)."""
    import flashinfer_jit_cache

    return [{"id": p.provider_id, "dist": p.distribution, "version": p.version, "dir": str(p.jit_cache_dir),
             "archs": sorted(p.cuda_architectures)} for p in flashinfer_jit_cache.get_jit_cache_providers()]


def toolchain() -> dict:
    """nvcc and ptxas: on PATH, under $CUDA_HOME/bin and /usr/local/cuda/bin, and what that bin/ holds."""
    home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    found = {t: [str(p) for p in (shutil.which(t), home / "bin" / t, Path("/usr/local/cuda/bin") / t)
                 if p and Path(p).exists()] for t in ("nvcc", "ptxas")}
    return {"found": found, "cuda_bin": sorted(p.name for p in (home / "bin").iterdir()) if (home / "bin").is_dir()
            else []}


def notices() -> dict:
    """The third-party notices (deploy/THIRD_PARTY_NOTICES) and what they say of the image: the paths they name that
    are missing, the distributions with no licence file among the files their RECORD lists, the libraries Pillow
    bundles (by name, without auditwheel's hash), the sections of Pillow's licence file, and the image's apt source."""
    text = NOTICES.read_text()
    paths = re.findall(r"(?<![\w.:/>])((?:/|site-packages/)[\w./*+-]*[\w/*])", text)
    missing = [p for p in paths if not glob.glob(str(SITE / p.removeprefix("site-packages/"))
                                                 if p.startswith("site-packages/") else p)]

    def licensed(f: md.PackagePath) -> bool:
        return bool(LICENCE_FILE.match(f.name)) or (f.parts[0].endswith(".dist-info") and "licenses" in f.parts[1:-1])

    pillow = next(SITE.glob("pillow-*.dist-info/licenses/LICENSE")).read_text()
    return {"sha256": sha256(NOTICES), "missing": missing,
            "unlicensed": sorted(d.metadata["Name"] for d in md.distributions()
                                 if not any(licensed(f) for f in d.files or [])),
            "pillow_libs": sorted({p.name.split(".")[0].rsplit("-", 1)[0] for p in (SITE / "pillow.libs").iterdir()}),
            "pillow_sections": [s.strip("\n").splitlines()[0] for s in pillow.split("\n----\n") if s.strip()],
            "apt_source": Path("/etc/apt/sources.list.d/ubuntu.sources").read_text()}


def imports() -> dict:
    import llm_mini_infer

    failed, names = [], []
    for mod in pkgutil.walk_packages(llm_mini_infer.__path__, "llm_mini_infer."):
        names.append(mod.name)
        try:
            __import__(mod.name)
        except Exception:
            failed.append(f"{mod.name}: {traceback.format_exc(limit=1).strip().splitlines()[-1]}")
    return {"modules": names, "failed": failed}


def preflight(model_data: Path) -> dict:
    """The entrypoint's preflight on the image's configuration with the host check Job's settings, over Gemma 4's own
    configuration files (as tests/deploy/test_deploy_contract.py): load, derive, /dev/shm, the caches, as the image's
    user."""
    from llm_mini_infer import entrypoint

    with tempfile.TemporaryDirectory() as tmp:
        target, assistant = Path(tmp) / "target", Path(tmp) / "assistant"
        target.mkdir()
        assistant.mkdir()
        for src, dst in (("gemma-4-31b-it-fp8-block.config.json", "config.json"),
                         ("gemma-4-31b-it.generation_config.json", "generation_config.json"),
                         ("gemma-4-31b-it.processor_config.json", "processor_config.json"),
                         ("gemma-4-31b-it.tokenizer_config.json", "tokenizer_config.json")):
            shutil.copy(model_data / src, target / dst)
        (target / "tokenizer.json").write_text("{}")
        (assistant / "config.json").write_text(json.dumps({"architectures": ["Gemma4AssistantForCausalLM"],
                                                           "model_type": "gemma4_assistant"}))
        sets = [f"model.path={target}", f"speculative.model={assistant}", *JOB_SETS]
        config = os.environ.get("LLM_MINI_INFER_CONFIG", entrypoint.CONFIG)
        return {"config": config, "problems": entrypoint.preflight(config, [x for s in sets for x in ("--set", s)])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-data", type=Path, required=True)
    args = ap.parse_args()
    facts: dict = {"errors": {}}
    steps = {
        "distributions": distributions,
        "package_metadata": package_metadata,
        "kernels": lambda: {p: d for p, d in files(KERNELS).items() if p.endswith(".so")},
        "kernel_record": lambda: (KERNELS / "nvcc-version.txt").read_text(),
        "kernel_lookup": kernel_lookup,
        "aot": lambda: files(SITE / "flashinfer_jit_cache" / "providers" / "sm120f" / "jit_cache"),
        "cubins": lambda: sorted(str(p.relative_to(SITE)) for p in SITE.rglob("*.cubin"))[:20],
        "cubin_package": lambda: importlib.util.find_spec("flashinfer_cubin") is not None,
        "aot_providers": aot_providers,
        "toolchain": toolchain,
        "compilers": lambda: {t: shutil.which(t) for t in ("gcc", "g++", "cc", "c++", "cicc", "nvlink", "fatbinary",
                                                           "cudafe++", "nvcc", "ptxas") if shutil.which(t)},
        "notices": notices,
        "imports": imports,
        "preflight": lambda: preflight(args.model_data),
        "version": lambda: __import__("llm_mini_infer.version", fromlist=["version"]).version(),
        "uid": os.getuid,
        "cache_writable": lambda: os.access("/cache", os.W_OK),
    }
    for name, fn in steps.items():
        try:
            facts[name] = fn()
        except Exception:
            facts["errors"][name] = traceback.format_exc()
    print(json.dumps(facts))


if __name__ == "__main__":
    main()
