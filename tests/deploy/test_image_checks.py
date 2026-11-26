"""The CPU checks of a built slim image (deploy/Dockerfile; docs/development.md, "Building the image"): what can be
checked without a GPU. The Actions workflow runs them on the runner against the image it built, before anything is
pushed (.github/workflows/image.yml); skipped without an image:

    LLM_MINI_INFER_IMAGE=llm-mini-infer:check LLM_MINI_INFER_IMAGE_REVISION=<full commit> \\
        uv run --locked --only-group image-checks pytest tests/deploy/test_image_checks.py

`docker image inspect` gives the labels, the user, the entrypoint and the port; tests/scripts/image_probe.py, run in the
image as its user (`--network none`, `--shm-size 2g`, tests/data mounted read-only), gives the rest:
- **versions:** every installed distribution at uv.lock's version for the image (no dev group, the extras cuda, images
  and image, less what the Dockerfile leaves out: flashinfer-cubin and the jit-cache providers of other GPUs), nothing
  else, llm-mini-infer at pyproject's; every file a RECORD lists present but FlashInfer's cut;
- **kernels:** both libraries at the Dockerfile's KERNELS_SHA256, with nvcc 13.0.88's version as their build recorded
  it, and the runtime's own lookup finds them;
- **FlashInfer:** the AOT modules the inventory recorded, by sha256, and no other; the shim finds SM120's provider
  alone; no cubin, no cubin package;
- **no nvcc, no ptxas, no compiler:** none on PATH or under the CUDA toolkit's bin/ (the kernels' keys read the
  build's record; FlashInfer's import takes torch's CUDA version);
- **notices:** /usr/share/doc/llm-mini-infer/THIRD_PARTY_NOTICES is deploy/THIRD_PARTY_NOTICES, and holds there: every
  path it names exists, the wheels without a licence file and Pillow's bundled libraries are the ones it names,
  Pillow's licence file has the sections it refers to, the apt source is the Dockerfile's Ubuntu snapshot;
- **imports:** every module of llm_mini_infer, Qwen's included;
- **configuration:** the entrypoint's preflight on the image's config.yaml with the host check Job's settings, over
  Gemma 4's own configuration files: load, derive, /dev/shm and the caches, as the image's user;
- **user:** not root, /cache writable; `llm_mini_infer.version.version()` is the build id `<version>+<commit>`;
- **licence:** the label, and llm-mini-infer's dist-info carrying the repository's README.md and LICENSE.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import pytest

from ..support.paths import DATA
from ..support.slim_image import ROOT, SOURCE, aot_modules, dockerfile_arg, installed, kernels, norm, version

IMAGE = os.environ.get("LLM_MINI_INFER_IMAGE")
REVISION = os.environ.get("LLM_MINI_INFER_IMAGE_REVISION")
pytestmark = pytest.mark.skipif(not IMAGE,
                                reason="no image to check: LLM_MINI_INFER_IMAGE=<ref> (the image workflow's)")
UID = 10001  # deploy/Dockerfile's llm-mini-infer user
NVCC_RELEASE = "V13.0.88"
NOTICES = ROOT / "deploy" / "THIRD_PARTY_NOTICES"


def docker(*args: str) -> str:
    return subprocess.run(["docker", *args], check=True, capture_output=True, text=True).stdout


@pytest.fixture(scope="module")
def config() -> dict:
    return json.loads(docker("image", "inspect", IMAGE))[0]["Config"]


@pytest.fixture(scope="module")
def facts() -> dict:
    out = docker("run", "--rm", "--network", "none", "--shm-size", "2g", "--entrypoint", "python3",
                 "-v", f"{ROOT / 'tests' / 'scripts' / 'image_probe.py'}:/checks/image_probe.py:ro",
                 "-v", f"{DATA}:/model-data:ro",
                 IMAGE, "-I", "-B", "/checks/image_probe.py", "--model-data", "/model-data")
    facts = json.loads(out.strip().splitlines()[-1])  # the last line: an import may print before it
    assert facts["errors"] == {}, "\n".join(facts["errors"].values())
    return facts


def test_the_labels_name_the_source_version_and_revision(config) -> None:
    labels = config.get("Labels") or {}
    assert labels.get("org.opencontainers.image.source") == SOURCE
    assert labels.get("org.opencontainers.image.version") == version()
    if REVISION:
        assert labels.get("org.opencontainers.image.revision") == REVISION


def test_the_licence_label(config) -> None:
    assert (config.get("Labels") or {}).get("org.opencontainers.image.licenses") == "Apache-2.0"


def test_the_package_carries_its_readme_and_licence_files(facts) -> None:
    meta = facts["package_metadata"]
    assert meta["license_expression"] == "Apache-2.0"
    assert meta["license_files"] == {"LICENSE": hashlib.sha256((ROOT / "LICENSE").read_bytes()).hexdigest()}
    readme = (ROOT / "README.md").read_text().strip()
    assert meta["readme_type"] == "text/markdown"
    assert meta["readme_sha256"] == hashlib.sha256(readme.encode()).hexdigest()


def test_the_user_entrypoint_and_port(config) -> None:
    assert (config.get("User") or "root").split(":")[0] == str(UID)
    assert config["Entrypoint"] == ["python3", "-m", "llm_mini_infer.entrypoint"] and config["Cmd"] == ["serve"]
    assert set(config.get("ExposedPorts") or {}) == {"8000/tcp"}


def test_every_distribution_is_the_locks_and_nothing_else(facts) -> None:
    have = {norm(n): v for n, v in facts["distributions"]["dists"].items()}
    want = installed()
    assert want["llm-mini-infer"] == version()
    assert {n: (have.get(n), v) for n, v in want.items() if have.get(n) != v} == {}
    assert {n: v for n, v in have.items() if n not in want} == {}


def test_every_record_holds_but_flashinfers_cut(facts) -> None:
    d = facts["distributions"]
    assert d["no_record"] == [] and d["missing"] == []
    # the cut removed what the sm120f provider's RECORD lists under jit_cache/ but the three modules (560-odd files)
    cut = "flashinfer_jit_cache/providers/sm120f/jit_cache/"
    assert list(d["cut"]) == [cut] and d["cut"][cut] > 0


def test_the_kernels_are_the_recorded_bytes_and_the_runtime_finds_them(facts) -> None:
    assert facts["kernels"] == kernels()
    assert NVCC_RELEASE in facts["kernel_record"]  # nvcc's --version as the build saw it: the keys' compiler
    prebuilt = Path("/opt/llm-mini-infer/kernels")
    assert [str(Path(p).relative_to(prebuilt)) for p in facts["kernel_lookup"]] == list(kernels())


def test_flashinfer_keeps_the_aot_modules_llm_mini_infer_loads_and_no_cubin(facts) -> None:
    prefix = "flashinfer_jit_cache/providers/sm120f/jit_cache/"
    assert {prefix + p: d for p, d in facts["aot"].items()} == aot_modules()
    assert facts["cubins"] == [] and facts["cubin_package"] is False


def test_flashinfer_finds_sm120s_provider_alone(facts) -> None:
    """The jit-cache shim finds one provider, SM120's, by its entry point: a shim without one imports all the same, and
    FlashInfer would then compile its modules at their first use, which the image cannot."""
    (p,) = facts["aot_providers"]
    assert p["id"] == "sm120f" and p["archs"] == ["sm120f"] and p["dist"] == "flashinfer-jit-cache-sm120f"
    assert p["version"] == installed()["flashinfer-jit-cache-sm120f"]
    assert p["dir"].endswith("site-packages/flashinfer_jit_cache/providers/sm120f/jit_cache")


def test_no_nvcc_ptxas_or_compiler(facts) -> None:
    assert facts["toolchain"]["found"] == {"nvcc": [], "ptxas": []}
    assert not {"nvcc", "nvcc.profile", "ptxas"} & set(facts["toolchain"]["cuda_bin"])
    assert facts["compilers"] == {}  # a GPU without prebuilt kernels is refused at startup


def listed(text: str, after: str) -> dict[str, str]:
    """The notices' `- name, name: what` list in the paragraph that starts with `after`: each name with its what."""
    paragraph = text[text.index(after):].split("\n\n")[0]
    return {n: what for names, what in re.findall(r"^- ([^:\n]+): (.*)$", paragraph, re.M) for n in names.split(", ")}


def test_the_notices_hold_for_the_image(facts) -> None:
    n, text = facts["notices"], NOTICES.read_text()
    assert n["sha256"] == hashlib.sha256(NOTICES.read_bytes()).hexdigest()
    assert n["missing"] == []
    # a new wheel without a licence file, or a library Pillow newly bundles, needs its licence in the notices
    assert sorted(map(norm, n["unlicensed"])) == sorted(map(norm, listed(text, "site-packages/*.dist-info/ names")))
    libs = listed(text, "Pillow's wheel")
    assert n["pillow_libs"] == sorted(libs)
    sections = [w.removeprefix("Pillow's ") for w in libs.values() if w.startswith("Pillow's ")]
    assert sections and set(sections) <= set(n["pillow_sections"])
    assert f"https://snapshot.ubuntu.com/ubuntu/{dockerfile_arg('UBUNTU_SNAPSHOT')}/" in n["apt_source"]


def test_every_module_imports(facts) -> None:
    imp = facts["imports"]
    assert imp["failed"] == []
    assert len(imp["modules"]) > 50 and any("qwen" in m for m in imp["modules"])


def test_the_configuration_passes_the_preflight(facts) -> None:
    assert facts["preflight"]["config"] == "/etc/llm-mini-infer/config.yaml"
    assert facts["preflight"]["problems"] == []


def test_it_runs_as_its_user_and_names_its_build(facts) -> None:
    assert facts["uid"] == UID and facts["cache_writable"] is True
    if REVISION:  # the workflow's GIT_COMMIT: the commit's first 7 characters
        assert facts["version"] == f"{version()}+{REVISION[:7]}"
    else:
        assert facts["version"].startswith(f"{version()}+")
