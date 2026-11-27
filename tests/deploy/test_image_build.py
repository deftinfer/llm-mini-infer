"""The slim image's build (docs/development.md, "Building the image"), on the CPU and offline:
- **the lock:** uv.lock is the only lock, for macOS and linux x86_64; the five packages of the GPU-validated stack,
  and what they constrain, are the versions the image recorded (tests/data/image/freeze.txt); every distribution an
  llm-mini-infer process imported there is locked; pyproject's lower bounds are the locked versions;
- **deploy/Dockerfile:** pinned bases, uv's Docker guide (uv mounted, the dependencies' layer, then the project),
  FlashInfer's cubin package and other GPUs' AOT providers left out and its AOT modules cut to the inventory's, the apt
  packages at the recorded versions (the image's apt list, the kernel build's toolchain), the kernels' build and check,
  the third-party notices, the runtime on the dependencies' stage and its layer order without nvcc or ptxas, the
  entrypoint, config path, port and a non-root user, the labels;
- **tests/gpu-image.Dockerfile:** the GPU tests' image on a published one, its packages unchanged, its apt packages
  from deploy/Dockerfile's debs stage, installed offline;
- **one version:** pyproject's, the chart's appVersion and default tag, the host check Job's image, and the image
  names in README.md, docs/operations.md and deploy/docker-compose.yaml;
- **.github/workflows/image.yml:** triggers, permissions, pins, tags, the checks before the push, the test image.
The build itself runs in Actions, whose checks of the built image are tests/deploy/test_image_checks.py."""

from __future__ import annotations

import json
import re
import tomllib

import yaml
from packaging.requirements import Requirement

from ..support.slim_image import (
    DATA,
    DOCKERFILE,
    EXTRAS,
    IMAGE_REPOSITORY,
    INVENTORY,
    LOCK,
    PYPROJECT,
    ROOT,
    SOURCE,
    STACK_LABEL,
    aot_modules,
    closure,
    dockerfile_arg,
    freeze,
    kernels,
    norm,
    not_installed,
    version,
)

WORKFLOW = ROOT / ".github" / "workflows" / "image.yml"
TEST_IMAGE = ROOT / "tests" / "gpu-image.Dockerfile"
APT = DATA / "apt.txt"
CHART = ROOT / "deploy" / "helm" / "llm-mini-infer"
# The five packages of the GPU-validated stack, and what they constrain on linux: FlashInfer's AOT providers by GPU,
# torch's and FlashInfer's CUDA stack, FlashInfer's FFI and NCCL bindings, and triton (torch's pin)
FIVE = {"torch", "flashinfer-python", "flashinfer-jit-cache", "flashinfer-cubin", "llguidance", "tokenizers", "pillow"}
# Pinned for the dev group alone, not in the image: torch's own release of torchvision, the backend of the reference
# image processors the tests compare with
DEV_PINNED = {"torchvision"}
# What the image leaves out of the lock's closure: FlashInfer's cubins (none is opened on SM120) and the AOT modules
# of every GPU but SM120's
LEFT_OUT = ("flashinfer-cubin", "flashinfer-jit-cache-sm80", "flashinfer-jit-cache-sm89", "flashinfer-jit-cache-sm90a",
            "flashinfer-jit-cache-sm100a", "flashinfer-jit-cache-sm103a")


def constrained(name: str) -> bool:
    return name.startswith(("nvidia-", "cuda-", "flashinfer-jit-cache-")) or name in {
        "triton", "apache-tvm-ffi", "nccl4py", "nccl-extensions"}


def pyproject() -> dict:
    return tomllib.loads(PYPROJECT.read_text())


def lock() -> dict:
    return tomllib.loads(LOCK.read_text())


# -- the lock ------------------------------------------------------------------------------------------------------

def test_the_five_and_what_they_constrain_are_the_recorded_stacks() -> None:
    """Each one the image installs is at the version of its recorded freeze (tests/data/image/freeze.txt); the ones it
    leaves out (flashinfer-cubin, the other GPUs' AOT providers) are at FlashInfer's release."""
    locked, frozen, left_out = closure(), freeze(), not_installed()
    parity = {n for n in locked if n in FIVE or constrained(n)}
    assert parity >= FIVE and len(parity) > 30 and left_out <= parity
    assert {n: (locked[n], frozen.get(n)) for n in parity - left_out if locked[n] != frozen.get(n)} == {}
    assert not left_out & set(frozen) and {locked[n].split("+")[0] for n in left_out} == {locked["flashinfer-python"]}
    for name, ver in (("torch", "2.14.1+cu130"), ("flashinfer-python", "0.7.2"),
                      ("flashinfer-jit-cache", "0.7.2+cu130"), ("flashinfer-jit-cache-sm120f", "0.7.2+cu130"),
                      ("llguidance", "1.9.1"), ("tokenizers", "0.23.3"), ("pillow", "13.0.0"), ("triton", "3.8.0"),
                      ("nvidia-nccl-cu13", "2.30.7")):
        assert locked[name] == ver == frozen[name]
    assert "vllm" not in locked and not DEV_PINNED & (set(locked) | set(frozen))  # never imported by llm_mini_infer


def test_the_cu12_flavour_of_cute_dsl_and_the_aarch64_providers_are_left_out() -> None:
    """nvidia-cutlass-dsl requires its cu12 libraries, but CuTe DSL loads the cu13 flavour under any CUDA 13 driver;
    and uv takes the jit-cache shim's requirements from its aarch64 wheel, which adds two providers with aarch64 wheels
    only: overrides drop the requirements, so neither the lock nor the image has them."""
    locked, names = closure(), {p["name"] for p in lock()["package"]}
    assert "nvidia-cutlass-dsl-libs-cu13" in locked and "nvidia-cutlass-dsl-libs-cu12" not in names
    never = {"nvidia-cutlass-dsl-libs-cu12", "flashinfer-jit-cache-sm110a", "flashinfer-jit-cache-sm121a"}
    overrides = [Requirement(r) for r in pyproject()["tool"]["uv"]["override-dependencies"]]
    assert {r.name: str(r.marker) for r in overrides} == dict.fromkeys(never, 'sys_platform == "never"')
    assert not never & names


def test_pyproject_pins_them_where_uv_resolves() -> None:
    """The pins are pyproject's [tool.uv] constraints, each at the locked version and the installed ones at the recorded
    freeze's (without the local `+cu130`, so macOS's CPU wheels match too); NCCL among them, at torch's own pin, and
    the dev group's torchvision."""
    uv = pyproject()["tool"]["uv"]
    pins = {norm(r.name): str(r.specifier) for r in map(Requirement, uv["constraint-dependencies"])}
    frozen, locked = freeze(), closure()
    dev = {p["name"]: p["version"] for p in lock()["package"] if p["name"] in DEV_PINNED}
    assert set(pins) == {n for n in locked if n in FIVE or constrained(n)} | DEV_PINNED and "nvidia-nccl-cu13" in pins
    assert all(spec == f"=={(locked.get(n) or dev[n]).split('+')[0]}" for n, spec in pins.items()), pins
    assert all(spec == f"=={frozen[n].split('+')[0]}" for n, spec in pins.items() if n in frozen), pins


def test_one_lock_for_the_mac_and_linux() -> None:
    lk = lock()
    assert set(lk["manifest"]) >= {"constraints", "overrides"}
    sources = {(p["name"], p["version"]): next(iter(p["source"].values())) for p in lk["package"]}
    assert sources[("torch", "2.14.1+cu130")] == sources[("torchvision", "0.29.1+cu130")] == (
        "https://download.pytorch.org/whl/cu130")
    assert sources[("torch", "2.14.1")] == "https://pypi.org/simple"  # the macOS CPU wheel
    # the jit-cache shim and its providers (the image installs SM120's alone) from FlashInfer's CUDA 13.0 index
    jit = {(n, v): s for (n, v), s in sources.items() if n.startswith("flashinfer-jit-cache")}
    assert {n for n, _ in jit} == {"flashinfer-jit-cache", "flashinfer-jit-cache-sm120f", *LEFT_OUT[1:]}
    assert set(jit.values()) == {"https://flashinfer.ai/whl/cu130"} and {v for _, v in jit} == {"0.7.2+cu130"}
    assert sources[("flashinfer-cubin", "0.7.2")] == "https://flashinfer.ai/whl"
    mac = closure(env={"sys_platform": "darwin", "platform_machine": "arm64", "platform_system": "Darwin",
                       "os_name": "posix", "python_version": "3.12", "python_full_version": "3.12.12",
                       "implementation_name": "cpython", "implementation_version": "3.12.12",
                       "platform_python_implementation": "CPython", "platform_release": "", "platform_version": "",
                       "extra": ""})
    assert mac["torch"] == "2.14.1" and not {n for n in mac if n.startswith(("nvidia-", "flashinfer"))}
    for name in ("requirements.txt", "lock.py", "install.py", "build-constraints.txt"):  # no second lock
        assert not (ROOT / "deploy" / "image" / name).exists()


def test_pyproject_declares_what_the_inventory_imported() -> None:
    """Every distribution whose modules a server process imported in the image (Gemma, Qwen, the host check) is in the
    image's closure of uv.lock."""
    inv = json.loads(INVENTORY.read_text())
    imported = {norm(d) for run in inv["runs"].values() for role in run["roles"].values() for d in role["dists"]}
    assert imported and imported - set(closure()) == set()


def test_the_lower_bounds_are_the_locked_versions() -> None:
    """Every declared requirement's minimum is what uv.lock holds, so the declared minimums are what the tests ran."""
    pp = pyproject()
    reqs = [*pp["project"]["dependencies"], *[r for rs in pp["project"]["optional-dependencies"].values() for r in rs],
            *[r for rs in pp["dependency-groups"].values() for r in rs]]
    locked: dict[str, set[str]] = {}
    for p in lock()["package"]:
        locked.setdefault(p["name"], set()).add(p["version"].split("+")[0])
    for r in map(Requirement, reqs):
        (spec,) = r.specifier
        assert spec.operator == ">=" and locked[norm(r.name)] == {spec.version}, (str(r), locked[norm(r.name)])
    (backend,) = map(Requirement, pp["build-system"]["requires"])
    assert backend.name == "hatchling" and str(backend.specifier).startswith("==")


# -- deploy/Dockerfile -----------------------------------------------------------------------------------------------

def stages() -> dict[str, list[str]]:
    """Each stage's instructions (continuation lines joined, comments dropped), by stage name."""
    out: dict[str, list[str]] = {"": []}
    cur = ""
    for ins in re.sub(r"\\\n\s*", " ", DOCKERFILE.read_text()).splitlines():
        ins = ins.strip()
        if not ins or ins.startswith("#"):
            continue
        if m := re.match(r"FROM \S+ AS (\S+)", ins):
            cur = m.group(1)
            out[cur] = []
        out[cur].append(ins)
    return out


def runs(stage: str) -> list[str]:
    return [i for i in stages()[stage] if i.startswith("RUN")]


def apt_pins(stage: str) -> dict[str, str]:
    """The `name=version` packages a stage's apt-get install names (its `-o` options aside)."""
    install = r"apt-get(?: -o \S+)* install"
    (run,) = [r for r in runs(stage) if re.search(install, r)]
    return dict(re.findall(r"\s([a-z0-9][a-z0-9.+-]*)=(\S+)", re.split(install, run, maxsplit=1)[1]))


def test_the_bases_are_pinned() -> None:
    text = DOCKERFILE.read_text()
    assert re.search(r"^ARG CUDA_BASE=nvidia/cuda:13\.0\.2-base-ubuntu24\.04@sha256:[0-9a-f]{64}$", text, re.M)
    assert re.search(r"^ARG UV_IMAGE=ghcr\.io/astral-sh/uv:[0-9.]+@sha256:[0-9a-f]{64}$", text, re.M)
    # the label's stack: the five's versions in the image, as its freeze records them
    stack, frozen = dict(p.split("==") for p in dockerfile_arg("STACK").split()), freeze()
    assert stack == {n: frozen[n] for n in ("torch", "flashinfer-python", "llguidance", "tokenizers", "pillow")}
    assert re.fullmatch(r"\d{8}T\d{6}Z", dockerfile_arg("UBUNTU_SNAPSHOT"))
    # runtime last: the default target, so a build without one is the release
    assert list(stages()) == ["", "uv", "os", "deps", "kernels", "debs", "test-debs", "project", "runtime"]
    # the uv the image builds with is the one that wrote the lock's format: mounted into each RUN that calls it, so no
    # layer holds it
    uses = [r for s in stages().values() for r in s if r.startswith("RUN") and re.search(r"\buv (sync|cache)", r)]
    assert len(uses) == 2 and all("--mount=from=uv,source=/uv,target=/bin/uv" in r for r in uses)
    assert not [i for s in stages().values() for i in s if i.startswith("COPY") and "--from=uv" in i]


def test_the_dependencies_layer_follows_uvs_guide_and_reads_only_the_lock() -> None:
    """Editing llm_mini_infer/ must not rebuild the 6.9 GB venv: its RUN binds pyproject.toml and uv.lock alone."""
    deps = stages()["deps"]
    env = " ".join(i for i in deps if i.startswith("ENV"))
    for kv in ("UV_COMPILE_BYTECODE=1", "UV_LINK_MODE=copy", "UV_PYTHON_DOWNLOADS=never",
               "UV_PROJECT_ENVIRONMENT=/opt/venv"):
        assert kv in env, kv
    (run,) = runs("deps")
    assert set(re.findall(r"type=bind,source=(\S+?),", run)) == {"pyproject.toml", "uv.lock"}
    assert "--mount=type=cache,target=/root/.cache/uv" in run
    extras = " ".join(f"--extra {e}" for e in EXTRAS)
    leave_out = " ".join(f"--no-install-package {n}" for n in LEFT_OUT)
    assert f"uv sync --locked --no-dev --no-install-project {extras} {leave_out}" in " ".join(run.split())
    (project,) = runs("project")
    assert f"uv sync --locked --no-dev --no-editable {extras} {leave_out}" in " ".join(project.split())
    # the shim's providers but SM120's
    (shim,) = [p for p in lock()["package"] if p["name"] == "flashinfer-jit-cache"]
    providers = {d["name"] for d in shim["dependencies"]}
    assert not_installed() == set(LEFT_OUT) == {"flashinfer-cubin", *providers - {"flashinfer-jit-cache-sm120f"}}
    assert not [i for i in stages()["project"] if i.startswith("COPY") and ("deploy/" in i or "tests" in i)]


def test_flashinfer_is_cut_to_the_inventorys_aot_modules_in_the_same_layer() -> None:
    """No cubin package (not installed: no process opened a cubin on SM120, tests/data/image/README.md), and SM120's
    AOT provider cut to the three modules llm-mini-infer loads in the RUN that installs it, its wheel cleaned out of the
    cache."""
    (run,) = runs("deps")
    provider = "flashinfer_jit_cache/providers/sm120f/jit_cache/"
    assert run.index("uv sync") < run.index(f"cd /opt/venv/lib/python3.12/site-packages/{provider[:-1]}")
    assert "! -name %s" in run and 'test "$(ls | wc -l)" = 3' in run
    assert run.endswith("uv cache clean flashinfer-jit-cache-sm120f") and "flashinfer_cubin" not in run
    names = dockerfile_arg("FLASHINFER_AOT").split()
    assert all(p.startswith(provider) for p in aot_modules())
    assert sorted(names) == sorted(p.removeprefix(provider).split("/")[0] for p in aot_modules())


def test_the_apt_packages_are_the_recorded_ones() -> None:
    """The runtime's apt pins are the image's recorded list (tests/data/image/apt.txt), the kernels stage's the
    toolchain's record (tests/data/image/toolchain.json: what the kernel build reads, by package), and none of the
    toolchain reaches the runtime; the base's own packages are upgraded from the same snapshot first."""
    record = dict(line.rsplit(" ", 1)[0].split("=", 1) for line in APT.read_text().splitlines()
                  if line and not line.startswith("#"))
    runtime, build = apt_pins("os"), apt_pins("kernels")
    assert {n: (v, record.get(n)) for n, v in runtime.items() if record.get(n) != v} == {}
    # build: what the kernel build reads; runtime: what a server process mapped (the inventory) and the interpreter.
    # nvcc 13.0.88 (the kernels' cache key) only in the build, which records its version beside the kernels for the
    # runtime
    toolchain = json.loads((DATA / "toolchain.json").read_text())
    assert build == toolchain["packages"] and build["cuda-nvcc-13-0"] == "13.0.88-1"
    assert not set(build) & set(record)
    # the base's Ubuntu packages upgraded from the snapshot alone (NVIDIA's moving repository set aside meanwhile, and
    # back for the kernels stage), then the runtime's installed
    (os_run,) = runs("os")
    steps = [s.strip() for s in os_run.split("&&")]
    i = steps.index("mv /etc/apt/sources.list.d/cuda.list /tmp/cuda.list")
    assert steps[i + 1:i + 3] == ["apt-get update", "apt-get upgrade -y"]
    assert steps[i + 3].startswith("apt-get install -y --no-install-recommends ")
    assert steps[i + 4] == "mv /tmp/cuda.list /etc/apt/sources.list.d/cuda.list"
    inv = json.loads(INVENTORY.read_text())
    mapped = {o.removeprefix("apt:") for run in inv["runs"].values() for r in run["roles"].values()
              for o in r["libs"] if o.startswith("apt:")}
    assert set(runtime) == mapped | {"python3.12", "python3.12-minimal"}


def test_the_kernels_build_reproducibly_and_match_their_recorded_bytes() -> None:
    ins = stages()["kernels"]
    assert "WORKDIR /opt/llm-mini-infer/src" in ins
    (build,) = [i for i in ins if "llm_mini_infer.kernels.build" in i]
    assert "LLM_MINI_INFER_KERNEL_CACHE=/opt/llm-mini-infer/kernels" in build
    # ptxas records each file's mtime in the line tables: ours and CUTLASS's are set to 0 before the build
    touch = build.index('find /opt/llm-mini-infer/src "$CUTLASS_HOME" -type f -exec touch -d @0')
    assert touch < build.index("kernels.build")
    # FlashInfer's CUTLASS at the include path the recorded bytes were built with: the line tables hold it
    assert any(i.startswith("ENV") and "CUTLASS_HOME=/usr/local/lib/python3.12/dist-packages/flashinfer/data/cutlass"
               in i for i in ins)
    assert any("$KERNELS_SHA256" in i and "sha256sum --check --strict" in i for i in ins)
    # KERNELS_SHA256 is the record the build checks; tests/data/image/README.md's table holds the same keys and bytes
    readme = (DATA / "README.md").read_text()
    assert len(kernels()) == 2
    for rel, digest in kernels().items():
        key, name = rel.split("/")
        assert f"`{name}` | `{key}`" in readme and re.fullmatch(r"[0-9a-f]{64}", digest)
        assert digest in readme
    assert build.endswith("&& cat nvcc-version.txt")  # nvcc's version recorded beside the libraries


def test_the_runtime_layers_and_behaviour() -> None:
    ins = stages()["runtime"]
    # on the dependencies' stage: its venv layer is the image's own, not a copy (a second 3 GB layer and cache blob)
    assert DOCKERFILE.read_text().count("FROM deps AS runtime") == 1
    layers = [i for i in ins if i.startswith(("COPY", "RUN", "ADD"))]
    assert not [i for i in layers if i.startswith(("RUN", "ADD"))]  # nothing unpacks the venv to add a layer
    assert all("--link" in i for i in layers) and not [i for i in layers if "/opt/venv /opt/venv" in i]
    order = [next(k for k in ("/opt/llm-mini-infer/kernels", "config.yaml", "THIRD_PARTY_NOTICES",
                              "--from=project /stage/")
                  if k in i) for i in layers]
    # the kernels, the configuration and the notices before Python, then llm-mini-infer; no nvcc or ptxas, nor anything
    # else of the CUDA toolkit's bin/ (the kernels' keys read nvcc's version from the build's record)
    assert order == ["/opt/llm-mini-infer/kernels", "config.yaml", "THIRD_PARTY_NOTICES", "--from=project /stage/"]
    assert not [i for i in layers if "cuda-13.0" in i or "nvcc" in i or "ptxas" in i]
    # the version and commit last: nothing after their ARGs but the ENV and LABEL that use them
    last = max(i for i, x in enumerate(ins) if x.startswith(("COPY", "RUN")))
    first = ins.index("ARG VERSION=unknown")
    assert first > last and ins[first + 1] == "ARG GIT_COMMIT=unknown"
    assert all(x.startswith(("ARG", "ENV", "LABEL")) for x in ins[first:])
    # the entrypoint, config path and port that compose, the chart and the host check Job use
    assert 'ENTRYPOINT ["python3", "-m", "llm_mini_infer.entrypoint"]' in ins and 'CMD ["serve"]' in ins
    assert "EXPOSE 8000" in ins
    env = " ".join(i for i in ins if i.startswith("ENV"))
    for kv in ("LLM_MINI_INFER_CONFIG=/etc/llm-mini-infer/config.yaml",
               "LLM_MINI_INFER_KERNEL_PREBUILT=/opt/llm-mini-infer/kernels", "HF_HOME=/cache/huggingface",
               "PATH=/opt/venv/bin:", "CUDA_HOME=/usr/local/cuda", "LLM_MINI_INFER_BUILD=${GIT_COMMIT}"):
        assert kv in env, kv
    assert "ENV FLASHINFER_DISABLE_JIT=1" in ins  # a missing FlashInfer module fails as missing, not in a build
    assert any("deploy/config.yaml /etc/llm-mini-infer/config.yaml" in i for i in layers)
    # the notices at their fixed path (tests/deploy/test_image_checks.py checks what they say against the image)
    assert "COPY --link deploy/THIRD_PARTY_NOTICES /usr/share/doc/llm-mini-infer/THIRD_PARTY_NOTICES" in layers
    # non-root, numeric (Kubernetes' runAsNonRoot can check it); the os stage creates it and its /cache
    assert "USER 10001:10001" in ins
    (os_run,) = runs("os")
    assert "useradd --uid 10001" in os_run and "install -d -o llm-mini-infer -g llm-mini-infer /cache" in os_run
    labels = " ".join(i for i in ins if i.startswith("LABEL"))
    for label in (f'org.opencontainers.image.source="{SOURCE}"', f'{STACK_LABEL}="${{STACK}}"',
                  'org.opencontainers.image.licenses="Apache-2.0"',
                  'org.opencontainers.image.version="${VERSION}"',
                  'org.opencontainers.image.revision="${GIT_REVISION}"'):
        assert label in labels, label


def test_the_build_context_has_what_the_dockerfile_reads() -> None:
    allowed = [line[1:].rstrip("/") for line in (ROOT / ".dockerignore").read_text().splitlines()
               if line.startswith("!")]
    text = re.sub(r"\\\n\s*", " ", DOCKERFILE.read_text())
    sources = re.findall(r"type=bind,source=(\S+?),", text)
    sources += [s for line in text.splitlines() if line.startswith("COPY") and "--from" not in line
                for s in line.split()[1:-1] if not s.startswith("--")]
    assert {"pyproject.toml", "uv.lock", "deploy/config.yaml", "llm_mini_infer", "README.md",
            "LICENSE"} <= set(sources)
    for src in sources:
        assert any(src == a or src.startswith(a + "/") for a in allowed), src
        assert (ROOT / src).exists(), src


def test_deploy_holds_only_what_deploys() -> None:
    assert sorted(p.name for p in (ROOT / "deploy").iterdir() if p.name != "__pycache__") == [
        "Dockerfile", "THIRD_PARTY_NOTICES", "config.yaml", "docker-compose.yaml", "helm"]


def test_pyproject_the_chart_and_its_default_tag_agree() -> None:
    """pyproject's version is the release (tag v<version>, image :<version>), the chart's appVersion and its default
    image tag; the host check Job (a plain manifest), the README's quickstart, operations.md and the compose file's
    comment name it too."""
    chart = yaml.safe_load((CHART / "Chart.yaml").read_text())
    values = yaml.safe_load((CHART / "values.yaml").read_text())
    assert chart["appVersion"] == version()
    assert values["image"] == {**values["image"], "repository": IMAGE_REPOSITORY, "tag": ""}
    deployment = (CHART / "templates" / "deployment.yaml").read_text()
    assert 'image: "{{ .Values.image.repository }}:{{ .Values.image.tag | default .Chart.AppVersion }}"' in deployment
    job = (ROOT / "deploy" / "helm" / "examples" / "hostcheck-job.yaml").read_text()
    assert re.findall(rf"image: ({re.escape(IMAGE_REPOSITORY)}:\S+)", job) == [f"{IMAGE_REPOSITORY}:{version()}"]
    for doc in ("README.md", "docs/operations.md", "deploy/docker-compose.yaml"):  # the quickstart's image
        text = (ROOT / doc).read_text()
        tags = [tag.rstrip('`)".,') for tag in re.findall(rf"{re.escape(IMAGE_REPOSITORY)}:(\S+)", text)]
        assert tags and set(tags) == {version()}, (doc, tags)


# -- .github/workflows/image.yml ------------------------------------------------------------------------------------

def workflow() -> dict:
    wf = yaml.safe_load(WORKFLOW.read_text())
    wf["on"] = wf.pop(True, wf.get("on"))  # YAML 1.1 reads `on` as true
    return wf


def steps() -> dict[str, dict]:
    return {s.get("name") or s["uses"].split("@")[0]: s for s in next(iter(workflow()["jobs"].values()))["steps"]}


def test_the_workflow_triggers_and_never_runs_on_a_fork() -> None:
    wf = workflow()
    assert set(wf["on"]) == {"push", "workflow_dispatch"}  # no pull_request: nothing a fork sends builds here
    # a branch builds by dispatch; a push that changes only docs builds nothing (a tag's push always runs)
    push = wf["on"]["push"]
    assert push["branches"] == ["main"] and push["tags"] == ["v*"] and set(push) == {"branches", "tags", "paths"}
    # a push to main builds when the build context or the checks' inputs change: every file .dockerignore lets in
    ignore = (ROOT / ".dockerignore").read_text().splitlines()
    context = {line[1:].rstrip("/") for line in ignore if line.startswith("!")}
    paths = push["paths"]
    for entry in context - {"README.md", "deploy"}:
        assert entry in paths or f"{entry}/**" in paths, entry
    assert {"deploy/Dockerfile", "deploy/config.yaml", "deploy/THIRD_PARTY_NOTICES", ".dockerignore"} <= set(paths)
    # everything the image checks import or mount
    assert {"tests/__init__.py", "tests/conftest.py", "tests/support/__init__.py", "tests/support/paths.py",
            "tests/support/slim_image.py", "tests/deploy/__init__.py", "tests/deploy/test_image_checks.py",
            "tests/scripts/image_probe.py", "tests/data/image/**",
            "tests/data/gemma-4-31b-it-fp8-block.config.json", "tests/data/gemma-4-31b-it.generation_config.json",
            "tests/data/gemma-4-31b-it.processor_config.json", "tests/data/gemma-4-31b-it.tokenizer_config.json",
            } <= set(paths)
    assert ".github/workflows/image.yml" in paths and paths[-1] == "!**/*.md"
    assert set(wf["on"]["workflow_dispatch"]["inputs"]) == {"publish", "cache", "test_image"}
    # read-only but for the two jobs, which push; a job's block replaces the whole set, so it restates contents
    assert wf["permissions"] == {"contents": "read"}
    assert list(wf["jobs"]) == ["image", "test-image"]
    for job in wf["jobs"].values():
        assert job["if"].startswith("vars.PUBLISH_IMAGE == 'true'")  # a repository's variable: a fork has none
        assert job["permissions"] == {"contents": "read", "packages": "write"}
        assert job["runs-on"] == "ubuntu-24.04"
    # the image is the repository's own: any repository that runs the workflow builds its own package
    assert wf["env"]["IMAGE"] == "ghcr.io/${{ github.repository }}"


def test_the_workflow_pins_its_actions_and_uses_only_its_own_token() -> None:
    text = WORKFLOW.read_text()
    uses = re.findall(r"uses: (\S+)", text)
    assert uses and all(re.fullmatch(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}", u) for u in uses), uses
    assert set(re.findall(r"secrets\.(\w+)", text)) == {"GITHUB_TOKEN"}
    assert re.search(r"moby/buildkit:v[0-9.]+@sha256:[0-9a-f]{64}", text)
    s = steps()
    build, publish = s["Build"]["with"], s["Publish"]["with"]
    assert build["load"] is True and "push" not in build  # checked before anything is pushed
    assert publish["push"] is True and publish["build-args"] == build["build-args"]
    assert "VERSION=${{ steps.tags.outputs.version }}" in build["build-args"]
    assert "mode=max" in build["cache-to"] and "buildcache-main" in build["cache-from"]
    assert build["no-cache"] == "${{ github.event_name == 'workflow_dispatch' && !inputs.cache }}"  # a cold build
    assert list(s).index("Check") < list(s).index("Publish")


def test_a_release_is_pyprojects_version_and_the_checks_are_the_tests() -> None:
    tags = steps()["Tags"]["run"]
    assert 'tomllib.load(open("pyproject.toml", "rb"))["project"]["version"]' in tags
    assert '[ "$GITHUB_REF_NAME" != "v$version" ]' in tags and "exit 1" in tags
    assert 'tags="$IMAGE:$version,$IMAGE:$commit"' in tags
    # a tag on a commit main published gets that commit's manifest itself: the same digest, not an index around it
    retag = steps()["Retag the commit's image"]["run"]
    assert 'imagetools create --prefer-index=false --tag "$t" "$IMAGE:$COMMIT"' in retag
    check = steps()["Check"]
    assert check["env"]["LLM_MINI_INFER_IMAGE"] == "llm-mini-infer:check"
    assert check["env"]["LLM_MINI_INFER_IMAGE_REVISION"] == "${{ github.sha }}"
    run = "run --locked --only-group image-checks pytest -p no:cacheprovider tests/deploy/test_image_checks.py"
    assert run in check["run"]
    assert "ARG UV_IMAGE=" in check["run"]  # the build's own uv, by digest


# -- tests/gpu-image.Dockerfile and the test-image job ----------------------------------------------------------------

def test_the_test_image_carries_the_build_toolchain() -> None:
    """The suite's compiler tests (FlashInfer's JIT, Triton's launcher, our kernels' build) need a compiler the release
    leaves out: deploy/Dockerfile's debs stage downloads, on os (the runtime's own userland), the kernels stage's
    packages at the kernels stage's versions, with nvcc's libnvptxcompiler at nvcc's and Python 3.12's headers at the
    interpreter's, and an SSH server and rsync, into /debs, after one apt-get update; test-debs holds /debs alone. The
    test image installs them from there, offline, and checks the compiler; its own ignore file keeps the root's
    .dockerignore in its tree."""
    (run,) = runs("debs")
    assert stages()["debs"][0] == "FROM os AS debs" and run.count("apt-get") == 3 and run.count(" update") == 1
    assert "-o Dir::Cache::archives=/debs install -y --no-install-recommends --download-only" in run
    # the set is closed: the same install, simulated with no lists left, needs nothing else
    assert " ".join(run.split()).endswith("rm -rf /debs/partial /debs/lock /var/lib/apt/lists/* "
                        "&& apt-get -s install --no-install-recommends /debs/*.deb >/dev/null")
    ours, kernels, os_stage = apt_pins("debs"), apt_pins("kernels"), apt_pins("os")
    assert {k: v for k, v in kernels.items() if k in ours} == kernels  # every kernels-stage package, at its version
    assert ours["libnvptxcompiler-13-0"] == ours["cuda-nvcc-13-0"] == "13.0.88-1"
    assert ours["python3.12-dev"] == ours["libpython3.12-dev"] == os_stage["python3.12"]
    assert set(ours) == {*kernels, "libnvptxcompiler-13-0", "python3.12-dev", "libpython3.12-dev"}
    assert re.search(r"\sopenssh-server\s+rsync\s", run)
    assert stages()["test-debs"] == ["FROM scratch AS test-debs", "COPY --from=debs /debs /debs"]
    text = re.sub(r"\\\n\s*", " ", TEST_IMAGE.read_text())
    (toolchain,) = [" ".join(line.split()) for line in text.splitlines() if line.startswith("RUN") and "/debs" in line]
    assert toolchain.startswith("RUN --mount=from=debs,source=/debs,target=/debs apt-get install -y "
                                "--no-install-recommends /debs/*.deb")
    assert "apt-get update" not in text  # offline: the image keeps no apt lists
    assert "nvcc --version | grep -q 'V13.0.88'" in toolchain
    assert "/opt/venv/bin/ninja" in toolchain and "nvvm/bin/cicc" in toolchain
    assert (ROOT / "tests" / "gpu-image.Dockerfile.dockerignore").is_file()


def test_the_test_image_is_a_published_image_with_tests_on_top() -> None:
    """The GPU tests run on the release's own layers: the test image is built FROM a published image by digest, adds an
    SSH server, the dev group (none of the image's packages may change) and the commit's tracked files, with
    llm_mini_infer/ a link to the installed package; only a dispatch publishes it, as :test-gpu-<commit>, and checks the
    layers."""
    text = re.sub(r"\\\n\s*", " ", TEST_IMAGE.read_text())
    froms = re.findall(r"^FROM (\S+)", text, re.M)
    assert froms == ["${UV_IMAGE}", "${TEST_DEBS}", "${TEST_BASE}"] and "USER root" in text
    assert "dpkg-query -W openssh-server rsync" in text
    (sync,) = [line for line in text.splitlines() if "uv sync" in line and not line.startswith("#")]
    assert "--mount=from=uv,source=/uv,target=/bin/uv" in sync
    assert "uv sync --locked --inexact --no-install-project --only-group dev" in sync
    assert "UV_PROJECT_ENVIRONMENT=/opt/venv UV_PYTHON=/usr/bin/python3.12" in sync
    assert "! grep -E '^ [-~] |virtual environment'" in sync
    assert "flashinfer_jit_cache/providers/sm120f/jit_cache | wc -l)\" = 3" in sync
    # FlashInfer's JIT is off in the release and on in the suite: any non-empty value turns it off, so it must be empty
    assert re.findall(r"^ENV FLASHINFER_DISABLE_JIT=(.*)$", text, re.M) == [""]
    assert sync.endswith("ln -s /opt/venv/lib/python3.12/site-packages/llm_mini_infer llm_mini_infer")
    job = workflow()["jobs"]["test-image"]
    assert job["needs"] == "image" and "inputs.test_image" in job["if"] and "workflow_dispatch" in job["if"]
    assert workflow()["jobs"]["image"]["outputs"]["cache"] == "${{ steps.tags.outputs.cache }}"
    st = {s.get("name") or s["uses"].split("@")[0]: s for s in job["steps"]}
    # the packages: test-debs on the release's os (the image job's cache), under a cache of their own, by digest
    assert st["Packages"]["id"] == "debs"  # TEST_DEBS reads steps.debs.outputs.digest
    debs = st["Packages"]["with"]
    assert debs["file"] == "deploy/Dockerfile" and debs["target"] == "test-debs" and debs["push"] is True
    assert debs["platforms"] == "linux/amd64" and debs["provenance"] is False and debs["sbom"] is False
    assert debs["tags"] == "${{ env.IMAGE }}:test-debs"
    assert debs["cache-to"] == "type=registry,ref=${{ env.IMAGE }}:buildcache-test-debs,mode=max"
    ref = "type=registry,ref=${{ env.IMAGE }}:buildcache-"
    assert debs["cache-from"].splitlines() == [f"{ref}test-debs", f"{ref}main",
                                               f"{ref}${{{{ needs.image.outputs.cache }}}}"]
    assert list(st).index("Packages") < list(st).index("Publish")
    base = st["Base and context"]["run"]
    # .dockerignore stays in the tree
    assert "git archive --format=tar HEAD -- . ':(exclude)llm_mini_infer' | tar" in base
    assert "tag=$IMAGE:test-gpu-${GITHUB_SHA::7}" in base
    publish = st["Publish"]["with"]
    assert publish["file"] == "tests/gpu-image.Dockerfile" and publish["push"] is True
    assert "TEST_BASE=${{ steps.base.outputs.base }}" in publish["build-args"]
    assert "TEST_DEBS=${{ env.IMAGE }}@${{ steps.debs.outputs.digest }}" in publish["build-args"]
    assert "assert test[:len(base)] == base" in st["The image's layers come first"]["run"]
