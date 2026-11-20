# llm-mini-infer's image (docs/operations.md, "The image"; docs/development.md, "Building the image").
# GitHub Actions builds, checks and publishes it to ghcr.io/deftinfer/llm-mini-infer (.github/workflows/image.yml). By hand:
#
#   docker buildx build --platform linux/amd64 -f deploy/Dockerfile -t llm-mini-infer:$(git rev-parse --short=7 HEAD) \
#       --build-arg VERSION=$(uv version --short) --build-arg GIT_COMMIT=$(git rev-parse --short=7 HEAD) \
#       --build-arg GIT_REVISION=$(git rev-parse HEAD) .
#
# nvidia/cuda's base image plus only what llm-mini-infer loads. Stages:
#   os       the base, its Ubuntu packages upgraded, Python 3.12 and the C runtime (apt), the llm-mini-infer user and
#            /cache
#   deps     /opt/venv: `uv sync --locked --no-dev --no-install-project` without flashinfer-cubin and the jit-cache
#            providers of other GPUs, then FlashInfer's AOT modules cut to the three llm-mini-infer loads
#   kernels  the compilers (apt) and our kernels, which must be the recorded bytes (KERNELS_SHA256, or the build fails)
#   debs, test-debs
#            the GPU tests' image's apt packages, downloaded on os (never in the release; tests/gpu-image.Dockerfile)
#   project  llm-mini-infer itself (`uv sync --locked --no-editable`), staged apart from the dependencies
#   runtime  deps itself, then our kernels, the configuration, the third-party notices and llm-mini-infer, in that
#            order, and the build's version and commit last: a Python-only change rebuilds the last layers, a kernel
#            change one more
# uv is mounted into each RUN that calls it, never copied: no layer holds it. The GPU tests' image (never a release)
# is built on a published one by tests/gpu-image.Dockerfile.
ARG CUDA_BASE=nvidia/cuda:13.0.2-base-ubuntu24.04@sha256:2ab6381d970b211fb93853796dc6707eb8a72575a375c422b17cf4d8b2641701
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.13.5@sha256:32a7bb39b496babbd997b0a39a4d66c126bdaf5661b302bf6f06abda8deca6ff
# The GPU-validated stack the image keeps (its label): the five Python packages pyproject pins (torch with its CUDA
# libraries, FlashInfer, llguidance, tokenizers, Pillow with torchvision) at the versions the GPU suite and the
# image's comparison runs; tests/data/image/ records the image's freeze, apt list and inventory on it
ARG STACK="torch==2.14.1+cu130 torchvision==0.29.1+cu130 flashinfer-python==0.7.2 llguidance==1.9.1 tokenizers==0.23.3 pillow==13.0.0"
# Ubuntu's archive at a fixed time, its security updates included: the os and kernels stages install their Ubuntu
# packages from it at pinned versions (tests/data/image/apt.txt and toolchain.json); the image's apt source names it,
# where the notices find the GPL and LGPL packages' sources
ARG UBUNTU_SNAPSHOT=20261018T000000Z

FROM ${UV_IMAGE} AS uv

FROM ${CUDA_BASE} AS os
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG DEBIAN_FRONTEND=noninteractive
# The Ubuntu snapshot (UBUNTU_SNAPSHOT) alone: the base's own packages are upgraded to their versions there (its tag is
# not rebuilt, and its digest above holds older packages, OpenSSL 3.0.13-0ubuntu3.6 among them), then the runtime's
# are installed at the newest there. NVIDIA's CUDA repository (the base's cuda.list) is set aside meanwhile: the base's
# CUDA packages (its runtime, the driver's forward compatibility) stay the base's, and nothing follows that repository's
# moving versions; the kernels stage installs from it. The runtime's packages: the interpreter, and the C and Python
# runtime an llm-mini-infer process mapped (tests/data/image/inventory.json).
ARG UBUNTU_SNAPSHOT
RUN printf 'Types: deb\nURIs: https://snapshot.ubuntu.com/ubuntu/%s/\nSuites: noble noble-updates noble-security\nComponents: main universe\nSigned-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg\n' \
        "$UBUNTU_SNAPSHOT" > /etc/apt/sources.list.d/ubuntu.sources \
 && mv /etc/apt/sources.list.d/cuda.list /tmp/cuda.list \
 && apt-get update \
 && apt-get upgrade -y \
 && apt-get install -y --no-install-recommends \
        libbz2-1.0=1.0.8-5.1ubuntu0.1 \
        libc6=2.39-0ubuntu8.9 \
        libexpat1=2.6.1-2ubuntu0.6 \
        libffi8=3.4.6-1build1 \
        libfribidi0=1.0.13-3build1 \
        libgcc-s1=14.2.0-4ubuntu2~24.04.1 \
        liblzma5=5.6.1+really5.4.5-1ubuntu0.3 \
        libpython3.12-minimal=3.12.3-1ubuntu0.17 \
        libpython3.12-stdlib=3.12.3-1ubuntu0.17 \
        libssl3t64=3.0.13-0ubuntu3.16 \
        libstdc++6=14.2.0-4ubuntu2~24.04.1 \
        python3.12=3.12.3-1ubuntu0.17 \
        python3.12-minimal=3.12.3-1ubuntu0.17 \
        zlib1g=1:1.3.dfsg-3.1ubuntu2.2 \
 && mv /tmp/cuda.list /etc/apt/sources.list.d/cuda.list \
 && rm -rf /var/lib/apt/lists/* \
 && groupadd --gid 10001 llm-mini-infer \
 && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin llm-mini-infer \
 && install -d -o llm-mini-infer -g llm-mini-infer /cache

FROM os AS deps
# uv's Docker guide: bytecode compiled at install (the server's user cannot write the venv), files copied out of the
# cache mount, the base image's Python only; the venv at the runtime's path. The runtime, built on this stage, keeps
# these variables, which nothing there reads: it has no uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=/usr/bin/python3.12 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_HTTP_TIMEOUT=300
WORKDIR /opt/llm-mini-infer/src
# FlashInfer's AOT modules llm-mini-infer loads (the inventory: batch prefill with BF16 KV at head dims 128 and 256, the
# latter with and without the sliding window). flashinfer-jit-cache is a shim that finds per-GPU providers of AOT
# modules by their entry points: it is installed with SM120's provider alone (flashinfer-jit-cache-sm120f), whose
# modules are cut to these three. flashinfer-cubin is not installed (no process opened a cubin on SM120): FlashInfer
# imports it only when it is there, and otherwise looks for cubins under its cache,
# $FLASHINFER_WORKSPACE_BASE/.cache/flashinfer/cubins.
# tests/deploy/test_image_checks.py checks the three modules by sha256.
ARG FLASHINFER_AOT="\
batch_prefill_with_kv_cache_dtype_q_bf16_dtype_kv_bf16_dtype_o_bf16_dtype_idx_i32_head_dim_qk_128_head_dim_vo_128_posenc_0_use_swa_False_use_logits_cap_False_f16qk_False_kv_stride_equal \
batch_prefill_with_kv_cache_dtype_q_bf16_dtype_kv_bf16_dtype_o_bf16_dtype_idx_i32_head_dim_qk_256_head_dim_vo_256_posenc_0_use_swa_False_use_logits_cap_False_f16qk_False_kv_stride_equal \
batch_prefill_with_kv_cache_dtype_q_bf16_dtype_kv_bf16_dtype_o_bf16_dtype_idx_i32_head_dim_qk_256_head_dim_vo_256_posenc_0_use_swa_True_use_logits_cap_False_f16qk_False_kv_stride_equal"
# The dependencies' layer: only a change to uv.lock or pyproject.toml (or to the instructions above) rebuilds it. The
# cut runs in the same RUN, so no layer holds what it removes; FlashInfer's AOT wheel leaves the cache mount too, which
# on the runner's disk would otherwise hold it unpacked for the rest of the build.
RUN --mount=from=uv,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --locked --no-dev --no-install-project --extra cuda --extra images --extra image \
        --no-install-package flashinfer-cubin \
        --no-install-package flashinfer-jit-cache-sm80 --no-install-package flashinfer-jit-cache-sm89 \
        --no-install-package flashinfer-jit-cache-sm90a --no-install-package flashinfer-jit-cache-sm100a \
        --no-install-package flashinfer-jit-cache-sm103a \
 && cd /opt/venv/lib/python3.12/site-packages/flashinfer_jit_cache/providers/sm120f/jit_cache \
 && for m in $FLASHINFER_AOT; do test -d "$m"; done \
 && find . -mindepth 1 -maxdepth 1 $(printf '! -name %s ' $FLASHINFER_AOT) -exec rm -r {} + \
 && test "$(ls | wc -l)" = 3 \
 && uv cache clean flashinfer-jit-cache-sm120f

FROM os AS kernels
ARG DEBIAN_FRONTEND=noninteractive
# What our kernel build reads (nvcc -dryrun, nvcc -M, dpkg -S: tests/data/image/toolchain.json), at the versions the
# kernels' recorded bytes were built with
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        binutils=2.42-4ubuntu2.10 \
        cuda-cccl-13-0=13.0.85-1 \
        cuda-crt-13-0=13.0.88-1 \
        cuda-cudart-dev-13-0=13.0.96-1 \
        cuda-nvcc-13-0=13.0.88-1 \
        g++=4:13.2.0-7ubuntu1 \
        g++-13-x86-64-linux-gnu=13.3.0-6ubuntu2~24.04.1 \
        gcc=4:13.2.0-7ubuntu1 \
        gcc-13-x86-64-linux-gnu=13.3.0-6ubuntu2~24.04.1 \
        libc6-dev=2.39-0ubuntu8.9 \
        libgcc-13-dev=13.3.0-6ubuntu2~24.04.1 \
        libnvvm-13-0=13.0.88-1 \
        libstdc++-13-dev=13.3.0-6ubuntu2~24.04.1 \
        linux-libc-dev=6.8.0-147.147 \
 && rm -rf /var/lib/apt/lists/*
# FlashInfer's CUTLASS headers (our GEMMs include them) at the include path the recorded bytes were built with: the
# device code's line tables record include paths as they do source paths, so the libraries keep their bytes only from
# the same directories
COPY --from=deps /opt/venv/lib/python3.12/site-packages/flashinfer/data/cutlass \
     /usr/local/lib/python3.12/dist-packages/flashinfer/data/cutlass
ENV CUDA_HOME=/usr/local/cuda \
    CUTLASS_HOME=/usr/local/lib/python3.12/dist-packages/flashinfer/data/cutlass
# Our CUDA kernels and the CUTLASS GEMMs, prebuilt for ARCHS (nvcc needs no GPU); the startup loads them from here.
# Only what their build reads comes in (build.py imports the standard library alone, and the sources include only each
# other), from /opt/llm-mini-infer/src, the directory the line tables hold. The jobs are capped by the builder's memory
# (build.py's log line names the choice).
WORKDIR /opt/llm-mini-infer/src
COPY llm_mini_infer/__init__.py llm_mini_infer/__init__.py
COPY llm_mini_infer/kernels/__init__.py llm_mini_infer/kernels/build.py llm_mini_infer/kernels/
COPY llm_mini_infer/kernels/csrc llm_mini_infer/kernels/csrc
COPY llm_mini_infer/kernels/cutlass llm_mini_infer/kernels/cutlass
# The GPUs to prebuild for (nvcc arch without sm_): 120 = SM120 (RTX PRO 6000 Blackwell, RTX 5090)
ARG ARCHS="120"
# With -lineinfo, ptxas writes each file's mtime (and size) into the line tables: our sources' and CUTLASS's are set
# to 0 first, as every build of these libraries must (the CUDA headers keep their packages' own mtimes), or no two
# builds give the same bytes
RUN find /opt/llm-mini-infer/src "$CUTLASS_HOME" -type f -exec touch -d @0 {} + \
 && LLM_MINI_INFER_KERNEL_CACHE=/opt/llm-mini-infer/kernels python3.12 -B -m llm_mini_infer.kernels.build \
        $(for a in $ARCHS; do echo --arch $a; done) \
 && cd /opt/llm-mini-infer/kernels && find . -name '*.so' -printf '%P\n' | sort | xargs sha256sum \
 && cat nvcc-version.txt
# Our kernels' bytes for ARCHS=120 (tests/deploy/test_image_checks.py reads them here): sha256, then the path under
# /opt/llm-mini-infer/kernels, whose directory is the cache key (sources, `nvcc --version`, flags). The build fails on
# any other bytes or key, so a kernel change updates these (docs/development.md, "The kernels' recorded bytes").
ARG KERNELS_SHA256="\
961c25fe5b60154fa4eefe9d6d23c23a11c4a9e945ee9ebb5e906fcc1da8ffa0:120-debd3f77f9d049cc/libllm_mini_infer_kernels.so \
f3e7d12493d607e2d8887fd995c753278e450a5d47a450757d958cc7280d8d98:cutlass-120a-da5aebbd96ce381f/libllm_mini_infer_cutlass.so"
RUN if [ "$ARCHS" = 120 ]; then \
        cd /opt/llm-mini-infer/kernels && for e in $KERNELS_SHA256; do echo "${e%%:*}  ${e#*:}"; done \
            | sha256sum --check --strict; \
    else \
        echo "ARCHS=$ARCHS: KERNELS_SHA256 holds ARCHS=120's libraries; not checked"; \
    fi

# The GPU tests' image's packages (tests/gpu-image.Dockerfile), never the release's: downloaded on the runtime's own
# userland (the runtime adds no package to os), so they and their dependencies install there offline, with no apt-get
# update. An SSH server and rsync (a development tool reaches a GPU host over SSH), and the build toolchain the suite's
# compiler tests need and serving does not (FlashInfer's JIT for the tiny models' head dims, Triton's launcher, our
# kernels' build), from the snapshot and NVIDIA's CUDA repository: the kernels stage's packages at its versions, nvcc's
# libnvptxcompiler at nvcc's, Python 3.12's headers at the interpreter's. The image workflow's test-image job publishes
# test-debs (/debs alone) under a registry cache of its own, so the mirrors are reached only when this list or os
# changes. The RUN's last step simulates that install with no lists left, so a set that misses a dependency fails here.
# runtime stays the last stage: the build's default target.
FROM os AS debs
ARG DEBIAN_FRONTEND=noninteractive
RUN mkdir /debs \
 && apt-get -o Acquire::Retries=10 update \
 && apt-get -o Acquire::Retries=10 -o Dir::Cache::archives=/debs install -y --no-install-recommends --download-only \
        openssh-server \
        rsync \
        binutils=2.42-4ubuntu2.10 \
        cuda-cccl-13-0=13.0.85-1 \
        cuda-crt-13-0=13.0.88-1 \
        cuda-cudart-dev-13-0=13.0.96-1 \
        cuda-nvcc-13-0=13.0.88-1 \
        g++=4:13.2.0-7ubuntu1 \
        g++-13-x86-64-linux-gnu=13.3.0-6ubuntu2~24.04.1 \
        gcc=4:13.2.0-7ubuntu1 \
        gcc-13-x86-64-linux-gnu=13.3.0-6ubuntu2~24.04.1 \
        libc6-dev=2.39-0ubuntu8.9 \
        libgcc-13-dev=13.3.0-6ubuntu2~24.04.1 \
        libnvptxcompiler-13-0=13.0.88-1 \
        libnvvm-13-0=13.0.88-1 \
        libpython3.12-dev=3.12.3-1ubuntu0.17 \
        libstdc++-13-dev=13.3.0-6ubuntu2~24.04.1 \
        linux-libc-dev=6.8.0-147.147 \
        python3.12-dev=3.12.3-1ubuntu0.17 \
 && rm -rf /debs/partial /debs/lock /var/lib/apt/lists/* \
 && apt-get -s install --no-install-recommends /debs/*.deb >/dev/null

FROM scratch AS test-debs
COPY --from=debs /debs /debs

FROM deps AS project
# llm-mini-infer's own wheel (hatchling, pinned in pyproject's [build-system]) into the venv, then staged for one COPY:
# the runtime's venv layer is not unpacked to add it
COPY pyproject.toml uv.lock ./
COPY README.md LICENSE ./
COPY llm_mini_infer llm_mini_infer
RUN --mount=from=uv,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable --extra cuda --extra images --extra image \
        --no-install-package flashinfer-cubin \
        --no-install-package flashinfer-jit-cache-sm80 --no-install-package flashinfer-jit-cache-sm89 \
        --no-install-package flashinfer-jit-cache-sm90a --no-install-package flashinfer-jit-cache-sm100a \
        --no-install-package flashinfer-jit-cache-sm103a \
 && site=lib/python3.12/site-packages \
 && mkdir -p /stage/opt/venv/bin /stage/opt/venv/$site \
 && mv /opt/venv/bin/llm-mini-infer /stage/opt/venv/bin/ \
 && mv /opt/venv/$site/llm_mini_infer /opt/venv/$site/llm_mini_infer-*.dist-info /stage/opt/venv/$site/

# On deps, whose venv layer is then the image's own: neither a second copy of it in the image and the cache, nor its
# compression and upload twice
FROM deps AS runtime
ARG CUDA_BASE
ARG STACK
# Caches: our kernels prebuilt into the image (read-only), the rest on the cache volume at /cache (compose's volume,
# the Helm chart's emptyDir or claim), so only a first start writes them: FlashInfer's rendered sources of the AOT
# modules it loads, CuTe DSL's compiles and the driver's PTX cache. The Hugging Face cache is there too: a model given
# as a repo id downloads into it at startup (llm_mini_infer/hub.py; HF_TOKEN from the environment), a mounted checkpoint
# (compose) never touches it.
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:${PATH} \
    CUDA_HOME=/usr/local/cuda \
    PYTHONUNBUFFERED=1 \
    LLM_MINI_INFER_CONFIG=/etc/llm-mini-infer/config.yaml \
    LLM_MINI_INFER_KERNEL_PREBUILT=/opt/llm-mini-infer/kernels \
    FLASHINFER_WORKSPACE_BASE=/cache \
    CUTE_DSL_CACHE_DIR=/cache/cute-dsl \
    CUDA_CACHE_PATH=/cache/nv \
    HF_HOME=/cache/huggingface \
    TOKENIZERS_PARALLELISM=false
# FlashInfer never builds a module here: one that the image lacks fails with its MissingJITCacheError ("JIT
# compilation is disabled"), not with an nvcc or ninja error. Any non-empty value turns the JIT off, "0" too
ENV FLASHINFER_DISABLE_JIT=1
# --link: each layer stands alone, so a rebuild neither unpacks the venv under it nor misses on a change above it.
# No nvcc or ptxas: our kernels' keys take nvcc's version from the build's record beside them (llm_mini_infer/kernels/
# build.py), and FlashInfer's import, which asks `nvcc --version` where nvcc exists (an SM 12.x GPU visible), then
# takes torch's CUDA version; flashinfer.triton, if imported, looks for ptxas and with none leaves Triton its own
COPY --link --from=kernels /opt/llm-mini-infer/kernels /opt/llm-mini-infer/kernels
COPY --link deploy/config.yaml /etc/llm-mini-infer/config.yaml
# The third-party notices: where each licence is in the image, NVIDIA's terms and where the GPL and LGPL packages'
# sources are; tests/deploy/test_image_checks.py checks what it says against the image
COPY --link deploy/THIRD_PARTY_NOTICES /usr/share/doc/llm-mini-infer/THIRD_PARTY_NOTICES
COPY --link --from=project /stage/ /
LABEL org.opencontainers.image.title="llm-mini-infer" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.description="A lean, all-Python LLM serving engine (nvidia/cuda plus uv.lock, at ${STACK})" \
      org.opencontainers.image.source="https://github.com/deftinfer/llm-mini-infer" \
      org.opencontainers.image.base.name="${CUDA_BASE}" \
      io.github.deftinfer.llm-mini-infer.stack="${STACK}"
USER 10001:10001
WORKDIR /
EXPOSE 8000
ENTRYPOINT ["python3", "-m", "llm_mini_infer.entrypoint"]
CMD ["serve"]
# The build's version and commit last (no layer follows): VERSION is pyproject's (a release's tag is v<VERSION>),
# GIT_COMMIT the build id (LLM_MINI_INFER_BUILD: `<version>+<commit>` in each response's engine marker,
# llm_mini_infer/version.py; the image's commit tag), GIT_REVISION the full commit
ARG VERSION=unknown
ARG GIT_COMMIT=unknown
ARG GIT_REVISION=${GIT_COMMIT}
ENV LLM_MINI_INFER_BUILD=${GIT_COMMIT}
LABEL org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${GIT_REVISION}"
