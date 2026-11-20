# The image's reference data

These files record the image's stack (`deploy/Dockerfile`) and what llm-mini-infer loads on it. `tests/deploy/test_image_build.py` and `tests/deploy/test_image_checks.py` check the image against them.

| file | what it holds |
|---|---|
| `freeze.txt` | the image's Python distributions, `name==version` |
| `apt.txt` | the image's apt packages, `package=version KiB` |
| `inventory.json` | what llm-mini-infer's processes imported, mapped, opened and started while serving Gemma 4 and Qwen3.8 and running the host check, by run and process role, and the FlashInfer modules they loaded (`aot_sha256`) |
| `toolchain.json` | the programs and the header packages that the kernel build uses, at their versions |

From them, tests check that the pinned stack's lock is the freeze's, that the runtime's apt packages are the interpreter and those its processes mapped, that the kernels stage's packages stay out of the runtime, and that FlashInfer's SM120 provider holds only the loaded modules. No process opened a cubin, so the image omits FlashInfer's cubin package.

## Recording them

When the stack changes (`pyproject.toml`'s pins, `CUDA_BASE`, `UBUNTU_SNAPSHOT` or the apt pins), the files are recorded again from the new image by serving it on a GPU. The first build needs them already, so write them by hand from the lock, the base and the previous record, and replace them after the build.

## The kernels' recorded bytes

The cache key and sha256 of each kernel library, as `KERNELS_SHA256` in `deploy/Dockerfile` records them ([the kernels' recorded bytes](../../../docs/development.md#the-kernels-recorded-bytes)). `tests/deploy/test_image_build.py` checks that the two agree.

| library | cache key | sha256 (ARCHS=120) |
|---|---|---|
| `libllm_mini_infer_kernels.so` | `120-50a18689d8204ca4` | `9f6c720a472fb0eed76487e41dcd38791669b67746b65a71ba2f7e7faa0821a1` |
| `libllm_mini_infer_cutlass.so` | `cutlass-120a-da5aebbd96ce381f` | `f3e7d12493d607e2d8887fd995c753278e450a5d47a450757d958cc7280d8d98` |
