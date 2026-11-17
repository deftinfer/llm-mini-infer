# Development

This document is for contributors. [architecture.md](architecture.md) tells how the engine works, and [operations.md](operations.md) how to run it.

## The repository

```
llm_mini_infer/    the package
  api/             the frontend: raw ASGI app, OpenAI types (msgspec), SSE, images, API keys, the engine client
  render/          chat templates (Hugging Face's Jinja environment), the tokenizer, message normalization
  output/          the streaming detokenizer, stop strings
  engine/          the engine process: the step loop, IPC (ZMQ + msgspec), messages
  sched/           the scheduler and the step-plan contract (`StepPlan`)
  cache/           the radix tree, the KV and state pools, the encoder cache
  grammar/         structured output with llguidance
  runner/          the runner contract (base.py), the fake runner, the CUDA runners (cuda/)
  models/          Gemma 4 and Qwen3.8, their vision towers, Gemma 4's assistant, FP8 loading
  kernels/         our CUDA kernels (csrc/), the CUTLASS GEMMs (cutlass/), their build, torch references
  vision/          image fetch, header probe, sizing per family, the image pool
  config.py        the configuration and its derived defaults
  launcher.py      process startup and supervision; entrypoint.py, the image's; cli.py, `llm-mini-infer`
  hub.py           Hugging Face downloads; metrics.py; hostcheck.py and synthetic.py, the host check
tests/             the CPU and GPU suites, a folder per subsystem: sched/, structured/, tool_calls/, api/, runner/,
                   models/, vision/, kernels/, deploy/ (the image and the chart), repo/ (the repository's own);
                   gpu-image.Dockerfile, the GPU tests' image
  support/         what the tests share: harnesses, fuzz workloads, tiny models, the paths (paths.py)
  scripts/         the fixtures, fuzz, plan-diff, numerics, profiling and multi-turn cache tools (python -m tests.scripts.<tool>)
  data/            test data
bench/             benchmarks (bench/README.md)
deploy/            the image (Dockerfile), its configuration, docker compose, the Helm chart
tools/             pinned command-line tools (fetch_tools.py) ([Tools](#tools))
docs/              architecture.md, operations.md, development.md, vllm-compatibility.md, benchmarks.md
.github/           CI (workflows/ci.yml, scripts/), the image (workflows/image.yml), the issue form (ISSUE_TEMPLATE/)
results/           every tool's output (git-ignored)
README.md, CHANGELOG.md, SECURITY.md, CONTRIBUTING.md, LICENSE   the root files
```

A test goes in the folder of the code it exercises. No test module imports another: what tests share lives in `tests/support/`.

## Setup

llm-mini-infer uses [uv](https://docs.astral.sh/uv/) and Python 3.12. One lock, `uv.lock`, covers macOS on Apple Silicon (torch's CPU wheels) and Linux x86_64 (torch's CUDA 13.0 wheels).

```bash
uv sync                                  # the package and the dev dependency group
uv run tests/scripts/fetch_fixtures.py   # the tests' model fixtures, about 120 MB
uv run tools/fetch_tools.py helm         # the chart's tests skip without it
uv run pytest -q                         # the CPU suite
uv run ruff check .
uv run llm-mini-infer serve --model tests/.fixtures/models/gemma-4-31b-it --runner fake --port 8000   # no GPU
```

On a Linux GPU host, `uv sync --extra cuda --extra images` adds FlashInfer, and the kernels need `nvcc` from a CUDA 13 toolkit. `tests/scripts/fetch_fixtures.py` downloads the model fixtures (no weights) into the git-ignored `tests/.fixtures/` and checks their sha256. A test without its fixture skips.

## The CPU suite

`uv run pytest -q` runs the tests that need no GPU: the scheduler and cache over the strict fake runner (which checks the step-plan contract), the CUDA runner and models on the torch reference kernels against transformers, the API and the server end to end, the Helm chart, and checks of the Dockerfile, workflows and lock. `-n auto --dist loadfile` runs it in parallel processes (pytest-xdist), one file to a process. A test whose timing bound needs a whole core carries the `serial` mark, and CI runs it alone after the others. CI runs the suite on Linux x86_64. A GPU test carries the `gpu` mark (`cutlass` or `flashinfer` when it also needs those libraries). The conftest skips it where its needs are absent; `pytest -m gpu` selects the GPU tests.

The tests that compare with the reference vLLM read its checkout from `LLM_MINI_INFER_VLLM_SRC` (default: a `vllm` checkout beside this repository), at the tag `REV` in `tests/support/vllm_ref.py`. They skip without a checkout and fail on one that lacks the tag. The fuzz suites run a few hundred examples per test. Copy a failing example worth keeping into `tests/data/fuzz_regressions/`, which every run replays.

## The GPU suite

On an SM120 GPU, the same `tests/` add the kernels against their references, CUDA graphs against eager runs, the FP8 GEMMs, batch invariance, and both CUDA runners through the engine loop. The suite runs in **the GPU tests' image**. Without an SM120 GPU, the GPU suite is optional for a contribution: CI runs the CPU suite.

To build the image on an SM120 host, follow the recipe in the header of `tests/gpu-image.Dockerfile`. It builds on a published `llm-mini-infer` image and adds the test tools. Then start a shell in it: `docker run --rm -it --gpus all -v llm-mini-infer-cache:/cache --entrypoint bash <the image>`. The working directory, `/opt/llm-mini-infer/test`, holds the tests. The maintainer publishes the image as `:test-gpu-<commit>` through the image workflow, on a dispatch with `test_image` and `publish` on.

In the shell, fetch the fixtures (`python3 tests/scripts/fetch_fixtures.py`) and check the host (`llm-mini-infer hostcheck --no-server`). Run `tests/api/test_api_cuda.py` alone (about 30 s), then `python3 -m pytest tests -m gpu` and the whole suite. The suite runs the CUDA runners under the sync check at error (`runner.cuda.sync_check`; `LLM_MINI_INFER_TEST_SYNC_CHECK=warn` or `off` for another mode). A sync in a step's launch fails its test, named at its line.

**The batch-invariance golden.** `tests/runner/test_batch_invariance_gpu.py` and `tests/runner/test_qwen_batch_invariance_gpu.py` compare their bits with `tests/data/golden/batch_invariance.json`: one entry per case, GPU (architecture and SM count, `sm_120-188sm`) and set of tiny weights, with the stack as metadata. A new GPU or an intended change needs a new entry, recorded on that GPU:

```bash
python3 -m pytest tests/runner/test_batch_invariance_gpu.py tests/runner/test_qwen_batch_invariance_gpu.py -q --refresh-golden
python -m tests.support.golden tests/data/golden/batch_invariance.json   # then copy the file out of the image and commit it
```

## Fuzzing

Fuzz every change to the scheduler or the cache. Each suite but `radix` sends random workloads (prefixes, images, grammars, drafts, aborts, priorities, memory pressure) to the scheduler and the strict fake runner. Run a full campaign before you merge:

```bash
uv run python -m tests.scripts.fuzz_campaign      # every suite x 4 seeds x 20,000 examples
uv run python -m tests.scripts.fuzz_campaign --replay results/fuzz/<time>/failures/<example>.json
```

The logs go to `results/fuzz/<time>/`. A new class of hazard gets its own suite (`SUITES` in `tests/scripts/fuzz_campaign.py`). `LLM_MINI_INFER_TEST_FULL_CHECKS=1` runs the strict checks' slower reference implementations, for a pass after a change to the checks. `tests.scripts.fuzz_profile` shows each suite's time.

## The plan-diff harness

```bash
uv run python -m tests.scripts.plandiff --base main                                 # the working tree against main
uv run python -m tests.scripts.plandiff --base main --new HEAD --expect-changed structured,window
```

The harness runs the fuzz suites' workloads through both trees and compares every `StepPlan`, mask set and step report. A refactor must report "identical". A change may change only the suites that `--expect-changed` names. With `--compare-draws` (after a change to the fuzz strategies) both trees draw the workloads, and a seed must draw the same ones in both. The fuzz workloads run without batch invariance's cuts. `LLM_MINI_INFER_TEST_BATCH_INVARIANT=1` turns them on, for a second pass with `fuzz_campaign --seeds 1 2 3 4`.

## Numerics

Three GPU tools measure accuracy against fp32 on the real checkpoints: `numerics_forced`, `numerics_fp32` and `numerics_kernels`. Run each inside the GPU tests' image in `/opt/llm-mini-infer/test` with `python3 -m tests.scripts.<tool>`; its docstring describes the subcommands. `numerics_forced` forces greedy references through each arm (batch invariance on or off), because greedy answers differ at near ties: the arms' acceptance must compare on the same tokens. `numerics_fp32` replays the captures in fp32, and `numerics_kernels` probes single kernels.

## Measurement rules

Follow these rules when you compare llm-mini-infer with vLLM, or one configuration with another.

- **Parity bars:** a variant passes when its top-1 agreement with vLLM is not significantly below the lowest agreement among vLLM's own variants (paired 95% bootstrap). Measure it teacher-forced over whole sequences, and for greedy decoding at batch 1 over the first 32 tokens. A fixed 90% greedy bar is unreachable: vLLM's own variants agree on only 72% to 79%.
- **FP8 noise:** FP8 widens the noise of every engine. Compare FP8 greedy outputs only with FP8 references.
- **vLLM's FP8 paths:** vLLM serves a block-FP8 checkpoint with DeepGEMM by default for Gemma 4, and on its CUTLASS path for the Qwen3.5 family. The kernel matters to vLLM's batch invariance. With DeepGEMM, two fresh vLLM servers matched on 57.2% of answers. With `VLLM_USE_DEEP_GEMM=0`, they matched on all. Name the path that each comparison uses. llm-mini-infer's FP8 GEMM equals the CUTLASS path ([numerics](vllm-compatibility.md#numerics)).
- **A cached prefix freezes one FP8 draw** for every request that reuses it. Average engines over draws (prefix cache off, or several chunkings), or compare per server instance. Never compare one instance with another as if it were the engine.
- **The standing paired test:** send the same corpus twice to both engines at a stated concurrency. Pair the answers and cluster them by source. A difference counts when it is at least 2 points with Benjamini-Hochberg q < 0.05 in both rounds, while each engine agrees with itself.
- **Answer reproducibility:** every kernel on the answer's path keeps `tests/runner/test_batch_invariance_gpu.py` green. See [batch invariance](architecture.md#batch-invariance).
- **Closed loops arrive in waves:** a p50 can fall in a wave's late group while the mean, p90 and p99 are lower. Repeat close cells, and report the mean and p90 beside the p50. Or use an open loop.
- **Under MTP:** compare TPOT per request, (e2e − TTFT) / (tokens − 1). Count acceptance as tokens delivered per verify step, because vLLM's counters keep drafts that it later trims. llm-mini-infer also delivers tokens that no verify step makes, its forced prefixes and an EOS-free finish's EOS (`llm_mini_infer:grammar_forced_tokens_total`, `llm_mini_infer:grammar_eos_free_finishes_total`): subtract them for its drafter's own acceptance.

## Building the image

`.github/workflows/image.yml` builds `deploy/Dockerfile` on GitHub's amd64 runners, checks it on the CPU (`tests/deploy/test_image_checks.py`), and publishes it. Start a branch build with `gh workflow run image.yml --ref <branch>`.

| trigger | tags |
|---|---|
| a push to main that changes the image's inputs (`paths` in `image.yml`) | `:<commit>` (7 characters, immutable) and `:main` |
| a tag `v<version>` | `:<version>` and `:<commit>`: a release |
| a dispatch on another branch | `:test-<branch>`. Inputs: `publish`, `cache`, `test_image` (with `publish` only) |

The workflow's jobs run only where the repository variable `PUBLISH_IMAGE` is true, and publish under the repository's own name (`ghcr.io/<owner>/llm-mini-infer`).

`deploy/Dockerfile`'s header has the command to build by hand, on amd64. The build keeps its layers in the registry, so a Python change rebuilds in minutes. The kernels are built in their own stage, so the runtime has no nvcc. The image serves only the families whose FlashInfer modules it holds: a new family needs its modules in `FLASHINFER_AOT`, the module count that the build, `tests/gpu-image.Dockerfile` and `tests/deploy/test_image_build.py` check, and the inventory's record (`tests/data/image/`).

### The kernels' recorded bytes

The same sources, flags, toolchain, directories and mtimes give the same kernel bytes. `KERNELS_SHA256` in `deploy/Dockerfile` records each library's sha256 and path, whose directory is the architecture and a hash of sources, nvcc's version and flags. A default build fails on other bytes.

A change to `llm_mini_infer/kernels/csrc/` or `llm_mini_infer/kernels/cutlass/`, even to a comment, changes the bytes. Update `KERNELS_SHA256` with the values that the failed build prints, and the table in `tests/data/image/README.md`: `tests/deploy/test_image_build.py` checks that they agree.

## Releasing

These are maintainer notes. Do them in this order:

1. Set the version with `uv version <version>`. Set it also in the chart's `appVersion`, in `deploy/helm/examples/hostcheck-job.yaml` and in `tests/data/image/freeze.txt`. Set it in the image names of `README.md`, `docs/operations.md` and `deploy/docker-compose.yaml` too. Increase the chart's own `version` if the chart changed. `tests/deploy/test_image_build.py` checks that `pyproject.toml`, the chart, the Job and those image names agree.
2. Write the version's entry in `CHANGELOG.md`. The tag message and the GitHub release body come from it.
3. Merge to main, create the annotated tag `v<version>` on the commit, and push the tag. The image workflow publishes `:<version>`.
4. Create the GitHub release for the tag.

## CI and style

`.github/workflows/ci.yml` runs on pushes to main, pull requests and dispatches (`gh workflow run ci.yml --ref <branch>`). A push to main that changes documentation only starts no job, and a pull request starts every job. Its jobs: ruff, the Helm chart (`helm lint`, `helm template`, kubeconform), the CPU suite in three parallel shards (by directory, each in two workers; a test whose servers leave the runner's memory no room for a second worker's takes the `serial` mark and runs after them, as `ci.yml` describes), `tools/`'s tests, and every fuzz suite on two seeds of 500 examples. The fuzz job runs only when `llm_mini_infer/`, `tests/`, the lock or `ci.yml` change beyond comments and docstrings (`.github/scripts/ci_changes.py`). The job `required` passes when every other job passed, and the fuzz job also when it was skipped. It is the check that main requires before a pull request merges.

## Tools

`uv run tools/fetch_tools.py` fetches helm, kubeconform, kind, kubectl and gh, pinned and checked, into `tools/bin/`. CI's `tools` job runs its test.

## Writing

Commit messages have a subject of at most 72 characters and the details in the body. Prose (Markdown, commit bodies) follows ASD-STE100's writing rules, without its dictionary. Write each paragraph or list item on one line. Code lines split at ruff's 120 characters, and docstrings and comments follow the same split and rules. The rules: sentences of at most 20 words for an instruction and 25 for a description, one instruction per sentence, the active voice, one term for one concept, noun groups of at most three words, paragraphs of one topic and at most six sentences, no idioms, and no history notes. Put code names in backticks.

## Licence notices

LICENSE covers the repository: our own files carry no licence headers. An adapted file keeps its notices and says it was changed. LICENSE's sections after the Apache text list these files. `tests/repo/test_licence_headers.py` checks that every listed file exists, that code adapted from vLLM or transformers carries its copyright line, that no file with an origin's notice is unlisted, that the Gemma 4 fixtures match their pins, and that the licence texts are whole.
