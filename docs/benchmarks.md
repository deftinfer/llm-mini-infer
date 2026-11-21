# Benchmarks

This page summarizes a comparison of llm-mini-infer with vLLM v0.33.0 and SGLang v0.5.23 on one GPU. The record is [bench/reports/v0.8.0/](../bench/reports/v0.8.0/README.md): its README holds the method, the definitions and every deviation, [summary.md](../bench/reports/v0.8.0/summary.md) and `summary.json` hold the tables, and `commands.sh` holds the commands. Where this page and the record differ, the record is right.

**The author of llm-mini-infer wrote the harness.** llm-mini-infer is built for this GPU and these two model families. vLLM and SGLang are general-purpose engines, run with their documented recipe flags.

## Conditions

- One NVIDIA RTX PRO 6000 Blackwell Server Edition at 600 W, on three hosts of the same type (`p1`, `p2`, `p3`).
- Gemma 4 31B FP8 with its assistant (5 drafts), and Qwen3.8 27B FP8 with its MTP layer (3 drafts). Every engine uses the same drafter and draft count.
- A context of 8,184 tokens and 64 sequences (SGLang on Qwen3.8: 48, below).
- vLLM v0.33.0 and SGLang v0.5.23 from their CUDA 13 wheels, pinned to the package versions of their vendor images.
- Synthetic workloads: scene (an image and a strict JSON schema; each engine's own answers) and chat (1,024-token prompts and 256-token replies under `ignore_eos`, so every engine decodes the same number of tokens).
- Streaming at temperature 0, with the client on the same host.
- llm-mini-infer's speed rows ran with batch invariance on, its default at this context. vLLM's and SGLang's ran with it off, their default.
- Each cell is one run, and no load was run again.
- Results move with versions.

## Gemma 4 31B FP8

| | llm-mini-infer | vLLM v0.33.0 | SGLang v0.5.23 |
|---|---|---|---|
| [TPOT p50, 1 in flight, scene (ms; `p1`)](../bench/reports/v0.8.0/summary.md#gemma-a1-1-in-flight-ms-tokens) | 5.86 | 7.07 | 12.27 |
| [TPOT p50, 1 in flight, chat (ms; `p1`)](../bench/reports/v0.8.0/summary.md#gemma-a1-1-in-flight-ms-tokens) | 13.97 | 19.21 | 23.13 |
| [Requests/s, 64 in flight, scene (`p1`)](../bench/reports/v0.8.0/summary.md#gemma-a1-64-in-flight-ms-tokens) | 7.90 (232 tokens a reply) | 6.58 (195) | 3.82 (176) |
| [Tokens/s, 64 in flight, chat (`p1`)](../bench/reports/v0.8.0/summary.md#gemma-a1-64-in-flight-ms-tokens) | 661 | 657 | 316 |
| [Highest rate within the latency bound, scene (requests/s; `p2`)](../bench/reports/v0.8.0/summary.md#gemma-a2-highest-rate-within-the-bound-requests-per-second) | 3.95 | 2.63 | none at or above 1.53, the lowest rate tried |
| [Start-up to the first answer, cold / warm (s; `p2`)](../bench/reports/v0.8.0/summary.md#gemma-start-up-seconds) | 24.1 / 21.6 | 205.1 / 68.2 | 140.9 / 36.5 |
| [Identical answers, two fresh servers, invariance on](../bench/reports/v0.8.0/summary.md#gemma-stability-answers-identical-between-two-passes-percent-95-wilson-interval) | 500/500 (`p2`) | 500/500 with DeepGEMM off (`p3`); 57.2% with its default DeepGEMM (`p2`) | —¹ |
| [Chat tokens/s at 64 in flight, invariance on / off (`p3`)](../bench/reports/v0.8.0/summary.md#gemma-cost-of-invariance-at-64-in-flight-family-i-both-modes-from-fresh-servers-on-one-host-other-mode--the-engines-non-default-mode) | 678 / 708 (4.2% less) | 602 / 661 (9.0% less)² | 175 / 329 (47% less)³ |

1. SGLang's deterministic mode gave 496–498 of 500 scene requests one and the same short answer (3 and 5 distinct answers per pass), so its answers stopped depending on the image. Its 99.2% identical is not compared.
2. vLLM's invariance holds only with DeepGEMM off, so its "on" figure also changes the FP8 kernel from DeepGEMM to CUTLASS.
3. In this mode SGLang's scene answers degenerate (26 tokens against 163). The chat row compares the same work.

The latency bound is an e2e p99 of 3,977 ms on scene. The cost row compares fresh servers on one host, each engine in both modes.

## Qwen3.8 27B FP8

| | llm-mini-infer | vLLM v0.33.0 | SGLang v0.5.23 |
|---|---|---|---|
| [TPOT p50, 1 in flight, scene (ms; `p3`)](../bench/reports/v0.8.0/summary.md#qwen-a1-1-in-flight-ms-tokens) | 5.91 | 9.90 | 9.69 |
| [TPOT p50, 1 in flight, chat (ms; `p3`)](../bench/reports/v0.8.0/summary.md#qwen-a1-1-in-flight-ms-tokens) | 7.45 | 9.57 | 9.47 |
| [Tokens/s, 64 in flight, chat (`p3`)](../bench/reports/v0.8.0/summary.md#qwen-a1-64-in-flight-ms-tokens) | 1,528 | 1,242 | 1,289⁴ |
| [Start-up to the first answer, cold / warm (s)](../bench/reports/v0.8.0/summary.md#qwen-start-up-seconds) | 27.7 / 24.9 (`p2`) | 212.4 / 75.2 (`p2`) | 155.3 / 61.8 (`p3`) |
| [Identical answers, two fresh servers, invariance on](../bench/reports/v0.8.0/summary.md#qwen-stability-answers-identical-between-two-passes-percent-95-wilson-interval) | 496/500 (`p2`) | refused⁵ | 320/500 (`p3`)⁶ |
| [Chat tokens/s at 64 in flight, invariance on / off](../bench/reports/v0.8.0/summary.md#qwen-cost-of-invariance-at-64-in-flight-a1-against-the-c-pass-1-load-of-the-other-mode-both-from-one-host-other-mode--the-engines-non-default-mode) | 1,528 / 1,515 (`p2`) | refused⁵ | 956 / 1,289 (`p3`)⁶ |

4. SGLang ran 48 sequences on Qwen3.8: at 64 its float32 SSM states do not fit. At 64 in flight, 16 requests queue in its server.
5. vLLM refuses batch invariance on Qwen3.8 ("not supported for GDN_ATTN").
6. SGLang's deterministic mode on Qwen3.8 also switches its attention from FlashInfer to Triton, so these figures include that change.

Every engine gave Qwen3.8's 500 stability requests only 13 to 20 distinct answers, so its stability rows rest on few distinct answers. The cost row compares a load of each mode on one host, after other loads on the same server. Neither llm-mini-infer nor vLLM has a rate within Qwen3.8's latency bound, and SGLang's rate search did not run on Qwen3.8.

## Disclosures

- llm-mini-infer is built for this GPU and these two families; the others are general-purpose engines.
- `p1` ran Gemma 4's latency and throughput rows. `p2` ran the rest of Gemma 4, and llm-mini-infer's and vLLM's Qwen3.8 runs. `p3` ran SGLang on Qwen3.8, Qwen3.8's latency and throughput rows for all three, vLLM's invariance without DeepGEMM and Gemma 4's cost of invariance.
- `p2` and `p3` have the same GPU, driver, CPU model and CPU quota. `p1` has the same CPU model with twice the quota. Its GPU gave 2 to 3% less bf16 and fp8 throughput in the GPU control.
- vLLM and SGLang ran from their wheels, not from their vendor images. Two more hosts checked the images against the wheels: vLLM's TPOT at 1 in flight differed by 1.3% and SGLang's by 5.0%, which fails the 5% bound set before the check. The comparison went on with the wheels.
- KV capacity, the tokens of one step (2,048 for llm-mini-infer, 8,192 for the others), the FP8 kernel and the attention backend are not matched. The record's README lists each arm's.
- SGLang on Qwen3.8 ran 48 sequences, not 64, and has no rate-search row.
- llm-mini-infer's speed rows ran with batch invariance on.
- Acceptance (tokens per verify step) compares only on chat: on scene, llm-mini-infer also delivers tokens that a grammar forces.
- Stability compares two passes on two fresh servers with the same seed and settings, not two hosts or two versions.
- Scene replies differ in length by engine (Gemma 4 at 1 in flight: 228, 192 and 157 tokens). Read scene rows beside their reply tokens.
- SGLang's deterministic mode gives Gemma 4's scene requests nearly one answer. On Qwen3.8 it also switches to Triton attention.
- vLLM's invariance depends on its FP8 kernel: two fresh servers matched for 57.2% of requests with DeepGEMM, its default on Gemma 4, and for all with `VLLM_USE_DEEP_GEMM=0`. Both arms' logs show invariance active.
- vLLM's cost of invariance (9.0%) includes the change from DeepGEMM to CUTLASS.
- One request failed in vLLM's Gemma 4 rate search, at 5.27 requests/s, past its saturation rate.
- Loads at 16 and 64 in flight run at the power limit, with the GPU clock up to 5.4% below the host's median. No load was run again.
- The cold start empties each engine's cache folders and advises the weights out of the page cache. llm-mini-infer's kernels come prebuilt in its image, and vLLM's and SGLang's cold starts include their compiles and graph captures.
- SGLang has no structured-output row: its first run stopped on a header missing from the test image.
- No multi-turn result is published.
- On `p1` and `p2`, llm-mini-infer's arm got the image's configuration file through a driver argument, as the image's entrypoint passes it. On `p3`, the arm's own command names the file. Every host served with the same file.

## The method

- **Families.** A1 measures latency and throughput at 1, 16 and 64 requests in flight. A2 finds the highest Poisson rate within the latency bound. C compares the answers of two fresh servers, invariance off and on. I measures the cost of invariance on one host. D times cold and warm starts. The record also has structured-output validity, vLLM's own draft counts and the vendor-image check.
- **The latency bound** is an e2e p99 of at most twice the median of the engines' e2e p50 at 1 in flight, set before any rate search. A probe passes with no failed or dropped request, a served rate of at least 0.95 of the offered rate, and its p99 under the bound.
- **Identical answers** have the same content, token count and finish reason.

The record's README gives [the full definitions](../bench/reports/v0.8.0/README.md#definitions).

## Versions and settings

- The record's README names the two measured commits of llm-mini-infer, from the project's history before its first public commit, and the digests of their GPU tests' images, which were not published. Their serving code (`llm_mini_infer/`, `deploy/`, `pyproject.toml`, `uv.lock`) is this tree's, apart from comments, docstrings, version numbers, pyproject's authors, URLs and ruff exclude, and one test marker's description.
- vLLM v0.33.0 and SGLang v0.5.23, each in a venv of its own. `freeze-diff.txt` compares each venv with its vendor image.
- Batch invariance: llm-mini-infer's `dev.runner.cuda.batch_invariant` (on by default on Gemma 4, and on Qwen3.8 up to a context of 32,768), vLLM's `VLLM_BATCH_INVARIANT=1` (with `VLLM_USE_DEEP_GEMM=0` to hold), and SGLang's `--enable-deterministic-inference`.
- `commands.sh` gives every server's command and flags.

## Running it again

The record's README tells [how to run it again](../bench/reports/v0.8.0/README.md#running-it-again), and `commands.sh` runs every family. To measure one load against a server of your own:

```bash
uv run python bench/load/run.py --model RedHatAI/gemma-4-31B-it-FP8-block --workload chat --concurrency 1,8,32 \
    --input-tokens 1024 --max-tokens 256 --ignore-eos
```

The benchmark's llm-mini-infer ran with the image's configuration file and `--set model.max_model_len=8184`. [bench/README.md](../bench/README.md) describes the tools.

## What not to conclude

- Nothing about other GPUs, MoE models or tensor parallelism.
- No ranking across hosts: where a row names two hosts, its cells come from different machines of the same type.
- Not that reproducible answers are llm-mini-infer's alone: vLLM gives 500 of 500 on Gemma 4 with DeepGEMM off.
- Not that invariance is free: on Gemma 4 chat it costs llm-mini-infer 4.2% of throughput on one host.
- Not a throughput win on Gemma 4 chat at 64 in flight: 661 against 657 tokens/s is a tie.
- Not that SGLang's deterministic mode reproduces Gemma 4's answers: it gives nearly every image one answer.
- Not that llm-mini-infer's Qwen3.8 answers are always identical: 496 of 500 were.
- Nothing about Qwen3.8 within its latency bound: no engine that was searched has a rate within it.
