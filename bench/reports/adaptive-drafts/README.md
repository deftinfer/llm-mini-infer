# A draft count per step: Gemma 4 on one GPU

This note measures `dev.speculative.adaptive`, a draft count per step chosen by the scheduler from the load and the acceptance it measures ([architecture.md](../../../docs/architecture.md#speculative-decoding)), against Gemma 4's fixed k = 5, with family K of `bench/engines/drive.py`. `bench/reports/check.py` checks this folder.

**What ran.** llm-mini-infer from its GPU tests' image `sha256:59cccfb06c55e8337eb2ba9904f14cbfe1e4fb9abd263cde8bdd0dcd586427f0`, built from the code this note was added with. `RedHatAI/gemma-4-31B-it-FP8-block` with `google/gemma-4-31B-it-assistant`, at the revisions the [v0.8.0 manifests](../v0.8.0/manifests/) record; the adaptive arm's counts 1, 2, 3 and 5. Context 8,184, 64 sequences, batch invariance on (its default at this context), the image's configuration. One NVIDIA RTX PRO 6000 Blackwell Server Edition at its 600 W power limit, driver 595.112.07, CUDA 13.2, on an AMD EPYC 9355 host with 13.6 cores of CPU quota; the GPU control (`bench/engines/gpu_control.py`) gave 406.5 bf16 TFLOPS at the start and 403.8 at the end. Clients: `bench/load/run.py` on the same host over the loopback interface, streaming, temperature 0, seed 0.

**Loads.** A1's ([v0.8.0](../v0.8.0/README.md#the-families)): at 1 in flight 150 scene and 80 chat requests, at 16 and 64 in flight 60 s each; and `mixed-c64`, 32 chat and 32 scene clients at once. Scene is the synthetic image workload with a strict JSON schema; chat has 1,024-token prompts and 256-token replies under `ignore_eos`. k = 5 and the adaptive arm ran every load, each arm on a new server, then the loads at 64 in flight twice more on new servers. Each cell's best fixed k, chosen from an earlier run of family K on the same GPU model, ran once.

Output tokens per second (the mixed load: both clients'), mean and range over the runs:

| load | k = 5 | adaptive | change | best fixed k | adaptive's steps by count |
|---|---|---|---|---|---|
| scene, 1 in flight | 163.5 | 163.4 | −0.1% | 5 | 3: 1%, 5: 99% |
| scene, 16 | 1,343.8 | 1,362.3 | +1.4% | 5 | 3: 1%, 5: 98% |
| scene, 64 | 1,847.8 (1,845.5–1,849.0, 3 runs) | 1,800.8 (1,783.4–1,826.9, 3 runs) | −2.5% | 2: 1,835.7 (−0.7%) | 1: 1%, 2: 1%, 3: 30%, 5: 67% |
| chat, 1 | 66.7 | 62.5 | −6.3% | 5 | 1: 78%, 2: 11%, 3: 3%, 5: 8% |
| chat, 16 | 507.7 | 524.8 | +3.4% | 3: 541.9 (+6.7%) | 1: 38%, 2: 39%, 3: 12%, 5: 12% |
| chat, 64 | 661.3 (661.3–661.3, 3 runs) | 733.8 (725.3–738.1, 3 runs) | +11.0% | 1: 814.9 (+23.2%) | 1: 67%, 2: 6%, 3: 12%, 5: 14% |
| mixed, 64 | 1,170.8 (1,169.2–1,173.9, 3 runs) | 1,009.3 (1,009.0–1,009.5, 3 runs) | −13.8% | 3: 1,200.8 (+2.6%) | 1: 82%, 2: 1%, 3: 14%, 5: 3% |

Where the table names k = 5 as the best fixed k, the earlier run's best was 5, or 3 within 0.3% of it (chat at 1 in flight); on chat at 64 in flight its k = 2 tied k = 1. Steps by count are the first run's (`llm_mini_infer:spec_decode_steps_by_count`).

**Start-up and memory** (each server's log, the three runs): k = 5 had 3,893 KV pages of 12.5 MiB, captured 112 CUDA graphs in 1.9–2.3 s and was ready in 21.3–23.8 s; the adaptive arm had 3,876 pages (17 fewer), captured 301 graphs in 2.7–2.8 s and was ready in 22.8–23.7 s.

**Reading.** On chat under load the count per step gains: +11.0% at 64 in flight and +3.4% at 16, about half of the best fixed count's gain. It loses on chat at 1 in flight (−6.3%, one run of each arm), on scene at 64 (−2.5%: the three runs' ranges do not overlap) and on the mixed load (−13.8%, against a spread of 0.4% between k = 5's runs). The engine's counters and a simulation of the policy point to two causes:
- At one decode row a step's share of accepted drafts is 0 or 1, and acceptance comes in runs. A run of rejections at a high count drives the estimate for the drafts past the first low; the policy then verifies one draft a step, so those positions are measured again only at its probes, and their estimate stays low.
- On the mixed load about half the steps carry prefill rows (at a fixed k = 1, 502 of 1,054 steps held more than the 128 tokens of 64 decode rows). Such a step drafts k, since the runners' token-bucket graphs draft k, and the next step verifies the policy's count, mostly 1: the sampling graphs, which run the drafter, took 11.9 s of device time over the load, against 5.5 s at a fixed k = 1. The policy prices each count on decode-only steps only.

The setting stays off by default.

**Running it again.** On a host with this GPU, from the GPU tests' image of the commit (`tests/gpu-image.Dockerfile`), with the two models in its Hugging Face cache:

```bash
python3 bench/engines/drive.py --out results/engines --model gemma --family K --engines ours --arm ours-adaptive --arm ours
python3 bench/engines/drive.py --out results/engines --model gemma --family K --engines ours --arm ours-k1 --load chat-c64
```
