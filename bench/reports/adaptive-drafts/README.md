# A draft count per step: Gemma 4 on one GPU

This note measures `dev.speculative.adaptive`, a draft count per step chosen by the scheduler from the load and the acceptance it measures ([architecture.md](../../../docs/architecture.md#speculative-decoding)), against Gemma 4's fixed k = 5, with family K of `bench/engines/drive.py`. `bench/reports/check.py` checks this folder.

**What ran.** The adaptive arm from llm-mini-infer's GPU tests' image `sha256:23c0fd04f3da90454b2c29fd93edc19f290e8eb4829925b98d4e4c965134482a`, built from the code this note was added with. k = 5 and the best fixed counts from the image of an earlier commit of the same change, whose code with the setting off is this one's: the plan-diff against main is identical, and k = 5 on this note's image gave the same tokens per second at the three loads at 64 in flight, within 0.2%. `RedHatAI/gemma-4-31B-it-FP8-block` with `google/gemma-4-31B-it-assistant`, at the revisions the [v0.8.0 manifests](../v0.8.0/manifests/) record; the adaptive arm's counts 1, 2, 3 and 5. Context 8,184, 64 sequences, batch invariance on (its default at this context), the image's configuration (structured output on, every step split into a forward graph and a sampling graph). Two hosts of one configuration: an NVIDIA RTX PRO 6000 Blackwell Server Edition at its 600 W power limit, driver 595.112.07, CUDA 13.2, on an AMD EPYC 9355 host with 13.6 cores of CPU quota; the GPU control (`bench/engines/gpu_control.py`) gave 406.5 and 411.1 bf16 TFLOPS at the two runs' starts and 403.8 and 407.0 at their ends. Clients: `bench/load/run.py` on the same host over the loopback interface, streaming, temperature 0, seed 0.

**Loads.** A1's ([v0.8.0](../v0.8.0/README.md#the-families)): at 1 in flight 150 scene and 80 chat requests, at 16 and 64 in flight 60 s each; and `mixed-c64`, 32 chat and 32 scene clients at once. Scene is the synthetic image workload with a strict JSON schema; chat has 1,024-token prompts and 256-token replies under `ignore_eos`. Each arm ran every load, each arm on a new server, then the loads at 64 in flight twice more on new servers. Each cell's best fixed k, chosen from an earlier run of family K on the same GPU model, ran once.

Output tokens per second (the mixed load: both clients'), mean and range over the runs:

| load | k = 5 | adaptive | change | best fixed k | adaptive's steps by count |
|---|---|---|---|---|---|
| scene, 1 in flight | 163.5 | 163.5 | 0.0% | 5 | 3: 1%, 5: 99% |
| scene, 16 | 1,343.8 | 1,357.3 | +1.0% | 5 | 3: 1%, 5: 98% |
| scene, 64 | 1,847.8 (1,845.5–1,849.0, 3 runs) | 1,813.8 (1,807.1–1,823.1, 3 runs) | −1.8% | 2: 1,835.7 (−0.7%) | 1: 1%, 2: 3%, 3: 77%, 5: 19% |
| chat, 1 | 66.7 | 67.1 | +0.6% | 5 | 1: 3%, 2: 1%, 3: 6%, 5: 90% |
| chat, 16 | 507.7 | 520.5 | +2.5% | 3: 541.9 (+6.7%) | 1: 11%, 2: 9%, 3: 45%, 5: 34% |
| chat, 64 | 661.3 (661.3–661.3, 3 runs) | 766.6 (763.7–768.0, 3 runs) | +15.9% | 1: 814.9 (+23.2%) | 1: 33%, 2: 15%, 3: 33%, 5: 20% |
| mixed, 64 | 1,170.8 (1,169.2–1,173.9, 3 runs) | 1,176.0 (1,165.4–1,187.7, 3 runs) | +0.4% | 3: 1,200.8 (+2.6%) | 1: 15%, 2: 2%, 3: 66%, 5: 17% |

Where the table names k = 5 as the best fixed k, the earlier run's best was 5, or 3 within 0.3% of it (chat at 1 in flight); on chat at 64 in flight its k = 2 tied k = 1. Steps by count are the first run's (`llm_mini_infer:spec_decode_steps_by_count`).

**Start-up and memory** (each server's log, on this note's host): k = 5 had 3,893 KV pages of 12.5 MiB, captured 112 CUDA graphs in 1.8 s and was ready in 20.3 s; the adaptive arm had 3,857 pages (36 fewer), captured 472 graphs (each count's decode graphs and the token buckets' sampling graphs) in 3.4–3.8 s and was ready in 24.1–25.6 s.

**Reading.** On chat under load the count per step gains: +15.9% at 64 in flight and +2.5% at 16, about two thirds and one third of the best fixed count's gain. At one request in flight it matches k = 5 (+0.6% on chat, 0.0% on scene, one run of each arm), and on the mixed load too (+0.4%; the two arms' ranges overlap). On scene at 64 in flight it loses 1.8%, beyond the spread of k = 5's runs (0.2%): it runs count 3 on 77% of the steps, where k = 5 delivers more. The policy prices each count on decode-only steps, but most of these steps carry prefill rows (at k = 5, 433 of 621 steps held more than 512 tokens, beyond the 384 of 64 decode rows), on which a wider verify costs little. The setting stays off by default.

**Running it again.** On a host with this GPU, from the GPU tests' image of the commit (`tests/gpu-image.Dockerfile`), with the two models in its Hugging Face cache:

```bash
python3 bench/engines/drive.py --out results/engines --model gemma --family K --engines ours --arm ours-adaptive --arm ours
python3 bench/engines/drive.py --out results/engines --model gemma --family K --engines ours --arm ours-k1 --load chat-c64
```
