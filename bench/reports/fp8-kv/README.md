# An FP8 KV cache for Gemma 4's full layers on one GPU

This note measures `cache.kv_cache_dtype fp8`, Gemma 4's full-attention layers' KV in FP8 E4M3 with the sliding layers' in BF16 ([architecture.md](../../../docs/architecture.md#sliding-window-models)), against the default BF16 KV: its capacity, its attention kernel's time and family L of `bench/engines/drive.py` at a context of 262,144. `bench/reports/check.py` checks this folder.

**What ran.** `RedHatAI/gemma-4-31B-it-FP8-block` with `google/gemma-4-31B-it-assistant`, k = 5, 64 sequences, batch invariance on (its default), the GPU tests' image of an earlier commit of this change (`sha256:a7ae5185978ed677d9dbcf4bba4f8236ce3d77b4097ab5d509c932ebda6ccc5f`), whose engine and kernels are this change's, its configuration (structured output on, every step split into a forward graph and a sampling graph); the arena load with this note's `drive.py`, whose only difference from the image's is that load's arguments. An NVIDIA RTX PRO 6000 Blackwell Server Edition at its 600 W power limit, driver 595.112.07, on an AMD EPYC 9355 host with 13.6 cores of CPU quota; the kernel timing on another host of the same GPU (AMD EPYC 9554), from the same kernel sources. Clients: `bench/load/run.py` on the same host over the loopback interface, streaming, temperature 0, seed 0. Every server was a new one. With batch invariance and greedy answers a load's outputs repeat exactly: repeated runs of the one-request decode and prefill cells agree within 0.2%, of the cached turns within 7%, of the arena load within 2%, and the tables give their means.

**Capacity** (each server's log; the arena load: `/metrics` every 2 s):

| | BF16 | FP8 |
|---|---|---|
| KV arena | 3,861 pages of 12.5 MiB | the same |
| a full page | 320 positions | 640 positions |
| requests of 4,096 tokens at once, without shared prefixes | 48 | 52 |
| requests of 262,144 tokens at once | 4 | 8 |
| the arena load: the most requests running at once | 25 | 32 (for under a minute of its 5.6 minutes: the 32 cold prefills run one after another) |
| the arena load: preemptions | 5 | 1 |
| the arena load: output tokens per second (2 runs) | 316.0, 316.3 | 389.6, 382.8 (+22%) |
| the arena load: tokens per verify step | 4.30 | 5.64 |

The arena load is one wave of 32 requests over distinct 32k prompts with replies of 4,096 tokens under `ignore_eos`. It replaced a first version (64 requests, 1,024-token replies) whose prefills kept the arena from filling on either arm, so it showed nothing. Each request also holds about 66 window pages whatever its length, so at short contexts FP8 adds few requests; at 262,144 it doubles them. The drafter accepts more of the FP8 arm's text on this load (the replies differ): its throughput gain is smaller than the rise in tokens per verify step, so these runs do not separate capacity from acceptance.

**Latency and load** (family L: three runs of each one-request cell; chat at 16 in flight one run; chat at 64 in flight over three prompt sets on three servers each, 60 s each, so one request more or less is about 0.6% of its tokens per second):

| load | measure | BF16 | FP8 | change |
|---|---|---|---|---|
| decode, 1 in flight, 1k prompt | output tokens/s | 61.3 | 59.0 | −3.8% |
| | ms per verify step | 29.46 | 29.49 | +0.1% |
| | tokens per verify step | 1.863 | 1.793 | −3.8% |
| decode, 1 in flight, 8k prompt | output tokens/s | 72.7 | 69.1 | −5.0% |
| | ms per verify step | 30.06 | 30.11 | +0.2% |
| | tokens per verify step | 3.262 | 3.041 | −6.8% |
| decode, 1 in flight, 32k prompt | output tokens/s | 28.6 | 27.9 | −2.4% |
| | ms per verify step | 31.81 | 31.92 | +0.3% |
| | tokens per verify step | 4.494 | 4.514 | +0.4% |
| cold prefill, 32k prompt | time to first token (p50) | 7.12 s | 7.34 s | +3.2% |
| cold prefill, 128k prompt | time to first token (p50) | 74.71 s | 78.12 s | +4.6% |
| a turn after a cached 32k prefix | time to first token (p50) | 132 ms | 135 ms | +1.8% |
| a turn after a cached 128k prefix | time to first token (p50) | 460 ms | 466 ms | +1.3% |
| chat, 16 in flight | output tokens/s | 507.7 | 503.5 | −0.8% |
| chat, 64 in flight | output tokens/s | 668.4 | 646.2 | −3.3% (−5.7%, +0.9%, −5.2%) |
| | ms per verify step | 131.8 | 131.4 | −0.3% (−1.1%, +2.7%, −2.5%) |
| | tokens per verify step | 1.872 | 1.828 | −2.3% (−4.2%, +1.6%, −4.4%) |
| | time to first token (p50) | 3.93 s | 3.52 s | −10.4% |

A verify step's time is a request's decode time over its verify steps: the engine's cost per step, apart from how many drafts the step accepts. Chat is bench/load/run.py's synthetic workload (1,024-token prompts, 256-token replies under `ignore_eos`); the one-request loads use its prompts at those lengths.

**The attention kernel** (`tests.scripts.numerics_kernels attn-timing --parts gemma-full --max-model-len 262144 --k 5`, at `--kv-dtype bf16` and `fp8`, interleaved, two runs each; eleven of the sixteen shapes it times, the others alike; microseconds per call of one layer; decode rows of 1 + k queries replayed from their decode bucket's CUDA graph, prefill rows eager; the fixed split is batch invariance's, the wave split its switch-off path):

| step | fixed split, BF16 | FP8 | change | wave split, BF16 | FP8 | change |
|---|---|---|---|---|---|---|
| 1 decode row over 8k keys | 80 | 86 | +6.4% | 52 | 53 | +2.0% |
| 1 decode row over 32k | 293 | 305 | +3.8% | 166 | 168 | +0.6% |
| 1 decode row over 128k | 1,126 | 1,190 | +5.8% | 603 | 630 | +4.5% |
| 1 decode row over 262k | 2,240 | 2,364 | +5.5% | 1,161 | 1,234 | +6.3% |
| 8 decode rows over 32k | 1,406 | 1,473 | +4.8% | 1,499 | 1,588 | +5.9% |
| 8 decode rows over 262k | 10,774 | 11,355 | +5.4% | 11,847 | 12,587 | +6.2% |
| 64 decode rows over 8k | 2,834 | 2,944 | +3.9% | 3,190 | 3,366 | +5.5% |
| 64 decode rows over 32k | 9,735 | 10,164 | +4.4% | 12,632 | 13,341 | +5.6% |
| a 256-token prefill chunk over 32k | 6,708 | 7,038 | +4.9% | 8,828 | 8,697 | −1.5% |
| a 256-token prefill chunk over 128k | 26,617 | 27,663 | +3.9% | 35,130 | 34,808 | −0.9% |
| a 2,048-token prefill chunk over 32k | 53,188 | 54,875 | +3.2% | 49,348 | 52,733 | +6.9% |

The kernel reads half the bytes but converts each tile of 16 keys to BF16 in shared memory before the BF16 path's arithmetic (so its outputs are those of a BF16 pool of the same values), and at these shapes the conversion costs more than the bytes save.

**Reading.** FP8 doubles the requests of 262,144 tokens that fit and, on a load where the arena binds, reaches all 32 requests at once (1 preemption) where BF16 runs at most 25 (5 preemptions). Per verify step the engine is as fast as with BF16 within 0.3% at 1 in flight and at 64 in flight (1.7% on a turn after a cached 128k prefix), but the conversion costs a long cold prefill 3.2% at 32k and 4.6% at 128k. The output tokens per second at 1k and 8k and in chat follow the drafts' acceptance on the text each arm generates, lower here for FP8 on those four cells; at 32k the loss is the slower prefill inside a 256-token reply. At 64 in flight the time to first token is lower with FP8 on all three prompt sets, a cause these runs do not isolate. On the same forced answers the two settings' acceptance can be compared directly (`tests.scripts.numerics_forced run --kv-cache-dtype fp8` beside a run without it, then `tests.scripts.numerics_fp32 pairs`). The setting stays off by default.

**Running it again.** On a host with this GPU, from the GPU tests' image of the commit (`tests/gpu-image.Dockerfile`), with the two models in its Hugging Face cache:

```bash
python3 bench/engines/drive.py --out results/engines --model gemma --family L --engines ours --context 262144 --arm ours-ctx262144 --arm ours-ctx262144-kvfp8
python3 -m tests.scripts.numerics_kernels attn-timing --parts gemma-full --max-model-len 262144 --k 5 --kv-dtype fp8
python3 -m tests.scripts.numerics_kernels attn-timing --parts gemma-full --max-model-len 262144 --k 5 --kv-dtype bf16
```
