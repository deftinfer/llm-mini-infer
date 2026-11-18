# Operations

This document tells how to deploy, configure and monitor llm-mini-infer's image. [architecture.md](architecture.md) tells how the engine works, and the [README](../README.md#the-api) gives the API. [vllm-compatibility.md](vllm-compatibility.md) gives the vLLM names, rules and settings that the server follows, and where it differs.

A server is one container on one GPU: the launcher (the main process, which supervises the others), the engine, the HTTP frontends (one by default) and the image pool, which decodes images ([architecture.md](architecture.md#processes)).

## Deploying

### The image

The image is `ghcr.io/deftinfer/llm-mini-infer:0.8.0` ([tags](development.md#building-the-image)). It holds a virtual environment from `uv.lock`, llm-mini-infer's kernels prebuilt for SM120, the configuration file `/etc/llm-mini-infer/config.yaml` ([below](#the-images-configuration-file)), and the third-party notices. The `docker` commands below take the image's name from `$IMAGE`. Set it first: `export IMAGE=ghcr.io/deftinfer/llm-mini-infer:0.8.0`.

The image runs as the user `llm-mini-infer` (uid 10001), listens on port 8000, and keeps its caches and downloads in `/cache`. Its entrypoint runs `serve` (the default: `llm-mini-infer serve` with `$LLM_MINI_INFER_CONFIG` and the further arguments), `hostcheck` ([the host check](#the-host-check)), or any other command as given.

Before the server starts, `serve` checks the configuration, the model (a mounted `config.json`, or a repo id that it downloads into a writable cache) and `/dev/shm`. A failed check exits 2 with a message that tells what to fix.

**Requirements:** those of the [quickstart](../README.md#quickstart), and 2 GB of `/dev/shm` (Docker's default is 64 MB).

### The image's model limit

The image's FlashInfer modules cover only the attention of Gemma 4 31B and Qwen3.8 27B (`FLASHINFER_AOT` in `deploy/Dockerfile`). The image has no nvcc, so it cannot build other modules, or its kernels for another GPU: on a GPU other than SM120, it fails at startup. A missing FlashInfer module fails with `MissingJITCacheError` (the image sets `FLASHINFER_DISABLE_JIT=1`). Another model family needs a new image ([development.md](development.md#building-the-image)).

### docker run

The [quickstart](../README.md#quickstart) gives the command. `--init` reaps child processes and passes SIGTERM to the launcher. The cache volume must be writable by uid 10001. On a machine with several GPUs, run one container per GPU (`--gpus device=1`), each with its own name and port.

`model.path` takes a repo id, downloaded into the cache volume (`HF_HOME=/cache/huggingface`) with `HF_TOKEN` if set. `HF_HUB_OFFLINE=1` uses only the cache. To serve a local checkpoint, mount it at `/models/checkpoint`, the image's default `model.path`:

```bash
docker run -d --name llm-mini-infer --gpus all --init --shm-size 2g -p 8000:8000 \
    -v llm-mini-infer-cache:/cache -v /models/gemma-4-31B-it-FP8-block:/models/checkpoint:ro -e LLM_MINI_INFER_API_KEY \
    "$IMAGE" serve --set speculative.model=google/gemma-4-31B-it-assistant
curl -s localhost:8000/health               # {"status":"ok","ready":true}
docker stop -t 60 llm-mini-infer                  # the default of 10 s cuts the drain short
```

A mounted checkpoint's served name is its path, unless `model.served_model_name` is set.

### docker compose

`deploy/docker-compose.yaml` runs one server on one GPU from a checkpoint on the host, with a restart policy and a 60 s stop grace:

```bash
export LLM_MINI_INFER_IMAGE=$IMAGE                                   # without it, compose builds deploy/Dockerfile
export LLM_MINI_INFER_MODEL_DIR=/models/gemma-4-31B-it-FP8-block    # required: mounted at /models/checkpoint
export LLM_MINI_INFER_API_KEY=$(openssl rand -hex 16)               # compose passes it, VLLM_API_KEY and HF_TOKEN
docker compose -f deploy/docker-compose.yaml up -d
```

Put the model's settings (the served name, the drafter) in the file's `command:`, as `--set` arguments after `serve`. Compose's volume is `<project>_llm-mini-infer-cache`, not `docker run`'s `llm-mini-infer-cache`. For several GPUs, run one project per GPU (`-p`, `LLM_MINI_INFER_PORT`, `device_ids`).

### Kubernetes

The Helm chart in `deploy/helm/llm-mini-infer/` runs one replica per GPU, and never two on one node:

```bash
helm install llm-mini-infer deploy/helm/llm-mini-infer --namespace llm-mini-infer --create-namespace \
    --set model.id=RedHatAI/gemma-4-31B-it-FP8-block \
    --set config.speculative.model=google/gemma-4-31B-it-assistant
```

For Qwen3.8 27B FP8, `--set model.id=Qwen/Qwen3.8-27B-FP8` alone: it drafts with the checkpoint's MTP layer. `values.yaml` documents every value. The main ones:
- `model.id` (required: the chart refuses to render without it): a repo id, or a path on the cache volume. `model.hfToken` and `apiKey` name existing Secrets: the chart creates none.
- `config`: llm-mini-infer's configuration file. It replaces the image's file, so its defaults repeat that file's settings, and add `server.drain_grace_s: 20`.
- `cache`: an `emptyDir` by default, so each new pod downloads the model. `cache.existingClaim` keeps it.
- `resources`: 10 CPUs and 32 GiB. Set limits, because the derived defaults read the CPU quota.
- `terminationGracePeriodSeconds` (90): the chart refuses a value 20 s or less above `drain_grace_s + drain_timeout_s` ([drain](#health-drain-and-restarts)).

The chart's ClusterIP Service listens on port 80.

## Configuration

### How settings are given

- A YAML file of what differs from the defaults: `llm-mini-infer serve --config <file>`. The image's entrypoint gives it `$LLM_MINI_INFER_CONFIG`.
- `--set group.key=value`, repeatable, the value parsed as YAML. A few flags too (`--model`, `--max-model-len`, `--port`, ...: `llm-mini-infer serve --help`).
- Environment variables for secrets and paths ([below](#environment-variables)).

Every setting has a default: a constant, or a value derived from the model's files, the GPU's memory or the CPU quota. The startup log (`effective configuration:`) and `GET /server_info` give every value with its reason. Invalid YAML, an unknown key, a wrong type or a dev setting outside `dev.` stops the load, and the container exits 2.

### The image's configuration file

`deploy/config.yaml` sets only what the image fixes:

| setting | value |
|---|---|
| `model.path` | `/models/checkpoint` |
| `runner.kind` | `cuda` |
| `scheduler.seed` | 0: a request without a seed draws from the server's seeded generator, so from a start the same requests in the same order get the same seeds |
| `structured_output.backend`, `disable_any_whitespace` | `guidance`, `true`: canonical forcing and compact JSON, fewer tokens ([grammar-aware heads](architecture.md#grammar-aware-heads)) |
| `server.port` | 8000 |

A deployment sets the rest with `--set`, the chart's `config`, or a file mounted over this one.

### Settings

`llm_mini_infer/config.py` comments the settings, and `GET /server_info` gives each effective value with its reason. The table lists the settings that a deployment is likely to set.

| setting | default | what it does |
|---|---|---|
| `model.path` | (required) | a checkpoint directory, or a Hugging Face repo id |
| `model.served_model_name` | the path as given | the name that clients send as `model` |
| `model.max_model_len` | `config.json`'s position limit (8,192 if none) | the context. A derived value is lowered, with a warning, to what the KV pool holds. A set value past the position limit, or that does not fit, stops the startup, naming what fits |
| `model.chat_template_kwargs` | `{}` | defaults for the chat template's variables |
| `sampling.temperature`, `top_p`, `top_k`, `min_p` | `generation_config.json`'s, else 1.0, 1.0, 0 (off), 0.0 | defaults for the fields that a request leaves out |
| `sampling.max_tokens` | `generation_config.json`'s `max_new_tokens`, else the room left in the context | the same |
| `scheduler.max_num_batched_tokens` | 2048 | the tokens of one step: prefill chunks and decode rows |
| `scheduler.max_num_seqs` | 64 | the requests that run at the same time |
| `scheduler.max_num_queued_reqs` | off | caps the requests in the server, running and queued: past it chat completions get vLLM's 503. Set `max_num_seqs` plus the queue depth wanted |
| `scheduler.max_num_queued_tokens` | off | caps the prompt tokens of requests with no output yet: past it chat completions get vLLM's 503. Set the TTFT target times the prefill throughput |
| `scheduler.policy` | `fcfs` | or `priority` ([the scheduler](architecture.md#the-scheduler)) |
| `scheduler.seed` | random | the seed of the seeds that requests without one get |
| `cache.enable_prefix_caching` | `true` | the radix prefix cache |
| `speculative.model` | none | a separate drafter: Gemma 4's assistant |
| `speculative.num_speculative_tokens` | 5 with an assistant, 3 with an MTP layer, else 0 | the drafts per step: 0 to 7 (Qwen3.8: at most 3) |
| `images.enabled` | on with a vision tower | image inputs. On Gemma 4, `false` gives the KV pool about 2.3 GiB more |
| `images.min_pixels`, `max_pixels` | `preprocessor_config.json`'s | Qwen3.8's resize bounds |
| `images.max_soft_tokens` | the checkpoint's, else 280 | Gemma 4's tokens per image: 70, 140, 280, 560 or 1120 |
| `images.max_per_request`, `max_bytes`, `fetch_timeout_s` | 16, 20 MiB, 10 | the images in one request, the bytes of one image, the seconds for one URL |
| `images.max_source_pixels` | 178,956,970 | the largest declared size before decoding (Pillow's decompression-bomb bound); it can only be lowered |
| `images.max_request_decode_bytes` | 715,827,880 | the bytes the request's images decode to, 4 a stored pixel, as vLLM's budget; 0: no limit |
| `images.max_request_download_mb` | 256 | the MiB the request's `http(s)` downloads may total, as vLLM's budget; 0: no limit |
| `images.allow_urls` | `false` | fetch `http(s)` URLs; private, loopback, link-local, metadata and reserved addresses refused ([access](#access)) |
| `images.allowed_networks` | none | CIDRs fetched from despite that, such as an in-cluster image store |
| `images.allowed_domains` | none | only these hostnames, on the URL and on every redirect |
| `images.workers` | the CPU quota less the engine's and the frontends' cores, 1 to 8 | the image pool's processes |
| `structured_output.backend` | `auto` | `guidance`, or `auto` and `xgrammar` ([how they differ](architecture.md#structured-output)) |
| `structured_output.disable_any_whitespace` | `false` | compact JSON, with `guidance` only |
| `structured_output.max_schema_bytes` | 1 MiB | the largest schema, the tools' included; a call grammar takes at most 4 × (their bytes + 128 and the name's bytes a tool) + 64 KiB |
| `tool_calls.parser` | the model family's, when its chat template renders that format; else `none` | `gemma4`, `qwen3_coder`, or `none`: requests with tools are 400s ([tool calls](architecture.md#tool-calls)) |
| `reasoning.parser` | the model family's, when its chat template renders the markers; else `none` | `gemma4`, `qwen3`, or `none`: no thought is parsed ([reasoning](architecture.md#reasoning)) |
| `reasoning.effort_map` | the family's: Qwen3.8 `minimal` and `low` to `low`, `high`, `xhigh` and `max` to `xhigh`; Gemma 4 none | `reasoning_effort` values onto the template's, a JSON object; an effort it leaves out passes as it is |
| `runner.kind` | `cuda` | or `fake` (random tokens, for tests). `cuda` without a GPU fails |
| `runner.cuda.memory_utilization` | 0.9 | the part of the GPU's memory that llm-mini-infer takes |
| `runner.cuda.fp8_cold_prefill` | `reuse` | FP8 checkpoints: `reuse`, `bf16` or `fp8` ([the precise prefix](architecture.md#the-precise-prefix)) |
| `server.host`, `port` | `0.0.0.0`, 8000 | where the frontends listen |
| `server.num_frontends` | 1 | frontend processes on one port |
| `server.drain_grace_s` | 0 | after SIGTERM, the time that new requests are still served |
| `server.drain_timeout_s` | 30 | then, the time that requests in flight have to complete |
| `server.request_id_header` | `true` | echo `X-Request-Id` |
| `server.logprobs` | `false` | answer `logprobs` and `top_logprobs` (off: a 400); the CUDA runner then keeps a buffer of its head's logits for them |
| `server.keep_alive_s` | 5 | the idle time of an HTTP connection |
| `server.stop_with_parent` | `false` | stop as on SIGTERM when the launcher's parent process exits |
| `engine.watchdog_s` | 60 | stop an engine whose step loop is stuck this long; 0 disables it |
| `engine.plan_at_boundary` | `false` | [plan at the forward/sampling boundary](architecture.md#planning-at-the-forwardsampling-boundary): Gemma 4 with structured output only |
| `observability.log_level`, `log_requests` | `info`, `true` | the log, and a line per chat request |
| `observability.debug_endpoints` | `false` | `/debug/*` timings |

### The dev settings

`dev.<group>.<key>` holds the settings of tests and benchmarks, which a deployment does not set. These switches turn a feature off, as a fallback: `dev.runner.cuda.graphs` (eager steps), `fused_ops` (torch ops), `pdl`, `batch_invariant` ([batch invariance](architecture.md#batch-invariance)), `always_split`, `sparse_head`, `dev.speculative.draft_universe`, `dev.structured_output.forced_prefix`, `eos_free_finish`, `dev.cache.decode_checkpoint_interval` (0) and `dev.cache.demote_superseded` (false). `llm_mini_infer/config.py` comments each, and [architecture.md](architecture.md) describes most. `always_split` and `pdl` keep the outputs the same; the other switches can change the GPU's rounding.

### Environment variables

| variable | what it does |
|---|---|
| `LLM_MINI_INFER_API_KEY`, `VLLM_API_KEY` | the API key (either one). Never in a configuration file |
| `HF_TOKEN`, `HF_HOME`, `HF_HUB_OFFLINE` | Hugging Face's token, cache (the image: `/cache/huggingface`) and offline mode |
| `LLM_MINI_INFER_CONFIG` | the file that the image's entrypoint and `llm-mini-infer hostcheck` read (`/etc/llm-mini-infer/config.yaml`) |
| `LLM_MINI_INFER_KERNEL_PREBUILT`, `LLM_MINI_INFER_KERNEL_CACHE` | the prebuilt kernels (the image: `/opt/llm-mini-infer/kernels`), and where kernels are built outside the image |
| `FLASHINFER_WORKSPACE_BASE`, `CUTE_DSL_CACHE_DIR`, `CUDA_CACHE_PATH` | compile caches (the image: in `/cache`) |

## Access

**The API key.** With `LLM_MINI_INFER_API_KEY` or `VLLM_API_KEY` set, every request needs `Authorization: Bearer <key>` except OPTIONS and `/health`, `/ready`, `/live` and `/metrics`, for probes and scrapes. Without a key every path is open, `/tokenize`, `/server_info` and `/debug/*` included, and the startup line warns when `server.host` is not a loopback address. `/metrics` names the model and counts traffic: keep it on a trusted network.

**Image URLs.** With `images.allow_urls` on, the server fetches `http(s)` image URLs and follows at most 5 redirects. It refuses loopback, private, link-local, cloud-metadata and reserved addresses, after DNS and on every redirect, unless `images.allowed_networks` holds them; with `images.allowed_domains` set, it fetches only those hostnames. A refusal is a 400 that names no address. The block is about addresses, not about what an address serves. It cannot catch a service on a public address (the node's own, a public load balancer or ingress in front of internal services, a name that also resolves publicly), a network that `images.allowed_networks` exempts, or a route through a proxy or mesh that rewrites destinations: the server uses no proxy and assumes its connections go straight out. A fetch's status, timing and errors still tell a client whether a public URL answers. Turn URLs on only for clients that may make the server fetch from any public host.

## Health, drain and restarts

`GET /health` (also `/ready`) answers 200 when the server is ready, and 503 during a drain or after the engine stopped. Nothing listens before the server is ready. `GET /live` answers 200 while the process serves HTTP: use it for liveness, because `/health` fails during a drain.

**SIGTERM drains:**
1. `/health` answers 503, so balancers stop sending requests.
2. For `server.drain_grace_s`, new requests are still served, with `Connection: close`.
3. Then new chat completions get a 503. Requests in flight, their upload included, have `server.drain_timeout_s` to complete, then fail.
4. The processes stop: exit 0, or 1 if one failed meanwhile.

The supervisor's grace period (`docker stop -t`, `stop_grace_period`, `terminationGracePeriodSeconds`) must exceed `drain_grace_s + drain_timeout_s` by about 20 s. A second signal fails the requests at once (exit 1). SIGINT drains without the grace.

**Other stops.** A signal before the server is ready ends the start: the engine's stacks go to the log, and the container exits 143 (130 for SIGINT). A signal during a model download ends it at once, with the same codes. With `server.stop_with_parent` on, the exit of the launcher's parent process (a wrapper such as `uv run`, if one started it) stops the server as SIGTERM would, within about two seconds; a parent of PID 1, or none (the launcher is PID 1), is not watched. Turn it on under a supervisor that is not PID 1 and may die without stopping the server, such as a test runner or a script. In a container it does nothing.

**Failures.** An engine or frontend that exits, or a step loop stuck for `engine.watchdog_s`, fails the requests in flight (503, or an error event on a stream), and the container exits 1. The launcher replaces an image pool worker that crashes or holds an image past its deadline (45 s at the default): that image fails with a 400, and nothing else stops. More than 3 worker deaths in 60 s stop the container. A non-finite result or a grammar error fails only its request, with a 500.

## Metrics

`GET /metrics` serves vLLM's metric names and buckets ([vLLM compatibility](vllm-compatibility.md#metrics)). Each metric's `# HELP` tells what it counts.

**vLLM's metrics** (the server exports only these):
- requests: `vllm:num_requests_running`, `_waiting` (it includes requests whose grammar is being built), `vllm:request_success_total{finished_reason}`, `vllm:num_preemptions_total`, `vllm:admission_rejections_total{reason}` (the queue caps' 503s);
- the cache: `vllm:kv_cache_usage_perc`, `vllm:prefix_cache_queries_total`, `_hits_total`, `vllm:cache_config_info`;
- tokens and latency: `vllm:prompt_tokens_total`, `generation_tokens_total`, the request histograms, `time_to_first_token_seconds`, `inter_token_latency_seconds`, `e2e_request_latency_seconds` and the queue, prefill and decode times ([clocks](vllm-compatibility.md#metrics));
- speculative decoding: `vllm:spec_decode_num_drafts_total`, `_num_draft_tokens_total`, `_num_accepted_tokens_total`, `_num_accepted_tokens_per_pos_total{position}`;
- `http_requests_total{method,status,handler}` ([which requests it counts](vllm-compatibility.md#metrics)).

**llm-mini-infer's own,** to watch:

| metric | what to watch |
|---|---|
| `llm_mini_infer:ttft_overhead_seconds` | the host's time to a stream's first byte, without device time. A slow host shows here first |
| `llm_mini_infer:gpu_idle_seconds_total`, `gpu_busy_seconds_total` | the GPU idle time caused by the CPU: idle over busy plus idle must stay at a few percent |
| `llm_mini_infer:requests_failed_total`, `grammar_errors_total`, `grammar_rejects_total` | failed requests (never expected), and schemas that did not compile, or not within the engine's compile time limit |
| `llm_mini_infer:chat_requests_inflight` | chat requests from the drain gate to the end of their response, those still being prepared included (the queue caps count them only once sent to the engine) |
| `llm_mini_infer:http_responses_total{handler,method,status}` | exact codes, probe paths aside: 499 if the client left before the status line |
| `llm_mini_infer:client_disconnects_total`, `engine_aborts_total{reason}` | clients that left, and requests that the engine stopped |
| `llm_mini_infer:page_waits_total`, `prefix_cache_lost_tokens_total{cause}` | waits for KV pages, and cached tokens computed again |
| `llm_mini_infer:image_stage_seconds{stage}`, `image_pool_restarts_total`, `_hangs_total` | the image path, and the pool's workers |
| `llm_mini_infer:attention_plans_kept_total` | must stay 0: a decode graph kept a plan, and its rows may attend wrongly |
| `llm_mini_infer:vision_plan_waits_total` | vision plans that waited for a free buffer: a lost overlap with the previous step, never a wrong plan. It rises under batch invariance when the steps in flight encode more than four images |
| `llm_mini_infer:gc_seconds_total{process,generation}`, `engine_phase_seconds_total{phase}` | time in Python's collector against wall time; a pause of 10 ms or more is also logged. Where the engine loop's time goes |

With several frontends, each one keeps its own request metrics and the per-step histograms of the steps that sent it outputs (`vllm:iteration_tokens_total`, `llm_mini_infer:engine_step_*`), so a scrape sees one frontend's share. The engine's other metrics are the same in all.

## Caches

### The prefix cache

The prefix cache is on by default ([architecture.md](architecture.md#the-cache)). A `cache_salt` separates the encoder cache too: one image under two salts is encoded twice.

The server sizes the pools at startup from the free memory, and logs them. The log warns when the KV pool holds fewer than `max_num_seqs` requests of `model.max_model_len` tokens (Gemma 4's KV arena: of 4,096 tokens). `llm_mini_infer:page_waits_total` and `vllm:num_preemptions_total` show the effect.

On traffic where conversations come back (agents, chats), these counters show whether the checkpoints suffice:
- `llm_mini_infer:prefix_cache_lost_tokens_total{cause="checkpoint"}` against `vllm:prefix_cache_hits_total` is the KV the cache held but could not resume from: no checkpoint was taken there, it was evicted, or it lies inside an image span.
- `llm_mini_infer:checkpoint_evictions_total{source,served}` separates checkpoints evicted after serving a resume (`served="yes"`, expected) from those evicted without serving one (`served="no"`). A `finish` checkpoint in the second case is a conversation that lost its resume point, or a client that does not send the reply back as generated (a chat template that drops the reasoning, re-serialized tool calls), whose next turn resumes at the previous prompt's end-of-prefill checkpoint.
- When the second rises, the conversations outnumber the checkpoint budget, by default `min(max_num_seqs, the slots that fit in 25% of the memory left)` ([architecture.md](architecture.md#pools)); a benchmark sets it with `dev.cache.checkpoint_slots`. `bench/load/run.py --turns N` gives the hit rate by turn index.

### Compile caches

The cache volume holds the downloads and the compile caches that the warm-up writes. To start clean, remove the volume: it holds only caches and downloads.

## Logging

Logs go to stderr as `LEVEL MM-DD HH:MM:SS.mmm [process] message`. The startup lines give the effective configuration, the pools, the ATen CPU capability, the compile and warm-up times and the features in effect. With `observability.log_requests`, each chat completion gets one line of metadata: `request id=<id> status=<code> finish=<reason> stream=<0|1> prompt=<n> cached=<n> images=<n> out=<n> ttft_ms=<ms> e2e_ms=<ms> receive_ms=<ms> metric_ttft_ms=<ms> metric_e2e_ms=<ms> verify_steps=<n> verify_tokens=<n>`. `ttft_ms` and `e2e_ms` count from the request's receipt, `receive_ms` is its body's upload, the `metric_` times are those the TTFT and e2e histograms observe (from the request parsed), and under speculative decoding `verify_steps` and `verify_tokens` are its verify steps and the tokens they delivered (`-` when a stop string ended it). The status is 499 when the client left before the response started. A stream whose client left later keeps its 200, with `finish=abort`.

The [precise prefix](architecture.md#the-precise-prefix) logs an INFO line, `request <id> refines a shared prefix`, with its PRECISE rows and pieces, when a request starts a refine. A WARNING, `evicting a refined checkpoint that served <n> resumes`, shows when a refined copy that served resumes is evicted. It shows when memory pressure evicts such a copy.

Request lines hold no contents. The engine loop, the frontends and the image workers log an exception's type and frames, but not its message, which can quote a request. Startup errors keep their messages, which name what to fix. Secrets are never logged.

**Request ids.** A client's `X-Request-Id` (1 to 256 printable ASCII characters, no spaces, quotes or backslashes) names the request in its `id`, its response header and the log. Without a valid one, the server makes a 24-digit hexadecimal id.

## The host check

`llm-mini-infer hostcheck` checks, in the container, that a machine runs llm-mini-infer as fast as it should. It needs the GPU for itself:

```bash
docker run --rm --gpus all --init --shm-size 2g -v llm-mini-infer-cache:/cache -e HF_TOKEN \
    "$IMAGE" hostcheck \
    --set model.path=RedHatAI/gemma-4-31B-it-FP8-block --set speculative.model=google/gemma-4-31B-it-assistant
```

It measures the GPU and the host (`--no-server` stops there), the server's start twice (a warm start over 60 s is `SLOW`), and load: Gemma 4's image requests with a JSON schema at 2 and 4 requests/s, at most 32 and 64 in flight, extras dropped (`--rates`, `--slots`). The synthetic workload is a scene task: a landscape or portrait test-pattern image, a generated prompt and strict schema (`--prompt-tokens`, `--tall-share`). Verdicts need a reference (`llm_mini_infer/hostcheck_references.json`) naming the run's GPU, power limit, model, drafter and workload; else the report has raw numbers.

For your own workload, give `--inputs DIR`, a mounted directory (layout in `--help`). `frames/` needs about 1,000 JPEG images at the defaults, and `reference.json` is the `as_reference` of a healthy host's report. `deploy/helm/examples/` has a Job for Kubernetes.

The report and the server logs go to `/cache/hostcheck/`. The exit status is 0 when no verdict failed (read the `SLOW` and `LOW` rows), 1 when one failed or the check itself failed (the report then holds `error` and what it measured before), and 2 when the check could not run. `llm-mini-infer hostcheck --help` lists every option.

## Troubleshooting

| symptom | cause |
|---|---|
| `[entrypoint] model.path ... has no config.json` | no checkpoint at `model.path`: mount one there, or give a Hugging Face repo id |
| `[entrypoint] configuration ...: invalid configuration: ...` | a wrong key or value in the file or a `--set`; the message names it |
| `... Gemma 4 sizes images by images.max_soft_tokens ...` | pixel bounds on Gemma 4: set `images.max_soft_tokens` |
| clients get 404 ``The model `X` does not exist. Valid aliases: ...`` | the client's name is not the served name (`model.served_model_name`) |
| clients get 401 | no `Authorization: Bearer` key, or the wrong one, on a path other than the probes' and `/metrics` |
| `model <repo id> (revision ...): RepositoryNotFoundError ...` | the repo id, the revision or `HF_TOKEN`; offline, a missing snapshot |
| `the Hugging Face cache ... is not writable` | the cache volume is missing, read-only, or another user's |
| `/dev/shm has N MiB free, the pixel arena needs M MiB` | more shared memory (`--shm-size`, `shm_size`, `shm.sizeLimit`) |
| `Free memory on device cuda:0 ... is less than ...` | another process holds GPU memory, or lower `runner.cuda.memory_utilization` |
| `runner.kind cuda (the default) needs a CUDA GPU ...` | no GPU in the container |
| `... needs llm-mini-infer's CUDA kernel library, which is unavailable: failed to build: nvcc not found ...` | in the image: a GPU other than SM120, which the message names. Outside it: install nvcc |
| `the FP8 GEMM library is unavailable: no SM 12.x GPU ...` | an FP8 checkpoint on a GPU other than SM 12.x |
| `SIGTERM while starting (waiting for ...)` | a stop before the server was ready, often a startup probe |
| `the engine loop made no progress for N s (engine.watchdog_s)` | a hung GPU call; the stacks are in the log |
| `frontend N is M outputs behind: pausing its admissions` | a starved frontend core |
| `the KV arena holds about N requests ...`, `the KV pool holds N requests ...` | too little GPU memory for `max_num_seqs` or `max_model_len` |
| `llm-mini-infer-pool-N exited unexpectedly ...`, `... past the job deadline ...` | an image pool worker died or hung on an image, which fails with a 400 |
| `... image pool worker deaths in 60 s, a crash loop ...` | workers die again and again: read the stacks and the node's memory |
| `engine.plan_at_boundary is set, but ...` | too little margin on this host: unset it |
| `cannot start N grammar compile threads ...` | an address-space limit (`ulimit -v`, `vm.overcommit_memory=2`) refuses their stacks. A frontend and the engine reserve 64 MiB each (two threads of 32 MiB) |
