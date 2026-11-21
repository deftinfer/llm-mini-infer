# llm-mini-infer

llm-mini-infer is an LLM inference server for one NVIDIA RTX Blackwell GPU (SM120). It serves Gemma 4 31B and Qwen3.8 27B through an OpenAI-compatible API: chat completions with streaming, images, reasoning, tool calls and JSON-schema structured output. It also serves `/tokenize`, `/detokenize` and vLLM-named metrics. It is built for low latency on these two models, and it keeps vLLM's request fields, metric names and settings, so a client made for vLLM works unchanged within the features that [the API](#the-api) lists ([vLLM compatibility](docs/vllm-compatibility.md)).

- [docs/operations.md](docs/operations.md): deploy and run it.
- [docs/architecture.md](docs/architecture.md): how the engine works.
- [docs/benchmarks.md](docs/benchmarks.md): the comparison with vLLM and SGLang.
- [docs/development.md](docs/development.md): work on it (tests, the image's build, releases).

## Performance

On one RTX PRO 6000 Blackwell Server Edition at 600 W, serving Gemma 4 31B FP8 with its assistant (5 drafts) at a context of 8,184, against vLLM v0.33.0 and SGLang v0.5.23 with the same drafter. The scene workload sends an image and asks for JSON under a strict schema, and the chat workload asks for 256-token replies to 1,024-token prompts. Each cell is one run:
- **Latency.** At 1 request in flight, the median time per output token after the first (TPOT p50) is 5.86 ms on scene, against vLLM's 7.07 and SGLang's 12.27: 17% under vLLM. On chat it is 13.97 ms against 19.21 and 23.13: 27% under vLLM ([table](bench/reports/v0.8.0/summary.md#gemma-a1-1-in-flight-ms-tokens)).
- **Saturated throughput.** At 64 requests in flight, scene gets 7.90 requests/s, against 6.58 and 3.82. Chat gets 661 tokens/s against vLLM's 657, a tie, and SGLang's 316 ([table](bench/reports/v0.8.0/summary.md#gemma-a1-64-in-flight-ms-tokens)).
- **Reproducible answers.** Batch invariance makes a request's output independent of the other requests in its step. With it on, two servers started from empty caches gave the same answer to 500 of 500 scene requests ([table](bench/reports/v0.8.0/summary.md#gemma-stability-answers-identical-between-two-passes-percent-95-wilson-interval)). Invariance is on by default on Gemma 4 at any context: each query's attention split comes from its own position. On host `p3` it costs 4.2% of chat throughput at a context of 8,184: 678 against 708 tokens/s ([table](bench/reports/v0.8.0/summary.md#gemma-cost-of-invariance-at-64-in-flight-family-i-both-modes-from-fresh-servers-on-one-host-other-mode--the-engines-non-default-mode)). vLLM also gave 500 of 500 with DeepGEMM, its default FP8 kernel on Gemma 4, turned off, on another host of the same type; with DeepGEMM it gave 57.2%. On Qwen3.8 27B, llm-mini-infer's two servers agreed on 496 of 500, over only 13 to 20 distinct answers, and vLLM refuses invariance.
- **Start-up.** With the caches of all three emptied, the first answer comes 24.1 s after the start, against 205.1 s and 140.9 s. llm-mini-infer's kernels come prebuilt in its image, and the others compile theirs at the start. A warm restart takes 21.6 s, against 68.2 s and 36.5 s ([table](bench/reports/v0.8.0/summary.md#gemma-start-up-seconds)).

| Gemma 4 31B FP8 | llm-mini-infer | vLLM v0.33.0 | SGLang v0.5.23 |
|---|---|---|---|
| TPOT p50 at 1 in flight, scene (ms; host `p1`) | 5.86 | 7.07 | 12.27 |
| Requests/s at 64 in flight, scene (`p1`) | 7.90 | 6.58 | 3.82 |
| Tokens/s at 64 in flight, chat (`p1`) | 661 (a tie with vLLM) | 657 | 316 |
| Start-up to the first answer, cold / warm (s; `p2`) | 24.1 / 21.6 | 205.1 / 68.2 | 140.9 / 36.5 |
| Identical answers, two fresh servers, invariance on | 500/500 (`p2`) | 500/500 with DeepGEMM off (`p3`); 57.2% with its default DeepGEMM (`p2`) | not compared |

The hosts `p1`, `p2` and `p3` are three machines with the same GPU model at 600 W. vLLM and SGLang ran from their wheels with their documented recipe flags, and their speed rows ran with invariance off, their default. KV capacity, the tokens of one step, the FP8 kernel and the attention backend are not matched. SGLang's deterministic mode gave nearly every image one answer, so its stability is not compared. The author of llm-mini-infer wrote the harness, and results move with versions. [docs/benchmarks.md](docs/benchmarks.md) gives Qwen3.8, every condition and what not to conclude.

To reproduce these numbers, run the image with `--set model.max_model_len=8184`: the benchmark did, with the image's configuration file.

## Quickstart

You need an SM120 GPU with enough memory, such as the RTX PRO 6000 Blackwell (96 GB), an NVIDIA driver for CUDA 13.0 and the NVIDIA Container Toolkit. This command serves Gemma 4 31B in FP8 with its assistant as the drafter. The first start downloads both (about 34 GB) into the cache volume.

```bash
export LLM_MINI_INFER_API_KEY=$(openssl rand -hex 16)
docker run -d --name llm-mini-infer --gpus all --init --shm-size 2g -p 8000:8000 \
    -v llm-mini-infer-cache:/cache -e LLM_MINI_INFER_API_KEY \
    ghcr.io/deftinfer/llm-mini-infer:0.8.0 serve \
    --set model.path=RedHatAI/gemma-4-31B-it-FP8-block \
    --set speculative.model=google/gemma-4-31B-it-assistant
docker logs -f llm-mini-infer            # wait for "ready in N s: http://0.0.0.0:8000"
curl -s localhost:8000/v1/chat/completions \
    -H "Authorization: Bearer $LLM_MINI_INFER_API_KEY" -H "Content-Type: application/json" \
    -d '{"model": "RedHatAI/gemma-4-31B-it-FP8-block",
         "messages": [{"role": "user", "content": "Name three prime numbers."}], "max_tokens": 64}'
```

For Qwen3.8 27B FP8, give `--set model.path=Qwen/Qwen3.8-27B-FP8` alone: it drafts with the checkpoint's MTP layer.

The image tag is a release ([tags](docs/development.md#building-the-image)). Add `-e HF_TOKEN` for a gated model. Each `--set` changes a [setting](docs/operations.md#settings). Stop the server with `docker stop -t 60 llm-mini-infer`, which lets it [drain](docs/operations.md#health-drain-and-restarts).

With the OpenAI client (`pip install openai`):

```python
import base64
import os

from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key=os.environ["LLM_MINI_INFER_API_KEY"])
model = "RedHatAI/gemma-4-31B-it-FP8-block"

# Text, streamed
stream = client.chat.completions.create(
    model=model, messages=[{"role": "user", "content": "Name three prime numbers."}], stream=True)
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)

# An image
with open("photo.jpg", "rb") as f:
    image = base64.b64encode(f.read()).decode()
reply = client.chat.completions.create(model=model, max_tokens=128, messages=[{"role": "user", "content": [
    {"type": "text", "text": "What is in this picture?"},
    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}}]}])
print(reply.choices[0].message.content)

# A JSON schema
schema = {"type": "object", "additionalProperties": False, "required": ["title", "year"],
          "properties": {"title": {"type": "string"}, "year": {"type": "integer"}}}
reply = client.chat.completions.create(
    model=model, temperature=0, messages=[{"role": "user", "content": "Describe a classic novel."}],
    response_format={"type": "json_schema", "json_schema": {"name": "book", "strict": True, "schema": schema}})
print(reply.choices[0].message.content)
```

## Models and GPUs

| Model | Checkpoints | Drafter |
|---|---|---|
| Gemma 4 31B (instruction-tuned) | [`google/gemma-4-31B-it`](https://huggingface.co/google/gemma-4-31B-it) (BF16), [`RedHatAI/gemma-4-31B-it-FP8-block`](https://huggingface.co/RedHatAI/gemma-4-31B-it-FP8-block) (block FP8) | the assistant, [`google/gemma-4-31B-it-assistant`](https://huggingface.co/google/gemma-4-31B-it-assistant): 5 drafts by default |
| Qwen3.8 27B | [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) (BF16), [`Qwen/Qwen3.8-27B-FP8`](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) (block FP8) | the checkpoint's MTP layer: 3 drafts |

The GPU is SM120. A GeForce RTX 50 card (at most 32 GB) is too small for these checkpoints. A server uses one GPU, and the container image runs only on SM120 ([the image's model limit](docs/operations.md#the-images-model-limit)). llm-mini-infer does not support the mixture-of-experts variants (Gemma 4 26B-A4B, Qwen3.5 MoE).

## What it does differently

- **One thread drives the GPU.** The scheduler, the cache, the grammar masks and the runner share one Python thread, which plans step N+1 while step N runs. Other processes do HTTP, tokenizing and image decoding.
- **A radix prefix cache for hybrid and sliding-window models.** The prefill pass writes state checkpoints at prompt ends and branch points. A client can ask for one at any message's end (`cache_control`).
- **Grammar-aware sampling.** When every sampled row has a mask and the masks allow at most 1,024 tokens, the LM head scores only those. With the image's `guidance` backend, tokens that a grammar forces at its start need no sampling steps ([grammar-aware heads](docs/architecture.md#grammar-aware-heads)).
- **Reproducible answers.** Sampling noise depends only on a request's seed and position. On Gemma 4, and on Qwen3.8 with an FP8 checkpoint up to a `model.max_model_len` of 32,768, the kernels are [batch-invariant](docs/architecture.md#batch-invariance), so a request's output can be independent of its batch ([measured](#performance)).
- **Exact images.** CPU processes decode and resize images with the reference processors' exact integer path. The GPU caches each image's embeddings.

## The API

| Endpoint | |
|---|---|
| `POST /v1/chat/completions` | chat completions, streamed (server-sent events) or not |
| `GET /v1/models` | the served model |
| `POST /tokenize`, `POST /detokenize` | a prompt or chat messages to token ids, and back, in vLLM's shapes |
| `GET /health` (also `/ready`), `GET /live` | readiness and liveness ([health](docs/operations.md#health-drain-and-restarts)) |
| `GET /metrics` | Prometheus metrics, with vLLM's names ([metrics](docs/operations.md#metrics)) |
| `GET /server_info` | the effective configuration, with the reason for each derived value |

**`/v1/chat/completions` accepts:**
- `messages` with the roles `system`, `developer`, `user`, `assistant` and `tool`. Content is a string, or `text` and `image_url` parts: a base64 data URI, or an `http(s)` URL when `images.allow_urls` is on (private addresses refused: [access](docs/operations.md#access)).
- `model` (another name than the served one is a 404), `max_tokens` or `max_completion_tokens`, `temperature`, `top_p`, `top_k`, `min_p`, `seed`, `stop` (up to 16), `stop_token_ids` (up to 16 distinct), `ignore_eos`, `stream`, `stream_options.include_usage`.
- `response_format`: `text`, `json_object`, or `json_schema` (`name`, `description`, `schema`, `strict`). llguidance enforces the schema, with or without `strict`.
- `tools` (function tools), `tool_choice` (`none`, `auto`, `required` or a named function) and `parallel_tool_calls`, with `tool_calls.parser` on ([tool calls](docs/architecture.md#tool-calls)).
- `reasoning_effort` (`none` to `max`) and `include_reasoning`, with `reasoning.parser` on: the thought comes back as `reasoning` ([reasoning](docs/architecture.md#reasoning)).
- `logprobs` and `top_logprobs` (0 to 20) with `server.logprobs: true` (off by default: a 400), with vLLM's checks, messages and values ([Sampling](docs/architecture.md#sampling)).
- `frequency_penalty`, `presence_penalty`, `repetition_penalty` and `logit_bias` with `server.penalties: true` (off by default: a 400), with vLLM's semantics and checks ([Sampling](docs/architecture.md#sampling)).
- The extension fields of vLLM's server: `chat_template_kwargs`, `add_generation_prompt`, `skip_special_tokens`, `include_stop_str_in_output`, `return_token_ids`, `cache_salt` (requests with different salts share no cached prompts or image embeddings), and `priority` (`scheduler.policy: priority` uses it).
- `cache_control: {"type": "ephemeral"}` on a message or its last part, in Anthropic's shape ([cache hints](docs/architecture.md#cache-hints)).

The server ignores `user`, and unknown keys inside a message, a tool call or its function (tool loops echo them back). It refuses with a 400 a top-level field that it does not know, and the features that it does not have: `n` other than 1, `thinking_token_budget`, `prompt_logprobs`, `min_tokens`, `echo`, `continue_final_message`, audio, video and file parts, and a chat template of the request's own. A value out of range is a 400, and so is a prompt that leaves no room for an output, as in vLLM ([request checks](docs/vllm-compatibility.md#request-checks)). `max_tokens` is a cap: the output stops where the context ends.

Responses carry `usage.prompt_tokens_details.cached_tokens` (on a stream, with `include_usage`) and the id `chatcmpl-<request id>`. `system_fingerprint` and the `x-llm-engine` header give the version, with `+<commit>` in a published image.

**Authentication.** With `LLM_MINI_INFER_API_KEY` (or `VLLM_API_KEY`) set, every path needs `Authorization: Bearer <key>` except OPTIONS and `/health`, `/ready`, `/live` and `/metrics`, which stay open for probes and scrapes. `/metrics` names the model and counts traffic: keep it off untrusted networks.

## Status

This is the first public release, before 1.0: settings and metrics can change between releases, and the [changelog](CHANGELOG.md) lists each change. It is tested on the RTX PRO 6000 Blackwell (SM120) only, with two model families and no mixture-of-experts models. [SECURITY.md](SECURITY.md) tells how to report a vulnerability, and [CONTRIBUTING.md](CONTRIBUTING.md) how to contribute.

## Licence

llm-mini-infer is licensed under the [Apache License, Version 2.0](LICENSE). It includes code adapted from vLLM and Hugging Face transformers and follows three of SGLang's designs (all Apache-2.0), and the image's FP8 GEMMs are compiled from NVIDIA CUTLASS (BSD-3-Clause). LICENSE holds the Apache text, then lists every origin, and ends with CUTLASS's licence. The image's `/usr/share/doc/llm-mini-infer/THIRD_PARTY_NOTICES` ([deploy/THIRD_PARTY_NOTICES](deploy/THIRD_PARTY_NOTICES)) tells where the licences of its third-party software are, and the sources of its GPL and LGPL packages.

## Acknowledgements

llm-mini-infer uses [PyTorch](https://pytorch.org), [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [CUTLASS](https://github.com/NVIDIA/cutlass), [llguidance](https://github.com/guidance-ai/llguidance), Hugging Face's [tokenizers](https://github.com/huggingface/tokenizers), [msgspec](https://github.com/msgspec/msgspec), [pyzmq](https://github.com/zeromq/pyzmq) and [uvicorn](https://github.com/Kludex/uvicorn). It learned from [vLLM](https://github.com/vllm-project/vllm), whose code it partly adapts, [SGLang](https://github.com/sgl-project/sglang), whose ReplaySSM verify (from [ReplaySSM](https://github.com/Johnny-Liou/ReplaySSM)) its GDN drafts follow, [mini-sglang](https://github.com/sgl-project/mini-sglang) and [LightLLM](https://github.com/ModelTC/lightllm); its tests' discipline owes much to [DeepInfer](https://github.com/deepseek-ai/DeepInfer). Hugging Face [transformers](https://github.com/huggingface/transformers) is its reference for the models.
