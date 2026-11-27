# vLLM compatibility

The reference is vLLM v0.33.0. A client, router, dashboard or deployment made for it works with llm-mini-infer unchanged:
- the chat API takes the same request fields and answers with the same response fields, error texts and status codes;
- `/metrics` serves the same metric names, help texts and buckets;
- tool calls and reasoning are parsed as the reference's `gemma4`, `qwen3_coder` and `qwen3` parsers parse them;
- each deploy setting has a counterpart or a fixed rule ([Deploy settings](#deploy-settings)).

This page lists the names and rules that follow the reference, and where the server differs from it ([Deviations from the reference](#deviations-from-the-reference)). `tests/api/test_openai_compat.py` and the API tests pin them.

## Metrics

- **Series.** The `vllm:*` series ([Metrics](operations.md#metrics)) keep vLLM's names, help texts and buckets, with the labels `model_name` and `engine` (`vllm:admission_rejections_total` and `vllm:tool_call_parser_invocations_total`: `model_name` only); counters end in `_total`.
- **Gaps.** vLLM's `vllm:num_requests_waiting_by_reason`, `prompt_tokens_by_source`, `prompt_tokens_cached`, `request_params_n`, `request_max_num_generation_tokens`, `request_num_preemptions`, `request_prefill_kv_computed_tokens`, `engine_sleep_state`, `mm_cache_queries`, `mm_cache_hits`, `external_prefix_cache_queries`, `external_prefix_cache_hits`, and v0.33.0's `prompt_tokens_cached_by_source` are not exported: a panel on them is empty. The dashboard of vLLM's Kubernetes deployment stack queries `vllm:gpu_cache_usage_perc`, `vllm:gpu_prefix_cache_hits_total` and `_queries_total`, which neither vLLM nor this server exports: read `vllm:kv_cache_usage_perc`, `vllm:prefix_cache_hits_total` and `_queries_total`.
- **HTTP counter.** `http_requests_total{method,status,handler}` is prometheus-fastapi-instrumentator's counter, as vLLM's server mounts it. It leaves out the paths vLLM leaves out (`/metrics`, `/health`, `/load`, `/ping`, `/version`, `/server_info`) and `/ready` and `/live`, and never sees a 401. `llm_mini_infer:http_responses_total` counts every response.
- **Clocks.** `vllm:time_to_first_token_seconds` and `vllm:e2e_request_latency_seconds` start once the request is parsed and checked, before its grammar, its images and its render: vLLM's arrival time, and the request line's `metric_ttft_ms` and `metric_e2e_ms`. The queue, prefill, decode and inference times and the inter-token latency run on the engine's clock. A one-token request's time per output token is 0. An aborted request appears in no finished-request series.

## Requests

- **API key.** `VLLM_API_KEY` works beside `LLM_MINI_INFER_API_KEY`, and either key is accepted. vLLM guards only `/v1`, `/v2`, `/inference` and `/cohere`; this server guards every path except `/health`, `/ready`, `/live` and `/metrics`, so `/tokenize`, `/detokenize`, `/server_info` and `/debug/*` need the key, and so does an unknown path (a 401, not a 404). OPTIONS reaches no handler (a 405 or a 404); vLLM answers CORS preflights. The 401 body is OpenAI-shaped, with `WWW-Authenticate: Bearer`; vLLM's is `{"error":"Unauthorized"}`. There is no `--api-key`: the key comes from the environment.
- **Cache salt.** `cache_salt` takes the reference's checks and texts. It also keys the encoder cache, which the reference (and SGLang) shares across salts.
- **Routes.** `/tokenize` and `/detokenize` take and answer vLLM's shapes, and `/server_info` keeps its name.
- **Request fields.** vLLM's extension fields that the server honours: `top_k`, `min_p`, `repetition_penalty` (with `server.penalties`), `stop_token_ids`, `ignore_eos`, `chat_template_kwargs`, `add_generation_prompt`, `skip_special_tokens`, `include_stop_str_in_output`, `return_token_ids`, `cache_salt`, and `priority` (lower first). `uuid` in an image part is ignored. `structured_outputs`, `guided_*` and `mm_processor_kwargs` are refused as unknown fields. A chat template from the request, `chat_template` or one not null in `chat_template_kwargs`, is always refused with the 400 vLLM gives without `--trust-request-chat-template` (this server has no such setting).
- **`stop_reason`** is what ended a reply: a `stop` string, one of the request's `stop_token_ids`, or a stop id from the model's generation config. Gemma 4's are `<turn|>` (106, a plain answer's end) and `<|tool_response>` (50, a call turn's), and Qwen3.8's is `<|endoftext|>` (248044). The tokenizer's own EOS gives null (Gemma 4's `<eos>`, Qwen3.8's `<|im_end|>`), and so does a grammar's end. With `ignore_eos`, an EOS listed in `stop_token_ids` is reported.
- **Ids.** `X-Request-Id` is always echoed (`server.request_id_header`); vLLM echoes it only with `--enable-request-id-headers`. The chat id is `chatcmpl-<X-Request-Id>`, as vLLM's. `system_fingerprint` is `llm-mini-infer-<version>`, where vLLM's default is `vllm-<version>[-<parallelism>]-<hash8>`, and the `x-llm-engine` header is `llm-mini-infer/<version>`.

## Tool calls

Tool calls are the reference's, with `--enable-auto-tool-choice --tool-call-parser gemma4` or `qwen3_coder`. Its request checks and 400s apply, and so do its normalized tools, parse, ids, deltas and finish reasons. These rules differ:
- the parser is derived from the model, when its chat template renders that format; `none` switches it off;
- `required` and a named tool are constrained on Gemma 4 too, and every tool's schema is enforced, `strict` or not;
- `response_format` with tools under `auto` lets Gemma 4 call a tool too; `required` or a named tool with `response_format` gets the call grammar on Gemma 4 too;
- the text after the last call is kept, streamed or not;
- control tokens spelled in a tool result stay text, a literal `<|image|>` included;
- markers count by token id: no call starts from text alone, Qwen's `<function=` without `<tool_call>` included;
- a Qwen `<function=` inside an argument's value stays in the value, where the reference ends the call and starts another;
- `tool_choice: "none"` takes Gemma 4's call out of the content too;
- streamed, a chunk holds one entry per call index, the arguments' last piece included;
- `vllm:tool_call_parser_invocations_total` counts only requests with tools;
- a `$defs` or `$ref` that the grammar cannot resolve is a 400, only where a grammar is built;
- unknown keys inside a message, a tool call or its function are ignored, as tool loops echo them back;
- an unknown top-level field stays a 400;
- a message that does not decode is refused before the tool-usage checks, which vLLM runs first;
- streamed and non-streamed responses give the same calls and content where vLLM's differ: whitespace after a Qwen call, and a truncated Qwen call;
- a Qwen parameter left open at its call's end is kept, as Gemma 4's; a key written twice keeps its first value;
- an argument's value nested deeper than 256 levels stays as written, a string. The value's own bracket is its first level;
- streamed, a call's arguments stop coming in pieces after 4,096 characters that begin no member. The rest comes at the call's end;
- a value under a `$ref` into the tool's `$defs` is typed by its definition, where vLLM keeps it a string;
- an object schema without `properties` and `additionalProperties` takes any keys with any values (a dict);
- inside such a dict, Gemma 4's bare values are typed by their syntax: integers, numbers, `true`, `false` and `null`;
- on Gemma 4 under `required` or a named tool, an extra key (`additionalProperties`) is refused when it sorts before a named one. The template writes it there;
- the grammars enforce `type`, `nullable`, `enum`, `const`, `properties`, `required`, `items`, `additionalProperties`, `anyOf`, `oneOf` and `$ref`. With `properties`, an absent `additionalProperties` allows no other key. They ignore the annotations, and read an `allOf` of one schema as that schema. They take `pattern`, `format`, the length, range and count bounds and `uniqueItems` without enforcing them. Any other keyword is a 400. So are an empty `enum`, a `$ref` cycle that can produce no value, and a schema outside the subset ([structured output](#structured-output)).

## Reasoning

Reasoning is the reference's, with `--reasoning-parser gemma4` or `qwen3`: `reasoning` in the message and the deltas, `reasoning_effort`, `include_reasoning`, `usage.completion_tokens_details.reasoning_tokens`, and the history under `reasoning` or `reasoning_content`. Where its stream and non-stream parse differently, both follow the stream: a cut output inside a thought the prompt opened, and a second thought. Its 400s apply. A `reasoning_effort` outside its seven values gets its message. A value the chat template refuses gets the template's 400. For example, Qwen3.8 refuses `reasoning_effort: "none"` with `chat_template_kwargs.enable_thinking: true`. These rules differ:
- the parser is derived from the model, when its chat template renders the markers; `none` switches it off;
- `reasoning_effort` goes through `reasoning.effort_map`, by default the family's, where vLLM passes it unmapped;
- a refused `reasoning.effort_map` stops the start with vLLM's message, which names the setting in place of `--reasoning-effort-map`;
- markers count by token id: a `</think>` the model spells with ordinary tokens stays text. vLLM's non-streamed parse reads it as the marker;
- with thinking on, `response_format` and the call grammars apply from the first token, the thought in front; vLLM waits for the end marker. After Qwen's closer they allow a blank line. Under a call grammar the closer comes before the calls;
- `include_reasoning: false` leaves the grammar as it is, and hides logprobs and token ids only with a reasoning parser;
- on Gemma 4 with thinking off, `required` and a named tool keep refusing the model's empty thought;
- with `tool_calls.parser` none, the text after a thought is parsed as content, not passed raw;
- Gemma 4's `<|"|>` in content is kept only with `skip_special_tokens: false`, and a request without tools is not parsed for calls (as above);
- the count and the deltas are the parse's at one token a step, whatever the step size;
- `thinking_token_budget` is a 400 (no thinking budget), and `stream_options.continuous_usage_stats` stays one: the per-chunk count is not served.

## Structured output

A `response_format` nested past 64 JSON levels gets the reference's 400, with its text and no param. This server takes a subset of JSON Schema in `response_format` and in each tool a call grammar holds. It refuses the rest with a 400 before it builds anything:
- **keywords:** a tool's schema takes the list above, and nothing else. A `response_format` also takes `prefixItems`, `minProperties`, `maxProperties` and `$anchor`. It ignores any other keyword, as llguidance does, except those llguidance acts on: `not`, `if`, `then`, `else`, `dependencies`, `dependentRequired`, `dependentSchemas`, `contains`, `minContains`, `maxContains`, `propertyNames`, `patternProperties`, `additionalItems`, `unevaluatedItems`, `unevaluatedProperties`, `$dynamicRef`, `$dynamicAnchor`, `$recursiveRef` and `$recursiveAnchor`. Those are refused with their JSON pointer, and so is a `contentSchema` that holds an `$id`;
- **no merging:** `anyOf`, `oneOf`, `$ref` and an `allOf` of one schema take only annotations, keys that start with `x-` and `$defs` beside them. llguidance merges any other key into them, so an unknown keyword there is refused too;
- **`x-guidance`:** a schema's own options win, and null counts as absent. Under `guidance`, a value that is not an object is ignored, as llguidance ignores it. Under `auto` and `xgrammar`, it is a 400;
- **`$ref`:** a definition in the root's `$defs` or `definitions`, or `#` in a `response_format`. Its name is read percent-decoded, then as a JSON Pointer token, as llguidance reads it: `#/$defs/a%20b` names `a b`, never `a%20b`, and `#/$defs/a/b` names nothing. An `$id` stands at the root only, and so does an `id` that is not a fragment in a document that names draft-04 (guidance-ai/llguidance#442, #444);
- **integers:** an integer bound outside the supported integer range is refused: an upper bound at or below the lowest 64-bit integer (an integer within 512 of it, a float within 1,024), and such a number in an `enum` or `const` that llguidance reads as an integer. Gemma 4's tools take both, as their grammar writes values as text;
- **caps,** a call grammar's tools counted together:
  - 32 levels of nesting, a `$ref`'s definition counted where it is used;
  - 400 properties in one object, and 5,000 in all;
  - 1,000 `enum` and `const` values, and 10,000 array items and object members inside them;
  - 1,000 `anyOf` and `oneOf` alternatives, and 1,000 definitions;
  - JSON nested 64 levels in a `response_format` (the reference's cap and text), 124 in the tools;
  - 16 KB of `pattern` and `x-guidance` text, and 400 strings with a `format` llguidance checks;
  - 32 optional properties in an object under `minProperties`, and 40,000 in all such objects, whose grammar grows with the square of their optional properties;
- **fixed output:** at most 4,096 bytes in a row that one schema fixes, per `response_format` and per tool. A run ends where the model chooses: a free value, an optional key, an alternative, the end of an array of no fixed length. The count includes keys, `const` values, an `enum`'s shared prefix and definitions used again. It also includes `minItems` copies, a pattern's `x{n}`, a number fixed by its bounds, `x-guidance` separators and a tool call's syntax. A `pattern` beside a `format` counts as fixed throughout, up to the pattern's longest string. The 400 names the JSON pointer and the length, under every backend: shorten the text there, or let the model write some of it. A pattern in syntax the count does not read, and an `x-guidance` whitespace pattern that fixes text, are refused.

## Deviations from the reference

- under `ignore_eos`, drafts past a drafted EOS are verified; the reference cuts them. Only the speculative counters differ;
- xgrammar's compact form (`disable_any_whitespace` under `auto` or `xgrammar`) is refused; the reference asks its pinned xgrammar 0.2.8 for `','` and `':'` separators, and that release still writes a space after a comma;
- schemas nest 32 levels, a `$ref`'s definition counted where it is used: stricter than the reference, which takes any schema llguidance compiles within its 64 JSON levels;
- tool grammars: the reference checks a parser-built tool grammar, a structural tag, at 80 JSON levels (64, and 16 for its wrapper); under Qwen's `auto` it can hold the `response_format` too. Here a `response_format` keeps the 64-level cap, and the tools their limits (32 levels, JSON 124);
- more than 4 stop strings: the reference refuses them (`VLLM_MAX_STOP_STRINGS`, a 400); this server takes 16;
- an image URL refused with a 4xx other than 408 and 429, or malformed: the reference answers 422 "Failed to fetch media from URL: ..." with param `image_url` (other failures keep their own error); this server a 400 "image: ..." with param `messages`. The download budget's 422 matches the reference;
- an image URL whose address is loopback, private, link-local, metadata or reserved, on the URL or a redirect: a 400 "image: ..." here; the reference checks no address, only `--allowed-media-domains`' names;
- `mm_processor_kwargs` and `media_io_kwargs`: the reference accepts `{}` and refuses any other value with a 400 naming `--trust-request-mm-kwargs`; this server refuses either field as unknown;
- the queue caps are checked twice, as the reference does: before a request's preparation and once it is prepared; a refusal at the second is a plain 503 here, in-band in a stream in the reference. With several frontends each checks its own counts; the reference shares the request count across its API servers, not the token count. It counts `vllm:admission_rejections_total` only with stat logging on, this server always;
- out-of-range sampling parameters (`temperature`, `top_p`, `top_k`, `min_p`, `max_tokens`, an empty stop string) get their 400 before the model's 404 and the queue caps' 503; the reference checks them after its render, so it answers those first;
- a grammar that fails to compile in the engine (rare: the frontend's own check answers 400 first) or misses its compile time limit ends as the engine's 500 "the engine rejected or failed the request" (in-band in a stream); the reference answers 400 "Invalid structured outputs: ..." (in-band in a stream);
- the engine's requests share one build of a grammar in flight (two when some want its forced prefix and some, with stop strings, do not), and a request waits for its build at most 60 s by default (`dev.structured_output.compile_timeout_s`, then a 500); the reference builds per request and waits without limit;
- a chat template slower than 30 s to render: the reference ends it with a 500; this server bounds the render at 2^20 template operations with a 400;
- the reference's GDN prefill leaves FlashInfer to choose its route; this server forces the per-sequence one at every step, so a prefill of few sequences can round differently ([Numerics](architecture.md#models));
- on Gemma 4, a thin strip of an image (1190 × 17 at 70 soft tokens) takes 70 soft tokens, as Hugging Face's processor gives it; the reference gives 69;
- a non-streamed request whose client left is a 499 in the request log and in `llm_mini_infer:http_responses_total`; the reference records an empty 200. `http_requests_total` counts it as the reference does;
- `top_logprobs` given as a float or a string is a 400; the reference converts 2.0 to 2;
- a model's image placeholder (Gemma 4's `<|image|>`) in a message's text is a 400 unless it matches an image; the reference accepts it;
- the content format (a string or parts) is detected from the template's source as the reference detected it before it tracked `namespace(...)` aliases of `messages`;
- an image whose header declares more than 178,956,970 pixels is a 400 before any decode; the reference answers 500;
- a schema, the tools' included, of more than `structured_output.max_schema_bytes` (1 MiB) is a 400; the reference has no cap;
- `logit_bias` is accepted under speculative decoding (the bias applies at every verify position); the reference refuses it there;
- more than 1,024 `logit_bias` entries is a 400 "Too many logit bias tokens: N. The max size is 1024."; the reference's v2 runner raises that text as an engine error;
- a `repetition_penalty` below 1.1754944e-38 or above 3.4028235e38 (float32's normal range) is a 400;
- a JSON number out of range (`1e999`) is a 400 at decoding ("Number out of range"), as are `NaN` and `Infinity` (not JSON: "JSON is malformed"), in a known field; the reference's parser (Python's `json`) takes them as infinity and NaN, so its range checks answer instead ("presence_penalty must be in [-2, 2], got nan."), and it clamps a `logit_bias` value of `1e999` to 100;
- `--override-generation-config` has no setting: the model's penalties are taken as they are;
- under `priority`, a request that preempts itself ends the planning of running requests for that step; the reference plans the rest.

## Deploy settings

Each row gives vLLM's setting (its default) and the server's (its default):

| vLLM | llm-mini-infer |
|---|---|
| `--served-model-name` (the model as given) | `model.served_model_name` (the model as given) |
| `--seed` (0) | `scheduler.seed` (random; the image's file sets 0) |
| `--structured-outputs-config` `backend`, `disable_any_whitespace` (`auto`, `false`) | `structured_output.backend`, `disable_any_whitespace` (`auto`, `false`; the image's file sets `guidance`, `true`) |
| `--default-chat-template-kwargs` (none) | `model.chat_template_kwargs` (`{}`) |
| `--mm-processor-kwargs` `max_soft_tokens` (the checkpoint's) | `images.max_soft_tokens` (the checkpoint's, else 280), server-wide only |
| `--gpu-memory-utilization` (0.92) | `runner.cuda.memory_utilization` (0.9) |
| `--kv-cache-dtype fp8` with `--kv-cache-dtype-skip-layers sliding_window` (`auto`) | `cache.kv_cache_dtype fp8` (`auto`): Gemma 4 only, E4M3 at scale 1.0 |
| `--enable-request-id-headers` (off) | `server.request_id_header` (`true`) |
| `--enable-auto-tool-choice`, `--tool-call-parser` (unset) | `tool_calls.parser` (the model family's) |
| `--reasoning-parser` (unset) | `reasoning.parser` (the model family's) |
| `--reasoning-effort-map` (none) | `reasoning.effort_map` (the family's) |
| `--max-num-queued-reqs` (off) | `scheduler.max_num_queued_reqs` (off), `--max-num-queued-reqs` |
| `--max-num-queued-tokens` (off) | `scheduler.max_num_queued_tokens` (off), `--max-num-queued-tokens` |
| `--max-logprobs` (20) | 20, with `server.logprobs` (`false`) |
| `VLLM_HTTP_TIMEOUT_KEEP_ALIVE` (5) | `server.keep_alive_s` (5) |
| `VLLM_MAX_IMAGE_PIXELS` (178,956,970) | `images.max_source_pixels` (178,956,970, only lowered) |
| `VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES` (715,827,880) | `images.max_request_decode_bytes` (715,827,880) |
| `VLLM_MAX_MEDIA_REQUEST_DOWNLOAD_SIZE_MB` (256) | `images.max_request_download_mb` (256) |
| `--allowed-media-domains` (none) | `images.allowed_domains` (none), on every redirect too |
| `VLLM_MEDIA_URL_ALLOW_REDIRECTS` (1) | none: at most 5 redirects, each checked |
| `VLLM_ALLOW_LONG_MAX_MODEL_LEN` (false; without it, a value up to config.json's `model_max_length` passes) | none: a `model.max_model_len` past the model's limit stops the start |

## Further rules

These rules follow the reference too.

### Request checks

- **Context.** A prompt of more than `model.max_model_len` tokens, an image counted as its placeholder, gets the reference's 400 (param `input_tokens`). Once its images are expanded, a longer prompt gets "Input length (N) exceeds model's maximum context length (M).". A prompt of exactly that length gets "max_tokens must be at least 1, got 0.".
- **`max_tokens`** is a cap: the output is clamped to the room left. A `max_tokens` above `model.max_model_len` gets the reference's 400 "max_tokens=N cannot be greater than max_model_len=M. ...".
- **Characters.** A text longer than `model.max_model_len` times the longest token's characters is refused before it is tokenized, with the reference's 400 (param `input_text`).
- **Images.** More than `images.max_per_request` images get "At most N image(s) may be provided in one prompt.".
- **Refusals** carry the reference's message, its `(parameter=P, value=V)` suffix and its `param`.
- **Sampling checks** run in the reference's order: `temperature` in [0, 2], `top_p` in (0, 1], `top_k` −1, 0 or at least 1, `min_p` in [0, 1], `max_tokens` at least 1, a signed 64-bit `seed` (−1 for none), and no empty stop string.
- **Penalties** need `server.penalties`; without it, a field set to a non-neutral value (`repetition_penalty` 1, the others 0, an empty or null `logit_bias` are neutral) is a 400 "penalties and logit_bias are off on this server (server.penalties)", with the field as param. `presence_penalty` and `frequency_penalty` take [-2, 2], `repetition_penalty` a finite number above 0. `logit_bias` keys go through `int()` (a bad key is a 400 naming it) and must be vocabulary ids; values are clamped to [-100, 100].
- **Other fields.** `max_completion_tokens` wins over `max_tokens`. An empty `stream_options` is accepted without `stream`. `"model": ""` names the served model. A `top_k` at or above the vocabulary size is off.
- **Stop ids** outside the vocabulary get the reference's 400. A repeated id counts once, and more than 16 distinct ids are a 400.
- **Unknown model.** A 404 "The model `X` does not exist. Valid aliases: ..." names the served models, on chat, `/tokenize` and `/detokenize`.
- **Logprobs** need `server.logprobs`; without it, a request for them is a 400. `top_logprobs` takes 0 to 20 (the reference's `--max-logprobs`), and null is 0. −1, the whole vocabulary, is over the limit. A value above 0 needs `logprobs: true`. A stream over the limit gets one compact JSON error event, then `[DONE]`.

### Rendering

- **A template error** is a 400 with the first Jinja `TemplateError`'s message alone.
- **`developer` messages** become `system` messages when the template does not mention a `developer` role. Several system messages, or one that is not first, then merge into one at the start.
- **Message parts.** For a template that takes parts, a tool message's text parts are joined, unless one is an image. For a template that takes a string, a message with images gets a placeholder in front of its text for each image that the text does not place.
- **Text that spells a special token** becomes that token, except in a tool result ([Tool calls](#tool-calls)).
- **The stop token** is never detokenized.

### Sampling, grammars and drafts

- **Greedy.** Only a `temperature` of 0 is greedy, and ties go to the lowest id. A `temperature` in (0, 0.01) is raised to 0.01. The filters run in the reference's order: min-p, top-k, top-p.
- **Penalties.** The bias is added, then `repetition_penalty` divides a positive logit and multiplies a negative one for any token of the prompt or the output, then `frequency_penalty` subtracts its value times the token's output count and `presence_penalty` subtracts once for a token in the output; the temperature and the filters follow. Logprobs are before all of them (`raw_logprobs`). A bias cannot lift a token that the grammar's mask refuses. Under speculative decoding they are exact: each verify position counts the drafts before it. With `server.penalties`, the model's `generation_config.json` penalties are the defaults, as the reference's `generation_config auto` takes them; Gemma 4 31B's and Qwen3.8 27B's set none.
- **Logprobs' values** are the reference's `raw_logprobs`: the logits after the grammar's mask, before the temperature and the filters. Ties sort by id. Each entry's token is decoded alone, with the byte-fallback correction over up to 4 tokens, and a floor of −9999.0. The sampled token comes first.
- **Masks** are llguidance's, as the reference's `guidance` backend computes them. A refused token is never sampled.
- **The grammar's end.** Under `guidance`, the grammar ends on the tokenizer's EOS token. Under `auto` and `xgrammar`, it ends on the model's stop ids. Under `ignore_eos`, the output after the grammar's end is not constrained.
- **Drafts and the grammar.** A draft that the grammar refuses is not counted as drafted.
- **A stop string under speculative decoding.** The usage's tokens and the logprobs' entries end at the stop. The speculative counters keep the tokens after it.
- **Speculative counters.** A draft is one verify step, and a preempted request's step counts.

### Images

- **EXIF** that cannot be read leaves the image as stored.
- **Transparency** (RGBA, LA, PA, or a `transparency` entry) is composited onto white.

### Scheduling and start-up

- **Preemption.** A page shortage preempts the lowest priority first, and a waiting request never preempts a running one.
- **`vllm:kv_cache_usage_perc`** leaves out pages that only the cache holds.
- **A shared GPU.** The server refuses to start when its memory share is not free.

### Numerics

- **FP8 GEMMs** give the same bytes as the reference's CUTLASS path (`VLLM_USE_DEEP_GEMM=0`). The reference's default on Gemma 4 is DeepGEMM, which rounds differently. Activations are quantized by the scale's reciprocal, except SiLU-and-mul's, which divide by it.
- **FP8 quantization.** The scale is max(absmax, 1e-10)/448, and a NaN becomes −448, in every quantizer. The reference's SiLU-and-mul and RMSNorm quantizers floor the scale at 1/(448·512) and make a NaN +448.
- **SiLU-and-mul** takes torch's x / (1 + e^−x), where the reference's fused kernel takes x · (1 / (1 + e^−x)). On Gemma 4's shapes, about 15% of the group scales differ by 1 to 3 ulps, and the quantized bytes are equal.
- **Fused ops** compute in fp32 and round to BF16 where the unfused ops round.
