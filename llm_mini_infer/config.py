"""Configuration: one typed msgspec Struct, from a YAML file plus CLI overrides (docs/architecture.md,
"Configuration"; the main settings: docs/operations.md, "Settings").

Three parts:
- **Settings**, the groups an operator sets (`model`, `scheduler`, `server`, ...): what to serve and how, its limits
  and its lifecycle.
- **The dev group**, `dev.<group>.<key>`: what tests, benchmarks and A/B runs set (the fake runner, pool and encoder
  sizes, kernel choices, the switches of features on by default, host pinning). A deployment takes their defaults,
  derived where they depend on the model, the GPU or the host, and sets none of them.
- **The server's state** (`STATE_FIELDS`): what the hub, `derive()`, the launcher and the engine fill in, never set.

Every setting but `model.path` has a default, a constant or one derived from the model files, the GPU or the host;
`derive()` fills the derived ones and records why. Pool sizes need the runner's capability report, so the engine
finishes them (`cache.pools`), and the launcher logs the effective configuration once the engine is ready.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Literal

import msgspec
import yaml

from .kernels.build import cgroup_cpu_quota  # the standard library only (the kernel build's job cap reads it too)
from .runner.mtp import MAX_GEMMA_MTP_K, MAX_MTP_K

log = logging.getLogger(__name__)


class _Group(msgspec.Struct, forbid_unknown_fields=True, kw_only=True):
    pass


class ModelConfig(_Group):
    # A local checkpoint directory, or a Hugging Face repo id (org/name) that the launcher downloads into the Hugging
    # Face cache first (llm_mini_infer.hub; the token from HF_TOKEN)
    path: str = ""
    revision: str | None = None  # a repo id's branch, tag or commit; default: main
    served_model_name: str | None = None  # default: the path as given, a repo id included (vLLM's rule)
    tokenizer: str | None = None  # directory with tokenizer.json; default: the model directory
    # Default: config.json's smallest position limit (8192 if none); a derived one is lowered to what the KV pool holds.
    # A set one past that limit is refused
    max_model_len: int | None = None
    # Defaults for the chat template's variables, under each request's chat_template_kwargs (the request wins): vLLM's
    # --default-chat-template-kwargs, e.g. {enable_thinking: false} for Qwen3.5/3.8, whose template thinks by default
    chat_template_kwargs: dict[str, Any] = {}


class SamplingDefaults(_Group):
    """Used when a request leaves the field out. Default: the model's generation_config.json."""

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    max_tokens: int | None = None


class SchedulerConfig(_Group):
    # Tokens per step, prefill chunks and decode rows together. Default 2048: larger steps hold more prefill beside the
    # decode rows, which lengthens their time per output token under load
    max_num_batched_tokens: int | None = None
    max_num_seqs: int | None = None  # default 64
    # The waiting requests' order: arrival, or (priority, arrival) with the lowest value first, which also picks the
    # running request a page shortage preempts first; a waiting request never preempts a running one
    policy: Literal["fcfs", "priority"] = "fcfs"
    seed: int | None = None  # seeds the per-request seeds drawn for requests without one
    # vLLM's queue caps (None: off), checked by each frontend before a chat completion's response starts: past one, a
    # 503 the client can retry on another instance. The requests in flight, waiting or running: size it as
    # max_num_seqs plus the queue depth wanted
    max_num_queued_reqs: int | None = None
    # The prompt tokens of the requests not yet given their first token, each whole (a prefix-cache hit or a prefill
    # in progress counts all of it), checked before the new request's own: size it as the TTFT target times the
    # prefill throughput
    max_num_queued_tokens: int | None = None


class CacheConfig(_Group):
    # The radix prefix cache (architecture.md, "The radix tree"); off, every prompt is computed from scratch
    enable_prefix_caching: bool = True


class SpeculativeConfig(_Group):
    """MTP speculative decoding (architecture.md, "Speculative decoding"): drafts from the checkpoint's MTP layer
    (Qwen3.5/3.8) or from a separate assistant model (Gemma 4)."""

    # A separate drafter's checkpoint, Gemma 4's assistant (google/gemma-4-31B-it-assistant): a local directory or a
    # Hugging Face repo id, fetched as model.path is (hub)
    model: str | None = None
    revision: str | None = None  # a repo id's branch, tag or commit; default: main
    # k, the drafts per step (0-7; the Qwen runner at most 3). Default: 5 with an assistant model, whose drafts keep
    # paying deep into the chain; 3 with the checkpoint's MTP layer; else 0 (off)
    num_speculative_tokens: int | None = None


class ImageConfig(_Group):
    """Image inputs (architecture.md, "Images"; `llm_mini_infer.vision`)."""

    enabled: bool | None = None  # default: on with a vision tower this server runs
    # Qwen3.5/3.8's resize bounds; default: preprocessor_config.json size.shortest_edge / size.longest_edge. Gemma 4
    # derives both from max_soft_tokens and refuses other values
    min_pixels: int | None = None
    max_pixels: int | None = None
    # Gemma 4 only: the most soft tokens one image takes (vLLM's --mm-processor-kwargs '{"max_soft_tokens": N}'; a 16:9
    # image takes 264 at 280, 66 at 70). Default: the checkpoint's (`vision.gemma.GemmaVision.from_model_dir`; 280 on
    # google/gemma-4-31B-it). Server-wide: a request's mm_processor_kwargs is an unknown field (400). A family sized by
    # pixels (Qwen) refuses it
    max_soft_tokens: Literal[70, 140, 280, 560, 1120] | None = None
    max_per_request: int = 16  # images in one request
    max_bytes: int = 20 << 20  # one image's bytes, decoded from a data URI or downloaded
    # An image's stored size (its header) at most: more is a 400 before the image pool decodes it (vision.probe).
    # Pillow's decompression-bomb bound (also VLLM_MAX_IMAGE_PIXELS' default); it can be lowered, not raised
    max_source_pixels: int = 178_956_970
    # A request's images, 4 bytes a pixel of their stored sizes, summed in prompt order, at most: more is a 400 before
    # any is decoded (VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES and its default, one image of max_source_pixels; 0: no limit)
    max_request_decode_bytes: int = 715_827_880
    # The MiB a request's http(s) image downloads receive, counted as they arrive, at most: more stops them with a 422
    # (VLLM_MAX_MEDIA_REQUEST_DOWNLOAD_SIZE_MB and its default; data URIs are not counted; 0: no limit)
    max_request_download_mb: int = 256
    # Fetch http(s) image URLs (data URIs are always taken). Loopback, private, link-local, metadata and reserved
    # addresses are refused, after DNS and on every redirect (vision.urlguard)
    allow_urls: bool = False
    # CIDRs fetched from despite that block, such as an in-cluster image store's ([10.0.0.0/8])
    allowed_networks: list[str] = []
    # Only these hostnames, exactly, on the URL and on every redirect (vLLM's --allowed-media-domains); empty: any
    allowed_domains: list[str] = []
    fetch_timeout_s: float = 10.0  # an http(s) image's whole download
    workers: int | None = None  # image pool processes; default: the CPU quota after the engine and frontends, capped


class StructuredOutputConfig(_Group):
    """Structured output: OpenAI's `response_format` (architecture.md, "Structured output";
    `llm_mini_infer.grammar`)."""

    enabled: bool | None = None  # default: on when llguidance is installed and the checkpoint has a tokenizer.json
    # Keys and defaults of vLLM's --structured-outputs-config (architecture.md, "Structured output";
    # grammar.spec): guidance is llguidance's JSON grammar; auto and xgrammar emulate xgrammar's JSON language. The
    # flag's disable_additional_properties is not supported (an unknown key)
    backend: Literal["auto", "guidance", "xgrammar"] = "auto"
    disable_any_whitespace: bool = False  # compact JSON (backend guidance only)
    # The bytes of `response_format.json_schema.schema` as received, at most: more is a 400 before the schema is decoded
    # (the grammar caches keep their texts whole, and each request clones its compiled grammar)
    max_schema_bytes: int = 1 << 20


class ToolCallConfig(_Group):
    """Tool calls (architecture.md, "Tool calls"; `llm_mini_infer.output.tool_calls`)."""

    # The format requests' tools are parsed and constrained in: vLLM's --tool-call-parser names (with
    # --enable-auto-tool-choice), gemma4 or qwen3_coder; none: a request with tools gets the 400s of a server without
    # a tool parser (tool_choice "none" and null are served). Default: the model family's when its chat template renders
    # that format (Gemma 4's <|tool_call>, Qwen3.5/3.8's <function=...><parameter=...>), else none
    parser: Literal["gemma4", "qwen3_coder", "none"] | None = None


class ReasoningConfig(_Group):
    """Reasoning output (architecture.md, "Reasoning"; `llm_mini_infer.output.reasoning`)."""

    # The thought's parser: vLLM's --reasoning-parser names, gemma4 or qwen3; none: no thought is parsed (the output is
    # content, tools parsed as without it). Default: the model family's when its chat template renders that family's
    # markers (Gemma 4's <|channel>, Qwen3.5/3.8's <think> and </think>), else none
    parser: Literal["gemma4", "qwen3", "none"] | None = None
    # A request's reasoning_effort onto the chat template's own values (vLLM's --reasoning-effort-map): an object of
    # efforts (none, minimal, low, medium, high, xhigh, max) to strings; an effort it lacks passes as it is. A set map
    # replaces the default whole. Default: the model family's (`output.reasoning.EFFORT_MAPS`): Qwen3.5/3.8 minimal and
    # low to low, medium to medium, high, xhigh and max to xhigh; Gemma 4 and others {}. Anything else stops the start
    # with vLLM's message (`output.reasoning.check_effort_map`)
    effort_map: Any = None


class CudaRunnerConfig(_Group):
    # Share of device memory for weights, activations and pools; startup fails when that much is not free
    memory_utilization: float = 0.9
    # A block-FP8 model's precise prefix (architecture.md, "The precise prefix"): "reuse" runs prompts on the FP8 path
    # and recomputes a cached prefix with BF16 activations at its second reuse. "bf16" runs every prefill row of a
    # prompt without a cache hit with BF16 activations, into a precise copy (unique prompts pay too); a prompt with a
    # hit resumes from the deeper copy, the precise one on a tie, and runs the rest on the FP8 path. "fp8" quantizes
    # every activation
    fp8_cold_prefill: Literal["reuse", "bf16", "fp8"] = "reuse"


class RunnerConfig(_Group):
    # The GPU, or the fake (random tokens: tests and benchmarks; dev.runner.fake). Never derived from what torch sees:
    # without a GPU the cuda runner fails at startup
    kind: Literal["fake", "cuda"] = "cuda"
    cuda: CudaRunnerConfig = msgspec.field(default_factory=CudaRunnerConfig)


class ServerConfig(_Group):
    host: str = "0.0.0.0"
    port: int = 8000
    num_frontends: int | None = None  # default 1 (one GPU); N processes share the port with SO_REUSEPORT
    drain_timeout_s: float = 30.0  # how long a drain waits for in-flight requests (then aborts them)
    # After SIGTERM: keep serving new requests this long while /health reports 503, so a load balancer that
    # health-checks stops routing here before the server refuses them (a balancer that does not retry a 503 would
    # fail them): set it above the balancer's detection time (its failure threshold x check interval). 0: refuse at
    # once. SIGINT (Ctrl-C) skips it.
    drain_grace_s: float = 0.0
    request_id_header: bool = True  # echo X-Request-Id on every response, as vLLM's --enable-request-id-headers
    # Chat completions' `logprobs` and `top_logprobs` (at most 20: the OpenAI API's bound). Off (a request that
    # asks is a 400), the head runs as without them; on, the CUDA runner keeps its head's logits in a buffer of
    # max_num_seqs x (1 + k) x the vocabulary in fp32, taken from the KV pool (403 MB at 64 sequences, k = 5 and a
    # vocabulary of 262,144)
    logprobs: bool = False
    # How long an idle HTTP connection stays open for its next request: uvicorn's timeout_keep_alive; its default, 5 s,
    # is that of VLLM_HTTP_TIMEOUT_KEEP_ALIVE. A proxy that pools connections to this server must close idle ones sooner
    keep_alive_s: float = 5.0


class EngineConfig(_Group):
    # The launcher stops an engine whose step loop made no progress this long (a hung device call): in-flight requests
    # fail and the server exits non-zero, so its supervisor restarts it. A loop turn is at most a step (well under 1 s)
    # or a 1 s idle poll. 0: off
    watchdog_s: float = 60.0
    # Plan at the forward/sampling boundary (architecture.md, "Planning at the forward/sampling boundary";
    # engine/core.py): when the step in flight is split, poll until its forward has ended before planning the next, so
    # a request that arrives during that forward joins the next step instead of the one after. Default: off (it leaves
    # the host only the sampling part, not the forward, to launch the next step); set, a margin the runner's lone
    # decode step finds short (`engine.core.derive_plan_at_boundary`) is logged as a warning. On CUDA: Gemma 4 with
    # structured output only
    plan_at_boundary: bool | None = None


class ObservabilityConfig(_Group):
    log_level: str = "info"
    # One line per chat request at INFO, metadata only (id, status, token counts, finish, timings; never prompts,
    # images or outputs): docs/operations.md, "Logging"
    log_requests: bool = True
    # /debug/stages, /debug/engine, /debug/images and /debug/reset, which need the API key when one is set: raw
    # per-request and per-step timings, the image pool's workers, and a reset of the timings
    debug_endpoints: bool = False


# -- the dev group: what tests, benchmarks and A/B runs set (`dev.<group>.<key>`; module docstring) -------------------


class DevSchedulerConfig(_Group):
    long_prefill_token_threshold: int = 0  # cap on one request's chunk per step; 0 = off
    # Decode rows x prefill tokens of a step that carries decode rows: its prefill gets at most this
    # divided by its decode rows (at least one checkpoint grid), which bounds the waiting it adds to them, summed.
    # Default: derived from the runner's measured step costs (sched.scheduler.derive_mixed_step_row_tokens); 0 = off
    # (every step may fill max_num_batched_tokens)
    mixed_step_row_tokens: int | None = None


class DevCacheConfig(_Group):
    # Default 16. Sliding-window models (Gemma 4): the window page; the full layers' pages hold the positions that fill
    # as many bytes (320 on Gemma 4 31B), in one KV arena with them (cache.pools.size_arena)
    page_size: int | None = None
    # Default: what is left after the state pool (architecture.md, "Pools"); sliding-window models: the KV arena's
    # pages, either kind
    num_pages: int | None = None
    num_state_slots: int | None = None  # default: max_num_seqs live slots + checkpoint_slots + null
    # Default: min(max_num_seqs, the slots in 25% of the memory after the live and null slots); 0 without recurrent
    # state or prefix caching
    checkpoint_slots: int | None = None
    decode_checkpoint_interval: int = 256  # hybrid models: checkpoint decoding requests every N positions; 0 = off
    # Checkpoints a conversation has moved past are evicted before every live one (architecture.md, "Eviction"); false:
    # every checkpoint least recently used first, for traffic that resends its prompts more than it extends them
    demote_superseded: bool = True
    # The precise prefix (runner.cuda.fp8_cold_prefill reuse; architecture.md, "The precise prefix"): the resumes a
    # cached FP8 checkpoint serves before the next requester recomputes its prefix with BF16 activations, and the
    # fewest tokens such a refine recomputes
    refine_after: int = 1
    refine_min: int = 256
    # How a refine's PRECISE rows are cut into steps (architecture.md, "The precise prefix"): at multiples of
    # refine_chunk_tokens from position 0 and at the refine's end only, each piece whole in one step, for one longer
    # decode gap per piece. A multiple of the checkpoint grid, at most max_num_batched_tokens - max_num_seqs x (1 + k)
    # (a larger piece would wait for steps with fewer decode rows); checked when set (derive(); the grid when the engine
    # starts). Default: derived (`sched.scheduler.derive_refine_chunk_tokens`: at most 2,048, with room for decode rows
    # of 1 + `mtp_k_max` tokens)
    refine_chunk_tokens: int | None = None


class DevSpeculativeConfig(_Group):
    # The drafter's LM head (runner/cuda/draft_head.py): the tokens it scores, ranked from the tokens the server emits
    # (0: the whole vocabulary), and its weights' dtype (fp8: a block-FP8 copy of its rows). Drafts change acceptance
    # and, where outputs depend on the batch, their rounding. Default: derived (below)
    draft_vocab: int | None = None
    draft_head_dtype: Literal["auto", "bfloat16", "fp8"] = "auto"
    # The grammar-universe draft head (architecture.md, "Grammar-aware heads"): on the steps the sparse head could
    # serve, with it on or off, the drafter scores only the union of every such step's masks. Default: on with drafts
    # and structured output
    draft_universe: bool | None = None
    # Gemma 4's assistant: its queries at RoPE position p + 1, the token after the last committed position p (as in
    # Hugging Face's assisted generation), instead of p (as in vLLM); the KV each reads, [0, p] and the sliding
    # window's last W positions, stays (`runner.cuda.gemma`). It changes acceptance and, where outputs depend on the
    # batch, their rounding
    draft_query_next: bool = False


class DevImageConfig(_Group):
    arena_mb: int | None = None  # shared memory for pixels; default: derived from max_pixels
    encoder_budget: int | None = None  # image tokens the encoder computes per step; default: derived (below)
    encoder_cache_tokens: int | None = None  # image embeddings kept on the device; default: 2 x encoder_budget
    # The image digest's bits in its span's radix key (vision.family.span_key); fewer force collisions, which every
    # cache tells apart by the whole digest (tests)
    span_key_bits: int = 62


class DevStructuredOutputConfig(_Group):
    threads: int | None = None  # llguidance's mask-filling threads; default: 1 (GRAMMAR_THREADS)
    # The forced prefix (architecture.md, "Grammar-aware heads"): the tokens a grammar forces at its start go into the
    # request's final prefill chunk as decided output tokens, not sampled in steps of their own. Default: on with
    # structured output
    forced_prefix: bool | None = None
    # The EOS-free finish (architecture.md, "Grammar-aware heads"): once a request's grammar has stopped (only the EOS
    # may follow), it finishes at once with the EOS appended and counted, unless it is alone in the step in flight.
    # Needs a grammar that ends on one EOS id. Default: on with structured output when its grammars do: always with the
    # guidance backend (the tokenizer's eos_token), with auto and xgrammar on a checkpoint with one stop id (Gemma 4's
    # has three and Qwen3.8's two: off)
    eos_free_finish: bool | None = None
    # The longest a request waits for its grammar's build in the engine, from its arrival there; past it the request
    # fails (a 500). The build goes on and fills the cache (architecture.md, "Structured output")
    compile_timeout_s: float = 60.0


class FakeRunnerConfig(_Group):
    step_ms: float = 5.0  # device time of one forward pass
    sample_ms: float = 0.0  # device time of sampling
    prefill_us_per_token: float = 0.0  # extra forward time per query token
    vocab_size: int | None = None  # default: config.json's, else 32000
    memory_gb: float = 62.0  # pool memory reported as free
    eos: list[tuple[int, int]] = []  # (seed, position) pairs where the fake samples EOS
    # (seed, position, tokens): from that position a request admitted with that seed samples these tokens (recorded
    # outputs: tool calls in tests), under masks and in drafts as its own samples; after them, the fake's again
    scripts: list[tuple[int, int, list[int]]] = []
    strict: bool = False  # check KV contents every step (O(context); for tests); always on for sliding-window models
    wait: Literal["spin", "sleep"] = "spin"  # how wait() passes device time; spin is CUDA's default behaviour
    accept_p: float = 0.8  # MTP: the chance that each draft is accepted (after the previous ones were)
    precise_rows: bool = False  # takes PRECISE rows, as a block-FP8 model's runner does (the precise prefix; tests)
    load_s: float = 0.0  # its "model load" takes this long: a slow or hung start (tests)
    # Reports batch invariance (`dev.runner.cuda.batch_invariant`'s scheduler side: a GDN model's prefill cuts on the
    # grid, every final chunk's forced prefix whole); only the test harness, which knows the requests' exact cuts,
    # checks them
    batch_invariant: bool = False


class DevCudaConfig(_Group):
    device: str = "cuda"  # "cpu" runs the same runner on the torch reference kernels (tests)
    dtype: Literal["auto", "bfloat16", "float32"] = "auto"  # auto: bfloat16 on CUDA, float32 on CPU
    attention: Literal["auto", "flashinfer", "torch"] = "auto"  # auto: FlashInfer on CUDA, torch on CPU
    gdn: Literal["auto", "flashinfer", "torch"] = "auto"  # GDN prefill kernel; decode is our fused kernel
    memory_gb: float | None = None  # pool memory to report instead of profiling (CPU default: 1 GiB)
    check: bool = False  # compare the device tables with the host mirrors after every step (syncs; tests)
    graphs: bool = True  # CUDA graphs: full graphs for decode steps, breakable graphs for prefill and mixed steps
    # Our CUDA kernels and the FP8 GEMMs; false: their torch references (the sampler stays fused on CUDA), with no CUDA
    # graphs or PDL; Gemma 4 then needs `attention: torch` (its full attention is our kernel)
    fused_ops: bool = True
    decode_buckets: list[int] | None = None  # batch sizes with a decode graph (default: default_decode_buckets)
    token_buckets: list[int] | None = None  # token counts with a breakable graph (default: default_token_buckets)
    vision_buckets: list[int] | None = None  # patch counts with a vision-tower graph ([]: eager; default: derived)
    fp8_gemm: Literal["auto", "cutlass", "sglang", "flashinfer", "torch"] = "auto"  # block-FP8 GEMMs (kernels.fp8)
    # The grammar-sparse head (architecture.md, "Grammar-aware heads"): on a step whose every sample token has a
    # grammar mask, the target's head scores only the union of the step's masks; other steps (a row without a mask, a
    # union over sparse_head_rows) take the dense head. Default: on with structured output
    sparse_head: bool | None = None
    # The tokens the sparse head scores at most (1-65,536), and the draft universe's capacity
    # (dev.speculative.draft_universe). Default: 1,024 (about 0.4% of the head)
    sparse_head_rows: int | None = None
    # Programmatic dependent launch (architecture.md, "CUDA graphs"): our kernels and the FP8 GEMMs may start while the
    # kernel before them ends; outputs stay bit for bit the serialized launches'. Default: on (needs fused_ops)
    pdl: bool | None = None
    # Always-split graphs (architecture.md, "CUDA graphs"): with structured output every step runs as a forward graph
    # and a sampling graph, masked or not, and the fused graphs are not captured. Default: on with structured output
    always_split: bool | None = None
    # The CUDA sync check (`runner.cuda.sync_check`): launch_forward and launch_sample run under torch's sync debug mode
    # and its blind spots' checks, a sync warning (warn) or raising (error, which ends the engine loop: the GPU suite's)
    sync_check: Literal["off", "warn", "error"] = "off"
    # A file (Gemma 4 with images): after startup, record CUDA memory history through one eager tower call at the
    # largest bucket and dump its snapshot there (`runner.cuda.memory_report`); its peak against free memory is logged
    memory_history: str | None = None
    # Batch invariance (answer reproducibility; architecture.md, "Batch invariance"): a row's bits do not depend on the
    # rows beside it, its MTP verify window's offset or the step's kind. Default: on for Gemma 4, and for Qwen3.8 with a
    # block-FP8 checkpoint, when model.max_model_len is at most 8,184 (BATCH_INVARIANT_MAX_LEN less 8) and the token
    # budget, the chunk cap and dev.cache.page_size allow it (`_invariance_blocked`). Forced on past that length, a
    # fixed split holds about a 32nd of the context instead of 256 keys
    batch_invariant: bool | None = None


class DevRunnerConfig(_Group):
    fake: FakeRunnerConfig = msgspec.field(default_factory=FakeRunnerConfig)  # runner.kind fake's
    cuda: DevCudaConfig = msgspec.field(default_factory=DevCudaConfig)


class DevServerConfig(_Group):
    cpus: list[int] | None = None  # cores for the frontend processes, one each (Linux); default: below the engine's
    spin_ms: float | None = None  # poll instead of sleeping while requests are in flight and this long after


class DevEngineConfig(_Group):
    cpu: int | None = None  # core for the engine thread (Linux); default: the last CPU this process may use
    idle_spin_ms: float | None = None  # poll instead of sleeping this long after the engine goes idle
    gc_freeze: bool = True
    inline_threshold: int = 256  # arrays of fewer bytes than this travel inside the message, others as frames


class DevConfig(_Group):
    scheduler: DevSchedulerConfig = msgspec.field(default_factory=DevSchedulerConfig)
    cache: DevCacheConfig = msgspec.field(default_factory=DevCacheConfig)
    speculative: DevSpeculativeConfig = msgspec.field(default_factory=DevSpeculativeConfig)
    images: DevImageConfig = msgspec.field(default_factory=DevImageConfig)
    structured_output: DevStructuredOutputConfig = msgspec.field(default_factory=DevStructuredOutputConfig)
    runner: DevRunnerConfig = msgspec.field(default_factory=DevRunnerConfig)
    server: DevServerConfig = msgspec.field(default_factory=DevServerConfig)
    engine: DevEngineConfig = msgspec.field(default_factory=DevEngineConfig)


class Config(msgspec.Struct, forbid_unknown_fields=True, kw_only=True):
    model: ModelConfig = msgspec.field(default_factory=ModelConfig)
    sampling: SamplingDefaults = msgspec.field(default_factory=SamplingDefaults)
    scheduler: SchedulerConfig = msgspec.field(default_factory=SchedulerConfig)
    cache: CacheConfig = msgspec.field(default_factory=CacheConfig)
    speculative: SpeculativeConfig = msgspec.field(default_factory=SpeculativeConfig)
    images: ImageConfig = msgspec.field(default_factory=ImageConfig)
    structured_output: StructuredOutputConfig = msgspec.field(default_factory=StructuredOutputConfig)
    tool_calls: ToolCallConfig = msgspec.field(default_factory=ToolCallConfig)
    reasoning: ReasoningConfig = msgspec.field(default_factory=ReasoningConfig)
    runner: RunnerConfig = msgspec.field(default_factory=RunnerConfig)
    server: ServerConfig = msgspec.field(default_factory=ServerConfig)
    engine: EngineConfig = msgspec.field(default_factory=EngineConfig)
    observability: ObservabilityConfig = msgspec.field(default_factory=ObservabilityConfig)
    dev: DevConfig = msgspec.field(default_factory=DevConfig)
    # The server's state, filled by the hub, derive(), the launcher and the engine (STATE_FIELDS): load() refuses them
    model_source: str = ""  # a downloaded model's repo@commit (llm_mini_infer.hub); model.path is then its local
                            # snapshot
    speculative_source: str = ""  # the same for a downloaded speculative.model
    eos_token_ids: list[int] = []
    grammar_eos_token_ids: list[int] = []  # the engine's: the EOS ids a finished grammar allows (grammar.engine)
    # The k the derived refine piece leaves room for (dev.cache.refine_chunk_tokens): with an assistant its largest,
    # whatever k is set, so k never changes the precise copy; else the set k (0: no drafter)
    mtp_k_max: int = 0
    vocab_size: int = 0
    cpu_budget: float = 0.0
    image_tokens_max: int = 0  # the most placeholder tokens one image takes (0: no images)
    pixel_arena: str = ""  # the launcher's shared pixel file (vision.arena)
    pool_dir: str = ""  # where the image pool's sockets are
    runner_info: dict[str, Any] = {}  # the engine's: the runner's capabilities and startup report (/server_info)
    reasons: dict[str, str] = {}


STATE_FIELDS = tuple(f.name for f in msgspec.structs.fields(Config)
                     if not (isinstance(f.type, type) and issubclass(f.type, _Group)))


def leaves(group: type[_Group] | type[Config], prefix: str = "", groups: bool = False) -> list[str]:
    """The dotted names of a group's settings, its subgroups' included (with `groups`, its subgroups' names)."""
    out = []
    for f in msgspec.structs.fields(group):
        if isinstance(f.type, type) and issubclass(f.type, _Group):
            out += [prefix + f.name] * groups + leaves(f.type, f"{prefix}{f.name}.", groups)
        elif not groups:
            out.append(prefix + f.name)
    return out


SETTINGS = tuple(k for k in leaves(Config) if k not in STATE_FIELDS and not k.startswith("dev."))
# The dev settings and the dev subgroups by the names they would have outside the dev group, where they are refused
DEV_NAMES = frozenset({*leaves(DevConfig), *leaves(DevConfig, groups=True)} - set(leaves(Config, groups=True)))


def _moved(data: dict[str, Any], prefix: str = "") -> list[str]:
    """The dev settings (and subgroups) a file or override sets outside the dev group."""
    out = []
    for key, value in data.items():
        name = prefix + key
        if name in DEV_NAMES:
            out.append(name)
        elif isinstance(value, dict) and name != "dev" and not prefix.startswith("dev"):
            out += _moved(value, name + ".")
    return out


def load(path: str | None, overrides: dict[str, Any]) -> Config:
    """YAML file, then dotted overrides (`scheduler.max_num_seqs: 128`), validated against the schema. The state
    fields are refused (a value there would be used as given, or overwritten), and so are dev settings given outside
    the dev group, naming where they go."""
    data: dict[str, Any] = {}
    if path:
        try:
            data = yaml.safe_load(Path(path).read_text()) or {}
        except yaml.YAMLError as exc:  # a ValueError, so that the CLI and the image's checks exit 2 for it
            raise ValueError(f"{path}: not valid YAML: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"{path}: expected a mapping")
    for dotted, value in overrides.items():
        node = data
        *parents, leaf = dotted.split(".")
        for p in parents:
            node = node.setdefault(p, {})
            if not isinstance(node, dict):
                raise ValueError(f"override {dotted}: {p} is not a group")
        node[leaf] = value
    state = sorted(k for k in data if k in STATE_FIELDS)
    if state:
        raise ValueError(f"invalid configuration: {', '.join(state)}: the server's own state (derived from the model "
                         "and the host, or filled by the launcher and the engine; /server_info shows it), not a "
                         "setting")
    moved = _moved(data)
    if moved:
        raise ValueError(f"invalid configuration: {', '.join(moved)}: dev settings (tests, benchmarks, A/B runs; a "
                         f"deployment takes their defaults): set {', '.join('dev.' + k for k in moved)}")
    try:
        return msgspec.convert(data, Config)
    except msgspec.ValidationError as exc:
        raise ValueError(f"invalid configuration: {exc}") from exc


def parse_override(text: str) -> tuple[str, Any]:
    key, sep, value = text.partition("=")
    if not sep:
        raise ValueError(f"expected key=value, got {text!r}")
    try:
        return key.strip(), yaml.safe_load(value)
    except yaml.YAMLError as exc:
        raise ValueError(f"{key.strip()}: the value is not valid YAML: {exc}") from exc


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


_MAX_LEN_KEYS = ("max_position_embeddings", "n_positions", "max_seq_len", "seq_length", "max_sequence_length")


DRAFT_VOCAB = 32_768  # the default draft vocabulary of a checkpoint's MTP layer


def derive(cfg: Config) -> Config:
    """Fill model- and host-derived defaults, recording a reason for each."""
    reasons: dict[str, str] = {}
    model_dir = Path(cfg.model.path)
    if not model_dir.is_dir():
        raise ValueError(f"model directory {model_dir} does not exist")
    hf = _read_json(model_dir / "config.json")
    text = hf.get("text_config", hf)
    gen = _read_json(model_dir / "generation_config.json")

    m = cfg.model
    if m.served_model_name is None:
        m.served_model_name = cfg.model_source.split("@")[0] or m.path
        reasons["model.served_model_name"] = "the repo id as given" if cfg.model_source else "the model path as given"
    found = [(k, int(text[k])) for k in _MAX_LEN_KEYS if isinstance(text.get(k), int)]
    key, limit = min(found, key=lambda kv: kv[1]) if found else (None, None)
    if m.max_model_len is None:
        if found:
            m.max_model_len = limit
            reasons["model.max_model_len"] = f"config.json {key}"
        else:
            m.max_model_len = 8192
            reasons["model.max_model_len"] = "config.json has no position limit; 8192"
    elif limit is not None and m.max_model_len > limit:
        # Refused, as in vLLM (where VLLM_ALLOW_LONG_MAX_MODEL_LEN lifts it; here nothing does)
        raise ValueError(f"model.max_model_len {m.max_model_len}: at most the model's position limit, config.json "
                         f"{key} {limit}")

    s = cfg.sampling
    keys = {"temperature": "temperature", "top_p": "top_p", "top_k": "top_k", "min_p": "min_p",
            "max_tokens": "max_new_tokens"}
    used = []
    for field, gkey in keys.items():
        if getattr(s, field) is None and gkey in gen:
            setattr(s, field, gen[gkey])
            used.append(f"{field}={gen[gkey]}")
    if used:
        reasons["sampling"] = "generation_config.json: " + ", ".join(used)
    if s.temperature is None:
        s.temperature = 1.0
    if s.top_p is None:
        s.top_p = 1.0
    if s.top_k is None:
        s.top_k = 0
    if s.min_p is None:
        s.min_p = 0.0

    eos = gen.get("eos_token_id", text.get("eos_token_id", hf.get("eos_token_id")))
    cfg.eos_token_ids = sorted({int(e) for e in (eos if isinstance(eos, list) else [eos]) if e is not None})
    reasons["eos_token_ids"] = ("generation_config.json eos_token_id" if "eos_token_id" in gen
                                else "config.json eos_token_id")
    if not cfg.eos_token_ids:
        reasons["eos_token_ids"] = ("none: no eos_token_id in generation_config.json or config.json (a request stops "
                                    "at max_tokens or a stop string)")
        log.warning("eos_token_ids: %s", reasons["eos_token_ids"])
    cfg.vocab_size = int(text.get("vocab_size", hf.get("vocab_size", 0)) or 0)

    sc = cfg.scheduler
    if sc.max_num_batched_tokens is None:
        sc.max_num_batched_tokens = 2048
        reasons["scheduler.max_num_batched_tokens"] = ("2048: larger steps lengthen the decode rows' time per output "
                                                       "token")
    if sc.max_num_seqs is None:
        sc.max_num_seqs = 64
        reasons["scheduler.max_num_seqs"] = "64; the state pool is sized from it"
    if cfg.dev.cache.page_size is None:
        cfg.dev.cache.page_size = 16
        reasons["dev.cache.page_size"] = "16 tokens, independent of the state size"
    sp = cfg.speculative
    mtp_layers = int(text.get("mtp_num_hidden_layers", 0) or 0)
    assistant = _check_assistant(cfg, hf)
    if sp.num_speculative_tokens is None:
        sp.num_speculative_tokens = 5 if assistant else 3 if mtp_layers else 0
        reasons["speculative.num_speculative_tokens"] = (
            "5 drafts from the assistant model (speculative.model): they keep paying deep into the chain (k = 5 beat "
            "k = 3 at every load measured on the Blackwell GPUs tested)"
            if assistant
            else "3 drafts from the checkpoint's MTP layer (the most the Qwen runner takes)" if mtp_layers
            else "off: the checkpoint has no MTP layer and no speculative.model is set")
    elif sp.num_speculative_tokens and not (mtp_layers or assistant):
        raise ValueError("speculative.num_speculative_tokens > 0 needs an MTP layer in the checkpoint "
                         "(mtp_num_hidden_layers) or a drafter model (speculative.model)")
    if not 0 <= sp.num_speculative_tokens <= MAX_MTP_K:
        raise ValueError(f"speculative.num_speculative_tokens must be 0-{MAX_MTP_K} (the verify window is 1 + k <= "
                         f"{1 + MAX_MTP_K} tokens)")
    cfg.mtp_k_max = max(MAX_GEMMA_MTP_K, sp.num_speculative_tokens) if assistant else sp.num_speculative_tokens
    if assistant:
        why = (f"the assistant model's largest k, whatever k is set (speculative.model; the verify window holds 1 + k "
               f"<= {1 + MAX_MTP_K} tokens)")
    elif not mtp_layers:
        why = "no drafter (the checkpoint has no MTP layer and no speculative.model is set)"
    else:
        why = "the set k: the checkpoint's MTP layer (Qwen) derives its refine piece from the set k"
    reasons["mtp_k_max"] = f"{cfg.mtp_k_max}: {why}"
    # Below either, the scheduler never plans a step: the server starts and hangs (our decode rows are 1 + k wide)
    if sc.max_num_seqs < 1:
        raise ValueError(f"scheduler.max_num_seqs {sc.max_num_seqs}: at least 1")
    for key in ("max_num_queued_reqs", "max_num_queued_tokens"):
        v = getattr(sc, key)
        if v is not None and v < 0:
            raise ValueError(f"scheduler.{key} {v}: 0 or more, or unset (no limit)")
    if sc.max_num_batched_tokens < 1 + sp.num_speculative_tokens:
        raise ValueError(f"scheduler.max_num_batched_tokens {sc.max_num_batched_tokens}: at least 1 + "
                         f"speculative.num_speculative_tokens = {1 + sp.num_speculative_tokens} (a decode row)")
    vocab = int(text.get("vocab_size", hf.get("vocab_size", 0)) or 0)
    dsp = cfg.dev.speculative
    if dsp.draft_vocab is None:
        # FR-Spec's draft vocabulary size, ranked online (runner/cuda/draft_head.py): on Qwen3.8 27B it shortens the
        # step a lone request decodes in by reading a fraction of the head, with acceptance equal to the whole head's
        # once ranked. Not for an assistant model (Gemma 4): its own head is narrow (1,024 wide: 512 MiB, against the
        # 27B's 2.5 GB), and a ranked head costs it acceptance, most right after startup
        dsp.draft_vocab = DRAFT_VOCAB if sp.num_speculative_tokens and vocab >= 4 * DRAFT_VOCAB and not assistant else 0
        reasons["dev.speculative.draft_vocab"] = (
            f"{DRAFT_VOCAB:,} of {vocab:,} tokens, ranked from the tokens this server emits (the drafter's head read "
            f"drops {100 * (1 - DRAFT_VOCAB / vocab):.0f}%; drafts change acceptance, outputs at most by rounding)"
            if dsp.draft_vocab else
            "0: the whole vocabulary (the assistant's own narrow head; a ranked one costs acceptance on Gemma 4)"
            if assistant and sp.num_speculative_tokens else
            "0: the whole vocabulary (no MTP, or a vocabulary under 4 x 32,768)")
    if dsp.draft_head_dtype == "auto":
        dsp.draft_head_dtype = "bfloat16"
        reasons["dev.speculative.draft_head_dtype"] = "the target head's dtype (an FP8 copy costs acceptance)"
    # A set refine piece, before the engine loads any weight (the scheduler checks its grid too)
    piece = cfg.dev.cache.refine_chunk_tokens
    if piece is not None and not 0 < piece <= sc.max_num_batched_tokens:
        raise ValueError(f"dev.cache.refine_chunk_tokens {piece}: positive, at most scheduler.max_num_batched_tokens "
                         f"({sc.max_num_batched_tokens}); unset, it is derived")
    room = sc.max_num_batched_tokens - sc.max_num_seqs * (1 + sp.num_speculative_tokens)
    if piece is not None and piece > room:
        raise ValueError(f"dev.cache.refine_chunk_tokens {piece}: at most {room} (max_num_batched_tokens - "
                         f"max_num_seqs x {1 + sp.num_speculative_tokens}: a piece must fit next to a full batch of "
                         "decode rows)")
    # A derived piece is at least one checkpoint grid (`sched.scheduler.derive_refine_chunk_tokens`), for a runner with
    # PRECISE rows (a block-FP8 checkpoint's; the fake's when told) under fp8_cold_prefill reuse: where none fits, the
    # scheduler refused it at startup, naming the piece as if it were set
    gemma = hf.get("model_type") in GEMMA4_TYPES or text.get("model_type") in GEMMA4_TYPES
    precise = cfg.dev.runner.fake.precise_rows if cfg.runner.kind == "fake" else bool(hf.get("quantization_config"))
    grid = 16 if gemma and cfg.runner.kind == "cuda" else 64  # the runners' checkpoint_grid
    if piece is None and precise and cfg.runner.cuda.fp8_cold_prefill == "reuse" and room < grid:
        raise ValueError(f"scheduler.max_num_batched_tokens {sc.max_num_batched_tokens} leaves {room} tokens next to "
                         f"max_num_seqs {sc.max_num_seqs} decode rows x {1 + sp.num_speculative_tokens}, under one "
                         f"{grid}-token piece of the precise prefix's refine: raise it, lower scheduler.max_num_seqs, "
                         "or set runner.cuda.fp8_cold_prefill bf16 or fp8")

    cfg.cpu_budget, why = cgroup_cpu_quota()
    reasons["cpu_budget"] = why
    if cfg.server.num_frontends is None:
        cfg.server.num_frontends = 1
        reasons["server.num_frontends"] = (f"1 for one GPU; CPU budget {cfg.cpu_budget:g} "
                                           f"(each frontend takes a core of it)")
    linux = sys.platform == "linux" and hasattr(os, "sched_getaffinity")
    ds, de = cfg.dev.server, cfg.dev.engine
    if de.cpu is None and linux:
        de.cpu = max(os.sched_getaffinity(0))
        reasons["dev.engine.cpu"] = "the last CPU in the affinity mask"
    if ds.cpus is None and linux and de.cpu is not None:
        allowed = sorted(c for c in os.sched_getaffinity(0) if c < de.cpu)
        if len(allowed) >= cfg.server.num_frontends:
            ds.cpus = allowed[-cfg.server.num_frontends:][::-1]
            reasons["dev.server.cpus"] = "the CPUs just below the engine's"
    # A warm core on the TTFT path: on a Linux VM an idle core takes longer over a request's CPU work (clocking down,
    # waking the vCPU). Spinning costs the frontend's and the engine's core while there is traffic, and nothing when
    # there is none.
    for group, key in ((ds, "spin_ms"), (de, "idle_spin_ms")):
        if getattr(group, key) is None:
            setattr(group, key, 20.0 if linux else 0.0)
            reasons[f"dev.{'server' if group is ds else 'engine'}.{key}"] = (
                "20 ms on Linux: a warm core on the TTFT path" if linux else "off: no pinning here")
    _derive_images(cfg, model_dir, reasons)
    _derive_structured(cfg, model_dir, reasons)
    _derive_tool_calls(cfg, hf, reasons)
    _derive_reasoning(cfg, hf, reasons)
    _derive_heads(cfg, hf, assistant, reasons)
    _derive_grammar_tokens(cfg, reasons)
    _derive_kernels(cfg, hf, reasons)
    cfg.reasons = reasons
    return cfg


def _check_assistant(cfg: Config, hf: dict[str, Any]) -> bool:
    """Whether `speculative.model` names a drafter for this target: a Gemma 4 assistant for a Gemma 4 checkpoint.
    Raises for any other pairing; the runner checks shapes (`models.gemma4.assistant.AssistantConfig.check_target`)."""
    path = cfg.speculative.model
    if not path:
        return False
    d = Path(path)
    if not (d / "config.json").is_file():
        raise ValueError(f"speculative.model {path} has no config.json (a local checkpoint or a Hugging Face repo id)")
    kind = _read_json(d / "config.json").get("model_type")
    target = hf.get("model_type") or (hf.get("text_config") or {}).get("model_type")
    if kind not in ("gemma4_assistant", "gemma4_unified_assistant") or target not in ("gemma4", "gemma4_text"):
        raise ValueError(f"speculative.model is a {kind!r} model for a {target!r} checkpoint: only Gemma 4's "
                         "assistant (gemma4_assistant, gemma4_unified_assistant) is served as a separate drafter")
    return True


GRAMMAR_THREADS = 1  # clones of one compiled grammar share its caches: more threads take longer (grammar.engine)


def tokenizer_json(cfg: Config) -> Path:
    return Path(cfg.model.tokenizer or cfg.model.path) / "tokenizer.json"


def tokenizer_eos_token(cfg: Config) -> str | None:
    """The tokenizer's EOS token by name: what transformers makes `eos_token_id` of, and so the EOS of the guidance
    backend (`llguidance.hf.from_tokenizer`), read without loading the tokenizer. transformers 5's rule:
    tokenizer_config.json's `eos_token`, unless special_tokens_map.json names one and tokenizer_config.json has no
    `added_tokens_decoder` (its legacy path); a string or an AddedToken's `content`. None if neither names one."""
    d = Path(cfg.model.tokenizer or cfg.model.path)
    tc = _read_json(d / "tokenizer_config.json")
    eos = tc.get("eos_token")
    if "added_tokens_decoder" not in tc:
        eos = _read_json(d / "special_tokens_map.json").get("eos_token", eos)
    if isinstance(eos, dict):
        eos = eos.get("content")
    return eos if isinstance(eos, str) and eos else None


QWEN3_5_TYPES = ("qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text")  # Qwen3.5 and Qwen3.8


def _derive_tool_calls(cfg: Config, hf: dict[str, Any], reasons: dict[str, str]) -> None:
    """tool_calls.parser's default: the family's format, when the chat template renders its calls. One that is set
    must fit the tokenizer (`output.tool_calls.missing_tokens`): refused as a bad value is, before anything starts."""
    tc = cfg.tool_calls
    if tc.parser not in (None, "none") and tokenizer_json(cfg).exists():
        from tokenizers import Tokenizer

        from .output.tool_calls import missing_tokens

        vocab = Tokenizer.from_file(str(tokenizer_json(cfg))).get_vocab(with_added_tokens=True)
        missing = missing_tokens(tc.parser, vocab)
        if missing:
            raise ValueError(f"invalid configuration: tool_calls.parser {tc.parser!r} does not fit the tokenizer, "
                             f"which has no {', '.join(missing)} - at `$.tool_calls.parser`")
    if tc.parser is not None:
        return
    from .output.tool_calls import template_format
    from .render.renderer import load_chat_templates

    types = {hf.get("model_type"), (hf.get("text_config") or {}).get("model_type")}
    family = ("gemma4", "Gemma 4") if types & set(GEMMA4_TYPES) else ("qwen3_coder", "Qwen3.5/3.8") \
        if types & set(QWEN3_5_TYPES) else (None, "")
    tok_dir = Path(cfg.model.tokenizer or cfg.model.path)
    templates = load_chat_templates(tok_dir, _read_json(tok_dir / "tokenizer_config.json"))
    text = "\n".join(templates.values()) if isinstance(templates, dict) else templates or ""
    if family[0] is None:
        tc.parser = "none"
        reasons["tool_calls.parser"] = "none: the model is neither Gemma 4 nor Qwen3.5/3.8"
    elif template_format(text) != family[0]:
        tc.parser = "none"
        reasons["tool_calls.parser"] = f"none: the chat template does not render {family[1]}'s tool calls"
    else:
        tc.parser = family[0]
        reasons["tool_calls.parser"] = f"{family[0]}: {family[1]}, whose chat template renders its tool calls"


def _derive_reasoning(cfg: Config, hf: dict[str, Any], reasons: dict[str, str]) -> None:
    """reasoning.parser's default: the family's, when the chat template renders its markers; one that is set must fit
    the tokenizer (`output.reasoning.missing_tokens`). reasoning.effort_map: checked as vLLM checks
    --reasoning-effort-map (the server stops with its message), else the model family's."""
    from .output.reasoning import EFFORT_MAPS, check_effort_map, missing_tokens, template_format

    rc = cfg.reasoning
    types = {hf.get("model_type"), (hf.get("text_config") or {}).get("model_type")}
    family = ("gemma4", "Gemma 4") if types & set(GEMMA4_TYPES) else ("qwen3", "Qwen3.5/3.8") \
        if types & set(QWEN3_5_TYPES) else (None, "")
    if rc.effort_map is None:
        rc.effort_map = dict(EFFORT_MAPS.get(family[0], {}))
        reasons["reasoning.effort_map"] = (f"{family[1]}'s: its chat template's values" if family[0] else
                                           "{}: the efforts pass as they are")
    else:
        check_effort_map(rc.effort_map)
    if rc.parser not in (None, "none") and tokenizer_json(cfg).exists():
        from tokenizers import Tokenizer

        vocab = Tokenizer.from_file(str(tokenizer_json(cfg))).get_vocab(with_added_tokens=True)
        missing = missing_tokens(rc.parser, vocab)
        if missing:
            raise ValueError(f"invalid configuration: reasoning.parser {rc.parser!r} does not fit the tokenizer, "
                             f"which has no {', '.join(missing)} - at `$.reasoning.parser`")
    if rc.parser is not None:
        return
    from .render.renderer import load_chat_templates

    tok_dir = Path(cfg.model.tokenizer or cfg.model.path)
    templates = load_chat_templates(tok_dir, _read_json(tok_dir / "tokenizer_config.json"))
    text = "\n".join(templates.values()) if isinstance(templates, dict) else templates or ""
    if family[0] is None:
        rc.parser = "none"
        reasons["reasoning.parser"] = "none: the model is neither Gemma 4 nor Qwen3.5/3.8"
    elif template_format(text) != family[0]:
        rc.parser = "none"
        reasons["reasoning.parser"] = f"none: the chat template does not render {family[1]}'s thought markers"
    else:
        rc.parser = family[0]
        reasons["reasoning.parser"] = f"{family[0]}: {family[1]}, whose chat template renders its thought markers"


def _derive_structured(cfg: Config, model_dir: Path, reasons: dict[str, str]) -> None:
    so = cfg.structured_output
    if so.enabled is None:
        try:
            import llguidance  # noqa: F401
        except ImportError:
            so.enabled = False
            reasons["structured_output.enabled"] = "off: llguidance is not installed"
        else:
            so.enabled = tokenizer_json(cfg).exists()
            reasons["structured_output.enabled"] = ("llguidance and the checkpoint's tokenizer.json" if so.enabled else
                                                    "off: no tokenizer.json")
    if not 0 < cfg.dev.structured_output.compile_timeout_s < math.inf:  # NaN too: the scheduler counts in ns
        raise ValueError(f"dev.structured_output.compile_timeout_s {cfg.dev.structured_output.compile_timeout_s}: "
                         "more than 0, and finite")
    if cfg.dev.structured_output.threads is None:
        cfg.dev.structured_output.threads = GRAMMAR_THREADS
        reasons["dev.structured_output.threads"] = (f"{GRAMMAR_THREADS}: masks of requests sharing a grammar do not "
                                                    "get faster on more threads (llguidance's shared caches)")
    if so.disable_any_whitespace and so.backend != "guidance":
        raise ValueError(f"structured_output.disable_any_whitespace needs structured_output.backend guidance "
                         f"(xgrammar's compact JSON is not reproduced), not {so.backend}")
    if so.backend == "guidance":
        eos = tokenizer_eos_token(cfg)
        if so.enabled and eos is None:
            raise ValueError("structured_output.backend guidance ends a grammar on the tokenizer's eos_token, and the "
                             "tokenizer (tokenizer_config.json, special_tokens_map.json) names no eos_token")
        reasons["structured_output.backend"] = (
            f"GuidanceBackend: llguidance's JSON grammar with canonical forcing, "
            f"{'compact' if so.disable_any_whitespace else 'any whitespace'}; " +
            (f"the tokenizer's eos_token {eos} ends a grammar" if eos else "the tokenizer names no eos_token"))
    else:
        reasons["structured_output.backend"] = (
            ("the default" if so.backend == "auto" else "as auto") +
            ": xgrammar's JSON language through llguidance (any whitespace, no forcing); the model's stop ids end a "
            "grammar")


SPARSE_HEAD_ROWS = 1024  # the grammar-aware heads' rows (architecture.md, "Grammar-aware heads")
GEMMA4_TYPES = ("gemma4", "gemma4_text")  # models.gemma4.model.MODEL_TYPES (this module imports no torch)


def _derive_heads(cfg: Config, hf: dict[str, Any], assistant: bool, reasons: dict[str, str]) -> None:
    """The grammar-aware heads' defaults (architecture.md, "Grammar-aware heads"): the sparse head with structured
    output, the draft universe with drafts too (Gemma 4's assistant, the checkpoint's MTP layer)."""
    o, sp, so, dsp = cfg.dev.runner.cuda, cfg.speculative, cfg.structured_output, cfg.dev.speculative
    gemma = hf.get("model_type") in GEMMA4_TYPES or (hf.get("text_config") or {}).get("model_type") in GEMMA4_TYPES
    drafts = bool(sp.num_speculative_tokens)  # from an assistant or the checkpoint's MTP layer (checked in `derive`)
    if o.sparse_head_rows is None:
        o.sparse_head_rows = SPARSE_HEAD_ROWS
        v = cfg.vocab_size
        reasons["dev.runner.cuda.sparse_head_rows"] = (
            (f"{SPARSE_HEAD_ROWS:,}: {100 * SPARSE_HEAD_ROWS / v:.2f}% of the {v:,}-row head" if v else
             f"{SPARSE_HEAD_ROWS:,}") + "; a step whose union is larger takes the dense head")
    elif not 0 < o.sparse_head_rows <= 65_536:
        raise ValueError(f"dev.runner.cuda.sparse_head_rows {o.sparse_head_rows}: 1-65,536 tokens")
    if o.sparse_head is None:
        o.sparse_head = bool(so.enabled)
        reasons["dev.runner.cuda.sparse_head"] = (
            "on: structured output; a step whose every sample token is masked scores the union of its masks"
            if o.sparse_head else "off: structured output is off")
    if dsp.draft_universe is None:
        dsp.draft_universe = drafts and bool(so.enabled)
        reasons["dev.speculative.draft_universe"] = (
            f"on: structured output and drafts ({'the assistant' if assistant else 'the MTP layer'}); on the steps "
            "the sparse head could serve, the drafter scores only the grammar's universe" if dsp.draft_universe else
            "off: structured output is off" if drafts else "off: no drafts (speculative.num_speculative_tokens 0)")
    elif dsp.draft_universe and not drafts:
        raise ValueError("dev.speculative.draft_universe: needs drafts (speculative.num_speculative_tokens > 0: Gemma "
                         "4's assistant, speculative.model, or the checkpoint's MTP layer)")
    if dsp.draft_query_next and not (gemma and assistant):
        raise ValueError("dev.speculative.draft_query_next: Gemma 4's assistant only (speculative.model)")


def _derive_grammar_tokens(cfg: Config, reasons: dict[str, str]) -> None:
    """The forced prefix's and the EOS-free finish's defaults (their comments in `DevStructuredOutputConfig`;
    architecture.md, "Grammar-aware heads"), alike for every model (both are the scheduler's)."""
    so, dso = cfg.structured_output, cfg.dev.structured_output
    if dso.forced_prefix is None:
        dso.forced_prefix = bool(so.enabled)
        reasons["dev.structured_output.forced_prefix"] = (
            "on: structured output; a grammar's forced first tokens go into the final prefill chunk" +
            ("" if so.backend == "guidance" else f" (backend {so.backend} forces none: no canonical forcing)")
            if dso.forced_prefix else "off: structured output is off")
    ids = sorted(set(cfg.eos_token_ids))
    one = bool(ids) and (so.backend == "guidance" or len(ids) == 1)
    if dso.eos_free_finish is None:
        dso.eos_free_finish = bool(so.enabled) and one
        reasons["dev.structured_output.eos_free_finish"] = (
            "on: structured output, its grammars ending on one EOS id; a request whose grammar stopped finishes with "
            "that EOS appended" if dso.eos_free_finish else
            "off: structured output is off" if not so.enabled else
            "off: the checkpoint names no EOS id" if not ids else
            f"off: backend {so.backend}'s grammars end on {len(ids)} EOS ids: the sample decides which")
    elif dso.eos_free_finish and not ids:
        raise ValueError("dev.structured_output.eos_free_finish: the checkpoint names no EOS id (eos_token_ids is "
                         "empty)")
    elif dso.eos_free_finish and not one:
        raise ValueError(f"dev.structured_output.eos_free_finish: backend {so.backend}'s grammars end on several EOS "
                         f"ids ({ids}), so the sample decides which: use backend guidance")


# dev.runner.cuda.batch_invariant's default context: kernels.attn512's fixed split keeps 256 keys a split up to the
# merge's 32 splits (`fixed_split_keys`)
BATCH_INVARIANT_MAX_LEN = 32 * 256


# Our kernels read and write BF16 (the Qwen runner's take any tensor they are given): fp32 on CUDA takes their torch
# references. Refused at load, and by both runners for a caller that builds one itself
FP32_KERNELS = "dev.runner.cuda.dtype float32 on CUDA needs dev.runner.cuda.fused_ops false (our kernels are BF16)"


def _derive_kernels(cfg: Config, hf: dict[str, Any], reasons: dict[str, str]) -> None:
    """The kernel and graph defaults (architecture.md, "CUDA graphs", "Batch invariance"): programmatic dependent
    launch, for both runners; always-split graphs, with structured output; the batch-invariant kernels
    (`_derive_batch_invariance`). On CUDA a set `engine.plan_at_boundary` is refused on the Qwen runner or without
    structured output, and fp32 with our kernels (`FP32_KERNELS`)."""
    o = cfg.dev.runner.cuda
    if cfg.runner.kind == "cuda" and o.device.split(":")[0] == "cuda" and o.dtype == "float32" and o.fused_ops:
        raise ValueError(FP32_KERNELS)
    gemma = hf.get("model_type") in GEMMA4_TYPES or (hf.get("text_config") or {}).get("model_type") in GEMMA4_TYPES
    short = (cfg.model.max_model_len or 0) + 8 <= BATCH_INVARIANT_MAX_LEN
    if o.pdl is None:
        o.pdl = True
        reasons["dev.runner.cuda.pdl"] = "on: each kernel of ours and each FP8 GEMM may start as the one before it ends"
    so = cfg.structured_output
    if o.always_split is None:
        o.always_split = bool(so.enabled)
        reasons["dev.runner.cuda.always_split"] = (
            "on: structured output; every step runs as a forward graph and a sampling graph, and the fused graphs are "
            "not captured" if o.always_split else "off: structured output is off")
    elif o.always_split and not so.enabled:
        raise ValueError("dev.runner.cuda.always_split: needs structured output (its graphs are the grammar masks' "
                         "split)")
    _derive_batch_invariance(cfg, hf, gemma, short, reasons)
    if cfg.engine.plan_at_boundary and cfg.runner.kind == "cuda" and not gemma:
        raise ValueError("engine.plan_at_boundary: Gemma 4 only on CUDA (the Qwen runner's prefill and mixed steps "
                         "launch breakable graphs whose eager GDN and attention breaks take 15-22 ms of host time on a "
                         "27B model, which the step in flight's outputs would wait for)")
    if cfg.engine.plan_at_boundary and cfg.runner.kind == "cuda" and not so.enabled:
        raise ValueError("engine.plan_at_boundary: needs structured output on CUDA (only its steps split into a "
                         "forward and a sampling)")


def _derive_batch_invariance(cfg: Config, hf: dict[str, Any], gemma: bool, short: bool,
                             reasons: dict[str, str]) -> None:
    """The default of `dev.runner.cuda.batch_invariant` (architecture.md, "Batch invariance"; the rule: its comment in
    `DevCudaConfig`). Raises when it is set on and `_invariance_blocked` gives a reason."""
    o = cfg.dev.runner.cuda
    long = (f"off: model.max_model_len {cfg.model.max_model_len} is past {BATCH_INVARIANT_MAX_LEN - 8:,}, where a "
            "decode row's keys may outgrow the full layers' fixed split (at most 32 splits of 256 keys: "
            f"{BATCH_INVARIANT_MAX_LEN:,}); forced on, a split holds about a 32nd of the context instead, so a "
            "decode row runs fewer, longer splits (one up to that many keys), and a prefill row one however long its "
            "history; set it to take that")
    from .sched.plan import CKPT_GRID

    # The Qwen runner cuts its prompts on the GDN prefill's 64-token grid too; its MTP layer writes KV of its own, so
    # its drafter's chain comes out of the forced prefix's room, where Gemma 4's assistant (no KV) leaves it all
    cant = _invariance_blocked(cfg, 1 if gemma else CKPT_GRID, draft_kv=not gemma)
    if gemma:
        if o.batch_invariant is None:
            o.batch_invariant = short and not cant
            reasons["dev.runner.cuda.batch_invariant"] = (
                "on: Gemma 4; a row's attention, sparse head and image bits do not depend on the rows beside it (nor "
                "its FP8 projections': CUTLASS at every M)" if o.batch_invariant else long if not short else
                f"off: {cant}")
        elif o.batch_invariant and cant:
            raise ValueError(f"dev.runner.cuda.batch_invariant: {cant}")
        return
    fp8 = bool(hf.get("quantization_config"))
    if o.batch_invariant is None:
        o.batch_invariant = short and fp8 and not cant
        reasons["dev.runner.cuda.batch_invariant"] = (
            "on: Qwen3.8 with a block-FP8 checkpoint; a row's attention, GDN prefill, in_proj_ba, sparse head and "
            "image bits do not depend on the rows beside it, and a prompt's chunks are cut on the GDN prefill's "
            "64-token grid (nor its FP8 projections': CUTLASS at every M)" if o.batch_invariant else
            long if not short else
            "off: a BF16 checkpoint, whose projections run at the step's M (cuBLAS picks its kernel by M); set it for "
            "the rest" if not fp8 else f"off: {cant}")
    elif o.batch_invariant and cant:
        raise ValueError(f"dev.runner.cuda.batch_invariant: {cant}")


def _invariance_blocked(cfg: Config, grid: int, draft_kv: bool) -> str:
    """Why batch invariance cannot run, or "": no room for a prefill's `grid` tokens and its forced prefix next to
    max_num_seqs − 1 decode rows, or a chunk cap below the grid (the scheduler's `_check_progress`), or a page size
    that is not a multiple of our attention kernels' 16-key KV tile (`kernels.attn_window`)."""
    from .sched.plan import reserve_positions

    sc, k = cfg.scheduler, cfg.speculative.num_speculative_tokens or 0
    chain = max(0, k - 1) if draft_kv else 0
    forced = reserve_positions(2, k) - chain if cfg.dev.structured_output.forced_prefix else 0
    need = grid + forced + (sc.max_num_seqs - 1) * (1 + k)
    if sc.max_num_batched_tokens < need:
        return (f"scheduler.max_num_batched_tokens {sc.max_num_batched_tokens} is below {need}: "
                f"{f'a prefill cut on the {grid}-token grid' if grid > 1 else 'a prefill token'}"
                f"{', with its forced prefix,' if forced else ''} next to max_num_seqs - 1 = {sc.max_num_seqs - 1} "
                f"decode rows of {1 + k} tokens")
    cap = cfg.dev.scheduler.long_prefill_token_threshold
    if grid > 1 and 0 < cap < grid:
        return (f"dev.scheduler.long_prefill_token_threshold {cap} is below the {grid}-token grid every chunk is "
                "cut on")
    if (cfg.dev.cache.page_size or 16) % 16:
        return (f"dev.cache.page_size {cfg.dev.cache.page_size} is not a multiple of 16 (our attention kernels' KV "
                "tile)")
    return ""


POOL_WORKERS_MAX = 8  # a worker decodes and resizes a typical image in milliseconds


def _derive_images(cfg: Config, model_dir: Path, reasons: dict[str, str]) -> None:
    """The `images` group's defaults (architecture.md, "The image pool"): from the checkpoint's image family, the CPU
    quota and the scheduler's budget."""
    from .vision import family as vf
    from .vision.probe import MAX_SOURCE_PIXELS

    im = cfg.images
    fam = None
    if im.enabled is not False:
        try:
            fam = vf.load(model_dir, min_pixels=im.min_pixels, max_pixels=im.max_pixels,
                          max_soft_tokens=im.max_soft_tokens)
        except (ValueError, KeyError, OSError) as exc:
            raise ValueError(f"images: {exc}") from exc
    if im.enabled is None:
        im.enabled = fam is not None
        reasons["images.enabled"] = (f"the checkpoint's {fam.name} vision tower" if fam is not None
                                     else "the checkpoint has no vision tower this server runs")
    elif im.enabled and fam is None:
        raise ValueError("images.enabled: the checkpoint has no vision tower this server runs")
    if not im.enabled:
        cfg.image_tokens_max = 0
        return
    if not 0 < im.max_source_pixels <= MAX_SOURCE_PIXELS:
        raise ValueError(f"images.max_source_pixels must be 1..{MAX_SOURCE_PIXELS} (Pillow's decompression-bomb bound, "
                         f"which the image pool's Pillow path enforces anyway), got {im.max_source_pixels}")
    for key in ("max_request_decode_bytes", "max_request_download_mb"):
        if getattr(im, key) < 0:
            raise ValueError(f"images.{key} must be 0 (no limit) or more, got {getattr(im, key)}")
    from .vision.urlguard import parse_domains, parse_networks

    parse_networks(im.allowed_networks)  # a ValueError names the bad entry
    parse_domains(im.allowed_domains)
    if fam.max_soft_tokens is not None:  # Gemma 4: sized by its soft-token budget; every later load site passes it
        if im.max_soft_tokens is None:
            im.max_soft_tokens = fam.max_soft_tokens
            reasons["images.max_soft_tokens"] = fam.max_soft_tokens_source
        side = fam.pooling * fam.patch
        pixel_reasons = (f"one soft token, {side} x {side} pixels: Gemma 4 sizes images by images.max_soft_tokens",
                         f"images.max_soft_tokens {fam.max_soft_tokens} x {fam.pooling}^2 patches of {fam.patch}^2 "
                         "pixels")
    else:
        reasons["images.max_soft_tokens"] = "unused: this family sizes images by images.min_pixels / max_pixels"
        pixel_reasons = ("preprocessor_config.json size.shortest_edge", "preprocessor_config.json size.longest_edge")
    if im.min_pixels is None:
        im.min_pixels = fam.min_pixels
        reasons["images.min_pixels"] = pixel_reasons[0]
    if im.max_pixels is None:
        im.max_pixels = fam.max_pixels
        reasons["images.max_pixels"] = pixel_reasons[1]
    cfg.image_tokens_max = fam.max_tokens()
    one = 3 * im.max_pixels  # the largest image's uint8 pixels
    dim = cfg.dev.images
    if not 0 <= dim.span_key_bits <= 62:
        raise ValueError("dev.images.span_key_bits must be 0-62 (the image digest's bits in a span's radix key)")
    if dim.arena_mb is None:
        dim.arena_mb = max(256, -(-4 * one // (1 << 20)))
        reasons["dev.images.arena_mb"] = (f"max(256 MiB, 4 images of max_pixels): pixels stay until their request "
                                          f"leaves the engine ({one / 2**20:.1f} MiB per image at most)")
    sc = cfg.scheduler
    largest = min(cfg.image_tokens_max, cfg.model.max_model_len or cfg.image_tokens_max)
    if dim.encoder_budget is None:
        dim.encoder_budget = max(sc.max_num_batched_tokens, largest)
        n = dim.encoder_budget // largest
        reasons["dev.images.encoder_budget"] = (f"max(max_num_batched_tokens, the largest image a prompt can hold: "
                                            f"{largest} tokens); {n} image"
                                            f"{'' if n == 1 else 's'} of {largest} tokens per step")
    elif dim.encoder_budget < largest:
        raise ValueError(f"dev.images.encoder_budget {dim.encoder_budget} < the largest image ({largest} tokens at "
                         f"max_pixels {im.max_pixels}): such an image could never be encoded")
    if dim.encoder_cache_tokens is None:
        dim.encoder_cache_tokens = 2 * dim.encoder_budget
        reasons["dev.images.encoder_cache_tokens"] = ("2 x encoder_budget: a step's images plus the spans still "
                                                      "prefilling")
    elif dim.encoder_cache_tokens < dim.encoder_budget:
        raise ValueError("dev.images.encoder_cache_tokens must be at least dev.images.encoder_budget")
    if im.workers is None:
        free = int(cfg.cpu_budget) - 1 - (cfg.server.num_frontends or 1)
        im.workers = max(1, min(POOL_WORKERS_MAX, free))
        reasons["images.workers"] = (f"the CPU budget {cfg.cpu_budget:g} minus the engine's core and "
                                     f"{cfg.server.num_frontends or 1} frontend core(s), at most {POOL_WORKERS_MAX}: "
                                     "one process and one thread each")


def effective_lines(cfg: Config) -> list[str]:
    """`name = value  (reason)` for every setting, then every dev setting and the server's state, for the startup
    log."""
    out = []
    for name in leaves(Config):
        if name != "reasons":
            value = cfg
            for part in name.split("."):
                value = getattr(value, part)
            out.append(_line(name, msgspec.to_builtins(value), cfg.reasons))
    return out


def _line(name: str, value: Any, reasons: dict[str, str]) -> str:
    why = reasons.get(name) or reasons.get(name.split(".")[0] if name.startswith("sampling.") else "")
    if isinstance(value, float) and math.isfinite(value):
        value = f"{value:g}"
    return f"{name} = {value}" + (f"  ({why})" if why else "")
