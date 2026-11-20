# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Metric names, help texts and histogram buckets are vLLM v0.30.0's (vllm/v1/metrics/,
# https://github.com/vllm-project/vllm), for compatibility with dashboards and routers built for vLLM.
"""Prometheus metrics in the text exposition format, written here (no prometheus_client; docs/architecture.md,
"Observability"; the metrics to watch: docs/operations.md, "Metrics"; each metric's meaning: its help text here).

Names and buckets match vLLM's (`v1/metrics/loggers.py`, `v1/metrics/buckets.py`), with the labels
`model_name` and `engine`, so dashboards and routers built for vLLM read them unchanged; and `http_requests_total`,
the counter vLLM's API server takes from prometheus-fastapi-instrumentator, labelled as there. A counter is exposed as
prometheus_client exposes vLLM's: its name with `_total` (`vllm:request_success` is `vllm:request_success_total`,
which those dashboards query), unless the name already ends so.

Request-level metrics are kept by each frontend process, and so are the per-step histograms it observes itself
(`vllm:iteration_tokens_total`, `llm_mini_infer:engine_step_*`, the `vit` image stage), for the steps that sent it
outputs (a step that sends none is in no frontend's). With more than one frontend, each scrape sees one frontend's
share of them; the engine's gauges, counters and accumulated histograms (`llm_mini_infer:step_decode_rows`, ...) are
the same everywhere.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Sequence

from .engine.messages import POPCOUNT_BUCKETS
from .runner.base import UNION_BUCKETS

REQUEST_LATENCY = (0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 120.0, 240.0,
                   480.0, 960.0, 1920.0, 7680.0)
TTFT = (0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0,
        160.0, 640.0, 2560.0)
ITL = (0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0)
ITERATION_TOKENS = (1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)
OVERHEAD = (0.00001, 0.000025, 0.00005, 0.0001, 0.00025, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25,
            1.0)
STEP = (0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 1.0)
STEP_ROWS = (*range(17), 24, 32, 48, 64, 96, 128, 256)
STEP_IMAGES = (*range(9), 16, 32, 64)


def request_token_buckets(max_model_len: int) -> tuple[float, ...]:
    """The 1-2-5 series (1, 2, 5, 10, ...) up to max_model_len, as the buckets of `vllm:request_*_tokens`."""
    out, mantissa, exp = [], (1, 2, 5), 1
    while True:
        for m in mantissa:
            v = m * exp
            if v > max_model_len:
                return tuple(out)
            out.append(v)
        exp *= 10


def _esc(v: str) -> str:
    return v.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _fmt(v: float) -> str:
    if v == float("inf"):
        return "+Inf"
    return repr(float(v)) if isinstance(v, float) else str(v)


class _Metric:
    kind = ""

    def __init__(self, name: str, doc: str, base: str):
        self.name = name
        self.doc = doc
        self.base = base  # rendered label set, e.g. 'model_name="m",engine="0"'

    def header(self) -> list[str]:
        return [f"# HELP {self.name} {self.doc}", f"# TYPE {self.name} {self.kind}"]


class Counter(_Metric):
    kind = "counter"

    def __init__(self, name: str, doc: str, base: str, labels: Sequence[str] = ()):
        super().__init__(name if name.endswith("_total") else name + "_total", doc, base)
        self.values: dict[tuple[str, ...], float] = {} if labels else {(): 0.0}
        self.label_names = tuple(labels)

    def inc(self, v: float = 1.0, *labels: str) -> None:
        self.values[labels] = self.values.get(labels, 0.0) + v

    def set_total(self, v: float, *labels: str) -> None:
        """For counters the engine accumulates: export its running total."""
        self.values[labels] = float(v)

    def render(self) -> list[str]:
        out = self.header()
        for labels, v in self.values.items():
            extra = ",".join(f'{k}="{_esc(x)}"' for k, x in zip(self.label_names, labels))
            inner = f"{self.base},{extra}" if self.base and extra else self.base or extra
            out.append(f"{self.name}{{{inner}}} {_fmt(v)}")
        return out


class Gauge(_Metric):
    kind = "gauge"

    def __init__(self, name: str, doc: str, base: str):
        super().__init__(name, doc, base)
        self.value = 0.0

    def set(self, v: float) -> None:
        self.value = v

    def render(self) -> list[str]:
        return [*self.header(), f"{self.name}{{{self.base}}} {_fmt(float(self.value))}"]


class Histogram(_Metric):
    kind = "histogram"

    def __init__(self, name: str, doc: str, base: str, buckets: Sequence[float], labels: Sequence[str] = ()):
        super().__init__(name, doc, base)
        self.buckets = tuple(buckets)
        self.label_names = tuple(labels)
        self.series: dict[tuple[str, ...], list] = {}  # labels -> [counts per bucket + inf, sum, count]

    def observe(self, v: float, *labels: str) -> None:
        s = self.series.get(labels)
        if s is None:
            s = self.series[labels] = [[0] * (len(self.buckets) + 1), 0.0, 0]
        s[0][bisect_left(self.buckets, v)] += 1
        s[1] += v
        s[2] += 1

    def set_cumulative(self, counts: Sequence[int], total: float, *labels: str) -> None:
        """For histograms the engine accumulates: its counts per bucket (the last one +Inf) and sum."""
        self.series[labels] = [list(counts), float(total), sum(counts)]

    def render(self) -> list[str]:
        out = self.header()
        for labels, (counts, total, n) in self.series.items():
            extra = "".join(f',{k}="{_esc(x)}"' for k, x in zip(self.label_names, labels))
            acc = 0
            for b, c in zip((*self.buckets, float("inf")), counts):
                acc += c
                out.append(f'{self.name}_bucket{{{self.base}{extra},le="{_fmt(b)}"}} {acc}')
            out.append(f"{self.name}_sum{{{self.base}{extra}}} {_fmt(total)}")
            out.append(f"{self.name}_count{{{self.base}{extra}}} {n}")
        return out


class Metrics:
    """All metrics one frontend process exports."""


    def __init__(self, model_name: str, max_model_len: int, engine: str = "0"):
        base = f'model_name="{_esc(model_name)}",engine="{engine}"'
        self._model_label = f'model_name="{_esc(model_name)}"'  # the parser metric's: no engine label
        self.tool_parser: Counter | None = None  # `enable_tool_parser`
        tok = request_token_buckets(max_model_len)
        self.running = Gauge("vllm:num_requests_running", "Number of requests in model execution batches.", base)
        self.waiting = Gauge("vllm:num_requests_waiting", "Number of requests waiting to be processed.", base)
        self.kv_usage = Gauge("vllm:kv_cache_usage_perc", "KV-cache usage. 1 means 100 percent usage.", base)
        self.prefix_queries = Counter("vllm:prefix_cache_queries",
                                      "Prefix cache queries, in terms of number of queried tokens.", base)
        self.prefix_hits = Counter("vllm:prefix_cache_hits",
                                   "Prefix cache hits, in terms of number of cached tokens.", base)
        self.preemptions = Counter("vllm:num_preemptions", "Cumulative number of preemption from the engine.", base)
        self.prompt_tokens = Counter("vllm:prompt_tokens", "Number of prefill tokens processed.", base)
        self.generation_tokens = Counter("vllm:generation_tokens", "Number of generation tokens processed.", base)
        self.request_success = Counter("vllm:request_success", "Count of successfully processed requests.", base,
                                       labels=("finished_reason",))
        self.request_prompt_tokens = Histogram("vllm:request_prompt_tokens", "Number of prefill tokens processed.",
                                               base, tok)
        self.request_generation_tokens = Histogram("vllm:request_generation_tokens",
                                                   "Number of generation tokens processed.", base, tok)
        self.request_max_tokens = Histogram("vllm:request_params_max_tokens",
                                            "Histogram of the max_tokens request parameter.", base, tok)
        self.iteration_tokens = Histogram("vllm:iteration_tokens_total",
                                          "Histogram of number of tokens per engine_step.", base, ITERATION_TOKENS)
        self.ttft = Histogram("vllm:time_to_first_token_seconds", "Histogram of time to first token in seconds.",
                              base, TTFT)
        self.itl = Histogram("vllm:inter_token_latency_seconds", "Histogram of inter-token latency in seconds.",
                             base, ITL)
        self.tpot = Histogram("vllm:request_time_per_output_token_seconds",
                              "Histogram of time_per_output_token_seconds per request.", base, ITL)
        self.e2e = Histogram("vllm:e2e_request_latency_seconds", "Histogram of e2e request latency in seconds.",
                             base, REQUEST_LATENCY)
        self.queue_time = Histogram("vllm:request_queue_time_seconds",
                                    "Histogram of time spent in WAITING phase for request.", base, REQUEST_LATENCY)
        self.inference_time = Histogram("vllm:request_inference_time_seconds",
                                        "Histogram of time spent in RUNNING phase for request.", base, REQUEST_LATENCY)
        self.prefill_time = Histogram("vllm:request_prefill_time_seconds",
                                      "Histogram of time spent in PREFILL phase for request.", base, REQUEST_LATENCY)
        self.decode_time = Histogram("vllm:request_decode_time_seconds",
                                     "Histogram of time spent in DECODE phase for request.", base, REQUEST_LATENCY)
        # llm_mini_infer: the prefix cache (architecture.md, "The cache")
        self.state_usage = Gauge("llm_mini_infer:state_slot_usage_perc",
                                 "Live recurrent-state slot usage (running requests). 1 means 100 percent.", base)
        self.checkpoint_usage = Gauge("llm_mini_infer:checkpoint_slot_usage_perc",
                                      "Checkpoint state slot usage (prefix cache; sliding-window models: the KV "
                                      "pages only windows hold). 1 means 100 percent.", base)
        self.cached_pages = Gauge("llm_mini_infer:prefix_cache_pages", "KV pages the prefix cache holds.", base)
        self.preempted_hits = Counter("llm_mini_infer:prefix_cache_preempted_hits",
                                      "Tokens served from the prefix cache to requests readmitted after preemption.",
                                      base)
        self.ckpt_hits = Counter("llm_mini_infer:checkpoint_hits_total",
                                 "Admissions that resumed from a checkpoint.", base)
        self.ckpt_misses = Counter("llm_mini_infer:checkpoint_misses_total",
                                   "Admissions of a hybrid or sliding-window model without a usable checkpoint.", base)
        self.ckpts_taken = Counter("llm_mini_infer:checkpoints_total",
                                   "Checkpoints created, by source (prefill end, branch point, client hint, decode "
                                   "interval, finish: a request leaving, finished or aborted; refine: a cached prefix "
                                   "recomputed with BF16 activations; preempt: a preempted request's donation).", base,
                                   labels=("source",))
        self.ckpt_resumes = Counter("llm_mini_infer:checkpoint_resumes_total",
                                    "Admissions resumed from a checkpoint, in either namespace, by the checkpoint's "
                                    "source (as llm_mini_infer:checkpoints_total): together, "
                                    "llm_mini_infer:checkpoint_hits_total.", base, labels=("source",))
        self.ckpt_evictions = Counter("llm_mini_infer:checkpoint_evictions_total",
                                      "Checkpoints evicted from the prefix cache, by source (as "
                                      "llm_mini_infer:checkpoints_total) and by whether they served a resume (yes) or "
                                      "none (no) since they were taken.", base, labels=("source", "served"))
        self.ckpts_skipped = Counter("llm_mini_infer:checkpoints_skipped_total",
                                     "Checkpoints not taken because no checkpoint slot could be freed.", base)
        self.cache_waits = Counter("llm_mini_infer:prefix_cache_waits_total",
                                   "Admissions deferred one step for KV or a checkpoint being written.", base)
        self.lost_tokens = Counter("llm_mini_infer:prefix_cache_lost_tokens_total",
                                   "Prompt tokens the prefix cache held for an admission but the admission recomputed "
                                   "(the cache's match within the prompt less its usable hit), by cause: writing (the "
                                   "KV, or the checkpoint ending it, that the plan under construction writes), "
                                   "checkpoint (a hybrid or sliding-window model's KV past the deepest checkpoint a "
                                   "hit may use: none was taken there, it was evicted, or it lies inside an image "
                                   "span), page (an attention-only model's partial page past its page-aligned hit), "
                                   "precise (an FP8 copy's hit above the precise prefix's copy the admission resumed "
                                   "from).", base,
                                   labels=("cause",))
        # the precise prefix (architecture.md, "The precise prefix"): block-FP8 models
        self.refines = Counter("llm_mini_infer:prefix_cache_refines_total",
                               "Admissions that recomputed a cached prompt prefix with BF16 activations (the precise "
                               "prefix: an FP8 copy reused often enough).", base)
        self.precise_tokens = Counter("llm_mini_infer:prefix_cache_precise_tokens_total",
                                      "Prompt tokens computed with BF16 activations (PRECISE rows): refines for the "
                                      "prefix cache, or with fp8_cold_prefill=bf16 every cold prompt's.", base)
        self.precise_hits = Counter("llm_mini_infer:prefix_cache_precise_hits_total",
                                    "Admissions that resumed from a prefix computed with BF16 activations.", base)
        self.precise_steps = Counter("llm_mini_infer:prefix_cache_precise_steps_total",
                                     "Steps that carried PRECISE rows: each dequantizes every weight and runs eagerly.",
                                     base)
        self.evicted_pages = Counter("llm_mini_infer:prefix_cache_evicted_pages_total",
                                     "KV pages freed by evicting prefix-cache entries.", base)
        self.evicted_ckpts = Counter("llm_mini_infer:prefix_cache_evicted_checkpoints_total",
                                     "Checkpoints evicted from the prefix cache (by source: "
                                     "llm_mini_infer:checkpoint_evictions_total).", base)
        self.evicted_window_pages = Counter("llm_mini_infer:prefix_cache_evicted_window_pages_total",
                                            "Window pages that evicting windows freed (sliding-window models: a "
                                            "window's pages that no other window or request held).", base)
        self.windows = Gauge("llm_mini_infer:prefix_cache_windows",
                             "Windows the prefix cache holds (sliding-window models' checkpoints).", base)
        self.failed = Counter("llm_mini_infer:requests_failed_total",
                              "Requests failed by a non-finite result (their own, or a rolled-back write their hit "
                              "used), an empty grammar mask or a token their grammar rejected.", base)
        # llm_mini_infer: FlashInfer's attention plans for MTP rows of uncertain KV length (`kernels.flashinfer`)
        self.attention_replans = Counter("llm_mini_infer:attention_replans_total",
                                         "FlashInfer attention plans made again with another KV split: the first "
                                         "split's chunk count could change with an MTP row's true KV length.", base)
        self.attention_plans_kept = Counter("llm_mini_infer:attention_plans_kept_total",
                                            "Decode-graph attention plans kept although the true KV lengths could "
                                            "change their chunk count (no safe split fit the captured grid): those "
                                            "rows may attend wrongly.", base)
        self.attention_plans_unsplit = Counter("llm_mini_infer:attention_plans_unsplit_total",
                                               "Attention plans made without a KV split because the startup self-check "
                                               "could not read FlashInfer's plan info (correct, slower).", base)
        self.vision_plan_waits = Counter("llm_mini_infer:vision_plan_waits_total",
                                         "Vision-tower attention plans that waited on the host for their page-locked "
                                         "buffer's last copy to run: more encoder calls in flight than the tower's "
                                         "plan buffers (four).", base)
        # llm_mini_infer: the grammar-aware heads (architecture.md, "Grammar-aware heads"; `runner.base.HeadCounters`)
        self.grammar_head_steps = Counter("llm_mini_infer:grammar_head_steps_total",
                                          "Steps with grammar-masked rows, by the LM head their sampling ran on: "
                                          "sparse (reason union: only the rows of the union of the step's masks), or "
                                          "dense because the sparse head is off (off: dev.runner.cuda.sparse_head), a "
                                          "sample token had no mask or a mask allowed every token (unconstrained), or "
                                          "the union exceeded dev.runner.cuda.sparse_head_rows (over_cap).", base,
                                          labels=("path", "reason"))
        self.grammar_head_union_tokens = Histogram("llm_mini_infer:grammar_head_union_tokens",
                                                   "Tokens in the union of a step's grammar masks (the LM-head rows "
                                                   "the sparse head computes), over the steps whose sample tokens all "
                                                   "had a mask.", base, UNION_BUCKETS)
        self.draft_universe_tokens = Gauge("llm_mini_infer:draft_universe_tokens",
                                           "Tokens in the draft universe, the union of the masks of every step the "
                                           "sparse head could take: the only rows the drafter's head (Gemma 4's "
                                           "assistant, Qwen's MTP layer) scores on such steps (0: off or empty).",
                                           base)
        self.draft_universe_steps = Counter("llm_mini_infer:draft_universe_steps_total",
                                            "Steps whose MTP drafts scored the draft universe alone.", base)
        self.draft_universe_rebuilds = Counter("llm_mini_infer:draft_universe_rebuilds_total",
                                               "Times the drafter's universe head was rebuilt: the draft universe "
                                               "grew.", base)
        # llm_mini_infer: overhead accounting (architecture.md, "Observability")
        self.stage = Histogram("llm_mini_infer:request_stage_seconds",
                               "TTFT-path stages: receive (the request body's receipt), render (its JSON decode and "
                               "checks, its grammar, its images' preparation and the template's render), tokenize, "
                               "to_admit (IPC + queue), to_launch, to_first_token (includes runner time), deliver "
                               "(engine send to first byte).", base, OVERHEAD, labels=("stage",))
        self.ttft_overhead = Histogram("llm_mini_infer:ttft_overhead_seconds",
                                       "Time to first byte from the request body's receipt, minus runner time of the "
                                       "steps before the first token, and minus the image stages on the path (fetch, "
                                       "the wait for the image pool).", base, OVERHEAD)
        self.step_cpu = Histogram("llm_mini_infer:engine_step_cpu_seconds",
                                  "Engine-thread time per loop turn, excluding blocking (poll, device) and the "
                                  "runner's launch calls.", base, STEP)
        self.step_runner_calls = Histogram("llm_mini_infer:engine_step_runner_calls_seconds",
                                           "CPU time of the runner's launch_forward and launch_sample per loop turn.",
                                           base, STEP)
        self.step_runner = Histogram("llm_mini_infer:engine_step_runner_seconds", "Runner (device) busy time per step.",
                                     base, STEP)
        self.step_decode_rows = Histogram("llm_mini_infer:step_decode_rows",
                                          "Decode (MTP verify) rows per step, every step the engine completed "
                                          "(vllm:iteration_tokens_total counts their 1 + k tokens with the prefill, "
                                          "and only steps with outputs).", base, STEP_ROWS)
        self.step_encoder_images = Histogram("llm_mini_infer:step_encoder_images", "Images the vision encoder ran for, "
                                             "per step (0: a step without an encoder call), every step the engine "
                                             "completed.", base, STEP_IMAGES)
        self.page_waits = Counter("llm_mini_infer:page_waits_total",
                                  "Admissions and running prefill chunks that waited a step for KV pages (decode rows "
                                  "short of pages preempt: vllm:num_preemptions).", base)
        self.gpu_busy = Counter("llm_mini_infer:gpu_busy_seconds_total", "Runner (device) busy time.", base)
        self.gpu_idle = Counter("llm_mini_infer:gpu_idle_seconds_total",
                                "Runner (device) idle time while work was queued: CPU-induced idle.", base)
        self.gpu_sample = Counter("llm_mini_infer:gpu_sample_seconds_total",
                                  "Runner (device) time of the sampling part of steps split into a forward and a "
                                  "sampling (grammar masks; every step with always-split graphs): part of busy.",
                                  base)
        self.boundary_wait = Counter("llm_mini_infer:boundary_wait_seconds_total",
                                     "Engine time polling for the end of the step in flight's forward before planning "
                                     "the next (engine.plan_at_boundary).", base)
        self.engine_phase = Counter("llm_mini_infer:engine_phase_seconds_total",
                                    "Engine loop time by phase: poll (the frontends' messages, blocking idle "
                                    "excluded), schedule (the next step's plan), launch_forward and launch_sample (the "
                                    "runner's host calls), wait (for the step in flight's results), process (those "
                                    "results folded in), send (the outputs to the frontends), masks (grammar masks), "
                                    "boundary (as llm_mini_infer:boundary_wait_seconds_total).", base,
                                    labels=("phase",))
        self.gc_seconds = Counter("llm_mini_infer:gc_seconds_total",
                                  "Time in Python's garbage collector (gc.callbacks), by process (engine, or this "
                                  "frontend) and generation.", base, labels=("process", "generation"))
        self.gc_collections = Counter("llm_mini_infer:gc_collections_total",
                                      "Python garbage collections, by process (engine, or this frontend) and "
                                      "generation.", base, labels=("process", "generation"))
        self.engine_aborts = Counter("llm_mini_infer:engine_aborts_total",
                                     "Aborts that found a live request in the engine, by reason. Reasons are "
                                     "disconnect, stop, failed, and shutdown.", base, labels=("reason",))
        self.step_output_errors = Counter("llm_mini_infer:step_output_errors_total",
                                          "Engine outputs whose statistics or pixel releases this frontend failed to "
                                          "process, each logged, by part: stats, release. Their requests' tokens were "
                                          "delivered.", base, labels=("part",))
        # llm_mini_infer: images (architecture.md, "Images")
        self.images = Counter("llm_mini_infer:images_total", "Images in the requests this frontend sent to the engine.",
                              base)
        self.pool_restarts = Counter("llm_mini_infer:image_pool_restarts_total",
                                     "Image pool workers retired: died (a native decoder crash, an out-of-memory "
                                     "kill) or killed at the job deadline, respawned unless the pool was stopping "
                                     "(one killed while it started at shutdown counts too): the same in every "
                                     "frontend.", base)
        self.pool_hangs = Counter("llm_mini_infer:image_pool_hangs_total",
                                  "Image pool workers the launcher killed for holding one image past the job deadline "
                                  "(a decoder hang), each a restart too: the same in every frontend.",
                                  base)
        self.pool_jobs_failed = Counter("llm_mini_infer:image_pool_jobs_failed_total",
                                        "Images whose pool worker stopped while decoding them (a crash, or past the "
                                        "job deadline): each failed its request with a 400 (this frontend's).", base)
        self.pool_jobs_resent = Counter("llm_mini_infer:image_pool_jobs_resent_total",
                                        "Images re-sent because their pool worker stopped before it decoded them, "
                                        "to another worker or to its respawn, once: a second such "
                                        "stop fails the image with a 503 (this frontend's).", base)
        self.image_stage = Histogram("llm_mini_infer:image_stage_seconds",
                                     "Image stages: fetch (all of a request's images), probe (header, sizing and "
                                     "hash, per request), pool (decode and resize in the image pool, per image), "
                                     "pool_wait (the request's wait for the pool after tokenizing), vit (the "
                                     "encoder's device time per step that encodes images).", base, OVERHEAD,
                                     labels=("stage",))
        self.encoder_hits = Counter("llm_mini_infer:encoder_cache_hits_total",
                                    "Image spans prefilled from embeddings already in the encoder cache.", base)
        self.encoder_misses = Counter("llm_mini_infer:encoder_cache_misses_total",
                                      "Images the encoder ran for (encoder cache misses).", base)
        self.encoder_tokens = Counter("llm_mini_infer:encoder_tokens_total",
                                      "Image embeddings the encoder computed.", base)
        self.encoder_time = Counter("llm_mini_infer:encoder_seconds_total", "Device time of the vision encoder.", base)
        self.structured = Counter("llm_mini_infer:structured_requests_total",
                                  "Requests with a response_format grammar the engine accepted.", base)
        self.grammar_compiles = Counter("llm_mini_infer:grammar_compiles_total",
                                        "Grammars the engine compiled (cache misses).", base)
        self.grammar_compile_time = Counter("llm_mini_infer:grammar_compile_seconds_total",
                                            "Engine time compiling grammars.", base)
        self.grammar_rejects = Counter("llm_mini_infer:grammar_rejects_total",
                                       "Requests whose grammar the engine could not compile, or not within "
                                       "dev.structured_output.compile_timeout_s.", base)
        self.grammar_masks = Counter("llm_mini_infer:grammar_masks_total", "Grammar masks the engine filled.", base)
        self.grammar_mask_time = Counter("llm_mini_infer:grammar_mask_seconds_total",
                                         "Engine time filling grammar masks (hidden behind the forward).", base)
        self.grammar_errors = Counter("llm_mini_infer:grammar_errors_total",
                                      "Requests failed by a token their grammar rejected (never expected).", base)
        self.grammar_end_waits = Counter("llm_mini_infer:grammar_end_waits_total",
                                         "Decode rows not planned because the request's row in flight could end its "
                                         "grammar (the device kept free for the next request).", base)
        self.dropped_samples = Counter("llm_mini_infer:dropped_samples_total",
                                       "Sampled rows whose request had already left (the step after a stop, an abort "
                                       "or a preemption), or that the same step failed (a non-finite result, an empty "
                                       "grammar mask): work the pipelined loop wasted.", base)
        # llm_mini_infer: the forced prefix, the EOS-free finish (architecture.md, "Grammar-aware heads") and the MTP
        # verify rows by grammar mask
        self.grammar_forced_requests = Counter("llm_mini_infer:grammar_forced_requests_total",
                                               "Requests whose final prefill chunk took the tokens their grammar "
                                               "forces at its start (dev.structured_output.forced_prefix).", base)
        self.grammar_forced_tokens = Counter("llm_mini_infer:grammar_forced_tokens_total",
                                             "Output tokens decided by the forced prefix: computed in the final "
                                             "prefill chunk, never sampled.", base)
        self.grammar_eos_free = Counter("llm_mini_infer:grammar_eos_free_finishes_total",
                                        "Requests finished with their grammar's EOS appended once the grammar stopped "
                                        "(dev.structured_output.eos_free_finish), by whether a row of theirs was in "
                                        "flight (yes: its sample is dropped; no: a step was never planned).",
                                        base, labels=("in_flight",))
        self.grammar_eos_free_gated = Counter("llm_mini_infer:grammar_eos_free_gated_total",
                                              "Requests whose stopped grammar's EOS was left to the step in flight: "
                                              "no other request stayed in it.", base)
        self.spec_first_rejections = Counter("llm_mini_infer:spec_first_rejections_total",
                                             "MTP verify rows with a rejected draft, by the grammar mask its target "
                                             "sample was drawn under: forced (one token), eos (the EOS alone: the end "
                                             "of a document), choice, free (unconstrained). Rows without masks are "
                                             "not counted.", base, labels=("mask",))
        self.spec_window_mask_tokens = Histogram("llm_mini_infer:spec_window_mask_tokens",
                                                 "Tokens the grammar masks of MTP verify rows allowed, every mask "
                                                 "verification read (the samples after the pending token and after "
                                                 "each accepted draft).", base, POPCOUNT_BUCKETS)
        self.backpressure_pauses = Counter("llm_mini_infer:frontend_backpressure_pauses_total",
                                           "Times a frontend lagging behind its outputs had its admissions paused.",
                                           base)
        self.paused_frontends = Gauge("llm_mini_infer:frontends_paused", "Frontends whose admissions are paused.", base)
        self.backpressure_waits = Counter("llm_mini_infer:frontend_backpressure_waits_total",
                                          "Engine loop turns that blocked on the poll because its only work was the "
                                          "waiting requests of paused frontends.", base)
        self.unauthorized = Counter("llm_mini_infer:unauthorized_requests_total",
                                    "Requests refused with 401: a missing or wrong API key.", base)
        # HTTP responses (`api.app.HttpTally`): http_requests_total, as prometheus-fastapi-instrumentator's, carries no
        # model_name or engine label
        self.http_requests = Counter("http_requests_total", "Total number of requests by method, status and handler.",
                                     "", labels=("method", "status", "handler"))
        self.http_responses = Counter("llm_mini_infer:http_responses_total",
                                      "HTTP responses by handler, method and status code, each counted once, as its "
                                      "status line is sent (a stream's with its first chunk); 499: the client left "
                                      "before it. Probe and scrape paths are not counted, as in http_requests_total.",
                                      base, labels=("handler", "method", "status"))
        self.client_disconnects = Counter("llm_mini_infer:client_disconnects_total",
                                          "Requests whose client disconnected before their response ended, by handler "
                                          "and whether they streamed: true, a stream cut short (before or after its "
                                          "first chunk); false, a request abandoned before its response, or while it "
                                          "sent its body.", base, labels=("handler", "stream"))
        self.cache_info = Gauge("vllm:cache_config_info", "Information of the LLMEngine CacheConfig", base)
        # vLLM's queue caps (`api.app.App.admit`): its name, help text and labels (no engine label), both reasons at 0
        self.admission_rejections = Counter("vllm:admission_rejections",
                                            "Number of admission attempts rejected by a queue limit, by reason.",
                                            self._model_label, labels=("reason",))
        for reason in ("max_num_queued_reqs", "max_num_queued_tokens"):
            self.admission_rejections.inc(0.0, reason)
        self.chat_inflight = Gauge("llm_mini_infer:chat_requests_inflight",
                                   "Chat completion requests this frontend holds, from the drain gate to the end of "
                                   "their response: the body's read, the preparation (grammar, images, render) and "
                                   "the response included. Requests the gate refuses and other routes are not "
                                   "counted.", base)
        self._base = base
        self.spec: SpecMetrics | None = None  # MTP: created by the first stats that carry it (only when MTP is on)

    def spec_metrics(self, k: int) -> SpecMetrics:
        """The speculative-decoding metrics, made by the first stats that carry them (`k` drafts a step)."""
        if self.spec is None:
            self.spec = SpecMetrics(self._base, k)
        return self.spec

    def set_cache_info(self, page_size: int, num_pages: int, num_state_slots: int, prefix_caching: bool = True) -> None:
        self.cache_info.set(1.0)
        self.cache_info.base += (f',block_size="{page_size}",num_gpu_blocks="{num_pages}",'
                                 f'num_state_slots="{num_state_slots}",'
                                 f'enable_prefix_caching="{prefix_caching!s}"')

    def set_gc(self, process: str, ns: Sequence[int], collections: Sequence[int]) -> None:
        """A process's garbage-collector counters (`gcstats.GcStats`), per generation."""
        for g, (t, n) in enumerate(zip(ns, collections)):
            self.gc_seconds.set_total(t / 1e9, process, str(g))
            self.gc_collections.set_total(n, process, str(g))

    def enable_tool_parser(self) -> None:
        """`vllm:tool_call_parser_invocations_total` (its name, help text and labels vLLM's), registered where tool
        calls are parsed, every label combination at 0: once per choice not streamed, once per step streamed, of the
        requests with tools."""
        c = self.tool_parser = Counter("vllm:tool_call_parser_invocations_total",
                                       "Total number of ToolParser invocations. Non-streaming increments once per "
                                       "choice; streaming increments once per delta.", self._model_label,
                                       labels=("mode", "outcome", "request_type"))
        for mode in ("streaming", "non_streaming"):
            for outcome in ("tool_call", "no_tool_call"):
                for request_type in ("chat_completions", "responses", "other"):
                    c.inc(0.0, mode, outcome, request_type)

    def render(self) -> bytes:
        lines: list[str] = []
        for m in vars(self).values():
            if isinstance(m, _Metric):
                lines.extend(m.render())
        if self.spec is not None:
            for m in vars(self.spec).values():
                lines.extend(m.render())
        return ("\n".join(lines) + "\n").encode()


def acceptance_buckets(k: int) -> tuple[float, ...]:
    """Tokens per verify step, 1 to 1 + k by quarters."""
    return tuple(1 + i / 4 for i in range(4 * k + 1))


class SpecMetrics:
    """The `vllm:spec_decode_*` counters (names and meaning of vLLM's `v1/spec_decode/metrics.py`): a draft is one
    request's verify step; and llm-mini-infer's per-request acceptance."""

    def __init__(self, base: str, k: int):
        self.num_drafts = Counter("vllm:spec_decode_num_drafts", "Number of spec decoding drafts.", base)
        self.num_draft_tokens = Counter("vllm:spec_decode_num_draft_tokens", "Number of draft tokens.", base)
        self.num_accepted_tokens = Counter("vllm:spec_decode_num_accepted_tokens", "Number of accepted tokens.", base)
        self.num_accepted_tokens_per_pos = Counter("vllm:spec_decode_num_accepted_tokens_per_pos",
                                                   "Accepted tokens per draft position.", base, labels=("position",))
        self.tokens_per_verify_step = Histogram("llm_mini_infer:request_tokens_per_verify_step",
                                                "Per request the engine finished with MTP verify steps (not one a stop "
                                                "string ended in the frontend): the tokens they delivered over their "
                                                "number, 1 to 1 + k (the mean acceptance length of one request: 1 + "
                                                "accepted drafts per verify step; a finish cuts the last step's "
                                                "tokens). The request log line carries both (verify_steps, "
                                                "verify_tokens).", base,
                                                acceptance_buckets(k))
