# Benchmarks

This directory holds the benchmarks that engine work runs again. Run each script from the repository's root with `uv run python bench/...`: `--help` lists its options. The results go to the git-ignored `results/` at the repository's root, unless `--out` gives a path.

| directory | what it measures | what it needs |
|---|---|---|
| [`server/`](server/README.md) | the server's own CPU cost: TTFT overhead, engine CPU per step, GPU idle time caused by the CPU, SSE throughput | the fixtures (`uv run tests/scripts/fetch_fixtures.py`); no GPU |
| [`load/`](#a-load-generator) | any OpenAI-compatible server's latency, throughput and answers | a running server; a tokenizer for scene and chat |
| [`structured/`](#structured-output) | the grammar backend's CPU cost, and the validity of structured answers | a model directory; a running server |
| [`engines/`](#comparing-engines) | llm-mini-infer, vLLM and SGLang on one GPU, under the same loads | a GPU host with the three engines installed |
| `reports/` | published benchmark reports; `reports/check.py` checks them for private data | nothing |

Against a server with an API key, give the key with `--api-key` or `OPENAI_API_KEY`.

## A load generator

`bench/load/run.py` sends chat completions to any OpenAI-compatible server (llm-mini-infer, vLLM, SGLang) and reports what a client sees, so two engines compare on the same requests and arrivals. Its workloads (`--workload`):
- `scene` (the default): the host check's synthetic workload, one new image per request, a system prompt shared by every request (`--prompt-tokens`), a share of portrait images (`--tall-share`) and a strict JSON schema. The image budget is the server's own setting (`images.max_soft_tokens`): a client cannot set it;
- `chat`: text prompts of `--input-tokens` tokens, no schema. `--shared-prefix-tokens N` starts every prompt with the same N tokens, so a later request measures its own part over a cached prefix. With `--turns N` and `--concurrency`, each client holds a conversation: its next request is the previous messages, the reply it got and a new user message of `--tool-tokens` tokens, and after N turns it starts another. Its later turns follow each engine's own replies, so only the first turn's answers compare across engines;
- `catalog`: text requests with the catalog schema (`tests/data/catalog.schema.json`: a schema with nested `$defs`, nullable enum lists and strict objects), each prompt different;
- a directory in the [host check's layout](../docs/operations.md#the-host-check).

A load is Poisson arrivals (`--rates 2,4 --slots 32,64`: at most that many in flight, an arrival beyond that dropped and counted) or a closed loop (`--concurrency 1,8,32`). It measures the requests that arrive in the `--duration` window, after `--warmup`. It streams at temperature 0 with thinking off. `--extra-body` replaces that default, so repeat `chat_template_kwargs` in it.

```bash
export OPENAI_API_KEY=$LLM_MINI_INFER_API_KEY        # a server as in the quickstart (README.md)
uv run python bench/load/run.py --model RedHatAI/gemma-4-31B-it-FP8-block --rates 2,4 --slots 32,64
uv run python bench/load/run.py --model RedHatAI/gemma-4-31B-it-FP8-block --workload chat --concurrency 1,8,32 \
    --input-tokens 1024 --max-tokens 256 --ignore-eos
```

The report gives, per load, the requests (succeeded, dropped, failed), the latencies (end to end, TTFT, TPOT), the tokens and the throughput. From `/metrics`, it adds the engine's counters for the window: acceptance, GPU idle time, the grammar-aware heads. The cache columns come from `usage.prompt_tokens_details.cached_tokens`: the hit rate (cached over prompt tokens) and the uncached tokens per request (p50, p90); a session load adds the hit rate by turn index (`cache_by_turn`). vLLM reports the field only when started with `--enable-prompt-tokens-details` (streamed, only in the final usage chunk that `stream_options.include_usage` asks for, which `run.py` sends), and SGLang only with `--enable-cache-report` (off by default); without them the columns show `-` (`null` in the report), not 0. llm-mini-infer reports it with every usage. A session's replies are kept in memory for its next turn, not in the report, which keeps each answer's digest (16 hex digits of its SHA-256), not its text. A short answer can be found from its digest, so keep a private workload's report private. The script exits 1 when a measured request failed.

Compare engines on the same GPU, checkpoint and settings, and run each load more than once.

**Comparing answers.** `bench/load/compare.py` puts runs side by side (each `[NAME=]PATH`, a report or a directory of reports) and counts the identical answers of each pair. Without batch invariance, answers under load change with the batches. Thus two passes of one server set the floor (`--floor A1,A2`), and every other pair is compared with it, with a 95% bootstrap interval.

An A/B test of one server setting: run each variant twice, each pass on a new server, with the same `--seed`:

```bash
uv run llm-mini-infer serve --model <model dir> --port 8000 &   # A1; A2 the same into ab/A2
until curl -sf http://127.0.0.1:8000/health > /dev/null; do sleep 5; done
uv run python bench/load/run.py --model <served name> --workload catalog --rates 2,4 --slots 32,64 \
    --out results/ab/A1/catalog.json
kill %1; wait
uv run llm-mini-infer serve --model <model dir> --port 8000 --set <the setting>=<the other value> &   # B1; B2 the same
until curl -sf http://127.0.0.1:8000/health > /dev/null; do sleep 5; done
uv run python bench/load/run.py --model <served name> --workload catalog --rates 2,4 --slots 32,64 \
    --out results/ab/B1/catalog.json
kill %1; wait
uv run python bench/load/compare.py results/ab/* --floor A1,A2 --floor B1,B2
```

A difference between A and B no larger than between A1 and A2 is noise. "not below the floor" means that the setting moves no more answers than the batches do, within the interval.

## Comparing engines

`bench/engines/drive.py` runs llm-mini-infer, vLLM and SGLang one after another on one GPU, each under the same loads from `bench/load/run.py` over the loopback interface. `bench/engines/arms.py` is the one place that holds each engine's server command and environment, with what is held equal and what is not. The driver runs a family of measurements as units: per arm, it starts the server and times the start (`bench/load/startup.py`), checks the arm (a cached prefix, valid compact JSON, no reasoning, drafts verified), runs the loads, saves `/metrics` around each load and the GPU's clock and power each second, then writes a manifest. A unit that finished is skipped when the family runs again; a load that crashed, timed out or wrote no report fails its unit (run.py's exit 1, some requests failed, is a result). Family `I` measures the cost of batch invariance: each engine in both modes under the c=64 scene and chat loads, on fresh servers. Family `L` runs the long contexts (decode, cold prefill and a short turn over a cached history, usually with `--engines ours`); `--context N` sets every arm's context (default 8,184; family `I`'s pair at any context is llm-mini-infer's default mode and the other, `bench/engines/arms.py`'s `default_batch_invariant`), `--load NAME` runs only that load of the family (repeatable; each unit keeps the named loads it has, a unit with none of them is left out), and `L`'s loads that do not fit it are skipped and named in the manifest. `--skip-validity` leaves the structured-output validity run out of family `D` (a named deviation in the manifest). `bench/engines/gpu_control.py` measures the GPU alone (matmul and copy throughput, clock, power), so two hosts, or one host over hours, compare by numbers.

```bash
python3 bench/engines/drive.py --out results/engines --model gemma --family A1 --dry-run   # the commands
python3 bench/engines/drive.py --out results/engines --model gemma --family A1
```

| need | llm-mini-infer | vLLM v0.33.0 | SGLang v0.5.23 |
|---|---|---|---|
| cached tokens in `usage` | always | `--enable-prompt-tokens-details` | `--enable-cache-report` |
| verify steps in `/metrics` | always | always | `--enable-metrics` |
| draft count k | `--set speculative.num_speculative_tokens=k` | `--speculative-config` with `"num_speculative_tokens": k` | `--speculative-num-steps k --speculative-eagle-topk 1 --speculative-num-draft-tokens k+1` |
| batch invariance | on by default (Qwen3.8: up to a context of 32,768; `dev.runner.cuda.batch_invariant`) | `VLLM_BATCH_INVARIANT=1` | `--enable-deterministic-inference` |
| llguidance, compact JSON | the image's configuration | `--structured-outputs-config '{"backend":"guidance","disable_any_whitespace":true}'` | `--grammar-backend llguidance --constrained-json-disable-any-whitespace` |

`run.py`'s options for this: `--requests N` (a closed loop of exactly N requests, after `--warmup-requests` from another seed), `--first-index` (several runs against one server never resend a request), `--find-rate LO,HI --bound-file PATH` (the highest Poisson rate within a latency bound), `--meta` (the arm, in the report) and `--think` (a session's pauses between turns). Tokens per verify step have one definition for every engine: the completion tokens after the first, over the server's verify steps (`spec` in each load). llm-mini-infer also delivers tokens no verify step makes (a grammar's forced tokens), so compare acceptance on the chat workload.

## Structured output

`bench/structured/grammar_cost.py` measures what llm-mini-infer's llguidance backend spends: building the tokenizer, compiling the corpus's schemas (`tests/support/schema_corpus.py`) and cloning a grammar. With `--catalog` it also compiles the catalog schema and fills a step's masks along its documents for 1 to 64 requests on 1 to 8 threads.

```bash
uv run python bench/structured/grammar_cost.py                    # the Gemma 4 fixture's tokenizer
uv run python bench/structured/grammar_cost.py --model <model dir> --backend guidance --compact --catalog
```

`bench/structured/validity.py` sends a running server requests with a `json_schema` response format: every schema of the corpus (`tests/support/schema_corpus.py`), and with `--n-catalog N` that many catalog requests, greedy and sampled, streamed and not. Each stop answer must validate against its schema (`jsonschema`, Draft 2020-12; no `time`, `date-time` or `uri` format checks). It exits 1 on any invalid answer (a repeated key is counted apart), error or unexpected finish.

```bash
uv run python bench/structured/validity.py --base-url http://127.0.0.1:8000 --model <served name>
```
