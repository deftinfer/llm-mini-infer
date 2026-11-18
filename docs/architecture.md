# Architecture

How llm-mini-infer works. [operations.md](operations.md) covers running it and its settings, and [development.md](development.md) working on it.

## Scope and decisions

- **Configuration is derived.** Every default comes from the model and the hardware, and every setting has an override. No default is tuned for one workload.
- **The goal** is lower latency than vLLM on one SM120 GPU, with the comparable settings held equal. The reference workloads are Gemma 4 31B FP8 with its assistant drafter, and Qwen3.8 27B FP8. [benchmarks.md](benchmarks.md) has the numbers.
- **vLLM compatibility.** The server speaks the OpenAI API with vLLM's extension fields, metrics and settings. [vllm-compatibility.md](vllm-compatibility.md) lists what matches and what differs.
- **Python only.** The server runs on CPython 3.12. Outside the CUDA kernels, it has no second language.
- **Models.** Gemma 4 31B (dense, instruction-tuned) and Qwen3.8 (hybrid Gated DeltaNet) are the supported families. The list of architectures is curated. AWQ, GPTQ, GGUF and bitsandbytes are not supported.
- **Hardware.** CUDA 13 and SM120. One server drives one GPU. The FP8 GEMM and the image's prebuilt kernels target `sm_120a`.
- **Kernels.** FlashInfer, CUTLASS and our own kernels where SM120 has none. GDN and mixture-of-experts layers use no Triton kernels, and nothing uses `torch.compile`: the engine uses CUDA graphs and hand-fused operations. The target for a start with cached weights is under 60 s.
- **Quantization.** FP8 with one FP32 scale per 128 × 128 block. On SM120, a mixture-of-experts family would take NVFP4 or MXFP4 experts only.
- **Licence.** Apache-2.0.
- **Not planned.** Expert, pipeline and context parallelism. Prefill and decode disaggregation. KV offload tiers. LoRA. Beam search, `n` above 1 and `best_of`. Legacy completion options (`echo`, suffix). Embeddings and reranking. Audio and video. AMD, Intel and CPU backends (the CPU path exists for tests). [Not yet](#not-yet) lists what may come.

## Overview

llm-mini-infer serves a small set of recent models (hybrid Gated DeltaNet and attention-only GQA models) on one NVIDIA GPU, behind an OpenAI-compatible API. It is written in Python. The model code is plain PyTorch, and its kernels are FlashInfer's, CUTLASS's and a small library of its own.

A request takes this path:
1. A frontend process accepts the HTTP request ([The frontend](#the-frontend)).
2. The frontend renders the chat template and tokenizes the prompt ([Rendering and tokenizing](#rendering-and-tokenizing)). The image pool prepares any images ([Images](#images)).
3. The frontend sends the request to the engine over ZMQ ([Frontend and engine messages](#frontend-and-engine-messages)). The engine's scheduler admits it and plans each step ([The scheduler](#the-scheduler)).
4. The step loop launches each plan on the GPU, one step ahead of the results ([The step loop](#the-step-loop)).
5. The frontend detokenizes the new tokens and streams them to the client ([Detokenizing and stop strings](#detokenizing-and-stop-strings)).

```
                 ┌────────────── frontend processes (one by default) ───────────────┐
 HTTP / SSE ───► │ api     chat completions · tokenize · models · health · metrics  │
                 │ render  chat template (Jinja) · tokenizer                        │
                 │ images  fetch · probe and sizing · content hashes · pool jobs    │
                 │ output  streaming detokenizer · stop strings                     │
                 └──────┬───────────────────────┬───────────────────────▲───────────┘
                        │ decode jobs           │ AddRequest, Abort, Ack│ StepOutputs
                        ▼                       ▼ ZMQ ipc, msgspec      │ per step
    ┌───────────────────┴─────────┐   ┌─────────┴───────────────────────┴───────────┐
    │ image pool (processes, one  │   │ engine process: one Python thread           │
    │ torch thread each): exact   │──►│  step loop · scheduler · cache (radix tree, │
    │ uint8 decode and resize,    │shm│  KV pages, state slots) · grammar masks     │
    │ pixels into shared memory   │   │  runner: PyTorch, FlashInfer, CUTLASS, our  │
    └─────────────────────────────┘   │  CUDA kernels, CUDA graphs                  │
                                      └─────────────────────────────────────────────┘
```

**Principles**
1. **One thread drives the GPU.** The engine process runs the scheduler, the cache, the grammar masks and the model runner on one Python thread. Its only other Python threads build grammars, mostly in llguidance's native code, and the native code it calls (FlashInfer, llguidance) releases the GIL. HTTP, templates, tokenizing and image decoding run in other processes.
2. **The scheduler never waits for the GPU.** Step N+1 is planned and launched before step N's tokens reach the CPU, and decode inputs stay on the GPU.
3. **Explicit runner state.** The runner keeps per-slot tables, and only the scheduler's step plans change them.
4. **Vectorized bookkeeping.** The state of running requests is struct-of-arrays (numpy) indexed by slot. Per-request Python runs only for requests whose state changes, and for grammars.
5. **Backends declare their capabilities.** At startup the runner reports its geometry, memory, step costs and features, and the engine sizes the pools, turns off what the runner cannot do and configures the scheduler from the report.
6. **Recurrent state is a cache type of its own,** with its own pool, not something stored in attention pages.
7. **Every setting but `model.path` has a default and an override.** A default that depends on the model, the GPU, the host or a startup measurement is derived, and its reason is logged and served.
8. **No kernel compiles or autotunes while serving.** Kernels are built ahead of time (the image carries them) or once at startup, before the server reports ready.

## Processes

One server is the launcher and up to three kinds of processes that it starts with `spawn`:

- **The launcher** is the main process. It downloads the model (and a drafter, `speculative.model`) when given as a Hugging Face repo id, derives the configuration, starts the image pool, the engine and the frontends in that order, and supervises them: it is the only process a supervisor signals, and it runs drains, stops and the engine's watchdog ([Health, drain and restarts](operations.md#health-drain-and-restarts)).
- **The engine** loads the model, sizes the pools from the runner's report, warms up, and then runs the step loop. Its one thread stamps a heartbeat in shared memory every loop turn, which the watchdog reads.
- **Frontends** (`server.num_frontends`) serve HTTP. Each binds its own socket on the server's port with `SO_REUSEPORT`, so the kernel spreads connections across them, and each has its own link to the engine. A frontend starts, with the engine's effective configuration, only once the engine is warm, so nothing is routed to a server before it can serve.
- **The image pool**, with images on, decodes and resizes images on the CPU ([The image pool](#the-image-pool)).

### Frontend and engine messages

Every frontend pushes into the engine's one ZMQ `ipc://` input socket (the launcher too, for `Shutdown`), and the engine pushes each frontend's outputs into a socket of its own. Messages are msgspec Structs (`array_like`, `gc=False`) behind a one-byte type frame; numpy arrays of 256 bytes or more travel as zero-copy frames. The frontend reads engine outputs from a callback on the socket's file descriptor, not through `zmq.asyncio`, which keeps the first token's path short.

- **Frontend to engine:** `AddRequest` (a request's token ids, parameters, image references, cache keys and grammar), `Abort` and `Ack`.
- **Engine to frontend:** `StepOutputs`, at most one per step for each frontend with results in it: each request's new tokens (and their logprobs, if it asked) and finish reason, the cached-token count on its first output, the engine's statistics, and the requests whose pixels the frontend may free.
- **Pixels never travel over ZMQ.** The image pool writes them into a shared-memory arena, and a request carries handles into it.
- **Backpressure:** the engine never blocks on a frontend. A frontend acknowledges the outputs it has processed, and one that falls too far behind gets no new admissions until it catches up; nothing is dropped.

### Who finishes a request

- **The engine** finishes a request for everything that needs no text: EOS, stop token ids, `max_tokens` or the context, and a grammar's end. It also ends a request it rejects at admission or one that fails ([Failures and rollback](#failures-and-rollback)), and aborts every request when it shuts down. It frees the request's resources in the same step.
- **The frontend** finishes it for stop strings, client disconnects and its own failures (a drain that timed out, the engine gone), and sends `Abort`; a request it ends while preparing it is never sent.
- A finish the engine detects wastes at most the one step already launched; a finish the frontend detects also wastes the steps launched before its `Abort` arrives, usually one or two.

## The step loop

The engine's loop runs one step ahead of the GPU:

```
loop:
  poll the frontends (non-blocking): new requests, aborts, acks
  schedule step N+1 → StepPlan (struct-of-arrays: pages, state ops, checkpoints)
  runner.launch_forward(N+1)          # small host-to-device copies; graph replay or eager
  wait for step N's tokens            # an event on the copy stream: the host waits here while N runs
  process N: request state (vectorized), grammars, finishes, frees, decode checkpoints
  send StepOutputs(N) to the frontends
  if N+1 has constrained requests: fill their grammar masks from N's tokens (llguidance, GIL released)
  runner.launch_sample(N+1)           # a split step's sampling and drafts (a fused step's ran in its forward)
```

When there is nothing to run, the loop blocks on the ZMQ poll, which also watches a pipe that the end of a grammar build writes to, so the loop wakes at once; while builds are its only work it blocks instead of turning. On Linux the engine thread is pinned to one core and each frontend to one below it. The engine and the frontends spin-poll while requests are in flight and for a short time after, so a request that arrives during traffic does not wait for an idle core to wake up.

**The cost of staying ahead** is at most one step of time to first token: a request that arrives while step N runs is planned into N+2. A finish costs at most one wasted step, and the scheduler avoids even that where it can: a request gets another row only while the rows in flight cannot reach its `max_tokens`, and a step that would serve only grammars about to end is skipped ([Structured output](#structured-output)).

### Planning at the forward/sampling boundary

When the step in flight is split into a forward and a sampling part ([CUDA graphs](#cuda-graphs)), `engine.plan_at_boundary` makes the loop first poll until that step's forward pass has ended on the device. A request that arrives during step N's forward then joins N+1 instead of N+2, and N+1's launch overlaps N's sampling part instead of its forward. Only which step a wall-clock arrival lands in changes. The price is a smaller margin: the host must plan and launch N+1 within N's sampling part, so a stall of the engine's thread can idle the GPU. It suits streaming clients. It needs Gemma 4 with structured output: the Qwen runner reports no forward's end, because the eager GDN and attention breaks of its prefill and mixed steps take milliseconds of host time, which the step in flight's outputs would wait for.

## The scheduler

**One token budget per step.** Each step serves running requests first, then admits waiting ones, within `scheduler.max_num_batched_tokens` tokens and `scheduler.max_num_seqs` running requests. Prefill chunks and decode rows mix in the same step, so long prompts never stall the decoding requests. A prefill counts the positions scheduled or served from the cache, which covers chunks, prefix hits and readmissions (the prompt and the tokens generated before the preemption).

**The mixed-step bound.** A step with n decode rows and P prefill tokens delays each decoding request by about c·P, where c is the prefill cost per token. So such a step takes at most K / n prefill tokens (at least one checkpoint grid, so prefills progress). A prompt costs the requests decoding beside it the same total however it is chunked, so the bound only spreads that cost: few decoding requests may wait through a long prefill (a burst's first steps), many through a short one (steady load). K comes from the prefill cost that the memory profile measures, so that a step's prefill delays its decoding requests by 3.5 s in sum (`MIXED_STEP_DELAY_MS`): as a time, the ranges that suit the two models overlap; as a token count, they do not. A verify row counts once, and a PRECISE token as its measured cost in FP8 tokens. A Gemma 4 image span and a refine's piece go whole into one step, past the bound.

**Admission, priority and preemption.**
- First come, first served by default. With `scheduler.policy: priority`, a request's `priority` orders the waiting queue (lower values first) and picks the running request a page shortage preempts first (the highest value, then the latest arrival). A waiting request never preempts a running one, and a step that preempted, or whose running prefill waited for pages, admits nothing new.
- Admission is cache-aware: siblings that share a prefix nobody has computed yet wait one step, so the first one fills the cache and the others hit it. A request waits when the plan under construction writes a deeper checkpoint on its path, a precise copy at least as deep as its hit, or the KV its branch point lies in; each such wait gives the next plan a deeper or precise hit, so waiting ends. A would-be refiner also waits while another request refines its checkpoint.
- Preemption is by recompute, cheap with the prefix cache, and exact: a preempted request donates its KV and live state to the cache as a finish does, a sample still in flight reaches it as a normal output, and its readmission computes only what is missing.
- Preempted requests keep their arrival order when readmitted, so the same request is not preempted again and again.

**Chunk boundaries follow the cache.** A prefill chunk never splits for block alignment, because checkpoints are taken inside the pass. Among checkpoint targets, only a client cache hint off the chunk's grid and a refine's end cut a chunk, with an exact checkpoint there. Other targets (the end of a prefill, a branch point) go on the grid of the chunk that contains them, less than a grid below the desired position, which saves a conversation one step per turn. Budget cuts round down to the absolute grid while targets lie beyond them. [Batch invariance](#batch-invariance) cuts hybrid models' prompts on the grid.

**Encoder scheduling.** An image is encoded in the step whose chunk reaches it, within a per-step encoder budget. Each image is reserved in the encoder cache as the chunk is planned; if the chunk then fails to get its pages, the reservation rolls back. A running request keeps the images of the spans it has not finished. A request whose next image finds the cache's rows taken gets no chunk, so it asks for no pages and preempts nothing, and a chunk preempts only requests behind it: when a request behind it holds those rows and its chunk waits for pages, the two wait for each other. Such a wait usually ends by itself (a decode finishes, a step completes), so the scheduler breaks it only when a plan comes out empty, preempted nothing and has no step in flight, which makes every later plan the same: the waiting chunk's request then gives its images back, and its next chunk takes them again or encodes again what was evicted meanwhile, as a readmission does. Gemma 4's image spans attend bidirectionally in its sliding layers, so every span stays whole in one row: a chunk never ends inside a span, and neither a checkpoint nor a hit lies inside one (a hit backs off to the deepest checkpoint at or below the span's start).

## The step plan

Each step the scheduler builds a `StepPlan`: numpy arrays in struct-of-arrays layout, handed to the runner by reference. `llm_mini_infer/sched/plan.py` and `llm_mini_infer/runner/base.py` define the contract; its rules:

- **Only plans change the runner's slot tables** (page-table rows, live state slots, sampling parameters and seeds, commit limits), apart from the device's own advance of the committed length `seq_len` and the pending token at each sample. Updates are stream-ordered, so they never race a forward pass in flight.
- **Decode input stays on the device.** A decode row reads its token from the slot's pending token (and its drafts) and its position from `seq_len`, which it advances by one plus its accepted drafts; the scheduler mirrors `seq_len` from the outputs and reserves pages for the steps in flight.
- **Results** are matched by their position in the plan, never by slot.
- **Visibility.** Step N+1 may use prompt KV and checkpoints that step N writes before N's tokens are known, which lets a sibling with the same uncached prefix start one step later.
- **Sampling noise** depends only on (seed, position, token) ([Sampling](#sampling)).
- **Speculative rows** verify `1 + k` tokens and return up to `1 + k`, never past the request's length limit or its first 8 stop tokens (the scheduler drops what follows any other).
- **Grammar masks** split a step: `launch_forward` runs the forward pass, the engine fills the masks, and `launch_sample` applies them and samples.

The fake runner (`llm_mini_infer/runner/fake.py`; its checks are in `runner/fake_checks.py`) checks every plan against the slot tables it keeps and raises on a violation; in strict mode it also checks the token at every KV position. Its tokens depend on the whole context, so a result matched to the wrong request, or state reused for the wrong prefix, changes outputs that tests predict.

## The cache

llm-mini-infer's prefix cache is a radix tree over token ids. It holds KV pages and checkpoints: on hybrid models recurrent-state slots, in a pool of their own; on sliding-window models windows of KV pages from the same arena as the full pages.

### Pools

- **KV pages.** Fixed-size pages (16 tokens by default) across all full-attention layers, Qwen3.8's MTP layer included. Pages are reference-counted, and the count is the single source of truth: a running request holds one reference per page-table entry, the tree one per page it keeps. The pool keeps no LRU of its own; pages only the tree holds are reclaimed by evicting tree nodes.
- **State slots** (hybrid models). Each slot holds the conv and recurrent states of every linear-attention layer for one position, plus the MTP drafter's input hidden vector. Page size never depends on state size.
- **The split.** Live slots first, one per running request; then, by default, the checkpoint budget, `min(max_num_seqs, the slots that fit in 25% of the memory left)`, none with prefix caching off; then KV gets the rest. A running request owns exactly one full-size state, its live slot: checkpoints are written straight into checkpoint slots, and a request that finishes, is aborted or is preempted donates its live slot. What the KV pool must hold is in [operations.md](operations.md#the-prefix-cache).

### Sliding-window models

Gemma 4's sliding layers need only each request's last `sliding_window` positions. Their KV lives in **window pages** that are released as soon as a row's window has passed them, and the full layers' KV lives in **full pages** shared through the prefix cache.

- **One KV arena.** Both kinds of page have equal bytes: a window page holds a page size of positions (16) of every sliding layer, and a full page as many positions of the full layers as fill the same bytes (320 on Gemma 4 31B). One free list and one reference count serve both, so the split follows the load and nothing fragments. Window pages keep FlashInfer's per-layer layout. A page reused as the other kind loses its contents.
- **Window checkpoints** play the hybrid state checkpoint's role: the request's own window pages, held by reference, taken at the same points. A hit shares pages of both kinds and copies the partial last page of each.
- **Under pressure,** in order: the cached windows of unlocked nodes (least recently used first), then cached full pages, then duplicate tree pages, then the windows on paths that running requests hold, and only then a preemption. In each pass, demoted windows go first ([Eviction](#eviction)). Unlocked windows go first because they are most of the cache's bytes. Windows on held paths go last because evicting them frees almost nothing (their pages are shared) and forces the shared prefix to be recomputed.

### The radix tree

- **Keys** are token ids. An image's span is keyed by its content: every position of the span carries 62 bits of `blake3(bytes, resize target)` above `2^62`, while the model sees placeholder ids. A node also keeps the full 256-bit digest of each span its writer had, and a match or an insert compares it with the request's, so a hit never crosses a different image, even one whose 62 bits collide (a crafted pair costs about 2^31 hashes). `cache_salt` namespaces the tree: requests with different salts (or none) never share an entry.
- **Nodes** cover key positions `[start, end)`. A node holds the KV pages that end inside it, a partial tail page, and at most one checkpoint, valid at the node's end.
- **Hits.** A model with full attention alone hits at its visible KV match, rounded down to a page. A hybrid model hits at the deepest node with both KV and a state checkpoint, a sliding-window model at the deepest with both KV and a window. Either way the hit stops one token short of what the admission prefills, so that token is computed and can be sampled from.
- **Visibility.** Every entry is stamped with the step that writes it, and a plan uses only entries from earlier steps. Entries the plan under construction writes are reported as pending, and a sibling waits one step for them.
- **Duplicate inserts.** A request that inserts a span the tree already holds gives the tree its pages wherever only the tree held its own, and under page pressure running requests' duplicate pages are adopted too, so a lone request never holds a prefix twice and blocks itself.

### Checkpoints

A hybrid model can resume only where a state checkpoint exists, and a sliding-window model where a window does. Checkpoints lie on a grid: 64 tokens on hybrid models (the GDN prefill kernel's chunk size), the window page on Gemma 4. They come from six places:

1. **The end of a prefill:** one token short of what the admission prefills, on the grid of the chunk that contains it, so a repeat of the same prompt hits.
2. **A branch point:** when a request's KV match is deeper than the deepest checkpoint on its path, its prefill takes a checkpoint where the prompts diverge.
3. **A client hint:** exact, where the marked message ends ([Cache hints](#cache-hints)).
4. **During decode:** every `dev.cache.decode_checkpoint_interval` positions; a running request keeps only its latest.
5. **At finish, abort or preemption:** at the last computed position, with no copy. On a hybrid model the live slot swaps kinds with a free checkpoint slot (evicting a checkpoint, in the order of [Eviction](#eviction), if none is free); on a sliding-window model the request's window pages become the window. The next turn of a conversation resumes there.
6. **A refine's end:** exact, in the precise copy ([The precise prefix](#the-precise-prefix)).

**Checkpoints inside a step.** A step's mid-chunk checkpoints cut their rows into consecutive calls of FlashInfer's GDN prefill, each checkpoint a segment's final fp32 state. The cuts lie on the 64-token grid from the row's start, where the per-sequence route computes the same chunks as one call ([Batch invariance](#batch-invariance)). FlashInfer 0.7.2's checkpointing variant, which emits a state every 64 tokens in one pass, is not used: on SM120 its outputs and states move between identical calls (fixed in 0.7.3, flashinfer-ai/flashinfer#7785). End-of-chunk and end-of-step checkpoints copy the live state slot.

**Speculative decoding and hits.** Qwen3.8's MTP layer's KV at position i depends on token i + 1, so on a hit of length L only its entry at L − 1 is stale. A checkpoint also stores the target's last hidden vector, so the drafter recomputes just that one position. Gemma 4's assistant writes no KV, so its hits need no repair.

### Cache hints

`cache_control` is an extension of the chat API, in Anthropic's shape ([the API](../README.md#the-api)).

- Without hints, a hybrid or sliding-window model resumes only at a checkpoint that a prefill's end, a branch point, a decode or a finish wrote. A hint adds an exact checkpoint at the end of the marked message, so the next request that shares that message resumes right after it. Off the chunk's grid, the scheduler ends the chunk there.
- The end of a message is found by rendering the conversation up to it. For templates that cannot render that prefix alone, a probe user turn is appended and the message ends at the special token that opens it. A hint that does not fall on a token boundary of the prompt is ignored. Of more than 4 hints (Anthropic's limit), the deepest 4 count.
- A hint never changes the prompt, but its cut can move the prompt's rounding, as any chunk boundary does. Entries live until eviction.

### Eviction

Cached entries are evicted before any running request is preempted.
- **Pages:** least recently used among unlocked leaves, including leaves whose writing step is still in flight. A leaf takes its checkpoint with it.
- **Checkpoint slots** (hybrid models): when a new checkpoint needs a slot and none is free, a demoted checkpoint first, then the least recently used, locked or not; their KV stays. The plan under construction protects the checkpoints it writes and those its `STATE_COPY`s read. Windows go under page pressure ([Sliding-window models](#sliding-window-models)).
- **Demoted checkpoints** are ones their conversation has moved past. Each time a request leaves a deeper checkpoint of its own on its path (an end-of-prefill or branch checkpoint as its chunk is planned, a decode checkpoint, the one at its finish or preemption), it demotes the checkpoints above it on that path: the one it resumed from, its branch checkpoints and its latest decode checkpoint, and its end-of-prefill checkpoints if it continues where an earlier request stopped (it resumed from a finish, preemption or decode checkpoint, or its prompt runs through the end of a cached path): a first request's end-of-prefill checkpoint may be the prefix its siblings resume from. Stay live: a checkpoint that another request resumed from since it was attached (for the one it resumed from, more than once), a client hint's, a refine's end and the precise copies, another request's preemption checkpoint, and one below which the tree forks before any deeper checkpoint. A demoted checkpoint is not dropped: it is evicted before every live one, and a hit makes it live again. `dev.cache.demote_superseded` (true) turns the rule off, for traffic that resends its prompts more than it extends them.
- **Locks** keep KV and defer the eviction of their windows. They are a retention policy, not a safety mechanism: pages stay allocated while any request references them, and every plan effect is stream-ordered.
- **Preempted requests** leave their KV and donated state in the cache, marked recently used.
- Entries still being written stay evictable, or a request short of pages could preempt itself every step.

### Failures and rollback

A bad row fails its request and the running requests that resumed from that request's writes still in flight.
- A request's cache writes stay **unverified** until its next finite sample. A non-finite result, an abort mid-prefill or a readmission that resumes elsewhere removes them, with the checkpoints attached to them and the entries below. What the request only read stays. A running request that only locked a removed node runs on but caches nothing more.
- Both runners check the logits of sample rows. Gemma 4's runner also checks the last hidden state of a row that does not sample. On FP8 rows the activation quantizer turns a NaN finite, so such a NaN goes unseen.
- The node a hit ends in, which other requests branch from, never carries the step of a request that only resumed from it, so it never rolls back with that request.
- A non-finite row or an empty grammar mask fails its request with a 500 (on a stream, an error event); the server goes on. A request preempted while such a sample was in flight does not fail then: the scheduler drops the sample, and the readmission samples that position again.

### The precise prefix

For block-FP8 models, `runner.cuda.fp8_cold_prefill` decides how a shared prefix is computed. A prefix that many requests share, such as a long system prompt, is computed once and every later request resumes from the cached copy. FP8 activation quantization turns differences below BF16's precision (where a chunk boundary falls, how attention splits its work) into different roundings, so an FP8 copy carries one draw of that rounding noise, and the draw tilts every answer that reuses it the same way.

- **`reuse` (the default):** prompts run on the FP8 path. Once a cached FP8 checkpoint has served `dev.cache.refine_after` resumes, the next request that resumes from it recomputes the span from the deepest precise copy on its path (or position 0) with BF16 activations against the dequantized FP8 weights, into a precise copy, and later requests resume from that copy. A span shorter than `dev.cache.refine_min` is not refined: a step with PRECISE rows runs eagerly and dequantizes every weight, which pays only for a long prefix.
- **The first reuse is free** because a conversation resumes each turn from the previous turn's checkpoint once: refining there would make every turn recompute the previous one first. A shared prompt is resumed from again and again, and is refined at its second reuse.
- **Two key spaces.** The precise copy lives in a key space beside the FP8 one. An admission matches both and resumes from the deeper hit, the precise one on a tie, or from the precise one when the FP8 hit lies less than `dev.cache.refine_min` above it and either was computed from the FP8 copy or lies less than a checkpoint grid above it. That span then runs on the FP8 path, and the precise copy stays locked. No page is overwritten. Requests that resumed from the precise copy insert their continuations into the FP8 key space, which then takes the precise copy's pages wherever the tree alone held the FP8 copy's.
- **Refines.** Two refines of one checkpoint never overlap. A refine is cut in fixed pieces at multiples of `dev.cache.refine_chunk_tokens` from position 0, each whole in one step, so every server start cuts the precise copy at the same boundaries whatever the load.
- **PRECISE rows** share steps with FP8 rows, which keep their bits. They delay the step's other rows too, so each PRECISE token counts at its measured cost against the mixed-step bound.
- **`bf16`** runs every chunk of a prompt without a cache hit with BF16 activations, into a precise copy. A prompt with any hit resumes from the deeper copy, the precise one on a tie, and runs the rest on the FP8 path. **`fp8`** quantizes every activation.

## Model execution

### The CUDA runner

The runner (`llm_mini_infer/runner/cuda/`) owns the model, the device pools and the CUDA graphs, and executes step plans. The Qwen runner (`runner.py`) handles the hybrid models. Gemma 4's runner (`gemma.py`) has its arena of window and full pages and its assistant drafter, and no state slots or GDN; it reuses the Qwen runner's staging, static input block and MTP verification. The two share the rest of the package: the graph capture, the FP8 path, the image encoder, the grammar-aware heads, the draft head and the sampler.

- **Input preparation on the GPU.** The per-slot tables live on the device; each step's inputs are built there from small host arrays copied in (pinned, double-buffered); device-to-host copies run on a copy stream. The step path waits on the device only for a page-locked buffer whose last copy has not run: the vision tower's attention with more than four encoder calls in flight (`FlashInferVisionAttention` counts those waits), and a draft head's rebuild.
- **The host mirror.** FlashInfer plans attention from host-side lengths. Without drafts every decode row advances by exactly one, so the runner keeps an exact host mirror of `seq_len` and never syncs. With drafts the accepted lengths are known only on the device, so attention is planned for the upper bound and the kernels read the true lengths on the device. A split-KV plan whose chunk count could change with the true length is made again with chunks whose count cannot; a decode graph whose captured grid fits no such plan keeps its first plan, and those rows may attend wrongly (logged and counted). A startup self-check verifies the FlashInfer internals this reads (`llm_mini_infer/kernels/flashinfer.py`).
- **Startup.** The runner loads the weights, profiles memory with synthetic full-budget steps, sizes the pools from `runner.cuda.memory_utilization`, runs one eager warm-up step per kernel variant, and captures the graphs before the server reports ready. Graphs are captured with Python's garbage collector off, and cuBLAS's handle is created before the first capture.
- **On a CPU** the same runner runs with the torch reference kernels and float32, which is how the tests run model code without a GPU.

### Models

Plain `nn.Module`s that get their batch metadata from a per-step context (`models.qwen3_5.backend.ForwardContext`, Gemma 4's `Gemma4Context`). One subpackage per architecture; `models/common.py` holds the helpers both use. transformers is never imported on the serving path: configurations are read as JSON, and the chat template environment and the tokenizer are loaded directly.

- **Gemma 4** (`llm_mini_infer/models/gemma4/`: `model`, `vision`, `assistant`, `backend`): the dense 31B instruction model. Five sliding layers (head dim 256) come before each full layer (head dim 512), whose K and V come from one projection (`attention_k_eq_v`). The KV pool stores only V for the full layers; the attention kernel derives K from it per key, which halves those layers' KV memory and reads. Its assistant drafter is a separate checkpoint (`speculative.model`).
- **Qwen3.8** (`llm_mini_infer/models/qwen3_5/`: `model`, `vision`, `backend`): the hybrid Qwen3.5 architecture (`Qwen3_5ForConditionalGeneration`), in which every fourth layer is gated full attention and the others are Gated DeltaNet (GDN) layers, plus the checkpoint's MTP layer.
- **Numerics.** The text models follow vLLM's CUDA path, rounding where it rounds, so differences between the two engines are limited by batch noise and not by rounding choices, except in Qwen3.8's prefills of few sequences, where the reference lets FlashInfer's GDN prefill pick its route (on SM120 the context-parallel one) and this engine forces the per-sequence one, so the two can round differently; the vision towers follow transformers'. Each fused kernel rounds where its torch reference rounds and only the order of its reductions differs, but for the GDN decode, verify and fold kernels, which keep that engine's arithmetic (fast exp).

### CUDA graphs

- **Full graphs** for decode-only steps, bucketed by batch size up to `max_num_seqs`, with a uniform `1 + k` query length under speculative decoding. They hold everything from input building to the sampler and the length advance, verification and the drafter included. Attention, FlashInfer's and ours, is planned outside the graph from the host mirror into fixed buffers.
- **Breakable graphs** for prefill and mixed steps, bucketed by token count up to `max_num_batched_tokens`. Attention and the GDN mixer run as eager breaks between captured segments. A break holds only weak references, so the graph pool does not pin its buffers.
- **Split graphs.** A step with grammar masks runs a forward graph, then a sampling graph that applies the masks; the sampling graph reads the forward's hidden states from a copy and runs the same kernels at the same shapes, so its tokens are the fused graph's bit for bit. With structured output on, both runners run every step this way (`dev.runner.cuda.always_split`) and capture no fused graphs, which saves capture time and graph memory; on Gemma 4 it also gives every step the forward-end event that [planning at the boundary](#planning-at-the-forwardsampling-boundary) needs.
- **Programmatic dependent launch** (`dev.runner.cuda.pdl`): the step's kernels of ours and the CUTLASS GEMMs launch with the programmatic-stream-serialization attribute, so a kernel may start while the one before it ends. Each waits for its predecessor (`griddepcontrol.wait`) before it touches memory the predecessor wrote, so outputs equal the serialized launches' bit for bit. FlashInfer launches this way by its own default and reads a row's KV length before it waits, so right after the device rewrites those lengths its attention runs serialized.
- Padding rows use slot 0: the null page, the null state slot, greedy sampling, never reported.

### Kernels

| Function | Kernel |
|---|---|
| Attention | FlashInfer for head dims up to 256 (prefill, decode, Gemma 4's sliding window). Gemma 4's full layers (head dim 512: FlashInfer's SM120 path for it would need a separate V plane) run `attn512`, our split-KV kernel with BF16 tensor cores that builds K from the stored V in registers |
| Batch-invariant attention | `attn_window`, our kernel for Gemma 4's sliding layers and, with no window, Qwen3.8's full layers ([Batch invariance](#batch-invariance)) |
| GDN prefill | FlashInfer's chunked GDN kernel on its per-sequence route at every step (its context-parallel route is no more accurate and slower at the default budget's chunk sizes), reading and writing each row's state slot in place |
| GDN decode and MTP verify | vLLM's fused GDN decode kernel, adapted into our library, and our verify kernel on its arithmetic |
| BF16 thin GEMMs | our small-M kernel, cuBLAS beyond: Qwen3.8's `in_proj_ba` up to 8 rows; Gemma 4's BF16 projections and head up to 2 rows |
| Block-FP8 GEMM | vLLM's CUTLASS SM120 blockwise kernel (FP32 scales), dispatched among three tile configurations by M and N; every configuration gives the same bits at every M |
| Norms, RoPE, conv, activations | our kernels: the norms (with the residual add, gated, Gemma 4's sandwich), the gated activations, QK-norm with RoPE and the paged KV write in one launch, the causal conv, the GDN prefill's input preparation |
| FP8 activation quantization | fused into the op that produces each FP8 GEMM's input; Gemma 4's attention output is quantized by a kernel of its own |
| Sampling | our fused counter-based sampler ([Sampling](#sampling)) |
| Vision attention | FlashInfer's ragged non-causal kernel, head dim 72 padded to 128; each tower's attention input (2-D RoPE, padding) is one kernel of ours |

**Our kernel library** (`llm_mini_infer/kernels/`) is plain CUDA with `extern "C"` launchers, built by nvcc alone into two shared libraries loaded with ctypes (the fused ops, and the CUTLASS GEMMs). Every kernel has a torch reference that defines its numerics and runs without a CUDA toolkit. Its build and cache: `llm_mini_infer/kernels/build.py` and [the environment variables](operations.md#environment-variables); the image's prebuilt copy: [development.md](development.md#building-the-image). Our attention kernels keep a split's partial outputs in fp32 and round once, where FlashInfer's paged prefill merges them through BF16.

### FP8

Block-FP8 checkpoints (one scale per 128 × 128 block) load through one path for both formats: Hugging Face's `fp8` method (Qwen3.8) and compressed-tensors `FP8_BLOCK` (Gemma 4). Every projection the checkpoint quantizes is an FP8 GEMM; embeddings, norms, the vision towers and the KV cache stay BF16. Activations are quantized dynamically per token and per 128-group by the kernels that produce them. The GEMM library needs an SM 12.x GPU; on a CPU the torch reference runs.

### Sampling

- **Counter-based Gumbel-max.** The noise depends only on (seed, position, token), never on the batch; its hash is exact integer arithmetic, so CPU and CUDA draw the same uniform value (their fp32 logs may round differently). A seeded request's tokens depend on its batch only through its logits ([Batch invariance](#batch-invariance)). Ties go to the lowest token id, and top-p's mass is summed in fp64.
- One fused kernel samples every row of a step for greedy and every filter, inside the graph. It matches the torch reference token for token, unless a top-p sum lands within about 1e-13 of its threshold.
- A request without a seed (or with -1) draws one from a generator seeded by `scheduler.seed`.
- **Logprobs** (`server.logprobs`, off by default) are vLLM's default `raw_logprobs`: from the logits the sampler read, before temperature and the filters, with the grammar's mask applied, over the whole vocabulary. An eager pass after the step's sampling computes the values of the rows that asked, so no step that asks for none runs another kernel. The sparse head's values are the dense head's, since its union holds every token the mask allows; under MTP each delivered token is scored at its own verify position. The forced prefix's tokens and the EOS-free finish's EOS have a mask that allows them alone: their logprob is 0. `runner/cuda/logprobs.py` and `output/logprobs.py` hold the details.

### Speculative decoding

Chain drafts of length k (`speculative.num_speculative_tokens`), verified in every step's body as device-input rows of `1 + k` queries:
- **Gemma 4's assistant** (`speculative.model`, `google/gemma-4-31B-it-assistant`): four query-only layers that read the target's KV, so they write no KV of their own. k is at most 7. Every draft step's query sits at the last committed position, as vLLM does (`dev.speculative.draft_query_next` puts it one later).
- **Qwen3.8's MTP layer:** one extra decoder layer of the checkpoint, fed the target's final hidden state. Its KV is part of the radix KV. On a GPU k is at most 3: the verify kernel holds the window and the fold, 1 + k tokens each, in 8 tokens.
- **Verification with the sampler's own noise.** A draft at position p is sampled with the same noise as the target's sample at p, and drafts are accepted while they equal the target's samples. The output is then exactly the target's sampled sequence: a seeded request gives the same tokens with drafting on or off, up to numeric noise. What changes the drafts changes acceptance, never which token the target samples.
- **GDN states under speculation.** A verify kernel folds the slot's previous commit into the live state (a replay of the accepted tokens' records), computes its window's outputs from that state in registers, and records the window's inputs for the next fold. So the state is read and written once a step, and a rejected draft costs no state slot.
- **The drafter's head** (`dev.speculative.draft_vocab`): by default Qwen's MTP layer scores a ranked subset of the vocabulary, re-ranked from the tokens the server emits, and Gemma 4's assistant the whole vocabulary (a cuBLAS GEMM at every M: a draft head's numerics change acceptance only). A token outside the subset is not drafted, which costs acceptance, never a wrong token. On the steps the [grammar-aware heads](#grammar-aware-heads) serve, either drafter scores the grammar's universe instead.

## Structured output

- **The grammar.** `response_format` (`json_object`, or `json_schema` with or without `strict`) becomes a grammar in the language of `structured_output.backend`. The frontend checks the schema against a subset of JSON Schema with caps (`grammar/bounds.py`, [the subset](vllm-compatibility.md#structured-output)), then with llguidance. The caps bound what llguidance's compile costs (guidance-ai/llguidance#436, #438). So an invalid schema is a 400 before anything runs, and so is one of more than `structured_output.max_schema_bytes`. The engine compiles the grammar against the tokenizer, caches it by its text and clones it per request.
- **Fixed text.** Under `guidance`'s canonical forcing, llguidance scans the bytes a grammar fixes ahead of the current state, at every mask and in the forced prefix (guidance-ai/llguidance#440). So the frontend counts a schema's longest run of fixed output from the schema (`grammar/forced.py`), and refuses one past 4,096 bytes. At that length, the first mask after a choice that enters the run takes under 3 ms on Gemma 4's tokenizer. The cap holds under every backend, so a schema builds under one exactly when it builds under the other.
- **Where it compiles.** The frontend builds and checks an uncached grammar off the event loop, on one of two threads with a large stack. It builds each once, however many requests need it. The engine builds on its two compile threads, one build per grammar and forced-prefix choice in flight, shared. A request whose grammar misses the cache waits COMPILING, outside the waiting heap, while the step loop goes on and later ready requests overtake it. The engine thread settles a finished build at the top of a turn and in the boundary wait (the caches and counters are touched only there), and the request then waits with its arrival number. A failed build, or one past `dev.structured_output.compile_timeout_s`, fails the request; a build is never stopped, and its end still fills the cache. The check compiles over a small byte tokenizer, so it meets llguidance's limits as the engine's compile does.
- **Two JSON languages,** chosen by `structured_output.backend`. `guidance` is llguidance's JSON grammar with canonical forcing, ending on the tokenizer's EOS token; with `structured_output.disable_any_whitespace`, compact JSON. `auto` (the default) and `xgrammar` emulate xgrammar's JSON language through llguidance (any whitespace, no forcing), ending on the model's stop tokens. The engine refuses to start when the model's stop tokens miss the grammar's EOS, since every finished document would then be a grammar error.
- **Masks.** llguidance fills a step's masks while its forward pass runs, after the step before it is processed ([The step loop](#the-step-loop)), with the GIL released and on one thread by default (`dev.structured_output.threads`: requests that share a grammar get no faster on more). The sampling graph applies them. A verify row gets `1 + k` masks, along its drafts.
- **The end of a grammar.** A complete JSON document's grammar allows only the EOS. When every request the next plan would decode has a row in flight that could deliver that EOS, and no request waits or prefills, the plan skips those rows, so a request that arrives meanwhile does not queue behind a wasted step.

### Grammar-aware heads

Gemma 4's vocabulary has 262,144 tokens and Qwen3.8's 248,320, and a grammar mask usually allows a few. With structured output on, on both models:
- **The sparse head** (`dev.runner.cuda.sparse_head`): on a step whose every sample token has a mask, the host ORs the step's masks into their union, and when it holds at most `dev.runner.cuda.sparse_head_rows` tokens, the sampling gathers only those rows of the target's LM head (BF16 in the FP8 checkpoints too) and samples over them, reading each row's mask through the token ids. Every token a mask allows is scored, so the masked choice is the dense head's up to the GEMM's rounding. Any other step takes the dense head.
- **The draft universe** (`dev.speculative.draft_universe`): the runner keeps the OR of the unions of every such step, and on those steps the drafter (Gemma 4's assistant, or Qwen3.8's MTP layer in place of its ranked subset) scores only that universe. A draft outside it can never equal the masked target sample, so acceptance cannot fall once the universe has converged. Past `sparse_head_rows` tokens it turns off for good.
- **The forced prefix** (`dev.structured_output.forced_prefix`): the tokens a grammar forces at its start (compact JSON's `{"` and a fixed first key) go into the request's final prefill chunk as decided output tokens, instead of being sampled one step at a time. The chunk takes as many as `max_tokens`, the step's budget and free pages allow. A hybrid model's prefill takes a checkpoint inside a row only on its 64-token grid, so there the forced tokens join the final chunk once every checkpoint of the step is placed, and a chunk whose prompt end carries a checkpoint off the grid samples them instead.
- **The EOS-free finish** (`dev.structured_output.eos_free_finish`): once a request's grammar is complete and only the EOS may follow, the request finishes with the EOS appended and counted, without a step to sample it, unless it is the only request left in the step in flight.

The forced prefix needs a grammar with canonical forcing, and the EOS-free finish one that ends on one EOS id: `guidance` grammars, and on Gemma 4 and Qwen3.8 (several stop tokens) not `auto` or `xgrammar` ones. Neither applies to a request with stop strings, whose finish the frontend counts.

Outputs, token counts and finish reasons are the same with each of these on or off: exactly on a CPU, and on a GPU up to numeric noise. The forced prefix's positions and the first sample after them run in the final prefill row instead of decode or verify rows, whose kernels round otherwise, even under batch invariance. Without batch invariance, any change in acceptance or in a step's rows also changes rounding.

## Tool calls

- **The format** is `tool_calls.parser`'s: Gemma 4's `<|tool_call>call:NAME{...}<tool_call|>` or Qwen3.8's `<tool_call>` XML, as their chat templates write calls. By default, the model family's format, when its template renders it.
- **The prompt.** The tools reach the template normalized (an absent `description` or `parameters` as null, other keys dropped), for every `tool_choice`. A tool message's own text is neutralized: a control token spelled there becomes the ordinary tokens of its spelling. A second render, with that text masked, locates it in the prompt.
- **The parse** runs in the frontend: one parser per request with tools, on the engine's token ids (`output/tool_calls.py`). A call starts only at its opener token. The parser gets a token only after the stop strings release its text. A streamed call gives its name first, then its arguments [in pieces](vllm-compatibility.md#tool-calls) (`output/tool_args.py`). Each piece extends the last, so the pieces add up to the non-streamed arguments however a step splits the tokens. A value nested deeper than 256 levels stays as written, a string. An exception in one request's output finishes that request alone, with an error.
- **The grammars.** A grammar in the model's own call syntax enforces `required` and a named tool. It is generated from every tool's schema (`grammar/tools.py`) and ends the turn as the model does: Gemma's `<|tool_response>` is part of it. Under `guidance`, a named tool's opening is the forced prefix. Under `auto`, `response_format` with tools takes calls or the schema's JSON.
- A request without tools builds no parser and no tool grammar.

## Reasoning

- **The parser** is `reasoning.parser`'s: Gemma 4's `<|channel>thought\n…<channel|>` or Qwen3.8's `<think>…</think>`. By default, the model family's, when its chat template renders the markers. It reads token ids in front of the tool-call parser (`output/reasoning.py`) and gives the reply's `reasoning`, `content` and calls.
- **Where the output starts** is read from the prompt's end. The prompt closed a thought, or thinking is off; Gemma's turn starts with an optional thought; or the prompt opened one (Qwen3.8's default). `enable_thinking` decides only where the prompt's end says nothing.
- **The count,** `usage.completion_tokens_details.reasoning_tokens`, is the thought's tokens the engine accepted, the markers left out. Tokens a stop string cuts count as the parse counts them whole.
- **The grammars.** With thinking on, a `response_format` or call grammar covers the output from its first token: the thought, then the schema or calls. With thinking off they are unchanged.
- **The request.** `reasoning_effort` reaches the template through `reasoning.effort_map`, and `none` turns thinking off. `include_reasoning: false` leaves the thought out of the reply.
- A request without tools, its prompt in no thought, starts parsing only at a marker: thinking off costs what it did.

## Batch invariance

With `dev.runner.cuda.batch_invariant` these parts give a row the same bits whatever rows share its step. It is on for Gemma 4 when `model.max_model_len` is at most 8,184 tokens, so that a row's keys fit the fixed split's 32 splits of 256, and for Qwen3.8 at the same contexts with a block-FP8 checkpoint. On both it needs a token budget that leaves a prefill room beside `max_num_seqs − 1` decode rows (a token, on Qwen3.8 a 64-token grid, plus a final chunk's forced prefix), pages of a multiple of 16 tokens (our attention kernels read a 16-key tile from one page) and, on Qwen3.8, no chunk cap below the grid.

- **Full attention** splits a decode row's keys at fixed absolute multiples of 256 and takes a prefill row's keys whole, instead of choosing the split by how many rows share the step: Gemma 4's full layers on `attn512`, Qwen3.8's on `attn_window`'s kernels with no window (`FullAttention`). With the switch off, Qwen3.8's run FlashInfer, whose query tile, KV tiles and split follow the batch.
- **Sliding attention** (Gemma 4) runs our `attn_window` kernel instead of FlashInfer's, whose query tile and KV split depend on the batch: K and V read by absolute page, key tiles at absolute multiples of 16, a decode row's keys split at absolute multiples of 256 and merged in order.
- **The GDN prefill** (Qwen3.8) takes every prompt token, a 1-token chunk's too (with the switch off and without MTP, a 1-token chunk takes the decode kernel). FlashInfer's per-sequence route runs one CTA per (sequence, value head), which walks the sequence's 64-token chunks in order with its fp32 state in registers, with no reduction across CTAs; a partial last chunk is masked to add exact zeros. So a sequence's bits are the same alone or packed with others, and the same cut into calls at multiples of 64 from its start as in one call.
- **Chunk boundaries** (Qwen3.8). The GDN prefill counts its chunks from each row's start, so every chunk is a whole number of 64-token grids from its start, unless it ends at the prompt's end or at an exact cut: a hint off the grid, a refine's end (whose PRECISE pieces keep their own cuts), or an image cut (below). A step that leaves a chunk less than a grid gives it nothing. A prompt is then chunked the same way whatever the budget: on the grid from its hit or its last exact cut, which from position 0, or from a hit on a checkpoint a prefill wrote, is the absolute grid.
- **The BF16 `in_proj_ba`** (Qwen3.8's GDN gates) runs `linear_rows` at every M: tensor cores with K cut into a fixed number of slices summed in order. Otherwise it runs our small-M kernel up to 8 rows and cuBLAS beyond, which picks its kernel by M.
- **The sparse head's GEMM** runs at one M, the most sample tokens a step can have.
- **The vision tower** encodes each image in its own call, since a shared call's GEMMs would be sized by every image in it.
- **The forced prefix** goes whole into a request's final prefill chunk: as many tokens as the request decides. A final chunk without room for them beside its prompt's end is cut short of it, and the next chunk takes them.

The FP8 projections need no switch: the default `dev.runner.cuda.fp8_gemm` runs CUTLASS, whose bits are the same at every M.

What stays batch-dependent with it on:
- **The dense head,** which a step takes unless every row it samples has a grammar mask and the masks' union is small enough ([Grammar-aware heads](#grammar-aware-heads)): every step of a request without a grammar, and any step that a constrained request shares with one. It runs at the step's sample rows, on kernels chosen by M.
- **Projections in BF16:** a BF16 checkpoint's, and an FP8 checkpoint's in PRECISE rows ([The precise prefix](#the-precise-prefix)), run at the step's token count, on kernels chosen by M.
- **Preemption.** A readmission recomputes in prefill rows the tokens its request last sampled (under MTP up to 1 + k) and any positions the cache evicted meanwhile, which the first run may have computed in decode rows, with other last bits.
- **An image the step cannot have** (Qwen3.8). When no grid point lies before an image that the step cannot encode, the chunk is cut exactly before it, which moves later chunks off the grid; whether that happens depends on the encoder cache's contents. It needs encoder pressure that the derived budgets leave to large images close together.
- **The drafter** keeps batch-dependent kernels (FlashInfer, the wave split, cuBLAS), so its drafts depend on the batch. That changes acceptance only: verification keeps the target's tokens. On the same tokens the switch does not change acceptance (within 0.01 tokens a verify step on Qwen3.8 27B FP8, 0.03 on Gemma 4 31B FP8, measured with `numerics_forced`: [development.md](development.md#numerics)).

So with an FP8 checkpoint and no PRECISE rows, on steps the sparse head takes, without preemption and (Qwen3.8) without an image cut, a greedy or seeded request's output does not depend on its batch. The prefix cache's contents are another matter:
- a prompt that resumes from an earlier answer's cached KV (computed by decode rows) can get other bits than one that recomputes it;
- on Qwen3.8 so can one that resumes from a checkpoint off the 64-token grid (a hint's, a finish's or a decode checkpoint's): its chunks then count from there, while a hit on a checkpoint that a prefill wrote on the grid replays the same chunks;
- under `runner.cuda.fp8_cold_prefill: reuse` a shared prefix's bits change once it is refined.

Forced on at a longer context, the full layers' fixed split holds about a 32nd of the context instead of 256 keys (at 262,144 tokens, 8,208), so decode rows run fewer, longer splits.

`tests/runner/test_batch_invariance_gpu.py` and `tests/runner/test_qwen_batch_invariance_gpu.py` check it bit for bit on tiny FP8 models (their docstrings tell what they compare). Their golden: [development.md](development.md#the-gpu-suite).

## Images

### The image pool

- **Sizing.** A pool of processes, each with one torch thread. By default, the CPU budget (the cgroup's quota, else the affinity mask; not `os.cpu_count()`) less the cores of the engine and the frontends, between 1 and 8; `images.workers` sets it.
- **Work.** A worker decodes (torchvision for JPEG, Pillow otherwise), applies EXIF orientation, composites a transparent image onto white as vLLM does, and resizes with the uint8 antialiased bicubic resize to the size the frontend computed, which reproduces the reference image processors' pixels exactly. It writes uint8 pixels into the shared-memory arena (in `/dev/shm` where it is writable).
- **The frontend waits for the pool,** so the engine never sees a request without its pixels. The pixels stay in the arena while the request is in the engine, since an eviction from the encoder cache or a preemption may need them again; once it has left and its last step has completed, the frontend frees them.
- **A worker that dies or hangs is replaced** under a new epoch ([what fails](operations.md#health-drain-and-restarts)). The death fails the one image the worker had started, with a 400 (the image may have caused it), and its other jobs are re-sent once to a live worker. Every job carries its epoch, and a worker refuses another epoch's job, so a re-sent image is never decoded twice into a freed arena range.

### The vision encoder and its cache

- The engine normalizes the pixels with the reference processor's formula, patchifies them and runs the vision tower on the GPU, with CUDA graphs per patch-count bucket. The vision tower's RoPE table is computed the same way on every CPU.
- **The encoder cache** keeps image embeddings on the device, keyed by the image's full 256-bit content hash and the request's `cache_salt`; the scheduler owns its bookkeeping and the runner its rows. A repeated image is encoded once per salt.
- **Pluggable per family** (`llm_mini_infer/vision/family.py`): sizing, preprocessing and the encoder sit behind one interface. Qwen3.8 sizes images by pixel bounds (`images.min_pixels`, `images.max_pixels`); Gemma 4 by its soft-token budget (`images.max_soft_tokens`).

## The frontend

### HTTP and streaming

- uvicorn with uvloop and httptools, binding the socket itself (`SO_REUSEPORT`). The application is raw ASGI: no FastAPI or pydantic on the request path.
- Requests are decoded and validated with msgspec Structs: an unknown field is a 400.
- **Server-sent events** are byte templates: a fixed prefix and suffix around the msgspec-encoded text. One event per engine step, so the tokens of one speculative step travel together. The first chunk carries the role only.
- The API is the OpenAI chat API with the extension fields of vLLM's server ([vLLM compatibility](vllm-compatibility.md)); the [README](../README.md#the-api) lists what it honours and refuses. Several choices per request (`n` above 1, `best_of`, beam search) are not planned.

### Rendering and tokenizing

- **Chat templates** render in Hugging Face's own sandboxed Jinja environment, vendored in `llm_mini_infer/render/jinja_env.py`. A template compiles once. Before rendering, the messages are converted by the content format detected from the template's source: a template that prints `content` gets its text parts joined, one that loops over it a list of parts.
- **Tokenizing** uses `tokenizers.Tokenizer.from_file("tokenizer.json")`, as the checkpoint ships it. Long repeated prompt parts are tokenized once: the text is cut at the added token the template renders most, which the tokenizer treats as a hard boundary, and pieces of 512 characters or more are cached. This is exact by construction.
- Both are checked against transformers on corpora of rendered chat-template cases for several model families (`tests/scripts/template_corpus.py`).
- **A budget.** Templates render on the frontend's event loop, and a template's cost grows with the conversation (Gemma 4's is quadratic in messages). A request's renders, its cache hints' included, run at most 2^20 template operations (attribute and item reads, calls), about 2 s on a laptop. Past it, the prompt's render makes the request a 400, and a hint's render drops that hint.

### Detokenizing and stop strings

- `tokenizers.decoders.DecodeStream`, primed with the shortest suffix of 4 to 23 prompt tokens that decodes cleanly (else the whole prompt), decodes each request's tokens incrementally.
- Stop strings hold back the text that could still become a match; a match sends `Abort` to the engine.

## Configuration

One YAML file and command-line overrides (a few named flags and `--set group.key=value`) make a typed msgspec Struct (`llm_mini_infer/config.py`) in three parts: the settings an operator sets, in groups; the `dev` group that tests and benchmarks set; and the server's state, which the hub, `derive()`, the launcher and the engine fill in. The API key and Hugging Face's token come from the environment and never enter the configuration. [operations.md](operations.md#configuration) tells how settings are given, and `config.py` comments each setting.

## Observability

`/metrics` uses vLLM's names and buckets ([vLLM compatibility](vllm-compatibility.md#metrics)), plus an `llm_mini_infer:` family ([operations.md](operations.md#metrics) lists them).

- **Stage timing.** Per-request stage timestamps feed `llm_mini_infer:request_stage_seconds` (its `# HELP` lists the stages). `llm_mini_infer:ttft_overhead_seconds` is a stream's time from its body's receipt to its first byte less the device time of the steps that computed it and its image stages: the host's share, queueing included, an upload through a proxy not.
- **Latency series.** TTFT and e2e end at the frontend's receipt of the first and the finishing outputs. [vLLM compatibility](vllm-compatibility.md#metrics) tells where they start and which series run on the engine's clock.
- **Speculative decoding's counters.** Every output token after a request's first is a verify step's (the speculative-decoding counters' drafts and accepted drafts, those of a row held for a preempted request included), a forced prefix's, the EOS an EOS-free finish appends or a readmission's first sample; every harness run checks it. A finish reports the request's verify steps and their tokens for its log line and `llm_mini_infer:request_tokens_per_verify_step`.
- **GPU idle due to the CPU.** From the runner's device timing per step, the engine counts device idle between consecutive steps when the later one was planned while the earlier was in flight, plus idle between a step's forward and its sampling (`llm_mini_infer:gpu_idle_seconds_total`, against `llm_mini_infer:gpu_busy_seconds_total`). Idle for lack of work is not counted.

## Testing

A CPU suite, a GPU suite on SM120, fuzz campaigns and a plan-diff harness guard the engine: [development.md](development.md#the-cpu-suite) describes them.

**Parity with a reference engine.** FP8 and batch composition move every engine's outputs, so the bars are relative to the engines' own variation, never bit equality. `bench/load/compare.py` measures that ([bench/README.md](../bench/README.md), `tests/scripts/parity_compare.py`). Where an engine's variants share one implementation (the image towers), transformers in fp32 is the reference.

## Where we differ

Each item says what vLLM and SGLang do, and what this engine does instead.

- **Arrivals.** vLLM drains its input queue once per iteration, before it schedules. SGLang's overlap loop polls once per iteration. Neither waits for a forward pass to end. With `engine.plan_at_boundary` (off by default), this engine does wait, so a request that arrives during a forward pass joins the next step ([planning at the boundary](#planning-at-the-forwardsampling-boundary)).
- **Step budget.** Neither engine bounds the product of decode rows and prefill tokens. vLLM keeps its budget in local integers, SGLang in its prefill adder, and both plan the first chunk and the later chunks in separate code paths. Here one `StepBudget` holds the bound, and one chunk planner serves every chunk, so the bound holds for all of them ([the mixed-step bound](#the-scheduler)).
- **Encoder cache.** vLLM evicts inside its capacity check and never undoes the eviction. SGLang reserves nothing across steps and encodes again what its cache evicted. Here a chunk reserves its images and gives them back if the chunk gets no pages, so a failed chunk leaves the cache as it was ([encoder scheduling](#the-scheduler)).
- **GDN checkpoints.** vLLM's GDN prefill emits no state inside one prefill. SGLang gathers the kernel's BF16 per-chunk states, which round at about 3.6e-3. Here a checkpoint is an FP32 state at the requested offset, made by cutting a row into calls on FlashInfer's per-sequence route. The in-kernel checkpoint variant gives different outputs for identical calls on SM120 ([checkpoints inside a step](#checkpoints)).
- **Cache entries.** Neither engine replaces a cache entry, or keys one by how it was computed. Here the precise copy of a prefix lives in a second key space beside the FP8 one, so no page changes under a request that reads it ([the precise prefix](#the-precise-prefix)).
- **Graphs.** Neither engine captures sampling. vLLM's graphs hold the forward pass. SGLang's hold the forward pass and its logits processor. Masks and samplers run eagerly after them. Here the graphs hold the sampler too, so a step adds no launch outside them ([CUDA graphs](#cuda-graphs)).

## Not yet

This list holds what may come. [Scope and decisions](#scope-and-decisions) lists what is not planned.

- Tensor parallelism, and any engine for more than one GPU.
- An Apple Silicon (MLX) runner.
- Mixture-of-experts models (Gemma 4 26B-A4B, Qwen3.5 MoE), and other model families.
- `prompt_logprobs`, penalties, `logit_bias`, `min_tokens`, a thinking budget.
- Validation on GPUs other than SM120.
- A cache-aware router's KV events.
