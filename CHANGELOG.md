# Changelog

## Unreleased

Breaking:
- The dev settings `dev.speculative.draft_head_dtype` and `dev.runner.cuda.prefill_split_from` are removed, and so is the value `flashinfer` of `dev.runner.cuda.fp8_gemm`. A configuration that sets one fails the load (exit 2) with a message that names it and says what holds now: the drafter's head is in the model dtype (`dev.runner.cuda.dtype`; on Gemma 4 the assistant's own head, on Qwen3.8 the target's), the full layers' prefill split starts at the kernels' threshold (`kernels.attn512.PREFILL_FROM`), and the FP8 GEMMs take `auto`, `cutlass`, `sglang` or `torch`.
- `/metrics` no longer serves `llm_mini_infer:spec_first_rejections_total` and `llm_mini_infer:spec_window_mask_tokens`, and the host check's report no longer has the `spec_first_rejections` key.

Changed:
- The image no longer ships torchvision, and the `images` extra no longer requires it. The image pool decodes JPEG with Pillow, as it decodes the other formats and as vLLM does, and resizes with torch alone; the pixels are the same. A pool worker starts faster and holds less memory; a large JPEG takes longer to decode.
- A truncated JPEG gets a 400 with Pillow's error, `image: OSError: image file is truncated (...)`, instead of torchvision's `RuntimeError`, which named its source file; a JPEG whose decode fails otherwise gets Pillow's text too, for example `broken data stream when reading image file`.
- A JPEG above Pillow's `MAX_IMAGE_PIXELS` (89,478,485 pixels) decodes with Pillow's `DecompressionBombWarning` in the log, as PNG and WebP do. `images.max_source_pixels` still refuses an image above twice that.
- An image is oriented by the image pool's decode alone, with Pillow's EXIF orientation, as vLLM orients it, and the frontend no longer reads EXIF or XMP. A non-square image whose orientation the frontend read otherwise than Pillow got a 400 at orientations 5 to 8 and is now served as Pillow orients it; one whose EXIF Pillow reads but cannot rewrite after turning it (crafted values) is now served as stored at every orientation, as vLLM serves it.
- A JPEG with more than 4,096 markers or 1 MiB of fill bytes before its frame header gets a 400 (`image: broken JPEG header: more than 4096 markers before the frame header`, or `... more than 1 MiB of fill bytes ...`), and so does a data URI with more than 1,024 characters before its `,` (`image: malformed data URI: more than 1024 characters before its ','`); real files hold a few hundred markers at most, no fill to speak of, and a few dozen characters before the `,`. A JPEG with bytes after its first frame header that the frontend could not walk (`marker expected`) is served where Pillow decodes it.
- `llm_mini_infer:image_stage_seconds`' `probe` stage no longer holds the image's hash, which now runs off the event loop while the pool decodes.
- A PNG text chunk that inflates past 1 MiB gets Pillow's 400 from the image pool instead of the frontend's `a PNG text chunk inflates past 1 MiB`. A WebP whose first chunk is not VP8, VP8L or VP8X gets the frontend's 400, `broken WebP header: no VP8, VP8L or VP8X chunk first`, instead of the pool's.

Added:
- `/debug/loop` (with `observability.debug_endpoints`) gives a frontend's event-loop lag: how late a timer due every millisecond ran, for at least the last two minutes. `bench/server/loop.py` measures it beside steady streams under image requests, crafted images and a stderr that stops reading.

Fixed:
- A crafted image no longer stalls every stream of its frontend. The frontend's header probe read EXIF, XMP and PNG text and walked every chunk of an image on the event loop, up to about a minute for one 20 MiB image; it now reads a bounded part of the header, and an image of 256 KiB or more is hashed, and its data URI decoded (with pybase64, which lets the loop run meanwhile), on a thread of the frontend's, on Linux on the cores the engine and the frontends leave.
- A stderr that stops reading no longer stops a frontend's streams through its log. A frontend's log lines go through a queue to a writer thread, so a blocked stderr blocks that thread instead of the event loop that every stream waits on (it froze them for as long as stderr took nothing). The queue holds 10,000 lines; past that a line is dropped and counted, and the count is logged in its place, with the next line or at the frontend's exit. A frontend that exits normally writes every line it queued, waiting at most 10 s for a stderr that takes nothing. On Linux the writer thread runs on the cores the engine and the frontends leave, not on its frontend's. The launcher, the engine and the image pool write directly, as before, and so does anything that writes to stderr without the log.
- An image's EXIF orientation follows Pillow's for malformed EXIF blocks. A block behind two `Exif\0\0` prefixes is read, as Pillow strips both: a JPEG at orientations 2 to 8 was served as stored and is now flipped or turned, and a PNG or WebP at 5 to 8 got a 400 and is now turned (at 2 to 4 they were already served as Pillow orients them). EXIF that Pillow cannot parse counts as orientation 1 even beside an XMP orientation, as in Pillow: a JPEG at 2 to 8 was flipped or turned by the XMP's value and is now served as stored, and a PNG or WebP at 5 to 8 got a 400 and is now served as stored (at 2 to 4 they already were). A TIFF header with its 42 written in the other byte order, or a big-endian 43, which Pillow reads as classic TIFF, is no longer taken for missing EXIF. An IFD0 entry whose data lies past the block's end ends IFD0 there, as in Pillow, whatever its tag: an Orientation after it no longer counts, and an XMP orientation does (a JPEG was flipped or turned by that Orientation, and a PNG or WebP at 5 to 8 got a 400).

## 0.8.0

This is the first public release of llm-mini-infer. It is an LLM inference server for Gemma 4 31B and Qwen3.8 27B. It runs on one NVIDIA SM120 GPU, such as the RTX PRO 6000 Blackwell. It ships as a container image with prebuilt kernels and as a Helm chart. The [README](README.md#quickstart) has the quickstart.

For a client (the HTTP API):
- Chat completions, streamed or not, follow vLLM v0.33.0's chat API for the features that the README lists. See [The API](README.md#the-api) and [vLLM compatibility](docs/vllm-compatibility.md), which lists the deviations.
- Image parts take a base64 data URI, or an `http(s)` URL when `images.allow_urls` is on. The server refuses a URL that names a private address. See [The API](README.md#the-api) and [Access](docs/operations.md#access).
- `response_format` takes a JSON schema, and the output follows it. A schema outside the supported subset gets a 400. See [Structured output](docs/vllm-compatibility.md#structured-output).
- Tool calls work on Gemma 4 and Qwen3.8: `tools`, `tool_choice`, tool results, and call arguments that stream in pieces. See [Tool calls](docs/vllm-compatibility.md#tool-calls).
- A reasoning parser returns the thought in `reasoning`, and `reasoning_effort` and `include_reasoning` control it. Qwen3.8 thinks by default, as its chat template does, and `reasoning_effort: "none"` turns thinking off. See [Reasoning](docs/architecture.md#reasoning).
- `cache_control` asks for a prefix-cache checkpoint at a message's end. See [Cache hints](docs/architecture.md#cache-hints).
- Requests with different `cache_salt` values share no cached prompts or image embeddings. See [The prefix cache](docs/operations.md#the-prefix-cache).
- `/tokenize`, `/detokenize` and `return_token_ids` give token ids in vLLM's shapes. See [Requests](docs/vllm-compatibility.md#requests).
- With `model.max_model_len` at 8,184 or less, and on Qwen3.8 a block-FP8 checkpoint, batch invariance is on. Then a request's output can be independent of the other requests in its step. See [Batch invariance](docs/architecture.md#batch-invariance).

For an operator:
- The container image holds the kernels prebuilt for SM120 and runs one server on one GPU. See [The image](docs/operations.md#the-image).
- The Helm chart runs one replica per GPU, and `model.id` names the model. See [Kubernetes](docs/operations.md#kubernetes).
- `GET /server_info` shows every effective setting and the reason for each derived value. See [Settings](docs/operations.md#settings). [Deploy settings](docs/vllm-compatibility.md#deploy-settings) gives the setting for each vLLM flag.
- Speculative decoding drafts with Qwen3.8's MTP layer, and on Gemma 4 with its assistant model when `speculative.model` names it. See [Speculative decoding](docs/architecture.md#speculative-decoding).
- The radix prefix cache keeps state checkpoints for hybrid and sliding-window models. It is on by default. See [The prefix cache](docs/operations.md#the-prefix-cache).
- With an API key set, every path needs the key except OPTIONS, `/health`, `/ready`, `/live` and `/metrics`. See [Access](docs/operations.md#access).
- `scheduler.max_num_queued_reqs` and `scheduler.max_num_queued_tokens` cap the queue with a 503. See [Settings](docs/operations.md#settings).
- SIGTERM drains the server. The container exits 1 when one of its processes failed. See [Health, drain and restarts](docs/operations.md#health-drain-and-restarts).
- `llm-mini-infer hostcheck` measures a machine's GPU, host, server start and load, and gives verdicts against a reference. One reference ships, for an RTX PRO 6000 at 600 W with Gemma 4 31B FP8 and its assistant; with no reference for the setup, the report gives the numbers alone. See [The host check](docs/operations.md#the-host-check).

For a dashboard:
- `/metrics` serves vLLM's metric names, help texts and buckets. [vLLM compatibility](docs/vllm-compatibility.md#metrics) lists the series that it does not export.
- The `vllm:spec_decode_*` series count drafts and accepted tokens, and the `vllm:prefix_cache_*` series count cache queries and hits. See [Metrics](docs/operations.md#metrics).
- `vllm:admission_rejections_total{reason}` counts the 503s of the queue caps. See [Metrics](docs/operations.md#metrics).
- The `llm_mini_infer:*` series show host overhead, GPU idle time, failed requests, image stages and lost prefix-cache tokens. See [Metrics](docs/operations.md#metrics).
