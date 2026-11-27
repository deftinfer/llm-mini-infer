# The server's overhead

These benchmarks measure what the Python server itself costs. The fake runner replaces the GPU (`runner.kind: fake`): it gives random tokens on a simulated device timeline, through the real scheduler, cache and API. The timings come from the server's own instrumentation (`/debug/stages`, `/debug/engine`), the CPU use from psutil, and the load from separate client processes. Speculative decoding is off.

```bash
uv run python bench/server/run.py all                    # one pass of everything
uv run python bench/server/run.py ttft [--warm] [--fake-wait sleep] [--images] [--response-format <schema.json>]
uv run python bench/server/run.py decode --batch 64 256
uv run python bench/server/run.py idle --conc 1 8 32
uv run python bench/server/run.py sse
uv run python bench/server/report.py results/server/*.json     # several runs as markdown tables
python3 bench/server/run.py ttft+idle --runner cuda --model <model dir>   # in the GPU tests' image: real device time
```

`--help` lists the options. The default model is the Gemma 4 31B fixture (`uv run tests/scripts/fetch_fixtures.py`). On a sliding-window model such as Gemma 4, the fake runner checks each row's KV at each step. These checks are in the runner's launch calls, outside the engine's CPU below, but they keep the engine busy. `--model tests/.fixtures/models/qwen3.8-27b` has no window.

## What it measures

| Measure | How | Target |
|---|---|---|
| TTFT overhead | one request at a time: a 1,024-token chat, streamed, 4 tokens out. The time from the request's body read to its first byte, less the device time of its steps (5 ms of prefill) | p50 < 1 ms, p99 < 2 ms |
| Engine CPU per decode step | 64 and 256 streams of 400 tokens, 10 ms steps. Per steady step (every request decodes): the loop's wall time less blocking and the runner's launch calls | < 0.5 ms at 64, < 1.5 ms at 256 |
| GPU idle time caused by the CPU | 5 ms steps, closed-loop clients at concurrency 1, 8 and 32, the 1,024-token chat with 64 tokens out, 15 s each. The device's gaps while a step waited, over busy time plus gaps | < 3% |
| SSE throughput, one frontend | 128 streams of 256 tokens, 0 ms steps, the events of the streams that completed in 10 s | reported |

## The event loop beside image requests

`bench/server/loop.py` measures what an image request costs the other streams of its frontend: one event loop serves them all, and an image request's parse, base64 decode, header probe and hash run on it. `arms` runs a server with images on and, beside steady streams, image requests of three kinds (a phone-sized JPEG, a 12-megapixel JPEG, a PNG) at a few a second and at as many as the server takes, the images of a folder (`--files`), and request logging into a stderr that stops reading for a while. Per arm it reports the lag of the frontend's timer (`/debug/loop`, which `observability.debug_endpoints` turns on), the gaps between the streams' chunks, and the image requests' TTFT and on-loop stages; the arms run interleaved, several times. `stages` times each step of an image request's path in one process, and whether it lets other threads run. `--help` lists the options.

```bash
uv run python bench/server/loop.py arms --repeats 5
uv run python bench/server/loop.py stages
uv run python bench/server/loop.py table results/server/<time>-loop-arms.json
```

## Caveats

- On macOS the engine's thread is not pinned, the cores differ and the clients share the machine. On Linux the engine pins its thread (`engine_affinity`), and `cgroup_throttled_ms` shows a CPU quota's throttling. Run the benchmarks on the serving host to judge it.
- `--runner cuda` on a GPU measures the CUDA runner's own launch cost.
