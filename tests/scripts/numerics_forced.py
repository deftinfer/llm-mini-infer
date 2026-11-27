#!/usr/bin/env python3
"""Teacher-forced runs of the real models through the real engine, for the numerics analysis
(tests/scripts/numerics_fp32.py): the same reference answers forced under each arm of batch invariance, with the
target's final hidden state at every position, the drafter's acceptance at every verify step and, eagerly, every layer's
residual stream, mixer and MLP outputs and a few kernels' own inputs and outputs. Qwen3.8 with its MTP layer
(`CudaRunner`) and Gemma 4 with its assistant (`GemmaRunner`); each arm runs in a process of its own (the switch changes
the model and process-wide flags).

    python3 -m tests.scripts.numerics_forced refs --model Qwen/Qwen3.8-27B-FP8 --workload scene,chat --indices 0-7
    python3 -m tests.scripts.numerics_forced run --refs <dir>/refs.json --arm off [--cache] [--concurrency 8] \\
        [--rids 0,3]
    python3 -m tests.scripts.numerics_forced run --refs <dir>/refs.json --arm on --no-graphs --taps --core-layers 3,31
    python3 -m tests.scripts.numerics_forced run --refs <dir>/refs.json --arm on --kv-dtype fp8 [--concurrency 8]
    python3 -m tests.scripts.numerics_forced run --refs <dir>/refs.json --arm on --kv-cache-dtype fp8
    python3 -m tests.scripts.numerics_forced same <dir>/on-kvfp8 <dir>/on-kv8
    (Gemma 4: --model RedHatAI/gemma-4-31B-it-FP8-block --assistant google/gemma-4-31B-it-assistant)

**The engine** (`Engine`) is the server's, in-process: the runner, its pools sized as `engine.process` sizes them from
the memory the runner reports (its memory profile; `--memory-gb` skips it), the scheduler with the server's settings and
the engine loop. The defaults are the server's for these checkpoints with structured output on, but one: k = 3 drafts on
both models (`--k`; the server drafts 5 with an assistant). The server's: `model.max_model_len` at the engine
benchmark's context (8,184), steps of 2,048 tokens, 64 sequences, pages of 16 positions (Gemma
4: window pages of 16, full pages of as many bytes), the sparse head, always-split graphs, PDL, the forced prefix and
the EOS-free finish, the precise prefix (`runner.cuda.fp8_cold_prefill` reuse; it acts on cache hits only), the
mixed-step bound derived from the profile's step costs (off with `--memory-gb`), images on where the checkpoint has a
tower. An arm sets `dev.runner.cuda.batch_invariant` (`on`, `off`; Gemma 4's five runner switches together, as
`make_cuda_runner` sets them), the prefix cache (`--cache`; Qwen's checkpoint slots as the server sizes them) and CUDA
graphs (`--no-graphs`: every step eager), and with `--kv-dtype` the KV an FP8 pool would hold (below), or with
`--kv-cache-dtype fp8` the real FP8 pool (`cache.kv_cache_dtype`: Gemma 4's full layers). Its name, and its folder
under `--out`: `<switch>[-kv8|-kv8all|-kvfp8][-cache][-eager]` (`arm_name`), e.g. `on`, `off-cache`, `on-kv8-eager`,
`on-kvfp8`.

**References** (`refs`; `build_requests`, `make_refs`): each request's prompt as the server builds it (the chat template
with thinking off, `Renderer.encode`, the frontend's image expansion), answered greedily by the engine under the
request's real grammar (llguidance over the checkpoint's tokenizer, the guidance backend's EOS), arm `on` with graphs
by default, one request at a time, the prefix cache off:
- `scene`: the host check's synthetic workload as bench/load/run.py builds it (`hostcheck.synthetic_inputs` at its
  defaults: a system prompt of `synthetic.PROMPT_TOKENS`, one generated image per request index, a share
  `synthetic.TALL_SHARE` of them portrait; the short user text; the strict schema as compact guidance JSON,
  `json_schema_grammar(schema, "guidance", disable_any_whitespace=True)`);
- `chat`: bench/load/run.py's `chat_workload` prompts (`--input-tokens`, 2,048 by default), text only, no grammar;
- `recall`: chat's prompts asking for the words in order from the first, so the answer reads the prompt's start (at a
  long context, past Gemma 4's sliding window); text only, no grammar.
A long context: `--input-tokens 32768 --max-model-len 65536` (a forced run takes the references' max_model_len); a
prompt whose answer could pass max_model_len is refused.
`<out>/refs.json` keeps, per request, what rebuilds its prompt (workload, index, its token parts with each image by its
index, the grammar's name and its forced prefix's length), the expanded prompt to check the rebuild against, the answer
(EOS stripped) and the finish.

**Forced runs** (`run`; `run_arm`) force every answer (`ForcedEach`: tests/support/forced.py's grammar, the reference's
next token alone at every position, then its EOS, each request's forced prefix its own grammar's), greedy, each
request's `max_tokens` its answer's length + 2. The drafts stay free: no draft universe (it would hold the reference's
own tokens) and the drafter's whole head (`--draft-vocab` 0), so the in-engine acceptance is the drafter's on the
reference. `--concurrency 1` runs each request alone; N keeps N in flight (a closed loop). Each request's capture goes
to `<out>/<arm>/r<rid>.pt` (`torch.save`; tensors on the CPU), and `<out>/<arm>/run.json` holds the settings, the whole
run's counters, each request's acceptance and the timings; a markdown table is printed. A request whose output is not
its reference (a grammar error) fails the run after run.json is written.

**The KV fake-quant** (`run --kv-dtype`; `KvFakeQuant`): each layer's KV write is followed, before that layer's
attention reads it, by the written slots rounded through float8_e4m3fn and back (`e4m3_round`: clamp to ±448 at the
scale, round to nearest even); `fp8` on the full-attention layers (Gemma 4's one stored plane, Qwen's K and V and its
MTP layer's), `fp8-all` on Gemma 4's sliding layers too. Every E4M3 value is a BF16 value, so a kernel that reads an FP8
pool by converting each tile to BF16 before today's arithmetic (FlashInfer's FP8 prefill does) reads exactly this BF16
pool: the arm's tokens, hidden states and acceptance are that design's, the drafter's and the precise prefix's included.
The scale is 1.0 unless `--kv-scales` sets per-layer powers of two (exact too); Gemma 4's full layers store k / rms(k)
per head, at most sqrt(512) = 22.6, never saturated. run.json's `kv_fake_quant`: the settings and, per layer and plane,
the values written, those saturated and the largest magnitude (and the power-of-two scales that would keep them in
range). The real pool (`--kv-cache-dtype fp8`, arm `-kvfp8`) stores the same rounding of the same BF16 values
(`gemma_ops.kv_fp8`) and its kernel reads them as this arm's attention does, so on the same references `on-kvfp8`
should equal `on-kv8` bit for bit (`same`: hidden states and acceptance per request), except where the full page of
twice the positions changes what a prefix-cache hit reuses (`-cache` arms). The fake-quant refuses an FP8 pool.

**Bit for bit** (`same A B`): per request, whether it is in both arm folders and its tokens, the positions with a hidden
state, the hidden states there and the acceptance at every verify step are the same; exit 1 when any request is not.

**The capture** (one dict per request; the shape the fp32 analysis reads):
- `model` ("qwen" | "gemma"), `model_dir`, `assistant_dir`, `arm`, `settings` (the engine's: the `Options`, the pools,
  the scheduler's `forced_prefix` and `eos_free_finish`, Gemma 4's `draft_query_next`, the runner's stats);
- `tokens` int64 [N]: the prompt and the reference answer; `prompt_len`; `forced`: the tokens the request's grammar
  forces at its start, which its final prefill chunk takes with `forced_prefix` (whole with batch invariance on, best
  effort off); `images`: (offset, tokens) of each image span;
  `rope` int64 [3, N]: Qwen's M-RoPE positions with images (`ImageFamily.positions`), else None (positions 0..N-1);
- `hidden` [N, D] in the runner's dtype (BF16 on CUDA, fp32 on the CPU): the target's final hidden state after its final
  norm, `hvalid` bool [N] where it was computed from reference inputs only (Capture.drain's rule: a row's tokens count
  up to its first input that is not the reference's). With graphs, from `static.hid_out` behind each step's forward:
  every token's on Qwen (its drafter reads them), the sample tokens' on Gemma 4 (every answer position under forcing,
  not the prompt's); eagerly, from the forward's output: every token's on both;
- `accept` int64 [R]: the drafts accepted at each verify step whose window starts on the reference, short of its last
  token (a window there verifies nothing of it), in order, and `accept_pos` int64 [R]: the position of that window's
  first token (the pending token; its drafts sit at accept_pos + 1, ...; the drafter read the hidden state at
  accept_pos − 1);
- `counters`: the whole run's (the scheduler's spec_drafts, spec_accepted and per position, the forced prefix, cache
  and precise-prefix counters; the runner's attention plan counters (replans, kept, unsplit) and head counters; steps
  and the bodies they ran: decode or breakable graphs, eager). With `--taps` a capture is written as its request
  finishes (host memory), its counters as of then (`counters["final"]` False); run.json has the final ones;
- `taps` (eager arms with `--taps` only): `kinds` per layer ("gdn" | "full" | "sliding"), `stream` [L + 1, N, D] (the
  residual stream entering layer l; [L]: the stream the final norm reads), `mixer` [L, N, D] (the attention or GDN
  module's output), `mlp` [L, N, D] (the MLP module's output), `tvalid` bool [N], all in the runner's dtype, and `core`
  ({layer: dict}, `--core-layers`, which needs `--taps`): a kernel's inputs and output in situ, on Qwen's GDN layers its
  prefill rows `{q, k, v, alpha, beta, o, init, rows}` (q and k L2-normalised, v, alpha = exp(g) and beta as the
  prefill kernel takes them, its output before the gated norm, each row's initial state fp32 [R, HV, V, K], zero at
  pos0 0; `rows` (pos0, n), concatenated in that order), on full-attention layers `{q, k, v, o, valid}` by position (q
  after its norm and RoPE, k and v as written to the KV pool, o the attention's output before Qwen's gate; Gemma 4's
  full layers store V alone, `attention_k_eq_v`: `v` is that plane and `k` is absent, K = RoPE(v ⊙ k_norm.weight)).
  Gemma 4's sliding layers have no core capture (FlashInfer rewrites an image span's rows in a second pass).

**Taps** replace module methods on the instances (eager only: a graph replay runs no Python), so every value is the
engine's own. Qwen (`models/qwen3_5/model.py` DecoderLayer): the stream entering layer l is the residual that
`input_layernorm.apply` returns (`x + residual` rounded once; layer 0's: the input embeddings, image rows included); the
mixer is `self_attn`'s or `linear_attn`'s return, the MLP `mlp`'s; the next stream is the fused add-norm's rounded sum
of the MLP output and the post-attention residual, itself the rounded sum of the stream and the mixer, so `stream[l + 1]
== dtype(dtype(stream[l] + mixer[l]) + mlp[l])` bit for bit wherever the add-norms sum in fp32 and round once (the torch
references do; each norm reads its sum before the rounding: tests/scripts/numerics_fp32.py); [L] is the final norm's own
sum (`norm.forward_add`). Gemma 4 (`models/gemma4/model.py`): the stream is each layer call's `residual` argument (layer
0's: the scaled embeddings, image rows unscaled); the mixer is the attention module's output (o_proj's), the MLP the MLP
module's; the layer's sandwich gives `r_mid = rnd(r + rms(mixer) * w_post_attention)` and the model loop's
`r_next = rnd((r_mid + rms(mlp) * w_post_feedforward) * layer_scalar)` (`gemma_ops.sandwich`, one rounding each); [L] is
the last sandwich's residual (computed again by the same op: the model discards it after the final norm). A tap keeps
only the step's tokens computed from reference inputs, copied to the host after each layer: a request's taps take (3L +
1) × N × D values of host memory (about 7 GB at N = 3,400 on either model), each request in flight its own.

**On the CPU** the same code runs the tiny models (tests/support/tiny_qwen.py, tests/support/tiny_gemma.py;
`Options(device="cpu")`: fp32, the torch reference kernels, the runner's 1 GiB of pools unless `memory_gb` or
`num_pages` say otherwise); their requests come as token parts (`make_refs`), images as {"image": i}
(`synthetic.image_jpeg(i)`).

**Shared** with tests/scripts/numerics_kernels.py and tests/scripts/numerics_fp32.py (this module imports no torch at
its top): `results_dir`, `parse_ids`, `resolve`, `md`, `HelpFormatter`.
"""

from __future__ import annotations

import argparse
import functools
import importlib.util
import json
import logging
import math
import os
import re
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from ..support.forced import Forced, ForcedMatcher
from ..support.paths import RESULTS, ROOT

MAX_LEN = 8184  # model.max_model_len: the engine benchmark's context (bench/engines/arms.py CONTEXT)
SOFT_TOKENS = 280  # Gemma 4's images.max_soft_tokens: the checkpoint's own budget
CHAT_TOKENS = 2048  # the chat workload's prompt tokens
MAX_TOKENS = 384  # a reference answer's tokens at most
DRAIN = 32  # captured steps kept on the device between drains (one host sync a drain)
ARMS = ("on", "off")
WORKLOADS = ("scene", "chat", "recall")
# recall: chat's prompt with its instruction turned around (bench/load/run.py's chat_workload words it so)
REVERSE, FORWARD = "Repeat these words in reverse order:", "Repeat these words in order, from the first:"
KV_DTYPES = ("fp8", "fp8-all")  # `--kv-dtype`: the full-attention layers' KV, or every layer's (Gemma 4's sliding too)
E4M3_MAX = 448.0  # float8_e4m3fn's largest finite value

log = logging.getLogger("numerics_forced")


# -- shared by the numerics tools --------------------------------------------------------------------------------------


def results_dir(tool: str) -> Path:
    """A numerics tool's default results folder, results/numerics/<tool>/<UTC time> (git-ignored), stamped as
    the repo's other tools stamp theirs."""
    return RESULTS / "numerics" / tool / time.strftime("%Y%m%d-%H%M%S", time.gmtime())


def parse_ids(text: str) -> list[int]:
    """`0-3,7,14` -> [0, 1, 2, 3, 7, 14]."""
    out: list[int] = []
    for part in filter(None, (p.strip() for p in text.split(","))):
        a, sep, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if sep else [int(a)]
    return out


@functools.cache
def resolve(path: str | Path) -> Path:
    """A local checkpoint folder, or a repo id fetched into (or found in) the Hugging Face cache (`llm_mini_infer.hub`),
    once a process."""
    from llm_mini_infer import hub

    return Path(hub.download(str(path), None) if hub.is_repo_id(str(path)) else path)


def md(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    """A markdown table: a cell None is "-", a float has three decimals, a list its items' cells (empty: "-")."""
    def cell(v: Any) -> str:
        if isinstance(v, float):
            return f"{v:.3f}"
        if isinstance(v, list):
            return " ".join(cell(x) for x in v) or "-"
        return "-" if v is None else str(v)

    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    return "\n".join(out + ["| " + " | ".join(cell(v) for v in r) + " |" for r in rows])


class HelpFormatter(argparse.RawDescriptionHelpFormatter):
    """The tools' help: descriptions as written, each option's default after its help unless the help names it or it
    is None, False or empty (the help says what that means)."""

    def _get_help_string(self, action: argparse.Action) -> str | None:
        h, d = action.help or "", action.default
        if not action.option_strings or "%(default)" in h or d is None or d is False or d == "" or \
                d is argparse.SUPPRESS:
            return action.help
        return f"{h} (default: %(default)s)"


def arm_name(switch: str, cache: bool, graphs: bool, kv_dtype: str | None = None, kv_scaled: bool = False,
             kv_cache_dtype: str = "auto") -> str:
    """`<switch>[-kv8|-kv8all|-kvfp8][-cache][-eager]`: `-kv8` the fake-quant (`s` after it when per-layer scales other
    than 1 are set), `-kvfp8` the real FP8 pool."""
    kv = "" if kv_dtype is None else "-kv8" + ("all" if kv_dtype == "fp8-all" else "") + ("s" if kv_scaled else "")
    if kv_cache_dtype == "fp8":
        kv += "-kvfp8"
    return switch + kv + ("-cache" if cache else "") + ("" if graphs else "-eager")


# -- the checkpoint ----------------------------------------------------------------------------------------------------


def model_kind(model_dir: Path) -> str:
    from llm_mini_infer.models.gemma4.model import is_gemma4

    return "gemma" if is_gemma4(model_dir) else "qwen"


def vocab_size(model_dir: Path) -> int:
    """The logits' width, as `config.derive` reads it (the runners' `cfg.vocab_size`)."""
    hf = json.loads((model_dir / "config.json").read_text())
    return int(hf.get("text_config", hf).get("vocab_size", hf.get("vocab_size", 0)) or 0)


def stop_ids(model_dir: Path) -> tuple[int, ...]:
    """The model's stop ids, as the server derives them (generation_config.json's, else config.json's)."""
    from .profile_verify import eos_ids

    return eos_ids(model_dir)


def grammar_eos(model_dir: Path) -> tuple[str | None, int]:
    """The EOS a finished grammar allows under the guidance backend: the tokenizer's eos_token by name
    (`config.tokenizer_eos_token`: `<|im_end|>` on Qwen3.8, `<eos>` on Gemma 4) and its id; a folder without a tokenizer
    (the tiny models) takes its first stop id."""
    from types import SimpleNamespace

    from llm_mini_infer.config import tokenizer_eos_token

    name = tokenizer_eos_token(SimpleNamespace(model=SimpleNamespace(tokenizer=None, path=str(model_dir))))
    path = model_dir / "tokenizer.json"
    if name and path.exists():
        from tokenizers import Tokenizer

        i = Tokenizer.from_file(str(path)).token_to_id(name)
        if i is not None:
            return name, int(i)
    return None, stop_ids(model_dir)[0]


def image_family(model_dir: Path, soft_tokens: int | None) -> Any:
    """The checkpoint's image family (None without a tower), Gemma 4's at `soft_tokens`."""
    from llm_mini_infer.vision import family as vf

    return vf.load(model_dir, max_soft_tokens=soft_tokens if model_kind(model_dir) == "gemma" else None)


@dataclass
class Options:
    """The engine's settings, the module docstring's defaults unless set (the server's but two: k, and Gemma 4's image
    budget `soft_tokens`, None for the checkpoint's). `memory_gb`: pool memory to report instead of the runner's memory
    profile (None: the profile on CUDA, 1 GiB on the CPU); `num_pages`, `ckpt_slots`: the pools' sizes instead of the
    server's sizing; `row_tokens`: `dev.scheduler.mixed_step_row_tokens` (None: derived from the profile's step costs,
    as the server derives it; 0: off); `draft_vocab` None: the server's (`config.DRAFT_VOCAB` on Qwen3.8, the whole head
    with an assistant)."""

    device: str = "cuda"
    dtype: str = "auto"  # auto: BF16 on CUDA, fp32 on the CPU
    k: int = 3
    max_model_len: int = MAX_LEN
    budget: int = 2048
    max_num_seqs: int = 64
    page_size: int = 16
    memory_gb: float | None = None
    num_pages: int | None = None
    ckpt_slots: int | None = None
    row_tokens: int | None = None
    precise: str = "reuse"
    soft_tokens: int | None = SOFT_TOKENS
    draft_vocab: int | None = None
    draft_universe: bool = True
    sparse_head: bool = True
    pdl: bool = True
    always_split: bool = True
    check: bool = False
    kv_cache_dtype: str = "auto"  # cache.kv_cache_dtype: fp8, Gemma 4's full layers in a real FP8 pool


# -- prompts -----------------------------------------------------------------------------------------------------------


@functools.cache
def image(i: int) -> bytes:
    """Image `i` of the scene workload (`synthetic.image_jpeg`)."""
    from llm_mini_infer.synthetic import image_jpeg

    return image_jpeg(i)


def _bench_load() -> Any:
    """bench/load/run.py, loaded from its file (bench/ is no package; tests/repo/test_bench_kit.py loads it so)."""
    name = "bench_load_run"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / "bench" / "load" / "run.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


def build_requests(model_dir: Path, workloads: Sequence[str], indices: Sequence[int], *,
                   soft_tokens: int | None = SOFT_TOKENS, input_tokens: int = CHAT_TOKENS,
                   seed: int = 0) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Each workload's requests at `indices` as the server builds their prompts (module docstring), rids from 0 in that
    order: (requests: rid, workload, index, parts (token lists and {"image": index}: the template's one placeholder
    token taken out, the frontend's expansion puts the span there), grammar (its name, or ""), forced (its forced
    prefix's tokens); the grammar texts by name)."""
    from tokenizers import Tokenizer

    from llm_mini_infer import synthetic
    from llm_mini_infer.hostcheck import KW, synthetic_inputs
    from llm_mini_infer.render.messages import to_template_messages
    from llm_mini_infer.render.renderer import Renderer

    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))  # bench/load/run.py's `load_tokenizer`
    r = Renderer(model_dir)
    fam = image_family(model_dir, soft_tokens)
    kw = r.filter_template_kwargs(dict(KW))  # thinking off, as the bench's requests ask
    reqs: list[dict[str, Any]] = []
    grammars: dict[str, str] = {}
    for wl in workloads:
        if wl == "scene":
            if fam is None:
                raise ValueError(f"{model_dir} has no vision tower: the scene workload needs one")
            inp = synthetic_inputs(lambda s: len(tok.encode(s).ids), synthetic.PROMPT_TOKENS,  # specials included,
                                   synthetic.TALL_SHARE)  # as the bench counts; the images as `image` gives them
            grammars[wl] = _scene_grammar(model_dir, inp.schema)
            forced = len(_grammars(model_dir).forced_prefix(grammars[wl]))
        elif wl in ("chat", "recall"):
            chat = _bench_load().chat_workload(tok, input_tokens, MAX_TOKENS, seed)
            forced = 0
        else:
            raise ValueError(f"workload {wl!r}: one of {', '.join(WORKLOADS)}")
        for i in indices:
            if wl == "scene":
                msgs = [{"role": "system", "content": [{"type": "text", "text": inp.prompt}]},
                        {"role": "user", "content": [{"type": "text", "text": inp.user_text},
                                                     {"type": "image_url", "image_url": {"url": f"image:{i}"}}]}]
            else:
                msgs = chat.body(i)["messages"]
                if wl == "recall":
                    if REVERSE not in msgs[0]["content"]:
                        raise ValueError(f"recall: chat's prompt no longer asks {REVERSE!r}")
                    msgs = [{**msgs[0], "content": msgs[0]["content"].replace(REVERSE, FORWARD, 1)}]
            conv = to_template_messages(msgs, r.content_format, r.supports_developer_role,
                                        fam.placeholder if fam is not None else "")
            ids = r.encode(r.render(conv, add_generation_prompt=True, **kw)).tolist()
            parts: list[Any] = [ids]
            if wl == "scene":
                if ids.count(fam.pad_id) != 1:
                    raise ValueError(f"scene {i}: {ids.count(fam.pad_id)} image placeholders in the rendered prompt")
                k = ids.index(fam.pad_id)
                parts = [ids[:k], {"image": i}, ids[k + 1:]]
            reqs.append(dict(rid=len(reqs), workload=wl, index=i, parts=parts, grammar=wl if wl in grammars else "",
                             forced=forced))
    return reqs, grammars


def _scene_grammar(model_dir: Path, schema: dict[str, Any]) -> str:
    from llm_mini_infer.grammar.spec import json_schema_grammar

    return json_schema_grammar(schema, "guidance", disable_any_whitespace=True)  # compact guidance JSON


def _grammars(model_dir: Path) -> Any:
    """The engine's grammars as the server makes them with the guidance backend: llguidance over the checkpoint's
    tokenizer, the logits' width, the tokenizer's eos_token (`engine.process._grammars`)."""
    from llm_mini_infer.grammar.engine import Grammars, LLGuidance

    name, _ = grammar_eos(model_dir)
    return Grammars(LLGuidance(model_dir / "tokenizer.json", vocab_size(model_dir), name or stop_ids(model_dir)))


class Pixels:
    """The requests' images in a pixel arena (`vision.arena`, sized for all of them) as the image pool writes them (the
    family's exact CPU path), and their spans as the frontend expands them (`api.images`): an image part becomes the
    family's span prefix, `size.tokens` placeholders and its suffix, its `ImageRef` keyed by the frontend's digest."""

    def __init__(self, fam: Any, requests: Iterable[dict[str, Any]]):
        from llm_mini_infer.vision import arena as px

        self.fam = fam
        need = sum(self._bytes(p["image"]) for q in requests for p in q["parts"] if isinstance(p, dict))
        self.path = px.arena_path(px.default_dir(tempfile.gettempdir()), f"numerics-{os.getpid()}")
        px.remove(self.path)
        # locked as a server's (llm_mini_infer.leftovers): a server starting meanwhile on this host leaves it alone
        self.path, self.lock = px.create_locked(self.path, max(need, px.ALIGN))
        self.arena = px.map_writable(self.path)
        self.next = 0

    def _bytes(self, i: int) -> int:
        from llm_mini_infer.vision.arena import ALIGN
        from llm_mini_infer.vision.probe import probe

        size = self.fam.size(probe(image(i)))
        return -(-3 * size.height * size.width // ALIGN) * ALIGN

    def expand(self, parts: Sequence[Any]) -> tuple[list[int], list[Any]]:
        """Token parts and image parts -> (the prompt's ids, its images' refs, their pixels in the arena)."""
        import struct

        import blake3

        from llm_mini_infer.engine.messages import ImageRef
        from llm_mini_infer.vision.arena import ALIGN
        from llm_mini_infer.vision.family import span_key, transposed
        from llm_mini_infer.vision.probe import probe

        ids: list[int] = []
        refs: list[Any] = []
        for part in parts:
            if not isinstance(part, dict):
                ids += [int(t) for t in part]
                continue
            data = image(part["image"])
            header = probe(data)
            size = self.fam.size(header)
            px = self.fam.pixels(data, header, size)
            if tuple(px.shape[1:]) != (size.height, size.width):  # Pillow turned the image: the decode decides
                size = transposed(size)
            px = px.numpy().reshape(-1)
            off = self.next
            self.arena.buf[off:off + len(px)] = px.tobytes()
            self.next += -(-len(px) // ALIGN) * ALIGN
            h = blake3.blake3(data)
            h.update(struct.pack("<II", size.height, size.width))
            digest = h.digest()
            ids += list(self.fam.span_prefix)
            refs.append(ImageRef(key=span_key(digest), offset=len(ids), tokens=size.tokens, grid_t=size.grid[0],
                                 grid_h=size.grid[1], grid_w=size.grid[2], height=size.height, width=size.width,
                                 pixels=off, digest=digest))
            ids += [self.fam.pad_id] * size.tokens + list(self.fam.span_suffix)
        return ids, refs

    def close(self) -> None:
        from llm_mini_infer.vision import arena as px

        self.arena.close()
        px.remove(self.path)
        if self.lock is not None:
            os.close(self.lock)


# -- the engine --------------------------------------------------------------------------------------------------------


class Engine:
    """The runner, the scheduler and the engine loop, built as `engine.process` builds them (module docstring), with
    `grammars` (the server's, or the forced ones) and `requests`' images in a pixel arena. `add`, then `drive` the loop
    until every request has finished; `out`, `finish` and `cached` (prompt tokens served from the cache) by rid; `modes`
    counts each launched step's body (`_mode`: decode or breakable graphs, eager)."""

    def __init__(self, model_dir: Path, assistant: Path | None, opts: Options, *, switch: str, cache: bool,
                 graphs: bool, grammars: Any, requests: Sequence[dict[str, Any]]):
        if switch not in ARMS:
            raise ValueError(f"arm {switch!r}: one of {', '.join(ARMS)}")
        if opts.k < 1:
            raise ValueError("k: at least 1 (these runs measure the drafter)")
        t0 = time.perf_counter()
        self.model_dir, self.assistant, self.kind = model_dir, assistant, model_kind(model_dir)
        gemma = self.kind == "gemma"
        if gemma != (assistant is not None):
            raise ValueError("Gemma 4 drafts with its assistant (google/gemma-4-31B-it-assistant), Qwen3.8 with its "
                             "MTP layer: an assistant for Gemma 4 only")
        inv = switch == "on"
        self.eos = stop_ids(model_dir)
        self.fam = image_family(model_dir, opts.soft_tokens)
        self.pixels = Pixels(self.fam, requests) if self.fam is not None else None
        try:
            self._build(opts, switch, inv, cache, graphs, grammars)
        except BaseException:
            if self.pixels is not None:  # the arena file outlives the process otherwise
                self.pixels.close()
            raise
        self.load_s = time.perf_counter() - t0

    def _build(self, opts: Options, switch: str, inv: bool, cache: bool, graphs: bool, grammars: Any) -> None:
        """The runner, its pools and the scheduler, in `engine.process`'s order."""
        import torch

        from llm_mini_infer.cache.pools import size_arena, size_pools
        from llm_mini_infer.config import DRAFT_VOCAB
        from llm_mini_infer.engine.core import EngineCore
        from llm_mini_infer.runner.base import PoolConfig
        from llm_mini_infer.runner.cuda.images import VisionOptions
        from llm_mini_infer.runner.mtp import MAX_GEMMA_MTP_K
        from llm_mini_infer.sched.plan import reserve_positions
        from llm_mini_infer.sched.scheduler import Scheduler, derive_mixed_step_row_tokens, derive_refine_chunk_tokens

        model_dir, assistant, gemma = self.model_dir, self.assistant, self.kind == "gemma"
        vision = None
        if self.fam is not None:  # the server's encoder budget and cache (`config._derive_images`)
            enc = max(opts.budget, min(self.fam.max_tokens(), opts.max_model_len))
            vision = VisionOptions(encoder_rows=2 * enc, encoder_budget=enc, arena=self.pixels.path,
                                   max_soft_tokens=self.fam.max_soft_tokens)
        vocab = vocab_size(model_dir)
        dv = opts.draft_vocab if opts.draft_vocab is not None else \
            DRAFT_VOCAB if not gemma and vocab >= 4 * DRAFT_VOCAB else 0
        kw = dict(device=opts.device, dtype=None if opts.dtype == "auto" else getattr(torch, opts.dtype),
                  max_num_seqs=opts.max_num_seqs, max_num_batched_tokens=opts.budget, max_model_len=opts.max_model_len,
                  memory_gb=opts.memory_gb, check=opts.check, graphs=graphs, grammar=True, vision=vision,
                  fp8_cold_prefill=opts.precise, mtp_k=opts.k, draft_vocab=dv, sparse_head=opts.sparse_head,
                  draft_universe=opts.draft_universe, pdl=opts.pdl, always_split=opts.always_split,
                  batch_invariant=inv)
        if gemma:  # the switch's five runner settings together (`runner.cuda.make_cuda_runner`)
            from llm_mini_infer.kernels.attn512 import SPLIT_KEYS as FULL_SPLIT_KEYS
            from llm_mini_infer.kernels.attn_window import SPLIT_KEYS
            from llm_mini_infer.runner.cuda.gemma import GemmaRunner

            self.runner: Any = GemmaRunner(
                model_dir, assistant=assistant, attn512_split_keys=FULL_SPLIT_KEYS if inv else 0,
                window_split_keys=SPLIT_KEYS if inv else 0, head_fixed_m=inv, encode_per_image=inv,
                kv_cache_dtype=opts.kv_cache_dtype, **kw)
        else:
            from llm_mini_infer.runner.cuda.runner import CudaRunner

            if opts.kv_cache_dtype != "auto":
                raise ValueError("--kv-cache-dtype fp8: Gemma 4's full layers only (Qwen3.8's is not built)")
            self.runner = CudaRunner(model_dir, **kw)
        r = self.runner
        caps = r.capabilities()
        max_len = min(opts.max_model_len, caps.max_positions or opts.max_model_len)  # the server's cap
        if opts.page_size % caps.page_size_multiple:
            raise ValueError(f"page_size {opts.page_size} must be a multiple of {caps.page_size_multiple}")
        reserve = reserve_positions(caps.max_steps_in_flight, caps.mtp_depth)
        if caps.window:  # one arena of window and full pages (`size_arena`)
            sizes: Any = size_arena(
                memory_bytes=caps.memory_free_bytes, kv_bytes_per_token=caps.kv_bytes_per_token,
                window_kv_bytes_per_token=caps.window_kv_bytes_per_token, window=caps.window,
                window_page_size=opts.page_size, max_num_seqs=opts.max_num_seqs, max_model_len=max_len,
                max_num_batched_tokens=opts.budget, reserve_tokens=reserve, num_pages=opts.num_pages,
                prefix_caching=cache)
            page_size, wps, grid = sizes.page_size, opts.page_size, opts.page_size
            slots, live = opts.max_num_seqs + 1, opts.max_num_seqs
        else:
            sizes = size_pools(
                memory_bytes=caps.memory_free_bytes, kv_bytes_per_token=caps.kv_bytes_per_token,
                state_slot_bytes=caps.state_slot_bytes, page_size=opts.page_size, max_num_seqs=opts.max_num_seqs,
                max_model_len=max_len, num_pages=opts.num_pages, checkpoint_slots=opts.ckpt_slots,
                prefix_caching=cache, reserve_tokens=reserve)
            if cache and caps.checkpoint_grid % opts.page_size:
                raise ValueError(f"page_size {opts.page_size} must divide the checkpoint grid {caps.checkpoint_grid}")
            page_size, wps, grid = opts.page_size, 0, caps.checkpoint_grid
            slots, live = sizes.num_state_slots, sizes.num_live_slots
        r.configure(PoolConfig(page_size=page_size, num_pages=sizes.num_pages, num_state_slots=slots,
                               max_num_seqs=opts.max_num_seqs, max_num_batched_tokens=opts.budget,
                               max_model_len=max_len, eos_token_ids=self.eos, mtp_k=caps.mtp_depth,
                               window_page_size=wps))
        row_tokens = opts.row_tokens if opts.row_tokens is not None else derive_mixed_step_row_tokens(
            decode_ms=caps.decode_step_ms, decode_rows=caps.decode_step_rows, prefill_ms=caps.prefill_step_ms,
            prefill_tokens=caps.prefill_step_tokens, max_num_batched_tokens=opts.budget,
            max_num_seqs=opts.max_num_seqs)[0]
        precise = opts.precise if caps.precise_rows else "fp8"
        cost = caps.precise_step_ms / caps.prefill_step_ms if caps.precise_step_ms and caps.prefill_step_ms else 1.0
        k_max = max(MAX_GEMMA_MTP_K, opts.k) if gemma else opts.k  # `config.mtp_k_max`
        piece = derive_refine_chunk_tokens(max_num_batched_tokens=opts.budget, max_num_seqs=opts.max_num_seqs,
                                           mtp_k_max=k_max, grid=grid,
                                           drafter="the assistant" if gemma else None)[0] if precise == "reuse" else 0
        images = self.fam is not None
        self.sched = Scheduler(
            max_num_seqs=opts.max_num_seqs, max_num_batched_tokens=opts.budget, max_model_len=max_len,
            page_size=page_size, num_pages=sizes.num_pages, num_state_slots=slots, eos_token_ids=self.eos,
            steps_in_flight=caps.max_steps_in_flight, mtp_k=caps.mtp_depth, num_live_slots=live,
            hybrid=caps.state_slot_bytes > 0, prefix_caching=cache, checkpoint_grid=grid,
            mixed_step_row_tokens=row_tokens, encoder_rows=caps.encoder_rows if images else 0,
            encoder_budget=caps.encoder_budget, positions=self.fam.positions if images else None, grammars=grammars,
            window=caps.window, window_page_size=wps, bidirectional_images=bool(caps.bidirectional_images and images),
            draft_kv=caps.draft_kv, precise=precise, precise_cost=cost, refine_chunk_tokens=piece, mtp_k_max=k_max,
            forced_prefix=True, eos_free_finish=True, batch_invariant=caps.batch_invariant)
        self.core = EngineCore(self.sched, r)
        self.modes: Counter[str] = Counter()
        mode = r._mode

        def counted(lay: Any, split: bool = False) -> tuple[str, int]:
            m = mode(lay, split)
            self.modes[m[0]] += 1
            return m

        r._mode = counted
        self.out: dict[int, list[int]] = {}
        self.finish: dict[int, int] = {}
        self.cached: dict[int, int] = {}
        self.settings = {
            **asdict(opts), "switch": switch, "batch_invariant": inv, "caching": cache, "graphs": bool(r.use_graphs),
            "max_model_len": max_len, "assistant": str(assistant) if assistant else None, "draft_vocab": dv,
            "page_size": page_size, "window_page_size": wps, "num_pages": sizes.num_pages,
            "num_state_slots": slots, "num_live_slots": live, "row_tokens": row_tokens, "precise": precise,
            "precise_cost": cost, "refine_chunk_tokens": piece, "eos_token_ids": list(self.eos),
            "memory_free_bytes": caps.memory_free_bytes, "images": images,
            "max_soft_tokens": self.fam.max_soft_tokens if images else None,
            "forced_prefix": self.sched.forced_prefix, "eos_free_finish": self.sched.eos_free_finish,
            "draft_query_next": bool(getattr(r, "draft_next", False)),
            "runner": json.loads(json.dumps(r.stats, default=str))}

    def expand(self, parts: Sequence[Any]) -> tuple[list[int], list[Any]]:
        if self.pixels is None:
            if any(isinstance(p, dict) for p in parts):
                raise ValueError(f"{self.model_dir} has no vision tower for the requests' images")
            return [int(t) for p in parts for t in p], []
        return self.pixels.expand(parts)

    def add(self, rid: int, ids: list[int], images: list[Any], *, max_tokens: int, grammar: str = "") -> None:
        """A greedy request (the server's sampling at temperature 0), stopping on the model's stop ids."""
        from llm_mini_infer.engine.messages import AddRequest

        self.out[rid] = []
        self.core.handle(AddRequest(rid=rid, frontend=0, prompt=np.asarray(ids, np.int32), max_tokens=max_tokens,
                                    temperature=0.0, top_p=1.0, top_k=0, min_p=0.0, seed=1 + rid, stop_token_ids=[],
                                    ignore_eos=False, grammar=grammar, images=list(images)))

    def drive(self, order: Sequence[int], add: Callable[[int], None], concurrency: int,
              finished: Callable[[int], None] | None = None, limit: int = 1_000_000) -> None:
        """Requests in `order`, `concurrency` in flight (each finished one replaced before the next turn), until all
        have finished; `finished(rid)` after each."""
        queue = list(order)[::-1]
        live: set[int] = set()
        n = 0
        while queue or self.core.busy:
            while queue and len(live) < concurrency:
                rid = queue.pop()
                add(rid)
                live.add(rid)
            rep = self.core.step()
            n += 1
            if n > limit:
                raise RuntimeError(f"not done after {limit} engine turns")
            for out in rep.outputs.values() if rep is not None else ():
                self.cached.update(zip(out.first_rids.tolist(), out.first_cached.tolist()))
                off = 0
                for rid, nn, fin in zip(out.rids.tolist(), out.n_new.tolist(), out.finish.tolist()):
                    self.out[rid].extend(out.tokens[off:off + nn].tolist())
                    off += nn
                    if fin:
                        self.finish[rid] = fin
                        live.discard(rid)
                        if finished is not None:
                            finished(rid)
        self.sched.check_accounting()

    def counters(self) -> dict[str, Any]:
        """The run's counters: the scheduler's (MTP, the forced prefix and the EOS-free finish, the cache, the precise
        prefix), the runner's attention plans and heads, and the steps by body."""
        import msgspec

        s, r = self.sched, self.runner
        d = s.spec_drafts
        return {
            "steps": self.core.steps, "spec_drafts": d, "spec_draft_tokens": s.spec_draft_tokens,
            "spec_accepted": s.spec_accepted, "spec_accepted_per_pos": s.spec_accepted_per_pos.tolist(),
            "tokens_per_verify": 1 + s.spec_accepted / d if d else None,
            "accept_per_pos": (s.spec_accepted_per_pos / d).tolist() if d else [],
            "attention": msgspec.to_builtins(r.attention_counters()), "heads": msgspec.to_builtins(r.head_counters()),
            "graphs": bool(r.use_graphs), "modes": dict(self.modes), "forced_requests": s.forced_requests,
            "forced_tokens": s.forced_tokens, "forced_declined": s.forced_declined, "eos_free": s.eos_free.tolist(),
            "end_waits": s.end_waits, "preemptions": s.num_preemptions, "prefix_queries": s.prefix_queries,
            "prefix_hits": s.prefix_hits, "ckpts_taken": dict(s.ckpts_taken), "refines": s.refines,
            "precise_tokens": s.precise_tokens, "image_cuts": s.image_cuts,
            "grammar_errors": s.grammars.errors if s.grammars is not None else 0, "final": not self.core.busy}

    def close(self) -> None:
        self.runner.shutdown()
        if self.pixels is not None:
            self.pixels.close()


# -- the forced grammar ------------------------------------------------------------------------------------------------


class ForcedEach(Forced):
    """`tests.support.forced.Forced` with each request's own forced prefix (`forced`: rid -> the tokens its grammar
    forces at its start: the schema grammar's on scene, none on chat), so a forced run's final chunks take the tokens
    the request's real grammar forces there. A grammar's text names its request (`forced:<rid>`)."""

    def __init__(self, vocab_size: int, eos: int, refs: dict[int, list[int]], forced: dict[int, int]):
        super().__init__(vocab_size, eos, refs, 0)
        self.prefix = forced

    def compile(self, text: str) -> int:  # type: ignore[override]
        return int(text.partition(":")[2])

    def instance(self, compiled: int) -> ForcedMatcher:  # type: ignore[override]
        return ForcedMatcher(self.refs[compiled], self.eos[0], self.prefix[compiled])


# -- the captures ------------------------------------------------------------------------------------------------------


def valid_rows(slot: np.ndarray, pos: np.ndarray, tok: np.ndarray, by_slot: dict[int, int],
               true: dict[int, np.ndarray]) -> list[tuple[int, int, int]]:
    """A step's tokens computed from reference inputs only, as (token, rid, position): `Capture.drain`'s rule, a row's
    tokens (in order) up to its first input that is not its request's reference (`true`: prompt and answer)."""
    out, ok_row = [], {}
    for j in range(len(slot)):
        s = int(slot[j])
        seq = true.get(by_slot.get(s, -1))
        if seq is None:
            continue
        p = int(pos[j])
        ok = ok_row.get(s, True) and p < len(seq) and int(tok[j]) == int(seq[p])
        ok_row[s] = ok
        if ok:
            out.append((j, by_slot[s], p))
    return out


@dataclass
class Request:
    """One request's captures as the run goes (`Recorder`): `true` its prompt and reference answer."""

    true: np.ndarray
    hidden: Any = None  # [N, D], allocated at the first state
    hvalid: np.ndarray | None = None
    accept: list[tuple[int, int]] = field(default_factory=list)  # (window's first position, accepted)
    taps: dict[str, Any] | None = None
    tvalid: np.ndarray | None = None
    gdn: dict[int, list[tuple[int, int, dict[str, Any], Any]]] = field(default_factory=dict)  # layer -> rows
    attn: dict[int, dict[str, Any]] = field(default_factory=dict)  # layer -> tensors by position
    last_step: int = -1  # the last step launched with one of its rows
    saved: bool = False

    def put_hidden(self, ps: Any, rows: Any) -> None:
        import torch

        if self.hidden is None:
            self.hidden = torch.zeros((len(self.true), rows.shape[1]), dtype=rows.dtype)
            self.hvalid = np.zeros(len(self.true), bool)
        self.hidden.index_copy_(0, ps, rows)
        self.hvalid[ps.numpy()] = True

    def put_tap(self, name: str, layer: int, ps: Any, rows: Any, layers: int) -> None:
        import torch

        if self.taps is None:
            n, d = len(self.true), rows.shape[1]
            self.taps = {"stream": torch.zeros((layers + 1, n, d), dtype=rows.dtype),
                         "mixer": torch.zeros((layers, n, d), dtype=rows.dtype),
                         "mlp": torch.zeros((layers, n, d), dtype=rows.dtype)}
            self.tvalid = np.zeros(n, bool)
        self.taps[name][layer].index_copy_(0, ps, rows)
        if name == "stream" and layer == 0:
            self.tvalid[ps.numpy()] = True

    def put_attn(self, layer: int, ps: Any, parts: dict[str, Any]) -> None:
        import torch

        d = self.attn.get(layer)
        if d is None:
            n = len(self.true)
            d = self.attn[layer] = {k: torch.zeros((n, *v.shape[1:]), dtype=v.dtype) for k, v in parts.items()}
            d["valid"] = torch.zeros(n, dtype=torch.bool)
        for k, v in parts.items():
            d[k].index_copy_(0, ps, v)
        d["valid"][ps] = True


class Recorder:
    """The captures of a forced run (module docstring), from wrappers on the runner's instance: `launch_forward` (each
    plan's slot -> rid and its verify rows), `wait` (their accepted counts), `launch_sample` (each split step's tokens:
    slot, position, input, whether it opens a verify window; with `hid_out`, the final hidden states the runner keeps
    there), and eagerly the model's `forward` (the step's tokens from reference inputs; without `hid_out`, their final
    states), with `taps` each layer's stream, mixer and MLP outputs, and `core_layers` their kernels' inputs and
    outputs. `hid_out` None: with graphs only (the eager forward's output holds every token's state, Gemma 4's prompt
    included). The captured steps are drained on the host once their results are back; a finished request's captures go
    to `on_ready(rid, request)` once no step of it is in flight (None: they stay in `reqs`)."""

    def __init__(self, engine: Engine, true: dict[int, list[int]], *, taps: bool = False,
                 core_layers: Sequence[int] = (), on_ready: Callable[[int, Request], None] | None = None,
                 hid_out: bool | None = None):
        r = engine.runner
        self.engine, self.r, self.sched = engine, r, engine.sched
        self.eager = not r.use_graphs
        if (taps or core_layers) and not self.eager:
            raise ValueError("taps and core layers need eager steps (graphs off): a graph replay runs no Python")
        self.hid_out = not self.eager if hid_out is None else hid_out
        if not (self.hid_out or self.eager):
            raise ValueError("the forward's states need eager steps (graphs off)")
        self.reqs = {rid: Request(np.asarray(seq, np.int64)) for rid, seq in true.items()}
        self.true = {rid: q.true for rid, q in self.reqs.items()}
        self.every = engine.kind == "qwen"  # `hid_out` holds every token's state (Qwen under MTP), else the samples'
        self.layers = len(r.model.layers)
        self.on_ready = on_ready
        self.rids: dict[int, dict[int, int]] = {}  # step -> slot -> rid
        self.rows: dict[int, list[tuple[int, int]]] = {}  # step -> (slot, drafts) until its wait
        self.verify: dict[tuple[int, int], int] = {}  # (step, slot) -> a verify row's tokens out (1 + accepted)
        self.records: list[tuple[Any, ...]] = []  # (step, slot, pos, tok, opens a window, states, their tokens)
        self.waiting: set[int] = set()  # finished requests not yet handed to `on_ready`
        self.waited = -1
        self.cur: dict[int, int] = {}  # the launching step's slot -> rid
        self.sel: tuple[Any, list[tuple[int, Any, Any]]] | None = None  # this forward's reference tokens
        self.vmap: dict[int, tuple[int, int]] = {}  # this forward's token -> (rid, position), reference tokens only
        self.kinds = self._kinds()
        self._wrap_steps()
        if self.eager and not (self.hid_out and not taps and not core_layers):  # one host sync a step
            self._wrap_forward()
        if taps:
            self._wrap_layers()
        if core_layers:
            self._wrap_core(core_layers)

    def _kinds(self) -> list[str]:
        if self.engine.kind == "qwen":
            return ["full" if layer.is_attn else "gdn" for layer in self.r.model.layers]
        from llm_mini_infer.models.gemma4.model import SLIDING

        return ["sliding" if layer.self_attn.kind == SLIDING else "full" for layer in self.r.model.layers]

    # -- the steps --------------------------------------------------------------------------------------------------

    def _wrap_steps(self) -> None:
        import torch

        r = self.r
        lf, ls, wt = r.launch_forward, r.launch_sample, r.wait

        def launch_forward(plan: Any, _internal: bool = False, _capture: Any = None) -> Any:
            if not _internal and _capture is None and len(plan.sched_slot):
                step, slots = int(plan.step_id), plan.sched_slot.tolist()
                by = {s: int(self.sched.fe_rid[s]) for s in slots}
                self.rids[step] = self.cur = by
                self.rows[step] = list(zip(slots, plan.sched_n_draft.tolist()))
                for rid in by.values():
                    if rid in self.reqs:
                        self.reqs[rid].last_step = step
            return lf(plan, _internal=_internal, _capture=_capture)

        def launch_sample(h: Any, n_masks: int = 0, _heads: Any = None) -> Any:
            sp = h.split
            if sp is not None and not h.internal:  # positions as the body computes them, before its advance
                lay, t = sp[3], sp[3].T
                s, tb = r.static, r.tables
                pin, slot = s.pos[:t], s.slot[:t]
                pos = torch.where(pin >= 0, pin, tb.seq_len.index_select(0, slot) + (-1 - pin).clamp(min=0))
                hid = idx = None
                if self.hid_out:
                    if self.every:
                        hid = s.hid_out[:t].clone()
                    else:
                        idx, hid = s.s_idx[:lay.n_stok].clone(), s.hid_out[:lay.n_stok].clone()
                self.records.append((int(h.step_id), slot.clone(), pos, r._ids(t), pin == -1, hid, idx))
            out = ls(h, n_masks, _heads)
            if len(self.records) >= DRAIN or self.waiting:
                self.drain()
            return out

        def wait(h: Any) -> Any:
            res = wt(h)
            rows = self.rows.pop(int(h.step_id), None)
            if rows is not None:
                for (slot, nd), n in zip(rows, res.n_out.tolist()):
                    if nd:
                        self.verify[(int(h.step_id), slot)] = int(n)
                self.waited = max(self.waited, int(h.step_id))
            return res

        r.launch_forward, r.launch_sample, r.wait = launch_forward, launch_sample, wait

    def finished(self, rid: int) -> None:
        if rid in self.reqs and self.on_ready is not None:
            self.waiting.add(rid)

    def drain(self, final: bool = False) -> None:
        """The captured steps whose results are back (`final`: all of them) onto the host: the accepted count of each
        verify window whose first token is the reference's, short of its last token (a window there has nothing of the
        reference to verify), and with graphs the states from reference inputs; then the finished requests none of
        whose steps is in flight go to `on_ready`."""
        import torch

        ready = [rec for rec in self.records if final or rec[0] <= self.waited]
        self.records = [rec for rec in self.records if not (final or rec[0] <= self.waited)]
        for step, slot, pos, tok, first, hid, idx in ready:
            slot, pos, tok, first = (x.cpu().numpy() for x in (slot, pos, tok, first))
            valid = valid_rows(slot, pos, tok, self.rids.pop(step, {}), self.true)
            ok = {j: (rid, p) for j, rid, p in valid}
            for j in np.flatnonzero(first).tolist():
                n = self.verify.pop((step, int(slot[j])), None)
                if n is not None and j in ok and ok[j][1] < len(self.true[ok[j][0]]) - 1:
                    rid, p = ok[j]
                    self.reqs[rid].accept.append((p, n - 1))
            if hid is None or not valid:
                continue
            row_of = {int(j): i for i, j in enumerate(idx.cpu().tolist() if idx is not None else range(len(slot)))}
            pick = [(row_of[j], rid, p) for j, rid, p in valid if j in row_of]
            if pick:
                rows = hid.index_select(0, torch.tensor([i for i, _, _ in pick], device=hid.device)).cpu()
                for rid, ks, ps in _groups([(rid, p) for _, rid, p in pick]):
                    self.reqs[rid].put_hidden(ps, rows.index_select(0, ks))
        for rid in sorted(self.waiting):
            q = self.reqs[rid]
            if final or q.last_step <= self.waited:
                self.on_ready(rid, q)
                self.waiting.discard(rid)
                q.saved = True
                q.hidden = q.taps = None
                q.gdn, q.attn = {}, {}

    # -- eager steps: the forward and its layers ----------------------------------------------------------------------

    def _wrap_forward(self) -> None:
        import torch

        model, fwd = self.r.model, self.r.model.forward

        def forward(input_ids: Any, ctx: Any, select: Any = None, embeds: Any = None) -> Any:
            if select is not None:  # under MTP both runners take every token's state outside decode graphs
                raise RuntimeError("an eager forward selected rows: the captures need every token's state")
            t = input_ids.shape[0]
            valid = valid_rows(self.r.static.slot[:t].cpu().numpy(), ctx.meta.positions[:t].cpu().numpy(),
                               input_ids.cpu().numpy(), self.cur, self.true)
            self.vmap = {j: (rid, p) for j, rid, p in valid}
            self.sel = None
            if valid:
                js = torch.tensor([j for j, _, _ in valid], device=input_ids.device)
                self.sel = (js, _groups([(rid, p) for _, rid, p in valid]))
            out = fwd(input_ids, ctx, select=select, embeds=embeds)
            if not self.hid_out:
                self._put(out, "hidden")
            self.sel = None
            self.vmap = {}
            return out

        model.forward = forward

    def _put(self, x: Any, name: str, layer: int = 0) -> None:
        """This forward's reference tokens' rows of `x` (a [T, D] tensor of the step's tokens) to their requests."""
        if self.sel is None:
            return
        js, groups = self.sel
        rows = x.index_select(0, js).cpu()
        for rid, ks, ps in groups:
            part = rows.index_select(0, ks)
            if name == "hidden":
                self.reqs[rid].put_hidden(ps, part)
            else:
                self.reqs[rid].put_tap(name, layer, ps, part, self.layers)

    def _out_tap(self, f: Callable[..., Any], name: str, i: int) -> Callable[..., Any]:
        def tapped(*a: Any) -> Any:
            out = f(*a)
            self._put(out, name, i)
            return out

        return tapped

    def _wrap_layers(self) -> None:
        """The taps (module docstring): each layer's stream, mixer and MLP outputs, by methods on the instances."""
        model, layers = self.r.model, self.r.model.layers
        last = len(layers) - 1
        for i, layer in enumerate(layers):
            mixer = layer.self_attn if self.kinds[i] != "gdn" else layer.linear_attn
            mixer.forward = self._out_tap(mixer.forward, "mixer", i)
            layer.mlp.forward = self._out_tap(layer.mlp.forward, "mlp", i)
            if self.engine.kind == "qwen":
                layer.input_layernorm.apply = self._qwen_stream(layer.input_layernorm.apply, i)
            else:
                layer.forward = self._gemma_stream(layer, layer.forward, i, model.norm.weight if i == last else None)
        if self.engine.kind == "qwen":  # the final norm's sum: the last MLP output and the post-attention residual
            end = layers[last]
            post, mlp = end.post_attention_layernorm.apply, end.mlp.forward
            res: list[Any] = [None]

            def post_apply(*a: Any) -> Any:
                h, res[0] = post(*a)
                return h, res[0]

            def mlp_end(*a: Any) -> Any:
                m = mlp(*a)
                if self.sel is not None:
                    self._put(model.norm.forward_add(m, res[0])[1], "stream", last + 1)
                return m

            end.post_attention_layernorm.apply, end.mlp.forward = post_apply, mlp_end

    def _qwen_stream(self, apply: Callable[..., Any], i: int) -> Callable[..., Any]:
        def tapped(*a: Any) -> Any:
            h, res = apply(*a)  # res: the stream entering layer i (x + residual, rounded; layer 0: x)
            self._put(res, "stream", i)
            return h, res

        return tapped

    def _gemma_stream(self, layer: Any, f: Callable[..., Any], i: int, norm: Any) -> Callable[..., Any]:
        from llm_mini_infer.kernels import gemma_ops as gops

        def tapped(x: Any, residual: Any, ctx: Any) -> Any:
            self._put(residual, "stream", i)
            m, mid = f(x, residual, ctx)
            if norm is not None and self.sel is not None:  # the last sandwich's residual, as the model loop forms it
                r, _ = gops.sandwich(m, mid, layer.post_feedforward_layernorm.weight, norm, layer.eps,
                                     scalar=layer.layer_scalar)
                self._put(r, "stream", i + 1)
            return m, mid

        return tapped

    # -- eager steps: kernels in situ ----------------------------------------------------------------------------------

    def _wrap_core(self, core_layers: Sequence[int]) -> None:
        """`--core-layers`: Qwen's GDN prefill (`GDN.prefill_rows`, the prefill kernel's call) and full attention (the
        target's `attn.run`), Gemma 4's full attention (`attn_f.run`), on the chosen model layers."""
        r, layers = self.r, self.r.model.layers
        for i in core_layers:
            if not 0 <= i < len(layers):
                raise ValueError(f"core layer {i}: the model has {len(layers)} layers")
            if self.kinds[i] == "sliding":
                raise ValueError(f"core layer {i}: a sliding layer (Gemma 4's core captures are its full layers')")
        gdn = {layers[i].linear_attn.index: i for i in core_layers if self.kinds[i] == "gdn"}
        full = {i for i in core_layers if self.kinds[i] == "full"}
        if gdn:
            self._wrap_gdn(gdn)
        if full and self.engine.kind == "qwen":
            idx = {layers[i].self_attn.index: i for i in full}
            run, pools = r.attn.run, r.pools

            def qwen_run(meta: Any, index: int, q: Any, out: Any) -> Any:
                res = run(meta, index, q, out)
                i = idx.get(index)
                if i is not None and self.sel is not None:
                    js = self.sel[0]
                    page, off = meta.kv_page.index_select(0, js), meta.kv_off.index_select(0, js)
                    kv = pools.kv[index]
                    self._put_attn(i, {"q": q.index_select(0, js), "k": kv[page, 0, off], "v": kv[page, 1, off],
                                       "o": out.index_select(0, js)})
                return res

            r.attn.run = qwen_run
        elif full:
            from llm_mini_infer.kernels.gemma_ops import kv_read

            mods = {id(layers[i].self_attn): i for i in full}
            run, pools = r.attn_f.run, r.pools

            def gemma_run(meta: Any, layer: Any, q: Any, out: Any) -> Any:
                res = run(meta, layer, q, out)
                i = mods.get(id(layer))  # the target's layers only (the assistant's are its own modules)
                if i is not None and self.sel is not None:
                    js = self.sel[0]
                    page, off = meta.f_page.index_select(0, js), meta.f_off.index_select(0, js)
                    kv = pools.fkv[layer.index]
                    planes = {"v": kv[page, off]} if layer.k_eq_v else {"k": kv[page, 0, off], "v": kv[page, 1, off]}
                    planes = {n: kv_read(x, q.dtype) for n, x in planes.items()}  # an FP8 pool's, dequantized
                    self._put_attn(i, {"q": q.index_select(0, js), **planes, "o": out.index_select(0, js)})
                return res

            r.attn_f.run = gemma_run

    def _put_attn(self, i: int, parts: dict[str, Any]) -> None:
        rows = {k: v.cpu() for k, v in parts.items()}
        for rid, ks, ps in self.sel[1]:
            self.reqs[rid].put_attn(i, ps, {k: v.index_select(0, ks) for k, v in rows.items()})

    def _wrap_gdn(self, layers: dict[int, int]) -> None:
        """The prefill kernel's call (`GDN.prefill`: q, k, v, alpha, beta, the layer's state pool, the rows' slots, the
        step's meta) on the chosen GDN layers, its rows from reference inputs kept with its output and their initial
        states."""
        g = self.r.gdn
        rows_of, prefill = g.prefill_rows, g.prefill

        def prefill_rows(meta: Any, layer: Any, x: Any, a: Any, b: Any) -> Any:
            i = layers.get(layer.index)
            if i is None or self.sel is None:
                return rows_of(meta, layer, x, a, b)
            tok0 = meta.num_tokens - x.shape[0]  # the step's first prefill token

            def kernel(q: Any, k: Any, v: Any, alpha: Any, beta: Any, pool: Any, slots: Any, m: Any) -> Any:
                init0 = pool[slots]  # a copy, before the call writes the final states over them
                o = prefill(q, k, v, alpha, beta, pool, slots, m)
                self._put_gdn(i, tok0, m.pf_cu_host, dict(q=q, k=k, v=v, alpha=alpha, beta=beta, o=o), init0)
                return o

            g.prefill = kernel
            try:
                return rows_of(meta, layer, x, a, b)
            finally:
                g.prefill = prefill

        g.prefill_rows = prefill_rows

    def _put_gdn(self, i: int, tok0: int, cu: np.ndarray, parts: dict[str, Any], init: Any) -> None:
        for r in range(len(cu) - 1):
            a, b = int(cu[r]), int(cu[r + 1])
            hits = [self.vmap.get(tok0 + j) for j in range(a, b)]
            if not hits or None in hits or len({rid for rid, _ in hits}) > 1:
                continue  # a row with an input off the reference (never under forcing: explicit tokens)
            rid, p0 = hits[0]
            self.reqs[rid].gdn.setdefault(i, []).append(
                (p0, b - a, {k: v[a:b].cpu() for k, v in parts.items()}, init[r].cpu()))


def _groups(pairs: list[tuple[int, int]]) -> list[tuple[int, Any, Any]]:
    """(rid, position) per row -> per rid: (rid, its rows' indices, their positions), int64 CPU tensors."""
    import torch

    by: dict[int, tuple[list[int], list[int]]] = {}
    for k, (rid, p) in enumerate(pairs):
        ks, ps = by.setdefault(rid, ([], []))
        ks.append(k)
        ps.append(p)
    return [(rid, torch.tensor(ks), torch.tensor(ps)) for rid, (ks, ps) in by.items()]


def accept_summary(accept: list[tuple[int, int]], k: int) -> dict[str, Any]:
    a = np.asarray([n for _, n in accept], np.int64)
    if not len(a):
        return {"verify_steps": 0, "accepted": 0, "tokens_per_verify": None, "per_pos": []}
    return {"verify_steps": len(a), "accepted": int(a.sum()), "tokens_per_verify": float(1 + a.mean()),
            "per_pos": [float((a > j).mean()) for j in range(k)]}


# -- the KV fake-quant -------------------------------------------------------------------------------------------------


def e4m3_round(x: Any, scale: float = 1.0) -> Any:
    """x stored as float8_e4m3fn at `scale` and read back in x's dtype: clamp(x / scale, ±448) (PTX's `.satfinite`: an
    overflow or an infinity saturates, a NaN stays NaN), rounded to nearest even, times `scale`. Every E4M3 value is a
    BF16 value, so with a power-of-two scale the read is exact: what an FP8 pool dequantized into BF16 holds."""
    import torch

    y = x.float() if scale == 1.0 else x.float() / scale
    y = y.clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).float()
    return (y if scale == 1.0 else y * scale).to(x.dtype)


def pow2_scale(amax: float) -> float:
    """The smallest power of two s >= 1 with amax / s <= 448 (1.0 for a NaN or an infinity)."""
    return 2.0 ** max(0, math.ceil(math.log2(amax / E4M3_MAX))) if amax > E4M3_MAX and math.isfinite(amax) else 1.0


def load_kv_scales(path: str | Path) -> dict[tuple[str, int], tuple[float, float]]:
    """`--kv-scales`: a JSON object {"<kind>:<pool index>": [K's scale, V's scale]} (kind "full" or "sliding"), or a
    forced run's run.json, whose `kv_fake_quant.pow2_scales` is one (the scales its largest values ask for)."""
    d = json.loads(Path(path).read_text())
    if "kv_fake_quant" in d:
        d = (d["kv_fake_quant"] or {}).get("pow2_scales")
        if d is None:
            raise ValueError(f"{path}: a forced run without --kv-dtype, so no KV scales")
    out = {}
    for key, ss in d.items():
        kind, _, i = key.partition(":")
        if kind not in ("full", "sliding") or not i.isdigit() or len(ss) != 2:
            raise ValueError(f"{path}: KV scales are {{\"full:3\": [k, v], ...}}, not {key!r}: {ss!r}")
        out[(kind, int(i))] = (float(ss[0]), float(ss[1]))
    return out


class KvFakeQuant:
    """`--kv-dtype`: the KV an FP8 pool would hold, in today's BF16 pools. Each layer's KV write goes through one of two
    context methods (`Gemma4Context.qkv_store`, Qwen's `ForwardContext.qk_rope_store`: the store ops' only callers);
    wrapped here, each rounds the step's written slots in place (`e4m3_round`) right after the store, before that
    layer's attention reads them, so a step's own new keys are rounded too. Graph capture runs the Python once: the
    rounding is in every graph. `fp8`: Gemma 4's full layers (its one stored plane with `attention_k_eq_v`, else K and
    V) and Qwen's every attention layer, the MTP layer included (the pool's last; K after its norm and RoPE, V);
    `fp8-all`: Gemma 4's sliding layers too. The drafter reads the rounded KV like the target, and the precise prefix
    and a hit's recomputed KV write through the same methods. Counters per layer and plane stay on the device (no host
    sync inside a graph): the values real tokens wrote (padding rows write the null page, 0), those beyond 448 × the
    scale (saturated by the clamp) and the largest magnitude. `scales`: (kind, pool index) -> (K's, V's), each a power
    of two (default 1.0, the engines' default without checkpoint scales; neither checkpoint ships any). Its temporaries
    count in the memory profile, so an arm's pools can hold a few pages fewer than its BF16 twin's (`--num-pages` pins
    them)."""

    SLOTS = 256  # (kind, layer) pairs counted

    def __init__(self, dtype: str, device: str, scales: dict[tuple[str, int], tuple[float, float]] | None = None):
        import torch

        if dtype not in KV_DTYPES:
            raise ValueError(f"KV dtype {dtype!r}: one of {', '.join(KV_DTYPES)}")
        for key, ss in (scales or {}).items():
            if not all(s > 0 and math.frexp(s)[0] == 0.5 for s in ss):
                raise ValueError(f"KV scales {ss} of {key}: powers of two (their dequantization is exact)")
        self.dtype, self.scales = dtype, dict(scales or {})
        self.slot: dict[tuple[str, int], int] = {}
        self.values = torch.zeros((self.SLOTS, 2), dtype=torch.int64, device=device)
        self.saturated = torch.zeros_like(self.values)
        self.amax = torch.zeros((self.SLOTS, 2), dtype=torch.float32, device=device)
        self._undo: list[tuple[Any, str, Any]] = []

    @property
    def scaled(self) -> bool:
        return any(s != 1.0 for ss in self.scales.values() for s in ss)

    def settings(self) -> dict[str, Any]:
        return {"dtype": self.dtype, "format": "float8_e4m3fn", "clamp": E4M3_MAX,
                "layers": "full and sliding" if self.dtype == "fp8-all" else "full",
                "scales": {f"{k}:{i}": list(ss) for (k, i), ss in sorted(self.scales.items())} or 1.0}

    def install(self) -> None:
        """The wrappers, before the engine is built (its memory profile and graph capture store KV)."""
        from llm_mini_infer.models.gemma4.backend import SLIDING, Gemma4Context
        from llm_mini_infer.models.qwen3_5.backend import ForwardContext

        gemma, qwen = Gemma4Context.qkv_store, ForwardContext.qk_rope_store
        every = self.dtype == "fp8-all"

        def gemma_store(ctx: Any, layer: Any, qkv: Any) -> Any:
            q = gemma(ctx, layer, qkv)
            sliding = layer.kind == SLIDING
            if every or not sliding:
                m, p = ctx.meta, ctx.pools
                kv = p.wkv[layer.index] if sliding else p.fkv[layer.index]
                planes = {1: kv} if kv.dim() == 4 else {0: kv[:, 0], 1: kv[:, 1]}  # k_eq_v: V alone
                self.round("sliding" if sliding else "full", layer.index, planes,
                           m.w_page if sliding else m.f_page, m.w_off if sliding else m.f_off)
            return q

        def qwen_store(ctx: Any, layer: Any, qkv: Any) -> Any:
            q = qwen(ctx, layer, qkv)
            kv = ctx.pools.kv[layer.index]
            self.round("full", layer.index, {0: kv[:, 0], 1: kv[:, 1]}, ctx.meta.kv_page, ctx.meta.kv_off)
            return q

        self._undo = [(Gemma4Context, "qkv_store", gemma), (ForwardContext, "qk_rope_store", qwen)]
        Gemma4Context.qkv_store, ForwardContext.qk_rope_store = gemma_store, qwen_store

    def uninstall(self) -> None:
        for cls, name, f in self._undo:
            setattr(cls, name, f)
        self._undo = []

    def round(self, kind: str, index: int, planes: dict[int, Any], page: Any, off: Any) -> None:
        """Rounds `planes` (0: K, 1: V; [pages, page_size, heads, dim] views) at the step's (`page`, `off`) slots."""
        import torch

        if not page.numel():
            return
        slot = self.slot.setdefault((kind, index), len(self.slot))
        if slot >= self.SLOTS:
            raise RuntimeError(f"more than {self.SLOTS} KV layers")
        real = (page != 0)[:, None, None]
        scales = self.scales.get((kind, index), (1.0, 1.0))
        if any(plane.dtype == torch.uint8 for plane in planes.values()):
            raise ValueError("the KV fake-quant rounds a BF16 pool: this one is FP8 already (--kv-cache-dtype fp8)")
        for j, plane in planes.items():
            x = plane[page, off]
            a = torch.where(real, x.abs().float(), 0.0)
            self.values[slot, j] += real.sum() * x[0].numel()
            self.saturated[slot, j] += (a > E4M3_MAX * scales[j]).sum()
            self.amax[slot, j] = torch.maximum(self.amax[slot, j], a.amax())
            plane[page, off] = e4m3_round(x, scales[j])

    def reset(self) -> None:
        """Zeroes the counters in place (a captured graph holds their addresses)."""
        for t in (self.values, self.saturated, self.amax):
            t.zero_()

    def report(self) -> dict[str, Any]:
        """The counters by layer (`kind:index`, K and V; a plane never written has 0 values), their totals, and the
        power-of-two scales (>= 1) that would have kept every value within 448 (`--kv-scales` takes them)."""
        vals, sat, amax = self.values.tolist(), self.saturated.tolist(), self.amax.tolist()
        layers = {f"{k}:{i}": {"values": vals[s], "saturated": sat[s], "amax": amax[s]}
                  for (k, i), s in sorted(self.slot.items())}
        return {**self.settings(), "layers": layers,
                "values": sum(sum(e["values"]) for e in layers.values()),
                "saturated": sum(sum(e["saturated"]) for e in layers.values()),
                "amax": max((max(e["amax"]) for e in layers.values()), default=0.0),
                "pow2_scales": {key: [pow2_scale(a) for a in e["amax"]] for key, e in layers.items()}}


# -- references --------------------------------------------------------------------------------------------------------


def make_refs(model: str | Path, requests: Sequence[dict[str, Any]], *, assistant: str | Path | None = None,
              grammars: dict[str, str] | None = None, opts: Options | None = None, max_tokens: int = MAX_TOKENS,
              switch: str = "on", cache: bool = False, graphs: bool = True, concurrency: int = 1,
              options: dict[str, Any] | None = None) -> dict[str, Any]:
    """Greedy references (module docstring): `requests` as `build_requests` gives them (rid, workload, index, parts,
    grammar, forced), each request's grammar its text in `grammars` (by name; "" none), through the engine of arm
    `switch`. Returns the references (refs.json's content); `options`: what built the requests, recorded."""
    from llm_mini_infer.engine.messages import OPENAI_FINISH

    model_dir = resolve(model)
    asst = resolve(assistant) if assistant else None
    opts = opts or Options()
    texts = grammars or {}
    unknown = sorted({q["grammar"] for q in requests if q["grammar"] and q["grammar"] not in texts})
    if unknown:
        raise ValueError(f"no grammar text for {', '.join(unknown)}")
    g = None
    if (model_dir / "tokenizer.json").exists():  # structured output on, as the server's with a tokenizer
        g = _grammars(model_dir)
    elif any(q["grammar"] for q in requests):
        raise ValueError(f"{model_dir} has no tokenizer.json: its requests can have no grammar")
    eng = Engine(model_dir, asst, opts, switch=switch, cache=cache, graphs=graphs, grammars=g, requests=requests)
    try:
        by = {q["rid"]: q for q in requests}
        prompts = {rid: eng.expand(q["parts"]) for rid, q in by.items()}
        max_len = eng.settings["max_model_len"]  # a forced run asks the answer's length + 2
        long = sorted(rid for rid, (ids, _) in prompts.items() if len(ids) + max_tokens + 2 > max_len)
        if long:
            raise ValueError(f"requests {long}: the prompt and {max_tokens} + 2 answer tokens pass max_model_len "
                             f"{max_len} (raise --max-model-len: 65,536 for --input-tokens 32768)")
        t0 = time.perf_counter()

        def add(rid: int) -> None:
            eng.add(rid, *prompts[rid], max_tokens=max_tokens, grammar=texts.get(by[rid]["grammar"], ""))

        eng.drive(list(by), add, concurrency,
                  lambda rid: log.info("reference %d: %d tokens", rid, len(eng.out[rid])))
        out = []
        for rid, q in by.items():
            ans = eng.out[rid]
            if ans and ans[-1] in eng.eos:
                ans = ans[:-1]
            ids, images = prompts[rid]
            out.append({**q, "prompt": ids, "prompt_len": len(ids),
                        "images": [[ref.offset, ref.tokens] for ref in images], "answer": ans,
                        "finish": OPENAI_FINISH.get(eng.finish.get(rid, 0), "none")})
        return {"model": eng.kind, "model_name": str(model), "model_dir": str(model_dir),
                "assistant_name": str(assistant) if assistant else None, "assistant_dir": str(asst) if asst else None,
                "options": {**(options or {}), "soft_tokens": opts.soft_tokens, "max_tokens": max_tokens},
                "settings": eng.settings, "counters": eng.counters(), "grammars": texts, "requests": out,
                "timings": {"load_s": round(eng.load_s, 2), "run_s": round(time.perf_counter() - t0, 2)}}
    finally:
        eng.close()


# -- forced runs -------------------------------------------------------------------------------------------------------


def run_arm(refs: dict[str, Any] | str | Path, *, switch: str, cache: bool = False, graphs: bool = True,
            taps: bool = False, core_layers: Sequence[int] = (), concurrency: int = 1,
            rids: Sequence[int] | None = None, out: str | Path | None = None, opts: Options | None = None,
            model: str | Path | None = None, assistant: str | Path | None = None,
            hid_out: bool | None = None, kv_dtype: str | None = None,
            kv_scales: dict[tuple[str, int], tuple[float, float]] | None = None) -> dict[str, Any]:
    """One arm's forced run of the references (`refs`: refs.json's content or its path; module docstring): its captures
    to `<out>/<arm>/r<rid>.pt` and `<out>/<arm>/run.json` (`out` default: the references' folder). `opts`: the engine's
    settings (default: `Options()` at the references' max_model_len), always with free drafts (no draft universe;
    `draft_vocab` None: the whole head) and the references' image budget. `taps` and `core_layers` (which needs `taps`)
    run with graphs off. `model`, `assistant`: other folders of the same checkpoints. `hid_out`: where the final states
    come from (`Recorder`; None: `hid_out` with graphs, the forward's output eagerly). `kv_dtype`, `kv_scales`: the KV
    fake-quant (`KvFakeQuant`), its counters in run.json's `kv_fake_quant`. Returns run.json's content; raises after
    writing it when an output is not its reference."""
    import torch

    from llm_mini_infer.grammar.engine import Grammars

    src = None
    if not isinstance(refs, dict):
        src = Path(refs)
        refs = json.loads(src.read_text())
    if (taps or core_layers) and graphs:
        raise ValueError("taps and core layers need eager steps: graphs off")
    if core_layers and not taps:
        raise ValueError("core layers need taps (--taps): their captures live in the taps (taps.core)")
    model_dir = resolve(model) if model else Path(refs["model_dir"])
    asst = resolve(assistant) if assistant else Path(refs["assistant_dir"]) if refs.get("assistant_dir") else None
    if kv_dtype == "fp8-all" and model_kind(model_dir) != "gemma":
        raise ValueError("--kv-dtype fp8-all adds Gemma 4's sliding layers: fp8 rounds every layer of this model")
    opts = opts or Options(max_model_len=refs.get("settings", {}).get("max_model_len", MAX_LEN))
    if kv_dtype is not None and opts.kv_cache_dtype != "auto":
        raise ValueError("--kv-dtype rounds a BF16 pool's KV as an FP8 pool would hold it; with --kv-cache-dtype fp8 "
                         "the pool is FP8 already")
    opts = replace(opts, draft_universe=False, draft_vocab=opts.draft_vocab or 0,
                   soft_tokens=refs.get("options", {}).get("soft_tokens", opts.soft_tokens))
    want = set(rids) if rids is not None else None
    reqs = [q for q in refs["requests"] if want is None or q["rid"] in want]
    if want is not None and want - {q["rid"] for q in reqs}:
        raise ValueError(f"no reference for rids {sorted(want - {q['rid'] for q in reqs})}")
    kvq = KvFakeQuant(kv_dtype, opts.device, kv_scales) if kv_dtype is not None else None
    arm = arm_name(switch, cache, graphs, kv_dtype, kvq is not None and kvq.scaled, opts.kv_cache_dtype)
    folder = Path(out) if out else src.parent if src else results_dir("forced")
    (folder / arm).mkdir(parents=True, exist_ok=True)
    _, eos = grammar_eos(model_dir)
    answers = {q["rid"]: [int(t) for t in q["answer"]] for q in reqs}
    forced = Grammars(ForcedEach(vocab_size(model_dir), eos, answers, {q["rid"]: int(q["forced"]) for q in reqs}))
    if kvq is not None:
        kvq.install()
    try:
        eng = Engine(model_dir, asst, opts, switch=switch, cache=cache, graphs=graphs, grammars=forced, requests=reqs)
    except BaseException:
        if kvq is not None:
            kvq.uninstall()
        raise
    early = taps  # big captures go to disk as their requests finish
    summary: dict[str, Any] = {}
    try:
        if kvq is not None:
            kvq.reset()  # the memory profile and graph capture stored KV too
        by = {q["rid"]: q for q in reqs}
        prompts = {rid: eng.expand(q["parts"]) for rid, q in by.items()}
        for rid, (ids, _) in prompts.items():
            if ids != by[rid]["prompt"]:
                raise ValueError(f"request {rid}: its prompt rebuilds differently from the references' (another "
                                 "checkpoint, tokenizer or image budget?)")

        conf = {**eng.settings, "arm": arm, "taps": taps, "core_layers": list(core_layers),
                "concurrency": concurrency, "teacher_forced": True,
                "kv_fake_quant": kvq.settings() if kvq is not None else None}

        def save(rid: int, q: Request) -> None:
            ids, images = prompts[rid]
            cap = capture(eng, q, arm, conf, ids, images, forced=int(by[rid]["forced"]),
                          kinds=rec.kinds if taps else None, final=not early)
            torch.save(cap, folder / arm / f"r{rid}.pt")
            summary[rid] = {"rid": rid, "workload": by[rid]["workload"], "index": by[rid]["index"],
                            "prompt_len": len(ids), "answer_len": len(answers[rid]), "cached": eng.cached.get(rid, 0),
                            "hvalid": int(q.hvalid.sum()) if q.hvalid is not None else 0,
                            "tvalid": int(q.tvalid.sum()) if q.tvalid is not None else None,
                            **accept_summary(q.accept, opts.k)}

        rec = Recorder(eng, {rid: ids + answers[rid] for rid, (ids, _) in prompts.items()}, taps=taps,
                       core_layers=core_layers, on_ready=save if early else None, hid_out=hid_out)
        t0 = time.perf_counter()

        def add(rid: int) -> None:
            eng.add(rid, *prompts[rid], max_tokens=len(answers[rid]) + 2, grammar=f"forced:{rid}")

        def finished(rid: int) -> None:
            rec.finished(rid)
            rec.drain()
            n = accept_summary(rec.reqs[rid].accept, opts.k)
            log.info("%s: request %d finished, %d verify steps, %s tokens a step (%.0f s)", arm, rid,
                     n["verify_steps"], f"{n['tokens_per_verify']:.3f}" if n["tokens_per_verify"] else "-",
                     time.perf_counter() - t0)

        eng.drive([q["rid"] for q in reqs], add, concurrency, finished)
        rec.drain(final=True)
        run_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        for rid, q in rec.reqs.items():
            if not q.saved:
                save(rid, q)
        wrong = sorted(rid for rid in by if eng.out[rid][:len(answers[rid])] != answers[rid])
        res = {"model": eng.kind, "model_dir": str(model_dir), "assistant_dir": str(asst) if asst else None,
               "arm": arm, "settings": conf, "counters": eng.counters(),
               "kv_fake_quant": kvq.report() if kvq is not None else None,
               "requests": [summary[rid] for rid in sorted(summary)], "wrong": wrong,
               "timings": {"load_s": round(eng.load_s, 2), "run_s": round(run_s, 2),
                           "save_s": round(time.perf_counter() - t1, 2)}}
        (folder / arm / "run.json").write_text(json.dumps(res, indent=1, default=str))
        if wrong or res["counters"]["grammar_errors"]:
            raise RuntimeError(f"{arm}: outputs off their references for rids {wrong}, "
                               f"{res['counters']['grammar_errors']} grammar errors")
        return res
    finally:
        try:
            eng.close()
        finally:
            if kvq is not None:
                kvq.uninstall()


def capture(eng: Engine, q: Request, arm: str, conf: dict[str, Any], ids: list[int], images: list[Any], *,
            forced: int, kinds: list[str] | None, final: bool) -> dict[str, Any]:
    """A request's capture (module docstring)."""
    import torch

    from llm_mini_infer.vision.family import ImageSize, Span

    r, n = eng.runner, len(q.true)
    rope = None
    if eng.kind == "qwen" and images:
        spans = [Span(ref.offset, ImageSize(ref.height, ref.width, (ref.grid_t, ref.grid_h, ref.grid_w), ref.tokens))
                 for ref in images]
        got = eng.fam.positions(n, spans)[0]
        rope = torch.from_numpy(np.asarray(got, np.int64)) if got is not None else None
    hidden = q.hidden if q.hidden is not None else torch.zeros((n, r.cfg.hidden_size), dtype=r.dtype)
    hvalid = q.hvalid if q.hvalid is not None else np.zeros(n, bool)
    counters = eng.counters()
    counters["final"] = final and counters["final"]
    cap: dict[str, Any] = {
        "model": eng.kind, "model_dir": str(eng.model_dir),
        "assistant_dir": str(eng.assistant) if eng.assistant else None, "arm": arm, "settings": conf,
        "tokens": torch.from_numpy(q.true.copy()), "prompt_len": len(ids), "forced": forced,
        "images": [(ref.offset, ref.tokens) for ref in images], "rope": rope, "hidden": hidden,
        "hvalid": torch.from_numpy(hvalid), "accept": torch.tensor([a for _, a in q.accept], dtype=torch.int64),
        "accept_pos": torch.tensor([p for p, _ in q.accept], dtype=torch.int64), "counters": counters}
    if kinds is not None:
        taps = q.taps or {}
        if not taps:  # no step of it ran eagerly with a reference input (never: its prompt is the reference's)
            d, L = r.cfg.hidden_size, len(kinds)
            taps = {"stream": torch.zeros((L + 1, n, d), dtype=r.dtype),
                    "mixer": torch.zeros((L, n, d), dtype=r.dtype), "mlp": torch.zeros((L, n, d), dtype=r.dtype)}
        core: dict[int, dict[str, Any]] = {}
        for i, rows in q.gdn.items():
            rows = sorted(rows, key=lambda row: row[0])
            core[i] = {k: torch.cat([row[2][k] for row in rows]) for k in rows[0][2]}
            core[i].update(init=torch.stack([row[3] for row in rows]), rows=[(p0, m) for p0, m, _, _ in rows])
        core.update(q.attn)
        tvalid = q.tvalid if q.tvalid is not None else np.zeros(n, bool)
        cap["taps"] = {"kinds": kinds, **taps, "tvalid": torch.from_numpy(tvalid), "core": core}
    return cap


# -- reports -----------------------------------------------------------------------------------------------------------


def table(rows: list[dict[str, Any]], cols: list[tuple[str, str]]) -> str:
    """Markdown (`md`): `cols` (key, header) of each row."""
    return md([h for _, h in cols], [[r.get(k) for k, _ in cols] for r in rows])


def refs_table(refs: dict[str, Any]) -> str:
    rows = [{**q, "n_images": len(q["images"]), "answer_len": len(q["answer"])} for q in refs["requests"]]
    return table(rows, [("rid", "rid"), ("workload", "workload"), ("index", "index"), ("prompt_len", "prompt"),
                        ("n_images", "images"), ("forced", "forced"), ("answer_len", "answer"), ("finish", "finish")])


def run_tables(res: dict[str, Any]) -> str:
    c = res["counters"]
    a = c["attention"]
    head = {"arm": res["arm"], "requests": len(res["requests"]),
            "tokens": sum(q["answer_len"] for q in res["requests"]), "drafts": c["spec_drafts"],
            "tpv": c["tokens_per_verify"], "per_pos": c["accept_per_pos"], "steps": c["steps"],
            "modes": " ".join(f"{k} {v}" for k, v in sorted(c["modes"].items())),
            "plans": f"{a['replans']}/{a['kept']}/{a['unsplit']}", "forced": c["forced_tokens"],
            "hits": c["prefix_hits"], "refines": c["refines"]}
    top = table([head], [("arm", "arm"), ("requests", "requests"), ("tokens", "answer tokens"),
                         ("drafts", "verify steps"), ("tpv", "tokens / verify"), ("per_pos", "accepted by position"),
                         ("steps", "steps"), ("modes", "bodies"), ("plans", "replans/kept/unsplit"),
                         ("forced", "forced tokens"), ("hits", "cache hits"), ("refines", "refines")])
    per = table(res["requests"], [("rid", "rid"), ("workload", "workload"), ("index", "index"),
                                  ("prompt_len", "prompt"), ("cached", "cached"), ("answer_len", "answer"),
                                  ("verify_steps", "verify"), ("tokens_per_verify", "tokens / verify"),
                                  ("per_pos", "by position"), ("hvalid", "states"), ("tvalid", "taps")])
    kv = res.get("kv_fake_quant")
    if not kv:
        return f"{top}\n\n{per}"
    rows = [{"layer": key, "values": e["values"], "saturated": e["saturated"], "amax": [round(a, 2) for a in e["amax"]],
             "scales": kv["pow2_scales"][key]} for key, e in kv["layers"].items()]
    lay = table(rows, [("layer", "KV layer"), ("values", "values K V"), ("saturated", "saturated K V"),
                       ("amax", "max |x| K V"), ("scales", "power-of-two scales")])
    return f"{top}\n\n{per}\n\n{kv['dtype']} KV, {kv['saturated']} of {kv['values']} values saturated:\n\n{lay}"


# -- the command line --------------------------------------------------------------------------------------------------


def _engine_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("the engine (the module docstring's defaults)")
    g.add_argument("--assistant", default=None, help="Gemma 4's assistant: a folder or a repo id")
    g.add_argument("--device", default="cuda", help="cuda, or cpu (the tiny models: fp32, the torch references)")
    g.add_argument("--dtype", default="auto", choices=("auto", "bfloat16", "float32"),
                   help="the runner's dtype (auto: BF16 on CUDA, fp32 on the CPU)")
    g.add_argument("--k", type=int, default=3,
                   help="drafts a step (the server's: 3 from Qwen3.8's MTP layer, 5 from an assistant)")
    g.add_argument("--max-model-len", type=int, default=None,
                   help=f"model.max_model_len (refs: {MAX_LEN:,}, batch invariance's default range; run: the "
                        "references')")
    g.add_argument("--budget", type=int, default=2048, help="scheduler.max_num_batched_tokens")
    g.add_argument("--max-num-seqs", type=int, default=64, help="scheduler.max_num_seqs")
    g.add_argument("--page-size", type=int, default=16, help="dev.cache.page_size (Gemma 4: the window page)")
    g.add_argument("--memory-gb", type=float, default=None,
                   help="pool memory to report instead of the runner's memory profile (which the mixed-step bound's "
                        "step costs come from)")
    g.add_argument("--num-pages", type=int, default=None, help="the KV pool's (Gemma 4: the arena's) pages")
    g.add_argument("--ckpt-slots", type=int, default=None, help="Qwen's checkpoint state slots (with --cache)")
    g.add_argument("--row-tokens", type=int, default=None,
                   help="dev.scheduler.mixed_step_row_tokens (default: derived from the profile's step costs; 0: off)")
    g.add_argument("--precise", default="reuse", choices=("reuse", "bf16", "fp8"), help="runner.cuda.fp8_cold_prefill")
    g.add_argument("--pdl", action=argparse.BooleanOptionalAction, default=True,
                   help="programmatic dependent launch (dev.runner.cuda.pdl)")
    g.add_argument("--sparse-head", action=argparse.BooleanOptionalAction, default=True,
                   help="the head over the grammar masks' tokens (dev.runner.cuda.sparse_head)")
    g.add_argument("--always-split", action=argparse.BooleanOptionalAction, default=True,
                   help="forward and sampling graphs apart at every step (dev.runner.cuda.always_split)")
    g.add_argument("--check", action="store_true", help="compare the device tables with the host's every step")
    g.add_argument("--kv-cache-dtype", default="auto", choices=("auto", "fp8"),
                   help="cache.kv_cache_dtype: fp8 stores Gemma 4's full layers' KV in a real FP8 pool (arm "
                        "-kvfp8; --kv-dtype emulates one in a BF16 pool)")
    g.add_argument("--cache", action="store_true", help="the prefix cache on (Qwen: with checkpoint slots)")
    g.add_argument("--graphs", action=argparse.BooleanOptionalAction, default=True,
                   help="CUDA graphs (--no-graphs: every step eager, which taps need)")
    g.add_argument("--concurrency", type=int, default=1, help="requests in flight (a closed loop)")


def _options(args: argparse.Namespace, **kw: Any) -> Options:
    return Options(device=args.device, dtype=args.dtype, k=args.k, max_model_len=args.max_model_len,
                   budget=args.budget, max_num_seqs=args.max_num_seqs, page_size=args.page_size,
                   memory_gb=args.memory_gb, num_pages=args.num_pages, ckpt_slots=args.ckpt_slots,
                   row_tokens=args.row_tokens, precise=args.precise, pdl=args.pdl, sparse_head=args.sparse_head,
                   always_split=args.always_split, check=args.check, kv_cache_dtype=args.kv_cache_dtype, **kw)


def cmd_refs(args: argparse.Namespace) -> None:
    if args.max_model_len is None:
        args.max_model_len = MAX_LEN
    workloads = [w for w in args.workload.split(",") if w]
    indices = parse_ids(args.indices)
    model_dir = resolve(args.model)
    soft = args.soft_tokens if model_kind(model_dir) == "gemma" else None
    reqs, grammars = build_requests(model_dir, workloads, indices, soft_tokens=soft, input_tokens=args.input_tokens,
                                    seed=args.seed)
    refs = make_refs(args.model, reqs, assistant=args.assistant, grammars=grammars,
                     opts=_options(args, soft_tokens=soft), max_tokens=args.max_tokens, switch=args.arm,
                     cache=args.cache, graphs=args.graphs, concurrency=args.concurrency,
                     options={"workloads": workloads, "indices": indices, "input_tokens": args.input_tokens,
                              "seed": args.seed})
    out = args.out or results_dir("forced")
    out.mkdir(parents=True, exist_ok=True)
    (out / "refs.json").write_text(json.dumps(refs, indent=1, default=str))
    print(refs_table(refs), flush=True)
    log.info("wrote %s", out / "refs.json")


def cmd_run(args: argparse.Namespace) -> None:
    if args.max_model_len is None:  # the references'
        args.max_model_len = json.loads(args.refs.read_text()).get("settings", {}).get("max_model_len", MAX_LEN)
    if args.kv_scales is not None and args.kv_dtype is None:
        raise SystemExit("--kv-scales needs --kv-dtype")
    res = run_arm(args.refs, switch=args.arm, cache=args.cache, graphs=args.graphs, taps=args.taps,
                  core_layers=parse_ids(args.core_layers), concurrency=args.concurrency,
                  rids=parse_ids(args.rids) if args.rids else None, out=args.out,
                  opts=_options(args, draft_universe=False, draft_vocab=args.draft_vocab), model=args.model,
                  assistant=args.assistant, kv_dtype=args.kv_dtype,
                  kv_scales=load_kv_scales(args.kv_scales) if args.kv_scales else None)
    print(run_tables(res), flush=True)


def same(a: str | Path, b: str | Path) -> list[dict[str, Any]]:
    """Two arms' forced runs request by request (the module docstring, bit for bit): for each `r<rid>.pt` in either
    folder, whether it is in both (`missing`: the folder without it), whether the tokens, the positions with a hidden
    state (`hvalid`), the hidden states there and the acceptance at every verify step (its drafts accepted and their
    positions) are the same, and where the hidden states differ, the rows and the largest difference. `ok`: all of
    them, over at least one position."""
    import torch

    def runs(folder: str | Path) -> dict[int, Path]:
        return {int(m[1]): f for f in Path(folder).glob("r*.pt") if (m := re.fullmatch(r"r(\d+)\.pt", f.name))}

    fa, fb = runs(a), runs(b)
    out = []
    for rid in sorted(fa.keys() | fb.keys()):
        if rid not in fa or rid not in fb:
            out.append({"rid": rid, "missing": str(b if rid in fa else a), "ok": False})
            continue
        ca, cb = (torch.load(f, map_location="cpu", weights_only=True) for f in (fa[rid], fb[rid]))
        tokens = torch.equal(ca["tokens"], cb["tokens"])
        masks = tokens and torch.equal(ca["hvalid"], cb["hvalid"])
        v = ca["hvalid"] if masks else torch.zeros(0, dtype=torch.bool)
        ha, hb = (ca["hidden"][v], cb["hidden"][v]) if masks else (torch.zeros(0), torch.zeros(0))
        rows = (ha != hb).flatten(1).any(1) if len(ha) else torch.zeros(0, dtype=torch.bool)
        accept = torch.equal(ca["accept"], cb["accept"]) and torch.equal(ca["accept_pos"], cb["accept_pos"])
        hidden = masks and not bool(rows.any())
        out.append({"rid": rid, "missing": None, "tokens": tokens, "masks": masks, "positions": int(v.sum()),
                    "hidden": hidden, "rows": int(rows.sum()),
                    "max_diff": float((ha.float() - hb.float()).abs().max()) if bool(rows.any()) else 0.0,
                    "accept": accept, "ok": tokens and hidden and accept and int(v.sum()) > 0})
    return out


def cmd_same(args: argparse.Namespace) -> None:
    res = same(args.a, args.b)
    if not res:
        raise SystemExit(f"no request in {args.a} or {args.b}")
    for r in res:
        if r["missing"]:
            print(f"r{r['rid']}: not in {r['missing']}")
            continue
        hidden = ("same" if r["hidden"] else f"differ ({r['rows']} rows, max {r['max_diff']:.4g})" if r["masks"]
                  else "at other positions")
        print(f"r{r['rid']}: {r['positions']} positions, tokens {'same' if r['tokens'] else 'differ'}, hidden "
              f"{hidden}, accept {'same' if r['accept'] else 'differ'}")
    bad = [r["rid"] for r in res if not r["ok"]]
    print(f"{len(res) - len(bad)} of {len(res)} requests bit for bit the same" + (f"; not: {bad}" if bad else ""))
    if bad:
        raise SystemExit(1)


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python3 -m tests.scripts.numerics_forced",
                                 description="\n\n".join(__doc__.split("\n\n")[:2]), formatter_class=HelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("refs", help="greedy references under the real grammars (refs.json)",
                       formatter_class=HelpFormatter)
    p.add_argument("--model", required=True, help="a checkpoint folder or a repo id")
    p.add_argument("--workload", default="scene", help="scene, chat, recall, or several (comma-separated)")
    p.add_argument("--indices", default="0-3", help="the workloads' request indices, e.g. 0-7,14")
    p.add_argument("--max-tokens", type=int, default=MAX_TOKENS, help="a reference answer's tokens at most")
    p.add_argument("--soft-tokens", type=int, default=SOFT_TOKENS,
                   help="Gemma 4's images.max_soft_tokens (the checkpoint's own budget: 280 on gemma-4-31B-it)")
    p.add_argument("--input-tokens", type=int, default=CHAT_TOKENS, help="chat, recall: each prompt's tokens")
    p.add_argument("--seed", type=int, default=0, help="chat, recall: the prompts' seed")
    p.add_argument("--arm", default="on", choices=ARMS, help="dev.runner.cuda.batch_invariant on or off")
    p.add_argument("--out", type=Path, default=None,
                   help="the folder for refs.json (default: results/numerics/forced/<UTC time>)")
    _engine_args(p)
    p.set_defaults(fn=cmd_refs)
    p = sub.add_parser("run", help="a forced run of one arm (<out>/<arm>/r<rid>.pt, run.json)",
                       formatter_class=HelpFormatter)
    p.add_argument("--refs", type=Path, required=True, help="refs.json")
    p.add_argument("--arm", required=True, choices=ARMS, help="dev.runner.cuda.batch_invariant on or off")
    p.add_argument("--model", default=None, help="the references' checkpoint in another folder")
    p.add_argument("--taps", action="store_true", help="each layer's stream, mixer and MLP outputs (--no-graphs)")
    p.add_argument("--core-layers", default="",
                   help="model layers whose kernels' inputs and outputs to keep (--taps), e.g. 3,31")
    p.add_argument("--rids", default="", help="the references to force (default: all), e.g. 0-3")
    p.add_argument("--draft-vocab", type=int, default=0, help="the drafter's head (0: the whole vocabulary)")
    p.add_argument("--kv-dtype", default=None, choices=KV_DTYPES,
                   help="round the KV each step writes through float8_e4m3fn (the module docstring): fp8 the "
                        "full-attention layers', fp8-all Gemma 4's sliding layers' too (default: BF16 KV)")
    p.add_argument("--kv-scales", type=Path, default=None,
                   help="per-layer power-of-two KV scales (`load_kv_scales`: a JSON object, or a run.json whose "
                        "largest values set them; default: 1.0)")
    p.add_argument("--out", type=Path, default=None, help="the run's folder, <out>/<arm>/ (default: refs.json's)")
    _engine_args(p)
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("same", help="two arms' runs bit for bit (tokens, hidden states, acceptance)",
                       formatter_class=HelpFormatter)
    p.add_argument("a", type=Path, help="an arm's folder (<out>/<arm>)")
    p.add_argument("b", type=Path, help="another arm's folder")
    p.set_defaults(fn=cmd_same)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    args.fn(args)


if __name__ == "__main__":
    main()
