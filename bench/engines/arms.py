"""The arms of the engine comparison (bench/engines/drive.py): each engine's server command and environment, the one
place that holds them.

Three engines serve each model in turn on one GPU, on the loopback interface, without an API key:
- `ours`: llm-mini-infer, from the GPU tests' image of the measured commit (its venv, its configuration file: the
  guidance backend with compact JSON, seed 0);
- `vllm`: vLLM v0.33.0, from its CUDA 13 release wheel in a venv of its own;
- `sglang`: SGLang v0.5.23, from its CUDA 13 wheels in a venv of its own.

**Held equal** on every arm: the context (`CONTEXT`: llm-mini-infer's batch invariance is on by its own default at
this length), the sequences in flight (`SEQS`; SGLang on Qwen3.8 `SGLANG_QWEN_SEQS`, a named deviation), the KV cache
in the model's dtype, each model's default image budget, prefix caching on, structured output with llguidance and
compact JSON, the same drafter and draft count (`k`), seed 0, no reasoning parser on vLLM or SGLang (every request
turns thinking off). **Not held equal**, recorded per arm: the
memory budget (vLLM 0.9, llm-mini-infer 0.9 by its default, SGLang its computed static fraction), the per-step token
budget, the FP8 GEMM kernel and the attention backend: each engine's default, unless an arm names an override.

**The environment** of each arm is clean: a short list of the container's own variables (`INHERIT`), the engine's
interpreter first in PATH, its own cache roots under `CACHE_ROOT/<engine>` (`cache_roots`: a cold start wipes exactly
these, and SGLang's `~/.cache/sglang`, where its JIT kernels build whatever `SGLANG_CACHE_DIR` says), the Hugging Face
cache offline, and the variables an arm sets (`Arm.env_set`). A manifest records the variables' names, never their
values.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

CONTEXT = 8184  # the context of every arm; llm-mini-infer's batch invariance needs at most 8,184 positions
SEQS = 64  # sequences in flight
HOST, PORT = "127.0.0.1", 8000
ENGINES = ("ours", "vllm", "sglang")
CACHE_ROOT = "/cache"
HF_HOME = "/root/hf"
# The engines' commands on the GPU tests' image with the two venvs beside its own (`--engine-bin` replaces one)
BIN = {"ours": "/opt/venv/bin/llm-mini-infer", "vllm": "/opt/vllm/bin/vllm", "sglang": "/opt/sglang/bin/sglang"}
SYSTEM_PATH = "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
# The container's variables every arm keeps (the GPU's visibility and the driver's libraries, the locale, HOME)
INHERIT = ("HOME", "LANG", "LC_ALL", "TERM", "LD_LIBRARY_PATH", "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES",
           "CUDA_VISIBLE_DEVICES")
# llm-mini-infer's image settings the release itself carries (deploy/Dockerfile): its configuration file, its
# prebuilt kernels and its build id
OURS_INHERIT = ("LLM_MINI_INFER_CONFIG", "LLM_MINI_INFER_KERNEL_PREBUILT", "LLM_MINI_INFER_BUILD")
# The image's configuration file (deploy/Dockerfile's LLM_MINI_INFER_CONFIG), which its entrypoint passes as --config:
# `llm-mini-infer serve` itself reads no variable, so the arm's command names the file
OURS_CONFIG = "/etc/llm-mini-infer/config.yaml"
# SGLang on Qwen3.8: its mamba state pool holds 5 fp32 slots a running request (3 + 2 tracking slots), ~49 GB at 64
# against ~41 GB left after the target and its MTP copy at its computed memory fraction; 48 is SGLang's own cap on
# running requests under speculative decoding
SGLANG_QWEN_SEQS = 48
SGLANG_QWEN_DEVIATION = (f"--max-running-requests {SGLANG_QWEN_SEQS} (SGLang's own value under speculative decoding), "
                         f"not {SEQS}: at {SEQS} its fp32 mamba state (5 slots a request, ~49 GB) does not fit the "
                         "~41 GB left after the weights; a load at a higher concurrency queues in the server")
# vLLM's batch invariance with DeepGEMM off: its FP8 GEMM choice moves to the next block-FP8 kernel (the log's
# "Selected" line names it); the arm beside the one that keeps DeepGEMM, vLLM's default kernel
VLLM_NO_DEEP_GEMM = (("VLLM_USE_DEEP_GEMM", "0"),)
VLLM_NO_DEEP_GEMM_DEVIATION = ("VLLM_USE_DEEP_GEMM=0: DeepGEMM off, so vLLM selects its next block-FP8 kernel (the "
                               "arm with DeepGEMM, vLLM's default, runs beside it)")


@dataclass(frozen=True)
class Model:
    key: str
    id: str  # the Hugging Face repo id, served under that name by every engine
    drafter: str | None  # the assistant model's repo id; None: the checkpoint's own MTP layer
    k: int  # the headline draft count: llm-mini-infer's default, and SGLang's recipe and default
    k_source: str


MODELS = {
    "gemma": Model("gemma", "RedHatAI/gemma-4-31B-it-FP8-block", "google/gemma-4-31B-it-assistant", 5,
                   "llm-mini-infer's default; SGLang's recipe and CI value (5 steps, top-k 1, 6 draft tokens)"),
    "qwen": Model("qwen", "Qwen/Qwen3.8-27B-FP8", None, 3,
                  "llm-mini-infer's default; SGLang's recipe and its EAGLE default (3 steps, top-k 1, 4 draft tokens)"),
}
# vLLM's own settings (the second table), each with its source
VLLM_OWN_K = {
    "gemma": ((1, "vLLM v0.33.0's code default: the assistant's n_predict"),
              (4, "vLLM v0.33.0's recipe, recipes/models/Google/gemma-4-31B-it.yaml")),
    "qwen": ((1, "the value vLLM v0.33.0's docs start at (an MTP layer needs an explicit k)"),),
}


@dataclass(frozen=True)
class Arm:
    """One server: an engine, a model, its draft count and batch invariance (None: the engine's default), and what
    else it changes. `name` is the unit's folder; every change from the engine's default arm is a named deviation."""
    engine: str
    model: str  # a key of MODELS
    k: int | None = None  # None: the model's headline k
    batch_invariant: bool | None = None  # None: the engine's default (llm-mini-infer: on at CONTEXT; the others: off)
    sets: tuple[str, ...] = ()  # llm-mini-infer: more --set values (the multi-turn cell's cache settings)
    seqs: int | None = None  # the sequences in flight; None: SEQS (a named deviation)
    env_set: tuple[tuple[str, str], ...] = ()  # variables set in the arm's environment (a named variant)
    attention: str | None = None  # an attention backend forced on top of the arm's own (a named fallback)
    drop: tuple[str, ...] = ()  # flags left out of the arm's command (a named deviation)
    extra: tuple[str, ...] = ()  # arguments added at the end of the arm's command (a named deviation)
    label: str = ""  # a suffix of the unit's folder for a variant the fields above don't name
    deviations: tuple[str, ...] = field(default=())  # why each override or drop was made, for the manifest

    @property
    def max_seqs(self) -> int:
        return self.seqs if self.seqs is not None else SEQS

    @property
    def draft_k(self) -> int:
        return self.k if self.k is not None else MODELS[self.model].k

    @property
    def name(self) -> str:
        parts = [self.engine]
        if self.k is not None and self.k != MODELS[self.model].k:
            parts.append(f"k{self.k}")
        if self.batch_invariant is not None and self.batch_invariant != default_batch_invariant(self.engine):
            parts.append("bi" if self.batch_invariant else "nobi")
        if self.label:
            parts.append(self.label)
        return "-".join(parts)


def default_batch_invariant(engine: str) -> bool:
    """Each engine's own mode at CONTEXT: llm-mini-infer's batch invariance is on by default at a context of at most
    8,184 positions on these block-FP8 checkpoints; vLLM's and SGLang's are off unless asked for."""
    return engine == "ours"


def batch_invariant(arm: Arm) -> bool:
    return default_batch_invariant(arm.engine) if arm.batch_invariant is None else arm.batch_invariant


def argv(arm: Arm, binary: str | None = None, config: str | None = None) -> list[str]:
    """The arm's server command (`binary`: the engine's command, default `BIN`; `config`: llm-mini-infer's
    configuration file, default `$LLM_MINI_INFER_CONFIG`, else `OURS_CONFIG`)."""
    m = MODELS[arm.model]
    exe = binary or BIN[arm.engine]
    k = arm.draft_k
    bi = batch_invariant(arm)
    seqs = arm.max_seqs
    if arm.engine == "ours":
        config = config or os.environ.get("LLM_MINI_INFER_CONFIG") or OURS_CONFIG
        cmd = [exe, "serve", "--config", config, "--set", f"model.path={m.id}"]
        if m.drafter:
            cmd += ["--set", f"speculative.model={m.drafter}"]
        cmd += ["--set", f"model.max_model_len={CONTEXT}", "--set", f"scheduler.max_num_seqs={seqs}",
                "--set", f"server.host={HOST}", "--set", f"server.port={PORT}",
                "--set", f"speculative.num_speculative_tokens={k}",
                "--set", "server.stop_with_parent=true"]  # it stops when the driver exits without stopping it
        if arm.batch_invariant is not None:
            cmd += ["--set", f"dev.runner.cuda.batch_invariant={'true' if bi else 'false'}"]
        for s in arm.sets:
            cmd += ["--set", s]
    elif arm.engine == "vllm":
        spec: dict = {"method": "mtp", "num_speculative_tokens": k}
        if m.drafter:
            spec = {"method": "mtp", "model": m.drafter, "num_speculative_tokens": k}
        cmd = [exe, "serve", m.id, "--host", HOST, "--port", str(PORT), "--max-model-len", str(CONTEXT),
               "--max-num-seqs", str(seqs), "--gpu-memory-utilization", "0.9", "--enable-prefix-caching",
               "--limit-mm-per-prompt", json.dumps({"image": 16}, separators=(",", ":")), "--seed", "0",
               "--enable-prompt-tokens-details",
               "--structured-outputs-config",
               json.dumps({"backend": "guidance", "disable_any_whitespace": True}, separators=(",", ":")),
               "--speculative-config", json.dumps(spec, separators=(",", ":"))]
    elif arm.engine == "sglang":
        cmd = [exe, "serve", "--model-path", m.id, "--host", HOST, "--port", str(PORT),
               "--max-running-requests", str(seqs), "--context-length", str(CONTEXT), "--enable-metrics",
               "--enable-cache-report", "--random-seed", "0",
               "--grammar-backend", "llguidance", "--constrained-json-disable-any-whitespace"]
        if arm.model == "gemma":
            # Gemma 4 takes triton on SM120 (its only practical backend there); the assistant becomes FROZEN_KV_MTP,
            # which has no default counts: steps k, top-k 1, k + 1 draft tokens
            cmd += ["--attention-backend", "triton", "--speculative-algorithm", "NEXTN",
                    "--speculative-draft-model-path", m.drafter or "", "--speculative-eagle-topk", "1",
                    "--speculative-num-steps", str(k), "--speculative-num-draft-tokens", str(k + 1),
                    "--mm-process-config", json.dumps({"image": {"max_soft_tokens": 280}}, separators=(",", ":"))]
        else:
            # Qwen3.8: flashinfer on SM120 (triton with batch invariance on, which keeps the radix cache), the
            # checkpoint's MTP layer through EAGLE, the mamba pool sized so the arm's sequences fit (5 slots each),
            # and SGLang's documented RTX PRO 6000 flag for MTP on the linear layers
            cmd += ["--trust-remote-code", "--attention-backend", "triton" if bi else "flashinfer",
                    "--speculative-algorithm", "EAGLE", "--speculative-eagle-topk", "1",
                    "--speculative-num-steps", str(k), "--speculative-num-draft-tokens", str(k + 1),
                    "--max-mamba-cache-size", str(seqs * 5), "--enable-linear-replayssm-spec"]
        if bi:
            cmd += ["--enable-deterministic-inference"]
    else:
        raise ValueError(f"unknown engine {arm.engine!r}")
    cmd += list(arm.extra)
    for flag in arm.drop:
        if flag not in cmd:
            raise ValueError(f"{arm.name}: drop {flag!r}, which its command does not hold")
        cmd.remove(flag)
    if arm.attention:
        if "--attention-backend" in cmd:
            cmd[cmd.index("--attention-backend") + 1] = arm.attention
        else:
            cmd += ["--attention-backend", arm.attention]
    return cmd


def cache_roots(engine: str) -> list[str]:
    """The engine's own cache directories: everything it compiles or writes at a start, and all a cold start wipes.
    The first is the one the arm's variables point at. SGLang v0.5.23 builds its JIT kernels into ~/.cache/sglang/jit
    whatever SGLANG_CACHE_DIR says, so its home cache is one of its roots."""
    roots = [f"{CACHE_ROOT}/{engine}"]
    if engine == "sglang":
        roots.append("~/.cache/sglang")
    return roots


def env(arm: Arm, binary: str | None = None, base: dict[str, str] | None = None) -> dict[str, str]:
    """The arm's environment: `INHERIT` from `base` (default: this process's), the engine's interpreter first in PATH,
    its cache roots, the Hugging Face cache offline, and the engine's switches."""
    base = dict(os.environ) if base is None else base
    exe = binary or BIN[arm.engine]
    root = cache_roots(arm.engine)[0]
    out = {k: base[k] for k in INHERIT if k in base}
    out.update(PATH=f"{os.path.dirname(exe)}:{SYSTEM_PATH}", PYTHONUNBUFFERED="1", CUDA_HOME="/usr/local/cuda",
               HF_HOME=HF_HOME, HF_HUB_OFFLINE="1", FLASHINFER_WORKSPACE_BASE=root, CUDA_CACHE_PATH=f"{root}/nv",
               TRITON_CACHE_DIR=f"{root}/triton", TORCHINDUCTOR_CACHE_DIR=f"{root}/inductor")
    out.setdefault("HOME", "/root")
    if arm.engine == "ours":
        out.update({k: base[k] for k in OURS_INHERIT if k in base})
        out.update(VIRTUAL_ENV=os.path.dirname(os.path.dirname(exe)), CUTE_DSL_CACHE_DIR=f"{root}/cute-dsl",
                   TOKENIZERS_PARALLELISM="false", FLASHINFER_DISABLE_JIT="1")  # the release image's setting
    elif arm.engine == "vllm":
        out["VLLM_CACHE_ROOT"] = root  # its torch.compile cache and DeepGEMM's JIT inside
        if batch_invariant(arm):
            out["VLLM_BATCH_INVARIANT"] = "1"
    elif arm.engine == "sglang":
        out["SGLANG_CACHE_DIR"] = root
    out.update(dict(arm.env_set))
    return out


def arms(model: str) -> dict[str, list[Arm]]:
    """The arms of each family for one model (bench/engines/drive.py runs them)."""
    def arm(engine: str, **kw) -> Arm:
        if model == "qwen" and engine == "sglang":  # its fp32 mamba state does not fit 64 sequences
            kw.update(seqs=SGLANG_QWEN_SEQS, deviations=(*kw.get("deviations", ()), SGLANG_QWEN_DEVIATION))
        return Arm(engine, model, **kw)

    head = [arm(e) for e in ENGINES]
    other_mode = [arm(e, batch_invariant=not default_batch_invariant(e)) for e in ENGINES]
    own = [Arm("vllm", model, k=k) for k, _ in VLLM_OWN_K[model]]
    # The cost of batch invariance (family I): each engine in both modes, on fresh servers; vLLM's invariance on Gemma
    # 4 with DeepGEMM off (vLLM refuses it on Qwen3.8's GDN attention)
    invariant = other_mode
    if model == "gemma":
        nodg = Arm("vllm", model, batch_invariant=True, env_set=VLLM_NO_DEEP_GEMM, label="nodg",
                   deviations=(VLLM_NO_DEEP_GEMM_DEVIATION,))
        invariant = [nodg if a.engine == "vllm" else a for a in other_mode]
        other_mode = [*other_mode, nodg]
    out = {"smoke": head + other_mode, "A1": head, "A2": head, "C": head + other_mode, "D": head, "S": head,
           "B": own, "I": [a for pair in zip(head, invariant, strict=True) for a in pair]}
    if model == "qwen":
        # The multi-turn cell: llm-mini-infer with superseded checkpoints demoted and not, at checkpoint slots below
        # the conversations in flight; vLLM's and SGLang's own caches (capacity not matched)
        slots = "dev.cache.checkpoint_slots=24"
        out["M"] = [Arm("ours", model, sets=("dev.cache.demote_superseded=true", slots), label="demote"),
                    Arm("ours", model, sets=("dev.cache.demote_superseded=false", slots), label="nodemote"),
                    Arm("vllm", model), arm("sglang")]
    return out


def k_source(arm: Arm) -> str:
    if arm.engine == "vllm" and arm.k is not None:
        return dict(VLLM_OWN_K[arm.model]).get(arm.k, "named by the arm")
    if arm.engine == "vllm":
        return "the same k as the other engines (vLLM's own: bench/engines/arms.py, VLLM_OWN_K)"
    return MODELS[arm.model].k_source
