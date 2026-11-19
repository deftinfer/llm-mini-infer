"""The container's entrypoint (deploy/Dockerfile; docs/operations.md, "The image").

    docker compose -f deploy/docker-compose.yaml up -d                       # serve: the image's default command
    docker compose -f deploy/docker-compose.yaml run --rm llm-mini-infer hostcheck
        # the host check (llm_mini_infer.hostcheck)
    docker compose -f deploy/docker-compose.yaml run --rm llm-mini-infer python3 -m llm_mini_infer.kernels.build --help
        # any other

**serve** (default) reads `$LLM_MINI_INFER_CONFIG` (default `/etc/llm-mini-infer/config.yaml`) with the extra arguments,
which are `llm-mini-infer serve`'s (`--model`, `--set group.key=value`, ...). It checks the container first: the model
directory is mounted, or a Hugging Face repo id (the model's or its drafter's) downloads into a writable cache, and
`/dev/shm` holds the pixel arena, once a killed server's files are removed (`leftovers`). A failed check stops it
with a message that names the setting to fix; an unwritable compile cache only logs a warning. Then it runs the
launcher in this process, which docker's init (`init: true`) or an orchestrator's signals: SIGTERM drains
(`launcher`), and before the drain starts (a download) it exits at once.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

log = logging.getLogger("llm_mini_infer.entrypoint")
CONFIG = "/etc/llm-mini-infer/config.yaml"
# The JIT caches that persist on the cache volume (the image prebuilds our kernels: LLM_MINI_INFER_KERNEL_CACHE is never
# written there)
CACHES = ("FLASHINFER_WORKSPACE_BASE", "CUTE_DSL_CACHE_DIR", "CUDA_CACHE_PATH")


def load(config: str, extra: list[str]) -> Any:
    """The configuration that `llm-mini-infer serve --config <config> <extra>` builds: the extra arguments parsed as
    `llm_mini_infer.cli` parses them (it reports the ones it does not know). Raises OSError or ValueError; a malformed
    argument exits through argparse with status 2, as the CLI does."""
    import argparse

    from . import config as cfgmod
    from .cli import overrides, serve_args

    args, _ = serve_args(argparse.ArgumentParser(add_help=False)).parse_known_args(extra)
    return cfgmod.load(args.config or config, overrides(args))


def preflight(config: str, extra: list[str]) -> list[str]:
    """Problems that would stop the server later with a less helpful message (empty: none)."""
    from . import config as cfgmod
    from . import hub

    problems = []
    try:
        cfg = load(config, extra)
    except (OSError, ValueError) as exc:
        return [f"configuration {config}: {exc}"]
    # A repo id, the model's or the drafter's: fetched now, and the launcher, in this process, reuses the snapshots
    if hub.is_repo_id(cfg.model.path) or hub.is_repo_id(cfg.speculative.model or ""):
        cache = hub.cache_dir()
        try:
            cache.mkdir(parents=True, exist_ok=True)
            writable = os.access(cache, os.W_OK)
        except OSError:
            writable = False
        if not writable:
            return [f"the Hugging Face cache {cache} is not writable: mount a volume there (HF_HOME)"]
        try:
            hub.resolve(cfg)
        except hub.HubError as exc:
            return [str(exc)]
    model = Path(cfg.model.path or "")
    if not (model / "config.json").is_file():
        return [f"model.path {model} has no config.json: mount the checkpoint there (compose: "
                "LLM_MINI_INFER_MODEL_DIR), or give a Hugging Face repo id"]
    try:
        cfgmod.derive(cfg)
    except ValueError as exc:
        return [f"configuration: {exc}"]
    if cfg.images.enabled and os.path.isdir("/dev/shm"):
        st = os.statvfs("/dev/shm")
        free, need = st.f_bavail * st.f_frsize, (cfg.dev.images.arena_mb or 0) << 20
        if free < need:
            problems.append(f"/dev/shm has {free >> 20} MiB free, the pixel arena needs {need >> 20} MiB "
                            "(dev.images.arena_mb): raise /dev/shm's size (compose: shm_size; docker run: "
                            "--shm-size; the chart: shm.sizeLimit)")
    for var in CACHES:
        path = os.environ.get(var)
        if path:
            try:
                Path(path).mkdir(parents=True, exist_ok=True)
                ok = os.access(path, os.W_OK)
            except OSError:
                ok = False
            if not ok:
                log.warning("%s=%s is not writable: its cache does not persist across starts (mount the cache "
                            "volume)", var, path)
    return problems


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    cmd = args.pop(0) if args and not args[0].startswith("-") else "serve"
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s.%(msecs)03d [entrypoint] %(message)s",
                        datefmt="%m-%d %H:%M:%S")  # `logutil.setup_logging`'s format (the launcher's, from the config)
    for noisy in ("httpx2", "httpcore2"):  # a line per request of a Hugging Face download
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if cmd == "serve":
        config = os.environ.get("LLM_MINI_INFER_CONFIG", CONFIG)
        log.info("llm-mini-infer %s, configuration %s", os.environ.get("LLM_MINI_INFER_BUILD", "(no build id)"), config)
        from . import leftovers
        from .launcher import exit_on_stop

        exit_on_stop()  # the checks may download
        leftovers.sweep()  # a killed server's arena, which /dev/shm's check would count (the launcher's finds none)
        problems = preflight(config, args)
        for p in problems:
            log.error("%s", p)
        if problems:
            sys.exit(2)  # a failed check: the CLI's code for a bad setting (a failure while serving exits 1)
        from .cli import main as cli

        cli(["serve", "--config", config, *args])
    elif cmd == "hostcheck":
        from .hostcheck import main as hostcheck

        sys.exit(hostcheck(args))
    else:
        os.execvp(cmd, [cmd, *args])


if __name__ == "__main__":
    main()
