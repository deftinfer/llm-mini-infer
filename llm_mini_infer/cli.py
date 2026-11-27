"""`llm-mini-infer serve --model <dir or repo id> [--runner cuda|fake] [--config x.yaml] [--set group.key=value ...]`;
`llm-mini-infer hostcheck` (llm_mini_infer.hostcheck)."""

from __future__ import annotations

import argparse
import sys
from typing import Any

from . import config


def serve_args(s: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """`serve`'s flags, added to `s` (the container's entrypoint parses them the same way for its checks)."""
    s.add_argument("--model", help="model directory (config.json, tokenizer.json, chat template) or a Hugging Face "
                   "repo id")
    s.add_argument("--runner", choices=["fake", "cuda"],
                   help="cuda (the default) or fake (random tokens: tests and benchmarks)")
    s.add_argument("--config", help="YAML configuration file")
    s.add_argument("--host")
    s.add_argument("--port", type=int)
    s.add_argument("--frontends", type=int, help="frontend processes (default 1)")
    s.add_argument("--served-model-name", nargs="+", metavar="NAME",
                   help="one or more aliases; completions and metrics use the first")
    s.add_argument("--max-model-len", type=int)
    s.add_argument("--max-num-seqs", type=int)
    s.add_argument("--max-num-batched-tokens", type=int)
    s.add_argument("--max-num-queued-reqs", type=int, help="503 past this many requests in flight (default: no limit)")
    s.add_argument("--max-num-queued-tokens", type=int,
                   help="503 once the prompts not yet given a token hold this many tokens (default: no limit)")
    s.add_argument("--page-size", type=int)
    s.add_argument("--step-ms", type=float, help="fake runner: device time per step")
    s.add_argument("--log-level")
    s.add_argument("--set", action="append", default=[], metavar="GROUP.KEY=VALUE",
                   help="override any setting, e.g. --set scheduler.policy=priority (repeatable)")
    return s


def overrides(args: argparse.Namespace) -> dict[str, Any]:
    """The settings that `serve`'s flags give, as `config.load` takes them: the named flags, then every `--set`, which
    wins. Raises ValueError for a malformed `--set`."""
    names = args.served_model_name
    served_name = names[0] if names and len(names) == 1 else names
    flags = {"model.path": args.model, "runner.kind": args.runner, "server.host": args.host,
             "server.port": args.port, "server.num_frontends": args.frontends,
             "model.served_model_name": served_name, "model.max_model_len": args.max_model_len,
             "scheduler.max_num_seqs": args.max_num_seqs, "scheduler.max_num_batched_tokens":
             args.max_num_batched_tokens, "scheduler.max_num_queued_reqs": args.max_num_queued_reqs,
             "scheduler.max_num_queued_tokens": args.max_num_queued_tokens, "dev.cache.page_size": args.page_size,
             "dev.runner.fake.step_ms": args.step_ms, "observability.log_level": args.log_level}
    out = {k: v for k, v in flags.items() if v is not None}
    out.update(config.parse_override(o) for o in args.set)
    return out


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else list(argv)
    if args[:1] == ["hostcheck"]:  # its own flags
        from .hostcheck import main as hostcheck

        sys.exit(hostcheck(args[1:]))
    parser = argparse.ArgumentParser(prog="llm-mini-infer")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("hostcheck", help="check this machine (the GPU and host, host speed, startup and a load) against "
                                     "reference numbers (llm-mini-infer hostcheck --help)")
    serve_args(sub.add_parser("serve", help="run the OpenAI-compatible server"))
    args = parser.parse_args(args)
    try:
        cfg = config.load(args.config, overrides(args))
        if not cfg.model.path:
            parser.error("--model (or model.path in the config file) is required")
    except ValueError as exc:
        parser.error(str(exc))
    from .launcher import serve

    sys.exit(serve(cfg))


if __name__ == "__main__":
    main()
