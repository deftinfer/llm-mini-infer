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
    s.add_argument(
        "--config",
        help="path to the YAML configuration file (default: application configuration)",
    )
    s.add_argument("--host", help="server bind address (default: config server.host)")
    s.add_argument("--port", type=int, help="TCP port for the server (default: config server.port)")
    s.add_argument("--frontends", type=int,
                   help="number of frontend processes (default: 1; overrides server.num_frontends)")
    s.add_argument(
        "--served-model-name",
        help="model name exposed by the API (default: config model.served_model_name)",
    )
    s.add_argument(
        "--max-model-len",
        type=int,
        help="maximum model context length in tokens (default: config model.max_model_len)",
    )
    s.add_argument(
        "--max-num-seqs",
        type=int,
        help="maximum concurrent sequences (default: config scheduler.max_num_seqs)",
    )
    s.add_argument(
        "--max-num-batched-tokens",
        type=int,
        help="maximum batched tokens per scheduler iteration (default: config scheduler.max_num_batched_tokens)",
    )
    s.add_argument(
        "--max-num-queued-reqs",
        type=int,
        help=(
            "maximum requests in flight before returning HTTP 503 "
            "(default: unlimited; overrides scheduler.max_num_queued_reqs)"
        ),
    )
    s.add_argument(
        "--max-num-queued-tokens",
        type=int,
        help=(
            "maximum queued prompt tokens before returning HTTP 503 "
            "(default: unlimited; overrides scheduler.max_num_queued_tokens)"
        ),
    )
    s.add_argument("--page-size", type=int, help="KV-cache page size in tokens (default: config dev.cache.page_size)")
    s.add_argument("--step-ms", type=float,
                   help="fake runner device time per step in milliseconds (overrides dev.runner.fake.step_ms)")
    s.add_argument("--log-level", help="logging level (default: config observability.log_level)")
    s.add_argument("--set", action="append", default=[], metavar="GROUP.KEY=VALUE",
                   help="override any setting, e.g. --set scheduler.policy=priority "
                        "(repeatable; takes precedence over individual CLI flags)")
    return s


def overrides(args: argparse.Namespace) -> dict[str, Any]:
    """The settings that `serve`'s flags give, as `config.load` takes them: the named flags, then every `--set`, which
    wins. Raises ValueError for a malformed `--set`."""
    flags = {"model.path": args.model, "runner.kind": args.runner, "server.host": args.host,
             "server.port": args.port, "server.num_frontends": args.frontends,
             "model.served_model_name": args.served_model_name, "model.max_model_len": args.max_model_len,
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
