# Contributing

Thank you for your interest in llm-mini-infer.

- Open an issue first for anything large, so that we can agree on the change before you write it.
- Before a pull request, run the CPU suite with `uv run pytest -q`, and run `uv run ruff check .`. Both must pass.
- A pull request's title and body become its squash commit's subject and body: [the pull request template](.github/pull_request_template.md) says how to write them.
- The GPU suite is optional. It needs an SM120 host: see [The GPU suite](docs/development.md#the-gpu-suite).
- Contributions are under the Apache License 2.0 of the repository ([LICENSE](LICENSE)).
- Do not paste or translate code from repositories under source-available or copyleft licences. Re-derive an idea from its description, and credit it in the pull request.
- Report a security problem as [SECURITY.md](SECURITY.md) says, not in an issue.

[docs/development.md](docs/development.md) has the rest: the setup, the writing rules and the commit style.
