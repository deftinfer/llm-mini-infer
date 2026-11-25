"""Every docs heading, anchor and repository path that the text cites exists.

A citation of a heading reads `architecture.md, "Heading"` (or `docs/architecture.md "Heading"`), of an anchor
`operations.md#anchor` (GitHub's slug of a heading), of a path `docs/...`, `tests/...`, `llm_mini_infer/...`,
`bench/...`, `deploy/...` or `tools/...`. The check walks the tree with pathlib and reads text files, so it needs no git
and runs in the GPU tests' image (where llm_mini_infer/ is a symlink to the installed package) and in an export.

A document resolves the way the citing file means it: a relative Markdown link against the citing file's folder, a bare
name in the citing file's folder, else in docs/, else at the root; a name with a folder first from the root.

Not citations (the narrow rules): a path with a glob (`*`), a placeholder (`<...>`, `{...}`, `[...]`) or an ellipsis;
a path under a generated folder (tests/.fixtures/, tools/bin/, results/, bench/results/); a path in a regular
expression (followed by a backslash); a document in a placeholder folder (DIR/x.md); the files that invent names for a
tool under test (SYNTHETIC); a bare file name without a heading, anchor or
link that is not one of the project's documents (tests write throwaway .md files). A checkout whose .gitattributes keeps
paths out of `git archive` (export-ignore) holds files an export lacks: they are not read, and a citation of one fails
here as it would in the export.
"""

from __future__ import annotations

import bisect
import os
import re
from collections import defaultdict
from functools import cache
from pathlib import Path

from tests.support.paths import ROOT

SKIP_NAMES = {".git", ".venv", "__pycache__", ".ruff_cache", ".pytest_cache", ".mypy_cache", "node_modules"}


def export_ignored() -> set[str]:
    """The paths .gitattributes keeps out of `git archive`, relative to the root (none without a .gitattributes)."""
    f = ROOT / ".gitattributes"
    if not f.is_file():
        return set()
    return {parts[0].strip("/") for line in f.read_text().splitlines()
            if (parts := line.split()) and not parts[0].startswith("#") and "export-ignore" in parts[1:]}


IGNORED = export_ignored()
SKIP_DIRS = {"tests/.fixtures", "results", "tools/bin"} | {p for p in IGNORED if (ROOT / p).is_dir()}
SKIP_FILES = {p for p in IGNORED if not (ROOT / p).is_dir()} | {
    "tests/repo/test_doc_citations.py",  # this file describes the forms it checks
}
# Tests that invent file names to feed a tool (a changed-files list, a repro command): not citations.
SYNTHETIC = {"tests/repo/test_repro_harness.py"}
# Files whose bytes the image records: a wrong citation there is skipped by file:line, never edited.
SKIP_CITATIONS: set[tuple[str, int]] = set()
SUFFIXES = {".py", ".md", ".yaml", ".yml", ".toml", ".cu", ".cuh", ".txt"}
KNOWN_DOCS = {"architecture", "operations", "development", "README", "vllm-compatibility", "benchmarks"}
GENERATED = ("tests/.fixtures", "tools/bin", "results", "bench/results")

DOC = r"(?<![\w./~-])((?:[\w.-]+/)*[\w-]+\.md)(?![\w])"
HEADING_AFTER = re.compile(r"(?:'s)?(?:,\s*|\s+)\\?[\"“]([^\"”\\]+?)\\?[\"”]")
ANCHOR_AFTER = re.compile(r"#([\w-]+)")
LINK_BEFORE = re.compile(r"\]\($")
PATH = re.compile(r"(?<![\w./~-])((?:docs|tests|llm_mini_infer|bench|deploy|tools)/[\w./-]*)")
LEAD = re.compile(r"^[ \t]*(?:(?:#+|//|\*|>)[ \t]*)?")


def _skipped_dir(rel: str) -> bool:
    return any(rel == d or rel.startswith(d + "/") for d in SKIP_DIRS)


def tracked_texts() -> list[tuple[str, Path]]:
    """(logical path relative to the root, real path) of every text file to read."""
    found: list[tuple[str, Path]] = []

    def walk(real: Path, logical: str) -> None:
        for dirpath, dirnames, filenames in os.walk(real, followlinks=False):
            rel_dir = Path(dirpath).relative_to(real).as_posix()
            base = logical if rel_dir == "." else f"{logical}/{rel_dir}".lstrip("/")
            dirnames[:] = sorted(
                d
                for d in dirnames
                if d not in SKIP_NAMES
                and not d.endswith(".egg-info")
                and not (Path(dirpath) / d).is_symlink()
                and not _skipped_dir(f"{base}/{d}".lstrip("/"))
            )
            for name in sorted(filenames):
                path = Path(dirpath) / name
                rel = f"{base}/{name}".lstrip("/")
                if path.is_symlink() or rel in SKIP_FILES:
                    continue
                if path.suffix in SUFFIXES or "Dockerfile" in name:
                    found.append((rel, path))

    walk(ROOT, "")
    package = ROOT / "llm_mini_infer"
    if package.is_symlink() and package.exists():
        walk(package.resolve(), "llm_mini_infer")
    return found


def flatten(text: str) -> tuple[str, list[int], list[int]]:
    """The text with each line's comment lead removed and the lines joined by a space, so a citation that wraps across
    comment lines reads whole; the start offset of each line in it and the line numbers."""
    parts: list[str] = []
    starts: list[int] = []
    pos = 0
    for line in text.splitlines():
        body = LEAD.sub("", line).rstrip()
        starts.append(pos)
        parts.append(body)
        pos += len(body) + 1
    return " ".join(parts), starts, list(range(1, len(starts) + 1))


def slug(heading: str) -> str:
    """GitHub's anchor of a heading, before the -1, -2 suffixes of repeats."""
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", heading)
    text = text.replace("`", "").replace("*", "").strip().lower()
    return re.sub(r"[^\w\- ]", "", text).replace(" ", "-")


@cache
def headings_of(rel: str) -> tuple[frozenset[str], frozenset[str]]:
    """The heading texts and the anchors of a Markdown file (code fences skipped)."""
    texts: set[str] = set()
    anchors: set[str] = set()
    seen: defaultdict[str, int] = defaultdict(int)
    fence = False
    for line in (ROOT / rel).read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith(("```", "~~~")):
            fence = not fence
            continue
        m = None if fence else re.match(r"^#{1,6}\s+(.*?)\s*#*\s*$", line)
        if not m:
            continue
        text = m.group(1)
        texts.add(text)
        texts.add(text.replace("`", ""))
        base = slug(text)
        anchors.add(base if not seen[base] else f"{base}-{seen[base]}")
        seen[base] += 1
    return frozenset(texts), frozenset(anchors)


def resolve_doc(cite: str, citing: str) -> str | None:
    """The root-relative path of the cited document, or None when no candidate exists."""
    folder = Path(citing).parent
    if cite.startswith(("./", "../")):
        candidates = [folder / cite]
    elif "/" in cite:
        candidates = [Path(cite), folder / cite]
    else:
        candidates = [folder / cite, Path("docs") / cite, Path(cite)]
    for c in candidates:
        norm = Path(os.path.normpath(c))
        if not norm.parts or norm.parts[0] == "..":
            continue
        if (ROOT / norm).is_file():
            return norm.as_posix()
    return None


def collect() -> tuple[list[tuple], list[tuple]]:
    """(doc citations, path citations); each is (citing file, line, ...) ready to check."""
    docs: list[tuple] = []
    paths: list[tuple] = []
    for rel, real in tracked_texts():
        if rel in SYNTHETIC:
            continue
        try:
            text = real.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        flat, starts, _ = flatten(text)
        for m in re.finditer(DOC, flat):
            cite = m.group(1)
            line = bisect.bisect_right(starts, m.start()) or 1
            if (rel, line) in SKIP_CITATIONS:
                continue
            after = flat[m.end() :]
            is_link = bool(LINK_BEFORE.search(flat[max(0, m.start() - 2) : m.start()]))
            heading = HEADING_AFTER.match(after)
            anchor = ANCHOR_AFTER.match(after)
            if cite.split("/")[0].isupper() and "/" in cite:
                continue
            known = Path(cite).stem in KNOWN_DOCS or "/" in cite
            if not (heading or anchor or is_link or known):
                continue
            target = resolve_doc(cite, rel)
            if target == "tests/repo/test_doc_citations.py":
                continue
            if target in IGNORED or (target and any(target.startswith(p + "/") for p in IGNORED)):
                target = None  # not in an export
            docs.append(
                (
                    rel,
                    line,
                    cite,
                    target,
                    heading.group(1).strip() if heading else None,
                    anchor.group(1) if anchor else None,
                )
            )
        for number, line_text in enumerate(text.splitlines(), 1):
            if (rel, number) in SKIP_CITATIONS:
                continue
            for m in PATH.finditer(line_text):
                path = m.group(1).rstrip(".,;:")
                nxt = line_text[m.end() : m.end() + 1]
                if (
                    nxt in ("*", "<", "{", "[", "$", "%", "\\")
                    or "..." in path
                    or "…" in line_text[m.start() : m.end() + 2]
                ):
                    continue
                if path.endswith("/") is False and path in ("docs", "tests", "tools", "deploy", "bench"):
                    continue
                if any(path == g or path.startswith(g + "/") for g in GENERATED):
                    continue
                paths.append((rel, number, path))
    return docs, paths


def test_cited_headings_and_anchors_exist():
    docs, _ = collect()
    problems = []
    for rel, line, cite, target, heading, anchor in docs:
        where = f"{rel}:{line}"
        if target is None:
            problems.append(f"{where}: {cite} does not exist")
            continue
        if not target.endswith(".md"):
            continue
        texts, anchors = headings_of(target)
        if heading is not None and heading not in texts:
            problems.append(f'{where}: {target} has no heading "{heading}"')
        if anchor is not None and anchor not in anchors:
            problems.append(f"{where}: {target} has no anchor #{anchor}")
    assert not problems, "\n".join(problems)


def test_cited_paths_exist():
    _, paths = collect()
    problems = [
        f"{rel}:{line}: {path} does not exist"
        for rel, line, path in paths
        if (not (ROOT / path).exists() and not (ROOT / path.rstrip("/")).exists())
        or any(path.rstrip("/") == p or path.startswith(p + "/") for p in IGNORED)
    ]
    assert not problems, "\n".join(problems)


def test_the_check_reads_enough():
    docs, paths = collect()
    headings = {(t, h) for _, _, _, t, h, _ in docs if h}
    anchors = {(t, a) for _, _, _, t, _, a in docs if a}
    print(
        f"{len(docs)} doc citations ({len(headings)} distinct headings, {len(anchors)} anchors),"
        f" {len(paths)} path citations ({len({p for _, _, p in paths})} distinct)"
    )
    assert len(docs) >= 150 and len(headings) >= 40
    assert len(paths) >= 50
