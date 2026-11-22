"""Every site that loads the image family sizes images alike: the frontend, the engine, the runner's tower and the
image pool all pass the derived images.max_soft_tokens, so none can fall back to the checkpoint's budget (Gemma 4:
processor_config.json's 280 under a setting of 70). The pixel bounds' check catches a site that forgets it at
startup, except the runner's, which passes no bounds: its tower buckets are checked here, and so are the launcher's
keyword arguments to the pool's workers (the budget and images.max_source_pixels), without the Gemma fixture."""

from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

from ..support.paths import DATA, ROOT

PACKAGE = ROOT / "llm_mini_infer"
FAMILY = "llm_mini_infer.vision.family"
# The load sites: a new one is added here deliberately, with its images.max_soft_tokens
SITES = {"llm_mini_infer/config.py": 1, "llm_mini_infer/api/app.py": 1, "llm_mini_infer/engine/process.py": 1,
         "llm_mini_infer/runner/cuda/images.py": 1, "llm_mini_infer/vision/pool.py": 1}


def family_loads(path: Path, root: Path = ROOT) -> list[ast.Call]:
    """The calls of `vision.family.load` in a module under `root`, however it is imported (`from ..vision import
    family as vf`, `from . import family as fam`, `import llm_mini_infer.vision.family`, `from .family import load`)."""
    tree = ast.parse(path.read_text())
    parts = path.relative_to(root).with_suffix("").parts
    package = ".".join(parts[:-1])  # what a relative import starts from (an __init__.py's own package)
    modules: set[str] = set()  # names bound to the module
    functions: set[str] = set()  # names bound to its load()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:  # relative: the module's package, then one level up per further dot
                up = package.split(".")
                up = up[:len(up) - node.level + 1]
                base = ".".join([*up, base] if base else up)
            for alias in node.names:
                if f"{base}.{alias.name}" == FAMILY:
                    modules.add(alias.asname or alias.name)
                elif base == FAMILY and alias.name == "load":
                    functions.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == FAMILY:
                    modules.add(alias.asname or alias.name)
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        on_module = isinstance(f, ast.Attribute) and f.attr == "load" and ast.unparse(f.value) in modules
        if on_module or (isinstance(f, ast.Name) and f.id in functions):
            calls.append(node)
    return calls


def test_every_image_family_load_passes_max_soft_tokens() -> None:
    """The five sites, each with the keyword: a site added without it, or a sixth one, fails here."""
    found: dict[str, int] = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        for call in family_loads(path):
            rel = str(path.relative_to(ROOT))
            found[rel] = found.get(rel, 0) + 1
            keywords = {k.arg for k in call.keywords}
            assert "max_soft_tokens" in keywords, f"{rel}:{call.lineno} loads the image family without " \
                                                  "max_soft_tokens (images.max_soft_tokens)"
    assert found == SITES


def test_the_launcher_passes_the_pool_its_settings(tmp_path) -> None:
    """The launcher spawns each image pool worker with the derived images.max_soft_tokens and images.max_source_pixels
    by keyword, under worker_main's own names (the worker's use of them: tests/vision/test_gemma_vision.py and
    tests/vision/test_vision.py), on google/gemma-4-31B-it's configs (280 soft tokens unless set): the Gemma fixture's
    server tests check it too, but they skip without it."""
    import inspect
    import multiprocessing as mp
    import shutil
    from types import SimpleNamespace

    from llm_mini_infer import config, launcher
    from llm_mini_infer.vision import arena as px_arena
    from llm_mini_infer.vision.pool import worker_main

    model = tmp_path / "gemma"
    model.mkdir()
    for name in ("config.json", "processor_config.json"):
        shutil.copy(DATA / f"gemma-4-31b-it.{name}", model / name)
    cfg = config.derive(config.load(None, {"model.path": str(model), "runner.kind": "fake",
                                           "images.max_soft_tokens": 70, "images.max_source_pixels": 1_000_000,
                                           "images.workers": 2, "dev.images.arena_mb": 1}))
    spawned = []

    class Recorded:  # the spawn context's Process, recorded and never started
        pid = None

        def __init__(self, *, target, args, kwargs, name):
            self.target, self.args, self.kwargs, self.name = target, args, kwargs, name
            spawned.append(self)

        def start(self) -> None:
            pass

    readies, locks = [], []
    try:
        pool = launcher._start_images(cfg, SimpleNamespace(Pipe=mp.Pipe, Process=Recorded, RawArray=mp.RawArray),
                                      str(tmp_path), locks)
        readies = pool.start()
        respawn = pool.spawn(1, epoch=3)  # a dead worker's respawn: the same settings, its new epoch
        readies.append(respawn)
        assert len(readies) == len(spawned) == 3 and pool.procs == [spawned[0], spawned[2]]
        for i, p in zip((0, 1, 1), spawned):
            assert p.target is worker_main and set(p.kwargs) == {"max_soft_tokens", "max_source_pixels", "epoch",
                                                                 "state"}
            got = inspect.signature(worker_main).bind(*p.args, **p.kwargs).arguments
            assert (got["model_dir"], got["min_pixels"], got["max_pixels"], got["index"]) == (str(model), 2304,
                                                                                               161_280, i)
            assert (got["max_soft_tokens"], got["max_source_pixels"], got["state"]) == (70, 1_000_000, pool.state)
        assert [p.kwargs["epoch"] for p in spawned] == [0, 0, 3] and pool.state.epoch(1) == 3
    finally:
        for r in readies:
            r.close()
        px_arena.remove(cfg.pixel_arena)
        for fd in locks:  # the arena's lock (llm_mini_infer.leftovers), after its removal as the launcher does
            os.close(fd)


def test_the_guard_sees_every_import_form(tmp_path) -> None:
    """The guard's resolver on the forms the package uses and the ones it could."""
    pkg = tmp_path / "llm_mini_infer" / "vision"
    pkg.mkdir(parents=True)
    src = ("from . import family as fam\nfrom .family import load\nimport llm_mini_infer.vision.family as f2\n"
           "from .. import vision\nfam.load('a')\nload('b')\nf2.load('c', max_soft_tokens=70)\nvision.load('d')\n")
    (pkg / "x.py").write_text(src)
    calls = family_loads(pkg / "x.py", tmp_path)
    assert [ast.unparse(c.args[0]) for c in calls] == ["'a'", "'b'", "'c'"]


@pytest.fixture(scope="module")
def tiny_mm(tmp_path_factory):
    pytest.importorskip("transformers")
    from ..support.tiny_gemma import make_tiny_gemma_mm

    return make_tiny_gemma_mm(tmp_path_factory.mktemp("g4mm"), depth=1)


@pytest.mark.parametrize("invariant", [True, False], ids=["invariant", "not-invariant"])
def test_the_runner_builds_its_tower_buckets_at_the_setting(tiny_mm, caplog, invariant) -> None:
    """The runner's site sizes nothing, so the family's check cannot catch it: the tiny model's checkpoint says 70
    soft tokens, the setting 280, and the runner's options carry 280 to the tower, whose graph buckets are the 280
    family's. With batch invariance (Gemma 4's default: one tower call per image) the 19 up to the first that holds its
    largest image (2,520 patches; at 70, 4); without it the 28 to a 2,048-token encoder budget's 18,432 patches (at 70,
    25: the dense region ends at the largest image). Its startup line names tower graphs only when the runner captures
    CUDA graphs: on CPU the tower runs eagerly."""
    import logging

    from llm_mini_infer import config
    from llm_mini_infer.runner.cuda import make_cuda_runner
    from llm_mini_infer.runner.cuda.images import family_buckets
    from llm_mini_infer.vision.gemma import GemmaVision

    cfg = config.derive(config.load(None, {"model.path": str(tiny_mm), "runner.kind": "cuda",
                                           "dev.runner.cuda.device": "cpu", "images.max_soft_tokens": 280,
                                           "dev.runner.cuda.batch_invariant": invariant}))
    assert cfg.images.max_soft_tokens == 280 and cfg.dev.images.encoder_budget == 2048
    with caplog.at_level(logging.INFO, logger="llm_mini_infer.runner.cuda.gemma"):
        runner = make_cuda_runner(cfg)
    enc = runner.img
    fam = {k: GemmaVision(boi_id=501, image_id=500, eoi_id=502, patch=4, max_soft_tokens=k) for k in (280, 70)}
    count, top, at70 = (19, "2,565", 4) if invariant else (28, "18,432", 25)
    assert enc.max_soft_tokens == 280 and enc.per_image is invariant
    assert enc.buckets == family_buckets(fam[280], 3, 2048, per_image=invariant) and len(enc.buckets) == count
    assert len(family_buckets(fam[70], 3, 2048, per_image=invariant)) == at70
    (line,) = [r.getMessage() for r in caplog.records if r.getMessage().startswith("vision tower loaded")]
    assert not runner.use_graphs
    assert line.endswith("images of at most 280 soft tokens; no tower graphs (CUDA graphs off): the tower runs "
                         "eagerly"), line
    assert enc.describe(True).endswith(f"images of at most 280 soft tokens; tower graphs for {count} buckets of "
                                       f"261-{top} patches")
