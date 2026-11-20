from __future__ import annotations

import contextlib
import ctypes
import fcntl
import os
import shutil
import stat
import sys
import tempfile

import pytest

# The helpers' asserts keep pytest's detailed failure messages, as they had in the test modules they came from
pytest.register_assert_rewrite("tests.support")

# The CUDA sync check of the runners the suite builds (`llm_mini_infer.runner.cuda.sync_check`): error; this variable
# set to warn or off for another. A runner on the CPU or the torch references never arms it
SYNC_CHECK_ENV = "LLM_MINI_INFER_TEST_SYNC_CHECK"
HELD = "held"  # in a test's private root (`private_tmp`): the lock its test holds while it runs


def pytest_configure(config) -> None:
    from llm_mini_infer.runner.cuda import sync_check

    sync_check.DEFAULT = os.environ.get(SYNC_CHECK_ENV, "error")
    _servers_root()


def _servers_root() -> None:
    """The servers' socket directories and pixel arenas go to a root of the tests' own (`llm_mini_infer.leftovers`,
    TEST_ROOT_ENV), never to /tmp or /dev/shm, where a starting server removes a killed one's files: no test sweeps the
    files of a server it did not start (another session's, a developer's). Set before the session's hooks and every
    test, and inherited by xdist's workers and the servers. One per user, kept between sessions, so a killed session's
    files are the next one's to sweep; its `shm` is a folder in /dev/shm where the host has one, so the arenas stay in
    shared memory as a server's do. A value already set is used as given."""
    from llm_mini_infer import leftovers

    if os.environ.get(leftovers.TEST_ROOT_ENV):
        return
    name = f"{leftovers.TEST_ROOT_PREFIX}{os.getuid()}"  # its servers log no warning for it (launcher._warn_test_root)
    root = os.path.join(leftovers.tmp_root(), name)  # /tmp: a socket's path holds about 104 bytes
    _own_folder(root)
    if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK):
        shm, link = os.path.join("/dev/shm", name), os.path.join(root, "shm")
        _own_folder(shm)
        with contextlib.suppress(FileExistsError):
            os.symlink(shm, link)
        if not os.path.islink(link) or os.readlink(link) != shm:
            raise pytest.UsageError(f"{link} is not a link to {shm}: remove it, or set {leftovers.TEST_ROOT_ENV}")
    os.environ[leftovers.TEST_ROOT_ENV] = root


def _own_folder(path: str) -> None:
    """`path`, made private when missing; refused unless it is a folder of this user's (another user could make it
    first in a shared /tmp)."""
    with contextlib.suppress(FileExistsError):
        os.mkdir(path, 0o700)
    st = os.lstat(path)
    if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.getuid():
        raise pytest.UsageError(f"{path} is not a folder of this user's (uid {st.st_uid}), and the test servers' files "
                                "need one: remove it, or set LLM_MINI_INFER_TEST_TMP")


@pytest.fixture
def private_tmp(monkeypatch):
    """A root of this test's own for the servers' socket directories and arenas (`_servers_root`), set for the test and
    the servers it starts, so that no other test's server sweeps what a killed one left before the test looks; with
    an `shm` folder (in /dev/shm) where the session's root has one. Removed at the end."""
    from llm_mini_infer import leftovers

    session = os.environ[leftovers.TEST_ROOT_ENV]
    root = tempfile.mkdtemp(prefix="t-", dir=session)
    held, made = None, [root]
    try:  # a root left without its lock (HELD) would stay for good: removed with the rest if the lock fails
        held = leftovers.locked_file(os.path.join(root, HELD))  # a killed session's go at its next start
        if os.path.isdir(os.path.join(session, "shm")):
            made.append(tempfile.mkdtemp(prefix="t-", dir=os.path.join(session, "shm")))
            os.symlink(made[-1], os.path.join(root, "shm"))
        monkeypatch.setenv(leftovers.TEST_ROOT_ENV, root)
        yield root
    finally:
        for path in made:
            shutil.rmtree(path, ignore_errors=True)
        if held is not None:
            os.close(held)


def remove_stale_roots(session: str) -> None:
    """The private roots (`private_tmp`) in `session` of tests whose pytest process was killed before their teardown,
    with what their servers left there, an arena in /dev/shm among them: the roots whose lock (HELD) nobody holds. A
    root whose test runs (another session's), or one without the lock file yet, stays."""
    try:
        names = sorted(name for name in os.listdir(session) if name.startswith("t-"))
    except OSError:
        return
    for name in names:
        root = os.path.join(session, name)
        try:
            fd = os.open(os.path.join(root, HELD), os.O_RDONLY | os.O_NOFOLLOW)
        except OSError:
            continue
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            shm = os.path.join(root, "shm")
            twin = os.readlink(shm) if os.path.islink(shm) else ""
            if os.path.dirname(twin) == os.path.join(session, "shm") and os.path.basename(twin).startswith("t-"):
                shutil.rmtree(twin, ignore_errors=True)
            shutil.rmtree(root, ignore_errors=True)
        except OSError:  # held: its test runs
            pass
        finally:
            os.close(fd)


try:
    _malloc_trim = ctypes.CDLL(None).malloc_trim  # glibc's; absent on macOS and musl
except (OSError, AttributeError):
    _malloc_trim = None
_TRIM_EVERY = 256 << 20  # bytes of growth between two trims within a test file
_trimmed_at = 0


def _resident() -> int:
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return 0


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item, nextitem):
    """The heap's free pages go back to the system (glibc's malloc_trim) after a test file's last test, and within a
    file whenever the process has grown by 256 MB. glibc keeps them otherwise, and a pytest process grows with every
    test that builds an application or a tokenizer in process: a worker measured 2.0 GB after
    tests/api/test_preparing.py, 0.8 GB trimmed, on a runner whose two workers and their servers share 7.8 GB."""
    global _trimmed_at
    yield
    if _malloc_trim is None:
        return
    last = nextitem is None or nextitem.path != item.path
    if last or _resident() - _trimmed_at > _TRIM_EVERY:
        _malloc_trim(0)
        _trimmed_at = _resident()


def _servers(load: bool):
    """The test servers' helper (tests/support/api_util.py): as this process imported it (only then did it launch
    servers), else imported now when `load`. None where it cannot load: the image checks' environment has none of its
    packages, and starts no server."""
    module = sys.modules.get("tests.support.api_util")
    if module is None and load:
        try:
            from tests.support import api_util as module
        except ImportError:
            return None
    return module


def _sweep(name: str, load: bool) -> None:
    """The helper's sweep `name` (`sweep`, `sweep_stale`), where it loads (`_servers`). Its error is a warning line on
    stderr, never raised: raised at the session's finish it would hide the tests' summary, at a worker's end stop
    xdist's controller."""
    try:
        servers = _servers(load)
        if servers is not None:
            getattr(servers, name)()
    except Exception as e:
        print(f"warning: the test servers' {name} failed: {type(e).__name__}: {e}", file=sys.stderr, flush=True)


@pytest.hookimpl(wrapper=True)
def pytest_sessionfinish(session):
    """No test server outlives its session (tests/support/api_util.py). After every other finish (the fixtures' last
    teardown, xdist's workers stopped), the servers this process launched and never stopped are killed; in xdist's
    controller, or a run without xdist, also those of the PID files whose pytest process is gone: a worker that was
    killed ran neither its teardown nor this hook."""
    try:
        return (yield)
    finally:
        controller = not hasattr(session.config, "workerinput")
        _sweep("sweep", load=controller)
        if controller:
            _sweep("sweep_stale", load=True)


def pytest_sessionstart(session) -> None:
    """A session starts without the servers of one that was killed (CI's serial run, after its workers' run hit its
    limit), then without its tests' private roots (`remove_stale_roots`): in xdist's controller, or a run without
    xdist, as at the finish below."""
    if not hasattr(session.config, "workerinput"):
        _sweep("sweep_stale", load=True)
        try:
            from llm_mini_infer import leftovers

            remove_stale_roots(os.environ[leftovers.TEST_ROOT_ENV])
        except Exception as e:  # as `_sweep`'s: a warning, never a stop
            print(f"warning: the tests' private roots' sweep failed: {type(e).__name__}: {e}", file=sys.stderr,
                  flush=True)


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node, error) -> None:
    """xdist's controller, a worker down with an error (killed, say by the OOM killer): its servers would hold their
    memory beside the workers still at work until the session ends. Swept at once: only servers whose pytest process
    is gone (`api_util.sweep_stale`)."""
    if error:
        _sweep("sweep_stale", load=True)


def pytest_addoption(parser) -> None:
    parser.addoption("--refresh-golden", action="store_true",
                     help="write or replace this GPU's entries of the committed goldens (tests/data/golden/, "
                          "tests/support/golden.py) instead of comparing runs with them")


# The GPU tests' marks (pyproject.toml's markers): what each needs, checked in this order, CUDA first, so that a host
# without it loads no library
NEEDS = {"cuda": None, "flashinfer": None, "gpu": None, "cutlass": None}


def _has(need: str, have: dict[str, bool]) -> bool:
    if need not in have:
        if need == "cuda":
            import torch

            have[need] = torch.cuda.is_available()
        elif need == "flashinfer":
            import importlib.util

            have[need] = importlib.util.find_spec("flashinfer") is not None
        else:
            from llm_mini_infer.kernels import build

            have[need] = (build.load() if need == "gpu" else build.load_cutlass()) is not None
    return have[need]


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items) -> None:
    """A test marked gpu, cutlass or flashinfer skips on a host without what its marks need, as skipifs with their
    reasons would: each level that carries one of the marks (the test, its class, its module) is checked, closest
    first, and the first whose marks' needs are missing gives the reason (cutlass's or flashinfer's over gpu's within a
    level). It runs after -m's deselection, so torch and the kernel library load only when a selected test carries a
    mark."""
    have: dict[str, bool] = {}
    for item in items:
        levels: list[list] = []  # [node, marks], closest first
        for node, mark in item.iter_markers_with_node():
            if mark.name in NEEDS:
                if levels and levels[-1][0] is node:
                    levels[-1][1].append(mark)
                else:
                    levels.append([node, [mark]])
        for _, marks in levels:
            ordered = sorted(marks, key=lambda m: m.name == "gpu")
            reasons = [m.kwargs["reason"] for m in ordered if "reason" in m.kwargs]
            if len(reasons) != 1:
                names = [m.name for m in marks]
                raise pytest.UsageError(f"{item.nodeid}: its {names} marks need one reason: {reasons}")
            names = {m.name for m in marks}
            if not all(_has(n, have) for n in NEEDS if n == "cuda" or n in names):
                item.add_marker(pytest.mark.skipif(True, reason=reasons[0]))
                break


@pytest.fixture
def refresh_golden(request) -> bool:
    """`--refresh-golden`: a golden test records its run (`golden.check`) instead of comparing it."""
    return bool(request.config.getoption("--refresh-golden"))


@pytest.fixture(autouse=True)
def _fused_ops_switch():
    """The fused-ops switch (`kernels.ops.enabled`), which a CUDA runner's constructor sets from `fused_ops`: as each
    test found it when it ends (on, the module's default, if the test imported it), so a runner built with fused_ops
    off leaves the next tests their kernels. A test that never imports the module does not import torch for it."""
    ops = sys.modules.get("llm_mini_infer.kernels.ops")
    before = ops.enabled() if ops is not None else True
    yield
    ops = sys.modules.get("llm_mini_infer.kernels.ops")
    if ops is not None and ops.enabled() != before:
        ops.enabled(before)


@pytest.fixture(autouse=True)
def _pdl_switch():
    """The kernel library's programmatic-dependent-launch flag (`kernels.ops.set_pdl`), which a CUDA runner's
    constructor sets from `pdl`: as each test found it when it ends (off, the library's default), so a runner built with
    PDL on leaves the next tests serialized launches. A test that never imports the module does not import torch for
    it."""
    ops = sys.modules.get("llm_mini_infer.kernels.ops")
    before = ops.pdl() if ops is not None else False
    yield
    ops = sys.modules.get("llm_mini_infer.kernels.ops")
    if ops is not None and ops.pdl() != before:
        ops.set_pdl(before)


