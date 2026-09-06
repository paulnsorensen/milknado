"""Tests for #171 — run_inline per-dispatch worktree isolation + merge-back.

Covers the ISOLATE (default, safe) vs THIS_BRANCH modes, the merge_back opt-out,
DONE terminalization, and the structural reuse of the executor's rebase-merge
machinery.
"""

from __future__ import annotations

import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import pytest

import milknado.adapters as adapters
from milknado.domains.common import MikadoNode, RebaseResult, WorktreeMode
from milknado.domains.common.protocols import GitPort
from milknado.domains.dispatch import ProcessOutcome
from milknado.domains.dispatch.isolate import IsolateContext
from milknado.mcp._core import NodeSummary, RunDict
from milknado.mcp.run import (
    milknado_run_inline,
    milknado_run_inline_poll,
    milknado_run_inline_start,
)
from milknado.mcp.server import open_graph
from milknado.mcp.todo_mutate import milknado_todo_add


def _call(tool: object, **kwargs: object) -> object:
    fn = getattr(tool, "fn", tool)
    if not callable(fn):
        raise TypeError(f"tool is not callable: {tool!r}")
    return fn(**kwargs)


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _init_repo(root: Path) -> None:
    """A real git repo on branch `main` with one commit — the dispatch branch."""
    root.mkdir(parents=True, exist_ok=True)
    _ = _git(root, "init", "-b", "main")
    _ = _git(root, "config", "user.email", "t@t.com")
    _ = _git(root, "config", "user.name", "T")
    _ = (root / "README").write_text("x")
    _ = _git(root, "add", ".")
    _ = _git(root, "commit", "-m", "init")


@pytest.fixture()
def worker_writes_pwd(worker_stub: Callable[[str], str]) -> str:
    """A worker that writes its cwd to a deliverable and exits 0."""
    return worker_stub("sh -c 'pwd > deliverable.txt'")


def _spy_worker_cwd(monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Capture the cwd handed to the spawned worker while still spawning it."""
    captured: dict[str, Path] = {}
    real = adapters.ProcessAdapter.run

    def spy(  # noqa: PLR0913 - mirrors ProcessAdapter.run for monkeypatch typing
        self: adapters.ProcessAdapter,
        argv: tuple[str, ...],
        cwd: Path,
        log_path: Path,
        stdin: bytes,
        env: dict[str, str],
        timeout: float,
        *,
        cancel_requested: Callable[[], bool] | None = None,
        on_started: Callable[[int], None] | None = None,
    ) -> ProcessOutcome:
        captured["cwd"] = cwd
        return real(
            self,
            argv,
            cwd,
            log_path,
            stdin,
            env,
            timeout,
            cancel_requested=cancel_requested,
            on_started=on_started,
        )

    monkeypatch.setattr(adapters.ProcessAdapter, "run", spy)
    return captured


def _wait_for_terminal(run_id: str, root: str, timeout: float = 10.0) -> RunDict:
    deadline = time.monotonic() + timeout
    last: RunDict | None = None
    while time.monotonic() < deadline:
        last = cast(RunDict, _call(milknado_run_inline_poll, run_id=run_id, project_root=root))
        if last["status"] in ("done", "failed"):
            return last
        time.sleep(0.05)
    raise AssertionError(f"run {run_id} did not finish; last={last}")


def _node(root: Path, node_id: int) -> MikadoNode:
    graph, _cfg = open_graph(root)
    try:
        node = graph.get_node(node_id)
        if node is None:
            raise AssertionError(f"node {node_id} not found")
        return node
    finally:
        graph.close()


def _squash_success(_worktree: Path, _onto: str, _message: str) -> bool:
    return True


def _rebase_success(_worktree: Path, _onto: str) -> RebaseResult:
    return RebaseResult(success=True)


def _rebase_failure(_worktree: Path, _onto: str) -> RebaseResult:
    return RebaseResult(success=False)


def _resolve_worker_oid(_ref: str) -> str:
    return "worker-oid"


def _ignore_compare_and_swap(_ref: str, _expected_oid: str, _new_oid: str) -> None:
    return None


class TestIsolateDefault:
    """Criterion 2: omitted worktree → ISOLATE + merge_back."""

    def test_runs_in_worktree_merges_back_and_tears_down(
        self, tmp_path: Path, worker_writes_pwd: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = tmp_path / "repo"
        _init_repo(root)
        captured = _spy_worker_cwd(monkeypatch)
        task = cast(
            NodeSummary,
            _call(milknado_todo_add, description="isolate", kind="task", project_root=str(root)),
        )

        result = cast(
            RunDict,
            _call(
                milknado_run_inline,
                node_id=task["id"],
                worker_cmd=worker_writes_pwd,
                project_root=str(root),
            ),
        )

        assert result["exit_code"] == 0
        assert result["status"] == "done"
        assert result["rebased"] is True
        # Worker ran in the isolated worktree, NOT the shared checkout.
        assert captured["cwd"] != root
        assert captured["cwd"].name.startswith("milknado-")
        # Deliverable landed on the caller's dispatch branch.
        assert (root / "deliverable.txt").exists()
        # Worktree torn down.
        node = _node(root, task["id"])
        assert node.worktree_path is not None
        assert not Path(node.worktree_path).exists()


class TestIsolateNoMergeBack:
    """Criterion 3: ISOLATE + merge_back=False leaves the branch/worktree in place."""

    def test_worktree_persists_and_caller_untouched(
        self, tmp_path: Path, worker_writes_pwd: str
    ) -> None:
        root = tmp_path / "repo"
        _init_repo(root)
        task = cast(
            NodeSummary,
            _call(milknado_todo_add, description="keep", kind="task", project_root=str(root)),
        )

        result = cast(
            RunDict,
            _call(
                milknado_run_inline,
                node_id=task["id"],
                worker_cmd=worker_writes_pwd,
                merge_back=False,
                project_root=str(root),
            ),
        )

        assert result["status"] == "done"
        assert result["rebased"] is None  # no merge attempted
        node = _node(root, task["id"])
        assert node.worktree_path is not None
        wt = Path(node.worktree_path)
        assert (wt / "deliverable.txt").exists()
        assert not (root / "deliverable.txt").exists()


class TestThisBranch:
    """Criterion 4: worktree=THIS_BRANCH runs in the shared checkout."""

    def test_runs_in_project_root_no_worktree(
        self, tmp_path: Path, worker_writes_pwd: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = tmp_path / "repo"
        _init_repo(root)
        captured = _spy_worker_cwd(monkeypatch)
        task = cast(
            NodeSummary,
            _call(milknado_todo_add, description="shared", kind="task", project_root=str(root)),
        )

        result = cast(
            RunDict,
            _call(
                milknado_run_inline,
                node_id=task["id"],
                worker_cmd=worker_writes_pwd,
                worktree=WorktreeMode.THIS_BRANCH,
                project_root=str(root),
            ),
        )

        assert result["status"] == "done"
        assert result["rebased"] is None  # merge_back is a no-op under THIS_BRANCH
        assert captured["cwd"] == root
        node = _node(root, task["id"])
        assert node.worktree_path in (None, "")
        # The diff landed in the shared checkout (worker wrote its cwd = root).
        assert (root / "deliverable.txt").exists()


class TestDoneTerminalization:
    """Criterion 5: exit 0 marks the node DONE-terminal; re-dispatch is refused."""

    def test_second_dispatch_refused_after_done(
        self, tmp_path: Path, worker_writes_pwd: str
    ) -> None:
        root = tmp_path / "repo"
        _init_repo(root)
        task = cast(
            NodeSummary,
            _call(milknado_todo_add, description="once", kind="task", project_root=str(root)),
        )

        first = cast(
            RunDict,
            _call(
                milknado_run_inline,
                node_id=task["id"],
                worker_cmd=worker_writes_pwd,
                project_root=str(root),
            ),
        )
        assert first["status"] == "done"
        assert _node(root, task["id"]).status.value == "done"

        with pytest.raises(ValueError, match="already done|set status back to pending"):
            _ = cast(
                RunDict,
                _call(
                    milknado_run_inline_start,
                    node_id=task["id"],
                    worker_cmd=worker_writes_pwd,
                    project_root=str(root),
                ),
            )

    def test_sync_second_dispatch_refused_without_spawning(
        self, tmp_path: Path, worker_writes_pwd: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The sync path refuses a DONE node up front and never spawns a worker —
        no run with a run_id it cannot finalize."""
        root = tmp_path / "repo"
        _init_repo(root)
        task = cast(
            NodeSummary,
            _call(milknado_todo_add, description="sonce", kind="task", project_root=str(root)),
        )

        first = cast(
            RunDict,
            _call(
                milknado_run_inline,
                node_id=task["id"],
                worker_cmd=worker_writes_pwd,
                project_root=str(root),
            ),
        )
        assert first["status"] == "done"

        spawned: list[tuple[str, ...]] = []
        real = adapters.ProcessAdapter.run

        def spy(  # noqa: PLR0913 - mirrors ProcessAdapter.run for monkeypatch typing
            self: adapters.ProcessAdapter,
            argv: tuple[str, ...],
            cwd: Path,
            log_path: Path,
            stdin: bytes,
            env: dict[str, str],
            timeout: float,
            *,
            cancel_requested: Callable[[], bool] | None = None,
            on_started: Callable[[int], None] | None = None,
        ) -> ProcessOutcome:
            spawned.append(argv)
            return real(
                self,
                argv,
                cwd,
                log_path,
                stdin,
                env,
                timeout,
                cancel_requested=cancel_requested,
                on_started=on_started,
            )

        monkeypatch.setattr(adapters.ProcessAdapter, "run", spy)

        with pytest.raises(ValueError, match="already done"):
            _ = cast(
                RunDict,
                _call(
                    milknado_run_inline,
                    node_id=task["id"],
                    worker_cmd=worker_writes_pwd,
                    project_root=str(root),
                ),
            )
        assert spawned == []


class TestMergeBackReuse:
    def test_compare_and_swap_uses_captured_target(
        self, tmp_path: Path, worker_writes_pwd: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = tmp_path / "repo"
        _init_repo(root)
        calls: list[tuple[str, str, str]] = []
        original = adapters.GitAdapter.compare_and_swap_ref

        def spy(
            self: adapters.GitAdapter,
            ref: str,
            expected_oid: str,
            new_oid: str,
        ) -> None:
            calls.append((ref, expected_oid, new_oid))
            return original(self, ref, expected_oid, new_oid)

        monkeypatch.setattr(adapters.GitAdapter, "compare_and_swap_ref", spy)
        task = cast(
            NodeSummary,
            _call(
                milknado_todo_add,
                description="reuse",
                kind="task",
                project_root=str(root),
            ),
        )
        result = cast(
            RunDict,
            _call(
                milknado_run_inline,
                node_id=task["id"],
                worker_cmd=worker_writes_pwd,
                project_root=str(root),
            ),
        )
        assert result["status"] == "done"
        assert len(calls) == 1
        assert calls[0][0] == "refs/heads/main"

    def test_branch_switch_cannot_redirect_merge_target(
        self, tmp_path: Path, worker_stub: Callable[[str], str]
    ) -> None:
        root = tmp_path / "repo"
        _init_repo(root)
        worker = tmp_path / "switch-worker.py"
        _ = worker.write_text(
            "import pathlib, subprocess\n"
            + f"root = pathlib.Path({str(root)!r})\n"
            + "subprocess.run(['git', '-C', str(root), 'switch', '-q', '-c', "
            + "'diversion'], check=True)\n"
            + "pathlib.Path('landed.txt').write_text('immutable-target')\n"
        )
        task = cast(
            NodeSummary,
            _call(
                milknado_todo_add,
                description="immutable target",
                kind="task",
                project_root=str(root),
            ),
        )
        result = cast(
            RunDict,
            _call(
                milknado_run_inline,
                node_id=task["id"],
                worker_cmd=worker_stub(f"{sys.executable} {worker}"),
                project_root=str(root),
            ),
        )
        assert result["status"] == "failed"
        assert _git(root, "branch", "--show-current").strip() == "diversion"
        assert not (root / "landed.txt").exists()
        assert result["worktree_preserved"]


class TestIsolateAsync:
    """The async start path also isolates + merges back (deferred, post-exit)."""

    def test_async_isolate_merges_back(self, tmp_path: Path, worker_writes_pwd: str) -> None:
        root = tmp_path / "repo"
        _init_repo(root)
        task = cast(
            NodeSummary,
            _call(milknado_todo_add, description="async", kind="task", project_root=str(root)),
        )

        started = cast(
            RunDict,
            _call(
                milknado_run_inline_start,
                node_id=task["id"],
                worker_cmd=worker_writes_pwd,
                project_root=str(root),
            ),
        )
        assert started["status"] == "running"
        run_id = started["run_id"]
        assert run_id is not None
        final = _wait_for_terminal(run_id, str(root))

        assert final["status"] == "done"
        assert final["rebased"] is True
        assert (root / "deliverable.txt").exists()
        node = _node(root, task["id"])
        assert node.worktree_path is not None
        assert not Path(node.worktree_path).exists()


class TestMergeBackFailure:
    """A merge-back that does not land (e.g. a rebase conflict) fails the node and
    preserves the worktree for inspection rather than falsely reporting DONE."""

    def test_sync_merge_failure_fails_node_and_preserves_worktree(
        self, tmp_path: Path, worker_writes_pwd: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import milknado.domains.dispatch.lifecycle as lifecycle_mod
        from milknado.domains.dispatch.isolate import MergeBackResult

        root = tmp_path / "repo"
        _init_repo(root)

        def _fail(_git: GitPort, _root: Path, ctx: IsolateContext) -> MergeBackResult:
            return MergeBackResult(rebased=False, worktree_preserved=str(ctx.worktree_path))

        monkeypatch.setattr(lifecycle_mod, "merge_back_isolated", _fail)
        task = cast(
            NodeSummary,
            _call(
                milknado_todo_add,
                description="conflict",
                kind="task",
                project_root=str(root),
            ),
        )
        result = cast(
            RunDict,
            _call(
                milknado_run_inline,
                node_id=task["id"],
                worker_cmd=worker_writes_pwd,
                project_root=str(root),
            ),
        )

        # Clean worker exit but the branch never landed → node FAILED, not DONE.
        assert result["exit_code"] == 0
        assert result["status"] == "failed"
        assert result["rebased"] is False
        node = _node(root, task["id"])
        assert node.status.value == "failed"
        preserved = result["worktree_preserved"]
        assert preserved is not None
        assert preserved == str(root / "milknado-1-conflict")
        assert Path(preserved).exists()
        assert node.worktree_path is None

    def test_sync_merge_back_raise_terminalizes_node_and_run(
        self, tmp_path: Path, worker_writes_pwd: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A merge-back that RAISES must finalize the node and run FAILED."""
        import milknado.domains.dispatch.lifecycle as lifecycle_mod
        from milknado.domains.common.errors import RebaseAbortError

        root = tmp_path / "repo"
        _init_repo(root)

        def _boom(_git: GitPort, _root: Path, ctx: IsolateContext) -> None:
            raise RebaseAbortError(ctx.worktree_path, "rebase --abort failed")

        monkeypatch.setattr(lifecycle_mod, "merge_back_isolated", _boom)
        task = cast(
            NodeSummary,
            _call(
                milknado_todo_add,
                description="raise",
                kind="task",
                project_root=str(root),
            ),
        )

        with pytest.raises(RebaseAbortError):
            _ = cast(
                RunDict,
                _call(
                    milknado_run_inline,
                    node_id=task["id"],
                    worker_cmd=worker_writes_pwd,
                    project_root=str(root),
                ),
            )

        assert _node(root, task["id"]).status.value == "failed"
        graph, _cfg = open_graph(root)
        try:
            latest = graph.recent_runs(1)[0]
        finally:
            graph.close()
        assert latest["status"] == "failed"

    def test_async_merge_failure_fails_node(
        self, tmp_path: Path, worker_writes_pwd: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import milknado.domains.dispatch.async_run as async_mod
        from milknado.domains.dispatch.isolate import MergeBackResult

        root = tmp_path / "repo"
        _init_repo(root)

        def _fail(_git: GitPort, _root: Path, ctx: IsolateContext) -> MergeBackResult:
            return MergeBackResult(rebased=False, worktree_preserved=str(ctx.worktree_path))

        monkeypatch.setattr(async_mod, "merge_back_isolated", _fail)
        task = cast(
            NodeSummary,
            _call(
                milknado_todo_add,
                description="asyncfail",
                kind="task",
                project_root=str(root),
            ),
        )
        started = cast(
            RunDict,
            _call(
                milknado_run_inline_start,
                node_id=task["id"],
                worker_cmd=worker_writes_pwd,
                project_root=str(root),
            ),
        )
        run_id = started["run_id"]
        assert run_id is not None
        final = _wait_for_terminal(run_id, str(root))

        assert final["status"] == "failed"
        assert final["rebased"] is False
        # The unlanded worktree is surfaced on the async poll result too.
        assert final["worktree_preserved"] == str(root / "milknado-1-asyncfail")


class TestFailClosedPreservation:
    """Merge-back failures preserve the isolated worktree for recovery."""

    def _ctx(self, root: Path) -> tuple[adapters.GitAdapter, IsolateContext]:

        wt = root / "milknado-1-x"
        git = adapters.GitAdapter(root)
        return git, IsolateContext(
            worktree_path=wt,
            worker_branch="milknado/1-x",
            target_branch="main",
            base_oid=git.resolve_ref("refs/heads/main"),
            node_id=1,
            description="x",
        )

    def test_target_mismatch_preserves_refs_before_merge_mutation(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from milknado.domains.dispatch.isolate import merge_back_isolated

        root = tmp_path / "repo"
        _init_repo(root)
        git, ctx = self._ctx(root)
        before = _git(root, "rev-parse", "refs/heads/main")
        squash = MagicMock()
        rebase = MagicMock()
        compare_and_swap = MagicMock()
        remove = MagicMock()
        monkeypatch.setattr(git, "current_branch", lambda: "other")
        monkeypatch.setattr(git, "squash_and_commit", squash)
        monkeypatch.setattr(git, "rebase", rebase)
        monkeypatch.setattr(git, "compare_and_swap_ref", compare_and_swap)
        monkeypatch.setattr(git, "remove_worktree", remove)

        result = merge_back_isolated(git, root, ctx)

        assert result.rebased is False
        assert result.worktree_preserved == str(ctx.worktree_path)
        assert _git(root, "rev-parse", "refs/heads/main") == before
        squash.assert_not_called()
        rebase.assert_not_called()
        compare_and_swap.assert_not_called()
        remove.assert_not_called()

    def test_unlanded_work_error_preserves_worktree(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A teardown refusal (UnlandedWorkError) preserves the worktree and reports
        rebased=False — the branch never landed on the dispatch branch, so the merge
        did not succeed even though the worker exited clean."""
        from milknado.domains.common.errors import UnlandedWorkError
        from milknado.domains.dispatch.isolate import merge_back_isolated

        root = tmp_path / "repo"
        _init_repo(root)
        git, ctx = self._ctx(root)
        monkeypatch.setattr(git, "squash_and_commit", _squash_success)
        monkeypatch.setattr(git, "rebase", _rebase_success)
        monkeypatch.setattr(git, "resolve_ref", _resolve_worker_oid)
        monkeypatch.setattr(git, "compare_and_swap_ref", _ignore_compare_and_swap)

        def _refuse(_worktree: Path, target: str = "HEAD") -> None:
            _ = target
            raise UnlandedWorkError(ctx.worktree_path, "dirty files:\nx")

        monkeypatch.setattr(git, "remove_worktree", _refuse)
        result = merge_back_isolated(git, root, ctx)

        assert result.rebased is False
        assert result.worktree_preserved == str(ctx.worktree_path)

    def test_unlanded_rebase_preserves_worktree(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A rebase that did not land (RebaseResult.success is False) is surfaced as
        rebased=False with the worktree preserved for inspection."""
        from milknado.domains.dispatch.isolate import merge_back_isolated

        root = tmp_path / "repo"
        _init_repo(root)
        git, ctx = self._ctx(root)
        monkeypatch.setattr(git, "squash_and_commit", _squash_success)
        monkeypatch.setattr(git, "rebase", _rebase_failure)
        result = merge_back_isolated(git, root, ctx)

        assert result.rebased is False
        assert result.worktree_preserved == str(ctx.worktree_path)


class TestIsolateWorkerFailure:
    """A worker that exits nonzero under ISOLATE must NOT merge its half-done branch
    back and must NOT be marked DONE — the merge-back gate keys on a clean exit.
    """

    def test_nonzero_exit_fails_node_and_skips_merge_back(
        self, tmp_path: Path, worker_stub: Callable[[str], str]
    ) -> None:
        root = tmp_path / "repo"
        _init_repo(root)
        task = cast(
            NodeSummary,
            _call(milknado_todo_add, description="boom", kind="task", project_root=str(root)),
        )

        # Worker produces a deliverable in its worktree, then fails.
        result = cast(
            RunDict,
            _call(
                milknado_run_inline,
                node_id=task["id"],
                worker_cmd=worker_stub("sh -c 'pwd > deliverable.txt; exit 7'"),
                project_root=str(root),
            ),
        )

        assert result["exit_code"] == 7
        assert result["status"] == "failed"
        assert result["rebased"] is None  # no merge-back attempted on a failed worker
        # The worker's diff never reached the caller's dispatch branch.
        assert not (root / "deliverable.txt").exists()
        assert _node(root, task["id"]).status.value == "failed"


class TestAsyncStartFailure:
    """A startup failure AFTER the node is claimed must finalize the node FAILED
    (never strand it RUNNING) and re-raise — the app-layer claim-release seam in
    milknado.app.run.run_inline_start."""

    def test_start_headless_failure_releases_claim_as_failed(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        worker_stub: Callable[[str], str],
    ) -> None:
        import milknado.domains.dispatch as dispatch_mod

        root = tmp_path / "repo"
        _init_repo(root)
        task = cast(
            NodeSummary,
            _call(milknado_todo_add, description="boom", kind="task", project_root=str(root)),
        )

        def _explode(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("spawn exploded")

        monkeypatch.setattr(dispatch_mod, "start_headless_async", _explode)

        with pytest.raises(RuntimeError, match="spawn exploded"):
            _ = cast(
                RunDict,
                _call(
                    milknado_run_inline_start,
                    node_id=task["id"],
                    worker_cmd=worker_stub("sh -c 'true'"),
                    worktree=WorktreeMode.THIS_BRANCH,
                    merge_back=False,
                    project_root=str(root),
                ),
            )

        # The claim was released with a fenced terminal write — not stranded RUNNING.
        assert _node(root, task["id"]).status.value == "failed"


def test_isolated_worktree_removes_checkout_when_base_moves(tmp_path: Path) -> None:
    from milknado.domains.common.errors import GitOperationError
    from milknado.domains.dispatch.isolate import create_isolated_worktree

    class Git:
        def __init__(self) -> None:
            self.removed: list[Path] = []

        def current_branch(self) -> str:
            return "main"

        def resolve_ref(self, ref: str) -> str:
            return "base-before" if ref == "refs/heads/main" else "base-after"

        def create_worktree(self, path: Path, _branch: str) -> None:
            path.mkdir(parents=True)

        def force_remove_worktree(self, path: Path) -> None:
            self.removed.append(path)

    git = Git()
    expected = tmp_path / "milknado-8-race"
    with pytest.raises(GitOperationError, match="checkout changed"):
        _ = create_isolated_worktree(
            cast(GitPort, cast(object, git)), tmp_path, 8, "race", "milknado-{node_id}-{slug}"
        )
    assert git.removed == [expected]


class TestMergeBackLock:
    """merge_back_isolated serializes concurrent merge-backs on a cross-process
    flock, so two dispatches never rebase onto the same dispatch branch at once."""

    def test_concurrent_merge_backs_do_not_overlap(self, tmp_path: Path) -> None:
        import threading

        from milknado.domains.dispatch.isolate import IsolateContext, merge_back_isolated

        root = tmp_path / "repo"
        _init_repo(root)
        base_oid = adapters.GitAdapter(root).resolve_ref("refs/heads/main")
        intervals: list[tuple[float, float]] = []
        record_lock = threading.Lock()

        class _Git:
            def current_branch(self) -> str:
                return "main"

            def squash_and_commit(self, _worktree: Path, _onto: str, _message: str) -> bool:
                start = time.monotonic()
                time.sleep(0.05)
                end = time.monotonic()
                with record_lock:
                    intervals.append((start, end))
                return True

            def rebase(self, _worktree: Path, _onto: str) -> RebaseResult:
                return RebaseResult(success=True)

            def resolve_ref(self, ref: str) -> str:
                return f"{ref}-oid"

            def compare_and_swap_ref(self, _ref: str, _expected_oid: str, _new_oid: str) -> None:
                return None

            def remove_worktree(self, _path: Path, target: str = "HEAD") -> None:
                _ = target
                return None

        def _run(n: int) -> None:
            ctx = IsolateContext(
                worktree_path=root / f"milknado-{n}-x",
                worker_branch=f"milknado/{n}-x",
                target_branch="main",
                base_oid=base_oid,
                node_id=n,
                description="x",
            )
            _ = merge_back_isolated(cast(GitPort, cast(object, _Git())), root, ctx)

        threads = [threading.Thread(target=_run, args=(n,)) for n in (1, 2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(intervals) == 2
        intervals.sort()
        # The flock forces the second critical section to start only after the
        # first has released — the intervals must not overlap.
        assert intervals[0][1] <= intervals[1][0]


def test_async_start_reports_lost_terminal_fence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import milknado.domains.dispatch as dispatch_mod
    from milknado.app.run import InlineRunRequest, run_inline_start

    root = tmp_path / "repo"
    _init_repo(root)
    task = cast(
        NodeSummary,
        _call(milknado_todo_add, description="async fence", kind="task", project_root=str(root)),
    )
    graph, cfg = open_graph(root)
    try:

        def _worker_failed(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("worker failed")

        def _terminal_write_failed(*_args: object, **_kwargs: object) -> bool:
            return False

        monkeypatch.setattr(dispatch_mod, "start_headless_async", _worker_failed)
        monkeypatch.setattr(type(graph), "mark_terminal", _terminal_write_failed)
        with pytest.raises(RuntimeError, match="startup terminal node write lost its fence"):
            _ = run_inline_start(
                graph,
                cfg,
                root,
                InlineRunRequest(
                    node_id=task["id"],
                    worker_cmd=None,
                    timeout_seconds=1,
                    worktree=WorktreeMode.THIS_BRANCH,
                    merge_back=False,
                ),
                use_tmux=False,
            )
    finally:
        graph.close()
