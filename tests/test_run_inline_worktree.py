"""Tests for #171 — run_inline per-dispatch worktree isolation + merge-back.

Covers the ISOLATE (default, safe) vs THIS_BRANCH modes, the merge_back opt-out,
DONE terminalization, and the structural reuse of the executor's rebase-merge
machinery.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pytest

import milknado.domains.dispatch.runner as runner_mod
import milknado.domains.execution.executor as executor_mod
from milknado.domains.common import WorktreeMode
from milknado.mcp_run import (
    milknado_run_inline,
    milknado_run_inline_poll,
    milknado_run_inline_start,
)
from milknado.mcp_server import open_graph
from milknado.mcp_todo_mutate import milknado_todo_add


def _call(tool, **kwargs):
    fn = getattr(tool, "fn", tool)
    return fn(**kwargs)


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=str(cwd), check=True, capture_output=True)


def _init_repo(root: Path) -> None:
    """A real git repo on branch `main` with one commit — the dispatch branch."""
    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.email", "t@t.com")
    _git(root, "config", "user.name", "T")
    (root / "README").write_text("x")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "init")


@pytest.fixture()
def worker_stub(tmp_path_factory, monkeypatch):
    """Install a `claude` shim that exec's its args, so a worker_cmd can run `sh -c`."""
    bindir = tmp_path_factory.mktemp("worker-stub-bin")
    stub = bindir / "claude"
    stub.write_text('#!/bin/sh\nexec "$@"\n')
    stub.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}:{__import__('os').environ.get('PATH', '')}")


@pytest.fixture()
def worker_writes_pwd(worker_stub):
    """A worker that writes its cwd to a deliverable and exits 0."""
    # Record the worker's cwd (proves ISOLATE ran outside project_root) and leave a
    # deliverable the merge-back can land.
    return "claude sh -c 'pwd > deliverable.txt'"


def _spy_worker_cwd(monkeypatch) -> dict:
    """Capture the cwd handed to the spawned worker while still spawning it."""
    captured: dict = {}
    real = runner_mod._spawn_worker

    def spy(cwd, *args, **kwargs):  # noqa: ANN002, ANN003
        captured["cwd"] = Path(cwd)
        return real(cwd, *args, **kwargs)

    monkeypatch.setattr(runner_mod, "_spawn_worker", spy)
    return captured


def _wait_for_terminal(run_id: str, root: str, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        last = _call(milknado_run_inline_poll, run_id=run_id, project_root=root)
        if last["status"] in ("done", "failed"):
            return last
        time.sleep(0.05)
    raise AssertionError(f"run {run_id} did not finish; last={last}")


def _node(root: Path, node_id: int):  # noqa: ANN202
    graph, _cfg = open_graph(root)
    try:
        return graph.get_node(node_id)
    finally:
        graph.close()


class TestIsolateDefault:
    """Criterion 2: omitted worktree → ISOLATE + merge_back."""

    def test_runs_in_worktree_merges_back_and_tears_down(
        self, tmp_path: Path, worker_writes_pwd, monkeypatch
    ) -> None:
        root = tmp_path / "repo"
        _init_repo(root)
        captured = _spy_worker_cwd(monkeypatch)
        task = _call(milknado_todo_add, description="isolate", kind="task", project_root=str(root))

        result = _call(
            milknado_run_inline,
            node_id=task["id"],
            worker_cmd=worker_writes_pwd,
            project_root=str(root),
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
        assert not Path(node.worktree_path).exists()


class TestIsolateNoMergeBack:
    """Criterion 3: ISOLATE + merge_back=False leaves the branch/worktree in place."""

    def test_worktree_persists_and_caller_untouched(
        self, tmp_path: Path, worker_writes_pwd
    ) -> None:
        root = tmp_path / "repo"
        _init_repo(root)
        task = _call(milknado_todo_add, description="keep", kind="task", project_root=str(root))

        result = _call(
            milknado_run_inline,
            node_id=task["id"],
            worker_cmd=worker_writes_pwd,
            merge_back=False,
            project_root=str(root),
        )

        assert result["status"] == "done"
        assert result["rebased"] is None  # no merge attempted
        node = _node(root, task["id"])
        wt = Path(node.worktree_path)
        # worktree + its deliverable survive; the caller checkout is untouched.
        assert wt.exists()
        assert (wt / "deliverable.txt").exists()
        assert not (root / "deliverable.txt").exists()


class TestThisBranch:
    """Criterion 4: worktree=THIS_BRANCH runs in the shared checkout."""

    def test_runs_in_project_root_no_worktree(
        self, tmp_path: Path, worker_writes_pwd, monkeypatch
    ) -> None:
        root = tmp_path / "repo"
        _init_repo(root)
        captured = _spy_worker_cwd(monkeypatch)
        task = _call(milknado_todo_add, description="shared", kind="task", project_root=str(root))

        result = _call(
            milknado_run_inline,
            node_id=task["id"],
            worker_cmd=worker_writes_pwd,
            worktree=WorktreeMode.THIS_BRANCH,
            project_root=str(root),
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

    def test_second_dispatch_refused_after_done(self, tmp_path: Path, worker_writes_pwd) -> None:
        root = tmp_path / "repo"
        _init_repo(root)
        task = _call(milknado_todo_add, description="once", kind="task", project_root=str(root))

        first = _call(
            milknado_run_inline,
            node_id=task["id"],
            worker_cmd=worker_writes_pwd,
            project_root=str(root),
        )
        assert first["status"] == "done"
        assert _node(root, task["id"]).status.value == "done"

        with pytest.raises(ValueError, match="already done|set status back to pending"):
            _call(
                milknado_run_inline_start,
                node_id=task["id"],
                worker_cmd=worker_writes_pwd,
                project_root=str(root),
            )


class TestMergeBackReuse:
    """Criterion 6: the ISOLATE merge-back reuses WorktreeManager.rebase_and_merge."""

    def test_delegates_to_executor_machinery(
        self, tmp_path: Path, worker_writes_pwd, monkeypatch
    ) -> None:
        root = tmp_path / "repo"
        _init_repo(root)
        calls: list[dict] = []
        original = executor_mod.WorktreeManager.rebase_and_merge

        def spy(self, worktree, feature_branch, node_id, description, worker_branch=None):  # noqa: ANN001, ANN202
            calls.append({"feature_branch": feature_branch, "worker_branch": worker_branch})
            return original(
                self, worktree, feature_branch, node_id, description, worker_branch=worker_branch
            )

        monkeypatch.setattr(executor_mod.WorktreeManager, "rebase_and_merge", spy)
        task = _call(milknado_todo_add, description="reuse", kind="task", project_root=str(root))

        _call(
            milknado_run_inline,
            node_id=task["id"],
            worker_cmd=worker_writes_pwd,
            project_root=str(root),
        )

        assert len(calls) == 1, "merge-back must go through the executor's rebase_and_merge"
        assert calls[0]["feature_branch"] == "main"  # the caller's dispatch branch
        assert calls[0]["worker_branch"] == f"milknado/{task['id']}-reuse"


class TestIsolateAsync:
    """The async start path also isolates + merges back (deferred, post-exit)."""

    def test_async_isolate_merges_back(self, tmp_path: Path, worker_writes_pwd) -> None:
        root = tmp_path / "repo"
        _init_repo(root)
        task = _call(milknado_todo_add, description="async", kind="task", project_root=str(root))

        started = _call(
            milknado_run_inline_start,
            node_id=task["id"],
            worker_cmd=worker_writes_pwd,
            project_root=str(root),
        )
        assert started["status"] == "running"
        final = _wait_for_terminal(started["run_id"], str(root))

        assert final["status"] == "done"
        assert final["rebased"] is True
        assert (root / "deliverable.txt").exists()
        node = _node(root, task["id"])
        assert not Path(node.worktree_path).exists()


class TestMergeBackFailure:
    """A merge-back that does not land (e.g. a rebase conflict) fails the node and
    preserves the worktree for inspection rather than falsely reporting DONE."""

    def test_sync_merge_failure_fails_node_and_preserves_worktree(
        self, tmp_path: Path, worker_writes_pwd, monkeypatch
    ) -> None:
        import milknado.domains.dispatch.lifecycle as lifecycle_mod
        from milknado.domains.dispatch.isolate import MergeBackResult

        root = tmp_path / "repo"
        _init_repo(root)

        def _fail(_root, ctx):  # noqa: ANN001, ANN202
            return MergeBackResult(rebased=False, worktree_preserved=str(ctx.worktree_path))

        monkeypatch.setattr(lifecycle_mod, "merge_back_isolated", _fail)
        task = _call(
            milknado_todo_add, description="conflict", kind="task", project_root=str(root)
        )

        result = _call(
            milknado_run_inline,
            node_id=task["id"],
            worker_cmd=worker_writes_pwd,
            project_root=str(root),
        )

        # Clean worker exit but the branch never landed → node FAILED, not DONE.
        assert result["exit_code"] == 0
        assert result["status"] == "failed"
        assert result["rebased"] is False
        node = _node(root, task["id"])
        assert node.status.value == "failed"
        # The preserved worktree is surfaced on the result and still on disk for
        # inspection; mark_failed nulls the node's worktree_path (shared convention).
        assert result["worktree_preserved"] == str(root / "milknado-1-conflict")
        assert Path(result["worktree_preserved"]).exists()
        assert node.worktree_path is None

    def test_async_merge_failure_fails_node(
        self, tmp_path: Path, worker_writes_pwd, monkeypatch
    ) -> None:
        import milknado.domains.dispatch.async_run as async_mod
        from milknado.domains.dispatch.isolate import MergeBackResult

        root = tmp_path / "repo"
        _init_repo(root)

        def _fail(_root, ctx):  # noqa: ANN001, ANN202
            return MergeBackResult(rebased=False, worktree_preserved=str(ctx.worktree_path))

        monkeypatch.setattr(async_mod, "merge_back_isolated", _fail)
        task = _call(
            milknado_todo_add, description="asyncfail", kind="task", project_root=str(root)
        )

        started = _call(
            milknado_run_inline_start,
            node_id=task["id"],
            worker_cmd=worker_writes_pwd,
            project_root=str(root),
        )
        final = _wait_for_terminal(started["run_id"], str(root))

        assert final["status"] == "failed"
        assert final["rebased"] is False
        # The unlanded worktree is surfaced on the async poll result too — the async
        # path persists it via the run row's `detail` column, mirroring the sync
        # dispatch dict key and milknado_run_cancel.
        assert final["worktree_preserved"] == str(root / "milknado-1-asyncfail")


class TestFailClosedPreservation:
    """isolate.merge_back_isolated's OWN fail-closed teardown preserves an unlanded
    worktree. The merge-failure tests above stub the whole function; these exercise
    its real branches by stubbing only WorktreeManager.rebase_and_merge one level
    down, so the preservation logic in isolate.py is what actually runs.
    """

    def _ctx(self, root: Path):  # noqa: ANN202
        from milknado.domains.dispatch.isolate import IsolateContext

        wt = root / "milknado-1-x"
        return IsolateContext(
            worktree_path=wt,
            worker_branch="milknado/1-x",
            feature_branch="main",
            node_id=1,
            description="x",
        )

    def test_unlanded_work_error_preserves_worktree(self, tmp_path: Path, monkeypatch) -> None:
        """A teardown refusal (UnlandedWorkError) preserves the worktree and still
        reports rebased=True — the branch landed but the dirty tree can't be torn down."""
        from milknado.domains.common.errors import UnlandedWorkError
        from milknado.domains.dispatch.isolate import merge_back_isolated

        root = tmp_path / "repo"
        _init_repo(root)
        ctx = self._ctx(root)

        def _refuse(self, *a, **k):  # noqa: ANN001, ANN002, ANN003, ANN202
            raise UnlandedWorkError(ctx.worktree_path, "dirty files:\nx")

        monkeypatch.setattr(executor_mod.WorktreeManager, "rebase_and_merge", _refuse)

        result = merge_back_isolated(root, ctx)

        assert result.rebased is True
        assert result.worktree_preserved == str(ctx.worktree_path)

    def test_unlanded_rebase_preserves_worktree(self, tmp_path: Path, monkeypatch) -> None:
        """A rebase that did not land (RebaseResult.success is False) is surfaced as
        rebased=False with the worktree preserved for inspection."""
        from milknado.domains.common.types import RebaseResult
        from milknado.domains.dispatch.isolate import merge_back_isolated

        root = tmp_path / "repo"
        _init_repo(root)
        ctx = self._ctx(root)

        monkeypatch.setattr(
            executor_mod.WorktreeManager,
            "rebase_and_merge",
            lambda self, *a, **k: RebaseResult(success=False),
        )

        result = merge_back_isolated(root, ctx)

        assert result.rebased is False
        assert result.worktree_preserved == str(ctx.worktree_path)


class TestIsolateWorkerFailure:
    """A worker that exits nonzero under ISOLATE must NOT merge its half-done branch
    back and must NOT be marked DONE — the merge-back gate keys on a clean exit.
    """

    def test_nonzero_exit_fails_node_and_skips_merge_back(
        self, tmp_path: Path, worker_stub
    ) -> None:
        root = tmp_path / "repo"
        _init_repo(root)
        task = _call(milknado_todo_add, description="boom", kind="task", project_root=str(root))

        # Worker produces a deliverable in its worktree, then fails.
        result = _call(
            milknado_run_inline,
            node_id=task["id"],
            worker_cmd="claude sh -c 'pwd > deliverable.txt; exit 7'",
            project_root=str(root),
        )

        assert result["exit_code"] == 7
        assert result["status"] == "failed"
        assert result["rebased"] is None  # no merge-back attempted on a failed worker
        # The worker's diff never reached the caller's dispatch branch.
        assert not (root / "deliverable.txt").exists()
        assert _node(root, task["id"]).status.value == "failed"
