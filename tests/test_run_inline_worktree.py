"""Tests for #171 — run_inline per-dispatch worktree isolation + merge-back.

Covers the ISOLATE (default, safe) vs THIS_BRANCH modes, the merge_back opt-out,
DONE terminalization, and the structural reuse of the executor's rebase-merge
machinery.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import milknado.adapters as adapters
from milknado.domains.common import RebaseResult, WorktreeMode
from milknado.mcp.run import (
    milknado_run_inline,
    milknado_run_inline_poll,
    milknado_run_inline_start,
)
from milknado.mcp.server import open_graph
from milknado.mcp.todo_mutate import milknado_todo_add


def _call(tool, **kwargs):
    fn = getattr(tool, "fn", tool)
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
    real = adapters.ProcessAdapter.run

    def spy(self, argv, cwd, *args, **kwargs):
        captured["cwd"] = Path(cwd)
        return real(self, argv, cwd, *args, **kwargs)

    monkeypatch.setattr(adapters.ProcessAdapter, "run", spy)
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


def _node(root: Path, node_id: int):
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

    def test_sync_second_dispatch_refused_without_spawning(
        self, tmp_path: Path, worker_writes_pwd, monkeypatch
    ) -> None:
        """The sync path refuses a DONE node up front and never spawns a worker —
        no run with a run_id it cannot finalize."""
        root = tmp_path / "repo"
        _init_repo(root)
        task = _call(milknado_todo_add, description="sonce", kind="task", project_root=str(root))

        first = _call(
            milknado_run_inline,
            node_id=task["id"],
            worker_cmd=worker_writes_pwd,
            project_root=str(root),
        )
        assert first["status"] == "done"

        spawned: list = []
        real = adapters.ProcessAdapter.run

        def spy(self, *args, **kwargs):
            spawned.append(args)
            return real(self, *args, **kwargs)

        monkeypatch.setattr(adapters.ProcessAdapter, "run", spy)

        with pytest.raises(ValueError, match="already done"):
            _call(
                milknado_run_inline,
                node_id=task["id"],
                worker_cmd=worker_writes_pwd,
                project_root=str(root),
            )
        assert spawned == []


class TestMergeBackReuse:
    def test_compare_and_swap_uses_captured_target(
        self, tmp_path: Path, worker_writes_pwd, monkeypatch
    ) -> None:
        root = tmp_path / "repo"
        _init_repo(root)
        calls: list[tuple[str, str, str]] = []
        original = adapters.GitAdapter.compare_and_swap_ref

        def spy(self, ref, expected_oid, new_oid):
            calls.append((ref, expected_oid, new_oid))
            return original(self, ref, expected_oid, new_oid)

        monkeypatch.setattr(adapters.GitAdapter, "compare_and_swap_ref", spy)
        task = _call(milknado_todo_add, description="reuse", kind="task", project_root=str(root))
        result = _call(
            milknado_run_inline,
            node_id=task["id"],
            worker_cmd=worker_writes_pwd,
            project_root=str(root),
        )
        assert result["status"] == "done"
        assert len(calls) == 1
        assert calls[0][0] == "refs/heads/main"

    def test_branch_switch_cannot_redirect_merge_target(self, tmp_path: Path, worker_stub) -> None:
        root = tmp_path / "repo"
        _init_repo(root)
        worker = tmp_path / "switch-worker.py"
        worker.write_text(
            "import pathlib, subprocess\n"
            f"root = pathlib.Path({str(root)!r})\n"
            "subprocess.run(['git', '-C', str(root), 'switch', '-q', '-c', "
            "'diversion'], check=True)\n"
            "pathlib.Path('landed.txt').write_text('immutable-target')\n"
        )
        task = _call(
            milknado_todo_add,
            description="immutable target",
            kind="task",
            project_root=str(root),
        )
        result = _call(
            milknado_run_inline,
            node_id=task["id"],
            worker_cmd=f"claude {sys.executable} {worker}",
            project_root=str(root),
        )
        assert result["status"] == "failed"
        assert _git(root, "branch", "--show-current").strip() == "diversion"
        assert not (root / "landed.txt").exists()
        assert result["worktree_preserved"]


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

        def _fail(_git, _root, ctx):
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

    def test_sync_merge_back_raise_terminalizes_node_and_run(
        self, tmp_path: Path, worker_writes_pwd, monkeypatch
    ) -> None:
        """A merge-back that RAISES (not just returns unlanded) must finalize the node
        and run FAILED before re-raising — never strand them on RUNNING."""
        import milknado.domains.dispatch.lifecycle as lifecycle_mod
        from milknado.domains.common.errors import RebaseAbortError

        root = tmp_path / "repo"
        _init_repo(root)

        def _boom(_git, _root, ctx):
            raise RebaseAbortError(ctx.worktree_path, "rebase --abort failed")

        monkeypatch.setattr(lifecycle_mod, "merge_back_isolated", _boom)
        task = _call(milknado_todo_add, description="raise", kind="task", project_root=str(root))

        with pytest.raises(RebaseAbortError):
            _call(
                milknado_run_inline,
                node_id=task["id"],
                worker_cmd=worker_writes_pwd,
                project_root=str(root),
            )

        assert _node(root, task["id"]).status.value == "failed"
        graph, _cfg = open_graph(root)
        try:
            latest = graph.recent_runs(1)[0]
        finally:
            graph.close()
        assert latest["status"] == "failed"

    def test_async_merge_failure_fails_node(
        self, tmp_path: Path, worker_writes_pwd, monkeypatch
    ) -> None:
        import milknado.domains.dispatch.async_run as async_mod
        from milknado.domains.dispatch.isolate import MergeBackResult

        root = tmp_path / "repo"
        _init_repo(root)

        def _fail(_git, _root, ctx):
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
    """Merge-back failures preserve the isolated worktree for recovery."""

    def _ctx(self, root: Path):
        from milknado.domains.dispatch.isolate import IsolateContext

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
        monkeypatch,
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

    def test_unlanded_work_error_preserves_worktree(self, tmp_path: Path, monkeypatch) -> None:
        """A teardown refusal (UnlandedWorkError) preserves the worktree and reports
        rebased=False — the branch never landed on the dispatch branch, so the merge
        did not succeed even though the worker exited clean."""
        from milknado.domains.common.errors import UnlandedWorkError
        from milknado.domains.dispatch.isolate import merge_back_isolated

        root = tmp_path / "repo"
        _init_repo(root)
        git, ctx = self._ctx(root)
        monkeypatch.setattr(git, "squash_and_commit", lambda *a, **k: True)
        monkeypatch.setattr(git, "rebase", lambda *a, **k: RebaseResult(success=True))
        monkeypatch.setattr(git, "resolve_ref", lambda *a, **k: "worker-oid")
        monkeypatch.setattr(git, "compare_and_swap_ref", lambda *a, **k: None)

        def _refuse(*a, **k):
            raise UnlandedWorkError(ctx.worktree_path, "dirty files:\nx")

        monkeypatch.setattr(git, "remove_worktree", _refuse)
        result = merge_back_isolated(git, root, ctx)

        assert result.rebased is False
        assert result.worktree_preserved == str(ctx.worktree_path)

    def test_unlanded_rebase_preserves_worktree(self, tmp_path: Path, monkeypatch) -> None:
        """A rebase that did not land (RebaseResult.success is False) is surfaced as
        rebased=False with the worktree preserved for inspection."""
        from milknado.domains.common.types import RebaseResult
        from milknado.domains.dispatch.isolate import merge_back_isolated

        root = tmp_path / "repo"
        _init_repo(root)
        git, ctx = self._ctx(root)
        monkeypatch.setattr(
            git,
            "squash_and_commit",
            lambda *a, **k: True,
        )
        monkeypatch.setattr(git, "rebase", lambda *a, **k: RebaseResult(success=False))
        result = merge_back_isolated(git, root, ctx)

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


class TestAsyncStartFailure:
    """A startup failure AFTER the node is claimed must finalize the node FAILED
    (never strand it RUNNING) and re-raise — the app-layer claim-release seam in
    milknado.app.run.run_inline_start."""

    def test_start_headless_failure_releases_claim_as_failed(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        import milknado.domains.dispatch as dispatch_mod

        root = tmp_path / "repo"
        _init_repo(root)
        task = _call(milknado_todo_add, description="boom", kind="task", project_root=str(root))

        def _explode(*args, **kwargs):
            raise RuntimeError("spawn exploded")

        monkeypatch.setattr(dispatch_mod, "start_headless_async", _explode)

        with pytest.raises(RuntimeError, match="spawn exploded"):
            _call(
                milknado_run_inline_start,
                node_id=task["id"],
                worker_cmd="claude sh -c 'true'",
                worktree=WorktreeMode.THIS_BRANCH,
                merge_back=False,
                project_root=str(root),
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

        def create_worktree(self, path: Path, branch: str) -> None:
            path.mkdir(parents=True)

        def force_remove_worktree(self, path: Path) -> None:
            self.removed.append(path)

    git = Git()
    expected = tmp_path / "milknado-8-race"
    with pytest.raises(GitOperationError, match="checkout changed"):
        create_isolated_worktree(git, tmp_path, 8, "race", "milknado-{node_id}-{slug}")  # type: ignore[arg-type]
    assert git.removed == [expected]


class TestMergeBackLock:
    """merge_back_isolated serializes concurrent merge-backs on a cross-process
    flock, so two dispatches never rebase onto the same dispatch branch at once."""

    def test_concurrent_merge_backs_do_not_overlap(self, tmp_path: Path, monkeypatch) -> None:
        import threading

        from milknado.domains.common.types import RebaseResult
        from milknado.domains.dispatch.isolate import IsolateContext, merge_back_isolated

        root = tmp_path / "repo"
        _init_repo(root)
        base_oid = adapters.GitAdapter(root).resolve_ref("refs/heads/main")
        intervals: list = []
        record_lock = threading.Lock()

        class _Git:
            def current_branch(self) -> str:
                return "main"

            def squash_and_commit(self, *args, **kwargs):
                start = time.monotonic()
                time.sleep(0.05)
                end = time.monotonic()
                with record_lock:
                    intervals.append((start, end))
                return True

            def rebase(self, *args, **kwargs):
                return RebaseResult(success=True)

            def resolve_ref(self, ref: str) -> str:
                return f"{ref}-oid"

            def compare_and_swap_ref(self, *args, **kwargs) -> None:
                return None

            def remove_worktree(self, *args, **kwargs) -> None:
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
            merge_back_isolated(_Git(), root, ctx)

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
    task = _call(milknado_todo_add, description="async fence", kind="task", project_root=str(root))
    graph, cfg = open_graph(root)
    try:
        monkeypatch.setattr(
            dispatch_mod,
            "start_headless_async",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("worker failed")),
        )
        monkeypatch.setattr(
            type(graph),
            "mark_terminal",
            lambda *_args, **_kwargs: False,
        )
        with pytest.raises(RuntimeError, match="startup terminal node write lost its fence"):
            run_inline_start(
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
