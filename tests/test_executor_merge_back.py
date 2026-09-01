"""Integration tests for the worker-branch merge-back path (vendored-loop-hardening #4).

The vendored ralph engine left a defect: a worker's rebased branch was never merged
back into the feature branch, so a DONE node deposited zero commits onto the feature
branch — work stranded on `milknado/<id>-*`. These tests drive `Executor.complete`
through a *real* git repo + linked worktree (not a fake) and assert the three
coordinated fixes land together:

1. The worker's squashed commit ends up on the feature branch (fast-forward merge-back).
2. The loop's own scaffolding (RALPH.md, .ralph-logs/) never reaches that commit
   (worktree-local `.git/info/exclude`).
3. The run's `.ralph-logs/` is preserved to `.milknado/logs/<node_id>/` before the
   worktree is destroyed (post-mortem evidence).

A fast-forward that cannot apply (dirty/diverged project root) must raise, never
silently skip — otherwise the stranded-work bug returns disguised as success.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from operator import attrgetter
from pathlib import Path
from typing import cast

import pytest

import milknado.domains.execution.executor as _executor_module
from milknado.adapters.git import GitAdapter
from milknado.domains.common.errors import GitOperationError
from milknado.domains.common.protocols import CrgPort, GitPort, LoopPort
from milknado.domains.common.types import RebaseResult
from milknado.domains.execution.executor import ExecutionConfig, Executor, WorktreeManager
from milknado.domains.graph import MikadoGraph

_EXCLUDE_LOOP_SCAFFOLDING = cast(
    Callable[[Path], None], attrgetter("_exclude_loop_scaffolding")(_executor_module)
)
_PRESERVE_RUN_LOGS = cast(
    Callable[[Path, int], None], attrgetter("_preserve_run_logs")(_executor_module)
)


def _git_port(value: object) -> GitPort:
    return cast(GitPort, value)


def _loop_port(value: object) -> LoopPort:
    return cast(LoopPort, value)


def _crg_port(value: object) -> CrgPort:
    return cast(CrgPort, value)


def _worker_run_ids(executor: Executor) -> dict[int, str]:
    return cast(dict[int, str], attrgetter("_worker_run_id_by_node")(executor))


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _init_project(root: Path) -> None:
    """A git repo at `root` with one commit on a `feature` branch checked out."""
    _ = _git(root, "init", "-q", "-b", "feature", str(root))
    _ = _git(root, "config", "user.email", "test@test.com")
    _ = _git(root, "config", "user.name", "Test")
    _ = (root / "README.md").write_text("seed\n")
    _ = _git(root, "add", ".")
    _ = _git(root, "commit", "-qm", "seed")


def _worker_did_work(worktree: Path) -> None:
    """Simulate a ralph run inside `worktree`: a real source change committed on the
    worker branch, plus the loop's own scaffolding left lying around uncommitted."""
    _ = (worktree / "feature.py").write_text("def added():\n    return 1\n")
    _ = _git(worktree, "add", "feature.py")
    _ = _git(worktree, "commit", "-qm", "wip: worker change")
    # Loop scaffolding the worker must NOT carry into the squashed commit.
    _ = (worktree / "RALPH.md").write_text("# loop scaffolding\n")
    logs = worktree / ".ralph-logs"
    logs.mkdir()
    _ = (logs / "run.log").write_text("iteration 1 log line\n")


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    root.mkdir()
    _init_project(root)
    return root


def _running_node(graph: MikadoGraph, worktree: Path, branch: str) -> int:
    _ = graph.add_node("Add the added() helper")
    graph.mark_running(1, worktree_path=str(worktree), branch_name=branch)
    return 1


class TestMergeBackIntegration:
    def test_worker_commit_lands_on_feature_branch(self, project: Path, tmp_path: Path) -> None:
        """After complete(), the worker's squashed commit is on the feature branch —
        the defect this curd fixes was the feature branch receiving zero commits."""
        git = GitAdapter(project)
        branch = "milknado/1-added"
        wt = project / "milknado-1-added"
        _ = git.create_worktree(wt, branch)
        _EXCLUDE_LOOP_SCAFFOLDING(wt)
        _worker_did_work(wt)

        graph = MikadoGraph(tmp_path / "g.db")
        try:
            _ = _running_node(graph, wt, branch)
            ex = Executor(
                graph=graph, git=git, ralph=_loop_port(_NoRalph()), crg=_crg_port(_NoCrg())
            )
            result = ex.complete(1, "feature")
        finally:
            graph.close()

        assert result.rebased is True
        # The feature branch HEAD now carries the worker's source change.
        feature_head = _git(project, "show", "HEAD:feature.py")
        assert "def added()" in feature_head
        # And it is exactly one new commit (squashed), not the worker's raw "wip" commit.
        subject = _git(project, "log", "-1", "--format=%s").strip()
        assert subject.startswith("feat(milknado-1):")

    def test_scaffolding_absent_from_feature_commit(self, project: Path, tmp_path: Path) -> None:
        """RALPH.md / .ralph-logs/ must never be staged into the merged-back commit."""
        git = GitAdapter(project)
        branch = "milknado/1-added"
        wt = project / "milknado-1-added"
        _ = git.create_worktree(wt, branch)
        _EXCLUDE_LOOP_SCAFFOLDING(wt)
        _worker_did_work(wt)

        graph = MikadoGraph(tmp_path / "g.db")
        try:
            _ = _running_node(graph, wt, branch)
            ex = Executor(
                graph=graph, git=git, ralph=_loop_port(_NoRalph()), crg=_crg_port(_NoCrg())
            )
            _ = ex.complete(1, "feature")
        finally:
            graph.close()

        tracked = _ = _git(project, "ls-tree", "-r", "--name-only", "HEAD").splitlines()
        assert "RALPH.md" not in tracked
        assert not any(p.startswith(".ralph-logs") for p in tracked)
        assert "feature.py" in tracked

    def test_run_logs_preserved_under_milknado(self, project: Path, tmp_path: Path) -> None:
        """The run's .ralph-logs/ survives worktree removal at .milknado/logs/<id>/."""
        git = GitAdapter(project)
        branch = "milknado/1-added"
        wt = project / "milknado-1-added"
        _ = git.create_worktree(wt, branch)
        _EXCLUDE_LOOP_SCAFFOLDING(wt)
        _worker_did_work(wt)

        graph = MikadoGraph(tmp_path / "g.db")
        try:
            _ = _running_node(graph, wt, branch)
            ex = Executor(
                graph=graph, git=git, ralph=_loop_port(_NoRalph()), crg=_crg_port(_NoCrg())
            )
            _ = ex.complete(1, "feature")
        finally:
            graph.close()

        assert not wt.exists()  # worktree was removed
        preserved = project / ".milknado" / "logs" / "1" / "run.log"
        assert preserved.is_file()
        assert "iteration 1 log line" in preserved.read_text()

    def test_fast_forward_failure_returns_failed_domain_outcome(
        self, project: Path, tmp_path: Path
    ) -> None:
        git = GitAdapter(project)
        branch = "milknado/1-added"
        wt = project / "milknado-1-added"
        _ = git.create_worktree(wt, branch)
        _EXCLUDE_LOOP_SCAFFOLDING(wt)
        _worker_did_work(wt)
        (wt / "addons").symlink_to("worker-addons")
        _ = _git(wt, "add", "addons")
        _ = _git(wt, "commit", "-qm", "add addons symlink")
        (project / "addons").symlink_to("local-addons")

        graph = MikadoGraph(tmp_path / "g.db")
        try:
            _ = graph.add_node("Add the added() helper")
            graph.mark_running(1, worktree_path=str(wt), branch_name=branch, run_id="run-1")
            graph.start_run("run-1", 1, "run.log", "2026-01-01T00:00:00+00:00", 300)
            ex = Executor(
                graph=graph, git=git, ralph=_loop_port(_NoRalph()), crg=_crg_port(_NoCrg())
            )
            _worker_run_ids(ex)[1] = "run-1"

            result = ex.complete(1, "feature")
            row = graph.get_run("run-1")
            assert result.rebased is False
            assert result.rebase_conflict is not None
            assert "untracked integration-checkout path collision: addons" in (
                result.rebase_conflict.detail
            )
            assert row is not None
            assert row["status"] == "failed"
            assert row["error"] == result.rebase_conflict.detail
            assert row["rebased"] == 0
            node = graph.get_node(1)
            assert node is not None
            assert node.status.value == "failed"
            assert node.worktree_path == str(wt)
            assert node.branch_name == branch

            with pytest.raises(ValueError, match="retry merge-back before redispatching"):
                _ = ex.dispatch(
                    1,
                    ExecutionConfig(
                        execution_agent="claude",
                        quality_gates=None,
                        worktree_pattern="milknado-{node_id}",
                        project_root=project,
                    ),
                )
        finally:
            graph.close()

        assert wt.exists(), "unlanded worktree must be preserved on merge-back failure"
        assert _git(wt, "rev-parse", "--abbrev-ref", "HEAD").strip() == branch

        (project / "addons").unlink()
        retry = WorktreeManager(_git_port(git)).rebase_and_merge(
            wt, "feature", 1, "Add the added() helper", branch
        )
        assert retry.success is True
        assert _git(project, "show", "HEAD:addons").strip() == "worker-addons"


class TestGitAdapterContract:
    """The merge-back relies on two adapter contract changes: squash_and_commit must
    report whether it committed, and fast_forward must advance the feature branch."""

    def test_squash_and_commit_returns_true_when_it_commits(self, project: Path) -> None:
        git = GitAdapter(project)
        branch = "milknado/1-x"
        wt = project / "milknado-1-x"
        _ = git.create_worktree(wt, branch)
        _ = (wt / "f.py").write_text("x = 1\n")
        _ = _git(wt, "add", "f.py")
        _ = _git(wt, "commit", "-qm", "wip")
        assert git.squash_and_commit(wt, "feature", "feat: squashed") is True

    def test_squash_and_commit_returns_false_when_nothing_to_commit(self, project: Path) -> None:
        git = GitAdapter(project)
        branch = "milknado/1-x"
        wt = project / "milknado-1-x"
        _ = git.create_worktree(wt, branch)  # worktree identical to feature — no work
        assert git.squash_and_commit(wt, "feature", "feat: squashed") is False

    def test_fast_forward_advances_feature_branch(self, project: Path) -> None:
        git = GitAdapter(project)
        branch = "milknado/1-x"
        wt = project / "milknado-1-x"
        _ = git.create_worktree(wt, branch)
        _ = (wt / "f.py").write_text("x = 1\n")
        _ = _git(wt, "add", "f.py")
        _ = _git(wt, "commit", "-qm", "worker commit")
        before = _git(project, "rev-parse", "HEAD").strip()
        git.fast_forward(branch)
        after = _git(project, "rev-parse", "HEAD").strip()
        assert after != before
        assert "x = 1" in _git(project, "show", "HEAD:f.py")

    def test_fast_forward_raises_on_non_ff(self, project: Path) -> None:
        git = GitAdapter(project)
        branch = "milknado/1-x"
        wt = project / "milknado-1-x"
        _ = git.create_worktree(wt, branch)
        _ = (wt / "f.py").write_text("x = 1\n")
        _ = _git(wt, "add", "f.py")
        _ = _git(wt, "commit", "-qm", "worker commit")
        # A divergent commit on feature makes the worker branch non-fast-forwardable.
        _ = (project / "g.py").write_text("y = 2\n")
        _ = _git(project, "add", "g.py")
        _ = _git(project, "commit", "-qm", "feature moves on")
        with pytest.raises(GitOperationError, match="merge --ff-only"):
            git.fast_forward(branch)


class TestExcludeLoopScaffolding:
    def test_appends_patterns_to_common_dir_exclude(self, project: Path) -> None:
        git = GitAdapter(project)
        wt = project / "milknado-1-x"
        _ = git.create_worktree(wt, "milknado/1-x")
        _EXCLUDE_LOOP_SCAFFOLDING(wt)

        common = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=str(wt),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        exclude = (Path(common) / "info" / "exclude").read_text()
        assert "RALPH.md" in exclude
        assert ".ralph-logs/" in exclude

    def test_idempotent_no_duplicate_lines(self, project: Path) -> None:
        git = GitAdapter(project)
        wt = project / "milknado-1-x"
        _ = git.create_worktree(wt, "milknado/1-x")
        _EXCLUDE_LOOP_SCAFFOLDING(wt)
        _EXCLUDE_LOOP_SCAFFOLDING(wt)

        common = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=str(wt),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        lines = (Path(common) / "info" / "exclude").read_text().splitlines()
        assert lines.count("RALPH.md") == 1
        assert lines.count(".ralph-logs/") == 1

    def test_non_git_path_is_silent_noop(self, tmp_path: Path) -> None:
        """A test double's bare path is not a git checkout — must not raise."""
        bogus = tmp_path / "not-a-worktree"
        bogus.mkdir()
        _EXCLUDE_LOOP_SCAFFOLDING(bogus)  # no exception

    def test_exclude_file_io_error_is_logged_not_raised(self, project: Path) -> None:
        """A real git checkout whose `info/exclude` cannot be written (here: it is a
        directory, so read_text/write_text raise OSError) must log and return, not
        abort dispatch — the function's documented best-effort contract."""
        git = GitAdapter(project)
        wt = project / "milknado-1-x"
        _ = git.create_worktree(wt, "milknado/1-x")
        common = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=str(wt),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        exclude = Path(common) / "info" / "exclude"
        if exclude.exists():
            exclude.unlink()
        exclude.mkdir()  # now read_text/write_text on a dir path raise OSError
        _EXCLUDE_LOOP_SCAFFOLDING(wt)  # must not raise


class TestPreserveRunLogs:
    def test_no_logs_is_noop(self, tmp_path: Path) -> None:
        wt = tmp_path / "proj" / "wt"
        wt.mkdir(parents=True)
        _PRESERVE_RUN_LOGS(wt, 7)
        assert not (tmp_path / "proj" / ".milknado").exists()

    def test_copies_logs_tree(self, tmp_path: Path) -> None:
        wt = tmp_path / "proj" / "wt"
        (wt / ".ralph-logs" / "sub").mkdir(parents=True)
        _ = (wt / ".ralph-logs" / "a.log").write_text("a\n")
        _ = (wt / ".ralph-logs" / "sub" / "b.log").write_text("b\n")
        _PRESERVE_RUN_LOGS(wt, 7)
        dest = tmp_path / "proj" / ".milknado" / "logs" / "7"
        assert (dest / "a.log").read_text() == "a\n"
        assert (dest / "sub" / "b.log").read_text() == "b\n"

    def test_retains_only_twenty_newest_log_files(self, tmp_path: Path) -> None:
        wt = tmp_path / "proj" / "wt"
        logs = wt / ".ralph-logs"
        logs.mkdir(parents=True)
        for index in range(22):
            path = logs / f"{index:02}.log"
            _ = path.write_text(f"{index}\n")
            os.utime(path, (index, index))

        _PRESERVE_RUN_LOGS(wt, 7)

        dest = tmp_path / "proj" / ".milknado" / "logs" / "7"
        assert {path.name for path in dest.iterdir()} == {
            f"{index:02}.log" for index in range(2, 22)
        }

    def test_nested_worktree_preserves_under_checkout_root(self, project: Path) -> None:
        """When the worktree is nested in a subdirectory under the project root
        (a legal `worktree_pattern`), logs must land under the main checkout root,
        not `worktree.parent`. Resolving via the common git dir is what makes this
        hold — `worktree.parent` here is `project/sub`, the wrong place."""
        git = GitAdapter(project)
        sub = project / "sub"
        sub.mkdir()
        wt = sub / "wt-1"
        _ = git.create_worktree(wt, "milknado/1-x")
        (wt / ".ralph-logs").mkdir()
        _ = (wt / ".ralph-logs" / "run.log").write_text("nested log\n")
        _PRESERVE_RUN_LOGS(wt, 1)
        # Correct: under the main checkout root.
        preserved = project / ".milknado" / "logs" / "1" / "run.log"
        assert preserved.read_text() == "nested log\n"
        # And NOT under worktree.parent (the old buggy location).
        assert not (sub / ".milknado").exists()


class TestRebaseAndMergeSkipsFastForwardWithoutCommit:
    def test_no_commit_skips_fast_forward(self, tmp_path: Path) -> None:
        """When squash_and_commit reports no commit, ff is skipped — there is nothing
        to merge back. The no-commit case itself is owned by a sibling verifier."""
        wt = tmp_path / "wt"
        wt.mkdir()
        git = _RecordingGit(committed=False)
        wm = WorktreeManager(_git_port(git))
        result = wm.rebase_and_merge(wt, "feature", 1, "desc", worker_branch="milknado/1-x")
        assert result.success is True
        assert git.fast_forwarded == []  # ff skipped: no commit produced

    def test_commit_triggers_fast_forward(self, tmp_path: Path) -> None:
        wt = tmp_path / "wt"
        wt.mkdir()
        git = _RecordingGit(committed=True)
        wm = WorktreeManager(_git_port(git))
        _ = wm.rebase_and_merge(wt, "feature", 1, "desc", worker_branch="milknado/1-x")
        assert git.fast_forwarded == ["milknado/1-x"]

    def test_missing_worker_branch_skips_fast_forward(self, tmp_path: Path) -> None:
        """A node with no branch_name skips fast-forward instead of using None."""
        wt = tmp_path / "wt"
        wt.mkdir()
        git = _RecordingGit(committed=True)
        wm = WorktreeManager(_git_port(git))
        _ = wm.rebase_and_merge(wt, "feature", 1, "desc", worker_branch=None)
        assert git.fast_forwarded == []

    def test_worktree_kept_when_fast_forward_raises(self, tmp_path: Path) -> None:
        wt = tmp_path / "wt"
        wt.mkdir()
        git = _RecordingGit(
            committed=True, ff_error=GitOperationError("merge --ff-only", "diverged")
        )
        wm = WorktreeManager(_git_port(git))
        with pytest.raises(GitOperationError, match="merge --ff-only"):
            _ = wm.rebase_and_merge(wt, "feature", 1, "desc", worker_branch="milknado/1-x")
        assert git.removed == [], "unlanded work must not be torn down"
        assert wt.exists()

    def test_success_path_removes_with_feature_branch_target(self, tmp_path: Path) -> None:
        """Removal checks the feature branch that received the landed work."""
        wt = tmp_path / "wt"
        wt.mkdir()
        git = _RecordingGit(committed=True)
        wm = WorktreeManager(_git_port(git))
        _ = wm.rebase_and_merge(wt, "feature", 1, "desc", worker_branch="milknado/1-x")
        assert git.removed == [wt]
        assert git.remove_targets == ["feature"]


class TestDispatchRelocationIntegration:
    """Relocation against real git: the preserved orphan still has the node's
    canonical branch checked out, so reusing the branch name (not just the
    path) would make `git worktree add -b` fail — the suffix must cover both."""

    def test_dirty_orphan_relocates_dispatch_with_real_git(
        self, project: Path, tmp_path: Path
    ) -> None:
        from milknado.domains.common.config import Gate
        from milknado.domains.execution.executor import ExecutionConfig

        git = GitAdapter(project)
        orphan = project / "milknado-1-add-the-added-helper"
        _ = git.create_worktree(orphan, "milknado/1-add-the-added-helper")
        _ = (orphan / "wip.py").write_text("at-risk work\n")

        graph = MikadoGraph(tmp_path / "g.db")
        try:
            _ = graph.add_node("Add the added() helper")
            ex = Executor(
                graph=graph,
                git=git,
                ralph=_loop_port(_DispatchRalph()),
                crg=_crg_port(_NoCrg()),
            )
            config = ExecutionConfig(
                execution_agent="claude",
                quality_gates=(Gate(command="true"),),
                worktree_pattern="milknado-{node_id}-{slug}",
                project_root=project,
            )
            result = ex.dispatch(1, config)
        finally:
            graph.close()

        assert result.worktree == project / "milknado-1-add-the-added-helper-2"
        assert result.worktree.is_dir(), "relocated worktree must really exist"
        branches = _git(project, "branch", "--list", "milknado/1-add-the-added-helper-2")
        assert "milknado/1-add-the-added-helper-2" in branches
        assert (orphan / "wip.py").read_text() == "at-risk work\n", "orphan preserved"

    def test_relocation_skips_slot_whose_branch_survives(
        self, project: Path, tmp_path: Path
    ) -> None:
        """Skip a relocation slot when its branch remains checked out."""
        from milknado.domains.common.config import Gate
        from milknado.domains.execution.executor import ExecutionConfig

        git = GitAdapter(project)
        orphan = project / "milknado-1-add-the-added-helper"
        _ = git.create_worktree(orphan, "milknado/1-add-the-added-helper")
        _ = (orphan / "wip.py").write_text("at-risk work\n")
        _ = _git(project, "branch", "milknado/1-add-the-added-helper-2")

        graph = MikadoGraph(tmp_path / "g.db")
        try:
            _ = graph.add_node("Add the added() helper")
            ex = Executor(
                graph=graph,
                git=git,
                ralph=_loop_port(_DispatchRalph()),
                crg=_crg_port(_NoCrg()),
            )
            config = ExecutionConfig(
                execution_agent="claude",
                quality_gates=(Gate(command="true"),),
                worktree_pattern="milknado-{node_id}-{slug}",
                project_root=project,
            )
            result = ex.dispatch(1, config)
        finally:
            graph.close()

        assert result.worktree == project / "milknado-1-add-the-added-helper-3"
        assert result.worktree.is_dir(), "relocated worktree must really exist"
        head_branch = _git(result.worktree, "rev-parse", "--abbrev-ref", "HEAD").strip()
        assert head_branch == "milknado/1-add-the-added-helper-3"


# --- minimal duck-typed doubles (the integration tests use a real GitAdapter) ---


@dataclass(frozen=True)
class _RunState:
    run_id: str


@dataclass(frozen=True)
class _Run:
    state: _RunState


class _DispatchRalph:
    """Just enough LoopPort for Executor.dispatch to run against real git."""

    def generate_ralph_md(
        self,
        brief: str,
        quality_gates: tuple[object, ...] | None,
        output_path: Path,
        prior_findings: str = "",
        findings_round: int | None = None,
    ) -> Path:
        _ = (brief, quality_gates, prior_findings, findings_round)
        _ = output_path.write_text("# ralph\n")
        return output_path

    def create_run(
        self,
        agent: str,
        ralph_dir: Path,
        ralph_file: Path,
        quality_gates: tuple[object, ...] | None,
        project_root: Path | None = None,
        commit_footer: str | None = None,
        base_oid: str | None = None,
        runtime_policy: object | None = None,
        run_id: str | None = None,
        completion_probe: Callable[[], bool] | None = None,
    ) -> _Run:
        _ = (
            agent,
            ralph_dir,
            ralph_file,
            quality_gates,
            project_root,
            commit_footer,
            base_oid,
            runtime_policy,
            completion_probe,
        )
        return _Run(_RunState(run_id or "run-1"))

    def start_run(self, run_id: str) -> None:
        _ = run_id

    def get_run(self, run_id: str) -> _Run | None:
        _ = run_id
        return None

    def is_run_alive(self, run_id: str) -> bool:
        _ = run_id
        return False


class _RecordingGit:
    _committed: bool
    _ff_error: BaseException | None

    def __init__(
        self,
        committed: bool,
        ff_error: BaseException | None = None,
    ) -> None:
        self._committed = committed
        self._ff_error = ff_error
        self.fast_forwarded: list[str] = []
        self.removed: list[Path] = []
        self.remove_targets: list[str] = []

    def current_branch(self) -> str:
        return "feature"

    def squash_and_commit(self, worktree: Path, onto: str, msg: str) -> bool:
        _ = (worktree, onto, msg)
        return self._committed

    def rebase(self, worktree: Path, onto: str) -> RebaseResult:
        _ = (worktree, onto)
        return RebaseResult(success=True)

    def fast_forward(self, branch: str) -> None:
        _ = branch
        if self._ff_error is not None:
            raise self._ff_error
        self.fast_forwarded.append(branch)

    def untracked_merge_collisions(self, worktree: Path) -> tuple[str, ...]:
        _ = worktree
        return ()

    def remove_worktree(self, path: Path, target: str = "HEAD") -> None:
        self.removed.append(path)
        self.remove_targets.append(target)

    def force_remove_worktree(self, path: Path) -> None:
        self.removed.append(path)


class _NoRalph:
    """Executor.complete never touches the ralph port; present only to satisfy ctor."""


class _NoCrg:
    def __getattr__(self, _name: str):
        raise AssertionError("crg must not be used during complete()")
