import logging
import subprocess
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from milknado.domains.common.config import Gate
from milknado.domains.common.errors import (
    GitOperationError,
    InvalidTransition,
    RebaseAbortError,
    TransientDispatchError,
    UnlandedWorkError,
)
from milknado.domains.common.types import (
    NodeStatus,
    RebaseResult,
)
from milknado.domains.dispatch._runstate import RUN_ID_RE
from milknado.domains.execution import (
    CompletionResult,
    DispatchResult,
    ExecutionConfig,
    Executor,
    WorktreeManager,
    get_dispatchable_nodes,
)
from milknado.domains.graph import MikadoGraph
from milknado.loop import RunStatus


@dataclass
class FakeRunState:
    run_id: str = "run-1"
    status: RunStatus = RunStatus.RUNNING


def _dead_thread() -> threading.Thread:
    """An already-finished thread: a run whose work is done."""
    thread = threading.Thread(target=lambda: None, daemon=True)
    thread.start()
    thread.join()
    return thread


@dataclass
class FakeRun:
    state: FakeRunState = field(default_factory=FakeRunState)
    # Default: the run's thread has already finished, so the executor's cancel
    # watcher takes its dead-thread exit and no suite dispatch leaks a polling
    # daemon thread. Tests exercising the live-cancel path use make_live().
    thread: threading.Thread = field(default_factory=_dead_thread)


class FakeGit:
    def __init__(self) -> None:
        self.created: list[tuple[Path, str]] = []
        self.removed: list[Path] = []
        self.force_removed: list[Path] = []
        self.remove_targets: list[str] = []
        self.commits: list[tuple[Path, str]] = []
        self.rebases: list[tuple[Path, str]] = []
        self.fast_forwards: list[str] = []
        self.rebase_result: RebaseResult = RebaseResult(success=True)

    def branch_exists(self, branch: str) -> bool:
        return False

    def create_worktree(self, path: Path, branch: str) -> Path:
        self.created.append((path, branch))
        return path

    def remove_worktree(self, path: Path, target: str = "HEAD") -> None:
        self.removed.append(path)
        self.remove_targets.append(target)

    def force_remove_worktree(self, path: Path) -> None:
        self.force_removed.append(path)
        self.removed.append(path)

    def rebase(self, worktree: Path, onto: str) -> RebaseResult:
        self.rebases.append((worktree, onto))
        return self.rebase_result

    def current_branch(self) -> str:
        return "main"

    def resolve_ref(self, ref: str) -> str:
        return f"{ref}-oid"

    def diff_for_review(self, worktree: Path, base_oid: str) -> str:
        return f"diff from {base_oid}"

    def commit_all(self, worktree: Path, message: str) -> None:
        self.commits.append((worktree, message))

    def squash_and_commit(self, worktree: Path, onto: str, msg: str) -> None:
        self.commits.append((worktree, msg))

    def fast_forward(self, branch: str) -> None:
        self.fast_forwards.append(branch)


class FakeRalph:
    def __init__(self, *, live: bool = False, id_prefix: str | None = None) -> None:
        # live=True: created runs get a live thread from the start, so the
        # cancel watcher stays on its poll loop until a stop kills it —
        # required to exercise the sentinel path without racing the watcher's
        # first dead-thread check. id_prefix namespaces run ids so a test's
        # watcher thread name never collides with another test's; the default
        # is per-instance unique so namespacing is not opt-in.
        self._live = live
        self._id_prefix = id_prefix or f"run-{uuid4().hex[:8]}"
        self.runs_created: list[dict[str, Any]] = []
        self.runs_started: list[str] = []
        self.generated: list[Path] = []
        self.generated_briefs: list[str] = []
        self.runs: dict[str, FakeRun] = {}
        self.stopped: list[str] = []
        self.force_stopped: list[str] = []
        # Stop-outcome configurability mirrors the real stop_and_join /
        # force_stop_and_join, which return False on join timeout and can
        # raise — required to pin the unconfirmed-stop (fail-closed) branches.
        self.stop_result = True
        self.stop_raises: Exception | None = None
        self.force_stop_result = True
        self.force_stop_raises: Exception | None = None
        self._stop_events: dict[str, threading.Event] = {}

    def make_live(self, run_id: str) -> None:
        """Give a run a live thread so the cancel watcher stays on its poll
        loop until a stop kills it — required to exercise the sentinel path."""
        stop = threading.Event()
        thread = threading.Thread(target=stop.wait, daemon=True)
        self._stop_events[run_id] = stop
        thread.start()
        self.runs[run_id].thread = thread

    def _kill(self, run_id: str) -> None:
        stop = self._stop_events.get(run_id)
        if stop is not None:
            stop.set()

    def create_run(
        self,
        agent: str,
        ralph_dir: Path,
        ralph_file: Path,
        commands: list[str],
        quality_gates: tuple[Gate, ...] | None,
        project_root: Path | None = None,
        commit_footer: str | None = None,
        base_oid: str | None = None,
        runtime_policy: Any | None = None,
        run_id: str | None = None,
    ) -> FakeRun:
        self.runs_created.append(
            {
                "agent": agent,
                "dir": ralph_dir,
                "commit_footer": commit_footer,
                "base_oid": base_oid,
            }
        )
        resolved_run_id = run_id or f"{self._id_prefix}-{len(self.runs_created)}"
        run = FakeRun(state=FakeRunState(run_id=resolved_run_id))
        self.runs[run.state.run_id] = run
        if self._live:
            self.make_live(run.state.run_id)
        return run

    def start_run(self, run_id: str) -> None:
        self.runs_started.append(run_id)

    def stop_run(self, run_id: str, timeout: float | None = None) -> bool:
        self.stopped.append(run_id)
        if self.stop_raises is not None:
            raise self.stop_raises
        if self.stop_result:
            self._kill(run_id)
        return self.stop_result

    def force_stop_run(self, run_id: str, timeout: float | None = None) -> bool:
        self.force_stopped.append(run_id)
        if self.force_stop_raises is not None:
            raise self.force_stop_raises
        if self.force_stop_result:
            self._kill(run_id)
        return self.force_stop_result

    def list_runs(self) -> list[Any]:
        return []

    def get_run(self, run_id: str) -> Any | None:
        return self.runs.get(run_id)

    def get_run_stdout(self, run_id: str) -> list[str]:
        return []

    def wait_for_next_completion(
        self,
        active_run_ids: set[str],
        timeout: float | None = None,
    ) -> tuple[str, bool]:
        raise RuntimeError("Not expected in executor tests")

    def poll_progress_events(self) -> list[Any]:
        return []

    def verify_spec(self, spec_text: str, graph_state: str) -> Any:
        from milknado.domains.common.protocols import VerifySpecResult

        return VerifySpecResult(outcome="done")

    def generate_ralph_md(
        self,
        brief: str,
        quality_gates: tuple[Gate, ...] | None,
        output_path: Path,
        prior_findings: str = "",
        findings_round: int | None = None,
    ) -> Path:
        self.generated.append(output_path)
        self.generated_briefs.append(brief)
        return output_path


class FakeCrg:
    def ensure_graph(self, project_root: Path) -> None:
        pass

    def get_impact_radius(self, files: list[str]) -> dict[str, Any]:
        return {"files": files}

    def get_architecture_overview(self) -> dict[str, Any]:
        return {"modules": []}

    def list_communities(self, sort_by: str = "size", min_size: int = 0) -> list[dict[str, Any]]:
        return []

    def list_flows(self, sort_by: str = "criticality", limit: int = 50) -> list[dict[str, Any]]:
        return []

    def get_bridge_nodes(self, top_n: int = 10) -> list[dict[str, Any]]:
        return []

    def get_hub_nodes(self, top_n: int = 10) -> list[dict[str, Any]]:
        return []


@pytest.fixture()
def config(tmp_path: Path) -> ExecutionConfig:
    return ExecutionConfig(
        execution_agent="claude",
        quality_gates=(Gate("uv run pytest"),),
        worktree_pattern="milknado-{node_id}-{slug}",
        project_root=tmp_path,
    )


@pytest.fixture()
def executor(graph: MikadoGraph) -> Executor:
    # Explicit prefix: these tests assert deterministic run ids ("run-1");
    # the default prefix is per-instance unique.
    return Executor(
        graph=graph,
        git=FakeGit(),
        ralph=FakeRalph(id_prefix="run"),
        crg=FakeCrg(),
    )


class TestGetDispatchableNodes:
    def test_empty_graph(self, graph: MikadoGraph) -> None:
        assert get_dispatchable_nodes(graph) == []

    def test_single_pending_leaf(self, graph: MikadoGraph) -> None:
        root = graph.add_node("root")
        leaf = graph.add_node("leaf", parent_id=root.id)
        result = get_dispatchable_nodes(graph)
        assert result == [leaf.id]
        assert root.id not in result

    def test_excludes_running_nodes(self, graph: MikadoGraph) -> None:
        graph.add_node("root")
        graph.mark_running(1)
        assert get_dispatchable_nodes(graph) == []

    def test_excludes_done_nodes(self, graph: MikadoGraph) -> None:
        graph.add_node("root")
        graph.mark_running(1)
        graph.mark_done(1)
        assert get_dispatchable_nodes(graph) == []

    def test_parallel_leaves_no_conflict(self, graph: MikadoGraph) -> None:
        root = graph.add_node("root")
        graph.add_node("leaf-a", parent_id=root.id)
        graph.add_node("leaf-b", parent_id=root.id)
        dispatchable = get_dispatchable_nodes(graph)
        assert 2 in dispatchable
        assert 3 in dispatchable

    def test_filters_conflicting_leaves(self, graph: MikadoGraph) -> None:
        root = graph.add_node("root")
        a = graph.add_node("leaf-a", parent_id=root.id)
        b = graph.add_node("leaf-b", parent_id=root.id)
        graph.set_file_ownership(a.id, ["shared.py"])
        graph.set_file_ownership(b.id, ["shared.py"])
        dispatchable = get_dispatchable_nodes(graph)
        assert len(dispatchable) == 1
        assert dispatchable[0] == a.id

    def test_active_owner_blocks_pending_conflict(self, graph: MikadoGraph) -> None:
        root = graph.add_node("root")
        active = graph.add_node("active", parent_id=root.id)
        pending = graph.add_node("pending", parent_id=root.id)
        graph.set_file_ownership(active.id, ["shared.py"])
        graph.set_file_ownership(pending.id, ["shared.py"])
        graph.mark_running(active.id)

        assert get_dispatchable_nodes(graph) == []

    def test_child_must_complete_before_parent(self, graph: MikadoGraph) -> None:
        root = graph.add_node("root")
        child = graph.add_node("child", parent_id=root.id)
        dispatchable = get_dispatchable_nodes(graph)
        assert child.id in dispatchable
        assert root.id not in dispatchable


class TestExecutorDispatch:
    def test_creates_worktree_and_starts_run(
        self,
        executor: Executor,
        graph: MikadoGraph,
        config: ExecutionConfig,
    ) -> None:
        graph.add_node("extract interface")
        result = executor.dispatch(1, config)

        assert isinstance(result, DispatchResult)
        assert result.node_id == 1
        assert RUN_ID_RE.match(result.run_id), "the ralph run_id matches the executor's id format"
        assert "extract-interface" in str(result.worktree)

    def test_marks_node_running(
        self,
        executor: Executor,
        graph: MikadoGraph,
        config: ExecutionConfig,
    ) -> None:
        graph.add_node("do work")
        executor.dispatch(1, config)
        node = graph.get_node(1)
        assert node is not None
        assert node.status == NodeStatus.RUNNING

    def test_stores_worktree_path(
        self,
        executor: Executor,
        graph: MikadoGraph,
        config: ExecutionConfig,
    ) -> None:
        graph.add_node("task")
        executor.dispatch(1, config)
        node = graph.get_node(1)
        assert node is not None
        assert node.worktree_path is not None
        assert "milknado-1-" in node.worktree_path

    def test_stores_branch_name(
        self,
        executor: Executor,
        graph: MikadoGraph,
        config: ExecutionConfig,
    ) -> None:
        graph.add_node("task")
        executor.dispatch(1, config)
        node = graph.get_node(1)
        assert node is not None
        assert node.branch_name is not None
        assert node.branch_name.startswith("milknado/1-")

    def test_commit_footer_threaded_to_create_run(
        self,
        graph: MikadoGraph,
        tmp_path: Path,
    ) -> None:
        """A set commit_footer on ExecutionConfig reaches the ralph port's create_run call."""
        footer_config = ExecutionConfig(
            execution_agent="claude",
            quality_gates=(Gate("uv run pytest"),),
            worktree_pattern="milknado-{node_id}-{slug}",
            project_root=tmp_path,
            commit_footer="Co-authored-by: Team <team@example.com>",
        )
        fake_ralph = FakeRalph()
        executor = Executor(graph=graph, git=FakeGit(), ralph=fake_ralph, crg=FakeCrg())
        graph.add_node("task")

        executor.dispatch(1, footer_config)

        footer = "Co-authored-by: Team <team@example.com>"
        assert fake_ralph.runs_created[0]["commit_footer"] == footer

    def test_generates_ralph_md(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
    ) -> None:
        fake_ralph = FakeRalph()
        ex = Executor(
            graph=graph,
            git=FakeGit(),
            ralph=fake_ralph,
            crg=FakeCrg(),
        )
        graph.add_node("build feature")
        ex.dispatch(1, replace(config, brief_prepend="Use the project gate."))

        assert len(fake_ralph.generated) == 1
        assert fake_ralph.generated[0].name == "RALPH.md"
        assert fake_ralph.generated_briefs[0].startswith("Use the project gate.")
        assert "# Task: build feature" in fake_ralph.generated_briefs[0]

    def test_dispatch_nonexistent_node_raises(
        self,
        executor: Executor,
        config: ExecutionConfig,
    ) -> None:
        with pytest.raises(ValueError, match="not found"):
            executor.dispatch(999, config)

    def test_creates_git_worktree(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
    ) -> None:
        fake_git = FakeGit()
        ex = Executor(
            graph=graph,
            git=fake_git,
            ralph=FakeRalph(),
            crg=FakeCrg(),
        )
        graph.add_node("refactor auth")
        ex.dispatch(1, config)
        assert len(fake_git.created) == 1
        path, branch = fake_git.created[0]
        assert "milknado-1-" in str(path)
        assert branch.startswith("milknado/1-")

    def test_starts_ralph_run(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
    ) -> None:
        fake_ralph = FakeRalph(id_prefix="run")
        ex = Executor(
            graph=graph,
            git=FakeGit(),
            ralph=fake_ralph,
            crg=FakeCrg(),
        )
        graph.add_node("do it")
        result = ex.dispatch(1, config)
        assert fake_ralph.runs_started == [result.run_id]
        assert RUN_ID_RE.match(result.run_id)

    def test_stores_run_id_in_graph(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
    ) -> None:
        ex = Executor(
            graph=graph,
            git=FakeGit(),
            ralph=FakeRalph(id_prefix="run"),
            crg=FakeCrg(),
        )
        graph.add_node("track run")
        result = ex.dispatch(1, config)
        node = graph.get_node(1)
        assert node is not None
        assert node.run_id == result.run_id
        assert RUN_ID_RE.match(node.run_id)

    def test_adopts_an_existing_claim_without_remarking_running(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
    ) -> None:
        """When the dispatching parent (the ralph MCP tool) has already claimed the
        node RUNNING under its run_id, the detached runner's dispatch must NOT
        re-mark RUNNING (an illegal RUNNING -> RUNNING transition that would kill the
        run) and must NOT clobber the claim's run_id — that run_id is the fence the
        parent's set_pid and the completion guard depend on. It only attaches the
        worktree, gated on the fence."""
        from milknado.domains.dispatch._runstate import now_iso

        ex = Executor(graph=graph, git=FakeGit(), ralph=FakeRalph(id_prefix="run"), crg=FakeCrg())
        graph.add_node("claimed upstream")
        claim_run_id = "node-1-20260101T000000Z-claim"
        assert graph.claim_node(1, claim_run_id, now=now_iso()) is True

        result = ex.dispatch(1, config, parent_run_id=claim_run_id)

        node = graph.get_node(1)
        assert node is not None
        assert node.status == NodeStatus.RUNNING
        assert node.run_id == claim_run_id, "the parent's fence run_id is preserved"
        assert node.worktree_path is not None and "milknado-1-" in node.worktree_path
        assert RUN_ID_RE.match(result.run_id), (
            "the ralph run_id is returned for completion waiting"
        )

    def test_complete_rejects_a_zombie_terminal_write_after_reclaim_mid_merge(
        self,
        graph: MikadoGraph,
        tmp_path: Path,
    ) -> None:
        """The spec's 'main risk': complete snapshots the node (run_id=A) at entry,
        then does the slow rebase-merge. If a concurrent dispatch reclaims the node
        under a fresh run_id during that merge window, the post-merge terminal write
        is fenced on the stale run_id A and must hit zero rows — the fresh owner's
        RUNNING row is left intact and no completion duration is recorded for the
        zombie. Without the run_id fence, complete would clobber the fresh run to
        DONE."""
        from milknado.domains.dispatch._runstate import now_iso

        dead_pid = 2**31 - 1  # unreapable: os.kill(pid, 0) -> ProcessLookupError
        fake_git = FakeGit()
        ex = Executor(graph=graph, git=fake_git, ralph=FakeRalph(), crg=FakeCrg())
        graph.add_node("zombie-completer")
        wt = tmp_path / "worktree"
        wt.mkdir()
        assert graph.claim_node(1, "run-A", now=now_iso()) is True
        graph.set_pid(1, "run-A", dead_pid)
        graph.set_worktree(1, "run-A", str(wt), "milknado/1-x")

        def _reclaim_during_merge(worktree: Path, onto: str) -> RebaseResult:
            # A second dispatch lands while run-A is mid-merge: run-A is presumed
            # dead (its pid is gone), so it is reclaimed and a fresh run-B claims.
            graph.try_reclaim(1, now=now_iso())
            assert graph.claim_node(1, "run-B", now=now_iso()) is True
            return RebaseResult(success=True)

        fake_git.rebase = _reclaim_during_merge  # ty: ignore[invalid-assignment]

        result = ex.complete(1, "main")

        assert result.rebased is True, "the merge itself succeeded"
        node = graph.get_node(1)
        assert node is not None
        assert node.status == NodeStatus.RUNNING, "fresh owner (run-B) not clobbered to DONE"
        assert node.run_id == "run-B", "run-B still owns the node; the zombie write was rejected"
        assert node.completed_at is None, "no completion recorded for the reclaimed zombie run"

    def test_cleans_up_worktree_on_dispatch_failure(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
    ) -> None:
        fake_git = FakeGit()
        fake_ralph = FakeRalph()
        fake_ralph.create_run = lambda **_kw: (_ for _ in ()).throw(  # type: ignore
            RuntimeError("ralph exploded"),
        )
        ex = Executor(
            graph=graph,
            git=fake_git,
            ralph=fake_ralph,
            crg=FakeCrg(),
        )
        graph.add_node("doomed task")
        with pytest.raises(RuntimeError, match="ralph exploded"):
            ex.dispatch(1, config)
        assert len(fake_git.removed) == 1

    def test_resets_node_to_pending_on_dispatch_failure(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
    ) -> None:
        fake_ralph = FakeRalph()
        fake_ralph.create_run = lambda **_kw: (_ for _ in ()).throw(  # type: ignore
            RuntimeError("ralph exploded"),
        )
        ex = Executor(
            graph=graph,
            git=FakeGit(),
            ralph=fake_ralph,
            crg=FakeCrg(),
        )
        graph.add_node("doomed task")
        with pytest.raises(RuntimeError, match="ralph exploded"):
            ex.dispatch(1, config)
        node = graph.get_node(1)
        assert node is not None
        assert node.status == NodeStatus.PENDING
        assert node.worktree_path is None
        assert node.branch_name is None

    def test_worktree_cleanup_runs_even_when_mark_pending_raises(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
    ) -> None:
        """Orphan worktree must be removed even if mark_pending raises InvalidTransition."""
        from milknado.domains.common.errors import InvalidTransition

        fake_git = FakeGit()
        fake_ralph = FakeRalph()
        fake_ralph.create_run = lambda **_kw: (_ for _ in ()).throw(  # type: ignore
            RuntimeError("failure after running"),
        )

        ex = Executor(graph=graph, git=fake_git, ralph=fake_ralph, crg=FakeCrg())
        graph.add_node("doomed")

        # Monkey-patch mark_pending to raise after mark_running succeeds
        original_mark_pending = graph.mark_pending

        def mark_pending_raises(node_id):
            raise InvalidTransition(node_id, NodeStatus.RUNNING, NodeStatus.PENDING, ())

        graph.mark_pending = mark_pending_raises  # type: ignore

        with pytest.raises(RuntimeError, match="failure after running"):
            ex.dispatch(1, config)

        # Worktree cleanup must have run despite mark_pending raising
        assert len(fake_git.removed) == 1
        assert 1 not in ex._wt._worktrees

        graph.mark_pending = original_mark_pending  # ty: ignore[invalid-assignment]  # restore

    def test_path_traversal_in_worktree_pattern_raises(
        self,
        executor: Executor,
        graph: MikadoGraph,
        config: ExecutionConfig,
    ) -> None:
        """worktree_pattern that escapes project_root must be rejected before create_worktree."""
        traversal_config = ExecutionConfig(
            execution_agent=config.execution_agent,
            quality_gates=config.quality_gates,
            worktree_pattern="../../etc/{node_id}-{slug}",
            project_root=config.project_root,
        )
        graph.add_node("sneaky task")
        with pytest.raises(ValueError, match="resolves outside project_root"):
            executor.dispatch(1, traversal_config)

    def test_path_traversal_via_abs_pattern_raises(
        self,
        executor: Executor,
        graph: MikadoGraph,
        config: ExecutionConfig,
    ) -> None:
        """Absolute worktree_pattern that leaves project_root is also rejected."""
        traversal_config = ExecutionConfig(
            execution_agent=config.execution_agent,
            quality_gates=config.quality_gates,
            worktree_pattern="/tmp/evil-{node_id}-{slug}",
            project_root=config.project_root,
        )
        graph.add_node("sneaky task 2")
        with pytest.raises(ValueError, match="resolves outside project_root"):
            executor.dispatch(1, traversal_config)

    def test_normal_worktree_pattern_does_not_raise(
        self,
        executor: Executor,
        graph: MikadoGraph,
        config: ExecutionConfig,
    ) -> None:
        """Baseline: a normal pattern resolves within project_root without error."""
        graph.add_node("safe task")
        result = executor.dispatch(1, config)
        assert result.node_id == 1


class TestExecutorComplete:
    def test_merge_back_rejects_mismatched_checkout_before_git_mutation(
        self,
        tmp_path: Path,
    ) -> None:
        fake_git = FakeGit()
        worktree = tmp_path / "worktree"
        worktree.mkdir()
        target_before = fake_git.current_branch()

        with pytest.raises(GitOperationError, match="merge-back target"):
            WorktreeManager(fake_git).rebase_and_merge(
                worktree,
                "feature",
                node_id=1,
                description="task",
            )

        assert fake_git.current_branch() == target_before
        assert fake_git.commits == []
        assert fake_git.rebases == []
        assert fake_git.fast_forwards == []
        assert fake_git.removed == []

    def test_completion_rejects_dispatch_target_switch_before_merge_mutation(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
    ) -> None:
        fake_git = FakeGit()
        checked_out = ["main"]
        fake_git.current_branch = lambda: checked_out[0]  # type: ignore[method-assign]
        ex = Executor(
            graph=graph,
            git=fake_git,
            ralph=FakeRalph(),
            crg=FakeCrg(),
        )
        graph.add_node("task")
        ex.dispatch(1, config)
        checked_out[0] = "other"

        with pytest.raises(GitOperationError, match="dispatch-time merge target"):
            ex.complete(1, "other")

        assert fake_git.commits == []
        assert fake_git.rebases == []
        assert fake_git.fast_forwards == []
        assert fake_git.removed == []
        node = graph.get_node(1)
        assert node is not None
        assert node.status is NodeStatus.RUNNING

    def test_marks_done_on_success(
        self,
        graph: MikadoGraph,
    ) -> None:
        fake_git = FakeGit()
        ex = Executor(
            graph=graph,
            git=fake_git,
            ralph=FakeRalph(),
            crg=FakeCrg(),
        )
        graph.add_node("task")
        graph.mark_running(1)
        result = ex.complete(1, "main")

        assert isinstance(result, CompletionResult)
        assert result.rebased is True
        node = graph.get_node(1)
        assert node is not None
        assert node.status == NodeStatus.DONE

    def test_completing_already_done_node_is_idempotent(
        self,
        graph: MikadoGraph,
        tmp_path: Path,
    ) -> None:
        """A duplicate completion of an already-DONE node must be a true no-op.

        A second completion signal for the same run (or a re-run of a finished
        node) reaches complete() with the node already terminal. DONE has no
        valid transitions, so an unguarded mark_done would raise
        InvalidTransition. Beyond not raising, the duplicate must not re-run the
        squash/rebase/worktree-remove side effects: mark_done leaves
        worktree_path set, so a node whose prior worktree cleanup failed would
        otherwise have its rebase_and_merge workflow run a second time.
        """
        fake_git = FakeGit()
        ex = Executor(
            graph=graph,
            git=fake_git,
            ralph=FakeRalph(),
            crg=FakeCrg(),
        )
        graph.add_node("task")
        wt = tmp_path / "worktree"
        wt.mkdir()
        graph.mark_running(1, worktree_path=str(wt))
        ex.complete(1, "main")
        first_node = graph.get_node(1)
        assert first_node is not None
        assert first_node.status == NodeStatus.DONE
        assert len(fake_git.commits) == 1  # first completion squashed exactly once

        # Second completion of the same node. The worktree dir still exists
        # (FakeGit.remove only records the path) and worktree_path is still set,
        # so an unguarded re-run would squash again. It must instead be a no-op:
        # not a crash, and not a second squash/rebase/remove.
        result = ex.complete(1, "main")

        assert result.rebased is True
        node = graph.get_node(1)
        assert node is not None
        assert node.status == NodeStatus.DONE
        assert len(fake_git.commits) == 1  # duplicate completion did not re-squash

    def test_completing_a_failed_node_is_a_noop(
        self,
        graph: MikadoGraph,
    ) -> None:
        """A FAILED node reaching complete() again is a terminal duplicate: a true
        no-op that returns rebased=False without raising or re-running the
        squash/rebase workflow. FAILED -> DONE is not a valid transition, so an
        unguarded mark_done would raise InvalidTransition instead.
        """
        fake_git = FakeGit()
        ex = Executor(
            graph=graph,
            git=fake_git,
            ralph=FakeRalph(),
            crg=FakeCrg(),
        )
        graph.add_node("task")
        graph.mark_failed(1)

        result = ex.complete(1, "main")

        assert result.rebased is False  # terminal-but-not-DONE no-op
        assert len(fake_git.commits) == 0  # short-circuited before rebase_and_merge
        node = graph.get_node(1)
        assert node is not None
        assert node.status == NodeStatus.FAILED

    @pytest.mark.parametrize("status", [NodeStatus.PENDING, NodeStatus.BLOCKED])
    def test_completing_a_non_terminal_node_raises(
        self,
        graph: MikadoGraph,
        status: NodeStatus,
    ) -> None:
        """A non-RUNNING, non-terminal node (PENDING/BLOCKED) is never a valid
        completion target. complete() must fail loud with InvalidTransition rather
        than silently no-op: a silent no-op would hide a real state-machine bug and,
        in the run loop, print a false completion for a node that never finished.
        """
        fake_git = FakeGit()
        ex = Executor(
            graph=graph,
            git=fake_git,
            ralph=FakeRalph(),
            crg=FakeCrg(),
        )
        graph.add_node("task")
        if status is NodeStatus.BLOCKED:
            graph.mark_blocked(1)

        with pytest.raises(InvalidTransition):
            ex.complete(1, "main")

        assert len(fake_git.commits) == 0  # raised before any squash/rebase
        node = graph.get_node(1)
        assert node is not None
        assert node.status == status

    def test_marks_failed_on_rebase_conflict(
        self,
        graph: MikadoGraph,
        tmp_path: Path,
    ) -> None:
        fake_git = FakeGit()
        fake_git.rebase_result = RebaseResult(success=False)
        ex = Executor(
            graph=graph,
            git=fake_git,
            ralph=FakeRalph(),
            crg=FakeCrg(),
        )
        graph.add_node("task")
        wt = tmp_path / "worktree"
        wt.mkdir()
        graph.mark_running(1, worktree_path=str(wt))
        result = ex.complete(1, "main")

        assert result.rebased is False
        node = graph.get_node(1)
        assert node is not None
        assert node.status == NodeStatus.FAILED

    def test_removes_worktree_on_success(
        self,
        graph: MikadoGraph,
        tmp_path: Path,
    ) -> None:
        fake_git = FakeGit()
        ex = Executor(
            graph=graph,
            git=fake_git,
            ralph=FakeRalph(),
            crg=FakeCrg(),
        )
        graph.add_node("task")
        wt = tmp_path / "worktree"
        wt.mkdir()
        graph.mark_running(1, worktree_path=str(wt))
        ex.complete(1, "main")
        assert wt in fake_git.removed

    def test_removal_targets_feature_branch_on_success(
        self,
        graph: MikadoGraph,
        tmp_path: Path,
    ) -> None:
        """The landed check must run against the branch work landed on, not a
        default ref — a wrong target would refuse freshly-landed worktrees."""
        fake_git = FakeGit()
        ex = Executor(graph=graph, git=fake_git, ralph=FakeRalph(), crg=FakeCrg())
        graph.add_node("task")
        wt = tmp_path / "worktree"
        wt.mkdir()
        graph.mark_running(1, worktree_path=str(wt))
        ex.complete(1, "main")
        assert fake_git.remove_targets == ["main"]

    def test_keeps_worktree_on_rebase_failure(
        self,
        graph: MikadoGraph,
        tmp_path: Path,
    ) -> None:
        """Fail-closed teardown: a conflicting rebase means the squashed commit
        never landed — the worktree is the only copy of that work and must be
        preserved, not force-removed as it was before."""
        fake_git = FakeGit()
        fake_git.rebase_result = RebaseResult(success=False)
        ex = Executor(
            graph=graph,
            git=fake_git,
            ralph=FakeRalph(),
            crg=FakeCrg(),
        )
        graph.add_node("task")
        wt = tmp_path / "worktree"
        wt.mkdir()
        graph.mark_running(1, worktree_path=str(wt))
        result = ex.complete(1, "main")
        assert result.rebased is False
        assert fake_git.removed == []
        assert fake_git.force_removed == []

    def test_keeps_worktree_on_commit_failure(
        self,
        graph: MikadoGraph,
        tmp_path: Path,
    ) -> None:
        """An exception mid-merge-back leaves work state unknown — fail closed
        and keep the worktree rather than destroying the evidence."""
        fake_git = FakeGit()
        fake_git.squash_and_commit = lambda *_args: (_ for _ in ()).throw(  # type: ignore
            RuntimeError("nothing to commit"),
        )
        ex = Executor(
            graph=graph,
            git=fake_git,
            ralph=FakeRalph(),
            crg=FakeCrg(),
        )
        graph.add_node("task")
        wt = tmp_path / "worktree"
        wt.mkdir()
        graph.mark_running(1, worktree_path=str(wt))
        result = ex.complete(1, "main")
        assert fake_git.removed == []
        assert result.rebased is False

    def test_clears_metadata_on_rebase_failure(
        self,
        graph: MikadoGraph,
        tmp_path: Path,
    ) -> None:
        fake_git = FakeGit()
        fake_git.rebase_result = RebaseResult(success=False)
        ex = Executor(
            graph=graph,
            git=fake_git,
            ralph=FakeRalph(),
            crg=FakeCrg(),
        )
        graph.add_node("task")
        wt = tmp_path / "worktree"
        wt.mkdir()
        graph.mark_running(1, worktree_path=str(wt), branch_name="milknado/1-task")
        ex.complete(1, "main")
        node = graph.get_node(1)
        assert node is not None
        assert node.worktree_path is None
        assert node.branch_name is None

    def test_returns_newly_ready_nodes(
        self,
        graph: MikadoGraph,
    ) -> None:
        ex = Executor(
            graph=graph,
            git=FakeGit(),
            ralph=FakeRalph(),
            crg=FakeCrg(),
        )
        root = graph.add_node("root")
        child = graph.add_node("child", parent_id=root.id)
        grandchild = graph.add_node("grandchild", parent_id=child.id)
        graph.mark_running(grandchild.id)
        result = ex.complete(grandchild.id, "main")
        # child becomes ready once grandchild is done; root stays excluded.
        assert child.id in result.newly_ready
        assert root.id not in result.newly_ready

    def test_commit_message_includes_node_id_and_description(
        self,
        graph: MikadoGraph,
        tmp_path: Path,
    ) -> None:
        fake_git = FakeGit()
        ex = Executor(
            graph=graph,
            git=fake_git,
            ralph=FakeRalph(),
            crg=FakeCrg(),
        )
        graph.add_node("Add user authentication")
        wt = tmp_path / "worktree"
        wt.mkdir()
        graph.mark_running(1, worktree_path=str(wt))
        ex.complete(1, "main")

        assert len(fake_git.commits) == 1
        _, msg = fake_git.commits[0]
        assert msg.startswith("feat(milknado-1): Add user authentication")
        assert "Milknado-Node: 1" in msg

    def test_complete_nonexistent_raises(
        self,
        executor: Executor,
    ) -> None:
        with pytest.raises(ValueError, match="not found"):
            executor.complete(999, "main")

    def test_rebase_abort_error_propagates(
        self,
        graph: MikadoGraph,
        tmp_path: Path,
    ) -> None:
        """RebaseAbortError must propagate from complete() — do not swallow."""
        fake_git = FakeGit()
        abort_error = RebaseAbortError(tmp_path / "worktree", stderr="abort failed")
        fake_git.rebase = lambda *_args: (_ for _ in ()).throw(abort_error)  # type: ignore

        ex = Executor(graph=graph, git=fake_git, ralph=FakeRalph(), crg=FakeCrg())
        graph.add_node("task")
        wt = tmp_path / "worktree"
        wt.mkdir()
        graph.mark_running(1, worktree_path=str(wt))

        with pytest.raises(RebaseAbortError):
            ex.complete(1, "main")

    def test_complete_logs_detail_on_generic_exception(
        self,
        graph: MikadoGraph,
        tmp_path: Path,
    ) -> None:
        """Generic exceptions from rebase yield failed result with detail."""
        fake_git = FakeGit()
        fake_git.squash_and_commit = lambda *_args: (_ for _ in ()).throw(  # type: ignore
            RuntimeError("commit failed: nothing to commit"),
        )
        ex = Executor(graph=graph, git=fake_git, ralph=FakeRalph(), crg=FakeCrg())
        graph.add_node("task")
        wt = tmp_path / "worktree"
        wt.mkdir()
        graph.mark_running(1, worktree_path=str(wt))
        result = ex.complete(1, "main")
        assert result.rebased is False


class TestEnsureCleanWorktree:
    def test_orphan_removal_failure_is_logged_not_raised(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
    ) -> None:
        """WorktreeManager.ensure_clean swallows remove_worktree errors."""
        call_count = 0

        class BoomGit(FakeGit):
            def remove_worktree(self, path: Path) -> None:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise OSError("disk full")

        fake_git = BoomGit()
        ex = Executor(graph=graph, git=fake_git, ralph=FakeRalph(), crg=FakeCrg())
        graph.add_node("first")
        graph.add_node("second")

        # Dispatch first node so it gets into _worktrees with an existing path
        result1 = ex.dispatch(1, config)
        # Simulate the worktree path existing
        result1.worktree.mkdir(parents=True, exist_ok=True)

        # Second dispatch of same node triggers _ensure_clean_worktree which calls remove
        # To test the exception path, manually stash the worktree and call ensure
        ex._wt._worktrees[1] = result1.worktree
        # Should not raise even though remove_worktree raises
        ex._wt.ensure_clean(1)

    def test_refusal_degrades_and_keeps_worktree(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """ensure_clean is pre-dispatch cleanup, not teardown of finished work:
        a fail-closed refusal must degrade (log what is at risk, keep the
        worktree) so dispatch can relocate instead of blocking the run loop."""
        wt = tmp_path / "milknado-1-task"
        wt.mkdir()

        class RefusingGit(FakeGit):
            def remove_worktree(self, path: Path, target: str = "HEAD") -> None:
                raise UnlandedWorkError(path, "dirty files:\n M src/app.py")

        ex = Executor(graph=graph, git=RefusingGit(), ralph=FakeRalph(), crg=FakeCrg())
        ex._wt._worktrees[1] = wt
        with caplog.at_level(logging.WARNING):
            ex._wt.ensure_clean(1)  # must not raise
        assert wt.exists()
        assert any(
            "dirty files" in r.getMessage() and str(wt) in r.getMessage() for r in caplog.records
        ), "refusal must be logged with the worktree path and what is at risk"


class TestWorktreeRefusalSemantics:
    def test_remove_propagates_refusal(self, tmp_path: Path) -> None:
        """WorktreeManager.remove is teardown of finished work — a refusal
        (dirty/unlanded) must hard-fail the caller, never warn-and-proceed:
        that swallow was the silent-destruction bug."""
        from milknado.domains.execution.executor import WorktreeManager

        class RefusingGit(FakeGit):
            def remove_worktree(self, path: Path, target: str = "HEAD") -> None:
                raise UnlandedWorkError(path, "unlanded commits (not on main):\nabc123 wip")

        wm = WorktreeManager(RefusingGit())
        with pytest.raises(UnlandedWorkError, match="unlanded commits"):
            wm.remove(1, tmp_path / "wt")

    def test_remove_still_swallows_non_refusal_failures(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A non-refusal failure destroyed nothing (the worktree is still on
        disk) — logged and swallowed so the node lifecycle keeps moving."""
        from milknado.domains.execution.executor import WorktreeManager

        class BrokenGit(FakeGit):
            def remove_worktree(self, path: Path, target: str = "HEAD") -> None:
                raise OSError("worktree locked")

        wm = WorktreeManager(BrokenGit())
        with caplog.at_level(logging.WARNING):
            wm.remove(1, tmp_path / "wt")  # must not raise
        assert any("Failed to remove worktree" in r.getMessage() for r in caplog.records)

    def test_discard_uses_only_the_forced_variant(self, tmp_path: Path) -> None:
        """discard IS the explicit destructive path: it must reach --force
        (force_remove_worktree) and never the fail-closed removal — otherwise a
        refused discard would resurrect the cleanup-blocks-dispatch bug."""
        from milknado.domains.execution.executor import WorktreeManager

        fake_git = FakeGit()
        wm = WorktreeManager(fake_git)
        wm.discard(1, tmp_path / "wt")
        assert fake_git.force_removed == [tmp_path / "wt"]
        assert fake_git.remove_targets == []  # fail-closed removal never invoked


class TestDispatchRelocation:
    def test_occupied_path_relocates_dispatch(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
    ) -> None:
        """A preserved (refused) orphan occupying the canonical worktree path
        must not block dispatch: the new worktree gets a deterministic -2
        suffix on both path and branch (the orphan still holds the original
        branch checked out), and the orphan survives untouched."""
        fake_git = FakeGit()
        ex = Executor(graph=graph, git=fake_git, ralph=FakeRalph(), crg=FakeCrg())
        graph.add_node("task")
        occupied = config.project_root / "milknado-1-task"
        occupied.mkdir()
        (occupied / "wip.py").write_text("at-risk work\n")

        result = ex.dispatch(1, config)

        assert result.worktree == config.project_root / "milknado-1-task-2"
        assert fake_git.created == [(result.worktree, "milknado/1-task-2")]
        assert (occupied / "wip.py").read_text() == "at-risk work\n"
        node = graph.get_node(1)
        assert node is not None
        assert node.worktree_path == str(result.worktree)
        assert node.branch_name == "milknado/1-task-2"

    def test_free_path_dispatches_without_suffix(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
    ) -> None:
        fake_git = FakeGit()
        ex = Executor(graph=graph, git=fake_git, ralph=FakeRalph(), crg=FakeCrg())
        graph.add_node("task")
        result = ex.dispatch(1, config)
        assert result.worktree == config.project_root / "milknado-1-task"
        assert fake_git.created == [(result.worktree, "milknado/1-task")]

    def test_two_preserved_orphans_take_the_next_slot(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
    ) -> None:
        fake_git = FakeGit()
        ex = Executor(graph=graph, git=fake_git, ralph=FakeRalph(), crg=FakeCrg())
        graph.add_node("task")
        (config.project_root / "milknado-1-task").mkdir()
        (config.project_root / "milknado-1-task-2").mkdir()
        result = ex.dispatch(1, config)
        assert result.worktree == config.project_root / "milknado-1-task-3"
        assert fake_git.created == [(result.worktree, "milknado/1-task-3")]


class TestExecutorFail:
    def test_marks_node_failed(
        self,
        executor: Executor,
        graph: MikadoGraph,
    ) -> None:
        graph.add_node("task")
        graph.mark_running(1)
        executor.fail(1)
        node = graph.get_node(1)
        assert node is not None
        assert node.status == NodeStatus.FAILED

    def test_clears_metadata_on_fail(
        self,
        executor: Executor,
        graph: MikadoGraph,
    ) -> None:
        graph.add_node("task")
        graph.mark_running(1, worktree_path="/tmp/wt", branch_name="milknado/1-task")
        executor.fail(1)
        node = graph.get_node(1)
        assert node is not None
        assert node.worktree_path is None
        assert node.branch_name is None

    def test_removes_worktree_on_fail(
        self,
        graph: MikadoGraph,
        tmp_path: Path,
    ) -> None:
        fake_git = FakeGit()
        ex = Executor(
            graph=graph,
            git=fake_git,
            ralph=FakeRalph(),
            crg=FakeCrg(),
        )
        graph.add_node("task")
        wt = tmp_path / "worktree"
        wt.mkdir()
        graph.mark_running(1, worktree_path=str(wt), branch_name="milknado/1-task")
        ex.fail(1)
        assert wt in fake_git.removed

    def test_cancel_releases_running_node_without_recording_failure(
        self,
        graph: MikadoGraph,
        tmp_path: Path,
    ) -> None:
        fake_git = FakeGit()
        executor = Executor(graph=graph, git=fake_git, ralph=FakeRalph(), crg=FakeCrg())
        graph.add_node("task")
        worktree = tmp_path / "worktree"
        worktree.mkdir()
        graph.mark_running(1, worktree_path=str(worktree), branch_name="milknado/1-task")

        executor.cancel(1)

        node = graph.get_node(1)
        assert node is not None
        assert node.status is NodeStatus.PENDING
        assert node.worktree_path is None
        assert node.branch_name is None
        assert worktree in fake_git.removed
        assert executor.get_attempt_count(1) == 0

    def test_cancel_releases_node_when_worktree_cleanup_preserves_unlanded_work(
        self,
        graph: MikadoGraph,
        tmp_path: Path,
    ) -> None:
        class RefusingGit(FakeGit):
            def remove_worktree(self, path: Path, target: str = "HEAD") -> None:
                raise UnlandedWorkError(path, "unlanded commits (not on main):\nabc123 wip")

        executor = Executor(graph=graph, git=RefusingGit(), ralph=FakeRalph(), crg=FakeCrg())
        graph.add_node("task")
        worktree = tmp_path / "worktree"
        worktree.mkdir()
        graph.mark_running(
            1,
            worktree_path=str(worktree),
            branch_name="milknado/1-task",
            run_id="run-1",
        )

        executor.cancel(1)

        node = graph.get_node(1)
        assert node is not None
        assert node.status is NodeStatus.PENDING
        assert node.worktree_path is None
        assert node.branch_name is None
        assert node.run_id is None
        assert worktree.exists()

    def test_refusal_propagates_and_aborts_the_transition(
        self,
        graph: MikadoGraph,
        tmp_path: Path,
    ) -> None:
        """fail() tears down finished work through the fail-closed path: a
        refusal hard-fails the operation before mark_failed, preserving both
        the worktree and the node's RUNNING claim for later reconcile."""

        class RefusingGit(FakeGit):
            def remove_worktree(self, path: Path, target: str = "HEAD") -> None:
                raise UnlandedWorkError(path, "unlanded commits (not on main):\nabc123 wip")

        ex = Executor(graph=graph, git=RefusingGit(), ralph=FakeRalph(), crg=FakeCrg())
        graph.add_node("task")
        wt = tmp_path / "worktree"
        wt.mkdir()
        graph.mark_running(1, worktree_path=str(wt), branch_name="milknado/1-task")
        with pytest.raises(UnlandedWorkError, match="unlanded commits"):
            ex.fail(1)
        node = graph.get_node(1)
        assert node is not None
        assert node.status == NodeStatus.RUNNING, "transition must not land on refusal"


class TestBuildCommitMessage:
    def test_subject_format(self) -> None:
        from milknado.domains.execution.executor import _build_commit_message

        msg = _build_commit_message(7, "Add login endpoint")
        assert msg.startswith("feat(milknado-7): Add login endpoint")

    def test_body_is_full_description(self) -> None:
        from milknado.domains.execution.executor import _build_commit_message

        desc = "Implement OAuth2 flow with PKCE"
        msg = _build_commit_message(3, desc)
        lines = msg.split("\n")
        assert desc in lines

    def test_trailer_format(self) -> None:
        from milknado.domains.execution.executor import _build_commit_message

        msg = _build_commit_message(42, "some task")
        assert "Milknado-Node: 42" in msg

    def test_long_description_truncated_in_subject(self) -> None:
        from milknado.domains.execution.executor import _build_commit_message

        desc = "x" * 80
        msg = _build_commit_message(1, desc)
        subject_line = msg.split("\n")[0]
        assert subject_line.endswith("...")
        assert len(subject_line) <= len("feat(milknado-1): ") + 60

    def test_long_description_preserved_in_body(self) -> None:
        from milknado.domains.execution.executor import _build_commit_message

        desc = "x" * 80
        msg = _build_commit_message(1, desc)
        assert desc in msg


class TestIsTransient:
    def test_os_error_is_transient(self) -> None:
        from milknado.domains.execution.executor import _is_transient

        assert _is_transient(OSError("no such file"))

    def test_timeout_expired_is_transient(self) -> None:
        from milknado.domains.execution.executor import _is_transient

        assert _is_transient(subprocess.TimeoutExpired(cmd="x", timeout=5))

    def test_exit_code_124_is_transient(self) -> None:
        from milknado.domains.execution.executor import _is_transient

        assert _is_transient(subprocess.CalledProcessError(124, "cmd"))

    def test_rate_limit_message_is_transient(self) -> None:
        from milknado.domains.execution.executor import _is_transient

        assert _is_transient(RuntimeError("429 rate limit exceeded"))

    def test_transient_dispatch_error_is_transient(self) -> None:
        from milknado.domains.execution.executor import _is_transient

        assert _is_transient(TransientDispatchError("retry me"))

    def test_value_error_not_transient(self) -> None:
        from milknado.domains.execution.executor import _is_transient

        assert not _is_transient(ValueError("bad input"))

    def test_runtime_error_not_transient(self) -> None:
        from milknado.domains.execution.executor import _is_transient

        assert not _is_transient(RuntimeError("something broke"))


class TestGetAttemptCount:
    def test_returns_zero_before_dispatch(
        self,
        executor: Executor,
        graph: MikadoGraph,
    ) -> None:
        graph.add_node("task")
        assert executor.get_attempt_count(1) == 0

    def test_returns_zero_after_first_success(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
    ) -> None:
        ex = Executor(graph=graph, git=FakeGit(), ralph=FakeRalph(), crg=FakeCrg())
        graph.add_node("task")
        ex.dispatch(1, config)
        assert ex.get_attempt_count(1) == 0

    def test_increments_on_transient_retry(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
    ) -> None:
        call_count = 0

        class BurstRalph(FakeRalph):
            def create_run(
                self,
                agent: str,
                ralph_dir: Path,
                ralph_file: Path,
                commands: list[str],
                quality_gates: tuple[Gate, ...] | None,
                project_root: Path | None = None,
                commit_footer: str | None = None,
                base_oid: str | None = None,
                runtime_policy: Any | None = None,
                run_id: str | None = None,
            ) -> FakeRun:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise TransientDispatchError("transient")
                return FakeRun(state=FakeRunState(run_id=run_id or "run-1"))

        config_retry = ExecutionConfig(
            execution_agent="claude",
            quality_gates=(Gate("uv run pytest"),),
            worktree_pattern="milknado-{node_id}-{slug}",
            project_root=config.project_root,
            dispatch_max_retries=2,
            dispatch_backoff_seconds=0.0,
        )
        ex = Executor(graph=graph, git=FakeGit(), ralph=BurstRalph(), crg=FakeCrg())
        graph.add_node("task")
        ex.dispatch(1, config_retry)
        assert ex.get_attempt_count(1) == 1

    def test_non_transient_raises_immediately(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
    ) -> None:
        class FailRalph(FakeRalph):
            def create_run(
                self,
                agent: str,
                ralph_dir: Path,
                ralph_file: Path,
                commands: list[str],
                quality_gates: tuple[Gate, ...] | None,
                project_root: Path | None = None,
                commit_footer: str | None = None,
                base_oid: str | None = None,
                runtime_policy: Any | None = None,
                run_id: str | None = None,
            ) -> FakeRun:
                raise ValueError("bad config")

        ex = Executor(graph=graph, git=FakeGit(), ralph=FailRalph(), crg=FakeCrg())
        graph.add_node("task")
        with pytest.raises(ValueError, match="bad config"):
            ex.dispatch(1, config)
        assert ex.get_attempt_count(1) == 0

    def test_transient_exhausted_raises(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
    ) -> None:
        class AlwaysTransient(FakeRalph):
            def create_run(
                self,
                agent: str,
                ralph_dir: Path,
                ralph_file: Path,
                commands: list[str],
                quality_gates: tuple[Gate, ...] | None,
                project_root: Path | None = None,
                commit_footer: str | None = None,
                base_oid: str | None = None,
                runtime_policy: Any | None = None,
                run_id: str | None = None,
            ) -> FakeRun:
                raise TransientDispatchError("always fails")

        config_retry = ExecutionConfig(
            execution_agent="claude",
            quality_gates=(),
            worktree_pattern="milknado-{node_id}-{slug}",
            project_root=config.project_root,
            dispatch_max_retries=1,
            dispatch_backoff_seconds=0.0,
        )
        ex = Executor(graph=graph, git=FakeGit(), ralph=AlwaysTransient(), crg=FakeCrg())
        graph.add_node("task")
        with pytest.raises(TransientDispatchError):
            ex.dispatch(1, config_retry)


def test_dispatch_refuses_claimed_node_before_worktree_mutation(
    executor: Executor, graph: MikadoGraph, config: ExecutionConfig
) -> None:
    graph.add_node("already claimed")
    assert graph.claim_node(1, "other-run", now="2026-01-01T00:00:00+00:00", pid=2**31 - 1)
    with pytest.raises(ValueError, match="already claimed; dispatch refused"):
        executor.dispatch(1, config)


def test_dispatch_rejects_parent_run_id_that_does_not_fence_node(
    executor: Executor, graph: MikadoGraph, config: ExecutionConfig
) -> None:
    graph.add_node("wrong parent")
    assert graph.claim_node(1, "parent-run", now="2026-01-01T00:00:00+00:00")
    with pytest.raises(InvalidTransition):
        executor.dispatch(1, config, parent_run_id="different-run")


def test_dispatch_rejects_lost_run_id_fence(
    graph: MikadoGraph,
    config: ExecutionConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fence loss after the ralph run started must fail closed: the live run
    is stopped and its runs row finalized before the worktree is discarded,
    so nothing leaks into a discarded worktree with a zombie 'running' row."""
    graph.add_node("lost fence")
    git = FakeGit()
    ralph = FakeRalph(id_prefix="fence")
    executor = Executor(graph=graph, git=git, ralph=ralph, crg=FakeCrg())
    monkeypatch.setattr(graph, "replace_run_id", lambda *args: False)
    with pytest.raises(ValueError, match="dispatch fence lost"):
        executor.dispatch(1, config)
    assert len(ralph.runs_started) == 1
    run_id = ralph.runs_started[0]
    assert RUN_ID_RE.match(run_id)
    assert ralph.force_stopped == [run_id]
    row = graph.get_run(run_id)
    assert row is not None, "the started ralph run's row must not be orphaned"
    assert row["status"] != "running", "row must not zombie as running"
    assert row["error"] == "dispatch aborted"
    assert git.removed, "confirmed stop: the worktree is discarded"


def test_dispatch_fence_loss_unconfirmed_stop_preserves_worktree(
    graph: MikadoGraph,
    config: ExecutionConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unconfirmed stop on the dispatch-abort path must fail closed: the
    runs row stays 'running' (no falsely-terminal write over a still-live
    loop) and the worktree is NOT discarded out from under it."""
    graph.add_node("wedged fence loss")
    git = FakeGit()
    ralph = FakeRalph(id_prefix="wedged-fence")
    ralph.force_stop_result = False
    executor = Executor(graph=graph, git=git, ralph=ralph, crg=FakeCrg())
    monkeypatch.setattr(graph, "replace_run_id", lambda *args: False)
    with pytest.raises(ValueError, match="dispatch fence lost"):
        executor.dispatch(1, config)
    assert len(ralph.runs_started) == 1
    run_id = ralph.runs_started[0]
    assert RUN_ID_RE.match(run_id)
    assert ralph.force_stopped == [run_id]
    row = graph.get_run(run_id)
    assert row is not None and row["status"] == "running", (
        "unconfirmed stop: row must stay running, not falsely terminal"
    )
    assert git.removed == [], "worktree must not be discarded under a live loop"


def test_dispatch_fence_loss_force_stop_raise_preserves_worktree(
    graph: MikadoGraph,
    config: ExecutionConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A force-stop that raises is an unconfirmed stop: same fail-closed
    policy — row stays 'running', worktree preserved."""
    graph.add_node("raising stop fence loss")
    git = FakeGit()
    ralph = FakeRalph(id_prefix="raising-fence")
    ralph.force_stop_raises = RuntimeError("loop manager wedged")
    executor = Executor(graph=graph, git=git, ralph=ralph, crg=FakeCrg())
    monkeypatch.setattr(graph, "replace_run_id", lambda *args: False)
    with pytest.raises(ValueError, match="dispatch fence lost"):
        executor.dispatch(1, config)
    assert len(ralph.runs_started) == 1
    run_id = ralph.runs_started[0]
    assert RUN_ID_RE.match(run_id)
    assert ralph.force_stopped == [run_id]
    row = graph.get_run(run_id)
    assert row is not None and row["status"] == "running"
    assert git.removed == []


def test_watcher_spawn_failure_after_start_stops_and_finalizes(
    graph: MikadoGraph,
    config: ExecutionConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Post-start raise window: if the cancel watcher cannot be spawned
    (runs_dir mkdir OSError / Thread.start RuntimeError), the just-started
    run's id never escapes _create_ralph_run, so the run must be stopped and
    its row finalized there — the caller's cleanup sees started_run_id=None."""
    import milknado.domains.execution.executor as executor_module

    graph.add_node("watcher spawn fails")
    ralph = FakeRalph(id_prefix="spawnfail")
    executor = Executor(graph=graph, git=FakeGit(), ralph=ralph, crg=FakeCrg())

    def _boom(root: Path) -> Path:
        raise OSError("read-only .milknado")

    monkeypatch.setattr(executor_module, "runs_dir", _boom)
    with pytest.raises(OSError, match="read-only .milknado"):
        executor._dispatch_once(1, config)  # direct: skip the transient-retry wrapper
    assert len(ralph.runs_started) == 1, "the run was started before the raise"
    run_id = ralph.runs_started[0]
    assert RUN_ID_RE.match(run_id)
    assert ralph.force_stopped == [run_id]
    row = graph.get_run(run_id)
    assert row is not None and row["status"] != "running", (
        "post-start failure must not leak a zombie 'running' row"
    )
    assert row["error"] == "dispatch aborted"


def test_watcher_spawn_failure_unconfirmed_stop_preserves_worktree(
    graph: MikadoGraph,
    config: ExecutionConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unconfirmed stop in the post-start raise window: the teardown result
    must cross the boundary into the caller's cleanup. The run id never
    escaped _create_ralph_run, so without the exception-carried outcome the
    cleanup would default stop_confirmed=True and force-discard the
    worktree under a possibly-live loop — the fence-loss fail-closed policy
    must hold at this seam too."""
    import milknado.domains.execution.executor as executor_module

    graph.add_node("watcher spawn fails wedged")
    git = FakeGit()
    ralph = FakeRalph(id_prefix="spawnwedged")
    ralph.force_stop_result = False
    executor = Executor(graph=graph, git=git, ralph=ralph, crg=FakeCrg())

    def _boom(root: Path) -> Path:
        raise OSError("read-only .milknado")

    monkeypatch.setattr(executor_module, "runs_dir", _boom)
    with pytest.raises(OSError, match="read-only .milknado"):
        executor._dispatch_once(1, config)  # direct: skip the transient-retry wrapper
    assert len(ralph.runs_started) == 1, "the run was started before the raise"
    run_id = ralph.runs_started[0]
    assert RUN_ID_RE.match(run_id)
    assert ralph.force_stopped == [run_id], (
        "the post-start teardown attempted the stop exactly once; the caller's "
        "cleanup must not re-stop a run it never started"
    )
    row = graph.get_run(run_id)
    assert row is not None and row["status"] == "running", (
        "unconfirmed stop: row must stay running, not falsely terminal"
    )
    assert git.removed == [], "worktree must not be discarded under a live loop"


def test_unconfirmed_stop_aborted_run_finalized_when_loop_self_exits(
    graph: MikadoGraph,
    config: ExecutionConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An aborted run whose stop was unconfirmed has no completion-path owner
    (its id never enters _worker_run_id_by_node). When the wedged loop later
    exits on its own, the still-observing cancel watcher must finalize the
    row with a neutral marker instead of letting it zombie 'running'
    forever."""
    graph.add_node("wedged fence loss self-exit")
    git = FakeGit()
    ralph = FakeRalph(live=True, id_prefix="selfexit")
    ralph.force_stop_result = False
    executor = Executor(graph=graph, git=git, ralph=ralph, crg=FakeCrg())
    monkeypatch.setattr(graph, "replace_run_id", lambda *args: False)
    with pytest.raises(ValueError, match="dispatch fence lost"):
        executor.dispatch(1, config)
    assert len(ralph.runs_started) == 1
    run_id = ralph.runs_started[0]
    assert RUN_ID_RE.match(run_id)
    row = graph.get_run(run_id)
    assert row is not None and row["status"] == "running"
    assert git.removed == [], "unconfirmed stop: worktree preserved"

    # The wedged loop exits on its own; the watcher observes the dead thread
    # and finalizes the ownerless row.
    ralph._stop_events[run_id].set()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        row = graph.get_run(run_id)
        if row is not None and row["status"] != "running":
            break
        time.sleep(0.05)
    row = graph.get_run(run_id)
    assert row is not None and row["status"] == "failed", (
        "self-exited aborted run must not zombie as running"
    )
    assert row["error"] == "loop exited after unconfirmed stop"
    assert git.removed == [], "the finalizer must not discard the preserved worktree"


def test_unconfirmed_stop_blocks_retry_from_starting_second_worker(
    graph: MikadoGraph,
    config: ExecutionConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Blocker 2: an unconfirmed stop during a transient dispatch failure
    must retain the node claim so the retry loop cannot reclaim the node and
    start a second worker (race-2) while the first's abandoned loop may
    still be alive and writing into the shared worktree."""
    import milknado.domains.execution.executor as executor_module

    graph.add_node("unconfirmed stop retry")
    ralph = FakeRalph(id_prefix="race")
    ralph.force_stop_result = False  # the post-start abort's stop never confirms
    executor = Executor(graph=graph, git=FakeGit(), ralph=ralph, crg=FakeCrg())

    def _boom(root: Path) -> Path:
        raise OSError("transient watcher-setup failure")

    monkeypatch.setattr(executor_module, "runs_dir", _boom)
    retry_config = replace(config, dispatch_max_retries=1, dispatch_backoff_seconds=0)

    with pytest.raises(ValueError, match="already claimed"):
        executor.dispatch(1, retry_config)

    assert len(ralph.runs_started) == 1, (
        "retry must not start a second worker while the first's stop is unconfirmed"
    )
    node = graph.get_node(1)
    assert node is not None
    assert node.status.value == "running", "claim retained, not released, under unconfirmed stop"


def test_stop_aborted_run_registers_unconfirmed_before_attempting_stop(
    graph: MikadoGraph,
) -> None:
    """High: _stop_aborted_run must register the run as unconfirmed BEFORE
    attempting the stop, not only on a failure branch after the call
    returns — otherwise the independently-polling cancel watcher can
    observe a dead thread and check membership in the gap before the id is
    ever added, permanently zombie-ing the row (nothing else owns it)."""
    ralph = FakeRalph(id_prefix="registerfirst")
    executor = Executor(graph=graph, git=FakeGit(), ralph=ralph, crg=FakeCrg())

    entered = threading.Event()
    release = threading.Event()
    original_force_stop_run = ralph.force_stop_run

    def _blocking_force_stop_run(run_id: str, timeout: float | None = None) -> bool:
        entered.set()
        release.wait(timeout=5)
        return original_force_stop_run(run_id, timeout=timeout)

    ralph.force_stop_run = _blocking_force_stop_run

    thread = threading.Thread(
        target=lambda: executor._stop_aborted_run("run-registerfirst", context="test")
    )
    thread.start()
    try:
        assert entered.wait(timeout=5), "force_stop_run must have been called"
        assert "run-registerfirst" in executor._unconfirmed_stop_run_ids, (
            "the run must already be registered as unconfirmed while the stop is in flight"
        )
    finally:
        release.set()
        thread.join(timeout=5)


def test_review_redispatch_fence_loss_stops_and_finalizes_fresh_run(
    graph: MikadoGraph,
    config: ExecutionConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The review-redispatch sibling of the dispatch fence-loss path: when
    replace_run_id loses the fence, the FRESH run must be stopped and its
    runs row finalized — not silently suppressed under contextlib.suppress
    with its row left to zombie as 'running'."""
    graph.add_node("redispatch fence lost")
    ralph = FakeRalph(id_prefix="rd")
    executor = Executor(graph=graph, git=FakeGit(), ralph=ralph, crg=FakeCrg())
    result = executor.dispatch(1, config)
    node = graph.get_node(1)
    assert node is not None
    monkeypatch.setattr(graph, "replace_run_id", lambda *args: False)
    with pytest.raises(ValueError, match="review redispatch fence lost"):
        executor._redispatch_review_round(node, config, result.worktree)
    assert len(ralph.runs_started) == 2
    old_run_id, new_run_id = ralph.runs_started
    assert old_run_id == result.run_id
    assert ralph.force_stopped == [new_run_id]
    row = graph.get_run(new_run_id)
    assert row is not None, "the fresh run's row must not be orphaned"
    assert row["status"] != "running", "fresh run's row must not zombie as running"
    assert row["error"] == "dispatch aborted"
    assert graph.get_node(1).run_id == old_run_id, "the old run id keeps the node fence"
    old_row = graph.get_run(old_run_id)
    assert old_row is not None and old_row["status"] != "running", (
        "the superseded worker run's row must be finalized unconditionally, "
        "before the fence check, regardless of the raise that follows"
    )
    assert old_row["detail"] == "superseded by review round 0 redispatch"


def test_review_redispatch_fence_loss_unconfirmed_stop_leaves_row_running(
    graph: MikadoGraph,
    config: ExecutionConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unconfirmed stop on the redispatch fence-loss path: the fresh run's
    row stays 'running' (recoverable), never a falsely-terminal write over a
    still-live loop."""
    graph.add_node("redispatch wedged fence")
    ralph = FakeRalph(id_prefix="rdwedged")
    ralph.force_stop_result = False
    executor = Executor(graph=graph, git=FakeGit(), ralph=ralph, crg=FakeCrg())
    result = executor.dispatch(1, config)
    node = graph.get_node(1)
    assert node is not None
    monkeypatch.setattr(graph, "replace_run_id", lambda *args: False)
    with pytest.raises(ValueError, match="review redispatch fence lost"):
        executor._redispatch_review_round(node, config, result.worktree)
    assert len(ralph.runs_started) == 2
    new_run_id = ralph.runs_started[1]
    assert ralph.force_stopped == [new_run_id]
    row = graph.get_run(new_run_id)
    assert row is not None and row["status"] == "running"


def test_review_redispatch_adopted_owner_fence_loss_stops_fresh_run(
    graph: MikadoGraph,
    config: ExecutionConfig,
) -> None:
    """The adopted-owner sibling branch: a node whose recorded run_id no
    longer matches the owner fence must also stop + finalize the fresh run."""
    graph.add_node("adopted fence lost")
    ralph = FakeRalph(id_prefix="adopted")
    executor = Executor(graph=graph, git=FakeGit(), ralph=ralph, crg=FakeCrg())
    result = executor.dispatch(1, config)
    node = graph.get_node(1)
    assert node is not None
    executor._owner_fence_by_node[1] = "other-owner-fence"
    with pytest.raises(ValueError, match="adopted owner fence lost"):
        executor._redispatch_review_round(node, config, result.worktree)
    assert len(ralph.runs_started) == 2
    old_run_id, new_run_id = ralph.runs_started
    assert old_run_id == result.run_id
    assert ralph.force_stopped == [new_run_id]
    row = graph.get_run(new_run_id)
    assert row is not None and row["status"] != "running"
    assert row["error"] == "dispatch aborted"
    old_row = graph.get_run(old_run_id)
    assert old_row is not None and old_row["status"] != "running", (
        "the superseded worker run's row must be finalized unconditionally, "
        "before the fence check, regardless of the raise that follows"
    )
    assert old_row["detail"] == "superseded by review round 0 redispatch"


def test_review_redispatch_finalizes_prior_round_even_when_fresh_run_fails_to_start(
    graph: MikadoGraph,
    config: ExecutionConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """High: the prior round's row must be finalized regardless of whether
    the new round can even be created — a _create_ralph_run failure (ralph-md
    generation, create_run, start_run) previously propagated before any of
    the finalize calls ran, leaving the prior round zombied 'running' forever."""
    graph.add_node("redispatch fresh run fails to start")
    ralph = FakeRalph(id_prefix="redispatchboom")
    executor = Executor(graph=graph, git=FakeGit(), ralph=ralph, crg=FakeCrg())
    result = executor.dispatch(1, config)
    node = graph.get_node(1)
    assert node is not None

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("ralph create_run boom")

    monkeypatch.setattr(ralph, "create_run", _boom)

    with pytest.raises(RuntimeError, match="ralph create_run boom"):
        executor._redispatch_review_round(node, config, result.worktree)

    prior_row = graph.get_run(result.run_id)
    assert prior_row is not None and prior_row["status"] != "running", (
        "the prior round's row must be finalized even though the fresh round never started"
    )
    assert prior_row["detail"] == "superseded by review round 0 redispatch"


def test_default_fake_ralph_id_prefixes_are_unique() -> None:
    """Default-constructed fakes must namespace their run ids per instance,
    so watcher thread names can never collide across tests."""
    first, second = FakeRalph(), FakeRalph()
    cfg = {
        "agent": "a",
        "ralph_dir": Path("/x"),
        "ralph_file": Path("/x/RALPH.md"),
        "commands": [],
        "quality_gates": None,
    }
    assert first.create_run(**cfg).state.run_id != second.create_run(**cfg).state.run_id


def test_finalize_worker_run_swallows_finish_run_failure(
    graph: MikadoGraph,
    monkeypatch: pytest.MonkeyPatch,
    caplog,  # noqa: ANN001
) -> None:
    """A failed finalize must not kill node completion: the finish_run raise
    is swallowed and logged loud, like the dispatch insert."""
    from milknado.domains.common.types import RunResult

    executor = Executor(graph=graph, git=FakeGit(), ralph=FakeRalph(), crg=FakeCrg())
    monkeypatch.setattr(graph, "get_run", lambda run_id: {"status": "running"})

    def _boom(run_id: str, result: Any) -> bool:
        raise RuntimeError("db wedged")

    monkeypatch.setattr(graph, "finish_run", _boom)
    result = RunResult(
        status="failed",
        exit_code=None,
        timed_out=False,
        ended_at="2026-07-24T00:00:00Z",
        error="x",
    )
    with caplog.at_level(logging.ERROR, logger="milknado.domains.execution.executor"):
        executor._finalize_worker_run("run-x", result)
    assert any("runs-row finalize failed" in r.message for r in caplog.records)


def test_watcher_retries_after_force_stop_raise(
    graph: MikadoGraph,
    config: ExecutionConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog,  # noqa: ANN001
) -> None:
    """force_stop_run raising must be retried with backoff: the failure is
    logged, the row stays running (fail-closed), and once the raise clears
    the watcher stops the loop and finalizes the row cancelled."""
    import milknado.domains.execution.executor as executor_module
    from milknado.domains.dispatch._runstate import request_cancel, runs_dir

    graph.add_node("force stop raise retries")
    ralph = FakeRalph(live=True, id_prefix="fstopraise")
    executor = Executor(graph=graph, git=FakeGit(), ralph=ralph, crg=FakeCrg())
    result = executor.dispatch(1, config)

    real_sleep = time.sleep
    monkeypatch.setattr(executor_module.time, "sleep", lambda secs: real_sleep(0.001))
    ralph.force_stop_raises = RuntimeError("stop boom")

    with caplog.at_level(logging.ERROR, logger="milknado.domains.execution.executor"):
        request_cancel(runs_dir(tmp_path), result.run_id)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and len(ralph.force_stopped) < 3:
            real_sleep(0.01)
    assert len(ralph.force_stopped) >= 3, "watcher must retry the raising force-stop"
    assert any("force-stop raised for cancelled ralph run" in r.message for r in caplog.records), (
        "the raise must be logged loud"
    )
    row = graph.get_run(result.run_id)
    assert row is not None and row["status"] == "running", (
        "fail-closed: a raising force-stop must not leave a falsely-terminal row"
    )

    # The transient failure clears: the retry stops the loop and finalizes cancelled.
    ralph.force_stop_raises = None
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        row = graph.get_run(result.run_id)
        if row is not None and row["status"] != "running":
            break
        real_sleep(0.05)
    row = graph.get_run(result.run_id)
    assert row is not None and row["error"] == "cancelled", (
        "watcher must retry the force-stop and finalize cancelled"
    )


def test_watcher_does_not_overwrite_a_completion_that_won_the_race(
    graph: MikadoGraph,
    config: ExecutionConfig,
    tmp_path: Path,
) -> None:
    """High: a cancel racing with natural completion must not win the DB
    fence as failed/cancelled. force_stop_and_join only confirms the thread
    exited, not why — if the loop's own try_commit_completion() won just
    before the force-stop landed, the watcher must leave the row alone for
    the normal completion path to finalize as done."""
    from milknado.domains.dispatch._runstate import request_cancel, runs_dir

    graph.add_node("cancel races completion")
    ralph = FakeRalph(live=True, id_prefix="racecomplete")
    executor = Executor(graph=graph, git=FakeGit(), ralph=ralph, crg=FakeCrg())
    result = executor.dispatch(1, config)

    # Simulate the loop committing completion just before the force-stop lands.
    ralph.runs[result.run_id].state.status = RunStatus.COMPLETED

    request_cancel(runs_dir(tmp_path), result.run_id)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not ralph.force_stopped:
        time.sleep(0.01)
    assert ralph.force_stopped == [result.run_id], "watcher must still confirm the stop"

    # Give the watcher's post-stop check a moment to run and (correctly) no-op.
    time.sleep(0.2)
    row = graph.get_run(result.run_id)
    assert row is not None and row["status"] == "running", (
        "a completed run must not be overwritten as cancelled by the racing watcher"
    )
