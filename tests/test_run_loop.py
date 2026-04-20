import collections
import io
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from milknado.domains.common.types import (
    MikadoNode,
    NodeStatus,
    RebaseResult,
)
from milknado.domains.execution import (
    ExecutionConfig,
    Executor,
    RunLoop,
)
from milknado.domains.execution.run_loop.display import (
    TuiState,
    _build_log_panel,
    _build_title,
    _build_worker_table,
    _render_overlay,
    _render_progress_bar,
)
from milknado.domains.graph import MikadoGraph


@dataclass
class FakeRunState:
    run_id: str = "run-1"


@dataclass
class FakeRun:
    state: FakeRunState = field(default_factory=FakeRunState)


class FakeGit:
    def __init__(self) -> None:
        self.created: list[tuple[Path, str]] = []
        self.removed: list[Path] = []
        self.rebase_result: RebaseResult = RebaseResult(success=True)

    def create_worktree(self, path: Path, branch: str) -> Path:
        self.created.append((path, branch))
        path.mkdir(parents=True, exist_ok=True)
        return path

    def remove_worktree(self, path: Path) -> None:
        self.removed.append(path)

    def rebase(self, worktree: Path, onto: str) -> RebaseResult:
        return self.rebase_result

    def current_branch(self) -> str:
        return "main"

    def commit_all(self, worktree: Path, message: str) -> None:
        pass

    def squash_and_commit(self, worktree: Path, onto: str, msg: str) -> None:
        pass


class FakeCrg:
    def ensure_graph(self, project_root: Path) -> None:
        pass

    def get_impact_radius(self, files: list[str]) -> dict[str, Any]:
        return {"files": files}

    def get_architecture_overview(self) -> dict[str, Any]:
        return {"modules": []}

    def list_communities(
        self, sort_by: str = "size", min_size: int = 0,
    ) -> list[dict[str, Any]]:
        return []

    def list_flows(
        self, sort_by: str = "criticality", limit: int = 50,
    ) -> list[dict[str, Any]]:
        return []

    def get_minimal_context(
        self, task: str = "", changed_files: list[str] | None = None,
    ) -> dict[str, Any]:
        return {}

    def get_bridge_nodes(self, top_n: int = 10) -> list[dict[str, Any]]:
        return []

    def get_hub_nodes(self, top_n: int = 10) -> list[dict[str, Any]]:
        return []

    def semantic_search_nodes(
        self, query: str, top_n: int = 5,
    ) -> list[dict[str, Any]]:
        return []

    def semantic_search(
        self, query: str, top_n: int = 5, detail_level: str = "minimal",
    ) -> list[dict[str, Any]]:
        return []


class FakeRalph:
    def __init__(self) -> None:
        self._run_counter = 0
        self._success: dict[str, bool] = {}
        self._pending_completions: list[tuple[str, bool]] = []

    def create_run(
        self,
        agent: str,
        ralph_dir: Path,
        ralph_file: Path,
        commands: list[str],
        quality_gates: list[str],
        project_root: Path | None = None,
    ) -> FakeRun:
        self._run_counter += 1
        run_id = f"run-{self._run_counter}"
        success = self._success.get(run_id, True)
        self._pending_completions.append((run_id, success))
        return FakeRun(state=FakeRunState(run_id=run_id))

    def start_run(self, run_id: str) -> None:
        pass

    def stop_run(self, run_id: str) -> None:
        pass

    def list_runs(self) -> list[Any]:
        return []

    def get_run(self, run_id: str) -> Any | None:
        return None

    def get_run_stdout(self, run_id: str) -> list[str]:
        return []

    def wait_for_next_completion(
        self,
        active_run_ids: set[str],
        timeout: float | None = None,
    ) -> tuple[str, bool]:
        for i, (run_id, success) in enumerate(self._pending_completions):
            if run_id in active_run_ids:
                self._pending_completions.pop(i)
                return run_id, success
        raise RuntimeError("No pending completions for active runs")

    def poll_progress_events(self) -> list[Any]:
        return []

    def verify_spec(self, spec_text: str, graph_state: str) -> Any:
        from milknado.domains.common.protocols import VerifySpecResult
        return VerifySpecResult(outcome="done")

    def generate_ralph_md(
        self,
        node: MikadoNode,
        context: str,
        quality_gates: list[str],
        output_path: Path,
    ) -> Path:
        return output_path

    def set_run_fails(self, run_id: str) -> None:
        self._success[run_id] = False


@pytest.fixture()
def config(tmp_path: Path) -> ExecutionConfig:
    return ExecutionConfig(
        execution_agent="claude",
        quality_gates=("uv run pytest",),
        worktree_pattern="milknado-{node_id}-{slug}",
        project_root=tmp_path,
    )


@pytest.fixture()
def fake_git() -> FakeGit:
    return FakeGit()


@pytest.fixture()
def fake_ralph() -> FakeRalph:
    return FakeRalph()


@pytest.fixture()
def fake_crg() -> FakeCrg:
    return FakeCrg()


@pytest.fixture()
def executor(
    graph: MikadoGraph,
    fake_git: FakeGit,
    fake_ralph: FakeRalph,
    fake_crg: FakeCrg,
) -> Executor:
    return Executor(graph=graph, git=fake_git, ralph=fake_ralph, crg=fake_crg)


@pytest.fixture()
def run_loop(
    executor: Executor,
    graph: MikadoGraph,
    fake_ralph: FakeRalph,
) -> RunLoop:
    return RunLoop(executor=executor, graph=graph, ralph=fake_ralph)


class TestRunLoopSingleNode:
    def test_solo_root_not_dispatched(
        self,
        run_loop: RunLoop,
        graph: MikadoGraph,
        config: ExecutionConfig,
    ) -> None:
        # Root is never dispatched as a work node; it's completed via verify_spec.
        # Without spec_text, root stays PENDING and root_done is False.
        graph.add_node("root goal")
        result = run_loop.run(config, "main")

        assert result.dispatched_total == 0
        assert result.completed_total == 0
        assert result.failed_total == 0
        assert result.root_done is False

    def test_solo_root_stays_pending_without_spec(
        self,
        run_loop: RunLoop,
        graph: MikadoGraph,
        config: ExecutionConfig,
    ) -> None:
        graph.add_node("root goal")
        run_loop.run(config, "main")

        root = graph.get_node(1)
        assert root is not None
        assert root.status == NodeStatus.PENDING


class TestRunLoopParentChild:
    def test_dispatches_leaf_not_root(
        self,
        run_loop: RunLoop,
        graph: MikadoGraph,
        config: ExecutionConfig,
    ) -> None:
        # Root is excluded from dispatch; only the leaf is dispatched.
        # Without spec_text, root stays PENDING.
        root = graph.add_node("root")
        leaf = graph.add_node("leaf", parent_id=root.id)

        result = run_loop.run(config, "main")

        assert result.dispatched_total == 1
        assert result.completed_total == 1
        assert result.root_done is False
        leaf_node = graph.get_node(leaf.id)
        root_node = graph.get_node(root.id)
        assert leaf_node is not None and leaf_node.status == NodeStatus.DONE
        assert root_node is not None and root_node.status == NodeStatus.PENDING

    def test_leaf_done_root_pending_without_spec(
        self,
        run_loop: RunLoop,
        graph: MikadoGraph,
        config: ExecutionConfig,
    ) -> None:
        root = graph.add_node("root")
        graph.add_node("leaf", parent_id=root.id)

        run_loop.run(config, "main")

        root_node = graph.get_node(root.id)
        assert root_node is not None
        assert root_node.status == NodeStatus.PENDING


class TestRunLoopParallelLeaves:
    def test_dispatches_parallel_leaves_not_root(
        self,
        run_loop: RunLoop,
        graph: MikadoGraph,
        config: ExecutionConfig,
    ) -> None:
        root = graph.add_node("root")
        graph.add_node("leaf-a", parent_id=root.id)
        graph.add_node("leaf-b", parent_id=root.id)

        result = run_loop.run(config, "main")

        assert result.dispatched_total == 2
        assert result.completed_total == 2
        # Root not dispatched; stays PENDING without spec_text.
        assert result.root_done is False


class TestRunLoopConcurrencyLimit:
    def test_respects_concurrency_limit(
        self,
        run_loop: RunLoop,
        graph: MikadoGraph,
        config: ExecutionConfig,
    ) -> None:
        root = graph.add_node("root")
        graph.add_node("a", parent_id=root.id)
        graph.add_node("b", parent_id=root.id)
        graph.add_node("c", parent_id=root.id)

        result = run_loop.run(config, "main", concurrency_limit=2)

        # 3 leaves dispatched, root excluded.
        assert result.dispatched_total == 3
        assert result.completed_total == 3
        assert result.root_done is False


class TestRunLoopFailure:
    def test_failed_leaf_marks_leaf_failed(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
        fake_git: FakeGit,
        fake_crg: FakeCrg,
    ) -> None:
        # Root is not dispatched; only the leaf is. Set the leaf's run to fail.
        ralph = FakeRalph()
        ralph._success["run-1"] = False

        executor = Executor(graph=graph, git=fake_git, ralph=ralph, crg=fake_crg)
        loop = RunLoop(executor=executor, graph=graph, ralph=ralph)

        root = graph.add_node("root")
        leaf = graph.add_node("will fail", parent_id=root.id)
        result = loop.run(config, "main")

        assert result.failed_total == 1
        assert result.root_done is False
        leaf_node = graph.get_node(leaf.id)
        assert leaf_node is not None
        assert leaf_node.status == NodeStatus.FAILED

    def test_failed_leaf_blocks_parent(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
        fake_git: FakeGit,
        fake_crg: FakeCrg,
    ) -> None:
        ralph = FakeRalph()
        ralph._success["run-1"] = False

        executor = Executor(graph=graph, git=fake_git, ralph=ralph, crg=fake_crg)
        loop = RunLoop(executor=executor, graph=graph, ralph=ralph)

        root = graph.add_node("root")
        graph.add_node("leaf", parent_id=root.id)
        result = loop.run(config, "main")

        assert result.failed_total == 1
        assert result.root_done is False
        root_node = graph.get_node(root.id)
        assert root_node is not None
        assert root_node.status == NodeStatus.PENDING


class TestRunLoopResult:
    def test_empty_graph_returns_immediately(
        self,
        run_loop: RunLoop,
        graph: MikadoGraph,
        config: ExecutionConfig,
    ) -> None:
        result = run_loop.run(config, "main")

        assert result.dispatched_total == 0
        assert result.completed_total == 0
        assert result.root_done is False


class TestRunLoopDispatchFailure:
    def test_dispatch_failure_marks_leaf_failed(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
        fake_git: FakeGit,
        fake_crg: FakeCrg,
    ) -> None:
        ralph = FakeRalph()
        ralph.generate_ralph_md = lambda *_a, **_kw: (_ for _ in ()).throw(  # type: ignore
            RuntimeError("ralph exploded"),
        )
        executor = Executor(graph=graph, git=fake_git, ralph=ralph, crg=fake_crg)
        loop = RunLoop(executor=executor, graph=graph, ralph=ralph)

        root = graph.add_node("root")
        leaf = graph.add_node("doomed", parent_id=root.id)
        result = loop.run(config, "main")

        assert result.dispatched_total == 0
        assert result.root_done is False
        leaf_node = graph.get_node(leaf.id)
        assert leaf_node is not None
        assert leaf_node.status == NodeStatus.FAILED

    def test_dispatch_failure_does_not_crash_loop(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
        fake_git: FakeGit,
        fake_crg: FakeCrg,
    ) -> None:
        ralph = FakeRalph()
        call_count = 0
        original_generate = ralph.generate_ralph_md

        def fail_first_only(*args: Any, **kwargs: Any) -> Path:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("ralph exploded")
            return original_generate(*args, **kwargs)

        ralph.generate_ralph_md = fail_first_only  # type: ignore
        executor = Executor(graph=graph, git=fake_git, ralph=ralph, crg=fake_crg)
        loop = RunLoop(executor=executor, graph=graph, ralph=ralph)

        root = graph.add_node("root")
        graph.add_node("doomed-leaf", parent_id=root.id)
        graph.add_node("good-leaf", parent_id=root.id)

        result = loop.run(config, "main")

        assert result.root_done is False
        good_leaf = graph.get_node(3)
        assert good_leaf is not None
        assert good_leaf.status == NodeStatus.DONE


class TestRunLoopRebaseConflicts:
    def test_rebase_conflict_details_surfaced_in_result(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
        fake_crg: FakeCrg,
    ) -> None:
        fake_git = FakeGit()
        fake_git.rebase_result = RebaseResult(
            success=False,
            conflicting_files=("src/models.py", "src/views.py"),
            detail="CONFLICT (content): Merge conflict in src/models.py",
        )
        ralph = FakeRalph()
        executor = Executor(graph=graph, git=fake_git, ralph=ralph, crg=fake_crg)
        loop = RunLoop(executor=executor, graph=graph, ralph=ralph)

        root = graph.add_node("root")
        leaf = graph.add_node("conflicting node", parent_id=root.id)
        result = loop.run(config, "main")

        assert result.failed_total == 1
        assert len(result.rebase_conflicts) == 1
        conflict = result.rebase_conflicts[0]
        assert conflict.node_id == leaf.id
        assert conflict.conflicting_files == ("src/models.py", "src/views.py")
        assert "Merge conflict" in conflict.detail

    def test_no_conflicts_yields_empty_tuple(
        self,
        run_loop: RunLoop,
        graph: MikadoGraph,
        config: ExecutionConfig,
    ) -> None:
        root = graph.add_node("root")
        graph.add_node("clean node", parent_id=root.id)
        result = run_loop.run(config, "main")

        assert result.rebase_conflicts == ()


class TestRunLoopFileConflicts:
    def test_serializes_conflicting_nodes(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
        fake_git: FakeGit,
        fake_ralph: FakeRalph,
        fake_crg: FakeCrg,
    ) -> None:
        executor = Executor(
            graph=graph, git=fake_git, ralph=fake_ralph, crg=fake_crg,
        )
        loop = RunLoop(executor=executor, graph=graph, ralph=fake_ralph)

        root = graph.add_node("root")
        a = graph.add_node("a", parent_id=root.id)
        b = graph.add_node("b", parent_id=root.id)
        graph.set_file_ownership(a.id, ["shared.py"])
        graph.set_file_ownership(b.id, ["shared.py"])

        result = loop.run(config, "main")

        # 2 leaves dispatched (serialized due to file conflict), root excluded.
        assert result.dispatched_total == 2
        assert result.completed_total == 2
        assert result.root_done is False


_RICH_DESC = (
    "US-204: split bundling\n\n## Reuse candidates\n- foo.py:123\n- bar.py:45"
)


def _make_tui_state(node_id: int) -> TuiState:
    return TuiState(
        tick=0,
        active={"run-1": node_id},
        logs=[],
        dispatched_at={"run-1": time.monotonic()},
        attempts={},
        progress_by_run={},
        completion_durations=[],
        stall_threshold=300.0,
        max_retries=2,
        exec_agent="claude",
    )


def _render_to_text(renderable: Any) -> str:
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, no_color=True, width=200)
    console.print(renderable)
    return buf.getvalue()


def _snapshot(renderable: Any) -> str:
    console = Console(record=True, width=200)
    console.print(renderable)
    return console.export_text()


class TestWorkerTableDescriptionSanitization:
    def test_description_cell_is_summarized(self, graph: MikadoGraph) -> None:
        node = graph.add_node(_RICH_DESC)
        state = _make_tui_state(node.id)
        rendered = _render_to_text(_build_worker_table(state, graph))

        assert "split bundling" in rendered
        assert "##" not in rendered
        assert "Reuse" not in rendered

    def test_description_cell_has_no_raw_newlines(self, graph: MikadoGraph) -> None:
        node = graph.add_node(_RICH_DESC)
        state = _make_tui_state(node.id)
        rendered = _render_to_text(_build_worker_table(state, graph))

        assert "\n\n" not in rendered


class TestRenderOverlayPreservesRawDescription:
    """Counterpoint: _render_overlay must NOT sanitize the description."""

    def test_overlay_rendered_contains_markdown_header(
        self, graph: MikadoGraph
    ) -> None:
        node = graph.add_node(_RICH_DESC)
        state = _make_tui_state(node.id)
        ralph = FakeRalph()
        panel = _render_overlay("run-1", state, graph, ralph)
        rendered = _render_to_text(panel)

        assert "##" in rendered

    def test_overlay_sanitization_is_nondestructive(
        self, graph: MikadoGraph
    ) -> None:
        node = graph.add_node(_RICH_DESC)
        state = _make_tui_state(node.id)
        _build_worker_table(state, graph)

        fresh = graph.get_node(node.id)
        assert fresh is not None
        assert "##" in fresh.description
        assert "- foo.py:123" in fresh.description
        assert "- bar.py:45" in fresh.description


# ---------------------------------------------------------------------------
# US-208: headless TUI snapshot tests
# ---------------------------------------------------------------------------


class TestBuildTitle:
    def test_shows_done_count(self, graph: MikadoGraph) -> None:
        n = graph.add_node("done-node")
        graph.mark_running(n.id)
        graph.mark_done(n.id)

        text = _snapshot(_build_title({}, graph))

        assert "1 done" in text

    def test_shows_failed_count(self, graph: MikadoGraph) -> None:
        n = graph.add_node("fail-node")
        graph.mark_failed(n.id)

        text = _snapshot(_build_title({}, graph))

        assert "1 failed" in text

    def test_shows_blocked_count(self, graph: MikadoGraph) -> None:
        n = graph.add_node("blocked-node")
        graph.mark_blocked(n.id)

        text = _snapshot(_build_title({}, graph))

        assert "1 blocked" in text

    def test_all_counts_aggregate(self, graph: MikadoGraph) -> None:
        done_n = graph.add_node("done")
        graph.mark_running(done_n.id)
        graph.mark_done(done_n.id)

        fail_n = graph.add_node("fail")
        graph.mark_failed(fail_n.id)

        blocked_n = graph.add_node("blocked")
        graph.mark_blocked(blocked_n.id)

        text = _snapshot(_build_title({}, graph))

        assert "1 done" in text
        assert "1 failed" in text
        assert "1 blocked" in text


class TestRenderProgressBar:
    def test_normal_returns_spinner_frame(self) -> None:
        result = _render_progress_bar("◜", elapsed=0.0, pct=None, stall_threshold=300.0)

        assert "◜" in result
        assert "⚠" not in result

    def test_stalled_includes_warning_glyph(self) -> None:
        result = _render_progress_bar("◜", elapsed=400.0, pct=None, stall_threshold=300.0)

        assert "⚠" in result

    def test_with_pct_shows_bar_and_percentage(self) -> None:
        result = _render_progress_bar("◜", elapsed=10.0, pct=70.0, stall_threshold=300.0)

        assert "70%" in result
        assert "█" in result

    def test_completed_full_bar(self) -> None:
        result = _render_progress_bar("◜", elapsed=30.0, pct=100.0, stall_threshold=300.0)

        assert "100%" in result
        assert "░" not in result


class TestBuildWorkerTableColumns:
    def test_elapsed_column_present(self, graph: MikadoGraph) -> None:
        node = graph.add_node("simple task")
        state = _make_tui_state(node.id)
        text = _snapshot(_build_worker_table(state, graph))

        assert "Elapsed" in text

    def test_eta_column_shows_unknown_when_no_history(self, graph: MikadoGraph) -> None:
        node = graph.add_node("simple task")
        state = _make_tui_state(node.id)
        text = _snapshot(_build_worker_table(state, graph))

        assert "~?" in text

    def test_files_column_shows_owned_files(self, graph: MikadoGraph) -> None:
        node = graph.add_node("task with files")
        graph.set_file_ownership(node.id, ["src/foo.py"])
        state = _make_tui_state(node.id)
        text = _snapshot(_build_worker_table(state, graph))

        assert "src/foo.py" in text

    def test_attempt_column_empty_on_first_attempt(self, graph: MikadoGraph) -> None:
        node = graph.add_node("fresh task")
        state = _make_tui_state(node.id)
        text = _snapshot(_build_worker_table(state, graph))

        assert "1/" not in text

    def test_attempt_column_shows_ratio_on_retry(self, graph: MikadoGraph) -> None:
        node = graph.add_node("retried task")
        state = TuiState(
            tick=0,
            active={"run-1": node.id},
            logs=[],
            dispatched_at={"run-1": time.monotonic()},
            attempts={node.id: 1},
            progress_by_run={},
            completion_durations=[],
            stall_threshold=300.0,
            max_retries=2,
            exec_agent="claude",
        )
        text = _snapshot(_build_worker_table(state, graph))

        assert "2/3" in text

    def test_description_is_single_line(self, graph: MikadoGraph) -> None:
        node = graph.add_node(_RICH_DESC)
        state = _make_tui_state(node.id)
        text = _snapshot(_build_worker_table(state, graph))

        lines_with_desc = [ln for ln in text.splitlines() if "split bundling" in ln]
        assert len(lines_with_desc) == 1


class _RalphWithStdout(FakeRalph):
    def __init__(self, lines: list[str]) -> None:
        super().__init__()
        self._lines = lines

    def get_run_stdout(self, run_id: str) -> list[str]:
        return self._lines


class TestRenderOverlayLogLines:
    def test_stdout_lines_appear_in_overlay(self, graph: MikadoGraph) -> None:
        node = graph.add_node("node-with-output")
        state = _make_tui_state(node.id)
        ralph = _RalphWithStdout(["line alpha", "line beta", "line gamma"])
        panel = _render_overlay("run-1", state, graph, ralph)
        text = _snapshot(panel)

        assert "line alpha" in text
        assert "line gamma" in text

    def test_only_last_100_stdout_lines_shown(self, graph: MikadoGraph) -> None:
        node = graph.add_node("many-lines")
        state = _make_tui_state(node.id)
        ralph = _RalphWithStdout([f"log-line-{i}" for i in range(150)])
        panel = _render_overlay("run-1", state, graph, ralph)
        text = _snapshot(panel)

        assert "log-line-149" in text
        assert "log-line-0" not in text

    def test_full_description_in_title_not_sanitized(self, graph: MikadoGraph) -> None:
        node = graph.add_node(_RICH_DESC)
        state = _make_tui_state(node.id)
        ralph = FakeRalph()
        panel = _render_overlay("run-1", state, graph, ralph)
        text = _snapshot(panel)

        assert "##" in text


# ---------------------------------------------------------------------------
# US-207: four integration paths
# ---------------------------------------------------------------------------


class TestStrictDrain:
    def test_no_new_dispatch_after_failure_in_flight_completes(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
        fake_git: FakeGit,
        fake_crg: FakeCrg,
    ) -> None:
        ralph = FakeRalph()
        ralph._success["run-1"] = False  # leaf-a fails, leaf-b (run-2) succeeds

        executor = Executor(graph=graph, git=fake_git, ralph=ralph, crg=fake_crg)
        loop = RunLoop(executor=executor, graph=graph, ralph=ralph)

        root = graph.add_node("root")
        graph.add_node("leaf-a", parent_id=root.id)
        graph.add_node("leaf-b", parent_id=root.id)

        result = loop.run(config, "main", strict=True)

        # Both leaves dispatched; root never dispatched after strict-failure
        assert result.dispatched_total == 2
        assert result.failed_total == 1
        assert result.completed_total == 1
        assert result.strict_exit is True
        assert result.root_done is False


class TestProtectedBranchGuard:
    @pytest.mark.skip(reason="protected-branch guard lives in app/run_command (β slice)")
    def test_protected_branch_raises_exit_2_before_log_created(
        self, tmp_path: Path
    ) -> None:
        import typer

        from milknado.app.run_command import _check_protected_branch
        from milknado.domains.common.config import MilknadoConfig

        cfg = MilknadoConfig(
            project_root=tmp_path,
            db_path=tmp_path / ".milknado" / "milknado.db",
            protected_branches=("main", "master"),
        )
        log_dir = tmp_path / ".milknado"

        with pytest.raises(typer.Exit) as exc_info:
            _check_protected_branch(cfg, "main", allow_protected=False)

        assert exc_info.value.exit_code == 2
        assert not any(log_dir.glob("run-*.log"))


class TestStalledWorkerGlyph:
    def test_stalled_worker_shows_warning_glyph_in_table(
        self, graph: MikadoGraph
    ) -> None:
        node = graph.add_node("long-running task")
        past = time.monotonic() - 400.0
        state = TuiState(
            tick=0,
            active={"run-1": node.id},
            logs=[],
            dispatched_at={"run-1": past},
            attempts={},
            progress_by_run={},
            completion_durations=[],
            stall_threshold=300.0,
            max_retries=2,
            exec_agent="claude",
        )
        rendered = _render_to_text(_build_worker_table(state, graph))

        assert "⚠" in rendered


class TestOrphanCleanupTransientRetries:
    def test_ensure_clean_worktree_called_each_attempt_no_stale_accumulation(
        self,
        graph: MikadoGraph,
        fake_git: FakeGit,
        fake_crg: FakeCrg,
        tmp_path: Path,
    ) -> None:
        from milknado.domains.common.errors import TransientDispatchError
        from milknado.domains.execution.executor import Executor as _Executor

        ralph = FakeRalph()
        call_count = 0
        original_generate = ralph.generate_ralph_md

        def fail_thrice(*args: Any, **kwargs: Any) -> Path:
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                raise TransientDispatchError("rate limited")
            return original_generate(*args, **kwargs)

        ralph.generate_ralph_md = fail_thrice  # type: ignore

        executor = _Executor(graph=graph, git=fake_git, ralph=ralph, crg=fake_crg)

        clean_calls: list[int] = []
        worktree_sizes: list[int] = []
        original_ensure = executor._ensure_clean_worktree

        def tracked_ensure(node_id: int) -> None:
            worktree_sizes.append(len(executor._worktrees))
            clean_calls.append(node_id)
            return original_ensure(node_id)

        executor._ensure_clean_worktree = tracked_ensure  # type: ignore

        retry_config = ExecutionConfig(
            execution_agent="claude",
            quality_gates=("uv run pytest",),
            worktree_pattern="milknado-{node_id}-{slug}",
            project_root=tmp_path,
            dispatch_max_retries=3,
            dispatch_backoff_seconds=0.0,
        )

        graph.add_node("transient node")
        executor.dispatch(1, retry_config)

        # Called once at the start of each _dispatch_once attempt (3 fail + 1 success = 4)
        assert len(clean_calls) == 4
        # At no point when _ensure_clean_worktree fires are there stale entries
        assert all(sz == 0 for sz in worktree_sizes)
        # Final state: successful dispatch recorded, not cleared
        assert 1 in executor._worktrees


class TestBuildLogPanel:
    def test_deque_maxlen_30_truncates_older_entries(self) -> None:
        logs: collections.deque[str] = collections.deque(maxlen=30)
        for i in range(40):
            logs.append(f"entry-{i}")
        text = _snapshot(_build_log_panel(logs))

        assert "entry-39" in text
        assert "entry-9" not in text

    def test_deque_within_maxlen_shows_all(self) -> None:
        logs: collections.deque[str] = collections.deque(maxlen=30)
        for i in range(20):
            logs.append(f"item-{i}")
        text = _snapshot(_build_log_panel(logs))

        assert "item-0" in text
        assert "item-19" in text

    def test_no_mid_line_cuts(self) -> None:
        complete_entry = "start:end"
        logs: collections.deque[str] = collections.deque(maxlen=30)
        logs.append(complete_entry)
        text = _snapshot(_build_log_panel(logs))

        assert "start:end" in text

    def test_empty_logs_shows_placeholder(self) -> None:
        text = _snapshot(_build_log_panel([]))

        assert "No events yet" in text


# ---------------------------------------------------------------------------
# Root completion via _maybe_verify_spec (not via dispatch)
# ---------------------------------------------------------------------------


class TestRootCompletionViaVerifySpec:
    def test_root_marked_done_by_verify_spec_after_all_leaves_done(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
        fake_git: FakeGit,
        fake_crg: FakeCrg,
    ) -> None:
        ralph = FakeRalph()
        executor = Executor(graph=graph, git=fake_git, ralph=ralph, crg=fake_crg)
        loop = RunLoop(executor=executor, graph=graph, ralph=ralph)

        root = graph.add_node("root goal")
        leaf = graph.add_node("leaf", parent_id=root.id)

        result = loop.run(config, "main", spec_text="spec: do the thing")

        assert result.root_done is True
        root_node = graph.get_node(root.id)
        leaf_node = graph.get_node(leaf.id)
        assert root_node is not None and root_node.status == NodeStatus.DONE
        assert leaf_node is not None and leaf_node.status == NodeStatus.DONE
        # Root was NOT dispatched — only the leaf was.
        assert result.dispatched_total == 1

    def test_root_not_dispatched_during_run(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
        fake_git: FakeGit,
        fake_crg: FakeCrg,
    ) -> None:
        dispatched_ids: list[int] = []
        ralph = FakeRalph()
        executor = Executor(graph=graph, git=fake_git, ralph=ralph, crg=fake_crg)
        original_dispatch = executor.dispatch

        def tracking_dispatch(node_id: int, cfg: Any) -> Any:
            dispatched_ids.append(node_id)
            return original_dispatch(node_id, cfg)

        executor.dispatch = tracking_dispatch  # type: ignore
        loop = RunLoop(executor=executor, graph=graph, ralph=ralph)

        root = graph.add_node("root goal")
        graph.add_node("leaf", parent_id=root.id)

        loop.run(config, "main", spec_text="spec: do the thing")

        assert root.id not in dispatched_ids

    def test_root_stays_pending_when_leaves_fail(
        self,
        graph: MikadoGraph,
        config: ExecutionConfig,
        fake_git: FakeGit,
        fake_crg: FakeCrg,
    ) -> None:
        ralph = FakeRalph()
        ralph._success["run-1"] = False

        executor = Executor(graph=graph, git=fake_git, ralph=ralph, crg=fake_crg)
        loop = RunLoop(executor=executor, graph=graph, ralph=ralph)

        root = graph.add_node("root goal")
        graph.add_node("leaf", parent_id=root.id)

        result = loop.run(config, "main", spec_text="spec: do the thing")

        assert result.root_done is False
        root_node = graph.get_node(root.id)
        assert root_node is not None
        assert root_node.status == NodeStatus.PENDING


# ---------------------------------------------------------------------------
# RalphifyAdapter.create_run passes log_dir
# ---------------------------------------------------------------------------


class TestRalphifyAdapterLogDir:
    def test_create_run_passes_log_dir_under_worktree(
        self, tmp_path: Path,
    ) -> None:
        from unittest.mock import MagicMock, patch

        ralph_dir = tmp_path / "wt-node-1"
        ralph_dir.mkdir()
        ralph_file = ralph_dir / "ralph.md"
        ralph_file.write_text("# task", encoding="utf-8")

        captured_configs: list[Any] = []

        def fake_create_run(config: Any) -> Any:
            captured_configs.append(config)
            fake_run = MagicMock()
            fake_run.state.run_id = "run-test"
            return fake_run

        from milknado.adapters.ralphify import RalphifyAdapter
        adapter = RalphifyAdapter(agent="claude")

        with patch.object(adapter._manager, "create_run", side_effect=fake_create_run):
            adapter.create_run(
                agent="claude",
                ralph_dir=ralph_dir,
                ralph_file=ralph_file,
                commands=[],
                quality_gates=[],
                project_root=None,
            )

        assert len(captured_configs) == 1
        cfg = captured_configs[0]
        assert cfg.log_dir == ralph_dir / ".ralph-logs"
