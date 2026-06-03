import itertools
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer
from typer.testing import CliRunner

from milknado.cli import app
from milknado.cli_tools import (
    _write_claude_worker_settings,
    _write_gemini_worker_settings,
    _write_worker_hooks,
)
from milknado.domains.common import default_config
from milknado.domains.common.agent_argv import WORKER_ALLOWED_TOOLS

FIXTURES = Path(__file__).parent / "fixtures"

runner = CliRunner()


def _unique_run_factory() -> MagicMock:
    counter = itertools.count(1)

    def _create_run(*_args: object, **_kwargs: object) -> MagicMock:
        run = MagicMock()
        run.state.run_id = f"run-{next(counter)}"
        return run

    mock = MagicMock(side_effect=_create_run)
    return mock


def _configure_ralph_mocks(
    ralph_cls: MagicMock,
    project_dir: Path,
    *,
    unique: bool = False,
) -> None:
    if unique:
        ralph_cls.return_value.create_run = _unique_run_factory()
    else:
        fake_run = MagicMock()
        fake_run.state.run_id = "run-1"
        ralph_cls.return_value.create_run.return_value = fake_run
    ralph_cls.return_value.generate_ralph_md.return_value = project_dir / "RALPH.md"

    def _wait_for_next_completion(
        active_run_ids: set[str],
        timeout: float | None = None,
    ) -> tuple[str, bool]:
        return next(iter(active_run_ids)), True

    ralph_cls.return_value.wait_for_next_completion.side_effect = _wait_for_next_completion
    ralph_cls.return_value.poll_progress_events.return_value = []


@pytest.fixture()
def project_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture()
def mock_adapters():
    with (
        patch("milknado.adapters.RalphifyAdapter") as ralph,
        patch("milknado.adapters.GitAdapter") as git,
        patch("milknado.adapters.CrgAdapter") as crg,
    ):
        crg.return_value.get_impact_radius.return_value = {}
        yield ralph, git, crg


class TestInit:
    @patch("milknado.adapters.crg.CrgAdapter")
    def test_creates_config_and_db(self, _mock_crg: MagicMock, project_dir: Path) -> None:
        result = runner.invoke(app, ["init", str(project_dir)])
        assert result.exit_code == 0
        assert (project_dir / "milknado.toml").exists()
        assert (project_dir / ".milknado" / "milknado.db").exists()

    @patch("milknado.adapters.crg.CrgAdapter")
    def test_idempotent(self, _mock_crg: MagicMock, project_dir: Path) -> None:
        runner.invoke(app, ["init", str(project_dir)])
        result = runner.invoke(app, ["init", str(project_dir)])
        assert result.exit_code == 0
        assert "already exists" in result.output

    @patch("milknado.adapters.crg.CrgAdapter")
    def test_config_has_defaults(self, _mock_crg: MagicMock, project_dir: Path) -> None:
        runner.invoke(app, ["init", str(project_dir)])
        content = (project_dir / "milknado.toml").read_text()
        assert "agent_family" in content
        assert "planning_agent" in content
        assert "execution_agent" in content
        assert "quality_gates" in content
        assert "concurrency_limit" in content

    @patch("milknado.adapters.crg.CrgAdapter")
    def test_calls_ensure_graph(self, mock_crg_cls: MagicMock, project_dir: Path) -> None:
        result = runner.invoke(app, ["init", str(project_dir)])
        assert result.exit_code == 0
        mock_crg_cls.return_value.ensure_graph.assert_called_once_with(project_dir)
        assert "Code-review-graph ready" in result.output

    @patch("milknado.adapters.crg.CrgAdapter")
    def test_init_crg_build_failure_exits(
        self, mock_crg_cls: MagicMock, project_dir: Path
    ) -> None:
        mock_crg_cls.return_value.ensure_graph.side_effect = subprocess.CalledProcessError(
            returncode=2, cmd="crg", stderr="boom"
        )
        result = runner.invoke(app, ["init", str(project_dir)])
        assert result.exit_code == 1
        assert "CRG build failed (exit 2)" in result.output
        assert "boom" in result.output


class TestIndex:
    @patch("milknado.adapters.crg.CrgAdapter")
    def test_index_rebuilds(self, mock_crg_cls: MagicMock, project_dir: Path) -> None:
        runner.invoke(app, ["init", str(project_dir)])
        result = runner.invoke(app, ["index", str(project_dir)])
        assert result.exit_code == 0
        assert "Code-review-graph rebuilt." in result.output
        mock_crg_cls.return_value.build_graph.assert_called_once_with(project_dir)

    @patch("milknado.adapters.crg.CrgAdapter")
    def test_index_crg_build_failure_exits(
        self, mock_crg_cls: MagicMock, project_dir: Path
    ) -> None:
        runner.invoke(app, ["init", str(project_dir)])
        mock_crg_cls.return_value.build_graph.side_effect = subprocess.CalledProcessError(
            returncode=3, cmd="crg", stderr="index boom"
        )
        result = runner.invoke(app, ["index", str(project_dir)])
        assert result.exit_code == 1
        assert "CRG build failed (exit 3)" in result.output
        assert "index boom" in result.output


class TestCrgCommand:
    def test_crg_no_index_exits(self, project_dir: Path) -> None:
        result = runner.invoke(app, ["crg", str(project_dir)])
        assert result.exit_code == 1
        assert "No CRG index" in result.output

    @patch("milknado.adapters.crg.CrgAdapter")
    def test_crg_prints_overview(self, mock_crg_cls: MagicMock, project_dir: Path) -> None:
        (project_dir / ".code-review-graph").mkdir(parents=True)
        (project_dir / ".code-review-graph" / "graph.db").write_text("", encoding="utf-8")
        mock_crg_cls.return_value.get_architecture_overview.return_value = {"hubs": ["a"]}
        result = runner.invoke(app, ["crg", str(project_dir)])
        assert result.exit_code == 0
        assert '"hubs"' in result.output


class TestStatus:
    def test_empty_graph(self, project_dir: Path) -> None:
        runner.invoke(app, ["init", str(project_dir)])
        result = runner.invoke(app, ["status", str(project_dir)])
        assert result.exit_code == 0
        assert "No nodes" in result.output

    def test_shows_nodes(self, project_dir: Path) -> None:
        runner.invoke(app, ["init", str(project_dir)])
        runner.invoke(
            app,
            ["add-node", "root goal", "--project-root", str(project_dir)],
        )
        result = runner.invoke(app, ["status", str(project_dir)])
        assert result.exit_code == 0
        assert "root goal" in result.output
        assert "1/1" not in result.output  # pending, not done
        assert "0/1" in result.output

    def test_shows_progress(self, project_dir: Path) -> None:
        from milknado.domains.common import default_config
        from milknado.domains.graph import MikadoGraph

        runner.invoke(app, ["init", str(project_dir)])
        config = default_config(project_dir)
        graph = MikadoGraph(config.db_path)
        graph.add_node("root")
        child = graph.add_node("child", parent_id=1)
        graph.mark_running(child.id)
        graph.mark_done(child.id)
        graph.close()

        result = runner.invoke(app, ["status", str(project_dir)])
        assert result.exit_code == 0
        assert "1/2" in result.output
        assert "50%" in result.output

    def test_shows_tree_structure(self, project_dir: Path) -> None:
        from milknado.domains.common import default_config
        from milknado.domains.graph import MikadoGraph

        runner.invoke(app, ["init", str(project_dir)])
        config = default_config(project_dir)
        graph = MikadoGraph(config.db_path)
        root = graph.add_node("Root goal")
        graph.add_node("Leaf A", parent_id=root.id)
        graph.add_node("Leaf B", parent_id=root.id)
        graph.close()

        result = runner.invoke(app, ["status", str(project_dir)])
        assert result.exit_code == 0
        assert "Root goal" in result.output
        assert "Leaf A" in result.output
        assert "Leaf B" in result.output
        assert "Ready" in result.output

    def test_shows_file_conflicts(self, project_dir: Path) -> None:
        from milknado.domains.common import default_config
        from milknado.domains.graph import MikadoGraph

        runner.invoke(app, ["init", str(project_dir)])
        config = default_config(project_dir)
        graph = MikadoGraph(config.db_path)
        root = graph.add_node("Root")
        a = graph.add_node("Node A", parent_id=root.id)
        b = graph.add_node("Node B", parent_id=root.id)
        graph.set_file_ownership(a.id, ["shared.py"])
        graph.set_file_ownership(b.id, ["shared.py"])
        graph.close()

        result = runner.invoke(app, ["status", str(project_dir)])
        assert result.exit_code == 0
        assert "Conflict" in result.output
        assert "shared.py" in result.output

    def test_shows_running_worktree(self, project_dir: Path) -> None:
        from milknado.domains.common import default_config
        from milknado.domains.graph import MikadoGraph

        runner.invoke(app, ["init", str(project_dir)])
        config = default_config(project_dir)
        graph = MikadoGraph(config.db_path)
        root = graph.add_node("Root")
        c = graph.add_node("Worker", parent_id=root.id)
        graph.mark_running(c.id)
        graph._conn.execute(
            "UPDATE nodes SET worktree_path = ? WHERE id = ?",
            ("/tmp/milknado-wt", c.id),
        )
        graph._conn.commit()
        graph.close()

        result = runner.invoke(app, ["status", str(project_dir)])
        assert result.exit_code == 0
        assert "/tmp/milknado-wt" in result.output

    @patch("milknado.adapters.ralphify.RalphifyAdapter")
    def test_enriches_running_node_with_run_state(
        self, mock_ralph_cls: MagicMock, project_dir: Path
    ) -> None:
        from milknado.domains.common import default_config
        from milknado.domains.graph import MikadoGraph

        runner.invoke(app, ["init", str(project_dir)])
        config = default_config(project_dir)
        graph = MikadoGraph(config.db_path)
        root = graph.add_node("Root")
        worker = graph.add_node("Worker", parent_id=root.id)
        graph.mark_running(worker.id, run_id="run-42")
        graph.close()

        fake_run = MagicMock()
        fake_run.status = "in_progress"
        mock_ralph_cls.return_value.get_run.return_value = fake_run

        result = runner.invoke(app, ["status", str(project_dir)])
        assert result.exit_code == 0
        mock_ralph_cls.return_value.get_run.assert_called_once_with("run-42")

    @patch("milknado.adapters.ralphify.RalphifyAdapter")
    def test_run_state_enrichment_is_best_effort(
        self, mock_ralph_cls: MagicMock, project_dir: Path
    ) -> None:
        from milknado.domains.common import default_config
        from milknado.domains.graph import MikadoGraph

        runner.invoke(app, ["init", str(project_dir)])
        config = default_config(project_dir)
        graph = MikadoGraph(config.db_path)
        root = graph.add_node("Root")
        worker = graph.add_node("Worker", parent_id=root.id)
        graph.mark_running(worker.id, run_id="run-99")
        graph.close()

        mock_ralph_cls.return_value.get_run.side_effect = RuntimeError("ralph down")

        result = runner.invoke(app, ["status", str(project_dir)])
        # Enrichment failure must not break the status render.
        assert result.exit_code == 0
        assert "Worker" in result.output


class TestAddNode:
    def test_add_root(self, project_dir: Path) -> None:
        runner.invoke(app, ["init", str(project_dir)])
        result = runner.invoke(
            app,
            ["add-node", "my goal", "--project-root", str(project_dir)],
        )
        assert result.exit_code == 0
        assert "Added node" in result.output

    def test_add_child(self, project_dir: Path) -> None:
        runner.invoke(app, ["init", str(project_dir)])
        runner.invoke(
            app,
            ["add-node", "parent", "--project-root", str(project_dir)],
        )
        result = runner.invoke(
            app,
            [
                "add-node",
                "child",
                "--parent",
                "1",
                "--project-root",
                str(project_dir),
            ],
        )
        assert result.exit_code == 0
        assert "Added node 2" in result.output

    def test_add_with_files(self, project_dir: Path) -> None:
        from milknado.domains.common import default_config
        from milknado.domains.graph import MikadoGraph

        runner.invoke(app, ["init", str(project_dir)])
        result = runner.invoke(
            app,
            [
                "add-node",
                "refactor auth",
                "--files",
                "src/auth.py",
                "--files",
                "src/login.py",
                "--project-root",
                str(project_dir),
            ],
        )
        assert result.exit_code == 0

        config = default_config(project_dir)
        graph = MikadoGraph(config.db_path)
        files = graph.get_file_ownership(1)
        graph.close()
        assert set(files) == {"src/auth.py", "src/login.py"}

    def test_blocks_running_parent(self, project_dir: Path) -> None:
        from milknado.domains.common import NodeStatus, default_config
        from milknado.domains.graph import MikadoGraph

        runner.invoke(app, ["init", str(project_dir)])
        runner.invoke(
            app,
            ["add-node", "parent", "--project-root", str(project_dir)],
        )

        config = default_config(project_dir)
        graph = MikadoGraph(config.db_path)
        graph.mark_running(1)
        graph.close()

        result = runner.invoke(
            app,
            [
                "add-node",
                "prereq",
                "--parent",
                "1",
                "--project-root",
                str(project_dir),
            ],
        )
        assert result.exit_code == 0
        assert "blocked" in result.output

        graph = MikadoGraph(config.db_path)
        parent = graph.get_node(1)
        graph.close()
        assert parent is not None
        assert parent.status == NodeStatus.BLOCKED


class TestPlanCommand:
    @patch("milknado.adapters.crg.CrgAdapter")
    @patch("milknado.domains.planning.Planner")
    def test_plan_success(
        self,
        mock_planner_cls: MagicMock,
        mock_crg_cls: MagicMock,
        project_dir: Path,
    ) -> None:
        mock_planner_cls.return_value.launch.return_value = _make_plan_result()
        result = runner.invoke(
            app,
            ["plan", "--spec", str(FIXTURES / "valid.md"), "--project-root", str(project_dir)],
        )
        assert result.exit_code == 0
        assert "Planning" in result.output

    @patch("milknado.adapters.crg.CrgAdapter")
    @patch("milknado.domains.planning.Planner")
    def test_plan_failure(
        self,
        mock_planner_cls: MagicMock,
        mock_crg_cls: MagicMock,
        project_dir: Path,
    ) -> None:
        mock_planner_cls.return_value.launch.return_value = _make_plan_result(
            success=False,
            exit_code=1,
            solver_status="NO_MANIFEST",
        )
        result = runner.invoke(
            app,
            ["plan", "--spec", str(FIXTURES / "valid.md"), "--project-root", str(project_dir)],
        )
        assert result.exit_code == 1

    @patch("milknado.adapters.crg.CrgAdapter")
    @patch("milknado.domains.planning.Planner")
    def test_plan_passes_prompt_prepend_from_config(
        self,
        mock_planner_cls: MagicMock,
        mock_crg_cls: MagicMock,
        project_dir: Path,
    ) -> None:
        """The plan command must wire config.planning_prompt_prepend into Planner.

        Pins the regression from 19d53ec: a CLI refactor dropped this kwarg, so
        the user's planning_prepend config was silently ignored. If the argument
        is dropped again, the kwarg assertion below fails.
        """
        (project_dir / "milknado.toml").write_text(
            "[milknado]\n"
            'agent_family = "claude"\n\n'
            "[milknado.prompts]\n"
            'planning_prepend = "team rule X"\n',
            encoding="utf-8",
        )
        mock_planner_cls.return_value.launch.return_value = _make_plan_result()

        result = runner.invoke(
            app,
            ["plan", "--spec", str(FIXTURES / "valid.md"), "--project-root", str(project_dir)],
        )

        assert result.exit_code == 0, result.output
        mock_planner_cls.assert_called_once()
        assert mock_planner_cls.call_args.kwargs["prompt_prepend"] == "team rule X"


class TestPlanInteractive:
    @patch("milknado.adapters.crg.CrgAdapter")
    @patch("milknado.domains.planning.Planner")
    def test_accept_first_iteration(
        self,
        mock_planner_cls: MagicMock,
        _mock_crg_cls: MagicMock,
        project_dir: Path,
    ) -> None:
        mock_planner_cls.return_value.launch.return_value = _make_plan_result(
            solver_status="OPTIMAL",
            change_count=3,
            batch_count=2,
        )
        result = runner.invoke(
            app,
            [
                "plan",
                "--interactive",
                "--spec",
                str(FIXTURES / "valid.md"),
                "--project-root",
                str(project_dir),
            ],
            input="1\n",
        )
        assert result.exit_code == 0, result.output
        assert "Plan iteration 1" in result.output
        assert "Choose next step" in result.output
        mock_planner_cls.return_value.launch.assert_called_once()

    @patch("milknado.adapters.crg.CrgAdapter")
    @patch("milknado.domains.planning.Planner")
    def test_revise_then_accept(
        self,
        mock_planner_cls: MagicMock,
        _mock_crg_cls: MagicMock,
        project_dir: Path,
    ) -> None:
        first = _make_plan_result(solver_status="NO_MANIFEST", success=False, exit_code=1)
        second = _make_plan_result(solver_status="OPTIMAL", success=True, exit_code=0)
        mock_planner_cls.return_value.launch.side_effect = [first, second]
        result = runner.invoke(
            app,
            [
                "plan",
                "--interactive",
                "--spec",
                str(FIXTURES / "valid.md"),
                "--project-root",
                str(project_dir),
            ],
            input="2\nAdd explicit API changes and ordering constraints.\n1\n",
        )
        assert result.exit_code == 0, result.output
        assert "Planner output was invalid for execution" in result.output
        assert "Plan iteration 2" in result.output
        assert mock_planner_cls.return_value.launch.call_count == 2
        second_goal = mock_planner_cls.return_value.launch.call_args_list[1].args[0]
        assert "User revision request" in second_goal
        assert "Add explicit API changes and ordering constraints." in second_goal
        assert "Prior planner result" in second_goal

    @patch("milknado.adapters.crg.CrgAdapter")
    @patch("milknado.domains.planning.Planner")
    def test_cancel_exits_one(
        self,
        mock_planner_cls: MagicMock,
        _mock_crg_cls: MagicMock,
        project_dir: Path,
    ) -> None:
        mock_planner_cls.return_value.launch.return_value = _make_plan_result()
        result = runner.invoke(
            app,
            [
                "plan",
                "--interactive",
                "--spec",
                str(FIXTURES / "valid.md"),
                "--project-root",
                str(project_dir),
            ],
            input="3\n",
        )
        assert result.exit_code == 1
        assert "Choose next step" in result.output

    @patch("milknado.adapters.crg.CrgAdapter")
    @patch("milknado.domains.planning.Planner")
    def test_invalid_selection_reprompts(
        self,
        mock_planner_cls: MagicMock,
        _mock_crg_cls: MagicMock,
        project_dir: Path,
    ) -> None:
        mock_planner_cls.return_value.launch.return_value = _make_plan_result()
        result = runner.invoke(
            app,
            [
                "plan",
                "--interactive",
                "--spec",
                str(FIXTURES / "valid.md"),
                "--project-root",
                str(project_dir),
            ],
            input="9\n1\n",
        )
        assert result.exit_code == 0, result.output
        assert "Please enter 1, 2, or 3." in result.output

    @patch("milknado.adapters.crg.CrgAdapter")
    @patch("milknado.domains.planning.Planner")
    def test_context_path_and_unknown_warning_printed(
        self,
        mock_planner_cls: MagicMock,
        _mock_crg_cls: MagicMock,
        project_dir: Path,
    ) -> None:
        mock_planner_cls.return_value.launch.return_value = _make_plan_result(
            solver_status="UNKNOWN",
            batch_count=1,
            context_path=str(project_dir / "ctx.md"),
        )
        result = runner.invoke(
            app,
            [
                "plan",
                "--interactive",
                "--spec",
                str(FIXTURES / "valid.md"),
                "--project-root",
                str(project_dir),
            ],
            input="1\n",
        )
        assert result.exit_code == 0, result.output
        assert "Planner context:" in result.output
        assert "solver returned UNKNOWN" in result.output

    @patch("milknado.adapters.crg.CrgAdapter")
    @patch("milknado.domains.planning.Planner")
    def test_accept_despite_invalid_status(
        self,
        mock_planner_cls: MagicMock,
        _mock_crg_cls: MagicMock,
        project_dir: Path,
    ) -> None:
        mock_planner_cls.return_value.launch.return_value = _make_plan_result(
            solver_status="NO_MANIFEST",
            success=False,
            exit_code=1,
        )
        result = runner.invoke(
            app,
            [
                "plan",
                "--interactive",
                "--spec",
                str(FIXTURES / "valid.md"),
                "--project-root",
                str(project_dir),
            ],
            input="1\n",
        )
        assert result.exit_code == 0, result.output
        assert "Accepting current plan despite invalid planner status." in result.output

    @patch("milknado.adapters.crg.CrgAdapter")
    @patch("milknado.domains.planning.Planner")
    def test_empty_feedback_keeps_goal_then_accept(
        self,
        mock_planner_cls: MagicMock,
        _mock_crg_cls: MagicMock,
        project_dir: Path,
    ) -> None:
        mock_planner_cls.return_value.launch.return_value = _make_plan_result()
        result = runner.invoke(
            app,
            [
                "plan",
                "--interactive",
                "--spec",
                str(FIXTURES / "valid.md"),
                "--project-root",
                str(project_dir),
            ],
            input="2\n \n1\n",
        )
        assert result.exit_code == 0, result.output
        assert "Empty feedback; keeping original goal for next iteration." in result.output
        # Goal unchanged across both iterations (empty feedback does not rebuild it).
        first_goal = mock_planner_cls.return_value.launch.call_args_list[0].args[0]
        second_goal = mock_planner_cls.return_value.launch.call_args_list[1].args[0]
        assert first_goal == second_goal

    @patch("milknado.adapters.crg.CrgAdapter")
    @patch("milknado.domains.planning.Planner")
    def test_max_iterations_without_acceptance_exits_one(
        self,
        mock_planner_cls: MagicMock,
        _mock_crg_cls: MagicMock,
        project_dir: Path,
    ) -> None:
        mock_planner_cls.return_value.launch.return_value = _make_plan_result()
        result = runner.invoke(
            app,
            [
                "plan",
                "--interactive",
                "--max-iterations",
                "2",
                "--spec",
                str(FIXTURES / "valid.md"),
                "--project-root",
                str(project_dir),
            ],
            input="2\nrevise once\n2\nrevise again\n",
        )
        assert result.exit_code == 1, result.output
        assert "Reached max iterations (2) without acceptance." in result.output


class TestIssueHelpers:
    @patch("milknado.cli_plan.subprocess.run")
    def test_fetch_issue_invalid_json_exits(self, mock_run: MagicMock) -> None:
        from milknado.cli_plan import _fetch_issue

        mock_run.return_value = MagicMock(returncode=0, stdout="not json", stderr="")
        with pytest.raises(typer.Exit) as exc:
            _fetch_issue("42")
        assert exc.value.exit_code == 1

    def test_issue_title_falls_back_to_number(self) -> None:
        from milknado.cli_plan import _issue_title

        assert _issue_title({"number": 7}) == "Issue 7"
        assert _issue_title({}) == "Issue"

    def test_materialize_issue_spec_rejects_empty_refs(self, project_dir: Path) -> None:
        from milknado.cli_plan import _materialize_issue_spec

        with pytest.raises(ValueError, match="issue_refs must not be empty"):
            _materialize_issue_spec([], project_dir)


def _make_plan_result(**kwargs: object) -> MagicMock:
    """Build a PlanResult-like mock with sensible defaults."""
    from milknado.domains.planning.planner import PlanResult

    defaults: dict[str, object] = {
        "success": True,
        "exit_code": 0,
        "context_path": None,
        "nodes_created": 3,
        "batch_count": 2,
        "oversized_count": 0,
        "solver_status": "OPTIMAL",
        "change_count": 4,
    }
    for k, v in kwargs.items():
        defaults[k] = v
    return PlanResult(**defaults)  # ty: ignore[invalid-argument-type, invalid-return-type]


class TestPlanSpecOption:
    def test_missing_spec_exits_nonzero(self) -> None:
        result = runner.invoke(app, ["plan"])
        assert result.exit_code != 0

    def test_nonexistent_file_exits_nonzero(self) -> None:
        result = runner.invoke(app, ["plan", "--spec", "/nonexistent/file.md"])
        assert result.exit_code != 0

    def test_non_md_extension_exits_1(self, project_dir: Path) -> None:
        result = runner.invoke(
            app,
            ["plan", "--spec", str(FIXTURES / "not_md.txt"), "--project-root", str(project_dir)],
        )
        assert result.exit_code == 1
        assert ".md" in result.output

    @patch("milknado.adapters.crg.CrgAdapter")
    @patch("milknado.domains.planning.Planner")
    def test_happy_path_exit_0_summary_printed(
        self,
        mock_planner_cls: MagicMock,
        mock_crg_cls: MagicMock,
        project_dir: Path,
    ) -> None:
        mock_planner_cls.return_value.launch.return_value = _make_plan_result()
        result = runner.invoke(
            app,
            ["plan", "--spec", str(FIXTURES / "valid.md"), "--project-root", str(project_dir)],
        )
        assert result.exit_code == 0
        assert "solver=" in result.output
        assert "created" in result.output

    @patch("milknado.adapters.crg.CrgAdapter")
    @patch("milknado.domains.planning.Planner")
    def test_heading_derived_as_goal(
        self,
        mock_planner_cls: MagicMock,
        mock_crg_cls: MagicMock,
        project_dir: Path,
    ) -> None:
        mock_planner_cls.return_value.launch.return_value = _make_plan_result()
        result = runner.invoke(
            app,
            ["plan", "--spec", str(FIXTURES / "valid.md"), "--project-root", str(project_dir)],
        )
        assert result.exit_code == 0
        assert "My Feature Goal" in result.output
        mock_planner_cls.return_value.launch.assert_called_once()
        call_args = mock_planner_cls.return_value.launch.call_args
        goal_arg = call_args.kwargs.get("goal") or (call_args.args[0] if call_args.args else None)
        assert goal_arg == "My Feature Goal"

    @patch("milknado.adapters.crg.CrgAdapter")
    @patch("milknado.domains.planning.Planner")
    def test_no_heading_uses_filename_stem(
        self,
        mock_planner_cls: MagicMock,
        mock_crg_cls: MagicMock,
        project_dir: Path,
    ) -> None:
        mock_planner_cls.return_value.launch.return_value = _make_plan_result()
        result = runner.invoke(
            app,
            [
                "plan",
                "--spec",
                str(FIXTURES / "no_heading.md"),
                "--project-root",
                str(project_dir),
            ],
        )
        assert result.exit_code == 0
        assert "no_heading" in result.output

    @patch("milknado.adapters.crg.CrgAdapter")
    @patch("milknado.domains.planning.Planner")
    def test_optimal_solver_exits_0_summary_has_solver(
        self,
        mock_planner_cls: MagicMock,
        mock_crg_cls: MagicMock,
        project_dir: Path,
    ) -> None:
        mock_planner_cls.return_value.launch.return_value = _make_plan_result(
            solver_status="OPTIMAL"
        )
        result = runner.invoke(
            app,
            ["plan", "--spec", str(FIXTURES / "valid.md"), "--project-root", str(project_dir)],
        )
        assert result.exit_code == 0
        assert "solver=OPTIMAL" in result.output

    @patch("milknado.adapters.crg.CrgAdapter")
    @patch("milknado.domains.planning.Planner")
    def test_one_oversized_summary(
        self,
        mock_planner_cls: MagicMock,
        mock_crg_cls: MagicMock,
        project_dir: Path,
    ) -> None:
        mock_planner_cls.return_value.launch.return_value = _make_plan_result(oversized_count=1)
        result = runner.invoke(
            app,
            ["plan", "--spec", str(FIXTURES / "valid.md"), "--project-root", str(project_dir)],
        )
        assert result.exit_code == 0
        assert "(1 oversized)" in result.output

    @patch("milknado.adapters.crg.CrgAdapter")
    @patch("milknado.domains.planning.Planner")
    def test_infeasible_exits_1(
        self,
        mock_planner_cls: MagicMock,
        mock_crg_cls: MagicMock,
        project_dir: Path,
    ) -> None:
        mock_planner_cls.return_value.launch.return_value = _make_plan_result(
            solver_status="INFEASIBLE",
            success=False,
            exit_code=0,
        )
        result = runner.invoke(
            app,
            ["plan", "--spec", str(FIXTURES / "valid.md"), "--project-root", str(project_dir)],
        )
        assert result.exit_code == 1
        assert "solver=INFEASIBLE" in result.output

    @patch("milknado.adapters.crg.CrgAdapter")
    @patch("milknado.domains.planning.Planner")
    def test_unknown_with_batches_exits_0_stderr_warning(
        self,
        mock_planner_cls: MagicMock,
        mock_crg_cls: MagicMock,
        project_dir: Path,
    ) -> None:
        mock_planner_cls.return_value.launch.return_value = _make_plan_result(
            solver_status="UNKNOWN",
            batch_count=2,
        )
        result = runner.invoke(
            app,
            ["plan", "--spec", str(FIXTURES / "valid.md"), "--project-root", str(project_dir)],
            catch_exceptions=False,
        )
        assert result.exit_code == 0

    @patch("milknado.adapters.crg.CrgAdapter")
    @patch("milknado.domains.planning.Planner")
    def test_crg_failure_degradation_still_runs(
        self,
        mock_planner_cls: MagicMock,
        mock_crg_cls: MagicMock,
        project_dir: Path,
    ) -> None:
        mock_crg_cls.return_value.ensure_graph.side_effect = RuntimeError("crg failed")
        mock_planner_cls.return_value.launch.return_value = _make_plan_result()
        result = runner.invoke(
            app,
            ["plan", "--spec", str(FIXTURES / "valid.md"), "--project-root", str(project_dir)],
        )
        assert result.exit_code == 0
        assert "solver=" in result.output


class TestPlanIssueOption:
    @staticmethod
    def _gh_ok(title: str = "Add --issue support", body: str = "Body text") -> MagicMock:
        import json as _json

        completed = MagicMock()
        completed.returncode = 0
        completed.stdout = _json.dumps(
            {
                "title": title,
                "body": body,
                "number": 42,
                "url": "https://example.com/issues/42",
            }
        )
        completed.stderr = ""
        return completed

    @patch("milknado.cli_plan.subprocess.run")
    @patch("milknado.adapters.crg.CrgAdapter")
    @patch("milknado.domains.planning.Planner")
    def test_issue_fetches_and_runs_planner(
        self,
        mock_planner_cls: MagicMock,
        _mock_crg_cls: MagicMock,
        mock_run: MagicMock,
        project_dir: Path,
    ) -> None:
        mock_run.return_value = self._gh_ok()
        mock_planner_cls.return_value.launch.return_value = _make_plan_result()

        result = runner.invoke(
            app,
            ["plan", "--issue", "42", "--project-root", str(project_dir)],
        )

        assert result.exit_code == 0, result.output
        assert "Add --issue support" in result.output  # title becomes goal
        mock_run.assert_called_once()
        argv = mock_run.call_args.args[0]
        assert argv[:3] == ["gh", "issue", "view"]
        assert "42" in argv
        spec_file = project_dir / ".milknado" / "issues" / "issue-42.md"
        assert spec_file.exists()
        assert spec_file.read_text().startswith("# Add --issue support")

    @patch("milknado.cli_plan.subprocess.run")
    @patch("milknado.adapters.crg.CrgAdapter")
    @patch("milknado.domains.planning.Planner")
    def test_issue_and_spec_combine_into_one_plan(
        self,
        mock_planner_cls: MagicMock,
        _mock_crg_cls: MagicMock,
        mock_run: MagicMock,
        project_dir: Path,
    ) -> None:
        mock_run.return_value = self._gh_ok(title="Issue title")
        mock_planner_cls.return_value.launch.return_value = _make_plan_result()

        result = runner.invoke(
            app,
            [
                "plan",
                "--spec",
                str(FIXTURES / "valid.md"),
                "--issue",
                "42",
                "--project-root",
                str(project_dir),
            ],
        )

        assert result.exit_code == 0, result.output
        spec_file = project_dir / ".milknado" / "issues" / "plan-valid-42.md"
        assert spec_file.exists()
        content = spec_file.read_text()
        assert content.startswith("# Plan for specs valid + issues #42")
        assert "## Spec: valid" in content
        assert "My Feature Goal" in content  # from fixture body
        assert "## #42: Issue title" in content

    def test_plan_with_neither_exits_one(self, project_dir: Path) -> None:
        result = runner.invoke(
            app,
            ["plan", "--project-root", str(project_dir)],
        )
        assert result.exit_code == 1
        assert "--spec or --issue" in result.output

    @patch("milknado.cli_plan.subprocess.run")
    def test_issue_gh_failure_exits_one(
        self,
        mock_run: MagicMock,
        project_dir: Path,
    ) -> None:
        failed = MagicMock()
        failed.returncode = 1
        failed.stdout = ""
        failed.stderr = "no such issue"
        mock_run.return_value = failed

        result = runner.invoke(
            app,
            ["plan", "--issue", "99999", "--project-root", str(project_dir)],
        )
        assert result.exit_code == 1
        assert "no such issue" in result.output

    @patch("milknado.cli.subprocess.run", side_effect=FileNotFoundError())
    def test_issue_gh_not_installed_exits_one(
        self,
        _mock_run: MagicMock,
        project_dir: Path,
    ) -> None:
        result = runner.invoke(
            app,
            ["plan", "--issue", "42", "--project-root", str(project_dir)],
        )
        assert result.exit_code == 1
        assert "gh" in result.output.lower()

    @patch("milknado.cli_plan.subprocess.run")
    @patch("milknado.adapters.crg.CrgAdapter")
    @patch("milknado.domains.planning.Planner")
    def test_multiple_issues_merged_into_one_spec(
        self,
        mock_planner_cls: MagicMock,
        _mock_crg_cls: MagicMock,
        mock_run: MagicMock,
        project_dir: Path,
    ) -> None:
        import json as _json

        def _gh(title: str, number: int, body: str) -> MagicMock:
            completed = MagicMock()
            completed.returncode = 0
            completed.stdout = _json.dumps(
                {
                    "title": title,
                    "body": body,
                    "number": number,
                    "url": f"https://example.com/issues/{number}",
                }
            )
            completed.stderr = ""
            return completed

        mock_run.side_effect = [
            _gh("First issue", 42, "Body A"),
            _gh("Second issue", 43, "Body B"),
        ]
        mock_planner_cls.return_value.launch.return_value = _make_plan_result()

        result = runner.invoke(
            app,
            [
                "plan",
                "--issue",
                "42",
                "--issue",
                "43",
                "--project-root",
                str(project_dir),
            ],
        )

        assert result.exit_code == 0, result.output
        assert mock_run.call_count == 2
        spec_file = project_dir / ".milknado" / "issues" / "issue-42-43.md"
        assert spec_file.exists()
        content = spec_file.read_text()
        assert content.startswith("# Plan for issues #42, #43")
        assert "## #42: First issue" in content
        assert "Body A" in content
        assert "## #43: Second issue" in content
        assert "Body B" in content
        # Goal derived from combined heading
        assert "Plan for issues #42, #43" in result.output

    @patch("milknado.cli_plan.subprocess.run")
    @patch("milknado.adapters.crg.CrgAdapter")
    @patch("milknado.domains.planning.Planner")
    def test_comma_separated_issues_accepted(
        self,
        mock_planner_cls: MagicMock,
        _mock_crg_cls: MagicMock,
        mock_run: MagicMock,
        project_dir: Path,
    ) -> None:
        import json as _json

        def _gh(title: str, number: int) -> MagicMock:
            c = MagicMock()
            c.returncode = 0
            c.stdout = _json.dumps(
                {
                    "title": title,
                    "body": f"body {number}",
                    "number": number,
                    "url": f"https://example.com/issues/{number}",
                }
            )
            c.stderr = ""
            return c

        mock_run.side_effect = [_gh("A", 42), _gh("B", 43), _gh("C", 44)]
        mock_planner_cls.return_value.launch.return_value = _make_plan_result()

        result = runner.invoke(
            app,
            [
                "plan",
                "--issue",
                "42,43",
                "--issue",
                "44",
                "--project-root",
                str(project_dir),
            ],
        )

        assert result.exit_code == 0, result.output
        assert mock_run.call_count == 3
        spec_file = project_dir / ".milknado" / "issues" / "issue-42-43-44.md"
        assert spec_file.exists()

    @patch("milknado.adapters.crg.CrgAdapter")
    @patch("milknado.domains.planning.Planner")
    def test_multiple_specs_comma_separated_merged(
        self,
        mock_planner_cls: MagicMock,
        _mock_crg_cls: MagicMock,
        project_dir: Path,
    ) -> None:
        mock_planner_cls.return_value.launch.return_value = _make_plan_result()
        result = runner.invoke(
            app,
            [
                "plan",
                "--spec",
                f"{FIXTURES / 'valid.md'},{FIXTURES / 'no_heading.md'}",
                "--project-root",
                str(project_dir),
            ],
        )
        assert result.exit_code == 0, result.output
        spec_file = project_dir / ".milknado" / "issues" / "plan-valid-no_heading.md"
        assert spec_file.exists()
        content = spec_file.read_text()
        assert content.startswith("# Plan for specs valid, no_heading")
        assert "## Spec: valid" in content
        assert "## Spec: no_heading" in content

    @patch("milknado.cli_plan.subprocess.run")
    def test_multi_issue_second_fetch_fails_exits_one(
        self,
        mock_run: MagicMock,
        project_dir: Path,
    ) -> None:
        ok = self._gh_ok()
        failed = MagicMock()
        failed.returncode = 1
        failed.stdout = ""
        failed.stderr = "bad issue"
        mock_run.side_effect = [ok, failed]

        result = runner.invoke(
            app,
            [
                "plan",
                "--issue",
                "42",
                "--issue",
                "9999",
                "--project-root",
                str(project_dir),
            ],
        )
        assert result.exit_code == 1
        assert "bad issue" in result.output


class TestToolsCheck:
    @patch("milknado.cli_tools.get_required_tool_status")
    def test_all_installed_exits_zero(self, mock_status: MagicMock) -> None:
        from milknado.domains.common.toolchain import ToolStatus

        mock_status.return_value = [
            ToolStatus(name="tilth", installed=True, path="/usr/bin/tilth"),
            ToolStatus(name="mergiraf", installed=True, path="/usr/bin/mergiraf"),
        ]
        result = runner.invoke(app, ["tools", "check"])
        assert result.exit_code == 0
        assert "tilth" in result.output
        assert "mergiraf" in result.output

    @patch("milknado.cli_tools.get_required_tool_status")
    def test_missing_tool_exits_nonzero(self, mock_status: MagicMock) -> None:
        from milknado.domains.common.toolchain import ToolStatus

        mock_status.return_value = [
            ToolStatus(name="tilth", installed=False, path=None),
            ToolStatus(name="mergiraf", installed=True, path="/usr/bin/mergiraf"),
        ]
        result = runner.invoke(app, ["tools", "check"])
        assert result.exit_code != 0
        assert "tilth" in result.output


class TestToolsInstall:
    @patch("milknado.cli_tools.install_missing_rust_tools")
    @patch("milknado.cli_tools.get_required_tool_status")
    def test_success_exits_zero(self, mock_status: MagicMock, mock_install: MagicMock) -> None:
        from milknado.domains.common.toolchain import ToolStatus

        mock_install.return_value = (["tilth"], [])
        mock_status.return_value = [
            ToolStatus(name="tilth", installed=True, path="/usr/bin/tilth"),
        ]
        result = runner.invoke(app, ["tools", "install"])
        assert result.exit_code == 0
        assert "tilth" in result.output

    @patch("milknado.cli_tools.install_missing_rust_tools")
    def test_failure_exits_nonzero(self, mock_install: MagicMock) -> None:
        mock_install.return_value = ([], ["mergiraf"])
        result = runner.invoke(app, ["tools", "install"])
        assert result.exit_code != 0
        assert "mergiraf" in result.output


class TestInitWithInstallRustTools:
    @patch("milknado.cli_tools.install_missing_rust_tools")
    @patch("milknado.cli_tools.get_required_tool_status")
    @patch("milknado.adapters.crg.CrgAdapter")
    def test_flag_triggers_install_on_success(
        self,
        _mock_crg: MagicMock,
        mock_status: MagicMock,
        mock_install: MagicMock,
        project_dir: Path,
    ) -> None:
        from milknado.domains.common.toolchain import ToolStatus

        mock_install.return_value = (["tilth"], [])
        mock_status.return_value = [
            ToolStatus(name="tilth", installed=True, path="/usr/bin/tilth"),
        ]
        result = runner.invoke(app, ["init", str(project_dir), "--install-rust-tools"])
        assert result.exit_code == 0
        mock_install.assert_called_once()

    @patch("milknado.cli_tools.install_missing_rust_tools")
    @patch("milknado.adapters.crg.CrgAdapter")
    def test_flag_exits_nonzero_on_install_failure(
        self,
        _mock_crg: MagicMock,
        mock_install: MagicMock,
        project_dir: Path,
    ) -> None:
        mock_install.return_value = ([], ["mergiraf"])
        result = runner.invoke(app, ["init", str(project_dir), "--install-rust-tools"])
        assert result.exit_code != 0
        assert "mergiraf" in result.output


class TestAgentsCheck:
    @patch("milknado.adapters.crg.CrgAdapter")
    def test_agents_check_prints_agent_fields(
        self,
        _mock_crg: MagicMock,
        project_dir: Path,
    ) -> None:
        runner.invoke(app, ["init", str(project_dir)])
        result = runner.invoke(
            app,
            ["agents", "check", "--project-root", str(project_dir)],
        )
        assert result.exit_code == 0
        assert "agent_family" in result.output
        assert "planning" in result.output
        assert "execution" in result.output
        assert "planning argv" in result.output


class TestRunCommand:
    def test_no_nodes_ready(
        self,
        mock_adapters: tuple[MagicMock, MagicMock, MagicMock],
        project_dir: Path,
    ) -> None:
        # The protected-branch guard now runs before the no-nodes check, so the
        # branch must resolve to a valid, non-protected name for this path.
        _mock_ralph_cls, mock_git_cls, _mock_crg_cls = mock_adapters
        mock_git_cls.return_value.current_branch.return_value = "feature-x"
        runner.invoke(app, ["init", str(project_dir)])
        result = runner.invoke(
            app,
            ["run", "--project-root", str(project_dir)],
        )
        assert result.exit_code == 0
        assert "No nodes ready" in result.output

    def test_no_nodes_ready_all_done(
        self,
        mock_adapters: tuple[MagicMock, MagicMock, MagicMock],
        project_dir: Path,
    ) -> None:
        from milknado.domains.common import default_config
        from milknado.domains.graph import MikadoGraph

        _mock_ralph_cls, mock_git_cls, _mock_crg_cls = mock_adapters
        mock_git_cls.return_value.current_branch.return_value = "feature-x"
        runner.invoke(app, ["init", str(project_dir)])
        config = default_config(project_dir)
        graph = MikadoGraph(config.db_path)
        graph.add_node("root")
        graph.mark_running(1)
        graph.mark_done(1)
        graph.close()

        result = runner.invoke(
            app,
            ["run", "--project-root", str(project_dir)],
        )
        assert result.exit_code == 0
        assert "No nodes ready" in result.output

    def test_dispatches_ready_nodes(
        self,
        mock_adapters: tuple[MagicMock, MagicMock, MagicMock],
        project_dir: Path,
    ) -> None:
        from milknado.domains.common import default_config
        from milknado.domains.graph import MikadoGraph

        mock_ralph_cls, _mock_git_cls, _mock_crg_cls = mock_adapters
        runner.invoke(app, ["init", str(project_dir)])
        config = default_config(project_dir)
        graph = MikadoGraph(config.db_path)
        root = graph.add_node("root goal")
        graph.add_node("leaf task", parent_id=root.id)
        graph.close()

        _configure_ralph_mocks(mock_ralph_cls, project_dir)

        result = runner.invoke(
            app,
            ["run", "--project-root", str(project_dir)],
        )
        assert result.exit_code == 0
        assert "Starting execution loop" in result.output
        assert "1 completed" in result.output

    def test_dispatches_multiple_parallel_leaves(
        self,
        mock_adapters: tuple[MagicMock, MagicMock, MagicMock],
        project_dir: Path,
    ) -> None:
        from milknado.domains.common import default_config
        from milknado.domains.graph import MikadoGraph

        mock_ralph_cls, _mock_git_cls, _mock_crg_cls = mock_adapters
        runner.invoke(app, ["init", str(project_dir)])
        config = default_config(project_dir)
        graph = MikadoGraph(config.db_path)
        root = graph.add_node("root")
        graph.add_node("leaf-a", parent_id=root.id)
        graph.add_node("leaf-b", parent_id=root.id)
        graph.close()

        _configure_ralph_mocks(mock_ralph_cls, project_dir, unique=True)

        result = runner.invoke(
            app,
            ["run", "--project-root", str(project_dir)],
        )
        assert result.exit_code == 0
        assert "2 completed" in result.output

    def test_skips_conflicting_nodes(
        self,
        mock_adapters: tuple[MagicMock, MagicMock, MagicMock],
        project_dir: Path,
    ) -> None:
        from milknado.domains.common import default_config
        from milknado.domains.graph import MikadoGraph

        mock_ralph_cls, _mock_git_cls, _mock_crg_cls = mock_adapters
        runner.invoke(app, ["init", str(project_dir)])
        config = default_config(project_dir)
        graph = MikadoGraph(config.db_path)
        root = graph.add_node("root")
        a = graph.add_node("leaf-a", parent_id=root.id)
        b = graph.add_node("leaf-b", parent_id=root.id)
        graph.set_file_ownership(a.id, ["shared.py"])
        graph.set_file_ownership(b.id, ["shared.py"])
        graph.close()

        _configure_ralph_mocks(mock_ralph_cls, project_dir, unique=True)

        result = runner.invoke(
            app,
            ["run", "--project-root", str(project_dir)],
        )
        assert result.exit_code == 0
        assert "2 completed" in result.output

    def test_strict_flag_maps_strict_exit_to_exit_1(
        self,
        mock_adapters: tuple[MagicMock, MagicMock, MagicMock],
        project_dir: Path,
    ) -> None:
        """US-102.1 — a strict-drain result must map to CLI exit code 1.

        The drain behaviour itself is covered by TestStrictDrain; this asserts
        the CLI wiring: --strict is threaded into loop.run and a True
        strict_exit becomes typer.Exit(code=1).
        """
        from milknado.domains.common import default_config
        from milknado.domains.execution.run_loop import RunLoopResult
        from milknado.domains.graph import MikadoGraph

        mock_ralph_cls, mock_git_cls, _mock_crg_cls = mock_adapters
        mock_git_cls.return_value.current_branch.return_value = "feature-x"
        runner.invoke(app, ["init", str(project_dir)])
        config = default_config(project_dir)
        graph = MikadoGraph(config.db_path)
        root = graph.add_node("root")
        graph.add_node("leaf-a", parent_id=root.id)
        graph.close()

        strict_result = RunLoopResult(
            root_done=False,
            dispatched_total=1,
            completed_total=0,
            failed_total=1,
            strict_exit=True,
        )
        with patch("milknado.domains.execution.RunLoop") as mock_loop_cls:
            mock_loop_cls.return_value.run.return_value = strict_result
            result = runner.invoke(
                app,
                ["run", "--strict", "--project-root", str(project_dir)],
            )

        assert result.exit_code == 1
        assert mock_loop_cls.return_value.run.call_args.kwargs["strict"] is True

    def test_non_strict_failure_does_not_exit_1(
        self,
        mock_adapters: tuple[MagicMock, MagicMock, MagicMock],
        project_dir: Path,
    ) -> None:
        """--strict is opt-in: a mid-run failure without the flag still exits 0.

        Guards against a naive ``failed_total -> exit 1`` regression that would
        break the opt-in contract; only strict_exit may force exit 1.
        """
        from milknado.domains.common import default_config
        from milknado.domains.execution.run_loop import RunLoopResult
        from milknado.domains.graph import MikadoGraph

        mock_ralph_cls, mock_git_cls, _mock_crg_cls = mock_adapters
        mock_git_cls.return_value.current_branch.return_value = "feature-x"
        runner.invoke(app, ["init", str(project_dir)])
        config = default_config(project_dir)
        graph = MikadoGraph(config.db_path)
        root = graph.add_node("root")
        graph.add_node("leaf-a", parent_id=root.id)
        graph.close()

        failed_result = RunLoopResult(
            root_done=False,
            dispatched_total=1,
            completed_total=0,
            failed_total=1,
            strict_exit=False,
        )
        with patch("milknado.domains.execution.RunLoop") as mock_loop_cls:
            mock_loop_cls.return_value.run.return_value = failed_result
            result = runner.invoke(
                app,
                ["run", "--project-root", str(project_dir)],
            )

        assert result.exit_code == 0
        assert mock_loop_cls.return_value.run.call_args.kwargs["strict"] is False

    def test_protected_branch_exits_2_without_dispatch(
        self,
        mock_adapters: tuple[MagicMock, MagicMock, MagicMock],
        project_dir: Path,
    ) -> None:
        """US-102.2 — running on a protected branch exits 2 with no side effects."""
        from milknado.domains.common import default_config
        from milknado.domains.graph import MikadoGraph

        mock_ralph_cls, mock_git_cls, _mock_crg_cls = mock_adapters
        mock_git_cls.return_value.current_branch.return_value = "main"
        runner.invoke(app, ["init", str(project_dir)])
        config = default_config(project_dir)
        graph = MikadoGraph(config.db_path)
        root = graph.add_node("root")
        graph.add_node("leaf-a", parent_id=root.id)
        graph.close()

        result = runner.invoke(
            app,
            ["run", "--project-root", str(project_dir)],
        )

        assert result.exit_code == 2
        assert "Starting execution loop" not in result.output
        assert "Refusing to run on protected branch" in result.output
        # No executor/worktree constructed: the adapter class is never instantiated.
        mock_ralph_cls.assert_not_called()
        mock_ralph_cls.return_value.create_run.assert_not_called()

    def test_allow_protected_bypasses_guard_on_protected_branch(
        self,
        mock_adapters: tuple[MagicMock, MagicMock, MagicMock],
        project_dir: Path,
    ) -> None:
        """US-102.2 — --allow-protected is the explicit opt-in past the guard."""
        from milknado.domains.common import default_config
        from milknado.domains.execution.run_loop import RunLoopResult
        from milknado.domains.graph import MikadoGraph

        mock_ralph_cls, mock_git_cls, _mock_crg_cls = mock_adapters
        mock_git_cls.return_value.current_branch.return_value = "main"
        runner.invoke(app, ["init", str(project_dir)])
        config = default_config(project_dir)
        graph = MikadoGraph(config.db_path)
        root = graph.add_node("root")
        graph.add_node("leaf-a", parent_id=root.id)
        graph.close()

        ok_result = RunLoopResult(
            root_done=True,
            dispatched_total=1,
            completed_total=1,
            failed_total=0,
        )
        with patch("milknado.domains.execution.RunLoop") as mock_loop_cls:
            mock_loop_cls.return_value.run.return_value = ok_result
            result = runner.invoke(
                app,
                ["run", "--allow-protected", "--project-root", str(project_dir)],
            )

        assert result.exit_code == 0
        assert "Starting execution loop" in result.output
        mock_loop_cls.return_value.run.assert_called_once()


# ── worker hook writers ───────────────────────────────────────────────────────


def test_write_claude_worker_settings_writes_resolved_tools(tmp_path: Path) -> None:
    _write_claude_worker_settings(tmp_path, tools=("Read", "Edit"))
    settings = json.loads((tmp_path / ".claude" / "settings.json").read_text(encoding="utf-8"))
    # The resolved tool tuple is what lands in the worker's allowlist.
    assert settings["permissions"]["allow"] == ["Read", "Edit"]
    pre = settings["hooks"]["PreToolUse"][0]
    assert pre["matcher"] == "Bash"
    assert pre["hooks"][0]["command"] == "rtk hook claude"


def test_write_gemini_worker_settings_writes_resolved_tools(tmp_path: Path) -> None:
    _write_gemini_worker_settings(tmp_path, tools=("tilth_search", "read_file"))
    settings = json.loads((tmp_path / ".gemini" / "settings.json").read_text(encoding="utf-8"))
    assert settings["includeTools"] == ["tilth_search", "read_file"]
    before = settings["hooks"]["BeforeTool"][0]
    assert before["matcher"] == "run_shell_command"
    assert before["hooks"][0]["command"] == "rtk hook gemini"


def test_write_worker_hooks_dispatches_to_family_writer(tmp_path: Path) -> None:
    # With rtk present and a default claude config, the dispatcher resolves the
    # family allowlist and routes to the claude writer.
    config = default_config(tmp_path)
    with patch("milknado.cli_tools.shutil.which", return_value="/usr/bin/rtk"):
        _write_worker_hooks(tmp_path, config)
    settings = json.loads((tmp_path / ".claude" / "settings.json").read_text(encoding="utf-8"))
    assert settings["permissions"]["allow"] == list(WORKER_ALLOWED_TOOLS["claude"])


def test_write_worker_hooks_skips_when_rtk_missing(tmp_path: Path) -> None:
    # No rtk on PATH → early return, no settings file written.
    config = default_config(tmp_path)
    with patch("milknado.cli_tools.shutil.which", return_value=None):
        _write_worker_hooks(tmp_path, config)
    assert not (tmp_path / ".claude").exists()


def test_write_worker_hooks_dispatches_to_gemini(tmp_path: Path) -> None:
    config = replace(default_config(tmp_path), agent_family="gemini")
    with patch("milknado.cli_tools.shutil.which", return_value="/usr/bin/rtk"):
        _write_worker_hooks(tmp_path, config)
    settings = json.loads((tmp_path / ".gemini" / "settings.json").read_text(encoding="utf-8"))
    assert settings["includeTools"] == list(WORKER_ALLOWED_TOOLS["gemini"])


def test_write_worker_hooks_dispatches_to_cursor(tmp_path: Path) -> None:
    config = replace(default_config(tmp_path), agent_family="cursor")
    with patch("milknado.cli_tools.shutil.which", return_value="/usr/bin/rtk"):
        _write_worker_hooks(tmp_path, config)
    hooks = json.loads((tmp_path / "hooks" / "hooks.json").read_text(encoding="utf-8"))
    assert hooks["hooks"][0]["command"] == "rtk hook cursor"


class TestPrintRunResult:
    def test_root_done_prints_success(self, capsys: pytest.CaptureFixture[str]) -> None:
        from milknado.cli_run import _print_run_result
        from milknado.domains.execution.run_loop import RunLoopResult

        _print_run_result(
            RunLoopResult(
                root_done=True,
                dispatched_total=1,
                completed_total=1,
                failed_total=0,
            )
        )
        assert "Root goal achieved" in capsys.readouterr().out

    def test_rebase_conflicts_rendered(self, capsys: pytest.CaptureFixture[str]) -> None:
        from milknado.cli_run import _print_run_result
        from milknado.domains.execution.executor import RebaseConflict
        from milknado.domains.execution.run_loop import RunLoopResult

        conflict = RebaseConflict(
            node_id=3,
            description="merge node 3",
            conflicting_files=("a.py", "b.py"),
            detail="rebase aborted",
        )
        _print_run_result(
            RunLoopResult(
                root_done=False,
                dispatched_total=2,
                completed_total=1,
                failed_total=1,
                rebase_conflicts=(conflict,),
            )
        )
        out = capsys.readouterr().out
        assert "Loop ended: 1 completed, 1 failed" in out
        assert "Rebase conflict — node 3" in out
        assert "a.py" in out
        assert "b.py" in out
        assert "rebase aborted" in out


def test_write_worker_hooks_unknown_family_skips(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = replace(default_config(tmp_path), agent_family="acme")
    with patch("milknado.cli_tools.shutil.which", return_value="/usr/bin/rtk"):
        _write_worker_hooks(tmp_path, config)
    assert "No hook template for family 'acme'" in capsys.readouterr().out
    assert not (tmp_path / ".claude").exists()


def test_merge_json_recovers_from_corrupt_file(tmp_path: Path) -> None:
    from milknado.cli_tools import _merge_json

    target = tmp_path / "settings.json"
    target.write_text("{not valid json", encoding="utf-8")
    _merge_json(target, {"key": "value"})
    assert json.loads(target.read_text(encoding="utf-8")) == {"key": "value"}


def test_ensure_plugins_loaded_announces_each_plugin(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from milknado._cli_helpers import _ensure_plugins_loaded

    config = default_config(tmp_path)
    local_plugin = MagicMock()
    local_plugin.meta.name = "local-hook"
    ep_plugin = MagicMock()
    ep_plugin.meta.name = "entry-hook"

    with (
        patch("milknado.plugins.load_plugins", return_value=[local_plugin]),
        patch("milknado.plugins.discover_entry_point_plugins", return_value=[ep_plugin]),
    ):
        loaded = _ensure_plugins_loaded(config)

    assert loaded == [local_plugin, ep_plugin]
    out = capsys.readouterr().out
    assert "Plugin loaded: local-hook" in out
    assert "Plugin loaded: entry-hook (entry point)" in out


def test_plan_exit_code_default_fallthrough() -> None:
    from milknado.cli_plan import _plan_exit_code

    # solver_status not in any special case, success True → final return 0.
    result = _make_plan_result(solver_status="UNKNOWN", batch_count=0, success=True)
    assert _plan_exit_code(result) == 0
