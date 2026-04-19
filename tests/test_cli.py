import itertools
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from milknado.cli import app

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
    ralph_cls: MagicMock, project_dir: Path, *, unique: bool = False,
) -> None:
    if unique:
        ralph_cls.return_value.create_run = _unique_run_factory()
    else:
        fake_run = MagicMock()
        fake_run.state.run_id = "run-1"
        ralph_cls.return_value.create_run.return_value = fake_run
    ralph_cls.return_value.generate_ralph_md.return_value = project_dir / "RALPH.md"

    def _wait_for_next_completion(active_run_ids: set[str]) -> tuple[str, bool]:
        return next(iter(active_run_ids)), True

    ralph_cls.return_value.wait_for_next_completion.side_effect = (
        _wait_for_next_completion
    )


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
    def test_creates_config_and_db(
        self, _mock_crg: MagicMock, project_dir: Path
    ) -> None:
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
    def test_config_has_defaults(
        self, _mock_crg: MagicMock, project_dir: Path
    ) -> None:
        runner.invoke(app, ["init", str(project_dir)])
        content = (project_dir / "milknado.toml").read_text()
        assert "agent_family" in content
        assert "planning_agent" in content
        assert "execution_agent" in content
        assert "quality_gates" in content
        assert "concurrency_limit" in content

    @patch("milknado.adapters.crg.CrgAdapter")
    def test_calls_ensure_graph(
        self, mock_crg_cls: MagicMock, project_dir: Path
    ) -> None:
        result = runner.invoke(app, ["init", str(project_dir)])
        assert result.exit_code == 0
        mock_crg_cls.return_value.ensure_graph.assert_called_once_with(
            project_dir
        )
        assert "Code-review-graph ready" in result.output


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
                "add-node", "child",
                "--parent", "1",
                "--project-root", str(project_dir),
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
                "add-node", "refactor auth",
                "--files", "src/auth.py",
                "--files", "src/login.py",
                "--project-root", str(project_dir),
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
                "add-node", "prereq",
                "--parent", "1",
                "--project-root", str(project_dir),
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
            success=False, exit_code=1, solver_status="NO_MANIFEST",
        )
        result = runner.invoke(
            app,
            ["plan", "--spec", str(FIXTURES / "valid.md"), "--project-root", str(project_dir)],
        )
        assert result.exit_code == 1


def _make_plan_result(**kwargs: object) -> MagicMock:
    """Build a PlanResult-like mock with sensible defaults."""
    from milknado.domains.planning.planner import PlanResult

    defaults = {
        "success": True,
        "exit_code": 0,
        "context_path": None,
        "nodes_created": 3,
        "batch_count": 2,
        "oversized_count": 0,
        "solver_status": "OPTIMAL",
        "change_count": 4,
    }
    defaults.update(kwargs)
    return PlanResult(**defaults)  # type: ignore[arg-type]


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
            ["plan", "--spec", str(FIXTURES / "no_heading.md"),
             "--project-root", str(project_dir)],
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
            solver_status="INFEASIBLE", success=False, exit_code=0,
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
            solver_status="UNKNOWN", batch_count=2,
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


class TestToolsCheck:
    @patch("milknado.cli.get_required_tool_status")
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

    @patch("milknado.cli.get_required_tool_status")
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
    @patch("milknado.cli.install_missing_rust_tools")
    @patch("milknado.cli.get_required_tool_status")
    def test_success_exits_zero(
        self, mock_status: MagicMock, mock_install: MagicMock
    ) -> None:
        from milknado.domains.common.toolchain import ToolStatus

        mock_install.return_value = (["tilth"], [])
        mock_status.return_value = [
            ToolStatus(name="tilth", installed=True, path="/usr/bin/tilth"),
        ]
        result = runner.invoke(app, ["tools", "install"])
        assert result.exit_code == 0
        assert "tilth" in result.output

    @patch("milknado.cli.install_missing_rust_tools")
    def test_failure_exits_nonzero(self, mock_install: MagicMock) -> None:
        mock_install.return_value = ([], ["mergiraf"])
        result = runner.invoke(app, ["tools", "install"])
        assert result.exit_code != 0
        assert "mergiraf" in result.output


class TestInitWithInstallRustTools:
    @patch("milknado.cli.install_missing_rust_tools")
    @patch("milknado.cli.get_required_tool_status")
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
        result = runner.invoke(
            app, ["init", str(project_dir), "--install-rust-tools"]
        )
        assert result.exit_code == 0
        mock_install.assert_called_once()

    @patch("milknado.cli.install_missing_rust_tools")
    @patch("milknado.adapters.crg.CrgAdapter")
    def test_flag_exits_nonzero_on_install_failure(
        self,
        _mock_crg: MagicMock,
        mock_install: MagicMock,
        project_dir: Path,
    ) -> None:
        mock_install.return_value = ([], ["mergiraf"])
        result = runner.invoke(
            app, ["init", str(project_dir), "--install-rust-tools"]
        )
        assert result.exit_code != 0
        assert "mergiraf" in result.output


class TestAgentsCheck:
    @patch("milknado.adapters.crg.CrgAdapter")
    def test_agents_check_prints_agent_fields(
        self, _mock_crg: MagicMock, project_dir: Path,
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
    def test_no_nodes_ready(self, project_dir: Path) -> None:
        runner.invoke(app, ["init", str(project_dir)])
        result = runner.invoke(
            app,
            ["run", "--project-root", str(project_dir)],
        )
        assert result.exit_code == 0
        assert "No nodes ready" in result.output

    def test_no_nodes_ready_all_done(self, project_dir: Path) -> None:
        from milknado.domains.common import default_config
        from milknado.domains.graph import MikadoGraph

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
        mock_ralph_cls, _mock_git_cls, _mock_crg_cls = mock_adapters
        runner.invoke(app, ["init", str(project_dir)])
        runner.invoke(
            app,
            ["add-node", "leaf task", "--project-root", str(project_dir)],
        )
        _configure_ralph_mocks(mock_ralph_cls, project_dir)

        result = runner.invoke(
            app,
            ["run", "--project-root", str(project_dir)],
        )
        assert result.exit_code == 0
        assert "Starting execution loop" in result.output
        assert "Root goal achieved" in result.output

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
        assert "Root goal achieved" in result.output

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
        assert "Root goal achieved" in result.output
