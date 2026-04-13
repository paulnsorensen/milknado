from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from milknado.cli import app

runner = CliRunner()


@pytest.fixture()
def project_dir(tmp_path: Path) -> Path:
    return tmp_path


class TestInit:
    def test_creates_config_and_db(self, project_dir: Path) -> None:
        result = runner.invoke(app, ["init", str(project_dir)])
        assert result.exit_code == 0
        assert (project_dir / "milknado.toml").exists()
        assert (project_dir / ".milknado" / "milknado.db").exists()

    def test_idempotent(self, project_dir: Path) -> None:
        runner.invoke(app, ["init", str(project_dir)])
        result = runner.invoke(app, ["init", str(project_dir)])
        assert result.exit_code == 0
        assert "already exists" in result.output

    def test_config_has_defaults(self, project_dir: Path) -> None:
        runner.invoke(app, ["init", str(project_dir)])
        content = (project_dir / "milknado.toml").read_text()
        assert "agent_command" in content
        assert "quality_gates" in content
        assert "concurrency_limit" in content


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
    @patch("milknado.domains.planning.planner.subprocess.run")
    def test_plan_success(
        self,
        mock_run: MagicMock,
        mock_crg_cls: MagicMock,
        project_dir: Path,
    ) -> None:
        mock_run.return_value = MagicMock(returncode=0)
        mock_crg_cls.return_value.get_architecture_overview.return_value = {}
        result = runner.invoke(
            app,
            ["plan", "extract service", "--project-root", str(project_dir)],
        )
        assert result.exit_code == 0
        assert "Planning" in result.output

    @patch("milknado.adapters.crg.CrgAdapter")
    @patch("milknado.domains.planning.planner.subprocess.run")
    def test_plan_failure(
        self,
        mock_run: MagicMock,
        mock_crg_cls: MagicMock,
        project_dir: Path,
    ) -> None:
        mock_run.return_value = MagicMock(returncode=1)
        mock_crg_cls.return_value.get_architecture_overview.return_value = {}
        result = runner.invoke(
            app,
            ["plan", "extract service", "--project-root", str(project_dir)],
        )
        assert result.exit_code == 1


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

    @patch("milknado.adapters.RalphifyAdapter")
    @patch("milknado.adapters.GitAdapter")
    @patch("milknado.adapters.CrgAdapter")
    def test_dispatches_ready_nodes(
        self,
        mock_crg_cls: MagicMock,
        _mock_git_cls: MagicMock,
        mock_ralph_cls: MagicMock,
        project_dir: Path,
    ) -> None:
        runner.invoke(app, ["init", str(project_dir)])
        runner.invoke(
            app,
            ["add-node", "leaf task", "--project-root", str(project_dir)],
        )

        fake_run = MagicMock()
        fake_run.id = "run-1"
        mock_ralph_cls.return_value.create_run.return_value = fake_run
        mock_ralph_cls.return_value.generate_ralph_md.return_value = (
            project_dir / "RALPH.md"
        )
        mock_crg_cls.return_value.get_impact_radius.return_value = {}

        result = runner.invoke(
            app,
            ["run", "--project-root", str(project_dir)],
        )
        assert result.exit_code == 0
        assert "Dispatched node 1" in result.output
        assert "1 node(s) dispatched" in result.output

    @patch("milknado.adapters.RalphifyAdapter")
    @patch("milknado.adapters.GitAdapter")
    @patch("milknado.adapters.CrgAdapter")
    def test_dispatches_multiple_parallel_leaves(
        self,
        mock_crg_cls: MagicMock,
        _mock_git_cls: MagicMock,
        mock_ralph_cls: MagicMock,
        project_dir: Path,
    ) -> None:
        from milknado.domains.common import default_config
        from milknado.domains.graph import MikadoGraph

        runner.invoke(app, ["init", str(project_dir)])
        config = default_config(project_dir)
        graph = MikadoGraph(config.db_path)
        root = graph.add_node("root")
        graph.add_node("leaf-a", parent_id=root.id)
        graph.add_node("leaf-b", parent_id=root.id)
        graph.close()

        fake_run = MagicMock()
        fake_run.id = "run-1"
        mock_ralph_cls.return_value.create_run.return_value = fake_run
        mock_ralph_cls.return_value.generate_ralph_md.return_value = (
            project_dir / "RALPH.md"
        )
        mock_crg_cls.return_value.get_impact_radius.return_value = {}

        result = runner.invoke(
            app,
            ["run", "--project-root", str(project_dir)],
        )
        assert result.exit_code == 0
        assert "2 node(s) dispatched" in result.output

    @patch("milknado.adapters.RalphifyAdapter")
    @patch("milknado.adapters.GitAdapter")
    @patch("milknado.adapters.CrgAdapter")
    def test_skips_conflicting_nodes(
        self,
        mock_crg_cls: MagicMock,
        _mock_git_cls: MagicMock,
        mock_ralph_cls: MagicMock,
        project_dir: Path,
    ) -> None:
        from milknado.domains.common import default_config
        from milknado.domains.graph import MikadoGraph

        runner.invoke(app, ["init", str(project_dir)])
        config = default_config(project_dir)
        graph = MikadoGraph(config.db_path)
        root = graph.add_node("root")
        a = graph.add_node("leaf-a", parent_id=root.id)
        b = graph.add_node("leaf-b", parent_id=root.id)
        graph.set_file_ownership(a.id, ["shared.py"])
        graph.set_file_ownership(b.id, ["shared.py"])
        graph.close()

        fake_run = MagicMock()
        fake_run.id = "run-1"
        mock_ralph_cls.return_value.create_run.return_value = fake_run
        mock_ralph_cls.return_value.generate_ralph_md.return_value = (
            project_dir / "RALPH.md"
        )
        mock_crg_cls.return_value.get_impact_radius.return_value = {}

        result = runner.invoke(
            app,
            ["run", "--project-root", str(project_dir)],
        )
        assert result.exit_code == 0
        assert "1 node(s) dispatched" in result.output
