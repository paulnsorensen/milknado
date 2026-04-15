from __future__ import annotations

import itertools
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from milknado.cli import app
from milknado.domains.planning import Planner

runner = CliRunner()


def _unique_run_factory() -> MagicMock:
    counter = itertools.count(1)

    def _create_run(*args: object, **kwargs: object) -> MagicMock:
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
        assert "agent_command" in content
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
    @patch.object(Planner, "_run_agent", return_value=0)
    def test_plan_success(
        self,
        _mock_run_agent: MagicMock,
        mock_crg_cls: MagicMock,
        project_dir: Path,
    ) -> None:
        mock_crg_cls.return_value.get_architecture_overview.return_value = {}
        result = runner.invoke(
            app,
            ["plan", "extract service", "--project-root", str(project_dir)],
        )
        assert result.exit_code == 0
        assert "Planning" in result.output

    @patch("milknado.adapters.crg.CrgAdapter")
    @patch.object(Planner, "_run_agent", return_value=1)
    def test_plan_failure(
        self,
        _mock_run_agent: MagicMock,
        mock_crg_cls: MagicMock,
        project_dir: Path,
    ) -> None:
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
