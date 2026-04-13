from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from milknado.adapters.ralphify import RalphifyAdapter, _build_ralph_content
from milknado.domains.common.types import MikadoNode


@pytest.fixture()
def mock_manager() -> MagicMock:
    return MagicMock()


@pytest.fixture()
def adapter(mock_manager: MagicMock) -> RalphifyAdapter:
    a = RalphifyAdapter.__new__(RalphifyAdapter)
    a._manager = mock_manager
    return a


class TestCreateRun:
    @patch("milknado.adapters.ralphify.RunConfig")
    def test_creates_config_and_delegates(
        self,
        mock_config_cls: MagicMock,
        adapter: RalphifyAdapter,
        mock_manager: MagicMock,
    ) -> None:
        mock_config = MagicMock()
        mock_config_cls.return_value = mock_config
        mock_manager.create_run.return_value = MagicMock(id="run-1")

        result = adapter.create_run(
            agent="claude",
            ralph_dir=Path("/project"),
            ralph_file=Path("/project/RALPH.md"),
            commands=["uv run pytest"],
            quality_gates=["uv run ruff check"],
        )

        mock_config_cls.assert_called_once_with(project_root=Path("/project"))
        mock_manager.create_run.assert_called_once_with(mock_config)
        assert result.id == "run-1"


class TestStartStopRun:
    def test_start_delegates(
        self, adapter: RalphifyAdapter, mock_manager: MagicMock
    ) -> None:
        adapter.start_run("run-1")
        mock_manager.start_run.assert_called_once_with("run-1")

    def test_stop_delegates(
        self, adapter: RalphifyAdapter, mock_manager: MagicMock
    ) -> None:
        adapter.stop_run("run-1")
        mock_manager.stop_run.assert_called_once_with("run-1")


class TestListAndGetRuns:
    def test_list_runs(
        self, adapter: RalphifyAdapter, mock_manager: MagicMock
    ) -> None:
        mock_manager.list_runs.return_value = []
        assert adapter.list_runs() == []

    def test_get_run_found(
        self, adapter: RalphifyAdapter, mock_manager: MagicMock
    ) -> None:
        run = MagicMock(id="run-1")
        mock_manager.get_run.return_value = run
        assert adapter.get_run("run-1") == run

    def test_get_run_not_found(
        self, adapter: RalphifyAdapter, mock_manager: MagicMock
    ) -> None:
        mock_manager.get_run.return_value = None
        assert adapter.get_run("missing") is None


class TestGenerateRalphMd:
    def test_writes_file(self, adapter: RalphifyAdapter, tmp_path: Path) -> None:
        node = MikadoNode(id=1, description="Extract interface")
        output = tmp_path / "RALPH.md"
        result = adapter.generate_ralph_md(
            node=node,
            context="Refactoring auth module",
            quality_gates=["uv run pytest"],
            output_path=output,
        )
        assert result == output
        assert output.exists()
        content = output.read_text()
        assert "Extract interface" in content
        assert "Refactoring auth module" in content
        assert "`uv run pytest`" in content


class TestBuildRalphContent:
    def test_includes_all_sections(self) -> None:
        node = MikadoNode(id=1, description="Do thing")
        content = _build_ralph_content(
            node, "some context", ["gate1", "gate2"]
        )
        assert content.startswith("# Do thing")
        assert "## Context" in content
        assert "some context" in content
        assert "- `gate1`" in content
        assert "- `gate2`" in content
