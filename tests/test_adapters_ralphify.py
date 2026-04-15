from __future__ import annotations

import queue
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from milknado.adapters.ralphify import RalphifyAdapter, _build_ralph_content
from milknado.domains.common.types import CompletionEvent, MikadoNode


@pytest.fixture()
def mock_manager() -> MagicMock:
    return MagicMock()


@pytest.fixture()
def adapter(mock_manager: MagicMock) -> RalphifyAdapter:
    a = RalphifyAdapter.__new__(RalphifyAdapter)
    a._manager = mock_manager
    a._queue = queue.Queue()
    a._emitter = MagicMock()
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
        mock_run = MagicMock(id="run-1")
        mock_manager.create_run.return_value = mock_run

        result = adapter.create_run(
            agent="claude",
            ralph_dir=Path("/project"),
            ralph_file=Path("/project/RALPH.md"),
            commands=["uv run pytest"],
            quality_gates=["uv run ruff check"],
        )

        mock_config_cls.assert_called_once_with(
            agent="claude",
            ralph_dir=Path("/project"),
            ralph_file=Path("/project/RALPH.md"),
            project_root=Path("/project"),
        )
        mock_manager.create_run.assert_called_once_with(mock_config)
        mock_run.add_listener.assert_called_once()
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


class TestCompletionEvents:
    def test_yields_on_run_stopped_event(
        self, adapter: RalphifyAdapter, mock_manager: MagicMock,
    ) -> None:
        from ralphify import EventType, RunStatus

        event = MagicMock()
        event.type = EventType.RUN_STOPPED
        event.run_id = "run-1"
        run = MagicMock()
        run.state.status = RunStatus.COMPLETED
        mock_manager.get_run.return_value = run
        adapter._queue.put(event)

        events = adapter.completion_events()
        result = next(events)
        assert result == CompletionEvent(run_id="run-1", success=True)

    def test_yields_failure_on_failed_run(
        self, adapter: RalphifyAdapter, mock_manager: MagicMock,
    ) -> None:
        from ralphify import EventType, RunStatus

        event = MagicMock()
        event.type = EventType.RUN_STOPPED
        event.run_id = "run-1"
        run = MagicMock()
        run.state.status = RunStatus.FAILED
        mock_manager.get_run.return_value = run
        adapter._queue.put(event)

        events = adapter.completion_events()
        result = next(events)
        assert result == CompletionEvent(run_id="run-1", success=False)

    def test_skips_non_stop_events(
        self, adapter: RalphifyAdapter, mock_manager: MagicMock,
    ) -> None:
        from ralphify import EventType, RunStatus

        noise = MagicMock()
        noise.type = EventType.ITERATION_COMPLETED
        noise.run_id = "run-1"

        stop = MagicMock()
        stop.type = EventType.RUN_STOPPED
        stop.run_id = "run-1"
        run = MagicMock()
        run.state.status = RunStatus.COMPLETED
        mock_manager.get_run.return_value = run

        adapter._queue.put(noise)
        adapter._queue.put(stop)

        events = adapter.completion_events()
        result = next(events)
        assert result == CompletionEvent(run_id="run-1", success=True)

    def test_yields_events_for_any_run(
        self, adapter: RalphifyAdapter, mock_manager: MagicMock,
    ) -> None:
        from ralphify import EventType, RunStatus

        event1 = MagicMock()
        event1.type = EventType.RUN_STOPPED
        event1.run_id = "run-99"

        event2 = MagicMock()
        event2.type = EventType.RUN_STOPPED
        event2.run_id = "run-1"

        run = MagicMock()
        run.state.status = RunStatus.COMPLETED
        mock_manager.get_run.return_value = run

        adapter._queue.put(event1)
        adapter._queue.put(event2)

        events = adapter.completion_events()
        first = next(events)
        assert first == CompletionEvent(run_id="run-99", success=True)
        second = next(events)
        assert second == CompletionEvent(run_id="run-1", success=True)

    def test_yields_failure_when_run_missing(
        self, adapter: RalphifyAdapter, mock_manager: MagicMock,
    ) -> None:
        from ralphify import EventType

        event = MagicMock()
        event.type = EventType.RUN_STOPPED
        event.run_id = "run-1"
        mock_manager.get_run.return_value = None
        adapter._queue.put(event)

        events = adapter.completion_events()
        result = next(events)
        assert result == CompletionEvent(run_id="run-1", success=False)


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
