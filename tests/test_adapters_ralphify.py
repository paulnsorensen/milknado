import queue
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ralphify import EventType

from milknado.adapters.ralphify import (
    MILKNADO_COMPLETION_SIGNAL,
    RalphifyAdapter,
    _build_ralph_content,
    _build_verify_prompt,
    _parse_verify_output,
)
from milknado.domains.common.protocols import VerifySpecResult
from milknado.domains.common.types import MikadoNode


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
            completion_signal=MILKNADO_COMPLETION_SIGNAL,
            stop_on_completion_signal=True,
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


class TestWaitForNextCompletion:
    def test_returns_on_run_stopped_event(
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

        run_id, success = adapter.wait_for_next_completion({"run-1"})
        assert run_id == "run-1"
        assert success is True

    def test_returns_false_on_failed_run(
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

        run_id, success = adapter.wait_for_next_completion({"run-1"})
        assert run_id == "run-1"
        assert success is False

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

        run_id, success = adapter.wait_for_next_completion({"run-1"})
        assert run_id == "run-1"
        assert success is True

    def test_skips_events_for_inactive_runs(
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

        run_id, success = adapter.wait_for_next_completion({"run-1"})
        assert run_id == "run-1"
        assert success is True

    def test_returns_false_when_run_missing(
        self, adapter: RalphifyAdapter, mock_manager: MagicMock,
    ) -> None:
        from ralphify import EventType

        event = MagicMock()
        event.type = EventType.RUN_STOPPED
        event.run_id = "run-1"
        mock_manager.get_run.return_value = None
        adapter._queue.put(event)

        run_id, success = adapter.wait_for_next_completion({"run-1"})
        assert run_id == "run-1"
        assert success is False


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

    def test_includes_completion_promise_instruction(self) -> None:
        node = MikadoNode(id=1, description="Do thing")
        content = _build_ralph_content(node, "ctx", ["gate1"])
        assert "## Completion" in content
        assert f"<promise>{MILKNADO_COMPLETION_SIGNAL}</promise>" in content

    def test_longer_context_injected_and_completion_preserved(self) -> None:
        """Longer context with Goal/Why chain/Your task — template passes it
        through verbatim and Completion block still ends the file."""
        node = MikadoNode(id=2, description="Batch node")
        longer_ctx = (
            "## Goal\n\nRefactor auth slice into its own domain.\n\n"
            "## Why chain (parent → grandparent → ...)\n\n"
            "### Extract interfaces from auth module\n\n"
            "## Your task\n\nUpdate callers to use new AuthService interface.\n\n"
            "## Files\n\n- `src/main.py`\n\n"
            "## Impact Radius\n\n_(CRG unavailable — impact radius skipped)_"
        )
        content = _build_ralph_content(node, longer_ctx, ["uv run pytest"])
        assert "## Goal" in content
        assert "## Why chain" in content
        assert "## Your task" in content
        assert "## Completion" in content
        assert f"<promise>{MILKNADO_COMPLETION_SIGNAL}</promise>" in content
        # Completion block is the last heading in the file
        completion_pos = content.rfind("## Completion")
        assert completion_pos != -1
        assert content[completion_pos:].strip().endswith(
            f"emit `<promise>{MILKNADO_COMPLETION_SIGNAL}</promise>` on its own line\n"
            "so the run can stop before the iteration budget."
        )


class TestBuildVerifyPrompt:
    def test_contains_spec_and_graph_state(self) -> None:
        prompt = _build_verify_prompt("do the thing", "node A: done")
        assert "do the thing" in prompt
        assert "node A: done" in prompt

    def test_contains_result_tags(self) -> None:
        prompt = _build_verify_prompt("spec", "state")
        assert "<result>done</result>" in prompt
        assert "<result>gaps</result>" in prompt
        assert "<goal_delta>" in prompt

    def test_contains_completion_signal(self) -> None:
        prompt = _build_verify_prompt("spec", "state")
        assert f"<promise>{MILKNADO_COMPLETION_SIGNAL}</promise>" in prompt

    def test_none_graph_state_rendered(self) -> None:
        prompt = _build_verify_prompt("spec", None)  # type: ignore[arg-type]
        assert "(no graph state)" in prompt


class TestParseVerifyOutput:
    def test_done_result(self) -> None:
        result = _parse_verify_output("<result>done</result>")
        assert result == VerifySpecResult(outcome="done")

    def test_gaps_with_delta(self) -> None:
        output = "<result>gaps</result>\n<goal_delta>missing X</goal_delta>"
        result = _parse_verify_output(output)
        assert result == VerifySpecResult(outcome="gaps", goal_delta="missing X")

    def test_gaps_without_delta(self) -> None:
        result = _parse_verify_output("<result>gaps</result>")
        assert result == VerifySpecResult(outcome="gaps", goal_delta=None)

    def test_unparseable_defaults_to_done(self) -> None:
        result = _parse_verify_output("no recognizable tags here")
        assert result == VerifySpecResult(outcome="done")


class TestVerifySpec:
    def test_no_agent_returns_done(self, adapter: RalphifyAdapter) -> None:
        adapter._agent = ""  # type: ignore[attr-defined]
        result = adapter.verify_spec("spec text", "state")
        assert result == VerifySpecResult(outcome="done")

    @patch("milknado.adapters.ralphify.RunManager")
    def test_done_signal(
        self, mock_manager_cls: MagicMock, adapter: RalphifyAdapter,
    ) -> None:
        adapter._agent = "claude"  # type: ignore[attr-defined]
        local_q: queue.Queue[MagicMock] = queue.Queue()
        _setup_verify_mocks(mock_manager_cls, local_q)

        _put_iteration(local_q, EventType.ITERATION_COMPLETED, "<result>done</result>")
        _put_stopped(local_q)

        result = adapter.verify_spec("spec", "graph")
        assert result == VerifySpecResult(outcome="done")

    @patch("milknado.adapters.ralphify.RunManager")
    def test_gaps_signal_with_delta(
        self, mock_manager_cls: MagicMock, adapter: RalphifyAdapter,
    ) -> None:
        adapter._agent = "claude"  # type: ignore[attr-defined]
        local_q: queue.Queue[MagicMock] = queue.Queue()
        _setup_verify_mocks(mock_manager_cls, local_q)

        output = "<result>gaps</result>\n<goal_delta>missing auth</goal_delta>"
        _put_iteration(local_q, EventType.ITERATION_COMPLETED, output)
        _put_stopped(local_q)

        result = adapter.verify_spec("spec", "graph")
        assert result == VerifySpecResult(outcome="gaps", goal_delta="missing auth")

    @patch("milknado.adapters.ralphify.RunManager")
    def test_timeout_returns_gaps(
        self, mock_manager_cls: MagicMock, adapter: RalphifyAdapter,
    ) -> None:
        import time

        adapter._agent = "claude"  # type: ignore[attr-defined]
        local_q: queue.Queue[MagicMock] = queue.Queue()
        mock_manager = _setup_verify_mocks(mock_manager_cls, local_q)

        with patch("milknado.adapters.ralphify.time") as mock_time:
            mock_time.monotonic.side_effect = [0.0, 0.0, 200.0]
            mock_time.sleep = time.sleep
            result = adapter.verify_spec("spec", "graph")

        mock_manager.stop_run.assert_called_once()
        assert result == VerifySpecResult(outcome="gaps", goal_delta="verification timed out")


def _setup_verify_mocks(
    mock_manager_cls: MagicMock,
    local_q: queue.Queue[MagicMock],
) -> MagicMock:
    mock_manager = MagicMock()
    mock_manager_cls.return_value = mock_manager
    mock_emitter = MagicMock()
    mock_emitter.queue = local_q
    mock_run = MagicMock()
    mock_run.state.run_id = "verify-1"
    mock_run.emitter = mock_emitter
    mock_manager.create_run.return_value = mock_run
    return mock_manager


def _put_iteration(
    q: queue.Queue[MagicMock], event_type: EventType, result_text: str,
) -> None:
    event = MagicMock()
    event.type = event_type
    event.data = {"result_text": result_text}
    q.put(event)


def _put_stopped(q: queue.Queue[MagicMock]) -> None:
    event = MagicMock()
    event.type = EventType.RUN_STOPPED
    q.put(event)
