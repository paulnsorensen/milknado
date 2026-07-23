import inspect
from dataclasses import FrozenInstanceError, fields, is_dataclass
from importlib.metadata import metadata
from typing import Literal, get_type_hints

import pytest
from textual.app import App

import milknado.domains.common as common
from milknado.app.run import (
    ActiveRunSnapshot,
    ExecutionRunStatus,
    ExecutionSnapshot,
    RunActionAvailability,
    TerminalRunSnapshot,
)
from milknado.domains.common import LoopPort, ProgressEvent, TerminalRunOutcome


def test_textual_is_available_as_runtime_dependency() -> None:
    assert App.__name__ == "App"
    requirements = metadata("milknado").get_all("Requires-Dist") or ()
    assert any(requirement.startswith("textual") for requirement in requirements)


def test_loop_port_exposes_typed_operator_control_contract() -> None:
    assert {"queue_guidance", "request_stop_run", "stop_run", "force_stop_run"} <= set(
        LoopPort.__dict__
    )
    assert list(inspect.signature(LoopPort.queue_guidance).parameters) == [
        "self",
        "run_id",
        "text",
    ]
    assert list(inspect.signature(LoopPort.request_stop_run).parameters) == [
        "self",
        "run_id",
    ]
    for method_name in ("stop_run", "force_stop_run"):
        signature = inspect.signature(getattr(LoopPort, method_name))
        assert list(signature.parameters) == ["self", "run_id", "timeout"]
        assert signature.parameters["timeout"].default is None
    completion_signature = inspect.signature(LoopPort.wait_for_next_completion)
    assert list(completion_signature.parameters) == ["self", "active_run_ids", "timeout"]
    assert completion_signature.parameters["timeout"].default is None
    assert get_type_hints(LoopPort.queue_guidance)["return"] is bool
    assert get_type_hints(LoopPort.request_stop_run)["return"] is type(None)
    assert get_type_hints(LoopPort.stop_run)["return"] is bool
    assert get_type_hints(LoopPort.force_stop_run)["return"] is bool
    assert (
        get_type_hints(LoopPort.wait_for_next_completion)["return"]
        == tuple[str, TerminalRunOutcome | ProgressEvent]
    )
    assert get_type_hints(LoopPort.wait_for_next_completion)["active_run_ids"] == set[str]
    assert TerminalRunOutcome == Literal["completed", "stopped", "failed"]
    assert common.TerminalRunOutcome is TerminalRunOutcome
    assert "TerminalRunOutcome" in common.__all__


def test_application_snapshots_are_frozen_slots_based_read_models() -> None:
    actions = RunActionAvailability(cancel_reason="already stopping")
    active = ActiveRunSnapshot(
        run_id="run-1",
        node_id=1,
        description="Implement snapshots",
        status=ExecutionRunStatus.RUNNING,
        progress="1/2",
        stop_requested=False,
        actions=actions,
        output=("line one",),
        pending_guidance=("continue",),
    )
    terminal = TerminalRunSnapshot(
        run_id="run-0",
        node_id=2,
        description="Stopped earlier",
        status=ExecutionRunStatus.STOPPED,
        output=("last output",),
        pending_guidance=("not delivered",),
    )
    snapshot = ExecutionSnapshot(
        goal="responsive-run-tui",
        active_runs=(active,),
        terminal_runs=(terminal,),
        completed=0,
        failed=0,
        stopped=1,
        available=1,
        event_lines=("dispatched",),
    )

    for value in (actions, active, terminal, snapshot):
        assert is_dataclass(value)
        assert type(value).__slots__ == tuple(field.name for field in fields(value))
    assert get_type_hints(RunActionAvailability) == {
        "cancel_reason": str | None,
        "guidance_reason": str | None,
        "force_stop_reason": str | None,
    }
    assert get_type_hints(ActiveRunSnapshot) == {
        "run_id": str,
        "node_id": int,
        "description": str,
        "status": ExecutionRunStatus,
        "progress": str | None,
        "stop_requested": bool,
        "actions": RunActionAvailability,
        "output": tuple[str, ...],
        "pending_guidance": tuple[str, ...],
    }
    assert get_type_hints(TerminalRunSnapshot) == {
        "run_id": str,
        "node_id": int,
        "description": str,
        "status": ExecutionRunStatus,
        "output": tuple[str, ...],
        "pending_guidance": tuple[str, ...],
    }
    assert get_type_hints(ExecutionSnapshot) == {
        "goal": str,
        "active_runs": tuple[ActiveRunSnapshot, ...],
        "terminal_runs": tuple[TerminalRunSnapshot, ...],
        "completed": int,
        "failed": int,
        "stopped": int,
        "available": int,
        "event_lines": tuple[str, ...],
    }
    assert isinstance(snapshot.active_runs, tuple)
    assert isinstance(snapshot.terminal_runs, tuple)
    assert isinstance(snapshot.event_lines, tuple)
    assert isinstance(active.output, tuple)
    assert isinstance(active.pending_guidance, tuple)
    with pytest.raises(FrozenInstanceError):
        active.stop_requested = True
    with pytest.raises(FrozenInstanceError):
        snapshot.event_lines = ("completed",)
