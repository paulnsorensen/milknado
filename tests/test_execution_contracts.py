import inspect
from importlib.metadata import metadata

from dataclasses import FrozenInstanceError, fields, is_dataclass
from typing import Literal, get_type_hints

import pytest
from textual.app import App

import milknado.domains.common as common
import milknado.domains.execution as execution
import milknado.domains.execution.run_loop as run_loop

from milknado.domains.common import LoopPort, TerminalRunOutcome
from milknado.domains.execution import ActiveRunSnapshot, ExecutionSnapshot
from milknado.domains.execution.run_loop import (
    ActiveRunSnapshot as RunLoopActiveRunSnapshot,
)
from milknado.domains.execution.run_loop import ExecutionSnapshot as RunLoopExecutionSnapshot
from milknado.loop import RunStatus


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
    assert get_type_hints(LoopPort.wait_for_next_completion)["return"] == tuple[
        str, Literal["completed", "stopped", "failed"]
    ]
    assert get_type_hints(LoopPort.wait_for_next_completion)["active_run_ids"] == set[str]
    assert TerminalRunOutcome == Literal["completed", "stopped", "failed"]
    assert common.TerminalRunOutcome is TerminalRunOutcome
    assert "TerminalRunOutcome" in common.__all__



def test_execution_snapshots_are_frozen_slots_based_read_models() -> None:
    active = ActiveRunSnapshot(
        run_id="run-1",
        node_id=1,
        description="Implement snapshots",
        status=RunStatus.RUNNING,
        progress="1/2",
        stop_requested=False,
        force_stop_available=True,
        unavailable_action_reasons=(("cancel", "already stopping"),),
        output=("line one",),
        pending_guidance=("continue",),
    )
    snapshot = ExecutionSnapshot(
        goal="responsive-run-tui",
        active_runs=(active,),
        completed=0,
        failed=0,
        available=1,
        event_lines=("dispatched",),
    )

    assert ActiveRunSnapshot is RunLoopActiveRunSnapshot
    assert ExecutionSnapshot is RunLoopExecutionSnapshot
    assert execution.ActiveRunSnapshot is ActiveRunSnapshot
    assert execution.ExecutionSnapshot is ExecutionSnapshot
    assert run_loop.ActiveRunSnapshot is ActiveRunSnapshot
    assert run_loop.ExecutionSnapshot is ExecutionSnapshot
    assert {"ActiveRunSnapshot", "ExecutionSnapshot"} <= set(execution.__all__)
    assert {"ActiveRunSnapshot", "ExecutionSnapshot"} <= set(run_loop.__all__)
    assert is_dataclass(active) and is_dataclass(snapshot)
    assert ActiveRunSnapshot.__slots__ == tuple(field.name for field in fields(ActiveRunSnapshot))
    assert ExecutionSnapshot.__slots__ == tuple(field.name for field in fields(ExecutionSnapshot))
    assert get_type_hints(ActiveRunSnapshot) == {
        "run_id": str,
        "node_id": int,
        "description": str,
        "status": RunStatus,
        "progress": str | None,
        "stop_requested": bool,
        "force_stop_available": bool,
        "unavailable_action_reasons": tuple[tuple[str, str], ...],
        "output": tuple[str, ...],
        "pending_guidance": tuple[str, ...],
    }
    assert get_type_hints(ExecutionSnapshot) == {
        "goal": str,
        "active_runs": tuple[ActiveRunSnapshot, ...],
        "completed": int,
        "failed": int,
        "available": int,
        "event_lines": tuple[str, ...],
    }
    assert isinstance(snapshot.active_runs, tuple)
    assert isinstance(snapshot.event_lines, tuple)
    assert isinstance(active.unavailable_action_reasons, tuple)
    assert isinstance(active.output, tuple)
    assert isinstance(active.pending_guidance, tuple)
    with pytest.raises(FrozenInstanceError):
        active.stop_requested = True
    with pytest.raises(FrozenInstanceError):
        snapshot.event_lines = ("completed",)
