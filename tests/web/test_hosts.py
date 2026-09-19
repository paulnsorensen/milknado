from typing import cast

from milknado.domains.graph import MikadoGraph
from milknado.web import (
    HostDependencies,
    ObserverHandlers,
    OwnerHandlers,
    WebCommands,
    observer_commands,
    owner_commands,
)


def test_host_builders_set_capabilities() -> None:
    assert owner_commands(OwnerHandlers()).session_input is None
    assert observer_commands(ObserverHandlers()).force_stop is None
    assert WebCommands().cancel is None


class _Runs:
    record: dict[str, object]

    def __init__(self, record: dict[str, object]) -> None:
        self.record = record

    def get(self, run_id: str) -> dict[str, object] | None:
        return self.record if run_id == self.record["run_id"] else None


class _Graph:
    runs: _Runs

    def __init__(self, record: dict[str, object]) -> None:
        self.runs = _Runs(record)


class _Controller:
    def __init__(self) -> None:
        self.cancelled: list[str] = []

    def session_input(self, run_id: str, command: object) -> bool:
        _ = (run_id, command)
        return True

    def cancel(self, run_id: str) -> None:
        self.cancelled.append(run_id)

    def force_stop(self, run_id: str, timeout: float = 10.0) -> bool:
        _ = (run_id, timeout)
        return True

    def stop_scheduling(self) -> None:
        return None


def test_owner_cancel_returns_updated_run_record() -> None:
    record: dict[str, object] = {"run_id": "run-1", "status": "cancelled"}
    controller = _Controller()
    graph = cast(MikadoGraph, cast(object, _Graph(record)))
    commands = owner_commands(controller, HostDependencies(graph=graph))

    assert commands.cancel is not None
    assert commands.cancel("run-1") == record
    assert controller.cancelled == ["run-1"]
