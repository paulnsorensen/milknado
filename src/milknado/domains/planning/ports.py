from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class PlanningProcessResult:
    exit_code: int
    stdout: str = ""
    stderr: str = ""


class PlanningProcessPort(Protocol):
    def run_agent(
        self,
        context_path: Path,
        command: str,
        project_root: Path,
    ) -> PlanningProcessResult: ...

    def run_validation(
        self,
        command: str,
        payload: dict[str, object],
        project_root: Path,
    ) -> PlanningProcessResult: ...


@dataclass(frozen=True)
class PlanningPorts:
    process: PlanningProcessPort
