from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

EditKind = Literal["add", "modify", "delete", "rename"]
SolverStatus = Literal["OPTIMAL", "FEASIBLE", "INFEASIBLE", "UNKNOWN"]


@dataclass(frozen=True)
class SymbolRef:
    name: str
    file: str


@dataclass(frozen=True)
class FileChange:
    id: str
    path: str
    edit_kind: EditKind = "modify"
    symbols: tuple[SymbolRef, ...] = ()
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class BatchPlan:
    batches: tuple[tuple[str, ...], ...]
    spread_report: dict[str, int]
    solver_status: SolverStatus
