from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

EditKind = Literal["add", "modify", "delete", "rename"]
SolverStatus = Literal["OPTIMAL", "FEASIBLE", "INFEASIBLE", "UNKNOWN"]
RelationshipReason = Literal["new_file", "new_import", "new_call", "new_type_use"]


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
    paths: tuple[str, ...] = ()

    def all_paths(self) -> tuple[str, ...]:
        """Return all paths this change touches. Falls back to (path,) if paths is empty."""
        return self.paths if self.paths else (self.path,)


@dataclass(frozen=True)
class NewRelationship:
    source_change_id: str
    dependant_change_id: str
    reason: RelationshipReason


@dataclass(frozen=True)
class Batch:
    index: int
    change_ids: tuple[str, ...]
    depends_on: tuple[int, ...]
    oversized: bool = False


@dataclass(frozen=True)
class SymbolSpread:
    symbol: SymbolRef
    spread: int


@dataclass(frozen=True)
class BatchPlan:
    batches: tuple[Batch, ...]
    spread_report: tuple[SymbolSpread, ...]
    solver_status: SolverStatus
