"""Typed evidence callbacks supplied to the graph-free loop runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from milknado.domains.common import HelperIdentity, ObservationKey, WorkerIdentity, WorkerOwner
from milknado.loop._process_identity import Descendant
from milknado.loop._process_registry import WorkerRegistry


class WorkerRecordView(Protocol):
    invocation_id: str
    snapshot_seq: int
    ready_generation: int
    helper_generation: int
    helper_pid: int | None
    helper_start_token: float | None
    observation_owner: str | None
    descendants: tuple[Descendant, ...]
    ended_at: str | None


class WorkerEvidence(Protocol):
    def with_deadline(self, deadline: float) -> WorkerEvidence: ...
    def record_worker(self, owner: WorkerOwner, worker: WorkerIdentity) -> None: ...
    def get_worker(self, invocation_id: str) -> WorkerRecordView | None: ...
    def record_helper(self, helper: HelperIdentity) -> None: ...
    def begin_worker_observation(self, key: ObservationKey) -> None: ...
    def commit_worker_observation(
        self, key: ObservationKey, descendants: tuple[Descendant, ...]
    ) -> None: ...
    def end_worker(
        self, invocation_id: str, snapshot_seq: int, helper_generation: int | None = None
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class ProtectionContext:
    evidence: WorkerEvidence
    owner: WorkerOwner
    db_path: Path
    registry: WorkerRegistry | None = None
