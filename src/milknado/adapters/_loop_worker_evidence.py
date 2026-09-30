"""Per-operation, deadline-scoped evidence callbacks for loop workers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from milknado.domains.common import HelperIdentity, ObservationKey, WorkerIdentity, WorkerOwner
from milknado.domains.graph import WorkerEvidenceStore, WorkerRecord
from milknado.loop._process_identity import Descendant


@dataclass(frozen=True, slots=True)
class LoopWorkerEvidence:
    db_path: Path
    deadline: float | None = None

    def with_deadline(self, deadline: float) -> LoopWorkerEvidence:
        return LoopWorkerEvidence(self.db_path, deadline)

    def record_worker(self, owner: WorkerOwner, worker: WorkerIdentity) -> None:
        with WorkerEvidenceStore(self.db_path, deadline=self.deadline) as store:
            store.record_worker(owner, worker)

    def get_worker(self, invocation_id: str) -> WorkerRecord | None:
        with WorkerEvidenceStore(self.db_path, deadline=self.deadline) as store:
            return store.get(invocation_id)

    def record_helper(self, helper: HelperIdentity) -> None:
        with WorkerEvidenceStore(self.db_path, deadline=self.deadline) as store:
            store.record_helper(helper)

    def begin_worker_observation(self, key: ObservationKey) -> None:
        with WorkerEvidenceStore(self.db_path, deadline=self.deadline) as store:
            store.begin(key)

    def commit_worker_observation(
        self, key: ObservationKey, descendants: tuple[Descendant, ...]
    ) -> None:
        with WorkerEvidenceStore(self.db_path, deadline=self.deadline) as store:
            store.commit(key, descendants)

    def end_worker(
        self, invocation_id: str, snapshot_seq: int, helper_generation: int | None = None
    ) -> None:
        with WorkerEvidenceStore(self.db_path, deadline=self.deadline) as store:
            store.end(invocation_id, snapshot_seq, helper_generation)
