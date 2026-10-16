"""Record graph attempt evidence around the release of a gated loop worker."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from milknado.domains.common import WorkerOwner
from milknado.domains.graph import MikadoGraph
from milknado.loop._process_gate import WorkerProcess


@dataclass(frozen=True, slots=True)
class LoopWorkerRelease:
    db_path: Path

    def release_worker(self, worker: WorkerProcess, owner: WorkerOwner) -> None:
        run_id = owner.graph_run_id
        if run_id is None:
            worker.release()
            return
        graph = MikadoGraph(self.db_path)
        try:
            attempts = graph.task_attempts
            fence = attempts.begin_release(run_id, worker.identity.invocation_id)
            worker.release()
            if fence is not None:
                attempts.finish_release(fence, worker.start_evidence.value)
        finally:
            graph.close()
