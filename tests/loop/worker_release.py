"""Worker release for tests that do not exercise attempt bookkeeping."""

from __future__ import annotations

from milknado.domains.common import WorkerOwner
from milknado.loop._process_gate import WorkerProcess


class PlainWorkerRelease:
    """Release the exec gate with no attempt bookkeeping."""

    def release_worker(self, worker: WorkerProcess, owner: WorkerOwner) -> None:
        del owner
        worker.release()
