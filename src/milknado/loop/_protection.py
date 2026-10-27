"""Protection context for a worker launched under a supervising milknado process."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import psutil

from milknado.domains.common import WorkerOwner
from milknado.loop._process_contract import ProtectionContext
from milknado.loop._process_registry import WorkerRegistry
from milknado.loop._worker_evidence import LoopWorkerEvidence
from milknado.loop._worker_release import LoopWorkerRelease
from milknado.project import load_project_config


def loop_protection(
    db_path: Path, owner: WorkerOwner, registry: WorkerRegistry | None = None
) -> ProtectionContext:
    """Protect a worker with DB-backed evidence and attempt-aware gate release."""
    return ProtectionContext(
        LoopWorkerEvidence(db_path), owner, db_path, registry, release=LoopWorkerRelease(db_path)
    )


def protection_from_env(env: Mapping[str, str], run_id: str) -> ProtectionContext:
    """Bind the current process as supervisor of the node named by `env`."""
    config = load_project_config(Path(env["MILKNADO_PROJECT_ROOT"]))
    supervisor = psutil.Process()
    owner = WorkerOwner(
        run_id,
        supervisor.pid,
        supervisor.create_time(),
        run_id,
        int(env["MILKNADO_NODE_ID"]),
    )
    return loop_protection(config.db_path, owner)
