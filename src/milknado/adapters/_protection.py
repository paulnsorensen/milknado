"""Protection context for a worker launched under a supervising milknado process."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import psutil

from milknado.adapters._loop_worker_evidence import LoopWorkerEvidence
from milknado.domains.common import MilknadoConfig, WorkerOwner, default_config, load_config
from milknado.loop._process_contract import ProtectionContext


def load_config_or_default(root: Path) -> MilknadoConfig:
    config_path = root / "milknado.toml"
    return load_config(config_path) if config_path.exists() else default_config(root)


def protection_from_env(env: Mapping[str, str], run_id: str) -> ProtectionContext:
    """Bind the current process as supervisor of the node named by `env`."""
    config = load_config_or_default(Path(env["MILKNADO_PROJECT_ROOT"]))
    supervisor = psutil.Process()
    owner = WorkerOwner(
        run_id,
        supervisor.pid,
        supervisor.create_time(),
        run_id,
        int(env["MILKNADO_NODE_ID"]),
    )
    return ProtectionContext(LoopWorkerEvidence(config.db_path), owner, config.db_path)
