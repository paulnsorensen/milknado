from __future__ import annotations

import contextlib
import json
import logging
import uuid
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from milknado.domains.execution.run_loop._result import RunLoopResult

_logger = logging.getLogger("milknado")
_MAX_RETAINED_LOGS = 20
_MAX_LOG_BYTES = 10 * 1024 * 1024
_MAX_LOG_AGE_SECONDS = 7 * 24 * 60 * 60


def _prune_old_logs(
    log_dir: Path,
    pattern: str,
    keep: int = _MAX_RETAINED_LOGS,
    active_run_ids: Iterable[str] = (),
) -> None:
    active = tuple(active_run_ids)
    cutoff = datetime.now(UTC).timestamp() - _MAX_LOG_AGE_SECONDS
    retained_bytes = 0
    retained_count = 0
    logs = sorted(log_dir.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in logs:
        try:
            stat = path.stat()
        except OSError:
            continue
        protected = any(run_id in path.name for run_id in active)
        expired = stat.st_mtime < cutoff
        over_limit = retained_count >= keep or retained_bytes + stat.st_size > _MAX_LOG_BYTES
        if not protected and (expired or over_limit):
            with contextlib.suppress(OSError):
                path.unlink()
            continue
        retained_count += 1
        retained_bytes += stat.st_size


@contextmanager
def configure_run_logging(
    project_root: Path,
    *,
    active_run_ids: Iterable[str] = (),
) -> Generator[Path, None, None]:
    log_dir = project_root / ".milknado"
    log_dir.mkdir(parents=True, exist_ok=True)
    _prune_old_logs(log_dir, "run-*.log", active_run_ids=active_run_ids)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    log_path = log_dir / f"run-{timestamp}-{uuid.uuid4().hex}.log"
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    previous_level = _logger.level
    _logger.setLevel(logging.INFO)
    _logger.addHandler(handler)
    try:
        yield log_path
    finally:
        _logger.removeHandler(handler)
        _logger.setLevel(previous_level)
        handler.flush()
        handler.close()
        _prune_old_logs(log_dir, "run-*.log", active_run_ids=active_run_ids)


def emit_final_telemetry(result: RunLoopResult, stopped: int, interrupted: bool) -> None:
    _logger.info(
        "FINAL_TELEMETRY %s",
        json.dumps(
            {
                "dispatched": result.dispatched_total,
                "completed": result.completed_total,
                "failed": result.failed_total,
                "stopped": stopped,
                "conflicts": len(result.rebase_conflicts),
                "root_done": result.root_done,
                "strict_exit": result.strict_exit,
                "interrupted": interrupted,
            }
        ),
    )


def ts() -> str:
    return datetime.now(UTC).strftime("%H:%M:%S")


__all__ = ["configure_run_logging", "emit_final_telemetry", "ts"]
