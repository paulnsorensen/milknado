from __future__ import annotations

import contextlib
import logging
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

_logger = logging.getLogger("milknado")
_MAX_RETAINED_LOGS = 20


def _prune_old_logs(
    log_dir: Path,
    pattern: str,
    keep: int = _MAX_RETAINED_LOGS,
    active_run_ids: Iterable[str] = (),
) -> None:
    active = tuple(active_run_ids)
    logs = sorted(log_dir.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    removable = [path for path in logs if not any(run_id in path.name for run_id in active)]
    protected_count = len(logs) - len(removable)
    for stale in removable[max(keep - protected_count, 0) :]:
        with contextlib.suppress(OSError):
            stale.unlink()


@contextmanager
def configure_run_logging(
    project_root: Path,
    *,
    active_run_ids: Iterable[str] = (),
) -> Generator[Path, None, None]:
    log_dir = project_root / ".milknado"
    log_dir.mkdir(parents=True, exist_ok=True)
    _prune_old_logs(log_dir, "run-*.log", active_run_ids=active_run_ids)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    log_path = log_dir / f"run-{timestamp}.log"
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


def ts() -> str:
    return datetime.now(UTC).strftime("%H:%M:%S")


__all__ = ["configure_run_logging", "ts"]
