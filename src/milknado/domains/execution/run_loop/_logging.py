from __future__ import annotations

import contextlib
import logging
import sys
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

_logger = logging.getLogger("milknado")

_MAX_RETAINED_LOGS = 20


def _prune_old_logs(log_dir: Path, pattern: str, keep: int = _MAX_RETAINED_LOGS) -> None:
    logs = sorted(log_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    for stale in logs[keep:]:
        with contextlib.suppress(OSError):
            stale.unlink()


@contextmanager
def configure_run_logging(project_root: Path) -> Generator[Path, None, None]:
    log_dir = project_root / ".milknado"
    log_dir.mkdir(parents=True, exist_ok=True)
    _prune_old_logs(log_dir, "run-*.log")
    _prune_old_logs(log_dir / "runs", "*.log")
    iso = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    log_path = log_dir / f"run-{iso}.log"
    handler = logging.FileHandler(str(log_path), mode="w", encoding="utf-8", delay=False)
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


def configure_stderr_logging() -> logging.Handler:
    """Install a formatted stderr handler for long-lived non-interactive surfaces.

    The MCP server has no single run to scope a log file to and speaks stdio —
    logging to stdout would corrupt the protocol stream, so this writes to
    stderr instead of reusing configure_run_logging's file handler.
    """
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    _logger.setLevel(logging.INFO)
    _logger.addHandler(handler)
    return handler


def ts() -> str:
    return datetime.now(UTC).strftime("%H:%M:%S")
