from __future__ import annotations

import logging
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

_logger = logging.getLogger("milknado")


@contextmanager
def configure_run_logging(project_root: Path) -> Generator[Path, None, None]:
    log_dir = project_root / ".milknado"
    log_dir.mkdir(parents=True, exist_ok=True)
    iso = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    log_path = log_dir / f"run-{iso}.log"
    handler = logging.FileHandler(str(log_path), mode="w", encoding="utf-8", delay=False)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
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
