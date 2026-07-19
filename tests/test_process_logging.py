from __future__ import annotations

import logging
import sys

from milknado.process_logging import configure_stderr_logging


def test_configure_stderr_logging_is_idempotent() -> None:
    logger = logging.getLogger("milknado")
    before = list(logger.handlers)
    try:
        first = configure_stderr_logging()
        second = configure_stderr_logging()

        assert second is first
        assert first.stream is sys.stderr
        assert first.formatter is not None
        assert first.formatter._fmt == "%(asctime)s %(levelname)s %(name)s: %(message)s"
    finally:
        logger.handlers[:] = before
