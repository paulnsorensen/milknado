from __future__ import annotations

import logging
import sys
from typing import TextIO, cast

from milknado.process_logging import configure_stderr_logging


def test_configure_stderr_logging_is_idempotent() -> None:
    logger = logging.getLogger("milknado")
    before = list(logger.handlers)
    try:
        first = configure_stderr_logging()
        second = configure_stderr_logging()

        assert second is first
        assert isinstance(first, logging.StreamHandler)
        stream_handler = cast("logging.StreamHandler[TextIO]", first)
        assert stream_handler.stream is sys.stderr
        assert stream_handler.formatter is not None
        assert stream_handler.formatter._fmt == "%(asctime)s %(levelname)s %(name)s: %(message)s"
    finally:
        logger.handlers[:] = before
