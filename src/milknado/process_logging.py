from __future__ import annotations

import logging
import sys
from typing import TextIO, cast

_logger = logging.getLogger("milknado")


def configure_stderr_logging() -> logging.Handler:
    """Install one formatted stderr handler for process entrypoints."""
    for handler in _logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            stream_handler = cast("logging.StreamHandler[TextIO]", handler)
            if stream_handler.stream is sys.stderr:
                return stream_handler
    stderr_handler: logging.StreamHandler[TextIO] = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    _logger.setLevel(logging.INFO)
    _logger.addHandler(stderr_handler)
    return stderr_handler
