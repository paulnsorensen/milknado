from __future__ import annotations

import logging
import sys

_logger = logging.getLogger("milknado")


def configure_stderr_logging() -> logging.Handler:
    """Install one formatted stderr handler for process entrypoints."""
    for handler in _logger.handlers:
        if isinstance(handler, logging.StreamHandler) and handler.stream is sys.stderr:
            return handler
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    _logger.setLevel(logging.INFO)
    _logger.addHandler(handler)
    return handler
