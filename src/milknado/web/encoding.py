"""JSON encoding for private web responses."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any, cast

import msgspec
from starlette.responses import JSONResponse


def _builtins(value: Any) -> Any:
    try:
        return msgspec.to_builtins(value)
    except (TypeError, ValueError):
        if is_dataclass(value):
            return asdict(cast(Any, value))
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, tuple):
            return [_builtins(item) for item in value]
        if isinstance(value, dict):
            return {key: _builtins(item) for key, item in value.items()}
        return value


def json_response(value: Any, status_code: int = 200) -> JSONResponse:
    return JSONResponse(_builtins(value), status_code=status_code)
