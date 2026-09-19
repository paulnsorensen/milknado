# pyright: reportAny=false, reportExplicitAny=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnannotatedClassAttribute=false, reportUnnecessaryCast=false, reportUnnecessaryIsInstance=false
"""JSON encoding for private web responses."""

from __future__ import annotations

from typing import cast

import msgspec
from starlette.responses import JSONResponse


def _builtins(value: object) -> object:
    return cast(object, msgspec.to_builtins(value))


def json_response(value: object, status_code: int = 200) -> JSONResponse:
    return JSONResponse(_builtins(value), status_code=status_code)
