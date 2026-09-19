"""Sorted route discovery for the private web API."""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Iterable
from typing import Any, Protocol, cast


class RouteModule(Protocol):
    ROUTES: Iterable[Any]


def discover_routes() -> list[Any]:
    modules = sorted(
        info.name for info in pkgutil.iter_modules(__path__) if info.name != "__init__"
    )
    routes: list[Any] = []
    for name in modules:
        module = cast(RouteModule, cast(object, importlib.import_module(f"{__package__}.{name}")))
        routes.extend(module.ROUTES)
    return routes
