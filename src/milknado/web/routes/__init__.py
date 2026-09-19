# pyright: reportAny=false, reportExplicitAny=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnannotatedClassAttribute=false, reportUnnecessaryCast=false, reportUnnecessaryIsInstance=false
"""Sorted route discovery for the private web API."""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Iterable
from typing import Protocol, cast

from starlette.routing import BaseRoute


class RouteModule(Protocol):
    ROUTES: Iterable[BaseRoute]


def discover_routes() -> list[BaseRoute]:
    modules = sorted(
        info.name for info in pkgutil.iter_modules(__path__) if info.name != "__init__"
    )
    routes: list[BaseRoute] = []
    for name in modules:
        module = cast(RouteModule, cast(object, importlib.import_module(f"{__package__}.{name}")))
        routes.extend(module.ROUTES)
    return routes
