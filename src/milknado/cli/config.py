"""Configuration-inspection commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from milknado.domains.common import (
    LoadedConfig,
    default_config,
    explain_view,
    load_config_details,
    resolved_view,
)

config_app = typer.Typer(name="config", help="Inspect effective configuration")


@config_app.command("show")
def show(
    resolved: Annotated[
        bool, typer.Option("--resolved", help="Print effective runtime config.")
    ] = False,
    explain: Annotated[
        bool, typer.Option("--explain", help="Print values with their source layer.")
    ] = False,
    flavor: Annotated[
        str | None, typer.Option("--flavor", help="Show one resolved flavor profile.")
    ] = None,
    project_root: Annotated[
        Path, typer.Option("--project-root", help="Project root directory")
    ] = Path("."),
) -> None:
    """Print resolved configuration or source-attributed effective values."""
    if resolved == explain:
        raise typer.BadParameter("choose exactly one of --resolved or --explain")
    root = project_root.resolve()
    details = _load_details(root)
    view = explain_view(details, flavor) if explain else resolved_view(details, flavor)
    typer.echo(json.dumps(view, indent=2, sort_keys=True))


def _load_details(project_root: Path) -> LoadedConfig:
    path = project_root / "milknado.toml"
    if path.exists():
        return load_config_details(path)
    return LoadedConfig(default_config(project_root), {}, {})
