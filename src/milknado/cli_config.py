"""`milknado config` — config management commands."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from milknado._cli_helpers import _find_config
from milknado.domains.common.config import default_config, global_config_path, load_config
from milknado.domains.common.types import TaskFlavor

config_app = typer.Typer(name="config", help="Config management commands")
console = Console()

_ALL_HARNESSES = ["claude", "opencode", "codex"]
_VALID_HARNESSES: frozenset[str] = frozenset(_ALL_HARNESSES)


def _parse_harnesses(harness: str | None) -> list[str]:
    if not harness:
        return list(_ALL_HARNESSES)
    harnesses = [h.strip() for h in harness.split(",")]
    bad = sorted(set(harnesses) - _VALID_HARNESSES)
    if bad:
        raise ValueError(f"unknown harness(es) {bad}; expected one of {sorted(_VALID_HARNESSES)}")
    return harnesses


def _parse_flavors(flavor: str | None) -> list[TaskFlavor]:
    if not flavor:
        return list(TaskFlavor)
    valid = sorted(f.value for f in TaskFlavor)
    try:
        return [TaskFlavor(f.strip()) for f in flavor.split(",")]
    except ValueError as exc:
        raise ValueError(f"{exc}; expected one of {valid}") from None


def _load_cfg(scope: str, project_root: Path):
    if scope == "global":
        path = global_config_path()
        if path.exists():
            return load_config(path, include_global=False)
        return default_config(path.parent)
    path = _find_config(project_root)
    if path.exists():
        return load_config(path, include_global=True)
    return default_config(project_root)


@config_app.command("sync")
def config_sync(
    scope: Annotated[str, typer.Option("--scope", help="project or global")] = "project",
    harness: Annotated[
        str | None, typer.Option("--harness", help="Harnesses csv: claude,opencode,codex")
    ] = None,
    flavor: Annotated[
        str | None, typer.Option("--flavor", help="Comma-separated flavors to sync")
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Print manifest without writing")
    ] = False,
    project_root: Annotated[
        Path, typer.Option("--project-root", help="Project root directory")
    ] = Path("."),
) -> None:
    """Render per-flavor agent-definition files from milknado.toml."""
    from milknado.domains.config_sync import sync

    if scope not in ("project", "global"):
        console.print(f"[red]Unknown scope {scope!r}; expected 'project' or 'global'.[/red]")
        raise typer.Exit(code=1)

    try:
        harnesses = _parse_harnesses(harness)
        flavor_list = _parse_flavors(flavor)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None

    project_root = project_root.resolve()
    cfg = _load_cfg(scope, project_root)
    from milknado.domains.config_sync._render import _SyncFilter

    results = sync(cfg, scope, _SyncFilter(harnesses, flavor_list), dry_run)

    action = "Would write" if dry_run else "Wrote"
    for rd in results:
        console.print(f"{action}: {rd.path}")

    if not dry_run:
        console.print("\n[dim]Claude Code: restart the session or run /agents to reload.[/dim]")
        console.print("[dim]MCP clients: run /mcp reconnect to reload.[/dim]")
