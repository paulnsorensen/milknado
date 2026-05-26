"""agents sub-commands."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from milknado._cli_helpers import _load_or_default
from milknado.domains.common.agent_argv import build_planning_subprocess

console = Console()

agents_app = typer.Typer(name="agents", help="Agent CLI compatibility")


@agents_app.command("check")
def agents_check(
    project_root: Annotated[
        Path, typer.Option("--project-root", help="Project root directory")
    ] = Path("."),
) -> None:
    """Print resolved planning/execution commands and sample planning argv."""
    project_root = project_root.resolve()
    config, _ = _load_or_default(project_root)
    console.print(f"[bold]agent_family[/bold]: {config.agent_family}")
    console.print(f"[bold]planning[/bold]: {config.planning_agent}")
    console.print(f"[bold]execution (ralphify)[/bold]: {config.execution_agent}")

    sample = project_root / ".milknado" / ".agent-check-sample.md"
    sample.parent.mkdir(parents=True, exist_ok=True)
    sample.write_text("# sample planning context\n", encoding="utf-8")
    try:
        argv, extra = build_planning_subprocess(
            sample,
            config.planning_agent,
        )
        redacted = {k: ("<stdin>" if k == "input" else v) for k, v in extra.items()}
        console.print(f"[bold]planning argv[/bold]: {argv}")
        if redacted:
            console.print(f"[bold]planning extras[/bold]: {redacted}")
    finally:
        sample.unlink(missing_ok=True)
