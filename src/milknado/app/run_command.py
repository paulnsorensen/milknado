"""Application-layer policy for the ``run`` command."""

from __future__ import annotations

import typer
from rich.console import Console

from milknado.domains.common import MilknadoConfig

console = Console()


def _check_protected_branch(
    cfg: MilknadoConfig,
    branch: str,
    allow_protected: bool,
) -> None:
    """Reject a run on a protected branch with exit code 2.

    Called before the executor or run log is built, so a rejected run leaves no
    side effects (US-102.2). Rejection is loud — the user is told which branch
    was refused and how to override. ``--allow-protected`` is the explicit opt-in.
    """
    if not allow_protected and branch in cfg.protected_branches:
        console.print(
            f"[red]Refusing to run on protected branch '{branch}'. "
            "Pass --allow-protected to override.[/red]"
        )
        raise typer.Exit(code=2)
