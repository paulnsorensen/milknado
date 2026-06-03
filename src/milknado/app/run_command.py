"""Application-layer policy for the ``run`` command."""

from __future__ import annotations

import typer
from rich.console import Console

from milknado.domains.common import MilknadoConfig

console = Console()


def check_protected_branch(
    cfg: MilknadoConfig,
    branch: str,
    allow_protected: bool,
) -> None:
    """Refuse to run on a protected or invalid branch with exit code 2.

    Called before the graph DB, executor, or run log is built, so a refused run
    leaves no side effects (US-102.2). Refusal is loud — the user is told which
    branch was refused and how to override. ``--allow-protected`` is the explicit
    opt-in for a protected *named* branch; a detached HEAD (``current_branch()``
    returns ``"HEAD"`` or ``""``) is refused unconditionally, since the run loop
    has no valid branch to rebase-merge completed nodes back onto (mirrors the
    headless guard in ``execution/headless.py``).
    """
    if branch in ("", "HEAD"):
        console.print(
            f"[red]Refusing to run on detached HEAD (branch {branch!r}); "
            "check out a named branch first.[/red]"
        )
        raise typer.Exit(code=2)
    if not allow_protected and branch in cfg.protected_branches:
        console.print(
            f"[red]Refusing to run on protected branch '{branch}'. "
            "Pass --allow-protected to override.[/red]"
        )
        raise typer.Exit(code=2)
