"""tools and plugin sub-commands, plus worker hook wiring."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from milknado.domains.common import MilknadoConfig
from milknado.domains.common.agent_argv import WORKER_ALLOWED_TOOLS, resolve_worker_tools
from milknado.domains.common.merge import deep_merge
from milknado.domains.common.toolchain import get_required_tool_status, install_missing_rust_tools

console = Console()

tools_app = typer.Typer(name="tools", help="Toolchain commands")
plugin_app = typer.Typer(name="plugin", help="Plugin management commands")


# ── Tool status ───────────────────────────────────────────────────────────────


def _print_tool_status() -> list[tuple[str, bool]]:
    statuses = get_required_tool_status()
    rows: list[tuple[str, bool]] = []
    for status in statuses:
        state = "ok" if status.installed else "missing"
        details = f" ({status.path})" if status.path else ""
        console.print(f"{status.name}: {state}{details}")
        rows.append((status.name, status.installed))
    return rows


def _install_rust_tools_or_exit() -> None:
    installed, failed = install_missing_rust_tools()
    for name in installed:
        console.print(f"[green]Installed {name}[/green]")
    if failed:
        console.print("[red]Failed to install:[/red] " + ", ".join(failed))
        console.print("Install Rust/cargo, then run: [bold]milknado tools install[/bold]")
        raise typer.Exit(code=1)


# ── Worker hook wiring ────────────────────────────────────────────────────────
# `rtk hook <family>` is a self-contained stdin→stdout hook processor shipped
# by rtk-ai/rtk. We wire it directly into each harness's project-scoped hook
# config — no `rtk init -g` (global install) required.


def _write_worker_hooks(project_root: Path, config: MilknadoConfig) -> None:
    if shutil.which("rtk") is None:
        console.print("[dim]rtk not on PATH; skipping hook wiring.[/dim]")
        return
    family = config.agent_family
    family_tools = config.worker_tools.get(family)
    tools = resolve_worker_tools(
        family,
        list(family_tools) if family_tools is not None else None,
    )
    if family == "claude":
        _write_claude_worker_settings(project_root, tools)
    elif family == "gemini":
        _write_gemini_worker_settings(project_root, tools)
    elif family == "cursor":
        _write_cursor_worker_hooks(project_root)
    else:
        console.print(f"[dim]No hook template for family '{family}'; skipping.[/dim]")


def _merge_json(path: Path, patch: dict) -> None:
    existing: dict = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing = {}
    existing = deep_merge(existing, patch, list_mode="merge_dedup")
    path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")


def _write_claude_worker_settings(
    project_root: Path,
    tools: tuple[str, ...] = WORKER_ALLOWED_TOOLS["claude"],
) -> None:
    path = project_root / ".claude" / "settings.json"
    path.parent.mkdir(exist_ok=True)
    _merge_json(
        path,
        {
            "permissions": {
                "allow": list(tools),
            },
            "hooks": {
                "PreToolUse": [
                    {
                        "matcher": "Bash",
                        "hooks": [{"type": "command", "command": "rtk hook claude"}],
                    },
                ],
            },
        },
    )
    console.print(f"Worker hooks written: {path}")


def _write_gemini_worker_settings(
    project_root: Path,
    tools: tuple[str, ...] = WORKER_ALLOWED_TOOLS["gemini"],
) -> None:
    path = project_root / ".gemini" / "settings.json"
    path.parent.mkdir(exist_ok=True)
    _merge_json(
        path,
        {
            "includeTools": list(tools),
            "hooks": {
                "BeforeTool": [
                    {
                        "matcher": "run_shell_command",
                        "hooks": [
                            {
                                "name": "rtk-rewrite",
                                "type": "command",
                                "command": "rtk hook gemini",
                                "timeout": 5000,
                            },
                        ],
                    },
                ],
            },
        },
    )
    console.print(f"Worker hooks written: {path}")


def _write_cursor_worker_hooks(project_root: Path) -> None:
    path = project_root / "hooks" / "hooks.json"
    path.parent.mkdir(exist_ok=True)
    _merge_json(
        path,
        {
            "hooks": [
                {
                    "type": "PreToolUse",
                    "matcher": "Bash",
                    "command": "rtk hook cursor",
                },
            ],
        },
    )
    console.print(f"Worker hooks written: {path}")


# ── tools commands ────────────────────────────────────────────────────────────


@tools_app.command("check")
def tools_check() -> None:
    """Check whether required Rust tools are available."""
    rows = _print_tool_status()
    if not all(installed for _, installed in rows):
        raise typer.Exit(code=1)


@tools_app.command("install")
def tools_install() -> None:
    """Install missing Rust tools used by milknado workflows."""
    _install_rust_tools_or_exit()
    _print_tool_status()


# ── plugin commands ───────────────────────────────────────────────────────────


@plugin_app.command("init")
def plugin_init(
    name: Annotated[str, typer.Argument(help="Plugin name")],
    target_dir: Annotated[
        Path, typer.Option("--target-dir", "-d", help="Directory to create plugin in")
    ] = Path("."),
) -> None:
    """Scaffold a new milknado plugin."""
    from milknado.plugins import scaffold_plugin

    try:
        result = scaffold_plugin(name, target_dir.resolve())
        console.print(f"Created plugin [bold]{name}[/bold] at {result.plugin_dir}")
        for f in result.files_created:
            console.print(f"  {f}")
    except FileExistsError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1) from None
