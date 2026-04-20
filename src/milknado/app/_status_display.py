from __future__ import annotations

import statistics
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from milknado.domains.common import MikadoNode, MilknadoConfig
    from milknado.domains.graph import MikadoGraph

console = Console()


def fetch_run_states(nodes: list[MikadoNode]) -> dict[str, str] | None:
    from milknado.domains.common import NodeStatus

    run_ids = [n.run_id for n in nodes if n.run_id and n.status == NodeStatus.RUNNING]
    if not run_ids:
        return None
    try:
        from milknado.adapters.ralphify import RalphifyAdapter

        ralph = RalphifyAdapter()
        states: dict[str, str] = {}
        for run_id in run_ids:
            run = ralph.get_run(run_id)
            if run:
                states[run_id] = getattr(run, "status", "unknown")
        return states or None
    except Exception:  # noqa: BLE001 — status enrichment is best-effort
        return None


def print_duration_table(nodes: list[MikadoNode]) -> None:
    table = Table(title="Node Durations", show_lines=False)
    table.add_column("ID", justify="right", style="dim")
    table.add_column("Description")
    table.add_column("Status")
    table.add_column("Duration (s)", justify="right")

    durations: list[float] = []
    for node in nodes:
        dur = node.completion_duration_seconds
        if dur is not None:
            durations.append(dur)
            dur_str = f"{dur:.1f}"
        else:
            dur_str = "-"
        table.add_row(str(node.id), node.description, node.status.value, dur_str)

    console.print(table)

    if not durations:
        console.print("[dim]No completed nodes with duration data.[/dim]")
        return

    mn = min(durations)
    med = statistics.median(durations)
    mx = max(durations)
    p95 = statistics.quantiles(durations, n=20)[18] if len(durations) >= 2 else durations[0]
    console.print(
        f"[bold]Duration summary:[/bold] "
        f"min={mn:.1f}s  median={med:.1f}s  p95={p95:.1f}s  max={mx:.1f}s"
    )


def print_status(graph: MikadoGraph, with_durations: bool) -> None:
    from milknado.domains.graph import render_tree

    nodes = graph.get_all_nodes()
    if not nodes:
        console.print("No nodes in graph. Run [bold]milknado plan[/bold] to start.")
        return
    console.print(render_tree(graph, run_states=fetch_run_states(nodes)))
    if with_durations:
        print_duration_table(nodes)


def check_agents_config(config: MilknadoConfig, project_root: Path) -> None:
    from milknado.domains.common.agent_argv import build_planning_subprocess

    console.print(f"[bold]agent_family[/bold]: {config.agent_family}")
    console.print(f"[bold]planning[/bold]: {config.planning_agent}")
    console.print(f"[bold]execution (ralphify)[/bold]: {config.execution_agent}")
    sample = project_root / ".milknado" / ".agent-check-sample.md"
    sample.parent.mkdir(parents=True, exist_ok=True)
    sample.write_text("# sample planning context\n", encoding="utf-8")
    try:
        argv, extra = build_planning_subprocess(sample, config.planning_agent)
        redacted = {k: ("<stdin>" if k == "input" else v) for k, v in extra.items()}
        console.print(f"[bold]planning argv[/bold]: {argv}")
        if redacted:
            console.print(f"[bold]planning extras[/bold]: {redacted}")
    finally:
        sample.unlink(missing_ok=True)
