from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from milknado.domains.common.protocols import ProgressEvent
from milknado.domains.common.types import NodeStatus

if TYPE_CHECKING:
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.table import Table

    from milknado.domains.graph import MikadoGraph

SPINNER_FRAMES = ("◜", "◝", "◞", "◟")
STALL_THRESHOLD_DEFAULT = 300
ETA_SAMPLE_SIZE_DEFAULT = 10
MAX_RETRIES_DEFAULT = 2
OVERLAY_TAIL_LINES = 100


@dataclass(frozen=True)
class WorkerStats:
    elapsed: float
    pct: float | None
    attempts: int
    files: list[str]


def elapsed_str(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def eta_str(avg_duration: float | None, elapsed: float) -> str:
    if avg_duration is None:
        return "~?"
    remaining = max(0.0, avg_duration - elapsed)
    return f"~{elapsed_str(remaining)}"


def files_cell(files: list[str]) -> str:
    s = ", ".join(files)
    return s[:59] + "…" if len(s) > 60 else s


def ts() -> str:
    return datetime.now(UTC).strftime("%H:%M:%S")


def is_stalled(elapsed: float, threshold: float) -> bool:
    return elapsed >= threshold


def render_progress_bar(
    frame: str, elapsed: float, pct: float | None, stall_threshold: float,
) -> str:
    if pct is not None:
        filled = int(pct) // 10
        bar = "█" * filled + "░" * (10 - filled)
        return f"[cyan]{bar}[/cyan] {int(pct)}%"
    if is_stalled(elapsed, stall_threshold):
        return f"[yellow]{frame} ⚠[/yellow]"
    return f"[cyan]{frame}[/cyan]"


def build_title(
    active_count: int, done: int, failed: int, blocked: int,
) -> str:
    return (
        f"milknado — {active_count} active | "
        f"[green]{done} done[/green] | [red]{failed} failed[/red] | "
        f"[dim]{blocked} blocked[/dim]"
    )


def build_log_panel(logs: Iterable[str]) -> Panel:
    from rich.panel import Panel

    items = list(logs)
    content = "\n".join(items) if items else "[dim]No events yet[/dim]"
    return Panel(content, title="Log", border_style="dim", expand=True)


def worker_stats_for(
    run_id: str,
    now: float,
    active: Mapping[str, int],
    dispatched_at: Mapping[str, float],
    attempts: Mapping[int, int],
    progress_by_run: Mapping[str, ProgressEvent],
    graph: MikadoGraph,
) -> WorkerStats:
    node_id = active[run_id]
    elapsed = now - dispatched_at.get(run_id, now)
    ev = progress_by_run.get(run_id)
    pct = ev.work / ev.total * 100 if ev and ev.total > 0 else None
    return WorkerStats(
        elapsed=elapsed,
        pct=pct,
        attempts=attempts.get(node_id, 0),
        files=graph.get_file_ownership(node_id),
    )


def _title_from_graph(graph: MikadoGraph, active_count: int) -> str:
    all_nodes = graph.get_all_nodes()
    done = sum(1 for n in all_nodes if n.status == NodeStatus.DONE)
    failed = sum(1 for n in all_nodes if n.status == NodeStatus.FAILED)
    blocked = sum(1 for n in all_nodes if n.status == NodeStatus.BLOCKED)
    return build_title(active_count, done, failed, blocked)


def build_worker_table(
    tick: int,
    now: float,
    active: Mapping[str, int],
    dispatched_at: Mapping[str, float],
    attempts: Mapping[int, int],
    progress_by_run: Mapping[str, ProgressEvent],
    completion_durations: Iterable[float],
    graph: MikadoGraph,
    stall_threshold: float,
    max_retries: int,
) -> Table:
    from rich.table import Table

    frame = SPINNER_FRAMES[tick % len(SPINNER_FRAMES)]
    durations = list(completion_durations)
    avg_dur = sum(durations) / len(durations) if len(durations) >= 3 else None

    table = Table(
        title=_title_from_graph(graph, len(active)),
        show_header=True,
        header_style="bold",
    )
    table.add_column("", width=12, no_wrap=True)
    table.add_column("ID", style="cyan", width=4, no_wrap=True)
    table.add_column("Description")
    table.add_column("Files", width=62)
    table.add_column("Elapsed", width=7, no_wrap=True)
    table.add_column("ETA", width=6, no_wrap=True)
    table.add_column("Attempt", width=9, no_wrap=True)

    for run_id, node_id in active.items():
        node = graph.get_node(node_id)
        if not node:
            continue
        stats = worker_stats_for(
            run_id, now, active, dispatched_at, attempts, progress_by_run, graph,
        )
        table.add_row(
            render_progress_bar(frame, stats.elapsed, stats.pct, stall_threshold),
            str(node_id),
            node.description,
            files_cell(stats.files),
            elapsed_str(stats.elapsed),
            eta_str(avg_dur, stats.elapsed),
            f"{stats.attempts + 1}/{max_retries + 1}" if stats.attempts > 0 else "",
            style="on red" if stats.attempts > 0 else "",
        )
    return table


def build_layout(table: Table, log_panel: Panel) -> Layout:
    from rich.layout import Layout

    layout = Layout()
    layout.split_column(
        Layout(table, name="table", ratio=3),
        Layout(log_panel, name="log", ratio=1),
    )
    return layout


def render_overlay(
    run_id: str,
    active: Mapping[str, int],
    graph: MikadoGraph,
    agent: str,
    stdout_lines: list[str],
) -> Panel:
    from rich.panel import Panel

    node_id = active.get(run_id)
    if node_id is None:
        return Panel(
            "[dim]worker not found[/dim]", title="Overlay", border_style="cyan",
        )
    node = graph.get_node(node_id)
    branch = node.branch_name if node else None
    desc = (node.description if node else str(node_id))[:40]
    tail = stdout_lines[-OVERLAY_TAIL_LINES:]
    stdout = "\n".join(tail) if tail else "[dim]no output yet[/dim]"
    content = (
        f"[bold]Branch:[/bold] {branch or '(pending)'}\n"
        f"[bold]Agent:[/bold]  {agent}\n\n"
        f"{stdout}"
    )
    return Panel(
        content,
        title=f"Worker {node_id} — {desc}",
        border_style="cyan",
        expand=True,
    )
