from __future__ import annotations

import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.table import Table

    from milknado.domains.common.protocols import ProgressEvent, RalphPort
    from milknado.domains.graph import MikadoGraph

_SPINNER_FRAMES = ("◜", "◝", "◞", "◟")
_US_PREFIX_RE = re.compile(r"^US-\d+:\s*")


@dataclass
class TuiState:
    tick: int
    active: dict[str, int]
    logs: Sequence[str]
    dispatched_at: dict[str, float]
    attempts: dict[int, int]
    progress_by_run: dict[str, ProgressEvent]
    completion_durations: Sequence[float]
    stall_threshold: float
    max_retries: int
    exec_agent: str


@dataclass(frozen=True)
class _WorkerStats:
    elapsed: float
    pct: float | None
    attempts: int
    files: list[str]


def _elapsed_str(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def _eta_str(avg_duration: float | None, elapsed: float) -> str:
    if avg_duration is None:
        return "~?"
    remaining = max(0.0, avg_duration - elapsed)
    return f"~{_elapsed_str(remaining)}"


def _files_cell(files: list[str]) -> str:
    s = ", ".join(files)
    return s[:59] + "…" if len(s) > 60 else s


def _summarize_description(description: str, max_chars: int = 80) -> str:
    text = description.split("\n", 1)[0]
    text = _US_PREFIX_RE.sub("", text)
    text = " ".join(text.split())
    if len(text) > max_chars:
        text = text[: max_chars - 1] + "…"
    return text


def _render_progress_bar(
    frame: str, elapsed: float, pct: float | None, stall_threshold: float
) -> str:
    if pct is not None:
        clamped_pct = max(0.0, min(100.0, pct))
        filled = max(0, min(10, int(clamped_pct) // 10))
        bar = "█" * filled + "░" * (10 - filled)
        return f"[cyan]{bar}[/cyan] {int(clamped_pct)}%"
    if elapsed >= stall_threshold:
        return f"[yellow]{frame} ⚠[/yellow]"
    return f"[cyan]{frame}[/cyan]"


def _build_title(active: dict[str, int], graph: MikadoGraph) -> str:
    from milknado.domains.common.types import NodeStatus

    all_nodes = graph.get_all_nodes()
    done_c = sum(1 for n in all_nodes if n.status == NodeStatus.DONE)
    failed_c = sum(1 for n in all_nodes if n.status == NodeStatus.FAILED)
    blocked_c = sum(1 for n in all_nodes if n.status == NodeStatus.BLOCKED)
    return (
        f"milknado — {len(active)} active | "
        f"[green]{done_c} done[/green] | [red]{failed_c} failed[/red] | "
        f"[dim]{blocked_c} blocked[/dim]"
    )


def _worker_stats(
    run_id: str, state: TuiState, graph: MikadoGraph, now: float
) -> _WorkerStats:
    node_id = state.active[run_id]
    elapsed = now - state.dispatched_at.get(run_id, now)
    ev = state.progress_by_run.get(run_id)
    pct = ev.work / ev.total * 100 if ev and ev.total > 0 else None
    attempts = state.attempts.get(node_id, 0)
    files = graph.get_file_ownership(node_id)
    return _WorkerStats(elapsed=elapsed, pct=pct, attempts=attempts, files=files)


def _build_worker_table(state: TuiState, graph: MikadoGraph) -> Table:
    from rich.table import Table

    frame = _SPINNER_FRAMES[state.tick % len(_SPINNER_FRAMES)]
    now = time.monotonic()
    durations = list(state.completion_durations)
    avg_dur = sum(durations) / len(durations) if len(durations) >= 3 else None

    table = Table(
        title=_build_title(state.active, graph), show_header=True, header_style="bold"
    )
    table.add_column("", width=12, no_wrap=True)
    table.add_column("ID", style="cyan", width=4, no_wrap=True)
    table.add_column("Description")
    table.add_column("Files", width=62)
    table.add_column("Elapsed", width=7, no_wrap=True)
    table.add_column("ETA", width=6, no_wrap=True)
    table.add_column("Attempt", width=9, no_wrap=True)

    for run_id, node_id in state.active.items():
        node = graph.get_node(node_id)
        if not node:
            continue
        stats = _worker_stats(run_id, state, graph, now)
        table.add_row(
            _render_progress_bar(frame, stats.elapsed, stats.pct, state.stall_threshold),
            str(node_id),
            _summarize_description(node.description),
            _files_cell(stats.files),
            _elapsed_str(stats.elapsed),
            _eta_str(avg_dur, stats.elapsed),
            f"{stats.attempts + 1}/{state.max_retries + 1}" if stats.attempts > 0 else "",
            style="on red" if stats.attempts > 0 else "",
        )
    return table


def _build_log_panel(logs: Sequence[str]) -> Panel:
    from rich.panel import Panel

    content = "\n".join(logs) or "[dim]No events yet[/dim]"
    return Panel(content, title="Log", border_style="dim", expand=True)


def _build_layout(state: TuiState, graph: MikadoGraph) -> Layout:
    from rich.layout import Layout

    layout = Layout()
    layout.split_column(
        Layout(_build_worker_table(state, graph), name="table", ratio=3),
        Layout(_build_log_panel(state.logs), name="log", ratio=1),
    )
    return layout


def _render_overlay(
    run_id: str, state: TuiState, graph: MikadoGraph, ralph: RalphPort
) -> Panel:
    from rich.panel import Panel

    node_id = state.active.get(run_id)
    if node_id is None:
        return Panel("[dim]worker not found[/dim]", title="Overlay", border_style="cyan")
    node = graph.get_node(node_id)
    branch = node.branch_name if node else None
    desc = (node.description if node else str(node_id))[:40]
    lines = ralph.get_run_stdout(run_id)[-100:]
    stdout = "\n".join(lines) if lines else "[dim]no output yet[/dim]"
    content = (
        f"[bold]Branch:[/bold] {branch or '(pending)'}\n"
        f"[bold]Agent:[/bold]  {state.exec_agent}\n\n"
        f"{stdout}"
    )
    return Panel(
        content,
        title=f"Worker {node_id} — {desc}",
        border_style="cyan",
        expand=True,
    )
