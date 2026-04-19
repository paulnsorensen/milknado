from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from milknado.domains.common.protocols import ProgressEvent
from milknado.domains.common.types import NodeStatus
from milknado.domains.execution.executor import RebaseConflict, get_dispatchable_nodes

if TYPE_CHECKING:
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table

    from milknado.domains.common.config import MilknadoConfig
    from milknado.domains.common.protocols import RalphPort
    from milknado.domains.execution.executor import ExecutionConfig, Executor
    from milknado.domains.graph import MikadoGraph
    from milknado.domains.planning.planner import Planner

_logger = logging.getLogger(__name__)
_SPINNER_FRAMES = ("◜", "◝", "◞", "◟")


@dataclass(frozen=True)
class RunLoopResult:
    root_done: bool
    dispatched_total: int
    completed_total: int
    failed_total: int
    rebase_conflicts: tuple[RebaseConflict, ...] = ()
    strict_exit: bool = False


def _configure_run_logging(project_root: Path) -> Path:
    log_dir = project_root / ".milknado"
    log_dir.mkdir(parents=True, exist_ok=True)
    iso = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    log_path = log_dir / f"run-{iso}.log"
    handler = logging.FileHandler(str(log_path), encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    handler.setLevel(logging.DEBUG)
    logging.getLogger("milknado").addHandler(handler)
    return log_path


def _elapsed_str(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def _eta_str(avg_duration: float | None, elapsed: float) -> str:
    if avg_duration is None:
        return "~?"
    remaining = max(0.0, avg_duration - elapsed)
    return f"~{_elapsed_str(remaining)}"


def _spinner_cell(frame: str, elapsed: float, stall: int, pct: float | None) -> str:
    if pct is not None:
        filled = int(pct) // 10
        bar = "█" * filled + "░" * (10 - filled)
        return f"[cyan]{bar}[/cyan] {int(pct)}%"
    if elapsed > stall:
        return f"[yellow]{frame} ⚠[/yellow]"
    return f"[cyan]{frame}[/cyan]"


def _files_cell(files: list[str]) -> str:
    s = ", ".join(files)
    return s[:59] + "…" if len(s) > 60 else s


def _ts() -> str:
    return datetime.now(UTC).strftime("%H:%M:%S")


class RunLoop:
    def __init__(
        self,
        executor: Executor,
        graph: MikadoGraph,
        ralph: RalphPort,
        config: MilknadoConfig | None = None,
        planner: Planner | None = None,
    ) -> None:
        self._executor = executor
        self._graph = graph
        self._ralph = ralph
        self._milknado_config = config
        self._planner = planner
        self._active: dict[str, int] = {}
        self._log_buffer: deque[str] = deque(maxlen=30)
        self._start_times: dict[str, float] = {}
        self._attempts: dict[int, int] = {}
        self._failure_triggered: bool = False
        self._progress: dict[str, ProgressEvent] = {}
        self._strict: bool = False
        self._tick: int = 0

    def run(
        self,
        config: ExecutionConfig,
        feature_branch: str,
        concurrency_limit: int = 4,
        strict: bool = False,
        spec_text: str | None = None,
        spec_path: Path | None = None,
    ) -> RunLoopResult:
        from rich.layout import Layout
        from rich.live import Live

        self._strict = strict
        layout = Layout()
        layout.split_column(Layout(name="table", ratio=3), Layout(name="log", ratio=1))

        dispatched = completed = failed = 0
        conflicts: list[RebaseConflict] = []

        with Live(layout, refresh_per_second=2) as live:
            self._refresh(layout, live)
            dispatched += self._dispatch_batch(config, concurrency_limit, live)
            self._refresh(layout, live)
            while self._active:
                run_id, success = self._ralph.wait_for_next_completion(
                    set(self._active.keys()), timeout=1800.0,
                )
                self._absorb_progress()
                c, f, cs = self._handle_completion(run_id, success, feature_branch, live)
                completed += c
                failed += f
                conflicts.extend(cs)
                if not (self._strict and self._failure_triggered):
                    dispatched += self._dispatch_batch(config, concurrency_limit, live)
                self._refresh(layout, live)

        self._try_verify_root(spec_text, spec_path, config)
        root = self._graph.get_root()
        return RunLoopResult(
            root_done=root is not None and root.status == NodeStatus.DONE,
            dispatched_total=dispatched,
            completed_total=completed,
            failed_total=failed,
            rebase_conflicts=tuple(conflicts),
            strict_exit=strict and self._failure_triggered,
        )

    def _refresh(self, layout: Layout, live: Live) -> None:
        layout["table"].update(self._build_table())
        layout["log"].update(self._build_log_panel())
        live.update(layout)

    def _absorb_progress(self) -> None:
        for ev in self._ralph.poll_progress_events():
            self._progress[ev.run_id] = ev

    def _try_verify_root(
        self, spec_text: str | None, spec_path: Path | None, config: ExecutionConfig,
    ) -> None:
        if not spec_text or self._failure_triggered:
            return
        root = self._graph.get_root()
        if not root or root.status != NodeStatus.PENDING:
            return
        result = self._ralph.verify_spec(spec_text, str(self._graph))
        if result.outcome == "done":
            self._graph.mark_done(root.id)
        elif result.outcome == "gaps" and self._planner and result.goal_delta:
            self._planner.replan_with_delta(result.goal_delta, config.project_root, spec_path)

    def _build_table(self) -> Table:
        from rich.table import Table

        self._tick += 1
        frame = _SPINNER_FRAMES[self._tick % len(_SPINNER_FRAMES)]
        now = time.monotonic()

        all_nodes = self._graph.get_all_nodes()
        done_c = sum(1 for n in all_nodes if n.status == NodeStatus.DONE)
        failed_c = sum(1 for n in all_nodes if n.status == NodeStatus.FAILED)
        blocked_c = sum(1 for n in all_nodes if n.status == NodeStatus.BLOCKED)

        cfg = self._milknado_config
        stall = cfg.stall_threshold_seconds if cfg else 300
        eta_n = cfg.eta_sample_size if cfg else 10
        max_r = cfg.dispatch_max_retries if cfg else 2

        durations = self._graph.recent_completion_durations(eta_n)
        avg_dur = sum(durations) / len(durations) if len(durations) >= 3 else None

        title = (
            f"milknado — {len(self._active)} active | "
            f"[green]{done_c} done[/green] | [red]{failed_c} failed[/red] | "
            f"[dim]{blocked_c} blocked[/dim]"
        )
        table = Table(title=title, show_header=True, header_style="bold")
        table.add_column("", width=12, no_wrap=True)
        table.add_column("ID", style="cyan", width=4, no_wrap=True)
        table.add_column("Description")
        table.add_column("Files", width=62)
        table.add_column("Elapsed", width=7, no_wrap=True)
        table.add_column("ETA", width=6, no_wrap=True)
        table.add_column("Attempt", width=9, no_wrap=True)

        for run_id, node_id in self._active.items():
            node = self._graph.get_node(node_id)
            if not node:
                continue
            elapsed = now - self._start_times.get(run_id, now)
            ev = self._progress.get(run_id)
            attempts = self._attempts.get(node_id, 0)
            files = self._graph.get_file_ownership(node_id)
            table.add_row(
                _spinner_cell(
                    frame, elapsed, stall,
                    ev.work / ev.total * 100 if ev and ev.total > 0 else None,
                ),
                str(node_id),
                node.description,
                _files_cell(files),
                _elapsed_str(elapsed),
                _eta_str(avg_dur, elapsed),
                f"{attempts + 1}/{max_r + 1}" if attempts > 0 else "",
                style="on red" if attempts > 0 else "",
            )
        return table

    def _build_log_panel(self) -> Panel:
        from rich.panel import Panel

        content = "\n".join(self._log_buffer) or "[dim]No events yet[/dim]"
        return Panel(content, title="Log", border_style="dim", expand=True)

    def _overlay_for_worker(self, worker_id: str) -> None:  # noqa: ARG002
        # Prototype-stage: overlay not yet implemented.
        # Future: poll .milknado/overlay-{worker_id}.sentinel for key input.
        pass

    def _dispatch_batch(
        self,
        config: ExecutionConfig,
        concurrency_limit: int,
        live: Live,
    ) -> int:
        if self._strict and self._failure_triggered:
            return 0
        available = concurrency_limit - len(self._active)
        if available <= 0:
            return 0
        dispatchable = get_dispatchable_nodes(self._graph)
        dispatched = 0
        for node_id in dispatchable[:available]:
            node = self._graph.get_node(node_id)
            desc = node.description if node else str(node_id)
            try:
                result = self._executor.dispatch(node_id, config)
            except Exception as exc:
                _logger.exception(
                    "Dispatch failed for node %d (%s): %s: %s",
                    node_id, desc, type(exc).__name__, exc,
                )
                self._executor.fail(node_id)
                self._log_buffer.append(
                    f"[{_ts()}] ✗ dispatch node {node_id}: {type(exc).__name__}"
                )
                continue
            self._active[result.run_id] = node_id
            self._start_times[result.run_id] = time.monotonic()
            self._log_buffer.append(f"[{_ts()}] → node {node_id}: {desc[:50]}")
            live.console.print(f"[cyan]→[/cyan] [{node_id}] {desc}")
            dispatched += 1
        return dispatched

    def _handle_completion(
        self,
        run_id: str,
        success: bool,
        feature_branch: str,
        live: Live,
    ) -> tuple[int, int, list[RebaseConflict]]:
        completed = failed = 0
        conflicts: list[RebaseConflict] = []

        node_id = self._active.pop(run_id)
        node = self._graph.get_node(node_id)
        desc = node.description if node else str(node_id)
        start = self._start_times.pop(run_id, time.monotonic())
        duration = time.monotonic() - start

        if success:
            result = self._executor.complete(node_id, feature_branch)
            self._graph._record_completion_duration(node_id, duration)
            if result.rebase_conflict:
                conflicts.append(result.rebase_conflict)
                files = ", ".join(result.rebase_conflict.conflicting_files)
                live.console.print(f"[red]✗[/red] [{node_id}] {desc} — conflict: {files}")
                self._log_buffer.append(f"[{_ts()}] ✗ node {node_id} conflict")
                self._attempts[node_id] = self._attempts.get(node_id, 0) + 1
                if self._strict:
                    self._failure_triggered = True
                failed += 1
            else:
                live.console.print(f"[green]✓[/green] [{node_id}] {desc}")
                self._log_buffer.append(f"[{_ts()}] ✓ node {node_id} in {int(duration)}s")
                completed += 1
        else:
            self._executor.fail(node_id)
            live.console.print(f"[red]✗[/red] [{node_id}] {desc}")
            self._log_buffer.append(f"[{_ts()}] ✗ node {node_id} failed")
            self._attempts[node_id] = self._attempts.get(node_id, 0) + 1
            if self._strict:
                self._failure_triggered = True
            failed += 1

        return completed, failed, conflicts
