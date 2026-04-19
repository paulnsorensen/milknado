from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from milknado.domains.common.errors import CompletionTimeout
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
_ETA_SAMPLE_SIZE_DEFAULT = 10
_STALL_THRESHOLD_DEFAULT = 300


@dataclass(frozen=True)
class VerifyOutcome:
    done: bool
    goal_delta: str | None = None


@dataclass(frozen=True)
class RunLoopResult:
    root_done: bool
    dispatched_total: int
    completed_total: int
    failed_total: int
    rebase_conflicts: tuple[RebaseConflict, ...] = ()
    strict_exit: bool = False
    verify_outcome: VerifyOutcome | None = None


@dataclass(frozen=True)
class _WorkerStats:
    elapsed: float
    pct: float | None
    attempts: int
    files: list[str]


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
        self._logs: deque[str] = deque(maxlen=30)
        self._dispatched_at: dict[str, float] = {}
        self._attempts: dict[int, int] = {}
        self._failure_triggered: bool = False
        self._progress_by_run: dict[str, ProgressEvent] = {}
        self._strict: bool = False
        self._tick: int = 0
        eta_n = config.eta_sample_size if config else _ETA_SAMPLE_SIZE_DEFAULT
        self._completion_durations: deque[float] = deque(maxlen=eta_n)
        self._overlay_state: str | None = None
        self._awaiting_node_digits: bool = False
        self._key_buffer: str = ""
        self._input_queue: queue.Queue[str] = queue.Queue()
        self._input_stop: threading.Event = threading.Event()
        self._input_thread: threading.Thread | None = None
        self._exec_config: ExecutionConfig | None = None

    def run(
        self,
        config: ExecutionConfig,
        feature_branch: str,
        concurrency_limit: int = 4,
        strict: bool = False,
        spec_text: str | None = None,
        spec_path: Path | None = None,
    ) -> RunLoopResult:
        from rich.live import Live

        self._strict = strict
        self._exec_config = config
        dispatched = completed = failed = 0
        conflicts: list[RebaseConflict] = []

        timeout = (
            self._milknado_config.completion_timeout_seconds
            if self._milknado_config
            else 1800.0
        )
        self._input_stop.clear()
        self._start_input_thread()
        try:
            with Live(self._build_layout(), refresh_per_second=2) as live:
                dispatched += self._dispatch_batch(config, concurrency_limit, live)
                self._drain_input()
                display = (
                    self._render_overlay(self._overlay_state)
                    if self._overlay_state else self._build_layout()
                )
                live.update(display)
                while self._active:
                    try:
                        run_id, success = self._ralph.wait_for_next_completion(
                            set(self._active.keys()), timeout=timeout,
                        )
                    except CompletionTimeout as ct:
                        _logger.warning(
                            "Completion timeout after %.1fs; active runs: %s",
                            ct.waited_seconds, sorted(ct.active_run_ids),
                        )
                        for timed_out_id in list(self._active):
                            nid = self._active.pop(timed_out_id)
                            self._executor.fail(nid)
                            self._append_log(f"[{_ts()}] ⏱ node {nid} timeout")
                            failed += 1
                        if self._strict:
                            self._failure_triggered = True
                        break
                    self._absorb_progress()
                    self._drain_input()
                    c, f, cs = self._handle_completion(run_id, success, feature_branch, live)
                    completed += c
                    failed += f
                    conflicts.extend(cs)
                    if not (self._strict and self._failure_triggered):
                        dispatched += self._dispatch_batch(config, concurrency_limit, live)
                    display = (
                        self._render_overlay(self._overlay_state)
                        if self._overlay_state else self._build_layout()
                    )
                    live.update(display)
        finally:
            self._stop_input_thread()

        verify_outcome = self._maybe_verify_spec(spec_text, spec_path, config)
        root = self._graph.get_root()
        return RunLoopResult(
            root_done=root is not None and root.status == NodeStatus.DONE,
            dispatched_total=dispatched,
            completed_total=completed,
            failed_total=failed,
            rebase_conflicts=tuple(conflicts),
            strict_exit=strict and self._failure_triggered,
            verify_outcome=verify_outcome,
        )

    def _absorb_progress(self) -> None:
        for ev in self._ralph.poll_progress_events():
            self._progress_by_run[ev.run_id] = ev

    def _append_log(self, msg: str) -> None:
        self._logs.append(msg)

    def _is_stalled(self, elapsed: float) -> bool:
        cfg = self._milknado_config
        stall = cfg.stall_threshold_seconds if cfg else _STALL_THRESHOLD_DEFAULT
        return elapsed >= stall

    def _render_progress_bar(self, frame: str, elapsed: float, pct: float | None) -> str:
        if pct is not None:
            filled = int(pct) // 10
            bar = "█" * filled + "░" * (10 - filled)
            return f"[cyan]{bar}[/cyan] {int(pct)}%"
        if self._is_stalled(elapsed):
            return f"[yellow]{frame} ⚠[/yellow]"
        return f"[cyan]{frame}[/cyan]"

    def _worker_stats_for(self, run_id: str, now: float) -> _WorkerStats:
        node_id = self._active[run_id]
        elapsed = now - self._dispatched_at.get(run_id, now)
        ev = self._progress_by_run.get(run_id)
        pct = ev.work / ev.total * 100 if ev and ev.total > 0 else None
        attempts = self._attempts.get(node_id, 0)
        files = self._graph.get_file_ownership(node_id)
        return _WorkerStats(elapsed=elapsed, pct=pct, attempts=attempts, files=files)

    def _build_title(self) -> str:
        all_nodes = self._graph.get_all_nodes()
        done_c = sum(1 for n in all_nodes if n.status == NodeStatus.DONE)
        failed_c = sum(1 for n in all_nodes if n.status == NodeStatus.FAILED)
        blocked_c = sum(1 for n in all_nodes if n.status == NodeStatus.BLOCKED)
        return (
            f"milknado — {len(self._active)} active | "
            f"[green]{done_c} done[/green] | [red]{failed_c} failed[/red] | "
            f"[dim]{blocked_c} blocked[/dim]"
        )

    def _build_worker_table(self) -> Table:
        from rich.table import Table

        self._tick += 1
        frame = _SPINNER_FRAMES[self._tick % len(_SPINNER_FRAMES)]
        now = time.monotonic()
        cfg = self._milknado_config
        max_r = cfg.dispatch_max_retries if cfg else 2

        durations = list(self._completion_durations)
        avg_dur = sum(durations) / len(durations) if len(durations) >= 3 else None

        table = Table(title=self._build_title(), show_header=True, header_style="bold")
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
            stats = self._worker_stats_for(run_id, now)
            table.add_row(
                self._render_progress_bar(frame, stats.elapsed, stats.pct),
                str(node_id),
                node.description,
                _files_cell(stats.files),
                _elapsed_str(stats.elapsed),
                _eta_str(avg_dur, stats.elapsed),
                f"{stats.attempts + 1}/{max_r + 1}" if stats.attempts > 0 else "",
                style="on red" if stats.attempts > 0 else "",
            )
        return table

    def _build_log_panel(self) -> Panel:
        from rich.panel import Panel

        content = "\n".join(self._logs) or "[dim]No events yet[/dim]"
        return Panel(content, title="Log", border_style="dim", expand=True)

    def _build_layout(self) -> Layout:
        from rich.layout import Layout

        layout = Layout()
        layout.split_column(
            Layout(self._build_worker_table(), name="table", ratio=3),
            Layout(self._build_log_panel(), name="log", ratio=1),
        )
        return layout

    # ── US-004: drill-in overlay ──────────────────────────────────────────

    def _start_input_thread(self) -> None:
        import sys
        if not sys.stdin.isatty():
            return
        try:
            import select
            import termios
            import tty
        except ImportError:
            return
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        stop = self._input_stop

        def _read_keys() -> None:
            try:
                while not stop.is_set():
                    readable, _, _ = select.select([sys.stdin], [], [], 0.1)
                    if readable:
                        ch = sys.stdin.read(1)
                        if ch:
                            self._input_queue.put(ch)
            finally:
                try:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                except Exception:  # noqa: BLE001
                    pass

        self._input_thread = threading.Thread(
            target=_read_keys, daemon=True, name="milknado-input",
        )
        self._input_thread.start()

    def _stop_input_thread(self) -> None:
        self._input_stop.set()
        if self._input_thread is not None:
            self._input_thread.join(timeout=0.5)
            self._input_thread = None

    def _handle_key(self, key: str) -> None:
        if key == "\x1b":
            self._overlay_state = None
            self._awaiting_node_digits = False
            self._key_buffer = ""
            return
        if key in ("l", "L") and not self._awaiting_node_digits:
            self._awaiting_node_digits = True
            self._key_buffer = ""
            return
        if self._awaiting_node_digits:
            if key.isdigit():
                self._key_buffer += key
            elif key in ("\r", "\n"):
                self._awaiting_node_digits = False
                if self._key_buffer:
                    target = int(self._key_buffer)
                    for run_id, node_id in self._active.items():
                        if node_id == target:
                            self._overlay_state = run_id
                            break
                self._key_buffer = ""

    def _drain_input(self) -> None:
        try:
            while True:
                self._handle_key(self._input_queue.get_nowait())
        except queue.Empty:
            pass

    def _render_overlay(self, run_id: str) -> Panel:
        from rich.panel import Panel

        node_id = self._active.get(run_id)
        if node_id is None:
            return Panel("[dim]worker not found[/dim]", title="Overlay", border_style="cyan")
        node = self._graph.get_node(node_id)
        branch = node.branch_name if node else None
        desc = (node.description if node else str(node_id))[:40]
        agent = self._exec_config.execution_agent if self._exec_config else "(unknown)"
        lines = self._ralph.get_run_stdout(run_id)[-100:]
        stdout = "\n".join(lines) if lines else "[dim]no output yet[/dim]"
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

    def _maybe_verify_spec(
        self, spec_text: str | None, spec_path: Path | None, config: ExecutionConfig,
    ) -> VerifyOutcome | None:
        if not spec_text or self._failure_triggered or self._active:
            return None
        root = self._graph.get_root()
        if root is None or root.status == NodeStatus.DONE:
            return None
        non_root_all_done = all(
            n.status == NodeStatus.DONE
            for n in self._graph.get_all_nodes()
            if n.id != root.id
        )
        if not non_root_all_done:
            return None
        result = self._ralph.verify_spec(spec_text, str(self._graph))
        outcome = VerifyOutcome(done=result.outcome == "done", goal_delta=result.goal_delta)
        if result.outcome == "done":
            self._graph.mark_done(root.id)
        elif result.outcome == "gaps" and self._planner and result.goal_delta:
            self._planner.replan_with_delta(result.goal_delta, config.project_root, spec_path)
        return outcome

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
                self._append_log(f"[{_ts()}] ✗ dispatch node {node_id}: {type(exc).__name__}")
                continue
            self._active[result.run_id] = node_id
            self._dispatched_at[result.run_id] = time.monotonic()
            self._append_log(f"[{_ts()}] → node {node_id}: {desc[:50]}")
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
        if self._overlay_state == run_id:
            self._overlay_state = None
        node = self._graph.get_node(node_id)
        desc = node.description if node else str(node_id)
        start = self._dispatched_at.pop(run_id, time.monotonic())
        duration = time.monotonic() - start

        if success:
            result = self._executor.complete(node_id, feature_branch)
            self._completion_durations.append(duration)
            self._graph._record_completion_duration(node_id, duration)
            if result.rebase_conflict:
                conflicts.append(result.rebase_conflict)
                files = ", ".join(result.rebase_conflict.conflicting_files)
                live.console.print(f"[red]✗[/red] [{node_id}] {desc} — conflict: {files}")
                self._append_log(f"[{_ts()}] ✗ node {node_id} conflict")
                self._attempts[node_id] = self._attempts.get(node_id, 0) + 1
                if self._strict:
                    self._failure_triggered = True
                failed += 1
            else:
                live.console.print(f"[green]✓[/green] [{node_id}] {desc}")
                self._append_log(f"[{_ts()}] ✓ node {node_id} in {int(duration)}s")
                completed += 1
        else:
            self._executor.fail(node_id)
            live.console.print(f"[red]✗[/red] [{node_id}] {desc}")
            self._append_log(f"[{_ts()}] ✗ node {node_id} failed")
            self._attempts[node_id] = self._attempts.get(node_id, 0) + 1
            if self._strict:
                self._failure_triggered = True
            failed += 1

        return completed, failed, conflicts
