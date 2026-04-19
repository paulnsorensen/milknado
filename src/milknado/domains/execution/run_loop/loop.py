from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

from milknado.domains.common.protocols import ProgressEvent
from milknado.domains.common.types import NodeStatus
from milknado.domains.execution.executor import RebaseConflict, get_dispatchable_nodes
from milknado.domains.execution.run_loop import display
from milknado.domains.execution.run_loop.input import InputController

if TYPE_CHECKING:
    from rich.console import RenderableType
    from rich.live import Live

    from milknado.domains.common.config import MilknadoConfig
    from milknado.domains.common.protocols import RalphPort
    from milknado.domains.execution.executor import ExecutionConfig, Executor
    from milknado.domains.graph import MikadoGraph


@dataclass(frozen=True)
class RunLoopResult:
    root_done: bool
    dispatched_total: int
    completed_total: int
    failed_total: int
    rebase_conflicts: tuple[RebaseConflict, ...] = ()


class RunLoop:
    def __init__(
        self,
        executor: Executor,
        graph: MikadoGraph,
        ralph: RalphPort,
        config: MilknadoConfig | None = None,
    ) -> None:
        self._executor = executor
        self._graph = graph
        self._ralph = ralph
        self._milknado_config = config
        self._active: dict[str, int] = {}
        self._logs: deque[str] = deque(maxlen=30)
        self._dispatched_at: dict[str, float] = {}
        self._attempts: dict[int, int] = {}
        self._progress_by_run: dict[str, ProgressEvent] = {}
        self._tick: int = 0
        eta_n = getattr(config, "eta_sample_size", display.ETA_SAMPLE_SIZE_DEFAULT)
        self._completion_durations: deque[float] = deque(maxlen=eta_n)
        self._overlay_state: str | None = None
        self._exec_config: ExecutionConfig | None = None
        self._input = InputController(
            on_overlay=self._open_overlay_for_node,
            on_clear=self._clear_overlay,
        )

    def run(
        self,
        config: ExecutionConfig,
        feature_branch: str,
        concurrency_limit: int = 4,
    ) -> RunLoopResult:
        from rich.live import Live

        self._exec_config = config
        dispatched = completed = failed = 0
        conflicts: list[RebaseConflict] = []

        self._input.start()
        try:
            with Live(self._render(), refresh_per_second=2) as live:
                dispatched += self._dispatch_batch(config, concurrency_limit, live)
                self._input.drain()
                live.update(self._render())
                while self._active:
                    run_id, success = self._ralph.wait_for_next_completion(
                        set(self._active.keys()),
                    )
                    self._absorb_progress()
                    self._input.drain()
                    c, f, cs = self._handle_completion(
                        run_id, success, feature_branch, live,
                    )
                    completed += c
                    failed += f
                    conflicts.extend(cs)
                    dispatched += self._dispatch_batch(
                        config, concurrency_limit, live,
                    )
                    live.update(self._render())
        finally:
            self._input.stop()

        root = self._graph.get_root()
        return RunLoopResult(
            root_done=root is not None and root.status == NodeStatus.DONE,
            dispatched_total=dispatched,
            completed_total=completed,
            failed_total=failed,
            rebase_conflicts=tuple(conflicts),
        )

    # ── Rendering ─────────────────────────────────────────────────────────

    def _render(self) -> RenderableType:
        self._tick += 1
        if self._overlay_state is not None:
            return self._build_overlay(self._overlay_state)
        return self._build_layout()

    def _build_layout(self) -> RenderableType:
        cfg = self._milknado_config
        stall = getattr(cfg, "stall_threshold_seconds", display.STALL_THRESHOLD_DEFAULT)
        max_r = getattr(cfg, "dispatch_max_retries", display.MAX_RETRIES_DEFAULT)
        table = display.build_worker_table(
            tick=self._tick,
            now=time.monotonic(),
            active=self._active,
            dispatched_at=self._dispatched_at,
            attempts=self._attempts,
            progress_by_run=self._progress_by_run,
            completion_durations=self._completion_durations,
            graph=self._graph,
            stall_threshold=stall,
            max_retries=max_r,
        )
        return display.build_layout(table, display.build_log_panel(self._logs))

    def _build_overlay(self, run_id: str) -> RenderableType:
        agent = (
            self._exec_config.execution_agent if self._exec_config else "(unknown)"
        )
        stdout = self._ralph.get_run_stdout(run_id)
        return display.render_overlay(
            run_id=run_id,
            active=self._active,
            graph=self._graph,
            agent=agent,
            stdout_lines=stdout,
        )

    # ── Overlay callbacks for InputController ─────────────────────────────

    def _open_overlay_for_node(self, target_node_id: int) -> None:
        for run_id, node_id in self._active.items():
            if node_id == target_node_id:
                self._overlay_state = run_id
                return

    def _clear_overlay(self) -> None:
        self._overlay_state = None

    # ── Core loop helpers ─────────────────────────────────────────────────

    def _absorb_progress(self) -> None:
        for ev in self._ralph.poll_progress_events():
            self._progress_by_run[ev.run_id] = ev

    def _dispatch_batch(
        self,
        config: ExecutionConfig,
        concurrency_limit: int,
        live: Live,
    ) -> int:
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
                live.console.print(
                    f"[red]✗[/red] [{node_id}] {desc} — dispatch failed: {exc}",
                )
                self._executor.fail(node_id)
                self._logs.append(
                    f"[{display.ts()}] ✗ dispatch node {node_id}: {type(exc).__name__}",
                )
                continue
            self._active[result.run_id] = node_id
            self._dispatched_at[result.run_id] = time.monotonic()
            self._logs.append(f"[{display.ts()}] → node {node_id}: {desc[:50]}")
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
            if result.rebase_conflict:
                conflicts.append(result.rebase_conflict)
                files = ", ".join(result.rebase_conflict.conflicting_files)
                live.console.print(
                    f"[red]✗[/red] [{node_id}] {desc} — rebase conflict: {files}",
                )
                self._logs.append(f"[{display.ts()}] ✗ node {node_id} conflict")
                self._attempts[node_id] = self._attempts.get(node_id, 0) + 1
                failed += 1
            else:
                live.console.print(f"[green]✓[/green] [{node_id}] {desc}")
                self._logs.append(
                    f"[{display.ts()}] ✓ node {node_id} in {int(duration)}s",
                )
                completed += 1
        else:
            self._executor.fail(node_id)
            live.console.print(f"[red]✗[/red] [{node_id}] {desc}")
            self._logs.append(f"[{display.ts()}] ✗ node {node_id} failed")
            self._attempts[node_id] = self._attempts.get(node_id, 0) + 1
            failed += 1

        return completed, failed, conflicts
