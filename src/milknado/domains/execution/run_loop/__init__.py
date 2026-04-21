from __future__ import annotations

import json
import logging
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

from milknado.domains.common.errors import CompletionTimeout
from milknado.domains.common.protocols import ProgressEvent
from milknado.domains.common.types import NodeStatus
from milknado.domains.execution.executor import RebaseConflict, get_dispatchable_nodes
from milknado.domains.execution.run_loop._completion import handle_completion
from milknado.domains.execution.run_loop._logging import configure_run_logging, ts
from milknado.domains.execution.run_loop._result import RunLoopResult, VerifyOutcome
from milknado.domains.execution.run_loop.display import (
    TuiState,
    _build_layout,
    _render_overlay,
    _summarize_description,
)
from milknado.domains.execution.run_loop.input import (
    InputState,
    drain_input,
    start_input_thread,
    stop_input_thread,
)

if TYPE_CHECKING:
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel

    from milknado.domains.common.config import MilknadoConfig
    from milknado.domains.common.protocols import RalphPort
    from milknado.domains.execution.executor import ExecutionConfig, Executor
    from milknado.domains.graph import MikadoGraph
    from milknado.domains.planning.planner import Planner

_logger = logging.getLogger("milknado")
_ETA_SAMPLE_SIZE_DEFAULT = 10
_STALL_THRESHOLD_DEFAULT = 300


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
        self._input: InputState = InputState()
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
        from rich.console import Console

        self._strict = strict
        self._exec_config = config
        timeout = (
            self._milknado_config.completion_timeout_seconds if self._milknado_config else 1800.0
        )

        with configure_run_logging(config.project_root) as log_path:
            Console().print(f"[dim]Log → {log_path}[/dim]")
            _logger.info("Run started feature_branch=%s", feature_branch)
            self._input.input_stop.clear()
            start_input_thread(self._input)
            dispatched, completed, failed, conflicts, interrupted = self._execute_run(
                config, feature_branch, concurrency_limit, timeout
            )

        self._emit_final_telemetry(dispatched, completed, failed, conflicts, interrupted)
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

    def _execute_run(
        self,
        config: ExecutionConfig,
        feature_branch: str,
        concurrency_limit: int,
        timeout: float,
    ) -> tuple[int, int, int, list[RebaseConflict], bool]:
        from rich.live import Live

        dispatched = completed = failed = 0
        conflicts: list[RebaseConflict] = []
        interrupted = False
        try:
            with Live(self._build_layout(), refresh_per_second=2) as live:
                d, f = self._dispatch_batch(config, concurrency_limit, live)
                dispatched += d
                failed += f
                drain_input(self._input, self._active)
                self._render_live_frame(live)
                while self._active:
                    try:
                        run_id, success = self._ralph.wait_for_next_completion(
                            set(self._active.keys()), timeout=timeout
                        )
                    except CompletionTimeout as ct:
                        failed += self._handle_completion_timeout(ct)
                        break
                    self._absorb_progress()
                    drain_input(self._input, self._active)
                    c, f, cs = handle_completion(self, run_id, success, feature_branch, live)
                    completed += c
                    failed += f
                    conflicts.extend(cs)
                    if not (self._strict and self._failure_triggered):
                        d, f = self._dispatch_batch(config, concurrency_limit, live)
                        dispatched += d
                        failed += f
                    self._render_live_frame(live)
        except KeyboardInterrupt:
            interrupted = True
            _logger.warning("Run interrupted by user (KeyboardInterrupt)")
            raise
        finally:
            stop_input_thread(self._input)
        return dispatched, completed, failed, conflicts, interrupted

    def _handle_completion_timeout(self, ct: CompletionTimeout) -> int:
        _logger.warning(
            "Completion timeout after %.1fs; active runs: %s",
            ct.waited_seconds,
            sorted(ct.active_run_ids),
        )
        newly_failed = 0
        for timed_out_id in list(self._active):
            nid = self._active.pop(timed_out_id)
            self._executor.fail(nid)
            self._logs.append(f"[{ts()}] ⏱ node {nid} timeout")
            newly_failed += 1
        if self._strict:
            self._failure_triggered = True
        return newly_failed

    def _render_live_frame(self, live: Live) -> None:
        display = (
            self._render_overlay(self._input.overlay_state)
            if self._input.overlay_state
            else self._build_layout()
        )
        live.update(display)

    def _emit_final_telemetry(
        self,
        dispatched: int,
        completed: int,
        failed: int,
        conflicts: list[RebaseConflict],
        interrupted: bool,
    ) -> None:
        root_node = self._graph.get_root()
        _logger.info(
            "FINAL_TELEMETRY %s",
            json.dumps(
                {
                    "dispatched": dispatched,
                    "completed": completed,
                    "failed": failed,
                    "conflicts": len(conflicts),
                    "root_done": root_node is not None and root_node.status == NodeStatus.DONE,
                    "strict_exit": self._strict and self._failure_triggered,
                    "interrupted": interrupted,
                }
            ),
        )

    def _absorb_progress(self) -> None:
        for ev in self._ralph.poll_progress_events():
            self._progress_by_run[ev.run_id] = ev

    def _tui_state(self) -> TuiState:
        cfg = self._milknado_config
        return TuiState(
            tick=self._tick,
            active=self._active,
            logs=list(self._logs),
            dispatched_at=self._dispatched_at,
            attempts=self._attempts,
            progress_by_run=self._progress_by_run,
            completion_durations=list(self._completion_durations),
            stall_threshold=cfg.stall_threshold_seconds if cfg else _STALL_THRESHOLD_DEFAULT,
            max_retries=cfg.dispatch_max_retries if cfg else 2,
            exec_agent=self._exec_config.execution_agent if self._exec_config else "(unknown)",
        )

    def _build_layout(self) -> Layout:
        self._tick += 1
        return _build_layout(self._tui_state(), self._graph)

    def _render_overlay(self, run_id: str) -> Panel:
        return _render_overlay(run_id, self._tui_state(), self._graph, self._ralph)

    def _maybe_verify_spec(
        self,
        spec_text: str | None,
        spec_path: Path | None,
        config: ExecutionConfig,
    ) -> VerifyOutcome | None:
        if not spec_text or self._failure_triggered or self._active:
            return None
        root = self._graph.get_root()
        if root is None or root.status == NodeStatus.DONE:
            return None
        non_root_all_done = all(
            n.status == NodeStatus.DONE for n in self._graph.get_all_nodes() if n.id != root.id
        )
        if not non_root_all_done:
            return None
        result = self._ralph.verify_spec(spec_text, str(self._graph))
        outcome = VerifyOutcome(done=result.outcome == "done", goal_delta=result.goal_delta)
        if result.outcome == "done":
            self._graph.mark_running(root.id)
            self._graph.mark_done(root.id)
        elif result.outcome == "gaps" and self._planner and result.goal_delta:
            self._planner.replan_with_delta(result.goal_delta, config.project_root, spec_path)
        return outcome

    def _dispatch_batch(
        self,
        config: ExecutionConfig,
        concurrency_limit: int,
        live: Live,
    ) -> tuple[int, int]:
        if self._strict and self._failure_triggered:
            return 0, 0
        available = concurrency_limit - len(self._active)
        if available <= 0:
            return 0, 0
        dispatchable = get_dispatchable_nodes(self._graph)
        dispatched = 0
        failed = 0
        for node_id in dispatchable[:available]:
            node = self._graph.get_node(node_id)
            desc = _summarize_description(node.description) if node else str(node_id)
            try:
                result = self._executor.dispatch(node_id, config)
            except Exception as exc:
                _logger.exception(
                    "Dispatch failed for node %d (%s): %s: %s",
                    node_id,
                    desc,
                    type(exc).__name__,
                    exc,
                )
                self._executor.fail(node_id)
                self._logs.append(f"[{ts()}] ✗ dispatch node {node_id}: {type(exc).__name__}")
                failed += 1
                if self._strict:
                    self._failure_triggered = True
                    break
                continue
            self._active[result.run_id] = node_id
            self._dispatched_at[result.run_id] = time.monotonic()
            self._logs.append(f"[{ts()}] → node {node_id}: {desc}")
            live.console.print(f"[cyan]→[/cyan] [{node_id}] {desc}")
            _logger.info("node_dispatched node_id=%d run_id=%s", node_id, result.run_id)
            dispatched += 1
        return dispatched, failed
