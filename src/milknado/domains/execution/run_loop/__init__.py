from __future__ import annotations

import dataclasses
import json
import logging
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING

from milknado.domains.common import ProgressEvent, TerminalRunOutcome, resolve_flavor_profile
from milknado.domains.common.errors import CompletionTimeout
from milknado.domains.common.types import NodeStatus
from milknado.domains.execution.executor import RebaseConflict, get_dispatchable_nodes
from milknado.domains.execution.run_loop._completion import handle_completion
from milknado.domains.execution.run_loop._logging import configure_run_logging, ts
from milknado.domains.execution.run_loop._result import RunLoopResult, VerifyOutcome
from milknado.domains.execution.run_loop.display import (
    ActiveRunSnapshot,
    ExecutionSnapshot,
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
from milknado.loop import RunStatus


__all__ = ["ActiveRunSnapshot", "ExecutionSnapshot", "RunLoop", "RunLoopResult"]

if TYPE_CHECKING:
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel

    from milknado.domains.common.config import MilknadoConfig
    from milknado.domains.common.protocols import LoopPort
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
        ralph: LoopPort,
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
        self._stopped_nodes: set[int] = set()
        self._completed = 0
        self._failed = 0
        self._listeners: set[Callable[[ExecutionSnapshot], None]] = set()
        self._listeners_lock = Lock()
        self._process_controls: Callable[[], None] | None = None
        self._completion_wait_started = 0.0
        self._scheduling_stopped = False
        self._scheduling_lock = Lock()




    def subscribe(self, listener: Callable[[ExecutionSnapshot], None]) -> Callable[[], None]:
        """Register a UI-neutral observer and return its idempotent removal callback."""
        with self._listeners_lock:
            self._listeners.add(listener)

        def unsubscribe() -> None:
            with self._listeners_lock:
                self._listeners.discard(listener)

        return unsubscribe

    def snapshot(self) -> ExecutionSnapshot:
        """Build a bounded immutable view without exposing loop-owned collections."""
        with self._graph.read_locked():
            root = self._graph.get_root()
            active_runs = tuple(
                self._active_snapshot(run_id, node_id)
                for run_id, node_id in sorted(self._active.items(), key=lambda item: item[1])
            )
            available = sum(
                node_id not in self._stopped_nodes
                for node_id in get_dispatchable_nodes(self._graph)
            )
        return ExecutionSnapshot(
            goal=root.description if root else "",
            active_runs=active_runs,
            completed=self._completed,
            failed=self._failed,
            available=available,
            event_lines=tuple(self._logs),
        )

    def _active_snapshot(self, run_id: str, node_id: int) -> ActiveRunSnapshot:
        run = self._ralph.get_run(run_id)
        state = run.state if run is not None else None
        status = getattr(state, "status", RunStatus.RUNNING)
        progress_event = self._progress_by_run.get(run_id)
        progress = None
        if progress_event is not None:
            progress = progress_event.message or f"{progress_event.work}/{progress_event.total}"
        reasons: list[tuple[str, str]] = []
        stop_requested = bool(getattr(state, "stop_requested", False))
        terminal_reasons = {
            RunStatus.COMPLETED: "run has completed",
            RunStatus.FAILED: "run has failed",
            RunStatus.STOPPED: "run has stopped",
        }
        terminal_reason = terminal_reasons.get(status)
        force_stop_available = not bool(getattr(state, "force_stop_requested", False))
        if terminal_reason is not None:
            force_stop_available = False
            reasons.extend(
                (
                    ("cancel", terminal_reason),
                    ("guidance", terminal_reason),
                    ("force_stop", terminal_reason),
                )
            )
        elif stop_requested:
            reasons.append(("cancel", "stop already requested"))
            reasons.append(("guidance", "run is stopping"))
        if not force_stop_available and terminal_reason is None:
            reasons.append(("force_stop", "force stop already requested"))
        node = self._graph.get_node(node_id)
        return ActiveRunSnapshot(
            run_id=run_id,
            node_id=node_id,
            description=_summarize_description(node.description) if node else str(node_id),
            status=status,
            progress=progress,
            stop_requested=stop_requested,
            force_stop_available=force_stop_available,
            unavailable_action_reasons=tuple(reasons),
            output=tuple(self._ralph.get_run_output_tail(run_id, 30)),
            pending_guidance=self._ralph.get_run_guidance(run_id),
        )

    def _publish_snapshot(self) -> None:
        snapshot = self.snapshot()
        with self._listeners_lock:
            listeners = tuple(self._listeners)
        for listener in listeners:
            try:
                listener(snapshot)
            except Exception:
                _logger.exception("execution snapshot listener failed")

    def queue_guidance(self, run_id: str, text: str) -> bool:
        accepted = self._ralph.queue_guidance(run_id, text)
        self._publish_snapshot()
        return accepted

    def cancel(self, run_id: str) -> None:
        self._ralph.request_stop_run(run_id)
        self._publish_snapshot()

    def force_stop(self, run_id: str, timeout: float = 10.0) -> bool:
        stopped = self._ralph.force_stop_run(run_id, timeout)
        self._publish_snapshot()
        return stopped

    def admit_stop_scheduling(self) -> None:
        """Atomically close scheduling admission before a control is queued."""
        with self._scheduling_lock:
            self._scheduling_stopped = True

    def stop_scheduling(self) -> None:
        """Prevent redispatch and ask every currently active run to stop."""
        self.admit_stop_scheduling()
        for run_id in self._active:
            self._ralph.request_stop_run(run_id)
        self._publish_snapshot()


    def run(
        self,
        config: ExecutionConfig,
        feature_branch: str,
        concurrency_limit: int = 4,
        strict: bool = False,
        spec_text: str | None = None,
        spec_path: Path | None = None,
        process_controls: Callable[[], None] | None = None,
    ) -> RunLoopResult:
        from rich.console import Console

        self._strict = strict
        self._exec_config = config
        self._process_controls = process_controls
        self._stopped_nodes.clear()
        self._completed = 0
        self._failed = 0
        if self._process_controls is not None:
            self._process_controls()
        self._publish_snapshot()
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
        verify_outcome = self._verify_if_scheduling_open(spec_text, spec_path, config)
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
                drain_input(self._input, self._active)
                if self._process_controls is not None:
                    self._process_controls()
                d, f = self._dispatch_if_scheduling_open(config, concurrency_limit, live)
                dispatched += d
                failed += f
                self._failed = failed
                self._publish_snapshot()
                while self._active:
                    d, c, f, cs, timed_out = self._poll_and_complete(
                        config, feature_branch, concurrency_limit, timeout, live
                    )
                    dispatched += d
                    completed += c
                    failed += f
                    conflicts.extend(cs)
                    self._completed = completed
                    self._failed = failed
                    self._publish_snapshot()
                    if timed_out:
                        break
        except KeyboardInterrupt:
            interrupted = True
            _logger.warning("Run interrupted by user (KeyboardInterrupt)")
            raise
        finally:
            stop_input_thread(self._input)
            self._publish_snapshot()
        return dispatched, completed, failed, conflicts, interrupted

    def _handle_completion_timeout(self, ct: CompletionTimeout) -> int:
        _logger.warning(
            "Completion timeout after %.1fs; active runs: %s",
            ct.waited_seconds,
            sorted(ct.active_run_ids),
        )
        newly_failed = 0
        for timed_out_id in list(self._active):
            nid = self._active[timed_out_id]
            if not self._ralph.stop_run(timed_out_id, timeout=10.0):
                _logger.error(
                    "worker did not exit after stop; preserving ownership node_id=%d run_id=%s",
                    nid,
                    timed_out_id,
                )
                continue
            self._active.pop(timed_out_id)
            self._executor.fail(nid)
            self._logs.append(f"[{ts()}] ⏱ node {nid} timeout")
            newly_failed += 1
        if self._strict:
            self._failure_triggered = True
        return newly_failed

    def _poll_and_complete(
        self,
        config: ExecutionConfig,
        feature_branch: str,
        concurrency_limit: int,
        timeout: float,
        live: Live,
    ) -> tuple[int, int, int, list[RebaseConflict], bool]:
        """One poll-and-handle iteration.

        Returns (dispatched, completed, failed, conflicts, timed_out).
        """
        if self._process_controls is not None:
            self._process_controls()
        wait_timeout = min(timeout, 0.1) if self._process_controls is not None else timeout
        try:
            run_id, outcome = self._ralph.wait_for_next_completion(
                set(self._active.keys()), timeout=wait_timeout
            )
        except CompletionTimeout as ct:
            if (
                self._process_controls is not None
                and time.monotonic() - self._completion_wait_started < timeout
            ):
                self._process_controls()
                return 0, 0, 0, [], False
            return 0, 0, self._handle_completion_timeout(ct), [], True
        if isinstance(outcome, ProgressEvent):
            self._progress_by_run[outcome.run_id] = outcome
            if self._process_controls is not None:
                self._process_controls()
            self._publish_snapshot()
            self._render_live_frame(live)
            return 0, 0, 0, [], False
        self._completion_wait_started = time.monotonic()
        self._absorb_progress()
        drain_input(self._input, self._active)
        c, f, cs = handle_completion(self, run_id, outcome, feature_branch, live)
        completed = c
        failed = f
        conflicts: list[RebaseConflict] = list(cs)
        dispatched = 0
        d, f2 = self._dispatch_if_scheduling_open(config, concurrency_limit, live)
        dispatched += d
        failed += f2
        self._render_live_frame(live)
        return dispatched, completed, failed, conflicts, False

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

    def _dispatch_if_scheduling_open(
        self,
        config: ExecutionConfig,
        concurrency_limit: int,
        live: Live,
    ) -> tuple[int, int]:
        with self._scheduling_lock:
            if self._scheduling_stopped:
                return 0, 0
            return self._dispatch_batch(config, concurrency_limit, live)

    def _verify_if_scheduling_open(
        self,
        spec_text: str | None,
        spec_path: Path | None,
        config: ExecutionConfig,
    ) -> VerifyOutcome | None:
        with self._scheduling_lock:
            if self._scheduling_stopped:
                return None
            return self._maybe_verify_spec(spec_text, spec_path, config)

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
        dispatchable = [
            node_id for node_id in get_dispatchable_nodes(self._graph) if node_id not in self._stopped_nodes
        ]
        dispatched = 0
        failed = 0
        for node_id in dispatchable[:available]:
            node = self._graph.get_node(node_id)
            desc = _summarize_description(node.description) if node else str(node_id)
            node_config = config
            if self._milknado_config is not None and node is not None:
                profile = resolve_flavor_profile(self._milknado_config, node.flavor)
                node_config = dataclasses.replace(
                    config,
                    execution_agent=profile.execution_agent,
                    quality_gates=profile.quality_gates,
                    review=profile.review,
                    review_agent=profile.review_agent,
                    review_max_rounds=profile.review_max_rounds,
                    on_reject=profile.on_reject,
                    session_mode=profile.session_mode,
                )
            if node_config.quality_gates is None:
                from milknado.domains.execution.completion import NO_GATES_CONFIGURED_MESSAGE

                _logger.error(
                    "preflight: node %d (%s): %s", node_id, desc, NO_GATES_CONFIGURED_MESSAGE
                )
                live.console.print(
                    f"[red]✗[/red] [{node_id}] {desc}: {NO_GATES_CONFIGURED_MESSAGE}"
                )
                self._executor.fail(node_id)
                self._logs.append(f"[{ts()}] ✗ node {node_id}: no quality_gates configured")
                failed += 1
                if self._strict:
                    self._failure_triggered = True
                    break
                continue
            try:
                result = self._executor.dispatch(node_id, node_config)
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
