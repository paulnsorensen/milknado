from __future__ import annotations

import dataclasses
import logging
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING

from typing_extensions import override

from milknado.domains.common import (
    ProgressEvent,
    SessionInput,
    TerminalRunOutcome,
    resolve_flavor_profile,
)
from milknado.domains.common.errors import CompletionTimeout
from milknado.domains.common.types import NodeStatus
from milknado.domains.execution._models import (
    NodeClaimRejected,
    PreservedWorkerRun,
    RebaseConflict,
)
from milknado.domains.execution.executor import get_dispatchable_nodes, get_execution_overview
from milknado.domains.execution.run_loop._idle import IdleGraphState, settle_once
from milknado.domains.execution.run_loop._logging import (
    configure_run_logging,
    emit_final_telemetry,
    ts,
)
from milknado.domains.execution.run_loop._node import IDLE_RESCAN_SECONDS, NodeDriverMixin
from milknado.domains.execution.run_loop._projection import project_state
from milknado.domains.execution.run_loop._result import (
    NodeLoopOutcome,
    RunLoopResult,
    VerifyOutcome,
)
from milknado.domains.execution.run_loop._stop import StopControlMixin
from milknado.domains.execution.run_loop.input import (
    InputState,
    drain_input,
    start_input_thread,
    stop_input_thread,
)
from milknado.domains.execution.run_loop.state import (
    ActiveRunFacts,
    ProjectionFacts,
    RunLoopState,
    TerminalRunState,
    summarize_description,
)
from milknado.domains.graph import ConcurrencyLimitReached
from milknado.loop import RunStatus

__all__ = ["RunLoop", "RunLoopResult", "TerminalRunOutcome"]

if TYPE_CHECKING:
    from milknado.domains.common.config import MilknadoConfig
    from milknado.domains.common.protocols import LoopPort
    from milknado.domains.execution.executor import ExecutionConfig, Executor
    from milknado.domains.graph import MikadoGraph
    from milknado.domains.planning import Planner

_logger = logging.getLogger("milknado")
_ETA_SAMPLE_SIZE_DEFAULT = 10
_STALL_THRESHOLD_DEFAULT = 300


class RunLoop(NodeDriverMixin, StopControlMixin):
    def __init__(
        self,
        executor: Executor,
        graph: MikadoGraph,
        loop: LoopPort,
        config: MilknadoConfig | None = None,
        planner: Planner | None = None,
        shutdown_requested: Callable[[], bool] | None = None,
    ) -> None:
        self._executor: Executor = executor
        self._graph: MikadoGraph = graph
        self._loop: LoopPort = loop
        self._milknado_config: MilknadoConfig | None = config
        self._planner: Planner | None = planner
        self._active: dict[str, int] = {}
        self._logs: deque[str] = deque(maxlen=30)
        self._dispatched_at: dict[str, float] = {}
        self._attempts: dict[int, int] = {}
        self._failure_triggered: bool = False
        self._progress_by_run: dict[str, ProgressEvent] = {}
        self._strict: bool = False
        self._log_path: str | None = None
        eta_n = config.eta_sample_size if config else _ETA_SAMPLE_SIZE_DEFAULT
        self._completion_durations: deque[float] = deque(maxlen=eta_n)
        self._input: InputState = InputState()
        self._exec_config: ExecutionConfig | None = None
        self._stopped_nodes: set[int] = set()
        self._terminal_runs: deque[TerminalRunState] = deque(maxlen=20)
        self._completed: int = 0
        self._failed: int = 0
        self._stopped: int = 0
        self._state_listener: Callable[[RunLoopState], None] | None = None
        self._process_controls: Callable[[], None] | None = None
        self._await_owner_work: bool = False
        self._capacity_deferred: bool = False
        self._deferred_retry_at: float = 0.0
        self._idle_sleep: Callable[[float], None] = time.sleep
        self._completion_wait_started: float = 0.0
        self._scheduling_stopped: bool = False
        self._shutdown_requested: Callable[[], bool] | None = shutdown_requested
        self._scheduling_lock: Lock = Lock()
        self._logged_blocks: set[tuple[int, int, tuple[str, ...]]] = set()
        self._spec: tuple[str | None, Path | None] = (None, None)
        self._idle_settled_graph: IdleGraphState | None = None
        self._verify_outcome: VerifyOutcome | None = None

    def set_state_listener(self, listener: Callable[[RunLoopState], None]) -> None:
        """Set the application-layer state sink used during execution."""
        self._state_listener = listener

    def state(self) -> RunLoopState:
        """Build a bounded immutable execution state for the application layer."""
        active_items = tuple(sorted(self._active.items(), key=lambda item: item[1]))
        goal, descriptions, available = get_execution_overview(
            self._graph,
            [node_id for _, node_id in active_items],
            self._stopped_nodes,
        )
        cfg = self._milknado_config
        facts = ProjectionFacts(
            goal=goal,
            active_runs=tuple(
                self._active_facts(run_id, node_id, descriptions.get(node_id, str(node_id)))
                for run_id, node_id in active_items
            ),
            terminal_runs=tuple(self._terminal_runs),
            completed=self._completed,
            failed=self._failed,
            stopped=self._stopped,
            available=available,
            event_lines=tuple(self._logs),
            execution_agent=(
                self._exec_config.execution_agent if self._exec_config else "(unknown)"
            ),
            log_path=self._log_path,
            completion_durations=tuple(self._completion_durations),
            stall_threshold_seconds=(
                cfg.stall_threshold_seconds if cfg else _STALL_THRESHOLD_DEFAULT
            ),
            max_attempts=cfg.dispatch_max_retries + 1 if cfg else 3,
        )
        return project_state(facts, time.monotonic())

    def _active_facts(self, run_id: str, node_id: int, description: str) -> ActiveRunFacts:
        run = self._loop.get_run(run_id)
        state = run.state if run is not None else None
        progress = self._progress_by_run.get(run_id)
        return ActiveRunFacts(
            run_id=run_id,
            node_id=node_id,
            description=description,
            status=getattr(state, "status", RunStatus.RUNNING),
            stop_requested=bool(getattr(state, "stop_requested", False)),
            force_stop_requested=bool(getattr(state, "force_stop_requested", False)),
            output=tuple(self._loop.get_run_output_tail(run_id, 30)),
            pending_guidance=tuple(self._loop.get_run_guidance(run_id)),
            dispatched_at=self._dispatched_at.get(run_id),
            prior_attempts=self._attempts.get(node_id, 0),
            progress_message=progress.message if progress else "",
            progress_work=progress.work if progress else None,
            progress_total=progress.total if progress else None,
            session=self._loop.get_run_session(run_id),
        )

    @override
    def _publish_state(self) -> None:
        listener = self._state_listener
        if listener is None:
            return
        try:
            listener(self.state())
        except Exception:
            _logger.exception(
                "execution state listener failed listener=%s",
                getattr(listener, "__qualname__", type(listener).__qualname__),
            )

    def queue_guidance(self, run_id: str, text: str) -> bool:
        accepted = self._loop.queue_guidance(run_id, text)
        self._publish_state()
        return accepted

    def session_input(self, run_id: str, command: SessionInput) -> bool:
        if run_id not in self._active:
            return False
        accepted = self._loop.session_input(run_id, command)
        self._publish_state()
        return accepted

    def cancel(self, run_id: str) -> None:
        self._loop.request_stop_run(run_id)
        self._publish_state()

    def run(
        self,
        config: ExecutionConfig,
        feature_branch: str,
        concurrency_limit: int = 4,
        strict: bool = False,
        spec_text: str | None = None,
        spec_path: Path | None = None,
        process_controls: Callable[[], None] | None = None,
        *,
        interactive: bool = True,
        await_owner_work: bool = False,
    ) -> RunLoopResult:
        self._strict = strict
        self._exec_config = config
        self._process_controls = process_controls
        self._await_owner_work = await_owner_work
        self._stopped_nodes.clear()
        self._logged_blocks.clear()
        self._terminal_runs.clear()
        self._completed = 0
        self._failed = 0
        self._stopped = 0
        self._log_path = None
        self._spec = (spec_text, spec_path)
        self._idle_settled_graph, self._verify_outcome = None, None
        if self._process_controls is not None:
            self._process_controls()
        timeout = (
            self._milknado_config.completion_timeout_seconds if self._milknado_config else None
        )
        self._publish_state()

        with configure_run_logging(config.project_root) as log_path:
            self._log_path = str(log_path)
            _logger.info("Run started feature_branch=%s", feature_branch)
            if interactive:
                self._input.input_stop.clear()
                start_input_thread(self._input)
            dispatched, completed, failed, conflicts, interrupted = self._execute_run(
                config, feature_branch, concurrency_limit, timeout, interactive
            )
            root = self._graph.get_root()
            result = RunLoopResult(
                root_done=root is not None and root.status == NodeStatus.DONE,
                dispatched_total=dispatched,
                completed_total=completed,
                failed_total=failed,
                rebase_conflicts=tuple(conflicts),
                strict_exit=strict and self._failure_triggered,
            )
            emit_final_telemetry(result, self._stopped, interrupted)

        verify_outcome = self._verify_if_scheduling_open(spec_text, spec_path, config)
        root = self._graph.get_root()
        return dataclasses.replace(
            result,
            root_done=root is not None and root.status == NodeStatus.DONE,
            strict_exit=strict and self._failure_triggered,
            verify_outcome=verify_outcome,
        )

    def _execute_run(
        self,
        config: ExecutionConfig,
        feature_branch: str,
        concurrency_limit: int,
        timeout: float | None,
        interactive: bool,
    ) -> tuple[int, int, int, list[RebaseConflict], bool]:
        dispatched = 0
        conflicts: list[RebaseConflict] = []
        interrupted = False
        try:
            if interactive:
                drain_input(self._input, self._active)
            if self._process_controls is not None:
                self._process_controls()
            added, failed = self._dispatch_if_scheduling_open(config, concurrency_limit)
            dispatched += added
            self._failed += failed
            self._completion_wait_started = time.monotonic()
            self._publish_state()
            idle_added = 0
            while self._active or (
                idle_added := self._wait_for_owner_work(config, concurrency_limit)
            ):
                dispatched += idle_added
                idle_added = 0
                added, completed, failed, new_conflicts, timed_out = self._poll_and_complete(
                    config, feature_branch, concurrency_limit, timeout, interactive
                )
                dispatched += added
                self._completed += completed
                self._failed += failed
                conflicts.extend(new_conflicts)
                self._publish_state()
                if timed_out:
                    break
        except KeyboardInterrupt:
            interrupted = True
            _logger.warning("Run interrupted by user (KeyboardInterrupt)")
            raise
        finally:
            if interactive:
                stop_input_thread(self._input)
            self._publish_state()
        return dispatched, self._completed, self._failed, conflicts, interrupted

    def _wait_for_owner_work(self, config: ExecutionConfig, concurrency_limit: int) -> int:
        """Wait while capacity or owner work may make a task dispatchable."""
        owner_wait = self._await_owner_work and self._process_controls is not None
        if not self._capacity_deferred and not owner_wait:
            return 0
        while True:
            if self._process_controls is not None:
                self._process_controls()
            if self._strict and self._failure_triggered:
                return 0
            with self._scheduling_lock:
                if self._scheduling_stopped or (
                    self._shutdown_requested is not None and self._shutdown_requested()
                ):
                    return 0
            self._idle_settled_graph = settle_once(
                self._graph,
                self._idle_settled_graph,
                lambda: self._verify_if_scheduling_open(*self._spec, config),
            )
            root = self._graph.get_root()
            if root is not None and root.status == NodeStatus.DONE:
                return 0
            added, failed = self._dispatch_if_scheduling_open(config, concurrency_limit)
            self._failed += failed
            if added:
                self._completion_wait_started = time.monotonic()
                self._publish_state()
                return added
            if failed:
                self._publish_state()
            if not self._capacity_deferred and not owner_wait:
                return 0
            self._idle_sleep(IDLE_RESCAN_SECONDS)

    def _poll_and_complete(
        self,
        config: ExecutionConfig,
        feature_branch: str,
        concurrency_limit: int,
        timeout: float | None,
        interactive: bool,
    ) -> tuple[int, int, int, list[RebaseConflict], bool]:
        """One poll-and-handle iteration.

        Returns (dispatched, completed, failed, conflicts, timed_out).
        """
        if self._process_controls is not None:
            self._process_controls()
        wait_timeout = timeout
        if self._process_controls is not None:
            wait_timeout = 0.1 if timeout is None else min(timeout, 0.1)
        elif self._capacity_deferred:
            wait_timeout = (
                IDLE_RESCAN_SECONDS if timeout is None else min(timeout, IDLE_RESCAN_SECONDS)
            )
        try:
            run_id, outcome = self._loop.wait_for_next_completion(
                set(self._active.keys()), timeout=wait_timeout
            )
        except CompletionTimeout as ct:
            if self._process_controls is not None and (
                timeout is None or time.monotonic() - self._completion_wait_started < timeout
            ):
                self._process_controls()
                dispatched, failed = self._retry_deferred_if_due(config, concurrency_limit)
                if dispatched:
                    self._completion_wait_started = time.monotonic()
                return dispatched, 0, failed, [], False
            if self._capacity_deferred and (
                timeout is None or time.monotonic() - self._completion_wait_started < timeout
            ):
                dispatched, failed = self._dispatch_if_scheduling_open(config, concurrency_limit)
                if dispatched:
                    self._completion_wait_started = time.monotonic()
                return dispatched, 0, failed, [], False
            return 0, 0, self._handle_completion_timeout(ct), [], True
        if isinstance(outcome, ProgressEvent):
            self._progress_by_run[outcome.run_id] = outcome
            if self._process_controls is not None:
                self._process_controls()
            else:
                self._completion_wait_started = time.monotonic()
            self._publish_state()
            return 0, 0, 0, [], False
        self._completion_wait_started = time.monotonic()
        if interactive:
            drain_input(self._input, self._active)
        transition = self._handle_terminal(run_id, outcome, feature_branch)
        if transition is None:
            return 0, 0, 0, [], True
        completed, failed, conflicts = transition
        dispatched, dispatch_failures = self._dispatch_if_scheduling_open(
            config, concurrency_limit
        )
        return dispatched, completed, failed + dispatch_failures, list(conflicts), False

    def _retry_deferred_if_due(
        self, config: ExecutionConfig, concurrency_limit: int
    ) -> tuple[int, int]:
        """Retry a capacity-deferred dispatch at the idle-rescan cadence."""
        if not self._capacity_deferred:
            return 0, 0
        now = time.monotonic()
        if now < self._deferred_retry_at:
            return 0, 0
        self._deferred_retry_at = now + IDLE_RESCAN_SECONDS
        return self._dispatch_if_scheduling_open(config, concurrency_limit)

    def _dispatch_if_scheduling_open(
        self,
        config: ExecutionConfig,
        concurrency_limit: int,
    ) -> tuple[int, int]:
        with self._scheduling_lock:
            if self._scheduling_stopped or (
                self._shutdown_requested is not None and self._shutdown_requested()
            ):
                return 0, 0
            return self._dispatch_batch(config, concurrency_limit)

    def _verify_if_scheduling_open(
        self,
        spec_text: str | None,
        spec_path: Path | None,
        config: ExecutionConfig,
    ) -> VerifyOutcome | None:
        with self._scheduling_lock:
            if self._scheduling_stopped or (
                self._shutdown_requested is not None and self._shutdown_requested()
            ):
                return None
            if spec_text:
                outcome = self._maybe_verify_spec(spec_text, spec_path, config)
                self._verify_outcome = outcome or self._verify_outcome
                return self._verify_outcome
            self._complete_root_if_settled()
            return None

    def _complete_root_if_settled(self) -> None:
        """Complete the root structurally when no spec was supplied to verify against."""
        if self._failure_triggered or self._active:
            return
        root = self._graph.get_root()
        if root is None:
            return
        if not any(n.id != root.id for n in self._graph.get_all_nodes()):
            return  # undecomposed goal: nothing was done, so nothing is achieved
        _ = self._graph.complete_root()

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
        result = self._loop.verify_spec(spec_text, str(self._graph))
        outcome = VerifyOutcome(done=result.outcome == "done", goal_delta=result.goal_delta)
        if result.outcome == "done":
            self._graph.mark_running(root.id)
            self._graph.mark_done(root.id)
        elif result.outcome == "gaps" and self._planner and result.goal_delta:
            _ = self._planner.replan_with_delta(result.goal_delta, config.project_root, spec_path)
        return outcome

    def _dispatch_batch(
        self,
        config: ExecutionConfig,
        concurrency_limit: int,
    ) -> tuple[int, int]:
        self._capacity_deferred = False
        if self._strict and self._failure_triggered:
            return 0, 0
        available = concurrency_limit - len(self._active)
        if available <= 0:
            return 0, 0
        exclusions = self._graph.dispatch_exclusions()
        dispatchable = [
            node_id
            for node_id in get_dispatchable_nodes(self._graph, self._logged_blocks)
            if node_id not in self._stopped_nodes and node_id not in exclusions
        ]
        dispatched = 0
        failed = 0
        for node_id in dispatchable[:available]:
            if self._shutdown_requested is not None and self._shutdown_requested():
                break
            node = self._graph.get_node(node_id)
            desc = summarize_description(node.description) if node else str(node_id)
            node_config = config
            if self._milknado_config is not None and node is not None:
                profile = resolve_flavor_profile(self._milknado_config, node.flavor)
                node_config = dataclasses.replace(
                    config,
                    execution_agent=profile.execution_agent,
                    quality_gates=profile.quality_gates,
                    brief_prepend=profile.brief_prepend,
                    review=profile.review,
                    review_agent=profile.review_agent,
                    review_max_rounds=profile.review_max_rounds,
                    review_timeout_seconds=profile.review_timeout_seconds,
                    on_reject=profile.on_reject,
                    session_mode=profile.session_mode,
                    max_iterations=profile.max_iterations,
                    attempt_timeout_seconds=float(profile.attempt_timeout_seconds),
                    completion_timeout_seconds=(
                        profile.attempt_timeout_seconds * profile.max_iterations
                    ),
                )
            try:
                result = self._executor.dispatch(node_id, node_config)
            except ConcurrencyLimitReached as exc:
                self._capacity_deferred = True
                _logger.info(
                    "node_deferred node_id=%d running=%d limit=%d",
                    node_id,
                    exc.running,
                    exc.limit,
                )
                break
            except NodeClaimRejected:
                self._capacity_deferred = True
                break
            except PreservedWorkerRun as exc:
                _ = self.confirm_preserved_stop(
                    NodeLoopOutcome(
                        node_id,
                        False,
                        str(exc),
                        ownership_preserved=True,
                        worker_run_id=exc.run_id,
                        owner_run_id=exc.owner_run_id,
                    )
                )
                self._logs.append(f"[{ts()}] ✗ dispatch node {node_id}: {type(exc).__name__}")
                failed += 1
                if self._strict:
                    self._failure_triggered = True
                    break
                continue
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
            _logger.info("node_dispatched node_id=%d run_id=%s", node_id, result.run_id)
            dispatched += 1
        return dispatched, failed
