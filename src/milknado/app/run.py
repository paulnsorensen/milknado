"""Application-layer policy and adapter wiring for the run and dispatch surfaces.

The CLI and MCP entry points parse input and call this module. This module owns
dispatch policy, execution configuration, worker validation, and adapter wiring.
"""

from __future__ import annotations

import logging
import shlex
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from queue import Queue
from threading import Event, Lock, Thread
from typing import TYPE_CHECKING, cast, final

from typing_extensions import override

from milknado.adapters import ProcessAdapter, TmuxAdapter
from milknado.domains.common import (
    GitOperationError,
    GitPort,
    MikadoNode,
    MilknadoConfig,
    NodeKind,
    NodeStatus,
    WorktreeMode,
    resolve_flavor_profile,
)

if TYPE_CHECKING:
    from milknado.domains.dispatch import IsolateContext
    from milknado.domains.execution import ExecutionConfig, RunLoop, RunLoopResult, RunLoopState
    from milknado.domains.graph import MikadoGraph

_logger = logging.getLogger(__name__)


class ExecutionRunStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass(frozen=True, slots=True)
class RunActionAvailability:
    cancel_reason: str | None = None
    guidance_reason: str | None = None
    force_stop_reason: str | None = None

    @property
    def can_cancel(self) -> bool:
        return self.cancel_reason is None

    @property
    def can_queue_guidance(self) -> bool:
        return self.guidance_reason is None

    @property
    def can_force_stop(self) -> bool:
        return self.force_stop_reason is None


@dataclass(frozen=True, slots=True)
class ActiveRunSnapshot:
    """Application-owned immutable read model for one active agent run."""

    run_id: str
    node_id: int
    description: str
    status: ExecutionRunStatus
    progress: str | None
    stop_requested: bool
    actions: RunActionAvailability
    output: tuple[str, ...]
    pending_guidance: tuple[str, ...] | None
    elapsed_seconds: float
    progress_pct: float | None
    eta_seconds: float | None
    attempt: int | None
    max_attempts: int | None
    stalled: bool


@dataclass(frozen=True, slots=True)
class TerminalRunSnapshot:
    """Bounded terminal record retained after an active run exits."""

    run_id: str
    node_id: int
    description: str
    status: ExecutionRunStatus
    output: tuple[str, ...]
    pending_guidance: tuple[str, ...] | None
    duration_seconds: float


@dataclass(frozen=True, slots=True)
class ExecutionSnapshot:
    """Immutable presentation state published by the application controller."""

    goal: str
    active_runs: tuple[ActiveRunSnapshot, ...]
    terminal_runs: tuple[TerminalRunSnapshot, ...]
    completed: int
    failed: int
    stopped: int
    available: int
    event_lines: tuple[str, ...]
    listener_errors: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ProtectedBranchRefusal(RuntimeError):
    """Structured refusal raised before dispatch side effects begin."""

    branch: str
    reason: str

    def __post_init__(self) -> None:
        RuntimeError.__init__(self, self.__str__())

    @override
    def __str__(self) -> str:
        return f"{self.reason} branch {self.branch!r}"


def ensure_dispatch_allowed(cfg: MilknadoConfig, branch: str, allow_protected: bool) -> None:
    """Raise a typed refusal for detached or protected branches."""
    if branch in ("", "HEAD"):
        raise ProtectedBranchRefusal(branch=branch, reason="detached")
    if not allow_protected and branch in cfg.protected_branches:
        raise ProtectedBranchRefusal(branch=branch, reason="protected")


def build_exec_config(config: MilknadoConfig, project_root: Path) -> ExecutionConfig:
    from milknado.domains.execution import ExecutionConfig

    return ExecutionConfig(
        execution_agent=config.execution_agent,
        quality_gates=config.quality_gates,
        worktree_pattern=config.worktree_pattern,
        project_root=project_root,
        brief_prepend=config.worker_brief_prepend,
    )


def resolve_feature_branch(project_root: Path) -> str:
    """Return the checkout's current branch name (adapter wiring for the CLI)."""
    from milknado.adapters import GitAdapter

    return GitAdapter(project_root).current_branch()


@dataclass
class _ControlRequest:
    operation: str
    args: tuple[object, ...]
    done: Event = field(default_factory=Event)
    result: object | None = None
    error: BaseException | None = None


@final
class ExecutionController:
    """UI-neutral application boundary for one execution run."""

    def __init__(
        self,
        loop: RunLoop,
        execution_config: ExecutionConfig,
        concurrency_limit: int,
        config: MilknadoConfig,
    ) -> None:
        self._loop = loop
        self._execution_config = execution_config
        self._concurrency_limit = concurrency_limit
        self._config = config
        self._controls: Queue[_ControlRequest] = Queue()
        self._state_lock = Lock()
        self._listeners: set[Callable[[ExecutionSnapshot], None]] = set()
        self._listener_errors: dict[int, str] = {}
        self._snapshot = self._project_snapshot(loop.state())
        self._running = False
        loop.set_state_listener(self._receive_state)

    def run(
        self,
        *,
        feature_branch: str,
        strict: bool = False,
        spec_text: str | None = None,
        spec_path: Path | None = None,
        allow_protected: bool = False,
    ) -> RunLoopResult:
        ensure_dispatch_allowed(self._config, feature_branch, allow_protected)

        with self._state_lock:
            if self._running:
                raise RuntimeError("execution controller is already running")
            self._running = True
        outcomes: Queue[RunLoopResult | BaseException] = Queue(maxsize=1)

        def execute() -> None:
            try:
                outcomes.put(
                    self._loop.run(
                        config=self._execution_config,
                        feature_branch=feature_branch,
                        concurrency_limit=self._concurrency_limit,
                        strict=strict,
                        spec_text=spec_text,
                        spec_path=spec_path,
                        process_controls=self._drain_controls,
                        interactive=False,
                    )
                )
            except BaseException as exc:
                outcomes.put(exc)
            finally:
                with self._state_lock:
                    self._running = False
                    self._reject_pending_controls()

        worker = Thread(target=execute, name="milknado-execution", daemon=True)
        worker.start()
        outcome = outcomes.get()
        worker.join()
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def snapshot(self) -> ExecutionSnapshot:
        """Return the controller's current immutable presentation snapshot."""
        with self._state_lock:
            return self._snapshot

    def subscribe(self, listener: Callable[[ExecutionSnapshot], None]) -> Callable[[], None]:
        """Subscribe to future snapshots and replay the current state once."""
        with self._state_lock:
            self._listeners.add(listener)
            snapshot = self._snapshot
        try:
            listener(snapshot)
        except BaseException:
            with self._state_lock:
                self._listeners.discard(listener)
            raise

        def unsubscribe() -> None:
            with self._state_lock:
                self._listeners.discard(listener)
                _ = self._listener_errors.pop(id(listener), None)
                errors = tuple(self._listener_errors.values())
                self._snapshot = replace(self._snapshot, listener_errors=errors)

        return unsubscribe

    def _receive_state(self, state: RunLoopState) -> None:
        snapshot = self._project_snapshot(state)
        with self._state_lock:
            snapshot = replace(snapshot, listener_errors=tuple(self._listener_errors.values()))
            self._snapshot = snapshot
            listeners = tuple(self._listeners)

        failures: list[tuple[int, str]] = []
        for listener in listeners:
            try:
                listener(snapshot)
            except Exception as exc:
                listener_name = getattr(listener, "__qualname__", type(listener).__qualname__)
                error = f"{listener_name}: {exc}"
                failures.append((id(listener), error))
                _logger.exception("execution snapshot listener failed: %s", error)
        if not failures:
            return

        with self._state_lock:
            self._listener_errors.update(failures)
            errors = tuple(self._listener_errors.values())
            self._snapshot = visible = replace(self._snapshot, listener_errors=errors)

        failed_ids = {listener_id for listener_id, _ in failures}
        for listener in listeners:
            if id(listener) in failed_ids:
                continue
            try:
                listener(visible)
            except Exception:
                listener_name = getattr(listener, "__qualname__", type(listener).__qualname__)
                _logger.exception(
                    "execution snapshot listener failed while publishing error: %s",
                    listener_name,
                )

    @staticmethod
    def _project_snapshot(state: RunLoopState) -> ExecutionSnapshot:
        active_runs = tuple(
            ActiveRunSnapshot(
                run_id=run.run_id,
                node_id=run.node_id,
                description=run.description,
                status=ExecutionRunStatus(run.status.value),
                progress=run.progress,
                stop_requested=run.stop_requested,
                actions=RunActionAvailability(
                    cancel_reason=run.actions.cancel_reason,
                    guidance_reason=run.actions.guidance_reason,
                    force_stop_reason=run.actions.force_stop_reason,
                ),
                output=run.output,
                pending_guidance=run.pending_guidance,
                elapsed_seconds=run.elapsed_seconds,
                progress_pct=run.progress_pct,
                eta_seconds=run.eta_seconds,
                attempt=run.attempt,
                max_attempts=run.max_attempts,
                stalled=run.stalled,
            )
            for run in state.active_runs
        )
        terminal_runs = tuple(
            TerminalRunSnapshot(
                run_id=run.run_id,
                node_id=run.node_id,
                description=run.description,
                status=ExecutionRunStatus(run.status.value),
                output=run.output,
                pending_guidance=run.pending_guidance,
                duration_seconds=run.duration_seconds,
            )
            for run in state.terminal_runs
        )
        return ExecutionSnapshot(
            goal=state.goal,
            active_runs=active_runs,
            terminal_runs=terminal_runs,
            completed=state.completed,
            failed=state.failed,
            stopped=state.stopped,
            available=state.available,
            event_lines=state.event_lines,
        )

    def queue_guidance(self, run_id: str, text: str) -> bool:
        return bool(self._control("queue_guidance", run_id, text))

    def cancel(self, run_id: str) -> None:
        _ = self._control("cancel", run_id)

    def force_stop(self, run_id: str, timeout: float = 10.0) -> bool:
        return bool(self._control("force_stop", run_id, timeout))

    def stop_scheduling(self) -> None:
        """Stop admitting new work and request terminal completion of active runs."""
        _ = self._control("stop_scheduling")

    def _control(self, operation: str, *args: object) -> object:
        request: _ControlRequest | None = None
        with self._state_lock:
            if self._running:
                if operation == "stop_scheduling":
                    self._loop.admit_stop_scheduling()
                request = _ControlRequest(operation=operation, args=args)
                self._controls.put(request)
        if request is None:
            operation_fn = cast(Callable[..., object], getattr(self._loop, operation))
            return operation_fn(*args)
        _ = request.done.wait()
        if request.error is not None:
            raise request.error
        return request.result

    def _drain_controls(self) -> None:
        while not self._controls.empty():
            request = self._controls.get_nowait()
            try:
                request.result = getattr(self._loop, request.operation)(*request.args)
            except BaseException as exc:
                request.error = exc
            finally:
                request.done.set()

    def _reject_pending_controls(self) -> None:
        while not self._controls.empty():
            request = self._controls.get_nowait()
            request.error = RuntimeError("execution has finished")
            request.done.set()


def build_execution_controller(
    graph: MikadoGraph,
    config: MilknadoConfig,
    project_root: Path,
) -> ExecutionController:
    """Compose the sole UI-facing execution API from application dependencies."""
    from milknado.adapters import CrgAdapter, GitAdapter, LoopAdapter
    from milknado.domains.dispatch import reconcile_orphaned_runs
    from milknado.domains.execution import Executor, RunLoop

    _ = reconcile_orphaned_runs(graph)
    ralph = LoopAdapter()
    executor = Executor(
        graph=graph,
        git=GitAdapter(project_root),
        ralph=ralph,
        crg=CrgAdapter(project_root),
    )
    loop = RunLoop(executor=executor, graph=graph, ralph=ralph, config=config)
    return ExecutionController(
        loop,
        execution_config=build_exec_config(config, project_root),
        concurrency_limit=config.concurrency_limit,
        config=config,
    )


def run_execution_loop(
    graph: MikadoGraph,
    config: MilknadoConfig,
    project_root: Path,
    feature_branch: str,
    strict: bool,
    allow_protected: bool = False,
) -> RunLoopResult:
    """Wire the executor + run loop and drive it to completion."""
    ensure_dispatch_allowed(config, feature_branch, allow_protected)
    from milknado.adapters import CrgAdapter, GitAdapter, LoopAdapter
    from milknado.domains.dispatch import reconcile_orphaned_runs
    from milknado.domains.execution import Executor, RunLoop

    _ = reconcile_orphaned_runs(graph)
    git = GitAdapter(project_root)
    ralph = LoopAdapter()
    crg = CrgAdapter(project_root)
    executor = Executor(graph=graph, git=git, ralph=ralph, crg=crg)
    loop = RunLoop(executor=executor, graph=graph, ralph=ralph, config=config)
    return loop.run(
        config=build_exec_config(config, project_root),
        feature_branch=feature_branch,
        concurrency_limit=config.concurrency_limit,
        strict=strict,
        interactive=False,
    )


def resolve_run_attach_target(graph: MikadoGraph, project_root: Path, run_id: str) -> str:
    """Resolve the tmux target for a durable run identifier."""

    from milknado.domains.dispatch import resolve_attach_target

    return resolve_attach_target(graph, TmuxAdapter(project_root), run_id)


def validate_worker_cmd(worker_cmd: str | None) -> None:
    """Validate an explicit worker command before dispatch.

    The eager MCP check is repeated when the environment fallback or built-in
    default is resolved in ``runner.resolve_worker_cmd``.
    """

    from milknado.domains.dispatch import validate_worker_argv

    if not worker_cmd or not worker_cmd.strip():
        return
    validate_worker_argv(shlex.split(worker_cmd))


def prepare_isolation(
    graph: MikadoGraph,
    git: GitPort,
    root: Path,
    node: MikadoNode,
    run_id: str,
    worktree: WorktreeMode,
    merge_back: bool,
    worktree_pattern: str,
) -> tuple[Path, IsolateContext | None]:
    from milknado.domains.dispatch import setup_isolated_worktree

    if worktree != WorktreeMode.ISOLATE:
        return root, None
    context = setup_isolated_worktree(graph, git, root, node, run_id, worktree_pattern)
    return context.worktree_path, context if merge_back else None


@dataclass(frozen=True)
class InlineRunRequest:
    node_id: int
    worker_cmd: str | None
    timeout_seconds: int
    worktree: WorktreeMode
    merge_back: bool


def _require_task_node(graph: MikadoGraph, node_id: int) -> MikadoNode:
    node = graph.get_node(node_id)
    if node is None:
        raise ValueError(f"node {node_id} not found")
    if node.kind != NodeKind.TASK:
        raise ValueError(
            f"node {node_id} has kind={node.kind.value}; only task nodes can be dispatched"
        )
    return node


def _git_for_inline_dispatch(
    cfg: MilknadoConfig,
    root: Path,
    request: InlineRunRequest,
    allow_protected: bool,
) -> GitPort:
    from milknado.adapters import GitAdapter

    git = GitAdapter(root)
    try:
        branch = git.current_branch()
    except GitOperationError as exc:
        non_repository = "not a git repository" in exc.detail.lower()
        if request.worktree is not WorktreeMode.THIS_BRANCH or not non_repository:
            raise
    else:
        ensure_dispatch_allowed(cfg, branch, allow_protected)
    return git


def run_inline(
    graph: MikadoGraph,
    cfg: MilknadoConfig,
    root: Path,
    request: InlineRunRequest,
    *,
    allow_protected: bool = False,
) -> dict[str, object]:
    """Dispatch one task to a blocking worker and return its run state."""
    git = _git_for_inline_dispatch(cfg, root, request, allow_protected)
    from milknado.domains.dispatch import SyncDispatchRequest, dispatch_node_sync

    _logger.info(
        "milknado_run_inline: node=%d timeout=%ds worktree=%s merge_back=%s",
        request.node_id,
        request.timeout_seconds,
        request.worktree.value,
        request.merge_back,
    )
    node = _require_task_node(graph, request.node_id)
    profile = resolve_flavor_profile(cfg, node.flavor)
    state = dispatch_node_sync(
        graph,
        git,
        SyncDispatchRequest(
            node_id=request.node_id,
            project_root=root,
            worker_cmd=request.worker_cmd,
            timeout_seconds=request.timeout_seconds,
            default_cmd=profile.execution_agent,
            process=ProcessAdapter(),
            brief_prepend=profile.brief_prepend,
            worktree_mode=request.worktree,
            merge_back=request.merge_back,
            worktree_pattern=cfg.worktree_pattern,
        ),
    )
    return state


def run_inline_start(
    graph: MikadoGraph,
    cfg: MilknadoConfig,
    root: Path,
    request: InlineRunRequest,
    use_tmux: bool,
    *,
    allow_protected: bool = False,
) -> dict[str, object]:
    """Start an asynchronous worker and return its initial run state."""
    git = _git_for_inline_dispatch(cfg, root, request, allow_protected)
    from milknado.app.project import open_graph
    from milknado.domains.dispatch import (
        AsyncRunRequest,
        GraphSessionPort,
        ensure_tmux_ready,
        make_run_id,
        now_iso,
        reclaim_stale_node,
        render_brief,
        start_headless_async,
    )

    class _GraphSessions(GraphSessionPort):
        @override
        def open_graph(self, project_root: Path) -> tuple[MikadoGraph, MilknadoConfig]:
            return open_graph(project_root)

    tmux: TmuxAdapter | None = None
    if use_tmux:
        # Explicit tmux requests fail closed before any claim.
        tmux = TmuxAdapter(root)
        ensure_tmux_ready(tmux)

    node = _require_task_node(graph, request.node_id)
    run_id = make_run_id(request.node_id)
    if node.status == NodeStatus.RUNNING:
        reclaim_stale_node(graph, request.node_id, fence_run_id=node.run_id)
    profile = resolve_flavor_profile(cfg, node.flavor)
    brief = render_brief(
        graph,
        request.node_id,
        prepend=profile.brief_prepend,
        project_root=root,
    )
    graph.claim_node_for_dispatch(request.node_id, run_id, now=now_iso())
    try:
        worker_cwd, merge_ctx = prepare_isolation(
            graph,
            git,
            root,
            node,
            run_id,
            request.worktree,
            request.merge_back,
            cfg.worktree_pattern,
        )
        ref = start_headless_async(
            AsyncRunRequest(
                project_root=root,
                node_id=request.node_id,
                brief=brief,
                worker_cmd=request.worker_cmd,
                timeout_seconds=request.timeout_seconds,
                run_id=run_id,
                default_cmd=profile.execution_agent,
                cwd=worker_cwd,
                merge_ctx=merge_ctx,
            ),
            _GraphSessions(),
            git,
            ProcessAdapter(),
            tmux,
        )
    except Exception:
        # Failed startup releases the claim with a fenced terminal write.
        if not graph.mark_terminal(request.node_id, run_id, NodeStatus.FAILED):
            raise RuntimeError(
                f"startup terminal node write lost its fence for node {request.node_id}"
            )
        raise
    _logger.info(
        "milknado_run_inline_start: node=%d run_id=%s timeout=%ds",
        request.node_id,
        ref.run_id,
        request.timeout_seconds,
    )
    return {
        "run_id": ref.run_id,
        "node_id": request.node_id,
        "status": "running",
        "log_path": str(ref.log_path),
    }
