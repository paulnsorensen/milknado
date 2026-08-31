"""Application-layer policy and adapter wiring for the run / dispatch surfaces.

The CLI ``run``/``attach`` commands and the MCP ``milknado_run_inline*`` tools are
thin: they parse I/O and call the functions here, which own the policy (protected
branch refusal, execution-config assembly, worker-cmd validation, worktree
isolation) and construct the adapters (git, loop, crg, process, tmux). Entry
modules therefore hold no inline dispatch policy and build no adapters.
"""

from __future__ import annotations

import logging
import shlex
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from queue import Queue
from threading import Event, Lock, Thread
from typing import TYPE_CHECKING

from milknado.adapters import ProcessAdapter, TmuxAdapter
from milknado.domains.common import (
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


@dataclass(frozen=True)
class ProtectedBranchRefusal:
    """Structured refusal returned before graph/log setup can create side effects."""

    branch: str
    reason: str


def check_protected_branch(
    cfg: MilknadoConfig,
    branch: str,
    allow_protected: bool,
) -> ProtectedBranchRefusal | None:
    """Return a typed refusal for detached or protected branches.

    The caller owns presentation and exit-code mapping. This keeps the policy
    reusable by MCP and other non-CLI entry points without importing Typer or
    Rich.
    """
    if branch in ("", "HEAD"):
        return ProtectedBranchRefusal(branch=branch, reason="detached")
    if not allow_protected and branch in cfg.protected_branches:
        return ProtectedBranchRefusal(branch=branch, reason="protected")
    return None


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


class ExecutionController:
    """UI-neutral application boundary for one execution run."""

    def __init__(
        self,
        loop: RunLoop,
        execution_config: ExecutionConfig,
        concurrency_limit: int,
    ) -> None:
        self._loop = loop
        self._execution_config = execution_config
        self._concurrency_limit = concurrency_limit
        self._controls: Queue[_ControlRequest] = Queue()
        self._state_lock = Lock()
        self._listeners: set[Callable[[ExecutionSnapshot], None]] = set()
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
    ) -> RunLoopResult:
        with self._state_lock:
            if self._running:
                raise RuntimeError("execution controller is already running")
            self._running = True
        result: RunLoopResult | None = None
        error: BaseException | None = None

        def execute() -> None:
            nonlocal result, error
            try:
                result = self._loop.run(
                    config=self._execution_config,
                    feature_branch=feature_branch,
                    concurrency_limit=self._concurrency_limit,
                    strict=strict,
                    spec_text=spec_text,
                    spec_path=spec_path,
                    process_controls=self._drain_controls,
                    interactive=False,
                )
            except BaseException as exc:
                error = exc
            finally:
                with self._state_lock:
                    self._running = False
                    self._reject_pending_controls()

        worker = Thread(target=execute, name="milknado-execution", daemon=True)
        worker.start()
        worker.join()
        if error is not None:
            raise error
        if result is None:
            raise RuntimeError("execution loop returned no result")
        return result

    def snapshot(self) -> ExecutionSnapshot:
        """Return the controller's current immutable presentation snapshot."""
        with self._state_lock:
            return self._snapshot

    def subscribe(self, listener: Callable[[ExecutionSnapshot], None]) -> Callable[[], None]:
        """Subscribe to future snapshots and replay the current state once."""
        with self._state_lock:
            self._listeners.add(listener)
            snapshot = self._snapshot
        listener(snapshot)

        def unsubscribe() -> None:
            with self._state_lock:
                self._listeners.discard(listener)

        return unsubscribe

    def _receive_state(self, state: RunLoopState) -> None:
        snapshot = self._project_snapshot(state)
        with self._state_lock:
            self._snapshot = snapshot
            listeners = tuple(self._listeners)
        for listener in listeners:
            try:
                listener(snapshot)
            except Exception:
                _logger.exception(
                    "execution snapshot listener failed listener=%s",
                    getattr(listener, "__qualname__", type(listener).__qualname__),
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
        self._control("cancel", run_id)

    def force_stop(self, run_id: str, timeout: float = 10.0) -> bool:
        return bool(self._control("force_stop", run_id, timeout))

    def stop_scheduling(self) -> None:
        """Stop admitting new work and request terminal completion of active runs."""
        self._control("stop_scheduling")

    def _control(self, operation: str, *args: object) -> object:
        request: _ControlRequest | None = None
        with self._state_lock:
            if self._running:
                if operation == "stop_scheduling":
                    self._loop.admit_stop_scheduling()
                request = _ControlRequest(operation=operation, args=args)
                self._controls.put(request)
        if request is None:
            return getattr(self._loop, operation)(*args)
        request.done.wait()
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

    reconcile_orphaned_runs(graph)
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
    )


def run_execution_loop(
    graph: MikadoGraph,
    config: MilknadoConfig,
    project_root: Path,
    feature_branch: str,
    strict: bool,
) -> RunLoopResult:
    """Wire the executor + run loop and drive it to completion."""
    from milknado.adapters import CrgAdapter, GitAdapter, LoopAdapter
    from milknado.domains.dispatch import reconcile_orphaned_runs
    from milknado.domains.execution import Executor, RunLoop

    reconcile_orphaned_runs(graph)
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
    """Resolve the tmux window target for a run (adapter wiring for ``attach``)."""
    from milknado.domains.dispatch import resolve_attach_target

    return resolve_attach_target(graph, TmuxAdapter(project_root), run_id)


def validate_worker_cmd(worker_cmd: str | None) -> None:
    """Reject an explicit worker_cmd whose executable isn't an allowed AI agent CLI.

    Eager pre-check on the MCP arg; the env fallback and built-in default are
    validated again where they're resolved (``runner.resolve_worker_cmd``). Both
    routes share ``validate_worker_argv``, so the allowlist lives in one place.
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


def run_inline(
    graph: MikadoGraph,
    cfg: MilknadoConfig,
    root: Path,
    request: InlineRunRequest,
) -> dict[str, object]:
    """Dispatch a node to a blocking subprocess worker; return the run state dict."""
    from milknado.adapters import GitAdapter
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
        GitAdapter(root),
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
) -> dict[str, object]:
    """Start an async worker (optionally in tmux); return the initial run state dict."""
    from milknado.adapters import GitAdapter
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
        def open_graph(self, project_root: Path) -> tuple[MikadoGraph, MilknadoConfig]:
            return open_graph(project_root)

    tmux: TmuxAdapter | None = None
    if use_tmux:
        # Fail closed BEFORE any claim: tmux was explicitly requested, so a
        # missing binary or unstartable server fails the dispatch loudly.
        tmux = TmuxAdapter(root)
        ensure_tmux_ready(tmux)
    git = GitAdapter(root)
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
        # Startup failed after the claim: release the claim with a fenced terminal
        # write so the node is not stranded RUNNING, then re-raise.
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
