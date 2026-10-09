from __future__ import annotations

import shlex
import threading
from collections.abc import Callable
from pathlib import Path
from threading import Condition, Event
from time import monotonic
from typing import Literal, cast, final

import psutil

from milknado.adapters._group_action_session import (
    GroupActionSession,
    drain_group_commands,
    record_group_command,
)
from milknado.adapters._loop_worker_evidence import LoopWorkerEvidence
from milknado.adapters.recovery import ExistingWorktreeRecovery
from milknado.domains.common import (
    MilknadoConfig,
    SessionAction,
    SessionContext,
    WorkerOwner,
    resolve_execution_agent_command,
    validate_worker_argv,
)
from milknado.domains.coordinator import (
    TurnIdentity,
    TurnPreflightError,
    TurnRunResult,
    TurnRuntimeRequest,
    TurnRuntimeResult,
)
from milknado.domains.graph import MikadoGraph, TaskAttempt
from milknado.loop._agent import AgentRunSpec
from milknado.loop._process_contract import ProtectionContext
from milknado.loop._process_gate import SpawnOptions
from milknado.loop._process_lifecycle import ProtectedWorker, spawn_protected
from milknado.loop._process_registry import WorkerRegistry
from milknado.loop.sessions import (
    ProviderSessionIdentity,
    RuntimePreflightError,
    RuntimeRecoveryRequest,
    RuntimeRequest,
    RuntimeSession,
    SessionChannel,
    start_or_resume,
)
from milknado.loop.sessions import RuntimeResult as NativeRuntimeResult
from milknado.loop.sessions._codex_policy import translate_argv


@final
class NativeCoordinatorTurns:
    def __init__(
        self, root: Path, config: MilknadoConfig, graph: MikadoGraph | None = None
    ) -> None:
        self.root = root
        self.config = config
        self.graph = graph
        self._active: dict[str, RuntimeSession | GroupActionSession] = {}
        self._stops: dict[str, Event] = {}
        self._lock = threading.Lock()
        self._done = Condition(self._lock)
        self._closing = False
        self._workers = WorkerRegistry()

    def runtime_session(
        self, provider_session_id: str
    ) -> RuntimeSession | GroupActionSession | None:
        with self._lock:
            return self._active.get(provider_session_id)

    def owner(self, turn_id: str, attempt: TaskAttempt | None = None) -> WorkerOwner:
        supervisor = psutil.Process()
        return WorkerOwner(
            turn_id,
            supervisor.pid,
            supervisor.create_time(),
            attempt.attempt_id if attempt else None,
            attempt.node_id if attempt else None,
        )

    def cancel(self, turn_id: str) -> bool:
        with self._lock:
            stop = self._stops.get(turn_id)
            if stop is None:
                return False
            stop.set()
            return True

    def shutdown(self, timeout: float = 10.0) -> None:
        deadline = monotonic() + timeout
        with self._done:
            self._closing = True
            for stop in self._stops.values():
                stop.set()
        stopped = self._workers.stop_all(deadline)
        with self._done:
            while self._stops and (remaining := deadline - monotonic()) > 0:
                _ = self._done.wait(remaining)
            if self._stops or not stopped:
                raise RuntimeError("native worker shutdown is unconfirmed")

    def _spawn(
        self, options: SpawnOptions, turn_id: str, attempt: TaskAttempt | None
    ) -> ProtectedWorker:
        graph = self.graph
        if attempt is not None:
            if graph is None:
                raise TurnPreflightError("execution group command owner is unavailable")
            with graph.synchronization_lock:
                if graph.groups.active_attempt(attempt.group_id) != attempt:
                    raise TurnPreflightError("execution group writer changed before launch")
                if not graph.goal_admission(attempt.node_id).allowed:
                    raise TurnPreflightError("goal review pauses task launch")
                return spawn_protected(options, self._protection(turn_id, attempt))
        return spawn_protected(options, self._protection(turn_id, None))

    def _protection(self, turn_id: str, attempt: TaskAttempt | None) -> ProtectionContext:
        return ProtectionContext(
            LoopWorkerEvidence(self.config.db_path),
            self.owner(turn_id, attempt),
            self.config.db_path,
            self._workers,
        )

    def _command(self, provider: str, cwd: Path) -> list[str]:
        override = self.config.execution_agent if provider == self.config.agent_family else None
        command = resolve_execution_agent_command(
            provider, execution_agent=override, tools=self.config.worker_tools.get(provider)
        )
        argv = shlex.split(command)
        validate_worker_argv(argv)
        if argv[0] != provider:
            raise ValueError("native provider command does not match provider")
        if provider == "codex" and translate_argv(tuple(argv), cwd).cwd != cwd.resolve():
            raise ValueError("native provider effective cwd differs from assigned worktree")
        return argv

    def _group_channel(self, request: TurnRuntimeRequest) -> SessionChannel:
        group, graph = request.group, self.graph
        if group is None or graph is None:
            raise TurnPreflightError("execution group command owner is unavailable")
        assert request.hooks.turn_id
        attempt = request.attempt
        if (
            attempt is None
            or attempt.group_id != group.id
            or graph.runs.get(attempt.attempt_id) is None
        ):
            raise TurnPreflightError("execution group run is unavailable")
        owner = request.hooks.turn_id

        def publish(
            _context: SessionContext,
            actions: tuple[SessionAction, ...],
            invocation_id: str,
            permissions: tuple[tuple[str, ...], tuple[tuple[str, str], ...]],
        ) -> None:
            _ = graph.commands.publish_capabilities(
                attempt.attempt_id, attempt.node_id, invocation_id, owner, actions, *permissions
            )

        channel = SessionChannel(
            sink=request.hooks.event,
            durable_drain=lambda: drain_group_commands(graph, attempt, owner),
            command_state_sink=lambda command, state: record_group_command(graph, command, state),
        )
        channel.set_capability_sink(
            publish,
            on_close=lambda invocation: graph.commands.close_owner(
                attempt.attempt_id, owner, invocation
            ),
        )
        return channel

    def _registration_callback(
        self, request: TurnRuntimeRequest, channel: SessionChannel, active_ids: set[str]
    ) -> Callable[[str], None]:
        def register(provider_id: str) -> None:
            with self._lock:
                if provider_id in self._active:
                    raise ValueError("provider session is already active")
            request.hooks.identity(provider_id)
            incarnation = channel.capture_incarnation()
            if incarnation is None:
                raise RuntimeError("provider session has no active channel")
            session: RuntimeSession | GroupActionSession = RuntimeSession(
                ProviderSessionIdentity(
                    cast(Literal["claude", "codex"], request.provider), provider_id
                ),
                channel,
                incarnation,
            )
            if request.group is not None:
                graph, attempt = self.graph, request.attempt
                if graph is None or attempt is None:
                    raise TurnPreflightError("execution group command owner is unavailable")
                capabilities = graph.commands.capabilities(attempt.attempt_id)
                if capabilities is None or capabilities.owner_incarnation != request.hooks.turn_id:
                    raise TurnPreflightError("execution group owner capabilities are unavailable")
                session = GroupActionSession(
                    session,
                    graph,
                    attempt.attempt_id,
                    request.hooks.turn_id,
                    capabilities.invocation_id,
                )
            with self._lock:
                if provider_id in self._active:
                    raise ValueError("provider session is already active")
                self._active[provider_id] = session
            active_ids.add(provider_id)

        return register

    def _prepare(self, request: TurnRuntimeRequest) -> tuple[Path, list[str]]:
        provider, group = request.provider, request.group
        if provider not in {"claude", "codex"}:
            raise TurnPreflightError("unsupported native provider")
        cwd = Path(group.worktree_path) if group else self.root
        if group is not None:
            if not ExistingWorktreeRecovery(self.root).restore(group):
                raise TurnPreflightError("execution group worktree does not match its branch")
        elif not cwd.is_absolute() or not cwd.is_dir():
            raise TurnPreflightError("coordinator project root is unavailable")
        try:
            return cwd, self._command(provider, cwd)
        except ValueError as error:
            raise TurnPreflightError(str(error)) from error

    def _resume(self, identity: TurnIdentity | None, cwd: Path) -> RuntimeRecoveryRequest | None:
        if identity is None:
            return None
        native_identity = ProviderSessionIdentity(
            cast(Literal["claude", "codex"], identity.family), identity.session_id
        )
        return RuntimeRecoveryRequest(native_identity, cwd)

    def _result(self, native: NativeRuntimeResult) -> TurnRuntimeResult:
        run = native.run
        return TurnRuntimeResult(
            TurnRunResult(run.session_id, run.terminal_confirmed) if run else None,
            native.recovery.turn_confirmed if native.recovery else None,
        )

    def _native_request(
        self, request: TurnRuntimeRequest, stop: Event, active_ids: set[str]
    ) -> RuntimeRequest:
        cwd, argv = self._prepare(request)
        hooks = request.hooks
        channel = (
            self._group_channel(request)
            if request.group is not None
            else SessionChannel(sink=hooks.event)
        )
        if stop.is_set():
            raise TurnPreflightError("native turn is shutting down")
        spec = AgentRunSpec(
            argv,
            request.prompt,
            timeout=self.config.completion_timeout_seconds,
            force_stop_event=stop,
            log_dir=None,
            iteration=1,
            cwd=cwd,
            spawn_worker=lambda options: self._spawn(options, hooks.turn_id, request.attempt),
            on_session_id=self._registration_callback(request, channel, active_ids),
        )
        return RuntimeRequest(spec, channel, self._resume(request.identity, cwd))

    def run(self, request: TurnRuntimeRequest) -> TurnRuntimeResult:
        active_ids: set[str] = set()
        channel: SessionChannel | None = None
        stop = Event()
        with self._done:
            if self._closing:
                raise TurnPreflightError("native turn is shutting down")
            self._stops[request.hooks.turn_id] = stop
        try:
            runtime_request = self._native_request(request, stop, active_ids)
            channel = runtime_request.channel
            try:
                return self._result(start_or_resume(runtime_request))
            except RuntimePreflightError as error:
                raise TurnPreflightError(str(error)) from error
        finally:
            with self._done:
                _ = self._stops.pop(request.hooks.turn_id, None)
                for active_id in active_ids:
                    current = self._active.get(active_id)
                    if current is not None and current.channel is channel:
                        del self._active[active_id]
                self._done.notify_all()
