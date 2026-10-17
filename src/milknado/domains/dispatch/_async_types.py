"""Immutable inputs for an asynchronous inline worker."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from milknado.domains.common import GitPort, SlotLease
from milknado.domains.dispatch.isolate import IsolateContext
from milknado.domains.dispatch.ports import (
    CompletionCheck,
    GraphSessionPort,
    ProcessPort,
    TmuxPort,
)


@dataclass(frozen=True)
class AsyncRunRequest:
    project_root: Path
    node_id: int
    brief: str
    worker_cmd: str | None
    timeout_seconds: int
    run_id: str
    default_cmd: str
    cwd: Path
    merge_ctx: IsolateContext | None = None
    lease: SlotLease | None = None
    max_node_attempts: int = 1
    completion_check: CompletionCheck | None = None
    base_oid: str | None = None


@dataclass(frozen=True)
class AsyncWorkerContext:
    request: AsyncRunRequest
    log_path: Path
    argv: tuple[str, ...]
    graph_sessions: GraphSessionPort
    git: GitPort
    process: ProcessPort
    tmux: TmuxPort | None = None
