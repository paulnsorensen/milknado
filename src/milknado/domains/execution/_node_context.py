from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from milknado.domains.common.agent_argv import NodeAgentSession
from milknado.domains.common.flavor_codec import Gate


@dataclass(frozen=True)
class ExecutionConfig:
    execution_agent: str
    quality_gates: tuple[Gate, ...] | None
    worktree_pattern: str
    project_root: Path
    brief_prepend: str | None = None
    dispatch_max_retries: int = 2
    dispatch_backoff_seconds: float = 5.0
    commit_footer: str | None = None
    agent_family: str = "claude"
    review: bool = False
    review_agent: str | None = None
    review_max_rounds: int = 0
    review_timeout_seconds: int = 1800
    on_reject: str = "warn"
    session_mode: str = "fresh"
    completion_timeout_seconds: int | None = None
    attempt_timeout_seconds: float | None = None
    max_iterations: int | None = None


@dataclass
class NodeExecutionContext:
    worker_run_id: str
    owner_fence: str | None
    worktree: Path
    session: NodeAgentSession | None
    target_branch: str
    target_oid: str
    base_oid: str | None
    review_round: int
    config: ExecutionConfig
