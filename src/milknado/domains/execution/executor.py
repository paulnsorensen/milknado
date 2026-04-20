from __future__ import annotations

import logging
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from milknado.domains.common.errors import InvalidTransition, TransientDispatchError
from milknado.domains.common.types import RebaseResult
from milknado.domains.execution._context import build_node_context

if TYPE_CHECKING:
    from milknado.domains.common.protocols import CrgPort, GitPort, RalphPort
    from milknado.domains.graph import MikadoGraph

_logger = logging.getLogger(__name__)

_TRANSIENT_EXIT_CODES = frozenset({124, 137, 143})
_TRANSIENT_MSG_RE = re.compile(
    r"(429|rate.?limit|too many requests)", re.IGNORECASE,
)


@dataclass(frozen=True)
class ExecutionConfig:
    execution_agent: str
    quality_gates: tuple[str, ...]
    worktree_pattern: str
    project_root: Path
    dispatch_max_retries: int = 2
    dispatch_backoff_seconds: float = 5.0


@dataclass(frozen=True)
class DispatchResult:
    node_id: int
    worktree: Path
    run_id: str


@dataclass(frozen=True)
class CompletionResult:
    node_id: int
    rebased: bool
    newly_ready: list[int]
    rebase_conflict: RebaseConflict | None = None


@dataclass(frozen=True)
class RebaseConflict:
    node_id: int
    description: str
    conflicting_files: tuple[str, ...]
    detail: str


def _build_commit_message(node_id: int, description: str) -> str:
    subject = description[:57] + "..." if len(description) > 60 else description
    return (
        f"feat(milknado-{node_id}): {subject}\n\n"
        f"{description}\n\n"
        f"Milknado-Node: {node_id}"
    )


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:30]


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, (OSError, subprocess.TimeoutExpired, TransientDispatchError)):
        return True
    if isinstance(exc, subprocess.CalledProcessError):
        if exc.returncode in _TRANSIENT_EXIT_CODES:
            return True
    if _TRANSIENT_MSG_RE.search(str(exc)):
        return True
    return False


def get_dispatchable_nodes(graph: MikadoGraph) -> list[int]:
    ready = graph.get_ready_nodes()
    ids = [n.id for n in ready]
    conflicts = graph.check_parallel_safety(ids)
    if conflicts:
        milknado_logger = logging.getLogger("milknado")
        for left_id, right_id, paths in conflicts:
            milknado_logger.info(
                "Node %d blocked by Node %d on shared files: %s",
                right_id, left_id, paths,
            )
    blocked = {c[1] for c in conflicts}
    return [nid for nid in ids if nid not in blocked]


class Executor:
    def __init__(
        self,
        graph: MikadoGraph,
        git: GitPort,
        ralph: RalphPort,
        crg: CrgPort,
    ) -> None:
        self._graph = graph
        self._git = git
        self._ralph = ralph
        self._crg = crg
        self._worktrees: dict[int, Path] = {}
        self._attempts_by_node: dict[int, int] = {}

    def _ensure_clean_worktree(self, node_id: int) -> None:
        if node_id in self._worktrees:
            wt = self._worktrees.pop(node_id)
            if wt.exists():
                try:
                    self._git.remove_worktree(wt)
                except Exception as exc:
                    _logger.warning(
                        "Failed to remove orphan worktree %s for node %d: %s",
                        wt, node_id, exc,
                    )

    def get_attempt_count(self, node_id: int) -> int:
        return self._attempts_by_node.get(node_id, 0)

    def dispatch(
        self, node_id: int, config: ExecutionConfig,
    ) -> DispatchResult:
        max_retries = config.dispatch_max_retries
        backoff = config.dispatch_backoff_seconds
        last_exc: BaseException | None = None
        for attempt in range(max_retries + 1):
            try:
                result = self._dispatch_once(node_id, config)
                self._attempts_by_node[node_id] = attempt
                return result
            except (InvalidTransition, ValueError):
                raise
            except Exception as exc:
                last_exc = exc
                self._attempts_by_node[node_id] = attempt
                if not _is_transient(exc) or attempt >= max_retries:
                    raise
                wait = backoff * (2 ** attempt)
                _logger.warning(
                    "Dispatch attempt %d/%d failed for node %d: %s. Retrying in %.1fs",
                    attempt + 1, max_retries + 1, node_id, exc, wait,
                )
                time.sleep(wait)
        raise last_exc or RuntimeError("dispatch exhausted retries")

    def _dispatch_once(
        self, node_id: int, config: ExecutionConfig
    ) -> DispatchResult:
        node = self._graph.get_node(node_id)
        if node is None:
            raise ValueError(f"Node {node_id} not found")

        self._ensure_clean_worktree(node_id)

        slug = _slugify(node.description)
        worktree_name = config.worktree_pattern.format(
            node_id=node_id, slug=slug,
        )
        wt_path = config.project_root / worktree_name
        branch = f"milknado/{node_id}-{slug}"

        self._git.create_worktree(wt_path, branch)
        self._worktrees[node_id] = wt_path
        try:
            self._graph.mark_running(
                node_id, worktree_path=str(wt_path), branch_name=branch,
            )

            context = build_node_context(node, self._graph, self._crg)
            ralph_path = self._ralph.generate_ralph_md(
                node, context, list(config.quality_gates),
                wt_path / "RALPH.md",
            )

            run = self._ralph.create_run(
                agent=config.execution_agent,
                ralph_dir=wt_path,
                ralph_file=ralph_path,
                commands=[],
                quality_gates=list(config.quality_gates),
                project_root=wt_path,
            )
            run_id = run.state.run_id
            self._graph.set_run_id(node_id, run_id)
            self._ralph.start_run(run_id)
            self._graph.set_dispatched_at(node_id)
        except Exception as exc:
            _logger.error(
                "Dispatch failed for node %d (%s): %s: %s",
                node_id,
                node.description[:80],
                type(exc).__name__,
                str(exc)[:200],
                exc_info=True,
            )
            self._graph.mark_pending(node_id)
            self._git.remove_worktree(wt_path)
            self._worktrees.pop(node_id, None)
            raise

        return DispatchResult(
            node_id=node_id, worktree=wt_path, run_id=run_id,
        )

    def complete(
        self, node_id: int, feature_branch: str
    ) -> CompletionResult:
        node = self._graph.get_node(node_id)
        if node is None:
            raise ValueError(f"Node {node_id} not found")

        worktree = Path(node.worktree_path) if node.worktree_path else None
        try:
            rebase_result = self._rebase_and_merge(
                worktree, feature_branch, node.id, node.description,
            )
        except Exception:
            rebase_result = RebaseResult(success=False)

        self._worktrees.pop(node_id, None)

        conflict: RebaseConflict | None = None
        if rebase_result.success:
            self._graph.mark_done(node_id)
            if node.dispatched_at is not None:
                completed_now = datetime.now(UTC)
                duration = (completed_now - node.dispatched_at).total_seconds()
                self._graph._record_completion_duration(node_id, duration)
        else:
            self._graph.mark_failed(node_id)
            if rebase_result.conflicting_files or rebase_result.detail:
                conflict = RebaseConflict(
                    node_id=node_id,
                    description=node.description,
                    conflicting_files=rebase_result.conflicting_files,
                    detail=rebase_result.detail,
                )

        newly_ready = get_dispatchable_nodes(self._graph) if rebase_result.success else []
        return CompletionResult(
            node_id=node_id,
            rebased=rebase_result.success,
            newly_ready=newly_ready,
            rebase_conflict=conflict,
        )

    def _rebase_and_merge(
        self,
        worktree: Path | None,
        feature_branch: str,
        node_id: int,
        description: str,
    ) -> RebaseResult:
        if not worktree or not worktree.exists():
            return RebaseResult(success=True)
        try:
            msg = _build_commit_message(node_id, description)
            self._git.squash_and_commit(worktree, feature_branch, msg)
            return self._git.rebase(worktree, feature_branch)
        finally:
            self._git.remove_worktree(worktree)

    def fail(self, node_id: int) -> None:
        self._ensure_clean_worktree(node_id)
        node = self._graph.get_node(node_id)
        if node and node.worktree_path:
            wt = Path(node.worktree_path)
            if wt.exists():
                self._git.remove_worktree(wt)
        self._graph.mark_failed(node_id)
