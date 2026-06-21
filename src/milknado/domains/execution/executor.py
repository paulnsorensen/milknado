from __future__ import annotations

import contextlib
import logging
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from milknado.domains.common.config import Gate
from milknado.domains.common.errors import (
    InvalidTransition,
    RebaseAbortError,
    TransientDispatchError,
)
from milknado.domains.common.types import (
    VALID_TRANSITIONS,
    MikadoNode,
    NodeStatus,
    RebaseResult,
)
from milknado.domains.execution._context import build_node_context

if TYPE_CHECKING:
    from milknado.domains.common.protocols import CrgPort, GitPort, LoopPort
    from milknado.domains.graph import MikadoGraph

_logger = logging.getLogger(__name__)

_TRANSIENT_EXIT_CODES = frozenset({124, 137, 143})
_TRANSIENT_MSG_RE = re.compile(
    r"(429|rate.?limit|too many requests)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ExecutionConfig:
    execution_agent: str
    quality_gates: tuple[Gate, ...] | None
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
    return f"feat(milknado-{node_id}): {subject}\n\n{description}\n\nMilknado-Node: {node_id}"


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:30]


_LOOP_SCAFFOLDING = ("RALPH.md", ".ralph-logs/")


def _git_common_dir(worktree: Path) -> Path | None:
    """Resolve a linked worktree's common git dir (the main checkout's `.git`).

    A linked worktree's `.git` is a gitlink file, not a dir, so the common dir is
    where shared state lives. Returns ``None`` when `worktree` is not a real git
    checkout (e.g. a test double's bare path) so callers can degrade best-effort.
    """
    try:
        common = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=worktree,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return None
    return Path(common)


def _exclude_loop_scaffolding(worktree: Path) -> None:
    """Mark the loop's own scaffolding (RALPH.md, .ralph-logs/) git-ignored so the
    worker's `git add -A` never stages it into the squashed commit.

    Git consults `info/exclude` only in the common git dir, resolved via
    `_git_common_dir`, and we append idempotently. Best-effort: a worktree that is
    not a real git checkout — e.g. a test double's bare path — has no exclude file
    to write, and an exclude-file I/O error (permissions, a non-UTF8 file) must not
    abort dispatch, so failures are logged, not raised.
    """
    common = _git_common_dir(worktree)
    if common is None:
        _logger.debug("Skipping scaffolding-exclude for %s: not a git checkout", worktree)
        return
    exclude = common / "info" / "exclude"
    try:
        exclude.parent.mkdir(parents=True, exist_ok=True)
        existing = exclude.read_text() if exclude.exists() else ""
        present = set(existing.splitlines())
        missing = [p for p in _LOOP_SCAFFOLDING if p not in present]
        if not missing:
            return
        prefix = "" if existing == "" or existing.endswith("\n") else "\n"
        exclude.write_text(existing + prefix + "\n".join(missing) + "\n")
    except (OSError, UnicodeDecodeError) as exc:
        _logger.warning("Failed to exclude loop scaffolding under %s: %s", exclude, exc)


def _preserve_run_logs(worktree: Path, node_id: int) -> None:
    """Copy the worktree's `.ralph-logs/` to `.milknado/logs/<node_id>/` in the
    main checkout root before the worktree is destroyed, so a run's post-mortem
    evidence survives worktree removal. `.milknado/` is already local, untracked
    state.

    The dest root is the main checkout, resolved from the worktree's common git
    dir rather than `worktree.parent` — `worktree_pattern` can legally nest the
    worktree in a subdirectory under the project root, so the parent is not always
    the root. Best-effort: a worktree with no logs (or a test double's bare path
    that is not a git checkout) leaves nothing to preserve, and a copy failure must
    not block worktree cleanup, so failures are logged, not raised.
    """
    logs = worktree / ".ralph-logs"
    if not logs.is_dir():
        return
    common = _git_common_dir(worktree)
    root = common.parent if common is not None else worktree.parent
    dest = root / ".milknado" / "logs" / str(node_id)
    try:
        shutil.copytree(logs, dest, dirs_exist_ok=True)
    except OSError as exc:
        _logger.warning("Failed to preserve run logs for node %d: %s", node_id, exc)


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
                right_id,
                left_id,
                paths,
            )
    blocked = {c[1] for c in conflicts}
    return [nid for nid in ids if nid not in blocked]


class WorktreeManager:
    """Tracks and manages worktree lifecycle: create, clean up, rebase-and-merge."""

    def __init__(self, git: GitPort) -> None:
        self._git = git
        self._worktrees: dict[int, Path] = {}

    def create(self, node_id: int, wt_path: Path, branch: str) -> None:
        self._git.create_worktree(wt_path, branch)
        self._worktrees[node_id] = wt_path
        _exclude_loop_scaffolding(wt_path)

    def ensure_clean(self, node_id: int) -> None:
        """Remove any tracked worktree for node_id, logging failures."""
        if node_id in self._worktrees:
            wt = self._worktrees.pop(node_id)
            if wt.exists():
                try:
                    self._git.remove_worktree(wt)
                except Exception as exc:
                    _logger.warning(
                        "Failed to remove orphan worktree %s for node %d: %s",
                        wt,
                        node_id,
                        exc,
                    )

    def remove(self, node_id: int, wt_path: Path) -> None:
        """Remove a specific worktree path and untrack it, logging failures."""
        self._worktrees.pop(node_id, None)
        try:
            self._git.remove_worktree(wt_path)
        except Exception as exc:
            _logger.warning(
                "Failed to remove worktree %s for node %d: %s",
                wt_path,
                node_id,
                exc,
            )

    def discard(self, node_id: int, wt_path: Path) -> None:
        """Best-effort worktree removal — all exceptions suppressed."""
        self._worktrees.pop(node_id, None)
        with contextlib.suppress(Exception):
            self._git.remove_worktree(wt_path)

    def rebase_and_merge(
        self,
        worktree: Path | None,
        feature_branch: str,
        node_id: int,
        description: str,
        worker_branch: str | None = None,
    ) -> RebaseResult:
        """Squash-commit, rebase onto feature_branch, fast-forward the feature
        branch onto the rebased worker branch, then preserve logs and remove the
        worktree.

        The fast-forward is what actually lands the worker's commit on the feature
        branch — without it the rebased work is stranded on the worker branch. It
        runs only when the squash produced a commit (a no-commit squash has nothing
        to merge) and the rebase succeeded. A fast-forward failure raises: a dirty
        or diverged project root overlapping the merge is a loud failure, not a
        silent skip.
        """
        if not worktree or not worktree.exists():
            return RebaseResult(success=True)
        try:
            msg = _build_commit_message(node_id, description)
            committed = self._git.squash_and_commit(worktree, feature_branch, msg)
            rebase_result = self._git.rebase(worktree, feature_branch)
            if rebase_result.success and committed and worker_branch:
                try:
                    self._git.fast_forward(worker_branch)
                except subprocess.CalledProcessError as exc:
                    raise RebaseAbortError(worktree, stderr=exc.stderr or "") from exc
            return rebase_result
        finally:
            _preserve_run_logs(worktree, node_id)
            self.remove(node_id, worktree)


class Executor:
    def __init__(
        self,
        graph: MikadoGraph,
        git: GitPort,
        ralph: LoopPort,
        crg: CrgPort,
    ) -> None:
        self._graph = graph
        self._ralph = ralph
        self._crg = crg
        self._wt = WorktreeManager(git)
        self._attempts_by_node: dict[int, int] = {}

    def get_attempt_count(self, node_id: int) -> int:
        return self._attempts_by_node.get(node_id, 0)

    def dispatch(
        self,
        node_id: int,
        config: ExecutionConfig,
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
                wait = backoff * (2**attempt)
                _logger.warning(
                    "Dispatch attempt %d/%d failed for node %d: %s. Retrying in %.1fs",
                    attempt + 1,
                    max_retries + 1,
                    node_id,
                    exc,
                    wait,
                )
                time.sleep(wait)
        raise last_exc or RuntimeError("dispatch exhausted retries")

    def _dispatch_once(self, node_id: int, config: ExecutionConfig) -> DispatchResult:
        node = self._graph.get_node(node_id)
        if node is None:
            raise ValueError(f"Node {node_id} not found")

        self._wt.ensure_clean(node_id)

        slug = _slugify(node.description)
        worktree_name = config.worktree_pattern.format(
            node_id=node_id,
            slug=slug,
        )
        wt_path = config.project_root / worktree_name

        resolved_root = config.project_root.resolve()
        resolved_wt = wt_path.resolve()
        if not resolved_wt.is_relative_to(resolved_root):
            raise ValueError(
                f"worktree_pattern resolves outside project_root: "
                f"{resolved_wt!r} is not under {resolved_root!r}"
            )

        branch = f"milknado/{node_id}-{slug}"

        # The ralph MCP path claims the node RUNNING in the dispatching parent
        # (cross-process mutual exclusion) before this detached runner gets here;
        # the in-process TUI / e2e path arrives with a still-PENDING node.
        already_claimed = node.status == NodeStatus.RUNNING and node.run_id is not None
        self._wt.create(node_id, wt_path, branch)
        try:
            if already_claimed:
                # Re-marking RUNNING would be an illegal RUNNING -> RUNNING transition
                # that kills the detached run on startup. Attach worktree metadata via a
                # fence-gated update instead, and do NOT clobber the claim's run_id — the
                # parent's set_pid and the completion fence both depend on it.
                self._graph.set_worktree(node_id, node.run_id, str(wt_path), branch)
                run_id = self._create_ralph_run(node, config, wt_path)
            else:
                self._graph.mark_running(
                    node_id,
                    worktree_path=str(wt_path),
                    branch_name=branch,
                )
                run_id = self._create_ralph_run(node, config, wt_path)
                self._graph.set_run_id(node_id, run_id)
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
            self._cleanup_failed_dispatch(node_id, wt_path)
            raise

        return DispatchResult(
            node_id=node_id,
            worktree=wt_path,
            run_id=run_id,
        )

    def _create_ralph_run(
        self,
        node: MikadoNode,
        config: ExecutionConfig,
        wt_path: Path,
    ) -> str:
        """Generate RALPH.md, create the run, start it, and return the run_id."""
        context = build_node_context(node, self._graph, self._crg)
        ralph_path = self._ralph.generate_ralph_md(
            node,
            context,
            config.quality_gates,
            wt_path / "RALPH.md",
        )
        run = self._ralph.create_run(
            agent=config.execution_agent,
            ralph_dir=wt_path,
            ralph_file=ralph_path,
            commands=[],
            quality_gates=config.quality_gates,
            project_root=wt_path,
        )
        run_id = run.state.run_id
        self._ralph.start_run(run_id)
        return run_id

    def _cleanup_failed_dispatch(self, node_id: int, wt_path: Path) -> None:
        """Best-effort state reset; worktree cleanup always runs.

        Release fenced on the node's run_id when it has one (the already-claimed
        ralph path owns the claim under run_id), so a failed dispatch cannot walk
        a node back to PENDING after a different run has re-claimed it. A node with
        no run_id (in-process TUI / a fresh dispatch that failed before set_run_id)
        has no fence to honour and falls back to the unconditional reset.
        """
        try:
            current = self._graph.get_node(node_id)
            if current and current.status == NodeStatus.RUNNING:
                if current.run_id is not None:
                    self._graph.release(node_id, current.run_id)
                else:
                    self._graph.mark_pending(node_id)
        except Exception as reset_exc:
            _logger.warning("state reset failed during cleanup: %s", reset_exc)
        finally:
            self._wt.discard(node_id, wt_path)

    def complete(self, node_id: int, feature_branch: str) -> CompletionResult:
        node = self._graph.get_node(node_id)
        if node is None:
            raise ValueError(f"Node {node_id} not found")

        # A terminal node (DONE/FAILED) reaching complete() again is a legitimate
        # duplicate — the same run reported done twice, or a re-run of an already-
        # finished node. Short-circuit *before* rebase_and_merge so its
        # squash/rebase/worktree-remove side effects are not re-run: mark_done
        # leaves worktree_path set, so without this guard a node whose prior
        # worktree cleanup failed would re-run the workflow. The prior result
        # stands; mirrors reconcile_node_status — the first completion wins.
        if node.status in (NodeStatus.DONE, NodeStatus.FAILED):
            return CompletionResult(
                node_id=node_id,
                rebased=node.status == NodeStatus.DONE,
                newly_ready=[],
                rebase_conflict=None,
            )

        # Any other non-RUNNING status (PENDING/BLOCKED) is not a valid completion
        # target. Silently treating it as a no-op would hide a real state-machine
        # bug — and in the run loop a worker-reported success would print a false
        # completion — so fail loud, exactly as the unguarded mark_done once did.
        if node.status != NodeStatus.RUNNING:
            raise InvalidTransition(
                node_id,
                node.status,
                NodeStatus.DONE,
                tuple(VALID_TRANSITIONS[node.status]),
            )

        worktree = Path(node.worktree_path) if node.worktree_path else None
        try:
            rebase_result = self._wt.rebase_and_merge(
                worktree,
                feature_branch,
                node.id,
                node.description,
                worker_branch=node.branch_name,
            )
        except RebaseAbortError:
            raise  # repo-corruption state: do not swallow
        except Exception as exc:
            _logger.error("Rebase-merge failed for node %d", node_id, exc_info=True)
            rebase_result = RebaseResult(success=False, detail=f"{type(exc).__name__}: {exc}")

        conflict: RebaseConflict | None = None
        if rebase_result.success:
            wrote = self._mark_terminal(node, NodeStatus.DONE)
            if wrote and node.dispatched_at is not None:
                completed_now = datetime.now(UTC)
                duration = (completed_now - node.dispatched_at).total_seconds()
                self._graph.record_completion_duration(node_id, duration)
        else:
            self._mark_terminal(node, NodeStatus.FAILED)
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

    def _mark_terminal(self, node: MikadoNode, status: NodeStatus) -> bool:
        """Write a node's terminal status, fenced on its run_id so a run that was
        reclaimed mid-flight cannot overwrite the fresh owner's row. A node with no
        run_id (in-process TUI legacy / direct test calls) has no fence to honour,
        so the transition is unconditional. Returns whether the write landed."""
        if node.run_id is None:
            if status is NodeStatus.DONE:
                self._graph.mark_done(node.id)
            else:
                self._graph.mark_failed(node.id)
            return True
        return self._graph.mark_terminal(node.id, node.run_id, status)

    def fail(self, node_id: int) -> None:
        self._wt.ensure_clean(node_id)
        node = self._graph.get_node(node_id)
        if node and node.worktree_path:
            wt = Path(node.worktree_path)
            if wt.exists():
                self._wt.remove(node_id, wt)
        self._graph.mark_failed(node_id)
