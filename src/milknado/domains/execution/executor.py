from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from milknado.domains.common.agent_argv import NodeAgentSession, capture_session_id
from milknado.domains.common.config import Gate
from milknado.domains.common.errors import (
    GitOperationError,
    InvalidTransition,
    RebaseAbortError,
    TransientDispatchError,
    UnlandedWorkError,
)
from milknado.domains.common.paths import slugify
from milknado.domains.common.protocols import GraphExecutionSnapshot
from milknado.domains.common.types import (
    VALID_TRANSITIONS,
    MikadoNode,
    NodeStatus,
    RebaseResult,
    RunResult,
)
from milknado.domains.dispatch import render_brief
from milknado.domains.dispatch._runstate import is_cancel_requested, make_run_id, runs_dir
from milknado.loop import RunStatus

if TYPE_CHECKING:
    from milknado.domains.common.protocols import CrgPort, GitPort, GraphReadPort, LoopPort
    from milknado.domains.graph import MikadoGraph

_logger = logging.getLogger(__name__)

# Cancel-watcher cadence for pid-less in-process ralph runs: poll the on-disk
# sentinel often enough that cancel_run's 8s confirm bound is comfortably met.
_RALPH_CANCEL_POLL_SECS = 0.25
_RALPH_CANCEL_STOP_TIMEOUT_SECS = 5.0
# Ceiling for the watcher's retry backoff on consecutive force-stop failures:
# a permanently wedged loop stays loud (first failure logs immediately) but
# the retry cadence caps at one error line per minute (~1.4k lines/day) instead
# of polling at full rate forever. Retry-until-stopped semantics are preserved.
_RALPH_CANCEL_RETRY_BACKOFF_MAX_SECS = 60.0

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
    brief_prepend: str | None = None
    dispatch_max_retries: int = 2
    dispatch_backoff_seconds: float = 5.0
    commit_footer: str | None = None
    agent_family: str = "claude"
    review: bool = False
    review_agent: str | None = None
    review_max_rounds: int = 0
    on_reject: str = "warn"
    session_mode: str = "fresh"
    completion_timeout_seconds: int | None = None


@dataclass(frozen=True)
class RuntimePolicy:
    session: NodeAgentSession | None = None


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
    redispatch: DispatchResult | None = None
    blocked: bool = False
    review_notification_failed: bool = False


@dataclass(frozen=True)
class RebaseConflict:
    node_id: int
    description: str
    conflicting_files: tuple[str, ...]
    detail: str


def _build_commit_message(node_id: int, description: str) -> str:
    subject = description[:57] + "..." if len(description) > 60 else description
    return f"feat(milknado-{node_id}): {subject}\n\n{description}\n\nMilknado-Node: {node_id}"


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
        retained = sorted(
            (path for path in dest.rglob("*") if path.is_file()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for stale in retained[20:]:
            stale.unlink(missing_ok=True)
    except OSError as exc:
        _logger.warning("Failed to preserve run logs for node %d: %s", node_id, exc)


_MAX_RELOCATION_SLOTS = 100


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, (OSError, subprocess.TimeoutExpired, TransientDispatchError)):
        return True
    if isinstance(exc, subprocess.CalledProcessError):
        if exc.returncode in _TRANSIENT_EXIT_CODES:
            return True
    if _TRANSIENT_MSG_RE.search(str(exc)):
        return True
    return False


def _dispatchable_node_ids(
    snapshot: GraphExecutionSnapshot,
    logged_blocks: set[tuple[int, int, tuple[str, ...]]] | None = None,
) -> list[int]:
    ready_ids = snapshot.ready_node_ids
    active_ids = set(snapshot.running_node_ids)
    blocked_relationships = {
        (right_id, left_id, paths)
        for left_id, right_id, paths in snapshot.parallel_conflicts
        if right_id in ready_ids and (left_id in active_ids or left_id in ready_ids)
    }
    if logged_blocks is None:
        newly_blocked = blocked_relationships
    else:
        newly_blocked = blocked_relationships - logged_blocks
        logged_blocks.intersection_update(blocked_relationships)
        logged_blocks.update(blocked_relationships)
    for right_id, left_id, paths in sorted(newly_blocked):
        _logger.info("Node %d blocked by Node %d on shared files: %s", right_id, left_id, paths)
    blocked_ids = {item[0] for item in blocked_relationships}
    return [node_id for node_id in ready_ids if node_id not in blocked_ids]


def get_dispatchable_nodes(
    graph: GraphReadPort,
    logged_blocks: set[tuple[int, int, tuple[str, ...]]] | None = None,
) -> list[int]:
    return _dispatchable_node_ids(graph.get_execution_snapshot([]), logged_blocks)


def get_execution_overview(
    graph: GraphReadPort,
    node_ids: list[int],
    excluded_node_ids: set[int] | None = None,
) -> tuple[str, dict[int, str], int]:
    snapshot = graph.get_execution_snapshot(node_ids)
    excluded = excluded_node_ids or set()
    available = sum(node_id not in excluded for node_id in _dispatchable_node_ids(snapshot))
    return (
        snapshot.root.description if snapshot.root is not None else "",
        {node.id: node.description for node in snapshot.nodes},
        available,
    )


class WorktreeManager:
    """Tracks and manages worktree lifecycle: create, clean up, rebase-and-merge."""

    def __init__(self, git: GitPort) -> None:
        self._git = git
        self._worktrees: dict[int, Path] = {}

    def current_branch(self) -> str:
        return self._git.current_branch()

    def resolve_ref(self, ref: str) -> str:
        return self._git.resolve_ref(ref)

    def create(self, node_id: int, wt_path: Path, branch: str) -> None:
        self._git.create_worktree(wt_path, branch)
        self._worktrees[node_id] = wt_path
        _exclude_loop_scaffolding(wt_path)

    def relocate_occupied(self, wt_path: Path, branch: str) -> tuple[Path, str]:
        """Relocate a dispatch whose canonical path or branch is still taken.

        A refused (dirty/unlanded) orphan stays on disk at the node's canonical
        worktree path with its branch checked out, so dispatch degrades by
        relocating: a deterministic `-<n>` suffix on both path and branch, first
        slot where BOTH are free wins. `git worktree remove` never deletes the
        branch, so a slot whose path was cleaned off disk can still own a live
        branch; advancing on the path alone would return that taken branch and
        make `git worktree add -b` fail. The preserved orphan is left for a human
        or an explicit discard; the run loop is never blocked by it.
        """
        if not wt_path.exists() and not self._git.branch_exists(branch):
            return wt_path, branch
        for n in range(2, _MAX_RELOCATION_SLOTS):
            candidate = wt_path.with_name(f"{wt_path.name}-{n}")
            candidate_branch = f"{branch}-{n}"
            if not candidate.exists() and not self._git.branch_exists(candidate_branch):
                _logger.warning(
                    "Worktree path %s is occupied by a preserved orphan; "
                    "relocating dispatch to %s",
                    wt_path,
                    candidate,
                )
                return candidate, candidate_branch
        raise RuntimeError(
            f"no free worktree slot for {wt_path} after {_MAX_RELOCATION_SLOTS - 2} relocations"
        )

    def ensure_clean(self, node_id: int) -> None:
        """Remove any tracked worktree for node_id; degrade on refusal.

        Pre-dispatch defensive cleanup, not teardown of finished work: a refusal
        (dirty/unlanded work) keeps the worktree, logs what is at risk, and lets
        dispatch proceed — `_dispatch_once` relocates to a fresh path rather
        than blocking the run loop on preserved work.
        """
        if node_id in self._worktrees:
            wt = self._worktrees.pop(node_id)
            if wt.exists():
                try:
                    self._git.remove_worktree(wt)
                except UnlandedWorkError as exc:
                    _logger.warning(
                        "Keeping worktree for node %d (dispatch relocates): %s", node_id, exc
                    )
                except Exception as exc:
                    _logger.warning(
                        "Failed to remove orphan worktree %s for node %d: %s",
                        wt,
                        node_id,
                        exc,
                    )

    def remove(self, node_id: int, wt_path: Path, target: str = "HEAD") -> None:
        """Remove a worktree and untrack it — fail-closed.

        A refusal (dirty or unlanded work, `UnlandedWorkError`) propagates: the
        caller's operation hard-fails rather than silently destroying or
        silently skipping. Any other removal failure destroyed nothing — the
        worktree is still on disk — so it is logged and swallowed to keep the
        node lifecycle moving, as before.
        """
        self._worktrees.pop(node_id, None)
        try:
            self._git.remove_worktree(wt_path, target)
        except UnlandedWorkError:
            raise
        except Exception as exc:
            _logger.warning(
                "Failed to remove worktree %s for node %d: %s",
                wt_path,
                node_id,
                exc,
            )

    def discard(self, node_id: int, wt_path: Path) -> None:
        """Explicitly destructive, best-effort removal — the ONLY path to
        `--force` (`force_remove_worktree`); all exceptions suppressed."""
        self._worktrees.pop(node_id, None)
        with contextlib.suppress(Exception):
            self._git.force_remove_worktree(wt_path)

    def rebase_and_merge(
        self,
        worktree: Path | None,
        feature_branch: str,
        node_id: int,
        description: str,
        worker_branch: str | None = None,
        target_branch: str | None = None,
    ) -> RebaseResult:
        """Squash, rebase, fast-forward, and remove only landed worktrees."""
        merge_target = target_branch or feature_branch
        current_target = self._git.current_branch()
        if feature_branch != merge_target or current_target != merge_target:
            raise GitOperationError(
                "merge-back target",
                f"dispatch target is {merge_target!r}; caller requested "
                f"{feature_branch!r}, checkout is {current_target!r}",
            )
        if not worktree or not worktree.exists():
            return RebaseResult(success=True)
        landed = False
        try:
            find_collisions = getattr(type(self._git), "untracked_merge_collisions", None)
            collisions = self._git.untracked_merge_collisions(worktree) if find_collisions else ()
            if collisions:
                paths = ", ".join(collisions)
                raise GitOperationError(
                    f"merge --ff-only {worker_branch or '(worker branch unavailable)'}",
                    f"untracked integration-checkout path collision: {paths}; "
                    f"target branch={merge_target!r}, worker worktree={worktree}",
                )
            msg = _build_commit_message(node_id, description)
            committed = self._git.squash_and_commit(worktree, merge_target, msg)
            rebase_result = self._git.rebase(worktree, merge_target)
            if rebase_result.success and committed and worker_branch:
                self._git.fast_forward(worker_branch)
            landed = rebase_result.success
            return rebase_result
        finally:
            _preserve_run_logs(worktree, node_id)
            if landed:
                self.remove(node_id, worktree, target=merge_target)
            else:
                _logger.error(
                    "Merge-back for node %d did not land; keeping worktree %s for inspection",
                    node_id,
                    worktree,
                )


class Executor:
    def __init__(
        self,
        graph: MikadoGraph,
        git: GitPort,
        ralph: LoopPort,
        crg: CrgPort,
    ) -> None:

        self._base_oid_by_node: dict[int, str | None] = {}
        self._worker_run_id_by_node: dict[int, str] = {}
        self._owner_fence_by_node: dict[int, str] = {}
        # Runs whose dispatch-abort force-stop was never confirmed: the cancel
        # watcher finalizes their rows when the wedged loop later self-exits
        # (no completion path owns these ids — they never enter
        # _worker_run_id_by_node).
        self._unconfirmed_stop_run_ids: set[str] = set()
        self._git = git
        self._target_branch_by_node: dict[int, str] = {}
        self._target_oid_by_node: dict[int, str] = {}
        self._graph = graph
        self._wt = WorktreeManager(git)
        self._ralph = ralph
        self._crg = crg
        self._attempts_by_node: dict[int, int] = {}
        self._config_by_node: dict[int, ExecutionConfig] = {}
        self._session_by_node: dict[int, NodeAgentSession] = {}
        self._review_round_by_node: dict[int, int] = {}

    def dispatch(
        self,
        node_id: int,
        config: ExecutionConfig,
        *,
        base_oid: str | None = None,
        parent_run_id: str | None = None,
    ) -> DispatchResult:
        self._review_enabled(config)
        max_retries = config.dispatch_max_retries
        backoff = config.dispatch_backoff_seconds
        last_exc: BaseException | None = None
        target_branch = self._wt.current_branch()
        target_oid = base_oid or self._wt.resolve_ref(target_branch)
        for attempt in range(max_retries + 1):
            try:
                result = self._dispatch_once(
                    node_id,
                    config,
                    base_oid=target_oid,
                    parent_run_id=parent_run_id,
                )
                self._base_oid_by_node[node_id] = target_oid
                self._worker_run_id_by_node[node_id] = result.run_id
                if parent_run_id is not None:
                    self._owner_fence_by_node[node_id] = parent_run_id
                self._target_branch_by_node[node_id] = target_branch
                self._target_oid_by_node[node_id] = target_oid
                self._attempts_by_node[node_id] = attempt
                self._config_by_node[node_id] = config
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

    def _resolve_worktree_path(
        self, node_id: int, node: MikadoNode, config: ExecutionConfig
    ) -> tuple[Path, str]:
        slug = slugify(node.description, max_length=30) or "node"
        worktree_name = config.worktree_pattern.format(node_id=node_id, slug=slug)
        wt_path = config.project_root / worktree_name
        resolved_root = config.project_root.resolve()
        resolved_wt = wt_path.resolve()
        if not resolved_wt.is_relative_to(resolved_root):
            raise ValueError(
                f"worktree_pattern resolves outside project_root: "
                f"{resolved_wt!r} is not under {resolved_root!r}"
            )
        return wt_path, f"milknado/{node_id}-{slug}"

    def _dispatch_once(
        self,
        node_id: int,
        config: ExecutionConfig,
        *,
        base_oid: str | None = None,
        parent_run_id: str | None = None,
    ) -> DispatchResult:
        node = self._graph.get_node(node_id)
        if node is None:
            raise ValueError(f"Node {node_id} not found")
        if (
            parent_run_id is None
            and node.status is NodeStatus.FAILED
            and node.worktree_path
            and node.branch_name
        ):
            raise ValueError(
                f"node {node_id} retains worker branch {node.branch_name!r} at "
                f"{node.worktree_path}; resolve and retry merge-back before redispatching"
            )

        adopted = parent_run_id is not None
        if adopted:
            if node.status is not NodeStatus.RUNNING or node.run_id != parent_run_id:
                raise InvalidTransition(
                    node_id,
                    node.status,
                    NodeStatus.RUNNING,
                    tuple(VALID_TRANSITIONS[node.status]),
                )
            owner_run_id = parent_run_id
        else:
            owner_run_id = make_run_id(node_id)
            if not self._graph.claim_node(
                node_id, owner_run_id, now=datetime.now(UTC).isoformat(), pid=os.getpid()
            ):
                raise ValueError(f"node {node_id} is already claimed; dispatch refused")

        create_attempted = False
        run_id: str | None = None
        wt_path, branch = self._resolve_worktree_path(node_id, node, config)
        try:
            self._wt.ensure_clean(node_id)
            wt_path, branch = self._wt.relocate_occupied(wt_path, branch)
            self._graph.set_worktree(node_id, owner_run_id, str(wt_path), branch)
            create_attempted = True
            self._wt.create(node_id, wt_path, branch)
            dispatch_base_oid = base_oid or self._wt.resolve_ref(self._wt.current_branch())
            run_id = self._create_ralph_run(node, config, wt_path, dispatch_base_oid)
            if not adopted and not self._graph.replace_run_id(node_id, owner_run_id, run_id):
                raise ValueError(f"dispatch fence lost for node {node_id}")
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
            self._cleanup_failed_dispatch(
                node_id,
                wt_path,
                owner_run_id=owner_run_id,
                release_claim=not adopted,
                discard_worktree=create_attempted,
                started_run_id=(
                    run_id
                    if run_id is not None
                    else getattr(exc, "_milknado_aborted_run_id", None)
                ),
                stop_confirmed=getattr(exc, "_milknado_stop_confirmed", None),
            )
            raise
        return DispatchResult(node_id=node_id, worktree=wt_path, run_id=run_id)

    def _create_ralph_run(
        self,
        node: MikadoNode,
        config: ExecutionConfig,
        wt_path: Path,
        base_oid: str,
        *,
        session: NodeAgentSession | None = None,
        prior_findings: str = "",
        findings_round: int | None = None,
    ) -> str:
        brief = render_brief(
            self._graph,
            node.id,
            prepend=config.brief_prepend,
            project_root=config.project_root,
        )
        ralph_path = self._ralph.generate_ralph_md(
            brief,
            config.quality_gates,
            wt_path / "RALPH.md",
            prior_findings=prior_findings,
            findings_round=findings_round,
        )
        ralph_run_id = make_run_id(node.id)
        create_kwargs: dict[str, Any] = {
            "agent": config.execution_agent,
            "ralph_dir": wt_path,
            "ralph_file": ralph_path,
            "commands": [],
            "quality_gates": config.quality_gates,
            "project_root": wt_path,
            "commit_footer": config.commit_footer,
            "base_oid": base_oid,
            "run_id": ralph_run_id,
        }
        if node.flavor == "review":
            create_kwargs["completion_probe"] = lambda run_id=ralph_run_id: (
                self._graph.latest_run_message(run_id, "review_terminal") is not None
            )
        if session is not None:
            create_kwargs["runtime_policy"] = RuntimePolicy(session=session)
        run = self._ralph.create_run(**create_kwargs)
        run_id = run.state.run_id
        if run_id != ralph_run_id:
            raise ValueError(f"ralph run id mismatch: requested {ralph_run_id!r}, got {run_id!r}")
        # Register the ralph run in the runs table BEFORE starting the loop so
        # review-verdict run_messages inserts satisfy the FK and run_list/poll
        # can see it (#296). Persistence must precede execution: an unregistered
        # run is worse than a failed dispatch, so a failed insert propagates
        # rather than being swallowed — the loop has not started yet (only
        # create_run ran, registering an in-memory RunManager entry but no
        # thread), so nothing needs teardown here. The row records
        # timeout_seconds (the completion timeout) and NO pid, mirroring the
        # async-worker path — so after a server crash fail_stale_running_runs
        # (reconcile.py) sweeps the orphaned row once started_at + timeout +
        # grace elapses; while the server lives, the headless completion-
        # timeout force-stop drives a live run to terminal at timeout (before
        # the sweep's timeout+grace window), so the row is never false-
        # flipped. No pid means cancel correctly routes through the sentinel
        # path, never _cancel_pid_run.
        log_dir = wt_path / ".ralph-logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "0000-dispatch.log").touch()
        self._graph.start_run(
            run_id,
            node.id,
            str(log_dir),
            datetime.now(UTC).isoformat(),
            config.completion_timeout_seconds,
        )
        self._ralph.start_run(run_id)
        try:
            self._spawn_ralph_cancel_watcher(run_id, config.project_root)
        except Exception as exc:
            # The run is live but its id never escapes _create_ralph_run, so
            # the caller's cleanup cannot see it: tear it down here with the
            # same confirmed-stop + fenced-finalize discipline as every other
            # dispatch-abort path, and carry the outcome across the boundary
            # on the propagating exception so _dispatch_once's cleanup honors
            # an UNCONFIRMED stop (preserve the worktree, fail loud) instead
            # of defaulting to discard under a possibly-live loop.
            stop_confirmed = self._stop_aborted_run(
                run_id, context=f"post-start setup failed for node {node.id}"
            )
            exc._milknado_aborted_run_id = run_id
            exc._milknado_stop_confirmed = stop_confirmed
            raise
        return run_id

    def _finalize_worker_run(self, run_id: str | None, result: RunResult) -> None:
        """Best-effort terminal write for an executor ralph run's runs row.

        Executor ralph rows are registered at dispatch (#296); without a
        matching finish they zombie as status='running' forever and pid-less
        cancel takes a sentinel path nothing observed. The finish_run fence
        (status='running') makes double-writes safe; the pre-check only keeps
        the fence's dropped-write warning out of the logs. A failed finalize
        must not kill node completion — loud log, like the dispatch insert.
        """
        if not run_id:
            return
        try:
            row = self._graph.get_run(run_id)
            if row is None or row.get("status") != "running":
                return
            self._graph.finish_run(run_id, result)
        except Exception:
            _logger.exception("runs-row finalize failed for ralph run %s", run_id)

    def _finish_node_worker_run(self, node_id: int, result: RunResult) -> None:
        self._finalize_worker_run(self._worker_run_id_by_node.get(node_id), result)

    def _spawn_ralph_cancel_watcher(self, run_id: str, project_root: Path) -> None:
        """Watch the on-disk cancel sentinel for a pid-less ralph run.

        cancel_run() marks a pid-less run via the async-sentinel path, but the
        in-process ralph loop never observes the sentinel — live cancel stalled
        for the full confirm bound and then raised. This daemon thread is the
        observer: on the sentinel it force-stops the loop and finalizes the
        row cancelled, which cancel_run's confirm poll then sees. A dead run
        thread (or a missing run record) exits the watcher — the executor
        completion path owns that row's finalize, except for an
        aborted-after-start run whose stop was never confirmed
        (_unconfirmed_stop_run_ids): no completion path owns those ids, so
        the watcher finalizes the row itself when the wedged loop self-exits
        rather than letting it zombie 'running' forever. If the force-stop
        fails or raises, the row is deliberately left running so
        cancel_run's confirm bound expires and raises the preserving error
        instead of tearing a worktree out from under a live worker — but
        the sentinel stays set, so the watcher keeps observing and retries
        the force-stop with a capped exponential backoff (doubling from the
        poll cadence to _RALPH_CANCEL_RETRY_BACKOFF_MAX_SECS): a transient
        stop failure must not permanently strand the cancel, and a
        permanently wedged loop must not log at full poll rate forever.
        """
        rdir = runs_dir(project_root)
        threading.Thread(
            target=self._watch_ralph_cancel,
            args=(run_id, rdir),
            name=f"ralph-cancel-watch-{run_id}",
            daemon=True,
        ).start()

    def _watch_ralph_cancel(self, run_id: str, rdir: Path) -> None:
        """Poll loop for one cancel watcher — see _spawn_ralph_cancel_watcher
        for the full behavioral contract this implements."""
        backoff = _RALPH_CANCEL_POLL_SECS
        while True:
            run = self._ralph.get_run(run_id)
            if run is None:
                # The run record is gone; nothing left to watch.
                self._unconfirmed_stop_run_ids.discard(run_id)
                return
            thread = getattr(run, "thread", None)
            if thread is not None and not thread.is_alive():
                self._finalize_dead_watched_thread(run_id)
                return
            if not is_cancel_requested(rdir, run_id):
                backoff = _RALPH_CANCEL_POLL_SECS
                time.sleep(_RALPH_CANCEL_POLL_SECS)
                continue
            stopped, backoff = self._attempt_watcher_force_stop(run_id, backoff)
            if stopped is None:
                continue
            if stopped:
                self._finalize_watcher_stop(run_id)
            return

    def _finalize_dead_watched_thread(self, run_id: str) -> None:
        """The watched thread already exited with no cancel ever requested:
        only an aborted-after-start run with an unconfirmed stop has no
        other owner — finalize it here so it doesn't zombie 'running'
        forever. The finish_run fence keeps any late finalize
        exactly-one-winner."""
        if run_id not in self._unconfirmed_stop_run_ids:
            return
        self._unconfirmed_stop_run_ids.discard(run_id)
        self._finalize_worker_run(
            run_id,
            RunResult(
                status="failed",
                exit_code=None,
                timed_out=False,
                ended_at=datetime.now(UTC).isoformat(),
                error="loop exited after unconfirmed stop",
            ),
        )

    def _attempt_watcher_force_stop(
        self, run_id: str, backoff: float
    ) -> tuple[bool | None, float]:
        """One force-stop attempt for a cancel-requested run.

        Returns ``(None, new_backoff)`` after a transient failure (logged
        and slept already) so the caller keeps polling, or ``(confirmed,
        backoff)`` once force_stop_run itself resolved either way.
        """
        try:
            stopped = self._ralph.force_stop_run(run_id, timeout=_RALPH_CANCEL_STOP_TIMEOUT_SECS)
        except Exception:
            _logger.exception(
                "force-stop raised for cancelled ralph run %s; row left "
                "running so cancel_run fails closed; retrying while the "
                "sentinel is set",
                run_id,
            )
            time.sleep(backoff)
            return None, min(backoff * 2, _RALPH_CANCEL_RETRY_BACKOFF_MAX_SECS)
        if not stopped:
            _logger.error(
                "force-stop could not confirm exit for cancelled ralph "
                "run %s; row left running so cancel_run fails closed; "
                "retrying while the sentinel is set",
                run_id,
            )
            time.sleep(backoff)
            return None, min(backoff * 2, _RALPH_CANCEL_RETRY_BACKOFF_MAX_SECS)
        return True, backoff

    def _finalize_watcher_stop(self, run_id: str) -> None:
        """A confirmed force-stop: finalize the row cancelled, unless the
        loop's own completion won the race just before the stop landed —
        force_stop_and_join only confirms the thread exited, not why, so a
        completed run must be left for the normal completion path to
        finalize with the real rebase outcome instead of being stomped."""
        completed_run = self._ralph.get_run(run_id)
        if completed_run is not None and completed_run.state.status is RunStatus.COMPLETED:
            return
        self._finalize_worker_run(
            run_id,
            RunResult(
                status="failed",
                exit_code=-1,
                timed_out=False,
                ended_at=datetime.now(UTC).isoformat(),
                error="cancelled",
            ),
        )

    def _review_enabled(self, config: ExecutionConfig) -> bool:
        if not config.review:
            return False
        if not config.review_agent or not config.review_agent.strip():
            raise ValueError("review=true requires a non-empty review_agent")
        return config.review_max_rounds > 0

    def _capture_session(
        self,
        node: MikadoNode,
        config: ExecutionConfig,
    ) -> NodeAgentSession | None:
        if config.session_mode != "resume":
            return None
        existing = self._session_by_node.get(node.id)
        if existing is not None:
            return existing
        worker_run_id = self._worker_run_id_by_node.get(node.id) or node.run_id
        if not worker_run_id:
            raise ValueError(f"node {node.id} has no worker run id to resume")
        lines = self._ralph.get_run_stdout(worker_run_id)
        output = "\n".join(lines)
        session_id: str | None = None
        try:
            session_id = capture_session_id(config.agent_family, output)
        except ValueError:
            for line in reversed(lines):
                try:
                    session_id = capture_session_id(config.agent_family, line)
                    break
                except ValueError:
                    continue
        if session_id is None:
            raise ValueError(
                f"node {node.id} worker output did not contain a resumable session id"
            )
        worktree = Path(node.worktree_path or config.project_root).resolve()
        session = NodeAgentSession(
            node_id=node.id,
            family=config.agent_family,
            session_id=session_id,
            worktree_path=str(worktree),
            created_at=datetime.now(UTC).isoformat(),
        )
        path = config.project_root / ".milknado" / "sessions" / f"node-{node.id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(session), indent=2) + "\n", encoding="utf-8")
        self._session_by_node[node.id] = session
        return session

    def _review_prompt(self, node: MikadoNode, worktree: Path, config: ExecutionConfig) -> str:
        base_oid = self._base_oid_by_node.get(node.id)
        if not base_oid:
            raise ValueError(f"node {node.id} has no dispatch base oid for review")
        diff = self._git.diff_for_review(worktree, base_oid) or "(no diff)"
        try:
            brief = (worktree / "RALPH.md").read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            brief = node.description
        spec_path = node.artifact_path
        if spec_path:
            candidate = Path(spec_path).expanduser()
            if not candidate.is_absolute():
                candidate = (config.project_root / candidate).resolve()
            spec_path = str(candidate)
        return (
            f"# Adversarial review: {node.description}\n\n"
            "## Generated node brief/context\n\n"
            f"{brief.rstrip()}\n\n"
            f"## Spec\n\n{spec_path or '(no spec path)'}\n\n"
            "Review the pinned worktree against the brief and spec. Report "
            "easy-cheese severity/dimension findings with evidence and fixes, "
            "then emit exactly one verdict tag:\n\n"
            "<verdict>approve</verdict>\n\nor\n\n"
            "<verdict>reject</verdict>\n\n"
            f"## Full base-to-worktree diff\n\n```diff\n{diff}\n```\n"
        )

    def _persist_review_findings(
        self,
        node: MikadoNode,
        worktree: Path,
        findings_md: str,
    ) -> None:
        slug = slugify(node.description, max_length=30) or str(node.id)
        path = worktree / ".cheese" / "age" / f"{slug}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(findings_md.rstrip() + "\n", encoding="utf-8")

    def _notify_review(
        self,
        node: MikadoNode,
        *,
        verdict: str,
        findings_md: str,
        round_number: int,
    ) -> bool:
        worker_run_id = self._worker_run_id_by_node.get(node.id) or node.run_id
        if not worker_run_id:
            _logger.error("node_review_notification_missing_run node_id=%d", node.id)
            return False
        now = datetime.now(UTC).isoformat()
        # Durable audit trail keyed by (node_id, round); deliberately has no FK
        # to runs, so this write cannot fail the way the #296 run_messages one did.
        try:
            self._graph.insert_node_review(node.id, round_number, verdict, findings_md, now)
        except Exception:
            _logger.exception(
                "node_review_table_write_failed node_id=%d round=%d", node.id, round_number
            )
        body = json.dumps(
            {"node_id": node.id, "verdict": verdict, "findings": findings_md},
            sort_keys=True,
        )
        try:
            self._graph.deposit_run_message(
                worker_run_id,
                "node_review",
                body,
                now,
            )
        except Exception:
            _logger.exception(
                "node_review_notification_failed node_id=%d run_id=%s",
                node.id,
                worker_run_id,
            )
            return False
        return True

    def _run_review(
        self,
        node: MikadoNode,
        worktree: Path,
        config: ExecutionConfig,
    ) -> tuple[bool, str]:
        self._capture_session(node, config)
        reviewer = getattr(self._ralph, "run_node_review", None)
        if reviewer is None:
            raise ValueError("configured adversarial review requires a LoopPort reviewer")
        result = reviewer(
            config.review_agent,
            self._review_prompt(node, worktree, config),
            worktree,
            config.project_root,
        )
        approved = bool(getattr(result, "approved", False))
        findings = str(getattr(result, "findings_md", "")).strip()
        self._persist_review_findings(node, worktree, findings or "reviewer returned no findings")
        return approved, findings

    def _redispatch_review_round(
        self,
        node: MikadoNode,
        config: ExecutionConfig,
        worktree: Path,
        *,
        prior_findings: str = "",
        findings_round: int | None = None,
    ) -> DispatchResult:
        owner_fence = self._owner_fence_by_node.get(node.id)
        prior_worker_run_id = self._worker_run_id_by_node.get(node.id)
        old_worker_run_id = prior_worker_run_id or node.run_id
        if not old_worker_run_id:
            raise ValueError(
                f"node {node.id} has no run fence/worker run id for review redispatch"
            )
        # The prior round is over (its review already ran) regardless of
        # whether the next round can be created below — finalize it here,
        # unconditionally and exactly once, so a fresh-run failure (raised
        # out of _create_ralph_run before any fence check runs) can never
        # leave it zombied 'running'. Only the recorded worker run id is
        # finalized — never the node.run_id fence fallback, which can be a
        # live parent run's row.
        review_round = self._review_round_by_node.get(node.id, 0)
        self._finalize_worker_run(
            prior_worker_run_id,
            RunResult(
                status="done",
                exit_code=0,
                timed_out=False,
                ended_at=datetime.now(UTC).isoformat(),
                detail=f"superseded by review round {review_round} redispatch",
            ),
        )
        base_oid = self._base_oid_by_node.get(node.id)
        if base_oid is None:
            base_oid = self._wt.resolve_ref(self._wt.current_branch())
        session = self._session_by_node.get(node.id)
        run_id = self._create_ralph_run(
            node,
            config,
            worktree,
            base_oid,
            session=session if config.session_mode == "resume" else None,
            prior_findings=prior_findings,
            findings_round=findings_round,
        )
        if owner_fence is None:
            if not self._graph.replace_run_id(node.id, old_worker_run_id, run_id):
                self._stop_aborted_run(
                    run_id, context=f"review redispatch fence lost for node {node.id}"
                )
                raise ValueError(f"review redispatch fence lost for node {node.id}")
        else:
            current = self._graph.get_node(node.id)
            if current is None or current.run_id != owner_fence:
                self._stop_aborted_run(
                    run_id, context=f"adopted owner fence lost for node {node.id}"
                )
                raise ValueError(f"adopted owner fence lost for node {node.id}")
        self._worker_run_id_by_node[node.id] = run_id

        return DispatchResult(node_id=node.id, worktree=worktree, run_id=run_id)

    def _cleanup_failed_dispatch(
        self,
        node_id: int,
        wt_path: Path,
        *,
        owner_run_id: str,
        release_claim: bool,
        discard_worktree: bool,
        started_run_id: str | None = None,
        stop_confirmed: bool | None = None,
    ) -> None:
        # The ralph run may already have been started when dispatch aborted
        # (fence loss / set_dispatched_at failure): stop it and finalize its
        # runs row before the worktree it writes into is discarded, so it
        # can't leak as a live loop with an unrecoverable zombie 'running'
        # row. On an UNCONFIRMED stop the row stays 'running' and the
        # worktree is kept — tearing it out from under a live loop is worse
        # than a recoverable leftover directory.
        #
        # stop_confirmed carries a teardown result across the post-start
        # window: _create_ralph_run already stopped the run there (its id
        # never escaped), so cleanup must NOT stop it again — only honor the
        # outcome. None means cleanup owns the stop attempt for
        # started_run_id (the fence-loss paths).
        if stop_confirmed is None:
            stop_confirmed = True
            if started_run_id is not None:
                stop_confirmed = self._stop_aborted_run(
                    started_run_id,
                    context=f"failed-dispatch cleanup for node {node_id}",
                )
        try:
            # An unconfirmed stop means the abandoned loop may still be
            # alive and writing into wt_path: releasing the claim here would
            # let the caller's transient-retry loop reclaim the node and
            # start a second worker (race) while the first is still live.
            # Withhold release until the stop is confirmed (or nothing was
            # ever started, where stop_confirmed defaults True) — the next
            # _dispatch_once attempt then hits claim_node failing and raises
            # ValueError, which the retry loop does not retry.
            if release_claim and stop_confirmed:
                self._graph.release(node_id, owner_run_id)
        except Exception:
            _logger.exception(
                "state reset failed during cleanup node_id=%d worktree=%s",
                node_id,
                wt_path,
            )
        finally:
            if discard_worktree:
                if stop_confirmed:
                    self._wt.discard(node_id, wt_path)
                else:
                    _logger.error(
                        "worktree %s preserved: ralph run %s was not confirmed "
                        "stopped during failed-dispatch cleanup for node %d",
                        wt_path,
                        started_run_id,
                        node_id,
                    )

    def _stop_aborted_run(self, run_id: str, *, context: str) -> bool:
        """Confirmed force-stop + fenced finalize for a ralph run abandoned
        after start.

        Shared teardown for every dispatch-abort path (dispatch fence loss,
        review-redispatch fence loss, post-start setup failure). The run's
        worktree is about to be discarded or reused, so the stop is a
        force-stop; the runs row is finalized only when the stop is
        confirmed. On an unconfirmed stop — force_stop_run returned False or
        raised — the row is deliberately left 'running' and False is
        returned so the caller keeps the worktree instead of tearing it out
        from under a still-live loop: the same fail-closed policy as the
        cancel watcher. The run id is registered in
        _unconfirmed_stop_run_ids so the cancel watcher — which was spawned
        before the abort on every path except the post-start window —
        finalizes the row (fenced, 'loop exited after unconfirmed stop')
        when the wedged loop later self-exits; these ids never enter
        _worker_run_id_by_node, so no completion path owns them. Accepted
        residual: a run aborted in the post-start window whose watcher
        itself failed to spawn has no observer, so on an unconfirmed stop
        there its row has no in-tree finalizer — the loud preserved-worktree
        error names the run id for manual recovery.
        """
        # Register before attempting the stop, not after failure: the cancel
        # watcher (already running independently) can observe a dead thread
        # the instant force_stop_run kills it, and checks this set at that
        # exact moment. Adding only on a failure branch left a window where
        # the watcher's check could land before this call ever completes,
        # permanently zombie-ing the row since nothing else would own it.
        self._unconfirmed_stop_run_ids.add(run_id)
        try:
            stopped = self._ralph.force_stop_run(run_id, timeout=_RALPH_CANCEL_STOP_TIMEOUT_SECS)
        except Exception:
            _logger.exception(
                "force-stop raised for aborted ralph run %s (%s); row left running",
                run_id,
                context,
            )
            return False
        if not stopped:
            _logger.error(
                "force-stop could not confirm exit for aborted ralph run %s (%s); "
                "row left running",
                run_id,
                context,
            )
            return False
        self._unconfirmed_stop_run_ids.discard(run_id)
        self._finalize_worker_run(
            run_id,
            RunResult(
                status="failed",
                exit_code=None,
                timed_out=False,
                ended_at=datetime.now(UTC).isoformat(),
                error="dispatch aborted",
            ),
        )
        return True

    def note_unconfirmed_stop(self, run_id: str) -> None:
        """Register a run whose stop request was not confirmed so its cancel
        watcher (if one exists) finalizes the row when the run later
        self-exits, instead of leaving a permanent 'running' zombie.

        Used by callers (e.g. headless dispatch) that issue their own
        stop_run/force_stop_run outside Executor's own abort paths
        (_stop_aborted_run) and so would otherwise never register the id.
        """
        self._unconfirmed_stop_run_ids.add(run_id)

    def _validate_completion_target(self, node_id: int, feature_branch: str) -> None:
        target_branch = self._target_branch_by_node.get(node_id)
        if target_branch is None:
            return
        current_branch = self._wt.current_branch()
        if feature_branch != target_branch or current_branch != target_branch:
            target_oid = self._target_oid_by_node.get(node_id, "(unknown)")
            raise GitOperationError(
                "dispatch-time merge target",
                f"dispatch captured {target_branch!r} at {target_oid}; "
                f"caller requested {feature_branch!r}, checkout is {current_branch!r}",
            )

    def _rebase_or_fail(
        self,
        worktree: Path | None,
        feature_branch: str,
        node: MikadoNode,
        target_branch: str | None = None,
    ) -> RebaseResult:
        try:
            return self._wt.rebase_and_merge(
                worktree,
                feature_branch,
                node.id,
                node.description,
                worker_branch=node.branch_name,
                target_branch=target_branch,
            )
        except RebaseAbortError:
            raise
        except GitOperationError as exc:
            _logger.error("Rebase-merge failed for node %d", node.id, exc_info=True)
            return RebaseResult(success=False, detail=f"{type(exc).__name__}: {exc}")
        except Exception as exc:
            _logger.error(
                "Rebase-merge failed for node %d: %s: %s",
                node.id,
                type(exc).__name__,
                exc,
                exc_info=True,
            )
            return RebaseResult(success=False, detail=f"{type(exc).__name__}: {exc}")

    def _finalize_completion(
        self, node: MikadoNode, rebase_result: RebaseResult
    ) -> CompletionResult:
        node_id = node.id
        conflict: RebaseConflict | None = None
        failure_reason = rebase_result.detail or (
            "merge-back did not integrate the worker deliverable"
        )
        if rebase_result.success:
            wrote = self._mark_terminal(node, NodeStatus.DONE)
            if wrote and node.dispatched_at is not None:
                completed_now = datetime.now(UTC)
                duration = (completed_now - node.dispatched_at).total_seconds()
                self._graph.record_completion_duration(node_id, duration)
        else:
            self._mark_terminal(node, NodeStatus.FAILED, preserve_recovery=True)
            if rebase_result.conflicting_files or rebase_result.detail:
                conflict = RebaseConflict(
                    node_id=node_id,
                    description=node.description,
                    conflicting_files=rebase_result.conflicting_files,
                    detail=failure_reason,
                )

        self._finish_node_worker_run(
            node_id,
            RunResult(
                status="done" if rebase_result.success else "failed",
                exit_code=0 if rebase_result.success else None,
                timed_out=False,
                ended_at=datetime.now(UTC).isoformat(),
                error=None if rebase_result.success else failure_reason,
                rebased=rebase_result.success,
            ),
        )

        newly_ready = get_dispatchable_nodes(self._graph) if rebase_result.success else []
        return CompletionResult(
            node_id=node_id,
            rebased=rebase_result.success,
            newly_ready=newly_ready,
            rebase_conflict=conflict,
        )

    def _handle_review_rejection(
        self,
        node: MikadoNode,
        worktree: Path,
        config: ExecutionConfig,
        findings: str,
        review_error: bool,
    ) -> tuple[CompletionResult | None, bool]:
        """Decide the outcome of a rejected review: redispatch, block, or (returning
        None) fall through to a warn-policy merge. The bool is whether the
        review-notification write itself failed, for the caller to propagate."""
        node_id = node.id
        round_number = self._review_round_by_node.get(node_id, 0)
        review_round = round_number + 1
        notification_failed = not self._notify_review(
            node, verdict="reject", findings_md=findings, round_number=review_round
        )
        if notification_failed:
            _logger.error(
                "node_review_notification_degraded node_id=%d policy=%s",
                node_id,
                config.on_reject,
            )
        if not review_error and round_number < config.review_max_rounds:
            self._review_round_by_node[node_id] = review_round
            redispatch = self._redispatch_review_round(
                node,
                config,
                worktree,
                prior_findings=findings,
                findings_round=review_round,
            )
            _logger.warning(
                "node_review_rejected node_id=%d round=%d/%d; redispatching",
                node_id,
                round_number + 1,
                config.review_max_rounds,
            )
            return (
                CompletionResult(
                    node_id=node_id,
                    rebased=False,
                    newly_ready=[],
                    redispatch=redispatch,
                    review_notification_failed=notification_failed,
                ),
                notification_failed,
            )
        if review_error or config.on_reject == "block":
            if node.run_id:
                blocked = self._graph.mark_blocked_fenced(node.id, node.run_id)
            else:
                self._graph.mark_blocked(node.id)
                blocked = True
            if not blocked:
                raise ValueError(f"review block fence lost for node {node_id}")
            self._finish_node_worker_run(
                node_id,
                RunResult(
                    status="failed",
                    exit_code=None,
                    timed_out=False,
                    ended_at=datetime.now(UTC).isoformat(),
                    detail=f"review blocked after round {review_round}",
                ),
            )
            _logger.warning("node_review_blocked node_id=%d worktree=%s", node_id, worktree)
            return (
                CompletionResult(
                    node_id=node_id,
                    rebased=False,
                    newly_ready=[],
                    blocked=True,
                    review_notification_failed=notification_failed,
                ),
                notification_failed,
            )
        _logger.warning(
            "node_review_rejected node_id=%d rounds_exhausted=%d; warn policy merges",
            node_id,
            config.review_max_rounds,
        )
        return None, notification_failed

    def _gate_review(
        self, node: MikadoNode, worktree: Path, config: ExecutionConfig
    ) -> tuple[CompletionResult | None, bool]:
        review_error = False
        try:
            approved, findings = self._run_review(node, worktree, config)
        except Exception as exc:
            review_error = True
            approved = False
            findings = f"reviewer failed: {type(exc).__name__}: {exc}"
            _logger.exception("adversarial review failed for node %d", node.id)
        if approved:
            return None, False
        return self._handle_review_rejection(node, worktree, config, findings, review_error)

    def complete(self, node_id: int, feature_branch: str) -> CompletionResult:
        node = self._graph.get_node(node_id)
        if node is None:
            raise ValueError(f"Node {node_id} not found")

        if node.status in (NodeStatus.DONE, NodeStatus.FAILED):
            return CompletionResult(
                node_id=node_id,
                rebased=node.status == NodeStatus.DONE,
                newly_ready=[],
                rebase_conflict=None,
            )

        if node.status != NodeStatus.RUNNING:
            raise InvalidTransition(
                node_id,
                node.status,
                NodeStatus.DONE,
                tuple(VALID_TRANSITIONS[node.status]),
            )

        self._validate_completion_target(node_id, feature_branch)
        worktree = Path(node.worktree_path) if node.worktree_path else None
        config = self._config_by_node.get(node_id)
        notification_failed = False
        if worktree is not None and config is not None and self._review_enabled(config):
            early, notification_failed = self._gate_review(node, worktree, config)
            if early is not None:
                return early

        target_branch = self._target_branch_by_node.get(node_id)
        rebase_result = self._rebase_or_fail(worktree, feature_branch, node, target_branch)
        result = self._finalize_completion(node, rebase_result)
        if notification_failed:
            return replace(result, review_notification_failed=True)
        return result

    def _mark_terminal(
        self, node: MikadoNode, status: NodeStatus, *, preserve_recovery: bool = False
    ) -> bool:
        """Write a fenced terminal state without clobbering recovered work."""
        if node.run_id is None:
            if status is NodeStatus.DONE:
                self._graph.mark_done(node.id)
            else:
                self._graph.mark_failed(node.id)
            return True
        return self._graph.mark_terminal(
            node.id, node.run_id, status, preserve_recovery=preserve_recovery
        )

    def fail(self, node_id: int, detail: str | None = None) -> None:
        self._wt.ensure_clean(node_id)
        node = self._graph.get_node(node_id)
        if node and node.worktree_path:
            wt = Path(node.worktree_path)
            if wt.exists():
                self._wt.remove(node_id, wt)
        self._graph.mark_failed(node_id)
        self._finish_node_worker_run(
            node_id,
            RunResult(
                status="failed",
                exit_code=None,
                timed_out=False,
                ended_at=datetime.now(UTC).isoformat(),
                detail=detail or "node failed",
            ),
        )

    def cancel(self, node_id: int) -> None:
        """Clean a stopped run and make its graph node schedulable next invocation."""
        node = self._graph.get_node(node_id)
        if node is None:
            raise ValueError(f"Node {node_id} not found")
        self._finish_node_worker_run(
            node_id,
            RunResult(
                status="failed",
                exit_code=-1,
                timed_out=False,
                ended_at=datetime.now(UTC).isoformat(),
                error="cancelled",
            ),
        )
        if node.worktree_path:
            worktree = Path(node.worktree_path)
            if worktree.exists():
                try:
                    self._wt.remove(node_id, worktree)
                except UnlandedWorkError as exc:
                    _logger.warning(
                        "Preserving stopped node %d worktree %s; releasing graph claim: %s",
                        node_id,
                        worktree,
                        exc,
                    )
        if node.run_id:
            self._graph.release(node_id, node.run_id)
        else:
            self._graph.mark_pending(node_id)

    def get_attempt_count(self, node_id: int) -> int:
        return self._attempts_by_node.get(node_id, 0)
