from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import shutil
import subprocess
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
from milknado.domains.common.types import (
    VALID_TRANSITIONS,
    MikadoNode,
    NodeStatus,
    RebaseResult,
)
from milknado.domains.dispatch._runstate import make_run_id
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
    commit_footer: str | None = None
    agent_family: str = "claude"
    review: bool = False
    review_agent: str | None = None
    review_max_rounds: int = 0
    on_reject: str = "warn"
    session_mode: str = "fresh"


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


def get_dispatchable_nodes(graph: MikadoGraph) -> list[int]:
    ready_ids = [node.id for node in graph.get_ready_nodes()]
    active_ids = [node.id for node in graph.get_all_nodes() if node.status is NodeStatus.RUNNING]
    conflicts = graph.check_parallel_safety([*active_ids, *ready_ids])
    blocked = {
        right_id
        for left_id, right_id, paths in conflicts
        if right_id in ready_ids and (left_id in active_ids or left_id in ready_ids)
    }
    milknado_logger = logging.getLogger("milknado")
    for left_id, right_id, paths in conflicts:
        if right_id in blocked:
            milknado_logger.info(
                "Node %d blocked by Node %d on shared files: %s",
                right_id,
                left_id,
                paths,
            )
    return [node_id for node_id in ready_ids if node_id not in blocked]


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
        self._git = git
        self._target_branch_by_node: dict[int, str] = {}
        self._target_oid_by_node: dict[int, str] = {}
        self._graph = graph
        self._wt = WorktreeManager(git)
        self._ralph = ralph
        self._crg = crg
        self._attempts_by_node: dict[int, int] = {}
        self._config_by_node: dict[int, ExecutionConfig] = {}
        self._review_round_by_node: dict[int, int] = {}
        self._session_by_node: dict[int, NodeAgentSession] = {}

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
        slug = _slugify(node.description)
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
        wt_path, branch = self._resolve_worktree_path(node_id, node, config)
        try:
            # Every filesystem mutation follows the fenced graph claim. A losing
            # contender exits above and cannot clean, relocate, or create anything.
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
    ) -> str:
        context = build_node_context(node, self._graph, self._crg)
        ralph_path = self._ralph.generate_ralph_md(
            node,
            context,
            config.quality_gates,
            wt_path / "RALPH.md",
        )
        create_kwargs: dict[str, Any] = {
            "agent": config.execution_agent,
            "ralph_dir": wt_path,
            "ralph_file": ralph_path,
            "commands": [],
            "quality_gates": config.quality_gates,
            "project_root": wt_path,
            "commit_footer": config.commit_footer,
            "base_oid": base_oid,
        }
        if session is not None:
            create_kwargs["runtime_policy"] = RuntimePolicy(session=session)
        run = self._ralph.create_run(**create_kwargs)
        run_id = run.state.run_id
        self._ralph.start_run(run_id)
        return run_id

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
        slug = _slugify(node.description) or str(node.id)
        path = worktree / ".cheese" / "age" / f"{slug}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(findings_md.rstrip() + "\n", encoding="utf-8")

    def _notify_review(
        self,
        node: MikadoNode,
        *,
        verdict: str,
        findings_md: str,
    ) -> bool:
        worker_run_id = self._worker_run_id_by_node.get(node.id) or node.run_id
        if not worker_run_id:
            _logger.error("node_review_notification_missing_run node_id=%d", node.id)
            return False
        body = json.dumps(
            {"node_id": node.id, "verdict": verdict, "findings": findings_md},
            sort_keys=True,
        )
        try:
            self._graph.deposit_run_message(
                worker_run_id,
                "node_review",
                body,
                datetime.now(UTC).isoformat(),
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
    ) -> DispatchResult:
        owner_fence = self._owner_fence_by_node.get(node.id)
        old_worker_run_id = self._worker_run_id_by_node.get(node.id) or node.run_id
        if not old_worker_run_id:
            raise ValueError(
                f"node {node.id} has no run fence/worker run id for review redispatch"
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
        )
        if owner_fence is None:
            if not self._graph.replace_run_id(node.id, old_worker_run_id, run_id):
                with contextlib.suppress(Exception):
                    self._ralph.stop_run(run_id, timeout=5.0)
                raise ValueError(f"review redispatch fence lost for node {node.id}")
        else:
            current = self._graph.get_node(node.id)
            if current is None or current.run_id != owner_fence:
                with contextlib.suppress(Exception):
                    self._ralph.stop_run(run_id, timeout=5.0)
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
    ) -> None:
        try:
            if release_claim:
                self._graph.release(node_id, owner_run_id)
        except Exception:
            _logger.exception(
                "state reset failed during cleanup node_id=%d worktree=%s",
                node_id,
                wt_path,
            )
        finally:
            if discard_worktree:
                self._wt.discard(node_id, wt_path)

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
            review_error = False
            try:
                approved, findings = self._run_review(node, worktree, config)
            except Exception as exc:
                review_error = True
                approved = False
                findings = f"reviewer failed: {type(exc).__name__}: {exc}"
                _logger.exception("adversarial review failed for node %d", node_id)
            if not approved:
                notification_failed = not self._notify_review(
                    node, verdict="reject", findings_md=findings
                )
                if notification_failed:
                    _logger.error(
                        "node_review_notification_degraded node_id=%d policy=%s",
                        node_id,
                        config.on_reject,
                    )
                round_number = self._review_round_by_node.get(node_id, 0)
                if not review_error and round_number < config.review_max_rounds:
                    self._review_round_by_node[node_id] = round_number + 1
                    redispatch = self._redispatch_review_round(node, config, worktree)
                    _logger.warning(
                        "node_review_rejected node_id=%d round=%d/%d; redispatching",
                        node_id,
                        round_number + 1,
                        config.review_max_rounds,
                    )
                    return CompletionResult(
                        node_id=node_id,
                        rebased=False,
                        newly_ready=[],
                        redispatch=redispatch,
                        review_notification_failed=notification_failed,
                    )
                if notification_failed and config.on_reject == "block":
                    return CompletionResult(
                        node_id=node_id,
                        rebased=False,
                        newly_ready=[],
                        review_notification_failed=True,
                    )
                if review_error or config.on_reject == "block":
                    if node.run_id:
                        blocked = self._graph.mark_blocked_fenced(node.id, node.run_id)
                    else:
                        self._graph.mark_blocked(node.id)
                        blocked = True
                    if not blocked:
                        raise ValueError(f"review block fence lost for node {node_id}")
                    _logger.warning(
                        "node_review_blocked node_id=%d worktree=%s", node_id, worktree
                    )
                    return CompletionResult(
                        node_id=node_id,
                        rebased=False,
                        newly_ready=[],
                        blocked=True,
                        review_notification_failed=notification_failed,
                    )
                _logger.warning(
                    "node_review_rejected node_id=%d rounds_exhausted=%d; warn policy merges",
                    node_id,
                    config.review_max_rounds,
                )

        target_branch = self._target_branch_by_node.get(node_id)
        rebase_result = self._rebase_or_fail(worktree, feature_branch, node, target_branch)
        result = self._finalize_completion(node, rebase_result)
        if notification_failed:
            return replace(result, review_notification_failed=True)
        return result

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

    def get_attempt_count(self, node_id: int) -> int:
        return self._attempts_by_node.get(node_id, 0)
