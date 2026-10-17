"""Node validation and quality-gate check shared by the inline dispatch entry points."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from milknado.domains.common import (
    FlavorProfile,
    GitOperationError,
    GitPort,
    MikadoNode,
    NodeKind,
    WorktreeMode,
)
from milknado.domains.dispatch import CompletionCheck
from milknado.domains.execution import build_completion_verifier

if TYPE_CHECKING:
    from milknado.domains.graph import MikadoGraph


def require_task_node(graph: MikadoGraph, node_id: int) -> MikadoNode:
    node = graph.get_node(node_id)
    if node is None:
        raise ValueError(f"node {node_id} not found")
    if node.kind != NodeKind.TASK:
        raise ValueError(
            f"node {node_id} has kind={node.kind.value}; only task nodes can be dispatched"
        )
    return node


def _checkout_head(git: GitPort) -> str | None:
    try:
        return git.resolve_ref("HEAD")
    except GitOperationError:
        return None


def inline_completion_check(
    profile: FlavorProfile, node: MikadoNode, git: GitPort, mode: WorktreeMode
) -> CompletionCheck:
    """Build the loop path's completion verifier for an inline-dispatched node.

    An isolated worker is judged against the base commit captured at dispatch.
    A worker in the shared checkout is judged against the checkout HEAD now.
    """
    in_place = mode != WorktreeMode.ISOLATE
    shared_base = _checkout_head(git) if in_place else None

    def check(worktree: Path, base_oid: str | None) -> str | None:
        verdict = build_completion_verifier(
            worktree,
            profile.quality_gates,
            base_oid=base_oid or shared_base,
            in_place=in_place,
            artifact_path=node.artifact_path,
        )()
        return None if verdict.ok else verdict.feedback

    return check
