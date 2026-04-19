from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from milknado.domains.common.types import RebaseResult
from milknado.domains.graph.traversals import walk_ancestors

if TYPE_CHECKING:
    from milknado.domains.common.protocols import CrgPort, GitPort, RalphPort
    from milknado.domains.common.types import MikadoNode
    from milknado.domains.graph import MikadoGraph


@dataclass(frozen=True)
class ExecutionConfig:
    execution_agent: str
    quality_gates: tuple[str, ...]
    worktree_pattern: str
    project_root: Path


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


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:30]


def get_dispatchable_nodes(graph: MikadoGraph) -> list[int]:
    ready = graph.get_ready_nodes()
    ids = [n.id for n in ready]
    conflicts = graph.check_parallel_safety(ids)
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

    def dispatch(
        self, node_id: int, config: ExecutionConfig
    ) -> DispatchResult:
        node = self._graph.get_node(node_id)
        if node is None:
            raise ValueError(f"Node {node_id} not found")

        slug = _slugify(node.description)
        worktree_name = config.worktree_pattern.format(
            node_id=node_id, slug=slug,
        )
        wt_path = config.project_root / worktree_name
        branch = f"milknado/{node_id}-{slug}"

        self._git.create_worktree(wt_path, branch)
        try:
            self._graph.mark_running(
                node_id, worktree_path=str(wt_path), branch_name=branch,
            )

            context = _build_node_context(node, self._graph, self._crg)
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
            )
            run_id = run.state.run_id
            self._graph.set_run_id(node_id, run_id)
            self._ralph.start_run(run_id)
        except Exception:
            self._graph.mark_pending(node_id)
            self._git.remove_worktree(wt_path)
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

        conflict: RebaseConflict | None = None
        if rebase_result.success:
            self._graph.mark_done(node_id)
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
            msg = f"feat(milknado-{node_id}): {description}"
            self._git.commit_all(worktree, msg)
            return self._git.rebase(worktree, feature_branch)
        finally:
            self._git.remove_worktree(worktree)

    def fail(self, node_id: int) -> None:
        node = self._graph.get_node(node_id)
        if node and node.worktree_path:
            wt = Path(node.worktree_path)
            if wt.exists():
                self._git.remove_worktree(wt)
        self._graph.mark_failed(node_id)


def _build_node_context(
    node: MikadoNode,
    graph: MikadoGraph,
    crg: CrgPort | None,
) -> str:
    """Build executor context by walking the ancestor chain.

    Single-node graph (node == root, no ancestors): omits the "Why chain"
    section; Goal and Your task both show the same node description.

    crg=None: Impact Radius section shows a degradation fallback line.
    """
    ancestors = walk_ancestors(graph, node.id)
    # ancestors[0] == node, ancestors[-1] == root
    root = ancestors[-1]
    sections = [f"## Goal\n\n{root.description}"]

    # Emit Why chain only when there are intermediate ancestors between node and root.
    # Single node (node is root) or direct child of root → no Why chain.
    why_nodes = ancestors[1:-1]  # between node and root, exclusive
    if why_nodes:
        why_parts = "\n".join(f"### {n.description}" for n in why_nodes)
        sections.append(f"## Why chain (parent → grandparent → ...)\n\n{why_parts}")

    sections.append(f"## Your task\n\n{node.description}")

    files = graph.get_file_ownership(node.id)
    if files:
        file_list = "\n".join(f"- `{f}`" for f in files)
        sections.append(f"## Files\n\n{file_list}")
    else:
        sections.append("## Files\n\n_(no files assigned)_")

    impact_section = _impact_radius_section(crg, files)
    sections.append(impact_section)

    return "\n\n".join(sections)


def _impact_radius_section(crg: CrgPort | None, files: list[str]) -> str:
    if crg is None:
        return "## Impact Radius\n\n_(CRG unavailable — impact radius skipped)_"
    if not files:
        return "## Impact Radius\n\n_(no files — impact radius skipped)_"
    impact = crg.get_impact_radius(files)
    return f"## Impact Radius\n\n{impact}"
