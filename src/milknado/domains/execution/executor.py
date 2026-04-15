from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from milknado.domains.common.types import NodeStatus

if TYPE_CHECKING:
    from milknado.domains.common.protocols import CrgPort, GitPort, RalphPort
    from milknado.domains.graph.graph import MikadoGraph


@dataclass(frozen=True)
class ExecutionConfig:
    agent_command: str
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


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:30]


def get_dispatchable_nodes(graph: MikadoGraph) -> list[int]:
    ready = graph.get_ready_nodes()
    pending = [n for n in ready if n.status == NodeStatus.PENDING]
    ids = [n.id for n in pending]
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

            context = _build_node_context(
                node.description,
                self._graph.get_file_ownership(node_id),
                self._crg,
            )
            ralph_path = self._ralph.generate_ralph_md(
                node, context, list(config.quality_gates),
                wt_path / "RALPH.md",
            )

            run = self._ralph.create_run(
                agent=config.agent_command,
                ralph_dir=wt_path,
                ralph_file=ralph_path,
                commands=[],
                quality_gates=list(config.quality_gates),
            )
            run_id = run.state.run_id
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
            rebased = self._rebase_and_merge(worktree, feature_branch)
        except Exception:
            rebased = False

        if rebased:
            self._graph.mark_done(node_id)
        else:
            self._graph.mark_failed(node_id)

        newly_ready = get_dispatchable_nodes(self._graph) if rebased else []
        return CompletionResult(
            node_id=node_id, rebased=rebased, newly_ready=newly_ready,
        )

    def _rebase_and_merge(
        self,
        worktree: Path | None,
        feature_branch: str,
    ) -> bool:
        if not worktree or not worktree.exists():
            return True
        try:
            self._git.commit_all(worktree, "feat(milknado): complete node")
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
    description: str,
    files: list[str],
    crg: CrgPort,
) -> str:
    sections = [f"# Task\n\n{description}"]
    if files:
        impact = crg.get_impact_radius(files)
        sections.append(f"# Impact Radius\n\n{impact}")
        file_list = "\n".join(f"- `{f}`" for f in files)
        sections.append(f"# Owned Files\n\n{file_list}")
    return "\n\n".join(sections)
