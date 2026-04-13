from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from milknado.domains.common import MikadoNode, NodeStatus
from milknado.domains.graph.graph import MikadoGraph

if TYPE_CHECKING:
    from rich.console import Console

STATUS_COLORS: dict[NodeStatus, str] = {
    NodeStatus.PENDING: "dim",
    NodeStatus.RUNNING: "cyan",
    NodeStatus.DONE: "green",
    NodeStatus.BLOCKED: "yellow",
    NodeStatus.FAILED: "red",
}

STATUS_ICONS: dict[NodeStatus, str] = {
    NodeStatus.PENDING: "○",
    NodeStatus.RUNNING: "◉",
    NodeStatus.DONE: "✓",
    NodeStatus.BLOCKED: "⊘",
    NodeStatus.FAILED: "✗",
}


@dataclass(frozen=True)
class GraphSummary:
    total: int
    done: int
    running: int
    failed: int
    blocked: int
    ready: list[MikadoNode]
    conflicts: list[tuple[int, int, list[str]]]

    @property
    def pct_complete(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.done / self.total) * 100


def summarize(graph: MikadoGraph) -> GraphSummary:
    nodes = graph.get_all_nodes()
    ready = graph.get_ready_nodes()
    ready_pending = [n for n in ready if n.status == NodeStatus.PENDING]
    ready_ids = [n.id for n in ready_pending]
    conflicts = graph.check_parallel_safety(ready_ids)

    return GraphSummary(
        total=len(nodes),
        done=sum(1 for n in nodes if n.status == NodeStatus.DONE),
        running=sum(1 for n in nodes if n.status == NodeStatus.RUNNING),
        failed=sum(1 for n in nodes if n.status == NodeStatus.FAILED),
        blocked=sum(1 for n in nodes if n.status == NodeStatus.BLOCKED),
        ready=ready_pending,
        conflicts=conflicts,
    )


def format_node(node: MikadoNode) -> str:
    icon = STATUS_ICONS[node.status]
    color = STATUS_COLORS[node.status]
    label = f"[{color}]{icon} [{node.id}] {node.description}[/{color}]"
    if node.status == NodeStatus.RUNNING and node.worktree_path:
        label += f" [dim]({node.worktree_path})[/dim]"
    return label


def render_tree(graph: MikadoGraph) -> str:
    from rich.console import Console
    from rich.tree import Tree

    root_node = graph.get_root()
    if root_node is None:
        return "[dim]No nodes in graph[/dim]"

    summary = summarize(graph)
    tree = Tree(format_node(root_node))
    _build_subtree(graph, root_node.id, tree)

    console = Console(record=True, width=120)
    console.print(tree)
    console.print()
    _print_summary(console, summary)
    return console.export_text()


def _build_subtree(
    graph: MikadoGraph, node_id: int, tree: Any
) -> None:
    for child in graph.get_children(node_id):
        branch = tree.add(format_node(child))
        _build_subtree(graph, child.id, branch)


def _print_summary(console: Console, summary: GraphSummary) -> None:
    pct = summary.pct_complete
    console.print(
        f"[bold]Progress:[/bold] {summary.done}/{summary.total} "
        f"({pct:.0f}%) — "
        f"[cyan]{summary.running} running[/cyan], "
        f"[red]{summary.failed} failed[/red], "
        f"[yellow]{summary.blocked} blocked[/yellow]"
    )

    if summary.ready:
        names = ", ".join(
            f"[{n.id}] {n.description}" for n in summary.ready
        )
        console.print(f"[bold]Ready:[/bold] {names}")

    if summary.conflicts:
        console.print("[bold red]Conflicts:[/bold red]")
        for a, b, files in summary.conflicts:
            console.print(f"  Nodes {a} ↔ {b}: {', '.join(files)}")
