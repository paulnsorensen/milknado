from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from milknado.domains.common import MikadoNode, NodeStatus

if TYPE_CHECKING:
    from rich.console import Console
    from rich.tree import Tree

    from milknado.domains.graph.graph import MikadoGraph

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
    active_worktrees: list[MikadoNode]

    @property
    def pct_complete(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.done / self.total) * 100


def summarize(graph: MikadoGraph, *, include_archived: bool = False) -> GraphSummary:
    nodes = graph.get_all_nodes(include_archived=include_archived)
    ready = graph.get_ready_nodes(include_archived=include_archived)
    ready_ids = [n.id for n in ready]
    conflicts = graph.check_parallel_safety(ready_ids)
    active = [n for n in nodes if n.status == NodeStatus.RUNNING and n.worktree_path]

    return GraphSummary(
        total=len(nodes),
        done=sum(1 for n in nodes if n.status == NodeStatus.DONE),
        running=sum(1 for n in nodes if n.status == NodeStatus.RUNNING),
        failed=sum(1 for n in nodes if n.status == NodeStatus.FAILED),
        blocked=sum(1 for n in nodes if n.status == NodeStatus.BLOCKED),
        ready=ready,
        conflicts=conflicts,
        active_worktrees=active,
    )


def format_node(node: MikadoNode) -> str:
    icon = STATUS_ICONS[node.status]
    color = STATUS_COLORS[node.status]
    label = f"[{color}]{icon} [{node.id}] {node.description}[/{color}]"
    if node.status == NodeStatus.RUNNING and node.worktree_path:
        label += f" [dim]({node.worktree_path})[/dim]"
    if node.archived_at is not None:
        # Escaped so rich renders the literal marker instead of parsing a tag.
        label += " [dim]\\[archived][/dim]"
    return label


def render_tree(
    graph: MikadoGraph,
    run_states: dict[str, str] | None = None,
    *,
    include_archived: bool = False,
) -> str:
    from rich.console import Console
    from rich.tree import Tree

    roots = graph.get_roots(include_archived=include_archived)
    if not roots:
        return "[dim]No nodes in graph[/dim]"
    console = Console(record=True, width=120)
    for root_node in roots:
        tree = Tree(format_node(root_node))
        _build_subtree(graph, root_node.id, tree, include_archived=include_archived)
        console.print(tree)
    console.print()
    _print_summary(console, summarize(graph, include_archived=include_archived), run_states)
    return console.export_text()


def _build_subtree(
    graph: MikadoGraph, node_id: int, tree: Tree, *, include_archived: bool = False
) -> None:
    children = graph.get_children(node_id, include_archived=include_archived)
    for child in children:
        branch = tree.add(format_node(child))
        _build_subtree(graph, child.id, branch, include_archived=include_archived)


def _print_summary(
    console: Console,
    summary: GraphSummary,
    run_states: dict[str, str] | None = None,
) -> None:
    pct = summary.pct_complete
    console.print(
        f"[bold]Progress:[/bold] {summary.done}/{summary.total} "
        + f"({pct:.0f}%) — "
        + f"[cyan]{summary.running} running[/cyan], "
        + f"[red]{summary.failed} failed[/red], "
        + f"[yellow]{summary.blocked} blocked[/yellow]"
    )

    if summary.active_worktrees:
        console.print(f"[bold]Active Worktrees ({len(summary.active_worktrees)}):[/bold]")
        for node in summary.active_worktrees:
            run_status = _run_status_label(node.run_id, run_states)
            console.print(
                f"  [cyan]◉[/cyan] [{node.id}] {node.description}"
                + f" [dim]({node.worktree_path})[/dim]"
                + f"{run_status}"
            )

    if summary.ready:
        names = ", ".join(f"[{n.id}] {n.description}" for n in summary.ready)
        console.print(f"[bold]Ready:[/bold] {names}")

    if summary.conflicts:
        console.print("[bold red]Conflicts:[/bold red]")
        for a, b, files in summary.conflicts:
            console.print(f"  Nodes {a} ↔ {b}: {', '.join(files)}")


def _run_status_label(
    run_id: str | None,
    run_states: dict[str, str] | None,
) -> str:
    if not run_id:
        return ""
    if not run_states:
        return " [dim]run: unknown[/dim]"
    status = run_states.get(run_id)
    if not status:
        return " [dim]run: unknown[/dim]"
    color = {"running": "cyan", "completed": "green", "failed": "red"}.get(
        status,
        "dim",
    )
    return f" [{color}]run: {status}[/{color}]"
