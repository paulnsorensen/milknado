from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from milknado.domains.common.protocols import CrgPort
    from milknado.domains.common.types import MikadoNode
    from milknado.domains.graph.graph import MikadoGraph


def build_planning_context(
    goal: str,
    crg: CrgPort,
    graph: MikadoGraph,
) -> str:
    resuming = len(graph.get_all_nodes()) > 0
    sections = [
        _goal_section(goal),
        _architecture_section(crg),
        _graph_section(graph),
        _instructions_section(resuming),
    ]
    return "\n\n".join(sections)


def _goal_section(goal: str) -> str:
    return f"# Goal\n\n{goal}"


def _architecture_section(crg: CrgPort) -> str:
    overview = crg.get_architecture_overview()
    formatted = json.dumps(overview, indent=2, default=str)
    return (
        "# Architecture Overview (from code-review-graph)\n\n"
        f"```json\n{formatted}\n```"
    )


def _graph_section(graph: MikadoGraph) -> str:
    nodes = graph.get_all_nodes()
    if not nodes:
        return "# Existing Graph\n\nNo existing nodes."

    lines = ["# Existing Graph\n"]
    lines.append(_progress_summary(nodes))

    for node in nodes:
        children = graph.get_children(node.id)
        files = graph.get_file_ownership(node.id)
        parts = [f"- [{node.id}] {node.description} ({node.status.value})"]
        if children:
            child_ids = ", ".join(str(c.id) for c in children)
            parts.append(f"  deps: [{child_ids}]")
        if files:
            parts.append(f"  files: {files}")
        lines.append("\n".join(parts))

    failed = [n for n in nodes if n.status.value == "failed"]
    if failed:
        lines.append("\n## Failed (need re-planning)\n")
        for node in failed:
            lines.append(f"- [{node.id}] {node.description}")

    ready = graph.get_ready_nodes()
    pending_ready = [
        n for n in ready if n.status.value == "pending"
    ]
    if pending_ready:
        lines.append("\n## Ready to Execute\n")
        for node in pending_ready:
            lines.append(f"- [{node.id}] {node.description}")

    return "\n".join(lines)


def _progress_summary(nodes: list[MikadoNode]) -> str:
    from collections import Counter
    counts = Counter(n.status.value for n in nodes)
    total = len(nodes)
    parts = [f"{total} total"]
    for status in ("done", "running", "pending", "blocked", "failed"):
        if counts[status]:
            parts.append(f"{counts[status]} {status}")
    return f"Progress: {', '.join(parts)}\n"


def _instructions_section(resuming: bool) -> str:
    add_node_usage = (
        "Use `milknado add-node` to add nodes to the graph:\n"
        "```\n"
        'milknado add-node "description" --parent <id> '
        "--files file1.py file2.py\n"
        "```"
    )

    if not resuming:
        return (
            "# Instructions\n\n"
            "Decompose the goal into a Mikado dependency graph.\n"
            "For each node, specify:\n"
            "- A short description of the work\n"
            "- Which files it will touch\n"
            "- Dependencies (which nodes must complete first)\n\n"
            f"{add_node_usage}\n\n"
            "Start with the root goal node, then add children "
            "for each prerequisite."
        )

    return (
        "# Instructions (resuming)\n\n"
        "The graph above shows prior progress. Do NOT recreate "
        "existing nodes.\n\n"
        "Review the current state and:\n"
        "- Add new child nodes for any remaining work\n"
        "- Re-plan around failed nodes if the approach needs to change\n"
        "- Ensure all pending nodes still make sense given completed work\n\n"
        f"{add_node_usage}"
    )
