from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from milknado.domains.common.protocols import CrgPort
    from milknado.domains.graph.graph import MikadoGraph


def build_planning_context(
    goal: str,
    crg: CrgPort,
    graph: MikadoGraph,
) -> str:
    sections = [
        _goal_section(goal),
        _architecture_section(crg),
        _graph_section(graph),
        _instructions_section(),
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
    for node in nodes:
        lines.append(
            f"- [{node.id}] {node.description} ({node.status.value})"
        )
    return "\n".join(lines)


def _instructions_section() -> str:
    return (
        "# Instructions\n\n"
        "Decompose the goal into a Mikado dependency graph.\n"
        "For each node, specify:\n"
        "- A short description of the work\n"
        "- Which files it will touch\n"
        "- Dependencies (which nodes must complete first)\n\n"
        "Use `milknado add-node` to add nodes to the graph:\n"
        "```\n"
        'milknado add-node "description" --parent <id> '
        "--files file1.py file2.py\n"
        "```\n\n"
        "Start with the root goal node, then add children "
        "for each prerequisite."
    )
