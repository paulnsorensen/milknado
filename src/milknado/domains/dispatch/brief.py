"""Render a markdown brief for a task node, derived from the graph."""

from __future__ import annotations

from milknado.domains.common import MikadoNode, NodeKind, NodeStatus
from milknado.domains.graph import MikadoGraph, walk_ancestors


def _done_prereqs(graph: MikadoGraph, node: MikadoNode) -> list[MikadoNode]:
    if node.parent_id is None:
        return []
    siblings = graph.get_children(node.parent_id)
    return [s for s in siblings if s.id != node.id and s.status == NodeStatus.DONE]


def _format_goal_context(chain: list[MikadoNode]) -> list[str]:
    if not chain:
        return ["(no parent goal)"]
    return [
        f"- [{a.kind.value} #{a.id}] {a.description}"
        for a in chain
        if a.kind in (NodeKind.ROADMAP, NodeKind.GOAL)
    ] or ["(no parent goal)"]


def render_brief(graph: MikadoGraph, node_id: int, *, prepend: str | None = None) -> str:
    node = graph.get_node(node_id)
    if node is None:
        raise ValueError(f"node {node_id} not found")

    # walk_ancestors returns [node, parent, ..., root]; drop node itself, reverse to root-first
    ancestors = walk_ancestors(graph, node_id)
    chain = list(reversed(ancestors[1:]))
    done = _done_prereqs(graph, node)
    files = graph.get_file_ownership(node_id)

    lines = [f"# Task: {node.description}", "", "## Goal context"]
    lines.extend(_format_goal_context(chain))
    lines.append("")

    lines.append("## Prerequisites already done")
    if done:
        lines.extend(f"- [#{d.id}] {d.description}" for d in done)
    else:
        lines.append("(none)")
    lines.append("")

    lines.append("## Relevant files")
    if files:
        lines.extend(f"- {f}" for f in files)
    else:
        lines.append("(no file hints registered)")
    lines.append("")

    lines.append("## Instructions")
    lines.append(
        "Complete the task above. Touch only files listed under "
        "'Relevant files' unless others are clearly needed. "
        "Report blockers in stdout if you cannot proceed. "
        "If you discover follow-up work, register it by calling "
        "milknado_track_follow_up with a one-line description rather than only "
        "printing it. "
        "As your final step, call milknado_deposit_result with run_id set to the "
        "MILKNADO_RUN_ID environment variable and payload set to your COMPLETE "
        "deliverable — the full text of what you produced, not a reference to "
        "content that lives only in this context. The deposited payload is what "
        "the coordinator reads back; anything left only in your reply is lost."
    )
    body = "\n".join(lines) + "\n"
    if prepend:
        return prepend.rstrip() + "\n\n" + body
    return body
