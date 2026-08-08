"""Render a markdown brief for a task node, derived from the graph."""

from __future__ import annotations

import os
from pathlib import Path

from milknado.domains.common import GraphReadPort, MikadoNode, NodeKind, NodeStatus
from milknado.domains.graph import walk_ancestors


def _done_prereqs(graph: GraphReadPort, node: MikadoNode) -> list[MikadoNode]:
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


def _resolve_spec_path(
    node: MikadoNode,
    chain: list[MikadoNode],
    project_root: Path | None = None,
) -> str | None:
    """Return the nearest durable spec as an absolute path when possible."""
    raw: str | None = node.artifact_path
    if raw is None:
        for ancestor in reversed(chain):
            if ancestor.artifact_path:
                raw = ancestor.artifact_path
                break
    if raw is None:
        return None
    path = Path(raw).expanduser()
    if project_root is None or path.is_absolute():
        return str(path.resolve()) if path.is_absolute() else str(path)

    root = project_root.resolve()
    data_home = Path(os.environ.get("XDG_DATA_HOME", "~/.local/share")).expanduser()
    project_name = root.name
    candidates: list[Path] = []
    if path.parts[:2] == (".cheese", "specs"):
        candidates.append(root / path)
    else:
        candidates.extend(
            [
                data_home / "cheese" / project_name / "specs" / path,
                root / path,
                root / ".cheese" / "specs" / path,
            ]
        )
    unique = list(dict.fromkeys(candidates))
    for candidate in unique:
        if candidate.exists():
            return str(candidate.resolve())
    return str(unique[0].resolve())


def _brief_header(
    title_prefix: str,
    node: MikadoNode,
    chain: list[MikadoNode],
    done: list[MikadoNode],
    done_label: str,
) -> list[str]:
    lines = [f"# {title_prefix}: {node.description}", "", "## Goal context"]
    lines.extend(_format_goal_context(chain))
    lines.append("")
    lines.append(f"## {done_label}")
    if done:
        lines.extend(f"- [#{d.id}] {d.description}" for d in done)
    else:
        lines.append("(none)")
    lines.append("")
    return lines


def _finish_brief(lines: list[str], instructions: str, prepend: str | None) -> str:
    lines.append("## Instructions")
    lines.append(instructions)
    body = "\n".join(lines) + "\n"
    if prepend:
        return prepend.rstrip() + "\n\n" + body
    return body


_CODER_INSTRUCTIONS = (
    "Complete the task above. Touch only files listed under "
    "'Relevant files' unless others are clearly needed. "
    "Report blockers in stdout if you cannot proceed. "
    "If you discover follow-up work, register it by calling "
    "milknado_track_follow_up with a one-line description rather than only "
    "printing it. "
    "As your final step, call milknado_deposit_result with run_id set to the "
    "MILKNADO_RUN_ID environment variable. If the task description defines a "
    "structured result contract, payload must be that exact structured result; "
    "otherwise payload must be your COMPLETE deliverable — the full text of what "
    "you produced, not a reference to content that lives only in this context. "
    "The deposited payload is what the coordinator reads back; anything left "
    "only in your reply is lost."
)

_REVIEW_INSTRUCTIONS = (
    "Review the diff produced by the work above against its brief and any "
    "referenced spec. Assess correctness, security, encapsulation, spec "
    "alignment, and complexity — do not merely confirm it compiles. Produce a "
    "severity-grouped findings report (blocker/high/medium/low), one bullet "
    "per finding with evidence and a fix recommendation. "
    "As your final step, call milknado_deposit_result with run_id set to the "
    "MILKNADO_RUN_ID environment variable and payload set to your COMPLETE "
    "findings report — the full markdown, not a reference to content that "
    "lives only in this context. Then call milknado_deposit_review with the same "
    "run_id, verdict exactly 'approve' or 'reject', and the same findings markdown."
)

_PLATE_INSTRUCTIONS = (
    "This is a terminal plating task. Stage and commit the completed work with "
    "a Conventional Commits message, then open or update a pull request using "
    "the `gh` CLI. `gh` is authenticated in this environment without "
    "sandboxing — do not run `gh` through a sandboxed or offline path. "
    "As your final step, call milknado_deposit_result with run_id set to the "
    "MILKNADO_RUN_ID environment variable and payload set to the PR URL (or "
    "commit SHA if no PR was opened) plus a summary of what shipped."
)


def render_brief(
    graph: GraphReadPort,
    node_id: int,
    *,
    prepend: str | None = None,
    project_root: Path | None = None,
) -> str:
    node = graph.get_node(node_id)
    if node is None:
        raise ValueError(f"node {node_id} not found")

    # walk_ancestors returns [node, parent, ..., root]; drop node itself, reverse to root-first
    ancestors = walk_ancestors(graph, node_id)
    chain = list(reversed(ancestors[1:]))
    done = _done_prereqs(graph, node)

    if node.flavor == "review":
        lines = _brief_header("Review", node, chain, done, "Work under review")
        return _finish_brief(lines, _REVIEW_INSTRUCTIONS, prepend)
    if node.flavor == "plate":
        lines = _brief_header("Plate", node, chain, done, "Work to publish")
        return _finish_brief(lines, _PLATE_INSTRUCTIONS, prepend)

    files = graph.get_file_ownership(node_id)
    lines = _brief_header("Task", node, chain, done, "Prerequisites already done")

    lines.append("## Relevant files")
    if files:
        lines.extend(f"- {f}" for f in files)
    else:
        lines.append("(no file hints registered)")
    lines.append("")

    spec_path = _resolve_spec_path(node, chain, project_root)
    if spec_path:
        lines.append("## Spec")
        lines.append(f"- {spec_path}")
        lines.append("")

    return _finish_brief(lines, _CODER_INSTRUCTIONS, prepend)
