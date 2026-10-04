"""Render a markdown brief for a task node, derived from the graph."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from milknado.domains.common import (
    GitOperationError,
    GitPort,
    GraphReadPort,
    MikadoNode,
    NodeKind,
    NodeStatus,
)
from milknado.domains.graph import walk_ancestors


@dataclass(frozen=True)
class WorkerOrientation:
    """Facts a worker needs at start, so it never has to probe its environment."""

    run_id: str
    worktree: Path
    branch: str | None


_ORIENTATION_NOTE = (
    "These values are final. Do not run pwd, ls, git status or echo of environment "
    "variables to learn them."
)


def current_branch_or_none(git: GitPort) -> str | None:
    """Return the checked-out branch, or None when the root is not a git checkout."""
    try:
        return git.current_branch()
    except GitOperationError:
        return None


def isolated_orientation(
    graph: GraphReadPort, git: GitPort, node_id: int, worker: tuple[str, Path]
) -> WorkerOrientation:
    """Orient a worker from `(run_id, cwd)`; a node whose worktree is cwd has its own branch."""
    run_id, cwd = worker
    node = graph.get_node(node_id)
    in_own_worktree = node is not None and node.worktree_path == str(cwd)
    branch = node.branch_name if node is not None and in_own_worktree else None
    return WorkerOrientation(run_id, cwd, branch or current_branch_or_none(git))


def _orientation_lines(node_id: int, orientation: WorkerOrientation) -> list[str]:
    return [
        "## Orientation",
        f"- run_id: {orientation.run_id}",
        f"- node_id: {node_id}",
        f"- worktree: {orientation.worktree}",
        f"- branch: {orientation.branch or '(unknown)'}",
        _ORIENTATION_NOTE,
        "",
    ]


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
    labels: tuple[str, str],
    node: MikadoNode,
    context: tuple[list[MikadoNode], list[MikadoNode]],
    orientation: WorkerOrientation | None,
) -> list[str]:
    """Render the heading; `labels` is (title_prefix, done_label), `context` is (chain, done)."""
    (title_prefix, done_label), (chain, done) = labels, context
    lines = [f"# {title_prefix}: {node.description}", ""]
    if orientation is not None:
        lines.extend(_orientation_lines(node.id, orientation))
    lines.append("## Goal context")
    lines.extend(_format_goal_context(chain))
    lines.append("")
    lines.append(f"## {done_label}")
    if done:
        lines.extend(f"- [#{d.id}] {d.description}" for d in done)
    else:
        lines.append("(none)")
    lines.append("")
    return lines


_RUN_ID_SOURCE = "@RUN_ID_SOURCE@"


def _finish_brief(
    lines: list[str], instructions: str, prepend: str | None, orientation: WorkerOrientation | None
) -> str:
    run_id_source = (
        "the MILKNADO_RUN_ID environment variable"
        if orientation is None
        else "the run_id stated under Orientation"
    )
    lines.append("## Instructions")
    lines.append(instructions.replace(_RUN_ID_SOURCE, run_id_source))
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
    "As your final step, call milknado_deposit_result with run_id set to "
    f"{_RUN_ID_SOURCE} and payload set to your COMPLETE "
    "deliverable — the full text of what you produced, not a reference to "
    "content that lives only in this context. The deposited payload is what "
    "the coordinator reads back; anything left only in your reply is lost."
)

_REVIEW_INSTRUCTIONS = (
    "Review the diff produced by the work above against its brief and any "
    "referenced spec. Assess correctness, security, encapsulation, spec "
    "alignment, and complexity — do not merely confirm it compiles. Produce a "
    "severity-grouped findings report (blocker/high/medium/low), one bullet "
    "per finding with evidence and a fix recommendation. "
    "As your final step, call milknado_deposit_result with run_id set to "
    f"{_RUN_ID_SOURCE} and payload set to your COMPLETE "
    "findings report — the full markdown, not a reference to content that "
    "lives only in this context. Then call milknado_deposit_review with the same "
    "run_id, verdict exactly 'approve' or 'reject', and the same findings markdown."
)

_PLATE_INSTRUCTIONS = (
    "This is a terminal plating task. Stage and commit the completed work with "
    "a Conventional Commits message, then open or update a pull request using "
    "the `gh` CLI. `gh` is authenticated in this environment without "
    "sandboxing — do not run `gh` through a sandboxed or offline path. "
    "As your final step, call milknado_deposit_result with run_id set to "
    f"{_RUN_ID_SOURCE} and payload set to the PR URL (or "
    "commit SHA if no PR was opened) plus a summary of what shipped."
)


def render_brief(
    graph: GraphReadPort,
    node_id: int,
    *,
    prepend: str | None = None,
    project_root: Path | None = None,
    orientation: WorkerOrientation | None = None,
) -> str:
    node = graph.get_node(node_id)
    if node is None:
        raise ValueError(f"node {node_id} not found")

    # walk_ancestors returns [node, parent, ..., root]; drop node itself, reverse to root-first
    ancestors = walk_ancestors(graph, node_id)
    chain = list(reversed(ancestors[1:]))
    done = _done_prereqs(graph, node)

    if node.flavor == "review":
        lines = _brief_header(("Review", "Work under review"), node, (chain, done), orientation)
        return _finish_brief(lines, _REVIEW_INSTRUCTIONS, prepend, orientation)
    if node.flavor == "plate":
        lines = _brief_header(("Plate", "Work to publish"), node, (chain, done), orientation)
        return _finish_brief(lines, _PLATE_INSTRUCTIONS, prepend, orientation)

    files = graph.files.for_node(node_id)
    labels = ("Task", "Prerequisites already done")
    lines = _brief_header(labels, node, (chain, done), orientation)

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

    return _finish_brief(lines, _CODER_INSTRUCTIONS, prepend, orientation)
