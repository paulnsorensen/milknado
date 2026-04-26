from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from milknado.domains.common.protocols import CrgPort, TilthPort
    from milknado.domains.common.types import MikadoNode
    from milknado.domains.graph import MikadoGraph


def build_planning_context(
    goal: str,
    crg: CrgPort | None,
    graph: MikadoGraph,
    *,
    spec_text: str | None = None,
    tilth: TilthPort | None = None,
    scope: Path | None = None,
) -> str:
    if spec_text == "":
        raise ValueError("spec_text must not be empty string — pass None to omit the spec section")
    resuming = len(graph.get_all_nodes()) > 0
    sections = [
        _goal_section(goal),
        _crg_compact_section(crg),
        _structural_section(tilth, scope),
        _graph_section(graph),
        _batching_section(),
        _instructions_section(resuming),
    ]
    if spec_text is not None:
        sections.append(_spec_section(spec_text))
    return "\n\n".join(sections)


def _goal_section(goal: str) -> str:
    return f"# Goal\n\n{goal}"


def _crg_compact_section(crg: CrgPort | None) -> str:
    header = "# Architecture (compact)"
    if crg is None:
        return f"{header}\n\n_(CRG unavailable — architecture overview skipped)_"
    try:
        communities = crg.list_communities()[:5]
        flows = crg.list_flows()[:3]
        bridges = crg.get_bridge_nodes(top_n=5)[:5]
        hubs = crg.get_hub_nodes(top_n=5)[:5]
    except Exception as exc:
        return f"{header}\n\n_(CRG query failed: {exc!s} — architecture overview skipped)_"

    lines = [header]
    lines.append("\n**Top communities:**")
    for c in communities:
        name = c.get("name", c.get("id", str(c)))
        lines.append(f"- {name}")

    lines.append("\n**Top flows:**")
    for f in flows:
        name = f.get("name", f.get("id", str(f)))
        lines.append(f"- {name}")

    lines.append("\n**Bridge nodes:**")
    for b in bridges:
        name = b.get("name", b.get("id", str(b)))
        lines.append(f"- {name}")

    lines.append("\n**Hub nodes:**")
    for h in hubs:
        name = h.get("name", h.get("id", str(h)))
        lines.append(f"- {name}")

    return "\n".join(lines)


def _structural_section(tilth: TilthPort | None, scope: Path | None) -> str:
    from milknado.domains.common.types import DegradationMarker, TilthMap

    header = "# Structural Map (tilth)"
    if tilth is None:
        return f"{header}\n\n_(tilth not available — structural map skipped)_"
    result = tilth.structural_map(scope or Path("."), 2000)
    if isinstance(result, DegradationMarker):
        return (
            f"{header}\n\n"
            f"_(tilth unavailable: {result.source} — {result.reason}. "
            "Structural analysis skipped.)_"
        )
    assert isinstance(result, TilthMap)
    data_lines = []
    for key, value in result.data.items():
        data_lines.append(f"- **{key}**: {value}")
    body = "\n".join(data_lines) if data_lines else "_(no structural data returned)_"
    return f"{header}\n\n{body}"


def _spec_section(spec_text: str) -> str:
    return f"# Spec\n\n{spec_text}"


def _batching_section() -> str:
    return (
        "# Batching\n\n"
        "The host runs a token-budgeted OR-tools solver over your manifest to group "
        "file-level changes into executor batches. Emit file-level changes and let the "
        "solver batch them — more granular changes produce better batch boundaries. "
        "Do not pre-group changes yourself."
    )


def _mcp_targeting_note() -> str:
    return (
        "**Required MCP inspection before you emit changes:**\n"
        "- Use `tilth` to inspect the target area and capture exact edit boundaries.\n"
        "- Use `tilth` or `code-review-graph` to discover impacted dependencies around that "
        "area.\n"
        "- For every change, include the target file, symbols, and `hash_anchors` with "
        '`"before"` / `"after"` values that bound the intended edit span.\n'
        "- Add dependency entries in the same `file + symbols + hash_anchors` format for any "
        "adjacent code that constrains the change.\n"
        "- Do not invent anchors or dependencies: only emit data you observed via MCP queries."
    )


def _graph_section(graph: MikadoGraph) -> str:
    nodes = graph.get_all_nodes()
    if not nodes:
        return "# Existing Graph\n\nNo existing nodes."

    lines = ["# Existing Graph\n"]
    lines.append(_progress_summary(nodes))

    for node in nodes:
        desc = _truncate_description(node.description)
        children = graph.get_children(node.id)
        files = graph.get_file_ownership(node.id)
        parts = [f"- [{node.id}] {desc} ({node.status.value})"]
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
            lines.append(f"- [{node.id}] {_truncate_description(node.description)}")

    ready = graph.get_ready_nodes()
    if ready:
        lines.append("\n## Ready to Execute\n")
        for node in ready:
            lines.append(f"- [{node.id}] {_truncate_description(node.description)}")

    return "\n".join(lines)


def _truncate_description(description: str) -> str:
    if not description:
        return description
    lines = description.splitlines()
    first_line = lines[0]
    has_more_content = any(line.strip() for line in lines[1:])
    if first_line and has_more_content:
        return first_line + " \u2026"
    if not first_line and has_more_content:
        return " \u2026"
    return first_line


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
    schema = (
        "```json\n"
        "{\n"
        '  "manifest_version": "milknado.plan.v2",\n'
        '  "goal": "One-line goal statement",\n'
        '  "goal_summary": "2-4 sentence summary: what / why / success criteria.'
        ' Used as the root Mikado node — every executor reads this.",\n'
        '  "spec_path": "path/to/spec.md or null",\n'
        '  "changes": [\n'
        "    {\n"
        '      "id": "c1",\n'
        '      "path": "src/foo.py",\n'
        '      "edit_kind": "modify",\n'
        '      "description": "Why this change is needed + which spec section drives it.'
        ' Causal changes explain the full rationale.",\n'
        '      "symbols": [\n'
        '        {"name": "FooClass", "file": "src/foo.py"}\n'
        "      ],\n"
        '      "hash_anchors": {\n'
        '        "before": "sha256:4c9f0e2b-target-before",\n'
        '        "after": "sha256:7b1a92d4-target-after"\n'
        "      },\n"
        '      "dependencies": [\n'
        "        {\n"
        '          "path": "src/bar.py",\n'
        '          "symbols": [{"name": "bar_call_site", "file": "src/bar.py"}],\n'
        '          "hash_anchors": {\n'
        '            "before": "sha256:8f2b3c44-dependency-before",\n'
        '            "after": "sha256:9a7d118e-dependency-after"\n'
        "          },\n"
        '          "reason": "Call site or import that constrains this change"\n'
        "        }\n"
        "      ],\n"
        '      "depends_on": []\n'
        "    },\n"
        "    {\n"
        '      "id": "c2",\n'
        '      "path": "src/bar.py",\n'
        '      "edit_kind": "add",\n'
        '      "description": "Respond to c1 signature change — update call site.",\n'
        '      "hash_anchors": {\n'
        '        "before": "sha256:12ac44e0-target-before",\n'
        '        "after": "sha256:21fd9b77-target-after"\n'
        "      },\n"
        '      "depends_on": ["c1"]\n'
        "    }\n"
        "  ],\n"
        '  "new_relationships": [\n'
        "    {\n"
        '      "source_change_id": "c1",\n'
        '      "dependant_change_id": "c2",\n'
        '      "reason": "new_import"\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "```"
    )

    enum_note = (
        "**Closed enums — do not invent values:**\n"
        '- `edit_kind`: one of `"add"`, `"modify"`, `"delete"`, `"rename"`.\n'
        '- `new_relationships[].reason`: one of `"new_file"`, `"new_import"`,'
        ' `"new_call"`, `"new_type_use"`.'
    )

    edge_note = (
        "> **Graph edges** are stored as 3-line rows: `(parent_id, child_id, relationship_type)`. "
        "The solver uses `depends_on` and `new_relationships` to build these rows automatically."
    )

    description_rules = (
        "**description field rules:**\n"
        "- Causal changes: explain *why* the change is needed,"
        " reference the spec section name if applicable.\n"
        "- Effect changes: be terse, reference the cause by id"
        " (e.g. `respond to c1 signature change`).\n"
        "- All descriptions must be non-empty — the executor has no other context."
    )

    granularity_note = (
        "Emit file-level changes. More granular is better — multiple changes can share"
        " a path when symbols differ (e.g. two classes in the same file)."
        " The solver batches them optimally."
    )

    mcp_targeting_note = _mcp_targeting_note()

    goal_summary_note = (
        "`goal_summary` is 2-4 sentences structured as **what / why / success criteria**."
        " It becomes the root Mikado node description that every executor reads."
        " Make it load-bearing."
    )

    if not resuming:
        return (
            "# Instructions\n\n"
            "Decompose the goal into a v2 change manifest.\n\n"
            f"{granularity_note}\n\n"
            f"{mcp_targeting_note}\n\n"
            f"{description_rules}\n\n"
            f"{goal_summary_note}\n\n"
            f"{edge_note}\n\n"
            f"{enum_note}\n\n"
            "Emit the manifest as a fenced ```json block (valid JSON, not YAML):\n\n"
            f"{schema}"
        )

    return (
        "# Instructions (resuming)\n\n"
        "The graph above shows prior progress. Do NOT recreate existing nodes.\n\n"
        "Review the current state and add change manifest entries for any remaining work.\n\n"
        f"{granularity_note}\n\n"
        f"{mcp_targeting_note}\n\n"
        f"{description_rules}\n\n"
        f"{goal_summary_note}\n\n"
        f"{edge_note}\n\n"
        f"{enum_note}\n\n"
        "Emit the manifest as a fenced ```json block (valid JSON, not YAML):\n\n"
        f"{schema}"
    )
