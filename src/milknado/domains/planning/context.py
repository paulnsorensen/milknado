from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from milknado.domains.planning.touch_sites import _touch_sites_section

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
        _touch_sites_section(spec_text, crg, tilth, scope),
        _graph_section(graph),
        _batching_section(),
        _instructions_section(resuming),
    ]
    if spec_text is not None:
        sections.append(_spec_section(spec_text))
    return "\n\n".join(sections)


def _goal_section(goal: str) -> str:
    return f"# Goal\n\n{goal}"


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
        '      "depends_on": []\n'
        "    },\n"
        "    {\n"
        '      "id": "c2",\n'
        '      "path": "src/bar.py",\n'
        '      "edit_kind": "add",\n'
        '      "description": "Respond to c1 signature change — update call site.",\n'
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

    # US-004: if 3 consecutive post-US-001+US-002 runs show multi_story_change_count>0
    # AND US-008 has shipped, open a follow-up to revert FileChange to one-per-SymbolRef.
    granularity_note = (
        "Emit file-level changes. More granular is better — one change per story,"
        " one story per change. The solver batches them optimally."
    )

    tests_note = (
        "**Every user story requires a tests/ change.** Emit a separate change entry"
        " for each test file and set `depends_on` to point at the implementation change:\n"
        "```json\n"
        '{"id": "c2", "path": "tests/test_foo.py", "edit_kind": "add",\n'
        ' "description": "Tests for FooClass — verifies §Feature A (depends on c1)",\n'
        ' "depends_on": ["c1"]}\n'
        "```"
    )

    bundling_anti_pattern = (
        "**Anti-pattern — bundling multiple stories into one change:**\n"
        "One change must cover exactly one user story. Even when two stories touch the"
        " same file, emit two separate changes — the solver merges safely:\n"
        "```json\n"
        "// BAD — c1 and c2 both claim src/foo.py and cover different stories\n"
        '{"id": "c1", "path": "src/foo.py", "description": "Implement US-001 and US-002"}\n'
        "\n"
        "// GOOD — one change per story; path collision is fine when stories are distinct\n"
        '{"id": "c1", "path": "src/foo.py", "description": "US-001: add FooClass (§ A)"}\n'
        '{"id": "c2", "path": "src/foo.py", "description": "US-002: add BarMethod (§ B)"}\n'
        "```"
    )

    bundled_description_anti_pattern = (
        "**Anti-pattern — bundled description:**\n"
        'A description such as `"Implement US-001, US-002, and US-003"` is a bundling'
        " signal. Split into one change per story with an individual description each."
    )

    reject_and_retry_rule = (
        "**Reject-and-retry rule:** Before returning, scan every change description"
        r" for the pattern `\bUS-\d{3}\b.*\bUS-\d{3}\b` (two or more story references"
        " in one description). If any description matches, you MUST split that change"
        " into one change per story and re-check every description before returning."
    )

    worked_example = (
        "**Worked example — 3 stories → 3 impl changes + 3 test changes:**\n"
        "```json\n"
        '{"id": "c1", "path": "src/auth.py", "edit_kind": "modify",\n'
        ' "description": "US-101: add token refresh endpoint (§ Auth)",\n'
        ' "depends_on": []},\n'
        '{"id": "c2", "path": "src/session.py", "edit_kind": "modify",\n'
        ' "description": "US-102: expire stale sessions (§ Session)",\n'
        ' "depends_on": []},\n'
        '{"id": "c3", "path": "src/audit.py", "edit_kind": "add",\n'
        ' "description": "US-103: write audit log on logout (§ Audit)",\n'
        ' "depends_on": ["c1"]},\n'
        '{"id": "c4", "path": "tests/test_auth.py", "edit_kind": "add",\n'
        ' "description": "US-101 tests — verifies refresh flow", "depends_on": ["c1"]},\n'
        '{"id": "c5", "path": "tests/test_session.py", "edit_kind": "add",\n'
        ' "description": "US-102 tests — verifies expiry logic", "depends_on": ["c2"]},\n'
        '{"id": "c6", "path": "tests/test_audit.py", "edit_kind": "add",\n'
        ' "description": "US-103 tests — verifies audit event written", "depends_on": ["c3"]}\n'
        "```"
    )

    goal_summary_note = (
        "`goal_summary` is 2-4 sentences structured as **what / why / success criteria**."
        " It becomes the root Mikado node description that every executor reads."
        " Make it load-bearing."
    )

    if not resuming:
        return (
            "# Instructions\n\n"
            "Decompose the goal into a v2 change manifest.\n\n"
            f"{tests_note}\n\n"
            f"{granularity_note}\n\n"
            f"{bundling_anti_pattern}\n\n"
            f"{bundled_description_anti_pattern}\n\n"
            f"{reject_and_retry_rule}\n\n"
            f"{worked_example}\n\n"
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
        f"{tests_note}\n\n"
        f"{granularity_note}\n\n"
        f"{bundling_anti_pattern}\n\n"
        f"{bundled_description_anti_pattern}\n\n"
        f"{reject_and_retry_rule}\n\n"
        f"{worked_example}\n\n"
        f"{description_rules}\n\n"
        f"{goal_summary_note}\n\n"
        f"{edge_note}\n\n"
        f"{enum_note}\n\n"
        "Emit the manifest as a fenced ```json block (valid JSON, not YAML):\n\n"
        f"{schema}"
    )
