from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from milknado.domains.common.types import DegradationMarker, TilthMap

if TYPE_CHECKING:
    from milknado.domains.common.protocols import CrgPort, TilthPort
    from milknado.domains.common.types import MikadoNode
    from milknado.domains.graph import MikadoGraph


def build_planning_context(
    spec_path: Path,
    crg: CrgPort,
    graph: MikadoGraph,
    *,
    execution_agent: str,
    tilth: TilthPort | None = None,
    project_root: Path | None = None,
) -> str:
    spec_text = spec_path.read_text(encoding="utf-8")
    resuming = len(graph.get_all_nodes()) > 0
    token_estimate = _estimate_tokens(spec_text)
    scope = project_root if project_root is not None else spec_path.parent
    sections = [
        _spec_section(spec_path, spec_text, token_estimate),
        _crg_usage_section(),
        _atom_budget_heuristics_section(token_estimate),
        _architecture_section(crg),
        _structural_section(tilth, scope),
        _graph_section(graph),
        _instructions_section(resuming, execution_agent),
    ]
    return "\n\n".join(sections)


def _spec_section(spec_path: Path, spec_text: str, token_estimate: int) -> str:
    return (
        f"# Spec\n\n"
        f"- path: `{spec_path}`\n"
        f"- estimated_tokens: {token_estimate}\n\n"
        f"```markdown\n{spec_text}\n```"
    )


def _crg_usage_section() -> str:
    return (
        "# CRG Token-Efficient Workflow\n\n"
        "Follow this workflow during planning:\n"
        "1. `get_minimal_context` first.\n"
        "2. Keep `detail_level=\"minimal\"` unless a high-risk area needs expansion.\n"
        "3. Prefer targeted graph queries over broad scans.\n"
        "4. Avoid loading full files unless strictly required.\n"
        "5. Keep the graph-call budget tight and summarize findings compactly."
    )


def _atom_budget_heuristics_section(spec_tokens: int) -> str:
    startup_overhead = 20_000
    graph_context = 4_000
    tool_reserve = 7_500
    effective_budget = 70_000 - startup_overhead - graph_context - tool_reserve - spec_tokens
    return (
        "# Atom Budget Heuristics\n\n"
        "Use tiktoken for all estimates and split/merge decisions.\n\n"
        f"- spec_tokens: {spec_tokens}\n"
        f"- startup_overhead: {startup_overhead}\n"
        f"- graph_context_reserve: {graph_context}\n"
        f"- tool_output_reserve: {tool_reserve}\n"
        f"- effective_code_budget: {max(0, effective_budget)}\n\n"
        "Formulas:\n"
        "- atom_read_tokens = tiktoken(read_payload_text)\n"
        "- atom_write_tokens = tiktoken(predicted_write_payload_text)\n"
        "- atom_total_tokens = startup_overhead + spec_tokens + graph_context_reserve + "
        "tool_output_reserve + atom_read_tokens + atom_write_tokens\n\n"
        "Thresholds:\n"
        "- <25k: merge candidate\n"
        "- 25k-40k: optimal\n"
        "- 40k-50k: tight\n"
        "- 50k-65k: split recommended\n"
        "- >65k: split required\n"
    )


def _architecture_section(crg: CrgPort) -> str:
    communities = crg.list_communities(sort_by="size")[:5]
    flows = crg.list_flows(sort_by="criticality", limit=3)
    overview = crg.get_architecture_overview()
    warnings = overview.get("warnings", []) if isinstance(overview, dict) else []

    lines = ["# Architecture Overview (compact, from code-review-graph)\n"]
    lines.append("## Top communities\n")
    if communities:
        for c in communities:
            name = c.get("name", "?")
            size = c.get("size", "?")
            cohesion = c.get("cohesion", "?")
            lines.append(f"- {name} (size={size}, cohesion={cohesion})")
    else:
        lines.append("- none detected")

    lines.append("\n## Top flows\n")
    if flows:
        for f in flows:
            name = f.get("name", "?")
            criticality = f.get("criticality", "?")
            lines.append(f"- {name} (criticality={criticality})")
    else:
        lines.append("- none detected")

    if warnings:
        lines.append("\n## Warnings\n")
        for w in warnings:
            lines.append(f"- {w}")

    return "\n".join(lines)


def _structural_section(tilth: TilthPort | None, scope: Path) -> str:
    if tilth is None:
        return (
            "# Structural Map (tilth)\n\n"
            "- status: not_configured\n"
            "- reason: tilth adapter not wired into planner"
        )
    result = tilth.structural_map(scope, budget_tokens=2_000)
    if isinstance(result, DegradationMarker):
        return (
            "# Structural Map (tilth)\n\n"
            "- status: degraded\n"
            f"- reason: {result.reason}\n"
            f"- detail: {result.detail}"
        )
    return _render_tilth_map(result)


def _render_tilth_map(tmap: TilthMap) -> str:
    formatted = json.dumps(tmap.data, indent=2, default=str)
    return (
        "# Structural Map (tilth)\n\n"
        f"- scope: {tmap.scope}\n"
        f"- budget_tokens: {tmap.budget_tokens}\n\n"
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
    if ready:
        lines.append("\n## Ready to Execute\n")
        for node in ready:
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


def _instructions_section(resuming: bool, execution_agent: str) -> str:
    manifest_contract = (
        "Return exactly one JSON object in a fenced `json` code block. "
        "No prose before or after the JSON.\n\n"
        "Schema:\n"
        "```json\n"
        "{\n"
        '  "manifest_version": "milknado.plan.v1",\n'
        '  "atoms": [\n'
        "    {\n"
        '      "id": "A1",\n'
        '      "description": "short action-oriented task",\n'
        '      "depends_on": ["A2"],\n'
        '      "files": ["src/path.py"],\n'
        '      "token_budget": {\n'
        '        "estimated_read_tokens": 12000,\n'
        '        "estimated_write_tokens": 6000,\n'
        '        "estimated_total_tokens": 45000,\n'
        '        "split_required": false\n'
        "      }\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "```"
    )

    if not resuming:
        return (
            "# Instructions\n\n"
            "Decompose the spec into a Mikado dependency graph.\n"
            "Treat each atom as a DB-ready node candidate.\n\n"
            "Budget each atom with token-awareness:\n"
            "- Use tiktoken for read/write/total estimates.\n"
            "- Fill `token_budget` per the Atom Budget Heuristics thresholds.\n"
            "- Set `split_required: true` when atom_total_tokens exceeds 65k.\n"
            "- Split nodes that exceed budget; merge undersized siblings when safe.\n\n"
            f"Execution agent target for run phase: `{execution_agent}`\n\n"
            "Dependency semantics:\n"
            "- `depends_on` lists prerequisite atoms.\n"
            "- If A depends_on B, B must complete before A.\n"
            "- In the edges table: `edges.parent_id = A` (dependent), "
            "`edges.child_id = B` (prerequisite).\n"
            "- Child finishes first; do not reverse.\n\n"
            "Fallback (only if manifest cannot be produced): use `milknado add-node`.\n\n"
            f"{manifest_contract}\n\n"
            "Start from leaf-prerequisites first, then parent atoms."
        )

    return (
        "# Instructions (resuming)\n\n"
        "The graph above shows prior progress. Do NOT recreate "
        "existing nodes.\n\n"
        "Review the current state and produce only net-new atoms:\n"
        "- Add new child nodes for any remaining work\n"
        "- Re-plan around failed nodes if the approach needs to change\n"
        "- Ensure all pending nodes still make sense given completed work\n\n"
        "Use tiktoken to keep atom budgets in range; fill `token_budget` "
        "per the Atom Budget Heuristics thresholds.\n\n"
        "Dependency semantics:\n"
        "- dependent atom = `edges.parent_id`; prerequisite = `edges.child_id`.\n"
        "- Child finishes first; do not reverse.\n\n"
        f"Execution agent target for run phase: `{execution_agent}`\n\n"
        "Fallback (only if manifest cannot be produced): use `milknado add-node`.\n\n"
        f"{manifest_contract}"
    )


def _estimate_tokens(text: str) -> int:
    import tiktoken  # type: ignore[import-not-found]

    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))
