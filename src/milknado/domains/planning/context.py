from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from milknado.domains.common.protocols import CrgPort
    from milknado.domains.common.types import MikadoNode
    from milknado.domains.graph import MikadoGraph


def build_planning_context(
    spec_path: Path,
    crg: CrgPort,
    graph: MikadoGraph,
    *,
    execution_agent: str,
) -> str:
    spec_text = spec_path.read_text(encoding="utf-8")
    resuming = len(graph.get_all_nodes()) > 0
    token_estimate = _estimate_tokens(spec_text)
    sections = [
        _spec_section(spec_path, spec_text, token_estimate),
        _crg_usage_section(),
        _atom_budget_heuristics_section(token_estimate),
        _architecture_section(crg),
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
            "Decompose the spec into a Mikado dependency graph.\n"
            "For each node, specify:\n"
            "- A short description of the work\n"
            "- Which files it will touch\n"
            "- Dependencies (which nodes must complete first)\n\n"
            "Budget each atom with token-awareness:\n"
            "- Include read tokens and expected write tokens.\n"
            "- Keep atom total near 50k-70k including overhead.\n"
            "- Split nodes that exceed budget; merge undersized siblings when safe.\n\n"
            f"Execution agent target for run phase: `{execution_agent}`\n\n"
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
        f"Execution agent target for run phase: `{execution_agent}`\n\n"
        f"{add_node_usage}"
    )


def _estimate_tokens(text: str) -> int:
    import tiktoken  # type: ignore[import-not-found]

    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))
