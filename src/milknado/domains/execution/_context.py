from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from milknado.domains.common.protocols import CrgPort
    from milknado.domains.common.types import MikadoNode
    from milknado.domains.graph import MikadoGraph

from milknado.domains.graph.traversals import walk_ancestors


def build_node_context(
    node: MikadoNode,
    graph: MikadoGraph,
    crg: CrgPort | None,
) -> str:
    ancestors = walk_ancestors(graph, node.id)
    root = ancestors[-1]
    sections = [f"## Goal\n\n{root.description}"]

    why_nodes = ancestors[1:-1]
    if why_nodes:
        why_parts = "\n".join(f"### {n.description}" for n in why_nodes)
        sections.append(f"## Why chain (parent → grandparent → ...)\n\n{why_parts}")

    sections.append(f"## Your task\n\n{node.description}")

    files = graph.get_file_ownership(node.id)
    if files:
        file_list = "\n".join(f"- `{f}`" for f in files)
        sections.append(f"## Files\n\n{file_list}")
    else:
        sections.append("## Files\n\n_(no files assigned)_")

    sections.append(_impact_radius_section(crg, files))
    return "\n\n".join(sections)


def _impact_radius_section(crg: CrgPort | None, files: list[str]) -> str:
    if crg is None:
        return "## Impact Radius\n\n_(CRG unavailable — impact radius skipped)_"
    if not files:
        return "## Impact Radius\n\n_(no files — impact radius skipped)_"
    try:
        impact = crg.get_impact_radius(files)
    except Exception as exc:
        return f"## Impact Radius\n\n_(CRG unavailable — impact radius skipped: {exc})_"
    return f"## Impact Radius\n\n{impact}"
