"""Pure renderers for roadmap graph documents."""

from __future__ import annotations

import html
import re

from milknado.domains.wiki.model import Lifecycle, RoadmapModel

_STATUS_STYLES = {
    "pending": ("#e2e8f0", "#475569", "#0f172a"),
    "running": ("#dbeafe", "#2563eb", "#172554"),
    "done": ("#dcfce7", "#16a34a", "#14532d"),
    "blocked": ("#fef3c7", "#d97706", "#451a03"),
    "failed": ("#fee2e2", "#dc2626", "#450a0a"),
}


def _node_id(slug: str) -> str:
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", slug):
        return slug
    return "node_" + re.sub(r"[^A-Za-z0-9_]", "_", slug)


def _node_ids(slugs: list[str]) -> dict[str, str]:
    counts: dict[str, int] = {}
    result: dict[str, str] = {}
    for slug in slugs:
        base = _node_id(slug)
        counts[base] = counts.get(base, 0) + 1
        suffix = counts[base]
        result[slug] = base if suffix == 1 else f"{base}_{suffix}"
    return result


def _label(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def _status_value(goal: object) -> str:
    status = getattr(goal, "status", None)
    return getattr(status, "value", None) or "pending"


def render_mermaid(model: RoadmapModel) -> str:
    """Render a deterministic Mermaid representation of a roadmap prereq DAG."""
    lines = ["graph TD"]
    node_ids = _node_ids(sorted(model.goals))
    for slug in sorted(model.goals):
        goal = model.goals[slug]
        lines.append(f'    {node_ids[slug]}["{_label(goal.title)}"]')
    edges: set[tuple[str, str]] = set()
    for slug in sorted(model.goals):
        for prereq in model.goals[slug].edges:
            if prereq in node_ids:
                edges.add((prereq, slug))
    lines.extend(
        f"    {node_ids[source]} --> {node_ids[target]}" for source, target in sorted(edges)
    )
    lines.extend(_class_definitions())
    lines.extend(
        f"    class {node_ids[slug]} {','.join(_classes(model.goals[slug]))}"
        for slug in sorted(model.goals)
    )
    return "\n".join(lines) + "\n"


def _classes(goal: object) -> list[str]:
    classes = [f"status-{_status_value(goal)}"]
    if getattr(goal, "lifecycle", Lifecycle.ACTIVE) is Lifecycle.DEPRECATED:
        classes.append("lifecycle-deprecated")
    return classes


def _class_definitions() -> list[str]:
    lines = []
    for status, (fill, stroke, text) in _STATUS_STYLES.items():
        lines.append(f"    classDef status-{status} fill:{fill},stroke:{stroke},color:{text}")
    lines.append(
        "    classDef lifecycle-deprecated stroke:#7c3aed,stroke-width:3px,stroke-dasharray:5 5"
    )
    return lines


def _svg(model: RoadmapModel) -> str:
    goals = sorted(model.goals)
    width, row_height = 520, 80
    height = max(120, 40 + row_height * len(goals))
    positions = {slug: (40, 40 + row_height * index) for index, slug in enumerate(goals)}
    lines = [f'<svg viewBox="0 0 {width} {height}" role="img">', "<defs>"]
    lines.append(
        '<marker id="arrow" markerWidth="8" markerHeight="8" '
        'refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z"/>'
        "</marker>"
    )
    lines.append("</defs>")
    for slug in goals:
        for prereq in model.goals[slug].edges:
            if prereq in positions:
                source_x, source_y = positions[prereq]
                target_x, target_y = positions[slug]
                lines.append(
                    f'<path d="M {source_x + 360} {source_y} L {target_x} {target_y}" '
                    'fill="none" stroke="#64748b" marker-end="url(#arrow)"/>'
                )
    for slug in goals:
        goal = model.goals[slug]
        fill, stroke, text = _STATUS_STYLES[_status_value(goal)]
        dash = (
            ' stroke-dasharray="6 4"'
            if Lifecycle.DEPRECATED is getattr(goal, "lifecycle", None)
            else ""
        )
        x, y = positions[slug]
        lines.append(
            f'<rect x="{x}" y="{y - 24}" width="360" height="48" rx="8" '
            f'fill="{fill}" stroke="{stroke}"{dash}/>'
        )
        lines.append(
            f'<text x="{x + 16}" y="{y + 5}" fill="{text}">{html.escape(goal.title)}</text>'
        )
    lines.append("</svg>")
    return "".join(lines)


def render_html(model: RoadmapModel) -> str:
    """Render a self-contained static HTML page containing the roadmap SVG."""
    graph = render_mermaid(model)
    title = html.escape(model.roadmap.slug)
    return (
        "<!doctype html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        f"<title>{title} roadmap</title>\n"
        "<style>body{font-family:system-ui,sans-serif;margin:2rem}"
        "svg{display:block;max-width:100%;height:auto}"
        "</style>\n"
        "</head>\n"
        "<body>\n"
        f"<h1>{title}</h1>\n"
        f"{_svg(model)}\n"
        "<details><summary>Mermaid source</summary>\n"
        f"<pre>{html.escape(graph)}</pre>\n"
        "</details>\n"
        "</body>\n"
        "</html>\n"
    )
