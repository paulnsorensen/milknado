from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from milknado.domains.common.types import DegradationMarker, TilthMap

if TYPE_CHECKING:
    from milknado.domains.common.protocols import CrgPort, TilthPort

MAX_FILES = 6
_TOKEN_BUDGET = 3000
_CHARS_PER_TOKEN = 4

_US_HEADING_RE = re.compile(r"^##\s+US-\d+[:\s]+(.+)", re.MULTILINE)
_H3_HEADING_RE = re.compile(r"^###\s+(.+)", re.MULTILINE)


def _extract_keywords(spec_text: str) -> list[str]:
    """Return unique queries from US-NNN headings then ### headings, in order."""
    seen: set[str] = set()
    keywords: list[str] = []

    for m in _US_HEADING_RE.finditer(spec_text):
        kw = m.group(1).strip()
        if kw and kw not in seen:
            seen.add(kw)
            keywords.append(kw)

    for m in _H3_HEADING_RE.finditer(spec_text):
        kw = m.group(1).strip()
        if kw and kw not in seen:
            seen.add(kw)
            keywords.append(kw)

    return keywords


def _safe_crg_search(crg: CrgPort, keyword: str) -> list[dict]:
    try:
        return crg.semantic_search(keyword, top_n=5, detail_level="minimal")
    except Exception:
        return []


def _safe_tilth_search(tilth: TilthPort, keyword: str) -> list:
    try:
        return tilth.search_symbol(keyword)
    except Exception:
        return []


def _path_from_crg_hit(hit: dict) -> str:
    return hit.get("file_path") or hit.get("path") or hit.get("file") or hit.get("name", "")


def _touch_sites_section(
    spec_text: str | None,
    crg: CrgPort | None,
    tilth: TilthPort | None,
    scope: Path | None,
) -> str:
    header = "# Probable Touch Sites"

    if spec_text is None:
        return _degrade_no_spec(header, tilth, scope)

    keywords = _extract_keywords(spec_text)
    if not keywords:
        return f"{header}\n\n_(no US-NNN blocks or ### headings found in spec)_"

    file_scores: dict[str, int] = {}
    zero_hit_kws: list[str] = []

    if crg is not None:
        for kw in keywords:
            hits = _safe_crg_search(crg, kw)
            if not hits:
                zero_hit_kws.append(kw)
            for r in hits:
                p = _path_from_crg_hit(r)
                if p:
                    file_scores[p] = file_scores.get(p, 0) + 1
    else:
        zero_hit_kws = list(keywords)

    file_locs: dict[str, tuple[int, int]] = {}
    if tilth is not None:
        for kw in zero_hit_kws:
            locs = _safe_tilth_search(tilth, kw)
            for loc in locs:
                p = str(loc.path)
                file_scores[p] = file_scores.get(p, 0) + 1
                if p not in file_locs:
                    file_locs[p] = (loc.line_start, loc.line_end)

    if not file_scores:
        return f"{header}\n\n_(no touch sites found for {len(keywords)} keywords)_"

    ranked = sorted(file_scores.items(), key=lambda x: -x[1])[:MAX_FILES]

    parts = [header]
    budget_chars = _TOKEN_BUDGET * _CHARS_PER_TOKEN

    for path_str, score in ranked:
        if budget_chars <= 0:
            break
        entry = _render_entry(path_str, score, file_locs.get(path_str), tilth, budget_chars)
        parts.append(entry)
        budget_chars -= len(entry)

    return "\n\n".join(parts)


def _render_entry(
    path_str: str,
    score: int,
    loc: tuple[int, int] | None,
    tilth: TilthPort | None,
    budget_chars: int,
) -> str:
    label = f"## `{path_str}` ({score} hit{'s' if score != 1 else ''})"
    if tilth is None or loc is None:
        return label

    try:
        content = tilth.read_section(Path(path_str), loc[0], loc[1])
    except Exception:
        return label

    if not content:
        return label

    snippet = content[:budget_chars]
    return f"{label}\n```\n{snippet}\n```"


def _degrade_no_spec(
    header: str,
    tilth: TilthPort | None,
    scope: Path | None,
) -> str:
    if tilth is None:
        return f"{header}\n\n_(spec not provided and tilth unavailable — touch sites skipped)_"

    src_scope = (scope / "src") if scope is not None else Path("src")
    result = tilth.structural_map(src_scope, 400)

    if isinstance(result, DegradationMarker):
        return f"{header}\n\n_(spec not provided; tilth structural map degraded: {result.reason})_"

    assert isinstance(result, TilthMap)
    lines = [header, "_(no spec provided — src/ structure)_"]
    for key, value in result.data.items():
        lines.append(f"- **{key}**: {value}")
    return "\n".join(lines)
