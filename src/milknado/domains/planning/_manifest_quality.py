from __future__ import annotations

import logging
import re
from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from milknado.domains.common.protocols import CrgPort
    from milknado.domains.planning.manifest import PlanChangeManifest

_US_REF_RE = re.compile(r"\bUS-\d{3}\b")
_logger = logging.getLogger(__name__)


def get_bundled_changes(manifest: PlanChangeManifest) -> list[str]:
    bundled = []
    for c in manifest.changes:
        refs = set(_US_REF_RE.findall(c.description or ""))
        if len(refs) >= 2:
            bundled.append(c.path)
    return bundled


def summarise_manifest_quality(manifest: PlanChangeManifest) -> dict[str, int]:
    if not manifest.changes:
        return {
            "impl_change_count": 0,
            "test_change_count": 0,
            "multi_story_change_count": 0,
            "max_us_refs_per_change": 0,
            "distinct_path_count": 0,
        }
    impl = sum(1 for c in manifest.changes if c.path.startswith("src/"))
    tests = sum(1 for c in manifest.changes if c.path.startswith("tests/"))
    us_per = [len(set(_US_REF_RE.findall(c.description or ""))) for c in manifest.changes]
    multi = sum(1 for n in us_per if n >= 2)
    return {
        "impl_change_count": impl,
        "test_change_count": tests,
        "multi_story_change_count": multi,
        "max_us_refs_per_change": max(us_per) if us_per else 0,
        "distinct_path_count": len({c.path for c in manifest.changes}),
    }


def append_reuse_candidates(
    manifest: PlanChangeManifest,
    crg: CrgPort | None,
) -> PlanChangeManifest:
    if crg is None:
        return manifest
    from milknado.domains.batching import FileChange

    updated: list[FileChange] = []
    for change in manifest.changes:
        query = __import__("pathlib").Path(change.path).stem
        try:
            hits = crg.semantic_search_nodes(query, top_n=5)
        except Exception as exc:
            _logger.warning("CRG reuse search failed for %s: %s", change.path, exc)
            hits = []
        if not hits:
            updated.append(change)
            continue
        lines = ["\n\n## Reuse candidates"]
        for h in hits:
            sym = h.get("symbol_name", "?")
            fpath = h.get("file_path", "?")
            summary = h.get("summary", "")
            lines.append(f"- `{sym}` ({fpath}): {summary}")
        updated.append(replace(change, description=change.description + "\n".join(lines)))
    return replace(manifest, changes=tuple(updated))
