"""Shared related-page projections for graph detail controls."""

from __future__ import annotations

from typing import cast

from milknado.domains.graph import NodeDetailSnapshot, SnapshotPage


def related_pages(detail: NodeDetailSnapshot | None) -> tuple[SnapshotPage[object], ...]:
    if detail is None:
        return ()
    return tuple(
        cast(SnapshotPage[object], page)
        for page in (
            detail.children,
            detail.ancestors,
            detail.prerequisite_ids,
            detail.dependent_ids,
            detail.reverse_dependents,
            detail.owned_files,
            detail.runs,
            detail.reviews,
            detail.sessions,
            detail.receipts,
            detail.artifacts,
        )
    )
