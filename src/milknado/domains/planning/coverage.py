from __future__ import annotations

import re

from milknado.domains.planning.manifest import PlanChangeManifest

_US_TOKEN_RE = re.compile(r"\bUS-\d{3}\b")


def coverage_check(manifest: PlanChangeManifest) -> list[str]:
    """Return impl change paths that lack test coverage.

    An impl change (path under src/) is covered when at least one tests/ change
    has a depends_on pointing to the impl change's id, OR shares at least one
    US-NNN ref in its description.
    """
    impl_changes = [c for c in manifest.changes if c.path.startswith("src/")]
    test_changes = [c for c in manifest.changes if c.path.startswith("tests/")]

    if not impl_changes:
        return []

    orphans: list[str] = []
    for impl in impl_changes:
        impl_refs = set(_US_TOKEN_RE.findall(impl.description))
        covered = any(
            impl.id in test.depends_on
            or bool(impl_refs and impl_refs & set(_US_TOKEN_RE.findall(test.description)))
            for test in test_changes
        )
        if not covered:
            orphans.append(impl.path)

    return sorted(orphans)
