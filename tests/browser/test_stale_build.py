"""AC-2: the committed build stays byte-stable with `web/`'s current source."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.browser.conftest import COMMITTED_STATIC_DIR, WEB_DIR, build_web_to, diff_build_trees

pytestmark = pytest.mark.browser

_INDEX_HTML = WEB_DIR / "index.html"


def test_committed_build_matches_current_source(tmp_path: Path) -> None:
    build_web_to(tmp_path)

    differing = diff_build_trees(COMMITTED_STATIC_DIR, tmp_path)

    assert differing == [], (
        "committed src/milknado/web/static is stale versus web/ source; "
        f"run `npm --prefix web run build` and commit these files: {differing}"
    )


def test_stale_build_is_detected_without_rebuilding(tmp_path: Path) -> None:
    original = _INDEX_HTML.read_text(encoding="utf-8")
    mutated = original.replace("</body>", "<!-- stale-build-marker -->\n  </body>")
    assert mutated != original, "expected </body> to exist in web/index.html"
    _ = _INDEX_HTML.write_text(mutated, encoding="utf-8")
    try:
        build_web_to(tmp_path)
        differing = diff_build_trees(COMMITTED_STATIC_DIR, tmp_path)
    finally:
        _ = _INDEX_HTML.write_text(original, encoding="utf-8")

    assert differing, "expected a mutated source build to differ from the committed build"
    assert "index.html" in differing, differing
