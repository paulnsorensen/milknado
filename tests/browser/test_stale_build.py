"""AC-2: the committed build stays byte-stable with `web/`'s current source."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from tests.browser.conftest import COMMITTED_STATIC_DIR, WEB_DIR, build_web_to, diff_build_trees

pytestmark = pytest.mark.browser


def test_committed_build_matches_current_source(tmp_path: Path) -> None:
    build_web_to(tmp_path)

    differing = diff_build_trees(COMMITTED_STATIC_DIR, tmp_path)

    assert differing == [], (
        "committed src/milknado/web/static is stale versus web/ source; "
        f"run `npm --prefix web run build` and commit these files: {differing}"
    )


def _copy_web_source(destination: Path) -> Path:
    """Copy `web/` so a test can mutate source without racing parallel workers."""
    _ = shutil.copytree(WEB_DIR, destination, ignore=shutil.ignore_patterns("node_modules"))
    (destination / "node_modules").symlink_to(WEB_DIR / "node_modules")
    return destination


def test_stale_build_is_detected_without_rebuilding(tmp_path: Path) -> None:
    web_copy = _copy_web_source(tmp_path / "web")
    index_html = web_copy / "index.html"
    original = index_html.read_text(encoding="utf-8")
    mutated = original.replace("</body>", "<!-- stale-build-marker -->\n  </body>")
    assert mutated != original, "expected </body> to exist in web/index.html"
    _ = index_html.write_text(mutated, encoding="utf-8")

    build_web_to(tmp_path / "out", web_dir=web_copy)
    differing = diff_build_trees(COMMITTED_STATIC_DIR, tmp_path / "out")

    assert differing, "expected a mutated source build to differ from the committed build"
    assert "index.html" in differing, differing
