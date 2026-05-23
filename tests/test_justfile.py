from __future__ import annotations

from pathlib import Path

_JUSTFILE = Path(__file__).resolve().parents[1] / "justfile"


def test_mcp_dev_recipe_exists_with_watcher() -> None:
    text = _JUSTFILE.read_text(encoding="utf-8")
    assert "\nmcp-dev:" in text
    assert "watchfiles" in text
    assert "src/milknado" in text
