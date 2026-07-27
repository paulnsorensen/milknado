from __future__ import annotations

from pathlib import Path

from scripts.check_file_lengths import _violations


def _write_source(root: Path, name: str, lines: int) -> None:
    path = root / "src" / "milknado" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("line\n" * lines, encoding="utf-8")


def test_violations_use_global_budget(tmp_path: Path) -> None:
    _write_source(tmp_path, "small.py", 3)

    assert _violations(tmp_path, 2, {}) == ["src/milknado/small.py: 3 lines (limit 2)"]


def test_violations_apply_measured_waiver(tmp_path: Path) -> None:
    _write_source(tmp_path, "large.py", 3)
    exceptions = {"src/milknado/large.py": {"max_lines": 3}}

    assert _violations(tmp_path, 2, exceptions) == []
