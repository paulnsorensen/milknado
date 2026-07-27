from __future__ import annotations

from pathlib import Path

from scripts.check_file_lengths import _violations, main


def _write_source(root: Path, name: str, lines: int) -> None:
    path = root / "src" / "milknado" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("line\n" * lines, encoding="utf-8")


def _write_pyproject(root: Path, max_lines: int, file_exceptions: str = "") -> None:
    (root / "pyproject.toml").write_text(
        "[tool.milknado.quality]\n"
        f"max_file_lines = {max_lines}\n\n"
        "[tool.milknado.quality.file_exceptions]\n"
        f"{file_exceptions}",
        encoding="utf-8",
    )


def test_violations_use_global_budget(tmp_path: Path) -> None:
    _write_source(tmp_path, "small.py", 3)

    assert _violations(tmp_path, 2, {}) == ["src/milknado/small.py: 3 lines (limit 2)"]


def test_violations_apply_measured_waiver(tmp_path: Path) -> None:
    _write_source(tmp_path, "large.py", 3)
    exceptions = {"src/milknado/large.py": {"max_lines": 3}}

    assert _violations(tmp_path, 2, exceptions) == []


def test_violations_flags_file_exceeding_waiver(tmp_path: Path) -> None:
    _write_source(tmp_path, "large.py", 4)
    exceptions = {"src/milknado/large.py": {"max_lines": 3}}

    assert _violations(tmp_path, 2, exceptions) == ["src/milknado/large.py: 4 lines (limit 3)"]


def test_main_passes_on_clean_tree(tmp_path: Path) -> None:
    _write_source(tmp_path, "small.py", 2)
    _write_pyproject(tmp_path, 3)

    assert main(tmp_path) == 0


def test_main_fails_and_prints_violation(tmp_path: Path, capsys) -> None:
    _write_source(tmp_path, "big.py", 4)
    _write_pyproject(tmp_path, 2)

    assert main(tmp_path) == 1
    out = capsys.readouterr().out
    assert "Source files exceed the 2-line budget:" in out
    assert "src/milknado/big.py: 4 lines (limit 2)" in out


def test_main_fails_on_missing_reason(tmp_path: Path, capsys) -> None:
    _write_source(tmp_path, "big.py", 4)
    _write_pyproject(
        tmp_path,
        2,
        '"src/milknado/big.py" = { max_lines = 5 }\n',
    )

    assert main(tmp_path) == 1
    out = capsys.readouterr().out
    assert "File-size waivers need reasons: src/milknado/big.py" in out


def test_main_fails_on_stale_waiver(tmp_path: Path, capsys) -> None:
    _write_source(tmp_path, "small.py", 2)
    _write_pyproject(
        tmp_path,
        3,
        '"src/milknado/gone.py" = { max_lines = 5, reason = "stale" }\n',
    )

    assert main(tmp_path) == 1
    out = capsys.readouterr().out
    assert "File-size waivers reference files that no longer exist: src/milknado/gone.py" in out
