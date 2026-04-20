from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock, patch

from milknado.domains.batching import (
    FileChange,
    SymbolRef,
    estimate_tokens,
    estimate_tokens_per_symbols,
)
from milknado.domains.batching.weights import HEADROOM
from milknado.domains.common.protocols import SymbolLocation


def test_delete_is_flat_80(tmp_path: Path) -> None:
    c = FileChange(id="1", path="x.py", edit_kind="delete")
    assert estimate_tokens(c, tmp_path) == 80


def test_rename_is_flat_120(tmp_path: Path) -> None:
    c = FileChange(id="1", path="x.py", edit_kind="rename")
    assert estimate_tokens(c, tmp_path) == 120


def test_modify_uses_tiktoken_on_existing_file(tmp_path: Path) -> None:
    f = tmp_path / "a.py"
    f.write_text("def hello():\n    return 42\n")
    c = FileChange(id="1", path="a.py", edit_kind="modify")
    with patch(
        "milknado.domains.batching.weights._tiktoken_count", return_value=10
    ):
        n = estimate_tokens(c, tmp_path)
    assert 5 < n < 100


def test_modify_falls_back_to_median_when_file_missing(tmp_path: Path) -> None:
    c = FileChange(id="1", path="missing.py", edit_kind="modify")
    # 150 lines * 10 tpl * 1.25 = 1875
    assert estimate_tokens(c, tmp_path) == 1875


def test_add_uses_median_lines_by_extension(tmp_path: Path) -> None:
    c = FileChange(id="1", path="new.py", edit_kind="add")
    assert estimate_tokens(c, tmp_path) == 1875  # 150 * 10 * 1.25


def test_unknown_extension_uses_defaults(tmp_path: Path) -> None:
    c = FileChange(id="1", path="new.xyz", edit_kind="add")
    assert estimate_tokens(c, tmp_path) == 1500  # 150 * 8 * 1.25


def test_rs_extension(tmp_path: Path) -> None:
    c = FileChange(id="1", path="lib.rs", edit_kind="add")
    # 200 * 11 * 1.25 = 2750
    assert estimate_tokens(c, tmp_path) == 2750


# --- estimate_tokens_per_symbols ---


def _make_tilth_port(slice_text: str, line_start: int = 1, line_end: int = 5) -> MagicMock:
    port = MagicMock()
    port.search_symbol.return_value = [
        SymbolLocation(path=Path("src/foo.py"), line_start=line_start, line_end=line_end)
    ]
    port.read_section.return_value = slice_text
    return port


def test_per_symbols_tilth_none_falls_back_to_path_level(tmp_path: Path) -> None:
    f = tmp_path / "a.py"
    f.write_text("x = 1\n")
    c = FileChange(id="1", path="a.py", edit_kind="modify")
    with patch("milknado.domains.batching.weights._tiktoken_count", return_value=20):
        result = estimate_tokens_per_symbols(c, tmp_path, None)
    assert result == math.ceil(20 * HEADROOM)


def test_per_symbols_no_symbols_falls_back_to_path_level(tmp_path: Path) -> None:
    f = tmp_path / "a.py"
    f.write_text("x = 1\n")
    port = _make_tilth_port("def foo(): pass")
    c = FileChange(id="1", path="a.py", edit_kind="modify", symbols=())
    with patch("milknado.domains.batching.weights._tiktoken_count", return_value=20):
        result = estimate_tokens_per_symbols(c, tmp_path, port)
    port.search_symbol.assert_not_called()
    assert result == math.ceil(20 * HEADROOM)


def test_per_symbols_resolves_symbols_via_tilth(tmp_path: Path) -> None:
    slice_text = "def foo():\n    return 42\n"
    port = _make_tilth_port(slice_text)
    c = FileChange(
        id="1", path="src/foo.py", edit_kind="modify",
        symbols=(SymbolRef(name="foo", file="src/foo.py"),),
    )
    result = estimate_tokens_per_symbols(c, tmp_path, port)
    port.search_symbol.assert_called_once_with("foo", glob="src/foo.py")
    assert result > 0
    assert result == math.ceil(
        len(__import__("tiktoken").get_encoding("cl100k_base").encode(slice_text)) * HEADROOM
    )


def test_per_symbols_multiple_symbols_concatenated(tmp_path: Path) -> None:
    fixed_loc = SymbolLocation(path=Path("src/foo.py"), line_start=1, line_end=3)
    port = MagicMock()
    port.search_symbol.return_value = [fixed_loc]
    port.read_section.return_value = "def placeholder(): pass"
    c = FileChange(
        id="1", path="src/foo.py", edit_kind="modify",
        symbols=(
            SymbolRef(name="foo", file="src/foo.py"),
            SymbolRef(name="bar", file="src/foo.py"),
        ),
    )
    result = estimate_tokens_per_symbols(c, tmp_path, port)
    assert port.search_symbol.call_count == 2
    assert result > 0


def test_per_symbols_symbol_not_found_degrades_to_path_level(tmp_path: Path) -> None:
    f = tmp_path / "src" / "foo.py"
    f.parent.mkdir()
    f.write_text("x = 1\n")
    port = MagicMock()
    port.search_symbol.return_value = []  # not found
    c = FileChange(
        id="1", path="src/foo.py", edit_kind="modify",
        symbols=(SymbolRef(name="missing", file="src/foo.py"),),
    )
    with patch("milknado.domains.batching.weights._tiktoken_count", return_value=15):
        result = estimate_tokens_per_symbols(c, tmp_path, port)
    assert result == math.ceil(15 * HEADROOM)
    warn_log = tmp_path / ".milknado" / "planning-context-warn.log"
    assert warn_log.exists()
    assert "missing" in warn_log.read_text()


def test_per_symbols_read_section_error_degrades(tmp_path: Path) -> None:
    port = MagicMock()
    port.search_symbol.return_value = [
        SymbolLocation(path=Path("src/foo.py"), line_start=1, line_end=5)
    ]
    port.read_section.side_effect = RuntimeError("tilth unavailable")
    c = FileChange(
        id="1", path="src/foo.py", edit_kind="modify",
        symbols=(SymbolRef(name="foo", file="src/foo.py"),),
    )
    with patch("milknado.domains.batching.weights._tiktoken_count", return_value=10):
        result = estimate_tokens_per_symbols(c, tmp_path, port)
    assert result == math.ceil(10 * HEADROOM)
    warn_log = tmp_path / ".milknado" / "planning-context-warn.log"
    assert warn_log.exists()


def test_per_symbols_add_uses_heuristic_ignores_tilth(tmp_path: Path) -> None:
    port = _make_tilth_port("should not be called")
    c = FileChange(
        id="1", path="new.py", edit_kind="add",
        symbols=(SymbolRef(name="foo", file="new.py"),),
    )
    result = estimate_tokens_per_symbols(c, tmp_path, port)
    port.search_symbol.assert_not_called()
    assert result == 1875  # 150 * 10 * 1.25


def test_per_symbols_delete_flat_cost_ignores_tilth(tmp_path: Path) -> None:
    port = _make_tilth_port("should not be called")
    c = FileChange(id="1", path="x.py", edit_kind="delete")
    assert estimate_tokens_per_symbols(c, tmp_path, port) == 80
    port.search_symbol.assert_not_called()
