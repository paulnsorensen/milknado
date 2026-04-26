from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock

import tiktoken

from milknado.domains.batching.change import FileChange, SymbolRef
from milknado.domains.batching.solver import _tokens_per_scc
from milknado.domains.batching.weights import (
    HEADROOM,
    NEW_FILE_LINES,
    TOKENS_PER_LINE,
    estimate_tokens_per_symbols,
)
from milknado.domains.common.protocols import SymbolLocation

_ENC = tiktoken.get_encoding("cl100k_base")


def _tilth(*, symbol_text: str, line_start: int = 1, line_end: int = 5) -> MagicMock:
    port = MagicMock()
    port.search_symbol.return_value = [
        SymbolLocation(path=Path("src/foo.py"), line_start=line_start, line_end=line_end)
    ]
    port.read_section.return_value = symbol_text
    return port


def test_single_symbol_uses_tiktoken_slice(tmp_path: Path) -> None:
    text = "def process(x):\n    return x * 2\n"
    port = _tilth(symbol_text=text)
    change = FileChange(
        id="c1",
        path="src/foo.py",
        edit_kind="modify",
        symbols=(SymbolRef(name="process", file="src/foo.py"),),
    )
    result = estimate_tokens_per_symbols(change, tmp_path, port)

    port.search_symbol.assert_called_once_with("process", glob="src/foo.py")
    expected = math.ceil(len(_ENC.encode(text)) * HEADROOM)
    assert result == expected


def test_multi_symbol_slices_concatenated(tmp_path: Path) -> None:
    slice_a = "def foo():\n    pass\n"
    slice_b = "def bar():\n    pass\n"
    port = MagicMock()
    port.search_symbol.return_value = [
        SymbolLocation(path=Path("src/foo.py"), line_start=1, line_end=3)
    ]
    port.read_section.side_effect = [slice_a, slice_b]

    change = FileChange(
        id="c1",
        path="src/foo.py",
        edit_kind="modify",
        symbols=(
            SymbolRef(name="foo", file="src/foo.py"),
            SymbolRef(name="bar", file="src/foo.py"),
        ),
    )
    result = estimate_tokens_per_symbols(change, tmp_path, port)

    assert port.search_symbol.call_count == 2
    combined = slice_a + "\n" + slice_b
    expected = math.ceil(len(_ENC.encode(combined)) * HEADROOM)
    assert result == expected


def test_shared_symbol_deduped_across_changes(tmp_path: Path) -> None:
    """_tokens_per_scc must not double-count a (file, name) seen in a prior change."""
    slice_text = "def shared():\n    pass\n"
    port = MagicMock()
    port.search_symbol.return_value = [
        SymbolLocation(path=Path("src/foo.py"), line_start=1, line_end=3)
    ]
    port.read_section.return_value = slice_text

    changes = [
        FileChange(
            id="c1",
            path="src/foo.py",
            edit_kind="modify",
            symbols=(SymbolRef(name="shared", file="src/foo.py"),),
        ),
        FileChange(
            id="c2",
            path="src/foo.py",
            edit_kind="modify",
            symbols=(SymbolRef(name="shared", file="src/foo.py"),),
        ),
    ]
    scc_members = {"s0": ["c1", "c2"]}
    sccs = ["s0"]

    result = _tokens_per_scc(changes, scc_members, sccs, tmp_path, port)

    # shared symbol only fetched once
    assert port.search_symbol.call_count == 1
    assert port.read_section.call_count == 1
    single_cost = math.ceil(len(_ENC.encode(slice_text)) * HEADROOM)
    assert result["s0"] == single_cost


def test_tilth_degradation_falls_back_to_path_dedup(tmp_path: Path) -> None:
    """With tilth_port=None, _tokens_per_scc deduplicates by path, not symbol."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "foo.py").write_text("x = 1\n")

    changes = [
        FileChange(
            id="c1",
            path="src/foo.py",
            edit_kind="modify",
            symbols=(SymbolRef(name="x", file="src/foo.py"),),
        ),
        FileChange(
            id="c2",
            path="src/foo.py",
            edit_kind="modify",
            symbols=(SymbolRef(name="y", file="src/foo.py"),),
        ),
    ]
    scc_members = {"s0": ["c1", "c2"]}
    sccs = ["s0"]

    result = _tokens_per_scc(changes, scc_members, sccs, tmp_path, None)

    # path seen after c1, c2 is skipped — total is only one file's cost
    single_cost = estimate_tokens_per_symbols(changes[0], tmp_path, None)
    assert result["s0"] == single_cost


def test_edit_kind_add_uses_new_file_lines_heuristic(tmp_path: Path) -> None:
    port = _tilth(symbol_text="should not be called")
    change = FileChange(
        id="c1",
        path="new_module.py",
        edit_kind="add",
        symbols=(SymbolRef(name="anything", file="new_module.py"),),
    )
    result = estimate_tokens_per_symbols(change, tmp_path, port)

    port.search_symbol.assert_not_called()
    expected = math.ceil(NEW_FILE_LINES["py"] * TOKENS_PER_LINE["py"] * HEADROOM)
    assert result == expected


def test_edit_kind_delete_flat_cost_ignores_symbols(tmp_path: Path) -> None:
    port = _tilth(symbol_text="should not be called")
    change = FileChange(
        id="c1",
        path="old.py",
        edit_kind="delete",
        symbols=(SymbolRef(name="anything", file="old.py"),),
    )
    result = estimate_tokens_per_symbols(change, tmp_path, port)

    port.search_symbol.assert_not_called()
    assert result == 80
