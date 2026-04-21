from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

from milknado.adapters.tilth import TilthAdapter
from milknado.domains.common.protocols import SymbolLocation
from milknado.domains.common.types import DegradationMarker, TilthMap


def _ok(stdout: str) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=stdout,
        stderr="",
    )


def _fail(stderr: str, code: int = 1) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=[],
        returncode=code,
        stdout="",
        stderr=stderr,
    )


class TestStructuralMap:
    @patch("milknado.adapters.tilth.shutil.which", return_value=None)
    def test_returns_marker_when_binary_missing(
        self,
        _which: MagicMock,
        tmp_path: Path,
    ) -> None:
        adapter = TilthAdapter()
        result = adapter.structural_map(tmp_path, budget_tokens=2000)
        assert isinstance(result, DegradationMarker)
        assert result.source == "tilth"
        assert result.reason == "binary_missing"

    @patch("milknado.adapters.tilth.subprocess.run")
    @patch("milknado.adapters.tilth.shutil.which", return_value="/usr/bin/tilth")
    def test_returns_map_on_success(
        self,
        _which: MagicMock,
        mock_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        payload = {"files": ["a.py"], "symbols": {}}
        mock_run.return_value = _ok(json.dumps(payload))
        adapter = TilthAdapter()

        result = adapter.structural_map(tmp_path, budget_tokens=1500)

        assert isinstance(result, TilthMap)
        assert result.scope == tmp_path
        assert result.budget_tokens == 1500
        assert result.data == payload
        argv = mock_run.call_args[0][0]
        assert argv[0] == "tilth"
        assert "--map" in argv and "--json" in argv
        assert "--scope" in argv and str(tmp_path) in argv
        assert "--budget" in argv and "1500" in argv

    @patch("milknado.adapters.tilth.subprocess.run")
    @patch("milknado.adapters.tilth.shutil.which", return_value="/usr/bin/tilth")
    def test_returns_marker_on_nonzero_exit(
        self,
        _which: MagicMock,
        mock_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_run.return_value = _fail("boom happened")
        adapter = TilthAdapter()

        result = adapter.structural_map(tmp_path, budget_tokens=100)

        assert isinstance(result, DegradationMarker)
        assert result.reason == "exec_failed"
        assert "boom happened" in result.detail

    @patch("milknado.adapters.tilth.subprocess.run")
    @patch("milknado.adapters.tilth.shutil.which", return_value="/usr/bin/tilth")
    def test_returns_marker_on_invalid_json(
        self,
        _which: MagicMock,
        mock_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_run.return_value = _ok("not-json")
        adapter = TilthAdapter()

        result = adapter.structural_map(tmp_path, budget_tokens=100)

        assert isinstance(result, DegradationMarker)
        assert result.reason == "invalid_json"

    @patch("milknado.adapters.tilth.subprocess.run")
    @patch("milknado.adapters.tilth.shutil.which", return_value="/usr/bin/tilth")
    def test_returns_marker_on_timeout(
        self,
        _which: MagicMock,
        mock_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="tilth", timeout=30)
        adapter = TilthAdapter()

        result = adapter.structural_map(tmp_path, budget_tokens=100)

        assert isinstance(result, DegradationMarker)
        assert result.reason == "exec_failed"
        assert "timed out" in result.detail

    @patch("milknado.adapters.tilth.subprocess.run")
    @patch("milknado.adapters.tilth.shutil.which", return_value="/usr/bin/tilth")
    def test_returns_marker_when_top_level_not_object(
        self,
        _which: MagicMock,
        mock_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_run.return_value = _ok(json.dumps([1, 2, 3]))
        adapter = TilthAdapter()

        result = adapter.structural_map(tmp_path, budget_tokens=100)

        assert isinstance(result, DegradationMarker)
        assert result.reason == "invalid_json"
        assert "top-level" in result.detail


class TestSearchSymbol:
    @patch("milknado.adapters.tilth.shutil.which", return_value=None)
    def test_returns_empty_when_binary_missing(self, _which: MagicMock) -> None:
        assert TilthAdapter().search_symbol("MyClass") == []

    @patch("milknado.adapters.tilth.subprocess.run")
    @patch("milknado.adapters.tilth.shutil.which", return_value="/usr/bin/tilth")
    def test_returns_locations_on_success(
        self,
        _which: MagicMock,
        mock_run: MagicMock,
    ) -> None:
        output_text = (
            "## src/foo.py:10-20 [definition]\n"
            "→ [10]   class MyClass:\n"
            "## src/bar.py:5-8 [usage]\n"
            "→ [5]   MyClass()\n"
        )
        mock_run.return_value = _ok(json.dumps({"output": output_text, "query": "MyClass"}))

        result = TilthAdapter().search_symbol("MyClass")

        assert result == [
            SymbolLocation(path=Path("src/foo.py"), line_start=10, line_end=20),
            SymbolLocation(path=Path("src/bar.py"), line_start=5, line_end=8),
        ]
        argv = mock_run.call_args[0][0]
        assert argv[0] == "tilth"
        assert "MyClass" in argv
        assert "--json" in argv
        assert "--search" not in argv

    @patch("milknado.adapters.tilth.subprocess.run")
    @patch("milknado.adapters.tilth.shutil.which", return_value="/usr/bin/tilth")
    def test_passes_glob_filter(
        self,
        _which: MagicMock,
        mock_run: MagicMock,
    ) -> None:
        output_text = "## src/foo.py:1-5 [definition]\n"
        mock_run.return_value = _ok(json.dumps({"output": output_text, "query": "Foo"}))

        TilthAdapter().search_symbol("Foo", glob="src/**/*.py")

        argv = mock_run.call_args[0][0]
        assert "--glob" in argv
        assert "src/**/*.py" in argv

    @patch("milknado.adapters.tilth.subprocess.run")
    @patch("milknado.adapters.tilth.shutil.which", return_value="/usr/bin/tilth")
    def test_single_line_match_sets_end_equal_start(
        self,
        _which: MagicMock,
        mock_run: MagicMock,
    ) -> None:
        output_text = "## src/mod.py:42 [usage]\n→ [42]   import MyClass\n"
        mock_run.return_value = _ok(json.dumps({"output": output_text, "query": "MyClass"}))

        result = TilthAdapter().search_symbol("MyClass")

        assert result == [SymbolLocation(path=Path("src/mod.py"), line_start=42, line_end=42)]

    @patch("milknado.adapters.tilth.subprocess.run")
    @patch("milknado.adapters.tilth.shutil.which", return_value="/usr/bin/tilth")
    def test_skips_non_header_lines(
        self,
        _which: MagicMock,
        mock_run: MagicMock,
    ) -> None:
        output_text = (
            '# Search: "fn" — 2 matches\n'
            "## src/ok.py:1-5 [definition]\n"
            "  some detail line without header format\n"
        )
        mock_run.return_value = _ok(json.dumps({"output": output_text, "query": "fn"}))

        result = TilthAdapter().search_symbol("fn")

        assert len(result) == 1
        assert result[0].path == Path("src/ok.py")

    @patch("milknado.adapters.tilth.subprocess.run")
    @patch("milknado.adapters.tilth.shutil.which", return_value="/usr/bin/tilth")
    def test_returns_empty_on_nonzero_exit(
        self,
        _which: MagicMock,
        mock_run: MagicMock,
    ) -> None:
        mock_run.return_value = _fail("error")
        assert TilthAdapter().search_symbol("x") == []

    @patch("milknado.adapters.tilth.subprocess.run")
    @patch("milknado.adapters.tilth.shutil.which", return_value="/usr/bin/tilth")
    def test_returns_empty_on_invalid_json(
        self,
        _which: MagicMock,
        mock_run: MagicMock,
    ) -> None:
        mock_run.return_value = _ok("not-json")
        assert TilthAdapter().search_symbol("x") == []

    @patch("milknado.adapters.tilth.subprocess.run")
    @patch("milknado.adapters.tilth.shutil.which", return_value="/usr/bin/tilth")
    def test_returns_empty_on_timeout(
        self,
        _which: MagicMock,
        mock_run: MagicMock,
    ) -> None:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="tilth", timeout=30)
        assert TilthAdapter().search_symbol("x") == []

    @patch("milknado.adapters.tilth.subprocess.run")
    @patch("milknado.adapters.tilth.shutil.which", return_value="/usr/bin/tilth")
    def test_returns_empty_when_output_not_a_dict(
        self,
        _which: MagicMock,
        mock_run: MagicMock,
    ) -> None:
        mock_run.return_value = _ok(json.dumps([1, 2, 3]))
        assert TilthAdapter().search_symbol("x") == []


class TestReadSection:
    @patch("milknado.adapters.tilth.shutil.which", return_value=None)
    def test_returns_empty_when_binary_missing(
        self,
        _which: MagicMock,
        tmp_path: Path,
    ) -> None:
        assert TilthAdapter().read_section(tmp_path / "f.py", 1, 10) == ""

    @patch("milknado.adapters.tilth.subprocess.run")
    @patch("milknado.adapters.tilth.shutil.which", return_value="/usr/bin/tilth")
    def test_returns_content_on_success(
        self,
        _which: MagicMock,
        mock_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_run.return_value = _ok("def foo():\n    pass\n")
        target = tmp_path / "src.py"

        result = TilthAdapter().read_section(target, 3, 7)

        assert result == "def foo():\n    pass\n"
        argv = mock_run.call_args[0][0]
        assert argv[0] == "tilth"
        assert str(target) in argv
        assert "--section" in argv and "3-7" in argv
        assert "--read" not in argv

    @patch("milknado.adapters.tilth.subprocess.run")
    @patch("milknado.adapters.tilth.shutil.which", return_value="/usr/bin/tilth")
    def test_returns_empty_on_nonzero_exit(
        self,
        _which: MagicMock,
        mock_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_run.return_value = _fail("oops")
        assert TilthAdapter().read_section(tmp_path / "f.py", 1, 5) == ""

    @patch("milknado.adapters.tilth.subprocess.run")
    @patch("milknado.adapters.tilth.shutil.which", return_value="/usr/bin/tilth")
    def test_returns_empty_on_timeout(
        self,
        _which: MagicMock,
        mock_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="tilth", timeout=30)
        assert TilthAdapter().read_section(tmp_path / "f.py", 1, 5) == ""
