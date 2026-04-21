from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

from milknado.adapters.tilth import TilthAdapter, _parse_symbol_headers, _run_tilth_json
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


class TestRunTilthJson:
    @patch("milknado.adapters.tilth.subprocess.run")
    def test_returns_dict_on_success(self, mock_run: MagicMock) -> None:
        data = {"key": "value"}
        mock_run.return_value = _ok(json.dumps(data))
        result = _run_tilth_json(["tilth", "search", "--json"])
        assert result == data

    @patch("milknado.adapters.tilth.subprocess.run")
    def test_returns_none_on_nonzero_exit(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _fail("error")
        assert _run_tilth_json(["tilth", "search"]) is None

    @patch("milknado.adapters.tilth.subprocess.run")
    def test_returns_none_on_invalid_json(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _ok("not-json")
        assert _run_tilth_json(["tilth", "search"]) is None

    @patch("milknado.adapters.tilth.subprocess.run")
    def test_returns_none_on_timeout(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="tilth", timeout=30)
        assert _run_tilth_json(["tilth", "search"]) is None

    @patch("milknado.adapters.tilth.subprocess.run")
    def test_returns_none_when_not_dict(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _ok(json.dumps([1, 2]))
        assert _run_tilth_json(["tilth", "search"]) is None


class TestParseSymbolHeaders:
    def test_parses_single_header(self) -> None:
        output = "## src/foo.py:10-20 [definition]\nsome content"
        result = _parse_symbol_headers(output)
        assert len(result) == 1
        assert result[0].path == Path("src/foo.py")
        assert result[0].line_start == 10
        assert result[0].line_end == 20

    def test_parses_multiple_headers(self) -> None:
        output = "## src/a.py:1-5 [definition]\n## src/b.py:10-15 [usage]\n"
        result = _parse_symbol_headers(output)
        assert len(result) == 2
        assert result[0].path == Path("src/a.py")
        assert result[1].path == Path("src/b.py")

    def test_handles_line_without_end(self) -> None:
        output = "## src/foo.py:42 [definition]"
        result = _parse_symbol_headers(output)
        assert len(result) == 1
        assert result[0].line_start == 42
        assert result[0].line_end == 42

    def test_skips_non_matching_lines(self) -> None:
        output = "no headers here\njust text"
        result = _parse_symbol_headers(output)
        assert result == []

    def test_empty_output_returns_empty_list(self) -> None:
        assert _parse_symbol_headers("") == []


class TestSearchSymbol:
    @patch("milknado.adapters.tilth.shutil.which", return_value=None)
    def test_returns_empty_when_binary_missing(self, _which: MagicMock) -> None:
        adapter = TilthAdapter()
        assert adapter.search_symbol("MyClass") == []

    @patch("milknado.adapters.tilth.subprocess.run")
    @patch("milknado.adapters.tilth.shutil.which", return_value="/usr/bin/tilth")
    def test_returns_locations_on_success(
        self, _which: MagicMock, mock_run: MagicMock
    ) -> None:
        output_text = "## src/foo.py:5-15 [definition]\ncode here"
        payload = {"output": output_text}
        mock_run.return_value = _ok(json.dumps(payload))

        adapter = TilthAdapter()
        result = adapter.search_symbol("MyClass")

        assert len(result) == 1
        assert result[0].path == Path("src/foo.py")
        assert result[0].line_start == 5

    @patch("milknado.adapters.tilth.subprocess.run")
    @patch("milknado.adapters.tilth.shutil.which", return_value="/usr/bin/tilth")
    def test_passes_glob_when_provided(
        self, _which: MagicMock, mock_run: MagicMock
    ) -> None:
        mock_run.return_value = _ok(json.dumps({"output": ""}))
        TilthAdapter().search_symbol("Foo", glob="*.py")
        argv = mock_run.call_args[0][0]
        assert "--glob" in argv
        assert "*.py" in argv

    @patch("milknado.adapters.tilth.subprocess.run")
    @patch("milknado.adapters.tilth.shutil.which", return_value="/usr/bin/tilth")
    def test_returns_empty_on_invalid_json(
        self, _which: MagicMock, mock_run: MagicMock
    ) -> None:
        mock_run.return_value = _ok("not-json")
        assert TilthAdapter().search_symbol("Foo") == []

    @patch("milknado.adapters.tilth.subprocess.run")
    @patch("milknado.adapters.tilth.shutil.which", return_value="/usr/bin/tilth")
    def test_returns_empty_on_timeout(
        self, _which: MagicMock, mock_run: MagicMock
    ) -> None:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="tilth", timeout=30)
        assert TilthAdapter().search_symbol("Foo") == []

    @patch("milknado.adapters.tilth.subprocess.run")
    @patch("milknado.adapters.tilth.shutil.which", return_value="/usr/bin/tilth")
    def test_returns_empty_when_output_not_string(
        self, _which: MagicMock, mock_run: MagicMock
    ) -> None:
        mock_run.return_value = _ok(json.dumps({"output": 42}))
        assert TilthAdapter().search_symbol("Foo") == []

    @patch("milknado.adapters.tilth.subprocess.run")
    @patch("milknado.adapters.tilth.shutil.which", return_value="/usr/bin/tilth")
    def test_handles_malformed_header_lines(
        self, _which: MagicMock, mock_run: MagicMock
    ) -> None:
        output_text = "## src/foo.py:notanumber [definition]"
        mock_run.return_value = _ok(json.dumps({"output": output_text}))
        result = TilthAdapter().search_symbol("Foo")
        assert result == []


class TestReadSection:
    @patch("milknado.adapters.tilth.shutil.which", return_value=None)
    def test_returns_empty_when_binary_missing(self, _which: MagicMock) -> None:
        assert TilthAdapter().read_section(Path("foo.py"), 1, 10) == ""

    @patch("milknado.adapters.tilth.subprocess.run")
    @patch("milknado.adapters.tilth.shutil.which", return_value="/usr/bin/tilth")
    def test_returns_stdout_on_success(
        self, _which: MagicMock, mock_run: MagicMock
    ) -> None:
        mock_run.return_value = _ok("def foo():\n    pass\n")
        result = TilthAdapter().read_section(Path("src/foo.py"), 1, 5)
        assert result == "def foo():\n    pass\n"
        argv = mock_run.call_args[0][0]
        assert "tilth" in argv
        assert "--section" in argv
        assert "1-5" in argv

    @patch("milknado.adapters.tilth.subprocess.run")
    @patch("milknado.adapters.tilth.shutil.which", return_value="/usr/bin/tilth")
    def test_returns_empty_on_nonzero_exit(
        self, _which: MagicMock, mock_run: MagicMock
    ) -> None:
        mock_run.return_value = _fail("error", code=1)
        assert TilthAdapter().read_section(Path("foo.py"), 1, 5) == ""

    @patch("milknado.adapters.tilth.subprocess.run")
    @patch("milknado.adapters.tilth.shutil.which", return_value="/usr/bin/tilth")
    def test_returns_empty_on_timeout(
        self, _which: MagicMock, mock_run: MagicMock
    ) -> None:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="tilth", timeout=30)
        assert TilthAdapter().read_section(Path("foo.py"), 1, 5) == ""
