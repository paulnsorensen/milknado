from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

from milknado.adapters.tilth import TilthAdapter
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
