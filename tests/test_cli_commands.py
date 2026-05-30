"""Tests for cli.py command bodies surfaced by the per-command split.

Covers the CRG-build commands (``index``/``crg``), the ``init`` CRG-failure
branch, and the best-effort ``_fetch_run_states`` status enrichment.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from milknado.cli import _fetch_run_states, app
from milknado.domains.common.types import MikadoNode, NodeStatus

runner = CliRunner()


def _running_node(node_id: int, run_id: str | None) -> MikadoNode:
    return MikadoNode(
        id=node_id,
        description=f"task {node_id}",
        parent_id=None,
        status=NodeStatus.RUNNING,
        run_id=run_id,
    )


class TestFetchRunStates:
    def test_returns_none_without_running_run_ids(self) -> None:
        nodes = [_running_node(1, None)]
        assert _fetch_run_states(nodes) is None

    def test_collects_status_per_running_run(self) -> None:
        nodes = [_running_node(1, "run-1"), _running_node(2, "run-2")]
        with patch("milknado.adapters.ralphify.RalphifyAdapter") as ralph_cls:
            ralph = ralph_cls.return_value
            ralph.get_run.side_effect = lambda rid: MagicMock(status=f"{rid}-status")
            states = _fetch_run_states(nodes)
        assert states == {"run-1": "run-1-status", "run-2": "run-2-status"}

    def test_swallows_adapter_errors(self) -> None:
        nodes = [_running_node(1, "run-1")]
        with patch("milknado.adapters.ralphify.RalphifyAdapter", side_effect=RuntimeError):
            assert _fetch_run_states(nodes) is None


class TestIndexCommand:
    def test_rebuilds_graph(self, tmp_path: Path) -> None:
        with (
            patch("milknado.cli._load_or_default", return_value=(MagicMock(), [])),
            patch("milknado.adapters.crg.CrgAdapter") as crg_cls,
        ):
            result = runner.invoke(app, ["index", str(tmp_path)])
        assert result.exit_code == 0
        assert "rebuilt" in result.output
        crg_cls.return_value.build_graph.assert_called_once()

    def test_reports_build_failure(self, tmp_path: Path) -> None:
        with (
            patch("milknado.cli._load_or_default", return_value=(MagicMock(), [])),
            patch("milknado.adapters.crg.CrgAdapter") as crg_cls,
        ):
            crg_cls.return_value.build_graph.side_effect = subprocess.CalledProcessError(
                returncode=2, cmd="crg", stderr="boom"
            )
            result = runner.invoke(app, ["index", str(tmp_path)])
        assert result.exit_code == 1
        assert "CRG build failed" in result.output
        assert "boom" in result.output


class TestCrgCommand:
    def test_exits_when_no_index(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["crg", str(tmp_path)])
        assert result.exit_code == 1
        assert "No CRG index" in result.output

    def test_prints_architecture_overview(self, tmp_path: Path) -> None:
        db = tmp_path / ".code-review-graph" / "graph.db"
        db.parent.mkdir(parents=True)
        db.touch()
        with patch("milknado.adapters.crg.CrgAdapter") as crg_cls:
            crg_cls.return_value.get_architecture_overview.return_value = {"nodes": 3}
            result = runner.invoke(app, ["crg", str(tmp_path)])
        assert result.exit_code == 0
        assert '"nodes": 3' in result.output


class TestInitCrgFailure:
    def test_reports_crg_failure_and_exits(self, tmp_path: Path) -> None:
        with patch("milknado.adapters.crg.CrgAdapter") as crg_cls:
            crg_cls.return_value.ensure_graph.side_effect = subprocess.CalledProcessError(
                returncode=3, cmd="crg", stderr="index exploded"
            )
            result = runner.invoke(app, ["init", str(tmp_path)])
        assert result.exit_code == 1
        assert "CRG build failed" in result.output
        assert "index exploded" in result.output
