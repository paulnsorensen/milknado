"""Targeted tests for coverage gaps in domains/dispatch/runner.py.

Covers:
- _resolve_worker_cmd explicit cmd branch (line 75)
- _inject_mcp_config — claude binary + .mcp.json present vs absent
- fail_stale_running_runs exception / missing-field skip paths (lines 307-318)
- find_terminal_runs_for_node corrupt-file skip path (lines 343-344)
- reconcile_node_status early return when node absent or not RUNNING (line 361)
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from milknado.domains.dispatch.runner import (
    _inject_mcp_config,
    _resolve_worker_cmd,
    fail_stale_running_runs,
    find_terminal_runs_for_node,
    reconcile_node_status,
)
from milknado.domains.graph import MikadoGraph

# ---------------------------------------------------------------------------
# _resolve_worker_cmd (line 75 — explicit non-empty cmd)
# ---------------------------------------------------------------------------


class TestResolveWorkerCmd:
    def test_explicit_cmd_is_split_and_returned(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When an explicit, non-empty cmd is passed it is shlex-split and returned."""
        monkeypatch.delenv("MILKNADO_WORKER_CMD", raising=False)
        result = _resolve_worker_cmd("claude --model sonnet")
        assert result[0] == "claude"
        assert "--model" in result
        assert "sonnet" in result

    def test_explicit_whitespace_only_falls_back_to_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Whitespace-only explicit is treated as absent; falls back to env var."""
        monkeypatch.setenv("MILKNADO_WORKER_CMD", "claude --version")
        result = _resolve_worker_cmd("   ")
        assert result[0] == "claude"


# ---------------------------------------------------------------------------
# _inject_mcp_config
# ---------------------------------------------------------------------------


class TestInjectMcpConfig:
    def test_injects_when_claude_binary_and_mcp_json_present(self, tmp_path: Path) -> None:
        mcp = tmp_path / ".mcp.json"
        mcp.write_text("{}")
        argv = ["claude", "run"]
        result = _inject_mcp_config(argv, tmp_path)
        assert "--mcp-config" in result
        assert str(mcp) in result

    def test_does_not_inject_when_no_mcp_json(self, tmp_path: Path) -> None:
        argv = ["claude", "run"]
        result = _inject_mcp_config(argv, tmp_path)
        assert "--mcp-config" not in result
        assert result == argv

    def test_does_not_inject_for_non_claude_binary(self, tmp_path: Path) -> None:
        mcp = tmp_path / ".mcp.json"
        mcp.write_text("{}")
        argv = ["/usr/bin/python3", "worker.py"]
        result = _inject_mcp_config(argv, tmp_path)
        assert "--mcp-config" not in result
        assert result == argv


# ---------------------------------------------------------------------------
# fail_stale_running_runs — exception / missing-field skip paths
# ---------------------------------------------------------------------------


def _runs_dir(project_root: Path) -> Path:
    from milknado.domains.dispatch.runner import _runs_dir as _rd

    return _rd(project_root)


class TestFailStaleRunningRunsEdgePaths:
    def _write_state(self, path: Path, data: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))

    def test_skips_corrupt_state_file(self, tmp_path: Path) -> None:
        """A state file with invalid JSON must be skipped, not crash."""
        rd = _runs_dir(tmp_path)
        rd.mkdir(parents=True, exist_ok=True)
        bad = rd / "node-10-20200101T000000Z-dead.state.json"
        bad.write_text("{not valid json")

        result = fail_stale_running_runs(tmp_path, 10)
        assert result == []

    def test_skips_run_without_started_at(self, tmp_path: Path) -> None:
        """A 'running' state missing started_at is skipped (cannot compute staleness)."""
        rd = _runs_dir(tmp_path)
        rd.mkdir(parents=True, exist_ok=True)
        state_path = rd / "node-11-20200101T000000Z-abcd.state.json"
        self._write_state(
            state_path,
            {
                "run_id": "node-11-20200101T000000Z-abcd",
                "node_id": 11,
                "status": "running",
                "timeout_seconds": 10,
                # started_at intentionally missing
                "exit_code": None,
                "timed_out": False,
                "ended_at": None,
                "log_path": str(rd / "node-11-20200101T000000Z-abcd.log"),
            },
        )
        result = fail_stale_running_runs(tmp_path, 11)
        assert result == []

    def test_skips_run_without_timeout_seconds(self, tmp_path: Path) -> None:
        """A 'running' state missing timeout_seconds is skipped."""
        rd = _runs_dir(tmp_path)
        rd.mkdir(parents=True, exist_ok=True)
        state_path = rd / "node-12-20200101T000000Z-beef.state.json"
        stale = (datetime.now(UTC) - timedelta(seconds=600)).isoformat()
        self._write_state(
            state_path,
            {
                "run_id": "node-12-20200101T000000Z-beef",
                "node_id": 12,
                "status": "running",
                "started_at": stale,
                # timeout_seconds intentionally missing
                "exit_code": None,
                "timed_out": False,
                "ended_at": None,
                "log_path": str(rd / "node-12-20200101T000000Z-beef.log"),
            },
        )
        result = fail_stale_running_runs(tmp_path, 12)
        assert result == []

    def test_skips_run_with_invalid_started_at_format(self, tmp_path: Path) -> None:
        """A 'running' state with a non-ISO started_at is skipped."""
        rd = _runs_dir(tmp_path)
        rd.mkdir(parents=True, exist_ok=True)
        state_path = rd / "node-13-20200101T000000Z-cafe.state.json"
        self._write_state(
            state_path,
            {
                "run_id": "node-13-20200101T000000Z-cafe",
                "node_id": 13,
                "status": "running",
                "started_at": "not-a-date",
                "timeout_seconds": 10,
                "exit_code": None,
                "timed_out": False,
                "ended_at": None,
                "log_path": str(rd / "node-13-20200101T000000Z-cafe.log"),
            },
        )
        result = fail_stale_running_runs(tmp_path, 13)
        assert result == []


# ---------------------------------------------------------------------------
# find_terminal_runs_for_node — corrupt state file skip path (lines 343-344)
# ---------------------------------------------------------------------------


class TestFindTerminalRunsForNode:
    def test_skips_corrupt_state_file(self, tmp_path: Path) -> None:
        """A corrupt state file must be skipped; result is still returned."""
        rd = _runs_dir(tmp_path)
        rd.mkdir(parents=True, exist_ok=True)
        bad = rd / "node-20-20200101T000000Z-dead.state.json"
        bad.write_text("{{bad json")

        result = find_terminal_runs_for_node(tmp_path, 20)
        assert result == []

    def test_returns_terminal_runs_ignoring_corrupt_siblings(self, tmp_path: Path) -> None:
        """Corrupt file is skipped; a valid terminal file is still returned."""
        rd = _runs_dir(tmp_path)
        rd.mkdir(parents=True, exist_ok=True)

        bad = rd / "node-21-20200101T000000Z-bad0.state.json"
        bad.write_text("{invalid}")

        good = rd / "node-21-20200101T000000Z-good.state.json"
        good.write_text(
            json.dumps(
                {
                    "run_id": "node-21-20200101T000000Z-good",
                    "node_id": 21,
                    "status": "done",
                    "exit_code": 0,
                }
            )
        )

        result = find_terminal_runs_for_node(tmp_path, 21)
        assert len(result) == 1
        assert result[0]["status"] == "done"


# ---------------------------------------------------------------------------
# reconcile_node_status — early return when node absent or not RUNNING (line 361)
# ---------------------------------------------------------------------------


class TestReconcileNodeStatus:
    def test_no_op_when_node_does_not_exist(self, graph: MikadoGraph) -> None:
        """reconcile_node_status must be a no-op when the node is absent."""
        # Should not raise
        reconcile_node_status(graph, 999, "done")

    def test_no_op_when_node_is_not_running(self, graph: MikadoGraph) -> None:
        """reconcile_node_status must not transition a node that isn't RUNNING."""
        from milknado.domains.common.types import NodeStatus

        graph.add_node("task")
        # Node is PENDING, not RUNNING
        reconcile_node_status(graph, 1, "done")
        node = graph.get_node(1)
        assert node is not None
        assert node.status == NodeStatus.PENDING

    def test_transitions_running_node_to_done(self, graph: MikadoGraph) -> None:
        """A RUNNING node with run_status='done' must be marked DONE."""
        from milknado.domains.common.types import NodeStatus

        graph.add_node("task")
        graph.mark_running(1)
        reconcile_node_status(graph, 1, "done")
        node = graph.get_node(1)
        assert node is not None
        assert node.status == NodeStatus.DONE

    def test_transitions_running_node_to_failed(self, graph: MikadoGraph) -> None:
        """A RUNNING node with run_status='failed' must be marked FAILED."""
        from milknado.domains.common.types import NodeStatus

        graph.add_node("task")
        graph.mark_running(1)
        reconcile_node_status(graph, 1, "failed")
        node = graph.get_node(1)
        assert node is not None
        assert node.status == NodeStatus.FAILED


# ---------------------------------------------------------------------------
# _async_worker OSError on log-write (lines 223-227)
# ---------------------------------------------------------------------------


class TestAsyncWorkerLogWriteFailure:
    def test_log_write_oserror_is_swallowed_and_state_still_written(self, tmp_path: Path) -> None:
        """When _async_worker's log-append fails with OSError the state file still
        gets written as 'failed' — the OSError from the log write is suppressed."""
        from milknado.domains.dispatch.runner import _async_worker, _runs_dir

        rd = _runs_dir(tmp_path)
        rd.mkdir(parents=True, exist_ok=True)

        # Use a directory as log_path — opening it for append raises OSError
        log_path = rd / "node-50"
        log_path.mkdir()  # directory, not a file → open("a") raises IsADirectoryError

        state_path = rd / "node-50-20200101T000000Z-abcd.state.json"
        base_state = {
            "run_id": "node-50-20200101T000000Z-abcd",
            "node_id": 50,
            "status": "running",
            "started_at": datetime.now(UTC).isoformat(),
            "log_path": str(log_path),
            "timeout_seconds": 1,
            "exit_code": None,
            "timed_out": False,
            "ended_at": None,
        }
        # Write initial running state
        state_path.write_text(json.dumps(base_state))

        # argv that doesn't exist → _execute raises FileNotFoundError → enters except block
        # log_path is a directory → log_fh open raises OSError → except OSError: pass fires
        _async_worker(
            state_path=state_path,
            project_root=tmp_path,
            log_path=log_path,
            brief="test brief",
            argv=["/nonexistent/binary/that/does/not/exist"],
            timeout=1,
            base_state=base_state,
        )

        # State must be written as failed despite the log-write OSError
        state = json.loads(state_path.read_text())
        assert state["status"] == "failed"
        assert state["exit_code"] == -1
