"""Tests for milknado_deposit_review — the structured review-verdict deposit seam."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from milknado.domains.common import RunResult
from milknado.mcp.run import milknado_deposit_review
from milknado.mcp.server import open_graph


def _call(tool, **kwargs):
    fn = getattr(tool, "fn", tool)
    return fn(**kwargs)


def _seed_run(root: Path, *, run_id: str, node_id: int, status: str = "running") -> None:
    graph, _cfg = open_graph(root)
    try:
        graph._conn.execute(
            "INSERT OR IGNORE INTO nodes (id, description, status, created_at) "
            "VALUES (?, ?, 'running', ?)",
            (node_id, f"seeded-{node_id}", datetime.now(UTC).isoformat()),
        )
        started_at = datetime.now(UTC).isoformat()
        log_path = str(root / ".milknado" / "runs" / f"{run_id}.log")
        graph.start_run(run_id, node_id, log_path, started_at, 10, None)
        if status != "running":
            graph.finish_run(
                run_id,
                RunResult(
                    status=status,
                    exit_code=0,
                    timed_out=False,
                    ended_at=datetime.now(UTC).isoformat(),
                    error=None,
                    rebased=False,
                    detail=None,
                ),
            )
    finally:
        graph.close()


class TestDepositReview:
    def test_deposit_rejects_malformed_run_id(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="invalid run_id"):
            _call(
                milknado_deposit_review,
                run_id="../etc/passwd",
                verdict="approve",
                findings_md="looks good",
                project_root=str(tmp_path),
            )

    def test_deposit_rejects_bad_verdict(self, tmp_path: Path) -> None:
        """A verdict outside approve|reject must fail fast, not be silently accepted
        or coerced — the spec requires exactly the two-valued enum."""
        run_id = "node-1-20260101T000000Z-abcd"
        _seed_run(tmp_path, run_id=run_id, node_id=1, status="running")
        with pytest.raises(ValueError, match="invalid verdict"):
            _call(
                milknado_deposit_review,
                run_id=run_id,
                verdict="maybe",
                findings_md="unclear",
                project_root=str(tmp_path),
            )

    def test_deposit_unknown_run_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="not found"):
            _call(
                milknado_deposit_review,
                run_id="node-1-20260101T000000Z-abcd",
                verdict="approve",
                findings_md="n/a",
                project_root=str(tmp_path),
            )

    def test_deposit_persists_approve_verdict_and_findings(self, tmp_path: Path) -> None:
        """A well-formed verdict must persist as its own role='review' message —
        distinct from role='result' — so both survive the process boundary
        alongside each other, matching the spec's 'alongside' requirement."""
        run_id = "node-1-20260101T000000Z-abcd"
        _seed_run(tmp_path, run_id=run_id, node_id=1, status="running")
        result = _call(
            milknado_deposit_review,
            run_id=run_id,
            verdict="approve",
            findings_md="# Findings\nNo issues found.",
            project_root=str(tmp_path),
        )
        assert result == {"run_id": run_id, "seq": 1}
        graph, _cfg = open_graph(tmp_path)
        try:
            stored = graph.latest_run_message(run_id, "review")
            assert stored == "approve\n# Findings\nNo issues found."
            assert graph.latest_run_message(run_id, "result") is None
        finally:
            graph.close()

    def test_deposit_persists_reject_verdict(self, tmp_path: Path) -> None:
        run_id = "node-1-20260101T000000Z-abcd"
        _seed_run(tmp_path, run_id=run_id, node_id=1, status="running")
        result = _call(
            milknado_deposit_review,
            run_id=run_id,
            verdict="reject",
            findings_md="missing test coverage",
            project_root=str(tmp_path),
        )
        assert result["seq"] == 1
        graph, _cfg = open_graph(tmp_path)
        try:
            assert graph.latest_run_message(run_id, "review") == "reject\nmissing test coverage"
        finally:
            graph.close()
