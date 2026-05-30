"""
End-to-end test: real 'claude --model sonnet' worker runs a single-leaf Mikado
graph through dispatch → completion → rebase.

All other test files mock the claude path; this file is the only place the real
Sonnet API is exercised in the test suite. Skipped automatically when the
``claude`` CLI or ANTHROPIC_API_KEY is absent so CI stays clean without
special orchestration.

Run just this file:
    uv run pytest tests/test_e2e_sonnet_worker.py -v

Exclude from fast runs:
    uv run pytest -m "not e2e"
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from milknado.adapters.git import GitAdapter
from milknado.adapters.ralphify import RalphifyAdapter
from milknado.domains.common.types import NodeStatus
from milknado.domains.execution import ExecutionConfig, Executor, run_node_to_completion
from milknado.domains.graph import MikadoGraph

# ---------------------------------------------------------------------------
# Skip guard
# ---------------------------------------------------------------------------

_AGENT = "claude --model claude-sonnet-4-6"
_TIMEOUT = 300.0  # 5-minute ceiling — generous but finite

_CLAUDE_MISSING = shutil.which("claude") is None
_KEY_MISSING = not os.environ.get("ANTHROPIC_API_KEY")
_SKIP_REASON = (
    "claude CLI not found"
    if _CLAUDE_MISSING
    else "ANTHROPIC_API_KEY not set"
    if _KEY_MISSING
    else ""
)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(bool(_SKIP_REASON), reason=_SKIP_REASON or "n/a"),
]

# ---------------------------------------------------------------------------
# Minimal CRG stub — satisfies CrgPort structurally; never actually called
# because the leaf node has no file ownership assigned.
# ---------------------------------------------------------------------------


class _StubCrg:
    def ensure_graph(self, project_root: Path) -> None:
        pass

    def get_impact_radius(self, files: list[str]) -> dict[str, Any]:
        return {}

    def get_architecture_overview(self) -> dict[str, Any]:
        return {}

    def list_communities(self, sort_by: str = "size", min_size: int = 0) -> list[dict[str, Any]]:
        return []

    def list_flows(self, sort_by: str = "criticality", limit: int = 50) -> list[dict[str, Any]]:
        return []

    def get_bridge_nodes(self, top_n: int = 10) -> list[dict[str, Any]]:
        return []

    def get_hub_nodes(self, top_n: int = 10) -> list[dict[str, Any]]:
        return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _init_git_repo(repo: Path) -> str:
    """Create a real git repo with one initial commit on 'feature/e2e'."""
    for cmd in [
        ["git", "init"],
        ["git", "config", "user.email", "test@milknado.test"],
        ["git", "config", "user.name", "Milknado Test"],
    ]:
        subprocess.run(cmd, cwd=repo, check=True, capture_output=True)
    (repo / "README.md").write_text("# e2e\n", encoding="utf-8")
    for cmd in [
        ["git", "add", "README.md"],
        ["git", "commit", "-m", "init"],
        ["git", "checkout", "-b", "feature/e2e"],
    ]:
        subprocess.run(cmd, cwd=repo, check=True, capture_output=True)
    return "feature/e2e"


def _build_graph(db_path: Path) -> tuple[MikadoGraph, int]:
    """Root → leaf graph. Returns (graph, leaf_id)."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    g = MikadoGraph(db_path)
    root = g.add_node("Create a greeting file in the repo")
    leaf = g.add_node(
        "Write 'hello world' to a new file called hello.txt",
        parent_id=root.id,
    )
    return g, leaf.id


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


def test_sonnet_worker_leaf_to_done(tmp_path: Path) -> None:
    """Full path: dispatch → real claude-sonnet worker → completion signal → rebase.

    The leaf task is intentionally trivial (create a single file) so the agent
    finishes in one iteration and the total wall-clock cost stays low.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    feature_branch = _init_git_repo(repo)

    graph, leaf_id = _build_graph(repo / ".milknado" / "graph.db")
    try:
        ralph = RalphifyAdapter(agent=_AGENT)
        exec_config = ExecutionConfig(
            execution_agent=_AGENT,
            quality_gates=(),
            worktree_pattern="milknado-wt-{node_id}-{slug}",
            project_root=repo,
        )
        executor = Executor(
            graph=graph,
            git=GitAdapter(repo),
            ralph=ralph,
            crg=_StubCrg(),
        )

        outcome = run_node_to_completion(
            executor=executor,
            ralph=ralph,
            node_id=leaf_id,
            exec_config=exec_config,
            feature_branch=feature_branch,
            timeout=_TIMEOUT,
        )

        assert outcome.success, f"worker run failed: {outcome.detail}"
        assert outcome.node_id == leaf_id
        leaf_node = graph.get_node(leaf_id)
        assert leaf_node is not None
        assert leaf_node.status == NodeStatus.DONE
    finally:
        graph.close()
