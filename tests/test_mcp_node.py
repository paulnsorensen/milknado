"""Native Workflow backend MCP tools: milknado_todo_claim, milknado_node_verify,
and the server-side completion gate on the terminal "done" transition.

Uses a real git repo so the claim tool's GitAdapter.create_worktree produces a
worktree on disk and node_verify runs real quality gates in it.
"""

from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from milknado.domains.common import NodeKind, NodeSpec, NodeStatus
from milknado.domains.execution.completion import NO_GATES_CONFIGURED_MESSAGE
from milknado.mcp._core import open_graph
from milknado.mcp.node import (
    GOAL_OWNER_ENV_VAR,
    milknado_goal_claim,
    milknado_goal_release,
    milknado_node_verify,
    milknado_todo_claim,
)
from milknado.mcp.todo_mutate import milknado_set_subtree_status, milknado_todo_set_status

_DEAD_PID = 2**31 - 1  # no process can hold this; os.kill(_, 0) -> ProcessLookupError


def _call(tool, **kwargs):
    fn = getattr(tool, "fn", tool)
    return fn(**kwargs)


def _init_git_repo(repo: Path) -> None:
    for cmd in (
        ["git", "init"],
        ["git", "config", "user.email", "t@milknado.test"],
        ["git", "config", "user.name", "Milknado Test"],
    ):
        subprocess.run(cmd, cwd=repo, check=True, capture_output=True)
    (repo / "README.md").write_text("# t\n", encoding="utf-8")
    for cmd in (["git", "add", "-A"], ["git", "commit", "-m", "init"]):
        subprocess.run(cmd, cwd=repo, check=True, capture_output=True)


def _write_config(repo: Path, *, gates: list[str]) -> None:
    gates_toml = ", ".join(f'"{g}"' for g in gates)
    (repo / "milknado.toml").write_text(
        f'[milknado]\nagent_family = "claude"\nquality_gates = [{gates_toml}]\n',
        encoding="utf-8",
    )


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    _init_git_repo(tmp_path)
    return tmp_path


def _add_task(repo: Path, description: str = "do the thing") -> int:
    graph, _cfg = open_graph(repo)
    try:
        return graph.add_node(description, spec=NodeSpec(kind=NodeKind.TASK)).id
    finally:
        graph.close()


# ── milknado_todo_claim ──────────────────────────────────────────────────────


def test_claim_marks_running_creates_worktree_writes_run_no_spawn(repo: Path) -> None:
    _write_config(repo, gates=["true"])
    node_id = _add_task(repo)

    payload = _call(milknado_todo_claim, node_id=node_id, project_root=str(repo))

    # Node is RUNNING with the run_id fence.
    graph, _cfg = open_graph(repo)
    try:
        node = graph.get_node(node_id)
        assert node is not None
        assert node.status == NodeStatus.RUNNING
        assert node.run_id == payload["run_id"]
        # runs row present and running.
        run = graph.get_run(payload["run_id"])
        assert run is not None
        assert run["status"] == "running"
        assert node.worktree_path == payload["worktree_path"]
    finally:
        graph.close()

    # Worktree exists on disk and is a real linked worktree.
    wt = Path(payload["worktree_path"])
    assert wt.is_dir()
    assert (wt / "README.md").exists()


def test_claim_returns_full_structured_payload(repo: Path) -> None:
    _write_config(repo, gates=["true"])
    node_id = _add_task(repo)
    payload = _call(milknado_todo_claim, node_id=node_id, project_root=str(repo))
    for key in (
        "run_id",
        "node_id",
        "brief",
        "flavor",
        "model",
        "tools",
        "worktree_path",
        "agent_type",
        "loop_mode",
        "max_iterations",
        "max_turns",
    ):
        assert key in payload, f"missing payload field {key}"
    assert payload["agent_type"] == "milknado:milknado-worker"
    assert payload["loop_mode"] == "redispatch"
    assert payload["max_iterations"] == 8
    assert payload["max_turns"] == 60
    assert isinstance(payload["tools"], list) and payload["tools"]
    # coordinator tools must never be handed to a worker.
    assert "mcp__milknado__milknado_todo_claim" not in payload["tools"]
    assert payload["brief"].startswith("# Task:")


def test_claim_honors_global_worker_tools(repo: Path) -> None:
    """The native tools list consults GLOBAL [milknado.worker.tools], like the subprocess path."""
    (repo / "milknado.toml").write_text(
        '[milknado]\nagent_family = "claude"\nquality_gates = ["true"]\n'
        '\n[milknado.worker.tools]\nclaude = ["Read", "Write"]\n',
        encoding="utf-8",
    )
    node_id = _add_task(repo)
    payload = _call(milknado_todo_claim, node_id=node_id, project_root=str(repo))
    # No "..." sentinel -> the override replaces the family default outright.
    assert payload["tools"] == ["Read", "Write"]


def test_claim_preserves_empty_flavor_tools_override(repo: Path) -> None:
    """A per-flavor `tools = []` is an intentional empty allowlist, not inherit."""
    (repo / "milknado.toml").write_text(
        '[milknado]\nagent_family = "claude"\nquality_gates = ["true"]\n'
        "\n[milknado.flavor.implement]\ntools = []\n",
        encoding="utf-8",
    )
    node_id = _add_task(repo)  # tasks default to the implement flavor
    payload = _call(milknado_todo_claim, node_id=node_id, project_root=str(repo))
    assert payload["tools"] == []


def test_claim_single_mode_includes_node_verify(repo: Path) -> None:
    """In loop_mode="single" the worker self-verifies, so node_verify must be allowlisted."""
    (repo / "milknado.toml").write_text(
        '[milknado]\nagent_family = "claude"\nquality_gates = ["true"]\n'
        '\n[milknado.flavor.implement]\nloop_mode = "single"\n',
        encoding="utf-8",
    )
    node_id = _add_task(repo)
    payload = _call(milknado_todo_claim, node_id=node_id, project_root=str(repo))
    assert payload["loop_mode"] == "single"
    assert "mcp__milknado__milknado_node_verify" in payload["tools"]


def test_claim_honors_custom_worktree_pattern(repo: Path) -> None:
    """_create_node_worktree threads cfg.worktree_pattern instead of a hardcoded name."""
    (repo / "milknado.toml").write_text(
        '[milknado]\nagent_family = "claude"\nquality_gates = ["true"]\n'
        'worktree_pattern = "wt-{node_id}-{slug}"\n',
        encoding="utf-8",
    )
    node_id = _add_task(repo)
    payload = _call(milknado_todo_claim, node_id=node_id, project_root=str(repo))
    assert Path(payload["worktree_path"]).name.startswith(f"wt-{node_id}-")


def test_claim_is_atomic_second_claim_loses(repo: Path) -> None:
    _write_config(repo, gates=["true"])
    node_id = _add_task(repo)
    _call(milknado_todo_claim, node_id=node_id, project_root=str(repo))
    with pytest.raises(ValueError, match="already running"):
        _call(milknado_todo_claim, node_id=node_id, project_root=str(repo))


def test_claim_refuses_non_task_node(repo: Path) -> None:
    _write_config(repo, gates=["true"])
    graph, _cfg = open_graph(repo)
    try:
        goal_id = graph.add_node("a goal", spec=NodeSpec(kind=NodeKind.GOAL)).id
    finally:
        graph.close()
    with pytest.raises(ValueError, match="only task nodes"):
        _call(milknado_todo_claim, node_id=goal_id, project_root=str(repo))


# ── milknado_todo_claim: per-node worktree policy (ADR-005) ─────────────────


def test_claim_worktree_false_override_skips_worktree(repo: Path) -> None:
    """worktree=False claims RUNNING in place: no worktree, node.worktree_path stays None."""
    _write_config(repo, gates=["true"])
    node_id = _add_task(repo)

    payload = _call(milknado_todo_claim, node_id=node_id, worktree=False, project_root=str(repo))

    assert payload["worktree_path"] is None
    graph, _cfg = open_graph(repo)
    try:
        node = graph.get_node(node_id)
        assert node is not None
        assert node.status == NodeStatus.RUNNING
        assert node.worktree_path is None
        run = graph.get_run(payload["run_id"])
        assert run is not None and run["status"] == "running"
    finally:
        graph.close()


def test_in_place_reclaim_clears_stale_isolated_metadata(repo: Path) -> None:
    from milknado.domains.dispatch import now_iso

    _write_config(repo, gates=["true"])
    node_id = _add_task(repo)
    stale = repo / "stale-worktree"
    stale.mkdir()
    graph, _cfg = open_graph(repo)
    try:
        assert graph.claim_node(node_id, "old-run", now=now_iso()) is True
        graph.set_worktree(node_id, "old-run", str(stale), "old-branch")
        graph.mark_terminal(node_id, "old-run", NodeStatus.FAILED)
    finally:
        graph.close()

    payload = _call(milknado_todo_claim, node_id=node_id, worktree=False, project_root=str(repo))
    (repo / "new-evidence.txt").write_text("current checkout\n", encoding="utf-8")
    verdict = _call(milknado_node_verify, run_id=payload["run_id"], project_root=str(repo))
    assert payload["worktree_path"] is None
    assert verdict == {"ok": True, "feedback": ""}
    graph, _cfg = open_graph(repo)
    try:
        node = graph.get_node(node_id)
        assert node is not None
        assert (node.worktree_path, node.branch_name) == (None, None)
    finally:
        graph.close()


def test_claim_worktree_none_defaults_to_profile(repo: Path) -> None:
    """worktree=None (default) follows the resolved flavor profile's worktree knob."""
    (repo / "milknado.toml").write_text(
        '[milknado]\nagent_family = "claude"\nquality_gates = ["true"]\n'
        "\n[milknado.flavor.implement]\nworktree = false\n",
        encoding="utf-8",
    )
    node_id = _add_task(repo)  # tasks default to the implement flavor
    payload = _call(milknado_todo_claim, node_id=node_id, project_root=str(repo))
    assert payload["worktree_path"] is None


def test_claim_worktree_true_override_wins_over_false_profile_default(repo: Path) -> None:
    """An explicit worktree=True override wins even when the flavor default is false."""
    (repo / "milknado.toml").write_text(
        '[milknado]\nagent_family = "claude"\nquality_gates = ["true"]\n'
        "\n[milknado.flavor.implement]\nworktree = false\n",
        encoding="utf-8",
    )
    node_id = _add_task(repo)
    payload = _call(milknado_todo_claim, node_id=node_id, worktree=True, project_root=str(repo))
    assert payload["worktree_path"] is not None
    assert Path(payload["worktree_path"]).is_dir()


# ── milknado_node_verify ─────────────────────────────────────────────────────


def test_node_verify_passing_gates_returns_ok(repo: Path) -> None:
    _write_config(repo, gates=["true"])
    node_id = _add_task(repo)
    claim = _call(milknado_todo_claim, node_id=node_id, project_root=str(repo))
    # Produce a stageable change so the change-rejection check passes.
    (Path(claim["worktree_path"]) / "out.txt").write_text("work\n", encoding="utf-8")

    verdict = _call(milknado_node_verify, run_id=claim["run_id"], project_root=str(repo))
    assert verdict == {"ok": True, "feedback": ""}


def test_node_verify_failing_gate_returns_not_ok_with_feedback(repo: Path) -> None:
    _write_config(repo, gates=["false"])
    node_id = _add_task(repo)
    claim = _call(milknado_todo_claim, node_id=node_id, project_root=str(repo))

    verdict = _call(milknado_node_verify, run_id=claim["run_id"], project_root=str(repo))
    assert verdict["ok"] is False
    assert verdict["feedback"]


def test_node_verify_unknown_run_raises(repo: Path) -> None:
    _write_config(repo, gates=["true"])
    with pytest.raises(ValueError, match="not found"):
        _call(
            milknado_node_verify,
            run_id="node-1-20260101T000000Z-deadbeef",
            project_root=str(repo),
        )


def test_node_verify_rejects_malformed_run_id(repo: Path) -> None:
    """run_id is an external input: a value that does not match RUN_ID_RE is
    rejected at the boundary before any graph lookup, not treated as not-found."""
    _write_config(repo, gates=["true"])
    with pytest.raises(ValueError, match="invalid run_id format"):
        _call(milknado_node_verify, run_id="../etc/passwd", project_root=str(repo))


def test_node_verify_unconfigured_gates_fails_closed(repo: Path) -> None:
    """When the flavor resolves quality_gates=None (unconfigured, not the
    explicit skip-gates `[]`), node_verify must fail closed with the
    actionable message rather than raising a TypeError from list(None)."""
    (repo / "milknado.toml").write_text('[milknado]\nagent_family = "claude"\n', encoding="utf-8")
    node_id = _add_task(repo)
    claim = _call(milknado_todo_claim, node_id=node_id, project_root=str(repo))

    verdict = _call(milknado_node_verify, run_id=claim["run_id"], project_root=str(repo))
    assert verdict["ok"] is False
    assert verdict["feedback"] == NO_GATES_CONFIGURED_MESSAGE


# ── completion gate on terminal "done" ───────────────────────────────────────


def test_done_rejected_without_passing_verify(repo: Path) -> None:
    _write_config(repo, gates=["true"])
    node_id = _add_task(repo)
    _call(milknado_todo_claim, node_id=node_id, project_root=str(repo))
    with pytest.raises(ValueError, match="has not returned ok=True"):
        _call(milknado_todo_set_status, node_id=node_id, status="done", project_root=str(repo))


def test_done_rejected_when_verify_failed(repo: Path) -> None:
    _write_config(repo, gates=["false"])
    node_id = _add_task(repo)
    claim = _call(milknado_todo_claim, node_id=node_id, project_root=str(repo))
    failed = _call(milknado_node_verify, run_id=claim["run_id"], project_root=str(repo))
    assert failed["ok"] is False
    with pytest.raises(ValueError, match="has not returned ok=True"):
        _call(milknado_todo_set_status, node_id=node_id, status="done", project_root=str(repo))


def test_done_accepted_after_passing_verify(repo: Path) -> None:
    _write_config(repo, gates=["true"])
    node_id = _add_task(repo)
    claim = _call(milknado_todo_claim, node_id=node_id, project_root=str(repo))
    (Path(claim["worktree_path"]) / "out.txt").write_text("work\n", encoding="utf-8")
    ok = _call(milknado_node_verify, run_id=claim["run_id"], project_root=str(repo))
    assert ok["ok"] is True

    result = _call(
        milknado_todo_set_status, node_id=node_id, status="done", project_root=str(repo)
    )
    assert result["status"] == "done"


def test_done_accepted_after_failed_then_passing_reverify(repo: Path) -> None:
    """The gate reads the LATEST verdict, not any verdict: a node that fails
    verify, gets more work, then re-verifies clean must be allowed to mark done.

    Guards the seq-DESC ordering in latest_run_message — if the gate read the
    first (failed) verdict instead, this redispatch-loop success path would
    deadlock a node that legitimately passed on a later iteration.
    """
    # Gate path resolves the node's quality_gates from config; rewrite the same
    # config file between verifies to flip the gate result deterministically.
    _write_config(repo, gates=["false"])
    node_id = _add_task(repo)
    claim = _call(milknado_todo_claim, node_id=node_id, project_root=str(repo))

    failed = _call(milknado_node_verify, run_id=claim["run_id"], project_root=str(repo))
    assert failed["ok"] is False

    # Worker does more work and the gate now passes.
    _write_config(repo, gates=["true"])
    (Path(claim["worktree_path"]) / "out.txt").write_text("more work\n", encoding="utf-8")
    passed = _call(milknado_node_verify, run_id=claim["run_id"], project_root=str(repo))
    assert passed["ok"] is True

    result = _call(
        milknado_todo_set_status, node_id=node_id, status="done", project_root=str(repo)
    )
    assert result["status"] == "done"


def test_subtree_done_gate_blocks_unverified_native_node(repo: Path) -> None:
    """A native-claimed task in the subtree blocks the bulk done until verified."""
    graph, _cfg = open_graph(repo)
    try:
        goal_id = graph.add_node("goal", spec=NodeSpec(kind=NodeKind.GOAL)).id
        task_id = graph.add_node("task", parent_id=goal_id, spec=NodeSpec(kind=NodeKind.TASK)).id
    finally:
        graph.close()
    _write_config(repo, gates=["true"])
    _call(milknado_todo_claim, node_id=task_id, project_root=str(repo))
    with pytest.raises(ValueError, match="has not returned ok=True"):
        _call(milknado_set_subtree_status, root_id=goal_id, status="done", project_root=str(repo))


def test_done_gate_does_not_block_manual_unclaimed_node(repo: Path) -> None:
    """A node with no owning run (manual/subprocess) is exempt from the verify gate."""
    _write_config(repo, gates=["true"])
    node_id = _add_task(repo, "manual node")
    result = _call(
        milknado_todo_set_status, node_id=node_id, status="done", project_root=str(repo)
    )
    assert result["status"] == "done"


def test_done_gate_exempts_subprocess_run_node(repo: Path) -> None:
    """A subprocess-completed node carries a run_id and a runs row but no native
    marker: the subprocess executor uses start_run, never milknado_todo_claim,
    so it deposits no CLAIM_ROLE/verify message. The gate must discriminate on
    that native-only marker, not on "run_id + runs row" — otherwise it would
    wrongly block a manual `done` on a node that never used the native backend.

    Pins the fix: shape a node exactly like a subprocess run (start_run + a
    result message, no claim/verify message) and assert the gate lets it through.
    """
    from milknado.domains.dispatch import now_iso

    _write_config(repo, gates=["true"])
    node_id = _add_task(repo, "subprocess node")
    run_id = "node-1-20260101T000000Z-deadbeef"

    graph, _cfg = open_graph(repo)
    try:
        # Reproduce the subprocess shape: CAS-claim fences run_id onto the node
        # (as the dispatcher does), a running run row, and a deposited result —
        # but NO claim/verify message, because the subprocess path never calls
        # milknado_todo_claim or milknado_node_verify.
        assert graph.claim_node(node_id, run_id, now=now_iso()) is True
        graph.start_run(run_id, node_id, "log", now_iso(), None)
        graph.deposit_run_message(run_id, "result", "subprocess deliverable", now_iso())
        node = graph.get_node(node_id)
        assert node is not None
        assert node.run_id == run_id
        assert graph.latest_run_message(run_id, "claim") is None
        assert graph.latest_run_message(run_id, "verify") is None
    finally:
        graph.close()

    result = _call(
        milknado_todo_set_status, node_id=node_id, status="done", project_root=str(repo)
    )
    assert result["status"] == "done"


# ── edge cases ───────────────────────────────────────────────────────────────


def test_claim_unknown_node_raises(repo: Path) -> None:
    _write_config(repo, gates=["true"])
    with pytest.raises(ValueError, match="not found"):
        _call(milknado_todo_claim, node_id=999, project_root=str(repo))


def test_claim_rolls_back_claim_when_worktree_creation_fails(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If worktree/run setup fails after the CAS claim, the node is released FAILED."""
    _write_config(repo, gates=["true"])
    node_id = _add_task(repo)

    def _boom(*_a, **_k):
        raise RuntimeError("worktree boom")

    monkeypatch.setattr("milknado.mcp.node.create_isolated_worktree", _boom)
    with pytest.raises(RuntimeError, match="worktree boom"):
        _call(milknado_todo_claim, node_id=node_id, project_root=str(repo))

    graph, _cfg = open_graph(repo)
    try:
        node = graph.get_node(node_id)
        assert node is not None
        # Released by the fenced terminal write — not stranded RUNNING.
        assert node.status == NodeStatus.FAILED
    finally:
        graph.close()


def test_node_verify_in_place_no_worktree_uses_project_root(repo: Path) -> None:
    """ADR-005/006: a node with no worktree_path is verified in project_root."""
    from datetime import UTC, datetime

    _write_config(repo, gates=["true"])
    node_id = _add_task(repo)
    run_id = "node-1-20260101T000000Z-deadbeef"
    graph, _cfg = open_graph(repo)
    try:
        graph.start_run(run_id, node_id, "log", datetime.now(UTC).isoformat(), None)
    finally:
        graph.close()
    # An in-place node with non-empty gates keeps the stageable-change check;
    # the repo has an untracked file relative to its own HEAD -> passes.
    (repo / "out.txt").write_text("work\n", encoding="utf-8")

    verdict = _call(milknado_node_verify, run_id=run_id, project_root=str(repo))
    assert verdict["ok"] is True


def test_node_verify_in_place_artifact_evidence_pass(repo: Path) -> None:
    """ADR-006: in-place node, explicitly-empty gates, artifact exists+non-empty -> ok."""
    _write_config(repo, gates=[])
    node_id = _add_task(repo)
    claim = _call(milknado_todo_claim, node_id=node_id, worktree=False, project_root=str(repo))
    assert claim["worktree_path"] is None

    graph, _cfg = open_graph(repo)
    try:
        graph._conn.execute(
            "UPDATE nodes SET artifact_path = ? WHERE id = ?", ("docs/out.md", node_id)
        )
        graph._conn.commit()
    finally:
        graph.close()
    (repo / "docs").mkdir()
    (repo / "docs" / "out.md").write_text("deliverable\n", encoding="utf-8")

    verdict = _call(milknado_node_verify, run_id=claim["run_id"], project_root=str(repo))
    assert verdict == {"ok": True, "feedback": ""}


def test_node_verify_in_place_artifact_missing_fails(repo: Path) -> None:
    """ADR-006: in-place node, explicitly-empty gates, artifact missing -> not ok."""
    _write_config(repo, gates=[])
    node_id = _add_task(repo)
    claim = _call(milknado_todo_claim, node_id=node_id, worktree=False, project_root=str(repo))

    graph, _cfg = open_graph(repo)
    try:
        graph._conn.execute(
            "UPDATE nodes SET artifact_path = ? WHERE id = ?", ("docs/out.md", node_id)
        )
        graph._conn.commit()
    finally:
        graph.close()

    verdict = _call(milknado_node_verify, run_id=claim["run_id"], project_root=str(repo))
    assert verdict["ok"] is False
    assert verdict["feedback"]


def test_resolve_model_extracts_flag_and_defaults() -> None:
    from milknado.mcp.node import _resolve_model

    assert _resolve_model("claude --model opus -p") == "opus"
    assert _resolve_model("claude --model=sonnet -p") == "sonnet"
    assert _resolve_model("codex exec --sandbox workspace-write") == "sonnet"


def test_latest_verify_ok_false_on_missing_or_malformed(repo: Path) -> None:
    from milknado.domains.dispatch import now_iso
    from milknado.domains.graph.status_flow import VERIFY_ROLE, latest_verify_ok

    _write_config(repo, gates=["true"])
    node_id = _add_task(repo)
    claim = _call(milknado_todo_claim, node_id=node_id, project_root=str(repo))
    graph, _cfg = open_graph(repo)
    try:
        # No verify message yet -> not ok.
        assert latest_verify_ok(graph, claim["run_id"]) is False
        # Malformed verdict body -> fail-closed.
        graph.deposit_run_message(claim["run_id"], VERIFY_ROLE, "not json", now_iso())
        assert latest_verify_ok(graph, claim["run_id"]) is False
    finally:
        graph.close()


# ── milknado_goal_claim / milknado_goal_release (ADR-003) ───────────────────


def _add_goal(repo: Path, description: str = "a goal") -> int:
    graph, _cfg = open_graph(repo)
    try:
        return graph.add_node(description, spec=NodeSpec(kind=NodeKind.GOAL)).id
    finally:
        graph.close()


def test_goal_claim_then_release_frees_it(repo: Path) -> None:
    goal_id = _add_goal(repo)

    payload = _call(milknado_goal_claim, goal_id=goal_id, owner="sess-1", project_root=str(repo))
    assert payload == {"goal_id": goal_id, "owner": "sess-1"}
    graph, _cfg = open_graph(repo)
    try:
        assert graph.get_node(goal_id).goal_run_id == "sess-1"
    finally:
        graph.close()

    released = _call(
        milknado_goal_release, goal_id=goal_id, owner="sess-1", project_root=str(repo)
    )
    assert released == {"goal_id": goal_id, "released": True}
    graph, _cfg = open_graph(repo)
    try:
        assert graph.get_node(goal_id).goal_run_id is None
    finally:
        graph.close()


def test_goal_claim_blocks_second_live_owner(repo: Path) -> None:
    goal_id = _add_goal(repo)
    _call(milknado_goal_claim, goal_id=goal_id, owner="sess-1", project_root=str(repo))

    with pytest.raises(ValueError, match="already claimed by owner 'sess-1'"):
        _call(milknado_goal_claim, goal_id=goal_id, owner="sess-2", project_root=str(repo))


def test_goal_claim_concurrent_public_calls_report_one_winner(repo: Path) -> None:
    goal_id = _add_goal(repo)
    barrier = Barrier(2)

    def claim(owner: str) -> tuple[str, str]:
        barrier.wait()
        try:
            result = _call(
                milknado_goal_claim,
                goal_id=goal_id,
                owner=owner,
                project_root=str(repo),
            )
        except ValueError as exc:
            return "lost", str(exc)
        return "won", result["owner"]

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(claim, ("owner-a", "owner-b")))
    assert sorted(result[0] for result in results) == ["lost", "won"]
    winner = next(result[1] for result in results if result[0] == "won")
    assert all(
        "already claimed by owner" in message
        for disposition, message in results
        if disposition == "lost"
    )
    graph, _cfg = open_graph(repo)
    try:
        claimed = graph.get_node(goal_id)
        assert claimed is not None
        assert claimed.goal_run_id == winner
    finally:
        graph.close()


def test_goal_release_is_fenced_on_owner(repo: Path) -> None:
    """Releasing with a non-owning token is a no-op; the claim survives."""
    goal_id = _add_goal(repo)
    _call(milknado_goal_claim, goal_id=goal_id, owner="sess-1", project_root=str(repo))

    released = _call(
        milknado_goal_release, goal_id=goal_id, owner="not-the-owner", project_root=str(repo)
    )
    assert released == {"goal_id": goal_id, "released": False}
    graph, _cfg = open_graph(repo)
    try:
        assert graph.get_node(goal_id).goal_run_id == "sess-1"
    finally:
        graph.close()


def test_goal_claim_reclaims_dead_pid_owner(repo: Path) -> None:
    """A claim whose recorded pid is provably dead is reclaimed for the new owner."""
    goal_id = _add_goal(repo)
    graph, _cfg = open_graph(repo)
    try:
        from milknado.domains.dispatch import now_iso

        assert graph.claim_goal(goal_id, "stale-owner", now=now_iso()) is True
        graph.set_goal_pid(goal_id, "stale-owner", _DEAD_PID)
    finally:
        graph.close()

    payload = _call(
        milknado_goal_claim, goal_id=goal_id, owner="fresh-owner", project_root=str(repo)
    )
    assert payload == {"goal_id": goal_id, "owner": "fresh-owner"}


def test_goal_claim_requires_owner_or_env(repo: Path) -> None:
    goal_id = _add_goal(repo)
    with pytest.raises(ValueError, match="owner is required"):
        _call(milknado_goal_claim, goal_id=goal_id, project_root=str(repo))


def test_goal_claim_falls_back_to_env_owner(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A sub-agent that inherited MILKNADO_GOAL_OWNER can claim/release without `owner`."""
    monkeypatch.setenv(GOAL_OWNER_ENV_VAR, "env-owner")
    goal_id = _add_goal(repo)

    payload = _call(milknado_goal_claim, goal_id=goal_id, project_root=str(repo))
    assert payload["owner"] == "env-owner"

    released = _call(milknado_goal_release, goal_id=goal_id, project_root=str(repo))
    assert released == {"goal_id": goal_id, "released": True}


def test_goal_claim_refuses_non_goal_node(repo: Path) -> None:
    task_id = _add_task(repo)
    with pytest.raises(ValueError, match="only goal nodes can be claimed"):
        _call(milknado_goal_claim, goal_id=task_id, owner="sess-1", project_root=str(repo))
