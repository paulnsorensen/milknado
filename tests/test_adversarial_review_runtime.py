from __future__ import annotations

import json
import queue
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from milknado.adapters.loop import (
    LoopAdapter,
    ReviewVerdict,
    _drain_review_run,
    _parse_review_output,
)
from milknado.domains.common import Gate, MikadoNode
from milknado.domains.common.agent_argv import (
    NodeAgentSession,
    build_resume_command,
    capture_session_id,
)
from milknado.domains.dispatch.brief import _resolve_spec_path
from milknado.domains.execution import (
    CompletionResult,
    DispatchResult,
    ExecutionConfig,
    Executor,
    RebaseConflict,
    run_node_to_completion,
)
from milknado.domains.execution.run_loop._completion import handle_completion
from milknado.loop._events import Event, EventType
from tests.test_execution import FakeCrg, FakeGit


@dataclass
class _Run:
    state: SimpleNamespace


class _ReviewRalph:
    def __init__(self, verdicts: list[bool]) -> None:
        self.verdicts = iter(verdicts)
        self.created: list[dict[str, Any]] = []
        self.started: list[str] = []
        self.stdout_requests: list[str] = []
        self.reviews: list[tuple[str, Path]] = []
        self.ralph_md_calls: list[dict[str, Any]] = []
        self._next_id = 0

    def create_run(self, **kwargs: Any) -> _Run:
        self._next_id += 1
        run_id = kwargs.get("run_id") or f"run-{self._next_id}"
        self.created.append({**kwargs, "run_id": run_id})
        return _Run(SimpleNamespace(run_id=run_id))

    def start_run(self, run_id: str) -> None:
        self.started.append(run_id)

    def stop_run(self, run_id: str, timeout: float | None = None) -> bool:
        return True

    def get_run_stdout(self, run_id: str) -> list[str]:
        self.stdout_requests.append(run_id)
        return ['{"session_id":"session-1"}']

    def run_node_review(
        self, agent: str, prompt: str, worktree: Path, project_root: Path
    ) -> ReviewVerdict:
        self.reviews.append((agent, worktree))
        return ReviewVerdict(approved=next(self.verdicts), findings_md="[P1][correctness] finding")

    def generate_ralph_md(
        self,
        brief: str,
        quality_gates: tuple[Gate, ...] | None,
        output_path: Path,
        prior_findings: str = "",
        findings_round: int | None = None,
    ) -> Path:
        self.ralph_md_calls.append(
            {"prior_findings": prior_findings, "findings_round": findings_round}
        )
        return output_path


def _config(root: Path, **overrides: Any) -> ExecutionConfig:
    values: dict[str, Any] = {
        "execution_agent": "claude --model opus --effort high",
        "quality_gates": (Gate("true"),),
        "worktree_pattern": "milknado-{node_id}-{slug}",
        "project_root": root,
        "review": True,
        "review_agent": "age",
        "review_max_rounds": 1,
        "session_mode": "resume",
        "on_reject": "block",
    }
    values.update(overrides)
    return ExecutionConfig(**values)


def _executor(graph, root: Path, ralph: _ReviewRalph, git: FakeGit | None = None) -> Executor:
    del root
    return Executor(
        graph=graph,
        git=cast(Any, git or FakeGit()),
        ralph=cast(Any, ralph),
        crg=cast(Any, FakeCrg()),
    )


def test_reject_redispatches_pinned_worktree_and_resumes_session(graph, tmp_path: Path) -> None:
    ralph = _ReviewRalph([False, True])
    executor = _executor(graph, tmp_path, ralph)
    graph.add_node("reviewed change")

    first = executor.dispatch(1, _config(tmp_path))
    rejected = executor.complete(1, "main")
    assert rejected.redispatch is not None
    assert rejected.redispatch.worktree == first.worktree
    assert ralph.created[1]["runtime_policy"].session.session_id == "session-1"
    assert ralph.created[1]["runtime_policy"].session.worktree_path == str(
        first.worktree.resolve()
    )

    approved = executor.complete(1, "main")
    assert approved.rebased is True
    assert graph.get_node(1).status.value == "done"
    session_file = tmp_path / ".milknado" / "sessions" / "node-1.json"
    assert json.loads(session_file.read_text())["session_id"] == "session-1"
    assert (
        first.worktree / ".cheese" / "age" / "reviewed-change.md"
    ).read_text() == "[P1][correctness] finding\n"


def test_redispatch_threads_findings_into_ralph_regeneration(graph, tmp_path: Path) -> None:
    """#298: the redispatched worker's RALPH.md regenerates with the rejecting
    round's findings threaded in memory — required reading, not re-derived."""
    ralph = _ReviewRalph([False, True])
    executor = _executor(graph, tmp_path, ralph)
    graph.add_node("findings thread")

    executor.dispatch(1, _config(tmp_path))
    rejected = executor.complete(1, "main")

    assert rejected.redispatch is not None
    assert ralph.ralph_md_calls[0] == {"prior_findings": "", "findings_round": None}
    assert ralph.ralph_md_calls[1]["prior_findings"] == "[P1][correctness] finding"
    assert ralph.ralph_md_calls[1]["findings_round"] == 1


def test_notify_review_dual_writes_node_reviews_and_run_messages(graph, tmp_path: Path) -> None:
    """#298: every notified verdict lands in node_reviews (audit trail, no runs
    dependency) AND in run_messages (FK satisfied by the curd-A runs row)."""
    ralph = _ReviewRalph([False, True])
    executor = _executor(graph, tmp_path, ralph)
    graph.add_node("verdict audit")

    executor.dispatch(1, _config(tmp_path))
    executor.complete(1, "main")

    rows = graph.node_reviews_for_node(1)
    assert len(rows) == 1
    assert rows[0]["verdict"] == "reject"
    assert rows[0]["round"] == 1
    assert rows[0]["findings"] == "[P1][correctness] finding"
    worker_run_id = ralph.started[0]
    assert graph.latest_run_message(worker_run_id, "node_review") is not None


def test_findings_delivered_in_memory_when_db_writes_fail(
    graph, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#298: the DB being down at redispatch must not lose the findings — the
    handoff is the in-memory thread, never a DB read-back."""
    ralph = _ReviewRalph([False, True])
    executor = _executor(graph, tmp_path, ralph)
    graph.add_node("db down redispatch")

    executor.dispatch(1, _config(tmp_path))
    monkeypatch.setattr(
        graph, "deposit_run_message", MagicMock(side_effect=RuntimeError("db down"))
    )
    monkeypatch.setattr(
        graph, "insert_node_review", MagicMock(side_effect=RuntimeError("db down"))
    )
    rejected = executor.complete(1, "main")

    assert rejected.review_notification_failed is True
    assert rejected.redispatch is not None
    assert ralph.ralph_md_calls[1]["prior_findings"] == "[P1][correctness] finding"
    assert ralph.ralph_md_calls[1]["findings_round"] == 1


def test_second_rejection_round_accumulates_audits_and_labels_round_2(
    graph, tmp_path: Path
) -> None:
    """#298: consecutive rejections are distinct rounds — node_reviews keeps one
    row per round (PK is (node_id, round)) and the round-2 RALPH regeneration
    is labeled round 2, so the worker can tell its second attempt was rejected
    again rather than mistaking it for a fresh dispatch."""
    ralph = _ReviewRalph([False, False, True])
    executor = _executor(graph, tmp_path, ralph)
    graph.add_node("two rejections")

    executor.dispatch(1, _config(tmp_path, review_max_rounds=2))
    first = executor.complete(1, "main")
    second = executor.complete(1, "main")
    assert first.redispatch is not None and second.redispatch is not None

    rows = graph.node_reviews_for_node(1)
    assert [r["round"] for r in rows] == [1, 2]
    assert all(r["verdict"] == "reject" for r in rows)
    assert ralph.ralph_md_calls[1]["findings_round"] == 1
    assert ralph.ralph_md_calls[2]["findings_round"] == 2
    assert ralph.ralph_md_calls[2]["prior_findings"] == "[P1][correctness] finding"


def test_zero_review_rounds_preserves_merge_path(graph, tmp_path: Path) -> None:
    ralph = _ReviewRalph([False])
    git = FakeGit()
    executor = _executor(graph, tmp_path, ralph, git)
    graph.add_node("no review")

    executor.dispatch(1, _config(tmp_path, review_max_rounds=0))
    result = executor.complete(1, "main")
    assert result.rebased is True
    assert not ralph.reviews


def test_warn_policy_merges_after_review_round_budget(graph, tmp_path: Path) -> None:
    ralph = _ReviewRalph([False, False])
    git = FakeGit()
    executor = _executor(graph, tmp_path, ralph, git)
    graph.add_node("warn review")

    executor.dispatch(1, _config(tmp_path, on_reject="warn"))
    redispatch = executor.complete(1, "main").redispatch
    assert redispatch is not None
    result = executor.complete(1, "main")
    assert result.rebased is True
    assert graph.get_node(1).status.value == "done"
    assert git.rebases


def test_block_policy_retains_pinned_worktree(
    graph, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ralph = _ReviewRalph([False, False])
    git = FakeGit()
    executor = _executor(graph, tmp_path, ralph, git)
    graph.add_node("blocked review")
    monkeypatch.setattr(graph, "deposit_run_message", MagicMock(return_value=1))

    first = executor.dispatch(1, _config(tmp_path))
    assert executor.complete(1, "main").redispatch is not None
    result = executor.complete(1, "main")
    node = graph.get_node(1)
    assert result.blocked is True
    assert node.status.value == "blocked"
    assert node.worktree_path == str(first.worktree)
    assert not git.rebases
    assert not git.removed


def test_durable_spec_path_prefers_xdg_corpus(tmp_path: Path, monkeypatch) -> None:
    xdg = tmp_path / "xdg"
    durable = xdg / "cheese" / tmp_path.name / "specs" / "accepted.md"
    durable.parent.mkdir(parents=True)
    durable.write_text("# Accepted\n", encoding="utf-8")
    monkeypatch.setenv("XDG_DATA_HOME", str(xdg))
    node = MikadoNode(id=1, description="spec", artifact_path="accepted.md")

    assert _resolve_spec_path(node, [], tmp_path) == str(durable.resolve())


def test_loop_adapter_resume_command_keeps_cwd_and_controls(tmp_path: Path) -> None:
    prompt_file = tmp_path / "RALPH.md"
    prompt_file.write_text("prompt", encoding="utf-8")
    session = NodeAgentSession(1, "claude", "s1", str(tmp_path), "now")
    adapter = LoopAdapter()
    run = adapter.create_run(
        "claude --model opus --effort high",
        tmp_path,
        prompt_file,
        [],
        (Gate("true"),),
        project_root=tmp_path,
        runtime_policy=SimpleNamespace(session=session),
    )
    assert "--resume s1" in run.config.agent
    assert "--model opus" in run.config.agent
    assert "--effort high" in run.config.agent
    assert run.config.project_root == tmp_path


def test_capture_session_handles_nested_json_and_failures(graph, tmp_path: Path) -> None:
    ralph = _ReviewRalph([True])
    ralph.get_run_stdout = lambda run_id: [  # type: ignore[method-assign]
        "not-json",
        '{"event":{"session_id":"nested"}}',
    ]
    executor = _executor(graph, tmp_path, ralph)
    node = MikadoNode(id=9, description="session", run_id="worker", worktree_path=str(tmp_path))
    session = executor._capture_session(node, _config(tmp_path, agent_family="codex"))
    assert session is not None
    assert session.session_id == "nested"
    assert executor._capture_session(node, _config(tmp_path, agent_family="codex")) == session

    with pytest.raises(ValueError, match="no worker run id"):
        executor._capture_session(
            MikadoNode(id=10, description="missing"),
            _config(tmp_path),
        )
    ralph.get_run_stdout = lambda run_id: ["{}"]  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="resumable session id"):
        executor._capture_session(
            MikadoNode(id=11, description="invalid", run_id="worker"),
            _config(tmp_path),
        )


def test_executor_review_helpers_cover_spec_and_missing_reviewer(graph, tmp_path: Path) -> None:
    ralph = _ReviewRalph([True])
    executor = _executor(graph, tmp_path, ralph)
    node = MikadoNode(id=12, description="helper", artifact_path="spec.md")
    executor._base_oid_by_node[node.id] = "base-oid"
    (tmp_path / "RALPH.md").write_text("generated context", encoding="utf-8")
    prompt = executor._review_prompt(node, tmp_path, _config(tmp_path, session_mode="fresh"))
    assert str((tmp_path / "spec.md").resolve()) in prompt
    assert "generated context" in prompt
    assert "diff from base-oid" in prompt
    executor._notify_review(node, verdict="reject", findings_md="finding", round_number=1)

    missing_reviewer = SimpleNamespace()
    no_reviewer = Executor(
        graph=graph,
        git=cast(Any, FakeGit()),
        ralph=cast(Any, missing_reviewer),
        crg=cast(Any, FakeCrg()),
    )
    with pytest.raises(ValueError, match="requires a LoopPort reviewer"):
        no_reviewer._run_review(node, tmp_path, _config(tmp_path, session_mode="fresh"))
    with pytest.raises(ValueError, match="no run fence"):
        executor._redispatch_review_round(
            node,
            _config(tmp_path, session_mode="fresh"),
            tmp_path,
        )


def test_review_true_without_reviewer_fails_before_dispatch(graph, tmp_path: Path) -> None:
    executor = _executor(graph, tmp_path, _ReviewRalph([True]))
    graph.add_node("missing reviewer")

    with pytest.raises(ValueError, match="review=true"):
        executor.dispatch(1, _config(tmp_path, review_agent="  "))


def test_adopted_review_keeps_parent_fence_and_notifies_worker(
    graph, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from milknado.domains.dispatch._runstate import now_iso

    ralph = _ReviewRalph([False, True])
    executor = _executor(graph, tmp_path, ralph)
    graph.add_node("adopted review")
    parent_fence = "parent-fence"
    assert graph.claim_node(1, parent_fence, now=now_iso())
    notified = MagicMock(return_value=1)
    monkeypatch.setattr(graph, "deposit_run_message", notified)

    executor.dispatch(1, _config(tmp_path), parent_run_id=parent_fence)
    rejected = executor.complete(1, "main")
    round_run_id = ralph.created[0]["run_id"]

    assert rejected.redispatch is not None
    assert graph.get_node(1).run_id == parent_fence
    assert ralph.stdout_requests == [round_run_id]
    assert notified.call_args.args[0] == round_run_id

    approved = executor.complete(1, "main")
    assert approved.rebased is True
    assert graph.get_node(1).status.value == "done"


def test_review_notification_failure_is_explicit_for_block_and_warn(
    graph, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for node_id, policy in ((1, "block"), (2, "warn")):
        graph.add_node(f"notification {policy}")
        executor = _executor(graph, tmp_path, _ReviewRalph([False, False]))
        executor.dispatch(node_id, _config(tmp_path, on_reject=policy))
        monkeypatch.setattr(
            graph,
            "deposit_run_message",
            MagicMock(side_effect=RuntimeError("notification store unavailable")),
        )

        first = executor.complete(node_id, "main")
        assert first.redispatch is not None
        result = executor.complete(node_id, "main")
        assert result.review_notification_failed is True
        if policy == "block":
            assert result.rebased is False
            assert result.blocked is True
            assert graph.get_node(node_id).status.value == "blocked"
        else:
            assert result.rebased is True
            assert graph.get_node(node_id).status.value == "done"


def test_review_prompt_requires_dispatch_base_oid(graph, tmp_path: Path) -> None:
    executor = _executor(graph, tmp_path, _ReviewRalph([True]))
    with pytest.raises(ValueError, match="dispatch base oid"):
        executor._review_prompt(
            MikadoNode(id=20, description="missing base"),
            tmp_path,
            _config(tmp_path, session_mode="fresh"),
        )


def test_review_redispatch_fails_closed_when_fence_changes(
    graph, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph.add_node("unfenced redispatch")
    executor = _executor(graph, tmp_path, _ReviewRalph([True]))
    node = graph.get_node(1)
    from milknado.domains.dispatch._runstate import now_iso

    assert node is not None
    executor._worker_run_id_by_node[1] = "worker-1"
    executor._base_oid_by_node[1] = "base-1"
    monkeypatch.setattr(graph, "replace_run_id", MagicMock(return_value=False))
    with pytest.raises(ValueError, match="fence lost"):
        executor._redispatch_review_round(node, _config(tmp_path), tmp_path)

    graph.add_node("adopted fence")
    assert graph.claim_node(2, "parent-fence", now=now_iso())
    executor._owner_fence_by_node[2] = "parent-fence"
    executor._worker_run_id_by_node[2] = "worker-2"
    executor._base_oid_by_node[2] = "base-2"
    original_get_node = graph.get_node

    def changed_fence(node_id: int) -> MikadoNode | None:
        current = original_get_node(node_id)
        if node_id == 2 and current is not None:
            current = replace(current, run_id="changed-fence")
        return current

    monkeypatch.setattr(graph, "get_node", changed_fence)
    node = original_get_node(2)
    assert node is not None
    with pytest.raises(ValueError, match="owner fence lost"):
        executor._redispatch_review_round(node, _config(tmp_path), tmp_path)


def test_review_failure_blocks_without_redispatch(
    graph, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FailingReviewRalph(_ReviewRalph):
        def run_node_review(
            self, agent: str, prompt: str, worktree: Path, project_root: Path
        ) -> ReviewVerdict:
            raise RuntimeError("review process failed")

    executor = _executor(graph, tmp_path, FailingReviewRalph([True]))
    graph.add_node("review failure")
    monkeypatch.setattr(graph, "deposit_run_message", MagicMock(return_value=1))
    executor.dispatch(1, _config(tmp_path))
    result = executor.complete(1, "main")
    assert result.blocked is True
    assert result.redispatch is None
    assert graph.get_node(1).status.value == "blocked"


def test_review_drain_reports_timeout_and_stop_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    class Manager:
        def __init__(self, stop_error: Exception | None = None) -> None:
            self.stop_error = stop_error
            self.stopped = False

        def stop_and_join(self, run_id: str, timeout: float) -> None:
            del run_id, timeout
            self.stopped = True
            if self.stop_error:
                raise self.stop_error

    timeout_manager = Manager()
    clock = iter((0.0, 1801.0))
    monkeypatch.setattr("milknado.adapters.loop.time.monotonic", lambda: next(clock))
    timed_out = _drain_review_run(timeout_manager, "timeout", queue.Queue())
    assert timed_out.approved is False
    assert timeout_manager.stopped is True

    empty_manager = Manager()
    empty_events = MagicMock()
    empty_events.get.side_effect = queue.Empty
    monkeypatch.setattr("milknado.adapters.loop.time.monotonic", lambda: 0.0)
    empty = _drain_review_run(empty_manager, "empty", empty_events)
    assert empty.approved is False
    assert empty_manager.stopped is True

    failed_manager = Manager(RuntimeError("stop failed"))
    failed_events = MagicMock()
    failed_events.get.side_effect = RuntimeError("event stream failed")
    failed = _drain_review_run(failed_manager, "failed", failed_events)
    assert failed.approved is False
    assert "event stream failed" in failed.findings_md


def test_agent_session_parser_rejects_bad_shapes() -> None:
    assert capture_session_id("codex", '[{"session_id":"s"}]') == "s"
    with pytest.raises(ValueError, match="could not parse"):
        capture_session_id("claude", "not-json")
    with pytest.raises(ValueError, match="no session_id"):
        capture_session_id("claude", "{}")
    with pytest.raises(ValueError, match="empty"):
        build_resume_command("", "claude", "s")
    with pytest.raises(ValueError, match="unsupported"):
        build_resume_command("claude", "unknown", "s")


def test_brief_resolves_cheese_path_and_missing_fallback(tmp_path: Path) -> None:
    node = MikadoNode(id=13, description="spec", artifact_path=".cheese/specs/a.md")
    assert _resolve_spec_path(node, [], tmp_path) == str(
        (tmp_path / ".cheese" / "specs" / "a.md").resolve()
    )
    no_artifact = MikadoNode(id=14, description="none")
    ancestor = MikadoNode(id=15, description="ancestor", artifact_path="missing.md")
    fallback = _resolve_spec_path(no_artifact, [ancestor], tmp_path)
    assert fallback is not None
    assert fallback.endswith("/specs/missing.md")
    assert _resolve_spec_path(no_artifact, [], tmp_path) is None


def test_adapter_stdout_and_review_parser_paths(tmp_path: Path) -> None:
    adapter = LoopAdapter()
    adapter._manager = cast(Any, SimpleNamespace(get_run=lambda run_id: None))
    assert adapter.get_run_stdout("missing") == []
    runs = [
        (
            SimpleNamespace(
                state=SimpleNamespace(last_captured_stdout="a\nb", last_result_text="unused")
            ),
            ["a", "b"],
        ),
        (
            SimpleNamespace(
                state=SimpleNamespace(last_captured_stdout=None, last_result_text="c")
            ),
            ["c"],
        ),
    ]
    for run, expected in runs:
        adapter._manager = cast(Any, SimpleNamespace(get_run=lambda run_id, run=run: run))
        assert adapter.get_run_stdout("run") == expected

    assert _parse_review_output("no tag")[0] is False
    assert _parse_review_output("<verdict>reject</verdict>")[0] is False


def test_loop_adapter_runs_bounded_review_in_pinned_worktree(monkeypatch, tmp_path: Path) -> None:
    class FakeManager:
        def __init__(self) -> None:
            self.emitter = None

        def create_run(self, config, emitter, run_id=None):
            del config
            self.emitter = emitter
            return SimpleNamespace(state=SimpleNamespace(run_id=run_id or "review-run"))

        def start_run(self, run_id: str) -> None:
            assert run_id == "review-run"
            self.emitter.queue.put(
                Event(
                    EventType.ITERATION_FAILED,
                    run_id,
                    {"echo_stdout": "<verdict>approve</verdict>"},
                )
            )
            self.emitter.queue.put(Event(EventType.RUN_STOPPED, run_id, {}))

    monkeypatch.setattr("milknado.adapters.loop.RunManager", FakeManager)
    adapter = LoopAdapter()
    verdict = adapter.run_node_review("age", "review prompt", tmp_path, tmp_path)
    assert verdict.approved is True


def test_adapter_review_drain_collects_iteration_output() -> None:
    events: queue.Queue[Any] = queue.Queue()
    events.put(
        Event(
            EventType.ITERATION_COMPLETED,
            "r",
            {"result_text": "<verdict>approve</verdict>"},
        )
    )
    events.put(Event(EventType.RUN_STOPPED, "r", {}))
    manager = SimpleNamespace()
    result = _drain_review_run(manager, "r", events)
    assert result.approved is True


class _HeadlessRoundExecutor:
    def __init__(self) -> None:
        self.round = 0

    def dispatch(self, node_id: int, config: ExecutionConfig, *, base_oid=None) -> DispatchResult:
        self.round += 1
        return DispatchResult(node_id, Path("/tmp/wt"), f"run-{self.round}")

    def complete(self, node_id: int, feature_branch: str) -> CompletionResult:
        if self.round == 1:
            self.round = 2
            return CompletionResult(
                node_id,
                rebased=False,
                newly_ready=[],
                redispatch=DispatchResult(node_id, Path("/tmp/wt"), "run-2"),
            )
        return CompletionResult(node_id, rebased=True, newly_ready=[])


class _HeadlessRoundRalph:
    def wait_for_next_completion(self, run_ids, timeout=None):
        return next(iter(run_ids)), "completed"


def test_headless_follows_review_redispatch() -> None:
    config = ExecutionConfig("agent", (), "wt", Path("/tmp"))
    outcome = run_node_to_completion(
        _HeadlessRoundExecutor(),
        _HeadlessRoundRalph(),
        17,
        config,
        "main",
        30.0,
    )
    assert outcome.success is True


def test_completion_handler_tracks_review_round_and_block_paths() -> None:
    def make_loop(result: CompletionResult):
        return SimpleNamespace(
            _active={"run-1": 1},
            _dispatched_at={"run-1": 0.0},
            _progress_by_run={},
            _input=SimpleNamespace(overlay_state=None),
            _graph=SimpleNamespace(get_node=lambda node_id: MikadoNode(node_id, "handler node")),
            _executor=SimpleNamespace(complete=lambda node_id, branch: result),
            _completion_durations=[],
            _logs=[],
            _attempts={},
            _strict=True,
            _failure_triggered=False,
        )

    live = SimpleNamespace(console=MagicMock())
    redispatch = CompletionResult(
        1,
        rebased=False,
        newly_ready=[],
        redispatch=DispatchResult(1, Path("/tmp/wt"), "run-2"),
    )
    loop = make_loop(redispatch)
    assert handle_completion(loop, "run-1", "completed", "main", live) == (0, 0, [])
    assert "run-2" in loop._active

    blocked = make_loop(CompletionResult(1, rebased=False, newly_ready=[], blocked=True))
    assert handle_completion(blocked, "run-1", "completed", "main", live)[1] == 1
    conflict = RebaseConflict(1, "handler node", ("a.py",), "conflict")
    failed = make_loop(
        CompletionResult(1, rebased=False, newly_ready=[], rebase_conflict=conflict)
    )
    assert handle_completion(failed, "run-1", "completed", "main", live)[2] == [conflict]
    passed = make_loop(CompletionResult(1, rebased=True, newly_ready=[]))
    assert handle_completion(passed, "run-1", "completed", "main", live)[0] == 1
