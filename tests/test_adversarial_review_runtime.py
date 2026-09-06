from __future__ import annotations

import json
import queue
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol, TypedDict, Unpack, cast
from unittest.mock import MagicMock

import pytest
from rich.live import Live

from milknado.adapters._loop_types import ReviewVerdict
from milknado.adapters.loop import (
    LoopAdapter,
    _drain_review_run,  # pyright: ignore[reportPrivateUsage]
    _parse_review_verdict,  # pyright: ignore[reportPrivateUsage]
)
from milknado.domains.common import (
    Gate,
    GitPort,
    MikadoNode,
    ProgressEvent,
    TerminalRunOutcome,
    VerifySpecResult,
)
from milknado.domains.common.agent_argv import (
    NodeAgentSession,
    build_resume_command,
    capture_session_id,
)
from milknado.domains.dispatch.brief import (
    _resolve_spec_path,  # pyright: ignore[reportPrivateUsage]
)
from milknado.domains.execution import (
    CompletionResult,
    DispatchResult,
    ExecutionConfig,
    Executor,
    RebaseConflict,
    RunLoop,
    run_node_to_completion,
)
from milknado.domains.execution._review import build_review_prompt
from milknado.domains.execution.executor import RuntimePolicy
from milknado.domains.execution.run_loop._completion import handle_completion
from milknado.domains.graph import MikadoGraph
from milknado.loop._events import (
    Event,
    EventData,
    EventType,
    IterationEndedData,
    NoData,
    NullEmitter,
    QueueEmitter,
)
from milknado.loop._run_types import RunConfig, RunState, RunStatus
from milknado.loop.manager import ManagedRun, RunManager
from tests.test_execution import FakeCrg, FakeGit


def _iteration_payload(
    *, result_text: str | None = None, echo_stdout: str | None = None
) -> IterationEndedData:
    """Full iteration payload matching what the loop engine actually emits."""
    return IterationEndedData(
        iteration=1,
        returncode=0,
        duration=0.0,
        detail="",
        log_file=None,
        result_text=result_text,
        echo_stdout=echo_stdout,
    )


@dataclass(frozen=True)
class _RunState:
    _run_id: str

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def status(self) -> RunStatus:
        return RunStatus.RUNNING

    @property
    def total(self) -> int:
        return 1

    @property
    def stop_requested(self) -> bool:
        return False

    @property
    def force_stop_requested(self) -> bool:
        return False


@dataclass(frozen=True)
class _Run:
    _state: _RunState

    @property
    def state(self) -> _RunState:
        return self._state


class _Console:
    def __init__(self) -> None:
        self.print_calls: list[tuple[object, ...]] = []

    def print(self, *args: object, **_kwargs: object) -> None:
        self.print_calls.append(args)


def _handler_loop(result: CompletionResult) -> RunLoop:
    def _get_node(node_id: int) -> MikadoNode:
        return MikadoNode(node_id, "handler node")

    def _complete(_node_id: int, _branch: str) -> CompletionResult:
        return result

    loop = SimpleNamespace(
        _active={"run-1": 1},
        _dispatched_at={"run-1": 0.0},
        _progress_by_run={},
        _input=SimpleNamespace(overlay_state=None),
        _graph=SimpleNamespace(get_node=_get_node),
        _executor=SimpleNamespace(complete=_complete),
        _completion_durations=deque[float](),
        _logs=deque[str](),
        _attempts={},
        _strict=True,
        _failure_triggered=False,
        _stopped_nodes=set(),
        _stopped=0,
        _terminal_runs=deque(),
        _ralph=SimpleNamespace(),
    )
    # The test double implements only the fields used by the completion handler.
    return cast(RunLoop, cast(object, loop))


def _handler_live() -> tuple[Live, _Console]:
    console = _Console()
    live_double = SimpleNamespace(console=console)
    # Live exposes a large concrete surface; only .console is exercised here,
    # so a single justified cast is the narrowest honest representation.
    return cast(Live, live_double), console  # pyright: ignore[reportInvalidCast]


class _ReviewRalph:
    def __init__(self, verdicts: list[bool]) -> None:
        self.verdicts: Iterator[bool] = iter(verdicts)
        self.created: list[dict[str, object]] = []
        self.started: list[str] = []
        self.stdout_requests: list[str] = []
        self.reviews: list[tuple[str, str, Path]] = []
        self.ralph_md_calls: list[dict[str, object]] = []
        self._next_id: int = 0

    def create_run(
        self,
        agent: str,
        ralph_dir: Path,
        ralph_file: Path,
        quality_gates: tuple[Gate, ...] | None,
        project_root: Path | None = None,
        commit_footer: str | None = None,
        base_oid: str | None = None,
        runtime_policy: object | None = None,
        run_id: str | None = None,
        completion_probe: Callable[[], bool] | None = None,
    ) -> _Run:
        self._next_id += 1
        resolved_run_id = run_id or f"run-{self._next_id}"
        self.created.append(
            {
                "agent": agent,
                "ralph_dir": ralph_dir,
                "ralph_file": ralph_file,
                "quality_gates": quality_gates,
                "project_root": project_root,
                "commit_footer": commit_footer,
                "base_oid": base_oid,
                "runtime_policy": runtime_policy,
                "run_id": resolved_run_id,
                "completion_probe": completion_probe,
            }
        )
        return _Run(_RunState(resolved_run_id))

    def start_run(self, run_id: str) -> None:
        self.started.append(run_id)

    def queue_guidance(self, run_id: str, text: str) -> bool:
        _ = run_id, text
        return False

    def request_stop_run(self, run_id: str) -> None:
        _ = run_id

    def stop_run(self, run_id: str, timeout: float | None = None) -> bool:
        _ = run_id, timeout
        return True

    def force_stop_run(self, run_id: str, timeout: float | None = None) -> bool:
        _ = run_id, timeout
        return True

    def list_runs(self) -> Sequence[_Run]:
        return ()

    def get_run(self, run_id: str) -> _Run | None:
        _ = run_id
        return None

    def is_run_alive(self, run_id: str) -> bool:
        _ = run_id
        return False

    def get_run_stdout(self, run_id: str) -> list[str]:
        self.stdout_requests.append(run_id)
        return ['{"session_id":"session-1"}']

    def get_run_failure_detail(self, run_id: str) -> str | None:
        _ = run_id
        return None

    def get_run_output_tail(self, run_id: str, max_lines: int) -> list[str]:
        _ = run_id, max_lines
        return []

    def get_run_guidance(self, run_id: str) -> tuple[str, ...]:
        _ = run_id
        return ()

    def wait_for_next_completion(
        self,
        active_run_ids: set[str],
        timeout: float | None = None,
    ) -> tuple[str, TerminalRunOutcome | ProgressEvent]:
        _ = active_run_ids, timeout
        raise RuntimeError("Not expected in review executor tests")

    def run_node_review(
        self, agent: str, prompt: str, worktree: Path, project_root: Path
    ) -> ReviewVerdict:
        _ = project_root
        self.reviews.append((agent, prompt, worktree))
        return ReviewVerdict(
            approved=next(self.verdicts),
            findings_md="[P1][correctness] finding",
        )

    def verify_spec(self, spec_text: str, graph_state: str) -> VerifySpecResult:
        _ = spec_text, graph_state
        return VerifySpecResult(outcome="done")

    def generate_ralph_md(
        self,
        brief: str,
        quality_gates: tuple[Gate, ...] | None,
        output_path: Path,
        prior_findings: str = "",
        findings_round: int | None = None,
    ) -> Path:
        _ = brief, quality_gates
        self.ralph_md_calls.append(
            {"prior_findings": prior_findings, "findings_round": findings_round}
        )
        return output_path


class _DispatchGitFixture(Protocol):
    def create_worktree(self, path: Path, branch: str) -> Path: ...

    def rebase(self, worktree: Path, onto: str) -> object: ...

    def fast_forward(self, branch: str) -> None: ...


def _as_git_port(git: _DispatchGitFixture) -> GitPort:
    return cast(GitPort, git)


class _ConfigOverrides(TypedDict, total=False):
    execution_agent: str
    quality_gates: tuple[Gate, ...] | None
    worktree_pattern: str
    project_root: Path
    review: bool
    review_agent: str | None
    review_max_rounds: int
    session_mode: str
    agent_family: str
    on_reject: str


def _config(root: Path, **overrides: Unpack[_ConfigOverrides]) -> ExecutionConfig:
    values: _ConfigOverrides = {
        "execution_agent": "claude --model opus --effort high",
        "quality_gates": (Gate(command="true"),),
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


def _executor(
    graph: MikadoGraph,
    root: Path,
    ralph: _ReviewRalph,
    git: FakeGit | None = None,
) -> Executor:
    del root
    return Executor(
        graph=graph,
        git=_as_git_port(git or FakeGit()),
        ralph=ralph,
        crg=FakeCrg(),
    )


def test_reject_redispatches_pinned_worktree_and_resumes_session(
    graph: MikadoGraph, tmp_path: Path
) -> None:
    ralph = _ReviewRalph([False, True])
    executor = _executor(graph, tmp_path, ralph)
    _ = graph.add_node("reviewed change")

    first = executor.dispatch(1, _config(tmp_path))
    rejected = executor.complete(1, "main")
    assert rejected.redispatch is not None
    assert rejected.redispatch.worktree == first.worktree
    runtime_policy = cast(RuntimePolicy, ralph.created[1]["runtime_policy"])
    assert runtime_policy.session is not None
    assert runtime_policy.session.session_id == "session-1"
    assert runtime_policy.session.worktree_path == str(first.worktree.resolve())

    approved = executor.complete(1, "main")
    assert approved.rebased is True
    node = graph.get_node(1)
    assert node is not None
    assert node.status.value == "done"
    session_file = tmp_path / ".milknado" / "sessions" / "node-1.json"
    assert json.loads(session_file.read_text())["session_id"] == "session-1"
    assert (
        first.worktree / ".cheese" / "age" / "reviewed-change.md"
    ).read_text() == "[P1][correctness] finding\n"


def test_redispatch_threads_findings_into_ralph_regeneration(
    graph: MikadoGraph, tmp_path: Path
) -> None:
    """#298: the redispatched worker's RALPH.md regenerates with the rejecting
    round's findings threaded in memory — required reading, not re-derived."""
    ralph = _ReviewRalph([False, True])
    executor = _executor(graph, tmp_path, ralph)
    _ = graph.add_node("findings thread")

    _ = executor.dispatch(1, _config(tmp_path))
    rejected = executor.complete(1, "main")

    assert rejected.redispatch is not None
    assert ralph.ralph_md_calls[0] == {"prior_findings": "", "findings_round": None}
    assert ralph.ralph_md_calls[1]["prior_findings"] == "[P1][correctness] finding"
    assert ralph.ralph_md_calls[1]["findings_round"] == 1


def test_notify_review_dual_writes_node_reviews_and_run_messages(
    graph: MikadoGraph, tmp_path: Path
) -> None:
    """#298: every notified verdict lands in node_reviews (audit trail, no runs
    dependency) AND in run_messages (FK satisfied by the curd-A runs row)."""
    ralph = _ReviewRalph([False, True])
    executor = _executor(graph, tmp_path, ralph)
    _ = graph.add_node("verdict audit")

    _ = executor.dispatch(1, _config(tmp_path))
    _ = executor.complete(1, "main")

    rows = graph.node_reviews_for_node(1)
    assert len(rows) == 1
    assert rows[0]["verdict"] == "reject"
    assert rows[0]["round"] == 1
    assert rows[0]["findings"] == "[P1][correctness] finding"
    worker_run_id = ralph.started[0]
    assert graph.latest_run_message(worker_run_id, "node_review") is not None


def test_findings_delivered_in_memory_when_db_writes_fail(
    graph: MikadoGraph, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#298: the DB being down at redispatch must not lose the findings — the
    handoff is the in-memory thread, never a DB read-back."""
    ralph = _ReviewRalph([False, True])
    executor = _executor(graph, tmp_path, ralph)
    _ = graph.add_node("db down redispatch")

    _ = executor.dispatch(1, _config(tmp_path))
    monkeypatch.setattr(
        graph, "deposit_run_message", MagicMock(side_effect=RuntimeError("db down"))
    )
    rejected = executor.complete(1, "main")

    assert rejected.review_notification_failed is True
    assert rejected.redispatch is not None
    assert ralph.ralph_md_calls[1]["prior_findings"] == "[P1][correctness] finding"
    assert ralph.ralph_md_calls[1]["findings_round"] == 1


def test_second_rejection_round_accumulates_audits_and_labels_round_2(
    graph: MikadoGraph, tmp_path: Path
) -> None:
    """#298: consecutive rejections are distinct rounds — node_reviews keeps one
    row per round (PK is (node_id, round)) and the round-2 RALPH regeneration
    is labeled round 2, so the worker can tell its second attempt was rejected
    again rather than mistaking it for a fresh dispatch."""
    ralph = _ReviewRalph([False, False, True])
    executor = _executor(graph, tmp_path, ralph)
    _ = graph.add_node("two rejections")

    _ = executor.dispatch(1, _config(tmp_path, review_max_rounds=2))
    first = executor.complete(1, "main")
    second = executor.complete(1, "main")
    assert first.redispatch is not None and second.redispatch is not None

    rows = graph.node_reviews_for_node(1)
    assert [r["round"] for r in rows] == [1, 2]
    assert all(r["verdict"] == "reject" for r in rows)
    assert ralph.ralph_md_calls[1]["findings_round"] == 1
    assert ralph.ralph_md_calls[2]["findings_round"] == 2
    assert ralph.ralph_md_calls[2]["prior_findings"] == "[P1][correctness] finding"


def test_zero_review_rounds_preserves_merge_path(graph: MikadoGraph, tmp_path: Path) -> None:
    ralph = _ReviewRalph([False])
    git = FakeGit()
    executor = _executor(graph, tmp_path, ralph, git)
    _ = graph.add_node("no review")

    _ = executor.dispatch(1, _config(tmp_path, review_max_rounds=0))
    result = executor.complete(1, "main")
    assert result.rebased is True
    assert not ralph.reviews


def test_warn_policy_merges_after_review_round_budget(graph: MikadoGraph, tmp_path: Path) -> None:
    ralph = _ReviewRalph([False, False])
    git = FakeGit()
    executor = _executor(graph, tmp_path, ralph, git)
    _ = graph.add_node("warn review")

    _ = executor.dispatch(1, _config(tmp_path, on_reject="warn"))
    redispatch = executor.complete(1, "main").redispatch
    assert redispatch is not None
    result = executor.complete(1, "main")
    assert result.rebased is True
    node = graph.get_node(1)
    assert node is not None
    assert node.status.value == "done"
    assert git.rebases


def test_block_policy_retains_pinned_worktree(
    graph: MikadoGraph, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ralph = _ReviewRalph([False, False])
    git = FakeGit()
    executor = _executor(graph, tmp_path, ralph, git)
    _ = graph.add_node("blocked review")
    monkeypatch.setattr(graph, "deposit_run_message", MagicMock(return_value=1))

    first = executor.dispatch(1, _config(tmp_path))
    assert executor.complete(1, "main").redispatch is not None
    result = executor.complete(1, "main")
    node = graph.get_node(1)
    assert result.blocked is True
    assert node is not None
    assert node.status.value == "blocked"
    assert node.worktree_path == str(first.worktree)
    assert not git.rebases
    assert not git.removed


def test_durable_spec_path_prefers_xdg_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    xdg = tmp_path / "xdg"
    durable = xdg / "cheese" / tmp_path.name / "specs" / "accepted.md"
    durable.parent.mkdir(parents=True)
    _ = durable.write_text("# Accepted\n", encoding="utf-8")
    monkeypatch.setenv("XDG_DATA_HOME", str(xdg))
    node = MikadoNode(id=1, description="spec", artifact_path="accepted.md")

    assert _resolve_spec_path(node, [], tmp_path) == str(durable.resolve())


def test_loop_adapter_resume_command_keeps_cwd_and_controls(tmp_path: Path) -> None:
    prompt_file = tmp_path / "RALPH.md"
    _ = prompt_file.write_text("prompt", encoding="utf-8")
    session = NodeAgentSession(1, "claude", "s1", str(tmp_path), "now")
    adapter = LoopAdapter()
    run = adapter.create_run(
        "claude --model opus --effort high",
        tmp_path,
        prompt_file,
        (Gate(command="true"),),
        project_root=tmp_path,
        runtime_policy=SimpleNamespace(session=session),
    )
    assert "--resume s1" in run.config.agent
    assert "--model opus" in run.config.agent
    assert "--effort high" in run.config.agent
    assert run.config.project_root == tmp_path


def test_capture_session_handles_nested_json_and_failures(
    graph: MikadoGraph, tmp_path: Path
) -> None:
    ralph = _ReviewRalph([True])
    ralph.get_run_stdout = lambda run_id: [  # type: ignore[method-assign]
        "not-json",
        '{"event":{"session_id":"nested"}}',
    ]
    executor = _executor(graph, tmp_path, ralph)
    node = MikadoNode(id=9, description="session", run_id="worker", worktree_path=str(tmp_path))
    session = executor._capture_session(  # pyright: ignore[reportPrivateUsage]
        node, _config(tmp_path, agent_family="codex")
    )
    assert session is not None
    assert session.session_id == "nested"
    assert (
        executor._capture_session(  # pyright: ignore[reportPrivateUsage]
            node, _config(tmp_path, agent_family="codex")
        )
        == session
    )

    with pytest.raises(ValueError, match="no worker run id"):
        _ = executor._capture_session(  # pyright: ignore[reportPrivateUsage]
            MikadoNode(id=10, description="missing"),
            _config(tmp_path),
        )
    ralph.get_run_stdout = lambda run_id: ["{}"]  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="resumable session id"):
        _ = executor._capture_session(  # pyright: ignore[reportPrivateUsage]
            MikadoNode(id=11, description="invalid", run_id="worker"),
            _config(tmp_path),
        )


def test_executor_review_helpers_cover_spec_and_missing_reviewer(
    graph: MikadoGraph, tmp_path: Path
) -> None:
    ralph = _ReviewRalph([True])
    git = FakeGit()
    executor = _executor(graph, tmp_path, ralph, git)
    node = MikadoNode(id=12, description="helper", artifact_path="spec.md")
    executor._base_oid_by_node[node.id] = "base-oid"  # pyright: ignore[reportPrivateUsage]
    _ = (tmp_path / "RALPH.md").write_text("generated context", encoding="utf-8")
    prompt = build_review_prompt(
        node,
        tmp_path,
        _config(tmp_path, session_mode="fresh").project_root,
        "direct helper diff",
    )
    assert str((tmp_path / "spec.md").resolve()) in prompt
    assert "generated context" in prompt
    assert "exactly one verdict tag" in prompt
    _ = executor._run_review(  # pyright: ignore[reportPrivateUsage]
        node, tmp_path, _config(tmp_path, session_mode="fresh")
    )
    assert "diff from base-oid" in ralph.reviews[0][1]
    _ = executor._notify_review(  # pyright: ignore[reportPrivateUsage]
        node, verdict="reject", findings_md="finding"
    )

    class _MissingReviewer:
        """Intentionally missing reviewer: no run_node_review method."""

    no_reviewer = Executor(
        graph=graph,
        git=_as_git_port(FakeGit()),
        ralph=_MissingReviewer(),  # pyright: ignore[reportArgumentType]
        crg=FakeCrg(),
    )
    with pytest.raises(ValueError, match="requires a LoopPort reviewer"):
        _ = no_reviewer._run_review(  # pyright: ignore[reportPrivateUsage]
            node, tmp_path, _config(tmp_path, session_mode="fresh")
        )
    with pytest.raises(ValueError, match="no run fence"):
        _ = executor._redispatch_review_round(  # pyright: ignore[reportPrivateUsage]
            node,
            _config(tmp_path, session_mode="fresh"),
            tmp_path,
        )


def test_review_true_without_reviewer_fails_before_dispatch(
    graph: MikadoGraph, tmp_path: Path
) -> None:
    executor = _executor(graph, tmp_path, _ReviewRalph([True]))
    _ = graph.add_node("missing reviewer")

    with pytest.raises(ValueError, match="review=true"):
        _ = executor.dispatch(1, _config(tmp_path, review_agent="  "))


def test_adopted_review_keeps_parent_fence_and_notifies_worker(
    graph: MikadoGraph, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from milknado.domains.dispatch._runstate import now_iso

    ralph = _ReviewRalph([False, True])
    executor = _executor(graph, tmp_path, ralph)
    _ = graph.add_node("adopted review")
    parent_fence = "parent-fence"
    assert graph.claim_node(1, parent_fence, now=now_iso())
    notified = MagicMock(return_value=1)
    monkeypatch.setattr(graph, "deposit_run_message", notified)

    _ = executor.dispatch(1, _config(tmp_path), parent_run_id=parent_fence)
    rejected = executor.complete(1, "main")
    round_run_id = ralph.created[0]["run_id"]

    assert rejected.redispatch is not None
    node = graph.get_node(1)
    assert node is not None
    assert node.run_id == parent_fence
    assert ralph.stdout_requests == [round_run_id]
    assert notified.call_args.args[0] == round_run_id

    approved = executor.complete(1, "main")
    assert approved.rebased is True
    node = graph.get_node(1)
    assert node is not None
    assert node.status.value == "done"


def test_review_notification_failure_is_explicit_for_block_and_warn(
    graph: MikadoGraph, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for node_id, policy in ((1, "block"), (2, "warn")):
        _ = graph.add_node(f"notification {policy}")
        executor = _executor(graph, tmp_path, _ReviewRalph([False, False]))
        _ = executor.dispatch(node_id, _config(tmp_path, on_reject=policy))
        monkeypatch.setattr(
            graph,
            "deposit_run_message",
            MagicMock(side_effect=RuntimeError("notification store unavailable")),
        )

        first = executor.complete(node_id, "main")
        assert first.redispatch is not None
        result = executor.complete(node_id, "main")
        assert result.review_notification_failed is True
        node = graph.get_node(node_id)
        assert node is not None
        if policy == "block":
            assert result.rebased is False
            assert result.blocked is True
            assert node.status.value == "blocked"
        else:
            assert result.rebased is True
            assert node.status.value == "done"


def test_review_prompt_requires_dispatch_base_oid(graph: MikadoGraph, tmp_path: Path) -> None:
    executor = _executor(graph, tmp_path, _ReviewRalph([True]))
    with pytest.raises(ValueError, match="dispatch base oid"):
        _ = executor._run_review(  # pyright: ignore[reportPrivateUsage]
            MikadoNode(id=20, description="missing base"),
            tmp_path,
            _config(tmp_path, session_mode="fresh"),
        )


def test_review_redispatch_fails_closed_when_fence_changes(
    graph: MikadoGraph, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = graph.add_node("unfenced redispatch")
    executor = _executor(graph, tmp_path, _ReviewRalph([True]))
    node = graph.get_node(1)
    from milknado.domains.dispatch._runstate import now_iso

    assert node is not None
    executor._worker_run_id_by_node[1] = "worker-1"  # pyright: ignore[reportPrivateUsage]
    executor._base_oid_by_node[1] = "base-1"  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(graph, "replace_run_id", MagicMock(return_value=False))
    with pytest.raises(ValueError, match="fence lost"):
        _ = executor._redispatch_review_round(  # pyright: ignore[reportPrivateUsage]
            node, _config(tmp_path), tmp_path
        )

    _ = graph.add_node("adopted fence")
    assert graph.claim_node(2, "parent-fence", now=now_iso())
    executor._owner_fence_by_node[2] = "parent-fence"  # pyright: ignore[reportPrivateUsage]
    executor._worker_run_id_by_node[2] = "worker-2"  # pyright: ignore[reportPrivateUsage]
    executor._base_oid_by_node[2] = "base-2"  # pyright: ignore[reportPrivateUsage]
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
        _ = executor._redispatch_review_round(  # pyright: ignore[reportPrivateUsage]
            node, _config(tmp_path), tmp_path
        )


def test_review_failure_blocks_without_redispatch(
    graph: MikadoGraph, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FailingReviewRalph(_ReviewRalph):
        def run_node_review(  # pyright: ignore[reportImplicitOverride]
            self, agent: str, prompt: str, worktree: Path, project_root: Path
        ) -> ReviewVerdict:
            raise RuntimeError("review process failed")

    executor = _executor(graph, tmp_path, FailingReviewRalph([True]))
    _ = graph.add_node("review failure")
    monkeypatch.setattr(graph, "deposit_run_message", MagicMock(return_value=1))
    _ = executor.dispatch(1, _config(tmp_path))
    result = executor.complete(1, "main")
    assert result.blocked is True
    assert result.redispatch is None
    node = graph.get_node(1)
    assert node is not None
    assert node.status.value == "blocked"


def test_review_findings_write_failure_still_audits_and_blocks(
    graph: MikadoGraph, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    git = FakeGit()
    executor = _executor(graph, tmp_path, _ReviewRalph([True]), git)
    _ = graph.add_node("findings write failure")

    def fail_write(*_args: object, **_kwargs: object) -> None:
        raise OSError("findings unavailable")

    monkeypatch.setattr(Path, "write_text", fail_write)
    _ = executor.dispatch(1, _config(tmp_path))
    result = executor.complete(1, "main")

    assert result.blocked is True
    assert result.redispatch is None
    assert not git.rebases
    rows = graph.node_reviews_for_node(1)
    assert rows[0]["verdict"] == "error"
    assert "findings unavailable" in rows[0]["findings"]


def test_review_drain_reports_timeout_and_stop_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    class Manager(RunManager):
        def __init__(self, stop_error: Exception | None = None) -> None:
            super().__init__()
            self.stop_error: Exception | None = stop_error
            self.stopped: bool = False

        def stop_and_join(  # pyright: ignore[reportImplicitOverride]
            self, run_id: str, timeout: float | None = None
        ) -> bool:
            del run_id, timeout
            self.stopped = True
            if self.stop_error:
                raise self.stop_error
            return True

    timeout_manager = Manager()
    clock = iter((0.0, 1801.0))

    def _advance_clock() -> float:
        return next(clock)

    monkeypatch.setattr("milknado.adapters.loop.time.monotonic", _advance_clock)
    timed_out = _drain_review_run(
        timeout_manager,
        "timeout",
        queue.Queue[Event[EventData]](),
    )
    assert timed_out.approved is False
    assert timeout_manager.stopped is True

    class EmptyEvents(queue.Queue[Event[EventData]]):
        def get(  # pyright: ignore[reportImplicitOverride]
            self, block: bool = True, timeout: float | None = None
        ) -> Event[EventData]:
            _ = block, timeout
            raise queue.Empty

    empty_manager = Manager()
    empty_events = EmptyEvents()

    def _zero_clock() -> float:
        return 0.0

    monkeypatch.setattr("milknado.adapters.loop.time.monotonic", _zero_clock)
    empty = _drain_review_run(
        empty_manager,
        "empty",
        empty_events,
    )
    assert empty.approved is False
    assert empty_manager.stopped is True

    class FailedEvents(queue.Queue[Event[EventData]]):
        def get(  # pyright: ignore[reportImplicitOverride]
            self, block: bool = True, timeout: float | None = None
        ) -> Event[EventData]:
            _ = block, timeout
            raise RuntimeError("event stream failed")

    failed_manager = Manager(RuntimeError("stop failed"))
    failed_events = FailedEvents()
    failed = _drain_review_run(
        failed_manager,
        "failed",
        failed_events,
    )
    assert failed.approved is False
    assert "event stream failed" in failed.findings_md


def test_agent_session_parser_rejects_bad_shapes() -> None:
    assert capture_session_id("codex", '[{"session_id":"s"}]') == "s"
    assert capture_session_id("omp", '{"sessionId":"omp-session"}') == "omp-session"
    assert build_resume_command("omp -p --model x", "omp", "omp-session") == (
        "omp -p --model x --resume omp-session"
    )
    with pytest.raises(ValueError, match="could not parse"):
        _ = capture_session_id("claude", "not-json")
    with pytest.raises(ValueError, match="no session_id"):
        _ = capture_session_id("claude", "{}")
    with pytest.raises(ValueError, match="empty"):
        _ = build_resume_command("", "claude", "s")
    with pytest.raises(ValueError, match="unsupported"):
        _ = build_resume_command("claude", "unknown", "s")


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


class _StubManager(RunManager):
    def __init__(self, runs: dict[str, ManagedRun]) -> None:
        super().__init__()
        self._stub_runs: dict[str, ManagedRun] = runs

    def get_run(  # pyright: ignore[reportImplicitOverride]
        self, run_id: str
    ) -> ManagedRun | None:
        return self._stub_runs.get(run_id)


def _managed_run(stdout: str | None, result_text: str | None) -> ManagedRun:
    return ManagedRun(
        config=RunConfig(agent="agent", ralph_dir=Path("."), ralph_file=Path("ralph.md")),
        state=RunState(
            run_id="run",
            last_captured_stdout=stdout,
            last_result_text=result_text,
        ),
        emitter=NullEmitter(),
    )


def test_adapter_stdout_and_review_parser_paths() -> None:
    adapter = LoopAdapter()

    adapter._manager = _StubManager({})  # pyright: ignore[reportPrivateUsage]
    assert adapter.get_run_stdout("missing") == []
    runs: list[tuple[ManagedRun, list[str]]] = [
        (_managed_run("a\nb", "unused"), ["a", "b"]),
        (_managed_run(None, "c"), ["c"]),
    ]
    for run, expected in runs:
        adapter._manager = _StubManager(  # pyright: ignore[reportPrivateUsage]
            {"run": run}
        )
        assert adapter.get_run_stdout("run") == expected

    assert _parse_review_verdict("no tag").error is True
    assert _parse_review_verdict("<verdict>reject</verdict>").approved is False
    assert (
        _parse_review_verdict("<verdict>approve</verdict><verdict>unknown</verdict>").error is True
    )
    assert _parse_review_verdict("<verdict>\nanalysis\n<verdict>approve</verdict>").error is True
    assert (
        _parse_review_verdict("<verdict>approve</verdict><verdict>approve</verdict>").error is True
    )
    assert _parse_review_verdict("<verdict>approve</verdict><verdict>reject").error is True


def test_loop_adapter_runs_bounded_review_in_pinned_worktree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    created_managers: list[FakeManager] = []

    class FakeManager:
        def __init__(self) -> None:
            self.emitter: QueueEmitter | None = None
            self.received_run_id: str | None = None
            created_managers.append(self)

        def create_run(
            self,
            config: RunConfig,
            emitter: QueueEmitter,
            run_id: str | None = None,
        ) -> _Run:
            _ = config
            self.emitter = emitter
            return _Run(_RunState(run_id or "review-run"))

        def start_run(self, run_id: str) -> None:
            self.received_run_id = run_id
            assert self.emitter is not None

            self.emitter.queue.put(
                Event(
                    EventType.ITERATION_FAILED,
                    run_id,
                    _iteration_payload(echo_stdout="<verdict>approve</verdict>"),
                )
            )
            self.emitter.queue.put(Event(EventType.RUN_STOPPED, run_id, NoData()))

    monkeypatch.setattr("milknado.adapters.loop.RunManager", FakeManager)
    adapter = LoopAdapter()
    verdict = adapter.run_node_review("age", "review prompt", tmp_path, tmp_path)
    assert verdict.approved is True
    # The adapter's __init__ creates its own manager; the review run uses the
    # most recently constructed one — assert on the instance that got the run.
    fake_manager = next(m for m in created_managers if m.received_run_id is not None)
    assert fake_manager.emitter is not None
    assert fake_manager.received_run_id == "review-run"


def test_adapter_review_drain_collects_iteration_output() -> None:
    events: queue.Queue[Event[EventData]] = queue.Queue()
    events.put(
        Event(
            EventType.ITERATION_COMPLETED,
            "r",
            _iteration_payload(result_text="", echo_stdout="<verdict>approve</verdict>"),
        )
    )
    events.put(Event(EventType.RUN_STOPPED, "r", NoData()))
    manager = RunManager()
    result = _drain_review_run(manager, "r", events)
    assert result.approved is True


class _HeadlessRoundExecutor:
    def __init__(self) -> None:
        self.round: int = 0

    def dispatch(
        self,
        node_id: int,
        config: ExecutionConfig,
        *,
        base_oid: str | None = None,
        parent_run_id: str | None = None,
    ) -> DispatchResult:
        _ = config, base_oid, parent_run_id
        self.round += 1
        return DispatchResult(node_id, Path("/tmp/wt"), f"run-{self.round}")

    def complete(self, node_id: int, feature_branch: str) -> CompletionResult:
        _ = feature_branch
        if self.round == 1:
            self.round = 2
            return CompletionResult(
                node_id,
                rebased=False,
                newly_ready=[],
                redispatch=DispatchResult(node_id, Path("/tmp/wt"), "run-2"),
            )
        return CompletionResult(node_id, rebased=True, newly_ready=[])

    def cancel(self, node_id: int) -> None:
        _ = node_id

    def fail(self, node_id: int, detail: str | None = None) -> None:
        _ = node_id, detail

    def note_unconfirmed_stop(self, run_id: str) -> None:
        _ = run_id


class _HeadlessRoundRalph:
    def wait_for_next_completion(
        self, active_run_ids: set[str], timeout: float | None = None
    ) -> tuple[str, TerminalRunOutcome]:
        _ = timeout
        return next(iter(active_run_ids)), "completed"

    def stop_run(self, run_id: str, timeout: float | None = None) -> bool:
        _ = run_id, timeout
        return True


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
    live, _console = _handler_live()
    redispatch = CompletionResult(
        1,
        rebased=False,
        newly_ready=[],
        redispatch=DispatchResult(1, Path("/tmp/wt"), "run-2"),
    )
    loop = _handler_loop(redispatch)
    assert handle_completion(loop, "run-1", "completed", "main", live) == (0, 0, [])
    assert "run-2" in loop._active  # pyright: ignore[reportPrivateUsage]

    blocked = _handler_loop(CompletionResult(1, rebased=False, newly_ready=[], blocked=True))
    assert handle_completion(blocked, "run-1", "completed", "main", live)[1] == 1
    conflict = RebaseConflict(1, "handler node", ("a.py",), "conflict")
    failed = _handler_loop(
        CompletionResult(1, rebased=False, newly_ready=[], rebase_conflict=conflict)
    )
    assert handle_completion(failed, "run-1", "completed", "main", live)[2] == [conflict]
    passed = _handler_loop(CompletionResult(1, rebased=True, newly_ready=[]))
    assert handle_completion(passed, "run-1", "completed", "main", live)[0] == 1


def test_completion_handler_surfaces_review_notification_failure() -> None:
    """A failed review notification reaches the operator, orthogonal to the outcome.

    The review ran but its verdict could not be delivered to the worker. Without a
    consumer the flag was set and discarded, so the operator saw a clean completion.
    """
    result = CompletionResult(1, rebased=True, newly_ready=[], review_notification_failed=True)
    loop = _handler_loop(result)
    live, console = _handler_live()

    completed, failed, conflicts = handle_completion(loop, "run-1", "completed", "main", live)

    # The node still completes normally — the notice does not change the outcome.
    assert (completed, failed, conflicts) == (1, 0, [])
    assert any("review notification failed" in entry for entry in loop._logs)  # pyright: ignore[reportPrivateUsage]
    printed = " ".join(str(call) for call in console.print_calls)
    assert "review notification failed" in printed


def test_completion_handler_surfaces_review_audit_failure() -> None:
    result = CompletionResult(
        1, rebased=False, newly_ready=[], blocked=True, review_audit_failed=True
    )
    loop = _handler_loop(result)
    live, console = _handler_live()

    completed, failed, conflicts = handle_completion(loop, "run-1", "completed", "main", live)

    assert (completed, failed, conflicts) == (0, 1, [])
    assert any("review audit failed" in entry for entry in loop._logs)  # pyright: ignore[reportPrivateUsage]
    printed = " ".join(str(call) for call in console.print_calls)
    assert "review audit failed" in printed


def test_approval_audit_survives_worktree_cleanup(graph: MikadoGraph, tmp_path: Path) -> None:
    ralph = _ReviewRalph([True])
    git = FakeGit()
    executor = _executor(graph, tmp_path, ralph, git)
    _ = graph.add_node("approved audit")

    _ = executor.dispatch(1, _config(tmp_path))
    result = executor.complete(1, "main")

    assert result.rebased is True
    assert git.removed
    rows = graph.node_reviews_for_node(1)
    assert [(row["round"], row["verdict"]) for row in rows] == [(1, "approve")]


def test_approval_audit_failure_blocks_before_merge(
    graph: MikadoGraph, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ralph = _ReviewRalph([True])
    git = FakeGit()
    executor = _executor(graph, tmp_path, ralph, git)
    _ = graph.add_node("audit failure")
    monkeypatch.setattr(
        graph, "insert_node_review", MagicMock(side_effect=RuntimeError("audit unavailable"))
    )

    _ = executor.dispatch(1, _config(tmp_path, on_reject="warn"))
    result = executor.complete(1, "main")

    assert result.blocked is True
    assert result.review_audit_failed is True
    assert not git.rebases
    node = graph.get_node(1)
    assert node is not None
    assert node.status.value == "blocked"


@pytest.mark.parametrize("policy", ["block", "warn"])
def test_rejection_audit_failure_blocks_before_merge(
    graph: MikadoGraph, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, policy: str
) -> None:
    ralph = _ReviewRalph([False])
    git = FakeGit()
    executor = _executor(graph, tmp_path, ralph, git)
    _ = graph.add_node(f"rejection audit failure {policy}")
    monkeypatch.setattr(
        graph, "insert_node_review", MagicMock(side_effect=RuntimeError("audit unavailable"))
    )

    _ = executor.dispatch(1, _config(tmp_path, on_reject=policy, review_max_rounds=1))
    executor._review_round_by_node[1] = 1  # pyright: ignore[reportPrivateUsage]
    result = executor.complete(1, "main")

    assert result.blocked is True
    assert result.review_audit_failed is True
    assert result.redispatch is None
    assert not git.rebases


def test_review_sequence_appends_after_executor_restart(tmp_path: Path) -> None:
    db = tmp_path / "review.db"
    graph = MikadoGraph(db)
    node = graph.add_node("restart sequence")
    assert graph.insert_node_review(node.id, "reject", "old", "2026-01-01T00:00:00+00:00") == 1
    graph.close()

    reopened = MikadoGraph(db)
    executor = _executor(reopened, tmp_path, _ReviewRalph([True]))
    audit = executor._notify_review(  # pyright: ignore[reportPrivateUsage]
        node, verdict="approve", findings_md="new"
    )

    assert audit.audit_succeeded is True
    assert [row["round"] for row in reopened.node_reviews_for_node(node.id)] == [1, 2]
    reopened.close()


class _MalformedReviewRalph(_ReviewRalph):
    def run_node_review(  # pyright: ignore[reportImplicitOverride]
        self, agent: str, prompt: str, worktree: Path, project_root: Path
    ) -> ReviewVerdict:
        _ = agent, prompt, worktree, project_root
        return _parse_review_verdict("progress only")


@pytest.mark.parametrize("policy", ["block", "warn"])
def test_invalid_review_blocks_without_worker_revision(
    graph: MikadoGraph, tmp_path: Path, policy: str
) -> None:
    ralph = _MalformedReviewRalph([])
    git = FakeGit()
    executor = _executor(graph, tmp_path, ralph, git)
    _ = graph.add_node(f"invalid review {policy}")

    _ = executor.dispatch(
        1,
        _config(tmp_path, on_reject=policy, review_max_rounds=3),
    )
    result = executor.complete(1, "main")

    assert result.blocked is True
    assert result.redispatch is None
    assert len(ralph.created) == 1
    assert not git.rebases
    assert executor._review_round_by_node.get(1, 0) == 0  # pyright: ignore[reportPrivateUsage]
    rows = graph.node_reviews_for_node(1)
    assert rows[0]["verdict"] == "error"
    assert rows[0]["findings"] == "progress only"
