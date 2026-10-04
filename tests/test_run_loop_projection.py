from dataclasses import FrozenInstanceError, replace

import pytest

from milknado.domains.execution.run_loop._projection import project_state
from milknado.domains.execution.run_loop.state import ActiveRunFacts, ProjectionFacts, RunLoopState
from milknado.loop import RunStatus


def facts(*runs: ActiveRunFacts, durations: tuple[float, ...] = ()) -> ProjectionFacts:
    return ProjectionFacts(
        goal="ship controller",
        active_runs=runs,
        terminal_runs=(),
        completed=0,
        failed=0,
        stopped=0,
        available=2,
        event_lines=(),
        execution_agent="codex",
        log_path=None,
        completion_durations=durations,
        stall_threshold_seconds=300,
        max_attempts=3,
    )


def active(**changes: object) -> ActiveRunFacts:
    base = ActiveRunFacts(
        run_id="run-1",
        node_id=7,
        description="build snapshots",
        status=RunStatus.RUNNING,
        stop_requested=False,
        force_stop_requested=False,
        output=("line",),
        pending_guidance=(),
        dispatched_at=100.0,
        prior_attempts=0,
    )
    return replace(base, **changes)


def test_empty_projection_is_deterministic_and_immutable() -> None:
    state = project_state(facts(), 125.0)
    assert state == RunLoopState(
        goal="ship controller",
        active_runs=(),
        terminal_runs=(),
        completed=0,
        failed=0,
        stopped=0,
        available=2,
        event_lines=(),
        execution_agent="codex",
        log_path=None,
    )
    assert state == project_state(facts(), 125.0)
    with pytest.raises(FrozenInstanceError):
        state.available = 3  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.parametrize(
    ("run", "durations", "now", "expected"),
    [
        (active(), (), 100.0, (0.0, None, False, None)),
        (active(dispatched_at=None), (), 200.0, (0.0, None, False, None)),
        (active(), (10.0, 20.0), 105.0, (5.0, None, False, None)),
        (active(), (10.0, 20.0, 30.0), 105.0, (5.0, 15.0, False, None)),
        (active(), (1.0, 2.0, 3.0), 200.0, (100.0, 0.0, False, None)),
        (active(dispatched_at=0.0), (), 299.0, (299.0, None, False, None)),
        (active(dispatched_at=0.0), (), 300.0, (300.0, None, True, None)),
        (
            active(dispatched_at=0.0, progress_work=3, progress_total=4),
            (),
            300.0,
            (300.0, None, False, 75.0),
        ),
        (active(progress_work=0, progress_total=0), (), 105.0, (5.0, None, False, None)),
    ],
)
def test_running_projection_boundaries(
    run: ActiveRunFacts,
    durations: tuple[float, ...],
    now: float,
    expected: tuple[float, float | None, bool, float | None],
) -> None:
    snapshot = project_state(facts(run, durations=durations), now).active_runs[0]
    actual = (
        snapshot.elapsed_seconds,
        snapshot.eta_seconds,
        snapshot.stalled,
        snapshot.progress_pct,
    )
    assert actual == expected


def test_stopped_run_actions_attempts_and_tails() -> None:
    run = active(
        status=RunStatus.STOPPED,
        stop_requested=True,
        force_stop_requested=True,
        progress_message="building",
        progress_work=1,
        progress_total=2,
        output=tuple(str(i) for i in range(35)),
        prior_attempts=2,
    )
    snapshot = project_state(facts(run), 105.0).active_runs[0]
    assert snapshot.progress == "building"
    assert snapshot.attempt == 3
    assert snapshot.max_attempts == 3
    assert snapshot.output == tuple(str(i) for i in range(5, 35))
    assert snapshot.actions.cancel_reason == "run has stopped"
    assert snapshot.actions.guidance_reason == "run has stopped"
    assert snapshot.actions.force_stop_reason == "run has stopped"
    assert project_state(facts(run), 105.0) == project_state(facts(run), 105.0)
