from dataclasses import FrozenInstanceError

import pytest

from milknado.domains.common import ProgressEvent
from milknado.domains.execution.run_loop._scheduler import Scheduler
from milknado.domains.execution.run_loop.state import TerminalRunState
from milknado.loop import RunStatus


def test_scheduler_lifecycle_records_immutable_history_and_attempts() -> None:
    scheduler = Scheduler(eta_sample_size=2)
    scheduler.admit_run("run-1", 7, 10.0)
    before = scheduler.view()
    scheduler.record_progress(ProgressEvent(run_id="run-1", work=1, total=2, message="half"))
    terminal = TerminalRunState("run-1", 7, "build", RunStatus.FAILED, (), (), 0.0)
    finished = scheduler.finish_run("run-1", terminal, 14.0)
    scheduler.record_completion(finished.duration)
    scheduler.record_failure(finished.node_id, strict=True)
    scheduler.count_outcomes(failed=1)

    after = scheduler.view()
    assert before.active[0].progress is None
    assert finished.node_id == 7 and finished.duration == 4.0
    assert after.active == ()
    assert after.terminal_runs[0].duration_seconds == 4.0
    assert after.completion_durations == (4.0,)
    assert after.failed == 1 and after.failure_triggered
    scheduler.admit_run("run-2", 7, 20.0)
    assert scheduler.view().active[0].prior_attempts == 1
    with pytest.raises(FrozenInstanceError):
        after.__setattr__("failed", 2)


def test_scheduler_deferral_stop_and_reset_sequence() -> None:
    scheduler = Scheduler(eta_sample_size=2)
    scheduler.defer_capacity()
    assert scheduler.retry_deferred(10.0, 1.0)
    assert not scheduler.retry_deferred(10.5, 1.0)
    assert scheduler.retry_deferred(11.0, 1.0)
    scheduler.begin_dispatch()
    scheduler.admit_run("run-1", 3, 12.0)
    scheduler.record_stop(3)
    scheduler.close_admission()

    stopped = scheduler.view()
    assert not stopped.capacity_deferred
    assert stopped.scheduling_stopped
    assert stopped.stopped_nodes == {3}
    assert stopped.stopped == 1
    assert scheduler.abandon_run("run-1") == 3
    assert scheduler.view().active == ()
    scheduler.reset_run()
    assert scheduler.view().stopped_nodes == frozenset()
    assert scheduler.view().stopped == 0
    assert scheduler.view().scheduling_stopped
