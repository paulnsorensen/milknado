"""Tests for the multi-run manager."""

import sys
import threading
import time
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
from _pytest.monkeypatch import MonkeyPatch

from milknado.loop._events import (
    EventType,
    QueueEmitter,
)
from milknado.loop._run_types import (
    RUN_ID_LENGTH,
    CompletionVerdict,
    RunConfig,
    RunState,
    RunStatus,
)
from milknado.loop.manager import ManagedRun, RunManager
from tests.loop.helpers import (
    MOCK_SUBPROCESS,
    drain_events,  # pyright: ignore[reportUnknownVariableType]
    event_types,  # pyright: ignore[reportUnknownVariableType]
    make_config,
    ok_proc,
)


class TestRunManagerCreateRun:
    def test_create_run_returns_managed_run(self, tmp_path: Path):
        manager = RunManager()
        config = make_config(tmp_path)
        managed = manager.create_run(config)

        assert isinstance(managed, ManagedRun)
        assert managed.config is config
        assert managed.state.status == RunStatus.PENDING
        assert managed.thread is None
        assert isinstance(managed.emitter, QueueEmitter)

    def test_create_run_assigns_unique_ids(self, tmp_path: Path):
        manager = RunManager()
        config = make_config(tmp_path)
        run1 = manager.create_run(config)
        run2 = manager.create_run(config)

        assert run1.state.run_id != run2.state.run_id

    def test_create_run_id_is_12_hex_chars(self, tmp_path: Path):
        manager = RunManager()
        config = make_config(tmp_path)
        managed = manager.create_run(config)

        run_id = managed.state.run_id
        assert len(run_id) == RUN_ID_LENGTH
        assert all(c in "0123456789abcdef" for c in run_id)


class TestRunManagerStartRun:
    @patch(MOCK_SUBPROCESS, side_effect=ok_proc)
    def test_start_run_starts_thread(self, mock_run: MagicMock, tmp_path: Path):  # pyright: ignore[reportUnusedParameter]
        manager = RunManager()
        config = make_config(tmp_path, max_iterations=1)
        managed = manager.create_run(config)
        run_id = managed.state.run_id

        manager.start_run(run_id)

        assert managed.thread is not None
        managed.thread.join(timeout=5)
        assert managed.state.status == RunStatus.COMPLETED

    @patch(MOCK_SUBPROCESS, side_effect=ok_proc)
    def test_start_run_thread_is_daemon(self, mock_run: MagicMock, tmp_path: Path):  # pyright: ignore[reportUnusedParameter]
        manager = RunManager()
        config = make_config(tmp_path, max_iterations=1)
        managed = manager.create_run(config)
        run_id = managed.state.run_id

        manager.start_run(run_id)

        assert managed.thread is not None
        assert managed.thread.daemon is True
        managed.thread.join(timeout=5)

    @patch(MOCK_SUBPROCESS, side_effect=ok_proc)
    def test_start_run_emits_events_to_queue(self, mock_run: MagicMock, tmp_path: Path):  # pyright: ignore[reportUnusedParameter]
        manager = RunManager()
        config = make_config(tmp_path, max_iterations=1)
        managed = manager.create_run(config)
        run_id = managed.state.run_id

        manager.start_run(run_id)
        assert managed.thread is not None
        managed.thread.join(timeout=5)

        events = drain_events(cast(QueueEmitter, managed.emitter))  # pyright: ignore[reportUnknownVariableType]
        types = event_types(events)
        assert EventType.RUN_STARTED in types
        assert EventType.RUN_STOPPED in types


class TestRunManagerStartRunGuards:
    @patch(MOCK_SUBPROCESS, side_effect=ok_proc)
    def test_start_run_raises_on_double_start(self, mock_run: MagicMock, tmp_path: Path):  # pyright: ignore[reportUnusedParameter]
        manager = RunManager()
        config = make_config(tmp_path, max_iterations=1)
        managed = manager.create_run(config)
        run_id = managed.state.run_id

        manager.start_run(run_id)
        with pytest.raises(RuntimeError, match="already been started"):
            manager.start_run(run_id)

        assert managed.thread is not None
        managed.thread.join(timeout=5)


class TestRunManagerInvalidRunId:
    def test_start_run_raises_key_error_for_unknown_id(self):
        manager = RunManager()
        with pytest.raises(KeyError, match="No run with ID 'nonexistent'"):
            manager.start_run("nonexistent")

    def test_stop_run_raises_key_error_for_unknown_id(self):
        manager = RunManager()
        with pytest.raises(KeyError, match="No run with ID 'nonexistent'"):
            manager.stop_run("nonexistent")

    def test_queue_guidance_raises_key_error_for_unknown_id(self):
        manager = RunManager()
        with pytest.raises(KeyError, match="No run with ID 'nonexistent'"):
            _ = manager.queue_guidance("nonexistent", "check this")


class TestRunManagerStopRun:
    @patch(MOCK_SUBPROCESS, side_effect=ok_proc)
    def test_stop_run_stops_running_run(self, mock_run: MagicMock, tmp_path: Path):  # pyright: ignore[reportUnusedParameter]
        manager = RunManager()
        config = make_config(tmp_path, max_iterations=100, delay=0.1)
        managed = manager.create_run(config)
        run_id = managed.state.run_id

        manager.start_run(run_id)
        time.sleep(0.05)

        manager.stop_run(run_id)
        assert managed.thread is not None
        managed.thread.join(timeout=5)

        assert managed.state.status == RunStatus.STOPPED

    @patch("milknado.loop.engine._run_iteration")
    def test_graceful_stop_wins_before_soft_completion(
        self, mock_iteration: MagicMock, tmp_path: Path
    ):
        verifier = MagicMock()
        manager = RunManager()
        managed = manager.create_run(make_config(tmp_path, completion_verifier=verifier))

        def simultaneous_stop(_config: RunConfig, state: RunState, *_args: object):
            state.mark_completed()
            state.request_stop()
            return False, True

        mock_iteration.side_effect = simultaneous_stop
        manager.start_run(managed.state.run_id)
        assert managed.thread is not None
        managed.thread.join(timeout=1)

        assert not managed.thread.is_alive()
        assert managed.state.status is RunStatus.STOPPED
        assert managed.state.completed == 1
        verifier.assert_not_called()

    @patch("milknado.loop.engine._run_iteration", return_value=(True, True))
    def test_graceful_stop_wins_while_completion_verifier_is_running(
        self, _mock_iteration: MagicMock, tmp_path: Path
    ):
        verifier_started = threading.Event()
        release_verifier = threading.Event()

        def blocking_verifier():
            verifier_started.set()
            assert release_verifier.wait(timeout=1)
            return CompletionVerdict(ok=True, feedback="")

        manager = RunManager()
        managed = manager.create_run(make_config(tmp_path, completion_verifier=blocking_verifier))
        manager.start_run(managed.state.run_id)
        assert verifier_started.wait(timeout=1)

        manager.stop_run(managed.state.run_id)
        release_verifier.set()
        assert managed.thread is not None
        managed.thread.join(timeout=1)

        assert not managed.thread.is_alive()
        assert managed.state.status is RunStatus.STOPPED

    @patch("milknado.loop.engine._run_iteration", return_value=(True, True))
    def test_graceful_stop_wins_when_blocked_completion_verifier_raises(
        self, _mock_iteration: MagicMock, tmp_path: Path
    ):
        verifier_started = threading.Event()
        release_verifier = threading.Event()

        def failing_verifier():
            verifier_started.set()
            assert release_verifier.wait(timeout=1)
            raise RuntimeError("verifier failed")

        manager = RunManager()
        managed = manager.create_run(make_config(tmp_path, completion_verifier=failing_verifier))
        manager.start_run(managed.state.run_id)
        assert verifier_started.wait(timeout=1)

        manager.stop_run(managed.state.run_id)
        release_verifier.set()
        assert managed.thread is not None
        managed.thread.join(timeout=1)

        assert not managed.thread.is_alive()
        assert managed.state.status is RunStatus.STOPPED

    def test_stop_and_join_without_started_thread_proves_exit(self, tmp_path: Path) -> None:
        manager = RunManager()
        managed = manager.create_run(make_config(tmp_path))

        assert manager.stop_and_join(managed.state.run_id, timeout=0.01) is True
        assert managed.state.stop_requested is True

    def test_stop_and_join_reports_live_thread_after_timeout(self, tmp_path: Path) -> None:
        manager = RunManager()
        managed = manager.create_run(make_config(tmp_path))
        thread = MagicMock()
        thread.is_alive.return_value = True  # pyright: ignore[reportAny]
        managed.thread = thread

        assert manager.stop_and_join(managed.state.run_id, timeout=0.25) is False
        thread.join.assert_called_once_with(timeout=0.25)  # pyright: ignore[reportAny]
        assert managed.state.stop_requested is True


class TestRunManagerForceStop:
    def test_force_stop_closes_guidance_and_reports_unstarted_cleanup(self, tmp_path: Path):
        manager = RunManager()
        managed = manager.create_run(make_config(tmp_path))

        assert managed.state.queue_guidance("stop after this turn") is True
        assert manager.force_stop_and_join(managed.state.run_id, timeout=0) is True
        assert managed.state.force_stop_requested is True
        assert managed.state.take_guidance() == ("stop after this turn",)
        assert managed.state.queue_guidance("later") is False

    def test_public_queue_guidance_delegates_to_run_state(self, tmp_path: Path):
        manager = RunManager()
        managed = manager.create_run(make_config(tmp_path))

        assert manager.queue_guidance(managed.state.run_id, "check the output") is True
        assert managed.state.pending_guidance == ("check the output",)

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX process-group behavior")
    def test_force_stop_reaps_live_worker_tree_and_classifies_stopped(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ):
        monkeypatch.setattr("milknado.loop.engine.validate_worker_argv", lambda _cmd: None)  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
        ready = tmp_path / "child-ready"
        stopped = tmp_path / "child-stopped"
        script = tmp_path / "worker.py"
        _ = script.write_text(
            "import signal, subprocess, sys, time\n"  # pyright: ignore[reportImplicitStringConcatenation]
            "from pathlib import Path\n"
            "def stop(*_):\n"
            "    Path(sys.argv[2]).write_text('stopped')\n"
            "    raise SystemExit(0)\n"
            "signal.signal(signal.SIGTERM, stop)\n"
            "child = '''import os, sys, time\\n"
            "from pathlib import Path\\n"
            "Path(sys.argv[1]).write_text(str(os.getpid()))\\n"
            "while True: time.sleep(1)\\n'''\n"
            "subprocess.Popen([sys.executable, '-c', child, sys.argv[1]])\n"
            "while True: time.sleep(1)\n",
            encoding="utf-8",
        )
        manager = RunManager()
        managed = manager.create_run(
            make_config(
                tmp_path,
                agent=f"{sys.executable} {script} {ready} {stopped}",
                max_iterations=1,
            )
        )
        manager.start_run(managed.state.run_id)

        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            _ = managed.state.wait_for_stop(timeout=0.01)
        assert ready.exists(), "descendant never reached its controllable ready point"

        assert manager.force_stop_and_join(managed.state.run_id, timeout=5) is True
        assert managed.thread is not None
        assert not managed.thread.is_alive()
        assert managed.state.status is RunStatus.STOPPED
        assert managed.state.completed == 0

        # The grandchild's SIGTERM handler races the parent's own (unhandled)
        # SIGTERM exit — force_stop_and_join only guarantees the parent thread
        # joined, not that the grandchild finished writing. Poll for completed
        # sentinel content rather than treating file creation as completion.
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if stopped.exists() and stopped.read_text(encoding="utf-8") == "stopped":
                break
            time.sleep(0.01)
        assert stopped.read_text(encoding="utf-8") == "stopped"


class TestRunManagerListAndGet:
    def test_list_runs_returns_all_runs(self, tmp_path: Path):
        manager = RunManager()
        config = make_config(tmp_path)

        managed1 = manager.create_run(config)
        managed2 = manager.create_run(config)
        managed3 = manager.create_run(config)

        runs = manager.list_runs()
        assert len(runs) == 3
        run_ids = {r.state.run_id for r in runs}
        assert managed1.state.run_id in run_ids
        assert managed2.state.run_id in run_ids
        assert managed3.state.run_id in run_ids

    def test_list_runs_empty(self):
        manager = RunManager()
        assert manager.list_runs() == []

    def test_get_run_returns_correct_run(self, tmp_path: Path):
        manager = RunManager()
        config = make_config(tmp_path)
        managed = manager.create_run(config)
        run_id = managed.state.run_id

        result = manager.get_run(run_id)
        assert result is managed

    def test_get_run_returns_none_for_unknown_id(self):
        manager = RunManager()
        assert manager.get_run("nonexistent") is None
