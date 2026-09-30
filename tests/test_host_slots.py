"""Host-wide worker slot pool: admission, release, config, and the nesting guard."""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Generator
from contextlib import closing
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest
from typer.testing import CliRunner
from typing_extensions import override

from milknado.adapters import FlockSlotPool
from milknado.app.project import open_graph
from milknado.cli import app
from milknado.domains.common import WORKER_CONTEXT_ENV, NodeStatus, load_config
from milknado.domains.common.config import Gate, global_config_path
from milknado.domains.common.errors import CompletionTimeout
from milknado.domains.common.protocols import LoopPort, SlotLease
from milknado.domains.common.types import MikadoNode, WorktreeMode
from milknado.domains.dispatch import make_run_id
from milknado.domains.execution import ExecutionConfig, Executor, PreservedWorkerRun
from milknado.domains.execution import executor as executor_module
from milknado.domains.execution import run_loop as run_loop_module
from milknado.domains.execution._models import CompletionResult, NodeClaimRejected
from milknado.domains.execution.run_loop import RunLoop
from milknado.domains.graph import ConcurrencyLimitReached, HostCapacityFull, MikadoGraph
from milknado.mcp.loop import milknado_run_loop_start
from milknado.mcp.run import milknado_run_inline, milknado_run_inline_start
from milknado.mcp.todo_mutate import milknado_todo_add
from tests.test_execution import FakeCrg, FakeGit, FakeLoop

_HOLDER = (
    "import sys, time\n"
    "from pathlib import Path\n"
    "from milknado.adapters import FlockSlotPool\n"
    "pool = FlockSlotPool(int(sys.argv[1]))\n"
    "lease = pool.acquire('held-run', 7, Path('/held'))\n"
    "print('ready', flush=True)\n"
    "time.sleep(600)\n"
)


class _Holder:
    def __init__(self, limit: int) -> None:
        self.proc: subprocess.Popen[str] = subprocess.Popen(
            [sys.executable, "-c", _HOLDER, str(limit)],
            stdout=subprocess.PIPE,
            text=True,
            env=os.environ.copy(),
        )
        assert self.proc.stdout is not None
        assert cast(str, self.proc.stdout.readline()).strip() == "ready"

    def kill(self) -> None:
        self.proc.send_signal(signal.SIGKILL)
        _ = self.proc.wait(timeout=10)


@pytest.fixture()
def holders() -> Generator[list[_Holder], None, None]:
    started: list[_Holder] = []
    yield started
    for holder in started:
        if holder.proc.poll() is None:
            holder.kill()
        if holder.proc.stdout is not None:
            holder.proc.stdout.close()


def _hold(holders: list[_Holder], limit: int) -> _Holder:
    holder = _Holder(limit)
    holders.append(holder)
    return holder


def _free(limit: int = 1) -> bool:
    try:
        FlockSlotPool(limit).acquire("probe", 0, Path("/probe")).release()
    except HostCapacityFull:
        return False
    return True


def _wait_free(limit: int = 1) -> bool:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if _free(limit):
            return True
        time.sleep(0.05)
    return False


class TestSlotPool:
    def test_admits_up_to_limit_then_reports_full(self) -> None:
        pool = FlockSlotPool(2)
        first = pool.acquire("r1", 1, Path("/a"))
        second = pool.acquire("r2", 2, Path("/b"))
        with pytest.raises(HostCapacityFull) as full:
            _ = pool.acquire("r3", 3, Path("/c"))
        assert isinstance(full.value, ConcurrencyLimitReached)
        assert (full.value.running, full.value.limit) == (2, 2)
        first.release()
        third = pool.acquire("r3", 3, Path("/c"))
        second.release()
        third.release()

    def test_release_is_idempotent_and_never_frees_another_slot(self) -> None:
        pool = FlockSlotPool(1)
        lease = pool.acquire("r1", 1, Path("/a"))
        lease.release()
        successor = pool.acquire("r2", 2, Path("/b"))
        lease.release()
        assert not _free()
        successor.release()

    def test_slot_body_names_the_holder(self) -> None:
        pool = FlockSlotPool(1)
        lease = pool.acquire("run-9", 9, Path("/proj"))
        (body,) = pool.holders()
        assert body["pid"] == os.getpid()
        assert (body["run_id"], body["node_id"], body["project_root"]) == ("run-9", 9, "/proj")
        assert "acquired_at" in body
        lease.release()
        assert pool.holders() == []

    def test_pool_lives_under_xdg_state_home(self, monkeypatch: pytest.MonkeyPatch) -> None:
        state = os.environ["XDG_STATE_HOME"]
        assert FlockSlotPool(1).directory == Path(state) / "milknado" / "worker-slots"
        monkeypatch.delenv("XDG_STATE_HOME")
        assert FlockSlotPool(1).directory == (
            Path.home() / ".local" / "state" / "milknado" / "worker-slots"
        )

    def test_sigkilled_holder_frees_its_slot(self, holders: list[_Holder]) -> None:
        holder = _hold(holders, 1)
        assert not _free()
        holder.kill()
        assert _wait_free()


def _executor(root: Path, limit: int) -> tuple[Executor, MikadoGraph, FakeLoop, ExecutionConfig]:
    root.mkdir(parents=True, exist_ok=True)
    graph = MikadoGraph(root / "graph.db")
    loop = FakeLoop(id_prefix=f"run-{root.name}")
    executor = Executor(graph=graph, git=FakeGit(), loop=loop, crg=FakeCrg())
    executor.use_host_capacity(FlockSlotPool(limit))
    config = ExecutionConfig(
        execution_agent="claude",
        quality_gates=(Gate(command="true"),),
        worktree_pattern="wt-{node_id}-{slug}",
        project_root=root,
        dispatch_max_retries=0,
    )
    return executor, graph, loop, config


class TestDispatchAdmission:
    def test_full_pool_defers_dispatch_from_any_project_without_spawning(
        self, tmp_path: Path, holders: list[_Holder]
    ) -> None:
        _ = _hold(holders, 1)
        for name in ("proj-a", "proj-b"):
            executor, graph, loop, config = _executor(tmp_path / name, 1)
            with closing(graph):
                node = graph.add_node("task")
                with pytest.raises(ConcurrencyLimitReached):
                    _ = executor.dispatch(node.id, config)
                assert loop.runs_created == []
                current = graph.get_node(node.id)
                assert current is not None
                assert current.status is NodeStatus.PENDING

    def test_dead_holder_admits_the_deferred_dispatch(
        self, tmp_path: Path, holders: list[_Holder]
    ) -> None:
        holder = _hold(holders, 1)
        executor, graph, loop, config = _executor(tmp_path / "proj", 1)
        with closing(graph):
            node = graph.add_node("task")
            with pytest.raises(HostCapacityFull):
                _ = executor.dispatch(node.id, config)
            holder.kill()
            assert _wait_free()
            _ = executor.dispatch(node.id, config)
            assert len(loop.runs_created) == 1
            assert not _free()

    def test_second_project_waits_while_first_holds_the_only_slot(self, tmp_path: Path) -> None:
        first, graph_a, _, config_a = _executor(tmp_path / "a", 1)
        second, graph_b, loop_b, config_b = _executor(tmp_path / "b", 1)
        with closing(graph_a), closing(graph_b):
            node_a = graph_a.add_node("task a")
            node_b = graph_b.add_node("task b")
            _ = first.dispatch(node_a.id, config_a)
            with pytest.raises(HostCapacityFull):
                _ = second.dispatch(node_b.id, config_b)
            assert loop_b.runs_created == []

    def test_claim_failure_releases_the_slot(self, tmp_path: Path) -> None:
        executor, graph, _, config = _executor(tmp_path / "proj", 1)
        with closing(graph):
            node = graph.add_node("task")
            assert graph.claim_node(node.id, "someone-else", now="2026-01-01T00:00:00+00:00")
            with pytest.raises(NodeClaimRejected):
                _ = executor.dispatch(node.id, config)
            assert _free()

    def test_graph_capacity_refusal_releases_the_slot(self, tmp_path: Path) -> None:
        executor, graph, _, config = _executor(tmp_path / "proj", 1)
        with closing(graph):
            busy = graph.add_node("busy")
            waiting = graph.add_node("waiting")
            for index in range(3):
                extra = graph.add_node(f"extra-{index}")
                assert graph.claim_node(extra.id, f"x{index}", now="2026-01-01T00:00:00+00:00")
            assert graph.claim_node(busy.id, "busy", now="2026-01-01T00:00:00+00:00")
            with pytest.raises(ConcurrencyLimitReached):
                _ = executor.dispatch(waiting.id, config)
            assert _free()


class TestSlotReleasedOnTerminalPaths:
    @pytest.mark.parametrize("path", ["done", "failed", "cancelled", "force-stopped"])
    def test_slot_is_freed_once_the_run_is_terminal(self, tmp_path: Path, path: str) -> None:
        executor, graph, _, config = _executor(tmp_path / "proj", 1)
        with closing(graph):
            node = graph.add_node("task")
            dispatched = executor.dispatch(node.id, config)
            assert not _free()
            if path == "done":
                _ = executor.complete(node.id, "main")
            elif path == "failed":
                executor.fail(node.id)
            elif path == "cancelled":
                assert executor.force_stop_run(dispatched.run_id)
                executor.cancel(node.id)
            else:
                assert executor.force_stop_run(dispatched.run_id)
                executor.finish_preserved_abort(node.id, dispatched.run_id, dispatched.run_id)
            assert _free()


class TestSlotKeptWhileAWorkerMayLive:
    def test_unconfirmed_force_stop_keeps_the_slot(self, tmp_path: Path) -> None:
        executor, graph, loop, config = _executor(tmp_path / "proj", 1)
        with closing(graph):
            node = graph.add_node("task")
            dispatched = executor.dispatch(node.id, config)
            loop.force_stop_result = False
            assert not executor.force_stop_run(dispatched.run_id)
            assert not _free()

    def test_preserved_worker_run_keeps_the_slot(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        executor, graph, loop, config = _executor(tmp_path / "proj", 1)
        loop.force_stop_result = False

        def _boom(_root: Path) -> Path:
            raise OSError("watcher setup failed after the worker started")

        monkeypatch.setattr(executor_module, "runs_dir", _boom)
        with closing(graph):
            node = graph.add_node("task")
            with pytest.raises(PreservedWorkerRun):
                _ = executor.dispatch(node.id, config)
            assert len(loop.runs_started) == 1
            assert not _free()

    def test_failed_slot_settle_does_not_mask_the_completion_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        executor, graph, _, _ = _executor(tmp_path / "proj", 1)

        def _fail_completion(_node_id: int, _branch: str) -> CompletionResult:
            raise RuntimeError("completion failed")

        def _fail_lookup(_node_id: int) -> MikadoNode | None:
            raise OSError("graph unreadable")

        monkeypatch.setattr(executor, "_complete", _fail_completion)
        monkeypatch.setattr(graph, "get_node", _fail_lookup)
        with closing(graph), pytest.raises(RuntimeError, match="completion failed"):
            _ = executor.complete(1, "main")


class _Admitted(BaseException):
    """Raised by the stub loop once the deferred node has been admitted."""


class _StallingLoop(FakeLoop):
    """Never completes a run; frees another project's slot on the first wait."""

    def __init__(self, other_lease: SlotLease) -> None:
        super().__init__(id_prefix="stall")
        self._other: SlotLease = other_lease
        self.timeouts: list[float | None] = []

    @override
    def wait_for_next_completion(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, active_run_ids: set[str], timeout: float | None = None
    ) -> tuple[str, object]:
        self.timeouts.append(timeout)
        if len(active_run_ids) == 2:
            raise _Admitted
        if timeout is None:
            raise AssertionError("an unbounded wait strands the deferred node")
        self._other.release()
        time.sleep(timeout)
        raise CompletionTimeout(active_run_ids, timeout)


class TestDeferredNodeRetry:
    def test_slot_freed_by_another_project_admits_the_deferred_node_while_a_run_is_active(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(run_loop_module, "IDLE_RESCAN_SECONDS", 0.05)
        other = FlockSlotPool(2).acquire("other-project", 1, Path("/other"))
        executor, graph, _, config = _executor(tmp_path / "proj", 2)
        loop = _StallingLoop(other)
        with closing(graph):
            root = graph.add_node("goal")
            _ = graph.add_node("first", parent_id=root.id)
            _ = graph.add_node("second", parent_id=root.id)
            driver = RunLoop(
                executor=executor, graph=graph, loop=cast(LoopPort, cast(object, loop))
            )
            with pytest.raises(_Admitted):
                _ = driver.run(config, "main", interactive=False)
        assert loop.timeouts[0] is not None

    def test_controller_loop_also_retries_the_deferred_node(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(run_loop_module, "IDLE_RESCAN_SECONDS", 0.05)
        other = FlockSlotPool(2).acquire("other-project", 1, Path("/other"))
        executor, graph, _, config = _executor(tmp_path / "proj", 2)
        loop = _StallingLoop(other)
        control_calls = 0

        def process_controls() -> None:
            nonlocal control_calls
            control_calls += 1
            if control_calls > 30:
                raise AssertionError("controls polled forever; the deferred node is stranded")

        with closing(graph):
            root = graph.add_node("goal")
            _ = graph.add_node("first", parent_id=root.id)
            _ = graph.add_node("second", parent_id=root.id)
            driver = RunLoop(
                executor=executor, graph=graph, loop=cast(LoopPort, cast(object, loop))
            )
            with pytest.raises(_Admitted):
                _ = driver.run(
                    config, "main", interactive=False, process_controls=process_controls
                )

    def test_a_full_pool_is_retried_at_most_once_per_idle_rescan_interval(
        self, tmp_path: Path
    ) -> None:
        executor, graph, _, config = _executor(tmp_path / "proj", 2)
        calls: list[float] = []
        clock = [10.0]

        def dispatch(*_args: object) -> tuple[int, int]:
            calls.append(clock[0])
            return 0, 0

        with closing(graph):
            driver = RunLoop(executor=executor, graph=graph, loop=FakeLoop(id_prefix="cadence"))
            setattr(driver, "_capacity_deferred", True)  # noqa: B010
            setattr(driver, "_dispatch_if_scheduling_open", dispatch)  # noqa: B010
            retry = cast(
                Callable[[ExecutionConfig, int], tuple[int, int]],
                getattr(driver, "_retry_deferred_if_due"),  # noqa: B009
            )
            interval = cast(float, getattr(run_loop_module, "IDLE_RESCAN_SECONDS"))  # noqa: B009
            monotonic = "milknado.domains.execution.run_loop.time.monotonic"
            with patch(monotonic, side_effect=lambda: clock[0]):
                _ = retry(config, 2)
                clock[0] += interval / 2
                _ = retry(config, 2)
                clock[0] += interval / 2
                _ = retry(config, 2)

        assert calls == [10.0, 10.0 + interval]


_NOW = "2026-01-01T00:00:00+00:00"


def _write_global(text: str) -> None:
    path = global_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(text, encoding="utf-8")


def _project_config(root: Path, body: str) -> Path:
    path = root / "milknado.toml"
    _ = path.write_text(f"[milknado]\n{body}", encoding="utf-8")
    return path


class TestHostWorkerLimitConfig:
    def test_default_is_six(self, tmp_path: Path) -> None:
        assert load_config(_project_config(tmp_path, "")).host_worker_limit == 6

    def test_global_value_applies(self, tmp_path: Path) -> None:
        _write_global("[milknado]\nhost_worker_limit = 3\n")
        assert load_config(_project_config(tmp_path, "")).host_worker_limit == 3

    def test_project_value_warns_and_is_ignored(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        _write_global("[milknado]\nhost_worker_limit = 5\n")
        with caplog.at_level(logging.WARNING):
            config = load_config(_project_config(tmp_path, "host_worker_limit = 9\n"))
        assert config.host_worker_limit == 5
        assert "host_worker_limit" in caplog.text

    def test_project_value_without_global_falls_back_to_default(self, tmp_path: Path) -> None:
        config = load_config(_project_config(tmp_path, "host_worker_limit = 9\n"))
        assert config.host_worker_limit == 6

    def test_zero_is_rejected_at_load(self, tmp_path: Path) -> None:
        _write_global("[milknado]\nhost_worker_limit = 0\n")
        with pytest.raises(ValueError, match="host_worker_limit"):
            _ = load_config(_project_config(tmp_path, ""))


class TestNestingGuard:
    def test_run_refuses_inside_a_worker_before_opening_any_graph(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(WORKER_CONTEXT_ENV, "1")
        result = CliRunner().invoke(app, ["run", "--project-root", str(tmp_path)])
        assert result.exit_code == 2
        assert "cannot start inside a milknado worker" in result.output
        assert not (tmp_path / ".milknado").exists()


def _loop_project(root: Path, *tasks: str) -> tuple[str, list[int]]:
    root.mkdir(parents=True, exist_ok=True)
    _ = _project_config(root, "")
    tool = cast(
        Callable[..., dict[str, object]], getattr(milknado_todo_add, "fn", milknado_todo_add)
    )
    ids = [
        cast(int, tool(description=t, kind="task", project_root=str(root))["id"]) for t in tasks
    ]
    return str(root), ids


def _git_init(root: Path) -> None:
    for args in (
        ["init", "-q", "-b", "main"],
        ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "i"],
    ):
        _ = subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def _start(root: str, node_id: int) -> dict[str, object]:
    tool = cast(
        Callable[..., dict[str, object]],
        getattr(milknado_run_loop_start, "fn", milknado_run_loop_start),
    )
    return tool(node_id=node_id, runner_cmd=f"{sys.executable} -c pass", project_root=root)


class TestLoopStart:
    def test_full_host_pool_defers_start_and_admits_after_holder_dies(
        self, tmp_path: Path, holders: list[_Holder]
    ) -> None:
        _write_global("[milknado]\nhost_worker_limit = 1\n")
        holder = _hold(holders, 1)
        root, (node_id,) = _loop_project(tmp_path / "proj", "task")
        _git_init(Path(root))
        deferred = _start(root, node_id)
        assert deferred["status"] == "deferred"
        assert (deferred["running"], deferred["limit"]) == (1, 1)
        assert deferred["run_id"] is None
        holder.kill()
        assert _wait_free()
        started = _start(root, node_id)
        assert started["status"] == "running"
        assert _wait_free()


class TestInlineStart:
    def _start(self, root: Path, node_id: int, worker_cmd: str) -> object:
        tool = cast(
            Callable[..., object],
            getattr(milknado_run_inline_start, "fn", milknado_run_inline_start),
        )
        return tool(
            node_id=node_id,
            worker_cmd=worker_cmd,
            worktree=WorktreeMode.THIS_BRANCH,
            project_root=str(root),
        )

    def test_full_pool_refuses_inline_start(
        self, tmp_path: Path, holders: list[_Holder], worker_stub: Callable[[str], str]
    ) -> None:
        _write_global("[milknado]\nhost_worker_limit = 1\n")
        _ = _hold(holders, 1)
        _, (node_id,) = _loop_project(tmp_path, "task")
        with pytest.raises(HostCapacityFull):
            _ = self._start(tmp_path, node_id, worker_stub("cat"))

    def test_slot_is_held_for_the_run_then_released(
        self, tmp_path: Path, worker_stub: Callable[[str], str]
    ) -> None:
        _write_global("[milknado]\nhost_worker_limit = 1\n")
        _, (node_id,) = _loop_project(tmp_path, "task")
        _ = self._start(tmp_path, node_id, worker_stub("sleep 2"))
        assert not _free()
        assert _wait_free()


def _run_inline_sync(root: Path, node_id: int, worker_cmd: str) -> dict[str, object]:
    tool = cast(
        Callable[..., dict[str, object]], getattr(milknado_run_inline, "fn", milknado_run_inline)
    )
    return tool(
        node_id=node_id,
        worker_cmd=worker_cmd,
        worktree=WorktreeMode.THIS_BRANCH,
        project_root=str(root),
    )


def _node_status(root: Path, node_id: int) -> NodeStatus:
    graph, _cfg = open_graph(root)
    with closing(graph):
        node = graph.get_node(node_id)
        assert node is not None
        return node.status


class TestInlineSync:
    def test_full_pool_refuses_sync_inline_without_spawning(
        self, tmp_path: Path, holders: list[_Holder], worker_stub: Callable[[str], str]
    ) -> None:
        _write_global("[milknado]\nhost_worker_limit = 1\n")
        _ = _hold(holders, 1)
        _, (node_id,) = _loop_project(tmp_path, "task")
        marker = tmp_path / "spawned"
        with pytest.raises(HostCapacityFull):
            _ = _run_inline_sync(tmp_path, node_id, worker_stub(f"touch {marker}"))
        assert not marker.exists()
        assert _node_status(tmp_path, node_id) is NodeStatus.PENDING

    def test_slot_is_held_while_the_blocking_worker_runs(
        self, tmp_path: Path, worker_stub: Callable[[str], str]
    ) -> None:
        _write_global("[milknado]\nhost_worker_limit = 1\n")
        _, (node_id,) = _loop_project(tmp_path, "task")
        outcome: list[dict[str, object]] = []
        worker = threading.Thread(
            target=lambda: outcome.append(
                _run_inline_sync(tmp_path, node_id, worker_stub("sleep 2"))
            )
        )
        worker.start()
        deadline = time.monotonic() + 10
        while _free() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not _free()
        worker.join(timeout=30)
        assert outcome[0]["status"] == "done"
        assert _free()


class TestDetachedRunner:
    def test_full_pool_defers_the_node_instead_of_failing_it(
        self, tmp_path: Path, holders: list[_Holder]
    ) -> None:
        _write_global("[milknado]\nhost_worker_limit = 1\n")
        root, (node_id,) = _loop_project(tmp_path / "proj", "task")
        _git_init(Path(root))
        _ = _project_config(
            Path(root), "quality_gates = []\n[milknado.flavor.implement]\nreview = false\n"
        )
        base_oid = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
        run_id = make_run_id(node_id)
        graph, _cfg = open_graph(Path(root))
        with closing(graph):
            graph.claim_node_for_dispatch(node_id, run_id, now=_NOW)
            graph.runs.start(run_id, node_id, str(tmp_path / "run.log"), _NOW, 60)
        holder = _hold(holders, 1)
        done = subprocess.run(
            [
                sys.executable,
                "-m",
                "milknado.mcp._loop_node_runner",
                *("--node-id", str(node_id), "--project-root", root, "--run-id", run_id),
                *("--target-branch", "main", "--base-oid", base_oid),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            env=os.environ.copy(),
        )
        assert holder.proc.poll() is None
        graph, _cfg = open_graph(Path(root))
        with closing(graph):
            node = graph.get_node(node_id)
            run = graph.runs.get(run_id)
        assert node is not None
        assert node.status is NodeStatus.PENDING, done.stderr
        assert run is not None
        assert run["status"] == "failed"
        assert "deferred" in (run["detail"] or "")


def test_slot_body_is_json(tmp_path: Path) -> None:
    lease: SlotLease = FlockSlotPool(1).acquire("r", 1, tmp_path)
    raw = (FlockSlotPool(1).directory / "slot-0").read_text(encoding="utf-8")
    assert json.loads(raw)["run_id"] == "r"
    lease.release()
