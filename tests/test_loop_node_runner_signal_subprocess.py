from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals required")

_RUNNER = """
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import milknado.adapters as adapters
import milknado.app._shutdown as shutdown
import milknado.app.project as project
import milknado.app.worker_recovery as recovery
import milknado.domains.common as common
import milknado.domains.execution as execution
from milknado.domains.execution import NodeLoopOutcome
from milknado.mcp import _loop_node_runner as runner

mode, marker, stopped, recorded, root = sys.argv[1:]
original_record = shutdown.ShutdownIntent.record

def record(self, signum, frame):
    original_record(self, signum, frame)
    Path(recorded).touch()

shutdown.ShutdownIntent.record = record

class Graph:
    def __init__(self):
        self.runs = self
    def set_pid(self, *args):
        pass
    def get_node(self, node_id):
        return None
    def finish(self, run_id, result):
        raise AssertionError('signal path wrote terminal result')
    def close(self):
        pass

class Loop:
    def __init__(self, **kwargs):
        pass
    def bind_shutdown_intent(self, callback):
        pass

class Executor:
    def __init__(self, **kwargs):
        pass
    def use_host_capacity(self, pool):
        pass

class RunLoop:
    def __init__(self, **kwargs):
        pass
    def run_node(self, *args, **kwargs):
        if mode == 'error':
            raise RuntimeError('worker failed after dispatch')
        return NodeLoopOutcome(1, False, ownership_preserved=True)
    def confirm_preserved_stop(self, outcome):
        assert outcome.ownership_preserved
        Path(marker).touch()
        while True:
            time.sleep(0.02)
    def force_stop_active(self, deadline):
        Path(stopped).touch()
        return True

profile = SimpleNamespace(
    execution_agent='test', quality_gates=(), worktree_pattern='wt-{node}',
    brief_prepend='', commit_footer=None, review=False, review_agent=None,
    review_max_rounds=0, review_timeout_seconds=1, on_reject=None,
    session_mode='redispatch', attempt_timeout_seconds=1, max_iterations=1,
)
cfg = SimpleNamespace(host_worker_limit=1, worktree_pattern='wt-{node}', commit_footer=None)
args = SimpleNamespace(
    project_root=root, run_id='run-1', node_id=1, target_branch='main', base_oid='base'
)
runner._parse_args = lambda argv: args
project.open_graph = lambda root: (Graph(), cfg)
recovery.reconcile_loop_workers = lambda graph: None
common.resolve_flavor_profile = lambda cfg, flavor: profile
adapters.GitAdapter = lambda root: object()
adapters.CrgAdapter = lambda root: object()
adapters.FlockSlotPool = lambda limit: object()
adapters.LoopAdapter = Loop
execution.ExecutionConfig = lambda **kwargs: object()
execution.Executor = Executor
execution.RunLoop = RunLoop
raise SystemExit(runner.main([]))
"""


@pytest.mark.parametrize("mode", ["normal", "error"])
def test_os_signal_during_preserved_stop_retry_exits_with_signal_status(
    tmp_path: Path, mode: str
) -> None:
    marker = tmp_path / "retrying"
    stopped = tmp_path / "stopped"
    recorded = tmp_path / "recorded"
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            _RUNNER,
            mode,
            str(marker),
            str(stopped),
            str(recorded),
            str(tmp_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 5
        while not marker.exists() and proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert marker.exists(), "runner exited before preserved-stop retry"
        os.kill(proc.pid, signal.SIGTERM)
        while not recorded.exists() and proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert recorded.exists()
        if proc.poll() is None:
            os.kill(proc.pid, signal.SIGINT)
        stdout, stderr = proc.communicate(timeout=4)
        assert proc.returncode == 128 + signal.SIGTERM, (stdout, stderr)
        assert stopped.exists()
    finally:
        if proc.poll() is None:
            proc.kill()
        _ = proc.communicate(timeout=2)
