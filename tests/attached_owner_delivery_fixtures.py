from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_ATTACHED_SOURCE = """\\
from pathlib import Path
import sys

from milknado.app.watch import AttachedWatchSource, WatchSnapshotSource, graph_command_admitter
from milknado.domains.common import SessionInput
from milknado.domains.graph import MikadoGraph

repo = Path(sys.argv[1])
db_path = Path(sys.argv[2])
run_id, request_id, text = sys.argv[3:]
graph = MikadoGraph(db_path)
try:
    source = AttachedWatchSource(
        WatchSnapshotSource(repo, db_path),
        graph_command_admitter(graph),
    )
    accepted = source.session_input(
        run_id, SessionInput(action="follow_up", text=text, request_id=request_id)
    )
    print("accepted" if accepted else "rejected", flush=True)
    raise SystemExit(0 if accepted else 1)
finally:
    graph.close()
"""


@dataclass(frozen=True)
class AttachedCommand:
    repo: Path
    db_path: Path
    run_id: str
    request_id: str
    text: str


def admit_from_process(request: AttachedCommand) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-c",
            _ATTACHED_SOURCE,
            str(request.repo),
            str(request.db_path),
            request.run_id,
            request.request_id,
            request.text,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
