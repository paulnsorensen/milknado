from __future__ import annotations

import textwrap
from pathlib import Path

from milknado.domains.common import SessionContext, SessionEvent, SessionInput
from milknado.loop._agent import AgentRunSpec
from milknado.loop.sessions import SessionChannel, run_session

_CONTEXT = SessionContext(family="claude", cwd="/repo")


def _spec(worker: Path, tmp_path: Path) -> AgentRunSpec:
    return AgentRunSpec(
        cmd=[str(worker)],
        prompt="initial",
        timeout=5.0,
        log_dir=None,
        iteration=1,
        capture_result_text=True,
        cwd=tmp_path,
    )


def _worker(tmp_path: Path) -> Path:
    worker = tmp_path / "claude"
    _ = worker.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import json
            import sys

            for line in sys.stdin:
                if json.loads(line).get("type") == "user":
                    print(
                        json.dumps({"type": "result", "subtype": "success", "result": "done"}),
                        flush=True,
                    )
                    break
            """
        ),
        encoding="utf-8",
    )
    worker.chmod(0o755)
    return worker


def test_runtime_uses_distinct_process_invocations_and_stable_turn_identity(
    tmp_path: Path,
) -> None:
    worker = _worker(tmp_path)
    publications: list[str] = []
    channel = SessionChannel(
        capability_sink=lambda _context, _actions, invocation, _permissions: publications.append(
            invocation
        )
    )

    _ = run_session(_spec(worker, tmp_path), channel)
    first_publications = tuple(publications)
    first = first_publications[-1]
    _ = run_session(_spec(worker, tmp_path), channel)
    second_publications = tuple(publications[len(first_publications) :])
    second = second_publications[-1]

    assert first
    assert second
    assert set(first_publications) == {first}
    assert set(second_publications) == {second}
    assert first != second


def test_close_clears_actions_after_pending_receipt_cleanup() -> None:
    states: list[tuple[str, str]] = []
    publications: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    channel = SessionChannel(
        command_state_sink=lambda command, state: states.append((command.text, state)),
        capability_sink=lambda _context, actions, _invocation, permissions: publications.append(
            (actions, permissions)
        ),
    )
    channel.start(_CONTEXT, ("steer", "approve"), invocation_id="process-1")
    assert channel.submit(SessionInput(action="steer", text="pending"))
    channel.publish(
        SessionEvent(
            kind="permission", text="approval", event_id="permission-1", state="requested"
        )
    )

    channel.close()

    assert channel.view().actions == ()
    assert ("pending", "rejected") in states
    assert all(actions for actions, _permissions in publications[:-1])
    assert publications[-1] == ((), ())
