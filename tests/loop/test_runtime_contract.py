from __future__ import annotations

import sys
import textwrap
from dataclasses import replace
from pathlib import Path
from typing import Literal, cast

import msgspec
import pytest

from milknado.domains.common import SessionContext, SessionEvent, SessionInput
from milknado.loop._agent import AgentRunSpec
from milknado.loop.sessions import (
    ProtocolStep,
    ProviderSessionIdentity,
    RecoveryReceipt,
    RuntimeRecoveryRequest,
    RuntimeRequest,
    RuntimeSession,
    SessionChannel,
    SessionProtocol,
    runtime_capabilities,
    start_or_resume,
    submit_runtime_action,
)
from milknado.loop.sessions._factory import create_protocol

FLOOR = {
    "start",
    "resume",
    "streamed_output",
    "user_input",
    "interrupt",
    "approval",
    "cancel",
    "terminal_result",
    "recovery_report",
}


def _active_protocol(family: Literal["claude", "codex"], cwd: Path) -> SessionProtocol:
    protocol = create_protocol((family,), cwd)
    assert protocol is not None
    started = protocol.start("work")
    if family == "claude":
        return protocol
    initialize = msgspec.json.decode(started.commands[0], type=dict[str, object])
    thread_step = protocol.receive(msgspec.json.encode({"id": initialize["id"], "result": {}}))
    thread = msgspec.json.decode(thread_step.commands[1], type=dict[str, object])
    turn_step = protocol.receive(
        msgspec.json.encode(
            {"id": thread["id"], "result": {"thread": {"id": "thread-1", "sessionId": "s-1"}}}
        )
    )
    turn = msgspec.json.decode(turn_step.commands[0], type=dict[str, object])
    _ = protocol.receive(
        msgspec.json.encode(
            {"id": turn["id"], "result": {"turn": {"id": "turn-1", "status": "inProgress"}}}
        )
    )
    return protocol


@pytest.mark.parametrize("family", ["claude", "codex"])
def test_provider_contract_declares_each_lifecycle_capability(
    family: Literal["claude", "codex"], tmp_path: Path
) -> None:
    protocol = _active_protocol(family, tmp_path)
    capabilities = runtime_capabilities(family)
    assert set(capabilities.floor) == FLOOR
    assert all(
        state in {"native", "runtime", "unsupported"} for state in capabilities.floor.values()
    )
    assert capabilities.floor["start"] == "native"
    assert capabilities.floor["resume"] == "native"
    assert capabilities.floor["recovery_report"] == "runtime"
    assert frozenset(protocol.actions) == capabilities.native_actions


def test_provider_specific_actions_are_explicit() -> None:
    claude = runtime_capabilities("claude")
    codex = runtime_capabilities("codex")
    assert claude.native_actions == frozenset({"follow_up", "interrupt", "approve", "deny"})
    assert codex.native_actions == frozenset({"steer", "interrupt", "approve", "deny"})
    assert "steer" in claude.unsupported_actions
    assert "follow_up" in codex.unsupported_actions


@pytest.mark.parametrize("family", ["claude", "codex"])
def test_recovery_identity_only_comes_from_provider_step(
    family: Literal["claude", "codex"],
) -> None:
    transcript = SessionEvent(kind="assistant", text="session_id: forged")
    assert ProviderSessionIdentity.from_step(family, ProtocolStep(events=(transcript,))) is None
    identity = ProviderSessionIdentity.from_step(
        family, ProtocolStep(events=(transcript,), session_id="provider-42")
    )
    assert identity is not None
    assert identity.session_id == "provider-42"
    request = RuntimeRecoveryRequest(identity=identity, cwd=Path("/repo"))
    assert request.identity == identity
    forged = cast(ProviderSessionIdentity, cast(object, "session_id: forged"))
    with pytest.raises(TypeError):
        _ = RuntimeRecoveryRequest(identity=forged, cwd=Path("/repo"))


def test_recovery_receipt_cannot_claim_unknown_turn_completed() -> None:
    identity = ProviderSessionIdentity.from_step("codex", ProtocolStep(session_id="thread-1"))
    assert identity is not None
    receipt = RecoveryReceipt(identity=identity, outcome="unknown_turn")
    assert not receipt.turn_confirmed
    with pytest.raises(ValueError):
        _ = replace(receipt, turn_confirmed=True)


@pytest.mark.parametrize(
    ("family", "supported", "unsupported"),
    [("claude", "follow_up", "steer"), ("codex", "steer", "follow_up")],
)
def test_action_port_uses_active_actions_and_rejects_unsupported(
    family: Literal["claude", "codex"],
    supported: Literal["follow_up", "steer"],
    unsupported: Literal["follow_up", "steer"],
    tmp_path: Path,
) -> None:
    protocol = _active_protocol(family, tmp_path)
    channel = SessionChannel()
    channel.start(SessionContext(family=family, cwd=str(tmp_path)), protocol.actions)
    session = RuntimeSession.from_step(family, ProtocolStep(session_id="provider-1"), channel)
    assert session is not None

    rejected = submit_runtime_action("provider-1", SessionInput(action=unsupported), session)
    assert rejected.state == "unsupported"
    assert channel.drain() == ()
    accepted = submit_runtime_action(
        "provider-1", SessionInput(action=supported, text="continue"), session
    )
    assert accepted.state == "queued"
    assert [command.action for command in channel.drain()] == [supported]
    interrupted = submit_runtime_action("provider-1", SessionInput(action="interrupt"), session)
    assert interrupted.state == "queued"
    assert [command.action for command in channel.drain()] == ["interrupt"]
    assert (
        submit_runtime_action("wrong-id", SessionInput(action="interrupt"), session).state
        == "unknown_session"
    )


def test_stale_handle_cannot_submit_after_channel_restart(tmp_path: Path) -> None:
    channel = SessionChannel()
    context = SessionContext(family="claude", cwd=str(tmp_path))
    channel.start(context, ("interrupt",), invocation_id="process-a")
    first = RuntimeSession.from_step("claude", ProtocolStep(session_id="same-id"), channel)
    assert first is not None

    channel.close()
    channel.start(context, ("interrupt",), invocation_id="process-b")
    second = RuntimeSession.from_step("claude", ProtocolStep(session_id="same-id"), channel)
    assert second is not None

    stale = submit_runtime_action("same-id", SessionInput(action="interrupt"), first)
    assert stale.state == "unknown_session"
    assert channel.drain() == ()
    current = submit_runtime_action("same-id", SessionInput(action="interrupt"), second)
    assert current.state == "queued"
    assert [command.action for command in channel.drain()] == ["interrupt"]


def test_resume_rejects_provider_and_worktree_mismatch(tmp_path: Path) -> None:
    identity = ProviderSessionIdentity("codex", "provider-1")
    spec = AgentRunSpec(
        cmd=["claude"], prompt="continue", timeout=1, log_dir=None, iteration=1, cwd=tmp_path
    )
    request = RuntimeRequest(
        spec=spec,
        channel=SessionChannel(),
        resume=RuntimeRecoveryRequest(identity=identity, cwd=tmp_path),
    )
    with pytest.raises(ValueError, match="provider family"):
        _ = start_or_resume(request)
    with pytest.raises(ValueError, match="worktree"):
        _ = start_or_resume(
            RuntimeRequest(
                spec=replace(spec, cmd=["codex"]),
                channel=SessionChannel(),
                resume=RuntimeRecoveryRequest(identity, tmp_path / "other"),
            )
        )


def test_resume_rejects_codex_effective_cwd_before_launch(tmp_path: Path) -> None:
    other = tmp_path / "other"
    other.mkdir()
    identity = ProviderSessionIdentity("codex", "thread-1")
    spec = AgentRunSpec(
        cmd=["codex", "--cd", str(other)],
        prompt="continue",
        timeout=1,
        log_dir=None,
        iteration=1,
        cwd=tmp_path,
    )
    with pytest.raises(ValueError, match="worktree"):
        _ = start_or_resume(
            RuntimeRequest(spec, SessionChannel(), RuntimeRecoveryRequest(identity, tmp_path))
        )


def test_claude_resume_uses_provider_identity_and_reports_confirmation(tmp_path: Path) -> None:
    script = tmp_path / "worker.py"
    _ = script.write_text(
        textwrap.dedent(
            """\
            import json
            import sys

            assert sys.argv[1:3] == ["--resume", "provider-1"]
            for raw in sys.stdin:
                if json.loads(raw).get("type") == "user":
                    print(json.dumps({"type": "result", "subtype": "success",
                                      "session_id": "provider-1", "result": "done"}), flush=True)
                    break
            """
        ),
        encoding="utf-8",
    )
    worker = tmp_path / "claude"
    worker.symlink_to(sys.executable)
    identity = ProviderSessionIdentity("claude", "provider-1")
    spec = AgentRunSpec(
        cmd=[str(worker), str(script)],
        prompt="continue",
        timeout=5,
        log_dir=None,
        iteration=1,
        cwd=tmp_path,
    )
    result = start_or_resume(
        RuntimeRequest(spec, SessionChannel(), RuntimeRecoveryRequest(identity, tmp_path))
    )
    assert result.run is not None and result.run.returncode == 0
    assert result.recovery == RecoveryReceipt(identity, "resumed", turn_confirmed=True)


@pytest.mark.parametrize("prefix", ((), ("app-server",)))
def test_codex_resume_uses_thread_identity_and_confirms_turn(
    tmp_path: Path, prefix: tuple[str, ...]
) -> None:
    worker = tmp_path / "codex"
    _ = worker.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import json
            import sys

            assert sys.argv[1:] == ["app-server"]
            for raw in sys.stdin:
                request = json.loads(raw)
                method = request.get("method")
                if method == "initialize":
                    result = {}
                elif method == "thread/resume":
                    assert request["params"]["threadId"] == "thread-1"
                    result = {"thread": {"id": "thread-1"}}
                elif method == "turn/start":
                    assert request["params"]["threadId"] == "thread-1"
                    result = {"turn": {"id": "turn-1", "status": "completed"}}
                else:
                    continue
                print(json.dumps({"id": request["id"], "result": result}), flush=True)
                if method == "turn/start":
                    break
            """
        ),
        encoding="utf-8",
    )
    worker.chmod(0o755)
    identity = ProviderSessionIdentity("codex", "thread-1")
    spec = AgentRunSpec(
        cmd=[str(worker), *prefix],
        prompt="continue",
        timeout=5,
        log_dir=None,
        iteration=1,
        cwd=tmp_path,
    )
    result = start_or_resume(
        RuntimeRequest(spec, SessionChannel(), RuntimeRecoveryRequest(identity, tmp_path))
    )
    assert result.run is not None and result.run.returncode == 0
    assert result.recovery == RecoveryReceipt(identity, "resumed", turn_confirmed=True)


def test_start_port_runs_provider_session(tmp_path: Path) -> None:
    script = tmp_path / "worker.py"
    _ = script.write_text(
        textwrap.dedent(
            """\
            import json
            import sys

            for raw in sys.stdin:
                if json.loads(raw).get("type") == "user":
                    print(json.dumps({
                        "type": "result", "subtype": "success",
                        "session_id": "provider-1", "result": "done",
                    }), flush=True)
                    break
            """
        ),
        encoding="utf-8",
    )
    worker = tmp_path / "claude"
    worker.symlink_to(sys.executable)
    spec = AgentRunSpec(
        cmd=[str(worker), str(script)],
        prompt="work",
        timeout=5,
        log_dir=None,
        iteration=1,
        cwd=tmp_path,
    )
    result = start_or_resume(RuntimeRequest(spec=spec, channel=SessionChannel()))
    assert result.recovery is None
    assert result.run is not None
    assert result.run.session_id == "provider-1"
    assert result.run.returncode == 0
