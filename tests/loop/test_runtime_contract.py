from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Literal, cast

import pytest

from milknado.domains.common import SessionEvent
from milknado.loop.sessions import (
    ProtocolStep,
    ProviderSessionIdentity,
    RecoveryReceipt,
    RuntimeRecoveryRequest,
    runtime_capabilities,
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


@pytest.mark.parametrize("family", ["claude", "codex"])
def test_provider_contract_declares_each_lifecycle_capability(
    family: Literal["claude", "codex"], tmp_path: Path
) -> None:
    protocol = create_protocol((family,), tmp_path)
    assert protocol is not None
    capabilities = runtime_capabilities(family)
    assert set(capabilities.floor) == FLOOR
    assert all(
        state in {"native", "runtime", "unsupported"} for state in capabilities.floor.values()
    )
    assert capabilities.floor["start"] == "native"
    assert capabilities.floor["resume"] == "unsupported"
    assert capabilities.floor["recovery_report"] == "unsupported"
    assert set(protocol.actions) <= capabilities.native_actions


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
    with pytest.raises(TypeError):
        _ = RuntimeRecoveryRequest(
            identity=cast(ProviderSessionIdentity, "session_id: forged"), cwd=Path("/repo")
        )


def test_recovery_receipt_cannot_claim_unknown_turn_completed() -> None:
    identity = ProviderSessionIdentity.from_step("codex", ProtocolStep(session_id="thread-1"))
    assert identity is not None
    receipt = RecoveryReceipt(identity=identity, outcome="unknown_turn")
    assert not receipt.turn_confirmed
    with pytest.raises(ValueError):
        _ = replace(receipt, turn_confirmed=True)
