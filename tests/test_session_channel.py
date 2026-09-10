from __future__ import annotations

from typing import Literal

import pytest

from milknado.domains.common import SessionContext, SessionEvent, SessionInput
from milknado.loop.sessions import SessionChannel

_CONTEXT = SessionContext(family="omp", cwd="/repo", base_oid="base")


def test_capacity_includes_inputs_waiting_for_vendor_receipts() -> None:
    channel = SessionChannel(max_inputs=1)
    channel.start(_CONTEXT, ("steer",))
    assert channel.submit(SessionInput(action="steer", text="first"))
    (sent,) = channel.drain()
    assert not channel.submit(SessionInput(action="steer", text="second"))
    channel.publish(
        SessionEvent(kind="user", text=sent.text, event_id=sent.request_id, state="delivered")
    )
    assert channel.submit(SessionInput(action="steer", text="third"))
    assert [
        (event.text, event.state) for event in channel.view().events if event.kind == "user"
    ] == [
        ("first", "delivered"),
        ("second", "rejected"),
        ("third", "queued"),
    ]


@pytest.mark.parametrize(
    ("decision", "terminal_state"),
    (("approve", "approved"), ("deny", "denied")),
)
def test_terminal_permission_decisions_release_capacity_and_close_cleanly(
    decision: Literal["approve", "deny"], terminal_state: Literal["approved", "denied"]
) -> None:
    channel = SessionChannel(max_inputs=1)
    channel.start(_CONTEXT, ("steer", "approve", "deny"))
    channel.publish(
        SessionEvent(
            kind="permission",
            text="write file",
            event_id="permission-1",
            state="requested",
        )
    )
    permission = channel.view().permissions[0]

    assert channel.submit(SessionInput(action=decision, request_id=permission.event_id))
    (submitted,) = channel.drain()
    channel.publish(
        SessionEvent(
            kind="permission",
            text="write file",
            event_id=submitted.request_id,
            state="submitted",
        )
    )
    channel.publish(
        SessionEvent(
            kind="permission",
            text="write file",
            event_id=submitted.request_id,
            state=terminal_state,
        )
    )

    assert channel.submit(SessionInput(action="steer", text="after decision"))
    channel.close()

    permission_receipts = [
        event
        for event in channel.view().events
        if event.kind == "user" and event.event_id == permission.event_id
    ]
    assert [event.state for event in permission_receipts] == ["submitted"]
    assert all(event.state != "unconfirmed" for event in channel.view().events)


def test_close_distinguishes_unsent_inputs_from_unconfirmed_delivery() -> None:
    channel = SessionChannel()
    channel.start(_CONTEXT, ("steer",))
    assert channel.submit(SessionInput(action="steer", text="sent"))
    _ = channel.drain()
    assert channel.submit(SessionInput(action="steer", text="unsent"))
    channel.close()
    assert [
        (event.text, event.state) for event in channel.view().events if event.kind == "user"
    ] == [
        ("sent", "unconfirmed"),
        ("unsent", "rejected"),
    ]
    assert channel.view().active is False


def test_failed_shutdown_preserves_unsent_input_until_its_receipt_is_saved() -> None:
    failed = False

    def persist(event: SessionEvent) -> None:
        nonlocal failed
        if event.kind == "user" and event.state == "rejected" and not failed:
            failed = True
            raise OSError("session store unavailable")

    channel = SessionChannel(sink=persist)
    channel.start(_CONTEXT, ("steer",))
    assert channel.submit(SessionInput(action="steer", text="unsent"))
    with pytest.raises(OSError, match="session store unavailable"):
        channel.close()
    with pytest.raises(RuntimeError, match="unpersisted"):
        channel.start(_CONTEXT, ("steer",))
    channel.close()
    assert [
        (event.text, event.state) for event in channel.view().events if event.kind == "user"
    ] == [
        ("unsent", "rejected"),
    ]
