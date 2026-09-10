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


def test_nonpositive_capacity_is_rejected() -> None:
    with pytest.raises(ValueError, match="capacities must be positive"):
        _ = SessionChannel(max_events=0)
    with pytest.raises(ValueError, match="capacities must be positive"):
        _ = SessionChannel(max_inputs=0)


def test_active_channel_cannot_replace_its_context() -> None:
    channel = SessionChannel()
    channel.start(_CONTEXT, ("steer",))

    with pytest.raises(RuntimeError, match="cannot replace an active session context"):
        channel.start(SessionContext(family="codex", cwd="/repo", base_oid="base"), ("steer",))


def test_submit_rejects_inactive_unsupported_and_unresolved_permission_inputs() -> None:
    channel = SessionChannel()
    assert not channel.submit(SessionInput(action="steer", text="inactive"))
    assert channel.view().events[-1].state == "rejected"

    channel.start(_CONTEXT, ("steer",))
    assert not channel.submit(SessionInput(action="follow_up", text="unsupported"))
    assert channel.view().events[-1].text == "unsupported"

    channel.start(_CONTEXT, ("approve",))
    assert not channel.submit(SessionInput(action="approve", request_id="missing"))
    assert channel.view().events[-1].state == "rejected"


def test_streaming_deltas_are_deferred_until_the_next_flush(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("milknado.loop.sessions._channel.time.monotonic", lambda: 1.0)
    persisted: list[SessionEvent] = []
    channel = SessionChannel(sink=persisted.append)
    channel.start(_CONTEXT, ("steer",))
    channel.publish(SessionEvent(kind="status", text="started", state="running"))
    delta = SessionEvent(
        kind="assistant",
        text="chunk",
        event_id="assistant-1",
        state="streaming",
        delta=True,
    )
    channel.publish(delta)

    channel.publish(SessionEvent(kind="status", text="flush", state="running"))
    assert [
        (event.kind, event.text, event.event_id, event.state, event.delta) for event in persisted
    ] == [
        ("status", "started", "", "running", False),
        ("assistant", "chunk", "1/assistant-1", "streaming", False),
        ("status", "flush", "", "running", False),
    ]


def test_failed_deferred_persistence_is_retried_without_losing_the_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("milknado.loop.sessions._channel.time.monotonic", lambda: 1.0)
    persisted: list[SessionEvent] = []
    fail_delta = True

    def sink(event: SessionEvent) -> None:
        nonlocal fail_delta
        if event.kind == "assistant" and fail_delta:
            fail_delta = False
            raise OSError("sink unavailable")
        persisted.append(event)

    channel = SessionChannel(sink=sink)
    channel.start(_CONTEXT, ("steer",))
    channel.publish(SessionEvent(kind="status", text="started", state="running"))
    delta = SessionEvent(
        kind="assistant",
        text="chunk",
        event_id="assistant-1",
        state="streaming",
        delta=True,
    )
    channel.publish(delta)
    with pytest.raises(OSError, match="sink unavailable"):
        channel.publish(SessionEvent(kind="status", text="flush", state="running"))

    channel.publish(SessionEvent(kind="status", text="retry", state="running"))
    assert [event.text for event in persisted] == ["started", "chunk", "retry"]


def test_event_retention_evicts_the_oldest_indexed_event() -> None:
    channel = SessionChannel(max_events=1)
    channel.start(_CONTEXT, ("steer",))
    channel.publish(SessionEvent(kind="status", text="first", event_id="one", state="running"))
    channel.publish(SessionEvent(kind="status", text="second", event_id="two", state="running"))

    assert [(event.event_id, event.text) for event in channel.view().events] == [
        ("1/two", "second")
    ]


def test_permission_cancellation_failure_does_not_block_a_later_close() -> None:
    persisted: list[SessionEvent] = []

    def sink(event: SessionEvent) -> None:
        if event.kind == "permission" and event.state == "cancelled":
            raise OSError("permission save failed")
        if event.kind == "status" and event.state == "stopped":
            raise OSError("stop save failed")
        persisted.append(event)

    channel = SessionChannel(sink=sink)
    channel.start(_CONTEXT, ("approve",))
    channel.publish(
        SessionEvent(
            kind="permission",
            text="write file",
            event_id="permission-1",
            state="requested",
        )
    )
    with pytest.raises(OSError, match="permission save failed"):
        channel.close()
    assert channel.view().active is False

    channel.set_sink(persisted.append)
    channel.close()
    assert [(event.kind, event.state) for event in persisted] == [
        ("permission", "requested"),
        ("permission", "cancelled"),
        ("status", "stopped"),
    ]
