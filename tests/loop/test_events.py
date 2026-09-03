"""Tests for the event types and emitter implementations."""

import queue
from datetime import UTC
from typing import cast

from milknado.loop._events import (
    LOG_ERROR,
    LOG_INFO,
    BoundEmitter,
    Event,
    EventData,
    EventType,
    NullEmitter,
    QueueEmitter,
)
from tests.loop.helpers import drain_events  # pyright: ignore[reportUnknownVariableType]


class TestEvent:
    def test_default_data_is_empty_dict(self):
        event = Event(type=EventType.LOG_MESSAGE, run_id="r1")  # pyright: ignore[reportUnknownVariableType]
        assert event.data == {}  # pyright: ignore[reportUnknownMemberType]

    def test_default_timestamp_is_utc(self):
        event = Event(type=EventType.RUN_STARTED, run_id="r1")  # pyright: ignore[reportUnknownVariableType]
        assert event.timestamp.tzinfo == UTC


class TestNullEmitter:
    def test_emit_does_not_raise(self):
        emitter = NullEmitter()
        event = Event(type=EventType.RUN_STARTED, run_id="x")  # pyright: ignore[reportUnknownVariableType]
        emitter.emit(event)  # should not raise  # pyright: ignore[reportUnknownArgumentType]


class TestQueueEmitter:
    def test_emit_pushes_to_queue(self):
        emitter = QueueEmitter()
        event = Event(
            type=EventType.ITERATION_STARTED,
            run_id="r1",
            data=cast(EventData, cast(object, {"iteration": 1})),
        )

        emitter.emit(event)

        assert not emitter.queue.empty()
        assert emitter.queue.get() is event

    def test_multiple_events_queued_in_order(self):
        emitter = QueueEmitter()
        e1 = Event(type=EventType.RUN_STARTED, run_id="r1")  # pyright: ignore[reportUnknownVariableType]
        e2 = Event(type=EventType.ITERATION_STARTED, run_id="r1")  # pyright: ignore[reportUnknownVariableType]
        e3 = Event(type=EventType.RUN_STOPPED, run_id="r1")  # pyright: ignore[reportUnknownVariableType]

        emitter.emit(e1)  # pyright: ignore[reportUnknownArgumentType]
        emitter.emit(e2)  # pyright: ignore[reportUnknownArgumentType]
        emitter.emit(e3)  # pyright: ignore[reportUnknownArgumentType]

        assert emitter.queue.get() is e1
        assert emitter.queue.get() is e2
        assert emitter.queue.get() is e3

    def test_accepts_external_queue(self):
        q: queue.Queue[Event] = queue.Queue()  # pyright: ignore[reportMissingTypeArgument, reportUnknownVariableType]
        emitter = QueueEmitter(q)  # pyright: ignore[reportUnknownArgumentType]
        event = Event(type=EventType.RUN_STARTED, run_id="r1")  # pyright: ignore[reportUnknownVariableType]

        emitter.emit(event)  # pyright: ignore[reportUnknownArgumentType]

        assert q.get() is event


class TestBoundEmitter:
    def test_emits_event_with_fixed_run_id(self):
        q = QueueEmitter()
        emit = BoundEmitter(q, "run-abc")
        emit(EventType.ITERATION_STARTED, {"iteration": 1})

        events = drain_events(q)  # pyright: ignore[reportUnknownVariableType]
        assert len(events) == 1  # pyright: ignore[reportUnknownArgumentType]
        assert events[0].run_id == "run-abc"
        assert events[0].type == EventType.ITERATION_STARTED
        assert events[0].data == {"iteration": 1}  # pyright: ignore[reportUnknownMemberType]

    def test_emits_empty_data_when_none_provided(self):
        q = QueueEmitter()
        emit = BoundEmitter(q, "run-xyz")
        emit(EventType.RUN_STOPPED)

        events = drain_events(q)  # pyright: ignore[reportUnknownVariableType]
        assert len(events) == 1  # pyright: ignore[reportUnknownArgumentType]
        assert events[0].data == {}  # pyright: ignore[reportUnknownMemberType]

    def test_multiple_events_share_run_id(self):
        q = QueueEmitter()
        emit = BoundEmitter(q, "run-123")
        emit(EventType.RUN_STARTED)
        emit(EventType.ITERATION_STARTED, {"iteration": 1})
        emit(EventType.RUN_STOPPED)

        events = drain_events(q)  # pyright: ignore[reportUnknownVariableType]
        assert all(e.run_id == "run-123" for e in events)  # pyright: ignore[reportUnknownVariableType]
        assert len(events) == 3  # pyright: ignore[reportUnknownArgumentType]

    def test_log_info_emits_log_message_at_info_level(self):
        q = QueueEmitter()
        emit = BoundEmitter(q, "run-log")
        emit.log_info("Waiting 5s...")

        events = drain_events(q)  # pyright: ignore[reportUnknownVariableType]
        assert len(events) == 1  # pyright: ignore[reportUnknownArgumentType]
        assert events[0].type == EventType.LOG_MESSAGE
        assert events[0].data["message"] == "Waiting 5s..."  # pyright: ignore[reportUnknownMemberType]
        assert events[0].data["level"] == LOG_INFO  # pyright: ignore[reportUnknownMemberType]

    def test_log_error_emits_log_message_at_error_level(self):
        q = QueueEmitter()
        emit = BoundEmitter(q, "run-log")
        emit.log_error("Something broke")

        events = drain_events(q)  # pyright: ignore[reportUnknownVariableType]
        assert len(events) == 1  # pyright: ignore[reportUnknownArgumentType]
        assert events[0].type == EventType.LOG_MESSAGE
        assert events[0].data["message"] == "Something broke"  # pyright: ignore[reportUnknownMemberType]
        assert events[0].data["level"] == LOG_ERROR  # pyright: ignore[reportUnknownMemberType]
        assert "traceback" not in events[0].data  # pyright: ignore[reportUnknownMemberType]

    def test_log_error_includes_traceback_when_provided(self):
        q = QueueEmitter()
        emit = BoundEmitter(q, "run-log")
        emit.log_error("Crashed", traceback="Traceback (most recent call last):\n  ...")

        events = drain_events(q)  # pyright: ignore[reportUnknownVariableType]
        assert len(events) == 1  # pyright: ignore[reportUnknownArgumentType]
        assert events[0].data["message"] == "Crashed"  # pyright: ignore[reportUnknownMemberType]
        assert events[0].data["level"] == LOG_ERROR  # pyright: ignore[reportUnknownMemberType]
        assert events[0].data["traceback"] == "Traceback (most recent call last):\n  ..."  # pyright: ignore[reportUnknownMemberType]
