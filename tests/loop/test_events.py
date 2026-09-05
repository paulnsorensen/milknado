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
    IterationStartedData,
    NoData,
    NullEmitter,
    QueueEmitter,
)
from tests.loop.helpers import drain_events, events_of_type


class TestEvent:
    def test_default_data_is_empty_dict(self):
        event: Event[NoData] = Event(type=EventType.LOG_MESSAGE, run_id="r1")
        assert event.data == {}

    def test_default_timestamp_is_utc(self):
        event: Event[NoData] = Event(type=EventType.RUN_STARTED, run_id="r1")
        assert event.timestamp.tzinfo == UTC


class TestNullEmitter:
    def test_emit_does_not_raise(self):
        emitter = NullEmitter()
        event: Event[NoData] = Event(type=EventType.RUN_STARTED, run_id="x")
        emitter.emit(event)  # should not raise


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
        e1: Event[NoData] = Event(type=EventType.RUN_STARTED, run_id="r1")
        e2: Event[NoData] = Event(type=EventType.ITERATION_STARTED, run_id="r1")
        e3: Event[NoData] = Event(type=EventType.RUN_STOPPED, run_id="r1")

        emitter.emit(e1)
        emitter.emit(e2)
        emitter.emit(e3)

        assert emitter.queue.get() is e1
        assert emitter.queue.get() is e2
        assert emitter.queue.get() is e3

    def test_accepts_external_queue(self):
        q: queue.Queue[Event[EventData]] = queue.Queue()
        emitter = QueueEmitter(q)
        event: Event[NoData] = Event(type=EventType.RUN_STARTED, run_id="r1")

        emitter.emit(event)

        assert q.get() is event


class TestBoundEmitter:
    def test_emits_event_with_fixed_run_id(self):
        q = QueueEmitter()
        emit = BoundEmitter(q, "run-abc")
        emit(EventType.ITERATION_STARTED, {"iteration": 1})

        events = drain_events(q)
        assert len(events) == 1
        assert events[0].run_id == "run-abc"
        assert events[0].type == EventType.ITERATION_STARTED
        iteration_event = events_of_type(events, EventType.ITERATION_STARTED)[0]
        assert iteration_event.data == IterationStartedData(iteration=1)

    def test_emits_empty_data_when_none_provided(self):
        q = QueueEmitter()
        emit = BoundEmitter(q, "run-xyz")
        emit(EventType.RUN_STOPPED)

        events = drain_events(q)
        assert len(events) == 1
        assert events[0].data == {}

    def test_multiple_events_share_run_id(self):
        q = QueueEmitter()
        emit = BoundEmitter(q, "run-123")
        emit(EventType.RUN_STARTED)
        emit(EventType.ITERATION_STARTED, {"iteration": 1})
        emit(EventType.RUN_STOPPED)

        events = drain_events(q)
        assert all(e.run_id == "run-123" for e in events)
        assert len(events) == 3

    def test_log_info_emits_log_message_at_info_level(self):
        q = QueueEmitter()
        emit = BoundEmitter(q, "run-log")
        emit.log_info("Waiting 5s...")

        events = drain_events(q)
        assert len(events) == 1
        assert events[0].type == EventType.LOG_MESSAGE
        log_event = events_of_type(events, EventType.LOG_MESSAGE)[0]
        assert log_event.data["message"] == "Waiting 5s..."
        assert log_event.data["level"] == LOG_INFO

    def test_log_error_emits_log_message_at_error_level(self):
        q = QueueEmitter()
        emit = BoundEmitter(q, "run-log")
        emit.log_error("Something broke")

        events = drain_events(q)
        assert len(events) == 1
        assert events[0].type == EventType.LOG_MESSAGE
        log_event = events_of_type(events, EventType.LOG_MESSAGE)[0]
        assert log_event.data["message"] == "Something broke"
        assert log_event.data["level"] == LOG_ERROR
        assert "traceback" not in log_event.data

    def test_log_error_includes_traceback_when_provided(self):
        q = QueueEmitter()
        emit = BoundEmitter(q, "run-log")
        emit.log_error("Crashed", traceback="Traceback (most recent call last):\n  ...")

        events = drain_events(q)
        assert len(events) == 1
        log_event = events_of_type(events, EventType.LOG_MESSAGE)[0]
        assert log_event.data["message"] == "Crashed"
        assert log_event.data["level"] == LOG_ERROR
        assert log_event.data.get("traceback") == "Traceback (most recent call last):\n  ..."
