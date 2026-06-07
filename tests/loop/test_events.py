"""Tests for the event types and emitter implementations."""

import queue
from datetime import UTC, datetime

from milknado.loop._events import (
    LOG_ERROR,
    LOG_INFO,
    BoundEmitter,
    Event,
    EventType,
    FanoutEmitter,
    NullEmitter,
    QueueEmitter,
)
from tests.loop.helpers import drain_events


class TestEvent:
    def test_to_dict_serialization(self):
        ts = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)
        event = Event(
            type=EventType.RUN_STARTED,
            run_id="abc123",
            data={"max_iterations": 5},
            timestamp=ts,
        )
        result = event.to_dict()

        assert result["type"] == "run_started"
        assert result["run_id"] == "abc123"
        assert result["data"] == {"max_iterations": 5}
        assert result["timestamp"] == "2025-01-15T12:00:00+00:00"

    def test_to_dict_empty_data(self):
        event = Event(type=EventType.RUN_STOPPED, run_id="x")

        result = event.to_dict()

        assert result["data"] == {}
        assert result["type"] == "run_stopped"

    def test_default_data_is_empty_dict(self):
        event = Event(type=EventType.LOG_MESSAGE, run_id="r1")
        assert event.data == {}

    def test_default_timestamp_is_utc(self):
        event = Event(type=EventType.RUN_STARTED, run_id="r1")
        assert event.timestamp.tzinfo == UTC


class TestNullEmitter:
    def test_emit_does_not_raise(self):
        emitter = NullEmitter()
        event = Event(type=EventType.RUN_STARTED, run_id="x")
        emitter.emit(event)  # should not raise

    def test_wants_agent_output_lines_false(self):
        assert NullEmitter().wants_agent_output_lines() is False


class TestQueueEmitter:
    def test_emit_pushes_to_queue(self):
        emitter = QueueEmitter()
        event = Event(type=EventType.ITERATION_STARTED, run_id="r1", data={"iteration": 1})

        emitter.emit(event)

        assert not emitter.queue.empty()
        assert emitter.queue.get() is event

    def test_multiple_events_queued_in_order(self):
        emitter = QueueEmitter()
        e1 = Event(type=EventType.RUN_STARTED, run_id="r1")
        e2 = Event(type=EventType.ITERATION_STARTED, run_id="r1")
        e3 = Event(type=EventType.RUN_STOPPED, run_id="r1")

        emitter.emit(e1)
        emitter.emit(e2)
        emitter.emit(e3)

        assert emitter.queue.get() is e1
        assert emitter.queue.get() is e2
        assert emitter.queue.get() is e3

    def test_accepts_external_queue(self):
        q: queue.Queue[Event] = queue.Queue()
        emitter = QueueEmitter(q)
        event = Event(type=EventType.RUN_STARTED, run_id="r1")

        emitter.emit(event)

        assert q.get() is event

    def test_wants_agent_output_lines_false(self):
        assert QueueEmitter().wants_agent_output_lines() is False


class TestBoundEmitter:
    def test_emits_event_with_fixed_run_id(self):
        q = QueueEmitter()
        emit = BoundEmitter(q, "run-abc")
        emit(EventType.ITERATION_STARTED, {"iteration": 1})

        events = drain_events(q)
        assert len(events) == 1
        assert events[0].run_id == "run-abc"
        assert events[0].type == EventType.ITERATION_STARTED
        assert events[0].data == {"iteration": 1}

    def test_emits_empty_data_when_none_provided(self):
        q = QueueEmitter()
        emit = BoundEmitter(q, "run-xyz")
        emit(EventType.RUN_PAUSED)

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
        assert events[0].data["message"] == "Waiting 5s..."
        assert events[0].data["level"] == LOG_INFO

    def test_log_error_emits_log_message_at_error_level(self):
        q = QueueEmitter()
        emit = BoundEmitter(q, "run-log")
        emit.log_error("Something broke")

        events = drain_events(q)
        assert len(events) == 1
        assert events[0].type == EventType.LOG_MESSAGE
        assert events[0].data["message"] == "Something broke"
        assert events[0].data["level"] == LOG_ERROR
        assert "traceback" not in events[0].data

    def test_agent_output_line_emits_event(self):
        q = QueueEmitter()
        emit = BoundEmitter(q, "run-peek")
        emit.agent_output_line("a line", "stdout", 3)

        events = drain_events(q)
        assert len(events) == 1
        assert events[0].type == EventType.AGENT_OUTPUT_LINE
        assert events[0].data == {
            "line": "a line",
            "stream": "stdout",
            "iteration": 3,
        }

    def test_wants_agent_output_lines_delegates(self):
        """BoundEmitter delegates to the underlying emitter dynamically."""
        q = QueueEmitter()
        emit = BoundEmitter(q, "run-1")
        assert emit.wants_agent_output_lines() is False

        null = NullEmitter()
        emit2 = BoundEmitter(null, "run-2")
        assert emit2.wants_agent_output_lines() is False

    def test_log_error_includes_traceback_when_provided(self):
        q = QueueEmitter()
        emit = BoundEmitter(q, "run-log")
        emit.log_error("Crashed", traceback="Traceback (most recent call last):\n  ...")

        events = drain_events(q)
        assert len(events) == 1
        assert events[0].data["message"] == "Crashed"
        assert events[0].data["level"] == LOG_ERROR
        assert events[0].data["traceback"] == "Traceback (most recent call last):\n  ..."


class TestFanoutEmitter:
    def test_broadcasts_to_all_emitters(self):
        q1 = QueueEmitter()
        q2 = QueueEmitter()
        fanout = FanoutEmitter([q1, q2])
        event = Event(type=EventType.RUN_STARTED, run_id="r1")

        fanout.emit(event)

        assert q1.queue.get() is event
        assert q2.queue.get() is event

    def test_empty_emitters_list(self):
        fanout = FanoutEmitter([])
        event = Event(type=EventType.RUN_STARTED, run_id="r1")
        fanout.emit(event)  # should not raise

    def test_fanout_with_null_emitter(self):
        null = NullEmitter()
        q = QueueEmitter()
        fanout = FanoutEmitter([null, q])
        event = Event(type=EventType.ITERATION_COMPLETED, run_id="r1")

        fanout.emit(event)

        assert q.queue.get() is event

    def test_wants_agent_output_lines_false_when_no_child_wants(self):
        fanout = FanoutEmitter([NullEmitter(), QueueEmitter()])
        assert fanout.wants_agent_output_lines() is False

    def test_wants_agent_output_lines_false_when_empty(self):
        assert FanoutEmitter([]).wants_agent_output_lines() is False
