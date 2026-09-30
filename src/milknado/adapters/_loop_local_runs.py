"""Drain local verifier and reviewer runs into explicit results."""

from __future__ import annotations

import logging
import queue
import re
import time

from milknado.adapters._loop_types import ReviewVerdict
from milknado.adapters._loop_types import event_text as _event_text
from milknado.domains.common import VerifySpecResult
from milknado.loop import EventType, RunManager
from milknado.loop._events import Event, EventData

_logger = logging.getLogger(__name__)


def drain_verify_run(
    local_manager: RunManager,
    run_id: str,
    ev_queue: queue.Queue[Event[EventData]],
) -> VerifySpecResult:
    _ITERATION_EVENTS = frozenset((EventType.ITERATION_COMPLETED, EventType.ITERATION_FAILED))
    output_parts: list[str] = []
    deadline = time.monotonic() + 120.0
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _ = local_manager.stop_and_join(run_id, timeout=5.0)
                return VerifySpecResult(outcome="gaps", goal_delta="verification timed out")
            try:
                event = ev_queue.get(timeout=remaining)
            except queue.Empty:
                _ = local_manager.stop_and_join(run_id, timeout=5.0)
                return VerifySpecResult(outcome="gaps", goal_delta="verification timed out")
            if event.type in _ITERATION_EVENTS:
                text = _event_text(event.data, "result_text")
                if text:
                    output_parts.append(text)
            elif event.type == EventType.RUN_STOPPED:
                break
    except Exception as exc:
        _logger.exception("verify_spec drain failed for run_id=%s", run_id)
        try:
            _ = local_manager.stop_and_join(run_id, timeout=5.0)
        except Exception:
            _logger.exception("verify_spec stop failed for run_id=%s", run_id)
        return VerifySpecResult(outcome="gaps", goal_delta=f"verification failed: {exc}")
    return _parse_verify_output("\n".join(output_parts))


class UnconfirmedReviewStop(RuntimeError):
    def __init__(self, run_id: str) -> None:
        super().__init__(f"reviewer stop was not confirmed: {run_id}")


def _require_review_stop(local_manager: RunManager, run_id: str) -> None:
    try:
        if local_manager.stop_and_join(run_id, timeout=5.0):
            return
    except Exception as exc:
        raise UnconfirmedReviewStop(run_id) from exc
    raise UnconfirmedReviewStop(run_id)


def drain_review_run(
    local_manager: RunManager,
    run_id: str,
    ev_queue: queue.Queue[Event[EventData]],
    timeout_seconds: float,
) -> ReviewVerdict:
    """Drain one reviewer run and convert its final output into a verdict."""
    output_parts: list[str] = []
    deadline = time.monotonic() + timeout_seconds
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _require_review_stop(local_manager, run_id)
                return ReviewVerdict(
                    approved=False,
                    findings_md="reviewer timed out before producing a verdict",
                    error=True,
                )
            try:
                event = ev_queue.get(timeout=remaining)
            except queue.Empty:
                _require_review_stop(local_manager, run_id)
                return ReviewVerdict(
                    approved=False,
                    findings_md="reviewer timed out before producing a verdict",
                    error=True,
                )
            if event.type in {EventType.ITERATION_COMPLETED, EventType.ITERATION_FAILED}:
                text = _event_text(event.data, "result_text", "echo_stdout")
                if text:
                    output_parts.append(text)
            elif event.type == EventType.RUN_STOPPED:
                break
    except UnconfirmedReviewStop:
        raise
    except Exception as exc:
        _logger.exception("node review drain failed for run_id=%s", run_id)
        _require_review_stop(local_manager, run_id)
        return ReviewVerdict(approved=False, findings_md=f"reviewer failed: {exc}", error=True)
    return _parse_review_verdict("\n".join(output_parts))


def _parse_verify_output(output: str) -> VerifySpecResult:

    if "<result>done</result>" in output:
        return VerifySpecResult(outcome="done")
    if "<result>gaps</result>" in output:
        m = re.search(r"<goal_delta>(.*?)</goal_delta>", output, re.DOTALL)
        delta = m.group(1).strip() if m else None
        return VerifySpecResult(outcome="gaps", goal_delta=delta)
    _logger.warning("verify_spec: unparseable output, returning gaps")
    return VerifySpecResult(outcome="gaps", goal_delta="verification produced no explicit result")


def _parse_review_verdict(output: str) -> ReviewVerdict:
    verdicts = list(re.finditer(r"<verdict>\s*(approve|reject|revise)\s*</verdict>", output, re.I))
    marker_count = len(re.findall(r"</?verdict\b", output, re.I))
    if len(verdicts) != 1 or marker_count != 2:
        _logger.warning("run_node_review: unparseable or conflicting reviewer output")
        findings = output.strip() or "reviewer produced no parseable <verdict> tag"
        return ReviewVerdict(approved=False, findings_md=findings, error=True)
    match = verdicts[0]
    findings_md = output[: match.start()].strip()
    return ReviewVerdict(approved=match.group(1).lower() == "approve", findings_md=findings_md)
