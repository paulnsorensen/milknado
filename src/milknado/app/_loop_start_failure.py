"""Persist detached-loop startup failures."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from milknado.domains.common import NodeStatus, RunFenceLostError, RunResult
from milknado.domains.dispatch import now_iso
from milknado.domains.graph import MikadoGraph

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoopStartupFailure:
    node_id: int
    run_id: str
    attempt_id: str
    run_started: bool
    cause: Exception


def record_start_failure(graph: MikadoGraph, failure: LoopStartupFailure) -> None:
    node_written = False
    persistence_error: Exception | None = None
    if failure.run_started:
        try:
            graph.runs.finish(
                failure.run_id,
                RunResult(
                    status="failed",
                    exit_code=-1,
                    timed_out=False,
                    ended_at=now_iso(),
                    rebased=False,
                    detail=f"start failed: {type(failure.cause).__name__}: {failure.cause}",
                ),
            )
        except RunFenceLostError:
            _logger.info("start failure run already finalized: %s", failure.run_id)
        except Exception as error:
            persistence_error = error
            _logger.exception("start failure run terminal write failed: run_id=%s", failure.run_id)
    outcome_error: Exception | None = None
    try:
        if graph.task_attempts.get(failure.attempt_id) is not None:
            graph.task_attempts.require_nonfailure(
                failure.attempt_id, failure.run_id, "startup_failed"
            )
    except Exception as error:
        outcome_error = error
        _logger.exception("start failure task outcome write failed: run_id=%s", failure.run_id)
    try:
        node_written = graph.mark_terminal(failure.node_id, failure.run_id, NodeStatus.FAILED)
    except Exception as error:
        persistence_error = persistence_error or error
        _logger.exception("start failure node terminal write failed: node_id=%d", failure.node_id)
    if outcome_error is not None:
        raise outcome_error
    if persistence_error is not None or node_written is False:
        detail = (
            "start failure persistence failed: terminal writes incomplete: "
            + f"node_written={node_written}"
        )
        _logger.error("%s run_id=%s node_id=%d", detail, failure.run_id, failure.node_id)
        raise RuntimeError(detail) from persistence_error
