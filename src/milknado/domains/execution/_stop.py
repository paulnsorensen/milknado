"""Stop one graph run and confirm its durable workers."""

from __future__ import annotations

import time

from milknado.domains.common import LoopPort

LOOP_CANCEL_STOP_TIMEOUT_SECS = 5.0


def stop_graph_run(
    loop: LoopPort, unconfirmed: set[str], run_id: str, timeout: float | None
) -> bool:
    deadline = time.monotonic() + (
        timeout if timeout is not None else LOOP_CANCEL_STOP_TIMEOUT_SECS
    )
    unconfirmed.add(run_id)
    try:
        stopped = loop.stop_run(run_id, timeout=max(0, deadline - time.monotonic()))
    finally:
        workers_stopped = loop.stop_run_workers(run_id, deadline)
    if stopped and workers_stopped:
        unconfirmed.discard(run_id)
    return stopped and workers_stopped


def force_stop_graph_run(
    loop: LoopPort, unconfirmed: set[str], run_id: str, timeout: float | None
) -> bool:
    deadline = time.monotonic() + (
        timeout if timeout is not None else LOOP_CANCEL_STOP_TIMEOUT_SECS
    )
    unconfirmed.add(run_id)
    try:
        stopped = loop.force_stop_run(run_id, timeout=max(0, deadline - time.monotonic()))
    finally:
        workers_stopped = loop.stop_run_workers(run_id, deadline)
    if stopped and workers_stopped:
        unconfirmed.discard(run_id)
    return stopped and workers_stopped
