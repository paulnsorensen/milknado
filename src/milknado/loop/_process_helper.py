"""Launch and confirm one fixed adapter-owned worker lifeline."""

from __future__ import annotations

import os
import select
import subprocess
import sys
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass

import psutil

from milknado.domains.common import HelperIdentity
from milknado.loop._process_contract import ProtectionContext
from milknado.loop._process_gate import WorkerProcess


@dataclass(frozen=True, slots=True)
class HelperStart:
    sequence: int
    generation: int
    deadline: float
    stop_deadline: Callable[[], float | None] | None = None


class UnconfirmedHelperExit(RuntimeError):
    pass


def _await_ready(
    helper: subprocess.Popen[str],
    expected: str,
    deadline: float,
    stop_deadline: Callable[[], float | None] | None = None,
) -> bool:
    if helper.stdout is None:
        return False
    fd = helper.stdout.fileno()
    frame = bytearray()
    while (remaining := deadline - time.monotonic()) > 0 and len(frame) < 256:
        if stop_deadline is not None and stop_deadline() is not None:
            return False
        timeout = min(remaining, 0.1) if stop_deadline else remaining
        ready, _, _ = select.select([fd], [], [], timeout)
        if not ready:
            if stop_deadline is None:
                return False
            continue
        if not (chunk := os.read(fd, 256 - len(frame))):
            return False
        frame.extend(chunk)
        if b"\n" in frame:
            line, _, extra = frame.partition(b"\n")
            try:
                return not extra and line.decode("utf-8") == expected
            except UnicodeDecodeError:
                return False
    return False


def stop_failed_helper(helper: subprocess.Popen[str], deadline: float) -> None:
    if helper.poll() is None:
        with suppress(ProcessLookupError):
            helper.terminate()
        try:
            _ = helper.wait(timeout=min(0.2, max(0, deadline - time.monotonic())))
        except subprocess.TimeoutExpired:
            if helper.poll() is None:
                with suppress(ProcessLookupError):
                    helper.kill()
    try:
        _ = helper.wait(timeout=max(0, deadline - time.monotonic()))
    except subprocess.TimeoutExpired as exc:
        raise UnconfirmedHelperExit("lifeline helper exit unconfirmed") from exc
    finally:
        if helper.stdout is not None:
            helper.stdout.close()
        if helper.stderr is not None:
            helper.stderr.close()


def start_helper(
    worker: WorkerProcess, context: ProtectionContext, request: HelperStart
) -> tuple[subprocess.Popen[str], int]:
    read_fd, write_fd = os.pipe()
    try:
        helper = subprocess.Popen(
            (
                sys.executable,
                "-m",
                "milknado.adapters._loop_lifeline",
                str(context.db_path),
                str(read_fd),
                worker.identity.invocation_id,
                str(request.generation),
            ),
            pass_fds=(read_fd,),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception:
        os.close(write_fd)
        raise
    finally:
        os.close(read_fd)
    try:
        identity = HelperIdentity(
            worker.identity.invocation_id,
            request.generation,
            helper.pid,
            psutil.Process(helper.pid).create_time(),
        )
        evidence = context.evidence.with_deadline(request.deadline)
        evidence.record_helper(identity)
        expected = (
            f"READY {identity.invocation_id} {identity.generation} {identity.pid} "
            f"{identity.start_token} {request.sequence}"
        )
        if not _await_ready(helper, expected, request.deadline - 0.5, request.stop_deadline):
            raise RuntimeError("lifeline READY mismatch")
        record = evidence.get_worker(worker.identity.invocation_id)
        if (
            record is None
            or record.ended_at is not None
            or record.observation_owner is not None
            or record.snapshot_seq != request.sequence
            or record.ready_generation != identity.generation
            or record.helper_generation != identity.generation
            or record.helper_pid != identity.pid
            or record.helper_start_token != identity.start_token
        ):
            raise RuntimeError("lifeline READY not durable")
        if request.stop_deadline is not None and request.stop_deadline() is not None:
            raise RuntimeError("lifeline replacement cancelled")
        return helper, write_fd
    except Exception:
        os.close(write_fd)
        stop_deadline = request.stop_deadline() if request.stop_deadline is not None else None
        limit = (
            min(request.deadline, stop_deadline) if stop_deadline is not None else request.deadline
        )
        stop_failed_helper(helper, limit)
        raise
