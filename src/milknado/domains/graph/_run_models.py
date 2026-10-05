from __future__ import annotations

from typing import TypedDict


class RunRow(TypedDict):
    run_id: str
    node_id: int
    status: str
    pid: int | None
    log_path: str
    started_at: str
    ended_at: str | None
    timed_out: int
    exit_code: int | None
    error: str | None
    timeout_seconds: int | None
    detail: str | None
    rebased: int | None
    verification_status: str | None  # noqa: V107
    verified_at: str | None


class RunRecord(TypedDict):
    run_id: str
    node_id: int
    status: str
    pid: int | None
    log_path: str
    started_at: str
    ended_at: str | None
    timed_out: bool
    exit_code: int | None
    error: str | None
    timeout_seconds: int | None
    detail: str | None
    rebased: bool | None
    verification_status: str | None  # noqa: V107
    verified_at: str | None
