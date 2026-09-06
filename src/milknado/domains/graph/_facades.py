"""Narrow synchronized sub-facades for MikadoGraph persistence clusters."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from contextlib import AbstractContextManager
from typing import Protocol

import milknado.domains.graph._persistence as _persistence
import milknado.domains.graph._reads as _reads
import milknado.domains.graph._run_persistence as _run_persistence
from milknado.domains.common import RunResult
from milknado.domains.graph._analytics_facade import synchronized


class _GraphHandle(Protocol):
    @property
    def synchronization_lock(self) -> AbstractContextManager[object]: ...

    @property
    def _conn(self) -> sqlite3.Connection: ...


class _SubFacade:
    def __init__(self, graph: _GraphHandle) -> None:
        self._graph: _GraphHandle = graph

    @property
    def synchronization_lock(self) -> AbstractContextManager[object]:
        return self._graph.synchronization_lock

    @property
    def _conn(self) -> sqlite3.Connection:
        return self._graph._conn  # pyright: ignore[reportPrivateUsage]


class _RunFacade(_SubFacade):
    @synchronized
    def start(  # noqa: PLR0913
        self,
        run_id: str,
        node_id: int,
        log_path: str,
        started_at: str,
        timeout_seconds: int | None,
        pid: int | None = None,
    ) -> None:
        _run_persistence.start_run(
            self._conn, run_id, node_id, log_path, started_at, timeout_seconds, pid
        )

    @synchronized
    def finish(self, run_id: str, result: RunResult) -> None:
        _run_persistence.finish_run(self._conn, run_id, result)

    @synchronized
    def set_pid(self, run_id: str, pid: int) -> None:
        _run_persistence.set_run_pid(self._conn, run_id, pid)

    @synchronized
    def get(self, run_id: str) -> _run_persistence.RunRecord | None:
        return _run_persistence.get_run(self._conn, run_id)

    @synchronized
    def for_node(self, node_id: int) -> list[_run_persistence.RunRecord]:
        return _run_persistence.runs_for_node(self._conn, node_id)

    @synchronized
    def latest_terminal(self, node_id: int, run_id: str) -> _run_persistence.RunRecord | None:
        return _run_persistence.latest_terminal_run(self._conn, node_id, run_id)

    @synchronized
    def latest_unowned_terminal(self, node_id: int) -> _run_persistence.RunRecord | None:
        return _reads.latest_unowned_terminal_run(self._conn, node_id)

    @synchronized
    def recent(self, limit: int) -> list[_run_persistence.RunRecord]:
        return _run_persistence.recent_runs(self._conn, limit)

    @synchronized
    def deposit_message(self, run_id: str, role: str, body: str, created_at: str) -> int:
        return _run_persistence.deposit_run_message(self._conn, run_id, role, body, created_at)

    @synchronized
    def deposit_review(self, run_id: str, verdict: str, findings: str, created_at: str) -> int:
        return _run_persistence.deposit_review_verdict(
            self._conn, run_id, verdict, findings, created_at
        )

    @synchronized
    def latest_message(self, run_id: str, role: str) -> str | None:
        return _run_persistence.latest_run_message(self._conn, run_id, role)

    @synchronized
    def insert_review(self, node_id: int, verdict: str, findings: str, created_at: str) -> int:
        return _run_persistence.insert_node_review(
            self._conn, node_id, verdict, findings, created_at
        )


class _FileFacade(_SubFacade):
    @synchronized
    def claim(self, node_id: int, files: list[str]) -> None:
        _persistence.set_file_ownership(self._conn, node_id, files)

    @synchronized
    def for_node(self, node_id: int) -> list[str]:
        return _persistence.get_file_ownership(self._conn, node_id)

    @synchronized
    def for_nodes(self, node_ids: Iterable[int] | None = None) -> dict[int, list[str]]:
        return _persistence.get_file_ownership_map(self._conn, node_ids)


class _GithubFacade(_SubFacade):
    @synchronized
    def attempt(self, goal_id: int) -> _persistence.GithubBindAttempt | None:
        return _persistence.get_github_bind_attempt(self._conn, goal_id)

    @synchronized
    def bind(self, goal_id: int, marker: str, issue_url: str | None, created_at: str) -> None:
        _persistence.set_github_bind_attempt(self._conn, goal_id, marker, issue_url, created_at)

    @synchronized
    def clear(self, goal_id: int) -> None:
        _persistence.clear_github_bind_attempt(self._conn, goal_id)


__all__ = ["_FileFacade", "_GithubFacade", "_RunFacade"]
