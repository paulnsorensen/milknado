# ruff: noqa: RUF100
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from typing import cast
from uuid import uuid4

from milknado.domains.coordinator.model import CoordinatorSession, EntityLink, ProviderBinding
from milknado.domains.graph import GoalReviewSubjectError, top_level_goal

_LINK_KINDS = frozenset(
    {
        "planning_decision",
        "graph_revision",
        "approval",
        "run",
        "provider_session",
        "execution_group",
        "recovery",
    }
)


def _session(row: tuple[str, int, str, str]) -> CoordinatorSession:
    return CoordinatorSession(
        id=str(row[0]),
        goal_id=row[1],
        provider=str(row[2]),
        created_at=str(row[3]),
    )


def get_coordinator(conn: sqlite3.Connection, session_id: str) -> CoordinatorSession | None:  # noqa
    row = cast(
        tuple[str, int, str, str] | None,
        conn.execute(
            "SELECT id, goal_id, provider, created_at FROM coordinator_sessions WHERE id = ?",
            (session_id,),
        ).fetchone(),
    )
    return _session(row) if row is not None else None


def _require_top_level_goal(conn: sqlite3.Connection, goal_id: int) -> None:
    row_factory = conn.row_factory
    if row_factory is None:
        conn.row_factory = sqlite3.Row
    try:
        _ = top_level_goal(conn, goal_id)
    except GoalReviewSubjectError as exc:
        raise ValueError("coordinator requires a top-level goal") from exc
    finally:
        conn.row_factory = row_factory


def start_coordinator(conn: sqlite3.Connection, goal_id: int, provider: str) -> CoordinatorSession:  # noqa
    if not provider.strip():
        raise ValueError("provider must not be empty")
    _require_top_level_goal(conn, goal_id)
    with conn:
        _ = conn.execute(
            "INSERT OR IGNORE INTO coordinator_sessions (id, goal_id, provider, created_at) "
            + "VALUES (?, ?, ?, ?)",
            (uuid4().hex, goal_id, provider, datetime.now(UTC).isoformat()),
        )
    row = cast(
        tuple[str, int, str, str] | None,
        conn.execute(
            "SELECT id, goal_id, provider, created_at FROM coordinator_sessions WHERE goal_id = ?",
            (goal_id,),
        ).fetchone(),
    )
    assert row is not None
    session = _session(row)
    if session.provider != provider:
        raise ValueError("goal already has a coordinator with a different provider")
    return session


def link_entity(conn: sqlite3.Connection, session_id: str, kind: str, entity_id: str) -> None:  # noqa
    if kind not in _LINK_KINDS:
        raise ValueError(f"unsupported coordinator link kind: {kind}")
    if not entity_id:
        raise ValueError("entity_id must not be empty")
    with conn:
        _ = conn.execute(
            "INSERT OR IGNORE INTO coordinator_links (session_id, kind, entity_id) "
            + "VALUES (?, ?, ?)",
            (session_id, kind, entity_id),
        )


def links_for_session(conn: sqlite3.Connection, session_id: str) -> tuple[EntityLink, ...]:  # noqa
    rows = cast(
        list[tuple[str, str]],
        conn.execute(
            "SELECT kind, entity_id FROM coordinator_links WHERE session_id = ? ORDER BY seq",
            (session_id,),
        ).fetchall(),
    )
    return tuple(EntityLink(str(row[0]), str(row[1])) for row in rows)


def bind_provider_session(  # noqa: V103
    conn: sqlite3.Connection, coordinator_id: str, binding: ProviderBinding
) -> None:
    if binding.scope_kind not in {"coordinator", "execution_group"}:
        raise ValueError("invalid provider binding scope")
    if binding.scope_kind == "coordinator" and binding.scope_id != coordinator_id:
        raise ValueError("coordinator provider binding has wrong scope identity")
    if not binding.scope_id or not binding.provider_session_id:
        raise ValueError("provider binding identities must not be empty")
    if binding.family not in {"claude", "codex"}:
        raise ValueError("unsupported provider family")
    with conn:
        _ = conn.execute(
            "INSERT INTO coordinator_provider_bindings "
            + "(coordinator_id, scope_kind, scope_id, provider_family, provider_session_id) "
            + "VALUES (?, ?, ?, ?, ?)",
            (
                coordinator_id,
                binding.scope_kind,
                binding.scope_id,
                binding.family,
                binding.provider_session_id,
            ),
        )


def provider_bindings_for_session(
    conn: sqlite3.Connection, coordinator_id: str
) -> tuple[ProviderBinding, ...]:
    rows = cast(
        list[tuple[str, str, str, str]],
        conn.execute(
            "SELECT scope_kind, scope_id, provider_family, provider_session_id "
            + "FROM coordinator_provider_bindings WHERE coordinator_id = ? "
            + "ORDER BY CASE scope_kind WHEN 'coordinator' THEN 0 ELSE 1 END, scope_id",
            (coordinator_id,),
        ).fetchall(),
    )
    return tuple(ProviderBinding(*row) for row in rows)
