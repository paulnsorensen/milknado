from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from milknado.domains.coordinator import ProviderBinding
from milknado.domains.coordinator.persistence import (
    bind_provider_session,
    provider_bindings_for_session,
)
from milknado.domains.coordinator.workflow import CoordinatorWorkflow
from milknado.domains.graph import MikadoGraph


@pytest.mark.parametrize(
    ("binding", "message"),
    [
        (
            ProviderBinding("task", "task-1", "codex", "provider-2"),
            "invalid provider binding scope",
        ),
        (
            ProviderBinding("coordinator", "other", "codex", "provider-2"),
            "wrong scope identity",
        ),
        (
            ProviderBinding("execution_group", "", "codex", "provider-2"),
            "identities must not be empty",
        ),
        (
            ProviderBinding("execution_group", "group-2", "codex", ""),
            "identities must not be empty",
        ),
        (
            ProviderBinding("execution_group", "group-2", "unknown", "provider-2"),
            "unsupported provider family",
        ),
    ],
)
def test_invalid_binding_keeps_durable_binding(
    tmp_path: Path, binding: ProviderBinding, message: str
) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    with closing(sqlite3.connect(graph.db_path)) as conn:
        session = CoordinatorWorkflow(graph, conn).start_goal("Goal", "codex")
        original = ProviderBinding("coordinator", session.id, "codex", "provider-1")
        bind_provider_session(conn, session.id, original)
        with pytest.raises(ValueError, match=message):
            bind_provider_session(conn, session.id, binding)
        assert provider_bindings_for_session(conn, session.id) == (original,)
    graph.close()
