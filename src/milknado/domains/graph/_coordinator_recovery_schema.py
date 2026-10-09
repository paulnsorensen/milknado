"""Coordinator recovery statements for the forward-only graph schema ladder."""

CREATE_PROVIDER_BINDINGS = (
    "CREATE TABLE IF NOT EXISTS coordinator_provider_bindings ("
    "coordinator_id TEXT NOT NULL REFERENCES coordinator_sessions(id) ON DELETE CASCADE, "
    "scope_kind TEXT NOT NULL CHECK (scope_kind IN ('coordinator', 'execution_group')), "
    "scope_id TEXT NOT NULL, "
    "provider_family TEXT NOT NULL CHECK (provider_family IN ('claude', 'codex')), "
    "provider_session_id TEXT NOT NULL, "
    "PRIMARY KEY (coordinator_id, scope_kind, scope_id), "
    "UNIQUE (coordinator_id, provider_family, provider_session_id), "
    "UNIQUE (provider_family, provider_session_id))"
)
CREATE_TURN_EVENTS = (
    "CREATE TABLE IF NOT EXISTS coordinator_turn_events ("
    "seq INTEGER PRIMARY KEY AUTOINCREMENT, "
    "coordinator_id TEXT NOT NULL, "
    "provider_family TEXT NOT NULL, "
    "provider_session_id TEXT NOT NULL, "
    "turn_id TEXT NOT NULL, "
    "status TEXT NOT NULL CHECK (status IN ('submitted', 'confirmed', 'unknown')), "
    "recorded_at TEXT NOT NULL, "
    "FOREIGN KEY (coordinator_id, provider_family, provider_session_id) "
    "REFERENCES coordinator_provider_bindings "
    "(coordinator_id, provider_family, provider_session_id) ON DELETE CASCADE)"
)
CREATE_UNKNOWN_TURN_INDEX = (
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_coordinator_turn_unknown "
    "ON coordinator_turn_events "
    "(coordinator_id, provider_family, provider_session_id, turn_id) "
    "WHERE status = 'unknown'"
)

CREATE_WEB_RECEIPTS = (
    "CREATE TABLE IF NOT EXISTS coordinator_web_receipts ("
    "command_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, "
    "command_hash TEXT NOT NULL, status TEXT NOT NULL, "
    "result_json TEXT NOT NULL)"
)
