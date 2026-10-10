"""Coordinator statements for the graph schema ladder."""

CREATE_SESSIONS = (
    "CREATE TABLE IF NOT EXISTS coordinator_sessions ("
    "id TEXT PRIMARY KEY, "
    "goal_id INTEGER NOT NULL UNIQUE REFERENCES nodes(id) ON DELETE CASCADE, "
    "provider TEXT NOT NULL, "
    "created_at TEXT NOT NULL)"
)
CREATE_LINKS = (
    "CREATE TABLE IF NOT EXISTS coordinator_links ("
    "seq INTEGER PRIMARY KEY AUTOINCREMENT, "
    "session_id TEXT NOT NULL REFERENCES coordinator_sessions(id) ON DELETE CASCADE, "
    "kind TEXT NOT NULL, "
    "entity_id TEXT NOT NULL, "
    "UNIQUE (session_id, kind, entity_id))"
)
CREATE_EVENTS = (
    "CREATE TABLE IF NOT EXISTS coordinator_events ("
    "seq INTEGER PRIMARY KEY AUTOINCREMENT, "
    "session_id TEXT NOT NULL REFERENCES coordinator_sessions(id) ON DELETE CASCADE, "
    "kind TEXT NOT NULL, "
    "text TEXT NOT NULL, "
    "entity_kind TEXT NOT NULL, "
    "entity_id TEXT NOT NULL, "
    "tool_name TEXT NOT NULL, "
    "status TEXT NOT NULL, "
    "duration_ms INTEGER, "
    "created_at TEXT NOT NULL, "
    "expires_at TEXT, "
    "operation_hash TEXT)"
)
CREATE_EVENTS_SESSION_INDEX = (
    "CREATE INDEX IF NOT EXISTS idx_coordinator_events_session "
    "ON coordinator_events(session_id, seq)"
)
CREATE_EVENTS_EXPIRY_INDEX = (
    "CREATE INDEX IF NOT EXISTS idx_coordinator_events_expiry "
    "ON coordinator_events(expires_at) WHERE expires_at IS NOT NULL"
)
CREATE_EVENTS_OPERATION_INDEX = (
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_coordinator_events_operation "
    "ON coordinator_events(session_id, operation_hash) "
    "WHERE operation_hash IS NOT NULL"
)

CREATE_TURN_LAUNCHES = (
    "CREATE TABLE IF NOT EXISTS coordinator_turn_launches ("
    + "command_id TEXT PRIMARY KEY, coordinator_id TEXT NOT NULL, "
    + "scope_kind TEXT NOT NULL, scope_id TEXT NOT NULL, state TEXT NOT NULL)"
)
CREATE_ACTIVE_TURN_INDEX = (
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_coordinator_turn_active "
    + "ON coordinator_turn_launches(coordinator_id, scope_kind, scope_id) "
    + "WHERE state = 'submitted'"
)

CREATE_EVENTS_STREAM_INDEX = (
    "CREATE INDEX IF NOT EXISTS idx_coordinator_events_stream "
    "ON coordinator_events(session_id, stream_key, seq) WHERE stream_key IS NOT NULL"
)

CREATE_WEB_RECEIPTS = (
    "CREATE TABLE IF NOT EXISTS coordinator_web_receipts ("
    "command_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, "
    "command_hash TEXT NOT NULL, status TEXT NOT NULL, "
    "result_json TEXT NOT NULL)"
)

CORE_MIGRATIONS: tuple[tuple[int, str], ...] = (
    (41, CREATE_SESSIONS),
    (42, CREATE_LINKS),
    (43, CREATE_EVENTS),
    (44, CREATE_EVENTS_SESSION_INDEX),
    (45, CREATE_EVENTS_EXPIRY_INDEX),
    (46, CREATE_EVENTS_OPERATION_INDEX),
    (47, CREATE_WEB_RECEIPTS),
)

CREATE_DISPATCHES = (
    "CREATE TABLE IF NOT EXISTS coordinator_dispatches ("
    "attempt_id TEXT PRIMARY KEY, "
    "session_id TEXT NOT NULL REFERENCES coordinator_sessions(id) ON DELETE CASCADE, "
    "group_id TEXT NOT NULL, "
    "node_id INTEGER NOT NULL, "
    "run_id TEXT NOT NULL, "
    "state TEXT NOT NULL)"
)
CREATE_ACTION_RECEIPTS = (
    "CREATE TABLE IF NOT EXISTS coordinator_action_receipts ("
    "command_id TEXT PRIMARY KEY, "
    "session_id TEXT NOT NULL REFERENCES coordinator_sessions(id) ON DELETE CASCADE, "
    "provider_session_id TEXT NOT NULL, "
    "action_hash TEXT NOT NULL, "
    "state TEXT NOT NULL, "
    "created_at TEXT NOT NULL)"
)
CREATE_PLANS = (
    "CREATE TABLE IF NOT EXISTS coordinator_plans ("
    "operation_id TEXT PRIMARY KEY, "
    "session_id TEXT NOT NULL REFERENCES coordinator_sessions(id) ON DELETE CASCADE, "
    "result_json TEXT)"
)

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

RESET_STATEMENTS = (
    "DELETE FROM coordinator_turn_events",
    "DELETE FROM coordinator_provider_bindings",
    "DELETE FROM coordinator_links",
    "DELETE FROM coordinator_events",
    "DELETE FROM coordinator_dispatches",
    "DELETE FROM coordinator_action_receipts",
    "DELETE FROM coordinator_plans",
    "DELETE FROM coordinator_sessions",
    "DELETE FROM coordinator_web_receipts",
    "DELETE FROM coordinator_turn_launches",
)
