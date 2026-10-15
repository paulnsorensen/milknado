"""Schema statements for the task-attempt ledger."""

CREATE_TASK_ATTEMPTS = (
    "CREATE TABLE IF NOT EXISTS task_attempts ("
    + "attempt_id TEXT PRIMARY KEY, "
    + "node_id INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE, "
    + "owner_run_id TEXT NOT NULL, active_invocation_id TEXT NOT NULL, "
    + "max_node_attempts INTEGER NOT NULL, start_state TEXT NOT NULL DEFAULT 'unknown', "
    + "outcome TEXT NOT NULL DEFAULT 'active', disposition TEXT, "
    + "failed_count INTEGER, reason TEXT)"
)

CREATE_NODE_OUTCOME_INDEX = (
    "CREATE INDEX IF NOT EXISTS idx_task_attempts_node_outcome "
    + "ON task_attempts(node_id, outcome)"
)

CREATE_RUN_ATTEMPT_INDEX = (
    "CREATE INDEX IF NOT EXISTS idx_runs_attempt_node ON runs(attempt_id, node_id)"
)

CREATE_LIVE_WORKER_INDEX = (
    "CREATE INDEX IF NOT EXISTS idx_run_workers_live_node "
    + "ON run_workers(node_id) WHERE ended_at IS NULL"
)
