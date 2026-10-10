"""Execution-group schema statements for the forward-only graph ladder."""

CREATE_EXECUTION_GROUPS = (
    "CREATE TABLE IF NOT EXISTS execution_groups ("
    "id TEXT PRIMARY KEY, graph_id TEXT NOT NULL, "
    "worktree_path TEXT NOT NULL UNIQUE, branch_name TEXT NOT NULL UNIQUE, "
    "provider_session_id TEXT UNIQUE, "
    "source_group_id TEXT REFERENCES execution_groups(id), "
    "active_node_id INTEGER REFERENCES nodes(id), active_run_id TEXT)"
)
CREATE_EXECUTION_GROUP_TASKS = (
    "CREATE TABLE IF NOT EXISTS execution_group_tasks ("
    "group_id TEXT NOT NULL REFERENCES execution_groups(id) ON DELETE CASCADE, "
    "node_id INTEGER NOT NULL UNIQUE REFERENCES nodes(id), position INTEGER NOT NULL, "
    "status TEXT CHECK (status IN ('done', 'failed', 'blocked')), result TEXT, "
    "PRIMARY KEY (group_id, node_id), UNIQUE (group_id, position))"
)
ADD_ATTEMPT_ID = "ALTER TABLE execution_groups ADD COLUMN active_attempt_id TEXT"
CREATE_GRAPH_ALTERNATIVES = (
    "CREATE TABLE IF NOT EXISTS graph_alternatives ("
    "id TEXT PRIMARY KEY, source_graph_id TEXT, "
    "source_group_id TEXT REFERENCES execution_groups(id), "
    "created_at TEXT NOT NULL)"
)
CREATE_GROUP_CLAIM_TRIGGER = (
    "CREATE TRIGGER IF NOT EXISTS group_writer_claim BEFORE UPDATE OF status, run_id ON nodes "
    "WHEN NEW.status = 'running' AND OLD.status != 'running' "
    "AND EXISTS (SELECT 1 FROM execution_group_tasks t "
    "JOIN execution_groups g ON g.id = t.group_id "
    "WHERE t.node_id = NEW.id "
    "AND (g.active_node_id IS NOT NEW.id OR g.active_attempt_id IS NOT NEW.run_id)) "
    "BEGIN SELECT RAISE(ABORT, 'group writer admission required'); END"
)

ADD_RESERVED_NODE_STATUS = "ALTER TABLE execution_groups ADD COLUMN active_node_status TEXT"
ADD_RESERVED_NODE_RUN_ID = "ALTER TABLE execution_groups ADD COLUMN active_node_run_id TEXT"

RESET_STATEMENTS = (
    "DELETE FROM execution_group_tasks",
    "DELETE FROM graph_alternatives",
    "DELETE FROM execution_groups",
)
