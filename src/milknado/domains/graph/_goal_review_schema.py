"""SQLite schema statements for top-level goal reviews."""

CREATE_GOAL_REVIEWS = (
    "CREATE TABLE IF NOT EXISTS goal_reviews ("
    + "review_id INTEGER PRIMARY KEY AUTOINCREMENT, "
    + "goal_id INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE, "
    + "goal_revision TEXT NOT NULL, evidence TEXT NOT NULL, "
    + "proposed_change TEXT NOT NULL, "
    + "decision TEXT NOT NULL CHECK (decision IN ('pending', 'accepted', 'rejected')), "
    + "affected_node_ids TEXT, reviewer TEXT NOT NULL, assessed_at TEXT NOT NULL, "
    + "decided_at TEXT, decided_by TEXT)"
)
CREATE_PENDING_GOAL_REVIEW_INDEX = (
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_goal_reviews_pending "
    + "ON goal_reviews(goal_id) WHERE decision = 'pending'"
)
