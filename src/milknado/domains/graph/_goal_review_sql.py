"""SQL fragments shared by goal-review admission queries."""

READY_NODE_ADMISSION_CTE = """
WITH RECURSIVE latest_reviews(goal_id, affected_node_ids, decision) AS (
    SELECT review.goal_id, review.affected_node_ids, review.decision
    FROM goal_reviews AS review
    WHERE review.review_id = (
        SELECT MAX(previous.review_id)
        FROM goal_reviews AS previous
        WHERE previous.goal_id = review.goal_id
    )
),
review_scope_roots(id) AS (
    SELECT goal_id
    FROM latest_reviews
    WHERE decision = 'pending' AND affected_node_ids IS NULL
    UNION
    SELECT CAST(scope.value AS INTEGER)
    FROM latest_reviews
    JOIN json_each(latest_reviews.affected_node_ids) AS scope
    WHERE decision = 'pending'
),
paused_review_nodes(id) AS (
    SELECT id FROM review_scope_roots
    UNION
    SELECT child.id
    FROM nodes AS child
    JOIN paused_review_nodes AS parent ON child.parent_id = parent.id
)
"""
READY_NODE_ADMISSION_FILTER = "n.id NOT IN (SELECT id FROM paused_review_nodes)"

__all__ = ["READY_NODE_ADMISSION_CTE", "READY_NODE_ADMISSION_FILTER"]
