"""SQL fragments shared by goal-review admission and ready-node queries."""

from milknado.domains.common import NodeKind

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

# A goal without a structural child is an undecomposed roadmap stub, not executable work.
# Prerequisite edges do not count: only nodes.parent_id marks decomposition.
DECOMPOSED_GOAL_FILTER = (
    f"(n.kind != '{NodeKind.GOAL.value}' "
    + "OR EXISTS (SELECT 1 FROM nodes c WHERE c.parent_id = n.id))"
)

# Ready: unpaused, decomposed, and every prerequisite (outgoing edge child) is done.
READY_NODE_PREDICATE = (
    READY_NODE_ADMISSION_FILTER
    + " AND "
    + DECOMPOSED_GOAL_FILTER
    + " AND EXISTS (SELECT 1 FROM edges i WHERE i.child_id = n.id)"
    + " AND NOT EXISTS (SELECT 1 FROM edges e JOIN nodes c ON c.id = e.child_id "
    + "WHERE e.parent_id = n.id AND c.status != 'done')"
)

__all__ = [
    "DECOMPOSED_GOAL_FILTER",
    "READY_NODE_ADMISSION_CTE",
    "READY_NODE_ADMISSION_FILTER",
    "READY_NODE_PREDICATE",
]
