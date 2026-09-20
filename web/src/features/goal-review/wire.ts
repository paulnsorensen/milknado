// Wire type bound to `GoalReviewRecord`
// (`src/milknado/domains/graph/goal_review.py`), returned by `GET /api/reviews`.
export type WireGoalReviewDecision = 'pending' | 'accepted' | 'rejected';

export interface WireGoalReview {
  review_id: number;
  goal_id: number;
  goal_revision: string;
  evidence: string;
  proposed_change: string;
  decision: WireGoalReviewDecision;
  affected_node_ids: number[] | null;
  reviewer: string;
  assessed_at: string;
  decided_at: string | null;
  decided_by: string | null;
}
