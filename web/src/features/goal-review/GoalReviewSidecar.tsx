// The `sidecar` contribution: the selected goal review's evidence, proposed
// change, held nodes, and the Accept/Reject decision buttons.
import type { ReactElement } from 'react';
import { useSyncExternalStore } from 'react';
import { getState, subscribe } from '../../app/store';
import { Milknado } from '../../design-system';
import { decideReview, getReviews, subscribeReviews } from './reviewsState';
import { clearReviewSelection, getSelectedReviewId, subscribeReviewSelection } from './selection';

export function GoalReviewSidecar(): ReactElement | null {
  const store = useSyncExternalStore(subscribe, getState);
  const reviews = useSyncExternalStore(subscribeReviews, getReviews);
  const selectedReviewId = useSyncExternalStore(subscribeReviewSelection, getSelectedReviewId);
  const { Button } = Milknado;
  const review = reviews.find((candidate) => candidate.review_id === selectedReviewId);

  if (!review) {
    return null;
  }

  const nodes = store.snapshot?.graph?.nodes ?? [];
  const heldNodes = (review.affected_node_ids ?? [])
    .map((nodeId) => nodes.find((node) => node.id === nodeId)?.description ?? String(nodeId));
  const decisionCapability = store.capabilities?.review_decision;

  function decide(decision: 'accepted' | 'rejected'): void {
    void decideReview(review!.review_id, decision).then(clearReviewSelection);
  }

  return (
    <div className="mk-goal-review-sidecar" style={{ width: 560 }}>
      <p>{review.evidence}</p>
      <p>{review.proposed_change}</p>
      {heldNodes.length > 0 && (
        <ul>
          {heldNodes.map((title, index) => (
            <li key={`${index}-${title}`}>{title}</li>
          ))}
        </ul>
      )}
      <Button disabled={!decisionCapability?.available} onClick={() => decide('accepted')}>
        Accept change
      </Button>
      <Button disabled={!decisionCapability?.available} onClick={() => decide('rejected')}>
        Reject change
      </Button>
      {decisionCapability && !decisionCapability.available && (
        <p role="note">{decisionCapability.reason}</p>
      )}
    </div>
  );
}
