// The `sidecar` contribution: the selected goal review's proposed change,
// evidence, held nodes, and the Accept/Reject decision buttons.
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
  const { Button, StatusBadge } = Milknado;
  const review = reviews.find((candidate) => candidate.review_id === selectedReviewId);

  if (!review) {
    return null;
  }

  const nodes = store.snapshot?.graph?.nodes ?? [];
  const heldNodes = (review.affected_node_ids ?? []).map((nodeId) => ({
    id: nodeId,
    title: nodes.find((node) => node.id === nodeId)?.description ?? String(nodeId),
  }));
  const decisionCapability = store.capabilities?.review_decision;

  function decide(decision: 'accepted' | 'rejected'): void {
    void decideReview(review!.review_id, decision).then(clearReviewSelection);
  }

  return (
    <div className="mk-goal-review mk-stack">
      <div className="mk-sidecar-head">
        <span className="mk-kicker is-live">Goal review {review.review_id}</span>
        <Button icon className="mk-btn-ctl" ariaLabel="Close the review" onClick={clearReviewSelection}>
          {'×'}
        </Button>
      </div>
      <h2 className="mk-sidecar-title">{store.snapshot?.goal ?? `Goal ${review.goal_id}`}</h2>
      <div className="mk-badge-row">
        <StatusBadge state="at-risk">Waits for a person</StatusBadge>
        <span className="mk-text-data mk-muted">
          goal {review.goal_id} {'·'} revision {review.goal_revision} {'·'} {review.reviewer}
        </span>
      </div>
      <p className="mk-text-body mk-muted">
        Milknado holds the affected nodes until a person decides. All other nodes continue.
      </p>
      <section className="mk-section">
        <span className="mk-kicker">Proposed change</span>
        <p className="mk-text-body">{review.proposed_change}</p>
      </section>
      <section className="mk-section">
        <span className="mk-kicker">Evidence</span>
        <p className="mk-text-body">{review.evidence}</p>
      </section>
      {heldNodes.length > 0 && (
        <section className="mk-section">
          <span className="mk-kicker">Affected nodes {'·'} held</span>
          <dl>
            {heldNodes.map((node) => (
              <div key={node.id} className="mk-kv" style={{ gridTemplateColumns: '56px 1fr' }}>
                <dt>node {node.id}</dt>
                <dd>{node.title}</dd>
              </div>
            ))}
          </dl>
        </section>
      )}
      <div className="mk-button-row">
        <Button variant="primary" disabled={!decisionCapability?.available} onClick={() => decide('accepted')}>
          Accept change
        </Button>
        <Button disabled={!decisionCapability?.available} onClick={() => decide('rejected')}>
          Reject change
        </Button>
      </div>
      {decisionCapability && !decisionCapability.available && (
        <p role="note" className="mk-note">
          {decisionCapability.reason}
        </p>
      )}
    </div>
  );
}