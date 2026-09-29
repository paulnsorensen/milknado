// The `rail-section` contribution: the pending goal reviews under a Reviews
// kicker, loaded on mount, with an empty state and an Open button per row.
import type { ReactElement } from 'react';
import { useEffect, useSyncExternalStore } from 'react';
import { setSelection } from '../../app/store';
import { Milknado } from '../../design-system';
import { getReviews, getReviewsError, loadReviews, subscribeReviews } from './reviewsState';
import { getSelectedReviewId, selectReview, subscribeReviewSelection } from './selection';

export function ReviewsRail(): ReactElement {
  const reviews = useSyncExternalStore(subscribeReviews, getReviews);
  const reviewsError = useSyncExternalStore(subscribeReviews, getReviewsError);
  const selectedReviewId = useSyncExternalStore(subscribeReviewSelection, getSelectedReviewId);
  const { Button } = Milknado;

  useEffect(() => {
    void loadReviews();
  }, []);

  // A review takes the sidecar over, so the node selection clears with it.
  function openReview(reviewId: number): void {
    setSelection(null);
    selectReview(reviewId);
  }

  return (
    <section className="mk-rail-section" aria-label="Reviews">
      <div className="mk-rail-head">
        <span className="mk-kicker">Reviews</span>
        <span className="mk-text-caption mk-muted">{reviews.length}</span>
      </div>
      {reviewsError && (
        <p role="alert" className="mk-rail-empty">
          {reviewsError}
        </p>
      )}
      {!reviewsError && reviews.length === 0 && (
        <p role="status" className="mk-rail-empty">
          No goal reviews are pending.
        </p>
      )}
      {reviews.map((review) => (
        <div
          key={review.review_id}
          className={
            review.review_id === selectedReviewId
              ? 'mk-rail-row mk-review-row is-selected'
              : 'mk-rail-row mk-review-row'
          }
        >
          <span className="mk-rail-row-copy">
            <span className="mk-rail-row-title">
              <span className="mk-accent-text" aria-hidden="true">
                {'▣ '}
              </span>
              Goal review {review.review_id}
            </span>
            <span className="mk-rail-row-sub">node {review.goal_id}</span>
          </span>
          <span className="mk-rail-row-state">{review.decision}</span>
          <Button variant="ghost" className="mk-btn-sm" onClick={() => openReview(review.review_id)}>
            Open
          </Button>
        </div>
      ))}
    </section>
  );
}