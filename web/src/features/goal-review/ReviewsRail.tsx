// The `rail-section` contribution: the list of pending goal reviews, loaded
// on mount, with an empty state and an Open button per row.
import type { ReactElement } from 'react';
import { useEffect, useSyncExternalStore } from 'react';
import { Milknado } from '../../design-system';
import { getReviews, getReviewsError, loadReviews, subscribeReviews } from './reviewsState';
import { selectReview } from './selection';

export function ReviewsRail(): ReactElement {
  const reviews = useSyncExternalStore(subscribeReviews, getReviews);
  const reviewsError = useSyncExternalStore(subscribeReviews, getReviewsError);
  const { Button } = Milknado;

  useEffect(() => {
    void loadReviews();
  }, []);

  if (reviewsError) {
    return <p role="alert">{reviewsError}</p>;
  }

  if (reviews.length === 0) {
    return <p role="status">No goal reviews are pending.</p>;
  }

  return (
    <ul className="mk-reviews-rail">
      {reviews.map((review) => (
        <li key={review.review_id}>
          <span>{review.evidence}</span>
          <Button onClick={() => selectReview(review.review_id)}>Open</Button>
        </li>
      ))}
    </ul>
  );
}
