// The `rail-section` Reviews rail and the `sidecar` goal review detail.
import { registerSlot } from '../../app/slots';
import { GoalReviewSidecar } from './GoalReviewSidecar';
import { ReviewSelectionSync } from './ReviewSelectionSync';
import { ReviewsRail } from './ReviewsRail';

export function register(): void {
  registerSlot('rail-section', () => <ReviewsRail />);
  registerSlot('sidecar', () => <GoalReviewSidecar />);
  registerSlot('provider', () => <ReviewSelectionSync />);
}
