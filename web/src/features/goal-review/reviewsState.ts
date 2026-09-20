// The pending-review list: loaded from `GET /api/reviews` and reloaded after
// every decision so the rail reflects the server's current pending set. A
// failed load surfaces inline on the reviews rail, not as a global toast,
// because it belongs to that decision-gating surface.
import { get, post } from '../../app/api';
import type { WireGoalReview, WireGoalReviewDecision } from './wire';

let reviews: WireGoalReview[] = [];
let reviewsError: string | null = null;
const listeners = new Set<() => void>();

function emit(): void {
  for (const listener of listeners) {
    listener();
  }
}

export function getReviews(): WireGoalReview[] {
  return reviews;
}

export function getReviewsError(): string | null {
  return reviewsError;
}

export function subscribeReviews(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export async function loadReviews(): Promise<void> {
  try {
    const loaded = await get<WireGoalReview[]>('/api/reviews');
    reviews = loaded ?? [];
    reviewsError = null;
    emit();
  } catch {
    reviewsError = 'Goal reviews are unavailable.';
    emit();
  }
}

export async function decideReview(
  reviewId: number,
  decision: Exclude<WireGoalReviewDecision, 'pending'>,
): Promise<void> {
  await post(`/api/reviews/${reviewId}/decision`, { decision });
  await loadReviews();
}

export function resetReviews(): void {
  reviews = [];
  reviewsError = null;
}
