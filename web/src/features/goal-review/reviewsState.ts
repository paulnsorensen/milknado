// The pending-review list: loaded from `GET /api/reviews` and reloaded after
// every decision so the rail reflects the server's current pending set.
import { get, post } from '../../app/api';
import type { WireGoalReview, WireGoalReviewDecision } from './wire';

let reviews: WireGoalReview[] = [];
const listeners = new Set<() => void>();

function emit(): void {
  for (const listener of listeners) {
    listener();
  }
}

export function getReviews(): WireGoalReview[] {
  return reviews;
}

export function subscribeReviews(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export async function loadReviews(): Promise<void> {
  const loaded = await get<WireGoalReview[]>('/api/reviews');
  reviews = loaded ?? [];
  emit();
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
}
