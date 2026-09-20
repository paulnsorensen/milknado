// The selected review id, kept separate from `store.selection` (which is
// reserved for graph-node selection) so opening a review does not clobber it.
let selectedReviewId: number | null = null;
const listeners = new Set<() => void>();

function emit(): void {
  for (const listener of listeners) {
    listener();
  }
}

export function getSelectedReviewId(): number | null {
  return selectedReviewId;
}

export function subscribeReviewSelection(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function selectReview(reviewId: number): void {
  selectedReviewId = reviewId;
  emit();
}

export function clearReviewSelection(): void {
  selectedReviewId = null;
  emit();
}

export function resetReviewSelection(): void {
  selectedReviewId = null;
}
