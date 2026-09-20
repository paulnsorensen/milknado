import { cleanup, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { get } from '../../app/api';
import { ReviewsRail } from './ReviewsRail';
import { resetReviews } from './reviewsState';
import { getSelectedReviewId, resetReviewSelection } from './selection';

vi.mock('../../app/api', () => ({ get: vi.fn(), post: vi.fn() }));

const REVIEW = {
  review_id: 1,
  goal_id: 1,
  goal_revision: 'rev',
  evidence: 'evidence text',
  proposed_change: 'change text',
  decision: 'pending' as const,
  affected_node_ids: null,
  reviewer: 'reviewer',
  assessed_at: '2026-01-01T00:00:00+00:00',
  decided_at: null,
  decided_by: null,
};

describe('ReviewsRail', () => {
  beforeEach(() => {
    resetReviews();
    resetReviewSelection();
    vi.mocked(get).mockReset();
  });

  afterEach(cleanup);

  it('shows an empty state when there are no pending reviews', async () => {
    vi.mocked(get).mockResolvedValue([]);
    render(<ReviewsRail />);

    await waitFor(() => screen.getByText('No goal reviews are pending.'));
  });

  it('lists a pending review and selects it on Open', async () => {
    vi.mocked(get).mockResolvedValue([REVIEW]);
    render(<ReviewsRail />);

    await waitFor(() => screen.getByText('evidence text'));
    screen.getByText('Open').click();

    expect(getSelectedReviewId()).toBe(1);
  });
});
