import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { post } from '../../app/api';
import { resetStore, setSnapshot } from '../../app/store';
import { GoalReviewSidecar } from './GoalReviewSidecar';
import { loadReviews, resetReviews } from './reviewsState';
import { resetReviewSelection, selectReview } from './selection';

vi.mock('../../app/api', () => ({
  get: vi.fn().mockResolvedValue([
    {
      review_id: 1,
      goal_id: 5,
      goal_revision: 'rev',
      evidence: 'evidence text',
      proposed_change: 'change text',
      decision: 'pending',
      affected_node_ids: [5],
      reviewer: 'reviewer',
      assessed_at: '2026-01-01T00:00:00+00:00',
      decided_at: null,
      decided_by: null,
    },
  ]),
  post: vi.fn().mockResolvedValue({}),
}));

function capabilities(overrides: Record<string, unknown> = {}) {
  return {
    session_input: { available: true, reason: null },
    cancel: { available: true, reason: null },
    force_stop: { available: true, reason: null },
    stop_scheduling: { available: true, reason: null },
    graph_edits: { available: true, reason: null },
    review_decision: { available: true, reason: null },
    git: { available: true, reason: null },
    owner: { available: false },
    ...overrides,
  };
}

describe('GoalReviewSidecar', () => {
  beforeEach(async () => {
    resetStore();
    resetReviews();
    resetReviewSelection();
    vi.mocked(post).mockClear();
    await loadReviews();
  });

  afterEach(cleanup);

  it('renders nothing without a selected review', () => {
    const { container } = render(<GoalReviewSidecar />);
    expect(container.firstChild).toBeNull();
  });

  it('shows evidence, proposed change and held nodes for the selected review', () => {
    setSnapshot({
      goal: null,
      graph: {
        nodes: [
          { id: 5, description: 'Held node', status: 'pending', parent_id: null, kind: 'goal', flavor: null },
        ],
        edges: [],
        root_ids: [5],
      },
      capabilities: capabilities(),
    });
    selectReview(1);
    render(<GoalReviewSidecar />);

    expect(screen.getByText('evidence text')).toBeTruthy();
    expect(screen.getByText('change text')).toBeTruthy();
    expect(screen.getByText('Held node')).toBeTruthy();
  });

  it('posts an accepted decision and clears the selection', () => {
    setSnapshot({
      goal: null,
      graph: { nodes: [], edges: [], root_ids: [] },
      capabilities: capabilities(),
    });
    selectReview(1);
    render(<GoalReviewSidecar />);

    screen.getByText('Accept change').click();

    expect(post).toHaveBeenCalledWith('/api/reviews/1/decision', { decision: 'accepted' });
  });
});
