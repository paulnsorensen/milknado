import { act, cleanup, render } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { resetStore, setSelection, setSnapshot } from '../../app/store';

function capabilities() {
  return {
    session_input: { available: true, reason: null },
    cancel: { available: true, reason: null },
    force_stop: { available: true, reason: null },
    stop_scheduling: { available: true, reason: null },
    graph_edits: { available: true, reason: null },
    review_decision: { available: true, reason: null },
    git: { available: true, reason: null },
    host_owner: { available: true, reason: null },
    owner: { available: false },
  };
}
import { getSelectedReviewId, resetReviewSelection, selectReview } from './selection';
import { ReviewSelectionSync } from './ReviewSelectionSync';

describe('ReviewSelectionSync', () => {
  beforeEach(() => {
    resetStore();
    resetReviewSelection();
  });

  afterEach(cleanup);

  it('clears an open review synchronously once a node is selected', () => {
    selectReview(1);
    render(<ReviewSelectionSync />);

    act(() => {
      setSelection(5);
    });

    expect(getSelectedReviewId()).toBeNull();
  });

  it('leaves the review open while no node is selected', () => {
    selectReview(1);
    render(<ReviewSelectionSync />);

    act(() => {
      setSelection(null);
    });

    expect(getSelectedReviewId()).toBe(1);
  });

  it('clears an open review once a run id resolves to a selected node', () => {
    setSnapshot({
      goal: null,
      graph: null,
      capabilities: capabilities(),
      active_runs: [{ run_id: 'run-1', node_id: 5, description: 'work', status: 'running' }],
    });
    selectReview(1);
    render(<ReviewSelectionSync />);

    act(() => {
      setSelection('run-1');
    });

    expect(getSelectedReviewId()).toBeNull();
  });
});
