import { act, cleanup, render } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { resetStore, setSelection } from '../../app/store';
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
});
