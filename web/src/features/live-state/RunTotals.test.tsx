import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { resetStore, setSnapshot } from '../../app/store';
import { mergeSnapshot, type RawStreamSnapshot } from './runtimeSnapshot';
import { RunTotals } from './RunTotals';

function baseSnapshot(): RawStreamSnapshot {
  return {
    goal: 'Ship it',
    graph: null,
    active_runs: [],
    event_lines: [],
  };
}

describe('RunTotals', () => {
  beforeEach(resetStore);
  afterEach(cleanup);

  it('renders nothing without a snapshot', () => {
    const { container } = render(<RunTotals />);
    expect(container.children.length).toBe(0);
  });

  it('marks omitted server totals as unknown', () => {
    setSnapshot(mergeSnapshot(baseSnapshot(), null));

    render(<RunTotals />);

    expect(screen.getByText('0 active · – completed · – failed · – stopped · – available')).toBeTruthy();
  });


  it('shows a zero count when a total is present but zero', () => {
    setSnapshot(mergeSnapshot({ ...baseSnapshot(), completed: 0 }, null));

    render(<RunTotals />);

    expect(screen.getByText(/0 completed/)).toBeTruthy();
  });
});
