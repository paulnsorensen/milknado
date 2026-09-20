import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { get } from '../../app/api';
import { resetStore, setSelection, setSnapshot } from '../../app/store';
import { mergeSnapshot, type RawStreamSnapshot } from '../live-state/runtimeSnapshot';
import { resetTab } from '../node-sidecar/detailTab';
import { resetDetail } from '../node-sidecar/nodeDetail';
import { NarrowDetail } from './NarrowDetail';

vi.mock('../../app/api', () => ({ get: vi.fn() }));

function snapshotWithNodes(): RawStreamSnapshot {
  return {
    goal: 'Ship it',
    graph: {
      nodes: [
        { id: 1, description: 'Goal', status: 'pending', parent_id: null, kind: 'goal', flavor: null },
      ],
      edges: [],
      root_ids: [1],
    },
    active_runs: [],
    event_lines: [],
  };
}

describe('NarrowDetail', () => {
  beforeEach(() => {
    resetStore();
    resetDetail();
    resetTab();
    vi.mocked(get).mockReset();
    vi.mocked(get).mockResolvedValue(null);
    setSnapshot(mergeSnapshot(snapshotWithNodes(), null));
    setSelection(1);
  });

  afterEach(cleanup);

  it('renders the node sidecar content full-width with a back control', () => {
    render(<NarrowDetail onBack={vi.fn()} />);

    expect(screen.getByRole('button', { name: 'Back to list' })).toBeInTheDocument();
  });

  it('calls onBack when the back control is used', () => {
    const onBack = vi.fn();
    render(<NarrowDetail onBack={onBack} />);

    fireEvent.click(screen.getByRole('button', { name: 'Back to list' }));

    expect(onBack).toHaveBeenCalledOnce();
  });
});
