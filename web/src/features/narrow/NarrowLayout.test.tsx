import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { get } from '../../app/api';
import { resetStore, setSnapshot } from '../../app/store';
import { mergeSnapshot, type RawStreamSnapshot } from '../live-state/runtimeSnapshot';
import { resetDetail, resetTab } from '../../shared/node-detail';
import { NarrowLayout } from './NarrowLayout';

vi.mock('../../app/api', () => ({ get: vi.fn() }));

function stubMatchMedia(matches: boolean): void {
  vi.stubGlobal(
    'matchMedia',
    vi.fn().mockReturnValue({ matches, addEventListener: vi.fn(), removeEventListener: vi.fn() }),
  );
}

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

describe('NarrowLayout', () => {
  beforeEach(() => {
    resetStore();
    resetDetail();
    resetTab();
    vi.mocked(get).mockReset();
    vi.mocked(get).mockResolvedValue(null);
    setSnapshot(mergeSnapshot(snapshotWithNodes(), null));
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    cleanup();
  });

  it('renders the wide layout above the narrow breakpoint', () => {
    stubMatchMedia(false);

    render(<NarrowLayout />);

    expect(screen.queryByLabelText('Jump to node')).not.toBeInTheDocument();
    expect(document.querySelector('[data-region="canvas"]')).toBeInTheDocument();
  });

  it('opening a node from the narrow list switches to the detail view, and back returns to the list', () => {
    stubMatchMedia(true);

    render(<NarrowLayout />);
    expect(screen.getByLabelText('Jump to node')).toBeInTheDocument();

    fireEvent.keyDown(screen.getByRole('treeitem', { name: 'Goal' }), { key: 'Enter' });
    expect(screen.getByRole('button', { name: 'Back to list' })).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Back to list' }));
    expect(screen.getByLabelText('Jump to node')).toBeInTheDocument();
  });
});
