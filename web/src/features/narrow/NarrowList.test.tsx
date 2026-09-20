import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { getState, resetStore, setSelection, setSnapshot } from '../../app/store';
import { mergeSnapshot, type RawStreamSnapshot } from '../live-state/runtimeSnapshot';
import { NarrowList } from './NarrowList';

function snapshotWithNodes(): RawStreamSnapshot {
  return {
    goal: 'Ship it',
    graph: {
      nodes: [
        { id: 1, description: 'Goal', status: 'pending', parent_id: null, kind: 'goal', flavor: null },
        { id: 2, description: 'Task one', status: 'pending', parent_id: 1, kind: 'task', flavor: null },
      ],
      edges: [{ parent_id: 1, child_id: 2 }],
      root_ids: [1],
    },
    active_runs: [],
    event_lines: [],
  };
}

describe('NarrowList', () => {
  beforeEach(() => {
    resetStore();
    setSnapshot(mergeSnapshot(snapshotWithNodes(), null));
  });

  afterEach(cleanup);

  it('shows the status strip and outline tree for every node', () => {
    render(<NarrowList onOpen={vi.fn()} />);

    expect(screen.getByText('Goal')).toBeInTheDocument();
    expect(screen.getByText('Task one')).toBeInTheDocument();
  });

  it('jumping to a known node id selects it', () => {
    render(<NarrowList onOpen={vi.fn()} />);

    fireEvent.change(screen.getByLabelText('Jump to node'), { target: { value: '2' } });
    fireEvent.click(screen.getByRole('button', { name: 'Jump' }));

    expect(getState().selection).toBe(2);
  });

  it('jumping to an unknown node id leaves the selection unchanged', () => {
    render(<NarrowList onOpen={vi.fn()} />);

    fireEvent.change(screen.getByLabelText('Jump to node'), { target: { value: '99' } });
    fireEvent.click(screen.getByRole('button', { name: 'Jump' }));

    expect(getState().selection).toBeNull();
  });

  it('disables the open-node bar until a node is selected', () => {
    render(<NarrowList onOpen={vi.fn()} />);

    expect(screen.getByRole('button', { name: 'Open node' })).toBeDisabled();
  });

  it('opens the selected node from the open-node bar', () => {
    setSelection(2);
    const onOpen = vi.fn();
    render(<NarrowList onOpen={onOpen} />);

    fireEvent.click(screen.getByRole('button', { name: 'Open node' }));

    expect(onOpen).toHaveBeenCalledWith(2);
  });

  it('opens a node from the outline tree', () => {
    const onOpen = vi.fn();
    render(<NarrowList onOpen={onOpen} />);

    fireEvent.keyDown(screen.getByRole('treeitem', { name: 'Task one' }), { key: 'Enter' });

    expect(onOpen).toHaveBeenCalledWith(2);
  });
});
