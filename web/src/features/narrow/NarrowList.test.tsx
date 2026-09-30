import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { getState, resetStore, setSelection, setSnapshot } from '../../app/store';
import { NoticeToasts } from '../errors/NoticeToasts';
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
    completed: 3,
    failed: 4,
    stopped: 5,
    available: 6,
    event_lines: [],
  };
}

describe('NarrowList', () => {
  beforeEach(() => {
    resetStore();
    setSnapshot(mergeSnapshot(snapshotWithNodes(), null));
  });

function setHostOwnerCapabilities(): void {
  const snapshot = getState().snapshot;
  if (snapshot === null) {
    throw new Error('snapshot not initialized');
  }
  setSnapshot({
    ...snapshot,
    capabilities: {
      ...snapshot.capabilities,
      host_owner: { available: true, reason: null },
      owner: { available: false, reason: 'No live owner is connected.' },
    },
  });
}

  afterEach(cleanup);

  it('shows the status strip and outline tree for every node', () => {
    render(<NarrowList onOpen={vi.fn()} />);

    expect(screen.getByText('Goal')).toBeInTheDocument();
    expect(screen.getByText('Task one')).toBeInTheDocument();
    expect(screen.getByText('0 active · 3 completed · 4 failed · 5 stopped · 6 available')).toBeInTheDocument();
  });

  it('uses the host owner role when the per-run owner is unavailable', () => {
    setHostOwnerCapabilities();
    render(<NarrowList onOpen={vi.fn()} />);

    expect(screen.getByText('Run active')).toBeInTheDocument();
    expect(screen.queryByText('Read-only')).toBeNull();
  });
  it('shows pending permission requests in the narrow header', () => {
    setSnapshot({
      ...getState().snapshot!,
      capabilities: {
        ...getState().capabilities!,
        owner: { available: true, permission_ids: ['perm-1'] },
      },
    });
    render(<NarrowList onOpen={vi.fn()} />);

    expect(screen.getByRole('status', { name: 'Pending permission requests' })).toHaveTextContent(
      'Permission requested',
    );
  });

  it('jumping to a known node id selects it', () => {
    render(<NarrowList onOpen={vi.fn()} />);

    fireEvent.change(screen.getByLabelText('Jump to node'), { target: { value: '2' } });
    fireEvent.click(screen.getByRole('button', { name: 'Jump' }));

    expect(getState().selection).toBe(2);
  });

  it('jumping to an unknown node id leaves the selection unchanged and shows a notice', () => {
    render(
      <>
        <NarrowList onOpen={vi.fn()} />
        <NoticeToasts />
      </>,
    );

    fireEvent.change(screen.getByLabelText('Jump to node'), { target: { value: '99' } });
    fireEvent.click(screen.getByRole('button', { name: 'Jump' }));

    expect(getState().selection).toBeNull();
    expect(screen.getByText('Node 99 was not found.')).toBeInTheDocument();
  });

  it('rejects empty jump input without a notice', () => {
    render(
      <>
        <NarrowList onOpen={vi.fn()} />
        <NoticeToasts />
      </>,
    );

    fireEvent.change(screen.getByLabelText('Jump to node'), { target: { value: '  ' } });
    fireEvent.click(screen.getByRole('button', { name: 'Jump' }));

    expect(getState().selection).toBeNull();
    expect(getState().notices).toHaveLength(0);
  });

  it.each(['abc', '0x1f', '1e1', '1.5', 'Infinity'])(
    'rejects non-numeric jump input %s with a notice',
    (value) => {
      render(
        <>
          <NarrowList onOpen={vi.fn()} />
          <NoticeToasts />
        </>,
      );

      fireEvent.change(screen.getByLabelText('Jump to node'), { target: { value } });
      fireEvent.click(screen.getByRole('button', { name: 'Jump' }));

      expect(getState().selection).toBeNull();
      expect(screen.getByText(`Node ${value} was not found.`)).toBeInTheDocument();
    },
  );

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
