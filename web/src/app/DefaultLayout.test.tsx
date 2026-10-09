import { act, cleanup, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { getState, resetStore, setCoordinatorGraph, setGraphView, setSnapshot } from './store';
import { mergeSnapshot } from '../features/live-state/runtimeSnapshot';

vi.mock('../design-system', () => ({
  Milknado: {
    MikadoGraph: (props: {
      compact?: boolean;
      nodes: Array<{ id: string | number; parent: string | number | null; extra?: Array<string | number> }>;
      onLayout: (layout: { lod: string; collapsed: string[] }) => void;
    }) => (
      <button
        data-compact={String(props.compact)}
        data-nodes={JSON.stringify(props.nodes)}
        onClick={() => props.onLayout({ lod: 'pill', collapsed: ['x'] })}
      >
        trigger
      </button>
    ),
  },
}));

import { DefaultLayout } from './DefaultLayout';

describe('DefaultLayout', () => {
  beforeEach(() => {
    resetStore();
  });

  afterEach(cleanup);

  it('keeps the numeric collapsed selection when the graph reports a layout', () => {
    setGraphView({ collapsed: [42] });

    render(<DefaultLayout />);
    screen.getByText('trigger').click();

    expect(getState().graphView.lod).toBe('pill');
    expect(getState().graphView.collapsed).toEqual([42]);
  });

  it('uses compact graph cards so layout spacing matches the card width', () => {
    render(<DefaultLayout />);

    expect(screen.getByText('trigger')).toHaveAttribute('data-compact', 'true');
  });

  it('passes non-parent prerequisites to the canvas without changing tree parents', () => {
    setCoordinatorGraph({
      nodes: [
        { id: 1, description: 'Goal', status: 'pending', parent_id: null, kind: 'goal', flavor: null },
        { id: 2, description: 'Task', status: 'pending', parent_id: 1, kind: 'task', flavor: null },
        { id: 3, description: 'Prerequisite', status: 'pending', parent_id: 1, kind: 'task', flavor: null },
      ],
      edges: [
        { parent_id: 1, child_id: 2 },
        { parent_id: 1, child_id: 3 },
        { parent_id: 2, child_id: 3 },
      ],
      root_ids: [1],
    });

    render(<DefaultLayout />);

    const nodes = JSON.parse(screen.getByText('trigger').getAttribute('data-nodes') ?? '[]') as Array<{
      id: number;
      parent: number | null;
      extra?: number[];
    }>;
    expect(nodes.find((node) => node.id === 2)).toMatchObject({ parent: 1, extra: [3] });
    expect(nodes.find((node) => node.id === 1)?.extra).toBeUndefined();
    expect(nodes.find((node) => node.id === 3)?.extra).toBeUndefined();
  });
  it('restores the untouched execution graph when the coordinator clears', () => {
    const ordinary = mergeSnapshot({
      goal: 'Ordinary goal',
      graph: { nodes: [{ id: 1, description: 'Ordinary task', status: 'pending', parent_id: null, kind: 'task', flavor: null }],
        edges: [], root_ids: [1] },
      active_runs: [], event_lines: [],
    }, null);
    setSnapshot(ordinary);
    setCoordinatorGraph({
      nodes: [{ id: 9, description: 'Coordinator task', status: 'pending', parent_id: null, kind: 'task', flavor: null }],
      edges: [], root_ids: [9],
    });
    render(<DefaultLayout />);
    const canvas = screen.getByText('trigger');
    expect(JSON.parse(canvas.getAttribute('data-nodes') ?? '[]')).toEqual([expect.objectContaining({ id: 9 })]);
    act(() => setCoordinatorGraph(null));
    expect(getState().snapshot).toBe(ordinary);
    expect(JSON.parse(canvas.getAttribute('data-nodes') ?? '[]')).toEqual([expect.objectContaining({ id: 1 })]);
  });

});
