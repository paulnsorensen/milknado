import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { getState, resetStore, setSelection, setSnapshot } from '../../app/store';
import { mergeSnapshot, type RawStreamSnapshot } from '../live-state/runtimeSnapshot';
import { GraphToolbarFeature } from './GraphToolbarFeature';

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

describe('GraphToolbarFeature', () => {
  beforeEach(() => {
    resetStore();
    setSnapshot(mergeSnapshot(snapshotWithNodes(), null));
  });

  afterEach(cleanup);

  it('writes the filter into the graph view state, and nothing else', () => {
    render(<GraphToolbarFeature />);

    screen.getByRole('button', { name: 'Ready' }).click();

    expect(getState().graphView.filter).toBe('ready');
  });

  it('jumping to a search result sets focus on that node', () => {
    render(<GraphToolbarFeature />);

    const input = screen.getByLabelText('Jump to node');
    fireEvent.change(input, { target: { value: 'Task' } });
    fireEvent.mouseDown(screen.getByRole('option', { name: /Task one/ }));

    expect(getState().graphView.focus).toBe(2);
  });

  it('toggling focus uses the current selection', () => {
    setSelection(2);
    render(<GraphToolbarFeature />);

    screen.getByRole('button', { name: 'Focus' }).click();

    expect(getState().graphView.focus).toBe(2);
  });

  it('collapse all fills collapsed with every node that has a child', () => {
    render(<GraphToolbarFeature />);

    screen.getByTitle('Collapse all groups').click();

    expect(getState().graphView.collapsed).toEqual([1]);
  });

  it('picking a node style sets the level of detail', () => {
    render(<GraphToolbarFeature />);

    screen.getByRole('button', { name: 'dots' }).click();

    expect(getState().graphView.lod).toBe('dot');
  });
});
