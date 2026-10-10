import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { getState, resetStore, setCoordinatorGraph, setSelection, setSnapshot } from '../../app/store';
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

  it('searches and collapses only the active coordinator graph, then restores execution nodes', () => {
    const execution = getState().snapshot;
    setSelection(2);
    setCoordinatorGraph({
      nodes: [
        { id: 10, description: 'Coordinator goal', status: 'pending', parent_id: null, kind: 'goal', flavor: null },
        { id: 11, description: 'Coordinator task', status: 'pending', parent_id: 10, kind: 'task', flavor: null },
      ],
      edges: [{ parent_id: 10, child_id: 11 }], root_ids: [10],
    });
    render(<GraphToolbarFeature />);
    expect(screen.getByRole('button', { name: 'Focus' })).toBeDisabled();
    const input = screen.getByLabelText('Jump to node');
    fireEvent.change(input, { target: { value: 'Task' } });
    expect(screen.getByRole('option', { name: /Coordinator task/ })).toBeInTheDocument();
    expect(screen.queryByRole('option', { name: /Task one/ })).not.toBeInTheDocument();
    fireEvent.mouseDown(screen.getByRole('option', { name: /Coordinator task/ }));
    expect(getState().graphView.focus).toBe(11);
    screen.getByTitle('Collapse all groups').click();
    expect(getState().graphView.collapsed).toEqual([10]);
    setCoordinatorGraph(null);
    expect(getState().snapshot).toBe(execution);
    fireEvent.change(input, { target: { value: 'Task' } });
    expect(screen.getByRole('option', { name: /Task one/ })).toBeInTheDocument();
  });

  it('switching graph mode clears collapse and focus, and the search follows the mode', () => {
    setCoordinatorGraph({
      nodes: [
        { id: 20, description: 'Roadmap', status: 'pending', parent_id: null, kind: 'roadmap', flavor: null },
        { id: 21, description: 'Planned goal', status: 'pending', parent_id: 20, kind: 'goal', flavor: null },
        { id: 22, description: 'Goal task', status: 'pending', parent_id: 21, kind: 'task', flavor: null },
      ],
      edges: [{ parent_id: 20, child_id: 21 }, { parent_id: 21, child_id: 22 }], root_ids: [20],
    });
    render(<GraphToolbarFeature />);
    const input = screen.getByLabelText('Jump to node');
    fireEvent.change(input, { target: { value: 'Roadmap' } });
    expect(screen.queryByRole('option', { name: /Roadmap/ })).not.toBeInTheDocument();
    screen.getByTitle('Collapse all groups').click();
    expect(getState().graphView.collapsed).toEqual([21]);

    fireEvent.click(screen.getByRole('button', { name: 'Roadmap' }));

    expect(getState().graphView).toMatchObject({ mode: 'roadmap', collapsed: [], focus: null });
    expect(screen.getByRole('button', { name: 'Roadmap' })).toHaveAttribute('aria-pressed', 'true');
    fireEvent.change(input, { target: { value: 'Goal task' } });
    expect(screen.queryByRole('option', { name: /Goal task/ })).not.toBeInTheDocument();
  });

  it('picking a node style sets the level of detail', () => {
    render(<GraphToolbarFeature />);

    screen.getByRole('button', { name: 'dots' }).click();

    expect(getState().graphView.lod).toBe('dot');
  });
});
