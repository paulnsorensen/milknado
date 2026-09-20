import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { getState, resetStore, setSnapshot } from '../../app/store';
import { mergeSnapshot, type RawStreamSnapshot } from '../live-state/runtimeSnapshot';
import { MINIMAP_NODE_LIMIT, MinimapFeature } from './MinimapFeature';

function wideSnapshot(nodeCount: number): RawStreamSnapshot {
  const nodes = Array.from({ length: nodeCount }, (_, index) => ({
    id: index + 1,
    description: `Node ${index + 1}`,
    status: 'pending' as const,
    parent_id: null,
    kind: 'goal' as const,
    flavor: null,
  }));
  return { goal: 'Ship it', graph: { nodes, edges: [], root_ids: nodes.map((n) => n.id) }, active_runs: [], event_lines: [] };
}

describe('MinimapFeature', () => {
  beforeEach(resetStore);
  afterEach(cleanup);

  it('caps the fed nodes at 40', () => {
    setSnapshot(mergeSnapshot(wideSnapshot(45), null));

    render(<MinimapFeature />);

    const overview = screen.getByRole('img', { name: 'Overview of the graph' });
    expect(overview.querySelectorAll('rect').length).toBe(MINIMAP_NODE_LIMIT);
  });

  it('jumping to a block sets focus on that node, and only that', () => {
    setSnapshot(mergeSnapshot(wideSnapshot(3), null));

    render(<MinimapFeature />);

    const overview = screen.getByRole('img', { name: 'Overview of the graph' });
    overview.querySelectorAll('rect')[1]?.dispatchEvent(new MouseEvent('click', { bubbles: true }));

    expect(getState().graphView).toMatchObject({ focus: 2, filter: null, collapsed: [] });
  });
});
