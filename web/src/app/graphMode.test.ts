import { beforeEach, describe, expect, it } from 'vitest';
import { graphNodesFor, visibleGraphNodes } from './graphMode';
import { getState, resetStore, setCoordinatorGraph, setGraphView } from './store';
import type { WireGraphSnapshot, WireNode } from './wire';

function node(id: number, kind: WireNode['kind'], parent_id: number | null, status: WireNode['status'] = 'pending'): WireNode {
  return { id, description: `node ${id}`, status, parent_id, kind, flavor: null };
}

// roadmap 1 → goals 2 (decomposed) and 3 (stub); goal 2 → tasks 4, 5;
// task 5 → nested task 6. Goal 7 sits under goal 2. Task 8 depends on task 4.
const GRAPH: WireGraphSnapshot = {
  nodes: [
    node(1, 'roadmap', null),
    node(2, 'goal', 1),
    node(3, 'goal', 1),
    node(4, 'task', 2, 'done'),
    node(5, 'task', 2, 'running'),
    node(6, 'task', 5, 'failed'),
    node(7, 'goal', 2, 'done'),
    node(8, 'task', 7, 'done'),
  ],
  edges: [
    { parent_id: 1, child_id: 2 },
    { parent_id: 1, child_id: 3 },
    { parent_id: 2, child_id: 4 },
    { parent_id: 2, child_id: 5 },
    { parent_id: 5, child_id: 6 },
    { parent_id: 2, child_id: 7 },
    { parent_id: 7, child_id: 8 },
    { parent_id: 8, child_id: 4 },
  ],
  root_ids: [1],
};

describe('graphNodesFor', () => {
  it('execution mode hides roadmap nodes and promotes their goals to roots', () => {
    const nodes = graphNodesFor(GRAPH, 'execution');

    expect(nodes.map((data) => data.id)).toEqual([2, 3, 4, 5, 6, 7, 8]);
    expect(nodes.find((data) => data.id === 2)).toMatchObject({ kind: 'subgoal', parent: null });
    expect(nodes.find((data) => data.id === 8)?.extra).toEqual([4]);
  });

  it('roadmap mode keeps roadmap and goal nodes with task rollups', () => {
    const nodes = graphNodesFor(GRAPH, 'roadmap');

    expect(nodes.map(({ id, kind, parent, state, statusText }) => ({ id, kind, parent, state, statusText }))).toEqual([
      { id: 1, kind: 'goal', parent: null, state: 'pending', statusText: '0/2 goals' },
      { id: 2, kind: 'subgoal', parent: 1, state: 'blocked', statusText: '1/3 · 1 blocked' },
      { id: 3, kind: 'subgoal', parent: 1, state: 'pending', statusText: 'no tasks yet' },
      { id: 7, kind: 'subgoal', parent: 2, state: 'done', statusText: '1/1 tasks' },
    ]);
  });

  it('a pending goal with running and no blocked tasks shows as running', () => {
    const graph: WireGraphSnapshot = {
      nodes: [node(1, 'goal', null), node(2, 'task', 1, 'running')],
      edges: [{ parent_id: 1, child_id: 2 }],
      root_ids: [1],
    };

    expect(graphNodesFor(graph, 'roadmap')).toMatchObject([
      { id: 1, state: 'running', statusText: '0/1 · 1 running' },
    ]);
  });
});

describe('visibleGraphNodes', () => {
  beforeEach(resetStore);

  it('projects the effective graph for the store graph mode', () => {
    expect(visibleGraphNodes(getState())).toEqual([]);

    setCoordinatorGraph(GRAPH);
    setGraphView({ mode: 'roadmap' });

    expect(visibleGraphNodes(getState()).map((data) => data.id)).toEqual([1, 2, 3, 7]);
  });
});
