import { beforeEach, describe, expect, it } from 'vitest';
import { visibleGraphNodes } from './graphMode';
import { getState, resetStore, setCoordinatorGraph, setGraphView, type GraphMode } from './store';
import type { GraphNodeData, WireGraphSnapshot, WireNode } from './wire';

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

function chain(nodes: WireNode[]): WireGraphSnapshot {
  return {
    nodes,
    edges: nodes
      .filter((child) => child.parent_id !== null)
      .map((child) => ({ parent_id: child.parent_id as number, child_id: child.id })),
    root_ids: nodes.filter((child) => child.parent_id === null).map((child) => child.id),
  };
}

function nodesFor(graph: WireGraphSnapshot, mode: GraphMode): GraphNodeData[] {
  resetStore();
  setCoordinatorGraph(graph);
  setGraphView({ mode });
  return visibleGraphNodes(getState());
}

describe('graph modes', () => {
  it('execution mode hides roadmap nodes and promotes their goals to roots', () => {
    const nodes = nodesFor(GRAPH, 'execution');

    expect(nodes.map((data) => data.id)).toEqual([2, 3, 4, 5, 6, 7, 8]);
    expect(nodes.find((data) => data.id === 2)).toMatchObject({ kind: 'subgoal', parent: null });
    expect(nodes.find((data) => data.id === 8)?.extra).toEqual([4]);
  });

  it('roadmap mode keeps roadmap and goal nodes with task rollups', () => {
    const nodes = nodesFor(GRAPH, 'roadmap');

    expect(nodes.map(({ id, kind, parent, state, statusText }) => ({ id, kind, parent, state, statusText }))).toEqual([
      { id: 1, kind: 'goal', parent: null, state: 'blocked', statusText: '0/2 · 1 blocked' },
      { id: 2, kind: 'subgoal', parent: 1, state: 'blocked', statusText: '2/4 · 1 blocked' },
      { id: 3, kind: 'subgoal', parent: 1, state: 'pending', statusText: 'no tasks yet' },
      { id: 7, kind: 'subgoal', parent: 2, state: 'done', statusText: '1/1 tasks' },
    ]);
  });

  it('a pending goal with running and no blocked tasks shows as running', () => {
    const graph = chain([node(1, 'goal', null), node(2, 'task', 1, 'running')]);

    expect(nodesFor(graph, 'roadmap')).toMatchObject([
      { id: 1, state: 'running', statusText: '0/1 · 1 running' },
    ]);
  });

  it('a goal with a blocked task shows as blocked', () => {
    const graph = chain([node(1, 'goal', null), node(2, 'task', 1, 'blocked')]);

    expect(nodesFor(graph, 'roadmap')).toMatchObject([
      { id: 1, state: 'blocked', statusText: '0/1 · 1 blocked' },
    ]);
  });

  it('a goal that already left pending keeps its own state', () => {
    const graph = chain([node(1, 'goal', null, 'done'), node(2, 'task', 1, 'running')]);

    expect(nodesFor(graph, 'roadmap')).toMatchObject([
      { id: 1, state: 'done', statusText: '0/1 · 1 running' },
    ]);
  });

  it('a goal holding only goals totals the tasks beneath them', () => {
    const graph = chain([
      node(1, 'goal', null),
      node(2, 'goal', 1),
      node(3, 'task', 2, 'done'),
      node(4, 'task', 2, 'blocked'),
      node(5, 'goal', 1),
    ]);

    expect(nodesFor(graph, 'roadmap').map(({ id, state, statusText }) => ({ id, state, statusText }))).toEqual([
      { id: 1, state: 'blocked', statusText: '1/3 · 1 blocked' },
      { id: 2, state: 'blocked', statusText: '1/2 · 1 blocked' },
      { id: 5, state: 'pending', statusText: 'no tasks yet' },
    ]);
  });

  it('a goal with its own tasks also totals its child goals', () => {
    const graph = chain([
      node(1, 'goal', null),
      node(2, 'task', 1, 'done'),
      node(3, 'goal', 1),
      node(4, 'task', 3, 'running'),
    ]);

    expect(nodesFor(graph, 'roadmap')).toMatchObject([
      { id: 1, state: 'running', statusText: '1/2 · 1 running' },
      { id: 3, statusText: '0/1 · 1 running' },
    ]);
  });

  it('a goal decomposed only into stub goals counts them as pending units', () => {
    const graph = chain([node(1, 'goal', null), node(2, 'goal', 1), node(3, 'goal', 1, 'done')]);

    expect(nodesFor(graph, 'roadmap')).toMatchObject([
      { id: 1, state: 'pending', statusText: '1/2 tasks' },
      { id: 2, statusText: 'no tasks yet' },
      { id: 3, statusText: 'no tasks yet' },
    ]);
  });

  it('a task directly under a roadmap is neither a goal rollup nor a promoted root', () => {
    const graph = chain([node(1, 'roadmap', null), node(2, 'task', 1), node(3, 'goal', 1)]);

    expect(nodesFor(graph, 'execution').find((data) => data.id === 2)).toMatchObject({ kind: 'task' });
    expect(nodesFor(graph, 'execution').find((data) => data.id === 3)).toMatchObject({ kind: 'subgoal' });
    expect(nodesFor(graph, 'roadmap').find((data) => data.id === 1)?.statusText).toBe('0/1 goals');
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
