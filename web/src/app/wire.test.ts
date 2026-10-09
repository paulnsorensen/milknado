import { describe, expect, it } from 'vitest';
import { toGraphNodes, type WireGraphSnapshot } from './wire';

describe('toGraphNodes', () => {
  it('keeps prerequisite order and duplicates without changing the graph', () => {
    const graph: WireGraphSnapshot = {
      nodes: [
        { id: 1, description: 'root', kind: 'goal', status: 'pending', parent_id: null, flavor: null },
        { id: 2, description: 'dependent', kind: 'goal', status: 'running', parent_id: 1, flavor: null },
        { id: 3, description: 'child', kind: 'task', status: 'pending', parent_id: 2, flavor: null },
        { id: 5, description: 'first prerequisite', kind: 'task', status: 'done', parent_id: null, flavor: null },
        { id: 6, description: 'second prerequisite', kind: 'task', status: 'blocked', parent_id: null, flavor: null },
      ],
      edges: [
        { parent_id: 2, child_id: 5 },
        { parent_id: 2, child_id: 1 },
        { parent_id: 2, child_id: 6 },
        { parent_id: 2, child_id: 3 },
        { parent_id: 2, child_id: 5 },
        { parent_id: 999, child_id: 5 },
        { parent_id: 2, child_id: 999 },
      ],
      root_ids: [1, 5, 6],
    };
    const original = structuredClone(graph);
    Object.freeze(graph.nodes);
    Object.freeze(graph.edges);
    for (const node of graph.nodes) Object.freeze(node);
    for (const edge of graph.edges) Object.freeze(edge);

    expect(toGraphNodes(graph)).toEqual([
      { id: 1, title: 'root', kind: 'goal', state: 'pending', parent: null, extra: undefined },
      { id: 2, title: 'dependent', kind: 'subgoal', state: 'running', parent: 1, extra: [5, 6, 5] },
      { id: 3, title: 'child', kind: 'task', state: 'pending', parent: 2, extra: undefined },
      { id: 5, title: 'first prerequisite', kind: 'task', state: 'done', parent: null, extra: undefined },
      { id: 6, title: 'second prerequisite', kind: 'task', state: 'blocked', parent: null, extra: undefined },
    ]);
    expect(graph).toEqual(original);
  });
});
