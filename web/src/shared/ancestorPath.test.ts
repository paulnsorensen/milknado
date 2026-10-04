import { describe, expect, it } from 'vitest';
import { ancestorPath } from './ancestorPath';

interface Node {
  id: number;
  parent: number | null;
}

const getId = (node: Node) => node.id;
const getParentId = (node: Node) => node.parent;

describe('ancestorPath', () => {
  it('walks from the root to the node, inclusive', () => {
    const nodes: Node[] = [
      { id: 1, parent: null },
      { id: 2, parent: 1 },
      { id: 3, parent: 2 },
    ];

    expect(ancestorPath(nodes, 3, getId, getParentId)).toEqual([
      { id: 1, parent: null },
      { id: 2, parent: 1 },
      { id: 3, parent: 2 },
    ]);
  });

  it('returns a single-item path for a root node', () => {
    const nodes: Node[] = [{ id: 1, parent: null }];

    expect(ancestorPath(nodes, 1, getId, getParentId)).toEqual([
      { id: 1, parent: null },
    ]);
  });

  it('returns an empty path when the node is missing', () => {
    const nodes: Node[] = [{ id: 1, parent: null }];

    expect(ancestorPath(nodes, 99, getId, getParentId)).toEqual([]);
  });

  it('returns null when the parent chain cycles', () => {
    const nodes: Node[] = [
      { id: 1, parent: 2 },
      { id: 2, parent: 1 },
    ];

    expect(ancestorPath(nodes, 1, getId, getParentId)).toBeNull();
  });
});
