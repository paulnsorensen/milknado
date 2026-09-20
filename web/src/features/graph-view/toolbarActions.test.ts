import { describe, expect, it } from 'vitest';
import type { GraphNodeData } from '../../app/wire';
import { clampZoom, collapsibleIds, ZOOM_MAX, ZOOM_MIN } from './toolbarActions';

function node(overrides: Partial<GraphNodeData> & Pick<GraphNodeData, 'id'>): GraphNodeData {
  return { title: String(overrides.id), kind: 'task', state: 'pending', parent: null, ...overrides };
}

describe('clampZoom', () => {
  it('keeps an in-range zoom unchanged', () => {
    expect(clampZoom(1)).toBe(1);
  });

  it('floors at the minimum', () => {
    expect(clampZoom(ZOOM_MIN - 1)).toBe(ZOOM_MIN);
  });

  it('ceils at the maximum', () => {
    expect(clampZoom(ZOOM_MAX + 1)).toBe(ZOOM_MAX);
  });
});

describe('collapsibleIds', () => {
  it('returns only nodes that have a child', () => {
    const nodes = [
      node({ id: 1, kind: 'goal', parent: null }),
      node({ id: 2, kind: 'subgoal', parent: 1 }),
      node({ id: 3, kind: 'task', parent: 2 }),
    ];

    expect(collapsibleIds(nodes)).toEqual([1, 2]);
  });

  it('returns an empty list with no children', () => {
    expect(collapsibleIds([node({ id: 1 })])).toEqual([]);
  });
});
