// Pure helpers for the graph-view toolbar: zoom clamping and the ids that
// "collapse all" / "expand all" act on (every node with a child).
import type { GraphNodeData } from '../../app/wire';

export const ZOOM_STEP = 0.1;
export const ZOOM_MIN = 0.1;
export const ZOOM_MAX = 2;

export function clampZoom(zoom: number): number {
  return Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, zoom));
}

/** The ids of nodes with at least one child — the collapsible groups. */
export function collapsibleIds(nodes: GraphNodeData[]): Array<string | number> {
  const parentIds = new Set(
    nodes.map((node) => node.parent).filter((parent): parent is string | number => parent != null),
  );
  return nodes.filter((node) => parentIds.has(node.id)).map((node) => node.id);
}
