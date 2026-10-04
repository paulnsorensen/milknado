/**
 * Walks a node's parent chain from itself up to the root, guarding against
 * cycles. Generic over any node shape that exposes an id and a parent id via
 * accessors, so callers can reuse it regardless of field naming.
 *
 * Returns the path ordered from the root to the node (inclusive), or `null`
 * if a cycle is detected before the walk completes.
 */
export function ancestorPath<T, Id>(
  nodes: T[],
  nodeId: Id,
  getId: (node: T) => Id,
  getParentId: (node: T) => Id | null,
): T[] | null {
  const byId = new Map<Id, T>(nodes.map((node) => [getId(node), node]));
  const path: T[] = [];
  const visited = new Set<Id>();
  let node = byId.get(nodeId);
  while (node !== undefined) {
    const id = getId(node);
    if (visited.has(id)) {
      return null;
    }
    path.unshift(node);
    visited.add(id);
    const parentId = getParentId(node);
    node = parentId === null ? undefined : byId.get(parentId);
  }
  return path;
}
