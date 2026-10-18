// The Mikado parent tree over the snapshot graph: a node's children are
// the prerequisites that must finish before it can.
import type { WireGraphSnapshot, WireNode } from '../../../app/wire';

export interface Tree {
  byId: Map<number, WireNode>;
  kids: Map<number, WireNode[]>;
  roots: WireNode[];
}

export function buildTree(graph: WireGraphSnapshot | null | undefined): Tree {
  const nodes = graph?.nodes ?? [];
  const byId = new Map(nodes.map((node) => [node.id, node]));
  const kids = new Map<number, WireNode[]>();
  for (const node of nodes) {
    if (node.parent_id !== null && byId.has(node.parent_id)) {
      const siblings = kids.get(node.parent_id) ?? [];
      siblings.push(node);
      kids.set(node.parent_id, siblings);
    }
  }
  const rootIds = graph?.root_ids?.length
    ? graph.root_ids
    : nodes.filter((node) => node.parent_id === null || !byId.has(node.parent_id)).map((n) => n.id);
  const roots = rootIds.map((id) => byId.get(id)).filter((node): node is WireNode => !!node);
  return { byId, kids, roots };
}

export function childrenOf(tree: Tree, id: number): WireNode[] {
  return tree.kids.get(id) ?? [];
}

export function subtreeIds(tree: Tree, id: number): number[] {
  const ids: number[] = [];
  const stack = [id];
  while (stack.length > 0) {
    const current = stack.pop() as number;
    ids.push(current);
    for (const child of childrenOf(tree, current)) {
      stack.push(child.id);
    }
  }
  return ids;
}

/** The path from the tree root down to `id`, inclusive. */
export function ancestry(tree: Tree, id: number): WireNode[] {
  const path: WireNode[] = [];
  const seen = new Set<number>();
  let current = tree.byId.get(id);
  while (current && !seen.has(current.id)) {
    seen.add(current.id);
    path.unshift(current);
    current = current.parent_id === null ? undefined : tree.byId.get(current.parent_id);
  }
  return path;
}

export function rootOf(tree: Tree, id: number): number | null {
  return ancestry(tree, id)[0]?.id ?? null;
}

export function isGoal(node: WireNode): boolean {
  return node.kind !== 'task';
}

export function allDone(tree: Tree, id: number): boolean {
  return subtreeIds(tree, id).every((nodeId) => tree.byId.get(nodeId)?.status === 'done');
}

/** A pending task whose prerequisites are all done can be dispatched now. */
export function isReady(tree: Tree, node: WireNode): boolean {
  return node.status === 'pending' && childrenOf(tree, node.id).every((kid) => kid.status === 'done');
}

export function openPrerequisites(tree: Tree, id: number): number {
  return childrenOf(tree, id).filter((kid) => kid.status !== 'done').length;
}

export function title(node: WireNode): string {
  const firstLine = node.description.split('\n', 1)[0].trim();
  return firstLine || `Node ${node.id}`;
}
