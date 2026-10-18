// Places a Mikado subtree as rows of cards: the root on top, each row of
// prerequisites below its dependant, squeezed to fit the pane width.
import type { Tree } from './tree';
import { allDone, childrenOf, subtreeIds } from './tree';

export interface LayoutOptions {
  width: number;
  cardWidth: number;
  rootWidth: number;
  height: number;
  gap: number;
  pitch: number;
  fold: boolean;
  unfold: Record<number, boolean>;
}

export interface PlacedNode {
  id: number;
  left: number;
  top: number;
  w: number;
  h: number;
  /** Count of done prerequisite nodes hidden under this card, 0 when unfolded. */
  folded: number;
  foldable: number;
}

export interface PlacedEdge {
  id: number;
  d: string;
}

export interface GraphLayout {
  width: number;
  height: number;
  nodes: PlacedNode[];
  edges: PlacedEdge[];
}

interface Slot {
  x: number;
  depth: number;
  w: number;
}

function visibleKids(tree: Tree, id: number, o: LayoutOptions, fold: Map<number, [number, number]>): number[] {
  const kids = childrenOf(tree, id);
  const done = kids.filter((kid) => allDone(tree, kid.id));
  const hidden = done.reduce((sum, kid) => sum + subtreeIds(tree, kid.id).length, 0);
  const folding = o.fold && !o.unfold[id] && done.length > 0;
  fold.set(id, [folding ? hidden : 0, hidden]);
  return (folding ? kids.filter((kid) => !done.includes(kid)) : kids).map((kid) => kid.id);
}

function spreadRow(row: number[], pos: Map<number, Slot>, o: LayoutOptions): void {
  const count = row.length;
  let w = o.cardWidth;
  if (count * w + (count - 1) * o.gap > o.width) {
    w = (o.width - (count - 1) * o.gap) / count;
    row.forEach((id, i) => { (pos.get(id) as Slot).x = w / 2 + i * (w + o.gap); });
  } else {
    for (let i = 1; i < count; i++) {
      const a = pos.get(row[i - 1]) as Slot;
      const b = pos.get(row[i]) as Slot;
      b.x = Math.max(b.x, a.x + w + o.gap);
    }
    const min = (pos.get(row[0]) as Slot).x - w / 2;
    const max = (pos.get(row[count - 1]) as Slot).x + w / 2;
    let shift = max > o.width ? o.width - max : 0;
    if (min + shift < 0) {
      shift = -min;
    }
    row.forEach((id) => { (pos.get(id) as Slot).x += shift; });
  }
  row.forEach((id) => { (pos.get(id) as Slot).w = w; });
}

function place(tree: Tree, rootId: number, o: LayoutOptions, fold: Map<number, [number, number]>): Map<number, Slot> {
  const pos = new Map<number, Slot>([[rootId, { x: o.width / 2, depth: 0, w: o.rootWidth }]]);
  let row = [rootId];
  for (let depth = 1; row.length > 0; depth++) {
    const next: number[] = [];
    for (const parentId of row) {
      const kids = visibleKids(tree, parentId, o, fold);
      const parentX = (pos.get(parentId) as Slot).x;
      kids.forEach((kid, i) => {
        next.push(kid);
        pos.set(kid, { x: parentX + (i - (kids.length - 1) / 2) * (o.cardWidth + o.gap), depth, w: o.cardWidth });
      });
    }
    if (next.length > 0) {
      spreadRow(next, pos, o);
    }
    row = next;
  }
  return pos;
}

export function layoutTree(tree: Tree, rootId: number, o: LayoutOptions): GraphLayout {
  if (!tree.byId.has(rootId)) {
    return { width: o.width, height: 0, nodes: [], edges: [] };
  }
  const fold = new Map<number, [number, number]>();
  const pos = place(tree, rootId, o, fold);
  const top = (depth: number): number => 6 + depth * o.pitch;
  const nodes: PlacedNode[] = [];
  const edges: PlacedEdge[] = [];
  let maxDepth = 0;
  for (const [id, slot] of pos) {
    maxDepth = Math.max(maxDepth, slot.depth);
    const [folded, foldable] = fold.get(id) ?? [0, 0];
    nodes.push({ id, left: slot.x - slot.w / 2, top: top(slot.depth), w: slot.w, h: o.height, folded, foldable });
    const parentId = tree.byId.get(id)?.parent_id;
    const parent = parentId === null || parentId === undefined ? undefined : pos.get(parentId);
    if (parent && id !== rootId) {
      const y1 = top(parent.depth) + o.height;
      const y2 = top(slot.depth);
      const mid = (y1 + y2) / 2;
      edges.push({ id, d: `M${parent.x} ${y1} C${parent.x} ${mid},${slot.x} ${mid},${slot.x} ${y2}` });
    }
  }
  return { width: o.width, height: top(maxDepth) + o.height + 10, nodes, edges };
}
