// The graph modes. `execution` hides roadmap nodes, so the canvas shows only
// goals and the tasks that execute them. `roadmap` keeps only roadmap and goal
// nodes and folds the tasks of each goal into a progress rollup on that goal.
import { effectiveGraph, type GraphMode, type StoreState } from './store';
import {
  toGraphNodes,
  type GraphNodeData,
  type WireGraphSnapshot,
  type WireNode,
  type WireNodeStatus,
} from './wire';

interface Rollup {
  total: number;
  done: number;
  running: number;
  blocked: number;
}

function keepNodes(graph: WireGraphSnapshot, keep: (node: WireNode) => boolean): WireGraphSnapshot {
  const kept = graph.nodes.filter(keep);
  const ids = new Set(kept.map((node) => node.id));
  const nodes = kept.map((node) =>
    node.parent_id !== null && !ids.has(node.parent_id) ? { ...node, parent_id: null } : node,
  );
  return {
    nodes,
    edges: graph.edges.filter((edge) => ids.has(edge.parent_id) && ids.has(edge.child_id)),
    root_ids: nodes.filter((node) => node.parent_id === null).map((node) => node.id),
  };
}

/** The nearest non-task ancestor: the goal (or roadmap) a node rolls up to. */
function rollupOwner(node: WireNode, byId: Map<number, WireNode>): number | null {
  let owner = node.parent_id === null ? undefined : byId.get(node.parent_id);
  while (owner?.kind === 'task') {
    owner = owner.parent_id === null ? undefined : byId.get(owner.parent_id);
  }
  return owner?.id ?? null;
}

function tally(rollup: Rollup, status: WireNodeStatus): void {
  rollup.total += 1;
  if (status === 'done') rollup.done += 1;
  if (status === 'running') rollup.running += 1;
  if (status === 'blocked' || status === 'failed') rollup.blocked += 1;
}

function count(rollups: Map<number, Rollup>, ownerId: number, status: WireNodeStatus): void {
  let rollup = rollups.get(ownerId);
  if (!rollup) {
    rollup = { total: 0, done: 0, running: 0, blocked: 0 };
    rollups.set(ownerId, rollup);
  }
  tally(rollup, status);
}

function addRollups(into: Rollup, other: Rollup): void {
  into.total += other.total;
  into.done += other.done;
  into.running += other.running;
  into.blocked += other.blocked;
}

/**
 * A goal totals its own tasks plus each child goal: a decomposed child adds its
 * rollup, and a stub child with no tasks yet adds itself as one unit.
 */
function foldChildGoals(nodes: WireNode[], rollups: Map<number, Rollup>): void {
  const childGoals = new Map<number, WireNode[]>();
  for (const node of nodes) {
    if (node.kind === 'goal' && node.parent_id !== null) {
      childGoals.set(node.parent_id, [...(childGoals.get(node.parent_id) ?? []), node]);
    }
  }
  const folded = new Map<number, Rollup>();
  const fold = (id: number): Rollup | undefined => {
    if (folded.has(id)) return folded.get(id);
    const total: Rollup = { total: 0, done: 0, running: 0, blocked: 0, ...rollups.get(id) };
    for (const child of childGoals.get(id) ?? []) {
      const childRollup = fold(child.id);
      if (childRollup) addRollups(total, childRollup);
      else tally(total, child.status);
    }
    if (total.total > 0) folded.set(id, total);
    return folded.get(id);
  };
  for (const node of nodes) {
    if (node.kind === 'goal') fold(node.id);
  }
  for (const [id, rollup] of folded) rollups.set(id, rollup);
}

/** Task rollups keyed by goal id, and goal rollups keyed by roadmap id. */
function rollupsFor(nodes: WireNode[]): Map<number, Rollup> {
  const byId = new Map(nodes.map((node) => [node.id, node]));
  const rollups = new Map<number, Rollup>();
  for (const node of nodes) {
    const ownerId = node.kind === 'task' ? rollupOwner(node, byId) : null;
    if (ownerId !== null && byId.get(ownerId)?.kind === 'goal') {
      count(rollups, ownerId, node.status);
    }
  }
  foldChildGoals(nodes, rollups);
  for (const node of nodes) {
    const parent = node.parent_id === null ? undefined : byId.get(node.parent_id);
    if (node.kind === 'goal' && parent?.kind === 'roadmap') {
      count(rollups, parent.id, rollupState(node, rollups.get(node.id)));
    }
  }
  return rollups;
}

function rollupText(node: WireNode, rollup: Rollup | undefined): string {
  const unit = node.kind === 'roadmap' ? 'goals' : 'tasks';
  if (!rollup) {
    return `no ${unit} yet`;
  }
  // A compact card fits one short line, so name only the most urgent count.
  const progress = `${rollup.done}/${rollup.total}`;
  if (rollup.blocked > 0) return `${progress} · ${rollup.blocked} blocked`;
  if (rollup.running > 0) return `${progress} · ${rollup.running} running`;
  return `${progress} ${unit}`;
}

/** A pending goal shows the most urgent state among its tasks. */
function rollupState(node: WireNode, rollup: Rollup | undefined): WireNodeStatus {
  if (node.status !== 'pending' || !rollup) return node.status;
  if (rollup.blocked > 0) return 'blocked';
  if (rollup.running > 0) return 'running';
  return node.status;
}

function roadmapNodes(graph: WireGraphSnapshot): GraphNodeData[] {
  const rollups = rollupsFor(graph.nodes);
  const byId = new Map(graph.nodes.map((node) => [node.id, node]));
  return toGraphNodes(keepNodes(graph, (node) => node.kind !== 'task')).map((data) => {
    const node = byId.get(data.id as number) as WireNode;
    const rollup = rollups.get(node.id);
    return { ...data, state: rollupState(node, rollup), statusText: rollupText(node, rollup) };
  });
}

// Goals under a hidden roadmap become roots but render as `subgoal`: the
// vendored design system sizes cards by kind alone and has no layout prop, so
// `goal` would widen them and break the one-row fit of a roadmap's goals.
function executionNodes(graph: WireGraphSnapshot): GraphNodeData[] {
  const roadmapIds = new Set(
    graph.nodes.filter((node) => node.kind === 'roadmap').map((node) => node.id),
  );
  const promoted = new Set(
    graph.nodes
      .filter(
        (node) =>
          node.kind === 'goal' && node.parent_id !== null && roadmapIds.has(node.parent_id),
      )
      .map((node) => node.id),
  );
  return toGraphNodes(keepNodes(graph, (node) => !roadmapIds.has(node.id))).map((data) =>
    promoted.has(data.id as number) ? { ...data, kind: 'subgoal' } : data,
  );
}

function graphNodesFor(graph: WireGraphSnapshot, mode: GraphMode): GraphNodeData[] {
  return mode === 'roadmap' ? roadmapNodes(graph) : executionNodes(graph);
}

/** The nodes the canvas, the toolbar, and graph navigation show. */
export function visibleGraphNodes(store: StoreState): GraphNodeData[] {
  const graph = effectiveGraph(store);
  return graph ? graphNodesFor(graph, store.graphView.mode) : [];
}
