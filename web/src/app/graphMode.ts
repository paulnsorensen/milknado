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

function count(rollups: Map<number, Rollup>, ownerId: number, status: WireNodeStatus): void {
  let rollup = rollups.get(ownerId);
  if (!rollup) {
    rollup = { total: 0, done: 0, running: 0, blocked: 0 };
    rollups.set(ownerId, rollup);
  }
  rollup.total += 1;
  if (status === 'done') rollup.done += 1;
  if (status === 'running') rollup.running += 1;
  if (status === 'blocked' || status === 'failed') rollup.blocked += 1;
}

/** Task rollups keyed by goal id, and goal rollups keyed by roadmap id. */
function rollupsFor(nodes: WireNode[]): Map<number, Rollup> {
  const byId = new Map(nodes.map((node) => [node.id, node]));
  const rollups = new Map<number, Rollup>();
  for (const node of nodes) {
    const ownerId = rollupOwner(node, byId);
    const owner = ownerId === null ? undefined : byId.get(ownerId);
    if (owner && (node.kind === 'task' || owner.kind === 'roadmap')) {
      count(rollups, owner.id, node.status);
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

// Goals under a hidden roadmap become roots but keep the sibling card width,
// so the goals of one roadmap still fit one canvas row.
function executionNodes(graph: WireGraphSnapshot): GraphNodeData[] {
  const roadmapIds = new Set(
    graph.nodes.filter((node) => node.kind === 'roadmap').map((node) => node.id),
  );
  const promoted = new Set(
    graph.nodes
      .filter((node) => node.parent_id !== null && roadmapIds.has(node.parent_id))
      .map((node) => node.id),
  );
  return toGraphNodes(keepNodes(graph, (node) => !roadmapIds.has(node.id))).map((data) =>
    promoted.has(data.id as number) ? { ...data, kind: 'subgoal' } : data,
  );
}

export function graphNodesFor(graph: WireGraphSnapshot, mode: GraphMode): GraphNodeData[] {
  return mode === 'roadmap' ? roadmapNodes(graph) : executionNodes(graph);
}

/** The nodes the canvas, the toolbar, and graph navigation show. */
export function visibleGraphNodes(store: StoreState): GraphNodeData[] {
  const graph = effectiveGraph(store);
  return graph ? graphNodesFor(graph, store.graphView.mode) : [];
}
