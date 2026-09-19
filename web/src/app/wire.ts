// Wire types bound to the landed `/api/*` JSON shapes. Extend with only the
// fields a feature reads; the server may send more, TypeScript ignores them.

export type WireNodeStatus = 'pending' | 'running' | 'done' | 'blocked' | 'failed';
export type WireNodeKind = 'roadmap' | 'goal' | 'task';

export interface WireNode {
  id: number;
  description: string;
  status: WireNodeStatus;
  parent_id: number | null;
  kind: WireNodeKind;
  flavor: string | null;
}

export interface WireEdge {
  parent_id: number;
  child_id: number;
}

export interface WireGraphSnapshot {
  nodes: WireNode[];
  edges: WireEdge[];
  root_ids: number[];
}

export interface WireCapability {
  available: boolean;
  reason: string | null;
}

export interface WireOwnerCapabilities {
  available: boolean;
  reason?: string;
  run_id?: string;
  node_id?: number;
  invocation_id?: string;
  owner_incarnation?: number;
  actions?: string[];
  permission_ids?: string[];
  published_at?: string;
}

export interface WireCapabilities {
  session_input: WireCapability;
  cancel: WireCapability;
  force_stop: WireCapability;
  stop_scheduling: WireCapability;
  graph_edits: WireCapability;
  review_decision: WireCapability;
  git: WireCapability;
  owner: WireOwnerCapabilities;
}

export interface WireExecutionSnapshot {
  goal: string | null;
  graph: WireGraphSnapshot | null;
  capabilities: WireCapabilities;
}

export interface GraphNodeData {
  id: string | number;
  title: string;
  kind: 'goal' | 'subgoal' | 'task';
  state: WireNodeStatus;
  parent: string | number | null;
}

function toDesignSystemKind(node: WireNode): 'goal' | 'subgoal' | 'task' {
  if (node.kind === 'task') {
    return 'task';
  }
  return node.parent_id === null ? 'goal' : 'subgoal';
}

export function toGraphNodes(graph: WireGraphSnapshot): GraphNodeData[] {
  return graph.nodes.map((node) => ({
    id: node.id,
    title: node.description,
    kind: toDesignSystemKind(node),
    state: node.status,
    parent: node.parent_id,
  }));
}
