import type { WireGraphSnapshot, WireNodeKind, WireNodeStatus } from '../../app/wire';

export interface Node {
  id: number;
  description: string;
  status: WireNodeStatus;
  parent_id: number | null;
  kind?: WireNodeKind;
  flavor?: string | null;
  artifact_path?: string | null;
}
export interface Record {
  seq: number;
  kind: string;
  text: string;
  entity_kind: string;
  entity_id: string;
  status: string;
  tool_name?: string;
  duration_ms?: number | null;
}
export interface Review {
  review_id: number;
  goal_id: number;
  decision: string;
  evidence: string;
  proposed_change: string;
}
export interface Run {
  run_id: string;
  node_id: number;
  status: string;
  detail?: string | null;
  error?: string | null;
  verification_status?: 'accepted' | 'rejected' | null;
  verified_at?: string | null;
}
export interface PlanProposal {
  id: string;
  status: string;
  manifest: {
    goal_summary: string;
    changes: Array<{ id: string; path: string; description: string; depends_on?: string[] }>;
    new_relationships?: Array<{
      source_change_id: string;
      dependant_change_id: string;
      reason: string;
    }>;
  };
}
export interface Snapshot {
  session: { id: string; goal_id: number; provider: string };
  goal: Node;
  nodes: Node[];
  edges: Array<{ parent_id: number; child_id: number }>;
  runs: Run[];
  reviews: Review[];
  proposals?: PlanProposal[];
  recovery: Record[];
  provider_turns: Array<{ provider_session_id: string; status: string }>;
  provider_bindings: Array<{ provider_session_id: string; scope_kind: string; scope_id: string }>;
  capability_floor: { [key: string]: string };
  native_actions: string[];
  unsupported_actions: string[];
  events: Record[];
  cursor: number;
}
export interface SessionSummary {
  id: string;
  goal_id: number;
  provider: string;
  created_at: string;
  description: string;
}
export interface Receipt {
  status: string;
  result: { id?: string } | string | null;
}

export function graphFrom(snapshot: Snapshot): WireGraphSnapshot {
  return {
    nodes: snapshot.nodes.map((node) => ({
      ...node, kind: node.kind ?? 'task', flavor: node.flavor ?? null,
    })),
    edges: snapshot.edges,
    root_ids: [snapshot.goal.id],
  };
}
