// Wire types for GET /api/nodes/{id}, bound to `NodeDetailResponse` in
// `src/milknado/domains/graph/snapshot_models.py`. Only the fields this
// feature reads are declared; the server may send more.
import type { WireNodeKind, WireNodeStatus } from '../../app/wire';

export type DetailPageState = 'loaded' | 'missing' | 'not_loaded' | 'not_stored';

export interface WireSnapshotPage<T> {
  items: T[] | null;
  offset: number;
  limit: number;
  total: number | null;
  has_more: boolean;
  state: DetailPageState;
}

export interface WireDetailNode {
  id: number;
  description: string;
  status: WireNodeStatus;
  parent_id: number | null;
  kind: WireNodeKind;
  flavor: string | null;
}

export interface WireRunRecord {
  run_id: string;
  node_id: number;
  status: string;
  started_at: string;
  ended_at: string | null;
  error: string | null;
}

export type WireSessionEventKind = 'assistant' | 'tool' | 'user' | 'error' | 'status' | 'permission';

export interface WireSessionEvent {
  kind: WireSessionEventKind;
  text: string;
  event_id: string;
  state: string;
  action: string | null;
}

export interface WireNodeSessionSnapshot {
  run_id: string;
  state: DetailPageState;
  event_history: WireSnapshotPage<WireSessionEvent>;
}

export interface WireNodeDetailSnapshot {
  node: WireDetailNode;
  description: string;
  parent: WireDetailNode | null;
  ancestors: WireSnapshotPage<WireDetailNode>;
  prerequisite_ids: WireSnapshotPage<number>;
  dependent_ids: WireSnapshotPage<number>;
  owned_files: WireSnapshotPage<string>;
  runs: WireSnapshotPage<WireRunRecord>;
  sessions: WireSnapshotPage<WireNodeSessionSnapshot>;
}

export interface WireNodeDetailResponse {
  node_id: number;
  request_generation: number;
  detail: WireNodeDetailSnapshot | null;
}
