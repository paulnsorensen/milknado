// Wire types the metaharness view reads beyond `app/wire.ts`: the live run
// session (`ActiveRunSnapshot` in `src/milknado/app/run_source.py`), node
// reviews, and `GET /api/metaharness`.
import type { WireActiveRun, WireExecutionSnapshot } from '../../app/wire';
import type { WireGoalReview } from '../goal-review/wire';
import type { WireNodeDetailSnapshot, WireSessionEvent, WireSnapshotPage } from '../../shared/node-detail/detailWire';

export interface WireSessionContext {
  family: string;
  cwd: string;
  base_oid: string;
}

export interface WireSessionView {
  context: WireSessionContext | null;
  events: WireSessionEvent[];
  actions: string[];
  active: boolean;
  permissions: WireSessionEvent[];
  owner_incarnation: string;
  invocation_id: string;
}

export interface WireLiveRun extends WireActiveRun {
  progress: string | null;
  progress_pct: number | null;
  attempt: number | null;
  max_attempts: number | null;
  stalled: boolean;
  elapsed_seconds: number;
  session?: WireSessionView;
}

export interface WireMhSnapshot extends WireExecutionSnapshot {
  active_runs?: WireLiveRun[];
  listener_errors?: string[];
  pending_goal_reviews?: WireGoalReview[];
}

export interface WireNodeReview {
  node_id: number;
  round: number;
  verdict: string;
  findings: string;
  created_at: string;
}

export interface WireDetailSession {
  run_id: string;
  session: WireSessionView | null;
  event_history: WireSnapshotPage<WireSessionEvent>;
}

export interface WireMhDetail extends Omit<WireNodeDetailSnapshot, 'sessions'> {
  reviews: WireSnapshotPage<WireNodeReview>;
  sessions: WireSnapshotPage<WireDetailSession>;
}

export interface WireMhDetailResponse {
  node_id: number;
  request_generation: number;
  detail: WireMhDetail | null;
}

export interface WireKnowledgePage {
  path: string;
  title: string;
  summary: string;
  updated_at: string;
}

export interface WireMetaharnessInfo {
  project: { name: string };
  concurrency: number | null;
  scheduling: { available: boolean; paused: boolean; reason: string | null };
  flavors: Record<string, { quality_gates: string[] }>;
  knowledge: WireKnowledgePage[];
}
