// Wire types for fields the `/api/stream` payload carries that
// `app/wire.ts` does not type: the stream sends a raw ExecutionSnapshot
// with no `capabilities`. `mergeSnapshot` restores the field from the
// store's last-known value so `setSnapshot` always sees a full snapshot.
import type { WireCapabilities, WireExecutionSnapshot } from '../../app/wire';

export type WireRunStatus = 'running' | 'completed' | 'failed' | 'stopped';

export interface WireActiveRun {
  run_id: string;
  node_id: number;
  description: string;
  status: WireRunStatus;
}

export interface RawStreamSnapshot extends Omit<WireExecutionSnapshot, 'capabilities'> {
  active_runs: WireActiveRun[];
  event_lines: string[];
  capabilities?: WireCapabilities;
}

export interface StreamSnapshot extends WireExecutionSnapshot {
  active_runs: WireActiveRun[];
  event_lines: string[];
}

const EMPTY_CAPABILITY = { available: false, reason: null };

export const DEFAULT_CAPABILITIES: WireCapabilities = {
  session_input: EMPTY_CAPABILITY,
  cancel: EMPTY_CAPABILITY,
  force_stop: EMPTY_CAPABILITY,
  stop_scheduling: EMPTY_CAPABILITY,
  graph_edits: EMPTY_CAPABILITY,
  review_decision: EMPTY_CAPABILITY,
  git: EMPTY_CAPABILITY,
  owner: { available: false },
};

export function mergeSnapshot(
  raw: RawStreamSnapshot,
  previousCapabilities: WireCapabilities | null,
): StreamSnapshot {
  return {
    ...raw,
    capabilities: raw.capabilities ?? previousCapabilities ?? DEFAULT_CAPABILITIES,
  };
}
