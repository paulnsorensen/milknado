import type { WireActiveRun, WireRunStatus } from '../live-state/runtimeSnapshot';

export interface RosterAgent {
  id: string;
  name: string;
  status: 'running' | 'idle';
  sub: string;
}

function toStatus(status: WireRunStatus): 'running' | 'idle' {
  return status === 'running' ? 'running' : 'idle';
}

export function toRosterAgents(runs: WireActiveRun[]): RosterAgent[] {
  return runs.map((run) => ({
    id: run.run_id,
    name: run.description,
    status: toStatus(run.status),
    sub: `node ${run.node_id}`,
  }));
}
