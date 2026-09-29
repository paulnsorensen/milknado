import type { WireActiveRun } from '../live-state/runtimeSnapshot';

export interface RosterAgent {
  id: string;
  name: string;
  status: 'running' | 'idle';
  sub: string;
  figure: string;
}


export function toRosterAgents(runs: WireActiveRun[]): RosterAgent[] {
  return runs.map((run) => ({
    id: run.run_id,
    name: run.description,
    status: run.status === 'running' ? 'running' : 'idle',
    sub: `node ${run.node_id}`,
    figure: run.status.toUpperCase(),
  }));
}
