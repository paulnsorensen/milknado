import type { ReactElement } from 'react';
import { useSyncExternalStore } from 'react';
import { getState, subscribe } from '../../app/store';
import { Milknado } from '../../design-system';
import type { StreamSnapshot } from '../live-state/runtimeSnapshot';
import { toRosterAgents } from './agents';

export function AgentRosterSection(): ReactElement {
  const state = useSyncExternalStore(subscribe, getState);
  const { AgentRoster } = Milknado;
  const snapshot = state.snapshot as Partial<StreamSnapshot> | null;
  const agents = toRosterAgents(snapshot?.active_runs ?? []);
  const running = agents.filter((agent) => agent.status === 'running').length;

  return (
    <AgentRoster
      agents={agents}
      title="Agents"
      count={`${running}/${agents.length} running`}
      className="mk-rail-section mk-agent-roster"
    />
  );
}