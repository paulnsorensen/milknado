import type { ReactElement } from 'react';
import { useSyncExternalStore } from 'react';
import { getState, subscribe } from '../../app/store';
import { Milknado } from '../../design-system';
import type { StreamSnapshot } from '../live-state/runtimeSnapshot';
import { toRosterAgents } from './agents';

export function AgentRosterSection(): ReactElement {
  const state = useSyncExternalStore(subscribe, getState);
  const { AgentRoster } = Milknado;
  const snapshot = state.snapshot as StreamSnapshot | null;
  const agents = snapshot ? toRosterAgents(snapshot.active_runs) : [];

  return <AgentRoster agents={agents} title="Agents" />;
}
