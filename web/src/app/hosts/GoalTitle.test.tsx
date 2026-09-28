import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import type { StreamSnapshot } from '../../features/live-state/runtimeSnapshot';
import { resetStore, setSnapshot } from '../store';
import type { WireExecutionSnapshot } from '../wire';
import { GoalTitle } from './GoalTitle';

const CAPABILITIES = {
  session_input: { available: false, reason: null },
  cancel: { available: false, reason: null },
  force_stop: { available: false, reason: null },
  stop_scheduling: { available: false, reason: null },
  graph_edits: { available: false, reason: null },
  review_decision: { available: false, reason: null },
  git: { available: false, reason: null },
  owner: { available: false },
};

function setGraphSnapshot(): void {
  const snapshot = {
    goal: 'Parked roadmap',
    capabilities: CAPABILITIES,
    graph: {
      nodes: [
        { id: 43, description: 'Parked roadmap', status: 'pending', parent_id: null, kind: 'roadmap', flavor: null },
        { id: 87, description: 'Running goal', status: 'running', parent_id: null, kind: 'goal', flavor: null },
        { id: 88, description: 'Running task', status: 'running', parent_id: 87, kind: 'task', flavor: null },
      ],
      edges: [{ parent_id: 87, child_id: 88 }],
      root_ids: [43, 87],
    },
    active_runs: [{ run_id: 'run-1', node_id: 88, description: 'Running task', status: 'running' }],
  } as unknown as WireExecutionSnapshot & Partial<StreamSnapshot>;
  setSnapshot(snapshot);
}

describe('GoalTitle', () => {
  beforeEach(resetStore);
  afterEach(cleanup);

  it('titles the header from the root containing the active run', () => {
    setGraphSnapshot();

    render(<GoalTitle />);

    expect(screen.getByRole('heading', { level: 1 }).textContent).toBe('Running goal');
    expect(screen.getByRole('combobox', { name: 'Root goal' })).toBeTruthy();
  });

  it('switches between visible roots without changing the active run', () => {
    setGraphSnapshot();
    render(<GoalTitle />);

    fireEvent.change(screen.getByRole('combobox', { name: 'Root goal' }), { target: { value: '43' } });

    expect(screen.getByRole('heading', { level: 1 }).textContent).toBe('Parked roadmap');
  });
});
