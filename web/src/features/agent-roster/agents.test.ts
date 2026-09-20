import { describe, expect, it } from 'vitest';
import type { WireActiveRun } from '../live-state/runtimeSnapshot';
import { toRosterAgents } from './agents';

function run(overrides: Partial<WireActiveRun> = {}): WireActiveRun {
  return {
    run_id: 'run-1',
    node_id: 3,
    description: 'Ship the tracer',
    status: 'running',
    ...overrides,
  };
}

describe('toRosterAgents', () => {
  it('maps a running run to a running agent', () => {
    const [agent] = toRosterAgents([run()]);

    expect(agent).toEqual({
      id: 'run-1',
      name: 'Ship the tracer',
      status: 'running',
      sub: 'node 3',
    });
  });

  it('maps a non-running status to idle', () => {
    const [agent] = toRosterAgents([run({ status: 'completed' })]);

    expect(agent.status).toBe('idle');
  });

  it('maps an empty run list to an empty roster', () => {
    expect(toRosterAgents([])).toEqual([]);
  });
});
