import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { resetStore, setSnapshot } from '../../app/store';
import { mergeSnapshot, type RawStreamSnapshot } from '../live-state/runtimeSnapshot';
import { AgentRosterSection } from './AgentRosterSection';

describe('AgentRosterSection', () => {
  beforeEach(resetStore);
  afterEach(cleanup);

  it('renders a roster row for each active run', () => {
    const raw: RawStreamSnapshot = {
      goal: 'Ship it',
      graph: null,
      active_runs: [
        { run_id: 'run-1', node_id: 1, description: 'Bake the roadmap', status: 'running' },
      ],
      event_lines: [],
    };
    setSnapshot(mergeSnapshot(raw, null));

    render(<AgentRosterSection />);
    const row = screen.getByRole('button', { name: /Bake the roadmap/ });
    expect(row.textContent).toContain('node 1');
    expect(row.textContent).toContain('RUNNING');
  });

  it('renders an empty roster with no snapshot yet', () => {
    const { container } = render(<AgentRosterSection />);
    expect(container.children.length).toBeGreaterThan(0);
  });
});
