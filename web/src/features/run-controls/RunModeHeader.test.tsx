import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { post } from '../../app/api';
import { resetStore, setSnapshot } from '../../app/store';
import { confirmPending, resetConfirm } from './confirmState';
import { RunModeHeader } from './RunModeHeader';

vi.mock('../../app/api', () => ({ post: vi.fn().mockResolvedValue({}) }));

function capabilities(overrides: Record<string, unknown> = {}) {
  return {
    session_input: { available: true, reason: null },
    cancel: { available: true, reason: null },
    force_stop: { available: true, reason: null },
    stop_scheduling: { available: true, reason: null },
    graph_edits: { available: true, reason: null },
    review_decision: { available: true, reason: null },
    git: { available: true, reason: null },
    owner: { available: false },
    ...overrides,
  };
}

describe('RunModeHeader', () => {
  beforeEach(() => {
    resetStore();
    resetConfirm();
    vi.mocked(post).mockClear();
  });

  afterEach(cleanup);

  it('shows the Observer badge and disables Stop scheduling with a reason', () => {
    setSnapshot({
      goal: null,
      graph: null,
      capabilities: capabilities({
        stop_scheduling: { available: false, reason: 'Stop scheduling is unavailable.' },
      }),
    });
    render(<RunModeHeader />);

    expect(screen.getByText('Observer')).toBeTruthy();
    expect(screen.getByText('Stop scheduling')).toBeDisabled();
    expect(screen.getByText('Stop scheduling is unavailable.')).toBeTruthy();
  });

  it('shows the Owner badge and posts once Confirm runs the pending request', () => {
    setSnapshot({
      goal: null,
      graph: null,
      capabilities: capabilities({ owner: { available: true, run_id: 'run-1' } }),
    });
    render(<RunModeHeader />);

    expect(screen.getByText('Owner')).toBeTruthy();

    screen.getByText('Stop scheduling').click();
    confirmPending();

    expect(post).toHaveBeenCalledWith('/api/scheduling/stop');
  });
});
