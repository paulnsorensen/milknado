import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { post } from '../../app/api';
import { resetStore, setSnapshot } from '../../app/store';
import { confirmPending, resetConfirm } from './confirmState';
import { RunControlsSidecar } from './RunControlsSidecar';

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
    owner: { available: true, run_id: 'run-1' },
    ...overrides,
  };
}

describe('RunControlsSidecar', () => {
  beforeEach(() => {
    resetStore();
    resetConfirm();
    vi.mocked(post).mockClear();
  });

  afterEach(cleanup);

  it('posts cancel once Confirm runs the pending request', () => {
    setSnapshot({ goal: null, graph: null, capabilities: capabilities() });
    render(<RunControlsSidecar />);

    screen.getByText('Cancel run').click();
    confirmPending();

    expect(post).toHaveBeenCalledWith('/api/runs/run-1/cancel');
  });

  it('disables Force stop and shows the server reason when unavailable', () => {
    setSnapshot({
      goal: null,
      graph: null,
      capabilities: capabilities({
        force_stop: { available: false, reason: 'Force stop is unavailable.' },
      }),
    });
    render(<RunControlsSidecar />);

    expect(screen.getByText('Force stop')).toBeDisabled();
    expect(screen.getByText('Force stop is unavailable.')).toBeTruthy();
  });
});
