import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { post } from '../../app/api';
import { resetStore, setSnapshot } from '../../app/store';
import { resetConfirm, confirmPending, getPendingConfirm } from './confirmState';
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
    host_owner: { available: false, reason: null },
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
  it('shows only the Read-only badge for watch mode', () => {
    setSnapshot({
      goal: null,
      graph: null,
      capabilities: capabilities(),
    });
    render(<RunModeHeader />);

    expect(screen.getByText('Read-only')).toBeTruthy();
    expect(screen.queryByText('Stop scheduling')).toBeNull();
  });

  it('does not offer a zero-count stop prompt before run totals arrive', () => {
    setSnapshot({
      goal: null,
      graph: null,
      capabilities: capabilities({
        host_owner: { available: true, reason: null },
        owner: { available: true, run_id: 'run-1' },
      }),
    });
    render(<RunModeHeader />);

    const button = screen.getByRole('button', { name: 'Stop scheduling' });
    expect(button).toBeDisabled();
    button.click();
    expect(getPendingConfirm()).toBeNull();
  });

  it('shows the Run active badge and posts once Confirm runs the pending request', () => {
    setSnapshot({
      goal: null,
      graph: null,
      active_runs: [
        { run_id: 'run-1', node_id: 1, description: 'First', status: 'running' },
        { run_id: 'run-2', node_id: 2, description: 'Second', status: 'running' },
      ],
      capabilities: capabilities({
        host_owner: { available: true, reason: null },
        owner: { available: true, run_id: 'run-1' },
      }),
    });
    render(<RunModeHeader />);

    expect(screen.getByText('Run active')).toBeTruthy();

    screen.getByText('Stop scheduling').click();
    expect(getPendingConfirm()).toMatchObject({
      prompt: 'Stop scheduling and stop 2 active runs?',
      dismissLabel: 'Keep running',
      confirmLabel: 'Stop runs',
    });
    confirmPending();

    expect(post).toHaveBeenCalledWith('/api/scheduling/stop');
  });
});
