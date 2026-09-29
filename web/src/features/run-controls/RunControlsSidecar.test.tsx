import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { post } from '../../app/api';
import { clearActions } from '../../app/actions';
import { clearSlots } from '../../app/slots';
import { resetStore, setSelection, setSnapshot } from '../../app/store';
import { confirmPending, resetConfirm } from './confirmState';
import { register } from './index';
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
    host_owner: { available: true, reason: null },
    owner: { available: true, run_id: 'run-1', node_id: 1 },
    ...overrides,
  };
}

describe('RunControlsSidecar', () => {
  beforeEach(() => {
    resetStore();
    resetConfirm();
    clearActions();
    register();
    vi.mocked(post).mockClear();
  });

  afterEach(() => {
    cleanup();
    clearSlots();
  });
  it('renders no controls without the selected owner run', () => {
    setSnapshot({ goal: null, graph: null, capabilities: capabilities() });

    render(<RunControlsSidecar />);

    expect(screen.queryByRole('button', { name: 'Cancel run' })).toBeNull();
    expect(screen.queryByRole('button', { name: 'Force stop' })).toBeNull();
  });

  it('posts cancel once Confirm runs the pending request', () => {
    setSnapshot({ goal: null, graph: null, capabilities: capabilities() });
    setSelection(1);
    render(<RunControlsSidecar />);

    screen.getByText('Cancel run').click();
    confirmPending();

    expect(post).toHaveBeenCalledWith('/api/runs/run-1/cancel');
  });

  it('hides Cancel run and Force stop for a watch host with the owner node selected', () => {
    setSnapshot({
      goal: null,
      graph: null,
      capabilities: capabilities({ host_owner: { available: false, reason: null } }),
    });
    setSelection(1);

    render(<RunControlsSidecar />);

    expect(screen.queryByRole('button', { name: 'Cancel run' })).toBeNull();
    expect(screen.queryByRole('button', { name: 'Force stop' })).toBeNull();
  });

  it('disables Force stop and shows the server reason when unavailable', () => {
    setSnapshot({
      goal: null,
      graph: null,
      capabilities: capabilities({
        force_stop: { available: false, reason: 'Force stop is unavailable.' },
      }),
    });
    setSelection(1);
    render(<RunControlsSidecar />);

    expect(screen.getByText('Force stop')).toBeDisabled();
    expect(screen.getByText('Force stop is unavailable.')).toBeTruthy();
  });
});
