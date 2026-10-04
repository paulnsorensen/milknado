import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { clearActions, dispatchAction } from '../../app/actions';
import { clearSlots } from '../../app/slots';
import { getState, resetStore, setSnapshot } from '../../app/store';
import { getPendingConfirm, resetConfirm } from './confirmState';
import { register } from './index';

function capabilities() {
  return {
    session_input: { available: true, reason: null },
    cancel: { available: true, reason: null },
    force_stop: { available: true, reason: null },
    stop_scheduling: { available: true, reason: null },
    graph_edits: { available: true, reason: null },
    review_decision: { available: true, reason: null },
    git: { available: true, reason: null },
    host_owner: { available: true, reason: null },
    owner: { available: true, run_id: 'run-1' },
  };
}

describe('scheduling.stop action', () => {
  beforeEach(() => {
    resetStore();
    resetConfirm();
    clearActions();
    register();
  });

  afterEach(clearSlots);

  it('pluralizes the confirm prompt for a single active run', () => {
    setSnapshot({
      goal: null,
      graph: null,
      capabilities: capabilities(),
      active_runs: [{ run_id: 'run-1', node_id: 1, description: 'work', status: 'running' }],
    });

    dispatchAction('scheduling.stop');

    expect(getPendingConfirm()).toMatchObject({
      prompt: 'Stop scheduling and stop 1 active run?',
    });
  });

  it('pushes a notice instead of a confirm prompt when run totals are unavailable', () => {
    setSnapshot({ goal: null, graph: null, capabilities: capabilities() });

    dispatchAction('scheduling.stop');

    expect(getPendingConfirm()).toBeNull();
    expect(getState().notices.map((notice) => notice.reason)).toContain(
      'Run totals are not available yet.',
    );
  });
});
