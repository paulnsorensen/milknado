import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { post } from '../../app/api';
import { resetStore, setSnapshot } from '../../app/store';
import { sendSessionCommand } from './sessionCommand';

vi.mock('../../app/api', () => ({ post: vi.fn().mockResolvedValue({}) }));

function snapshotWithOwner(overrides: Record<string, unknown> = {}) {
  return {
    goal: null,
    graph: null,
    capabilities: {
      session_input: { available: true, reason: null },
      cancel: { available: true, reason: null },
      force_stop: { available: true, reason: null },
      stop_scheduling: { available: true, reason: null },
      graph_edits: { available: true, reason: null },
      review_decision: { available: true, reason: null },
      git: { available: true, reason: null },
      owner: { available: true, run_id: 'run-1', owner_incarnation: 3, invocation_id: 'inv-1', permission_ids: [], ...overrides },
    },
  };
}

describe('sendSessionCommand', () => {
  beforeEach(() => {
    resetStore();
    vi.mocked(post).mockClear();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it('posts the action with a fresh command id and the string owner incarnation', async () => {
    setSnapshot(snapshotWithOwner());

    await sendSessionCommand('steer', { text: 'Please slow down' });

    expect(post).toHaveBeenCalledTimes(1);
    const [path, body] = vi.mocked(post).mock.calls[0] as [string, Record<string, unknown>];
    expect(path).toBe('/api/runs/run-1/session-input');
    expect(body).toMatchObject({
      action: 'steer',
      text: 'Please slow down',
      request_id: '',
      owner_incarnation: '3',
      invocation_id: 'inv-1',
    });
    expect(typeof body.command_id).toBe('string');
    expect((body.command_id as string).length).toBeGreaterThan(0);
  });

  it('mints a different command id per call', async () => {
    setSnapshot(snapshotWithOwner());

    await sendSessionCommand('approve', { requestId: 'perm-1' });
    await sendSessionCommand('approve', { requestId: 'perm-1' });

    const [firstBody, secondBody] = vi.mocked(post).mock.calls.map((call) => call[1] as Record<string, unknown>);
    expect(firstBody.command_id).not.toBe(secondBody.command_id);
  });

  it('does nothing without an owned run', async () => {
    await sendSessionCommand('steer', { text: 'x' });
    expect(post).not.toHaveBeenCalled();
  });
});
