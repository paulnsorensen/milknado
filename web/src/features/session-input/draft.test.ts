import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { post } from '../../app/api';
import { resetStore, setSnapshot } from '../../app/store';
import { focusSessionInput, getDraft, queueGuidance, registerInputEl, resetDraft, setDraft } from './draft';

vi.mock('../../app/api', () => ({ post: vi.fn().mockResolvedValue({}) }));

describe('draft', () => {
  beforeEach(() => {
    resetStore();
    resetDraft();
    vi.mocked(post).mockClear();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it('stores and clears the draft text', () => {
    setDraft('Hold off on the migration');
    expect(getDraft()).toBe('Hold off on the migration');
  });

  it('focuses the registered input element', () => {
    const input = document.createElement('input');
    const focusSpy = vi.spyOn(input, 'focus');
    registerInputEl(input);

    focusSessionInput();

    expect(focusSpy).toHaveBeenCalledTimes(1);
  });

  it('does nothing without a registered input element', () => {
    expect(() => focusSessionInput()).not.toThrow();
  });

  it('sends the draft as a follow-up and clears it', async () => {
    setSnapshot({
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
        owner: { available: true, run_id: 'run-1' },
      },
    });
    setDraft('Keep going');

    await queueGuidance();

    expect(getDraft()).toBe('');
    expect(post).toHaveBeenCalledWith(
      '/api/runs/run-1/session-input',
      expect.objectContaining({ action: 'follow_up', text: 'Keep going' }),
    );
  });

  it('does nothing with an empty draft', () => {
    queueGuidance();
    expect(post).not.toHaveBeenCalled();
  });
});
