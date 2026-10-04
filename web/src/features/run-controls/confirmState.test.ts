import { describe, expect, it, vi } from 'vitest';
import {
  confirmPending,
  dismissConfirm,
  getPendingConfirm,
  requestConfirm,
  resetConfirm,
} from './confirmState';

describe('confirmState', () => {
  it('has no pending request by default', () => {
    resetConfirm();
    expect(getPendingConfirm()).toBeNull();
  });

  it('runs the action once on confirm and clears the pending request', () => {
    resetConfirm();
    const action = vi.fn();
    requestConfirm({
      prompt: 'Force stop the run?',
      body: 'The run stops now.',
      dismissLabel: 'Keep the run',
      confirmLabel: 'Force stop',
      action,
    });

    expect(getPendingConfirm()).toMatchObject({
      prompt: 'Force stop the run?',
      body: 'The run stops now.',
      dismissLabel: 'Keep the run',
      confirmLabel: 'Force stop',
    });

    confirmPending();

    expect(action).toHaveBeenCalledTimes(1);
    expect(getPendingConfirm()).toBeNull();
  });

  it('uses safe defaults when a caller only supplies prompt and action', () => {
    resetConfirm();
    requestConfirm({ prompt: 'Cancel this run?', action: vi.fn() });

    expect(getPendingConfirm()).toMatchObject({
      body: 'The action runs once. It cannot be undone from here.',
      dismissLabel: 'Dismiss',
      confirmLabel: 'Confirm',
    });
  });

  it('runs nothing on dismiss', () => {
    resetConfirm();
    const action = vi.fn();
    requestConfirm({ prompt: 'Force stop the run?', action });
    dismissConfirm();

    expect(action).not.toHaveBeenCalled();
    expect(getPendingConfirm()).toBeNull();
  });
});
