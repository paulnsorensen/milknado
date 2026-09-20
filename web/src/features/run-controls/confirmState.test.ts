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
    requestConfirm('Cancel this run?', action);

    expect(getPendingConfirm()?.prompt).toBe('Cancel this run?');

    confirmPending();

    expect(action).toHaveBeenCalledTimes(1);
    expect(getPendingConfirm()).toBeNull();
  });

  it('runs nothing on dismiss', () => {
    resetConfirm();
    const action = vi.fn();
    requestConfirm('Force stop this run?', action);

    dismissConfirm();

    expect(action).not.toHaveBeenCalled();
    expect(getPendingConfirm()).toBeNull();
  });
});
