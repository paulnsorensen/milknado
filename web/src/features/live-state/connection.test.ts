import { beforeEach, describe, expect, it, vi } from 'vitest';
import {
  getConnectionStatus,
  resetConnectionStatus,
  setConnectionStatus,
  subscribeConnection,
} from './connection';

describe('connection status', () => {
  beforeEach(resetConnectionStatus);

  it('starts connected', () => {
    expect(getConnectionStatus()).toBe('connected');
  });

  it('notifies subscribers on a change', () => {
    const listener = vi.fn();
    const unsubscribe = subscribeConnection(listener);

    setConnectionStatus('reconnecting');

    expect(getConnectionStatus()).toBe('reconnecting');
    expect(listener).toHaveBeenCalledTimes(1);
    unsubscribe();
  });

  it('does not notify when the status is unchanged', () => {
    const listener = vi.fn();
    subscribeConnection(listener);

    setConnectionStatus('connected');

    expect(listener).not.toHaveBeenCalled();
  });
});
