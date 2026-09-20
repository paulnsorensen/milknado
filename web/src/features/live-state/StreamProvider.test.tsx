import { cleanup, render } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { get } from '../../app/api';
import { getState, resetStore } from '../../app/store';
import { resetConnectionStatus, getConnectionStatus } from './connection';
import { StreamProvider } from './StreamProvider';

vi.mock('../../app/api', () => ({ get: vi.fn().mockResolvedValue(null) }));

type Handler = (event: MessageEvent<string>) => void;

class FakeEventSource {
  static instances: FakeEventSource[] = [];
  private listeners = new Map<string, Set<Handler>>();
  closed = false;

  constructor(public url: string) {
    FakeEventSource.instances.push(this);
  }

  addEventListener(type: string, handler: Handler): void {
    const set = this.listeners.get(type) ?? new Set();
    set.add(handler);
    this.listeners.set(type, set);
  }

  removeEventListener(type: string, handler: Handler): void {
    this.listeners.get(type)?.delete(handler);
  }

  close(): void {
    this.closed = true;
  }

  emit(type: string, data?: string): void {
    for (const handler of this.listeners.get(type) ?? []) {
      handler({ data } as MessageEvent<string>);
    }
  }
}

describe('StreamProvider', () => {
  beforeEach(() => {
    resetStore();
    resetConnectionStatus();
    FakeEventSource.instances = [];
    vi.stubGlobal('EventSource', FakeEventSource);
  });

  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
    vi.clearAllMocks();
  });

  it('opens one EventSource on /api/stream', () => {
    render(<StreamProvider />);

    expect(FakeEventSource.instances).toHaveLength(1);
    expect(FakeEventSource.instances[0].url).toBe('/api/stream');
  });

  it('feeds a snapshot event into the store', () => {
    render(<StreamProvider />);
    const raw = { goal: 'Ship it', graph: null, active_runs: [], event_lines: ['hi'] };

    FakeEventSource.instances[0].emit('snapshot', JSON.stringify(raw));

    expect(getState().snapshot?.goal).toBe('Ship it');
    expect(getConnectionStatus()).toBe('connected');
  });

  it('marks reconnecting and probes the snapshot endpoint on error', () => {
    render(<StreamProvider />);

    FakeEventSource.instances[0].emit('error');

    expect(getConnectionStatus()).toBe('reconnecting');
    expect(get).toHaveBeenCalledWith('/api/snapshot');
  });

  it('closes the EventSource on unmount', () => {
    const { unmount } = render(<StreamProvider />);
    const instance = FakeEventSource.instances[0];

    unmount();

    expect(instance.closed).toBe(true);
  });
});
