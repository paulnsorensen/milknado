import { act, cleanup, fireEvent, render, renderHook, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { getState, resetStore } from '../../app/store';
import { useCoordinatorSession } from '../coordinator/useCoordinatorSession';
import { NarrowLayout } from './NarrowLayout';

const response = (body: unknown) => Promise.resolve({ ok: true, status: 200, json: async () => body } as Response);

const snapshot = {
  session: { id: 'session-1', goal_id: 1, provider: 'claude' },
  goal: { id: 1, description: 'Narrow goal', status: 'pending', parent_id: null },
  nodes: [{ id: 1, description: 'Narrow goal', status: 'pending', parent_id: null }],
  edges: [], runs: [], reviews: [], recovery: [], provider_turns: [], provider_bindings: [],
  capability_floor: {}, native_actions: [], unsupported_actions: [], events: [], cursor: 0,
  proposals: [{ id: 'plan-1', status: 'pending', manifest: {
    goal_summary: 'Narrow plan', changes: [{ id: 'change-1', path: 'src/goal.py', description: 'Apply goal' }],
  } }],
};

describe('narrow coordinator', () => {
  beforeEach(() => { resetStore(); localStorage.clear(); });
  afterEach(() => { cleanup(); vi.unstubAllGlobals(); localStorage.clear(); });

  it('mounts one cockpit for narrow intake and proposal approval', async () => {
    vi.stubGlobal('matchMedia', vi.fn().mockReturnValue({ matches: true, addEventListener: vi.fn(), removeEventListener: vi.fn() }));
    vi.stubGlobal('fetch', vi.fn((input: RequestInfo | URL, options?: RequestInit) => {
      const url = String(input);
      if (url === '/api/coordinators/commands' && options?.method === 'POST') return response({ status: 'accepted', result: { id: 'session-1' } });
      if (url === '/api/coordinators') return response([]);
      if (url.endsWith('/commands')) return response({ status: 'accepted', result: {} });
      if (url.endsWith('/snapshot')) return response(snapshot);
      return response({});
    }));
    render(<NarrowLayout />);
    expect(await screen.findAllByRole('complementary', { name: 'Coordinator cockpit' })).toHaveLength(1);
    fireEvent.change(screen.getByLabelText('Goal'), { target: { value: 'Narrow goal' } });
    fireEvent.click(screen.getByRole('button', { name: 'Start goal' }));
    expect(await screen.findByRole('button', { name: 'Approve plan-1' })).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Approve plan-1' }));
    await waitFor(() => expect(vi.mocked(fetch).mock.calls.some(([url, options]) =>
      String(url).endsWith('/commands') && String(options?.body).includes('"proposal_id":"plan-1"'))).toBe(true));
    expect(screen.getAllByRole('complementary', { name: 'Coordinator cockpit' })).toHaveLength(1);
  });
  it('keeps one cockpit in the wide layout', async () => {
    vi.stubGlobal('matchMedia', vi.fn().mockReturnValue({ matches: false, addEventListener: vi.fn(), removeEventListener: vi.fn() }));
    vi.stubGlobal('fetch', vi.fn(() => response([])));
    render(<NarrowLayout />);
    expect(await screen.findAllByRole('complementary', { name: 'Coordinator cockpit' })).toHaveLength(1);
  });
  it.each([
    ['accepted', { status: 'accepted', result: { id: 'session-a' } }],
    ['rejected', { status: 'rejected', result: 'Denied' }],
    ['conflict', { status: 409, reason: 'Goal A is stale' }],
  ])('ignores a late %s goal receipt after responsive remount', async (_, receipt) => {
    let matches = true;
    const listeners = new Set<() => void>();
    vi.stubGlobal('matchMedia', vi.fn(() => ({
      get matches() { return matches; },
      addEventListener: (_event: string, listener: () => void) => { listeners.add(listener); },
      removeEventListener: (_event: string, listener: () => void) => { listeners.delete(listener); },
    })));
    let finishStart!: (value: Response) => void;
    const pendingStart = new Promise<Response>((resolve) => { finishStart = resolve; });
    vi.stubGlobal('fetch', vi.fn((input: RequestInfo | URL, options?: RequestInit) => {
      const url = String(input);
      if (url === '/api/coordinators/commands' && options?.method === 'POST') return pendingStart;
      if (url === '/api/coordinators') return response([{ id: 'session-b', description: 'Goal B', provider: 'codex' }]);
      if (url.endsWith('/snapshot')) return response({
        ...snapshot,
        session: { id: 'session-b', goal_id: 2, provider: 'codex' },
        goal: { id: 2, description: 'Goal B', status: 'pending', parent_id: null },
        nodes: [{ id: 2, description: 'Goal B', status: 'pending', parent_id: null }],
      });
      return response({});
    }));
    render(<NarrowLayout />);
    fireEvent.change(await screen.findByLabelText('Goal'), { target: { value: 'Goal A' } });
    fireEvent.click(screen.getByRole('button', { name: 'Start goal' }));
    expect(vi.mocked(fetch).mock.calls.filter(([url, options]) =>
      String(url) === '/api/coordinators/commands' && options?.method === 'POST')).toHaveLength(1);
    act(() => { matches = false; for (const listener of listeners) listener(); });
    fireEvent.click(await screen.findByRole('button', { name: 'Goal B · codex' }));
    expect(await screen.findByText('Goal B', { selector: 'strong' })).toBeInTheDocument();
    await act(async () => {
      finishStart('reason' in receipt
        ? { status: 409, json: async () => ({ reason: receipt.reason }) } as Response
        : await response(receipt) as Response);
    });
    expect(localStorage.getItem('milknado.coordinator.session')).toBe('session-b');
    expect(getState().coordinatorGraph?.nodes[0]?.description).toBe('Goal B');
    expect(getState().notices).toEqual([]);
    expect(vi.mocked(fetch).mock.calls.filter(([url, options]) =>
      String(url) === '/api/coordinators/commands' && options?.method === 'POST')).toHaveLength(1);
  });
  it('does not publish stale session command or snapshot conflicts', async () => {
    localStorage.setItem('milknado.coordinator.session', 'session-a');
    let finishSnapshot!: (value: Response) => void;
    let finishCommand!: (value: Response) => void;
    const pendingSnapshot = new Promise<Response>((resolve) => { finishSnapshot = resolve; });
    const pendingCommand = new Promise<Response>((resolve) => { finishCommand = resolve; });
    vi.stubGlobal('fetch', vi.fn((input: RequestInfo | URL, options?: RequestInit) => {
      const url = String(input);
      if (url === '/api/coordinators') return response([]);
      if (url.includes('session-a/snapshot')) return pendingSnapshot;
      if (url.includes('session-a/commands') && options?.method === 'POST') return pendingCommand;
      if (url.includes('session-b/snapshot')) return response({ ...snapshot,
        session: { id: 'session-b', goal_id: 2, provider: 'codex' },
        nodes: [{ id: 2, description: 'Goal B', status: 'pending', parent_id: null }],
      });
      return response({});
    }));
    const { result } = renderHook(() => useCoordinatorSession());
    await waitFor(() => expect(vi.mocked(fetch).mock.calls.some(([url]) =>
      String(url).includes('session-a/snapshot'))).toBe(true));
    act(() => { void result.current.send('plan_goal'); });
    act(() => { result.current.selectSession('session-b'); });
    await waitFor(() => expect(getState().coordinatorGraph?.nodes[0]?.description).toBe('Goal B'));
    await act(async () => {
      finishSnapshot({ status: 409, json: async () => ({ reason: 'Old snapshot' }) } as Response);
      finishCommand({ status: 409, json: async () => ({ reason: 'Old command' }) } as Response);
    });
    expect(getState().notices).toEqual([]);
    expect(getState().coordinatorGraph?.nodes[0]?.description).toBe('Goal B');
    expect(localStorage.getItem('milknado.coordinator.session')).toBe('session-b');
  });
  it('does not publish a stale post-start discovery conflict', async () => {
    let finishDiscovery!: (value: Response) => void;
    const pendingDiscovery = new Promise<Response>((resolve) => { finishDiscovery = resolve; });
    let discoveryCalls = 0;
    vi.stubGlobal('fetch', vi.fn((input: RequestInfo | URL, options?: RequestInit) => {
      const url = String(input);
      if (url === '/api/coordinators') return ++discoveryCalls === 1 ? response([]) : pendingDiscovery;
      if (url === '/api/coordinators/commands' && options?.method === 'POST') {
        return response({ status: 'accepted', result: { id: 'session-a' } });
      }
      if (url.includes('session-b/snapshot')) return response({ ...snapshot,
        session: { id: 'session-b', goal_id: 2, provider: 'codex' },
        nodes: [{ id: 2, description: 'Goal B', status: 'pending', parent_id: null }],
      });
      return response(snapshot);
    }));
    const { result } = renderHook(() => useCoordinatorSession());
    await waitFor(() => expect(result.current.available).toBe(true));
    act(() => { result.current.setGoal('Goal A'); });
    await act(async () => {
      void result.current.start({ preventDefault: () => {} } as Parameters<typeof result.current.start>[0]);
      await waitFor(() => expect(discoveryCalls).toBe(2));
    });
    act(() => { result.current.selectSession('session-b'); });
    await waitFor(() => expect(getState().coordinatorGraph?.nodes[0]?.description).toBe('Goal B'));
    await act(async () => { finishDiscovery({ status: 409,
      json: async () => ({ reason: 'Old discovery' }) } as Response); });
    expect(getState().notices).toEqual([]);
    expect(localStorage.getItem('milknado.coordinator.session')).toBe('session-b');
  });
});
