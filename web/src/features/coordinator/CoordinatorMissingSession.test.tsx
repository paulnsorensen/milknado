import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { getState, resetStore } from '../../app/store';
import { CoordinatorCockpit } from './CoordinatorCockpit';

const response = (body: unknown) => Promise.resolve({ ok: true, status: 200, json: async () => body } as Response);
const missing = { ok: false, status: 404 } as Response;
const snapshot = (id: string) => ({
  session: { id, goal_id: 1, provider: 'codex' },
  goal: { id: 1, description: `Goal ${id}`, status: 'pending', parent_id: null },
  nodes: [{ id: 1, description: `Goal ${id}`, status: 'pending', parent_id: null }],
  edges: [], runs: [], reviews: [], recovery: [], provider_turns: [], provider_bindings: [],
  capability_floor: {}, native_actions: [], unsupported_actions: [], events: [], cursor: 0,
});

describe('missing coordinator session', () => {
  beforeEach(() => {
    resetStore();
    localStorage.clear();
    vi.stubGlobal('fetch', vi.fn());
  });
  afterEach(() => {
    cleanup();
    localStorage.clear();
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  it('clears a missing persisted session and stops polling after one notice', async () => {
    localStorage.setItem('milknado.coordinator.session', 'missing');
    vi.mocked(fetch).mockImplementation((input) =>
      String(input) === '/api/coordinators' ? response([]) : Promise.resolve(missing));
    vi.useFakeTimers();
    render(<CoordinatorCockpit />);
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });
    expect(screen.getByRole('button', { name: 'Start goal' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'New goal' })).not.toBeInTheDocument();
    expect(localStorage.getItem('milknado.coordinator.session')).toBeNull();
    expect(getState().notices.map((notice) => notice.reason)).toEqual(['Coordinator session was not found.']);
    await act(async () => { await vi.advanceTimersByTimeAsync(5000); });
    expect(vi.mocked(fetch).mock.calls.filter(([url]) => String(url).includes('/snapshot'))).toHaveLength(1);
  });

  it('ignores a late 404 from the previous selection', async () => {
    localStorage.setItem('milknado.coordinator.session', 'session-a');
    let finishOld!: (value: Response) => void;
    vi.mocked(fetch).mockImplementation((input) => {
      const url = String(input);
      if (url === '/api/coordinators') return response([{ id: 'session-b', description: 'Goal B', provider: 'codex' }]);
      if (url.includes('session-a/snapshot')) return new Promise<Response>((resolve) => { finishOld = resolve; });
      return response(snapshot('session-b'));
    });
    render(<CoordinatorCockpit />);
    fireEvent.click(await screen.findByRole('button', { name: 'Goal B · codex' }));
    expect(await screen.findByText('Goal session-b', { selector: 'strong' })).toBeInTheDocument();
    await act(async () => { finishOld(missing); });
    expect(localStorage.getItem('milknado.coordinator.session')).toBe('session-b');
    expect(screen.getByText('Goal session-b', { selector: 'strong' })).toBeInTheDocument();
    expect(getState().notices).toEqual([]);
  });

  it('keeps a non-404 failure selected for the next poll', async () => {
    localStorage.setItem('milknado.coordinator.session', 'session-a');
    let reads = 0;
    vi.mocked(fetch).mockImplementation((input) => {
      if (String(input) === '/api/coordinators') return response([]);
      reads += 1;
      return reads === 1
        ? Promise.resolve({ ok: false, status: 503 } as Response)
        : response(snapshot('session-a'));
    });
    vi.useFakeTimers();
    render(<CoordinatorCockpit />);
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });
    expect(localStorage.getItem('milknado.coordinator.session')).toBe('session-a');
    expect(getState().notices.map((notice) => notice.reason)).toEqual(['Failed to load coordinator status.']);
    await act(async () => { await vi.advanceTimersByTimeAsync(2000); });
    expect(screen.getByText('Goal session-a', { selector: 'strong' })).toBeInTheDocument();
    expect(reads).toBe(2);
  });
  it('ignores an aborted older 404 after a command refresh', async () => {
    localStorage.setItem('milknado.coordinator.session', 'session-a');
    let finishPoll!: (value: Response) => void;
    let pollSignal: AbortSignal | undefined;
    let reads = 0;
    vi.mocked(fetch).mockImplementation((input, options) => {
      if (String(input) === '/api/coordinators') return response([]);
      if (options?.method === 'POST') return response({ status: 'accepted', result: {} });
      reads += 1;
      if (reads === 2) {
        pollSignal = options?.signal ?? undefined;
        return new Promise<Response>((resolve) => { finishPoll = resolve; });
      }
      return response(snapshot('session-a'));
    });
    vi.useFakeTimers();
    render(<CoordinatorCockpit />);
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });
    await act(async () => { await vi.advanceTimersByTimeAsync(2000); });
    expect(reads).toBe(2);
    fireEvent.click(screen.getByRole('button', { name: 'Propose plan' }));
    await act(async () => { await Promise.resolve(); await Promise.resolve(); });
    expect(reads).toBe(3);
    expect(pollSignal?.aborted).toBe(true);
    await act(async () => { finishPoll(missing); });
    expect(localStorage.getItem('milknado.coordinator.session')).toBe('session-a');
    expect(screen.getByText('Goal session-a', { selector: 'strong' })).toBeInTheDocument();
    expect(getState().notices).toEqual([]);
  });
});
