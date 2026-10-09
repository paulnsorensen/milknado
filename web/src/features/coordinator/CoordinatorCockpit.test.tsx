import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { getState, resetStore, setSelection } from '../../app/store';
import { getActiveTab, resetTab } from '../../shared/node-detail';
import { CoordinatorCockpit } from './CoordinatorCockpit';

const response = (body: unknown) => Promise.resolve({ ok: true, status: 200, json: async () => body });

describe('CoordinatorCockpit', () => {
  beforeEach(() => {
    resetStore();
    resetTab();
    localStorage.clear();
    vi.stubGlobal('fetch', vi.fn());
  });
  afterEach(() => {
    cleanup();
    localStorage.clear();
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  it('hides controls when the server has no coordinator port', async () => {
    vi.mocked(fetch).mockResolvedValue({ ok: false, status: 409 } as Response);
    render(<CoordinatorCockpit />);
    await waitFor(() => expect(fetch).toHaveBeenCalledWith('/api/coordinators'));
    expect(screen.queryByRole('button', { name: 'Start goal' })).not.toBeInTheDocument();
    expect(screen.queryByRole('complementary', { name: 'Coordinator cockpit' })).not.toBeInTheDocument();
  });

  it('finds durable sessions and opens one without browser storage', async () => {
    vi.mocked(fetch).mockImplementation((input) => {
      const url = String(input);
      if (url === '/api/coordinators') return response([{ id: 'session-2', description: 'Recovered goal', provider: 'codex' }]) as Promise<Response>;
      return response({
        session: { id: 'session-2', goal_id: 2, provider: 'codex' },
        goal: { id: 2, description: 'Recovered goal', status: 'pending', parent_id: null },
        nodes: [
          { id: 2, description: 'Recovered goal', status: 'pending', parent_id: null },
          { id: 3, description: 'First', status: 'pending', parent_id: 2 },
          { id: 4, description: 'Second', status: 'pending', parent_id: 2 },
        ],
        edges: [{ parent_id: 2, child_id: 3 }, { parent_id: 2, child_id: 4 }, { parent_id: 3, child_id: 4 }],
        runs: [], reviews: [], recovery: [], provider_turns: [], provider_bindings: [],
        capability_floor: {}, native_actions: [], unsupported_actions: [], events: [], cursor: 0,
      }) as Promise<Response>;
    });
    render(<CoordinatorCockpit />);
    fireEvent.click(await screen.findByRole('button', { name: /Recovered goal/ }));
    expect(await screen.findByText('Recovered goal', { selector: 'strong' })).toBeInTheDocument();
    expect(localStorage.getItem('milknado.coordinator.session')).toBe('session-2');
    expect(getState().coordinatorGraph?.edges).toContainEqual({ parent_id: 3, child_id: 4 });
  });

  it('starts a goal and shows node evidence from its coordinator snapshot', async () => {
    vi.mocked(fetch).mockImplementation((input, options) => {
      const url = String(input);
      if (url === '/api/coordinators') return response([{ id: 'session-1', description: 'Ship goal', provider: 'claude' }]) as Promise<Response>;
      if (options?.method === 'POST') return response({ status: 'accepted', result: { id: 'session-1' } }) as Promise<Response>;
      return response({
        session: { id: 'session-1', goal_id: 1, provider: 'claude' },
        goal: { id: 1, description: 'Ship goal', status: 'pending', parent_id: null },
        nodes: [{ id: 1, description: 'Ship goal', status: 'pending', parent_id: null, artifact_path: 'notes/result.md' }],
        edges: [],
        runs: [{ run_id: 'run-1', node_id: 1, status: 'completed', detail: 'Checks passed', verification_status: 'accepted', verified_at: '2026-01-01' }],
        reviews: [], recovery: [], provider_turns: [], provider_bindings: [],
        capability_floor: { stream: 'supported' }, native_actions: [], unsupported_actions: [],
        events: [{ seq: 1, kind: 'run_transition', text: 'Run complete', entity_kind: 'node', entity_id: '1', status: 'done' }], cursor: 1,
      }) as Promise<Response>;
    });
    render(<CoordinatorCockpit />);
    fireEvent.change(await screen.findByLabelText('Goal'), { target: { value: 'Ship goal' } });
    fireEvent.click(screen.getByRole('button', { name: 'Start goal' }));
    await waitFor(() => expect(screen.getByText('Ship goal', { selector: 'strong' })).toBeInTheDocument());
    setSelection(1);
    expect(await screen.findByText(/Run complete/)).toBeInTheDocument();
    expect(screen.getByText('notes/result.md')).toBeInTheDocument();
    expect(screen.getByText(/Checks passed/)).toBeInTheDocument();
    expect(screen.getByRole('region', { name: 'Completion verification' }).textContent).toContain('accepted');
    expect(getState().coordinatorGraph?.nodes.map((node) => node.id)).toEqual([1]);
    fireEvent.click(screen.getByRole('button', { name: 'Open changes and diff' }));
    expect(getActiveTab()).toBe('changes');
  });
  it('shows goal approval and scoped recovery while a child task is selected', async () => {
    localStorage.setItem('milknado.coordinator.session', 'session-3');
    vi.mocked(fetch).mockImplementation((input, options) => {
      if (String(input) === '/api/coordinators') return response([]) as Promise<Response>;
      if (options?.method === 'POST') return response({ status: 'accepted', result: {} }) as Promise<Response>;
      return response({
        session: { id: 'session-3', goal_id: 1, provider: 'claude' },
        goal: { id: 1, description: 'Goal', status: 'pending', parent_id: null },
        nodes: [
          { id: 1, description: 'Goal', status: 'pending', parent_id: null },
          { id: 2, description: 'Task', status: 'running', parent_id: 1 },
        ],
        edges: [{ parent_id: 1, child_id: 2 }],
        runs: [], reviews: [{ review_id: 7, goal_id: 1, decision: 'pending', evidence: 'Review evidence', proposed_change: 'Approve plan' }],
        recovery: [
          { seq: 4, kind: 'recovery', entity_kind: 'coordinator', entity_id: 'session-3', status: 'resumed', text: 'provider recovery result' },
          { seq: 5, kind: 'recovery', entity_kind: 'execution_group', entity_id: 'group-1', status: 'unavailable', text: 'provider recovery result' },
        ],
        provider_turns: [], provider_bindings: [], capability_floor: {}, native_actions: [], unsupported_actions: [], events: [], cursor: 5,
      }) as Promise<Response>;
    });
    render(<CoordinatorCockpit />);
    await screen.findByText('Goal', { selector: 'strong' });
    setSelection(2);
    expect((await screen.findByRole('region', { name: 'Approvals' })).textContent).toContain('Review evidence');
    expect(screen.getByRole('button', { name: 'Approve' })).toBeInTheDocument();
    const recovery = screen.getByRole('region', { name: 'Recovery' });
    expect(recovery.textContent).toContain('coordinator session-3: resumed');
    expect(recovery.textContent).toContain('execution_group group-1: unavailable');
    fireEvent.click(screen.getByRole('button', { name: 'Approve' }));
    await waitFor(() => expect(vi.mocked(fetch).mock.calls.some(([url, options]) =>
      String(url).endsWith('/commands') && options?.body?.toString().includes('"review_id":7'))).toBe(true));
  });

  it('reviews proposed changes before sending an apply or reject command', async () => {
    localStorage.setItem('milknado.coordinator.session', 'session-4');
    vi.mocked(fetch).mockImplementation((input, options) => {
      if (String(input) === '/api/coordinators') return response([]) as Promise<Response>;
      if (options?.method === 'POST') return response({ status: 'accepted', result: {} }) as Promise<Response>;
      return response({
        session: { id: 'session-4', goal_id: 4, provider: 'codex' },
        goal: { id: 4, description: 'Goal', status: 'pending', parent_id: null },
        nodes: [{ id: 4, description: 'Goal', status: 'pending', parent_id: null }],
        edges: [],
        proposals: [
          { id: 'plan-a', status: 'pending', manifest: { goal_summary: 'Ship', changes: [{ id: 'c1', path: 'src/a.py', description: 'Change A' }] } },
          { id: 'plan-b', status: 'pending', manifest: {
            goal_summary: 'Ship',
            changes: [{ id: 'c2', path: 'src/b.py', description: 'Change B', depends_on: ['c1'] }],
            new_relationships: [{ source_change_id: 'c1', dependant_change_id: 'c2', reason: 'new_import' }],
          } },
        ],
        runs: [], reviews: [], recovery: [], provider_turns: [], provider_bindings: [],
        capability_floor: {}, native_actions: [], unsupported_actions: [], events: [], cursor: 0,
      }) as Promise<Response>;
    });
    render(<CoordinatorCockpit />);
    const section = await screen.findByRole('region', { name: 'Plan proposals' });
    expect(section.textContent).toContain('src/a.py');
    expect(section.textContent).toContain('Change B');
    expect(section.textContent).toContain('Depends on: c1');
    expect(section.textContent).toContain('c1 → c2: new_import');
    fireEvent.click(screen.getByRole('button', { name: 'Approve plan-a' }));
    await waitFor(() => expect(screen.getByRole('button', { name: 'Reject plan-b' })).not.toBeDisabled());
    fireEvent.click(screen.getByRole('button', { name: 'Reject plan-b' }));
    await waitFor(() => {
      const bodies = vi.mocked(fetch).mock.calls
        .filter(([, options]) => options?.method === 'POST')
        .map(([, options]) => JSON.parse(String(options?.body)) as { proposal_id: string; decision: string });
      expect(bodies).toEqual(expect.arrayContaining([
        expect.objectContaining({ proposal_id: 'plan-a', decision: 'accepted' }),
        expect.objectContaining({ proposal_id: 'plan-b', decision: 'rejected' }),
      ]));
    });
  });

  it('ignores a command response after switching coordinator sessions', async () => {
    localStorage.setItem('milknado.coordinator.session', 'session-a');
    let finishCommand!: (response: Response) => void;
    const command = new Promise<Response>((resolve) => { finishCommand = resolve; });
    const snapshot = (id: string) => ({
      session: { id, goal_id: 1, provider: 'codex' },
      goal: { id: 1, description: `Goal ${id}`, status: 'pending', parent_id: null },
      nodes: [{ id: 1, description: `Goal ${id}`, status: 'pending', parent_id: null }],
      edges: [],
      runs: [], reviews: [], recovery: [], provider_turns: [], provider_bindings: [],
      capability_floor: {}, native_actions: [], unsupported_actions: [], events: [], cursor: 0,
    });
    vi.mocked(fetch).mockImplementation((input, options) => {
      const url = String(input);
      if (url === '/api/coordinators') return response([
        { id: 'session-a', description: 'Goal A', provider: 'codex' },
        { id: 'session-b', description: 'Goal B', provider: 'codex' },
      ]) as Promise<Response>;
      if (options?.method === 'POST') return command;
      return response(snapshot(url.includes('session-a') ? 'session-a' : 'session-b')) as Promise<Response>;
    });
    render(<CoordinatorCockpit />);
    expect(await screen.findByText('Goal session-a', { selector: 'strong' })).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Propose plan' }));
    fireEvent.click(screen.getByRole('button', { name: 'Goal B · codex' }));
    expect(await screen.findByText('Goal session-b', { selector: 'strong' })).toBeInTheDocument();
    await act(async () => { finishCommand(await response({ status: 'accepted', result: {} }) as Response); });
    expect(screen.getByText('Goal session-b', { selector: 'strong' })).toBeInTheDocument();
    expect(getState().coordinatorGraph?.nodes[0]?.description).toBe('Goal session-b');
    expect(screen.queryByText(/plan_goal: accepted/)).not.toBeInTheDocument();
    expect(vi.mocked(fetch).mock.calls.filter(([url]) => String(url).includes('session-a/snapshot'))).toHaveLength(1);
  });

  it('keeps the live snapshot when the current session is selected again', async () => {
    localStorage.setItem('milknado.coordinator.session', 'session-a');
    vi.mocked(fetch).mockImplementation((input) => {
      if (String(input) === '/api/coordinators') return response([
        { id: 'session-a', description: 'Goal A', provider: 'codex' },
      ]) as Promise<Response>;
      return response({
        session: { id: 'session-a', goal_id: 1, provider: 'codex' },
        goal: { id: 1, description: 'Goal A', status: 'pending', parent_id: null },
        nodes: [{ id: 1, description: 'Goal A', status: 'pending', parent_id: null }],
        edges: [],
        runs: [], reviews: [], recovery: [], provider_turns: [], provider_bindings: [],
        capability_floor: {}, native_actions: [], unsupported_actions: [], events: [], cursor: 0,
      }) as Promise<Response>;
    });
    render(<CoordinatorCockpit />);
    expect(await screen.findByText('Goal A', { selector: 'strong' })).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Goal A · codex' }));
    expect(screen.getByText('Goal A', { selector: 'strong' })).toBeInTheDocument();
    expect(getState().coordinatorGraph?.nodes[0]?.description).toBe('Goal A');
  });
  it('bounds delayed polls, aborts cleanup, and rejects a stale same-session response', async () => {
    localStorage.setItem('milknado.coordinator.session', 'session-a');
    let finishPoll!: (response: Response) => void;
    const delayed = new Promise<Response>((resolve) => { finishPoll = resolve; });
    let snapshotReads = 0;
    let pollSignal: AbortSignal | undefined;
    const snapshot = (description: string) => ({
      session: { id: 'session-a', goal_id: 1, provider: 'codex' },
      goal: { id: 1, description, status: 'pending', parent_id: null },
      nodes: [{ id: 1, description, status: 'pending', parent_id: null }],
      edges: [], runs: [], reviews: [], recovery: [], provider_turns: [], provider_bindings: [],
      capability_floor: {}, native_actions: [], unsupported_actions: [], events: [], cursor: 0,
    });
    vi.mocked(fetch).mockImplementation((input, options) => {
      if (String(input) === '/api/coordinators') return response([]) as Promise<Response>;
      if (options?.method === 'POST') return response({ status: 'accepted', result: {} }) as Promise<Response>;
      snapshotReads += 1;
      if (snapshotReads === 2) { pollSignal = options?.signal ?? undefined; return delayed; }
      return response(snapshot(snapshotReads === 1 ? 'Before command' : 'After command')) as Promise<Response>;
    });
    vi.useFakeTimers();
    const view = render(<CoordinatorCockpit />);
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });
    expect(screen.getByText('Before command', { selector: 'strong' })).toBeInTheDocument();
    await act(async () => { await vi.advanceTimersByTimeAsync(2000); });
    expect(snapshotReads).toBe(2);
    await act(async () => { await vi.advanceTimersByTimeAsync(10000); });
    expect(snapshotReads).toBe(2);
    fireEvent.click(screen.getByRole('button', { name: 'Propose plan' }));
    await act(async () => { await Promise.resolve(); await Promise.resolve(); });
    expect(snapshotReads).toBe(3);
    expect(screen.getByText('After command', { selector: 'strong' })).toBeInTheDocument();
    expect(pollSignal?.aborted).toBe(true);
    await act(async () => { finishPoll(await response(snapshot('Stale poll')) as Response); });
    expect(screen.getByText('After command', { selector: 'strong' })).toBeInTheDocument();
    view.unmount();
    await act(async () => { await vi.advanceTimersByTimeAsync(10000); });
    expect(snapshotReads).toBe(3);
    expect(getState().notices).toEqual([]);
  });

  it('aborts a pending snapshot on unmount without reporting cancellation', async () => {
    localStorage.setItem('milknado.coordinator.session', 'session-a');
    let pendingSignal: AbortSignal | undefined;
    vi.mocked(fetch).mockImplementation((input, options) => {
      if (String(input) === '/api/coordinators') return response([]) as Promise<Response>;
      pendingSignal = options?.signal ?? undefined;
      return new Promise<Response>(() => {});
    });
    const view = render(<CoordinatorCockpit />);
    await waitFor(() => expect(pendingSignal).toBeDefined());
    view.unmount();
    expect(pendingSignal?.aborted).toBe(true);
    expect(getState().notices).toEqual([]);
  });

  it('aborts the old session snapshot and ignores its late result', async () => {
    localStorage.setItem('milknado.coordinator.session', 'session-a');
    let finishOld!: (response: Response) => void;
    let oldSignal: AbortSignal | undefined;
    vi.mocked(fetch).mockImplementation((input, options) => {
      const url = String(input);
      if (url === '/api/coordinators') return response([
        { id: 'session-a', description: 'Goal A', provider: 'codex' },
        { id: 'session-b', description: 'Goal B', provider: 'codex' },
      ]) as Promise<Response>;
      if (url.includes('session-a/snapshot')) {
        oldSignal = options?.signal ?? undefined;
        return new Promise<Response>((resolve) => { finishOld = resolve; });
      }
      return response({ session: { id: 'session-b', goal_id: 2, provider: 'codex' },
        goal: { id: 2, description: 'Goal B', status: 'pending', parent_id: null },
        nodes: [{ id: 2, description: 'Goal B', status: 'pending', parent_id: null }],
        edges: [], runs: [], reviews: [], recovery: [], provider_turns: [], provider_bindings: [],
        capability_floor: {}, native_actions: [], unsupported_actions: [], events: [], cursor: 0,
      }) as Promise<Response>;
    });
    render(<CoordinatorCockpit />);
    await waitFor(() => expect(oldSignal).toBeDefined());
    fireEvent.click(screen.getByRole('button', { name: 'Goal B · codex' }));
    expect(await screen.findByText('Goal B', { selector: 'strong' })).toBeInTheDocument();
    expect(oldSignal?.aborted).toBe(true);
    await act(async () => { finishOld(await response({ goal: { description: 'Stale A' } }) as Response); });
    expect(screen.getByText('Goal B', { selector: 'strong' })).toBeInTheDocument();
    expect(getState().notices).toEqual([]);
  });
});
