import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { get } from '../../app/api';
import { resetStore, setSelection, setSnapshot } from '../../app/store';
import { resetDetail, resetTab } from '../../shared/node-detail';
import { NodeSidecar } from './NodeSidecar';
import { resetReviewSelection } from '../goal-review/selection';
import { detailResponse } from './testFixtures';

vi.mock('../../app/api', () => ({ get: vi.fn() }));

describe('NodeSidecar runs', () => {
  beforeEach(() => {
    resetStore();
    resetDetail();
    resetTab();
    resetReviewSelection();
    vi.mocked(get).mockReset();
  });

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  it('renders the empty caption without a selection', () => {
    render(<NodeSidecar />);
    expect(
      screen.getByText('Select a node to inspect its details.'),
    ).toBeTruthy();
  });

  it('renders the empty-runs state for a selected node with no runs', async () => {
    vi.mocked(get).mockResolvedValue(detailResponse());
    setSelection(7);

    render(<NodeSidecar />);

    expect(await screen.findByText('This node has no runs yet.')).toBeTruthy();
    expect(screen.getByText('Bake the roadmap')).toBeTruthy();
  });

  it('keeps watch metrics inside the Run group and waits for capabilities', async () => {
    vi.mocked(get).mockResolvedValue(detailResponse());
    setSelection(7);
    render(<NodeSidecar />);

    expect(screen.queryByText('unavailable')).toBeNull();
    setSnapshot({
      goal: null,
      graph: null,
      capabilities: {
        session_input: { available: false, reason: null },
        cancel: { available: false, reason: null },
        force_stop: { available: false, reason: null },
        stop_scheduling: { available: false, reason: null },
        graph_edits: { available: false, reason: null },
        review_decision: { available: false, reason: null },
        git: { available: false, reason: null },
        host_owner: { available: false, reason: null },
        owner: { available: false },
      },
    });

    const run = await screen.findByRole('region', { name: 'Run' });
    expect(run).toHaveTextContent('ETAunavailable');
    expect(run).toHaveTextContent('Attemptunavailable');
    expect(run).toHaveTextContent('guidanceunavailable');
    expect(
      screen.queryByRole('region', { name: 'Watch mode availability' }),
    ).toBeNull();
  });

  it('renders a run row for each run', async () => {
    vi.mocked(get).mockResolvedValue(
      detailResponse({
        runs: {
          items: [
            {
              run_id: 'run-1',
              node_id: 7,
              status: 'running',
              started_at: '',
              ended_at: null,
              error: null,
            },
          ],
          offset: 0,
          limit: 50,
          total: 1,
          has_more: false,
          state: 'loaded',
        },
      }),
    );
    setSelection(7);

    render(<NodeSidecar />);

    expect(await screen.findByText('run-1', { exact: false })).toBeTruthy();
    expect(screen.getByText('Completed')).toBeTruthy();
    expect(screen.getByText('none')).toBeTruthy();
  });

  it('keys error rows by run id, not by index, when two runs share the same error', async () => {
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    vi.mocked(get).mockResolvedValue(
      detailResponse({
        runs: {
          items: [
            {
              run_id: 'run-1',
              node_id: 7,
              status: 'failed',
              started_at: '',
              ended_at: '',
              error: 'boom',
            },
            {
              run_id: 'run-2',
              node_id: 7,
              status: 'failed',
              started_at: '',
              ended_at: '',
              error: 'boom',
            },
          ],
          offset: 0,
          limit: 50,
          total: 2,
          has_more: false,
          state: 'loaded',
        },
      }),
    );
    setSelection(7);

    render(<NodeSidecar />);

    expect((await screen.findAllByRole('alert')).length).toBe(2);
    expect(screen.getAllByText('boom')).toHaveLength(2);
    const keyWarning = errorSpy.mock.calls.some((call) =>
      String(call[0]).includes('two children with the same key'),
    );
    expect(keyWarning).toBe(false);
  });
});
