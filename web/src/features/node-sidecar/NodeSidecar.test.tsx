import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { get } from '../../app/api';
import { resetStore, setSelection } from '../../app/store';
import { resetDetail } from './nodeDetail';
import { resetTab, setActiveTab } from './detailTab';
import { NodeSidecar } from './NodeSidecar';
import type { WireNodeDetailResponse } from './detailWire';

vi.mock('../../app/api', () => ({ get: vi.fn() }));

function detailResponse(overrides: Partial<WireNodeDetailResponse['detail']> = {}): WireNodeDetailResponse {
  return {
    node_id: 7,
    request_generation: 1,
    detail: {
      node: { id: 7, description: 'Bake the roadmap', status: 'running', parent_id: null, kind: 'task', flavor: null },
      description: 'Bake the roadmap',
      parent: null,
      ancestors: { items: [], offset: 0, limit: 50, total: 0, has_more: false, state: 'loaded' },
      prerequisite_ids: { items: [], offset: 0, limit: 50, total: 0, has_more: false, state: 'loaded' },
      dependent_ids: { items: [], offset: 0, limit: 50, total: 0, has_more: false, state: 'loaded' },
      owned_files: { items: [], offset: 0, limit: 50, total: 0, has_more: false, state: 'loaded' },
      runs: { items: [], offset: 0, limit: 50, total: 0, has_more: false, state: 'loaded' },
      sessions: { items: [], offset: 0, limit: 50, total: 0, has_more: false, state: 'loaded' },
      ...overrides,
    },
  };
}

describe('NodeSidecar', () => {
  beforeEach(() => {
    resetStore();
    resetDetail();
    resetTab();
    vi.mocked(get).mockReset();
  });

  afterEach(cleanup);

  it('renders nothing without a numeric selection', () => {
    const { container } = render(<NodeSidecar />);
    expect(container.children.length).toBe(0);
  });

  it('renders the empty-runs state for a selected node with no runs', async () => {
    vi.mocked(get).mockResolvedValue(detailResponse());
    setSelection(7);

    render(<NodeSidecar />);

    expect(await screen.findByText('This node has no runs yet.')).toBeTruthy();
    expect(screen.getByText('Bake the roadmap')).toBeTruthy();
  });

  it('renders a run row for each run', async () => {
    vi.mocked(get).mockResolvedValue(
      detailResponse({
        runs: {
          items: [{ run_id: 'run-1', node_id: 7, status: 'running', started_at: '', ended_at: null, error: null }],
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
  });

  it('renders the details tab body when the details tab is active', async () => {
    vi.mocked(get).mockResolvedValue(detailResponse());
    setSelection(7);
    setActiveTab('details');

    render(<NodeSidecar />);

    expect(await screen.findByText('Parent')).toBeTruthy();
  });
});
