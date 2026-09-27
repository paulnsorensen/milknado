import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { get } from '../../app/api';
import { getState, resetStore, setSelection } from '../../app/store';
import { resetDetail, resetTab, setActiveTab, type WireNodeDetailResponse } from '../../shared/node-detail';
import { NodeSidecar } from './NodeSidecar';
import { DetailsTabSection } from './TabSections';

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

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

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

  it('keys error rows by run id, not by index, when two runs share the same error', async () => {
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    vi.mocked(get).mockResolvedValue(
      detailResponse({
        runs: {
          items: [
            { run_id: 'run-1', node_id: 7, status: 'failed', started_at: '', ended_at: '', error: 'boom' },
            { run_id: 'run-2', node_id: 7, status: 'failed', started_at: '', ended_at: '', error: 'boom' },
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
    const keyWarning = errorSpy.mock.calls.some((call) =>
      String(call[0]).includes('two children with the same key'),
    );
    expect(keyWarning).toBe(false);
  });

  it('calls onClose on the close button when provided', async () => {
    vi.mocked(get).mockResolvedValue(detailResponse());
    setSelection(7);
    const onClose = vi.fn();

    render(<NodeSidecar onClose={onClose} />);
    await screen.findByText('Bake the roadmap');
    fireEvent.click(screen.getByRole('button', { name: 'Close the sidecar' }));

    expect(onClose).toHaveBeenCalledOnce();
    expect(getState().selection).toBe(7);
  });

  it('clears the selection on the close button when onClose is omitted', async () => {
    vi.mocked(get).mockResolvedValue(detailResponse());
    setSelection(7);

    render(<NodeSidecar />);
    await screen.findByText('Bake the roadmap');
    fireEvent.click(screen.getByRole('button', { name: 'Close the sidecar' }));

    expect(getState().selection).toBeNull();
  });

  it('renders the details tab body when the details tab is active', async () => {
    vi.mocked(get).mockResolvedValue(detailResponse());
    setSelection(7);
    setActiveTab('details');

    render(
      <>
        <NodeSidecar />
        <DetailsTabSection />
      </>,
    );

    expect(await screen.findByText('Parent')).toBeTruthy();
  });
});
