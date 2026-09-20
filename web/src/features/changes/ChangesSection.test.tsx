import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { get } from '../../app/api';
import { getDetailState, resetTab, setActiveTab, type DetailState } from '../../shared/node-detail';
import { resetChanges } from './changesState';
import { ChangesSection } from './ChangesSection';

vi.mock('../../app/api', () => ({ get: vi.fn() }));
vi.mock('../../shared/node-detail', async (importOriginal) => ({
  ...(await importOriginal<object>()),
  getDetailState: vi.fn(),
  subscribeDetail: vi.fn(() => () => {}),
}));

const emptyPage = { items: [], offset: 0, limit: 50, total: 0, has_more: false, state: 'loaded' as const };

function detailStateWithRun(runId: string | null): DetailState {
  return {
    nodeId: 7,
    page: 0,
    sessionPage: 0,
    detail: {
      node_id: 7,
      request_generation: 1,
      detail: {
        node: { id: 7, description: '', status: 'running', parent_id: null, kind: 'task', flavor: null },
        description: '',
        parent: null,
        ancestors: emptyPage,
        prerequisite_ids: emptyPage,
        dependent_ids: emptyPage,
        owned_files: emptyPage,
        runs: runId
          ? { ...emptyPage, items: [{ run_id: runId, node_id: 7, status: 'running', started_at: '', ended_at: null, error: null }] }
          : emptyPage,
        sessions: emptyPage,
      },
    },
  };
}

describe('ChangesSection', () => {
  beforeEach(() => {
    resetTab();
    resetChanges();
    vi.mocked(get).mockResolvedValue([{ path: 'a.py', status: 'modified', added: 1, removed: 0, old_path: null }]);
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('diff text', { status: 200 })));
  });

  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
    vi.clearAllMocks();
  });

  it('renders nothing outside the changes tab', () => {
    vi.mocked(getDetailState).mockReturnValue(detailStateWithRun('run-1'));
    const { container } = render(<ChangesSection />);
    expect(container.children.length).toBe(0);
  });

  it('lists changed files and shows the diff for the selected file', async () => {
    vi.mocked(getDetailState).mockReturnValue(detailStateWithRun('run-1'));
    setActiveTab('changes');

    render(<ChangesSection />);

    const fileButton = await screen.findByText('a.py');
    fileButton.click();

    expect(await screen.findByText('diff text')).toBeTruthy();
    expect(fetch).toHaveBeenCalledWith('/api/runs/run-1/diff?path=a.py');
  });

  it('shows an empty state with no changed files', async () => {
    vi.mocked(get).mockResolvedValue([]);
    vi.mocked(getDetailState).mockReturnValue(detailStateWithRun('run-1'));
    setActiveTab('changes');

    render(<ChangesSection />);

    expect(await screen.findByText('No changed files yet.')).toBeTruthy();
  });
});
