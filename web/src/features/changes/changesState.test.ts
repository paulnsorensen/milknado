import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { get } from '../../app/api';
import { getChangesState, resetChanges, selectPath, setRunId } from './changesState';

vi.mock('../../app/api', () => ({
  get: vi.fn().mockResolvedValue([{ path: 'a.py', status: 'modified', added: 1, removed: 0, old_path: null }]),
}));

describe('changesState', () => {
  beforeEach(() => {
    resetChanges();
    vi.mocked(get).mockClear();
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('diff text', { status: 200 })));
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.clearAllMocks();
  });

  it('fetches the file list when the run id changes', async () => {
    setRunId('run-1');
    await Promise.resolve();

    expect(get).toHaveBeenCalledWith('/api/runs/run-1/changes');
    expect(getChangesState().files).toEqual([{ path: 'a.py', status: 'modified', added: 1, removed: 0, old_path: null }]);
  });

  it('resets the file list when the run id changes again', async () => {
    setRunId('run-1');
    await Promise.resolve();

    setRunId('run-2');
    expect(getChangesState().files).toEqual([]);
    expect(getChangesState().selectedPath).toBeNull();
  });

  it('fetches the diff text for the selected file', async () => {
    setRunId('run-1');
    await Promise.resolve();

    selectPath('a.py');
    await new Promise((resolve) => setTimeout(resolve, 0));

    expect(fetch).toHaveBeenCalledWith('/api/runs/run-1/diff?path=a.py');
    expect(getChangesState().diffText).toBe('diff text');
  });

  it('does nothing with no run id', () => {
    selectPath('a.py');
    expect(getChangesState().selectedPath).toBeNull();
  });
});
