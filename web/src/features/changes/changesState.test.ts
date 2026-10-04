import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ApiError, get } from '../../app/api';
import * as store from '../../app/store';
import { getChangesState, resetChanges, selectPath, setRunId } from './changesState';

vi.mock('../../app/api', async () => {
  const actual = await vi.importActual<typeof import('../../app/api')>('../../app/api');
  return {
    ApiError: actual.ApiError,
    get: vi.fn().mockResolvedValue([{ path: 'a.py', status: 'modified', added: 1, removed: 0, old_path: null }]),
  };
});

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

  it('treats a missing changes endpoint as the no-changes state', async () => {
    const noticeSpy = vi.spyOn(store, 'pushNotice');
    vi.mocked(get).mockRejectedValueOnce(new ApiError('GET failed with status 404.', 404));

    setRunId('run-missing');
    await Promise.resolve();

    expect(getChangesState().files).toEqual([]);
    expect(getChangesState().selectedPath).toBeNull();
    expect(noticeSpy).not.toHaveBeenCalled();
  });

  it('pushes one notice when the changes endpoint fails', async () => {
    const noticeSpy = vi.spyOn(store, 'pushNotice');
    vi.mocked(get).mockRejectedValueOnce(new ApiError('GET failed with status 500.', 500));

    setRunId('run-failed');
    await Promise.resolve();

    expect(noticeSpy).toHaveBeenCalledTimes(1);
    expect(noticeSpy).toHaveBeenCalledWith('Could not load changed files.');
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
