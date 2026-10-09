import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ApiError, get, post } from './api';
import { getState, resetStore } from './store';

describe('api', () => {
  beforeEach(() => {
    resetStore();
    vi.stubGlobal('fetch', vi.fn());
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('throws a typed ApiError with the response status for a non-409 failure', async () => {
    vi.mocked(fetch).mockResolvedValue(new Response('', { status: 500 }));

    await expect(get('/api/thing')).rejects.toBeInstanceOf(ApiError);
    await expect(get('/api/thing')).rejects.toMatchObject({ status: 500 });
  });
  it('passes an optional cancellation signal through GET', async () => {
    const controller = new AbortController();
    vi.mocked(fetch).mockResolvedValue(new Response('{}'));

    await get('/api/thing', controller.signal);

    expect(fetch).toHaveBeenCalledWith('/api/thing', expect.objectContaining({ signal: controller.signal }));
  });
  it('keeps a live conflict notice and suppresses a scope that expires during JSON read', async () => {
    let finishJson!: (value: { reason: string }) => void;
    const pendingJson = new Promise<{ reason: string }>((resolve) => { finishJson = resolve; });
    let jsonStarted = false;
    vi.mocked(fetch).mockResolvedValue({ status: 409, json: () => {
      jsonStarted = true;
      return pendingJson;
    } } as Response);
    let active = true;
    const pending = post('/api/commands', {}, () => active);
    await vi.waitFor(() => expect(jsonStarted).toBe(true));
    active = false;
    finishJson({ reason: 'Obsolete conflict' });
    await expect(pending).resolves.toBeNull();
    expect(getState().notices).toEqual([]);

    await get('/api/thing');
    expect(getState().notices.map((notice) => notice.reason)).toEqual(['Obsolete conflict']);
  });
});
