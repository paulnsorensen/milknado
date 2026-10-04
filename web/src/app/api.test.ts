import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ApiError, get } from './api';

describe('api', () => {
  beforeEach(() => {
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
});
