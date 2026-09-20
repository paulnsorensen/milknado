import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { get } from '../../app/api';
import {
  followNewest,
  getDetailState,
  pageNext,
  pagePrevious,
  resetDetail,
  selectNode,
  sessionPageNext,
  sessionPagePrevious,
} from './nodeDetail';

vi.mock('../../app/api', () => ({ get: vi.fn().mockResolvedValue({ node_id: 7, request_generation: 1, detail: null }) }));

describe('nodeDetail', () => {
  beforeEach(() => {
    resetDetail();
    vi.mocked(get).mockClear();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it('fetches the node on selection and resets paging', async () => {
    selectNode(7);
    await Promise.resolve();

    expect(getDetailState().nodeId).toBe(7);
    expect(getDetailState().page).toBe(0);
    expect(getDetailState().sessionPage).toBe(0);
    expect(get).toHaveBeenCalledWith('/api/nodes/7?request_generation=1&page=0&session_event_page=0');
  });

  it('pages the detail request forward and backward, clamped at zero', async () => {
    selectNode(7);
    await Promise.resolve();

    pageNext();
    await Promise.resolve();
    expect(getDetailState().page).toBe(1);

    pagePrevious();
    await Promise.resolve();
    expect(getDetailState().page).toBe(0);

    pagePrevious();
    expect(getDetailState().page).toBe(0);
  });

  it('pages the session transcript and follows the newest page', async () => {
    selectNode(7);
    await Promise.resolve();

    sessionPageNext();
    await Promise.resolve();
    expect(getDetailState().sessionPage).toBe(1);

    sessionPagePrevious();
    await Promise.resolve();
    expect(getDetailState().sessionPage).toBe(0);

    sessionPageNext();
    await Promise.resolve();
    followNewest();
    await Promise.resolve();
    expect(getDetailState().sessionPage).toBe(0);
  });

  it('does nothing with no node selected', () => {
    pageNext();
    sessionPageNext();
    expect(getDetailState().nodeId).toBeNull();
  });
});
