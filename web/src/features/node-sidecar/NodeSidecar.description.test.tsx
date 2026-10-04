import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { get } from '../../app/api';
import { resetStore, setSelection } from '../../app/store';
import { resetDetail, resetTab } from '../../shared/node-detail';
import { NodeSidecar } from './NodeSidecar';
import { resetReviewSelection } from '../goal-review/selection';
import { detailResponse } from './testFixtures';

vi.mock('../../app/api', () => ({ get: vi.fn() }));

describe('NodeSidecar description', () => {
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

  it('clamps long descriptions until the expand control is used', async () => {
    const description = 'A'.repeat(600);
    vi.mocked(get).mockResolvedValue(detailResponse({ description }));
    setSelection(7);
    const scrollHeight = vi
      .spyOn(HTMLElement.prototype, 'scrollHeight', 'get')
      .mockReturnValue(600);
    const clientHeight = vi
      .spyOn(HTMLElement.prototype, 'clientHeight', 'get')
      .mockReturnValue(100);

    render(<NodeSidecar />);

    expect(
      await screen.findByRole('button', { name: 'Expand description' }),
    ).toBeTruthy();
    expect(screen.getByRole('heading', { name: description })).not.toHaveClass(
      'is-expanded',
    );
    fireEvent.click(screen.getByRole('button', { name: 'Expand description' }));
    expect(screen.getByRole('heading', { name: description })).toHaveClass(
      'is-expanded',
    );
    scrollHeight.mockRestore();
    clientHeight.mockRestore();
  });
});
