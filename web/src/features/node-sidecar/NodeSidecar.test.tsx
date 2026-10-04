import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { get } from '../../app/api';
import { getState, resetStore, setSelection } from '../../app/store';
import { resetDetail, resetTab } from '../../shared/node-detail';
import { NodeSidecar } from './NodeSidecar';
import { resetReviewSelection } from '../goal-review/selection';
import { detailResponse } from './testFixtures';

vi.mock('../../app/api', () => ({ get: vi.fn() }));

describe('NodeSidecar', () => {
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
});