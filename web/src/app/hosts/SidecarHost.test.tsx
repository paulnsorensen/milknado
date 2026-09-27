import { act, cleanup, render } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { resetTab, setActiveTab } from '../../shared/node-detail';
import { resetStore, setSelection } from '../store';
import { SidecarHost } from './SidecarHost';

vi.mock('./renderSlot', () => ({
  renderSlot: () => [],
}));

describe('SidecarHost', () => {
  beforeEach(() => {
    resetStore();
    resetTab();
  });

  afterEach(cleanup);

  it('hides the node-detail tab strip until a node is selected', () => {
    const { container } = render(<SidecarHost />);

    expect(container.querySelector('[data-region="sidecar-tab"]')).toBeNull();

    act(() => setSelection(5));

    expect(container.querySelector('[data-region="sidecar-tab"]')).not.toBeNull();
  });

  it('widens the sidecar only for a selected node on the changes tab', () => {
    setActiveTab('changes');
    const { container } = render(<SidecarHost />);

    expect(container.querySelector('[data-region="sidecar"]')?.className).not.toContain('is-wide');

    act(() => setSelection(5));

    expect(container.querySelector('[data-region="sidecar"]')?.className).toContain('is-wide');
  });
});
