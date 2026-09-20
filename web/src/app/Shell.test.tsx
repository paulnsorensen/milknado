import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { get } from './api';
import { clearSlots, registerSlot } from './slots';

vi.mock('./api', () => ({ get: vi.fn() }));
vi.mock('./DefaultLayout', () => ({ DefaultLayout: () => <div data-testid="default-layout" /> }));

import { Shell } from './Shell';

describe('Shell', () => {
  afterEach(() => {
    clearSlots();
    cleanup();
  });

  it('throws when more than one layout contribution is registered', () => {
    vi.mocked(get).mockResolvedValue(null);
    registerSlot('layout', () => <div>Layout A</div>);
    registerSlot('layout', () => <div>Layout B</div>);

    expect(() => render(<Shell />)).toThrow('Expected at most one layout contribution, found 2.');
  });

  it('renders the single registered layout contribution', () => {
    vi.mocked(get).mockResolvedValue(null);
    registerSlot('layout', () => <div>Layout Marker</div>);

    render(<Shell />);

    expect(screen.getByText('Layout Marker')).toBeInTheDocument();
  });

  it('renders the default layout when no layout is registered', () => {
    vi.mocked(get).mockResolvedValue(null);

    render(<Shell />);

    expect(screen.getByTestId('default-layout')).toBeInTheDocument();
  });
});
