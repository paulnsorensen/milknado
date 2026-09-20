import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { getState, resetStore, setGraphView } from './store';

vi.mock('../design-system', () => ({
  Milknado: {
    MikadoGraph: (props: { onLayout: (layout: { lod: string; collapsed: string[] }) => void }) => (
      <button onClick={() => props.onLayout({ lod: 'pill', collapsed: ['x'] })}>trigger</button>
    ),
  },
}));

import { DefaultLayout } from './DefaultLayout';

describe('DefaultLayout', () => {
  beforeEach(() => {
    resetStore();
  });

  afterEach(cleanup);

  it('keeps the numeric collapsed selection when the graph reports a layout', () => {
    setGraphView({ collapsed: [42] });

    render(<DefaultLayout />);
    screen.getByText('trigger').click();

    expect(getState().graphView.lod).toBe('pill');
    expect(getState().graphView.collapsed).toEqual([42]);
  });
});
