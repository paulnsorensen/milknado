import { beforeEach, describe, expect, it } from 'vitest';
import { getState, resetStore, setGraphView, setSelection } from './store';

describe('store', () => {
  beforeEach(resetStore);

  it('round-trips a selection', () => {
    setSelection(7);
    expect(getState().selection).toBe(7);
  });

  it('merges a graph view patch without dropping other fields', () => {
    setGraphView({ zoom: 1.5 });
    setGraphView({ filter: 'ready' });
    expect(getState().graphView).toMatchObject({ zoom: 1.5, filter: 'ready' });
  });
});
