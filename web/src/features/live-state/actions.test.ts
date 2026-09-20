import { beforeEach, describe, expect, it } from 'vitest';
import { getState, resetStore, setSelection, setSnapshot } from '../../app/store';
import { openEvents, selectNextRun, selectPreviousRun } from './actions';
import { mergeSnapshot, type RawStreamSnapshot } from './runtimeSnapshot';

function snapshotWithRuns(...runIds: string[]): void {
  const raw: RawStreamSnapshot = {
    goal: 'Ship the tracer',
    graph: null,
    active_runs: runIds.map((run_id, index) => ({
      run_id,
      node_id: index,
      description: `Run ${run_id}`,
      status: 'running',
    })),
    event_lines: [],
  };
  setSnapshot(mergeSnapshot(raw, null));
}

describe('live-state actions', () => {
  beforeEach(resetStore);

  it('selects the first run when nothing is selected', () => {
    snapshotWithRuns('a', 'b');

    selectNextRun();

    expect(getState().selection).toBe('a');
  });

  it('cycles forward with wraparound', () => {
    snapshotWithRuns('a', 'b');

    selectNextRun();
    selectNextRun();

    expect(getState().selection).toBe('b');
  });

  it('cycles backward with wraparound', () => {
    snapshotWithRuns('a', 'b');
    setSelection('a');

    selectPreviousRun();

    expect(getState().selection).toBe('b');
  });

  it('does nothing without active runs', () => {
    snapshotWithRuns();

    selectNextRun();

    expect(getState().selection).toBeNull();
  });

  it('opens the events sidecar', () => {
    openEvents();

    expect(getState().activeSidecar).toBe('events');
  });
});
