import { beforeEach, describe, expect, it } from 'vitest';
import { getState, resetStore, setSelection, setSnapshot } from '../../app/store';
import type { WireExecutionSnapshot } from '../../app/wire';
import { resetTab, getActiveTab } from '../node-sidecar/detailTab';
import {
  clearSelection,
  openSelectedDetails,
  selectFirstChild,
  selectNextSibling,
  selectParent,
  selectPreviousSibling,
  toggleSelectedCollapsed,
} from './graphNav';

function snapshot(): WireExecutionSnapshot {
  return {
    goal: 'Ship it',
    graph: {
      nodes: [
        { id: 1, description: 'root', status: 'pending', parent_id: null, kind: 'goal', flavor: null },
        { id: 2, description: 'a', status: 'pending', parent_id: 1, kind: 'task', flavor: null },
        { id: 3, description: 'b', status: 'pending', parent_id: 1, kind: 'task', flavor: null },
        { id: 4, description: 'c', status: 'pending', parent_id: 2, kind: 'task', flavor: null },
      ],
      edges: [],
      root_ids: [1],
    },
    capabilities: {
      session_input: { available: true, reason: null },
      cancel: { available: true, reason: null },
      force_stop: { available: true, reason: null },
      stop_scheduling: { available: true, reason: null },
      graph_edits: { available: true, reason: null },
      review_decision: { available: true, reason: null },
      git: { available: true, reason: null },
      owner: { available: false },
    },
  };
}

describe('graphNav', () => {
  beforeEach(() => {
    resetStore();
    resetTab();
    setSnapshot(snapshot());
  });

  it('selects the first sibling when nothing is selected', () => {
    selectNextSibling();
    expect(getState().selection).toBe(1);
  });

  it('cycles forward through siblings with wraparound', () => {
    setSelection(2);
    selectNextSibling();
    expect(getState().selection).toBe(3);
    selectNextSibling();
    expect(getState().selection).toBe(2);
  });

  it('cycles backward through siblings with wraparound', () => {
    setSelection(2);
    selectPreviousSibling();
    expect(getState().selection).toBe(3);
  });

  it('selects the parent', () => {
    setSelection(4);
    selectParent();
    expect(getState().selection).toBe(2);
  });

  it('does nothing at the root when there is no parent', () => {
    setSelection(1);
    selectParent();
    expect(getState().selection).toBe(1);
  });

  it('selects the first child', () => {
    setSelection(2);
    selectFirstChild();
    expect(getState().selection).toBe(4);
  });

  it('does nothing on a leaf with no children', () => {
    setSelection(4);
    selectFirstChild();
    expect(getState().selection).toBe(4);
  });

  it('clears the selection', () => {
    setSelection(2);
    clearSelection();
    expect(getState().selection).toBeNull();
  });

  it('opens the Details tab for the selected node', () => {
    setSelection(2);
    openSelectedDetails();
    expect(getActiveTab()).toBe('details');
  });

  it('does not open a tab with no selection', () => {
    openSelectedDetails();
    expect(getActiveTab()).toBe('session');
  });

  it('toggles the selected node in and out of collapsed', () => {
    setSelection(2);
    toggleSelectedCollapsed();
    expect(getState().graphView.collapsed).toEqual([2]);
    toggleSelectedCollapsed();
    expect(getState().graphView.collapsed).toEqual([]);
  });
});
