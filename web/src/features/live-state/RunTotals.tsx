// The rail footer: the graph's status strip, the run totals and the owner run.
import type { ReactElement } from 'react';
import { useSyncExternalStore } from 'react';
import { getState, subscribe } from '../../app/store';
import { toGraphNodes } from '../../app/wire';
import { Milknado } from '../../design-system';
import type { StreamSnapshot } from './runtimeSnapshot';
import { formatRunTotals } from '../../shared/runTotals';

export function RunTotals(): ReactElement | null {
  const state = useSyncExternalStore(subscribe, getState);
  const { StatusStrip } = Milknado;
  const snapshot = state.snapshot as Partial<StreamSnapshot> | null;

  if (!snapshot) {
    return null;
  }

  const nodes = snapshot.graph ? toGraphNodes(snapshot.graph) : [];
  const totals = formatRunTotals(snapshot);
  const runId = state.capabilities?.owner.run_id;

  return (
    <div className="mk-rail-footer">
      <StatusStrip nodes={nodes} short />
      <span className="mk-text-caption mk-muted">{totals}</span>
      {runId && <span className="mk-text-data mk-muted">run {runId}</span>}
    </div>
  );
}
