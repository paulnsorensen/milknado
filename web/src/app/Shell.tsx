import type { ReactElement } from 'react';
import { useEffect, useSyncExternalStore } from 'react';
import { Milknado } from '../design-system';
import { get } from './api';
import { DialogHost } from './DialogHost';
import { registerFeatures } from './registry';
import './shell.css';
import { getState, setGraphView, setSelection, setSnapshot, subscribe } from './store';
import { toGraphNodes, type WireExecutionSnapshot } from './wire';

registerFeatures();

function loadSnapshot(): void {
  void get<WireExecutionSnapshot>('/api/snapshot').then((snapshot) => {
    if (snapshot) {
      setSnapshot(snapshot);
    }
  });
}

/** The Main artboard: header, rail, canvas, dock and sidecar regions. */
export function Shell(): ReactElement {
  const state = useSyncExternalStore(subscribe, getState);
  const { MikadoGraph } = Milknado;

  useEffect(loadSnapshot, []);

  const nodes = state.snapshot?.graph ? toGraphNodes(state.snapshot.graph) : [];

  return (
    <div className="mk-shell">
      <div data-region="header" />
      <div data-region="rail" />
      <div data-region="canvas">
        <MikadoGraph
          nodes={nodes}
          selected={state.selection}
          onSelect={setSelection}
          focus={state.graphView.focus}
          filter={state.graphView.filter}
          lod={state.graphView.lod}
          zoom={state.graphView.zoom}
          collapsed={state.graphView.collapsed}
          onLayout={(layout) => setGraphView({ lod: layout.lod, collapsed: layout.collapsed })}
          height="auto"
        />
      </div>
      <div data-region="dock" />
      <DialogHost />
    </div>
  );
}
