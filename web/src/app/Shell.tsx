import type { ReactElement } from 'react';
import { useEffect, useSyncExternalStore } from 'react';
import { Milknado } from '../design-system';
import { get } from './api';
import { DialogHost } from './DialogHost';
import { CanvasBanner, CanvasOverlay, CanvasToolbar } from './hosts/CanvasChromeHost';
import { DockHost } from './hosts/DockHost';
import { HeaderHost } from './hosts/HeaderHost';
import { ProviderHost } from './hosts/ProviderHost';
import { RailHost } from './hosts/RailHost';
import { SidecarHost } from './hosts/SidecarHost';
import { ToastHost } from './hosts/ToastHost';
import { registerFeatures } from './registry';
import './shell.css';
import { getSlot } from './slots';
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

/** The default Main artboard region tree: header, rail, canvas, dock and sidecar. */
function DefaultLayout(): ReactElement {
  const state = useSyncExternalStore(subscribe, getState);
  const { MikadoGraph } = Milknado;

  const nodes = state.snapshot?.graph ? toGraphNodes(state.snapshot.graph) : [];

  return (
    <>
      <HeaderHost />
      <RailHost />
      <div data-region="canvas">
        <CanvasBanner />
        <CanvasToolbar />
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
        <CanvasOverlay />
      </div>
      <DockHost />
      <SidecarHost />
    </>
  );
}

/** The Main artboard: header, rail, canvas, dock and sidecar regions. */
export function Shell(): ReactElement {
  useEffect(loadSnapshot, []);

  const layouts = getSlot('layout');

  return (
    <div className="mk-shell">
      <ProviderHost />
      {layouts.length > 0 ? layouts[0]() : <DefaultLayout />}
      <DialogHost />
      <ToastHost />
    </div>
  );
}
