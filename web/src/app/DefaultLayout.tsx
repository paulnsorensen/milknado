// The default Main artboard region tree: header, rail, canvas, dock and
// sidecar. Shared by `Shell.tsx`'s fallback and the narrow feature's
// `WideLayout`, so wide viewports render one region tree either way.
import type { ReactElement } from 'react';
import { useSyncExternalStore } from 'react';
import { Milknado } from '../design-system';
import { CanvasBanner, CanvasOverlay, CanvasToolbar } from './hosts/CanvasChromeHost';
import { DockHost } from './hosts/DockHost';
import { HeaderHost } from './hosts/HeaderHost';
import { RailHost } from './hosts/RailHost';
import { SidecarHost } from './hosts/SidecarHost';
import { getState, setGraphView, setSelection, subscribe } from './store';
import { toGraphNodes } from './wire';

export function DefaultLayout(): ReactElement {
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
