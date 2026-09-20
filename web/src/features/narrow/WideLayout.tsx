// The default Main artboard region tree, rendered above the 400px
// breakpoint. Mirrors `Shell.tsx`'s (unexported) `DefaultLayout` so the
// `layout` slot contribution can fall back to it at wide viewports.
import type { ReactElement } from 'react';
import { useSyncExternalStore } from 'react';
import { CanvasBanner, CanvasOverlay, CanvasToolbar } from '../../app/hosts/CanvasChromeHost';
import { DockHost } from '../../app/hosts/DockHost';
import { HeaderHost } from '../../app/hosts/HeaderHost';
import { RailHost } from '../../app/hosts/RailHost';
import { SidecarHost } from '../../app/hosts/SidecarHost';
import { getState, setGraphView, setSelection, subscribe } from '../../app/store';
import { Milknado } from '../../design-system';
import { toGraphNodes } from '../../app/wire';

export function WideLayout(): ReactElement {
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
