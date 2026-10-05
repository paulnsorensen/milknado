// The default Main artboard region tree: rail | main (header, toolbar,
// canvas, dock) | sidecar. Shared by `Shell.tsx`'s fallback and the narrow
// feature's `WideLayout`, so wide viewports render one region tree either way.
import type { ReactElement } from 'react';
import { useRef, useSyncExternalStore } from 'react';
import { Milknado } from '../design-system';
import { CanvasBanner, CanvasOverlay, CanvasToolbar } from './hosts/CanvasChromeHost';
import { DockHost } from './hosts/DockHost';
import { HeaderHost } from './hosts/HeaderHost';
import { RailHost } from './hosts/RailHost';
import { SidecarHost } from './hosts/SidecarHost';
import { useRegionSize } from './hosts/useRegionSize';
import { getState, setGraphView, setSelection, subscribe } from './store';
import { toGraphNodes } from './wire';

export function DefaultLayout(): ReactElement {
  const state = useSyncExternalStore(subscribe, getState);
  const canvasRef = useRef<HTMLDivElement | null>(null);
  const canvasSize = useRegionSize(canvasRef);
  const { MikadoGraph } = Milknado;

  const graph = state.coordinatorGraph ?? state.snapshot?.graph;
  const nodes = graph ? toGraphNodes(graph) : [];

  return (
    <>
      <RailHost />
      <main data-region="main">
        <HeaderHost />
        <CanvasBanner />
        <CanvasToolbar />
        <div data-region="canvas" ref={canvasRef}>
          <MikadoGraph
            nodes={nodes}
            selected={state.selection}
            onSelect={setSelection}
            focus={state.graphView.focus}
            filter={state.graphView.filter}
            lod={state.graphView.lod}
            zoom={state.graphView.zoom}
            collapsed={state.graphView.collapsed}
            onLayout={(layout) => setGraphView({ lod: layout.lod })}
            compact
            autoLod={false}
            width={canvasSize?.width}
            height={canvasSize?.height ?? 'auto'}
          />
          <CanvasOverlay />
        </div>
        <DockHost />
      </main>
      <SidecarHost />
    </>
  );
}