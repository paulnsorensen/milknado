// The `toolbar` slot contribution. Every control writes only the store's
// graph view state — none sends a command or an API write (AC-13). "Hide
// done" is local UI state: `GraphView` (store.ts) has no field for it, and
// wiring it into the canvas would require editing the shell's MikadoGraph
// call, both out of this feature's scope.
import type { ReactElement } from 'react';
import { useState, useSyncExternalStore } from 'react';
import { getState, setGraphView, subscribe } from '../../app/store';
import { toGraphNodes } from '../../app/wire';
import { Milknado } from '../../design-system';
import { clampZoom, collapsibleIds, ZOOM_STEP } from './toolbarActions';

export function GraphToolbarFeature(): ReactElement {
  const state = useSyncExternalStore(subscribe, getState);
  const [hideDone, setHideDone] = useState(false);
  const { GraphToolbar } = Milknado;
  const nodes = state.snapshot?.graph ? toGraphNodes(state.snapshot.graph) : [];
  const { graphView, selection } = state;
  const zoom = graphView.zoom ?? 1;

  return (
    <GraphToolbar
      nodes={nodes}
      filter={graphView.filter}
      onFilter={(filter) => setGraphView({ filter })}
      hideDone={hideDone}
      onHideDone={setHideDone}
      focus={graphView.focus != null && graphView.focus === selection}
      canFocus={selection != null}
      onFocus={(active) => setGraphView({ focus: active ? selection : null })}
      lod={graphView.lod}
      onLod={(lod) => setGraphView({ lod })}
      onJump={(id) => setGraphView({ focus: id })}
      onCollapseAll={() => setGraphView({ collapsed: collapsibleIds(nodes) })}
      onExpandAll={() => setGraphView({ collapsed: [] })}
      onZoomIn={() => setGraphView({ zoom: clampZoom(zoom + ZOOM_STEP) })}
      onZoomOut={() => setGraphView({ zoom: clampZoom(zoom - ZOOM_STEP) })}
      onFit={() => setGraphView({ zoom: 1 })}
    />
  );
}
