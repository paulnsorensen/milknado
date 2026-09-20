// The `canvas-overlay` slot contribution: an overview of up to 40 visible
// nodes. A minimap jump writes only the store's graph view state (AC-13).
import type { ReactElement } from 'react';
import { useSyncExternalStore } from 'react';
import { getState, setGraphView, subscribe } from '../../app/store';
import { toGraphNodes } from '../../app/wire';
import { Milknado } from '../../design-system';

export const MINIMAP_NODE_LIMIT = 40;

export function MinimapFeature(): ReactElement {
  const state = useSyncExternalStore(subscribe, getState);
  const { Minimap } = Milknado;
  const nodes = state.snapshot?.graph
    ? toGraphNodes(state.snapshot.graph).slice(0, MINIMAP_NODE_LIMIT)
    : [];

  return <Minimap nodes={nodes} selected={state.selection} onJump={(id) => setGraphView({ focus: id })} />;
}
