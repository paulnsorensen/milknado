// The `toolbar` slot contribution. Every control writes only the store's
// graph view state — none sends a command or an API write (AC-13). "Hide
// done" is local UI state: `GraphView` (store.ts) has no field for it, and
// wiring it into the canvas would require editing the shell's MikadoGraph
// call, both out of this feature's scope.
import type { ReactElement } from 'react';
import { useState, useSyncExternalStore } from 'react';
import { visibleGraphNodes } from '../../app/graphMode';
import { getState, setGraphView, subscribe, type GraphMode } from '../../app/store';
import { Milknado } from '../../design-system';
import { clampZoom, collapsibleIds, ZOOM_STEP } from './toolbarActions';

const MODES: Array<{ id: GraphMode; label: string }> = [
  { id: 'execution', label: 'Execution' },
  { id: 'roadmap', label: 'Roadmap' },
];

// Collapse and focus name nodes of one mode, so a mode switch clears them.
function GraphModeSwitch({ mode }: { mode: GraphMode }): ReactElement {
  return (
    <div className="mk mk-seg" role="group" aria-label="Graph mode">
      {MODES.map((option) => (
        <button
          key={option.id}
          type="button"
          className={mode === option.id ? 'mk-seg-opt is-on' : 'mk-seg-opt'}
          aria-pressed={mode === option.id}
          onClick={() => setGraphView({ mode: option.id, collapsed: [], focus: null })}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

export function GraphToolbarFeature(): ReactElement {
  const state = useSyncExternalStore(subscribe, getState);
  const [hideDone, setHideDone] = useState(false);
  const { GraphToolbar } = Milknado;
  const nodes = visibleGraphNodes(state);
  const { graphView, selection } = state;
  const zoom = graphView.zoom ?? 1;
  const canFocus = selection != null && nodes.some((node) => node.id === selection);

  return (
    <div className="mk-graph-toolbar-row">
      <GraphModeSwitch mode={graphView.mode} />
      <GraphToolbar
        nodes={nodes}
        filter={graphView.filter}
        onFilter={(filter) => setGraphView({ filter })}
        hideDone={hideDone}
        onHideDone={setHideDone}
        focus={canFocus && graphView.focus === selection}
        canFocus={canFocus}
        onFocus={(active) => setGraphView({ focus: active && canFocus ? selection : null })}
        lod={graphView.lod}
        onLod={(lod) => setGraphView({ lod })}
        onJump={(id) => setGraphView({ focus: id })}
        onCollapseAll={() => setGraphView({ collapsed: collapsibleIds(nodes) })}
        onExpandAll={() => setGraphView({ collapsed: [] })}
        onZoomIn={() => setGraphView({ zoom: clampZoom(zoom + ZOOM_STEP) })}
        onZoomOut={() => setGraphView({ zoom: clampZoom(zoom - ZOOM_STEP) })}
        onFit={() => setGraphView({ zoom: 1 })}
      />
    </div>
  );
}
