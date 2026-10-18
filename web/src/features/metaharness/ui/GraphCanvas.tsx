// The Mikado subtree as positioned cards joined by curved edges. The
// compact variant lives in the left pane; the expanded one is the overlay.
import type { MouseEvent, ReactElement } from 'react';
import type { WireNode } from '../../../app/wire';
import type { GraphLayout, PlacedNode } from '../model/layout';
import { childrenOf, isGoal, title } from '../model/tree';
import { patchMh } from '../viewState';
import { selectNode } from './actions';
import { nodeMeta, runPercent, stateClass } from './nodeView';
import type { MhModel } from './useMhModel';

export type CanvasMode = 'compact' | 'expanded';

function kindLabel(node: WireNode, model: MhModel): string {
  if (isGoal(node)) {
    return node.kind;
  }
  const flavor = node.flavor ?? 'task';
  return model.liveByNode.has(node.id) ? `${flavor} · w${node.id}` : flavor;
}

interface CardProps {
  placed: PlacedNode;
  node: WireNode;
  model: MhModel;
  mode: CanvasMode;
  fold: boolean;
}

function FoldToggle({ placed }: { placed: PlacedNode }): ReactElement {
  const toggle = (event: MouseEvent): void => {
    event.stopPropagation();
    patchMh((ui) => ({ unfold: { ...ui.unfold, [placed.id]: !ui.unfold[placed.id] } }));
  };
  const folded = placed.folded > 0;
  return (
    <button type="button" className="mh-fold" onClick={toggle}
      title={folded ? 'Show done prerequisites' : 'Fold done prerequisites'}>
      {folded ? `✓ ${placed.folded}` : 'fold'}
    </button>
  );
}

function Card({ placed, node, model, mode, fold }: CardProps): ReactElement {
  const selected = model.ui.sel === node.id;
  const running = node.status === 'running' && !model.needs.has(node.id);
  const onClick = (event: MouseEvent<HTMLDivElement>): void => {
    event.stopPropagation();
    if (mode === 'expanded') {
      patchMh({ expanded: false });
      selectNode(model, node.id);
      return;
    }
    selectNode(model, node.id, event.currentTarget);
  };
  const onDoubleClick = (event: MouseEvent): void => {
    event.stopPropagation();
    if (isGoal(node) && childrenOf(model.tree, node.id).length > 0) {
      patchMh({ scopeRoot: node.id, sel: node.id, expanded: false, pop: null });
    }
  };
  return (
    <div data-mh-node={node.id} data-mode={mode} role="button" tabIndex={0}
      className={`mh-node ${stateClass(node, model)}${selected ? ' is-selected' : ''}`}
      title={isGoal(node) ? `${title(node)} · double-click to focus` : title(node)}
      style={{ left: placed.left, top: placed.top, width: placed.w, height: placed.h }}
      onClick={onClick} onDoubleClick={onDoubleClick}
      onKeyDown={(event) => { if (event.key === 'Enter') { selectNode(model, node.id, event.currentTarget); } }}>
      {mode === 'expanded' && (
        <div className="mh-node-kind"><span className="mh-dot" /><span className="mh-node-kind-text">{kindLabel(node, model)}</span></div>
      )}
      <div className="mh-node-head">
        {mode === 'compact' && <span className="mh-dot" />}
        <span className="mh-node-title">{title(node)}</span>
      </div>
      <div className="mh-node-meta">
        <span>{nodeMeta(node, model)}</span>
        {fold && placed.foldable > 0 && <FoldToggle placed={placed} />}
      </div>
      {running && <Progress pct={runPercent(model.liveByNode.get(node.id))} />}
    </div>
  );
}

function Progress({ pct }: { pct: number | null }): ReactElement {
  return (
    <div className={`mh-node-progress${pct === null ? ' is-indeterminate' : ''}`}>
      <div style={{ width: pct === null ? '100%' : `${pct}%` }} />
    </div>
  );
}
function edgeClass(node: WireNode | undefined, model: MhModel): string {
  if (!node) {
    return '';
  }
  return model.needs.has(node.id) ? 'is-need' : `is-${node.status}`;
}

interface CanvasProps {
  layout: GraphLayout;
  model: MhModel;
  mode: CanvasMode;
  fold: boolean;
}

export function GraphCanvas({ layout, model, mode, fold }: CanvasProps): ReactElement {
  return (
    <div className={`mh-canvas is-${mode}`} style={{ width: layout.width, height: layout.height }}>
      <svg width={layout.width} height={layout.height} className="mh-edges" aria-hidden="true">
        {layout.edges.map((edge) => (
          <path key={edge.id} d={edge.d} className={`mh-edge ${edgeClass(model.tree.byId.get(edge.id), model)}`} />
        ))}
      </svg>
      {layout.nodes.map((placed) => {
        const node = model.tree.byId.get(placed.id);
        return node ? <Card key={placed.id} placed={placed} node={node} model={model} mode={mode} fold={fold} /> : null;
      })}
    </div>
  );
}
