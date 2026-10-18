// The Mikado subtree as positioned cards joined by curved edges. The
// compact variant lives in the left pane; the expanded one is the overlay.
import type { MouseEvent, ReactElement } from 'react';
import type { WireNode } from '../../../app/wire';
import type { GraphLayout, PlacedNode } from '../model/layout';
import { childrenOf, isGoal, title } from '../model/tree';
import { patchMh } from '../viewState';
import { selectNode } from './actions';
import { dropProps, endDrag, openEdit, startDrag } from './edits';
import { Icon } from './Icon';
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

function cardClass(node: WireNode, model: MhModel): string {
  const { sel, drag } = model.ui;
  return [
    'mh-node', stateClass(node, model),
    sel === node.id ? 'is-selected' : '',
    drag?.id === node.id ? 'is-dragging' : '',
    drag?.over === node.id ? 'is-drop-target' : '',
  ].filter(Boolean).join(' ');
}

function AddButton({ node }: { node: WireNode }): ReactElement {
  return (
    <button type="button" className="mh-node-add" title="Add a prerequisite" draggable={false}
      aria-label={`Add a prerequisite to ${title(node)}`}
      onClick={(event) => { event.stopPropagation(); openEdit('add', node.id); }}>
      <Icon name="plus" size={12} />
    </button>
  );
}

function cardTitle(node: WireNode, model: MhModel): string {
  const focus = isGoal(node) && childrenOf(model.tree, node.id).length > 0 ? ' · double-click to focus' : '';
  return `${title(node)}${focus}${model.editable ? ' · drag onto another node to move it' : ''}`;
}

function Card({ placed, node, model, mode, fold }: CardProps): ReactElement {
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
    <div data-mh-node={node.id} data-mode={mode} role="button" tabIndex={0} className={cardClass(node, model)}
      title={cardTitle(node, model)} aria-label={`${title(node)} · ${nodeMeta(node, model)}`}
      style={{ left: placed.left, top: placed.top, width: placed.w, height: placed.h }}
      onClick={onClick} onDoubleClick={onDoubleClick} draggable={model.editable}
      onDragStart={(event) => startDrag(event, node.id)} onDragEnd={endDrag} {...dropProps(model, node.id)}
      onKeyDown={(event) => { if (event.key === 'Enter') { selectNode(model, node.id, event.currentTarget); } }}>
      {model.editable && <AddButton node={node} />}
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
