// Graph-edit intents: the shared add/edit/move/archive dialogs, run
// cancel behind a confirmation, and drag-to-move between cards.
import type { DOMAttributes, DragEvent } from 'react';
import { openDialog } from '../../graph-edits/dialogState';
import { requestConfirm } from '../../run-controls/confirmState';
import { allowed, cancelNodeRun, moveUnder } from '../data/commands';
import { canAdopt, title } from '../model/tree';
import type { DropTarget } from '../viewState';
import { flash, getMh, patchMh } from '../viewState';
import type { MhModel } from './useMhModel';

export type EditKind = 'add' | 'edit' | 'move' | 'archive';

/** Opens a graph-edit dialog; `add` takes the new node's parent. */
export function openEdit(kind: EditKind, nodeId: number | null): void {
  if (allowed('graph_edits')) {
    patchMh({ pop: null });
    openDialog(kind, nodeId);
  }
}

export function openGoalForm(): void {
  if (allowed('coordinator')) {
    patchMh({ goalForm: true, pop: null });
  }
}

export function confirmCancel(model: MhModel, nodeId: number): void {
  const live = model.liveByNode.get(nodeId);
  const node = model.tree.byId.get(nodeId);
  if (!live || !node || !allowed('cancel')) {
    return;
  }
  requestConfirm({
    prompt: `Cancel the run on ${title(node)}?`,
    body: 'The worker stops. Its worktree and branch stay for review.',
    dismissLabel: 'Keep the run',
    confirmLabel: 'Cancel run',
    action: () => void cancelNodeRun(live.run_id).then((ok) => ok && flash(`Run on node ${nodeId} cancelled`)),
  });
}

function parentOf(target: DropTarget): number | null {
  return target === 'root' ? null : target;
}

export function startDrag(event: DragEvent, nodeId: number): void {
  event.stopPropagation();
  event.dataTransfer.effectAllowed = 'move';
  event.dataTransfer.setData('text/plain', String(nodeId));
  patchMh({ drag: { id: nodeId, over: null }, pop: null });
}

export function endDrag(): void {
  if (getMh().drag) {
    patchMh({ drag: null });
  }
}

/** Accepts the drag over `target` when the dragged node can move there. */
export function dragOver(event: DragEvent, model: MhModel, target: DropTarget): void {
  const drag = getMh().drag;
  if (!drag || !canAdopt(model.tree, drag.id, parentOf(target))) {
    return;
  }
  event.preventDefault();
  event.dataTransfer.dropEffect = 'move';
  if (drag.over !== target) {
    patchMh({ drag: { ...drag, over: target } });
  }
}

export function dragLeave(target: DropTarget): void {
  const drag = getMh().drag;
  if (drag?.over === target) {
    patchMh({ drag: { ...drag, over: null } });
  }
}

function movedText(model: MhModel, nodeId: number, parentId: number | null): string {
  const moved = model.tree.byId.get(nodeId);
  const parent = parentId === null ? undefined : model.tree.byId.get(parentId);
  const name = moved ? title(moved) : `Node ${nodeId}`;
  return parent ? `${name} is now a prerequisite of ${title(parent)}` : `${name} is now a top-level goal`;
}

export function dropOn(event: DragEvent, model: MhModel, target: DropTarget): void {
  const drag = getMh().drag;
  patchMh({ drag: null });
  const parentId = parentOf(target);
  if (!drag || !canAdopt(model.tree, drag.id, parentId)) {
    return;
  }
  event.preventDefault();
  event.stopPropagation();
  if (!allowed('graph_edits')) {
    return;
  }
  void moveUnder(drag.id, parentId).then((ok) => ok && flash(movedText(model, drag.id, parentId)));
}

/** Drag handlers for one drop target, spread onto its element. */
type DropProps = Pick<DOMAttributes<HTMLElement>, 'onDragOver' | 'onDragLeave' | 'onDrop'>;

export function dropProps(model: MhModel, target: DropTarget): DropProps {
  return {
    onDragOver: (event) => dragOver(event, model, target),
    onDragLeave: () => dragLeave(target),
    onDrop: (event) => dropOn(event, model, target),
  };
}
