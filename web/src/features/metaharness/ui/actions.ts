// View-level intents: select a node, open its decision, resolve a need.
import { answerPermission, decideGoalReview, retryNode, skipNode } from '../data/commands';
import type { Need } from '../model/needs';
import { nextNeed } from '../model/needs';
import { rootOf } from '../model/tree';
import { flash, getMh, patchMh } from '../viewState';
import type { MhModel } from './useMhModel';

export type NeedAction = 'approve' | 'deny' | 'accept' | 'reject' | 'retry' | 'note' | 'skip' | 'reply';

const POPOVER_HEIGHT = 300;

export function focusComposer(): void {
  setTimeout(() => document.querySelector<HTMLInputElement>('[data-mh-composer]')?.focus(), 30);
}

export function anchorPopover(id: number, element?: Element | null): void {
  const target = element ?? document.querySelector(`[data-mh-node="${id}"][data-mode="compact"]`);
  if (!target) {
    return;
  }
  const rect = target.getBoundingClientRect();
  const y = Math.max(12, Math.min(rect.top - 8, window.innerHeight - POPOVER_HEIGHT));
  patchMh({ pop: { id, x: rect.right + 12, y } });
}

export function selectNode(model: MhModel, id: number, element?: Element): void {
  const projectId = rootOf(model.tree, id);
  const sameProject = projectId === getMh().projectId;
  patchMh({
    sel: id,
    view: 'thread',
    pop: null,
    projectId,
    scopeRoot: sameProject ? getMh().scopeRoot : projectId,
  });
  if (model.needs.has(id)) {
    if (element) {
      anchorPopover(id, element);
    } else {
      requestAnimationFrame(() => anchorPopover(id));
    }
  }
}

export function nextDecision(model: MhModel): void {
  const need = nextNeed(model.needs, getMh().sel);
  if (need) {
    patchMh({ expanded: false, scopeRoot: rootOf(model.tree, need.nodeId) });
    selectNode(model, need.nodeId);
  }
}

async function run(need: Need, action: NeedAction, model: MhModel): Promise<boolean> {
  const live = model.liveByNode.get(need.nodeId);
  switch (action) {
    case 'approve':
    case 'deny':
      return !!live && !!need.permissionId && answerPermission(live, need.permissionId, action === 'approve');
    case 'accept':
    case 'reject':
      return need.reviewId !== undefined && decideGoalReview(need.reviewId, action === 'accept' ? 'accepted' : 'rejected');
    case 'retry':
      return retryNode(need.nodeId);
    case 'skip':
      return skipNode(need.nodeId);
    default:
      return false;
  }
}

const DONE_TEXT: Partial<Record<NeedAction, (id: number) => string>> = {
  approve: (id) => `Approved the request on node ${id}`,
  deny: (id) => `Denied the request on node ${id}`,
  accept: (id) => `Accepted the goal change on node ${id}`,
  reject: (id) => `Rejected the goal change on node ${id}`,
  retry: (id) => `Node ${id} queued for retry`,
  skip: (id) => `Node ${id} skipped and marked done`,
};

export async function resolveNeed(need: Need, action: NeedAction, model: MhModel): Promise<void> {
  patchMh({ pop: null, sel: need.nodeId, view: 'thread', compose: action === 'note' ? 'retry' : 'message' });
  if (action === 'note' || action === 'reply') {
    focusComposer();
    return;
  }
  if (await run(need, action, model)) {
    flash(DONE_TEXT[action]?.(need.nodeId) ?? 'Done');
  }
}
