// Facts about the selected node shared by the chat header, the composer,
// and the details panel: who answers it and where a message goes.
import type { WireNode } from '../../../app/wire';
import { canSteer } from '../data/commands';
import type { NodeData } from '../data/nodeData';
import type { NeedAction } from './actions';
import type { Need, NeedKind } from '../model/needs';
import { isGoal, title } from '../model/tree';
import { workerName } from '../model/thread';
import type { WireLiveRun } from '../wire';
import type { MhModel } from './useMhModel';

export type Delivery = 'steer' | 'retry' | 'note';

export interface Current {
  node: WireNode;
  who: string;
  need: Need | undefined;
  live: WireLiveRun | undefined;
  delivery: Delivery;
  runId: string | null;
}

export function currentNode(model: MhModel): WireNode | undefined {
  const { sel, projectId } = model.ui;
  return model.tree.byId.get(sel ?? projectId ?? -1) ?? model.tree.roots[0];
}

export function latestRunId(data: NodeData, nodeId: number): string | null {
  if (data.nodeId !== nodeId) {
    return null;
  }
  const runs = [...(data.detail?.runs.items ?? [])].sort((a, b) => a.started_at.localeCompare(b.started_at));
  return runs.at(-1)?.run_id ?? null;
}

export function describe(model: MhModel, node: WireNode, data: NodeData): Current {
  const live = model.liveByNode.get(node.id);
  const runId = live?.run_id ?? latestRunId(data, node.id);
  const who = isGoal(node) ? 'coordinator' : runId ? workerName(node.id) : 'unassigned';
  const need = model.needs.get(node.id);
  const delivery: Delivery = model.ui.compose === 'retry' && need?.kind === 'failed'
    ? 'retry' : canSteer(live) ? 'steer' : 'note';
  return { node, who, need, live, delivery, runId };
}

export const DELIVERY_TEXT: Record<Delivery, string> = {
  steer: 'delivered on next poll',
  retry: 'retries the node with your note',
  note: 'saved to the brief',
};

export function placeholder(current: Current): string {
  const { node, who, delivery } = current;
  if (delivery === 'retry') {
    return `Note for the retry of ${title(node)}…`;
  }
  if (isGoal(node)) {
    return `Message the coordinator about ${title(node)}…`;
  }
  return delivery === 'steer' ? `Redirect ${who}…` : `Message ${who === 'unassigned' ? `node ${node.id}` : who}…`;
}

export function initials(who: string): string {
  if (who === 'coordinator') {
    return 'co';
  }
  if (who === 'reviewer') {
    return 'rv';
  }
  return who.startsWith('worker-') ? `w${who.slice(7)}` : '—';
}

export function avatarClass(who: string): string {
  if (who === 'coordinator' || who === 'reviewer') {
    return `is-${who}`;
  }
  return who.startsWith('worker') ? 'is-worker' : 'is-none';
}

export interface ActionSpec {
  label: string;
  tone: 'primary' | 'secondary' | 'ghost' | 'danger';
  action: NeedAction;
}

export const NEED_ACTIONS: Record<NeedKind, ActionSpec[]> = {
  permission: [
    { label: 'Approve', tone: 'primary', action: 'approve' },
    { label: 'Deny', tone: 'danger', action: 'deny' },
    { label: 'Reply…', tone: 'ghost', action: 'reply' },
  ],
  review: [
    { label: 'Accept change', tone: 'primary', action: 'accept' },
    { label: 'Reject change', tone: 'secondary', action: 'reject' },
    { label: 'Reply…', tone: 'ghost', action: 'reply' },
  ],
  failed: [
    { label: 'Retry', tone: 'primary', action: 'retry' },
    { label: 'Retry with note', tone: 'secondary', action: 'note' },
    { label: 'Skip', tone: 'ghost', action: 'skip' },
  ],
};

/** The failure reason from the latest failed run, or the need's generic summary. */
export function needSummary(need: Need, data: NodeData): string {
  if (need.kind !== 'failed' || data.nodeId !== need.nodeId) {
    return need.summary;
  }
  const failed = (data.detail?.runs.items ?? []).filter((run) => run.status === 'failed' && run.error);
  const latest = failed.sort((a, b) => a.started_at.localeCompare(b.started_at)).at(-1);
  return latest?.error ? `Last run failed: ${latest.error}` : need.summary;
}
