// Every write the metaharness view makes. Each call goes through
// `app/api`, so a 409 surfaces the server's reason as a notice.
import { post } from '../../../app/api';
import { getState, pushNotice } from '../../../app/store';
import type { WireCapability, WireNode } from '../../../app/wire';
import { moveNode } from '../../graph-edits/commands';
import { cancelRun } from '../../run-controls/commands';
import type { WireLiveRun } from '../wire';
import { loadInfo, setPaused } from './info';

type Gated = 'graph_edits' | 'cancel' | 'coordinator';

/** True when the server offers `name`; otherwise shows the server's reason. */
export function allowed(name: Gated): boolean {
  const capability: WireCapability | undefined = getState().capabilities?.[name];
  if (capability?.available) {
    return true;
  }
  pushNotice(capability?.reason ?? 'This action is unavailable.');
  return false;
}

type SessionAction = 'steer' | 'approve' | 'deny';

export function canSteer(run: WireLiveRun | undefined): run is WireLiveRun {
  const capability = getState().capabilities?.session_input;
  return !!run && !!capability?.available && (run.session?.actions ?? []).includes('steer');
}

async function sessionInput(run: WireLiveRun, action: SessionAction, text = '', requestId = ''): Promise<boolean> {
  const owner = getState().capabilities?.owner;
  const result = await post(`/api/runs/${encodeURIComponent(run.run_id)}/session-input`, {
    action,
    text,
    request_id: requestId,
    command_id: crypto.randomUUID(),
    owner_incarnation: run.session?.owner_incarnation || String(owner?.owner_incarnation ?? ''),
    invocation_id: run.session?.invocation_id || (owner?.invocation_id ?? ''),
  });
  return result !== null;
}

export function steer(run: WireLiveRun, text: string): Promise<boolean> {
  return sessionInput(run, 'steer', text);
}

export function answerPermission(run: WireLiveRun, permissionId: string, approve: boolean): Promise<boolean> {
  return sessionInput(run, approve ? 'approve' : 'deny', '', permissionId);
}

export async function addNote(nodeId: number, text: string): Promise<boolean> {
  return (await post(`/api/nodes/${nodeId}/note`, { text })) !== null;
}

export async function retryNode(nodeId: number, note: string | null = null): Promise<boolean> {
  return (await post(`/api/nodes/${nodeId}/retry`, { note })) !== null;
}

export async function skipNode(nodeId: number): Promise<boolean> {
  return (await post(`/api/nodes/${nodeId}/skip`, { note: null })) !== null;
}

export async function decideGoalReview(reviewId: number, decision: 'accepted' | 'rejected'): Promise<boolean> {
  return (await post(`/api/reviews/${reviewId}/decision`, { decision })) !== null;
}

export async function queueFix(text: string, parent: WireNode | undefined): Promise<boolean> {
  const body = { description: `Fix: ${text}`, parent_id: parent?.id ?? null, kind: 'task' };
  return (await post('/api/nodes', body)) !== null;
}

export async function setLoopPaused(paused: boolean): Promise<boolean> {
  const result = await post<{ paused: boolean }>(`/api/scheduling/${paused ? 'pause' : 'resume'}`);
  if (result === null) {
    return false;
  }
  setPaused(result.paused);
  void loadInfo();
  return true;
}

export async function moveUnder(nodeId: number, parentId: number | null): Promise<boolean> {
  return (await moveNode(nodeId, parentId)) !== null;
}

export async function cancelNodeRun(runId: string): Promise<boolean> {
  return (await cancelRun(encodeURIComponent(runId))) !== null;
}

interface Receipt {
  status: string;
  result: unknown;
}

/** Creates a goal and its coordinator session through `start_goal`. */
export async function startGoal(description: string, provider: 'claude' | 'codex'): Promise<boolean> {
  const receipt = await post<Receipt>('/api/coordinators/commands', {
    kind: 'start_goal', command_id: crypto.randomUUID(), description, provider,
  });
  if (receipt && receipt.status !== 'accepted') {
    pushNotice(String(receipt.result ?? receipt.status));
  }
  return receipt?.status === 'accepted';
}
