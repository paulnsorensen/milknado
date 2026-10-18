// Composes one chat thread per node from the backend's records: run
// lifecycle, worker session events, reviewer verdicts, coordinator
// records, and the operator notes saved into the node brief.
import type { WireNode } from '../../../app/wire';
import type { WireGoalReview } from '../../goal-review/wire';
import type { WireRunRecord, WireSessionEvent } from '../../../shared/node-detail/detailWire';
import type { WireLiveRun, WireMhDetail, WireNodeReview } from '../wire';
import type { Tree } from './tree';
import { childrenOf, isReady, title } from './tree';

export type MsgKind = 'me' | 'agent' | 'tool' | 'gate';
export type Tone = 'ok' | 'bad' | 'run' | 'mute';

export interface Msg {
  key: string;
  kind: MsgKind;
  who: string;
  text: string;
  tone: Tone;
  at: string;
}

export interface CoordinatorRecord {
  seq: number;
  kind: string;
  text: string;
  status: string;
  tool_name?: string;
}

const NOTE_SEPARATOR = '\n\nOperator note: ';
const MAX_TEXT = 1500;

/** The brief body below the title line, and the operator notes appended after it. */
export function splitBrief(description: string): { brief: string; notes: string[] } {
  const [brief, ...notes] = description.split(NOTE_SEPARATOR);
  const [first, ...rest] = brief.trim().split('\n');
  const body = rest.join('\n').trim();
  return { brief: body || first.trim(), notes: notes.map((note) => note.trim()).filter(Boolean) };
}

export function shortRunId(runId: string): string {
  return runId.replace(/^run-/, '').slice(0, 8);
}

export function workerName(nodeId: number): string {
  return `worker-${nodeId}`;
}

export function ago(iso: string | null | undefined, now: number): string {
  const time = iso ? Date.parse(iso) : NaN;
  if (Number.isNaN(time)) {
    return '';
  }
  const minutes = Math.max(0, Math.round((now - time) / 60000));
  if (minutes < 1) {
    return 'now';
  }
  if (minutes < 60) {
    return `${minutes}m`;
  }
  const hours = Math.round(minutes / 60);
  return hours < 48 ? `${hours}h` : `${Math.round(hours / 24)}d`;
}

function clip(text: string, limit = MAX_TEXT): string {
  const trimmed = text.trim();
  return trimmed.length > limit ? `${trimmed.slice(0, limit - 1)}…` : trimmed;
}

function msg(key: string, kind: MsgKind, who: string, text: string, tone: Tone = 'mute', at = ''): Msg {
  return { key, kind, who, text, tone, at };
}

function gateTone(text: string): Tone {
  if (/\bFAIL/i.test(text)) {
    return 'bad';
  }
  return /\bPASS/i.test(text) ? 'ok' : 'run';
}

/** A tool call that runs a configured quality gate renders as a gate pill. */
export function eventMsg(event: WireSessionEvent, who: string, key: string, gates: string[] = []): Msg | null {
  const text = event.text ?? '';
  if (!text.trim()) {
    return null;
  }
  switch (event.kind) {
    case 'assistant':
      return msg(key, 'agent', who, clip(text));
    case 'tool': {
      const line = clip(text.split('\n', 1)[0], 160);
      return gates.some((gate) => line.startsWith(gate))
        ? msg(key, 'gate', 'gate', line, gateTone(line))
        : msg(key, 'tool', who, line);
    }
    case 'user':
      return msg(key, 'me', 'you', clip(text));
    case 'error':
      return msg(key, 'gate', who, clip(text, 200), 'bad');
    case 'permission':
      return msg(key, 'agent', who, `Asks permission: ${clip(text, 400)}`);
    default:
      return msg(key, 'gate', who, clip(text, 200), 'mute');
  }
}

function runEnd(run: WireRunRecord, now: number): Msg | null {
  const at = ago(run.ended_at, now);
  if (run.status === 'done') {
    return msg(`${run.run_id}:end`, 'gate', 'gate', 'Gate PASS · run done', 'ok', at);
  }
  if (run.status === 'failed') {
    const cancelled = run.error === 'cancelled';
    const text = cancelled ? 'Run cancelled' : `Gate FAIL · ${clip(run.error ?? 'run failed', 160)}`;
    return msg(`${run.run_id}:end`, 'gate', 'gate', text, cancelled ? 'mute' : 'bad', at);
  }
  return null;
}

function reviewMsg(review: WireNodeReview, now: number): Msg {
  const verdict = review.verdict.charAt(0).toUpperCase() + review.verdict.slice(1);
  const findings = review.findings.trim() ? ` ${clip(review.findings, 800)}` : '';
  return msg(`review:${review.round}:${review.created_at}`, 'agent', 'reviewer',
    `${verdict} · round ${review.round}.${findings}`, 'mute', ago(review.created_at, now));
}

function runEvents(run: WireRunRecord, detail: WireMhDetail | null, live: WireLiveRun | undefined): WireSessionEvent[] {
  if (live?.run_id === run.run_id && live.session) {
    return live.session.events;
  }
  const session = detail?.sessions.items?.find((item) => item.run_id === run.run_id);
  return session?.event_history.items ?? session?.session?.events ?? [];
}

/** Sorts after every ISO timestamp; `ago` renders it as an empty age. */
const AFTER_ALL = '￿';

function byTime<T>(items: T[], at: (item: T) => string | null): T[] {
  return [...items].sort((a, b) => (at(a) ?? '').localeCompare(at(b) ?? ''));
}

export interface TaskThreadInput {
  node: WireNode;
  detail: WireMhDetail | null;
  live: WireLiveRun | undefined;
  now: number;
  gates?: string[];
}

export function taskThread({ node, detail, live, now, gates = [] }: TaskThreadInput): Msg[] {
  const who = workerName(node.id);
  const runs = byTime(detail?.runs.items ?? [], (run) => run.started_at);
  if (live && !runs.some((run) => run.run_id === live.run_id)) {
    // Node detail can lag a new dispatch. The live run starts after every
    // recorded review, so it shows none and earlier runs keep their own.
    runs.push({ run_id: live.run_id, node_id: node.id, status: 'running', started_at: AFTER_ALL, ended_at: null, error: null });
  }
  const reviews = byTime(detail?.reviews.items ?? [], (review) => review.created_at);
  const thread: Msg[] = [];
  runs.forEach((run, index) => {
    thread.push(msg(`${run.run_id}:start`, 'gate', 'gate', `Run ${shortRunId(run.run_id)} dispatched`, 'run', ago(run.started_at, now)));
    runEvents(run, detail, live).forEach((event, i) => {
      const message = eventMsg(event, who, `${run.run_id}:${event.event_id || i}`, gates);
      if (message) {
        thread.push(message);
      }
    });
    const end = runEnd(run, now);
    if (end) {
      thread.push(end);
    }
    const nextStart = runs[index + 1]?.started_at ?? AFTER_ALL;
    reviews.filter((r) => r.created_at >= run.started_at && r.created_at < nextStart)
      .forEach((review) => thread.push(reviewMsg(review, now)));
  });
  splitBrief(node.description).notes.forEach((note, i) => thread.push(msg(`note:${i}`, 'me', 'you', note)));
  if (thread.length === 0) {
    thread.push(msg('idle', 'gate', 'gate', 'Not dispatched yet · brief the worker before it starts'));
  }
  return thread;
}

export interface GoalThreadInput {
  node: WireNode;
  tree: Tree;
  reviews: WireGoalReview[];
  records: CoordinatorRecord[];
}

export function goalThread({ node, tree, reviews, records }: GoalThreadInput): Msg[] {
  const kids = childrenOf(tree, node.id);
  const done = kids.filter((kid) => kid.status === 'done').length;
  const running = kids.filter((kid) => kid.status === 'running').length;
  const ready = kids.filter((kid) => isReady(tree, kid)).length;
  const parts = [`${title(node)}: ${done} of ${kids.length} prerequisites done.`];
  if (running || ready) {
    parts.push(`${running} running, ${ready} ready for a worker.`);
  }
  const thread = [msg('summary', 'agent', 'coordinator', parts.join(' '))];
  for (const record of records.slice(-40)) {
    if (!record.text.trim()) {
      continue;
    }
    const key = `record:${record.seq}`;
    thread.push(record.tool_name
      ? msg(key, 'tool', 'coordinator', clip(`${record.tool_name} ${record.text}`, 160))
      : msg(key, 'agent', 'coordinator', clip(record.text)));
  }
  for (const review of reviews.filter((r) => r.goal_id === node.id && r.decision === 'pending')) {
    thread.push(msg(`goal-review:${review.review_id}:evidence`, 'agent', 'reviewer', clip(review.evidence)));
    thread.push(msg(`goal-review:${review.review_id}`, 'agent', 'coordinator',
      `Proposed change: ${clip(review.proposed_change)} Accept it, or reject and keep the current goal?`));
  }
  splitBrief(node.description).notes.forEach((note, i) => thread.push(msg(`note:${i}`, 'me', 'you', note)));
  return thread;
}
