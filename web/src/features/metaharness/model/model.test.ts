import { describe, expect, it } from 'vitest';
import type { WireGraphSnapshot, WireNode } from '../../../app/wire';
import type { WireGoalReview } from '../../goal-review/wire';
import { parseUnifiedDiff } from '../data/runDiff';
import type { WireLiveRun, WireMhDetail, WireMhSnapshot } from '../wire';
import { deriveIssues, deriveStats, projectCards } from './insights';
import { layoutTree } from './layout';
import { deriveNeeds, nextNeed } from './needs';
import { goalThread, splitBrief, taskThread } from './thread';
import { ancestry, buildTree, canAdopt, isReady } from './tree';

function node(id: number, parent: number | null, status: WireNode['status'], kind: WireNode['kind'] = 'task'): WireNode {
  return { id, description: `Node ${id}`, status, parent_id: parent, kind, flavor: kind === 'task' ? 'implement' : null };
}

const GRAPH: WireGraphSnapshot = {
  nodes: [
    node(1, null, 'running', 'goal'),
    node(2, 1, 'done'),
    node(3, 1, 'blocked'),
    node(4, 3, 'running'),
    node(5, 3, 'pending'),
    node(6, 1, 'failed'),
    node(7, 1, 'blocked'),
  ],
  edges: [],
  root_ids: [1],
};

function run(nodeId: number, extra: Partial<WireLiveRun> = {}): WireLiveRun {
  return {
    run_id: `run-${nodeId}`, node_id: nodeId, description: '', status: 'running', progress: null,
    progress_pct: null, attempt: null, max_attempts: null, stalled: false, elapsed_seconds: 0, ...extra,
  };
}

const REVIEW: WireGoalReview = {
  review_id: 9, goal_id: 1, goal_revision: 'r', evidence: 'evidence', proposed_change: 'change',
  decision: 'pending', affected_node_ids: null, reviewer: 'coordinator', assessed_at: '', decided_at: null, decided_by: null,
};

describe('tree', () => {
  const tree = buildTree(GRAPH);

  it('walks ancestry and readiness from the parent links', () => {
    expect(ancestry(tree, 4).map((n) => n.id)).toEqual([1, 3, 4]);
    expect(isReady(tree, tree.byId.get(5)!)).toBe(true);
    expect(buildTree({ ...GRAPH, root_ids: [] }).roots.map((n) => n.id)).toEqual([1]);
  });

  it('folds done prerequisites and keeps cards inside the pane', () => {
    const options = { width: 300, cardWidth: 100, rootWidth: 140, height: 56, gap: 8, pitch: 84, fold: true, unfold: {} };
    const folded = layoutTree(tree, 1, options);
    expect(folded.nodes.map((n) => n.id)).not.toContain(2);
    expect(folded.nodes.find((n) => n.id === 1)?.folded).toBe(1);
    expect(folded.nodes.every((n) => n.left >= 0 && n.left + n.w <= 300 + 0.001)).toBe(true);
    expect(folded.edges).toHaveLength(folded.nodes.length - 1);
    const open = layoutTree(tree, 1, { ...options, unfold: { 1: true } });
    expect(open.nodes.map((n) => n.id)).toContain(2);
    expect(layoutTree(tree, 99, options).nodes).toEqual([]);
  });
});

describe('needs', () => {
  it('ranks a permission request over a review over a failure', () => {
    const permission = run(6, { session: { context: null, events: [], actions: [], active: true, owner_incarnation: '', invocation_id: '',
      permissions: [{ kind: 'permission', text: 'git push', event_id: 'p1', state: 'requested', action: null }] } });
    const needs = deriveNeeds(GRAPH.nodes, [permission], [REVIEW]);
    expect(needs.get(6)).toMatchObject({ kind: 'permission', permissionId: 'p1', summary: 'git push' });
    expect(needs.get(1)).toMatchObject({ kind: 'review', reviewId: 9 });
    expect(nextNeed(needs, 1)?.nodeId).toBe(6);
    expect(nextNeed(needs, 6)?.nodeId).toBe(1);
    expect(nextNeed(new Map(), null)).toBeUndefined();
  });
});

describe('thread', () => {
  const detail = {
    runs: { items: [{ run_id: 'run-a', node_id: 6, status: 'failed', started_at: '2026-01-01T00:00:00Z', ended_at: '2026-01-01T00:10:00Z', error: 'coverage 88%' }] },
    reviews: { items: [{ node_id: 6, round: 1, verdict: 'reject', findings: 'Missing tests.', created_at: '2026-01-01T00:05:00Z' }] },
    sessions: { items: [{ run_id: 'run-a', session: null, event_history: { items: [
      { kind: 'assistant', text: 'Working.', event_id: 'e1', state: '', action: null },
      { kind: 'tool', text: 'edit a.js\nmore', event_id: 'e2', state: '', action: null },
      { kind: 'tool', text: 'just check-llm · FAIL', event_id: 'e4', state: '', action: null },
      { kind: 'status', text: ' ', event_id: 'e3', state: '', action: null },
    ] } }] },
  } as unknown as WireMhDetail;

  it('orders run lifecycle, events, verdicts, and operator notes', () => {
    const failed = { ...node(6, 1, 'failed'), description: 'Strip\n\nOperator note: cover the empty state' };
    const thread = taskThread({ node: failed, detail, live: undefined, now: Date.parse('2026-01-01T01:00:00Z'), gates: ['just check-llm'] });
    expect(thread.map((m) => [m.kind, m.text])).toEqual([
      ['gate', 'Run a dispatched'],
      ['agent', 'Working.'],
      ['tool', 'edit a.js'],
      ['gate', 'just check-llm · FAIL'],
      ['gate', 'Gate FAIL · coverage 88%'],
      ['agent', 'Reject · round 1. Missing tests.'],
      ['me', 'cover the empty state'],
    ]);
    expect(thread[0].at).toBe('1h');
    expect(taskThread({ node: node(5, 3, 'pending'), detail: null, live: undefined, now: 0 })[0].text).toContain('Not dispatched');
  });

  it('summarises a goal with its pending review and coordinator records', () => {
    const thread = goalThread({ node: GRAPH.nodes[0], tree: buildTree(GRAPH), reviews: [REVIEW],
      records: [{ seq: 1, kind: 'turn', text: 'Dispatched 4.', status: 'ok' }, { seq: 2, kind: 'tool', text: 'x', status: 'ok', tool_name: 'plan' }] });
    expect(thread.map((m) => m.text)).toEqual([
      'Node 1: 1 of 4 prerequisites done.',
      'Dispatched 4.',
      'plan x',
      'evidence',
      'Proposed change: change Accept it, or reject and keep the current goal?',
    ]);
    expect(splitBrief('Title\nBody text\n\nOperator note: n1')).toEqual({ brief: 'Body text', notes: ['n1'] });
  });
});

describe('insights', () => {
  const tree = buildTree(GRAPH);
  const snapshot = { goal: null, graph: GRAPH, capabilities: null, completed: 3, failed: 1, stopped: 2,
    active_runs: [run(4, { stalled: true })], listener_errors: ['boom'] } as unknown as WireMhSnapshot;

  it('lists stalled runs, listener errors, and stuck blocked nodes', () => {
    expect(deriveIssues(snapshot, tree).map((issue) => issue.id)).toEqual(['listener:0', 'stalled:run-4', 'blocked:7']);
  });

  it('derives stats and project cards from the snapshot', () => {
    expect(deriveStats({ snapshot, tree, concurrency: 4, paused: true }).map((s) => s.value)).toEqual(['7', '75%', '3', '1/4']);
    expect(projectCards(tree, new Map([[6, { nodeId: 6, kind: 'failed', summary: '' }]]))).toEqual([
      { id: 1, title: 'Node 1', needCount: 1, running: 1, pct: 14 },
    ]);
  });
});

describe('parseUnifiedDiff', () => {
  it('keeps hunks and changed lines, drops file headers', () => {
    const lines = parseUnifiedDiff('diff --git a/x b/x\nindex 1..2\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-old\n+new\n same');
    expect(lines).toEqual([
      { mark: '@', text: '@@ -1 +1 @@' }, { mark: '-', text: 'old' }, { mark: '+', text: 'new' }, { mark: ' ', text: 'same' },
    ]);
  });
});

describe('canAdopt', () => {
  const tree = buildTree(GRAPH);

  it('accepts a sibling or the top level as the new parent', () => {
    expect(canAdopt(tree, 6, 3)).toBe(true);
    expect(canAdopt(tree, 3, null)).toBe(true);
  });

  it('rejects the current parent, the node itself, and its own prerequisites', () => {
    expect(canAdopt(tree, 4, 3)).toBe(false);
    expect(canAdopt(tree, 3, 3)).toBe(false);
    expect(canAdopt(tree, 3, 4)).toBe(false);
    expect(canAdopt(tree, 1, null)).toBe(false);
    expect(canAdopt(tree, 99, 1)).toBe(false);
  });
});
