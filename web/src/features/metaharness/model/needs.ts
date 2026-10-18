// The decisions that wait on a person: a worker's permission request, a
// pending goal review, and a node whose last run failed its gate.
import type { WireNode } from '../../../app/wire';
import type { WireGoalReview } from '../../goal-review/wire';
import type { WireLiveRun } from '../wire';

export type NeedKind = 'permission' | 'review' | 'failed';

export interface Need {
  nodeId: number;
  kind: NeedKind;
  summary: string;
  reviewId?: number;
  runId?: string;
  permissionId?: string;
}

export const NEED_LABEL: Record<NeedKind, string> = {
  permission: 'question',
  review: 'verdict',
  failed: 'gate failed',
};

export const NEED_KICKER: Record<NeedKind, string> = {
  permission: 'Worker asks',
  review: 'Goal review',
  failed: 'Gate failed',
};

/** One need per node; a permission request outranks a review, which outranks a failure. */
export function deriveNeeds(
  nodes: WireNode[],
  runs: WireLiveRun[],
  reviews: WireGoalReview[],
): Map<number, Need> {
  const ids = new Set(nodes.map((node) => node.id));
  const needs = new Map<number, Need>();
  for (const node of nodes) {
    if (node.status === 'failed') {
      needs.set(node.id, { nodeId: node.id, kind: 'failed', summary: 'The last run failed its quality gate.' });
    }
  }
  for (const review of reviews) {
    if (review.decision === 'pending' && ids.has(review.goal_id)) {
      needs.set(review.goal_id, {
        nodeId: review.goal_id,
        kind: 'review',
        summary: review.proposed_change,
        reviewId: review.review_id,
      });
    }
  }
  for (const run of runs) {
    const request = run.session?.permissions.at(-1);
    if (request && ids.has(run.node_id)) {
      needs.set(run.node_id, {
        nodeId: run.node_id,
        kind: 'permission',
        summary: request.text,
        runId: run.run_id,
        permissionId: request.event_id,
      });
    }
  }
  return needs;
}

/** The need after `current` in node-id order, wrapping around. */
export function nextNeed(needs: Map<number, Need>, current: number | null): Need | undefined {
  const ids = [...needs.keys()].sort((a, b) => a - b);
  if (ids.length === 0) {
    return undefined;
  }
  const index = current === null ? -1 : ids.indexOf(current);
  return needs.get(ids[(index + 1) % ids.length]);
}
