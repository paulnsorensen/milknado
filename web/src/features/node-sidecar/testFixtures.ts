import type { WireNodeDetailResponse } from '../../shared/node-detail';

export function detailResponse(
  overrides: Partial<WireNodeDetailResponse['detail']> = {},
): WireNodeDetailResponse {
  return {
    node_id: 7,
    request_generation: 1,
    detail: {
      node: {
        id: 7,
        description: 'Bake the roadmap',
        status: 'running',
        parent_id: null,
        kind: 'task',
        flavor: null,
      },
      description: 'Bake the roadmap',
      parent: null,
      ancestors: {
        items: [],
        offset: 0,
        limit: 50,
        total: 0,
        has_more: false,
        state: 'loaded',
      },
      prerequisite_ids: {
        items: [],
        offset: 0,
        limit: 50,
        total: 0,
        has_more: false,
        state: 'loaded',
      },
      dependent_ids: {
        items: [],
        offset: 0,
        limit: 50,
        total: 0,
        has_more: false,
        state: 'loaded',
      },
      owned_files: {
        items: [],
        offset: 0,
        limit: 50,
        total: 0,
        has_more: false,
        state: 'loaded',
      },
      runs: {
        items: [],
        offset: 0,
        limit: 50,
        total: 0,
        has_more: false,
        state: 'loaded',
      },
      sessions: {
        items: [],
        offset: 0,
        limit: 50,
        total: 0,
        has_more: false,
        state: 'loaded',
      },
      ...overrides,
    },
  };
}

export const EMPTY_CAPABILITIES = {
  session_input: { available: false, reason: null },
  cancel: { available: false, reason: null },
  force_stop: { available: false, reason: null },
  stop_scheduling: { available: false, reason: null },
  graph_edits: { available: false, reason: null },
  review_decision: { available: false, reason: null },
  git: { available: false, reason: null },
  host_owner: { available: false, reason: null },
  owner: { available: false },
};

export type { WireNodeDetailResponse };
