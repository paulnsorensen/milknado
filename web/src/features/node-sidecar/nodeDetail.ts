// Fetches and pages GET /api/nodes/{id}. One node is tracked at a time: a
// new node id resets both the detail page and the session transcript page.
import { get } from '../../app/api';
import type { WireNodeDetailResponse } from './detailWire';

export interface DetailState {
  nodeId: number | null;
  page: number;
  sessionPage: number;
  detail: WireNodeDetailResponse | null;
}

let state: DetailState = { nodeId: null, page: 0, sessionPage: 0, detail: null };
const listeners = new Set<() => void>();
let requestGeneration = 0;

function emit(): void {
  for (const listener of listeners) {
    listener();
  }
}

export function getDetailState(): DetailState {
  return state;
}

export function subscribeDetail(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

async function fetchDetail(): Promise<void> {
  const { nodeId, page, sessionPage } = state;
  if (nodeId === null) {
    return;
  }
  const generation = ++requestGeneration;
  const params = new URLSearchParams({
    request_generation: String(generation),
    page: String(page),
    session_event_page: String(sessionPage),
  });
  const response = await get<WireNodeDetailResponse>(`/api/nodes/${nodeId}?${params.toString()}`);
  if (response !== null && generation === requestGeneration) {
    state = { ...state, detail: response };
    emit();
  }
}

export function selectNode(nodeId: number | null): void {
  if (state.nodeId === nodeId) {
    return;
  }
  state = { nodeId, page: 0, sessionPage: 0, detail: null };
  emit();
  void fetchDetail();
}

export function pageNext(): void {
  if (state.nodeId === null) {
    return;
  }
  state = { ...state, page: state.page + 1 };
  emit();
  void fetchDetail();
}

export function pagePrevious(): void {
  if (state.nodeId === null || state.page === 0) {
    return;
  }
  state = { ...state, page: state.page - 1 };
  emit();
  void fetchDetail();
}

export function sessionPageNext(): void {
  if (state.nodeId === null) {
    return;
  }
  state = { ...state, sessionPage: state.sessionPage + 1 };
  emit();
  void fetchDetail();
}

export function sessionPagePrevious(): void {
  if (state.nodeId === null || state.sessionPage === 0) {
    return;
  }
  state = { ...state, sessionPage: state.sessionPage - 1 };
  emit();
  void fetchDetail();
}

export function followNewest(): void {
  if (state.nodeId === null || state.sessionPage === 0) {
    return;
  }
  state = { ...state, sessionPage: 0 };
  emit();
  void fetchDetail();
}

export function resetDetail(): void {
  state = { nodeId: null, page: 0, sessionPage: 0, detail: null };
  emit();
}
