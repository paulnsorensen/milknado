// A minimal, dependency-free client store. Slot and feature code read a
// snapshot with `getState()` and re-render on `subscribe(listener)`.
import type { WireCapabilities, WireExecutionSnapshot } from './wire';

export type GraphFilter = 'ready' | 'running' | 'blocked' | null;
export type GraphLod = 'card' | 'pill' | 'dot';

export interface GraphView {
  filter: GraphFilter;
  focus: string | number | null;
  lod: GraphLod | undefined;
  zoom: number | undefined;
  collapsed: Array<string | number>;
}

export interface Notice {
  id: string;
  reason: string;
}

export interface StoreState {
  snapshot: WireExecutionSnapshot | null;
  capabilities: WireCapabilities | null;
  selection: string | number | null;
  activeSidecar: string | null;
  graphView: GraphView;
  notices: Notice[];
}

type Listener = () => void;

function initialState(): StoreState {
  return {
    snapshot: null,
    capabilities: null,
    selection: null,
    activeSidecar: null,
    graphView: { filter: null, focus: null, lod: undefined, zoom: undefined, collapsed: [] },
    notices: [],
  };
}

let state = initialState();
const listeners = new Set<Listener>();

function emit(): void {
  for (const listener of listeners) {
    listener();
  }
}

export function getState(): StoreState {
  return state;
}

export function subscribe(listener: Listener): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function setSnapshot(snapshot: WireExecutionSnapshot): void {
  state = { ...state, snapshot, capabilities: snapshot.capabilities };
  emit();
}

export function setSelection(selection: string | number | null): void {
  state = { ...state, selection };
  emit();
}

export function setActiveSidecar(activeSidecar: string | null): void {
  state = { ...state, activeSidecar };
  emit();
}

function sameCollapsed(a: GraphView['collapsed'], b: GraphView['collapsed']): boolean {
  return a.length === b.length && a.every((value, index) => value === b[index]);
}

function sameGraphView(a: GraphView, b: GraphView): boolean {
  return (
    a.filter === b.filter &&
    a.focus === b.focus &&
    a.lod === b.lod &&
    a.zoom === b.zoom &&
    sameCollapsed(a.collapsed, b.collapsed)
  );
}

// The graph view may report the same layout on every render pass; emitting
// unconditionally would re-render the subscriber that produced it, looping.
export function setGraphView(patch: Partial<GraphView>): void {
  const next = { ...state.graphView, ...patch };
  if (sameGraphView(next, state.graphView)) {
    return;
  }
  state = { ...state, graphView: next };
  emit();
}

const NOTICE_TTL_MS = 6000;

export function removeNotice(id: string): void {
  state = { ...state, notices: state.notices.filter((notice) => notice.id !== id) };
  emit();
}

export function pushNotice(reason: string): void {
  const notice: Notice = { id: `${Date.now()}-${state.notices.length}`, reason };
  state = { ...state, notices: [...state.notices, notice] };
  emit();
  setTimeout(() => removeNotice(notice.id), NOTICE_TTL_MS);
}

export function resetStore(): void {
  state = initialState();
  emit();
}
