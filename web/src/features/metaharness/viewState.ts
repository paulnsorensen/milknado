// UI state for the metaharness view: which project and node are in focus,
// which pane overlays are open, and the transient toast.
export type MhView = 'thread' | 'diff';
export type MhSide = 'details' | 'knowledge' | 'issues' | 'stats' | null;

export interface MhPop {
  id: number;
  x: number;
  y: number;
}

export interface MhState {
  projectId: number | null;
  scopeRoot: number | null;
  sel: number | null;
  view: MhView;
  side: MhSide;
  expanded: boolean;
  railOpen: boolean;
  pop: MhPop | null;
  unfold: Record<number, boolean>;
  toast: string | null;
  /** The composer sends a retry note instead of a message. */
  compose: 'message' | 'retry';
}

const TOAST_MS = 3200;

function initial(): MhState {
  return {
    projectId: null,
    scopeRoot: null,
    sel: null,
    view: 'thread',
    side: null,
    expanded: false,
    railOpen: false,
    pop: null,
    unfold: {},
    toast: null,
    compose: 'message',
  };
}

let state = initial();
let toastTimer: ReturnType<typeof setTimeout> | undefined;
const listeners = new Set<() => void>();

export function getMh(): MhState {
  return state;
}

export function subscribeMh(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function patchMh(patch: Partial<MhState> | ((current: MhState) => Partial<MhState>)): void {
  const delta = typeof patch === 'function' ? patch(state) : patch;
  state = { ...state, ...delta };
  for (const listener of listeners) {
    listener();
  }
}

export function flash(text: string): void {
  clearTimeout(toastTimer);
  patchMh({ toast: text });
  toastTimer = setTimeout(() => patchMh({ toast: null }), TOAST_MS);
}

export function resetMh(): void {
  clearTimeout(toastTimer);
  state = initial();
}
