// Which wide layout the dashboard shows: the classic canvas or the
// metaharness view. The choice persists per browser.
export type ViewMode = 'classic' | 'metaharness';

export const VIEW_MODE_KEY = 'milknado.view';

const listeners = new Set<() => void>();

function readStored(): ViewMode {
  if (window.location.hash === '#metaharness') {
    return 'metaharness';
  }
  try {
    return window.localStorage.getItem(VIEW_MODE_KEY) === 'metaharness' ? 'metaharness' : 'classic';
  } catch {
    return 'classic';
  }
}

let mode: ViewMode = readStored();

export function getViewMode(): ViewMode {
  return mode;
}

export function subscribeViewMode(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function setViewMode(next: ViewMode): void {
  mode = next;
  try {
    window.localStorage.setItem(VIEW_MODE_KEY, next);
  } catch {
    // Storage can be unavailable in private windows; the choice then lasts for this page only.
  }
  for (const listener of listeners) {
    listener();
  }
}

export function resetViewMode(): void {
  mode = readStored();
}
