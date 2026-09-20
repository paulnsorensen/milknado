// Tracks the narrow-viewport media query so the `layout` slot can switch
// between the wide default layout and the narrow list/detail layout.
import { useSyncExternalStore } from 'react';

export const NARROW_QUERY = '(max-width: 400px)';

function subscribe(listener: () => void): () => void {
  if (typeof window.matchMedia !== 'function') {
    return () => {};
  }
  const media = window.matchMedia(NARROW_QUERY);
  media.addEventListener('change', listener);
  return () => media.removeEventListener('change', listener);
}

function isNarrow(): boolean {
  if (typeof window.matchMedia !== 'function') {
    return false;
  }
  return window.matchMedia(NARROW_QUERY).matches;
}

/** True at or below the narrow-viewport breakpoint; re-renders on resize. */
export function useNarrowViewport(): boolean {
  return useSyncExternalStore(subscribe, isNarrow, () => false);
}
