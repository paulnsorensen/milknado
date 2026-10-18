// A subscribable value for `useSyncExternalStore`, shared by the view's
// server-backed stores.
import { useSyncExternalStore } from 'react';

export interface Cell<T> {
  get: () => T;
  set: (value: T) => void;
  subscribe: (listener: () => void) => () => void;
}

export function cell<T>(initial: T): Cell<T> {
  let value = initial;
  const listeners = new Set<() => void>();
  return {
    get: () => value,
    set: (next) => {
      value = next;
      for (const listener of listeners) {
        listener();
      }
    },
    subscribe: (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
}

export function useCell<T>(source: Cell<T>): T {
  return useSyncExternalStore(source.subscribe, source.get);
}
