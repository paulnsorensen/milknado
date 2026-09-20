// The `banner` slot contribution: the snapshot's listener errors. `wire.ts`
// does not type `listener_errors` (no other feature reads it), so this
// reads it directly off the raw snapshot payload.
import type { ReactElement } from 'react';
import { useSyncExternalStore } from 'react';
import { getState, subscribe } from '../../app/store';

interface SnapshotWithListenerErrors {
  listener_errors?: string[];
}

export function ErrorBanner(): ReactElement | null {
  const store = useSyncExternalStore(subscribe, getState);
  const errors = (store.snapshot as SnapshotWithListenerErrors | null)?.listener_errors ?? [];

  if (errors.length === 0) {
    return null;
  }

  return (
    <div role="alert" className="mk-error-banner">
      {errors.map((error) => (
        <p key={error}>{error}</p>
      ))}
    </div>
  );
}
