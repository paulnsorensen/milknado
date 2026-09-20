// The `dialog` slot contribution: the one confirmation prompt for every
// run-controls action. Confirm runs the pending action once; Dismiss runs
// nothing.
import type { ReactElement } from 'react';
import { useSyncExternalStore } from 'react';
import { Milknado } from '../../design-system';
import { confirmPending, dismissConfirm, getPendingConfirm, subscribeConfirm } from './confirmState';

export function ConfirmDialog(): ReactElement | null {
  const pending = useSyncExternalStore(subscribeConfirm, getPendingConfirm);
  const { Button } = Milknado;

  if (!pending) {
    return null;
  }

  return (
    <div role="alertdialog" aria-label={pending.prompt} className="mk-confirm-dialog">
      <p>{pending.prompt}</p>
      <Button onClick={confirmPending}>Confirm</Button>
      <Button onClick={dismissConfirm}>Dismiss</Button>
    </div>
  );
}
