// The `dialog` slot contribution: the one confirmation prompt for every
// run-controls action. Confirm runs the pending action once; Dismiss runs
// nothing.
import type { ReactElement } from 'react';
import { useSyncExternalStore } from 'react';
import { Milknado } from '../../design-system';
import { Dialog } from '../../shared/dialog/Dialog';
import { confirmPending, dismissConfirm, getPendingConfirm, subscribeConfirm } from './confirmState';

export function ConfirmDialog(): ReactElement | null {
  const pending = useSyncExternalStore(subscribeConfirm, getPendingConfirm);
  const { Button } = Milknado;

  if (!pending) {
    return null;
  }

  return (
    <Dialog
      role="alertdialog"
      title={pending.prompt}
      onClose={dismissConfirm}
      actions={
        <>
          <Button onClick={dismissConfirm}>Dismiss</Button>
          <Button variant="primary" onClick={confirmPending}>
            Confirm
          </Button>
        </>
      }
    >
      <p className="mk-dialog-body">The action runs once. It cannot be undone from here.</p>
    </Dialog>
  );
}