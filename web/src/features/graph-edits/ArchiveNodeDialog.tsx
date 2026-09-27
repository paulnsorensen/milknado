// The `dialog` slot contribution confirming an archive, posted to
// `POST /api/nodes/{id}/archive`.
import type { ReactElement } from 'react';
import { useSyncExternalStore } from 'react';
import { Milknado } from '../../design-system';
import { Dialog } from '../../shared/dialog/Dialog';
import { archiveNode } from './commands';
import { closeDialog, getDialogState, subscribeDialog } from './dialogState';

export function ArchiveNodeDialog(): ReactElement | null {
  const dialog = useSyncExternalStore(subscribeDialog, getDialogState);
  const { Button } = Milknado;

  if (dialog.kind !== 'archive' || dialog.nodeId === null) {
    return null;
  }
  const nodeId = dialog.nodeId;

  function submit(): void {
    void archiveNode(nodeId);
    closeDialog();
  }

  return (
    <Dialog
      role="alertdialog"
      title="Archive this node and its subtree?"
      label="Archive node"
      actions={
        <>
          <Button onClick={closeDialog}>Cancel</Button>
          <Button variant="primary" onClick={submit}>
            Archive node
          </Button>
        </>
      }
    >
      <p className="mk-dialog-body">The node and every node under it leave the graph. Done work stays in the worktree.</p>
    </Dialog>
  );
}
