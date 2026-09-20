// The `dialog` slot contribution confirming an archive, posted to
// `POST /api/nodes/{id}/archive`.
import type { ReactElement } from 'react';
import { useSyncExternalStore } from 'react';
import { Milknado } from '../../design-system';
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
    <div role="alertdialog" aria-label="Archive node" className="mk-archive-node-dialog">
      <p>Archive this node and its subtree?</p>
      <Button onClick={submit}>Archive node</Button>
      <Button onClick={closeDialog}>Cancel</Button>
    </div>
  );
}
