// The `dialog` slot contribution for moving the selected node to a new
// parent, posted to `POST /api/nodes/{id}/move`.
import type { ReactElement } from 'react';
import { useState, useSyncExternalStore } from 'react';
import { getState, subscribe } from '../../app/store';
import { Milknado } from '../../design-system';
import { Dialog } from '../../shared/dialog/Dialog';
import { moveNode } from './commands';
import { closeDialog, getDialogState, subscribeDialog } from './dialogState';

export function MoveNodeDialog(): ReactElement | null {
  const dialog = useSyncExternalStore(subscribeDialog, getDialogState);
  const store = useSyncExternalStore(subscribe, getState);
  const { Button } = Milknado;
  const [parentId, setParentId] = useState('');

  if (dialog.kind !== 'move' || dialog.nodeId === null) {
    return null;
  }
  const nodeId = dialog.nodeId;
  const nodes = store.snapshot?.graph?.nodes ?? [];

  function submit(): void {
    void moveNode(nodeId, parentId === '' ? null : Number(parentId));
    setParentId('');
    closeDialog();
  }

  return (
    <Dialog
      title="Move this node"
      label="Move node"
      onClose={closeDialog}
      actions={
        <>
          <Button onClick={closeDialog}>Cancel</Button>
          <Button variant="primary" onClick={submit}>
            Move node
          </Button>
        </>
      }
    >
      <div className="mk-fields">
        <div className="mk-field">
          <label htmlFor="mk-move-parent">New parent</label>
          <select
            id="mk-move-parent"
            className="mk-input"
            value={parentId}
            onChange={(event) => setParentId(event.target.value)}
          >
            <option value="">None</option>
            {nodes
              .filter((candidate) => candidate.id !== nodeId)
              .map((candidate) => (
                <option key={candidate.id} value={candidate.id}>
                  {candidate.description}
                </option>
              ))}
          </select>
        </div>
      </div>
    </Dialog>
  );
}
