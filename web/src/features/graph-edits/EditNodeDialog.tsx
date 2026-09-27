// The `dialog` slot contribution for editing the selected node's
// description and flavor, posted to `PATCH /api/nodes/{id}`.
import type { ReactElement } from 'react';
import { useEffect, useState, useSyncExternalStore } from 'react';
import { getState, subscribe } from '../../app/store';
import { Milknado } from '../../design-system';
import { Dialog } from '../../shared/dialog/Dialog';
import { editNode } from './commands';
import { closeDialog, getDialogState, subscribeDialog } from './dialogState';

export function EditNodeDialog(): ReactElement | null {
  const dialog = useSyncExternalStore(subscribeDialog, getDialogState);
  const store = useSyncExternalStore(subscribe, getState);
  const { Button } = Milknado;
  const node = store.snapshot?.graph?.nodes.find((candidate) => candidate.id === dialog.nodeId);
  const [description, setDescription] = useState(node?.description ?? '');
  const [flavor, setFlavor] = useState(node?.flavor ?? '');

  // Seed only on a node switch; a live snapshot must not overwrite
  // in-progress edits.
  useEffect(() => {
    setDescription(node?.description ?? '');
    setFlavor(node?.flavor ?? '');
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dialog.nodeId]);

  if (dialog.kind !== 'edit' || dialog.nodeId === null) {
    return null;
  }
  const nodeId = dialog.nodeId;

  function submit(): void {
    void editNode(nodeId, {
      description: description.trim() === '' ? undefined : description,
      flavor: flavor.trim() === '' ? null : flavor,
    });
    closeDialog();
  }

  return (
    <Dialog
      title="Edit node"
      onClose={closeDialog}
      actions={
        <>
          <Button onClick={closeDialog}>Cancel</Button>
          <Button variant="primary" onClick={submit}>
            Save changes
          </Button>
        </>
      }
    >
      <div className="mk-fields">
        <div className="mk-field">
          <label htmlFor="mk-edit-description">Description</label>
          <textarea
            id="mk-edit-description"
            className="mk-input"
            aria-label="Edit description"
            rows={3}
            value={description}
            onChange={(event) => setDescription(event.target.value)}
          />
        </div>
        <div className="mk-field">
          <label htmlFor="mk-edit-flavor">Flavor</label>
          <input
            id="mk-edit-flavor"
            className="mk-input"
            aria-label="Edit flavor"
            value={flavor}
            onChange={(event) => setFlavor(event.target.value)}
          />
        </div>
      </div>
    </Dialog>
  );
}
