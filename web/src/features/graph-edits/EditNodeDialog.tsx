// The `dialog` slot contribution for editing the selected node's
// description and flavor, posted to `PATCH /api/nodes/{id}`.
import type { ReactElement } from 'react';
import { useEffect, useState, useSyncExternalStore } from 'react';
import { getState, subscribe } from '../../app/store';
import { Milknado } from '../../design-system';
import { editNode } from './commands';
import { closeDialog, getDialogState, subscribeDialog } from './dialogState';

export function EditNodeDialog(): ReactElement | null {
  const dialog = useSyncExternalStore(subscribeDialog, getDialogState);
  const store = useSyncExternalStore(subscribe, getState);
  const { Button } = Milknado;
  const node = store.snapshot?.graph?.nodes.find((candidate) => candidate.id === dialog.nodeId);
  const [description, setDescription] = useState(node?.description ?? '');
  const [flavor, setFlavor] = useState(node?.flavor ?? '');

  useEffect(() => {
    setDescription(node?.description ?? '');
    setFlavor(node?.flavor ?? '');
  }, [dialog.nodeId, node?.description, node?.flavor]);

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
    <div role="dialog" aria-label="Edit node" className="mk-edit-node-dialog">
      <label>
        Description
        <input
          aria-label="Edit description"
          value={description}
          onChange={(event) => setDescription(event.target.value)}
        />
      </label>
      <label>
        Flavor
        <input
          aria-label="Edit flavor"
          value={flavor}
          onChange={(event) => setFlavor(event.target.value)}
        />
      </label>
      <Button onClick={submit}>Save changes</Button>
      <Button onClick={closeDialog}>Cancel</Button>
    </div>
  );
}
