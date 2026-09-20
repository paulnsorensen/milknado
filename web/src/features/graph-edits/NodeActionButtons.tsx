// The `sidecar-action` contribution: Edit, Move and Archive for the
// currently selected node, gated on the `graph_edits` capability.
import type { ReactElement } from 'react';
import { useSyncExternalStore } from 'react';
import { getState, subscribe } from '../../app/store';
import { Milknado } from '../../design-system';
import { openDialog } from './dialogState';

export function NodeActionButtons(): ReactElement | null {
  const store = useSyncExternalStore(subscribe, getState);
  const { Button } = Milknado;
  const capability = store.capabilities?.graph_edits;
  const selection = store.selection;

  if (!capability || typeof selection !== 'number') {
    return null;
  }

  return (
    <div className="mk-node-actions">
      <Button disabled={!capability.available} onClick={() => openDialog('edit', selection)}>
        Edit node
      </Button>
      <Button disabled={!capability.available} onClick={() => openDialog('move', selection)}>
        Move node
      </Button>
      <Button disabled={!capability.available} onClick={() => openDialog('archive', selection)}>
        Archive node
      </Button>
      {!capability.available && <p role="note">{capability.reason}</p>}
    </div>
  );
}
