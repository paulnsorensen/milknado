// The `rail-action` contribution: opens the Add node dialog, gated on the
// server's `graph_edits` capability.
import type { ReactElement } from 'react';
import { useSyncExternalStore } from 'react';
import { getState, subscribe } from '../../app/store';
import { Milknado } from '../../design-system';
import { openDialog } from './dialogState';

export function AddNodeButton(): ReactElement | null {
  const store = useSyncExternalStore(subscribe, getState);
  const { Button } = Milknado;
  const capability = store.capabilities?.graph_edits;

  if (!capability) {
    return null;
  }

  return (
    <div className="mk-graph-edits-rail">
      <Button disabled={!capability.available} onClick={() => openDialog('add')}>
        Add node
      </Button>
      {!capability.available && <p role="note">{capability.reason}</p>}
    </div>
  );
}
