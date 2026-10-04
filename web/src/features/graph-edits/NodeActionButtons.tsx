// The `sidecar-action` contribution: Edit, Move and Archive for the
// currently selected node, gated on the `graph_edits` capability.
import type { ReactElement } from 'react';
import { useSyncExternalStore } from 'react';
import { getState, selectedNodeId, subscribe } from '../../app/store';
import { Milknado } from '../../design-system';
import { openDialog } from './dialogState';

interface NodeActionRowProps {
  disabled: boolean;
  nodeId: number;
}

function NodeActionRow({ disabled, nodeId }: NodeActionRowProps): ReactElement {
  const { Button } = Milknado;
  return (
    <div className="mk-button-row">
      <Button className="mk-btn-sm" disabled={disabled} onClick={() => openDialog('edit', nodeId)}>
        Edit node
      </Button>
      <Button className="mk-btn-sm" disabled={disabled} onClick={() => openDialog('move', nodeId)}>
        Move node
      </Button>
      <Button className="mk-btn-sm" disabled={disabled} onClick={() => openDialog('archive', nodeId)}>
        Archive node
      </Button>
    </div>
  );
}

export function NodeActionButtons(): ReactElement | null {
  const store = useSyncExternalStore(subscribe, getState);
  const capability = store.capabilities?.graph_edits;
  const nodeId = selectedNodeId(store);

  if (!capability || nodeId === null) {
    return null;
  }

  return (
    <section className="mk-section" aria-label="Node actions">
      <NodeActionRow disabled={!capability.available} nodeId={nodeId} />
      {!capability.available && (
        <p role="note" className="mk-text-caption mk-muted">
          {capability.reason}
        </p>
      )}
    </section>
  );
}
