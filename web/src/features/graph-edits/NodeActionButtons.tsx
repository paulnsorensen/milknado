// The `sidecar-action` contribution: Edit, Move and Archive for the
// currently selected node, gated on the `graph_edits` capability.
import type { ReactElement } from "react";
import { useSyncExternalStore } from "react";
import { getState, selectedNodeId, subscribe } from "../../app/store";
import { Milknado } from "../../design-system";
import { openDialog } from "./dialogState";

export function NodeActionButtons(): ReactElement | null {
  const store = useSyncExternalStore(subscribe, getState);
  const { Button } = Milknado;
  const capability = store.capabilities?.graph_edits;
  const nodeId = selectedNodeId(store);

  if (!capability || nodeId === null) {
    return null;
  }

  return (
    <section className="mk-section" aria-label="Node actions">
      <div className="mk-button-row">
        <Button
          className="mk-btn-sm"
          disabled={!capability.available}
          onClick={() => openDialog("edit", nodeId)}
        >
          Edit node
        </Button>
        <Button
          className="mk-btn-sm"
          disabled={!capability.available}
          onClick={() => openDialog("move", nodeId)}
        >
          Move node
        </Button>
        <Button
          className="mk-btn-sm"
          disabled={!capability.available}
          onClick={() => openDialog("archive", nodeId)}
        >
          Archive node
        </Button>
      </div>
      {!capability.available && (
        <p role="note" className="mk-text-caption mk-muted">
          {capability.reason}
        </p>
      )}
    </section>
  );
}
