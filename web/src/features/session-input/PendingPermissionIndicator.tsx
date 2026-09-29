import type { ReactElement } from 'react';
import { useSyncExternalStore } from 'react';
import { canActOnSelectedRun, getState, setSelection, subscribe } from '../../app/store';
import { Milknado } from '../../design-system';

/** The `header-control` contribution: pending permissions remain visible without a node selection. */
export function PendingPermissionIndicator(): ReactElement | null {
  const store = useSyncExternalStore(subscribe, getState);
  const { Button, StatusBadge } = Milknado;
  const owner = store.capabilities?.owner;
  const permissionCount = owner?.permission_ids?.length ?? 0;

  if (canActOnSelectedRun(store) || !owner?.available || permissionCount === 0) {
    return null;
  }

  const label = permissionCount === 1 ? 'Permission requested' : `${permissionCount} permissions requested`;
  const nodeId = owner.node_id;

  return (
    <div className="mk-button-row" role="status" aria-label="Pending permission requests">
      {nodeId === undefined ? (
        <StatusBadge state="at-risk">{label}</StatusBadge>
      ) : (
        <Button
          className="mk-btn-sm"
          ariaLabel={`${label} · node ${nodeId}`}
          onClick={() => setSelection(nodeId)}
        >
          <StatusBadge state="at-risk">{label}</StatusBadge>
        </Button>
      )}
    </div>
  );
}
