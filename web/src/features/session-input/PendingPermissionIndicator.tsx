import type { ReactElement } from 'react';
import { useSyncExternalStore } from 'react';
import { getState, subscribe } from '../../app/store';
import { Milknado } from '../../design-system';

/** The `header-control` contribution: pending permissions remain visible without a node selection. */
export function PendingPermissionIndicator(): ReactElement | null {
  const store = useSyncExternalStore(subscribe, getState);
  const { StatusBadge } = Milknado;
  const owner = store.capabilities?.owner;
  const permissionCount = owner?.permission_ids?.length ?? 0;

  const hidesForSelectedNode =
    typeof store.selection === 'number' &&
    (owner?.node_id === undefined || owner?.node_id === store.selection);
  if (hidesForSelectedNode || !owner?.available || permissionCount === 0) {
    return null;
  }

  const label = permissionCount === 1 ? 'Permission requested' : `${permissionCount} permissions requested`;
  return (
    <div className="mk-button-row" role="status" aria-label="Pending permission requests">
      <StatusBadge state="at-risk">{label}</StatusBadge>
    </div>
  );
}
