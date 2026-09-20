import type { ReactElement } from 'react';
import { useSyncExternalStore } from 'react';
import { getState, subscribe } from '../../app/store';
import { Milknado } from '../../design-system';
import { sendSessionCommand } from './sessionCommand';

/** The `sidecar-action` contribution: Approve/Deny for the oldest pending permission request. */
export function PermissionActions(): ReactElement | null {
  const store = useSyncExternalStore(subscribe, getState);
  const { Button } = Milknado;
  const permissionId = store.capabilities?.owner.permission_ids?.[0];

  if (!permissionId) {
    return null;
  }

  return (
    <div className="mk-permission-actions">
      <Button onClick={() => void sendSessionCommand('approve', { requestId: permissionId })}>Approve</Button>
      <Button onClick={() => void sendSessionCommand('deny', { requestId: permissionId })}>Deny</Button>
    </div>
  );
}
