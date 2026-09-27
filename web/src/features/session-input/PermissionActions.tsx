import type { ReactElement } from 'react';
import { useSyncExternalStore } from 'react';
import { getState, subscribe } from '../../app/store';
import { Milknado } from '../../design-system';
import { sendSessionCommand } from './sessionCommand';

/** The `sidecar-action` contribution: Approve/Deny for the oldest pending permission request. */
export function PermissionActions(): ReactElement | null {
  const store = useSyncExternalStore(subscribe, getState);
  const { Button, StatusBadge } = Milknado;
  const permissionId = store.capabilities?.owner.permission_ids?.[0];

  if (!permissionId) {
    return null;
  }

  return (
    <section className="mk-permission" aria-label="Permission request">
      <div className="mk-rail-head">
        <StatusBadge state="at-risk">Permission requested</StatusBadge>
        <span className="mk-text-data" style={{ color: 'var(--mk-at-risk)' }}>
          {permissionId}
        </span>
      </div>
      <p className="mk-text-body">The agent waits for a decision on this request.</p>
      <div className="mk-button-row">
        <Button
          variant="primary"
          className="mk-btn-sm"
          onClick={() => void sendSessionCommand('approve', { requestId: permissionId })}
        >
          Approve
        </Button>
        <Button className="mk-btn-sm" onClick={() => void sendSessionCommand('deny', { requestId: permissionId })}>
          Deny
        </Button>
      </div>
    </section>
  );
}