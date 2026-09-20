// The `header-control` contribution: the run-mode badge (owner vs
// observer) and Stop scheduling, gated by capabilities like the server.
import type { ReactElement } from 'react';
import { useSyncExternalStore } from 'react';
import { getState, subscribe } from '../../app/store';
import { Milknado } from '../../design-system';
import { stopScheduling } from './commands';
import { requestConfirm } from './confirmState';

export function RunModeHeader(): ReactElement | null {
  const store = useSyncExternalStore(subscribe, getState);
  const { Button, StatusBadge } = Milknado;
  const capabilities = store.capabilities;

  if (!capabilities) {
    return null;
  }

  const isOwner = capabilities.owner.available;
  const stop = capabilities.stop_scheduling;

  return (
    <div className="mk-run-mode">
      <StatusBadge state={isOwner ? 'running' : 'pending'}>
        {isOwner ? 'Owner' : 'Observer'}
      </StatusBadge>
      <Button
        disabled={!stop.available}
        onClick={() => requestConfirm('Stop scheduling?', () => void stopScheduling())}
      >
        Stop scheduling
      </Button>
      {!stop.available && <p role="note">{stop.reason}</p>}
    </div>
  );
}
