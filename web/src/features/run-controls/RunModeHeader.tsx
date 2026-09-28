// The `header-control` contribution: the run-mode badge (Run active for an
// owner, Read-only for an observer) and Stop scheduling, gated by
// capabilities like the server.
import type { ReactElement } from 'react';
import { useSyncExternalStore } from 'react';
import { getState, subscribe } from '../../app/store';
import { Milknado } from '../../design-system';
import { ownerLabel } from '../../shared/ownerLabel';
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
  const activeRuns = store.snapshot?.active_runs;

  return (
    <div className="mk-button-row">
      <StatusBadge state={isOwner ? 'running' : 'pending'}>{ownerLabel(isOwner).badge}</StatusBadge>
      {isOwner && (
        <Button
          disabled={!stop.available || activeRuns === undefined}
          title={
            activeRuns === undefined
              ? 'Run totals are not available yet.'
              : stop.available
                ? undefined
                : (stop.reason ?? undefined)
          }
          onClick={() => {
            if (activeRuns === undefined) {
              return;
            }
            requestConfirm({
              prompt: `Stop scheduling and stop ${activeRuns.length} active runs?`,
              body: 'Milknado dispatches no more nodes. Each active run stops after its current turn. Done work stays in the graph.',
              dismissLabel: 'Keep running',
              confirmLabel: 'Stop runs',
              action: () => void stopScheduling(),
            });
          }}
        >
          Stop scheduling
        </Button>
      )}
      {isOwner && !stop.available && <p role="note">{stop.reason}</p>}
    </div>
  );
}