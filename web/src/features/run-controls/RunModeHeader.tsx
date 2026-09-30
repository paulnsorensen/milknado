// The `header-control` contribution: the run-mode badge (Run active for an
// owner, Read-only for an observer) and Stop scheduling, gated by
// capabilities like the server.
import type { ReactElement } from 'react';
import { useSyncExternalStore } from 'react';
import { dispatchAction } from '../../app/actions';
import { getState, subscribe } from '../../app/store';
import { Milknado } from '../../design-system';
import { ownerLabel } from '../../shared/ownerLabel';

interface StopSchedulingButtonProps {
  stopAvailable: boolean;
  stopReason: string | null;
  activeRunsKnown: boolean;
}

function StopSchedulingButton({
  stopAvailable,
  stopReason,
  activeRunsKnown,
}: StopSchedulingButtonProps): ReactElement {
  const { Button } = Milknado;
  return (
    <Button
      disabled={!stopAvailable || !activeRunsKnown}
      title={
        !activeRunsKnown
          ? 'Run totals are not available yet.'
          : stopAvailable
            ? undefined
            : (stopReason ?? undefined)
      }
      onClick={() => {
        if (!activeRunsKnown) {
          return;
        }
        dispatchAction('scheduling.stop');
      }}
    >
      Stop scheduling
    </Button>
  );
}

export function RunModeHeader(): ReactElement | null {
  const store = useSyncExternalStore(subscribe, getState);
  const { StatusBadge } = Milknado;
  const capabilities = store.capabilities;

  if (!capabilities) {
    return null;
  }

  const isOwner = capabilities.host_owner.available;
  const stop = capabilities.stop_scheduling;
  const activeRuns = store.snapshot?.active_runs;

  return (
    <div className="mk-button-row">
      <StatusBadge state={isOwner ? 'running' : 'pending'}>{ownerLabel(isOwner).badge}</StatusBadge>
      {isOwner && (
        <StopSchedulingButton
          stopAvailable={stop.available}
          stopReason={stop.reason}
          activeRunsKnown={activeRuns !== undefined}
        />
      )}
      {isOwner && !stop.available && <p role="note">{stop.reason}</p>}
    </div>
  );
}