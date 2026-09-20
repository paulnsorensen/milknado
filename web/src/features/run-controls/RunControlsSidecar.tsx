// The `sidecar-action` contribution: Cancel run and Force stop. Cancel is
// always active (the server rejects it with a domain reason when there is
// no live run); Force stop is gated by capabilities, matching the server's
// owner-only enforcement.
import type { ReactElement } from 'react';
import { useSyncExternalStore } from 'react';
import { getState, subscribe } from '../../app/store';
import { Milknado } from '../../design-system';
import { cancelRun, forceStopRun } from './commands';
import { requestConfirm } from './confirmState';

export function RunControlsSidecar(): ReactElement | null {
  const store = useSyncExternalStore(subscribe, getState);
  const { Button } = Milknado;
  const capabilities = store.capabilities;

  if (!capabilities) {
    return null;
  }

  const runId = capabilities.owner.run_id ?? '';
  const forceStop = capabilities.force_stop;

  return (
    <div className="mk-run-controls">
      <Button
        disabled={runId === ''}
        onClick={() => requestConfirm('Cancel this run?', () => void cancelRun(runId))}
      >
        Cancel run
      </Button>
      <Button
        disabled={!forceStop.available}
        onClick={() => requestConfirm('Force stop this run?', () => void forceStopRun(runId))}
      >
        Force stop
      </Button>
      {!forceStop.available && <p role="note">{forceStop.reason}</p>}
    </div>
  );
}
