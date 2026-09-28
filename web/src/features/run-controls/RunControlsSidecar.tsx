// The `sidecar-action` contribution: Cancel run and Force stop for the
// owner's run. Cancel is always active (the server rejects it with a domain
// reason when there is no live run); Force stop is gated by capabilities,
// matching the server's owner-only enforcement.
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
    <section className="mk-section" aria-label="Run controls">
      <div className="mk-button-row">
        <Button
          className="mk-btn-sm"
          disabled={runId === ''}
          onClick={() =>
            requestConfirm({
              prompt: 'Cancel this run?',
              action: () => void cancelRun(runId),
              dismissLabel: 'Keep the run',
              confirmLabel: 'Cancel run',
            })
          }
        >
          Cancel run
        </Button>
        <Button
          className="mk-btn-sm"
          disabled={!forceStop.available}
          onClick={() =>
            requestConfirm({
              prompt: 'Force stop the run?',
              body: 'The run stops now. It does not wait for the current turn. Changes that are not committed stay in the worktree.',
              dismissLabel: 'Keep the run',
              confirmLabel: 'Force stop',
              action: () => void forceStopRun(runId),
            })
          }
        >
          Force stop
        </Button>
      </div>
      {!forceStop.available && (
        <p role="note" className="mk-text-caption mk-muted">
          {forceStop.reason}
        </p>
      )}
    </section>
  );
}