// The `sidecar-action` contribution: Cancel run and Force stop for the
// owner's selected active run. A selected non-owner node sees disabled
// controls with the server reason; no node selection renders no controls.
import type { ReactElement } from 'react';
import { useSyncExternalStore } from 'react';
import { canActOnSelectedRun, getState, selectedNodeId, subscribe } from '../../app/store';
import { Milknado } from '../../design-system';
import { cancelRun, forceStopRun } from './commands';
import { requestConfirm } from './confirmState';

export function RunControlsSidecar(): ReactElement | null {
  const store = useSyncExternalStore(subscribe, getState);
  const { Button } = Milknado;
  const capabilities = store.capabilities;
  const canAct = canActOnSelectedRun(store);

  if (!capabilities || selectedNodeId(store) === null) {
    return null;
  }

  const runId = capabilities.owner.run_id ?? '';
  const forceStop = capabilities.force_stop;
  return (
    <section className="mk-section" aria-label="Run controls">
      <div className="mk-button-row">
        <Button
          className="mk-btn-sm"
          disabled={!canAct || runId === ''}
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
          disabled={!canAct || !forceStop.available}
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