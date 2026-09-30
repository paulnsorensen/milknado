// The `sidecar-action` contribution: Cancel run and Force stop for the
// owner's selected active run. A selected non-owner node sees disabled
// controls with the server reason; no node selection, or a watch host
// (host_owner unavailable), renders no controls.
import type { ReactElement } from 'react';
import { useSyncExternalStore } from 'react';
import { dispatchAction } from '../../app/actions';
import { canActOnSelectedRun, getState, selectedNodeId, subscribe } from '../../app/store';
import { Milknado } from '../../design-system';

interface RunControlButtonsProps {
  canAct: boolean;
  runId: string;
  forceStopAvailable: boolean;
}

function RunControlButtons({ canAct, runId, forceStopAvailable }: RunControlButtonsProps): ReactElement {
  const { Button } = Milknado;
  return (
    <div className="mk-button-row">
      <Button
        className="mk-btn-sm"
        disabled={!canAct || runId === ''}
        onClick={() => dispatchAction('run.cancel')}
      >
        Cancel run
      </Button>
      <Button
        className="mk-btn-sm"
        disabled={!canAct || !forceStopAvailable}
        onClick={() => dispatchAction('run.force-stop')}
      >
        Force stop
      </Button>
    </div>
  );
}

export function RunControlsSidecar(): ReactElement | null {
  const store = useSyncExternalStore(subscribe, getState);
  const capabilities = store.capabilities;
  const canAct = canActOnSelectedRun(store);

  if (!capabilities || selectedNodeId(store) === null || !capabilities.host_owner.available) {
    return null;
  }

  const runId = capabilities.owner.run_id ?? '';
  const forceStop = capabilities.force_stop;
  return (
    <section className="mk-section" aria-label="Run controls">
      <RunControlButtons canAct={canAct} runId={runId} forceStopAvailable={forceStop.available} />
      {!forceStop.available && (
        <p role="note" className="mk-text-caption mk-muted">
          {forceStop.reason}
        </p>
      )}
    </section>
  );
}