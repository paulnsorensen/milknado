// The `run-controls` and `header-control` contributions, their shared
// confirmation dialog, and the `run.*`/`scheduling.stop` action ids.
import { registerAction } from '../../app/actions';
import { registerSlot } from '../../app/slots';
import { getState, pushNotice } from '../../app/store';
import { cancelRun, forceStopRun, stopScheduling } from './commands';
import { ConfirmDialog } from './ConfirmDialog';
import { CANCEL_RUN_CONFIRM, FORCE_STOP_CONFIRM } from './confirmOptions';
import { requestConfirm } from './confirmState';
import { RunControlsSidecar } from './RunControlsSidecar';
import { RunModeHeader } from './RunModeHeader';

function currentRunId(): string | undefined {
  return getState().capabilities?.owner.run_id;
}

export function register(): void {
  registerSlot('run-controls', () => <RunControlsSidecar />);
  registerSlot('header-control', () => <RunModeHeader />);
  registerSlot('dialog', () => <ConfirmDialog />);

  registerAction('run.cancel', () => {
    const runId = currentRunId() ?? '';
    requestConfirm({ ...CANCEL_RUN_CONFIRM, action: () => void cancelRun(runId) });
  });
  registerAction('run.force-stop', () => {
    const runId = currentRunId() ?? '';
    requestConfirm({ ...FORCE_STOP_CONFIRM, action: () => void forceStopRun(runId) });
  });
  registerAction('scheduling.stop', () => {
    const activeRuns = getState().snapshot?.active_runs;
    if (activeRuns === undefined) {
      pushNotice('Run totals are not available yet.');
      return;
    }
    requestConfirm({
      prompt: `Stop scheduling and stop ${activeRuns.length} active ${activeRuns.length === 1 ? 'run' : 'runs'}?`,
      body: 'Milknado dispatches no more nodes. Each active run stops after its current turn. Done work stays in the graph.',
      dismissLabel: 'Keep running',
      confirmLabel: 'Stop runs',
      action: () => void stopScheduling(),
    });
  });
}
