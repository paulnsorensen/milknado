// The `sidecar-action` and `header-control` run controls, their shared
// confirmation dialog, and the `run.*`/`scheduling.stop` action ids.
import { registerAction } from '../../app/actions';
import { registerSlot } from '../../app/slots';
import { getState } from '../../app/store';
import { cancelRun, forceStopRun, stopScheduling } from './commands';
import { ConfirmDialog } from './ConfirmDialog';
import { requestConfirm } from './confirmState';
import { RunControlsSidecar } from './RunControlsSidecar';
import { RunModeHeader } from './RunModeHeader';

function currentRunId(): string | undefined {
  return getState().capabilities?.owner.run_id;
}

export function register(): void {
  registerSlot('sidecar-action', () => <RunControlsSidecar />);
  registerSlot('header-control', () => <RunModeHeader />);
  registerSlot('dialog', () => <ConfirmDialog />);

  registerAction('run.cancel', () => {
    const runId = currentRunId() ?? '';
    requestConfirm('Cancel this run?', () => void cancelRun(runId));
  });
  registerAction('run.force-stop', () => {
    const runId = currentRunId() ?? '';
    requestConfirm('Force stop this run?', () => void forceStopRun(runId));
  });
  registerAction('scheduling.stop', () => {
    requestConfirm('Stop scheduling?', () => void stopScheduling());
  });
}
