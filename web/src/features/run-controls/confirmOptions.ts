// The cancel and force-stop confirm copy, shared by the `run.cancel` /
// `run.force-stop` action handlers and the RunControlsSidecar buttons that
// dispatch them, so both surfaces show the same prompt.
export const CANCEL_RUN_CONFIRM = {
  prompt: 'Cancel this run?',
  dismissLabel: 'Keep the run',
  confirmLabel: 'Cancel run',
};

export const FORCE_STOP_CONFIRM = {
  prompt: 'Force stop the run?',
  body: 'The run stops now. It does not wait for the current turn. Changes that are not committed stay in the worktree.',
  dismissLabel: 'Keep the run',
  confirmLabel: 'Force stop',
};
