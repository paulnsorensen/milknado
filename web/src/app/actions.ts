// The user actions a feature can bind a handler to. The shell dispatches by
// id (a keyboard shortcut, a menu item); a feature registers the handler.
export const ACTION_IDS = [
  'run.select-next',
  'run.select-previous',
  'events.open',
  'changes.open',
  'detail.page-next',
  'detail.page-previous',
  'session.page-next',
  'session.page-previous',
  'session.follow-newest',
  'session.focus-input',
  'session.queue-guidance',
  'run.cancel',
  'run.force-stop',
  'scheduling.stop',
  'help.open',
] as const;

export type ActionId = (typeof ACTION_IDS)[number];

export type ActionHandler = () => void;

const actions = new Map<ActionId, ActionHandler>();

export function registerAction(id: ActionId, handler: ActionHandler): void {
  actions.set(id, handler);
}

export function dispatchAction(id: ActionId): void {
  actions.get(id)?.();
}

export function clearActions(): void {
  actions.clear();
}
