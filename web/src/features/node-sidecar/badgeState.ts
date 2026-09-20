// A shared status-to-badge mapper: the wire status strings (node status,
// run status) do not always match the design system's `State` union, so an
// unknown value falls back to 'pending' instead of an unsafe cast.
const BADGE_STATES = ['pending', 'ready', 'running', 'at-risk', 'failed', 'blocked', 'done'] as const;

export type BadgeState = (typeof BADGE_STATES)[number];

// Wire statuses without an identically named badge state.
const STATUS_ALIASES: Record<string, BadgeState> = {
  completed: 'done',
  stopped: 'blocked',
};

export function toBadgeState(status: string): BadgeState {
  const alias = STATUS_ALIASES[status];
  if (alias) {
    return alias;
  }
  return (BADGE_STATES as readonly string[]).includes(status) ? (status as BadgeState) : 'pending';
}
