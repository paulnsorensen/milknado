// The owner-capability labels shared by the rail header, the narrow layout,
// and the run-mode header: an owner is running the goal, an observer watches.
export interface OwnerLabel {
  kicker: 'Run' | 'Watch';
  badge: 'Run active' | 'Read-only';
}

export function ownerLabel(owner: boolean): OwnerLabel {
  return owner ? { kicker: 'Run', badge: 'Run active' } : { kicker: 'Watch', badge: 'Read-only' };
}
