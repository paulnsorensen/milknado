import type { ReactElement } from 'react';
import { useSyncExternalStore } from 'react';
import { ownerLabel } from '../../shared/ownerLabel';
import { getState, subscribe } from '../store';

/** The mode kicker (Run for an owner, Watch for an observer) over the goal title. */
export function GoalTitle(): ReactElement {
  const store = useSyncExternalStore(subscribe, getState);
  const owner = store.capabilities?.owner.available ?? false;

  return (
    <div className="mk-goal-title">
      <span className="mk-kicker">{ownerLabel(owner).kicker}</span>
      <h1>{store.snapshot?.goal ?? 'Milknado'}</h1>
    </div>
  );
}
