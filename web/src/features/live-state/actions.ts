// Handlers for the `events.open`, `run.select-next`, and `run.select-previous`
// action ids (declared in app/actions.ts) that this feature registers.
import { getState, setActiveSidecar, setSelection } from '../../app/store';
import type { StreamSnapshot } from './runtimeSnapshot';

function runIds(): string[] {
  const snapshot = getState().snapshot as StreamSnapshot | null;
  return snapshot?.active_runs.map((run) => run.run_id) ?? [];
}

function selectRunAt(offset: number): void {
  const ids = runIds();
  if (ids.length === 0) {
    return;
  }
  const current = getState().selection;
  const index = typeof current === 'string' ? ids.indexOf(current) : -1;
  const next = index === -1 ? 0 : (index + offset + ids.length) % ids.length;
  setSelection(ids[next]);
}

export function selectNextRun(): void {
  selectRunAt(1);
}

export function selectPreviousRun(): void {
  selectRunAt(-1);
}

export function openEvents(): void {
  setActiveSidecar('events');
}
