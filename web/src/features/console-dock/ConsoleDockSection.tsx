import type { ReactElement } from 'react';
import { useSyncExternalStore } from 'react';
import { getState, subscribe } from '../../app/store';
import { Milknado } from '../../design-system';
import type { StreamSnapshot } from '../live-state/runtimeSnapshot';
import { toConsoleLines } from './lines';

export function ConsoleDockSection(): ReactElement {
  const state = useSyncExternalStore(subscribe, getState);
  const { Console } = Milknado;
  const snapshot = state.snapshot as StreamSnapshot | null;
  const lines = snapshot ? toConsoleLines(snapshot.event_lines) : [];

  return <Console lines={lines} input={false} />;
}
