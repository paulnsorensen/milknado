// The `dock` slot contribution: the coordinator bar with the newest event
// line and an Events toggle, plus the events console while it is open. The
// store's `activeSidecar === 'events'` is the open flag, so the `events.open`
// action (the `e` key) opens the same console.
import type { ReactElement } from 'react';
import { useSyncExternalStore } from 'react';
import { getState, setActiveSidecar, subscribe } from '../../app/store';
import { Milknado } from '../../design-system';
import { getConnectionStatus, subscribeConnection } from '../live-state/connection';
import type { StreamSnapshot } from '../live-state/runtimeSnapshot';
import { toConsoleLines } from './lines';

export function ConsoleDockSection(): ReactElement {
  const state = useSyncExternalStore(subscribe, getState);
  const connectionStatus = useSyncExternalStore(subscribeConnection, getConnectionStatus);
  const { Button, Console } = Milknado;
  const snapshot = state.snapshot as Partial<StreamSnapshot> | null;
  const eventLines = snapshot?.event_lines ?? [];
  const open = state.activeSidecar === 'events';
  const live = connectionStatus === 'connected';
  const last = eventLines[eventLines.length - 1] ?? 'No events yet';

  return (
    <>
      {open && (
        <div className="mk-dock-console">
          <Console
            lines={toConsoleLines(eventLines)}
            tabs={['Events']}
            activeTab="Events"
            input={false}
            emptyTitle="No events yet"
            emptyHint="Events appear when a run starts."
          />
        </div>
      )}
      <div className="mk-coordinator-bar">
        <span className={live ? 'mk-live-dot is-live' : 'mk-live-dot'} aria-hidden="true" />
        <span className={live ? 'mk-kicker is-live' : 'mk-kicker'}>Coordinator</span>
        <span className="mk-last-event">{last}</span>
        <Button
          variant="ghost"
          count={eventLines.length}
          onClick={() => setActiveSidecar(open ? null : 'events')}
        >
          {open ? 'Hide events' : 'Events'}
        </Button>
      </div>
    </>
  );
}