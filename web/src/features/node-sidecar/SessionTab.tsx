import type { ReactElement } from 'react';
import { dispatchAction } from '../../app/actions';
import { Milknado } from '../../design-system';
import type { WireSessionEvent, WireSessionEventKind } from '../../shared/node-detail';

export interface SessionTabProps {
  events: WireSessionEvent[];
  hasMore: boolean;
}

type LineTone = 'plain' | 'result' | 'noise' | 'warn';

const TONE_FOR: Partial<Record<WireSessionEventKind, LineTone>> = {
  status: 'noise',
  error: 'warn',
  permission: 'warn',
};

/** The Session tab body: the transcript console, plus paging and follow-newest. */
export function SessionTab({ events, hasMore }: SessionTabProps): ReactElement {
  const { Console, Button } = Milknado;
  const lines = events.map((event) => ({
    time: event.kind,
    text: event.text,
    tone: TONE_FOR[event.kind] ?? 'plain',
  }));

  return (
    <div className="mk-stack">
      <Console
        className="mk-session"
        lines={lines}
        tabs={[]}
        input={false}
        emptyTitle="No session transcript yet"
        emptyHint="Lines appear when the agent starts its first turn."
      />
      <div className="mk-button-row">
        <Button className="mk-btn-sm" onClick={() => dispatchAction('session.page-previous')}>
          Previous
        </Button>
        <Button className="mk-btn-sm" disabled={!hasMore} onClick={() => dispatchAction('session.page-next')}>
          Next
        </Button>
        <Button variant="ghost" className="mk-btn-sm" onClick={() => dispatchAction('session.follow-newest')}>
          Follow newest
        </Button>
      </div>
    </div>
  );
}