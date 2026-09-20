import type { ReactElement } from 'react';
import { dispatchAction } from '../../app/actions';
import { Milknado } from '../../design-system';
import type { WireSessionEvent } from './detailWire';

export interface SessionTabProps {
  events: WireSessionEvent[];
}

/** The Session tab body: the transcript, plus paging and follow-newest. */
export function SessionTab({ events }: SessionTabProps): ReactElement {
  const { Console, Button } = Milknado;
  const lines = events.map((event) => ({ time: '', text: `${event.kind}: ${event.text}` }));

  return (
    <div>
      <Console lines={lines} input={false} emptyTitle="No session activity yet." />
      <Button onClick={() => dispatchAction('session.page-previous')}>Previous</Button>
      <Button onClick={() => dispatchAction('session.page-next')}>Next</Button>
      <Button onClick={() => dispatchAction('session.follow-newest')}>Follow newest</Button>
    </div>
  );
}
