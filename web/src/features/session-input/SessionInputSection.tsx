import type { ReactElement } from 'react';
import { useSyncExternalStore } from 'react';
import { getState, subscribe } from '../../app/store';
import { Milknado } from '../../design-system';
import { getDraft, registerInputEl, setDraft, subscribeDraft } from './draft';
import { sendSessionCommand } from './sessionCommand';

type SendAction = 'steer' | 'follow_up' | 'interrupt';

/** The `sidecar-section` contribution: the guidance draft and its send actions. */
export function SessionInputSection(): ReactElement {
  const store = useSyncExternalStore(subscribe, getState);
  const draft = useSyncExternalStore(subscribeDraft, getDraft);
  const { Button } = Milknado;
  const sessionInput = store.capabilities?.session_input;

  function send(action: SendAction): void {
    const text = draft;
    setDraft('');
    void sendSessionCommand(action, { text });
  }

  if (sessionInput && !sessionInput.available) {
    return <p role="note">{sessionInput.reason ?? 'Session input is not available.'}</p>;
  }

  return (
    <div className="mk-session-input">
      <textarea
        aria-label="Session guidance"
        ref={registerInputEl}
        value={draft}
        onChange={(event) => setDraft(event.target.value)}
      />
      <Button onClick={() => send('steer')}>Steer</Button>
      <Button onClick={() => send('follow_up')}>Follow up</Button>
      <Button onClick={() => send('interrupt')}>Interrupt</Button>
    </div>
  );
}
