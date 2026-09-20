import type { ReactElement } from 'react';
import { useSyncExternalStore } from 'react';
import { getState, subscribe } from '../../app/store';
import { Milknado } from '../../design-system';
import { getDraft, registerInputEl, setDraft, subscribeDraft } from './draft';
import { sendSessionCommand } from './sessionCommand';

type MessageAction = 'steer' | 'follow_up';

/** The `sidecar-section` contribution: the guidance draft and its send actions. */
export function SessionInputSection(): ReactElement {
  const store = useSyncExternalStore(subscribe, getState);
  const draft = useSyncExternalStore(subscribeDraft, getDraft);
  const { Button } = Milknado;
  const sessionInput = store.capabilities?.session_input;
  const actions = store.capabilities?.owner?.actions ?? [];

  function send(action: MessageAction): void {
    const text = draft;
    void sendSessionCommand(action, { text }).then((sent) => {
      if (sent && getDraft() === text) {
        setDraft('');
      }
    });
  }

  // No session backend reads interrupt text, so the draft stays for a later send.
  function interrupt(): void {
    void sendSessionCommand('interrupt', { text: '' });
  }

  if (!sessionInput?.available) {
    return <p role="note">{sessionInput?.reason ?? 'Session input is not available.'}</p>;
  }

  const textEmpty = draft.trim() === '';

  return (
    <div className="mk-session-input">
      <textarea
        aria-label="Session guidance"
        ref={registerInputEl}
        value={draft}
        onChange={(event) => setDraft(event.target.value)}
      />
      <Button onClick={() => send('steer')} disabled={textEmpty || !actions.includes('steer')}>
        Steer
      </Button>
      <Button
        onClick={() => send('follow_up')}
        disabled={textEmpty || !actions.includes('follow_up')}
      >
        Follow up
      </Button>
      <Button onClick={interrupt} disabled={!actions.includes('interrupt')}>
        Interrupt
      </Button>
    </div>
  );
}
