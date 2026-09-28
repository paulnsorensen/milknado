import type { ReactElement } from 'react';
import { useState, useSyncExternalStore } from 'react';
import { getState, subscribe } from '../../app/store';
import { Milknado } from '../../design-system';
import { getDraft, registerInputEl, setDraft, subscribeDraft } from './draft';
import { sendSessionCommand } from './sessionCommand';

type MessageAction = 'steer' | 'follow_up';
type SessionAction = MessageAction | 'interrupt';
const SESSION_ACTIONS: SessionAction[] = ['steer', 'follow_up', 'interrupt'];

/** The `sidecar-section` contribution: the guidance draft and its send actions. */
export function SessionInputSection(): ReactElement {
  const store = useSyncExternalStore(subscribe, getState);
  const draft = useSyncExternalStore(subscribeDraft, getDraft);
  const { Button } = Milknado;
  const sessionInput = store.capabilities?.session_input;
  const actions = store.capabilities?.owner?.actions ?? [];
  const firstAllowedAction =
    SESSION_ACTIONS.find((action) => actions.includes(action)) ?? 'steer';
  const [selectedAction, setSelectedAction] = useState<SessionAction>(firstAllowedAction);
  const activeAction = actions.includes(selectedAction) ? selectedAction : firstAllowedAction;

  function sendMessage(action: MessageAction): void {
    const text = draft;
    void sendSessionCommand(action, { text }).then((sent) => {
      if (sent && getDraft() === text) {
        setDraft('');
      }
    });
  }

  function sendSelectedAction(): void {
    if (activeAction === 'interrupt') {
      void sendSessionCommand('interrupt', { text: '' }).then((sent) => {
        if (sent) {
          setSelectedAction(firstAllowedAction);
        }
      });
      return;
    }
    sendMessage(activeAction);
  }

  if (!sessionInput?.available) {
    return (
      <p role="note" className="mk-note">
        {sessionInput?.reason ?? 'Session input is not available.'}
      </p>
    );
  }

  const textEmpty = draft.trim() === '';

  return (
    <section className="mk-stack" aria-label="Session input">
      <span className="mk-kicker is-live">Session input</span>
      <textarea
        className="mk-input"
        aria-label="Session guidance"
        rows={2}
        placeholder="Guidance for the agent"
        ref={registerInputEl}
        value={draft}
        onChange={(event) => setDraft(event.target.value)}
      />
      <div className="mk mk-seg" role="group" aria-label="Session action">
        <button
          type="button"
          className={activeAction === 'steer' ? 'mk-seg-opt is-on' : 'mk-seg-opt'}
          aria-pressed={activeAction === 'steer'}
          onClick={() => setSelectedAction('steer')}
          disabled={!actions.includes('steer')}
        >
          Steer
        </button>
        <button
          type="button"
          className={activeAction === 'follow_up' ? 'mk-seg-opt is-on' : 'mk-seg-opt'}
          aria-pressed={activeAction === 'follow_up'}
          onClick={() => setSelectedAction('follow_up')}
          disabled={!actions.includes('follow_up')}
        >
          Follow up
        </button>
        <button
          type="button"
          className={activeAction === 'interrupt' ? 'mk-seg-opt is-on' : 'mk-seg-opt'}
          aria-pressed={activeAction === 'interrupt'}
          onClick={() => setSelectedAction('interrupt')}
          disabled={!actions.includes('interrupt')}
        >
          Interrupt
        </button>
      </div>
      <Button
        variant="primary"
        className="mk-btn-sm mk-session-send"
        onClick={sendSelectedAction}
        disabled={
          activeAction === 'interrupt'
            ? !actions.includes('interrupt')
            : textEmpty || !actions.includes(activeAction)
        }
      >
        Send
      </Button>
    </section>
  );
}