// The message composer. A running worker gets the text as steering on its
// next poll; a failed node retries with it; anything else saves it to the
// node brief, where the next worker reads it.
import type { ReactElement } from 'react';
import { useState } from 'react';
import { addNote, retryNode, steer } from '../data/commands';
import { loadNode } from '../data/nodeData';
import { isGoal } from '../model/tree';
import { flash, getMh, patchMh } from '../viewState';
import type { Current } from './current';
import { avatarClass, DELIVERY_TEXT, placeholder } from './current';
import { Icon } from './Icon';

async function deliver(current: Current, text: string): Promise<boolean> {
  const { node, live, delivery } = current;
  if (delivery === 'steer' && live) {
    return steer(live, text);
  }
  if (delivery === 'retry') {
    const sent = await retryNode(node.id, text);
    if (sent) {
      patchMh({ compose: 'message' });
      flash(`Node ${node.id} retrying with your note`);
    }
    return sent;
  }
  const saved = await addNote(node.id, text);
  if (saved) {
    flash(`Saved to the brief for node ${node.id}`);
  }
  return saved;
}

export function Composer({ current }: { current: Current }): ReactElement {
  const [draft, setDraft] = useState('');
  const [sending, setSending] = useState(false);
  const ready = draft.trim() !== '' && !sending;
  const send = (): void => {
    const text = draft.trim();
    if (!ready) {
      return;
    }
    setSending(true);
    void deliver(current, text).then((sent) => {
      if (!sent) {
        return;
      }
      setDraft('');
      // The operator may have moved on while the send was in flight. Leave
      // the new selection's view and data alone.
      if (getMh().sel === current.node.id) {
        patchMh({ view: 'thread' });
        void loadNode(current.node.id, isGoal(current.node));
      }
    }).finally(() => setSending(false));
  };
  return (
    <div className="mh-composer-wrap">
      <div className="mh-composer">
        <input data-mh-composer aria-label="Message" placeholder={placeholder(current)} value={draft}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter') {
              event.preventDefault();
              send();
            }
            if (event.key === 'Escape' && current.delivery === 'retry') {
              patchMh({ compose: 'message' });
            }
          }} />
        <div className="mh-composer-row">
          <span className="mh-to"><span className={`mh-av-dot ${avatarClass(current.who)}`} />{current.who}</span>
          <span className="mh-delivery">{DELIVERY_TEXT[current.delivery]}</span>
          <button type="button" className={`mh-send${ready ? ' is-ready' : ''}`} title="Send (Enter)"
            aria-label="Send" onClick={send} disabled={!ready}><Icon name="send" size={15} /></button>
        </div>
      </div>
    </div>
  );
}
