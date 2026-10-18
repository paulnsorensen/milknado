// The message list: operator bubbles on the right, agent bubbles grouped
// by speaker on the left, tool calls as chips, run gates as divider pills.
import type { ReactElement } from 'react';
import { useEffect, useRef } from 'react';
import type { Msg } from '../model/thread';
import { avatarClass, initials } from './current';
import { Icon } from './Icon';

function gapTop(messages: Msg[], index: number): number {
  const message = messages[index];
  const previous = messages[index - 1];
  if (!previous) {
    return 0;
  }
  if (previous.who === message.who && message.kind === 'agent' && previous.kind === 'agent') {
    return 2;
  }
  return message.kind === 'me' || previous.kind === 'me' ? 14 : 12;
}

function AgentBubble({ messages, index }: { messages: Msg[]; index: number }): ReactElement {
  const message = messages[index];
  const previous = messages[index - 1];
  const next = messages[index + 1];
  const grouped = previous?.kind === 'agent' && previous.who === message.who;
  const nextSame = next?.kind === 'agent' && next.who === message.who;
  const shape = grouped ? (nextSame ? 'is-mid' : 'is-last') : 'is-first';
  return (
    <div className="mh-msg-agent" style={{ marginTop: gapTop(messages, index) }}>
      <span className={`mh-av ${avatarClass(message.who)}`} style={{ opacity: nextSame ? 0 : 1 }}>{initials(message.who)}</span>
      <div className="mh-msg-col">
        {!grouped && (
          <span className={`mh-msg-name ${avatarClass(message.who)}`}>{message.who}
            {message.at && <span className="mh-msg-at">{message.at}</span>}</span>
        )}
        <div className={`mh-bubble ${shape}`}>{message.text}</div>
      </div>
    </div>
  );
}

function MessageRow({ messages, index }: { messages: Msg[]; index: number }): ReactElement {
  const message = messages[index];
  switch (message.kind) {
    case 'me':
      return (
        <div className="mh-msg-me" style={{ marginTop: gapTop(messages, index) }}>
          <div className="mh-bubble-me">{message.text}</div>
          {message.at && <span className="mh-msg-at">{message.at}</span>}
        </div>
      );
    case 'tool':
      return (
        <div className="mh-msg-tool"><span><Icon name="terminal" size={13} />{message.text}</span></div>
      );
    case 'gate':
      return (
        <div className={`mh-msg-gate is-${message.tone}`}>
          <span className="mh-rule" />
          <span className="mh-gate-pill" title={message.text}><span className="mh-dot" /><span className="mh-gate-text">{message.text}</span>
            {message.at && <span className="mh-gate-at">{message.at}</span>}</span>
          <span className="mh-rule is-right" />
        </div>
      );
    default:
      return <AgentBubble messages={messages} index={index} />;
  }
}

interface ThreadProps {
  messages: Msg[];
  typingWho: string | null;
}

export function Thread({ messages, typingWho }: ThreadProps): ReactElement {
  const scroller = useRef<HTMLDivElement | null>(null);
  const last = messages.at(-1)?.key;
  useEffect(() => {
    const element = scroller.current;
    if (element) {
      element.scrollTop = element.scrollHeight;
    }
  }, [last, messages.length, typingWho]);
  return (
    <div className="mh-thread" ref={scroller} aria-label="Conversation" role="log">
      <div className="mh-thread-inner">
        {messages.map((message, index) => (
          <div key={message.key} className="mh-msg"><MessageRow messages={messages} index={index} /></div>
        ))}
        {typingWho && (
          <div className="mh-msg-agent mh-typing" aria-label={`${typingWho} is working`}>
            <span className={`mh-av ${avatarClass(typingWho)}`}>{initials(typingWho)}</span>
            <div className="mh-typing-dots"><span /><span /><span /></div>
          </div>
        )}
      </div>
    </div>
  );
}
