// The center pane: the selected node's conversation, or its diff, with
// the composer below and the header's next-decision and diff controls.
import type { ReactElement } from 'react';
import { useEffect } from 'react';
import { useCell } from '../data/cell';
import type { NodeData } from '../data/nodeData';
import { nodeData } from '../data/nodeData';
import { diffTotals, loadRunDiff, runDiff } from '../data/runDiff';
import { goalThread, taskThread } from '../model/thread';
import type { Msg } from '../model/thread';
import { isGoal, title } from '../model/tree';
import { patchMh } from '../viewState';
import { nextDecision } from './actions';
import { Composer } from './Composer';
import type { Current } from './current';
import { avatarClass, initials, needSummary } from './current';
import { DiffView } from './DiffView';
import { confirmCancel } from './edits';
import { Icon } from './Icon';
import { Thread } from './Thread';
import type { MhModel } from './useMhModel';

function threadFor(model: MhModel, current: Current, data: NodeData): Msg[] {
  const { node, live } = current;
  const detail = data.nodeId === node.id ? data.detail : null;
  if (isGoal(node)) {
    return goalThread({ node, tree: model.tree, reviews: model.reviews, records: data.nodeId === node.id ? data.records : [] });
  }
  const gates = model.info?.flavors[node.flavor ?? 'implement']?.quality_gates ?? [];
  return taskThread({ node, detail, live, now: Date.now(), gates });
}

function typingWho(current: Current): string | null {
  const last = current.live?.session?.events.at(-1);
  return current.live && (!last || last.kind === 'user') ? current.who : null;
}

function tag(current: Current): { text: string; tone: string } {
  if (current.need) {
    return { text: 'needs you', tone: 'need' };
  }
  return { text: current.node.status, tone: current.node.status };
}

function ChatHeader({ model, current }: { model: MhModel; current: Current }): ReactElement {
  const diff = useCell(runDiff);
  const { node, who } = current;
  const showDiff = diff.runId === current.runId && diff.files.length > 0 && model.ui.view !== 'diff';
  const totals = diffTotals(diff.files);
  const badge = tag(current);
  return (
    <header className="mh-chat-head">
      <span className={`mh-av is-lg ${avatarClass(who)}`}>{initials(who)}</span>
      <div className="mh-chat-title">
        <div className="mh-chat-title-row"><span className="mh-chat-name">{title(node)}</span>
          <span className={`mh-tag is-${badge.tone}`}>{badge.text}</span></div>
        <span className="mh-kicker-line">#{node.id} · {who}{node.flavor ? ` · ${node.flavor}` : ''}</span>
      </div>
      <div className="mh-chat-actions">
        {model.needs.size > 0 && (
          <button type="button" className="mh-needs" title="Jump to the next decision (N)" onClick={() => nextDecision(model)}>
            Needs you<span className="mh-needs-count">{model.needs.size}</span></button>
        )}
        {showDiff && (
          <button type="button" className="mh-pill-btn" onClick={() => patchMh({ view: 'diff', pop: null })}>
            <Icon name="diff" size={14} />Diff<span className="mh-mono mh-add">+{totals.added}</span>
            <span className="mh-mono mh-del">−{totals.removed}</span></button>
        )}
        {isGoal(node) && model.tree.kids.has(node.id) && (
          <button type="button" className="mh-round-btn" title="Focus graph on this goal" aria-label="Focus graph on this goal"
            onClick={() => patchMh({ scopeRoot: node.id })}><Icon name="focus" size={15} /></button>
        )}
        {current.live && (
          <button type="button" className="mh-round-btn" title="Cancel this run" aria-label="Cancel run"
            onClick={() => confirmCancel(model, node.id)}><Icon name="stop" size={15} /></button>
        )}
        <button type="button" className={`mh-round-btn${model.ui.side === 'details' ? ' is-on' : ''}`} title="Details"
          aria-label="Details" onClick={() => patchMh((ui) => ({ side: ui.side === 'details' ? null : 'details' }))}>
          <Icon name="info" size={16} /></button>
      </div>
    </header>
  );
}

export function ChatPane({ model, current }: { model: MhModel; current: Current | null }): ReactElement {
  const data = useCell(nodeData);
  const diff = useCell(runDiff);
  const runId = current?.runId ?? null;
  useEffect(() => { void loadRunDiff(runId); }, [runId]);
  if (!current) {
    return (
      <section className="mh-chat"><div className="mh-empty">Waiting for the graph snapshot…</div></section>
    );
  }
  const showDiff = model.ui.view === 'diff' && diff.runId === runId && (diff.files.length > 0 || diff.loading);
  const summary = current.need ? needSummary(current.need, data) : '';
  return (
    <section className="mh-chat" aria-label="Conversation pane">
      <ChatHeader model={model} current={current} />
      {showDiff
        ? <DiffView diff={diff} current={current} summary={summary} model={model} />
        : <Thread messages={threadFor(model, current, data)} typingWho={typingWho(current)} />}
      <Composer current={current} />
      {model.ui.toast && (
        <div className="mh-toast" role="status"><Icon name="check" size={15} />{model.ui.toast}</div>
      )}
    </section>
  );
}
