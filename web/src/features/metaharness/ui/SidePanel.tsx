// The right panel: node details, the knowledge pages workers read, open
// issues with a one-click fix task, and the session stats.
import type { ReactElement } from 'react';
import type { WireNode } from '../../../app/wire';
import { useCell } from '../data/cell';
import { queueFix } from '../data/commands';
import { nodeData } from '../data/nodeData';
import { deriveIssues, deriveStats } from '../model/insights';
import { ago, splitBrief } from '../model/thread';
import { childrenOf, isGoal, rootOf, title } from '../model/tree';
import { flash, patchMh } from '../viewState';
import type { Current } from './current';
import { stateClass } from './nodeView';
import { Icon } from './Icon';
import type { MhModel } from './useMhModel';

function NodeLink({ node, model, icon }: { node: WireNode; model: MhModel; icon?: boolean }): ReactElement {
  return (
    <button type="button" className="mh-link-row" onClick={() => patchMh({ sel: node.id, view: 'thread' })}>
      {icon ? <Icon name="unblocks" size={14} /> : <span className={`mh-dot ${stateClass(node, model)}`} />}{title(node)}
    </button>
  );
}

function gateResult(node: WireNode): { text: string; tone: string } {
  switch (node.status) {
    case 'failed':
      return { text: 'FAIL', tone: 'bad' };
    case 'done':
      return { text: 'PASS', tone: 'ok' };
    case 'running':
      return { text: 'running…', tone: 'run' };
    default:
      return { text: 'not run', tone: 'mute' };
  }
}

function Gates({ model, node }: { model: MhModel; node: WireNode }): ReactElement {
  const kids = childrenOf(model.tree, node.id);
  if (isGoal(node)) {
    return <div className="mh-gate-row"><span>all prerequisites done</span>
      <span className="mh-muted">{kids.filter((kid) => kid.status === 'done').length}/{kids.length}</span></div>;
  }
  const gates = model.info?.flavors[node.flavor ?? 'implement']?.quality_gates ?? [];
  const result = gateResult(node);
  if (gates.length === 0) {
    return <div className="mh-gate-row"><span>no gates configured</span><span className={`is-${result.tone}`}>{result.text}</span></div>;
  }
  return <>{gates.map((gate) => (
    <div key={gate} className="mh-gate-row"><span>{gate}</span><span className={`is-${result.tone}`}>{result.text}</span></div>
  ))}</>;
}

function Details({ model, current }: { model: MhModel; current: Current }): ReactElement {
  const data = useCell(nodeData);
  const { node, live } = current;
  const detail = data.nodeId === node.id ? data.detail : null;
  const kids = childrenOf(model.tree, node.id);
  const parent = node.parent_id === null ? undefined : model.tree.byId.get(node.parent_id);
  const dependents = (detail?.dependent_ids.items ?? []).filter((id) => id !== node.parent_id)
    .map((id) => model.tree.byId.get(id)).filter((dep): dep is WireNode => !!dep);
  const context = live?.session?.context;
  const worktree = context?.cwd ?? node.worktree_path;
  const rows: Array<[string, string]> = [
    ['agent', current.who],
    ['branch', node.branch_name ?? '—'],
    ['worktree', worktree?.split('/').at(-1) ?? '—'],
    ['harness', context?.family ?? '—'],
    ['iteration', live?.attempt ? `${live.attempt} / ${live.max_attempts ?? '?'}` : '—'],
    ['runs', String(detail?.runs.total ?? detail?.runs.items?.length ?? 0)],
  ];
  return (
    <div className="mh-side-stack">
      <section className="mh-side-sec"><span className="mh-label">Brief</span>
        <p className="mh-brief">{splitBrief(node.description).brief || title(node)}</p></section>
      <dl className="mh-kv">{rows.map(([key, value]) => (
        <div key={key}><dt>{key}</dt><dd>{value}</dd></div>))}</dl>
      <section className="mh-side-sec"><span className="mh-label">Quality gates</span><Gates model={model} node={node} /></section>
      <section className="mh-side-sec"><span className="mh-label">Prerequisites</span>
        {kids.map((kid) => <NodeLink key={kid.id} node={kid} model={model} />)}
        {kids.length === 0 && <span className="mh-faint">None. This is a leaf.</span>}</section>
      {(parent || dependents.length > 0) && (
        <section className="mh-side-sec"><span className="mh-label">Unblocks</span>
          {[parent, ...dependents].filter((dep): dep is WireNode => !!dep).map((dep) => (
            <NodeLink key={dep.id} node={dep} model={model} icon />))}</section>
      )}
    </div>
  );
}

function updatedLabel(age: string): string {
  if (!age) {
    return '';
  }
  return age === 'now' ? 'just now' : `${age} ago`;
}

function Knowledge({ model }: { model: MhModel }): ReactElement {
  const pages = model.info?.knowledge ?? [];
  const now = Date.now();
  return (
    <div className="mh-side-list">
      <span className="mh-muted">Every worker reads these before its first iteration.</span>
      {pages.map((page) => (
        <div key={page.path} className="mh-card" title={page.path}>
          <span className="mh-card-head"><span className="mh-strong">{page.title}</span>
            <span className="mh-faint">{updatedLabel(ago(page.updated_at, now))}</span></span>
          <span className="mh-muted">{page.summary}</span>
        </div>
      ))}
      {pages.length === 0 && <span className="mh-faint">No wiki pages under .hallouminate/wiki.</span>}
    </div>
  );
}

function Issues({ model }: { model: MhModel }): ReactElement {
  const issues = deriveIssues(model.snapshot, model.tree);
  const queue = (text: string, nodeId: number | null): void => {
    const scope = nodeId ?? model.ui.projectId ?? model.tree.roots[0]?.id ?? null;
    const goalId = scope === null ? null : (rootOf(model.tree, scope) ?? scope);
    void queueFix(text, goalId === null ? undefined : model.tree.byId.get(goalId))
      .then((ok) => ok && flash('Fix task queued for the next free worker'));
  };
  return (
    <div className="mh-side-list">
      {issues.map((issue) => (
        <div key={issue.id} className="mh-card">
          <span className="mh-card-head"><span className={`mh-sev is-${issue.severity}`}>{issue.severity}</span>
            <button type="button" className="mh-chip-btn" onClick={() => queue(issue.text, issue.nodeId)}>Queue fix</button></span>
          <span className="mh-body">{issue.text}</span>
          <span className="mh-mono mh-faint">{issue.target}</span>
        </div>
      ))}
      {issues.length === 0 && <span className="mh-faint">No open issues.</span>}
    </div>
  );
}

function Stats({ model }: { model: MhModel }): ReactElement {
  const stats = deriveStats({
    snapshot: model.snapshot, tree: model.tree,
    concurrency: model.info?.concurrency ?? null, paused: model.info?.scheduling.paused ?? false,
  });
  return (
    <div className="mh-stats">{stats.map((stat) => (
      <div key={stat.label} className="mh-stat"><span className="mh-label">{stat.label}</span>
        <span className="mh-stat-value">{stat.value}</span><span className="mh-faint">{stat.sub}</span></div>
    ))}</div>
  );
}

function openIssues(count: number): string {
  return `${count} open ${count === 1 ? 'issue' : 'issues'}`;
}

const HEAD = {
  details: (current: Current | null) => [`Node #${current?.node.id ?? ''}`, current ? title(current.node) : ''],
  knowledge: () => ['Project', 'Knowledge'],
  issues: (_current: Current | null, model: MhModel) => ['Health', openIssues(deriveIssues(model.snapshot, model.tree).length)],
  stats: () => ['Session', 'Stats'],
} as const;

export function SidePanel({ model, current }: { model: MhModel; current: Current | null }): ReactElement | null {
  const side = model.ui.side;
  if (side === null) {
    return null;
  }
  const [kicker, heading] = HEAD[side](current, model);
  return (
    <aside className="mh-side" aria-label={heading}>
      <div className="mh-side-head">
        <div className="mh-side-title"><span className="mh-side-kicker">{kicker}</span><span className="mh-side-name">{heading}</span></div>
        <button type="button" className="mh-round-btn is-sm" aria-label="Close panel" onClick={() => patchMh({ side: null })}>
          <Icon name="close" size={14} /></button>
      </div>
      <div className="mh-side-body">
        {side === 'details' && current && <Details model={model} current={current} />}
        {side === 'knowledge' && <Knowledge model={model} />}
        {side === 'issues' && <Issues model={model} />}
        {side === 'stats' && <Stats model={model} />}
      </div>
    </aside>
  );
}
