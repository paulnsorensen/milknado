// Floating surfaces over the three panes: the decision popover beside a
// node that needs you, the edge-hover rail, and the expanded graph.
import type { ReactElement } from 'react';
import { pushNotice } from '../../../app/store';
import { useCell } from '../data/cell';
import { setLoopPaused } from '../data/commands';
import { nodeData } from '../data/nodeData';
import { diffTotals, runDiff } from '../data/runDiff';
import { deriveIssues } from '../model/insights';
import { layoutTree } from '../model/layout';
import { NEED_KICKER } from '../model/needs';
import { shortRunId } from '../model/thread';
import { title } from '../model/tree';
import { flash, patchMh } from '../viewState';
import { setViewMode } from '../viewMode';
import type { Current } from './current';
import { needSummary } from './current';
import { GraphCanvas } from './GraphCanvas';
import type { IconName } from './Icon';
import { Icon } from './Icon';
import { NeedButtons } from './NeedButtons';
import type { MhModel } from './useMhModel';

function PopoverStats({ model, current }: { model: MhModel; current: Current }): ReactElement {
  const diff = useCell(runDiff);
  const stats: Array<[string, string]> = [];
  if (diff.runId === current.runId && diff.files.length > 0) {
    const totals = diffTotals(diff.files);
    stats.push([`${diff.files.length} files`, 'muted'], [`+${totals.added}`, 'add'], [`−${totals.removed}`, 'del']);
  }
  if (current.need?.kind === 'failed') {
    stats.push(['gate FAIL', 'del']);
  } else {
    stats.push([`workers ${model.snapshot?.active_runs?.length ?? 0}/${model.info?.concurrency ?? '—'}`, 'muted']);
  }
  if (current.runId) {
    stats.push([`run ${shortRunId(current.runId)}`, 'run']);
  }
  return <div className="mh-pop-stats">{stats.map(([text, tone]) => <span key={text} className={`is-${tone}`}>{text}</span>)}</div>;
}

export function DecisionPopover({ model, current }: { model: MhModel; current: Current | null }): ReactElement | null {
  const data = useCell(nodeData);
  const diff = useCell(runDiff);
  const pop = model.ui.pop;
  const need = pop ? model.needs.get(pop.id) : undefined;
  if (!pop || !need || current?.node.id !== pop.id) {
    return null;
  }
  const hasDiff = diff.runId === current.runId && diff.files.length > 0;
  return (
    <>
      <div className="mh-scrim" onClick={() => patchMh({ pop: null })} />
      <div className="mh-pop" role="dialog" aria-label={NEED_KICKER[need.kind]} style={{ left: pop.x, top: pop.y }}>
        <div className="mh-pop-head"><span className="mh-side-kicker">{NEED_KICKER[need.kind]}</span>
          <span className="mh-mono mh-faint">#{need.nodeId}</span></div>
        <span className="mh-pop-title">{title(current.node)}</span>
        <span className="mh-pop-summary">{needSummary(need, data)}</span>
        <PopoverStats model={model} current={current} />
        <NeedButtons need={need} model={model} />
        {hasDiff && (
          <button type="button" className="mh-text-btn" onClick={() => patchMh({ view: 'diff', pop: null })}>
            <Icon name="diff" size={14} />Open full diff<Icon name="arrow" size={11} /></button>
        )}
      </div>
    </>
  );
}

interface RailItem {
  icon: IconName;
  label: string;
  count: string;
  go: () => void;
}

function toggleLoop(model: MhModel): void {
  const scheduling = model.info?.scheduling;
  if (!scheduling?.available) {
    pushNotice(scheduling?.reason ?? 'Pausing the run loop is unavailable.');
    return;
  }
  void setLoopPaused(!scheduling.paused).then((ok) => ok && flash(scheduling.paused ? 'Run loop resumed' : 'Run loop paused'));
}

export function Rail({ model }: { model: MhModel }): ReactElement {
  const paused = model.info?.scheduling.paused ?? false;
  const issues = deriveIssues(model.snapshot, model.tree).length;
  const close = { railOpen: false };
  const items: RailItem[] = [
    { icon: 'projects', label: 'All projects', count: '', go: () => patchMh({ ...close, projectId: null }) },
    { icon: 'book', label: 'Knowledge', count: String(model.info?.knowledge.length ?? 0), go: () => patchMh({ ...close, side: 'knowledge' }) },
    { icon: 'warning', label: 'Issues', count: issues ? String(issues) : '', go: () => patchMh({ ...close, side: 'issues' }) },
    { icon: 'chart', label: 'Stats', count: '', go: () => patchMh({ ...close, side: 'stats' }) },
    { icon: paused ? 'play' : 'pause', label: paused ? 'Resume run loop' : 'Pause run loop', count: '', go: () => { patchMh(close); toggleLoop(model); } },
    { icon: 'classic', label: 'Classic view', count: '', go: () => setViewMode('classic') },
  ];
  return (
    <>
      <div className="mh-rail-edge" onMouseEnter={() => patchMh({ railOpen: true })} />
      {model.ui.railOpen && (
        <nav className="mh-rail" aria-label="Metaharness menu" onMouseLeave={() => patchMh(close)}>
          <div className="mh-brand is-rail"><span className="mh-logo" />{model.info?.project.name ?? 'milknado'}</div>
          {items.map((item) => (
            <button key={item.label} type="button" className="mh-rail-item" onClick={item.go}>
              <Icon name={item.icon} size={17} /><span>{item.label}</span><span className="mh-rail-count">{item.count}</span>
            </button>
          ))}
        </nav>
      )}
    </>
  );
}

export function ExpandedGraph({ model }: { model: MhModel }): ReactElement | null {
  const rootId = model.ui.projectId ?? model.tree.roots[0]?.id;
  const root = rootId === undefined ? undefined : model.tree.byId.get(rootId);
  if (!model.ui.expanded || !root) {
    return null;
  }
  const layout = layoutTree(model.tree, root.id, {
    width: 1040, cardWidth: 156, rootWidth: 200, height: 92, gap: 14, pitch: 124, fold: false, unfold: {},
  });
  return (
    <div className="mh-expanded" role="dialog" aria-label="Expanded graph">
      <div className="mh-expanded-head">
        <span className="mh-expanded-title">{title(root)}</span><span className="mh-faint">full graph · done nodes unfolded</span>
        <span className="mh-keys"><kbd>G</kbd>or<kbd>Esc</kbd>to close</span>
        <button type="button" className="mh-round-btn" aria-label="Close expanded graph" onClick={() => patchMh({ expanded: false })}>
          <Icon name="collapse" size={15} /></button>
      </div>
      <div className="mh-expanded-body"><GraphCanvas layout={layout} model={model} mode="expanded" fold={false} /></div>
    </div>
  );
}
