// The left pane: project picker or the focused Mikado subtree, with
// breadcrumbs above and the run-loop toggle below.
import type { ReactElement } from 'react';
import { pushNotice } from '../../../app/store';
import { setLoopPaused } from '../data/commands';
import { projectCards } from '../model/insights';
import { layoutTree } from '../model/layout';
import { ancestry, title } from '../model/tree';
import type { DropTarget } from '../viewState';
import { flash, patchMh } from '../viewState';
import { dropProps, openAdd, openGoalForm } from './edits';
import { GraphCanvas } from './GraphCanvas';
import { Icon } from './Icon';
import type { MhModel } from './useMhModel';

export const GRAPH_WIDTH = 340;

function crumbClass(model: MhModel, target: DropTarget, current: boolean): string {
  return [current ? 'is-current' : '', model.ui.drag?.over === target ? 'is-drop-target' : ''].filter(Boolean).join(' ');
}

function Crumbs({ model }: { model: MhModel }): ReactElement {
  const { projectId, scopeRoot } = model.ui;
  const chain = projectId === null ? [] : ancestry(model.tree, scopeRoot ?? projectId);
  return (
    <nav className="mh-crumbs" aria-label="Graph scope">
      <button type="button" className={crumbClass(model, 'root', !chain.length)} {...dropProps(model, 'root')}
        title={model.editable ? 'Drop a node here to make it a top-level goal' : undefined}
        onClick={() => patchMh({ projectId: null, pop: null })}>Projects</button>
      {chain.map((node, i) => (
        <span key={node.id} className="mh-crumb">
          <Icon name="caret" size={10} />
          <button type="button" className={crumbClass(model, node.id, i === chain.length - 1)} {...dropProps(model, node.id)}
            onClick={() => patchMh({ scopeRoot: node.id, pop: null })}>{title(node)}</button>
        </span>
      ))}
    </nav>
  );
}

function Projects({ model }: { model: MhModel }): ReactElement {
  const name = model.info?.project.name ?? '';
  return (
    <div className="mh-projects">
      {projectCards(model.tree, model.needs).map((card) => (
        <button key={card.id} type="button" className={`mh-project${card.needCount ? ' is-need' : ''}`}
          onClick={() => patchMh({ projectId: card.id, scopeRoot: card.id, sel: card.id, view: 'thread' })}>
          <span className="mh-project-head"><span className="mh-project-title">{card.title}</span>
            <span className="mh-mono mh-faint">{name} #{card.id}</span></span>
          <span className="mh-project-meta">
            {card.needCount ? `${card.needCount} need you · ${card.running} working`
              : card.running ? `calm · ${card.running} working` : 'idle'}
          </span>
          <span className="mh-bar"><span style={{ width: `${card.pct}%` }} /></span>
        </button>
      ))}
      <button type="button" className="mh-new-goal" onClick={openGoalForm}>
        <Icon name="plus" size={14} />New goal
      </button>
      {model.tree.roots.length === 0 && <p className="mh-faint">No goals yet. Start one here or with milknado plan.</p>}
    </div>
  );
}

function LoopToggle({ model }: { model: MhModel }): ReactElement {
  const scheduling = model.info?.scheduling;
  const paused = scheduling?.paused ?? false;
  const working = (model.snapshot?.active_runs ?? []).filter((run) =>
    model.ui.projectId === null || ancestry(model.tree, run.node_id)[0]?.id === model.ui.projectId).length;
  const toggle = (): void => {
    if (!scheduling?.available) {
      pushNotice(scheduling?.reason ?? 'Pausing the run loop is unavailable.');
      return;
    }
    void setLoopPaused(!paused).then((ok) => ok && flash(paused ? 'Run loop resumed' : 'Run loop paused'));
  };
  return (
    <button type="button" className={`mh-loop${paused ? ' is-paused' : working ? ' is-live' : ''}`}
      title="Pause or resume the run loop" onClick={toggle}>
      <span className="mh-dot" />{paused ? 'paused' : `${working} working`}
    </button>
  );
}

export function GraphPane({ model }: { model: MhModel }): ReactElement {
  const { projectId, scopeRoot, unfold } = model.ui;
  const root = projectId === null ? null : (scopeRoot ?? projectId);
  const layout = root === null ? null : layoutTree(model.tree, root, {
    width: GRAPH_WIDTH - 24, cardWidth: 100, rootWidth: 140, height: 56, gap: 8, pitch: 84, fold: true, unfold,
  });
  return (
    <section className="mh-graph" style={{ width: GRAPH_WIDTH }} aria-label="Mikado graph">
      <div className="mh-graph-top">
        <div className="mh-brand">
          <span className="mh-logo" />
          <span>{model.info?.project.name ?? 'milknado'}</span>
          <span className="mh-brand-actions">
            <button type="button" className="mh-icon-btn" title={root === null ? 'New goal' : 'Add a node (A)'}
              aria-label={root === null ? 'New goal' : 'Add a node'}
              onClick={openAdd}>
              <Icon name="plus" size={14} />
            </button>
            <button type="button" className="mh-icon-btn" title="Expand graph (G)" aria-label="Expand graph"
              onClick={() => patchMh((ui) => ({ expanded: !ui.expanded, pop: null }))}>
              <Icon name="expand" size={14} />
            </button>
          </span>
        </div>
        <Crumbs model={model} />
      </div>
      <div className="mh-graph-body">
        {layout ? <GraphCanvas layout={layout} model={model} mode="compact" fold /> : <Projects model={model} />}
      </div>
      <div className="mh-graph-foot">
        <LoopToggle model={model} />
        <span className="mh-hint">{model.editable ? 'drag a node onto another to move it' : 'dbl-click goal to focus'}</span>
      </div>
    </section>
  );
}
