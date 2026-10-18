// The metaharness view: Mikado graph | conversation | details, with the
// decision popover, the edge rail, and the expanded graph on top.
import type { ReactElement } from 'react';
import { useEffect } from 'react';
import { useCell } from '../data/cell';
import { loadInfo } from '../data/info';
import { loadNode, nodeData } from '../data/nodeData';
import { nextNeed } from '../model/needs';
import { isGoal, rootOf } from '../model/tree';
import { getMh, patchMh } from '../viewState';
import { nextDecision } from './actions';
import { ChatPane } from './ChatPane';
import { currentNode, describe } from './current';
import { GraphPane } from './GraphPane';
import { DecisionPopover, ExpandedGraph, Rail } from './Overlays';
import { SidePanel } from './SidePanel';
import type { MhModel } from './useMhModel';
import { useMhModel } from './useMhModel';
import '../fonts.css';
import '../metaharness.css';
import '../metaharness-chat.css';

const INFO_REFRESH_MS = 30000;
const NODE_REFRESH_MS = 400;

function closeTopmost(): void {
  const ui = getMh();
  if (ui.pop) {
    patchMh({ pop: null });
  } else if (ui.expanded) {
    patchMh({ expanded: false });
  } else if (ui.view === 'diff') {
    patchMh({ view: 'thread' });
  } else if (ui.side) {
    patchMh({ side: null });
  }
}

function useShortcuts(model: MhModel): void {
  useEffect(() => {
    const onKey = (event: KeyboardEvent): void => {
      const target = event.target as HTMLElement | null;
      if (event.key === 'Escape') {
        closeTopmost();
        return;
      }
      if (target?.tagName === 'INPUT' || target?.tagName === 'TEXTAREA' || event.metaKey || event.ctrlKey || event.altKey) {
        return;
      }
      if (event.key.toLowerCase() === 'g') {
        patchMh((ui) => ({ expanded: !ui.expanded, pop: null }));
      } else if (event.key.toLowerCase() === 'n') {
        nextDecision(model);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [model]);
}

function useInitialFocus(model: MhModel): void {
  const ready = model.tree.roots.length > 0;
  useEffect(() => {
    const ui = getMh();
    if (!ready || ui.sel !== null || ui.projectId !== null) {
      return;
    }
    const need = nextNeed(model.needs, null);
    const projectId = need ? rootOf(model.tree, need.nodeId) : model.tree.roots[0].id;
    patchMh({ projectId, scopeRoot: projectId, sel: need?.nodeId ?? projectId });
  }, [ready, model]);
}

function useNodeData(model: MhModel, nodeId: number | null, goal: boolean): void {
  const snapshot = model.snapshot;
  useEffect(() => {
    const timer = setTimeout(() => void loadNode(nodeId, goal), nodeData.get().nodeId === nodeId ? NODE_REFRESH_MS : 0);
    return () => clearTimeout(timer);
  }, [nodeId, goal, snapshot]);
}

export function MetaharnessView(): ReactElement {
  const model = useMhModel();
  const data = useCell(nodeData);
  const node = currentNode(model);
  const current = node ? describe(model, node, data) : null;
  useEffect(() => {
    void loadInfo();
    const timer = setInterval(() => void loadInfo(), INFO_REFRESH_MS);
    return () => clearInterval(timer);
  }, []);
  useShortcuts(model);
  useInitialFocus(model);
  useNodeData(model, node?.id ?? null, node ? isGoal(node) : false);
  return (
    <div className="mh" data-view="metaharness">
      <Rail model={model} />
      <GraphPane model={model} />
      <ChatPane model={model} current={current} />
      <SidePanel model={model} current={current} />
      <ExpandedGraph model={model} />
      <DecisionPopover model={model} current={current} />
    </div>
  );
}
