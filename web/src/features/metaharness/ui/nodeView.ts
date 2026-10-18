// Card text and state classes for a graph node, shared by the canvas and
// the details panel.
import type { WireNode } from '../../../app/wire';
import { NEED_LABEL } from '../model/needs';
import { childrenOf, isGoal, isReady, openPrerequisites } from '../model/tree';
import type { WireLiveRun } from '../wire';
import type { MhModel } from './useMhModel';

/** The run's reported progress percent, or null when the worker reports none. */
export function runPercent(run: WireLiveRun | undefined): number | null {
  const pct = run?.progress_pct;
  if (pct === null || pct === undefined) {
    return null;
  }
  return Math.round(Math.min(100, Math.max(0, pct <= 1 ? pct * 100 : pct)));
}

function runProgress(run: WireLiveRun | undefined): string {
  const pct = runPercent(run);
  if (pct !== null) {
    return `${pct}%`;
  }
  const minutes = Math.round((run?.elapsed_seconds ?? 0) / 60);
  return minutes < 1 ? 'starting' : `${minutes}m`;
}

export function stateClass(node: WireNode, model: MhModel): string {
  return model.needs.has(node.id) ? 'is-need' : `is-${node.status}`;
}

export function nodeMeta(node: WireNode, model: MhModel): string {
  const need = model.needs.get(node.id);
  if (need) {
    return `needs you · ${NEED_LABEL[need.kind]}`;
  }
  const kids = childrenOf(model.tree, node.id);
  if (isGoal(node)) {
    return `${kids.filter((kid) => kid.status === 'done').length}/${kids.length} done`;
  }
  const open = openPrerequisites(model.tree, node.id);
  switch (node.status) {
    case 'running':
      return `w${node.id} · ${runProgress(model.liveByNode.get(node.id))}`;
    case 'blocked':
      return `blocked · ${open} prereqs`;
    case 'pending':
      return isReady(model.tree, node) ? 'ready' : `waiting · ${open} prereqs`;
    default:
      return node.status;
  }
}
