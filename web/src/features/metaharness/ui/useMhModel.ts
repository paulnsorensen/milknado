// One derived model per render: the snapshot tree, live runs by node, the
// needs-you set, and the view state.
import { useMemo, useSyncExternalStore } from 'react';
import { getState, subscribe } from '../../../app/store';
import type { WireGoalReview } from '../../goal-review/wire';
import { useCell } from '../data/cell';
import { info } from '../data/info';
import type { Need } from '../model/needs';
import { deriveNeeds } from '../model/needs';
import type { Tree } from '../model/tree';
import { buildTree } from '../model/tree';
import type { MhState } from '../viewState';
import { getMh, subscribeMh } from '../viewState';
import type { WireLiveRun, WireMetaharnessInfo, WireMhSnapshot } from '../wire';

export interface MhModel {
  snapshot: WireMhSnapshot | null;
  tree: Tree;
  needs: Map<number, Need>;
  liveByNode: Map<number, WireLiveRun>;
  reviews: WireGoalReview[];
  info: WireMetaharnessInfo | null;
  ui: MhState;
}

export function useMhModel(): MhModel {
  const store = useSyncExternalStore(subscribe, getState);
  const ui = useSyncExternalStore(subscribeMh, getMh);
  const metaharness = useCell(info);
  const snapshot = store.snapshot as WireMhSnapshot | null;
  const derived = useMemo(() => {
    const tree = buildTree(snapshot?.graph);
    const runs = snapshot?.active_runs ?? [];
    const reviews = snapshot?.pending_goal_reviews ?? [];
    return {
      tree,
      reviews,
      liveByNode: new Map(runs.map((run) => [run.node_id, run])),
      needs: deriveNeeds([...tree.byId.values()], runs, reviews),
    };
  }, [snapshot]);
  return { snapshot, ui, info: metaharness, ...derived };
}
