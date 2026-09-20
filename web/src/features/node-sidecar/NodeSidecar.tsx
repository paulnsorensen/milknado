import type { ReactElement } from 'react';
import { useEffect, useSyncExternalStore } from 'react';
import { getState, subscribe } from '../../app/store';
import { Milknado } from '../../design-system';
import { toGraphNodes } from '../../app/wire';
import { toBadgeState } from './badgeState';
import { getActiveTab, subscribeTab } from './detailTab';
import { DetailsTab } from './DetailsTab';
import { detailHasMore, getDetailState, selectNode, sessionHasMore, subscribeDetail } from './nodeDetail';
import { SessionTab } from './SessionTab';

/** The `sidecar` slot contribution: the node detail panel for the selected node. */
export function NodeSidecar(): ReactElement | null {
  const store = useSyncExternalStore(subscribe, getState);
  const detailState = useSyncExternalStore(subscribeDetail, getDetailState);
  const activeTab = useSyncExternalStore(subscribeTab, getActiveTab);
  const { AncestorPath, StatusBadge } = Milknado;

  const nodeId = typeof store.selection === 'number' ? store.selection : null;

  useEffect(() => {
    selectNode(nodeId);
  }, [nodeId]);

  if (nodeId === null) {
    return null;
  }

  const detail = detailState.detail?.detail ?? null;
  const nodes = store.snapshot?.graph ? toGraphNodes(store.snapshot.graph) : [];
  const title = detail?.description ?? '';
  const runs = detail?.runs.items ?? [];
  const events = detail?.sessions.items?.[0]?.event_history.items ?? [];

  return (
    <div className="mk-node-sidecar">
      <AncestorPath nodes={nodes} id={nodeId} />
      <h2>{title}</h2>
      {detail && <StatusBadge state={toBadgeState(detail.node.status)} />}
      <ul>
        {runs.length === 0 && <li>This node has no runs yet.</li>}
        {runs.map((run) => (
          <li key={run.run_id}>
            {run.run_id} <StatusBadge state={toBadgeState(run.status)} />
          </li>
        ))}
      </ul>
      {activeTab === 'session' && (
        <SessionTab events={events} hasMore={sessionHasMore(detailState)} />
      )}
      {activeTab === 'details' && detail && (
        <DetailsTab detail={detail} hasMore={detailHasMore(detailState)} />
      )}
    </div>
  );
}
