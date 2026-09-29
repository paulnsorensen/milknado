// The `sidecar-section` contributions for the Session and Details tabs:
// each renders its tab body only for a selected node while its tab is active.
import type { ReactElement } from 'react';
import { useSyncExternalStore } from 'react';
import { getState, selectedNodeId, subscribe } from '../../app/store';
import {
  detailHasMore,
  detailTabId,
  detailTabPanelId,
  getActiveTab,
  getDetailState,
  sessionHasMore,
  subscribeDetail,
  subscribeTab,
  type DetailTab,
} from '../../shared/node-detail';
import { DetailsTab } from './DetailsTab';
import { SessionTab } from './SessionTab';

function useTabBody(tab: DetailTab) {
  const store = useSyncExternalStore(subscribe, getState);
  const detailState = useSyncExternalStore(subscribeDetail, getDetailState);
  const activeTab = useSyncExternalStore(subscribeTab, getActiveTab);
  const selected = selectedNodeId(store) !== null;
  const active = selected && activeTab === tab;
  return { active, detailState, selected };
}

export function SessionTabSection(): ReactElement | null {
  const { active, detailState, selected } = useTabBody('session');
  if (!selected) {
    return null;
  }
  const events = detailState.detail?.detail?.sessions.items?.[0]?.event_history.items ?? [];
  return (
    <div
      id={detailTabPanelId('session')}
      role="tabpanel"
      aria-labelledby={detailTabId('session')}
      hidden={!active}
    >
      {active && <SessionTab events={events} hasMore={sessionHasMore(detailState)} />}
    </div>
  );
}

export function DetailsTabSection(): ReactElement | null {
  const { active, detailState, selected } = useTabBody('details');
  if (!selected) {
    return null;
  }
  const detail = detailState.detail?.detail ?? null;
  return (
    <div
      id={detailTabPanelId('details')}
      role="tabpanel"
      aria-labelledby={detailTabId('details')}
      hidden={!active}
    >
      {active && detail && <DetailsTab detail={detail} hasMore={detailHasMore(detailState)} />}
    </div>
  );
}
