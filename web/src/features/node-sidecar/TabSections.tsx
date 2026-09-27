// The `sidecar-section` contributions for the Session and Details tabs:
// each renders its tab body only for a selected node while its tab is active.
import type { ReactElement } from 'react';
import { useSyncExternalStore } from 'react';
import { getState, subscribe } from '../../app/store';
import {
  detailHasMore,
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
  const active = typeof store.selection === 'number' && activeTab === tab;
  return { active, detailState };
}

export function SessionTabSection(): ReactElement | null {
  const { active, detailState } = useTabBody('session');
  if (!active) {
    return null;
  }
  const events = detailState.detail?.detail?.sessions.items?.[0]?.event_history.items ?? [];
  return <SessionTab events={events} hasMore={sessionHasMore(detailState)} />;
}

export function DetailsTabSection(): ReactElement | null {
  const { active, detailState } = useTabBody('details');
  const detail = detailState.detail?.detail ?? null;
  if (!active || !detail) {
    return null;
  }
  return <DetailsTab detail={detail} hasMore={detailHasMore(detailState)} />;
}
