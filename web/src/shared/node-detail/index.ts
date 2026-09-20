// Public surface for the shared node-detail state: the active sidecar tab
// and the paged GET /api/nodes/{id} data. Owned here, not by any one
// feature, because `node-sidecar`, `changes`, `session-input` and
// `shortcuts` all read or drive it.
export type { DetailTab } from './detailTab';
export { getActiveTab, setActiveTab, subscribeTab, resetTab } from './detailTab';

export type { DetailState } from './nodeDetail';
export {
  getDetailState,
  subscribeDetail,
  selectNode,
  pageNext,
  pagePrevious,
  sessionPageNext,
  sessionPagePrevious,
  followNewest,
  resetDetail,
  detailHasMore,
  sessionHasMore,
} from './nodeDetail';

export type {
  DetailPageState,
  WireSnapshotPage,
  WireDetailNode,
  WireRunRecord,
  WireSessionEventKind,
  WireSessionEvent,
  WireNodeSessionSnapshot,
  WireNodeDetailSnapshot,
  WireNodeDetailResponse,
} from './detailWire';
