// The selected node's backend records: `GET /api/nodes/{id}` for runs,
// reviews and session history, plus the coordinator session records when
// a coordinator owns the selected goal.
import { get } from '../../../app/api';
import type { CoordinatorRecord } from '../model/thread';
import type { WireMhDetail, WireMhDetailResponse } from '../wire';
import { cell } from './cell';

export interface NodeData {
  nodeId: number | null;
  detail: WireMhDetail | null;
  records: CoordinatorRecord[];
}

interface SessionSummary {
  id: string;
  goal_id: number;
}

export const nodeData = cell<NodeData>({ nodeId: null, detail: null, records: [] });

let generation = 0;
const silent = (): boolean => false;

async function coordinatorRecords(goalId: number): Promise<CoordinatorRecord[]> {
  const sessions = await get<SessionSummary[]>('/api/coordinators', undefined, silent).catch(() => null);
  const session = sessions?.find((candidate) => candidate.goal_id === goalId);
  if (!session) {
    return [];
  }
  const snapshot = await get<{ events: CoordinatorRecord[] }>(
    `/api/coordinators/${encodeURIComponent(session.id)}/snapshot`, undefined, silent,
  ).catch(() => null);
  return snapshot?.events ?? [];
}

export async function loadNode(nodeId: number | null, isGoal: boolean): Promise<void> {
  const current = ++generation;
  if (nodeData.get().nodeId !== nodeId) {
    nodeData.set({ nodeId, detail: null, records: [] });
  }
  if (nodeId === null) {
    return;
  }
  const params = new URLSearchParams({ request_generation: String(current), limit: '100' });
  const [response, records] = await Promise.all([
    get<WireMhDetailResponse>(`/api/nodes/${nodeId}?${params.toString()}`, undefined, silent).catch(() => null),
    isGoal ? coordinatorRecords(nodeId) : Promise.resolve([]),
  ]);
  if (current === generation) {
    nodeData.set({ nodeId, detail: response?.detail ?? null, records });
  }
}
