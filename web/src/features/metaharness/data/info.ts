// `GET /api/metaharness`: project name, worker capacity, run-loop pause
// state, per-flavor quality gates, and the knowledge pages workers read.
import { get } from '../../../app/api';
import type { WireMetaharnessInfo } from '../wire';
import { cell } from './cell';

export const info = cell<WireMetaharnessInfo | null>(null);

export async function loadInfo(): Promise<void> {
  try {
    info.set(await get<WireMetaharnessInfo>('/api/metaharness', undefined, () => false));
  } catch {
    info.set(null);
  }
}

export function setPaused(paused: boolean): void {
  const current = info.get();
  if (current) {
    info.set({ ...current, scheduling: { ...current.scheduling, paused } });
  }
}
