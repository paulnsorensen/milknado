// A run's worktree diff: `GET /api/runs/{id}/changes` for the file list,
// then `GET /api/runs/{id}/diff?path=` for each file's unified diff.
import { get } from '../../../app/api';
import type { WireChangedFile } from '../../changes/changesWire';
import { fetchDiffText } from '../../changes/diffText';
import { cell } from './cell';

export type DiffMark = '+' | '-' | '@' | ' ';

export interface DiffLine {
  mark: DiffMark;
  text: string;
}

export interface DiffFile {
  path: string;
  added: number;
  removed: number;
  lines: DiffLine[];
}

export interface RunDiff {
  runId: string | null;
  files: DiffFile[];
  loading: boolean;
}

const MAX_FILES = 12;
const MAX_LINES = 400;

export const runDiff = cell<RunDiff>({ runId: null, files: [], loading: false });

export function parseUnifiedDiff(text: string): DiffLine[] {
  const lines: DiffLine[] = [];
  for (const raw of text.split('\n')) {
    if (/^(diff --git|index |--- |\+\+\+ |new file mode|deleted file mode)/.test(raw)) {
      continue;
    }
    if (raw.startsWith('@@')) {
      lines.push({ mark: '@', text: raw });
    } else if (raw.startsWith('+') || raw.startsWith('-')) {
      lines.push({ mark: raw[0] as DiffMark, text: raw.slice(1) });
    } else if (raw.startsWith(' ')) {
      lines.push({ mark: ' ', text: raw.slice(1) });
    }
  }
  return lines.slice(0, MAX_LINES);
}

export async function loadRunDiff(runId: string | null): Promise<void> {
  if (runId === runDiff.get().runId && (runDiff.get().loading || runDiff.get().files.length > 0)) {
    return;
  }
  runDiff.set({ runId, files: [], loading: runId !== null });
  if (runId === null) {
    return;
  }
  const changed = await get<WireChangedFile[]>(`/api/runs/${runId}/changes`, undefined, () => false)
    .catch(() => null) ?? [];
  const files = await Promise.all(changed.slice(0, MAX_FILES).map(async (file) => ({
    path: file.path,
    added: file.added,
    removed: file.removed,
    lines: parseUnifiedDiff(await fetchDiffText(runId, file.path).catch(() => '')),
  })));
  if (runDiff.get().runId === runId) {
    runDiff.set({ runId, files, loading: false });
  }
}

export function diffTotals(files: DiffFile[]): { added: number; removed: number } {
  return files.reduce((sum, file) => ({ added: sum.added + file.added, removed: sum.removed + file.removed }), { added: 0, removed: 0 });
}
