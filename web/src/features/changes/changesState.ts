// Tracks the changed-file list and the selected file's diff for the run
// associated with the currently selected node.
import { get } from '../../app/api';
import { fetchDiffText } from './diffText';
import type { WireChangedFile } from './changesWire';

export interface ChangesState {
  runId: string | null;
  files: WireChangedFile[];
  selectedPath: string | null;
  diffText: string;
}

let state: ChangesState = { runId: null, files: [], selectedPath: null, diffText: '' };
const listeners = new Set<() => void>();

function emit(): void {
  for (const listener of listeners) {
    listener();
  }
}

export function getChangesState(): ChangesState {
  return state;
}

export function subscribeChanges(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

async function fetchFiles(runId: string): Promise<void> {
  const files = await get<WireChangedFile[]>(`/api/runs/${runId}/changes`);
  if (files !== null && state.runId === runId) {
    state = { ...state, files };
    emit();
  }
}

export function setRunId(runId: string | null): void {
  if (runId === state.runId) {
    return;
  }
  state = { runId, files: [], selectedPath: null, diffText: '' };
  emit();
  if (runId !== null) {
    void fetchFiles(runId);
  }
}

export function selectPath(path: string): void {
  if (state.runId === null) {
    return;
  }
  const runId = state.runId;
  state = { ...state, selectedPath: path, diffText: '' };
  emit();
  void fetchDiffText(runId, path).then((text) => {
    if (state.runId === runId && state.selectedPath === path) {
      state = { ...state, diffText: text };
      emit();
    }
  });
}

export function resetChanges(): void {
  state = { runId: null, files: [], selectedPath: null, diffText: '' };
  emit();
}
