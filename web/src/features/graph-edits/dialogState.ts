// The single pending graph-edit dialog: which kind is open and, for
// edit/move/archive, which node it targets. Only one dialog is open at once.
export type DialogKind = 'add' | 'edit' | 'move' | 'archive' | null;

export interface DialogState {
  kind: DialogKind;
  nodeId: number | null;
}

let pending: DialogState = { kind: null, nodeId: null };
const listeners = new Set<() => void>();

function emit(): void {
  for (const listener of listeners) {
    listener();
  }
}

export function getDialogState(): DialogState {
  return pending;
}

export function subscribeDialog(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function openDialog(kind: Exclude<DialogKind, null>, nodeId: number | null = null): void {
  pending = { kind, nodeId };
  emit();
}

export function closeDialog(): void {
  pending = { kind: null, nodeId: null };
  emit();
}

export function resetDialog(): void {
  pending = { kind: null, nodeId: null };
}
