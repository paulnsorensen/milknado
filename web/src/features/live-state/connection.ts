// A tiny pub-sub for the stream connection status, parallel to app/store.ts
// but scoped to this feature: the shell's store has no field for it.
export type ConnectionStatus = 'connected' | 'reconnecting';

type Listener = () => void;

let status: ConnectionStatus = 'connected';
const listeners = new Set<Listener>();

export function getConnectionStatus(): ConnectionStatus {
  return status;
}

export function setConnectionStatus(next: ConnectionStatus): void {
  if (next === status) {
    return;
  }
  status = next;
  for (const listener of listeners) {
    listener();
  }
}

export function subscribeConnection(listener: Listener): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function resetConnectionStatus(): void {
  status = 'connected';
}
