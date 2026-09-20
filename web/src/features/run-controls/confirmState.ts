// The single pending confirmation: a prompt plus the action Confirm runs.
// Dismiss (or a second request) clears it without running anything.
export interface ConfirmRequest {
  prompt: string;
  action: () => void;
}

let pending: ConfirmRequest | null = null;
const listeners = new Set<() => void>();

function emit(): void {
  for (const listener of listeners) {
    listener();
  }
}

export function getPendingConfirm(): ConfirmRequest | null {
  return pending;
}

export function subscribeConfirm(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function requestConfirm(prompt: string, action: () => void): void {
  pending = { prompt, action };
  emit();
}

export function dismissConfirm(): void {
  pending = null;
  emit();
}

export function confirmPending(): void {
  const request = pending;
  pending = null;
  emit();
  request?.action();
}

export function resetConfirm(): void {
  pending = null;
}
