// The single pending confirmation: copy and action details for the prompt.
// Dismiss (or a second request) clears it without running anything.
export interface ConfirmRequest {
  prompt: string;
  body: string;
  dismissLabel: string;
  confirmLabel: string;
  action: () => void;
}

export interface ConfirmOptions {
  prompt: string;
  action: () => void;
  body?: string;
  dismissLabel?: string;
  confirmLabel?: string;
}

const DEFAULT_BODY = 'The action runs once. It cannot be undone from here.';
const DEFAULT_DISMISS_LABEL = 'Dismiss';
const DEFAULT_CONFIRM_LABEL = 'Confirm';

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

export function requestConfirm(options: ConfirmOptions): void {
  pending = {
    prompt: options.prompt,
    body: options.body ?? DEFAULT_BODY,
    dismissLabel: options.dismissLabel ?? DEFAULT_DISMISS_LABEL,
    confirmLabel: options.confirmLabel ?? DEFAULT_CONFIRM_LABEL,
    action: options.action,
  };
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
