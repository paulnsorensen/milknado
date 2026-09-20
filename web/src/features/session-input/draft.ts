// The guidance draft text, the input element it types into (for the
// `session.focus-input` action), and `queueGuidance` (the
// `session.queue-guidance` action): send the draft as a follow-up.
import { sendSessionCommand } from './sessionCommand';

let draft = '';
let inputEl: HTMLTextAreaElement | HTMLInputElement | null = null;
const listeners = new Set<() => void>();

function emit(): void {
  for (const listener of listeners) {
    listener();
  }
}

export function getDraft(): string {
  return draft;
}

export function setDraft(text: string): void {
  draft = text;
  emit();
}

export function subscribeDraft(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function registerInputEl(el: HTMLTextAreaElement | HTMLInputElement | null): void {
  inputEl = el;
}

export function focusSessionInput(): void {
  inputEl?.focus();
}

export function queueGuidance(): void {
  const text = draft;
  if (text === '') {
    return;
  }
  setDraft('');
  void sendSessionCommand('follow_up', { text });
}

export function resetDraft(): void {
  draft = '';
  inputEl = null;
  emit();
}
