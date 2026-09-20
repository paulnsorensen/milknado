// The Help dialog's open/closed flag: a plain pub-sub so the header control
// (a click) and the `help.open` action (a keypress) share one state.
let open = false;
const listeners = new Set<() => void>();

function emit(): void {
  for (const listener of listeners) {
    listener();
  }
}

export function isHelpOpen(): boolean {
  return open;
}

export function subscribeHelp(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function openHelp(): void {
  open = true;
  emit();
}

export function closeHelp(): void {
  open = false;
  emit();
}

export function resetHelp(): void {
  open = false;
}
