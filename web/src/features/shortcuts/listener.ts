// Matches a keydown event's `event.key` against KEY_BINDINGS. Does nothing
// while a text input or textarea has focus, so typing is never intercepted.
import { KEY_BINDINGS } from './keyMap';

function isTextEntry(target: EventTarget | null): boolean {
  return target instanceof HTMLElement && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA');
}

export function handleShortcutKey(event: KeyboardEvent): void {
  if (isTextEntry(event.target)) {
    return;
  }
  const shortcut = KEY_BINDINGS.find((binding) => binding.key === event.key);
  if (shortcut === undefined) {
    return;
  }
  event.preventDefault();
  shortcut.run();
}
