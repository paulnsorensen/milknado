// Matches a keydown event's `event.key` against KEY_BINDINGS. Does nothing
// while a text input, textarea, select, or contenteditable element has
// focus, or while a dialog is open, so typing and dialog interaction are
// never intercepted.
import { KEY_BINDINGS } from './keyMap';

const DIALOG_SELECTOR = '[role="dialog"], [role="alertdialog"]';

function isTextEntry(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) {
    return false;
  }
  if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.tagName === 'SELECT') {
    return true;
  }
  if (target.isContentEditable) {
    return true;
  }
  return target.closest(DIALOG_SELECTOR) !== null;
}

function isDialogOpen(): boolean {
  return document.querySelector(DIALOG_SELECTOR) !== null;
}

export function handleShortcutKey(event: KeyboardEvent): void {
  if (isDialogOpen() || isTextEntry(event.target)) {
    return;
  }
  const shortcut = KEY_BINDINGS.find((binding) => binding.key === event.key);
  if (shortcut === undefined) {
    return;
  }
  event.preventDefault();
  shortcut.run();
}
