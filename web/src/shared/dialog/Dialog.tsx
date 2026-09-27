// The one modal card every feature dialog renders into: a scrim over the
// shell, a serif title, the body, and right-aligned actions.
import type { KeyboardEvent, ReactElement, ReactNode } from 'react';
import { useEffect, useRef } from 'react';

const FOCUSABLE_SELECTOR =
  'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

export interface DialogProps {
  /** The visible heading. */
  title: string;
  /** The accessible name; defaults to the title. */
  label?: string;
  role?: 'dialog' | 'alertdialog';
  wide?: boolean;
  /** Runs once on an Escape keypress; omit to leave Escape unhandled. */
  onClose?: () => void;
  children?: ReactNode;
  actions: ReactNode;
}

export function Dialog({
  title,
  label = title,
  role = 'dialog',
  wide = false,
  onClose,
  children,
  actions,
}: DialogProps): ReactElement {
  const dialogRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const previouslyFocused = document.activeElement as HTMLElement | null;
    dialogRef.current?.focus();
    return () => {
      previouslyFocused?.focus();
    };
  }, []);

  useEffect(() => {
    if (!onClose) {
      return;
    }
    function handleEscape(event: globalThis.KeyboardEvent): void {
      if (event.key === 'Escape') {
        onClose?.();
      }
    }
    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, [onClose]);

  function trapTab(event: KeyboardEvent<HTMLDivElement>): void {
    if (event.key !== 'Tab') {
      return;
    }
    const dialog = dialogRef.current;
    if (!dialog) {
      return;
    }
    const focusable = Array.from(dialog.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR));
    if (focusable.length === 0) {
      event.preventDefault();
      dialog.focus();
      return;
    }
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    const active = document.activeElement;
    const onCard = active === dialog || !focusable.includes(active as HTMLElement);
    if (event.shiftKey && (onCard || active === first)) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && (onCard || active === last)) {
      event.preventDefault();
      first.focus();
    }
  }

  return (
    <div className="mk-scrim">
      <div
        ref={dialogRef}
        role={role}
        aria-modal="true"
        aria-label={label}
        tabIndex={-1}
        className={wide ? 'mk-dialog is-wide' : 'mk-dialog'}
        onKeyDown={trapTab}
      >
        <h2 className="mk-dialog-title">{title}</h2>
        {children}
        <div className="mk-dialog-actions">{actions}</div>
      </div>
    </div>
  );
}
