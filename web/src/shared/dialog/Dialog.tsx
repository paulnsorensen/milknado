// The one modal card every feature dialog renders into: a scrim over the
// shell, a serif title, the body, and right-aligned actions.
import type { ReactElement, ReactNode } from 'react';
import { useEffect, useRef } from 'react';

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
    function handleKeyDown(event: KeyboardEvent): void {
      if (event.key === 'Escape') {
        onClose?.();
      }
    }
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  return (
    <div className="mk-scrim">
      <div
        ref={dialogRef}
        role={role}
        aria-modal="true"
        aria-label={label}
        tabIndex={-1}
        className={wide ? 'mk-dialog is-wide' : 'mk-dialog'}
      >
        <h2 className="mk-dialog-title">{title}</h2>
        {children}
        <div className="mk-dialog-actions">{actions}</div>
      </div>
    </div>
  );
}
