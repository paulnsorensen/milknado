// The one modal card every feature dialog renders into: a scrim over the
// shell, a serif title, the body, and right-aligned actions.
import type { ReactElement, ReactNode } from 'react';

export interface DialogProps {
  /** The visible heading. */
  title: string;
  /** The accessible name; defaults to the title. */
  label?: string;
  role?: 'dialog' | 'alertdialog';
  wide?: boolean;
  children?: ReactNode;
  actions: ReactNode;
}

export function Dialog({
  title,
  label = title,
  role = 'dialog',
  wide = false,
  children,
  actions,
}: DialogProps): ReactElement {
  return (
    <div className="mk-scrim">
      <div role={role} aria-modal="true" aria-label={label} className={wide ? 'mk-dialog is-wide' : 'mk-dialog'}>
        <h2 className="mk-dialog-title">{title}</h2>
        {children}
        <div className="mk-dialog-actions">{actions}</div>
      </div>
    </div>
  );
}
