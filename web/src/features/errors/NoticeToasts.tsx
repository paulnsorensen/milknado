// The `toast` slot contribution: every store notice, most recent last.
// `api.ts` pushes a notice with the server's domain reason on a 409.
import type { ReactElement } from 'react';
import { useSyncExternalStore } from 'react';
import { getState, removeNotice, subscribe } from '../../app/store';
import { Milknado } from '../../design-system';

export function NoticeToasts(): ReactElement {
  const store = useSyncExternalStore(subscribe, getState);
  const { Button, StatusGlyph } = Milknado;

  return (
    <>
      {store.notices.map((notice) => (
        <p key={notice.id} role="alert" className="mk-toast">
          <StatusGlyph state="at-risk" />
          <span>{notice.reason}</span>
          <Button variant="ghost" className="mk-btn-sm" ariaLabel="Dismiss" onClick={() => removeNotice(notice.id)}>
            Dismiss
          </Button>
        </p>
      ))}
    </>
  );
}