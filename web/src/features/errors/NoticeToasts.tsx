// The `toast` slot contribution: every store notice, most recent last.
// `api.ts` pushes a notice with the server's domain reason on a 409.
import type { ReactElement } from 'react';
import { useSyncExternalStore } from 'react';
import { getState, removeNotice, subscribe } from '../../app/store';
import { Milknado } from '../../design-system';

export function NoticeToasts(): ReactElement {
  const store = useSyncExternalStore(subscribe, getState);
  const { Button } = Milknado;

  return (
    <div className="mk-toasts">
      {store.notices.map((notice) => (
        <p key={notice.id} role="alert">
          <span>{notice.reason}</span>
          <Button icon ariaLabel="Close notice" onClick={() => removeNotice(notice.id)}>
            ×
          </Button>
        </p>
      ))}
    </div>
  );
}
