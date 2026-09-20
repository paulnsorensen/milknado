// The `toast` slot contribution: every store notice, most recent last.
// `api.ts` pushes a notice with the server's domain reason on a 409.
import type { ReactElement } from 'react';
import { useSyncExternalStore } from 'react';
import { getState, subscribe } from '../../app/store';

export function NoticeToasts(): ReactElement {
  const store = useSyncExternalStore(subscribe, getState);

  return (
    <div className="mk-toasts">
      {store.notices.map((notice) => (
        <p key={notice.id} role="alert">
          {notice.reason}
        </p>
      ))}
    </div>
  );
}
