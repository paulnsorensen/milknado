// The `status` slot contribution: visible only while the stream is down.
import type { ReactElement } from 'react';
import { useSyncExternalStore } from 'react';
import { getConnectionStatus, subscribeConnection } from './connection';

export function StatusIndicator(): ReactElement | null {
  const status = useSyncExternalStore(subscribeConnection, getConnectionStatus);
  if (status !== 'reconnecting') {
    return null;
  }
  return <p role="status">Connection lost. The browser tries again.</p>;
}
