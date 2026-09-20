// The one EventSource for the whole app: it feeds every store update from
// `/api/stream` and probes `/api/snapshot` after a stream error so a
// dropped session (401) redirects through the existing api-client hook.
import type { ReactElement } from 'react';
import { useEffect } from 'react';
import { get } from '../../app/api';
import { getState, pushNotice, setSnapshot } from '../../app/store';
import type { WireExecutionSnapshot } from '../../app/wire';
import { setConnectionStatus } from './connection';
import { mergeSnapshot, type RawStreamSnapshot } from './runtimeSnapshot';

const STREAM_URL = '/api/stream';

export function StreamProvider(): ReactElement | null {
  useEffect(() => {
    const source = new EventSource(STREAM_URL);

    const handleSnapshot = (event: MessageEvent<string>): void => {
      let raw: RawStreamSnapshot;
      try {
        raw = JSON.parse(event.data) as RawStreamSnapshot;
      } catch {
        pushNotice('Received a malformed update from the server.');
        return;
      }
      setSnapshot(mergeSnapshot(raw, getState().capabilities));
      setConnectionStatus('connected');
    };

    const handleError = (): void => {
      setConnectionStatus('reconnecting');
      void get<WireExecutionSnapshot>('/api/snapshot').catch(() => {});
    };

    source.addEventListener('snapshot', handleSnapshot);
    source.addEventListener('error', handleError);

    return () => {
      source.removeEventListener('snapshot', handleSnapshot);
      source.removeEventListener('error', handleError);
      source.close();
    };
  }, []);

  return null;
}
