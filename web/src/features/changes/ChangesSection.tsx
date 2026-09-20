import type { ReactElement } from 'react';
import { useEffect, useSyncExternalStore } from 'react';
import { Milknado } from '../../design-system';
import { getActiveTab, subscribeTab } from '../node-sidecar/detailTab';
import { getDetailState, subscribeDetail } from '../node-sidecar/nodeDetail';
import { getChangesState, selectPath, setRunId, subscribeChanges } from './changesState';

/** The `sidecar-section` contribution for the Changes tab: a file list and its diff. */
export function ChangesSection(): ReactElement | null {
  const activeTab = useSyncExternalStore(subscribeTab, getActiveTab);
  const detailState = useSyncExternalStore(subscribeDetail, getDetailState);
  const changesState = useSyncExternalStore(subscribeChanges, getChangesState);
  const { Button } = Milknado;

  const runId = detailState.detail?.detail?.runs.items?.[0]?.run_id ?? null;

  useEffect(() => {
    setRunId(runId);
  }, [runId]);

  if (activeTab !== 'changes') {
    return null;
  }

  return (
    <div className="mk-changes-section">
      {changesState.files.length === 0 && <p>No changed files yet.</p>}
      <ul>
        {changesState.files.map((file) => (
          <li key={file.path}>
            <Button
              variant={changesState.selectedPath === file.path ? 'primary' : 'secondary'}
              onClick={() => selectPath(file.path)}
            >
              {file.path}
            </Button>
          </li>
        ))}
      </ul>
      {changesState.selectedPath && <pre>{changesState.diffText}</pre>}
    </div>
  );
}
