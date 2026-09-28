import type { ReactElement } from 'react';
import { useEffect, useSyncExternalStore } from 'react';
import { getState, subscribe } from '../../app/store';
import {
  detailTabId,
  detailTabPanelId,
  getActiveTab,
  getDetailState,
  subscribeDetail,
  subscribeTab,
} from '../../shared/node-detail';
import { getChangesState, selectPath, setRunId, subscribeChanges } from './changesState';

type DiffLineKind = 'add' | 'del' | 'hunk' | 'ctx';

function diffLineKind(line: string): DiffLineKind {
  if (line.startsWith('@@')) {
    return 'hunk';
  }
  if (line.startsWith('+++') || line.startsWith('---')) {
    return 'hunk';
  }
  if (line.startsWith('+')) {
    return 'add';
  }
  if (line.startsWith('-')) {
    return 'del';
  }
  return 'ctx';
}

/** The `sidecar-section` contribution for the Changes tab: a file table and the selected diff. */
export function ChangesSection(): ReactElement | null {
  const activeTab = useSyncExternalStore(subscribeTab, getActiveTab);
  const store = useSyncExternalStore(subscribe, getState);
  const detailState = useSyncExternalStore(subscribeDetail, getDetailState);
  const changesState = useSyncExternalStore(subscribeChanges, getChangesState);

  const runId = detailState.detail?.detail?.runs.items?.[0]?.run_id ?? null;

  useEffect(() => {
    setRunId(runId);
  }, [runId]);

  if (typeof store.selection !== 'number') {
    return null;
  }

  const { files, selectedPath, diffText } = changesState;
  const diffLines = diffText === '' ? [] : diffText.split('\n');

  return (
    <div
      id={detailTabPanelId('changes')}
      role="tabpanel"
      aria-labelledby={detailTabId('changes')}
      hidden={activeTab !== 'changes'}
      className="mk-console mk-well mk-stack"
    >
      {files.length === 0 && (
        <div className="mk-console-empty">
          <div>No changed files yet.</div>
          <div>Files that the agent changes appear here.</div>
        </div>
      )}
      {files.length > 0 && (
        <>
          <span className="mk-text-caption mk-muted">{files.length} files changed</span>
          <div>
            <div className="mk-kicker mk-file-head">
              <span>St</span>
              <span>Path</span>
              <span>+</span>
              <span>{'−'}</span>
            </div>
            {files.map((file) => (
              <button
                key={file.path}
                type="button"
                className={selectedPath === file.path ? 'mk-file-row is-selected' : 'mk-file-row'}
                aria-pressed={selectedPath === file.path}
                onClick={() => selectPath(file.path)}
              >
                <span>{file.status}</span>
                <span className="mk-file-path">{file.path}</span>
                <span className="mk-file-add">+{file.added}</span>
                <span className="mk-file-del">
                  {'−'}
                  {file.removed}
                </span>
              </button>
            ))}
          </div>
        </>
      )}
      {selectedPath && (
        <>
          <div className="mk-diff-path">{selectedPath}</div>
          <div role="region" aria-label="Unified diff" tabIndex={0} className="mk-diff">
            {diffLines.map((line, index) => (
              <div key={`${index}-${line}`} className={`mk-dl mk-dl-${diffLineKind(line)}`}>
                <span aria-hidden="true" />
                <span>{line}</span>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
