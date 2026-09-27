// The `sidecar` slot contribution: the selected node's header (ancestor
// path, title, badges) and its run summary. The tab bodies are separate
// `sidecar-section` contributions (TabSections.tsx) so the tab strip sits
// between the run summary and the active body.
import type { ReactElement } from 'react';
import { useEffect, useSyncExternalStore } from 'react';
import { getState, setSelection, subscribe } from '../../app/store';
import { Milknado } from '../../design-system';
import { toGraphNodes } from '../../app/wire';
import { toBadgeState } from './badgeState';
import { getDetailState, selectNode, subscribeDetail, type WireRunRecord } from '../../shared/node-detail';

function RunRows({ run }: { run: WireRunRecord }): ReactElement {
  return (
    <>
      <div className="mk-kv">
        <dt>Run</dt>
        <dd>{run.run_id}</dd>
      </div>
      <div className="mk-kv">
        <dt>Status</dt>
        <dd>{run.status}</dd>
      </div>
      <div className="mk-kv">
        <dt>Started</dt>
        <dd>{run.started_at}</dd>
      </div>
      <div className="mk-kv">
        <dt>Ended</dt>
        <dd>{run.ended_at ?? 'running'}</dd>
      </div>
    </>
  );
}

export function NodeSidecar(): ReactElement | null {
  const store = useSyncExternalStore(subscribe, getState);
  const detailState = useSyncExternalStore(subscribeDetail, getDetailState);
  const { AncestorPath, Button, StatusBadge } = Milknado;

  const nodeId = typeof store.selection === 'number' ? store.selection : null;

  useEffect(() => {
    selectNode(nodeId);
  }, [nodeId]);

  if (nodeId === null) {
    return null;
  }

  const detail = detailState.detail?.detail ?? null;
  const nodes = store.snapshot?.graph ? toGraphNodes(store.snapshot.graph) : [];
  const runs = detail?.runs.items ?? [];
  const errors = runs.map((run) => run.error).filter((error): error is string => error !== null);

  return (
    <div className="mk-stack">
      <div className="mk-sidecar-head">
        <AncestorPath nodes={nodes} id={nodeId} onSelect={setSelection} />
        <Button icon className="mk-btn-ctl" ariaLabel="Close the sidecar" onClick={() => setSelection(null)}>
          {'×'}
        </Button>
      </div>
      <h2 className="mk-sidecar-title">{detail?.description ?? ''}</h2>
      <div className="mk-badge-row">
        {detail && <StatusBadge state={toBadgeState(detail.node.status)} />}
        {detail?.node.flavor && <span className="mk-node-flavor">{detail.node.flavor}</span>}
        <span className="mk-text-data mk-muted">node {nodeId}</span>
      </div>
      <section className="mk-section" aria-label="Run">
        <span className="mk-kicker">Run</span>
        {runs.length === 0 && <p className="mk-text-caption mk-faint">This node has no runs yet.</p>}
        {runs.length > 0 && (
          <dl>
            {runs.map((run) => (
              <RunRows key={run.run_id} run={run} />
            ))}
          </dl>
        )}
      </section>
      {errors.map((error) => (
        <div key={error} role="alert" className="mk-alert">
          <span className="mk-glyph mk-glyph-at-risk" aria-hidden="true" />
          <span>{error}</span>
        </div>
      ))}
    </div>
  );
}