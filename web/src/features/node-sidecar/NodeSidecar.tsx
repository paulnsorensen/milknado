// The `sidecar` slot contribution: the selected node's header (ancestor
// path, title, badges) and its run summary. The tab bodies are separate
// `sidecar-section` contributions (TabSections.tsx) so the tab strip sits
// between the run summary and the active body.
import type { ReactElement } from 'react';
import {
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  useSyncExternalStore,
} from 'react';
import {
  getState,
  selectedNodeId,
  setSelection,
  subscribe,
} from '../../app/store';
import { Milknado } from '../../design-system';
import { toGraphNodes } from '../../app/wire';
import { toBadgeState } from './badgeState';
import { SidecarAncestorPath } from './SidecarAncestorPath';
import {
  getDetailState,
  selectNode,
  subscribeDetail,
  type WireRunRecord,
} from '../../shared/node-detail';
import {
  getSelectedReviewId,
  subscribeReviewSelection,
} from '../goal-review/selection';

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
        <dt>Completed</dt>
        <dd>{run.ended_at ?? 'none'}</dd>
      </div>
    </>
  );
}
export interface NodeSidecarProps {
  /** Runs on the close button instead of clearing the selection. */
  onClose?: () => void;
}

export function NodeSidecar({
  onClose,
}: NodeSidecarProps = {}): ReactElement | null {
  const store = useSyncExternalStore(subscribe, getState);
  const detailState = useSyncExternalStore(subscribeDetail, getDetailState);
  const selectedReviewId = useSyncExternalStore(
    subscribeReviewSelection,
    getSelectedReviewId,
  );
  const { Button, StatusBadge } = Milknado;
  const [descriptionExpanded, setDescriptionExpanded] = useState(false);
  const [descriptionExpandable, setDescriptionExpandable] = useState(false);
  const descriptionRef = useRef<HTMLHeadingElement>(null);

  const nodeId = selectedNodeId(store);
  const detail = detailState.detail?.detail ?? null;

  useEffect(() => {
    selectNode(nodeId);
    setDescriptionExpanded(false);
    setDescriptionExpandable(false);
  }, [nodeId]);
  useLayoutEffect(() => {
    const title = descriptionRef.current;
    if (!title || descriptionExpanded) {
      return;
    }
    const updateExpandable = () => {
      // The clamped box keeps the full text in its scroll extent.
      setDescriptionExpandable(title.scrollHeight > title.clientHeight + 1);
    };
    updateExpandable();
    if (typeof ResizeObserver === 'undefined') {
      return;
    }
    const observer = new ResizeObserver(updateExpandable);
    observer.observe(title);
    return () => observer.disconnect();
  }, [detail?.description, descriptionExpanded]);

  if (nodeId === null) {
    if (selectedReviewId !== null) {
      return null;
    }
    return (
      <p className="mk-text-caption mk-muted">
        Select a node to inspect its details.
      </p>
    );
  }

  const nodes = store.snapshot?.graph ? toGraphNodes(store.snapshot.graph) : [];
  const runs = detail?.runs.items ?? [];
  const errors = runs
    .filter(
      (run): run is WireRunRecord & { error: string } => run.error !== null,
    )
    .map((run) => ({ runId: run.run_id, error: run.error }));

  return (
    <div className="mk-stack">
      <div className="mk-sidecar-head">
        <SidecarAncestorPath nodes={nodes} nodeId={nodeId} />
        <Button
          icon
          className="mk-btn-ctl"
          ariaLabel="Close the sidecar"
          onClick={onClose ?? (() => setSelection(null))}
        >
          {'×'}
        </Button>
      </div>
      <div className="mk-sidecar-description">
        <h2
          ref={descriptionRef}
          className={
            descriptionExpanded
              ? 'mk-sidecar-title is-expanded'
              : 'mk-sidecar-title'
          }
        >
          {detail?.description ?? ''}
        </h2>
        {descriptionExpandable && (
          <button
            type="button"
            className="mk-btn mk-btn-sm mk-btn-ghost"
            aria-expanded={descriptionExpanded}
            onClick={() => setDescriptionExpanded((expanded) => !expanded)}
          >
            {descriptionExpanded
              ? 'Collapse description'
              : 'Expand description'}
          </button>
        )}
      </div>
      <div className="mk-badge-row">
        {detail && <StatusBadge state={toBadgeState(detail.node.status)} />}
        {detail?.node.flavor && (
          <span className="mk-node-flavor">{detail.node.flavor}</span>
        )}
        <span className="mk-text-data mk-muted">node {nodeId}</span>
      </div>
      <section className="mk-section" aria-label="Run">
        <span className="mk-kicker">Run</span>
        {runs.length === 0 && (
          <p className="mk-text-caption mk-faint">This node has no runs yet.</p>
        )}
        {runs.length > 0 && (
          <dl>
            {runs.map((run) => (
              <RunRows key={run.run_id} run={run} />
            ))}
          </dl>
        )}
        {store.capabilities !== null &&
          !store.capabilities.host_owner.available && (
            <dl>
              <div className="mk-kv">
                <dt>ETA</dt>
                <dd>unavailable</dd>
              </div>
              <div className="mk-kv">
                <dt>Attempt</dt>
                <dd>unavailable</dd>
              </div>
              <div className="mk-kv">
                <dt>guidance</dt>
                <dd>unavailable</dd>
              </div>
            </dl>
          )}
      </section>
      {errors.map(({ runId, error }) => (
        <div key={runId} role="alert" className="mk-alert">
          <span className="mk-glyph mk-glyph-at-risk" aria-hidden="true" />
          <span>{error}</span>
        </div>
      ))}
    </div>
  );
}
