// The `sidecar` slot contribution: the selected node's header (ancestor
// path, title, badges) and its run summary. The tab bodies are separate
// `sidecar-section` contributions (TabSections.tsx) so the tab strip sits
// between the run summary and the active body.
import type { ReactElement } from "react";
import {
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  useSyncExternalStore,
} from "react";
import {
  getState,
  selectedNodeId,
  setSelection,
  subscribe,
} from "../../app/store";
import { Milknado } from "../../design-system";
import { toGraphNodes, type GraphNodeData } from "../../app/wire";
import { toBadgeState } from "./badgeState";
import { summarizePathTitle } from "./pathTitle";
import {
  getDetailState,
  selectNode,
  subscribeDetail,
  type WireRunRecord,
} from "../../shared/node-detail";
import {
  getSelectedReviewId,
  subscribeReviewSelection,
} from "../goal-review/selection";

const PATH_ITEM_LIMIT = 4;

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
        <dd>{run.ended_at ?? "none"}</dd>
      </div>
    </>
  );
}
function ancestorPath(nodes: GraphNodeData[], nodeId: number): GraphNodeData[] {
  const byId = new Map(nodes.map((node) => [String(node.id), node]));
  const path: GraphNodeData[] = [];
  const visited = new Set<string>();
  let node = byId.get(String(nodeId));
  while (node && !visited.has(String(node.id))) {
    path.unshift(node);
    visited.add(String(node.id));
    node = node.parent === null ? undefined : byId.get(String(node.parent));
  }
  return path;
}

function AccessibleAncestorPath({
  nodes,
  fullNodes,
  nodeId,
}: {
  nodes: GraphNodeData[];
  fullNodes: GraphNodeData[];
  nodeId: number;
}): ReactElement {
  const pathRef = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => {
    const path = ancestorPath(fullNodes, nodeId);
    const visiblePath =
      path.length > PATH_ITEM_LIMIT
        ? [path[0], ...path.slice(-(PATH_ITEM_LIMIT - 1))]
        : path;
    pathRef.current
      ?.querySelectorAll<HTMLButtonElement>(".mk-path-item")
      .forEach((button, index) => {
        const node = visiblePath[index];
        if (node) {
          button.title = node.title;
          button.setAttribute("aria-label", summarizePathTitle(node.title));
        }
      });
  }, [fullNodes, nodeId]);

  return (
    <div ref={pathRef} className="mk-path-wrapper">
      <Milknado.AncestorPath
        nodes={nodes}
        id={nodeId}
        max={PATH_ITEM_LIMIT}
        onSelect={setSelection}
      />
    </div>
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
      const clone = title.cloneNode(true) as HTMLElement;
      clone.classList.add("is-expanded");
      clone.style.position = "absolute";
      clone.style.visibility = "hidden";
      clone.style.width = `${title.clientWidth}px`;
      title.parentElement?.append(clone);
      const fullHeight = clone.getBoundingClientRect().height;
      const collapsedHeight = title.getBoundingClientRect().height;
      clone.remove();
      setDescriptionExpandable(fullHeight > collapsedHeight + 1);
    };
    updateExpandable();
    if (typeof ResizeObserver === "undefined") {
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
  const pathNodes = nodes.map((node) => ({
    ...node,
    title: summarizePathTitle(node.title),
  }));
  const runs = detail?.runs.items ?? [];
  const errors = runs
    .filter(
      (run): run is WireRunRecord & { error: string } => run.error !== null,
    )
    .map((run) => ({ runId: run.run_id, error: run.error }));

  return (
    <div className="mk-stack">
      <div className="mk-sidecar-head">
        <AccessibleAncestorPath
          nodes={pathNodes}
          fullNodes={nodes}
          nodeId={nodeId}
        />
        <Button
          icon
          className="mk-btn-ctl"
          ariaLabel="Close the sidecar"
          onClick={onClose ?? (() => setSelection(null))}
        >
          {"×"}
        </Button>
      </div>
      <div className="mk-sidecar-description">
        <h2
          ref={descriptionRef}
          className={
            descriptionExpanded
              ? "mk-sidecar-title is-expanded"
              : "mk-sidecar-title"
          }
        >
          {detail?.description ?? ""}
        </h2>
        {descriptionExpandable && (
          <button
            type="button"
            className="mk-btn mk-btn-sm mk-btn-ghost"
            aria-expanded={descriptionExpanded}
            onClick={() => setDescriptionExpanded((expanded) => !expanded)}
          >
            {descriptionExpanded
              ? "Collapse description"
              : "Expand description"}
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
        {store.capabilities !== null && !store.capabilities.host_owner.available && (
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