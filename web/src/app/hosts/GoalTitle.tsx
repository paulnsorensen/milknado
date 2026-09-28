import type { ReactElement } from 'react';
import { useState, useSyncExternalStore } from 'react';
import { ownerLabel } from '../../shared/ownerLabel';
import type { StreamSnapshot } from '../../features/live-state/runtimeSnapshot';
import { getState, subscribe } from '../store';
import type { WireNode } from '../wire';

function rootForNode(nodes: WireNode[], nodeId: number): WireNode | null {
  const byId = new Map(nodes.map((node) => [node.id, node]));
  const visited = new Set<number>();
  let node = byId.get(nodeId) ?? null;
  while (node !== null && node.parent_id !== null) {
    if (visited.has(node.id)) {
      return null;
    }
    visited.add(node.id);
    node = byId.get(node.parent_id) ?? null;
  }
  return node;
}

/** The mode kicker and the active root goal title, with a visible root switcher when needed. */
export function GoalTitle(): ReactElement {
  const store = useSyncExternalStore(subscribe, getState);
  const owner = store.capabilities?.owner.available ?? false;
  const snapshot = store.snapshot as Partial<StreamSnapshot> | null;
  const [selectedRootId, setSelectedRootId] = useState<number | null>(null);
  const graph = snapshot?.graph;
  const roots = graph
    ? graph.root_ids
        .map((id) => graph.nodes.find((node) => node.id === id))
        .filter((node): node is WireNode => node !== undefined)
    : [];
  const runningNodeId = snapshot?.active_runs?.[0]?.node_id;
  const runningRoot = runningNodeId === undefined ? null : rootForNode(graph?.nodes ?? [], runningNodeId);
  const defaultRootId = runningRoot?.id ?? roots[0]?.id ?? null;
  const selectedRoot = roots.find((root) => root.id === selectedRootId) ?? null;
  const root = selectedRoot ?? roots.find((candidate) => candidate.id === defaultRootId) ?? null;

  return (
    <div className="mk-goal-title">
      <span className="mk-kicker">{ownerLabel(owner).kicker}</span>
      <h1 className="t-display">{root?.description ?? snapshot?.goal ?? 'Milknado'}</h1>
      {roots.length > 1 && (
        <select
          aria-label="Root goal"
          className="mk-root-switcher"
          value={root?.id ?? ''}
          onChange={(event) => setSelectedRootId(Number(event.target.value))}
        >
          {roots.map((candidate) => (
            <option key={candidate.id} value={candidate.id}>
              {candidate.description}
            </option>
          ))}
        </select>
      )}
    </div>
  );
}
