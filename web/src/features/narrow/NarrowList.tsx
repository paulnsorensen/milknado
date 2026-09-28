// The narrow list view: a header bar, the mode kicker and goal title, the
// status strip and totals, a jump-to-node input, the outline tree, and an
// explicit "Open node" bar for the selected node.
import type { FormEvent, ReactElement } from 'react';
import { useState, useSyncExternalStore } from 'react';
import { getState, pushNotice, setSelection, subscribe } from '../../app/store';
import { Milknado } from '../../design-system';
import { formatRunTotals } from '../../shared/runTotals';
import { toGraphNodes } from '../../app/wire';
import { ownerLabel } from '../../shared/ownerLabel';
import { PendingPermissionIndicator } from '../session-input/PendingPermissionIndicator';
import { Wordmark } from '../../shared/Wordmark';

export interface NarrowListProps {
  onOpen: (id: string | number) => void;
}

function toggle(collapsed: Array<string | number>, id: string | number): Array<string | number> {
  return collapsed.includes(id) ? collapsed.filter((value) => value !== id) : [...collapsed, id];
}


/** The `layout` slot's list view, shown at or below the 400px breakpoint. */
export function NarrowList({ onOpen }: NarrowListProps): ReactElement {
  const state = useSyncExternalStore(subscribe, getState);
  const [collapsed, setCollapsed] = useState<Array<string | number>>([]);
  const [jumpValue, setJumpValue] = useState('');
  const { StatusBadge, StatusStrip, OutlineTree, Button } = Milknado;

  const nodes = state.snapshot?.graph ? toGraphNodes(state.snapshot.graph) : [];
  const totals = formatRunTotals(state.snapshot);
  const selectedNode = nodes.find((node) => node.id === state.selection) ?? null;
  const owner = state.capabilities?.host_owner.available ?? false;

  function jumpToNode(event: FormEvent<HTMLFormElement>): void {
    event.preventDefault();
    const trimmed = jumpValue.trim();
    if (trimmed === '') {
      return;
    }
    if (!/^\d+$/.test(trimmed)) {
      pushNotice(`Node ${trimmed} was not found.`);
      return;
    }
    const id = Number(trimmed);
    if (!nodes.some((node) => node.id === id)) {
      pushNotice(`Node ${id} was not found.`);
      return;
    }
    setSelection(id);
  }

  return (
    <div className="mk-narrow-root mk-narrow-list">
      <header className="mk-narrow-bar">
        <Wordmark />
        <div className="mk-button-row">
          <StatusBadge state={owner ? 'running' : 'pending'}>{ownerLabel(owner).badge}</StatusBadge>
          <PendingPermissionIndicator />
        </div>
      </header>
      <div className="mk-narrow-intro">
        <span className="mk-kicker">{ownerLabel(owner).kicker}</span>
        <h1 className="mk-sidecar-title">{state.snapshot?.goal ?? 'Milknado'}</h1>
        <StatusStrip nodes={nodes} />
        <span className="mk-narrow-totals mk-text-caption mk-muted">{totals}</span>
        <form className="mk-narrow-jump" onSubmit={jumpToNode}>
          <label htmlFor="mk-narrow-jump-input" className="mk-kicker">
            Jump to node
          </label>
          <input
            id="mk-narrow-jump-input"
            className="mk-input"
            inputMode="numeric"
            placeholder="Node id"
            value={jumpValue}
            onChange={(event) => setJumpValue(event.target.value)}
          />
          <Button type="submit">Jump</Button>
        </form>
      </div>
      <div className="mk-narrow-tree">
        <OutlineTree
          nodes={nodes}
          selected={state.selection}
          onSelect={setSelection}
          onOpen={onOpen}
          collapsed={collapsed}
          onToggle={(id) => setCollapsed((current) => toggle(current, id))}
          label="Graph outline"
        />
      </div>
      <div className="mk-narrow-open-bar">
        <div className="mk-narrow-selected">
          <span className="mk-text-label">{selectedNode?.title ?? 'No node selected'}</span>
          <span className="mk-text-caption mk-muted">{selectedNode ? `node ${selectedNode.id}` : ''}</span>
        </div>
        <Button variant="primary" disabled={selectedNode === null} onClick={() => selectedNode && onOpen(selectedNode.id)}>
          Open node
        </Button>
      </div>
    </div>
  );
}