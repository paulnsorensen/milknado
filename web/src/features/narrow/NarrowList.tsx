// The narrow list view: header, status strip, a jump-to-node input, the
// outline tree, and an explicit "Open node" bar for the selected node.
import type { FormEvent, ReactElement } from 'react';
import { useState, useSyncExternalStore } from 'react';
import { getState, setSelection, subscribe } from '../../app/store';
import { Milknado } from '../../design-system';
import { toGraphNodes } from '../../app/wire';

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
  const { StatusStrip, OutlineTree, Button } = Milknado;

  const nodes = state.snapshot?.graph ? toGraphNodes(state.snapshot.graph) : [];
  const selectedNode = nodes.find((node) => node.id === state.selection) ?? null;

  function jumpToNode(event: FormEvent<HTMLFormElement>): void {
    event.preventDefault();
    const id = Number(jumpValue);
    if (Number.isNaN(id) || !nodes.some((node) => node.id === id)) {
      return;
    }
    setSelection(id);
  }

  return (
    <div className="mk-narrow-root mk-narrow-list">
      <h1>{state.snapshot?.goal ?? 'Milknado'}</h1>
      <StatusStrip nodes={nodes} />
      <form className="mk-narrow-jump" onSubmit={jumpToNode}>
        <label htmlFor="mk-narrow-jump-input">Jump to node</label>
        <input
          id="mk-narrow-jump-input"
          value={jumpValue}
          onChange={(event) => setJumpValue(event.target.value)}
        />
        <Button type="submit">Jump</Button>
      </form>
      <OutlineTree
        nodes={nodes}
        selected={state.selection}
        onSelect={setSelection}
        onOpen={onOpen}
        collapsed={collapsed}
        onToggle={(id) => setCollapsed((current) => toggle(current, id))}
      />
      <div className="mk-narrow-open-bar">
        <Button
          disabled={selectedNode === null}
          onClick={() => selectedNode && onOpen(selectedNode.id)}
        >
          Open node
        </Button>
      </div>
    </div>
  );
}
