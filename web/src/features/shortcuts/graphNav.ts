// Graph-key navigation and view toggles: reads the current selection and the
// graph snapshot directly from the store, bypassing the action registry.
import { getState, setGraphView, setSelection } from '../../app/store';
import { Milknado } from '../../design-system';
import { toGraphNodes, type GraphNodeData } from '../../app/wire';
import { setActiveTab } from '../../shared/node-detail';

function graphNodes(): GraphNodeData[] {
  const graph = getState().snapshot?.graph;
  return graph ? toGraphNodes(graph) : [];
}

function selectSiblingBy(offset: number): void {
  const nodes = graphNodes();
  if (nodes.length === 0) {
    return;
  }
  const { byId, children } = Milknado.tree.index(nodes);
  const selection = getState().selection;
  const current = selection !== null ? byId[String(selection)] : undefined;
  const siblings =
    current === undefined || current.parent === null
      ? nodes.filter((node) => node.parent === null)
      : (children[String(current.parent)] ?? []);
  if (siblings.length === 0) {
    return;
  }
  const index = current === undefined ? -1 : siblings.findIndex((node) => node.id === current.id);
  const next = index === -1 ? 0 : (index + offset + siblings.length) % siblings.length;
  setSelection(siblings[next].id);
}

export function selectNextSibling(): void {
  selectSiblingBy(1);
}

export function selectPreviousSibling(): void {
  selectSiblingBy(-1);
}

export function selectParent(): void {
  const nodes = graphNodes();
  const { byId } = Milknado.tree.index(nodes);
  const selection = getState().selection;
  const current = selection !== null ? byId[String(selection)] : undefined;
  if (current?.parent == null) {
    return;
  }
  setSelection(current.parent);
}

export function selectFirstChild(): void {
  const nodes = graphNodes();
  const { byId, children } = Milknado.tree.index(nodes);
  const selection = getState().selection;
  const current = selection !== null ? byId[String(selection)] : undefined;
  if (current === undefined) {
    return;
  }
  const child = children[String(current.id)]?.[0];
  if (child === undefined) {
    return;
  }
  setSelection(child.id);
}

export function clearSelection(): void {
  setSelection(null);
}

export function openSelectedDetails(): void {
  if (getState().selection !== null) {
    setActiveTab('details');
  }
}

export function toggleSelectedCollapsed(): void {
  const { selection, graphView } = getState();
  if (selection === null) {
    return;
  }
  const collapsed = graphView.collapsed.includes(selection)
    ? graphView.collapsed.filter((id) => id !== selection)
    : [...graphView.collapsed, selection];
  setGraphView({ collapsed });
}
