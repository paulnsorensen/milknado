// Milknado — component types. Documentation for consumers; the bundle is plain JS.
// Load order: tokens.css, bundle.css, react 18, react-dom 18, bundle.js → window.Milknado.
import type * as React from 'react';

/** A node or agent state. `ready` is a pending leaf that can be dispatched. */
export type State = 'pending' | 'ready' | 'running' | 'at-risk' | 'failed' | 'blocked' | 'done';
/** An agent state. `listening` is the coordinator waiting for input. */
export type AgentStatus = 'running' | 'idle' | 'listening' | 'at-risk';
export type NodeKind = 'goal' | 'subgoal' | 'task';
export type Flavor = 'goal' | 'implement' | 'spike' | 'review' | 'prototype';

export interface ButtonProps {
  /** primary: accent outline. secondary: line-strong outline. ghost: no outline. Default secondary. */
  variant?: 'primary' | 'secondary' | 'ghost';
  /** A 28px square button with one glyph and an ariaLabel. */
  icon?: boolean;
  /** A leading text glyph: '+', '▷', '⤓', '¶'. */
  glyph?: string;
  /** A count shown in caption after the label: "Dispatch ready 3". */
  count?: number | string;
  disabled?: boolean;
  ariaLabel?: string;
  title?: string;
  type?: 'button' | 'submit';
  className?: string;
  onClick?: (e: React.MouseEvent) => void;
  children?: React.ReactNode;
}
export const Button: React.FC<ButtonProps>;

export interface StatusGlyphProps { state: State | AgentStatus; className?: string; }
/** The 10px status mark in currentColor. */
export const StatusGlyph: React.FC<StatusGlyphProps>;

export interface StatusBadgeProps { state: State | AgentStatus; className?: string; /** Overrides the default word. */ children?: React.ReactNode; }
export const StatusBadge: React.FC<StatusBadgeProps>;

export interface ContextMeterProps {
  /** 0–100. Under 50 accent; 50–80 at-risk; over 80 blocked. */
  percent: number;
  /** The word left of the figure. false hides the head row. Default "context". */
  label?: string | false;
  className?: string;
}
export const ContextMeter: React.FC<ContextMeterProps>;

export interface AgentRowProps {
  /** A raptor name, or the role for the coordinator and the reviewer. */
  name: string;
  status?: AgentStatus;
  /** The caption line: "node 6 · iter 4/8". */
  sub?: string;
  /** The mono figure on the right: "PASS", "gate…", "2h 04m". */
  figure?: string;
  selected?: boolean;
  className?: string;
  onClick?: () => void;
}
export const AgentRow: React.FC<AgentRowProps>;

export interface Agent { id?: string; name: string; status?: AgentStatus; sub?: string; figure?: string; }
export interface AgentRosterProps {
  agents: Agent[];
  /** The id (or name) of the selected agent. */
  selected?: string;
  onSelect?: (id: string) => void;
  /** The kicker. Default "Agents". */
  title?: string;
  /** The count on the right: "3/4 workers". */
  count?: string;
  className?: string;
}
export const AgentRoster: React.FC<AgentRosterProps>;

export interface ConsoleLine { time: string; text: string; tone?: 'plain' | 'result' | 'noise' | 'warn'; }
export interface ConsoleProps {
  lines: ConsoleLine[];
  tabs?: string[];
  activeTab?: string;
  onTab?: (tab: string) => void;
  /** Controlled draft. Omit for an internal draft. */
  value?: string;
  onChange?: (value: string) => void;
  onSend?: (text: string) => void;
  placeholder?: string;
  sendLabel?: string;
  /** false hides the guidance input. */
  input?: boolean;
  /** false keeps the scroll position when lines change. */
  autoScroll?: boolean;
  emptyTitle?: string;
  emptyHint?: string;
  className?: string;
  style?: React.CSSProperties;
}
export const Console: React.FC<ConsoleProps>;

export interface GraphNodeProps {
  title: string;
  kind?: NodeKind;
  state?: State;
  /** Overrides the status word: "goal · 9 / 12". */
  statusText?: string;
  flavor?: Flavor | string;
  /** The caption line: "needs Stats dashboard", "Blocked by node 8". */
  meta?: string;
  /** The assigned agent, shown as an accent chip. */
  agent?: string;
  /** An artifact awaiting review: true, or its kind ("plan"). */
  artifact?: boolean | string;
  /** 0–100; shown only when state is running. */
  progress?: number;
  /** The 160px one-line style used inside MikadoGraph. */
  compact?: boolean;
  selected?: boolean;
  className?: string;
  style?: React.CSSProperties;
  onClick?: (e: React.MouseEvent) => void;
}
export const GraphNode: React.FC<GraphNodeProps>;

export interface GraphNodeData {
  id: string | number;
  title: string;
  kind?: NodeKind;
  state?: State;
  statusText?: string;
  flavor?: Flavor | string;
  meta?: string;
  agent?: string;
  artifact?: boolean | string;
  progress?: number;
  /** The tree parent. The root has none. */
  parent?: string | number | null;
  /** Extra prerequisites, drawn dashed. */
  extra?: Array<string | number>;
}
export interface StatusCounts { total: number; done: number; running: number; ready: number; blocked: number; failed: number; pending: number; }
export interface StatusStripProps { nodes?: GraphNodeData[]; counts?: StatusCounts; /** "done / total" only. */ short?: boolean; /** false hides the figure. */ label?: boolean; className?: string; }
/** The counts of a set of nodes as one stacked bar. */
export const StatusStrip: React.FC<StatusStripProps>;

export interface GroupNodeProps { title: string; descendants?: GraphNodeData[]; counts?: StatusCounts; selected?: boolean; className?: string; style?: React.CSSProperties; onClick?: (e: React.MouseEvent) => void; }
/** A collapsed subtree with its counts. */
export const GroupNode: React.FC<GroupNodeProps>;

export type Lod = 'card' | 'pill' | 'dot';
export interface MikadoGraphProps {
  /** Sub-goal ids to show as GroupNodes. */
  collapsed?: Array<string | number>;
  onExpand?: (id: string | number) => void;
  /** Collapse a subtree whose nodes are all done. Default true. */
  collapseDone?: boolean;
  /** A node id: show it with its ancestors and descendants; fade the rest. */
  focus?: string | number | null;
  /** Keep matching nodes and their ancestors. */
  filter?: 'ready' | 'running' | 'blocked' | null;
  hideDone?: boolean;
  /** The requested level of detail. */
  lod?: Lod;
  /** The canvas zoom; sets lod when lod is absent. */
  zoom?: number;
  /** Step down the level of detail when a level does not fit. Default true. */
  autoLod?: boolean;
  /** elbow (rounded corners, default), curve (one S-curve) or square. */
  edgeStyle?: 'elbow' | 'curve' | 'square';
  /** Over this many edges, only hot and critical edges draw. Default 200. */
  edgeBudget?: number;
  /** Reports positions, visible ids, collapsed ids and the lod in use. */
  onLayout?: (layout: { positions: Record<string, { x: number; y: number }>; visible: Array<string | number>; collapsed: string[]; lod: Lod }) => void;
  nodes: GraphNodeData[];
  selected?: string | number | null;
  /** Called with null on a canvas click. */
  onSelect?: (id: string | number | null) => void;
  width?: number;
  /** A number, or 'auto' to fit the content. */
  height?: number | 'auto';
  /** Vertical distance between levels. Default 100. */
  levelGap?: number;
  /** false renders full-size nodes with meta, agent and artifact. Default true. */
  compact?: boolean;
  showFlavor?: boolean;
  /** The bottom-left hint. false hides it. */
  hint?: string | false;
  className?: string;
}
export const MikadoGraph: React.FC<MikadoGraphProps>;

export interface GraphToolbarProps {
  nodes: GraphNodeData[];
  filter?: 'ready' | 'running' | 'blocked' | null; onFilter?: (f: 'ready' | 'running' | 'blocked' | null) => void;
  hideDone?: boolean; onHideDone?: (v: boolean) => void;
  focus?: boolean; canFocus?: boolean; onFocus?: (v: boolean) => void;
  lod?: Lod; onLod?: (l: Lod) => void;
  onJump?: (id: string | number) => void;
  onCollapseAll?: () => void; onExpandAll?: () => void;
  onZoomIn?: () => void; onZoomOut?: () => void; onFit?: () => void;
  /** The breadcrumb text for a search result. */
  pathFor?: (id: string | number) => string;
  className?: string;
}
export const GraphToolbar: React.FC<GraphToolbarProps>;

export interface MinimapProps { nodes: GraphNodeData[]; selected?: string | number | null; viewport?: { x: number; y: number; w: number; h: number }; onJump?: (id: string | number) => void; width?: number; height?: number; /** Show groups, not nodes, over this count. Default 200. */ groupAbove?: number; className?: string; }
export const Minimap: React.FC<MinimapProps>;

export interface OutlineTreeProps { nodes: GraphNodeData[]; selected?: string | number | null; onSelect?: (id: string | number) => void; onOpen?: (id: string | number) => void; collapsed?: Array<string | number>; onToggle?: (id: string | number) => void; query?: string; label?: string; className?: string; }
export const OutlineTree: React.FC<OutlineTreeProps>;

export interface AncestorPathProps { nodes: GraphNodeData[]; id?: string | number | null; onSelect?: (id: string | number) => void; /** Visible items. Default 4. */ max?: number; className?: string; }
export const AncestorPath: React.FC<AncestorPathProps>;

/** The level of detail for a zoom: card at 0.8 and up, pill at 0.5, dot below. */
export function lodFor(zoom?: number): Lod;
/** Tree helpers the components share. */
export const tree: { index(nodes: GraphNodeData[]): { byId: Record<string, GraphNodeData>; children: Record<string, GraphNodeData[]> }; ancestors(id: string | number, byId: Record<string, GraphNodeData>): GraphNodeData[]; descendants(id: string | number, children: Record<string, GraphNodeData[]>): GraphNodeData[]; counts(nodes: GraphNodeData[]): StatusCounts; };

declare global { interface Window { Milknado: { Button: typeof Button; StatusGlyph: typeof StatusGlyph; StatusBadge: typeof StatusBadge; ContextMeter: typeof ContextMeter; AgentRow: typeof AgentRow; AgentRoster: typeof AgentRoster; Console: typeof Console; GraphNode: typeof GraphNode; StatusStrip: typeof StatusStrip; GroupNode: typeof GroupNode; MikadoGraph: typeof MikadoGraph; GraphToolbar: typeof GraphToolbar; Minimap: typeof Minimap; OutlineTree: typeof OutlineTree; AncestorPath: typeof AncestorPath; lodFor: typeof lodFor; tree: typeof tree; }; } }
