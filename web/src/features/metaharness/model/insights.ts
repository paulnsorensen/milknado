// Read-only projections over the snapshot: the project cards, open
// issues worth a fix task, and the session stats tiles.
import type { WireMhSnapshot } from '../wire';
import type { Need } from './needs';
import type { Tree } from './tree';
import { openPrerequisites, subtreeIds, title } from './tree';

export interface ProjectCard {
  id: number;
  title: string;
  needCount: number;
  running: number;
  pct: number;
}

export function projectCards(tree: Tree, needs: Map<number, Need>): ProjectCard[] {
  return tree.roots.map((root) => {
    const ids = subtreeIds(tree, root.id);
    const nodes = ids.map((id) => tree.byId.get(id)).filter((node) => node !== undefined);
    const done = nodes.filter((node) => node.status === 'done').length;
    return {
      id: root.id,
      title: title(root),
      needCount: ids.filter((id) => needs.has(id)).length,
      running: nodes.filter((node) => node.kind === 'task' && node.status === 'running').length,
      pct: nodes.length ? Math.round((done / nodes.length) * 100) : 0,
    };
  });
}

export type IssueSeverity = 'high' | 'medium';

export interface Issue {
  id: string;
  severity: IssueSeverity;
  text: string;
  target: string;
  nodeId: number | null;
}

export function deriveIssues(snapshot: WireMhSnapshot | null, tree: Tree): Issue[] {
  const issues: Issue[] = (snapshot?.listener_errors ?? []).map((text, i) => ({
    id: `listener:${i}`, severity: 'high', text, target: 'snapshot listener', nodeId: null,
  }));
  for (const run of snapshot?.active_runs ?? []) {
    if (run.stalled) {
      const node = tree.byId.get(run.node_id);
      issues.push({
        id: `stalled:${run.run_id}`,
        severity: 'high',
        text: `Run ${run.run_id.slice(0, 8)} stopped reporting progress.`,
        target: `#${run.node_id} · ${node ? title(node) : run.description}`,
        nodeId: run.node_id,
      });
    }
  }
  for (const node of tree.byId.values()) {
    if (node.status === 'blocked' && openPrerequisites(tree, node.id) === 0) {
      issues.push({
        id: `blocked:${node.id}`,
        severity: 'medium',
        text: `${title(node)} is blocked but has no open prerequisites.`,
        target: `#${node.id} · blocked`,
        nodeId: node.id,
      });
    }
  }
  return issues;
}

export interface Stat {
  label: string;
  value: string;
  sub: string;
}

export interface StatsInput {
  snapshot: WireMhSnapshot | null;
  tree: Tree;
  concurrency: number | null;
  paused: boolean;
}

export function deriveStats({ snapshot, tree, concurrency, paused }: StatsInput): Stat[] {
  const nodes = [...tree.byId.values()];
  const count = (status: string): number => nodes.filter((node) => node.status === status).length;
  const completed = snapshot?.completed ?? 0;
  const failed = snapshot?.failed ?? 0;
  const finished = completed + failed;
  const active = snapshot?.active_runs?.length ?? 0;
  return [
    { label: 'Nodes', value: String(nodes.length), sub: `${count('done')} done · ${active} running` },
    { label: 'Gate pass', value: finished ? `${Math.round((completed / finished) * 100)}%` : '—', sub: `${finished} finished runs` },
    { label: 'Runs', value: String(completed), sub: `${failed} failed · ${snapshot?.stopped ?? 0} stopped` },
    { label: 'Workers', value: `${active}/${concurrency ?? '—'}`, sub: paused ? 'loop paused' : 'loop on' },
  ];
}
