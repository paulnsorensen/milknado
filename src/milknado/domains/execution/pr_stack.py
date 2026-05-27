"""Per-node PR stacking: group completed branches by file-overlap, open stacked PRs."""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from milknado.domains.graph import MikadoGraph

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StackedPr:
    node_id: int
    branch: str
    base_branch: str
    pr_url: str


def build_overlap_groups(node_ids: list[int], graph: MikadoGraph) -> list[list[int]]:
    """Group node IDs into overlap clusters via Union-Find on shared file ownership."""
    if not node_ids:
        return []

    ownership: dict[int, set[str]] = {nid: set(graph.get_file_ownership(nid)) for nid in node_ids}

    parent: dict[int, int] = {nid: nid for nid in node_ids}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        parent[find(x)] = find(y)

    for left, right in combinations(node_ids, 2):
        if ownership[left] & ownership[right]:
            union(left, right)

    groups: dict[int, list[int]] = {}
    for nid in node_ids:
        root = find(nid)
        groups.setdefault(root, []).append(nid)

    return list(groups.values())


def _topo_sort_group(group: list[int], graph: MikadoGraph) -> list[int]:
    """Topological sort within a group using Mikado edges; tie-break by node_id."""
    group_set = set(group)
    in_degree: dict[int, int] = {nid: 0 for nid in group}
    children_in_group: dict[int, list[int]] = {nid: [] for nid in group}

    for nid in group:
        node = graph.get_node(nid)
        if node and node.parent_id and node.parent_id in group_set:
            in_degree[nid] += 1
            children_in_group[node.parent_id].append(nid)

    # Process nodes with zero in-degree first (i.e. prerequisites before dependents)
    queue = sorted(nid for nid, deg in in_degree.items() if deg == 0)
    result: list[int] = []
    while queue:
        nid = queue.pop(0)
        result.append(nid)
        for child in sorted(children_in_group[nid]):
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)
                queue.sort()

    # Append any remaining (cycles or disconnected — shouldn't happen, but be safe)
    remaining = sorted(nid for nid in group if nid not in result)
    result.extend(remaining)
    return result


def open_stacked_prs(
    completed_branches: dict[int, str],
    graph: MikadoGraph,
    base_branch: str,
    project_root: Path,
) -> list[StackedPr]:
    """Group completed branches by file overlap and open stacked PRs via gh CLI.

    Limitation: pr_stack is sound only for file-independent nodes. Each node's
    worktree branches from feature_branch HEAD and stage_for_pr never advances
    feature_branch, so an overlapping dependent is NOT rebased onto its
    prerequisite's staged branch during execution — its diff is computed against
    feature_branch, not the prerequisite. We stack the *PRs* (base = prior staged
    branch) but not the underlying branches. See flag-for-followup.
    """
    if not completed_branches:
        return []

    node_ids = list(completed_branches.keys())
    groups = build_overlap_groups(node_ids, graph)
    results: list[StackedPr] = []

    for group in groups:
        ordered = _topo_sort_group(group, graph)
        current_base = base_branch
        for nid in ordered:
            branch = completed_branches[nid]
            node = graph.get_node(nid)
            title = node.description[:72] if node else f"milknado node {nid}"
            body = node.description if node else f"Milknado node {nid}"
            pr_url = _gh_create_pr(
                head=branch,
                base=current_base,
                title=title,
                body=body,
                project_root=project_root,
            )
            results.append(
                StackedPr(node_id=nid, branch=branch, base_branch=current_base, pr_url=pr_url)
            )
            current_base = branch

    return results


def _gh_create_pr(
    head: str,
    base: str,
    title: str,
    body: str,
    project_root: Path,
) -> str:
    """Run `gh pr create` and return the PR URL."""
    result = subprocess.run(
        ["gh", "pr", "create", "--head", head, "--base", base, "--title", title, "--body", body],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=True,
    )
    pr_url = result.stdout.strip()
    _logger.info("pr_created head=%s base=%s url=%s", head, base, pr_url)
    return pr_url
