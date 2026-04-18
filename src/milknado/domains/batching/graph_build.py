from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from milknado.domains.batching.change import FileChange, NewRelationship, SymbolRef
from milknado.domains.common.protocols import CrgPort


def build_change_graph(
    changes: Sequence[FileChange],
    crg: CrgPort | None = None,
    new_relationships: Sequence[NewRelationship] = (),
) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...], dict[str, tuple[SymbolRef, ...]]]:
    """Return (nodes, edges, symbols_by_node) for the given change set.

    Edge sources come from three independent sources, all unioned together:

    1. CRG impact radius — when ``crg`` is provided, each changed file is queried
       for structural dependants. Only edges between two nodes that are already in
       ``changes`` are kept; cross-boundary blast-radius results are ignored.
       Pass ``crg=None`` (e.g. in tests or when no graph is built) to skip this
       step entirely — the result is still valid, just structurally thinner.

    2. Explicit ``depends_on`` — each ``FileChange`` may declare ids it must come
       after. These are validated against the known id set; unknown ids raise.

    3. ``new_relationships`` — caller-supplied ``NewRelationship`` records for
       relationships the CRG doesn't yet know about (new files, new imports, etc.).

    Returns:
        nodes: change ids in input order
        edges: deduplicated (src_id, dst_id) pairs, sorted for determinism
        symbols_by_node: maps each change id to its declared symbol refs
    """
    path_to_id = {c.path: c.id for c in changes}
    known_ids = {c.id for c in changes}
    seen_edges: set[tuple[str, str]] = set()

    def _add_edge(src: str, dst: str) -> None:
        if src != dst:
            seen_edges.add((src, dst))

    for change in changes:
        if crg is not None:
            result = crg.get_impact_radius([change.path])
            for src_path, dst_path in _parse_impact_dict(result, change.path, set(path_to_id)):
                # _parse_impact_dict filters to known_paths = set(path_to_id),
                # so both paths are always present in path_to_id.
                _add_edge(path_to_id[src_path], path_to_id[dst_path])

    for change in changes:
        for dep_id in change.depends_on:
            if dep_id not in known_ids:
                raise ValueError(f"unknown depends_on id: {dep_id}")
            _add_edge(dep_id, change.id)

    for rel in new_relationships:
        if rel.source_change_id not in known_ids:
            raise ValueError(f"unknown edge endpoint: {rel.source_change_id}")
        if rel.dependant_change_id not in known_ids:
            raise ValueError(f"unknown edge endpoint: {rel.dependant_change_id}")
        _add_edge(rel.source_change_id, rel.dependant_change_id)

    nodes = tuple(c.id for c in changes)
    edges = tuple(sorted(seen_edges))
    symbols_by_node = {c.id: c.symbols for c in changes}
    return nodes, edges, symbols_by_node


def _parse_impact_dict(
    result: dict[str, Any],
    source_path: str,
    known_paths: set[str],
) -> list[tuple[str, str]]:
    """Parse CRG's opaque dict. Returns list of (src_path, dst_path) path pairs.

    Narrow except clauses: only absorb shape-variation errors (KeyError,
    TypeError, AttributeError) that arise from the untyped dict contract.
    Programmer errors propagate.
    """
    pairs: list[tuple[str, str]] = []
    edges = result.get("edges")
    if isinstance(edges, list):
        for edge in edges:
            try:
                src = edge["src"]
                dst = edge["dst"]
            except (KeyError, TypeError):
                continue
            if src in known_paths and dst in known_paths:
                pairs.append((src, dst))
    for key in ("impacted_files", "files"):
        files = result.get(key)
        if isinstance(files, list):
            for f in files:
                if isinstance(f, str) and f in known_paths and f != source_path:
                    pairs.append((source_path, f))
            break
    return pairs


def contract_sccs(
    nodes: Sequence[str],
    edges: Sequence[tuple[str, str]],
) -> tuple[dict[str, str], tuple[tuple[str, str], ...]]:
    """Collapse graph cycles into single SCC nodes using Tarjan's algorithm.

    A Strongly Connected Component (SCC) is a maximal set of nodes where every
    node is reachable from every other node via directed edges. In a change graph
    this means a cycle: two files that mutually import each other, or a ring of
    ``depends_on`` declarations. Cyclic nodes *must* land in the same batch — the
    solver cannot legally separate them — so we collapse each SCC into one
    representative node before handing the graph to CP-SAT.

    SCC id is the lexicographically smallest member id, making it stable across
    runs regardless of discovery order.

    The returned ``dag_edges`` are the edges *between* distinct SCCs after
    collapsing. Self-loops (src_scc == dst_scc) are excluded, and duplicate edges
    are deduplicated. The result is always a DAG (no cycles), which CP-SAT can
    then enforce as ``batch_of[src] <= batch_of[dst]`` ordering constraints.

    Implementation note: Tarjan's algorithm is iterative (explicit call stack)
    rather than recursive to avoid Python's default recursion limit on large
    change sets.

    Returns:
        scc_of: maps every node id to its SCC id (a node that forms its own
                SCC maps to itself)
        dag_edges: sorted, deduplicated (src_scc_id, dst_scc_id) pairs
    """
    adj: dict[str, list[str]] = defaultdict(list)
    for src, dst in edges:
        adj[src].append(dst)

    state = _TarjanState()
    for n in nodes:
        if n not in state.index:
            _strongconnect(n, adj, state)

    dag_seen: set[tuple[str, str]] = set()
    for src, dst in edges:
        s, d = state.scc_of[src], state.scc_of[dst]
        if s != d:
            dag_seen.add((s, d))

    return state.scc_of, tuple(sorted(dag_seen))


@dataclass
class _TarjanState:
    index_counter: int = 0
    stack: list[str] = field(default_factory=list)
    on_stack: set[str] = field(default_factory=set)
    index: dict[str, int] = field(default_factory=dict)
    lowlink: dict[str, int] = field(default_factory=dict)
    scc_of: dict[str, str] = field(default_factory=dict)


def _strongconnect(start: str, adj: dict[str, list[str]], s: _TarjanState) -> None:
    call_stack: list[tuple[str, int]] = [(start, 0)]
    while call_stack:
        v, i = call_stack[-1]
        if i == 0:
            s.index[v] = s.lowlink[v] = s.index_counter
            s.index_counter += 1
            s.stack.append(v)
            s.on_stack.add(v)
        advanced = _advance_neighbor(v, i, adj[v], call_stack, s)
        if not advanced:
            _finish_node(v, call_stack, s)


def _advance_neighbor(
    v: str, i: int, neighbors: list[str], call_stack: list[tuple[str, int]], s: _TarjanState
) -> bool:
    while i < len(neighbors):
        w = neighbors[i]
        i += 1
        if w not in s.index:
            call_stack[-1] = (v, i)
            call_stack.append((w, 0))
            return True
        if w in s.on_stack:
            s.lowlink[v] = min(s.lowlink[v], s.index[w])
    return False


def _finish_node(v: str, call_stack: list[tuple[str, int]], s: _TarjanState) -> None:
    call_stack.pop()
    if call_stack:
        parent, _ = call_stack[-1]
        s.lowlink[parent] = min(s.lowlink[parent], s.lowlink[v])
    if s.lowlink[v] != s.index[v]:
        return
    members: list[str] = []
    while True:
        w = s.stack.pop()
        s.on_stack.discard(w)
        members.append(w)
        if w == v:
            break
    scc_id = min(members)
    for m in members:
        s.scc_of[m] = scc_id


def symbols_by_scc(
    scc_of: dict[str, str],
    symbols_by_node: dict[str, tuple[SymbolRef, ...]],
) -> dict[str, tuple[SymbolRef, ...]]:
    """Union symbols across all nodes in each SCC."""
    result: dict[str, list[SymbolRef]] = defaultdict(list)
    for node_id, syms in symbols_by_node.items():
        scc_id = scc_of[node_id]
        for sym in syms:
            if sym not in result[scc_id]:
                result[scc_id].append(sym)
    return {k: tuple(v) for k, v in result.items()}
