from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from milknado.domains.batching.change import FileChange, NewRelationship, SymbolRef
from milknado.domains.common.protocols import CrgPort


def _build_path_to_ids(changes: Sequence[FileChange]) -> dict[str, list[str]]:
    """Map each path to all change ids that touch it."""
    path_to_ids: dict[str, list[str]] = {}
    for c in changes:
        for p in c.all_paths():
            path_to_ids.setdefault(p, []).append(c.id)
    return path_to_ids


def build_change_graph(
    changes: Sequence[FileChange],
    crg: CrgPort | None = None,
    new_relationships: Sequence[NewRelationship] = (),
) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...], dict[str, tuple[SymbolRef, ...]]]:
    """Return (nodes, edges, symbols_by_node) for the given change set.

    Edges come from three sources: CRG impact radius (symbol-aware routing),
    explicit ``depends_on`` declarations, and ``new_relationships``.
    Only edges between nodes present in ``changes`` are kept.
    """
    path_to_ids = _build_path_to_ids(changes)
    id_to_change = {c.id: c for c in changes}
    known_ids = set(id_to_change)
    seen_edges: set[tuple[str, str]] = set()

    def _add_edge(src: str, dst: str) -> None:
        if src != dst:
            seen_edges.add((src, dst))

    for change in changes:
        if crg is not None:
            result = crg.get_impact_radius([change.path])
            for src_id, dst_id in _parse_impact_dict(result, change.path, path_to_ids, id_to_change):
                _add_edge(src_id, dst_id)

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


def _parse_qualified(endpoint: str) -> tuple[str, str | None]:
    """Split a qualified CRG endpoint into (path, symbol_name | None)."""
    if "::" in endpoint:
        path, symbol = endpoint.split("::", 1)
        return path, symbol
    return endpoint, None


def _resolve_ids_for_endpoint(
    path: str,
    symbol: str | None,
    path_to_ids: dict[str, list[str]],
    id_to_change: dict[str, FileChange],
) -> list[str]:
    """Return the change ids attributed to a single CRG edge endpoint."""
    ids = path_to_ids.get(path, [])
    if len(ids) == 0:
        return []
    if len(ids) == 1:
        return list(ids)
    if symbol is None:
        return list(ids)
    matches = [
        cid for cid in ids
        if any(sr.name == symbol and sr.file == path for sr in id_to_change[cid].symbols)
    ]
    if len(matches) == 1:
        return matches
    return list(ids)


def _parse_impact_dict(
    result: dict[str, Any],
    source_path: str,
    path_to_ids: dict[str, list[str]],
    id_to_change: dict[str, FileChange],
) -> list[tuple[str, str]]:
    """Parse CRG's opaque dict. Returns deduplicated (src_id, dst_id) pairs.

    Supports qualified endpoints ``"path::symbol"`` for symbol-level routing
    when multiple changes share a path. Falls back to fan-out when symbol is
    unknown or ambiguous.

    Narrow except clauses: only absorb shape-variation errors (KeyError,
    TypeError) that arise from the untyped dict contract. Programmer errors
    propagate.
    """
    pairs: set[tuple[str, str]] = set()

    edges = result.get("edges")
    if isinstance(edges, list):
        for edge in edges:
            try:
                raw_src = edge["src"]
                raw_dst = edge["dst"]
            except (KeyError, TypeError):
                continue
            src_path, src_sym = _parse_qualified(raw_src)
            dst_path, dst_sym = _parse_qualified(raw_dst)
            src_ids = _resolve_ids_for_endpoint(src_path, src_sym, path_to_ids, id_to_change)
            dst_ids = _resolve_ids_for_endpoint(dst_path, dst_sym, path_to_ids, id_to_change)
            for s in src_ids:
                for d in dst_ids:
                    pairs.add((s, d))

    for key in ("impacted_files", "files"):
        files = result.get(key)
        if isinstance(files, list):
            src_ids = path_to_ids.get(source_path, [])
            for f in files:
                if not isinstance(f, str) or f == source_path:
                    continue
                dst_ids = path_to_ids.get(f, [])
                for s in src_ids:
                    for d in dst_ids:
                        pairs.add((s, d))
            break

    return list(pairs)


def _validate_no_symbol_overlap(changes: Sequence[FileChange]) -> None:
    """Raise ValueError if two changes declare ownership of the same symbol in the same file.

    Ownership means the symbol's file is one of the change's own paths. A symbol
    referenced from a different file (cross-file spread tracking) is not an ownership
    declaration and does not trigger the check.

    Two distinct changes claiming to implement the same symbol in a file they both own is
    a genuine data error — duplicate work declared.
    """
    owned_paths = {c.id: set(c.all_paths()) for c in changes}
    path_symbol_to_ids: dict[tuple[str, str], list[str]] = {}
    for c in changes:
        for sr in c.symbols:
            if sr.file not in owned_paths[c.id]:
                continue
            key = (sr.file, sr.name)
            path_symbol_to_ids.setdefault(key, []).append(c.id)
    for (path, symbol), ids in path_symbol_to_ids.items():
        if len(ids) > 1:
            raise ValueError(f"overlapping symbol: {path}::{symbol} declared by {ids}")


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
