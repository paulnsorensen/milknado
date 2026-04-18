from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple

from milknado.domains.batching.change import ChangeGraph, FileChange, NewRelationship, SymbolRef
from milknado.domains.common.protocols import CrgPort


class ContractedGraph(NamedTuple):
    """Typed return for contract_sccs: SCC membership map and the resulting DAG edges."""
    scc_of: dict[str, str]
    dag_edges: tuple[tuple[str, str], ...]


def _build_path_to_ids(changes: Sequence[FileChange]) -> dict[str, list[str]]:
    """Map each path to the change ids that touch it."""
    path_to_ids: dict[str, list[str]] = {}
    for c in changes:
        path_to_ids.setdefault(c.path, []).append(c.id)
    return path_to_ids


def build_change_graph(
    changes: Sequence[FileChange],
    crg: CrgPort | None = None,
    new_relationships: Sequence[NewRelationship] = (),
) -> ChangeGraph:
    """Return a ChangeGraph (nodes, edges, symbols_by_node) for the given change set.

    Edges come from three sources: CRG impact radius, explicit ``depends_on``
    declarations, and ``new_relationships``. Only edges between nodes present
    in ``changes`` are kept.

    ``crg=None`` is the valid "no structural graph" path — used in tests and
    when no CRG database has been built yet. The graph is still correct; it
    just lacks CRG-derived precedence edges.
    """
    path_to_ids = _build_path_to_ids(changes)
    id_to_change = {c.id: c for c in changes}
    seen_edges: set[tuple[str, str]] = set()
    _collect_crg_edges(changes, crg, path_to_ids, id_to_change, seen_edges)
    _collect_depends_on_edges(changes, id_to_change, seen_edges)
    _collect_new_relationship_edges(new_relationships, id_to_change, seen_edges)
    return ChangeGraph(
        nodes=tuple(c.id for c in changes),
        edges=tuple(sorted(seen_edges)),
        symbols_by_node={c.id: c.symbols for c in changes},
    )


def _add_edge(edges: set[tuple[str, str]], src: str, dst: str) -> None:
    if src != dst:
        edges.add((src, dst))


def _collect_crg_edges(
    changes: Sequence[FileChange],
    crg: CrgPort | None,
    path_to_ids: dict[str, list[str]],
    id_to_change: dict[str, FileChange],
    out: set[tuple[str, str]],
) -> None:
    if crg is None:
        return
    for change in changes:
        result = crg.get_impact_radius([change.path])
        for src_id, dst_id in _parse_impact_dict(result, change.path, path_to_ids, id_to_change):
            _add_edge(out, src_id, dst_id)


def _collect_depends_on_edges(
    changes: Sequence[FileChange],
    id_to_change: dict[str, FileChange],
    out: set[tuple[str, str]],
) -> None:
    for change in changes:
        for dep_id in change.depends_on:
            if dep_id not in id_to_change:
                raise ValueError(f"unknown depends_on id: {dep_id}")
            _add_edge(out, dep_id, change.id)


def _collect_new_relationship_edges(
    new_relationships: Sequence[NewRelationship],
    id_to_change: dict[str, FileChange],
    out: set[tuple[str, str]],
) -> None:
    for rel in new_relationships:
        if rel.source_change_id not in id_to_change:
            raise ValueError(f"unknown edge endpoint: {rel.source_change_id}")
        if rel.dependant_change_id not in id_to_change:
            raise ValueError(f"unknown edge endpoint: {rel.dependant_change_id}")
        _add_edge(out, rel.source_change_id, rel.dependant_change_id)


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
    if len(ids) == 1 or symbol is None:
        return list(ids)
    matches = [
        cid for cid in ids
        if any(sr.name == symbol and sr.file == path for sr in id_to_change[cid].symbols)
    ]
    if len(matches) == 1:
        return matches
    return list(ids)


def _edge_endpoints(edge: Any) -> tuple[str, str] | None:
    """Extract (src_raw, dst_raw) from an edge record.

    Supports the CRG ``GraphEdge`` dataclass (``source_qualified`` /
    ``target_qualified``) and dict forms using either ``src``/``dst`` or
    ``source``/``target`` keys. Returns ``None`` if the shape is unrecognised.
    """
    if isinstance(edge, dict):
        src = edge.get("src") or edge.get("source")
        dst = edge.get("dst") or edge.get("target")
    else:
        src = getattr(edge, "source_qualified", None)
        dst = getattr(edge, "target_qualified", None)
    if isinstance(src, str) and isinstance(dst, str):
        return src, dst
    return None


def _pairs_from_edge_list(
    edges: list[Any],
    path_to_ids: dict[str, list[str]],
    id_to_change: dict[str, FileChange],
) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for edge in edges:
        endpoints = _edge_endpoints(edge)
        if endpoints is None:
            continue
        raw_src, raw_dst = endpoints
        src_path, src_sym = _parse_qualified(raw_src)
        dst_path, dst_sym = _parse_qualified(raw_dst)
        src_ids = _resolve_ids_for_endpoint(src_path, src_sym, path_to_ids, id_to_change)
        dst_ids = _resolve_ids_for_endpoint(dst_path, dst_sym, path_to_ids, id_to_change)
        for s in src_ids:
            for d in dst_ids:
                pairs.add((s, d))
    return pairs


def _pairs_from_impacted_files(
    files: list[Any],
    source_path: str,
    path_to_ids: dict[str, list[str]],
) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    src_ids = path_to_ids.get(source_path, [])
    for f in files:
        if not isinstance(f, str) or f == source_path:
            continue
        dst_ids = path_to_ids.get(f, [])
        for s in src_ids:
            for d in dst_ids:
                pairs.add((s, d))
    return pairs


def _parse_impact_dict(
    result: dict[str, Any],
    source_path: str,
    path_to_ids: dict[str, list[str]],
    id_to_change: dict[str, FileChange],
) -> list[tuple[str, str]]:
    """Parse CRG's opaque impact result into deduped (src_id, dst_id) pairs.

    Accepts both CRG ``GraphEdge`` dataclass items and dict shapes for the
    ``edges`` key. Falls back to ``impacted_files``/``files`` for path-only
    fan-out when edge data is absent.
    """
    pairs: set[tuple[str, str]] = set()
    edges = result.get("edges")
    if isinstance(edges, list):
        pairs |= _pairs_from_edge_list(edges, path_to_ids, id_to_change)
    for key in ("impacted_files", "files"):
        files = result.get(key)
        if isinstance(files, list):
            pairs |= _pairs_from_impacted_files(files, source_path, path_to_ids)
            break
    return list(pairs)


def validate_no_symbol_overlap(changes: Sequence[FileChange]) -> None:
    """Raise ValueError if two changes declare the same symbol on the same file.

    Ownership means the symbol's file equals the change's own path. A symbol
    whose file belongs to another change (cross-file spread tracking) is not
    an ownership declaration and does not trigger the check.
    """
    path_symbol_to_ids: dict[tuple[str, str], list[str]] = {}
    for c in changes:
        for sr in c.symbols:
            if sr.file != c.path:
                continue
            path_symbol_to_ids.setdefault((sr.file, sr.name), []).append(c.id)
    for (path, symbol), ids in path_symbol_to_ids.items():
        if len(ids) > 1:
            raise ValueError(f"overlapping symbol: {path}::{symbol} declared by {ids}")


def contract_sccs(
    nodes: Sequence[str],
    edges: Sequence[tuple[str, str]],
) -> ContractedGraph:
    """Collapse graph cycles into single SCC nodes using Tarjan's algorithm.

    A Strongly Connected Component (SCC) is a maximal set of nodes where every
    node is reachable from every other node. Cyclic nodes must land in the
    same batch, so we collapse each SCC into one representative node before
    handing the graph to CP-SAT. SCC id is the lexicographically smallest
    member, making it stable across runs. The returned ``dag_edges`` are the
    edges between distinct SCCs (always a DAG). Iterative Tarjan is used to
    avoid Python's recursion limit on large change sets.

    When the graph is acyclic (common case), every node is its own trivial
    SCC — ``scc_of[n] == n`` for all nodes and ``dag_edges`` equals the input
    edges unchanged.
    """
    adj = _build_adjacency(edges)
    scc_of = _compute_sccs(nodes, adj)
    return ContractedGraph(scc_of=scc_of, dag_edges=_extract_dag_edges(edges, scc_of))


def _build_adjacency(edges: Sequence[tuple[str, str]]) -> dict[str, list[str]]:
    adj: dict[str, list[str]] = defaultdict(list)
    for src, dst in edges:
        adj[src].append(dst)
    return adj


def _compute_sccs(nodes: Sequence[str], adj: dict[str, list[str]]) -> dict[str, str]:
    state = _TarjanState()
    for n in nodes:
        if n not in state.index:
            _strongconnect(n, adj, state)
    return state.scc_of


def _extract_dag_edges(
    edges: Sequence[tuple[str, str]], scc_of: dict[str, str]
) -> tuple[tuple[str, str], ...]:
    dag_seen: set[tuple[str, str]] = set()
    for src, dst in edges:
        s, d = scc_of[src], scc_of[dst]
        if s != d:
            dag_seen.add((s, d))
    return tuple(sorted(dag_seen))


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
