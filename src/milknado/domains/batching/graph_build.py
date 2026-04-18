from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from milknado.domains.batching.change import FileChange, SymbolRef
from milknado.domains.common.protocols import CrgPort


def build_change_graph(
    changes: Sequence[FileChange],
    crg: CrgPort | None = None,
    extra_edges: Sequence[tuple[str, str]] = (),
) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...], dict[str, tuple[SymbolRef, ...]]]:
    """Return (nodes, edges, symbols_by_node)."""
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
                src_id = path_to_id.get(src_path)
                dst_id = path_to_id.get(dst_path)
                if src_id is not None and dst_id is not None:
                    _add_edge(src_id, dst_id)

    for change in changes:
        for dep_id in change.depends_on:
            if dep_id not in known_ids:
                raise ValueError(f"unknown depends_on id: {dep_id}")
            _add_edge(dep_id, change.id)

    for src, dst in extra_edges:
        if src not in known_ids:
            raise ValueError(f"unknown edge endpoint: {src}")
        if dst not in known_ids:
            raise ValueError(f"unknown edge endpoint: {dst}")
        _add_edge(src, dst)

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
    """Tarjan's algorithm (iterative). Returns (scc_of: node->scc_id, dag_edges).

    SCC ids are stable: min member node id.
    dag_edges are deduped and exclude self-loops.
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
