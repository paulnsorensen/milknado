"""Iterative Tarjan SCC algorithm for change-graph cycle detection.

Recursive Tarjan hits Python's default recursion limit (~1000 frames) on
graphs with long dependency chains. This implementation replaces the call
stack with an explicit ``call_stack`` list of ``(node, neighbor_index)``
tuples, where ``neighbor_index`` tracks how far we've advanced through each
node's adjacency list between iterations of the outer while loop.

State machine shape:
- ``_strongconnect`` drives the outer loop, initialising a node on first
  visit (``neighbor_index == 0``) then delegating to the two helpers.
- ``_advance_neighbor`` steps forward through the adjacency list; if it
  finds an unvisited neighbor it pushes it onto ``call_stack`` and returns
  ``True`` (the outer loop will re-enter for the new node on next iteration).
- ``_finish_node`` fires when all neighbors are exhausted; it pops the node,
  propagates the lowlink to its parent, and emits an SCC when
  ``lowlink[v] == index[v]``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass
class _TarjanState:
    index_counter: int = 0
    stack: list[str] = field(default_factory=list)
    on_stack: set[str] = field(default_factory=set)
    index: dict[str, int] = field(default_factory=dict)
    lowlink: dict[str, int] = field(default_factory=dict)
    scc_of: dict[str, str] = field(default_factory=dict)


def compute_sccs(nodes: Sequence[str], adj: dict[str, list[str]]) -> dict[str, str]:
    state = _TarjanState()
    for n in nodes:
        if n not in state.index:
            _strongconnect(n, adj, state)
    return state.scc_of


def _strongconnect(start: str, adj: dict[str, list[str]], s: _TarjanState) -> None:
    """Drive the iterative DFS from ``start``, processing one node per iteration."""
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
    """Scan neighbors of ``v`` starting at index ``i``; push the first unvisited one.

    Returns ``True`` when a new node was pushed (caller should loop again),
    ``False`` when all neighbors are exhausted (caller should call _finish_node).
    """
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
    """Pop ``v`` from call_stack, propagate lowlink to parent, and emit SCC if ``v`` is a root."""
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
