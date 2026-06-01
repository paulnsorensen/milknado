# NIH Decisions — Justified Reinventions on Record

Where milknado hand-rolls something a library could provide, and the reinvention
is *deliberate*. Each entry records the trade-off so a future reviewer (or an
`/age` NIH pass) doesn't re-litigate a settled call. A finding filed only to put
the decision on record — not to demand a change — belongs here.

## Hand-rolled iterative Tarjan SCC (#78)

`src/milknado/domains/batching/_tarjan.py` implements strongly-connected-component
detection by hand — `compute_sccs`, exercised through the caller-facing
`contract_sccs` wrapper. The change-graph cycle detection in the batching domain
needs SCC contraction (collapse each cycle to one DAG node) before batch
planning can topologically order the work.

### Why not a library

- **`graphlib` (stdlib, 3.9+) has no SCC.** It provides only `TopologicalSorter`,
  which assumes an acyclic graph — it can *detect* a cycle but won't *contract*
  one. SCC is exactly the piece it's missing.
- **`networkx` is not a dependency and isn't worth adding.** `networkx.strongly_connected_components`
  would do it, but pulling a heavyweight graph library in for one algorithm is a
  bad weight trade. Verified not in `pyproject.toml` and not importable in the
  environment.

### Why iterative, not the textbook recursive Tarjan

Recursive Tarjan recurses once per graph node, so a long dependency chain blows
Python's default recursion limit (~1000 frames). The implementation replaces the
call stack with an explicit `call_stack` of `(node, neighbor_index)` tuples — a
state machine across three helpers (`_strongconnect` drives the outer loop,
`_advance_neighbor` steps the adjacency list, `_finish_node` pops and emits an
SCC when `lowlink[v] == index[v]`).

`tests/batching/test_adversarial.py::TestContractSCCs::test_large_linear_chain`
proves the requirement is real, not hypothetical: a 1000-node linear chain. The
in-code docstring records the measurement — iterative completes in <5 ms; the
equivalent recursive form raises `RecursionError` at the default limit.

### Status

On record, no change. The recursion-limit rationale holds and there is no
drop-in replacement without a new heavyweight dependency. Issue **#78** is
labelled `documented` and kept open as a filterable pointer to this page (per
the 2026-05 MCP steel-thread review, tracker #81). The justification also lives
inline at the top of `_tarjan.py` — keep the two in sync if either changes.
