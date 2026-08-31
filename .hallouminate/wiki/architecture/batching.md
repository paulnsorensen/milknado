# Batching Domain — Parallel Batch Solver

The batching slice (`src/milknado/domains/batching/`) takes a flat set of file-level
`FileChange`s and partitions them into an ordered list of `Batch`es, where each batch is one
ralph-loop execution context. Batches that don't depend on each other run in parallel; the
solver minimizes a token-budgeted cost so each batch fits one model context window. Entry
point: `plan_batches` in `solver.py`.

## The model (`change.py`)

Pure dataclasses, all frozen:

- `FileChange` — `id`, `path`, `edit_kind`, `symbols` (`SymbolRef` tuple), optional
  `hash_anchors`/`dependencies`, `depends_on` (explicit precedence), `description`.
- `NewRelationship` — a precedence edge (`source → dependant`) with a diagnostic `reason`.
  Batching does not branch on `reason`; it's metadata for tracing.
- `Batch` — `index`, `change_ids`, inter-batch `depends_on` (batch indices), `oversized`.
- `BatchPlan` — `batches`, `spread_report` (`SymbolSpread` per cross-batch symbol),
  `solver_status` (`OPTIMAL`/`FEASIBLE`/`INFEASIBLE`/`UNKNOWN`).
- `ChangeGraph` — raw precedence graph: `nodes`, `edges`, `symbols_by_node`.

## Pipeline (`solver.py::_run_solver`)

1. **Build precedence graph** — `build_change_graph` (`graph_build.py`) collects edges from
   three sources: CRG impact radius (`get_impact_radius` per path, parsed loosely to tolerate
   dataclass or dict edge shapes), explicit `depends_on`, and `new_relationships`. Only edges
   between in-set nodes are kept; `crg=None` is the valid "no structural graph" path.
2. **Contract cycles** — `contract_sccs` collapses each strongly-connected component into one
   representative node (SCC id = lexicographically smallest member, stable across runs). The
   result is a DAG of SCCs (`dag_edges`). Cyclic changes *must* share a batch, so they're
   fused before solving.
3. **Estimate tokens per SCC** — `_tokens_per_scc` sums `estimate_tokens`
   (`weights.py`) across members, de-duplicating changes that share a path.
4. **Partition oversized** — SCCs whose token cost exceeds `budget` are flagged; they get
   their own isolated batch (passthrough) rather than failing the solve.
5. **Build + solve the CP-SAT model** — `build_model` (`_model.py`) then `_two_pass_solve`.
6. **Extract batches** — remap solver batch indices to a dense 0..N, recompute inter-batch
   deps from `dag_edges`, build the spread report.

Pre-solve validation (fail-loud): `_validate_unique_ids` and `validate_no_symbol_overlap`
(two changes owning the same `path::symbol` is a conflict → `ValueError`). Empty input
returns an empty `OPTIMAL` plan.

## Tarjan SCC (`_tarjan.py`)

Cycle detection uses an **iterative** Tarjan implementation, not recursive. Recursive Tarjan
hits Python's ~1000-frame recursion limit on long dependency chains; the iterative version
replaces the call stack with an explicit `(node, neighbor_index)` list and runs a 1000-node
chain in <5ms. NIH note (#78): stdlib `graphlib` offers only topo-sort (no SCC), and
`networkx` is too heavy for one algorithm — so it's hand-rolled. State machine:
`_strongconnect` drives the outer loop, `_advance_neighbor` pushes unvisited neighbors,
`_finish_node` pops and emits an SCC when `lowlink[v] == index[v]`.

## The CP-SAT solver (`_model.py`, `solver.py`)

Depends on **`ortools`** (`from ortools.sat.python import cp_model`) — Google's CP-SAT
constraint solver. The decision variable is `batch_of[scc]` ∈ [0, K-1] (which batch each SCC
lands in). Constraints:

- **Ordering**: for every DAG edge `src→dst`, `batch_of[src] <= batch_of[dst]` — precedence
  is preserved across batches.
- **Budget**: per batch, `Σ tokens_by_scc[s] * in_batch[(s,b)] <= budget`.
- **Oversized isolation**: an oversized SCC is constrained `!=` every other SCC's batch, so
  it sits alone.

**Two-pass lexicographic solve** (`_two_pass_solve`): pass 1 minimizes `total_cost` (half the
time budget); pass 2 pins `cost == cost*` and minimizes total symbol *spread* (how many
batches a shared symbol is split across) — a tie-breaker that keeps related edits together
without sacrificing cost. `MODEL_INVALID` raises (structural bug); `INFEASIBLE`/`UNKNOWN`
return an empty/partial plan with that status. `_worse_status` reports the conservative
status across both passes.

The cost function (`_build_total_cost`) is per-batch: a fixed `batch_size_cost` (ralph
startup overhead, 2000 tokens/batch when non-empty) plus a token cost scaled by a
batch-size-dependent multiplier `(k*12 - 1)*10` — larger batches pay more per token, nudging
the solver toward more, smaller batches. All values are ×100 to stay integer for CP-SAT.

## Startup cost — lazy loading (PR #380)

`ortools` (`cp_model`) is a ~145ms native import and `weights.py`'s tiktoken setup
adds more; eagerly importing `solver.py`/`weights.py` at module load put both on
**every** `milknado` CLI invocation (even `--help`) and every `milknado-mcp`
startup, since `domains/batching/__init__.py` used to import `plan_batches`
top-level. Fix: `DUMB_ZONE_BUDGET` moved into the dependency-free `change.py`
module, and the crust exposes `plan_batches`/`estimate_tokens` via a PEP 562
`__getattr__` in `__init__.py` — ortools and tiktoken now load only on first
real use (first solve). Measured via `python -X importtime`: the batching crust
dropped from ~163ms to ~2.7ms in the CLI import graph, and `ortools` disappeared
from the MCP server's import graph entirely. A regression test asserts neither
`milknado.cli` nor `milknado.mcp.server` imports any `ortools` module at
startup.

This closed the ortools slice of MCP/CLI cold start but not the rest: `fastmcp`
(~225ms) and a transitive `pandas` import (~86ms, via `mcp`'s `key_value`
adapter) remain, plus ortools itself is still a hard dependency rather than an
optional extra — both tracked as open follow-ups in #381, not yet decided.

## Weights (`weights.py`)

`estimate_tokens` uses path-level costs:

1. **Whole-file tiktoken** for an existing modified file, multiplied by `HEADROOM` (1.25).
2. **Line heuristic** — `NEW_FILE_LINES[ext] * TOKENS_PER_LINE[ext] * HEADROOM` for adds
   and missing modified files. `delete`/`rename` use a flat cost (80/120).

Within an SCC, changes sharing one path are counted once so multiple symbol-scoped manifest
entries for the same file do not inflate the batch cost.

tiktoken uses the `cl100k_base` encoder, cached via a vendored blob
(`src/milknado/_vendor/tiktoken-cache/`) so it works offline; encoder is `lru_cache`d.

## Budget (`DUMB_ZONE_BUDGET = 120_000`)

The default per-batch ceiling. One batch is one implementation context window; past ~120K
the model degrades ("dumb zone") rather than improving — so the budget caps a batch below
that threshold instead of maximizing fill. `SolverConfig.budget` defaults to 150K but the
public `plan_batches` default and the planner bridge both pass `DUMB_ZONE_BUDGET`.
