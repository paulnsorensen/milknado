# Batching Domain

Token-budgeted, precedence-respecting batch planner for Milknado. Given a list
of file changes and a token budget, it returns a `BatchPlan` whose batches form
a DAG of ralphify iterations where each batch fits within the budget and all
declared dependencies are satisfied.

## Why This Exists

When Milknado plans a Mikado execution graph, it needs to submit changes to a
code-review-graph (CRG) context window that has a finite token budget. Naively
dumping all changes at once may exceed the budget; naively splitting them
arbitrarily may violate dependency order. This domain solves the scheduling
problem optimally (or near-optimally within a time limit) using a CP-SAT
constraint solver.

## Public API

```python
from milknado.domains.batching import (
    plan_batches,
    FileChange, SymbolRef,
    NewRelationship,
    Batch, SymbolSpread, BatchPlan,
)
```

Key types (all `frozen=True` dataclasses):

```
FileChange
  id: str                           — stable caller-assigned identifier
  path: str                         — repo-relative file path
  edit_kind: EditKind               — "add" | "modify" | "delete" | "rename"
  symbols: tuple[SymbolRef, ...]    — optional: symbols this change touches
  depends_on: tuple[str, ...]       — explicit predecessor change ids

SymbolRef
  name: str   — symbol name (function, class, etc.)
  file: str   — file the symbol lives in

NewRelationship
  source_change_id: str
  dependant_change_id: str
  reason: RelationshipReason        — "new_file" | "new_import" | "new_call" | "new_type_use"

Batch
  index: int
  change_ids: tuple[str, ...]
  depends_on: tuple[int, ...]       — indices of prerequisite batches (Mikado DAG edges)
  oversized: bool                   — True when this batch exceeds the token budget

SymbolSpread
  symbol: SymbolRef
  spread: int                       — max(batch_index) − min(batch_index) across batches touching this symbol

BatchPlan
  batches: tuple[Batch, ...]
  spread_report: tuple[SymbolSpread, ...]
  solver_status: SolverStatus       — "OPTIMAL" | "FEASIBLE" | "INFEASIBLE" | "UNKNOWN"
```

## What a Batch Is

One batch = one ralphify iteration = one LLM context window.

- **Intra-batch ordering** is handled in-pass: the LLM resolves the ordering of
  changes within the same batch without any sequencing guarantee from the planner.
- **Inter-batch edges** (`Batch.depends_on`) force sequencing: ralph will not
  start batch N until every batch listed in `depends_on` has completed.
- **Sibling batches** (batches with no dependency relationship) fan out: ralph
  runs them concurrently as independent ralphify iterations.

## Degenerate Singleton Case (Q4)

When the change graph is acyclic, Tarjan's SCC algorithm produces one SCC per
node (every node is its own SCC of size 1). In that case:

- `dag_edges` equals the original input edges — no contraction occurs.
- The solver treats each change independently; no two changes are forced into
  the same batch by structural reasons alone.
- Budget constraints and the lexicographic objective still determine which
  changes share a batch.

This is the common case for most Mikado graphs where no circular dependencies
exist between in-flight changes.

## Concurrent Execution (Q5)

The `Batch.depends_on` field encodes the Mikado DAG that ralph uses to fan out
work. Independent branches run concurrently; dependent chains run sequentially.

**Worked example — chain plus detached node:**

```
Input changes:
  a (50 tokens)  →  b (30 tokens)
  b (30 tokens)  →  c (40 tokens)
  g (20 tokens)      (no edges)

Budget: 100 tokens

Output BatchPlan:
  batches = [
    Batch(index=0, change_ids=("a", "g"), depends_on=(),   oversized=False),
    Batch(index=1, change_ids=("b",),     depends_on=(0,), oversized=False),
    Batch(index=2, change_ids=("c",),     depends_on=(1,), oversized=False),
  ]
  solver_status = "OPTIMAL"
```

Ralph fans out batch 0 (a and g run concurrently in one ralphify iteration),
waits for batch 0 to complete, then runs batch 1, then batch 2.

`g` has no predecessors so it lands in the earliest available batch alongside
`a` — the solver co-batches them because their combined token cost (70) fits
within the 100-token budget.

## Oversized SCCs (Q2)

When a single SCC's estimated token count exceeds the budget, splitting it is
impossible (the nodes form a cycle and must remain together). Rather than
returning `INFEASIBLE`, the planner passes it through:

- The oversized SCC participates in CP-SAT as a batch-index decision variable
  like any other SCC — it is not pre-assigned a fixed index.
- Mutual-exclusion constraints (`batch_of[oversized] != batch_of[other]`) force
  each oversized SCC into its own batch, alone.
- The budget constraint is waived for that batch.
- The batch is marked `oversized=True`.
- Ordering constraints into and out of oversized batches are enforced by the
  same CP-SAT precedence relations used everywhere else.

Ralph treats oversized batches as **linear ralph loops** — one symbol-level
task at a time inside that batch — rather than a single LLM context window.

**Worked example — oversized passthrough:**

```
Input: one FileChange("big_refactor") touching a 120_000-token refactor.
Budget: 70_000.

Output BatchPlan:
  batches = [
    Batch(index=0, change_ids=("big_refactor",), depends_on=(), oversized=True),
  ]
  solver_status = "OPTIMAL"
```

`INFEASIBLE` is never returned for a single oversized SCC. It can only be
returned when the solver cannot find any valid assignment at all (which cannot
happen when there is exactly one SCC).

## New Relationships (Q3)

`new_relationships: tuple[NewRelationship, ...]` is the typed replacement for
the old `extra_edges` parameter of `plan_batches` and `build_change_graph`.

**When to use it**: CRG is authoritative for existing code. Pass
`new_relationships` only for changes that introduce edges the CRG graph has not
yet indexed — typically because the code being added or modified does not exist
yet in the CRG knowledge graph.

Valid `reason` values:

| Reason | When to use |
|--------|-------------|
| `"new_file"` | A new file is being created; it will import or be imported by another change |
| `"new_import"` | An existing file gains a new import statement that CRG hasn't seen |
| `"new_call"` | A new function call is being introduced between two changed files |
| `"new_type_use"` | A new type reference is being introduced between two changed files |

Each `NewRelationship` declares that `source_change_id` must be batched before
(or in the same batch as) `dependant_change_id`.

## Solver Objective

The solver uses a **lexicographic two-pass** strategy. BIG constants and ALPHA
weights are not used.

**Pass 1 — minimise batch count:**
- Objective: `minimise max_batch_idx`
- Time budget: `time_limit_s / 2`
- Captures `K* = solver.value(max_batch_idx)` from the solution.

**Pass 2 — minimise symbol spread:**
- Additional constraint: `max_batch_idx == K*`  (batch count is now fixed)
- Objective: `minimise sum(spread_vars.values())`
- Time budget: remaining half of `time_limit_s`

**Status resolution:**
- If pass 1 is `INFEASIBLE` → return `INFEASIBLE`.
- If pass 1 is `UNKNOWN` → return `UNKNOWN`.
- Otherwise: `solver_status` = worse of the two passes' statuses. Pass 2 can
  only relax (it adds a constraint and changes the objective), so it may
  downgrade `OPTIMAL` → `FEASIBLE` but never upgrade.

This guarantees that reducing the batch count by 1 always takes strict
precedence over any improvement in symbol spread, regardless of graph size.

| Status | Meaning |
|--------|---------|
| `OPTIMAL` | Provably optimal solution found |
| `FEASIBLE` | Valid solution found but not proven optimal (hit time limit) |
| `INFEASIBLE` | No valid assignment exists (cannot occur for a single oversized SCC) |
| `UNKNOWN` | Time limit hit before any solution was found |

## What Spread Means

For each `SymbolSpread(symbol, spread)` entry in `BatchPlan.spread_report`:

```
spread = max(batch_index) − min(batch_index)
```

across all batches that contain a change touching `symbol`.

- `spread == 0`: every change touching this symbol landed in the same batch.
  This is ideal — the reviewer sees all related edits in one ralphify iteration.
- Higher values flag symbols that are fragmented across the plan; the reviewer
  must inspect multiple batches to understand the full impact on that symbol.

Only symbols that appear in **more than one SCC** are reported. Symbols touched
by exactly one SCC always have `spread == 0` and are omitted.

## Architecture

The pipeline has three stages:

```
FileChange list
      │
      ▼
[graph_build]  →  (nodes, edges, symbols_by_node)
      │
      ▼
[contract_sccs]  →  (scc_of, dag_edges, sym_by_scc)
      │
      ▼
[solver / CP-SAT]  →  BatchPlan
```

### Stage 1: Graph Build (`graph_build.py`)

`build_change_graph` constructs a directed graph where nodes are change ids and
edges represent "must come before" relationships. Edges come from three sources:

1. **CRG impact radius** — if a `CrgPort` is provided, each changed file is
   queried for structural dependants within the CRG knowledge graph. Only edges
   where both endpoints are already in the change set are kept. CRG edges carry
   symbol-level qualified names (`"path::symbol"`); when multiple `FileChange`s
   share a path, edge attribution routes to the specific change whose declared
   symbols match the qualified name.

   When `crg=None` (tests, or no graph built), this step is skipped.

2. **Explicit `depends_on`** — each `FileChange` may declare ids it must
   follow. Unknown ids raise immediately.

3. **`new_relationships`** — typed `NewRelationship` records for edges CRG
   hasn't yet indexed (new files, new imports, new calls, new type uses).

All three sources are unioned into a single edge set; duplicates are discarded.

**`_validate_no_symbol_overlap`** enforces that two `FileChange`s on the same
path may not declare the same symbol name. That is a data error (duplicate work
declared) and raises `ValueError`.

### Stage 2: SCC Contraction (`graph_build.py`)

`contract_sccs` runs Tarjan's iterative algorithm to find all Strongly Connected
Components. Each SCC is collapsed to its lexicographically smallest member id.
The result is a DAG — CP-SAT cannot handle cycles.

**Symbol union**: `symbols_by_scc` unions symbol refs from all members of each
SCC so the solver can track spread at the SCC level.

### Stage 3: CP-SAT Solver (`solver.py`)

`plan_batches` passes the contracted graph to Google OR-Tools' CP-SAT solver.

**Decision variables:**
- `batch_of[s]` — integer in `[0, K-1]`: batch index assigned to SCC `s`.
- `in_batch[(s, b)]` — boolean indicator: `1` iff `batch_of[s] == b`.

**Constraints:**

Budget (one per batch index):
```
sum(tokens[s] * in_batch[(s, b)]  for all s) <= budget
```

Ordering (one per DAG edge):
```
batch_of[src] <= batch_of[dst]
```

**Batch-level DAG extraction** (post-solve):
For each original change-graph edge `(src_id, dst_id)`, the solver maps both
endpoints to their batch indices. Cross-batch edges are deduplicated and stored
in `Batch.depends_on`.

### Token Estimation (`weights.py`)

| Edit kind | Method |
|-----------|--------|
| `modify` | `tiktoken` encodes the file on disk, result × 1.25 headroom |
| `add` | `NEW_FILE_LINES[ext] × TOKENS_PER_LINE[ext] × 1.25` |
| `delete` | flat 80 tokens |
| `rename` | flat 120 tokens |

For `modify`, if the file cannot be read (`OSError`), the estimate falls back to
the `add` heuristic. The 1.25 factor accounts for context tokens the model will
also process.

## Invariants and Guarantees

- Change ids must be unique; duplicates raise `ValueError`.
- `depends_on` ids must refer to ids in the same input set; unknown ids raise.
- `new_relationships` endpoints must also be in the input set; unknown endpoints raise.
- Two `FileChange`s on the same path may not declare the same symbol name; raises `ValueError`.
- If no changes are given, returns an empty `BatchPlan` with status `OPTIMAL`.
- Batch indices in the output are compact `0..N-1`; no empty batches in the result.
- Within a batch, changes are ordered by their position in the input list.
- `spread_report` only contains symbols appearing in more than one SCC.
