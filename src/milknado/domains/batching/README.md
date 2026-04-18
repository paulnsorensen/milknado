# Batching Domain

Token-budgeted, precedence-respecting batch planner for Milknado. Given a list
of file changes and a token budget, it returns an ordered sequence of batches
where each batch fits within the budget and all declared dependencies are
satisfied.

## Why This Exists

When Milknado plans a Mikado execution graph, it needs to submit changes to a
code-review-graph (CRG) context window that has a finite token budget. Naively
dumping all changes at once may exceed the budget; naively splitting them
arbitrarily may violate dependency order. This domain solves the scheduling
problem optimally (or near-optimally within a time limit) using a CP-SAT
constraint solver.

## Domain Model

```
FileChange
  id: str              — stable identifier for this change (caller-assigned)
  path: str            — repo-relative file path
  edit_kind: EditKind  — "add" | "modify" | "delete" | "rename"
  symbols: tuple[SymbolRef, ...]   — optional: symbols this change touches
  depends_on: tuple[str, ...]      — explicit predecessors (change ids)

SymbolRef
  name: str   — symbol name (function, class, etc.)
  file: str   — file the symbol lives in

BatchPlan
  batches: tuple[tuple[str, ...], ...]  — ordered batches, each a tuple of change ids
  spread_report: dict[str, int]         — symbol key → batch spread (0 = co-located)
  solver_status: SolverStatus           — "OPTIMAL" | "FEASIBLE" | "INFEASIBLE" | "UNKNOWN"
```

A `BatchPlan` is INFEASIBLE when any single SCC (see below) already exceeds the
token budget on its own — splitting it further is impossible. UNKNOWN means the
solver hit the time limit without finding a solution.

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
   where *both* endpoints are already in the change set are kept. This
   automatically captures implicit coupling: if `auth.py` imports `tokens.py`
   and both are being changed, CRG will surface that edge.

   When `crg=None` (tests, or no graph built), this step is skipped. The
   result is still valid — it just relies on explicit edges only.

2. **Explicit `depends_on`** — each `FileChange` may declare ids it must
   follow. These are validated; referencing an unknown id raises immediately.

3. **`extra_edges`** — caller-supplied (src_id, dst_id) pairs for cases where
   neither CRG nor `depends_on` captures the relationship.

All three sources are unioned into a single edge set; duplicates are discarded.

### Stage 2: SCC Contraction (`graph_build.py`)

`contract_sccs` runs Tarjan's iterative algorithm to find all Strongly Connected
Components (SCCs) in the change graph.

**What is an SCC?** A maximal set of nodes where every node is reachable from
every other via directed edges. In a change graph this is a *cycle*: two files
that mutually import each other, or a ring of `depends_on` declarations. Because
cyclic nodes can never be ordered relative to each other, they *must* land in the
same batch.

**Contraction**: each SCC is collapsed into a single representative node (the
lexicographically smallest member id). The result is always a DAG — CP-SAT
cannot handle cycles, and now it does not need to. Edges between collapsed SCCs
form the `dag_edges` that the solver will enforce as ordering constraints.

**Symbol union**: `symbols_by_scc` unions the symbol refs from all members of
each SCC into a single set, so the solver can track symbol spread at the SCC
level.

Example:

```
FileChanges: A, B, C, D
Edges: A→B, B→A, B→C, C→D   (A and B form a cycle)

SCCs: {A,B} (scc_id="A"), {C} (scc_id="C"), {D} (scc_id="D")
dag_edges: ("A","C"), ("C","D")   ← A/B cycle contracted to "A"
```

### Stage 3: CP-SAT Solver (`solver.py`)

`plan_batches` passes the contracted graph to Google OR-Tools' CP-SAT solver.

#### Decision Variables

- `batch_of[s]` — integer in `[0, K-1]` where K = number of SCCs. Represents
  the batch index assigned to SCC `s`. Batches are numbered 0, 1, 2, … and
  executed in that order.

- `in_batch[(s, b)]` — boolean indicator: `1` iff `batch_of[s] == b`. Required
  because CP-SAT cannot directly express `sum(tokens[s] for s if batch_of[s]==b)`
  — the boolean linearisation makes this a standard linear constraint.

#### Constraints

**Budget** (one per batch index):
```
sum(tokens[s] * in_batch[(s, b)]  for all s) <= budget
```
Ensures no single batch exceeds the token budget. Batches may be empty.

**Ordering** (one per DAG edge):
```
batch_of[src] <= batch_of[dst]
```
Encodes the dependency: `src` must land in a batch whose index is ≤ `dst`'s
batch. Because batches execute in index order, this guarantees `src` is reviewed
before `dst`. Equal indices mean they land in the *same* batch — also valid.

#### Objective

```
Minimise:  max_batch_idx * BIG  +  ALPHA * sum(spread_vars)
```

**Primary term** (`max_batch_idx * BIG`): use the fewest batches. `BIG = 10_000`
ensures that reducing the batch count by 1 always outweighs any possible
improvement in the secondary term, regardless of graph size. Think of BIG as
"batch count is infinitely more important than spread."

**Secondary term** (`ALPHA * sum(spread_vars)`): minimise symbol spread. For
each symbol that appears in more than one SCC, the spread is:
```
spread[sym] = max(batch_of[s] for s touching sym) - min(batch_of[s] for s touching sym)
```
A spread of 0 means all changes touching a symbol are co-located in one batch —
ideal for reviewers. `ALPHA = 1` makes this a tie-breaker; raise it to push the
solver harder to co-locate symbols at the cost of potentially using more batches.

#### Time Limit and Status

The solver runs for at most `time_limit_s` seconds (default 10). The returned
`solver_status` reflects the outcome:

| Status | Meaning |
|--------|---------|
| `OPTIMAL` | Provably optimal solution found |
| `FEASIBLE` | Valid solution found but not proven optimal (hit time limit) |
| `INFEASIBLE` | No valid assignment exists (a single SCC exceeds the budget) |
| `UNKNOWN` | Time limit hit before any solution was found |

`INFEASIBLE` and `UNKNOWN` both return an empty `BatchPlan`. The caller must
decide how to handle these (raise the budget, split the change set, or abort).

### Token Estimation (`weights.py`)

Each `FileChange` gets a token estimate before entering the solver:

| Edit kind | Method |
|-----------|--------|
| `modify` | `tiktoken` encodes the file on disk, result × 1.25 headroom |
| `add` | `NEW_FILE_LINES[ext] × TOKENS_PER_LINE[ext] × 1.25` |
| `delete` | flat 80 tokens |
| `rename` | flat 120 tokens |

For `modify`, if the file cannot be read (any `OSError`), the estimate falls
back to the same heuristic as `add`. The 1.25 headroom factor accounts for
context tokens (comments, surrounding code) that the model will also process.

Language-specific multipliers in `TOKENS_PER_LINE` and `NEW_FILE_LINES` reflect
real token density differences: Rust/Java are more verbose than Markdown.

## Public API

```python
from milknado.domains.batching import plan_batches, FileChange, BatchPlan, SymbolRef

plan: BatchPlan = plan_batches(
    changes=[
        FileChange(id="auth", path="src/auth.py", edit_kind="modify"),
        FileChange(id="tokens", path="src/tokens.py", edit_kind="modify",
                   depends_on=("auth",)),
    ],
    budget=70_000,
    crg=None,          # omit for no structural edge discovery
    root=Path("."),    # repo root for tiktoken reads
    time_limit_s=10.0,
)

for i, batch in enumerate(plan.batches):
    print(f"Batch {i}: {batch}")
```

## Invariants and Guarantees

- Change ids must be unique across the input; duplicates raise `ValueError`.
- `depends_on` ids must refer to ids in the same input set; unknown ids raise.
- `extra_edges` endpoints must also be in the input set; unknown endpoints raise.
- If no changes are given, returns an empty `BatchPlan` with status `OPTIMAL`.
- Batch indices in the output are compact (0..N-1); no empty batches in the result.
- Within a batch, changes are ordered by their position in the input list.
- The `spread_report` only contains symbols that appear in *more than one SCC*
  (symbols in a single SCC always have spread 0 and are omitted).
