"""CP-SAT batch planner.

Time limit: 10s. BIG=10_000. ALPHA=1.
Timeout status maps to FEASIBLE if a solution was found, else UNKNOWN.
No solution found -> UNKNOWN.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from ortools.sat.python import cp_model

from milknado.domains.batching.change import BatchPlan, FileChange, SolverStatus, SymbolRef
from milknado.domains.batching.graph_build import build_change_graph, contract_sccs, symbols_by_scc
from milknado.domains.batching.weights import estimate_tokens
from milknado.domains.common.protocols import CrgPort

# BIG makes batch-count minimisation the primary objective: any gain of one fewer
# batch (saving BIG in objective) outweighs any improvement in symbol spread.
BIG = 10_000
# ALPHA weights the secondary objective (symbol spread). Raise it to push the
# solver harder to keep related symbols in the same batch; lower it to let the
# solver trade spread for fewer batches more freely.
ALPHA = 1

STATUS_OPTIMAL: SolverStatus = "OPTIMAL"
STATUS_FEASIBLE: SolverStatus = "FEASIBLE"
STATUS_INFEASIBLE: SolverStatus = "INFEASIBLE"
STATUS_UNKNOWN: SolverStatus = "UNKNOWN"


def _status_name(status: object) -> SolverStatus:
    if status == cp_model.OPTIMAL:
        return STATUS_OPTIMAL
    if status == cp_model.FEASIBLE:
        return STATUS_FEASIBLE
    if status == cp_model.INFEASIBLE:
        return STATUS_INFEASIBLE
    if status == cp_model.MODEL_INVALID:
        raise RuntimeError(
            "CP-SAT reported MODEL_INVALID \u2014 solver model has a structural bug"
        )
    return STATUS_UNKNOWN


def _build_model(
    sccs: list[str],
    dag_edges: tuple[tuple[str, str], ...],
    tokens_by_scc: dict[str, int],
    sym_by_scc: dict[str, tuple[SymbolRef, ...]],
    budget: int,
) -> tuple[cp_model.CpModel, dict[str, cp_model.IntVar], dict[str, cp_model.IntVar]]:
    """Build the CP-SAT optimisation model that assigns SCCs to numbered batches.

    Decision variables
    ------------------
    ``batch_of[s]``  — integer in [0, K-1], the batch index assigned to SCC s.
    ``in_batch[(s,b)]`` — boolean indicator: 1 iff batch_of[s] == b. Needed
       because CP-SAT cannot express linear sums over conditional integer vars
       directly; the boolean linearisation lets us write the budget constraint.

    Constraints
    -----------
    1. **Budget** (one per batch index b):
         sum(tokens[s] * in_batch[(s,b)]  for all s) <= budget
       Ensures no single batch exceeds the token budget. Batches are allowed to
       be empty (all indicators 0), which is how the solver creates fewer than K
       batches in practice.

    2. **Ordering** (one per DAG edge (src, dst)):
         batch_of[src] <= batch_of[dst]
       Encodes the precedence requirement: every dependency must land in a batch
       that is executed *before or at the same time as* its dependant. Because
       batches are executed sequentially (batch 0 first), batch index order is
       execution order.

    3. **Spread per symbol** (see ``_build_spread_vars``):
       For each symbol that appears in more than one SCC, a spread variable
       tracks (max_batch - min_batch) across the SCCs that reference it. This
       penalises splitting a cross-file symbol across many batches, which would
       require a reviewer to context-switch across reviews.

    Objective
    ---------
    Minimise:  max_batch_idx * BIG  +  ALPHA * sum(spread_vars)

    The first term is the primary objective: use the fewest batches possible.
    BIG is large enough (10 000) that any reduction of 1 in the batch count
    outweighs any possible improvement in the spread term, no matter how large
    the graph. The second term is a tie-breaker that prefers assignments where
    symbols that appear in multiple SCCs are grouped into adjacent batches.

    Returns
    -------
    model      — the fully-constrained CpModel, ready to solve
    batch_of   — SCC id → integer decision variable
    spread_vars — symbol key → spread integer variable (for reporting)
    """
    model = cp_model.CpModel()
    K = len(sccs)
    batch_of = {s: model.new_int_var(0, K - 1, f"b_{s}") for s in sccs}
    in_batch: dict[tuple[str, int], cp_model.IntVar] = {}
    for s in sccs:
        for b in range(K):
            in_batch[(s, b)] = model.new_bool_var(f"ib_{s}_{b}")
            model.add(batch_of[s] == b).only_enforce_if(in_batch[(s, b)])
            model.add(batch_of[s] != b).only_enforce_if(in_batch[(s, b)].negated())

    for b in range(K):
        model.add(
            sum(tokens_by_scc[s] * in_batch[(s, b)] for s in sccs) <= budget
        )

    for src, dst in dag_edges:
        model.add(batch_of[src] <= batch_of[dst])

    spread_vars = _build_spread_vars(model, batch_of, sym_by_scc, sccs, K)

    max_batch_idx = model.new_int_var(0, K - 1, "max_batch_idx")
    model.add_max_equality(max_batch_idx, list(batch_of.values()))
    model.minimize(max_batch_idx * BIG + ALPHA * sum(spread_vars.values()))
    return model, batch_of, spread_vars


def _build_spread_vars(
    model: cp_model.CpModel,
    batch_of: dict[str, cp_model.IntVar],
    sym_by_scc: dict[str, tuple[SymbolRef, ...]],
    sccs: list[str],
    K: int,
) -> dict[str, cp_model.IntVar]:
    """Build spread = max(batch) - min(batch) per multi-SCC symbol."""
    sym_to_sccs: dict[str, list[str]] = {}
    for s in sccs:
        for sym in sym_by_scc.get(s, ()):
            key = f"{sym.file}:{sym.name}"
            sym_to_sccs.setdefault(key, []).append(s)

    spread_vars: dict[str, cp_model.IntVar] = {}
    for key, scc_list in sym_to_sccs.items():
        if len(scc_list) < 2:
            continue
        b_vars = [batch_of[s] for s in scc_list]
        lo = model.new_int_var(0, K - 1, f"lo_{key}")
        hi = model.new_int_var(0, K - 1, f"hi_{key}")
        model.add_min_equality(lo, b_vars)
        model.add_max_equality(hi, b_vars)
        spread = model.new_int_var(0, K - 1, f"sp_{key}")
        model.add(spread == hi - lo)
        spread_vars[key] = spread
    return spread_vars


def _extract_solution(
    solver: cp_model.CpSolver,
    sccs: list[str],
    batch_of: dict[str, cp_model.IntVar],
    scc_members: dict[str, list[str]],
    input_order: dict[str, int],
) -> tuple[tuple[str, ...], ...]:
    raw: dict[int, list[str]] = {}
    for s in sccs:
        b = solver.value(batch_of[s])
        raw.setdefault(b, []).extend(scc_members[s])
    batches = []
    for b in sorted(raw):
        members = sorted(raw[b], key=lambda x: input_order.get(x, 0))
        batches.append(tuple(members))
    # Compact: remap batch indices to 0..N-1 (already done by sorting)
    return tuple(batches)


def _validate_unique_ids(changes: Sequence[FileChange]) -> None:
    seen: set[str] = set()
    for c in changes:
        if c.id in seen:
            raise ValueError(f"duplicate change id: {c.id}")
        seen.add(c.id)


def _group_sccs(
    nodes: tuple[str, ...], scc_of: dict[str, str]
) -> tuple[list[str], dict[str, list[str]]]:
    scc_members: dict[str, list[str]] = {}
    for node_id in nodes:
        scc_members.setdefault(scc_of[node_id], []).append(node_id)
    return list(scc_members), scc_members


def _tokens_per_scc(
    changes: Sequence[FileChange],
    scc_members: dict[str, list[str]],
    sccs: list[str],
    root: Path,
) -> dict[str, int]:
    tokens_by_change = {c.id: estimate_tokens(c, root) for c in changes}
    return {s: sum(tokens_by_change[cid] for cid in scc_members[s]) for s in sccs}


def plan_batches(
    changes: Sequence[FileChange],
    budget: int = 70_000,
    *,
    crg: CrgPort | None = None,
    extra_edges: Sequence[tuple[str, str]] = (),
    time_limit_s: float = 10.0,
    root: Path | None = None,
) -> BatchPlan:
    """Compute token-budgeted, precedence-respecting batches."""
    if time_limit_s < 0:
        raise ValueError(f"time_limit_s must be non-negative, got {time_limit_s}")
    if not changes:
        return BatchPlan(batches=(), spread_report={}, solver_status=STATUS_OPTIMAL)

    _validate_unique_ids(changes)
    if root is None:
        root = Path.cwd()

    nodes, edges, sym_by_node = build_change_graph(changes, crg, extra_edges)
    scc_of, dag_edges = contract_sccs(nodes, edges)
    sccs, scc_members = _group_sccs(nodes, scc_of)
    tokens_by_scc = _tokens_per_scc(changes, scc_members, sccs, root)

    if any(t > budget for t in tokens_by_scc.values()):
        return BatchPlan(batches=(), spread_report={}, solver_status=STATUS_INFEASIBLE)

    sym_by_scc_map = symbols_by_scc(scc_of, sym_by_node)
    model, batch_of, spread_vars = _build_model(
        sccs, dag_edges, tokens_by_scc, sym_by_scc_map, budget
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    status_name = _status_name(solver.solve(model))
    if status_name in (STATUS_INFEASIBLE, STATUS_UNKNOWN):
        return BatchPlan(batches=(), spread_report={}, solver_status=status_name)

    input_order = {c.id: i for i, c in enumerate(changes)}
    batches = _extract_solution(solver, sccs, batch_of, scc_members, input_order)
    spread_report = {key: int(solver.value(var)) for key, var in spread_vars.items()}
    return BatchPlan(batches=batches, spread_report=spread_report, solver_status=status_name)
