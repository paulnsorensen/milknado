"""CP-SAT batch planner — lexicographic two-pass solver with oversized passthrough."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from ortools.sat.python import cp_model

from milknado.domains.batching.change import (
    Batch,
    BatchPlan,
    FileChange,
    NewRelationship,
    SolverStatus,
    SymbolRef,
    SymbolSpread,
)
from milknado.domains.batching.graph_build import (
    _validate_no_symbol_overlap,
    build_change_graph,
    contract_sccs,
    symbols_by_scc,
)
from milknado.domains.batching.weights import estimate_tokens
from milknado.domains.common.protocols import CrgPort

STATUS_OPTIMAL: SolverStatus = "OPTIMAL"
STATUS_FEASIBLE: SolverStatus = "FEASIBLE"
STATUS_INFEASIBLE: SolverStatus = "INFEASIBLE"
STATUS_UNKNOWN: SolverStatus = "UNKNOWN"

_STATUS_RANK = {STATUS_OPTIMAL: 3, STATUS_FEASIBLE: 2, STATUS_UNKNOWN: 1, STATUS_INFEASIBLE: 0}


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


def _worse_status(a: SolverStatus, b: SolverStatus) -> SolverStatus:
    return a if _STATUS_RANK[a] <= _STATUS_RANK[b] else b


def _add_ordering_constraints(
    model: cp_model.CpModel,
    dag_edges: tuple[tuple[str, str], ...],
    batch_of: dict[str, cp_model.IntVar],
    fixed_batch_of: dict[str, int],
) -> None:
    for src, dst in dag_edges:
        sf, df = fixed_batch_of.get(src), fixed_batch_of.get(dst)
        if sf is None and df is None:
            model.add(batch_of[src] <= batch_of[dst])
        elif sf is not None and df is None:
            model.add(sf <= batch_of[dst])
        elif sf is None and df is not None:
            model.add(batch_of[src] <= df)
        # both fixed: ordering guaranteed by index assignment


def _build_model(
    sccs: list[str],
    dag_edges: tuple[tuple[str, str], ...],
    tokens_by_scc: dict[str, int],
    sym_by_scc: dict[str, tuple[SymbolRef, ...]],
    budget: int,
    oversized_indices: set[int],
    fixed_batch_of: dict[str, int],
) -> tuple[cp_model.CpModel, dict[str, cp_model.IntVar], dict[str, cp_model.IntVar], cp_model.IntVar]:
    """Build CP-SAT model for normal SCCs; oversized SCCs use fixed indices."""
    model = cp_model.CpModel()
    K = len(sccs)
    normal_sccs = [s for s in sccs if s not in fixed_batch_of]
    batch_of: dict[str, cp_model.IntVar] = {
        s: model.new_int_var(0, K - 1, f"b_{s}") for s in normal_sccs
    }
    for s in normal_sccs:
        for ov_idx in oversized_indices:
            model.add(batch_of[s] != ov_idx)

    normal_batch_indices = [b for b in range(K) if b not in oversized_indices]
    in_batch: dict[tuple[str, int], cp_model.IntVar] = {}
    for s in normal_sccs:
        for b in normal_batch_indices:
            in_batch[(s, b)] = model.new_bool_var(f"ib_{s}_{b}")
            model.add(batch_of[s] == b).only_enforce_if(in_batch[(s, b)])
            model.add(batch_of[s] != b).only_enforce_if(in_batch[(s, b)].negated())

    for b in normal_batch_indices:
        model.add(sum(tokens_by_scc[s] * in_batch[(s, b)] for s in normal_sccs) <= budget)

    _add_ordering_constraints(model, dag_edges, batch_of, fixed_batch_of)
    spread_vars = _build_spread_vars(model, batch_of, sym_by_scc, normal_sccs, K)

    max_batch_idx = model.new_int_var(0, K - 1, "max_batch_idx")
    if normal_sccs:
        model.add_max_equality(max_batch_idx, list(batch_of.values()))
    else:
        max_fixed = max(fixed_batch_of.values()) if fixed_batch_of else 0
        model.add(max_batch_idx == max_fixed)

    return model, batch_of, spread_vars, max_batch_idx


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


def _build_spread_report(
    solver: cp_model.CpSolver,
    spread_vars: dict[str, cp_model.IntVar],
    sym_by_node: dict[str, tuple[SymbolRef, ...]],
) -> tuple[SymbolSpread, ...]:
    key_to_sym: dict[str, SymbolRef] = {
        f"{sym.file}:{sym.name}": sym
        for syms in sym_by_node.values()
        for sym in syms
    }
    return tuple(
        SymbolSpread(symbol=sym, spread=int(solver.value(var)))
        for key, var in spread_vars.items()
        if (sym := key_to_sym.get(key)) is not None
    )


def _extract_solution(
    solver: cp_model.CpSolver,
    sccs: list[str],
    batch_of: dict[str, cp_model.IntVar],
    scc_members: dict[str, list[str]],
    input_order: dict[str, int],
    dag_edges: tuple[tuple[str, str], ...],
    oversized_indices: set[int],
    fixed_batch_of: dict[str, int],
) -> tuple[Batch, ...]:
    raw: dict[int, list[str]] = {}
    scc_to_batch: dict[str, int] = {}
    for s in sccs:
        b = fixed_batch_of[s] if s in fixed_batch_of else solver.value(batch_of[s])
        raw.setdefault(b, []).extend(scc_members[s])
        scc_to_batch[s] = b

    sorted_batch_indices = sorted(raw)
    remap = {old: new for new, old in enumerate(sorted_batch_indices)}
    remapped_oversized = {remap[b] for b in sorted_batch_indices if b in oversized_indices}

    batch_deps: dict[int, set[int]] = {remap[b]: set() for b in sorted_batch_indices}
    for src_scc, dst_scc in dag_edges:
        src_b = remap[scc_to_batch[src_scc]]
        dst_b = remap[scc_to_batch[dst_scc]]
        if src_b != dst_b:
            batch_deps[dst_b].add(src_b)

    batches = []
    for old_b in sorted_batch_indices:
        new_b = remap[old_b]
        members = sorted(raw[old_b], key=lambda x: input_order.get(x, 0))
        depends_on = tuple(sorted(batch_deps[new_b]))
        is_oversized = new_b in remapped_oversized
        batches.append(Batch(
            index=new_b,
            change_ids=tuple(members),
            depends_on=depends_on,
            oversized=is_oversized,
        ))
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


def _partition_oversized(
    sccs: list[str], tokens_by_scc: dict[str, int], budget: int
) -> tuple[list[str], list[str]]:
    """Return (oversized_sccs, normal_sccs) partitioned by token budget."""
    oversized = [s for s in sccs if tokens_by_scc[s] > budget]
    return oversized, [s for s in sccs if tokens_by_scc[s] <= budget]


def _two_pass_solve(
    model: cp_model.CpModel,
    max_batch_idx: cp_model.IntVar,
    spread_vars: dict[str, cp_model.IntVar],
    time_limit_s: float,
) -> tuple[cp_model.CpSolver, SolverStatus]:
    """Lexicographic solve: minimise batch count, then minimise spread."""
    solver = cp_model.CpSolver()
    model.minimize(max_batch_idx)
    solver.parameters.max_time_in_seconds = time_limit_s / 2
    status1 = _status_name(solver.solve(model))
    if status1 in (STATUS_INFEASIBLE, STATUS_UNKNOWN):
        return solver, status1
    k_star = int(solver.value(max_batch_idx))
    model.add(max_batch_idx == k_star)
    model.minimize(sum(spread_vars.values()) if spread_vars else max_batch_idx)
    solver.parameters.max_time_in_seconds = time_limit_s / 2
    status2 = _status_name(solver.solve(model))
    return solver, _worse_status(status1, status2)


def plan_batches(
    changes: Sequence[FileChange],
    budget: int = 70_000,
    *,
    crg: CrgPort | None = None,
    new_relationships: Sequence[NewRelationship] = (),
    time_limit_s: float = 10.0,
    root: Path | None = None,
) -> BatchPlan:
    """Compute token-budgeted, precedence-respecting batches."""
    if time_limit_s < 0:
        raise ValueError(f"time_limit_s must be non-negative, got {time_limit_s}")
    if not changes:
        return BatchPlan(batches=(), spread_report=(), solver_status=STATUS_OPTIMAL)

    _validate_unique_ids(changes)
    _validate_no_symbol_overlap(changes)
    if root is None:
        root = Path.cwd()
    nodes, edges, sym_by_node = build_change_graph(changes, crg, new_relationships)
    scc_of, dag_edges = contract_sccs(nodes, edges)
    sccs, scc_members = _group_sccs(nodes, scc_of)
    tokens_by_scc = _tokens_per_scc(changes, scc_members, sccs, root)
    oversized_sccs, normal_sccs = _partition_oversized(sccs, tokens_by_scc, budget)
    fixed_batch_of: dict[str, int] = {s: i for i, s in enumerate(oversized_sccs)}
    oversized_indices: set[int] = set(fixed_batch_of.values())
    sym_by_scc_map = symbols_by_scc(scc_of, sym_by_node)
    model, batch_of, spread_vars, max_batch_idx = _build_model(
        sccs, dag_edges, tokens_by_scc, sym_by_scc_map, budget,
        oversized_indices, fixed_batch_of,
    )
    if not normal_sccs:
        input_order = {c.id: i for i, c in enumerate(changes)}
        batches = _extract_solution(
            cp_model.CpSolver(), sccs, batch_of, scc_members, input_order, dag_edges,
            oversized_indices, fixed_batch_of,
        )
        return BatchPlan(batches=batches, spread_report=(), solver_status=STATUS_OPTIMAL)

    solver, status_name = _two_pass_solve(model, max_batch_idx, spread_vars, time_limit_s)
    if status_name in (STATUS_INFEASIBLE, STATUS_UNKNOWN):
        return BatchPlan(batches=(), spread_report=(), solver_status=status_name)
    input_order = {c.id: i for i, c in enumerate(changes)}
    batches = _extract_solution(
        solver, sccs, batch_of, scc_members, input_order, dag_edges,
        oversized_indices, fixed_batch_of,
    )
    spread_report = _build_spread_report(solver, spread_vars, sym_by_node)
    return BatchPlan(batches=batches, spread_report=spread_report, solver_status=status_name)
