"""CP-SAT batch planner — lexicographic two-pass solver with oversized passthrough."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
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
    ContractedGraph,
    build_change_graph,
    contract_sccs,
    symbols_by_scc,
    validate_no_symbol_overlap,
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


@dataclass(frozen=True)
class _ModelInputs:
    """Inputs required to build the CP-SAT model for a contracted change graph."""
    sccs: list[str]
    dag_edges: tuple[tuple[str, str], ...]
    tokens_by_scc: dict[str, int]
    sym_by_scc: dict[str, tuple[SymbolRef, ...]]
    budget: int
    oversized_sccs: set[str]


@dataclass
class _ModelBundle:
    """Built CP-SAT model plus the decision/objective variables the solver needs."""
    model: cp_model.CpModel
    batch_of: dict[str, cp_model.IntVar]
    spread_vars: dict[str, cp_model.IntVar] = field(default_factory=dict)
    max_batch_idx: cp_model.IntVar | None = None


@dataclass(frozen=True)
class _Snapshot:
    """Solver-independent snapshot of decision variable values."""
    batch_of: dict[str, int]
    spread_of: dict[str, int]


def _add_ordering_constraints(
    model: cp_model.CpModel,
    dag_edges: tuple[tuple[str, str], ...],
    batch_of: dict[str, cp_model.IntVar],
) -> None:
    for src, dst in dag_edges:
        model.add(batch_of[src] <= batch_of[dst])


def _isolate_oversized(
    model: cp_model.CpModel,
    sccs: list[str],
    oversized_sccs: set[str],
    batch_of: dict[str, cp_model.IntVar],
) -> None:
    """Each oversized SCC must occupy a batch alone."""
    for ov in oversized_sccs:
        for other in sccs:
            if other != ov:
                model.add(batch_of[ov] != batch_of[other])


def _add_budget_constraints(
    model: cp_model.CpModel,
    normal_sccs: list[str],
    batch_of: dict[str, cp_model.IntVar],
    inputs: _ModelInputs,
) -> None:
    if not normal_sccs:
        return
    K = len(batch_of)
    in_batch: dict[tuple[str, int], cp_model.IntVar] = {}
    for s in normal_sccs:
        for b in range(K):
            in_batch[(s, b)] = model.new_bool_var(f"ib_{s}_{b}")
            model.add(batch_of[s] == b).only_enforce_if(in_batch[(s, b)])
            model.add(batch_of[s] != b).only_enforce_if(in_batch[(s, b)].negated())
    for b in range(K):
        model.add(
            sum(inputs.tokens_by_scc[s] * in_batch[(s, b)] for s in normal_sccs)
            <= inputs.budget
        )


def _build_model(inputs: _ModelInputs) -> _ModelBundle:
    """Build the CP-SAT model. Oversized SCCs are isolated by mutual-exclusion."""
    model = cp_model.CpModel()
    K = len(inputs.sccs)
    batch_of: dict[str, cp_model.IntVar] = {
        s: model.new_int_var(0, K - 1, f"b_{s}") for s in inputs.sccs
    }
    _isolate_oversized(model, inputs.sccs, inputs.oversized_sccs, batch_of)
    normal_sccs = [s for s in inputs.sccs if s not in inputs.oversized_sccs]
    _add_budget_constraints(model, normal_sccs, batch_of, inputs)
    _add_ordering_constraints(model, inputs.dag_edges, batch_of)
    spread_vars = _build_spread_vars(model, batch_of, inputs.sym_by_scc, normal_sccs, K)
    max_batch_idx = model.new_int_var(0, K - 1, "max_batch_idx")
    model.add_max_equality(max_batch_idx, list(batch_of.values()))
    return _ModelBundle(
        model=model,
        batch_of=batch_of,
        spread_vars=spread_vars,
        max_batch_idx=max_batch_idx,
    )


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
    spread_of: dict[str, int],
    sym_by_node: dict[str, tuple[SymbolRef, ...]],
) -> tuple[SymbolSpread, ...]:
    key_to_sym: dict[str, SymbolRef] = {
        f"{sym.file}:{sym.name}": sym
        for syms in sym_by_node.values()
        for sym in syms
    }
    return tuple(
        SymbolSpread(symbol=sym, spread=value)
        for key, value in spread_of.items()
        if (sym := key_to_sym.get(key)) is not None
    )


def _extract_solution(
    batch_of: dict[str, int],
    scc_members: dict[str, list[str]],
    input_order: dict[str, int],
    dag_edges: tuple[tuple[str, str], ...],
    oversized_sccs: set[str],
) -> tuple[Batch, ...]:
    raw: dict[int, list[str]] = {}
    oversized_batch_indices: set[int] = set()
    for scc_id, b in batch_of.items():
        raw.setdefault(b, []).extend(scc_members[scc_id])
        if scc_id in oversized_sccs:
            oversized_batch_indices.add(b)
    sorted_batch_indices = sorted(raw)
    remap = {old: new for new, old in enumerate(sorted_batch_indices)}
    remapped_oversized = {remap[b] for b in sorted_batch_indices if b in oversized_batch_indices}
    batch_deps = _compute_batch_deps(dag_edges, batch_of, remap)
    batches: list[Batch] = []
    for old_b in sorted_batch_indices:
        new_b = remap[old_b]
        members = sorted(raw[old_b], key=lambda x: input_order.get(x, 0))
        batches.append(Batch(
            index=new_b,
            change_ids=tuple(members),
            depends_on=tuple(sorted(batch_deps[new_b])),
            oversized=new_b in remapped_oversized,
        ))
    return tuple(batches)


def _compute_batch_deps(
    dag_edges: tuple[tuple[str, str], ...],
    batch_of: dict[str, int],
    remap: dict[int, int],
) -> dict[int, set[int]]:
    batch_deps: dict[int, set[int]] = {new: set() for new in remap.values()}
    for src_scc, dst_scc in dag_edges:
        src_b = remap[batch_of[src_scc]]
        dst_b = remap[batch_of[dst_scc]]
        if src_b != dst_b:
            batch_deps[dst_b].add(src_b)
    return batch_deps


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
    # Dedup SCC mass by path: when multiple changes in an SCC share the same
    # path, only the first-seen change contributes its token estimate — the
    # file's token mass doesn't multiply with change count. See weights.py
    # for the "path dedup, first-seen" calibration notes.
    tokens_by_change = {c.id: estimate_tokens(c, root) for c in changes}
    change_by_id = {c.id: c for c in changes}

    def _scc_tokens(member_ids: list[str]) -> int:
        seen_paths: set[str] = set()
        total = 0
        for cid in member_ids:
            path = change_by_id[cid].path
            if path not in seen_paths:
                seen_paths.add(path)
                total += tokens_by_change[cid]
        return total

    return {s: _scc_tokens(scc_members[s]) for s in sccs}


def _partition_oversized(
    sccs: list[str], tokens_by_scc: dict[str, int], budget: int
) -> tuple[list[str], list[str]]:
    """Return (oversized_sccs, normal_sccs) partitioned by token budget."""
    oversized = [s for s in sccs if tokens_by_scc[s] > budget]
    return oversized, [s for s in sccs if tokens_by_scc[s] <= budget]


def _take_snapshot(
    solver: cp_model.CpSolver,
    batch_of: dict[str, cp_model.IntVar],
    spread_vars: dict[str, cp_model.IntVar],
) -> _Snapshot:
    return _Snapshot(
        batch_of={k: int(solver.value(v)) for k, v in batch_of.items()},
        spread_of={k: int(solver.value(v)) for k, v in spread_vars.items()},
    )


def _two_pass_solve(
    bundle: _ModelBundle,
    time_limit_s: float,
) -> tuple[_Snapshot | None, SolverStatus]:
    """Lexicographic solve: minimise batch count, then minimise spread.

    If pass-2 degrades a usable pass-1 solution to UNKNOWN or INFEASIBLE,
    the pass-1 snapshot is returned so callers can still emit a valid plan.
    """
    solver = cp_model.CpSolver()
    model = bundle.model
    model.minimize(bundle.max_batch_idx)
    solver.parameters.max_time_in_seconds = time_limit_s / 2
    status1 = _status_name(solver.solve(model))
    if status1 in (STATUS_INFEASIBLE, STATUS_UNKNOWN):
        return None, status1
    pass1 = _take_snapshot(solver, bundle.batch_of, bundle.spread_vars)
    k_star = int(solver.value(bundle.max_batch_idx))
    model.add(bundle.max_batch_idx == k_star)
    model.minimize(
        sum(bundle.spread_vars.values()) if bundle.spread_vars else bundle.max_batch_idx
    )
    solver.parameters.max_time_in_seconds = time_limit_s / 2
    status2 = _status_name(solver.solve(model))
    if status2 in (STATUS_INFEASIBLE, STATUS_UNKNOWN):
        # Pass 2 failed to improve; keep the valid pass-1 solution.
        return pass1, status1
    return _take_snapshot(solver, bundle.batch_of, bundle.spread_vars), _worse_status(
        status1, status2
    )


# Public API: the 6 knobs are intentional — budget, time_limit_s, and root
# each configure different concerns (solver scale, solver timeout, filesystem
# root for token estimation) while ``crg`` and ``new_relationships`` are
# optional orchestration hooks. Internal helpers obey the 4-param budget.
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
    validate_no_symbol_overlap(changes)
    return _run_solver(
        changes,
        budget=budget,
        crg=crg,
        new_relationships=new_relationships,
        time_limit_s=time_limit_s,
        root=root if root is not None else Path.cwd(),
    )


def _run_solver(
    changes: Sequence[FileChange],
    *,
    budget: int,
    crg: CrgPort | None,
    new_relationships: Sequence[NewRelationship],
    time_limit_s: float,
    root: Path,
) -> BatchPlan:
    graph = build_change_graph(changes, crg, new_relationships)
    contracted: ContractedGraph = contract_sccs(graph.nodes, graph.edges)
    sccs, scc_members = _group_sccs(graph.nodes, contracted.scc_of)
    tokens_by_scc = _tokens_per_scc(changes, scc_members, sccs, root)
    oversized_list, _ = _partition_oversized(sccs, tokens_by_scc, budget)
    inputs = _ModelInputs(
        sccs=sccs,
        dag_edges=contracted.dag_edges,
        tokens_by_scc=tokens_by_scc,
        sym_by_scc=symbols_by_scc(contracted.scc_of, graph.symbols_by_node),
        budget=budget,
        oversized_sccs=set(oversized_list),
    )
    bundle = _build_model(inputs)
    snapshot, status = _two_pass_solve(bundle, time_limit_s)
    if snapshot is None:
        return BatchPlan(batches=(), spread_report=(), solver_status=status)
    input_order = {c.id: i for i, c in enumerate(changes)}
    batches = _extract_solution(
        snapshot.batch_of, scc_members, input_order, contracted.dag_edges, inputs.oversized_sccs,
    )
    return BatchPlan(
        batches=batches,
        spread_report=_build_spread_report(snapshot.spread_of, graph.symbols_by_node),
        solver_status=status,
    )
