"""CP-SAT batch planner — lexicographic two-pass solver with oversized passthrough."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from ortools.sat.python import cp_model

from milknado.domains.batching._model import (
    ModelBundle,
    ModelInputs,
    Snapshot,
    build_model,
)
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
)
from milknado.domains.batching.weights import (
    estimate_tokens_per_symbols,  # noqa: F401 — used in _tokens_per_scc
)
from milknado.domains.common.protocols import CrgPort, TilthPort

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


def _take_snapshot(
    solver: cp_model.CpSolver,
    batch_of: dict[str, cp_model.IntVar],
    spread_vars: dict[str, cp_model.IntVar],
) -> Snapshot:
    return Snapshot(
        batch_of={k: int(solver.value(v)) for k, v in batch_of.items()},
        spread_of={k: int(solver.value(v)) for k, v in spread_vars.items()},
    )


def _two_pass_solve(
    bundle: ModelBundle,
    time_limit_s: float,
) -> tuple[Snapshot | None, SolverStatus]:
    solver = cp_model.CpSolver()
    model = bundle.model
    cost_objective = bundle.total_cost if bundle.total_cost is not None else bundle.max_batch_idx
    model.minimize(cost_objective)  # type: ignore
    solver.parameters.max_time_in_seconds = time_limit_s / 2
    status1 = _status_name(solver.solve(model))
    if status1 in (STATUS_INFEASIBLE, STATUS_UNKNOWN):
        return None, status1
    pass1 = _take_snapshot(solver, bundle.batch_of, bundle.spread_vars)
    cost_star = int(solver.value(cost_objective))  # type: ignore
    model.add(cost_objective == cost_star)
    model.minimize(
        sum(bundle.spread_vars.values()) if bundle.spread_vars else cost_objective  # type: ignore
    )
    solver.parameters.max_time_in_seconds = time_limit_s / 2
    status2 = _status_name(solver.solve(model))
    if status2 in (STATUS_INFEASIBLE, STATUS_UNKNOWN):
        return pass1, status1
    return _take_snapshot(solver, bundle.batch_of, bundle.spread_vars), _worse_status(
        status1, status2
    )


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
    tilth_port: TilthPort | None = None,
) -> dict[str, int]:
    change_by_id = {c.id: c for c in changes}

    def _scc_tokens(member_ids: list[str]) -> int:
        seen_sym_keys: set[tuple[str, str]] = set()
        seen_paths: set[str] = set()
        total = 0
        for cid in member_ids:
            change = change_by_id[cid]
            if tilth_port is not None and change.symbols:
                new_syms = tuple(
                    sym for sym in change.symbols if (sym.file, sym.name) not in seen_sym_keys
                )
                if new_syms:
                    filtered = FileChange(
                        id=change.id,
                        path=change.path,
                        edit_kind=change.edit_kind,
                        symbols=new_syms,
                    )
                    total += estimate_tokens_per_symbols(filtered, root, tilth_port)
                    for sym in new_syms:
                        seen_sym_keys.add((sym.file, sym.name))
            else:
                if change.path not in seen_paths:
                    seen_paths.add(change.path)
                    total += estimate_tokens_per_symbols(change, root, None)
        return total

    return {s: _scc_tokens(scc_members[s]) for s in sccs}


def _partition_oversized(
    sccs: list[str], tokens_by_scc: dict[str, int], budget: int
) -> tuple[list[str], list[str]]:
    oversized = [s for s in sccs if tokens_by_scc[s] > budget]
    return oversized, [s for s in sccs if tokens_by_scc[s] <= budget]


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


# Public API: the 7 knobs are intentional — budget, time_limit_s, and root
# each configure different concerns (solver scale, solver timeout, filesystem
# root for token estimation) while ``crg``, ``new_relationships``, and
# ``tilth_port`` are optional orchestration hooks.
def plan_batches(
    changes: Sequence[FileChange],
    budget: int = 150_000,
    *,
    crg: CrgPort | None = None,
    new_relationships: Sequence[NewRelationship] = (),
    time_limit_s: float = 10.0,
    root: Path | None = None,
    tilth_port: TilthPort | None = None,
) -> BatchPlan:
    """Compute token-budgeted, precedence-respecting batches."""
    if time_limit_s < 0:
        raise ValueError(f"time_limit_s must be non-negative, got {time_limit_s}")
    if not changes:
        return BatchPlan(batches=(), spread_report=(), solver_status=STATUS_OPTIMAL)
    _validate_unique_ids(changes)
    from milknado.domains.batching.graph_build import validate_no_symbol_overlap

    validate_no_symbol_overlap(changes)
    return _run_solver(
        changes,
        budget=budget,
        crg=crg,
        new_relationships=new_relationships,
        time_limit_s=time_limit_s,
        root=root if root is not None else Path.cwd(),
        tilth_port=tilth_port,
    )


def _run_solver(
    changes: Sequence[FileChange],
    *,
    budget: int,
    crg: CrgPort | None,
    new_relationships: Sequence[NewRelationship],
    time_limit_s: float,
    root: Path,
    tilth_port: TilthPort | None,
) -> BatchPlan:
    graph = build_change_graph(changes, crg, new_relationships)
    contracted: ContractedGraph = contract_sccs(graph.nodes, graph.edges)
    sccs, scc_members = _group_sccs(graph.nodes, contracted.scc_of)
    tokens_by_scc = _tokens_per_scc(changes, scc_members, sccs, root, tilth_port)
    oversized_list, _ = _partition_oversized(sccs, tokens_by_scc, budget)
    inputs = ModelInputs(
        sccs=sccs,
        dag_edges=contracted.dag_edges,
        tokens_by_scc=tokens_by_scc,
        sym_by_scc=symbols_by_scc(contracted.scc_of, graph.symbols_by_node),
        budget=budget,
        oversized_sccs=set(oversized_list),
    )
    bundle = build_model(inputs)
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
