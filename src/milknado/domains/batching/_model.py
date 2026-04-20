"""CP-SAT model builder and two-pass solver for batch planning."""
from __future__ import annotations

from dataclasses import dataclass, field

from ortools.sat.python import cp_model

from milknado.domains.batching.change import SymbolRef


@dataclass(frozen=True)
class ModelInputs:
    sccs: list[str]
    dag_edges: tuple[tuple[str, str], ...]
    tokens_by_scc: dict[str, int]
    sym_by_scc: dict[str, tuple[SymbolRef, ...]]
    budget: int
    oversized_sccs: set[str]


@dataclass
class ModelBundle:
    model: cp_model.CpModel
    batch_of: dict[str, cp_model.IntVar]
    spread_vars: dict[str, cp_model.IntVar] = field(default_factory=dict)
    max_batch_idx: cp_model.IntVar | None = None
    total_cost: cp_model.IntVar | None = None


@dataclass(frozen=True)
class Snapshot:
    batch_of: dict[str, int]
    spread_of: dict[str, int]


def build_model(inputs: ModelInputs) -> ModelBundle:
    model = cp_model.CpModel()
    K = len(inputs.sccs)
    batch_of: dict[str, cp_model.IntVar] = {
        s: model.new_int_var(0, K - 1, f"b_{s}") for s in inputs.sccs
    }
    _isolate_oversized(model, inputs.sccs, inputs.oversized_sccs, batch_of)
    normal_sccs = [s for s in inputs.sccs if s not in inputs.oversized_sccs]
    in_batch = _build_in_batch_vars(model, normal_sccs, batch_of, K)
    _add_budget_constraints(model, normal_sccs, in_batch, inputs)
    _add_ordering_constraints(model, inputs.dag_edges, batch_of)
    spread_vars = _build_spread_vars(model, batch_of, inputs.sym_by_scc, normal_sccs, K)
    max_batch_idx = model.new_int_var(0, K - 1, "max_batch_idx")
    model.add_max_equality(max_batch_idx, list(batch_of.values()))
    total_cost = _build_total_cost(model, normal_sccs, in_batch, inputs, K)
    return ModelBundle(
        model=model,
        batch_of=batch_of,
        spread_vars=spread_vars,
        max_batch_idx=max_batch_idx,
        total_cost=total_cost,
    )


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
    for ov in oversized_sccs:
        for other in sccs:
            if other != ov:
                model.add(batch_of[ov] != batch_of[other])


def _add_budget_constraints(
    model: cp_model.CpModel,
    normal_sccs: list[str],
    in_batch: dict[tuple[str, int], cp_model.IntVar],
    inputs: ModelInputs,
) -> None:
    if not normal_sccs:
        return
    K = max((b for _, b in in_batch), default=-1) + 1
    for b in range(K):
        model.add(
            sum(inputs.tokens_by_scc[s] * in_batch[(s, b)] for s in normal_sccs)
            <= inputs.budget
        )


def _build_in_batch_vars(
    model: cp_model.CpModel,
    normal_sccs: list[str],
    batch_of: dict[str, cp_model.IntVar],
    K: int,
) -> dict[tuple[str, int], cp_model.IntVar]:
    in_batch: dict[tuple[str, int], cp_model.IntVar] = {}
    for s in normal_sccs:
        for b in range(K):
            in_batch[(s, b)] = model.new_bool_var(f"ib_{s}_{b}")
            model.add(batch_of[s] == b).only_enforce_if(in_batch[(s, b)])
            model.add(batch_of[s] != b).only_enforce_if(in_batch[(s, b)].negated())
    return in_batch


def _build_total_cost(
    model: cp_model.CpModel,
    normal_sccs: list[str],
    in_batch: dict[tuple[str, int], cp_model.IntVar],
    inputs: ModelInputs,
    K: int,
) -> cp_model.IntVar:
    if not normal_sccs:
        return model.new_constant(0)

    size_cost_x100 = [_batch_size_cost(k) * 100 for k in range(K + 1)]
    file_mult_x100 = [
        100 + (k * 12 - 1) * 10 if k > 0 else 100
        for k in range(K + 1)
    ]

    sum_tokens = sum(inputs.tokens_by_scc.values())
    max_size_cost = max(size_cost_x100)
    max_mult = max(file_mult_x100)
    big_m = max_size_cost + sum_tokens * max_mult

    per_batch_costs: list[cp_model.IntVar] = []
    for b in range(K):
        size_b = model.new_int_var(0, len(normal_sccs), f"size_b{b}")
        model.add(size_b == sum(in_batch[(s, b)] for s in normal_sccs))
        file_b = model.new_int_var(0, sum_tokens, f"file_b{b}")
        model.add(
            file_b == sum(inputs.tokens_by_scc[s] * in_batch[(s, b)] for s in normal_sccs)
        )
        at_size: dict[int, cp_model.IntVar] = {}
        for k in range(K + 1):
            at_size[k] = model.new_bool_var(f"atsize_b{b}_k{k}")
            model.add(size_b == k).only_enforce_if(at_size[k])
            model.add(size_b != k).only_enforce_if(at_size[k].negated())
        model.add_exactly_one(list(at_size.values()))
        size_cost_b = model.new_int_var(0, max_size_cost, f"sizecost_b{b}")
        model.add(
            size_cost_b == sum(at_size[k] * size_cost_x100[k] for k in range(K + 1))
        )
        file_cost_b = model.new_int_var(0, sum_tokens * max_mult, f"filecost_b{b}")
        for k in range(K + 1):
            model.add(file_cost_b == file_b * file_mult_x100[k]).only_enforce_if(at_size[k])
        cost_b = model.new_int_var(0, big_m, f"cost_b{b}")
        model.add(cost_b == size_cost_b + file_cost_b)
        per_batch_costs.append(cost_b)

    total = model.new_int_var(0, big_m * K, "total_cost")
    model.add(total == sum(per_batch_costs))
    return total


def _build_spread_vars(
    model: cp_model.CpModel,
    batch_of: dict[str, cp_model.IntVar],
    sym_by_scc: dict[str, tuple[SymbolRef, ...]],
    sccs: list[str],
    K: int,
) -> dict[str, cp_model.IntVar]:
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


def _batch_size_cost(k: int) -> int:
    from milknado.domains.batching.weights import batch_size_cost
    return batch_size_cost(k)
