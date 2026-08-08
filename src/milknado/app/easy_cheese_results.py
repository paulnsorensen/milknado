from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from easy_cheese_schemas import (
    AgentWriterView,
    ContractVersion,
    Criterion,
    CriterionDisposition,
    CriterionResultWriterView,
    CurdResult,
    CurdResultWriterView,
    EvidenceRef,
    SemanticCurd,
    WriterViewKind,
    normalize_agent_value,
    supported_version_for,
)

from milknado.app.easy_cheese_bindings import (
    BoundMilknadoExecution,
    CurdExecution,
    bind_milknado_execution,
)
from milknado.app.easy_cheese_evidence import node_evidence, prepare_artifact_directory
from milknado.app.easy_cheese_execution import AppliedMilknadoPlan
from milknado.app.easy_cheese_outcomes import (
    PHYSICAL_OUTCOME_REASON_MAX,
    MilknadoCriterionOutcome,
    parse_milknado_physical_outcome,
)
from milknado.domains.common import MikadoNode, NodeStatus
from milknado.domains.graph import MikadoGraph
from milknado.domains.planning import MilknadoPlanV2


@dataclass(frozen=True)
class _NodeFact:
    status: str
    state: str
    result: str | None
    evidence: EvidenceRef
    outcomes: dict[tuple[str, str], MilknadoCriterionOutcome]


@dataclass(frozen=True)
class _AggregationContext:
    plan: MilknadoPlanV2
    bound: BoundMilknadoExecution
    facts: dict[int, _NodeFact]
    contract_version: ContractVersion


def _status_value(node: MikadoNode) -> str:
    value = node.status.value
    if not isinstance(value, str):
        raise ValueError("Mikado node has an invalid status")
    return value


def _status_class(status: str) -> str:
    if status == NodeStatus.DONE.value:
        return "done"
    if status == NodeStatus.FAILED.value:
        return "failed"
    if status in {
        NodeStatus.PENDING.value,
        NodeStatus.RUNNING.value,
        "in_progress",
        NodeStatus.BLOCKED.value,
    }:
        return "blocked"
    raise ValueError(f"unsupported Mikado node status {status!r}")


def _reason_for_node(node_id: int, status: str, result: str | None) -> str:
    reason = f"physical node {node_id} is {status}"
    if status == NodeStatus.FAILED.value and result is None:
        return f"{reason} without a deposited result"
    return f"{reason}; physical work remains unresolved"


def _bounded_reason(reasons: list[str]) -> str:
    reason = "; ".join(dict.fromkeys(reasons))
    if len(reason) <= PHYSICAL_OUTCOME_REASON_MAX:
        return reason
    return reason[: PHYSICAL_OUTCOME_REASON_MAX - 13] + " [truncated]"


def _read_node_facts(
    graph: MikadoGraph,
    bound: BoundMilknadoExecution,
    artifact_directory: Path,
) -> dict[int, _NodeFact]:
    latest_results = graph.latest_results_for_nodes(tuple(bound.nodes))
    facts: dict[int, _NodeFact] = {}
    for node_id, node in bound.nodes.items():
        status = _status_value(node)
        result = latest_results.get(node_id)
        outcomes = (
            parse_milknado_physical_outcome(result, bound.expected_criteria[node_id])
            if status == NodeStatus.DONE.value
            else {}
        )
        facts[node_id] = _NodeFact(
            status=status,
            state=_status_class(status),
            result=result,
            evidence=node_evidence(node_id, status, result, artifact_directory),
            outcomes=outcomes,
        )
    return facts


def _disposition(states: list[str]) -> CriterionDisposition:
    if "failed" in states:
        return CriterionDisposition.FAILED
    if "blocked" in states:
        return CriterionDisposition.BLOCKED
    if "skipped" in states:
        return CriterionDisposition.SKIPPED
    return CriterionDisposition.PASSED


def _criterion_result(
    curd: SemanticCurd,
    criterion: Criterion,
    execution: CurdExecution,
    facts: dict[int, _NodeFact],
) -> tuple[CriterionResultWriterView, tuple[EvidenceRef, ...], list[str], list[int]]:
    node_order = list(dict.fromkeys(node_id for _change_id, node_id in execution.items))
    states: list[str] = []
    reasons: list[str] = []
    key = (curd.curd_id, criterion.criterion_id)
    for node_id in node_order:
        fact = facts[node_id]
        if fact.state != "done":
            states.append(fact.state)
            reasons.append(_reason_for_node(node_id, fact.status, fact.result))
            continue
        outcome = fact.outcomes.get(key)
        if outcome is None:
            states.append("blocked")
            reasons.append(
                f"physical node {node_id} is done without a typed outcome for "
                f"{curd.curd_id}/{criterion.criterion_id}"
            )
            continue
        states.append(outcome.disposition.value)
        if outcome.reason is not None:
            reasons.append(outcome.reason)

    disposition = _disposition(states)
    reason = None if disposition is CriterionDisposition.PASSED else _bounded_reason(reasons)
    evidence = tuple(facts[node_id].evidence for node_id in node_order)
    return (
        CriterionResultWriterView(
            disposition=disposition,
            evidence_keys=tuple(item.evidence_id for item in evidence),
            reason=reason,
        ),
        evidence,
        reasons,
        node_order,
    )


def _normalize_curd_result(
    context: _AggregationContext,
    curd: SemanticCurd,
    curd_index: int,
) -> CurdResult:
    execution = context.bound.items_by_curd[curd.curd_id]
    criterion_rows: list[CriterionResultWriterView] = []
    evidence: dict[str, EvidenceRef] = {}
    unresolved: list[str] = []
    node_order: list[int] = []
    for criterion in curd.criteria:
        row, row_evidence, reasons, row_nodes = _criterion_result(
            curd, criterion, execution, context.facts
        )
        criterion_rows.append(row)
        evidence.update((item.evidence_id, item) for item in row_evidence)
        if row.disposition is not CriterionDisposition.PASSED:
            unresolved.extend(reasons)
        for node_id in row_nodes:
            if node_id not in node_order:
                node_order.append(node_id)

    writer = AgentWriterView(
        kind=WriterViewKind.CURD_RESULT,
        payload=CurdResultWriterView(
            criterion_results=tuple(criterion_rows),
            deliverables=(),
            unresolved_work=tuple(dict.fromkeys(unresolved)),
        ),
    )
    normalized = normalize_agent_value(
        writer,
        {
            "result_id": (
                f"{context.plan.source_plan_ref.plan_id}/revision/"
                f"{context.plan.source_plan_ref.revision}/result/{curd_index}"
            ),
            "source_plan_ref": context.plan.source_plan_ref,
            "source_curd_ref": execution.source_curd_ref,
            "expected_criterion_ids": [criterion.criterion_id for criterion in curd.criteria],
            "evidence": evidence,
            "deliverables": {},
            "runtime_refs": [f"milknado://nodes/{node_id}/outcome" for node_id in node_order],
            "contract_version": context.contract_version,
        },
    )
    if not isinstance(normalized, CurdResult):
        raise TypeError("Easy Cheese normalization returned a non-CurdResult value")
    return normalized


def collect_milknado_results(
    graph: MikadoGraph,
    plan: MilknadoPlanV2,
    application: AppliedMilknadoPlan,
    *,
    artifact_directory: str | Path,
) -> tuple[CurdResult, ...]:
    evidence_directory = prepare_artifact_directory(artifact_directory)
    bound = bind_milknado_execution(graph, plan, application)
    contract_version = supported_version_for(CurdResult)
    if contract_version is None:
        raise ValueError("Easy Cheese does not expose a supported CurdResult version")
    context = _AggregationContext(
        plan,
        bound,
        _read_node_facts(graph, bound, evidence_directory),
        contract_version,
    )
    return tuple(
        _normalize_curd_result(context, curd, curd_index)
        for curd_index, curd in enumerate(plan.semantic_plan.curds, start=1)
    )


__all__ = ["collect_milknado_results"]
