from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, cast

from easy_cheese_schemas import (
    AgentWriterView,
    CurdPlan,
    CurdResult,
    CurdResultWriterView,
    SourceCurdRef,
    SourcePlanRef,
    WriterViewKind,
    canonical_digest,
    normalize_agent_value,
    supported_version_for,
    validate_contract,
)

if TYPE_CHECKING:
    from milknado.domains.graph import MikadoGraph


def normalize_curd_result(  # noqa: PLR0913 - Coordinates host-owned result fields.
    node_id: int,
    writer_view: CurdResultWriterView,
    graph: MikadoGraph,
    evidence: Mapping[str, object],
    deliverables: Mapping[str, object],
    *,
    provenance_refs: Sequence[str] = (),
) -> CurdResult:
    """Normalize one admitted task's writer view into a canonical result."""
    identity = graph.get_curd_node(node_id)
    if identity is None:
        raise ValueError(f"node {node_id} has no admitted Curd identity")
    plan_record = graph.get_curd_plan(identity.plan_id, identity.revision)
    if plan_record is None:
        raise ValueError(f"Curd plan {identity.plan_id!r} revision {identity.revision} is missing")
    result_version = supported_version_for(CurdResult)
    plan_version = supported_version_for(CurdPlan)
    if result_version is None or plan_version is None:
        raise ValueError("Easy Cheese does not expose supported Curd contract versions")
    plan_artifact = validate_contract(
        plan_record.canonical_bytes,
        CurdPlan,
        supported_version=plan_version,
    )
    plan = cast(CurdPlan, plan_artifact.value)
    if plan.plan_id != identity.plan_id or plan.revision != identity.revision:
        raise ValueError("stored Curd plan identity does not match its node mapping")
    if plan.digest != plan_record.digest:
        raise ValueError("stored Curd plan digest does not match its canonical payload")
    try:
        curd = next(curd for curd in plan.curds if curd.curd_id == identity.curd_id)
    except StopIteration as error:
        raise ValueError(
            f"stored Curd plan has no curd {identity.curd_id!r} for node {node_id}"
        ) from error
    curd_digest = canonical_digest(curd)
    if curd_digest != identity.curd_digest:
        raise ValueError("stored Curd identity digest does not match its plan")

    normalized = normalize_agent_value(
        AgentWriterView(kind=WriterViewKind.CURD_RESULT, payload=writer_view),
        {
            "contract_version": result_version,
            "result_id": f"{plan.plan_id}/curd/{curd.curd_id}/result",
            "source_plan_ref": SourcePlanRef(
                plan_id=plan.plan_id,
                revision=plan.revision,
                digest=plan.digest,
            ),
            "source_curd_ref": SourceCurdRef(
                curd_id=curd.curd_id,
                digest=curd_digest,
            ),
            "expected_criterion_ids": tuple(criterion.criterion_id for criterion in curd.criteria),
            "evidence": evidence,
            "deliverables": deliverables,
            "provenance_refs": tuple(provenance_refs),
        },
    )
    if not isinstance(normalized, CurdResult):
        raise TypeError("Easy Cheese normalization returned a non-CurdResult value")
    return normalized


__all__ = ["normalize_curd_result"]
