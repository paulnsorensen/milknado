from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest
from easy_cheese_schemas import (
    COMPILED_TRANSITION_REGISTRY,
    ArtifactRef,
    ArtifactResolutionError,
    ContractValidationError,
    ContractVersion,
    CriterionDisposition,
    CurdPlan,
    CurdResult,
    PlannerResult,
    TransitionError,
    UnsupportedProjection,
    canonical_digest,
    list_conformance_fixtures,
    load_conformance_fixture,
    normalize_agent_output,
    project_curd_block,
    read_conformance_fixture,
    resolve_artifact,
    validate_contract,
    validate_transition,
)

from milknado.app import (
    MilknadoCriterionOutcome,
    apply_milknado_plan_to_graph,
    collect_milknado_results,
    encode_milknado_physical_outcome,
)
from milknado.domains.batching import Batch, BatchPlan
from milknado.domains.common import NodeStatus, RunResult
from milknado.domains.graph import MikadoGraph
from milknado.domains.planning import import_milknado

Fixture = dict[str, Any]
ERROR_TYPES = {
    "ArtifactResolutionError": ArtifactResolutionError,
    "ContractValidationError": ContractValidationError,
    "TransitionError": TransitionError,
}

FIXTURE_DIGESTS = {
    "contract-cases.json": "7939bef1b4aefc15ef1748249f3bb25b837f0237c7485a5df2df88afa635f4b5",
    "normalization-cases.json": "f7720c619e239e480a4f6db3c772ccb294b6df765c8c95d9ed2517aac7f61ffe",
}


def _fixture(name: str) -> Fixture:
    return cast(Fixture, load_conformance_fixture(name))


CONTRACT_CASES = _fixture("contract-cases.json")["cases"]
NORMALIZATION_CASES = _fixture("normalization-cases.json")["cases"]


def _version(case: Fixture) -> ContractVersion:
    return ContractVersion(**case["supported_version"])


def _validated_observation(case: Fixture) -> Fixture:
    value = validate_contract(
        case["input"], case["schema_uri"], supported_version=_version(case)
    ).value
    if case["schema_uri"].endswith("/curd-plan"):
        assert isinstance(value, CurdPlan)
        return {
            "type": type(value).__name__,
            "plan_id": value.plan_id,
            "revision": value.revision,
            "curd_ids": [curd.curd_id for curd in value.curds],
            "criterion_ids": [
                criterion.criterion_id for curd in value.curds for criterion in curd.criteria
            ],
        }
    assert isinstance(value, CurdResult)
    return {
        "type": type(value).__name__,
        "result_id": value.result_id,
        "disposition": value.disposition.value,
        "expected_criterion_ids": list(value.expected_criterion_ids),
        "criterion_result_ids": [result.criterion_id for result in value.criterion_results],
    }


def _normalized_observation(case: Fixture) -> Fixture:
    invocation = case["input"]
    value = normalize_agent_output(invocation["writer_view"], invocation["host_invocation"]).value
    assert isinstance(value, PlannerResult)
    return {
        "type": type(value).__name__,
        "disposition": value.disposition.value,
        "request_id": value.request_id,
        "plan_id": value.plan.plan_id if value.plan else None,
        "curd_ids": [curd.curd_id for curd in value.plan.curds] if value.plan else [],
        "uncertainty_scopes": [item.scope.value for item in value.unresolved_work],
    }


def _artifact_observation(case: Fixture, tmp_path: Path) -> Fixture:
    repository_root = tmp_path / case["name"]
    repository_root.mkdir()
    if source := case["input"].get("source"):
        source_path = repository_root / source["path"]
        source_path.parent.mkdir(parents=True)
        source_path.write_text(source["content"], encoding="utf-8")
    resolved = resolve_artifact(
        ArtifactRef(**case["input"]["artifact"]),
        repository_root=repository_root,
        artifact_directory=tmp_path / "retained",
    )
    return {
        "type": type(resolved).__name__,
        "role": resolved.role,
        "media_type": resolved.media_type,
        "content": Path(resolved.path).read_text(encoding="utf-8"),
    }


def _transition_observation(case: Fixture) -> Fixture:
    transition = validate_transition(COMPILED_TRANSITION_REGISTRY, **case["input"])
    return {
        "type": type(transition).__name__,
        "source": transition.source,
        "destination": transition.destination,
        "payload_schema_uri": transition.payload_schema_uri,
    }


def _projection_observation(case: Fixture) -> Fixture:
    plan = validate_contract(
        case["input"], case["schema_uri"], supported_version=_version(case)
    ).value
    assert isinstance(plan, CurdPlan)
    projected = project_curd_block(plan)
    assert isinstance(projected, UnsupportedProjection)
    return {
        "type": type(projected).__name__,
        "target": projected.target,
        "curd_id": projected.curd_id,
        "field": projected.field,
        "reason": projected.reason,
    }


def _dispatch(case: Fixture, tmp_path: Path) -> Fixture:
    operation = case["operation"]
    if operation == "validate_contract":
        return _validated_observation(case)
    if operation == "normalize_agent_output":
        return _normalized_observation(case)
    if operation == "resolve_artifact":
        return _artifact_observation(case, tmp_path)
    if operation == "validate_transition":
        return _transition_observation(case)
    if operation == "project_legacy":
        return _projection_observation(case)
    raise AssertionError(f"unhandled fixture operation: {operation}")


def test_consumes_the_versioned_fixture_bytes_from_the_installed_package() -> None:
    assert list_conformance_fixtures() == (
        "contract-cases.json",
        "normalization-cases.json",
    )
    for name in list_conformance_fixtures():
        payload = read_conformance_fixture(name)
        assert hashlib.sha256(payload).hexdigest() == FIXTURE_DIGESTS[name]
        assert json.loads(payload) == load_conformance_fixture(name)
        assert load_conformance_fixture(name)["fixture_version"] == "1.0"


@pytest.mark.parametrize("case", CONTRACT_CASES, ids=lambda case: case["name"])
def test_contract_fixture_agreement(case: Fixture, tmp_path: Path) -> None:
    expected = case["expected"]
    if expected["status"] == "accept":
        assert _dispatch(case, tmp_path) == expected["value"]
        return
    with pytest.raises(ERROR_TYPES[expected["error_type"]]) as error:
        _dispatch(case, tmp_path)
    assert str(error.value) == expected["message"]


@pytest.mark.parametrize("case", NORMALIZATION_CASES, ids=lambda case: case["name"])
def test_normalization_fixture_agreement_is_exact_and_deterministic(
    case: Fixture,
) -> None:
    first = normalize_agent_output(case["writer_view"], case["host_invocation"])
    second = normalize_agent_output(case["writer_view"], case["host_invocation"])
    assert json.loads(first.canonical_bytes) == case["expected_canonical_output"]
    assert first.canonical_bytes == second.canonical_bytes


def test_fixture_plan_maps_and_aggregates_without_semantic_reinterpretation(
    graph: MikadoGraph, tmp_path: Path
) -> None:
    case = next(
        case
        for case in CONTRACT_CASES
        if case["operation"] == "validate_contract"
        and case["schema_uri"].endswith("/curd-plan")
        and case["expected"]["status"] == "accept"
    )
    plan = validate_contract(
        case["input"], case["schema_uri"], supported_version=_version(case)
    ).value
    assert isinstance(plan, CurdPlan)
    imported = import_milknado(plan)

    assert imported.semantic_plan == plan
    assert imported.source_plan_ref.plan_id == plan.plan_id
    assert imported.source_plan_ref.revision == plan.revision
    assert imported.source_plan_ref.digest == plan.digest
    expected_refs = {curd.curd_id: canonical_digest(curd) for curd in plan.curds}
    assert {
        item.source_curd_ref.curd_id: item.source_curd_ref.digest for item in imported.items
    } == expected_refs

    change_ids = tuple(item.change.id for item in imported.items)
    batch_plan = BatchPlan(
        batches=(Batch(index=0, change_ids=change_ids, depends_on=()),),
        spread_report=(),
        solver_status="FEASIBLE",
    )
    applied = apply_milknado_plan_to_graph(graph, batch_plan, imported)
    node_id = applied.task_node_ids[0]
    outcome = encode_milknado_physical_outcome(
        tuple(
            MilknadoCriterionOutcome(
                curd.curd_id,
                criterion.criterion_id,
                CriterionDisposition.PASSED,
                None,
            )
            for curd in plan.curds
            for criterion in curd.criteria
        )
    )
    run_id = f"conformance-{node_id}"
    graph.start_run(
        run_id,
        node_id,
        log_path="run.log",
        started_at=datetime.now(UTC).isoformat(),
        timeout_seconds=None,
    )
    graph.deposit_run_message(run_id, "result", outcome, datetime.now(UTC).isoformat())
    graph.finish_run(
        run_id,
        RunResult(
            status="done",
            exit_code=0,
            timed_out=False,
            ended_at=datetime.now(UTC).isoformat(),
        ),
    )
    graph.set_todo_status(node_id, NodeStatus.RUNNING)
    graph.set_todo_status(node_id, NodeStatus.DONE)
    results = collect_milknado_results(
        graph,
        imported,
        applied,
        artifact_directory=tmp_path / "artifacts",
    )

    assert len(results) == len(plan.curds)
    for result, curd in zip(results, plan.curds, strict=True):
        assert result.source_plan_ref == imported.source_plan_ref
        assert result.source_curd_ref.curd_id == curd.curd_id
        assert result.source_curd_ref.digest == canonical_digest(curd)
        assert result.expected_criterion_ids == tuple(
            criterion.criterion_id for criterion in curd.criteria
        )
        assert tuple(row.criterion_id for row in result.criterion_results) == (
            result.expected_criterion_ids
        )
        assert all(row.disposition.value == "passed" for row in result.criterion_results)
