from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock

import pytest
from easy_cheese_schemas import (
    BoundedScope,
    ContractVersion,
    Criterion,
    CriterionDisposition,
    CurdPlan,
    IdentityAction,
    IdentityLineage,
    SemanticCurd,
    SourcePlanRef,
    canonical_bytes,
    canonical_digest,
    resolve_artifact,
)

from milknado.app import (
    AppliedMilknadoPlan,
    MilknadoCriterionOutcome,
    apply_milknado_plan_to_graph,
    collect_milknado_results,
    encode_milknado_physical_outcome,
)
from milknado.domains.batching import Batch, BatchPlan
from milknado.domains.common import NodeStatus, RunResult
from milknado.domains.graph import MikadoGraph
from milknado.domains.planning import MilknadoPlanV2, import_milknado

_SCHEMA_URI = "https://schemas.easy-cheese.dev/curd-plan"
_VERSION = ContractVersion(schema_uri=_SCHEMA_URI, major="1", minor="0")


def _curd(
    curd_id: str, paths: tuple[str, ...], criterion_ids: tuple[str, ...] = ()
) -> SemanticCurd:
    criterion_ids = criterion_ids or (f"criterion-{curd_id}",)
    return SemanticCurd(
        curd_id=curd_id,
        outcome=f"deliver {curd_id}",
        scope=BoundedScope(paths=paths),
        inputs=(),
        outputs=(f"{curd_id}-output",),
        dependencies=(),
        criteria=tuple(
            Criterion(
                criterion_id=criterion_id,
                description=f"{criterion_id} is delivered",
                check=f"check {criterion_id}",
            )
            for criterion_id in criterion_ids
        ),
        lineage=IdentityLineage(identity_action=IdentityAction.NEW),
    )


def _plan(*curds: SemanticCurd) -> CurdPlan:
    unsigned = {
        "contract_version": _VERSION,
        "plan_id": "result-plan",
        "revision": 1,
        "objective": "aggregate physical outcomes",
        "curds": curds,
        "context": None,
        "parent_plan_ref": None,
    }
    return CurdPlan(
        contract_version=_VERSION,
        plan_id="result-plan",
        revision=1,
        digest=canonical_digest(unsigned),
        objective="aggregate physical outcomes",
        curds=curds,
    )


def _batch_plan(
    imported: MilknadoPlanV2,
    groups: tuple[tuple[str, ...], ...] | None = None,
) -> BatchPlan:
    groups = groups or (tuple(item.change.id for item in imported.items),)
    return BatchPlan(
        batches=tuple(
            Batch(index=index, change_ids=change_ids, depends_on=())
            for index, change_ids in enumerate(groups)
        ),
        spread_report=(),
        solver_status="FEASIBLE",
    )


def _apply(
    graph: MikadoGraph,
    imported: MilknadoPlanV2,
    groups: tuple[tuple[str, ...], ...] | None = None,
) -> AppliedMilknadoPlan:
    return apply_milknado_plan_to_graph(graph, _batch_plan(imported, groups), imported)


def _set_status(graph: MikadoGraph, node_id: int, status: NodeStatus) -> None:
    if status is NodeStatus.DONE:
        graph.set_todo_status(node_id, NodeStatus.RUNNING)
    graph.set_todo_status(node_id, status)


def _deposit_result(graph: MikadoGraph, node_id: int, body: str, *, failed: bool = False) -> None:
    run_id = f"run-{node_id}-{datetime.now(UTC).timestamp()}"
    graph.start_run(
        run_id,
        node_id,
        log_path="run.log",
        started_at=datetime.now(UTC).isoformat(),
        timeout_seconds=None,
    )
    graph.deposit_run_message(run_id, "result", body, datetime.now(UTC).isoformat())
    graph.finish_run(
        run_id,
        RunResult(
            status="failed" if failed else "done",
            exit_code=1 if failed else 0,
            timed_out=False,
            ended_at=datetime.now(UTC).isoformat(),
        ),
    )


def _outcomes(
    imported: MilknadoPlanV2,
    change_ids: tuple[str, ...],
    disposition: CriterionDisposition = CriterionDisposition.PASSED,
    reason: str | None = None,
) -> tuple[MilknadoCriterionOutcome, ...]:
    curd_ids = {
        item.source_curd_ref.curd_id for item in imported.items if item.change.id in change_ids
    }
    return tuple(
        MilknadoCriterionOutcome(
            curd.curd_id,
            criterion.criterion_id,
            disposition,
            reason,
        )
        for curd in imported.semantic_plan.curds
        if curd.curd_id in curd_ids
        for criterion in curd.criteria
    )


def _deposit_outcomes(
    graph: MikadoGraph,
    node_id: int,
    outcomes: tuple[MilknadoCriterionOutcome, ...],
) -> None:
    _deposit_result(graph, node_id, encode_milknado_physical_outcome(outcomes))


@pytest.fixture()
def artifact_directory(tmp_path: Path) -> Path:
    return tmp_path / "artifacts"


def test_complete_result_preserves_exact_criteria_and_node_evidence(
    graph: MikadoGraph, artifact_directory: Path
) -> None:
    imported = import_milknado(_plan(_curd("one", ("does/not/exist.py",), ("first", "second"))))
    applied = _apply(graph, imported)
    node_id = applied.task_node_ids[0]
    _deposit_outcomes(
        graph, node_id, _outcomes(imported, applied.batch_plan.batches[0].change_ids)
    )
    _set_status(graph, node_id, NodeStatus.DONE)

    result = collect_milknado_results(
        graph, imported, applied, artifact_directory=artifact_directory
    )[0]

    assert result.expected_criterion_ids == ("first", "second")
    assert tuple(row.criterion_id for row in result.criterion_results) == ("first", "second")
    assert result.source_plan_ref == imported.source_plan_ref
    assert result.source_curd_ref == imported.items[0].source_curd_ref
    assert result.disposition.value == "passed"
    assert result.result_id == "result-plan/revision/1/result/1"
    assert result.contract_version.schema_uri == "https://schemas.easy-cheese.dev/curd-result"
    evidence = result.criterion_results[0].evidence[0]
    deposited = encode_milknado_physical_outcome(
        _outcomes(imported, applied.batch_plan.batches[0].change_ids)
    )
    payload = {"node_id": node_id, "status": "done", "result": deposited}
    resolved = resolve_artifact(
        evidence.artifact, artifact_directory=artifact_directory / "resolved"
    )
    assert Path(resolved.path).read_bytes() == canonical_bytes(payload)
    assert evidence.artifact.digest == canonical_digest(payload)
    fingerprint = evidence.artifact.digest.removeprefix("sha256:")
    assert evidence.evidence_id == f"milknado/node/{node_id}/outcome/{fingerprint}"
    assert evidence.artifact.artifact_id == evidence.evidence_id


def test_done_without_typed_criterion_outcomes_is_blocked(
    graph: MikadoGraph, artifact_directory: Path
) -> None:
    imported = import_milknado(_plan(_curd("one", ("one.py",))))
    applied = _apply(graph, imported)
    node_id = applied.task_node_ids[0]
    _set_status(graph, node_id, NodeStatus.DONE)

    result = collect_milknado_results(
        graph, imported, applied, artifact_directory=artifact_directory
    )[0]

    assert result.disposition.value == "blocked"
    assert result.criterion_results[0].reason == (
        f"physical node {node_id} is done without a typed outcome for one/criterion-one"
    )


def test_physical_fan_out_preserves_order_and_requires_each_node_to_pass(
    graph: MikadoGraph, artifact_directory: Path
) -> None:
    imported = import_milknado(_plan(_curd("fan", ("first.py", "second.py"), ("first", "second"))))
    groups = tuple((item.change.id,) for item in imported.items)
    applied = _apply(graph, imported, groups)
    for batch, node_id in zip(applied.batch_plan.batches, applied.task_node_ids, strict=True):
        _deposit_outcomes(graph, node_id, _outcomes(imported, batch.change_ids))
        _set_status(graph, node_id, NodeStatus.DONE)

    first = collect_milknado_results(
        graph, imported, applied, artifact_directory=artifact_directory
    )
    second = collect_milknado_results(
        graph, imported, applied, artifact_directory=artifact_directory
    )

    assert first == second
    result = first[0]
    assert result.runtime_refs == tuple(
        f"milknado://nodes/{node_id}/outcome" for node_id in applied.task_node_ids
    )
    assert all(row.disposition.value == "passed" for row in result.criterion_results)
    assert all(len(row.evidence) == 2 for row in result.criterion_results)


def test_explicit_criterion_outcomes_can_differ(
    graph: MikadoGraph, artifact_directory: Path
) -> None:
    imported = import_milknado(_plan(_curd("mixed", ("mixed.py",), ("pass", "skip"))))
    applied = _apply(graph, imported)
    node_id = applied.task_node_ids[0]
    _deposit_outcomes(
        graph,
        node_id,
        (
            MilknadoCriterionOutcome("mixed", "pass", CriterionDisposition.PASSED, None),
            MilknadoCriterionOutcome(
                "mixed", "skip", CriterionDisposition.SKIPPED, "not applicable"
            ),
        ),
    )
    _set_status(graph, node_id, NodeStatus.DONE)

    result = collect_milknado_results(
        graph, imported, applied, artifact_directory=artifact_directory
    )[0]

    assert tuple(row.disposition.value for row in result.criterion_results) == (
        "passed",
        "skipped",
    )
    assert result.criterion_results[1].reason == "not applicable"
    assert result.unresolved_work == ("not applicable",)


def test_failed_and_blocked_statuses_are_not_inferred_from_result_text(
    graph: MikadoGraph, artifact_directory: Path
) -> None:
    imported = import_milknado(
        _plan(
            _curd("failed", ("failed.py",)),
            _curd("blocked", ("blocked.py",)),
        )
    )
    groups = tuple((item.change.id,) for item in imported.items)
    applied = _apply(graph, imported, groups)
    failed_id, blocked_id = applied.task_node_ids
    malformed_typed = json.dumps(
        {"schema": "milknado.easy-cheese-physical-outcome.v1", "broken": True}
    )
    _deposit_result(graph, failed_id, malformed_typed, failed=True)
    _set_status(graph, failed_id, NodeStatus.FAILED)
    _set_status(graph, blocked_id, NodeStatus.BLOCKED)

    results = collect_milknado_results(
        graph, imported, applied, artifact_directory=artifact_directory
    )

    assert tuple(result.disposition.value for result in results) == ("failed", "blocked")
    assert results[0].criterion_results[0].reason == (
        f"physical node {failed_id} is failed; physical work remains unresolved"
    )
    assert results[1].criterion_results[0].reason == (
        f"physical node {blocked_id} is blocked; physical work remains unresolved"
    )


def test_done_skipped_signal_produces_skipped_rows_with_reason(
    graph: MikadoGraph, artifact_directory: Path
) -> None:
    imported = import_milknado(_plan(_curd("skipped", ("skipped.py",))))
    applied = _apply(graph, imported)
    node_id = applied.task_node_ids[0]
    reason = "physical executor excluded this change by policy"
    _deposit_outcomes(
        graph,
        node_id,
        _outcomes(
            imported,
            applied.batch_plan.batches[0].change_ids,
            CriterionDisposition.SKIPPED,
            reason,
        ),
    )
    _set_status(graph, node_id, NodeStatus.DONE)

    result = collect_milknado_results(
        graph, imported, applied, artifact_directory=artifact_directory
    )[0]

    assert result.disposition.value == "skipped"
    assert result.unresolved_work == (reason,)
    assert result.criterion_results[0].reason == reason


@pytest.mark.parametrize(
    "body",
    [
        '{"schema":"milknado.easy-cheese-physical-outcome.v1","criterion_results":[]}',
        (
            '{"schema":"milknado.easy-cheese-physical-outcome.v1",'
            '"criterion_results":[{"curd_id":"one","criterion_id":"criterion-one",'
            '"disposition":"passed","reason":"must be null"}]}'
        ),
        (
            '{"schema":"milknado.easy-cheese-physical-outcome.v1",'
            '"criterion_results":[{"curd_id":"wrong","criterion_id":"criterion-one",'
            '"disposition":"passed","reason":null}]}'
        ),
    ],
)
def test_malformed_done_physical_outcome_is_rejected(
    graph: MikadoGraph, artifact_directory: Path, body: str
) -> None:
    imported = import_milknado(_plan(_curd("one", ("one.py",))))
    applied = _apply(graph, imported)
    node_id = applied.task_node_ids[0]
    _deposit_result(graph, node_id, body)
    _set_status(graph, node_id, NodeStatus.DONE)

    with pytest.raises(ValueError, match="malformed physical outcome"):
        collect_milknado_results(graph, imported, applied, artifact_directory=artifact_directory)


def test_pending_nodes_emit_unstarted_criterion_results(
    graph: MikadoGraph, artifact_directory: Path
) -> None:
    imported = import_milknado(_plan(_curd("pending", ("pending.py",))))
    applied = _apply(graph, imported)
    node_id = applied.task_node_ids[0]

    result = collect_milknado_results(
        graph, imported, applied, artifact_directory=artifact_directory
    )[0]

    reason = f"physical node {node_id} is pending; physical work remains unresolved"
    assert result.disposition.value == "blocked"
    assert result.unresolved_work == (reason,)


def test_one_batch_node_can_cover_multiple_curds_deterministically(
    graph: MikadoGraph, artifact_directory: Path
) -> None:
    imported = import_milknado(
        _plan(_curd("first", ("first.py",)), _curd("second", ("second.py",)))
    )
    applied = _apply(graph, imported)
    node_id = applied.task_node_ids[0]
    _deposit_outcomes(
        graph, node_id, _outcomes(imported, applied.batch_plan.batches[0].change_ids)
    )
    _set_status(graph, node_id, NodeStatus.DONE)

    results = collect_milknado_results(
        graph, imported, applied, artifact_directory=artifact_directory
    )

    assert len(results) == 2
    assert all(result.disposition.value == "passed" for result in results)
    assert all(
        result.runtime_refs == (f"milknado://nodes/{node_id}/outcome",) for result in results
    )


def test_result_collection_rejects_source_plan_mismatch(
    graph: MikadoGraph, artifact_directory: Path
) -> None:
    imported = import_milknado(_plan(_curd("one", ("one.py",))))
    applied = _apply(graph, imported)
    wrong = AppliedMilknadoPlan(
        SourcePlanRef("wrong", 1, imported.source_plan_ref.digest),
        applied.batch_plan,
        applied.task_node_ids,
        applied.goal_node_id,
    )

    with pytest.raises(ValueError, match="source reference"):
        collect_milknado_results(graph, imported, wrong, artifact_directory=artifact_directory)


def test_result_collection_rejects_swapped_batch_nodes(
    graph: MikadoGraph, artifact_directory: Path
) -> None:
    imported = import_milknado(_plan(_curd("one", ("one.py", "two.py"))))
    groups = tuple((item.change.id,) for item in imported.items)
    applied = _apply(graph, imported, groups)
    swapped = AppliedMilknadoPlan(
        applied.source_plan_ref,
        applied.batch_plan,
        tuple(reversed(applied.task_node_ids)),
        applied.goal_node_id,
    )

    with pytest.raises(ValueError, match="batch index"):
        collect_milknado_results(graph, imported, swapped, artifact_directory=artifact_directory)


def test_result_collection_rejects_tampered_file_ownership(
    graph: MikadoGraph, artifact_directory: Path
) -> None:
    imported = import_milknado(_plan(_curd("one", ("one.py",))))
    applied = _apply(graph, imported)
    graph.set_file_ownership(applied.task_node_ids[0], ["different.py"])

    with pytest.raises(ValueError, match="file ownership"):
        collect_milknado_results(graph, imported, applied, artifact_directory=artifact_directory)


def test_result_collection_uses_bulk_graph_reads(
    graph: MikadoGraph, artifact_directory: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    imported = import_milknado(_plan(_curd("bulk", ("bulk.py",))))
    applied = _apply(graph, imported)
    get_nodes = Mock(wraps=graph.get_nodes)
    get_ownership = Mock(wraps=graph.get_file_ownership_map)
    latest_results = Mock(wraps=graph.latest_results_for_nodes)
    monkeypatch.setattr(graph, "get_nodes", get_nodes)
    monkeypatch.setattr(graph, "get_file_ownership_map", get_ownership)
    monkeypatch.setattr(graph, "latest_results_for_nodes", latest_results)

    collect_milknado_results(graph, imported, applied, artifact_directory=artifact_directory)

    assert get_nodes.call_count == 1
    assert get_ownership.call_count == 1
    assert latest_results.call_count == 1


def test_content_addressed_evidence_identity_changes_with_latest_result(
    graph: MikadoGraph, artifact_directory: Path
) -> None:
    imported = import_milknado(_plan(_curd("one", ("one.py",))))
    applied = _apply(graph, imported)
    node_id = applied.task_node_ids[0]
    passed = _outcomes(imported, applied.batch_plan.batches[0].change_ids)
    _deposit_outcomes(graph, node_id, passed)
    _set_status(graph, node_id, NodeStatus.DONE)
    first = (
        collect_milknado_results(graph, imported, applied, artifact_directory=artifact_directory)[
            0
        ]
        .criterion_results[0]
        .evidence[0]
    )

    changed = (
        MilknadoCriterionOutcome(
            "one", "criterion-one", CriterionDisposition.SKIPPED, "later policy"
        ),
    )
    _deposit_outcomes(graph, node_id, changed)
    second = (
        collect_milknado_results(graph, imported, applied, artifact_directory=artifact_directory)[
            0
        ]
        .criterion_results[0]
        .evidence[0]
    )

    assert first.evidence_id != second.evidence_id
    assert first.artifact.artifact_id != second.artifact.artifact_id
    assert first.artifact.uri != second.artifact.uri


def test_physical_outcome_encoder_rejects_invalid_reason_policy() -> None:
    with pytest.raises(ValueError, match="must not include a reason"):
        MilknadoCriterionOutcome("curd", "criterion", CriterionDisposition.PASSED, "not allowed")
    with pytest.raises(ValueError, match="require a non-blank reason"):
        MilknadoCriterionOutcome("curd", "criterion", CriterionDisposition.SKIPPED, None)
