from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest
from easy_cheese_schemas import (
    ArtifactRef,
    BoundedScope,
    ContractValidationError,
    ContractVersion,
    Criterion,
    CriterionDisposition,
    CriterionResultWriterView,
    CurdPlan,
    CurdResultWriterView,
    EvidenceKind,
    EvidenceRef,
    IdentityAction,
    IdentityLineage,
    SemanticCurd,
    canonical_bytes,
    canonical_digest,
)

from milknado.app.curd_results import normalize_curd_result
from milknado.domains.common import NodeKind, NodeSpec
from milknado.domains.graph import MikadoGraph
from milknado.domains.planning import admit_curd_plan
from milknado.mcp.curd import milknado_curd_plan_admit
from tests.graph_helpers import graph_conn

_SCHEMA_URI = "https://schemas.easy-cheese.dev/curd-plan"
_VERSION = ContractVersion(schema_uri=_SCHEMA_URI, major="1", minor="0")


def _curd(
    curd_id: str,
    paths: tuple[str, ...],
    dependencies: tuple[str, ...] = (),
    criterion_ids: tuple[str, ...] = ("criterion",),
) -> SemanticCurd:
    return SemanticCurd(
        curd_id=curd_id,
        outcome=f"deliver {curd_id}",
        scope=BoundedScope(paths=paths),
        inputs=(),
        outputs=(f"{curd_id}-output",),
        dependencies=dependencies,
        criteria=tuple(
            Criterion(
                criterion_id=f"{curd_id}/{criterion_id}",
                description=f"{criterion_id} is delivered",
                check=f"check {criterion_id}",
            )
            for criterion_id in criterion_ids
        ),
        lineage=IdentityLineage(identity_action=IdentityAction.NEW),
    )


def _plan(*curds: SemanticCurd) -> CurdPlan:
    return CurdPlan.signed(
        contract_version=_VERSION,
        plan_id="admission-plan",
        revision=1,
        objective="admit the plan",
        curds=curds,
    )


def test_admission_creates_tasks_claims_files_and_orients_dependencies(
    tmp_path: Path,
) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    try:
        plan = _plan(
            _curd("prepare", ("src/prepare.py",)),
            _curd("verify", ("tests/test_verify.py",), dependencies=("prepare",)),
        )
        admission = admit_curd_plan(plan, graph)

        goal = graph.get_node(admission.goal_id)
        assert goal is not None and goal.kind is NodeKind.GOAL
        tasks = [graph.get_node(node_id) for node_id in admission.curd_node_ids.values()]
        assert all(task is not None and task.kind is NodeKind.TASK for task in tasks)
        assert [task.description for task in tasks if task is not None] == [
            "deliver prepare",
            "deliver verify",
        ]
        assert graph.files.for_node(admission.curd_node_ids["prepare"]) == ["src/prepare.py"]
        assert graph.files.for_node(admission.curd_node_ids["verify"]) == ["tests/test_verify.py"]
        edge = cast(
            tuple[int, int] | None,
            graph_conn(graph)
            .execute(
                "SELECT parent_id, child_id FROM edges WHERE parent_id = ?",
                (admission.curd_node_ids["verify"],),
            )
            .fetchone(),
        )
        assert edge is not None
        assert tuple(edge) == (
            admission.curd_node_ids["verify"],
            admission.curd_node_ids["prepare"],
        )
        with pytest.raises(TypeError):
            cast(dict[str, int], admission.curd_node_ids)["extra"] = 99
    finally:
        graph.close()


def test_admission_persists_one_plan_payload_and_each_curd_identity(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    try:
        plan = _plan(_curd("one", ("one.py",)), _curd("two", ("two.py",)))
        admission = admit_curd_plan(plan, graph)

        stored_plan = graph.get_curd_plan(plan.plan_id, plan.revision)
        assert stored_plan is not None
        assert stored_plan.digest == plan.digest
        assert stored_plan.canonical_bytes.endswith(b"\n")
        assert graph_conn(graph).execute("SELECT COUNT(*) FROM curd_plans").fetchone()[0] == 1
        for curd in plan.curds:
            stored_node = graph.get_curd_node(admission.curd_node_ids[curd.curd_id])
            assert stored_node is not None
            assert stored_node.curd_id == curd.curd_id
            assert stored_node.curd_digest == canonical_digest(curd)
    finally:
        graph.close()


def test_admission_uses_an_existing_parent_and_rejects_a_missing_parent(
    tmp_path: Path,
) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    try:
        parent = graph.add_node("parent", spec=NodeSpec(kind=NodeKind.GOAL))
        plan = _plan(_curd("one", ("one.py",)))

        with pytest.raises(ValueError, match="parent_id 999 not found"):
            _ = admit_curd_plan(plan, graph, parent_id=999)

        admission = admit_curd_plan(plan, graph, parent_id=parent.id)
        task = graph.get_node(admission.curd_node_ids["one"])
        assert admission.goal_id == parent.id
        assert task is not None and task.parent_id == parent.id
    finally:
        graph.close()


def test_curd_plan_storage_is_idempotent_and_detects_conflicts(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    try:
        plan = _plan(_curd("one", ("one.py",)))
        admission = admit_curd_plan(plan, graph)
        stored = graph.get_curd_plan(plan.plan_id, plan.revision)
        assert stored is not None

        graph.record_curd_plan(
            plan.plan_id,
            plan.revision,
            plan.digest,
            stored.canonical_bytes,
        )
        with pytest.raises(ValueError, match="already recorded with different content"):
            graph.record_curd_plan(
                plan.plan_id,
                plan.revision,
                plan.digest,
                b"different",
            )

        assert graph.get_curd_plan("missing", 1) is None
        assert graph.get_curd_node(max(admission.curd_node_ids.values()) + 1) is None
    finally:
        graph.close()


def test_result_normalization_reorders_criteria_and_keeps_execution_private(
    tmp_path: Path,
) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    try:
        plan = _plan(_curd("one", ("one.py",), criterion_ids=("first", "second")))
        admission = admit_curd_plan(plan, graph)
        artifact = ArtifactRef(
            artifact_id="evidence-artifact",
            role="verification",
            uri="file://evidence",
            digest="sha256:" + "1" * 64,
            size_bytes=1,
            media_type="text/plain",
        )
        evidence = EvidenceRef(
            evidence_id="evidence",
            kind=EvidenceKind.VERIFICATION,
            artifact=artifact,
        )
        writer = CurdResultWriterView(
            criterion_results=(
                CriterionResultWriterView(
                    criterion_id="one/second",
                    disposition=CriterionDisposition.PASSED,
                    evidence_keys=("evidence",),
                ),
                CriterionResultWriterView(
                    criterion_id="one/first",
                    disposition=CriterionDisposition.PASSED,
                    evidence_keys=("evidence",),
                ),
            )
        )

        result = normalize_curd_result(
            admission.curd_node_ids["one"],
            writer,
            graph,
            {"evidence": evidence},
            {},
            provenance_refs=("host/provenance",),
        )

        assert result.expected_criterion_ids == ("one/first", "one/second")
        assert tuple(row.criterion_id for row in result.criterion_results) == (
            "one/first",
            "one/second",
        )
        assert result.provenance_refs == ("host/provenance",)
        assert not hasattr(result, "runtime_refs")
    finally:
        graph.close()


def test_result_normalization_rejects_missing_or_corrupt_identity(tmp_path: Path) -> None:
    graph = MikadoGraph(tmp_path / "graph.db")
    writer = CurdResultWriterView(
        criterion_results=(
            CriterionResultWriterView(
                criterion_id="one/criterion-1",
                disposition=CriterionDisposition.PASSED,
                evidence_keys=("evidence",),
            ),
        )
    )
    try:
        with pytest.raises(ValueError, match="has no admitted Curd identity"):
            _ = normalize_curd_result(999, writer, graph, {}, {})

        plan = _plan(_curd("one", ("one.py",)))
        admission = admit_curd_plan(plan, graph)
        _ = graph_conn(graph).execute(
            "UPDATE curd_plans SET digest = ? WHERE plan_id = ? AND revision = ?",
            ("sha256:" + "0" * 64, plan.plan_id, plan.revision),
        )
        graph_conn(graph).commit()
        with pytest.raises(ValueError, match="digest does not match"):
            _ = normalize_curd_result(
                admission.curd_node_ids["one"],
                writer,
                graph,
                {},
                {},
            )

        _ = graph_conn(graph).execute("PRAGMA foreign_keys = OFF")
        _ = graph_conn(graph).execute("DELETE FROM curd_plans")
        graph_conn(graph).commit()
        _ = graph_conn(graph).execute("PRAGMA foreign_keys = ON")
        with pytest.raises(ValueError, match="Curd plan .* is missing"):
            _ = normalize_curd_result(
                admission.curd_node_ids["one"],
                writer,
                graph,
                {},
                {},
            )
    finally:
        graph.close()


def test_mcp_admits_a_valid_plan(tmp_path: Path) -> None:
    response = milknado_curd_plan_admit(
        canonical_bytes(_plan(_curd("one", ("one.py",)))).decode(),
        project_root=str(tmp_path),
    )
    goal_id = cast(int, response["goal_id"])
    node_ids = cast(Mapping[str, int], response["curd_node_ids"])
    assert goal_id > 0
    assert set(node_ids) == {"one"}
    assert node_ids["one"] > 0


def test_mcp_rejects_malformed_plan_before_graph_open() -> None:
    with pytest.raises(ContractValidationError):
        _ = milknado_curd_plan_admit('{"objective": "missing required fields"}')
