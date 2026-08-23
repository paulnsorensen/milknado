from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

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
    SourceCurdRef,
    SourcePlanRef,
    canonical_digest,
)

import milknado.app.easy_cheese_bindings as bindings
import milknado.app.easy_cheese_evidence as evidence
import milknado.app.easy_cheese_execution as execution
import milknado.app.easy_cheese_outcomes as outcomes
import milknado.app.easy_cheese_results as results
import milknado.domains.planning.easy_cheese as easy_cheese
import milknado.domains.planning.manifest as manifest_module
from milknado.app.easy_cheese_execution import AppliedMilknadoPlan
from milknado.domains.batching import Batch, BatchPlan
from milknado.domains.graph import MikadoGraph
from milknado.domains.planning import MANIFEST_VERSION, import_milknado

_SCHEMA_URI = "https://schemas.easy-cheese.dev/curd-plan"
_VERSION = ContractVersion(schema_uri=_SCHEMA_URI, major="1", minor="0")


def _curd(*paths: str) -> SemanticCurd:
    return SemanticCurd(
        curd_id="curd-a",
        outcome="deliver curd-a",
        scope=BoundedScope(paths=paths, excluded_paths=()),
        inputs=(),
        outputs=("output",),
        dependencies=(),
        criteria=(Criterion(criterion_id="criterion-a", description="done", check="check"),),
        lineage=IdentityLineage(identity_action=IdentityAction.NEW),
    )


def _plan(*paths: str) -> CurdPlan:
    unsigned = {
        "contract_version": _VERSION,
        "plan_id": "plan-a",
        "revision": 1,
        "objective": "deliver plan-a",
        "curds": (_curd(*paths),),
        "context": None,
        "parent_plan_ref": None,
    }
    return CurdPlan(
        contract_version=_VERSION,
        plan_id="plan-a",
        revision=1,
        digest=canonical_digest(unsigned),
        objective="deliver plan-a",
        curds=(_curd(*paths),),
    )


def _imported(*paths: str):
    return import_milknado(_plan(*paths))


def _batch_plan(*batches: Batch) -> BatchPlan:
    return BatchPlan(batches=batches, spread_report=(), solver_status="OPTIMAL")


def _bypass_dataclass(cls: type[object], **values: object) -> object:
    instance = object.__new__(cls)
    for name, value in values.items():
        object.__setattr__(instance, name, value)
    return instance


def _imported(*paths: str):
    return import_milknado(_plan(*paths))


def _batch_plan(*batches: Batch) -> BatchPlan:
    return BatchPlan(batches=batches, spread_report=(), solver_status="OPTIMAL")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_plan_ref", "wrong", "applied plan must contain a SourcePlanRef"),
        ("batch_plan", object(), "applied plan must contain a BatchPlan"),
        ("task_node_ids", [1], "task_node_ids must be a tuple"),
        ("task_node_ids", (), "task node count must match batch count"),
        ("task_node_ids", (0,), "task node IDs must be positive integers"),
        ("task_node_ids", (True,), "task node IDs must be positive integers"),
        ("task_node_ids", (1, 1), "task node IDs must be unique"),
        ("goal_node_id", 0, "goal_node_id must be a positive integer or None"),
        ("goal_node_id", True, "goal_node_id must be a positive integer or None"),
    ],
)
def test_applied_plan_rejects_invalid_invariants(field: str, value: object, message: str) -> None:
    imported = _imported("src/a.py")
    batch_plan = _batch_plan(
        Batch(index=0, change_ids=(imported.items[0].change.id,), depends_on=())
    )
    values = {
        "source_plan_ref": imported.source_plan_ref,
        "batch_plan": batch_plan,
        "task_node_ids": (1,),
        "goal_node_id": None,
    }
    values[field] = value
    if field == "task_node_ids" and value == (1, 1):
        values["batch_plan"] = _batch_plan(
            Batch(index=0, change_ids=(imported.items[0].change.id,), depends_on=()),
            Batch(index=1, change_ids=(imported.items[0].change.id,), depends_on=(0,)),
        )
    candidate = _bypass_dataclass(AppliedMilknadoPlan, **values)
    with pytest.raises(ValueError, match=message):
        AppliedMilknadoPlan.__post_init__(candidate)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("paths", "batch_plan", "message"),
    [
        (
            ("src/a.py",),
            _batch_plan(
                Batch(
                    index=0,
                    change_ids=("plan-a/curd/1/path/1", "plan-a/curd/1/path/1"),
                    depends_on=(),
                )
            ),
            "batch plan change IDs must be unique",
        ),
        (
            ("src/a.py",),
            _batch_plan(Batch(index=0, change_ids=("unknown",), depends_on=())),
            r"batch/plan change coverage mismatch: "
            r"missing=\['plan-a/curd/1/path/1'\], unknown=\['unknown'\]",
        ),
        (
            ("src/a.py", "src/b.py"),
            _batch_plan(
                Batch(index=0, change_ids=("plan-a/curd/1/path/1",), depends_on=()),
                Batch(index=0, change_ids=("plan-a/curd/1/path/2",), depends_on=()),
            ),
            "batch indexes must be unique",
        ),
    ],
)
def test_apply_plan_rejects_invalid_batch_coverage(
    tmp_path: Path, paths: tuple[str, ...], batch_plan: BatchPlan, message: str
) -> None:
    imported = _imported(*paths)
    graph = MikadoGraph(tmp_path / "coverage.db")
    with pytest.raises(ValueError, match=message):
        execution.apply_milknado_plan_to_graph(graph, batch_plan, imported)


def test_apply_plan_with_parent_keeps_goal_out_of_task_ids(tmp_path: Path) -> None:
    imported = _imported("src/a.py")
    batch = Batch(index=0, change_ids=(imported.items[0].change.id,), depends_on=())
    graph = MikadoGraph(tmp_path / "parent.db")
    parent = graph.add_node("existing goal")

    application = execution.apply_milknado_plan_to_graph(
        graph, _batch_plan(batch), imported, parent_id=parent.id
    )

    assert application.goal_node_id is None
    assert len(application.task_node_ids) == 1
    assert graph.get_node(application.task_node_ids[0]) is not None


def test_apply_plan_rejects_wrong_plan_type(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="plan must be a MilknadoPlanV2"):
        execution.apply_milknado_plan_to_graph(
            MikadoGraph(tmp_path / "types.db"),
            _batch_plan(),
            object(),  # type: ignore[arg-type]
        )


def test_apply_plan_rejects_wrong_batch_plan_type(tmp_path: Path) -> None:
    imported = _imported("src/a.py")
    with pytest.raises(ValueError, match="batch_plan must be a BatchPlan"):
        execution.apply_milknado_plan_to_graph(
            MikadoGraph(tmp_path / "batch-type.db"),
            object(),
            imported,  # type: ignore[arg-type]
        )


def _valid_application(tmp_path: Path) -> tuple[MikadoGraph, object, AppliedMilknadoPlan]:
    imported = _imported("src/a.py", "src/b.py")
    batch = Batch(
        index=0,
        change_ids=tuple(item.change.id for item in imported.items),
        depends_on=(),
    )
    graph = MikadoGraph(tmp_path / "binding.db")
    application = execution.apply_milknado_plan_to_graph(graph, _batch_plan(batch), imported)
    return graph, imported, application


def _variant(application: AppliedMilknadoPlan, **changes: object) -> AppliedMilknadoPlan:
    values = {
        "source_plan_ref": application.source_plan_ref,
        "batch_plan": application.batch_plan,
        "task_node_ids": application.task_node_ids,
        "goal_node_id": application.goal_node_id,
    }
    values.update(changes)
    return _bypass_dataclass(AppliedMilknadoPlan, **values)  # type: ignore[return-value]


def test_index_plan_items_rejects_unknown_source_curd() -> None:
    imported = _imported("src/a.py")
    item = imported.items[0]
    unknown_item = easy_cheese.MilknadoPlanItem(
        item.change,
        SourceCurdRef("unknown-curd", item.source_curd_ref.digest),
    )
    candidate = _plan_variant(imported, items=(unknown_item,))
    with pytest.raises(ValueError, match="plan item refers to unknown source curd 'unknown-curd'"):
        bindings._index_plan_items(candidate)  # type: ignore[arg-type]


def test_index_plan_items_rejects_duplicate_change_id() -> None:
    imported = _imported("src/a.py", "src/b.py")
    candidate = _plan_variant(imported, items=(imported.items[0], imported.items[0]))
    with pytest.raises(ValueError, match="plan contains duplicate change ID"):
        bindings._index_plan_items(candidate)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("mutator", "exception", "message"),
    [
        (
            lambda graph, plan, app: object(),
            TypeError,
            "application must be an AppliedMilknadoPlan",
        ),
        (
            lambda graph, plan, app: _variant(
                app,
                source_plan_ref=SourcePlanRef("other", 1, plan.source_plan_ref.digest),
            ),
            ValueError,
            "applied plan source reference does not match imported plan",
        ),
        (
            lambda graph, plan, app: _variant(
                app,
                batch_plan=_batch_plan(
                    Batch(index=0, change_ids=(plan.items[0].change.id,) * 2, depends_on=())
                ),
            ),
            ValueError,
            "applied batch plan contains duplicate change IDs",
        ),
        (
            lambda graph, plan, app: _variant(
                app,
                batch_plan=_batch_plan(Batch(index=0, change_ids=("unknown",), depends_on=())),
            ),
            ValueError,
            "applied batch coverage mismatch",
        ),
        (
            lambda graph, plan, app: _variant(app, task_node_ids=(99999,)),
            ValueError,
            "applied plan refers to nonexistent Mikado node 99999",
        ),
        (
            lambda graph, plan, app: _variant(
                app,
                batch_plan=_batch_plan(
                    Batch(
                        index=9,
                        change_ids=tuple(item.change.id for item in plan.items),
                        depends_on=(),
                    )
                ),
            ),
            ValueError,
            "node .* batch index .* does not match 9",
        ),
        (
            lambda graph, plan, app: (
                graph.update_node(app.task_node_ids[0], description="wrong"),
                app,
            )[1],
            ValueError,
            "node .* description does not match its applied batch",
        ),
        (
            lambda graph, plan, app: (
                graph.set_file_ownership(app.task_node_ids[0], ["wrong.py"]),
                app,
            )[1],
            ValueError,
            "file ownership does not match its applied changes",
        ),
    ],
)
def test_bind_execution_rejects_invalid_application(
    tmp_path: Path, mutator: object, exception: type[Exception], message: str
) -> None:
    graph, plan, application = _valid_application(tmp_path)
    candidate = mutator(graph, plan, application)  # type: ignore[operator]
    with pytest.raises(exception, match=message):
        bindings.bind_milknado_execution(graph, plan, candidate)


def test_prepare_artifact_directory_rejects_file(tmp_path: Path) -> None:
    file_path = tmp_path / "artifact-file"
    file_path.write_text("not a directory", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact directory is not usable"):
        evidence.prepare_artifact_directory(file_path)


def test_prepare_artifact_directory_rejects_non_pathlike() -> None:
    with pytest.raises(ValueError, match="artifact directory is not usable: None"):
        evidence.prepare_artifact_directory(None)  # type: ignore[arg-type]


def test_persist_evidence_rejects_conflicting_bytes(tmp_path: Path) -> None:
    first = evidence.node_evidence(1, "done", "result", tmp_path)
    destination = Path(first.artifact.uri.removeprefix("file://"))
    destination.write_bytes(b"conflicting")

    with pytest.raises(ValueError, match="could not persist evidence at"):
        evidence.node_evidence(1, "done", "result", tmp_path)


def test_change_model_rejects_duplicate_excluded_paths() -> None:
    with pytest.raises(ValueError, match="excluded paths must be unique"):
        manifest_module._ChangeModel(
            id="change",
            path="src/change.py",
            description="change",
            excluded_paths=["src/other.py", "src/other.py"],
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("change", object(), "Milknado plan items must contain a FileChange"),
        ("source_curd_ref", object(), "Milknado plan items must contain a SourceCurdRef"),
    ],
)
def test_plan_item_rejects_wrong_types(field: str, value: object, message: str) -> None:
    imported = _imported("src/a.py")
    item = imported.items[0]
    with pytest.raises(ValueError, match=message):
        easy_cheese.MilknadoPlanItem(
            value if field == "change" else item.change,
            value if field == "source_curd_ref" else item.source_curd_ref,
        )


def _plan_variant(plan: object, **changes: object) -> object:
    values = {
        "manifest": plan.manifest,  # type: ignore[attr-defined]
        "source_plan_ref": plan.source_plan_ref,  # type: ignore[attr-defined]
        "semantic_plan": plan.semantic_plan,  # type: ignore[attr-defined]
        "items": plan.items,  # type: ignore[attr-defined]
    }
    values.update(changes)
    return _bypass_dataclass(easy_cheese.MilknadoPlanV2, **values)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("manifest", object(), "Milknado plan must contain a PlanChangeManifest"),
        ("source_plan_ref", object(), "Milknado plan must contain a SourcePlanRef"),
        ("semantic_plan", object(), "Milknado plan must contain a CurdPlan"),
        ("items", [], "Milknado plan items must be a tuple"),
        ("items", (object(),), "Milknado plan items must be MilknadoPlanItem values"),
    ],
)
def test_plan_rejects_wrong_component_types(field: str, value: object, message: str) -> None:
    imported = _imported("src/a.py")
    values = {
        "manifest": imported.manifest,
        "source_plan_ref": imported.source_plan_ref,
        "semantic_plan": imported.semantic_plan,
        "items": imported.items,
    }
    values[field] = value
    candidate = _bypass_dataclass(easy_cheese.MilknadoPlanV2, **values)
    with pytest.raises(ValueError, match=message):
        easy_cheese.MilknadoPlanV2.__post_init__(candidate)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda plan: _plan_variant(
                plan,
                source_plan_ref=SourcePlanRef("other", 1, plan.source_plan_ref.digest),
            ),
            "source plan reference does not match semantic plan",
        ),
        (
            lambda plan: _plan_variant(plan, manifest=replace(plan.manifest, goal="other")),
            "manifest goal does not match semantic plan",
        ),
        (
            lambda plan: _plan_variant(
                plan, manifest=replace(plan.manifest, goal_summary="other")
            ),
            "manifest summary does not match semantic plan",
        ),
        (
            lambda plan: _plan_variant(
                plan, manifest=replace(plan.manifest, manifest_version="other")
            ),
            "manifest version does not match deterministic projection",
        ),
        (
            lambda plan: _plan_variant(plan, manifest=replace(plan.manifest, spec_path="spec.md")),
            "manifest spec path does not match deterministic projection",
        ),
        (
            lambda plan: _plan_variant(
                plan, manifest=replace(plan.manifest, new_relationships=(object(),))
            ),
            "manifest relationships do not match deterministic projection",
        ),
        (
            lambda plan: _plan_variant(plan, items=()),
            "items must exactly match the deterministic semantic projection",
        ),
        (
            lambda plan: _plan_variant(plan, manifest=replace(plan.manifest, changes=())),
            "manifest changes must exactly match the deterministic semantic projection",
        ),
    ],
)
def test_plan_rejects_inconsistent_projection(mutator: object, message: str) -> None:
    candidate = mutator(_imported("src/a.py"))  # type: ignore[operator]
    with pytest.raises(ValueError, match=message):
        easy_cheese.MilknadoPlanV2.__post_init__(candidate)  # type: ignore[arg-type]


def test_scope_duplicate_guards_are_explicit() -> None:
    curd = SimpleNamespace(scope=SimpleNamespace(paths=("src/a.py", "src/a.py")))
    with pytest.raises(ValueError, match="curd 'curd-a' contains duplicate scope paths"):
        easy_cheese._scope_paths(SimpleNamespace(curd_id="curd-a", scope=curd.scope))

    curd = SimpleNamespace(
        curd_id="curd-a",
        scope=SimpleNamespace(excluded_paths=("src/a.py", "src/a.py")),
    )
    with pytest.raises(ValueError, match="curd 'curd-a' contains duplicate excluded paths"):
        easy_cheese._scope_excluded_paths(curd)


@pytest.mark.parametrize(
    "case",
    [
        (
            "",
            "criterion",
            CriterionDisposition.PASSED,
            None,
            ValueError,
            "curd_id must be non-empty text",
        ),
        (
            "curd",
            "",
            CriterionDisposition.PASSED,
            None,
            ValueError,
            "criterion_id must be non-empty text",
        ),
        (
            "curd",
            "criterion",
            "passed",
            None,
            TypeError,
            "disposition must be a CriterionDisposition",
        ),
        (
            "curd",
            "criterion",
            CriterionDisposition.FAILED,
            " ",
            ValueError,
            "non-passed criterion outcomes require a non-blank reason",
        ),
        (
            "curd",
            "criterion",
            CriterionDisposition.FAILED,
            "x" * 4097,
            ValueError,
            "criterion outcome reason must be at most 4096 characters",
        ),
    ],
)
def test_criterion_outcome_rejects_invalid_values(
    case: tuple[str, str, object, str | None, type[Exception], str],
) -> None:
    curd_id, criterion_id, disposition, reason, exception, message = case
    with pytest.raises(exception, match=message):
        outcomes.MilknadoCriterionOutcome(  # type: ignore[arg-type]
            curd_id, criterion_id, disposition, reason
        )


def test_criterion_outcome_rejects_reason_on_passed() -> None:
    with pytest.raises(ValueError, match="passed criterion outcomes must not include a reason"):
        outcomes.MilknadoCriterionOutcome(
            "curd", "criterion", CriterionDisposition.PASSED, "reason"
        )


@pytest.mark.parametrize(
    ("criterion_results", "exception", "message"),
    [
        ([], ValueError, "criterion_results must be a non-empty tuple"),
        ((), ValueError, "criterion_results must be a non-empty tuple"),
        ((object(),), TypeError, "criterion_results must contain MilknadoCriterionOutcome values"),
        (
            (
                outcomes.MilknadoCriterionOutcome(
                    "curd", "criterion", CriterionDisposition.PASSED, None
                ),
            )
            * 2,
            ValueError,
            "criterion outcome pairs must be unique",
        ),
    ],
)
def test_encode_outcome_rejects_invalid_rows(
    criterion_results: object, exception: type[Exception], message: str
) -> None:
    with pytest.raises(exception, match=message):
        outcomes.encode_milknado_physical_outcome(criterion_results)  # type: ignore[arg-type]


def _physical_payload(rows: object, *, schema: str = outcomes.PHYSICAL_OUTCOME_SCHEMA) -> str:
    return json.dumps({"schema": schema, "criterion_results": rows})


@pytest.mark.parametrize(
    ("result", "message"),
    [
        ("not-json", None),
        (_physical_payload([], schema="other"), None),
        (
            json.dumps({"schema": outcomes.PHYSICAL_OUTCOME_SCHEMA}),
            "top-level object expected exactly",
        ),
        (
            json.dumps(
                {
                    "schema": outcomes.PHYSICAL_OUTCOME_SCHEMA,
                    "criterion_results": [],
                    "extra": True,
                }
            ),
            "top-level object expected exactly",
        ),
        (_physical_payload([1]), "criterion_results[0] must be an object"),
        (
            _physical_payload(
                [
                    {
                        "curd_id": "curd",
                        "criterion_id": "criterion",
                        "disposition": 1,
                        "reason": None,
                    }
                ]
            ),
            "criterion_results[0].disposition must be text",
        ),
        (
            _physical_payload(
                [
                    {
                        "curd_id": "curd",
                        "criterion_id": "criterion",
                        "disposition": "unsupported",
                        "reason": None,
                    }
                ]
            ),
            "criterion_results[0].disposition is unsupported",
        ),
        (
            '{"schema":"milknado.easy-cheese-physical-outcome.v1","criterion_results":'
            '[{"curd_id":"curd","curd_id":"curd","criterion_id":"criterion",'
            '"disposition":"passed","reason":null}]}',
            "duplicate JSON keys in criterion_results[0]",
        ),
        (
            '{"schema":"milknado.easy-cheese-physical-outcome.v1",'
            '"schema":"milknado.easy-cheese-physical-outcome.v1","criterion_results":[]}',
            "duplicate JSON keys in top-level object",
        ),
        (
            _physical_payload(
                [
                    {
                        "curd_id": "curd",
                        "criterion_id": "criterion",
                        "disposition": "passed",
                        "reason": None,
                    },
                    {
                        "curd_id": "curd",
                        "criterion_id": "criterion",
                        "disposition": "passed",
                        "reason": None,
                    },
                ]
            ),
            "duplicate criterion outcome pair",
        ),
    ],
)
def test_parse_physical_outcome_rejects_malformed_payload(
    result: str, message: str | None
) -> None:
    if message is None:
        assert outcomes.parse_milknado_physical_outcome(result, ()) == {}
        return
    with pytest.raises(ValueError, match=re.escape(message)):
        outcomes.parse_milknado_physical_outcome(result, (("curd", "criterion"),))


def test_status_and_reason_rejection_helpers() -> None:
    with pytest.raises(ValueError, match="Mikado node has an invalid status"):
        results._status_value(SimpleNamespace(status=SimpleNamespace(value=1)))
    with pytest.raises(ValueError, match="unsupported Mikado node status 'unknown'"):
        results._status_class("unknown")
    assert (
        results._reason_for_node(3, "failed", None)
        == "physical node 3 is failed without a deposited result"
    )
    truncated = results._bounded_reason(["x" * (outcomes.PHYSICAL_OUTCOME_REASON_MAX + 100)])
    assert len(truncated) <= outcomes.PHYSICAL_OUTCOME_REASON_MAX
    assert truncated.endswith(" [truncated]")


def test_normalization_rejects_non_curd_result(monkeypatch: pytest.MonkeyPatch) -> None:
    curd = _curd("curd-a", "src/a.py")
    fact = results._NodeFact(
        status="done",
        state="done",
        result=None,
        evidence=SimpleNamespace(evidence_id="evidence"),  # type: ignore[arg-type]
        outcomes={
            ("curd-a", "criterion-curd-a"): outcomes.MilknadoCriterionOutcome(
                "curd-a", "criterion-curd-a", CriterionDisposition.PASSED, None
            )
        },
    )
    context = SimpleNamespace(
        plan=SimpleNamespace(source_plan_ref=SourcePlanRef("plan-a", 1, "sha256:" + "0" * 64)),
        bound=SimpleNamespace(
            items_by_curd={
                "curd-a": SimpleNamespace(
                    items=(("change", 1),), source_curd_ref=SimpleNamespace()
                )
            }
        ),
        facts={1: fact},
        contract_version=SimpleNamespace(),
    )
    monkeypatch.setattr(results, "normalize_agent_value", lambda *_args: object())
    with pytest.raises(TypeError, match="normalization returned a non-CurdResult value"):
        results._normalize_curd_result(context, curd, 1)


def test_collect_results_rejects_missing_supported_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph, plan, application = _valid_application(tmp_path)
    monkeypatch.setattr(results, "supported_version_for", lambda _value: None)
    with pytest.raises(ValueError, match="does not expose a supported CurdResult version"):
        results.collect_milknado_results(
            graph, plan, application, artifact_directory=tmp_path / "artifacts"
        )


assert MANIFEST_VERSION == "milknado.plan.v2"
assert SourceCurdRef
