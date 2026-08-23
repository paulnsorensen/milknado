from __future__ import annotations

import copy
import json
from collections.abc import Callable
from dataclasses import replace

import pytest
from easy_cheese_schemas import (
    BoundedScope,
    ContractValidationError,
    ContractVersion,
    Criterion,
    CurdPlan,
    IdentityAction,
    IdentityLineage,
    SemanticCurd,
    SourceCurdRef,
    SourcePlanRef,
    canonical_bytes,
    canonical_digest,
)

import milknado.domains.planning.easy_cheese as easy_cheese_importer
from milknado.domains.planning import (
    MANIFEST_VERSION,
    MilknadoPlanItem,
    MilknadoPlanV2,
    PlanChangeManifest,
    import_milknado,
)

_SCHEMA_URI = "https://schemas.easy-cheese.dev/curd-plan"
_VERSION = ContractVersion(schema_uri=_SCHEMA_URI, major="1", minor="0")
_DIGEST = "sha256:" + "0" * 64


def _curd(
    curd_id: str,
    paths: tuple[str, ...],
    dependencies: tuple[str, ...] = (),
    excluded_paths: tuple[str, ...] = (),
) -> SemanticCurd:
    return SemanticCurd(
        curd_id=curd_id,
        outcome=f"deliver {curd_id}",
        scope=BoundedScope(paths=paths, excluded_paths=excluded_paths),
        inputs=(),
        outputs=(f"{curd_id}-output",),
        dependencies=dependencies,
        criteria=(
            Criterion(
                criterion_id=f"criterion-{curd_id}",
                description=f"{curd_id} is delivered",
                check=f"check {curd_id}",
            ),
        ),
        lineage=IdentityLineage(identity_action=IdentityAction.NEW),
    )


def _plan(
    curds: tuple[SemanticCurd, ...],
    *,
    plan_id: str = "demo-plan",
    objective: str = "deliver the demo",
) -> CurdPlan:
    unsigned = {
        "contract_version": _VERSION,
        "plan_id": plan_id,
        "revision": 1,
        "objective": objective,
        "curds": curds,
        "context": None,
        "parent_plan_ref": None,
    }
    return CurdPlan(
        contract_version=_VERSION,
        plan_id=plan_id,
        revision=1,
        digest=canonical_digest(unsigned),
        objective=objective,
        curds=curds,
    )


def _plan_with_dependency() -> CurdPlan:
    return _plan(
        (
            _curd("prepare", ("src/z.py", "src/a.py")),
            _curd(
                "verify",
                ("src/a.py", "tests/test_demo.py"),
                dependencies=("prepare",),
            ),
        )
    )


def _raw(plan: CurdPlan) -> dict[str, object]:
    return json.loads(canonical_bytes(plan))


def test_import_maps_paths_items_and_source_references() -> None:
    plan = _plan_with_dependency()

    imported = import_milknado(plan)
    assert imported.semantic_plan == plan

    assert imported.source_plan_ref == SourcePlanRef("demo-plan", 1, plan.digest)
    assert imported.manifest.manifest_version == MANIFEST_VERSION
    assert imported.manifest.goal == "demo-plan"
    assert imported.manifest.goal_summary == "deliver the demo"
    assert imported.manifest.spec_path is None
    assert imported.manifest.new_relationships == ()
    assert [change.path for change in imported.manifest.changes] == [
        "src/a.py",
        "src/z.py",
        "src/a.py",
        "tests/test_demo.py",
    ]
    assert all(change.edit_kind == "modify" for change in imported.manifest.changes)
    descriptions = [change.description for change in imported.manifest.changes]
    assert descriptions[0] == descriptions[1]
    assert descriptions[2] == descriptions[3]
    assert descriptions[0].startswith("deliver prepare\n\nAcceptance criteria:")
    assert "- prepare/criterion-prepare: prepare is delivered" in descriptions[0]
    assert "Check: check prepare" in descriptions[0]
    assert 'Top-level schema: "milknado.easy-cheese-physical-outcome.v1".' in descriptions[0]

    prepare_digest = canonical_digest(plan.curds[0])
    verify_digest = canonical_digest(plan.curds[1])
    assert [item.source_curd_ref for item in imported.items] == [
        SourceCurdRef("prepare", prepare_digest),
        SourceCurdRef("prepare", prepare_digest),
        SourceCurdRef("verify", verify_digest),
        SourceCurdRef("verify", verify_digest),
    ]
    assert [item.change for item in imported.items] == list(imported.manifest.changes)


def test_source_curd_digests_are_computed_once_per_validation_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan_with_dependency()
    calls: list[str] = []
    original_digest = easy_cheese_importer.canonical_digest

    def counted_digest(value: object) -> str:
        if isinstance(value, SemanticCurd):
            calls.append(value.curd_id)
        return original_digest(value)

    monkeypatch.setattr(easy_cheese_importer, "canonical_digest", counted_digest)
    easy_cheese_importer.import_milknado(plan)

    assert calls == ["prepare", "verify", "prepare", "verify"]


def test_dependency_topology_is_complete_and_deterministic() -> None:
    imported = import_milknado(_plan_with_dependency())
    changes = imported.manifest.changes
    prepare_completion = changes[1].id

    assert changes[0].depends_on == ()
    assert changes[1].depends_on == (changes[0].id,)
    assert changes[2].depends_on == (prepare_completion,)
    assert changes[3].depends_on == (changes[2].id,)
    assert all("src/" not in change.id for change in changes)


def test_exclusions_are_copied_to_each_physical_change() -> None:
    plan = _plan(
        (
            _curd(
                "bounded",
                ("src/z.py", "src/a.py"),
                excluded_paths=("vendor/generated.py", "tests/fixtures.py"),
            ),
        )
    )

    imported = import_milknado(plan)

    assert [change.path for change in imported.manifest.changes] == ["src/a.py", "src/z.py"]
    assert all(
        change.excluded_paths == ("tests/fixtures.py", "vendor/generated.py")
        for change in imported.manifest.changes
    )
    assert all(change.path not in change.excluded_paths for change in imported.manifest.changes)


def test_dense_dependency_topology_has_linear_edge_count() -> None:
    upstream_paths = tuple(f"src/upstream/{index:03d}.py" for index in range(60))
    downstream_paths = tuple(f"src/downstream/{index:03d}.py" for index in range(60))
    plan = _plan(
        (
            _curd("upstream", upstream_paths),
            _curd("downstream", downstream_paths, dependencies=("upstream",)),
        )
    )

    imported = import_milknado(plan)
    changes = imported.manifest.changes
    edge_count = sum(len(change.depends_on) for change in changes)

    assert edge_count == 2 * len(changes) - 3 * 2 + 1
    upstream_completion_index = len(upstream_paths) - 1
    downstream_gate_index = len(upstream_paths)
    upstream_completion = changes[upstream_completion_index].id
    downstream_gate = changes[downstream_gate_index].id
    assert changes[downstream_gate_index].depends_on == (upstream_completion,)
    assert changes[downstream_gate_index + 1].depends_on == (downstream_gate,)
    assert changes[-1].depends_on == (
        downstream_gate,
        *(change.id for change in changes[downstream_gate_index + 1 : -1]),
    )


def test_repeated_projection_and_canonical_inputs_are_identical() -> None:
    plan = _plan_with_dependency()

    typed = import_milknado(plan)
    canonical_json = canonical_bytes(plan)

    assert typed == import_milknado(canonical_json)
    assert typed == import_milknado(canonical_json.decode())
    assert typed == import_milknado(_raw(plan))


def test_same_path_in_sequential_curds_is_preserved() -> None:
    plan = _plan(
        (
            _curd("first", ("src/shared.py",)),
            _curd("second", ("src/shared.py",)),
        )
    )

    imported = import_milknado(plan)

    assert [change.path for change in imported.manifest.changes] == [
        "src/shared.py",
        "src/shared.py",
    ]
    assert imported.manifest.changes[0].id != imported.manifest.changes[1].id


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(
            lambda raw: raw["contract_version"].update({"major": "2"}),
            id="unsupported-major",
        ),
        pytest.param(
            lambda raw: raw["contract_version"].update(
                {"schema_uri": "https://schemas.example.invalid/curd-plan"}
            ),
            id="wrong-schema",
        ),
        pytest.param(
            lambda raw: raw.update({"unexpected": True}),
            id="unknown-field",
        ),
        pytest.param(
            lambda raw: raw.update({"digest": _DIGEST}),
            id="bad-digest",
        ),
        pytest.param(
            lambda raw: raw["curds"][0]["scope"].update({"paths": ["/absolute.py"]}),
            id="absolute-path",
        ),
        pytest.param(
            lambda raw: raw["curds"][0]["scope"].update({"paths": ["../outside.py"]}),
            id="traversal-path",
        ),
        pytest.param(
            lambda raw: raw["curds"][0]["scope"].update(
                {"paths": ["src/duplicate.py", "src/duplicate.py"]}
            ),
            id="duplicate-path-within-curd",
        ),
    ],
)
def test_invalid_canonical_input_is_rejected_before_projection(
    mutate: Callable[[dict[str, object]], None],
) -> None:
    raw = copy.deepcopy(_raw(_plan_with_dependency()))
    mutate(raw)

    with pytest.raises(ContractValidationError):
        import_milknado(raw)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        pytest.param(
            lambda raw: raw["contract_version"].update({"minor": "1"}),
            "future contract minor",
            id="future-minor",
        ),
        pytest.param(
            lambda raw: raw["curds"][1].update(
                {"curd_id": raw["curds"][0]["curd_id"], "dependencies": []}
            ),
            "curd_id .* must be unique",
            id="duplicate-curd-id",
        ),
        pytest.param(
            lambda raw: raw["curds"][1].update({"dependencies": ["missing"]}),
            "references undeclared curd",
            id="unknown-dependency",
        ),
        pytest.param(
            lambda raw: (
                raw["curds"][0].update({"dependencies": ["verify"]}),
                raw["curds"][1].update({"dependencies": ["prepare"]}),
            ),
            "curd dependencies must be acyclic",
            id="cyclic-dependencies",
        ),
    ],
)
def test_malformed_semantic_plan_is_rejected_before_projection(
    mutate: Callable[[dict[str, object]], object],
    message: str,
) -> None:
    raw = copy.deepcopy(_raw(_plan_with_dependency()))
    mutate(raw)

    with pytest.raises(ContractValidationError, match=message):
        import_milknado(raw)


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param(None, id="null"),
        pytest.param([], id="array"),
        pytest.param("not-json", id="invalid-json"),
    ],
)
def test_non_object_plan_input_is_rejected_at_validation_boundary(
    raw: str | bytes | dict[str, object],
) -> None:
    with pytest.raises(ContractValidationError):
        import_milknado(raw)


def test_plan_contract_has_no_scheduler_fields() -> None:
    plan = _plan_with_dependency()

    scheduler_fields = ("batches", "waves", "worktrees", "retries", "scheduler")
    assert not any(hasattr(plan, name) for name in scheduler_fields)


def _physical_semantic_plan() -> CurdPlan:
    return _plan((_curd("curd", ("curd.py",)),), plan_id="plan", objective="objective")


def _rebuild_plan(
    imported: MilknadoPlanV2,
    *,
    manifest: PlanChangeManifest | None = None,
    source_plan_ref: SourcePlanRef | None = None,
    items: tuple[MilknadoPlanItem, ...] | None = None,
) -> MilknadoPlanV2:
    return MilknadoPlanV2(
        manifest=manifest or imported.manifest,
        source_plan_ref=source_plan_ref or imported.source_plan_ref,
        semantic_plan=imported.semantic_plan,
        items=items if items is not None else imported.items,
    )


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "reordered"])
def test_plan_v2_rejects_items_outside_deterministic_projection(mutation: str) -> None:
    imported = import_milknado(
        _plan(
            (_curd("curd", ("first.py", "second.py")),),
            plan_id="plan",
            objective="objective",
        )
    )
    if mutation == "missing":
        items = imported.items[:-1]
    elif mutation == "duplicate":
        items = (*imported.items, imported.items[-1])
    else:
        items = tuple(reversed(imported.items))

    with pytest.raises(ValueError, match="deterministic semantic projection"):
        _rebuild_plan(imported, items=items)


def test_plan_v2_rejects_swapped_source_curd_references() -> None:
    imported = import_milknado(
        _plan(
            (_curd("first", ("first.py",)), _curd("second", ("second.py",))),
            plan_id="plan",
            objective="objective",
        )
    )
    first, second = imported.items
    swapped = (
        MilknadoPlanItem(first.change, second.source_curd_ref),
        MilknadoPlanItem(second.change, first.source_curd_ref),
    )

    with pytest.raises(ValueError, match="deterministic semantic projection"):
        _rebuild_plan(imported, items=swapped)


def test_plan_v2_rejects_manifest_change_outside_deterministic_projection() -> None:
    imported = import_milknado(_physical_semantic_plan())
    changed = replace(imported.manifest.changes[0], path="different.py")
    manifest = replace(imported.manifest, changes=(changed,))

    with pytest.raises(ValueError, match="manifest changes.*deterministic semantic projection"):
        _rebuild_plan(imported, manifest=manifest)


def test_plan_v2_rejects_source_plan_ref_mismatch() -> None:
    semantic_plan = _physical_semantic_plan()
    imported = import_milknado(semantic_plan)

    with pytest.raises(ValueError, match="source plan reference"):
        _rebuild_plan(
            imported,
            source_plan_ref=SourcePlanRef("other-plan", 1, semantic_plan.digest),
        )
