from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TypeAlias

from easy_cheese_schemas import (
    CurdPlan,
    SemanticCurd,
    SourceCurdRef,
    SourcePlanRef,
    canonical_digest,
    supported_version_for,
    validate_contract,
    validate_curd_plan,
)

from milknado.domains.batching import FileChange
from milknado.domains.planning.manifest import MANIFEST_VERSION, PlanChangeManifest

CurdPlanInput: TypeAlias = CurdPlan | bytes | str | Mapping[str, object]
PHYSICAL_OUTCOME_SCHEMA = "milknado.easy-cheese-physical-outcome.v1"


@dataclass(frozen=True)
class MilknadoPlanItem:
    change: FileChange
    source_curd_ref: SourceCurdRef

    def __post_init__(self) -> None:
        if not isinstance(self.change, FileChange):
            raise ValueError("Milknado plan items must contain a FileChange")
        if not isinstance(self.source_curd_ref, SourceCurdRef):
            raise ValueError("Milknado plan items must contain a SourceCurdRef")


@dataclass(frozen=True)
class MilknadoPlanV2:
    manifest: PlanChangeManifest
    source_plan_ref: SourcePlanRef
    semantic_plan: CurdPlan
    items: tuple[MilknadoPlanItem, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, PlanChangeManifest):
            raise ValueError("Milknado plan must contain a PlanChangeManifest")
        if not isinstance(self.source_plan_ref, SourcePlanRef):
            raise ValueError("Milknado plan must contain a SourcePlanRef")
        if not isinstance(self.semantic_plan, CurdPlan):
            raise ValueError("Milknado plan must contain a CurdPlan")
        if not isinstance(self.items, tuple):
            raise ValueError("Milknado plan items must be a tuple")
        if any(not isinstance(item, MilknadoPlanItem) for item in self.items):
            raise ValueError("Milknado plan items must be MilknadoPlanItem values")

        expected_plan_ref = SourcePlanRef(
            self.semantic_plan.plan_id,
            self.semantic_plan.revision,
            self.semantic_plan.digest,
        )
        if self.source_plan_ref != expected_plan_ref:
            raise ValueError("source plan reference does not match semantic plan")
        if self.manifest.goal != self.semantic_plan.plan_id:
            raise ValueError("manifest goal does not match semantic plan")
        if self.manifest.goal_summary != self.semantic_plan.objective:
            raise ValueError("manifest summary does not match semantic plan")
        if self.manifest.manifest_version != MANIFEST_VERSION:
            raise ValueError("manifest version does not match deterministic projection")
        if self.manifest.spec_path is not None:
            raise ValueError("manifest spec path does not match deterministic projection")
        if self.manifest.new_relationships:
            raise ValueError("manifest relationships do not match deterministic projection")

        expected_items = _projected_items(self.semantic_plan)
        if self.items != expected_items:
            raise ValueError("items must exactly match the deterministic semantic projection")
        expected_changes = tuple(item.change for item in expected_items)
        if self.manifest.changes != expected_changes:
            raise ValueError(
                "manifest changes must exactly match the deterministic semantic projection"
            )


def _validated_plan(raw: CurdPlanInput) -> CurdPlan:
    if isinstance(raw, CurdPlan):
        return validate_curd_plan(raw)
    artifact = validate_contract(raw, CurdPlan, supported_version_for(CurdPlan))
    if not isinstance(artifact.value, CurdPlan):
        raise TypeError("Easy Cheese validation returned a non-CurdPlan value")
    return validate_curd_plan(artifact.value)


def _change_id(plan_id: str, curd_ordinal: int, path_ordinal: int) -> str:
    return f"{plan_id}/curd/{curd_ordinal}/path/{path_ordinal}"


def _scope_paths(curd: SemanticCurd) -> tuple[str, ...]:
    paths = tuple(sorted(curd.scope.paths))
    if len(paths) != len(set(paths)):
        raise ValueError(f"curd {curd.curd_id!r} contains duplicate scope paths")
    return paths


def _scope_excluded_paths(curd: SemanticCurd) -> tuple[str, ...]:
    paths = tuple(sorted(curd.scope.excluded_paths))
    if len(paths) != len(set(paths)):
        raise ValueError(f"curd {curd.curd_id!r} contains duplicate excluded paths")
    return paths


def _change_description(curd: SemanticCurd) -> str:
    lines = [curd.outcome, "", "Acceptance criteria:"]
    for criterion in curd.criteria:
        lines.extend(
            (
                f"- {curd.curd_id}/{criterion.criterion_id}: {criterion.description}",
                f"  Check: {criterion.check}",
            )
        )
    lines.extend(
        (
            "",
            "Structured result contract:",
            "Deposit one JSON result row for every unique curd/criterion pair assigned "
            "to the node.",
            f'Top-level schema: "{PHYSICAL_OUTCOME_SCHEMA}".',
            "Each row has exactly curd_id, criterion_id, disposition, and reason; "
            "disposition is passed, failed, blocked, or skipped. Passed requires a "
            "null reason; every other disposition requires a non-blank reason.",
        )
    )
    return "\n".join(lines)


def _projected_items(plan: CurdPlan) -> tuple[MilknadoPlanItem, ...]:
    source_refs = {
        curd.curd_id: SourceCurdRef(curd.curd_id, canonical_digest(curd)) for curd in plan.curds
    }
    changes_by_curd: dict[str, tuple[FileChange, ...]] = {}
    for curd_ordinal, curd in enumerate(plan.curds, start=1):
        excluded_paths = _scope_excluded_paths(curd)
        changes_by_curd[curd.curd_id] = tuple(
            FileChange(
                id=_change_id(plan.plan_id, curd_ordinal, path_ordinal),
                path=path,
                edit_kind="modify",
                description=_change_description(curd),
                excluded_paths=excluded_paths,
            )
            for path_ordinal, path in enumerate(_scope_paths(curd), start=1)
        )

    change_order = {
        change.id: order
        for order, change in enumerate(
            change for curd in changes_by_curd.values() for change in curd
        )
    }
    completion_by_curd = {
        curd.curd_id: changes_by_curd[curd.curd_id][-1].id for curd in plan.curds
    }
    projected: list[MilknadoPlanItem] = []
    for curd in plan.curds:
        curd_changes = changes_by_curd[curd.curd_id]
        external_dependencies = tuple(
            completion_by_curd[dependency_id] for dependency_id in curd.dependencies
        )
        first_id = curd_changes[0].id
        middle_ids = tuple(change.id for change in curd_changes[1:-1])
        for ordinal, change in enumerate(curd_changes):
            if len(curd_changes) == 1 or ordinal == 0:
                dependency_ids = external_dependencies
            elif ordinal == len(curd_changes) - 1:
                dependency_ids = (first_id, *middle_ids)
            else:
                dependency_ids = (first_id,)
            depends_on = tuple(sorted(dependency_ids, key=change_order.__getitem__))
            projected.append(
                MilknadoPlanItem(
                    replace(change, depends_on=depends_on),
                    source_refs[curd.curd_id],
                )
            )
    return tuple(projected)


def _project_plan(plan: CurdPlan) -> MilknadoPlanV2:
    items = _projected_items(plan)
    manifest = PlanChangeManifest(
        manifest_version=MANIFEST_VERSION,
        goal=plan.plan_id,
        goal_summary=plan.objective,
        spec_path=None,
        changes=tuple(item.change for item in items),
        new_relationships=(),
    )
    return MilknadoPlanV2(
        manifest=manifest,
        source_plan_ref=SourcePlanRef(plan.plan_id, plan.revision, plan.digest),
        semantic_plan=plan,
        items=items,
    )


def import_milknado(raw: CurdPlanInput) -> MilknadoPlanV2:
    """Validate a semantic CurdPlan, then project it into physical plan-v2 changes."""
    return _project_plan(_validated_plan(raw))


__all__ = [
    "MilknadoPlanItem",
    "MilknadoPlanV2",
    "PHYSICAL_OUTCOME_SCHEMA",
    "import_milknado",
]
