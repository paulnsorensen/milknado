from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from milknado.domains.planning.manifest import AtomTokenBudget, PlanManifest


class ThresholdBand(StrEnum):
    MERGE = "merge"
    OPTIMAL = "optimal"
    TIGHT = "tight"
    SPLIT_RECOMMENDED = "split_recommended"
    SPLIT_REQUIRED = "split_required"


class DynamicMultiplier(StrEnum):
    MINIMAL = "minimal"
    TYPICAL = "typical"
    HEAVY = "heavy"
    RETRY = "retry"


MULTIPLIER_FACTORS: dict[DynamicMultiplier, float] = {
    DynamicMultiplier.MINIMAL: 0.8,
    DynamicMultiplier.TYPICAL: 1.0,
    DynamicMultiplier.HEAVY: 1.3,
    DynamicMultiplier.RETRY: 1.5,
}


class ConfidenceLevel(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


CONFIDENCE_BUFFER: dict[ConfidenceLevel, float] = {
    ConfidenceLevel.HIGH: 0.0,
    ConfidenceLevel.MEDIUM: 0.05,
    ConfidenceLevel.LOW: 0.10,
}

WRITE_PRESSURE_THRESHOLD = 0.40


@dataclass(frozen=True)
class BudgetViolation:
    atom_id: str
    kind: str
    detail: str


def classify_band(total_tokens: int) -> ThresholdBand:
    if total_tokens < 25_000:
        return ThresholdBand.MERGE
    if total_tokens < 40_000:
        return ThresholdBand.OPTIMAL
    if total_tokens < 50_000:
        return ThresholdBand.TIGHT
    if total_tokens < 65_000:
        return ThresholdBand.SPLIT_RECOMMENDED
    return ThresholdBand.SPLIT_REQUIRED


def apply_multiplier(
    total_tokens: int, multiplier: DynamicMultiplier,
) -> int:
    return int(total_tokens * MULTIPLIER_FACTORS[multiplier])


def apply_confidence_buffer(
    total_tokens: int, confidence: ConfidenceLevel,
) -> int:
    return int(total_tokens * (1.0 + CONFIDENCE_BUFFER[confidence]))


def has_write_pressure(budget: AtomTokenBudget) -> bool:
    total = budget.estimated_total_tokens
    if total <= 0:
        return False
    return (budget.estimated_write_tokens / total) > WRITE_PRESSURE_THRESHOLD


def validate_manifest_budgets(
    manifest: PlanManifest,
    multiplier: DynamicMultiplier = DynamicMultiplier.TYPICAL,
    confidence: ConfidenceLevel = ConfidenceLevel.HIGH,
) -> list[BudgetViolation]:
    violations: list[BudgetViolation] = []
    for atom in manifest.atoms:
        if atom.token_budget is None:
            violations.append(BudgetViolation(
                atom_id=atom.id,
                kind="missing_budget",
                detail="atom has no token_budget; confidence forced to low",
            ))
            continue
        violations.extend(_check_atom(atom.id, atom.token_budget, multiplier, confidence))
    return violations


def _check_atom(
    atom_id: str,
    budget: AtomTokenBudget,
    multiplier: DynamicMultiplier,
    confidence: ConfidenceLevel,
) -> list[BudgetViolation]:
    violations: list[BudgetViolation] = []
    buffered = apply_confidence_buffer(budget.estimated_total_tokens, confidence)
    adjusted = apply_multiplier(buffered, multiplier)
    band = classify_band(adjusted)
    if band in (ThresholdBand.SPLIT_RECOMMENDED, ThresholdBand.SPLIT_REQUIRED):
        violations.append(BudgetViolation(
            atom_id=atom_id,
            kind=f"band_{band.value}",
            detail=f"adjusted total {adjusted} in {band.value} band",
        ))
    if budget.split_required and band not in (
        ThresholdBand.SPLIT_RECOMMENDED, ThresholdBand.SPLIT_REQUIRED,
    ):
        violations.append(BudgetViolation(
            atom_id=atom_id,
            kind="self_flagged_split",
            detail="atom self-flagged split_required; host classification disagrees",
        ))
    if has_write_pressure(budget):
        violations.append(BudgetViolation(
            atom_id=atom_id,
            kind="write_pressure",
            detail=(
                f"write_tokens {budget.estimated_write_tokens} "
                f"exceeds {WRITE_PRESSURE_THRESHOLD:.0%} of total "
                f"{budget.estimated_total_tokens}"
            ),
        ))
    return violations
