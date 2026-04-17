from __future__ import annotations

import pytest

from milknado.domains.planning.budget import (
    ConfidenceLevel,
    DynamicMultiplier,
    ThresholdBand,
    apply_confidence_buffer,
    apply_multiplier,
    classify_band,
    has_write_pressure,
    validate_manifest_budgets,
)
from milknado.domains.planning.manifest import (
    AtomTokenBudget,
    PlanAtom,
    PlanManifest,
)


def _atom(
    atom_id: str,
    read: int = 0,
    write: int = 0,
    total: int = 0,
    split: bool = False,
    with_budget: bool = True,
) -> PlanAtom:
    budget = AtomTokenBudget(
        estimated_read_tokens=read,
        estimated_write_tokens=write,
        estimated_total_tokens=total,
        split_required=split,
    ) if with_budget else None
    return PlanAtom(
        id=atom_id, description=atom_id, depends_on=[], files=[],
        token_budget=budget,
    )


def _manifest(*atoms: PlanAtom) -> PlanManifest:
    return PlanManifest(manifest_version="milknado.plan.v1", atoms=list(atoms))


class TestClassifyBand:
    @pytest.mark.parametrize(
        ("total", "expected"),
        [
            (0, ThresholdBand.MERGE),
            (24_999, ThresholdBand.MERGE),
            (25_000, ThresholdBand.OPTIMAL),
            (39_999, ThresholdBand.OPTIMAL),
            (40_000, ThresholdBand.TIGHT),
            (49_999, ThresholdBand.TIGHT),
            (50_000, ThresholdBand.SPLIT_RECOMMENDED),
            (64_999, ThresholdBand.SPLIT_RECOMMENDED),
            (65_000, ThresholdBand.SPLIT_REQUIRED),
            (1_000_000, ThresholdBand.SPLIT_REQUIRED),
        ],
    )
    def test_boundaries(self, total: int, expected: ThresholdBand) -> None:
        assert classify_band(total) == expected


class TestApplyMultiplier:
    @pytest.mark.parametrize(
        ("multiplier", "expected"),
        [
            (DynamicMultiplier.MINIMAL, 8_000),
            (DynamicMultiplier.TYPICAL, 10_000),
            (DynamicMultiplier.HEAVY, 13_000),
            (DynamicMultiplier.RETRY, 15_000),
        ],
    )
    def test_factors(self, multiplier: DynamicMultiplier, expected: int) -> None:
        assert apply_multiplier(10_000, multiplier) == expected


class TestApplyConfidenceBuffer:
    @pytest.mark.parametrize(
        ("confidence", "expected"),
        [
            (ConfidenceLevel.HIGH, 10_000),
            (ConfidenceLevel.MEDIUM, 10_500),
            (ConfidenceLevel.LOW, 11_000),
        ],
    )
    def test_buffers(self, confidence: ConfidenceLevel, expected: int) -> None:
        assert apply_confidence_buffer(10_000, confidence) == expected


class TestHasWritePressure:
    def test_below_threshold(self) -> None:
        budget = AtomTokenBudget(0, 3_900, 10_000, False)
        assert has_write_pressure(budget) is False

    def test_at_threshold(self) -> None:
        budget = AtomTokenBudget(0, 4_000, 10_000, False)
        assert has_write_pressure(budget) is False

    def test_above_threshold(self) -> None:
        budget = AtomTokenBudget(0, 4_100, 10_000, False)
        assert has_write_pressure(budget) is True

    def test_zero_total(self) -> None:
        budget = AtomTokenBudget(0, 100, 0, False)
        assert has_write_pressure(budget) is False


class TestValidateManifestBudgets:
    def test_merge_band_yields_no_violations(self) -> None:
        manifest = _manifest(_atom("A1", total=20_000, write=1000))
        assert validate_manifest_budgets(manifest) == []

    def test_optimal_band_yields_no_violations(self) -> None:
        manifest = _manifest(_atom("A1", total=30_000, write=1000))
        assert validate_manifest_budgets(manifest) == []

    def test_tight_band_yields_no_violations(self) -> None:
        manifest = _manifest(_atom("A1", total=45_000, write=1000))
        assert validate_manifest_budgets(manifest) == []

    def test_split_recommended_flagged(self) -> None:
        manifest = _manifest(_atom("A1", total=55_000, write=1000))
        violations = validate_manifest_budgets(manifest)
        assert [v.kind for v in violations] == ["band_split_recommended"]

    def test_split_required_flagged(self) -> None:
        manifest = _manifest(_atom("A1", total=70_000, write=1000))
        violations = validate_manifest_budgets(manifest)
        assert [v.kind for v in violations] == ["band_split_required"]

    def test_missing_budget_is_flagged(self) -> None:
        manifest = _manifest(_atom("A1", with_budget=False))
        violations = validate_manifest_budgets(manifest)
        assert [v.kind for v in violations] == ["missing_budget"]

    def test_write_pressure_flagged_at_41_percent(self) -> None:
        manifest = _manifest(_atom("A1", total=20_000, write=8_200))
        violations = validate_manifest_budgets(manifest)
        assert any(v.kind == "write_pressure" for v in violations)

    def test_no_write_pressure_at_39_percent(self) -> None:
        manifest = _manifest(_atom("A1", total=20_000, write=7_800))
        violations = validate_manifest_budgets(manifest)
        assert not any(v.kind == "write_pressure" for v in violations)

    def test_heavy_multiplier_promotes_band(self) -> None:
        # 45k (tight at typical) * 1.3 = 58.5k -> split_recommended
        manifest = _manifest(_atom("A1", total=45_000, write=1000))
        violations = validate_manifest_budgets(
            manifest, multiplier=DynamicMultiplier.HEAVY,
        )
        assert any(v.kind == "band_split_recommended" for v in violations)

    def test_low_confidence_buffer_promotes_band(self) -> None:
        # 48k * 1.10 = 52.8k -> split_recommended
        manifest = _manifest(_atom("A1", total=48_000, write=1000))
        violations = validate_manifest_budgets(
            manifest, confidence=ConfidenceLevel.LOW,
        )
        assert any(v.kind == "band_split_recommended" for v in violations)

    def test_self_flagged_split_when_host_disagrees(self) -> None:
        manifest = _manifest(_atom("A1", total=20_000, write=1000, split=True))
        violations = validate_manifest_budgets(manifest)
        assert any(v.kind == "self_flagged_split" for v in violations)

    def test_self_flagged_split_silent_when_host_agrees(self) -> None:
        manifest = _manifest(_atom("A1", total=60_000, write=1000, split=True))
        violations = validate_manifest_budgets(manifest)
        assert all(v.kind != "self_flagged_split" for v in violations)

    def test_never_raises_on_empty_manifest(self) -> None:
        manifest = PlanManifest(manifest_version="milknado.plan.v1", atoms=[])
        assert validate_manifest_budgets(manifest) == []
