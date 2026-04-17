from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from milknado.domains.planning.budget import (
    BudgetViolation,
    ConfidenceLevel,
    DynamicMultiplier,
)
from milknado.domains.planning.manifest import PlanAtom, PlanManifest

CALIBRATION_FILENAME = "calibration.jsonl"


def record_budget_snapshot(
    project_root: Path,
    manifest: PlanManifest,
    violations: list[BudgetViolation],
    multiplier: DynamicMultiplier = DynamicMultiplier.TYPICAL,
    confidence: ConfidenceLevel = ConfidenceLevel.HIGH,
) -> Path:
    milknado_dir = project_root / ".milknado"
    milknado_dir.mkdir(parents=True, exist_ok=True)
    path = milknado_dir / CALIBRATION_FILENAME
    timestamp = datetime.now(UTC).isoformat()

    violations_by_atom: dict[str, list[BudgetViolation]] = {}
    for v in violations:
        violations_by_atom.setdefault(v.atom_id, []).append(v)

    with path.open("a", encoding="utf-8") as f:
        for atom in manifest.atoms:
            record = _build_record(
                atom=atom,
                timestamp=timestamp,
                atom_violations=violations_by_atom.get(atom.id, []),
                multiplier=multiplier,
                confidence=confidence,
            )
            f.write(json.dumps(record) + "\n")
    return path


def _build_record(
    atom: PlanAtom,
    timestamp: str,
    atom_violations: list[BudgetViolation],
    multiplier: DynamicMultiplier,
    confidence: ConfidenceLevel,
) -> dict[str, object]:
    budget = atom.token_budget
    return {
        "timestamp": timestamp,
        "atom_id": atom.id,
        "estimated_read_tokens": budget.estimated_read_tokens if budget else None,
        "estimated_write_tokens": budget.estimated_write_tokens if budget else None,
        "estimated_total_tokens": budget.estimated_total_tokens if budget else None,
        "split_required": budget.split_required if budget else None,
        "violations": [
            {"kind": v.kind, "detail": v.detail} for v in atom_violations
        ],
        "multiplier": multiplier.value,
        "confidence": confidence.value,
    }
