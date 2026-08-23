from __future__ import annotations

import json
from dataclasses import dataclass

from easy_cheese_schemas import CriterionDisposition, canonical_bytes

from milknado.domains.planning import PHYSICAL_OUTCOME_SCHEMA

PHYSICAL_OUTCOME_REASON_MAX = 4096
_PHYSICAL_OUTCOME_KEYS = frozenset({"schema", "criterion_results"})
_CRITERION_OUTCOME_KEYS = frozenset({"curd_id", "criterion_id", "disposition", "reason"})


@dataclass(frozen=True)
class MilknadoCriterionOutcome:
    curd_id: str
    criterion_id: str
    disposition: CriterionDisposition
    reason: str | None

    def __post_init__(self) -> None:
        if type(self.curd_id) is not str or not self.curd_id:
            raise ValueError("curd_id must be non-empty text")
        if type(self.criterion_id) is not str or not self.criterion_id:
            raise ValueError("criterion_id must be non-empty text")
        if not isinstance(self.disposition, CriterionDisposition):
            raise TypeError("disposition must be a CriterionDisposition")
        if self.disposition is CriterionDisposition.PASSED:
            if self.reason is not None:
                raise ValueError("passed criterion outcomes must not include a reason")
            return
        if type(self.reason) is not str or not self.reason.strip():
            raise ValueError("non-passed criterion outcomes require a non-blank reason")
        if len(self.reason) > PHYSICAL_OUTCOME_REASON_MAX:
            raise ValueError(
                f"criterion outcome reason must be at most "
                f"{PHYSICAL_OUTCOME_REASON_MAX} characters"
            )


def encode_milknado_physical_outcome(
    criterion_results: tuple[MilknadoCriterionOutcome, ...],
) -> str:
    """Encode the public worker-result protocol consumed by result aggregation."""
    if not isinstance(criterion_results, tuple) or not criterion_results:
        raise ValueError("criterion_results must be a non-empty tuple")
    if any(not isinstance(row, MilknadoCriterionOutcome) for row in criterion_results):
        raise TypeError("criterion_results must contain MilknadoCriterionOutcome values")
    keys = tuple((row.curd_id, row.criterion_id) for row in criterion_results)
    if len(set(keys)) != len(keys):
        raise ValueError("criterion outcome pairs must be unique")
    payload = {
        "schema": PHYSICAL_OUTCOME_SCHEMA,
        "criterion_results": [
            {
                "curd_id": row.curd_id,
                "criterion_id": row.criterion_id,
                "disposition": row.disposition.value,
                "reason": row.reason,
            }
            for row in criterion_results
        ],
    }
    return canonical_bytes(payload).decode()


class _ParsedObject(dict[str, object]):
    def __init__(self, pairs: list[tuple[str, object]]) -> None:
        values: dict[str, object] = {}
        duplicates: list[str] = []
        for key, value in pairs:
            if key in values:
                duplicates.append(key)
            values[key] = value
        super().__init__(values)
        self.duplicates = tuple(duplicates)


def _malformed(message: str) -> ValueError:
    return ValueError(f"malformed physical outcome: {message}")


def _exact_keys(payload: dict[str, object], expected: frozenset[str], label: str) -> None:
    if isinstance(payload, _ParsedObject) and payload.duplicates:
        raise _malformed(f"duplicate JSON keys in {label}")
    missing = expected - set(payload)
    unknown = set(payload) - expected
    if missing or unknown:
        raise _malformed(
            f"{label} expected exactly {sorted(expected)!r}, "
            f"missing={sorted(missing)!r}, unknown={sorted(unknown)!r}"
        )


def parse_milknado_physical_outcome(
    result: str | None,
    expected_pairs: tuple[tuple[str, str], ...],
) -> dict[tuple[str, str], MilknadoCriterionOutcome]:
    if result is None:
        return {}
    try:
        payload = json.loads(result, object_pairs_hook=_ParsedObject)
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict) or payload.get("schema") != PHYSICAL_OUTCOME_SCHEMA:
        return {}
    _exact_keys(payload, _PHYSICAL_OUTCOME_KEYS, "top-level object")
    raw_rows = payload["criterion_results"]
    if not isinstance(raw_rows, list) or not raw_rows:
        raise _malformed("criterion_results must be a non-empty array")

    outcomes: dict[tuple[str, str], MilknadoCriterionOutcome] = {}
    for index, raw_row in enumerate(raw_rows):
        if not isinstance(raw_row, dict):
            raise _malformed(f"criterion_results[{index}] must be an object")
        _exact_keys(raw_row, _CRITERION_OUTCOME_KEYS, f"criterion_results[{index}]")
        raw_disposition = raw_row["disposition"]
        if type(raw_disposition) is not str:
            raise _malformed(f"criterion_results[{index}].disposition must be text")
        try:
            disposition = CriterionDisposition(raw_disposition)
        except ValueError as exc:
            raise _malformed(f"criterion_results[{index}].disposition is unsupported") from exc
        try:
            outcome = MilknadoCriterionOutcome(
                curd_id=raw_row["curd_id"],
                criterion_id=raw_row["criterion_id"],
                disposition=disposition,
                reason=raw_row["reason"],
            )
        except (TypeError, ValueError) as exc:
            raise _malformed(f"criterion_results[{index}]: {exc}") from exc
        key = (outcome.curd_id, outcome.criterion_id)
        if key in outcomes:
            raise _malformed(f"duplicate criterion outcome pair {key!r}")
        outcomes[key] = outcome

    expected = set(expected_pairs)
    actual = set(outcomes)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise _malformed(f"criterion coverage mismatch: missing={missing!r}, unknown={unknown!r}")
    return outcomes


__all__ = ["MilknadoCriterionOutcome", "encode_milknado_physical_outcome"]
