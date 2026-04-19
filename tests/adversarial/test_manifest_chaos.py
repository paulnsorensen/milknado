"""Adversarial tests for parse_manifest_from_output.

Priority order: invalid inputs → edge cases → integration paths → happy path.
"""
from __future__ import annotations

import json

from milknado.domains.planning.manifest import (
    MANIFEST_VERSION,
    parse_manifest_from_output,
)


def _wrap(payload: dict) -> str:
    """Wrap a dict in a fenced JSON block as the planner would emit it."""
    return f"```json\n{json.dumps(payload)}\n```"


def _minimal_change(
    cid: str = "c1",
    path: str = "src/foo.py",
    description: str = "Add feature",
) -> dict:
    return {
        "id": cid,
        "path": path,
        "edit_kind": "modify",
        "symbols": [],
        "depends_on": [],
        "description": description,
    }


def _minimal_manifest(**overrides) -> dict:
    base = {
        "manifest_version": MANIFEST_VERSION,
        "goal": "some goal",
        "goal_summary": "some summary",
        "spec_path": None,
        "changes": [_minimal_change()],
        "new_relationships": [],
    }
    base.update(overrides)
    return base


class TestUnicodeInFields:
    """Probe Unicode edge cases in goal / goal_summary / description."""

    def test_null_byte_in_goal_returns_none(self) -> None:
        # Null bytes are not valid JSON strings in all parsers; probe behavior.
        payload = _minimal_manifest(goal="goal\x00value")
        result = parse_manifest_from_output(_wrap(payload))
        # goal is technically a non-empty string with a null byte — parser
        # currently accepts any non-empty str. Confirm it doesn't crash.
        # Expected: either None (rejected) or PlanChangeManifest (accepted silently).
        # A crash here would be score-95.
        assert result is None or result.goal == "goal\x00value"

    def test_bom_in_goal_non_empty_accepted(self) -> None:
        # BOM (\ufeff) at start — non-empty, so should be accepted.
        payload = _minimal_manifest(goal="\ufeffThis is a BOM-prefixed goal")
        result = parse_manifest_from_output(_wrap(payload))
        assert result is not None
        assert result.goal.startswith("\ufeff")

    def test_rtl_override_in_description(self) -> None:
        # RTL override char U+202E is non-empty; parser should accept silently.
        change = _minimal_change(description="\u202emalicious text")
        payload = _minimal_manifest(changes=[change])
        result = parse_manifest_from_output(_wrap(payload))
        assert result is not None

    def test_10kb_emoji_in_goal_summary(self) -> None:
        # 10KB of emoji — no length cap, should be accepted without crash.
        big_summary = "🧀" * 3000  # ~12KB
        payload = _minimal_manifest(goal_summary=big_summary)
        result = parse_manifest_from_output(_wrap(payload))
        assert result is not None
        assert result.goal_summary == big_summary

    def test_whitespace_only_goal_rejected(self) -> None:
        # Spec says goal must be non-empty; the guard strips before checking,
        # so whitespace-only values are rejected.
        payload = _minimal_manifest(goal="   ")
        result = parse_manifest_from_output(_wrap(payload))
        assert result is None

    def test_whitespace_only_description_rejected(self) -> None:
        change = _minimal_change(description="   ")
        payload = _minimal_manifest(changes=[change])
        result = parse_manifest_from_output(_wrap(payload))
        assert result is None

    def test_newline_only_description_rejected(self) -> None:
        change = _minimal_change(description="\n\n")
        payload = _minimal_manifest(changes=[change])
        result = parse_manifest_from_output(_wrap(payload))
        assert result is None

    def test_literal_json_inside_description(self) -> None:
        # JSON-within-description — make sure the parser doesn't try to re-parse it.
        json_desc = '{"key": "value", "nested": [1,2,3]}'
        change = _minimal_change(description=json_desc)
        payload = _minimal_manifest(changes=[change])
        result = parse_manifest_from_output(_wrap(payload))
        assert result is not None
        assert result.changes[0].description == json_desc

    def test_control_chars_in_description(self) -> None:
        # Control chars (\x01-\x1f) that are technically valid JSON escapes.
        change = _minimal_change(description="Change\x01\x07\x1f boundary")
        payload = _minimal_manifest(changes=[change])
        result = parse_manifest_from_output(_wrap(payload))
        assert result is not None

    def test_1mb_goal_summary(self) -> None:
        # 1MB goal_summary — no length cap, should not crash or OOM.
        big = "x" * (1024 * 1024)
        payload = _minimal_manifest(goal_summary=big)
        result = parse_manifest_from_output(_wrap(payload))
        assert result is not None
        assert len(result.goal_summary) == 1024 * 1024


class TestMassiveManifest:
    def test_1000_changes_linear_deps(self) -> None:
        # 1000 changes in a linear chain — O(n^2) dep lookup would be slow.
        changes = []
        for i in range(1000):
            dep = [f"c{i - 1}"] if i > 0 else []
            changes.append({
                "id": f"c{i}",
                "path": f"src/file_{i}.py",
                "edit_kind": "modify",
                "symbols": [],
                "depends_on": dep,
                "description": f"Change {i}",
            })
        payload = _minimal_manifest(changes=changes)
        result = parse_manifest_from_output(_wrap(payload))
        assert result is not None
        assert len(result.changes) == 1000


class TestDuplicateIds:
    def test_duplicate_change_ids_rejected(self) -> None:
        changes = [
            _minimal_change("c1"),
            _minimal_change("c1", path="src/bar.py"),  # same id, different path
        ]
        payload = _minimal_manifest(changes=changes)
        result = parse_manifest_from_output(_wrap(payload))
        assert result is None

    def test_same_path_different_ids_accepted(self) -> None:
        # Same path but different IDs — valid (multiple symbols in one file).
        changes = [
            _minimal_change("c1", path="src/foo.py"),
            _minimal_change("c2", path="src/foo.py"),
        ]
        payload = _minimal_manifest(changes=changes)
        result = parse_manifest_from_output(_wrap(payload))
        assert result is not None
        assert len(result.changes) == 2


class TestSpecPathTraversal:
    def test_path_traversal_accepted_without_sanitizing(self) -> None:
        # spec_path is user-supplied; the parser stores it verbatim.
        # This probes whether the parser sanitizes or rejects path traversals.
        payload = _minimal_manifest(spec_path="../../etc/passwd")
        result = parse_manifest_from_output(_wrap(payload))
        # Parser currently stores spec_path verbatim — no sanitization.
        # Finding: path traversal strings pass through unvalidated.
        assert result is not None
        assert result.spec_path == "../../etc/passwd"

    def test_absolute_path_in_spec_path_accepted(self) -> None:
        payload = _minimal_manifest(spec_path="/etc/passwd")
        result = parse_manifest_from_output(_wrap(payload))
        assert result is not None
        assert result.spec_path == "/etc/passwd"

    def test_spec_path_wrong_type_rejected(self) -> None:
        payload = _minimal_manifest(spec_path=42)
        result = parse_manifest_from_output(_wrap(payload))
        assert result is None


class TestMalformedJson:
    def test_no_fenced_block_returns_none(self) -> None:
        result = parse_manifest_from_output("No JSON here at all")
        assert result is None

    def test_invalid_json_in_block_returns_none(self) -> None:
        result = parse_manifest_from_output("```json\n{broken json\n```")
        assert result is None

    def test_json_array_root_returns_none(self) -> None:
        result = parse_manifest_from_output("```json\n[1, 2, 3]\n```")
        assert result is None

    def test_empty_fenced_block_returns_none(self) -> None:
        result = parse_manifest_from_output("```json\n\n```")
        assert result is None

    def test_v1_manifest_version_rejected(self) -> None:
        payload = _minimal_manifest(manifest_version="milknado.plan.v1")
        result = parse_manifest_from_output(_wrap(payload))
        assert result is None

    def test_missing_goal_field_rejected(self) -> None:
        payload = _minimal_manifest()
        del payload["goal"]
        result = parse_manifest_from_output(_wrap(payload))
        assert result is None

    def test_empty_goal_rejected(self) -> None:
        payload = _minimal_manifest(goal="")
        result = parse_manifest_from_output(_wrap(payload))
        assert result is None

    def test_goal_is_int_rejected(self) -> None:
        payload = _minimal_manifest(goal=42)
        result = parse_manifest_from_output(_wrap(payload))
        assert result is None

    def test_empty_goal_summary_rejected(self) -> None:
        payload = _minimal_manifest(goal_summary="")
        result = parse_manifest_from_output(_wrap(payload))
        assert result is None

    def test_changes_is_null_rejected(self) -> None:
        payload = _minimal_manifest(changes=None)
        result = parse_manifest_from_output(_wrap(payload))
        assert result is None

    def test_changes_is_string_rejected(self) -> None:
        payload = _minimal_manifest(changes="not a list")
        result = parse_manifest_from_output(_wrap(payload))
        assert result is None

    def test_depends_on_unknown_id_rejected(self) -> None:
        changes = [_minimal_change("c1")]
        changes[0] = dict(changes[0], depends_on=["nonexistent"])
        payload = _minimal_manifest(changes=changes)
        result = parse_manifest_from_output(_wrap(payload))
        assert result is None

    def test_invalid_edit_kind_rejected(self) -> None:
        change = dict(_minimal_change(), edit_kind="explode")
        payload = _minimal_manifest(changes=[change])
        result = parse_manifest_from_output(_wrap(payload))
        assert result is None

    def test_empty_description_rejected(self) -> None:
        change = dict(_minimal_change(), description="")
        payload = _minimal_manifest(changes=[change])
        result = parse_manifest_from_output(_wrap(payload))
        assert result is None

    def test_description_missing_entirely_rejected(self) -> None:
        change = {
            "id": "c1",
            "path": "src/foo.py",
            "edit_kind": "modify",
            "symbols": [],
            "depends_on": [],
        }
        payload = _minimal_manifest(changes=[change])
        result = parse_manifest_from_output(_wrap(payload))
        assert result is None

    def test_multiple_fenced_blocks_uses_first(self) -> None:
        # If two fenced blocks exist, regex uses the first one.
        valid = _wrap(_minimal_manifest())
        garbage = "```json\n{bad}\n```"
        result = parse_manifest_from_output(valid + "\n" + garbage)
        assert result is not None

    def test_second_fenced_block_invalid_first_wins(self) -> None:
        invalid_first = "```json\n{bad}\n```"
        valid = _wrap(_minimal_manifest())
        result = parse_manifest_from_output(invalid_first + "\n" + valid)
        # First block is bad JSON — parser returns None even though second block is valid.
        assert result is None
