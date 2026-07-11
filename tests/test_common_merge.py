"""Tests for the shared deep_merge helper (domains/common/merge.py)."""

from __future__ import annotations

from milknado.domains.common.merge import deep_merge


def test_replace_mode_merges_dicts_and_replaces_lists() -> None:
    base = {"a": {"x": 1, "y": 2}, "tags": ["a", "b"], "scalar": 1}
    override = {"a": {"y": 3, "z": 4}, "tags": ["c"], "scalar": 2}

    result = deep_merge(base, override, list_mode="replace")

    assert result == {"a": {"x": 1, "y": 3, "z": 4}, "tags": ["c"], "scalar": 2}


def test_replace_mode_does_not_mutate_inputs() -> None:
    base = {"a": {"x": 1}}
    override = {"a": {"y": 2}}

    deep_merge(base, override, list_mode="replace")

    assert base == {"a": {"x": 1}}
    assert override == {"a": {"y": 2}}


def test_merge_dedup_mode_extends_lists_and_dedups_by_json_value() -> None:
    base = {"tools": ["Bash", "Read"]}
    override = {"tools": ["Read", "Write"]}

    result = deep_merge(base, override, list_mode="merge_dedup")

    assert result == {"tools": ["Bash", "Read", "Write"]}


def test_merge_dedup_mode_still_recurse_merges_dicts() -> None:
    base = {"hooks": {"PreToolUse": ["a"]}, "other": 1}
    override = {"hooks": {"PreToolUse": ["a", "b"]}, "other": 2}

    result = deep_merge(base, override, list_mode="merge_dedup")

    assert result == {"hooks": {"PreToolUse": ["a", "b"]}, "other": 2}


def test_new_key_with_list_value_is_taken_as_is_in_both_modes() -> None:
    base: dict = {}
    override = {"tags": ["a", "a"]}

    replace_result = deep_merge(base, override, list_mode="replace")
    dedup_result = deep_merge(base, override, list_mode="merge_dedup")

    assert replace_result == {"tags": ["a", "a"]}
    assert dedup_result == {"tags": ["a", "a"]}
