"""Pure serialization core for the wiki <-> milknado roadmap crossover.

Pins the deterministic key derivation and the surgical markdown rewrites that
keep human-authored Intent/Acceptance byte-identical while milknado overwrites
only the harvest block + status/last_synced frontmatter.
"""

from __future__ import annotations

from datetime import date

import pytest

from milknado.domains.wiki._serialize import (
    HARVEST_END,
    HARVEST_START,
    compute_goal_ref,
    compute_roadmap_ref,
    extract_section,
    extract_title,
    load_frontmatter,
    parse_wikilink,
    read_prereqs,
    replace_harvest_block,
    set_frontmatter_field,
    slugify,
    validate_slug,
)

GOAL_FILE = """---
kind: goal
slug: wire-export-path
roadmap: hallouminate-milknado-crossover
created: 2026-06-08
status: pending
flavor: implement
prereqs: [define-schema]
---
# Wire the export path

## Intent
Make the export path durable so roadmap state survives a db loss.

## Acceptance
- harvest block written without touching Intent

## Outcome
<!-- milknado:harvest:start -->
result: pending
<!-- milknado:harvest:end -->
"""


class TestKeyDerivation:
    def test_goal_ref_is_deterministic(self) -> None:
        a = compute_goal_ref("road", "goal", "2026-06-08")
        b = compute_goal_ref("road", "goal", "2026-06-08")
        assert a == b

    def test_goal_ref_varies_by_each_component(self) -> None:
        base = compute_goal_ref("road", "goal", "2026-06-08")
        assert compute_goal_ref("other", "goal", "2026-06-08") != base
        assert compute_goal_ref("road", "other", "2026-06-08") != base
        assert compute_goal_ref("road", "goal", "2026-06-09") != base

    def test_goal_ref_accepts_date_objects(self) -> None:
        assert compute_goal_ref("road", "goal", date(2026, 6, 8)) == compute_goal_ref(
            "road", "goal", "2026-06-08"
        )

    def test_roadmap_ref_differs_from_goal_ref(self) -> None:
        assert compute_roadmap_ref("road", "2026-06-08") != compute_goal_ref(
            "road", "road", "2026-06-08"
        )


class TestFrontmatter:
    def test_load_frontmatter_reads_keys(self) -> None:
        fm = load_frontmatter(GOAL_FILE)
        assert fm["slug"] == "wire-export-path"
        assert fm["prereqs"] == ["define-schema"]

    def test_load_frontmatter_empty_when_absent(self) -> None:
        assert load_frontmatter("# no frontmatter here") == {}

    def test_set_field_rewrites_existing_key(self) -> None:
        out = set_frontmatter_field(GOAL_FILE, "status", "done")
        assert load_frontmatter(out)["status"] == "done"
        # other frontmatter untouched
        assert load_frontmatter(out)["slug"] == "wire-export-path"

    def test_set_field_inserts_missing_key(self) -> None:
        # Callers quote values that YAML would otherwise coerce (e.g. timestamps),
        # so the wiki file keeps a stable, human-readable string.
        out = set_frontmatter_field(GOAL_FILE, "last_synced", '"2026-06-08T14:00:00Z"')
        assert load_frontmatter(out)["last_synced"] == "2026-06-08T14:00:00Z"

    def test_set_field_leaves_body_byte_identical(self) -> None:
        out = set_frontmatter_field(GOAL_FILE, "status", "done")
        body_marker = "# Wire the export path"
        assert out[out.index(body_marker) :] == GOAL_FILE[GOAL_FILE.index(body_marker) :]


class TestSections:
    def test_extract_section_returns_intent(self) -> None:
        intent = extract_section(GOAL_FILE, "Intent")
        assert intent is not None
        assert "durable" in intent
        assert "Acceptance" not in intent

    def test_extract_section_missing_returns_none(self) -> None:
        assert extract_section(GOAL_FILE, "Nonexistent") is None

    def test_extract_title_returns_h1(self) -> None:
        assert extract_title(GOAL_FILE) == "Wire the export path"

    def test_extract_title_missing_returns_none(self) -> None:
        assert extract_title("no title here") is None


class TestHarvestBlock:
    def test_replace_harvest_block_swaps_inner_only(self) -> None:
        out = replace_harvest_block(GOAL_FILE, "result: done · tasks: 5 done / 0 failed")
        assert "result: done · tasks: 5 done / 0 failed" in out
        assert "result: pending" not in out

    def test_replace_harvest_block_keeps_intent_acceptance_byte_identical(self) -> None:
        out = replace_harvest_block(GOAL_FILE, "result: done")
        intent_before = extract_section(GOAL_FILE, "Intent")
        intent_after = extract_section(out, "Intent")
        assert intent_before == intent_after
        accept_before = extract_section(GOAL_FILE, "Acceptance")
        accept_after = extract_section(out, "Acceptance")
        assert accept_before == accept_after

    def test_replace_harvest_block_idempotent_markers(self) -> None:
        out = replace_harvest_block(GOAL_FILE, "result: done")
        assert out.count(HARVEST_START) == 1
        assert out.count(HARVEST_END) == 1

    def test_replace_harvest_block_appends_when_markers_absent(self) -> None:
        text = "---\nkind: goal\n---\n# Title\n\n## Intent\nwhy\n"
        out = replace_harvest_block(text, "result: done")
        assert HARVEST_START in out
        assert HARVEST_END in out
        assert "result: done" in out
        # original content preserved verbatim at the head
        assert out.startswith(text)

    def test_replace_harvest_block_literal_backslash_escape(self) -> None:
        """re.sub replacement-string metacharacters in inner must not be interpreted.

        A Windows path or agent output containing \\n / \\t / \g<...> / \1 would
        previously be silently mangled or raise re.error.  The bytes must round-trip
        verbatim.
        """
        dangerous = r"summary: C:\new\g<x> and \1 and \t"
        out = replace_harvest_block(GOAL_FILE, dangerous)
        assert dangerous in out, f"inner was mangled; got:\n{out}"

    def test_replace_harvest_block_backreference_does_not_raise(self) -> None:
        """\g<name> in inner must not raise re.error (unmatched group reference)."""
        inner = r"path: C:\new\test and \g<bad_group>"
        # Must not raise — previously blew up with re.error
        out = replace_harvest_block(GOAL_FILE, inner)
        assert inner in out


class TestSlugify:
    def test_slugify_basic(self) -> None:
        assert slugify("Wire the Export Path!") == "wire-the-export-path"

    def test_slugify_collapses_separators(self) -> None:
        assert slugify("  foo / bar __baz  ") == "foo-bar-baz"


class TestValidateSlug:
    def test_accepts_normal_slug(self) -> None:
        assert validate_slug("demo-roadmap") == "demo-roadmap"

    @pytest.mark.parametrize("bad", ["", "../escape", "a/b", "a\\b", ".."])
    def test_rejects_unsafe(self, bad: str) -> None:
        with pytest.raises(ValueError, match="unsafe roadmap slug"):
            validate_slug(bad)


class TestParseWikilink:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("[[target]]", "target"),
            ("[[target|alias]]", "target"),
            ("bare-slug", "bare-slug"),
            ("[[multi-word-target]]", "multi-word-target"),
            ("[[t|long alias text]]", "t"),
        ],
    )
    def test_parses_wikilink_forms(self, value: str, expected: str) -> None:
        assert parse_wikilink(value) == expected

    def test_bare_string_passes_through(self) -> None:
        assert parse_wikilink("define-schema") == "define-schema"

    def test_unclosed_bracket_passes_through(self) -> None:
        # Malformed "[[" has no closing "]]" — must pass through verbatim,
        # not raise or silently drop the string.
        assert parse_wikilink("[[") == "[["

    def test_empty_target_passes_through(self) -> None:
        # "[[]]" matches the wikilink shape but has an empty target segment —
        # the [^|\]]+ group requires 1+ chars, so the regex won't match.
        assert parse_wikilink("[[]]") == "[[]]"

    def test_double_pipe_uses_first_segment_as_target(self) -> None:
        # "[[a|b|c]]" — the first "|" is the alias separator; "b|c" is the alias.
        # The target must be "a", not "b|c" or a raised error.
        assert parse_wikilink("[[a|b|c]]") == "a"


class TestReadPrereqs:
    def test_down_only_parses_wikilinks(self) -> None:
        fm = {"down": ["[[a]]", "[[b|alias]]"]}
        assert read_prereqs(fm) == ["a", "b"]

    def test_prereqs_only_passes_through(self) -> None:
        fm = {"prereqs": ["a", "b"]}
        assert read_prereqs(fm) == ["a", "b"]

    def test_union_of_prereqs_and_down(self) -> None:
        fm = {"prereqs": ["a"], "down": ["[[b]]", "[[a]]"]}
        result = read_prereqs(fm)
        # union: a from prereqs, b from down, a deduplicated
        assert result == ["a", "b"]

    def test_order_preserving_dedup(self) -> None:
        fm = {"prereqs": ["x", "y"], "down": ["[[y]]", "[[z]]"]}
        assert read_prereqs(fm) == ["x", "y", "z"]

    def test_empty_frontmatter_returns_empty(self) -> None:
        assert read_prereqs({}) == []

    def test_neither_field_returns_empty(self) -> None:
        assert read_prereqs({"kind": "goal"}) == []

    def test_non_list_prereqs_raises(self) -> None:
        with pytest.raises(ValueError, match="prereqs"):
            read_prereqs({"prereqs": "not-a-list"})

    def test_non_str_entry_in_prereqs_raises(self) -> None:
        with pytest.raises(ValueError, match="prereqs"):
            read_prereqs({"prereqs": [1, 2]})

    def test_non_list_down_raises(self) -> None:
        with pytest.raises(ValueError, match="`down`"):
            read_prereqs({"down": "not-a-list"})

    def test_non_str_entry_in_down_raises(self) -> None:
        with pytest.raises(ValueError, match="`down`"):
            read_prereqs({"down": [1]})

    def test_explicit_empty_down_list_returns_empty(self) -> None:
        # An explicit `down: []` is falsy, so `or []` triggers the same path as
        # missing `down:`.  Both must return []; this pins the coercion contract.
        assert read_prereqs({"down": []}) == []
