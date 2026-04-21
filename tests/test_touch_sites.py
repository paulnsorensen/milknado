from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock

from milknado.domains.common.protocols import SymbolLocation
from milknado.domains.common.types import DegradationMarker, TilthMap
from milknado.domains.planning.touch_sites import (
    _CHARS_PER_TOKEN,
    _TOKEN_BUDGET,
    MAX_FILES,
    _extract_keywords,
    _touch_sites_section,
)

HEADER = "# Probable Touch Sites"

SIMPLE_SPEC = """\
## US-001: Add auth middleware
### TokenValidation
## US-002: Rate limiting
### RequestThrottling
"""


def _make_crg(hits_per_keyword: dict[str, list[dict]] | None = None) -> MagicMock:
    crg = MagicMock()
    hits_per_keyword = hits_per_keyword or {}

    def semantic_search(query: str, top_n: int = 5, detail_level: str = "minimal") -> list[dict]:  # noqa: ARG001
        return hits_per_keyword.get(query, [])

    crg.semantic_search.side_effect = semantic_search
    return crg


def _make_tilth(
    symbol_results: dict[str, list[SymbolLocation]] | None = None,
    section_content: str = "def foo(): pass\n",
) -> MagicMock:
    tilth = MagicMock()
    symbol_results = symbol_results or {}

    def search_symbol(keyword: str, glob: str | None = None) -> list[SymbolLocation]:  # noqa: ARG001
        return symbol_results.get(keyword, [])

    tilth.search_symbol.side_effect = search_symbol
    tilth.read_section.return_value = section_content
    return tilth


class TestExtractKeywords:
    def test_extracts_us_headings(self) -> None:
        kws = _extract_keywords(SIMPLE_SPEC)
        assert "Add auth middleware" in kws
        assert "Rate limiting" in kws

    def test_extracts_h3_headings(self) -> None:
        kws = _extract_keywords(SIMPLE_SPEC)
        assert "TokenValidation" in kws
        assert "RequestThrottling" in kws

    def test_us_before_h3(self) -> None:
        kws = _extract_keywords(SIMPLE_SPEC)
        us_idx = kws.index("Add auth middleware")
        h3_idx = kws.index("TokenValidation")
        assert us_idx < h3_idx

    def test_deduplicates(self) -> None:
        spec = "## US-001: Duplicate\n### Duplicate\n"
        kws = _extract_keywords(spec)
        assert kws.count("Duplicate") == 1

    def test_empty_spec(self) -> None:
        assert _extract_keywords("") == []


class TestHappyPath:
    def test_header_present(self) -> None:
        crg = _make_crg({"Add auth middleware": [{"file_path": "src/auth.py"}]})
        tilth = _make_tilth()
        result = _touch_sites_section(SIMPLE_SPEC, crg, tilth, None)
        assert result.startswith(HEADER)

    def test_crg_hits_appear_in_output(self) -> None:
        crg = _make_crg(
            {
                "Add auth middleware": [{"file_path": "src/auth.py"}],
                "TokenValidation": [{"file_path": "src/tokens.py"}],
            }
        )
        tilth = _make_tilth()
        result = _touch_sites_section(SIMPLE_SPEC, crg, tilth, None)
        assert "src/auth.py" in result
        assert "src/tokens.py" in result

    def test_output_under_token_budget(self) -> None:
        crg = _make_crg(
            {
                "Add auth middleware": [{"file_path": "src/auth.py"}],
            }
        )
        tilth = _make_tilth(section_content="x" * 100)
        result = _touch_sites_section(SIMPLE_SPEC, crg, tilth, None)
        assert len(result) <= _TOKEN_BUDGET * _CHARS_PER_TOKEN + 500  # header overhead allowed

    def test_read_section_called_for_tilth_hits(self) -> None:
        loc = SymbolLocation(Path("src/auth.py"), 10, 20)
        crg = _make_crg()
        tilth = _make_tilth(symbol_results={"Add auth middleware": [loc]})
        _touch_sites_section(SIMPLE_SPEC, crg, tilth, None)
        tilth.read_section.assert_called_once_with(Path("src/auth.py"), 10, 20)


class TestNoSpecFallback:
    def test_header_present(self) -> None:
        tilth = MagicMock()
        tilth.structural_map.return_value = TilthMap(
            scope=Path("src"),
            budget_tokens=400,
            data={"src/foo.py": "~120 tokens"},
        )
        result = _touch_sites_section(None, None, tilth, Path("."))
        assert result.startswith(HEADER)

    def test_uses_structural_map(self) -> None:
        tilth = MagicMock()
        tilth.structural_map.return_value = TilthMap(
            scope=Path("src"),
            budget_tokens=400,
            data={"src/foo.py": "~120 tokens"},
        )
        result = _touch_sites_section(None, None, tilth, Path("."))
        assert "src/foo.py" in result
        tilth.structural_map.assert_called_once()

    def test_uses_src_subfolder(self) -> None:
        tilth = MagicMock()
        tilth.structural_map.return_value = TilthMap(
            scope=Path("src"),
            budget_tokens=400,
            data={},
        )
        _touch_sites_section(None, None, tilth, Path("/project"))
        call_scope = tilth.structural_map.call_args[0][0]
        assert str(call_scope).endswith("src")

    def test_degrades_when_tilth_none(self) -> None:
        result = _touch_sites_section(None, None, None, None)
        assert HEADER in result
        assert "skipped" in result.lower() or "unavailable" in result.lower()

    def test_degrades_on_structural_map_marker(self) -> None:
        tilth = MagicMock()
        tilth.structural_map.return_value = DegradationMarker(
            source="tilth",
            reason="timeout",
        )
        result = _touch_sites_section(None, None, tilth, None)
        assert HEADER in result
        assert "degraded" in result.lower() or "timeout" in result.lower()


class TestCrgMissTilthFallback:
    def test_falls_back_to_search_symbol_on_zero_crg_hits(self) -> None:
        crg = _make_crg()  # all keywords → empty
        loc = SymbolLocation(Path("src/tokens.py"), 5, 15)
        tilth = _make_tilth(symbol_results={"Add auth middleware": [loc]})
        result = _touch_sites_section(SIMPLE_SPEC, crg, tilth, None)
        assert "src/tokens.py" in result

    def test_search_symbol_called_for_missed_keywords(self) -> None:
        crg = _make_crg()
        tilth = _make_tilth()
        _touch_sites_section(SIMPLE_SPEC, crg, tilth, None)
        assert tilth.search_symbol.call_count == len(_extract_keywords(SIMPLE_SPEC))

    def test_no_crg_uses_all_keywords_as_zero_hit(self) -> None:
        loc = SymbolLocation(Path("src/foo.py"), 1, 10)
        tilth = _make_tilth(
            symbol_results={"Add auth middleware": [loc]},
            section_content="code here",
        )
        result = _touch_sites_section(SIMPLE_SPEC, None, tilth, None)
        assert "src/foo.py" in result


class TestBudgetExhaustion:
    def test_capped_at_max_files(self) -> None:
        hits = {f"kw{i}": [{"file_path": f"src/file{i}.py"}] for i in range(20)}
        spec = "\n".join(f"## US-{i:03d}: kw{i}" for i in range(20))
        crg = _make_crg(hits)
        tilth = _make_tilth()
        result = _touch_sites_section(spec, crg, tilth, None)
        # Count how many distinct files appear by counting label occurrences
        file_entries = re.findall(r"## `src/file\d+\.py`", result)
        assert len(file_entries) <= MAX_FILES

    def test_early_truncation_when_budget_exhausted(self) -> None:
        big_content = "x" * (_TOKEN_BUDGET * _CHARS_PER_TOKEN * 2)
        crg = _make_crg(
            {
                "Add auth middleware": [{"file_path": "src/auth.py"}],
                "TokenValidation": [{"file_path": "src/tok.py"}],
            }
        )
        tilth = _make_tilth(
            symbol_results={},
            section_content=big_content,
        )
        tilth.search_symbol.return_value = []
        # Inject locs by having crg return hits so read_section is called
        result = _touch_sites_section(SIMPLE_SPEC, crg, tilth, None)
        # Budget exhaustion means not all budget*chars worth of content fits
        assert len(result) < _TOKEN_BUDGET * _CHARS_PER_TOKEN * 3


class TestNoKeywordsFound:
    def test_message_when_no_us_or_h3_headings(self) -> None:
        bare_spec = "This spec has no headings at all.\nJust prose.\n"
        result = _touch_sites_section(bare_spec, None, None, None)
        assert HEADER in result
        assert "no US-NNN blocks" in result or "no touch sites" in result.lower()
