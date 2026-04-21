"""US-002: _instructions_section prompt quality tests."""

from __future__ import annotations

import pytest

from milknado.domains.planning.context import _instructions_section


@pytest.fixture()
def fresh_section() -> str:
    return _instructions_section(resuming=False)


@pytest.fixture()
def resuming_section() -> str:
    return _instructions_section(resuming=True)


class TestInstructionsSectionUS002:
    def test_no_merge_into_one_change_phrase(self, fresh_section: str) -> None:
        assert "merge into one change" not in fresh_section

    def test_no_merge_into_one_change_phrase_resuming(self, resuming_section: str) -> None:
        assert "merge into one change" not in resuming_section

    def test_bundling_anti_pattern_block_present(self, fresh_section: str) -> None:
        assert "Anti-pattern — bundling multiple stories into one change" in fresh_section

    def test_bundling_anti_pattern_block_present_resuming(self, resuming_section: str) -> None:
        assert "Anti-pattern — bundling multiple stories into one change" in resuming_section

    def test_c1_c2_same_path_counter_example(self, fresh_section: str) -> None:
        # Both c1 and c2 reference src/foo.py in the anti-pattern block
        assert '"id": "c1", "path": "src/foo.py"' in fresh_section
        assert '"id": "c2", "path": "src/foo.py"' in fresh_section

    def test_c1_c2_same_path_counter_example_resuming(self, resuming_section: str) -> None:
        assert '"id": "c1", "path": "src/foo.py"' in resuming_section
        assert '"id": "c2", "path": "src/foo.py"' in resuming_section

    def test_tests_note_block_present(self, fresh_section: str) -> None:
        assert "Every user story requires a tests/ change" in fresh_section

    def test_tests_note_block_present_resuming(self, resuming_section: str) -> None:
        assert "Every user story requires a tests/ change" in resuming_section

    def test_tests_note_elevated_before_granularity(self, fresh_section: str) -> None:
        tests_pos = fresh_section.index("Every user story requires a tests/ change")
        granularity_pos = fresh_section.index("Emit file-level changes")
        assert tests_pos < granularity_pos

    def test_tests_note_example_shows_depends_on(self, fresh_section: str) -> None:
        assert '"depends_on": ["c1"]' in fresh_section

    def test_tests_note_example_shows_tests_path(self, fresh_section: str) -> None:
        assert '"path": "tests/test_foo.py"' in fresh_section

    def test_bundled_description_anti_pattern_present(self, fresh_section: str) -> None:
        assert "Anti-pattern — bundled description" in fresh_section

    def test_bundled_description_anti_pattern_present_resuming(
        self, resuming_section: str
    ) -> None:
        assert "Anti-pattern — bundled description" in resuming_section
