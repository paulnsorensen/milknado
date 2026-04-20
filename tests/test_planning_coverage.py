from __future__ import annotations

from milknado.domains.batching import FileChange
from milknado.domains.planning.coverage import coverage_check
from milknado.domains.planning.manifest import MANIFEST_VERSION, PlanChangeManifest


def _manifest(*changes: FileChange) -> PlanChangeManifest:
    return PlanChangeManifest(
        manifest_version=MANIFEST_VERSION,
        goal="test",
        goal_summary="test",
        spec_path=None,
        changes=changes,
        new_relationships=(),
    )


def _change(
    cid: str,
    path: str,
    description: str = "",
    depends_on: tuple[str, ...] = (),
) -> FileChange:
    return FileChange(id=cid, path=path, description=description, depends_on=depends_on)


class TestCoverageCheckNoImplChanges:
    def test_returns_empty_when_no_src_changes(self) -> None:
        manifest = _manifest(_change("c1", "tests/test_foo.py", "US-001 tests"))
        assert coverage_check(manifest) == []

    def test_returns_empty_for_empty_manifest(self) -> None:
        assert coverage_check(_manifest()) == []


class TestCoverageCheckCoveredViaDependsOn:
    def test_covered_when_test_depends_on_impl_id(self) -> None:
        manifest = _manifest(
            _change("c1", "src/foo.py", "US-001 implement feature"),
            _change("c2", "tests/test_foo.py", "US-001 tests", depends_on=("c1",)),
        )
        assert coverage_check(manifest) == []

    def test_covered_with_nested_tests_path(self) -> None:
        manifest = _manifest(
            _change("c1", "src/foo.py", "US-002 impl"),
            _change("c2", "tests/unit/test_foo.py", "US-002 tests", depends_on=("c1",)),
        )
        assert coverage_check(manifest) == []

    def test_multiple_impl_all_covered_via_depends_on(self) -> None:
        manifest = _manifest(
            _change("c1", "src/a.py", "US-001 impl"),
            _change("c2", "tests/test_a.py", "US-001 tests", depends_on=("c1",)),
            _change("c3", "src/b.py", "US-002 impl"),
            _change("c4", "tests/test_b.py", "US-002 tests", depends_on=("c3",)),
        )
        assert coverage_check(manifest) == []


class TestCoverageCheckCoveredViaSharedUsRef:
    def test_covered_when_test_shares_us_ref(self) -> None:
        manifest = _manifest(
            _change("c1", "src/foo.py", "US-001 implement feature"),
            _change("c2", "tests/test_foo.py", "US-001 add tests"),
        )
        assert coverage_check(manifest) == []

    def test_covered_with_multiple_refs_partial_overlap(self) -> None:
        manifest = _manifest(
            _change("c1", "src/foo.py", "US-001 US-002 impl"),
            _change("c2", "tests/test_foo.py", "US-002 tests"),
        )
        assert coverage_check(manifest) == []

    def test_one_test_can_cover_multiple_impls_via_shared_ref(self) -> None:
        manifest = _manifest(
            _change("c1", "src/a.py", "US-003 impl a"),
            _change("c2", "src/b.py", "US-003 impl b"),
            _change("c3", "tests/test_ab.py", "US-003 tests"),
        )
        assert coverage_check(manifest) == []


class TestCoverageCheckOrphans:
    def test_orphan_when_no_test_change_exists(self) -> None:
        manifest = _manifest(_change("c1", "src/foo.py", "US-001 impl"))
        assert coverage_check(manifest) == ["src/foo.py"]

    def test_orphan_when_test_has_no_depends_on_and_no_shared_ref(self) -> None:
        manifest = _manifest(
            _change("c1", "src/foo.py", "US-001 impl"),
            _change("c2", "tests/test_bar.py", "US-002 unrelated tests"),
        )
        assert coverage_check(manifest) == ["src/foo.py"]

    def test_orphan_when_impl_has_no_us_ref(self) -> None:
        manifest = _manifest(
            _change("c1", "src/foo.py", "refactor internals"),
            _change("c2", "tests/test_foo.py", "US-001 tests"),
        )
        assert coverage_check(manifest) == ["src/foo.py"]

    def test_partial_coverage_returns_only_orphans(self) -> None:
        manifest = _manifest(
            _change("c1", "src/a.py", "US-001 impl"),
            _change("c2", "tests/test_a.py", "US-001 tests"),
            _change("c3", "src/b.py", "US-002 impl"),
        )
        assert coverage_check(manifest) == ["src/b.py"]

    def test_result_is_sorted(self) -> None:
        manifest = _manifest(
            _change("c1", "src/z.py", "US-001 impl"),
            _change("c2", "src/a.py", "US-002 impl"),
            _change("c3", "src/m.py", "US-003 impl"),
        )
        assert coverage_check(manifest) == ["src/a.py", "src/m.py", "src/z.py"]


class TestCoverageCheckDependsOnPrecedesSharedRef:
    def test_depends_on_covers_even_with_no_shared_ref(self) -> None:
        manifest = _manifest(
            _change("c1", "src/foo.py", "US-001 impl"),
            _change("c2", "tests/test_foo.py", "unrelated description", depends_on=("c1",)),
        )
        assert coverage_check(manifest) == []

    def test_depends_on_wrong_id_does_not_cover(self) -> None:
        manifest = _manifest(
            _change("c1", "src/foo.py", "US-001 impl"),
            _change("c2", "tests/test_foo.py", "US-999 tests", depends_on=("c99",)),
        )
        assert coverage_check(manifest) == ["src/foo.py"]
