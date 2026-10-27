"""Slice guard: each rule fails on a planted violation and passes once it is reverted."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from scripts import check_slices as guard
from tests.slice_fixture import BACK_EDGE, MAP, SRC, TREE, git

# Each case commits a fixture tree to a real git repository for the SB-07 base.
pytestmark = pytest.mark.real_process


@dataclass
class Repo:
    root: Path
    capsys: pytest.CaptureFixture[str]

    def write(self, path: str, text: str) -> None:
        (self.root / path).parent.mkdir(parents=True, exist_ok=True)
        _ = (self.root / path).write_text(text, encoding="utf-8")

    def source(self, name: str) -> Path:
        return self.root / SRC / name

    def commit(self) -> None:
        git(self.root, "add", "-A")
        git(self.root, "commit", "-q", "--allow-empty", "-m", "base")

    def guard(self, *args: str) -> tuple[int, str]:
        _ = self.capsys.readouterr()
        code = guard.main(list(args), self.root)
        captured = self.capsys.readouterr()
        return code, captured.out + captured.err

    def adopt(self) -> None:
        """Commit an adopted tree: the slice map and its baseline."""
        assert self.guard("baseline", "--init")[0] == 0
        self.commit()

    def fails(self, *expected: str) -> str:
        code, out = self.guard("check", "--base", "HEAD", "--strict")
        assert code == 1, out
        for line in expected:
            assert line in out
        return out

    def passes(self) -> None:
        code, out = self.guard("check", "--base", "HEAD", "--strict")
        assert (code, "slice guard: pass" in out) == (0, True), out


@pytest.fixture
def repo(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> Repo:
    fixture = Repo(tmp_path, capsys)
    for name, text in TREE.items():
        fixture.write(f"{SRC}/{name}", text)
    fixture.write("slices.toml", MAP)
    git(tmp_path, "init", "-q")
    return fixture


def test_the_clean_fixture_passes(repo: Repo) -> None:
    repo.adopt()
    repo.passes()


def test_sb00_an_unresolved_first_party_import_fails_and_blocks_init(repo: Repo) -> None:
    repo.write(f"{SRC}/_cli.py", "import shop.missing\nfrom shop.orders import place\n")
    code, out = repo.guard("baseline", "--init")
    assert code == 1
    assert f"SB-00 unresolved import: {SRC}/_cli.py -> shop.missing" in out
    assert not (repo.root / guard.BASELINE_FILE).exists()
    entry = f'{{"rule": "SB-00", "from": "{SRC}/_cli.py", "to": "shop.missing"}}'
    repo.write(guard.BASELINE_FILE, f'{{"version": 1, "entries": [{entry}]}}\n')
    repo.commit()
    _ = repo.fails("SB-00 unresolved import")  # A baseline entry never hides SB-00.


@pytest.mark.parametrize(
    "bypass",
    [
        "from shop.orders._place import place\n",
        "import shop.orders._place\n",
        "from shop.orders import _place\n",
        "from shop import orders\norders._place.place()\n",
        "from .orders import _place\n",
        "def a():\n    from shop import orders as api\n    api._place.place()\n\n\n"
        + "def b():\n    from shop import pricing as api\n",  # One alias per function scope.
        "from shop import orders as api\n\n\n"  # A default is evaluated outside the function.
        + "def f(x=api._place.place):\n    from shop import pricing as api\n",
        "from shop import orders as api\n\n\n"  # So is a decorator.
        + "@api._place.place\ndef f():\n    from shop import pricing as api\n",
        "from shop import orders as api\n\n\n"  # A comprehension does not see the class scope.
        + "class C:\n    from shop import pricing as api\n    X = [api._place for _ in (1,)]\n",
        "def setup():\n    global api\n    import shop.orders as api\n\n\napi._place.place()\n",
        "def outer():\n    api = None\n\n    def inner():\n        nonlocal api\n"
        + "        import shop.orders as api\n\n    api._place.place()\n",
    ],
)
def test_sb01_each_form_of_a_crust_bypass_fails(repo: Repo, bypass: str) -> None:
    repo.adopt()
    repo.write(f"{SRC}/_cli.py", bypass + "from shop.orders import place\n")
    _ = repo.fails(f"SB-01 crust bypass: {SRC}/_cli.py -> {SRC}/orders/_place.py")
    repo.write(f"{SRC}/_cli.py", TREE["_cli.py"])
    repo.passes()


def test_sb02_and_sb03_an_undeclared_back_edge_fails_as_a_cycle(repo: Repo) -> None:
    repo.adopt()
    repo.write(f"{SRC}/pricing/_tiers.py", BACK_EDGE)
    _ = repo.fails(
        f"SB-02 undeclared dependency: {SRC}/pricing/_tiers.py -> {SRC}/orders/__init__.py",
        f"SB-03 dependency cycle: {SRC}/orders/_place.py -> {SRC}/pricing/__init__.py",
        f"SB-03 dependency cycle: {SRC}/pricing/_tiers.py -> {SRC}/orders/__init__.py",
    )


def test_sb03_a_new_edge_inside_a_baselined_cycle_fails(repo: Repo) -> None:
    repo.write(f"{SRC}/pricing/_tiers.py", BACK_EDGE)
    repo.adopt()
    repo.passes()
    crust = TREE["pricing/__init__.py"] + "from shop.orders import place\n"
    repo.write(f"{SRC}/pricing/__init__.py", crust)
    out = repo.fails(
        f"SB-03 dependency cycle: {SRC}/pricing/__init__.py -> {SRC}/orders/__init__.py"
    )
    assert "_tiers.py ->" not in out


def test_sb04_an_unmapped_file_and_a_symlink_fail(repo: Repo) -> None:
    repo.adopt()
    repo.write(f"{SRC}/_extra.py", "")
    repo.source("pricing/_link.py").symlink_to(repo.source("pricing/_tiers.py"))
    _ = repo.fails(
        f"SB-04 unmapped source file: {SRC}/_extra.py -> unmapped",
        f"SB-04 unmapped source file: {SRC}/pricing/_link.py -> symlink",
    )


def test_sb04_a_symlinked_directory_fails(repo: Repo) -> None:
    repo.adopt()
    repo.source("orders/_mirror").symlink_to(repo.source("pricing"))
    _ = repo.fails(f"SB-04 unmapped source file: {SRC}/orders/_mirror -> symlink")
    assert f"{SRC}/orders/_mirror/_tiers.py" not in repo.guard("check", "--base", "HEAD")[1]
    repo.source("orders/_mirror").unlink()
    repo.passes()


def test_sb06_a_missing_required_edge_fails(repo: Repo) -> None:
    repo.adopt()
    repo.write(f"{SRC}/orders/_place.py", "def place():\n    return 1\n")
    _ = repo.fails("SB-06 missing required dependency: orders -> pricing")


def test_sb07_an_inline_directive_fails_and_a_string_does_not(repo: Repo) -> None:
    repo.adopt()
    repo.write(f"{SRC}/_cli.py", TREE["_cli.py"] + "NOTE = '# sbguard-ignore'\n")
    repo.passes()
    repo.write(f"{SRC}/_cli.py", TREE["_cli.py"] + "import os  # sbguard-ignore\n")
    _ = repo.fails(
        f"SB-07 suppression or baseline growth: {SRC}/_cli.py -> inline:sbguard-ignore:"
    )


def test_sb07_baseline_growth_and_deletion_fail_against_the_base(repo: Repo) -> None:
    repo.adopt()
    repo.write(f"{SRC}/_extra.py", "")
    entry = f'{{"rule": "SB-04", "from": "{SRC}/_extra.py", "to": "unmapped"}}'
    repo.write(guard.BASELINE_FILE, f'{{"version": 1, "entries": [{entry}]}}\n')
    _ = repo.fails(f"baseline:grew:SB-04|{SRC}/_extra.py|unmapped")
    repo.source("_extra.py").unlink()
    (repo.root / guard.BASELINE_FILE).unlink()
    _ = repo.fails(
        f"SB-07 suppression or baseline growth: {guard.BASELINE_FILE} -> baseline:deleted"
    )


def test_the_adoption_change_may_add_the_baseline(repo: Repo) -> None:
    _ = (repo.root / "slices.toml").rename(repo.root / "draft.toml")
    repo.commit()
    _ = (repo.root / "draft.toml").rename(repo.root / "slices.toml")
    repo.write(f"{SRC}/_extra.py", "")
    assert repo.guard("baseline", "--init")[0] == 0
    repo.passes()


def test_a_fixed_violation_cannot_pay_for_a_new_one(repo: Repo) -> None:
    repo.write(f"{SRC}/_old.py", "")
    repo.adopt()
    repo.source("_old.py").unlink()
    repo.write(f"{SRC}/_new.py", "")
    assert "STALE baseline entry" in repo.fails(f"{SRC}/_new.py -> unmapped")
    code, out = repo.guard("check", "--base", "HEAD")
    assert (code, "STALE" in out) == (1, False)


def test_shrink_removes_fixed_entries_and_never_adds(repo: Repo) -> None:
    repo.write(f"{SRC}/_old.py", "")
    repo.adopt()
    repo.source("_old.py").unlink()
    repo.write(f"{SRC}/_new.py", "")
    assert repo.guard("baseline", "--shrink")[0] == 0
    text = (repo.root / guard.BASELINE_FILE).read_text(encoding="utf-8")
    assert guard.decode(text, "baseline") == set()
    assert repo.guard("baseline", "--init")[0] == 1  # init refuses to overwrite.


def test_shrink_without_a_baseline_refuses(repo: Repo) -> None:
    code, out = repo.guard("baseline", "--shrink")
    assert (code, "run `baseline --init` first" in out) == (1, True)


def test_a_suppression_with_a_reason_accepts_one_identity_and_goes_stale(repo: Repo) -> None:
    repo.adopt()
    suppression = (
        f'rule = "SB-04"\nfrom = "{SRC}/_extra.py"\nto = "unmapped"\nreason = "generated"'
    )
    repo.write("slices.toml", f"{MAP}\n[[suppress]]\n{suppression}\n")
    repo.write(f"{SRC}/_extra.py", "")
    repo.passes()
    repo.source("_extra.py").unlink()
    _ = repo.fails("STALE [[suppress]] entry")


def test_a_missing_base_is_an_input_error(repo: Repo, monkeypatch: pytest.MonkeyPatch) -> None:
    repo.adopt()
    monkeypatch.delenv("GATE_BASE", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    code, out = repo.guard("check")
    assert (code, "no base" in out) == (2, True)
    monkeypatch.setenv("GATE_BASE", "HEAD")
    assert repo.guard("check")[0] == 0
