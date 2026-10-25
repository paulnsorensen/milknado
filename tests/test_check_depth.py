"""Depth ratchet: the ratchet modes, tighten, the baseline checks, and the gaming cases."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from scripts.depth_baseline import BASELINE_FILE
from tests.depth_fixture import CRUST, GROWN, INTERNAL, MAP, SRC, Tree, make_tree
from tests.slice_fixture import git


@pytest.fixture
def tree(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> Tree:
    return make_tree(tmp_path, capsys)


def test_crust_growth_blocks_in_block_mode_and_reports_in_report_mode(tree: Tree) -> None:
    tree.init("block")
    assert tree.depth("check")[0] == 0
    tree.write(f"{SRC}/orders/__init__.py", GROWN)
    code, out = tree.depth("check")
    assert code == 1, out
    assert "regressed orders M08.crust_logic_statements 1 -> 2" in out
    assert "regressed orders M08.share 1/6 -> 2/7" in out
    (tree.root / BASELINE_FILE).unlink()
    tree.write(f"{SRC}/orders/__init__.py", CRUST)
    tree.init("report")
    tree.write(f"{SRC}/orders/__init__.py", GROWN)
    code, out = tree.depth("check")
    assert code == 0, out
    assert "regressed orders M08.crust_logic_statements" in out


def test_internal_padding_cannot_hide_crust_growth(tree: Tree) -> None:
    tree.init()
    padding = "".join(f"\n\ndef pad{n}():\n    return {n}\n" for n in range(20))
    tree.write(f"{SRC}/orders/_place.py", INTERNAL + padding)
    tree.write(f"{SRC}/orders/__init__.py", GROWN)
    code, out = tree.depth("check")
    assert code == 1
    assert "regressed orders M08.crust_logic_statements 1 -> 2" in out
    assert "regressed orders M08.share" not in out


def test_new_exports_and_unused_exports_regress(tree: Tree) -> None:
    tree.init()
    crust = CRUST.replace("'unused']", "'unused', 'extra']\nextra = 1")
    tree.write(f"{SRC}/orders/__init__.py", crust)
    code, out = tree.depth("check")
    assert code == 1
    assert "regressed orders M09.public_surface 3 -> 4" in out
    assert "regressed orders M12.unused_exports 1 -> 2" in out


def test_an_internal_simplification_reports_denominator_shrink(tree: Tree) -> None:
    tree.init()
    simpler = "def place(order):\n    return order\n\n\ndef unused():\n    return 0\n"
    tree.write(f"{SRC}/orders/_place.py", simpler)
    code, out = tree.depth("check")
    assert code == 1
    assert "regressed orders M08.share 1/6 -> 1/3 denominator-shrink" in out


def test_tighten_lowers_improved_caps_and_never_raises_one(tree: Tree) -> None:
    tree.init()
    smaller = CRUST.replace(", 'unused']", "]").replace(", unused", "")
    tree.write(f"{SRC}/orders/__init__.py", smaller)
    code, out = tree.depth("check")
    assert code == 0, out
    assert "improved orders M09.public_surface 3 -> 2" in out
    code, out = tree.depth("tighten")
    assert (code, "improved orders M09.public_surface 3 -> 2" in out) == (0, True)
    assert (tree.caps()["M09.public_surface"], tree.caps()["M12.unused_exports"]) == (2, 0)
    tree.write(f"{SRC}/orders/__init__.py", CRUST)
    code, out = tree.depth("tighten")
    assert code == 1
    assert "tighten refuses" in out
    assert tree.caps()["M09.public_surface"] == 2


def test_tighten_without_a_change_reports_no_cap_change(tree: Tree) -> None:
    tree.init()
    assert tree.depth("tighten") == (0, "tighten: no cap changed\n")


def test_tighten_holds_a_share_gain_from_internal_growth(tree: Tree) -> None:
    tree.init()
    tree.write(f"{SRC}/orders/_place.py", INTERNAL + "\n\ndef more():\n    return place(None)\n")
    code, out = tree.depth("tighten")
    assert code == 0, out
    assert "held orders M08.share 1/6 -> 1/7" in out
    assert tree.caps()["M08.share"] == "1/6"
    assert tree.depth("tighten", "--accept-denominator-gains")[0] == 0
    assert tree.caps()["M08.share"] == "1/7"


def test_new_and_orphaned_slices(tree: Tree) -> None:
    tree.init()
    renamed = MAP.replace('name = "orders"', 'name = "sales"').replace('["orders"]', '["sales"]')
    tree.write("slices.toml", renamed)
    code, out = tree.depth("check")
    assert code == 1
    assert "new sales: no baseline" in out
    assert "orphan orders" in out
    assert tree.depth("tighten", "--adopt-new")[0] == 1  # An orphan blocks tighten.


def test_adopt_new_adds_a_baseline_for_a_new_slice(tree: Tree) -> None:
    extra = (
        f'[[slice]]\nname = "extra"\npaths = ["{SRC}/extra"]\n'
        f'crust = ["{SRC}/extra/__init__.py"]\nshape = "thin-facade"\nmay_depend_on = []\n\n'
    )
    tree.write("slices.toml", MAP.replace("[[slice]]", extra + "[[slice]]", 1))
    tree.write(f"{SRC}/extra/__init__.py", "")
    tree.init()
    document = tree.baseline()
    del cast(dict[str, object], document["slices"])["extra"]
    tree.save(document)
    assert "new extra: no baseline" in tree.depth("check")[1]
    assert tree.depth("tighten", "--adopt-new")[0] == 0
    assert tree.depth("check")[1].strip() == "depth ratchet: pass (2 slices, mode block)"


def test_a_missing_cap_is_reported_as_new(tree: Tree) -> None:
    tree.init()
    document = tree.baseline()
    del cast(dict[str, dict[str, object]], document["slices"])["orders"]["M09.public_surface"]
    tree.save(document)
    code, out = tree.depth("check")
    assert (code, "new orders M09.public_surface = 3 (no cap)" in out) == (0, True)


def test_init_refuses_to_overwrite_and_an_old_measurement_is_an_input_error(tree: Tree) -> None:
    tree.init()
    assert tree.depth("init")[0] == 1
    document = tree.baseline()
    document["measurement_version"] = 0
    tree.save(document)
    code, out = tree.depth("check")
    assert (code, "measurement_version 0 is not 1" in out) == (2, True)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mode", "loud"),
        ("slices", {"orders": {"M08.share": "half"}}),
        ("slices", {"orders": {"M09.public_surface": -1}}),
        ("slices", {"orders": {"M09.public_surface": True}}),
        ("slices", {"orders": {"M99.unknown": 1}}),
    ],
)
def test_an_invalid_baseline_is_an_input_error(tree: Tree, field: str, value: object) -> None:
    tree.init()
    document = tree.baseline()
    document[field] = value
    tree.save(document)
    assert tree.depth("check")[0] == 2


def test_a_missing_or_broken_baseline_is_an_input_error(tree: Tree) -> None:
    assert tree.depth("check")[0] == 2
    tree.write(BASELINE_FILE, "{")
    assert tree.depth("check")[0] == 2


def test_check_names_computed_imports_and_large_crusts(tree: Tree) -> None:
    body = "".join(f"    x{n} = {n}\n" for n in range(201))
    tree.write(f"{SRC}/orders/__init__.py", f"import importlib\n\n\ndef big():\n{body}")
    tree.write(f"{SRC}/_cli.py", "import importlib\n\nimportlib.import_module('shop.orders')\n")
    tree.init("report")
    code, out = tree.depth("check")
    assert code == 0
    assert "advisory M08.size-aware orders: crust holds 201/206" in out
    assert f"coverage: computed imports that no number follows: {SRC}/_cli.py:3" in out


@pytest.mark.real_process
def test_diff_baseline_lists_loosening(tree: Tree) -> None:
    tree.init()
    git(tree.root, "init", "-q")
    assert tree.depth("diff-baseline", "--base", "HEAD")[0] == 2  # No commit holds a baseline.
    git(tree.root, "add", "-A")
    git(tree.root, "commit", "-q", "-m", "base")
    assert tree.depth("diff-baseline", "--base", "HEAD") == (0, "no loosening\n")
    document = tree.baseline()
    document["mode"] = "report"
    slices = cast(dict[str, dict[str, object]], document["slices"])
    slices["orders"]["M09.public_surface"] = 9
    del slices["orders"]["M12.unused_exports"]
    tree.save(document)
    code, out = tree.depth("diff-baseline", "--base", "HEAD")
    assert code == 0
    assert "loosened mode block -> report" in out
    assert "loosened raised orders M09.public_surface 3 -> 9" in out
    assert "loosened removed orders M12.unused_exports" in out
    tree.save({**document, "slices": {}})
    assert "loosened removed slice orders" in tree.depth("diff-baseline", "--base", "HEAD")[1]
    (tree.root / BASELINE_FILE).unlink()
    code, out = tree.depth("diff-baseline", "--base", "HEAD")
    assert (code, out) == (2, f"error: the working tree has no {BASELINE_FILE}\n")
    tree.save({**document, "measurement_version": 0})
    git(tree.root, "commit", "-q", "-am", "an older measurement")
    (tree.root / BASELINE_FILE).unlink()
    assert tree.depth("diff-baseline", "--base", "HEAD")[0] == 2  # Even for an older base.
