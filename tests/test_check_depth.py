"""Depth ratchet: the metric definitions, the ratchet modes, tighten, and the gaming cases."""

from __future__ import annotations

import ast
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest

from scripts import check_depth as depth
from scripts import depth_metrics as metrics
from scripts.slice_map import MapError

SRC = "src/shop"
MAP = f"""version = 1

[policy]
language = "python"
unmapped_files = "error"
suppressions = "human-only"
roots = ["src"]

[[slice]]
name = "orders"
paths = ["{SRC}/orders"]
crust = ["{SRC}/orders/__init__.py"]
shape = "thin-facade"
may_depend_on = []

[[spine]]
name = "app"
paths = ["{SRC}/__init__.py", "{SRC}/_cli.py"]
may_depend_on = ["orders"]
"""
CRUST = (
    "from shop.orders._place import place, unused\n\n__all__ = ['place', 'total', 'unused']\n\n\n"
    "def total(items):\n    return sum(items)\n"
)
INTERNAL = (
    "def place(order, retry=1):\n    if order and retry:\n"
    "        order.save()\n        order.log()\n"
    "    return order\n\n\ndef unused():\n    return 0\n"
)
GROWN = CRUST + "    print(items)\n"
TREE = {
    "__init__.py": "",
    "_cli.py": "from shop.orders import place, total\n\nprint(place, total)\n",
    "orders/__init__.py": CRUST,
    "orders/_place.py": INTERNAL,
}


def _statements(source: str) -> int:
    return metrics.logic_statements(ast.parse(source))


def _complexity(source: str) -> int:
    return metrics.complexity(ast.parse(source))


@pytest.mark.parametrize(
    ("source", "count"),
    [
        ("def f(x):\n    '''Doc.'''\n    if x:\n        a = 1\n        b = 2\n", 3),
        ("X = [i for i in range(3)]\nif X:\n    Y = 1\n", 0),
        ("F = lambda x: x + 1\n", 1),
        ("def f():\n    def g():\n        return 1\n    return g\n", 2),
        ("class C:\n    X = 1\n\n    def m(self):\n        return self.X\n", 1),
    ],
)
def test_logic_statements_count_function_bodies_only(source: str, count: int) -> None:
    assert _statements(source) == count


def test_reformatting_does_not_change_a_count() -> None:
    one = "def f(x):\n    return [y for y in x if y]\n"
    two = "def f(\n    x,\n):\n    return [\n        y\n        for y in x\n        if y\n    ]\n"
    assert (_statements(one), _complexity(one)) == (_statements(two), _complexity(two))


@pytest.mark.parametrize(
    ("source", "count"),
    [
        ("def f():\n    return 1\n", 1),
        ("def f(a, b=1, *, c=2):\n    if a and b or c:\n        pass\n", 6),
        (
            "def f(x):\n    for i in x:\n        try:\n            pass\n"
            + "        except ValueError:\n            pass\n    return 1 if x else 2\n",
            4,
        ),
        (
            "def f(x):\n    match x:\n        case 1 if x:\n            pass\n"
            + "        case _:\n            pass\n",
            3,
        ),
        ("def f(x):\n    return [y for y in x if y]\n", 3),
        ("def f(x):\n    def g(y):\n        if y:\n            pass\n    return g\n", 3),
        ("F = lambda x=1: x\n", 2),
    ],
)
def test_complexity_counts_decisions_and_defaults_per_function(source: str, count: int) -> None:
    assert _complexity(source) == count


def test_module_init_counts_top_level_logic_only() -> None:
    source = (
        '"""Doc."""\nimport os\nfrom typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    pass\n'
        "X = 1\nY = os.getcwd()\nprint(Y)\n\n\nclass C:\n    pass\n"
    )
    assert metrics.module_init(ast.parse(source)) == 2


@pytest.mark.parametrize(
    ("source", "names"),
    [
        ("from a import b, _c\n__all__ = ('b', 'b')\n", ["b"]),
        ("from a import b, _c\nX = 1\n_Y = 2\ndef f(): pass\n", ["X", "b", "f"]),
        ("__all__: list[str] = ['b']\n", ["b"]),
        ("import a.b\nZ: int = 1\nclass K: pass\nasync def g(): pass\n", ["K", "Z", "a", "g"]),
    ],
)
def test_exports_read_a_literal_all_or_the_public_names(source: str, names: list[str]) -> None:
    assert metrics.exports(ast.parse(source)) == names


@pytest.mark.parametrize(
    "source",
    [
        "__all__ = names()\n",
        "__all__ = ['b']\n__all__ += ['c']\n",
        "__all__ = ['b']\n__all__.extend(['c'])\n",
        "__all__ = ['b', 1]\n",
    ],
)
def test_a_computed_or_reused_all_is_an_input_error(source: str) -> None:
    with pytest.raises(MapError):
        _ = metrics.exports(ast.parse(source))


@dataclass
class Tree:
    root: Path
    capsys: pytest.CaptureFixture[str]

    def write(self, path: str, text: str) -> None:
        (self.root / path).parent.mkdir(parents=True, exist_ok=True)
        _ = (self.root / path).write_text(text, encoding="utf-8")

    def depth(self, *args: str) -> tuple[int, str]:
        _ = self.capsys.readouterr()
        code = depth.main(list(args), self.root)
        captured = self.capsys.readouterr()
        return code, captured.out + captured.err

    def values(self) -> dict[str, str]:
        code, out = self.depth("measure", "--json")
        assert code == 0, out
        return cast(dict[str, dict[str, dict[str, str]]], json.loads(out))["orders"]["values"]

    def baseline(self) -> dict[str, object]:
        text = (self.root / depth.BASELINE_FILE).read_text(encoding="utf-8")
        return cast(dict[str, object], json.loads(text))

    def save(self, document: dict[str, object]) -> None:
        _ = (self.root / depth.BASELINE_FILE).write_text(json.dumps(document), encoding="utf-8")

    def caps(self) -> dict[str, object]:
        return cast(dict[str, dict[str, object]], self.baseline()["slices"])["orders"]

    def init(self, mode: str = "block") -> None:
        assert self.depth("init", "--mode", mode)[0] == 0


@pytest.fixture
def tree(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> Tree:
    fixture = Tree(tmp_path, capsys)
    fixture.write("slices.toml", MAP)
    for name, text in TREE.items():
        fixture.write(f"{SRC}/{name}", text)
    return fixture


def test_the_fixture_numbers(tree: Tree) -> None:
    assert tree.values() == {
        "M08.crust_logic_statements": "1",
        "M08.share": "1/6",
        "M09.public_surface": "3",
        "M10.crust_complexity": "1",
        "M10.share": "1/6",
        "M12.unused_exports": "1",
    }


def test_measure_prints_a_table(tree: Tree) -> None:
    code, out = tree.depth("measure")
    assert code == 0
    assert out.splitlines()[1].split() == ["orders", "1", "1/6", "3", "1", "1/6", "1", "0"]


@pytest.mark.parametrize(
    ("source", "unused"),
    [
        ("from shop.orders import unused\n\nprint(unused)\n", "0"),
        ("from shop.orders import unused\n", "1"),  # An unreferenced import is not a use.
        ("from shop import orders\n\norders.unused()\n", "0"),
        ("import shop.orders\n\nshop.orders.unused()\n", "0"),
        ("import shop.orders as o\n\nprint(o)\n", "0"),  # A bare module use uses every export.
        ("from shop.orders import *\n", "0"),
        ("from shop.orders._place import unused\n\nprint(unused)\n", "1"),  # A bypass is no use.
        ("from shop.orders import unused\n\n__all__ = ['unused']\n", "0"),  # A re-export is a use.
    ],
)
def test_m12_counts_each_use_form_through_the_crust(tree: Tree, source: str, unused: str) -> None:
    tree.write(f"{SRC}/__init__.py", source)
    assert tree.values()["M12.unused_exports"] == unused


def test_crust_growth_blocks_in_block_mode_and_reports_in_report_mode(tree: Tree) -> None:
    tree.init("block")
    assert tree.depth("check")[0] == 0
    tree.write(f"{SRC}/orders/__init__.py", GROWN)
    code, out = tree.depth("check")
    assert code == 1, out
    assert "regressed orders M08.crust_logic_statements 1 -> 2" in out
    assert "regressed orders M08.share 1/6 -> 2/7" in out
    (tree.root / depth.BASELINE_FILE).unlink()
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
    tree.write(depth.BASELINE_FILE, "{")
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


def _git(root: Path, *args: str) -> None:
    _ = subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=root,
        check=True,
        capture_output=True,
    )


@pytest.mark.real_process
def test_diff_baseline_lists_loosening(tree: Tree) -> None:
    tree.init()
    _git(tree.root, "init", "-q")
    assert tree.depth("diff-baseline", "--base", "HEAD")[0] == 2  # No commit holds a baseline.
    _git(tree.root, "add", "-A")
    _git(tree.root, "commit", "-q", "-m", "base")
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
