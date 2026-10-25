"""Depth metrics: the definition of each number and each use form that M12 counts."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from scripts import depth_metrics as metrics
from scripts.slice_map import MapError
from tests.depth_fixture import SRC, Tree, make_tree


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


@pytest.fixture
def tree(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> Tree:
    return make_tree(tmp_path, capsys)


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
        ("__all__ = names()\n", "1"),  # A computed `__all__` outside the crust is no error.
        ("from shop.orders import unused\n\n__all__ = names()\n", "0"),  # It may re-export.
        (
            "def a():\n    import shop.orders._place as api\n    api.unused()\n\n\n"
            + "def b():\n    import shop.orders as api\n",
            "1",  # Each function scope binds its own `api`.
        ),
    ],
)
def test_m12_counts_each_use_form_through_the_crust(tree: Tree, source: str, unused: str) -> None:
    tree.write(f"{SRC}/__init__.py", source)
    assert tree.values()["M12.unused_exports"] == unused
