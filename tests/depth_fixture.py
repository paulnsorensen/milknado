"""The source tree and helpers that the depth ratchet tests share."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest

from scripts import check_depth as depth
from scripts.depth_baseline import BASELINE_FILE

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
        text = (self.root / BASELINE_FILE).read_text(encoding="utf-8")
        return cast(dict[str, object], json.loads(text))

    def save(self, document: dict[str, object]) -> None:
        _ = (self.root / BASELINE_FILE).write_text(json.dumps(document), encoding="utf-8")

    def caps(self) -> dict[str, object]:
        return cast(dict[str, dict[str, object]], self.baseline()["slices"])["orders"]

    def init(self, mode: str = "block") -> None:
        assert self.depth("init", "--mode", mode)[0] == 0


def make_tree(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> Tree:
    """Write the depth fixture tree and its slice map under `tmp_path`."""
    fixture = Tree(tmp_path, capsys)
    fixture.write("slices.toml", MAP)
    for name, text in TREE.items():
        fixture.write(f"{SRC}/{name}", text)
    return fixture
